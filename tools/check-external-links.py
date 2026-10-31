#!/usr/bin/env python3
"""Check the external links in docs/**/*.qmd (#2487).

    tools/check-external-links.py [--docs DIR] [--url URL ...] [--list]
                                  [--summary FILE] [--timeout S] [--retries N]
                                  [--backoff S] [--jobs N]

Collects every `https://` URL in `DIR/**/*.qmd` (default `docs/`, relative to the
repo root), fetches each one following redirects, and requires a 2xx. For a URL
with a `#fragment` it also requires the final page to carry `id="<fragment>"` or
`name="<fragment>"` -- a renamed heading on a linked page is the failure this
exists to catch (`docs/model-file/mbma.qmd` links a section of the ferx book by
its anchor).

Why this is not a docs-lint rule: `crates/docs-lint` depends on no crate and makes
no network call by design (#969, #1163), so its R4 checks internal links only.
And why it is not a per-PR gate: an external site that is down for an hour would
redden every PR opened in that hour for a defect none of them made. So
`.github/workflows/external-links.yml` runs it weekly and on dispatch, never on
`pull_request` (pinned by `tests/check_external_links_script.rs`).

`--url` adds URLs to check (any http/https scheme) on top of the collected ones.
The workflow's dispatch input uses it to prove the job fails on a broken link or
anchor without committing one; the tests use it to point the checker at a local
server. `--list` prints what would be checked, with locations, and fetches
nothing.

Exit status: 0 every link passed, 1 at least one failed, 2 usage error.

Python stdlib only, so it runs on a bare runner and on a developer's machine with
no install step.
"""

import argparse
import concurrent.futures
import html
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

USER_AGENT = "ferx-docs-linkcheck/1 (+https://github.com/FeRx-NLME/ferx-core)"

# Pages larger than this are cut off before the fragment search. The largest
# page linked today is well under 1 MB.
MAX_BODY_BYTES = 8 * 1024 * 1024

# Hosts that cannot be checked the ordinary way, each with the reason. Match is on
# the host or any parent domain (`onlinelibrary.wiley.com` covers
# `ascpt.onlinelibrary.wiley.com`).
#
#   "skip"      -- not fetched at all; reported as skipped, never as a failure.
#   "no-follow" -- fetched without following redirects; a 3xx counts as a pass.
HOST_POLICY = {
    # A registered DOI answers 302 to the publisher and an unregistered one 404,
    # so the resolver itself says whether the DOI exists. Following the redirect
    # can land on a publisher that refuses non-browser clients: measured
    # 2026-10-31, doi.org/10.1002/psp4.12808 -> 302 -> Wiley -> 403, while the
    # unregistered doi.org/10.1002/psp4.99999999 -> 404 at the resolver.
    "doi.org": "no-follow",
    # Wiley's Online Library answers 403 to non-browser clients (a bot
    # challenge), whether or not the article exists: measured 2026-10-31 on the
    # live article linked from docs/.
    "onlinelibrary.wiley.com": "skip",
}

# `https://` up to whitespace or a character that ends a URL in markdown, HTML or
# a Quarto shortcode. Trailing sentence punctuation is stripped afterwards.
URL_RE = re.compile(r"https://[^\s<>()\[\]{}\"'`|\\]+")
TRAILING = ".,;:!?*_~"


def collect(docs_dir):
    """Map each URL found in `docs_dir/**/*.qmd` to its `file:line` locations."""
    found = {}
    for path in sorted(docs_dir.rglob("*.qmd")):
        if "_site" in path.relative_to(docs_dir).parts:
            continue
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in URL_RE.finditer(line):
                url = m.group(0).rstrip(TRAILING)
                if urllib.parse.urlsplit(url).netloc:
                    found.setdefault(url, []).append(f"{rel}:{lineno}")
    return found


def policy_for(host):
    host = host.lower()
    for domain, policy in HOST_POLICY.items():
        if host == domain or host.endswith("." + domain):
            return domain, policy
    return None, None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_FOLLOW = urllib.request.build_opener()
_NO_FOLLOW = urllib.request.build_opener(_NoRedirect)


def fetch(url, follow, want_body, timeout):
    """One request. Returns `(status, final_url, body_or_None, retry_after)`.

    A 3xx under `follow=False`, and every 4xx/5xx, comes back as a status rather
    than an exception; network errors raise `OSError`.
    """
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    opener = _FOLLOW if follow else _NO_FOLLOW
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(MAX_BODY_BYTES) if want_body else None
            return resp.status, resp.geturl(), body, None
    except urllib.error.HTTPError as e:
        return e.code, url, None, e.headers.get("Retry-After") if e.headers else None


def fragment_present(body, fragment):
    text = body.decode("utf-8", errors="replace")
    frag = urllib.parse.unquote(fragment)
    for attr in ("id", "name"):
        for value in re.findall(
            rf"""\s{attr}\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))""", text
        ):
            if html.unescape(next((v for v in value if v), "")) == frag:
                return True
    return False


def check_page(page, fragments, timeout, retries, backoff):
    """Check one page (URL without fragment) and every fragment linked on it.

    Returns a list of `(url, ok, detail)`, one per fragment-qualified URL (or one
    for the bare page when no fragment is linked).
    """
    host = urllib.parse.urlsplit(page).hostname or ""
    domain, policy = policy_for(host)
    urls = [page + "#" + f if f else page for f in fragments]
    if policy == "skip":
        return [(u, None, f"skipped: host {domain} is on the skip list") for u in urls]
    follow = policy != "no-follow"
    want_body = any(fragments)

    detail = ""
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff * 2 ** (attempt - 1))
        try:
            status, final, body, retry_after = fetch(page, follow, want_body, timeout)
        except (OSError, ValueError) as e:
            detail = f"error: {getattr(e, 'reason', e)}"
            continue
        if 200 <= status < 300 or (not follow and 300 <= status < 400):
            break
        detail = f"HTTP {status}"
        if status == 429 and retry_after and retry_after.isdigit():
            time.sleep(min(int(retry_after), 60))
        if 400 <= status < 500 and status not in (408, 425, 429):
            return [(u, False, detail) for u in urls]
    else:
        return [(u, False, detail + f" (after {retries + 1} attempts)") for u in urls]

    out = []
    for frag, url in zip(fragments, urls):
        if not frag:
            out.append((url, True, f"HTTP {status}"))
        elif body is None:
            out.append((url, False, f"HTTP {status} but no page body to search for #{frag}"))
        elif fragment_present(body, frag):
            out.append((url, True, f"HTTP {status}, #{frag} present"))
        else:
            where = f" (redirected to {final})" if final != page else ""
            out.append((url, False, f"no id=\"{frag}\" or name=\"{frag}\" on the page{where}"))
    return out


def write_summary(path, results, locations):
    failed = [r for r in results if r[1] is False]
    skipped = [r for r in results if r[1] is None]
    passed = len(results) - len(failed) - len(skipped)
    lines = ["## External links in docs/", ""]
    lines.append(
        f"{len(results)} URLs: **{passed} passed**, **{len(failed)} failed**, "
        f"{len(skipped)} skipped."
    )
    lines.append("")
    if failed:
        lines += ["| URL | Problem | Linked from |", "|---|---|---|"]
        for url, _, detail in failed:
            where = ", ".join(locations.get(url, ["`--url`"]))
            lines.append(f"| <{url}> | {detail} | {where} |")
        lines.append("")
    if skipped:
        lines.append("Skipped (see `HOST_POLICY` in `tools/check-external-links.py`):")
        lines.append("")
        for url, _, detail in skipped:
            lines.append(f"- <{url}> -- {detail}")
        lines.append("")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv):
    ap = argparse.ArgumentParser(
        description="Check https:// links (and #anchors) in docs/**/*.qmd.")
    ap.add_argument("--docs", default=str(REPO_ROOT / "docs"),
                    help="directory to scan for *.qmd (default: <repo>/docs)")
    ap.add_argument("--url", action="append", default=[],
                    help="extra URL to check; repeatable, or space-separated")
    ap.add_argument("--list", action="store_true",
                    help="print the URLs that would be checked and exit")
    ap.add_argument("--summary", help="append a markdown report to this file")
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds per request")
    ap.add_argument("--retries", type=int, default=3, help="retries per page")
    ap.add_argument("--backoff", type=float, default=2.0,
                    help="first retry delay in seconds, doubled each retry")
    ap.add_argument("--jobs", type=int, default=6, help="pages fetched in parallel")
    args = ap.parse_args(argv)

    docs_dir = Path(args.docs)
    if not docs_dir.is_dir():
        print(f"check-external-links: no such directory: {docs_dir}", file=sys.stderr)
        return 2
    locations = collect(docs_dir)
    for extra in (u for arg in args.url for u in arg.split()):
        if urllib.parse.urlsplit(extra).scheme not in ("http", "https"):
            print(f"check-external-links: not an http(s) URL: {extra}", file=sys.stderr)
            return 2
        locations.setdefault(extra, ["`--url`"])

    if args.list:
        for url in sorted(locations):
            print(f"{url}\t{', '.join(locations[url])}")
        return 0

    # One fetch per page, however many of its fragments are linked.
    pages = {}
    for url in locations:
        page, _, frag = url.partition("#")
        pages.setdefault(page, []).append(frag)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(check_page, page, frags, args.timeout, args.retries, args.backoff)
            for page, frags in pages.items()
        ]
        for fut in futures:
            results.extend(fut.result())
    results.sort(key=lambda r: r[0])

    for url, ok, detail in results:
        mark = "SKIP" if ok is None else ("ok  " if ok else "FAIL")
        print(f"{mark} {url} -- {detail}")
        if ok is False:
            print(f"     linked from: {', '.join(locations.get(url, []))}")
    if args.summary:
        write_summary(args.summary, results, locations)

    failed = sum(1 for r in results if r[1] is False)
    print(f"\n{len(results)} URLs checked, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
