#!/usr/bin/env python3
"""Quarto post-render step: publish the curated llms.txt and llms-full.txt (#3073).

Run by `quarto render` as the project's `post-render` script (see `_quarto.yml`),
so it runs inside the `Deploy Docs` workflow and in `tools/render-docs.sh`, and
nothing it writes is committed.

`website: llms-txt: true` makes Quarto write a Markdown twin (`<page>.llms.md`)
next to every rendered page, plus its own `llms.txt`: a flat, unordered list of
every page. That generated index is replaced here by the curated `docs/llms.txt`
(sections, usage notes, the agent page first), and `llms-full.txt` is built from
the twins: every page's Markdown, one after another, each under its URL.

A partial render (`quarto preview`, `quarto render <one page>`) writes twins for
the pages it rendered only, so `llms-full.txt` is rebuilt only on a full render
(`QUARTO_PROJECT_RENDER_ALL`); the curated `llms.txt` is copied either way, so the
site never serves Quarto's generated index in its place.

Python stdlib only.
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def site_url() -> str:
    """`website.site-url` from `_quarto.yml`, with a trailing slash."""
    for line in (PROJECT_DIR / "_quarto.yml").read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition(":")
        if key == "site-url" and value.strip():
            url = value.strip().strip("\"'")
            return url if url.endswith("/") else url + "/"
    sys.exit("_llms.py: no website.site-url in _quarto.yml")


def main() -> None:
    out = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR") or "_site")
    if not out.is_absolute():
        out = PROJECT_DIR / out
    curated = PROJECT_DIR / "llms.txt"
    (out / "llms.txt").write_text(curated.read_text(encoding="utf-8"), encoding="utf-8")

    if not os.environ.get("QUARTO_PROJECT_RENDER_ALL"):
        return

    twins = sorted(out.rglob("*.llms.md"))
    if not twins:
        # The curated llms.txt promises the twins; a Quarto without
        # `llms-txt` support (older than 1.9) would publish a broken promise.
        sys.exit(
            "_llms.py: rendered no *.llms.md twins; Quarto >= 1.9 is needed for "
            "`website: llms-txt`"
        )
    base = site_url()
    # The agent page first, as in llms.txt; then the rest in path order.
    first = out / "agents.llms.md"
    twins.sort(key=lambda p: (p != first, p.relative_to(out).as_posix()))
    parts = [curated.read_text(encoding="utf-8").rstrip() + "\n"]
    for twin in twins:
        url = base + twin.relative_to(out).as_posix()
        parts.append(f"\n---\n\nSource: {url}\n\n{twin.read_text(encoding='utf-8').strip()}\n")
    (out / "llms-full.txt").write_text("".join(parts), encoding="utf-8")
    print(f"_llms.py: llms.txt + llms-full.txt from {len(twins)} page(s)")


if __name__ == "__main__":
    main()
