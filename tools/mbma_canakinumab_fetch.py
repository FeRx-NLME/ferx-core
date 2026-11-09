"""Fetch the canakinumab / rheumatoid-arthritis MBMA dataset and derive the ferx layout.

Usage:
    python3 tools/mbma_canakinumab_fetch.py [--from <zip|csv>] [--out <path>] \\
        [--allow-new-hash]
    python3 tools/mbma_canakinumab_fetch.py --check [--out <path>]

Source: Case Study 2 of

    Bracis C, Taneja A, Lyauk YK, Barcomb H, de la Pena A, Celliere G.
    Model-Based Meta-Analysis With MonolixSuite: A Tutorial for Longitudinal
    Categorical and Continuous Data.
    CPT Pharmacometrics Syst Pharmacol. 2026;15:e70158.
    doi:10.1002/psp4.70158 (PMC12896392)

9 studies, 15 arms, 140 longitudinal ACR20 responder proportions.

THE DATA ARE NOT REDISTRIBUTABLE, AND THIS SCRIPT NEVER WRITES INTO THE TRACKED
TREE. The article is CC BY-NC-ND 4.0, the supplementary zip carries no licence of
its own, and a ferx-layout CSV is an adaptation, which ND forbids distributing
(#1913). So neither the upstream file nor the derived one is committed anywhere:
the default output, `data/external/mbma_canakinumab.csv`, is git-ignored, and
`tests/mbma_canakinumab_case_study.rs` fails if anything under `data/external/` is
ever tracked. Run this script locally, then run the anchor:

    python3 tools/mbma_canakinumab_fetch.py
    tools/linux-test.sh -- test --no-default-features \\
        --features ferx-core/ci,ferx-core/slow-tests \\
        --test mbma_canakinumab_case_study -- --nocapture

The download is Europe PMC's `supplementaryFiles` endpoint, an outer zip that is
assembled per request (fresh timestamps every time), so only the inner CSV can be
hash-pinned. The endpoint is slow (minutes for ~10 MB), has returned HTTP 500, and
has served an inner CSV with the wrong hash that a request half an hour later got
right (#2818), hence the retries: a transport error, an unreadable zip and a wrong
hash each fail one attempt, and all attempts share one `--deadline-seconds` budget,
so a trickling download cannot outlast the CI step's cap (#1938). The attempts are
spread over that budget, doubling waits from 15 s up to 120 s, so a short outage is
waited out; none starts in the budget's last quarter, which is kept for one full
download; and every phase of an attempt, the response headers included, ends at the
deadline (#2824; see `read_within` for the one fallback). `--from` takes a local copy of the outer zip, of
`PSP4-15-e70158-s002.zip`, or of the bare `data_9studies.csv`, and skips the
download.

The nightly `slow-tests.yml` core leg runs this script, then `--check`, and sets
FERX_REQUIRE_EXTERNAL_DATA=1, so a failed fetch is a red job rather than a skipped
anchor. Downloading at test time redistributes nothing. The file is deliberately
NOT put in the Actions cache or uploaded as an artifact: a cache entry written on
`main` can be restored by any branch's workflow, a fork PR's included, so caching
it would hand the data to anyone who opens a PR.

Transformations, all mechanical:
    ID      <- STUDY   (study is the unit of between-study variability)
    ARM     <- ARM     (the occasion for the between-arm kappa)
    TIME    <- TIME    (weeks)
    DV      <- transY / sqrt(Narm)
    NARM    <- Narm    (kappa weight and residual weight)
    ABATA, ADALI, CANAKI <- DRUG, as 0/1 indicators (placebo is the reference)
    MDV     <- 0

`DV` is `transY / sqrt(Narm)` and not `logit(Y / 100)`: `transY` is what the
published Monolix run fitted, and `Y` is rounded, so recomputing the logit from it
moves the weighted value by up to 5.5e-3. Every other column (`Y`, `Publication`,
`TRT`, `DRUGinteger`, the three baseline covariates) is dropped; the published
final model (run r03) uses none of them.
"""

import argparse
import csv
import hashlib
import http.client
import io
import math
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12896392/supplementaryFiles"
INNER_ZIP = "PSP4-15-e70158-s002.zip"
INNER_CSV = "supplemental/case_study_2/data_9studies.csv"
DEFAULT_OUT = "data/external/mbma_canakinumab.csv"
# The attempts are spread over the `--deadline-seconds` budget (#2824): the waits
# double from BACKOFF_BASE_SECONDS up to BACKOFF_CAP_SECONDS, and no attempt starts
# in the last RESERVE_FRACTION of the budget, which is kept for one full download.
# At the default 720 s, with fast failures, the attempts start at 0, 15, 45, 105,
# 225, 345, 465 and 540 s; the 180 s reserve is about 1.03x the slowest download CI
# has measured (174 s, slow-tests run 42608366633), so it has little headroom; see
# #2824. There is no sleep after the last attempt (#1938).
BACKOFF_BASE_SECONDS = 15
BACKOFF_CAP_SECONDS = 120
RESERVE_FRACTION = 0.25
# A non-zip reply is quoted in the log up to this many bytes.
QUOTE_BYTES = 1024

# sha256 of the upstream `data_9studies.csv` (inside PSP4-15-e70158-s002.zip).
UPSTREAM_SHA256 = "d71c7e0882e95eae4f096387a95c33ed1c4fa0fe85eb57d317886dee94553b27"
# sha256 of the derived ferx-layout CSV this script writes, and what `--check`
# verifies. The anchor test does no hashing: it fingerprints the file by its shape
# and per-column sums instead (`check_population` in the test).
DERIVED_SHA256 = "ecc4a9fb284983c8be7f66b84bf295ffa09ff714f10e8e4d00c962a78c8cf738"

EXPECTED_COLUMNS = [
    "STUDY", "ARM", "DRUG", "DRUGinteger", "TIME", "Y", "Publication", "Narm",
    "TRT", "DiseaseDuration", "CReactiveProtein", "SwollenJointCount", "transY",
]
EXPECTED_ROWS = 140
EXPECTED_STUDIES = 9
EXPECTED_ARMS = 15
DRUGS = ("abata", "adali", "canaki")
OUT_COLUMNS = ["ID", "ARM", "TIME", "DV", "NARM", "ABATA", "ADALI", "CANAKI", "MDV"]


class AttemptFailed(Exception):
    """One download attempt that did not yield the pinned upstream CSV."""

    def __init__(self, log, sha=None):
        super().__init__(log)
        self.log = log  # the attempt's line in the final report
        self.sha = sha  # the wrong sha256, when the payload parsed but hashed wrong


def open_within(url, timeout, deadline):
    """`urlopen(url)`, or `TimeoutError` if its response headers are not in by `deadline`.

    `urlopen(timeout=...)` limits each socket operation, so a server that trickles its
    headers one byte at a time resets it with every byte, and name resolution is not
    covered at all (#2824). So `urlopen` runs in a daemon thread and the caller waits
    for it only until the deadline. A thread left behind dies with the process: the
    retry loop starts no attempt after the deadline. An exception raised in the thread
    is re-raised here, type and text intact. The thread gets the full `timeout`, not
    the time left: the `join` is the one deadline gate, so a header phase cut by the
    deadline is always reported as the deadline, never as a socket `timed out` that
    happens to fire at the same instant (#2834 review row 2).
    """
    box = {}

    def work():
        try:
            # URL is the https constant above, never user input (asserted in main).
            box["r"] = urllib.request.urlopen(url, timeout=timeout)  # nosec B310
        except BaseException as e:  # noqa: BLE001 — handed to the caller as is
            box["e"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        raise TimeoutError("deadline reached before the response headers")
    if "e" in box:
        raise box["e"]
    return box["r"]


def read_within(url, timeout, deadline):
    """`(body, Content-Type)` of `url`; `TimeoutError` once `deadline` (monotonic) passes.

    `urlopen(timeout=...)` limits each socket operation, not the download: a server
    that trickles bytes never trips it (#1938). So the headers are waited for only
    until the deadline (`open_within`), and before every body read the socket timeout
    is re-set to `min(timeout, time left)`: no read outlasts the deadline, including
    one that stalls after the body has started.
    """
    with open_within(url, timeout, deadline) as r:
        # The connection's socket, to re-cap per read. It is private CPython API, so
        # if it is ever missing the fetch still works and only loses the re-cap: the
        # worst case is then the deadline plus one `timeout`, which the workflow's
        # budget test already counts.
        sock = getattr(getattr(r.fp, "raw", None), "_sock", None)
        parts = []
        while True:
            # Python ≥ 3.12's `read1` closes the connection itself once Content-Length
            # has arrived, and a `settimeout` on that closed socket raises EBADF: the
            # body is complete, so stop before re-capping (#2820 review r2).
            if r.isclosed():
                break
            left = deadline - time.monotonic()
            if left <= 0:
                got = sum(len(p) for p in parts)
                raise TimeoutError(f"deadline reached mid-download ({got} bytes received)")
            if sock is not None:
                sock.settimeout(min(timeout, left))
            try:
                chunk = r.read1(1 << 16)
            except socket.timeout:
                # A read cut by the time left is the deadline, and is reported as one;
                # a read cut by `timeout` is a stalled server, and stays a timeout.
                if left < timeout:
                    got = sum(len(p) for p in parts)
                    raise TimeoutError(
                        f"deadline reached mid-download ({got} bytes received)"
                    ) from None
                raise
            if not chunk:
                break
            parts.append(chunk)
        body = b"".join(parts)
        # `read1` returns b"" when the server closes early, where `read()` raised;
        # `length` is what Content-Length promised and has not arrived.
        if r.length:
            raise http.client.IncompleteRead(body, r.length)
        return body, r.headers.get("Content-Type")


def attempt(url, timeout, deadline):
    """The pinned upstream CSV bytes, or `AttemptFailed`.

    A payload that downloads but is not a readable zip, or whose CSV hashes wrong,
    fails the attempt exactly like a transport error: the endpoint has served a
    wrong-hash payload that the next request got right (#2818).
    """
    try:
        blob, content_type = read_within(url, timeout, deadline)
    except Exception as e:  # noqa: BLE001 — any failure is retried, then reported
        raise AttemptFailed(f"error: {type(e).__name__}: {e}") from None
    # The endpoint always answers with a zip; anything else is a garbled reply, not
    # a bare CSV to hash.
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        # Quote it whole, up to QUOTE_BYTES: on CI it has been the same 292-byte reply
        # every time (sha256 dd011235…, served with HTTP 2xx, #2824), and its text is
        # the only clue what it was.
        quote = f"{blob[:QUOTE_BYTES]!r}"
        if len(blob) > QUOTE_BYTES:
            quote += f" (cut at {QUOTE_BYTES} of {len(blob)} bytes)"
        raise AttemptFailed(
            f"unreadable payload: not a zip ({len(blob)} bytes, "
            f"{content_type or 'no content-type'}): {quote}"
        )
    try:
        raw = upstream_csv(blob)
    except (ValueError, zipfile.BadZipFile, KeyError, EOFError) as e:
        raise AttemptFailed(f"unreadable payload ({len(blob)} bytes): {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != UPSTREAM_SHA256:
        raise AttemptFailed(f"sha256 {got}", sha=got)
    return raw


def fetch_upstream(attempts, timeout, deadline_seconds):
    """Retry `attempt` until it returns the pinned CSV, or the attempts or the total
    `deadline_seconds` budget run out; then exit with a report naming every attempt."""
    deadline = time.monotonic() + deadline_seconds
    reserve = RESERVE_FRACTION * deadline_seconds
    # No attempt starts after this: the reserve is kept for one full download.
    latest_start = deadline - reserve
    failed = []
    reserve_reached = False
    for i in range(attempts):
        # The one deadline gate: no attempt starts without time left.
        if time.monotonic() >= deadline:
            break
        print(f"downloading {URL} (attempt {i + 1}/{attempts}) ...", file=sys.stderr)
        try:
            return attempt(URL, timeout, deadline)
        except AttemptFailed as e:
            failed.append(e)
            print(f"attempt {i + 1}: {e.log}", file=sys.stderr)
        if i + 1 >= attempts:  # no sleep after the last attempt
            break
        now = time.monotonic()
        if now >= latest_start:  # the attempt at `latest_start` was the last one
            reserve_reached = True
            break
        wait = min(BACKOFF_BASE_SECONDS * 2**i, BACKOFF_CAP_SECONDS)
        time.sleep(max(0.0, min(wait, latest_start - now)))
    out_of_time = time.monotonic() >= deadline
    sys.exit(
        fetch_report(
            failed, out_of_time, deadline_seconds, reserve if reserve_reached else None
        )
    )


def fetch_report(failed, out_of_time, deadline_seconds, reserve=None):
    """The exit message once every attempt failed: the verdict, then one line each.

    `reserve` is set when the loop stopped at `latest_start` with attempts left; it is
    reported only when the deadline itself had not passed, which wins.

    The verdict reads the complete downloads' hashes only. A transport error or an
    unreadable payload says nothing about the file, so it neither confirms nor
    contradicts a repeated hash.
    """
    n = len(failed)
    hashes = [e.sha for e in failed if e.sha is not None]
    distinct = set(hashes)
    if len(hashes) >= 2 and len(distinct) == 1:
        lines = [
            f"upstream {INNER_CSV} hashed sha256 {hashes[0]} on all {len(hashes)} "
            f"complete downloads, not the pinned {UPSTREAM_SHA256}.",
            "The same wrong hash on every complete download means the upstream file "
            "really changed: inspect it, then re-pin UPSTREAM_SHA256 (and "
            "DERIVED_SHA256) deliberately.",
        ]
    else:
        lines = [
            f"no attempt of {n} delivered the pinned upstream {INNER_CSV} "
            f"(sha256 {UPSTREAM_SHA256}).",
        ]
        if len(distinct) >= 2:
            lines.append(
                f"{len(distinct)} different wrong hashes mean the endpoint is flaky, not "
                "that the file changed: re-run the fetch, and do not re-pin."
            )
        else:
            lines.append(
                "Fewer than two complete downloads cannot tell a changed file from a "
                "flaky endpoint: re-run the fetch."
            )
    if out_of_time:
        lines.append(f"Stopped at the {deadline_seconds:g} s --deadline-seconds budget.")
    elif reserve is not None:
        lines.append(
            f"No attempt starts in the last {reserve:g} s of the {deadline_seconds:g} s "
            "--deadline-seconds budget: too little time for a full download."
        )
    lines += [f"  attempt {i + 1}: {e.log}" for i, e in enumerate(failed)]
    return "\n".join(lines)


def upstream_csv(blob):
    """The bytes of `data_9studies.csv`, from the outer zip, the inner zip or the CSV.

    Raises `ValueError` on a zip that holds neither."""
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        return blob
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    if INNER_CSV in names:
        return z.read(INNER_CSV)
    inner = [n for n in names if os.path.basename(n) == INNER_ZIP]
    if len(inner) != 1:
        raise ValueError(f"neither {INNER_CSV} nor {INNER_ZIP} found in the zip ({len(names)} entries)")
    return upstream_csv(z.read(inner[0]))


def derive(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    with io.StringIO(raw.decode("utf-8")) as f:
        header = next(csv.reader(f))
    if header != EXPECTED_COLUMNS:
        sys.exit(f"unexpected header {header}")
    if len(rows) != EXPECTED_ROWS:
        sys.exit(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len({r["STUDY"] for r in rows}) != EXPECTED_STUDIES:
        sys.exit("unexpected number of studies")
    narm = {}
    for r in rows:
        if r["DRUG"] not in DRUGS + ("placebo",):
            sys.exit(f"unexpected DRUG {r['DRUG']!r}")
        # The drug indicators are the whole covariate model: a recoded ARM/DRUG
        # pairing would silently move an arm onto the wrong drug effect.
        if (r["ARM"] == "0") != (r["DRUG"] == "placebo"):
            sys.exit(f"ARM 0 must be exactly the placebo arms: {r}")
        key = (r["STUDY"], r["ARM"])
        if narm.setdefault(key, r["Narm"]) != r["Narm"]:
            sys.exit(f"Narm not constant within study/arm {key}")
    if len(narm) != EXPECTED_ARMS:
        sys.exit(f"expected {EXPECTED_ARMS} arms, got {len(narm)}")

    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(OUT_COLUMNS)
    for r in rows:
        n = int(r["Narm"])
        dv = float(r["transY"]) / math.sqrt(n)
        w.writerow(
            [r["STUDY"], r["ARM"], r["TIME"], repr(dv), n]
            + [int(r["DRUG"] == d) for d in DRUGS]
            + [0]
        )
    return out.getvalue().encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", help="local outer zip, inner zip, or data_9studies.csv")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--allow-new-hash",
        action="store_true",
        help="accept a new DERIVED sha256; the upstream pin is never waived",
    )
    ap.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="most download attempts (default 10; at the default deadline the "
        "budget, not the count, ends the loop: up to 8 attempts fit, with fast "
        "failures)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="seconds any one socket operation may block (default 120)",
    )
    ap.add_argument(
        "--deadline-seconds",
        type=float,
        default=720,
        help="wall-time budget for all attempts and backoffs together (default 720 = "
        "12 min: plus one --timeout, inside slow-tests.yml's 15-min step cap)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="only verify that --out already holds the pinned derived file; exit 1 if not",
    )
    a = ap.parse_args()
    if a.attempts < 1:
        ap.error(f"--attempts: must be at least 1, got {a.attempts}")
    if not (a.timeout > 0 and a.deadline_seconds > 0):
        ap.error("--timeout and --deadline-seconds must be positive")
    if a.check:
        try:
            got = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
        except OSError as e:
            sys.exit(f"--check: cannot read {a.out}: {e}")
        if got != DERIVED_SHA256:
            sys.exit(f"--check: {a.out} sha256 {got} != pinned {DERIVED_SHA256}")
        print(f"ok: {a.out} is the pinned derived file")
        return
    # The loopback arm exists for `tests/mbma_canakinumab_fetch_script.rs`, which
    # points URL at a local fake server; the constant itself is https.
    if not (URL.startswith("https://") or URL.startswith("http://127.0.0.1:")):
        sys.exit(f"refusing a non-https download URL: {URL}")

    if a.src:
        try:
            raw = upstream_csv(open(a.src, "rb").read())
        except (ValueError, zipfile.BadZipFile) as e:
            sys.exit(f"--from {a.src}: {e}")
        got = hashlib.sha256(raw).hexdigest()
        if got != UPSTREAM_SHA256:
            sys.exit(f"upstream {INNER_CSV} sha256 {got} != pinned {UPSTREAM_SHA256}")
    else:
        # The upstream pin is checked inside the retry loop, so a wrong hash is a
        # failed attempt and never an accepted payload; `--allow-new-hash` does not
        # reach it.
        raw = fetch_upstream(a.attempts, a.timeout, a.deadline_seconds)

    derived = derive(raw)
    got = hashlib.sha256(derived).hexdigest()
    if got != DERIVED_SHA256 and not a.allow_new_hash:
        sys.exit(f"derived sha256 {got} != pinned {DERIVED_SHA256} (pass --allow-new-hash)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(a.out) or ".", delete=False) as t:
        t.write(derived)
    os.replace(t.name, a.out)
    print(f"wrote {a.out} ({EXPECTED_ROWS} rows, sha256 {got})")


if __name__ == "__main__":
    main()
