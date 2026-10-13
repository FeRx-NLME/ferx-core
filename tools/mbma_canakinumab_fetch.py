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
hash-pinned. The endpoint is slow (minutes for ~10 MB) and has returned HTTP 500,
hence the retries. `--from` takes a local copy of the outer zip, of
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
import io
import math
import os
import sys
import tempfile
import time
import urllib.request
import zipfile

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12896392/supplementaryFiles"
INNER_ZIP = "PSP4-15-e70158-s002.zip"
INNER_CSV = "supplemental/case_study_2/data_9studies.csv"
DEFAULT_OUT = "data/external/mbma_canakinumab.csv"

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


def download(attempts=4, timeout=600):
    last = None
    for i in range(attempts):
        try:
            print(f"downloading {URL} (attempt {i + 1}/{attempts}) ...", file=sys.stderr)
            # URL is the https constant above, never user input (asserted in main).
            with urllib.request.urlopen(URL, timeout=timeout) as r:  # nosec B310
                return r.read()
        except Exception as e:  # noqa: BLE001 — any failure is retried, then reported
            last = e
            time.sleep(10 * (i + 1))
    sys.exit(f"download failed after {attempts} attempts: {last}")


def upstream_csv(blob):
    """The bytes of `data_9studies.csv`, from the outer zip, the inner zip or the CSV."""
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        return blob
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    if INNER_CSV in names:
        return z.read(INNER_CSV)
    inner = [n for n in names if os.path.basename(n) == INNER_ZIP]
    if len(inner) != 1:
        sys.exit(f"neither {INNER_CSV} nor {INNER_ZIP} found in the zip ({len(names)} entries)")
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
    ap.add_argument("--allow-new-hash", action="store_true")
    ap.add_argument(
        "--check",
        action="store_true",
        help="only verify that --out already holds the pinned derived file; exit 1 if not",
    )
    a = ap.parse_args()
    if a.check:
        try:
            got = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
        except OSError as e:
            sys.exit(f"--check: cannot read {a.out}: {e}")
        if got != DERIVED_SHA256:
            sys.exit(f"--check: {a.out} sha256 {got} != pinned {DERIVED_SHA256}")
        print(f"ok: {a.out} is the pinned derived file")
        return
    if not URL.startswith("https://"):
        sys.exit(f"refusing a non-https download URL: {URL}")

    blob = open(a.src, "rb").read() if a.src else download()
    raw = upstream_csv(blob)
    got = hashlib.sha256(raw).hexdigest()
    if got != UPSTREAM_SHA256:
        sys.exit(f"upstream {INNER_CSV} sha256 {got} != pinned {UPSTREAM_SHA256}")

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
