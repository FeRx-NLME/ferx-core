"""Convert the published naproxen/osteoarthritis MBMA dataset to ferx layout.

Usage:
    python3 data/mbma_naproxen_source.py <path-to>/PSP4-7-288-s007.csv \\
        [output.csv] [--allow-new-hash]

Source: `PSP4-7-288-s007.csv`, supplementary material to

    Boucher M, Bennetts M. Many Flavors of Model-Based Meta-Analysis:
    Part II - Modeling Summary Level Longitudinal Responses.
    CPT Pharmacometrics Syst Pharmacol. 2018;7(5):288-297.
    doi:10.1002/psp4.12299

published under CC BY-NC 4.0. 18 trials x 2 arms, longitudinal WOMAC pain.
See `data/mbma_naproxen.README.md` for the provenance and reuse terms of the
committed copy, and `data/mbma_naproxen.LICENSE` for the licence itself.

Transformations, all of them mechanical:
    ID     <- STUD   (study is the unit of between-study variability)
    ARM    <- NFLG   (0 = placebo arm, 1 = naproxen arm; kept as the occasion
                      for a between-arm kappa, unused by the shipped model,
                      which is BSV-only)
    TRT    <- NFLG   (the same indicator, read as a treatment covariate)
    DV     <- WP / WPSE, the SE-weighted response
    NIND   <- NTRT   (subjects behind the arm)
FLARE and TIME pass through unchanged; WPSE passes through its value but is
reformatted to 9 decimals, which the script verifies is exact for every value in
the input rather than assuming it.

`DV` is computed here rather than read from the file: the upstream CSV carries the
arm mean (`WP`) and its standard error (`WPSE`), not their ratio.

The weighting scheme is the published one: the prediction is divided by the arm's
reported standard error and the residual variance is FIXED to 1, so each arm mean
enters the likelihood with weight 1/SE^2. Boucher & Bennetts, Methods: "As the
weights were based on the observed SEs, sigma was fixed to 1 in the modeling."

Pass `--allow-new-hash` to write an output whose sha256 differs from the
committed fixture's. That is a deliberate refresh, and the new hash has to be
recorded in `data/mbma_naproxen.README.md`; every other check still applies.
"""

import csv
import hashlib
import sys

DEFAULT_DST = "data/mbma_naproxen.csv"

# The published file's exact shape. `NFLG` is written into both `ARM` and `TRT`
# without inspecting its contents, so on its own nothing downstream would notice
# if a re-issued supplementary file renamed a column or recoded the flag — the
# previous derivation at least tested `TRT == "Naproxen"` and would have miscoded
# loudly. Three checks replace that, and they catch different things:
#   * the shape checks below reject a file that is not the one this script was
#     written for;
#   * `check_naproxen_arms_improve` reads what the flag *means*, so a file that
#     kept the shape but swapped 0 and 1 fails instead of silently emitting an
#     inverted `TRT` (which would estimate the naproxen effect on Emax with the
#     wrong sign, and nothing further down the pipeline would notice);
#   * the output sha256 is compared against the committed fixture's, which is the
#     invariant `data/mbma_naproxen.README.md` states — any difference at all is a
#     finding, and has to be acknowledged with `--allow-new-hash`.
EXPECTED_COLUMNS = {"STUD", "NFLG", "TIME", "NTRT", "FLARE", "WP", "WPSE"}
EXPECTED_ROWS = 122
EXPECTED_FLAGS = {"0", "1"}
EXPECTED_STUDIES = 18
# sha256 of the derived CSV committed at `data/mbma_naproxen.csv`, also pinned in
# `data/mbma_naproxen.README.md`.
EXPECTED_SHA256 = "4370af48bfda41872f4b7f09a44a4a31c72c80582099c58aa67c965c84641ab4"

# `WPSE` is written to 9 decimals but `DV` is computed from the full-precision
# parse, so a value needing a 10th decimal would leave the committed `DV` divided
# by a different number than the `WPSE` the model divides the prediction by —
# silently perturbing the `1/WPSE^2` weighting the whole analysis rests on.
WPSE_DECIMALS = 9


def check_shape(src, columns, rows):
    """Reject an input file that is not the PSP4-7-288-s007.csv derived from."""
    if columns != EXPECTED_COLUMNS:
        print(
            f"error: {src} has columns {sorted(columns)}, expected "
            f"{sorted(EXPECTED_COLUMNS)}. This is not the PSP4-7-288-s007.csv this "
            "script derives from; see data/mbma_naproxen.README.md.",
            file=sys.stderr,
        )
        return False
    if len(rows) != EXPECTED_ROWS:
        print(
            f"error: {src} has {len(rows)} rows, expected {EXPECTED_ROWS}.",
            file=sys.stderr,
        )
        return False
    flags = {r["NFLG"].strip() for r in rows}
    if flags != EXPECTED_FLAGS:
        print(
            f"error: NFLG holds {sorted(flags)}, expected a 0/1 placebo/naproxen "
            "indicator. TRT would be miscoded.",
            file=sys.stderr,
        )
        return False
    return True


def check_naproxen_arms_improve(rows):
    """Check that `NFLG == 1` is the arm with *less* pain, not more.

    The shape checks pass unchanged on a re-issued file that recoded the flag
    (1 = placebo), and `TRT` would then be inverted. Read the response instead:
    in each study, compare the two arms at the last time point they share. Every
    one of the 18 published trials has the `NFLG == 1` arm lower there, so a
    two-thirds majority is a wide margin, and an inverted flag scores zero.
    """
    by_study = {}
    for r in rows:
        arms = by_study.setdefault(r["STUD"].strip(), {})
        arms.setdefault(r["NFLG"].strip(), {})[float(r["TIME"])] = float(r["WP"])

    paired, favouring = 0, 0
    for arms in by_study.values():
        placebo, naproxen = arms.get("0", {}), arms.get("1", {})
        shared = set(placebo) & set(naproxen)
        if not shared:
            continue
        paired += 1
        last = max(shared)
        if naproxen[last] < placebo[last]:
            favouring += 1

    if paired != EXPECTED_STUDIES:
        print(
            f"error: {paired} studies have both a 0 and a 1 arm at a shared time, "
            f"expected {EXPECTED_STUDIES}. The 0/1 flag is not the arm indicator "
            "this script reads it as.",
            file=sys.stderr,
        )
        return False
    if favouring * 3 < paired * 2:
        print(
            f"error: only {favouring} of {paired} studies have the NFLG==1 arm "
            "below the NFLG==0 arm at their last shared time. NFLG does not mean "
            "1 = naproxen here, so ARM and TRT would be inverted and the naproxen "
            "effect on Emax would estimate with the wrong sign.",
            file=sys.stderr,
        )
        return False
    return True


def derive(rows):
    """Build the output lines, or return None if a row cannot be derived exactly."""
    out = ["ID,ARM,TIME,DV,TRT,FLARE,WPSE,NIND,MDV"]
    for r in rows:
        stud, flag = r["STUD"].strip(), r["NFLG"].strip()
        wpse = float(r["WPSE"].strip())
        if float(f"{wpse:.{WPSE_DECIMALS}f}") != wpse:
            print(
                f"error: WPSE {r['WPSE'].strip()!r} (study {stud}, time "
                f"{r['TIME'].strip()}) is not exact at {WPSE_DECIMALS} decimals. DV "
                "would be divided by a different value than the WPSE written beside "
                "it, perturbing the 1/WPSE^2 weighting.",
                file=sys.stderr,
            )
            return None
        dv = float(r["WP"].strip()) / wpse
        out.append(
            ",".join(
                [
                    stud,
                    flag,
                    r["TIME"].strip(),
                    f"{dv:.6f}",
                    flag,
                    r["FLARE"].strip(),
                    f"{wpse:.{WPSE_DECIMALS}f}",
                    r["NTRT"].strip(),
                    "0",
                ]
            )
        )
    return out


def main(argv):
    args = [a for a in argv[1:] if a != "--allow-new-hash"]
    allow_new_hash = len(args) != len(argv) - 1
    if not args:
        print(__doc__)
        return 2
    src, dst = args[0], (args[1] if len(args) > 1 else DEFAULT_DST)

    with open(src) as fh:
        reader = csv.DictReader(fh)
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    if not check_shape(src, columns, rows):
        return 1
    if not check_naproxen_arms_improve(rows):
        return 1

    out = derive(rows)
    if out is None:
        return 1

    text = "\n".join(out) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != EXPECTED_SHA256 and not allow_new_hash:
        print(
            f"error: the derived CSV is sha256 {digest}, expected "
            f"{EXPECTED_SHA256}. Per data/mbma_naproxen.README.md that difference "
            "is a finding: work out what changed before accepting it. Re-run with "
            "--allow-new-hash to write it anyway, and record the new hash in the "
            "README.",
            file=sys.stderr,
        )
        return 1

    with open(dst, "w") as fh:
        fh.write(text)

    studies = {r["STUD"].strip() for r in rows}
    arms = {(r["STUD"].strip(), r["NFLG"].strip()) for r in rows}
    note = "" if digest == EXPECTED_SHA256 else f", sha256 {digest} (NEW)"
    print(
        f"wrote {dst}: {len(rows)} rows, {len(studies)} studies, "
        f"{len(arms)} arms{note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
