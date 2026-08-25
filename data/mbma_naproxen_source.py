"""Convert the published naproxen/osteoarthritis MBMA dataset to ferx layout.

Usage:
    python3 data/mbma_naproxen_source.py <path-to>/PSP4-7-288-s007.csv \\
        [output.csv]

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
FLARE, TIME and WPSE pass through unchanged.

`DV` is computed here rather than read from the file: the upstream CSV carries the
arm mean (`WP`) and its standard error (`WPSE`), not their ratio.

The weighting scheme is the published one: the prediction is divided by the arm's
reported standard error and the residual variance is FIXED to 1, so each arm mean
enters the likelihood with weight 1/SE^2. Boucher & Bennetts, Methods: "As the
weights were based on the observed SEs, sigma was fixed to 1 in the modeling."
"""

import csv
import sys

DEFAULT_DST = "data/mbma_naproxen.csv"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    src, dst = argv[1], (argv[2] if len(argv) > 2 else DEFAULT_DST)

    with open(src) as fh:
        rows = list(csv.DictReader(fh))

    out = ["ID,ARM,TIME,DV,TRT,FLARE,WPSE,NIND,MDV"]
    for r in rows:
        wpse = float(r["WPSE"])
        dv = float(r["WP"]) / wpse
        out.append(
            ",".join(
                [
                    r["STUD"],
                    r["NFLG"],
                    r["TIME"],
                    f"{dv:.6f}",
                    r["NFLG"],
                    r["FLARE"],
                    f"{wpse:.9f}",
                    r["NTRT"],
                    "0",
                ]
            )
        )

    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")

    studies = {r["STUD"] for r in rows}
    arms = {(r["STUD"], r["NFLG"]) for r in rows}
    print(f"wrote {dst}: {len(rows)} rows, {len(studies)} studies, {len(arms)} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
