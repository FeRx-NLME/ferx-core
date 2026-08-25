"""Convert the published naproxen/osteoarthritis MBMA dataset to ferx layout.

Usage:
    python3 data/mbma_naproxen_source.py <path-to>/naproxen_data_transformed_se.csv \\
        [output.csv]

Source: `naproxen_data_transformed_se.csv`, supplementary material to

    Bracis C, Taneja A, Lyauk YK, Barcomb H, de la Peña A, Cellière G.
    Model-Based Meta-Analysis With MonolixSuite: A Tutorial for Longitudinal
    Categorical and Continuous Data. CPT Pharmacometrics Syst Pharmacol.
    2026;15:e70158.

18 trials x 2 arms, longitudinal WOMAC pain. See `data/mbma_naproxen.README.md`
for the provenance and reuse terms of the committed copy.

Transformations, all of them mechanical:
    ID     <- STUD   (study is the unit of between-study variability)
    DV     <- transWP = WP / WPSE   (the SE-weighted response, already in the file)
    TRT    <- 1 for Naproxen, 0 for Placebo
    ARM    <- ARM, kept for the between-treatment-arm occasion (unused by the
              Case Study 1 model, which is BSV-only)
FLARE, WPSE and Nindiv pass through unchanged.

The weighting scheme is the published one: the prediction is divided by the arm's
reported standard error and the residual variance is FIXED to 1, so each arm mean
enters the likelihood with weight 1/SE^2.
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
        trt = 1 if r["TRT"].strip().lower() == "naproxen" else 0
        out.append(
            ",".join(
                [
                    r["STUD"],
                    r["ARM"],
                    r["TIME"],
                    f'{float(r["transWP"]):.6f}',
                    str(trt),
                    r["FLARE"],
                    f'{float(r["WPSE"]):.9f}',
                    r["Nindiv"],
                    "0",
                ]
            )
        )

    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")

    studies = {r["STUD"] for r in rows}
    arms = {(r["STUD"], r["ARM"]) for r in rows}
    print(f"wrote {dst}: {len(rows)} rows, {len(studies)} studies, {len(arms)} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
