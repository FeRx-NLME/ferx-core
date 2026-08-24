"""Convert the published naproxen/osteoarthritis MBMA dataset to ferx layout.

Source: supplementary material to Bracis et al., CPT:PSP 2026;15:e70158
(`naproxen_data_transformed_se.csv`), 18 trials x 2 arms, longitudinal WOMAC
pain.

Transformations, all of them mechanical:
  ID     <- STUD   (study is the unit of between-study variability)
  DV     <- transWP = WP / WPSE   (the SE-weighted response, already in the file)
  TRT    <- 1 for Naproxen, 0 for Placebo
  ARM_ID <- ARM, kept for the between-treatment-arm occasion (unused by the
            Case Study 1 model, which is BSV-only)
FLARE, WPSE, Nindiv pass through unchanged.

The weighting scheme is the published one: the prediction is divided by the
arm's reported standard error and the residual variance is FIXED to 1, so each
arm mean enters the likelihood with weight 1/SE^2.
"""
import csv
import sys

SRC = "/Users/ron/git/mbma_introduction_book/source/supplemental/case_study_1/naproxen_data_transformed_se.csv"
DST = "data/mbma_naproxen.csv"

with open(SRC) as fh:
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

with open(DST, "w") as fh:
    fh.write("\n".join(out) + "\n")

studies = {r["STUD"] for r in rows}
arms = {(r["STUD"], r["ARM"]) for r in rows}
print(f"wrote {DST}: {len(rows)} rows, {len(studies)} studies, {len(arms)} arms")
sys.exit(0)
