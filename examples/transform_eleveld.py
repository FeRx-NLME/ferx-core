#!/usr/bin/env python3
"""Transform Eleveld's $PRED benchmark CSV into a ferx event-driven dataset.

The NONMEM model in FeRx-NLME discussion #179 is a $PRED closed form: it has no
dose records and hardcodes D=100 (with an extra *100 in the CP expression). The
ferx port uses the event-driven analytical `one_cpt_oral`, which needs:

  * a dose row per subject at TIME=0 with AMT = D*100 = 10000, EVID=1, CMT=1
  * DV pointing at the natural-scale concentration (the RDV column), since
    `log(DV) ~ additive` logs both sides itself. The source `DV` column is
    already log(conc), so we must NOT reuse it.

Input columns:  NSIM,ID,TIME,DOSE,RDV,DV
Output columns: ID,TIME,DV,AMT,EVID,CMT,MDV

Usage:
    python3 transform_eleveld.py simdata/data.1.txt --nsim 1 -o data_nsim1.csv
"""
import argparse
import csv
import sys

DOSE_AMT = 10000.0  # D (100) * the extra *100 factor in the CP expression


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="source CSV (NSIM,ID,TIME,DOSE,RDV,DV)")
    ap.add_argument("--nsim", type=int, default=1,
                    help="simulation replicate to extract (default: 1)")
    ap.add_argument("-o", "--outfile", default="-",
                    help="output CSV path (default: stdout)")
    args = ap.parse_args()

    with open(args.infile, newline="") as fh:
        rows = list(csv.DictReader(fh))

    rows = [r for r in rows if int(r["NSIM"]) == args.nsim]
    if not rows:
        sys.exit(f"no rows with NSIM={args.nsim} in {args.infile}")

    out = open(args.outfile, "w", newline="") if args.outfile != "-" else sys.stdout
    w = csv.writer(out)
    w.writerow(["ID", "TIME", "DV", "AMT", "EVID", "CMT", "MDV"])

    seen = set()
    for r in rows:
        sid = r["ID"]
        if sid not in seen:
            seen.add(sid)
            # one dose event per subject at t=0
            w.writerow([sid, 0, ".", DOSE_AMT, 1, 1, 1])
        # observation: DV = RDV (natural-scale concentration)
        w.writerow([sid, r["TIME"], r["RDV"], ".", 0, 1, 0])

    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
