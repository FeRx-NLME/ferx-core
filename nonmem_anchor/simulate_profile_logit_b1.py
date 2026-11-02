# #2111 / #2562 anchor B1 (synthetic, fixed seed). Pure Python, no dependencies.
#
#   python3 simulate_profile_logit_b1.py [outdir]   -> profile_logit_b1.csv
#
# A binary MBMA-like draw: per study (ID = study), visit and arm, NPAT Bernoulli patients
# with logit p = TVL + PLACEBO[STUDY, TIME] + ETA_S + EMAX * TRT, one row per patient. The
# model (`profile_logit_b1.ferx`) reads them through a `[binary_model]` predictor. These are
# the draws `simulate_profile_logit_b2.py` aggregates into observed proportions: the same
# seed and the same order of calls.
import math, os, random, sys

random.seed(2111)
out = sys.argv[1] if len(sys.argv) > 1 else "."
NSTUDY, NPAT = 10, 30
TIMES = [0.0, 4.0, 8.0, 12.0]
TVL, EMAX, SD_ETA = -0.5, 1.0, 0.5
PLC_MEAN = [-0.4, 0.0, 0.2, 0.2]


def expit(x):
    return 1.0 / (1.0 + math.exp(-x))


rows = ["ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT"]
for s in range(1, NSTUDY + 1):
    eta = random.gauss(0.0, SD_ETA)
    plc = [m + random.gauss(0.0, 0.3) for m in PLC_MEAN]
    for t, pl in zip(TIMES, plc):
        for trt in (0, 1):
            p = expit(TVL + pl + eta + EMAX * trt)
            for _ in range(NPAT):
                rows.append(f"{s},{s},{t:g},{1 if random.random() < p else 0},1,0,0,{trt}")
open(os.path.join(out, "profile_logit_b1.csv"), "w").write("\n".join(rows) + "\n")
print(len(rows) - 1)
