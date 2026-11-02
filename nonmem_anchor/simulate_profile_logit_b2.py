# #2111 anchor B2 (synthetic, fixed seed). Pure Python, no dependencies.
#
#   python3 simulate_profile_logit_b2.py [outdir]   -> profile_logit_b2.csv
#
# A binary MBMA-like draw, aggregated: per study (ID = study), visit and arm, NPAT
# Bernoulli patients with logit p = TVL + PLACEBO[STUDY, TIME] + ETA_S + EMAX * TRT, and
# DV the observed proportion. The model (`profile_logit_b2.ferx`) reads it as a bounded
# continuous endpoint, y = inv_logit(TVL + PLACEBO + ETA_S + EMAX * TRT), additive error.
# The individual Bernoulli draws are made (and discarded) so the random stream stays the
# one the binary fixture of #2111 is drawn from.
import math, os, random, sys

random.seed(2111)
out = sys.argv[1] if len(sys.argv) > 1 else "."
NSTUDY, NPAT = 10, 30
TIMES = [0.0, 4.0, 8.0, 12.0]
TVL, EMAX, SD_ETA = -0.5, 1.0, 0.5
PLC_MEAN = [-0.4, 0.0, 0.2, 0.2]


def expit(x):
    return 1.0 / (1.0 + math.exp(-x))


rows = ["ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT,NARM"]
for s in range(1, NSTUDY + 1):
    eta = random.gauss(0.0, SD_ETA)
    plc = [m + random.gauss(0.0, 0.3) for m in PLC_MEAN]
    for t, pl in zip(TIMES, plc):
        for trt in (0, 1):
            p = expit(TVL + pl + eta + EMAX * trt)
            ys = [1 if random.random() < p else 0 for _ in range(NPAT)]
            rows.append(f"{s},{s},{t:g},{sum(ys)/NPAT:.6f},1,0,0,{trt},{NPAT}")
open(os.path.join(out, "profile_logit_b2.csv"), "w").write("\n".join(rows) + "\n")
print(len(rows) - 1)
