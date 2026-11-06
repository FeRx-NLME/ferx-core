# #2603 anchor B1-probit (synthetic, fixed seed). Pure Python, no dependencies.
#
#   python3 simulate_probit_b1.py [outdir]   -> probit_b1.csv
#
# Fixture B1 of #2111/#2562 (`simulate_profile_logit_b1.py`) drawn under a probit link: per
# study (ID = study), visit and arm, NPAT Bernoulli patients with
# P(Y = 1) = Phi(TVL + PLACEBO[STUDY, TIME] + ETA_S + EMAX * TRT), one row per patient.
# Truth is on the probit scale (about 1/1.7 of B1's logit values), so the arm-visit
# probabilities again run from about 0.2 to 0.8 and the link is live everywhere. The model
# (`probit_b1.ferx`) reads them through a `[binary_model] probit` predictor.
import math, os, random, sys

random.seed(2603)
out = sys.argv[1] if len(sys.argv) > 1 else "."
NSTUDY, NPAT = 10, 30
TIMES = [0.0, 4.0, 8.0, 12.0]
TVL, EMAX, SD_ETA = -0.3, 0.6, 0.3
PLC_MEAN = [-0.25, 0.0, 0.12, 0.12]


def phi_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


rows = ["ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT"]
ps = []
for s in range(1, NSTUDY + 1):
    eta = random.gauss(0.0, SD_ETA)
    plc = [m + random.gauss(0.0, 0.18) for m in PLC_MEAN]
    for t, pl in zip(TIMES, plc):
        for trt in (0, 1):
            p = phi_cdf(TVL + pl + eta + EMAX * trt)
            ps.append(p)
            for _ in range(NPAT):
                rows.append(f"{s},{s},{t:g},{1 if random.random() < p else 0},1,0,0,{trt}")
open(os.path.join(out, "probit_b1.csv"), "w").write("\n".join(rows) + "\n")
print(len(rows) - 1, "rows; arm-visit p from %.3f to %.3f" % (min(ps), max(ps)))
