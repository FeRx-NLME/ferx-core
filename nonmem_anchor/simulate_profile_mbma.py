"""Synthetic MBMA dataset and NONMEM control stream for the #1065 profile anchor (A2).

12 studies x 2 arms (OCC 1 placebo, OCC 2 drug) x 4 visits; ID = study. Arm means:

  y = TVE0 + PLACEBO[study, visit] + KAPPA_ARM
      - TRT * EMAX * exp(ETA_EM) * TIME / (TIME + ET50) + SE * eps

PLACEBO is an unstructured study-by-visit effect, centred within each study (the
`sum_to_zero_within` truth). The drug effect is nonlinear in the study's ETA_EM (sd
0.5), and the kappa is on the same rows as the block: the two things the existing
level_se anchor does not show NONMEM.

ETA_EM's sensitivity varies by visit (`t / (t + ET50)`), so it is NOT orthogonal to the
within-study contrast, and the EBE moves with the levels. That is what makes the
frozen-EBE shortcut the profile solve avoids measurably wrong here. A random effect
whose sensitivity is the same on every row of a study (`TVE0 * exp(ETA_E0)`, tried
first) is orthogonal to a sum-to-zero contrast, the EBE does not depend on the levels,
and the shortcut is exact: a degenerate fixture. Only one subject-level eta: with ID =
study and a level at every visit, the binder lets the block share a subject with one.
Fixed seed, standard-library `random` only, so the committed files are
reproducible from this script alone. Writes

  nonmem_anchor/profile_mbma.csv   the data (ferx and NONMEM read the same file)
  nonmem_anchor/profile_mbma.ctl   every level a THETA, the last visit dependent
"""
import math
import random

random.seed(1065)

N_STUDY = 12
TIMES = [1, 2, 4, 8]
TVE0, EMAX, ET50 = 45.0, 8.0, 2.0
SD_EM, SD_KAPPA, SD_PLACEBO = 0.5, 1.0, 1.5

rows = ["ID,STUDY,OCC,TRT,TIME,DV,MDV,SE"]
for s in range(1, N_STUDY + 1):
    raw = [random.gauss(0.0, SD_PLACEBO) for _ in TIMES]
    mean = sum(raw) / len(raw)
    placebo = [p - mean for p in raw]
    eta_em = random.gauss(0.0, SD_EM)
    for occ in (1, 2):
        trt = occ - 1
        kappa = random.gauss(0.0, SD_KAPPA)
        for t, pl in zip(TIMES, placebo):
            se = random.uniform(0.6, 1.2)
            drug = trt * EMAX * math.exp(eta_em) * t / (t + ET50)
            y = TVE0 + pl + kappa - drug + se * random.gauss(0.0, 1.0)
            rows.append(f"{s},{s},{occ},{trt},{t},{y:.4f},0,{se:.4f}")

with open("nonmem_anchor/profile_mbma.csv", "w") as fh:
    fh.write("\n".join(rows) + "\n")

# The control stream. ferx orders the levels by (STUDY, TIME) and, under
# sum_to_zero_within, makes each study's LAST visit the dependent one, so THETA(4 +
# 3(s-1) + j) is ferx's theta[3 + 3(s-1) + j].
pred = []
thetas = []
for s in range(1, N_STUDY + 1):
    base = 4 + 3 * (s - 1)
    for j, t in enumerate(TIMES):
        if j < 3:
            rhs = f"THETA({base + j})"
            thetas.append(f" (-50, 0.1, 50) ; PLACEBO[STUDY={s},TIME={t}]")
        else:
            rhs = f"-(THETA({base})+THETA({base + 1})+THETA({base + 2}))"
        pred.append(f" IF (STUDY.EQ.{s}.AND.TIME.EQ.{t}) PLA = {rhs}")

ctl = f"""$PROBLEM ferx #1065 profile anchor A2: unstructured placebo, BTAV kappa, drug effect nonlinear in ETA
$INPUT ID STUDY OCC TRT TIME DV MDV SE
$DATA profile_mbma.csv IGNORE=@
$PRED
 TVE0 = THETA(1)
 EMAX = THETA(2)
 ET50 = THETA(3)
 PLA = 0
{chr(10).join(pred)}
 KAP = ETA(2) * (2 - OCC) + ETA(3) * (OCC - 1)
 BASE = TVE0 + PLA + KAP
 DRUG = TRT * EMAX * EXP(ETA(1)) * TIME / (TIME + ET50)
 IPRED = BASE - DRUG
 Y = IPRED + SE * EPS(1)
$THETA
 (0, 45, 200)    ; TVE0
 (0, 8, 100)     ; EMAX
 (0.01, 2, 100)  ; ET50
{chr(10).join(thetas)}
$OMEGA 0.25     ; ETA_EM
$OMEGA BLOCK(1) 1   ; KAPPA_ARM, OCC 1
$OMEGA BLOCK(1) SAME ; KAPPA_ARM, OCC 2
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=10 SIGL=12 NSIG=4 FORMAT=s1PE23.16
$COVARIANCE MATRIX=R SIGL=12
"""
with open("nonmem_anchor/profile_mbma.ctl", "w") as fh:
    fh.write(ctl)
print(f"wrote {len(rows) - 1} rows, {N_STUDY} studies, {3 * N_STUDY + 3} THETA")
