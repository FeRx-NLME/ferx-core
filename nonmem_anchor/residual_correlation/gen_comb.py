"""#1081 review r1 #9: combined-error twin. Proportional part F*THETA(9)*RESP and
weighted additive part WPSE*RESA, each correlated within the arm by the same C(rho):
RESP = sum_k L[j,k] EPS(k), RESA = sum_k L[j,k] EPS(5+k), SIGMA = I(10) FIX.
=> R_jk = C_jk (f_j f_k sp^2 + w_j w_k).  FOCEI (INTER): R depends on f(eta)."""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_ctl import CHOL, TH, OM

SDP = 0.03

def ctl(struct, rho):
    lo, hi = {'cs': (-0.24, 0.99), 'ar1': (-0.99, 0.99), 'car': (0.001, 0.999)}[struct]
    th = '\n'.join(f'  {v} FIX' for v in TH)
    return f"""$PROBLEM ferx #1081 naproxen MBMA, combined error, within-arm correlation {struct} rho={rho} EVAL FOCEI
; Data: napx1081.csv. R_jk = C_jk (f_j f_k THETA(9)^2 + WPSE_j WPSE_k)
$INPUT ID ARM TIME DV TRT FLARE WPSE MDV L2 VIS T1 T2 T3 T4 T5
$DATA napx1081.csv IGNORE=@
$PRED
  E0   = THETA(1) + THETA(4)*FLARE + ETA(1)
  EMAX = THETA(2) + THETA(5)*FLARE + THETA(6)*TRT + ETA(2)
  ET50 = THETA(3)*EXP(THETA(7)*TRT)
  EFF  = EMAX*TIME/(TIME + ET50)
  F    = E0 - EFF
  RHO  = THETA(8)
{CHOL[struct]}
  RESP = L1*EPS(1) + L2C*EPS(2) + L3*EPS(3) + L4*EPS(4) + L5*EPS(5)
  RESA = L1*EPS(6) + L2C*EPS(7) + L3*EPS(8) + L4*EPS(9) + L5*EPS(10)
  IPRED = F
  Y = F + F*THETA(9)*RESP + WPSE*RESA
$THETA
{th}
  ({lo}, {rho}, {hi}) FIX
  {SDP} FIX
$OMEGA {OM[0]} FIX
$OMEGA {OM[1]} FIX
""" + "$SIGMA 1 FIX\n" * 10 + """$ESTIMATION METHOD=1 INTER MAXEVAL=0 NOABORT POSTHOC PRINT=1
$TABLE ID ARM TIME DV IPRED L2 VIS ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=comb.tab
"""

RUNS = [('cs', 0.0, 'comb_cs_r0'), ('cs', 0.5, 'comb_cs_p05'), ('ar1', -0.4, 'comb_ar1_m04'),
        ('ar1', 0.5, 'comb_ar1_p05'), ('car', 0.3, 'comb_car_p03'), ('car', 0.8, 'comb_car_p08')]
if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    for struct, rho, name in RUNS:
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        open(f'{d}/{name}.ctl', 'w').write(ctl(struct, rho))
        shutil.copy(os.path.join(base, 'napx1081.csv'), f'{d}/')
    print('ok')
