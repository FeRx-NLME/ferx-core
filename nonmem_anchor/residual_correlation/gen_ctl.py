"""Generate NONMEM L2 control streams for #1081.
Each arm is one L2 record (EPS shared across its rows). SIGMA = diag(1) FIX over
5 EPS; row VIS=j carries RES = sum_k L[j,k](rho) EPS(k), L = Cholesky of the arm
correlation C(rho); Y = F + WPSE*RES  =>  R_arm = W C W.  rho = THETA(8)."""
import shutil
import sys, os
TH = [5.1972, 1.1580, 0.6901, 0.9629, 0.8197, 0.7930, -1.1747]
OM = [0.6215**2, 0.7387**2]

CHOL = {
'cs': """
; compound symmetry: column-k off-diagonal a_k = (rho - S_{k-1})/sqrt(1 - S_{k-1})
  A1 = RHO
  S1 = A1*A1
  A2 = (RHO - S1)/SQRT(1 - S1)
  S2 = S1 + A2*A2
  A3 = (RHO - S2)/SQRT(1 - S2)
  S3 = S2 + A3*A3
  A4 = (RHO - S3)/SQRT(1 - S3)
  S4 = S3 + A4*A4
  L1=0
  L2C=0
  L3=0
  L4=0
  L5=0
  IF (VIS.EQ.1) L1 = 1
  IF (VIS.EQ.2) THEN
    L1 = A1
    L2C = SQRT(1 - S1)
  ENDIF
  IF (VIS.EQ.3) THEN
    L1 = A1
    L2C = A2
    L3 = SQRT(1 - S2)
  ENDIF
  IF (VIS.EQ.4) THEN
    L1 = A1
    L2C = A2
    L3 = A3
    L4 = SQRT(1 - S3)
  ENDIF
  IF (VIS.EQ.5) THEN
    L1 = A1
    L2C = A2
    L3 = A3
    L4 = A4
    L5 = SQRT(1 - S4)
  ENDIF
""",
'ar1': """
; discrete AR(1) on the visit index: L[j,k] = rho^(j-k) * s_k, s_1 = 1, s_k = sqrt(1-rho^2)
  SR = SQRT(1 - RHO*RHO)
  P1 = RHO
  P2 = RHO*RHO
  P3 = P2*RHO
  P4 = P3*RHO
  L1=0
  L2C=0
  L3=0
  L4=0
  L5=0
  IF (VIS.EQ.1) L1 = 1
  IF (VIS.EQ.2) THEN
    L1 = P1
    L2C = SR
  ENDIF
  IF (VIS.EQ.3) THEN
    L1 = P2
    L2C = P1*SR
    L3 = SR
  ENDIF
  IF (VIS.EQ.4) THEN
    L1 = P3
    L2C = P2*SR
    L3 = P1*SR
    L4 = SR
  ENDIF
  IF (VIS.EQ.5) THEN
    L1 = P4
    L2C = P3*SR
    L3 = P2*SR
    L4 = P1*SR
    L5 = SR
  ENDIF
""",
'car': """
; continuous-time AR(1): corr = rho^|dt|; L[j,k] = rho^(t_j - t_k) * s_k,
; s_1 = 1, s_k = sqrt(1 - rho^(2 (t_k - t_{k-1}))).  0 < rho < 1.
  LR = LOG(RHO)
  Q2 = 0
  Q3 = 0
  Q4 = 0
  Q5 = 0
  IF (VIS.GE.2) Q2 = SQRT(1 - EXP(2*LR*(T2-T1)))
  IF (VIS.GE.3) Q3 = SQRT(1 - EXP(2*LR*(T3-T2)))
  IF (VIS.GE.4) Q4 = SQRT(1 - EXP(2*LR*(T4-T3)))
  IF (VIS.GE.5) Q5 = SQRT(1 - EXP(2*LR*(T5-T4)))
  L1=0
  L2C=0
  L3=0
  L4=0
  L5=0
  IF (VIS.EQ.1) L1 = 1
  IF (VIS.GE.2) L1 = EXP(LR*(TIME-T1))
  IF (VIS.GE.2) L2C = EXP(LR*(TIME-T2))*Q2
  IF (VIS.GE.3) L3 = EXP(LR*(TIME-T3))*Q3
  IF (VIS.GE.4) L4 = EXP(LR*(TIME-T4))*Q4
  IF (VIS.GE.5) L5 = EXP(LR*(TIME-T5))*Q5
""",
}

def ctl(struct, rho, estimate):
    fix = '' if estimate else ' FIX'
    lo, hi = {'cs': (-0.24, 0.99), 'ar1': (-0.99, 0.99), 'car': (0.001, 0.999)}[struct]
    th = '\n'.join(f'  {v}{fix}' for v in TH)
    est = ('$ESTIMATION METHOD=1 MAXEVAL=9999 SIGDIG=4 NOABORT PRINT=5\n$COVARIANCE MATRIX=R UNCONDITIONAL'
           if estimate else '$ESTIMATION METHOD=1 MAXEVAL=0 NOABORT POSTHOC PRINT=1')
    return f"""$PROBLEM ferx #1081 naproxen MBMA, within-arm residual correlation {struct} rho={rho} {'EST' if estimate else 'EVAL'}
; Data: napx1081.csv (natural-scale DV = DV*WPSE; L2 = arm; VIS = visit index; T1..T5 arm visit times)
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
  RES = L1*EPS(1) + L2C*EPS(2) + L3*EPS(3) + L4*EPS(4) + L5*EPS(5)
  IPRED = F
  Y = F + WPSE*RES
$THETA
{th}
  ({lo}, {rho}, {hi}){fix}
$OMEGA {OM[0]}{fix}
$OMEGA {OM[1]}{fix}
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
{est}
$TABLE ID ARM TIME DV IPRED L2 VIS ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE={struct}_{'est' if estimate else 'eval'}.tab
"""

RUNS = [('cs', 0.0, False, 'cs_r0'), ('cs', 0.5, False, 'cs_p05'), ('cs', -0.2, False, 'cs_m02'),
        ('ar1', 0.5, False, 'ar1_p05'), ('ar1', -0.4, False, 'ar1_m04'),
        ('car', 0.8, False, 'car_p08'), ('car', 0.3, False, 'car_p03'),
        ('cs', 0.3, True, 'cs_est'), ('ar1', 0.3, True, 'ar1_est'), ('car', 0.5, True, 'car_est')]
if __name__ == '__main__':
    for struct, rho, est, name in RUNS:
        os.makedirs(name, exist_ok=True)
        open(f'{name}/{name}.ctl', 'w').write(ctl(struct, rho, est))
        shutil.copy('napx1081.csv', f'{name}/')
    print('ok')
