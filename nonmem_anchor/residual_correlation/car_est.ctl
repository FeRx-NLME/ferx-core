$PROBLEM ferx #1081 naproxen MBMA, within-arm residual correlation car rho=0.5 EST
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

  RES = L1*EPS(1) + L2C*EPS(2) + L3*EPS(3) + L4*EPS(4) + L5*EPS(5)
  IPRED = F
  Y = F + WPSE*RES
$THETA
  5.1972
  1.158
  0.6901
  0.9629
  0.8197
  0.793
  -1.1747
  (0.001, 0.5, 0.999)
$OMEGA 0.3862622500000001
$OMEGA 0.54567769
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=1 MAXEVAL=9999 SIGDIG=4 NOABORT PRINT=5
$COVARIANCE MATRIX=R UNCONDITIONAL
$TABLE ID ARM TIME DV IPRED L2 VIS ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=car_est.tab
