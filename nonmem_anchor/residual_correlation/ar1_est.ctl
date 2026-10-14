$PROBLEM ferx #1081 naproxen MBMA, within-arm residual correlation ar1 rho=0.3 EST
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
  (-0.99, 0.3, 0.99)
$OMEGA 0.3862622500000001
$OMEGA 0.54567769
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=1 MAXEVAL=9999 SIGDIG=4 NOABORT PRINT=5
$COVARIANCE MATRIX=R UNCONDITIONAL
$TABLE ID ARM TIME DV IPRED L2 VIS ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=ar1_est.tab
