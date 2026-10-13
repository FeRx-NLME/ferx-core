$PROBLEM ferx #1081 naproxen MBMA, within-arm residual correlation cs rho=0.3 EST
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
  (-0.24, 0.3, 0.99)
$OMEGA 0.3862622500000001
$OMEGA 0.54567769
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=1 MAXEVAL=9999 SIGDIG=4 NOABORT PRINT=5
$COVARIANCE MATRIX=R UNCONDITIONAL
$TABLE ID ARM TIME DV IPRED L2 VIS ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=cs_est.tab
