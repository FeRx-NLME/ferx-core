$PROBLEM In-domain inverse-Gaussian absorption -- NONMEM FOCEI anchor for ferx analytic pk one_cpt_ig (#790)
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA ig_indomain_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)
  COMP=(CENTRAL,DEFOBS)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)*EXP(ETA(2))
  MAT = THETA(3)
  CV2 = THETA(4)
  K20 = CL/V
  PI  = 3.14159265358979312
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  F1  = 0.0

$DES
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ARG = -(TAD - MAT)**2 / (2.0*CV2*MAT*TAD)
  RIN = PODO*SQRT(MAT/(2.0*PI*CV2*TAD**3))*EXP(ARG)
  DADT(1) = 0.0
  DADT(2) = RIN - K20*A(2)

$ERROR
  IPRED = A(2)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1,  4.5,  100)   ; 1 CL
  (5.0,  45.0, 500)   ; 2 V
  (0.05, 2.5,  24)    ; 3 MAT
  (0.001,0.4,  10)    ; 4 CV2
$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V
$SIGMA
  0.0225  ; proportional residual variance
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NSIG=3 SIGL=9 NOABORT
$COVARIANCE
