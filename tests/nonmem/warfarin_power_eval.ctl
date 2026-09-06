$PROBLEM Warfarin 1-cpt oral - power residual error, evaluation at the ferx inits (#1182)
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED + IPRED**THETA(4)*EPS(1)
$THETA (0, 0.13)   ; TVCL
$THETA (0, 8.0)    ; TVV
$THETA (0, 1.0)    ; TVKA
$THETA (0.01, 1.3, 10) ; RUV_POW
$OMEGA 0.09        ; ETA_CL
$OMEGA 0.04        ; ETA_V
$OMEGA 0.30        ; ETA_KA
$SIGMA 0.01        ; sigma^2 (SD 0.1)
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=0 PRINT=5 NOABORT
