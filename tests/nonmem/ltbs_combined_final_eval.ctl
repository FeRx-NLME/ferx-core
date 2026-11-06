$PROBLEM 1-cpt oral, LTBS combined proportional + additive error, MAXEVAL=0 at the FOCEI fit's printed final estimates (#2729)
$DATA ltbs_combined.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)
  S2 = V
$ERROR
  ; ferx's LTBS floor, spelled out: the prediction is floored before the log and
  ; the same floored value divides ADD.
  IPR = F
  IF (IPR.LT.1E-12) IPR = 1E-12
  IPRED = LOG(IPR)
  W = SQRT(THETA(4)**2 + THETA(5)**2/IPR**2)
  Y = IPRED + W*EPS(1)
$THETA (0, 2.00036)  ; TVCL
$THETA (0, 9.34966)  ; TVV
$THETA (0, 1.53330)  ; TVKA
$THETA (0, 0.104379) ; PROP (log scale)
$THETA (0, 0.0321163) ; ADD (natural DV units)
$OMEGA 0.0648868        ; ETA_CL
$OMEGA 0.0401850        ; ETA_V
$SIGMA 1 FIX
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=0 PRINT=5 NOABORT
