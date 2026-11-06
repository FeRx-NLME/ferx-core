$PROBLEM 1-cpt oral, LTBS combined proportional + additive error, MAXEVAL=0, FOCE (#2729)
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
$THETA (0, 2.2)    ; TVCL
$THETA (0, 9.0)    ; TVV
$THETA (0, 1.3)    ; TVKA
$THETA (0, 0.12)   ; PROP (log scale)
$THETA (0, 0.07)   ; ADD (natural DV units)
$OMEGA 0.09        ; ETA_CL
$OMEGA 0.04        ; ETA_V
$SIGMA 1 FIX
$ESTIMATION METHOD=COND MAXEVAL=0 PRINT=5 NOABORT
