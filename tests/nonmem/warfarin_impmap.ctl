$PROBLEM Warfarin 1-cpt oral - ferx-core IMPMAP cross-check (issue #270)
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
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.2)   ; TVCL
$THETA (0, 10.0)  ; TVV
$THETA (0, 1.5)   ; TVKA
$OMEGA 0.09       ; ETA_CL
$OMEGA 0.04       ; ETA_V
$OMEGA 0.30       ; ETA_KA
$SIGMA 0.0004     ; proportional variance (= 0.02 SD, matching the .ferx init)
; IMPMAP (Importance Sampling assisted by Mode A Posteriori).
$ESTIMATION METHOD=IMPMAP INTERACTION NITER=200 ISAMPLE=300 PRINT=10 SEED=12345 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME NOPRINT ONEHEADER FILE=sdtab_impmap
