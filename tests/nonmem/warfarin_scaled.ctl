$PROBLEM Warfarin constant output scaling (S2 = V*10) - ferx-core obs_scale cross-check (issue #367)
; ferx `[scaling] obs_scale = 10` divides the concentration prediction by 10.
; NONMEM reproduces that with S2 = V*10 (TRANS2 gives F = A2/S2 = (A2/V)/10).
; Additive error is used on purpose: a proportional error model is invariant to a
; constant output scale, so it could not detect a scaling bug. The DV column in
; warfarin_scaled.csv is the warfarin DV divided by 10 so the scaled prediction
; and the data live on the same scale.
$DATA warfarin_scaled.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V*10
$ERROR
  IPRED = F
  Y = IPRED + EPS(1)
$THETA (0, 0.13)  ; TVCL
$THETA (0, 7.7)   ; TVV
$THETA (0, 0.8)   ; TVKA
$OMEGA 0.09       ; ETA_CL
$OMEGA 0.04       ; ETA_V
$OMEGA 0.30       ; ETA_KA
$SIGMA 0.01       ; additive variance on the /10 scale
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 NSIG=3 SIGL=9 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME IPRED NOPRINT ONEHEADER FILE=sdtab_scaled
