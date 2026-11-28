$PROBLEM #1716 ETA shape anchor: Box-Cox on CL, lambda = -0.5
; As eta_shape_bc_pos.ctl with the other sign of lambda: the transform's
; skew flips, so a sign error in the kernel cannot pass both arms.
; Differs from eta_shape_null.ctl by the transform alone.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  ETATR = (EXP(ETA(1))**THETA(3) - 1)/THETA(3)
  CL = THETA(1)*EXP(ETATR)
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
  -0.5 FIX             ; LAMBDA
$OMEGA
  0.2   ; ETA_CL
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER FILE=eta_shape_bc_neg.tab FORMAT=s1PE23.16
