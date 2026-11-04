$PROBLEM #1716 density anchor: the Gaussian NULL under IMP, for eta_density_logistic
; eta_shape_null.ctl with eta_density_logistic.ctl's $ESTIMATION line. Two IMP engines
; differ by ~0.06 on this plain Gaussian model already, so the logistic anchor compares
; what the density adds: (logistic - null) in NONMEM against the same in ferx.

; Data, model and parameter vector as eta_shape_null.ctl.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
$OMEGA
  0.2   ; ETA_CL
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR (ferx declares `~ 0.1 (sd)`; NONMEM takes the variance)
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=50 ISAMPLE=3000 SEED=12345 PRINT=1 NOABORT FORMAT=s1PE23.16
