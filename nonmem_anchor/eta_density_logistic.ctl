$PROBLEM #1716 density anchor: `density logistic` as F^-1(PHI(ETA)) under IMP
; ferx: `omega ETA_CL ~ 0.2 density logistic`, CL = TVCL * EXP(ETA_CL): the effect is
; logistic with mean 0 and variance 0.2, scale B = SQRT(3*0.2)/PI. The same marginal
; model in NONMEM: ETA(1) ~ N(0, 1) FIX, X = B*LOG(P/(1-P)), P = PHI(ETA(1)), i.e. the
; logistic quantile of a standard normal's CDF. Not an affine map, so the Laplace
; objectives differ; IMP estimates the exact marginal -2LL on both sides, which must
; agree within Monte Carlo error (ferx: `method = imp`, `imp_eval_only = true`).
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  B  = THETA(3)
  P  = PHI(ETA(1))
  X  = B*LOG(P/(1 - P))
  CL = THETA(1)*EXP(X)
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
  0.24656177762459994 FIX   ; B = SQRT(3*0.2)/PI
$OMEGA
  1 FIX ; ETA_CL, standard normal
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=50 ISAMPLE=3000 SEED=12345 PRINT=1 NOABORT FORMAT=s1PE23.16
