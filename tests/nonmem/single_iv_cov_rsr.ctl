$PROBLEM single_iv_cov_rsr: one-population twin of mixture_iv_cov (#1956), R^-1 S R^-1
; CL = TVCL*EXP(ETA), V = TVV on mixture_iv.csv, at the one-population MLE found by
; tests/nonmem/mixture_iv_cov_score_oracle.py --single (6 sig. fig.), MAXEVAL=0.
$INPUT ID TIME DV AMT EVID CMT WT
$DATA mixture_iv.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  CL = THETA(1) * EXP(ETA(1))
  V  = THETA(2)
  S1 = V

$ERROR
  IPRED = F
  Y = IPRED * (1.0 + EPS(1))

$THETA
  (0.01, 1.76783, 100.0)    ; TVCL
  (0.1, 9.77524, 1000.0)    ; V
$OMEGA 0.09 FIX
$SIGMA 0.04 FIX

$ESTIMATION METHOD=1 INTER MAXEVAL=0 NOABORT
$COVARIANCE PRINT=E
