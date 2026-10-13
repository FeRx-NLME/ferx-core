$PROBLEM tte_exponential Laplace + $COV (#1781)
; #1781: external reference for the covariance step on a TTE model. The SE rows of the
; .ext files (-1000000001) are what docs/estimation/saem.qmd and
; tests/sampler_covariance_non_gaussian.rs quote. Data: data/tte_exponential.csv.
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA tte_exponential_cov.csv IGNORE=@
$PRED
  LAMBDA = THETA(1)*EXP(ETA(1))
  CHZ = LAMBDA*TIME
  SUR = EXP(-CHZ)
  IF (DV.EQ.1) Y = LAMBDA*SUR
  IF (DV.EQ.0) Y = SUR
$THETA (0.001, 0.05, 10)
$OMEGA 0.09
$EST METHOD=COND LAPLACE LIKELIHOOD MAXEVAL=9999 PRINT=5 SIGDIG=4
$COV MATRIX=R UNCONDITIONAL PRINT=E
