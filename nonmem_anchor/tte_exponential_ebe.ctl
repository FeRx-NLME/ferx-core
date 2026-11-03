$PROBLEM tte_exponential POSTHOC at the Laplace optimum -- EBE anchor for #1885
; #1885: the analytic half of the individual-survival anchor. Same model and data as
; tte_exponential_cov_focei.ctl, evaluated (MAXEVAL=0 POSTHOC) at that run's final
; estimates, read from results/tte_exponential_cov_focei.ext (-1000000000 row):
; THETA1 = 3.32303E-02, OMEGA(1,1) = 2.84734E-01. ETA1 is then each subject's conditional
; mode and CHZ = LAMBDA*exp(ETA1)*TIME the individual cumulative hazard at its record.
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA tte_exponential_cov.csv IGNORE=@
$PRED
  LAMBDA = THETA(1)*EXP(ETA(1))
  CHZ = LAMBDA*TIME
  SUR = EXP(-CHZ)
  IF (DV.EQ.1) Y = LAMBDA*SUR
  IF (DV.EQ.0) Y = SUR
$THETA 3.32303E-02 FIX
$OMEGA 2.84734E-01 FIX
$EST METHOD=COND LAPLACE LIKELIHOOD MAXEVAL=0 POSTHOC PRINT=1
$TABLE ID TIME DV CHZ ETA1 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE15.8
       FILE=tte_exponential_ebe.tab
