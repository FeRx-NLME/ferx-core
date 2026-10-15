$PROBLEM Two co-timed observations the reader splits one ULP apart, constant covariates
; ferx twin: tied_obs_split_fit.ferx, an [odes] model (the engines the split reached).
; Cases (one subject each), tie at TIME 4, bolus 100 at 0, ALAG1 0.5 on every dose:
;  1 obs, EVID=2, obs   2 obs, obs, EVID=2   3 obs, obs (control)
;  4 obs, dose 50, obs  5 obs, obs, dose 50 (control)
; Expected PRED: 7.0469 at TIME 4 in every case; at TIME 8, 4.7237 (1-3) and 8.2471 (4-5).
; X is constant, so every subject reaches ferx's dense ODE engine.
; The two tied observations carry different DVs, so a record the engine fails to read moves
; the OFV and the POSTHOC ETA, not only the sdtab row. Subjects 1-3 (and 4-5) share their DVs,
; so their per-subject OFV terms and IPREDs must agree within each group.
$INPUT ID TIME DV MDV EVID AMT CMT X
$DATA tied_obs_split.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)*X*EXP(ETA(1))
  V  = THETA(2)
  ALAG1 = THETA(3)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED + EPS(1)
$THETA (1 FIX) (10 FIX) (0.5 FIX)
$OMEGA 0.01 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=1 MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME EVID X DV PRED IPRED ETA1 NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=tied_obs_split.tab
