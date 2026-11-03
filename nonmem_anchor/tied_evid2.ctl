$PROBLEM #1810: tied obs / EVID=2 / dose -- whose X governs the interval ending at the tie? (ODE)
; ferx twin: tied_evid2_fit.ferx (same CSV). d/dt(A) = TVR*X, readout A + SLOPE*X.
; Cases (one subject each), tie at TIME 4, bolus 100 at 0:
;  1 obs, EVID=2   2 EVID=2, obs   3 obs, EVID=2 at 4.000001   4 obs1, EVID=2, obs2
;  5 dose, EVID=2  6 EVID=2, dose  7 obs1, obs2 (control)
; Expected PRED (#1809 triage, NONMEM 7.6.0): 114,132 / 118,136 / 114,132 / 114,134 /
;  204 / 200 / 114,134
; The production term is PROD, not RATE: RATE is a reserved NONMEM data-item label.
; ETA(1) on PROD with OMEGA 0.01 matches the ferx twin's ETA_R; PRED (ETA = 0) is unchanged.
$INPUT ID TIME DV MDV EVID AMT CMT X
$DATA tied_evid2.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=9
$MODEL COMP=(A DEFDOSE DEFOBS)
$PK
  PROD = THETA(1)*X*EXP(ETA(1))
  RO   = THETA(2)*X
$DES
  DADT(1) = PROD
$ERROR
  IPRED = A(1) + RO
  Y = IPRED + EPS(1)
$THETA (1 FIX) (10 FIX)
$OMEGA 0.01 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME EVID X PRED IPRED NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=tied_evid2.tab
