$PROBLEM #1969 parameter time: A_0 re-seed reading TIME at an EVID=4 reset row (reset_seed)
; Two periods; period 2 restarts TIME at 0 with an EVID=4 dose. BASE steps at TIME 24 and
; reaches the prediction only through A_0(1), seeded at each period start: the reset row's own
; $PK, at TIME 0 as written, governs the period-2 seed. THETA(4) (the step) is FIXED: no record
; that seeds reads it, so it is not identifiable.
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA param_time_reset_seed.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  BASE = THETA(3)
  IF (TIME.GE.24) BASE = THETA(3)*THETA(4)
  A_0(1) = BASE
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1)) + EPS(2)
$THETA
  (0.01, 2.0, 100)   ; TVCL
  (1.0, 20.0, 500)   ; TVV
  (0.1, 40.0, 1000)  ; TVBASE
  3.0 FIX            ; BSTEP
$OMEGA 0.09
$OMEGA 0.04
$SIGMA 0.01
$SIGMA 0.0001 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME EVID BASE CL IPRED ETA1 ETA2 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16 FILE=param_time_reset_seed.tab
