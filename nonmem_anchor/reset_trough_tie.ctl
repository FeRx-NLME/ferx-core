$PROBLEM ferx #2728 reset_trough_tie -- an observation between a co-timed EVID=3 reset and a dose
; One state, A_0(1) re-seeded at every reset. The seed steps at TIME 30 (BSTEP, FIX), so a reset
; whose $PK read anything but its written TIME would seed the wrong level.
; ID 1: EVID=3, obs, dose at 30 (the #2728 case: the obs reads the reset state).
; ID 2: obs, EVID=3, dose (control: the obs reads the pre-reset state).
; ID 3: EVID=3, dose, obs (post-dose). ID 4: EVID=3, obs, no dose.
; ID 5: EVID=3, obs, obs, dose. ID 6: a second period restarting TIME at 0, then ID 1's tie.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA reset_trough_tie.csv IGNORE=@
$SUBROUTINE ADVAN6 TOL=9
$MODEL
  COMP=(CENTRAL, DEFDOSE, DEFOBS)
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  A0 = THETA(3)
  IF (TIME.GE.30) A0 = THETA(3)*THETA(4)
  IF (A_0FLG.EQ.1) A_0(1) = A0
  S1 = V
$DES
  DADT(1) = -CL/V*A(1)
$ERROR
  IPRED = F
  Y = F + EPS(1)
$THETA
  (1.0 FIX)
  (10.0 FIX)
  (50.0 FIX)
  (2.0 FIX)
$OMEGA 0.04
$SIGMA 0.01
$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC
$TABLE ID TIME EVID PRED IPRED ETA1 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16 FILE=reset_trough_tie.tab
