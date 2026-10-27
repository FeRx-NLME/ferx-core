$PROBLEM #1969 parameter time: F1 step on TIME read at a period-2 dose (reset_f)
; Two periods; period 2 restarts TIME at 0 with an EVID=4 dose. F1 is read only at a dose
; record, so the dose's own $PK governs: the period-2 dose at TIME 0 (as written) takes F1 = 1.
; Oral 1-cpt, observations on the central compartment (CMT 2), none co-timed with a dose.
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA param_time_reset_f.csv IGNORE=@
$SUBROUTINE ADVAN2 TRANS2
$PK
  FB = 1
  IF (TIME.GE.24) FB = THETA(3)
  F1 = FB
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(4)
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1)) + EPS(2)
$THETA
  (0.01, 2.0, 100)   ; TVCL
  (1.0, 20.0, 500)   ; TVV
  (0.05, 0.5, 1)     ; TVFSTEP
  (0.1, 1.2, 10)     ; TVKA
$OMEGA 0.09
$OMEGA 0.04
$SIGMA 0.01
$SIGMA 0.0001 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME EVID FB CL IPRED ETA1 ETA2 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16 FILE=param_time_reset_f.tab
