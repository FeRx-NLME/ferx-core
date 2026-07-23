$PROBLEM oral_central_inf_advan2_f1
; #376: 1-cpt ORAL model (ADVAN2), infusion (RATE>0) into CMT=2 (central,
; depot bypass), F=1. Direct NONMEM anchor for the event-driven central-
; infusion path (PR #350), previously anchored only transitively via the
; superposition reference. AMT=100 RATE=25 -> 4 h infusion into central.
$INPUT ID TIME DV AMT CMT EVID RATE SS II
$DATA oral_central_inf_advan2.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = 5
  V  = 50
  KA = 1
  S2 = V
$ERROR
  Y = F * (1 + EPS(1))
$THETA (5.0 FIX)   ; CL (unused; NONMEM needs >=1 THETA)
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME CMT PRED NOPRINT ONEHEADER FILE=oral_central_inf_advan2_f1.tab FORMAT=,1PE17.10
