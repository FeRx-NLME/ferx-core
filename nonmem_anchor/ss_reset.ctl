$PROBLEM ferx #1576 anchor: an SS=1 record resets the system at its record (ADVAN2)
; IDs 1-6: an SS=1 regimen with a later SS=1 (2: at 120; 3: at 400, < 50 II), a later non-SS
; dose (4, control); a bolus before a mid-timeline SS=1 (5); SS=1 alone (6).
; ID 41: SS=1 on every dose record, q12h (the TDM idiom).
; IDs 51-53: a bolus row before (51) / after (52) a co-timed SS=1 row; SS=1 alone (53).
; Pure evaluation at fixed thetas (MAXEVAL=0); PRED is the reference for
; tests/ss_reset_nonmem_anchor.rs.
$DATA ss_reset.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT MDV II SS RATE
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  KA = THETA(3)
  S2 = V
$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))
$THETA 2.0 FIX  20.0 FIX  0.15 FIX
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 NOABORT
$TABLE ID TIME EVID PRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16 FILE=sdtab
