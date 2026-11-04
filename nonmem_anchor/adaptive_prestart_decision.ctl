$PROBLEM ferx #2125 adaptive regimens whose first decision precedes the first record
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA adaptive_prestart_decision.csv IGNORE=@
$SUBROUTINE ADVAN13 TOL=9
$MODEL
  COMP=(CENTRAL, DEFDOSE)
$PK
  CL   = THETA(1)
  V    = THETA(2)
  K    = CL/V
  BASE = THETA(3)*WT
  A_0(1) = BASE
$DES
  DADT(1) = -K*A(1)
$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1+EPS(1))
$THETA
  (5.0 FIX)
  (50.0 FIX)
  (10.0 FIX)
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME EVID WT IPRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE17.10 FILE=adaptive_prestart_decision.tab
