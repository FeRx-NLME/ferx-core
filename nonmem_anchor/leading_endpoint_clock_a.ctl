$PROBLEM ferx #3095 leading_endpoint_clock_a -- a record before the first dose starts A_0
$INPUT ID TIME DV MDV EVID AMT CMT LEAD
$DATA leading_endpoint_clock_nm.csv IGNORE=@
$SUBROUTINE ADVAN2 TRANS2
$PK
  KA = THETA(1)
  CL = THETA(2)
  V  = THETA(3)
  S2 = V
  A_0(2) = THETA(4)
$ERROR
  IPRED = F
  Y = IPRED + EPS(1)
$THETA
  (1.0 FIX)
  (2.0 FIX)
  (20.0 FIX)
  (200.0 FIX)
$OMEGA 0 FIX
$SIGMA 0.1 FIX
$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME EVID IPRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE17.10 FILE=leading_endpoint_clock_a.tab
