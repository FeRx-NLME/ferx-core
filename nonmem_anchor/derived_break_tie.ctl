$PROBLEM ferx #2534 derived_break_tie -- a trough before a co-timed dose, with an infusion end or a lag arrival one ULP below it
$INPUT ID TIME DV MDV EVID AMT RATE CMT
$DATA derived_break_tie.csv IGNORE=@
$SUBROUTINE ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT, DEFDOSE)
  COMP=(CENTRAL)
$PK
  K     = THETA(1)
  ALAG1 = THETA(2)
  KA    = 1
$DES
  DADT(1) = -KA*A(1)
  DADT(2) = KA*A(1) - K*A(2)
$ERROR
  IPRED = A(2)
  Y = IPRED*(1+EPS(1))
$THETA
  (0.1 FIX)
  (2.96 FIX)
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME EVID IPRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE17.10 FILE=derived_break_tie.tab
