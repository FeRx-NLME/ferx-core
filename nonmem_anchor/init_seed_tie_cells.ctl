$PROBLEM ferx #2121 init_seed_tie_cells -- A_0 seed at a tied earliest instant: which co-timed record's $PK?
$INPUT ID TIME DV MDV EVID AMT CMT SS II WT
$DATA init_seed_tie_cells.csv IGNORE=@
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
$TABLE ID TIME EVID WT IPRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE17.10 FILE=init_seed_tie_cells.tab
