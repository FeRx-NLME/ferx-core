$PROBLEM #1909: an ADDL-implied dose tied with a pk-only row (ADVAN1, CL = TVCL*X)
; 1 EVID=2 at an implied dose (12) / 2 EVID=0,MDV=1 there / 3 an observation there /
; 4 no record at 12, X changes at the next observation / 5 constant X, observation at 12 /
; 6 explicit-dose twin of 1, EVID=2 listed before the dose / 7 the same, listed after it /
; 8 case 1 with an ADDL infusion (RATE=50) / 9 EVID=2 at the third dose (24), no record at 12
$INPUT ID TIME DV MDV EVID AMT RATE CMT ADDL II X
$DATA addl_tie.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)*X*EXP(ETA(1))
  V  = THETA(2)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0.5 FIX) (10 FIX)
$OMEGA 0.01 FIX
$SIGMA 0.01 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME EVID MDV X PRED NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=addl_tie.tab
