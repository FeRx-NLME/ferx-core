$PROBLEM #1909: which record's $PK sets F1 for an ADDL-implied dose (F1 = 1/X)
; 1 implied dose at 12: parent X=1, previous record (8) X=2, next record (14) X=4
; 2 implied dose tied with an EVID=2 row at 12 (X=2), next record (13) X=4
; 3 implied dose tied with an observation at 12 (X=2) / 4 constant-X control
$INPUT ID TIME DV MDV EVID AMT RATE CMT ADDL II X
$DATA addl_dose.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  KA = THETA(3)
  F1 = 1/X
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0.5 FIX) (10 FIX) (1 FIX)
$OMEGA 0.01 FIX
$SIGMA 0.01 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME EVID MDV X PRED NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=addl_f.tab
