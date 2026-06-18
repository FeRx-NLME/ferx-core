$PROBLEM zero-order input into the oral depot (ADVAN2 TRANS2, D1 modeled duration) - ferx #400 anchor
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA oral_depot_d1.csv IGNORE=@
$SUBROUTINE ADVAN2 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
  KA = THETA(3)
  D1 = THETA(4)   ; modeled zero-order infusion duration into the depot (cmt 1)
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED + IPRED*EPS(1)
$THETA
  5  FIX   ; CL
  50 FIX   ; V
  1  FIX   ; KA
  5  FIX   ; D1  -> rate = AMT/D1 = 20 over 5 h
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=1 INTER POSTHOC
$TABLE ID TIME CMT PRED IPRED NOPRINT ONEHEADER NOAPPEND FILE=sdtab1
