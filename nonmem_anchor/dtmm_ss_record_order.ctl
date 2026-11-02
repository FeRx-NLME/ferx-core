$PROBLEM #2473 a record listed before a same-TIME SS=1 dose reads the pre-dose history, one after it the SS state
$INPUT ID TIME AMT DV MDV EVID SS II
$DATA dtmm_ss_record_order.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
$ERROR
  IPRED = A(1)/V
  Y = IPRED + EPS(1)
$THETA 1 FIX
$THETA 10 FIX
$OMEGA 0 FIX
$SIGMA 1 FIX
$ESTIMATION MAXEVAL=0
$TABLE ID TIME EVID SS IPRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16 FILE=dtmm_ss_record_order.tab
