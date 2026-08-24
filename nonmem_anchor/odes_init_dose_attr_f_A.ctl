$PROBLEM ferx #1046 anchor F-A -- A_0 seeded from F1: does the seed get bioavailability applied?
$INPUT ID TIME DV MDV EVID AMT CMT
$DATA odes_init_dose_attr.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  CL = THETA(1)
  V  = THETA(2)
  K  = CL/V
  F1 = THETA(3)
  A_0(1) = F1*100

$DES
  DADT(1) = -K*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1+EPS(1))

$THETA
  (5.0 FIX)   ; CL
  (50.0 FIX)  ; V
  (0.5 FIX)   ; F1 -- scales the t=1 dose, and is read by A_0(1)

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME IPRED NOPRINT ONEHEADER FILE=sdtab.tab
