$PROBLEM ferx #1046 anchor F-B -- control: A_0 seeded from an ORDINARY parameter of the same value
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
  FSEED = THETA(4)
  A_0(1) = FSEED*100

$DES
  DADT(1) = -K*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1+EPS(1))

$THETA
  (5.0 FIX)   ; CL
  (50.0 FIX)  ; V
  (0.5 FIX)   ; F1    -- scales the t=1 dose, NOT read by A_0
  (0.5 FIX)   ; FSEED -- ordinary parameter, same value, read by A_0(1)

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME IPRED NOPRINT ONEHEADER FILE=sdtab.tab
