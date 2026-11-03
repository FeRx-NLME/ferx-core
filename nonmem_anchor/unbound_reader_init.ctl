$PROBLEM ferx #1898 anchor A -- A_0(2) seeded from S, an ordinary $PK variable no PK role uses
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA unbound_reader.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  CL = THETA(1)
  V  = THETA(2)
  KA = THETA(3)
  S  = THETA(4)*WT     ; 30 (WT 60) / 40 (WT 80): read only by the initial condition
  A_0(2) = S
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED*(1+EPS(1))

$THETA
  (5.0 FIX)   ; CL
  (50.0 FIX)  ; V
  (1.5 FIX)   ; KA
  (0.5 FIX)   ; S per kg

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME WT IPRED NOPRINT ONEHEADER FILE=unbound_reader_init.tab
