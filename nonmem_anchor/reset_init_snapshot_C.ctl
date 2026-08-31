$PROBLEM ferx #1133 anchor C -- A_0 re-seed at an EVID=3/4 reset: whose $PK snapshot?
; WT is 140 on the reset row and 200 on the NEXT record -- the discriminator.
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA reset_init_snapshot_split.csv IGNORE=@

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
  (5.0 FIX)   ; CL  -- deliberately NOT a function of WT, so WT reaches the
  (50.0 FIX)  ; V   -- prediction ONLY through the A_0 seed
  (10.0 FIX)  ; BASE per kg: A_0(1) = 10*WT (WT=70 -> 700, 140 -> 1400, 200 -> 2000)

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME WT IPRED NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=reset_init_snapshot_C.tab
