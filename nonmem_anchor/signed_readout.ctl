$PROBLEM ferx #1020 anchor -- signed ODE readout (change from baseline)
; One-compartment IV bolus written as $DES, read out through an arbitrary output
; expression that crosses zero: Y = A(1)/V - BASE. NONMEM applies no
; non-negativity guard to $ERROR output, so IPRED goes negative once the
; concentration falls below BASE. Anchor for ferx's Form C
; `[scaling] y = central / V - BASE`, which the ODE overshoot guard clamped at 0
; before #1020. All parameters FIX, MAXEVAL=0: this is a prediction anchor.
$INPUT ID TIME DV MDV EVID AMT CMT
$DATA signed_readout.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  CL   = THETA(1)
  V    = THETA(2)
  BASE = THETA(3)
  K    = CL/V

$DES
  DADT(1) = -K*A(1)

$ERROR
  IPRED = A(1)/V - BASE
  Y = IPRED + EPS(1)

$THETA
  (4.0 FIX)   ; CL
  (12.0 FIX)  ; V
  (2.0 FIX)   ; BASE

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME IPRED NOPRINT ONEHEADER FILE=sdtab.tab
