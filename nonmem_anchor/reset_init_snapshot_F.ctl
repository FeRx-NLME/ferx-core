$PROBLEM ferx #1133 anchor F -- OFV oracle: an ETA inside A_0, re-seeded at the reset
; Arms A-E are MAXEVAL=0 with $OMEGA 0 FIX, so they pin the VALUE path only. The FOCEI
; `h` matrix is the analytic Jacobian, so a wrong d(init)/d(eta) at the reset moves the
; reported objective -- which makes the Dual2 reset seed externally anchorable. This arm
; puts ETA(1) inside A_0 and reads #OBJV under POSTHOC, so it is that oracle.
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA reset_init_snapshot.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  CL   = THETA(1)
  V    = THETA(2)
  K    = CL/V
  BASE = THETA(3)*WT*EXP(ETA(1))
  A_0(1) = BASE

$DES
  DADT(1) = -K*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1+EPS(1))

$THETA
  (5.0 FIX)   ; CL  -- still NOT a function of WT, so WT reaches the prediction
  (50.0 FIX)  ; V   -- only through the A_0 seed
  (10.0 FIX)  ; BASE per kg: A_0(1) = 10*WT*exp(ETA(1))

$OMEGA 0.09    ; ETA(1) on the baseline -- this is what makes #OBJV a gradient oracle
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER POSTHOC
$TABLE ID TIME WT IPRED NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=reset_init_snapshot_F.tab
