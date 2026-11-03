$PROBLEM ferx-core #1565 MM ODE basin-shift reference (eval_fix)
; #1565: NONMEM 7.6.0 FOCEI evaluated (MAXEVAL=0, MCETA=200) at the estimates ferx reaches
; after the basin-shift restart (ferx OFV 775.394). Result: OBJV 775.414 (see .ext).
; Without MCETA, NONMEM's cold eta search puts subject 54 in the higher mode: OBJV 789.192.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA ../fixtures/mm_basin_shift/mm_n200.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=10
$MODEL COMP=(DEPOT DEFDOSE) COMP=(CENTRAL)
$PK
  VMAX = THETA(1)*EXP(ETA(1))
  KM   = THETA(2)
  V    = THETA(3)*EXP(ETA(2))
  KA   = THETA(4)*EXP(ETA(3))
$DES
  DADT(1) = -KA*A(1)
  DADT(2) = KA*A(1)/V - VMAX*A(2)/(KM+A(2))
$ERROR
  IPRED = A(2)
  Y = IPRED*(1+EPS(1))
$THETA (0.1,1.194396,50) (0.1,3.895956,100) (1,10.503601,200) (0.05,0.956276,20)
$OMEGA 0.091849 0.105154 0.198653
$SIGMA 0.010605
$ESTIMATION METHOD=1 INTER MAXEVAL=0 NOABORT PRINT=5 SIGDIGITS=4 MCETA=200
