$PROBLEM MM SS-absorption anchor for ferx-core #867 (nonlinear periodic steady state)
; SS=1 dose into a first-order (KA) depot feeding a central compartment with
; Michaelis-Menten elimination (VM, KM). Saturable, mean input 100/12=8.33 < VM=15,
; so a periodic SS exists. NONMEM computes it with its own general-ODE SS solver;
; PRED is the reference for ferx's Anderson-accelerated nonlinear SS (#867).
$DATA ss_mm.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT MDV II SS
$SUBROUTINES ADVAN13 TOL=9
$MODEL COMP=(DEPOT DEFDOSE) COMP=(CENTRAL DEFOBS)
$PK
  KA = THETA(1)
  VM = THETA(2)
  KM = THETA(3)
  S2 = 1
$DES
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - VM*A(2)/(KM + A(2))
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA 1.0 FIX 15.0 FIX 300.0 FIX
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FORMAT=,1PE20.13 FILE=sdtab1
