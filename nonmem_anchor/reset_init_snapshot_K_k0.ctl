$PROBLEM ferx #1153 anchor K_k0 -- K's data, but occasion 3 (reset row only) gets KAPPA = 0
; The pre-#1153 ferx convention: an occasion only a reset row carries had no kappa group,
; so the reset re-seeded A_0 under KAPPA = 0. Identical to K except that occasion 3 has no
; ETA. The distance between K's and this run's #OBJV is what arm K resolves.
$INPUT ID TIME DV MDV EVID AMT CMT WT OCC
$DATA reset_init_snapshot_occ_only.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  KAPPA = 0
  IF (OCC.EQ.1) KAPPA = ETA(2)
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)
  K    = CL/V
  BASE = THETA(3)*WT*EXP(KAPPA)
  A_0(1) = BASE

$DES
  DADT(1) = -K*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1+EPS(1))

$THETA
  (5.0 FIX)   ; CL
  (50.0 FIX)  ; V
  (10.0 FIX)  ; BASE per kg

$OMEGA 0.09   ; ETA_CL
$OMEGA 0.04   ; KAPPA occasion 1
$SIGMA 0.01

$ESTIMATION MAXEVAL=0 METHOD=1 INTER POSTHOC
$TABLE ID TIME OCC IPRED ETA1 ETA2 NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=reset_init_snapshot_K_k0.tab
