$PROBLEM ferx #1153 anchor K -- an occasion ONLY the EVID=3 reset row carries gets its own ETA
; The reset row at t=8 carries OCC=3; every observation and the dose carry OCC=1. The IOV
; ETA enters A_0 only, so ETA(3) (occasion 3) is read by the reset re-seed alone and is
; informed by the three post-reset observations; NONMEM estimates it like any other
; occasion ETA. K_k0 is the same data with occasion 3 given no ETA (KAPPA = 0 there),
; which is what ferx did before #1153.
$INPUT ID TIME DV MDV EVID AMT CMT WT OCC
$DATA reset_init_snapshot_occ_only.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  KAPPA = 0
  IF (OCC.EQ.1) KAPPA = ETA(2)
  IF (OCC.EQ.3) KAPPA = ETA(3)
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
$OMEGA 0.04   ; KAPPA occasion 3 (only the reset row carries it)
$SIGMA 0.01

$ESTIMATION MAXEVAL=0 METHOD=1 INTER POSTHOC
$TABLE ID TIME OCC IPRED ETA1 ETA2 ETA3 NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=reset_init_snapshot_K.tab
