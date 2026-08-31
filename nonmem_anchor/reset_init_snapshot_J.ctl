$PROBLEM ferx #1133 anchor J -- which OCCASION does $PK use at an EVID=3 reset row?
; WT is FLAT at 70, so the only thing that moves across the reset is the OCC column: the
; reset row at t=8 carries OCC=2 while every record before it carries OCC=1. A_0 reads a
; THETA-selected occasion multiplier (not an ETA), so the answer is deterministic and needs
; no EBE: seeding under the reset row's own OCC=2 gives (10*70*3)/50 = 42.0 at t=8, while
; seeding under the preceding record's OCC=1 gives (10*70*1)/50 = 14.0.
$INPUT ID TIME DV MDV EVID AMT CMT WT OCC
$DATA reset_init_snapshot_occ.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  CL   = THETA(1)
  V    = THETA(2)
  K    = CL/V
  OCCF = 1
  IF (OCC.EQ.2) OCCF = THETA(4)
  BASE = THETA(3)*WT*OCCF
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
  (3.0 FIX)   ; occasion-2 multiplier on the baseline

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME WT OCC IPRED NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=reset_init_snapshot_J.tab
