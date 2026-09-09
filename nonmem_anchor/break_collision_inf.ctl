$PROBLEM #1186 break-time collision twin: depot <- first_order(ka, lag=LAGR) with ALAG1=LAGC; central bolus at 0 and a 1 h INFUSION (RATE=100) at 8.2 -- the infusion-victim arm
; Exact twin of the ferx unit fixture (zz_measure_1186::spec): compartment 1 is an inert
; dose carrier (F1=0) whose record captures PODO/TDOS; the lagged first-order input
; RIN = PODO*KA*exp(-KA*(T-ONSET)) feeds the DEPOT (A(2)); DEPOT -> CENTRAL at KA;
; CENTRAL clears at CL/V. Doses at t=0 (cmt 3) and t=8.2 (cmt 3) are plain boluses.
; ONSET = TDOS + LAGC + LAGR = (0 + 0.3) + 7.9 = 8.200000000000001 in double precision,
; i.e. 1.78e-15 past the t=8.2 bolus -- the coincidence ferx applies that bolus twice on.
; NONMEM is event-typed: each dose record is applied exactly once regardless, so its
; PRED at 8.2001 is the "applied once" reference (closed form 144.04, twice = 244.04).
$INPUT ID TIME DV AMT RATE EVID CMT MDV
$DATA break_collision_inf.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(CARRIER,DEFDOSE)
  COMP=(DEPOT)
  COMP=(CENTRAL,DEFOBS)
$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)
  KA   = THETA(3)
  LAGC = THETA(4)
  LAGR = THETA(5)
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ONSET = TDOS + LAGC + LAGR
  F1   = 0.0
  K    = CL/V
$DES
  TR  = T - ONSET
  RIN = 0.0
  IF (TR.GT.0.0) RIN = PODO*KA*EXP(-KA*TR)
  DADT(1) = 0.0
  DADT(2) = RIN - KA*A(2)
  DADT(3) = KA*A(2) - K*A(3)
$ERROR
  IPRED = A(3)
  Y = IPRED + EPS(1)
$THETA
  1.0 FIX   ; CL
  10.0 FIX  ; V
  1.0 FIX   ; KA
  0.3 FIX   ; LAGC (ALAG1)
  7.9 FIX   ; LAGR (route lag)
$OMEGA 0 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME PRED IPRED CMT EVID NOPRINT ONEHEADER FORMAT=s1PE15.8 FILE=break_collision_inf.tab
