; Joint PK-TTE event-time SIMULATION anchor — ferx Slice 2.2 (#564).
;
; NONMEM simulates each subject's cumulative-hazard trajectory CHZ(t)=A(3) (oral 1-cpt
; PK + drug-driven hazard h=H0*exp(BETA*Cc), Cc=A(2)/V) with BSV on CL drawn by $SIM.
; CHZ(t) is tabled on a dense grid; compare.R inverts CHZ(T) = -log(U), one U per
; subject, for the event time (the inverse-CDF step is mechanical and identical across
; tools — ferx does it internally via the Slice 2.2 root-finder; its correctness is
; pinned separately by the PIT/KS goodness-of-fit unit test).
;
; Truth (matches simulate.R / the Slice 2.1 anchor): CL=1 V=10 KA=1 H0=0.015 BETA=0.25,
; omega(CL)=0.09, single dose=100, horizon=24.
; Run: nmfe76 sim.ctl sim.lst  ->  sim.tab

$PROBLEM  Joint PK-TTE event-time simulation (ferx anchor #564 Slice 2.2)
$INPUT    ID TIME DV EVID AMT CMT MDV
$DATA     simtemplate.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)
  COMP=(CENTRAL)
  COMP=(CUMHAZ)
$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)
  KA   = THETA(3)
  H0   = THETA(4)
  BETA = THETA(5)
  KE   = CL/V
$DES
  CONC    = A(2)/V
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - KE*A(2)
  DADT(3) =  H0*EXP(BETA*CONC)        ; cumulative hazard (= ferx __chz)
$ERROR
  CHZ = A(3)
  Y   = CHZ + EPS(1)
$THETA
  1.0   FIX   ; 1 CL
  10.0  FIX   ; 2 V
  1.0   FIX   ; 3 KA
  0.015 FIX   ; 4 H0
  0.25  FIX   ; 5 BETA
$OMEGA
  0.09        ; ETA(1) on CL — simulated by $SIM
$SIGMA
  0 FIX       ; dummy zero-variance EPS (NM-TRAN requires one for population data)
$SIMULATION (20260629) ONLYSIMULATION
$TABLE ID TIME CMT CHZ NOPRINT ONEHEADER NOAPPEND FILE=sim.tab
