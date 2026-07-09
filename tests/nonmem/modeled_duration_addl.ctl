; NONMEM anchor for ADDL + modeled infusion DURATION (RATE=-2 -> D1); ferx-core #722.
;
; ADVAN1 TRANS2 one-compartment IV. The dose is a modeled-duration infusion
; (RATE=-2 -> D1 = THETA(3)) REPEATED via ADDL=2, II=24 -> three infusions at
; t=0, 24, 48, each rate AMT/D1 = 100/5 = 20 over a 5 h window. This is the exact
; regimen ferx #722 fixes: before the fix ferx expanded the ADDL doses as boluses
; (only the first stayed a modeled infusion); NONMEM keeps ALL of them modeled
; infusions. Observations mid-infusion (t=26, 50) discriminate infusion vs bolus.
; CL=5, V=50 (k=0.1) FIX, eta/eps=0 -> the simulated Y is the exact ADVAN1 profile.
;
; Reproduce: nmfe76 modeled_duration_addl.ctl modeled_duration_addl.lst  (data file
; is ../../data/modeled_duration_addl_ref.csv; copy it next to this .ctl to run).
; NONMEM IPRED values are embedded in tests/modeled_duration.rs
; (`modeled_duration_addl_matches_nonmem`).
$PROBLEM Modeled duration + ADDL (RATE=-2 -> D1, ADDL=2 II=24)
$INPUT ID TIME DV EVID AMT CMT RATE MDV ADDL II
$DATA modeled_duration_addl_ref.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  D1 = THETA(3)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA 5 FIX 50 FIX 5 FIX
$OMEGA 0 FIX
$SIGMA 0 FIX
$SIMULATION (20260707) ONLYSIMULATION SUBPROBLEMS=1
$TABLE ID TIME IPRED Y NOPRINT NOAPPEND ONEHEADER FILE=md_addl.tab
