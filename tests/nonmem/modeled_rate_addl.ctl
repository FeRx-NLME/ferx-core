; NONMEM anchor for ADDL + modeled infusion RATE (RATE=-1 -> R1); ferx-core #722.
;
; ADVAN1 TRANS2 one-compartment IV. The dose is a modeled-RATE infusion
; (RATE=-1 -> R1 = THETA(3)) REPEATED via ADDL=2, II=24 -> three infusions at
; t=0, 24, 48, each rate R1 = 20 over AMT/R1 = 100/20 = 5 h. This is the RATE=-1
; twin of modeled_duration_addl.ctl (RATE=-2 -> D1=5): R1=20 delivers the exact
; same infusion (20 over 5 h) specified as a rate instead of a duration, so the
; profile matches -- this anchor pins the RATE=-1 ADDL code path. Before ferx #722
; the ADDL doses expanded as boluses (only the first stayed a modeled infusion);
; NONMEM keeps ALL of them modeled infusions. Observations mid-infusion (t=26, 50)
; discriminate infusion vs bolus.
; CL=5, V=50 (k=0.1) FIX, eta/eps=0 -> the simulated Y is the exact ADVAN1 profile.
;
; Reproduce: nmfe76 modeled_rate_addl.ctl modeled_rate_addl.lst  (data file is
; ../../data/modeled_rate_addl_ref.csv; copy it next to this .ctl to run). NONMEM
; IPRED values are embedded in tests/modeled_duration.rs
; (`modeled_rate_addl_matches_nonmem`).
$PROBLEM Modeled rate + ADDL (RATE=-1 -> R1, ADDL=2 II=24)
$INPUT ID TIME DV EVID AMT CMT RATE MDV ADDL II
$DATA modeled_rate_addl_ref.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  R1 = THETA(3)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA 5 FIX 50 FIX 20 FIX
$OMEGA 0 FIX
$SIGMA 0 FIX
$SIMULATION (20260709) ONLYSIMULATION SUBPROBLEMS=1
$TABLE ID TIME IPRED Y NOPRINT NOAPPEND ONEHEADER FILE=mr_addl.tab
