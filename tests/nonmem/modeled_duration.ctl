; NONMEM anchor for modeled infusion DURATION (RATE=-2 -> D1); ferx-core #324.
;
; ADVAN1 TRANS2 one-compartment IV with the infusion duration modeled as the
; $PK parameter D1 (NONMEM reads RATE=-2 from the data and uses D1 as the
; duration; the rate is then AMT/D1). Fixed thetas, eta=0, MAXEVAL=0 -> NONMEM
; PRED is the exact ADVAN1 solution. With CL=5, V=50 (k=0.1), AMT=100, D1=5:
; the infusion rate is 20 over a 5 h window, matching
; `one_cpt_infusion_closed_form` in tests/modeled_duration.rs.
;
; Reproduce: nmfe75 modeled_duration.ctl modeled_duration.lst  (data file is
; ../../data/modeled_duration_ref.csv; copy it next to this .ctl to run).
$PROBLEM Modeled infusion duration (RATE=-2 -> D1)
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA modeled_duration_ref.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
  D1 = THETA(3)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED * (1 + EPS(1))
$THETA 5 FIX 50 FIX 5 FIX
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
