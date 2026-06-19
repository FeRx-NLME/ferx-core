; NONMEM anchor for modeled infusion RATE (RATE=-1 -> R1); ferx-core #324.
;
; ADVAN1 TRANS2 one-compartment IV with the infusion rate modeled as the $PK
; parameter R1 (NONMEM reads RATE=-1 from the data and uses R1 as the rate; the
; duration is then AMT/R1). Fixed thetas, eta=0, MAXEVAL=0 -> NONMEM PRED is the
; exact ADVAN1 solution. With CL=5, V=50 (k=0.1), AMT=100, R1=20: the infusion
; rate is 20 over a 5 h window — the IDENTICAL physical infusion as the RATE=-2
; D1=5 anchor (tests/nonmem/modeled_duration.ctl), so NONMEM's IPRED values are
; byte-identical and match `one_cpt_infusion_closed_form` in
; tests/modeled_rate.rs. (At F=1 ferx's rate-scaling and NONMEM's duration-
; scaling coincide; F!=1 shape divergence for rate-defined infusions is a tracked
; follow-up.)
;
; Reproduce: nmfe75 modeled_rate.ctl modeled_rate.lst  (data file is
; ../../data/modeled_rate_ref.csv; copy it next to this .ctl to run).
$PROBLEM Modeled infusion rate (RATE=-1 -> R1)
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA modeled_rate_ref.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
  R1 = THETA(3)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED * (1 + EPS(1))
$THETA 5 FIX 50 FIX 20 FIX
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
