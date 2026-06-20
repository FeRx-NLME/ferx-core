; NONMEM provenance for ferx-core #419: bioavailability F on a rate-defined
; (RATE>0) infusion. NONMEM holds the data RATE and scales the *duration* to
; F*AMT/RATE, so the full F*AMT is delivered at the data rate over a shortened
; window. ferx now reproduces this (it previously scaled the rate instead - the
; bug #419 fixed). This control file is committed for provenance; the matching
; IPRED is the exact duration-scaled ADVAN1 closed form encoded directly in the
; Rust anchors (pk/mod.rs::iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form
; and modeled_rate.rs::modeled_rate_under_f_scales_duration), which run in CI
; without NONMEM.
;
; CL=5, V=50 (k=0.1), AMT=100, RATE=20, F1=0.5
;   -> rate held at 20, duration = F*AMT/RATE = 0.5*100/20 = 2.5 h
;   -> IPRED(t) = 20/(V*k) * (1 - exp(-k*t))                  for t <= 2.5 h
;                 IPRED(2.5) * exp(-k*(t-2.5))                for t  > 2.5 h
;   (plateau 20/(V*k) = 4.0)
$PROBLEM ferx-core #419 - F on a rate-defined (RATE>0) infusion (ADVAN1)
$DATA bioavailability_infusion.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
  F1 = THETA(3)
  S1 = V
$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))
$THETA  5.0 FIX  50.0 FIX  0.5 FIX   ; CL V F1
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
