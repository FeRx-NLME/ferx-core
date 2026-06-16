$PROBLEM Per-compartment F1/F2 + ALAG2, mixed oral(cmt1)+IV(cmt2) dose
; Cross-check anchor for ferx-core compartment-indexed bioavailability/lag (Fn/ALAGn, issue #369).
; A subject is dosed BOTH into the depot (CMT=1, oral, F1) and into the
; central compartment (CMT=2, IV bolus, F2 with absorption lag ALAG2).
; One-compartment disposition (CL,V) with first-order absorption (KA).
; Pure evaluation at fixed thetas, eta=0, so PRED is the analytical solution.
;
; CL=5  V=50  KA=1.5  F1=0.70  F2=0.40  ALAG2=2.0
; Run with NONMEM 7.5.1 (nmfe75); PRED in sdtab1 reproduces the closed-form
; reference hardcoded in tests/two_cmt_dose_nonmem.rs to all printed digits.
$DATA ../../data/two_cmt_dose_ref.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL    = THETA(1)
  V     = THETA(2)
  KA    = THETA(3)
  F1    = THETA(4)
  F2    = THETA(5)
  ALAG2 = THETA(6)
  S2    = V
$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))
$THETA  5.0 FIX  50.0 FIX  1.5 FIX  0.70 FIX  0.40 FIX  2.0 FIX  ; CL V KA F1 F2 ALAG2
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
