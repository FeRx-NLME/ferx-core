$PROBLEM Joint PK-TTE, RATE>0 into the absorption compartment -- external anchor for #1187
; ferx's joint PK-TTE cumulative hazard has two production routes: the #570 shared single
; solve (`ode_predictions_and_chz` -> `active_infusions`) and the two-solve fallback
; (`ode_cumhaz_hazard` -> `ode_dense_solve_states` -> `gated_infusions`). Only the second
; resolved infusions through the unguarded helper, so a RATE>0 dose into a built-in
; absorption compartment was delivered TWICE on that arm -- once by the convolved
; `R_in_inf` kernel and again as a plain `+rate`.
;
; NONMEM integrates the augmented system ONCE and has no such split, so it is the outside
; reference. Identical to `../pktte-engine-split/pktte_eta0.ctl` except the dose carries
; RATE=50 (AMT=100 -> T=2h zero-order fill of the depot). ADVAN2's zero-order-into-depot
; behaviour is what ferx's `first_order()` + `R_in_inf` convolution reproduces, already
; anchored by `tests/infusion_absorption_nonmem_anchor.rs`.
;
; Compared object is the CUMULATIVE HAZARD A(3), not the OFV: a likelihood model (F_FLAG=1)
; carries different additive constants in NONMEM than in ferx. A(3) is constant-free.
;
; MAXEVAL=0 POSTHOC, every THETA FIX, OMEGA 0 FIX so both tools evaluate at eta = 0.

$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA pktte_inf.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)
  COMP=(CENTRAL)
  COMP=(CHZ)

$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)
  KA   = THETA(3)
  H0   = THETA(4)
  BETA = THETA(5)
  K20  = CL/V

$DES
  CONCD   = A(2)/V
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - K20*A(2)
  DADT(3) =  H0*EXP(BETA*CONCD)

$ERROR
  CONC = A(2)/V
  HAZ  = H0*EXP(BETA*CONC)
  CHZ  = A(3)
  SUR  = EXP(-CHZ)
  IPRED = CONC
  F_FLAG = 0
  IF (CMT.EQ.3) F_FLAG = 1
  ; Single-level IFs only: NM-TRAN error 326 rejects a random variable assigned inside a
  ; NESTED if-structure, so the CMT/DV test is folded into one condition per branch.
  Y = IPRED*(1.0 + EPS(1))
  IF (CMT.EQ.3.AND.DV.EQ.1) Y = SUR*HAZ
  IF (CMT.EQ.3.AND.DV.EQ.0) Y = SUR

$THETA
  1.00 FIX   ; 1 CL   (L/h)
  10.0 FIX   ; 2 V    (L)
  1.00 FIX   ; 3 KA   (1/h)
  0.02 FIX   ; 4 H0   baseline hazard
  0.50 FIX   ; 5 BETA concentration effect on log-hazard

$OMEGA
  0 FIX      ; IIV CL -- zeroed so ferx and NONMEM are compared at the SAME point (eta=0)

$SIGMA
  0.01       ; proportional residual variance (0.10^2)

$ESTIMATION METHOD=1 LAPLACE INTER NUMERICAL SLOW MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME CMT DV IPRED CHZ HAZ ETA1 MDV NOPRINT ONEHEADER NOAPPEND FILE=pktte_inf.tab
