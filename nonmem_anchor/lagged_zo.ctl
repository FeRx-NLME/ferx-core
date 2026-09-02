$PROBLEM Lagged zero-order absorption -- NONMEM anchor for ferx zero_order(dur=DUR, lag=LAG) [#1171]
; ferx:   d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central ;  y = central/V
; NONMEM: ADVAN13 $DES carries elimination only; the zero-order input is NONMEM's own
;         duration-modeled dose (RATE=-2 in the data + D1=DUR) shifted by ALAG1=LAG.
;         Unlike the PODO/F1=0 trick in mixed_zero_first.ctl, this superposes MULTIPLE
;         doses correctly -- required here, because the second dose's window must open
;         while drug from the first is still present.
;
; This is the single-lag (lag_cmt = 0) case, the only member of the per-route-lag family
; NONMEM can express: ALAGn is per-compartment, so a route lag COMPOSED with a
; compartment lag needs two compartments and stops being the same object (see the note
; in tests/dose_form_lag_nonmem_anchor.rs).
;
; MAXEVAL=0 POSTHOC at fixed THETA: the target of this anchor is the per-record IPRED
; column, not the OFV. ferx's OFV path (compute_predictions_with_tv) is already correct;
; the defect is in the sdtab IPRED path (compute_predictions_with_states), so an
; OFV-only anchor would pass vacuously in both directions.

$INPUT ID TIME DV AMT RATE EVID CMT MDV
$DATA lagged_zo_nm.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(CENTRAL,DEFDOSE,DEFOBS)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)*EXP(ETA(2))
  DUR = THETA(3)
  LAG = THETA(4)
  D1    = DUR
  ALAG1 = LAG
  K10   = CL/V

$DES
  DADT(1) = -K10*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  5.0  FIX   ; 1 CL  (L/h)
  50.0 FIX   ; 2 V   (L)
  2.0  FIX   ; 3 DUR (h)  zero-order input duration
  1.5  FIX   ; 4 LAG (h)  zero-order route lag

$OMEGA
  0.09   ; IIV CL
  0.04   ; IIV V

$SIGMA
  0.0225 ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME DV PRED IPRED CWRES ETA1 ETA2 MDV NOPRINT ONEHEADER FILE=lagged_zo.tab
