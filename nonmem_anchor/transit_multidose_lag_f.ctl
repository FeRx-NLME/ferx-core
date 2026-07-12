$PROBLEM one_cpt_transit (N=3) + lagtime + F + MULTIPLE doses -- NONMEM anchor for ferx [#719]
; Reference fit for ferx's `one_cpt_transit(..., lagtime=LAG, f=FB)`. Because a lagtime/F mapping
; is present, ferx routes each subject to its `transit()` ODE twin (#735); this control is the
; NONMEM cross-check for that twin path under an estimated absorption lag, a fixed bioavailability,
; and multiple overlapping doses.
;
; Same EXPLICIT INTEGER-N transit chain as transit_multidose_cov.ctl (n=3 -> 4 transit
; compartments 1..4 + central 5, every transfer rate ktr=(n+1)/MTT), so multi-dose superposition
; is NONMEM-native. Lag and bioavailability are the compartment-1 dose attributes ALAG1 / F1 --
; both NONMEM-native, so there is NO hand-coded $DES time shift (which choked LSODA in the
; first_order_alag experiment; a native ALAG1 restarts the integrator cleanly at the lagged dose).
;
;   * ALAG1 (THETA5) -- estimated absorption lagtime; delays every dose into compartment 1.
;   * F1 (THETA6)    -- bioavailability, FIXed to 0.70. F is structurally non-identifiable on
;                       single-route oral data (only CL/F, V/F are), so both engines FIX it; the
;                       anchor verifies ferx APPLIES F (scales the absorbed amount) identically to
;                       NONMEM, not that it estimates it.
;   * CL = TVCL*(WT/70)**0.75  -- a fixed allometric covariate that composes on the twin path.
;
; Dose CMT 1 (matches ferx). Obs CMT 5 here; the ferx copy (data/transit_multidose_lag_f.csv)
; re-keys obs to CMT 1 (ferx's single disposition compartment). Data simulated FROM ferx
; (tests/gen_transit_multidose_lag_f_anchor.rs), so NONMEM also recovers the truths.

$INPUT ID TIME DV EVID AMT CMT MDV WT
$DATA transit_multidose_lag_f.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(TR1,DEFDOSE)   ; 1 = first transit compartment (dose lands here; ALAG1/F1 apply)
  COMP=(TR2)           ; 2
  COMP=(TR3)           ; 3
  COMP=(TR4)           ; 4 = (n+1)th transit compartment; its outflow feeds central
  COMP=(CENTRAL,DEFOBS); 5 = central (observed)

$PK
  TVCL  = THETA(1)
  TVV   = THETA(2)
  TVMTT = THETA(3)
  NN    = THETA(4)      ; transit count n, FIXed = 3

  CL  = TVCL*(WT/70.0)**0.75*EXP(ETA(1))
  V   = TVV*EXP(ETA(2))
  MTT = TVMTT

  KTR = (NN + 1.0)/MTT
  K10 = CL/V

  ALAG1 = THETA(5)      ; absorption lagtime (estimated), delays the dose into compartment 1
  F1    = THETA(6)      ; bioavailability (FIX 0.70); scales the dose into compartment 1

$DES
  DADT(1) = -KTR*A(1)
  DADT(2) =  KTR*A(1) - KTR*A(2)
  DADT(3) =  KTR*A(2) - KTR*A(3)
  DADT(4) =  KTR*A(3) - KTR*A(4)
  DADT(5) =  KTR*A(4) - K10*A(5)

$ERROR
  IPRED = A(5)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1,  9.0,  100)    ; 1 TVCL
  (1.0,  60.0, 500)    ; 2 TVV
  (0.05, 3.0,  24)     ; 3 TVMTT
  3 FIX                ; 4 NN   (integer transit count -> explicit 4-transit chain)
  (0.001, 0.5, 3.0)    ; 5 ALAG1  absorption lagtime (estimated)
  0.7 FIX              ; 6 F1     bioavailability (non-identifiable single-route oral -> fixed)

$OMEGA
  0.09    ; IIV CL (ETA1)
  0.09    ; IIV V  (ETA2)

$SIGMA
  0.01    ; proportional residual variance (0.1^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV WT NOPRINT ONEHEADER FILE=transit_multidose_lag_f.tab
