$PROBLEM one_cpt_transit (N=3) + WT-on-CL covariate + MULTIPLE doses -- NONMEM anchor for ferx [#719]
; Reference fit for ferx's ANALYTIC `pk one_cpt_transit(cl,v,n,mtt)` closed form under a continuous
; TV covariate (WT on CL, power model) and MULTIPLE overlapping doses (100 mg q6h x 5). No
; lagtime/F, so ferx uses its analytic exponential-tilting superposition (not the ODE twin); this
; control is the NONMEM cross-check for that analytic multi-dose + covariate path.
;
; EXPLICIT INTEGER-N TRANSIT CHAIN. For integer n the Savic density
;   R_in(t) = D * ktr * (ktr*t)^n * exp(-ktr*t) / n!,  ktr = (n+1)/MTT
; is exactly the input rate into central from an Erlang chain of (n+1) first-order transit
; compartments (dose in the first, flux from the (n+1)-th into central, every transfer rate ktr).
; For n = 3 that is 4 transit compartments (1..4) + central (5). NONMEM integrates the chain with
; its NATIVE multi-dose bookkeeping, so superposition across the five q6h doses is automatic -- no
; hand-coded $DES sum over PODO/TDOS (which sees only the most-recent dose). ferx's
; single-compartment one_cpt_transit(n=3 FIX) yields the identical central profile.
;
; Dose lands in CMT 1 (matches ferx's dose CMT 1). Observations are on CMT 5 (central) here; the
; ferx copy (data/transit_multidose_cov.csv) re-keys obs to CMT 1 (ferx's single disposition
; compartment) -- same DV, likelihood-identical. Data simulated FROM ferx
; (tests/gen_transit_multidose_cov_anchor.rs), so NONMEM also recovers the truths.

$INPUT ID TIME DV EVID AMT CMT MDV WT
$DATA transit_multidose_cov.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(TR1,DEFDOSE)   ; 1 = first transit compartment (dose lands here)
  COMP=(TR2)           ; 2
  COMP=(TR3)           ; 3
  COMP=(TR4)           ; 4 = (n+1)=4th transit compartment; its outflow feeds central
  COMP=(CENTRAL,DEFOBS); 5 = central (observed)

$PK
  TVCL  = THETA(1)
  TVV   = THETA(2)
  TVMTT = THETA(3)
  NN    = THETA(4)      ; transit count n, FIXed = 3 (integer -> explicit 4-transit chain)
  DWTCL = THETA(5)      ; WT-on-CL power

  CL  = TVCL*(WT/70.0)**DWTCL*EXP(ETA(1))
  V   = TVV*EXP(ETA(2))
  MTT = TVMTT

  KTR = (NN + 1.0)/MTT
  K10 = CL/V

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
  (0.1,  9.0,  100)   ; 1 TVCL
  (1.0,  60.0, 500)   ; 2 TVV
  (0.05, 3.0,  24)    ; 3 TVMTT
  3 FIX               ; 4 NN   (integer transit count -> explicit 4-transit chain)
  (0.0,  0.75, 2.0)   ; 5 DWTCL  WT-on-CL power

$OMEGA
  0.01    ; IIV CL (ETA1)
  0.09    ; IIV V  (ETA2)

$SIGMA
  0.01    ; proportional residual variance (0.1^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV WT NOPRINT ONEHEADER FILE=transit_multidose_cov.tab
