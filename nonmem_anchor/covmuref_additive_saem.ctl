$PROBLEM Additive renal covariate on CL, SAEM with a nonlinear MU -- anchor for ferx covariate mu-referencing [#619]
; Reference SAEM fit for ferx's multi-theta (covariate) mu-referencing.
; 1-cpt IV bolus; CL carries an additive renal gradient and IIV:
;
;     CL_i = (THETA(1) + (CRCL_i - 90) * THETA(2)) * EXP(ETA(1))
;
; which is not log-linear in a single theta, so the mu-reference is the
; *nonlinear* MU_1 = LOG(THETA(1) + (CRCL-90)*THETA(2)). NONMEM accepts a MU
; that is nonlinear in THETA (its EM update is then iterative rather than the
; one-line linear solve); ferx's covariate mu-reference group does the same
; re-fit of (THETA(1), THETA(2)) to the population of individual log-CL values.
;
; The dataset covmuref_additive.csv is simulated FROM this model
; (nonmem_anchor/simulate_covmuref_data.py, same truths), so this is a matched
; (well-specified) fit: NONMEM and ferx should agree AND both recover the
; data-generating values.
;
; Truths: TVCL=5, TH_CRCL=0.05, TVV=50, omega^2 = 0.09 (CL), 0.04 (V),
;         sigma^2 = 0.01. CRCL is constant within each subject.

$INPUT ID TIME DV AMT EVID CMT CRCL
$DATA covmuref_additive.csv IGNORE=@

$SUBROUTINES ADVAN1 TRANS2

$PK
  MU_1 = LOG(THETA(1) + (CRCL - 90.0)*THETA(2))   ; nonlinear in THETA
  MU_2 = LOG(THETA(3))
  CL = EXP(MU_1 + ETA(1))
  V  = EXP(MU_2 + ETA(2))
  S1 = V

$ERROR
  IPRED = F
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.0,  4.0,  100)   ; 1 TVCL     (L/h)   truth 5
  (0.0,  0.02, 5.0)   ; 2 TH_CRCL  (L/h per mL/min)  truth 0.05
  (1.0,  40.0, 500)   ; 3 TVV      (L)     truth 50

$OMEGA
  0.1    ; IIV CL   truth 0.09
  0.1    ; IIV V    truth 0.04

$SIGMA
  0.02   ; proportional residual variance, truth 0.01

$ESTIMATION METHOD=SAEM INTERACTION NBURN=2000 NITER=1000 ISAMPLE=10 PRINT=200 SEED=619 NOABORT
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=10 ISAMPLE=3000 PRINT=1 SEED=619 NOABORT
; No $COVARIANCE: point estimates only (printing R aborts this build on the SAEM+IMP chain).
$TABLE ID TIME DV IPRED CWRES NOPRINT ONEHEADER FILE=covmuref_additive_saem.tab
