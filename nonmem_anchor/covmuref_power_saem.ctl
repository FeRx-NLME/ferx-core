$PROBLEM Allometric CL with estimated exponent, SAEM with a linear MU -- anchor for ferx covariate mu-referencing [#619]
; Reference SAEM fit for the second shape of ferx's covariate mu-referencing:
;
;     CL_i = THETA(1) * (WT_i/70)**THETA(2) * EXP(ETA(1))
;
; i.e. MU_1 = LOG(THETA(1)) + THETA(2)*LOG(WT/70) -- linear in THETA, NONMEM's
; efficient MU form. ferx used to mu-reference only THETA(1) here (the power
; factor hides THETA(2) from the single-anchor detector) and left THETA(2) on
; the eta-frozen numerical M-step; with #619 both thetas form one covariate
; mu-reference group and are re-fitted jointly.
;
; Data covmuref_power.csv is simulated FROM this model
; (nonmem_anchor/simulate_covmuref_data.py). Truths: TVCL=5, TH_WT=0.75,
; TVV=50, omega^2 = 0.09 (CL), 0.04 (V), sigma^2 = 0.01. WT constant per subject.

$INPUT ID TIME DV AMT EVID CMT WT
$DATA covmuref_power.csv IGNORE=@

$SUBROUTINES ADVAN1 TRANS2

$PK
  MU_1 = LOG(THETA(1)) + THETA(2)*LOG(WT/70.0)
  MU_2 = LOG(THETA(3))
  CL = EXP(MU_1 + ETA(1))
  V  = EXP(MU_2 + ETA(2))
  S1 = V

$ERROR
  IPRED = F
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.0,  4.0,  100)   ; 1 TVCL   (L/h)  truth 5
  (-2.0, 0.3,  3.0)   ; 2 TH_WT  allometric exponent, truth 0.75
  (1.0,  40.0, 500)   ; 3 TVV    (L)    truth 50

$OMEGA
  0.1    ; IIV CL   truth 0.09
  0.1    ; IIV V    truth 0.04

$SIGMA
  0.02   ; proportional residual variance, truth 0.01

$ESTIMATION METHOD=SAEM INTERACTION NBURN=2000 NITER=1000 ISAMPLE=10 PRINT=200 SEED=619 NOABORT
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=10 ISAMPLE=3000 PRINT=1 SEED=619 NOABORT
$TABLE ID TIME DV IPRED CWRES NOPRINT ONEHEADER FILE=covmuref_power_saem.tab
