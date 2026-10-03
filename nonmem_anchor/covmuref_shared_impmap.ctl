$PROBLEM two_cpt_oral_cov with a SHARED allometric exponent, IMPMAP -- anchor for ferx #1620
; Reference IMPMAP fit for two covariate mu-references that read the same theta:
;
;     CL_i = THETA(1) * (WT/70)**THETA(6) * (CRCL/100)**THETA(7) * EXP(ETA(1))
;     V1_i = THETA(2) * (WT/70)**THETA(6) * EXP(ETA(2))
;
; THETA(6) (THETA_WT) is shared by CL and V1. Before #1620 ferx declined the second
; covariate mu-reference ("already belongs") and let the first one's M-step move
; THETA_WT with ETA_V1 left free to absorb the change; SAEM put THETA_WT at 0.0101.
; With #1620 the two form one joint group whose M-step re-centres both etas.
;
; This is examples/two_cpt_oral_cov.ferx + data/two_cpt_oral_cov.csv, unchanged
; except that the observation rows carry CMT=2 (ADVAN4's central compartment) and
; '.' is written as 0. That the two files describe the same observations is checked
; by FOCEI: NONMEM on this file and ferx on the bundled CSV reach the same OFV,
; -1199.3264. The ferx side is covmuref_shared_saem_fit.ferx with method = impmap.
; Both engines start from the bundled model's values.
;
; MU form, linear in THETA for both typical values (NONMEM's efficient form).

$INPUT ID TIME DV EVID AMT CMT RATE MDV WT CRCL
$DATA covmuref_shared.csv IGNORE=@

$SUBROUTINES ADVAN4 TRANS4

$PK
  MU_1 = LOG(THETA(1)) + THETA(6)*LOG(WT/70.0) + THETA(7)*LOG(CRCL/100.0)
  MU_2 = LOG(THETA(2)) + THETA(6)*LOG(WT/70.0)
  MU_3 = LOG(THETA(3))
  MU_4 = LOG(THETA(4))
  MU_5 = LOG(THETA(5))
  CL = EXP(MU_1 + ETA(1))
  V2 = EXP(MU_2 + ETA(2))
  Q  = EXP(MU_3 + ETA(3))
  V3 = EXP(MU_4 + ETA(4))
  KA = EXP(MU_5 + ETA(5))
  S2 = V2

$ERROR
  IPRED = F
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1, 4.0, 100.0)   ; 1 TVCL
  (1.0, 40.0, 500.0)  ; 2 TVV1
  (0.1, 8.0, 100.0)   ; 3 TVQ
  (1.0, 80.0, 500.0)  ; 4 TVV2
  (0.01, 1.0, 10.0)   ; 5 TVKA
  (0.01, 0.6, 5.0)    ; 6 THETA_WT   (shared by CL and V1)
  (0.01, 0.3, 5.0)    ; 7 THETA_CRCL

$OMEGA
  0.15   ; ETA_CL
  0.15   ; ETA_V1
  0.08   ; ETA_Q
  0.08   ; ETA_V2
  0.20   ; ETA_KA

$SIGMA
  0.0016 ; proportional, = (0.04 SD)^2

$ESTIMATION METHOD=IMPMAP INTERACTION NITER=200 ISAMPLE=300 PRINT=20 SEED=1620 NOABORT
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=10 ISAMPLE=3000 PRINT=1 SEED=1620 NOABORT
$TABLE ID TIME DV IPRED NOPRINT ONEHEADER FILE=covmuref_shared_impmap.tab FORMAT=s1PE15.8
