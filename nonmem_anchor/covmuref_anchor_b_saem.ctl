$PROBLEM two_cpt_oral_cov, V1 typical value reads TVV2 (ETA_V2's plain anchor) -- SAEM anchor for ferx #1630 case B
; V1's typical value reads TVV2, the plain single-anchor mu-reference of ETA_V2:
;
;     V1_i = THETA(4) * THETA(9) * (WT/70)**THETA(8) * EXP(ETA(2))
;     V2_i = THETA(4) * EXP(ETA(4))
;
; An exact reparameterisation of the control V1 = TVV1 * (WT/70)**THETA(8) with
; TVV1 = THETA(4)*THETA(9). Before #1630 ferx SAEM declined V1's covariate
; mu-reference, and its covariate exponent sat on the eta-frozen numerical M-step
; (THETA_WT_V1 0.68 / 1.07 / 0.63 over three seeds, FOCEI 0.577). The ferx side is
; covmuref_anchor_b_saem_fit.ferx.
;
; Data: covmuref_shared.csv (see covmuref_shared_saem.ctl). THETA numbering follows
; the ferx model, with its unused, FIXed TVV1 (THETA 2) kept FIXed here.

$INPUT ID TIME DV EVID AMT CMT RATE MDV WT CRCL
$DATA covmuref_shared.csv IGNORE=@

$SUBROUTINES ADVAN4 TRANS4

$PK
  MU_1 = LOG(THETA(1)) + THETA(6)*LOG(WT/70.0) + THETA(7)*LOG(CRCL/100.0)
  MU_2 = LOG(THETA(4)) + LOG(THETA(9)) + THETA(8)*LOG(WT/70.0)
  MU_3 = LOG(THETA(3))
  MU_4 = LOG(THETA(4))
  MU_5 = LOG(THETA(5))
  CL = EXP(MU_1 + ETA(1))
  V2 = EXP(MU_2 + ETA(2))
  Q  = EXP(MU_3 + ETA(3))
  V3 = EXP(MU_4 + ETA(4))
  KA = EXP(MU_5 + ETA(5))
  DUM = THETA(2)
  S2 = V2

$ERROR
  IPRED = F
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1, 4.0, 100.0)   ; 1 TVCL
  40.0 FIX            ; 2 TVV1 (unused)
  (0.1, 8.0, 100.0)   ; 3 TVQ
  (1.0, 80.0, 500.0)  ; 4 TVV2
  (0.01, 1.0, 10.0)   ; 5 TVKA
  (0.01, 0.6, 5.0)    ; 6 THETA_WT
  (0.01, 0.3, 5.0)    ; 7 THETA_CRCL
  (0.01, 0.6, 5.0)    ; 8 THETA_WT_V1
  (0.01, 0.5, 50.0)   ; 9 TH_R

$OMEGA
  0.15   ; ETA_CL
  0.15   ; ETA_V1
  0.08   ; ETA_Q
  0.08   ; ETA_V2
  0.20   ; ETA_KA

$SIGMA
  0.0016 ; proportional, = (0.04 SD)^2

$ESTIMATION METHOD=SAEM INTERACTION NBURN=2000 NITER=1000 ISAMPLE=10 PRINT=200 SEED=1630 NOABORT
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=10 ISAMPLE=3000 PRINT=1 SEED=1630 NOABORT
