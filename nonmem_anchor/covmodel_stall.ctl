$PROBLEM two-cpt oral with three covariate thetas (#1290 anchor)
; The NONMEM twin of examples/two_cpt_oral_covmodel.ferx, started from the same
; initial estimates, so the reported OFV / thetas are directly comparable with
; the ferx fit of that file.
$INPUT ID TIME DV EVID AMT CMT RATE MDV WT CRCL
$DATA covmodel_stall.csv IGNORE=@
; covmodel_stall.csv is data/two_cpt_oral_cov.csv with observation CMT
; recoded 1 -> 2: ferx `two_cpt_oral` reads CMT=1 observations from the central
; compartment, ADVAN4 numbers the depot 1 and the central 2.
$SUBROUTINE ADVAN4 TRANS4

$PK
  CL = THETA(1) * (WT/70)**THETA(6) * (CRCL/100)**THETA(7) * EXP(ETA(1))
  V2 = THETA(2) * (WT/70)**THETA(8) * EXP(ETA(2))
  Q  = THETA(3) * EXP(ETA(3))
  V3 = THETA(4) * EXP(ETA(4))
  KA = THETA(5) * EXP(ETA(5))
  S2 = V2

$ERROR
  IPRED = F
  Y = F * (1 + EPS(1))

$THETA
  (0.1, 4.0,  100.0)   ; TVCL
  (1.0, 40.0, 500.0)   ; TVV1
  (0.1, 8.0,  100.0)   ; TVQ
  (1.0, 80.0, 500.0)   ; TVV2
  (0.01, 1.0, 10.0)    ; TVKA
  (0.01, 0.6, 5.0)     ; THETA_CL_WT
  (0.01, 0.3, 5.0)     ; THETA_CL_CRCL
  (0.01, 0.6, 5.0)     ; THETA_V1_WT

$OMEGA
  0.15   ; ETA_CL
  0.15   ; ETA_V1
  0.08   ; ETA_Q
  0.08   ; ETA_V2
  0.20   ; ETA_KA

$SIGMA
  0.0016  ; PROP_ERR  (ferx declares `~ 0.04 (sd)`; NONMEM takes the variance)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE MATRIX=R
