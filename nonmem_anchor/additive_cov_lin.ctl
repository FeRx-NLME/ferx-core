$PROBLEM additive (+) covariate effect on CL (#1313 anchor, arm A)
; CL = TVCL*EXP(ETA(1)) + THETA(6)*(WT - 70)
;
; The object under test is the *combination*: ferx's `CL ~ WT linear(center =
; 70) +` desugars to exactly this sum, where every covariate relation before
; #1313 was a factor. NONMEM writes it directly, so this is an ordinary
; anchored comparison — no exception applies.
;
; MAXEVAL=0 with POSTHOC: nothing is estimated, so both engines evaluate the
; same parameter vector and the comparison is of arithmetic, not of where two
; optimizers stop. PRED is the population prediction (ETA = 0), where the
; additive term is the only covariate effect live at all.
;
; Non-degeneracy: WT runs 45.0 .. 93.7 about a centre of 70, so the term takes
; BOTH signs across the 30 subjects, and at THETA(6) = 0.04 it moves CL by
; -1.00 .. +0.95 against a TVCL of 4.0 — i.e. by up to 25%. A term that only
; ever added a small positive amount would be satisfied by an implementation
; that got the sign or the magnitude wrong.
$INPUT ID TIME DV EVID AMT CMT RATE MDV WT CRCL
$DATA covmodel_stall.csv IGNORE=@
; covmodel_stall.csv is data/two_cpt_oral_cov.csv with observation CMT
; recoded 1 -> 2: ferx `two_cpt_oral` reads CMT=1 observations from the central
; compartment, ADVAN4 numbers the depot 1 and the central 2.
$SUBROUTINE ADVAN4 TRANS4

$PK
  CLWGT = THETA(6) * (WT - 70)
  CL = THETA(1) * EXP(ETA(1)) + CLWGT
  V2 = THETA(2) * EXP(ETA(2))
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
  (-1.0, 0.04, 1.0)    ; THETA_CL_WT, the additive slope

$OMEGA
  0.15   ; ETA_CL
  0.15   ; ETA_V1
  0.08   ; ETA_Q
  0.08   ; ETA_V2
  0.20   ; ETA_KA

$SIGMA
  0.0016  ; PROP_ERR  (ferx declares `~ 0.04 (sd)`; NONMEM takes the variance)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME PRED IPRED NOPRINT ONEHEADER FILE=additive_cov_lin.tab FORMAT=s1PE23.16
