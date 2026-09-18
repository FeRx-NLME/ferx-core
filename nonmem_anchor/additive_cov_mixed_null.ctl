$PROBLEM NULL TWIN of multiplicative AND additive covariate effects on one CL (#1313 anchor, arm B)
; CL = TVCL*(WT/70)**THETA(6)*EXP(ETA(1)) + THETA(7)*(CRCL - 100)
;
; Arm A anchors the additive combination on its own. This arm anchors the
; *placement rule* that comes with it: a parameter carrying both kinds of
; relation must read as (factors on the typical value) + (terms), never as
; a factor multiplied into one addend of the sum. The two readings differ
; numerically — `TVCL*f*EXP(ETA) + t` is not `TVCL*f*(EXP(ETA) + t)` and is not
; `(TVCL*EXP(ETA) + t)*f` — so NONMEM's own spelling settles it.
;
; MAXEVAL=0 with POSTHOC: nothing is estimated, so the comparison is of
; arithmetic rather than of where two optimizers stop.
;
; Non-degeneracy, both effects and both signs: WT runs 45.0 .. 93.7 about a
; centre of 70 (so the power factor spans 0.79 .. 1.16 at THETA(6) = 0.6), and
; CRCL runs 46.5 .. 150.0 about a centre of 100, so the additive term spans
; -1.07 .. +1.00 at THETA(7) = 0.02 against a TVCL of 4.0. Neither is a small
; perturbation of the other, and the two covariates are different columns, so
; an implementation that applied one effect twice cannot pass.
$INPUT ID TIME DV EVID AMT CMT RATE MDV WT CRCL
$DATA covmodel_stall.csv IGNORE=@
$SUBROUTINE ADVAN4 TRANS4

$PK
  CLCRCL = THETA(7) * (CRCL - 100)
  CL = THETA(1) * (WT/70)**THETA(6) * EXP(ETA(1)) + CLCRCL
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
  (0.01, 0.6, 5.0)     ; THETA_CL_WT, the multiplicative exponent
  0 FIX                ; THETA_CL_CRCL held at zero — the null twin

$OMEGA
  0.15   ; ETA_CL
  0.15   ; ETA_V1
  0.08   ; ETA_Q
  0.08   ; ETA_V2
  0.20   ; ETA_KA

$SIGMA
  0.0016  ; PROP_ERR

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT
; The null twin of the arm above: identical in every respect except that the
; additive slope is held at zero. The OFV *difference* between the two is what
; the anchor compares, which cancels a baseline ferx-vs-NONMEM offset that is
; present with no covariate model at all (see the test module docs).
$TABLE ID TIME PRED IPRED NOPRINT ONEHEADER FILE=additive_cov_mixed_null.tab FORMAT=s1PE23.16
