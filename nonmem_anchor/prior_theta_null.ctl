; #254 parameter-prior anchor — STEP 1, the null control.
;
; One-compartment oral warfarin, with the structural parameters written on the
; LOG scale (CL = EXP(THETA(1))) so that a NONMEM $PRIOR NWPRI normal prior on
; THETA(1) is a normal prior on exactly the coordinate ferx penalizes. This run
; carries NO prior: it fixes the common baseline the priored run is compared
; against, so the anchor is a DELTA and every normalization constant — NONMEM's
; and ferx's alike — cancels.
;
; Estimation is evaluation-only (MAXEVAL=0) at a fixed point, deliberately: the
; quantity under test is the OBJECTIVE, not an optimizer's path to a minimum, and
; pinning the point removes any chance that the two engines' minimizers land
; somewhere different and confound the comparison.
$PROBLEM warfarin 1-cpt oral, log-parameterised, no prior

$INPUT ID TIME DV EVID AMT DROP RATE MDV
$DATA ../data/warfarin.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  CL = EXP(THETA(1) + ETA(1))
  V  = EXP(THETA(2) + ETA(2))
  KA = EXP(THETA(3))
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED * (1 + EPS(1))

; LOG-scale typical values. exp(-2.0217) = 0.13246, exp(2.0450) = 7.7297,
; exp(-0.3212) = 0.72526 — the unpriored ferx optimum, so both engines are
; evaluated at the same point.
$THETA -2.0000      ; log CL
$THETA  2.0450      ; log V
$THETA -0.3212      ; log KA

$OMEGA 0.0455       ; IIV CL
$OMEGA 0.0145       ; IIV V

$SIGMA 0.01055      ; proportional

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 PRINT=1 NOABORT
