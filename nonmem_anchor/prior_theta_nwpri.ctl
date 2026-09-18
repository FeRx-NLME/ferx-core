; #254 parameter-prior anchor — the PRIORED run (step 2 of 2).
;
; Paired with `prior_theta_nwpri_centered.ctl`, which is identical except for ONE
; number: `$THETAP`'s first entry. That is deliberate — the anchor is a difference
; of two NONMEM runs varying a single input, so every additive constant NONMEM
; carries (its own normalisation of the prior density, the data term, whatever
; either engine includes or drops) cancels exactly, and what remains is the
; prior's quadratic contribution alone.
;
;   centered run : THETAP(1) = -2.0000    == THETA(1), so its quadratic term is 0
;   this run     : THETAP(1) = -1.8971200 == log(0.15), so it is not
;
;   expected difference = ((-2.0000 + 1.8971200) / 0.30)^2 = 0.1176027...
;
; Why NTHETA=3 and not 1: NWPRI's `NTHETA` must cover every THETA in the
; estimation problem, so thetas 2 and 3 are given prior means exactly equal to
; their own values (zero quadratic contribution) and a variance of 1e6 (flat).
; They contribute the same constant to both runs and drop out.
;
; `CL = EXP(THETA(1) + ETA(1))` makes THETA(1) log CL — the same coordinate ferx
; penalizes for an identity-packed theta — so this is a normal prior on the same
; quantity, with the same mean and the same SD, as the ferx twin.
;
; MAXEVAL=0: the quantity under test is the OBJECTIVE, not an optimizer's path to
; a minimum. Pinning the point removes any chance the two engines' minimizers
; land somewhere different and confound the comparison.
$PROBLEM warfarin 1-cpt oral, log-parameterised, NWPRI on log CL (offset)

$INPUT ID TIME DV EVID AMT DROP RATE MDV
$DATA ../data/warfarin.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PRIOR NWPRI NTHETA=3 NETA=2 NTHP=3 NETP=0

$PK
  CL = EXP(THETA(1) + ETA(1))
  V  = EXP(THETA(2) + ETA(2))
  KA = EXP(THETA(3))
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED * (1 + EPS(1))

$THETA -2.0000      ; log CL
$THETA  2.0450      ; log V
$THETA -0.3212      ; log KA

$OMEGA 0.0455       ; IIV CL
$OMEGA 0.0145       ; IIV V

$SIGMA 0.01055      ; proportional

; --- prior block -----------------------------------------------------------
; THETAP(1) = log(0.15); the other two sit on their own values (inert).
$THETAP (-1.8971200 FIX) (2.0450 FIX) (-0.3212 FIX)

; Lower triangle of the prior VARIANCE matrix, as a BLOCK — NWPRI requires the
; `BLOCK(n) FIX` form here and rejects per-element `FIX` with "INPUTS SPECIFIED
; TO ROUTINE NWPRI ARE INAPPROPRIATE".
;
;   0.30^2 = 0.09 on log CL (the prior under test);
;   100 on the other two — flat for a log-scale parameter of order 1, and they
;   already sit on their own prior means, so their quadratic term is 0 in BOTH
;   runs and cancels in the difference.
$THETAPV BLOCK(3) FIX
 0.09
 0.0   100.0
 0.0   0.0    100.0

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 PRINT=1 NOABORT
