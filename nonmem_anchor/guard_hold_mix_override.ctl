$PROBLEM Held covariance: a [mixture] override FIX at ferx's internal rail (#2809)
; ferx holds a per-class override collapsed onto its internal lower guard
; (omega = exp(-12) = 6.1442123533282098E-6) in the covariance step, so every
; other standard error is conditional on it at its estimate. "Held" means FIX,
; and this is the NONMEM run of exactly that: the #2809 fixture
; (tests/data/guard_hold/mix_override_collapse.csv, two classes, class 2
; simulated with no CL variability, generator beside it; copied here unchanged
; as guard_hold_mix_override.csv) at ferx's FOCEI estimate, with the class-2 CL
; variance FIXed on the rail, MAXEVAL=0 and $COV MATRIX=R.
;
; ferx's `omega(2) ETA_CL` is a second ETA here, used only when MIXNUM.EQ.2
; (ETA(1) is the base, class 1). An ETA a class does not use leaves that class's
; FOCE marginal unchanged (its prior and Hessian terms cancel), so the two
; spellings are the same likelihood. ferx's `logit(1) = MIXL` is P(1) = expit(MIXL).
;
; $COV runs at SIGL=12 (reference_nonmem_cov_precision). The comparison and the
; accept rule are in nonmem_anchor/README.md and the #2809 PR.
$INPUT ID TIME DV AMT EVID CMT WT
$DATA guard_hold_mix_override.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$MIX
  NSPOP = 2
  P(1) = 1 / (1 + EXP(-THETA(4)))
  P(2) = 1 - P(1)

$PK
  IF (MIXNUM.EQ.1) THEN
    CL = THETA(1) * EXP(ETA(1))
  ELSE
    CL = THETA(2) * EXP(ETA(2))
  ENDIF
  V  = THETA(3)
  S1 = V

$ERROR
  IPRED = F
  Y = F * (1 + EPS(1))

$THETA (0.01, 1.2921355862556707, 100.0)    ; TVCL1
$THETA (0.01, 2.526598380988927, 100.0)     ; TVCL2
$THETA (0.1, 10.040303488207734, 1000.0)    ; TVV
$THETA (-10.0, -0.11953563542469542, 10.0)  ; MIXL
$OMEGA 0.06227533113210565                  ; ETA_CL, class 1 (the base)
$OMEGA 6.1442123533282098E-6 FIX            ; ETA_CL_MIX2, on ferx's rail
$SIGMA 0.010606851598516967                 ; EPS^2 (ferx: SD 0.10298957033853946)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC SIGL=12 NSIG=4 PRINT=1
$COVARIANCE MATRIX=R SIGL=12 PRINT=E
$TABLE ID TIME DV PRED IPRED NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=guard_hold_mix_override.tab
