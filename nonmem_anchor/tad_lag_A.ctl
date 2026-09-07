$PROBLEM #1070 TAD-dependent DES x estimated lagtime with IIV -- single dose
; Anchor for ferx issue #1070. The ODE right-hand side reads time-after-dose,
; and the dose arrival t_dose+ALAG1 carries an eta, so TAD itself depends on
; eta_lag with dTAD/dALAG1 = -1. ferx lifts TAD as an f64 constant in its
; analytic sensitivity walk, dropping that term.
;
; Single dose, so time-after-dose needs no MTIME machinery: the only arrival is
; at 0+ALAG1, hence TD1 = ALAG1 exactly. Before the arrival A(1)=0, so the
; negative TADX in that window contributes nothing (DADT = 0) -- which is why
; this matches ferx, whose timeline simply starts at the arrival.
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA tad_lag_A.csv IGNORE=@
;
; WT carries a FIXED-at-zero exponent: (WT/70)**THETA(5) with THETA(5)=0 FIX is
; exactly 1, so WT has no effect on any prediction. It is present only because
; ferx routes a subject to its event-driven predictor when a covariate column
; actually varies within the subject, and the plain (dense) predictor currently
; returns NaN for a TAD-reading RHS under a lagtime (ferx #1110). Because the
; exponent is zero, every per-event covariate snapshot is identical, so this
; anchor does not exercise the time-varying-covariate snapshot convention and
; cannot be invalidated by ferx #1073's change to where the timeline breaks.
;
; NOTE (ferx #1124): this WT workaround is NO LONGER REQUIRED. #1124 routes any
; model-time-reading RHS to the event-driven predictor directly, so a TAD-reading
; model reaches it without needing a varying covariate to force the routing. WT is
; kept here only so a passing anchor is not perturbed -- removing it drops
; THETA(5) and would need a fresh NONMEM run and a new committed OFV. Do NOT copy
; this trick into a new anchor; see mr_tad.ctl, which has no WT column.
$SUBROUTINES ADVAN13 TOL=9
$MODEL COMP=(CENTRAL,DEFDOSE)
$PK
  CL    = THETA(1)*(WT/70)**THETA(5)*EXP(ETA(1))
  V     = THETA(2)
  ALAG1 = THETA(4)*EXP(ETA(2))
  TD1   = ALAG1
$DES
  TADX = T - TD1
  DADT(1) = -(CL/V)*A(1)*(1.0 + THETA(3)*TADX)
$ERROR
  IPRED = A(1)/V
  Y     = IPRED*(1.0 + EPS(1))
$THETA
 (0.001, 1.0)   ; TVCL
 (0.001, 20.0)  ; TVV
 (0.0,   0.3)   ; THETA_TAD
 (0.001, 0.5)   ; TVLAG
 (0.0 FIX)      ; WT exponent -- zero, see header
$OMEGA
 0.09   ; ETA_CL
 0.02   ; ETA_LAG
$SIGMA
 0.01   ; proportional, variance
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1
$TABLE ID TIME IPRED CWRES NOPRINT ONEHEADER FILE=tad_lag_A.tab
