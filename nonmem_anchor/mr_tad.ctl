$PROBLEM #1124 TAD-dependent DES on a closed-form-eligible absorption model
; Anchor for ferx issue #1124. The ODE right-hand side reads time-after-dose on a
; model whose absorption ferx can serve in CLOSED FORM (a single built-in
; `first_order` input-rate forcing into a linear one-compartment disposition).
; That fast path never evaluates the right-hand side, so the TAD term was dropped
; from the predictions entirely -- output was bit-identical to the same model with
; the term deleted. This anchor pins the corrected route against NONMEM.
;
; TWO DOSES, and no lagtime:
;   - Two doses are required. Under a single dose TAD == TAFD, so a one-dose
;     anchor cannot distinguish a TAD defect from a TAFD one and would agree with
;     a wrong engine. The second dose at t=12 re-anchors TAD strictly inside the
;     observation range, with records on both sides (11.5 and 12.5).
;   - No lagtime, so unlike tad_lag_B this needs no MTIME/MTDIFF: TDOS switches
;     exactly AT a dose record, and NONMEM already breaks the integration there.
;     (Verified against the independent RK4 reference in
;     mr_tad_deterministic_truth.py rather than assumed -- MTIME construction is
;     where the tad_lag anchors went wrong twice, see this file's README entry.)
;   - No WT column. tad_lag_{A,B} carry a FIXED-zero-exponent WT purely to force
;     ferx onto its event-driven predictor (#1110); #1124 routes a model-time-
;     reading RHS there directly, so that workaround is obsolete here.
;
; NONMEM uses an explicit DEPOT compartment; the ferx twin is a one-state model
; whose `first_order(ka=KA)` forcing delivers KA*D*exp(-KA*(t-t_dose)) into
; central -- exactly the outflow of this depot. Same system, same dataset.
$INPUT ID TIME DV MDV EVID AMT CMT
$DATA mr_tad.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=9
$MODEL COMP=(DEPOT,DEFDOSE) COMP=(CENTRAL)
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  KA = THETA(3)
$DES
  TDOS = 0.0
  IF (T.GE.12.0) TDOS = 12.0
  TADX = T - TDOS
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - (CL/V)*A(2)*(1.0 + THETA(4)*TADX)
$ERROR
  IPRED = A(2)/V
  Y     = IPRED*(1.0 + EPS(1))
$THETA
 (0.001, 1.0)   ; TVCL
 (0.001, 20.0)  ; TVV
 (0.001, 0.8)   ; TVKA
 (0.0,   0.3)   ; THETA_TAD
$OMEGA
 0.09   ; ETA_CL
$SIGMA
 0.01   ; proportional, variance
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1
$TABLE ID TIME IPRED CWRES NOPRINT ONEHEADER FILE=mr_tad.tab
