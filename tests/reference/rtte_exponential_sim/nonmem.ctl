$PROBLEM RTTE simulate cross-tool anchor (Phase 3 Slice 3.3) — NONMEM fits ferx-simulated data
; Clock-forward (Andersen-Gill) constant-hazard RTTE with a shared log-rate frailty.
; The dataset rtte_sim.csv was produced by ferx's OWN simulator
; (`cargo run --bin rtte_sim_anchor --features survival`), from the truth
; TVLAMBDA = 0.15, omega^2(log rate) = 0.09, horizon = 20. Fitting it with an
; INDEPENDENT engine (NONMEM) must recover that truth — the external corroboration
; that the ferx RTTE *simulator* is correct (a biased sampler would shift the
; recovered parameters away from the truth here).
;
; Same telescoping-AG $PRED as tests/reference/rtte_exponential/nonmem.ctl: track the
; previous event time TLAST and contribute the survival over each inter-event gap
; times the hazard at an event, which telescopes to L = prod_k h * exp(-H(T)).
;
; Data: rtte_sim.csv (ID,TIME,DV,EVID,CMT,MDV). DV=1 event, DV=0 admin censor at t=20.
$INPUT ID TIME DV EVID CMT MDV
$DATA rtte_sim.csv IGNORE=@

$PRED
  IF (NEWIND.NE.2) TLAST = 0        ; reset the gap clock at each new subject
  LAMBDA = THETA(1)*EXP(ETA(1))     ; individual constant hazard (frailty on log rate)
  CUMHAZ = LAMBDA*(TIME - TLAST)    ; H over the gap since the previous event
  SURV   = EXP(-CUMHAZ)
  Y = SURV                          ; default (DV=0 censor): survival over the final gap
  IF (DV.EQ.1) Y = SURV*LAMBDA      ; event: density over the gap
  IF (DV.EQ.1) TLAST = TIME         ; advance the clock AFTER using the old TLAST

$THETA (0, 0.15)                    ; TVLAMBDA (population rate)
$OMEGA 0.09                         ; frailty variance (var of log rate)

; LAPLACE likelihood (exact 2nd-order eta-Hessian) — the NONMEM analogue of ferx FOCEI.
$ESTIMATION METHOD=COND LAPLACE LIKELIHOOD NUMERICAL SLOW NOABORT PRINT=5 MAXEVAL=9999
$COVARIANCE UNCONDITIONAL SLOW
$TABLE ID TIME DV LAMBDA NOPRINT ONEHEADER FILE=rtte_sim_sdtab
