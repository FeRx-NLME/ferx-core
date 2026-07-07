$PROBLEM RTTE constant-hazard with shared frailty (Phase 3 Slice 3.1 anchor)
; Clock-forward (Andersen-Gill) repeated time-to-event, constant hazard.
; No ODE is needed for a constant hazard: track the previous event time TLAST and
; contribute the survival over each inter-event gap times the hazard at an event,
; which telescopes to the AG likelihood  L = prod_k h * exp(-H(T)).
;
; Data: rtte_exp.csv (ID,TIME,DV,EVID,CMT,MDV). DV=1 event, DV=0 admin censor at t=20.
; A pure-TTE model: every row is a likelihood contribution, so Y is the likelihood and
; $EST carries LIKELIHOOD (no F_FLAG, which is only for mixed continuous+likelihood data).
$INPUT ID TIME DV EVID CMT MDV
$DATA rtte_exp.csv IGNORE=@

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

; LAPLACE likelihood (exact 2nd-order η-Hessian) — the NONMEM analogue of ferx FOCEI.
; The LIKELIHOOD option marks Y as an individual likelihood (F_FLAG=1), so no $SIGMA
; is present or needed.
$ESTIMATION METHOD=COND LAPLACE LIKELIHOOD NUMERICAL SLOW NOABORT PRINT=5 MAXEVAL=9999
$COVARIANCE UNCONDITIONAL SLOW
$TABLE ID TIME DV LAMBDA NOPRINT ONEHEADER FILE=rtte_sdtab
