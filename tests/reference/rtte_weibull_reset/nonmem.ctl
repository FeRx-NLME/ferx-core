$PROBLEM Clock-reset (gap-time) Weibull RTTE with frailty (Phase 3 Slice 3.2 anchor)
; Renewal RTTE: the hazard clock resets at each event, so each row contributes the
; Weibull survival/density over its inter-event gap DEL = TIME - TLAST. Pure-TTE model
; ($EST ... LIKELIHOOD, Y is the likelihood; no F_FLAG). No $DES needed — the closed-form
; Weibull is evaluated on the gap and TLAST is advanced after each event.
;
; Data: rtte_weibull_reset.csv (ID,TIME,DV,EVID,CMT,MDV). DV=1 event, DV=0 censor at t=30.
$INPUT ID TIME DV EVID CMT MDV
$DATA rtte_weibull_reset.csv IGNORE=@

$PRED
  IF (NEWIND.NE.2) TLAST = 0        ; reset the gap clock at each new subject
  SCALE = THETA(1)*EXP(ETA(1))      ; Weibull scale with frailty (H = (t/scale)^shape)
  SHAPE = THETA(2)
  DEL = TIME - TLAST                 ; gap duration since the previous event (clock reset)
  CH  = (DEL/SCALE)**SHAPE           ; H(DEL)
  SUR = EXP(-CH)
  HAZ = (SHAPE/SCALE)*(DEL/SCALE)**(SHAPE-1)
  Y = SUR                            ; censored: survival over the final gap
  IF (DV.EQ.1) Y = SUR*HAZ           ; event: density over the gap
  IF (DV.EQ.1) TLAST = TIME          ; advance the clock AFTER using the old TLAST

$THETA (0, 5.0)                      ; TVSCALE
$THETA (0, 1.5)                      ; TVSHAPE
$OMEGA 0.09                          ; frailty variance (var of log scale)

$ESTIMATION METHOD=COND LAPLACE LIKELIHOOD NUMERICAL SLOW NOABORT PRINT=5 MAXEVAL=9999
$COVARIANCE UNCONDITIONAL SLOW
