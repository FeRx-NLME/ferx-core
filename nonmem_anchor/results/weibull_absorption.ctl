$PROBLEM Weibull absorption -- NONMEM anchor for ferx weibull(td, beta) [#322 Phase 2]
; Reference fit for ferx's built-in weibull(td, beta) input-rate function.
; Weibull density absorption straight into a 1-cpt central compartment.
;
; NOTE on the data: igd_oral.csv was simulated from a TRANSIT model, so this
; Weibull fit is mildly mis-specified (the Weibull density approximates the
; delayed-absorption shape). That is still a valid IMPLEMENTATION anchor:
; ferx-weibull and NONMEM-weibull should return the same OFV/THETA on identical
; data even when neither is the true DGP. As with the igd anchor, the check is the
; OBJECTIVE AT THE SHARED OPTIMUM, not the optimiser path (a mis-specified ridge
; lets NONMEM's gradient FOCEI and ferx's derivative-free BOBYQA stall at
; different points). For a matched Weibull-truth dataset, regenerate from
; examples/weibull_absorption.ferx --simulate (see README.md).

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA igd_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); weibull feeds central directly
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)*EXP(ETA(2))
  TD   = THETA(3)          ; Weibull scale (h)
  BETA = THETA(4)          ; Weibull shape (dimensionless)

  K20 = CL/V

  ; PODO (last oral dose amount) and TDOS (its time) are NOT reserved variables --
  ; capture them at the dose record; NONMEM carries them forward for $DES to read.
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0: no bolus; PODO drives the Weibull input in $DES (ferx weibull()
  ; convention -- integral R_in = dose, fed straight into central). With
  ; bioavailability, BIO*PODO.
  F1   = 0.0

$DES
  ; Time after the most recent dose (TDOS captured in $PK); = T here (dose at t=0).
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; R_in(tad) = PODO * (BETA/TD) * (TAD/TD)^(BETA-1) * exp(-(TAD/TD)^BETA)
  XW  = TAD/TD
  RIN = PODO*(BETA/TD)*XW**(BETA-1.0)*EXP(-XW**BETA)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; Weibull input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,  5.0,  100)   ; 1 CL   (L/h)
  (5.0,  50.0, 500)   ; 2 V    (L)
  (0.05, 2.0,  24)    ; 3 TD   (h)   Weibull scale
  (0.1,  1.5,  10)    ; 4 BETA       Weibull shape

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=weibull.tab
