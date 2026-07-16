$PROBLEM Per-route absorption lag -- NONMEM anchor for ferx FR1*first_order(ka=KA1)+FR2*first_order(ka=KA2, lag=LAG2)
; Reference fit for ferx's PER-ROUTE absorption lag: two first-order (Bateman)
; pathways feeding a 1-cpt central, split by a dose fraction FR1 / (1-FR1), where the
; second (delayed-release) pathway carries its OWN onset lag LAG2. Mirrors
; examples/per_route_lag_absorption.ferx and the parallel anchor
; (nonmem_anchor/parallel_first_order.ctl) -- same $DES + F1=0 + PODO trick, with the
; DR pathway's TAD shifted by LAG2 (its input switches on only after t = TDOS + LAG2).
;
; The dataset per_route_lag_oral.csv is simulated FROM this model
; (nonmem_anchor/simulate_per_route_lag_data.py, same truths), so this is a matched
; (well-specified) fit: ferx and NONMEM should agree on OFV/THETA AND both recover the
; data-generating values.
;
; Label symmetry is broken by the per-route lag itself: pathway 1 (lag 0) and
; pathway 2 (lag LAG2 > 0) are no longer exchangeable. The KA1 > KA2 bounds are kept
; for a clean fast-IR / slow-DR reading; use the same convention in the ferx fit.

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA per_route_lag_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); first-order sum feeds central
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)*EXP(ETA(2))
  FR1  = THETA(3)          ; fraction of dose through the immediate-release pathway
  KA1  = THETA(4)          ; IR-pathway absorption rate (1/h)
  KA2  = THETA(5)          ; DR-pathway absorption rate (1/h)
  LAG2 = THETA(6)          ; DR-pathway onset lag (h) -- the per-route lag

  FR2  = 1.0 - FR1
  K20  = CL/V

  ; PODO (last oral dose amount) and TDOS (its time) captured at the dose record;
  ; NONMEM carries them forward for $DES to read.
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0: no bolus; PODO drives the two first-order inputs in $DES (each pathway's
  ; integral R_in = FRi*dose, so the total delivered is the dose).
  F1   = 0.0

$DES
  ; Time after the most recent dose (TDOS captured in $PK); = T here (dose at t=0).
  TAD  = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; IR pathway: first-order input from the dose, no lag.
  FO1  = FR1*PODO*KA1*EXP(-KA1*TAD)
  ; DR pathway: first-order input shifted by the per-route lag LAG2 -- switches on
  ; only after TAD > LAG2, then decays from its own onset.
  TAD2 = TAD - LAG2
  FO2  = 0.0
  IF (TAD2.GT.0.0) FO2 = FR2*PODO*KA2*EXP(-KA2*TAD2)
  RIN  = FO1 + FO2               ; per-route-lag input rate (IR now + DR after LAG2)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; dual first-order input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,   5.0,  100)   ; 1 CL    (L/h)
  (5.0,   50.0, 500)   ; 2 V     (L)
  (0.001, 0.5,  0.999) ; 3 FR1         IR-pathway fraction
  (0.3,   1.2,  24)    ; 4 KA1   (1/h) IR pathway (bounded > KA2)
  (0.05,  0.8,  1.1)   ; 5 KA2   (1/h) DR pathway
  (0.001, 3.0,  12)    ; 6 LAG2  (h)   DR-pathway per-route lag

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=per_route_lag.tab
