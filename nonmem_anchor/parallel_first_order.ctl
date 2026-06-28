$PROBLEM Parallel dual first-order absorption -- NONMEM anchor for ferx FR1*first_order(ka=KA1)+FR2*first_order(ka=KA2) [#505]
; Reference fit for ferx's parallel (dual first-order) absorption: two first-order
; (Bateman) absorption pathways feeding a 1-cpt central compartment, split by a dose
; fraction FR1 / (1-FR1). Mirrors examples/parallel_absorption.ferx and the biphasic
; IG anchor (nonmem_anchor/freijer_biphasic_ig.ctl) -- same $DES + F1=0 + PODO trick,
; with each IG density replaced by a first-order input KA*exp(-KA*tad).
;
; The dataset parallel_oral.csv is simulated FROM this model
; (nonmem_anchor/simulate_parallel_data.py, same truths), so this is a matched
; (well-specified) fit: ferx and NONMEM should agree on OFV/THETA AND both recover
; the data-generating values.
;
; Identifiability: the two pathways are exchangeable (pathway 1 <-> 2 with
; FR1 <-> 1-FR1, KA1 <-> KA2). The KA1 > KA2 bounds below (fast vs slow) break that
; label symmetry; keep the same convention in the ferx fit when comparing.

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA parallel_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); first-order sum feeds central
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)*EXP(ETA(2))
  FR1 = THETA(3)           ; fraction of dose through the fast pathway
  KA1 = THETA(4)           ; fast-pathway absorption rate (1/h)
  KA2 = THETA(5)           ; slow-pathway absorption rate (1/h)

  FR2 = 1.0 - FR1
  K20 = CL/V

  ; PODO (last oral dose amount) and TDOS (its time) captured at the dose record;
  ; NONMEM carries them forward for $DES to read.
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0: no bolus; PODO drives the two first-order inputs in $DES (each pathway's
  ; integral R_in = FRi*dose, so the total delivered is the dose).
  F1  = 0.0

$DES
  ; Time after the most recent dose (TDOS captured in $PK); = T here (dose at t=0).
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; Each pathway: first-order (Bateman) input FOi = FRi * PODO * KAi * exp(-KAi*tad).
  FO1 = FR1*PODO*KA1*EXP(-KA1*TAD)
  FO2 = FR2*PODO*KA2*EXP(-KA2*TAD)
  RIN = FO1 + FO2                ; parallel input rate (dose split FR1 / 1-FR1)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; dual first-order input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,   5.0,  100)   ; 1 CL    (L/h)
  (5.0,   50.0, 500)   ; 2 V     (L)
  (0.001, 0.6,  0.999) ; 3 FR1         fast-pathway fraction
  (0.5,   1.5,  24)    ; 4 KA1   (1/h) fast pathway (bounded > KA2 to break label symmetry)
  (0.01,  0.3,  0.5)   ; 5 KA2   (1/h) slow pathway

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=parallel_first_order.tab
