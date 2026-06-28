$PROBLEM Mixed zero+first-order absorption -- NONMEM anchor for ferx FZO1*first_order(ka=KA)+FZO*zero_order(dur=DUR) [#505]
; Reference fit for ferx's mixed absorption: a zero-order (constant-rate over a
; modeled duration) input plus a first-order (Bateman) input feeding a 1-cpt central
; compartment in parallel, split by a dose fraction FZO / (1-FZO). Mirrors
; examples/mixed_absorption.ferx and the biphasic IG anchor -- same $DES + F1=0 +
; PODO trick, with one IG density replaced by a first-order input and the other by a
; zero-order rectangle.
;
; The dataset mixed_oral.csv is simulated FROM this model
; (nonmem_anchor/simulate_mixed_data.py, same truths), so this is a matched
; (well-specified) fit: ferx and NONMEM should agree on OFV/THETA AND both recover
; the data-generating values.
;
; No pathway-label symmetry here (zero-order and first-order are distinguishable), so
; no ordering bound is needed -- unlike the parallel / biphasic anchors.

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA mixed_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); zero+first sum feeds central
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)*EXP(ETA(2))
  FZO = THETA(3)           ; fraction of dose through the zero-order pathway
  KA  = THETA(4)           ; first-order absorption rate (1/h)
  DUR = THETA(5)           ; zero-order input duration (h)

  FZO1 = 1.0 - FZO         ; first-order-pathway fraction
  K20  = CL/V

  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0: no bolus; PODO drives the two inputs in $DES (zero-order pathway delivers
  ; FZO*dose over DUR, first-order pathway delivers FZO1*dose -- total = dose).
  F1  = 0.0

$DES
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; First-order pathway: RFO = FZO1 * PODO * KA * exp(-KA*tad).
  RFO = FZO1*PODO*KA*EXP(-KA*TAD)
  ; Zero-order pathway: constant FZO*PODO/DUR over (0, DUR], 0 after the cutoff.
  RZO = 0.0
  IF (TAD.LE.DUR) RZO = FZO*PODO/DUR
  RIN = RFO + RZO                ; mixed input rate (dose split FZO / 1-FZO)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; zero+first-order input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,   5.0,  100)   ; 1 CL    (L/h)
  (5.0,   50.0, 500)   ; 2 V     (L)
  (0.001, 0.4,  0.999) ; 3 FZO         zero-order-pathway fraction
  (0.05,  1.0,  24)    ; 4 KA    (1/h) first-order pathway
  (0.05,  3.0,  24)    ; 5 DUR   (h)   zero-order duration

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=mixed_zero_first.tab
