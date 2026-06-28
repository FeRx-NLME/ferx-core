$PROBLEM Biphasic Freijer & Post inverse-Gaussian absorption -- NONMEM anchor for ferx FR1*igd()+FR2*igd() [#388]
; Reference fit for ferx's pathway-fraction mechanism: two inverse-Gaussian
; absorption pathways feeding a 1-cpt central compartment, split by a dose
; fraction FR1 / (1-FR1) -- the Freijer & Post (1997) BIPHASIC form. Mirrors
; examples/biphasic_igd_absorption.ferx and nonmem_anchor/freijer_ig.ctl.
;
; The dataset biphasic_ig_oral.csv is simulated FROM this biphasic model
; (nonmem_anchor/simulate_biphasic_ig_data.py, same truths), so unlike the
; single-IG anchor this is a matched (well-specified) fit: ferx and NONMEM should
; agree on OFV/THETA AND both recover the data-generating values.
;
; Identifiability: the two pathways are exchangeable (pathway 1 <-> 2 with
; FR1 <-> 1-FR1). The MAT1 < MAT2 bounds below (fast vs slow) break that label
; symmetry; keep the same convention in the ferx fit when comparing.

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA biphasic_ig_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); IG sum feeds central
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL    = THETA(1)*EXP(ETA(1))
  V     = THETA(2)*EXP(ETA(2))
  FR1   = THETA(3)         ; fraction of dose through the fast pathway
  MAT1  = THETA(4)         ; fast-pathway mean absorption time (h)
  MAT2  = THETA(5)         ; slow-pathway mean absorption time (h)
  CV21 = THETA(6)         ; fast-pathway relative dispersion (Var/mean^2) = ferx CV2_1
  CV22 = THETA(7)         ; slow-pathway relative dispersion             = ferx CV2_2

  FR2 = 1.0 - FR1
  K20 = CL/V
  PI  = 3.14159265358979312

  ; PODO (last oral dose amount) and TDOS (its time) are NOT reserved variables --
  ; capture them at the dose record; NONMEM carries them forward for $DES to read.
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0: no bolus; PODO drives the two IG inputs in $DES (ferx igd() convention --
  ; each pathway's integral R_in = FRi*dose, so the total delivered is the dose).
  F1  = 0.0

$DES
  ; Time after the most recent dose (TDOS captured in $PK); = T here (dose at t=0).
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; Each pathway: IGi = PODO * sqrt(MATi/(2*pi*CV2i*tad^3)) * exp(-(tad-MATi)^2/(2*CV2i*MATi*tad))
  AG1 = -(TAD - MAT1)**2 / (2.0*CV21*MAT1*TAD)
  IG1 = PODO*SQRT(MAT1/(2.0*PI*CV21*TAD**3))*EXP(AG1)
  AG2 = -(TAD - MAT2)**2 / (2.0*CV22*MAT2*TAD)
  IG2 = PODO*SQRT(MAT2/(2.0*PI*CV22*TAD**3))*EXP(AG2)
  RIN = FR1*IG1 + FR2*IG2        ; biphasic input rate (dose split FR1 / 1-FR1)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; biphasic IG input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,   5.0,  100)   ; 1 CL    (L/h)
  (5.0,   50.0, 500)   ; 2 V     (L)
  (0.001, 0.6,  0.999) ; 3 FR1         fast-pathway fraction
  (0.05,  0.5,  2.0)   ; 4 MAT1  (h)   fast pathway (bounded < MAT2 to break label symmetry)
  (2.0,   4.0,  24)    ; 5 MAT2  (h)   slow pathway
  (0.001, 0.2,  10)    ; 6 CV21       fast-pathway dispersion
  (0.001, 0.5,  10)    ; 7 CV22       slow-pathway dispersion

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=biphasic_ig.tab
