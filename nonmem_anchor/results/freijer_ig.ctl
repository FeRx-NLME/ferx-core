$PROBLEM Freijer & Post inverse-Gaussian absorption -- NONMEM anchor for ferx igd() [#347]
; Reference fit for ferx's built-in igd(mat, cv2) input-rate function.
; Inverse-Gaussian density absorption straight into a 1-cpt central compartment.
;
; NOTE on the data: transit_oral.csv was simulated from a TRANSIT model, so this
; IG fit is mildly mis-specified (IG approximates the delayed-absorption shape).
; That is still a valid IMPLEMENTATION anchor: ferx-igd and NONMEM-igd should
; return the same OFV/THETA on identical data even when neither is the true DGP.
; For a matched IG-truth dataset, see README.md (regenerate from an IG model).

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA transit_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); IG feeds central directly
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)*EXP(ETA(2))
  MAT = THETA(3)           ; mean absorption time (h)
  CV2 = THETA(4)           ; relative dispersion (Var/mean^2) of the absorption time

  K20 = CL/V
  PI  = 3.14159265358979312

  ; PODO (last oral dose amount) and TDOS (its time) are NOT reserved variables --
  ; capture them at the dose record; NONMEM carries them forward for $DES to read.
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0: no bolus; PODO drives the IG input in $DES (ferx igd() convention --
  ; integral R_in = dose, fed straight into central). With bioavailability, BIO*PODO.
  F1  = 0.0

$DES
  ; Time after the most recent dose (TDOS captured in $PK); = T here (dose at t=0).
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; R_in(tad) = PODO * sqrt(MAT/(2*pi*CV2*tad^3)) * exp(-(tad-MAT)^2/(2*CV2*MAT*tad))
  ARG = -(TAD - MAT)**2 / (2.0*CV2*MAT*TAD)
  RIN = PODO*SQRT(MAT/(2.0*PI*CV2*TAD**3))*EXP(ARG)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; IG input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,  5.0,  100)   ; 1 CL   (L/h)
  (5.0,  50.0, 500)   ; 2 V    (L)
  (0.05, 2.0,  24)    ; 3 MAT  (h)   mean absorption time
  (0.001,0.3,  10)    ; 4 CV2        relative dispersion

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=igd.tab
