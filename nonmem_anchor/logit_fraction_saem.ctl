$PROBLEM Logit-normal pathway fraction, SAEM with MU_ referencing -- anchor for ferx logit mu-referencing [#918]
; Reference SAEM fit for ferx's logit-normal mu-referencing. Parallel dual
; first-order absorption (fast KA1 / slow KA2) into a 1-cpt central compartment,
; split by a dose fraction FR1 that carries IIV *on the logit scale*:
;
;     FR1_i = 1 / (1 + EXP(-(MU_3 + ETA(3))))  ,  MU_3 = THETA(3)
;
; This is the NONMEM-canonical logit mu-reference: THETA(3) is declared on the
; logit scale, so MU_3 = THETA(3) with no link function, and the SAEM M-step
; updates it by the closed-form mean-eta shift. ferx detects the same structure
; automatically from `FR1 = 1/(1+exp(-(LOGIT_FR1 + ETA_FR1)))` (issue #918);
; before that fix it warned "individual parameter(s) not mu-referenced: FR1" and
; sent THETA(3) through the numeric M-step instead.
;
; The dataset logit_fraction_oral.csv is simulated FROM this model
; (nonmem_anchor/simulate_logit_fraction_data.py, same truths), so this is a
; matched (well-specified) fit: NONMEM and ferx should agree AND both recover the
; data-generating values.
;
; Truths: CL=5, V=50, FR1=0.6 (logit 0.405465), KA1=2.0, KA2=0.2,
;         omega^2 = 0.09 (CL), 0.09 (V), 0.25 (logit FR1), sigma^2 = 0.0064.
;
; Identifiability: the two pathways are exchangeable (pathway 1 <-> 2 with
; FR1 <-> 1-FR1, KA1 <-> KA2). The KA1 > KA2 bounds below (fast vs slow) break
; that label symmetry; keep the same convention in the ferx fit when comparing.

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA logit_fraction_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); first-order sum feeds central
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  MU_1 = LOG(THETA(1))
  MU_2 = LOG(THETA(2))
  MU_3 = THETA(3)          ; already on the logit scale -- identity link

  CL  = EXP(MU_1 + ETA(1))
  V   = EXP(MU_2 + ETA(2))
  FR1 = 1.0/(1.0 + EXP(-(MU_3 + ETA(3))))   ; logit-normal fraction in (0,1)
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
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  FO1 = FR1*PODO*KA1*EXP(-KA1*TAD)
  FO2 = FR2*PODO*KA2*EXP(-KA2*TAD)
  RIN = FO1 + FO2                ; parallel input rate (dose split FR1 / 1-FR1)
  DADT(1) = 0.0                  ; depot is an inert dose carrier
  DADT(2) = RIN - K20*A(2)       ; dual first-order input straight into central

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,   5.0,  100)    ; 1 CL    (L/h)
  (5.0,   50.0, 500)    ; 2 V     (L)
  (-10.0, -0.405465, 10.0) ; 3 LOGIT_FR1   logit of the fast-pathway fraction (start FR1 = 0.4)
  (0.5,   2.0,  24)     ; 4 KA1   (1/h) fast pathway (bounded > KA2 to break label symmetry)
  (0.01,  0.2,  0.5)    ; 5 KA2   (1/h) slow pathway

$OMEGA
  0.09    ; IIV CL
  0.09    ; IIV V
  0.25    ; IIV logit FR1

$SIGMA
  0.0064  ; proportional residual variance (0.08^2)

$ESTIMATION METHOD=SAEM INTERACTION NBURN=2000 NITER=1000 ISAMPLE=10 PRINT=200 SEED=918 NOABORT
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=10 ISAMPLE=3000 PRINT=1 SEED=918 NOABORT
; No $COVARIANCE: this anchor compares point estimates, not standard errors, and
; printing the R matrix aborts this NONMEM 7.5.1 build on the SAEM+IMP chain.
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=logit_fraction_saem.tab
