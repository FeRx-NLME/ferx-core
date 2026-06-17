$PROBLEM Savic 2007 transit-compartment absorption -- NONMEM anchor for ferx transit() [#343]
; Reference fit for ferx's built-in transit(n, mtt) input-rate function.
; One-compartment oral disposition; Savic continuous-N transit absorption.
; Fit this, then fit examples/transit_savic.ferx in ferx on the SAME data and
; compare OFV + THETA estimates (see README.md).

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA transit_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = depot   (oral dose lands here; bolus suppressed by F1=0)
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  TVCL  = THETA(1)
  TVV   = THETA(2)
  TVKA  = THETA(3)
  TVMTT = THETA(4)
  TVN   = THETA(5)

  CL  = TVCL*EXP(ETA(1))
  V   = TVV *EXP(ETA(2))
  KA  = TVKA
  MTT = TVMTT
  NN  = TVN

  KTR = (NN + 1.0)/MTT
  K20 = CL/V

  ; ---- ln Gamma(NN+1) via Lanczos g=7, n=9 --------------------------------
  ; Byte-for-byte the coefficients in ferx src/stats/special.rs::ln_gamma, so
  ; the gamma special-function is NOT a source of ferx-vs-NONMEM discrepancy.
  ; ferx evaluates with x = NN+1, x' = x-1 = NN; reflection branch (x<0.5) is
  ; never reached here because NN >= 0.
  XX  = NN
  AA  = 0.99999999999980993
  AA  = AA + 676.5203681218851      / (XX + 1.0)
  AA  = AA - 1259.1392167224028     / (XX + 2.0)
  AA  = AA + 771.32342877765313     / (XX + 3.0)
  AA  = AA - 176.61502916214059     / (XX + 4.0)
  AA  = AA + 12.507343278686905     / (XX + 5.0)
  AA  = AA - 0.13857109526572012    / (XX + 6.0)
  AA  = AA + 9.9843695780195716E-06 / (XX + 7.0)
  AA  = AA + 1.5056327351493116E-07 / (XX + 8.0)
  TG  = XX + 7.5
  LNG = 0.91893853320467274 + (XX + 0.5)*LOG(TG) - TG + LOG(AA)   ; = ln Gamma(NN+1)

  ; ---- dose routing: feed transit(), not a bolus --------------------------
  ; PODO (last oral dose amount) and TDOS (its time) are NOT reserved variables
  ; in NONMEM -- capture them at the dose record. NONMEM carries a variable's
  ; value forward across records, so $DES reads them on later observation rows.
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  ; F1=0 suppresses the depot bolus; PODO drives R_in in $DES. The NONMEM analogue
  ; of ferx delivering the dose as R_in and suppressing the instantaneous bolus
  ; (no double counting; integral R_in = dose). With bioavailability, use BIO*PODO.
  F1  = 0.0

$DES
  ; Time after the most recent dose: T is the absolute integration time and TDOS
  ; the captured dose time, so TAD = T - TDOS (= T here, the dose being at t=0).
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; R_in(tad) = PODO * KTR*(KTR*tad)^NN * exp(-KTR*tad) / Gamma(NN+1), log-domain.
  LNR = LOG(PODO) + LOG(KTR) + NN*LOG(KTR*TAD) - KTR*TAD - LNG
  RIN = EXP(LNR)
  DADT(1) = RIN - KA*A(1)
  DADT(2) = KA*A(1) - K20*A(2)

$ERROR
  IPRED = A(2)/V                 ; central concentration (mg/L)
  Y = IPRED*(1.0 + EPS(1))       ; proportional residual error

$THETA
  (0.1,  5.0,  100)   ; 1 TVCL  (L/h)
  (5.0,  50.0, 500)   ; 2 TVV   (L)
  (0.05, 1.0,  24)    ; 3 TVKA  (1/h)
  (0.05, 1.0,  24)    ; 4 TVMTT (h)   KTR=(N+1)/MTT
  (0.1,  3.0,  30)    ; 5 TVN   transit compartments (continuous)

$OMEGA
  0.09    ; IIV CL  (~30% CV)
  0.09    ; IIV V   (~30% CV)

$SIGMA
  0.0225  ; proportional residual variance (0.15^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=savic.tab
