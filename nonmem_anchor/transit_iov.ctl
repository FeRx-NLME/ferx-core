$PROBLEM one_cpt_transit + IOV on CL -- NONMEM anchor for ferx analytic transit under IOV [#719]
; Reference fit for ferx's analytic `pk one_cpt_transit(cl,v,n,mtt)` under inter-occasion
; variability (IOV on CL). `one_cpt_transit` is transit-density-directly-into-central (a single
; disposition compartment; its ferx ODE twin is `d/dt(central) = transit(n,mtt) - (CL/V)*central`),
; so this is ONE compartment: F1=0 suppresses the depot bolus and the Savic density is delivered
; as R_in in $DES. IOV on CL is one ETA per occasion (OCC 1/2/3), sharing one IOV variance via
; $OMEGA BLOCK SAME. Data (nonmem_anchor/transit_iov.csv) is one dose per occasion, three
; occasions 48 h apart with near-complete washout, so the $DES single-dose R_in (PODO/TDOS)
; equals ferx's dose superposition. Fit this, then fit ferx on the SAME csv and compare OFV +
; THETA/OMEGA (generator: tests/gen_transit_iov_anchor.rs prints the ferx side).

$INPUT ID TIME DV EVID AMT CMT MDV OCC
$DATA transit_iov.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(CENTRAL,DEFDOSE,DEFOBS)   ; 1 = central (dose lands here; bolus suppressed by F1=0)

$PK
  TVCL  = THETA(1)
  TVV   = THETA(2)
  TVMTT = THETA(3)
  TVN   = THETA(4)

  ; ---- IOV on CL: one ETA per occasion, one shared IOV variance (BLOCK SAME) ----
  OCC1 = 0
  OCC2 = 0
  OCC3 = 0
  IF (OCC.EQ.1) OCC1 = 1
  IF (OCC.EQ.2) OCC2 = 1
  IF (OCC.EQ.3) OCC3 = 1
  IOVCL = OCC1*ETA(2) + OCC2*ETA(3) + OCC3*ETA(4)

  CL  = TVCL*EXP(IOVCL)
  V   = TVV *EXP(ETA(1))
  MTT = TVMTT
  NN  = TVN

  KTR = (NN + 1.0)/MTT
  K10 = CL/V

  ; ---- ln Gamma(NN+1) via Lanczos g=7, n=9 -------------------------------------
  ; Byte-for-byte the coefficients in ferx src/stats/special.rs::ln_gamma, so the gamma
  ; special-function is NOT a source of ferx-vs-NONMEM discrepancy. x = NN+1, x' = NN.
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

  ; ---- dose routing: feed transit(), not a bolus -------------------------------
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  F1  = 0.0

$DES
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ; R_in(tad) = PODO * KTR*(KTR*tad)^NN * exp(-KTR*tad) / Gamma(NN+1), log-domain.
  LNR = LOG(PODO) + LOG(KTR) + NN*LOG(KTR*TAD) - KTR*TAD - LNG
  RIN = EXP(LNR)
  DADT(1) = RIN - K10*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1,  9.0,  100)   ; 1 TVCL
  (1.0,  30.0, 500)   ; 2 TVV
  (0.05, 1.0,  24)    ; 3 TVMTT   KTR=(N+1)/MTT
  (0.1,  3.0,  30)    ; 4 TVN     transit compartments (continuous)

$OMEGA
  0.09    ; IIV V (ETA1)

$OMEGA BLOCK(1) 0.04   ; IOV CL (ETA2), occasion 1
$OMEGA BLOCK(1) SAME   ;              occasion 2 (same variance)
$OMEGA BLOCK(1) SAME   ;              occasion 3 (same variance)

$SIGMA
  0.01    ; proportional residual variance (0.1^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV OCC NOPRINT ONEHEADER FILE=transit_iov.tab
