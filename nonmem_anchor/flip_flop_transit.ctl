$PROBLEM Flip-flop 1-cpt transit -- NONMEM anchor for ferx one_cpt_transit ODE reroute
; ke = CL/V = 2/4 = 0.5 >= KTR = (N+1)/MTT = 4/20 = 0.2 (flip-flop regime).
; transit() R_in delivered directly into central (no separate KA) -- the exact ODE
; ferx's flip-flop reroute integrates: d/dt(central) = transit(n,mtt) - (CL/V)*central,
; conc = A(central)/V. MAXEVAL=0 + zero OMEGA/SIGMA -> PRED is the typical profile.
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA flip_flop_transit.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(CENTRAL,DEFOBS,DEFDOSE)
$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)
  MTT = THETA(3)
  NN  = THETA(4)
  KTR = (NN + 1.0)/MTT
  K20 = CL/V
  ; ln Gamma(NN+1) via Lanczos g=7 (byte-identical to ferx ln_gamma / savic_transit.ctl)
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
  LNG = 0.91893853320467274 + (XX + 0.5)*LOG(TG) - TG + LOG(AA)
  ; capture oral dose amount/time; F1=0 suppresses the bolus, PODO drives R_in
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  F1  = 0.0
$DES
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  LNR = LOG(PODO) + LOG(KTR) + NN*LOG(KTR*TAD) - KTR*TAD - LNG
  RIN = EXP(LNR)
  DADT(1) = RIN - K20*A(1)
$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1.0 + EPS(1))
$THETA
  2.0  FIX   ; CL   -> ke = CL/V = 0.5
  4.0  FIX   ; V
  20.0 FIX   ; MTT  -> KTR = 4/20 = 0.2  (flip-flop: ke 0.5 >= KTR 0.2)
  3.0  FIX   ; N
$OMEGA 0 FIX
$SIGMA 0 FIX
$SIMULATION (20260707) ONLYSIMULATION SUBPROBLEMS=1
$TABLE ID TIME IPRED Y NOPRINT NOAPPEND ONEHEADER FILE=ff.tab
