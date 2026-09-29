$PROBLEM one_cpt_transit + IOV on CL and MTT, per-dose kernel (fixed at dose) -- ferx #1569 anchor
; Reference for ferx's transit ODE twin under IOV on an ABSORPTION parameter, with doses still
; absorbing when their occasion's successor begins. Each dose keeps the MTT of its own dose
; record; the disposition (K10) is the current record's -- NONMEM end-of-interval, the same
; convention as ferx (#1073). $DES superposes all four doses' Savic densities.
;
; Holding a per-dose MTT: NM-TRAN zeroes an ETA-dependent variable whose IF fails (WARNING 3)
; and rejects one set in a nested IF (error 326), so a kernel cannot simply be carried from the
; dose record. What IS carried is the dose's OCCASION (data, not random): ODn below. Every
; record then rebuilds dose n's MTT from that occasion's ETA through Lagrange weights in
; ODn = 1..3 -- no IF at all, so the ETA dependence (and its derivatives) is defined everywhere.
;
; transit_iov.ctl reads the CURRENT MTT/NN for the one dose it tracks (PODO/TDOS). That is the
; rule #1569 removes: an MTT switch mid-absorption splices two densities and does not conserve
; the dose's mass. With IOV on CL only (that anchor's design) the two rules coincide.
;
; Data (transit_iov_mtt.csv, simulate_absorption_iov_carryover_data.py): 24 subjects, doses of
; 100 at 0/6/12/18 h. Occasion 2 begins at the OBSERVATION record t = 8.5 while dose 2 is still
; ~85 % unabsorbed; occasion 3 begins at the DOSE record t = 18 while dose 3 is still ~15 %
; unabsorbed. FOCEI estimate from the simulation truth; ferx is evaluated at NONMEM's optimum
; on the same csv (tests/transit_iov_mtt_nonmem_anchor.rs).

$INPUT ID TIME DV EVID AMT CMT MDV OCC
$DATA transit_iov_mtt.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=12
$MODEL
  COMP=(CENTRAL,DEFDOSE,DEFOBS)   ; 1 = central (dose lands here; bolus suppressed by F1=0)

$PK
  TVCL  = THETA(1)
  TVV   = THETA(2)
  TVMTT = THETA(3)
  TVN   = THETA(4)

  ; ---- IOV on CL (current record) ----
  OCC1 = 0
  OCC2 = 0
  OCC3 = 0
  IF (OCC.EQ.1) OCC1 = 1
  IF (OCC.EQ.2) OCC2 = 1
  IF (OCC.EQ.3) OCC3 = 1
  IOVCL = OCC1*ETA(3) + OCC2*ETA(4) + OCC3*ETA(5)
  CL  = TVCL*EXP(ETA(1) + IOVCL)
  V   = TVV *EXP(ETA(2))
  K10 = CL/V

  ; ---- each occasion's MTT (IOV on MTT: ETA(6..8)) ----
  MTO1 = TVMTT*EXP(ETA(6))
  MTO2 = TVMTT*EXP(ETA(7))
  MTO3 = TVMTT*EXP(ETA(8))
  NN   = TVN

  ; ---- ln Gamma(NN+1) via Lanczos g=7, n=9 -------------------------------------
  ; Byte-for-byte the coefficients in ferx src/stats/special.rs::ln_gamma.
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

  ; ---- per dose: its time, amount and occasion, captured at its dose record ----------
  IF (NEWIND.NE.2) THEN
    NDS = 0
    OD1 = 1
    OD2 = 1
    OD3 = 1
    OD4 = 1
  ENDIF
  NEWD = 0
  IF (AMT.GT.0.0.AND.CMT.EQ.1) NEWD = 1
  NDS = NDS + NEWD
  IF (NEWD.EQ.1.AND.NDS.EQ.1) THEN
    TD1 = TIME
    AM1 = AMT
    OD1 = OCC
  ENDIF
  IF (NEWD.EQ.1.AND.NDS.EQ.2) THEN
    TD2 = TIME
    AM2 = AMT
    OD2 = OCC
  ENDIF
  IF (NEWD.EQ.1.AND.NDS.EQ.3) THEN
    TD3 = TIME
    AM3 = AMT
    OD3 = OCC
  ENDIF
  IF (NEWD.EQ.1.AND.NDS.EQ.4) THEN
    TD4 = TIME
    AM4 = AMT
    OD4 = OCC
  ENDIF
  ; ---- each dose's kernel from ITS occasion (Lagrange weights: 1 at ODn, 0 at the others)
  MT1 = MTO1*(2-OD1)*(3-OD1)/2 + MTO2*(OD1-1)*(3-OD1) + MTO3*(OD1-1)*(OD1-2)/2
  MT2 = MTO1*(2-OD2)*(3-OD2)/2 + MTO2*(OD2-1)*(3-OD2) + MTO3*(OD2-1)*(OD2-2)/2
  MT3 = MTO1*(2-OD3)*(3-OD3)/2 + MTO2*(OD3-1)*(3-OD3) + MTO3*(OD3-1)*(OD3-2)/2
  MT4 = MTO1*(2-OD4)*(3-OD4)/2 + MTO2*(OD4-1)*(3-OD4) + MTO3*(OD4-1)*(OD4-2)/2
  KT1 = (NN + 1.0)/MT1
  KT2 = (NN + 1.0)/MT2
  KT3 = (NN + 1.0)/MT3
  KT4 = (NN + 1.0)/MT4
  F1  = 0.0

$DES
  ; R_in = sum over arrived doses of AMn * KTn*(KTn*tad)^NN * exp(-KTn*tad) / Gamma(NN+1)
  RIN = 0.0
  TA1 = T - TD1
  TA2 = T - TD2
  TA3 = T - TD3
  TA4 = T - TD4
  IF (NDS.GE.1.AND.TA1.GT.0.0) RIN = RIN + EXP(LOG(AM1*KT1) + NN*LOG(KT1*TA1) - KT1*TA1 - LNG)
  IF (NDS.GE.2.AND.TA2.GT.0.0) RIN = RIN + EXP(LOG(AM2*KT2) + NN*LOG(KT2*TA2) - KT2*TA2 - LNG)
  IF (NDS.GE.3.AND.TA3.GT.0.0) RIN = RIN + EXP(LOG(AM3*KT3) + NN*LOG(KT3*TA3) - KT3*TA3 - LNG)
  IF (NDS.GE.4.AND.TA4.GT.0.0) RIN = RIN + EXP(LOG(AM4*KT4) + NN*LOG(KT4*TA4) - KT4*TA4 - LNG)
  DADT(1) = RIN - K10*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1,  5.0,  100)   ; 1 TVCL
  (1.0,  30.0, 500)   ; 2 TVV
  (0.05, 3.0,  24)    ; 3 TVMTT
  (0.1,  3.0,  30)    ; 4 TVN

$OMEGA
  0.09    ; IIV CL (ETA1)
  0.04    ; IIV V  (ETA2)

$OMEGA BLOCK(1) 0.04   ; IOV CL (ETA3), occasion 1
$OMEGA BLOCK(1) SAME   ;        (ETA4), occasion 2
$OMEGA BLOCK(1) SAME   ;        (ETA5), occasion 3
$OMEGA BLOCK(1) 0.09   ; IOV MTT (ETA6), occasion 1
$OMEGA BLOCK(1) SAME   ;         (ETA7), occasion 2
$OMEGA BLOCK(1) SAME   ;         (ETA8), occasion 3

$SIGMA
  0.01    ; proportional residual variance (0.1^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME DV IPRED PRED MDV OCC NOPRINT ONEHEADER FORMAT=s1PE21.13 FILE=transit_iov_mtt.tab
