$PROBLEM one_cpt_ig + IOV on CL and MAT, per-dose kernel (fixed at dose) -- ferx #1569 anchor
; The inverse-Gaussian counterpart of transit_iov_mtt.ctl: ferx's igd() ODE twin under IOV on
; an ABSORPTION parameter (MAT), with doses still absorbing when their occasion's successor
; begins. Each dose keeps the MAT of its own dose record; the disposition (K10) is the current
; record's (NONMEM end-of-interval, as ferx #1073). $DES superposes all four doses' Freijer &
; Post densities.
;
; The per-dose MAT is rebuilt at every record from the dose's captured OCCASION (ODn) through
; Lagrange weights in ODn = 1..3 -- see transit_iov_mtt.ctl for why an ETA-dependent kernel
; cannot simply be carried from the dose record.
;
; Data (ig_iov_mat.csv, simulate_absorption_iov_carryover_data.py): the transit_iov_mtt design
; -- 24 subjects, doses of 100 at 0/6/12/18 h, occasion 2 from the OBSERVATION record t = 8.5,
; occasion 3 from the DOSE record t = 18 -- simulated from this model. FOCEI estimate from the
; simulation truth; ferx is evaluated at NONMEM's optimum (tests/ig_iov_mat_nonmem_anchor.rs).

$INPUT ID TIME DV EVID AMT CMT MDV OCC
$DATA ig_iov_mat.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=12
$MODEL
  COMP=(CENTRAL,DEFDOSE,DEFOBS)   ; 1 = central (dose lands here; bolus suppressed by F1=0)

$PK
  TVCL  = THETA(1)
  TVV   = THETA(2)
  TVMAT = THETA(3)
  TVCV2 = THETA(4)

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

  ; ---- each occasion's MAT (IOV on MAT: ETA(6..8)) ----
  MAO1 = TVMAT*EXP(ETA(6))
  MAO2 = TVMAT*EXP(ETA(7))
  MAO3 = TVMAT*EXP(ETA(8))
  CV2  = TVCV2
  PI   = 3.14159265358979312

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
  ; ---- each dose's MAT from ITS occasion (Lagrange weights: 1 at ODn, 0 at the others)
  MA1 = MAO1*(2-OD1)*(3-OD1)/2 + MAO2*(OD1-1)*(3-OD1) + MAO3*(OD1-1)*(OD1-2)/2
  MA2 = MAO1*(2-OD2)*(3-OD2)/2 + MAO2*(OD2-1)*(3-OD2) + MAO3*(OD2-1)*(OD2-2)/2
  MA3 = MAO1*(2-OD3)*(3-OD3)/2 + MAO2*(OD3-1)*(3-OD3) + MAO3*(OD3-1)*(OD3-2)/2
  MA4 = MAO1*(2-OD4)*(3-OD4)/2 + MAO2*(OD4-1)*(3-OD4) + MAO3*(OD4-1)*(OD4-2)/2
  F1  = 0.0

$DES
  ; R_in = sum over arrived doses of AMn*sqrt(MAn/(2 pi CV2 tad^3))*exp(-(tad-MAn)^2/(2 CV2 MAn tad))
  RIN = 0.0
  TA1 = T - TD1
  TA2 = T - TD2
  TA3 = T - TD3
  TA4 = T - TD4
  IF (NDS.GE.1.AND.TA1.GT.0.0) RIN = RIN + AM1*SQRT(MA1/(2.0*PI*CV2*TA1**3))*EXP(-(TA1-MA1)**2/(2.0*CV2*MA1*TA1))
  IF (NDS.GE.2.AND.TA2.GT.0.0) RIN = RIN + AM2*SQRT(MA2/(2.0*PI*CV2*TA2**3))*EXP(-(TA2-MA2)**2/(2.0*CV2*MA2*TA2))
  IF (NDS.GE.3.AND.TA3.GT.0.0) RIN = RIN + AM3*SQRT(MA3/(2.0*PI*CV2*TA3**3))*EXP(-(TA3-MA3)**2/(2.0*CV2*MA3*TA3))
  IF (NDS.GE.4.AND.TA4.GT.0.0) RIN = RIN + AM4*SQRT(MA4/(2.0*PI*CV2*TA4**3))*EXP(-(TA4-MA4)**2/(2.0*CV2*MA4*TA4))
  DADT(1) = RIN - K10*A(1)

$ERROR
  IPRED = A(1)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1,  5.0,  100)   ; 1 TVCL
  (1.0,  30.0, 500)   ; 2 TVV
  (0.05, 3.0,  24)    ; 3 TVMAT
  (0.01, 0.5,  5)     ; 4 TVCV2

$OMEGA
  0.09    ; IIV CL (ETA1)
  0.04    ; IIV V  (ETA2)

$OMEGA BLOCK(1) 0.04   ; IOV CL (ETA3), occasion 1
$OMEGA BLOCK(1) SAME   ;        (ETA4), occasion 2
$OMEGA BLOCK(1) SAME   ;        (ETA5), occasion 3
$OMEGA BLOCK(1) 0.09   ; IOV MAT (ETA6), occasion 1
$OMEGA BLOCK(1) SAME   ;         (ETA7), occasion 2
$OMEGA BLOCK(1) SAME   ;         (ETA8), occasion 3

$SIGMA
  0.01    ; proportional residual variance (0.1^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME DV IPRED PRED MDV OCC NOPRINT ONEHEADER FORMAT=s1PE21.13 FILE=ig_iov_mat.tab
