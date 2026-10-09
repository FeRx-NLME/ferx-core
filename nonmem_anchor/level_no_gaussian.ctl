$PROBLEM Level block read by a TTE-only subject -- anchor for #1797
; A `theta PLACEBO[STUDY, contrast = ref]` block read by the hazard of a joint PK-TTE
; model, on data with two subjects that carry NO Gaussian observation: 98 (STUDY=1) and
; 99 (STUDY=3), one exact event each at t = 7. Before #1797 ferx indexed every such
; subject at level 1, so 99's hazard read STUDY=1's theta.
;
; NONMEM has no level-index object: $PK reads the record's own STUDY, so its value IS the
; closed form `H0 = TVH0 * exp(PLACEBO[own STUDY])`. That is the reference here.
;
; Compared object: per-subject OBJ from the .phi vs 2 x ferx `individual_nll`. $OMEGA 0
; FIX and no omega on the ferx side, so both sides are evaluated at eta = 0 and ferx's
; individual_nll carries no 1/2 ln(omega) constant. The ferx<->NONMEM per-subject constant
; for a proportional-error PK record and an exact TTE event was measured to be zero on
; pktte_tdep (#1166).
;
; Non-degeneracy: 98 and 99 straddle the fix -- 98 reads level 1 either way (the
; control), 99 reads level 3 only after it. Their hazards differ by e^1.
;
; MAXEVAL=0 POSTHOC, every THETA FIX at the values ferx evaluates.

$INPUT ID TIME DV EVID AMT CMT MDV STUDY
$DATA level_no_gaussian.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(CENTRAL,DEFDOSE)
  COMP=(CHZ)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)
  ; contrast = ref: STUDY=1 is the reference level, fixed at 0.
  PLC = 0
  IF (STUDY.EQ.2) PLC = THETA(4)
  IF (STUDY.EQ.3) PLC = THETA(5)
  H0  = THETA(3)*EXP(PLC)
  K10 = CL/V

$DES
  DADT(1) = -K10*A(1)
  DADT(2) =  H0

$ERROR
  IPRED = A(1)/V
  CHZ   = A(2)
  SUR   = EXP(-CHZ)
  F_FLAG = 0
  IF (CMT.EQ.2) F_FLAG = 1
  Y = IPRED*(1.0 + EPS(1))
  IF (CMT.EQ.2.AND.DV.EQ.1) Y = SUR*H0
  IF (CMT.EQ.2.AND.DV.EQ.0) Y = SUR

$THETA
  1.00 FIX   ; 1 TVCL (L/h)
  20.0 FIX   ; 2 TVV  (L)
  0.05 FIX   ; 3 TVH0 baseline hazard
  0.50 FIX   ; 4 PLACEBO[STUDY=2]
  1.00 FIX   ; 5 PLACEBO[STUDY=3]

$OMEGA
  0 FIX      ; IIV CL -- zeroed so both sides are compared at eta = 0

$SIGMA
  0.01       ; proportional residual variance (0.10^2)

$ESTIMATION METHOD=1 LAPLACE INTER NUMERICAL SLOW MAXEVAL=0 POSTHOC PRINT=1 NOABORT
            FORMAT=s1PE23.16
$TABLE ID TIME CMT DV IPRED CHZ H0 STUDY MDV NOPRINT ONEHEADER NOAPPEND
       FORMAT=s1PE15.8 FILE=level_no_gaussian.tab
