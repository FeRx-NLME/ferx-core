$PROBLEM Joint PK-TTE at the EBEs, live ETA -- external anchor for #1885
; #1885: `predict_survival_at_etas_diag` evaluates each subject's survival curve at its own
; eta-hat. Every other ODE-side TTE anchor runs `$OMEGA 0 FIX`, so none can see whether the
; individual curve reads the subject's eta. Here the IIV on CL is live (0.09) but FIX, and
; every THETA and SIGMA is FIX: MAXEVAL=0 POSTHOC is then the inner problem alone, so the
; ETA1 column is NONMEM's conditional mode at the stated point.
;
; Cloned from pktte_inf.ctl / pktte_tdep.ctl: depot + central + CHZ augmented system,
; h = H0*exp(BETA*C). Oral 100 into the depot at 0/12/24, so the later doses land on
; residual drug. Data from pktte_ebe_gen.py: 8 subjects, 8 PK observations each, one event
; record each (exact or right-censored), EVID=2 rows on 2..48 step 2. CHZ/HAZ on the EVID=2
; rows are the individual curves the ferx predictor is compared with.
;
; FORMAT=s1PE15.8: 9 significant figures (see pktte_inf.ctl for why the default 5 is a floor).

$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA pktte_ebe.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)
  COMP=(CENTRAL)
  COMP=(CHZ)

$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)
  KA   = THETA(3)
  H0   = THETA(4)
  BETA = THETA(5)
  K20  = CL/V

$DES
  CONCD   = A(2)/V
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - K20*A(2)
  DADT(3) =  H0*EXP(BETA*CONCD)

$ERROR
  CONC = A(2)/V
  HAZ  = H0*EXP(BETA*CONC)
  CHZ  = A(3)
  SUR  = EXP(-CHZ)
  IPRED = CONC
  F_FLAG = 0
  IF (CMT.EQ.3) F_FLAG = 1
  Y = IPRED*(1.0 + EPS(1))
  IF (CMT.EQ.3.AND.DV.EQ.1) Y = SUR*HAZ
  IF (CMT.EQ.3.AND.DV.EQ.0) Y = SUR

$THETA
  1.00 FIX   ; 1 CL   (L/h)
  10.0 FIX   ; 2 V    (L)
  1.00 FIX   ; 3 KA   (1/h)
  0.01 FIX   ; 4 H0   baseline hazard
  0.30 FIX   ; 5 BETA concentration effect on log-hazard

$OMEGA
  0.09 FIX   ; IIV CL -- LIVE, so the EBE is non-zero; FIX so only the inner problem runs

$SIGMA
  0.01 FIX   ; proportional residual variance (0.10^2)

$ESTIMATION METHOD=1 LAPLACE INTER NUMERICAL SLOW MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME EVID CMT DV IPRED CHZ HAZ ETA1 MDV NOPRINT ONEHEADER NOAPPEND
       FORMAT=s1PE15.8 FILE=pktte_ebe.tab
