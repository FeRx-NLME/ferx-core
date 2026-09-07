$PROBLEM Joint PK-TTE, TIME-DEPENDENT hazard -- anchor for #1166
; Anchor for the #1166 fix: a Gompertz-in-time x exposure hazard on a FULLY AUTONOMOUS PK
; block. Nothing committed exercises that combination -- `grep 'hazard *=.*TIME'` over
; src/ tests/ examples/ docs/ is empty -- and the identity trick `* (1.0 + 0.0*TIME)` proves
; ROUTING only: a mishandled model time inside the shared solve's segment integration would
; be invisible to it. Here TIME is LIVE in the hazard.
;
; Twin: `pktte_eta0.ctl` in ~/NONMEMruns/pktte-engine-split/ is this stream with GAM = 0.
; The two differ by exp(GAM*t) on the integrand, so the committed GAM=0 CHZ column is the
; one-variable comparator for "the time term is live".
;
; Compared objects:
;   (i)  A(3) at every TTE record vs BOTH ferx arms, with per-arm bounds.
;   (ii) per-subject OBJ from the .phi vs 2 x ferx `individual_nll` on arm A. Sound because
;        the ferx<->NONMEM per-subject constant was measured to be ZERO on the GAM=0 run
;        (2*nll == OBJ, no n*log(2pi)); run ferx at ode_reltol 1e-9 / abstol 1e-11 or the
;        ~1e-3 tolerance noise at ferx's defaults swamps the comparison.
;
; Single dose at t=0 per subject, so T == TIME == TAD == TAFD here: this stream anchors the
; VALUE of a time-reading hazard, not which spelling of time was read. The spellings are
; separated by the parser unit tests, not by this run.
;
; MAXEVAL=0 POSTHOC, every THETA FIX at the values ferx evaluates, OMEGA 0 FIX so ferx and
; NONMEM are compared at the same point (eta = 0).

$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA pktte.csv IGNORE=@

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
  GAM  = THETA(6)
  K20  = CL/V

$DES
  CONCD   = A(2)/V
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - K20*A(2)
  ; Gompertz baseline in T x exposure. The PK block above stays autonomous.
  DADT(3) =  H0*EXP(GAM*T)*EXP(BETA*CONCD)

$ERROR
  CONC = A(2)/V
  ; T is not available in $ERROR; TIME is the same quantity at a data record.
  HAZ  = H0*EXP(GAM*TIME)*EXP(BETA*CONC)
  CHZ  = A(3)
  SUR  = EXP(-CHZ)
  IPRED = CONC
  F_FLAG = 0
  IF (CMT.EQ.3) F_FLAG = 1
  ; Single-level IFs only: NM-TRAN error 326 rejects a random variable assigned inside a
  ; NESTED if-structure, so the CMT/DV test is folded into one condition per branch.
  Y = IPRED*(1.0 + EPS(1))
  IF (CMT.EQ.3.AND.DV.EQ.1) Y = SUR*HAZ
  IF (CMT.EQ.3.AND.DV.EQ.0) Y = SUR

$THETA
  1.00 FIX   ; 1 CL   (L/h)
  10.0 FIX   ; 2 V    (L)
  1.00 FIX   ; 3 KA   (1/h)
  0.02 FIX   ; 4 H0   baseline hazard
  0.50 FIX   ; 5 BETA concentration effect on log-hazard
  0.15 FIX   ; 6 GAM  Gompertz rate -- exp(GAM*t) spans 1.01x to 3.49x over the TTE
             ;        record range (t = 0.07 .. 8.32 h), so the time term is live at
             ;        every record rather than only at the late ones.

$OMEGA
  0 FIX      ; IIV CL -- zeroed so ferx and NONMEM are compared at the SAME point (eta=0)

$SIGMA
  0.01       ; proportional residual variance (0.10^2)

$ESTIMATION METHOD=1 LAPLACE INTER NUMERICAL SLOW MAXEVAL=0 POSTHOC PRINT=1 NOABORT
; FORMAT=s1PE15.8: the 5-s.f. default cost #1187 a false bound, and the arm-A/arm-B
; accuracy spread this anchor has to resolve is ~1e-9 vs ~3e-6.
$TABLE ID TIME CMT DV IPRED CHZ HAZ ETA1 MDV NOPRINT ONEHEADER NOAPPEND
       FORMAT=s1PE15.8 FILE=pktte_tdep.tab
