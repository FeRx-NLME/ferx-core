$PROBLEM Joint PK-RTTE, clock-forward, drug-driven ODE hazard -- anchor for #3152
; A repeated-event hazard that reads the PK, integrated as a $DES state A(3). Clock forward
; (Andersen-Gill): each record scores the gap since the previous one as A(3)(t_k) - A(3)(t_k-1),
; carried in OLDCHZ across a subject's records. ferx twin: pkrtte_fwd_fit.ferx, compared in
; tests/pkrtte_fwd_nonmem_anchor.rs on the per-subject OBJ (.phi).
;
; Data (pkrtte_fwd.csv): 8 subjects, oral doses of 50, 100 or 200 at 0/12/24, censored at 36.
; Events sit near peaks, near troughs, and on gaps that span a dose (10.5 -> 26, 3 -> 14), so
; the exposure moves inside each gap; subject 6 has no event.
;
; MAXEVAL=0 POSTHOC, every THETA FIX, OMEGA 0 FIX: ferx and NONMEM at the same point (eta = 0).

$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA pkrtte_fwd.csv IGNORE=@

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

$DES
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - CL/V*A(2)
  DADT(3) =  H0*EXP(BETA*A(2)/V)

$ERROR
  IF (NEWIND.NE.2) OLDCHZ = 0
  CHZ = A(3) - OLDCHZ
  IF (CMT.EQ.3) OLDCHZ = A(3)
  HAZ = H0*EXP(BETA*A(2)/V)
  Y = EXP(-CHZ)
  IF (DV.EQ.1) Y = EXP(-CHZ)*HAZ

$THETA
  2.00 FIX   ; 1 CL   (L/h)
  10.0 FIX   ; 2 V    (L)
  1.00 FIX   ; 3 KA   (1/h)
  0.03 FIX   ; 4 H0   baseline hazard (1/h)
  0.20 FIX   ; 5 BETA concentration effect on log-hazard (L/mg)

$OMEGA
  0 FIX      ; IIV CL, zeroed so ferx and NONMEM are compared at eta = 0

$ESTIMATION METHOD=COND LAPLACE LIKELIHOOD MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME CMT DV CHZ HAZ MDV NOPRINT ONEHEADER NOAPPEND
       FORMAT=s1PE15.8 FILE=pkrtte_fwd.tab
