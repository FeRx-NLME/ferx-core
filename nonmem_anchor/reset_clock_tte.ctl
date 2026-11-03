$PROBLEM #1978 reset clock: a joint PK-TTE hazard reading T/TIME after an EVID=4 restart (written)
; Period 1 is PK only; period 2 opens with an EVID=4 dose at TIME 0 and carries one TTE
; record on CMT 2. The Gompertz term exp(GAM*T) is read by $DES (integrated into the
; cumulative hazard) and exp(GAM*TIME) by $ERROR (the hazard at an exact event).
; ADVAN6, not ADVAN13: in the image these were measured with, ADVAN13 aborts with
; `ERROR IN LSODA: CODE -3` on the first record whenever an ETA is live (the committed
; evid_reset_lag_advan13.ctl does too once its $OMEGA is unfixed). T in $DES is passed
; by PREDPP, not by the solver; ADVAN14 (CVODES) gives the same OBJ to <= 2e-5 on every arm.
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA reset_clock_tte.csv IGNORE=@
$SUBROUTINE ADVAN6 TOL=10
$MODEL COMP=(CENTRAL,DEFDOSE) COMP=(CHZ)
$PK
  CL   = THETA(1)*EXP(ETA(1))
  V    = THETA(2)
  H0   = THETA(3)
  BETA = THETA(4)
  GAM  = THETA(5)
$DES
  DADT(1) = -CL/V*A(1)
  DADT(2) = H0*EXP(GAM*T)*EXP(BETA*A(1)/V)
$ERROR
  CONC = A(1)/V
  HAZ  = H0*EXP(GAM*TIME)*EXP(BETA*CONC)
  CHZ  = A(2)
  SUR  = EXP(-CHZ)
  IPRED = CONC
  F_FLAG = 0
  IF (CMT.EQ.2) F_FLAG = 1
  Y = IPRED*(1 + EPS(1))
  IF (CMT.EQ.2.AND.DV.EQ.1) Y = SUR*HAZ
  IF (CMT.EQ.2.AND.DV.EQ.0) Y = SUR
$THETA
  (0.01, 1.0, 100)   ; TVCL
  (0.1, 10.0, 500)   ; TVV
  (0.0001, 0.02, 10) ; H0
  (-5, 0.15, 5)      ; BETA
  (-1, 0.05, 1)      ; GAM
$OMEGA 0.09
$SIGMA 0.01
$ESTIMATION METHOD=1 LAPLACE INTER MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME CMT EVID DV IPRED CHZ HAZ ETA1 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16
       FILE=reset_clock_tte.tab
