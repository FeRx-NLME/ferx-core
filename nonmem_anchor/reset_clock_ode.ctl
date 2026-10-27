$PROBLEM #1978 reset clock: what $DES T reads after an EVID=4 restart (written)
; Two periods per subject; period 2 opens with an EVID=4 dose at TIME 0. Elimination
; rises linearly in T, read in $DES: k = CL/V*(1 + SLP*T). A smooth term, not a step: a
; step at a dose time puts a discontinuity on an integration boundary, which ADVAN6 rejects
; (`NUMERICAL DIFFICULTIES WITH INTEGRATION ROUTINE` at the dose at 24). Compartment 2 (PROBE) integrates T itself, so its
; amount at a record is the integral of T since the last reset: t^2/2 if T restarts with
; TIME, ((t+73)^2 - 73^2)/2 if it keeps running past period 1 (73 = ferx's shift).
; ADVAN6, not ADVAN13: in the image these were measured with, ADVAN13 aborts with
; `ERROR IN LSODA: CODE -3` on the first record whenever an ETA is live (the committed
; evid_reset_lag_advan13.ctl does too once its $OMEGA is unfixed). T in $DES is passed
; by PREDPP, not by the solver; ADVAN14 (CVODES) gives the same OBJ to <= 2e-5 on every arm.
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA reset_clock_ode.csv IGNORE=@
$SUBROUTINE ADVAN6 TOL=10
$MODEL COMP=(CENTRAL,DEFDOSE) COMP=(PROBE)
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  SLP = THETA(3)
$DES
  DADT(1) = -CL/V*(1 + SLP*T)*A(1)
  DADT(2) = T
$ERROR
  IPRED = A(1)/V
  PROBE = A(2)
  Y = IPRED*(1 + EPS(1)) + EPS(2)
$THETA
  (0.01, 2.0, 100)   ; TVCL
  (1.0, 20.0, 500)   ; TVV
  (-0.1, 0.02, 1)    ; SLP
$OMEGA 0.09
$OMEGA 0.04
$SIGMA 0.01
$SIGMA 0.0001 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME EVID IPRED PROBE ETA1 ETA2 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16
       FILE=reset_clock_ode.tab
