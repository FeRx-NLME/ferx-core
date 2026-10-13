$PROBLEM ferx iov_occasion anchor B -- the occasion an EVID=3 reset row re-seeds A_0 under
; Arm A's crossover with the reset written as an EVID=3 row followed by a separate EVID=1
; dose row at the same time. The occasion is DERIVED from TIME (the data's OCC column is read
; but not used), so both rows at t=12 run $PK under occasion 2. On ferx's side the EVID=3 row
; carries no dose, so an occasion label on it that no dose or observation shares names no
; kappa group, which arm A's EVID=4 row cannot reach.
$INPUT ID TIME DV MDV EVID AMT CMT OCC
$DATA iov_rule_reset_split.csv IGNORE=@

; Closed form, as arm A.
$SUBROUTINE ADVAN1 TRANS2

$PK
  OCCT = 1
  IF (TIME.GE.12) OCCT = 2
  KAPPA = 0
  IF (OCCT.EQ.1) KAPPA = ETA(2)
  IF (OCCT.EQ.2) KAPPA = ETA(3)
  CL   = THETA(1)*EXP(ETA(1) + KAPPA)
  V    = THETA(2)
  BASE = THETA(3)*EXP(KAPPA)
  A_0(1) = BASE
  S1   = V

$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))

$THETA
  (0.01, 5.0, 100.0)    ; TVCL
  (1.0, 50.0, 500.0)    ; TVV
  (0.01, 100.0, 1000.0) ; TVBASE

$OMEGA 0.04             ; ETA_CL
$OMEGA BLOCK(1) 0.09    ; KAPPA_B occasion 1
$OMEGA BLOCK(1) SAME    ; occasion 2
$SIGMA 0.0025           ; PROP, sd 0.05

$ESTIMATION MAXEVAL=0 METHOD=1 INTER PRINT=1 NOABORT
$TABLE ID TIME EVID OCCT IPRED ETA1 ETA2 ETA3 NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=iov_rule_reset_B.tab
