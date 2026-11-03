$PROBLEM ferx iov_occasion anchor A -- the occasion an EVID=4 reset row re-seeds A_0 under
; A two-period crossover: period 2 opens at t=12 with one EVID=4 row (reset + 100 mg). The
; occasion is DERIVED from TIME, as ferx's `iov_occasion = time(12)` / `= dose` derive it, and
; the data's OCC column is read but not used. KAPPA enters BASE, which reaches the prediction
; only through A_0, i.e. at t=0 and at the reset's re-seed, and CL. So the reset row's own
; $PK (occasion 2) decides the seed, and a seed under occasion 1 or under no occasion moves
; the objective.
$INPUT ID TIME DV MDV EVID AMT CMT OCC
$DATA iov_rule_reset.csv IGNORE=@

; Closed form: A_0 is a $PK feature on every ADVAN, and ferx's side integrates [odes], so
; the two engines share no solver.
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
$TABLE ID TIME EVID OCCT IPRED ETA1 ETA2 ETA3 NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=iov_rule_reset_A.tab
