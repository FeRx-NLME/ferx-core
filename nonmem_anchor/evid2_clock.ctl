$PROBLEM #1809: does a leading EVID=2 / EVID=0,MDV=1 record start the clock? (TIME-in-RHS model)
; ferx twin: evid2_clock_fit.ferx (same CSV). Zero initial state, no doses (except ID 5),
; d/dt(resp) = KOUT*((PLB*exp(-TF*T) + EFF) - resp), EFF from the EXPO data column.
;  ID 1: EVID=2 at 0, EXPO constant 3          (the static path ferx used to drop)
;  ID 2: EVID=2 at 0 with EXPO 0, then 3       (the per-record path)
;  ID 3: EVID=0/MDV=1 at 0, EXPO constant      (option A2)
;  ID 4: EVID=0/MDV=1 at 0, EXPO 0 then 3      (option A2, per-record path)
;  ID 5: EVID=2 at 0, bolus 2 into RESP at 2, obs 4 / 8, EXPO constant
; Expected (closed form, = the #1809 triage's NONMEM 7.6.0 PRED to 4 decimals):
;  IDs 1-4 at 2/4/8(/12): 3.9030 / 5.7482 / 6.6383 (/ 6.2937); ID 5 at 4/8: 6.9612 / 7.0845
$INPUT ID TIME DV EVID MDV AMT CMT EXPO
$DATA evid2_clock.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=9
$MODEL COMP=(RESP DEFDOSE DEFOBS)
$PK
  KOUT = THETA(1)
  PLB  = THETA(2)*EXP(ETA(1))
  TF   = THETA(3)
  EFF  = THETA(4)*EXPO/(THETA(5)+EXPO)
$DES
  DADT(1) = KOUT*((PLB*EXP(-TF*T) + EFF) - A(1))
$ERROR
  IPRED = A(1)
  Y = IPRED + EPS(1)
$THETA 0.25 FIX  6.0 FIX  0.15 FIX  8.0 FIX  2.0 FIX
$OMEGA 0.04 FIX
$SIGMA 0.25 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME EVID MDV EXPO PRED NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=evid2_clock.tab
