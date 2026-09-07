$PROBLEM modelsearch anchor (#1181) - LAGTIME(ON) evaluated (MAXEVAL=0) at ferx's estimates
; lag.ctl's minimizer stopped (MINIMIZATION TERMINATED) at -287.053 with ALAG1 = 0.0025;
; ferx reached -287.629 at ALAG1 = 0.00613 on the same flat direction. This evaluates
; NONMEM's objective at ferx's point, so the two engines are compared on the same model
; at the same parameters.
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  ALAG1 = THETA(4)*EXP(ETA(4))
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.1327108)     ; TVCL
$THETA (0, 7.7426766)     ; TVV
$THETA (0, 0.818823)      ; TVKA
$THETA (0, 0.0061279)     ; TVALAG
$OMEGA 0.02861324892854351
$OMEGA 0.009576881524064917
$OMEGA 0.336970828848554
$OMEGA 6.14421235332821E-06
$SIGMA 0.00010935459985     ; 0.010457274968254479^2
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=0 PRINT=5 NOABORT
