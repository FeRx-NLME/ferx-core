$PROBLEM modelsearch anchor (#1181) - LAGTIME(ON): 1-cpt oral with a lagged, eta-bearing absorption
; ferx candidate: pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG), ALAG = TVALAG * exp(ETA_ALAG).
; Inits: the base's final estimates (base.ext), ALAG = t_first / 2 = 0.25, omega 0.01
; (Pharmpy's absorption_delay strategy), as ferx derives them.
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
$THETA (0, 0.132695)   ; TVCL
$THETA (0, 7.73771)    ; TVV
$THETA (0, 0.810796)   ; TVKA
$THETA (0, 0.25)       ; TVALAG
$OMEGA 0.0285884       ; ETA_CL
$OMEGA 0.00959179      ; ETA_V
$OMEGA 0.33588         ; ETA_KA
$OMEGA 0.01            ; ETA_ALAG
$SIGMA 0.000111621     ; proportional variance
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
