$PROBLEM modelsearch anchor (#1181) - LAGTIME(ON);PERIPHERALS(1): 2-cpt oral with a lagged absorption
; ferx candidate: pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA, lagtime=ALAG).
; Inits: the base's final estimates (base.ext); Q = CL, V2 = 0.05 * V, ALAG = 0.25 with
; omega 0.01, as ferx derives them.
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN4 TRANS4
$PK
  CL = THETA(1)*EXP(ETA(1))
  V2 = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  Q  = THETA(4)
  V3 = THETA(5)
  ALAG1 = THETA(6)*EXP(ETA(4))
  S2 = V2
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.132695)   ; TVCL
$THETA (0, 7.73771)    ; TVV
$THETA (0, 0.810796)   ; TVKA
$THETA (0, 0.132695)   ; TVQ  = CL
$THETA (0, 0.3868855)  ; TVV2 = 0.05 * V
$THETA (0, 0.25)       ; TVALAG
$OMEGA 0.0285884       ; ETA_CL
$OMEGA 0.00959179      ; ETA_V
$OMEGA 0.33588         ; ETA_KA
$OMEGA 0.01            ; ETA_ALAG
$SIGMA 0.000111621     ; proportional variance
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
