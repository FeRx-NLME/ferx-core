$PROBLEM modelsearch anchor (#1181) - PERIPHERALS(1): 2-cpt oral
; ferx candidate: pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA), Q = TVQ, V2 = TVV2 (no eta).
; Inits: the base's final estimates (base.ext); Q = CL, V2 = 0.05 * V (Pharmpy's
; add_peripheral_compartment rule), as ferx derives them.
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN4 TRANS4
$PK
  CL = THETA(1)*EXP(ETA(1))
  V2 = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  Q  = THETA(4)
  V3 = THETA(5)
  S2 = V2
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.132695)   ; TVCL
$THETA (0, 7.73771)    ; TVV
$THETA (0, 0.810796)   ; TVKA
$THETA (0, 0.132695)   ; TVQ  = CL
$THETA (0, 0.3868855)  ; TVV2 = 0.05 * V
$OMEGA 0.0285884       ; ETA_CL
$OMEGA 0.00959179      ; ETA_V
$OMEGA 0.33588         ; ETA_KA
$SIGMA 0.000111621     ; proportional variance
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
