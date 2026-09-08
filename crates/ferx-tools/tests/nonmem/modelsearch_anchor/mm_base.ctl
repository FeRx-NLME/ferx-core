$PROBLEM modelsearch ODE-candidate anchor (#1257) - base: warfarin 1-cpt oral, FOCEI, MAXEVAL=0
; The control for mm_eval.ctl: the same objective at the same parameter
; vector, with the analytic first-order elimination the ferx twin
; (mm_base.ferx) also uses. Evaluation only - nothing here is minimised, so
; the two engines are compared on the model and not on an optimizer.
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.2)   ; TVCL
$THETA (0, 10.0)  ; TVV
$THETA (0, 1.5)   ; TVKA
$OMEGA 0.09       ; ETA_CL
$OMEGA 0.04       ; ETA_V
$OMEGA 0.30       ; ETA_KA
$SIGMA 0.0004     ; proportional variance (= 0.02 SD)
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=0 PRINT=1 NOABORT
