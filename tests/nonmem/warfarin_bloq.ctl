$PROBLEM Warfarin M3 BLOQ (proportional error, LLOQ=2.0) - ferx-core cross-check (issue #367)
; Cross-check for ferx's M3 likelihood + analytic M3 inner EBE gradient.
; ferx model: examples/warfarin_bloq.ferx (one_cpt_oral, DV ~ proportional(PROP_ERR),
; bloq_method = m3). On CENS=1 rows the DV cell carries the LLOQ (2.0) and the row
; contributes −logΦ((LLOQ−f)/√V) to the objective.
;
; NONMEM reproduces M3 with the F_FLAG / mixed-likelihood pattern under LAPLACE:
; censored rows return Φ((LLOQ−IPRED)/SD) as a likelihood (F_FLAG=1); quantified
; rows use the ordinary proportional residual. ferx's PROP_ERR is a proportional
; *SD* coefficient, so SD = THETA(4)*IPRED with EPS(1) ~ N(0,1) (SIGMA fixed to 1).
$DATA warfarin_bloq.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV CENS
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V
$ERROR
  LLOQ  = 2.0
  IPRED = F
  SD    = THETA(4)*IPRED          ; proportional SD, matches ferx proportional(PROP_ERR)
  IF (CENS.EQ.1) THEN
    F_FLAG = 1
    Y = PHI((LLOQ - IPRED)/SD)     ; M3: censored row contributes Φ((LLOQ−f)/SD)
  ELSE
    F_FLAG = 0
    Y = IPRED + SD*EPS(1)
  ENDIF
$THETA (0, 0.2)    ; TVCL
$THETA (0, 10.0)   ; TVV
$THETA (0, 1.5)    ; TVKA
$THETA (0, 0.02)   ; PROP_ERR (proportional SD)
$OMEGA 0.09        ; ETA_CL
$OMEGA 0.04        ; ETA_V
$OMEGA 0.30        ; ETA_KA
$SIGMA 1 FIX       ; EPS(1) ~ N(0,1); the proportional SD lives in THETA(4)
$ESTIMATION METHOD=1 LAPLACE INTER MAXEVAL=9999 NSIG=3 SIGL=9 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME IPRED CENS NOPRINT ONEHEADER FILE=sdtab_bloq
