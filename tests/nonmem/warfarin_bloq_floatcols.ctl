$PROBLEM Warfarin M3 BLOQ, integer columns written float-formatted (ferx-core #1496)
; #1496 anchor: does NONMEM read a float-formatted whole number ("1.0") in an
; integer data column as that integer? Two runs, identical but for how the dataset
; spells EVID, MDV, CENS and ADDL:
;   warfarin_bloq_intcols.csv    1 / 0 / 2
;   warfarin_bloq_floatcols.csv  1.0 / 0.0 / 2.0   (every other cell byte-identical)
; Each column is live: every dose row carries II=24, ADDL=2 (doses at 0, 24, 48 h);
; 10 observation rows carry CENS=1; the observation at ID 1, TIME 0.5 carries MDV=1
; with its real DV, so it is excluded only if MDV is read as 1.
; NONMEM 7.6.0, both runs: 109 observation records, #OBJV 279.35595553971399, and
; final estimates identical to every printed digit. With that one row at MDV=0
; instead: 110 records, #OBJV 281.06112134711015, so the MDV cell is live here.
; Model: tests/nonmem/warfarin_bloq.ctl unchanged but for $INPUT (II ADDL appended),
; $DATA and $TABLE FILE. Its committed warfarin_bloq.lst is NONMEM 7.5.1
; (-216.78754391843171); on 7.6.0 the same integer file gives -216.78754377790631, so
; that 1.4e-7 gap is the version, not the data format.
; Cross-check for ferx's M3 likelihood + analytic M3 inner EBE gradient.
; ferx model: examples/warfarin_bloq.ferx (one_cpt_oral, DV ~ proportional(PROP_ERR),
; bloq_method = m3). On CENS=1 rows the DV cell carries the LLOQ (2.0) and the row
; contributes −logΦ((LLOQ−f)/√V) to the objective.
;
; NONMEM reproduces M3 with the F_FLAG / mixed-likelihood pattern under LAPLACE:
; censored rows return Φ((LLOQ−IPRED)/SD) as a likelihood (F_FLAG=1); quantified
; rows use the ordinary proportional residual. ferx's PROP_ERR is a proportional
; *SD* coefficient, so SD = THETA(4)*IPRED with EPS(1) ~ N(0,1) (SIGMA fixed to 1).
$DATA warfarin_bloq_floatcols.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV CENS II ADDL
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
$TABLE ID TIME IPRED CENS NOPRINT ONEHEADER FILE=sdtab_bloq_floatcols
