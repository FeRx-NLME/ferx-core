$PROBLEM #1716 ETA shape anchor: NULL TWIN, plain log-normal CL
; Every shaped arm (eta_shape_bc_pos / _bc_neg / _td) differs from this stream
; by the transform on CL alone, so ferx and NONMEM are compared on the
; difference each transform makes to the objective (see
; tests/eta_shape_nonmem_anchor.rs), and on IPRED per record.
; MAXEVAL=0 POSTHOC INTERACTION: nothing is estimated, so both engines evaluate
; the same parameter vector and the comparison is of arithmetic.
; Multi-dose 1-cpt IV (100 at 0, 12, 24 h), 30 subjects; ADVAN1 and ferx's
; `one_cpt_iv` key the doses and observations on CMT 1 alike, so ferx reads the
; same eta_shape.csv.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
$OMEGA
  0.2   ; ETA_CL
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR (ferx declares `~ 0.1 (sd)`; NONMEM takes the variance)
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER FILE=eta_shape_null.tab FORMAT=s1PE23.16
