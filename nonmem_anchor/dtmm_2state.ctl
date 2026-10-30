$PROBLEM #2473 DTMM, 2 states: P(state at record k | state at record k-1), PREVIOUS-DV $PRED
; PDV is the previous observed DV (-1 on a subject's first observed record, which is
; conditioned on: Y = 1). MDV = 1 rows are not scored. The stay probabilities are
; 1 - the off-diagonal, exactly as ferx fills the diagonal of a `type = dtmm` block.
$INPUT ID TIME DV MDV CMT DOSE PDV
$DATA dtmm_2state.csv IGNORE=@
$PRED
  P01 = 1/(1 + EXP(-(THETA(1) + THETA(2)*DOSE + ETA(1))))
  P10 = 1/(1 + EXP(-(THETA(3) + ETA(2))))
  Y = 1
  IF (PDV.EQ.0.AND.DV.EQ.0) Y = 1 - P01
  IF (PDV.EQ.0.AND.DV.EQ.1) Y = P01
  IF (PDV.EQ.1.AND.DV.EQ.0) Y = P10
  IF (PDV.EQ.1.AND.DV.EQ.1) Y = 1 - P10
$THETA
  (-10, -0.5, 10)    ; A_AS
  (-10, 0.2, 10)     ; B_DOSE
  (-10, -0.5, 10)    ; A_SA
$OMEGA 0.3 0.3
$ESTIMATION METHOD=COND LAPLACE LIKE MAXEVAL=9999 SIGL=9 NSIG=3 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME DV PDV P01 P10 ETA1 ETA2 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16
       FILE=dtmm_2state.tab
