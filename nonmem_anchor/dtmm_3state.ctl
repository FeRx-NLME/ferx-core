$PROBLEM #2473 DTMM, 3 ordered states: cumulative logit conditioned on the previous DV
; P(s <= 0 | j) = inv_logit(A1 - G_j + ETA), P(s <= 1 | j) = inv_logit(A1 + EXP(D2) - G_j + ETA),
; G_0 = 0, G_1 = THETA(3), G_2 = THETA(4). PDV is the previous observed DV (-1 on a
; subject's first observed record, conditioned on). MDV = 1 rows are not scored.
$INPUT ID TIME DV MDV CMT DOSE PDV
$DATA dtmm_3state.csv IGNORE=@
$PRED
  GP = 0
  IF (PDV.EQ.1) GP = THETA(3)
  IF (PDV.EQ.2) GP = THETA(4)
  C0 = 1/(1 + EXP(-(THETA(1) - GP + ETA(1))))
  C1 = 1/(1 + EXP(-(THETA(1) + EXP(THETA(2)) - GP + ETA(1))))
  Y = 1
  IF (PDV.GE.0.AND.DV.EQ.0) Y = C0
  IF (PDV.GE.0.AND.DV.EQ.1) Y = C1 - C0
  IF (PDV.GE.0.AND.DV.EQ.2) Y = 1 - C1
$THETA
  (-10, 0.1, 10)     ; A1
  (-10, 0.1, 10)     ; D2
  (-10, 1, 10)       ; G1
  (-10, 2, 10)       ; G2
$OMEGA 0.3
$ESTIMATION METHOD=COND LAPLACE LIKE MAXEVAL=9999 SIGL=9 NSIG=3 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME DV PDV C0 C1 ETA1 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16
       FILE=dtmm_3state.tab
