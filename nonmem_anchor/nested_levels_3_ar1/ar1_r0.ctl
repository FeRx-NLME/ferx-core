$PROBLEM ferx #1082 PR5 anchor V7 -- exact3 + within-arm residual correlation ar1_r0 (MAXEVAL=0)
; exact3 (../nested_levels_3_eval.ctl) at its optimum; L2 = ARMID (one EPS vector per arm);
; VIS = visit index in the arm; T1..T5 = the arm's visit times. rho = THETA(5) FIX.
$INPUT ID STUDY ARM L2 STUDYID=DROP SLOT TIME DV NARM DOSE VIS T1 T2 T3 T4 T5
$DATA nested_levels_3_ar1.csv IGNORE=@
$PRED
  EST = 0
  IF(STUDY.EQ.1) EST = ETA(2)
  IF(STUDY.EQ.2) EST = ETA(3)
  KAR = 0
  IF(SLOT.EQ.1) KAR = ETA(4)
  IF(SLOT.EQ.2) KAR = ETA(5)
  IF(SLOT.EQ.3) KAR = ETA(6)
  IF(SLOT.EQ.4) KAR = ETA(7)
  EMAX = THETA(2)*EXP(ETA(1) + EST + KAR/SQRT(NARM))
  IPRED = THETA(1) + EMAX*DOSE/(THETA(3)+DOSE)*TIME/(TIME+THETA(4))
  RHO = THETA(5)

; discrete AR(1) on the visit index: L[j,k] = rho^(j-k) * s_k, s_1 = 1, s_k = sqrt(1-rho^2)
  SR = SQRT(1 - RHO*RHO)
  P1 = RHO
  P2 = RHO*RHO
  P3 = P2*RHO
  P4 = P3*RHO
  L1=0
  L2C=0
  L3=0
  L4=0
  L5=0
  IF (VIS.EQ.1) L1 = 1
  IF (VIS.EQ.2) THEN
    L1 = P1
    L2C = SR
  ENDIF
  IF (VIS.EQ.3) THEN
    L1 = P2
    L2C = P1*SR
    L3 = SR
  ENDIF
  IF (VIS.EQ.4) THEN
    L1 = P3
    L2C = P2*SR
    L3 = P1*SR
    L4 = SR
  ENDIF
  IF (VIS.EQ.5) THEN
    L1 = P4
    L2C = P3*SR
    L3 = P2*SR
    L4 = P1*SR
    L5 = SR
  ENDIF

  RES = L1*EPS(1) + L2C*EPS(2) + L3*EPS(3) + L4*EPS(4) + L5*EPS(5)
  SD = SQRT(0.255164)
  Y = IPRED + SD*RES
$THETA (0, 10.1818) FIX (0, 22.0140) FIX (0.1, 33.4048) FIX (0.01, 3.20150) FIX
$THETA (-0.99, 0, 0.99) FIX
$OMEGA 0.0699388 FIX     ; ETA(1) sponsor
$OMEGA BLOCK(1) 0.0904362 FIX ; ETA(2) study slot 1
$OMEGA BLOCK(1) SAME   ; ETA(3) study slot 2
$OMEGA BLOCK(1) 3.75478 FIX ; ETA(4) arm slot 1 (unweighted gamma^2)
$OMEGA BLOCK(1) SAME
$OMEGA BLOCK(1) SAME
$OMEGA BLOCK(1) SAME
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=0 NOABORT POSTHOC PRINT=1 FORMAT=s1PE23.16
$TABLE ID STUDY SLOT TIME IPRED ETA1 ETA2 ETA3 ETA4 ETA5 ETA6 ETA7 NOPRINT NOAPPEND ONEHEADER FORMAT=s1PE23.16 FILE=ar1_r0.tab
