$PROBLEM ferx #1082 PR2 anchor O1 -- exact3 evaluated at its own optimum (MAXEVAL=0)
; ID = SPONSOR (top level). Below it: STUDY (2 per sponsor, unweighted) and
; ARM (2 per study, weighted gamma^2/NARM), coded as occasion-style ETA slots
; with $OMEGA SAME. Per-sponsor likelihood is independent => ordinary FOCEI.
$INPUT ID STUDY ARM ARMID=DROP STUDYID=DROP SLOT TIME DV NARM DOSE
$DATA nested_levels_3.csv IGNORE=@
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
  Y = IPRED + EPS(1)
$THETA (0, 10.1818) (0, 22.0140) (0.1, 33.4048) (0.01, 3.20150)
$OMEGA 0.0699388 FIX     ; ETA(1) sponsor
$OMEGA BLOCK(1) 0.0904362 FIX ; ETA(2) study slot 1
$OMEGA BLOCK(1) SAME   ; ETA(3) study slot 2
$OMEGA BLOCK(1) 3.75478 FIX ; ETA(4) arm slot 1 (unweighted gamma^2)
$OMEGA BLOCK(1) SAME
$OMEGA BLOCK(1) SAME
$OMEGA BLOCK(1) SAME
$SIGMA 0.255164 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC
$TABLE ID STUDY SLOT TIME IPRED ETA1 ETA2 ETA3 ETA4 ETA5 ETA6 ETA7 NOPRINT NOAPPEND ONEHEADER FORMAT=s1PE15.8 FILE=nested_levels_3_eval.tab
