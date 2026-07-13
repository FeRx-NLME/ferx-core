$PROBLEM 2-state CTMM anchor for ferx-core issue #759
; Continuous-time Markov (panel/snapshot) likelihood via the closed-form 2-state
; transition-probability matrix P(dt) = expm(Q*dt), scored with F_FLAG=1.
; Q = [[-q01, q01], [q10, -q10]], q01=exp(TH1), q10=exp(TH2), lambda=q01+q10.
;   P00=(q10+q01*e)/lam  P01=q01*(1-e)/lam
;   P10=(q10*(1-e))/lam  P11=(q01+q10*e)/lam ,  e=exp(-lam*dt)
; Each record carries the previous state (PSTATE) and the gap (DT); the first
; record per subject is the initial condition (MDV=1), matching ferx which
; conditions on each subject's first observed state.
$DATA ctmm_2state.csv IGNORE=@
$INPUT ID DV PSTATE DT MDV
$PRED
  F_FLAG = 1
  Q01 = EXP(THETA(1))
  Q10 = EXP(THETA(2))
  LAM = Q01 + Q10
  E   = EXP(-LAM*DT)
  P00 = (Q10 + Q01*E)/LAM
  P01 = Q01*(1.0 - E)/LAM
  P10 = Q10*(1.0 - E)/LAM
  P11 = (Q01 + Q10*E)/LAM
  PR = P00
  IF (PSTATE.EQ.0 .AND. DV.EQ.1) PR = P01
  IF (PSTATE.EQ.1 .AND. DV.EQ.0) PR = P10
  IF (PSTATE.EQ.1 .AND. DV.EQ.1) PR = P11
  ; ETA(1)*0: a dummy random effect NONMEM requires for multi-subject data; the
  ; model is fixed-effects (Omega FIX 0), so it contributes nothing.
  Y = PR + ETA(1)*0
$THETA -0.7   ; log q01 (awake -> asleep)
$THETA -1.2   ; log q10 (asleep -> awake)
$OMEGA 0 FIX
$ESTIMATION METHOD=0 MAXEVAL=9999 PRINT=5 NOABORT
