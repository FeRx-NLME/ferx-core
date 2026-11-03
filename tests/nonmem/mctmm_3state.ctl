$PROBLEM 3-state minimal CTMM (mCTMM) anchor for ferx-core issue #2474
; Schindler & Karlsson (AAPS J 2017) minimal continuous-time Markov model, scored
; as a panel/snapshot CTMM likelihood with F_FLAG=1 (as tests/nonmem/ctmm_2state.ctl).
; Steady state: proportional odds  F_k = 1/(1+exp(-(CUT_k - LP))),  LP = BX*X,
;   P0 = F1, P1 = F2 - F1, P2 = 1 - F2,  CUT1 = THETA(2), CUT2 = CUT1 + EXP(THETA(3)).
; Adjacent rates: q(k->k+1) = P(k+1)/(TAU*(P(k)+P(k+1))), q(k+1->k) = P(k)/(TAU*(...)),
;   TAU = EXP(THETA(1)) the mean equilibration time.
; P(dt) = expm(Q*dt) in closed form (Sylvester; Q has eigenvalues 0, L1, L2):
;   P = (Q2 + S*Q + M*I)/M + E1*(Q2 - L2*Q)/(L1*(L1-L2)) + E2*(Q2 - L1*Q)/(L2*(L2-L1)),
;   S = A+B+C+D, M = A*C + A*D + B*D, L1,2 = (-S +- SQRT(S*S-4M))/2.
; Each record carries the previous state (PSTATE) and the gap (DT); the first record
; per subject is the initial condition (MDV=1), as ferx conditions on the first state.
$DATA mctmm_3state.csv IGNORE=@
$INPUT ID DV PSTATE DT X MDV
$PRED
  F_FLAG = 1
  TAU  = EXP(THETA(1))
  CUT1 = THETA(2)
  CUT2 = CUT1 + EXP(THETA(3))
  LP   = THETA(4)*X
  F1 = 1.0/(1.0 + EXP(-(CUT1 - LP)))
  F2 = 1.0/(1.0 + EXP(-(CUT2 - LP)))
  P0 = F1
  P1 = F2 - F1
  P2 = 1.0 - F2
  A = P1/(TAU*(P0 + P1))
  B = P0/(TAU*(P0 + P1))
  C = P2/(TAU*(P1 + P2))
  D = P1/(TAU*(P1 + P2))
  Q11 = -A
  Q12 = A
  Q13 = 0.0
  Q21 = B
  Q22 = -B - C
  Q23 = C
  Q31 = 0.0
  Q32 = D
  Q33 = -D
  S  = A + B + C + D
  M  = A*C + A*D + B*D
  SQ = SQRT(S*S - 4.0*M)
  L1 = (-S + SQ)/2.0
  L2 = (-S - SQ)/2.0
  E1 = EXP(L1*DT)
  E2 = EXP(L2*DT)
  W0 = 1.0/M
  W1 = E1/(L1*(L1 - L2))
  W2 = E2/(L2*(L2 - L1))
  R11 = Q11*Q11 + Q12*Q21 + Q13*Q31
  P11 = W0*(R11 + S*Q11 + M*1.0) + W1*(R11 - L2*Q11) + W2*(R11 - L1*Q11)
  R12 = Q11*Q12 + Q12*Q22 + Q13*Q32
  P12 = W0*(R12 + S*Q12 + M*0.0) + W1*(R12 - L2*Q12) + W2*(R12 - L1*Q12)
  R13 = Q11*Q13 + Q12*Q23 + Q13*Q33
  P13 = W0*(R13 + S*Q13 + M*0.0) + W1*(R13 - L2*Q13) + W2*(R13 - L1*Q13)
  R21 = Q21*Q11 + Q22*Q21 + Q23*Q31
  P21 = W0*(R21 + S*Q21 + M*0.0) + W1*(R21 - L2*Q21) + W2*(R21 - L1*Q21)
  R22 = Q21*Q12 + Q22*Q22 + Q23*Q32
  P22 = W0*(R22 + S*Q22 + M*1.0) + W1*(R22 - L2*Q22) + W2*(R22 - L1*Q22)
  R23 = Q21*Q13 + Q22*Q23 + Q23*Q33
  P23 = W0*(R23 + S*Q23 + M*0.0) + W1*(R23 - L2*Q23) + W2*(R23 - L1*Q23)
  R31 = Q31*Q11 + Q32*Q21 + Q33*Q31
  P31 = W0*(R31 + S*Q31 + M*0.0) + W1*(R31 - L2*Q31) + W2*(R31 - L1*Q31)
  R32 = Q31*Q12 + Q32*Q22 + Q33*Q32
  P32 = W0*(R32 + S*Q32 + M*0.0) + W1*(R32 - L2*Q32) + W2*(R32 - L1*Q32)
  R33 = Q31*Q13 + Q32*Q23 + Q33*Q33
  P33 = W0*(R33 + S*Q33 + M*1.0) + W1*(R33 - L2*Q33) + W2*(R33 - L1*Q33)
  PR = P11
  IF (PSTATE.EQ.0 .AND. DV.EQ.1) PR = P12
  IF (PSTATE.EQ.0 .AND. DV.EQ.2) PR = P13
  IF (PSTATE.EQ.1 .AND. DV.EQ.0) PR = P21
  IF (PSTATE.EQ.1 .AND. DV.EQ.1) PR = P22
  IF (PSTATE.EQ.1 .AND. DV.EQ.2) PR = P23
  IF (PSTATE.EQ.2 .AND. DV.EQ.0) PR = P31
  IF (PSTATE.EQ.2 .AND. DV.EQ.1) PR = P32
  IF (PSTATE.EQ.2 .AND. DV.EQ.2) PR = P33
  ; ETA(1)*0: a dummy random effect NONMEM requires for multi-subject data; the
  ; model is fixed-effects (Omega FIX 0), so it contributes nothing.
  Y = PR + ETA(1)*0
$THETA 0.5    ; LTAU  log mean equilibration time
$THETA -0.3   ; C1    first cut-point
$THETA 0.2    ; LD2   log increment to the second cut-point
$THETA 0.3    ; BX    covariate effect on the linear predictor
$OMEGA 0 FIX
$ESTIMATION METHOD=0 MAXEVAL=9999 PRINT=5 NOABORT SIGL=12 NSIG=5
$COVARIANCE MATRIX=R
