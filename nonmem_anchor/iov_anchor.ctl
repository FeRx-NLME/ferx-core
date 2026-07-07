$PROBLEM IOV simulate anchor (#723) - IOV on V, 1-cpt IV bolus, obs at t=0 per occasion
; IPRED at each occasion's t=0 observation = Dose/V (fresh EVID=4 reset+bolus, no
; elapsed time, no carryover). log IPRED = log(AMT) - log(V) - ETA_V - KAPPA_occ, so
; the within-subject between-occasion variance of log(IPRED) = Var(KAPPA) = omega^2_IOV.
$INPUT ID TIME DV AMT EVID CMT MDV OCC
$DATA iov_anchor.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  KAPPA = 0
  IF(OCC.EQ.1) KAPPA = ETA(3)
  IF(OCC.EQ.2) KAPPA = ETA(4)
  IF(OCC.EQ.3) KAPPA = ETA(5)
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2) + KAPPA)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA
  1.0    ; TVCL
  10.0   ; TVV
$OMEGA
  0.09   ; ETA(1) CL BSV
  0.04   ; ETA(2) V  BSV
$OMEGA BLOCK(1)
  0.04   ; ETA(3) IOV variance (omega^2_IOV)
$OMEGA BLOCK(1) SAME  ; ETA(4) occasion 2
$OMEGA BLOCK(1) SAME  ; ETA(5) occasion 3
$SIGMA
  0.0025 ; proportional residual, SD 0.05 (IPRED is noise-free; kept for parity)
$SIMULATION (20260707) ONLYSIMULATION SUBPROBLEMS=1
$TABLE ID TIME OCC IPRED Y NOPRINT NOAPPEND ONEHEADER FILE=iov_anchor.tab
