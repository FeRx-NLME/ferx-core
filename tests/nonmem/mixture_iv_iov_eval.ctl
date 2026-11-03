$PROBLEM #1932 eval-only anchor: tests/nonmem/mixture_iv_iov.ctl at its own final estimates, MAXEVAL=0 POSTHOC
; Same model and data as mixture_iv_iov.ctl (#985). $THETA is set to that run's .ext
; -1000000000 row (as printed, 6 significant figures) and MAXEVAL=0, so NONMEM only evaluates
; the objective and the POSTHOC EBEs at one fixed point. ferx's outer_maxiter = 0 evaluates
; the same point. The table pass runs at MIXNUM = MIXEST, so IPRED / IWRES / CWRES / ETA1-3
; are the most-probable class's, which is what ferx reports (mixest_etas / mixest_kappas).
; ETA2 and ETA3 are the occasion-1 and occasion-2 kappas.
$INPUT ID TIME DV AMT EVID CMT OCC WT
$DATA mixture_iv_iov.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  IF (MIXNUM.EQ.1) TVCL = THETA(1)
  IF (MIXNUM.EQ.2) TVCL = THETA(2)
  OCC1 = 0
  OCC2 = 0
  IF (OCC.EQ.1) OCC1 = 1
  IF (OCC.EQ.2) OCC2 = 1
  KACL = OCC1*ETA(2) + OCC2*ETA(3)
  CL = TVCL * EXP(ETA(1) + KACL)
  V  = THETA(3)
  S1 = V

$MIX
  NSPOP = 2
  P(1) = THETA(4)
  P(2) = 1.0 - THETA(4)

$ERROR
  MIXE = MIXEST
  IPRED = F
  ; proportional RUV, SIGMA(1,1) = 0.04 FIX, so SD = 0.2 * IPRED
  W = 0.2 * IPRED
  IF (W.EQ.0) W = 1
  IWRES = (DV - IPRED) / W
  Y = IPRED * (1.0 + EPS(1))

$THETA
  (0.01, 1.05700, 100.0)     ; TVCL1  (mixture_iv_iov.ext final)
  (0.01, 3.32950, 100.0)     ; TVCL2
  (0.1, 10.2591, 1000.0)     ; V
  (0.001, 0.525988, 0.999)   ; P(1)
$OMEGA 0.05 FIX          ; IIV on CL
$OMEGA BLOCK(1) 0.02 FIX ; IOV on CL, occasion 1
$OMEGA BLOCK(1) SAME     ; IOV on CL, occasion 2
$SIGMA 0.04 FIX          ; proportional RUV

$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC NOABORT
$TABLE ID TIME OCC EVID DV MIXE PRED IPRED IWRES CWRES ETA1 ETA2 ETA3
       NOAPPEND NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=mixture_iv_iov_eval.tab
