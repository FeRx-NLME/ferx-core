$PROBLEM IOV CWRES anchor (FOCEI): 1-cpt IV, IIV + per-occasion kappa on CL, no mixture, MAXEVAL=0 POSTHOC
; The #985 mixture_iv_iov data with the $MIX removed: one TVCL (the geometric mean of that
; anchor's two class finals, 1.057 and 3.3295), V at its final. MAXEVAL=0 evaluates one fixed
; point, which ferx's outer_maxiter = 0 reproduces. ETA2 / ETA3 are the occasion-1 / -2 kappas.
; What the table is for: NONMEM's CWRES linearizes over all three ETAs, the kappas included.
$INPUT ID TIME DV AMT EVID CMT OCC WT
$DATA mixture_iv_iov.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  OCC1 = 0
  OCC2 = 0
  IF (OCC.EQ.1) OCC1 = 1
  IF (OCC.EQ.2) OCC2 = 1
  KACL = OCC1*ETA(2) + OCC2*ETA(3)
  CL = THETA(1) * EXP(ETA(1) + KACL)
  V  = THETA(2)
  S1 = V

$ERROR
  IPRED = F
  W = 0.2 * IPRED
  IF (W.EQ.0) W = 1
  IWRES = (DV - IPRED) / W
  Y = IPRED * (1.0 + EPS(1))

$THETA
  (0.01, 1.876, 100.0)    ; TVCL
  (0.1, 10.2591, 1000.0)  ; V
$OMEGA 0.05 FIX          ; IIV on CL
$OMEGA BLOCK(1) 0.02 FIX ; IOV on CL, occasion 1
$OMEGA BLOCK(1) SAME     ; IOV on CL, occasion 2
$SIGMA 0.04 FIX          ; proportional RUV, SD 0.2

$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC NOABORT
$TABLE ID TIME OCC EVID DV PRED IPRED IWRES CWRES ETA1 ETA2 ETA3
       NOAPPEND NOPRINT ONEHEADER FORMAT=s1PE17.10 FILE=iov_cwres_focei.tab
