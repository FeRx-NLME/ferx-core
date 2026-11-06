$PROBLEM Plasma + urine, per-CMT LTBS combined error, MAXEVAL=0 at the FOCEI fit's printed final estimates (#2729)
; The natural-scale DV column is dropped; NONMEM reads LNDV, its log.
$DATA ltbs_combined_cmt.csv IGNORE=@
$INPUT ID TIME NATDV=DROP DV EVID AMT CMT MDV
$SUBROUTINES ADVAN13 TOL=12
$MODEL COMP=(DEPOT DEFDOSE) COMP=(CENTRAL DEFOBS) COMP=(URINE)
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)
  FE = THETA(4)
  S2 = V
$DES
  DADT(1) = -KA*A(1)
  DADT(2) = KA*A(1) - CL/V*A(2)
  DADT(3) = FE*CL/V*A(2)
$ERROR
  ; Plasma (CMT=2) and urine (CMT=3) each carry their own PROP and ADD. ferx's
  ; LTBS floor, spelled out: F is floored before it is logged and divides ADD.
  PROP = THETA(5)
  ADD  = THETA(6)
  IF (CMT.EQ.3) THEN
    PROP = THETA(7)
    ADD  = THETA(8)
  ENDIF
  IPR = F
  IF (IPR.LT.1E-12) IPR = 1E-12
  IPRED = LOG(IPR)
  W = SQRT(PROP**2 + ADD**2/IPR**2)
  Y = IPRED + W*EPS(1)
$THETA (0, 2.04299)  ; TVCL
$THETA (0, 9.97957)  ; TVV
$THETA (0, 1.43556)  ; TVKA
$THETA (0, 0.389745, 1) ; FE
$THETA (0, 0.0916699) ; PROP_P (log scale)
$THETA (0, 0.0282388) ; ADD_P (mg/L)
$THETA (0, 0.227097) ; PROP_U (log scale)
$THETA (0, 0.004)    ; ADD_U (mg)
$OMEGA 0.0639221        ; ETA_CL
$OMEGA 0.0314136        ; ETA_V
$SIGMA 1 FIX
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=0 PRINT=5 NOABORT
