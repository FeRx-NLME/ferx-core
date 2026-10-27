$PROBLEM proportional error with residual variances below 1e-12 (#2441 anchor, k = 1e6)
; variance_floor_prop_k1.ctl on the same data in units a million times larger: AMT and DV
; times 1e6. Every residual variance here is above 1e-6, so the old ferx floor of 1e-12
; never applied; the stream is identical to the k = 1 one but for $DATA and $TABLE FILE.
; The two objectives must differ by exactly 2*n*ln(1e6) for n = 108 observations.
; Data: make_variance_floor_prop.py.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA variance_floor_prop_k1e6.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  CL = THETA(1) * EXP(ETA(1))
  V  = THETA(2) * EXP(ETA(2))
  S1 = V

$ERROR
  IPRED = F
  Y = F * (1 + EPS(1))

$THETA
  (0.01, 1.0, 100.0) FIX   ; TVCL
  (0.1, 10.0, 1000.0) FIX  ; TVV

$OMEGA
  0.09 FIX  ; ETA_CL
  0.04 FIX  ; ETA_V

$SIGMA
  0.01 FIX  ; PROP_ERR (ferx declares `~ 0.1 (sd)`; NONMEM takes the variance)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME DV PRED IPRED ETA1 ETA2 NOPRINT NOAPPEND ONEHEADER FORMAT=s1PE23.16
  FILE=variance_floor_prop_k1e6.tab
