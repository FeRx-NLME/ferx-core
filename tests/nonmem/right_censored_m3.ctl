$PROBLEM Right-censored M3 upper-tail anchor for ferx-core issue #297
$DATA right_censored_m3.csv IGNORE=@
$INPUT ID DV CENS
$PRED
  ; Right-censored observation above ULOQ:
  ; F = 12, ULOQ = DV = 10, W = 2, so z = (F - ULOQ) / W = 1.
  F_FLAG = 1
  F = THETA(1) + ETA(1)*0
  W = 2
  Y = PHI((F - DV) / W)
$THETA 12 FIX
$OMEGA 0.01 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=0 NOABORT
