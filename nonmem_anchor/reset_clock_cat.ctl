$PROBLEM #1978 reset clock: what $ERROR TIME reads at a binary record after an EVID=3 restart (written)
; Two periods per subject, no doses; period 2 opens with an EVID=3 reset at TIME 0. The
; binary endpoint's log-odds is linear in TIME, read in $ERROR. PREDPP (ADVAN1, an empty
; compartment) is there only so the EVID=3 row is a reset as it is in ferx.
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA reset_clock_cat.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  CL = 1
  V  = 1
$ERROR
  TT = TIME
  LP = THETA(1) + THETA(2)*TIME + ETA(1)
  P  = 1/(1 + EXP(-LP))
  Y = P
  IF (DV.NE.1) Y = 1 - P
$THETA
  (-10, -1.0, 10)    ; TH0
  (-1, 0.03, 1)      ; THT
$OMEGA 0.5
$ESTIMATION METHOD=COND LAPLACE LIKE MAXEVAL=9999 PRINT=5 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME EVID TT LP P ETA1 NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16
       FILE=reset_clock_cat.tab
