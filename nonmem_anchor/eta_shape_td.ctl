$PROBLEM #1716 ETA shape anchor: heavy-tailed (t-distribution) on CL, nu = 5
; PsN `transform` / Petersson et al. 2009 third-order series:
; ETATR = ETA*(1 + (ETA^2+1)/(4 NU) + (5 ETA^4+16 ETA^2+3)/(96 NU^2)
;              + (3 ETA^6+19 ETA^4+17 ETA^2-15)/(384 NU^3)), CL = TVCL*EXP(ETATR).
; Differs from eta_shape_null.ctl by this transform alone.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  E1 = ETA(1)
  T1 = (E1**2 + 1)/(4*THETA(3))
  T2 = (5*E1**4 + 16*E1**2 + 3)/(96*THETA(3)**2)
  T3 = (3*E1**6 + 19*E1**4 + 17*E1**2 - 15)/(384*THETA(3)**3)
  ETATR = E1*(1 + T1 + T2 + T3)
  CL = THETA(1)*EXP(ETATR)
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
  5.0 FIX              ; NU
$OMEGA
  0.2   ; ETA_CL
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER FILE=eta_shape_td.tab FORMAT=s1PE23.16
