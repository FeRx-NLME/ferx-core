$PROBLEM Infusion (RATE>0) into a first-order absorption depot — reference for ferx-core #719 (gap 2)
; Anchors ferx's infusion-into-built-in-absorption support (the dose is a zero-order source
; feeding the absorption kernel, so R_in becomes the convolution R_in_inf = (F*amt/T)*[G(t)-
; G(t-T)]) against NONMEM's native ADVAN2 zero-order-into-depot behaviour: a RATE>0 dose into the
; depot fills it at a constant rate over T = AMT/RATE = 100/25 = 4 h, and first-order KA then
; carries it to central — exactly the convolution of the first-order kernel with the infusion.
; Pure evaluation at fixed thetas (MAXEVAL=0); PRED is the reference for the ferx test.
$DATA inf_first_order.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
  KA = THETA(3)
  S2 = V
$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))
$THETA 1.0 FIX  20.0 FIX  0.6 FIX   ; CL V KA
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
