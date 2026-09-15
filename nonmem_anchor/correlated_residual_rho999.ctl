$PROBLEM correlated combined residual error, fixed sigma block at rho = -0.999
; #1307. The twin of `correlated_residual_combined.ctl`, which declares
; rho = 0.5 — four widths inside ferx's Fisher-z estimation rail, and therefore
; blind to the defect this anchors. Here
;
;   sigma_prop = 0.03, sigma_add = 0.3, Cov = -0.999 * 0.03 * 0.3 = -0.008991
;
; so rho = -0.999, which is past the rail at -tanh(3) = -0.995055.
;
; Two things are deliberate about the numbers, and neither is cosmetic. Every
; THETA, the OMEGA and the whole SIGMA block are FIX, so the objective is a pure
; function of the declared SIGMA — nothing can absorb a substituted rho. And
; `sigma_prop * IPRED` (~0.12-0.3 over this dataset) is of the same size as
; `sigma_add` (0.3), so the residual variance
;
;   IPRED^2*s1^2 + s2^2 + 2*IPRED*rho*s1*s2
;
; nearly cancels at rho = -1 and the objective is violently sensitive to rho
; there. A balanced block is what makes this an oracle: the first attempt used
; s1 = 0.2 / s2 = 0.3 at rho = +0.999, where the cross term is a twentieth of
; the diagonal, the whole rho = 0.995 -> 0.999 effect was 0.012 OFV, and the
; fixture's own FOCE-vs-NONMEM baseline gap (0.005-0.02) swamped it.
$DATA correlated_residual_combined.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1) * EXP(ETA(1))
  V  = THETA(2)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED + IPRED*EPS(1) + EPS(2)
$THETA
  (0.01, 1.0, 10.0) FIX
  (0.1, 10.0, 100.0) FIX
$OMEGA 0.04 FIX
$SIGMA BLOCK(2) FIX
  0.0009
  -0.008991
  0.09
$ESTIMATION METHOD=1 MAXEVAL=0 NOABORT
