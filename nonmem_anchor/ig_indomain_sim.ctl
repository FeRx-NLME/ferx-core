$PROBLEM Simulate in-domain inverse-Gaussian absorption data (ferx #790 analytic-IG anchor)
; In-domain IG truth (ke = CL/V = 0.10 < 1/(2*MAT*CV2) = 0.833), so a FOCEI fit
; lands inside the exponential-tilting closed form's convergence domain and the
; ferx analytic `pk one_cpt_ig` closed form actually runs (no flip-flop reroute).
$INPUT ID TIME DV AMT EVID CMT MDV
$DATA transit_oral.csv IGNORE=@

$SUBROUTINES ADVAN13 TOL=9
$MODEL
  COMP=(DEPOT,DEFDOSE)     ; 1 = inert dose carrier (F1=0); IG feeds central directly
  COMP=(CENTRAL,DEFOBS)    ; 2 = central (amount; concentration = A(2)/V)

$PK
  CL  = THETA(1)*EXP(ETA(1))
  V   = THETA(2)*EXP(ETA(2))
  MAT = THETA(3)
  CV2 = THETA(4)
  K20 = CL/V
  PI  = 3.14159265358979312
  IF (AMT.GT.0.0.AND.CMT.EQ.1) PODO = AMT
  IF (AMT.GT.0.0.AND.CMT.EQ.1) TDOS = TIME
  F1  = 0.0

$DES
  TAD = T - TDOS
  IF (TAD.LE.1.0E-10) TAD = 1.0E-10
  ARG = -(TAD - MAT)**2 / (2.0*CV2*MAT*TAD)
  RIN = PODO*SQRT(MAT/(2.0*PI*CV2*TAD**3))*EXP(ARG)
  DADT(1) = 0.0
  DADT(2) = RIN - K20*A(2)

$ERROR
  IPRED = A(2)/V
  Y = IPRED*(1.0 + EPS(1))

$THETA
  5.0 FIX    ; CL  (L/h)   truth
  50.0 FIX   ; V   (L)     truth
  2.0 FIX    ; MAT (h)     truth  (in-domain: 1/(2*MAT*CV2)=0.833 > ke=0.10)
  0.3 FIX    ; CV2         truth
$OMEGA
  0.09 FIX   ; IIV CL
  0.09 FIX   ; IIV V
$SIGMA
  0.0225 FIX ; proportional residual variance (0.15^2)

$SIMULATION (20240710) ONLYSIMULATION
$TABLE ID TIME DV AMT EVID CMT MDV NOAPPEND NOPRINT ONEHEADER FILE=ig_sim.tab
