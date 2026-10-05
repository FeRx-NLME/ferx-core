$PROBLEM #1661 wide-ODE anchor: simulate the circulating-cell data
; Sequential PK/PD: 2-cpt PK from data columns drives a 4-transit myelosuppression
; chain (prol -> tr1 -> tr2 -> tr3 -> circ) with linear drug effect and feedback.
; 14 individual parameters on 4 THETA + 3 ETA, which is what the anchor is about:
; ferx used to run any ODE model with more than 12 individual parameters on finite
; differences. Truth: BAS 5, MTT 100 h, SLOPE 0.1 L/mg, GAM 0.17,
; OMEGA 0.1 / 0.05 / 0.1, proportional SD 0.15.
$INPUT ID TIME AMT CMT EVID MDV DV V1I K10I K12I K21I
$DATA wide_ode_myelo_template.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=10
$MODEL COMP=(CENT DEFDOSE) COMP=(PERI) COMP=(PROL) COMP=(TR1) COMP=(TR2)
       COMP=(TR3) COMP=(CIRC DEFOBS)

$PK
V1  = V1I
K10 = K10I
K12 = K12I
K21 = K21I
BAS   = THETA(1)*EXP(ETA(1))
MTT   = THETA(2)*EXP(ETA(2))
SLOPE = THETA(3)*EXP(ETA(3))
GAM   = THETA(4)
KTR   = 4/MTT
F3 = BAS
F4 = BAS
F5 = BAS
F6 = BAS
F7 = BAS

$DES
CP    = A(1)/V1
EDRUG = SLOPE*CP
FEED  = (BAS/A(7))**GAM
DADT(1) = -K10*A(1) - K12*A(1) + K21*A(2)
DADT(2) =  K12*A(1) - K21*A(2)
DADT(3) =  KTR*A(3)*(1 - EDRUG)*FEED - KTR*A(3)
DADT(4) =  KTR*A(3) - KTR*A(4)
DADT(5) =  KTR*A(4) - KTR*A(5)
DADT(6) =  KTR*A(5) - KTR*A(6)
DADT(7) =  KTR*A(6) - KTR*A(7)

$ERROR
IPRED = F
Y = IPRED*(1 + EPS(1))

$THETA 5 100 0.1 0.17
$OMEGA 0.1 0.05 0.1
$SIGMA 0.0225

$SIMULATION (1661) ONLYSIMULATION
$TABLE ID TIME AMT CMT EVID MDV DV V1I K10I K12I K21I NOPRINT NOHEADER NOAPPEND
       FORMAT=s1PE17.9 FILE=wide_ode_myelo_sim.tab
