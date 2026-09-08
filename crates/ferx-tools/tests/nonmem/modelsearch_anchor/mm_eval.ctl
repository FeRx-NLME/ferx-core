$PROBLEM modelsearch ODE-candidate anchor (#1257) - ELIMINATION(MM), MAXEVAL=0
; The NONMEM twin of the candidate `ferx modelsearch` writes for
; `ELIMINATION(MM)` from mm_base.ferx:
;
;   [structural_model]
;     ode_template one_cpt_oral(cl=CL, v=V, ka=KA)
;   [odes]
;     d/dt(central) = KA * depot - ((CL * KM / (KM + central / V)) / V) * central
;
; i.e. Pharmpy's parameterisation, `CLMM*KM*C/(KM + C)` with `C = A(2)/V` and
; `Vmax = CL*KM`, on the same disposition. ferx keeps the base model's `CL`
; as the Michaelis-Menten clearance (Pharmpy renames it to CLMM), which is
; what carries its initial estimate and its eta across the move - so THETA(1)
; is the same theta as in mm_base.ctl, at the same value.
;
; Evaluation only, at the inits the search derives: the base's own theta plus
; KM = max(DV)/2 = 6.9409, bounded above by 1.5*max(DV) = 20.8227. TOL/ATOL
; are tight because the FOCEI objective amplifies solver error; the ferx side
; pins ode_reltol = 1e-10, ode_abstol = 1e-12 for the same reason.
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN13 TOL=9 ATOL=12
$MODEL COMP=(DEPOT, DEFDOSE) COMP=(CENTRAL, DEFOBS)
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  KM = THETA(4)
  S2 = V
$DES
  CONC = A(2)/V
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - (CL*KM/(KM + CONC))/V*A(2)
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.2)               ; TVCL  (the Michaelis-Menten clearance)
$THETA (0, 10.0)              ; TVV
$THETA (0, 1.5)               ; TVKA
$THETA (0, 6.9409, 20.8227)   ; TVKM  = max(DV)/2, upper 1.5*max(DV)
$OMEGA 0.09                   ; ETA_CL
$OMEGA 0.04                   ; ETA_V
$OMEGA 0.30                   ; ETA_KA
$SIGMA 0.0004                 ; proportional variance (= 0.02 SD)
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=0 PRINT=1 NOABORT
