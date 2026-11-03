$PROBLEM #852 ADVAN5 liver model - FOCEI evaluation at fixed parameters
; depot -> trans1 -> trans2 -> liver <-> central <-> periph, written as rate constants.
; K12 = K23 = K34 (one transit rate: a defective system matrix), liver drain
; (QH + CLINT)/V4 ~ 107/h against a terminal phase near 0.02/h (stiff).
; Dataset: advan5_liver.csv (DV simulated by advan5_liver_sim.ctl, seed 852).
$INPUT ID TIME AMT RATE CMT EVID MDV DV SS II
$DATA advan5_liver.csv IGNORE=@
$SUBROUTINES ADVAN5
$MODEL NCOMP=6
  COMP=(DEPOT,DEFDOSE)
  COMP=(TRANS1)
  COMP=(TRANS2)
  COMP=(LIVER)
  COMP=(CENTRAL,DEFOBS)
  COMP=(PERIPH)
$PK
  KTR   = THETA(1)*EXP(ETA(2))
  CLINT = THETA(2)*EXP(ETA(1))
  QH    = THETA(3)
  V4    = THETA(4)
  V     = THETA(5)
  Q     = THETA(6)
  VP    = THETA(7)
  F1    = THETA(8)
  ALAG1 = THETA(9)
  K12 = KTR
  K23 = KTR
  K34 = KTR
  K45 = QH/V4
  K40 = CLINT/V4
  K54 = QH/V
  K56 = Q/V
  K65 = Q/VP
  S5  = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 1.2, 50)      ; KTR
  (0.1, 20, 500)       ; CLINT
  (1, 17.4, 100)       ; QH
  (0.01, 0.35, 10)     ; V4
  (1, 30, 500)         ; V
  (0.1, 5, 100)        ; Q
  (1, 200, 2000)       ; VP
  (0.01, 0.8, 1)       ; F1
  (0, 0.3, 5)          ; ALAG1
$OMEGA 0.09 ; ETA_CLINT
$OMEGA 0.04 ; ETA_KTR
$SIGMA 0.01 ; PROP
$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC PRINT=1
$TABLE ID TIME EVID PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER NOAPPEND FORMAT=,1PE20.12 FILE=advan5_liver.tab
