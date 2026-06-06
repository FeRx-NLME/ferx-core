# Covariate Model

This example demonstrates a two-compartment oral model with body weight (WT) and creatinine clearance (CRCL) as covariates on clearance.

## Model File (`two_cpt_oral_cov.ferx`)

```
# Two-compartment oral PK model with covariates

[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV1(50.0, 0.1, 1000.0)
  theta TVQ(10.0, 0.01, 200.0)
  theta TVV2(100.0, 0.1, 5000.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta THETA_CRCL(0.5, 0.01, 2.0)

  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_Q  ~ 0.04
  omega ETA_V2 ~ 0.09
  omega ETA_KA ~ 0.25

  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * (CRCL/100)^THETA_CRCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 500
  covariance = true
```

The optional [`[covariates]`](../model-file/covariates.md) block declares which
dataset columns are covariates and their type. It is not required, but when
present it is authoritative and lets ferx validate the columns and emit a
covariate table — see below.

## Covariate Effects

The clearance equation includes two covariate effects:

```
CL = TVCL * (WT/70)^THETA_WT * (CRCL/100)^THETA_CRCL * exp(ETA_CL)
```

- **(WT/70)^THETA_WT**: Allometric scaling of clearance with body weight, centered at 70 kg. THETA_WT is estimated (expected ~0.75 for CL).
- **(CRCL/100)^THETA_CRCL**: Renal function effect on clearance, centered at 100 mL/min. THETA_CRCL is estimated.

## Data Requirements

The dataset must include `WT` and `CRCL` columns:

```csv
ID,TIME,DV,EVID,AMT,CMT,MDV,WT,CRCL
1,0,.,1,100,1,1,72.5,105
1,0.5,12.3,0,.,.,0,72.5,105
1,1.0,18.7,0,.,.,0,72.5,105
```

By default, covariate columns are automatically detected -- any column not in the standard NONMEM set is treated as a covariate. Covariate names are **case-sensitive** and must match the names used in `[individual_parameters]` exactly.

Adding a [`[covariates]`](../model-file/covariates.md) block makes this explicit and authoritative: only the listed columns are treated as covariates, each declared column must exist in the data and be numerically coded (categoricals must be integer-coded), and a covariate used in the model but not declared produces a warning recommending it be declared.

## Running

```bash
ferx examples/two_cpt_oral_cov.ferx --data data/two_cpt_oral_cov.csv
```

When the model declares a `[covariates]` block, ferx also writes
`two_cpt_oral_cov-covtab.csv` alongside the usual `-sdtab.csv` and `-fit.yaml`:
one row per input record (`ID,TIME,EVID` plus the declared covariate columns),
which the R package and covariate-diagnostic tooling consume.

## Notes

- FOCEI is used here because the proportional error model creates an interaction between random effects and residual error
- Covariate centering (dividing by 70 for weight, 100 for CRCL) improves numerical stability and makes the typical value (TVCL) interpretable as the clearance for a 70 kg patient with CRCL of 100 mL/min
- The estimated covariate exponents (THETA_WT, THETA_CRCL) have standard errors that can be used to test significance
