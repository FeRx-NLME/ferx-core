# Two-Compartment IV Bolus

This example fits a two-compartment IV bolus model with four random effects.

## Model File (`two_cpt_iv.ferx`)

```
# Two-compartment IV bolus PK model

[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV1(50.0, 0.1, 1000.0)
  theta TVQ(10.0, 0.01, 200.0)
  theta TVV2(100.0, 0.1, 5000.0)

  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_Q  ~ 0.04
  omega ETA_V2 ~ 0.09

  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 500
  covariance = true
```

## Model Description

- **Structure**: Two-compartment model with central (V1) and peripheral (V2) compartments connected by intercompartmental clearance (Q)
- **Route**: Intravenous bolus (dose goes directly into central compartment)
- **Random effects**: Log-normal on all four PK parameters
- **Parameters**:
  - CL: Systemic clearance (L/h)
  - V1: Central volume of distribution (L)
  - Q: Intercompartmental clearance (L/h)
  - V2: Peripheral volume of distribution (L)

The bi-exponential concentration profile is characterized by a rapid distribution phase (alpha) followed by a slower elimination phase (beta).

## Running

```bash
ferx examples/two_cpt_iv.ferx --data data/two_cpt_iv.csv
```

## Notes

- Two-compartment models have more parameters and may need more iterations to converge
- The `global_search = true` option can help if convergence is difficult
- Consider reducing random effects (e.g., fixing Q or V2 variability) if the model is overparameterized
