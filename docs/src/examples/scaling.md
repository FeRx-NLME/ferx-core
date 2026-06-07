# Unit Scaling (`[scaling]`)

This example demonstrates the `[scaling]` block with `obs_scale`, which divides every model prediction by a constant before computing residuals. The complete model file is [`examples/warfarin_scaled.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_scaled.ferx).

## When to use

Use `obs_scale` when:
- The dose (`AMT`) and the observed concentration (`DV`) are in different units
- You prefer to keep AMT and DV in their natural assay units rather than converting the CSV
- The NONMEM model uses a `SCALE` statement (`A(1)/S1`) to convert compartment amounts to concentrations

The dataset for this example records `AMT` in micrograms (100 mg = 100 000 µg) while `DV` is in mg/L. Without scaling the model would predict in µg/L — a factor-1000 mismatch. Setting `obs_scale = 1000` divides every prediction by 1000 so predictions are expressed in the same mg/L units as the observations.

## Dataset

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV
1,0,.,1,100000.0,1,0,1
1,0.5,5.37,0,.,1,0,0
...
```

AMT is 100 000 µg (= 100 mg). DV is in mg/L.

## Model file

This is the contents of [`examples/warfarin_scaled.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_scaled.ferx):

```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
  obs_scale = 1000

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
  gradient   = ad
```

The `[scaling]` block sits between `[error_model]` and `[fit_options]`. The `obs_scale` value is applied to every prediction before the residual is formed:

```
residual = DV - IPRED / obs_scale
```

PK parameter values (`CL`, `V`) remain in L/h and L — only the unit of the predicted concentration changes from µg/L to mg/L.

## Running the fit

```bash
ferx examples/warfarin_scaled.ferx --data data/warfarin_scaled.csv
```

Estimates should match the standard warfarin fit (`TVCL ≈ 0.134`, `TVV ≈ 8.1`, `TVKA ≈ 1.0`) because the same underlying PK is fitted — only the AMT unit differs.

## Tips

- **Equivalent NONMEM code**: `S1 = V / 1000` (where V is in L and DV is in mg/L while AMT is in µg) is equivalent to `obs_scale = 1000` combined with a V in litres.
- **ODE models**: for ODE models with `obs_cmt=central` in concentration units (not amount), use `obs_scale` rather than dividing inside the ODE. If the ODE state is in amounts and `DV` is concentration, use `obs_scale = V` (a parameter expression) — see [`examples/warfarin_ode_lagtime.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_ode_lagtime.ferx) for this pattern.
- **Multi-analyte**: when multiple observation compartments have different unit scales, use the `[scaling]` block with per-CMT entries — see `examples/scaling_multi_analyte.ferx` for the syntax.
