# ODE Model with Absorption Lag Time

This example demonstrates the `LAGTIME` keyword in `[individual_parameters]` for ODE models. The complete model file is [`examples/warfarin_ode_lagtime.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_ode_lagtime.ferx).

## When to use

Use `LAGTIME` (ODE version) when:
- The drug shows a measurable delay between dosing and the rise in concentration
- You want IIV on the lag time itself (log-normal, like CL or V)
- Your structural model is ODE-based and you cannot use the analytical solver's `lagtime=` argument

For analytical models the lag time is passed directly to the solver:

```
pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=TLAG)
```

For ODE models the equivalent is to assign `LAGTIME` in `[individual_parameters]` — the event handler delays every dose event for that subject by the individual lag time, without any change to the ODE equations.

## Model file

This is the contents of [`examples/warfarin_ode_lagtime.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_ode_lagtime.ferx):

```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1,    0.1,  500.0)
  theta TVKA(1.0,   0.01,  50.0)
  theta TVLAG(0.5,  0.01,   5.0)

  omega ETA_CL  ~ 0.07
  omega ETA_V   ~ 0.02
  omega ETA_KA  ~ 0.40
  omega ETA_LAG ~ 0.09

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV  * exp(ETA_V)
  KA      = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL/V * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 500
  covariance = true
```

Two features work together here:

1. **`LAGTIME`** in `[individual_parameters]`: assigning to the reserved name `LAGTIME` delays every dose event for that subject. No keyword in `[structural_model]` or `[odes]` is needed.

2. **`obs_scale = V`**: the ODE state `central` is in amounts (mg), but `DV` is in concentrations (mg/L). Dividing by V converts amount to concentration before the residual is formed. This is equivalent to writing `d/dt(central) = KA * depot / V - (CL/V) * central` and using a concentration-scale ODE state.

## Running the fit

```bash
ferx examples/warfarin_ode_lagtime.ferx --data data/warfarin_ode_lagtime.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_ode_lagtime.ferx", "data/warfarin_ode_lagtime.csv")?;
println!("TVLAG = {:.3} h", result.theta["TVLAG"].estimate);
// Individual lag times are in result.individual_estimates["LAGTIME"]
```

## Interpreting output

The fit YAML includes `TVLAG` in the `theta:` section:

```yaml
theta:
  TVCL:
    estimate: 0.134
    ...
  TVLAG:
    estimate: 0.489
    se: 0.041
    rse_pct: 8.4
```

Individual `LAGTIME` values are returned in `FitResult.individual_estimates["LAGTIME"]` alongside `CL`, `V`, and `KA`.

## Tips

- **IIV on LAGTIME**: log-normal IIV (`ETA_LAG ~ 0.09`) is natural because lag times must be positive. Starting variance 0.09 corresponds to ~30% CV.
- **LAGTIME vs. transit compartments**: a lag time is appropriate when there is an identifiable delay before absorption starts. Transit compartments (see [Transit Absorption](transit-absorption.md)) are better when the delay distributes over time (smooth absorption rise). The two approaches are not equivalent.
- **Negative DV at lag time**: if DV observations fall before the estimated lag time for a subject, IPRED is zero at those times, which produces large IWRES. Consider filtering with `[data_selection]` or re-examining the lag-time prior.
- **Amount vs. concentration ODE states**: using `obs_scale = V` (as in this example) keeps the ODE equations in amount units and is marginally faster than dividing every flux by V inside the ODE.
