# Stochastic Differential Equations (SDE)

This example extends a one-compartment ODE model with a `[diffusion]` block that adds continuous process noise (SDE / EKF). The complete model file is [`examples/warfarin_sde.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_sde.ferx).

## When to use

Add `[diffusion]` process noise when:
- IWRES shows autocorrelation that cannot be absorbed by the residual error alone (Durbin-Watson < 1.5)
- Structural misspecification is suspected (e.g. unmodelled absorption variability, time-varying elimination)
- You want a data-driven stochastic extension rather than adding a transit compartment or a new mechanistic model

The SDE approach inflates the residual variance at each observation by the integrated process covariance (`V_total = P[state] + σ²(PRED)`), estimated via an Extended Kalman Filter (EKF). `DIFF_CENTRAL` captures the diffusion coefficient on the central compartment state; it is estimated on the variance scale.

## Dataset

Same as the standard warfarin ODE example — `data/warfarin_sde.csv` uses the standard warfarin format:

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV
1,0,.,1,100,1,0,1
1,0.5,5.37,0,.,1,0,0
...
```

## Model file

This is the contents of [`examples/warfarin_sde.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_sde.ferx):

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
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL / V) * central

[diffusion]
  central ~ 0.01

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method          = foce
  maxiter         = 300
  covariance      = true
  gradient_method = fd
```

The `[diffusion]` block declares `central ~ 0.01`, meaning the initial variance estimate for the diffusion coefficient on the `central` compartment is 0.01 (variance scale, not SD). The EKF propagates this uncertainty forward in time between observations.

**Important**: SDE uses finite-difference gradients (`gradient_method = fd`); the analytic `Dual2` sensitivity path does not cover the EKF. This makes SDE fits slower than equivalent analytical models; the EKF evaluation itself is also more expensive than a plain ODE step.

## Running the fit

```bash
ferx examples/warfarin_sde.ferx --data data/warfarin_sde.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/warfarin_sde.ferx", "data/warfarin_sde.csv")?;
println!("DIFF_CENTRAL = {:.4}", result.diffusion["central"].estimate);
```

## Interpreting output

The fit YAML gains a `diffusion:` section:

```yaml
diffusion:
  central:
    estimate: 0.008432
    se: 0.001102
```

A small `DIFF_CENTRAL` (near zero) indicates the process noise is negligible and the standard ODE model is adequate. A large value relative to the mean compartment level suggests real stochastic variability; check whether the DW statistic improves (moves toward 2.0) compared to the deterministic ODE.

## Tips

- **Compare OFV**: fit the deterministic ODE first. The SDE model adds one parameter (`DIFF_CENTRAL`); a Δ OFV > 3.84 (χ²₁ at 5%) justifies the addition. Note that the parameter is at a lower boundary (variance ≥ 0), so the asymptotic null is a 50:50 mixture of χ²₀ and χ²₁ — use a Δ OFV > 2.71 as the 5% threshold.
- **Multiple diffusion states**: add one `state ~ init_variance` line per ODE state. In practice only the observed-compartment state benefits from diffusion; depot-compartment diffusion is rarely estimable from concentration data.
- **Speed**: SDE fits are ~5–10× slower than equivalent analytical models. Use a release build (`cargo build --release`) and allow more outer iterations.
- **SDE + ADDL**: steady-state dosing (`SS=1`) and `ADDL` are supported with the SDE solver; the EKF is reset to the analytical steady-state covariance at each SS dose event.
