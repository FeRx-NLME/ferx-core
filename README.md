# ferx-core

[![CI](https://github.com/FeRx-NLME/ferx-core/actions/workflows/ci.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/FeRx-NLME/ferx-core/branch/main/graph/badge.svg)](https://codecov.io/gh/FeRx-NLME/ferx-core)

A high-performance Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics, written in Rust. Implements FOCE/FOCEI estimation with analytical PK solutions and an optional ODE solver, similar to NONMEM.

## Quick Start

```bash
# Build
cargo build --release

# Fit a model
cargo run --release --bin ferx -- examples/warfarin.ferx --data data/warfarin.csv

# Fit with simulated data (uses [simulation] block)
cargo run --release --bin ferx -- examples/warfarin.ferx --simulate
```

Output files: `{model}-fit.yaml` (parameter estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Model File Format (.ferx)

Models are defined in a simple DSL. Here is a one-compartment oral PK model for warfarin:

```
[parameters]
  theta TVCL(0.2, 0.001, 10.0)     # name(initial, lower, upper)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09              # between-subject variability (variance)
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02            # residual error

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

## Structural Models

| Model | Syntax |
|-------|--------|
| 1-compartment IV bolus | `pk one_cpt_iv_bolus(cl=CL, v=V)` |
| 1-compartment oral | `pk one_cpt_oral(cl=CL, v=V, ka=KA)` |
| 1-compartment infusion | `pk one_cpt_infusion(cl=CL, v=V)` |
| 2-compartment IV bolus | `pk two_cpt_iv_bolus(cl=CL, v1=V1, q=Q, v2=V2)` |
| 2-compartment oral | `pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)` |
| 2-compartment infusion | `pk two_cpt_infusion(cl=CL, v1=V1, q=Q, v2=V2)` |
| ODE-based | Define equations in an `[odes]` block |

## Estimation Methods

Set via `method` in `[fit_options]`:

| Method | Description |
|--------|-------------|
| `foce` | First-Order Conditional Estimation |
| `focei` | FOCE with Interaction (default) |
| `gn` | Gauss-Newton (BHHH) with Levenberg-Marquardt damping |
| `gn_hybrid` | Gauss-Newton followed by FOCEI polish |
| `saem` | Stochastic Approximation EM |

### Optimizers

For FOCE/FOCEI, the outer optimizer can be set via `optimizer` in `[fit_options]`:

| Optimizer | Description |
|-----------|-------------|
| `slsqp` | NLopt Sequential Least Squares Programming (default) |
| `lbfgs` | NLopt L-BFGS |
| `mma` | NLopt Method of Moving Asymptotes |

## Data Format

Input data uses NONMEM-format CSV with columns:

- **Required**: `ID`, `TIME`, `DV`, `EVID`, `AMT`, `CMT`
- **Optional**: `RATE`, `MDV`, `II`, `SS`
- **Covariates**: Any additional columns are auto-detected

EVID codes: 0 = observation, 1 = dose, 4 = reset + dose.

## Examples

The `examples/` directory contains ready-to-run models:

| File | Description |
|------|-------------|
| `warfarin.ferx` | 1-compartment oral (warfarin PK) |
| `two_cpt_iv.ferx` | 2-compartment IV bolus |
| `two_cpt_oral_cov.ferx` | 2-compartment oral with covariates (WT, CRCL) |
| `mm_oral.ferx` | Michaelis-Menten elimination via ODE |

## R Package

An R wrapper package (`ferx`) provides `ferx_fit()`, `ferx_simulate()`, and `ferx_predict()` functions that call into this Rust engine via [extendr](https://extendr.github.io/). Source is at `../ferx`.

### Installation

```r
# Build the Rust backend and load the package
withr::with_dir("path/to/ferx", {
  system("cd src/rust && cargo build --release")
  devtools::load_all()
})
```

### Fitting a model

```r
result <- ferx_fit(
  model = "warfarin.ferx",
  data  = "warfarin.csv",
  method = "foce"        # or "focei"
)

result                   # prints summary with estimates and SEs
result$theta             # named vector of fixed-effect estimates
result$omega             # BSV covariance matrix
result$sigma             # residual error estimates
result$se_theta          # standard errors (NULL if covariance step failed)
result$sdtab             # data.frame with ID, TIME, DV, PRED, IPRED, CWRES, IWRES, ETA1..n
```

### Simulation and VPC

```r
sim <- ferx_simulate("warfarin.ferx", "warfarin.csv", n_sim = 100, seed = 42)
# Returns data.frame with SIM, ID, TIME, IPRED, DV_SIM

library(vpc)
obs <- read.csv("warfarin.csv")
vpc(obs = obs, sim = sim, sim_cols = list(dv = "DV_SIM"))
```

### Population predictions

```r
preds <- ferx_predict("warfarin.ferx", "warfarin.csv")
# Returns data.frame with ID, TIME, PRED (predictions at eta = 0)
```

## License

MIT — see [LICENSE](LICENSE).
