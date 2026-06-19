# Introduction

ferx-core is a high-performance Nonlinear Mixed Effects (NLME) modeling engine written in Rust. It is designed for population pharmacokinetic (PopPK) analysis, implementing the same statistical methodology as established tools like NONMEM, Monolix, and Pumas.

## Key Features

- **FOCE/FOCEI estimation** -- First-Order Conditional Estimation with optional interaction, the gold standard for PopPK
- **SAEM estimation** -- Stochastic Approximation EM for robust convergence on complex models
- **Gauss-Newton (BHHH)** -- Fast outer optimizer with Levenberg-Marquardt damping, plus a GN+FOCEI hybrid
- **Importance Sampling (IMP) and SIR** -- For exact likelihood estimation and posterior sampling
- **Analytical PK solutions** -- Built-in one-, two-, and three-compartment models (IV bolus, oral, infusion) with numerical stability guarantees
- **ODE solver** -- Dormand-Prince RK45 adaptive integrator for custom kinetic models (e.g. Michaelis-Menten)
- **SDE / diffusion models** -- Extended Kalman Filter path for stochastic differential equations via the `[diffusion]` block
- **Automatic differentiation** -- Enzyme-based AD for fast, exact gradients
- **Multiple error models** -- Additive, proportional, combined, and log-transform-both-sides (LTBS); per-CMT multi-endpoint models for joint PK/PD
- **Inter-occasion variability (IOV)** -- `kappa` random effects with FOCE/FOCEI and SAEM support
- **Lagtime / ALAG** -- Absorption lag for analytical, AD, and ODE paths
- **LOQ censoring** -- Beal M3 method for below-LLOQ and above-ULOQ observations
- **NCA-based starting values** -- `inits_from_nca` for automatic theta initialization from the data
- **Simple model DSL** -- Declarative `.ferx` model files that read like equations
- **NONMEM-compatible data** -- Reads standard NONMEM CSV datasets directly
- **Covariate support** -- Time-constant and time-varying covariates with automatic detection
- **Parallel estimation** -- Per-subject computations parallelized via Rayon
- **Neural network covariates** -- MLP-based covariate mapper behind the `nn` feature (experimental)

## How It Compares

| Feature | ferx-core | NONMEM | Monolix | Pumas |
|---------|-----------|--------|---------|-------|
| Language | Rust | Fortran | C++ | Julia |
| FOCE/FOCEI | Yes | Yes | No | Yes |
| SAEM | Yes | No | Yes | Yes |
| Analytical PK | Yes | Yes | Yes | Yes |
| ODE models | Yes | Yes | Yes | Yes |
| Auto-diff | Yes (Enzyme) | No | No | Yes (ForwardDiff) |
| Open source | Yes | No | No | No |

## Architecture Overview

ferx-core uses a two-level optimization structure:

- **Outer loop**: Optimizes population parameters (theta, omega, sigma) using NLopt BOBYQA (default), SLSQP, L-BFGS, MMA, built-in BFGS, Newton trust-region, or Gauss-Newton (BHHH)
- **Inner loop**: For each subject, finds empirical Bayes estimates (EBEs) of random effects by minimizing individual negative log-likelihood

Parameters are internally transformed for unconstrained optimization: theta and sigma are log-transformed, and omega uses Cholesky factorization to guarantee positive-definiteness.

## Project Structure

```
src/
  types.rs          -- Core data structures
  api.rs            -- Public API (fit, simulate, predict)
  parser/           -- .ferx model file parser
  pk/               -- Analytical PK solutions
  ode/              -- ODE solver and predictions
  estimation/       -- FOCE/FOCEI, SAEM, Gauss-Newton, IMP, SIR, trust-region
  stats/            -- Likelihood and residual computations
  io/               -- Data reading and output writing
  ad/               -- Automatic differentiation (Enzyme)
  nn/               -- Neural network components (DCM / NODE)
  bin/run_model.rs  -- CLI binary
```

## License

ferx-core is released under the [MIT License](https://github.com/FeRx-NLME/ferx-core/blob/main/LICENSE).
