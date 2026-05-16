# Stochastic Differential Equations (SDE / Diffusion)

The `[diffusion]` block adds continuous stochastic noise to ODE state variables.
This models *system noise* — structural uncertainty that accumulates between
observations — as opposed to measurement noise (sigma) or inter-individual
variability (omega).

## What is system noise?

In a standard ODE-based NLME model the trajectory of each subject is
deterministic once the individual parameters are fixed.  In an SDE model the
ODE is replaced by an Itô stochastic differential equation:

```
dX = f(X, t, θ) dt  +  diag(σ_w) dW
```

where `dW` is a vector of independent Wiener increments.  The diagonal entries
`σ_w` are the *diffusion standard deviations*; their squares `σ²_w` are the
**diffusion variances** declared in `[diffusion]`.

FeRx estimates the diffusion variances with an Extended Kalman Filter (EKF):
the state covariance matrix `P` is propagated alongside the ODE trajectory and
updated at each observation.  The total observation variance is:

```
V_total = P_ekf[obs_cmt, obs_cmt]  +  V_residual
```

where `V_residual` is the variance from the `[error_model]` (sigma-based).

## Declaring diffusion variances

```
[diffusion]
  STATE_NAME ~ initial_value
  STATE_NAME ~ initial_value FIX
```

- `STATE_NAME` must be one of the state names listed in `[structural_model]`.
- The value after `~` is the **initial estimate of the variance** σ²_w (≥ 0).
- `FIX` pins the parameter at its initial value (not estimated).
- Each declared state gets a population parameter named `DIFF_<STATE>` (e.g.,
  `DIFF_CENTRAL`).  These appear in `theta_names` of the fit result alongside
  the regular structural parameters.

## How diffusion differs from sigma and omega

| | **sigma** | **omega** | **diffusion** |
|---|---|---|---|
| What it is | Measurement noise (assay/residual error) | Between-subject variability | Within-subject system noise |
| When it acts | At each observation time | At model initialisation (between subjects) | Continuously along the trajectory |
| What it affects | Observation residuals only | Individual parameter values | State trajectories between doses |
| NONMEM analogy | `EPS` / `SIGMA` | `ETA` / `OMEGA` | No direct analogy |

A large estimated `DIFF_CENTRAL` relative to sigma suggests that the ODE
structure is missing an important mechanism (e.g., a compartment, a feedback
loop, or a covariate effect).

## When to use SDE models

The primary signal is residual autocorrelation.  If the IWRES Durbin-Watson
statistic is low (< 1.5) in a fitted ODE model, adding a diffusion term on the
slow-varying compartment often absorbs the systematic drift.

## Worked example: 1-cpt IV with central diffusion

```
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
  central ~ 0.5       # initial estimate for DIFF_CENTRAL

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
```

The estimated `DIFF_CENTRAL` gives the variance of the stochastic driving noise
on the central compartment amount per unit time.  A value near zero indicates
the ODE alone explains the data; a large value suggests model misspecification.

## Constraints and incompatibilities

| Situation | Behaviour |
|---|---|
| `[diffusion]` on an analytical PK model (`pk ...`) | Parse-time error |
| `method = saem` with `[diffusion]` | Hard error at fit time |
| `gradient_method = ad` with `[diffusion]` | Warning emitted; automatically switched to `fd` |
| State name not in `states = [...]` | Parse-time error |
| Negative initial value | Parse-time error |

## ferx-r usage note

When using FeRx from R via the ferx-r package, the `[diffusion]` block is part
of the `.ferx` model file and requires no special R-side argument.  The
estimated diffusion variances appear in the returned fit object alongside other
theta parameters.  See the ferx-r documentation (`?ferx_fit`) for details on
accessing `theta_names` and interpreting `uses_sde` in the fit object.
