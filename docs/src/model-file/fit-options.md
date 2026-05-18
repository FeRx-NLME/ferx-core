# Fit Options

The optional `[fit_options]` block configures the estimation method and optimizer settings.

## Syntax

```
[fit_options]
  key = value
```

## General Options

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `method` | `foce`, `focei`, `saem` | `focei` | Estimation method |
| `maxiter` | integer | `500` | Maximum outer loop iterations |
| `covariance` | `true`, `false` | `true` | Compute covariance matrix and standard errors |
| `optimizer` | `slsqp`, `lbfgs`, `nlopt_lbfgs`, `mma`, `bfgs`, `bobyqa`, `trust_region` | `slsqp` | Optimization algorithm |
| `inner_maxiter` | integer | `200` | Max iterations for the inner (per-subject EBE) optimizer |
| `inner_tol` | float | `1e-4` | Gradient-norm convergence tolerance for the inner (per-subject EBE) optimizer. The default of `1e-4` matches the precision of typical NLME engines (NONMEM's default inner-loop SIGDIGITS is ~3, equivalent to ~`1e-3`). Tighter values (e.g. `1e-6`, `1e-8`) over-converge the EBE relative to the Sheiner–Beal linearisation error and can slow FOCEI fits by 10–15× without measurable change in the final OFV. Use a tighter value only if you're comparing post-hoc EBE values across runs at high precision. |
| `steihaug_max_iters` | integer | `50` | Max CG iterations for the Steihaug subproblem (only used when `optimizer = trust_region`) |
| `global_search` | `true`, `false` | `false` | Run NLopt CRS2-LM (Controlled Random Search with Local Mutation) as a gradient-free global pre-search before the local optimizer. CRS2-LM samples within the parameter bounds; the local optimizer (e.g. `bobyqa`, `slsqp`) starts from the best point found. Useful for poorly-identified models — when the local optimizer can land in a degenerate basin (collapsed ETA, V/Q swap, parameters at bounds) from a far-from-truth start, the global pre-search usually escapes it. Adds the pre-search budget on top of the local optimisation, but typically more efficient than running multiple full fits from scratch. Requires a full NLopt build (e.g. `brew install nlopt` or `apt install libnlopt-dev`); a clear warning is emitted if CRS2-LM is unavailable. |
| `global_maxeval` | integer | `200 * (n_params + 1)` | Maximum evaluations of the FOCE objective during the global pre-search. Each eval is a full inner-loop pass over all subjects, so this is the dominant cost of `global_search = true`. The default (`0` → auto) is empirically enough to escape bad basins on 10–20 parameter PK models without dominating the wall time of the subsequent local refine. |
| `bloq_method` | `drop`, `m3` | `drop` | How to handle rows with `CENS=1`. `m3` enables Beal's M3 likelihood (see [BLOQ example](../examples/bloq.md)). |
| `mu_referencing` | `true`, `false` | `true` | Re-centre inner-loop ETA estimates on the current population mean (auto-detected from `[individual_parameters]`). See the [FAQ entry](../faq.md#do-i-need-to-use-mu-referencing-in-my-model-definitions-like-in-nonmem--nlmixr2) for details. Set `false` to reproduce pre-automatic-mu behaviour. |
| `iov_column` | string | — | Name of the occasion column in the dataset (e.g. `OCC`). Required when the model uses `kappa` or `block_kappa` declarations. The column must contain integer occasion indices. Case-insensitive. Only supported with `foce` / `focei` — not `saem`. See [IOV documentation](../estimation/iov.md). |
| `optimizer_trace` | `true`, `false` | `false` | Write a per-iteration CSV to `/tmp/ferx_trace_<pid>_<ts>.csv`. The path is stored in `FitResult::trace_path`. Useful for diagnosing convergence problems or comparing optimizers. See [Optimizer Trace](#optimizer-trace). |

## Estimation Methods

### FOCEI (default)
```
method = focei
```
FOCE with Interaction. Includes the dependence of the residual error on random effects. More accurate than FOCE when the error model depends on individual predictions, but slightly slower.

### FOCE
```
method = foce
```
First-Order Conditional Estimation. Linearizes the model around the empirical Bayes estimates. Fast and reliable for most models.

### SAEM
```
method = saem
```
Stochastic Approximation EM. Uses Metropolis-Hastings sampling instead of MAP optimization for random effects. More robust to local minima; recommended for complex models with many random effects.

## SAEM-Specific Options

| Key | Default | Description |
|-----|---------|-------------|
| `n_exploration` | `150` | Phase 1 iterations (step size = 1) |
| `n_convergence` | `250` | Phase 2 iterations (step size = 1/k) |
| `n_mh_steps` | `3` | Metropolis-Hastings steps per subject per iteration |
| `adapt_interval` | `50` | Iterations between MH step-size adaptation |
| `seed` | `12345` | RNG seed for reproducibility |

## SIR (Sampling Importance Resampling)

SIR provides non-parametric parameter uncertainty estimates as an optional post-estimation step. Requires `covariance = true`.

| Key | Default | Description |
|-----|---------|-------------|
| `sir` | `false` | Enable SIR uncertainty estimation |
| `sir_samples` | `1000` | Number of proposal samples (M) |
| `sir_resamples` | `250` | Number of resampled vectors (m) |
| `sir_seed` | `12345` | RNG seed for reproducibility |
| `sir_keep_samples` | `false` | Retain resampled parameter vectors for `simulate_with_uncertainty()` |
| `sir_df` | `5.0` | Degrees of freedom for the Student-t proposal; higher values approach a normal proposal |

See [SIR documentation](../estimation/sir.md) for details.

## Optimizer Choices

| Optimizer | Description | Recommended For |
|-----------|-------------|-----------------|
| `slsqp` | Sequential Least Squares Programming (NLopt) | General use (default) |
| `bfgs` | Built-in BFGS quasi-Newton | When NLopt is unavailable |
| `lbfgs` | Limited-memory BFGS | Large parameter spaces |
| `nlopt_lbfgs` | NLopt L-BFGS | Alternative L-BFGS |
| `mma` | Method of Moving Asymptotes (NLopt) | Constrained problems |
| `bobyqa` | NLopt BOBYQA — derivative-free quadratic interpolation | Noisy or non-smooth objectives where FD gradients are unreliable |
| `trust_region` | Newton trust-region with Steihaug CG subproblem (argmin) | Well-conditioned problems where second-order curvature helps convergence; tune `steihaug_max_iters` |

Notes:
- `bobyqa` does not use gradients, so it is robust to small discontinuities in
  the FOCE surface caused by EBE re-estimation, but it converges more slowly
  than gradient-based methods on smooth problems.
- `trust_region` uses a finite-difference Hessian of the OFV-at-fixed-EBEs.
  Each Hessian costs O(n²) OFV evaluations, so it is fastest when the number
  of packed parameters is small. Increase `steihaug_max_iters` when the
  parameter count exceeds the default of 50.

## Parameter Scaling and EBE Convergence

| Key | Default | Description |
|-----|---------|-------------|
| `scale_params` | `true` | Scale packed parameters to O(1) before passing to the optimizer. Improves conditioning when log-transformed parameters span several orders of magnitude. Set to `false` to disable (reproduces pre-PR2 behaviour). |
| `max_unconverged_frac` | `0.1` | Fraction of subjects (with at least `min_obs_for_convergence_check` observations) allowed to have unconverged EBEs before the outer optimizer rejects the step (returns OFV = ∞). Set to `1.0` to disable the guard. |
| `min_obs_for_convergence_check` | `2` | Subjects with fewer than this many observations are excluded from the `max_unconverged_frac` check (they still run normally). |

## Options That Don't Apply to the Selected Method

If you set an option that the chosen estimation method doesn't consume
(e.g. `n_convergence` with `method = focei`, or `optimizer` with
`method = saem`), `fit()` emits a warning listing the option, the selected
method, and the keys that *are* available for that method. The option is
ignored — the fit still proceeds.

For a chained fit (`method = [saem, focei]`), an option is kept if it applies
to *any* stage in the chain, so SAEM and FOCEI keys can be mixed without
warnings.

## Global Search

Setting `global_search = true` runs a gradient-free pre-search (NLopt CRS2-LM) before the local optimizer. This helps escape local minima on challenging datasets.

The number of global evaluations is auto-scaled based on the number of parameters and observations, or can be set explicitly with `global_maxeval`.

## Examples

Standard FOCEI with defaults:
```
[fit_options]
  method     = focei
  maxiter    = 300
  covariance = true
```

FOCEI with global search:
```
[fit_options]
  method        = focei
  maxiter       = 500
  covariance    = true
  global_search = true
```

SAEM with custom settings:
```
[fit_options]
  method        = saem
  n_exploration = 200
  n_convergence = 300
  n_mh_steps    = 5
  seed          = 42
  covariance    = true
```

FOCEI with SIR uncertainty:
```
[fit_options]
  method        = focei
  covariance    = true
  sir           = true
  sir_samples   = 1000
  sir_resamples = 250
  sir_seed      = 42
```

Derivative-free BOBYQA fit:
```
[fit_options]
  method        = foce
  optimizer     = bobyqa
  maxiter       = 300
  inner_maxiter = 100
  inner_tol     = 1e-6
```

Trust-region with tuned CG subproblem:
```
[fit_options]
  method             = foce
  optimizer          = trust_region
  maxiter            = 200
  steihaug_max_iters = 30
```

FOCE with Inter-Occasion Variability:
```
[fit_options]
  method     = foce
  iov_column = OCC
  covariance = true
```

Enable optimizer trace and EBE guard:
```
[fit_options]
  method                        = foce
  optimizer_trace               = true
  max_unconverged_frac          = 0.5
  min_obs_for_convergence_check = 3
```

## Optimizer Trace

When `optimizer_trace = true`, a CSV is written to `/tmp/ferx_trace_<pid>_<ts>.csv` and the path is stored in `FitResult::trace_path`. Each row is one outer iteration.

| Column | Populated by | Description |
|--------|-------------|-------------|
| `iter` | all | Iteration number |
| `method` | all | `foce`, `focei`, `gn`, `gn_hybrid`, `saem` |
| `phase` | gn_hybrid, saem | `focei` (polish) or `explore`/`converge` |
| `ofv` | all | Objective function value |
| `wall_ms` | all | Wall time for this iteration (ms) |
| `grad_norm` | BFGS, NLopt gradient-mode | Gradient norm |
| `step_norm` | BFGS | Step size |
| `inner_iter_count` | (reserved) | Reserved for future per-iteration inner-loop count; currently `NA` |
| `optimizer` | FOCE/FOCEI | Active NLopt algorithm |
| `lm_lambda` | GN | Levenberg-Marquardt damping factor |
| `ofv_delta` | GN | Change in OFV from previous iteration |
| `step_accepted` | GN | Whether the GN step was accepted |
| `cond_nll` | SAEM | Conditional observation NLL |
| `gamma` | SAEM | SAEM step-size (1 in exploration, 1/k in convergence) |
| `mh_accept_rate` | SAEM | Mean Metropolis-Hastings acceptance rate across subjects |
| `n_ebe_unconverged` | FOCE/FOCEI | Subjects whose inner optimizer did not converge |
| `n_ebe_fallback` | FOCE/FOCEI | Subjects that fell back to Nelder-Mead |

Unused columns contain `NA`. The trace is buffered and flushed when the fit ends; if a run is killed mid-iteration the buffered tail may be lost.
