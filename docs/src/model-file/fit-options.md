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
| `method` | `foce`, `focei`, `saem`, `imp` (only as final chain stage) | `focei` | Estimation method (or single-stage method in a chain). See [chained fits](#chained-fits). |
| `maxiter` | integer | `500` | Maximum outer loop iterations |
| `covariance` | `true`, `false` | `true` | Compute covariance matrix and standard errors |
| `covariance_fallback` | `none`, `sir` | `none` | What to do when the FD Hessian is not positive definite (non-PD). `none` leaves the covariance step as failed. `sir` runs SIR with a fallback proposal covariance built from the `\|eigenvalue\|`-rectified Hessian, inflated 4×; the YAML reports SIR-based 95% CIs instead of Hessian SEs. Requires `covariance = true`. Ignored when the Hessian succeeds. |
| `fd_hessian_step` | float | `1e-2` | Initial relative step size for the finite-difference Hessian used in the covariance step. The actual perturbation for parameter `i` is `fd_hessian_step * (1 + |x_hat[i]|)`. ferx automatically halves this value up to 8 times if any diagonal stencil evaluates to a non-finite value, so in most cases the default requires no tuning. Set explicitly only if the automatic reduction is insufficient (try `0.1`) or if FD noise is the main concern on a very smooth surface (try `1e-3`). Must be positive and finite. |
| `optimizer` | `bobyqa`, `slsqp`, `bfgs`, `lbfgs`, `nlopt_lbfgs`, `mma`, `trust_region` | `bobyqa` | Outer-loop optimisation algorithm. Applies to `method = foce` / `focei` and to the FOCEI polish phase of `method = gn_hybrid`. Ignored (with a runtime warning) under `method = saem` / `imp` / `gn` — see the [Outer Optimizers](../estimation/optimizers.md) page for the applicability matrix and the rationale for the BOBYQA default. |
| `inner_maxiter` | integer | `200` | Max iterations for the inner (per-subject EBE) optimizer |
| `inner_tol` | float | `1e-4` | Gradient-norm convergence tolerance for the inner (per-subject EBE) optimizer. The default of `1e-4` matches the precision of typical NLME engines (NONMEM's default inner-loop SIGDIGITS is ~3, equivalent to ~`1e-3`). Tighter values (e.g. `1e-6`, `1e-8`) over-converge the EBE relative to the Sheiner–Beal linearisation error and can slow FOCEI fits by 10–15× without measurable change in the final OFV. Use a tighter value only if you're comparing post-hoc EBE values across runs at high precision. |
| `steihaug_max_iters` | integer | adaptive | Max CG iterations for the Steihaug subproblem (only used when `optimizer = trust_region`). Default (unset) uses `ceil(sqrt(n_params)).clamp(5, n_params)` — typically 5 for standard NLME models. Set explicitly (e.g. `steihaug_max_iters = 50`) to pin the budget. |
| `global_search` | `true`, `false` | `false` | Run NLopt CRS2-LM (Controlled Random Search with Local Mutation) as a gradient-free global pre-search before the local optimizer. CRS2-LM samples within the parameter bounds; the local optimizer (e.g. `bobyqa`, `slsqp`) starts from the best point found. Useful for poorly-identified models — when the local optimizer can land in a degenerate basin (collapsed ETA, V/Q swap, parameters at bounds) from a far-from-truth start, the global pre-search usually escapes it. Adds the pre-search budget on top of the local optimisation, but typically more efficient than running multiple full fits from scratch. Requires a full NLopt build (e.g. `brew install nlopt` or `apt install libnlopt-dev`); a clear warning is emitted if CRS2-LM is unavailable. |
| `global_maxeval` | integer | `200 * (n_params + 1)` | Maximum evaluations of the FOCE objective during the global pre-search. Each eval is a full inner-loop pass over all subjects, so this is the dominant cost of `global_search = true`. The default (`0` → auto) is empirically enough to escape bad basins on 10–20 parameter PK models without dominating the wall time of the subsequent local refine. |
| `bloq_method` | `drop`, `m3` | `drop` | How to handle rows with `CENS=1`. `m3` enables Beal's M3 likelihood (see [BLOQ example](../examples/bloq.md)). |
| `mu_referencing` | `true`, `false` | `true` | Re-centre inner-loop ETA estimates on the current population mean (auto-detected from `[individual_parameters]`). See the [FAQ entry](../faq.md#do-i-need-to-use-mu-referencing-in-my-model-definitions-like-in-nonmem--nlmixr2) for details. Set `false` to reproduce pre-automatic-mu behaviour. |
| `iov_column` | string | — | Name of the occasion column in the dataset (e.g. `OCC`). Required when the model uses `kappa` or `block_kappa` declarations. The column must contain integer occasion indices. Case-insensitive. Only supported with `foce` / `focei` — not `saem`. See [IOV documentation](iov.md). |
| `optimizer_trace` | `true`, `false` | `false` | Write a per-iteration CSV to `/tmp/ferx_trace_<pid>_<ts>.csv`. The path is stored in `FitResult::trace_path`. Useful for diagnosing convergence problems or comparing optimizers. See [Optimizer Trace](#optimizer-trace). |
| `reconverge_gradient_interval` | integer ≥ 0 | `0` | How often to re-solve each subject's inner EBE loop during the population gradient instead of holding the EBEs (η̂) and FOCE Hessian fixed. `0` (default) never reconverges — the cheap fixed-EBE gradient is used. `1` reconverges on every gradient evaluation; `N` reconverges on evaluations `0, N, 2N, …` and uses the cheap gradient in between (amortizing the cost by ~`N`× while still periodically correcting the search direction). The fixed-EBE gradient omits the inner solution's response to θ/Ω, so on ill-conditioned fits a gradient optimizer (e.g. `slsqp`) can stall well above the derivative-free (`bobyqa`) optimum; reconverging recovers the full surface at roughly **5–6× the per-gradient cost** at interval `1`. IOV models (`kappa`/`block_kappa`) always reconverge and ignore this setting. See the [FOCE/FOCEI page](../estimation/foce.md#gradient-accuracy-vs-cost). |
| `inits_from_nca` | `true`, `false`, `nca`, `nca_sweep`, `nca_ebe` | `false` | Derive NCA-based starting values from the data before the optimizer loop. `true` (alias for `nca_sweep`) and `false`/`off` toggle the default strategy; the three named values pick a strategy explicitly (see [NCA-based starting values](#nca-based-starting-values)). Fixed thetas are never overwritten; covariate-effect thetas (no mu-referencing link) keep the model default. Most useful with `method = trust_region` or `method = gn` / `method = gn_hybrid`, where bad starting values can cause early stalling. The same estimation is available without running a fit via the CLI flag `--inits-from-nca[=METHOD]` and (in ferx-r) the `ferx_inits_from_nca()` function. |

## NCA-based starting values

`inits_from_nca` estimates starting values directly from the data using
non-compartmental analysis (NCA), then optionally refines parameters NCA cannot
estimate. All three strategies are NCA-based; they differ only in how much
refinement runs on top:

| Value | Strategy | What it does | Typical cost |
|-------|----------|--------------|--------------|
| `nca` | NCA only | Per-subject NCA (AUC, terminal slope, Wagner–Nelson Ka, biexponential peeling for 2/3-cpt) pooled to population geometric means. Leaves parameters NCA can't reach (peripheral Q/V2, all ODE/PD thetas) at the model default. | < 5 ms |
| `nca_sweep` | NCA + sweep | Runs `nca`, then sweeps every remaining non-fixed theta over a log-space rRMSE grid using population predictions (etas = 0). Model-agnostic — also covers ODE/PD models. **This is what `true` selects.** | < 500 ms (analytical) |
| `nca_ebe` | NCA + EBE sweep | Like `nca_sweep` but evaluates the grid with empirical Bayes estimates (etas ≠ 0); more accurate under large IIV (omega > ~0.2). Falls back to `nca_sweep` for ODE models. | < 500 ms (analytical) |

The CL eta's omega is also updated from inter-subject CV² when ≥ 3 subjects have
a valid NCA estimate.

When `nca_sweep` is enabled but the fit fails to converge or the OFV looks
suspiciously high, try `nca_ebe`.

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
| `n_mh_steps` | `20` | Block Metropolis-Hastings steps per subject per iteration. Also sizes the componentwise decorrelating kernel that prevents block-Ω collapse (`max(2, n_mh_steps / n_eta)` sweeps; multi-η models only — skipped when `n_eta < 2`). When `n_leapfrog > 0`, this applies to subjects that fall back to MH (see below); HMC subjects use one proposal per iteration regardless. |
| `n_leapfrog` | `0` | Leapfrog steps per HMC proposal (0 = use MH; see below). When > 0, subjects for which HMC is unavailable (ODE model, missing analytical PK path, non-finite Ω, unsupported TV-cov path) fall back to MH using `n_mh_steps` proposals. |
| `adapt_interval` | `50` | Iterations between step-size adaptation |
| `omega_burnin` | `20` | Initial exploration iterations during which Ω (and Ω<sub>IOV</sub>) are held at their starting values while the MH chain warms up. Clamped to `n_exploration`; set `0` to disable. Prevents the Ω collapse described in the SAEM page. |
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

## Importance Sampling (IMP)

The `imp` stage estimates the marginal log-likelihood by Monte-Carlo
importance sampling, giving a lower-bias `−2 log L` than the FOCE/Laplace
OFV when subject posteriors of η are non-Gaussian (e.g. sparsely-sampled
PK). Use it as a final chain stage:

```
[fit_options]
  method        = [focei, imp]
  is_samples    = 1000
  is_proposal_df = 5
  is_seed       = 12345
```

| Key | Default | Description |
|-----|---------|-------------|
| `is_samples` | `1000` | Importance samples K per subject. 2000–5000 recommended for publication-quality MC SE. |
| `is_proposal_df` | `5.0` | Student-t proposal degrees of freedom (≥ 1). Lower = heavier tails. |
| `is_seed` | `42` | RNG seed. Same seed → identical `−2 log L`. |
| `is_low_ess_threshold` | `0.1` | Subjects with normalized ESS below this fraction get flagged in the result. Set `0` to silence. |

See [Importance Sampling documentation](../estimation/importance-sampling.md)
for the algorithm, IOV caveats, and tuning guidance.

## Optimizer Choices

| Optimizer | Description | Recommended For |
|-----------|-------------|-----------------|
| `bobyqa` | NLopt BOBYQA — derivative-free quadratic interpolation | General use (default); robust on noisy / non-smooth FOCE surfaces (ODE/PD, sparse data, Hill-ridge models) |
| `slsqp` | Sequential Least Squares Programming (NLopt) | Smooth, well-conditioned analytical PK models where you want gradient-based convergence; pair with `reconverge_gradient_interval` if it stalls |
| `bfgs` | Built-in BFGS quasi-Newton | When NLopt is unavailable |
| `lbfgs` | Limited-memory BFGS | Large parameter spaces |
| `nlopt_lbfgs` | NLopt L-BFGS | Alternative L-BFGS |
| `mma` | Method of Moving Asymptotes (NLopt) | Constrained problems |
| `trust_region` | Newton trust-region with Steihaug CG subproblem (argmin) | Well-conditioned problems where second-order curvature helps convergence |

Notes:
- `bobyqa` does not use gradients, so it is robust to small discontinuities in
  the FOCE surface caused by EBE re-estimation, but it converges more slowly
  than gradient-based methods on smooth problems.
- `trust_region` uses an AD-based gradient (same `subject_nll_pop_grad` as the
  outer FOCE optimizer) and a BHHH approximate Hessian (`H ≈ 4 Σ gᵢgᵢᵀ`).
  The BHHH matrix is always positive semi-definite, so the Steihaug subproblem
  is well-conditioned even near constraints. The Steihaug CG budget defaults to
  `ceil(sqrt(n_params))` — typically 5 for standard NLME models, which is far
  cheaper than the previous FD-Hessian path (O(n²) OFV evaluations per Hessian).

## Parameter Scaling and EBE Convergence

| Key | Default | Description |
|-----|---------|-------------|
| `scale_params` | `false` | Divide each packed (log/Cholesky) coordinate by its initial magnitude before passing it to the optimizer. **Off by default since issue #99:** the scaling-enabled path only ever runs on log/Cholesky coordinates, where dividing by `\|log value\|` is counterproductive — e.g. `ln(V)=ln(20)≈3` gets scale 3, turning the optimizer's unit step into a ≈20× multiplicative jump in V, which overshoots and (via the uniform SLSQP gradient cap) starves the step in other dimensions such as OMEGA, halting short of the minimum. The OFV value is unchanged at any fixed point, but the optimizer *trajectory* and stop point are not. Set to `true` only for experimentation. |
| `max_unconverged_frac` | `0.1` | Fraction of subjects (with at least `min_obs_for_convergence_check` observations) allowed to have unconverged EBEs before the outer optimizer rejects the step (returns OFV = ∞). Set to `1.0` to disable the guard. |
| `min_obs_for_convergence_check` | `2` | Subjects with fewer than this many observations are excluded from the `max_unconverged_frac` check (they still run normally). |
| `stagnation_guard` | `true` | Short-circuit the NLopt-based outer optimizers once recent evals show no OFV improvement above 1e-3 over a window of `3*(n+1).max(50)` evals. This lets SLSQP / L-BFGS terminate quickly via their own xtol/ftol on numerically-flat plateaus (e.g. γ-bearing FOCEI problems) instead of grinding through the remaining `outer_maxiter` budget at full inner-loop cost. Set to `false` to let the optimizer run to its natural termination criterion — useful when debugging or for problems with very slow but real OFV improvements below the threshold. |

## Options That Don't Apply to the Selected Method

If you set an option that the chosen estimation method doesn't consume
(e.g. `n_convergence` with `method = focei`, or `optimizer` with
`method = saem`), `fit()` emits a warning listing the option, the selected
method, and the keys that *are* available for that method. The option is
ignored — the fit still proceeds.

For a chained fit (`method = [saem, focei]`), an option is kept if it applies
to *any* stage in the chain, so SAEM and FOCEI keys can be mixed without
warnings.

## Multi-Start Optimization

| Key | Default | Description |
|-----|---------|-------------|
| `n_starts` | `1` | Number of independent optimization runs. `1` disables multi-start (no behaviour change). When `> 1`, all starts run in parallel via rayon; the converged run with the lowest OFV is returned. Start 0 always uses the exact initial values from the model file. |
| `start_sigma` | `0.3` | Log-space perturbation applied to initial theta values for starts 1..n. Log-packed thetas are multiplied by `exp(N(0, start_sigma))`; thetas with negative lower bounds are shifted additively. |
| `multi_start_seed` | `42` | RNG seed for the multi-start theta perturbations. Independent of `seed` (SAEM) so that changing the SAEM seed does not silently alter which perturbed starting points are used in FOCE multi-start runs. |

Multi-start is most useful for models prone to local minima: nonlinear elimination (Michaelis-Menten), full-block omega, or many covariate parameters. On an 8-core machine `n_starts = 8` costs the same wall-clock time as a single run but has ~8× lower probability of a local minimum.

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
| `mh_accept_rate` | SAEM | Mean acceptance rate across all subjects (MH or HMC). In mixed HMC/MH runs (`n_leapfrog > 0` with some MH-fallback subjects) this is an aggregate across both samplers. |
| `n_ebe_unconverged` | FOCE/FOCEI | Subjects whose inner optimizer did not converge |
| `n_ebe_fallback` | FOCE/FOCEI | Subjects that fell back to Nelder-Mead |

Unused columns contain `NA`. The trace is buffered and flushed when the fit ends; if a run is killed mid-iteration the buffered tail may be lost.
