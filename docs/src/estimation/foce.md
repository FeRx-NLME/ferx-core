# FOCE / FOCEI

> **Maturity: stable** — see [Feature Maturity](../maturity.md) for what this means.

First-Order Conditional Estimation (FOCE) and its interaction variant (FOCEI) are the primary estimation methods in ferx-core. They use a two-level nested optimization to find maximum likelihood estimates of population parameters.

## Algorithm Overview

### Outer Loop (Population Parameters)

The outer loop optimizes the population parameters \\( \theta \\), \\( \Omega \\), and \\( \sigma \\) by minimizing the population objective function value (OFV):

\\[ \text{OFV} = -2 \log L = \sum_{i=1}^{N} \text{OFV}_i \\]

Parameters are internally transformed for unconstrained optimization:
- **Theta**: Log-transformed (ensures positivity)
- **Omega**: Cholesky factorization (ensures positive-definiteness)
- **Sigma**: Log-transformed (ensures positivity)

### Inner Loop (Empirical Bayes Estimates)

For each subject, the inner loop finds the empirical Bayes estimate (EBE) of the random effects \\( \hat{\eta}_i \\) by minimizing the individual negative log-likelihood:

\\[ -\log p(\eta_i | y_i, \theta, \Omega, \sigma) = \frac{1}{2} \left[ \eta_i^T \Omega^{-1} \eta_i + \log|\Omega| + \sum_j \left( \frac{(y_{ij} - f_{ij})^2}{V_{ij}} + \log V_{ij} \right) \right] \\]

The inner loop uses BFGS, falling back to Nelder-Mead simplex if BFGS fails.

### Gradient route (AD vs FD)

The BFGS gradient is computed by reverse-mode **automatic differentiation (AD)** when available, and by central **finite differences (FD)** otherwise. AD requires the crate built with the `autodiff` feature (the Enzyme toolchain) *and* an analytical PK model; it is resolved **per subject**, so individual subjects fall back to FD for ODE models, steady-state (SS) doses, system resets (EVID 3/4), or time-varying covariates on a model the event-driven AD path doesn't yet support. AD is the default (`gradient_method = auto`) and is typically on par with FD on small models and faster as the parameter count grows.

The startup banner reports the route **actually resolved** across the population — not just the requested setting — so a silent AD→FD fallback is visible:

```
  gradient: AD (single-snapshot)  [requested: auto]
```

When the population splits across routes, the banner shows per-route subject counts, e.g. `AD (event-driven) ×118, FD ×3`. A build without the `autodiff` feature always reads `FD  [requested: auto; autodiff not compiled in]`.

This `gradient:` line appears for gradient-driven estimators (FOCE/FOCEI/GN) and for `imp`, which reuses the EBE Hessian built via the same route. SAEM is sampling-based and reports its E-step kernel on a `sampler:` line instead — see [SAEM](saem.md).

If the EBE search wanders into a region where the individual NLL evaluates to a non-finite value (for example, an ODE model whose integration blows up at extreme \\( \eta \\)), that point is treated as the worst possible objective rather than aborting the fit. The subject is reported as non-converged and estimation continues for the remaining subjects.

### Gradient accuracy vs cost (reconverged EBEs)

By default the population gradient holds each subject's EBEs (\\( \hat\eta \\)) and FOCE Hessian fixed while perturbing the population parameters — the *fixed-EBE* gradient. This is cheap, but it omits the response of the inner solution to \\( \theta \\) and \\( \Omega \\). On well-conditioned problems that omitted term is negligible and a gradient optimizer matches the derivative-free `bobyqa` (the default; see [Outer Optimizers](optimizers.md)). On **ill-conditioned** problems it is not: the omitted term is what separates a true descent direction from a flat-looking plateau, and `slsqp` can report `converged` at an OFV far above the `bobyqa` optimum — which is exactly why `bobyqa` is the default.

Set `reconverge_gradient_interval = 1` (in `[fit_options]`) to re-solve the inner EBE loop at every finite-difference perturbation, recovering the full surface. This costs roughly **5–6×** per gradient — reserve it for fits whose OFV looks suspiciously high. IOV models (`kappa`/`block_kappa`) always reconverge, so the setting is a no-op there; it only changes non-IOV fits.

To amortize that cost, `reconverge_gradient_interval = N` reconverges only every `N`-th gradient evaluation and uses the cheap fixed-EBE gradient in between — the periodic correction is often enough to keep the optimizer off the plateau at a fraction of the always-on cost. The default `0` disables reconverging entirely (cheap fixed-EBE gradient).

**Validation** — cefepime pediatric population PK, 2-compartment IV infusion, combined error, no IOV (5937 subjects / 17380 observations). All ferx rows use FOCEI and the same likelihood convention, so OFVs are directly comparable:

| Configuration | OFV | Wall time |
|---|---:|---:|
| `slsqp` (fixed-EBE gradient) | 68,252 | 390 s |
| `slsqp` + reconverge every 10 (`interval = 10`) | 66,118 | 633 s |
| `slsqp` + reconverge every 5 (`interval = 5`) | 66,056 | 1,004 s |
| `slsqp` + reconverge every eval (`interval = 1`) | **65,485** | 1,871 s |
| `bobyqa` (derivative-free, **default**) | 65,598 | 315 s |
| ferx OFV evaluated at NONMEM's final estimates | 67,514 | — |

The fixed-EBE gradient stalls `slsqp` ~2,650 OFV units above `bobyqa`. Reconverging the EBEs on every gradient evaluation closes the entire gap and reaches an optimum marginally below `bobyqa`'s — and below the point NONMEM converged to (NONMEM's estimates score 67,514 under ferx's likelihood), confirming the stall was a gradient-bias artifact, not a worse model.

A larger `reconverge_gradient_interval` trades that accuracy back for speed: reconverging every 5th or 10th gradient still closes ~80% of the stall, but the OFV degrades monotonically as the interval grows (the cheap fixed-EBE gradients in between bias the direction enough that `slsqp` declares convergence slightly early). On this problem every interval setting is dominated by `bobyqa` on *both* OFV and wall time — which is why `bobyqa` is the default. The reconverged-`slsqp` path earns its keep when a gradient optimizer is required (e.g. parameter count high enough that derivative-free search scales poorly, or downstream tooling that consumes the optimizer's gradient).

## FOCE vs FOCEI

### Standard FOCE

Uses linearized predictions around the EBEs:

\\[ f_0 = f(\hat{\eta}) - H \hat{\eta} \\]

where \\( H \\) is the Jacobian matrix \\( \partial f / \partial \eta \\). The per-subject objective is:

\\[ \text{OFV}_i = (y - f_0)^T \tilde{R}^{-1} (y - f_0) + \log|\tilde{R}| \\]

where \\( \tilde{R} = H \Omega H^T + R(f_0) \\).

### FOCEI (Interaction)

Uses individual predictions directly without linearization. ferx-core implements the Laplace approximation form from Almquist et al. (2015):

\\[ \text{OFV}_i = (y - \hat{f})^T V^{-1} (y - \hat{f}) + \hat{\eta}^T \Omega^{-1} \hat{\eta} + \log|\tilde{R}| \\]

FOCEI is more accurate when the residual variance depends on the predicted value (proportional or combined error models), because it captures the interaction between random effects and residual error.

> Almquist, J., et al. (2015). *Comparison of maximum a posteriori and conditional likelihood estimation in the FOCE method.* PAGE 24, Abstr 3516.

## Optimizer Options

The outer optimizer is configured independently of the estimation method. See **[Outer Optimizers](optimizers.md)** for a full description of all available algorithms (SLSQP, BOBYQA, trust-region, BFGS, L-BFGS, MMA) and guidance on when to use each.

Set via `[fit_options]`:

```
[fit_options]
  method    = focei
  optimizer = slsqp    # override the default (bobyqa)
```

## Global Search

Enable gradient-free pre-search to help escape local minima:

```
[fit_options]
  method        = foce
  global_search = true
  global_maxeval = 2000
```

The pre-search uses NLopt's CRS2-LM algorithm (Controlled Random Search) to explore the parameter space before handing off to the local optimizer. The number of evaluations auto-scales with model complexity if `global_maxeval` is not set.

## Covariance Step

When `covariance = true`, ferx-core computes the variance-covariance matrix of the parameter estimates (the **R-matrix**: the inverse observed Fisher information) at the converged solution. This provides:

- **Standard errors (SE)** for all parameters
- **Relative standard errors (%RSE)** for assessing estimation precision
- **Omega SEs** via delta method from the Cholesky parameterization

### How the Hessian is built

The covariance is `2·H⁻¹`, where `H` is the finite-difference Hessian of the objective. Two details make the SEs match NONMEM's `$COVARIANCE`:

1. **The EBEs are reconverged at every perturbed point.** Like NONMEM's covariance step, ferx re-solves the inner conditional-estimation loop (warm-started from the converged EBEs) at each finite-difference point. Holding the EBEs fixed omits the response of the conditional optimum to the population parameters, which gives a Hessian with the wrong curvature — indefinite even on well-conditioned surfaces such as warfarin — and was the source of the spurious eigenvalue clipping in issue [#129](https://github.com/FeRx-NLME/ferx-core/issues/129).
2. **The Hessian is a central difference of the analytical gradient** (issue [#209](https://github.com/FeRx-NLME/ferx-core/issues/209)): `H[:,k] ≈ (g(x̂+hₖeₖ) − g(x̂−hₖeₖ)) / 2hₖ`. This is `2·n_free` gradient evaluations instead of `~2·n_free²` objective evaluations, and it avoids the catastrophic cancellation of a four-point second difference of the OFV. The θ part of that gradient reuses H-matrix columns for mu-referenced parameters (issue [#196](https://github.com/FeRx-NLME/ferx-core/issues/196)). IOV models, whose κ block has no fixed-EBE analytical gradient, fall back to a second difference of the reconverged objective.

The factor of two is because the objective is `−2·logL`: its Hessian is twice the observed information, so the covariance needs `2·H⁻¹` to be on the right scale.

The covariance objective is `2·pop_nll` for **both** FOCE and FOCEI — the Ω penalty is already inside each per-subject marginal and must not be added again. FOCEI's Almquist–Laplace marginal carries `η̂ᵀΩ⁻¹η̂ + log|Ω|` explicitly; FOCE's Sheiner–Beal marginal carries it through `R̃ = HΩHᵀ + R` (equivalent by the Woodbury identity). An earlier version added the Ω prior a second time on the FOCE path, double-counting Ω and under-stating the FOCE omega SEs by ~31% — fixed in issue [#243](https://github.com/FeRx-NLME/ferx-core/issues/243).

For a mixed block + diagonal Ω, the structural-zero cross-block off-diagonals (which are not estimated) are excluded from the covariance parameter set like FIX parameters; otherwise their flat Hessian diagonal aborts the step. (The same #243 change exposed and fixed this for both FOCE and FOCEI.)

### NONMEM cross-check

On `data/warfarin.csv` (10 subjects, 1-cpt oral, proportional error) against NONMEM 7.5.1 `$COVARIANCE MATRIX=R`, **FOCEI** (`$EST METHOD=1 INTER`):

| Parameter | ferx SE | NONMEM SE | rel. diff |
|-----------|---------|-----------|-----------|
| TVCL      | 0.00712 | 0.00710   | +0.3% |
| TVV       | 0.2350  | 0.2401    | −2.1% |
| TVKA      | 0.1506  | 0.1486    | +1.3% |
| PROP_ERR (SD) | 0.000832 | 0.000835 | −0.4% |
| ω²(CL)    | 0.01291 | 0.01279   | +0.9% |
| ω²(V)     | 0.00403 | 0.00431   | −6.4% |
| ω²(KA)    | 0.1530  | 0.1504    | +1.8% |

and **FOCE** (`$EST METHOD=1`, no `INTER`), where ferx's OFV/estimates already match NONMEM (OFV −280.17 vs −280.36) and, after the #243 covariance fix, the omega SEs do too:

| Parameter | ferx SE | NONMEM SE | rel. diff |
|-----------|---------|-----------|-----------|
| TVCL      | 0.00661 | 0.00663   | −0.3% |
| TVV       | 0.2375  | 0.2340    | +1.5% |
| TVKA      | 0.1209  | 0.1245    | −2.8% |
| PROP_ERR (SD) | 0.000922 | 0.000941 | −2.0% |
| ω²(CL)    | 0.01312 | 0.01280   | +2.5% |
| ω²(V)     | 0.00454 | 0.00430   | +5.7% |
| ω²(KA)    | 0.1510  | 0.1607    | −6.1% |

(autodiff build; the FD-Jacobian build agrees to within ~10% on the ω block.) Both methods are guarded by `tests/warfarin_covariance_nonmem.rs` within a 20% band.

### Hessian regularization

Because the EBEs reconverge, the Hessian is positive-definite on well-conditioned surfaces and no regularization is needed. As a safety net for genuinely ill-conditioned or near-singular problems, ferx-core still inverts via a symmetric eigendecomposition and clips eigenvalues below `max(λ_max · 1e-10, 1e-12)` to that floor before reconstructing `H⁻¹`, adding a warning to `FitResult.warnings`:

```
Covariance step regularized: eigenvalue floor applied to FD Hessian
  (1 of 7 free-block eigenvalues clipped; min eig = -3.21e-08, floor = 4.56e-09).
Standard errors should be interpreted with care.
```

SEs in the regularised directions are inflated (since `1/λ_clipped` is larger than the would-be true `1/λ`). With the reconverging covariance step this warning should now be rare; if it fires, the surface is genuinely near-identifiable and the SEs in those directions warrant scrutiny. A Hessian whose spectrum is genuinely indefinite — every eigenvalue ≤ 0 — still fails outright with the standard `Covariance step failed` message.

### SIR fallback for non-PD Hessians

When a model is poorly identified or converges to a saddle point, the FD Hessian can have every free-block eigenvalue ≤ 0, making Hessian inversion impossible. Setting `covariance_fallback = sir` triggers a SIR (Sampling Importance Resampling) run instead of leaving the covariance step as failed:

```
[fit_options]
  covariance_fallback = sir
```

ferx-core constructs a proposal covariance by taking the absolute values of the Hessian eigenvalues, inflating by 4× (heavier tails to account for the non-PD correction), and running the same SIR sampler used by `sir = true`. The result is reported as 95% credible intervals in the fit YAML; `covariance_status` is set to `sir_fallback`.

Use this option when:
- The Hessian is non-PD at convergence due to model mis-specification or weak identifiability
- You want distributional uncertainty estimates without re-fitting with a more constrained model
- You need to proceed to a downstream analysis despite a non-PD Hessian

The SIR fallback inherits all `sir_*` settings (`sir_samples`, `sir_df`, `sir_resamples`, etc.) from `[fit_options]`.

## Convergence

The outer loop terminates when any of:
- The gradient norm falls below `outer_gtol` (default 1e-6)
- The maximum number of iterations (`maxiter`) is reached
- The optimizer reports convergence (NLopt `XtolReached` or `FtolReached`)

The inner loop terminates when the gradient norm falls below `inner_tol` (default `1e-4`, matching NONMEM's ~3 SIGDIGITS inner-loop precision — see [`fit-options.md`](../model-file/fit-options.md#general-options) for tuning guidance) or `inner_maxiter` (default 200) iterations are reached.

## Warm Starting

The inner loop warm-starts EBE estimation from the previous outer iteration's EBEs. This significantly reduces computation time, especially in later iterations when parameters change slowly.

## Configuration Example

```
[fit_options]
  method     = focei
  maxiter    = 500
  covariance = true
  # optimizer omitted → uses the default (bobyqa); set explicitly to switch
```
