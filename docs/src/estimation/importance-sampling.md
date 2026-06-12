# Importance Sampling (IMP)

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

The `imp` stage estimates the marginal log-likelihood

\\[
-2 \\log L(\\theta) \\;=\\; -2 \\sum_i \\log p(y_i \\mid \\theta)
\\;=\\; -2 \\sum_i \\log \\int p(y_i \\mid \\eta, \\theta)\\, p(\\eta \\mid \\theta)\\, d\\eta
\\]

by Monte-Carlo importance sampling. The IS Monte-Carlo estimator of
`p(yᵢ|θ)` is unbiased; the reported `−2 log L` carries a small Jensen
bias from the `log(·)` transform that vanishes as `K → ∞` and is
typically dominated by the much larger Laplace-approximation bias of
`ofv` in the sparse-data / strongly-nonlinear regime IMP targets.
Analogous to NONMEM's `$EST METHOD=IMP EONLY=1` and Monolix's
"Importance Sampling" likelihood method.

## When to use it

The Laplace approximation that produces the FOCE/FOCEI OFV assumes each
subject's posterior of η is Gaussian-shaped. That assumption is fine for
**well-sampled** PK (≥ 5–6 obs per subject across the elimination phase)
but breaks down for:

- **Sparse data** — e.g. routine TDM with 1–3 troughs per patient.
- **Strong nonlinearity** — Michaelis–Menten elimination,
  target-mediated drug disposition, transit-compartment absorption.
- **PD with categorical / binary endpoints** (likelihood
  surface non-quadratic in η).

In those regimes the FOCE OFV is biased — typically *under*-stated, so
naïve AIC/BIC comparisons favour over-parameterised models. IMP gives a
much-lower-bias estimate at extra MC cost.

## How it runs

`imp` is usually the terminal stage of a chain, but can also run on its own:

```
[fit_options]
  method        = [focei, imp]   # estimate with FOCEI, then evaluate the IS-LL
  is_samples    = 2000           # K, samples per subject
  is_proposal_df = 5             # ν, Student-t tail weight
  is_seed       = 12345
```

```
[fit_options]
  method = imp                   # standalone: IS-LL at the initial parameters
```

Rules:

- **Standalone is allowed.** `method = imp` evaluates the IS-LL at the
  **initial parameters** — IMP derives the EBEs and per-subject Jacobian it
  needs (the proposal centre/scale) at those parameters via a FOCE inner loop,
  rather than from a preceding estimator. It does not estimate the parameters;
  it reports the −2 log L there (handy for scoring imported/fixed parameter
  sets). When chained after an estimator (`[focei, imp]`), it uses that stage's
  EBEs/Jacobian instead.
- **Must appear at most once.**
- **Must be terminal.** `methods = [imp, focei]` (or any `imp` mid-chain) is
  rejected — a following stage would overwrite the parameters and make the IS
  result meaningless.

When IMP runs after an estimator, `FitResult.method` reports that estimating
stage (e.g. `FOCEI`); for a standalone `imp` it reports `IMP`. Either way
`FitResult.method_chain` preserves the full chain and the IS-LL lands on
`FitResult.importance_sampling`.

## Algorithm

For each subject *i* with EBE η̂ᵢ and inner-loop Jacobian Jᵢ = ∂f/∂η at η̂ᵢ:

1. **Build proposal scale Σᵢ:**
   - Compute Sheiner–Beal posterior Hessian
     \\(H_i = J_i^\\top R_i^{-1} J_i + \\Omega^{-1}\\) at η̂ᵢ
     (\\(R_i\\) = diagonal residual variance at the EBE).
   - Add a small ridge \\(\\lambda I\\) for numerical safety
     (λ = max(10⁻⁶ · trace(H)/d, 10⁻¹⁰)).
   - Cholesky-factor; on failure fall back to Σᵢ = Ω (broad prior-scale
     proposal — gives a valid but noisy LL estimate).

2. **Draw K samples** \\(\\eta_{ik} \\sim t_\\nu(\\hat\\eta_i, \\Sigma_i)\\)
   (multivariate Student-t).

3. **Compute log importance weights:**
   \\[
   \\log w_{ik} = \\log p(y_i \\mid \\eta_{ik}, \\theta)
                + \\log p(\\eta_{ik} \\mid \\theta)
                - \\log q(\\eta_{ik}).
   \\]

4. **Subject marginal LL** via log-sum-exp:
   \\[
   \\log \\hat p(y_i \\mid \\theta)
   \\;=\\; \\operatorname{lse}_k \\log w_{ik} - \\log K.
   \\]

5. **Per-subject effective sample size:**
   \\(\\mathrm{ESS}_i = 1 / \\sum_k \\tilde w_{ik}^2\\) (normalised weights).
   The result reports the across-subject min and median of ESS/K, plus
   any subjects below `is_low_ess_threshold` (default 10%).

6. **Monte-Carlo standard error.** For a self-normalised IS estimator the
   asymptotic per-subject variance (Geweke 1989) is
   \\[
   \\operatorname{Var}\\bigl(\\log \\hat p(y_i \\mid \\theta)\\bigr)
   \\;\\approx\\; \\frac{1}{K}\\left(\\frac{1}{\\mathrm{ESS}_i / K} - 1\\right).
   \\]
   Aggregating across subjects (LL is a sum of independent per-subject
   log-marginals) and converting to the `−2 log L` scale,
   \\[
   \\operatorname{SE}(-2 \\log L_{IS})
   \\;=\\; 2 \\sqrt{\\sum_i \\operatorname{Var}\\bigl(\\log \\hat p(y_i \\mid \\theta)\\bigr)}.
   \\]
   Subjects with degenerate ESS (\\(\\mathrm{ESS}_i / K = 0\\) — complete
   proposal collapse) fall back to a per-subject variance of `1.0`, which
   produces a *finite* but inflated SE rather than a NaN. A separate
   warning is emitted on `FitResult.warnings` listing the collapsed
   subjects; the IS-LL itself remains usable as a point estimate.

7. **Population −2 log L** = `−2 · Σᵢ log p̂(yᵢ | θ)`, reported on
   `FitResult.importance_sampling.minus2_log_likelihood` together with
   `mc_standard_error`.

## Tuning

- **K = `is_samples`** controls accuracy. The MC SE scales as
  \\(1/\\sqrt{K}\\); halve it by quadrupling K.
  Default 1000 is fine for a smoke test; bump to 2000–5000 for any
  reported LL.
- **ν = `is_proposal_df`** controls the proposal tails. The default of 5
  is robust to mild proposal misspecification (Geweke 1989); raise
  toward 30+ to recover a near-Gaussian proposal when the posterior is
  known to be light-tailed.
- **Low-ESS subjects** signal a proposal that doesn't match the
  posterior well — usually a sign the EBE didn't converge or the
  Hessian was near-singular. The IS estimate is still unbiased, just
  noisier. Investigate by re-running with a tighter `inner_tol` or
  inspecting the `FitResult.subjects[i]` diagnostics.
- **Reproducibility.** `is_seed` defaults to `42` when unset (so two
  fits with the same `is_samples` and `is_proposal_df` produce identical
  `−2 log L`). Set it explicitly to vary the RNG stream. The full set of
  `is_*` defaults is listed in the
  [`[fit_options]` reference](../model-file/fit-options.md#importance-sampling-imp).

## SDE / [diffusion] models (not supported)

IMP refuses to run on models with a `[diffusion]` block. The EKF
process-noise variance that inflates the residual variance for SDE
models is not yet threaded through the IS observation-likelihood path,
so the marginal would be silently biased. Use FOCE / FOCEI for the
Laplace OFV on SDE models; IMP for SDE is tracked as a follow-up.

## IOV (inter-occasion variability)

For models with `kappa` declarations, IMP performs **joint sampling** of
both η (between-subject variability) and κ (between-occasion variability).
The proposal is built from the full (n_eta + n_kappa × n_occasions)
posterior Hessian, and the IS weights include both the η and κ priors.

The result struct flags this via `kappa_treatment = "marginalized"`
(YAML) and the CLI prints a notice. The reported `−2 log L` integrates
over both η and κ uncertainty, making it directly comparable to
NONMEM `$EST METHOD=IMP LAPLACIAN=1` on κ.

## Cost

K × n_subjects predict calls per IMP run. For warfarin (32 subjects)
with K=2000 this is ≈ 64k predicts — seconds to a minute on a typical
laptop. The cost is linear in K and trivially parallel over subjects
(uses the same Rayon worker pool as the inner loop).

## Output

In the CLI:

```
--- Importance Sampling (marginal log-likelihood) ---
  -2 log L (IS): 1284.52  (MC SE = 0.41, K = 2000, ν = 5)
  ESS / K: min = 0.18, median = 0.62
  Low-ESS subjects (3): 12=0.18, 27=0.21, 34=0.22
```

In the fit YAML:

```yaml
importance_sampling:
  minus2_log_likelihood: 1284.524300
  mc_standard_error: 0.412700
  n_samples: 2000
  proposal_df: 5.0000
  ess_min: 0.1800
  ess_median: 0.6200
  kappa_treatment: marginalized  # or not_applicable if no IOV
  low_ess_subjects:
    - id: "12"
      ess_fraction: 0.1800
    - id: "27"
      ess_fraction: 0.2100
    - id: "34"
      ess_fraction: 0.2200
```

The Rust `FitResult.importance_sampling` field carries the same data
typed as `Option<ImportanceSamplingResult>`.

## See also

- [SAEM](saem.md) — common upstream stage for `methods = [saem, imp]`.
- [SIR](sir.md) — a different importance-sampling-flavoured procedure
  that targets *parameter uncertainty* (CIs on θ/Ω/σ), not the marginal
  likelihood. The name overlap is unfortunate; the two methods are
  unrelated in goal and implementation.
