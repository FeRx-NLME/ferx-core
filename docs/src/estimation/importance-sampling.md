# Importance Sampling (IMP)

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

`imp` is, **by default, a Monte-Carlo EM estimator** — the equivalent of
NONMEM `$EST METHOD=IMP`. It maximises the importance-sampled marginal
likelihood, updating θ/Ω/σ each iteration. Set `imp_eval_only = true`
(NONMEM `EONLY=1`) to instead *evaluate* the marginal log-likelihood

<div>
\[
-2 \\log L(\\theta) \\;=\\; -2 \\sum_i \\log p(y_i \\mid \\theta)
\\;=\\; -2 \\sum_i \\log \\int p(y_i \\mid \\eta, \\theta)\\, p(\\eta \\mid \\theta)\\, d\\eta
\]
</div>

at the **fixed** input parameters without estimating them.

> **⚠️ Behaviour change.** Before this release `imp` only *evaluated*
> `−2 log L` at fixed parameters (the `imp_eval_only` behaviour). It is now an
> estimator by default. If you used `method = imp` (or `[focei, imp]`) purely to
> *score* a fit, add `imp_eval_only = true` to keep the old behaviour.

### Mapping to NONMEM

| ferx | NONMEM |
|------|--------|
| `method = imp` (default) | `$EST METHOD=IMP` (estimator) |
| `method = imp` + `imp_eval_only = true` | `$EST METHOD=IMP EONLY=1` (evaluate) |
| `method = impmap` | `$EST METHOD=IMPMAP` (estimator) |

The IS Monte-Carlo estimator of `p(yᵢ|θ)` is unbiased; the reported `−2 log L`
carries a small Jensen bias from the `log(·)` transform that vanishes as
`K → ∞` and is typically dominated by the much larger Laplace-approximation
bias of `ofv` in the sparse-data / strongly-nonlinear regime IMP targets.

## How the estimator works

IMP is a Monte-Carlo EM (MCEM) loop sharing all of its machinery with
[IMPMAP](impmap.md) except the **proposal re-centering strategy**:

- On the **first** iteration it finds each subject's conditional mode and
  first-order variance via a FOCE inner loop, and centres the
  importance-sampling proposal there.
- On **every subsequent** iteration it re-centres the proposal from the
  *previous* iteration's importance-sample mean and covariance — it does **not**
  re-run the inner loop. This is the one-line difference from IMPMAP (which
  re-derives the mode/variance every iteration) and the source of IMP's
  lower per-iteration cost.

Each iteration's M-step updates θ/Ω/σ from the importance-weighted posterior
moments (closed-form Ω and log-mu-referenced θ; a small BOBYQA step for σ and
non-mu-ref θ), exactly as in IMPMAP. The reported estimate is the running mean
of the parameter vector over the final `imp_averaging` iterations, and `ofv` is
a final FOCE Laplace pass for AIC/BIC comparability.

> **Which number matches NONMEM's reported OFV?** NONMEM `METHOD=IMP` reports the
> importance-sampling Monte-Carlo **marginal** `−2 log L` (the `.ext`/`.lst`
> `#OBJV`), *not* a Laplace value. ferx's `ofv` is a Laplace pass, so it will
> **not** equal NONMEM's IMP `#OBJV` on data where the Laplace approximation and
> the true marginal diverge (sparse / strongly nonlinear). The NONMEM-comparable
> number is the marginal, evaluated at the final estimates and surfaced on
> `FitResult.importance_sampling.minus2_log_likelihood` (with its Monte-Carlo SE
> on `.mc_standard_error`) for **estimating** `imp`/`impmap` runs too — not only
> the evaluation-only path. If IMPMAP is configured with a Gaussian proposal
> (`impmap_proposal_df = normal`), it is replaced by a finite-`t` proposal for
> this final marginal eval, so the heavier tails keep the importance weights bounded.

### Rich data: prefer IMPMAP or warm-start

Because IMP's proposal lags one iteration behind the parameters, it is
**fragile on rich data**: when the conditional posterior of η is razor-sharp
(many observations per subject), a large early M-step moves the posterior past
the lagged proposal and the effective sample size collapses. This is the
documented weakness of `METHOD=IMP` and the reason NONMEM offers IMPMAP. On
rich data, either use [`impmap`](impmap.md) (re-centres every iteration, robust)
or **warm-start IMP from FOCEI** (`methods = [focei, imp]`) so the per-iteration
steps are small and the lagged proposal stays overlapped. IMP is well-suited to
**sparse data**, where the broad posterior keeps the proposal overlapped.

```
[fit_options]
  method        = imp            # estimate by importance-sampling MCEM
  imp_iterations = 200            # MCEM iterations
  imp_samples    = 1000           # K, samples per subject per iteration
  imp_averaging  = 50             # terminal iterations to average
  imp_proposal_df = 5             # ν, Student-t tail weight (or `normal` → MVN)
  imp_seed       = 12345
```

```
[fit_options]
  method = [focei, imp]          # robust on rich data: warm-start IMP from FOCEI
```

## Evaluation-only mode (`imp_eval_only = true`)

With `imp_eval_only = true`, `imp` does **not** estimate: it evaluates the IS
`−2 log L` at the fixed input parameters and reports it on
`FitResult.importance_sampling` (handy for scoring imported / fixed parameter
sets, or comparing models on a lower-bias likelihood than the Laplace OFV).

### When evaluation-only is useful

The Laplace approximation behind the FOCE/FOCEI OFV assumes each subject's
posterior of η is Gaussian-shaped — fine for **well-sampled** PK (≥ 5–6 obs
per subject) but biased for **sparse data**, **strong nonlinearity**
(Michaelis–Menten, TMDD, transit absorption), or **categorical/binary PD**.
There the FOCE OFV is typically *under*-stated, so naïve AIC/BIC favours
over-parameterised models; the IS `−2 log L` is much lower-bias.

```
[fit_options]
  method        = [focei, imp]   # estimate with FOCEI, then score the IS-LL
  imp_eval_only  = true
  imp_samples    = 2000
  imp_proposal_df = 5
```

```
[fit_options]
  method       = imp             # standalone: IS-LL at the initial parameters
  imp_eval_only = true
```

Rules for the **evaluation-only** mode:

- **Standalone is allowed.** `method = imp` + `imp_eval_only` evaluates the IS-LL
  at the **initial parameters** — deriving the EBEs/Jacobian (the proposal
  centre/scale) there via a FOCE inner loop. When chained after an estimator
  (`[focei, imp]`), it uses that stage's EBEs/Jacobian instead.
- **Must appear at most once** (true for both modes).
- **Must be terminal.** `methods = [imp, focei]` with `imp_eval_only` is rejected
  — a following stage would overwrite the parameters and make the IS result
  meaningless. (The *estimating* `imp` has no such restriction; it may lead or
  sit mid-chain like any estimator.)

For the estimating `imp`, `FitResult.method` reports `IMP`. For an
evaluation-only `imp` after an estimator it reports that estimating stage (e.g.
`FOCEI`); standalone evaluation-only reports `IMP`. Either way
`FitResult.method_chain` preserves the full chain, and the IS-LL lands on
`FitResult.importance_sampling`.

## Algorithm

This is the per-subject importance-sampling kernel — the E-step of the
estimator and the whole of the evaluation-only mode. The estimating loop simply
re-runs this kernel each iteration (re-centring the proposal from the previous
iteration's sample moments) and feeds the weighted moments into the M-step.

For each subject *i* with proposal centre η̂ᵢ (the EBE on the first
iteration / in evaluation-only mode) and inner-loop Jacobian Jᵢ = ∂f/∂η at η̂ᵢ:

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
<div>
   \[
   \\log w_{ik} = \\log p(y_i \\mid \\eta_{ik}, \\theta)
                + \\log p(\\eta_{ik} \\mid \\theta)
                - \\log q(\\eta_{ik}).
   \]
</div>

4. **Subject marginal LL** via log-sum-exp:
<div>
   \[
   \\log \\hat p(y_i \\mid \\theta)
   \\;=\\; \\operatorname{lse}_k \\log w_{ik} - \\log K.
   \]
</div>

5. **Per-subject effective sample size:**
   \\(\\mathrm{ESS}\_i = 1 / \\sum\_k \\tilde w\_{ik}^2\\) (normalised weights).
   The result reports the across-subject min and median of ESS/K, plus
   any subjects below `imp_low_ess_threshold` (default 10%).

6. **Monte-Carlo standard error.** For a self-normalised IS estimator the
   asymptotic per-subject variance (Geweke 1989) is
<div>
   \[
   \\operatorname{Var}\\bigl(\\log \\hat p(y_i \\mid \\theta)\\bigr)
   \\;\\approx\\; \\frac{1}{K}\\left(\\frac{1}{\\mathrm{ESS}_i / K} - 1\\right).
   \]
</div>
   Aggregating across subjects (LL is a sum of independent per-subject
   log-marginals) and converting to the `−2 log L` scale,
<div>
   \[
   \\operatorname{SE}(-2 \\log L_{IS})
   \\;=\\; 2 \\sqrt{\\sum_i \\operatorname{Var}\\bigl(\\log \\hat p(y_i \\mid \\theta)\\bigr)}.
   \]
</div>
   Subjects with degenerate ESS (\\(\\mathrm{ESS}_i / K = 0\\) — complete
   proposal collapse) fall back to a per-subject variance of `1.0`, which
   produces a *finite* but inflated SE rather than a NaN. A separate
   warning is emitted on `FitResult.warnings` listing the collapsed
   subjects; the IS-LL itself remains usable as a point estimate.

7. **Population −2 log L** = `−2 · Σᵢ log p̂(yᵢ | θ)`, reported on
   `FitResult.importance_sampling.minus2_log_likelihood` together with
   `mc_standard_error`.

## Tuning

- **`imp_iterations` / `imp_averaging`** (estimator only) set the number of MCEM
  iterations and how many terminal iterations are averaged into the reported
  estimate. Defaults 200 / 50. Under-running shows up as estimates still drifting
  with the MC noise; raise `imp_iterations` if the trace hasn't stabilised.
- **K = `imp_samples`** controls accuracy. The MC SE scales as
  \\(1/\\sqrt{K}\\); halve it by quadrupling K.
  Default 1000 is fine for a smoke test; bump to 2000–5000 for any
  reported LL.
- **ν = `imp_proposal_df`** controls the proposal tails. The default of 5
  is robust to mild proposal misspecification (Geweke 1989); raise
  toward 30+ to recover a near-Gaussian proposal when the posterior is
  known to be light-tailed.
- **Low-ESS subjects** signal a proposal that doesn't match the
  posterior well — usually a sign the EBE didn't converge or the
  Hessian was near-singular. The IS estimate is still unbiased, just
  noisier. Investigate by re-running with a tighter `inner_tol` or
  inspecting the `FitResult.subjects[i]` diagnostics.
- **Reproducibility.** `imp_seed` defaults to `42` when unset (so two
  fits with the same `imp_samples` and `imp_proposal_df` produce identical
  `−2 log L`). Set it explicitly to vary the RNG stream. The full set of
  `imp_*` defaults is listed in the
  [`[fit_options]` reference](../model-file/fit-options.md#importance-sampling-imp).

## SDE / [diffusion] models (not supported)

IMP refuses to run on models with a `[diffusion]` block. The EKF
process-noise variance that inflates the residual variance for SDE
models is not yet threaded through the IS observation-likelihood path,
so the marginal would be silently biased. Use FOCE / FOCEI for the
Laplace OFV on SDE models; IMP for SDE is tracked as a follow-up.

## IOV (inter-occasion variability)

> IOV is supported only in **evaluation-only** mode (`imp_eval_only = true`). The
> estimating `imp` (like `impmap`) refuses IOV models for now — the κ M-step is
> a planned follow-up. Use SAEM or FOCEI to *estimate* IOV models, then score
> with an evaluation-only `imp`.

For models with `kappa` declarations, evaluation-only IMP performs
**joint sampling** of
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

## NONMEM comparison

ferx's estimating `imp` is validated against NONMEM 7.5.1 `METHOD=IMP` on the
warfarin example (10-subject extract, `data/warfarin.csv`). Both engines are
warm-started from a FOCE/FOCEI pass (ferx `methods = [focei, imp]`; NONMEM
`$EST METHOD=COND INTERACTION` → `$EST METHOD=IMP INTERACTION NITER=100
ISAMPLE=1000 SEED=12345`, control stream `tests/nonmem/warfarin_imp.ctl`).

| Parameter | NONMEM `METHOD=IMP` | ferx `[focei, imp]` |
|-----------|--------------------:|--------------------:|
| TVCL      | 0.1264              | 0.1327              |
| TVV       | 7.723               | 7.737               |
| TVKA      | 0.8857              | 0.811               |
| ω²(CL)    | 0.0304              | 0.0286              |
| ω²(V)     | 0.0096              | 0.0096              |
| ω²(KA)    | 0.3405              | 0.336               |
| σ (SD)    | 0.0105              | 0.0106              |
| IMP marginal `−2 log L` | −285.69 | −285.93 ± 0.07 |
| Laplace `ofv`           | (−286.00 at COND) | −286.00 |

Both engines start from the same FOCEI basin (NONMEM `METHOD=COND` OFV −286.00,
identical to ferx's FOCEI). NONMEM's IMP MCEM then drifts slightly off it toward
the importance-sampled marginal optimum (TVKA up to 0.886), while ferx's
warm-started IMP holds near the FOCEI optimum — both stable and agreeing within
the cross-engine + Monte-Carlo margin. TVKA is the least-identified parameter on
this small extract (ETA_KA variance ≈ 0.34, high shrinkage); CL/V and the
variance components agree to a few percent.

Compare like with like: NONMEM's reported IMP `#OBJV` (−285.69) is the
importance-sampling **marginal** `−2 log L`, so the matching ferx number is
`importance_sampling.minus2_log_likelihood` (−285.93 ± 0.07), *not* ferx's
Laplace `ofv` (−286.00, which instead coincides with NONMEM's COND OBJ). The
small residual gap is the parameter-estimate difference (ferx's TVKA sits a
little below NONMEM's), not an OFV-definition difference — both objectives drop
the same `Nobs·log(2π)` constant (NONMEM "WITHOUT CONSTANT"). The cross-check
lives in `tests/warfarin_imp_nonmem.rs`.

## See also

- [SAEM](saem.md) — common upstream stage for `methods = [saem, imp]`.
- [SIR](sir.md) — a different importance-sampling-flavoured procedure
  that targets *parameter uncertainty* (CIs on θ/Ω/σ), not the marginal
  likelihood. The name overlap is unfortunate; the two methods are
  unrelated in goal and implementation.
