# IMPMAP — Importance Sampling assisted by Mode A Posteriori

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

`impmap` (alias `importance_sampling_map`) is a **Monte-Carlo EM (MCEM)
estimator**, equivalent to NONMEM `$EST METHOD=IMPMAP`. It estimates the
population parameters θ/Ω/σ by maximizing the importance-sampled marginal
likelihood.

Both [`imp`](importance-sampling.md) and `impmap` are MCEM estimators; they
differ only in how the proposal is re-centered. `impmap` re-derives each
subject's conditional mode and first-order variance **every** iteration (robust
on rich data), whereas `imp` re-centers from the previous iteration's sample
moments after the first iteration (cheaper, but can stall on rich data).
`imp` can also run in evaluation-only mode (`is_eval_only = true`), which
`impmap` has no equivalent of.

> **NONMEM's description.** *"Sometimes for highly dimensioned PK/PD problems
> with very rich data the importance sampling method does not advance the
> objective function well or even diverges. For this the IMPMAP method may be
> used. At each iteration, conditional modes and conditional first-order
> variances are evaluated as in the ITS or FOCE method, not just on the first
> iteration as is done with IMP method. These are then used as parameters to the
> multivariate normal proposal density for the Monte Carlo importance sampling
> step."*

## When to use it

IMPMAP targets the regime where a plain importance-sampling EM stalls or
diverges: **high-dimensional models with rich data**, where the individual
posteriors are sharp and a proposal not re-centered on the mode samples the
likelihood poorly. Re-evaluating the conditional mode and first-order variance
**every** iteration keeps the proposal aligned with the posterior throughout.

For routine 1–3 compartment PK with a few random effects, FOCEI is faster and
deterministic — reach for IMPMAP when FOCEI struggles or you want a
sampling-based cross-check that does not rely on the Laplace approximation.

## How it runs

Standalone, or as a chain stage warm-started by a preceding estimator:

```
[fit_options]
  method             = importance_sampling_map   # or: impmap
  impmap_iterations  = 200
  impmap_samples     = 300
  impmap_proposal_df = normal      # multivariate normal (NONMEM default)
  impmap_seed        = 12345
```

```
[fit_options]
  method = [focei, impmap]         # FOCEI warm-start, then IMPMAP refine
```

Because `impmap` is an estimator, in a chain like `[focei, impmap]` the reported
`FitResult.method` is `IMPMAP` and `method_chain` preserves the full sequence.

## Algorithm

Each MCEM iteration, at the current parameters θ⁽ᵗ⁾, Ω⁽ᵗ⁾, σ⁽ᵗ⁾:

1. **MAP re-center (E-step A).** Run the FOCE inner loop for every subject to get
   the conditional mode η̂ᵢ and Jacobian Jᵢ, and form the first-order-variance
   Hessian \\(H_i = J_i^\\top R_i^{-1} J_i + \\Omega^{-1}\\).

2. **Importance sampling (E-step B).** Draw `K = impmap_samples` samples
   \\(\\eta_{ik} \\sim q(\\hat\\eta_i, H_i^{-1})\\) — a multivariate normal by
   default (`impmap_proposal_df = normal`), or a Student-t for heavier tails —
   with self-normalized weights
   \\(\\tilde w_{ik} \\propto p(y_i\\mid\\eta_{ik},\\theta)\\,p(\\eta_{ik}\\mid\\Omega)/q(\\eta_{ik})\\).

3. **M-step.**
   - **Ω** (closed form): \\(\\Omega = \\tfrac1N \\sum_i \\sum_k \\tilde w_{ik}\\,\\eta_{ik}\\eta_{ik}^\\top\\),
     then structural off-diagonals zeroed, FIX entries restored, free diagonals
     floored to stay positive-definite.
   - **θ** (mu-referenced): the log-typical value is shifted by the
     importance-weighted population mean random effect,
     \\(\\log\\theta \\mathrel{+}= \\tfrac1N\\sum_i \\sum_k \\tilde w_{ik}\\,\\eta_{ik}\\).
     Re-MAPing at the new θ drives the mean random effect toward zero, so this
     fixed-point iteration resolves the θ–η confounding that would otherwise
     inflate Ω.
   - **θ** (non-mu-ref) and **σ**: maximize the importance-weighted observation
     likelihood \\(\\sum_i \\sum_k \\tilde w_{ik}\\log p(y_i\\mid\\eta_{ik},\\theta,\\sigma)\\)
     with a derivative-free NLopt BOBYQA step.

The reported estimate is the running mean of the parameter vector over the final
`impmap_averaging` iterations. A FOCE-Laplace `ofv` is computed at the final
parameters for AIC/BIC comparability with FOCE/FOCEI/SAEM.

> **Mu-referencing is required.** The closed-form `log θ += mean(η)` shift is the
> EM-correct typical-value update for log-normal random effects, so it is always
> applied for log-mu-referenced parameters (`CL = TVCL * exp(ETA_CL)`),
> independent of the `mu_referencing` fit option — that option only governs
> inner-loop centering. Like NONMEM's EM methods, IMPMAP needs a mu-referenced
> parameterization: a model with random effects whose typical values are *not*
> log-mu-referenced emits a warning and may estimate those typical values poorly
> (θ and the η mean are confounded over the importance samples). Use FOCEI for
> such models.

## Validation against NONMEM

On the bundled warfarin example (1-cpt oral, proportional error, 3 mu-referenced
ETAs), ferx IMPMAP (`impmap_iterations = 150`, `impmap_samples = 500`, MVN
proposal, seed 12345) matches NONMEM 7.5.1 `METHOD=IMPMAP` on the same
model/data (`tests/nonmem/warfarin_impmap.ctl`):

| Parameter | NONMEM IMPMAP | ferx IMPMAP |
|-----------|--------------:|------------:|
| TVCL | 0.135 | 0.1327 |
| TVV | 7.89 | 7.737 |
| TVKA | 0.730 | 0.811 |
| ω²(ETA_CL) | 0.0291 | 0.0286 |
| ω²(ETA_V) | 0.0101 | 0.0096 |
| ω²(ETA_KA) | 0.340 | 0.336 |
| σ (PROP, SD) | 0.01034 | 0.01056 |
| −2 log L | −284.92 | −286.00 |

The well-determined CL/V structure and all three variance components agree to a
few percent; TVKA is the least-identified parameter on this 10-subject extract
(ETA_KA variance ≈ 0.34, high shrinkage) and carries the loosest band. The OFVs
(NONMEM "without constant" vs ferx's Laplace OFV) agree to ~1 unit, the usual
cross-engine margin. This comparison is asserted by the gated
`ferx_impmap_matches_nonmem_impmap_on_warfarin` test (nightly, `slow-tests`); a
companion `impmap_converges_to_focei_on_warfarin` test checks agreement with
ferx's own FOCEI.

## Multi-start MAP (`impmap_mceta`)

In high-dimensional models (e.g. FREM with many covariates, ≥ 5 ETAs) the
per-subject MAP optimization can converge to a local mode, degrading
importance-sampling efficiency (low ESS) and biasing the M-step. The
`impmap_mceta` option (analogous to NONMEM `MCETA`) adds random starting
points drawn from N(0, Ω) and keeps the start with the lowest individual NLL:

```
[fit_options]
  method        = impmap
  impmap_mceta  = 3       # 1 warm-start + 3 random starts per subject
```

The default is `0` (single warm-start, matching previous behaviour).
`impmap_mceta = 3` is a good choice for FREM models. The cost is roughly
`(1 + mceta) ×` the MAP step per iteration, but since E-step B (IS draws)
dominates, the total wall-time increase is typically 15–40%.

## Cost

Per iteration ≈ one MAP inner loop (parallel over subjects) plus
`K × n_subjects` predicts for the IS step plus a short weighted BOBYQA M-step.
Total cost is roughly `impmap_iterations ×` the cost of a single `imp`
evaluation. Everything is parallelized over subjects via the shared Rayon pool.

## Not yet supported

- **IOV (`[iov]` / `kappa`)** — the κ sufficient statistics and Ω_iov update are
  a planned follow-up. IOV models are refused; use SAEM or FOCEI.
- **SDE / `[diffusion]`** — refused for the same reason `imp` refuses them (the
  EKF process-noise variance is not threaded through the IS observation
  likelihood). Use FOCE / FOCEI.

## See also

- [Importance Sampling (IMP)](importance-sampling.md) — the sample-moment
  re-centered sibling estimator (and its evaluation-only mode) that shares
  IMPMAP's per-subject IS kernel and M-step.
- [SAEM](saem.md) — the other sampling-based estimator; a common warm-start
  (`methods = [saem, impmap]`).
