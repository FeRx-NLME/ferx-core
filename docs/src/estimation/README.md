# Estimation Methods

ferx-core separates two orthogonal choices: the **statistical method** (how random effects are handled) and the **outer optimizer** (the algorithm that minimises the population objective).

## Statistical methods

- **[FOCE / FOCEI](foce.md)** — First-Order Conditional Estimation (with or without interaction). The workhorse of population PK, using nested MAP optimization to find maximum likelihood estimates.

- **[Gauss-Newton (BHHH)](gauss-newton.md)** — A fast alternative that exploits the nonlinear-least-squares structure of the FOCE objective. Converges in 10–30 iterations using the outer product of per-subject gradients as an approximate Hessian. Available as pure GN (`gn`) or a GN+FOCEI hybrid (`gn_hybrid`).

- **[SAEM](saem.md)** — Stochastic Approximation Expectation-Maximization. Uses MCMC sampling for random effects, providing more robust convergence on complex models with many random effects.

- **[IMPMAP](impmap.md)** — Importance Sampling assisted by Mode A Posteriori (NONMEM `METHOD=IMPMAP`). A Monte-Carlo EM estimator that re-centers a per-subject importance-sampling proposal on the conditional mode every iteration. Targets high-dimensional, rich-data models where a non-MAP importance-sampling EM stalls. Runs standalone (`method = impmap`) or chained (`methods = [focei, impmap]`).

## Post-estimation steps

These chain after a primary method via `methods = [...]` and refine the result rather than re-estimating from scratch:

- **[Importance Sampling (IMP)](importance-sampling.md)** — Monte-Carlo marginal likelihood estimate (`−2 log L`) that is more accurate than the Laplace approximation. Useful for model comparison when FOCE OFV may be biased (sparse data, complex random effect structure). Chain as `methods = [saem, imp]` or `methods = [focei, imp]`.

- **[SIR](sir.md)** — Sampling Importance Resampling. Provides non-parametric 95% CIs for all parameters, more robust than the asymptotic covariance matrix. Appended automatically when `sir = true` in `[fit_options]`.

## Outer optimizers

The outer loop optimizer is selected independently of the method. See **[Outer Optimizers](optimizers.md)** for a full comparison of BOBYQA, SLSQP, trust-region, BFGS, and the others. Short version: keep the default (`bobyqa` — derivative-free, robust on ODE/PD models and sparse data); switch to `slsqp` for smooth analytical PK models where the gradient is reliable.

---

## Quick comparison

| | FOCE/FOCEI | Gauss-Newton | SAEM |
|---|---|---|---|
| Random effect estimation | MAP (optimization) | MAP (optimization) | MCMC (sampling) |
| Convergence speed | Medium (~50–100 evals) | Fast (~10–30 iters) | Slower (~400 iters) |
| Local minima robustness | Can get stuck | Can get stuck | More robust |
| Gradient required | Yes (AD or FD) | Yes (FD, per-subject) | No (E-step) |
| Stochastic | No | No | Yes |
| OFV comparable to FOCE | — | Yes (post-GN Laplace) | Yes (post-SAEM Laplace) |
| Best for | General use | Fast model development | Complex models, many ETAs |

## Choosing a method

**Start with FOCEI** for standard 1–3 compartment models with 2–4 random effects. It is deterministic and comparable to NONMEM FOCEI.

**Try Gauss-Newton (`gn` or `gn_hybrid`)** when you want fast turnaround during model development, or when FOCE is slow to converge on a well-conditioned model.

**Switch to SAEM** when:
- FOCE fails to converge or produces implausible estimates
- The model has many random effects (>4 ETAs)
- You suspect the FOCE solution is a local minimum
- The model has complex nonlinear random effect structure

**Chain with IMP** (`methods = [focei, imp]` or `[saem, imp]`) when you need a more accurate marginal likelihood for model selection — especially on sparse data where the Laplace approximation is biased.

**Add SIR** (`sir = true`) when the covariance step fails or when you want non-parametric uncertainty intervals.

## Method chaining

Methods can be chained so that each stage warm-starts from the previous result:

```
[fit_options]
  methods = [saem, focei, imp]
```

This runs SAEM to find the basin → FOCEI polishes to the marginal optimum → IMP computes an exact marginal log-likelihood. Any two- or three-stage combination is valid; the most common chains are:

| Chain | Use case |
|-------|----------|
| `[saem, focei]` | Robust convergence + precise FOCE estimates |
| `[gn, focei]` | Equivalent to `gn_hybrid` |
| `[focei, imp]` | Exact marginal likelihood after a standard fit |
| `[saem, focei, imp]` | Most thorough: robust start + polished estimates + exact OFV |
