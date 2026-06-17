# Bayesian estimation (MCMC)

`method = bayes` runs full MCMC Bayesian estimation — it draws from the joint
posterior

\\[
p(\theta, \Omega, \Sigma, \{\eta_i\} \mid y) \propto
\Big[\prod_i p(y_i \mid \theta, \eta_i, \Sigma)\,\mathcal{N}(\eta_i \mid 0, \Omega)\Big]\;
p(\theta)\,p(\Omega)\,p(\Sigma)
\\]

rather than returning a single point estimate. It targets parity with NONMEM
`METHOD=BAYES`. This is a **Gibbs-within-HMC** sampler (the "Path A" design): it
reuses the per-subject HMC/MH kernels and the conjugate sufficient statistics
the other estimators already build, so it needs no population-level gradient.

> **IOV.** Inter-occasion variability (per-occasion `kappa`) is supported for
> **zero-mean kappas** — `κ ~ N(0, Ω_iov)` added to an existing mu-reference, the
> `exp(η + κ)` form. Kappas that anchor to their own θ (kappa mu-references) are
> rejected.

## The sampler

Each sweep, for each chain, cycles these blocks:

1. **η block** — samples \\( \eta_i \mid \theta, \Omega, \Sigma, \kappa, y \\) for
   every subject with a \\( \mathrm{chol}(\Omega) \\)-preconditioned block
   Metropolis kernel, or gradient-guided **HMC** when available (an autodiff
   build, `n_leapfrog > 0`, an analytical-PK subject, no IOV).

1b. **κ block** (IOV models) — samples each per-occasion
   \\( \kappa_{ik} \mid \eta, \theta, \Omega, \Omega_{iov}, y \\) holding η fixed.

2. **Ω block** — a conjugate **inverse-Wishart** draw
   \\( \Omega \mid \{\eta_i\} \sim \mathcal{W}^{-1}(\nu_0 + N,\ \Lambda_0 + \textstyle\sum_i \eta_i\eta_i^\top) \\),
   with structural zeros / fixed entries re-imposed and the diagonal floored.

2c. **Ω_iov block** (IOV) — the same conjugate inverse-Wishart draw from the
   per-occasion kappa scatter \\( \textstyle\sum_i\sum_k \kappa_{ik}\kappa_{ik}^\top \\).
   The posterior `OMEGA_IOV(i,j)` entries appear in the summaries.

3. **mu-referenced θ block** — for `P_i = θ·exp(η_i)` the population mean
   \\( \mu = \log\theta \\) has the exact Gaussian full conditional
   \\( \mu \sim \mathcal{N}(\mu_{\text{old}} + \bar\eta,\ \Omega/N) \\). The draw
   is applied as a shift \\( s \\): \\( \theta \leftarrow \theta\,e^{s} \\) and
   \\( \eta_i \leftarrow \eta_i - s \\), which leaves each individual parameter
   \\( \log\varphi_i = \mu + \eta_i \\) unchanged (only the η-prior moves). This
   always-accepted Gibbs move is what gives the sampler good mixing.

4. **(remaining θ, σ) block** — a random-walk Metropolis step in unconstrained
   space (log where the lower bound is ≥ 0) for any non-mu-referenced θ and for
   σ, with objective \\( \sum_i \mathrm{nll}_i \\) (η and Ω fixed, so the η-prior
   term cancels in the acceptance ratio).

Step sizes adapt during warmup toward a ~0.234 acceptance rate. Post-warmup
draws are thinned by `bayes_thin` and summarized.

## Options

| Key | Default | Meaning |
|-----|--------:|---------|
| `bayes_warmup` | 1000 | Warmup (burn-in + adaptation) sweeps per chain, discarded. |
| `bayes_iters`  | 1000 | Retained sampling sweeps per chain (before thinning). |
| `bayes_chains` | 4 | Independent chains (distinct seeds; used for split-R̂). |
| `bayes_thin`   | 1 | Keep every `bayes_thin`-th sampling draw. |
| `bayes_seed`   | — | Base RNG seed; chain `c` derives its own. |
| `n_leapfrog`   | 0 | Leapfrog steps for the HMC η kernel (autodiff builds; `0` ⇒ block MH). |

## Output

Posterior summaries are reported on `FitResult.bayes` and in the `.fit.yaml`
`bayes:` section — per parameter: `mean`, `sd`, `q025`, `median`, `q975`,
split-`rhat`, `ess_bulk` / `ess_tail`, and `mcse`; plus `n_chains`, `n_warmup`,
`n_draws_per_chain`, `n_divergent`, and `max_rhat`. The point-estimate fields
(`theta`/`omega`/`sigma`) carry the posterior means; the covariance / SIR steps
are skipped (posterior credible intervals replace the Hessian covariance).

A `max_rhat` above ~1.01 indicates the chains have not converged — increase
`bayes_warmup` / `bayes_iters`. This is also the threshold for the reported
`converged` flag. Use **at least 2 chains** (`bayes_chains >= 2`): split-R̂ on a
single chain only compares the two halves of one trajectory and cannot detect
between-chain non-convergence, so a near-1 `max_rhat` from one chain is weak
evidence; a single-chain run is warned about for this reason.

## Example

```toml
[fit_options]
  method       = bayes
  bayes_warmup = 1000
  bayes_iters  = 2000
  bayes_chains = 4
  bayes_seed   = 1
```

```
--- Bayesian posterior (Gibbs-within-HMC) ---
  chains = 4, warmup = 1000, draws/chain = 2000, max R-hat = 1.002
  param              mean         sd       2.5%      97.5%   Rhat      ESS
  TVCL             0.1329     0.0081     0.1172     0.1496  1.000     7900
  TVV              7.7445     0.2858     7.2093     8.3310  1.000     7960
  TVKA             0.8280     0.1620     0.5605     1.1975  1.000     7930
```

## Priors

Defaults are weakly-informative (≈ flat given typical data): \\( \Omega \sim
\mathcal{W}^{-1}(n_\eta + 2,\ \Omega_0) \\) centred at the initial Ω, and
\\( \theta,\ \log\sigma \sim \mathcal{N}(\text{init},\ \mathrm{sd}=10) \\) on the
unconstrained scale. User-specifiable priors in the `.ferx` DSL are planned.

## Validation

On the bundled warfarin model the posterior means match the FOCEI point estimate
and NONMEM `METHOD=BAYES` (TVCL ≈ 0.133, TVV ≈ 7.74, TVKA ≈ 0.83), with
`max_rhat` ≈ 1.00. Ω posterior means run slightly above the MLE — the expected
inverse-Wishart posterior-mean bias at small subject counts.

> NONMEM reports `SIGMA` as the variance of `EPS`; ferx parameterises
> proportional error as \\( \mathrm{Var} = (f\sigma)^2 \\) and reports σ as the
> SD, so compare `ferx σ ≈ √(NONMEM SIGMA)`.
