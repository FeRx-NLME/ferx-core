# SAEM

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

Stochastic Approximation Expectation-Maximization (SAEM) is an alternative estimation method that uses MCMC sampling for random effects instead of MAP optimization. It is more robust to local minima and can handle complex random effect structures, including models with **inter-occasion variability (IOV)**.

## Algorithm Overview

SAEM replaces the deterministic inner loop of FOCE with stochastic sampling, following the Monolix convention with a two-phase step-size schedule.

### References

- Delyon, Lavielle, Moulines (1999). *Convergence of a stochastic approximation version of the EM algorithm.* Annals of Statistics, 94--128.
- Kuhn & Lavielle (2004). *Coupling a stochastic approximation version of EM with an MCMC procedure.* ESAIM: Probability and Statistics 8:115--131.

## Two-Phase Schedule

### Phase 1: Exploration (iterations 1 to K1)

Step size \\( \gamma_k = 1 \\). The algorithm explores the parameter space rapidly, with the sufficient statistics being fully replaced each iteration. This allows fast movement toward the basin of the MLE.

Default: 150 iterations.

### Phase 2: Convergence (iterations K1+1 to K1+K2)

Step size \\( \gamma_k = 1/(k - K_1) \\). The algorithm performs a decreasing-weight average, which guarantees almost-sure convergence to the MLE under regularity conditions.

Default: 250 iterations.

## Per-Iteration Steps

Each SAEM iteration consists of:

### 1. E-Step: Sampling

For each subject, sample from the conditional distribution of random effects:

<div>
\[ p(\eta_i | y_i, \theta, \Omega, \sigma) \]
</div>

Two samplers are available:

#### Metropolis-Hastings (default, `n_leapfrog = 0`)

Two MH kernels run per subject per SAEM iteration (the mixture of Kuhn & Lavielle 2004):

**Kernel 1 — block proposal**: run `n_mh_steps` symmetric random-walk steps \\( \eta_{\text{prop}} = \eta_{\text{current}} + \delta_i \cdot L \cdot z \\), with \\( z \sim N(0, I) \\) and \\( L = \text{chol}(\Omega) \\). This mixes the joint scale efficiently.

**Kernel 2 — componentwise sweep**: for multi-η models, follow the block move with \\( \max(2, \lfloor \texttt{n\_mh\_steps} / n_\eta \rfloor) \\) sweeps that perturb one coordinate at a time, \\( \eta_j' = \eta_j + \delta_i^{\text{cw}} \cdot \sqrt{\Omega_{jj}} \cdot z \\), holding the others fixed. Because the block proposal is shaped by \\( \text{chol}(\Omega) \\), once \\( \Omega \\) drifts toward a high correlation the block move can only travel along that near-degenerate direction; the single-draw Ω M-step then feeds the correlation back into \\( \Omega \\), and during the \\( \gamma=1 \\) exploration phase this compounds into a runaway collapse toward a **near rank-1 \\( \Omega \\)** (every off-diagonal correlation \\( \to \pm 1 \\), one variance \\( \to 0 \\)). A componentwise move can always shift a single η independently of \\( \Omega \\)'s off-diagonals, so the sampled draws are not forced collinear. The kernel is skipped for single-η models (no off-diagonal to decorrelate).

Both kernels are symmetric in \\( \eta \\), so the proposal density cancels and the acceptance log-ratio is the difference of `individual_nll` values, which encodes the prior \\( N(0, \Omega) \\) plus the observation likelihood.

**Acceptance**: \\( \min(1, \exp(\text{NLL}\_{\text{current}} - \text{NLL}\_{\text{prop}})) \\). Target acceptance rate: 40%.

#### HMC (Hamiltonian Monte Carlo, `n_leapfrog > 0`)

When `n_leapfrog` is set to a positive integer (e.g. `3`), one HMC proposal replaces the `n_mh_steps` MH proposals per subject per iteration. HMC uses the gradient of the individual NLL to make longer, more directed moves through the posterior:

<div>
\[ H(\eta, p) = \text{NLL}(\eta) + \tfrac{1}{2} \|p\|^2 \]
</div>

with momentum \\( p \sim N(0, I) \\) and a standard velocity Störmer-Verlet (leapfrog) integrator. Acceptance is on \\( \Delta H \\), targeting ~65% acceptance.

**Requirements**: an analytical PK model (no ODE). The HMC gradient is the exact analytic `Dual2` η-gradient (the same one FOCEI uses) — no autodiff. A warning is emitted if `n_leapfrog > 0` but the model is out of the analytic provider's scope.

**Per-subject fallback to MH**: `hmc_step` silently falls back to MH for a subject when any of the following conditions hold:
- The model uses an ODE (`[odes]` block present)
- The model has no analytical PK path (no `tv_fn` — pure ODE-only models)
- The Ω matrix has a non-finite log-determinant (degenerate variance)
- The model is outside the analytic gradient's scope (time-varying covariates, oral infusion, SS+reset, expression scaling)

In a single run with `n_leapfrog > 0`, different subjects can therefore use different samplers. The acceptance rate reported in verbose output and the optimizer trace is an aggregate across all subjects; in mixed HMC/MH runs the target (65% for HMC, 40% for MH) may not be meaningful for the aggregate. The `n_mh_steps` option governs the number of proposals for MH-fallback subjects even when `n_leapfrog > 0`.

The E-step sampling is parallelized across subjects using Rayon.

The startup banner reports the resolved E-step kernel on a `sampler:` line, e.g. `sampler:  Metropolis-Hastings random walk` or `sampler:  HMC (3 leapfrog steps, Dual2 analytic gradients)`. If `n_leapfrog > 0` but HMC is unavailable (an ODE model, or one outside the analytic gradient's scope), the line says so and reflects the MH fallback. Because SAEM is sampling-based rather than gradient-driven, it does not print the `gradient:` line that FOCE/FOCEI use — the `gradient_method` option only governs the inner EBE/Hessian step (used for diagnostics, and consumed by a following `imp` stage), not the SAEM iterations themselves.

### 2. Stochastic Approximation Update

Update the sufficient statistic for \\( \Omega \\):

<div>
\[ S_2 \leftarrow (1 - \gamma_k^\Omega) \cdot S_2 + \gamma_k^\Omega \cdot \frac{1}{N} \sum_{i=1}^{N} \eta_i \eta_i^T \]
</div>

**Damped Ω step.** The θ/σ M-step uses the full \\( \gamma_k \\) (1.0 during exploration), but the Ω sufficient statistic uses a *capped* step \\( \gamma_k^\Omega = \min(\gamma_k, 0.1) \\). With the full \\( \gamma=1 \\), Ω would be overwritten every exploration iteration by a single warm-started, not-yet-equilibrated MCMC draw; for a correlated block that one snapshot is biased toward the chain's current correlation, and the bias feeds back through \\( \mathrm{chol}(\Omega) \\) into the next proposal — the same rank-1 runaway the componentwise kernel guards against, here attacked from the M-step side. Capping the Ω learning rate during exploration averages those draws (Robbins-Monro) and breaks the feedback while θ still moves at full speed. The cap applies during exploration only; in the convergence phase it is lifted and Ω uses the full decaying \\( \gamma_k = 1/(k-K_1) \\) — the same Robbins-Monro schedule as θ — so the SA estimate settles correctly (by then the chain is equilibrated, so the single-draw overwrite risk no longer applies).

### 3. M-Step for Omega (Closed Form)

<div>
\[ \Omega_k = S_2 \]
</div>

with structurally-zero entries (cross-block off-diagonals, standalone-vs-block off-diagonals, and all off-diagonals in a fully-diagonal Ω) zeroed out — the SA accumulator \\( (1/N) \sum_i \eta_i \eta_i^T \\) is dense by construction, but the model declares which entries are free parameters. Without this projection the chain feeds spurious sampling correlations into the next iteration's MH proposal Cholesky and Ω drifts toward rank-deficiency. Free diagonals are then floored at `1e-6` to keep them away from zero (which would collapse the per-eta proposal scale `δ·chol(Ω)`); this floor does not by itself guarantee positive-definiteness of a full block Ω.

**Burn-in.** This M-step is suppressed for the first `omega_burnin` iterations (default 20, clamped to `n_exploration`): Ω is held at its starting value while the MH chain warms up. The MH proposal scale is \\( \delta_i \cdot \mathrm{chol}(\Omega) \\), so Ω and the sampler are coupled. On sparse data (few observations per subject) a cold-start chain (η = 0, only `n_mh_steps` proposals) produces a tiny \\( (1/N) \sum_i \eta_i \eta_i^T \\); with \\( \gamma_1 = 1 \\) the M-step would install that as Ω on iteration 1, shrinking the proposal, which keeps the chain near zero — a self-reinforcing collapse that dumps between-subject variability into the residual error. The SA statistic \\( S_2 \\) is still refreshed each burn-in iteration (at the damped rate \\( \gamma_k^\Omega \\), so it is a running average of the warming chain), so the first Ω update after burn-in reflects the warmed-up chain rather than the cold-start spread. Set `omega_burnin = 0` to disable the burn-in; note that the damped Ω step above now also guards this cold-start collapse on its own (it is a strict generalisation — continuous rather than for the first `omega_burnin` iterations only), so disabling the burn-in no longer reproduces the collapse by itself.

### 4. M-Step for Theta and Sigma (Optimization)

Minimize the conditional observation negative log-likelihood with ETAs held fixed:

<div>
\[ \sum_{i=1}^{N} \sum_{j=1}^{n_i} \left[ \frac{1}{2} \log V_{ij} + \frac{1}{2} \frac{(y_{ij} - f_{ij})^2}{V_{ij}} \right] \]
</div>

When `mu_referencing = true` (the default), ferx detects lognormal parameters from the `[individual_parameters]` block and applies the **closed-form EM update** for those thetas instead of running NLopt:

<div>
\[ \log \theta_j \leftarrow \log \theta_j + \gamma_k \cdot \overline{\eta_j} \]
</div>

where \\( \overline{\eta_j} = (1/N) \sum_i \eta_{i,j} \\) is the empirical mean of the post-MH random effects for the eta paired with \\( \theta_j \\). For a log-mu-referenced model where \\( \log P_i = \log \theta + \eta_i \\) with \\( \eta_i \sim N(0, \omega^2) \\), this is exactly the M-step that maximises the complete-data log-likelihood, scaled by the SA step size \\( \gamma_k \\). After the update the etas are re-centred by the same shift so they remain deviations from the new \\( \log \theta_j \\).

NLopt still runs for any remaining thetas (non-mu-referenced) and for sigma — the closed-form-updated thetas are pinned at their new values for the NLopt call.

The NLopt M-step uses **BOBYQA** (derivative-free trust-region with quadratic interpolation). The earlier gradient-based SLSQP path was found to lock onto one side of the Emax-Hill identifiability ridge on the dense-Emax PKPD benchmark (under-estimating EMAX by ~40% at virtually identical OFV); BOBYQA's quadratic trust-region exploration lands much closer to truth and ~40% faster on that benchmark (no FD-gradient eval per parameter), while remaining numerically equivalent (ΔOFV < 0.1) on simpler PK-only models.

When `mu_referencing = false`, the full NLopt M-step runs for all thetas as before.

The number of NLopt evaluations saved is stored in `FitResult::saem_mu_ref_m_step_evals_saved`, accumulated across SAEM iterations as `2 × mstep_maxiter × n_mu_ref_pairs` per outer step (one finite-difference probe pair per pinned mu-ref dimension, capped at `mstep_maxiter` NLopt gradient requests). The field is `None` when mu-referencing is off or method ≠ SAEM.

When `n_leapfrog > 0`, `FitResult::saem_n_subjects_hmc` records how many subjects used HMC at least once during the E-step (the remainder used MH fallback). The field is `None` for MH-only runs. The fit YAML also emits `saem_n_subjects_hmc` and `saem_n_subjects_mh` when the field is `Some`.

### 5. Adaptive Step Sizes

Every `adapt_interval` iterations, the per-subject step sizes \\( \delta_i \\) (MH) or leapfrog step sizes (HMC) are adjusted based on acceptance rate:
- If acceptance rate exceeds the target: increase \\( \delta_i \\) by 10% (up to 5.0)
- If acceptance rate falls below the target: decrease \\( \delta_i \\) by 10% (down to 0.01)

The block kernel and the componentwise kernel carry independent per-subject scales (the componentwise scale is adapted toward the higher 1-D optimum), and the kappa MH (IOV) scale is adapted the same way.

Target acceptance rates: 40% for the block MH kernel, 44% for the componentwise kernel, 65% for HMC.

## Post-SAEM Finalization

After the SAEM iterations complete:

1. **EBE Refinement**: Run the standard FOCE inner loop (BFGS optimization) warm-started from the SAEM ETAs to obtain final empirical Bayes estimates
2. **Combined-error additive-collapse repair**: for `combined(PROP, ADD)`
   residual-error models, *only when SAEM has driven the additive component
   `ADD` onto its lower bound*, ferx runs a final FOCEI marginal-likelihood
   polish from the SAEM estimates (retrying from the model's initial parameters
   if `ADD` is still pinned) and adopts it **only if it lowers the marginal
   OFV**. Fits whose `ADD` converged to a healthy non-zero value are left
   untouched. This guards against point-η SAEM M-steps overfitting the
   low-concentration tail and collapsing `ADD` when the marginal likelihood
   identifies a non-zero additive term. It is a safety net, not a substitute
   for checking `ADD` identifiability yourself (RSE, σ-correlation, profile
   likelihood); on sparse data prefer the importance-sampling −2LL over the
   Laplace OFV when judging whether the additive term is real.
3. **FOCE OFV**: Compute the objective function using the FOCE/Laplace approximation, so AIC and BIC are directly comparable with FOCE results
4. **Covariance Step**: Optionally compute standard errors via finite-difference Hessian (same method as FOCE)
5. **Diagnostics**: Compute PRED, IPRED, CWRES, IWRES for each subject

For sparsely-sampled data where the Laplace OFV is biased, you can
append an importance-sampling stage that estimates `−2 log L` by Monte
Carlo:

```
method = [saem, imp]
```

See [Importance Sampling (IMP)](importance-sampling.md).

## Inter-Occasion Variability (IOV)

`method = saem` supports models with `kappa` declarations (`n_kappa > 0`). IOV is handled with a per-occasion Gibbs Metropolis-Hastings step interleaved with the standard eta MH:

- **E-step**: After sampling \\( \eta \\), one MH proposal is made for each occasion's \\( \kappa_k \\). Both the eta and kappa samplers target the correct conditional distributions: \\( p(\eta | \kappa, \theta, \text{data}) \\) and \\( p(\kappa_k | \eta, \theta, \text{data}) \\) respectively.
- **SA update**: \\( S_2^{\text{iov}} \leftarrow (1 - \gamma_k) S_2^{\text{iov}} + \gamma_k \cdot \frac{1}{N_{\text{occ}}} \sum_i \sum_k \kappa_{ik} \kappa_{ik}^T \\)
- **M-step**: \\( \Omega_{\text{iov}} = S_2^{\text{iov}} \\) (analytic update, same structure as the BSV omega M-step).

No additional configuration is required; `method = saem` works for both BSV-only and IOV models.

## Configuration

```
[fit_options]
  method        = saem
  n_exploration = 150      # Phase 1 iterations
  n_convergence = 250      # Phase 2 iterations
  n_mh_steps    = 20       # block-MH steps per subject per iteration (ignored when n_leapfrog > 0)
  n_leapfrog    = 0        # Set > 0 (e.g. 3) to use HMC instead of MH
  adapt_interval = 50      # Step-size adaptation frequency
  omega_burnin  = 20       # Iterations to hold Ω fixed while the chain warms up
  seed          = 12345    # RNG seed for reproducibility
  covariance    = true     # Compute standard errors
```

## Tuning Guide

### Not Converging

- Increase `n_exploration` (e.g., 300) to give more time for basin finding
- Increase `n_convergence` (e.g., 500) for a longer averaging window
- Raise `n_mh_steps` further (e.g. 30-50) for better mixing in the E-step
  on hard surfaces — the default 20 is calibrated to escape the basin trap
  observed on Emax PKPD with stressful initial values (see below) and to
  drive the componentwise kernel that prevents block-Ω collapse, but
  ODE-with-Form-C readouts may need more proposals to fully decorrelate
  samples between M-step calls.

### PD-curve thetas collapse on cold start (Emax / sigmoid readouts)

A failure mode specific to models that read population thetas through a Form
C `[scaling]` block (e.g. `y[CMT=N] = E0 + EMAX * effect^GAMMA / ...`) and
score them via a per-CMT additive `[error_model]`: from stressful initial
values (e.g. 1.5× truth) the M-step can lock the PD-curve thetas into a
degenerate basin where `E0 → 0`, `EMAX` and `EC50` blow up, and `GAMMA`
collapses below 1. The likelihood at the bad basin is only modestly worse
than at truth (~150 OFV units on a 100-subject benchmark), so SAEM doesn't
back out on its own.

The underlying cause is MCMC sample correlation: with an early default
`n_mh_steps = 3` the chain didn't decorrelate enough between SAEM outer
iterations, so the single-draw stochastic M-step received sticky correlated
ETAs that biased the population-θ update toward the basin. The default was
raised to 10 and then to 20 (alongside the componentwise kernel and the
damped Ω step), which resolves this reliably across seeds at modest extra
wall on the affected model and ~0% on simpler PK-only models.

If you still see this signature (E0 hitting its lower bound, EMAX large,
EC50 large) on a related Emax/Hill model:

1. Try `n_mh_steps = 20` or `n_mh_steps = 50`
2. Warm-start from a FOCEI fit (`method = [focei, saem]`)
3. Run with several seeds and keep the lowest OFV

### Ω Collapses / Residual Error Inflates

On sparse data (few observations per subject) the variance components can collapse toward zero on the first iterations while the residual error absorbs the between-subject variability (e.g. tiny `omega` with a large additive `sigma`). The default `omega_burnin = 20` and the damped Ω SA step guard against this by keeping Ω near its starting value while the chain warms up. If it still occurs, raise `omega_burnin` (e.g. 40) and/or `n_mh_steps` so the chain reaches a representative spread before Ω is first estimated, then polish with `method = [saem, focei]`.

### Block Ω correlations near ±1 (rank-1 collapse)

For a `block_omega` (correlated) random-effects block, a faulty E-step/M-step coupling can drive **every** off-diagonal correlation toward ±1 while one variance collapses toward zero — a near rank-1 Ω that FOCEI on the same data does not show. The mechanism is the block proposal (preconditioned by `chol(Ω)`) plus the single-draw Ω M-step feeding correlation back into Ω during the `γ=1` exploration phase. ferx guards this by default with two mechanisms — the **componentwise MH kernel** and the **damped Ω SA step** described above — so on a poorly-identified 2-cpt model the SAEM Ω now matches the FOCEI/NONMEM estimate (e.g. corr(CL,V1) ≈ 0.67, corr(V1,V2) ≈ 0.4) across seeds instead of collapsing to ≈0.99. If you still see inflated block correlations, raise `n_mh_steps` (this also raises the componentwise sweep count `n_mh_steps / n_eta`) and run several seeds.

### Slow Convergence

- Decrease `n_exploration` and `n_convergence` if parameters stabilize early
- Use `adapt_interval = 25` for faster step-size adaptation

### Reproducibility

- Always set `seed` for reproducible results
- Different seeds will produce slightly different estimates due to the stochastic nature of the algorithm

## Output

The SAEM iteration progress is printed to stderr:

```
SAEM: 10 subjects, 3 ETAs, 400 total iter (150 explore + 250 converge)
  SAEM iter    1/400 [explore] γ=1.000  condNLL=95.244
  SAEM iter   50/400 [explore] γ=1.000  condNLL=56.705
  SAEM iter  150/400 [explore] γ=1.000  condNLL=46.071
  SAEM iter  200/400 [converge] γ=0.020  condNLL=36.799
  SAEM iter  400/400 [converge] γ=0.004  condNLL=38.096
SAEM iterations complete. Computing final EBEs and OFV...
SAEM completed. Final OFV = ...
```

### Why `γ` (gamma) is shown

\\( \gamma_k \\) is the **stochastic-approximation step size**, not a model quantity — it is intrinsic to the SAEM algorithm and tells you which phase the run is in and how aggressively the estimates are still moving:

- **Exploration** (`[explore]`, \\( k \le K_1 \\)): \\( \gamma_k = 1 \\). Each iteration fully replaces the running sufficient statistics, so the chain roams freely toward the basin of the MLE.
- **Convergence** (`[converge]`, \\( k > K_1 \\)): \\( \gamma_k = 1/(k - K_1) \\) decays toward zero. Updates shrink into a decreasing-weight average that damps the Monte-Carlo noise so the estimates settle. The printed value (e.g. `0.020`, `0.004`) is exactly this decaying weight applied to the Ω, θ, and σ updates each iteration.

### Why `condNLL` and not `OFV`

During the iterations ferx prints `condNLL`, the **conditional** (joint) negative log-likelihood summed over subjects, evaluated at the *current MH/HMC-sampled* etas:

<div>
\[ \text{condNLL} = \sum_{i=1}^{N} \text{NLL}(\eta_i^{\text{sampled}}) \]
</div>

This is a cheap per-iteration progress signal. It is **not** the marginal objective function value (OFV): it is evaluated at one stochastic draw of the random effects rather than integrated over their distribution, so unlike the FOCE/FOCEI outer-loop OFV it is noisy, will not decrease monotonically, and is **not comparable across runs** for model selection.

The true marginal **OFV** (\\( -2 \log L \\) via the Laplace approximation, directly comparable with FOCE for AIC/BIC) is expensive and is therefore computed only once, after the iterations finish — this is the `Final OFV = ...` line. See [Post-SAEM Finalization](#post-saem-finalization).

(`NLL` = *negative log-likelihood*, i.e. \\( -\log L \\). NONMEM's `OFV = -2 \log L` is essentially `2 × NLL` plus a constant.)

`condNLL` should generally decrease during the exploration phase and stabilize during convergence.
