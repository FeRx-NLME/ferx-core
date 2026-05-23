# SAEM

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

\\[ p(\eta_i | y_i, \theta, \Omega, \sigma) \\]

Two samplers are available:

#### Metropolis-Hastings (default, `n_leapfrog = 0`)

Run `n_mh_steps` symmetric random-walk MH iterations per subject per SAEM iteration.

**Proposal**: \\( \eta_{\text{prop}} = \eta_{\text{current}} + \delta_i \cdot L \cdot z \\), with \\( z \sim N(0, I) \\) and \\( L = \text{chol}(\Omega) \\). The schedule is identical across both phases — only the SA step size \\( \gamma_k \\) changes between exploration and convergence.

The MH kernel is symmetric in \\( \eta \\), so the proposal density cancels and the acceptance log-ratio is the difference of `individual_nll` values, which encodes the prior \\( N(0, \Omega) \\) plus the observation likelihood.

**Acceptance**: \\( \min(1, \exp(\text{NLL}_{\text{current}} - \text{NLL}_{\text{prop}})) \\). Target acceptance rate: 40%.

#### HMC (Hamiltonian Monte Carlo, `n_leapfrog > 0`)

When `n_leapfrog` is set to a positive integer (e.g. `3`), one HMC proposal replaces the `n_mh_steps` MH proposals per subject per iteration. HMC uses the gradient of the individual NLL to make longer, more directed moves through the posterior:

\\[ H(\eta, p) = \text{NLL}(\eta) + \tfrac{1}{2} \|p\|^2 \\]

with momentum \\( p \sim N(0, I) \\) and a standard velocity Störmer-Verlet (leapfrog) integrator. Acceptance is on \\( \Delta H \\), targeting ~65% acceptance.

**Requirements**: `autodiff` feature enabled (default) and an analytical PK model (no ODE). A warning is emitted if `n_leapfrog > 0` but the `autodiff` feature is absent.

**Per-subject fallback to MH**: `hmc_step` silently falls back to MH for a subject when any of the following conditions hold:
- The model uses an ODE (`[odes]` block present)
- The model has no analytical PK path (no `tv_fn` — pure ODE-only models)
- The Ω matrix has a non-finite log-determinant (degenerate variance)
- The subject has time-varying covariates and either the PK model does not support the event-driven AD path or the model has a lag time

In a single run with `n_leapfrog > 0`, different subjects can therefore use different samplers. The acceptance rate reported in verbose output and the optimizer trace is an aggregate across all subjects; in mixed HMC/MH runs the target (65% for HMC, 40% for MH) may not be meaningful for the aggregate. The `n_mh_steps` option governs the number of proposals for MH-fallback subjects even when `n_leapfrog > 0`.

The E-step sampling is parallelized across subjects using Rayon.

### 2. Stochastic Approximation Update

Update the sufficient statistic for \\( \Omega \\):

\\[ S_2 \leftarrow (1 - \gamma_k) \cdot S_2 + \gamma_k \cdot \frac{1}{N} \sum_{i=1}^{N} \eta_i \eta_i^T \\]

### 3. M-Step for Omega (Closed Form)

\\[ \Omega_k = S_2 \\]

with structurally-zero entries (cross-block off-diagonals, standalone-vs-block off-diagonals, and all off-diagonals in a fully-diagonal Ω) zeroed out — the SA accumulator \\( (1/N) \sum_i \eta_i \eta_i^T \\) is dense by construction, but the model declares which entries are free parameters. Without this projection the chain feeds spurious sampling correlations into the next iteration's MH proposal Cholesky and Ω drifts toward rank-deficiency.

### 4. M-Step for Theta and Sigma (Optimization)

Minimize the conditional observation negative log-likelihood with ETAs held fixed:

\\[ \sum_{i=1}^{N} \sum_{j=1}^{n_i} \left[ \frac{1}{2} \log V_{ij} + \frac{1}{2} \frac{(y_{ij} - f_{ij})^2}{V_{ij}} \right] \\]

When `mu_referencing = true` (the default), ferx detects lognormal parameters from the `[individual_parameters]` block and applies the **closed-form EM update** for those thetas instead of running NLopt:

\\[ \log \theta_j \leftarrow \log \theta_j + \gamma_k \cdot \overline{\eta_j} \\]

where \\( \overline{\eta_j} = (1/N) \sum_i \eta_{i,j} \\) is the empirical mean of the post-MH random effects for the eta paired with \\( \theta_j \\). For a log-mu-referenced model where \\( \log P_i = \log \theta + \eta_i \\) with \\( \eta_i \sim N(0, \omega^2) \\), this is exactly the M-step that maximises the complete-data log-likelihood, scaled by the SA step size \\( \gamma_k \\). After the update the etas are re-centred by the same shift so they remain deviations from the new \\( \log \theta_j \\).

NLopt still runs for any remaining thetas (non-mu-referenced) and for sigma — the closed-form-updated thetas are pinned at their new values for the NLopt call.

When `mu_referencing = false`, the full NLopt M-step runs for all thetas as before.

The number of NLopt evaluations saved is stored in `FitResult::saem_mu_ref_m_step_evals_saved`, accumulated across SAEM iterations as `2 × mstep_maxiter × n_mu_ref_pairs` per outer step (one finite-difference probe pair per pinned mu-ref dimension, capped at `mstep_maxiter` NLopt gradient requests). The field is `None` when mu-referencing is off or method ≠ SAEM.

When `n_leapfrog > 0`, `FitResult::saem_n_subjects_hmc` records how many subjects used HMC at least once during the E-step (the remainder used MH fallback). The field is `None` for MH-only runs. The fit YAML also emits `saem_n_subjects_hmc` and `saem_n_subjects_mh` when the field is `Some`.

### 5. Adaptive Step Sizes

Every `adapt_interval` iterations, the per-subject step sizes \\( \delta_i \\) (MH) or leapfrog step sizes (HMC) are adjusted based on acceptance rate:
- If acceptance rate exceeds the target: increase \\( \delta_i \\) by 10% (up to 5.0)
- If acceptance rate falls below the target: decrease \\( \delta_i \\) by 10% (down to 0.01)

Target acceptance rates: 40% for MH, 65% for HMC.

## Post-SAEM Finalization

After the SAEM iterations complete:

1. **EBE Refinement**: Run the standard FOCE inner loop (BFGS optimization) warm-started from the SAEM ETAs to obtain final empirical Bayes estimates
2. **FOCE OFV**: Compute the objective function using the FOCE/Laplace approximation, so AIC and BIC are directly comparable with FOCE results
3. **Covariance Step**: Optionally compute standard errors via finite-difference Hessian (same method as FOCE)
4. **Diagnostics**: Compute PRED, IPRED, CWRES, IWRES for each subject

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
  n_mh_steps    = 3        # MH steps per subject per iteration (ignored when n_leapfrog > 0)
  n_leapfrog    = 0        # Set > 0 (e.g. 3) to use HMC instead of MH
  adapt_interval = 50      # Step-size adaptation frequency
  seed          = 12345    # RNG seed for reproducibility
  covariance    = true     # Compute standard errors
```

## Tuning Guide

### Not Converging

- Increase `n_exploration` (e.g., 300) to give more time for basin finding
- Increase `n_convergence` (e.g., 500) for a longer averaging window
- Increase `n_mh_steps` (e.g., 5-10) for better mixing in the E-step

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
  SAEM iter    1/400 [explore] gamma=1.000  condNLL=95.244
  SAEM iter   50/400 [explore] gamma=1.000  condNLL=56.705
  SAEM iter  150/400 [explore] gamma=1.000  condNLL=46.071
  SAEM iter  200/400 [converge] gamma=0.020  condNLL=36.799
  SAEM iter  400/400 [converge] gamma=0.004  condNLL=38.096
SAEM iterations complete. Computing final EBEs and OFV...
```

The `condNLL` is the conditional observation negative log-likelihood (not the final OFV). It should generally decrease during the exploration phase and stabilize during convergence.
