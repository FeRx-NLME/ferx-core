# ferx-core

[![CI](https://github.com/FeRx-NLME/ferx-core/actions/workflows/ci.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/ci.yml)
[![Slow tests](https://github.com/FeRx-NLME/ferx-core/actions/workflows/slow-tests.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/slow-tests.yml)
[![Docs](https://github.com/FeRx-NLME/ferx-core/actions/workflows/docs.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/FeRx-NLME/ferx-core/branch/main/graph/badge.svg)](https://codecov.io/gh/FeRx-NLME/ferx-core)
[![CodeFactor](https://www.codefactor.io/repository/github/ferx-nlme/ferx-core/badge)](https://www.codefactor.io/repository/github/ferx-nlme/ferx-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics, written in Rust. Implements FOCEI and SAEM estimation with analytical PK solutions and ODE solvers.

Additional features:
- PK-PD and multi-analyte modeling
- BLQ likelihood modeling
- Importance Sampling & SIR
- Deep Compartmental Models & Neural ODEs
- Stochastic differential equations
- Simulation with uncertainty
- Various optimizers
- ... and more

## Quick Start

```bash
# Build
cargo build --release

# Fit a model
cargo run --release --bin ferx -- examples/warfarin.ferx --data data/warfarin.csv

# Fit with simulated data (uses [simulation] block)
cargo run --release --bin ferx -- examples/warfarin.ferx --simulate
```

Output files: `{model}-fit.yaml` (parameter estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Model File Format (.ferx)

Models are defined in a simple DSL. Here is a one-compartment oral PK model for warfarin:

```
[parameters]
  theta TVCL(0.2, 0.001, 10.0)     # name(initial, lower, upper)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09              # between-subject variability (variance)
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02            # residual error

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

## Structural Models

| Model | Syntax |
|-------|--------|
| 1-compartment IV (bolus and/or infusion) | `pk one_cpt_iv(cl=CL, v=V)` |
| 1-compartment oral | `pk one_cpt_oral(cl=CL, v=V, ka=KA)` |
| 2-compartment IV (bolus and/or infusion) | `pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)` |
| 2-compartment oral | `pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)` |
| 3-compartment IV (bolus and/or infusion) | `pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)` |
| 3-compartment oral | `pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)` |
| ODE-based | Define equations in an `[odes]` block |

For IV models, the closed form (bolus vs infusion) is chosen per dose event from the `RATE` column — a subject can mix bolus and infusion records.

## Estimation Methods

Set via `method` in `[fit_options]`:

| Method | Description |
|--------|-------------|
| `foce` | First-Order Conditional Estimation |
| `focei` | FOCE with Interaction (default) |
| `gn` | Gauss-Newton (BHHH) with Levenberg-Marquardt damping |
| `gn_hybrid` | Gauss-Newton followed by FOCEI polish |
| `saem` | Stochastic Approximation EM |
| `imp` | Importance Sampling (typically chained after another method for OFV evaluation) |

Methods can be chained (e.g. `method = saem, focei, imp`) to run sequentially.

### Optimizers

For FOCE/FOCEI, the outer optimizer can be set via `optimizer` in `[fit_options]`:

| Optimizer | Description |
|-----------|-------------|
| `slsqp` | NLopt Sequential Least Squares Programming (default) |
| `lbfgs` | NLopt L-BFGS |
| `mma` | NLopt Method of Moving Asymptotes |
| `bfgs` | Built-in BFGS |
| `bobyqa` | NLopt BOBYQA (derivative-free) |
| `trust_region` | Newton trust-region (argmin + Steihaug CG) |

## Vine Copula Between-Subject Variability (`omega_dist = vine`)

### What it is and why it matters

Every NLME model ever fitted in NONMEM, nlmixr2, or the standard ferx FOCE/SAEM path makes the same assumption: the between-subject random effects η follow a **multivariate normal** distribution, η ~ N(0, Ω). The OMEGA matrix fully describes the shape of that distribution — its variances and covariances.

This assumption is convenient but restrictive. Real patient populations sometimes show:

- **Heavy tails** — more extreme subjects than a Gaussian predicts, inflating individual EBE-based residuals
- **Skewed marginals** — clearance distributions with a longer upper tail than lower (or vice versa)
- **Tail-dependent correlations** — subjects at the extremes of CL and V may cluster together more (or less) strongly than subjects near the centre of the distribution. A Pearson correlation in Ω cannot distinguish between "highly correlated only in the tails" and "uniformly correlated everywhere"

These features can cause:
- Biased population parameter estimates (θ, Ω) when the Gaussian misspecification is severe
- Inflated residuals and poor CWRES/IWRES diagnostics in specific subject groups
- Suboptimal individual predictions (IPREDs) for outlier subjects

`omega_dist = vine` replaces the multivariate-Gaussian BSV with a **D-vine pair-copula construction**. The marginal distribution of each η_i remains Gaussian (so the OMEGA diagonal is still interpretable), but the joint dependence structure is estimated from the data and can be non-Gaussian. Estimation runs via SAEM with a vine-prior Metropolis-Hastings E-step.

---

### For pharmacometricians

**The intuition**

In a standard fit, OMEGA simultaneously describes two things: how spread out each ETA is, and how the ETAs are related to each other. The vine approach separates these:

1. **How spread out is each ETA?** — still described by the OMEGA diagonal and reported as the marginal SD. Nothing changes here.
2. **How are the ETAs related to each other?** — instead of a single Pearson correlation, a bivariate *copula* is fitted for each ETA pair. Different copula families can capture different shapes of dependence.

The copula families available are:

| Family | What it captures |
|--------|-----------------|
| `gaussian` | The usual symmetric correlation (same as standard OMEGA off-diagonal) |
| `student_t` | Symmetric correlation but with heavier joint tails — extreme subjects in *both* ETAs simultaneously are more likely than a Gaussian predicts |
| `clayton` | **Lower** tail dependence — patients with unusually *low* values in both parameters tend to cluster together |
| `gumbel` | **Upper** tail dependence — patients with unusually *high* values in both parameters tend to cluster together |
| `frank` | Symmetric dependence without tail effects, more flexible than Gaussian |

Families are selected automatically by AIC on the first SAEM M-step and frozen for the rest of the run.

**When should you try this?**

- When η histograms or κ–η scatterplots look non-normal or show curvature at the extremes
- When you observe a subset of subjects with consistently poor IPREDs despite adequate average fit
- As a diagnostic: if every pair-copula selects `gaussian`, your standard model BSV structure is confirmed adequate
- When comparing competing structural models and you want the fairest possible OFV baseline

**Quick start**

Add `omega_dist = vine` to any model that already has a `method = saem` line:

```
[fit_options]
  method     = saem
  omega_dist = vine
```

For a SAEM exploration + FOCEI convergence workflow (recommended for final runs):

```
[fit_options]
  method     = saem, focei
  omega_dist = vine
```

The vine structure is fitted during the SAEM phase. The subsequent FOCEI step uses the Gaussian-equivalent OMEGA for the Laplace approximation (identical to standard chained estimation) while the vine distribution drives simulation.

**Reading the output**

The vine section in the console and YAML output reports, for each ETA pair at each tree level:

- **Family** — which copula was selected
- **Parameter** with approximate SE — e.g. `rho=0.431 (SE≈0.087)` for a Gaussian or Student-t pair, `theta=1.82 (SE≈0.41)` for Clayton/Gumbel/Frank
- **Kendall τ** — rank-based correlation, comparable across all families (unlike Pearson r which only makes sense for Gaussian)
- **λL / λU** — lower and upper tail dependence coefficients; values above ~0.1 indicate meaningful tail clustering

The `ΔOFV (Gaussian − corrected)` value in the output is the most important summary: a positive value means the vine model fits the data better than a Gaussian BSV on the same structural model, with full accounting for the extra copula parameters in the OFV baseline. Rule of thumb: ΔOFV > 3.84 per additional copula parameter (chi-squared approximation) suggests a meaningful improvement.

> **Note on the two OFV values reported:** The standard FOCE OFV uses the Gaussian-equivalent OMEGA and is printed for consistency with Gaussian fits. The *vine-corrected OFV* replaces the Gaussian BSV prior with the vine prior at the final EBEs — this is the value used for AIC and BIC and is the fair comparator to a Gaussian FOCEI OFV.

**IOV models**

Vine fits fully support models with inter-occasion variability (`kappa` declarations). The vine applies to the BSV (η) layer only. The IOV (κ) layer retains a Gaussian Ω_iov. This is scientifically appropriate: the vine captures cross-subject structure in the BSV distribution, while IOV captures within-subject occasion-to-occasion variability — a lower-signal process for which a vine copula would not be identifiable from typical NLME datasets.

---

### For statisticians

**Model**

The joint BSV density is factored as a product of Gaussian marginals and a D-vine copula density:

```
p(η₁, …, η_d) = ∏ᵢ φ(ηᵢ; μᵢ, σᵢ)  ×  c_vine(Φ(z₁), …, Φ(z_d))
```

where zᵢ = (ηᵢ − μᵢ)/σᵢ and uᵢ = Φ(zᵢ) is the probability integral transform (PIT). The D-vine copula density decomposes over d−1 tree levels into bivariate pair-copula densities via the sequential h-function recursion (Aas et al. 2009):

```
c(u₁,…,u_d) = ∏_{k=1}^{d-1} ∏_{j=0}^{d-k-1} c_{j,j+k|j+1,…,j+k-1}(u_{j|·}, u_{j+k|·})
```

The D-vine ordering is fixed at 0, 1, …, d−1 (no permutation selection in the current implementation).

**Estimation**

SAEM with a vine-prior Metropolis-Hastings E-step. The complete-data log-likelihood uses:

```
log p(ηᵢ) = Σⱼ [−½zᵢⱼ² − ½log(2π) − log σⱼ] + log c_vine(u₁,…,u_d)
```

The M-step separates into:
- **Marginals** (MLE, closed form): μⱼ ← mean(ηⱼ), σⱼ ← std(ηⱼ) from current SAEM samples
- **Pair-copulas** (IFM — Inference Functions for Margins): each pair copula is independently refitted by maximising the pair log-likelihood at the conditional pseudo-observations obtained by sequential h-function application through the vine tree structure. Family selection (AIC over the five candidate families above) occurs at the first M-step and is frozen thereafter to prevent log-prior discontinuities from disturbing the chain.

**Vine-corrected OFV**

The `vine_corrected_ofv` replaces the Gaussian prior in the standard FOCE population NLL with the vine prior at the post-SAEM final EBEs:

```
OFV_vine = OFV_FOCE  +  2 × Σᵢ [ log p_vine(η̂ᵢ) − log p_Gauss(η̂ᵢ; Ω_equiv) ]
```

This correction removes the Laplace-approximation error from the prior mismatch and yields an OFV that is directly comparable to a Gaussian FOCEI OFV on the same data.

**AIC / BIC**

```
k  =  k_theta  +  k_Ω (free Cholesky elements)  +  k_sigma  +  Σ_{pairs} n_params(family)
AIC = OFV_vine + 2k
BIC = OFV_vine + k × ln(N_obs)
```

Student-t pair-copulas contribute 2 (ρ and ν); all other families contribute 1.

**Copula parameter standard errors**

Per-pair SEs are approximate under the IFM assumption (pseudo-observations treated as fixed). The observed-information Hessian is computed by finite differences on the transformed scale:

| Family | Transform ψ(θ) | Back-transform SE |
|--------|---------------|-------------------|
| Gaussian / Student-t ρ | arctanh(ρ) | SE_ρ ≈ (1 − ρ²) × SE_ψ |
| Student-t ν | ln(ν) | SE_ν ≈ ν × SE_ψ |
| Clayton θ | ln(θ) | SE_θ ≈ θ × SE_ψ |
| Gumbel θ | ln(θ − 1) | SE_θ ≈ (θ − 1) × SE_ψ |
| Frank θ | identity | SE_θ = SE_ψ |

Rigorous SEs require a Godambe sandwich estimator (not yet implemented) that accounts for the uncertainty in the marginal PIT.

**Simulation**

Draws from the fitted vine use the inverse Rosenblatt transform (exact, O(d²) per draw): d independent U(0,1) variates are mapped through sequential h-inverse functions from the outermost tree level inward, then inverted through the Gaussian marginals. No MCMC is required.

---

### Fit options reference

| Option | Values | Default | Notes |
|--------|--------|---------|-------|
| `omega_dist` | `gaussian`, `vine` | `gaussian` | Must also set `method = saem` |
| `saem_n_exploration` | integer | 200 | Iterations for vine family selection and chain warm-up |
| `saem_n_convergence` | integer | 100 | Iterations for parameter convergence |
| `saem_omega_burnin` | integer | 50 | Iterations before vine M-step begins (chain warm-up) |
| `saem_n_mh_steps` | integer | 3 | MH proposals per subject per SAEM iteration |
| `covariance` | `true`/`false` | `false` | Covariance step for θ/σ SEs; copula SEs always computed |

---

### Output reference (vine additions)

Console output adds a `--- Vine Copula ---` block after the standard OMEGA/SIGMA table. YAML output adds a `vine_copula:` top-level key. Example (3 ETAs, tree 1 has 2 pairs, tree 2 has 1):

```yaml
vine_copula:
  marginals:
    ETA_CL:  { mean: -0.008, sd: 0.302 }
    ETA_V:   { mean:  0.003, sd: 0.194 }
    ETA_KA:  { mean:  0.011, sd: 0.556 }
  trees:
    - tree: 1                              # unconditional pairs
      pairs:
        - label: "ETA_CL ~ ETA_V"
          family: student_t
          rho:    0.431
          rho_se: 0.087                    # approximate SE (IFM)
          nu:     8.2
          nu_se:  3.1
          kendall_tau: 0.283
          tail_dep_lower: 0.124
          tail_dep_upper: 0.124
        - label: "ETA_V ~ ETA_KA"
          family: clayton
          theta:    1.24
          theta_se: 0.38
          kendall_tau: 0.383
          tail_dep_lower: 0.402
    - tree: 2                              # conditioned on ETA_V
      pairs:
        - label: "ETA_CL ~ ETA_KA | ETA_V"
          family: gaussian
          rho:    0.104
          rho_se: 0.096
          kendall_tau: 0.066

  ofv_vine_corrected:       1243.218       # use this for model comparison
  delta_ofv_vine_advantage:   12.441       # positive = vine improves fit vs Gaussian BSV
```

All standard output (theta, omega, sigma, AIC, BIC, sdtab, EBEs) is unchanged. AIC and BIC are computed from `ofv_vine_corrected` with the augmented parameter count.

---

### Limitations and roadmap

- **Variable ordering**: the D-vine ordering is currently fixed as declared in `[parameters]`. Optimal ordering or R-vine structure selection is not yet implemented.
- **Copula SEs**: approximate only (IFM assumption). Godambe sandwich SEs are a planned improvement.
- **HMC E-step**: the vine path always uses Metropolis-Hastings, regardless of `saem_n_leapfrog`. The gradient of the vine log-prior is not yet implemented, so HMC is silently ignored for vine fits (a warning is emitted). HMC (via the `autodiff` feature) is only available for Gaussian SAEM with analytical PK models.
- **ODE structural models**: fully supported — `[odes]`-based models work with `omega_dist = vine`. The MH E-step evaluates the ODE solver inside each proposal exactly as in Gaussian SAEM.

---

### References

- Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics* **44**, 182–198.
- Joe, H. (1996). Families of m-variate distributions with given margins and m(m−1)/2 bivariate dependence parameters. *IMS Lecture Notes–Monograph Series* **28**, 120–141.
- Delattre, M., Lavielle, M., Poursat, M.-A. (2014). A note on BIC in mixed-effects models. *Electronic Journal of Statistics* **8**(1), 456–475.

---

## Data Format

Input data uses NONMEM-format CSV with columns:

- **Required**: `ID`, `TIME`, `DV`, `EVID`, `AMT`, `CMT`
- **Optional**: `RATE`, `MDV`, `II`, `SS`
- **Covariates**: Any additional columns are auto-detected

EVID codes: 0 = observation, 1 = dose, 4 = reset + dose.

## Examples

The `examples/` directory contains ready-to-run models:

| File | Description |
|------|-------------|
| `warfarin.ferx` | 1-compartment oral (warfarin PK) |
| `two_cpt_iv.ferx` | 2-compartment IV bolus |
| `two_cpt_oral_cov.ferx` | 2-compartment oral with covariates (WT, CRCL) |
| `mm_oral.ferx` | Michaelis-Menten elimination via ODE |

## R Package

An R wrapper package (`ferx`) provides `ferx_fit()`, `ferx_simulate()`, and `ferx_predict()` functions that call into this Rust engine via [extendr](https://extendr.github.io/). Source is at `../ferx`.

### Installation

```r
# Build the Rust backend and load the package
withr::with_dir("path/to/ferx", {
  system("cd src/rust && cargo build --release")
  devtools::load_all()
})
```

### Fitting a model

```r
result <- ferx_fit(
  model = "warfarin.ferx",
  data  = "warfarin.csv",
  method = "foce"        # or "focei"
)

result                   # prints summary with estimates and SEs
result$theta             # named vector of fixed-effect estimates
result$omega             # BSV covariance matrix
result$sigma             # residual error estimates
result$se_theta          # standard errors (NULL if covariance step failed)
result$sdtab             # data.frame with ID, TIME, DV, PRED, IPRED, CWRES, IWRES, ETA1..n
```

### Simulation and VPC

```r
sim <- ferx_simulate("warfarin.ferx", "warfarin.csv", n_sim = 100, seed = 42)
# Returns data.frame with SIM, ID, TIME, IPRED, DV_SIM

library(vpc)
obs <- read.csv("warfarin.csv")
vpc(obs = obs, sim = sim, sim_cols = list(dv = "DV_SIM"))
```

### Population predictions

```r
preds <- ferx_predict("warfarin.ferx", "warfarin.csv")
# Returns data.frame with ID, TIME, PRED (predictions at eta = 0)
```

## License

MIT — see [LICENSE](LICENSE).
