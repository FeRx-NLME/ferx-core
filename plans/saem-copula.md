# Plan: SAEM-Copula (`omega_dist = vine`) — R-Vine & Zero-Mean Flexible Marginals for SAEM

## Goal

Add an opt-in `omega_dist = vine` SAEM estimator variant (#225) that models the
between-subject random-effect vector $\eta \in \mathbb{R}^d$ using zero-mean
flexible parametric marginals (`Normal`, `StudentT`, `SkewNormal`, Azzalini `SkewT`)
coupled by a regular vine (R-vine) pair-copula construction (`Independence`,
`Gaussian`, `StudentT`, `Frank`, and $0^\circ / 90^\circ / 180^\circ / 270^\circ$
rotations of `Clayton` and `Gumbel`).

The existing `omega_dist = gaussian` (default) SAEM code path is frozen and bit-identical.
Discrete subpopulations are intentionally out of scope here and handled by `[mixture]`.

## Three-Uses Consistency

One fitted `VineCopulaOmega` object drives all three downstream operations:
1. **Reporting**: marginal family, SD, skewness, excess kurtosis, R-vine tree
   structure, pair-copula families/rotations, parameters + observed-information SEs,
   Kendall's $\tau$, lower/upper tail-dependence coefficients, and the secondary
   Gaussian-equivalent $\Omega$.
2. **Simulation & Diagnostics**: `simulate()` and `npde` draw $\eta$ via the exact
   R-vine inverse-Rosenblatt transform (Dissmann et al. 2013, Algorithm 2.2) pushed
   through each marginal's quantile function.
3. **Individual Estimation & Marginal Likelihood**: MAP EBEs minimize
   $\ell_{\text{obs},i}(\eta) - \log p_{\text{vine}}(\eta)$ (with multi-start
   secondary-mode and Hessian PD checks), and the reported marginal objective
   $-2 \log L(\Psi)$ is evaluated by adaptive Owen-scrambled Sobol QMC importance
   sampling (`run_importance_sampling_vine`) with a defensive vine-prior mixture
   proposal — directly comparable to Gaussian `saem` + `saem_ofv = imp`.

## Statistical Design

### 1. Identifiability of Marginals and Mu-Referencing
Every marginal family pins its location parameter analytically so $\mathbb{E}[\eta_j] \equiv 0$:
- `Normal(sd)`: $\xi = 0$, $k = 1$.
- `StudentT(scale, nu)` ($\nu > 2.05$): $\xi = 0$, $\text{SD} = s\sqrt{\nu/(\nu - 2)}$, $k = 2$.
- `SkewNormal(omega, alpha)`: $\xi = -\omega \delta \sqrt{2/\pi}$ with $\delta = \alpha/\sqrt{1+\alpha^2}$, $k = 2$.
- `SkewT(omega, alpha, nu)` ($\nu > 2.05$): $\xi = -\omega \delta \sqrt{\nu/\pi}\,\Gamma((\nu-1)/2)/\Gamma(\nu/2)$, $k = 3$.

During SAEM's $\phi$-frozen mu-referencing step ($\phi_i = \mu(\theta) + \eta_i$),
$\mu(\theta)$ steps along the preconditioned vine prior score
$-\sum_i \nabla_\eta \log p_{\text{vine}}(\eta_i)$ and re-centers $\eta_i$ before the
marginal/copula M-step, satisfying Louis's finite-sample score identity with zero PK/ODE
evaluations and reducing identically to `mean(eta)` in the Gaussian case.

### 2. Two-Stage Exploration & Robbins-Monro Score/Information SA
- **Phase 1a ($1 \le k \le K_{\text{select}}$)**: Warm up under a Gaussian R-vine so
  $(\theta, \sigma, \Omega)$ reach the posterior basin first; buffer the last $M$
  re-centered conditional draws.
- **At $k = K_{\text{select}}$**: Select per-ETA marginal families (when
  `vine_marginals = auto`) and the Dissmann max-$|\tau|$ R-vine spanning trees +
  pair-copula families by BIC on the $1/M$-weighted pooled conditional log-likelihood
  (penalty $p \ln N$ on $N$ subjects, including `Independence` at $p = 0$). Freeze the
  selected structure and families for the rest of the run.
- **Phase 1b ($K_{\text{select}} < k \le K_1$) & Phase 2 ($k > K_1$)**: Update
  unconstrained marginal and pair-copula parameters via Robbins-Monro stochastic
  approximation of the score and positive-definite observed information
  ($\bar g_k, \bar I_k$), eliminating single-draw MLE Jensen and boundary bias.

### 3. Guards & Non-Goals
- Refuse `omega_dist = vine` without `saem` (`E_OMEGA_DIST_NO_SAEM`), in chains with
  Gaussian-prior estimators (`E_OMEGA_DIST_CHAIN`), with `sir = true` (`E_OMEGA_DIST_SIR`),
  and with `[mixture]` (`E_OMEGA_DIST_MIXTURE`).
- HMC proposals (`saem_n_leapfrog > 0`) emit a warning and fall back to the two-kernel
  (block + componentwise) Metropolis-Hastings E-step.

### 4. Cross-PR Merge Checklist (`#1721` → `#2669` → `#2703`, and Superseding `#193` / `ferx-r#133`)
- **Supersedes `#193` (`ferx-core`) and `ferx-r#133`**: Built cleanly on post-`#2939`
  `main` (`7d15fe88`), replacing the older D-vine / IFM / Laplace-approximation prototype
  with Dissmann R-vines, zero-mean flexible marginals, Robbins-Monro score/information SA,
  multi-start Vine MAP EBEs, and Sobol QMC importance-sampled marginal OFV.
- **Interaction with `#1721` / `#2669` / `#2703` (`[eta_shape]` & declarative `density` / `copula` lines)**:
  Whichever branch merges second onto `main` MUST add a validation check in
  `src/api/validation.rs` (and error code registry rows in `src/diagnostics.rs` +
  `docs/file-formats/check-report.qmd`) refusing `omega_dist = vine` when the model also
  declares `[eta_shape]`, `omega ... density ...`, or `copula (...) ~ ...` in
  `[parameters]`, preventing two competing random-effect priors from being active at once.
