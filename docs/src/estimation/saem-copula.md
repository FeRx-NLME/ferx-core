# Vine-Copula SAEM (`omega_dist = vine`)

Setting `omega_dist = vine` in `[fit_options]` replaces the multivariate-normal
Ω with a **D-vine copula** over the marginal η distributions. This allows the
joint distribution of random effects to exhibit non-Gaussian dependence
structure — in particular, lower or upper tail dependence that a Gaussian
copula (inherent to MVN) cannot represent regardless of the correlation
parameter.

## Motivation

Physiological correlation between individual parameters such as clearance (CL)
and volume of distribution (V) is not always symmetric. In some populations,
subjects with very low clearance also tend to have very low volume
(lower-tail dependence), while the upper tail of the joint distribution is
nearly independent. A multivariate-normal Ω fits the same Pearson correlation
to both tails and systematically misrepresents this asymmetry.

## Model

Each marginal η is modelled as Gaussian (mean 0, standard deviation σ\_k). The
joint distribution is constructed via a D-vine copula:

$$
f(\eta_1, \ldots, \eta_d) = \prod_{k=1}^d \phi(\eta_k; 0, \sigma_k) \times \prod_{\text{trees}} c_{\text{pair}}(u_i, u_j)
$$

where $u_k = \Phi(\eta_k / \sigma_k)$ are the probability integral transforms
(PITs) and $c_{\text{pair}}$ is the selected pair-copula density.

### D-Vine Structure

The D-vine uses the natural ordering 1, 2, …, d. Tree 1 contains pairs
(1,2), (2,3), …; tree 2 conditions on the central variables using sequential
h-functions. For two random effects this reduces to a single bivariate
pair-copula.

### Pair-Copula Families

The following families are considered at each tree level:

| Family | Tail dependence | Notes |
|--------|----------------|-------|
| Gaussian | None | Equivalent to MVN at this tree level |
| Student-t | Both (symmetric) | Degrees of freedom fitted |
| Clayton | Lower only | λ\_L = 2^{-1/θ} |
| Gumbel | Upper only | λ\_U = 2^{-1/θ} |
| Frank | None | Symmetric, non-elliptical |

Family selection is done once using AIC/BIC at the first M-step that has
sufficient MH samples and is then **frozen** for the remainder of the run.
Keeping the family fixed avoids discontinuities in the sufficient statistics
during the convergence phase.

## Algorithm

The vine SAEM follows the same two-phase schedule as standard SAEM but
replaces the Gaussian sufficient-statistics update with an
Inference-Functions-for-Margins (IFM) M-step:

1. **E-step**: per-subject Metropolis-Hastings sampling of η in two kernels —
   a block proposal with Cholesky covariance from the previous iteration's
   sample scatter matrix, followed by a componentwise (single-coordinate)
   decorrelating sweep. The componentwise kernel is essential here: the block
   proposal can only move η along the current (possibly near-degenerate)
   dependence direction, so on its own it lets correlation run away toward a
   rank-1 collapse (see *Decorrelation safeguards* below).
2. **M-step (marginals)**: update each σ\_k from the sample standard deviations
   of the accepted η chain.
3. **M-step (copula)**: compute PITs $u_{ik} = \Phi(\hat\eta_{ik}/\sigma_k)$
   for all subjects and tree levels, then fit the pair-copula parameters to
   these pseudo-observations.

The MH acceptance ratio uses the vine log-prior, so subjects with lower tail
dependence in their data accumulate acceptance that pushes the copula toward a
Clayton or Student-t family.

### Decorrelation safeguards

A pure block-proposal SAEM with full-rate (γ=1) sufficient-statistic updates
during exploration is prone to a **rank-1 collapse**: a single warm-started,
not-yet-equilibrated MCMC snapshot is biased toward the chain's current
dependence, and that bias feeds back through the proposal Cholesky (and, for the
vine, through the copula log-prior) into the next E-step — driving every pairwise
correlation toward ±1. Weakly identified random effects are hit first. The
symptom is a `ggpairs` plot of the EBE etas with every pair pinned near ±1.

Three safeguards, all active only during the exploration phase, prevent this:

- a componentwise decorrelating MH kernel (step 1 above), which can always move
  one η independently of the off-diagonal structure;
- a damped Robbins-Monro step (capped learning rate) for the Gaussian-equivalent
  Ω that builds the block proposal's Cholesky;
- a matching damped step for the pair-copula parameters, so a collinear snapshot
  cannot lock a pair-copula at extreme dependence.

In the convergence phase the caps are lifted and the usual decaying γ schedule
takes over. The marginal scales σ\_k are not a dependence driver and update at
full rate throughout.

### `omega_burnin`

During the first `omega_burnin` exploration iterations the marginals and copula
are held fixed at their starting values. This prevents the MH chain from
collapsing before it has warmed up — the same Ω-burnin logic that guards
standard SAEM.

## OFV Interpretation

The reported OFV uses the same Gaussian FOCE formula as the standard method
(see [SAEM](saem.md)). This means the raw ΔOFV between a vine fit and a
Gaussian fit understates the vine's advantage: the Gaussian prior term in the
Laplace approximation is identical for both, regardless of which copula was
fitted.

To see the true model advantage, replace the Gaussian prior with the vine
prior at the vine EBEs:

$$
\text{OFV}_{\text{corrected}} = \text{OFV}_{\text{FOCE}} + 2\sum_i \bigl[\ell^\text{vine}_{\text{prior},i} - \ell^\text{Gauss}_{\text{prior},i}\bigr]
$$

where both terms are evaluated with fully normalised log densities at the same
EBEs. For Clayton-distributed data a corrected ΔOFV of 20–100 units is typical
for 200 subjects at τ = 0.5, compared to a raw FOCE ΔOFV of < 5.

## Usage

```text
[fit_options]
  method       = saem
  omega_dist   = vine
  n_exploration  = 300
  n_convergence  = 200
  omega_burnin   = 50
```

The `omega_dist` key is only accepted by SAEM. Pairing it with a `saem → focei`
chain is rejected at run time.

## Output

After a successful vine fit the YAML output contains a `vine_copula:` section:

```yaml
vine_copula:
  marginals:
    - [ETA_CL, 0.0, 0.394]
    - [ETA_V,  0.0, 0.298]
  trees:
    - tree: 1
      pairs:
        - label: "ETA_CL,ETA_V"
          family: clayton
          params:
            - [theta, 1.92]
          kendall_tau: 0.489
          tail_dep_lower: 0.685
```

The console output mirrors this under a `--- Vine Copula ---` banner.

## Validation

A Clayton-copula simulation study is implemented in `tests/vine_validation.rs`
(Tier 3 slow test). It generates 200 subjects from a Clayton(θ=2) joint
distribution (τ=0.50, λ\_L=0.707), fits both Gaussian SAEM and vine SAEM, and
checks:

- Both methods recover TVCL and TVV within 30%.
- Vine OFV ≤ Gaussian OFV on the training data.
- Vine identifies a tail-dependent family and λ\_L > 0.20.
- Vine Kendall τ is within 0.20 of the true value (0.50).

The test also prints the corrected ΔOFV to quantify the true advantage.

Run with:

```bash
cargo test --features slow-tests --test vine_validation -- --nocapture
```
