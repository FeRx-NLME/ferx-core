/// D-vine copula distribution over per-subject random effects η.
///
/// Implements [`RandomEffectDistribution`] by decomposing the joint η density
/// into Gaussian marginals + a D-vine of bivariate pair-copulas. The vine
/// captures non-Gaussian and asymmetric tail dependence that a purely Gaussian
/// Ω cannot represent.
///
/// # Conventions
///
/// - **Marginals**: per-dimension Gaussian (`μ_i`, `σ_i`) fit by MLE from the
///   pooled SAEM η samples each M-step.
/// - **D-vine ordering**: fixed 0, 1, …, d−1. Variable permutation / R-vine
///   structure selection is deferred to a later phase.
/// - **Pair families**: AIC-selected on the first M-step call and frozen
///   thereafter; only the scalar parameters are updated in subsequent M-steps.
///   Mid-SAEM family switching would introduce discontinuities in `log_prior`
///   and destabilise the Markov chain.
/// - **Pseudo-observations**: Φ-PIT using the fitted marginal, not rank-based.
///
/// # References
/// - Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). Pair-copula
///   constructions of multiple dependence. Insurance: Mathematics and
///   Economics 44:182–198.
use crate::stats::copula::{BivariateCopula, CopulaFamily};
use crate::stats::random_effects::RandomEffectDistribution;
use crate::types::{ModelParameters, OmegaMatrix};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;

/// Minimum standard deviation allowed for a marginal Gaussian.
/// Keeps Φ-PIT well-defined and mirrors `SAEM_OMEGA_DIAG_FLOOR`.
const MARGINAL_STD_FLOOR: f64 = 1e-3;

// ---------------------------------------------------------------------------
// D-vine density evaluation
// ---------------------------------------------------------------------------

/// Evaluate the log-density of a D-vine copula at `u ∈ (0,1)^d`.
///
/// `pair_copulas[k][j]` is the pair-copula at tree level k+1, pair j.
/// Level k has `d − k − 1` entries. For d == 1 returns 0 (no pairs).
///
/// All inputs are clamped to (1e-12, 1−1e-12) before use; NaN/Inf inputs
/// return 1e20 (treated as an extremely unlikely state).
pub(crate) fn dvine_log_density(u: &[f64], pair_copulas: &[Vec<CopulaFamily>]) -> f64 {
    let d = u.len();
    if d <= 1 {
        return 0.0;
    }

    // `left[j]` and `right[j]` are the conditional pseudo-observations fed
    // into each vine level. Initially they are just u.
    //
    // After processing level k, pair j:
    //   new_left[j]  = h(left[j]  | right[j+1]; copula[k][j])  ← h(left|right)
    //   new_right[j] = h(right[j+1] | left[j]; copula[k][j])   ← h(right|left)
    //
    // For all five families (Gaussian, Student-t, Clayton, Gumbel, Frank) the
    // copula is symmetric C(u,v)=C(v,u), so ∂C(u,v)/∂u = h(v,u) — i.e.
    // h(right|left) can be evaluated by calling h(right, left) with the same
    // formula.
    let clamp01 = |x: f64| -> f64 {
        if !x.is_finite() {
            return 0.5;
        }
        x.clamp(1e-12, 1.0 - 1e-12)
    };

    let mut left: Vec<f64> = u.iter().map(|&x| clamp01(x)).collect();
    let mut right: Vec<f64> = left.clone();

    let mut log_dens = 0.0;
    for k in 0..d - 1 {
        let n_pairs = d - k - 1;
        let mut new_left = vec![0.0f64; n_pairs];
        let mut new_right = vec![0.0f64; n_pairs];
        let cop_level = &pair_copulas[k];

        for j in 0..n_pairs {
            let l = left[j];
            let r = right[j + 1];
            let cop = &cop_level[j];
            log_dens += cop.log_density(l, r);
            new_left[j] = clamp01(cop.h(l, r));
            new_right[j] = clamp01(cop.h(r, l));
        }
        left = new_left;
        right = new_right;
    }

    if !log_dens.is_finite() {
        return -1e20;
    }
    log_dens
}

// ---------------------------------------------------------------------------
// VineCopulaOmega
// ---------------------------------------------------------------------------

/// Random-effect distribution using Gaussian marginals + D-vine pair-copulas.
///
/// Constructed once at the start of `run_saem_vine` from the initial model
/// parameters; updated in-place by each `mstep_update` call.
#[derive(Clone, Debug)]
pub struct VineCopulaOmega {
    /// Number of ETAs (= d).
    pub d: usize,
    /// Per-dimension marginal means (updated each M-step).
    pub marginal_means: Vec<f64>,
    /// Per-dimension marginal standard deviations (updated each M-step).
    pub marginal_stds: Vec<f64>,
    /// D-vine pair-copula families. `pair_copulas[k][j]` is at tree k+1, pair j.
    /// After the first M-step the family types are frozen; only parameters change.
    pub pair_copulas: Vec<Vec<CopulaFamily>>,
    /// Whether AIC-based family selection has run at least once.
    families_selected: bool,
    /// SA sufficient statistic (1/N) Σ ηᵢηᵢᵀ — updated with step size γ.
    sample_s2: DMatrix<f64>,
    /// Gaussian-equivalent OmegaMatrix derived from `sample_s2`.
    /// Reported as the OMEGA output and used for the MH proposal scale.
    omega_equiv: OmegaMatrix,
    /// Initial omega — reference for eta_names, diagonal flag, free_mask.
    initial_omega: OmegaMatrix,
    /// Per-eta fixed flags.
    omega_fixed: Vec<bool>,
    /// Initial omega matrix values (for restoring fixed entries after SA).
    initial_matrix: DMatrix<f64>,
    /// Per-pair pseudo-observations from the last `fit_vine` call.
    /// `per_pair_pseudo_obs[k][j]` = (lefts, rights) for tree k+1, pair j.
    /// Used to compute approximate SEs via observed-information Hessian.
    per_pair_pseudo_obs: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
}

impl VineCopulaOmega {
    /// Construct from the initial model parameters.
    pub fn from_init_params(init_params: &ModelParameters) -> Self {
        let omega = init_params.omega.clone();
        let d = omega.dim();
        let marginal_means = vec![0.0f64; d];
        let marginal_stds: Vec<f64> = (0..d)
            .map(|i| {
                let v = omega.matrix[(i, i)];
                v.max(MARGINAL_STD_FLOOR * MARGINAL_STD_FLOOR).sqrt()
            })
            .collect();

        // Default: independent Gaussian copulas (ρ=0) before the first M-step.
        let pair_copulas: Vec<Vec<CopulaFamily>> = (0..d.saturating_sub(1))
            .map(|k| {
                (0..d - k - 1)
                    .map(|_| CopulaFamily::Gaussian(crate::stats::copula::GaussianCopula::new(0.0)))
                    .collect()
            })
            .collect();

        let sample_s2 = omega.matrix.clone();
        let initial_matrix = omega.matrix.clone();
        let initial_omega = omega.clone();
        let omega_equiv = omega.clone();
        let omega_fixed = init_params.omega_fixed.clone();

        Self {
            d,
            marginal_means,
            marginal_stds,
            pair_copulas,
            families_selected: false,
            sample_s2,
            omega_equiv,
            initial_omega,
            omega_fixed,
            initial_matrix,
            per_pair_pseudo_obs: Vec::new(),
        }
    }

    /// Compute pseudo-observations from η samples using the current marginals.
    ///
    /// Returns `u[j][i] = Φ((η[j][i] − μ_i) / σ_i)` clamped to (1e-8, 1−1e-8).
    pub(crate) fn pit_from_samples(&self, sampled_etas: &[Vec<f64>]) -> Vec<Vec<f64>> {
        sampled_etas
            .iter()
            .map(|eta| {
                (0..self.d)
                    .map(|i| {
                        let z = (eta[i] - self.marginal_means[i]) / self.marginal_stds[i];
                        crate::stats::special::normal_cdf(z).clamp(1e-8, 1.0 - 1e-8)
                    })
                    .collect()
            })
            .collect()
    }

    /// Update marginal means and std devs from the current η samples (MLE).
    fn update_marginals(&mut self, sampled_etas: &[Vec<f64>]) {
        let n = sampled_etas.len() as f64;
        for i in 0..self.d {
            let mean = sampled_etas.iter().map(|e| e[i]).sum::<f64>() / n;
            let var = sampled_etas
                .iter()
                .map(|e| (e[i] - mean) * (e[i] - mean))
                .sum::<f64>()
                / n;
            self.marginal_means[i] = mean;
            self.marginal_stds[i] = var.sqrt().max(MARGINAL_STD_FLOOR);
        }
    }

    /// SA update for `sample_s2` and re-build `omega_equiv` with structural
    /// constraints applied (free_mask, fixed entries, diagonal floor).
    fn update_sample_cov(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        let d = self.d;
        let n = sampled_etas.len() as f64;
        let mut eta_outer = DMatrix::zeros(d, d);
        for eta in sampled_etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= n;

        self.sample_s2 = (1.0 - gamma) * &self.sample_s2 + gamma * &eta_outer;

        let mut new_mat = self.sample_s2.clone();
        for i in 0..d {
            for j in 0..d {
                if !self.initial_omega.free_mask[(i, j)] {
                    new_mat[(i, j)] = 0.0;
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                let fi = self.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = self.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    new_mat[(i, j)] = self.initial_matrix[(i, j)];
                }
            }
        }
        for i in 0..d {
            if !self.omega_fixed.get(i).copied().unwrap_or(false) && new_mat[(i, i)] < 1e-6 {
                new_mat[(i, i)] = 1e-6;
            }
        }

        self.omega_equiv = OmegaMatrix::from_matrix(
            new_mat,
            self.initial_omega.eta_names.clone(),
            self.initial_omega.diagonal,
        );
    }

    /// Fit (or re-fit) all pair-copulas from the pseudo-observation matrix.
    ///
    /// When `select_families == true` the best family is chosen by AIC.
    /// When false, the existing family label is kept and only the parameter
    /// is updated. On fitting failure the current copula is left unchanged.
    fn fit_vine(&mut self, pseudo_obs: &[Vec<f64>], select_families: bool) {
        let d = self.d;
        if d <= 1 {
            return;
        }
        let n = pseudo_obs.len();
        if n < 4 {
            return; // too few samples to fit reliably
        }

        // u_cols[i] = all n samples of dimension i (column vector of PIT values).
        let mut u_cols: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                pseudo_obs
                    .iter()
                    .map(|row| row[i].clamp(1e-8, 1.0 - 1e-8))
                    .collect()
            })
            .collect();

        let mut new_pair_pseudo = vec![Vec::new(); d.saturating_sub(1)];

        for k in 0..d - 1 {
            let n_pairs = d - k - 1;
            // Store left/right h-transforms for the next level.
            let mut new_left_cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n_pairs];
            let mut new_right_cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n_pairs];

            for j in 0..n_pairs {
                let lefts: &[f64] = &u_cols[j];
                let rights: &[f64] = &u_cols[j + 1];

                // Store per-pair pseudo-observations for SE computation.
                new_pair_pseudo[k].push((lefts.to_vec(), rights.to_vec()));

                // Fit or re-fit the pair-copula.
                if select_families {
                    if let Ok(cop) = CopulaFamily::select(lefts, rights) {
                        self.pair_copulas[k][j] = cop;
                    }
                } else {
                    self.pair_copulas[k][j] = self.pair_copulas[k][j].refit(lefts, rights);
                }

                // Compute h-transforms for the next vine level.
                let cop = &self.pair_copulas[k][j];
                for s in 0..n {
                    let l = lefts[s];
                    let r = rights[s];
                    new_left_cols[j][s] = cop.h(l, r).clamp(1e-8, 1.0 - 1e-8);
                    new_right_cols[j][s] = cop.h(r, l).clamp(1e-8, 1.0 - 1e-8);
                }
            }

            // For the next level: u_cols has n_pairs+1 entries.
            // u_cols[j] = new_left_cols[j] for j=0..n_pairs-1.
            // u_cols[n_pairs] = new_right_cols[n_pairs-1] (the last right h-transform).
            let mut next_u: Vec<Vec<f64>> = new_left_cols;
            next_u.push(new_right_cols[n_pairs - 1].clone());
            u_cols = next_u;
        }
        self.per_pair_pseudo_obs = new_pair_pseudo;
    }
}

impl RandomEffectDistribution for VineCopulaOmega {
    fn log_prior(&self, eta: &[f64]) -> f64 {
        // Negative log joint density = −log p(η) = −log[∏ f_i(η_i) × c(u_1,…,u_d)]
        //   = Σ_i [−log f_i(η_i)] − log c(u_1,…,u_d)
        // For Gaussian marginals:
        //   −log f_i(η_i) = 0.5 z_i² + 0.5 log(2π) + log σ_i
        // where z_i = (η_i − μ_i) / σ_i and u_i = Φ(z_i).
        let mut log_prior = 0.0;
        let mut u = vec![0.0f64; self.d];
        for i in 0..self.d {
            let z = (eta[i] - self.marginal_means[i]) / self.marginal_stds[i];
            // Negative log marginal density.
            log_prior +=
                0.5 * z * z + 0.5 * (2.0 * std::f64::consts::PI).ln() + self.marginal_stds[i].ln();
            u[i] = crate::stats::special::normal_cdf(z).clamp(1e-12, 1.0 - 1e-12);
        }

        // Subtract vine log-density (vine density > 1 lowers prior, < 1 raises it).
        if self.d > 1 {
            let vine_ld = dvine_log_density(&u, &self.pair_copulas);
            log_prior -= vine_ld;
        }

        if !log_prior.is_finite() {
            1e20
        } else {
            log_prior
        }
    }

    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        if sampled_etas.is_empty() {
            return;
        }
        self.update_marginals(sampled_etas);
        self.update_sample_cov(sampled_etas, gamma);
        let pseudo_obs = self.pit_from_samples(sampled_etas);
        let select = !self.families_selected;
        self.fit_vine(&pseudo_obs, select);
        if select {
            self.families_selected = true;
        }
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        // Phase 4: sample from the Gaussian-equivalent Ω for reporting purposes.
        // Proper vine inverse-Rosenblatt sampling is deferred to Phase 5.
        let d = self.d;
        let l = &self.omega_equiv.chol;
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
                (l * DVector::from_column_slice(&z))
                    .iter()
                    .copied()
                    .collect()
            })
            .collect()
    }

    fn proposal_chol(&self) -> &DMatrix<f64> {
        &self.omega_equiv.chol
    }

    fn to_omega_matrix(&self) -> &OmegaMatrix {
        &self.omega_equiv
    }
}

// ---------------------------------------------------------------------------
// VineFitParams — extracted summary for reporting
// ---------------------------------------------------------------------------

/// Summary of one fitted bivariate pair-copula.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PairCopulaSummary {
    /// Family name: "gaussian", "student_t", "clayton", "gumbel", or "frank".
    pub family: String,
    /// Copula parameter(s). For Gaussian/Student-t: [("rho", ρ)], plus for
    /// Student-t [("nu", ν)]. For Clayton/Gumbel/Frank: [("theta", θ)].
    pub params: Vec<(String, f64)>,
    /// Approximate standard errors for each parameter (same ordering as `params`).
    /// Computed via observed-information Hessian on the transformed scale with
    /// delta-method back to the natural scale. Labeled "approximate" because
    /// pseudo-observations are treated as fixed (IFM approximation).
    #[serde(default)]
    pub se: Vec<(String, f64)>,
    /// Kendall's rank correlation τ derived from the fitted parameters.
    pub kendall_tau: f64,
    /// Lower tail dependence λ_L = lim_{u→0} P(V ≤ u | U ≤ u).
    /// 0 for Gaussian, Frank; 2^{−1/θ} for Clayton; 0 for Gumbel.
    /// For Student-t: 2 · t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))).
    pub tail_dep_lower: f64,
    /// Upper tail dependence λ_U = lim_{u→1} P(V > u | U > u).
    /// 0 for Gaussian, Frank, Clayton; 2 − 2^{1/θ} for Gumbel.
    /// For Student-t: same formula as lower (symmetric).
    pub tail_dep_upper: f64,
}

/// One vine tree level: all pair-copulas at this level together with their
/// variable labels.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VineTreeSummary {
    /// Tree level (1 = unconditional, 2 = conditioned on one variable, …).
    pub tree: usize,
    /// Per-pair summaries. Entry j describes the pair at position j in the
    /// D-vine ordering at this tree level.
    pub pairs: Vec<VinePairEntry>,
}

/// One pair within a vine tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VinePairEntry {
    /// Human-readable pair label, e.g. "ETA_CL ~ ETA_V | ETA_KA".
    pub label: String,
    pub copula: PairCopulaSummary,
}

/// Complete vine-copula fit summary attached to `FitResult` and written to
/// the YAML / console output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VineFitParams {
    /// Fitted marginal Gaussian: (name, mean, sd) per ETA dimension.
    pub marginals: Vec<(String, f64, f64)>,
    /// Vine tree summaries, one entry per tree level (length d−1 for d ETAs).
    pub trees: Vec<VineTreeSummary>,
}

// ---------------------------------------------------------------------------
// Closed-form Kendall's τ and tail-dependence coefficients
// ---------------------------------------------------------------------------

fn kendall_tau_gaussian(rho: f64) -> f64 {
    (2.0 / std::f64::consts::PI) * rho.asin()
}

fn kendall_tau_student_t(rho: f64) -> f64 {
    (2.0 / std::f64::consts::PI) * rho.asin()
}

fn kendall_tau_clayton(theta: f64) -> f64 {
    theta / (theta + 2.0)
}

fn kendall_tau_gumbel(theta: f64) -> f64 {
    1.0 - 1.0 / theta
}

/// Kendall's τ for the Frank copula via the Debye D₁ function.
/// D₁(θ) = (1/θ) ∫₀^θ t/(e^t − 1) dt, computed with 200-point quadrature.
fn kendall_tau_frank(theta: f64) -> f64 {
    if theta.abs() < 1e-8 {
        return 0.0;
    }
    let n = 200usize;
    let dt = theta / n as f64;
    // Trapezoidal rule: integrand = t / (exp(t) - 1)
    let integrand = |t: f64| -> f64 {
        if t.abs() < 1e-10 {
            return 1.0; // lim_{t→0} t/(e^t-1) = 1
        }
        t / (t.exp() - 1.0)
    };
    let mut integral = 0.5 * (integrand(0.0) + integrand(theta));
    for i in 1..n {
        integral += integrand(i as f64 * dt);
    }
    integral *= dt;
    let d1 = integral / theta; // Debye function D₁(θ)
    1.0 - 4.0 * (1.0 - d1) / theta
}

fn tail_dep_student_t(rho: f64, nu: f64) -> f64 {
    if rho <= -1.0 + 1e-8 {
        return 0.0;
    }
    let x = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
    2.0 * crate::stats::copula::student_t_cdf(x, nu + 1.0)
}

// ---------------------------------------------------------------------------
// VineCopulaOmega → VineFitParams extraction
// ---------------------------------------------------------------------------

impl VineCopulaOmega {
    /// Extract a human-readable summary of the fitted vine for reporting.
    ///
    /// `eta_names` should be `init_params.omega.eta_names` (the ETA labels from
    /// the model file). Falls back to "ETA_1", "ETA_2", … if None.
    pub fn to_fit_params(&self, eta_names: &[String]) -> VineFitParams {
        let d = self.d;
        let name = |i: usize| -> String {
            eta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("ETA_{}", i + 1))
        };

        // Marginals.
        let marginals: Vec<(String, f64, f64)> = (0..d)
            .map(|i| (name(i), self.marginal_means[i], self.marginal_stds[i]))
            .collect();

        // Tree summaries.
        let mut trees: Vec<VineTreeSummary> = Vec::new();
        for k in 0..d.saturating_sub(1) {
            let n_pairs = d - k - 1;
            let mut pairs: Vec<VinePairEntry> = Vec::new();

            for j in 0..n_pairs {
                let left_var = j;
                let right_var = j + k + 1;
                let cond_set: Vec<String> = (j + 1..=j + k).map(|c| name(c)).collect();

                let label = if cond_set.is_empty() {
                    format!("{} ~ {}", name(left_var), name(right_var))
                } else {
                    format!(
                        "{} ~ {} | {}",
                        name(left_var),
                        name(right_var),
                        cond_set.join(", ")
                    )
                };

                let cop = &self.pair_copulas[k][j];
                let mut summary = pair_copula_summary(cop);
                // Attach approximate SEs if pseudo-observations are available.
                if let Some(pair_pseudo) = self
                    .per_pair_pseudo_obs
                    .get(k)
                    .and_then(|level| level.get(j))
                {
                    summary.se = pair_copula_se(cop, &pair_pseudo.0, &pair_pseudo.1);
                }
                pairs.push(VinePairEntry {
                    label,
                    copula: summary,
                });
            }

            trees.push(VineTreeSummary { tree: k + 1, pairs });
        }

        VineFitParams { marginals, trees }
    }

    /// Reconstruct a vine distribution from a fitted summary (`VineFitParams`).
    ///
    /// Inverse of [`to_fit_params`](Self::to_fit_params): rebuilds the Gaussian
    /// marginals and the D-vine pair-copulas (families + parameters) so the
    /// result can [`draw_eta`](Self::draw_eta) and evaluate
    /// [`density`](Self::density). The fitting-only state (sufficient
    /// statistics, pseudo-observations) is set to neutral defaults — this is for
    /// *using* an already-fitted vine (simulation, density), not resuming a fit.
    ///
    /// Pair-copula parameters are read by name: Gaussian `rho`; Student-t `rho`
    /// and `nu`; Clayton/Gumbel/Frank `theta`. An unrecognised family or a
    /// missing parameter falls back to an independence (ρ=0 Gaussian) copula.
    pub fn from_fit_params(params: &VineFitParams) -> Self {
        use crate::stats::copula::{
            ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula,
        };

        let d = params.marginals.len();
        let eta_names: Vec<String> = params.marginals.iter().map(|(n, _, _)| n.clone()).collect();
        let marginal_means: Vec<f64> = params.marginals.iter().map(|(_, m, _)| *m).collect();
        let marginal_stds: Vec<f64> = params
            .marginals
            .iter()
            .map(|(_, _, s)| {
                if *s > MARGINAL_STD_FLOOR {
                    *s
                } else {
                    MARGINAL_STD_FLOOR
                }
            })
            .collect();

        let get = |ps: &[(String, f64)], key: &str| -> Option<f64> {
            ps.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
        };
        let independence = || CopulaFamily::Gaussian(GaussianCopula::new(0.0));

        // params.trees[k] is tree level k+1; entry j is pair j at that level,
        // in the same order `to_fit_params` emitted them (the D-vine order).
        let mut pair_copulas: Vec<Vec<CopulaFamily>> = Vec::with_capacity(d.saturating_sub(1));
        for tree in &params.trees {
            let level: Vec<CopulaFamily> = tree
                .pairs
                .iter()
                .map(|p| {
                    let ps = &p.copula.params;
                    match p.copula.family.as_str() {
                        "gaussian" => get(ps, "rho")
                            .map(|rho| CopulaFamily::Gaussian(GaussianCopula::new(rho)))
                            .unwrap_or_else(independence),
                        "student_t" => match (get(ps, "rho"), get(ps, "nu")) {
                            (Some(rho), Some(nu)) => {
                                CopulaFamily::StudentT(StudentTCopula::new(rho, nu))
                            }
                            _ => independence(),
                        },
                        "clayton" => get(ps, "theta")
                            .map(|t| CopulaFamily::Clayton(ClaytonCopula::new(t)))
                            .unwrap_or_else(independence),
                        "gumbel" => get(ps, "theta")
                            .map(|t| CopulaFamily::Gumbel(GumbelCopula::new(t)))
                            .unwrap_or_else(independence),
                        "frank" => get(ps, "theta")
                            .map(|t| CopulaFamily::Frank(FrankCopula::new(t)))
                            .unwrap_or_else(independence),
                        _ => independence(),
                    }
                })
                .collect();
            pair_copulas.push(level);
        }
        // Defensive: if the summary was malformed (e.g. no trees for d ≥ 2),
        // fall back to an all-independence vine of the right shape.
        if pair_copulas.len() != d.saturating_sub(1) {
            pair_copulas = (0..d.saturating_sub(1))
                .map(|k| (0..d - k - 1).map(|_| independence()).collect())
                .collect();
        }

        // Neutral fitting-only state. A diagonal Omega from the marginal
        // variances keeps `omega_equiv` / `initial_*` self-consistent; none of
        // these fields are read by `draw_eta` / `log_prior`.
        let mut diag = DMatrix::<f64>::zeros(d, d);
        for i in 0..d {
            diag[(i, i)] = marginal_stds[i] * marginal_stds[i];
        }
        let omega = OmegaMatrix::from_matrix(diag.clone(), eta_names, true);

        Self {
            d,
            marginal_means,
            marginal_stds,
            pair_copulas,
            families_selected: true,
            sample_s2: diag.clone(),
            omega_equiv: omega.clone(),
            initial_omega: omega,
            omega_fixed: vec![false; d],
            initial_matrix: diag,
            per_pair_pseudo_obs: Vec::new(),
        }
    }

    /// Joint log-density `log p(η)` of the fitted vine at `eta` (natural log).
    /// Equal to `−log_prior(eta)`; exposed for simulation / visualisation
    /// consumers that want the density rather than the SAEM prior penalty.
    pub fn log_density(&self, eta: &[f64]) -> f64 {
        -self.log_prior(eta)
    }

    /// Joint density `p(η)` of the fitted vine at `eta`.
    pub fn density(&self, eta: &[f64]) -> f64 {
        self.log_density(eta).exp()
    }

    /// Draw one d-dimensional ETA sample from this D-vine distribution.
    ///
    /// Uses the inverse Rosenblatt transform (Aas et al. 2009): independent
    /// uniforms w[0..d] are sequentially mapped through h-inverse functions from
    /// the outermost tree inward, then inverted through Gaussian marginals.
    pub fn draw_eta<R: rand::Rng>(&self, rng: &mut R) -> Vec<f64> {
        use crate::stats::special::normal_quantile;
        use rand_distr::Open01;

        let d = self.d;
        if d == 0 {
            return vec![];
        }
        let w: Vec<f64> = (0..d).map(|_| rng.sample(Open01)).collect();
        if d == 1 {
            let z = normal_quantile(w[0].clamp(1e-12, 1.0 - 1e-12));
            return vec![self.marginal_means[0] + self.marginal_stds[0] * z];
        }

        // V-table: vt[j][k] = v_{j,k} for 0 ≤ j ≤ k < d (0-indexed).
        // v_{j,j} = u[j] (the copula-uniform for variable j).
        // v_{j,k} = h(v_{j,k-1} | v_{j+1,k}; pair_copulas[k-j-1][j]).
        //
        // During simulation, vt grows column by column as each variable is
        // generated. Only the columns up to the current index are read/written.
        let mut vt = vec![vec![0.0_f64; d]; d];

        // Generate u[0] (first variable, no conditioning).
        let u0 = w[0].clamp(1e-12, 1.0 - 1e-12);
        vt[0][0] = u0;

        for i in 1..d {
            // Apply h_inv from the outermost tree (level i-1) down to tree 0.
            // j=0 → tree_level = i-1 (outermost), j=i-1 → tree_level = 0 (innermost).
            let mut tmp = w[i].clamp(1e-12, 1.0 - 1e-12);
            for j in 0..i {
                let tree_level = i - 1 - j;
                let pair_idx = j;
                let cond = vt[j][i - 1].clamp(1e-12, 1.0 - 1e-12);
                tmp = self.pair_copulas[tree_level][pair_idx]
                    .h_inv(tmp.clamp(1e-12, 1.0 - 1e-12), cond)
                    .clamp(1e-12, 1.0 - 1e-12);
            }
            vt[i][i] = tmp;

            // Update V-table for future iterations.
            // j goes from i-1 down to 0 so that vt[j+1][i] is ready before vt[j][i].
            for j in (0..i).rev() {
                let tree_level = i - j - 1;
                let pair_idx = j;
                let l = vt[j][i - 1].clamp(1e-12, 1.0 - 1e-12);
                let r = vt[j + 1][i].clamp(1e-12, 1.0 - 1e-12);
                vt[j][i] = self.pair_copulas[tree_level][pair_idx]
                    .h(l, r)
                    .clamp(1e-12, 1.0 - 1e-12);
            }
        }

        // Invert Gaussian marginals: u[k] → η[k] = μ[k] + σ[k] * Φ⁻¹(u[k]).
        (0..d)
            .map(|k| {
                let z = normal_quantile(vt[k][k]);
                self.marginal_means[k] + self.marginal_stds[k] * z
            })
            .collect()
    }
}

fn pair_copula_summary(cop: &CopulaFamily) -> PairCopulaSummary {
    match cop {
        CopulaFamily::Gaussian(c) => {
            let rho = c.rho;
            PairCopulaSummary {
                family: "gaussian".into(),
                params: vec![("rho".into(), rho)],
                se: vec![],
                kendall_tau: kendall_tau_gaussian(rho),
                tail_dep_lower: 0.0,
                tail_dep_upper: 0.0,
            }
        }
        CopulaFamily::StudentT(c) => {
            let (rho, nu) = (c.rho, c.nu);
            let td = tail_dep_student_t(rho, nu);
            PairCopulaSummary {
                family: "student_t".into(),
                params: vec![("rho".into(), rho), ("nu".into(), nu)],
                se: vec![],
                kendall_tau: kendall_tau_student_t(rho),
                tail_dep_lower: td,
                tail_dep_upper: td,
            }
        }
        CopulaFamily::Clayton(c) => {
            let theta = c.theta;
            PairCopulaSummary {
                family: "clayton".into(),
                params: vec![("theta".into(), theta)],
                se: vec![],
                kendall_tau: kendall_tau_clayton(theta),
                tail_dep_lower: (2.0f64).powf(-1.0 / theta),
                tail_dep_upper: 0.0,
            }
        }
        CopulaFamily::Gumbel(c) => {
            let theta = c.theta;
            PairCopulaSummary {
                family: "gumbel".into(),
                params: vec![("theta".into(), theta)],
                se: vec![],
                kendall_tau: kendall_tau_gumbel(theta),
                tail_dep_lower: 0.0,
                tail_dep_upper: 2.0 - (2.0f64).powf(1.0 / theta),
            }
        }
        CopulaFamily::Frank(c) => {
            let theta = c.theta;
            PairCopulaSummary {
                family: "frank".into(),
                params: vec![("theta".into(), theta)],
                se: vec![],
                kendall_tau: kendall_tau_frank(theta),
                tail_dep_lower: 0.0,
                tail_dep_upper: 0.0,
            }
        }
    }
}

/// Approximate SE for each parameter of a fitted pair-copula.
///
/// Uses the observed-information Hessian on a transformed scale:
/// - Gaussian/Student-t ρ: Fisher-z transform `ψ = arctanh(ρ)`
/// - Student-t ν: log transform `ψ = ln(ν)`
/// - Clayton/Gumbel θ > 1: log-shift `ψ = ln(θ − 1)` for Gumbel, log `ψ = ln(θ)` for Clayton
/// - Frank θ: identity
///
/// FD step size: `h = 1e-5`. SE = 1/√max(|H|, 1e-12). Delta-method maps SE_ψ → SE_θ.
/// Returns `Vec<(name, se)>` with the same ordering as `PairCopulaSummary::params`.
fn pair_copula_se(cop: &CopulaFamily, u: &[f64], v: &[f64]) -> Vec<(String, f64)> {
    use crate::stats::copula::BivariateCopula;

    if u.len() < 4 {
        return match cop {
            CopulaFamily::StudentT(_) => {
                vec![("rho".into(), f64::NAN), ("nu".into(), f64::NAN)]
            }
            _ => vec![("theta_or_rho".into(), f64::NAN)],
        };
    }

    let log_lik = |cop_eval: &CopulaFamily| -> f64 {
        u.iter()
            .zip(v.iter())
            .map(|(&ui, &vi)| cop_eval.log_density(ui, vi))
            .sum::<f64>()
    };

    let fd_hessian_diag = |ll_center: f64, ll_plus: f64, ll_minus: f64, h: f64| -> f64 {
        -(ll_plus - 2.0 * ll_center + ll_minus) / (h * h)
    };

    let h = 1e-5_f64;

    match cop {
        CopulaFamily::Gaussian(c) => {
            let rho = c.rho;
            let psi = rho.atanh(); // Fisher-z
            let make = |dpsi: f64| -> CopulaFamily {
                CopulaFamily::Gaussian(crate::stats::copula::GaussianCopula {
                    rho: (psi + dpsi).tanh().clamp(-0.9999, 0.9999),
                })
            };
            let ll0 = log_lik(cop);
            let ll_p = log_lik(&make(h));
            let ll_m = log_lik(&make(-h));
            let info = fd_hessian_diag(ll0, ll_p, ll_m, h).max(0.0);
            let se_psi = if info > 1e-12 {
                1.0 / info.sqrt()
            } else {
                f64::NAN
            };
            let d_rho_d_psi = 1.0 - rho * rho; // sech²(ψ)
            vec![("rho".into(), (d_rho_d_psi * se_psi).abs())]
        }
        CopulaFamily::StudentT(c) => {
            let (rho, nu) = (c.rho, c.nu);
            let psi_rho = rho.atanh();
            let psi_nu = nu.ln();
            // ρ diagonal
            let make_rho = |dp: f64| -> CopulaFamily {
                CopulaFamily::StudentT(crate::stats::copula::StudentTCopula {
                    rho: (psi_rho + dp).tanh().clamp(-0.9999, 0.9999),
                    nu,
                })
            };
            let ll0 = log_lik(cop);
            let ll_p = log_lik(&make_rho(h));
            let ll_m = log_lik(&make_rho(-h));
            let info_rho = fd_hessian_diag(ll0, ll_p, ll_m, h).max(0.0);
            let se_psi_rho = if info_rho > 1e-12 {
                1.0 / info_rho.sqrt()
            } else {
                f64::NAN
            };
            let d_rho = (1.0 - rho * rho) * se_psi_rho;
            // ν diagonal
            let make_nu = |dp: f64| -> CopulaFamily {
                CopulaFamily::StudentT(crate::stats::copula::StudentTCopula {
                    rho,
                    nu: (psi_nu + dp).exp().max(2.001),
                })
            };
            let ll_p2 = log_lik(&make_nu(h));
            let ll_m2 = log_lik(&make_nu(-h));
            let info_nu = fd_hessian_diag(ll0, ll_p2, ll_m2, h).max(0.0);
            let se_psi_nu = if info_nu > 1e-12 {
                1.0 / info_nu.sqrt()
            } else {
                f64::NAN
            };
            let d_nu = nu * se_psi_nu;
            vec![("rho".into(), d_rho.abs()), ("nu".into(), d_nu.abs())]
        }
        CopulaFamily::Clayton(c) => {
            let theta = c.theta;
            let psi = theta.max(1e-8).ln();
            let make = |dp: f64| -> CopulaFamily {
                CopulaFamily::Clayton(crate::stats::copula::ClaytonCopula {
                    theta: (psi + dp).exp().max(1e-6),
                })
            };
            let ll0 = log_lik(cop);
            let ll_p = log_lik(&make(h));
            let ll_m = log_lik(&make(-h));
            let info = fd_hessian_diag(ll0, ll_p, ll_m, h).max(0.0);
            let se_psi = if info > 1e-12 {
                1.0 / info.sqrt()
            } else {
                f64::NAN
            };
            vec![("theta".into(), (theta * se_psi).abs())]
        }
        CopulaFamily::Gumbel(c) => {
            let theta = c.theta;
            let psi = (theta - 1.0).max(1e-8).ln();
            let make = |dp: f64| -> CopulaFamily {
                CopulaFamily::Gumbel(crate::stats::copula::GumbelCopula {
                    theta: 1.0 + (psi + dp).exp(),
                })
            };
            let ll0 = log_lik(cop);
            let ll_p = log_lik(&make(h));
            let ll_m = log_lik(&make(-h));
            let info = fd_hessian_diag(ll0, ll_p, ll_m, h).max(0.0);
            let se_psi = if info > 1e-12 {
                1.0 / info.sqrt()
            } else {
                f64::NAN
            };
            let d_theta = (theta - 1.0) * se_psi; // ∂θ/∂ψ = exp(ψ) = θ − 1
            vec![("theta".into(), d_theta.abs())]
        }
        CopulaFamily::Frank(c) => {
            let theta = c.theta;
            let make = |dp: f64| -> CopulaFamily {
                CopulaFamily::Frank(crate::stats::copula::FrankCopula { theta: theta + dp })
            };
            let ll0 = log_lik(cop);
            let ll_p = log_lik(&make(h));
            let ll_m = log_lik(&make(-h));
            let info = fd_hessian_diag(ll0, ll_p, ll_m, h).max(0.0);
            let se = if info > 1e-12 {
                1.0 / info.sqrt()
            } else {
                f64::NAN
            };
            vec![("theta".into(), se.abs())]
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::copula::GaussianCopula;
    use crate::types::test_helpers::analytical_model;
    use crate::types::GradientMethod;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    fn make_dist() -> VineCopulaOmega {
        let model = analytical_model(GradientMethod::Auto);
        VineCopulaOmega::from_init_params(&model.default_params)
    }

    /// D-vine log density for d=1 is always 0 (no pairs).
    #[test]
    fn dvine_log_density_d1_is_zero() {
        let cop: Vec<Vec<CopulaFamily>> = vec![];
        assert_eq!(dvine_log_density(&[0.5], &cop), 0.0);
    }

    /// D-vine with all Gaussian(ρ=0) copulas (independence) has log density 0.
    #[test]
    fn dvine_independence_density_is_zero() {
        let d = 3;
        let pair_copulas: Vec<Vec<CopulaFamily>> = (0..d - 1)
            .map(|k| {
                (0..d - k - 1)
                    .map(|_| CopulaFamily::Gaussian(GaussianCopula::new(0.0)))
                    .collect()
            })
            .collect();
        let u = vec![0.2, 0.5, 0.8];
        let ld = dvine_log_density(&u, &pair_copulas);
        assert!(
            ld.abs() < 1e-10,
            "independence vine log density should be 0, got {ld}"
        );
    }

    /// from_init_params constructs with correct dimensions and positive stds.
    #[test]
    fn vine_omega_construction() {
        let dist = make_dist();
        assert_eq!(dist.marginal_means.len(), dist.d);
        assert_eq!(dist.marginal_stds.len(), dist.d);
        assert_eq!(dist.pair_copulas.len(), dist.d.saturating_sub(1));
        for &s in &dist.marginal_stds {
            assert!(s > 0.0);
        }
    }

    /// from_fit_params rebuilds a usable vine from a summary: families and
    /// parameters are restored, and density / draw_eta work on the result.
    #[test]
    fn vine_from_fit_params_rebuilds_clayton() {
        let summary = VineFitParams {
            marginals: vec![
                ("ETA_CL".to_string(), 0.0, 0.39),
                ("ETA_V".to_string(), 0.0, 0.30),
            ],
            trees: vec![VineTreeSummary {
                tree: 1,
                pairs: vec![VinePairEntry {
                    label: "ETA_CL ~ ETA_V".to_string(),
                    copula: PairCopulaSummary {
                        family: "clayton".to_string(),
                        params: vec![("theta".to_string(), 2.0)],
                        se: vec![],
                        kendall_tau: 0.5,
                        tail_dep_lower: 0.707,
                        tail_dep_upper: 0.0,
                    },
                }],
            }],
        };

        let dist = VineCopulaOmega::from_fit_params(&summary);
        assert_eq!(dist.d, 2);
        assert_eq!(dist.pair_copulas.len(), 1);
        assert_eq!(dist.pair_copulas[0].len(), 1);
        match &dist.pair_copulas[0][0] {
            CopulaFamily::Clayton(c) => assert!((c.theta - 2.0).abs() < 1e-12),
            other => panic!("expected Clayton, got {other:?}"),
        }
        assert!((dist.marginal_stds[0] - 0.39).abs() < 1e-12);

        // Density is finite & positive; the dependence raises it on the diagonal.
        let p = dist.density(&[0.1, 0.1]);
        assert!(
            p.is_finite() && p > 0.0,
            "density should be positive, got {p}"
        );

        // draw_eta produces a finite 2-vector.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let eta = dist.draw_eta(&mut rng);
        assert_eq!(eta.len(), 2);
        assert!(eta.iter().all(|x| x.is_finite()));
    }

    /// to_fit_params → from_fit_params preserves the marginals and vine shape.
    #[test]
    fn vine_fit_params_round_trip_shape() {
        let dist = make_dist();
        let summary = dist.to_fit_params(&[]);
        let rebuilt = VineCopulaOmega::from_fit_params(&summary);
        assert_eq!(rebuilt.d, dist.d);
        assert_eq!(rebuilt.pair_copulas.len(), dist.pair_copulas.len());
        for i in 0..dist.d {
            assert!((rebuilt.marginal_stds[i] - dist.marginal_stds[i]).abs() < 1e-9);
        }
    }

    /// log_prior is finite and positive for a typical η.
    #[test]
    fn vine_log_prior_finite_and_positive() {
        let dist = make_dist();
        let eta = vec![0.3_f64; dist.d];
        let lp = dist.log_prior(&eta);
        assert!(lp.is_finite(), "log_prior should be finite, got {lp}");
        assert!(
            lp > 0.0,
            "log_prior (neg log density) should be positive near zero"
        );
    }

    /// At η=0 with ρ=0 vine, log_prior equals the Gaussian marginal sum.
    #[test]
    fn vine_log_prior_zero_eta_equals_gaussian_marginal_sum() {
        let dist = make_dist();
        let d = dist.d;
        let eta = vec![0.0_f64; d];

        // Gaussian marginal neg-log-density at z=0: 0.5*log(2π) + log(σ).
        // Vine (ρ=0) adds 0.
        let expected: f64 = (0..d)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI).ln() + dist.marginal_stds[i].ln())
            .sum();
        let actual = dist.log_prior(&eta);
        assert!(
            (actual - expected).abs() < 1e-9,
            "log_prior(0) = {actual:.6e}, expected {expected:.6e}"
        );
    }

    /// mstep_update runs without panicking, updates marginals, and selects families.
    #[test]
    fn vine_mstep_updates_marginals_and_selects_families() {
        let mut dist = make_dist();
        let d = dist.d;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let etas: Vec<Vec<f64>> = (0..30)
            .map(|_| {
                (0..d)
                    .map(|_| 0.5 + 0.2 * rng.sample::<f64, _>(StandardNormal))
                    .collect()
            })
            .collect();

        let mean_before = dist.marginal_means[0];
        dist.mstep_update(&etas, 1.0);

        // Mean should move toward 0.5.
        let mean_after = dist.marginal_means[0];
        assert!(
            (mean_after - 0.5).abs() < (mean_before - 0.5).abs() + 0.2,
            "mean should move toward 0.5: before={mean_before:.3}, after={mean_after:.3}"
        );
        assert!(
            dist.families_selected,
            "families should be selected after first mstep"
        );
    }

    /// proposal_chol is square with positive diagonal.
    #[test]
    fn vine_proposal_chol_valid() {
        let dist = make_dist();
        let l = dist.proposal_chol();
        let d = dist.d;
        assert_eq!(l.nrows(), d);
        assert_eq!(l.ncols(), d);
        for i in 0..d {
            assert!(
                l[(i, i)] > 0.0,
                "Cholesky diagonal [{i},{i}] should be positive"
            );
        }
    }

    /// log_prior is consistent across two calls with the same η.
    #[test]
    fn vine_log_prior_deterministic() {
        let dist = make_dist();
        let eta = vec![0.1_f64; dist.d];
        let lp1 = dist.log_prior(&eta);
        let lp2 = dist.log_prior(&eta);
        assert_eq!(lp1, lp2);
    }

    /// `draw_eta` marginals should match the configured means and variances.
    ///
    /// With 4 000 draws and initial marginals σ² ≈ 0.09 (from the standard
    /// warfarin model), sample mean should be within 0.03 and sample variance
    /// within 20 % of the true value with overwhelming probability.
    #[test]
    /// `pair_copula_se` returns a finite positive SE for a Gaussian copula with
    /// 50 paired pseudo-observations drawn from the copula's own distribution.
    #[test]
    fn pair_copula_se_gaussian_is_finite_positive() {
        use crate::stats::copula::{BivariateCopula, GaussianCopula};
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let rho_true = 0.6;
        let cop = CopulaFamily::Gaussian(GaussianCopula::new(rho_true));
        // Draw n=50 (u,v) pairs from a Gaussian copula using the Gaussian copula
        // CDF: u ~ Uniform, v = h_inv(w | u) for w ~ Uniform.
        let n = 50usize;
        let mut u_data = vec![0.0f64; n];
        let mut v_data = vec![0.0f64; n];
        for i in 0..n {
            let u: f64 = rng.gen_range(0.01..0.99);
            let w: f64 = rng.gen_range(0.01..0.99);
            let v = cop.h_inv(w, u).clamp(0.01, 0.99);
            u_data[i] = u;
            v_data[i] = v;
        }
        let ses = pair_copula_se(&cop, &u_data, &v_data);
        assert_eq!(ses.len(), 1);
        assert_eq!(ses[0].0, "rho");
        let se = ses[0].1;
        assert!(
            se.is_finite() && se > 0.0,
            "SE should be finite positive, got {se}"
        );
        // SE for rho from n=50 should be in a reasonable range (rough bound: SE < 0.5)
        assert!(se < 0.5, "SE seems implausibly large: {se}");
    }

    fn draw_eta_marginals_match_configured() {
        use rand::SeedableRng;
        let dist = make_dist();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let n = 4_000;
        let mut sums = vec![0.0_f64; dist.d];
        let mut sumsq = vec![0.0_f64; dist.d];
        for _ in 0..n {
            let eta = dist.draw_eta(&mut rng);
            for (k, &v) in eta.iter().enumerate() {
                sums[k] += v;
                sumsq[k] += v * v;
            }
        }
        for k in 0..dist.d {
            let mean = sums[k] / n as f64;
            let var = sumsq[k] / n as f64 - mean * mean;
            let true_var = dist.marginal_stds[k] * dist.marginal_stds[k];
            assert!(
                mean.abs() < 0.05,
                "dimension {k}: sample mean {mean:.4} too far from 0"
            );
            assert!(
                (var - true_var).abs() / true_var < 0.25,
                "dimension {k}: sample var {var:.4} more than 25% from true {true_var:.4}"
            );
        }
    }
}
