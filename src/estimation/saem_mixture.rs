//! SAEM under a mixture model (`[mixture]`, #985).
//!
//! FOCE/FOCEI marginalise the latent class analytically (`estimation::mixture`,
//! the K-fold log-sum-exp `L_i = Σ_k p_ik L_ik`). SAEM instead **samples** the
//! class indicator: each E-step draws `z_i ~ Categorical(PMIX_i)` from the
//! current per-subject posterior, then runs the usual η-MCMC *within the drawn
//! class* (issue #985). The M-step partitions the sufficient statistics by
//! sampled class:
//!
//! * **θ / σ** — the class-switched typical values (`if MIXNUM == k …`) fall out
//!   of the ordinary θ/σ M-step for free, provided every per-subject likelihood
//!   evaluation runs under a [`MixtureClassGuard`] for that subject's drawn
//!   class: `∂pred/∂TVCL2 = 0` for a class-1 subject, so pooling all subjects in
//!   one NLopt call estimates each class's clearance from its own members. (The
//!   closed-form mu-reference M-step is disabled for mixtures — the θ↔η pairing
//!   is class-dependent — so `run_saem` routes to the full NLopt M-step.)
//! * **Ω overrides** (`omega(k)`) — per-class diagonal SA sufficient statistics,
//!   `s2_diag[k][j] = SA-avg over subjects with z_i=k of η_ij²`.
//! * **mixing coefficients** — a small NLopt maximising the SA-averaged expected
//!   complete-data mixing log-likelihood `Σ_i Σ_k r̄_ik ln p_ik(θ_mix)` over the
//!   thetas the mixing expressions depend on. `r̄_ik` is the Robbins-Monro
//!   average of the sampled class indicators, so this is exact for constant
//!   mixing (reduces to `p_k = mean_i r̄_ik`) and for covariate-dependent logit
//!   mixing alike.
//!
//! Per-class σ overrides (`sigma(k)`) are the one piece not yet re-estimated
//! under SAEM — they are held at their initial values and a warning is emitted;
//! everything else (class-switched θ, Ω overrides, constant/covariate mixing) is
//! estimated.

use crate::estimation::mixture::combine_subject;
use crate::parser::model_parser::eval_mixing_log_probs;
use crate::pk::EventPkParams;
use crate::stats::likelihood::{individual_nll_into, individual_nll_iov};
use crate::types::{
    CompiledModel, MixtureParams, ModelParameters, OmegaMatrix, Population, Subject,
};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::RngExt;
use std::collections::HashMap;

/// Floor for a per-class Ω override diagonal, mirroring the BSV `Ω` diagonal
/// floor — keeps each class Ω positive-definite when a class samples a near-zero
/// spread early.
const OVERRIDE_OMEGA_DIAG_FLOOR: f64 = 1e-8;

/// Mutable SAEM mixture state carried across the outer loop, alongside
/// `SaemState`. Present only when `model.mixture.is_some()`.
pub(crate) struct SaemMixture {
    pub n_classes: usize,
    /// Current per-class Omega / Sigma (index 0 == base class 1). Base entries
    /// are refreshed from `SaemState` each iteration via [`Self::sync_base`];
    /// override entries are updated by the mixture M-step.
    pub mp: MixtureParams,
    /// Per-subject drawn class this iteration (0-based).
    pub classes: Vec<usize>,
    /// Robbins-Monro average of the sampled class indicators, `r̄_ik`
    /// (`[subject][class]`) — the sufficient statistic the mixing M-step
    /// consumes.
    pub rbar: Vec<Vec<f64>>,
    /// Per-class per-eta SA sufficient statistic for the Ω-override diagonals,
    /// `s2_diag[class][eta]`.
    pub s2_diag: Vec<Vec<f64>>,
    /// Held (initial) values of the σ overrides, keyed `(class, sigma_index)`.
    pub sigma_override_init: HashMap<(usize, usize), f64>,
    /// Initial values of the Ω-override diagonals, keyed `(class, eta_index)`.
    /// Applied by [`Self::sync_base`] while the Ω burn-in is still running, so
    /// the overrides are *held at their inits* exactly like the base Ω — even
    /// though their SA statistic is warming up in the background (#987 review).
    pub omega_override_init: HashMap<(usize, usize), f64>,
    /// `false` while the Ω burn-in is running: [`Self::sync_base`] then applies
    /// `omega_override_init` rather than the (still-warming) SA statistic.
    pub omega_stat_active: bool,
    /// Theta indices the mixing expressions depend on (detected numerically at
    /// build time), excluding any theta marked `FIX`. Only these are moved by
    /// the mixing M-step.
    pub mixing_theta_idx: Vec<usize>,
}

impl SaemMixture {
    /// Build from the initial `ModelParameters` (must carry `mixture`). Uses the
    /// population's covariates both to detect which thetas the mixing expressions
    /// depend on and to seed each subject's responsibility average.
    pub(crate) fn build(
        model: &CompiledModel,
        init_params: &ModelParameters,
        population: &Population,
    ) -> Self {
        let mp = init_params
            .mixture
            .as_ref()
            .expect("SaemMixture::build on non-mixture params")
            .clone();
        let n_classes = mp.omega.len();
        let n_eta = model.n_eta;
        let n_subjects = population.subjects.len();
        let spec = model
            .mixture
            .as_ref()
            .expect("SaemMixture::build on non-mixture model");

        // Per-subject initial mixing probabilities (each under that subject's own
        // covariates — so covariate-dependent logit mixing starts from the right,
        // subject-specific responsibilities rather than a covariate-0 default).
        let base_lp: Vec<Vec<f64>> = population
            .subjects
            .iter()
            .map(|s| eval_mixing_log_probs(spec, &init_params.theta, &s.covariates))
            .collect();

        // Detect which thetas the mixing expressions depend on: perturb each and
        // see whether any `ln p_ik` moves for *any* subject. Probing against the
        // real covariates (not an empty map) is what catches a covariate that
        // enters purely multiplicatively and is 0 at the default — e.g. `BWT*SEX`
        // with `SEX = 0` would show no θ-dependence against a covariate-0 probe.
        let mut mixing_theta_idx = Vec::new();
        for j in 0..init_params.theta.len() {
            let mut t = init_params.theta.clone();
            t[j] += 1e-3 * t[j].abs().max(1.0);
            let moves = population.subjects.iter().zip(&base_lp).any(|(s, blp)| {
                let lp = eval_mixing_log_probs(spec, &t, &s.covariates);
                lp.iter().zip(blp).any(|(a, b)| (a - b).abs() > 1e-9)
            });
            // A `FIX`-ed theta must never be moved by the mixing M-step. FIX is
            // enforced elsewhere by collapsing the *packed* theta bounds, which
            // `mstep_mixing` (working on natural-scale user bounds) cannot see —
            // so drop fixed thetas here instead (#987 review).
            let fixed = init_params.theta_fixed.get(j).copied().unwrap_or(false);
            if moves && !fixed {
                mixing_theta_idx.push(j);
            }
        }

        // Per-class Ω-override SA statistic seeded from the init class Ω diagonal.
        let s2_diag: Vec<Vec<f64>> = (0..n_classes)
            .map(|c| (0..n_eta).map(|j| mp.omega[c].matrix[(j, j)]).collect())
            .collect();

        // Ω-override init snapshot — what `sync_base` applies during burn-in.
        let omega_override_init: HashMap<(usize, usize), f64> = mp
            .omega_override_addr
            .iter()
            .map(|&(c, j)| ((c, j), mp.omega[c].matrix[(j, j)]))
            .collect();

        // σ-override init snapshot (held constant under SAEM).
        let sigma_override_init: HashMap<(usize, usize), f64> = mp
            .sigma_override_addr
            .iter()
            .map(|&(c, s)| ((c, s), mp.sigma[c].values[s]))
            .collect();

        // r̄ initialised per subject from that subject's initial mixing probs.
        let rbar: Vec<Vec<f64>> = base_lp
            .iter()
            .map(|blp| blp.iter().map(|l| l.exp()).collect())
            .collect();

        SaemMixture {
            n_classes,
            mp,
            classes: vec![0usize; n_subjects],
            rbar,
            s2_diag,
            sigma_override_init,
            omega_override_init,
            omega_stat_active: false,
            mixing_theta_idx,
        }
    }

    /// True if any per-class σ override exists (held, not re-estimated).
    pub(crate) fn has_sigma_override(&self) -> bool {
        !self.mp.sigma_override_addr.is_empty()
    }

    /// Refresh every class's Ω / σ from the current base (SAEM state) plus the
    /// per-class overrides. Called at the top of each iteration so per-class
    /// views built from `mp` share the just-updated base.
    pub(crate) fn sync_base(&mut self, base_omega: &OmegaMatrix, base_sigma: &[f64]) {
        for c in 0..self.n_classes {
            self.mp.omega[c] = base_omega.clone();
            self.mp.sigma[c].values = base_sigma.to_vec();
        }
        // Restore Ω-override diagonals: the per-class SA statistic once the Ω
        // burn-in is over, the declared init value while it is still running
        // (mirroring the base Ω, which is likewise held at its init during
        // burn-in while `s2` warms up).
        let addrs = self.mp.omega_override_addr.clone();
        for &(c, j) in &addrs {
            let v = if self.omega_stat_active {
                self.s2_diag[c][j]
            } else {
                self.omega_override_init
                    .get(&(c, j))
                    .copied()
                    .unwrap_or(self.s2_diag[c][j])
            };
            self.mp.omega[c].matrix[(j, j)] = v.max(OVERRIDE_OMEGA_DIAG_FLOOR);
        }
        // Restore σ overrides to their held init values.
        for (&(c, s), &v) in &self.sigma_override_init {
            self.mp.sigma[c].values[s] = v;
        }
    }

    pub(crate) fn class_omega(&self, c: usize) -> &OmegaMatrix {
        &self.mp.omega[c]
    }
    pub(crate) fn class_sigma(&self, c: usize) -> &[f64] {
        &self.mp.sigma[c].values
    }

    /// M-step for the Ω-override diagonals: SA-update each override entry from
    /// the class-`c` subjects' `mean η_j²`. `gamma` is the SA step size.
    pub(crate) fn mstep_omega_overrides(&mut self, etas: &[Vec<f64>], gamma: f64) {
        let addrs = self.mp.omega_override_addr.clone();
        let fixed = self.mp.omega_override_fixed.clone();
        for (idx, &(c, j)) in addrs.iter().enumerate() {
            if fixed[idx] {
                continue;
            }
            let mut sum = 0.0;
            let mut n = 0usize;
            for (i, cls) in self.classes.iter().enumerate() {
                if *cls == c {
                    sum += etas[i][j] * etas[i][j];
                    n += 1;
                }
            }
            if n > 0 {
                let m = sum / n as f64;
                self.s2_diag[c][j] = (1.0 - gamma) * self.s2_diag[c][j] + gamma * m;
            }
        }
    }

    /// True when class `c` carries an Ω override on eta coordinate `j`.
    fn overrides_eta(&self, c: usize, j: usize) -> bool {
        self.mp.omega_override_addr.contains(&(c, j))
    }

    /// Per-class held σ overrides as `[class][(sigma_index, held_value)]` — what
    /// the θ/σ M-step substitutes into the base σ vector for a subject drawn into
    /// that class. Empty inner vectors for classes with no override.
    pub(crate) fn class_sigma_overrides(&self) -> Vec<Vec<(usize, f64)>> {
        let mut out = vec![Vec::new(); self.n_classes];
        for (&(c, s), &v) in &self.sigma_override_init {
            out[c].push((s, v));
        }
        for v in out.iter_mut() {
            v.sort_unstable_by_key(|&(s, _)| s);
        }
        out
    }

    /// Class-partitioned second-moment statistic for the **base** Ω.
    ///
    /// `Σ = mean_i η_i η_iᵀ` pooled over *all* subjects is the right sufficient
    /// statistic only when every class shares the base Ω. With an `omega(k)`
    /// override on coordinate `j`, class `k`'s members are drawn from a
    /// *different* variance, so pooling them into the base entry biases it toward
    /// the mixture-wide spread. Entry `(j, l)` is therefore averaged over the
    /// subjects whose drawn class overrides neither `j` nor `l` (#987 review).
    ///
    /// Returns `(mean, counts)`; the caller SA-updates only entries with a
    /// non-zero count (a coordinate overridden by *every* class has no base
    /// members and keeps its previous value).
    pub(crate) fn base_eta_outer(
        &self,
        etas: &[Vec<f64>],
        n_eta: usize,
    ) -> (DMatrix<f64>, DMatrix<usize>) {
        let mut sum = DMatrix::zeros(n_eta, n_eta);
        let mut counts = DMatrix::zeros(n_eta, n_eta);
        for (i, &c) in self.classes.iter().enumerate() {
            let eta = &etas[i];
            for j in 0..n_eta {
                if self.overrides_eta(c, j) {
                    continue;
                }
                for l in 0..n_eta {
                    if self.overrides_eta(c, l) {
                        continue;
                    }
                    sum[(j, l)] += eta[j] * eta[l];
                    counts[(j, l)] += 1;
                }
            }
        }
        for j in 0..n_eta {
            for l in 0..n_eta {
                if counts[(j, l)] > 0 {
                    sum[(j, l)] /= counts[(j, l)] as f64;
                }
            }
        }
        (sum, counts)
    }

    /// SA-update the Robbins-Monro responsibility average from the classes drawn
    /// this iteration. `gamma` is the SA step size.
    pub(crate) fn update_rbar(&mut self, gamma: f64) {
        for (i, cls) in self.classes.iter().enumerate() {
            for k in 0..self.n_classes {
                let ind = if k == *cls { 1.0 } else { 0.0 };
                self.rbar[i][k] = (1.0 - gamma) * self.rbar[i][k] + gamma * ind;
            }
        }
    }
}

/// Draw a class (0-based) for one subject from its current posterior
/// `PMIX_i ∝ p_ik · exp(−nll_ik)`, where `nll_ik` is the complete-data
/// individual objective at the subject's *current* η (and κ, for IOV) evaluated
/// under class `k`'s parameters. The final per-subject posterior is recomputed at
/// the converged parameters after the loop (via `mixture_ofv`), so only the
/// sampled class is returned here.
///
/// Must run on the same thread that owns the `MixtureClassGuard` — the caller
/// (the rayon E-step closure) enters this per subject, so the guard set inside
/// reaches the structural `MIXNUM` branch correctly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_class(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    mix: &SaemMixture,
    omega_iov: Option<&OmegaMatrix>,
    scratch: &mut EventPkParams,
    rng: &mut StdRng,
) -> usize {
    use crate::parser::model_parser::MixtureClassGuard;
    let spec = model.mixture.as_ref().expect("draw_class non-mixture");
    let k = mix.n_classes;
    let logp = eval_mixing_log_probs(spec, theta, &subject.covariates);
    let mut nll = vec![0.0f64; k];
    for (c, slot) in nll.iter_mut().enumerate() {
        let _g = MixtureClassGuard::enter(c + 1);
        *slot = if kappas.is_empty() {
            individual_nll_into(
                model,
                subject,
                theta,
                eta,
                mix.class_omega(c),
                mix.class_sigma(c),
                scratch,
            )
        } else {
            individual_nll_iov(
                model,
                subject,
                theta,
                eta,
                kappas,
                mix.class_omega(c),
                omega_iov,
                mix.class_sigma(c),
            )
        };
    }
    // combine_subject gives PMIX_ik ∝ exp(logp − nll), softmax-normalised.
    let (_contrib, pmix, _mixest) = combine_subject(&logp, &nll);
    // Sample z ~ Categorical(pmix).
    let u: f64 = rng.random::<f64>();
    let mut acc = 0.0;
    for (c, &p) in pmix.iter().enumerate() {
        acc += p;
        if u <= acc {
            return c;
        }
    }
    k - 1
}

/// M-step for the mixing thetas: maximise the SA-averaged expected complete-data
/// mixing log-likelihood `Σ_i Σ_k r̄_ik ln p_ik(θ_mix)` over `mix.mixing_theta_idx`,
/// holding all other thetas fixed. Writes the updated values back into `theta`.
///
/// Covers constant mixing (closed form `p_k = mean_i r̄_ik`, which the optimiser
/// recovers) and covariate-dependent logit mixing (a weighted multinomial fit)
/// with one code path.
///
/// Runs *after* the θ/σ M-step and overwrites `theta[mixing_theta_idx]`. This
/// assumes the mixing thetas are dedicated to the mixing expression (the
/// conventional parameterisation — `logit(k) = MIXL + …`, NONMEM `P(k)=THETA(j)`
/// — and how every example here is written). A theta that also drives a
/// structural typical value would be double-owned: the residual-likelihood
/// estimate from the θ/σ M-step would be discarded here. There is no clean joint
/// estimate for such a shared parameter; if that pattern is ever needed, split it
/// into two thetas (one for structure, one for mixing). [`mixing_structural_overlap`]
/// detects the pattern up front so `run_saem` can reject it with a clear error.
pub(crate) fn mstep_mixing(
    model: &CompiledModel,
    population: &Population,
    mix: &SaemMixture,
    theta: &mut [f64],
    theta_lower: &[f64],
    theta_upper: &[f64],
    maxiter: u32,
) {
    let idx = &mix.mixing_theta_idx;
    if idx.is_empty() {
        return;
    }
    let spec = model.mixture.as_ref().expect("mstep_mixing non-mixture");
    let m = idx.len();

    // Negative expected complete-data mixing log-likelihood as a function of the
    // free mixing-theta subvector.
    let theta_base = theta.to_vec();
    let rbar = &mix.rbar;
    let obj = |x: &[f64], _grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let mut t = theta_base.clone();
        for (p, &j) in idx.iter().enumerate() {
            t[j] = x[p];
        }
        let mut nll = 0.0;
        for (i, subject) in population.subjects.iter().enumerate() {
            let lp = eval_mixing_log_probs(spec, &t, &subject.covariates);
            for k in 0..mix.n_classes {
                if rbar[i][k] > 0.0 {
                    nll -= rbar[i][k] * lp[k];
                }
            }
        }
        if nll.is_finite() {
            nll
        } else {
            1e20
        }
    };

    let mut x: Vec<f64> = idx.iter().map(|&j| theta[j]).collect();
    let lower: Vec<f64> = idx.iter().map(|&j| theta_lower[j]).collect();
    let upper: Vec<f64> = idx.iter().map(|&j| theta_upper[j]).collect();
    for p in 0..m {
        x[p] = x[p].clamp(lower[p], upper[p]);
    }

    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Bobyqa,
        m,
        obj,
        nlopt::Target::Minimize,
        (),
    );
    opt.set_lower_bounds(&lower).ok();
    opt.set_upper_bounds(&upper).ok();
    opt.set_maxeval(maxiter * (m as u32 + 1)).ok();
    opt.set_ftol_rel(1e-5).ok();
    let _ = opt.optimize(&mut x);

    for (p, &j) in idx.iter().enumerate() {
        theta[j] = x[p].clamp(lower[p], upper[p]);
    }
}

/// Detect mixing thetas that *also* drive the structural model — the pattern
/// [`mstep_mixing`] cannot handle (it would discard the θ/σ M-step's structural
/// estimate). A theta is flagged if perturbing it changes a subject's observation
/// likelihood under any class (the residual/structural path), which a purely-
/// mixing theta (`MIXL`, only in the class logit) never does. Returns the offending
/// theta indices; `run_saem` turns a non-empty result into a clear error so the
/// user splits the parameter rather than getting a silently wrong SAEM fit (#987
/// review). One-time build-cost: `|mixing_theta_idx| · n_classes · n_subjects` NLL
/// evaluations, and `mixing_theta_idx` is a handful of thetas.
pub(crate) fn mixing_structural_overlap(
    model: &CompiledModel,
    init_params: &ModelParameters,
    population: &Population,
    mixing_theta_idx: &[usize],
) -> Vec<usize> {
    use crate::parser::model_parser::MixtureClassGuard;
    if mixing_theta_idx.is_empty() {
        return Vec::new();
    }
    let n_classes = model.mixture.as_ref().map(|m| m.n_classes).unwrap_or(1);
    let n_eta = model.n_eta;
    let eta0 = vec![0.0f64; n_eta];
    let omega = &init_params.omega;
    let sigma = &init_params.sigma.values;
    let mut scratch = EventPkParams::default();
    let mut overlap = Vec::new();
    for &j in mixing_theta_idx {
        let mut moved = false;
        let mut bumped_theta = init_params.theta.clone();
        bumped_theta[j] += 1e-3 * bumped_theta[j].abs().max(1.0);
        'probe: for subject in &population.subjects {
            if subject.observations.is_empty() {
                continue;
            }
            for c in 0..n_classes {
                let _g = MixtureClassGuard::enter(c + 1);
                let base = individual_nll_into(
                    model,
                    subject,
                    &init_params.theta,
                    &eta0,
                    omega,
                    sigma,
                    &mut scratch,
                );
                let bumped = individual_nll_into(
                    model,
                    subject,
                    &bumped_theta,
                    &eta0,
                    omega,
                    sigma,
                    &mut scratch,
                );
                if base.is_finite() && bumped.is_finite() && (bumped - base).abs() > 1e-9 {
                    moved = true;
                    break 'probe;
                }
            }
        }
        if moved {
            overlap.push(j);
        }
    }
    overlap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use rand::SeedableRng;

    // Constant (covariate-free) two-class mixing on clearance.
    const CONST_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    // Covariate-dependent logit mixing (WT enters the class logit).
    const COV_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BWT(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    fn read_pop(csv: &str) -> Population {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// Small 1-cpt IV dataset, `n_per` subjects at two weights.
    fn iv_csv(n_per: usize) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        let mut sid = 0;
        for (g, &wt) in [60.0_f64, 90.0].iter().enumerate() {
            let cl = if g == 0 { 1.0 } else { 3.0 };
            for _ in 0..n_per {
                sid += 1;
                s.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
                for (ti, t) in [0.5_f64, 1.0, 2.0, 4.0].iter().enumerate() {
                    let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                    let dv = c * (1.0 + 0.02 * ((sid + ti) as f64).sin());
                    s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
                }
            }
        }
        s
    }

    #[test]
    fn build_detects_constant_mixing_theta() {
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        // Only MIXL (theta index 3) enters the mixing expression.
        assert_eq!(mix.mixing_theta_idx, vec![3]);
        assert_eq!(mix.n_classes, 2);
    }

    #[test]
    fn build_detects_covariate_mixing_thetas() {
        let model = parse_model_string(COV_MODEL).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        // MIXL (3) and BWT (4) both enter the class logit.
        assert_eq!(mix.mixing_theta_idx, vec![3, 4]);
    }

    /// A covariate that enters the class logit purely multiplicatively and is 0
    /// at its default value must still be detected — the detection probes real
    /// subject covariates, not a covariate-0 map (#987 review).
    #[test]
    fn build_detects_zero_default_product_covariate() {
        const MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BSEX(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL + BSEX*SEX

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        // Half the subjects have SEX = 1 so the product term is non-zero for
        // some subject even though SEX defaults to 0.
        let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,SEX\n");
        for sid in 1..=4 {
            let sex = sid % 2; // alternate 1,0,1,0
            csv.push_str(&format!("{sid},0,0,100,1,1,{sex}\n"));
            csv.push_str(&format!("{sid},1,5.0,0,0,1,{sex}\n"));
        }
        let model = parse_model_string(MODEL).unwrap();
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), Some(&["SEX"]), None).unwrap();
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        // MIXL (3) and BSEX (4) both moved p_k for at least one subject.
        assert_eq!(mix.mixing_theta_idx, vec![3, 4]);
    }

    #[test]
    fn update_rbar_is_robbins_monro_average() {
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(1));
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        // Force known classes and a fresh rbar, then take one SA step.
        mix.classes = vec![0, 1];
        mix.rbar = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        mix.update_rbar(0.5);
        // Subject 0 drew class 0: rbar_0 = 0.5*[0.5,0.5] + 0.5*[1,0] = [0.75,0.25].
        assert!((mix.rbar[0][0] - 0.75).abs() < 1e-12);
        assert!((mix.rbar[0][1] - 0.25).abs() < 1e-12);
        // Subject 1 drew class 1: [0.25, 0.75].
        assert!((mix.rbar[1][0] - 0.25).abs() < 1e-12);
        assert!((mix.rbar[1][1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn mstep_mixing_recovers_constant_frequency() {
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(5)); // 10 subjects
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        // Target class-1 fraction q = 0.3 for every subject (constant mixing ⇒
        // the mixing M-step should drive p(1) = mean_i r̄_i1 = 0.3).
        let q = 0.3;
        for r in mix.rbar.iter_mut() {
            *r = vec![q, 1.0 - q];
        }
        let mut theta = model.default_params.theta.clone();
        mstep_mixing(
            &model,
            &pop,
            &mix,
            &mut theta,
            &model.default_params.theta_lower,
            &model.default_params.theta_upper,
            50,
        );
        // p(1) = σ(MIXL) must recover q.
        let p1 = 1.0 / (1.0 + (-theta[3]).exp());
        assert!((p1 - q).abs() < 1e-3, "recovered p(1) {p1} vs {q}");
    }

    #[test]
    fn draw_class_returns_valid_categorical() {
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(1)); // 2 subjects
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        let mut scratch = EventPkParams::default();
        let mut rng = StdRng::seed_from_u64(1);
        let eta = vec![0.0_f64];
        // Every draw must be a valid class index (0 or 1).
        for _ in 0..50 {
            let z = draw_class(
                &model,
                &pop.subjects[0],
                &model.default_params.theta,
                &eta,
                &[],
                &mix,
                None,
                &mut scratch,
                &mut rng,
            );
            assert!(z < 2, "class index in range");
        }
    }

    #[test]
    fn mstep_omega_overrides_noop_without_overrides() {
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(1));
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        let before = mix.s2_diag.clone();
        // No omega(k) overrides declared ⇒ the SA update touches nothing.
        mix.classes = vec![0, 1];
        mix.mstep_omega_overrides(&[vec![0.5], vec![-0.5]], 1.0);
        assert_eq!(mix.s2_diag, before);
    }

    #[test]
    fn overlap_empty_for_dedicated_mixing_theta() {
        // MIXL is only in the class logit, never in a typical value ⇒ no overlap.
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        let overlap =
            mixing_structural_overlap(&model, &model.default_params, &pop, &mix.mixing_theta_idx);
        assert!(
            overlap.is_empty(),
            "dedicated MIXL must not flag: {overlap:?}"
        );
    }

    #[test]
    fn overlap_detects_theta_shared_with_structure() {
        // TVV drives V structurally AND appears in the mixing logit — the shared
        // parameter SAEM cannot fit; the probe must flag its index (2).
        const SHARED: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = 0.01 * TVV

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = parse_model_string(SHARED).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        // TVV (index 2) must be detected as a mixing theta AND flagged as shared.
        assert!(mix.mixing_theta_idx.contains(&2));
        let overlap =
            mixing_structural_overlap(&model, &model.default_params, &pop, &mix.mixing_theta_idx);
        assert_eq!(
            overlap,
            vec![2],
            "TVV shared with structure must be flagged"
        );
    }

    // Per-class Ω / σ overrides on top of the constant-mixing model.
    const OVERRIDE_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.25
  sigma(2) EPS ~ 0.09

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// A `FIX`-ed mixing theta must never enter `mixing_theta_idx`: FIX is
    /// enforced elsewhere by collapsing the *packed* bounds, which `mstep_mixing`
    /// (working on natural-scale user bounds) cannot see (#987 review).
    #[test]
    fn build_excludes_fixed_mixing_theta() {
        let fixed_model = CONST_MODEL.replace(
            "theta MIXL(0.0, -10.0, 10.0)",
            "theta MIXL(0.4, -10.0, 10.0, FIX)",
        );
        let model = parse_model_string(&fixed_model).unwrap();
        assert!(
            model.default_params.theta_fixed[3],
            "MIXL must parse as FIX"
        );
        let pop = read_pop(&iv_csv(2));
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        assert!(
            mix.mixing_theta_idx.is_empty(),
            "FIXed MIXL must not be optimised: {:?}",
            mix.mixing_theta_idx
        );
    }

    /// End of the same story: with the mixing theta FIXed, the mixing M-step is a
    /// no-op even when the responsibilities point somewhere else entirely.
    #[test]
    fn mstep_mixing_holds_fixed_theta() {
        let fixed_model = CONST_MODEL.replace(
            "theta MIXL(0.0, -10.0, 10.0)",
            "theta MIXL(0.4, -10.0, 10.0, FIX)",
        );
        let model = parse_model_string(&fixed_model).unwrap();
        let pop = read_pop(&iv_csv(5));
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        for r in mix.rbar.iter_mut() {
            *r = vec![0.05, 0.95]; // far from σ(0.4) ≈ 0.599
        }
        let mut theta = model.default_params.theta.clone();
        let before = theta[3];
        mstep_mixing(
            &model,
            &pop,
            &mix,
            &mut theta,
            &model.default_params.theta_lower,
            &model.default_params.theta_upper,
            50,
        );
        assert_eq!(theta[3], before, "FIXed MIXL must not move");
    }

    /// The base Ω statistic must be built only from the subjects that share the
    /// base entry — an `omega(2)` override on ETA_CL means class-2 members are
    /// drawn from a different variance and must not pollute the base ω(ETA_CL).
    /// The un-overridden coordinate (ETA_V) still pools every subject.
    #[test]
    fn base_eta_outer_excludes_override_class_members() {
        let model = parse_model_string(OVERRIDE_MODEL).unwrap();
        let pop = read_pop(&iv_csv(2)); // 4 subjects
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        assert_eq!(mix.mp.omega_override_addr, vec![(1, 0)]);
        mix.classes = vec![0, 0, 1, 1];
        // ETA_CL (coord 0) large only for the class-2 members, so pooling would
        // be visibly different from the class-1-only mean.
        let etas = vec![
            vec![0.1, 1.0],
            vec![0.3, 2.0],
            vec![5.0, 3.0],
            vec![7.0, 4.0],
        ];
        let (mean, counts) = mix.base_eta_outer(&etas, 2);
        // (0,0): only classes without an override on ETA_CL → subjects 0 and 1.
        assert_eq!(counts[(0, 0)], 2);
        assert!((mean[(0, 0)] - (0.01 + 0.09) / 2.0).abs() < 1e-12);
        // (1,1): no class overrides ETA_V → all four subjects.
        assert_eq!(counts[(1, 1)], 4);
        assert!((mean[(1, 1)] - (1.0 + 4.0 + 9.0 + 16.0) / 4.0).abs() < 1e-12);
        // Cross term (0,1) touches ETA_CL, so it too drops the override class.
        assert_eq!(counts[(0, 1)], 2);
        assert!((mean[(0, 1)] - (0.1 * 1.0 + 0.3 * 2.0) / 2.0).abs() < 1e-12);
    }

    /// With no override declared the partitioned statistic must reduce exactly to
    /// the pooled `mean_i η_i η_iᵀ` the non-mixture path uses.
    #[test]
    fn base_eta_outer_reduces_to_pooled_without_overrides() {
        let model = parse_model_string(CONST_MODEL).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        mix.classes = vec![0, 1, 0, 1];
        let etas = vec![vec![0.2], vec![-0.5], vec![1.1], vec![0.0]];
        let (mean, counts) = mix.base_eta_outer(&etas, 1);
        assert_eq!(counts[(0, 0)], 4);
        let pooled = (0.04 + 0.25 + 1.21 + 0.0) / 4.0;
        assert!((mean[(0, 0)] - pooled).abs() < 1e-12);
    }

    /// During the Ω burn-in the override is held at its declared init even though
    /// its SA statistic is already accumulating — mirroring the base Ω, which is
    /// likewise held while `s2` warms up (#987 review).
    #[test]
    fn sync_base_holds_override_at_init_during_burnin() {
        let model = parse_model_string(OVERRIDE_MODEL).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mut mix = SaemMixture::build(&model, &model.default_params, &pop);
        let base_omega = model.default_params.omega.clone();
        let base_sigma = model.default_params.sigma.values.clone();
        // Drive the SA statistic well away from the init.
        mix.classes = vec![1, 1, 1, 1];
        mix.mstep_omega_overrides(
            &[
                vec![2.0, 0.0],
                vec![2.0, 0.0],
                vec![2.0, 0.0],
                vec![2.0, 0.0],
            ],
            1.0,
        );
        assert!((mix.s2_diag[1][0] - 4.0).abs() < 1e-12);

        // Burn-in: the declared 0.25 is what the E-step sees.
        mix.omega_stat_active = false;
        mix.sync_base(&base_omega, &base_sigma);
        assert!((mix.class_omega(1).matrix[(0, 0)] - 0.25).abs() < 1e-12);

        // After burn-in the statistic takes over.
        mix.omega_stat_active = true;
        mix.sync_base(&base_omega, &base_sigma);
        assert!((mix.class_omega(1).matrix[(0, 0)] - 4.0).abs() < 1e-12);
    }

    /// The θ/σ M-step scores a class-`k` subject under that class's held σ, so
    /// the override addresses must surface as `[class][(index, value)]`.
    #[test]
    fn class_sigma_overrides_reports_held_values() {
        let model = parse_model_string(OVERRIDE_MODEL).unwrap();
        let pop = read_pop(&iv_csv(1));
        let mix = SaemMixture::build(&model, &model.default_params, &pop);
        let over = mix.class_sigma_overrides();
        assert_eq!(over.len(), 2);
        assert!(over[0].is_empty(), "class 1 has no σ override");
        assert_eq!(over[1].len(), 1);
        assert_eq!(over[1][0].0, 0);
        // `sigma` is declared as a variance and stored as the SD.
        assert!((over[1][0].1 - 0.09_f64.sqrt()).abs() < 1e-12);
    }

    /// Regression for the class-unaware κ MH sweep (#987 review): under a
    /// mixture, per-occasion κ must be proposed inside the subject's *drawn*
    /// class. Sampling every subject's κ against the class-1 `MIXNUM` branch
    /// forces class-2 members to absorb the whole `TVCL2 / TVCL1` ratio into κ,
    /// which then inflates the Ω_IOV sufficient statistic.
    ///
    /// The data are strongly bimodal in CL with *no* real IOV, and every
    /// parameter except Ω_IOV is FIXed at truth — so a correct sweep leaves
    /// ω_IOV near its (small) init while the class-blind one drives it toward
    /// `ln(TVCL2/TVCL1)² ≈ 2.6`.
    #[test]
    fn saem_mixture_iov_kappa_is_sampled_within_drawn_class() {
        const MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0, FIX)
  theta TVCL2(5.0, 0.01, 100.0, FIX)
  theta TVV(10.0, 0.1, 1000.0, FIX)
  theta MIXL(0.0, -10.0, 10.0, FIX)
  omega ETA_CL ~ 0.01 FIX
  kappa KAPPA_CL ~ 0.04
  sigma EPS ~ 0.01 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL + KAPPA_CL) else TVCL2 * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        // 8 subjects, 2 occasions each; half at CL = 1 (class 1), half at CL = 5
        // (class 2). Dosed at the start of each occasion, no IOV in the truth.
        let mut csv = String::from(
            "ID,TIME,DV,AMT,EVID,CMT,OCC
",
        );
        for sid in 1..=8 {
            let cl: f64 = if sid <= 4 { 1.0 } else { 5.0 };
            for occ in 1..=2 {
                let t0 = 24.0 * (occ - 1) as f64;
                csv.push_str(&format!(
                    "{sid},{t0},0,100,1,1,{occ}
"
                ));
                for dt in [0.5_f64, 1.0, 2.0, 4.0, 8.0] {
                    let c = (100.0 / 10.0) * (-(cl / 10.0) * dt).exp();
                    csv.push_str(&format!(
                        "{sid},{},{c:.6},0,0,1,{occ}
",
                        t0 + dt
                    ));
                }
            }
        }
        let model = parse_model_string(MODEL).unwrap();
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();

        let opts = crate::types::FitOptions {
            method: crate::types::EstimationMethod::Saem,
            saem_n_exploration: 20,
            saem_n_convergence: 20,
            saem_omega_burnin: 2,
            saem_seed: Some(7),
            run_covariance_step: false,
            verbose: false,
            iov_column: Some("OCC".to_string()),
            ..Default::default()
        };

        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("mixture + IOV SAEM fit");
        let omega_iov = res
            .omega_iov
            .as_ref()
            .expect("IOV model must report Omega_IOV");
        let w = omega_iov[(0, 0)];
        // Measured: 0.0065 with the class-aware sweep, 0.165 without it.
        assert!(
            w < 0.05,
            "omega_IOV must not absorb the between-class CL ratio (got {w}); \
             the kappa MH sweep is sampling outside the drawn class"
        );
    }

    #[test]
    fn run_saem_rejects_mixing_theta_shared_with_structure() {
        // End-to-end: fit() must error clearly (before any iteration) rather than
        // silently double-owning the shared parameter (#987 review).
        const SHARED: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = 0.01 * TVV

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = parse_model_string(SHARED).unwrap();
        let pop = read_pop(&iv_csv(2));
        let mut opts = crate::types::FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        let err = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect_err("shared mixing/structural theta under SAEM must be rejected");
        assert!(
            err.contains("mixing-coefficient theta also drives the structural model")
                && err.contains("TVV"),
            "got: {err}"
        );
    }
}
