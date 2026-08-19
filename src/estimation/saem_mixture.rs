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
    /// Theta indices the mixing expressions depend on (detected numerically at
    /// build time). Only these are moved by the mixing M-step.
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
            if moves {
                mixing_theta_idx.push(j);
            }
        }

        // Per-class Ω-override SA statistic seeded from the init class Ω diagonal.
        let s2_diag: Vec<Vec<f64>> = (0..n_classes)
            .map(|c| (0..n_eta).map(|j| mp.omega[c].matrix[(j, j)]).collect())
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
        // Restore Ω-override diagonals from the per-class SA statistic.
        let addrs = self.mp.omega_override_addr.clone();
        for &(c, j) in &addrs {
            self.mp.omega[c].matrix[(j, j)] = self.s2_diag[c][j].max(OVERRIDE_OMEGA_DIAG_FLOOR);
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
