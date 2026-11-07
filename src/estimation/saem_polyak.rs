//! SAEM's accumulation phase as stochastic approximation on the **marginal score**, with
//! Polyak–Ruppert averaging and a zero-variance control variate
//! (`saem_accumulation = polyak`, #2721).
//!
//! SAEM is stochastic root-finding on `h(ψ) = Σᵢ E[sᵢ(ηᵢ, ψ) | yᵢ]`, the marginal score by
//! Fisher's identity, and its default accumulation phase is one particular recursion on it:
//! the closed-form EM update for Ω and the mu-referenced θ, and the numerical (`bobyqa`)
//! or `score_sa` step for the rest, each with steps `γₖ = 1/k`. This module replaces it,
//! inside the scope [`AccumScope`] admits, by:
//!
//! - **One recursion on the parameter vector** `ψ ← ψ + γₖ·I_c⁻¹·ĥ`, where `ĥ` is the
//!   summed complete-data score at the current draws and `I_c` the complete-data expected
//!   information — the gain the closed forms use, now applied jointly to every
//!   coordinate.
//! - **Polyak–Ruppert averaging.** Steps `γₖ = k^{−2/3}`, and the estimate is the average
//!   of the iterates over the second half of accumulation. The averaged iterate's
//!   asymptotic variance is optimal whatever the gain (Polyak & Juditsky 1992), so a
//!   coordinate whose information is mostly missing — 70–99 % for σ and most of Ω on
//!   #1445's sparse design — is no longer stuck at the rate its gain allows.
//! - **A zero-variance control variate.** Under each subject's posterior `E[∇η l] = 0`
//!   (Stein), and the adjoint returns `∇η l` exactly. Each subject's score is used as
//!   `sᵢ − Bᵢ·∇η lᵢ`, with `Bᵢ = Cov(s, ∇η l)·Var(∇η l)⁻¹` estimated from a burst of
//!   posterior draws at the start of accumulation. `Bᵢ` is fixed before the draws it is
//!   applied to, so the expectation of the score, and with it the fixed point, is
//!   unchanged; only its variance falls.
//!
//! **Why the gain is not the Louis observed information.** `I_obs⁻¹` (the Newton gain)
//! accelerates a direction with observed-to-complete ratio `λ` by `1/λ`, and with it the
//! Monte Carlo noise of the one-draw score: the iterates' fluctuation grows as
//! `γ(1 − λ)/(2λ)` statistical variances. Measured on #1445's fixture (importance-sampled
//! −2 log L; this module's arm 3181.67 ± 0.48 over 10 seeds, the classic one
//! 3183.46 ± 0.68): the Louis gain with `λ` floored at 0.01 landed at 3193.9–3212.5 on
//! every seed tried (3); floored at 0.1 at 3221 and 3347 (2 seeds); floored at 0.25 above
//! 3190 on 1–2 seeds in 10 (up to 3334); capped at a full EM step per direction above 3190
//! on 1 seed in 10 with the control variate and 5 in 10 without (up to 3354). At most 2×
//! acceleration (floor 0.5) never failed but gave 3181.22 ± 0.54, within two standard
//! errors of none, so it is not adopted. In the directions where the gain would matter,
//! the binding constraint is the Monte Carlo noise, which averaging and the control
//! variate reduce and acceleration amplifies.
//!
//! The coordinates, each in the parameterisation SAEM itself steps in:
//!
//! - a mu-referenced θ on its mu scale (the packed value, or the natural one for an
//!   additive anchor): prior score `[Ω⁻¹η]ⱼ`, after which the anchored η is re-centred so
//!   the individual parameter `φ = μ + η` is unchanged, as the closed form does;
//! - every other free θ and every free σ, packed: the exact data score at fixed η
//!   (`obs_nll_subject_grad`) and its expected Fisher information;
//! - the free entries of Ω: prior score `½(Ω⁻¹ηηᵀΩ⁻¹ − Ω⁻¹)` and expected information
//!   `½tr(Ω⁻¹E_rΩ⁻¹E_c)`.

use crate::estimation::fixed_eta_gradient::{obs_nll_subject_grad, obs_nll_subject_grad_fisher};
use crate::estimation::saem::MhScratch;
use crate::pk::EventPkParams;
use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, Population, Subject};
use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Posterior draws per subject in the burst that estimates the gain and the control
/// variate's coefficients, each `n_mh_steps` kernel steps apart.
pub(crate) const POLYAK_BURST_DRAWS: usize = 50;

/// Exponent of the accumulation step `γₖ = k^{−α}`; `α ∈ (½, 1)` is the Polyak–Ruppert
/// range.
pub(crate) const POLYAK_STEP_EXPONENT: f64 = 2.0 / 3.0;

/// Largest step any coordinate takes in one iteration (packed units), as `score_sa`'s
/// trust region.
const POLYAK_MAX_STEP: f64 = 0.5;

/// A mu-referenced θ and its η.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MuCoord {
    pub theta: usize,
    pub eta: usize,
    /// `P = THETA + ETA`: the mu scale is the natural θ, not the packed one.
    pub additive: bool,
}

/// Which parameters the accumulation moves, and in which coordinates.
#[derive(Clone, Debug)]
pub(crate) struct AccumScope {
    pub mu: Vec<MuCoord>,
    /// Indices into the packed `[θ, σ]` vector.
    pub data: Vec<usize>,
    /// Free, non-FIX lower-triangle Ω entries `(a, b)`, `a >= b`.
    pub om: Vec<(usize, usize)>,
    pub n_theta: usize,
    pub n_sigma: usize,
    pub n_eta: usize,
}

impl AccumScope {
    pub(crate) fn n(&self) -> usize {
        self.mu.len() + self.data.len() + self.om.len()
    }

    /// Build the coordinates. `mu` are the closed-form pairs `run_saem` resolved (plain
    /// and additive); every other free θ and every free σ is data-scored.
    pub(crate) fn new(params: &ModelParameters, mu: Vec<MuCoord>, n_eta: usize) -> Self {
        let n_theta = params.theta.len();
        let n_sigma = params.sigma.values.len();
        let mut data = Vec::new();
        for t in 0..n_theta {
            let fixed = params.theta_fixed.get(t).copied().unwrap_or(false)
                || params.theta_lower[t] >= params.theta_upper[t];
            if !fixed && !mu.iter().any(|m| m.theta == t) {
                data.push(t);
            }
        }
        for k in 0..n_sigma {
            if !params.sigma_fixed.get(k).copied().unwrap_or(false) {
                data.push(n_theta + k);
            }
        }
        let mut om = Vec::new();
        for a in 0..n_eta {
            for b in 0..=a {
                let fixed = params.omega_fixed.get(a).copied().unwrap_or(false)
                    || params.omega_fixed.get(b).copied().unwrap_or(false);
                if params.omega.free_mask[(a, b)] && !fixed {
                    om.push((a, b));
                }
            }
        }
        Self {
            mu,
            data,
            om,
            n_theta,
            n_sigma,
            n_eta,
        }
    }
}

/// The symmetric unit `E` of Ω entry `(a, b)`: `e_a e_bᵀ + e_b e_aᵀ` off the diagonal,
/// `e_a e_aᵀ` on it.
fn sym_unit(n: usize, a: usize, b: usize) -> DMatrix<f64> {
    let mut e = DMatrix::zeros(n, n);
    e[(a, b)] = 1.0;
    e[(b, a)] = 1.0;
    e
}

/// The current point.
pub(crate) struct Point<'a> {
    pub theta: &'a [f64],
    pub sigma: &'a [f64],
    pub omega_inv: &'a DMatrix<f64>,
    pub packs_log: &'a [bool],
    /// Packed `[θ, σ]` bounds (a pinned coordinate has `lower == upper`).
    pub lower: &'a [f64],
    pub upper: &'a [f64],
}

/// The prior part of the complete-data score in [`AccumScope`]'s order (the data block
/// left at zero): `log N(φ; μ, Ω)` with `η = φ − μ`, differentiated in the mu-referenced
/// μ and the free Ω entries.
pub(crate) fn prior_score(sc: &AccumScope, eta: &[f64], oi: &DMatrix<f64>) -> DVector<f64> {
    let (n_mu, n_data) = (sc.mu.len(), sc.data.len());
    let ev = DVector::from_column_slice(eta);
    let oe = oi * &ev;
    let gm = (&oe * oe.transpose() - oi) * 0.5;
    let mut s = DVector::<f64>::zeros(sc.n());
    for (r, m) in sc.mu.iter().enumerate() {
        s[r] = oe[m.eta];
    }
    for (r, &(a, b)) in sc.om.iter().enumerate() {
        s[n_mu + n_data + r] = if a == b { gm[(a, a)] } else { 2.0 * gm[(a, b)] };
    }
    s
}

/// The prior part of the complete-data **expected** information, per subject: `Ω⁻¹` on μ,
/// `½tr(Ω⁻¹E_rΩ⁻¹E_c)` on Ω, zero across (the cross term is odd in η).
pub(crate) fn prior_info(sc: &AccumScope, oi: &DMatrix<f64>) -> DMatrix<f64> {
    let (n_mu, n_data) = (sc.mu.len(), sc.data.len());
    let mut h = DMatrix::<f64>::zeros(sc.n(), sc.n());
    for (r, m) in sc.mu.iter().enumerate() {
        for (c, m2) in sc.mu.iter().enumerate() {
            h[(r, c)] = oi[(m.eta, m2.eta)];
        }
    }
    let units: Vec<DMatrix<f64>> = sc
        .om
        .iter()
        .map(|&(a, b)| sym_unit(sc.n_eta, a, b))
        .collect();
    for (r, er) in units.iter().enumerate() {
        let oero = oi * er * oi;
        for (c, ec) in units.iter().enumerate() {
            h[(n_mu + n_data + r, n_mu + n_data + c)] = 0.5 * (&oero * ec).trace();
        }
    }
    h
}

/// One subject's complete-data score at one draw and, when asked, the expected Fisher
/// information of its data block (row-major over the packed `[θ, σ]`). `None` when the
/// data score or information is out of scope or not finite.
fn complete_score(
    sc: &AccumScope,
    model: &CompiledModel,
    subject: &Subject,
    eta: &[f64],
    p: &Point,
    want_fisher: bool,
    pk: &mut EventPkParams,
) -> Option<(DVector<f64>, Option<Vec<f64>>)> {
    let (g, fis) = if sc.data.is_empty() {
        (Vec::new(), None)
    } else if want_fisher {
        let (_, g, f) = obs_nll_subject_grad_fisher(
            model,
            subject,
            p.theta,
            p.sigma,
            eta,
            p.packs_log,
            p.lower,
            p.upper,
            sc.n_theta,
            sc.n_sigma,
            pk,
        );
        (g, Some(f?))
    } else {
        let (_, g) = obs_nll_subject_grad(
            model,
            subject,
            p.theta,
            p.sigma,
            eta,
            p.packs_log,
            p.lower,
            p.upper,
            sc.n_theta,
            sc.n_sigma,
            pk,
        );
        (g, None)
    };
    let mut s = prior_score(sc, &eta[..sc.n_eta], p.omega_inv);
    for (r, &i) in sc.data.iter().enumerate() {
        s[sc.mu.len() + r] = -g[i];
    }
    s.iter().all(|v| v.is_finite()).then_some((s, fis))
}

/// The accumulation-phase state.
pub(crate) struct PolyakAccum {
    pub sc: AccumScope,
    /// The data block's SA-averaged expected Fisher information (θ/σ), as `score_sa`
    /// keeps it; the prior blocks' information is exact at the current Ω each iteration.
    data_info: DMatrix<f64>,
    /// Per-subject control-variate coefficients `Bᵢ` (`n × n_eta`); `None` when off.
    cv: Option<Vec<DMatrix<f64>>>,
    avg_sum: DVector<f64>,
    avg_omega: DMatrix<f64>,
    avg_n: usize,
    /// `Σ Var(s − B·∇η l) / Σ Var(s)` over the burst (1 without the control variate).
    pub cv_ratio: f64,
}

/// Per-subject burst moments.
struct Moments {
    k: f64,
    s: DVector<f64>,
    ss: DMatrix<f64>,
    e: DVector<f64>,
    ee: DMatrix<f64>,
    se: DMatrix<f64>,
    fisher: DMatrix<f64>,
}

/// One subject's burst: `draws` posterior draws, `thin` kernel steps apart, continuing the
/// subject's chain (`eta`, `nll` advanced in place). `None` when a score or gradient is
/// out of scope.
#[allow(clippy::too_many_arguments)]
fn burst_subject(
    sc: &AccumScope,
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    omega: &OmegaMatrix,
    p: &Point,
    eta: &mut [f64],
    nll: &mut f64,
    step_scale: f64,
    draws: usize,
    thin: usize,
    want_cv: bool,
    rng: &mut impl Rng,
) -> Option<Moments> {
    let n = sc.n();
    let nd = sc.data.len();
    let nn = sc.n_theta + sc.n_sigma;
    let mut m = Moments {
        k: 0.0,
        s: DVector::zeros(n),
        ss: DMatrix::zeros(n, n),
        e: DVector::zeros(sc.n_eta),
        ee: DMatrix::zeros(sc.n_eta, sc.n_eta),
        se: DMatrix::zeros(n, sc.n_eta),
        fisher: DMatrix::zeros(nd, nd),
    };
    let mut mh = MhScratch::default();
    mh.begin_subject(model, subject, p.theta, sc.n_eta);
    let mut pk = EventPkParams::default();
    for _ in 0..draws {
        let (_, v) = crate::estimation::saem::mh_steps(
            eta, *nll, subject, model, p.theta, omega, p.sigma, step_scale, None, rng, thin,
            &mut mh, None, None,
        );
        *nll = v;
        let (s, f) = complete_score(sc, model, subject, eta, p, true, &mut pk)?;
        if let Some(f) = f {
            for (r, &i) in sc.data.iter().enumerate() {
                for (c, &j) in sc.data.iter().enumerate() {
                    m.fisher[(r, c)] += f[i * nn + j];
                }
            }
        }
        if want_cv {
            let e = crate::sens::ode_provider::adjoint::exact_inner_gradient(
                model, subject, params, eta, None,
            )?;
            let e = DVector::from_column_slice(&e[..sc.n_eta]);
            if !e.iter().all(|v| v.is_finite()) {
                return None;
            }
            m.ee += &e * e.transpose();
            m.se += &s * e.transpose();
            m.e += e;
        }
        m.ss += &s * s.transpose();
        m.s += s;
        m.k += 1.0;
    }
    Some(m)
}

impl PolyakAccum {
    /// Run the burst at the current point and build the gain and the control variate.
    /// `etas` / `nlls` are each subject's chain state and are advanced in place.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start(
        sc: AccumScope,
        model: &CompiledModel,
        population: &Population,
        params: &ModelParameters,
        p: &Point,
        etas: &mut [Vec<f64>],
        nlls: &mut [f64],
        step_scales: &[f64],
        thin: usize,
        want_cv: bool,
        seed: u64,
    ) -> Result<Self, String> {
        use rand::SeedableRng;
        use rayon::prelude::*;
        let n = sc.n();
        let n_eta = sc.n_eta;
        let nd = sc.data.len();
        // Per-subject seeded streams, folded in subject order, so the result does not
        // depend on the thread count (#703).
        let per: Vec<Option<(Moments, Vec<f64>, f64)>> = population
            .subjects
            .par_iter()
            .enumerate()
            .map(|(i, subject)| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let mut eta = etas[i].clone();
                let mut nll = nlls[i];
                let m = burst_subject(
                    &sc,
                    model,
                    subject,
                    params,
                    &params.omega,
                    p,
                    &mut eta,
                    &mut nll,
                    step_scales[i],
                    POLYAK_BURST_DRAWS,
                    thin,
                    want_cv,
                    &mut rng,
                )?;
                Some((m, eta, nll))
            })
            .collect();
        let mut data_info = DMatrix::<f64>::zeros(nd, nd);
        let (mut var_cv, mut var_s) = (0.0, 0.0);
        let mut cv = Vec::with_capacity(population.subjects.len());
        for (i, r) in per.into_iter().enumerate() {
            let Some((m, eta, nll)) = r else {
                return Err(format!(
                    "the complete-data score or the exact η-gradient is out of scope for subject {}",
                    population.subjects[i].id
                ));
            };
            etas[i] = eta;
            nlls[i] = nll;
            let k = m.k;
            data_info += &m.fisher / k;
            if want_cv {
                let sbar = &m.s / k;
                let ebar = &m.e / k;
                let vs = &m.ss / k - &sbar * sbar.transpose();
                let ve = &m.ee / k - &ebar * ebar.transpose();
                let cse = &m.se / k - &sbar * ebar.transpose();
                let b = match ve.try_inverse() {
                    Some(vi) => &cse * vi,
                    None => DMatrix::zeros(n, n_eta),
                };
                var_s += vs.trace();
                var_cv += (&vs - &b * cse.transpose()).trace();
                cv.push(b);
            }
        }
        Ok(Self {
            avg_sum: DVector::zeros(n),
            avg_omega: DMatrix::zeros(n_eta, n_eta),
            avg_n: 0,
            sc,
            data_info,
            cv: want_cv.then_some(cv),
            cv_ratio: if want_cv && var_s > 0.0 {
                var_cv / var_s
            } else {
                1.0
            },
        })
    }

    /// The summed (control-variate corrected) score over subjects at their current draws,
    /// and the summed expected Fisher information of the data block there.
    pub(crate) fn score(
        &self,
        model: &CompiledModel,
        population: &Population,
        params: &ModelParameters,
        p: &Point,
        etas: &[Vec<f64>],
    ) -> Option<(DVector<f64>, DMatrix<f64>)> {
        use rayon::prelude::*;
        let sc = &self.sc;
        let nd = sc.data.len();
        let nn = sc.n_theta + sc.n_sigma;
        let per: Vec<Option<(DVector<f64>, Option<Vec<f64>>)>> = population
            .subjects
            .par_iter()
            .zip(etas.par_iter())
            .enumerate()
            .map_init(EventPkParams::default, |pk, (i, (subject, eta))| {
                let (mut s, f) = complete_score(sc, model, subject, eta, p, nd > 0, pk)?;
                if let Some(cv) = &self.cv {
                    let e = crate::sens::ode_provider::adjoint::exact_inner_gradient(
                        model, subject, params, eta, None,
                    )?;
                    let e = DVector::from_column_slice(&e[..sc.n_eta]);
                    if !e.iter().all(|v| v.is_finite()) {
                        return None;
                    }
                    s -= &cv[i] * e;
                }
                Some((s, f))
            })
            .collect();
        let mut h = DVector::<f64>::zeros(sc.n());
        let mut fisher = DMatrix::<f64>::zeros(nd, nd);
        for r in per {
            let (s, f) = r?;
            h += s;
            if let Some(f) = f {
                for (a, &i) in sc.data.iter().enumerate() {
                    for (b, &j) in sc.data.iter().enumerate() {
                        fisher[(a, b)] += f[i * nn + j];
                    }
                }
            }
        }
        fisher.iter().all(|v| v.is_finite()).then_some((h, fisher))
    }

    /// The step `γ·I_c⁻¹·ĥ` at the current point. The prior blocks' information is exact
    /// at the current Ω (`n_subjects·prior_info`), so on Ω and μ without the control
    /// variate the step is the EM-type convex combination `γ(S − Ω)` / `γ·mean(η)`. The data
    /// block's is the SA average of the expected Fisher (updated here at the same γ). The
    /// data block's step is clamped per coordinate to [`POLYAK_MAX_STEP`]. `None` when an
    /// information block is not positive definite.
    pub(crate) fn direction(
        &mut self,
        h: &DVector<f64>,
        fisher: &DMatrix<f64>,
        oi: &DMatrix<f64>,
        n_subjects: usize,
        gamma: f64,
    ) -> Option<DVector<f64>> {
        let sc = &self.sc;
        let (n_mu, nd) = (sc.mu.len(), sc.data.len());
        let g = gamma.clamp(0.0, 1.0);
        self.data_info += (fisher - &self.data_info) * g;
        let mut info = prior_info(sc, oi) * n_subjects as f64;
        info.view_mut((n_mu, n_mu), (nd, nd))
            .copy_from(&self.data_info);
        let mut step = info.cholesky()?.solve(h) * g;
        for r in n_mu..n_mu + nd {
            step[r] = step[r].clamp(-POLYAK_MAX_STEP, POLYAK_MAX_STEP);
        }
        step.iter().all(|v| v.is_finite()).then_some(step)
    }

    /// Add the current point to the Polyak–Ruppert average.
    pub(crate) fn accumulate(&mut self, coords: &DVector<f64>, omega: &DMatrix<f64>) {
        self.avg_sum += coords;
        self.avg_omega += omega;
        self.avg_n += 1;
    }

    /// The averaged coordinates and Ω, when any iterate was averaged.
    pub(crate) fn average(&self) -> Option<(DVector<f64>, DMatrix<f64>)> {
        (self.avg_n > 0).then(|| {
            let k = self.avg_n as f64;
            (&self.avg_sum / k, &self.avg_omega / k)
        })
    }
}

/// The accumulation step `γₖ = k^{−α}` for the `k`-th accumulation iteration (1-based).
pub(crate) fn polyak_gamma(k_acc: usize) -> f64 {
    (k_acc.max(1) as f64).powf(-POLYAK_STEP_EXPONENT)
}

#[cfg(test)]
#[path = "saem_polyak_tests.rs"]
mod tests;
