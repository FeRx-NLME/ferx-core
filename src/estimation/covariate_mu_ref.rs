//! Covariate-aware (multi-theta) mu-referencing M-step for SAEM and IMP/IMPMAP
//! (#619).
//!
//! A typical value that reads several thetas — `CL = (TVCL + (CRCL-90)*TH_CRCL)
//! * exp(ETA_CL)`, `CL = TVCL * (WT/70)^TH_WT * exp(ETA_CL)` — has no single
//! anchor theta, so the closed-form shift `log θ += γ·mean(η)` does not apply and
//! every theta in it used to fall to the **eta-frozen** numerical M-step. That
//! channel maximises the observation likelihood with each subject's sampled `η_i`
//! held fixed, and it is exactly the wrong coordinate system for a covariate
//! slope: once the MH sampler has let `η_i` absorb the covariate-correlated part
//! of the between-subject variation, the slope sees almost no gradient, and the
//! two drift together until the slope reaches a bound (the fluconazole renal
//! gradient of #619 landed on `0`, 480 OFV units above NONMEM).
//!
//! The fix is the mu-referencing augmentation: treat `φ_i = g(A_i(θ)) + η_i` —
//! the individual parameter on the mu scale — as the latent variable instead of
//! `η_i`. Given the E-step's `φ_i`, the complete-data likelihood in the group's
//! thetas is
//!
//! ```text
//!   Σ_i ½ (φ_i − g(A_i(θ)))ᵀ Ω⁻¹ (φ_i − g(A_i(θ)))   +   Σ_i −log p(y_i | φ_i, θ)
//! ```
//!
//! and the θ that minimises it is the M-step. Two engines, chosen per group:
//!
//! - **Exact** ([`CovariateMuGroup::solve_exact`]) when every covariate the
//!   typical value reads is constant within each subject. Then `P_i = g⁻¹(φ_i)`
//!   does not depend on θ at all, the data term is a constant, and the M-step is
//!   a small nonlinear least-squares fit of `g(A_i(θ))` to the `φ_i` — solved by
//!   Gauss–Newton with Levenberg–Marquardt damping. For `g(A_i) = log θ + c_i`
//!   this reduces to the classical `log θ += mean(η)`.
//! - **Numerical** ([`CovariateMuGroup::solve_numerical`]) when a covariate
//!   varies within a subject. `P_i(t) = g⁻¹(φ_i + g(A_i(t;θ)) − g(A_i(t₀;θ)))`
//!   keeps a θ-dependence through the within-subject ratio, so the data term is
//!   kept and the sum above is minimised by a few BOBYQA iterations over the
//!   group's thetas — still in the φ-frozen coordinates, which is what removes
//!   the drift.
//!
//! In both cases the caller re-centres each subject's eta by the realised change
//! in its mu, `η_i −= g(A_i(θ_new)) − g(A_i(θ_old))`, so `φ_i` is unchanged by
//! the M-step (the same bookkeeping the single-anchor shift does with the
//! population mean). With a block Ω the residuals of the *other* eta components
//! enter the quadratic form through the off-diagonal of Ω⁻¹; the exact engine
//! folds them into the target (`φ_i + Σ_{l≠k} (W_kl / W_kk) η_il`), the numerical
//! one evaluates the full form.
//!
//! What this is **not**: a change to `mu_refs`. An eta whose typical value also
//! matches a single-anchor pattern (the power form) keeps that `MuRef` for the
//! inner-loop centring, `suggest_start` and reporting; only the SAEM / IMP
//! M-steps prefer the group. FOCE / FOCEI / Laplace fits are byte-identical with
//! and without a group.

use crate::parser::model_parser::eval_typical_value;
use crate::types::{CompiledModel, CovariateMuRef, MuTransform, Population, Subject};
use nalgebra::{DMatrix, DVector};

/// An IIV variance below this carries no between-subject information about the
/// group's thetas: the prior term pins `φ_i ≈ g(A_i(θ_old))` and the exact
/// engine would return `θ_old` forever (the #411 freeze). Same threshold as the
/// single-anchor guard in `run_mcem`.
pub(crate) const WEAK_GROUP_IIV_VAR: f64 = 1e-3;

/// Levenberg–Marquardt iteration cap for [`CovariateMuGroup::solve_exact`].
const EXACT_MAX_ITER: usize = 40;

/// One covariate mu-reference resolved against the model and the data.
pub(crate) struct CovariateMuGroup<'m> {
    /// Index into `model.eta_names`.
    pub eta_idx: usize,
    /// Indices into `model.theta_names`, ascending.
    pub theta_idx: Vec<usize>,
    pub transform: MuTransform,
    /// Whether any covariate the typical value reads changes within a subject.
    /// Decides the engine: exact NLS when `false`, prior + data when `true`.
    pub time_varying: bool,
    spec: &'m CovariateMuRef,
}

/// What the estimator hands a group step: the current point, its bounds, the
/// current Ω, and one eta vector per subject (SAEM's draw, or IMP's posterior
/// mean — the prior term is quadratic, so the mean is sufficient).
pub(crate) struct GroupStepInput<'a> {
    /// Full natural-scale theta at the start of the step.
    pub theta: &'a [f64],
    pub theta_lower: &'a [f64],
    pub theta_upper: &'a [f64],
    pub theta_fixed: &'a [bool],
    /// Per-theta packing, `true` = log (see `theta_packs_log`). Only the
    /// numerical engine optimises in packed coordinates.
    pub theta_packs_log: &'a [bool],
    pub omega: &'a DMatrix<f64>,
    pub etas: &'a [Vec<f64>],
}

/// Resolve `model.covariate_mu_refs` into the groups the M-step will run, and
/// say why any was left out.
///
/// A group is dropped — its thetas then stay on the numerical M-step exactly as
/// before #619 — when:
///
/// - a theta of the group is also the anchor of a single-anchor pair on a
///   **different** eta (`plain_pairs`), or of an earlier group: two closed-form
///   channels moving one theta in the same iteration has no joint optimum;
/// - the eta's initial IIV variance is below [`WEAK_GROUP_IIV_VAR`];
/// - every theta of the group is `FIX`ed (nothing to move).
///
/// The single-anchor pair on the group's **own** eta is not a conflict: it is
/// the same eta, and the caller drops that pair in favour of the group.
pub(crate) fn resolve_covariate_mu_groups<'m>(
    model: &'m CompiledModel,
    population: &Population,
    plain_pairs: &[(usize, usize)],
    theta_fixed: &[bool],
    omega: &DMatrix<f64>,
) -> (Vec<CovariateMuGroup<'m>>, Vec<String>) {
    let mut groups: Vec<CovariateMuGroup<'m>> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut claimed: Vec<usize> = Vec::new();
    for spec in &model.covariate_mu_refs {
        let Some(eta_idx) = model.eta_names.iter().position(|n| n == &spec.eta_name) else {
            continue;
        };
        let theta_idx: Option<Vec<usize>> = spec
            .theta_names
            .iter()
            .map(|n| model.theta_names.iter().position(|t| t == n))
            .collect();
        let Some(theta_idx) = theta_idx else {
            continue;
        };
        let names = spec.theta_names.join(", ");
        if theta_idx
            .iter()
            .all(|&t| theta_fixed.get(t).copied().unwrap_or(false))
        {
            continue;
        }
        let shared_with_pair = plain_pairs
            .iter()
            .filter(|&&(_t, e)| e != eta_idx)
            .find(|&&(t, _e)| theta_idx.contains(&t));
        if let Some(&(t, e)) = shared_with_pair {
            notes.push(format!(
                "covariate mu-reference on {} (typical value of {} reads {}) is not used: {} is \
                 also the mu-reference anchor of {}, and one theta cannot take two closed-form \
                 updates in the same iteration; {} stay on the numerical M-step (#619).",
                spec.eta_name,
                spec.eta_name.trim_start_matches("ETA_"),
                names,
                model.theta_names.get(t).map(String::as_str).unwrap_or("?"),
                model.eta_names.get(e).map(String::as_str).unwrap_or("?"),
                names
            ));
            continue;
        }
        if let Some(&t) = theta_idx.iter().find(|t| claimed.contains(t)) {
            notes.push(format!(
                "covariate mu-reference on {} (reads {}) is not used: {} already belongs to \
                 another covariate mu-reference; {} stay on the numerical M-step (#619).",
                spec.eta_name,
                names,
                model.theta_names.get(t).map(String::as_str).unwrap_or("?"),
                names
            ));
            continue;
        }
        let var = omega
            .get((eta_idx, eta_idx))
            .copied()
            .unwrap_or(f64::INFINITY);
        if var < WEAK_GROUP_IIV_VAR {
            notes.push(format!(
                "covariate mu-reference on {} (reads {}) is not used: its random effect has \
                 negligible variance (ω² < {WEAK_GROUP_IIV_VAR:.0e}), so the population of \
                 individual values carries no information about the typical value; {} stay on \
                 the numerical M-step (#619).",
                spec.eta_name, names, names
            ));
            continue;
        }
        let time_varying = spec.covariate_names.iter().any(|c| {
            population
                .subjects
                .iter()
                .any(|s| covariate_varies_within(s, c))
        });
        claimed.extend(theta_idx.iter().copied());
        groups.push(CovariateMuGroup {
            eta_idx,
            theta_idx,
            transform: spec.transform,
            time_varying,
            spec,
        });
    }
    (groups, notes)
}

/// Whether covariate `name` takes more than one value across a subject's record
/// snapshots. `subject.covariates` is the first non-missing value; the per-event
/// snapshots are empty when the dataset has no time-varying covariates at all.
fn covariate_varies_within(subject: &Subject, name: &str) -> bool {
    let base = subject.covariates.get(name).copied();
    let differs = |v: Option<f64>| match (base, v) {
        (Some(b), Some(x)) => !(b == x || (b.is_nan() && x.is_nan())),
        (None, Some(x)) => !x.is_nan(),
        _ => false,
    };
    subject
        .dose_covariates
        .iter()
        .chain(subject.obs_covariates.iter())
        .chain(subject.pk_only_covariates.iter())
        .chain(subject.reset_covariates.iter())
        .any(|m| differs(m.get(name).copied()))
}

impl CovariateMuGroup<'_> {
    /// Thetas of the group the step may move.
    fn free_thetas(&self, theta_fixed: &[bool]) -> Vec<usize> {
        self.theta_idx
            .iter()
            .copied()
            .filter(|&t| !theta_fixed.get(t).copied().unwrap_or(false))
            .collect()
    }

    /// `g(A_i(θ))` for one subject: `log A_i` under the lognormal link, `A_i`
    /// itself under the logit link (the typical value is already on the logit
    /// scale). `NaN` when the lognormal typical value is not positive — the
    /// caller treats a `NaN` mu as "this θ is not admissible".
    pub fn mu(&self, theta: &[f64], subject: &Subject) -> f64 {
        let a = eval_typical_value(&self.spec.typical, theta, &subject.covariates);
        match self.transform {
            MuTransform::Log => {
                if a.is_finite() && a > 0.0 {
                    a.ln()
                } else {
                    f64::NAN
                }
            }
            MuTransform::Logit | MuTransform::Identity | MuTransform::LogitProbability => a,
        }
    }

    /// [`Self::mu`] over the population.
    pub fn mus(&self, theta: &[f64], population: &Population) -> Vec<f64> {
        population
            .subjects
            .iter()
            .map(|s| self.mu(theta, s))
            .collect()
    }

    /// Row `k` of Ω⁻¹ divided by its diagonal, or `None` when Ω is not
    /// invertible / the eta is uncorrelated. Used to fold the other components'
    /// residuals into the exact engine's target.
    fn cross_weights(&self, omega: &DMatrix<f64>) -> Option<Vec<f64>> {
        let n = omega.nrows();
        let k = self.eta_idx;
        if k >= n {
            return None;
        }
        // Diagonal Ω: no cross terms, skip the inverse.
        let has_off_diag = (0..n).any(|l| l != k && omega[(k, l)] != 0.0);
        if !has_off_diag {
            return None;
        }
        let w = omega.clone().try_inverse()?;
        let wkk = w[(k, k)];
        if !(wkk.is_finite() && wkk > 0.0) {
            return None;
        }
        Some(
            (0..n)
                .map(|l| if l == k { 0.0 } else { w[(k, l)] / wkk })
                .collect(),
        )
    }

    /// Exact M-step for a time-constant group: the θ minimising
    /// `Σ_i (t_i − g(A_i(θ)))²`, `t_i = g(A_i(θ_old)) + η_ik + Σ_{l≠k} (W_kl/W_kk) η_il`,
    /// by Gauss–Newton with LM damping, started at `θ_old` and kept inside the
    /// bounds. Returns the **full** natural theta vector with the group's free
    /// entries replaced, or `None` when the group has no free theta or the
    /// starting point already has an inadmissible typical value.
    pub fn solve_exact(
        &self,
        population: &Population,
        input: &GroupStepInput<'_>,
    ) -> Option<Vec<f64>> {
        let free = self.free_thetas(input.theta_fixed);
        if free.is_empty() {
            return None;
        }
        let n = population.subjects.len();
        let mu_old = self.mus(input.theta, population);
        if mu_old.iter().any(|m| !m.is_finite()) {
            return None;
        }
        let cross = self.cross_weights(input.omega);
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let eta_i = &input.etas[i];
                let mut t = mu_old[i] + eta_i.get(self.eta_idx).copied().unwrap_or(0.0);
                if let Some(cw) = &cross {
                    for (l, &c) in cw.iter().enumerate() {
                        if c != 0.0 {
                            t += c * eta_i.get(l).copied().unwrap_or(0.0);
                        }
                    }
                }
                t
            })
            .collect();

        let residuals = |theta: &[f64]| -> Option<(Vec<f64>, f64)> {
            let mut r = Vec::with_capacity(n);
            let mut ss = 0.0;
            for (i, s) in population.subjects.iter().enumerate() {
                let m = self.mu(theta, s);
                if !m.is_finite() {
                    return None;
                }
                let ri = target[i] - m;
                ss += ri * ri;
                r.push(ri);
            }
            Some((r, ss))
        };
        let clamp = |t: usize, v: f64| -> f64 {
            let lo = input
                .theta_lower
                .get(t)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let hi = input.theta_upper.get(t).copied().unwrap_or(f64::INFINITY);
            v.clamp(lo, hi)
        };

        let mut theta = input.theta.to_vec();
        let (mut r, mut ss) = residuals(&theta)?;
        let d = free.len();
        let mut lambda = 1e-3;
        for _ in 0..EXACT_MAX_ITER {
            // Jacobian of the residual, central differences per free theta.
            let mut jac = DMatrix::<f64>::zeros(n, d);
            for (j, &t) in free.iter().enumerate() {
                let h = 1e-6 * theta[t].abs().max(1.0);
                let mut up = theta.clone();
                up[t] += h;
                let mut dn = theta.clone();
                dn[t] -= h;
                let (Some((ru, _)), Some((rd, _))) = (residuals(&up), residuals(&dn)) else {
                    return Some(theta);
                };
                for i in 0..n {
                    jac[(i, j)] = (ru[i] - rd[i]) / (2.0 * h);
                }
            }
            let rv = DVector::from_vec(r.clone());
            let jtj = jac.transpose() * &jac;
            let g = jac.transpose() * &rv;
            let mut accepted = false;
            for _ in 0..12 {
                let mut a = jtj.clone();
                for j in 0..d {
                    a[(j, j)] += lambda * jtj[(j, j)].max(1e-12);
                }
                let Some(delta) = a.clone().cholesky().map(|c| c.solve(&(-&g))) else {
                    lambda *= 10.0;
                    continue;
                };
                let mut trial = theta.clone();
                for (j, &t) in free.iter().enumerate() {
                    trial[t] = clamp(t, theta[t] + delta[j]);
                }
                match residuals(&trial) {
                    Some((rt, sst)) if sst <= ss => {
                        let step: f64 = free
                            .iter()
                            .map(|&t| (trial[t] - theta[t]).abs() / theta[t].abs().max(1.0))
                            .fold(0.0, f64::max);
                        theta = trial;
                        r = rt;
                        ss = sst;
                        lambda = (lambda * 0.3).max(1e-12);
                        accepted = true;
                        if step < 1e-9 {
                            return Some(theta);
                        }
                        break;
                    }
                    _ => lambda *= 10.0,
                }
            }
            if !accepted {
                break;
            }
        }
        Some(theta)
    }

    /// Numerical M-step for a time-varying group: minimise the prior quadratic
    /// form plus the data term over the group's free thetas, in packed
    /// coordinates, by BOBYQA warm-started at `θ_old`. `data_nll(theta, shift)`
    /// must return `−log p(y | φ, θ)` summed over subjects with every sample of
    /// subject `i`'s eta component `k` shifted by `shift[i]` (that is
    /// `g(A_i(θ_old)) − g(A_i(θ))`, the amount that keeps `φ_i` fixed). Returns
    /// the full natural theta vector, or `None` when nothing can move.
    pub fn solve_numerical(
        &self,
        population: &Population,
        input: &GroupStepInput<'_>,
        maxiter: u32,
        data_nll: &dyn Fn(&[f64], &[f64]) -> f64,
    ) -> Option<Vec<f64>> {
        let free = self.free_thetas(input.theta_fixed);
        if free.is_empty() {
            return None;
        }
        let n = population.subjects.len();
        let mu_old = self.mus(input.theta, population);
        if mu_old.iter().any(|m| !m.is_finite()) {
            return None;
        }
        let w = input.omega.clone().try_inverse().unwrap_or_else(|| {
            let mut d = DMatrix::zeros(input.omega.nrows(), input.omega.ncols());
            for j in 0..input.omega.nrows() {
                d[(j, j)] = 1.0 / input.omega[(j, j)].max(1e-12);
            }
            d
        });
        let k = self.eta_idx;
        let packs = |t: usize| input.theta_packs_log.get(t).copied().unwrap_or(true);
        let pack = |t: usize, v: f64| if packs(t) { v.max(1e-10).ln() } else { v };
        let unpack = |t: usize, v: f64| if packs(t) { v.exp() } else { v };
        let d = free.len();
        let x0: Vec<f64> = free.iter().map(|&t| pack(t, input.theta[t])).collect();
        let lower: Vec<f64> = free
            .iter()
            .map(|&t| pack(t, input.theta_lower.get(t).copied().unwrap_or(1e-10)))
            .collect();
        let upper: Vec<f64> = free
            .iter()
            .map(|&t| {
                let hi = input.theta_upper.get(t).copied().unwrap_or(1e9);
                if packs(t) {
                    hi.min(1e9).ln()
                } else {
                    hi
                }
            })
            .collect();

        let base_theta = input.theta.to_vec();
        let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            let mut theta = base_theta.clone();
            for (j, &t) in free.iter().enumerate() {
                theta[t] = unpack(t, xv[j]);
            }
            let mu_new = self.mus(&theta, population);
            if mu_new.iter().any(|m| !m.is_finite()) {
                return 1e20;
            }
            let shift: Vec<f64> = (0..n).map(|i| mu_old[i] - mu_new[i]).collect();
            // Prior term: ½ η'ᵀ W η' with η'_k = η_k + shift.
            let mut prior = 0.0;
            for (i, eta) in input.etas.iter().enumerate() {
                let mut e = DVector::from_column_slice(eta);
                if k < e.len() {
                    e[k] += shift[i];
                }
                prior += 0.5 * (e.transpose() * &w * &e)[(0, 0)];
            }
            let data = data_nll(&theta, &shift);
            let v = prior + data;
            if v.is_finite() {
                v
            } else {
                1e20
            }
        };
        let mut opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Bobyqa,
            d,
            obj,
            nlopt::Target::Minimize,
            (),
        );
        opt.set_lower_bounds(&lower).ok()?;
        opt.set_upper_bounds(&upper).ok()?;
        opt.set_maxeval(maxiter.max(1) * (d as u32 + 2)).ok()?;
        opt.set_ftol_rel(1e-5).ok()?;
        let mut x = x0
            .iter()
            .zip(lower.iter().zip(upper.iter()))
            .map(|(&v, (&lo, &hi))| v.clamp(lo, hi))
            .collect::<Vec<f64>>();
        match opt.optimize(&mut x) {
            Ok(_) | Err(_) => {}
        }
        let mut theta = base_theta;
        for (j, &t) in free.iter().enumerate() {
            theta[t] = unpack(t, x[j]);
        }
        Some(theta)
    }
}

#[cfg(test)]
#[path = "covariate_mu_ref_tests.rs"]
mod tests;
