//! The conjugate mu-reference move of the Bayes sampler, extended to covariate
//! coefficients (#1626).
//!
//! Every η the move covers has an individual parameter on its mu scale,
//! `φ_ij = μ_ij(θ) + η_ij`, where `μ_ij` is
//!
//! - `ln θ_t` for a plain log mu-reference `P = θ_t·exp(η)` (an **anchor**), or
//! - `g(A_ij(θ))` for a covariate mu-reference (#619) — e.g.
//!   `ln TVCL + θ_WT·ln(WT_i/70) + θ_CRCL·ln(CRCL_i/100)` for
//!   `CL = TVCL·(WT/70)^θ_WT·(CRCL/100)^θ_CRCL·exp(η)`.
//!
//! When every `μ_ij` is **affine** in a coordinate vector `β` — `ln θ` for a
//! typical value, `θ` itself for an exponent or a linear slope — the model is a
//! linear regression of the `φ_i` on fixed covariates:
//! `φ_i = c_i + X_i β + η_i`, `η_i ~ N(0, Ω)`. With a flat prior on `β`, the full
//! conditional of `β` given `φ` and `Ω` is Gaussian:
//!
//! ```text
//!   P  = Σ_i X_iᵀ Ω⁻¹ X_i                       (precision)
//!   β | φ, Ω  ~  N(β_cur + P⁻¹ Σ_i X_iᵀ Ω⁻¹ η_i,  P⁻¹)
//! ```
//!
//! (the mean written relative to the current point, since `φ_i − c_i = X_i β_cur + η_i`).
//! With `X_i = I` this is the historical move `μ ~ N(μ_old + η̄, Ω/N)`.
//!
//! **The Gaussian is a proposal, corrected by Metropolis–Hastings on the real
//! likelihood** ([`MuDesign::draw`]). The draw re-centres
//! `η_i ← η_i − (μ_i(θ_new) − μ_i(θ_old))` to hold every design `φ_i`, then
//! accepts with the full target/proposal ratio, scored with the same per-subject
//! likelihood every other Gibbs block uses. That makes the move exact for any
//! typical value, whatever this module believes about it. When the belief is
//! right, the ratio is 1 and the move is the exact Gibbs draw. The belief is
//! that the design is affine and that the data depend on its thetas only
//! through the `φ_i`: covariates constant within a subject, no free theta read
//! elsewhere, no eta read by a second typical value
//! (`CovariateMuGroup::needs_data_term == false`).
//!
//! An earlier version trusted the belief and accepted every draw. Two review
//! rounds on #1951 found inputs where it was false:
//!
//! - an odd term, `(θ − θ₀)³`, that a numerical probe took for affine;
//! - `log(exp(θ·c))`, which the clamped DSL `log` makes piecewise;
//! - an anchor θ that also appears in its own exponent.
//!
//! Proving exactness form by form kept growing, so the correction now carries
//! correctness. The structural affinity check below only keeps the proposal
//! good, so that the acceptance stays near 1. It also decides which coefficients
//! leave the random walk.
//!
//! **Bounds.** A draw that lands outside `[theta_lower, theta_upper]` has zero
//! target and is rejected before the likelihood is evaluated; the chain stays
//! put.

use crate::estimation::covariate_mu_ref::{resolve_covariate_mu_groups, GroupMember};
use crate::parser::model_parser::{BinOp, Expression};
use crate::types::{CompiledModel, ModelParameters, MuTransform, OmegaMatrix, Population, Subject};
use nalgebra::{Cholesky, DMatrix, DVector};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

/// Rank floor of [`full_rank_columns`]: a column is kept when more than this
/// fraction of its (normalised) squared length is not explained by the columns
/// already kept. 1e-10 sits far below any covariate a dataset can identify and
/// far above rounding (~1e-15).
const RANK_TOL: f64 = 1e-10;

/// How one η's mu is computed.
enum MuRow<'m> {
    /// `P = θ·exp(η)`: `μ = ln θ_t`.
    Anchor { theta_idx: usize },
    /// A covariate mu-reference member: `μ = g(A_i(θ))`, with the indices of
    /// the thetas its typical value reads.
    Member(GroupMember<'m>, Vec<usize>),
}

impl MuRow<'_> {
    fn mu(&self, theta: &[f64], subject: &Subject) -> f64 {
        match self {
            MuRow::Anchor { theta_idx } => {
                let t = theta[*theta_idx];
                if t > 0.0 {
                    t.ln()
                } else {
                    f64::NAN
                }
            }
            MuRow::Member(m, _) => m.mu(theta, subject),
        }
    }
}

/// One coordinate of `β`: which theta, and whether `β = ln θ` (`log`) or `β = θ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BetaCoord {
    pub theta_idx: usize,
    pub log: bool,
}

impl BetaCoord {
    fn to_beta(self, theta: f64) -> f64 {
        if self.log {
            theta.ln()
        } else {
            theta
        }
    }
    fn to_theta(self, beta: f64) -> f64 {
        if self.log {
            beta.exp()
        } else {
            beta
        }
    }
}

/// The resolved design of the conjugate move: one mu row per η, the `β`
/// coordinates, and each subject's (constant) design matrix `X_i`.
pub(crate) struct MuDesign<'m> {
    rows: Vec<MuRow<'m>>,
    pub coords: Vec<BetaCoord>,
    /// `x[i]` is `n_eta × coords.len()`.
    x: Vec<DMatrix<f64>>,
}

/// What [`MuDesign::draw`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DrawOutcome {
    /// The draw was inside every bound and the chain moved.
    Moved,
    /// The draw fell outside a bound and was rejected; nothing changed.
    RejectedOutOfBounds,
    /// The Metropolis–Hastings correction rejected the draw; nothing changed.
    RejectedMetropolis,
    /// The conditional could not be formed (a non-PD Ω or precision, or a
    /// non-finite μ); nothing changed.
    Skipped,
}

impl<'m> MuDesign<'m> {
    /// Resolve the design, or `None` when some η is not covered by an anchor or
    /// by an affine covariate mu-reference (the move then does not run, and the
    /// caller falls back to the random-walk block for every theta).
    ///
    /// `allow_groups = false` restricts the design to plain anchors — the
    /// pre-#1626 move — and is what a mixture model uses (the class-switched
    /// typical values have not been derived for a group).
    pub(crate) fn build(
        model: &'m CompiledModel,
        population: &Population,
        params: &ModelParameters,
        allow_groups: bool,
    ) -> Option<Self> {
        let n_eta = model.n_eta;
        if n_eta == 0 || population.subjects.is_empty() {
            return None;
        }
        let is_free = |t: usize| !params.theta_fixed.get(t).copied().unwrap_or(false);

        // Plain log anchors on a free theta.
        let mut anchor: Vec<Option<usize>> = vec![None; n_eta];
        for (ei, ename) in model.eta_names.iter().enumerate() {
            if let Some(mr) = model.mu_refs.get(ename) {
                if mr.log_transformed() {
                    if let Some(ti) = model.theta_names.iter().position(|t| t == &mr.theta_name) {
                        if is_free(ti) {
                            anchor[ei] = Some(ti);
                        }
                    }
                }
            }
        }
        let mut rows: Vec<Option<MuRow<'m>>> = anchor
            .iter()
            .map(|a| a.map(|theta_idx| MuRow::Anchor { theta_idx }))
            .collect();
        // Coordinates the anchors fix to the log scale.
        let anchor_thetas: Vec<usize> = anchor.iter().flatten().copied().collect();
        let mut coords: Vec<BetaCoord> = Vec::new();
        for &t in &anchor_thetas {
            if !coords.iter().any(|c| c.theta_idx == t) {
                coords.push(BetaCoord {
                    theta_idx: t,
                    log: true,
                });
            }
        }

        if allow_groups {
            let plain_pairs: Vec<(usize, usize)> = anchor
                .iter()
                .enumerate()
                .filter_map(|(e, a)| a.map(|t| (t, e)))
                .collect();
            let (groups, _notes) = resolve_covariate_mu_groups(
                model,
                population,
                &plain_pairs,
                &params.theta_fixed,
                &params.omega.matrix,
            );
            for group in groups {
                if group.needs_data_term {
                    continue;
                }
                let Some(group_coords) = affine_coords(&group.members, &is_free, &params.theta)
                else {
                    continue;
                };
                // A theta already a coordinate (an anchor, or another group's)
                // must keep its scale.
                let mut merged: Coords = coords.iter().map(|c| (c.theta_idx, c.log)).collect();
                let mut ok = true;
                for c in &group_coords {
                    match merge(merged.clone(), vec![(c.theta_idx, c.log)]) {
                        Some(m) => merged = m,
                        None => ok = false,
                    }
                }
                if !ok {
                    continue;
                }
                for c in group_coords {
                    if !coords.iter().any(|k| k.theta_idx == c.theta_idx) {
                        coords.push(c);
                    }
                }
                for member in group.members {
                    let e = member.eta_idx;
                    let ts: Vec<usize> = member
                        .theta_names()
                        .iter()
                        .filter_map(|n| model.theta_names.iter().position(|t| t == n))
                        .collect();
                    rows[e] = Some(MuRow::Member(member, ts));
                }
            }
        }

        let rows: Vec<MuRow<'m>> = rows.into_iter().collect::<Option<Vec<_>>>()?;
        // Drop anchor coordinates no row reads any more (an anchor superseded
        // by its eta's group, whose theta the group did not keep).
        let mut design = MuDesign {
            rows,
            coords,
            x: Vec::new(),
        };
        design
            .coords
            .retain(|c| design_reads(&design.rows, c.theta_idx));
        if design.coords.is_empty() {
            return None;
        }
        let x = design.design_matrices(&params.theta, population)?;
        // Keep only coordinates the data can identify through the φ_i; the
        // rest go back to the random walk (`theta_indices` no longer lists
        // them), so no free theta is ever left without a move.
        let kept = full_rank_columns(&x, design.coords.len());
        if kept.is_empty() {
            return None;
        }
        design.coords = kept.iter().map(|&k| design.coords[k]).collect();
        design.x = x
            .into_iter()
            .map(|xi| xi.select_columns(kept.iter()))
            .collect();
        Some(design)
    }

    /// The thetas the move draws.
    pub(crate) fn theta_indices(&self) -> Vec<usize> {
        self.coords.iter().map(|c| c.theta_idx).collect()
    }

    /// `μ_i(θ)` for every η of one subject.
    pub(crate) fn mus(&self, theta: &[f64], subject: &Subject) -> Vec<f64> {
        self.rows.iter().map(|r| r.mu(theta, subject)).collect()
    }

    fn beta(&self, theta: &[f64]) -> Vec<f64> {
        self.coords
            .iter()
            .map(|c| c.to_beta(theta[c.theta_idx]))
            .collect()
    }

    fn theta_at(&self, theta: &[f64], beta: &[f64]) -> Vec<f64> {
        let mut out = theta.to_vec();
        for (c, &b) in self.coords.iter().zip(beta) {
            out[c.theta_idx] = c.to_theta(b);
        }
        out
    }

    /// `X_i` by central differences in `β` — exact up to rounding, because the
    /// affinity was proven from the expression when the design was resolved.
    fn design_matrices(&self, theta: &[f64], population: &Population) -> Option<Vec<DMatrix<f64>>> {
        let p = self.coords.len();
        let n_eta = self.rows.len();
        let beta = self.beta(theta);
        let mut x: Vec<DMatrix<f64>> = population
            .subjects
            .iter()
            .map(|_| DMatrix::zeros(n_eta, p))
            .collect();
        for k in 0..p {
            let h = step_for(self.coords[k], beta[k]);
            let mut bp = beta.clone();
            bp[k] += h;
            let mut bm = beta.clone();
            bm[k] -= h;
            let tp = self.theta_at(theta, &bp);
            let tm = self.theta_at(theta, &bm);
            for (i, s) in population.subjects.iter().enumerate() {
                let up = self.mus(&tp, s);
                let dn = self.mus(&tm, s);
                for j in 0..n_eta {
                    let d = (up[j] - dn[j]) / (2.0 * h);
                    if !d.is_finite() {
                        return None;
                    }
                    x[i][(j, k)] = d;
                }
            }
        }
        Some(x)
    }

    /// The full conditional of `β` given the current `η` (i.e. given `φ`) and
    /// `Ω`: returns `(mean, precision)`, both in `β` coordinates.
    pub(crate) fn conditional(
        &self,
        theta: &[f64],
        etas: &[Vec<f64>],
        omega: &OmegaMatrix,
    ) -> Option<(DVector<f64>, DMatrix<f64>)> {
        let p = self.coords.len();
        let w = Cholesky::new(omega.matrix.clone())?.inverse();
        let mut prec = DMatrix::<f64>::zeros(p, p);
        let mut grad = DVector::<f64>::zeros(p);
        for (xi, eta) in self.x.iter().zip(etas) {
            let xtw = xi.transpose() * &w;
            prec += &xtw * xi;
            grad += &xtw * DVector::from_column_slice(eta);
        }
        let chol = Cholesky::new(prec.clone())?;
        let shift = chol.solve(&grad);
        let beta = DVector::from_vec(self.beta(theta));
        let mean = beta + shift;
        if mean.iter().all(|v| v.is_finite()) {
            Some((mean, prec))
        } else {
            None
        }
    }

    /// One draw of `β` from its full conditional, as a full natural theta
    /// vector (no bound check, no re-centring).
    #[cfg(test)]
    pub(crate) fn sample_theta(
        &self,
        theta: &[f64],
        etas: &[Vec<f64>],
        omega: &OmegaMatrix,
        rng: &mut impl Rng,
    ) -> Option<Vec<f64>> {
        let (mean, prec) = self.conditional(theta, etas, omega)?;
        let l = Cholesky::new(prec)?.l();
        let z = DVector::from_iterator(
            mean.len(),
            (0..mean.len()).map(|_| rng.sample(StandardNormal)),
        );
        // v = L⁻ᵀ z has covariance (L Lᵀ)⁻¹ = P⁻¹.
        let v = l.transpose().solve_upper_triangular(&z)?;
        let beta_new: Vec<f64> = (0..mean.len()).map(|k| mean[k] + v[k]).collect();
        Some(self.theta_at(theta, &beta_new))
    }

    /// The move: propose `β'` from the Gaussian conditional, re-centre every
    /// `η_i` so its design `φ_i` is unchanged, and accept with the
    /// Metropolis–Hastings ratio on the **real** target.
    ///
    /// `subject_nlls(θ, η)` returns the per-subject negative log-likelihood at
    /// a state, in subject order; the log target of the whole state up to a
    /// constant, without the flat prior on `β`, is minus their sum — the same
    /// likelihood every other block scores. The proposal is
    /// `q(β' | s) = N(β'; m(s), P⁻¹)`, and the reverse proposal is the same
    /// formula at the proposed state, so
    ///
    /// ```text
    ///   log α = log_target(s') − log_target(s) + log q(β | s') − log q(β' | s)
    /// ```
    ///
    /// The **current** state's term is not recomputed: `nll` carries it in, and
    /// a `Moved` draw carries the proposed state's vector back out, so the
    /// caller's cached baseline stays exact without a second pass over the
    /// population. The move costs one population likelihood sweep, not two, and
    /// the caller needs no refresh after it — three sweeps per MCMC sweep down
    /// to two, with bit-identical draws, since the term dropped was a second
    /// evaluation of a state the caller had already evaluated.
    ///
    /// The re-centring is a shear `η' = η − (μ(β') − μ(β))` with unit Jacobian,
    /// and applying it from `s'` with `β` returns `s`, so this is a valid MH
    /// step for **any** typical value — exactness does not rest on the
    /// structural affinity proof (codex review of #1951, round 2). When the
    /// design is genuinely affine and the data depend on its thetas only
    /// through the `φ_i`, the two ratios cancel and `log α = 0` up to rounding:
    /// the move is the exact Gibbs draw. A draw outside a bound has zero
    /// target and is rejected before the target is evaluated.
    ///
    /// Returns the outcome and `log α` (`NaN` when no ratio was formed).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &self,
        theta: &mut [f64],
        etas: &mut [Vec<f64>],
        omega: &OmegaMatrix,
        lower: &[f64],
        upper: &[f64],
        population: &Population,
        rng: &mut impl Rng,
        nll: &mut [f64],
        subject_nlls: &mut dyn FnMut(&[f64], &[Vec<f64>]) -> Vec<f64>,
    ) -> (DrawOutcome, f64) {
        // Checked before a single RNG draw or mutation, not at the point of
        // use: `conditional` would silently `zip`-truncate a short `etas`, and
        // a short `nll` would panic in `copy_from_slice` *after* θ and η were
        // already committed — a partial move, which no `DrawOutcome` describes.
        debug_assert_eq!(
            etas.len(),
            population.subjects.len(),
            "one eta vector per subject"
        );
        debug_assert_eq!(
            nll.len(),
            population.subjects.len(),
            "one cached NLL per subject"
        );
        let Some((mean, prec)) = self.conditional(theta, etas, omega) else {
            return (DrawOutcome::Skipped, f64::NAN);
        };
        let Some(chol) = Cholesky::new(prec.clone()) else {
            return (DrawOutcome::Skipped, f64::NAN);
        };
        let l = chol.l();
        let z = DVector::from_iterator(
            mean.len(),
            (0..mean.len()).map(|_| rng.sample(StandardNormal)),
        );
        // v = L⁻ᵀ z has covariance (L Lᵀ)⁻¹ = P⁻¹.
        let Some(v) = l.transpose().solve_upper_triangular(&z) else {
            return (DrawOutcome::Skipped, f64::NAN);
        };
        let beta_old = DVector::from_vec(self.beta(theta));
        let beta_new = &mean + &v;
        let theta_new = self.theta_at(theta, beta_new.as_slice());
        if !self
            .coords
            .iter()
            .all(|c| within_bounds(theta_new[c.theta_idx], c.theta_idx, lower, upper))
        {
            return (DrawOutcome::RejectedOutOfBounds, f64::NAN);
        }
        let mut etas_new: Vec<Vec<f64>> = Vec::with_capacity(etas.len());
        for (s, eta) in population.subjects.iter().zip(etas.iter()) {
            let old = self.mus(theta, s);
            let new = self.mus(&theta_new, s);
            let e: Vec<f64> = eta
                .iter()
                .zip(new.iter().zip(&old))
                .map(|(e, (n, o))| e - (n - o))
                .collect();
            if e.iter().any(|v| !v.is_finite()) {
                return (DrawOutcome::Skipped, f64::NAN);
            }
            etas_new.push(e);
        }
        let Some((mean_rev, _)) = self.conditional(&theta_new, &etas_new, omega) else {
            return (DrawOutcome::Skipped, f64::NAN);
        };
        let quad = |x: &DVector<f64>, m: &DVector<f64>| {
            let d = x - m;
            (d.transpose() * &prec * &d)[(0, 0)]
        };
        // log q(β | s') − log q(β' | s); P and its normaliser are shared.
        let log_q_ratio = -0.5 * quad(&beta_old, &mean_rev) + 0.5 * quad(&beta_new, &mean);
        // −Σ nll(s') + Σ nll(s): the caller's cached baseline is the second
        // term, so only the proposal is evaluated here.
        let nll_new = subject_nlls(&theta_new, &etas_new);
        debug_assert_eq!(nll_new.len(), nll.len(), "one proposed NLL per subject");
        let log_alpha = nll.iter().sum::<f64>() - nll_new.iter().sum::<f64>() + log_q_ratio;
        if !log_alpha.is_finite() && log_alpha != f64::INFINITY {
            return (DrawOutcome::RejectedMetropolis, log_alpha);
        }
        if log_alpha < 0.0 && rng.random::<f64>().ln() >= log_alpha {
            return (DrawOutcome::RejectedMetropolis, log_alpha);
        }
        for (eta, e) in etas.iter_mut().zip(etas_new) {
            *eta = e;
        }
        theta.copy_from_slice(&theta_new);
        nll.copy_from_slice(&nll_new);
        (DrawOutcome::Moved, log_alpha)
    }
}

/// Whether `v` lies inside theta `t`'s declared bounds (inclusive; a missing
/// bound is unbounded).
pub(crate) fn within_bounds(v: f64, t: usize, lower: &[f64], upper: &[f64]) -> bool {
    let lo = lower.get(t).copied().unwrap_or(f64::NEG_INFINITY);
    let hi = upper.get(t).copied().unwrap_or(f64::INFINITY);
    v.is_finite() && v >= lo && v <= hi
}

fn design_reads(rows: &[MuRow<'_>], t: usize) -> bool {
    rows.iter().any(|r| match r {
        MuRow::Anchor { theta_idx } => *theta_idx == t,
        MuRow::Member(_, ts) => ts.contains(&t),
    })
}

/// A finite-difference step for one `β` coordinate. Any step is exact for an
/// affine function; this one stays in a sane region of θ.
fn step_for(c: BetaCoord, beta: f64) -> f64 {
    if c.log {
        0.25
    } else {
        0.25 * beta.abs().max(0.1)
    }
}

/// One `β` coordinate set: `(theta_idx, log)` pairs, at most one per theta.
type Coords = Vec<(usize, bool)>;

/// Union of two coordinate sets, or `None` when one theta would need both
/// scales (`TVCL * exp(TVCL * WT)` is affine in neither).
fn merge(mut a: Coords, b: Coords) -> Option<Coords> {
    for (t, log) in b {
        match a.iter().find(|&&(u, _)| u == t) {
            Some(&(_, l)) if l != log => return None,
            Some(_) => {}
            None => a.push((t, log)),
        }
    }
    Some(a)
}

/// Whether `e` is free of every free theta and of everything that varies
/// within a subject or with η: literals, covariates, FIX-ed thetas, and any
/// arithmetic or unary function of those.
fn is_const(e: &Expression, is_free: &dyn Fn(usize) -> bool) -> bool {
    match e {
        Expression::Literal(_) | Expression::Covariate(_) | Expression::CovariateIdx(_) => true,
        Expression::Theta(t) => !is_free(*t),
        Expression::BinOp(a, _, b) | Expression::Power(a, b) => {
            is_const(a, is_free) && is_const(b, is_free)
        }
        Expression::UnaryFn(_, a) => is_const(a, is_free),
        _ => false,
    }
}

/// The coordinates in which `e` itself is **affine**, by structure: a sum of
/// constant multiples of free thetas (natural scale) and of logs of
/// log-affine factors. `None` for anything else — a power or product of two
/// theta-dependent terms, a clamped function (`log`, `ln`, `sqrt`) of a
/// theta-dependent argument, a conditional, a level gather, an NN output.
fn lin_form(e: &Expression, is_free: &dyn Fn(usize) -> bool) -> Option<Coords> {
    if is_const(e, is_free) {
        return Some(Vec::new());
    }
    match e {
        Expression::Theta(t) => Some(vec![(*t, false)]),
        Expression::BinOp(a, BinOp::Add | BinOp::Sub, b) => {
            merge(lin_form(a, is_free)?, lin_form(b, is_free)?)
        }
        Expression::BinOp(a, BinOp::Mul, b) if is_const(a, is_free) => lin_form(b, is_free),
        Expression::BinOp(a, BinOp::Mul | BinOp::Div, b) if is_const(b, is_free) => {
            lin_form(a, is_free)
        }
        // No `log`/`ln` of a theta-dependent argument: the evaluator computes
        // `ln(max(x, 1e-30))`, so `log(exp(θ·c))` is `max(θ·c, ln 1e-30)` —
        // flat below the floor, not affine (codex review of #1951, round 2).
        _ => None,
    }
}

/// The coordinates in which `ln(e)` is affine, by structure: a product or
/// quotient of free thetas (log scale), constants, constant powers of such
/// factors, `exp` of an affine term, and a constant base raised to an affine
/// exponent — `TVCL * (WT/70)^THETA_WT * exp(THETA_SEX * SEX)`.
fn log_form(e: &Expression, is_free: &dyn Fn(usize) -> bool) -> Option<Coords> {
    if is_const(e, is_free) {
        return Some(Vec::new());
    }
    match e {
        Expression::Theta(t) => Some(vec![(*t, true)]),
        Expression::BinOp(a, BinOp::Mul | BinOp::Div, b) => {
            merge(log_form(a, is_free)?, log_form(b, is_free)?)
        }
        Expression::Power(base, exp) if is_const(base, is_free) => lin_form(exp, is_free),
        Expression::Power(base, exp) if is_const(exp, is_free) => log_form(base, is_free),
        Expression::UnaryFn(f, a) if f == "exp" => lin_form(a, is_free),
        // No `sqrt` either: the evaluator clamps its argument at 0.
        _ => None,
    }
}

/// The `β` coordinates of a group: the scale of each free theta its members
/// read, proven from the parsed typical values — `g(A_i(θ))` is affine in
/// them — or `None` when any member is not of a recognised affine form.
///
/// Structural, not numerical, on purpose: probing `μ` at a few points cannot
/// prove affinity, because an odd term such as `(θ − θ₀)³` cancels from every
/// symmetric second difference about `θ₀` (codex review of #1951). A form the
/// walker does not recognise stays on the random walk, which is always valid.
fn affine_coords(
    members: &[GroupMember<'_>],
    is_free: &dyn Fn(usize) -> bool,
    theta0: &[f64],
) -> Option<Vec<BetaCoord>> {
    let mut coords: Coords = Vec::new();
    for m in members {
        let form = match m.transform {
            MuTransform::Log => log_form(m.typical(), is_free)?,
            MuTransform::Logit | MuTransform::Identity | MuTransform::LogitProbability => {
                lin_form(m.typical(), is_free)?
            }
        };
        coords = merge(coords, form)?;
    }
    // `β = ln θ` needs θ > 0 at the start (the exp map keeps it there).
    if coords
        .iter()
        .any(|&(t, log)| log && !(theta0.get(t).copied().unwrap_or(f64::NAN) > 0.0))
    {
        return None;
    }
    Some(
        coords
            .into_iter()
            .map(|(theta_idx, log)| BetaCoord { theta_idx, log })
            .collect(),
    )
}

/// The coordinates, in order, that keep the stacked design `[X_1; …; X_N]` at
/// full column rank: a coordinate whose column is zero (every subject at the
/// covariate's reference value) or a combination of the columns already kept
/// (a covariate constant across subjects duplicates the typical value) is
/// left out. Its conditional precision would be singular, and adding a ridge
/// would be a prior; on the random walk it is sampled under its declared one.
///
/// Rank is judged on the column-normalised Gram matrix, by its Cholesky with a
/// pivot floor of `RANK_TOL`.
fn full_rank_columns(x: &[DMatrix<f64>], p: usize) -> Vec<usize> {
    let mut gram = DMatrix::<f64>::zeros(p, p);
    for xi in x {
        gram += xi.transpose() * xi;
    }
    let mut kept: Vec<usize> = Vec::new();
    for k in 0..p {
        if !(gram[(k, k)] > 0.0) {
            continue;
        }
        let mut trial = kept.clone();
        trial.push(k);
        let n = trial.len();
        let g = DMatrix::from_fn(n, n, |a, b| {
            let (i, j) = (trial[a], trial[b]);
            gram[(i, j)] / (gram[(i, i)] * gram[(j, j)]).sqrt()
        });
        // LDLᵀ-style check: the smallest Cholesky pivot of a correlation
        // matrix is the fraction of the new column not explained by the kept.
        let ok = Cholesky::new(g)
            .map(|c| {
                let l = c.l();
                (0..n).all(|d| l[(d, d)] * l[(d, d)] > RANK_TOL)
            })
            .unwrap_or(false);
        if ok {
            kept = trial;
        }
    }
    kept
}

#[cfg(test)]
#[path = "bayes_mu_design_tests.rs"]
mod tests;
