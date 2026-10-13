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
//! The draw holds every `φ_i` fixed and re-centres `η_i ← φ_i − μ_i(θ_new)`, so
//! the data likelihood is untouched and only the η prior moves. That is exact
//! only when the data depend on the group's thetas through the `φ_i` alone —
//! covariates constant within a subject, no free theta read elsewhere, no eta
//! read by a second typical value — which is precisely
//! `CovariateMuGroup::needs_data_term == false`; a group that fails it (or is
//! not affine in any coordinates) is left to the random-walk block.
//!
//! **Bounds.** A draw that lands outside `[theta_lower, theta_upper]` is
//! rejected and the chain stays put. That is an exact Metropolis–Hastings step,
//! not an approximation: the proposal is the untruncated conditional, the
//! target is that conditional times the indicator of the box, so the
//! acceptance ratio is the indicator itself.

use crate::estimation::covariate_mu_ref::{resolve_covariate_mu_groups, GroupMember};
use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, Population, Subject};
use nalgebra::{Cholesky, DMatrix, DVector};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

/// Relative tolerance of the affinity check: a coordinate whose second
/// difference exceeds this (relative to the size of `μ`) is not linear on that
/// scale. Rounding in `ln θ + θ_WT·ln(WT/70)` is ~1e-15, a genuinely curved
/// typical value is O(1).
const AFFINE_TOL: f64 = 1e-8;

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
                let free: Vec<usize> = group
                    .theta_idx
                    .iter()
                    .copied()
                    .filter(|&t| is_free(t))
                    .collect();
                let Some(group_coords) =
                    affine_coords(&group.members, &free, &anchor_thetas, params, population)
                else {
                    continue;
                };
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
        design.x = design.design_matrices(&params.theta, population)?;
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
    /// affinity was checked when the design was resolved.
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

    /// The Gibbs move: draw `β`, reject it if any drawn theta leaves its
    /// bounds, otherwise set `theta` and re-centre every `η_i` so each `φ_i`
    /// is unchanged.
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
    ) -> DrawOutcome {
        let Some(theta_new) = self.sample_theta(theta, etas, omega, rng) else {
            return DrawOutcome::Skipped;
        };
        if !self
            .coords
            .iter()
            .all(|c| within_bounds(theta_new[c.theta_idx], c.theta_idx, lower, upper))
        {
            return DrawOutcome::RejectedOutOfBounds;
        }
        let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(etas.len());
        for s in &population.subjects {
            let old = self.mus(theta, s);
            let new = self.mus(&theta_new, s);
            let d: Vec<f64> = new.iter().zip(&old).map(|(n, o)| n - o).collect();
            if d.iter().any(|v| !v.is_finite()) {
                return DrawOutcome::Skipped;
            }
            deltas.push(d);
        }
        for (eta, d) in etas.iter_mut().zip(&deltas) {
            for (e, dj) in eta.iter_mut().zip(d) {
                *e -= dj;
            }
        }
        theta.copy_from_slice(&theta_new);
        DrawOutcome::Moved
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
/// affine function; this one is large enough that the affinity check sees a
/// curvature of O(1) and small enough to stay in a sane region of θ.
fn step_for(c: BetaCoord, beta: f64) -> f64 {
    if c.log {
        0.25
    } else {
        0.25 * beta.abs().max(0.1)
    }
}

/// Decide the scale of each free theta of a group so every member's `μ` is
/// affine in the coordinates, or `None` when no such choice exists.
///
/// A theta that is also an anchor is held to the log scale. Otherwise the log
/// scale is tried first for a positive theta with a non-negative lower bound
/// (a typical value), then the natural scale (an exponent, a slope). Each
/// choice is checked per coordinate by a second difference, then jointly along
/// a mixed direction to rule out cross terms (`θ_a·θ_b`).
fn affine_coords(
    members: &[GroupMember<'_>],
    free: &[usize],
    anchor_thetas: &[usize],
    params: &ModelParameters,
    population: &Population,
) -> Option<Vec<BetaCoord>> {
    let theta0 = &params.theta;
    let mus = |theta: &[f64]| -> Vec<f64> {
        population
            .subjects
            .iter()
            .flat_map(|s| members.iter().map(move |m| m.mu(theta, s)))
            .collect()
    };
    let base = mus(theta0);
    if base.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scale = base.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let affine_along = |dirs: &[(BetaCoord, f64)]| -> bool {
        let shifted = |sign: f64| {
            let mut t = theta0.clone();
            for &(c, h) in dirs {
                t[c.theta_idx] = c.to_theta(c.to_beta(theta0[c.theta_idx]) + sign * h);
            }
            mus(&t)
        };
        let (up, dn) = (shifted(1.0), shifted(-1.0));
        up.iter()
            .zip(&dn)
            .zip(&base)
            .all(|((u, d), b)| (u - 2.0 * b + d).abs() <= AFFINE_TOL * scale)
    };
    let mut coords = Vec::with_capacity(free.len());
    for &t in free {
        let v = theta0[t];
        let positive = v > 0.0 && params.theta_lower.get(t).copied().unwrap_or(0.0) >= 0.0;
        let candidates: &[bool] = if anchor_thetas.contains(&t) {
            &[true]
        } else if positive {
            &[true, false]
        } else {
            &[false]
        };
        let chosen = candidates.iter().copied().find(|&log| {
            let c = BetaCoord { theta_idx: t, log };
            affine_along(&[(c, step_for(c, c.to_beta(v)))])
        })?;
        coords.push(BetaCoord {
            theta_idx: t,
            log: chosen,
        });
    }
    // Jointly: a mixed direction catches a product of two coordinates.
    let mixed: Vec<(BetaCoord, f64)> = coords
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            let sign = if k % 2 == 0 { 1.0 } else { -0.7 };
            (c, sign * step_for(c, c.to_beta(theta0[c.theta_idx])))
        })
        .collect();
    if !affine_along(&mixed) {
        return None;
    }
    Some(coords)
}

#[cfg(test)]
#[path = "bayes_mu_design_tests.rs"]
mod tests;
