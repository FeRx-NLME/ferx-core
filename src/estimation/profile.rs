//! Profiled θ level blocks (#1065).
//!
//! **The rule.** A `profile`d block's levels are, at every outer evaluation, the exact
//! minimiser over the block of the same FOCE/FOCEI objective the unprofiled model would
//! minimise. The minimiser is found by Newton on the **total** gradient — the EBEs
//! re-solved at every point, so `dη̂/dβ` is in it — and never by a frozen-EBE shortcut:
//! "solve the levels at the current η̂, re-solve η̂, repeat" converges to a point 0.05 OFV
//! per study above the minimum once the prediction is nonlinear in η (the M3 probe on
//! #1065). So the OFV, the structural θ and their standard errors are those of the
//! unprofiled fit, and the levels never enter the optimizer's vector.
//!
//! Three pieces:
//!
//! - [`ProfileCtx::solve`]: the Newton solve. Its matrix is the subject objective's
//!   Hessian in the levels **with η̂, H and κ̂ held** — exactly `XᵀC̃⁻¹X`, since a profiled
//!   block enters the prediction affinely with a data-only coefficient (the parser's
//!   `E_PROFILE_NOT_LINEAR`) and the residual variance does not read it
//!   (`E_PROFILE_SCOPE`). `C̃` therefore comes from the objective's own builder
//!   (`subject_nll`), never a re-derivation — including a `correlation … by` line's dense
//!   `R` over its blocks, with kappas and nested levels (#2110), which that builder
//!   carries for the objective already. The gradient is the **total** gradient of each
//!   subject's term over its own levels ([`crate::api::active_levels`]): the adjoint's
//!   exact one wherever the production outer gradient is analytic (#2109), otherwise a
//!   central difference with the EBE re-solved at each point. On a model linear in (η, κ)
//!   the Newton step is the exact GLS solution and the solve takes one step; otherwise a
//!   few.
//! - [`optimize_profiled`]: the outer loop over the structural parameters only. Its
//!   gradient is the production outer gradient at `β̂`, with the levels held the way
//!   `FIX` holds a θ: by the envelope theorem `dO_p/dθ = ∂O/∂θ |_{β = β̂}`, because
//!   `∇_β O = 0` there.
//! - [`ProfileCtx::covariance`]: `covariance_method = r` as the FD Hessian of the
//!   **profile** objective over the structural parameters, re-profiling the levels at
//!   every stencil point — the Schur complement `A − BD⁻¹Bᵀ` of the full Hessian — and
//!   the level blocks of the full `R⁻¹` from `D = ∇²_ββ O` and `B = ∇²_θβ O`, both FD per
//!   subject over its active levels. `s` and `rsr` (#2108) are the unprofiled estimators
//!   over the same coordinates: `rsr` combines that matrix with the score cross-product,
//!   which makes its structural block the efficient-score sandwich.
//!
//! The levels are held through a **driver-local** template whose `theta_fixed` marks
//! them, never through the fit's: `n_parameters`, BIC and `fixed:` in the output read the
//! fit's template, where the levels are estimated parameters (#2103 review #9).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::estimation::covariance::{fd_ofv_hessian, CovStepOutcome};
use crate::estimation::inner_optimizer::{
    find_ebe_for_stage, run_inner_loop_warm_map, InnerHessianSeed, InnerLoopStats, InnerSolvePolicy,
};
use crate::estimation::outer_optimizer::{
    bobyqa_initial_step, bobyqa_stop_needs_polish, drive_polish_legs, ebe_guard_rejects,
    flat_theta_candidates, flat_theta_probe_moves, flat_theta_probe_value, flat_theta_warning,
    joint_warm_start, joint_warm_starts, max_scaled_deviation, population_gradient,
    resolve_outer_ftol, runs_outer_optimizer, subject_nll, OuterFdDeclineLog, OuterResult,
    OuterTrial, PolishEnd, INIT_ESCAPE_STEP_S, MAX_POLISH_LEGS, POLISH_FTOL_REL, POLISH_XTOL_REL,
};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_mu_k, fd_step, fd_step_floor, fd_step_floors, pack_params,
    pack_with_bounds, packed_fixed_mask, theta_packs_log, unpack_params, PackedBounds, PackedStart,
};
use crate::parser::model_parser::LevelRule;
use crate::types::{
    CompiledModel, CovarianceMethod, FitOptions, ModelParameters, Optimizer, Population,
};

/// Newton steps a profile solve may take before the evaluation is declared failed. A
/// model linear in (η, κ) takes one; the nonlinear F2 fixture of #1065 takes three.
pub(crate) const MAX_PROFILE_ITER: usize = 50;

/// Stop when every level's total gradient, in units of its own conditional standard
/// error (`|g_k| / sqrt(M_kk)`, the distance to the optimum measured in SEs), is below
/// this. Dimensionless, so it means the same on every data scale.
///
/// Set from the measured floor of the FD gradient, not below it. Forcing the solve past
/// convergence on F1 (#1065) and watching the criterion at the optimum, it wanders between
/// 3e-8 and 9.6e-7 from step to step (the re-solved EBEs' noise at
/// [`PROFILE_INNER_TOL`]); a tolerance below that makes the loop spin to
/// [`MAX_PROFILE_ITER`]. `1e-5` is 10× the worst. At that residual the OFV is off by
/// ~`½·L·1e-10` and the envelope gradient by a first-order term measured at 3e-5 relative
/// (`profile_tests`, G1).
pub(crate) const PROFILE_TOL: f64 = 1e-5;

/// The inner EBE tolerance the profile solve uses when the fit's `inner_tol` is looser.
/// The total gradient differences the re-solved objective at a step of `FD_STEP_SE`
/// standard errors; a warm-started inner solve that already met a loose tolerance would
/// barely move η̂, and the difference would drift toward the frozen-EBE partial the
/// profile exists to avoid. Measured on F1: at `1e-9` 37% of the re-solves ended in the
/// Nelder–Mead fallback (the BFGS stop is unreachable there), which was most of the
/// solve's time; at `1e-7` 0.4% did, with the same gradient floor. It is the stencil
/// tolerance every other reconverged stencil now shares (`STENCIL_INNER_TOL`, #2261).
pub(crate) const PROFILE_INNER_TOL: f64 = crate::estimation::inner_optimizer::STENCIL_INNER_TOL;

/// The central-difference step of the total gradient, in conditional standard errors.
const FD_STEP_SE: f64 = 1e-2;

/// The FD step of the covariance step's `D` and `B` off the exact route (#2109), in
/// conditional standard errors for a level and the global `fd_hessian_step · (1 + |x|)` for
/// a structural coordinate. `B` keeps the global step, not [`profile_fd_hessian_step`]: at
/// `3e-3` C1's `Cov(θ, β)` gap is 1.7e-3, at `1e-2` 9.3e-4 (macOS arm64), with `A_p` on the
/// stencil too. A mixed θ × level difference whose level side steps in SEs has its own error
/// balance. Against the exact `B` alone it is not small: 8.6e-3 relative at `1e-2` on the
/// global fixture, falling as the step squared (`level_blocks_over_levels_are_the_theta_hessian`).
const COV_STEP_SE: f64 = 1e-1;

/// The relative step of a profiled fit's structural (`A_p`) covariance stencil when
/// `fd_hessian_step` is left at its default.
///
/// The profile objective's FD Hessian is ~10× further from the exact one than the
/// unprofiled stencil at the same step. Against the unprofiled step's analytic R-matrix,
/// which is exact and reads no step, the worst profiled `r` SE gap at `inner_tol = 1e-8` is
/// 7.5e-3 at the global default `1e-2` on F2 (`profile_mbma`, nonlinear in η), and 3.9e-3 /
/// 4.5e-3 / 5.2e-3 on F1, car1 and the nested fixture. The unprofiled FD R is 8e-4 off at
/// `1e-2` on F2. At `3e-3` the four are 7.4e-4, 3.5e-4, 4.0e-4 and 5.1e-4: O(h²).
///
/// Below this the re-solved objectives' noise over `h²` takes over, and where depends on the
/// inner tolerance. At `inner_tol = 1e-8` the nested fixture is 2.3e-4 at `2e-3`, 8.6e-3 at
/// `1e-3` and 7.0e-2 at `5e-4`. At the default `inner_tol`, which leaves the stencil's
/// re-solves at `STENCIL_INNER_TOL = 1e-7`, the floor sits higher: 5.2e-3 at `1e-2`, 1.3e-3 at
/// `3e-3` and already 2.5e-3 at `2e-3`. That is #2414 review row 1, macOS arm64. So `3e-3`
/// is the better of the two at the default and only ~4× better than `1e-2` there. An
/// explicit `fd_hessian_step` is used as written (`FitOptions::fd_hessian_step_set`).
pub(crate) const PROFILE_FD_HESSIAN_STEP: f64 = 3e-3;

/// The relative step of a profiled fit's `A_p` stencil: the user's `fd_hessian_step` when
/// they set one, else [`PROFILE_FD_HESSIAN_STEP`]. `B` keeps the global step
/// ([`COV_STEP_SE`]).
pub(crate) fn profile_fd_hessian_step(options: &FitOptions) -> f64 {
    if options.fd_hessian_step_set {
        options.fd_hessian_step
    } else {
        PROFILE_FD_HESSIAN_STEP
    }
}

/// The relative noise of a subject's re-solved objective, below which a backtracking
/// step is not asked to show a decrease. On F1 (#1065) the objective of a group at its
/// optimum wanders by ~1e-9 between re-solves at [`PROFILE_INNER_TOL`]; a 1e-12 floor
/// rejected every step from a point within noise of the optimum.
const OFV_NOISE: f64 = 1e-9;

/// Halvings of a Newton step before the solve gives up.
const MAX_BACKTRACK: usize = 30;

/// The solved coordinates and how the subjects share them.
///
/// **Two spaces.** The solve's unknowns are the θ in `coords`, but each subject is
/// differenced over the **levels** it reads (#2135). The two differ only at a
/// sum-to-zero dependent level, `−(sum of the group's other levels)`: in θ a subject that
/// reads it reads every θ of the group, which under a global `sum_to_zero` is the whole
/// block, so its Newton matrix took ~k²/2 objective evaluations and its gradient 2k EBE
/// re-solves at k = 799 (S1, 100 × 8). Over levels it reads its own 8, the dependent one
/// among them. The dependent becomes a coordinate of its own (a [`Border`]) and the
/// group's levels are tied by one linear constraint, `Σ y = c`, whose multiplier the
/// Newton step solves for ([`bordered_step`]).
///
/// A subject's term is a function of its own levels only, so it is evaluated off the
/// constraint by moving a **slack**: a θ of the dependent's group the subject does not
/// read. Shifting the dependent by `d` is `θ_slack −= d`; shifting a level it reads by
/// `d` is that θ `+= d` and `θ_slack −= d`, so the dependent stays put. Off the
/// constraint the subject's term is then an extension of the objective that agrees with
/// it on the constraint, which is all a linearly constrained Newton step reads: on the
/// constraint `y = Tθ + c`, the θ-space gradient and Hessian are `Tᵀg` and `TᵀMT`
/// ([`ProfileGroup::theta_gradient`], [`ProfileGroup::theta_matrix`]), exactly the ones
/// the θ-space differences gave.
///
/// A subject that reads every θ of a dependent's group has no slack. It keeps the θ
/// form: its levels are the group's θ, and the dependent is read through them.
#[derive(Debug, Clone)]
pub(crate) struct ProfileLayout {
    /// θ index of every coordinate the solve sets: each free θ of a `profile`d block,
    /// less any level ferx froze (#1891), which is solved around as an offset. Sorted.
    pub(crate) coords: Vec<usize>,
    /// Per subject, in population order, the levels it is differenced over, sorted by
    /// [`LevelRead::level`].
    pub(crate) reads: Vec<Vec<LevelRead>>,
    /// Coordinates no two subjects couple fall into independent groups, each solved on
    /// its own: a study's levels under `sum_to_zero_within`, the whole block under a
    /// global `sum_to_zero`, whose dependent level ties every level.
    pub(crate) groups: Vec<ProfileGroup>,
    /// The dependent levels some subject reads through a slack.
    pub(crate) borders: Vec<Border>,
    /// `(group, index within the group)` per position of `coords`.
    local: Vec<(usize, usize)>,
    /// Per position of `coords`, the box its block declares (#2112), `±∞` when the block
    /// declares none. A `NegSum` dependent has no θ and no bound.
    pub(crate) lower: Vec<f64>,
    pub(crate) upper: Vec<f64>,
}

/// One level a subject's term is differenced over.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LevelRead {
    /// The layout's level index: a position of [`ProfileLayout::coords`] below
    /// `coords.len()`, else `coords.len()` plus a [`ProfileLayout::borders`] index.
    pub(crate) level: usize,
    /// The θ moves, `(θ index, coefficient)`, that shift this level by one unit and leave
    /// every other level the subject reads where it was.
    pub(crate) moves: Vec<(usize, f64)>,
}

/// A sum-to-zero dependent level solved as a coordinate of its own, tied to the free
/// levels it sums by one bordered constraint.
#[derive(Debug, Clone)]
pub(crate) struct Border {
    /// Its group, and its index in that group's [`ProfileGroup::borders`].
    group: usize,
    slot: usize,
}

/// One independent set of coordinates and the subjects that read them.
#[derive(Debug, Clone)]
pub(crate) struct ProfileGroup {
    /// Positions into [`ProfileLayout::coords`], sorted.
    pub(crate) coords: Vec<usize>,
    /// Subject indices, sorted.
    pub(crate) subjects: Vec<usize>,
    /// Indices into [`ProfileLayout::borders`]; `borders[j]` is the group's level
    /// `coords.len() + j`, after its θ.
    pub(crate) borders: Vec<usize>,
    /// Per entry of `coords`, the index into `borders` of the dependent that sums it.
    border_of: Vec<Option<usize>>,
}

impl ProfileGroup {
    /// The group's level count: its θ, then its borders.
    fn n_levels(&self) -> usize {
        self.coords.len() + self.borders.len()
    }

    /// `Tᵀg`: the gradient over the group's θ from one over its levels, a θ moving the
    /// dependent that sums it by `−1`.
    fn theta_gradient(&self, gy: &[f64]) -> Vec<f64> {
        let n = self.coords.len();
        (0..n)
            .map(|a| match self.border_of[a] {
                Some(j) => gy[a] - gy[n + j],
                None => gy[a],
            })
            .collect()
    }

    /// `TᵀMT`: the Hessian over the group's θ from one over its levels. Without borders
    /// it is `m` itself, bit for bit.
    fn theta_matrix(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.coords.len();
        DMatrix::from_fn(n, n, |a, b| {
            let (ja, jb) = (
                self.border_of[a].map(|j| n + j),
                self.border_of[b].map(|j| n + j),
            );
            let mut v = m[(a, b)];
            if let Some(jb) = jb {
                v -= m[(a, jb)];
            }
            if let Some(ja) = ja {
                v -= m[(ja, b)];
                if let Some(jb) = jb {
                    v += m[(ja, jb)];
                }
            }
            v
        })
    }

    /// The diagonal of [`Self::theta_matrix`], without forming it.
    fn theta_diagonal(&self, m: &DMatrix<f64>) -> Vec<f64> {
        let n = self.coords.len();
        (0..n)
            .map(|a| match self.border_of[a] {
                Some(j) => m[(a, a)] - 2.0 * m[(a, n + j)] + m[(n + j, n + j)],
                None => m[(a, a)],
            })
            .collect()
    }
}

/// The Newton step over a group's θ, `−(TᵀMT)⁻¹Tᵀg`, as the bordered system over its
/// levels: `min ½ΔᵀMΔ + gᵀΔ` subject to `AΔ = 0`, one row of `A` per border (the border
/// and every θ it sums). With `M` block diagonal over `blocks` (the levels no subject
/// couples), `Δ = −M⁻¹(g + Aᵀλ)` and `(AM⁻¹Aᵀ)λ = −AM⁻¹g`: a Cholesky per block and one
/// of the borders' `nb × nb` system, where the θ form is one dense Cholesky over the
/// whole group. `None` when a block or the border system is not positive definite (a
/// level no subject reads directly has no curvature of its own); the caller then takes
/// the θ form, which is positive definite whenever the levels are identified.
///
/// `held` marks the group's θ the step must leave where they are (#2112, a bounded
/// solve's working set): each is dropped from its block and so from every border row,
/// and its step is 0. A dependent level has no bound and is never held, so the
/// constraint still binds over the levels left free. An empty mask is the unbounded step.
fn bordered_step(
    g: &ProfileGroup,
    m: &DMatrix<f64>,
    gy: &[f64],
    blocks: &[Vec<usize>],
    held: &[bool],
) -> Option<Vec<f64>> {
    let n = g.coords.len();
    let nb = g.borders.len();
    let row_of = |l: usize| if l >= n { Some(l - n) } else { g.border_of[l] };
    // Columns: M⁻¹g, then M⁻¹Aᵀ per border. A held θ's rows stay zero.
    let mut sol = DMatrix::<f64>::zeros(g.n_levels(), 1 + nb);
    for blk in blocks {
        let kept: Vec<usize>;
        let blk = if held.iter().any(|&h| h) {
            kept = blk
                .iter()
                .copied()
                .filter(|&l| l >= n || !held[l])
                .collect();
            &kept
        } else {
            blk
        };
        let k = blk.len();
        if k == 0 {
            continue;
        }
        let chol = DMatrix::from_fn(k, k, |a, b| m[(blk[a], blk[b])]).cholesky()?;
        let mut rhs = DMatrix::zeros(k, 1 + nb);
        for (a, &l) in blk.iter().enumerate() {
            rhs[(a, 0)] = gy[l];
            if let Some(j) = row_of(l) {
                rhs[(a, 1 + j)] = 1.0;
            }
        }
        let x = chol.solve(&rhs);
        for (a, &l) in blk.iter().enumerate() {
            for c in 0..=nb {
                sol[(l, c)] = x[(a, c)];
            }
        }
    }
    // A·M⁻¹[g, Aᵀ]: each row of A sums its levels.
    let mut a_sol = DMatrix::<f64>::zeros(nb, 1 + nb);
    for l in 0..g.n_levels() {
        if let Some(j) = row_of(l) {
            for c in 0..=nb {
                a_sol[(j, c)] += sol[(l, c)];
            }
        }
    }
    let s = a_sol.columns(1, nb).into_owned();
    let lambda = -s.cholesky()?.solve(&a_sol.column(0).into_owned());
    let step: Vec<f64> = (0..n)
        .map(|a| -(sol[(a, 0)] + (0..nb).map(|j| sol[(a, 1 + j)] * lambda[j]).sum::<f64>()))
        .collect();
    step.iter().all(|v| v.is_finite()).then_some(step)
}

/// The Newton step over a group's θ with the θ in `held` left where they are: the
/// bordered system when the group has borders and it is positive definite, else the θ
/// form `−(TᵀMT)_FF⁻¹ (Tᵀg)_F` over the free θ `F`. With nothing held this is the
/// unbounded solve's step, code path for code path (#2112, B1).
fn restricted_step(
    g: &ProfileGroup,
    m: &DMatrix<f64>,
    gy: &[f64],
    blocks: &[Vec<usize>],
    held: &[bool],
) -> Result<Vec<f64>, ProfileFailure> {
    if !g.borders.is_empty() {
        if let Some(s) = bordered_step(g, m, gy, blocks, held) {
            return Ok(s);
        }
    }
    let grad = g.theta_gradient(gy);
    if !held.iter().any(|&h| h) {
        let chol = g
            .theta_matrix(m)
            .cholesky()
            .ok_or(ProfileFailure::Singular)?;
        return Ok((-chol.solve(&DVector::from_vec(grad))).data.into());
    }
    let free: Vec<usize> = (0..held.len()).filter(|&a| !held[a]).collect();
    let mut step = vec![0.0; held.len()];
    if free.is_empty() {
        return Ok(step);
    }
    let h = g.theta_matrix(m);
    let chol = DMatrix::from_fn(free.len(), free.len(), |a, b| h[(free[a], free[b])])
        .cholesky()
        .ok_or(ProfileFailure::Singular)?;
    let d = -chol.solve(&DVector::from_iterator(
        free.len(),
        free.iter().map(|&a| grad[a]),
    ));
    for (j, &a) in free.iter().enumerate() {
        step[a] = d[j];
    }
    Ok(step)
}

/// The level-space gradient of the group's quadratic model at the θ displacement `p`:
/// `gy + M·T·p`, a θ moving the dependent that sums it by `−1`.
fn model_gradient(g: &ProfileGroup, m: &DMatrix<f64>, gy: &[f64], p: &[f64]) -> Vec<f64> {
    let n = g.coords.len();
    let mut y = DVector::zeros(g.n_levels());
    for (a, &v) in p.iter().enumerate() {
        y[a] += v;
        if let Some(j) = g.border_of[a] {
            y[n + j] -= v;
        }
    }
    let my = m * y;
    gy.iter().zip(my.iter()).map(|(a, b)| a + b).collect()
}

/// A bounded Newton step (#2112): `Δ = argmin ½ΔᵀMΔ + gᵀΔ` over the group's θ subject to
/// `lo ≤ β + Δ ≤ hi`, by a primal active-set method (Nocedal–Wright Alg. 16.3).
///
/// The working set starts empty: a θ already on a bound whose step points out of the box
/// is blocked at step length 0 on the first pass, which is the same set a seeded start
/// would hold (measured: seeding it by the gradient's sign changed no result). Each pass
/// takes the equality step with the working set held
/// ([`restricted_step`]); a step that would cross a bound is cut at the first one hit,
/// and that θ joins the working set. At the minimum over the working set's subspace, the
/// θ whose multiplier has the wrong sign (in SEs, `se`) leaves — the worst one only, a
/// rule against cycling. `M` is positive definite, so this terminates; past
/// [`box_qp_max_iter`] passes it is a [`ProfileFailure::BoxQp`].
///
/// Not "clamp the unconstrained step": with a coupled `M` the other coordinates of a
/// clamped step stay at their unconstrained optimum, which is not the constrained one.
///
/// Returns the step and, per θ, the bound it ends on (`None` when it ends inside), which
/// the caller writes as the level itself so activity is a bit-exact predicate. With the
/// working set empty throughout, the step is [`restricted_step`]'s with nothing held.
#[allow(clippy::too_many_arguments)]
fn box_qp_step(
    g: &ProfileGroup,
    m: &DMatrix<f64>,
    gy: &[f64],
    blocks: &[Vec<usize>],
    beta: &[f64],
    lo: &[f64],
    hi: &[f64],
    se: &[f64],
) -> Result<(Vec<f64>, Vec<Option<f64>>), ProfileFailure> {
    let n = beta.len();
    let mut pinned: Vec<Option<f64>> = vec![None; n];
    let mut p = vec![0.0; n];
    let mut moved = false;
    for _ in 0..box_qp_max_iter(n) {
        let held: Vec<bool> = pinned.iter().map(Option::is_some).collect();
        let gp;
        let gy_p = if moved {
            gp = model_gradient(g, m, gy, &p);
            &gp
        } else {
            gy
        };
        let d = restricted_step(g, m, gy_p, blocks, &held)?;
        // The ratio test: the first bound a free θ would cross.
        let mut alpha = 1.0;
        let mut block = None;
        for k in (0..n).filter(|&k| !held[k]) {
            let x = beta[k] + p[k];
            let (t, b) = if d[k] < 0.0 && lo[k].is_finite() {
                ((lo[k] - x) / d[k], lo[k])
            } else if d[k] > 0.0 && hi[k].is_finite() {
                ((hi[k] - x) / d[k], hi[k])
            } else {
                continue;
            };
            if t < alpha {
                alpha = t.max(0.0);
                block = Some((k, b));
            }
        }
        for k in 0..n {
            p[k] += alpha * d[k];
        }
        moved = true;
        if let Some((k, b)) = block {
            p[k] = b - beta[k];
            pinned[k] = Some(b);
            continue;
        }
        // The minimum over the working set's subspace: release the worst θ whose
        // multiplier says the objective falls into the box, or stop.
        if !held.iter().any(|&h| h) {
            return Ok((p, pinned));
        }
        let r = g.theta_gradient(&model_gradient(g, m, gy, &p));
        let worst = (0..n)
            .filter_map(|k| {
                let lambda = match pinned[k] {
                    Some(b) if b == lo[k] => r[k],
                    Some(_) => -r[k],
                    None => return None,
                };
                (lambda < 0.0).then_some((k, lambda * se[k]))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        match worst {
            Some((k, _)) => pinned[k] = None,
            None => return Ok((p, pinned)),
        }
    }
    Err(ProfileFailure::BoxQp)
}

/// The levels at step length `alpha` along `step` from `beta` (#2112): a θ the full step
/// ends on a bound (`pinned`) gets the bound value itself at `alpha = 1`, since `β + (u − β)`
/// can round to one ulp below `u` and activity is tested by exact equality; every other
/// level is clamped into the box, which only absorbs rounding (the box is convex) and is
/// the identity with no box.
fn trial_levels(
    beta: &[f64],
    step: &[f64],
    pinned: &[Option<f64>],
    alpha: f64,
    lo: &[f64],
    hi: &[f64],
) -> Vec<f64> {
    beta.iter()
        .zip(step)
        .enumerate()
        .map(|(k, (b, s))| match pinned[k] {
            Some(bound) if alpha == 1.0 => bound,
            _ => (b + alpha * s).clamp(lo[k], hi[k]),
        })
        .collect()
}

/// The pass cap of [`box_qp_step`] on `n` θ: each θ can join and leave the working set
/// once on a typical run, plus slack.
fn box_qp_max_iter(n: usize) -> usize {
    2 * n + 10
}

impl ProfileLayout {
    /// The layout of `model`'s profiled blocks on `population` (level columns written),
    /// or `None` when the model profiles nothing. A level `template` FIXes is left out.
    pub(crate) fn build(
        model: &CompiledModel,
        population: &Population,
        template: &ModelParameters,
    ) -> Option<Self> {
        let blocks = &model.indiv_param_partials.theta_blocks;
        let active_levels = crate::api::active_levels(model, population);
        if active_levels.is_empty() {
            return None;
        }
        let frozen = |k: usize| template.theta_fixed.get(k).copied().unwrap_or(false);
        let n_subj = population.subjects.len();
        // Per subject: the θ it couples (every θ of a dependent it reads, which is what
        // ties a group together) and its reads before positions exist — a θ with the
        // slack of the dependent it shares, or a dependent `(block, level)` with its range
        // and slack.
        enum Raw {
            Theta(usize, Option<usize>),
            Dep((usize, usize), std::ops::Range<usize>, usize),
        }
        let mut theta_sets: Vec<Vec<usize>> = vec![Vec::new(); n_subj];
        let mut raw: Vec<Vec<Raw>> = (0..n_subj).map(|_| Vec::new()).collect();
        for al in &active_levels {
            let (di, decl) = blocks
                .decls
                .iter()
                .enumerate()
                .find(|(_, d)| d.name == al.block)?;
            for (i, levels) in al.per_subject.iter().enumerate() {
                let direct: Vec<usize> = levels
                    .iter()
                    .filter_map(|&l| match decl.spec.levels[l] {
                        LevelRule::Free(k) => Some(k as usize),
                        LevelRule::NegSum(..) => None,
                    })
                    .collect();
                // The dependents read, each with its slack (`None`: no θ of its group
                // is left unread, so the subject keeps the θ form).
                let mut deps: Vec<(std::ops::Range<usize>, Option<usize>)> = Vec::new();
                for &l in levels {
                    let LevelRule::NegSum(a, b) = decl.spec.levels[l] else {
                        continue;
                    };
                    let range = a as usize..b as usize;
                    // A reference level, or a sum of frozen levels: a constant.
                    if !range.clone().any(|k| !frozen(k)) {
                        continue;
                    }
                    theta_sets[i].extend(range.clone().filter(|&k| !frozen(k)));
                    let slack = range.clone().find(|k| !direct.contains(k));
                    match slack {
                        Some(s) => raw[i].push(Raw::Dep((di, l), range.clone(), s)),
                        None => raw[i].extend(
                            range
                                .clone()
                                .filter(|&k| !frozen(k))
                                .map(|k| Raw::Theta(k, None)),
                        ),
                    }
                    deps.push((range, slack));
                }
                for &k in direct.iter().filter(|&&k| !frozen(k)) {
                    theta_sets[i].push(k);
                    let slack = deps
                        .iter()
                        .find(|(r, _)| r.contains(&k))
                        .and_then(|(_, s)| *s);
                    raw[i].push(Raw::Theta(k, slack));
                }
            }
        }
        let mut coords: Vec<usize> = theta_sets.iter().flatten().copied().collect();
        coords.sort_unstable();
        coords.dedup();
        let position = |k: usize| coords.binary_search(&k).expect("a collected θ index");
        let active: Vec<Vec<usize>> = theta_sets
            .iter()
            .map(|s| {
                let mut p: Vec<usize> = s.iter().map(|&k| position(k)).collect();
                p.sort_unstable();
                p.dedup();
                p
            })
            .collect();

        // Union-find over positions: a subject couples every coordinate it reads, a
        // dependent's reader every θ of its group.
        let mut parent: Vec<usize> = (0..coords.len()).collect();
        union_sets(&mut parent, &active);
        let mut root_group: std::collections::HashMap<usize, usize> = Default::default();
        let mut groups: Vec<ProfileGroup> = Vec::new();
        let mut local = vec![(0, 0); coords.len()];
        for (p, slot) in local.iter_mut().enumerate() {
            let r = find_root(&mut parent, p);
            let g = *root_group.entry(r).or_insert_with(|| {
                groups.push(ProfileGroup {
                    coords: Vec::new(),
                    subjects: Vec::new(),
                    borders: Vec::new(),
                    border_of: Vec::new(),
                });
                groups.len() - 1
            });
            *slot = (g, groups[g].coords.len());
            groups[g].coords.push(p);
            groups[g].border_of.push(None);
        }
        for (i, set) in active.iter().enumerate() {
            if let Some(&p) = set.first() {
                groups[local[p].0].subjects.push(i);
            }
        }

        let n = coords.len();
        let mut borders: Vec<Border> = Vec::new();
        let mut border_id: std::collections::HashMap<(usize, usize), usize> = Default::default();
        let reads: Vec<Vec<LevelRead>> = raw
            .into_iter()
            .map(|subject| {
                let mut out: Vec<LevelRead> = subject
                    .into_iter()
                    .map(|r| match r {
                        Raw::Theta(k, slack) => LevelRead {
                            level: position(k),
                            moves: std::iter::once((k, 1.0))
                                .chain(slack.map(|s| (s, -1.0)))
                                .collect(),
                        },
                        Raw::Dep(key, range, slack) => {
                            let id = *border_id.entry(key).or_insert_with(|| {
                                let range: Vec<usize> =
                                    range.filter(|&k| !frozen(k)).map(position).collect();
                                let group = local[range[0]].0;
                                let slot = groups[group].borders.len();
                                for &p in &range {
                                    groups[group].border_of[local[p].1] = Some(slot);
                                }
                                groups[group].borders.push(borders.len());
                                borders.push(Border { group, slot });
                                borders.len() - 1
                            });
                            LevelRead {
                                level: n + id,
                                moves: vec![(slack, -1.0)],
                            }
                        }
                    })
                    .collect();
                out.sort_by_key(|r| r.level);
                out.dedup_by_key(|r| r.level);
                out
            })
            .collect();
        let mut boxes: std::collections::HashMap<usize, (f64, f64)> = Default::default();
        for d in &blocks.decls {
            let Some(b) = d.profile_box else { continue };
            for r in &d.spec.levels {
                if let LevelRule::Free(k) = *r {
                    boxes.insert(k as usize, b);
                }
            }
        }
        let (lower, upper) = coords
            .iter()
            .map(|k| {
                boxes
                    .get(k)
                    .copied()
                    .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
            })
            .unzip();
        Some(Self {
            coords,
            reads,
            groups,
            borders,
            local,
            lower,
            upper,
        })
    }

    /// The coordinates `theta` sets exactly on a bound of their block's box (#2112), as
    /// `(θ index, the bound, whether it is the upper one)`. Exact equality: the bounded
    /// solve writes the bound value itself into a level that reaches it.
    pub(crate) fn at_bound(&self, theta: &[f64]) -> Vec<(usize, f64, bool)> {
        self.coords
            .iter()
            .enumerate()
            .filter_map(|(p, &k)| {
                let v = theta[k];
                if v == self.lower[p] {
                    Some((k, v, false))
                } else if v == self.upper[p] {
                    Some((k, v, true))
                } else {
                    None
                }
            })
            .collect()
    }

    /// The group of level `u` and its index among the group's levels.
    fn level_local(&self, u: usize) -> (usize, usize) {
        match u.checked_sub(self.coords.len()) {
            None => self.local[u],
            Some(b) => {
                let b = &self.borders[b];
                (b.group, self.groups[b.group].coords.len() + b.slot)
            }
        }
    }

    /// The layout's level count: every coordinate, then every border.
    fn n_levels(&self) -> usize {
        self.coords.len() + self.borders.len()
    }

    /// The global level index of a group's local level `l`.
    fn level_of(&self, g: &ProfileGroup, l: usize) -> usize {
        match l.checked_sub(g.coords.len()) {
            None => g.coords[l],
            Some(j) => self.coords.len() + g.borders[j],
        }
    }
}

/// The union-find root of `a`.
fn find_root(parent: &mut [usize], mut a: usize) -> usize {
    while parent[a] != a {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    a
}

/// Join every member of each set (sorted indices) under one root.
fn union_sets(parent: &mut [usize], sets: &[Vec<usize>]) {
    for set in sets {
        for w in set.windows(2) {
            let (ra, rb) = (find_root(parent, w[0]), find_root(parent, w[1]));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
}

/// The blocks of `n` levels no set couples: each a sorted list, a level no set reads a
/// block of its own.
fn coupled_blocks(n: usize, sets: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    union_sets(&mut parent, sets);
    let mut by_root: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for l in 0..n {
        by_root
            .entry(find_root(&mut parent, l))
            .or_default()
            .push(l);
    }
    by_root.into_values().collect()
}

/// `params` with each `(read, shift)` applied: the reads' θ moves, scaled.
fn shifted(
    params: &ModelParameters,
    reads: &[LevelRead],
    shifts: &[(usize, f64)],
) -> ModelParameters {
    let mut p = params.clone();
    for &(j, d) in shifts {
        for &(k, c) in &reads[j].moves {
            p.theta[k] += c * d;
        }
    }
    p
}

/// Warm start for the next solve: the levels and each subject's joint mode, `η̂` then each
/// cell's κ̂ ([`joint_warm_starts`]), so an IOV subject's next solve does not restart κ at 0
/// (#2261).
#[derive(Debug, Clone)]
pub(crate) struct ProfileWarm {
    pub(crate) beta: Vec<f64>,
    pub(crate) etas: Option<Vec<DVector<f64>>>,
}

impl ProfileWarm {
    /// The template's own level values, and a cold inner start.
    pub(crate) fn from_template(layout: &ProfileLayout, template: &ModelParameters) -> Self {
        Self {
            beta: layout.coords.iter().map(|&k| template.theta[k]).collect(),
            etas: None,
        }
    }

    fn from_solve(layout: &ProfileLayout, s: &ProfileSolve) -> Self {
        Self {
            beta: layout.coords.iter().map(|&k| s.params.theta[k]).collect(),
            etas: Some(joint_warm_starts(&s.etas, &s.kappas)),
        }
    }
}

/// A converged profile solve at one structural point.
#[derive(Debug, Clone)]
pub(crate) struct ProfileSolve {
    /// The structural parameters with the solved levels written into `theta`.
    pub(crate) params: ModelParameters,
    pub(crate) etas: Vec<DVector<f64>>,
    pub(crate) h_matrices: Vec<DMatrix<f64>>,
    pub(crate) kappas: Vec<Vec<DVector<f64>>>,
    /// `2·Σ nllᵢ` at `params`, the same sum `pop_nll` takes.
    pub(crate) ofv: f64,
    /// The most Newton steps any group took.
    pub(crate) iterations: usize,
    /// The largest stopping criterion over the groups at exit (≤ [`PROFILE_TOL`]).
    pub(crate) criterion: f64,
    /// Per coordinate, `1 / sqrt(M_kk)`: the conditional standard error the step sizes
    /// are measured in.
    pub(crate) se_scale: Vec<f64>,
    /// Per level of the layout (coordinates, then borders), `1 / sqrt(M_yy)` from the
    /// Newton matrix over levels: the scale a level's own differences step in.
    pub(crate) level_scale: Vec<f64>,
    /// The EBE convergence of this evaluation, for the outer loop's guard and published
    /// counts (#2130 review #2): the population inner loop at the start levels, plus every
    /// subject whose final mode came from a re-solve that did not converge or fell back to
    /// Nelder–Mead.
    pub(crate) ebe: InnerLoopStats,
    /// The subjects (population indices, sorted) whose level gradient fell back to finite
    /// differences at some Newton step, on a solve whose gradient is the adjoint's
    /// ([`profile_gradient_is_analytic`]), less those off their mode below
    /// `min_obs_for_convergence_check` ([`excused_off_mode`], #2408). Always empty on the FD route, which is
    /// the model's own choice, not a fallback.
    pub(crate) fd_gradient_subjects: Vec<usize>,
    /// Level gradients the adjoint served in this solve, one per subject per Newton step.
    pub(crate) n_adjoint_gradients: usize,
    /// Per subject, whether its final mode may not be one ([`off_mode`]): the covariance
    /// step's exact level blocks then take the re-solved differences for it.
    pub(crate) off_mode: Vec<bool>,
    /// Per subject, whether its final mode is off one on a subject below
    /// `min_obs_for_convergence_check` ([`excused_off_mode`]): it still takes the re-solved
    /// differences, but the FD-fallback warnings do not name it (#2408).
    pub(crate) off_mode_excused: Vec<bool>,
}

/// Why a profile solve did not return a point. The outer evaluation is then
/// non-finite, the convention a failed inner solve follows: never silently accepted.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProfileFailure {
    /// [`MAX_PROFILE_ITER`] steps without meeting [`PROFILE_TOL`].
    NotConverged { criterion: f64 },
    /// The Newton matrix of a group is not positive definite: some level is not
    /// identified at this point.
    Singular,
    /// No step length decreased the objective.
    LineSearch,
    /// The objective was not finite at the start point.
    NonFinite,
    /// The bounded step's active-set QP (#2112) did not settle within its pass cap.
    BoxQp,
}

/// One subject's converged inner state.
#[derive(Debug, Clone)]
struct SubjectState {
    eta: DVector<f64>,
    h: DMatrix<f64>,
    kappas: Vec<DVector<f64>>,
    nll: f64,
    /// The re-solve that produced this mode did not converge, or fell back to
    /// Nelder–Mead. `false` for a mode from the population inner loop, whose own counts
    /// are carried separately.
    unconverged: bool,
    fallback: bool,
    /// The mode may not be one: its solve, from the population inner loop or a re-solve,
    /// did not converge or fell back to Nelder–Mead ([`off_mode`]). Its level gradient
    /// then takes the re-solved differences, never the adjoint's identity at a non-mode.
    off_mode: bool,
    /// Off its mode on a subject below `min_obs_for_convergence_check`
    /// ([`excused_off_mode`]): the re-solved differences still serve it, but it is not
    /// reported as a fallback.
    excused: bool,
}

/// Whether an inner solve's result may be off its mode, so the adjoint's mode identity
/// does not hold there (#2401 review row 2). Measured on `nested_levels_3` at
/// `inner_maxiter = 3`: the adjoint level gradient at such a state was 4.6 conditional SEs
/// from the converged one, the re-solved differences from the same state 2.4e-5.
fn off_mode(ebe: &crate::estimation::inner_optimizer::EbeResult) -> bool {
    !ebe.converged || ebe.used_fallback
}

/// Whether `subject` has fewer observations than `min_obs_for_convergence_check`, the
/// filter the inner loop's `n_unconverged` count applies (`run_inner_loop_warm_map`).
fn below_min_obs(subject: &crate::types::Subject, min_obs: u32) -> bool {
    subject.observations.len() < (min_obs as usize).max(1)
}

/// Whether an off-mode solve is excused from the FD-fallback warnings (#2408): its subject
/// has fewer observations than `min_obs_for_convergence_check`, the subjects the fit's
/// unconverged-EBE count already excuses. Such a subject keeps the re-solved level
/// gradient, the correct one at a non-mode, but is not named. A Nelder–Mead fallback is
/// excused too: outside ODE+IOV a BFGS that stops short always falls back, so excusing
/// only the bare stop would leave the excuse unreachable on every other model.
fn excused_off_mode(
    ebe: &crate::estimation::inner_optimizer::EbeResult,
    subject: &crate::types::Subject,
    min_obs: u32,
) -> bool {
    off_mode(ebe) && below_min_obs(subject, min_obs)
}

impl SubjectState {
    /// The mode as a re-solve's warm start, κ̂ included (#2261).
    fn warm(&self) -> DVector<f64> {
        joint_warm_start(&self.eta, &self.kappas)
    }
}

/// Everything a profile solve reads.
pub(crate) struct ProfileCtx<'a> {
    pub(crate) model: &'a CompiledModel,
    pub(crate) population: &'a Population,
    pub(crate) layout: &'a ProfileLayout,
    pub(crate) options: &'a FitOptions,
    /// The inner EBE tolerance of this solve: [`PROFILE_INNER_TOL`], or tighter when the
    /// covariance step asks for it.
    pub(crate) inner_tol: f64,
}

impl ProfileCtx<'_> {
    /// Subject `i`'s objective term at `params`, its EBE re-solved from `warm` (a joint
    /// mode, [`SubjectState::warm`]).
    fn resolve(&self, i: usize, params: &ModelParameters, warm: &DVector<f64>) -> SubjectState {
        let subject = &self.population.subjects[i];
        let mu_k = compute_mu_k(self.model, &params.theta, self.options.mu_referencing);
        let ebe = find_ebe_for_stage(
            self.model,
            subject,
            params,
            self.options.inner_maxiter,
            self.inner_tol,
            Some(warm.as_slice()),
            Some(&mu_k),
            0,
            self.options,
        );
        let nll = subject_nll(
            self.model,
            subject,
            params,
            &ebe.eta,
            &ebe.h_matrix,
            &ebe.kappas,
            self.options.interaction,
        );
        let min_obs = self.options.min_obs_for_convergence_check;
        SubjectState {
            off_mode: off_mode(&ebe),
            excused: excused_off_mode(&ebe, subject, min_obs),
            unconverged: !ebe.converged,
            eta: ebe.eta,
            h: ebe.h_matrix,
            kappas: ebe.kappas,
            nll,
            fallback: ebe.used_fallback,
        }
    }

    /// Subject `i`'s objective term at `params` with its EBE, H and κ̂ held at `s`, under
    /// `interaction` (the fit's own for the objective; `false` for the Newton matrix).
    fn held(&self, i: usize, params: &ModelParameters, s: &SubjectState, interaction: bool) -> f64 {
        subject_nll(
            self.model,
            &self.population.subjects[i],
            params,
            &s.eta,
            &s.h,
            &s.kappas,
            interaction,
        )
    }

    /// The Newton matrix of subject `i` over its levels ([`ProfileLayout::reads`]):
    /// `XᵀC̃⁻¹X`, the Hessian in those levels of the subject's **FOCE** term
    /// (`interaction = false`) with η̂, H and κ̂ held. That term is the Sheiner–Beal
    /// marginal `(y − f(η̂) + Hη̂)ᵀ C̃⁻¹ (·) + log|C̃|`, `C̃ = HΩHᵀ + R`, so it is quadratic
    /// in the levels and a forward second difference is exact up to rounding at any step
    /// (`1 + 2k + k(k−1)/2` evaluations, where a central stencil takes `2k²`); `steps`
    /// sets it per level.
    ///
    /// Not the fit's own term under FOCEI: that one is the conditional (Laplace) form, whose
    /// held-η̂ Hessian is `XᵀR⁻¹X` — the curvature with the random effects *not* free to
    /// absorb a level shift, larger than the total one. Measured on F1 (#1065): Newton on
    /// it converges linearly, ~1/30 per step, where `XᵀC̃⁻¹X` lands in one.
    fn held_hessian(
        &self,
        i: usize,
        params: &ModelParameters,
        s: &SubjectState,
        steps: &[f64],
    ) -> DMatrix<f64> {
        let reads = &self.layout.reads[i];
        let k = reads.len();
        let at = |moves: &[(usize, f64)]| self.held(i, &shifted(params, reads, moves), s, false);
        let f0 = at(&[]);
        let f1: Vec<f64> = (0..k).map(|a| at(&[(a, steps[a])])).collect();
        let mut m = DMatrix::zeros(k, k);
        for a in 0..k {
            let h = steps[a];
            m[(a, a)] = (at(&[(a, 2.0 * h)]) - 2.0 * f1[a] + f0) / (h * h);
            for b in 0..a {
                let g = steps[b];
                let v = (at(&[(a, h), (b, g)]) - f1[a] - f1[b] + f0) / (h * g);
                m[(a, b)] = v;
                m[(b, a)] = v;
            }
        }
        m
    }

    /// The objective the adjoint differentiates: the fit's own. Under the profile's
    /// additive-error scope the residual variance reads no η, so FOCE and FOCEI are one
    /// objective here (a `foce` arm of G2 on F2 measured the same gaps to every printed
    /// digit); the choice is kept for when that scope widens.
    fn objective(&self) -> crate::sens::ode_provider::adjoint::Objective {
        use crate::sens::ode_provider::adjoint::Objective;
        if self.options.interaction {
            Objective::Focei
        } else {
            Objective::Foce
        }
    }

    /// Whether subject `i` reads a level whose θ log-packs in `template`. The adjoint's
    /// packed derivatives are read as `∂/∂θ_k`, which holds only where θ `k` packs as
    /// itself. A level does today (its template box is open, even under #2402's bounds);
    /// one that log-packed would be off by a factor of θ, so the subject declines to the
    /// re-solved differences, which step in θ, instead (#2401 review row 1).
    fn reads_a_log_packed_level(&self, i: usize, template: &ModelParameters) -> bool {
        self.layout.reads[i]
            .iter()
            .flat_map(|r| &r.moves)
            .any(|&(k, _)| theta_packs_log(template.theta_lower[k]))
    }

    /// The exact total gradient of subject `i`'s term over its levels (#2109): one adjoint
    /// recording at the subject's mode `s`, read along each level's `moves`. `pt` is the
    /// packed point of the parameters `s` is the mode of.
    ///
    /// The adjoint's packed gradient is the total derivative `dFᵢ/dx` through
    /// `Ψ = F − wᵀ∇_η l`, so `dη̂/dβ` is in it, as it is in the re-solved differences of
    /// [`Self::total_gradient_fd`]; and a level's `moves` are a direction in θ, so the
    /// derivative along them is exactly the quantity those differences take, a slack's
    /// `−1` included. `None` where the recorder declines the subject (its scope or width,
    /// or a point where the mode identity cannot be formed).
    fn total_gradient_adjoint(
        &self,
        i: usize,
        pt: &crate::sens::ode_provider::adjoint::PackedPoint<'_>,
        s: &SubjectState,
    ) -> Option<Vec<f64>> {
        use crate::sens::ode_provider::adjoint::subject_adjoint_gradient_at;
        if self.reads_a_log_packed_level(i, pt.template) {
            return None;
        }
        let subject = &self.population.subjects[i];
        let mode = crate::estimation::covariance::stacked_mode(
            self.model,
            subject,
            &s.eta,
            Some(&s.kappas),
        )?;
        let g = subject_adjoint_gradient_at(self.model, subject, pt, &mode, self.objective())?;
        // Every θ the moves touch packs as itself (checked above), so `∂/∂x_k` is `∂/∂θ_k`.
        let out: Vec<f64> = self.layout.reads[i]
            .iter()
            .map(|r| r.moves.iter().map(|&(k, c)| c * g[k]).sum())
            .collect();
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// The total gradient of subject `i`'s term over its levels: central differences with
    /// the EBE re-solved at every point, warm-started from the subject's current mode. The
    /// route where the adjoint does not serve the solve, and its oracle.
    fn total_gradient_fd(
        &self,
        i: usize,
        params: &ModelParameters,
        s: &SubjectState,
        steps: &[f64],
    ) -> Vec<f64> {
        let reads = &self.layout.reads[i];
        let warm = s.warm();
        steps
            .iter()
            .enumerate()
            .map(|(a, &h)| {
                let at = |d: f64| {
                    self.resolve(i, &shifted(params, reads, &[(a, d)]), &warm)
                        .nll
                };
                (at(h) - at(-h)) / (2.0 * h)
            })
            .collect()
    }

    /// Solve the profiled levels at the structural point `params` (whose level values are
    /// ignored), from `warm`.
    pub(crate) fn solve(
        &self,
        params: &ModelParameters,
        warm: &ProfileWarm,
    ) -> Result<ProfileSolve, ProfileFailure> {
        let layout = self.layout;
        let mut p = params.clone();
        // A start outside a bounded block's box (#2112) is clamped into it; with no box
        // the clamp is the identity.
        for (pos, (&k, &b)) in layout.coords.iter().zip(&warm.beta).enumerate() {
            p.theta[k] = b.clamp(layout.lower[pos], layout.upper[pos]);
        }
        let mu_k = compute_mu_k(self.model, &p.theta, self.options.mu_referencing);
        // The start point's EBEs through the population inner loop, so the guarded
        // multi-start (`inner_restarts`) sees every evaluation as it does unprofiled.
        // Per subject, whether its start mode is one: the adjoint level gradient is exact only
        // there (#2401 review row 2).
        let (etas, hms, start_stats, kaps, start_off_mode) = run_inner_loop_warm_map(
            self.model,
            self.population,
            &p,
            self.options.inner_maxiter,
            self.inner_tol,
            warm.etas.as_deref(),
            Some(&mu_k),
            self.options.min_obs_for_convergence_check as usize,
            self.options.inner_restarts,
            InnerSolvePolicy {
                seed: InnerHessianSeed::for_options(self.options),
                ..InnerSolvePolicy::for_stage(self.options, self.model)
            },
            None,
            |subject, ebe| {
                (
                    off_mode(ebe),
                    excused_off_mode(ebe, subject, self.options.min_obs_for_convergence_check),
                )
            },
        );
        let mut states: Vec<SubjectState> = (0..self.population.subjects.len())
            .into_par_iter()
            .map(|i| {
                let mut s = SubjectState {
                    eta: etas[i].clone(),
                    h: hms[i].clone(),
                    kappas: kaps.get(i).cloned().unwrap_or_default(),
                    nll: 0.0,
                    unconverged: false,
                    fallback: false,
                    off_mode: start_off_mode.get(i).is_some_and(|m| m.0),
                    excused: start_off_mode.get(i).is_some_and(|m| m.1),
                };
                s.nll = self.held(i, &p, &s, self.options.interaction);
                s
            })
            .collect();
        if !states.iter().all(|s| s.nll.is_finite()) {
            return Err(ProfileFailure::NonFinite);
        }

        // Each group is independent: its subjects read only its coordinates.
        let theta0 = p.theta.clone();
        let route = GradientRoute {
            adjoint: profile_gradient_is_analytic(self.model, self.options),
            #[cfg(test)]
            force_decline: FORCE_PROFILE_GRADIENT_DECLINE.with(std::cell::Cell::get),
        };
        let outcomes: Vec<Result<GroupOutcome, ProfileFailure>> = layout
            .groups
            .par_iter()
            .map(|g| {
                let sub: Vec<SubjectState> =
                    g.subjects.iter().map(|&i| states[i].clone()).collect();
                self.solve_group(g, &p, &theta0, sub, &route)
            })
            .collect();
        let mut fd_gradient_subjects = Vec::new();
        let mut n_adjoint_gradients = 0;
        let mut iterations = 0;
        let mut criterion = 0.0_f64;
        let mut se_scale = vec![f64::NAN; layout.coords.len()];
        let mut level_scale = vec![f64::NAN; layout.n_levels()];
        for (g, out) in layout.groups.iter().zip(outcomes) {
            let out = out?;
            fd_gradient_subjects.extend(out.fd_subjects.iter().map(|&a| g.subjects[a]));
            n_adjoint_gradients += out.n_adjoint;
            iterations = iterations.max(out.iterations);
            criterion = criterion.max(out.criterion);
            for (j, &pos) in g.coords.iter().enumerate() {
                p.theta[layout.coords[pos]] = out.beta[j];
                se_scale[pos] = out.se[j];
            }
            for (l, &v) in out.level_se.iter().enumerate() {
                level_scale[layout.level_of(g, l)] = v;
            }
            for (&i, s) in g.subjects.iter().zip(out.states) {
                states[i] = s;
            }
        }
        let ofv = 2.0 * states.iter().map(|s| s.nll).sum::<f64>();
        let ebe = InnerLoopStats {
            n_unconverged: start_stats
                .n_unconverged
                .max(states.iter().filter(|s| s.unconverged).count()),
            n_fallback: start_stats.n_fallback + states.iter().filter(|s| s.fallback).count(),
            n_start_rejected: start_stats.n_start_rejected,
        };
        let (mut etas, mut h_matrices, mut kappas) = (Vec::new(), Vec::new(), Vec::new());
        let off_mode: Vec<bool> = states.iter().map(|s| s.off_mode).collect();
        let off_mode_excused: Vec<bool> = states.iter().map(|s| s.excused).collect();
        for s in states {
            etas.push(s.eta);
            h_matrices.push(s.h);
            kappas.push(s.kappas);
        }
        Ok(ProfileSolve {
            params: p,
            etas,
            h_matrices,
            kappas,
            ofv,
            iterations,
            criterion,
            se_scale,
            level_scale,
            ebe,
            fd_gradient_subjects: {
                fd_gradient_subjects.sort_unstable();
                fd_gradient_subjects
            },
            n_adjoint_gradients,
            off_mode,
            off_mode_excused,
        })
    }

    /// Newton with backtracking on one group, from the levels in `theta0`. The Newton
    /// matrix and the gradient are summed over the group's levels, each subject
    /// differenced over its own ([`ProfileLayout`]); the step is taken in θ.
    fn solve_group(
        &self,
        g: &ProfileGroup,
        params: &ModelParameters,
        theta0: &[f64],
        mut states: Vec<SubjectState>,
        route: &GradientRoute,
    ) -> Result<GroupOutcome, ProfileFailure> {
        let layout = self.layout;
        let n = g.n_levels();
        // Positions in `g.subjects` whose gradient fell back to FD at some step.
        let mut fd_subjects = std::collections::BTreeSet::new();
        let mut n_adjoint = 0;
        let mut p = params.clone();
        // The group's box (#2112), `±∞` where its block declares none. `solve` clamped the
        // start into it, so every iterate is feasible.
        let lo: Vec<f64> = g.coords.iter().map(|&c| layout.lower[c]).collect();
        let hi: Vec<f64> = g.coords.iter().map(|&c| layout.upper[c]).collect();
        let bounded = lo.iter().chain(&hi).any(|v| v.is_finite());
        let mut beta: Vec<f64> = g.coords.iter().map(|&c| theta0[layout.coords[c]]).collect();
        // Each subject's levels, as indices among the group's.
        let subj_loc: Vec<Vec<usize>> = g
            .subjects
            .iter()
            .map(|&i| {
                layout.reads[i]
                    .iter()
                    .map(|r| layout.level_local(r.level).1)
                    .collect()
            })
            .collect();
        let blocks = coupled_blocks(n, &subj_loc);
        let mut iterations = 0;
        // The step of the Newton matrix's difference: one SE once a pass has measured
        // it, a unit step before. The term is quadratic, so either is exact up to
        // rounding; the SE step only keeps the differences on the objective's own scale.
        let mut m_steps = vec![1.0; n];
        loop {
            // The Newton matrix first: its diagonal sizes the gradient's steps.
            let parts: Vec<DMatrix<f64>> = g
                .subjects
                .par_iter()
                .zip(states.par_iter())
                .zip(subj_loc.par_iter())
                .map(|((&i, s), loc)| {
                    let st: Vec<f64> = loc.iter().map(|&l| m_steps[l]).collect();
                    self.held_hessian(i, &p, s, &st)
                })
                .collect();
            let mut m = DMatrix::zeros(n, n);
            for (part, loc) in parts.iter().zip(&subj_loc) {
                for (a, &la) in loc.iter().enumerate() {
                    for (b, &lb) in loc.iter().enumerate() {
                        m[(la, lb)] += part[(a, b)];
                    }
                }
            }
            let se: Vec<f64> = g
                .theta_diagonal(&m)
                .into_iter()
                .map(|d| (d.is_finite() && d > 0.0).then(|| 1.0 / d.sqrt()))
                .collect::<Option<_>>()
                .ok_or(ProfileFailure::Singular)?;
            // A level's own scale. Without borders it is `se`; a θ no subject reads
            // directly has no curvature of its own and is never stepped.
            let level_se: Vec<f64> = (0..n)
                .map(|l| {
                    let d = m[(l, l)];
                    if d.is_finite() && d > 0.0 {
                        1.0 / d.sqrt()
                    } else {
                        1.0
                    }
                })
                .collect();
            m_steps.clone_from(&level_se);
            let steps: Vec<f64> = level_se.iter().map(|s| FD_STEP_SE * s).collect();
            // On the adjoint route, one packed point for the group: every subject is at `p`.
            let x = route.adjoint.then(|| pack_params(&p));
            let pt = x
                .as_deref()
                .and_then(|x| crate::sens::ode_provider::adjoint::PackedPoint::new(&p, x));
            // Per subject: the gradient, and whether the adjoint served it.
            let parts: Vec<(Vec<f64>, bool)> = g
                .subjects
                .par_iter()
                .zip(states.par_iter())
                .zip(subj_loc.par_iter())
                .map(|((&i, s), loc)| {
                    if let Some(pt) = pt.as_ref().filter(|_| !s.off_mode && !route.declines(i)) {
                        if let Some(gr) = self.total_gradient_adjoint(i, pt, s) {
                            return (gr, true);
                        }
                    }
                    let st: Vec<f64> = loc.iter().map(|&l| steps[l]).collect();
                    (self.total_gradient_fd(i, &p, s, &st), false)
                })
                .collect();
            if route.adjoint {
                for (a, ((_, served), s)) in parts.iter().zip(&states).enumerate() {
                    if *served {
                        n_adjoint += 1;
                    } else if !s.excused {
                        fd_subjects.insert(a);
                    }
                }
            }
            let mut grad_levels = vec![0.0; n];
            for ((part, _), loc) in parts.iter().zip(&subj_loc) {
                for (v, &l) in part.iter().zip(loc) {
                    grad_levels[l] += v;
                }
            }
            if !grad_levels.iter().all(|v| v.is_finite()) {
                return Err(ProfileFailure::NonFinite);
            }
            let grad = g.theta_gradient(&grad_levels);
            // The projected gradient (#2112): at a bound only the component pointing into
            // the box counts. Inside the box, or with no box, it is `|g_k|·se_k`.
            let criterion = grad
                .iter()
                .zip(&se)
                .enumerate()
                .map(|(k, (g, s))| {
                    if beta[k] == lo[k] {
                        (-g).max(0.0) * s
                    } else if beta[k] == hi[k] {
                        g.max(0.0) * s
                    } else {
                        (g * s).abs()
                    }
                })
                .fold(0.0_f64, f64::max);
            if criterion <= PROFILE_TOL {
                return Ok(GroupOutcome {
                    beta,
                    states,
                    iterations,
                    criterion,
                    se,
                    level_se,
                    fd_subjects: fd_subjects.into_iter().collect(),
                    n_adjoint,
                });
            }
            if iterations == MAX_PROFILE_ITER {
                return Err(ProfileFailure::NotConverged { criterion });
            }
            // Unbounded: the Newton step. Bounded: the box QP's, which is that step when
            // no bound is reached; a θ it ends on a bound gets the bound value itself.
            let (step, pinned) = if bounded {
                box_qp_step(g, &m, &grad_levels, &blocks, &beta, &lo, &hi, &se)?
            } else {
                let held = vec![false; g.coords.len()];
                let s = restricted_step(g, &m, &grad_levels, &blocks, &held)?;
                (s, vec![None; g.coords.len()])
            };
            let slope: f64 = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
            let o0: f64 = states.iter().map(|s| s.nll).sum();
            let mut alpha = 1.0;
            let mut accepted = None;
            for _ in 0..MAX_BACKTRACK {
                // The box is convex, so `β + αΔ` stays in it for `α ≤ 1`; the clamp only
                // absorbs rounding, and is the identity with no box.
                let trial = trial_levels(&beta, &step, &pinned, alpha, &lo, &hi);
                let mut q = p.clone();
                for (&c, &b) in g.coords.iter().zip(&trial) {
                    q.theta[layout.coords[c]] = b;
                }
                let next: Vec<SubjectState> = g
                    .subjects
                    .par_iter()
                    .zip(states.par_iter())
                    .map(|(&i, s)| self.resolve(i, &q, &s.warm()))
                    .collect();
                let o: f64 = next.iter().map(|s| s.nll).sum();
                // Armijo, with a floor at the objective's own noise: within it, the step
                // from a point already at the optimum cannot show a decrease.
                if o.is_finite() && o - o0 <= 1e-4 * alpha * slope + OFV_NOISE * (1.0 + o0.abs()) {
                    accepted = Some((trial, q, next));
                    break;
                }
                alpha *= 0.5;
            }
            let (trial, q, next) = accepted.ok_or(ProfileFailure::LineSearch)?;
            beta = trial;
            p = q;
            states = next;
            iterations += 1;
        }
    }
}

struct GroupOutcome {
    beta: Vec<f64>,
    states: Vec<SubjectState>,
    iterations: usize,
    criterion: f64,
    se: Vec<f64>,
    /// Per level of the group, `1 / sqrt(M_yy)`.
    level_se: Vec<f64>,
    /// Positions in the group's `subjects` whose level gradient fell back to FD.
    fd_subjects: Vec<usize>,
    /// Level gradients the adjoint served.
    n_adjoint: usize,
}

/// Where a solve's level gradients come from: the adjoint, a subject at a time with the
/// FD fallback, or FD throughout. Decided once per solve, on the caller's thread.
struct GradientRoute {
    adjoint: bool,
    /// Test switch: the subject (population index) whose adjoint gradient is declined.
    #[cfg(test)]
    force_decline: Option<usize>,
}

impl GradientRoute {
    fn declines(&self, _i: usize) -> bool {
        #[cfg(test)]
        if self.force_decline == Some(_i) {
            return true;
        }
        false
    }
}

/// Whether the profile solve's level gradient is the adjoint's (#2109): the predicate
/// the outer gradient routes on (`envelope_gradient`), so the two cannot disagree and
/// `gradient = fd` keeps the whole profile on finite differences. A subject the recorder
/// declines at runtime still takes the reconverged differences, and is reported.
pub(crate) fn profile_gradient_is_analytic(model: &CompiledModel, options: &FitOptions) -> bool {
    crate::sens::provider::analytic_outer_gradient_for_interaction(model, options.interaction)
}

/// The warning for subjects whose profile-solve level gradient fell back to finite
/// differences on the adjoint route, `during` naming the phase ("during this fit", "during
/// the covariance step"). `None` when there are none.
pub(crate) fn profile_fd_gradient_warning(
    population: &Population,
    subjects: &std::collections::BTreeSet<usize>,
    during: &str,
) -> Option<String> {
    let (count, example) = fd_fallback_subjects(population, *subjects.first()?, subjects.len());
    Some(format!(
        "{count} could not be given the exact analytic gradient of the profile solve \
         {during}{example}, and their profiled levels' gradients used reconverged finite \
         differences, which are correct but slower."
    ))
}

/// The two parts every profile FD-fallback warning opens with: `"N of M subjects"` and
/// `" (e.g. subject <id>)"` naming the `first` of the `n` subjects.
fn fd_fallback_subjects(population: &Population, first: usize, n: usize) -> (String, String) {
    let example = population
        .subjects
        .get(first)
        .map(|s| format!(" (e.g. subject {})", s.id))
        .unwrap_or_default();
    (
        format!("{n} of {} subjects", population.subjects.len()),
        example,
    )
}

/// The packed value of θ `k` at `value`, as `pack_params` would write it.
fn packed_theta(template: &ModelParameters, k: usize, value: f64) -> f64 {
    if theta_packs_log(template.theta_lower[k]) {
        value.max(1e-10).ln()
    } else {
        value
    }
}

/// The driver-local template: the fit's, with the solved levels held.
pub(crate) fn hold_template(layout: &ProfileLayout, template: &ModelParameters) -> ModelParameters {
    let mut hold = template.clone();
    for &k in &layout.coords {
        hold.theta_fixed[k] = true;
    }
    hold
}

/// [`hold_template`] undone: the fit's template, its profiled levels free θ again, as the
/// unprofiled step scores them.
fn unheld_template(layout: &ProfileLayout, hold: &ModelParameters) -> ModelParameters {
    let mut unheld = hold.clone();
    for &k in &layout.coords {
        unheld.theta_fixed[k] = false;
    }
    unheld
}

/// The packed point `x` with its level coordinates at the solve's `β̂`.
fn solved_point(
    layout: &ProfileLayout,
    hold: &ModelParameters,
    x: &[f64],
    solve: &ProfileSolve,
) -> Vec<f64> {
    let mut x_hat = x.to_vec();
    for &k in &layout.coords {
        x_hat[k] = packed_theta(hold, k, solve.params.theta[k]);
    }
    x_hat
}

/// The outer gradient at a solved point: the production gradient, the levels held at
/// `β̂` in `hold` (envelope theorem). Indexed over the whole packed vector; the held
/// coordinates' entries are not read.
///
/// `declines` should be built by [`profiled_decline_log`], as [`optimize_profiled`]'s
/// is: then a subject that declines the analytic gradient at
/// runtime is reconverged too (#2136).
pub(crate) fn envelope_gradient(
    ctx: &ProfileCtx<'_>,
    hold: &ModelParameters,
    x: &[f64],
    solve: &ProfileSolve,
    bounds: &PackedBounds,
    grad_idx: &mut usize,
    declines: &OuterFdDeclineLog,
) -> Vec<f64> {
    let x_hat = solved_point(ctx.layout, hold, x, solve);
    // Off the analytic route, every evaluation's gradient reconverges the EBEs. The FD
    // route's default is the fixed-EBE gradient, which drops `dη̂/dθ`: an approximation the
    // unprofiled fit's optimizer tolerates, but the profile's L-BFGS stopped on it 0.42
    // OFV above the optimum on S1 (#1065, 100 × 8), where the reconverged gradient reaches
    // it. The envelope theorem removes `dβ̂/dθ`, not `dη̂/dθ`. The analytic gradient is
    // exact and is left alone (the schedule would replace it), and the IOV FD route
    // reconverges already. On the analytic route a subject can still decline at runtime
    // and take the per-subject salvage; that one is keyed on the subject's own route, by
    // the reconverging `declines` log, not on this model-level predicate (#2136).
    let analytic = profile_gradient_is_analytic(ctx.model, ctx.options);
    let opts = FitOptions {
        reconverge_gradient_interval: if analytic {
            ctx.options.reconverge_gradient_interval
        } else {
            1
        },
        ..ctx.options.clone()
    };
    population_gradient(
        &x_hat,
        ctx.population.subjects.len(),
        hold,
        ctx.model,
        ctx.population,
        &solve.etas,
        &solve.h_matrices,
        &solve.kappas,
        bounds,
        &opts,
        grad_idx,
        OuterTrial::unknown(),
        declines,
    )
}

#[cfg(test)]
thread_local! {
    /// How many times [`optimize_profiled`] replaced its incumbent on this thread, so a
    /// test can tell evaluations that improved from those that did not.
    static INCUMBENT_UPDATES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Test switch: every log [`profiled_decline_log`] builds on this thread treats every
    /// subject's analytic outer gradient as declined on a **non-IOV** model (the IOV
    /// assembly does not read the forced flags), so a test can reach a runtime decline
    /// inside [`optimize_profiled`] on a fixture the adjoint serves.
    pub(crate) static FORCE_PROFILED_DECLINES: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test switch: every profile solve started on this thread declines this subject's
    /// (population index) adjoint level gradient, so it takes the FD fallback (#2109).
    pub(crate) static FORCE_PROFILE_GRADIENT_DECLINE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    /// Test switch: every covariance step started on this thread declines this subject's
    /// (population index) exact level blocks, so they take the FD fallback (#2109).
    pub(crate) static FORCE_PROFILE_COV_DECLINE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    /// Covariance steps on this thread whose `A_p` was the exact Schur complement (#2109),
    /// so a test can tell which engine a step ran on.
    pub(crate) static PROFILE_COV_SCHUR: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// The outer-gradient decline log of a profiled fit: reconverging, so a subject that
/// declines the analytic gradient at runtime takes the reconverged salvage rather than the
/// held-EBE one (#2136, see [`envelope_gradient`]). The one constructor both of
/// [`optimize_profiled`]'s gradient sites use.
pub(crate) fn profiled_decline_log(n_subjects: usize) -> OuterFdDeclineLog {
    let log = OuterFdDeclineLog::reconverging(n_subjects);
    #[cfg(test)]
    if FORCE_PROFILED_DECLINES.with(std::cell::Cell::get) {
        for i in 0..n_subjects {
            log.force_decline(i);
        }
    }
    log
}

/// The flat-θ pre-flight (#826) of a profiled fit: each free structural θ the profile
/// objective ignores at the start `x0`, with its warning. The rule is the unprofiled
/// pre-flight's (`flat_theta_candidates`, then the perturbation probe), on the objective
/// this loop minimises: the envelope gradient at the start's solve, and the probe
/// re-profiled. A probe whose solve fails counts as moving the objective, so it never
/// freezes.
fn profiled_flat_thetas(
    ctx: &ProfileCtx<'_>,
    hold: &ModelParameters,
    x0: &[f64],
    bounds: &PackedBounds,
) -> Vec<(usize, String)> {
    let n_theta = hold.theta.len();
    let is_free = |i: usize| !hold.theta_fixed[i];
    if !(0..n_theta).any(is_free) {
        return Vec::new();
    }
    let params = unpack_params(x0, hold);
    let Ok(solve) = ctx.solve(&params, &ProfileWarm::from_template(ctx.layout, hold)) else {
        return Vec::new();
    };
    // A log of its own, dropped, as the unprofiled pre-flight's: a decline here is not one
    // the optimizer's gradient took. The same goes for these solves' own level-gradient
    // fallbacks (`ProfileSolve::fd_gradient_subjects`): the fit's warning counts the
    // evaluations the optimizer ran, not the pre-flight's probes (#2401 review row 3).
    let grad = envelope_gradient(
        ctx,
        hold,
        x0,
        &solve,
        bounds,
        &mut 0,
        &profiled_decline_log(ctx.population.subjects.len()),
    );
    let Some(candidates) = flat_theta_candidates(&grad, n_theta, is_free) else {
        return Vec::new();
    };
    let warm = ProfileWarm::from_solve(ctx.layout, &solve);
    candidates
        .into_iter()
        .filter_map(|i| {
            let ti = params.theta[i];
            let probe = flat_theta_probe_value(ti, hold.theta_lower[i], hold.theta_upper[i])?;
            let mut p = params.clone();
            p.theta[i] = probe;
            let probed = ctx.solve(&p, &warm).ok()?.ofv;
            if flat_theta_probe_moves(solve.ofv, probed) {
                return None;
            }
            let name = hold.theta_names.get(i).map_or("<theta>", |s| s.as_str());
            Some((i, flat_theta_warning(name, ti)))
        })
        .collect()
}

/// The outer algorithm of a profiled fit: `auto` is L-BFGS on the envelope gradient,
/// whichever engine serves it, and the explicit NLopt choices are honoured;
/// `check_profiled_levels` refuses the rest.
///
/// Not the unprofiled rule (BOBYQA when the outer gradient is FD). That rule exists
/// because an FD gradient through a re-solved inner loop is noisy, and BOBYQA never
/// differentiates. Here the objective is a smooth, exactly re-solved profile, and its
/// gradient is checked against FD of that profile (G1, 5.5e-5 relative on the FD
/// engine). Measured on F2 (#1065): BOBYQA stopped 0.0185 OFV above NONMEM's optimum
/// after 110 evaluations, L-BFGS 1.5e-8 above it after 26 (7.9 s against 22.9 s).
///
/// `Optimizer::Bfgs` and `Optimizer::Lbfgs` — the hand-rolled loops, reachable only by
/// constructing the variant in Rust, since the parser reads `bfgs` and `lbfgs` as
/// `nlopt_lbfgs` (#483) — also run NLopt L-BFGS here, the profiled loop having no
/// hand-rolled arm (#2130 review #6). The docs say so.
fn profiled_algorithm(options: &FitOptions) -> nlopt::Algorithm {
    match options.optimizer {
        Optimizer::Bobyqa => nlopt::Algorithm::Bobyqa,
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::Auto | Optimizer::NloptLbfgs | Optimizer::Bfgs | Optimizer::Lbfgs => {
            nlopt::Algorithm::Lbfgs
        }
        // Refused by `check_profiled_levels` before a fit gets here.
        Optimizer::TrustRegion | Optimizer::Mma => nlopt::Algorithm::Lbfgs,
    }
}

/// `ftol_rel` of a profiled fit's gradient-based outer optimizer (L-BFGS, SLSQP) when
/// the user set no `outer_ftol` (#2285).
///
/// The unprofiled default, `1e-6`, is relative to the OFV, and on the F2 anchor
/// (`nonmem_anchor/profile_mbma_fit.ferx`, OFV ≈ 102) it stops L-BFGS on any step under
/// 1e-4. L-BFGS met it mid-descent: from the start (45, 7.5, 2) the trace's last steps
/// were 3.1e-4, 2.1e-5, 5.0e-5 and the run stopped `converged: true` 3.9e-4 above
/// NONMEM's optimum. #2280's start stopped 1.6e-3 above it after a 1.2e-5 step. That
/// default is set for the unprofiled objective, whose re-solved inner loop carries
/// ~1e-3 of noise. The profile objective is re-solved to [`OFV_NOISE`] (1e-9 relative),
/// so a stop it can reach sits much lower. `1e-8` is 10× that floor.
///
/// It is still a test on one iteration's change, so it does not by itself put the run
/// on the optimum: over a 27-start grid around F2's start it stopped up to 8.5e-5 short
/// (#2285's plan, M4), and #2280's start then stopped 1.6e-3 short on `xtol_rel`
/// instead (M1). Whether a stop is a minimum is decided by the verification leg
/// ([`PROFILED_VERIFY_GAIN`]), not by this tolerance.
///
/// BOBYQA keeps the unprofiled rule: it is derivative-free, and the tight stop would
/// only spend its budget.
pub(crate) const PROFILED_GRADIENT_FTOL_REL: f64 = 1e-8;

/// Ceiling on `xtol_rel` for a profiled fit's gradient-based outer optimizer (L-BFGS,
/// SLSQP), applied to every leg (#2285). NLopt's L-BFGS tests `xtol_rel` on a single
/// iteration's step, so at the unprofiled `outer_xtol = 1e-4` one short line-search
/// step ends the run: #2280's own A2 start stopped `XtolReached` 1.584e-3 above
/// NONMEM's optimum. Verification legs at that tolerance do not help either: from
/// (45, 10, 2) every leg ended on a short step and gained under
/// [`PROFILED_VERIFY_GAIN`] while still 2.5e-5 short (M9). A tighter user `outer_xtol`
/// wins, as for the #1782 polish legs. Not lower than this: at 1e-12 `main`'s L-BFGS
/// ends five of seven starts on NLopt `Failure` (M5).
pub(crate) const PROFILED_GRADIENT_XTOL_REL: f64 = 1e-6;

/// A profiled gradient-based fit is converged when a fresh L-BFGS / SLSQP leg from its
/// best point lowers the OFV by no more than this (#2285). Absolute OFV, like A2's band.
/// Measured on F2 over #2285's seven starts and a 27-start grid around them (macOS
/// arm64, #2285's plan M8/M10): a leg from a stop short of the optimum gained at least
/// 2.14e-5, and one from a genuine optimum at most 6.0e-8 (the earlier S1 fixture,
/// |OFV| 3579), so on that grid 1e-6 sat 21× below the one and 17× above the other. It
/// is not a bound beyond it: on a wider grid `main`'s start (48, 12, 3.2) took a leg
/// that gained under 1e-6 while 1.03e-5 short, and was reported converged (#2340). A
/// leg that gains more is followed by another, up to `MAX_POLISH_LEGS`, the cap the
/// #1782 BOBYQA polish uses.
///
/// The threshold is absolute while the profile solve's noise ([`OFV_NOISE`]) is
/// relative, so a large |OFV| eats into the margin. Measured on the largest fixture,
/// S1 (|OFV| 4236, Linux aarch64 at #2338's head): one leg from its optimum, gaining
/// 5.5e-9, 182× under the threshold.
pub(crate) const PROFILED_VERIFY_GAIN: f64 = 1e-6;

/// The `xtol_rel` [`optimize_profiled`] sets for `algo`: `outer_xtol` capped at
/// [`PROFILED_GRADIENT_XTOL_REL`] for a gradient method, `outer_xtol` for BOBYQA.
pub(crate) fn profiled_xtol(algo: nlopt::Algorithm, options: &FitOptions) -> f64 {
    match algo {
        nlopt::Algorithm::Bobyqa => options.outer_xtol,
        _ => options.outer_xtol.min(PROFILED_GRADIENT_XTOL_REL),
    }
}

/// One verification leg as [`optimize_profiled`] ran it, for the tests: the `xtol_rel`
/// and `ftol_rel` read back from the leg itself, the gain it reported to the loop, and the incumbent's
/// OFV before and after it.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct VerifyLeg {
    xtol: Option<f64>,
    ftol: Option<f64>,
    gain: f64,
    before: f64,
    after: f64,
}

#[cfg(test)]
thread_local! {
    /// The verification legs [`optimize_profiled`] ran on this thread, for the tests.
    static VERIFY_LEGS: std::cell::RefCell<Vec<VerifyLeg>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

type NloptReturn = Result<(nlopt::SuccessState, f64), (nlopt::FailState, f64)>;

/// Does a profiled run that returned `result` need a verification leg before it may be
/// called converged? Every stop of a gradient method that could be a minimum: any
/// tolerance or success status, and `Failure` / `RoundoffLimited`, which `main`'s L-BFGS
/// returns at the optimum under tight stops (M5). Not a spent budget or time, not a
/// cancelled run, not a bad argument.
///
/// BOBYQA's are the stops the unprofiled #1782 polish takes, `bobyqa_stop_needs_polish`:
/// its three tolerance stops (#2336). Before, a profiled BOBYQA fit took its first
/// `FtolReached` as converged, 3.46 OFV above A2's optimum on `main` at `5bd862b8`.
///
/// `Failure` / `RoundoffLimited` only once the incumbent has `left_init`: L-BFGS also
/// returns `Failure` when its first step fails, and a leg from that same start gains
/// nothing, which would report the initial estimates converged. That is the case the
/// unprofiled path's #751 guard (`failure_is_converged_plateau`) refuses; such a stop
/// keeps the status verdict, not converged. A tolerance stop needs no such guard: it
/// was a stop before #2285 too.
pub(crate) fn profiled_stop_needs_verify(
    algo: nlopt::Algorithm,
    result: &NloptReturn,
    cancelled: bool,
    left_init: bool,
) -> bool {
    if matches!(algo, nlopt::Algorithm::Bobyqa) {
        return bobyqa_stop_needs_polish(algo, result, false, cancelled);
    }
    !cancelled
        && match result {
            Ok((status, _)) => !matches!(
                status,
                nlopt::SuccessState::MaxEvalReached | nlopt::SuccessState::MaxTimeReached
            ),
            Err((fail, _)) => {
                left_init
                    && matches!(
                        fail,
                        nlopt::FailState::Failure | nlopt::FailState::RoundoffLimited
                    )
            }
        }
}

/// The NLopt status alone as a verdict: the rule for a run no leg verified (a spent
/// budget, a cancelled run).
fn status_is_converged(result: &NloptReturn) -> bool {
    matches!(
        result,
        Ok((
            nlopt::SuccessState::Success
                | nlopt::SuccessState::FtolReached
                | nlopt::SuccessState::XtolReached
                | nlopt::SuccessState::StopValReached,
            _
        ))
    )
}

/// `converged` of a profiled fit from how its verification legs ended and the last
/// leg's return. `Settled` on a verifiable stop means the last leg gained no more than
/// [`PROFILED_VERIFY_GAIN`]: converged, whatever NLopt's status, `Failure` included.
/// A leg cap, a spent budget or a leg that could not start is not converged. Otherwise
/// (nothing to verify, or the last leg ended on a budget) the status decides.
pub(crate) fn profiled_converged(
    algo: nlopt::Algorithm,
    end: PolishEnd,
    last: &NloptReturn,
    cancelled: bool,
    left_init: bool,
) -> bool {
    match end {
        PolishEnd::Settled if profiled_stop_needs_verify(algo, last, cancelled, left_init) => true,
        PolishEnd::Settled | PolishEnd::NotNeeded => status_is_converged(last),
        PolishEnd::LegCapReached { .. } | PolishEnd::BudgetSpent | PolishEnd::Skipped => false,
    }
}

/// The warning a profiled fit gets for how its verification ended, or `None` when it is
/// converged. A function rather than inline `format!`s so each message is tested where
/// it is written.
///
/// A budget that runs out *during* a leg ends the loop `Settled` on `MaxEvalReached`, not
/// `BudgetSpent` (which `drive_polish_legs` returns only when nothing is left before a
/// leg), and gets the same advice.
pub(crate) fn profiled_verdict_warning(
    end: PolishEnd,
    last: &NloptReturn,
    converged: bool,
) -> Option<String> {
    let budget = || {
        "Profiled outer optimizer spent its evaluation budget before its stop was \
         verified, so it is not reported as converged; increase maxiter."
            .to_string()
    };
    match end {
        PolishEnd::LegCapReached { last_gain } => Some(format!(
            "Profiled outer optimizer was still improving after {MAX_POLISH_LEGS} \
             verification legs (the last lowered the objective by {last_gain:.3e}), so \
             it is not reported as converged; re-run from these estimates."
        )),
        PolishEnd::BudgetSpent => Some(budget()),
        PolishEnd::Settled if matches!(last, Ok((nlopt::SuccessState::MaxEvalReached, _))) => {
            Some(budget())
        }
        _ if !converged => {
            let status = match last {
                Ok((s, _)) => format!("{s:?}"),
                Err((f, _)) => format!("{f:?}"),
            };
            Some(format!(
                "Profiled outer optimizer stopped without converging ({status})."
            ))
        }
        _ => None,
    }
}

/// The `ftol_rel` [`optimize_profiled`] sets for `algo`: the user's `outer_ftol` when
/// set, else [`PROFILED_GRADIENT_FTOL_REL`] for a gradient method and the unprofiled
/// `resolve_outer_ftol` for BOBYQA.
pub(crate) fn profiled_ftol(
    algo: nlopt::Algorithm,
    model: &CompiledModel,
    options: &FitOptions,
) -> f64 {
    match (options.outer_ftol, algo) {
        (Some(user), _) => user,
        (None, nlopt::Algorithm::Bobyqa) => {
            resolve_outer_ftol(model.has_non_gaussian(), model.is_ode_based(), None)
        }
        (None, _) => PROFILED_GRADIENT_FTOL_REL,
    }
}

/// The `(xtol_rel, ftol_rel)` every verification leg of a profiled fit runs at. A
/// gradient method's legs keep the first leg's stops ([`profiled_xtol`],
/// [`profiled_ftol`]): it is the gain, not a tighter stop, that verifies them (#2285).
/// BOBYQA's are the #1782 polish legs, at the `POLISH_*` ceilings the unprofiled path
/// polishes at, a tighter user value winning (#2336): restarted at its own relative
/// stops, BOBYQA would stop again wherever the first leg did.
pub(crate) fn profiled_leg_tols(
    algo: nlopt::Algorithm,
    model: &CompiledModel,
    options: &FitOptions,
) -> (f64, f64) {
    let (xtol, ftol) = (
        profiled_xtol(algo, options),
        profiled_ftol(algo, model, options),
    );
    match algo {
        nlopt::Algorithm::Bobyqa => (xtol.min(POLISH_XTOL_REL), ftol.min(POLISH_FTOL_REL)),
        _ => (xtol, ftol),
    }
}

/// The outer loop of a fit with `profile`d blocks: NLopt over the structural packed
/// coordinates, every evaluation a profile solve. Returns the same `OuterResult` the
/// unprofiled path does, its `params` carrying the solved levels and the fit's own
/// `theta_fixed`.
pub(crate) fn optimize_profiled(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let layout = ProfileLayout::build(model, population, init_params)
        .expect("optimize_profiled is called for a model with a profiled block");
    let mut hold = hold_template(&layout, init_params);
    let PackedStart {
        packed: mut x0,
        bounds,
        ..
    } = pack_with_bounds(&hold);
    clamp_to_bounds(&mut x0, &bounds);
    // The fit's template, with any θ the flat-θ pre-flight freezes marked: its
    // `theta_fixed` is what the result publishes, as the unprofiled path's frozen template.
    let mut fit_template = init_params.clone();
    let mut warnings: Vec<String> = Vec::new();
    let runs = runs_outer_optimizer(options);
    {
        let ctx = ProfileCtx {
            model,
            population,
            layout: &layout,
            options,
            inner_tol: options.inner_tol.min(PROFILE_INNER_TOL),
        };
        // The flat-θ pre-flight (#826), on the profile objective. Only when the optimizer
        // runs, as in `optimize_population`.
        if runs {
            for (i, w) in profiled_flat_thetas(&ctx, &hold, &x0, &bounds) {
                let v = unpack_params(&x0, &hold).theta[i];
                for t in [&mut hold, &mut fit_template] {
                    t.theta[i] = v;
                    t.theta_fixed[i] = true;
                }
                warnings.push(w);
            }
        }
    }
    let hold = hold;
    let fixed = packed_fixed_mask(&hold);
    let free: Vec<usize> = (0..x0.len()).filter(|&i| !fixed[i]).collect();
    let ctx = ProfileCtx {
        model,
        population,
        layout: &layout,
        options,
        inner_tol: options.inner_tol.min(PROFILE_INNER_TOL),
    };
    // Reconverging: a subject that declines the analytic gradient at runtime takes the
    // reconverged salvage, not the held-EBE one (#2136, see `envelope_gradient`).
    let declines = profiled_decline_log(population.subjects.len());
    let algo = profiled_algorithm(options);

    // The incumbent: its solve warm-starts the next evaluation (#1290: a warm start from
    // a rejected trial can stall a line search).
    struct Incumbent {
        x: Vec<f64>,
        solve: Option<ProfileSolve>,
        warm: ProfileWarm,
        /// The evaluation that found it, recorded in its checkpoint.
        iter: usize,
    }
    let best = Mutex::new(Incumbent {
        x: x0.clone(),
        solve: None,
        warm: ProfileWarm::from_template(&layout, init_params),
        iter: 0,
    });
    let n_evals = AtomicUsize::new(0);
    let grad_idx = Mutex::new(0usize);
    let full = |xs: &[f64]| {
        let mut x = x0.clone();
        for (&i, &v) in free.iter().zip(xs) {
            x[i] = v;
        }
        x
    };
    let ebe_accum = Mutex::new(EbeAccum::default());
    // The subjects whose profile-solve level gradient fell back to FD at any evaluation.
    let fd_gradients = Mutex::new(std::collections::BTreeSet::new());
    let evaluate_once = |x: &[f64]| -> Option<ProfileSolve> {
        n_evals.fetch_add(1, Ordering::Relaxed);
        let params = unpack_params(x, &hold);
        let warm = best.lock().unwrap().warm.clone();
        let solve = ctx.solve(&params, &warm).ok()?;
        fd_gradients
            .lock()
            .unwrap()
            .extend(solve.fd_gradient_subjects.iter().copied());
        {
            let mut acc = ebe_accum.lock().unwrap();
            acc.max_unconverged = acc.max_unconverged.max(solve.ebe.n_unconverged);
            acc.total_fallback += solve.ebe.n_fallback;
            // The unprofiled loop's EBE guard, on the same counts: too many unconverged
            // subjects (or one hard-rejected at its start) and the trial is a wall.
            if ebe_guard_rejects(
                &solve.ebe,
                population.subjects.len(),
                solve.ofv,
                options.max_unconverged_frac,
            ) {
                acc.n_guarded += 1;
                return None;
            }
        }
        let mut b = best.lock().unwrap();
        if b.solve.as_ref().is_none_or(|s| solve.ofv < s.ofv) {
            b.x = x.to_vec();
            b.warm = ProfileWarm::from_solve(&layout, &solve);
            b.solve = Some(solve.clone());
            b.iter = n_evals.load(Ordering::Relaxed);
            #[cfg(test)]
            INCUMBENT_UPDATES.with(|c| c.set(c.get() + 1));
        }
        Some(solve)
    };
    // Every evaluation, improving or not, then checks whether a checkpoint (#755) is due,
    // and writes the incumbent — the best point, its levels at β̂ — not this evaluation's
    // (#1317): a resume unpacks them as the start's levels, which warm-start its first
    // profile solve. Keyed on the incumbent rather than on an improvement, so a run whose
    // incumbent was found before the interval elapsed still writes once it has. Only a
    // feasible point is ever an incumbent (#1617).
    let evaluate = |x: &[f64]| -> Option<ProfileSolve> {
        let out = evaluate_once(x);
        if crate::io::checkpoint::is_due() {
            let b = best.lock().unwrap();
            if let Some(solve) = &b.solve {
                crate::io::checkpoint::maybe_write(
                    b.iter,
                    solve.ofv,
                    &solved_point(&layout, &hold, &b.x, solve),
                );
            }
        }
        out
    };
    // Optimizer trace: one row per feasible evaluation, every packed coordinate's value
    // (levels at β̂) and the gradient handed to NLopt (`NA` on held coordinates).
    let trace_row = |solve: &ProfileSolve, grad: Option<&[f64]>| {
        if !crate::estimation::trace::is_active() {
            return;
        }
        let grad_norm = grad.map(|g| free.iter().map(|&i| g[i] * g[i]).sum::<f64>().sqrt());
        let grads = grad.map(|g| {
            (0..g.len())
                .map(|i| if fixed[i] { f64::NAN } else { g[i] })
                .collect::<Vec<_>>()
        });
        crate::estimation::trace::write_foce(
            n_evals.load(Ordering::Relaxed),
            if options.interaction { "focei" } else { "foce" },
            solve.ofv,
            grad_norm,
            None,
            match algo {
                nlopt::Algorithm::Bobyqa => "bobyqa",
                nlopt::Algorithm::Slsqp => "slsqp",
                _ => "nlopt_lbfgs",
            },
            Some(solve.ebe.n_unconverged),
            Some(solve.ebe.n_fallback),
            &crate::estimation::parameterization::coordinate_values(&solve.params),
            grads.as_deref(),
        );
    };

    let mut converged = false;
    if runs && !free.is_empty() {
        let lower: Vec<f64> = free.iter().map(|&i| bounds.lower[i]).collect();
        let upper: Vec<f64> = free.iter().map(|&i| bounds.upper[i]).collect();
        let objective = |xs: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
            if crate::cancel::is_cancelled(&options.cancel) {
                if let Some(g) = grad {
                    g.fill(0.0);
                }
                return 1e20;
            }
            let x = full(xs);
            match evaluate(&x) {
                Some(solve) => {
                    if let Some(g) = grad {
                        let mut idx = grad_idx.lock().unwrap();
                        let mut gr = envelope_gradient(
                            &ctx, &hold, &x, &solve, &bounds, &mut idx, &declines,
                        );
                        for &i in &free {
                            if !gr[i].is_finite() {
                                gr[i] = 0.0;
                            }
                        }
                        for (gi, &i) in g.iter_mut().zip(&free) {
                            *gi = gr[i];
                        }
                        trace_row(&solve, Some(&gr));
                    } else {
                        trace_row(&solve, None);
                    }
                    if options.verbose {
                        eprintln!(
                            "Eval {:>4}: OFV = {:.6} (profile: {} Newton step{}, gradient {:.1e} SE, \
                             {} adjoint level gradients, {} subjects on FD)",
                            n_evals.load(Ordering::Relaxed),
                            solve.ofv,
                            solve.iterations,
                            if solve.iterations == 1 { "" } else { "s" },
                            solve.criterion,
                            solve.n_adjoint_gradients,
                            solve.fd_gradient_subjects.len()
                        );
                    }
                    solve.ofv
                }
                None => {
                    // A failed profile solve is a wall, as a failed inner solve is; the
                    // gradient pushes back toward the box centre.
                    if let Some(g) = grad {
                        for (k, gi) in g.iter_mut().enumerate() {
                            *gi = 100.0 * (xs[k] - 0.5 * (lower[k] + upper[k]));
                        }
                    }
                    1e20
                }
            }
        };
        let mut opt = nlopt::Nlopt::new(algo, free.len(), &objective, nlopt::Target::Minimize, ());
        opt.set_lower_bounds(&lower).unwrap();
        opt.set_upper_bounds(&upper).unwrap();
        let per_iter = free.len() as u32 + 1;
        let mut budget = (options.outer_maxiter as u32).saturating_mul(per_iter);
        let bobyqa = matches!(algo, nlopt::Algorithm::Bobyqa);
        if bobyqa {
            budget = budget.saturating_add(40 * per_iter);
            opt.set_initial_step(&bobyqa_initial_step(&lower, &upper))
                .unwrap();
        }
        opt.set_maxeval(budget.max(1)).unwrap();
        opt.set_xtol_rel(profiled_xtol(algo, options)).unwrap();
        opt.set_ftol_rel(profiled_ftol(algo, model, options))
            .unwrap();
        let (leg_xtol, leg_ftol) = profiled_leg_tols(algo, model, options);
        let mut xs: Vec<f64> = free.iter().map(|&i| x0[i]).collect();
        let first = opt.optimize(&mut xs);
        if options.verbose {
            eprintln!("Profiled outer optimizer finished: {first:?}");
        }
        let cancelled = || crate::cancel::is_cancelled(&options.cancel);
        // Has the incumbent moved off the start? Packed coordinates, unscaled here.
        let left_init = || {
            let b = best.lock().unwrap();
            max_scaled_deviation(&b.x, &x0) >= INIT_ESCAPE_STEP_S
        };
        // #2285: a gradient method's stop is not evidence of a minimum — `ftol_rel` and
        // `xtol_rel` are tests on one iteration, which L-BFGS meets mid-descent, and
        // `main`'s L-BFGS returns `Failure` at a genuine optimum. So run a fresh leg from
        // the best point at the same stops; the fit is converged when a leg gains no
        // more than `PROFILED_VERIFY_GAIN`. Every leg feeds the same incumbent and shares
        // the budget, the #1782 loop.
        //
        // #2336: a BOBYQA tolerance stop is no better evidence — on A2 it stopped
        // `FtolReached` 3.46 OFV above the optimum — so it gets the #1782 polish legs:
        // a restart from the best point at the `POLISH_*` tolerances and a fresh trust
        // radius, judged on the same gain.
        let (result, end) = drive_polish_legs(
            first,
            PROFILED_VERIFY_GAIN,
            |r| profiled_stop_needs_verify(algo, r, cancelled(), left_init()),
            || budget.saturating_sub(n_evals.load(Ordering::Relaxed) as u32),
            |left| {
                let (bx, before) = {
                    let b = best.lock().unwrap();
                    (b.x.clone(), b.solve.as_ref()?.ofv)
                };
                let mut leg =
                    nlopt::Nlopt::new(algo, free.len(), &objective, nlopt::Target::Minimize, ());
                leg.set_lower_bounds(&lower).unwrap();
                leg.set_upper_bounds(&upper).unwrap();
                leg.set_maxeval(left).unwrap();
                leg.set_xtol_rel(leg_xtol).unwrap();
                leg.set_ftol_rel(leg_ftol).unwrap();
                if bobyqa {
                    leg.set_initial_step(&bobyqa_initial_step(&lower, &upper))
                        .unwrap();
                }
                let mut lx: Vec<f64> = free.iter().map(|&i| bx[i]).collect();
                let r = leg.optimize(&mut lx);
                let after = best
                    .lock()
                    .unwrap()
                    .solve
                    .as_ref()
                    .map_or(before, |s| s.ofv);
                let gain = before - after;
                #[cfg(test)]
                VERIFY_LEGS.with(|v| {
                    v.borrow_mut().push(VerifyLeg {
                        xtol: leg.get_xtol_rel(),
                        ftol: leg.get_ftol_rel(),
                        gain,
                        before,
                        after,
                    })
                });
                if options.verbose {
                    eprintln!(
                        "Profiled verification leg from OFV {before:.9}: {r:?}, gain {gain:.3e}"
                    );
                }
                Some((r, gain))
            },
        );
        converged = profiled_converged(algo, end, &result, cancelled(), left_init());
        warnings.extend(profiled_verdict_warning(end, &result, converged));
    } else {
        // Evaluation only (`maxiter = 0`, NONMEM `MAXEVAL=0`), or nothing to search: the
        // levels are still solved at the initial structural point.
        if let Some(solve) = evaluate(&x0) {
            trace_row(&solve, None);
        }
    }

    let (x_best, solve) = {
        let b = best.lock().unwrap();
        (b.x.clone(), b.solve.clone())
    };
    let Some(solve) = solve else {
        warnings.push(
            "Profile solve failed at every evaluated point: no objective is available.".to_string(),
        );
        let params = unpack_params(&x0, &fit_template);
        let ebe = ebe_accum.into_inner().unwrap();
        return failed_result(params, population, model, warnings, &ebe);
    };
    let x_hat = solved_point(&layout, &hold, &x_best, &solve);

    let cov = if options.run_covariance_step {
        let (out, _) = ctx.covariance_at_estimate(
            &hold,
            &x_hat,
            &free,
            &solve.params,
            &solve.etas,
            &solve.kappas,
        );
        warnings.extend(out.warnings.iter().cloned());
        Some(out)
    } else {
        None
    };
    // #1303: the verdict a consumer keys on is gated on the objective it describes.
    if let Some(w) =
        crate::estimation::outer_optimizer::gate_converged_on_objective(&mut converged, solve.ofv)
    {
        warnings.push(w);
    }
    // Per-subject outer FD fallbacks that happened in the outer loop, from its runtime log,
    // as `optimize_population` reports them (#1154).
    if let Some(w) =
        crate::estimation::outer_optimizer::outer_fd_fallback_warning(model, population, &declines)
    {
        warnings.push(w);
    }
    if let Some(w) = profile_fd_gradient_warning(
        population,
        &fd_gradients.into_inner().unwrap(),
        "during this fit",
    ) {
        warnings.push(w);
    }
    let mut params = solve.params.clone();
    params.theta_fixed = fit_template.theta_fixed.clone();
    let ebe = ebe_accum.into_inner().unwrap();
    OuterResult {
        params,
        ofv: solve.ofv,
        converged,
        n_iterations: n_evals.load(Ordering::Relaxed),
        eta_hats: solve.etas,
        h_matrices: solve.h_matrices,
        kappas: solve.kappas,
        covariance_matrix: cov.as_ref().and_then(|c| c.matrix.clone()),
        covariance_method: cov.as_ref().and_then(|c| c.method),
        covariance_wall_time_secs: cov.as_ref().map_or(0.0, |c| c.wall_time_secs),
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: ebe.n_guarded as u32,
        max_unconverged_subjects: ebe.max_unconverged as u32,
        total_ebe_fallbacks: ebe.total_fallback as u32,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal: cov.and_then(|c| c.sir_fallback_proposal),
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        packed_estimate: Some(x_hat),
        left_init: None,
        mixture_posteriors: None,
        vi: None,
    }
}

/// The warning `covariance_at_estimate` returns when the profile solve at the estimates
/// fails, on the inline `fit()` step and on `replay_covariance` alike.
/// A function rather than an inline `format!` so the `classify_warning` table test
/// (`types_tests.rs`) classifies the text this site emits, not a copy of it (#2158).
pub(crate) fn profile_solve_failed_msg(e: &ProfileFailure) -> String {
    format!(
        "Covariance step failed: the profile solve did not converge at the \
         estimates ({e:?}). SE estimates not available."
    )
}

/// The warning `covariance_at_estimate` returns when the EBE guard rejects the profile
/// solve at the estimates (#2214), as it would reject that solve as an optimizer
/// evaluation. A function so the `classify_warning` table test classifies the text this
/// site emits (#2158).
///
/// It names the trigger that fired. `ebe_guard_rejects` rejects any subject rejected at its
/// inner start whatever `max_unconverged_frac` is, and otherwise only an unconverged
/// fraction above a non-negative `max_unconverged_frac`, so with no start rejection the
/// fraction is what was exceeded.
///
/// `held`: the rejected solve is the second, held-layout re-solve a level on its bound
/// takes (#2420), which the text names, so a failure says which of the two fired.
pub(crate) fn profile_resolve_guarded_msg(
    ebe: &InnerLoopStats,
    n_subj: usize,
    max_unconverged_frac: f64,
    held: bool,
) -> String {
    let reason = if ebe.n_start_rejected > 0 {
        "any subject rejected at its inner start, whatever max_unconverged_frac is".to_string()
    } else {
        format!("an unconverged fraction above max_unconverged_frac = {max_unconverged_frac}")
    };
    let which = if held {
        "the profile re-solve with the levels on their bound held there"
    } else {
        "the profile solve at the estimates"
    };
    format!(
        "Covariance step failed: {which} left {} of {n_subj} subjects' EBEs unconverged and \
         rejected {} at their inner start. The EBE guard rejects {reason}. SE estimates not \
         available.",
        ebe.n_unconverged, ebe.n_start_rejected
    )
}

/// A θ name that makes the covariance step's re-profiling solve fail as `Singular`, for the
/// tests: `fit()` has no natural point where the optimizer's solve succeeds and the same
/// solve from its own mode fails (#2214). A name on the model rather than a thread-local,
/// because `fit()` runs the step on a pool thread.
#[cfg(test)]
pub(crate) const FAIL_COVARIANCE_RESOLVE: &str = "FAILS_COV_RESOLVE";

/// A θ name that makes the covariance step run its held-layout re-solve (#2112) cold and at
/// `inner_maxiter = 0`, so its EBEs stop short while the first re-solve's, warm at their
/// modes, do not (#2420). A name on the model for the reason above: `run_covariance` runs
/// the step on a pool thread too.
#[cfg(test)]
pub(crate) const STARVE_HELD_RESOLVE: &str = "STARVES_HELD_RESOLVE";

/// The warning of a profiled covariance step whose profile-objective Hessian `A_p` is not
/// positive definite, its eigenvalues listed; the step then hands SIR the rectified
/// fallback proposal. A function so the `classify_warning` table test classifies the text
/// this site emits (#2158).
pub(crate) fn profile_non_pd_msg(eigvals: &[f64]) -> String {
    let eig = eigvals
        .iter()
        .map(|&v| crate::estimation::covariance::fmt_eig(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Covariance step failed: the Hessian of the profile objective is not positive \
         definite (eigenvalues [{eig}]). SE estimates not available."
    )
}

/// The covariance step of a profiled fit, re-run on its estimates (`run_covariance`), or
/// `None` when `model` profiles nothing. The levels are re-profiled from the fit's own
/// values first, so a reloaded fit gets the step the fit ran — not the unprofiled `R`
/// over every level, which at a few hundred levels is the stencil profiling exists to
/// avoid. Every `covariance_method` replays as `fit()` runs it (#2108).
///
/// `x_hat` is the packed point the step runs at — the fit's own packed vector when
/// `run_covariance` can reuse it — not a repack of `params`, which can differ in the last
/// bits through the log and Cholesky transforms. With the outcome come the levels the step
/// held on their bound and the levels it ran at ([`ProfileCtx::covariance_at_estimate`]).
pub(crate) fn replay_covariance(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    x_hat: &[f64],
    eta_hats: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> Option<(CovStepOutcome, Option<ResolvedLevels>)> {
    let layout = ProfileLayout::build(model, population, params)?;
    let hold = hold_template(&layout, params);
    let fixed = packed_fixed_mask(&hold);
    let free: Vec<usize> = (0..x_hat.len()).filter(|&i| !fixed[i]).collect();
    let ctx = ProfileCtx {
        model,
        population,
        layout: &layout,
        options,
        inner_tol: options.inner_tol.min(PROFILE_INNER_TOL),
    };
    Some(ctx.covariance_at_estimate(&hold, x_hat, &free, params, eta_hats, kappas))
}

/// What a profiled covariance step's re-solve set, when one passed the EBE guard
/// ([`ProfileCtx::covariance_at_estimate`]).
pub(crate) struct ResolvedLevels {
    /// The levels on their bound, which the step held (#2112), as
    /// [`ProfileLayout::at_bound`] lists them.
    pub(crate) pinned: Vec<(usize, f64, bool)>,
    /// Every profiled level as `(θ index, value)` at the solve the step's matrix is at: the
    /// held-layout re-solve when a level is held, else the first. Not the incoming estimates
    /// when the re-solve moved them (#2419).
    pub(crate) levels: Vec<(usize, f64)>,
}

/// EBE convergence over a profiled run, as the unprofiled NLopt loop keeps it: the worst
/// unconverged count, the fallbacks, and the evaluations the EBE guard rejected.
#[derive(Default)]
struct EbeAccum {
    max_unconverged: usize,
    total_fallback: usize,
    n_guarded: usize,
}

/// An `OuterResult` for a fit whose profile solve never succeeded.
fn failed_result(
    params: ModelParameters,
    population: &Population,
    model: &CompiledModel,
    warnings: Vec<String>,
    ebe: &EbeAccum,
) -> OuterResult {
    let n = population.subjects.len();
    OuterResult {
        params,
        ofv: f64::INFINITY,
        converged: false,
        n_iterations: 0,
        eta_hats: vec![DVector::zeros(model.n_eta); n],
        h_matrices: population
            .subjects
            .iter()
            .map(|s| DMatrix::zeros(s.obs_times.len(), model.n_eta))
            .collect(),
        kappas: vec![Vec::new(); n],
        covariance_matrix: None,
        covariance_method: None,
        covariance_wall_time_secs: 0.0,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: ebe.n_guarded as u32,
        max_unconverged_subjects: ebe.max_unconverged as u32,
        total_ebe_fallbacks: ebe.total_fallback as u32,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal: None,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        packed_estimate: None,
        left_init: None,
        mixture_posteriors: None,
        vi: None,
    }
}

/// The inflation of the non-PD SIR fallback proposal, as the unprofiled step's.
const NON_PD_PROPOSAL_INFLATION: f64 = 4.0;

/// The full packed covariance of a profiled fit (levels included) from the structural
/// block `cov_s` (`nf × nf`, over `free`), `D` per group and `B` (`nf × L`, both OFV
/// scale), with `H = [[A, B], [Bᵀ, D]]`:
/// - `Cov(θ, β) = −Cov(θ)·B·D⁻¹`,
/// - `Cov(β) = d_scale·D⁻¹ + D⁻¹Bᵀ·Cov(θ)·B·D⁻¹`.
///
/// `d_scale = 2` and `cov_s = 2·A_p⁻¹` give the blocks of `2·H⁻¹`; the non-PD SIR proposal
/// passes its own inflated `cov_s` and `d_scale`. `None` when a group's `D` is not
/// positive definite.
fn assemble_profiled_covariance(
    layout: &ProfileLayout,
    free: &[usize],
    n: usize,
    cov_s: &DMatrix<f64>,
    d_groups: &[DMatrix<f64>],
    b: &DMatrix<f64>,
    d_scale: f64,
) -> Option<DMatrix<f64>> {
    let nf = free.len();
    let l = layout.coords.len();
    // D⁻¹ per group, and D⁻¹Bᵀ (L × nf).
    let mut d_inv_bt = DMatrix::zeros(l, nf);
    let mut d_inv: Vec<DMatrix<f64>> = Vec::with_capacity(layout.groups.len());
    for (g, d) in layout.groups.iter().zip(d_groups) {
        let inv = d.clone().cholesky()?.inverse();
        for (a, &pa) in g.coords.iter().enumerate() {
            for s in 0..nf {
                d_inv_bt[(pa, s)] = (0..g.coords.len())
                    .map(|c| inv[(a, c)] * b[(s, g.coords[c])])
                    .sum();
            }
        }
        d_inv.push(inv);
    }
    // Cov(θ, β) = −Cov(θ)·B·D⁻¹ = −Cov(θ)·(D⁻¹Bᵀ)ᵀ.
    let cov_sb = -(cov_s * d_inv_bt.transpose());
    // Cov(β) = d_scale·D⁻¹ + (D⁻¹Bᵀ)·Cov(θ)·(D⁻¹Bᵀ)ᵀ.
    let mut cov_bb = &d_inv_bt * cov_s * d_inv_bt.transpose();
    for (g, inv) in layout.groups.iter().zip(&d_inv) {
        for (a, &pa) in g.coords.iter().enumerate() {
            for (c, &pc) in g.coords.iter().enumerate() {
                cov_bb[(pa, pc)] += d_scale * inv[(a, c)];
            }
        }
    }
    let mut full = DMatrix::zeros(n, n);
    for a in 0..nf {
        for c in 0..nf {
            full[(free[a], free[c])] = cov_s[(a, c)];
        }
        for (p, &k) in layout.coords.iter().enumerate() {
            full[(free[a], k)] = cov_sb[(a, p)];
            full[(k, free[a])] = cov_sb[(a, p)];
        }
    }
    for (p, &k) in layout.coords.iter().enumerate() {
        for (q, &j) in layout.coords.iter().enumerate() {
            full[(k, j)] = cov_bb[(p, q)];
        }
    }
    Some(full)
}

impl ProfileCtx<'_> {
    /// The profile objective at the packed point `x` (levels ignored), re-profiled from
    /// `warm`; `NaN` when the solve fails. The covariance step's stencil scores the same
    /// solve and also collects its FD gradient fallbacks.
    #[cfg(test)]
    pub(crate) fn profile_ofv(&self, hold: &ModelParameters, x: &[f64], warm: &ProfileWarm) -> f64 {
        self.solve(&unpack_params(x, hold), warm)
            .map_or(f64::NAN, |s| s.ofv)
    }

    /// The covariance step at an estimate: re-profile the levels from `params`' own values
    /// and from `eta_hats` with `kappas` (the joint start, #2261), then [`Self::covariance`]
    /// at that solve. `fit()` and `replay_covariance` both go through here, so a replay
    /// handed the fit's estimates, EBEs and κ̂ repeats the inline step bit for bit (#2173).
    ///
    /// The re-solve is an evaluation like any other (#2214): it goes through the optimizer's
    /// EBE guard, a rejected re-solve failing the step with a warning that names its counts,
    /// and the step's wall time includes it. The held-layout re-solve below is guarded the
    /// same way (#2420). On the exact route (#2109) it runs at the
    /// step's inner tolerance ([`Self::at_cov_tol`]), since every block is read at its modes;
    /// so does the held-layout re-solve below, for the same reason.
    ///
    /// Also returns what the re-solve set ([`ResolvedLevels`]): the levels it set on their
    /// bound, which the step held (#2112), and every level where the step ran. `None` when
    /// the step failed before a re-solve passed the guard, so a caller reporting them falls
    /// back to its own estimates.
    pub(crate) fn covariance_at_estimate(
        &self,
        hold: &ModelParameters,
        x: &[f64],
        free: &[usize],
        params: &ModelParameters,
        eta_hats: &[DVector<f64>],
        kappas: &[Vec<DVector<f64>>],
    ) -> (CovStepOutcome, Option<ResolvedLevels>) {
        let start = Instant::now();
        let failed = |reason: String| CovStepOutcome {
            matrix: None,
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![reason],
            sir_fallback_proposal: None,
            method: None,
        };
        let fail = |reason: String| (failed(reason), None);
        let layout = self.layout;
        let warm = ProfileWarm {
            beta: layout.coords.iter().map(|&k| params.theta[k]).collect(),
            etas: (eta_hats.len() == self.population.subjects.len())
                .then(|| joint_warm_starts(eta_hats, kappas)),
        };
        // The exact route reads every block at this solve's modes, so it runs at the step's
        // inner tolerance, behind the guard below (#2417 review rows 1–2). The FD route and
        // `s` keep the solve they had: the stencil re-solves at that tolerance itself.
        let exact = self.options.covariance_method != CovarianceMethod::CrossProduct
            && self.cov_is_analytic();
        let solver = if exact {
            self.at_cov_tol()
        } else {
            ProfileCtx { ..*self }
        };
        let solved = solver.solve(params, &warm);
        #[cfg(test)]
        let solved = if self
            .model
            .theta_names
            .iter()
            .any(|n| n == FAIL_COVARIANCE_RESOLVE)
        {
            Err(ProfileFailure::Singular)
        } else {
            solved
        };
        let solve = match solved {
            Ok(s) => s,
            Err(e) => return fail(profile_solve_failed_msg(&e)),
        };
        let n_subj = self.population.subjects.len();
        let limit = self.options.max_unconverged_frac;
        if ebe_guard_rejects(&solve.ebe, n_subj, solve.ofv, limit) {
            return fail(profile_resolve_guarded_msg(
                &solve.ebe, n_subj, limit, false,
            ));
        }
        // On a converged start `solve_group` takes no Newton step, so β̂ is unchanged and this
        // overwrite is a bitwise no-op; it matters when the re-solve moves the levels — a
        // replay handed levels other than the fit's β̂ (`run_covariance` on edited estimates).
        let x_hat = solved_point(layout, hold, x, &solve);
        let pinned = layout.at_bound(&solve.params.theta);
        let resolved = |solve: &ProfileSolve| ResolvedLevels {
            pinned: pinned.clone(),
            levels: layout
                .coords
                .iter()
                .map(|&k| (k, solve.params.theta[k]))
                .collect(),
        };
        if pinned.is_empty() {
            return (
                self.covariance(hold, &x_hat, free, &solve, start),
                Some(resolved(&solve)),
            );
        }
        // A level on its bound (#2112) is held there for the step: the step runs on the
        // layout with those levels frozen (#1891's offset path), so `A_p` re-profiles only
        // the free levels and no stencil point can move a level on or off its bound, and the
        // held levels get the zero rows of a FIX θ. `pinned` is a function of the estimates
        // and the declared box alone, so a replay freezes the same levels.
        // The estimates with the fit's own FIX mask: the solve's `params` carry the hold
        // template's, which fixes every level.
        let mut frozen = solve.params.clone();
        frozen.theta_fixed = unheld_template(layout, hold).theta_fixed;
        for &(k, _, _) in &pinned {
            frozen.theta_fixed[k] = true;
        }
        let Some(held_layout) = ProfileLayout::build(self.model, self.population, &frozen) else {
            let out = failed(profile_solve_failed_msg(&ProfileFailure::Singular));
            return (out, Some(resolved(&solve)));
        };
        // `solver`'s tolerance: on the exact route the held re-solve is the solve every block
        // is read at.
        let held_ctx = ProfileCtx {
            layout: &held_layout,
            ..solver
        };
        let held_warm = ProfileWarm {
            beta: held_layout
                .coords
                .iter()
                .map(|&k| solve.params.theta[k])
                .collect(),
            etas: Some(joint_warm_starts(&solve.etas, &solve.kappas)),
        };
        #[cfg(test)]
        let starved = self
            .model
            .theta_names
            .iter()
            .any(|n| n == STARVE_HELD_RESOLVE)
            .then(|| FitOptions {
                inner_maxiter: 0,
                ..self.options.clone()
            });
        #[cfg(test)]
        let (held_ctx, held_warm) = match &starved {
            Some(o) => (
                ProfileCtx {
                    options: o,
                    ..held_ctx
                },
                ProfileWarm {
                    etas: None,
                    ..held_warm
                },
            ),
            None => (held_ctx, held_warm),
        };
        let held_solve = match held_ctx.solve(&frozen, &held_warm) {
            Ok(s) => s,
            Err(e) => return (failed(profile_solve_failed_msg(&e)), Some(resolved(&solve))),
        };
        // The held re-solve is an evaluation as the first one is, and on the exact route the
        // one every block is read from, so it goes through the same guard (#2420).
        if ebe_guard_rejects(&held_solve.ebe, n_subj, held_solve.ofv, limit) {
            let out = failed(profile_resolve_guarded_msg(
                &held_solve.ebe,
                n_subj,
                limit,
                true,
            ));
            return (out, Some(resolved(&solve)));
        }
        let x_hat = solved_point(&held_layout, hold, &x_hat, &held_solve);
        (
            held_ctx.covariance(hold, &x_hat, free, &held_solve, start),
            Some(resolved(&held_solve)),
        )
    }

    /// `covariance_method = r` for a profiled fit, as the full packed covariance the
    /// unprofiled step would publish (levels included), so every downstream reader —
    /// `se_theta`, the level SEs and their delta method for a `NegSum` dependent, SIR's
    /// proposal — reads it unchanged.
    ///
    /// With `H = [[A, B], [Bᵀ, D]]` the OFV Hessian over (structural, levels):
    /// - `Cov(θ) = 2·A_p⁻¹`, `A_p = A − BD⁻¹Bᵀ` the Hessian of the profile objective. On
    ///   the adjoint route it is that Schur complement of the exact blocks
    ///   ([`Self::exact_covariance`], #2109); otherwise it is differenced with the levels
    ///   re-profiled at every stencil point. Holding them fixed would give `A` instead, and
    ///   `2·A⁻¹` is too small.
    /// - `Cov(β) = 2·D⁻¹ + D⁻¹Bᵀ Cov(θ) B D⁻¹`, `Cov(θ, β) = −Cov(θ) B D⁻¹`: the level
    ///   blocks of `2·H⁻¹`. `D` is the total Hessian (EBE re-solved), not the Newton
    ///   matrix; the two differ when the model is nonlinear in η.
    ///
    /// `covariance_method = s` and `rsr` (#2108) are the unprofiled estimators over the same
    /// packed coordinates, levels included, so their θ blocks are what the unprofiled step
    /// reports: `s` is `S⁻¹` ([`Self::cross_product_covariance`]), built without the stencil
    /// as unprofiled; `rsr` combines the matrix above with `S`
    /// ([`Self::covariance_from_profile_hessian`]).
    ///
    /// `start` is when the step began, the re-profiling solve before it included.
    pub(crate) fn covariance(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
        start: Instant,
    ) -> CovStepOutcome {
        let options = self.options;
        if options.covariance_method == CovarianceMethod::CrossProduct {
            return match self.cross_product_covariance(hold, x_hat, solve) {
                Ok(matrix) => CovStepOutcome {
                    matrix: Some(matrix),
                    wall_time_secs: start.elapsed().as_secs_f64(),
                    warnings: Vec::new(),
                    sir_fallback_proposal: None,
                    method: Some(CovarianceMethod::CrossProduct),
                },
                Err(reason) => CovStepOutcome {
                    matrix: None,
                    wall_time_secs: start.elapsed().as_secs_f64(),
                    warnings: vec![reason],
                    sir_fallback_proposal: None,
                    method: None,
                },
            };
        }
        let cov_ctx = self.at_cov_tol();
        let warm = ProfileWarm::from_solve(self.layout, solve);
        // The subjects whose level gradient fell back to FD in the stencil's solves. The
        // first of them is at the estimate itself, so it sees what the re-solve before the
        // step saw; collecting that one too was a second gate on the same inputs.
        let fd_gradients = Mutex::new(std::collections::BTreeSet::new());
        let f = |xv: &[f64]| match cov_ctx.solve(&unpack_params(xv, hold), &warm) {
            Ok(s) => {
                fd_gradients
                    .lock()
                    .unwrap()
                    .extend(s.fd_gradient_subjects.iter().copied());
                s.ofv
            }
            Err(_) => f64::NAN,
        };
        let mut out = if cov_ctx.cov_is_analytic() {
            // No stencil runs, so the estimate's own solve is the one whose fallbacks the
            // step reports.
            fd_gradients
                .lock()
                .unwrap()
                .extend(solve.fd_gradient_subjects.iter().copied());
            cov_ctx.exact_covariance(hold, x_hat, free, solve, start)
        } else {
            self.covariance_stencil(hold, x_hat, free, solve, start, &cov_ctx, &f, None)
        };
        if let Some(w) = profile_fd_gradient_warning(
            self.population,
            &fd_gradients.into_inner().unwrap(),
            "during the covariance step",
        ) {
            out.warnings.push(w);
        }
        out
    }

    /// Whether the covariance step's level blocks and `A_p` are the adjoint's exact ones
    /// (#2109): the profile gradient's engine ([`profile_gradient_is_analytic`], so
    /// `gradient = fd` keeps the step on finite differences too), and the switch the
    /// unprofiled step's exact R-matrix reads (`analytic_cov_hessian`).
    fn cov_is_analytic(&self) -> bool {
        self.options.analytic_cov_hessian && profile_gradient_is_analytic(self.model, self.options)
    }

    /// The covariance step's context: [`Self`] at the step's inner tolerance
    /// (`effective_cov_inner_tol`, never looser than this solve's own).
    fn at_cov_tol(&self) -> Self {
        ProfileCtx {
            inner_tol: self
                .options
                .effective_cov_inner_tol(self.model.uses_closed_form_ltbs_inner())
                .min(self.inner_tol),
            ..*self
        }
    }

    /// [`Self::covariance`] on the adjoint (#2109), at `solve`: the estimate's re-profiling
    /// solve, which [`Self::covariance_at_estimate`] ran at the step's inner tolerance and
    /// guarded. At its modes every subject's exact Hessian block over the structural
    /// coordinates and its own levels gives `A`, `B` and `D` at once ([`Self::level_blocks`]).
    /// At `β̂`, where the profile gradient is zero, `A_p = A − BD⁻¹Bᵀ` is the exact Hessian
    /// of the profile objective, which replaces the stencil's ~`2·nf²` population re-solves
    /// (89–91% of the step on S1).
    ///
    /// A subject the adjoint declines takes the re-solved differences for its own `Aᵢ`,
    /// `Dᵢ` and `Bᵢ` and is reported; `A_p` stays a Schur complement, with that one term on
    /// finite differences (#2417 review row 3).
    fn exact_covariance(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
        start: Instant,
    ) -> CovStepOutcome {
        let fail = |reason: String| CovStepOutcome {
            matrix: None,
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![reason],
            sir_fallback_proposal: None,
            method: None,
        };
        let blocks = match self.level_blocks(hold, x_hat, free, solve) {
            Ok(v) => v,
            Err(reason) => return fail(reason),
        };
        if crate::cancel::is_cancelled(&self.options.cancel) {
            return fail(crate::estimation::covariance::COV_CANCELLED_MSG.to_string());
        }
        // A subject off its mode below `min_obs_for_convergence_check` took the re-solved
        // blocks, as it must, but is not named (#2408).
        let reported: Vec<usize> = blocks
            .fd_subjects
            .iter()
            .copied()
            .filter(|&i| !solve.off_mode_excused.get(i).copied().unwrap_or(false))
            .collect();
        let declined = profile_fd_level_blocks_warning(self.population, &reported);
        if self.options.verbose {
            eprintln!(
                "  Covariance step (profile): exact level blocks, {} of {} subjects on \
                 finite differences",
                blocks.fd_subjects.len(),
                self.population.subjects.len()
            );
        }
        let a = blocks
            .a
            .as_ref()
            .expect("the adjoint route gives every subject an Aᵢ");
        let Some(a_p) = schur_profile_hessian(self.layout, a, &blocks.d_groups, &blocks.b) else {
            return fail(LEVELS_NOT_PD.to_string());
        };
        #[cfg(test)]
        PROFILE_COV_SCHUR.with(|c| c.set(c.get() + 1));
        let mut out = self.covariance_from_profile_hessian(
            hold,
            x_hat,
            free,
            solve,
            &a_p,
            start,
            Some(blocks),
        );
        out.warnings.extend(declined);
        out
    }

    /// [`Self::covariance`] from the profile objective `f` on: the stencil of `A_p`, then
    /// the level blocks (`blocks`, when the caller has them).
    #[allow(clippy::too_many_arguments)]
    fn covariance_stencil<F: Fn(&[f64]) -> f64 + Sync>(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
        start: Instant,
        cov_ctx: &ProfileCtx<'_>,
        f: &F,
        blocks: Option<LevelBlocks>,
    ) -> CovStepOutcome {
        let fail = |reason: String| CovStepOutcome {
            matrix: None,
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![reason],
            sir_fallback_proposal: None,
            method: None,
        };
        let options = self.options;
        let n = x_hat.len();
        let f0 = f(x_hat);
        if !f0.is_finite() {
            return fail(PROFILE_NOT_FINITE_AT_ESTIMATE.to_string());
        }
        let floor = fd_step_floors(hold, &self.model.default_params);
        let Some(stencil) = fd_ofv_hessian(
            n,
            x_hat,
            free,
            &floor,
            profile_fd_hessian_step(options),
            f0,
            f,
            options,
        ) else {
            return fail(crate::estimation::covariance::COV_CANCELLED_MSG.to_string());
        };
        if !stencil.diag_nan.is_empty() || !stencil.offdiag_nan.is_empty() {
            return fail(
                "Covariance step failed: the profile objective is not finite at some \
                 finite-difference point. SE estimates not available."
                    .to_string(),
            );
        }
        let nf = free.len();
        let a_p = DMatrix::from_fn(nf, nf, |a, b| stencil.hess[(free[a], free[b])]);
        cov_ctx.covariance_from_profile_hessian(hold, x_hat, free, solve, &a_p, start, blocks)
    }

    /// The rest of [`Self::covariance`] once `A_p` (the profile objective's OFV Hessian over
    /// `free`) is known: `D` and `B` (`blocks`, or [`Self::level_blocks`] when `None`), then
    /// the full matrix — or, when `A_p` is not positive definite, no matrix and the non-PD
    /// SIR fallback proposal over every packed coordinate. `self` is the covariance step's
    /// context (its inner tolerance).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn covariance_from_profile_hessian(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
        a_p: &DMatrix<f64>,
        start: Instant,
        blocks: Option<LevelBlocks>,
    ) -> CovStepOutcome {
        let fail = |reason: String| CovStepOutcome {
            matrix: None,
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![reason],
            sir_fallback_proposal: None,
            method: None,
        };
        let options = self.options;
        let n = x_hat.len();
        let nf = free.len();
        // `R⁻¹` over the structural coordinates, or — when `A_p` is not positive definite —
        // the non-PD fallback's |λ|-rectified proposal (`build_non_pd_fallback_proposal`,
        // as the unprofiled step builds it, at the same 4× inflation), with the reason.
        let (cov_s, non_pd) = match a_p.clone().cholesky() {
            Some(ch) => (ch.inverse() * 2.0, None),
            None => {
                let sym = (a_p + a_p.transpose()) * 0.5;
                let Some(eigvals) = crate::estimation::covariance::extract_eigenvalues(&sym) else {
                    return fail(
                        "Covariance step failed: could not compute eigenvalues of the Hessian \
                         of the profile objective (it may contain NaN or Inf). SE estimates \
                         not available."
                            .to_string(),
                    );
                };
                let units = crate::estimation::covariance::free_coordinate_units(
                    hold,
                    x_hat,
                    free,
                    &self.model.nn_weight_theta_ranges(),
                );
                let all: Vec<usize> = (0..nf).collect();
                let proposal = crate::estimation::covariance::build_non_pd_fallback_proposal(
                    &sym,
                    &all,
                    nf,
                    NON_PD_PROPOSAL_INFLATION,
                    &units,
                );
                (proposal, Some(profile_non_pd_msg(&eigvals)))
            }
        };

        // D (per group) and B, per subject over its own levels, the EBE re-solved.
        let blocks = match blocks.map_or_else(|| self.level_blocks(hold, x_hat, free, solve), Ok) {
            Ok(v) => v,
            Err(reason) => return fail(non_pd.unwrap_or(reason)),
        };
        let (d_groups, b) = (blocks.d_groups, blocks.b);
        // The proposal is `4·|H|⁻¹` where the covariance is `2·H⁻¹`: its `D⁻¹` term carries
        // the same factor 2, so the level blocks are those of the rectified, inflated
        // block inverse.
        let d_scale = if non_pd.is_some() {
            NON_PD_PROPOSAL_INFLATION
        } else {
            2.0
        };
        let Some(full) =
            assemble_profiled_covariance(self.layout, free, n, &cov_s, &d_groups, &b, d_scale)
        else {
            return fail(non_pd.unwrap_or_else(|| LEVELS_NOT_PD.to_string()));
        };
        if let Some(reason) = non_pd {
            return CovStepOutcome {
                matrix: None,
                wall_time_secs: start.elapsed().as_secs_f64(),
                warnings: vec![reason],
                sir_fallback_proposal: Some(full),
                method: None,
            };
        }
        // `rsr`: `R⁻¹SR⁻¹` with `full` as `R⁻¹`, through the unprofiled step's own
        // `combine_covariance`. `full`'s structural rows are `[A_p⁻¹, −A_p⁻¹BD⁻¹]` (NLL
        // scale), so each subject's score `sᵢ = (sᵢθ, sᵢβ)` maps to `A_p⁻¹(sᵢθ − BD⁻¹sᵢβ)`:
        // the structural block is `A_p⁻¹(Σ sᵢᵉᶠᶠsᵢᵉᶠᶠᵀ)A_p⁻¹` over the efficient scores, the
        // θθ block of the unprofiled sandwich. The level blocks are that sandwich's too.
        let (matrix, method) = if options.covariance_method == CovarianceMethod::Sandwich {
            let (s, free_all, _) = match self.score_cross_product(hold, x_hat, solve) {
                Ok(v) => v,
                Err(reason) => return fail(reason),
            };
            let s = crate::estimation::covariance::embed_free(&s, &free_all, n);
            let sandwich = crate::estimation::covariance::combine_covariance(
                CovarianceMethod::Sandwich,
                full,
                &s,
                &[],
            )
            .expect("the sandwich inverts nothing");
            (sandwich, CovarianceMethod::Sandwich)
        } else {
            (full, CovarianceMethod::Hessian)
        };
        if options.verbose {
            eprintln!("  Covariance step successful (profile)");
        }
        CovStepOutcome {
            matrix: Some(matrix),
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: Vec::new(),
            sir_fallback_proposal: None,
            method: Some(method),
        }
    }

    /// The score cross-product `S = Σᵢ sᵢsᵢᵀ` of the unprofiled step
    /// (`assemble_score_cross_product`, the fixed-mode scores at the solve's modes) over
    /// every free packed coordinate, the profiled levels included, with those coordinates
    /// and the template they are free in ([`unheld_template`]): the levels are scored as the
    /// free θ they are in the fit's template, not held.
    fn score_cross_product(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        solve: &ProfileSolve,
    ) -> Result<(DMatrix<f64>, Vec<usize>, ModelParameters), String> {
        let unheld = unheld_template(self.layout, hold);
        let PackedStart { bounds, fixed, .. } = pack_with_bounds(&unheld);
        let free_all: Vec<usize> = (0..x_hat.len()).filter(|&i| !fixed[i]).collect();
        let s = crate::estimation::covariance::assemble_score_cross_product(
            x_hat,
            &unheld,
            self.model,
            self.population,
            &solve.etas,
            &solve.h_matrices,
            &solve.kappas,
            &bounds,
            self.options,
            &free_all,
        )?;
        if crate::cancel::is_cancelled(&self.options.cancel) {
            return Err(crate::estimation::covariance::COV_CANCELLED_MSG.to_string());
        }
        Ok((s, free_all, unheld))
    }

    /// `covariance_method = s` for a profiled fit: `S⁻¹` over every free packed coordinate,
    /// the levels included, as the unprofiled step computes it. Like that step it reads no
    /// `R`, so neither the profile stencil nor `D` and `B` are built, and a rank-deficient
    /// `S` — more free coordinates than subjects, which a level block per study makes
    /// common — is refused rather than floored.
    fn cross_product_covariance(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        solve: &ProfileSolve,
    ) -> Result<DMatrix<f64>, String> {
        let (s, free_all, unheld) = self.score_cross_product(hold, x_hat, solve)?;
        let units = crate::estimation::covariance::free_coordinate_units(
            &unheld,
            x_hat,
            &free_all,
            &self.model.nn_weight_theta_ranges(),
        );
        let inv = crate::estimation::covariance::invert_cross_product(&s, &units)
            .ok_or_else(|| crate::estimation::covariance::S_SINGULAR_MSG.to_string())?;
        if self.options.verbose {
            eprintln!("  Covariance step successful (profile)");
        }
        Ok(crate::estimation::covariance::embed_free(
            &inv,
            &free_all,
            x_hat.len(),
        ))
    }

    /// `D` per group (OFV scale, over the group's coordinates) and `B` (`nf × L`, OFV
    /// scale), summed over subjects, each over the structural free coordinates and its own
    /// levels: a subject reading a global `sum_to_zero`'s dependent costs its own `k`
    /// levels, not the block's (#2135). `D` and `B` are mapped to θ after (`TᵀDT`, `BT`).
    ///
    /// On the adjoint route ([`Self::cov_is_analytic`]) each subject's are exact
    /// ([`Self::subject_blocks_adjoint`]), and so is `A = Σ Aᵢ`; a subject the adjoint
    /// declines, or whose mode may not be one, takes the re-solved differences for all three
    /// ([`Self::subject_blocks_fd`], [`Self::subject_structural_fd`]) and is listed in
    /// [`LevelBlocks::fd_subjects`]. Mixing the two over subjects is sound: each subject's
    /// term is its own. The FD route builds no `A`: its `A_p` is the stencil's.
    fn level_blocks(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
    ) -> Result<LevelBlocks, String> {
        let layout = self.layout;
        let nf = free.len();
        let l = layout.coords.len();
        let (theta_steps, level_steps) = self.block_steps(hold, x_hat, free, solve)?;
        let adjoint = self.cov_is_analytic();
        // A declined subject's `Aᵢ` steps as the profile stencil does (#2414); its `Bᵢ` keeps
        // `B`'s global step on either route (`COV_STEP_SE`). With both at one step, the mixed
        // step's worst correlation-scale gap to the all-exact one on F2 is 1.3e-2 / 1.3e-3 /
        // 1.9e-3 / 3.5e-3 / 1.2e-2 at 1e-2 / 3e-3 / 1e-3 / 3e-4 / 1e-4 (macOS arm64):
        // truncation above `3e-3`, the re-solved differences' noise below. `Bᵢ` at the
        // global `1e-2` instead moves it to 1.6e-3, inside the same band.
        let a_steps: Vec<f64> = free
            .iter()
            .map(|&i| {
                let floor = fd_step_floor(hold, &self.model.default_params, i);
                fd_step(profile_fd_hessian_step(self.options), x_hat[i], floor)
            })
            .collect();
        let pt = adjoint
            .then(|| crate::sens::ode_provider::adjoint::PackedPoint::new(hold, x_hat))
            .flatten();
        #[cfg(test)]
        let forced = FORCE_PROFILE_COV_DECLINE.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let forced: Option<usize> = None;
        // Per subject: (D_i over its levels, B_i nf × its levels, A_i on the adjoint route,
        // whether they are the adjoint's).
        #[allow(clippy::type_complexity)]
        let parts: Vec<(DMatrix<f64>, DMatrix<f64>, Option<DMatrix<f64>>, bool)> =
            (0..self.population.subjects.len())
                .into_par_iter()
                .map(|i| {
                    let exact = pt
                        .as_ref()
                        .filter(|_| !solve.off_mode[i] && forced != Some(i))
                        .and_then(|pt| self.subject_blocks_adjoint(i, pt, free, solve));
                    match exact {
                        Some((a, b, d)) => (d, b, Some(a), true),
                        None => {
                            let (d, b) = self.subject_blocks_fd(
                                i,
                                hold,
                                x_hat,
                                free,
                                solve,
                                &theta_steps,
                                &level_steps,
                            );
                            let a = adjoint.then(|| {
                                self.subject_structural_fd(i, hold, x_hat, free, solve, &a_steps)
                            });
                            (d, b, a, false)
                        }
                    }
                })
                .collect();
        let fd_subjects: Vec<usize> = (0..parts.len()).filter(|&i| !parts[i].3).collect();
        let a = adjoint.then(|| {
            // In subject order, so the sum does not depend on the thread count.
            let mut acc = DMatrix::zeros(nf, nf);
            for (_, _, ai, _) in &parts {
                acc += ai
                    .as_ref()
                    .expect("the adjoint route gives every subject an Aᵢ");
            }
            acc
        });
        if adjoint {
            // The recordings' tape buffers, released on every worker (`analytic_cov_assembly`).
            rayon::broadcast(|_| crate::sens::rtape::rtape_trim(0));
        }
        let mut d_levels: Vec<DMatrix<f64>> = layout
            .groups
            .iter()
            .map(|g| DMatrix::zeros(g.n_levels(), g.n_levels()))
            .collect();
        let mut b_levels: DMatrix<f64> = DMatrix::zeros(nf, layout.n_levels());
        for (i, (di, bi, _, _)) in parts.iter().enumerate() {
            let reads = &layout.reads[i];
            for (a, ra) in reads.iter().enumerate() {
                let (g, la) = layout.level_local(ra.level);
                for (c, rc) in reads.iter().enumerate() {
                    d_levels[g][(la, layout.level_local(rc.level).1)] += di[(a, c)];
                }
                for s in 0..nf {
                    b_levels[(s, ra.level)] += bi[(s, a)];
                }
            }
        }
        // To θ: `D = TᵀD_yT` per group, and `B = B_yT`, a θ moving the dependent that
        // sums it by `−1`.
        let d_groups: Vec<DMatrix<f64>> = layout
            .groups
            .iter()
            .zip(&d_levels)
            .map(|(g, d)| g.theta_matrix(d))
            .collect();
        let b = DMatrix::from_fn(nf, l, |s, pos| {
            let (g, a) = layout.local[pos];
            match layout.groups[g].border_of[a] {
                Some(j) => b_levels[(s, pos)] - b_levels[(s, l + layout.groups[g].borders[j])],
                None => b_levels[(s, pos)],
            }
        });
        if !d_groups.iter().all(|d| d.iter().all(|v| v.is_finite()))
            || !b.iter().all(|v| v.is_finite())
            || !a.iter().flatten().all(|v| v.is_finite())
        {
            return Err(
                "Covariance step failed: the objective is not finite at some \
                 finite-difference point of the profiled levels. SE estimates not available."
                    .to_string(),
            );
        }
        Ok(LevelBlocks {
            d_groups,
            b,
            a,
            fd_subjects,
        })
    }

    /// The FD steps of [`Self::subject_blocks_fd`]: per structural free coordinate the
    /// covariance stencil's, and per level `COV_STEP_SE` of its conditional SE. `Err` when a
    /// level has no curvature to step in.
    fn block_steps(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let theta_steps: Vec<f64> = free
            .iter()
            .map(|&i| {
                let floor = fd_step_floor(hold, &self.model.default_params, i);
                fd_step(self.options.fd_hessian_step, x_hat[i], floor)
            })
            .collect();
        let level_steps: Vec<f64> = solve.level_scale.iter().map(|s| COV_STEP_SE * s).collect();
        let stepped = |s: f64| s.is_finite() && s > 0.0;
        if !solve.se_scale.iter().all(|&s| stepped(s))
            || !self
                .layout
                .reads
                .iter()
                .flatten()
                .all(|r| stepped(level_steps[r.level]))
        {
            return Err(
                "Covariance step failed: a profiled level has no curvature at the estimate. \
                 SE estimates not available."
                    .to_string(),
            );
        }
        Ok((theta_steps, level_steps))
    }

    /// Subject `i`'s `(Dᵢ, Bᵢ)` (OFV scale; `k × k` over its levels and `nf × k`) by central
    /// differences of its term over the structural free coordinates and its own levels, the
    /// EBE re-solved at every point. The route where the adjoint does not serve the
    /// covariance step, and its oracle.
    #[allow(clippy::too_many_arguments)]
    fn subject_blocks_fd(
        &self,
        i: usize,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
        theta_steps: &[f64],
        level_steps: &[f64],
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let nf = free.len();
        let reads = &self.layout.reads[i];
        let k = reads.len();
        let warm = &joint_warm_start(&solve.etas[i], &solve.kappas[i]);
        // A packed point moved along structural coordinates `ds` and the subject's levels
        // `dl`, scored as subject `i`'s term (OFV scale). `x_hat` already carries `β̂` in
        // the level coordinates.
        let at = |ds: &[(usize, f64)], dl: &[(usize, f64)]| -> f64 {
            let mut x = x_hat.to_vec();
            for &(a, d) in ds {
                x[free[a]] += d;
            }
            let p = shifted(&unpack_params(&x, hold), reads, dl);
            2.0 * self.resolve(i, &p, warm).nll
        };
        let f0 = at(&[], &[]);
        let mut d = DMatrix::zeros(k, k);
        for a in 0..k {
            let h = level_steps[reads[a].level];
            d[(a, a)] = (at(&[], &[(a, h)]) - 2.0 * f0 + at(&[], &[(a, -h)])) / (h * h);
            for c in 0..a {
                let g = level_steps[reads[c].level];
                let v = (at(&[], &[(a, h), (c, g)])
                    - at(&[], &[(a, h), (c, -g)])
                    - at(&[], &[(a, -h), (c, g)])
                    + at(&[], &[(a, -h), (c, -g)]))
                    / (4.0 * h * g);
                d[(a, c)] = v;
                d[(c, a)] = v;
            }
        }
        let mut b = DMatrix::zeros(nf, k);
        for s in 0..nf {
            let hs = theta_steps[s];
            for a in 0..k {
                let h = level_steps[reads[a].level];
                b[(s, a)] = (at(&[(s, hs)], &[(a, h)])
                    - at(&[(s, hs)], &[(a, -h)])
                    - at(&[(s, -hs)], &[(a, h)])
                    + at(&[(s, -hs)], &[(a, -h)]))
                    / (4.0 * hs * h);
            }
        }
        (d, b)
    }

    /// Subject `i`'s `Aᵢ` (OFV scale, `nf × nf`) by central differences of its term over the
    /// structural free coordinates, its levels held at `β̂` and its EBE re-solved at every
    /// point, at `steps`: the exact route's `Aᵢ` for a subject the adjoint declines, so
    /// `A_p` stays a Schur complement with that one term on finite differences (#2417
    /// review row 3). `2·nf²` re-solves of one subject, where the population stencil it
    /// replaces re-profiles every subject at each of its points.
    fn subject_structural_fd(
        &self,
        i: usize,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
        steps: &[f64],
    ) -> DMatrix<f64> {
        let nf = free.len();
        let warm = &joint_warm_start(&solve.etas[i], &solve.kappas[i]);
        // `x_hat` carries `β̂` in the level coordinates, which `hold` keeps fixed.
        let at = |ds: &[(usize, f64)]| -> f64 {
            let mut x = x_hat.to_vec();
            for &(a, d) in ds {
                x[free[a]] += d;
            }
            2.0 * self.resolve(i, &unpack_params(&x, hold), warm).nll
        };
        let f0 = at(&[]);
        let mut a = DMatrix::zeros(nf, nf);
        for s in 0..nf {
            let h = steps[s];
            a[(s, s)] = (at(&[(s, h)]) - 2.0 * f0 + at(&[(s, -h)])) / (h * h);
            for t in 0..s {
                let g = steps[t];
                let v = (at(&[(s, h), (t, g)]) - at(&[(s, h), (t, -g)]) - at(&[(s, -h), (t, g)])
                    + at(&[(s, -h), (t, -g)]))
                    / (4.0 * h * g);
                a[(s, t)] = v;
                a[(t, s)] = v;
            }
        }
        a
    }

    /// Subject `i`'s exact `(Aᵢ, Bᵢ, Dᵢ)` (OFV scale; `nf × nf`, `nf × k`, `k × k`) at its
    /// mode in `solve` (#2109): one block of the adjoint's covariance Hessian
    /// `∂²F̂ᵢ = Uᵀ∇²Ψ U`, the EBE's response in it, over the structural free coordinates and
    /// every θ the subject's level `moves` touch. With `M` the moves (θ touched × levels),
    /// `Dᵢ = MᵀH_LL M` and `Bᵢ = H_SL M`, the quantities the re-solved differences of
    /// [`Self::subject_blocks_fd`] take, a slack's `−1` included; `Aᵢ = H_SS`.
    ///
    /// `None` where the recorder declines the subject (its scope or width, a mode at which
    /// `H` is singular, a non-finite block) or a level it reads log-packs.
    fn subject_blocks_adjoint(
        &self,
        i: usize,
        pt: &crate::sens::ode_provider::adjoint::PackedPoint<'_>,
        free: &[usize],
        solve: &ProfileSolve,
    ) -> Option<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> {
        use crate::sens::ode_provider::adjoint::subject_adjoint_cov_block_at;
        if self.reads_a_log_packed_level(i, pt.template) {
            return None;
        }
        let subject = &self.population.subjects[i];
        let mode = crate::estimation::covariance::stacked_mode(
            self.model,
            subject,
            &solve.etas[i],
            Some(&solve.kappas[i]),
        )?;
        let reads = &self.layout.reads[i];
        let mut touched: Vec<usize> = reads
            .iter()
            .flat_map(|r| r.moves.iter().map(|&(k, _)| k))
            .collect();
        touched.sort_unstable();
        touched.dedup();
        let nf = free.len();
        let axes: Vec<usize> = free.iter().chain(&touched).copied().collect();
        let h =
            subject_adjoint_cov_block_at(self.model, subject, pt, &mode, self.objective(), &axes)?;
        let m = DMatrix::from_fn(touched.len(), reads.len(), |t, a| {
            reads[a]
                .moves
                .iter()
                .filter(|&&(k, _)| k == touched[t])
                .map(|&(_, c)| c)
                .sum()
        });
        let (nt, k) = (touched.len(), reads.len());
        // ×2: the per-subject NLL to the OFV (see `analytic_cov_assembly`).
        let a = h.view((0, 0), (nf, nf)) * 2.0;
        let b = h.view((0, nf), (nf, nt)) * &m * 2.0;
        let d = m.transpose() * h.view((nf, nf), (nt, nt)) * &m * 2.0;
        debug_assert_eq!(d.shape(), (k, k));
        let finite = |x: &DMatrix<f64>| x.iter().all(|v| v.is_finite());
        (finite(&a) && finite(&b) && finite(&d)).then_some((a, b, d))
    }
}

/// [`ProfileCtx::level_blocks`]' result.
pub(crate) struct LevelBlocks {
    /// `D` per group, over the group's θ (OFV scale).
    pub(crate) d_groups: Vec<DMatrix<f64>>,
    /// `B`, structural free × the layout's coordinates (OFV scale).
    pub(crate) b: DMatrix<f64>,
    /// `A = Σ Aᵢ` over the structural free coordinates (OFV scale) on the adjoint route,
    /// each `Aᵢ` the adjoint's or, for a subject in `fd_subjects`, its re-solved
    /// differences. `None` on the FD route.
    pub(crate) a: Option<DMatrix<f64>>,
    /// The subjects (population indices) whose `Dᵢ` and `Bᵢ` took the re-solved
    /// differences: on the FD route every subject, which only the adjoint route reports.
    pub(crate) fd_subjects: Vec<usize>,
}

/// The exact Hessian of the profile objective over the structural coordinates at `β̂`:
/// `A_p = A − BD⁻¹Bᵀ`, with `D` block diagonal over the layout's groups. `None` when a
/// group's `D` is not positive definite.
fn schur_profile_hessian(
    layout: &ProfileLayout,
    a: &DMatrix<f64>,
    d_groups: &[DMatrix<f64>],
    b: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    let nf = a.nrows();
    let mut a_p = a.clone();
    for (g, d) in layout.groups.iter().zip(d_groups) {
        let ch = d.clone().cholesky()?;
        let bg = DMatrix::from_fn(nf, g.coords.len(), |s, c| b[(s, g.coords[c])]);
        a_p -= &bg * ch.solve(&bg.transpose());
    }
    Some(a_p.symmetric_part())
}

/// The covariance step's warning for subjects whose Hessian blocks fell back to finite
/// differences on the adjoint route. `None` when there are none.
pub(crate) fn profile_fd_level_blocks_warning(
    population: &Population,
    subjects: &[usize],
) -> Option<String> {
    let (count, example) = fd_fallback_subjects(population, *subjects.first()?, subjects.len());
    Some(format!(
        "{count} could not be given the exact analytic Hessian of their objective terms in the \
         covariance step{example}. Those terms' Hessians used reconverged finite differences, \
         which are correct but slower."
    ))
}

const PROFILE_NOT_FINITE_AT_ESTIMATE: &str =
    "Covariance step failed: the profile objective is not finite at the estimate. \
     SE estimates not available.";

const LEVELS_NOT_PD: &str = "Covariance step failed: the Hessian of the objective in the \
     profiled levels is not positive definite. SE estimates not available.";

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
