//! The exact per-subject covariance Hessian `∂²F̂/∂x²` of the profile objective
//! `F̂(x) = F(x, η̂(x))`, by the gradient's identity one order up.
//!
//! `η̂` is the root of `g = ∇_η l`. For a fixed `w`, `F̂(x) = Ψ_w(x, η̂(x))` with
//! `Ψ_w = F − wᵀg`, since `g` vanishes along the mode. Differentiating twice,
//!
//! ```text
//!   ∇²F̂ = Uᵀ ∇²Ψ_w U + Σ_k (∂Ψ_w/∂η_k) η̂″_k,     U = [I; η′],  η′ = dη̂/dx = −H⁻¹ ∂ₓg,
//! ```
//!
//! and `w = H⁻¹∇_η F` (the gradient's `w`) zeroes `∂Ψ_w/∂η`, so the second-order mode
//! response never appears: the Hessian is the second derivative of one scalar along the
//! mode's tangent directions `u_a = (e_a, η′e_a)`.
//!
//! Everything comes from one primitive, the subject's **covariance plan** ([`CovPlan`]): the
//! objective `F` (FOCE, FOCEI, or the AGQ marginal under either anchor) and `g`, through the
//! same recorders and kernel as the gradient, recorded once on the recording type with every
//! packed coordinate, η (and, under [`NnReduction`], the networks' raw outputs) an input slot
//! (#2786). The predictions carry an inner η-dual over the recording type, so `G` (and under
//! the exact anchor `Q`) are values the kernel reads.
//!
//! - The plan's value forward and one reverse seeded with each output give the gradients of
//!   `F` and of every `g_k` with respect to every input: `∇_η F`, `H` and `∂g` along any
//!   direction at once (`mode_response`).
//! - Per pass, the plan's forward along the pass's directions `u_a` (its tangent layout) and
//!   one forward-over-reverse sweep seeded with `Ψ_w` on the value lanes give, on the
//!   adjoints' tangent lanes, their Hessian columns `∇_v(∂Ψ_w/∂u_a) = ∇²Ψ_w u_a`, contracted
//!   with every `u_b` (`psi_along`). A structured subject ([`st`]) takes one pass of every
//!   direction; the recorded kernel one pass per direction ([`pass_width`], #2980).
//!
//! There are no third-order sensitivity tables and no finite differences, and FOCE's
//! frozen-variance mode (whose `R⁰` moves with θ through `f(η = 0)`) is just another term
//! of `g`.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::estimation::parameterization::{packed_segments, rho_packed_start, theta_packs_log};
use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
use crate::sens::focei_kernel::{chol_logdet, focei_htilde, OmegaT};
use crate::sens::num::vm::Layout;
use crate::sens::num::{log_sum_exp, PkNum};

#[path = "ode_adjoint_hessian_basis.rs"]
pub(super) mod basis;
use basis::{Basis, Lanes, Moved, MuFold, PlanKey, PlanRun};
#[path = "ode_adjoint_hessian_struct.rs"]
pub(super) mod st;

/// A direction of a pass: the packed coordinate it moves, the `[covariate_nn]` raw output
/// it moves (the reduced basis, [`NnReduction`]), and its η tangent (empty: none).
#[derive(Clone)]
pub(super) struct Dir {
    x: Option<usize>,
    z: Option<usize>,
    eta: Vec<f64>,
}

impl Dir {
    fn x(i: usize, eta: Vec<f64>) -> Self {
        Dir { x: Some(i), z: None, eta }
    }

    #[cfg(feature = "nn")]
    fn eta(k: usize, n: usize) -> Self {
        Dir { x: None, z: None, eta: (0..n).map(|m| if m == k { 1.0 } else { 0.0 }).collect() }
    }

    #[cfg(feature = "nn")]
    fn z(k: usize) -> Self {
        Dir { x: None, z: Some(k), eta: Vec::new() }
    }
}

/// The exact per-subject covariance Hessian of `objective` in packed coordinates
/// (`pack_params` order; the per-subject NLL Hessian, which the caller scales by 2 to the
/// OFV), or `None` outside the adjoint's scope. `eta_hat` is the mode (the stacked
/// `[η, κ]` under IOV) for `unpack_params(x)`.
#[cfg(test)]
pub(crate) fn subject_adjoint_cov_hessian(
    model: &CompiledModel, subject: &Subject, template: &ModelParameters, x: &[f64],
    eta_hat: &[f64], objective: Objective,
) -> Option<DMatrix<f64>> {
    let pt = PackedPoint::new(template, x)?;
    subject_adjoint_cov_hessian_at(model, subject, &pt, eta_hat, objective)
}

/// `subject_adjoint_cov_hessian` (the per-subject entry point, which unpacks its own `x`) at a [`PackedPoint`] the caller unpacked once for its
/// population.
pub(crate) fn subject_adjoint_cov_hessian_at(
    model: &CompiledModel, subject: &Subject, pt: &PackedPoint<'_>, eta_hat: &[f64],
    objective: Objective,
) -> Option<DMatrix<f64>> {
    match cov_hessian(model, subject, pt, eta_hat, objective, HessianSel::Full)? {
        HessianOut::Full(h) => Some(h),
        HessianOut::Diag(_) | HessianOut::Block(..) => None,
    }
}

/// The diagonal of [`subject_adjoint_cov_hessian_at`] along the packed `axes` (in `axes`
/// order): `∂²Fᵢ/∂x_a²` for each `a`, at `axes.len()` directions instead of `x.len()` and
/// without the dense block — what the #2159 identity-scale pre-flight reads its candidates'
/// curvature from. `None` outside the adjoint's scope or for an axis past `x`.
pub(crate) fn subject_adjoint_cov_diag_at(
    model: &CompiledModel, subject: &Subject, pt: &PackedPoint<'_>, eta_hat: &[f64],
    objective: Objective, axes: &[usize],
) -> Option<Vec<f64>> {
    if axes.iter().any(|&i| i >= pt.x.len()) {
        return None;
    }
    match cov_hessian(model, subject, pt, eta_hat, objective, HessianSel::Diag(axes))? {
        HessianOut::Diag(d) => Some(d),
        HessianOut::Full(_) | HessianOut::Block(..) => None,
    }
}

/// The dense sub-block of [`subject_adjoint_cov_hessian_at`] over the packed `axes` (rows and
/// columns in `axes` order): `∂²Fᵢ/∂x_a∂x_b` for `a, b ∈ axes`, at `axes.len()` directions
/// instead of `x.len()` — what a profiled fit's covariance step reads its structural and
/// level blocks from (#2109). `None` outside the adjoint's scope or for an axis past `x`.
pub(crate) fn subject_adjoint_cov_block_at(
    model: &CompiledModel, subject: &Subject, pt: &PackedPoint<'_>, eta_hat: &[f64],
    objective: Objective, axes: &[usize],
) -> Option<DMatrix<f64>> {
    subject_adjoint_cov_block_with_gradient_at(model, subject, pt, eta_hat, objective, axes)
        .map(|(_, h)| h)
}

/// [`subject_adjoint_cov_block_at`] and, from the same pass, the total packed gradient along
/// `axes` (`dFᵢ/dx_a`, in `axes` order): the `∇Ψ_w` the block's sweeps form anyway. A caller
/// that needs both (a log-packed coordinate's second derivative takes `−∂/∂x` on its diagonal)
/// records one pass, not a Hessian and a second gradient.
pub(crate) fn subject_adjoint_cov_block_with_gradient_at(
    model: &CompiledModel, subject: &Subject, pt: &PackedPoint<'_>, eta_hat: &[f64],
    objective: Objective, axes: &[usize],
) -> Option<(Vec<f64>, DMatrix<f64>)> {
    if axes.iter().any(|&i| i >= pt.x.len()) {
        return None;
    }
    match cov_hessian(model, subject, pt, eta_hat, objective, HessianSel::Block(axes))? {
        HessianOut::Block(g, h) => Some((g, h)),
        HessianOut::Full(_) | HessianOut::Diag(_) => None,
    }
}

/// Which part of the covariance Hessian a pass assembles.
#[derive(Clone, Copy)]
enum HessianSel<'a> {
    /// The full `x.len()`-square matrix.
    Full,
    /// The diagonal entries along these packed axes.
    Diag(&'a [usize]),
    /// The dense block over these packed axes.
    Block(&'a [usize]),
}

/// A covariance Hessian, the diagonal a [`HessianSel::Diag`] asked for, or the block a
/// [`HessianSel::Block`] asked for with the total gradient along its axes.
enum HessianOut {
    Full(DMatrix<f64>),
    Diag(Vec<f64>),
    Block(Vec<f64>, DMatrix<f64>),
}

impl HessianOut {
    fn all_finite(&self) -> bool {
        match self {
            HessianOut::Full(h) => h.iter().all(|v| v.is_finite()),
            HessianOut::Diag(d) => d.iter().all(|v| v.is_finite()),
            HessianOut::Block(g, h) => {
                g.iter().all(|v| v.is_finite()) && h.iter().all(|v| v.is_finite())
            }
        }
    }
}

/// The dispatch shared by [`subject_adjoint_cov_hessian_at`] and
/// [`subject_adjoint_cov_diag_at`].
fn cov_hessian(
    model: &CompiledModel, subject: &Subject, pt: &PackedPoint<'_>, eta_hat: &[f64],
    objective: Objective, sel: HessianSel<'_>,
) -> Option<HessianOut> {
    if super::marginal_refused(model, subject, objective) {
        // The refusal sentinel is constant: its Hessian is the zero matrix.
        let n = pt.x.len();
        return match sel {
            HessianSel::Full => Some(HessianOut::Full(DMatrix::zeros(n, n))),
            HessianSel::Diag(axes) => Some(HessianOut::Diag(vec![0.0; axes.len()])),
            HessianSel::Block(axes) => Some(HessianOut::Block(
                vec![0.0; axes.len()],
                DMatrix::zeros(axes.len(), axes.len()),
            )),
        };
    }
    let sc = super::scope(model, subject, pt, eta_hat)?;
    hessian_with(&sc, subject, pt.template, pt.x, objective, sel)
}

/// The exact mode response `dη̂/dx` (one `n_eta`-vector, stacked `[η, κ]` under IOV, per
/// packed coordinate) at the EBE `eta_hat` of `unpack_params(x)`: the EBE warm-start
/// predictor's Jacobian (Almquist Eq. 48), FOCE's frozen-variance mode included.
#[cfg(test)]
pub(crate) fn subject_adjoint_eta_dx(
    model: &CompiledModel, subject: &Subject, template: &ModelParameters, x: &[f64],
    eta_hat: &[f64], interaction: bool,
) -> Option<Vec<DVector<f64>>> {
    let pt = PackedPoint::new(template, x)?;
    subject_adjoint_eta_dx_at(model, subject, &pt, eta_hat, interaction)
}

/// `subject_adjoint_eta_dx` (the per-subject entry point, which unpacks its own `x`) at a [`PackedPoint`].
#[cfg(test)]
fn subject_adjoint_eta_dx_at(
    model: &CompiledModel, subject: &Subject, pt: &PackedPoint<'_>, eta_hat: &[f64],
    interaction: bool,
) -> Option<Vec<DVector<f64>>> {
    let (template, x) = (pt.template, pt.x);
    let objective = if interaction { Objective::Focei } else { Objective::Foce };
    let sc = super::scope(model, subject, pt, eta_hat)?;
    let ctx = LikCtx::new(sc.model, subject, sc.params);
    let so = stacked_omega(sc.model, subject, template, sc.params, sc.width);
    let plan = CovPlan::new(&ctx, &sc, subject, template, x, objective, &so, None);
    let (_, eta_x) = mode_response(&plan, &x_dirs(x.len()))?;
    let cols: Vec<DVector<f64>> =
        (0..x.len()).map(|i| DVector::from_fn(sc.width, |k, _| eta_x[(k, i)])).collect();
    cols.iter().all(|c| c.iter().all(|v| v.is_finite())).then_some(cols)
}

/// Whether the profile Hessian serves a model's non-Gaussian endpoints (TTE, binary, CTMM)
/// under every objective (FOCE/FOCEI, Laplace and AGQ at either anchor, IOV included): every
/// endpoint with an exact channel. The model-level
/// clause the covariance step's decline walk reads; a subject can still bail at run time.
pub(crate) fn cov_non_gaussian_served(model: &CompiledModel, objective: Objective) -> bool {
    cfg!(feature = "survival")
        && match objective {
            Objective::Foce | Objective::Focei | Objective::Agq { .. } => true,
        }
        && crate::estimation::inner_optimizer::non_gaussian_endpoints_served(model)
}

/// The packed axes as pass directions, without an η tangent.
#[cfg(test)]
fn x_dirs(n: usize) -> Vec<Dir> {
    (0..n).map(|i| Dir::x(i, Vec::new())).collect()
}

/// The mode's first-order response at the scoped point: `w = H⁻¹∇_η F` and the mode's
/// response `−H⁻¹∂g` along each of `along` (`N × along.len()`; `η′ = dη̂/dx` along the
/// packed axes), from the plan's value forward and one reverse seeded with each output
/// (`F`, `g_k`).
fn mode_response(plan: &CovPlan<'_>, along: &[Dir]) -> Option<(Vec<f64>, DMatrix<f64>)> {
    let n = plan.pn();
    // ∇_η F, H = ∂g/∂η and ∂g along `along`: from the structured kernel where it serves, else
    // from the recorded composition's inputs' adjoints.
    let (grad_f, h, gx) = match plan.struct_mode(along) {
        Some((g, h, gx)) => (DVector::from_vec(g), h, gx),
        None => {
            let run = plan.run(&[], 0)?;
            let unit = |o: usize| -> Vec<f64> {
                (0..=n).map(|c| if c == o { 1.0 } else { 0.0 }).collect()
            };
            let rows = plan.reverse(&run, &(0..=n).map(unit).collect::<Vec<_>>())?;
            (
                DVector::from_fn(n, |k, _| rows[0][plan.eta_slot(k)]),
                DMatrix::from_fn(n, n, |o, k| rows[1 + o][plan.eta_slot(k)]),
                DMatrix::from_fn(n, along.len(), |o, j| plan.along(&rows[1 + o], &along[j])),
            )
        }
    };
    // Nonsingular, not necessarily definite: see `mode_solve`.
    let hs = h.symmetric_part();
    let solve: Box<dyn Fn(&DMatrix<f64>) -> Option<DMatrix<f64>>> = match hs.clone().cholesky() {
        Some(c) => Box::new(move |b| Some(c.solve(b))),
        None => {
            let lu = hs.full_piv_lu();
            if !lu.is_invertible() {
                return None;
            }
            Box::new(move |b| lu.solve(b))
        }
    };
    let wv = solve(&DMatrix::from_column_slice(n, 1, grad_f.as_slice()))?;
    let eta_x = -solve(&gx)?;
    (wv.iter().chain(eta_x.iter()).all(|v| v.is_finite()))
        .then(|| (wv.iter().copied().collect(), eta_x))
}

/// Directions per pass of `n_dirs`. A structured subject's pass carries all of them: its
/// compiled kernels are the reduce's, one tangent per run, and the plans' bases, run once
/// whatever the directions. A recorded kernel's pass carries one: that kernel's code grows with
/// the width a pass carries while its arithmetic does not (#2980).
fn pass_width(structured: bool, n_dirs: usize) -> usize {
    #[cfg(test)]
    if let Some(w) = PASS_WIDTH.with(std::cell::Cell::get) {
        return w.max(1);
    }
    if structured {
        // A layout carries at most `u8::MAX` tangents (`Layout::tan`).
        n_dirs.clamp(1, usize::from(u8::MAX))
    } else {
        1
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: every pass carries exactly this many directions (the last one fewer).
    pub(crate) static PASS_WIDTH: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn hessian_with(
    sc: &Scoped, subject: &Subject, template: &ModelParameters, x: &[f64], objective: Objective,
    sel: HessianSel<'_>,
) -> Option<HessianOut> {
    let (model, params) = (sc.model, sc.params);
    let n = x.len();
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega(model, subject, template, params, sc.width);
    // A block's axes are few, so the reduced basis buys nothing: it goes along them.
    #[cfg(feature = "nn")]
    if let Some(red) = NnReduction::build(model, subject, template, &params.theta)
        .filter(|_| !nn_reduce_off() && !matches!(sel, HessianSel::Block(_)))
    {
        let plan =
            CovPlan::new(&ctx, sc, subject, template, x, objective, &so, Some(red.z.as_slice()));
        if let Some(h) = red.hessian(model, &plan, template, x, sc.width, sel) {
            #[cfg(test)]
            NN_REDUCED_HITS.with(|c| c.set(c.get() + 1));
            return Some(h);
        }
    }
    let plan = CovPlan::new(&ctx, sc, subject, template, x, objective, &so, None);
    let axes: Vec<usize> = match sel {
        HessianSel::Full => (0..n).collect(),
        HessianSel::Diag(axes) | HessianSel::Block(axes) => axes.to_vec(),
    };
    let along: Vec<Dir> = axes.iter().map(|&i| Dir::x(i, Vec::new())).collect();
    let (w, eta_x) = mode_response(&plan, &along)?;
    // Uᵀ∇²Ψ U along u_a = (e_a, η′e_a).
    let dirs: Vec<Dir> = axes
        .iter()
        .enumerate()
        .map(|(c, &i)| Dir::x(i, (0..sc.width).map(|k| eta_x[(k, c)]).collect()))
        .collect();
    let out = match sel {
        HessianSel::Full => HessianOut::Full(psi_along(&plan, &w, &dirs)?.1),
        HessianSel::Block(_) => {
            let (g, h) = psi_along(&plan, &w, &dirs)?;
            HessianOut::Block(g.as_slice().to_vec(), h)
        }
        HessianSel::Diag(_) => HessianOut::Diag(psi_diag(&plan, &w, &dirs)?),
    };
    out.all_finite().then_some(out)
}

/// `∇Ψ_w` and `∇²Ψ_w` along `dirs`, `Ψ_w = F − wᵀg`, by forward over reverse: per pass, the
/// directions as the forward tangents, and one sweep of `Ψ_w` whose adjoints' tangent `a` is
/// the column `∇²Ψ_w u_a`, read along every direction. Symmetrised: the two triangles agree
/// to rounding.
fn psi_along(plan: &CovPlan<'_>, w: &[f64], dirs: &[Dir]) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let n = dirs.len();
    let mut grad = DVector::<f64>::zeros(n);
    let mut hess = DMatrix::<f64>::zeros(n, n);
    psi_lanes(plan, w, dirs, |i, g, along| {
        grad[i] = g;
        for (j, d) in dirs.iter().enumerate() {
            hess[(j, i)] = along(d);
        }
    })?;
    Some((grad, hess.symmetric_part()))
}

/// The diagonal of [`psi_along`]'s `∇²Ψ_w` along `dirs`: each lane read along its own
/// direction only, `O(n)` reads instead of `O(n²)` and no `n × n` matrix.
fn psi_diag(plan: &CovPlan<'_>, w: &[f64], dirs: &[Dir]) -> Option<Vec<f64>> {
    let mut diag = vec![0.0; dirs.len()];
    psi_lanes(plan, w, dirs, |i, _, along| diag[i] = along(&dirs[i]))?;
    Some(diag)
}

/// The passes [`psi_along`] and [`psi_diag`] share: per pass [`pass_width`] of `dirs` as the
/// plan's tangents and one forward-over-reverse sweep of `Ψ_w = F − wᵀg` (under the test
/// switch, a reverse per direction), then `lane(i, ∂Ψ_w/∂u_i, along)` for each direction `i`
/// in the pass, where `along(d)` reads `d`'s component of the column `∇²Ψ_w u_i`.
fn psi_lanes(
    plan: &CovPlan<'_>, w: &[f64], dirs: &[Dir],
    mut lane: impl FnMut(usize, f64, &dyn Fn(&Dir) -> f64),
) -> Option<()> {
    let declined = plan.st_declined.get();
    match psi_passes(plan, w, dirs, &mut lane) {
        // The structured form did not hold at this point ([`CovPlan::run`]): every pass again,
        // on the recorded kernels.
        None if !declined && plan.st_declined.get() => psi_passes(plan, w, dirs, &mut lane),
        r => r,
    }
}

/// [`psi_lanes`]' passes at the width the plan's kernel takes.
fn psi_passes(
    plan: &CovPlan<'_>, w: &[f64], dirs: &[Dir],
    lane: &mut impl FnMut(usize, f64, &dyn Fn(&Dir) -> f64),
) -> Option<()> {
    let n = plan.pn();
    let width = pass_width(plan.structured(), dirs.len());
    let l = 1 + width;
    plan.prepare(dirs);
    for (c, chunk) in dirs.chunks(width).enumerate() {
        let run = plan.run(chunk, width)?;
        let out = run.out();
        let rows = if reverse_per_direction() {
            // Seed `a`: `F`'s tangent lane `a` and `−w_k` on each `g_k`'s.
            let seeds: Vec<Vec<f64>> = (0..chunk.len())
                .map(|a| {
                    let mut s = vec![0.0; (1 + n) * l];
                    s[1 + a] = 1.0;
                    for k in 0..n {
                        s[(1 + k) * l + 1 + a] = -w[k];
                    }
                    s
                })
                .collect();
            plan.reverse(&run, &seeds)?
        } else {
            // One sweep: `Ψ_w`'s seed on the value lanes, every direction's column on the
            // adjoints' tangent lanes.
            let mut s = vec![0.0; (1 + n) * l];
            s[0] = 1.0;
            for k in 0..n {
                s[(1 + k) * l] = -w[k];
            }
            plan.reverse_tangent(&run, &s)?
        };
        for a in 0..chunk.len() {
            let psi_a = match run.psi.borrow().as_ref() {
                Some(p) => p[a],
                None => out[1 + a] - (0..n).map(|k| w[k] * out[(1 + k) * l + 1 + a]).sum::<f64>(),
            };
            lane(c * width + a, psi_a, &|d: &Dir| plan.along(&rows[a], d));
        }
    }
    Some(())
}

#[cfg(test)]
thread_local! {
    /// Test switch (#2980): the columns by a reverse per direction (reverse over forward), the
    /// forward-over-reverse sweep's oracle.
    pub(crate) static REVERSE_PER_DIRECTION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

fn reverse_per_direction() -> bool {
    #[cfg(test)]
    return REVERSE_PER_DIRECTION.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

#[cfg(all(test, feature = "nn"))]
thread_local! {
    /// Covariance Hessians assembled in [`NnReduction`]'s basis on this thread.
    pub(crate) static NN_REDUCED_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Test switch: assemble every Hessian along the packed axes.
    pub(crate) static NN_REDUCE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "nn")]
fn nn_reduce_off() -> bool {
    #[cfg(test)]
    return NN_REDUCE_OFF.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// The covariance Hessian of a `[covariate_nn]` subject in a reduced basis.
///
/// The network weights reach the likelihood only through the networks' raw outputs `z`
/// (`nn_output_chain_supported`), and with subject-static inputs `z` depends on the weights
/// alone. So with `v = (x_r, x_w, η)` (the other packed coordinates, the weights, the random
/// effects) and `Ψ(v) = Φ(x_r, z(x_w), η)`,
///
/// ```text
///   Uᵀ∇²Ψ U = Cᵀ ∇²Φ_B C + Σ_k (∂Φ/∂z_k) ∇²z_k      (on the weight block)
/// ```
///
/// where `B` is the basis `{(e_r, η′e_r)} ∪ {e_{z_k}} ∪ {e_{η_j}}` and `C` writes each packed
/// direction in it: `e_r` itself, a weight `a` as `Σ_k J_z[k, a] e_{z_k} + Σ_j η′[j, a] e_{η_j}`
/// with `η′` along `a` equal to `η′_z J_z[:, a]`. The passes through the recorders are then over
/// `|B| = n_r + n_z + N` directions instead of `n` (16 against 147 on the 141-weight
/// `warfarin_dcm`), and the weights' own curvature comes from the networks alone.
///
#[cfg(feature = "nn")]
pub(super) struct NnReduction<'s> {
    /// Raw outputs per network at the current weights.
    pub(super) z: Vec<Vec<f64>>,
    /// Per network: its first packed weight coordinate, its weight count, and its inputs.
    pub(super) nets: Vec<(usize, usize, &'s std::collections::HashMap<String, f64>)>,
}

#[cfg(feature = "nn")]
impl<'s> NnReduction<'s> {
    /// `None` when the reduction does not hold (a weight read directly, an input that
    /// varies over the subject's records) or does not pay (fewer weights than outputs).
    pub(super) fn build(
        model: &CompiledModel, subject: &'s Subject, template: &ModelParameters, theta: &[f64],
    ) -> Option<Self> {
        if model.covariate_nns.is_empty()
            || !crate::sens::provider::nn_output_chain_supported(model)
        {
            return None;
        }
        let mut z = Vec::new();
        let mut nets = Vec::new();
        for nn in &model.covariate_nns {
            let n_w = nn.mapper.mlp().n_weights();
            let end = nn.weights_offset + n_w;
            if end > theta.len() || end > template.theta.len() {
                return None;
            }
            let names = nn.mapper.input_names();
            let rep = crate::estimation::nn_theta_gradient::static_nn_input_map(subject, names)?;
            // Every recording reads the subject-static map or a per-record one; both must
            // give these outputs.
            if names.iter().any(|k| rep.get(k) != subject.covariates.get(k)) {
                return None;
            }
            z.push(nn.mapper.forward_raw(&theta[nn.weights_offset..end], rep).ok()?);
            nets.push((nn.weights_offset, n_w, rep));
        }
        let n_w: usize = nets.iter().map(|n| n.1).sum();
        let n_z: usize = z.iter().map(Vec::len).sum();
        (n_w > n_z).then_some(NnReduction { z, nets })
    }

    fn hessian(
        &self, model: &CompiledModel, plan: &CovPlan<'_>, template: &ModelParameters, x: &[f64],
        width: usize, sel: HessianSel<'_>,
    ) -> Option<HessianOut> {
        let n = x.len();
        let n_z: usize = self.z.iter().map(Vec::len).sum();
        let mut is_w = vec![false; n];
        for &(off, n_w, _) in &self.nets {
            is_w[off..off + n_w].iter_mut().for_each(|b| *b = true);
        }
        let rest: Vec<usize> = (0..n).filter(|&i| !is_w[i]).collect();
        // The networks: J_z = ∂z/∂x_w and ∇²z_k, per network over its own weights.
        let (jz, hz) = self.network_jets(model, plan.subject, template, x)?;

        // The mode's response along the rest axes and the outputs.
        let along: Vec<Dir> =
            rest.iter().map(|&i| Dir::x(i, Vec::new())).chain((0..n_z).map(Dir::z)).collect();
        let (w, resp) = mode_response(plan, &along)?;
        let basis: Vec<Dir> = rest
            .iter()
            .enumerate()
            .map(|(r, &i)| Dir::x(i, (0..width).map(|k| resp[(k, r)]).collect()))
            .chain((0..n_z).map(Dir::z))
            .chain((0..width).map(|k| Dir::eta(k, width)))
            .collect();
        let (gb, hb) = psi_along(plan, &w, &basis)?;

        // C: each packed axis in the basis.
        let (nr, nb) = (rest.len(), basis.len());
        let mut c = DMatrix::<f64>::zeros(nb, n);
        for (r, &i) in rest.iter().enumerate() {
            c[(r, i)] = 1.0;
        }
        for (a, ja) in &jz {
            for (k, &d) in ja.iter().enumerate() {
                c[(nr + k, *a)] = d;
            }
            for j in 0..width {
                c[(nr + n_z + j, *a)] = (0..n_z).map(|k| resp[(j, nr + k)] * ja[k]).sum();
            }
        }
        let out = match sel {
            HessianSel::Full => {
                let mut hess = c.transpose() * hb * &c;
                for (k, net_h) in hz.iter().enumerate() {
                    let gk = gb[nr + k];
                    for &((a, b), v) in net_h {
                        hess[(a, b)] += gk * v;
                    }
                }
                HessianOut::Full(hess.symmetric_part())
            }
            HessianSel::Block(_) => unreachable!("a block is assembled along its own axes"),
            // `cᵢᵀ H_B cᵢ` per axis plus the networks' own curvature on the diagonal, with no
            // `n × n` product.
            HessianSel::Diag(axes) => {
                let mut own = vec![0.0; n];
                for (k, net_h) in hz.iter().enumerate() {
                    let gk = gb[nr + k];
                    for &((a, b), v) in net_h {
                        if a == b {
                            own[a] += gk * v;
                        }
                    }
                }
                let hbs = hb.symmetric_part();
                HessianOut::Diag(
                    axes.iter()
                        .map(|&a| {
                            let ca = c.column(a);
                            (ca.transpose() * &hbs * ca)[(0, 0)] + own[a]
                        })
                        .collect(),
                )
            }
        };
        out.all_finite().then_some(out)
    }

    /// `J_z` as `(packed weight axis, ∂z/∂x_a over all outputs)` and per output `k` the
    /// nonzero `∂²z_k/∂x_a∂x_b` over its network's weights, through θ's packing chain.
    #[allow(clippy::type_complexity)]
    fn network_jets(
        &self, model: &CompiledModel, subject: &Subject, template: &ModelParameters, x: &[f64],
    ) -> Option<(Vec<(usize, Vec<f64>)>, Vec<Vec<((usize, usize), f64)>>)> {
        let n_z: usize = self.z.iter().map(Vec::len).sum();
        let mut jz: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut hz: Vec<Vec<((usize, usize), f64)>> = vec![Vec::new(); n_z];
        let mut z0 = 0;
        let n_th = template.theta.len();
        for (net, &(off, n_w, ref cov)) in self.nets.iter().enumerate() {
            let nn = &self.z[net];
            let mlp_out = nn.len();
            let mut jcols: Vec<Vec<f64>> = vec![vec![0.0; n_z]; n_w];
            // The network as a plan over θ's packed coordinates (#2786): its outputs' rows from
            // one reverse per output, and per output the Hessian columns along the weights in
            // passes of [`pass_width`] (a recorded kernel's), one forward-over-reverse sweep per
            // pass (as `psi_lanes`, #2980).
            let record = |lv: &[kj::Tr]| -> Option<Vec<kj::Tr>> {
                let th = unpack_theta(template, |m| lv[m]);
                let out = model.covariate_nns[net].mapper.forward_raw_g(&th[off..off + n_w], cov);
                (out.len() == mlp_out).then_some(out)
            };
            let variant = super::kernel::variant_of(&(7u8, net, off, n_w));
            let width = pass_width(false, n_w);
            let run = |dirs: &[usize]| {
                let (layout, l) = if dirs.is_empty() {
                    (Layout::F64, 1)
                } else {
                    (Layout { tan: width as u8, ..Layout::F64 }, 1 + width)
                };
                let mut inputs = vec![0.0; n_th * l];
                for (m, &v) in x[..n_th].iter().enumerate() {
                    inputs[m * l] = v;
                }
                for (p, &i) in dirs.iter().enumerate() {
                    inputs[i * l + 1 + p] = 1.0;
                }
                super::kernel::kernel_run(
                    model,
                    subject,
                    variant.clone(),
                    layout,
                    &inputs,
                    mlp_out,
                    &record,
                )
            };
            let kr = run(&[])?;
            let unit = |o: usize| -> Vec<f64> {
                (0..mlp_out).map(|c| if c == o { 1.0 } else { 0.0 }).collect()
            };
            let rows = kr.reverse_values(&(0..mlp_out).map(unit).collect::<Vec<_>>())?;
            for (o, row) in rows.iter().enumerate() {
                for i in off..off + n_w {
                    jcols[i - off][z0 + o] = row[i];
                }
            }
            let l = 1 + width;
            for blk in (off..off + n_w).collect::<Vec<_>>().chunks(width) {
                let kr = run(blk)?;
                // One forward-over-reverse sweep, a seed per output on its value lane: tangent
                // `p` of each weight's adjoint in row `o` is output `o`'s column along `blk[p]`.
                let rows = if reverse_per_direction() {
                    None
                } else {
                    let mut seeds = vec![0.0; mlp_out * mlp_out * l];
                    for o in 0..mlp_out {
                        seeds[o * mlp_out * l + o * l] = 1.0;
                    }
                    Some(kr.reverse_tangent_flat(&seeds, mlp_out)?)
                };
                let row_w = n_th * l;
                for o in 0..mlp_out {
                    // Under the oracle switch, a reverse per direction seeded on tangent lane
                    // `p`, whose value lanes are the same column.
                    let column: Box<dyn Fn(usize, usize) -> f64 + '_> = if let Some(r) = &rows {
                        Box::new(move |p, j| r[o * row_w + j * l + 1 + p])
                    } else {
                        let seeds: Vec<Vec<f64>> = (0..blk.len())
                            .map(|p| {
                                let mut s = vec![0.0; mlp_out * l];
                                s[o * l + 1 + p] = 1.0;
                                s
                            })
                            .collect();
                        let adj = kr.reverse(&seeds)?;
                        Box::new(move |p, j| adj[p][j * l])
                    };
                    for (p, &i) in blk.iter().enumerate() {
                        for j in off..off + n_w {
                            let v = column(p, j);
                            if v != 0.0 {
                                hz[z0 + o].push(((i, j), v));
                            }
                        }
                    }
                }
            }
            jz.extend(jcols.into_iter().enumerate().map(|(a, col)| (off + a, col)));
            z0 += mlp_out;
        }
        Some((jz, hz))
    }
}
/// A subject's covariance pass as a composition of compiled plans (#2786): the prediction
/// plan (its `(f, G)`, and `Q` under the exact anchor, as lanes), FOCE's population prediction
/// at `η = 0`, the non-Gaussian term's plan, and the likelihood kernel's plan, which takes all
/// of those as inputs with the packed coordinates and η and returns `F` and `g = ∇_η l`. Under
/// the quadrature (more than one node) the kernel runs in two stages: the first returns the
/// nodes `b_j` (a function of the anchor), the node predictions (and term values) are the
/// plans' at `b_j`, and the second scores them. Every plan runs at the value layout
/// (`mode_response`) or along a pass's directions (`psi_lanes`, the plans' tangent layout);
/// one reverse — multi-seed, or forward over reverse — runs back through every stage, θ's
/// packing chain closed by hand (its second derivative included).
pub(super) struct CovPlan<'a> {
    ctx: &'a LikCtx<'a>,
    sc: &'a Scoped<'a>,
    subject: &'a Subject,
    template: &'a ModelParameters,
    objective: Objective,
    so: &'a StackedOmega,
    nn_z: Option<&'a [Vec<f64>]>,
    x: &'a [f64],
    nx: usize,
    nz: usize,
    /// θ and `dθ/dx`, `d²θ/dx²` per coordinate (log- or identity-packed).
    th: Vec<f64>,
    d1: Vec<f64>,
    d2: Vec<f64>,
    second: bool,
    variant: kj::Desc,
    /// The θ the prediction plans carry on their η axes ([`MuFold`], #2982).
    fold: Option<MuFold>,
    /// The axes the Hessian's directions move ([`Self::prepare`]); `None`: every plan along
    /// the passes' directions.
    moved: std::cell::RefCell<Option<Moved>>,
    /// Each plan's run along its input axes, kept for every pass ([`basis`]).
    bases: std::cell::RefCell<std::collections::HashMap<PlanKey, Basis>>,
    /// The structured kernel's rows at this point ([`st`]), formed on the first pass.
    st_rows: std::cell::RefCell<Option<st::Rows>>,
    /// The structured kernel's reduce plan, once looked up.
    st_reduce: std::cell::RefCell<[Option<std::sync::Arc<kj::Plan>>; 2]>,
    /// A quadrature subject's rows at its nodes ([`st::NodeRows`]).
    st_node_rows: std::cell::RefCell<Option<st::NodeRows>>,
    /// Set where the structured form does not hold at this point ([`Self::run`]): every later
    /// pass takes the recorded kernel, one direction a pass ([`pass_width`]).
    st_declined: std::cell::Cell<bool>,
}

/// The kernel's input slots for one subject's pass: per observation the prediction plan's
/// base lanes (`f`, `G`, `Q`), FOCE's population predictions, the non-Gaussian term's
/// `(q, ∇q, ∇²q)`, η, the packed coordinates, then (second stage) the nodes' predictions and
/// term values.
#[derive(Clone, Copy)]
struct Slots {
    n_obs: usize,
    /// The kernel's prediction rows: the observations, then an SDE's variance rows.
    n_rows: usize,
    lb: usize,
    pop: usize,
    has_pop: bool,
    ng: usize,
    nb: usize,
    has_ng: bool,
    eta: usize,
    x: usize,
    nf: usize,
    nq_node: usize,
    n_nodes: usize,
    end: usize,
}

impl Slots {
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize, n_obs: usize, n_rows: usize, lb: usize, has_pop: bool, nb: Option<usize>,
        nx: usize, n_nodes: usize,
    ) -> Self {
        let pop = n_rows * lb;
        let ng = pop + if has_pop { n_obs } else { 0 };
        let nbv = nb.unwrap_or(0);
        let eta = ng + nb.map_or(0, |b| 1 + b + b * (b + 1) / 2);
        let x = eta + n;
        let nf = x + nx;
        let nq_node = nf + n_nodes * n_rows;
        let end = nq_node + if nb.is_some() { n_nodes } else { 0 };
        Slots {
            n_obs,
            n_rows,
            lb,
            pop,
            has_pop,
            ng,
            nb: nbv,
            has_ng: nb.is_some(),
            eta,
            x,
            nf,
            nq_node,
            n_nodes,
            end,
        }
    }
    fn f(&self, j: usize, lane: usize) -> usize {
        j * self.lb + lane
    }
    fn ng_g(&self, k: usize) -> usize {
        self.ng + 1 + k
    }
    fn ng_h(&self, k: usize, l: usize) -> usize {
        let (k, l) = if l <= k { (k, l) } else { (l, k) };
        self.ng + 1 + self.nb + k * (k + 1) / 2 + l
    }
}

/// A plan run kept for its reverse: the plan, its forward, its layout.
/// A plan's runs along `n` input axes at one tangent each (`inputs(a)`: axis `a`'s inputs at
/// that layout): one forward with the primal shared ([`kj::Plan::forward_rows`], #2980), or —
/// where some chunk has no such kernel — one forward per axis. Either way one [`Kept`] per axis,
/// as a one-tangent run keeps it; `None` where a run fails or the guard does not hold.
fn rows_or_each(
    pl: &PlanArc, n: usize, inputs: impl Fn(usize) -> (Vec<f64>, Vec<f64>), base: Layout,
    n_obs: usize,
) -> Option<Vec<Kept>> {
    let lt = Layout { tan: 1, ..base };
    let (l, w) = (base.lanes(), lt.lanes());
    // The packed inputs' values (the base lanes) and, per axis, their tangents.
    let packed: Vec<(Vec<f64>, Vec<f64>)> = (0..n).map(&inputs).collect();
    let split = |v: &[f64], tan: bool| -> Vec<f64> {
        v.chunks(w).flat_map(|s| if tan { &s[l..] } else { &s[..l] }).copied().collect()
    };
    if let Some(&(ref u0, ref p0)) = packed.first().filter(|_| pl.rows_serve(base)) {
        let (u, p) = (split(u0, false), split(p0, false));
        let tu: Vec<f64> = packed
            .iter()
            .flat_map(|(u1, p1)| {
                let mut t = split(u1, true);
                t.extend(split(p1, true));
                t
            })
            .collect();
        if let Some(rf) = pl.forward_rows(base, &u, &p, &tu, n) {
            if !pl.rows_guard(&rf) {
                return None;
            }
            let rf = std::rc::Rc::new(rf);
            return Some(
                (0..n)
                    .map(|sd| {
                        // Each output's values, then this axis's tangents: a one-tangent run's.
                        let mut du = vec![0.0; (n_obs + 1) * w];
                        for o in 0..n_obs {
                            let (v, t) = pl.rows_out(&rf, o, sd);
                            du[o * w..o * w + l].copy_from_slice(v);
                            du[o * w + l..(o + 1) * w].copy_from_slice(t);
                        }
                        du[n_obs * w] = 1.0;
                        Kept { plan: pl.clone(), fwd: KeptFwd::Rows(rf.clone()), layout: lt, du }
                    })
                    .collect(),
            );
        }
    }
    packed
        .into_iter()
        .map(|(u, p)| {
            let mut du = vec![0.0; (n_obs + 1) * w];
            let fwd = pl.forward_layout(lt, &u, &p, &mut du)?;
            (du[n_obs * w] == 1.0).then(|| Kept {
                plan: pl.clone(),
                fwd: KeptFwd::One(fwd),
                layout: lt,
                du,
            })
        })
        .collect()
}

/// A kept run's forward: its own, or its share of a forward along several axes at once.
enum KeptFwd {
    One(kj::Fwd),
    Rows(std::rc::Rc<kj::RowsFwd>),
}

impl KeptFwd {
    /// Its own forward (a run along a pass's directions always has one).
    fn one(&self) -> Option<&kj::Fwd> {
        match self {
            KeptFwd::One(f) => Some(f),
            KeptFwd::Rows(_) => None,
        }
    }
}

struct Kept {
    plan: PlanArc,
    fwd: KeptFwd,
    layout: Layout,
    du: Vec<f64>,
}

impl Kept {
    /// Output `o`'s base lane `x` along tangent `a` (`None`: the value).
    fn out(&self, o: usize, x: usize, a: Option<usize>) -> f64 {
        let (l, lb) = (self.layout.lanes(), self.layout.base_lanes());
        let kt = self.layout.tan as usize;
        self.du[o * l + a.map_or(x, |a| lb + x * kt + a)]
    }
}

/// One forward of the composition, kept for the reverse.
pub(super) struct CovRun {
    slots: Slots,
    kt: usize,
    /// The packed axis each tangent moves.
    dir_x: Vec<Option<usize>>,
    pred: Option<PlanRun>,
    /// An SDE subject's EKF variance rows, a plan over θ then σ.
    var: Option<PlanRun>,
    pop: Option<PlanRun>,
    ng: Option<PlanRun>,
    /// Per node: its predictions', its term's and its variance rows' plans.
    nodes: Vec<(Option<PlanRun>, Option<PlanRun>, Option<PlanRun>)>,
    k1: Option<super::kernel::KernelRun>,
    /// The recorded likelihood kernel; `None` where the pass takes the structured kernel.
    k2: Option<super::kernel::KernelRun>,
    /// The structured kernel's inputs (every kernel input's value, then its tangents): the
    /// pass's [`st`] form instead of `k2`.
    st_inputs: Option<Vec<f64>>,
    /// The structured pass's `ψ_a`, set by its sweep.
    psi: std::cell::RefCell<Option<Vec<f64>>>,
}

impl CovRun {
    /// The kernel's outputs: per output (`F`, then each `g_k`), its value and `kt` tangents
    /// (the value alone at the value layout).
    pub(super) fn out(&self) -> &[f64] {
        self.k2.as_ref().map_or(&[][..], |k| &k.out[..])
    }
}

impl<'a> CovPlan<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &'a LikCtx<'a>, sc: &'a Scoped<'a>, subject: &'a Subject,
        template: &'a ModelParameters, x: &'a [f64], objective: Objective, so: &'a StackedOmega,
        nn_z: Option<&'a [Vec<f64>]>,
    ) -> Self {
        let nz = nn_z.map_or(0, |z| z.iter().map(Vec::len).sum());
        let n_th = template.theta.len();
        let logp = |m: usize| theta_packs_log(template.theta_lower[m]);
        let th: Vec<f64> = (0..n_th).map(|m| if logp(m) { x[m].exp() } else { x[m] }).collect();
        let d1: Vec<f64> = (0..n_th).map(|m| if logp(m) { th[m] } else { 1.0 }).collect();
        let d2: Vec<f64> = (0..n_th).map(|m| if logp(m) { th[m] } else { 0.0 }).collect();
        let second = matches!(objective, Objective::Agq { exact: true, .. });
        let (ka, n_agq) = match objective {
            Objective::Foce => (0u8, 0),
            Objective::Focei => (1, 0),
            Objective::Agq { n_agq, exact } => (2 + exact as u8, n_agq),
        };
        let variant = super::kernel::variant_of(&(
            6u8,
            (ka, n_agq),
            sc.width,
            &so.placed.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
        ));
        let fold =
            if basis::mu_fold_off() { None } else { MuFold::at(sc.model, subject, &th, &sc.eta) };
        CovPlan {
            ctx,
            sc,
            subject,
            template,
            objective,
            so,
            nn_z,
            x,
            nx: x.len(),
            nz,
            th,
            d1,
            d2,
            second,
            variant,
            fold,
            moved: std::cell::RefCell::new(None),
            bases: std::cell::RefCell::new(std::collections::HashMap::new()),
            st_rows: std::cell::RefCell::new(None),
            st_reduce: std::cell::RefCell::new([None, None]),
            st_node_rows: std::cell::RefCell::new(None),
            st_declined: std::cell::Cell::new(false),
        }
    }

    /// The subject's random-effect width.
    fn pn(&self) -> usize {
        self.sc.width
    }

    /// The slot of η `k` in a reverse row (`x`, the network outputs, η).
    fn eta_slot(&self, k: usize) -> usize {
        self.nx + self.nz + k
    }

    /// `rowᵀ·d` over a reverse row (one value per `x`, network output and η).
    fn along(&self, row: &[f64], d: &Dir) -> f64 {
        let mut s = 0.0;
        if let Some(i) = d.x {
            s += row[i];
        }
        if let Some(z) = d.z {
            s += row[self.nx + z];
        }
        for (k, &e) in d.eta.iter().enumerate() {
            if e != 0.0 {
                s += e * row[self.eta_slot(k)];
            }
        }
        s
    }

    /// A plan's inputs at `layout` (`kt` tangent blocks): η (value, one-hot over `n` base
    /// tangents, tangent `a` = `eta_t(k, a)`) and the parameter slots (value, tangent `a` =
    /// `p_t(i, a)`).
    #[allow(clippy::too_many_arguments)]
    fn plan_inputs(
        layout: Layout, eta: &[f64], eta_t: &dyn Fn(usize, usize) -> f64, params: &[f64],
        p_t: &dyn Fn(usize, usize) -> f64, kt: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let (l, lb) = (layout.lanes(), layout.base_lanes());
        let mut u = vec![0.0; eta.len() * l];
        for (k, &v) in eta.iter().enumerate() {
            u[k * l] = v;
            if layout.n > 0 {
                u[k * l + 1 + k] = 1.0;
            }
            for a in 0..kt {
                u[k * l + lb + a] = eta_t(k, a);
            }
        }
        let mut p = vec![0.0; params.len() * l];
        for (i, &v) in params.iter().enumerate() {
            p[i * l] = v;
            for a in 0..kt {
                p[i * l + lb + a] = p_t(i, a);
            }
        }
        (u, p)
    }

    /// θ's tangent along direction `a` of `dirs` (`dθ/dx` along the packed axis it moves).
    fn theta_t(&self, dirs: &[Dir], i: usize, a: usize) -> f64 {
        if i < self.th.len() && dirs.get(a).and_then(|d| d.x) == Some(i) {
            self.d1[i]
        } else {
            0.0
        }
    }

    /// The parameter slots of a prediction plan: θ, and under the reduced basis the network
    /// outputs after it (their tangents: [`Self::param_t`]).
    fn pred_values(&self) -> Vec<f64> {
        let mut p = self.th.clone();
        if let Some(z) = self.nn_z {
            p.extend(z.iter().flatten());
        }
        p
    }

    /// Run a subject's prediction plan at `(θ, eta)` along `eta_t` (base layout `base`).
    #[allow(clippy::too_many_arguments)]
    fn run_pred(
        &self, model: &CompiledModel, eta: &[f64], eta_t: &dyn Fn(usize, usize) -> f64,
        lanes: Lanes<'_>, base: Layout, kt: usize,
    ) -> Option<Vec<Kept>> {
        let layout = Layout { tan: kt as u8, ..base };
        let params = self.pred_values();
        let p_t = |i: usize, a: usize| self.param_t(lanes, i, a);
        let n_obs = self.subject.observations.len();
        let mut serve = |pl: &PlanArc| -> Option<Vec<Kept>> {
            // Along input axes: every axis's run at once, the primal shared (#2980) — or, where a
            // chunk has no such kernel, one run per axis.
            if let Lanes::Axes(ax) = lanes {
                return rows_or_each(
                    pl,
                    ax.len(),
                    |a| {
                        let ax1 = &ax[a..a + 1];
                        let et1 = |k: usize, _: usize| eta_t(k, a);
                        let p1 = |i: usize, _: usize| self.param_t(Lanes::Axes(ax1), i, 0);
                        Self::plan_inputs(Layout { tan: 1, ..base }, eta, &et1, &params, &p1, 1)
                    },
                    base,
                    n_obs,
                );
            }
            let (u, p) = Self::plan_inputs(layout, eta, eta_t, &params, &p_t, kt);
            let mut du = vec![0.0; (n_obs + 1) * layout.lanes()];
            let fwd = pl.forward_layout(layout, &u, &p, &mut du)?;
            (du[n_obs * layout.lanes()] == 1.0)
                .then(|| vec![Kept { plan: pl.clone(), fwd: KeptFwd::One(fwd), layout, du }])
        };
        match self.nn_z {
            Some(z) => super::with_plan_nn(model, self.subject, &self.th, eta, z, None, &mut serve),
            None => super::with_plan(
                model,
                self.subject,
                self.sc.layout.as_deref(),
                &self.th,
                eta,
                &mut serve,
            ),
        }
    }

    /// Run an SDE subject's variance plan (the EKF's `p_obs` rows, parameter slots a prediction
    /// plan's — θ, and under the reduced basis the network outputs — then σ) at `(θ, σ, eta)`
    /// along `eta_t`; σ's tangent along a direction moving its packed coordinate is `dσ/dx = σ`.
    fn run_var(
        &self, eta: &[f64], eta_t: &dyn Fn(usize, usize) -> f64, lanes: Lanes<'_>, base: Layout,
        kt: usize,
    ) -> Option<Vec<Kept>> {
        // Several axes: one run each (a single axis is the one-tangent run below).
        if let Lanes::Axes(ax) = lanes {
            if ax.len() > 1 {
                return (0..ax.len())
                    .map(|a| {
                        let et1 = |k: usize, _: usize| eta_t(k, a);
                        self.run_var(eta, &et1, Lanes::Axes(&ax[a..a + 1]), base, 1)
                            .and_then(|mut v| v.pop())
                    })
                    .collect();
            }
        }
        let layout = Layout { tan: kt as u8, ..base };
        let sigma = self.sc.params.sigma.values.as_slice();
        let mut params = self.pred_values();
        params.extend_from_slice(sigma);
        let p_t = |i: usize, a: usize| self.param_t(lanes, i, a);
        let n_obs = self.subject.observations.len();
        let mut serve = |pl: &PlanArc| -> Option<Vec<Kept>> {
            let (u, p) = Self::plan_inputs(layout, eta, eta_t, &params, &p_t, kt);
            let mut du = vec![0.0; (n_obs + 1) * layout.lanes()];
            let fwd = pl.forward_layout(layout, &u, &p, &mut du)?;
            (du[n_obs * layout.lanes()] == 1.0)
                .then(|| vec![Kept { plan: pl.clone(), fwd: KeptFwd::One(fwd), layout, du }])
        };
        match self.nn_z {
            Some(z) => super::with_plan_nn(
                self.sc.model,
                self.subject,
                &self.th,
                eta,
                z,
                Some(sigma),
                &mut serve,
            ),
            None => super::with_plan_in(
                self.sc.model,
                self.subject,
                self.sc.layout.as_deref(),
                &self.th,
                Some(sigma),
                eta,
                &mut serve,
            ),
        }
    }

    /// Run the subject's non-Gaussian term plan at `(θ, eta)` along `eta_t` (base `base`).
    fn run_term(
        &self, eta: &[f64], eta_t: &dyn Fn(usize, usize) -> f64, lanes: Lanes<'_>, base: Layout,
        kt: usize,
    ) -> Option<Vec<Kept>> {
        // Several axes: one run each (a single axis is the one-tangent run below).
        if let Lanes::Axes(ax) = lanes {
            if ax.len() > 1 {
                return (0..ax.len())
                    .map(|a| {
                        let et1 = |k: usize, _: usize| eta_t(k, a);
                        self.run_term(eta, &et1, Lanes::Axes(&ax[a..a + 1]), base, 1)
                            .and_then(|mut v| v.pop())
                    })
                    .collect();
            }
        }
        let layout = Layout { tan: kt as u8, ..base };
        let plans = super::terms::non_gaussian_plans(self.sc.model, self.subject, &self.th, eta)?;
        let p_t = |i: usize, a: usize| self.param_t(lanes, i, a);
        let (u, p) = Self::plan_inputs(layout, eta, eta_t, &self.th, &p_t, kt);
        plans.iter().find_map(|pl| {
            let mut du = vec![0.0; 2 * layout.lanes()];
            let fwd = pl.forward_layout(layout, &u, &p, &mut du)?;
            (du[layout.lanes()] == 1.0)
                .then(|| vec![Kept { plan: pl.clone(), fwd: KeptFwd::One(fwd), layout, du }])
        })
    }

    /// One forward of the composition along `dirs` as `width` tangent lanes (the value layout
    /// with none; a pass's lanes past its directions carry zeros, so a ragged last pass runs
    /// at the same layout as the others).
    pub(super) fn run(&self, dirs: &[Dir], width: usize) -> Option<CovRun> {
        let (model, subject, sc) = (self.sc.model, self.subject, self.sc);
        let kt = if dirs.is_empty() { 0 } else { width };
        if dirs.len() > kt {
            return None;
        }
        let lk = 1 + kt;
        let n_obs = subject.observations.len();
        let pn = sc.width;
        let nq = if self.second { pn * (pn + 1) / 2 } else { 0 };
        let eta = &sc.eta[..pn];
        let eta_t =
            |k: usize, a: usize| dirs.get(a).and_then(|d| d.eta.get(k)).copied().unwrap_or(0.0);
        // The predictions with their η-jets, and an SDE's variance rows beside them.
        let base = Layout { n: pn as u16, m: nq as u16, th: false, wide: false, tan: 0 };
        let n_par = self.th.len() + self.nz;
        let n_sig = self.sc.params.sigma.values.len();
        let pred = if n_obs > 0 {
            Some(self.plan_run(
                PlanKey::Pred,
                dirs,
                pn,
                n_par,
                true,
                &eta_t,
                kt,
                &|lanes, et, k| self.run_pred(model, eta, et, lanes, base, k),
            )?)
        } else {
            None
        };
        let var = if n_obs > 0 && model.is_sde() {
            Some(self.plan_run(
                PlanKey::Var,
                dirs,
                pn,
                n_par + n_sig,
                true,
                &eta_t,
                kt,
                &|lanes, et, k| self.run_var(eta, et, lanes, base, k),
            )?)
        } else {
            None
        };
        let n_rows = n_obs * (1 + usize::from(var.is_some()));
        let lb = 1 + pn + nq;
        let (interaction, n_agq) = match self.objective {
            Objective::Foce => (false, 1),
            Objective::Focei => (true, 1),
            Objective::Agq { n_agq, .. } => (true, n_agq),
        };
        let has_ng = !subject.obs_records.is_empty();
        let is_agq = matches!(self.objective, Objective::Agq { .. });
        // FOCE's population prediction at η = 0 (beside a non-Gaussian term production takes
        // the conditional variance).
        let pop = if n_obs > 0
            && !is_agq
            && !interaction
            && model.error_spec.has_f_dependent_variance()
        {
            let zero_t = |_: usize, _: usize| 0.0;
            let zeros = vec![0.0; pn];
            Some(self.plan_run(
                PlanKey::Pop,
                dirs,
                pn,
                n_par,
                false,
                &zero_t,
                kt,
                &|lanes, et, k| self.run_pred(sc.pop_model, &zeros, et, lanes, Layout::F64, k),
            )?)
        } else {
            None
        };
        // The non-Gaussian term `(q, ∇q, ∇²q)` over the BSV block.
        let nb = model.n_eta;
        let ng = if has_ng {
            let base = super::terms::layout2(nb);
            Some(self.plan_run(
                PlanKey::Term,
                dirs,
                nb,
                self.th.len(),
                true,
                &eta_t,
                kt,
                &|lanes, et, k| self.run_term(&sc.eta[..nb], et, lanes, base, k),
            )?)
        } else {
            None
        };
        let n_nodes = if is_agq && n_agq > 1 { agq_grid(n_agq, sc.width).len() } else { 0 };
        let slots = Slots::new(
            pn,
            n_obs,
            n_rows,
            lb,
            pop.is_some(),
            ng.as_ref().map(|_| nb),
            self.nx,
            n_nodes,
        );
        // The kernel's inputs: value, then the `kt` tangents.
        let n_base = slots.nf;
        let mut inputs = vec![0.0; slots.end * lk];
        fn put_in(inputs: &mut [f64], lk: usize, slot: usize, v: f64, t: &dyn Fn(usize) -> f64) {
            inputs[slot * lk] = v;
            for a in 0..lk - 1 {
                inputs[slot * lk + 1 + a] = t(a);
            }
        }
        macro_rules! put {
            ($slot:expr, $v:expr, $t:expr) => {
                put_in(&mut inputs, lk, $slot, $v, $t)
            };
        }
        let po = |pr: &PlanRun, j: usize, x: usize, a: Option<usize>| self.plan_out(pr, j, x, a);
        if let Some(pk) = pred.as_ref() {
            for j in 0..n_obs {
                for x in 0..lb {
                    put!(slots.f(j, x), po(pk, j, x, None), &|a| po(pk, j, x, Some(a)));
                }
            }
        }
        if let Some(vk) = var.as_ref() {
            for j in 0..n_obs {
                for x in 0..lb {
                    put!(slots.f(n_obs + j, x), po(vk, j, x, None), &|a| po(vk, j, x, Some(a)));
                }
            }
        }
        if let Some(pp) = pop.as_ref() {
            for j in 0..n_obs {
                put!(slots.pop + j, po(pp, j, 0, None), &|a| po(pp, j, 0, Some(a)));
            }
        }
        if let Some(t) = ng.as_ref() {
            put!(slots.ng, po(t, 0, 0, None), &|a| po(t, 0, 0, Some(a)));
            for k in 0..nb {
                put!(slots.ng_g(k), po(t, 0, 1 + k, None), &|a| po(t, 0, 1 + k, Some(a)));
                for l in 0..=k {
                    let h = super::terms::h_lane(nb, k, l);
                    put!(slots.ng_h(k, l), po(t, 0, h, None), &|a| po(t, 0, h, Some(a)));
                }
            }
        }
        for k in 0..pn {
            put!(slots.eta + k, sc.eta[k], &|a| eta_t(k, a));
        }
        for i in 0..self.nx {
            put!(slots.x + i, self.x[i], &|a| {
                if dirs.get(a).and_then(|d| d.x) == Some(i) {
                    1.0
                } else {
                    0.0
                }
            });
        }
        let record_k1 = |lv: &[kj::Tr]| self.record(&slots, lv, true);
        let record_k2 = |lv: &[kj::Tr]| self.record(&slots, lv, false);
        let layout_k = if kt > 0 { Layout { tan: kt as u8, ..Layout::F64 } } else { Layout::F64 };
        // The quadrature's nodes: stage 1, then the node predictions (and term values) there.
        let mut nodes = Vec::new();
        // A structured quadrature subject's pass takes its nodes from the structured stage 1.
        let st_nodes = if kt > 0 && slots.n_nodes > 0 && self.structured() {
            self.struct_nodes(&inputs[..n_base * lk], &slots, kt)
        } else {
            None
        };
        let k1 = if slots.n_nodes > 0 {
            let k1 = match st_nodes {
                Some(_) => None,
                None => Some(super::kernel::kernel_run(
                    model,
                    subject,
                    super::kernel::variant_of(&(self.variant.bytes(), 1u8, slots.end)),
                    layout_k,
                    &inputs[..n_base * lk],
                    slots.n_nodes * pn,
                    &record_k1,
                )?),
            };
            let bout: &[f64] = match (&st_nodes, &k1) {
                (Some(b), _) => b,
                (None, Some(k)) => &k.out,
                (None, None) => return None,
            };
            for nd in 0..slots.n_nodes {
                let bv: Vec<f64> = (0..pn).map(|k| bout[(nd * pn + k) * lk]).collect();
                let bt = |k: usize, a: usize| bout[(nd * pn + k) * lk + 1 + a];
                let fb = if n_obs > 0 {
                    let fb = self.plan_run(
                        PlanKey::Node(nd),
                        dirs,
                        pn,
                        n_par,
                        true,
                        &bt,
                        kt,
                        &|lanes, et, k| self.run_pred(model, &bv, et, lanes, Layout::F64, k),
                    )?;
                    for j in 0..n_obs {
                        put!(slots.nf + nd * n_rows + j, po(&fb, j, 0, None), &|a| po(
                            &fb,
                            j,
                            0,
                            Some(a)
                        ));
                    }
                    Some(fb)
                } else {
                    None
                };
                let fv = if n_obs > 0 && model.is_sde() {
                    let fv = self.plan_run(
                        PlanKey::NodeVar(nd),
                        dirs,
                        pn,
                        n_par + n_sig,
                        true,
                        &bt,
                        kt,
                        &|lanes, et, k| self.run_var(&bv, et, lanes, Layout::F64, k),
                    )?;
                    for j in 0..n_obs {
                        put!(slots.nf + nd * n_rows + n_obs + j, po(&fv, j, 0, None), &|a| po(
                            &fv,
                            j,
                            0,
                            Some(a)
                        ));
                    }
                    Some(fv)
                } else {
                    None
                };
                let qb = if ng.is_some() {
                    let bb: Vec<f64> = bv[..nb].to_vec();
                    let t = self.plan_run(
                        PlanKey::NodeTerm(nd),
                        dirs,
                        nb,
                        self.th.len(),
                        true,
                        &bt,
                        kt,
                        &|lanes, et, k| self.run_term(&bb, et, lanes, Layout::F64, k),
                    )?;
                    put!(slots.nq_node + nd, po(&t, 0, 0, None), &|a| po(&t, 0, 0, Some(a)));
                    Some(t)
                } else {
                    None
                };
                nodes.push((fb, qb, fv));
            }
            k1
        } else {
            None
        };
        // A tangent pass of a structured subject keeps the kernel's inputs for [`st`] instead of
        // running the recorded kernel — where its rows hold at this point. Where they do not (a
        // row off its domain, FOCE past its Woodbury gate, where `foce_f` leaves Woodbury's form,
        // or a structured stage 1 that did not form), this pass is laid out at the structured
        // width and, under quadrature, has no recorded stage 1: it is declined here and
        // [`psi_lanes`] reruns every pass on the recorded kernels, one direction each.
        let structured = kt > 0 && self.structured();
        if structured
            && !((slots.n_nodes == 0 || st_nodes.is_some())
                && self.struct_rows_hold(&inputs, &slots, kt))
        {
            self.st_declined.set(true);
            return None;
        }
        let k2 = if structured {
            None
        } else {
            Some(super::kernel::kernel_run(
                model,
                subject,
                super::kernel::variant_of(&(self.variant.bytes(), 2u8, slots.end)),
                layout_k,
                &inputs,
                1 + pn,
                &record_k2,
            )?)
        };
        let st_inputs = structured.then(|| inputs.clone());
        Some(CovRun {
            slots,
            kt,
            dir_x: (0..kt).map(|a| dirs.get(a).and_then(|d| d.x)).collect(),
            pred,
            var,
            pop,
            ng,
            nodes,
            k1,
            k2,
            st_inputs,
            psi: std::cell::RefCell::new(None),
        })
    }

    /// The kernel's computation on the recording type from its inputs `lv`: stage 1 (`nodes`)
    /// the quadrature's nodes `b_j`, else `[F, g_0 … g_{N−1}]`.
    fn record(&self, sl: &Slots, lv: &[kj::Tr], nodes: bool) -> Option<Vec<kj::Tr>> {
        type T = kj::Tr;
        let c = |v: f64| <T as PkNum>::from_f64(v);
        let (ctx, sc, subject, template) = (self.ctx, self.sc, self.subject, self.template);
        let model = sc.model;
        let n_obs = sl.n_obs;
        let n_rows = sl.n_rows;
        let pn = sc.width;
        let f: Vec<T> = (0..n_rows).map(|j| lv[sl.f(j, 0)]).collect();
        let g: Vec<Vec<T>> =
            (0..n_rows).map(|j| (0..pn).map(|k| lv[sl.f(j, 1 + k)]).collect()).collect();
        let eo: Vec<T> = (0..pn).map(|k| lv[sl.eta + k]).collect();
        let xl = &lv[sl.x..sl.x + self.nx];
        let xo = |i: usize| xl[i];
        let theta: Vec<T> = unpack_theta(template, xo);
        let p = population_params(model, subject, template, sc.params, self.so, &theta, xo)?;
        let fpop: Option<Vec<T>> = sl.has_pop.then(|| (0..n_obs).map(|j| lv[sl.pop + j]).collect());
        let ng = sl.has_ng.then(|| {
            let nb = sl.nb;
            crate::sens::focei_kernel::NonGauss {
                q: lv[sl.ng],
                g: (0..pn).map(|k| if k < nb { lv[sl.ng_g(k)] } else { c(0.0) }).collect(),
                h: (0..pn)
                    .map(|k| {
                        (0..pn)
                            .map(|l| if k < nb && l < nb { lv[sl.ng_h(k, l)] } else { c(0.0) })
                            .collect()
                    })
                    .collect(),
            }
        });
        let (fv, gv) = match self.objective {
            Objective::Foce | Objective::Focei => {
                let interaction = self.objective == Objective::Focei;
                match ng.as_ref() {
                    None => objective_terms(ctx, interaction, &f, &g, &eo, fpop.as_deref(), &p)?,
                    Some(ng) => super::objective_ng::<T>(
                        ctx,
                        interaction,
                        &f,
                        &g,
                        &eo,
                        fpop.as_deref(),
                        ng,
                        &p,
                    )?,
                }
            }
            Objective::Agq { n_agq, exact } => {
                // The anchor H_a — the exact Hessian over the predictions' (f, G, Q), or
                // FOCEI's H̃ — jittered as production's proposal is.
                let mut h_a: Vec<Vec<T>> = if exact {
                    let q: Vec<Vec<T>> = (0..n_rows)
                        .map(|j| (0..pn * (pn + 1) / 2).map(|t| lv[sl.f(j, 1 + pn + t)]).collect())
                        .collect();
                    crate::sens::focei_kernel::exact_inner_hessian_kernel(ctx, &f, &g, &q, &eo, &p)?
                } else {
                    focei_htilde(ctx, &f, &g, &eo, &p)?
                };
                if let Some(ng) = ng.as_ref() {
                    for k in 0..pn {
                        for l in 0..pn {
                            h_a[k][l] = h_a[k][l] + ng.h[k][l];
                        }
                    }
                }
                let nh = h_a.len();
                let lh = factor_anchor(&mut h_a, p.omega_inv(nh))?;
                let grid = agq_grid(n_agq, sc.width);
                if nodes {
                    // b = η + √2·L⁻ᵀz per node.
                    let mut out = Vec::with_capacity(grid.len() * pn);
                    for (z, _) in &grid {
                        let mut step = vec![c(0.0); pn];
                        for i in (0..pn).rev() {
                            let mut acc = c(z[i]);
                            for j in (i + 1)..pn {
                                acc = acc - lh[j][i] * step[j];
                            }
                            step[i] = acc / lh[i][i];
                        }
                        out.extend((0..pn).map(|k| eo[k] + c(std::f64::consts::SQRT_2) * step[k]));
                    }
                    return Some(out);
                }
                let half_logdet = c(0.5) * chol_logdet(&lh);
                let q_hat = ng.as_ref().map_or(c(0.0), |n| n.q);
                let a = if n_agq == 1 {
                    inner_l(ctx, &f, &eo, &p, None)? + q_hat + half_logdet
                } else {
                    // The nodes again on this stage's inputs (the same arithmetic as stage 1).
                    let mut terms: Vec<T> = Vec::with_capacity(grid.len());
                    for (nd, (z, cw)) in grid.iter().enumerate() {
                        let mut step = vec![c(0.0); pn];
                        for i in (0..pn).rev() {
                            let mut acc = c(z[i]);
                            for j in (i + 1)..pn {
                                acc = acc - lh[j][i] * step[j];
                            }
                            step[i] = acc / lh[i][i];
                        }
                        let b: Vec<T> = (0..pn)
                            .map(|k| eo[k] + c(std::f64::consts::SQRT_2) * step[k])
                            .collect();
                        let fb: Vec<T> = (0..n_rows).map(|j| lv[sl.nf + nd * n_rows + j]).collect();
                        let qb = if sl.has_ng { lv[sl.nq_node + nd] } else { c(0.0) };
                        terms.push(c(*cw) - (inner_l(ctx, &fb, &b, &p, None)? + qb));
                    }
                    half_logdet - log_sum_exp(&terms)
                };
                let mut lg = inner_grad(ctx, &f, &g, &eo, &p, None)?;
                if let Some(ng) = ng.as_ref() {
                    for k in 0..pn {
                        lg[k] = lg[k] + ng.g[k];
                    }
                }
                (a, lg)
            }
        };
        Some(std::iter::once(fv).chain(gv).collect())
    }

    /// One reverse of `run` seeded with each row of `seeds` (laid out as the kernel's
    /// outputs, `1 + kt` lanes each): per seed, the adjoint of every packed coordinate,
    /// network output and η (`nx + nz + N`).
    pub(super) fn reverse(&self, r: &CovRun, seeds: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        self.sweep(r, seeds, false)
    }

    /// Forward over reverse (#2980): one sweep of `run` seeded with `seed` (laid out as the
    /// kernel's outputs; its value lanes `ū`, its tangent lanes zero) carrying every adjoint's
    /// tangents along the run's directions. Per direction `a`, the row [`Self::reverse`] gives
    /// for the seed `ū` on the outputs' tangent lane `a`: the second derivative of `ūᵀ·out`
    /// along direction `a`, one sweep for every direction.
    ///
    /// Each kernel input's adjoint lanes stand in for that reverse's seeds: tangent `a` of the
    /// value lane's adjoint is seed `a`'s value-lane adjoint, and the value lane's adjoint is
    /// seed `a`'s adjoint on tangent lane `a` (zero on the others — tangent lanes do not mix).
    pub(super) fn reverse_tangent(&self, r: &CovRun, seed: &[f64]) -> Option<Vec<Vec<f64>>> {
        if r.kt == 0 {
            return None;
        }
        self.sweep(r, &[seed.to_vec()], true)
    }

    /// [`Self::reverse`] (`tangent` false: a reverse per seed) and [`Self::reverse_tangent`]
    /// (`tangent`: one seed, its adjoints' tangents carried; a row per direction).
    fn sweep(&self, r: &CovRun, seeds: &[Vec<f64>], tangent: bool) -> Option<Vec<Vec<f64>>> {
        let sl = &r.slots;
        let lk = 1 + r.kt;
        let ns = if tangent { r.kt } else { seeds.len() };
        let pn = self.pn();
        let width = self.nx + self.nz + pn;
        let mut rows = vec![vec![0.0; width]; ns];
        // Stage 2, then the nodes back to their locations, then stage 1.
        let mut pending = None;
        let mut adj = if tangent {
            match r.st_inputs.as_ref() {
                Some(_) => {
                    let (adj, psi, pend) = self.struct_adjoint(r, &seeds[0])?;
                    *r.psi.borrow_mut() = Some(psi);
                    pending = Some(pend);
                    vec![adj]
                }
                None => vec![r.k2.as_ref()?.reverse_tangent(&seeds[0])?.to_vec()],
            }
        } else {
            r.k2.as_ref()?.reverse(seeds)?
        };
        if sl.n_nodes > 0 {
            let mut b_seeds = vec![vec![0.0; sl.n_nodes * pn * lk]; adj.len()];
            for (nd, (fb, qb, fv)) in r.nodes.iter().enumerate() {
                let mut node =
                    |kept: &PlanRun, slot: &dyn Fn(usize) -> usize, n_out: usize| -> Option<()> {
                        let back =
                            self.plan_reverse(kept, r, &adj, slot, n_out, tangent, &mut rows)?;
                        for (sd, ub) in back.iter().enumerate() {
                            for k in 0..ub.len() / lk {
                                for x in 0..lk {
                                    b_seeds[sd][(nd * pn + k) * lk + x] += ub[k * lk + x];
                                }
                            }
                        }
                        Some(())
                    };
                if let Some(fb) = fb.as_ref() {
                    node(fb, &|j| sl.nf + nd * sl.n_rows + j, sl.n_obs)?;
                }
                if let Some(fv) = fv.as_ref() {
                    node(fv, &|j| sl.nf + nd * sl.n_rows + sl.n_obs + j, sl.n_obs)?;
                }
                if let Some(qb) = qb.as_ref() {
                    node(qb, &|_| sl.nq_node + nd, 1)?;
                }
            }
            match (pending.take(), r.k1.as_ref()) {
                // Stage 1 structured: its forward over reverse, then the anchor rows.
                (Some(pend), _) => self.struct_finish(r, &mut adj[0], pend, Some(&b_seeds[0]))?,
                (None, Some(k1)) => {
                    let adj1 = if tangent {
                        vec![k1.reverse_tangent(&b_seeds[0])?.to_vec()]
                    } else {
                        k1.reverse(&b_seeds)?
                    };
                    for (a2, a1) in adj.iter_mut().zip(&adj1) {
                        for (x, y) in a2.iter_mut().zip(a1) {
                            *x += y;
                        }
                    }
                }
                (None, None) => return None,
            }
        }
        if let Some(pend) = pending.take() {
            self.struct_finish(r, &mut adj[0], pend, None)?;
        }
        // The predictions, FOCE's population predictions and the term, back to θ and η.
        if let Some(pk) = r.pred.as_ref() {
            let lb = sl.lb;
            let slot = |j, x| sl.f(j, x);
            let e =
                self.plan_reverse_lanes(pk, r, &adj, &slot, sl.n_obs, lb, tangent, &mut rows)?;
            self.eta_into(&e, lk, tangent, &mut rows);
        }
        if let Some(vk) = r.var.as_ref() {
            let lb = sl.lb;
            let e = self.plan_reverse_lanes(
                vk,
                r,
                &adj,
                &|j, x| sl.f(sl.n_obs + j, x),
                sl.n_obs,
                lb,
                tangent,
                &mut rows,
            )?;
            self.eta_into(&e, lk, tangent, &mut rows);
        }
        if let Some(pp) = r.pop.as_ref() {
            self.plan_reverse(pp, r, &adj, &|j| sl.pop + j, sl.n_obs, tangent, &mut rows)?;
        }
        if let Some(t) = r.ng.as_ref() {
            let nb = sl.nb;
            // The term plan's base layout (`run`): value, gradient and Hessian over the BSV block.
            let lb = super::terms::layout2(nb).base_lanes();
            let slot = |_: usize, x: usize| -> usize {
                if x == 0 {
                    sl.ng
                } else if x <= nb {
                    sl.ng_g(x - 1)
                } else {
                    let (k, l) = untri(x - 1 - nb);
                    sl.ng_h(k, l)
                }
            };
            let e = self.plan_reverse_lanes(t, r, &adj, &slot, 1, lb, tangent, &mut rows)?;
            self.eta_into(&e, lk, tangent, &mut rows);
        }
        // The kernel's own η and packed coordinates.
        for (sd, row) in rows.iter_mut().enumerate() {
            for k in 0..pn {
                row[self.eta_slot(k)] += seed_value(&adj, tangent, sd, (sl.eta + k) * lk);
            }
            for i in 0..self.nx {
                row[i] += seed_value(&adj, tangent, sd, (sl.x + i) * lk);
            }
        }
        rows.iter().flatten().all(|v| v.is_finite()).then_some(rows)
    }

    /// The value lanes of a plan's η adjoints (`e`, per seed, `lk` lanes per η) into `rows`
    /// (under `tangent`, each direction's: see [`seed_value`]).
    fn eta_into(&self, e: &[Vec<f64>], lk: usize, tangent: bool, rows: &mut [Vec<f64>]) {
        let Some(first) = e.first() else { return };
        for (sd, row) in rows.iter_mut().enumerate() {
            for k in 0..first.len() / lk {
                row[self.eta_slot(k)] += seed_value(e, tangent, sd, k * lk);
            }
        }
    }

    /// A plan whose outputs are kernel inputs read at base lane 0 only (`slot(j)` per output):
    /// its reverse seeded with the kernel's adjoints of them; θ, network outputs and η into
    /// `rows`. Returns the η-input adjoints per seed (all lanes), for a node's location.
    fn plan_reverse(
        &self, kept: &PlanRun, r: &CovRun, adj: &[Vec<f64>], slot: &dyn Fn(usize) -> usize,
        n_out: usize, tangent: bool, rows: &mut [Vec<f64>],
    ) -> Option<Vec<Vec<f64>>> {
        let s2 = |j: usize, x: usize| if x == 0 { slot(j) } else { usize::MAX };
        self.plan_reverse_lanes(kept, r, adj, &s2, n_out, 1, tangent, rows)
    }

    /// The general case of [`Self::plan_reverse`]: output `j`'s base lane `x < nl` is kernel
    /// input `slot(j, x)` (`usize::MAX`: not an input). Under `tangent`, `adj` is the one
    /// forward-over-reverse buffer and the plan's reverse is its own forward over reverse, so
    /// the plan's adjoints come back in the same form.
    #[allow(clippy::too_many_arguments)]
    fn plan_reverse_lanes(
        &self, pr: &PlanRun, r: &CovRun, adj: &[Vec<f64>], slot: &dyn Fn(usize, usize) -> usize,
        n_out: usize, nl: usize, tangent: bool, rows: &mut [Vec<f64>],
    ) -> Option<Vec<Vec<f64>>> {
        let kept = match pr {
            PlanRun::Direct(k) => k,
            // A basis is only built for forward-over-reverse passes ([`CovPlan::plan_run`]).
            PlanRun::Basis(u) if tangent => {
                return self.basis_reverse(u, r, adj, slot, n_out, nl, rows);
            }
            PlanRun::Basis(_) => return None,
        };
        let (kt, lk) = (r.kt, 1 + r.kt);
        let (l, lb) = (kept.layout.lanes(), kept.layout.base_lanes());
        let ns = adj.len();
        let w = n_out * l;
        let mut du_bar = kj::Buf::new(ns * w);
        for (sd, a) in adj.iter().enumerate() {
            for j in 0..n_out {
                for x in 0..nl {
                    let s = slot(j, x);
                    if s == usize::MAX {
                        continue;
                    }
                    du_bar[sd * w + j * l + x] = a[s * lk];
                    for t in 0..kt {
                        du_bar[sd * w + j * l + lb + x * kt + t] = a[s * lk + 1 + t];
                    }
                }
            }
        }
        let (n_eta, n_par) = (kept.plan.n_eta(), kept.plan.n_theta());
        let (nu, np) = (n_eta * l, n_par * l);
        let (mut ub, mut pb) = (kj::Buf::new(ns * nu), kj::Buf::new(ns * np));
        let swept = if tangent {
            kept.plan.reverse_tangent_from(kept.fwd.one()?, 1, &mut ub, &mut pb, &du_bar)
        } else {
            kept.plan.reverse_from(kept.fwd.one()?, kept.layout, ns, &mut ub, &mut pb, &du_bar)
        };
        if !swept {
            return None;
        }
        let n_th = self.th.len();
        let s0 = packed_segments(self.template).sigma_start();
        let sigma = self.sc.params.sigma.values.as_slice();
        if tangent {
            // Per direction `a`: θ's value adjoint's tangent `a` through dθ/dx, and (on the
            // axis `a` moves) its value adjoint through d²θ/dx² — the reverse over forward's
            // two terms, each seed's lanes read off the one buffer ([`seed_value`]).
            for (a, row) in rows.iter_mut().enumerate() {
                for i in 0..n_par {
                    let (pv, pa) = (pb[i * l], pb[i * l + lb + a]);
                    if i < n_th {
                        row[i] += pa * self.d1[i];
                        if r.dir_x[a] == Some(i) {
                            row[i] += pv * self.d2[i];
                        }
                    } else if i - n_th < self.nz {
                        row[self.nx + (i - n_th)] += pa;
                    } else {
                        let k = i - n_th - self.nz;
                        row[s0 + k] += pa * sigma[k];
                        if r.dir_x[a] == Some(s0 + k) {
                            row[s0 + k] += pv * sigma[k];
                        }
                    }
                }
            }
            let mut e = vec![0.0; n_eta * lk];
            for k in 0..n_eta {
                e[k * lk] = ub[k * l];
                for t in 0..kt {
                    e[k * lk + 1 + t] = ub[k * l + lb + t];
                }
            }
            return Some(vec![e]);
        }
        let mut eta_back = Vec::with_capacity(ns);
        for (sd, row) in rows.iter_mut().enumerate() {
            let pb = &pb[sd * np..(sd + 1) * np];
            for i in 0..n_par {
                let pv = pb[i * l];
                if i < n_th {
                    // θ = θ(x_i): its value's adjoint through dθ/dx, and each tangent's
                    // (`dθ/dx·ẋ_a`) through d²θ/dx²·ẋ_a.
                    row[i] += pv * self.d1[i];
                    for t in 0..kt {
                        if r.dir_x[t] == Some(i) {
                            row[i] += pb[i * l + lb + t] * self.d2[i];
                        }
                    }
                } else if i - n_th < self.nz {
                    // A network output under the reduced basis.
                    row[self.nx + (i - n_th)] += pv;
                } else {
                    // An SDE variance plan's σ = exp(x), after the outputs: dσ/dx = d²σ/dx² = σ.
                    let k = i - n_th - self.nz;
                    row[s0 + k] += pv * sigma[k];
                    for t in 0..kt {
                        if r.dir_x[t] == Some(s0 + k) {
                            row[s0 + k] += pb[i * l + lb + t] * sigma[k];
                        }
                    }
                }
            }
            let ub = &ub[sd * nu..(sd + 1) * nu];
            // η: the plan's own η inputs (the kernel's η slots — the node plans' are their
            // locations, returned to the caller).
            let mut e = vec![0.0; n_eta * lk];
            for k in 0..n_eta {
                e[k * lk] = ub[k * l];
                for t in 0..kt {
                    e[k * lk + 1 + t] = ub[k * l + lb + t];
                }
            }
            eta_back.push(e);
        }
        Some(eta_back)
    }
}

/// Seed `sd`'s value-lane adjoint at lane offset `at` (a location's `lk` lanes start there):
/// row `sd` of a reverse per seed, or — from one forward-over-reverse buffer (`tangent`),
/// whose tangent `sd` is that seed's value-lane adjoint ([`CovPlan::reverse_tangent`]) —
/// the value adjoint's tangent along direction `sd`.
fn seed_value(adj: &[Vec<f64>], tangent: bool, sd: usize, at: usize) -> f64 {
    if tangent {
        adj[0][at + 1 + sd]
    } else {
        adj[sd][at]
    }
}

/// `(k, l)`, `l ≤ k`, of the packed lower-triangle index `t = k(k+1)/2 + l`.
fn untri(t: usize) -> (usize, usize) {
    let mut k = (((8 * t + 1) as f64).sqrt() as usize).saturating_sub(1) / 2;
    while (k + 1) * (k + 2) / 2 <= t {
        k += 1;
    }
    while k * (k + 1) / 2 > t {
        k -= 1;
    }
    (k, t - k * (k + 1) / 2)
}
/// `(F, ∇_η l)` for FOCE / FOCEI (`objective`).
fn objective_terms<T: PkNum>(
    ctx: &LikCtx, interaction: bool, f: &[T], g: &[Vec<T>], eta: &[T], fpop: Option<&[T]>,
    p: &ParT<T>,
) -> Option<(T, Vec<T>)> {
    objective(ctx, interaction, f, g, eta, fpop, p)
}

/// Ω (Cholesky entries), σ, ρ, the serial ρ and the custom residual magnitude as the
/// kernel reads them, from the packed coordinates `xo` through their packing chains.
fn population_params<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel, subject: &Subject, template: &ModelParameters, params: &ModelParameters,
    so: &StackedOmega, theta: &[T], xo: impl Fn(usize) -> T,
) -> Option<ParT<T>> {
    let mut l: Vec<Vec<T>> =
        so.chol.iter().map(|r| r.iter().map(|&v| T::from_f64(v)).collect()).collect();
    for (idx, at, _) in &so.placed {
        let e = if at[0].0 == at[0].1 { xo(*idx).exp() } else { xo(*idx) };
        for &(i, j) in at {
            l[i][j] = e;
        }
    }
    let seg = packed_segments(template);
    let rho_start = rho_packed_start(template);
    let ruv_mult = match model.ruv_magnitude.as_ref().filter(|rm| rm.is_active()) {
        Some(rm) => Some(
            (0..subject.observations.len())
                .map(|j| rm.eval_subject_obs_g(&model.error_spec, subject, j, theta))
                .collect::<Option<Vec<_>>>()?,
        ),
        None => None,
    };
    Some(ParT {
        omega: OmegaT::from_chol(l)?,
        sigma: (0..params.sigma.values.len()).map(|k| xo(seg.sigma_start() + k).exp()).collect(),
        // ρ = tanh(z), the Fisher-z packing (`rho_chain` = 1 − ρ²).
        rho: (0..params.residual_correlations.len())
            .map(|k| {
                let e = (xo(rho_start + k) * T::from_f64(2.0)).exp();
                (e - T::from_f64(1.0)) / (e + T::from_f64(1.0))
            })
            .collect(),
        serial_rho: model.residual_serial.as_ref().map(|s| theta[s.rho_theta]),
        theta: theta.to_vec(),
        ruv_mult,
    })
}

/// θ from its packed coordinates on any number type: `exp(x)` when log-packed.
fn unpack_theta<T: PkNum>(template: &ModelParameters, xd: impl Fn(usize) -> T) -> Vec<T> {
    (0..template.theta.len())
        .map(|m| if theta_packs_log(template.theta_lower[m]) { xd(m).exp() } else { xd(m) })
        .collect()
}
