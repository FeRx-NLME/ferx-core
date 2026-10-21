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
//! Everything comes from one primitive, a `pass`: the objective `F` (FOCE, FOCEI, or the
//! AGQ marginal under either anchor) and `g`, through the same recorders and kernel as the
//! gradient, over a forward **first-order** dual along up to `K` directions in `(x, η)`
//! whose components are scalars on the reverse tape ([`R`]) — the packed coordinates and η
//! (and, under [`NnReduction`], the networks' raw outputs) are the tape's leaves. The
//! predictions carry an inner η-dual, so `G` (and under the exact anchor `Q`) are
//! differentiated with them.
//!
//! - One pass along no direction, swept once per output, gives the gradients of `F` and
//!   of every `g_k` with respect to every leaf: `∇_η F`, `H` and `∂g` along any direction
//!   at once (`mode_response`).
//! - A pass along `u_1 … u_K`, swept once per tangent component `∂Ψ_w/∂u_a`, gives the
//!   `K` Hessian columns `∇_v(∂Ψ_w/∂u_a) = ∇²Ψ_w u_a`, contracted with every `u_b`
//!   (`psi_along`): reverse over forward, so the Hessian costs `⌈n/K⌉` passes.
//!
//! There are no third-order sensitivity tables and no finite differences, and FOCE's
//! frozen-variance mode (whose `R⁰` moves with θ through `f(η = 0)`) is just another term
//! of `g`.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::estimation::parameterization::{packed_segments, rho_packed_start, theta_packs_log};
use crate::sens::focei_kernel::{chol, chol_logdet, focei_htilde, OmegaT};
use crate::sens::rtape::DualH;

/// Directions per pass: the outer dual's width, and the lanes of each reverse sweep.
const K: usize = 8;

/// First order along the pass's directions, over `B` (taped: `R` at the bottom).
type Out<B = R> = DualT<K, B>;

/// The tape nodes a pass along `K` directions may grow to before `psi_along` spends its
/// directions over more, narrower passes (a node costs ~`20 + 8K` bytes with the sweep's
/// adjoints): ~350 MB per worker thread.
pub(super) const TAPE_NODE_BUDGET: usize = 4 << 20;

/// The leaves of a pass's tape: each packed coordinate, each `[covariate_nn]` raw output
/// of the reduced basis, each η.
struct Leaves<const N: usize> {
    x: Vec<u32>,
    z: Vec<u32>,
    eta: [u32; N],
}

impl<const N: usize> Leaves<N> {
    /// `∇_v s · d`: lane `lane` of a sweep's adjoints `adj` (node-major, `K` lanes) along
    /// the direction `d`.
    fn along(&self, adj: &[f64], lane: usize, d: &Dir<N>) -> f64 {
        let at = |i: u32| adj[i as usize * K + lane];
        let mut s = 0.0;
        if let Some(i) = d.x {
            s += at(self.x[i]);
        }
        if let Some(k) = d.z {
            s += at(self.z[k]);
        }
        for k in 0..N {
            if d.eta[k] != 0.0 {
                s += d.eta[k] * at(self.eta[k]);
            }
        }
        s
    }
}

/// A recorded pass: `F`, `g = ∇_η l`, the leaves they were taped from, and the tape's
/// length.
struct Pass<const N: usize> {
    f: Out,
    g: [Out; N],
    leaves: Leaves<N>,
    nodes: usize,
}

/// A direction of a pass: the packed coordinate it moves, the `[covariate_nn]` raw output
/// it moves (the reduced basis, [`NnReduction`]), and its η tangent.
#[derive(Clone, Copy)]
struct Dir<const N: usize> {
    x: Option<usize>,
    z: Option<usize>,
    eta: [f64; N],
}

impl<const N: usize> Dir<N> {
    fn x(i: usize, eta: [f64; N]) -> Self {
        Dir {
            x: Some(i),
            z: None,
            eta,
        }
    }

    #[cfg(feature = "nn")]
    fn eta(k: usize) -> Self {
        Dir {
            x: None,
            z: None,
            eta: std::array::from_fn(|m| if m == k { 1.0 } else { 0.0 }),
        }
    }

    #[cfg(feature = "nn")]
    fn z(k: usize) -> Self {
        Dir {
            x: None,
            z: Some(k),
            eta: [0.0; N],
        }
    }
}

/// The exact per-subject covariance Hessian of `objective` in packed coordinates
/// (`pack_params` order; the per-subject NLL Hessian, which the caller scales by 2 to the
/// OFV), or `None` outside the adjoint's scope. `eta_hat` is the mode (the stacked
/// `[η, κ]` under IOV) for `unpack_params(x)`.
pub(crate) fn subject_adjoint_cov_hessian(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    objective: Objective,
) -> Option<DMatrix<f64>> {
    let mut sc = scope(model, subject, template, x, eta_hat)?;
    let bucket = hessian_bucket(sc.width, objective)?;
    sc.eta.resize(bucket, 0.0);
    sc.bucket = bucket;
    let exact = matches!(objective, Objective::Agq { exact: true, .. });
    // A nested dual value is up to ~6 kB and the solvers keep several per state on the
    // stack: this runs on the fit pool's workers (`FIT_RAYON_STACK_SIZE`), on the caller's
    // thread so every fit-scoped override (`bloq_method`, …) it reads is the fit's.
    let h = match (bucket, exact) {
        (4, true) => hessian_with::<4, 10, DualH<4, 10, R>>(&sc, subject, template, x, objective),
        (8, true) => hessian_with::<8, 36, DualH<8, 36, R>>(&sc, subject, template, x, objective),
        (4, _) => hessian_with::<4, 10, DualT<4, R>>(&sc, subject, template, x, objective),
        (6, _) => hessian_with::<6, 21, DualT<6, R>>(&sc, subject, template, x, objective),
        (8, _) => hessian_with::<8, 36, DualT<8, R>>(&sc, subject, template, x, objective),
        (24, _) => hessian_with::<24, 300, DualT<24, R>>(&sc, subject, template, x, objective),
        (48, _) => hessian_with::<48, 1176, DualT<48, R>>(&sc, subject, template, x, objective),
        _ => None,
    };
    crate::sens::rtape::rtape_trim(TAPE_NODE_BUDGET / 4);
    h
}

/// The Hessian's dispatch widths: its own, shorter ladder (the covariance step runs once per
/// fit, so padding is cheap while every bucket instantiates the recorders over a nested
/// dual).
const HESSIAN_BUCKETS: [usize; 5] = [4, 6, 8, 24, 48];

/// The exact anchor's Hessian widths. Its inner η-dual carries `Q` as well as `(f, G)`, so
/// a value is `(1 + K) · (1 + N + N(N+1)/2)` taped scalars (16 bytes): ~2 kB at `N = 4`,
/// ~6.5 kB at `N = 8`; it stops at 8 (a 5- or 6-effect subject pads to 8), as the inner
/// recordings' `Q` payload is quadratic in the width.
const EXACT_ANCHOR_HESSIAN_BUCKETS: [usize; 2] = [4, 8];

/// The widest subject the exact anchor's Hessian serves (pinned by the width-rule test).
#[cfg(test)]
pub(crate) const EXACT_ANCHOR_HESSIAN_WIDTH: usize =
    EXACT_ANCHOR_HESSIAN_BUCKETS[EXACT_ANCHOR_HESSIAN_BUCKETS.len() - 1];

/// The Hessian dispatch bucket for a subject `width` random effects wide (the stacked or
/// compact `[η, κ]` under IOV) under `objective`, or `None` when none serves it: past the
/// widest bucket, or past `EXACT_ANCHOR_HESSIAN_BUCKETS` under the exact anchor. The one
/// width rule the dispatch runs on and [`cov_width_declines`] reports, so the covariance
/// step's decline diagnostics cannot promise a route the dispatch refuses.
pub(crate) fn hessian_bucket(width: usize, objective: Objective) -> Option<usize> {
    let ladder: &[usize] = if matches!(objective, Objective::Agq { exact: true, .. }) {
        &EXACT_ANCHOR_HESSIAN_BUCKETS
    } else {
        &HESSIAN_BUCKETS
    };
    ladder.iter().copied().find(|&b| b >= width)
}

/// Whether `subject` is too wide for any Hessian bucket under `objective`
/// ([`hessian_bucket`]); `false` when its width is not known (an unprepared nested-level
/// subject, which the model-level and per-subject declines report).
pub(crate) fn cov_width_declines(
    model: &CompiledModel,
    subject: &Subject,
    objective: Objective,
) -> bool {
    super::subject_width(model, subject).is_some_and(|w| hessian_bucket(w, objective).is_none())
}

/// The exact mode response `dη̂/dx` (one `n_eta`-vector, stacked `[η, κ]` under IOV, per
/// packed coordinate) at the EBE `eta_hat` of `unpack_params(x)`: the EBE warm-start
/// predictor's Jacobian (Almquist Eq. 48), FOCE's frozen-variance mode included.
pub(crate) fn subject_adjoint_eta_dx(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    interaction: bool,
) -> Option<Vec<DVector<f64>>> {
    let objective = if interaction {
        Objective::Focei
    } else {
        Objective::Foce
    };
    let mut sc = scope(model, subject, template, x, eta_hat)?;
    let bucket = hessian_bucket(sc.width, objective)?;
    sc.eta.resize(bucket, 0.0);
    sc.bucket = bucket;
    fn go<const N: usize, const M: usize, J: Jet<N>>(
        sc: &Scoped,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
        objective: Objective,
    ) -> Option<Vec<DVector<f64>>> {
        let ctx = LikCtx::new(sc.model, subject, &sc.params);
        let so = stacked_omega::<N>(sc.model, subject, template, &sc.params, sc.width);
        let pass = |dirs: &[Dir<N>]| {
            pass::<N, M, J>(&ctx, sc, subject, template, x, objective, &so, None, dirs)
        };
        let (_, eta_x, _) = mode_response::<N>(&pass, &x_dirs(x.len()))?;
        let cols: Vec<DVector<f64>> = (0..x.len())
            .map(|i| DVector::from_fn(sc.width, |k, _| eta_x[(k, i)]))
            .collect();
        cols.iter()
            .all(|c| c.iter().all(|v| v.is_finite()))
            .then_some(cols)
    }
    let r = match bucket {
        4 => go::<4, 10, DualT<4, R>>(&sc, subject, template, x, objective),
        6 => go::<6, 21, DualT<6, R>>(&sc, subject, template, x, objective),
        8 => go::<8, 36, DualT<8, R>>(&sc, subject, template, x, objective),
        24 => go::<24, 300, DualT<24, R>>(&sc, subject, template, x, objective),
        48 => go::<48, 1176, DualT<48, R>>(&sc, subject, template, x, objective),
        _ => None,
    };
    crate::sens::rtape::rtape_trim(TAPE_NODE_BUDGET / 4);
    r
}

/// The packed axes as pass directions, without an η tangent.
fn x_dirs<const N: usize>(n: usize) -> Vec<Dir<N>> {
    (0..n).map(|i| Dir::x(i, [0.0; N])).collect()
}

/// The mode's first-order response at the scoped point: `w = H⁻¹∇_η F` and the mode's
/// response `−H⁻¹∂g` along each of `along` (`N × along.len()`; `η′ = dη̂/dx` along the
/// packed axes), from one pass along no direction swept once per output (`F`, `g_k`), and
/// that pass's tape length.
fn mode_response<const N: usize>(
    pass: &impl Fn(&[Dir<N>]) -> Option<Pass<N>>,
    along: &[Dir<N>],
) -> Option<([f64; N], DMatrix<f64>, usize)> {
    let p = pass(&[])?;
    let outs: Vec<R> = std::iter::once(p.f.v)
        .chain(p.g.iter().map(|g| g.v))
        .collect();
    // ∇_η F, H = ∂g/∂η and ∂g along `along`, from the leaves' adjoints.
    let mut grad_f = DVector::<f64>::zeros(N);
    let mut h = DMatrix::<f64>::zeros(N, N);
    let mut gx = DMatrix::<f64>::zeros(N, along.len());
    for (c, chunk) in outs.chunks(K).enumerate() {
        let lanes: [R; K] = std::array::from_fn(|l| chunk.get(l).copied().unwrap_or(R::c(0.0)));
        let adj = rsweep::<K>(&lanes);
        for l in 0..chunk.len() {
            let at_eta = |k: usize| adj[p.leaves.eta[k] as usize * K + l];
            match c * K + l {
                0 => (0..N).for_each(|k| grad_f[k] = at_eta(k)),
                o => {
                    for k in 0..N {
                        h[(o - 1, k)] = at_eta(k);
                    }
                    for (j, d) in along.iter().enumerate() {
                        gx[(o - 1, j)] = p.leaves.along(adj, l, d);
                    }
                }
            }
        }
    }
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
    let wv = solve(&DMatrix::from_column_slice(N, 1, grad_f.as_slice()))?;
    let eta_x = -solve(&gx)?;
    (wv.iter().chain(eta_x.iter()).all(|v| v.is_finite()))
        .then(|| (std::array::from_fn(|k| wv[k]), eta_x, p.nodes))
}

/// Directions per `psi_along` pass: `K`, or fewer when a pass along `K` would outgrow
/// [`TAPE_NODE_BUDGET`]: a pass along `m` directions tapes up to `1 + 1.25m` times the
/// `nodes` of one along none (measured 5.4–10.5× at `m = 8` on the covariance fixtures).
pub(super) fn pass_width(nodes: usize) -> usize {
    let w = ((TAPE_NODE_BUDGET / nodes.max(1)).saturating_sub(1) * 4 / 5).clamp(1, K);
    #[cfg(test)]
    return PASS_WIDTH_CAP.with(|c| c.get()).map_or(w, |cap| w.min(cap));
    #[cfg(not(test))]
    w
}

#[cfg(test)]
thread_local! {
    /// Test switch: cap `psi_along`'s directions per pass, as a large tape would.
    pub(crate) static PASS_WIDTH_CAP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn hessian_with<const N: usize, const M: usize, J: Jet<N>>(
    sc: &Scoped,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    objective: Objective,
) -> Option<DMatrix<f64>> {
    let (model, params) = (sc.model, &sc.params);
    let n = x.len();
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega::<N>(model, subject, template, params, sc.width);
    #[cfg(feature = "nn")]
    if let Some(red) =
        NnReduction::build(model, subject, template, &params.theta).filter(|_| !nn_reduce_off())
    {
        let pass = |dirs: &[Dir<N>]| {
            pass::<N, M, J>(
                &ctx,
                sc,
                subject,
                template,
                x,
                objective,
                &so,
                Some(red.z.as_slice()),
                dirs,
            )
        };
        if let Some(h) = red.hessian::<N>(model, &pass, template, x, sc.width) {
            #[cfg(test)]
            NN_REDUCED_HITS.with(|c| c.set(c.get() + 1));
            return Some(h);
        }
    }
    let pass = |dirs: &[Dir<N>]| {
        pass::<N, M, J>(&ctx, sc, subject, template, x, objective, &so, None, dirs)
    };
    let (w, eta_x, nodes) = mode_response::<N>(&pass, &x_dirs(n))?;
    // Uᵀ∇²Ψ U along u_a = (e_a, η′e_a).
    let dirs: Vec<Dir<N>> = (0..n)
        .map(|i| Dir::x(i, std::array::from_fn(|k| eta_x[(k, i)])))
        .collect();
    let (_, hess) = psi_along(&pass, &w, &dirs, pass_width(nodes))?;
    hess.iter().all(|v| v.is_finite()).then_some(hess)
}

/// `∇Ψ_w` and `∇²Ψ_w` along `dirs`, `Ψ_w = F − wᵀg`, by reverse over forward: per pass,
/// `width` (≤ `K`) of the directions as the forward tangents, and one sweep whose lane `a`
/// is `∂Ψ_w/∂u_a`, so its leaf adjoints are the column `∇²Ψ_w u_a`, read along every
/// direction. Symmetrised: the two triangles agree to rounding.
fn psi_along<const N: usize>(
    pass: &impl Fn(&[Dir<N>]) -> Option<Pass<N>>,
    w: &[f64; N],
    dirs: &[Dir<N>],
    width: usize,
) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let n = dirs.len();
    let mut grad = DVector::<f64>::zeros(n);
    let mut hess = DMatrix::<f64>::zeros(n, n);
    for (c, chunk) in dirs.chunks(width.clamp(1, K)).enumerate() {
        let p = pass(chunk)?;
        let psi = p.f
            - <Out as PkNum>::wsum3(
                0.0,
                (0..N).map(|k| (Out::constant(w[k]), p.g[k], Out::constant(1.0))),
            );
        let adj = rsweep::<K>(&psi.g);
        for a in 0..chunk.len() {
            let i = c * width.clamp(1, K) + a;
            grad[i] = psi.g[a].v;
            for (j, d) in dirs.iter().enumerate() {
                hess[(j, i)] = p.leaves.along(adj, a, d);
            }
        }
    }
    Some((grad, hess.symmetric_part()))
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
/// The outer gradient (`run`) reads the same reduction: the outputs as tape leaves, and the
/// weights' adjoint as one backward pass through each network.
#[cfg(feature = "nn")]
pub(super) struct NnReduction {
    /// Raw outputs per network at the current weights.
    pub(super) z: Vec<Vec<f64>>,
    /// Per network: its first packed weight coordinate, its weight count, and its inputs.
    pub(super) nets: Vec<(usize, usize, std::collections::HashMap<String, f64>)>,
}

#[cfg(feature = "nn")]
impl NnReduction {
    /// `None` when the reduction does not hold (a weight read directly, an input that
    /// varies over the subject's records) or does not pay (fewer weights than outputs).
    pub(super) fn build(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        theta: &[f64],
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
            if names
                .iter()
                .any(|k| rep.get(k) != subject.covariates.get(k))
            {
                return None;
            }
            z.push(nn.mapper.forward_raw_g(&theta[nn.weights_offset..end], rep));
            nets.push((nn.weights_offset, n_w, rep.clone()));
        }
        let n_w: usize = nets.iter().map(|n| n.1).sum();
        let n_z: usize = z.iter().map(Vec::len).sum();
        (n_w > n_z).then_some(NnReduction { z, nets })
    }

    fn hessian<const N: usize>(
        &self,
        model: &CompiledModel,
        pass: &impl Fn(&[Dir<N>]) -> Option<Pass<N>>,
        template: &ModelParameters,
        x: &[f64],
        width: usize,
    ) -> Option<DMatrix<f64>> {
        let n = x.len();
        let n_z: usize = self.z.iter().map(Vec::len).sum();
        let mut is_w = vec![false; n];
        for &(off, n_w, _) in &self.nets {
            is_w[off..off + n_w].iter_mut().for_each(|b| *b = true);
        }
        let rest: Vec<usize> = (0..n).filter(|&i| !is_w[i]).collect();
        // The networks: J_z = ∂z/∂x_w and ∇²z_k, per network over its own weights.
        let (jz, hz) = self.network_jets(model, template, x)?;

        // The mode's response along the rest axes and the outputs.
        let along: Vec<Dir<N>> = rest
            .iter()
            .map(|&i| Dir::x(i, [0.0; N]))
            .chain((0..n_z).map(Dir::z))
            .collect();
        let (w, resp, nodes) = mode_response::<N>(pass, &along)?;
        // The bucket's padding is inert: no packed axis or output moves it, so the basis
        // spans the active effects only.
        if (width..N).any(|j| resp.row(j).iter().any(|&v| v != 0.0)) {
            return None;
        }
        let basis: Vec<Dir<N>> = rest
            .iter()
            .enumerate()
            .map(|(r, &i)| Dir::x(i, std::array::from_fn(|k| resp[(k, r)])))
            .chain((0..n_z).map(Dir::z))
            .chain((0..width).map(Dir::eta))
            .collect();
        let (gb, hb) = psi_along(pass, &w, &basis, pass_width(nodes))?;

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
        let mut hess = c.transpose() * hb * &c;
        for (k, net_h) in hz.iter().enumerate() {
            let gk = gb[nr + k];
            for &((a, b), v) in net_h {
                hess[(a, b)] += gk * v;
            }
        }
        let hess = hess.symmetric_part();
        hess.iter().all(|v| v.is_finite()).then_some(hess)
    }

    /// `J_z` as `(packed weight axis, ∂z/∂x_a over all outputs)` and per output `k` the
    /// nonzero `∂²z_k/∂x_a∂x_b` over its network's weights, through θ's packing chain.
    #[allow(clippy::type_complexity)]
    fn network_jets(
        &self,
        model: &CompiledModel,
        template: &ModelParameters,
        x: &[f64],
    ) -> Option<(Vec<(usize, Vec<f64>)>, Vec<Vec<((usize, usize), f64)>>)> {
        let n_z: usize = self.z.iter().map(Vec::len).sum();
        let mut jz: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut hz: Vec<Vec<((usize, usize), f64)>> = vec![Vec::new(); n_z];
        let mut z0 = 0;
        for (net, &(off, n_w, ref cov)) in self.nets.iter().enumerate() {
            let nn = &self.z[net];
            let mlp_out = nn.len();
            let mut jcols: Vec<Vec<f64>> = vec![vec![0.0; n_z]; n_w];
            // Reverse over forward, as `psi_along`: the weights in blocks of `K` as the
            // tangents, and per output one sweep whose lane `p` is `∂z_o/∂x_{blk[p]}`.
            for blk in (off..off + n_w).collect::<Vec<_>>().chunks(K) {
                rtape_reset();
                let leaves: Vec<R> = x[..template.theta.len()]
                    .iter()
                    .map(|&v| rinput(v))
                    .collect();
                let th = unpack_theta(template, |m| DualT {
                    v: leaves[m],
                    g: std::array::from_fn(|p| {
                        R::c(if blk.get(p) == Some(&m) { 1.0 } else { 0.0 })
                    }),
                });
                let out: Vec<Out> = model.covariate_nns[net]
                    .mapper
                    .forward_raw_g(&th[off..off + n_w], cov);
                if out.len() != mlp_out {
                    return None;
                }
                for (o, y) in out.iter().enumerate() {
                    for (p, &i) in blk.iter().enumerate() {
                        jcols[i - off][z0 + o] = y.g[p].v;
                    }
                    let adj = rsweep::<K>(&y.g);
                    for (p, &i) in blk.iter().enumerate() {
                        for j in off..off + n_w {
                            let v = adj[leaves[j].i as usize * K + p];
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
/// The inner η-dual the predictions carry, over the tape: `(f, G)` (`DualT`) or
/// `(f, G, Q)` (`DualH`, the exact anchor's).
trait Jet<const N: usize>: crate::sens::num::vm::VmNum {
    /// `η_k`'s value (a leaf) with its unit tangent.
    fn eta(v: R, k: usize) -> Self;
    /// A taped scalar with no η-dependence of its own.
    fn lift(v: R) -> Self;
    /// Component `c`: the value (0), `∂/∂η_k` (`1 + k`), packed `∂²/∂η_k∂η_l` (`1 + N + tri(k, l)`).
    fn part(&self, c: usize) -> R;
}

impl<const N: usize> Jet<N> for DualT<N, R> {
    fn eta(v: R, k: usize) -> Self {
        let mut g = [R::c(0.0); N];
        g[k] = R::c(1.0);
        DualT { v, g }
    }
    fn lift(v: R) -> Self {
        DualT {
            v,
            g: [R::c(0.0); N],
        }
    }
    fn part(&self, c: usize) -> R {
        if c == 0 {
            self.v
        } else {
            self.g[c - 1]
        }
    }
}

impl<const N: usize, const M: usize> Jet<N> for DualH<N, M, R> {
    fn eta(v: R, k: usize) -> Self {
        let mut g = [R::c(0.0); N];
        g[k] = R::c(1.0);
        DualH {
            v,
            g,
            h: [R::c(0.0); M],
        }
    }
    fn lift(v: R) -> Self {
        DualH {
            v,
            g: [R::c(0.0); N],
            h: [R::c(0.0); M],
        }
    }
    fn part(&self, c: usize) -> R {
        match c {
            0 => self.v,
            c if c <= N => self.g[c - 1],
            c => self.h[c - 1 - N],
        }
    }
}

/// The objective `F` and `g = ∇_η l` at the mode, first order along `dirs` over a fresh
/// reverse tape whose leaves are the packed coordinates, the reduced basis's outputs and η.
#[allow(clippy::too_many_arguments)]
fn pass<const N: usize, const M: usize, J: Jet<N>>(
    ctx: &LikCtx,
    sc: &Scoped,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    objective: Objective,
    so: &StackedOmega,
    nn_z: Option<&[Vec<f64>]>,
    dirs: &[Dir<N>],
) -> Option<Pass<N>> {
    debug_assert!(dirs.len() <= K);
    let (model, params, eta) = (sc.model, &sc.params, &sc.eta[..]);
    let n_obs = subject.observations.len();
    let seed = |i: usize| -> [R; K] {
        std::array::from_fn(|a| {
            R::c(if dirs.get(a).is_some_and(|d| d.x == Some(i)) {
                1.0
            } else {
                0.0
            })
        })
    };
    let tan = |k: usize| -> [R; K] {
        std::array::from_fn(|a| R::c(dirs.get(a).map_or(0.0, |d| d.eta[k])))
    };
    let lift_g = |g: [R; K]| -> [J; K] { g.map(J::lift) };

    rtape_reset();
    let xl: Vec<R> = x.iter().map(|&v| rinput(v)).collect();
    let el: [R; N] = std::array::from_fn(|k| rinput(eta[k]));

    // The reduced basis (`NnReduction`): the networks' raw outputs as leaves and as their
    // own directions, read by every recording below on both of its number types.
    #[cfg(feature = "nn")]
    let mut zl: Vec<R> = Vec::new();
    #[cfg(feature = "nn")]
    let preset = nn_z.map(|z| {
        let mut outer: Vec<Vec<Out<J>>> = Vec::with_capacity(z.len());
        let mut plain: Vec<Vec<Out>> = Vec::with_capacity(z.len());
        for net in z {
            let (mut o, mut p) = (Vec::with_capacity(net.len()), Vec::with_capacity(net.len()));
            for &v in net {
                let flat = zl.len();
                let leaf = rinput(v);
                zl.push(leaf);
                let g: [R; K] = std::array::from_fn(|a| {
                    R::c(if dirs.get(a).is_some_and(|d| d.z == Some(flat)) {
                        1.0
                    } else {
                        0.0
                    })
                });
                o.push(DualT {
                    v: J::lift(leaf),
                    g: lift_g(g),
                });
                p.push(DualT { v: leaf, g });
            }
            outer.push(o);
            plain.push(p);
        }
        NnPresetGuard::enter(vec![Box::new(outer), Box::new(plain)])
    });
    #[cfg(not(feature = "nn"))]
    let _ = nn_z;
    #[cfg(not(feature = "nn"))]
    let zl: Vec<R> = Vec::new();

    // The predictions over the outer dual, θ through its packing chain and η along its
    // tangent, each η also carrying its inner unit tangent.
    let th: Vec<Out<J>> = unpack_theta(template, |i| DualT {
        v: J::lift(xl[i]),
        g: lift_g(seed(i)),
    });
    let et: Vec<Out<J>> = (0..N)
        .map(|k| DualT {
            v: J::eta(el[k], k),
            g: lift_g(tan(k)),
        })
        .collect();
    let preds = record(
        model,
        subject,
        sc.layout.as_ref(),
        &params.theta,
        eta,
        &th,
        &et,
    )?;
    if preds.len() != n_obs {
        return None;
    }
    let comp = |y: &Out<J>, c: usize| -> Out {
        DualT {
            v: y.v.part(c),
            g: y.g.map(|d| d.part(c)),
        }
    };
    let f: Vec<Out> = preds.iter().map(|y| comp(y, 0)).collect();
    let g: Vec<[Out; N]> = preds
        .iter()
        .map(|y| std::array::from_fn(|k| comp(y, 1 + k)))
        .collect();

    // The population parameters along the same directions, each through its packing chain.
    let xo = |i: usize| -> Out {
        DualT {
            v: xl[i],
            g: seed(i),
        }
    };
    let theta = unpack_theta(template, xo);
    let p = population_params(model, subject, template, params, so, &theta, xo)?;
    let eo: [Out; N] = std::array::from_fn(|k| DualT {
        v: el[k],
        g: tan(k),
    });

    let out = match objective {
        Objective::Foce | Objective::Focei => {
            let interaction = objective == Objective::Focei;
            // FOCE scores R⁰ at the population prediction f(η = 0) on the same θ.
            let fpop: Option<Vec<Out>> =
                if !interaction && model.error_spec.has_f_dependent_variance() {
                    let et0 = vec![Out::<J>::constant(0.0); N];
                    let p0 = record(
                        sc.pop_model,
                        subject,
                        sc.layout.as_ref(),
                        &params.theta,
                        &[0.0; N],
                        &th,
                        &et0,
                    )?;
                    (p0.len() == n_obs).then(|| p0.iter().map(|y| comp(y, 0)).collect())
                } else {
                    None
                };
            objective_terms(ctx, interaction, &f, &g, &eo, fpop.as_deref(), &p)
        }
        Objective::Agq { n_agq, exact } => {
            // The anchor H_a — the exact Hessian over an η-dual whose predictions carry
            // (f, G, Q), or FOCEI's H̃ — jittered as production's proposal is.
            let mut h_a: Vec<Vec<Out>> = if exact {
                type E<const N: usize, const M: usize> = DualH<N, M, Out>;
                let q = |y: &Out<J>| -> [Out; M] { std::array::from_fn(|t| comp(y, 1 + N + t)) };
                let fh: Vec<E<N, M>> = preds
                    .iter()
                    .zip(&g)
                    .map(|(y, gj)| DualH {
                        v: comp(y, 0),
                        g: *gj,
                        h: q(y),
                    })
                    .collect();
                let eh: [E<N, M>; N] = std::array::from_fn(|k| {
                    let mut gk = [Out::constant(0.0); N];
                    gk[k] = Out::constant(1.0);
                    DualH {
                        v: eo[k],
                        g: gk,
                        h: [Out::constant(0.0); M],
                    }
                });
                let l = inner_l(
                    ctx,
                    &fh,
                    &eh,
                    &p.map(|v| DualH {
                        v,
                        g: [Out::constant(0.0); N],
                        h: [Out::constant(0.0); M],
                    }),
                    None,
                )?;
                (0..N)
                    .map(|a| (0..N).map(|b| l.hess(a, b)).collect())
                    .collect()
            } else {
                focei_htilde(ctx, &f, &g, &eo, &p)?
            };
            regularise(&mut h_a);
            // Production's (and the gradient's) fallback when the regularised anchor is not
            // positive-definite: the prior precision `Ω⁻¹` scales the grid and enters `½log|·|`
            // (`build_proposal`), so the R-matrix differentiates that objective too.
            let lh = chol(&h_a).or_else(|| chol(&p.omega.inv))?;
            let half_logdet = Out::constant(0.5) * chol_logdet(&lh);
            let a = if n_agq == 1 {
                inner_l(ctx, &f, &eo, &p, None)? + half_logdet
            } else {
                let th_o: Vec<Out> = unpack_theta(template, xo);
                let mut terms: Vec<Out> = Vec::new();
                for (z, c) in agq_grid::<N>(n_agq, sc.width) {
                    // b = η + √2·L⁻ᵀz, and the predictions recorded at b.
                    let mut step = [Out::constant(0.0); N];
                    for i in (0..N).rev() {
                        let mut acc = Out::constant(z[i]);
                        for j in (i + 1)..N {
                            acc = acc - lh[j][i] * step[j];
                        }
                        step[i] = acc / lh[i][i];
                    }
                    let b: [Out; N] = std::array::from_fn(|k| {
                        eo[k] + Out::constant(std::f64::consts::SQRT_2) * step[k]
                    });
                    let bv: Vec<f64> = b.iter().map(|v| v.val()).collect();
                    let fb = record(
                        model,
                        subject,
                        sc.layout.as_ref(),
                        &params.theta,
                        &bv,
                        &th_o,
                        &b,
                    )
                    .filter(|q| q.len() == n_obs)?;
                    terms.push(Out::constant(c) - inner_l(ctx, &fb, &b, &p, None)?);
                }
                let mx = terms
                    .iter()
                    .map(|t| t.val())
                    .fold(f64::NEG_INFINITY, f64::max);
                let one = Out::constant(1.0);
                let sum = <Out as PkNum>::wsum3(
                    0.0,
                    terms
                        .iter()
                        .map(|&t| (one, (t - Out::constant(mx)).exp(), one)),
                );
                half_logdet - (Out::constant(mx) + sum.ln())
            };
            Some((a, inner_grad(ctx, &f, &g, &eo, &p, None)?))
        }
    };
    // A recording that did not read the seeded outputs differentiated the wrong function.
    #[cfg(feature = "nn")]
    if preset.as_ref().is_some_and(NnPresetGuard::missed) {
        return None;
    }
    #[cfg(feature = "nn")]
    drop(preset);
    let (f, g) = out?;
    Some(Pass {
        f,
        g,
        leaves: Leaves {
            x: xl.iter().map(|r| r.i).collect(),
            z: zl.iter().map(|r| r.i).collect(),
            eta: el.map(|r| r.i),
        },
        nodes: crate::sens::rtape::rtape_len(),
    })
}

/// `(F, ∇_η l)` for FOCE / FOCEI (`objective`).
fn objective_terms<const N: usize>(
    ctx: &LikCtx,
    interaction: bool,
    f: &[Out],
    g: &[[Out; N]],
    eta: &[Out; N],
    fpop: Option<&[Out]>,
    p: &ParT<Out>,
) -> Option<(Out, [Out; N])> {
    objective::<Out, N>(ctx, interaction, f, g, eta, fpop, p)
}

/// The tensor Gauss–Hermite grid over the active dimensions (inert padding at `z = 0`):
/// per node its `z` and `c = Σ log w + ‖z‖²`, as the gradient's grid.
fn agq_grid<const N: usize>(n_agq: usize, active: usize) -> Vec<([f64; N], f64)> {
    let (nodes, _, logw) = crate::estimation::agq::cached_gauss_hermite(n_agq);
    let mut out = Vec::new();
    let mut idx = [0usize; N];
    loop {
        let z: [f64; N] = std::array::from_fn(|k| if k < active { nodes[idx[k]] } else { 0.0 });
        out.push((z, (0..active).map(|k| logw[idx[k]] + z[k] * z[k]).sum()));
        let mut k = 0;
        while k < active {
            idx[k] += 1;
            if idx[k] < n_agq {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
        if k == active {
            return out;
        }
    }
}

/// Ω (Cholesky entries), σ, ρ, the serial ρ and the custom residual magnitude as the
/// kernel reads them, from the packed coordinates `xo` through their packing chains.
fn population_params(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    params: &ModelParameters,
    so: &StackedOmega,
    theta: &[Out],
    xo: impl Fn(usize) -> Out,
) -> Option<ParT<Out>> {
    let mut l: Vec<Vec<Out>> = so
        .chol
        .iter()
        .map(|r| r.iter().map(|&v| Out::constant(v)).collect())
        .collect();
    for (idx, at, _) in &so.placed {
        let e = if at[0].0 == at[0].1 {
            xo(*idx).exp()
        } else {
            xo(*idx)
        };
        for &(i, j) in at {
            l[i][j] = e;
        }
    }
    let seg = packed_segments(template);
    let rho_start = rho_packed_start(template);
    let ruv_mult = match model.ruv_magnitude.as_ref().filter(|rm| rm.is_active()) {
        Some(rm) => Some(
            (0..subject.observations.len())
                .map(|j| {
                    rm.eval_obs_g(
                        theta,
                        subject.obs_cov(j),
                        subject.readout_time(j),
                        rm.tad_for(subject, j),
                    )
                })
                .collect::<Option<Vec<_>>>()?,
        ),
        None => None,
    };
    Some(ParT {
        omega: OmegaT::from_chol(l)?,
        sigma: (0..params.sigma.values.len())
            .map(|k| xo(seg.sigma_start() + k).exp())
            .collect(),
        // ρ = tanh(z), the Fisher-z packing (`rho_chain` = 1 − ρ²).
        rho: (0..params.residual_correlations.len())
            .map(|k| {
                let e = (xo(rho_start + k) * Out::constant(2.0)).exp();
                (e - Out::constant(1.0)) / (e + Out::constant(1.0))
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
        .map(|m| {
            if theta_packs_log(template.theta_lower[m]) {
                xd(m).exp()
            } else {
                xd(m)
            }
        })
        .collect()
}
