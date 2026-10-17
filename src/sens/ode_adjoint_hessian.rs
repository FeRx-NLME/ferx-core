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
//! AGQ marginal under either anchor) and `g`, over a forward second-order dual along up to
//! `K` directions in `(x, η)`, through the same recorders and kernel as the gradient. The
//! predictions carry an inner η-dual, so `G` (and under the exact anchor `Q`) are
//! differentiated with them. Passes along the η axes give `∇_η F` and `H`; along the x
//! axes, `∂ₓg`; along pairs of the `u_a`, `Uᵀ∇²Ψ_w U`. There are no third-order
//! sensitivity tables and no finite differences, and FOCE's frozen-variance mode (whose
//! `R⁰` moves with θ through `f(η = 0)`) is just another term of `g`.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::estimation::parameterization::{packed_segments, rho_packed_start, theta_packs_log};
use crate::sens::focei_kernel::{chol, chol_logdet, focei_htilde, OmegaT};
use crate::sens::rtape::DualH;

/// Directions per pass: the outer dual's width. A pass carries `K(K+1)/2` second-order
/// entries per value, so the Hessian is assembled from passes over pairs of `K/2`-blocks.
const K: usize = 8;
const MK: usize = K * (K + 1) / 2;

/// Second order along the pass's directions, over `B`.
type Out<B = f64> = DualH<K, MK, B>;

/// A direction of a pass: the packed coordinate it moves (`None` for a pure η
/// direction) and its η tangent.
type Dir<const N: usize> = (Option<usize>, [f64; N]);

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
    let mut sc = scope(
        model,
        subject,
        template,
        x,
        eta_hat,
        objective != Objective::Foce,
    )?;
    // Its own, shorter bucket ladder (the covariance step runs once per fit, so padding is
    // cheap while every bucket instantiates the recorders over a nested dual). The inner
    // η-dual carries `(f, G)`; only the exact anchor, which differentiates `Q` as well,
    // carries it too, and only up to width 4 so a value stays a few kilobytes.
    let bucket = HESSIAN_BUCKETS.into_iter().find(|&b| b >= sc.width)?;
    sc.eta.resize(bucket, 0.0);
    sc.bucket = bucket;
    let exact = matches!(objective, Objective::Agq { exact: true, .. });
    // A nested dual value is up to ~10 kB and the solvers keep several per state on the
    // stack: this runs on the fit pool's workers (`FIT_RAYON_STACK_SIZE`), on the caller's
    // thread so every fit-scoped override (`bloq_method`, …) it reads is the fit's.
    match (bucket, exact) {
        (4, true) => hessian_with::<4, 10, DualH<4, 10, f64>>(&sc, subject, template, x, objective),
        (_, true) => None,
        (4, _) => hessian_with::<4, 10, DualT<4, f64>>(&sc, subject, template, x, objective),
        (8, _) => hessian_with::<8, 36, DualT<8, f64>>(&sc, subject, template, x, objective),
        (24, _) => hessian_with::<24, 300, DualT<24, f64>>(&sc, subject, template, x, objective),
        _ => None,
    }
}

/// The Hessian's dispatch widths (see [`subject_adjoint_cov_hessian`]).
const HESSIAN_BUCKETS: [usize; 3] = [4, 8, 24];

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
    let mut sc = scope(model, subject, template, x, eta_hat, interaction)?;
    let bucket = HESSIAN_BUCKETS.into_iter().find(|&b| b >= sc.width)?;
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
        let pass =
            |dirs: &[Dir<N>]| pass::<N, M, J>(&ctx, sc, subject, template, x, objective, &so, dirs);
        let (_, eta_x) = mode_response::<N>(&pass, x.len())?;
        let cols: Vec<DVector<f64>> = (0..x.len())
            .map(|i| DVector::from_fn(sc.width, |k, _| eta_x[(k, i)]))
            .collect();
        cols.iter()
            .all(|c| c.iter().all(|v| v.is_finite()))
            .then_some(cols)
    }
    match bucket {
        4 => go::<4, 10, DualT<4, f64>>(&sc, subject, template, x, objective),
        8 => go::<8, 36, DualT<8, f64>>(&sc, subject, template, x, objective),
        24 => go::<24, 300, DualT<24, f64>>(&sc, subject, template, x, objective),
        _ => None,
    }
}

/// The mode's first-order response at the scoped point: `w = H⁻¹∇_η F` and
/// `η′ = dη̂/dx = −H⁻¹∂ₓg` (`N × n`), from passes along the η and the x axes.
fn mode_response<const N: usize>(
    pass: &impl Fn(&[Dir<N>]) -> Option<(Out, [Out; N])>,
    n: usize,
) -> Option<([f64; N], DMatrix<f64>)> {
    // ∇_η F and H = ∂g/∂η, along the η axes.
    let mut grad_f = DVector::<f64>::zeros(N);
    let mut h = DMatrix::<f64>::zeros(N, N);
    for block in (0..N).collect::<Vec<_>>().chunks(K) {
        let dirs: Vec<Dir<N>> = block
            .iter()
            .map(|&k| {
                (
                    None,
                    std::array::from_fn(|m| if m == k { 1.0 } else { 0.0 }),
                )
            })
            .collect();
        let (fv, lg) = pass(&dirs)?;
        for (a, &k) in block.iter().enumerate() {
            grad_f[k] = fv.g[a];
            for i in 0..N {
                h[(i, k)] = lg[i].g[a];
            }
        }
    }
    let hc = h.symmetric_part().cholesky()?;
    let wv = hc.solve(&grad_f);
    // ∂ₓg along the x axes.
    let mut gx = DMatrix::<f64>::zeros(N, n);
    for block in (0..n).collect::<Vec<_>>().chunks(K) {
        let dirs: Vec<Dir<N>> = block.iter().map(|&i| (Some(i), [0.0; N])).collect();
        let (_, lg) = pass(&dirs)?;
        for (a, &i) in block.iter().enumerate() {
            for k in 0..N {
                gx[(k, i)] = lg[k].g[a];
            }
        }
    }
    Some((std::array::from_fn(|k| wv[k]), -hc.solve(&gx)))
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
    let pass =
        |dirs: &[Dir<N>]| pass::<N, M, J>(&ctx, sc, subject, template, x, objective, &so, dirs);
    let (w, eta_x) = mode_response::<N>(&pass, n)?;

    // Uᵀ∇²Ψ U along u_a = (e_a, η′e_a): one pass when every coordinate fits, else one per
    // pair of K/2-blocks.
    let dir = |i: usize| -> Dir<N> { (Some(i), std::array::from_fn(|k| eta_x[(k, i)])) };
    let blocks: Vec<Vec<usize>> = if n <= K {
        vec![(0..n).collect()]
    } else {
        (0..n)
            .collect::<Vec<_>>()
            .chunks(K / 2)
            .map(<[usize]>::to_vec)
            .collect()
    };
    let mut hess = DMatrix::<f64>::zeros(n, n);
    for (bi, a) in blocks.iter().enumerate() {
        for b in &blocks[bi..] {
            let idx: Vec<usize> = if std::ptr::eq(a, b) {
                a.clone()
            } else {
                a.iter().chain(b).copied().collect()
            };
            let (fv, lg) = pass(&idx.iter().map(|&i| dir(i)).collect::<Vec<_>>())?;
            let psi = fv
                - <Out as PkNum>::wsum3(
                    0.0,
                    (0..N).map(|k| (Out::constant(w[k]), lg[k], Out::constant(1.0))),
                );
            for (p, &i) in idx.iter().enumerate() {
                for (q, &j) in idx.iter().enumerate() {
                    hess[(i, j)] = psi.h[crate::sens::vtape::tri(p, q)];
                }
            }
        }
    }
    hess.iter().all(|v| v.is_finite()).then_some(hess)
}

/// The inner η-dual the predictions carry: `(f, G)` (`DualT`) or `(f, G, Q)`
/// (`DualH`, the exact anchor's).
trait Jet<const N: usize>: crate::sens::num::vm::VmNum {
    /// `η_k`'s value with its unit tangent.
    fn eta(v: f64, k: usize) -> Self;
    /// Component `c`: the value (0), `∂/∂η_k` (`1 + k`), packed `∂²/∂η_k∂η_l` (`1 + N + tri(k, l)`).
    fn part(&self, c: usize) -> f64;
}

impl<const N: usize> Jet<N> for DualT<N, f64> {
    fn eta(v: f64, k: usize) -> Self {
        let mut g = [0.0; N];
        g[k] = 1.0;
        DualT { v, g }
    }
    fn part(&self, c: usize) -> f64 {
        if c == 0 {
            self.v
        } else {
            self.g[c - 1]
        }
    }
}

impl<const N: usize, const M: usize> Jet<N> for DualH<N, M, f64> {
    fn eta(v: f64, k: usize) -> Self {
        let mut g = [0.0; N];
        g[k] = 1.0;
        DualH { v, g, h: [0.0; M] }
    }
    fn part(&self, c: usize) -> f64 {
        match c {
            0 => self.v,
            c if c <= N => self.g[c - 1],
            c => self.h[c - 1 - N],
        }
    }
}

/// The objective `F` and `g = ∇_η l` at the mode, second order along `dirs`.
#[allow(clippy::too_many_arguments)]
fn pass<const N: usize, const M: usize, J: Jet<N>>(
    ctx: &LikCtx,
    sc: &Scoped,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    objective: Objective,
    so: &StackedOmega,
    dirs: &[Dir<N>],
) -> Option<(Out, [Out; N])> {
    debug_assert!(dirs.len() <= K);
    let (model, params, eta) = (sc.model, &sc.params, &sc.eta[..]);
    let n_obs = subject.observations.len();
    let seed = |i: usize| -> [f64; K] {
        std::array::from_fn(|a| {
            if dirs.get(a).is_some_and(|d| d.0 == Some(i)) {
                1.0
            } else {
                0.0
            }
        })
    };
    let tan =
        |k: usize| -> [f64; K] { std::array::from_fn(|a| dirs.get(a).map_or(0.0, |d| d.1[k])) };
    let zero_h = || [J::from_f64(0.0); MK];

    // The predictions over the outer dual, θ through its packing chain and η along its
    // tangent, each η also carrying its inner unit tangent.
    let th: Vec<Out<J>> = unpack_theta(template, |i| DualH {
        v: J::from_f64(x[i]),
        g: seed(i).map(J::from_f64),
        h: zero_h(),
    });
    let et: Vec<Out<J>> = (0..N)
        .map(|k| DualH {
            v: J::eta(eta[k], k),
            g: tan(k).map(J::from_f64),
            h: zero_h(),
        })
        .collect();
    let preds = record(model, subject, &params.theta, eta, &th, &et)?;
    if preds.len() != n_obs {
        return None;
    }
    let comp = |y: &Out<J>, c: usize| -> Out {
        DualH {
            v: y.v.part(c),
            g: y.g.map(|d| d.part(c)),
            h: y.h.map(|d| d.part(c)),
        }
    };
    let f: Vec<Out> = preds.iter().map(|y| comp(y, 0)).collect();
    let g: Vec<[Out; N]> = preds
        .iter()
        .map(|y| std::array::from_fn(|k| comp(y, 1 + k)))
        .collect();

    // The population parameters along the same directions, each through its packing chain.
    let xo = |i: usize| -> Out {
        DualH {
            v: x[i],
            g: seed(i),
            h: [0.0; MK],
        }
    };
    let theta = unpack_theta(template, xo);
    let p = population_params(model, subject, template, params, so, &theta, xo)?;
    let eo: [Out; N] = std::array::from_fn(|k| DualH {
        v: eta[k],
        g: tan(k),
        h: [0.0; MK],
    });

    match objective {
        Objective::Foce | Objective::Focei => {
            let interaction = objective == Objective::Focei;
            // FOCE scores R⁰ at the population prediction f(η = 0) on the same θ.
            let fpop: Option<Vec<Out>> =
                if !interaction && model.error_spec.has_f_dependent_variance() {
                    let et0 = vec![Out::<J>::constant(0.0); N];
                    let p0 = record(model, subject, &params.theta, &[0.0; N], &th, &et0)?;
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
            let lh = chol(&h_a)?;
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
                    let bv: Vec<f64> = b.iter().map(|v| v.v).collect();
                    let fb = record(model, subject, &params.theta, &bv, &th_o, &b)
                        .filter(|q| q.len() == n_obs)?;
                    terms.push(Out::constant(c) - inner_l(ctx, &fb, &b, &p, None)?);
                }
                let mx = terms.iter().map(|t| t.v).fold(f64::NEG_INFINITY, f64::max);
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
    }
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
