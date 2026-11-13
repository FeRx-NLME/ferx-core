//! The AGQ / Laplace gradient through the subject's compiled plan (#2786): [`run_agq`]'s
//! quantity, with no taped recording.
//!
//! - At `η̂`, the second-order forward is kept ([`plan_forward`]).
//! - The objective `A` and `∇_η l(η̂)` come from the subject's kernel plan ([`kernel`]),
//!   one multi-seed reverse giving every output's adjoint at every leaf. A node's
//!   predictions are leaves of it and its location `b_j` an output: a node's reverse turns
//!   `A`'s adjoint of its predictions into `Gᵀā`, which one more kernel reverse seeded on
//!   the `b_j` carries to the leaves `b_j` is built from — exact, with `G` never formed.
//! - Reverse 1 of the kept forward is seeded with `A`'s adjoints of `f`, `G` and `Q`. Its
//!   η lanes are the exact `∂A/∂η` through the predictions (the third-order `∂Q/∂η`
//!   contraction included, no eigen-directions), and its θ lanes are `∂A/∂θ` there.
//! - Reverse 2 seeds `−wᵀ∇_η l` and the mode-response rows, all fixed combinations of
//!   the `∇_η l` lanes.
//! - A node's predictions are a value forward at `b_j` (of the plan whose guard holds
//!   there), kept for that value-only reverse.
//! - An endpoint-only subject (no prediction rows) has no plan: its term alone.
use super::*;
use crate::sens::num::vm::Layout;

/// The quadrature (AGQ, Laplace) gradient of a subject through its plans; `None` where a
/// plan cannot be recorded.
#[allow(clippy::too_many_arguments)]
pub(super) fn agq_planned<const N: usize, const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    template: &ModelParameters,
    params: &ModelParameters,
    x: &[f64],
    eta: &[f64],
    ctx: &LikCtx,
    so: &StackedOmega,
    pv: &ParT<f64>,
    grid: &[([f64; N], f64, [f64; N])],
    h_inner: &DMatrix<f64>,
    exact: bool,
    active: usize,
    held: bool,
    ng: Option<&AgqNg>,
    eta_dx: Option<&mut ModeOut>,
) -> Option<Vec<f64>> {
    let theta = &params.theta;
    let n_obs = subject.observations.len();
    let n_bsv = model.n_eta;
    let full = terms::layout2(N);
    let lf = full.lanes();
    let mode_fwd = if n_obs == 0 {
        None
    } else {
        Some(plan_forward(model, subject, lay, theta, eta, full)?)
    };
    let p1 = PredLanes::<N> {
        du: mode_fwd.as_ref().map_or(&[][..], |p| p.1.as_slice()),
    };
    let n_th = theta.len();
    // The node predictions, values only, each forward kept for its one reverse: the tape takes
    // a node as a block of `b` ([`node_block`]), whose reverse returns `Gᵀ·adjoint` — all `b`
    // needs — and the node's θ adjoint with it, so `G` itself is never formed.
    type Node = (Vec<f64>, PlanArc, std::sync::Arc<kj_fwd::Fwd>);
    let mut nodes: Vec<Node> = Vec::with_capacity(grid.len());
    for (_, _, b) in grid {
        if n_obs == 0 {
            continue;
        }
        let (plan, du, f) = plan_forward(model, subject, lay, theta, &b[..], Layout::F64)?;
        nodes.push((du, plan, std::sync::Arc::new(f)));
    }

    // ---- the objective and ∇_η l from the kernel plan --------------------------------
    // Every input that moves between calls is a leaf: the predictions and their jets at η̂
    // (`Q` too, which only the exact anchor reads), η̂, the parameters, the non-Gaussian
    // term's jets and its node values, and per node its grid point, weight, location and
    // predictions.
    use kernel::Leaf;
    let mut inp = kernel::Inputs::default();
    let fr: Vec<Leaf> = (0..n_obs).map(|j| inp.leaf(p1.v(j))).collect();
    let gr: Vec<[Leaf; N]> = (0..n_obs)
        .map(|j| std::array::from_fn(|k| inp.leaf(p1.g(j, k))))
        .collect();
    let qr: Vec<[Leaf; M]> = (0..n_obs)
        .map(|j| {
            std::array::from_fn(|t| {
                if exact {
                    inp.leaf(p1.q(j, t))
                } else {
                    Leaf::c(0.0)
                }
            })
        })
        .collect();
    let er: [Leaf; N] = std::array::from_fn(|k| inp.leaf(eta[k]));
    let (pr, om) = param_leaves(pv, so, &mut inp);
    let ng_r = ng.map(|ng| ng_leaves::<N>(&ng.jets, &mut inp));
    let node_qr: Vec<Leaf> = ng
        .map(|ng| ng.nodes.iter().map(|(q, _)| inp.leaf(*q)).collect())
        .unwrap_or_default();
    let node_gq: Vec<Vec<Leaf>> = ng
        .map(|ng| {
            ng.nodes
                .iter()
                .map(|(_, gq)| gq.iter().take(n_bsv).map(|&v| inp.leaf(v)).collect())
                .collect()
        })
        .unwrap_or_default();
    #[allow(clippy::type_complexity)]
    let grid_l: Vec<([Leaf; N], Leaf, [Leaf; N])> = grid
        .iter()
        .map(|(z, c, b0)| {
            (
                z.map(|v| inp.leaf(v)),
                inp.leaf(*c),
                b0.map(|v| inp.leaf(v)),
            )
        })
        .collect();
    let node_f: Vec<Vec<Leaf>> = nodes
        .iter()
        .map(|(fv, _, _)| fv.iter().map(|&v| inp.leaf(v)).collect())
        .collect();
    let n_nodes_b = nodes.len();
    let n_out = 1 + N + n_nodes_b * N;
    let variant = kernel::variant_of(&(
        4u8,
        (N, M),
        exact,
        active,
        grid.len(),
        n_nodes_b,
        ng.map(|n| (n.jets.g.len(), n.nodes.len())),
        &so.placed.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
    ));
    type T = kj_tr::Tr;
    let kr = kernel::kernel_run(
        model,
        subject,
        variant,
        Layout::F64,
        &inp.vals,
        n_out,
        &|lv| {
            use crate::sens::num::PkNum;
            let t = |l: Leaf| l.tr(lv);
            let c = T::from_f64;
            let one = c(1.0);
            let fr: Vec<T> = fr.iter().map(|&l| t(l)).collect();
            let gr: Vec<[T; N]> = gr.iter().map(|r| r.map(t)).collect();
            let qr: Vec<[T; M]> = qr.iter().map(|r| r.map(t)).collect();
            let er: [T; N] = er.map(t);
            let pr = param_tr(model, subject, &pr, lv)?;
            let ngr = ng_r.as_ref().map(|n| ng_tr(n, lv));
            let mut h_a: Vec<Vec<T>> = if exact {
                crate::sens::focei_kernel::exact_inner_hessian_kernel(
                    ctx, &fr, &gr, &qr, &er, &pr, active,
                )?
            } else {
                crate::sens::focei_kernel::focei_htilde(ctx, &fr, &gr, &er, &pr)?
            };
            if let Some(ngr) = ngr.as_ref() {
                for k in 0..n_bsv {
                    for l in 0..n_bsv {
                        h_a[k][l] = h_a[k][l] + ngr.h[k][l];
                    }
                }
            }
            let lh = factor_anchor(&mut h_a, &pr.omega.inv)?;
            let half_logdet = c(0.5) * crate::sens::focei_kernel::chol_logdet(&lh);
            let ng_q = ngr.as_ref().map_or(c(0.0), |n| n.q);
            let mut bs: Vec<T> = Vec::with_capacity(n_nodes_b * N);
            let a = if grid_l.is_empty() {
                inner_l(ctx, &fr, &er, &pr, None)? + ng_q + half_logdet
            } else {
                let mut terms: Vec<T> = Vec::with_capacity(grid_l.len());
                for (j, (z, cw, b0)) in grid_l.iter().enumerate() {
                    // b = η̂ + √2·L⁻ᵀz.
                    let mut step = [c(0.0); N];
                    for i in (0..N).rev() {
                        let mut acc = t(z[i]);
                        for jj in (i + 1)..N {
                            acc = acc - lh[jj][i] * step[jj];
                        }
                        step[i] = acc / lh[i][i];
                    }
                    let b: [T; N] =
                        std::array::from_fn(|k| er[k] + c(std::f64::consts::SQRT_2) * step[k]);
                    let fb: Vec<T> = match node_f.get(j) {
                        Some(f) => {
                            bs.extend(b);
                            f.iter().map(|&l| t(l)).collect()
                        }
                        None => Vec::new(),
                    };
                    let qb = match ng {
                        Some(_) => {
                            let gq = &node_gq[j];
                            t(node_qr[j])
                                + T::wsum3(
                                    0.0,
                                    (0..n_bsv).map(|k| (t(gq[k]), b[k] - t(b0[k]), one)),
                                )
                        }
                        None => c(0.0),
                    };
                    terms.push(t(*cw) - (inner_l(ctx, &fb, &b, &pr, None)? + qb));
                }
                // The log-sum-exp's shift: the terms' largest, recorded, with no derivative (the
                // sum is invariant to it).
                let mx = terms[1..]
                    .iter()
                    .fold(terms[0], |m, &x| {
                        T::select_if(crate::sens::num::BranchCmp::Gt, x, m, x, m)
                    })
                    .value_only();
                let sum = T::wsum3(0.0, terms.iter().map(|&x| (one, (x - mx).exp(), one)));
                c(0.5 * active as f64 * std::f64::consts::PI.ln()) + half_logdet - (mx + sum.ln())
            };
            let mut lg = inner_grad(ctx, &fr, &gr, &er, &pr, None)?;
            if let Some(ngr) = ngr.as_ref() {
                for k in 0..N {
                    lg[k] = lg[k] + ngr.g[k];
                }
            }
            let mut outs = vec![a];
            outs.extend(lg);
            outs.extend(bs);
            Some(outs)
        },
    )?;
    // The adjoint rows of `A` and of each `∇_η l` component.
    let unit =
        |o: usize| -> Vec<f64> { (0..n_out).map(|c| if c == o { 1.0 } else { 0.0 }).collect() };
    let mut base = kr.reverse_values(&(0..=N).map(unit).collect::<Vec<_>>())?;
    // `A` through every node's predictions: `ā` of them, back through the node's plan to its
    // location (`Gᵀā`) and θ, and the locations' adjoints through the kernel to the leaves.
    let mut theta_nodes = vec![0.0; n_th];
    if n_nodes_b > 0 {
        let mut seed = vec![0.0; n_out];
        for (j, (_, plan, ff)) in nodes.iter().enumerate() {
            let du: Vec<f64> = node_f[j].iter().map(|&l| l.adj(&base[0])).collect();
            if du.iter().all(|&x| x == 0.0) {
                continue;
            }
            let (mut ub, mut pb) = (vec![0.0; N], vec![0.0; n_th]);
            if !plan.reverse_from(ff, Layout::F64, 1, &mut ub, &mut pb, &du) {
                return None;
            }
            seed[1 + N + j * N..1 + N + (j + 1) * N].copy_from_slice(&ub[..N]);
            for (t, d) in theta_nodes.iter_mut().zip(&pb) {
                *t += d;
            }
        }
        if seed.iter().any(|&x| x != 0.0) {
            let corr = kr.reverse_values(&[seed])?.pop()?;
            for (a, c) in base[0].iter_mut().zip(&corr) {
                *a += c;
            }
        }
    }
    let ba = |o: usize, v: Leaf| v.adj(&base[o]);

    // ---- reverse 1: A through the predictions at η̂ ------------------------------------
    // Seeds of output combination `c` (over the lanes `[A, ∇_η l]`) on (f, G, Q).
    let seed_preds = |c: &[f64], du: &mut [f64]| {
        let comb = |v: Leaf| -> f64 {
            c.iter()
                .enumerate()
                .filter(|(_, &k)| k != 0.0)
                .map(|(o, &k)| k * ba(o, v))
                .sum()
        };
        for j in 0..n_obs {
            du[j * lf] = comb(fr[j]);
            for k in 0..N {
                du[j * lf + 1 + k] = comb(gr[j][k]);
            }
            if exact {
                for t in 0..M {
                    du[j * lf + 1 + N + t] = comb(qr[j][t]);
                }
            }
        }
    };
    let (nu, np, w_out) = (N * lf, n_th * lf, n_obs * lf);
    let mut a_seed = vec![0.0; 1 + N];
    a_seed[0] = 1.0;
    let mut du_bar = vec![0.0; w_out];
    seed_preds(&a_seed, &mut du_bar);
    let (mut ub, mut pb) = (vec![0.0; nu], vec![0.0; np]);
    if let Some((plan, _, fwd)) = mode_fwd.as_ref() {
        if !plan.reverse_from(fwd, full, 1, &mut ub, &mut pb, &du_bar) {
            return None;
        }
    }
    let mut grad_eta: [f64; N] = std::array::from_fn(|k| ba(0, er[k]) + ub[k * lf]);
    if let (Some(ngr), Some(ng)) = (ng_r.as_ref(), ng) {
        let d = ng_eta_contraction(&ng.jets, ngr, |v| ba(0, v))?;
        for k in 0..n_bsv {
            grad_eta[k] += d[k];
        }
    }
    // A's θ adjoint: at η̂, then through every node's predictions (their blocks' reverses).
    let mut theta_a: Vec<f64> = (0..n_th).map(|j| pb[j * lf]).collect();
    for (t, d) in theta_a.iter_mut().zip(&theta_nodes) {
        *t += d;
    }

    // ---- the mode solve ----------------------------------------------------------------
    let grad_l: Vec<f64> = (0..N).map(|k| kr.value(1 + k)).collect();
    let bvec = DVector::from_column_slice(&grad_eta);
    let mut at_mode = true;
    let w = match mode_solve(h_inner, &bvec, &grad_l, || {
        let mut g = inner_search_gradient::<N>(model, subject, lay, theta, eta, ctx, pv, None)?;
        if let Some(ng) = ng {
            for (gk, qk) in g.iter_mut().zip(&ng.jets.g) {
                *gk += qk;
            }
        }
        Some(g)
    }) {
        Some(w) => w,
        None if held => {
            at_mode = false;
            DVector::zeros(N)
        }
        None => return None,
    };

    // ---- reverse 2: −wᵀ∇_η l and the rows, from the kept forward -----------------------
    let mut coefs: Vec<Vec<f64>> = vec![std::iter::once(0.0).chain(w.iter().map(|v| -v)).collect()];
    let want_rows = matches!(&eta_dx, Some(o) if o.want_dx) && at_mode;
    if want_rows {
        coefs.extend(eta_dx_coefs::<N>(
            h_inner,
            active.min(N),
            n_bsv.min(active).min(N),
        ));
    }
    let seeds = coefs.len();
    let mut du_bar = vec![0.0; seeds * w_out];
    for (s, c) in coefs.iter().enumerate() {
        seed_preds(c, &mut du_bar[s * w_out..(s + 1) * w_out]);
    }
    let (mut ub2, mut pb2) = (vec![0.0; seeds * nu], vec![0.0; seeds * np]);
    if let Some((plan, _, fwd)) = mode_fwd.as_ref() {
        if !plan.reverse_from(fwd, full, seeds, &mut ub2, &mut pb2, &du_bar) {
            return None;
        }
    }
    let theta_of = |s: usize| -> Vec<f64> { (0..n_th).map(|j| pb2[s * np + j * lf]).collect() };
    let comb_adj = |c: &[f64], v: Leaf| -> f64 {
        c.iter()
            .enumerate()
            .filter(|(_, &k)| k != 0.0)
            .map(|(o, &k)| k * ba(o, v))
            .sum()
    };
    let psi_c: Vec<f64> = std::iter::once(1.0).chain(w.iter().map(|v| -v)).collect();
    let psi_of = |v: Leaf| comb_adj(&psi_c, v);
    let mut theta_adj = theta_a;
    for (t, d) in theta_adj.iter_mut().zip(theta_of(0)) {
        *t += d;
    }
    if let (Some(ngr), Some(_)) = (ng_r.as_ref(), ng) {
        let (a_q, a_g, a_h) = ng_adjoints(ngr, n_bsv, psi_of);
        let mut parts = vec![NgSeed {
            eta: &eta[..n_bsv],
            a_q,
            a_g: &a_g,
            a_h: &a_h,
        }];
        for ((_, _, b), &q) in grid.iter().zip(&node_qr) {
            parts.push(NgSeed {
                eta: &b[..n_bsv],
                a_q: psi_of(q),
                a_g: &[],
                a_h: &[],
            });
        }
        for (t, d) in theta_adj
            .iter_mut()
            .zip(ng_theta_adjoint(model, subject, theta, &parts)?)
        {
            *t += d;
        }
    }
    let out = packed_gradient(
        model,
        template,
        params,
        x.len(),
        &theta_adj,
        &psi_of,
        &pr,
        &om,
        so,
        pv,
    );
    if let Some(o) = eta_dx {
        o.eta_dx = None;
        o.hessian = (o.want_hessian && at_mode && !iov_frem_mismatch(model, subject))
            .then(|| h_inner.view((0, 0), (active, active)).into_owned())
            .filter(|h| h.iter().all(|v| v.is_finite()));
        let n_rows = seeds - 1;
        if want_rows && n_rows > 0 {
            let row_c = |r: usize| coefs[1 + r].as_slice();
            let mut tas: Vec<Vec<f64>> = (0..n_rows).map(|r| theta_of(1 + r)).collect();
            if let Some(ngr) = ng_r.as_ref() {
                let ng_seeds: Vec<_> = (0..n_rows)
                    .map(|r| ng_adjoints(ngr, n_bsv, |v| comb_adj(row_c(r), v)))
                    .collect();
                let d = ng_theta_adjoint_seeds(model, subject, theta, &eta[..n_bsv], &ng_seeds)?;
                for (ta, d) in tas.iter_mut().zip(d) {
                    for (t, dv) in ta.iter_mut().zip(d) {
                        *t += dv;
                    }
                }
            }
            let mut cols = vec![DVector::<f64>::zeros(n_rows); x.len()];
            for (r, ta) in tas.iter().enumerate() {
                let of = |v: Leaf| comb_adj(row_c(r), v);
                let row =
                    packed_gradient(model, template, params, x.len(), ta, &of, &pr, &om, so, pv);
                for (c, v) in cols.iter_mut().zip(row) {
                    c[r] = -v;
                }
            }
            let ok = cols.iter().all(|c| c.iter().all(|v| v.is_finite()));
            o.eta_dx = ok.then_some(cols);
        }
    }
    Some(out)
}
