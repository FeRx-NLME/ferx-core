//! Matrix products and solves as calls of a compiled plan (#2980).
//!
//! A recording inlines whatever its generic code computes. For the linear-ODE path that is the
//! degree-13 Padé approximant of a matrix exponential (`ode::linear`): six `n×n` products and a
//! solve with a matrix right-hand side, each written out lane by lane — kernels of tens of
//! thousands of straight-line instructions that stream through the instruction cache on every
//! call (on `transit_2cpt` they were 78% of an ordinary fit's cycles at a tenth of the rest's
//! instructions per cycle). As calls, the same products and solves run here: compact loops,
//! the plan's kernel only the glue around them.
//!
//! **Lanes.** A slot holds a layout's lanes (`sens::num::vm::Layout`): the value, the η-jet and
//! its second-order payload, then `tan` tangent blocks. The lane arithmetic of a product is
//! bilinear: lane `L` of `a·b` is `Σ κ_{L,p,q}·a_p·b_q` over a fixed table κ — the lowering's
//! own rule (`rhs_ir`'s `mul`: `g_k = a_k b + a b_k`, `h_t = a_h b + a b_h + a_k b_l + a_l b_k`;
//! a tangent block by the product rule). So a matrix product's lane `L` is `Σ κ_{L,p,q}·A_p·B_q`
//! and its reverse `Ā_p += κ·C̄_L·B_qᵀ`, `B̄_q += κ·A_pᵀ·C̄_L` — both bilinear again. A solve
//! `Q·X = P` holds lane by lane: `Σ κ_{L,p,q}·Q_p·X_q = P_L`, where `X_L` enters only through
//! `Q₀·X_L` and every other `X_q` has a lower lane index, so the lanes solve in order against
//! the one factorization of `Q₀`; its reverse is the transposed system in the opposite order.
//!
//! **Forward over reverse** (`Sweep::TangentReverse`): the base layout's reverse differentiated
//! along the tangent blocks — the same bilinear forms with the product rule over the blocks.
//!
//! The value lane is the value path's own arithmetic: the products accumulate in its order and
//! the solve calls its LU (`sens::linsolve`), so a plan's value agrees with the `f64` run bit
//! for bit.

use super::*;

/// What a matrix call computes, over `n×n` row-major matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatOp {
    /// `C = A·B`: inputs `A`, `B`.
    Mul,
    /// `X = Q⁻¹·P`: inputs `Q`, `P`.
    Solve,
}

/// A matrix call: its operation and size.
pub(crate) struct MatSpec {
    pub(super) op: MatOp,
    pub(super) n: usize,
}

/// The lane of base lane `i` in tangent block `t` (`0`: the base lane itself): a slot holds its
/// base lanes, then per base lane its `tan` tangents (`rhs_ir::tangent_packed`).
#[inline(always)]
fn tl(lb: usize, tan: usize, t: usize, i: usize) -> usize {
    if t == 0 {
        i
    } else {
        lb + i * tan + (t - 1)
    }
}

/// The bilinear lane table of `layout`: `(L, p, q, κ)` with lane `L` of a product
/// `Σ κ·a_p·b_q`.
pub(super) fn kappa(layout: Layout) -> Vec<(usize, usize, usize, f64)> {
    let base = kappa_base(layout);
    let lb = layout.base_lanes();
    let tan = layout.tan as usize;
    let mut out = base.clone();
    for t in 1..=tan {
        for &(l, p, q, c) in &base {
            out.push((tl(lb, tan, t, l), tl(lb, tan, t, p), q, c));
            out.push((tl(lb, tan, t, l), p, tl(lb, tan, t, q), c));
        }
    }
    out
}

/// [`kappa`] of the base layout (no tangent blocks).
fn kappa_base(layout: Layout) -> Vec<(usize, usize, usize, f64)> {
    let n = layout.n as usize;
    let mut out = vec![(0, 0, 0, 1.0)];
    for k in 0..n {
        out.push((1 + k, 1 + k, 0, 1.0));
        out.push((1 + k, 0, 1 + k, 1.0));
    }
    for k in 0..n {
        for (l, t) in super::ir::pairs(&layout, k) {
            let h = 1 + n + t;
            out.push((h, h, 0, 1.0));
            out.push((h, 0, h, 1.0));
            out.push((h, 1 + k, 1 + l, 1.0));
            out.push((h, 1 + l, 1 + k, 1.0));
        }
    }
    out
}

/// Lane `p` of entry `(i, j)` of a matrix stored entry by entry, `l` lanes each.
#[inline(always)]
fn at(n: usize, l: usize, i: usize, j: usize, p: usize) -> usize {
    (i * n + j) * l + p
}

/// `C[L] += κ·A[p]·B[q]` for every table entry: a product of lane-stacked matrices. The value
/// lane (`(0, 0, 0)`, first) accumulates in the value path's order (`ode::linear`'s
/// `accumulate`: `i`, then `k`, then `j`).
fn mul_lanes(
    n: usize,
    l: usize,
    kap: &[(usize, usize, usize, f64)],
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    for &(ll, p, q, k) in kap {
        for i in 0..n {
            for m in 0..n {
                let aik = a[at(n, l, i, m, p)];
                if aik == 0.0 {
                    continue;
                }
                let s = k * aik;
                for j in 0..n {
                    c[at(n, l, i, j, ll)] += if k == 1.0 {
                        aik * b[at(n, l, m, j, q)]
                    } else {
                        s * b[at(n, l, m, j, q)]
                    };
                }
            }
        }
    }
}

impl MatSpec {
    pub(super) fn n_in(&self) -> usize {
        2 * self.n * self.n
    }
    pub(super) fn n_out(&self) -> usize {
        self.n * self.n
    }

    /// The call over `layout`: `inputs` the two matrices' lanes, entry by entry; `out` the
    /// result's.
    pub(super) fn run(&self, layout: Layout, inputs: &[f64], out: &mut [f64]) -> bool {
        let (n, l) = (self.n, layout.lanes());
        let nn = n * n;
        let (a, b) = inputs.split_at(nn * l);
        out[..nn * l].fill(0.0);
        let kap = kappa(layout);
        match self.op {
            MatOp::Mul => mul_lanes(n, l, &kap, a, b, out),
            MatOp::Solve => {
                let Some(lu) = Lu::of(a, n, l, 0) else {
                    return false;
                };
                // Lane by lane, in order: `Q₀·X_L = P_L − Σ_{(p,q)≠(0,L)} κ·Q_p·X_q`.
                let mut rhs = vec![0.0; nn];
                for ll in 0..l {
                    for i in 0..n {
                        for j in 0..n {
                            rhs[i * n + j] = b[at(n, l, i, j, ll)];
                        }
                    }
                    for &(lt, p, q, k) in &kap {
                        if lt != ll || (p == 0 && q == ll) {
                            continue;
                        }
                        for i in 0..n {
                            for m in 0..n {
                                let qim = a[at(n, l, i, m, p)];
                                if qim == 0.0 {
                                    continue;
                                }
                                for j in 0..n {
                                    rhs[i * n + j] -= k * qim * out[at(n, l, m, j, q)];
                                }
                            }
                        }
                    }
                    let x = lu.solve_cols(&rhs);
                    for i in 0..n {
                        for j in 0..n {
                            out[at(n, l, i, j, ll)] = x[i * n + j];
                        }
                    }
                }
            }
        }
        out[..nn * l].iter().all(|v| v.is_finite())
    }

    /// The reverse: `out_adj` the result's adjoints per seed, into `in_adj` per seed the two
    /// inputs'. Under [`Sweep::Reverse`] the adjoint of every lane of `layout`; under
    /// [`Sweep::TangentReverse`] (one seed) the base layout's reverse with its tangents along
    /// `layout`'s blocks.
    pub(super) fn run_rev(
        &self,
        layout: Layout,
        sweep: Sweep,
        inputs: &[f64],
        seeds: usize,
        out_adj: &[f64],
        in_adj: &mut [f64],
    ) -> bool {
        let (n, l) = (self.n, layout.lanes());
        let nn = n * n;
        let mut fwd = vec![0.0; nn * l];
        if self.op == MatOp::Solve && !self.run(layout, inputs, &mut fwd) {
            return false;
        }
        let (a, b) = inputs.split_at(nn * l);
        // The table the reverse transposes: every lane's (`Reverse`), or the base layout's with
        // the product rule over the tangent blocks (`TangentReverse`: the adjoint's block `t`
        // is its tangent `t`, so `Ā_{t,p}` gains `κ·(C̄_{t,L}·B_{0,q} + C̄_{0,L}·B_{t,q})`,
        // exactly `kappa(layout)` read with the adjoint in the tangent's place).
        let kap = match sweep {
            Sweep::Reverse => kappa(layout),
            Sweep::TangentReverse => kappa(layout),
            // Forwards (a tangent-rows forward runs a matrix call at one tangent per seed).
            Sweep::Forward | Sweep::TangentRows | Sweep::TangentReverseRows => return false,
        };
        let lb = layout.base_lanes();
        for sd in 0..seeds {
            let cb = &out_adj[sd * nn * l..(sd + 1) * nn * l];
            let ia = &mut in_adj[sd * 2 * nn * l..(sd + 1) * 2 * nn * l];
            let (abar, bbar) = ia.split_at_mut(nn * l);
            match self.op {
                MatOp::Mul => match sweep {
                    Sweep::TangentReverse => mul_rev_tangent(n, l, lb, &kap, a, b, cb, abar, bbar),
                    _ => mul_rev(n, l, &kap, a, b, cb, abar, bbar),
                },
                MatOp::Solve => {
                    let ok = match sweep {
                        Sweep::TangentReverse => {
                            solve_rev_tangent(n, l, lb, layout, a, &fwd, cb, abar, bbar)
                        }
                        _ => solve_rev(n, l, &kap, a, &fwd, cb, abar, bbar),
                    };
                    if !ok {
                        return false;
                    }
                }
            }
        }
        in_adj.iter().all(|v| v.is_finite())
    }
}

/// `Ā_p += κ·C̄_L·B_qᵀ`, `B̄_q += κ·A_pᵀ·C̄_L` over `kap`: the reverse of [`mul_lanes`].
#[allow(clippy::too_many_arguments)]
fn mul_rev(
    n: usize,
    l: usize,
    kap: &[(usize, usize, usize, f64)],
    a: &[f64],
    b: &[f64],
    cb: &[f64],
    abar: &mut [f64],
    bbar: &mut [f64],
) {
    for &(ll, p, q, k) in kap {
        for i in 0..n {
            for j in 0..n {
                let c = k * cb[at(n, l, i, j, ll)];
                if c == 0.0 {
                    continue;
                }
                for m in 0..n {
                    abar[at(n, l, i, m, p)] += c * b[at(n, l, m, j, q)];
                    bbar[at(n, l, m, j, q)] += c * a[at(n, l, i, m, p)];
                }
            }
        }
    }
}

/// The forward over reverse of a product: the base layout's [`mul_rev`] with its tangents.
/// Block `t` of an adjoint is its tangent along direction `t`; `Ā_{0,p} += κ·C̄_{0,L}·B_{0,q}`
/// and `Ā_{t,p} += κ·(C̄_{t,L}·B_{0,q} + C̄_{0,L}·B_{t,q})`, `B̄` alike.
#[allow(clippy::too_many_arguments)]
fn mul_rev_tangent(
    n: usize,
    l: usize,
    lb: usize,
    kap: &[(usize, usize, usize, f64)],
    a: &[f64],
    b: &[f64],
    cb: &[f64],
    abar: &mut [f64],
    bbar: &mut [f64],
) {
    let tan = l / lb - 1;
    // The base table: the entries of `kap` inside block 0.
    let base: Vec<_> = kap.iter().copied().filter(|e| e.0 < lb).collect();
    for &(ll, p, q, k) in &base {
        for i in 0..n {
            for j in 0..n {
                for m in 0..n {
                    let (cv, av, bv) = (
                        cb[at(n, l, i, j, ll)],
                        a[at(n, l, i, m, p)],
                        b[at(n, l, m, j, q)],
                    );
                    abar[at(n, l, i, m, p)] += k * cv * bv;
                    bbar[at(n, l, m, j, q)] += k * av * cv;
                    for t in 1..=tan {
                        let (tll, tp, tq) =
                            (tl(lb, tan, t, ll), tl(lb, tan, t, p), tl(lb, tan, t, q));
                        let (ct, at_, bt) = (
                            cb[at(n, l, i, j, tll)],
                            a[at(n, l, i, m, tp)],
                            b[at(n, l, m, j, tq)],
                        );
                        abar[at(n, l, i, m, tp)] += k * (ct * bv + cv * bt);
                        bbar[at(n, l, m, j, tq)] += k * (at_ * cv + av * ct);
                    }
                }
            }
        }
    }
}

/// A dense LU (partial pivoting) of one lane of a lane-stacked matrix, through the value path's
/// own routines, for column solves and transposed ones.
struct Lu {
    n: usize,
    f: Vec<f64>,
    piv: Vec<f64>,
}

impl Lu {
    fn of(m: &[f64], n: usize, l: usize, lane: usize) -> Option<Self> {
        let mut f: Vec<f64> = (0..n * n).map(|e| m[e * l + lane]).collect();
        let mut piv = vec![0.0; crate::sens::linsolve::lu_piv_len::<f64>(n)];
        crate::sens::linsolve::lu_factor_in_place_g(&mut f, &mut piv, n).then_some(Lu { n, f, piv })
    }
    /// `Q⁻¹·R` for an `n×n` row-major `R`, column by column (the value path's order).
    fn solve_cols(&self, r: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n * n];
        let mut col = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                col[i] = r[i * n + j];
            }
            crate::sens::linsolve::lu_solve_in_place_g(&self.f, &self.piv, n, &mut col);
            for i in 0..n {
                x[i * n + j] = col[i];
            }
        }
        x
    }
}

/// `Q⁻ᵀ·R` from a factorization of `Qᵀ` (the transposed solves of the reverse).
fn transpose_lu(m: &[f64], n: usize, l: usize, lane: usize) -> Option<Lu> {
    let t: Vec<f64> = (0..n * n)
        .map(|e| {
            let (i, j) = (e / n, e % n);
            m[(j * n + i) * l + lane]
        })
        .collect();
    let mut piv = vec![0.0; crate::sens::linsolve::lu_piv_len::<f64>(n)];
    let mut f = t;
    crate::sens::linsolve::lu_factor_in_place_g(&mut f, &mut piv, n).then_some(Lu { n, f, piv })
}

/// The reverse of [`MatOp::Solve`]: with `X` known, the multipliers `Λ_L` of `Σ κ·Q_p·X_q = P_L`
/// solve the transposed system in decreasing lane order (`Q₀ᵀ·Λ_L = X̄_L − Σ κ·Q_pᵀ·Λ_{L'}`
/// over the equations `L' ≠ L` reading `X_L`), then `P̄_L = Λ_L` and `Q̄_p −= κ·Λ_{L'}·X_qᵀ`.
#[allow(clippy::too_many_arguments)]
fn solve_rev(
    n: usize,
    l: usize,
    kap: &[(usize, usize, usize, f64)],
    q: &[f64],
    x: &[f64],
    xb: &[f64],
    qbar: &mut [f64],
    pbar: &mut [f64],
) -> bool {
    let nn = n * n;
    let Some(lu) = transpose_lu(q, n, l, 0) else {
        return false;
    };
    let mut lam = vec![0.0; nn * l];
    let mut rhs = vec![0.0; nn];
    for ll in (0..l).rev() {
        for i in 0..n {
            for j in 0..n {
                rhs[i * n + j] = xb[at(n, l, i, j, ll)];
            }
        }
        for &(lt, p, qq, k) in kap {
            if qq != ll || (p == 0 && lt == ll) {
                continue;
            }
            // `X_ll` read by equation `lt` through `Q_p`: subtract `κ·Q_pᵀ·Λ_lt`.
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for m in 0..n {
                        s += q[at(n, l, m, i, p)] * lam[at(n, l, m, j, lt)];
                    }
                    rhs[i * n + j] -= k * s;
                }
            }
        }
        let y = lu.solve_cols(&rhs);
        for i in 0..n {
            for j in 0..n {
                lam[at(n, l, i, j, ll)] = y[i * n + j];
            }
        }
    }
    for e in 0..nn * l {
        pbar[e] += lam[e];
    }
    for &(lt, p, qq, k) in kap {
        for i in 0..n {
            for m in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += lam[at(n, l, i, j, lt)] * x[at(n, l, m, j, qq)];
                }
                qbar[at(n, l, i, m, p)] -= k * s;
            }
        }
    }
    true
}

/// The forward over reverse of a solve: [`solve_rev`] on the base layout with its tangents
/// along `layout`'s blocks. `Λ` solves block by block: block 0 the base reverse, block `t`
/// `Q₀ᵀ·Λ_t = X̄_t − (Q_{t,0})ᵀ·Λ₀ − Σ κ·(Q_{t,p}ᵀ·Λ_{0,L'} + Q_{0,p}ᵀ·Λ_{t,L'})` lane by lane;
/// then `P̄`, `Q̄` by the product rule.
#[allow(clippy::too_many_arguments)]
fn solve_rev_tangent(
    n: usize,
    l: usize,
    lb: usize,
    layout: Layout,
    q: &[f64],
    x: &[f64],
    xb: &[f64],
    qbar: &mut [f64],
    pbar: &mut [f64],
) -> bool {
    let nn = n * n;
    let tan = l / lb - 1;
    let base = kappa_base(layout);
    let Some(lu) = transpose_lu(q, n, l, 0) else {
        return false;
    };
    let mut lam = vec![0.0; nn * l];
    let mut rhs = vec![0.0; nn];
    // `Σ_m Q[m][i]·Λ[m][j]` of two lanes.
    let qt_lam = |qo: usize, lo: usize, lam: &[f64], i: usize, j: usize| -> f64 {
        (0..n)
            .map(|m| q[at(n, l, m, i, qo)] * lam[at(n, l, m, j, lo)])
            .sum()
    };
    let lt_ = |blk: usize, i: usize| tl(lb, tan, blk, i);
    for blk in 0..=tan {
        for ll in (0..lb).rev() {
            for i in 0..n {
                for j in 0..n {
                    rhs[i * n + j] = xb[at(n, l, i, j, lt_(blk, ll))];
                }
            }
            for &(lt, p, qq, k) in &base {
                if qq != ll {
                    continue;
                }
                let diag = p == 0 && lt == ll;
                for i in 0..n {
                    for j in 0..n {
                        let mut s = 0.0;
                        if blk == 0 {
                            if !diag {
                                s += qt_lam(p, lt, &lam, i, j);
                            }
                        } else {
                            // The tangent of `Q_pᵀ·Λ_lt`: `Q_{t,p}ᵀ·Λ_{0,lt} + Q_{0,p}ᵀ·Λ_{t,lt}`,
                            // the diagonal term's second half being the unknown itself.
                            s += qt_lam(lt_(blk, p), lt, &lam, i, j);
                            if !diag {
                                s += qt_lam(p, lt_(blk, lt), &lam, i, j);
                            }
                        }
                        rhs[i * n + j] -= k * s;
                    }
                }
            }
            let y = lu.solve_cols(&rhs);
            for i in 0..n {
                for j in 0..n {
                    lam[at(n, l, i, j, lt_(blk, ll))] = y[i * n + j];
                }
            }
        }
    }
    for e in 0..nn * l {
        pbar[e] += lam[e];
    }
    for &(lt, p, qq, k) in &base {
        for i in 0..n {
            for m in 0..n {
                let mut s0 = 0.0;
                for j in 0..n {
                    s0 += lam[at(n, l, i, j, lt)] * x[at(n, l, m, j, qq)];
                }
                qbar[at(n, l, i, m, p)] -= k * s0;
                for t in 1..=tan {
                    let mut st = 0.0;
                    for j in 0..n {
                        st += lam[at(n, l, i, j, lt_(t, lt))] * x[at(n, l, m, j, qq)]
                            + lam[at(n, l, i, j, lt)] * x[at(n, l, m, j, lt_(t, qq))];
                    }
                    qbar[at(n, l, i, m, lt_(t, p))] -= k * st;
                }
            }
        }
    }
    true
}

/// Record `op` of two `n×n` row-major matrices as one call (see [`MatSpec`]): the inputs' registers
/// as the call's arguments, the result as its outputs.
pub(crate) fn record_mat(op: MatOp, a: &[Tr], b: &[Tr], n: usize) -> Option<Vec<Tr>> {
    let spec = Arc::new(MatSpec { op, n });
    let inputs: Vec<f64> = a.iter().chain(b).map(|v| v.v).collect();
    let mut out = vec![0.0; n * n];
    if !spec.run(Layout::F64, &inputs, &mut out) {
        fail_here("a matrix call's record-time run failed");
        return None;
    }
    let id = rec(|r| {
        r.loops.push(CallKind::Mat(spec));
        (r.loops.len() - 1) as u32
    });
    for v in a.iter().chain(b) {
        let a = v.r as u32;
        rec(|r| r.code.push(Flat::LoopArg { a }));
    }
    rec(|r| r.code.push(Flat::Loop { id }));
    Some(
        out.iter()
            .enumerate()
            .map(|(k, &v)| {
                let r = node(None, |d| Flat::LoopOut { d, k: k as u32 });
                Tr::mk(v, r, true)
            })
            .collect(),
    )
}
