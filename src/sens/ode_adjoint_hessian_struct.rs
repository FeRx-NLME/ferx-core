//! The covariance Hessian's structured kernel (#2980).
//!
//! A pass's likelihood kernel is one recorded program over every observation; its forward over
//! reverse sweeps all of it once per direction. FOCE's and FOCEI's diagonal objectives have a
//! structure that sweep does not use: each row yields a few scalars `r_jk` from its own
//! prediction (FOCE also its population prediction) and the residual parameters, and the
//! objective reads the rows only through **aggregates** `A_k = Σ_j r_jk·m_k(G_j)` with
//! `m_k ∈ {1, G, GGᵀ}`:
//!
//! - FOCEI: `(ℓ, dl, α)` → `Σℓ`, `B = Σ dl·G`, `S = Σ α·GGᵀ`;
//!   `Ψ_w = ½[Σℓ + ηᵀΩ⁻¹η + log|Ω| + log|Ω⁻¹ + S|] − wᵀ(Ω⁻¹η + ½B)`;
//! - FOCE (Woodbury): `(iv·e², iv·e, iv, log V)` → `E`, `c = Σ iv·e·G`, `S = Σ iv·GGᵀ`, `LR`;
//!   with `A = I + LᵀSL` (`Ω = LLᵀ`), `Ψ_w = ½[E + 2cᵀη + ηᵀSη − (c+Sη)ᵀLA⁻¹Lᵀ(c+Sη) + LR +
//!   log|A|] − wᵀ(Ω⁻¹η − c)`;
//! - Laplace (one node, the exact anchor): `(ℓ, ℓ′, ℓ″)` → `Σℓ`, `B = Σ ℓ′·G`,
//!   `S = Σ ℓ″·GGᵀ`, `T = Σ ℓ′·Q` (`Q` the prediction's packed η-Hessian, a fourth monomial);
//!   `H = Ω⁻¹ + ½(S + T)`, `Ψ_w = ½[Σℓ + ηᵀΩ⁻¹η + log|Ω|] + ½log|H| − wᵀ(Ω⁻¹η + ½B)`; one
//!   node under FOCEI's anchor the same with `H̃ = Ω⁻¹ + Σ α·GGᵀ` from FOCEI's rows;
//! - AGQ (either anchor): the anchor's aggregates as above, and per node `N_d = Σℓ` over the
//!   rows at the node's predictions. Stage 1 `b_d = η + √2·L⁻ᵀz_d` (`LLᵀ` the factored
//!   anchor) and stage 2 `Ψ_w = ½log|H| − LSE_d[c_d − ½(prior(b_d) + N_d)] − wᵀ(Ω⁻¹η + ½B)` are
//!   two reduces over the aggregates; the node plans run between them, as they do between the
//!   recorded kernel's two stages.
//!
//! So the adjoint tangents the sweep would return are assembled from: the rows, once per
//! subject on a second-order dual over their local inputs (the same row functions production's
//! kernel calls); the **reduce** `R(η, x, A, w)` — a small recorded program over the
//! aggregates, no rows — swept along the pass's directions; and per direction the bilinear
//! algebra of the monomials over the rows. Exact: the same derivatives by the chain rule
//! through the aggregates; the oracles that check the sweep check this.

use super::basis::Input;
use super::*;
use crate::sens::dual2::Dual2;
use crate::sens::focei_kernel::{
    chol, foce_row, foce_woodbury_ok, focei_gauss_row, gauss_row_l, laplace_row, prior_of,
    structured_rows_supported,
};

/// A row's local dual: its prediction, its population prediction (FOCE), then up to `NL − 2`
/// σ packed coordinates.
const NL: usize = 5;

#[cfg(test)]
thread_local! {
    /// Test switch: every pass through the recorded likelihood kernel — the structured
    /// kernel's oracle.
    pub(crate) static STRUCT_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Passes this thread served by the structured kernel.
    pub(crate) static STRUCT_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn struct_off() -> bool {
    #[cfg(test)]
    return STRUCT_OFF.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// What a row scalar multiplies in its aggregate.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mono {
    One,
    G,
    Gg,
    /// The prediction's packed η-Hessian (Laplace's exact anchor).
    Q,
}

/// The objective's structured form.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Focei,
    Foce,
    Laplace,
    /// One node under FOCEI's Gauss–Newton anchor.
    LaplaceGn,
    /// Several nodes, FOCEI's anchor or the exact one.
    AgqGn,
    AgqExact,
}

impl Family {
    /// The aggregates: `(row scalar, monomial)`; a scalar may feed several.
    fn aggs(self) -> &'static [(usize, Mono)] {
        match self.rows() {
            Family::Foce => &[(0, Mono::One), (1, Mono::G), (2, Mono::Gg), (3, Mono::One)],
            Family::Laplace => &[(0, Mono::One), (1, Mono::G), (2, Mono::Gg), (1, Mono::Q)],
            _ => &[(0, Mono::One), (1, Mono::G), (2, Mono::Gg)],
        }
    }
    /// Row scalars.
    fn n_scalars(self) -> usize {
        match self {
            Family::Foce => 4,
            _ => 3,
        }
    }
    /// The family whose rows this one reads: FOCEI's `(ℓ, ∂ℓ/∂f, α)` under the Gauss–Newton
    /// anchor, the exact anchor's `(ℓ, ℓ′, ℓ″)` and `Q`.
    fn rows(self) -> Family {
        match self {
            Family::LaplaceGn | Family::AgqGn => Family::Focei,
            Family::AgqExact => Family::Laplace,
            f => f,
        }
    }
    /// Whether the objective reads quadrature nodes.
    fn nodes(self) -> bool {
        matches!(self, Family::AgqGn | Family::AgqExact)
    }
}

/// A subject's rows at the pass-independent point.
pub(super) struct Rows {
    family: Family,
    /// Local lane of the population prediction (FOCE), and the σ packed coordinates (lanes
    /// after it).
    pop_lane: Option<usize>,
    sig: Vec<usize>,
    /// Per row its scalars on the local dual.
    s: Vec<Vec<Dual2<NL>>>,
    g: Vec<Vec<f64>>,
    /// Per row its packed η-Hessian (Laplace; empty otherwise).
    q: Vec<Vec<f64>>,
    /// The aggregates, concatenated in [`Family::monos`] order.
    aggr: Vec<f64>,
}

/// A quadrature subject's rows at its nodes: per node and observation `ℓ` at the node's
/// prediction on the rows' local dual, and per node `N_d = Σℓ`.
pub(super) struct NodeRows {
    l: Vec<Vec<Dual2<NL>>>,
    agg: Vec<f64>,
}

/// A pass's anchor-row tangents, gathered once: per row its local lanes' kernel slots, and per
/// row and direction `u` (local), `Ġ`, `Q̇` and each scalar's `ṙ = ∇r·u`; and the aggregates'
/// tangents along each direction (`adot`, `kt × na`).
pub(super) struct AnchorTan {
    tg: Vec<f64>,
    slots: Vec<[usize; NL]>,
    stride: usize,
    o_g: usize,
    o_q: usize,
    o_r: usize,
    nla: usize,
    adot: Vec<f64>,
}

/// The anchor rows' back-propagation, waiting for every contribution to the aggregates'
/// adjoint: stage 2's, and (quadrature) stage 1's through the nodes' locations.
pub(super) struct Pending {
    at: AnchorTan,
    a_val: Vec<f64>,
    a_tan: Vec<f64>,
}

#[inline]
fn tri(k: usize, l: usize) -> usize {
    let (k, l) = if l <= k { (k, l) } else { (l, k) };
    k * (k + 1) / 2 + l
}

/// `d` on the rows' stored dual: its lanes first, the rest exact zeros.
fn widen<const L: usize>(d: Dual2<L>) -> Dual2<NL> {
    let mut o = Dual2::<NL>::constant(d.value);
    for i in 0..L {
        o.grad[i] = d.grad[i];
        for c in 0..L {
            o.hess[i][c] = d.hess[i][c];
        }
    }
    o
}

/// Each aggregate's offset in the aggregate vector, and the total width.
fn offsets(aggs: &[(usize, Mono)], m: usize) -> (Vec<usize>, usize) {
    let mut at = Vec::with_capacity(aggs.len());
    let mut w = 0;
    for (_, mono) in aggs {
        at.push(w);
        w += match mono {
            Mono::One => 1,
            Mono::G => m,
            Mono::Gg | Mono::Q => m * (m + 1) / 2,
        };
    }
    (at, w)
}

/// The symmetric `Q` with `Σ_{k≥l} P_kl·G_kG_l = GᵀQG` of a packed lower triangle `P`.
fn sym(p: &[f64], m: usize) -> Vec<Vec<f64>> {
    let mut q = vec![vec![0.0; m]; m];
    for k in 0..m {
        for l in 0..=k {
            let v = p[tri(k, l)];
            if k == l {
                q[k][k] = v;
            } else {
                q[k][l] = 0.5 * v;
                q[l][k] = 0.5 * v;
            }
        }
    }
    q
}

fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// An aggregate's adjoint prepared for the rows: a scalar, a vector, or (`Gg`) the symmetric
/// `Q` of its packed triangle, row-major — a view into a pass's [`Preps`].
#[derive(Clone, Copy)]
enum Prep<'a> {
    One(f64),
    G(&'a [f64]),
    Gg(&'a [f64]),
}

/// Every aggregate's prepared adjoint, for one or several adjoint vectors (a pass's value and
/// each direction's tangent), in one buffer: formed once per pass, not per row, and allocated
/// once.
struct Preps {
    buf: Vec<f64>,
    off: Vec<usize>,
    width: usize,
    monos: Vec<Mono>,
}

impl Preps {
    /// The prepared adjoints of `n` adjoint vectors, vector `v`'s aggregate `k` read from
    /// `src(v)[at[k]..]`.
    fn new<'s>(
        aggs: &[(usize, Mono)],
        at: &[usize],
        m: usize,
        n: usize,
        src: impl Fn(usize) -> &'s [f64],
    ) -> Self {
        let mut off = Vec::with_capacity(aggs.len());
        let mut width = 0;
        for &(_, mono) in aggs {
            off.push(width);
            width += match mono {
                Mono::One => 1,
                Mono::G => m,
                Mono::Q => m * (m + 1) / 2,
                Mono::Gg => m * m,
            };
        }
        let mut buf = vec![0.0; n * width];
        for v in 0..n {
            let a = src(v);
            for (k, &(_, mono)) in aggs.iter().enumerate() {
                let a = &a[at[k]..];
                let o = &mut buf[v * width + off[k]..];
                match mono {
                    Mono::One => o[0] = a[0],
                    Mono::G => o[..m].copy_from_slice(&a[..m]),
                    Mono::Q => o[..m * (m + 1) / 2].copy_from_slice(&a[..m * (m + 1) / 2]),
                    Mono::Gg => {
                        for kk in 0..m {
                            for l in 0..=kk {
                                let x = a[tri(kk, l)];
                                let x = if kk == l { x } else { 0.5 * x };
                                o[kk * m + l] = x;
                                o[l * m + kk] = x;
                            }
                        }
                    }
                }
            }
        }
        Preps {
            buf,
            off,
            width,
            monos: aggs.iter().map(|a| a.1).collect(),
        }
    }

    /// Vector `v`'s aggregate `k`.
    #[inline]
    fn get(&self, v: usize, k: usize) -> Prep<'_> {
        let o = &self.buf[v * self.width + self.off[k]..];
        match self.monos[k] {
            Mono::One => Prep::One(o[0]),
            Mono::G | Mono::Q => {
                let w = self.off.get(k + 1).copied().unwrap_or(self.width) - self.off[k];
                Prep::G(&o[..w])
            }
            Mono::Gg => {
                let w = self.off.get(k + 1).copied().unwrap_or(self.width) - self.off[k];
                Prep::Gg(&o[..w])
            }
        }
    }
}

/// `y = Q·v` into a reused buffer (`Q` row-major, `y.len()` square).
#[inline]
fn mv_into(q: &[f64], v: &[f64], y: &mut [f64]) {
    let m = y.len();
    for (k, yk) in y.iter_mut().enumerate() {
        let r = &q[k * m..(k + 1) * m];
        let mut acc = 0.0;
        for l in 0..m {
            acc += r[l] * v[l];
        }
        *yk = acc;
    }
}

/// A row's scalar adjoint `⟨a, m(G)⟩` and its `G`-adjoint `r·∂⟨a, m(G)⟩/∂G` into `dg`, given
/// the aggregate's prepared adjoint `a`; `qg` is scratch of width `m`.
fn back(a: Prep<'_>, g: &[f64], r: f64, dg: &mut [f64], qg: &mut [f64]) -> f64 {
    match a {
        Prep::One(v) => v,
        Prep::G(av) => {
            for (d, x) in dg.iter_mut().zip(av) {
                *d += r * x;
            }
            dotv(av, g)
        }
        Prep::Gg(q) => {
            mv_into(q, g, &mut qg[..g.len()]);
            for (d, x) in dg.iter_mut().zip(qg.iter()) {
                *d += 2.0 * r * x;
            }
            dotv(g, qg)
        }
    }
}

/// [`back`]'s tangent along a direction moving the aggregate adjoint by `ad`, the scalar by
/// `rd` and `G` by `gd`: the scalar adjoint's tangent, and the `G`-adjoint's into `dgd`. `qg`
/// is `P·G` as [`back`] left it (a `Gg` aggregate; direction-free), `sc` scratch of width `2m`.
#[allow(clippy::too_many_arguments)]
fn back_t(
    a: Prep<'_>,
    ad: Prep<'_>,
    g: &[f64],
    gd: &[f64],
    r: f64,
    rd: f64,
    qg: &[f64],
    dgd: &mut [f64],
    sc: &mut [f64],
) -> f64 {
    match (a, ad) {
        (Prep::One(_), Prep::One(v)) => v,
        (Prep::G(av), Prep::G(adv)) => {
            for ((d, x), y) in dgd.iter_mut().zip(av).zip(adv) {
                *d += rd * x + r * y;
            }
            dotv(adv, g) + dotv(av, gd)
        }
        (Prep::Gg(q), Prep::Gg(qd)) => {
            let m = g.len();
            let (qdg, qgd) = sc.split_at_mut(m);
            mv_into(qd, g, qdg);
            mv_into(q, gd, &mut qgd[..m]);
            for k in 0..m {
                dgd[k] += 2.0 * (rd * qg[k] + r * qdg[k] + r * qgd[k]);
            }
            dotv(g, qdg) + 2.0 * dotv(gd, qg)
        }
        _ => unreachable!("an aggregate's value and tangent have one monomial"),
    }
}

/// A row's contribution to an aggregate's tangent: `ṙ·m(G) + r·ṁ(G; Ġ)` into `out`.
fn aggr_t(mono: Mono, g: &[f64], gd: &[f64], r: f64, rd: f64, m: usize, out: &mut [f64]) {
    match mono {
        Mono::One => out[0] += rd,
        // `G`, or `Q` (a vector of its own length).
        Mono::G | Mono::Q => {
            for k in 0..g.len() {
                out[k] += rd * g[k] + r * gd[k];
            }
        }
        Mono::Gg => {
            for k in 0..m {
                for l in 0..=k {
                    out[tri(k, l)] += rd * g[k] * g[l] + r * (gd[k] * g[l] + g[k] * gd[l]);
                }
            }
        }
    }
}

impl CovPlan<'_> {
    /// The structured family of this subject's passes, if any: FOCEI or FOCE on the diagonal
    /// path with rows of the supported kind and at most `NL − 2` σ, outside the oracles'
    /// switches. (FOCE's Woodbury gate is checked on the rows, [`Self::rows`].)
    fn family(&self) -> Option<Family> {
        let fam = match self.objective {
            Objective::Focei => Family::Focei,
            Objective::Foce => Family::Foce,
            Objective::Agq {
                n_agq: 1,
                exact: true,
            } => Family::Laplace,
            Objective::Agq {
                n_agq: 1,
                exact: false,
            } => Family::LaplaceGn,
            Objective::Agq { exact: true, .. } => Family::AgqExact,
            Objective::Agq { exact: false, .. } => Family::AgqGn,
        };
        (!struct_off()
            && !reverse_per_direction()
            && self.nn_z.is_none()
            && self.sc.params.sigma.values.len() + 2 <= NL
            && structured_rows_supported(self.ctx))
        .then_some(fam)
    }

    /// Whether this subject's passes take the structured kernel.
    pub(super) fn structured(&self) -> bool {
        self.family().is_some()
    }

    /// The rows at this point (it does not move between passes): formed once, from the
    /// predictions `f(j)`, the population predictions `fp(j)` (FOCE under an `f`-dependent
    /// variance), the η-gradients `g(j, k)` and the packed coordinates `x(i)`. `None` where the
    /// structured form does not hold here (a row off its domain, FOCE off its Woodbury gate).
    fn rows(
        &self,
        f: &dyn Fn(usize) -> f64,
        fp: &dyn Fn(usize) -> Option<f64>,
        gv: &dyn Fn(usize, usize) -> f64,
        qv: &dyn Fn(usize, usize) -> f64,
        xv: &dyn Fn(usize) -> f64,
    ) -> Option<std::cell::Ref<'_, Rows>> {
        if self.st_rows.borrow().is_none() {
            let family = self.family()?;
            let pn = self.pn();
            let n_obs = self.subject.observations.len();
            let pop_lane = (family == Family::Foce).then_some(1);
            let s0 = packed_segments(self.template).sigma_start();
            let sig: Vec<usize> = (0..self.sc.params.sigma.values.len())
                .map(|k| s0 + k)
                .collect();
            let lane0 = 1 + usize::from(pop_lane.is_some());
            let (rs, l_chol) = match lane0 + sig.len() {
                0..=2 => self.eval_rows::<2>(family, n_obs, f, fp, xv, &sig, lane0),
                3 => self.eval_rows::<3>(family, n_obs, f, fp, xv, &sig, lane0),
                4 => self.eval_rows::<4>(family, n_obs, f, fp, xv, &sig, lane0),
                _ => self.eval_rows::<NL>(family, n_obs, f, fp, xv, &sig, lane0),
            }?;
            let aggs = family.aggs();
            let (at, na) = offsets(aggs, pn);
            let nq = if family.rows() == Family::Laplace {
                pn * (pn + 1) / 2
            } else {
                0
            };
            let mut s = Vec::with_capacity(n_obs);
            let mut g = Vec::with_capacity(n_obs);
            let mut qs = Vec::with_capacity(n_obs);
            let mut aggr = vec![0.0; na];
            for (j, r) in rs.into_iter().enumerate() {
                let gj: Vec<f64> = (0..pn).map(|k| gv(j, k)).collect();
                let qj: Vec<f64> = (0..nq).map(|t| qv(j, t)).collect();
                for (k, &(si, mono)) in aggs.iter().enumerate() {
                    let v = if mono == Mono::Q { &qj } else { &gj };
                    let zero = vec![0.0; v.len()];
                    aggr_t(mono, v, &zero, 0.0, r[si].value, pn, &mut aggr[at[k]..]);
                }
                s.push(r);
                g.push(gj);
                qs.push(qj);
            }
            if family == Family::Foce {
                // Production's gate: `1 + tr(LᵀSL)` (`S` the third aggregate).
                let l = &l_chol;
                let sq = sym(&aggr[at[2]..at[2] + pn * (pn + 1) / 2], pn);
                let mut tr = 0.0;
                for c in 0..pn {
                    for a in 0..pn {
                        for b in 0..pn {
                            tr += l[a][c] * sq[a][b] * l[b][c];
                        }
                    }
                }
                if !foce_woodbury_ok(tr) {
                    return None;
                }
            }
            *self.st_rows.borrow_mut() = Some(Rows {
                family,
                pop_lane,
                sig,
                s,
                g,
                q: qs,
                aggr,
            });
        }
        let rows = self.st_rows.borrow();
        std::cell::Ref::filter_map(rows, Option::as_ref).ok()
    }

    /// The population parameters on `Dual2<L>`: σ's packed coordinates `sig` on lanes
    /// `lane0 ..`, everything else constant.
    fn local_params<const L: usize>(
        &self,
        xv: &dyn Fn(usize) -> f64,
        sig: &[usize],
        lane0: usize,
    ) -> Option<crate::sens::focei_kernel::ParT<Dual2<L>>> {
        let xo = |i: usize| -> Dual2<L> {
            let v = xv(i);
            match sig.iter().position(|&s| s == i) {
                Some(q) => Dual2::var(v, lane0 + q),
                None => Dual2::constant(v),
            }
        };
        let theta: Vec<Dual2<L>> = unpack_theta(self.template, xo);
        population_params(
            self.sc.model,
            self.subject,
            self.template,
            self.sc.params,
            self.so,
            &theta,
            xo,
        )
    }

    /// The anchor rows' scalars on `Dual2<L>`, `L` the live local lanes (a dense second-order
    /// dual costs `L²` per operation), widened to the stored [`NL`]; and Ω's factor's values
    /// (FOCE's Woodbury gate).
    #[allow(clippy::too_many_arguments)]
    fn eval_rows<const L: usize>(
        &self,
        family: Family,
        n_obs: usize,
        f: &dyn Fn(usize) -> f64,
        fp: &dyn Fn(usize) -> Option<f64>,
        xv: &dyn Fn(usize) -> f64,
        sig: &[usize],
        lane0: usize,
    ) -> Option<(Vec<Vec<Dual2<NL>>>, Vec<Vec<f64>>)> {
        let p = self.local_params::<L>(xv, sig, lane0)?;
        let mut out = Vec::with_capacity(n_obs);
        for j in 0..n_obs {
            let fd = Dual2::<L>::var(f(j), 0);
            let r: Vec<Dual2<L>> = match family.rows() {
                Family::Focei => {
                    let (l, dl, al) = focei_gauss_row(self.ctx, j, fd, &p)?;
                    vec![l, dl, al]
                }
                Family::Foce => {
                    // `R⁰` at the population prediction where the variance reads `f`, else at
                    // anything (it does not).
                    let r_eval = match fp(j) {
                        Some(v) => Dual2::var(v, 1),
                        None => Dual2::constant(f(j)),
                    };
                    foce_row(self.ctx, j, fd, r_eval, &p)?.to_vec()
                }
                _ => laplace_row(self.ctx, j, fd, &p)?.to_vec(),
            };
            out.push(r.into_iter().map(widen).collect());
        }
        let pn = self.pn();
        let l = if family == Family::Foce {
            let lc = p.omega_chol(pn);
            (0..pn)
                .map(|a| (0..pn).map(|c| lc[a][c].value).collect())
                .collect()
        } else {
            Vec::new()
        };
        Some((out, l))
    }

    /// The quadrature node rows' `ℓ` on `Dual2<L>` (prediction, then σ), widened.
    fn eval_node_rows<const L: usize>(
        &self,
        fb: &dyn Fn(usize, usize) -> f64,
        xv: &dyn Fn(usize) -> f64,
        n_nodes: usize,
        n_obs: usize,
        s0: usize,
        n_sig: usize,
    ) -> Option<Vec<Vec<Dual2<NL>>>> {
        let sig: Vec<usize> = (s0..s0 + n_sig).collect();
        let p = self.local_params::<L>(xv, &sig, 1)?;
        (0..n_nodes)
            .map(|d| {
                (0..n_obs)
                    .map(|j| gauss_row_l(self.ctx, j, Dual2::<L>::var(fb(d, j), 0), &p).map(widen))
                    .collect()
            })
            .collect()
    }

    /// The rows at the quadrature's nodes (formed once, as [`Self::rows`]): `fb(d, j)` node
    /// `d`'s prediction of observation `j`, `xv` the packed coordinates.
    fn node_rows(
        &self,
        fb: &dyn Fn(usize, usize) -> f64,
        xv: &dyn Fn(usize) -> f64,
        n_nodes: usize,
    ) -> Option<std::cell::Ref<'_, NodeRows>> {
        if self.st_node_rows.borrow().is_none() {
            let n_obs = self.subject.observations.len();
            let s0 = packed_segments(self.template).sigma_start();
            let n_sig = self.sc.params.sigma.values.len();
            let l = match 1 + n_sig {
                0..=2 => self.eval_node_rows::<2>(fb, xv, n_nodes, n_obs, s0, n_sig),
                3 => self.eval_node_rows::<3>(fb, xv, n_nodes, n_obs, s0, n_sig),
                4 => self.eval_node_rows::<4>(fb, xv, n_nodes, n_obs, s0, n_sig),
                _ => self.eval_node_rows::<NL>(fb, xv, n_nodes, n_obs, s0, n_sig),
            }?;
            let agg = l.iter().map(|r| r.iter().map(|x| x.value).sum()).collect();
            *self.st_node_rows.borrow_mut() = Some(NodeRows { l, agg });
        }
        let rows = self.st_node_rows.borrow();
        std::cell::Ref::filter_map(rows, Option::as_ref).ok()
    }

    /// [`Self::rows`] from a pass's kernel inputs `inputs` (value, then `kt` tangents per slot).
    fn rows_at(&self, inputs: &[f64], sl: &Slots, kt: usize) -> Option<std::cell::Ref<'_, Rows>> {
        let (pn, lk) = (self.pn(), 1 + kt);
        let val = |s: usize| inputs[s * lk];
        self.rows(
            &|j| val(sl.f(j, 0)),
            &|j| sl.has_pop.then(|| val(sl.pop + j)),
            &|j, k| val(sl.f(j, 1 + k)),
            &|j, t| val(sl.f(j, 1 + pn + t)),
            &|i| val(sl.x + i),
        )
    }

    /// Whether the structured form holds at a pass's kernel inputs: the anchor rows, and a
    /// quadrature subject's node rows, form here ([`Self::struct_adjoint`] reads both). Where
    /// they do not, the pass takes the recorded kernel instead.
    pub(super) fn struct_rows_hold(&self, inputs: &[f64], sl: &Slots, kt: usize) -> bool {
        let Some(nodes) = self.rows_at(inputs, sl, kt).map(|r| r.family.nodes()) else {
            return false;
        };
        if !nodes || sl.n_nodes == 0 {
            return true;
        }
        let val = |s: usize| inputs[s * (1 + kt)];
        self.node_rows(
            &|d, j| val(sl.nf + d * sl.n_rows + j),
            &|i| val(sl.x + i),
            sl.n_nodes,
        )
        .is_some()
    }

    /// The anchor rows' tangents along a pass's `kt` directions, read from the kernel inputs
    /// `inputs` (value, then `kt` tangents per slot), and the aggregates' tangents.
    fn anchor_tan(&self, rows: &Rows, inputs: &[f64], sl: &Slots, kt: usize) -> AnchorTan {
        let lk = 1 + kt;
        let pn = self.pn();
        let aggs = rows.family.aggs();
        let (at, na) = offsets(aggs, pn);
        let nq = rows.q.first().map_or(0, Vec::len);
        let n_obs = sl.n_obs;
        let kk = rows.family.n_scalars();
        let lane0 = 1 + usize::from(rows.pop_lane.is_some());
        let nla = lane0 + rows.sig.len();
        let (o_g, o_q, o_r) = (NL, NL + pn, NL + pn + nq);
        let stride = o_r + kk;
        let mut tg = vec![0.0; n_obs * kt * stride];
        let mut slots = vec![[usize::MAX; NL]; n_obs];
        for j in 0..n_obs {
            let ls = &mut slots[j];
            ls[0] = sl.f(j, 0);
            if rows.pop_lane.is_some() && sl.has_pop {
                ls[1] = sl.pop + j;
            }
            for (q, &i) in rows.sig.iter().enumerate() {
                ls[lane0 + q] = sl.x + i;
            }
            let mut gather = |s: usize, at: usize| {
                let src = &inputs[s * lk + 1..(s + 1) * lk];
                for (a, v) in src.iter().enumerate() {
                    tg[(j * kt + a) * stride + at] = *v;
                }
            };
            for (lane, &s) in ls.iter().enumerate() {
                if s != usize::MAX {
                    gather(s, lane);
                }
            }
            for k in 0..pn + nq {
                gather(sl.f(j, 1 + k), o_g + k);
            }
            for a in 0..kt {
                let t = &mut tg[(j * kt + a) * stride..(j * kt + a + 1) * stride];
                for si in 0..kk {
                    let d = &rows.s[j][si].grad;
                    t[o_r + si] = (0..nla).map(|q| d[q] * t[q]).sum();
                }
            }
        }
        let mut adot = vec![0.0; kt * na];
        for j in 0..n_obs {
            let (g, q) = (&rows.g[j][..], &rows.q[j][..]);
            for a in 0..kt {
                let t = &tg[(j * kt + a) * stride..(j * kt + a + 1) * stride];
                let ad = &mut adot[a * na..(a + 1) * na];
                for (k, &(si, mono)) in aggs.iter().enumerate() {
                    let (v, vd) = if mono == Mono::Q {
                        (q, &t[o_q..o_r])
                    } else {
                        (g, &t[o_g..o_q])
                    };
                    aggr_t(
                        mono,
                        v,
                        vd,
                        rows.s[j][si].value,
                        t[o_r + si],
                        pn,
                        &mut ad[at[k]..],
                    );
                }
            }
        }
        AnchorTan {
            tg,
            slots,
            stride,
            o_g,
            o_q,
            o_r,
            nla,
            adot,
        }
    }

    /// Stage 1's input `(η, x, A)` along `kt` tangents: η and x from the kernel inputs, the
    /// anchor's aggregates and their tangents.
    fn stage1_in(
        &self,
        rows: &Rows,
        at: &AnchorTan,
        inputs: &[f64],
        sl: &Slots,
        kt: usize,
    ) -> Vec<f64> {
        let (pn, nx, lk) = (self.pn(), self.nx, 1 + kt);
        let na = rows.aggr.len();
        let mut rin = vec![0.0; (pn + nx + na) * lk];
        for k in 0..pn {
            rin[k * lk..(k + 1) * lk]
                .copy_from_slice(&inputs[(sl.eta + k) * lk..(sl.eta + k + 1) * lk]);
        }
        for i in 0..nx {
            let o = (pn + i) * lk;
            rin[o..o + lk].copy_from_slice(&inputs[(sl.x + i) * lk..(sl.x + i + 1) * lk]);
        }
        for t in 0..na {
            let o = (pn + nx + t) * lk;
            rin[o] = rows.aggr[t];
            for a in 0..kt {
                rin[o + 1 + a] = at.adot[a * na + t];
            }
        }
        rin
    }

    /// A structured quadrature subject's node locations along a pass's `kt` directions, laid
    /// out as the recorded stage 1's outputs (`(d·N + k)·(1 + kt)`: value, then tangents), with
    /// the anchor rows built on the way. `None`: not a structured quadrature subject here.
    pub(super) fn struct_nodes(&self, inputs: &[f64], sl: &Slots, kt: usize) -> Option<Vec<f64>> {
        let family = self.family().filter(|f| f.nodes())?;
        let pn = self.pn();
        let lk = 1 + kt;
        let rows = self.rows_at(inputs, sl, kt)?;
        let at = self.anchor_tan(&rows, inputs, sl, kt);
        let rin = self.stage1_in(&rows, &at, inputs, sl, kt);
        let nb = sl.n_nodes * pn;
        let n_in = rin.len() / lk;
        if let Some(out) = self.nodes_rows(family, &rin, kt, nb) {
            return Some(out);
        }
        let mut out = vec![0.0; nb * lk];
        let mut one = vec![0.0; n_in * 2];
        for a in 0..kt.max(1) {
            for i in 0..n_in {
                one[2 * i] = rin[i * lk];
                one[2 * i + 1] = if kt > 0 { rin[i * lk + 1 + a] } else { 0.0 };
            }
            let kr = self.reduce_run(family, 1, &one, 1, nb)?;
            for o in 0..nb {
                if a == 0 {
                    out[o * lk] = kr.out[o * 2];
                }
                if kt > 0 {
                    out[o * lk + 1 + a] = kr.out[o * 2 + 1];
                }
            }
        }
        Some(out)
    }

    /// The stage-1 reduce's node locations through the rows kernels (#2980): its values once,
    /// each direction a seed of their tangents (`rin` and the result laid out as
    /// [`Self::struct_nodes`]). `None` where the plan has no rows kernels or its guard does not
    /// hold; the caller then runs a direction at a time.
    fn nodes_rows(&self, family: Family, rin: &[f64], kt: usize, nb: usize) -> Option<Vec<f64>> {
        let lk = 1 + kt;
        let n_in = rin.len() / lk;
        if kt == 0 {
            return None;
        }
        let vals: Vec<f64> = (0..n_in).map(|i| rin[i * lk]).collect();
        let kr = self.reduce_run(family, 1, &vals, 0, nb)?;
        let plan = kr.plan();
        if !plan.rows_serve(Layout::F64) {
            return None;
        }
        let mut tu = vec![0.0; kt * n_in];
        for a in 0..kt {
            for i in 0..n_in {
                tu[a * n_in + i] = rin[i * lk + 1 + a];
            }
        }
        let rf = plan.forward_rows(Layout::F64, &[], &vals, &tu, kt)?;
        if !plan.rows_guard(&rf) {
            return None;
        }
        let mut out = vec![0.0; nb * lk];
        for o in 0..nb {
            for a in 0..kt {
                let (v, t) = plan.rows_out(&rf, o, a);
                out[o * lk] = v[0];
                out[o * lk + 1 + a] = t[0];
            }
        }
        Some(out)
    }

    /// The reduce's forward over reverse seeded on its output, along `kt` tangents (`rin`: each
    /// input's value, then its tangents), in the same layout: per input its value adjoint, then
    /// its adjoint's tangent along each. One tangent per run, for the reason the bases take one
    /// axis per run: the kernel's code grows with its width, its calls with their count.
    ///
    /// `stage` 2 is the objective (one output, seeded `1`); stage 1 the nodes' locations, seeded
    /// with `seed` (per output its value lane, then its tangent along each direction).
    fn reduce_for(
        &self,
        family: Family,
        stage: u8,
        rin: &[f64],
        kt: usize,
        seed: &[f64],
    ) -> Option<Vec<f64>> {
        let lk = 1 + kt;
        let n_in = rin.len() / lk;
        let n_out = seed.len() / lk;
        if let Some(out) = self.reduce_rows(family, stage, rin, kt, seed) {
            return Some(out);
        }
        let mut out = vec![0.0; n_in * lk];
        let mut one = vec![0.0; n_in * 2];
        let mut s1 = vec![0.0; n_out * 2];
        for a in 0..kt {
            for i in 0..n_in {
                one[2 * i] = rin[i * lk];
                one[2 * i + 1] = rin[i * lk + 1 + a];
            }
            for o in 0..n_out {
                s1[2 * o] = seed[o * lk];
                s1[2 * o + 1] = seed[o * lk + 1 + a];
            }
            let kr = self.reduce_run(family, stage, &one, 1, n_out)?;
            let radj = kr.reverse_tangent(&s1)?;
            for i in 0..n_in {
                if a == 0 {
                    out[i * lk] = radj[2 * i];
                }
                out[i * lk + 1 + a] = radj[2 * i + 1];
            }
        }
        Some(out)
    }

    /// [`Self::reduce_for`] through the rows kernels (#2980): the reduce's values and adjoints
    /// once, each direction a seed of its tangents. `None` where its plan has no rows kernels or
    /// its guard does not hold at these values; the caller then runs a direction at a time.
    fn reduce_rows(
        &self,
        family: Family,
        stage: u8,
        rin: &[f64],
        kt: usize,
        seed: &[f64],
    ) -> Option<Vec<f64>> {
        let lk = 1 + kt;
        let (n_in, n_out) = (rin.len() / lk, seed.len() / lk);
        if kt == 0 {
            return None;
        }
        let vals: Vec<f64> = (0..n_in).map(|i| rin[i * lk]).collect();
        let kr = self.reduce_run(family, stage, &vals, 0, n_out)?;
        let plan = kr.plan();
        if !plan.rows_serve(Layout::F64) {
            return None;
        }
        let mut tu = vec![0.0; kt * n_in];
        for a in 0..kt {
            for i in 0..n_in {
                tu[a * n_in + i] = rin[i * lk + 1 + a];
            }
        }
        let rf = plan.forward_rows(Layout::F64, &[], &vals, &tu, kt)?;
        if !plan.rows_guard(&rf) {
            return None;
        }
        let du: Vec<f64> = (0..n_out).map(|o| seed[o * lk]).collect();
        let mut dut = vec![0.0; kt * n_out];
        for a in 0..kt {
            for o in 0..n_out {
                dut[a * n_out + o] = seed[o * lk + 1 + a];
            }
        }
        let (mut pb, mut pt) = (vec![0.0; n_in], vec![0.0; kt * n_in]);
        if !plan.reverse_rows(&rf, &du, Some(&dut), &mut [], &mut pb, &mut [], &mut pt) {
            return None;
        }
        let mut out = vec![0.0; n_in * lk];
        for i in 0..n_in {
            out[i * lk] = pb[i];
            for a in 0..kt {
                out[i * lk + 1 + a] = pt[a * n_in + i];
            }
        }
        Some(out)
    }

    /// The reduce `R(η, x, A, w)` along `kt` tangents (`rin`: each input's value, then its
    /// tangents): a small recorded program, no rows.
    fn reduce_run(
        &self,
        family: Family,
        stage: u8,
        rin: &[f64],
        kt: usize,
        n_out: usize,
    ) -> Option<super::kernel::KernelRun> {
        let layout = Layout {
            tan: kt as u8,
            ..Layout::F64
        };
        let si = usize::from(stage == 1);
        // The subject's reduce plan, looked up once (its key hashes the subject and the
        // parameters); every later run goes straight to it unless a guard sends it back.
        let mine = self.st_reduce.borrow()[si].clone();
        if let Some(kr) =
            mine.and_then(|p| super::kernel::KernelRun::of_plan(&p, layout, rin, n_out))
        {
            return Some(kr);
        }
        let pn = self.pn();
        let nx = self.nx;
        let (at, na) = offsets(family.aggs(), pn);
        let grid = match self.objective {
            Objective::Agq { n_agq, .. } if family.nodes() => agq_grid(n_agq, pn),
            _ => Vec::new(),
        };
        let nn = grid.len();
        let (model, subject, template, sc, so, ctx) = (
            self.sc.model,
            self.subject,
            self.template,
            self.sc,
            self.so,
            self.ctx,
        );
        let record = |lv: &[kj::Tr]| -> Option<Vec<kj::Tr>> {
            type T = kj::Tr;
            let c = |v: f64| <T as PkNum>::from_f64(v);
            let eta: Vec<T> = lv[..pn].to_vec();
            let xo = |i: usize| lv[pn + i];
            let theta: Vec<T> = unpack_theta(template, xo);
            let p = population_params(model, subject, template, sc.params, so, &theta, xo)?;
            let ag = |i: usize| lv[pn + nx + i];
            let node_agg = |d: usize| lv[pn + nx + na + d];
            let w = |k: usize| lv[pn + nx + na + nn + k];
            let oi = p.omega_inv(pn).clone();
            let mat = |off: usize| -> Vec<Vec<T>> {
                (0..pn)
                    .map(|k| (0..pn).map(|l| ag(off + tri(k, l))).collect())
                    .collect()
            };
            let oieta: Vec<T> = (0..pn)
                .map(|k| (0..pn).fold(c(0.0), |s, l| s + oi[k][l] * eta[l]))
                .collect();
            match family {
                Family::Focei => {
                    let s = mat(at[2]);
                    let a: Vec<Vec<T>> = (0..pn)
                        .map(|k| (0..pn).map(|l| oi[k][l] + s[k][l]).collect())
                        .collect();
                    let ld = chol_logdet(&chol(&a)?);
                    let mut wg = c(0.0);
                    for k in 0..pn {
                        wg = wg + w(k) * (oieta[k] + c(0.5) * ag(at[1] + k));
                    }
                    Some(vec![
                        c(0.5) * (ag(at[0]) + prior_of(ctx, &eta, &p) + ld) - wg,
                    ])
                }
                Family::Foce => {
                    let s = mat(at[2]);
                    let cv: Vec<T> = (0..pn).map(|k| ag(at[1] + k)).collect();
                    let l = p.omega_chol(pn).clone();
                    // A = I + LᵀSL; v = Lᵀ(c + Sη).
                    let sl: Vec<Vec<T>> = (0..pn)
                        .map(|a| {
                            (0..pn)
                                .map(|j| (0..pn).fold(c(0.0), |acc, b| acc + s[a][b] * l[b][j]))
                                .collect()
                        })
                        .collect();
                    let a: Vec<Vec<T>> = (0..pn)
                        .map(|i| {
                            (0..pn)
                                .map(|j| {
                                    let v = (0..pn).fold(c(0.0), |acc, k| acc + l[k][i] * sl[k][j]);
                                    if i == j {
                                        v + c(1.0)
                                    } else {
                                        v
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    let cs: Vec<T> = (0..pn)
                        .map(|k| cv[k] + (0..pn).fold(c(0.0), |acc, m| acc + s[k][m] * eta[m]))
                        .collect();
                    let v: Vec<T> = (0..pn)
                        .map(|i| (0..pn).fold(c(0.0), |acc, k| acc + l[k][i] * cs[k]))
                        .collect();
                    let la = chol(&a)?;
                    let z = crate::sens::focei_kernel::chol_solve(&la, &v);
                    let vz = (0..pn).fold(c(0.0), |acc, k| acc + v[k] * z[k]);
                    let ce = (0..pn).fold(c(0.0), |acc, k| acc + cv[k] * eta[k]);
                    let ese = (0..pn).fold(c(0.0), |acc, k| {
                        acc + (0..pn).fold(c(0.0), |a2, m| a2 + eta[k] * s[k][m] * eta[m])
                    });
                    let quad = ag(at[0]) + c(2.0) * ce + ese - vz;
                    let mut wg = c(0.0);
                    for k in 0..pn {
                        wg = wg + w(k) * (oieta[k] - cv[k]);
                    }
                    Some(vec![c(0.5) * (quad + ag(at[3]) + chol_logdet(&la)) - wg])
                }
                Family::Laplace => {
                    // The exact anchor `H = ½(Ω⁻¹ + Ω⁻ᵀ) + ½(S + T)`, factored by production's rule.
                    let (s, t) = (mat(at[2]), mat(at[3]));
                    let mut h: Vec<Vec<T>> = (0..pn)
                        .map(|k| {
                            (0..pn)
                                .map(|l| {
                                    c(0.5) * (oi[k][l] + oi[l][k]) + c(0.5) * (s[k][l] + t[k][l])
                                })
                                .collect()
                        })
                        .collect();
                    let lh = factor_anchor(&mut h, &oi)?;
                    let mut wg = c(0.0);
                    for k in 0..pn {
                        wg = wg + w(k) * (oieta[k] + c(0.5) * ag(at[1] + k));
                    }
                    Some(vec![
                        c(0.5) * (ag(at[0]) + prior_of(ctx, &eta, &p)) + c(0.5) * chol_logdet(&lh)
                            - wg,
                    ])
                }
                Family::LaplaceGn | Family::AgqGn | Family::AgqExact => {
                    // The anchor: FOCEI's `H̃ = Ω⁻¹ + S`, or the exact `½(Ω⁻¹ + Ω⁻ᵀ) + ½(S + T)`;
                    // factored by production's rule.
                    let s = mat(at[2]);
                    let t = (family == Family::AgqExact).then(|| mat(at[3]));
                    let mut h: Vec<Vec<T>> = (0..pn)
                        .map(|k| {
                            (0..pn)
                                .map(|l| match &t {
                                    None => oi[k][l] + s[k][l],
                                    Some(t) => {
                                        c(0.5) * (oi[k][l] + oi[l][k])
                                            + c(0.5) * (s[k][l] + t[k][l])
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    let lh = factor_anchor(&mut h, &oi)?;
                    let half_logdet = c(0.5) * chol_logdet(&lh);
                    // `wᵀg` (stage 2 only: stage 1's inputs stop at the aggregates).
                    let wg = || {
                        (0..pn).fold(c(0.0), |acc, k| {
                            acc + w(k) * (oieta[k] + c(0.5) * ag(at[1] + k))
                        })
                    };
                    if family == Family::LaplaceGn {
                        return Some(vec![
                            c(0.5) * (ag(at[0]) + prior_of(ctx, &eta, &p)) + half_logdet - wg(),
                        ]);
                    }
                    // b = η + √2·L⁻ᵀz per node (the recorded kernel's arithmetic).
                    let nodes: Vec<Vec<T>> = grid
                        .iter()
                        .map(|(z, _)| {
                            let mut step = vec![c(0.0); pn];
                            for i in (0..pn).rev() {
                                let mut acc = c(z[i]);
                                for j in (i + 1)..pn {
                                    acc = acc - lh[j][i] * step[j];
                                }
                                step[i] = acc / lh[i][i];
                            }
                            (0..pn)
                                .map(|k| eta[k] + c(std::f64::consts::SQRT_2) * step[k])
                                .collect()
                        })
                        .collect();
                    if stage == 1 {
                        return Some(nodes.into_iter().flatten().collect());
                    }
                    let terms: Vec<T> = grid
                        .iter()
                        .zip(&nodes)
                        .enumerate()
                        .map(|(d, ((_, cw), b))| {
                            c(*cw) - c(0.5) * (prior_of(ctx, b, &p) + node_agg(d))
                        })
                        .collect();
                    Some(vec![
                        half_logdet - crate::sens::num::log_sum_exp(&terms) - wg(),
                    ])
                }
            }
        };
        let tag = match family {
            Family::Focei => 7u8,
            Family::Foce => 8u8,
            Family::Laplace => 9u8,
            Family::LaplaceGn => 10u8,
            Family::AgqGn => 11u8,
            Family::AgqExact => 12u8,
        };
        // Shared across subjects: the reduce reads no subject data. Its literals are Ω's
        // placed factor and the parameters at this point, so their bits key it.
        let lits: Vec<u64> = so
            .chol
            .iter()
            .flatten()
            .chain(&sc.params.theta)
            .chain(&sc.params.sigma.values)
            .map(|v| v.to_bits())
            .collect();
        let kr = super::kernel::kernel_run_shared(
            model,
            subject,
            super::kernel::variant_of(&(self.variant.bytes(), tag, stage, nn, pn, nx, lits)),
            layout,
            rin,
            n_out,
            &record,
        )?;
        self.st_reduce.borrow_mut()[si] = Some(kr.plan().clone());
        Some(kr)
    }

    /// The structured pass's forward over reverse, stage 2: the buffer
    /// [`super::kernel::KernelRun::reverse_tangent`] returns for the kernel at `seed` (`Ψ_w`: `1`
    /// on `F`, `−w` on `g`, value lanes only) — per kernel input its value adjoint and its
    /// tangent along each direction — on every input but the anchor rows, whose
    /// back-propagation [`Self::struct_finish`] completes once every contribution to the
    /// aggregates' adjoint is in; and `ψ_a`.
    pub(super) fn struct_adjoint(
        &self,
        r: &CovRun,
        seed: &[f64],
    ) -> Option<(Vec<f64>, Vec<f64>, Pending)> {
        let inputs = r.st_inputs.as_ref()?;
        let (sl, kt) = (&r.slots, r.kt);
        let lk = 1 + kt;
        let pn = self.pn();
        let nx = self.nx;
        if seed.first() != Some(&1.0) {
            return None;
        }
        let w: Vec<f64> = (0..pn).map(|k| -seed[(1 + k) * lk]).collect();
        let val = |s: usize| inputs[s * lk];
        let tan = |s: usize, a: usize| inputs[s * lk + 1 + a];
        let rows = self.rows_at(inputs, sl, kt)?;
        let family = rows.family;
        let na = rows.aggr.len();
        let at = self.anchor_tan(&rows, inputs, sl, kt);
        let nn = if family.nodes() { sl.n_nodes } else { 0 };
        let node_slot = |d: usize, j: usize| sl.nf + d * sl.n_rows + j;
        let nodes = if nn > 0 {
            Some(self.node_rows(&|d, j| val(node_slot(d, j)), &|i| val(sl.x + i), nn)?)
        } else {
            None
        };
        // A node row's local tangents along direction `a`: its prediction's, then σ's.
        let s0 = packed_segments(self.template).sigma_start();
        let n_sig = self.sc.params.sigma.values.len();
        let nla = 1 + n_sig;
        let node_u = |d: usize, j: usize, a: usize| -> [f64; NL] {
            let mut u = [0.0; NL];
            u[0] = tan(node_slot(d, j), a);
            for q in 0..n_sig {
                u[1 + q] = tan(sl.x + s0 + q, a);
            }
            u
        };
        // The node aggregates' tangents.
        let mut ndot = vec![0.0; kt * nn];
        if let Some(nr) = nodes.as_ref() {
            for d in 0..nn {
                for (j, x) in nr.l[d].iter().enumerate() {
                    for a in 0..kt {
                        let u = node_u(d, j, a);
                        ndot[a * nn + d] += (0..nla).map(|q| x.grad[q] * u[q]).sum::<f64>();
                    }
                }
            }
        }
        // Stage 2 along the directions: (η, x, A, N, w), each value then `kt` tangents.
        let mut rin = self.stage1_in(&rows, &at, inputs, sl, kt);
        let base = rin.len() / lk;
        rin.resize((base + nn + pn) * lk, 0.0);
        if let Some(nr) = nodes.as_ref() {
            for d in 0..nn {
                rin[(base + d) * lk] = nr.agg[d];
                for a in 0..kt {
                    rin[(base + d) * lk + 1 + a] = ndot[a * nn + d];
                }
            }
        }
        for k in 0..pn {
            rin[(base + nn + k) * lk] = w[k];
        }
        let mut s2 = vec![0.0; lk];
        s2[0] = 1.0;
        let radj = self.reduce_for(family, 2, &rin, kt, &s2)?;
        let a_val: Vec<f64> = (0..na).map(|t| radj[(pn + nx + t) * lk]).collect();
        let mut a_tan = vec![0.0; kt * na];
        for a in 0..kt {
            for t in 0..na {
                a_tan[a * na + t] = radj[(pn + nx + t) * lk + 1 + a];
            }
        }
        let mut adj = vec![0.0; sl.end * lk];
        for k in 0..pn {
            adj[(sl.eta + k) * lk..(sl.eta + k + 1) * lk]
                .copy_from_slice(&radj[k * lk..(k + 1) * lk]);
        }
        for i in 0..nx {
            adj[(sl.x + i) * lk..(sl.x + i + 1) * lk]
                .copy_from_slice(&radj[(pn + i) * lk..(pn + i + 1) * lk]);
        }
        // `ψ_a`: the directional derivative, by the chain rule through the aggregates — the
        // reduce's own η and x, the anchor's aggregates, the nodes'.
        let mut psi = vec![0.0; kt];
        for (a, pa) in psi.iter_mut().enumerate() {
            let mut v = 0.0;
            for k in 0..pn {
                v += radj[k * lk] * tan(sl.eta + k, a);
            }
            for i in 0..nx {
                v += radj[(pn + i) * lk] * tan(sl.x + i, a);
            }
            for t in 0..na {
                v += a_val[t] * at.adot[a * na + t];
            }
            for d in 0..nn {
                v += radj[(base + d) * lk] * ndot[a * nn + d];
            }
            *pa = v;
        }
        // The node rows: `N_d`'s adjoint back to the node predictions and σ.
        if let Some(nr) = nodes.as_ref() {
            for d in 0..nn {
                let (nv, nt) = (
                    radj[(base + d) * lk],
                    &radj[(base + d) * lk + 1..(base + d + 1) * lk],
                );
                for (j, x) in nr.l[d].iter().enumerate() {
                    let ls = |lane: usize| {
                        if lane == 0 {
                            node_slot(d, j)
                        } else {
                            sl.x + s0 + lane - 1
                        }
                    };
                    for lane in 0..nla {
                        adj[ls(lane) * lk] += nv * x.grad[lane];
                    }
                    for a in 0..kt {
                        let u = node_u(d, j, a);
                        for lane in 0..nla {
                            let hu: f64 = (0..nla).map(|q| x.hess[lane][q] * u[q]).sum();
                            adj[ls(lane) * lk + 1 + a] += nt[a] * x.grad[lane] + nv * hu;
                        }
                    }
                }
            }
        }
        #[cfg(test)]
        STRUCT_PASSES.with(|c| c.set(c.get() + 1));
        Some((adj, psi, Pending { at, a_val, a_tan }))
    }

    /// The structured pass's forward over reverse, completed: under quadrature, stage 1's
    /// forward over reverse seeded with the node locations' adjoints `b_seed` (laid out as
    /// [`Self::struct_nodes`]' output; the node plans' reverse) adds its η, x and aggregate
    /// adjoints; then the aggregates' adjoint back through the anchor rows into `adj`.
    pub(super) fn struct_finish(
        &self,
        r: &CovRun,
        adj: &mut [f64],
        pending: Pending,
        b_seed: Option<&[f64]>,
    ) -> Option<()> {
        let inputs = r.st_inputs.as_ref()?;
        let (sl, kt) = (&r.slots, r.kt);
        let lk = 1 + kt;
        let pn = self.pn();
        let nx = self.nx;
        let rows = self.st_rows.borrow();
        let rows = rows.as_ref()?;
        let Pending {
            at,
            mut a_val,
            mut a_tan,
        } = pending;
        let family = rows.family;
        let aggs = family.aggs();
        let (offs, na) = offsets(aggs, pn);
        if let Some(bs) = b_seed {
            let rin = self.stage1_in(rows, &at, inputs, sl, kt);
            let radj = self.reduce_for(family, 1, &rin, kt, bs)?;
            for k in 0..pn {
                for x in 0..lk {
                    adj[(sl.eta + k) * lk + x] += radj[k * lk + x];
                }
            }
            for i in 0..nx {
                for x in 0..lk {
                    adj[(sl.x + i) * lk + x] += radj[(pn + i) * lk + x];
                }
            }
            for t in 0..na {
                a_val[t] += radj[(pn + nx + t) * lk];
                for a in 0..kt {
                    a_tan[a * na + t] += radj[(pn + nx + t) * lk + 1 + a];
                }
            }
        }
        let pv = Preps::new(aggs, &offs, pn, 1, |_| &a_val[..]);
        let pt = Preps::new(aggs, &offs, pn, kt, |a| &a_tan[a * na..(a + 1) * na]);
        let nq = rows.q.first().map_or(0, Vec::len);
        let kk = family.n_scalars();
        let AnchorTan {
            tg,
            slots,
            stride,
            o_g,
            o_q,
            o_r,
            nla,
            ..
        } = at;
        let (mut dg, mut dq, mut ds) = (vec![0.0; pn], vec![0.0; nq], vec![0.0; kk]);
        let (mut dgd, mut dqd, mut dsd) = (vec![0.0; pn], vec![0.0; nq], vec![0.0; kk]);
        // Per aggregate, `P·G` of a `Gg` aggregate's value adjoint at the row: direction-free.
        let mut qg = vec![0.0; aggs.len() * pn];
        let mut sc = vec![0.0; 2 * pn];
        for j in 0..sl.n_obs {
            let (g, q) = (&rows.g[j][..], &rows.q[j][..]);
            let (rj, ls) = (&rows.s[j], &slots[j]);
            dg.fill(0.0);
            dq.fill(0.0);
            ds.fill(0.0);
            for (k, &(si, mono)) in aggs.iter().enumerate() {
                let (v, dv) = if mono == Mono::Q {
                    (q, &mut dq[..])
                } else {
                    (g, &mut dg[..])
                };
                ds[si] += back(
                    pv.get(0, k),
                    v,
                    rj[si].value,
                    dv,
                    &mut qg[k * pn..(k + 1) * pn],
                );
            }
            for (lane, &s) in ls.iter().enumerate().take(nla) {
                if s != usize::MAX {
                    adj[s * lk] += (0..kk).map(|k| ds[k] * rj[k].grad[lane]).sum::<f64>();
                }
            }
            for k in 0..pn {
                adj[sl.f(j, 1 + k) * lk] += dg[k];
            }
            for t in 0..nq {
                adj[sl.f(j, 1 + pn + t) * lk] += dq[t];
            }
            for a in 0..kt {
                let t = &tg[(j * kt + a) * stride..(j * kt + a + 1) * stride];
                let u = &t[..nla];
                dgd.fill(0.0);
                dqd.fill(0.0);
                dsd.fill(0.0);
                for (k, &(si, mono)) in aggs.iter().enumerate() {
                    let (v, vd, dvd) = if mono == Mono::Q {
                        (q, &t[o_q..o_r], &mut dqd[..])
                    } else {
                        (g, &t[o_g..o_q], &mut dgd[..])
                    };
                    dsd[si] += back_t(
                        pv.get(0, k),
                        pt.get(a, k),
                        v,
                        vd,
                        rj[si].value,
                        t[o_r + si],
                        &qg[k * pn..(k + 1) * pn],
                        dvd,
                        &mut sc,
                    );
                }
                for (lane, &s) in ls.iter().enumerate().take(nla) {
                    if s == usize::MAX {
                        continue;
                    }
                    let mut v = 0.0;
                    for k in 0..kk {
                        let hu: f64 = rj[k].hess[lane][..nla]
                            .iter()
                            .zip(u)
                            .map(|(h, x)| h * x)
                            .sum();
                        v += ds[k] * hu + dsd[k] * rj[k].grad[lane];
                    }
                    adj[s * lk + 1 + a] += v;
                }
                for k in 0..pn {
                    adj[sl.f(j, 1 + k) * lk + 1 + a] += dgd[k];
                }
                for t in 0..nq {
                    adj[sl.f(j, 1 + pn + t) * lk + 1 + a] += dqd[t];
                }
            }
        }
        Some(())
    }

    /// The mode response's `(∇_η F, H = ∂g/∂η, ∂g along each of `along`)` for a structured
    /// subject: the kernel's adjoints of `F` and `g` from the rows and one reduce run along `w`
    /// (its tangents `−∂g/∂·`), and the plan's from the prediction basis's `J` (`Jᵀ` of the
    /// kernel's adjoints, a folded θ through `∂μ/∂θ`) — no plan reverse, no kernel run. `None`:
    /// not structured, or a plan the basis does not serve (FOCE's population plan is not
    /// assembled here); the recorded path then answers.
    pub(super) fn struct_mode(
        &self,
        along: &[Dir],
    ) -> Option<(Vec<f64>, DMatrix<f64>, DMatrix<f64>)> {
        let family = self.family()?;
        let model = self.sc.model;
        let pn = self.pn();
        let nx = self.nx;
        let n_obs = self.subject.observations.len();
        let f_dep = model.error_spec.has_f_dependent_variance();
        if along.is_empty() || n_obs == 0 {
            return None;
        }
        // FOCE under an `f`-dependent variance reads the population predictions too.
        let need_pop = family == Family::Foce && f_dep;
        self.prepare(along);
        let kt = super::pass_width(true, along.len());
        let nq = if self.second { pn * (pn + 1) / 2 } else { 0 };
        let base = Layout {
            n: pn as u16,
            m: nq as u16,
            th: false,
            wide: false,
            tan: 0,
        };
        let eta = &self.sc.eta[..pn];
        let zero = |_: usize, _: usize| 0.0;
        // Only a basis serves here: a plan the basis is not narrower for declines before running
        // along the directions (that run would be discarded, and the recorded path answers).
        let axes_only = |lanes: Lanes<'_>| matches!(lanes, Lanes::Axes(_));
        let pr = self.plan_run(
            PlanKey::Pred,
            along,
            pn,
            self.th.len() + self.nz,
            true,
            &zero,
            kt,
            &|lanes, et, k| {
                axes_only(lanes)
                    .then(|| self.run_pred(model, eta, et, lanes, base, k))
                    .flatten()
            },
        )?;
        if !matches!(pr, PlanRun::Basis(_)) {
            return None;
        }
        if need_pop {
            let zeros = vec![0.0; pn];
            let pp = self.plan_run(
                PlanKey::Pop,
                along,
                pn,
                self.th.len() + self.nz,
                false,
                &zero,
                kt,
                &|lanes, et, k| {
                    axes_only(lanes)
                        .then(|| {
                            self.run_pred(self.sc.pop_model, &zeros, et, lanes, Layout::F64, k)
                        })
                        .flatten()
                },
            )?;
            if !matches!(pp, PlanRun::Basis(_)) {
                return None;
            }
        }
        // Quadrature: the nodes' locations by stage 1 (values), and each node plan's basis there.
        let n_par = self.th.len() + self.nz;
        let (rin1, node_b) = if family.nodes() {
            let aggr = {
                let bases = self.bases.borrow();
                let (_, lb0, v0, _) = bases.get(&PlanKey::Pred)?.parts();
                let rows = self.rows(
                    &|j| v0[j * lb0],
                    &|_| None,
                    &|j, k| v0[j * lb0 + 1 + k],
                    &|j, t| v0[j * lb0 + 1 + pn + t],
                    &|i| self.x[i],
                )?;
                rows.aggr.clone()
            };
            let rin1: Vec<f64> = eta
                .iter()
                .chain(self.x.iter())
                .chain(&aggr)
                .copied()
                .collect();
            let nn = match self.objective {
                Objective::Agq { n_agq, .. } => agq_grid(n_agq, pn).len(),
                _ => 0,
            };
            let kr = self.reduce_run(family, 1, &rin1, 0, nn * pn)?;
            let b: Vec<Vec<f64>> = (0..nn)
                .map(|d| (0..pn).map(|k| kr.out[d * pn + k]).collect())
                .collect();
            (rin1, b)
        } else {
            (Vec::new(), Vec::new())
        };
        let nn = node_b.len();
        for (d, bv) in node_b.iter().enumerate() {
            let pr = self.plan_run(
                PlanKey::Node(d),
                along,
                pn,
                n_par,
                true,
                &zero,
                kt,
                &|lanes, et, k| {
                    axes_only(lanes)
                        .then(|| self.run_pred(model, bv, et, lanes, Layout::F64, k))
                        .flatten()
                },
            )?;
            if !matches!(pr, PlanRun::Basis(_)) {
                return None;
            }
        }
        let bases = self.bases.borrow();
        let (axes, lb, vals, jac) = bases.get(&PlanKey::Pred)?.parts();
        let n_ax = axes.len();
        let node_parts: Vec<_> = (0..nn)
            .map(|d| bases.get(&PlanKey::Node(d)).map(|b| b.parts()))
            .collect::<Option<Vec<_>>>()?;
        let pop = if need_pop {
            Some(bases.get(&PlanKey::Pop)?.parts())
        } else {
            None
        };
        let rows = self.rows(
            &|j| vals[j * lb],
            &|j| pop.map(|(_, plb, pv, _)| pv[j * plb]),
            &|j, k| vals[j * lb + 1 + k],
            &|j, t| vals[j * lb + 1 + pn + t],
            &|i| self.x[i],
        )?;
        let aggs = family.aggs();
        let (at, na) = offsets(aggs, pn);
        let nodes = if nn > 0 {
            Some(self.node_rows(
                &|d, j| {
                    let (_, nlb, nv, _) = node_parts[d];
                    nv[j * nlb]
                },
                &|i| self.x[i],
                nn,
            )?)
        } else {
            None
        };
        // The reduce at `w = 0` along each `w_o`: values `∂F/∂·`, tangent `o` `−∂g_o/∂·`.
        let lk = 1 + pn;
        let n_in = pn + nx + na + nn + pn;
        let mut rin = vec![0.0; n_in * lk];
        if let Some(nr) = nodes.as_ref() {
            for d in 0..nn {
                rin[(pn + nx + na + d) * lk] = nr.agg[d];
            }
        }
        for k in 0..pn {
            rin[k * lk] = eta[k];
        }
        for i in 0..nx {
            rin[(pn + i) * lk] = self.x[i];
        }
        for t in 0..na {
            rin[(pn + nx + t) * lk] = rows.aggr[t];
        }
        for o in 0..pn {
            rin[(pn + nx + na + nn + o) * lk + 1 + o] = 1.0;
        }
        let mut s2 = vec![0.0; lk];
        s2[0] = 1.0;
        let radj = self.reduce_for(family, 2, &rin, pn, &s2)?;
        // Quadrature: `F`'s route through the nodes (`g` reads none). `N_d`'s adjoint back to the
        // node predictions and σ, through each node plan's `Jᵀ` to its location (`b̄`) and to θ
        // (its axes, a folded θ through the fold), then stage 1's reverse to η, x and the
        // anchor's aggregates.
        let n_th = self.th.len();
        let width = self.nx + self.nz + pn;
        let mut node_row = vec![0.0; width];
        let (mut e0, mut x0, mut a0) = (vec![0.0; pn], vec![0.0; nx], vec![0.0; na]);
        if let Some(nr) = nodes.as_ref() {
            let s0 = packed_segments(self.template).sigma_start();
            let n_sig = self.sc.params.sigma.values.len();
            let mut b_bar = vec![0.0; nn * pn];
            for d in 0..nn {
                let an = radj[(pn + nx + na + d) * lk];
                let (nax, nlb, _, njac) = node_parts[d];
                let mut on = vec![0.0; nax.len()];
                for (j, x) in nr.l[d].iter().enumerate() {
                    let fa = an * x.grad[0];
                    for q in 0..n_sig {
                        x0[s0 + q] += an * x.grad[1 + q];
                    }
                    if fa != 0.0 {
                        for (bb, v) in on.iter_mut().enumerate() {
                            *v += fa * njac[j * nlb * nax.len() + bb];
                        }
                    }
                }
                for (bb, &ax) in nax.iter().enumerate() {
                    match ax {
                        Input::Eta(k) => b_bar[d * pn + k] += on[bb],
                        Input::Param(i) if i < n_th => node_row[i] += on[bb] * self.d1[i],
                        Input::Param(i) => node_row[self.nx + (i - n_th)] += on[bb],
                    }
                }
                if let Some(f) = self.fold.as_ref() {
                    for i in 0..n_th {
                        if let Some(dd) = f.of(i) {
                            let sm: f64 = dd
                                .iter()
                                .filter(|e| e.0 < pn)
                                .map(|&(k, m)| m * b_bar[d * pn + k])
                                .sum();
                            node_row[i] += sm * self.d1[i];
                        }
                    }
                }
            }
            let kr1 = self.reduce_run(family, 1, &rin1, 0, nn * pn)?;
            let r1 = kr1.reverse_one(&b_bar)?;
            for k in 0..pn {
                e0[k] += r1[k];
            }
            for i in 0..nx {
                x0[i] += r1[pn + i];
            }
            for t in 0..na {
                a0[t] += r1[pn + nx + t];
            }
        }
        // Per output `y` (`F`, then each `g_o`): the aggregate adjoint, then back to the rows.
        let ny = 1 + pn;
        let sign = |y: usize| if y == 0 { 1.0 } else { -1.0 };
        let lane = |y: usize| if y == 0 { 0 } else { y };
        let mut kf = vec![vec![0.0; n_obs * lb]; ny];
        // The kernel's adjoints of the population predictions (FOCE's local lane 1).
        let mut kp = vec![vec![0.0; n_obs]; ny];
        let mut ke = vec![vec![0.0; pn]; ny];
        let mut kx = vec![vec![0.0; nx]; ny];
        for y in 0..ny {
            let f0 = |v: &[f64], i: usize| if y == 0 { v[i] } else { 0.0 };
            for k in 0..pn {
                ke[y][k] = sign(y) * radj[k * lk + lane(y)] + f0(&e0, k);
            }
            for i in 0..nx {
                kx[y][i] = sign(y) * radj[(pn + i) * lk + lane(y)] + f0(&x0, i);
            }
            let ay: Vec<f64> = (0..na)
                .map(|t| sign(y) * radj[(pn + nx + t) * lk + lane(y)] + f0(&a0, t))
                .collect();
            let py = Preps::new(aggs, &at, pn, 1, |_| &ay[..]);
            let kk = family.n_scalars();
            for j in 0..n_obs {
                let (g, q) = (&rows.g[j], &rows.q[j]);
                let rj = &rows.s[j];
                let mut dg = vec![0.0; pn];
                let mut dq = vec![0.0; q.len()];
                let mut sc = vec![0.0; pn];
                let mut ds = vec![0.0; kk];
                for (k, &(si, mono)) in aggs.iter().enumerate() {
                    let (v, dv) = if mono == Mono::Q {
                        (&q[..], &mut dq[..])
                    } else {
                        (&g[..], &mut dg[..])
                    };
                    ds[si] += back(py.get(0, k), v, rj[si].value, dv, &mut sc);
                }
                kf[y][j * lb] += (0..kk).map(|k| ds[k] * rj[k].grad[0]).sum::<f64>();
                if rows.pop_lane.is_some() {
                    kp[y][j] += (0..kk).map(|k| ds[k] * rj[k].grad[1]).sum::<f64>();
                }
                for k in 0..pn {
                    kf[y][j * lb + 1 + k] += dg[k];
                }
                for (t, &v) in dq.iter().enumerate() {
                    kf[y][j * lb + 1 + pn + t] += v;
                }
                let lane0 = 1 + usize::from(rows.pop_lane.is_some());
                for (qi, &i) in rows.sig.iter().enumerate() {
                    kx[y][i] += (0..kk).map(|k| ds[k] * rj[k].grad[lane0 + qi]).sum::<f64>();
                }
            }
        }
        // Rows over (x, network outputs, η): `Jᵀ` of the kernel's adjoints on each axis, then the
        // kernel's own η and x.
        let width = self.nx + self.nz + pn;
        let n_th = self.th.len();
        let fold = self.fold.as_ref();
        let mut out = vec![vec![0.0; width]; ny];
        for (y, row) in out.iter_mut().enumerate() {
            let mut on_axis = vec![0.0; n_ax];
            for rr in 0..n_obs * lb {
                let a = kf[y][rr];
                if a != 0.0 {
                    for (b, s) in on_axis.iter_mut().enumerate() {
                        *s += a * jac[rr * n_ax + b];
                    }
                }
            }
            let mut eta_ax = vec![0.0; pn];
            for (b, &ax) in axes.iter().enumerate() {
                match ax {
                    Input::Eta(k) => {
                        row[self.eta_slot(k)] += on_axis[b];
                        eta_ax[k] = on_axis[b];
                    }
                    Input::Param(i) if i < n_th => row[i] += on_axis[b] * self.d1[i],
                    Input::Param(i) => row[self.nx + (i - n_th)] += on_axis[b],
                }
            }
            // The population predictions: `Jᵀ` on their axes; an η axis there carries only a
            // folded θ (η is held at 0), so it reaches θ through the fold and not `η̂`.
            let mut pop_eta = vec![0.0; pn];
            if let Some((pax, plb, _, pjac)) = pop {
                let np = pax.len();
                let mut on_p = vec![0.0; np];
                for j in 0..n_obs {
                    let a = kp[y][j];
                    if a != 0.0 {
                        for (b, s) in on_p.iter_mut().enumerate() {
                            *s += a * pjac[j * plb * np + b];
                        }
                    }
                }
                for (b, &ax) in pax.iter().enumerate() {
                    match ax {
                        Input::Eta(k) => pop_eta[k] = on_p[b],
                        Input::Param(i) if i < n_th => row[i] += on_p[b] * self.d1[i],
                        Input::Param(i) => row[self.nx + (i - n_th)] += on_p[b],
                    }
                }
            }
            if let Some(f) = fold {
                for i in 0..n_th {
                    if let Some(d) = f.of(i) {
                        let s: f64 = d
                            .iter()
                            .filter(|e| e.0 < pn)
                            .map(|&(k, m)| m * (eta_ax[k] + pop_eta[k]))
                            .sum();
                        row[i] += s * self.d1[i];
                    }
                }
            }
            for k in 0..pn {
                row[self.eta_slot(k)] += ke[y][k];
            }
            for i in 0..nx {
                row[i] += kx[y][i];
            }
        }
        for (v, n) in out[0].iter_mut().zip(&node_row) {
            *v += n;
        }
        let grad_f: Vec<f64> = (0..pn).map(|k| out[0][self.eta_slot(k)]).collect();
        let h = DMatrix::from_fn(pn, pn, |o, k| out[1 + o][self.eta_slot(k)]);
        let gx = DMatrix::from_fn(pn, along.len(), |o, j| self.along(&out[1 + o], &along[j]));
        out.iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some((grad_f, h, gx))
    }
}
