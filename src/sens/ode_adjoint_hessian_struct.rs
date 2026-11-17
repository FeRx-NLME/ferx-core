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
//!   log|A|] − wᵀ(Ω⁻¹η − c)`.
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
    chol, foce_row, foce_woodbury_ok, focei_gauss_row, prior_of, structured_rows_supported,
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
}

/// The objective's structured form.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Focei,
    Foce,
}

impl Family {
    fn monos(self) -> &'static [Mono] {
        match self {
            Family::Focei => &[Mono::One, Mono::G, Mono::Gg],
            Family::Foce => &[Mono::One, Mono::G, Mono::Gg, Mono::One],
        }
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
    /// The aggregates, concatenated in [`Family::monos`] order.
    aggr: Vec<f64>,
}

#[inline]
fn tri(k: usize, l: usize) -> usize {
    let (k, l) = if l <= k { (k, l) } else { (l, k) };
    k * (k + 1) / 2 + l
}

/// Each monomial's offset in the aggregate vector, and the total width.
fn offsets(monos: &[Mono], m: usize) -> (Vec<usize>, usize) {
    let mut at = Vec::with_capacity(monos.len());
    let mut w = 0;
    for mono in monos {
        at.push(w);
        w += match mono {
            Mono::One => 1,
            Mono::G => m,
            Mono::Gg => m * (m + 1) / 2,
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

fn mv(q: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    q.iter()
        .map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// An aggregate's adjoint prepared for the rows: a scalar, a vector, or (`Gg`) the symmetric
/// `Q` of its packed triangle — formed once per pass, not per row.
enum Prep {
    One(f64),
    G(Vec<f64>),
    Gg(Vec<Vec<f64>>),
}

impl Prep {
    fn of(mono: Mono, a: &[f64], m: usize) -> Self {
        match mono {
            Mono::One => Prep::One(a[0]),
            Mono::G => Prep::G(a[..m].to_vec()),
            Mono::Gg => Prep::Gg(sym(a, m)),
        }
    }
}

/// `y = Q·v` into a reused buffer.
fn mv_into(q: &[Vec<f64>], v: &[f64], y: &mut [f64]) {
    for (yk, r) in y.iter_mut().zip(q) {
        *yk = r.iter().zip(v).map(|(a, b)| a * b).sum();
    }
}

/// A row's scalar adjoint `⟨a, m(G)⟩` and its `G`-adjoint `r·∂⟨a, m(G)⟩/∂G` into `dg`, given
/// the aggregate's prepared adjoint `a`; `qg` is scratch of width `m`.
fn back(a: &Prep, g: &[f64], r: f64, dg: &mut [f64], qg: &mut [f64]) -> f64 {
    match a {
        Prep::One(v) => *v,
        Prep::G(av) => {
            for (d, x) in dg.iter_mut().zip(av) {
                *d += r * x;
            }
            dotv(av, g)
        }
        Prep::Gg(q) => {
            mv_into(q, g, qg);
            for (d, x) in dg.iter_mut().zip(qg.iter()) {
                *d += 2.0 * r * x;
            }
            dotv(g, qg)
        }
    }
}

/// [`back`]'s tangent along a direction moving the aggregate adjoint by `ad`, the scalar by
/// `rd` and `G` by `gd`: the scalar adjoint's tangent, and the `G`-adjoint's into `dgd`.
/// `sc` is scratch of width `3m`.
#[allow(clippy::too_many_arguments)]
fn back_t(
    a: &Prep,
    ad: &Prep,
    g: &[f64],
    gd: &[f64],
    r: f64,
    rd: f64,
    dgd: &mut [f64],
    sc: &mut [f64],
) -> f64 {
    match (a, ad) {
        (Prep::One(_), Prep::One(v)) => *v,
        (Prep::G(av), Prep::G(adv)) => {
            for ((d, x), y) in dgd.iter_mut().zip(av).zip(adv) {
                *d += rd * x + r * y;
            }
            dotv(adv, g) + dotv(av, gd)
        }
        (Prep::Gg(q), Prep::Gg(qd)) => {
            let m = g.len();
            let (qg, rest) = sc.split_at_mut(m);
            let (qdg, qgd) = rest.split_at_mut(m);
            mv_into(q, g, qg);
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
        Mono::G => {
            for k in 0..m {
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
            Objective::Agq { .. } => return None,
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
            let xo = |i: usize| -> Dual2<NL> {
                let v = xv(i);
                match sig.iter().position(|&s| s == i) {
                    Some(q) => Dual2::var(v, lane0 + q),
                    None => Dual2::constant(v),
                }
            };
            let theta: Vec<Dual2<NL>> = unpack_theta(self.template, xo);
            let p = population_params(
                self.sc.model,
                self.subject,
                self.template,
                self.sc.params,
                self.so,
                &theta,
                xo,
            )?;
            let monos = family.monos();
            let (at, na) = offsets(monos, pn);
            let mut s = Vec::with_capacity(n_obs);
            let mut g = Vec::with_capacity(n_obs);
            let mut aggr = vec![0.0; na];
            for j in 0..n_obs {
                let fd = Dual2::var(f(j), 0);
                let r: Vec<Dual2<NL>> = match family {
                    Family::Focei => {
                        let (l, dl, al) = focei_gauss_row(self.ctx, j, fd, &p)?;
                        vec![l, dl, al]
                    }
                    Family::Foce => {
                        // `R⁰` at the population prediction where the variance reads `f`, else
                        // at anything (it does not).
                        let r_eval = match fp(j) {
                            Some(v) => Dual2::var(v, 1),
                            None => Dual2::constant(f(j)),
                        };
                        foce_row(self.ctx, j, fd, r_eval, &p)?.to_vec()
                    }
                };
                let gj: Vec<f64> = (0..pn).map(|k| gv(j, k)).collect();
                for (k, mono) in monos.iter().enumerate() {
                    let zero = vec![0.0; pn];
                    aggr_t(*mono, &gj, &zero, 0.0, r[k].value, pn, &mut aggr[at[k]..]);
                }
                s.push(r);
                g.push(gj);
            }
            if family == Family::Foce {
                // Production's gate: `1 + tr(LᵀSL)` (`S` the third aggregate).
                let l = p.omega_chol(pn);
                let sq = sym(&aggr[at[2]..at[2] + pn * (pn + 1) / 2], pn);
                let mut tr = 0.0;
                for c in 0..pn {
                    for a in 0..pn {
                        for b in 0..pn {
                            tr += l[a][c].value * sq[a][b] * l[b][c].value;
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
                aggr,
            });
        }
        let rows = self.st_rows.borrow();
        std::cell::Ref::filter_map(rows, Option::as_ref).ok()
    }

    /// The reduce `R(η, x, A, w)` along `kt` tangents (`rin`: each input's value, then its
    /// tangents): a small recorded program, no rows.
    fn reduce_run(
        &self,
        family: Family,
        rin: &[f64],
        kt: usize,
    ) -> Option<super::kernel::KernelRun> {
        let pn = self.pn();
        let nx = self.nx;
        let (at, na) = offsets(family.monos(), pn);
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
            let w = |k: usize| lv[pn + nx + na + k];
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
            }
        };
        let layout = Layout {
            tan: kt as u8,
            ..Layout::F64
        };
        let tag = match family {
            Family::Focei => 7u8,
            Family::Foce => 8u8,
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
        super::kernel::kernel_run_shared(
            model,
            subject,
            super::kernel::variant_of(&(self.variant.bytes(), tag, pn, nx, lits)),
            layout,
            rin,
            1,
            &record,
        )
    }

    /// Per row, its local inputs' tangents along direction `a` (`tan(slot, a)`): the
    /// prediction, the population prediction, σ.
    fn local_t(&self, rows: &Rows, sl: &Slots, j: usize, tan: &dyn Fn(usize) -> f64) -> [f64; NL] {
        let mut u = [0.0; NL];
        u[0] = tan(sl.f(j, 0));
        let mut lane = 1;
        if rows.pop_lane.is_some() {
            u[1] = if sl.has_pop { tan(sl.pop + j) } else { 0.0 };
            lane = 2;
        }
        for (q, &i) in rows.sig.iter().enumerate() {
            u[lane + q] = tan(sl.x + i);
        }
        u
    }

    /// The structured pass's forward over reverse: the buffer
    /// [`super::kernel::KernelRun::reverse_tangent`] returns for the kernel at `seed` (`Ψ_w`: `1`
    /// on `F`, `−w` on `g`, value lanes only) — per kernel input its value adjoint and its
    /// tangent along each direction — and `ψ_a`.
    pub(super) fn struct_adjoint(&self, r: &CovRun, seed: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
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
        let rows = self.rows(
            &|j| val(sl.f(j, 0)),
            &|j| sl.has_pop.then(|| val(sl.pop + j)),
            &|j, k| val(sl.f(j, 1 + k)),
            &|i| val(sl.x + i),
        )?;
        let family = rows.family;
        let monos = family.monos();
        let (at, na) = offsets(monos, pn);
        let n_obs = sl.n_obs;
        let ud = |j: usize, a: usize| self.local_t(&rows, sl, j, &|s| tan(s, a));
        let dot = |d: &Dual2<NL>, u: &[f64; NL]| (0..NL).map(|q| d.grad[q] * u[q]).sum::<f64>();
        // The aggregates' tangents along each direction.
        let mut adot = vec![vec![0.0; na]; kt];
        for (a, ad) in adot.iter_mut().enumerate() {
            for j in 0..n_obs {
                let u = ud(j, a);
                let gd: Vec<f64> = (0..pn).map(|k| tan(sl.f(j, 1 + k), a)).collect();
                for (k, mono) in monos.iter().enumerate() {
                    let rk = &rows.s[j][k];
                    aggr_t(
                        *mono,
                        &rows.g[j],
                        &gd,
                        rk.value,
                        dot(rk, &u),
                        pn,
                        &mut ad[at[k]..],
                    );
                }
            }
        }
        // The reduce along the directions: (η, x, A, w), each value then `kt` tangents.
        let n_in = pn + nx + na + pn;
        let mut rin = vec![0.0; n_in * lk];
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
                rin[o + 1 + a] = adot[a][t];
            }
        }
        for k in 0..pn {
            rin[(pn + nx + na + k) * lk] = w[k];
        }
        let kr = self.reduce_run(family, &rin, kt)?;
        let mut rs = vec![0.0; lk];
        rs[0] = 1.0;
        let radj = kr.reverse_tangent(&rs)?;
        let a_val: Vec<f64> = (0..na).map(|t| radj[(pn + nx + t) * lk]).collect();
        let a_tan: Vec<Vec<f64>> = (0..kt)
            .map(|a| (0..na).map(|t| radj[(pn + nx + t) * lk + 1 + a]).collect())
            .collect();
        let pv: Vec<Prep> = (0..monos.len())
            .map(|k| Prep::of(monos[k], &a_val[at[k]..], pn))
            .collect();
        let pt: Vec<Vec<Prep>> = (0..kt)
            .map(|a| {
                (0..monos.len())
                    .map(|k| Prep::of(monos[k], &a_tan[a][at[k]..], pn))
                    .collect()
            })
            .collect();
        let mut adj = vec![0.0; sl.end * lk];
        for k in 0..pn {
            adj[(sl.eta + k) * lk..(sl.eta + k + 1) * lk]
                .copy_from_slice(&radj[k * lk..(k + 1) * lk]);
        }
        for i in 0..nx {
            adj[(sl.x + i) * lk..(sl.x + i + 1) * lk]
                .copy_from_slice(&radj[(pn + i) * lk..(pn + i + 1) * lk]);
        }
        // Local lane `p` of row `j` → its kernel slot.
        let lane_slot = |j: usize, p: usize| -> Option<usize> {
            match (p, rows.pop_lane) {
                (0, _) => Some(sl.f(j, 0)),
                (1, Some(_)) => sl.has_pop.then_some(sl.pop + j),
                (p, pl) => {
                    let q = p - 1 - usize::from(pl.is_some());
                    rows.sig.get(q).map(|&i| sl.x + i)
                }
            }
        };
        let kk = monos.len();
        let (mut dg, mut dgd, mut gd) = (vec![0.0; pn], vec![0.0; pn], vec![0.0; pn]);
        let (mut ds, mut dsd) = (vec![0.0; kk], vec![0.0; kk]);
        let mut scratch = vec![0.0; 3 * pn];
        for j in 0..n_obs {
            let g = &rows.g[j];
            let rj = &rows.s[j];
            // Values: each scalar's adjoint and the `G`-adjoint.
            dg.fill(0.0);
            for k in 0..kk {
                ds[k] = back(&pv[k], g, rj[k].value, &mut dg, &mut scratch[..pn]);
            }
            for p in 0..NL {
                if let Some(s) = lane_slot(j, p) {
                    adj[s * lk] += (0..kk).map(|k| ds[k] * rj[k].grad[p]).sum::<f64>();
                }
            }
            for k in 0..pn {
                adj[sl.f(j, 1 + k) * lk] += dg[k];
            }
            // Tangents along each direction.
            for a in 0..kt {
                let u = ud(j, a);
                for (k, v) in gd.iter_mut().enumerate() {
                    *v = tan(sl.f(j, 1 + k), a);
                }
                dgd.fill(0.0);
                for k in 0..kk {
                    dsd[k] = back_t(
                        &pv[k],
                        &pt[a][k],
                        g,
                        &gd,
                        rj[k].value,
                        dot(&rj[k], &u),
                        &mut dgd,
                        &mut scratch,
                    );
                }
                for p in 0..NL {
                    let Some(s) = lane_slot(j, p) else { continue };
                    let v: f64 = (0..kk)
                        .map(|k| {
                            let hu: f64 = (0..NL).map(|q| rj[k].hess[p][q] * u[q]).sum();
                            ds[k] * hu + dsd[k] * rj[k].grad[p]
                        })
                        .sum();
                    adj[s * lk + 1 + a] += v;
                }
                for k in 0..pn {
                    adj[sl.f(j, 1 + k) * lk + 1 + a] += dgd[k];
                }
            }
        }
        // `ψ_a`: the value adjoints along each direction's input tangents.
        let psi: Vec<f64> = (0..kt)
            .map(|a| (0..sl.end).map(|s| adj[s * lk] * tan(s, a)).sum())
            .collect();
        #[cfg(test)]
        STRUCT_PASSES.with(|c| c.set(c.get() + 1));
        Some((adj, psi))
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
        if along.is_empty() || n_obs == 0 || (family == Family::Foce && f_dep) {
            return None;
        }
        self.prepare(along);
        let kt = super::pass_width();
        let base = Layout {
            n: pn as u16,
            m: 0,
            th: false,
            wide: false,
            tan: 0,
        };
        let eta = &self.sc.eta[..pn];
        let zero = |_: usize, _: usize| 0.0;
        let pr = self.plan_run(
            PlanKey::Pred,
            along,
            pn,
            self.th.len() + self.nz,
            true,
            &zero,
            kt,
            &|lanes, et| self.run_pred(model, eta, et, lanes, base, kt),
        )?;
        if !matches!(pr, PlanRun::Basis(_)) {
            return None;
        }
        let bases = self.bases.borrow();
        let (axes, lb, vals, jac) = bases.get(&PlanKey::Pred)?.parts();
        let n_ax = axes.len();
        let rows = self.rows(
            &|j| vals[j * lb],
            &|_| None,
            &|j, k| vals[j * lb + 1 + k],
            &|i| self.x[i],
        )?;
        let monos = family.monos();
        let (at, na) = offsets(monos, pn);
        // The reduce at `w = 0` along each `w_o`: values `∂F/∂·`, tangent `o` `−∂g_o/∂·`.
        let lk = 1 + pn;
        let n_in = pn + nx + na + pn;
        let mut rin = vec![0.0; n_in * lk];
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
            rin[(pn + nx + na + o) * lk + 1 + o] = 1.0;
        }
        let kr = self.reduce_run(family, &rin, pn)?;
        let mut rs = vec![0.0; lk];
        rs[0] = 1.0;
        let radj = kr.reverse_tangent(&rs)?;
        // Per output `y` (`F`, then each `g_o`): the aggregate adjoint, then back to the rows.
        let ny = 1 + pn;
        let sign = |y: usize| if y == 0 { 1.0 } else { -1.0 };
        let lane = |y: usize| if y == 0 { 0 } else { y };
        let mut kf = vec![vec![0.0; n_obs * lb]; ny];
        let mut ke = vec![vec![0.0; pn]; ny];
        let mut kx = vec![vec![0.0; nx]; ny];
        for y in 0..ny {
            for k in 0..pn {
                ke[y][k] = sign(y) * radj[k * lk + lane(y)];
            }
            for i in 0..nx {
                kx[y][i] = sign(y) * radj[(pn + i) * lk + lane(y)];
            }
            let ay: Vec<f64> = (0..na)
                .map(|t| sign(y) * radj[(pn + nx + t) * lk + lane(y)])
                .collect();
            let py: Vec<Prep> = (0..monos.len())
                .map(|k| Prep::of(monos[k], &ay[at[k]..], pn))
                .collect();
            for j in 0..n_obs {
                let g = &rows.g[j];
                let rj = &rows.s[j];
                let mut dg = vec![0.0; pn];
                let mut qg = vec![0.0; pn];
                let ds: Vec<f64> = (0..monos.len())
                    .map(|k| back(&py[k], g, rj[k].value, &mut dg, &mut qg))
                    .collect();
                kf[y][j * lb] += (0..monos.len()).map(|k| ds[k] * rj[k].grad[0]).sum::<f64>();
                for k in 0..pn {
                    kf[y][j * lb + 1 + k] += dg[k];
                }
                let lane0 = 1 + usize::from(rows.pop_lane.is_some());
                for (q, &i) in rows.sig.iter().enumerate() {
                    kx[y][i] += (0..monos.len())
                        .map(|k| ds[k] * rj[k].grad[lane0 + q])
                        .sum::<f64>();
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
            if let Some(f) = fold {
                for i in 0..n_th {
                    if let Some(d) = f.of(i) {
                        let s: f64 = d
                            .iter()
                            .filter(|e| e.0 < pn)
                            .map(|&(k, m)| m * eta_ax[k])
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
        let grad_f: Vec<f64> = (0..pn).map(|k| out[0][self.eta_slot(k)]).collect();
        let h = DMatrix::from_fn(pn, pn, |o, k| out[1 + o][self.eta_slot(k)]);
        let gx = DMatrix::from_fn(pn, along.len(), |o, j| self.along(&out[1 + o], &along[j]));
        out.iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some((grad_f, h, gx))
    }
}
