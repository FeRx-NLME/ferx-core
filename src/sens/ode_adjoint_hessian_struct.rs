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
//!   `H = Ω⁻¹ + ½(S + T)`, `Ψ_w = ½[Σℓ + ηᵀΩ⁻¹η + log|Ω|] + ½log|H| − wᵀ(Ω⁻¹η + ½B)`.
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
    chol, foce_row, foce_woodbury_ok, focei_gauss_row, laplace_row, prior_of,
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
}

impl Family {
    /// The aggregates: `(row scalar, monomial)`; a scalar may feed several.
    fn aggs(self) -> &'static [(usize, Mono)] {
        match self {
            Family::Focei => &[(0, Mono::One), (1, Mono::G), (2, Mono::Gg)],
            Family::Foce => &[(0, Mono::One), (1, Mono::G), (2, Mono::Gg), (3, Mono::One)],
            Family::Laplace => &[(0, Mono::One), (1, Mono::G), (2, Mono::Gg), (1, Mono::Q)],
        }
    }
    /// Row scalars.
    fn n_scalars(self) -> usize {
        match self {
            Family::Focei | Family::Laplace => 3,
            Family::Foce => 4,
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
    /// Per row its packed η-Hessian (Laplace; empty otherwise).
    q: Vec<Vec<f64>>,
    /// The aggregates, concatenated in [`Family::monos`] order.
    aggr: Vec<f64>,
}

#[inline]
fn tri(k: usize, l: usize) -> usize {
    let (k, l) = if l <= k { (k, l) } else { (l, k) };
    k * (k + 1) / 2 + l
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
            Mono::Q => Prep::G(a[..m * (m + 1) / 2].to_vec()),
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
/// `rd` and `G` by `gd`: the scalar adjoint's tangent, and the `G`-adjoint's into `dgd`. `qg`
/// is `P·G` as [`back`] left it (a `Gg` aggregate; direction-free), `sc` scratch of width `2m`.
#[allow(clippy::too_many_arguments)]
fn back_t(
    a: &Prep,
    ad: &Prep,
    g: &[f64],
    gd: &[f64],
    r: f64,
    rd: f64,
    qg: &[f64],
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
            let aggs = family.aggs();
            let (at, na) = offsets(aggs, pn);
            let nq = if family == Family::Laplace {
                pn * (pn + 1) / 2
            } else {
                0
            };
            let mut s = Vec::with_capacity(n_obs);
            let mut g = Vec::with_capacity(n_obs);
            let mut qs = Vec::with_capacity(n_obs);
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
                    Family::Laplace => laplace_row(self.ctx, j, fd, &p)?.to_vec(),
                };
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
                q: qs,
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
        let layout = Layout {
            tan: kt as u8,
            ..Layout::F64
        };
        // The subject's reduce plan, looked up once (its key hashes the subject and the
        // parameters); every later run goes straight to it unless a guard sends it back.
        let mine = self.st_reduce.borrow().clone();
        if let Some(kr) = mine.and_then(|p| super::kernel::KernelRun::of_plan(&p, layout, rin, 1)) {
            return Some(kr);
        }
        let pn = self.pn();
        let nx = self.nx;
        let (at, na) = offsets(family.aggs(), pn);
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
            }
        };
        let tag = match family {
            Family::Focei => 7u8,
            Family::Foce => 8u8,
            Family::Laplace => 9u8,
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
            super::kernel::variant_of(&(self.variant.bytes(), tag, pn, nx, lits)),
            layout,
            rin,
            1,
            &record,
        )?;
        *self.st_reduce.borrow_mut() = Some(kr.plan().clone());
        Some(kr)
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
            &|j, t| val(sl.f(j, 1 + pn + t)),
            &|i| val(sl.x + i),
        )?;
        let family = rows.family;
        let aggs = family.aggs();
        let (at, na) = offsets(aggs, pn);
        let nq = rows.q.first().map_or(0, Vec::len);
        let n_obs = sl.n_obs;
        let kk = family.n_scalars();
        // The live local lanes: prediction, population prediction (FOCE), σ.
        let lane0 = 1 + usize::from(rows.pop_lane.is_some());
        let nla = lane0 + rows.sig.len();
        // Per row, its local lanes' kernel slots; per row and direction, the gathered tangents
        // `u` (local), `Ġ`, `Q̇` and each scalar's `ṙ = ∇r·u` — read once, used by both loops.
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
        // The aggregates' tangents along each direction.
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
                rin[o + 1 + a] = adot[a * na + t];
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
        let pv: Vec<Prep> = aggs
            .iter()
            .enumerate()
            .map(|(k, &(_, mono))| Prep::of(mono, &a_val[at[k]..], pn))
            .collect();
        let pt: Vec<Vec<Prep>> = (0..kt)
            .map(|a| {
                aggs.iter()
                    .enumerate()
                    .map(|(k, &(_, mono))| Prep::of(mono, &a_tan[a][at[k]..], pn))
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
        let (mut dg, mut dq, mut ds) = (vec![0.0; pn], vec![0.0; nq], vec![0.0; kk]);
        let (mut dgd, mut dqd, mut dsd) = (vec![0.0; pn], vec![0.0; nq], vec![0.0; kk]);
        // Per aggregate, `P·G` of a `Gg` aggregate's value adjoint at the row: direction-free.
        let mut qg = vec![0.0; aggs.len() * pn];
        let mut sc = vec![0.0; 2 * pn];
        for j in 0..n_obs {
            let (g, q) = (&rows.g[j][..], &rows.q[j][..]);
            let (rj, ls) = (&rows.s[j], &slots[j]);
            // Values: each scalar's adjoint (summed over the aggregates it feeds) and the `G`-
            // and `Q`-adjoints.
            dg.fill(0.0);
            dq.fill(0.0);
            ds.fill(0.0);
            for (k, &(si, mono)) in aggs.iter().enumerate() {
                let (v, dv) = if mono == Mono::Q {
                    (q, &mut dq[..])
                } else {
                    (g, &mut dg[..])
                };
                ds[si] += back(&pv[k], v, rj[si].value, dv, &mut qg[k * pn..(k + 1) * pn]);
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
            // Tangents along each direction.
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
                        &pv[k],
                        &pt[a][k],
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
        let pr = self.plan_run(
            PlanKey::Pred,
            along,
            pn,
            self.th.len() + self.nz,
            true,
            &zero,
            kt,
            &|lanes, et, k| self.run_pred(model, eta, et, lanes, base, k),
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
            &|j, t| vals[j * lb + 1 + pn + t],
            &|i| self.x[i],
        )?;
        let aggs = family.aggs();
        let (at, na) = offsets(aggs, pn);
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
            let py: Vec<Prep> = aggs
                .iter()
                .enumerate()
                .map(|(k, &(_, mono))| Prep::of(mono, &ay[at[k]..], pn))
                .collect();
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
                    ds[si] += back(&py[k], v, rj[si].value, dv, &mut sc);
                }
                kf[y][j * lb] += (0..kk).map(|k| ds[k] * rj[k].grad[0]).sum::<f64>();
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
