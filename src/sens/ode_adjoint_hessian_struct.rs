//! The covariance Hessian's structured FOCEI kernel (#2980).
//!
//! A pass's likelihood kernel is one recorded program over every observation; its forward over
//! reverse sweeps all of it once per direction. The FOCEI diagonal objective has a structure that
//! sweep does not use:
//!
//! ```text
//!   Ψ_w = ½·Σ_j ℓ_j − ½·Σ_j dl_j·(wᵀG_j) + R(η, x, S),     S = Σ_j α_j·G_j·G_jᵀ
//!   R   = ½·[ηᵀΩ⁻¹η + log|Ω| + log|Ω⁻¹ + S|] − wᵀΩ⁻¹η
//! ```
//!
//! where each row's `(ℓ_j, dl_j, α_j)` ([`crate::sens::focei_kernel::focei_gauss_row`]) reads
//! only its prediction `f_j` and the residual parameters, and the η-gradients `G_j` enter
//! through `S` and `wᵀG_j` alone. So the adjoint tangents the sweep would return are assembled
//! here from three small pieces:
//!
//! - the rows, evaluated once per subject on a second-order dual over their local inputs
//!   (`f_j` and σ's packed coordinates) — the same function production's kernel calls;
//! - the reduce `R`, a small recorded program over `(η, x, S, w)` — `O(m²)` inputs, no rows —
//!   swept along the pass's directions;
//! - per direction, the bilinear algebra of `S` and `wᵀG` over the rows (`O(n_obs·m²)`).
//!
//! Exact: the same derivatives, assembled by the chain rule through `S`; the oracles that check
//! the sweep check this too.

use super::basis::Input;
use super::*;
use crate::sens::dual2::Dual2;
use crate::sens::focei_kernel::{chol, focei_gauss_row, prior_of, structured_rows_supported};

/// A row's local dual: its prediction, then up to `NL − 1` σ packed coordinates.
const NL: usize = 4;

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

/// A subject's rows at the pass-independent point: per row `(ℓ, dl, α)` on the local dual, its
/// `G` values, and `S`.
pub(super) struct Rows {
    /// The σ packed coordinates, in local-lane order (lane `1 + q`).
    sig: Vec<usize>,
    s: Vec<[Dual2<NL>; 3]>,
    g: Vec<Vec<f64>>,
    /// `S` packed lower-triangular, entry `k(k+1)/2 + l` for `l ≤ k`.
    smat: Vec<f64>,
}

#[inline]
fn tri(k: usize, l: usize) -> usize {
    let (k, l) = if l <= k { (k, l) } else { (l, k) };
    k * (k + 1) / 2 + l
}

impl CovPlan<'_> {
    /// Whether this subject's passes take the structured kernel: FOCEI on the diagonal path
    /// with rows of the supported kind and at most `NL − 1` σ, outside the oracles' switches.
    pub(super) fn structured(&self) -> bool {
        self.objective == Objective::Focei
            && !struct_off()
            && !reverse_per_direction()
            && self.nn_z.is_none()
            && self.sc.params.sigma.values.len() < NL
            && structured_rows_supported(self.ctx)
    }

    /// The rows at this point (the point does not move between passes): formed once, from the
    /// predictions `f(j)`, their η-gradients `g(j, k)` and the packed coordinates `x(i)`.
    fn rows(
        &self,
        f: &dyn Fn(usize) -> f64,
        gv: &dyn Fn(usize, usize) -> f64,
        xv: &dyn Fn(usize) -> f64,
    ) -> Option<std::cell::Ref<'_, Rows>> {
        if self.st_rows.borrow().is_none() {
            let pn = self.pn();
            let n_obs = self.subject.observations.len();
            let s0 = packed_segments(self.template).sigma_start();
            let sig: Vec<usize> = (0..self.sc.params.sigma.values.len())
                .map(|k| s0 + k)
                .collect();
            let xo = |i: usize| -> Dual2<NL> {
                let v = xv(i);
                match sig.iter().position(|&s| s == i) {
                    Some(q) => Dual2::var(v, 1 + q),
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
            let mut s = Vec::with_capacity(n_obs);
            let mut g = Vec::with_capacity(n_obs);
            let mut smat = vec![0.0; pn * (pn + 1) / 2];
            for j in 0..n_obs {
                let (l, dl, al) = focei_gauss_row(self.ctx, j, Dual2::var(f(j), 0), &p)?;
                let gj: Vec<f64> = (0..pn).map(|k| gv(j, k)).collect();
                for k in 0..pn {
                    for m in 0..=k {
                        smat[tri(k, m)] += al.value * gj[k] * gj[m];
                    }
                }
                s.push([l, dl, al]);
                g.push(gj);
            }
            *self.st_rows.borrow_mut() = Some(Rows { sig, s, g, smat });
        }
        let rows = self.st_rows.borrow();
        std::cell::Ref::filter_map(rows, Option::as_ref).ok()
    }

    /// The reduce `R(η, x, S, w)` along `kt` tangents (`rin`: each input's value, then its
    /// tangents): a small recorded program, no rows.
    fn reduce_run(&self, rin: &[f64], kt: usize) -> Option<super::kernel::KernelRun> {
        let pn = self.pn();
        let (nx, np) = (self.nx, pn * (pn + 1) / 2);
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
            let oi = p.omega_inv(pn).clone();
            let mut a: Vec<Vec<T>> = vec![vec![c(0.0); pn]; pn];
            for k in 0..pn {
                for m in 0..pn {
                    a[k][m] = oi[k][m] + lv[pn + nx + tri(k, m)];
                }
            }
            let ld = chol_logdet(&chol(&a)?);
            let mut wpe = c(0.0);
            for k in 0..pn {
                for m in 0..pn {
                    wpe = wpe + lv[pn + nx + np + k] * oi[k][m] * eta[m];
                }
            }
            Some(vec![c(0.5) * (prior_of(ctx, &eta, &p) + ld) - wpe])
        };
        let layout = Layout {
            tan: kt as u8,
            ..Layout::F64
        };
        super::kernel::kernel_run(
            model,
            subject,
            super::kernel::variant_of(&(self.variant.bytes(), 7u8, pn, nx)),
            layout,
            rin,
            1,
            &record,
        )
    }

    /// The mode response's `(∇_η F, H = ∂g/∂η, ∂g along each of `along`)` for a structured
    /// subject: the kernel's adjoints of `F` and `g` from the rows and one reduce run along `w`
    /// (its tangents `−∂(Ω⁻¹η)/∂·`), and the plan's from the prediction basis's `J` (`Jᵀ` of the
    /// kernel's adjoints, a folded θ through `∂μ/∂θ`) — no plan reverse, no kernel run. `None`:
    /// not structured, or the basis did not serve; the recorded path then answers.
    pub(super) fn struct_mode(
        &self,
        along: &[Dir],
    ) -> Option<(Vec<f64>, DMatrix<f64>, DMatrix<f64>)> {
        if !self.structured() || along.is_empty() {
            return None;
        }
        let model = self.sc.model;
        let pn = self.pn();
        let (nx, np) = (self.nx, pn * (pn + 1) / 2);
        let n_obs = self.subject.observations.len();
        if n_obs == 0 {
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
        let rows = self.rows(&|j| vals[j * lb], &|j, k| vals[j * lb + 1 + k], &|i| {
            self.x[i]
        })?;
        // The reduce at `w = 0` along each `w_o`: values `∂R₀/∂·`, tangent `o` `−∂(Ω⁻¹η)_o/∂·`.
        let lk = 1 + pn;
        let n_in = pn + nx + np + pn;
        let mut rin = vec![0.0; n_in * lk];
        for k in 0..pn {
            rin[k * lk] = eta[k];
        }
        for i in 0..nx {
            rin[(pn + i) * lk] = self.x[i];
        }
        for t in 0..np {
            rin[(pn + nx + t) * lk] = rows.smat[t];
        }
        for o in 0..pn {
            rin[(pn + nx + np + o) * lk + 1 + o] = 1.0;
        }
        let kr = self.reduce_run(&rin, pn)?;
        let mut rs = vec![0.0; lk];
        rs[0] = 1.0;
        let radj = kr.reverse_tangent(&rs)?;
        let mut q0 = vec![vec![0.0; pn]; pn];
        for k in 0..pn {
            for m in 0..=k {
                let v = radj[(pn + nx + tri(k, m)) * lk];
                if k == m {
                    q0[k][k] = v;
                } else {
                    q0[k][m] = 0.5 * v;
                    q0[m][k] = 0.5 * v;
                }
            }
        }
        // Per output `y` (`F`, then each `g_o`): the kernel's adjoints of the predictions' base
        // lanes (`f_j`, `G_j`), of η and of x.
        let ny = 1 + pn;
        let mut kf = vec![vec![0.0; n_obs * lb]; ny];
        let mut ke = vec![vec![0.0; pn]; ny];
        let mut kx = vec![vec![0.0; nx]; ny];
        for k in 0..pn {
            ke[0][k] = radj[k * lk];
            for o in 0..pn {
                ke[1 + o][k] = -radj[k * lk + 1 + o];
            }
        }
        for i in 0..nx {
            kx[0][i] = radj[(pn + i) * lk];
            for o in 0..pn {
                kx[1 + o][i] = -radj[(pn + i) * lk + 1 + o];
            }
        }
        for j in 0..n_obs {
            let [l, dl, al] = &rows.s[j];
            let g = &rows.g[j];
            let qg: Vec<f64> = (0..pn)
                .map(|k| (0..pn).map(|m| q0[k][m] * g[m]).sum())
                .collect();
            let c_al: f64 = (0..pn).map(|k| g[k] * qg[k]).sum();
            kf[0][j * lb] += 0.5 * l.grad[0] + c_al * al.grad[0];
            for k in 0..pn {
                kf[0][j * lb + 1 + k] += 2.0 * al.value * qg[k];
                kf[1 + k][j * lb] += 0.5 * dl.grad[0] * g[k];
                kf[1 + k][j * lb + 1 + k] += 0.5 * dl.value;
            }
            for (q, &i) in rows.sig.iter().enumerate() {
                kx[0][i] += 0.5 * l.grad[1 + q] + c_al * al.grad[1 + q];
                for o in 0..pn {
                    kx[1 + o][i] += 0.5 * dl.grad[1 + q] * g[o];
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
            for r in 0..n_obs * lb {
                let a = kf[y][r];
                if a != 0.0 {
                    for (b, s) in on_axis.iter_mut().enumerate() {
                        *s += a * jac[r * n_ax + b];
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

    /// The structured pass's forward over reverse: the buffer [`super::kernel::KernelRun::reverse_tangent`]
    /// returns for the kernel at `seed` (`Ψ_w`: `1` on `F`, `−w` on `g`, value lanes only) — per
    /// kernel input its value adjoint and its tangent along each direction — and `ψ_a`.
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
        let tan = |s: usize, a: usize| inputs[s * lk + 1 + a];
        let val = |s: usize| inputs[s * lk];
        let rows = self.rows(&|j| val(sl.f(j, 0)), &|j, k| val(sl.f(j, 1 + k)), &|i| {
            val(sl.x + i)
        })?;
        let ns = rows.sig.len();
        let np = pn * (pn + 1) / 2;
        // Per direction and row: the local inputs' tangents, the scalars' tangents, `Ġ`.
        let n_obs = sl.n_obs;
        let ud = |j: usize, a: usize| -> [f64; NL] {
            let mut u = [0.0; NL];
            u[0] = tan(sl.f(j, 0), a);
            for (q, &i) in rows.sig.iter().enumerate() {
                u[1 + q] = tan(sl.x + i, a);
            }
            u
        };
        let dot = |d: &Dual2<NL>, u: &[f64; NL]| (0..NL).map(|q| d.grad[q] * u[q]).sum::<f64>();
        // `Ṡ_a`.
        let mut sdot = vec![vec![0.0; np]; kt];
        for (a, sd) in sdot.iter_mut().enumerate() {
            for j in 0..n_obs {
                let u = ud(j, a);
                let (al, g) = (&rows.s[j][2], &rows.g[j]);
                let ald = dot(al, &u);
                for k in 0..pn {
                    let gk = tan(sl.f(j, 1 + k), a);
                    for m in 0..=k {
                        let gm = tan(sl.f(j, 1 + m), a);
                        sd[tri(k, m)] += ald * g[k] * g[m] + al.value * (gk * g[m] + g[k] * gm);
                    }
                }
            }
        }
        // The reduce along the directions: inputs (η, x, S, w), each value then `kt` tangents.
        let n_in = pn + nx + np + pn;
        let mut rin = vec![0.0; n_in * lk];
        for k in 0..pn {
            rin[k * lk..(k + 1) * lk]
                .copy_from_slice(&inputs[(sl.eta + k) * lk..(sl.eta + k + 1) * lk]);
        }
        for i in 0..nx {
            let at = (pn + i) * lk;
            rin[at..at + lk].copy_from_slice(&inputs[(sl.x + i) * lk..(sl.x + i + 1) * lk]);
        }
        for t in 0..np {
            let at = (pn + nx + t) * lk;
            rin[at] = rows.smat[t];
            for a in 0..kt {
                rin[at + 1 + a] = sdot[a][t];
            }
        }
        for k in 0..pn {
            rin[(pn + nx + np + k) * lk] = w[k];
        }
        let kr = self.reduce_run(&rin, kt)?;
        let mut rs = vec![0.0; lk];
        rs[0] = 1.0;
        let radj = kr.reverse_tangent(&rs)?;
        // `∂R/∂S` as the symmetric `Q` with `Σ_{k≥m} R_S·G_kG_m = GᵀQG`, value and tangents.
        let q_of = |lane: usize| -> Vec<Vec<f64>> {
            let mut q = vec![vec![0.0; pn]; pn];
            for k in 0..pn {
                for m in 0..=k {
                    let v = radj[(pn + nx + tri(k, m)) * lk + lane];
                    if k == m {
                        q[k][k] = v;
                    } else {
                        q[k][m] = 0.5 * v;
                        q[m][k] = 0.5 * v;
                    }
                }
            }
            q
        };
        let mv = |q: &[Vec<f64>], v: &[f64]| -> Vec<f64> {
            (0..pn)
                .map(|k| (0..pn).map(|m| q[k][m] * v[m]).sum())
                .collect()
        };
        let q0 = q_of(0);
        let mut adj = vec![0.0; sl.end * lk];
        // η and x: the reduce's own, then the rows' σ terms below.
        for k in 0..pn {
            adj[(sl.eta + k) * lk..(sl.eta + k + 1) * lk]
                .copy_from_slice(&radj[k * lk..(k + 1) * lk]);
        }
        for i in 0..nx {
            adj[(sl.x + i) * lk..(sl.x + i + 1) * lk]
                .copy_from_slice(&radj[(pn + i) * lk..(pn + i + 1) * lk]);
        }
        let qd: Vec<Vec<Vec<f64>>> = (0..kt).map(|a| q_of(1 + a)).collect();
        for j in 0..n_obs {
            let [l, dl, al] = &rows.s[j];
            let g = &rows.g[j];
            let qg = mv(&q0, g);
            let c_dl = -0.5 * (0..pn).map(|k| w[k] * g[k]).sum::<f64>();
            let c_al: f64 = (0..pn).map(|k| g[k] * qg[k]).sum();
            let coef = [0.5, c_dl, c_al];
            let ds = [l, dl, al];
            // Values: the prediction, `G`, σ.
            adj[sl.f(j, 0) * lk] += (0..3).map(|s| coef[s] * ds[s].grad[0]).sum::<f64>();
            for k in 0..pn {
                adj[sl.f(j, 1 + k) * lk] += -0.5 * w[k] * dl.value + 2.0 * al.value * qg[k];
            }
            for (q, &i) in rows.sig.iter().enumerate() {
                adj[(sl.x + i) * lk] += (0..3).map(|s| coef[s] * ds[s].grad[1 + q]).sum::<f64>();
            }
            // Tangents along each direction.
            for a in 0..kt {
                let u = ud(j, a);
                let gd: Vec<f64> = (0..pn).map(|k| tan(sl.f(j, 1 + k), a)).collect();
                let qdg = mv(&qd[a], g);
                let qgd = mv(&q0, &gd);
                let cd_dl = -0.5 * (0..pn).map(|k| w[k] * gd[k]).sum::<f64>();
                let cd_al: f64 = (0..pn).map(|k| g[k] * qdg[k] + 2.0 * gd[k] * qg[k]).sum();
                let cdot = [0.0, cd_dl, cd_al];
                // `Σ_s [c_s·∇²s·u̇ + ċ_s·∇s]` on local lane `p`.
                let local = |p: usize| -> f64 {
                    (0..3)
                        .map(|s| {
                            let hu: f64 = (0..NL).map(|q| ds[s].hess[p][q] * u[q]).sum();
                            coef[s] * hu + cdot[s] * ds[s].grad[p]
                        })
                        .sum()
                };
                adj[sl.f(j, 0) * lk + 1 + a] += local(0);
                for (q, &i) in rows.sig.iter().enumerate() {
                    adj[(sl.x + i) * lk + 1 + a] += local(1 + q);
                }
                let dld = dot(dl, &u);
                let ald = dot(al, &u);
                for k in 0..pn {
                    adj[sl.f(j, 1 + k) * lk + 1 + a] += -0.5 * w[k] * dld
                        + 2.0 * (ald * qg[k] + al.value * qdg[k] + al.value * qgd[k]);
                }
            }
        }
        let _ = ns;
        // `ψ_a`: the value adjoints along each direction's input tangents.
        let psi: Vec<f64> = (0..kt)
            .map(|a| (0..sl.end).map(|s| adj[s * lk] * tan(s, a)).sum())
            .collect();
        #[cfg(test)]
        STRUCT_PASSES.with(|c| c.set(c.get() + 1));
        Some((adj, psi))
    }
}
