//! The M-step of a non-Gaussian random-effect distribution (#1716,
//! `crate::stats::eta_density`): its Ω block and its shape θ (ν, μ) from the latent draws.
//!
//! A Gaussian block has a sufficient statistic, `E[zzᵀ]`, and keeps its closed form. A
//! density group generally has none, so its parameters `ψ` move along the exact score of
//! the complete-data prior `Σᵢ ln p(zᵢ; ψ)`, preconditioned by its information:
//!
//! - **SAEM** ([`DensityMstep::sa_step`]) takes one step per iteration on the current draws,
//!   `ψ ← ψ + γ I⁻¹ s`, with the information `I` (the sum of per-subject score outer
//!   products) averaged at the same `γ`. This is the "expected gradients, single
//!   iteration" rule NONMEM applies to parameters without a sufficient statistic
//!   (*Introduction to NONMEM 7.5*, §I.34). By Fisher's identity its fixed point is the
//!   MLE.
//! - **MCEM** ([`DensityMstep::maximise`]) maximises the importance-weighted prior of one
//!   iteration's draws outright, by the same Fisher scoring run to convergence.
//!
//! Both backtrack so that a step never lowers the objective it is a step on.
//!
//! `ψ` is unconstrained: a univariate group's `ln ω`, a t block's Cholesky factor (log
//! diagonal, raw off-diagonal), and a shape θ on the logit of its position between its
//! bounds. A FIXed ω or θ is left out of `ψ`. The score comes from forward duals of the
//! one prior implementation ([`crate::stats::eta_density::correction_g`] beside the
//! Gaussian term), so it is exact.

use crate::sens::num::PkNum;
use crate::sens::rtape::DualT;
use crate::stats::eta_density::{correction_g, EtaDensity};
use crate::types::CompiledModel;
use nalgebra::{DMatrix, DVector};

/// Where a group's free parameters sit in `ψ`.
#[derive(Debug, Clone)]
struct GroupLayout {
    /// Whether Ω's block is free (not FIX).
    omega_free: bool,
    /// The shape θ's index and bounds, if it is free.
    shape: Option<(usize, f64, f64)>,
    /// `n(n+1)/2` Cholesky entries (row-major lower triangle) when `omega_free`.
    n_chol: usize,
}

impl GroupLayout {
    fn dim(&self) -> usize {
        self.n_chol + usize::from(self.shape.is_some())
    }
}

/// The density groups' M-step state: per group, the SA-averaged information.
#[derive(Debug, Clone)]
pub(crate) struct DensityMstep {
    layouts: Vec<GroupLayout>,
    info: Vec<Option<DMatrix<f64>>>,
}

/// `logit((θ − lo)/(hi − lo))`, and back.
fn to_unit(theta: f64, lo: f64, hi: f64) -> f64 {
    let u = ((theta - lo) / (hi - lo)).clamp(1e-12, 1.0 - 1e-12);
    (u / (1.0 - u)).ln()
}

fn from_unit<T: PkNum>(phi: T, lo: f64, hi: f64) -> T {
    let one = T::from_f64(1.0);
    T::from_f64(lo) + T::from_f64(hi - lo) * (one / (one + (-phi).exp()))
}

impl DensityMstep {
    /// The state for `model`'s density groups, or `None` for a Gaussian model. `omega_fixed`
    /// and `theta_fixed` are per ETA and per θ; `theta_bounds` are the natural-scale bounds.
    pub(crate) fn new(
        model: &CompiledModel,
        omega_fixed: &[bool],
        theta_fixed: &[bool],
        theta_bounds: &[(f64, f64)],
    ) -> Option<Self> {
        if model.eta_densities.is_empty() {
            return None;
        }
        let layouts: Vec<GroupLayout> = model
            .eta_densities
            .iter()
            .map(|d| {
                let n = d.etas.len();
                let omega_free = !d
                    .etas
                    .iter()
                    .any(|&a| omega_fixed.get(a).copied().unwrap_or(false));
                let shape = d.shape_thetas.first().and_then(|&k| {
                    let (lo, hi) = theta_bounds[k];
                    (!theta_fixed.get(k).copied().unwrap_or(false) && lo < hi)
                        .then_some((k, lo, hi))
                });
                GroupLayout {
                    omega_free,
                    shape,
                    n_chol: if omega_free { n * (n + 1) / 2 } else { 0 },
                }
            })
            .collect();
        let info = vec![None; layouts.len()];
        Some(Self { layouts, info })
    }

    /// The θ indices this M-step owns: every free shape θ. The data M-step must leave them
    /// alone (they enter the prior only).
    pub(crate) fn owned_thetas(&self) -> Vec<usize> {
        self.layouts
            .iter()
            .filter_map(|l| l.shape.map(|s| s.0))
            .collect()
    }

    /// One SAEM step for every group on the current latent draws `etas` (one per subject),
    /// at SA gain `gamma`, writing the new Ω blocks into `omega` and shape θs into `theta`
    /// (natural scale).
    pub(crate) fn sa_step(
        &mut self,
        model: &CompiledModel,
        etas: &[Vec<f64>],
        omega: &mut DMatrix<f64>,
        theta: &mut [f64],
        gamma: f64,
    ) {
        let draws: Vec<(&[f64], f64)> = etas.iter().map(|e| (e.as_slice(), 1.0)).collect();
        for g in 0..self.layouts.len() {
            let layout = self.layouts[g].clone();
            if layout.dim() == 0 {
                continue;
            }
            let d = &model.eta_densities[g];
            let psi = pack(d, &layout, omega, theta);
            let Some((s, j)) = score_and_info(d, &layout, &psi, &draws, omega, theta) else {
                continue;
            };
            let g_eff = gamma.clamp(0.0, 1.0);
            let info = match self.info[g].take() {
                Some(i) => &i + (&j - &i) * g_eff,
                None => j,
            };
            if let Some(step) = info.clone().lu().solve(&s) {
                let psi_new = backtrack(d, &layout, &psi, &(step * g_eff), &draws, omega, theta);
                unpack(d, &layout, &psi_new, omega, theta);
            }
            self.info[g] = Some(info);
        }
    }

    /// The MCEM M-step: maximise `Σ w ln p(z; ψ)` over importance-weighted draws
    /// (`(z, w)`, the weights of each subject summing to 1) by Fisher scoring.
    pub(crate) fn maximise(
        &self,
        model: &CompiledModel,
        draws: &[(&[f64], f64)],
        omega: &mut DMatrix<f64>,
        theta: &mut [f64],
    ) {
        for (g, layout) in self.layouts.iter().enumerate() {
            if layout.dim() == 0 {
                continue;
            }
            let d = &model.eta_densities[g];
            for _ in 0..100 {
                let psi = pack(d, layout, omega, theta);
                let Some((s, j)) = score_and_info(d, layout, &psi, draws, omega, theta) else {
                    break;
                };
                let Some(step) = j.lu().solve(&s) else {
                    break;
                };
                let psi_new = backtrack(d, layout, &psi, &step, draws, omega, theta);
                let moved = (&psi_new - &psi).amax();
                unpack(d, layout, &psi_new, omega, theta);
                if moved < 1e-10 {
                    break;
                }
            }
        }
    }
}

/// Bayes (#1716): one random-walk Metropolis move on `ln ω` of a univariate density
/// group, for its full conditional `∝ Πᵢ p(zᵢ; ω, θ) · IG(ω; a₀, b₀)`. The prior is the
/// inverse gamma the conjugate update gives a Gaussian variance, so the two kinds of ETA
/// share one prior. Moves `omega` in place; returns whether the move was accepted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_log_omega(
    d: &EtaDensity,
    etas: &[Vec<f64>],
    omega: &mut f64,
    theta: &[f64],
    a0: f64,
    b0: f64,
    step: f64,
    rng: &mut impl rand::Rng,
) -> bool {
    use rand_distr::{Distribution, StandardNormal, Uniform};
    debug_assert_eq!(d.etas.len(), 1, "univariate groups only");
    let a = d.etas[0];
    let local = EtaDensity {
        kind: d.kind,
        etas: vec![0],
        shape_thetas: if d.shape_thetas.is_empty() {
            vec![]
        } else {
            vec![0]
        },
    };
    let shape = d.shape_thetas.first().map_or(0.0, |&k| theta[k]);
    let ln_2pi = std::f64::consts::TAU.ln();
    // ln of the full conditional at u = ln ω, the Jacobian `+u` included.
    let log_target = |u: f64| -> f64 {
        let w = u.exp();
        let oi = DMatrix::from_element(1, 1, 1.0 / w);
        let mut lp = -(a0 + 1.0) * u - b0 / w + u;
        for e in etas {
            let z = e[a];
            let c = crate::stats::eta_density::correction(
                std::slice::from_ref(&local),
                &[z],
                &oi,
                &[shape],
            );
            lp += -0.5 * (z * z / w + u + ln_2pi + c);
        }
        lp
    };
    let u0 = omega.ln();
    let n: f64 = StandardNormal.sample(rng);
    let u1 = u0 + step * n;
    let (l0, l1) = (log_target(u0), log_target(u1));
    let log_u = Uniform::new(0.0f64, 1.0)
        .map_or(0.0, |d| d.sample(rng))
        .max(f64::MIN_POSITIVE)
        .ln();
    if l1.is_finite() && log_u < l1 - l0 {
        *omega = u1.exp();
        true
    } else {
        false
    }
}

/// `ψ` of group `d` at the current `omega` and `theta`.
fn pack(d: &EtaDensity, l: &GroupLayout, omega: &DMatrix<f64>, theta: &[f64]) -> DVector<f64> {
    let mut v = Vec::with_capacity(l.dim());
    if l.omega_free {
        let n = d.etas.len();
        let block = DMatrix::from_fn(n, n, |i, j| omega[(d.etas[i], d.etas[j])]);
        let chol = block.clone().cholesky().map(|c| c.l()).unwrap_or_else(|| {
            DMatrix::from_diagonal(&block.diagonal().map(|x| x.max(1e-12).sqrt()))
        });
        for i in 0..n {
            for j in 0..=i {
                v.push(if i == j {
                    chol[(i, i)].ln()
                } else {
                    chol[(i, j)]
                });
            }
        }
    }
    if let Some((k, lo, hi)) = l.shape {
        v.push(to_unit(theta[k], lo, hi));
    }
    DVector::from_vec(v)
}

/// Write `ψ` back into `omega` and `theta`.
fn unpack(
    d: &EtaDensity,
    l: &GroupLayout,
    psi: &DVector<f64>,
    omega: &mut DMatrix<f64>,
    theta: &mut [f64],
) {
    if l.omega_free {
        let n = d.etas.len();
        let lch = chol_of(n, &psi.as_slice()[..l.n_chol]);
        for (i, &a) in d.etas.iter().enumerate() {
            for (j, &b) in d.etas.iter().enumerate() {
                omega[(a, b)] = (0..n).map(|k| lch[i][k] * lch[j][k]).sum();
            }
        }
    }
    if let Some((k, lo, hi)) = l.shape {
        theta[k] = from_unit(psi[l.n_chol], lo, hi);
    }
}

/// The lower Cholesky factor from its packed entries (log diagonal).
fn chol_of<T: PkNum>(n: usize, p: &[T]) -> Vec<Vec<T>> {
    let mut l = vec![vec![T::from_f64(0.0); n]; n];
    let mut at = 0;
    for (i, row) in l.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate().take(i + 1) {
            *x = if i == j { p[at].exp() } else { p[at] };
            at += 1;
        }
    }
    l
}

/// `ln p(z_g; ψ)` (the full group log-density, normalising constants included) over `T`.
fn group_log_density<T: PkNum>(
    d: &EtaDensity,
    l: &GroupLayout,
    psi: &[T],
    z: &[f64],
    omega: &DMatrix<f64>,
    theta: &[f64],
) -> T {
    let n = d.etas.len();
    let c = |v: f64| T::from_f64(v);
    // L of the block: from ψ when free, from Ω when FIXed.
    let lch: Vec<Vec<T>> = if l.omega_free {
        chol_of(n, &psi[..l.n_chol])
    } else {
        let block = DMatrix::from_fn(n, n, |i, j| omega[(d.etas[i], d.etas[j])]);
        let lf = block.clone().cholesky().map(|ch| ch.l()).unwrap_or(block);
        (0..n)
            .map(|i| (0..n).map(|j| c(lf[(i, j)])).collect())
            .collect()
    };
    // L⁻¹ by forward substitution, then Ω⁻¹ = L⁻ᵀL⁻¹ and ln|Ω| = 2 Σ ln L_ii.
    let mut li = vec![vec![c(0.0); n]; n];
    for i in 0..n {
        li[i][i] = c(1.0) / lch[i][i];
        for j in 0..i {
            let mut s = c(0.0);
            for k in j..i {
                s = s + lch[i][k] * li[k][j];
            }
            li[i][j] = -s / lch[i][i];
        }
    }
    let mut oi = vec![vec![c(0.0); n]; n];
    for a in 0..n {
        for b in 0..n {
            let mut s = c(0.0);
            for row in &li {
                s = s + row[a] * row[b];
            }
            oi[a][b] = s;
        }
    }
    let mut log_det = c(0.0);
    for (i, row) in lch.iter().enumerate() {
        log_det = log_det + c(2.0) * row[i].ln();
    }
    let zt: Vec<T> = z.iter().map(|&v| c(v)).collect();
    let mut q = c(0.0);
    for a in 0..n {
        for b in 0..n {
            q = q + oi[a][b] * zt[a] * zt[b];
        }
    }
    // The group on its own: η positions 0..n, its shape θ at index 0.
    let local = EtaDensity {
        kind: d.kind,
        etas: (0..n).collect(),
        shape_thetas: if d.shape_thetas.is_empty() {
            vec![]
        } else {
            vec![0]
        },
    };
    let shape_val = match l.shape {
        Some((_, lo, hi)) => from_unit(psi[l.n_chol], lo, hi),
        None => c(d.shape_thetas.first().map_or(0.0, |&k| theta[k])),
    };
    let corr = correction_g(&[local], &zt, |a, b| oi[a][b], |_| shape_val);
    let ln_2pi = std::f64::consts::TAU.ln();
    c(-0.5) * (q + log_det + c(n as f64 * ln_2pi) + corr)
}

/// The group's latent draw out of a full η vector.
fn group_z(d: &EtaDensity, eta: &[f64]) -> Vec<f64> {
    d.etas.iter().map(|&a| eta[a]).collect()
}

/// `(s, J)`: the weighted score `Σ w ∇ψ ln p` and the weighted outer-product information
/// `Σ w ∇ψ ln p ∇ψ ln pᵀ`, from one forward dual per `ψ` coordinate per draw. `None` when a
/// term is not finite.
fn score_and_info(
    d: &EtaDensity,
    l: &GroupLayout,
    psi: &DVector<f64>,
    draws: &[(&[f64], f64)],
    omega: &DMatrix<f64>,
    theta: &[f64],
) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let m = l.dim();
    let mut s = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, m);
    let mut gi = DVector::zeros(m);
    for &(eta, w) in draws {
        let z = group_z(d, eta);
        for k in 0..m {
            let p: Vec<DualT<1, f64>> = (0..m)
                .map(|i| DualT {
                    v: psi[i],
                    g: [if i == k { 1.0 } else { 0.0 }],
                })
                .collect();
            gi[k] = group_log_density(d, l, &p, &z, omega, theta).g[0];
        }
        if !gi.iter().all(|v| v.is_finite()) {
            return None;
        }
        s += &gi * w;
        j += &gi * gi.transpose() * w;
    }
    Some((s, j))
}

/// `Σ w ln p(z; ψ)` at `f64`.
fn objective(
    d: &EtaDensity,
    l: &GroupLayout,
    psi: &DVector<f64>,
    draws: &[(&[f64], f64)],
    omega: &DMatrix<f64>,
    theta: &[f64],
) -> f64 {
    draws
        .iter()
        .map(|&(eta, w)| {
            w * group_log_density(d, l, psi.as_slice(), &group_z(d, eta), omega, theta)
        })
        .sum()
}

/// `psi + step`, halved until the objective on these draws does not fall (at most 30
/// halvings; `psi` itself if none qualifies).
fn backtrack(
    d: &EtaDensity,
    l: &GroupLayout,
    psi: &DVector<f64>,
    step: &DVector<f64>,
    draws: &[(&[f64], f64)],
    omega: &DMatrix<f64>,
    theta: &[f64],
) -> DVector<f64> {
    let f0 = objective(d, l, psi, draws, omega, theta);
    let mut t = 1.0;
    for _ in 0..30 {
        let cand = psi + step * t;
        let f = objective(d, l, &cand, draws, omega, theta);
        if f.is_finite() && f >= f0 {
            return cand;
        }
        t *= 0.5;
    }
    psi.clone()
}

#[cfg(test)]
#[path = "eta_density_mstep_tests.rs"]
mod tests;
