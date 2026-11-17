//! Parameter-expanded SAEM (PX-SAEM) for the random-effect covariance.
//!
//! **Expansion.** Write `η_i = A η*_i` with `η*_i ~ N(0, Ω*)`; the original model is `A = I`.
//! The expected complete-data log-likelihood separates as
//! `E[log p(y_i | θ, A η*_i)] + E[log N(η*_i; 0, Ω*)]`, so `Ω*` is the usual statistic
//! `S = mean(ηηᵀ)` and `A` maximises the *data* term alone, with the draws `η_i` held. The
//! reduced parameter is `Ω = A S Aᵀ` (Liu, Rubin & Wu 1998, Biometrika 85, "Parameter
//! expansion to accelerate EM: the PX-EM algorithm"; SAEM adaptation: Lavielle & Meza
//! 2007, Statistics and Computing 17, "A parameter expansion version of the SAEM
//! algorithm"). Taken as an ECM step — `θ` and `Ω*` first at `A = I`, then `A` given them —
//! the iteration is a valid generalised EM.
//!
//! **`A` respects Ω's structure.** `A` is lower triangular with a free entry `(j, l)` only
//! where Ω has a free `(j, l)` covariance (the diagonal always): `A S Aᵀ` then has Ω's
//! block pattern, so a diagonal Ω stays diagonal and a block Ω keeps its blocks. With a
//! diagonal `A` the expansion only rescales variances (correlations are untouched); with
//! the block pattern it can also move the correlations inside a block.
//!
//! **Why it helps near a variance boundary.** When the data say little about `η_j` the draws
//! are prior noise, the data term *falls* as the scale of `η_j` grows, and a scale below 1
//! contracts `ω_jj` in one step instead of at the missing-information rate (→ 1 as
//! `ω_jj → 0`).
//!
//! **Fixed point.** At the maximum likelihood estimate `E[η_l ∂_j ℓ(η)]` equals the prior's
//! own term minus `δ_jl` (Stein's identity on the conditional), so the derivative of the
//! data term in every entry of `A` vanishes in expectation and the stationary points are
//! those of the unexpanded algorithm: the *same* likelihood is maximised. `A = I`
//! reproduces the plain update bit for bit.
//!
//! `A` is one damped Newton step of `Σ_i ℓ_i(A η_i)` from `A = I`, on the exact gradient and
//! Hessian of the individual objective with the prior's own `η`-term removed, backtracked on
//! the objective itself.

use nalgebra::{DMatrix, DVector};

use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, Population, Subject};

/// Bounds of a diagonal entry of `A` (a step cannot more than quadruple or quarter a scale).
const DIAG_MIN: f64 = 0.25;
const DIAG_MAX: f64 = 4.0;
/// Bound of an off-diagonal entry of `A`.
const OFF_MAX: f64 = 2.0;
/// Halvings of the Newton step before it is refused (`A = I`).
const BACKTRACKS: usize = 6;

/// The free entries of the lower-triangular `A` and their bounds.
pub(crate) struct PxLayout {
    n: usize,
    idx: Vec<(usize, usize)>,
}

impl PxLayout {
    /// Diagonal `A`: variances rescale, correlations are untouched. Test oracle only: the run
    /// uses [`PxLayout::block`], which equals this on a diagonal Ω.
    #[cfg(test)]
    pub(crate) fn diagonal(n: usize) -> Self {
        Self {
            n,
            idx: (0..n).map(|j| (j, j)).collect(),
        }
    }

    /// Lower-triangular `A` with a free `(j, l)` entry wherever Ω has a free covariance.
    pub(crate) fn block(omega: &OmegaMatrix) -> Self {
        let n = omega.dim();
        let mut idx = Vec::new();
        for j in 0..n {
            for l in 0..=j {
                if j == l || (!omega.diagonal && omega.free_mask[(j, l)]) {
                    idx.push((j, l));
                }
            }
        }
        Self { n, idx }
    }

    pub(crate) fn len(&self) -> usize {
        self.idx.len()
    }

    /// The parameter vector of `A = I`.
    pub(crate) fn base(&self) -> Vec<f64> {
        self.idx
            .iter()
            .map(|&(j, l)| if j == l { 1.0 } else { 0.0 })
            .collect()
    }

    pub(crate) fn bounds(&self) -> Vec<(f64, f64)> {
        self.idx
            .iter()
            .map(|&(j, l)| {
                if j == l {
                    (DIAG_MIN, DIAG_MAX)
                } else {
                    (-OFF_MAX, OFF_MAX)
                }
            })
            .collect()
    }

    /// `A` for the parameter vector `p` (identity outside the free entries).
    pub(crate) fn matrix(&self, p: &[f64]) -> DMatrix<f64> {
        let mut a = DMatrix::identity(self.n, self.n);
        for (&(j, l), &v) in self.idx.iter().zip(p) {
            a[(j, l)] = v;
        }
        a
    }
}

/// A subject's individual objective over the stacked `[η, κ…]` vector (just `η`
/// without IOV): value and exact gradient at fixed `(θ, Ω, σ)`.
pub(crate) struct StackedObjective<'a> {
    model: &'a CompiledModel,
    subject: &'a Subject,
    params: &'a ModelParameters,
    kappas: Option<(&'a [Vec<f64>], &'a OmegaMatrix)>,
    n_eta: usize,
    k_lens: Vec<usize>,
    mult: Option<Vec<Vec<f64>>>,
    pub width: usize,
}

impl<'a> StackedObjective<'a> {
    pub(crate) fn new(
        model: &'a CompiledModel,
        subject: &'a Subject,
        params: &'a ModelParameters,
        kappas: Option<(&'a [Vec<f64>], &'a OmegaMatrix)>,
        n_eta: usize,
    ) -> Self {
        let k_lens: Vec<usize> = kappas
            .map(|(k, _)| k.iter().map(|v| v.len()).collect())
            .unwrap_or_default();
        let width = n_eta + k_lens.iter().sum::<usize>();
        let mult = if kappas.is_some() {
            model.ruv_obs_mult(subject, &params.theta)
        } else {
            None
        };
        Self {
            model,
            subject,
            params,
            kappas,
            n_eta,
            k_lens,
            mult,
            width,
        }
    }

    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        let (theta, omega, sigma) = (
            &self.params.theta,
            &self.params.omega,
            &self.params.sigma.values,
        );
        match self.kappas {
            Some((_, omega_iov)) => {
                let mut off = self.n_eta;
                let ks: Vec<Vec<f64>> = self
                    .k_lens
                    .iter()
                    .map(|&l| {
                        let v = x[off..off + l].to_vec();
                        off += l;
                        v
                    })
                    .collect();
                crate::stats::likelihood::individual_nll_iov(
                    self.model,
                    self.subject,
                    theta,
                    &x[..self.n_eta],
                    &ks,
                    omega,
                    Some(omega_iov),
                    sigma,
                )
            }
            None => crate::stats::likelihood::individual_nll(
                self.model,
                self.subject,
                theta,
                x,
                omega,
                sigma,
            ),
        }
    }

    pub(crate) fn grad(&self, x: &[f64]) -> Option<Vec<f64>> {
        let (theta, omega, sigma) = (
            &self.params.theta,
            &self.params.omega,
            &self.params.sigma.values,
        );
        let g = match self.kappas {
            Some((_, omega_iov)) => {
                crate::estimation::inner_optimizer::analytic_eta_nll_gradient_iov(
                    self.model,
                    self.subject,
                    theta,
                    x,
                    omega,
                    omega_iov,
                    sigma,
                    self.mult.as_deref(),
                )?
            }
            None => crate::estimation::inner_optimizer::analytic_eta_nll_gradient(
                self.model,
                self.subject,
                theta,
                x,
                omega,
                sigma,
            )?,
        };
        (g.len() == self.width && g.iter().all(|v| v.is_finite())).then_some(g)
    }
}

/// One damped Newton step on `F(p)` from `p = base`.
///
/// `grad = ∂F/∂p|base` and `hess = ∂²F/∂p²|base`; `objective(p)` evaluates `F`. The Hessian is
/// eigen-floored at a thousandth of its largest eigenvalue so an indefinite direction (the
/// data term is not convex away from the optimum) takes a short step; every entry is clamped to
/// `bounds`. Returns `base` unless a backtracked step lowers `F`.
pub(crate) fn px_newton(
    grad: &[f64],
    hess: &DMatrix<f64>,
    base: &[f64],
    bounds: &[(f64, f64)],
    mut objective: impl FnMut(&[f64]) -> f64,
) -> Vec<f64> {
    let n = grad.len();
    if n == 0 || !grad.iter().all(|v| v.is_finite()) || !hess.iter().all(|v| v.is_finite()) {
        return base.to_vec();
    }
    let sym = (hess + hess.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    let lmax = eig.eigenvalues.iter().cloned().fold(0.0_f64, f64::max); // nan-ok: guarded finite above
    if lmax <= 0.0 {
        return base.to_vec();
    }
    let floor = 1e-3 * lmax;
    let inv = eig.eigenvalues.map(|v| 1.0 / v.max(floor));
    let g = DVector::from_column_slice(grad);
    let step = &eig.eigenvectors * DMatrix::from_diagonal(&inv) * eig.eigenvectors.transpose() * &g;
    let f0 = objective(base);
    if !f0.is_finite() {
        return base.to_vec();
    }
    let mut t = 1.0_f64;
    for _ in 0..BACKTRACKS {
        let p: Vec<f64> = (0..n)
            .map(|j| (base[j] - t * step[j]).clamp(bounds[j].0, bounds[j].1))
            .collect();
        let f = objective(&p);
        if f.is_finite() && f < f0 {
            return p;
        }
        t *= 0.5;
    }
    base.to_vec()
}

/// Subject `i`'s occasion effects as the IOV objective wants them (`None` without IOV).
fn kappa_arg<'a>(
    omega_iov: Option<&'a OmegaMatrix>,
    kappas: &'a [Vec<Vec<f64>>],
    i: usize,
) -> Option<(&'a [Vec<f64>], &'a OmegaMatrix)> {
    match (omega_iov, kappas.get(i)) {
        (Some(o), Some(k)) if !k.is_empty() => Some((k.as_slice(), o)),
        _ => None,
    }
}

/// `∂F/∂p` and `∂²F/∂p²` at the matrix `a` for `F(p) = Σ_i nll_data_i(A(p) η_i)`, the individual
/// objective with the prior's `η`-term removed (`∂η_j/∂A_jl = η_l`). `None` when the exact
/// gradient or Hessian is unavailable for some subject.
pub(crate) fn px_terms_at(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
    layout: &PxLayout,
    a: &DMatrix<f64>,
) -> Option<(Vec<f64>, DMatrix<f64>)> {
    use crate::sens::ode_provider::adjoint::exact_inner_hessian;
    let omega: &OmegaMatrix = &params.omega;
    let omega_iov = params.omega_iov.as_ref();
    let n_eta = layout.n;
    let np = layout.len();
    let terms: Vec<Option<(Vec<f64>, DMatrix<f64>)>> =
        crate::par::map(population.subjects.len(), |i| {
            let subject = &population.subjects[i];
            let kap = kappa_arg(omega_iov, kappas, i);
            let st = StackedObjective::new(model, subject, params, kap, n_eta);
            let eta = &etas[i];
            let scaled: Vec<f64> = (a * DVector::from_column_slice(eta))
                .iter()
                .copied()
                .collect();
            let mut x = scaled.clone();
            if let Some((k, _)) = kap {
                x.extend(k.iter().flatten().copied());
            }
            let g = st.grad(&x)?;
            let h = exact_inner_hessian(model, subject, params, &x)?;
            let om_eta = &omega.inv * DVector::from_column_slice(&scaled);
            let gd: Vec<f64> = (0..n_eta).map(|j| g[j] - om_eta[j]).collect();
            let ga: Vec<f64> = layout.idx.iter().map(|&(j, l)| gd[j] * eta[l]).collect();
            let mut ha = DMatrix::zeros(np, np);
            for (p, &(j, l)) in layout.idx.iter().enumerate() {
                for (q, &(j2, l2)) in layout.idx.iter().enumerate() {
                    ha[(p, q)] = eta[l] * eta[l2] * (h[(j, j2)] - omega.inv[(j, j2)]);
                }
            }
            Some((ga, ha))
        });
    let mut grad = vec![0.0; np];
    let mut hess = DMatrix::zeros(np, np);
    for t in &terms {
        let (g, h) = t.as_ref()?;
        for p in 0..np {
            grad[p] += g[p];
        }
        hess += h;
    }
    Some((grad, hess))
}

/// `F(A) = Σ_i nll_data_i(A η_i)` (see [`px_terms_at`]).
pub(crate) fn px_objective(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
    a: &DMatrix<f64>,
) -> f64 {
    let omega: &OmegaMatrix = &params.omega;
    let omega_iov = params.omega_iov.as_ref();
    let n_eta = a.nrows();
    let parts: Vec<f64> = crate::par::map(population.subjects.len(), |i| {
        let subject = &population.subjects[i];
        let kap = kappa_arg(omega_iov, kappas, i);
        let st = StackedObjective::new(model, subject, params, kap, n_eta);
        let scaled: Vec<f64> = (a * DVector::from_column_slice(&etas[i]))
            .iter()
            .copied()
            .collect();
        let mut x = scaled.clone();
        if let Some((k, _)) = kap {
            x.extend(k.iter().flatten().copied());
        }
        let v = DVector::from_column_slice(&scaled);
        st.value(&x) - 0.5 * (v.transpose() * &omega.inv * &v)[(0, 0)]
    });
    parts.iter().sum()
}

/// The expansion matrix `A` for the current draws (`A = I` when nothing lowers the data term),
/// or `None` when the exact gradient/Hessian is unavailable for some subject (the caller then
/// takes `A = I`).
///
/// `kappas[i]` are subject `i`'s current occasion effects (held fixed; empty without IOV).
pub(crate) fn px_matrix(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
    layout: &PxLayout,
) -> Option<DMatrix<f64>> {
    let n = layout.n;
    let ident = DMatrix::identity(n, n);
    let (grad, hess) = px_terms_at(model, population, params, etas, kappas, layout, &ident)?;
    let p = px_newton(&grad, &hess, &layout.base(), &layout.bounds(), |p| {
        px_objective(model, population, params, etas, kappas, &layout.matrix(p))
    });
    Some(layout.matrix(&p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadratic data term `Σ_i ½ (y_i − α η_i)² / s²` has the closed-form optimum
    /// `α* = Σ y η / Σ η²`; one Newton step lands on it exactly (the quadratic is its own
    /// model), and at the optimum the step is the identity.
    #[test]
    fn px_newton_hits_the_closed_form_optimum_of_a_quadratic() {
        let y = [1.0, -0.4, 2.2, 0.7];
        let eta = [0.5, -0.9, 1.1, 0.2];
        let f = |a: &[f64]| -> f64 {
            y.iter()
                .zip(&eta)
                .map(|(y, e)| 0.5 * (y - a[0] * e).powi(2))
                .sum()
        };
        let grad = [y.iter().zip(&eta).map(|(y, e)| -e * (y - e)).sum::<f64>()];
        let hess = DMatrix::from_element(1, 1, eta.iter().map(|e| e * e).sum::<f64>());
        let star: f64 = y.iter().zip(&eta).map(|(y, e)| y * e).sum::<f64>()
            / eta.iter().map(|e| e * e).sum::<f64>();
        let bounds = [(0.25, 4.0)];
        let a = px_newton(&grad, &hess, &[1.0], &bounds, f);
        assert!((a[0] - star).abs() < 1e-12, "{} vs {star}", a[0]);
        // At the optimum (gradient zero) the step is the identity: A = I exactly, so the
        // expanded statistic A S Aᵀ is S bit for bit.
        let at_opt = |a: &[f64]| -> f64 {
            y.iter()
                .zip(&eta)
                .map(|(y, e)| 0.5 * (y - star * a[0] * e).powi(2))
                .sum()
        };
        assert_eq!(px_newton(&[0.0], &hess, &[1.0], &bounds, at_opt), vec![1.0]);
    }

    /// A step that does not lower the objective is refused (`A = I`), an indefinite Hessian
    /// is declined even when a step would lower the objective, and non-finite input is refused.
    #[test]
    fn px_newton_refuses_a_step_that_raises_the_objective() {
        let b = [(0.25, 4.0)];
        let hess = DMatrix::from_element(1, 1, 1.0);
        // The objective rises along every alpha != 1.
        let f = |a: &[f64]| (a[0] - 1.0).abs() + 1.0 + if a[0] == 1.0 { 0.0 } else { 0.1 };
        assert_eq!(px_newton(&[0.5], &hess, &[1.0], &b, f), vec![1.0]);
        // Indefinite: a smooth objective with its minimum at 0.5 would accept any step toward
        // it; the indefinite curvature must still decline.
        let neg = DMatrix::from_element(1, 1, -2.0);
        let down = |a: &[f64]| (a[0] - 0.5).powi(2);
        assert_eq!(px_newton(&[0.5], &neg, &[1.0], &b, down), vec![1.0]);
        assert_eq!(
            px_newton(&[f64::NAN], &hess, &[1.0], &b, |_| 1.0),
            vec![1.0]
        );
        // The step direction: a positive gradient lowers alpha.
        let r = px_newton(&[1.0], &hess, &[1.0], &b, down);
        assert!(r[0] < 1.0, "{r:?}");
        // The full step (to alpha = 1 - 1.6 -> clamped to 0.25) raises the objective; only a
        // halved one lowers it, so a single try would refuse and the backtracking must find it.
        let g = |a: &[f64]| (a[0] - 0.9).powi(2);
        let r = px_newton(&[1.6], &hess, &[1.0], &b, g);
        assert!(r[0] > 0.25 && r[0] < 1.0, "{r:?}");
    }

    // ---- model-level tests on the warfarin example ------------------------------------

    use crate::parser::model_parser::parse_model_file;
    use crate::read_nonmem_csv;
    use std::path::Path;

    fn warfarin() -> (CompiledModel, Population) {
        let model = parse_model_file(Path::new("examples/warfarin.ferx")).expect("model");
        let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data");
        (model, pop)
    }

    /// Conditional modes of the first `k` subjects at the default parameters.
    fn modes(model: &CompiledModel, pop: &Population, k: usize) -> (Population, Vec<Vec<f64>>) {
        let p = &model.default_params;
        let sub: Vec<Subject> = pop.subjects.iter().take(k).cloned().collect();
        let etas: Vec<Vec<f64>> = sub
            .iter()
            .map(|s| {
                let r = crate::estimation::inner_optimizer::find_ebe(
                    model, s, p, 200, 1e-10, None, None, 0,
                );
                r.eta.iter().copied().collect()
            })
            .collect();
        (
            Population {
                subjects: sub,
                covariate_names: vec![],
                dv_column: pop.dv_column.clone(),
                input_columns: vec![],
                exclusions: None,
                warnings: vec![],
            },
            etas,
        )
    }

    /// The assembled PX gradient and Hessian are the derivatives of the objective they claim to
    /// be, for the diagonal layout AND a full lower-triangular one: central finite differences
    /// of `F(A)` for the gradient, and of the gradient re-evaluated at the moved `A` for the
    /// Hessian. This is the dual-vs-FD parity check for the new derivative path; it sees a sign
    /// error, a missing `η_l η_l'` factor, or the prior's own term left in the data part. The
    /// draws are the modes displaced by a deterministic spread, so the data term has both signs
    /// of slope.
    #[test]
    fn px_gradient_and_hessian_match_finite_differences() {
        let (model, pop) = warfarin();
        let n_eta = model.n_eta;
        let p = &model.default_params;
        let (pop, mut etas) = modes(&model, &pop, 6);
        for (i, e) in etas.iter_mut().enumerate() {
            for (j, x) in e.iter_mut().enumerate() {
                *x += 0.02 * (((i * 3 + j) % 5) as f64 - 2.0);
            }
        }
        let kappas = vec![Vec::new(); etas.len()];
        let full = PxLayout {
            n: n_eta,
            idx: (0..n_eta)
                .flat_map(|j| (0..=j).map(move |l| (j, l)))
                .collect(),
        };
        for (name, layout) in [("diagonal", PxLayout::diagonal(n_eta)), ("full", full)] {
            let ident = DMatrix::identity(n_eta, n_eta);
            let (g, h) =
                px_terms_at(&model, &pop, p, &etas, &kappas, &layout, &ident).expect("terms");
            let base = layout.base();
            let f = |q: &[f64]| px_objective(&model, &pop, p, &etas, &kappas, &layout.matrix(q));
            let step = 1e-5;
            let mut worst_g = 0.0_f64;
            let mut worst_h = 0.0_f64;
            for a in 0..layout.len() {
                let mut qp = base.clone();
                let mut qm = base.clone();
                qp[a] += step;
                qm[a] -= step;
                let fd = (f(&qp) - f(&qm)) / (2.0 * step);
                assert!(fd.is_finite() && g[a].is_finite());
                worst_g = worst_g.max((fd - g[a]).abs() / (1.0 + g[a].abs())); // nan-ok: asserted finite above
                for b in 0..layout.len() {
                    let grad_at = |q: &[f64]| -> f64 {
                        let (gg, _) = px_terms_at(
                            &model,
                            &pop,
                            p,
                            &etas,
                            &kappas,
                            &layout,
                            &layout.matrix(q),
                        )
                        .unwrap();
                        gg[a]
                    };
                    let mut bp = base.clone();
                    let mut bm = base.clone();
                    bp[b] += step;
                    bm[b] -= step;
                    let fdh = (grad_at(&bp) - grad_at(&bm)) / (2.0 * step);
                    assert!(fdh.is_finite() && h[(a, b)].is_finite());
                    worst_h = worst_h.max((fdh - h[(a, b)]).abs() / (1.0 + h[(a, b)].abs()));
                    // nan-ok: asserted finite above
                }
            }
            eprintln!("px FD parity ({name}): gradient {worst_g:e}, Hessian {worst_h:e}");
            assert!(worst_g < 1e-5, "{name}: gradient parity {worst_g:e}");
            assert!(worst_h < 1e-4, "{name}: Hessian parity {worst_h:e}");
        }
    }

    /// At the conditional modes the draws carry no spread, only the prior's shrinkage, which
    /// the data term would undo: every diagonal scale is above 1 and the objective falls.
    /// Regression it catches: the prior's own `η`-term left in the data part (that would make
    /// `A` chase the prior and sit at 1 here), or a step in the wrong direction.
    #[test]
    fn px_scale_expands_shrunken_modes_and_lowers_the_objective() {
        let (model, pop) = warfarin();
        let n_eta = model.n_eta;
        let p = &model.default_params;
        let (pop, etas) = modes(&model, &pop, usize::MAX);
        let kappas = vec![Vec::new(); etas.len()];
        let layout = PxLayout::diagonal(n_eta);
        let a = px_matrix(&model, &pop, p, &etas, &kappas, &layout).expect("matrix");
        eprintln!("A at modes {a}");
        assert!((0..n_eta).all(|j| a[(j, j)] > 1.0), "{a}");
        let ident = DMatrix::identity(n_eta, n_eta);
        assert!(
            px_objective(&model, &pop, p, &etas, &kappas, &a)
                < px_objective(&model, &pop, p, &etas, &kappas, &ident)
        );
    }

    /// The block layout follows Ω's free pattern: a diagonal Ω gets a diagonal `A`, a block Ω
    /// gets the lower triangle of its block only. Regression it catches: an `A` entry across
    /// blocks (or on a diagonal Ω) would push `A S Aᵀ` off Ω's structure.
    #[test]
    fn block_layout_follows_the_omega_pattern() {
        let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let diag = OmegaMatrix::from_diagonal(&[0.1, 0.2, 0.3], names.clone());
        assert_eq!(PxLayout::block(&diag).idx, vec![(0, 0), (1, 1), (2, 2)]);
        let m = DMatrix::from_row_slice(3, 3, &[0.4, 0.1, 0.0, 0.1, 0.3, 0.0, 0.0, 0.0, 0.2]);
        let blk = OmegaMatrix::from_matrix(m, names, false);
        assert_eq!(
            PxLayout::block(&blk).idx,
            vec![(0, 0), (1, 0), (1, 1), (2, 2)]
        );
        // A = I at the base point, whatever the layout.
        let l = PxLayout::block(&blk);
        assert_eq!(l.matrix(&l.base()), DMatrix::identity(3, 3));
    }
}
