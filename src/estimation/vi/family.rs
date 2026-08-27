//! Variational families: the tractable `q_φ(η)` that VI fits to each subject's
//! random-effect posterior.
//!
//! This trait is the extension point of the VI implementation. Everything above
//! it — the ELBO assembly, the optimizer, the reporting — is family-agnostic, so
//! a richer posterior (a Gaussian mixture, a normalizing flow, or a joint
//! posterior over the *population* parameters as in Janssen et al.'s `full_advi`)
//! arrives as a new `impl`, not a new estimator.
//!
//! # Reparameterization
//!
//! Every family samples as `η = T_φ(ε)` for a standard-normal `ε` fixed
//! independently of `φ`. That is what makes `∂η/∂φ` exist and the ELBO gradient
//! low-variance: the randomness sits in `ε`, which carries no gradient, so
//! differentiating the sampled `η` is just differentiating `T_φ`.
//!
//! [`VariationalFamily::sample`] applies `T_φ`; [`VariationalFamily::chain_to_phi`]
//! applies its transposed Jacobian to an incoming `∂L/∂η`.
//!
//! # Positivity
//!
//! Scale parameters are stored on the **log** scale (`MeanField`'s `log σ`,
//! `FullRank`'s Cholesky diagonal), so `φ` is unconstrained and the implied
//! covariance is positive-definite by construction. Adam can then step `φ`
//! freely without projection. This differs from Janssen et al., who use
//! `softplus`; `log` is chosen here to match the Cholesky-Ω convention already
//! used by [`crate::estimation::parameterization`], so `Ω` and `S` live in the
//! same transform.

use nalgebra::{DMatrix, DVector};

use crate::types::OmegaMatrix;

/// `KL(q_φ ‖ N(0, Ω))` and its derivatives, when the family has it in closed form.
///
/// Having this analytically is what keeps the default VI path low-variance: the
/// ELBO splits as `E_q[log p(y|η)] − KL(q ‖ N(0,Ω))`, and only the first term
/// then needs Monte Carlo. See the module docs of [`super`].
#[derive(Debug, Clone)]
pub struct KlTerm {
    /// The divergence itself (nats).
    pub value: f64,
    /// `∂KL/∂φ`, in the family's own `φ` layout.
    pub d_phi: Vec<f64>,
    /// `∂KL/∂Ω`, as a dense symmetric matrix — deliberately *not* in any packed
    /// optimizer coordinates. The family does not know how `Ω` is parameterized;
    /// the caller chains this into whatever coordinates it optimizes (or ignores
    /// it entirely and uses the closed-form `Ω` update instead).
    pub d_omega: DMatrix<f64>,
}

/// A tractable approximate posterior over one subject's random effects.
///
/// Implementations own their dimension (`n_eta`) so callers never have to keep
/// passing it, and are `Send + Sync` so subjects can be processed in parallel.
pub trait VariationalFamily: Send + Sync {
    /// Dimension of `η`.
    fn n_eta(&self) -> usize;

    /// Number of unconstrained variational parameters per subject.
    fn n_params(&self) -> usize;

    /// Human-readable name, for reporting.
    fn label(&self) -> &'static str;

    /// Initial `φ`: mean zero and covariance `Ω`, i.e. `q` starts *at the prior*.
    ///
    /// Starting at the prior rather than at a random point makes a fit
    /// reproducible by default and gives the first iteration a sane ELBO (the KL
    /// term is exactly zero).
    fn init(&self, omega: &OmegaMatrix) -> Vec<f64>;

    /// The reparameterized draw `η = T_φ(ε)`. `eps` has length `n_eta()`.
    fn sample(&self, phi: &[f64], eps: &[f64]) -> Vec<f64>;

    /// Accumulate `(∂η/∂φ)ᵀ · g_eta` into `out`.
    ///
    /// **Accumulates** (`+=`) rather than overwriting, so a caller averaging over
    /// several Monte-Carlo draws can call it once per draw.
    fn chain_to_phi(&self, phi: &[f64], eps: &[f64], g_eta: &[f64], out: &mut [f64]);

    /// Closed-form `KL(q_φ ‖ N(0, Ω))`, or `None` when this family has none.
    ///
    /// `None` is a supported answer, not an error: the caller falls back to the
    /// Monte-Carlo KL path ([`super::elbo`]) and reports having done so. A family
    /// with an intractable KL — a mixture, a normalizing flow — therefore only has
    /// to implement [`Self::log_density`].
    fn kl_to_normal(&self, phi: &[f64], omega: &OmegaMatrix) -> Option<KlTerm>;

    /// `log q_φ(η)` and `∂ log q_φ(η) / ∂η`.
    ///
    /// Only the Monte-Carlo-KL path (`vi_kl = mc`, or a family with no closed form)
    /// needs this; the analytic-KL path never evaluates the variational density.
    fn log_density(&self, phi: &[f64], eta: &[f64]) -> (f64, Vec<f64>);

    /// Posterior mean and covariance implied by `φ`.
    ///
    /// The mean is what VI reports in place of an EBE; the covariance is the
    /// per-subject posterior uncertainty, which VI gets for free where
    /// FOCE/Laplace need a Hessian.
    fn moments(&self, phi: &[f64]) -> (DVector<f64>, DMatrix<f64>);
}

/// Smallest variance a family will initialize to, guarding `ln(0)` when a model
/// declares a degenerate `Ω` diagonal.
const MIN_INIT_VAR: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Mean-field (diagonal) Gaussian
// ---------------------------------------------------------------------------

/// Diagonal Gaussian `q(η) = N(μ, diag(exp(2·s)))`.
///
/// `φ = [μ (d) | s (d)]` where `s = log σ`. Costs `O(d)` per subject, so this is
/// the right choice once `n_eta` is large — at the price of not representing
/// posterior correlation between random effects, which biases the ELBO whenever
/// the true posterior is correlated (it usually is, e.g. CL/V).
#[derive(Debug, Clone, Copy)]
pub struct MeanField {
    n_eta: usize,
}

impl MeanField {
    pub fn new(n_eta: usize) -> Self {
        Self { n_eta }
    }

    /// `σ_k = exp(s_k)` for each coordinate.
    fn sds(&self, phi: &[f64]) -> Vec<f64> {
        (0..self.n_eta).map(|k| phi[self.n_eta + k].exp()).collect()
    }
}

impl VariationalFamily for MeanField {
    fn n_eta(&self) -> usize {
        self.n_eta
    }

    fn n_params(&self) -> usize {
        2 * self.n_eta
    }

    fn label(&self) -> &'static str {
        "mean_field"
    }

    fn init(&self, omega: &OmegaMatrix) -> Vec<f64> {
        let d = self.n_eta;
        let mut phi = vec![0.0; 2 * d];
        for k in 0..d {
            let var = omega.matrix[(k, k)].max(MIN_INIT_VAR);
            phi[d + k] = 0.5 * var.ln();
        }
        phi
    }

    fn sample(&self, phi: &[f64], eps: &[f64]) -> Vec<f64> {
        let d = self.n_eta;
        (0..d).map(|k| phi[k] + phi[d + k].exp() * eps[k]).collect()
    }

    fn chain_to_phi(&self, phi: &[f64], eps: &[f64], g_eta: &[f64], out: &mut [f64]) {
        let d = self.n_eta;
        for k in 0..d {
            // ∂η_k/∂μ_k = 1
            out[k] += g_eta[k];
            // ∂η_k/∂s_k = exp(s_k)·ε_k
            out[d + k] += g_eta[k] * phi[d + k].exp() * eps[k];
        }
    }

    fn kl_to_normal(&self, phi: &[f64], omega: &OmegaMatrix) -> Option<KlTerm> {
        let d = self.n_eta;
        let mu = DVector::from_iterator(d, phi[..d].iter().copied());
        let sds = self.sds(phi);
        let oinv = &omega.inv;

        // KL = ½[tr(Ω⁻¹S) + μᵀΩ⁻¹μ − d + log|Ω| − log|S|]
        let mut tr = 0.0;
        let mut log_det_s = 0.0;
        for k in 0..d {
            let s_kk = sds[k] * sds[k];
            tr += oinv[(k, k)] * s_kk;
            log_det_s += 2.0 * phi[d + k];
        }
        let oinv_mu = oinv * &mu;
        let quad = mu.dot(&oinv_mu);
        let value = 0.5 * (tr + quad - d as f64 + omega.log_det - log_det_s);

        let mut d_phi = vec![0.0; 2 * d];
        for k in 0..d {
            // ∂KL/∂μ = Ω⁻¹μ
            d_phi[k] = oinv_mu[k];
            // ∂KL/∂S_kk = ½(Ω⁻¹_kk − 1/S_kk), and ∂S_kk/∂s_k = 2·S_kk, so
            // ∂KL/∂s_k = S_kk·Ω⁻¹_kk − 1.
            let s_kk = sds[k] * sds[k];
            d_phi[d + k] = s_kk * oinv[(k, k)] - 1.0;
        }

        Some(KlTerm {
            value,
            d_phi,
            d_omega: d_kl_d_omega(oinv, &self.moments(phi).1, &mu),
        })
    }

    fn log_density(&self, phi: &[f64], eta: &[f64]) -> (f64, Vec<f64>) {
        let d = self.n_eta;
        let mut lq = -0.5 * (d as f64) * std::f64::consts::TAU.ln();
        let mut g = vec![0.0; d];
        for k in 0..d {
            let s = phi[d + k].exp();
            let z = (eta[k] - phi[k]) / s;
            lq += -0.5 * z * z - phi[d + k];
            g[k] = -z / s;
        }
        (lq, g)
    }

    fn moments(&self, phi: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        let d = self.n_eta;
        let mean = DVector::from_iterator(d, phi[..d].iter().copied());
        let mut cov = DMatrix::zeros(d, d);
        for k in 0..d {
            let s = phi[d + k].exp();
            cov[(k, k)] = s * s;
        }
        (mean, cov)
    }
}

// ---------------------------------------------------------------------------
// Full-rank Gaussian
// ---------------------------------------------------------------------------

/// Number of lower-triangular entries of a `d × d` matrix.
pub const fn n_tril(d: usize) -> usize {
    d * (d + 1) / 2
}

/// Row-major index of lower-triangular entry `(i, j)`, `j ≤ i`.
#[inline]
pub const fn tril_index(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// Full-rank Gaussian `q(η) = N(μ, LLᵀ)`.
///
/// `φ = [μ (d) | vech(L) (d(d+1)/2)]`, row-major lower triangle, with the
/// **diagonal stored as its logarithm** so `L` always has positive diagonal and
/// `S = LLᵀ` is positive-definite for any `φ`.
///
/// This is the family Janssen et al. use, and the default here: the CL/V
/// posterior correlation that `MeanField` cannot represent is exactly the sort
/// of structure that makes the mean-field ELBO a loose bound.
#[derive(Debug, Clone, Copy)]
pub struct FullRank {
    n_eta: usize,
}

impl FullRank {
    pub fn new(n_eta: usize) -> Self {
        Self { n_eta }
    }

    /// Materialize `L` from `φ`, exponentiating the stored diagonal.
    fn lower(&self, phi: &[f64]) -> DMatrix<f64> {
        let d = self.n_eta;
        let mut l = DMatrix::zeros(d, d);
        for i in 0..d {
            for j in 0..=i {
                let raw = phi[d + tril_index(i, j)];
                l[(i, j)] = if i == j { raw.exp() } else { raw };
            }
        }
        l
    }
}

impl VariationalFamily for FullRank {
    fn n_eta(&self) -> usize {
        self.n_eta
    }

    fn n_params(&self) -> usize {
        self.n_eta + n_tril(self.n_eta)
    }

    fn label(&self) -> &'static str {
        "full_rank"
    }

    fn init(&self, omega: &OmegaMatrix) -> Vec<f64> {
        let d = self.n_eta;
        let mut phi = vec![0.0; self.n_params()];
        // `omega.chol` is the lower Cholesky factor, so S = LLᵀ = Ω exactly.
        for i in 0..d {
            for j in 0..=i {
                let v = omega.chol[(i, j)];
                phi[d + tril_index(i, j)] = if i == j {
                    v.max(MIN_INIT_VAR.sqrt()).ln()
                } else {
                    v
                };
            }
        }
        phi
    }

    fn sample(&self, phi: &[f64], eps: &[f64]) -> Vec<f64> {
        let d = self.n_eta;
        let l = self.lower(phi);
        (0..d)
            .map(|i| {
                let mut acc = phi[i];
                for j in 0..=i {
                    acc += l[(i, j)] * eps[j];
                }
                acc
            })
            .collect()
    }

    fn chain_to_phi(&self, phi: &[f64], eps: &[f64], g_eta: &[f64], out: &mut [f64]) {
        let d = self.n_eta;
        for i in 0..d {
            // ∂η_i/∂μ_i = 1
            out[i] += g_eta[i];
            for j in 0..=i {
                // ∂η_i/∂L_ij = ε_j; the diagonal carries an extra factor L_ii
                // from d(exp(raw))/d(raw).
                let raw = phi[d + tril_index(i, j)];
                let dl_draw = if i == j { raw.exp() } else { 1.0 };
                out[d + tril_index(i, j)] += g_eta[i] * eps[j] * dl_draw;
            }
        }
    }

    fn kl_to_normal(&self, phi: &[f64], omega: &OmegaMatrix) -> Option<KlTerm> {
        let d = self.n_eta;
        let mu = DVector::from_iterator(d, phi[..d].iter().copied());
        let l = self.lower(phi);
        let s = &l * l.transpose();
        let oinv = &omega.inv;

        // log|S| = 2·Σ log L_ii — read straight off the stored log-diagonal.
        let log_det_s: f64 = 2.0 * (0..d).map(|i| phi[d + tril_index(i, i)]).sum::<f64>();
        let tr = (oinv * &s).trace();
        let oinv_mu = oinv * &mu;
        let quad = mu.dot(&oinv_mu);
        let value = 0.5 * (tr + quad - d as f64 + omega.log_det - log_det_s);

        // ∂KL/∂L = Ω⁻¹L − L⁻ᵀ. The second term is where −½log|S| lands: the
        // (i,i) entry of L⁻ᵀ is 1/L_ii, matching d(−Σ log L_ii)/dL_ii.
        let oinv_l = oinv * &l;
        let l_inv_t = invert_lower_triangular(&l)?.transpose();

        let mut d_phi = vec![0.0; self.n_params()];
        for i in 0..d {
            d_phi[i] = oinv_mu[i];
            for j in 0..=i {
                let dk_dl = oinv_l[(i, j)] - l_inv_t[(i, j)];
                let dl_draw = if i == j { l[(i, i)] } else { 1.0 };
                d_phi[d + tril_index(i, j)] = dk_dl * dl_draw;
            }
        }

        Some(KlTerm {
            value,
            d_phi,
            d_omega: d_kl_d_omega(oinv, &s, &mu),
        })
    }

    fn log_density(&self, phi: &[f64], eta: &[f64]) -> (f64, Vec<f64>) {
        let d = self.n_eta;
        let l = self.lower(phi);
        let diff = DVector::from_iterator(d, (0..d).map(|k| eta[k] - phi[k]));

        // z = L⁻¹·diff by forward substitution; then S⁻¹·diff = L⁻ᵀ·z by back
        // substitution. Cheaper and better-conditioned than forming S⁻¹.
        let mut z = vec![0.0; d];
        for i in 0..d {
            let mut acc = diff[i];
            for j in 0..i {
                acc -= l[(i, j)] * z[j];
            }
            z[i] = acc / l[(i, i)];
        }
        let mut sinv_diff = vec![0.0; d];
        for i in (0..d).rev() {
            let mut acc = z[i];
            for j in (i + 1)..d {
                acc -= l[(j, i)] * sinv_diff[j];
            }
            sinv_diff[i] = acc / l[(i, i)];
        }

        let quad: f64 = z.iter().map(|zi| zi * zi).sum();
        let log_det_s: f64 = 2.0 * (0..d).map(|i| phi[d + tril_index(i, i)]).sum::<f64>();
        let lq = -0.5 * (quad + log_det_s + (d as f64) * std::f64::consts::TAU.ln());
        let g = sinv_diff.iter().map(|v| -v).collect();
        (lq, g)
    }

    fn moments(&self, phi: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        let d = self.n_eta;
        let mean = DVector::from_iterator(d, phi[..d].iter().copied());
        let l = self.lower(phi);
        let cov = &l * l.transpose();
        (mean, cov)
    }
}

/// `∂KL/∂Ω = ½(Ω⁻¹ − Ω⁻¹SΩ⁻¹ − Ω⁻¹μμᵀΩ⁻¹)`, treating `Ω` as a free symmetric
/// matrix.
///
/// Shared by both families — the `Ω`-side derivative depends on `q` only through
/// its moments, so it is the same formula whatever the family.
fn d_kl_d_omega(oinv: &DMatrix<f64>, s: &DMatrix<f64>, mu: &DVector<f64>) -> DMatrix<f64> {
    let oinv_s_oinv = oinv * s * oinv;
    let oinv_mu = oinv * mu;
    let outer = &oinv_mu * oinv_mu.transpose();
    (oinv - oinv_s_oinv - outer) * 0.5
}

/// Invert a lower-triangular matrix by forward substitution.
///
/// `None` when any diagonal entry is zero — unreachable for a `φ`-built `L`
/// (the diagonal is `exp(·)`), but `Ω`-derived callers are not guaranteed that.
fn invert_lower_triangular(l: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let d = l.nrows();
    let mut inv = DMatrix::zeros(d, d);
    for col in 0..d {
        for i in col..d {
            let mut acc = if i == col { 1.0 } else { 0.0 };
            for j in col..i {
                acc -= l[(i, j)] * inv[(j, col)];
            }
            let lii = l[(i, i)];
            if lii == 0.0 || !lii.is_finite() {
                return None;
            }
            inv[(i, col)] = acc / lii;
        }
    }
    Some(inv)
}

#[cfg(test)]
#[path = "family_tests.rs"]
mod tests;
