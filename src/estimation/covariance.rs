//! Covariance / standard-error subsystem (moved verbatim from
//! `estimation::outer_optimizer` in refactor T4). The FD-of-OFV Hessian step,
//! the eigen-floor inverse, the score cross-product, the non-PD SIR fallback,
//! and the progress reporter all live here. `outer_optimizer` retains only the
//! population optimizers, the outer-gradient family, `OuterResult`, and
//! `pop_nll`/`pop_nll_opts` (imported below).

use crate::estimation::cov_diagnostics::{
    cov_objective_method, cov_objective_route, estimate_minimises_cov_objective,
    format_non_stationary_warning, format_offdiag_nan_warning, format_quadrature_declined_warning,
    format_quadrature_objective_note, format_regularized_warning, format_salvage_note,
    r_falls_back_to_s, CovHessianSource, CovObjectiveRoute, CovRegularizationFacts,
    CovScopeDecline, NonStationaryFacts, OdeToleranceFacts, S_RANK_DEFICIENT_REASON,
};
use crate::estimation::inner_optimizer::find_ebe_for_stage;
use crate::estimation::outer_optimizer::pop_nll_opts;
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::types::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;
use std::collections::HashSet;

#[cfg(test)]
thread_local! {
    /// Test-only: objectives evaluated by the covariance step's FD probe (`select_fd_step`)
    /// and stencil (`fd_ofv_stencil`), accumulated on the thread that called the step. The
    /// count is gathered across the rayon workers by an atomic local to each call and added
    /// here once the stencil returns, so concurrent tests cannot see each other's counts and
    /// no lock is involved. It is what pins that `covariance_method = s` builds no `R` (#1803).
    pub(crate) static FD_OBJECTIVE_EVALS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Outcome of the FD covariance step. `matrix` is the n×n covariance with FIX
/// rows/cols zeroed; `warnings` carries non-fatal notes (regularisation applied,
/// off-diagonal FD stencil failures, etc.). Empty when everything was clean.
pub(crate) struct CovarianceOutput {
    pub matrix: DMatrix<f64>,
    pub warnings: Vec<String>,
    /// The estimator `matrix` actually is. The requested one, except where #1781 replaced an
    /// `R⁻¹` / `R⁻¹SR⁻¹` that would have inverted a floored `R` at an estimate the objective
    /// never optimised by `S⁻¹` ([`crate::estimation::cov_diagnostics::r_falls_back_to_s`]).
    pub method: CovarianceMethod,
}

/// Return type of [`compute_covariance`].
pub(crate) enum CovarianceStepResult {
    /// Covariance computed (possibly with non-fatal warnings).
    Success(CovarianceOutput),
    /// Structurally unusable. Carries a complete user-facing warning message
    /// (already ends with "SE estimates not available.").
    Unusable(String),
    /// FD Hessian symmetrised free-block has no positive eigenvalues — cannot
    /// be inverted. Carries the warning message and a ready-to-use fallback
    /// proposal covariance (full packed space, zeros for FIX params) built
    /// from `|eigenvalue|`-rectified Hessian, inflated 4×.
    FailedNonPd {
        reason: String,
        fallback_proposal: DMatrix<f64>,
    },
}

/// The Omega matrix to inspect when diagnosing a non-finite covariance base OFV.
///
/// For a mixture (#984 review) the base (class-1) Omega can be well-conditioned
/// while a per-class `omega(k)` override has collapsed — the actual cause of the
/// non-finite OFV. Return the worst-conditioned Omega (smallest minimum
/// eigenvalue) across all classes, so the emitted reason names Omega collapse
/// rather than misattributing it to a model-evaluation overflow. Non-mixture
/// models return the single base Omega unchanged.
fn diagnostic_omega(params_at: &ModelParameters) -> &DMatrix<f64> {
    params_at
        .mixture
        .as_ref()
        .and_then(|mp| {
            mp.omega.iter().map(|o| &o.matrix).min_by(|a, b| {
                let min_eig = |m| {
                    extract_eigenvalues(m)
                        .and_then(|ev| ev.last().copied())
                        .unwrap_or(f64::INFINITY)
                };
                min_eig(a)
                    .partial_cmp(&min_eig(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .unwrap_or(&params_at.omega.matrix)
}

/// Human-readable label for the packed parameter at position `packed_idx`.
/// E.g. `"theta[CL]"`, `"omega[ETA_1, ETA_2]"`, `"sigma[1]"`.
///
/// Uses names from `template` directly (`theta_names`, `omega.eta_names`) rather
/// than from the `CompiledModel`, so the label is correct even when a test
/// constructs a `ModelParameters` whose dimensions differ from the test model.
pub(crate) fn packed_param_label(packed_idx: usize, template: &ModelParameters) -> String {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_omega = omega_packed_len(n_eta, template.omega.diagonal);
    let n_sigma = template.sigma.values.len();
    let n_iov = template
        .omega_iov
        .as_ref()
        .map_or(0, |m| omega_packed_len(m.dim(), m.diagonal));

    if packed_idx < n_theta {
        let name = template
            .theta_names
            .get(packed_idx)
            .map(String::as_str)
            .unwrap_or("?");
        format!("theta[{}]", name)
    } else if packed_idx < n_theta + n_omega {
        let omega_idx = packed_idx - n_theta;
        // Decode a packed Ω index back to (row, col) via the centralized packing
        // order (single source: `lower_tri_entries`; `omega_idx < n_omega` here, so
        // the index is always in range). Cold path (labelling), so the Vec is fine.
        let (row, col) =
            crate::estimation::parameterization::lower_tri_entries(n_eta, template.omega.diagonal)
                [omega_idx];
        let nr = template
            .omega
            .eta_names
            .get(row)
            .map(String::as_str)
            .unwrap_or("?");
        let nc = template
            .omega
            .eta_names
            .get(col)
            .map(String::as_str)
            .unwrap_or("?");
        format!("omega[{}, {}]", nr, nc)
    } else if packed_idx < n_theta + n_omega + n_sigma {
        let idx = packed_idx - n_theta - n_omega + 1;
        format!("sigma[{}]", idx)
    } else if packed_idx < n_theta + n_omega + n_sigma + n_iov {
        let idx = packed_idx - n_theta - n_omega - n_sigma + 1;
        format!("kappa[{}]", idx)
    } else if let Some(mix) = template.mixture.as_ref().filter(|_| {
        packed_idx < n_theta + n_omega + n_sigma + n_iov + mixture_override_len(template)
    }) {
        // Mixture per-class Ω/Σ override segment (#983): appended after kappa in
        // pack order — Ω overrides first, then Σ, each `(class, eta|sigma idx)`.
        // Label as `omega[<base>_MIX{class}]` / `sigma[<base>_MIX{class}]`, the
        // same names `coordinate_names` emits, instead of the `packed[N]` fallback.
        // The mixing-logit coefficients are ordinary thetas and are already
        // labelled by the theta branch above.
        let ov = packed_idx - (n_theta + n_omega + n_sigma + n_iov);
        let n_omega_ov = mix.omega_override_addr.len();
        if ov < n_omega_ov {
            let (c, e) = mix.omega_override_addr[ov];
            let base = template
                .omega
                .eta_names
                .get(e)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("OMEGA({},{})", e + 1, e + 1));
            format!("omega[{}_MIX{}]", base, c + 1)
        } else {
            let (c, s) = mix.sigma_override_addr[ov - n_omega_ov];
            let base = template
                .sigma
                .names
                .get(s)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("SIGMA({})", s + 1));
            format!("sigma[{}_MIX{}]", base, c + 1)
        }
    } else {
        format!("packed[{}]", packed_idx)
    }
}

/// Number of packed mixture-override coordinates (Ω overrides + Σ overrides),
/// or 0 for a non-mixture `template`. Mirrors the pack-order tail appended by
/// `pack_params`.
fn mixture_override_len(template: &ModelParameters) -> usize {
    template.mixture.as_ref().map_or(0, |m| {
        m.omega_override_addr.len() + m.sigma_override_addr.len()
    })
}

/// Format a single eigenvalue for display: `"0"`, fixed-4, or scientific-3.
///
/// The exact-zero branch handles rank-deficient inputs (e.g. a parameter block
/// that is entirely FIX) where `SymmetricEigen` returns eigenvalue `0.0` exactly.
/// Any non-zero value — even 1e-300 — uses fixed or scientific notation instead.
pub(crate) fn fmt_eig(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs >= 1e-4 && abs < 1e5 {
        format!("{:.4}", v)
    } else {
        format!("{:.3e}", v)
    }
}

/// Eigenvalues of `sym` sorted descending. Returns `None` if any eigenvalue is non-finite,
/// or if `sym` is empty.
///
/// The empty case is reachable, not defensive: a model with **no random effects** has a
/// 0×0 Omega, and the non-finite-objective diagnostic below asks this function to inspect
/// it precisely when the fit has already gone wrong. `nalgebra`'s `SymmetricEigen::new`
/// panics outright on a 0×0 input ("Unable to compute the symmetric tridiagonal
/// decomposition of an empty matrix"), so an `n_eta = 0` fit whose objective went
/// non-finite aborted with that message instead of reporting the objective — the diagnostic
/// killed the run it was there to explain. `None` is the existing "cannot diagnose" signal
/// and every caller already handles it.
///
/// One predicate, not `nrows() == 0 || ncols() == 0`: those two reject exactly the same
/// inputs here, so either could be deleted with the guard still passing its own test —
/// the redundant-gate hole from #1229 / #1255.
pub(crate) fn extract_eigenvalues(sym: &DMatrix<f64>) -> Option<Vec<f64>> {
    if sym.is_empty() {
        return None;
    }
    let eig = SymmetricEigen::new(sym.clone());
    if eig.eigenvalues.iter().any(|l| !l.is_finite()) {
        return None;
    }
    let mut eigvals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    eigvals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Some(eigvals)
}

/// Format a diagnostic warning for a non-positive-definite covariance Hessian.
pub(crate) fn format_non_pd_warning(eigvals: &[f64]) -> String {
    let fmt = eigvals
        .iter()
        .map(|&v| fmt_eig(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Covariance step: Hessian is not positive definite. \
         Eigenvalues: [{}]. SE estimates not available.",
        fmt
    )
}

/// Largest condition number permitted for the non-PD fallback proposal. The
/// eigenvalue magnitudes are floored at `λ_max_abs / COND` so a near-zero
/// curvature direction can't blow its proposal variance up without bound (see
/// [`build_non_pd_fallback_proposal`]).
pub(crate) const FALLBACK_PROPOSAL_MAX_COND: f64 = 1e8;

/// Build a SIR proposal covariance for the non-PD-Hessian fallback path.
///
/// This is the standard eigenvalue-modification heuristic: the symmetrised
/// free-block Hessian has at least one non-positive eigenvalue, so it cannot be
/// inverted into a covariance directly. We take each eigenvalue's *magnitude*
/// `|λ_i|` as the curvature in that direction, and use `inflation / |λ_i|` as the
/// corresponding proposal variance (`inflation`× wider than the inverted
/// absolute Hessian).
///
/// The magnitudes are floored **relative to the largest** at
/// `|λ|_max / FALLBACK_PROPOSAL_MAX_COND` rather than at a fixed absolute value.
/// A fixed floor (e.g. `1e-10`) is not scale-invariant: on a well-scaled Hessian
/// a near-zero eigenvalue would yield a proposal variance of `inflation / 1e-10`
/// ≈ 1e10, scattering every SIR draw far outside the parameter bounds so the
/// fallback degenerates to "all samples had invalid weights". The relative floor
/// caps the proposal's condition number at `FALLBACK_PROPOSAL_MAX_COND`, keeping
/// the draws in a usable range while still giving the weakly-identified
/// directions the widest proposal.
///
/// `inflation = 4.0` is the recommended default: heavier tails account for the
/// uncertainty introduced by the non-PD correction.
///
/// Relative to the largest is still not invariant to rescaling *one* coordinate, which is
/// what writing an identity-packed θ in other units does. So the floor is taken on the
/// equilibrated `C = D⁻¹ H D⁻¹` of [`invert_psd_with_floor_unit_free`], with the same `units`
/// mask and so the same `dᵢ`, and the proposal mapped back, `D⁻¹ · C̃⁻¹ · D⁻¹` (#2179). With
/// no coordinate to rescale `D = I` and `H` is used as it is.
///
/// The result is embedded into the full packed-parameter covariance (zeros for
/// FIX parameters) and explicitly symmetrised, since the eigen-reconstruction
/// `V·diag·Vᵀ` can leave sub-ULP asymmetry that a downstream Cholesky rejects.
pub(crate) fn build_non_pd_fallback_proposal(
    hess_free_sym: &DMatrix<f64>,
    free_idx: &[usize],
    n_full: usize,
    inflation: f64,
    units: &[CoordUnits],
) -> DMatrix<f64> {
    let n = hess_free_sym.nrows();
    debug_assert_eq!(units.len(), n, "one flag per coordinate");
    let scales = unit_coordinate_scales(hess_free_sym, units);
    let cond = match &scales {
        Some(d) => DMatrix::from_fn(n, n, |i, j| hess_free_sym[(i, j)] / (d[i] * d[j])),
        None => hess_free_sym.clone(),
    };
    let eig = SymmetricEigen::new(cond);
    // Largest absolute eigenvalue anchors the relative floor. Guard the
    // all-zero block (max_abs == 0) with a tiny absolute fallback so the floor
    // stays positive and we never divide by zero.
    let max_abs = eig
        .eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let floor = (max_abs / FALLBACK_PROPOSAL_MAX_COND).max(1e-10);
    // Proposal covariance eigenvalues: inflation / max(|λ_i|, floor).
    let inv_eigs: DVector<f64> = eig.eigenvalues.map(|v| inflation / v.abs().max(floor));
    // Reconstruct: C_free = V * diag(inv_eigs) * V^T, then symmetrise to remove
    // any floating-point asymmetry from the matrix products.
    let cov_free_raw =
        &eig.eigenvectors * DMatrix::from_diagonal(&inv_eigs) * eig.eigenvectors.transpose();
    let mut cov_free = (&cov_free_raw + cov_free_raw.transpose()) * 0.5;
    if let Some(d) = &scales {
        for i in 0..n {
            for j in 0..n {
                cov_free[(i, j)] /= d[i] * d[j];
            }
        }
    }
    embed_free(&cov_free, free_idx, n_full)
}

/// One perturbation of `x̂` the FD stencil evaluates: a single coordinate (the diagonal
/// stencil) or a pair (the off-diagonal one). A *spec*, not a materialised x-vector — see
/// [`eval_fd_point`].
#[derive(Clone, Copy)]
enum FdPert {
    Single {
        i: usize,
        di: f64,
    },
    Pair {
        i: usize,
        di: f64,
        j: usize,
        dj: f64,
    },
}

/// The diagonal stencil's specs at step `eps`, in the order both [`select_fd_step`] and
/// [`fd_ofv_stencil`] lay them out: `+hₐ, −hₐ` for each free coordinate `a`, with
/// `hₐ = eps·(sᵢ + |x̂ᵢ|)` ([`fd_step`], `sᵢ = floor[i]`, #2147). One builder for both, so the
/// probe's accepted pass is the stencil's diagonal by construction rather than by two loops
/// happening to agree (#1935).
fn fd_diag_specs(x_hat: &[f64], free_idx: &[usize], eps: f64, floor: &[f64]) -> Vec<FdPert> {
    let mut specs = Vec::with_capacity(2 * free_idx.len());
    for &i in free_idx {
        let hi = fd_step(eps, x_hat[i], floor[i]);
        specs.push(FdPert::Single { i, di: hi });
        specs.push(FdPert::Single { i, di: -hi });
    }
    specs
}

/// Evaluate `ofv` at one perturbed point. Each call clones `x_hat` once, so a `par_iter` over
/// specs keeps only ~n_threads vectors live instead of all ~2·nf² perturbed points (#298).
fn eval_fd_point<F: Fn(&[f64]) -> f64>(x_hat: &[f64], p: FdPert, ofv: &F) -> f64 {
    let mut xv = x_hat.to_vec();
    match p {
        FdPert::Single { i, di } => xv[i] += di,
        FdPert::Pair { i, di, j, dj } => {
            xv[i] += di;
            xv[j] += dj;
        }
    }
    ofv(&xv)
}

/// What [`select_fd_step`] settled on.
pub(crate) struct FdStep {
    /// The step to build the stencil with.
    pub eps: f64,
    /// How many times `initial_eps` was halved to reach `eps`.
    pub n_halvings: usize,
    /// The accepted pass's objectives `[f(x̂+h₀e₀), f(x̂−h₀e₀), f(x̂+h₁e₁), …]` at `eps` — the
    /// stencil's own diagonal points, in its spec order, for [`fd_ofv_stencil`] to take
    /// instead of evaluating them a second time (#1935). `None` when every halving failed:
    /// `eps` then names a step no pass evaluated, so there is nothing to reuse.
    pub diag: Option<Vec<f64>>,
}

/// Choose a finite-difference step that keeps all free-parameter diagonal
/// stencils finite, starting from `initial_eps` and halving up to
/// `MAX_HALVINGS` times.
///
/// If every halving fails (all stencils still non-finite at
/// `initial_eps / 2^MAX_HALVINGS`), returns the final eps anyway — the FD loop will
/// detect and report the remaining failures.
///
/// The probe is on the scalar-OFV second-difference stencil `(f₊ − 2·f₀ + f₋)/h²`, and its
/// points are exactly the diagonal points of [`fd_ofv_stencil`] — the only FD Hessian builder,
/// on every route. So the accepted pass is not a proxy for the stencil's diagonal, it *is* the
/// diagonal, and [`FdStep::diag`] hands it over rather than having the stencil recompute
/// `2·n_free` reconverged population objectives (#1935).
///
/// Each pass evaluates its `2·n_free` points in parallel, the same way the stencil does — the
/// objective reconverges every subject serially, so before #1935 the probe was the one part of
/// the step that ran on one thread. A pass therefore runs in full even when an early point is
/// already non-finite; that costs extra work only on the rare halving path. `find_ebe` is
/// deterministic per point and `collect()` keeps the order, so the values are the serial
/// loop's bit for bit.
///
/// Cooperative cancel: a cancelled point returns `NaN` without evaluating, and a cancelled
/// pass returns at once with `diag: None`; the caller checks the token.
pub(crate) fn select_fd_step<F: Fn(&[f64]) -> f64 + Sync>(
    x_hat: &[f64],
    free_idx: &[usize],
    floor: &[f64],
    initial_eps: f64,
    f0: f64,
    ofv: &F,
    options: &FitOptions,
) -> FdStep {
    const MAX_HALVINGS: usize = 8;
    let mut eps = initial_eps;
    for halvings in 0..MAX_HALVINGS {
        let specs = fd_diag_specs(x_hat, free_idx, eps, floor);
        let report = cov_progress("step probe", specs.len(), options.verbose);
        let vals: Vec<f64> = specs
            .par_iter()
            .map(|&p| {
                if crate::cancel::is_cancelled(&options.cancel) {
                    report();
                    return f64::NAN;
                }
                let v = eval_fd_point(x_hat, p, ofv);
                report();
                v
            })
            .collect();
        if crate::cancel::is_cancelled(&options.cancel) {
            return FdStep {
                eps,
                n_halvings: halvings,
                diag: None,
            };
        }
        // Mirror the diagonal stencil the FD loop actually computes — (fp − 2·f0 + fm) / hi² —
        // including the division. A finite numerator can still overflow once divided by a tiny
        // hi², and the FD loop rejects on the quotient, so accepting the step here on the
        // numerator alone would hand back an eps the loop then rejects.
        let all_ok = free_idx.iter().enumerate().all(|(a, &i)| {
            let hi = fd_step(eps, x_hat[i], floor[i]);
            let h_ii = (vals[2 * a] - 2.0 * f0 + vals[2 * a + 1]) / (hi * hi);
            h_ii.is_finite()
        });
        if all_ok {
            return FdStep {
                eps,
                n_halvings: halvings,
                diag: Some(vals),
            };
        }
        eps *= 0.5;
    }
    FdStep {
        eps,
        n_halvings: MAX_HALVINGS,
        diag: None,
    }
}

/// Combine the observed-information inverse `r_inv = R⁻¹` (already `2·H_ofv⁻¹`)
/// and the score cross-product `S` into the covariance estimator selected by
/// `method`:
///   - `Hessian`      → `R⁻¹`            (model-based; `S` ignored)
///   - `CrossProduct` → `S⁻¹`            (empirical information)
///   - `Sandwich`     → `R⁻¹ S R⁻¹`      (Huber–White, robust)
///
/// Returns `None` only for `CrossProduct`, when `S` is not strictly
/// positive-definite — singular *or* merely rank-deficient (fewer subjects than
/// free parameters, or collinear scores). Unlike the Hessian path, a
/// rank-deficient `S` is **rejected** rather than eigenvalue-floored: `S⁻¹` of a
/// regularised `S` would silently report finite-but-fictitious SEs in the
/// unidentified directions, so the cross-product estimator requires a full-rank
/// `S`. `Sandwich` never inverts `S`, so it stays defined even when `S` is
/// rank-deficient.
///
/// `unit_coords` flags the coordinates whose units the floor on `S` must not see
/// ([`invert_psd_with_floor_unit_free`], #2145).
pub(crate) fn combine_covariance(
    method: CovarianceMethod,
    r_inv: DMatrix<f64>,
    s: &DMatrix<f64>,
    unit_coords: &[CoordUnits],
) -> Option<DMatrix<f64>> {
    match method {
        CovarianceMethod::Hessian => Some(r_inv),
        CovarianceMethod::Sandwich => Some(&r_inv * s * &r_inv),
        CovarianceMethod::CrossProduct => invert_cross_product(s, unit_coords),
    }
}

/// `S⁻¹`, accepted only when `S` is full-rank (no eigenvalue clipped); a rank-deficient or
/// indefinite `S` yields `None`. The whole of the `CrossProduct` estimator — it reads no `R` —
/// which is why `covariance_method = s` can call it without building one (#1803).
///
/// The rank test is the floor of [`invert_psd_with_floor_unit_free`], so an identity-packed θ
/// written in large or small units is not refused as rank-deficient (#2145).
pub(crate) fn invert_cross_product(
    s: &DMatrix<f64>,
    unit_coords: &[CoordUnits],
) -> Option<DMatrix<f64>> {
    match invert_psd_with_floor_unit_free(s, unit_coords) {
        Some(inv) if inv.n_clipped == 0 => Some(inv.inverse),
        _ => None,
    }
}

/// Whether the requested estimator reads the observed information `R` at all (#1803).
///
/// `S⁻¹` does not: like NONMEM's `$COV MATRIX=S`, it needs only the score cross-product, so
/// under `covariance_method = s` the step builds no `R` — no `2·n²`-objective stencil, and
/// none of `R`'s vetoes (a flat or non-finite diagonal, a spectrum with no positive eigenvalue),
/// which said nothing about whether `S⁻¹` exists. An exhaustive match, so a new estimator has
/// to declare which side it is on.
pub(crate) fn needs_r(method: CovarianceMethod) -> bool {
    match method {
        CovarianceMethod::Hessian | CovarianceMethod::Sandwich => true,
        CovarianceMethod::CrossProduct => false,
    }
}

/// Population objectives `R`'s finite-difference route evaluates at `n` free coordinates,
/// `None` when the count overflows `usize`: `2·n` for the step probe (`select_fd_step`), whose
/// accepted pass is the stencil's diagonal (#1935), then four per pair in `fd_ofv_stencil` —
/// `2·n²` in all. Each one re-converges every subject's EBEs. This is the count when the first
/// step is accepted; each halving `select_fd_step` needs adds `2·n` (up to eight passes), so it
/// is a lower bound. The single source of the count every cost warning quotes (#1803); the
/// analytic `R` assembly, where it applies, evaluates none of them.
pub(crate) fn fd_r_objective_evals(n: usize) -> Option<usize> {
    n.checked_mul(n)?.checked_mul(2)
}

/// The step's reason when `S` is singular or rank-deficient under `covariance_method = s`.
pub(crate) const S_SINGULAR_MSG: &str =
    "Covariance step failed: the score cross-product matrix S is \
     singular or rank-deficient (covariance_method = s); typically fewer subjects than free \
     parameters, or collinear per-subject scores. Use covariance_method = r or rsr, which \
     never invert S; both need R positive definite at the estimate, which after saem, imp, \
     impmap or vi it may not be. SE estimates not available.";

/// The free-block matrix `m` (ordered as `free_idx`) embedded into the full `n × n` packed
/// space, FIX rows and columns left at zero.
pub(crate) fn embed_free(m: &DMatrix<f64>, free_idx: &[usize], n: usize) -> DMatrix<f64> {
    let mut full = DMatrix::zeros(n, n);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            full[(i, j)] = m[(a, b)];
        }
    }
    full
}

/// [`combine_covariance`] applied to the returned inverse **and** to the eigen-floor's
/// unclipped reference, so both go through the same estimator (#1508 review §1).
///
/// This is the wiring the regularization diagnostic depends on and the reason it is a function
/// rather than two lines inside `compute_covariance`: under `covariance_method = rsr` the
/// returned covariance is `R⁻¹ S R⁻¹`, so the *same* floored `R` yields a different reported
/// inflation for a different `S` — `S` can suppress or concentrate the floored eigendirection.
/// Combining the reference with `R⁻¹` instead of with the selected estimator would put the old,
/// `S`-blind number back, and inside `compute_covariance` no Tier-1 test could reach it.
///
/// `None` propagates [`combine_covariance`]'s only failure (a rank-deficient `S` under
/// `covariance_method = s`). The reference is `None` whenever the caller has nothing to compare
/// — nothing clipped, or an estimator that never inverts `R`.
pub(crate) fn combine_covariance_and_reference(
    method: CovarianceMethod,
    r_inv: DMatrix<f64>,
    r_inv_ref: Option<DMatrix<f64>>,
    s: &DMatrix<f64>,
    unit_coords: &[CoordUnits],
) -> Option<(DMatrix<f64>, Option<DMatrix<f64>>)> {
    let reference = r_inv_ref.and_then(|ri| combine_covariance(method, ri, s, unit_coords));
    Some((
        combine_covariance(method, r_inv, s, unit_coords)?,
        reference,
    ))
}

/// Assemble the per-subject score cross-product `S = Σᵢ gᵢgᵢᵀ` over the free
/// parameter block, where `gᵢ = ∂(−logLᵢ)/∂θ` is subject `i`'s contribution to
/// the population score (the same per-subject gradient the Gauss–Newton optimizer
/// uses for its BHHH step). `S` is NONMEM's `S` matrix; combined with the
/// observed-information `R` it yields the `S⁻¹` and `R⁻¹SR⁻¹` covariance forms.
///
/// The result is `n_free × n_free`, ordered to match `free_idx`. Caller embeds it
/// (or its inverse) back into the full packed space.
/// Warning recorded when a cooperative cancel ([`crate::cancel::CancelFlag`])
/// is observed mid-covariance-step. The step is gated at entry too (so a flag
/// set before it starts skips it entirely); this message covers a flag flipped
/// *during* the long finite-difference / score loops, which short-circuit and
/// return [`CovarianceStepResult::Unusable`] so the fit still finishes (without
/// standard errors) instead of running the cancelled work to completion.
pub(super) const COV_CANCELLED_MSG: &str =
    "Covariance step cancelled before completion; standard errors not available.";

/// Throttle stride for the covariance progress reporter: at most ~20 lines per
/// loop, but always at least one (`max(1)` guards `total < 20`).
pub(crate) fn cov_progress_step(total: usize) -> usize {
    (total / 20).max(1)
}

/// Whether the `n`-th completed item (1-based) should emit a progress line:
/// every `step` items, plus the final item so the loop always reports 100%.
pub(crate) fn cov_progress_should_print(n: usize, total: usize, step: usize) -> bool {
    n % step == 0 || n == total
}

/// Estimated seconds remaining, extrapolated from observed wall-clock
/// throughput: `elapsed · (total − n) / n`. Returns 0 before any item finishes
/// or before any wall-clock has elapsed (avoids a divide-by-zero / Inf ETA).
pub(crate) fn cov_progress_eta(total: usize, n: usize, elapsed: f64) -> f64 {
    if n > 0 && elapsed > 0.0 {
        (total - n) as f64 * elapsed / n as f64
    } else {
        0.0
    }
}

/// Wall-clock progress reporter for the covariance step's parallel loops.
///
/// Returns a closure to be called once per completed item from inside a rayon
/// `par_iter().map(...)`. When `verbose`, it prints a throttled
/// `n/total (~Ns left)` line to stderr (matching the existing
/// `Computing covariance matrix...` style). The ETA extrapolates from observed
/// wall-clock throughput, so it already absorbs the rayon speed-up rather than
/// assuming serial per-item cost. Parallel out-of-order completion keeps the
/// count monotone but makes the ETA noisy early; it tightens as the loop runs.
///
/// The returned closure is `Fn + Sync` (atomic counter + `Instant`), so it can
/// be shared across the rayon worker threads.
fn cov_progress(label: &'static str, total: usize, verbose: bool) -> impl Fn() + Sync {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let done = AtomicUsize::new(0);
    let start = std::time::Instant::now();
    let step = cov_progress_step(total);
    move || {
        if !verbose {
            return;
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if !cov_progress_should_print(n, total, step) {
            return;
        }
        let eta = cov_progress_eta(total, n, start.elapsed().as_secs_f64());
        eprintln!("  [covariance] {label} {n}/{total} (~{eta:.0}s left)");
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_score_cross_product(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
    free_idx: &[usize],
) -> Result<DMatrix<f64>, String> {
    let n_free = free_idx.len();
    let n_subj = population.subjects.len();
    let quadrature_options = options.agq_nodes().map(|_| FitOptions {
        inner_tol: options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner()),
        inner_restarts: 0, // match the covariance step's reconvergence policy
        ..options.clone()
    });
    let quadrature = quadrature_options.as_ref().map(|score_options| {
        crate::estimation::agq::SubjectScoreContext::new(
            model,
            template,
            x_hat,
            score_options,
            bounds,
            crate::estimation::outer_optimizer::reconverge_this_eval(options, 0),
        )
        // The covariance step's own tolerance, not the outer gradient's stencil one (#2261).
        .with_fd_inner_tol(score_options.inner_tol)
    });
    // A mixture's subject marginal is `−log Σₖ pᵢₖ e^{−nllᵢₖ}`, not one class's nll
    // (#1956). Scoring it through the single-population path below ran every subject
    // under the rayon worker's default `MIXNUM = 1` with the base Ω/Σ, so the
    // class-2-only θ and the mixing θ had identically-zero score columns: `s` was
    // refused as singular and `rsr` silently lost those coordinates. The mixture path
    // computes its own per-class modes at `x̂`, so `eta_hats` / `h_matrices` /
    // `kappas` (the MIXEST class's) are not read for it.
    let mixture = model.mixture.as_ref().map(|_| {
        crate::estimation::mixture::MixtureScoreContext::new(
            model, population, template, x_hat, options,
        )
    });

    // Per-subject scores in parallel (mirrors `build_gn_system`).
    //
    // Convention: each row is the **fixed-mode** score, the per-subject gradient at x̂
    // with η̂, `H` and κ̂ held at their x̂ values (`fixed_ebe_subject_score`). The FOCEI
    // EBE-response term `tᵢ = ½·∂log|H̃|/∂η̂·dη̂/dθ` it adds back is identically zero
    // today (#338 Part A removed the θ block; the ω/σ blocks are #1817), so S carries
    // neither that term nor the θ-dependence of the held prediction Jacobian. The
    // FD-built R reconverges η̂ at every stencil point and is unaffected. NONMEM's
    // `MATRIX=S` differentiates the total `−2LLᵢ(x)`, η̂ re-estimated at every
    // perturbed x. The held score is biased at the MLE (`Σᵢ gᵢ ≠ 0`), so the two S
    // differ by a convention gap that more subjects do not close — measured on
    // `mixture_iv` and its one-population twin in `tests/mixture_cov_s_nonmem.rs`.
    // Closing it is #1817's territory. FOCE (`!interaction`) uses the Sheiner–Beal
    // gradient, which has no `log|H̃|` term, so `tᵢ` is not called for it.
    let report = cov_progress("score matrix", n_subj, options.verbose);
    let scores: Vec<Result<Vec<f64>, String>> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            // Cooperative cancel: skip the per-subject gradient and return a
            // cheap zero score so the in-flight rayon queue drains fast. The
            // caller (`compute_covariance`) re-checks the flag and discards this
            // matrix before it is used, so the placeholder is never trusted.
            if crate::cancel::is_cancelled(&options.cancel) {
                report();
                return Ok(vec![0.0; x_hat.len()]);
            }
            let kap_i = if i < kappas.len() {
                kappas[i].as_slice()
            } else {
                &[]
            };
            if let Some(context) = &mixture {
                let score = context.score(model, population, i, options);
                report();
                return Ok(score);
            }
            if let Some(context) = &quadrature {
                // The subject score has NLL units already and cannot contain the
                // optimizer's population-level EBE penalty. Never square a failed row.
                let score = context.score(&population.subjects[i], eta_hats[i].as_slice(), kap_i)
                    .ok_or_else(|| format!("Covariance step failed: could not obtain a converged, finite quadrature score for subject {}. SE estimates not available.", population.subjects[i].id));
                report();
                return score;
            }
            let gi = fixed_ebe_subject_score(
                x_hat,
                template,
                model,
                population,
                i,
                &eta_hats[i],
                &h_matrices[i],
                kap_i,
                bounds,
                options,
            );
            report();
            Ok(gi)
        })
        .collect();

    let mut s = DMatrix::zeros(n_free, n_free);
    for gi in scores {
        let gi = gi?;
        let gi_free = DVector::from_iterator(n_free, free_idx.iter().map(|&k| gi[k]));
        s.ger(1.0, &gi_free, &gi_free, 1.0); // s += gi_free * gi_freeᵀ (full outer product)
    }
    if s.iter().any(|v| !v.is_finite()) {
        return Err(
            "Covariance step failed: non-finite score cross-product. SE estimates not available."
                .into(),
        );
    }
    Ok(s)
}

/// One subject's single-population score `∂nllᵢ/∂x` at fixed modes (η̂, `H`, κ̂) — the
/// row `assemble_score_cross_product` squares into S, and (#1956) each class's term
/// of a mixture subject's score. `subject_nll_pop_grad` (closed form or central FD),
/// plus the FOCEI `log|H̃|` EBE-response term `tᵢ`, which is identically zero today
/// (#338 Part A); see the convention comment in `assemble_score_cross_product`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixed_ebe_subject_score(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    kappas: &[DVector<f64>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let (_, mut gi) = crate::estimation::gauss_newton::subject_nll_pop_grad(
        x, template, model, population, subj_idx, eta_hat, h_matrix, kappas, bounds, options,
    );
    if options.interaction {
        if let Some(ti) = crate::estimation::gauss_newton::subject_eta_response_correction(
            None, x, template, model, population, subj_idx, eta_hat, h_matrix, kappas, bounds,
            options,
        ) {
            for (g, t) in gi.iter_mut().zip(ti.iter()) {
                *g += *t;
            }
        }
    }
    gi
}

/// Test hook: subject ids whose analytic covariance term declines, so a test can reach the
/// #1514 salvage (`AnalyticCovAssembly::Partial`) without changing any prediction. Global, not
/// thread-local: the assembly runs on the rayon workers. Ids are test-unique by convention.
#[cfg(test)]
pub(super) mod forced_cov_decline {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    static IDS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

    pub(in crate::estimation) fn decline(id: &str) {
        IDS.lock().unwrap().insert(id.to_string());
    }

    pub(in crate::estimation) fn declines(id: &str) -> bool {
        IDS.lock().unwrap().contains(id)
    }
}

/// The objective the analytic R-matrix differentiates: the one the fit minimised (see
/// `analytic_cov_hessian`).
fn cov_objective(options: &FitOptions) -> crate::sens::ode_provider::adjoint::Objective {
    use crate::sens::ode_provider::adjoint::Objective;
    match options.agq_nodes() {
        Some(n_agq) => Objective::Agq {
            n_agq,
            exact: options.hessian_anchor() != HessianAnchor::GaussNewton,
        },
        None if options.interaction => Objective::Focei,
        None => Objective::Foce,
    }
}

/// **Every** clause that kept this fit off the exact analytic covariance R-matrix (#520 C2).
///
/// All of them, not the first: the remedy sentence promises that an action "moves the fit onto
/// the analytic route", and that is only true when the action clears every clause that
/// declined. Dropping `gradient = fd` does not move a non-Gaussian model, and enabling
/// `analytic_cov_hessian` does not move a mixture — both were promised unconditionally before
/// (#1508 review §3). The three model-level gates are therefore collected side by side rather
/// than short-circuited, and the per-subject width clause asks
/// `sens::ode_provider::adjoint::cov_width_declines`, the rule the Hessian dispatch itself
/// runs on — so a five-effect Laplace fit with `analytic_cov_hessian = false` is told that
/// enabling the flag would not move it (Copilot review on #2093).
///
/// Returns an empty vec only when the analytic route was in fact taken — so a caller that has
/// already observed `analytic_cov_hessian(..) == None` and still finds nothing here has found a
/// genuine drift between this walk and the assembly, and gets
/// [`CovScopeDecline::PerSubjectBail`] instead of silence.
///
/// `iov` is `n_kappa > 0` because the gate declines both of the other combinations
/// (`!iov && n_kappa > 0` and `iov && n_kappa == 0`), so it is the only value that can pass.
///
/// Called only when a regularization warning is about to be emitted on the FD route, i.e. after
/// a stencil that has just spent `2·n_free²` reconverged population objectives.
fn analytic_cov_declines(
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
    is_mixture: bool,
) -> Vec<CovScopeDecline> {
    let mut out = analytic_cov_scope_declines(model, population, options, is_mixture);
    if out.is_empty() {
        out.push(CovScopeDecline::PerSubjectBail);
    }
    out
}

/// How much the step budget grows for a step tightened by [`cov_ode_override`].
/// Tightening the default `ode_reltol` 100× costs an order-`p` stepper ~100^(1/p) more steps:
/// ~2.5× for rk45, ~4.6× for rosenbrock23, whose step control runs on an O(h³) local error.
/// 10 covers both with 2× headroom; it is a cap, paid only by a solve that needs it.
const COV_STENCIL_ODE_MAX_STEPS_FACTOR: usize = 10;

/// The ODE solver override a covariance step whose matrices are finite-differenced runs under
/// (#520), or `None` to run every step at the fit's own tolerance. Which steps take it is
/// [`compute_covariance_for`]'s call: one that differences `R`'s objective, and one that reads
/// the score cross-product `S` alone — whose per-subject θ scores are differenced too, forward
/// or central, even on the closed-form gradient paths (#520 review round 2). The whole step,
/// not those matrices alone, so no estimate mixes tolerances (round 6).
///
/// The stencil divides by `h²`, so it amplifies integration noise the fit's optimizer averages
/// over: on the 3-cpt IV ODE fixture (180 subjects, `analytic_cov_hessian = false`) the default
/// `1e-4` / `1e-6` left the SEs 2–7× too large, silently, and once regularized an SE 4400× too
/// large; `1e-6` / `1e-8` put them on the plateau, unchanged to three figures out to `1e-10`,
/// for 0.95–1.17× the stencil's wall time. So `reltol` / `abstol` drop to
/// [`COV_FD_PLATEAU_RELTOL`] / [`COV_FD_PLATEAU_ABSTOL`] each, where they are looser — a fit
/// already tighter keeps its own — and the step budget grows by
/// [`COV_STENCIL_ODE_MAX_STEPS_FACTOR`]. Every other field is the fit's own override, so a
/// caller's `ode_method` reaches the stencil.
///
/// The SEs then come from a marginally different objective than the one the fit converged on,
/// as NONMEM's `$COVARIANCE` re-evaluation also allows: a central second difference needs no
/// stationarity, only a smooth objective, and the tighter one is the smoother.
///
/// `None` when the model has no integrator at all (a closed form without an absorption ODE
/// twin) or when the tolerance already sits on the plateau. Read on the thread the step runs
/// on, where the fit's override is armed, so the tolerance compared is the one the fit
/// actually integrates at.
///
/// A model solved by matrix exponential (#852) is **not** exempt: the exponential declines a
/// segment it cannot propagate (possibly at a perturbed stencil point only), and a declined
/// segment is integrated at the tolerance armed here (#520 review round 3). The exact segments
/// ignore the tolerance, so on those the override changes no number.
///
/// Keyed on the twin's *existence*, not on [`OdeToleranceFacts::from_route`]'s per-subject
/// walk: the flip-flop reroute ([`crate::pk::effective_model_for_eval`]) is decided per
/// `(θ, η)`, so a transit / inverse-Gaussian subject can reach its twin at a perturbed stencil
/// point only, which no per-subject prediction sees (#520 review). Arming the override on a
/// model whose subjects never integrate changes no number; it costs a pool lease.
fn cov_ode_override(
    model: &CompiledModel,
    options: &FitOptions,
) -> Option<crate::ode::solver::OdeSolverOverride> {
    use crate::estimation::cov_diagnostics::{COV_FD_PLATEAU_ABSTOL, COV_FD_PLATEAU_RELTOL};
    let spec = model.ode_spec.as_ref().or_else(|| {
        model
            .absorption_ode_equivalent
            .as_ref()
            .and_then(|eq| eq.built().ode_spec.as_ref())
    })?;
    let eff = spec.effective_solver_opts();
    if eff.reltol <= COV_FD_PLATEAU_RELTOL && eff.abstol <= COV_FD_PLATEAU_ABSTOL {
        return None;
    }
    Some(crate::ode::solver::OdeSolverOverride {
        reltol: Some(eff.reltol.min(COV_FD_PLATEAU_RELTOL)),
        abstol: Some(eff.abstol.min(COV_FD_PLATEAU_ABSTOL)),
        max_steps: Some(
            eff.max_steps
                .saturating_mul(COV_STENCIL_ODE_MAX_STEPS_FACTOR),
        ),
        ..options.ode_solver_override()
    })
}

/// The warning for a covariance step [`cov_ode_override`] meant to tighten but could not,
/// because the scope carrying the tighter tolerance could not be built (`reason`).
pub(crate) fn stencil_scope_failure_warning(reason: &str) -> String {
    use crate::estimation::cov_diagnostics::{
        COV_FD_PLATEAU_ABSTOL, COV_FD_PLATEAU_RELTOL, ODE_TOLERANCE_NOT_TIGHTENED_TOKEN,
    };
    format!(
        "{ODE_TOLERANCE_NOT_TIGHTENED_TOKEN}: Covariance step: the finite-difference matrices \
         could not be run at ode_reltol = {COV_FD_PLATEAU_RELTOL:.0e} / ode_abstol = \
         {COV_FD_PLATEAU_ABSTOL:.0e} ({reason}) and ran at the fit's own ODE tolerance instead; \
         finite differences amplify integration noise, so these standard errors may be \
         inflated. Set ode_reltol / ode_abstol to at most those values to compute them at the \
         plateau (#520)."
    )
}

/// Which route the covariance step is expected to build `R` on, predicted before the fit from
/// the same walk [`analytic_cov_declines`] reports (#2054): the model-level gates, then
/// `cov_width_declines` per subject, then the salvage cut [`salvages_declined`] the assembly
/// itself applies. The prediction can still be wrong when a subject bails at run time (a trial
/// point the recorder cannot form) — then that subject joins the salvaged minority, or, past
/// the cut, the whole population takes the stencil — which is why every message built from it
/// is worded as an expectation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredictedRRoute {
    /// Every subject on the analytic assembly: no finite-difference objective at all.
    Analytic,
    /// A strict minority of subjects declines; the `2·n²`-point stencil runs over those
    /// `declined` of `n_subjects` alone (#1514).
    Salvage { declined: usize, n_subjects: usize },
    /// The whole-population stencil.
    Stencil,
}

/// Whether the assembly salvages `declined` of `n_subjects` declining subjects on their own
/// stencil, rather than taking the population stencil — the one cut both
/// [`analytic_cov_assembly`] and [`predicted_r_route`] read. `declined > 0` is the caller's.
fn salvages_declined(declined: usize, n_subjects: usize) -> bool {
    (declined as f64) < SALVAGE_MAX_DECLINED_FRACTION * n_subjects as f64
}

/// The covariance step's expected `R` route; see [`PredictedRRoute`].
///
/// What this decides is the *cost* of `R`: one profile-Hessian pass per subject instead of
/// `fd_r_objective_evals(n)` reconverged population objectives, so the large-problem reroute to
/// `S` and the pre-fit cost note, both written for the stencil, do not apply on
/// [`PredictedRRoute::Analytic`].
pub(crate) fn predicted_r_route(
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
    is_mixture: bool,
) -> PredictedRRoute {
    if !analytic_cov_model_declines(model, options, is_mixture).is_empty() {
        return PredictedRRoute::Stencil;
    }
    let n_subjects = population.subjects.len();
    match width_declined_subjects(model, population, options) {
        0 => PredictedRRoute::Analytic,
        // The mixture objective is not a per-subject sum: no salvage, all or nothing.
        _ if is_mixture => PredictedRRoute::Stencil,
        declined if salvages_declined(declined, n_subjects) => PredictedRRoute::Salvage {
            declined,
            n_subjects,
        },
        _ => PredictedRRoute::Stencil,
    }
}

/// Whether the covariance step is expected to take the analytic R-matrix for every subject:
/// [`predicted_r_route`] is [`PredictedRRoute::Analytic`], i.e. no clause of
/// [`analytic_cov_declines`]'s walk fires.
pub(crate) fn analytic_r_expected(
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
    is_mixture: bool,
) -> bool {
    predicted_r_route(model, population, options, is_mixture) == PredictedRRoute::Analytic
}

/// The clauses of [`analytic_cov_declines`], empty when none fires.
fn analytic_cov_scope_declines(
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
    is_mixture: bool,
) -> Vec<CovScopeDecline> {
    let mut out = analytic_cov_model_declines(model, options, is_mixture);
    if width_declined_subjects(model, population, options) > 0 {
        out.push(CovScopeDecline::TooManyRandomEffects);
    }
    out
}

/// The model-level clauses of [`analytic_cov_declines`] — every one that sends the whole
/// population to the stencil, whatever the subjects look like.
fn analytic_cov_model_declines(
    model: &CompiledModel,
    options: &FitOptions,
    is_mixture: bool,
) -> Vec<CovScopeDecline> {
    let mut out: Vec<CovScopeDecline> = Vec::new();
    if !options.analytic_cov_hessian {
        out.push(CovScopeDecline::Disabled);
    }
    // A mixture has its own exact assembly (`mixture::mixture_cov_hessian`), for the
    // FOCE/FOCEI class objectives `mixture_ofv` scores.
    if is_mixture
        && !matches!(
            cov_objective(options),
            crate::sens::ode_provider::adjoint::Objective::Foce
                | crate::sens::ode_provider::adjoint::Objective::Focei
        )
    {
        out.push(CovScopeDecline::Mixture);
    }
    // The adjoint's profile Hessian serves every estimator and anchor; what it declines is
    // the model (a shape no recorder serves, a non-Gaussian endpoint outside FOCE/FOCEI's
    // exact term), a subject wider than
    // its buckets, or the user's `gradient = fd` opt-out from analytic derivatives.
    if !crate::sens::ode_provider::adjoint::recorder_in_scope(model) {
        out.push(CovScopeDecline::ModelOutOfScope);
    }
    if crate::types::GradientMethod::forced_fd(model) {
        out.push(CovScopeDecline::GradientFd);
    }
    if model.has_non_gaussian()
        && !crate::sens::ode_provider::adjoint::cov_non_gaussian_served(
            model,
            cov_objective(options),
        )
    {
        out.push(CovScopeDecline::NonGaussianEndpoint);
    }
    out
}

/// How many subjects are too wide for the profile Hessian's buckets under the fit's
/// objective — the per-subject clause, through the rule the Hessian dispatch itself runs on.
fn width_declined_subjects(
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
) -> usize {
    let objective = cov_objective(options);
    population
        .subjects
        .iter()
        .filter(|s| crate::sens::ode_provider::adjoint::cov_width_declines(model, s, objective))
        .count()
}

/// Compute the parameter covariance matrix at convergence (the R-matrix:
/// inverse observed Fisher information).
///
/// The Hessian is built by finite differences that **reconverge the inner EBE
/// loop at every perturbed point** — matching how NONMEM's `$COVARIANCE` step
/// works. Holding the EBEs fixed (the previous behaviour) gives a Hessian with
/// the wrong curvature, indefinite even on well-conditioned surfaces like
/// warfarin, which forced eigenvalue clipping (#129) and inflated the SEs.
///
/// Two stencils:
/// - **non-IOV**: central FD of the analytical population gradient (issue #209),
///   `H[:,k] ≈ (g(x̂+hₖeₖ) − g(x̂−hₖeₖ)) / 2hₖ` — `2·n_free` gradient evaluations.
///   The θ part reuses H-matrix columns for mu-referenced parameters (issue #196).
/// - **IOV**: second differences of the reconverged OFV (the kappa block has no
///   fixed-EBE analytical gradient).
///
/// The returned covariance is `2·H⁻¹`: the objective is `−2·logL`, so its Hessian
/// is twice the observed information.
///
/// Returns [`CovarianceStepResult::Unusable`] when the FD Hessian is structurally
/// unusable (non-finite or zero-diagonal entries, or eigenvalues that diverge to
/// NaN/Inf so no proposal can be built). When the symmetrised free-block Hessian
/// is near-singular or has negative eigenvalues — a common FD noise artefact on
/// well-conditioned surfaces (see issue #129) — it is regularised by clipping
/// eigenvalues to a small positive floor before inversion, and the returned
/// `warning` records what was done. When the Hessian has finite eigenvalues but
/// no positive curvature at all (all eigenvalues ≤ 0), returns
/// [`CovarianceStepResult::FailedNonPd`], carrying the eigenvalue list formatted
/// as a warning together with an `|eigenvalue|`-rectified proposal covariance the
/// caller can hand to SIR when `covariance_fallback = sir`.
///
/// The estimator assembled from the Hessian `R` is selected by
/// [`FitOptions::covariance_method`] — `R⁻¹` (default), the score cross-product
/// `S⁻¹`, or the sandwich `R⁻¹SR⁻¹` (see [`assemble_score_cross_product`]).
/// How much of the population the exact analytic R-matrix (#436) could assemble.
///
/// Returned instead of an `Option` because a declining subject is a *route*, not a failure
/// (#1514). The observed information is `Σᵢ Rᵢ`, and every term in that sum is the second
/// derivative of one subject's own marginal: a subject this assembly declines can have its own
/// `Rⱼ` finite-differenced — from the same objective, at the same point, warm-started from the
/// same modes — and added in, without any other subject's term changing.
///
/// That is not the thing the old all-or-nothing comment ruled out. "A Hessian half-assembled
/// from two different approximations would be neither" describes averaging two estimates *of
/// the same matrix*; here each term is an estimate of itself. The fit-side gradient has
/// assembled the population exactly this way since #466 (`population_gradient_sens_mixed`).
///
/// Separability is what licenses it, and it holds for both halves of `cov_ofv`:
/// `reconverge_point` warm-starts every perturbed point from the *fit's* modes (a constant
/// across the whole stencil), so `x ↦ η̂ᵢ(x)` is a deterministic per-subject map; and `pop_nll`
/// is a per-subject sum, as is the AGQ branch. The one non-separable objective, `mixture_ofv`,
/// never reaches here — `compute_covariance` sends a mixture to its own all-or-nothing
/// assembly (`mixture::mixture_cov_hessian`).
pub(super) enum AnalyticCovAssembly {
    /// Every subject assembled analytically: the complete `∂²OFV/∂x²`.
    Full(DMatrix<f64>),
    /// `hess` is the sum over the in-scope subjects only. `declined` holds the **population
    /// indices** of the subjects whose terms still have to be finite-differenced and added.
    /// Non-empty, and always a strict minority (see [`SALVAGE_MAX_DECLINED_FRACTION`]).
    Partial {
        hess: DMatrix<f64>,
        declined: Vec<usize>,
    },
    /// The analytic route does not apply to this fit at all: the objective is not the one this
    /// assembly differentiates (exact-anchor Laplace), the modes handed in do not match the
    /// population, the step was cancelled, or so many subjects declined that the whole-
    /// population stencil is the cheaper way to the same matrix.
    Unavailable,
}

/// At or above this fraction of declining subjects, take the whole-population stencil instead
/// of salvaging (#1514).
///
/// The salvage is not free: the declined subjects get their own `select_fd_step` probe and
/// their own `2·n_free²`-point stencil, on top of an analytic pass that assembled almost
/// nothing. Measured on clofarabine (FOCEI, 3-state ODE, 56 subjects, **all 56** outside the
/// analytic scope): 7.91 s on the population stencil against 8.22 s rebuilt as 56 per-subject
/// stencils — 81/81 cells identical to 7 significant figures, +4 % wall for nothing. Half is
/// the conservative cut: the salvage's win comes from the analytic majority carrying the
/// population, and at parity there is no majority to carry it.
///
/// `>=`, so a single-subject population that declines takes the population stencil, which is
/// the same matrix by one fewer route.
const SALVAGE_MAX_DECLINED_FRACTION: f64 = 0.5;

/// A subject's mode as the adjoint's recordings take it: `η̂`, followed under IOV by every
/// occasion's `κ̂` (the stacked `[η, κ₁…κ_K]`), or `None` when the `κ̂` do not match the
/// subject's occasions.
pub(crate) fn stacked_mode<'a>(
    model: &CompiledModel,
    subject: &Subject,
    eta_hat: &'a DVector<f64>,
    kappas: Option<&Vec<DVector<f64>>>,
) -> Option<std::borrow::Cow<'a, [f64]>> {
    if model.n_kappa == 0 {
        return Some(std::borrow::Cow::Borrowed(eta_hat.as_slice()));
    }
    let kap = kappas?;
    if kap.len() != crate::stats::stacked_layout::group_count(subject)
        || kap.iter().any(|k| k.len() != model.n_kappa)
    {
        return None;
    }
    Some(std::borrow::Cow::Owned(
        eta_hat
            .iter()
            .copied()
            .chain(kap.iter().flat_map(|k| k.iter().copied()))
            .collect(),
    ))
}

/// Test watch: `(template address, count)` — the subjects [`per_subject_adjoint`] handed to
/// its closure for that one template, on any worker (keyed so concurrent tests do not count).
#[cfg(test)]
pub(crate) static SUBJECT_SWEEP_WATCH: std::sync::Mutex<Option<(usize, usize)>> =
    std::sync::Mutex::new(None);

/// The per-subject adjoint sweep the analytic R-matrix ([`analytic_cov_assembly`]) and the
/// #2159 identity-scale pre-flight ([`exact_candidate_derivatives`]) share: the
/// population-level gates, one [`PackedPoint`](crate::sens::ode_provider::adjoint::PackedPoint)
/// for every subject, each subject's stacked mode, cooperative cancellation and the test
/// decline hook, then `f(subject, point, mode, objective)` per subject in parallel. `None`
/// when a gate refuses the whole population; a `None` entry is a subject `f` (or its mode)
/// declined.
#[allow(clippy::too_many_arguments)]
fn per_subject_adjoint<R: Send>(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
    f: impl Fn(
            &Subject,
            &crate::sens::ode_provider::adjoint::PackedPoint<'_>,
            &[f64],
            crate::sens::ode_provider::adjoint::Objective,
        ) -> Option<R>
        + Sync,
) -> Option<Vec<Option<R>>> {
    // `eta_hats` indexes `population.subjects` positionally everywhere below, and the
    // `par_iter().zip()` that follows would silently *truncate* to the shorter of the two —
    // dropping the tail subjects from both the sum and the declined list, i.e. returning a
    // `Full` matrix missing terms. Under the old all-or-nothing shape that was a wrong matrix
    // too; here it would additionally be reported as complete, so it is refused outright.
    if eta_hats.len() != population.subjects.len() {
        return None;
    }
    // `gradient = fd` opts out of analytic derivatives, the covariance step's included.
    if crate::types::GradientMethod::forced_fd(model) {
        return None;
    }
    // The objective differentiated is the one the fit minimised: the FOCE/FOCEI marginal,
    // or — `agq_nodes()` is `Some` for `method = laplace` at any node count and
    // `method = focei` with `n_agq > 1` — the quadrature marginal under its own anchor (`H̃`,
    // or Laplace's exact `H`). Reporting `H̃`-derived SEs for a fit anchored on `H` would be
    // "standard errors for a likelihood it never optimised" (PR #953 review finding 1), so
    // the objective is taken from the same options the fit ran on.
    //
    // The adjoint's profile Hessian (`sens::ode_provider::adjoint::subject_adjoint_cov_hessian`)
    // differentiates every one of these objectives — FOCE (its frozen-variance mode
    // included), FOCEI, and the quadrature marginal under either anchor, Laplace's exact `H`
    // carried by an inner η-dual — so there is no anchor or estimator to decline here.
    let objective = cov_objective(options);
    // One unpack for the population: every subject is at `x`.
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(template, x)?;
    Some(
        population
            .subjects
            .par_iter()
            .zip(eta_hats.par_iter())
            .enumerate()
            .map(|(i, (subject, eta_hat))| {
                let b = stacked_mode(model, subject, eta_hat, kappas.get(i))?;
                // Cooperative cancel. The FD stencil checks on every perturbed point; this loop
                // is `2·(n_theta+n_eta)+1` provider evaluations plus an O(n_eta³·n_obs)
                // assembly per subject, and for the default `covariance_method = r` there is no
                // later checkpoint — so without this, the Ctrl-C affordance `saem.rs`
                // advertises before the covariance step (#893) was inoperative. Returning
                // `None` alone would drop into the *more* expensive FD stencil; the caller
                // re-checks the flag and reports cancelled instead (PR #953 review finding 10).
                if crate::cancel::is_cancelled(&options.cancel) {
                    return None;
                }
                #[cfg(test)]
                if forced_cov_decline::declines(&subject.id) {
                    return None;
                }
                #[cfg(test)]
                if let Ok(mut watch) = SUBJECT_SWEEP_WATCH.lock() {
                    if let Some((addr, n)) = watch.as_mut() {
                        if *addr == template as *const ModelParameters as usize {
                            *n += 1;
                        }
                    }
                }
                f(subject, &pt, &b, objective)
            })
            .collect(),
    )
}

/// The slope and the curvature of the OFV along each packed axis in `axes`, exact: `∂OFV/∂x_a`
/// (the adjoint gradient) and `∂²OFV/∂x_a²` (the R-matrix's diagonal along `axes`, at
/// `axes.len()` directions per subject), both of [`cov_objective`]'s objective at the **same**
/// modes, so the pair is one function's first and second derivative. What the #2159
/// identity-scale pre-flight reads its candidates' Newton step and curvature from.
///
/// The modes are re-solved warm from `warm` at `cov_inner_tol`, as the covariance step does
/// before its own R-matrix: both derivatives assume `∂lᵢ/∂η = 0` at the mode, which a fit's
/// `inner_tol` only approximates (#2291 review). `None` — and the caller probes instead —
/// unless every subject converges there and is served exactly, when the exact R-matrix is
/// switched off (`analytic_cov_hessian = false`), under `gradient = fd`, or on cancel. A
/// partial sum would be a different objective's derivatives.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exact_candidate_derivatives(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    warm: &[DVector<f64>],
    options: &FitOptions,
    axes: &[usize],
) -> Option<Vec<(f64, f64)>> {
    use crate::sens::ode_provider::adjoint::{
        subject_adjoint_cov_diag_at, subject_adjoint_gradient_at,
    };
    if !options.analytic_cov_hessian || warm.len() != population.subjects.len() {
        return None;
    }
    let cov_inner_tol = options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());
    let (_, ehs, _, kaps, all_converged) = reconverge_population_checked(
        x,
        model,
        population,
        template,
        warm,
        options,
        cov_inner_tol,
        true,
    );
    if !all_converged {
        return None;
    }
    let per_subject = per_subject_adjoint(
        model,
        population,
        template,
        x,
        &ehs,
        &kaps,
        options,
        |subject, pt, b, objective| {
            let g = subject_adjoint_gradient_at(model, subject, pt, b, objective)?;
            let d = subject_adjoint_cov_diag_at(model, subject, pt, b, objective, axes)?;
            let pairs: Vec<(f64, f64)> = axes
                .iter()
                .zip(d)
                .map(|(&a, h)| (g.get(a).copied().unwrap_or(f64::NAN), h))
                .collect();
            pairs
                .iter()
                .all(|(g, h)| g.is_finite() && h.is_finite())
                .then_some(pairs)
        },
    )?;
    if crate::cancel::is_cancelled(&options.cancel) {
        return None;
    }
    // In subject order, so the sum does not depend on the thread count.
    let mut acc = vec![(0.0, 0.0); axes.len()];
    for pairs in per_subject {
        for (a, (g, h)) in acc.iter_mut().zip(pairs?) {
            // ×2: the per-subject NLL to the OFV (see `analytic_cov_assembly`).
            a.0 += 2.0 * g;
            a.1 += 2.0 * h;
        }
    }
    Some(acc)
}

/// Assemble the exact analytic R-matrix (#436): `Σᵢ ∂²Fᵢ/∂x²` in packed coordinates, scaled to
/// the OFV convention, for as much of the population as is in scope.
///
/// Parallel subject assembly with a fixed-subject-order reduction, preserving deterministic
/// results while distributing the quadrature node sweeps across workers.
pub(super) fn analytic_cov_assembly(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x_hat: &[f64],
    eta_hats: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> AnalyticCovAssembly {
    use crate::sens::ode_provider::adjoint::subject_adjoint_cov_hessian_at;
    let n = x_hat.len();
    let Some(per_subject) = per_subject_adjoint(
        model,
        population,
        template,
        x_hat,
        eta_hats,
        kappas,
        options,
        |subject, pt, b, objective| {
            // The adjoint's profile Hessian (`Uᵀ∇²Ψ U`): every objective, either anchor.
            let h = subject_adjoint_cov_hessian_at(model, subject, pt, b, objective)?;
            if h.nrows() != n || h.ncols() != n || h.iter().any(|v| !v.is_finite()) {
                return None;
            }
            // ×2 — the OFV convention, and the one place this can be silently wrong.
            //
            // `subject_adjoint_cov_hessian` is the second derivative of `subject_packed_gradient`,
            // which is `∂Fᵢ/∂x` with `OFV = 2·Σᵢ Fᵢ` (see `population_gradient_sens_mixed`'s
            // `grad[k] += 2.0 * gi[k]`). The stencil this replaces differences `2·pop_nll`, so
            // `hess` here must be `∂²OFV/∂x²`, and the caller's `covariance = 2·H⁻¹` assumes it.
            // Summing the per-subject Hessians unscaled yields exactly half of that, which inflates
            // every standard error by √2 — with no other symptom, since the matrix stays symmetric,
            // positive-definite and plausibly sized.
            Some(2.0 * h)
        },
    ) else {
        return AnalyticCovAssembly::Unavailable;
    };
    // The R-matrix passes record on each worker's reverse tape, which keeps its buffers
    // between subjects so they are not reallocated per pass. Past the last subject nothing
    // else needs them at that size, so release them on every worker of the pool (#2093):
    // a 1M-node tape — what `rtape_trim`'s per-pass threshold lets a worker keep — is
    // ~100 MB (offsets, entries and eight adjoint lanes per node), and stayed pinned for the
    // rest of a long-lived process (an R session). This does not lower the step's own peak,
    // which is the tape at its largest during the passes: measured +4–5 MB of peak RSS over
    // `main` without the covariance step and +13–27 MB with it on the bundled models.
    rayon::broadcast(|_| crate::sens::rtape::rtape_trim(0));
    let mut acc = DMatrix::<f64>::zeros(n, n);
    let mut declined: Vec<usize> = Vec::new();
    for (i, h) in per_subject.into_iter().enumerate() {
        match h {
            Some(h) => acc += h,
            None => declined.push(i),
        }
    }
    // A cancel surfaces as a decline from every subject the flag caught. Salvaging those on
    // the FD stencil would be precisely the work the cancel asked to stop, and the caller
    // reads `Unavailable` + the flag as "cancelled" (PR #953 review finding 10).
    if crate::cancel::is_cancelled(&options.cancel) {
        return AnalyticCovAssembly::Unavailable;
    }
    if declined.is_empty() {
        return AnalyticCovAssembly::Full(acc);
    }
    if !salvages_declined(declined.len(), population.subjects.len()) {
        return AnalyticCovAssembly::Unavailable;
    }
    AnalyticCovAssembly::Partial {
        hess: acc,
        declined,
    }
}

/// The subjects at `idx`, in `idx` order, as their own `Population`, together with the matching
/// warm-start modes.
///
/// `pop_nll_opts` sums over whatever population it is handed, so this is all it takes to make
/// the covariance objective evaluate one subset's share of itself (#1514). `warm` is indexed
/// positionally against the returned subjects, which is why it is built here rather than at
/// each call site: the two vectors have to be permuted by the same `idx` or the salvaged
/// subject is warm-started from a different subject's modes.
///
/// `exclusions` and `covariate_names` are carried over unchanged and `warnings` is dropped —
/// the first two are read by downstream consumers of a `Population`, the third is a record of
/// how the dataset was parsed and would be duplicated into nothing here.
pub(super) fn subset_population(
    population: &Population,
    eta_hats: &[DVector<f64>],
    idx: &[usize],
) -> (Population, Vec<DVector<f64>>) {
    let pop = Population {
        subjects: idx
            .iter()
            .map(|&i| population.subjects[i].clone())
            .collect(),
        covariate_names: population.covariate_names.clone(),
        dv_column: population.dv_column.clone(),
        input_columns: population.input_columns.clone(),
        exclusions: population.exclusions.clone(),
        warnings: Vec::new(),
    };
    let warm = idx.iter().map(|&i| eta_hats[i].clone()).collect();
    (pop, warm)
}

/// The output of one reconverged-OFV second-difference stencil.
///
/// `pub(super)` so `salvage_tests` (`covariance_salvage_tests.rs`) can run the *production* stencil
/// over an arbitrary subject set and check that the population's equals the sum of the per-subject
/// ones — the premise #1514 rests on. A test that rebuilt the difference formulas to do that would
/// be comparing two copies rather than one formula on two inputs.
pub(super) struct FdStencil {
    /// `n`×`n`, zero outside the `free_idx` block and at every entry whose stencil was
    /// non-finite (those are reported through the two sets instead, because a stored zero is
    /// indistinguishable from genuinely flat curvature).
    pub(super) hess: DMatrix<f64>,
    /// Free indices whose diagonal stencil was non-finite.
    pub(super) diag_nan: HashSet<usize>,
    /// Free indices appearing in a non-finite cross-partial stencil.
    pub(super) offdiag_nan: HashSet<usize>,
}

/// Reconverged-OFV second-difference Hessian: 3-point diagonal, 4-point off-diagonal, over
/// `free_idx`, of whatever objective `ofv` evaluates.
///
/// It recomputes the marginal curvature end-to-end (`a = ∂f/∂η` and the `log|H̃|` EBE response
/// included) at every perturbed point, so it serves FOCE, FOCEI and IOV, and
/// additive/proportional/combined error uniformly — no envelope approximation, no held-fixed
/// `a`. `ofv` dispatches on the kappa count internally, so the same stencil is correct for the
/// IOV (joint η, κ) and non-IOV (η-only) cases.
///
/// Extracted from `compute_covariance` by #1514 so the whole-population route and the
/// per-subject salvage are **one** implementation. They differ only in which subjects `ofv`
/// sums; a second copy of the difference formulas would be a second place for the `4·hᵢ·hⱼ` to
/// go wrong, and the parity test between the two routes would then be comparing two copies
/// rather than one formula on two inputs.
///
/// `diag`, when given, is the `2·n_free` diagonal objectives already evaluated at this `eps` —
/// [`FdStep::diag`], through [`fd_ofv_hessian`] — and only the off-diagonal points are
/// evaluated here (#1935). `None` evaluates every point.
///
/// Returns `None` when the step was cancelled mid-stencil.
///
/// #256: flattened to one `par_iter` over all ~2·n_free² perturbed OFV points (each point's
/// EBE re-solve iterated serially over subjects inside `ofv`) instead of a serial loop firing
/// a per-subject `par_iter` at every point — removing the fork/join overhead of a rayon
/// barrier per point. The scoring pass after the re-solve is not covered by that: it keeps
/// `pop_nll_opts`'s own subject `par_iter` (always for AGQ, size-gated otherwise), which nests
/// inside this loop — see `reconverge_population`.
pub(super) fn fd_ofv_stencil<F: Fn(&[f64]) -> f64 + Sync>(
    n: usize,
    x_hat: &[f64],
    free_idx: &[usize],
    floor: &[f64],
    eps: f64,
    f0: f64,
    diag: Option<&[f64]>,
    ofv: &F,
    options: &FitOptions,
) -> Option<FdStencil> {
    let nf = free_idx.len();
    let hsteps: Vec<f64> = free_idx
        .iter()
        .map(|&i| fd_step(eps, x_hat[i], floor[i]))
        .collect();
    // Flat list of perturbation SPECS (not materialised x-vectors): 2 per diagonal (±hᵢ), then
    // 4 per (a<b) off-diagonal pair. Each par_iter task clones `x_hat` once and applies its
    // spec, so only ~n_threads vectors are live at a time instead of all ~2·nf² perturbed
    // points held resident for the whole reduction (the pre-#298 O(nf²·np) footprint) (#298).
    let mut specs: Vec<FdPert> = Vec::with_capacity(2 * nf + 2 * nf * nf);
    specs.extend(fd_diag_specs(x_hat, free_idx, eps, floor));
    let n_diag = specs.len();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..nf {
        for b in (a + 1)..nf {
            let (i, j) = (free_idx[a], free_idx[b]);
            let (hi, hj) = (hsteps[a], hsteps[b]);
            for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)] {
                specs.push(FdPert::Pair {
                    i,
                    di: si * hi,
                    j,
                    dj: sj * hj,
                });
            }
            pairs.push((a, b));
        }
    }
    // #1935: the diagonal points are the step probe's accepted pass when the caller has it, so
    // only the off-diagonal specs are evaluated here. Same specs at the same `eps` through the
    // same `eval_fd_point`, and `find_ebe` is deterministic, so this is the value the stencil
    // would have computed — skipped, not approximated.
    let skip = match diag {
        Some(d) => {
            assert_eq!(
                d.len(),
                n_diag,
                "fd_ofv_stencil: reused diagonal has {} points, the stencil {}",
                d.len(),
                n_diag
            );
            n_diag
        }
        None => 0,
    };
    let report = cov_progress("Hessian", specs.len() - skip, options.verbose);
    let mut vals: Vec<f64> = Vec::with_capacity(specs.len());
    if let Some(d) = diag {
        vals.extend_from_slice(d);
    }
    vals.par_extend(specs[skip..].par_iter().map(|&p| {
        // Cooperative cancel: skip this point's EBE reconvergence and
        // return NaN so the queue drains; bailed on below.
        if crate::cancel::is_cancelled(&options.cancel) {
            report();
            return f64::NAN;
        }
        let v = eval_fd_point(x_hat, p, ofv);
        report();
        v
    }));
    if crate::cancel::is_cancelled(&options.cancel) {
        return None;
    }
    let mut hess = DMatrix::zeros(n, n);
    let mut diag_nan: HashSet<usize> = HashSet::new();
    let mut offdiag_nan: HashSet<usize> = HashSet::new();
    // Diagonal: (f(x+h) − 2f(x) + f(x−h)) / h².
    for a in 0..nf {
        let i = free_idx[a];
        let hi = hsteps[a];
        let h_ii = (vals[2 * a] - 2.0 * f0 + vals[2 * a + 1]) / (hi * hi);
        if h_ii.is_finite() {
            hess[(i, i)] = h_ii;
        } else {
            diag_nan.insert(i);
        }
    }
    // Off-diagonal: (f++ − f+− − f−+ + f−−) / (4 hᵢ hⱼ).
    let mut off = n_diag;
    for &(a, b) in &pairs {
        let (i, j) = (free_idx[a], free_idx[b]);
        let (hi, hj) = (hsteps[a], hsteps[b]);
        let (fpp, fpm, fmm, fmp) = (vals[off], vals[off + 1], vals[off + 2], vals[off + 3]);
        off += 4;
        let h_ij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
        if h_ij.is_finite() {
            hess[(i, j)] = h_ij;
            hess[(j, i)] = h_ij;
        } else {
            offdiag_nan.insert(i);
            offdiag_nan.insert(j);
        }
    }
    Some(FdStencil {
        hess,
        diag_nan,
        offdiag_nan,
    })
}

/// The FD-of-OFV Hessian as the covariance step builds it: [`select_fd_step`] sizes the step,
/// then [`fd_ofv_stencil`] differences at it, taking the probe's accepted pass as its diagonal.
///
/// One function for the composition, so the reuse of #1935 is a property of the route rather
/// than of each call site: without halving the step costs `2·n_free²` objectives (it was
/// `2·n_free + 2·n_free²`), and each failed halving pass adds `2·n_free`. Every halving pass
/// but the last is still thrown away, since its points are at a step the stencil does not use.
///
/// Returns `None` when the step was cancelled, in the probe or the stencil.
pub(super) fn fd_ofv_hessian<F: Fn(&[f64]) -> f64 + Sync>(
    n: usize,
    x_hat: &[f64],
    free_idx: &[usize],
    floor: &[f64],
    initial_eps: f64,
    f0: f64,
    ofv: &F,
    options: &FitOptions,
) -> Option<FdStencil> {
    #[cfg(test)]
    reconverge_log::record_fd_hessian(initial_eps);
    // Adaptively select the FD step: halve up to 8× until all free-parameter diagonal stencils
    // are finite. Most models use the initial step; halving only kicks in when the OFV
    // overflows at the default perturbation size.
    let step = select_fd_step(x_hat, free_idx, floor, initial_eps, f0, ofv, options);
    if crate::cancel::is_cancelled(&options.cancel) {
        return None;
    }
    if options.verbose && step.n_halvings > 0 {
        eprintln!(
            "  [covariance] Adaptive FD step: reduced {:.3e} → {:.3e} ({} halving{})",
            initial_eps,
            step.eps,
            step.n_halvings,
            if step.n_halvings == 1 { "" } else { "s" }
        );
    }
    fd_ofv_stencil(
        n,
        x_hat,
        free_idx,
        floor,
        step.eps,
        f0,
        step.diag.as_deref(),
        ofv,
        options,
    )
}

/// Test-only record of every [`reconverge_population`] call (#1935), written by
/// [`map_subjects`] from the branch it actually takes: one population objective per call, with
/// the population it covered and whether its subjects were re-solved in parallel. Process-global, because the calls run on rayon workers; tests run concurrently,
/// so a test tags its subjects' ids with a unique prefix and reads back only its own entries.
#[cfg(test)]
pub(super) mod reconverge_log {
    use crate::types::Population;
    use std::sync::Mutex;

    /// `(first subject id, subject count, parallel_subjects)` per call.
    static CALLS: Mutex<Vec<(String, usize, bool)>> = Mutex::new(Vec::new());

    pub(in crate::estimation) fn record(pop: &Population, parallel_subjects: bool) {
        let id = pop
            .subjects
            .first()
            .map(|s| s.id.clone())
            .unwrap_or_default();
        CALLS
            .lock()
            .unwrap()
            .push((id, pop.subjects.len(), parallel_subjects));
    }

    /// `initial_eps` of every [`super::fd_ofv_hessian`] call: the witness that a covariance
    /// step sized its FD step through the probe rather than going straight to the stencil.
    /// Keyed by the step value, which a test makes unique to itself.
    static FD_HESSIAN_CALLS: Mutex<Vec<f64>> = Mutex::new(Vec::new());

    pub(in crate::estimation) fn record_fd_hessian(initial_eps: f64) {
        FD_HESSIAN_CALLS.lock().unwrap().push(initial_eps);
    }

    /// Remove and count the `fd_ofv_hessian` calls made with exactly `initial_eps`.
    pub(in crate::estimation) fn take_fd_hessian(initial_eps: f64) -> usize {
        let mut calls = FD_HESSIAN_CALLS.lock().unwrap();
        let before = calls.len();
        calls.retain(|&e| e != initial_eps);
        before - calls.len()
    }

    /// Remove and return `(subject count, parallel_subjects)` for every call whose first
    /// subject id starts with `prefix`.
    pub(in crate::estimation) fn take(prefix: &str) -> Vec<(usize, bool)> {
        let mut calls = CALLS.lock().unwrap();
        let (mine, rest): (Vec<_>, Vec<_>) = calls
            .drain(..)
            .partition(|(id, _, _)| id.starts_with(prefix));
        *calls = rest;
        mine.into_iter().map(|(_, n, p)| (n, p)).collect()
    }
}

/// `f(i)` for every subject index of `pop`, in subject order — in parallel or serially, as
/// `parallel` says. The subject loop of [`reconverge_population`], kept separate so the
/// dispatch and its `cfg(test)` log entry are one place: the log records the branch this
/// function takes, and a test witnesses that the parallel branch really runs concurrently.
fn map_subjects<T: Send, F: Fn(usize) -> T + Sync>(
    pop: &Population,
    parallel: bool,
    f: F,
) -> Vec<T> {
    let n = pop.subjects.len();
    if parallel {
        #[cfg(test)]
        reconverge_log::record(pop, true);
        (0..n).into_par_iter().map(&f).collect()
    } else {
        #[cfg(test)]
        reconverge_log::record(pop, false);
        (0..n).map(f).collect()
    }
}

/// Re-solve the inner EBE loop for every subject of `pop` at the packed point `xv`,
/// warm-started from `warm[i]`.
///
/// NONMEM reconverges the conditional estimates at every perturbed point in its covariance
/// step; holding η̂/H fixed gives a Hessian with the wrong curvature — indefinite even on
/// warfarin, which previously forced eigenvalue clipping (#129) and inflated the SEs.
///
/// `parallel_subjects` picks the loop for the EBE re-solve, the dominant cost of a point.
/// Inside the stencil and the step probe it is `false`: those parallelise over perturbed
/// POINTS, and a per-point subject fork/join is what #256 removed. At a single point evaluated
/// on its own — the base reconvergence at `x̂` and the salvage route's subset base — there is
/// no point loop to parallelise, so it is `true`; before #1935 those ran on one thread and
/// were over half of a 16-thread covariance step. `find_ebe` is deterministic per subject and
/// `collect()` keeps subject order, so both loops return the same EBEs bit for bit and
/// `pop_nll`'s summation order is unchanged (#703).
///
/// This flag does not reach the scoring pass that follows: `pop_nll_opts` keeps its own
/// subject `par_iter` (size-gated by `parallelize_cheap_subject_pass` for FOCE/FOCEI,
/// unconditional for AGQ), so at a point inside the stencil or probe that pass still nests in
/// the point loop. That is the stencil's arrangement since #256, which the probe now shares;
/// rayon's work-stealing runs a nested `par_iter` inline when no worker is idle.
///
/// Under `cfg(test)` every call is logged (`reconverge_log`), so a test can pin how many
/// population objectives a covariance step ran and which of them reconverged in parallel.
///
/// The covariance step reconverges at its own tolerance (`cov_inner_tol`), decoupled from the
/// fit's `inner_tol`: the second-difference-of-OFV R-matrix is far more sensitive to EBE
/// precision than the fit, so LTBS tightens it by default (the `g = ln(f)` Hessian needs it)
/// and any model can opt in. Defaults to `inner_tol` for non-LTBS (byte-identical). See
/// [`FitOptions::effective_cov_inner_tol`].
///
/// **`pop` is a parameter, not `self`'s population**, because the per-subject salvage (#1514)
/// runs this same reconvergence over the declined subjects alone. `warm` is indexed
/// positionally against `pop.subjects`, so a subset must carry the matching subset of modes —
/// and because the warm start is the *fit's* modes rather than the previous perturbed point,
/// `x ↦ η̂ᵢ(x)` is the same deterministic map on either set. Had it been path-dependent, the
/// population stencil could not have been decomposed per subject at all.
#[allow(clippy::too_many_arguments)]
pub(super) fn reconverge_population(
    xv: &[f64],
    model: &CompiledModel,
    pop: &Population,
    template: &ModelParameters,
    warm: &[DVector<f64>],
    options: &FitOptions,
    cov_inner_tol: f64,
    parallel_subjects: bool,
) -> (
    ModelParameters,
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    Vec<Vec<DVector<f64>>>,
) {
    let (params, ehs, hms, kaps, _) = reconverge_population_checked(
        xv,
        model,
        pop,
        template,
        warm,
        options,
        cov_inner_tol,
        parallel_subjects,
    );
    (params, ehs, hms, kaps)
}

/// [`reconverge_population`], and whether every subject's inner solve converged.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn reconverge_population_checked(
    xv: &[f64],
    model: &CompiledModel,
    pop: &Population,
    template: &ModelParameters,
    warm: &[DVector<f64>],
    options: &FitOptions,
    cov_inner_tol: f64,
    parallel_subjects: bool,
) -> (
    ModelParameters,
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    Vec<Vec<DVector<f64>>>,
    bool,
) {
    let params = unpack_params(xv, template);
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let ebes = map_subjects(pop, parallel_subjects, |i: usize| {
        find_ebe_for_stage(
            model,
            &pop.subjects[i],
            &params,
            options.inner_maxiter,
            cov_inner_tol,
            Some(warm[i].as_slice()),
            Some(&mu_k),
            0,
            options,
        )
    });
    let n = pop.subjects.len();
    let mut ehs = Vec::with_capacity(n);
    let mut hms = Vec::with_capacity(n);
    let mut kaps = Vec::with_capacity(n);
    let mut all_converged = true;
    for ebe in ebes {
        all_converged &= ebe.converged;
        ehs.push(ebe.eta);
        hms.push(ebe.h_matrix);
        kaps.push(ebe.kappas);
    }
    (params, ehs, hms, kaps, all_converged)
}

/// [`compute_covariance_for`] for an estimate the step's own objective produced — the form
/// every unit test that drives the step directly wants. Production goes through
/// [`run_covariance_step_inner`], which names the estimator explicitly.
#[cfg(test)]
pub(crate) fn compute_covariance(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> CovarianceStepResult {
    compute_covariance_for(
        options.method,
        x_hat,
        template,
        model,
        population,
        eta_hats,
        h_matrices,
        kappas,
        options,
    )
}

/// The covariance step for an estimate produced by `fit_method`, which may differ from
/// `options.method` — the method whose objective the step differentiates. They differ exactly
/// on the #1781 quadrature route, where a SAEM / IMP / IMPMAP estimate is scored on the
/// `method = laplace, n_agq = n` marginal; `fit_method` is what decides whether that estimate
/// minimised the objective ([`r_falls_back_to_s`]).
///
/// Every matrix of one step runs at **one** ODE tolerance (#520 review round 6): `S`, each
/// analytic and each finite-differenced term of `R`, and the modes, `H` and κ̂ they are built
/// at. Mixing tolerances would combine the derivatives of two different objectives into a
/// matrix that is the derivative of neither. Which tolerance:
/// - the fit's own, when [`cov_ode_override`] has nothing to tighten (no integrator, or one
///   already on the plateau), and when `R` is the exact analytic R-matrix for every subject
///   and the estimator reads `R` — no finite difference of the objective carries the
///   curvature, so `rsr`'s `S` stays with the `R` it is paired with;
/// - otherwise the plateau, for the whole step ([`tightened_covariance_step`]): the FD and
///   hybrid `R` routes, under `r` and `rsr` alike, and `s`, whose `S` is finite-differenced on
///   its own.
///
/// `R`'s route is known only once the analytic assembly has run, so a step expected to be
/// fully analytic ([`predicted_r_route`]) runs at the fit's tolerance first and is abandoned
/// right after the assembly — before `S` or any stencil — if a subject declines at run time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_covariance_for(
    fit_method: crate::types::EstimationMethod,
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> CovarianceStepResult {
    let at_fit_tolerance = |require_analytic_r| {
        covariance_step(
            fit_method,
            x_hat,
            template,
            model,
            population,
            eta_hats,
            h_matrices,
            kappas,
            options,
            require_analytic_r,
        )
    };
    let Some(tight) = cov_ode_override(model, options) else {
        return at_fit_tolerance(false).expect("a step that may take any route always completes");
    };
    if needs_r(options.covariance_method)
        && predicted_r_route(model, population, options, template.mixture.is_some())
            == PredictedRRoute::Analytic
    {
        if let Some(out) = at_fit_tolerance(true) {
            return out;
        }
    }
    tightened_covariance_step(
        fit_method, x_hat, template, model, population, eta_hats, h_matrices, kappas, options,
        tight,
    )
}

/// [`compute_covariance_for`]'s step run whole under the covariance ODE override `tight`
/// (#520): one scope around everything the step solves, so no part of it can run at the fit's
/// tolerance. Its first solve re-finds the modes, `H` and κ̂ from the fit's modes at the
/// tighter tolerance, and those replace the fit's everywhere the step reads them — `S` is
/// built at them, the stencil warm-starts from them — so the step is exactly the one a fit
/// pinned to the plateau would run from those modes. A mixture's step reads no single-
/// population mode (`S` and the K-fold objective re-find their own per class), so it keeps
/// the fit's.
///
/// A scope that cannot be built (a pool that cannot be leased) runs the step at the fit's
/// tolerance with the fit's modes — never without a covariance step, and never silently:
/// [`stencil_scope_failure_warning`] leads a successful step's warnings.
#[allow(clippy::too_many_arguments)]
fn tightened_covariance_step(
    fit_method: crate::types::EstimationMethod,
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
    tight: crate::ode::solver::OdeSolverOverride,
) -> CovarianceStepResult {
    let step =
        |eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>]| {
            covariance_step(
                fit_method, x_hat, template, model, population, eta_hats, h_matrices, kappas,
                options, false,
            )
            .expect("a step that may take any route always completes")
        };
    let run = || {
        // Test-only: the stencil counts its objectives on the thread it runs on, which is a
        // worker of the scope's pool; hand the count back to the caller's.
        #[cfg(test)]
        let evals_before = FD_OBJECTIVE_EVALS.with(|c| c.get());
        let out = if template.mixture.is_some() {
            step(eta_hats, h_matrices, kappas)
        } else {
            let (_, e, h, k) = reconverge_population(
                x_hat,
                model,
                population,
                template,
                eta_hats,
                options,
                options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner()),
                true,
            );
            step(&e, &h, &k)
        };
        // ...and take it back off the worker, so the count leaves this thread only through
        // the return value. Under a one-thread pool (CI's `RAYON_NUM_THREADS=1`) two tests
        // running this step on the same cached scope pool share the worker, which steals the
        // other test's `run` while this one waits in a `join`; without the reset that run's
        // stencil stayed on the worker's counter inside this run's window and was counted
        // twice — `100` objectives against a `50`-point stencil, on two unrelated PRs.
        // Restoring nests correctly because a stolen job completes before this frame resumes.
        #[cfg(test)]
        let out = (
            out,
            FD_OBJECTIVE_EVALS.with(|c| c.replace(evals_before)) - evals_before,
        );
        out
    };
    match crate::api::with_fit_scope_ode(options, tight, run) {
        #[cfg(test)]
        Ok((out, evals)) => {
            FD_OBJECTIVE_EVALS.with(|c| c.set(c.get() + evals));
            out
        }
        #[cfg(not(test))]
        Ok(out) => out,
        Err(e) => note_scope_failure(step(eta_hats, h_matrices, kappas), &e.to_string()),
    }
}

/// `out`, a step run at the fit's tolerance because the tightened scope could not be built
/// (`reason`), saying so whatever it returned. A success leads its warnings with
/// [`stencil_scope_failure_warning`]; a failure carries a single message, so the note is
/// appended to it in prose — without the `W_` token, which would reclassify the failure as a
/// `covariance_step` note (#520 review round 7).
fn note_scope_failure(mut out: CovarianceStepResult, reason: &str) -> CovarianceStepResult {
    match &mut out {
        CovarianceStepResult::Success(o) => {
            o.warnings.insert(0, stencil_scope_failure_warning(reason))
        }
        CovarianceStepResult::Unusable(msg)
        | CovarianceStepResult::FailedNonPd { reason: msg, .. } => {
            use crate::estimation::cov_diagnostics::{
                COV_FD_PLATEAU_ABSTOL, COV_FD_PLATEAU_RELTOL,
            };
            msg.push_str(&format!(
                " The step ran at the fit's own ODE tolerance: it could not be run at \
                 ode_reltol = {COV_FD_PLATEAU_RELTOL:.0e} / ode_abstol = \
                 {COV_FD_PLATEAU_ABSTOL:.0e} ({reason})."
            ));
        }
    }
    out
}

/// The failure naming every coordinate whose Hessian diagonal is unusable. Each entry already
/// closes its own parenthesis, so the entries are joined with a bare `; `.
fn ill_conditioned_entries_reason(problem_params: &[String]) -> String {
    format!(
        "Covariance step failed: Hessian has ill-conditioned entries for the following \
         parameter(s) — {}. SE estimates not available.",
        problem_params.join("; ")
    )
}

/// One covariance step at the ODE tolerance armed on the calling thread — see
/// [`compute_covariance_for`], which chooses it. `None` only under `require_analytic_r`, when
/// the analytic assembly did not cover every subject: the step stops right after it, before
/// `S` or any stencil is built.
#[allow(clippy::too_many_arguments)]
fn covariance_step(
    fit_method: crate::types::EstimationMethod,
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
    require_analytic_r: bool,
) -> Option<CovarianceStepResult> {
    let n = x_hat.len();
    let initial_eps = options.fd_hessian_step;
    if initial_eps <= 0.0 || !initial_eps.is_finite() {
        return Some(CovarianceStepResult::Unusable(format!(
            "Covariance step failed: fd_hessian_step must be positive and finite, got {}. \
             SE estimates not available.",
            initial_eps
        )));
    }
    // One walk for the box and the FIX mask (#1252) — `compute_bounds` builds
    // the mask internally to pin FIX-ed coordinates, so taking it here costs
    // nothing and saves the second walk this function used to do further down.
    // The packed start is dropped: `x_hat` is the caller's, not the template's.
    let PackedStart {
        bounds,
        fixed: fixed_mask,
        ..
    } = pack_with_bounds(template);
    // Mixture models (#983 Phase 6) build the FD Hessian on the K-fold mixture
    // objective and skip the single-population reconvergence / analytic R-matrix
    // (both single-population-only). Gated on `template.mixture`.
    let is_mixture = template.mixture.is_some();
    // #1352: the covariate-NN regularizer, when the estimator that produced `x_hat` minimised
    // it (`None` for an unregularized fit, and for SAEM / IMP / IMPMAP / Bayes, which never
    // apply the penalty, so their SEs are the curvature of what they fitted).
    let nn_reg = cov_nn_regularizer(fit_method, model, population, options);

    // `h_matrices` (the H from the fit) is read only by `S`
    // (`assemble_score_cross_product`): `R` reconverges the EBEs at every perturbed
    // point and recomputes H there.

    let cov_inner_tol = options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());

    // Every reconverged point starts at the fit's joint mode, κ̂ included, so an IOV
    // subject does not re-solve its κ from 0 at each of the stencil's points (#2261).
    // `fit()` and `run_covariance` hand the same η̂ and κ̂, so the repeat stays bit for bit.
    // Unlike the FD gradient stencils, the points keep `cov_inner_tol` rather than the
    // stencil tolerance (`STENCIL_INNER_TOL`): measured on `nested_levels_3` and
    // `mbma_placebo`, tightening them moved no standard error by more than 1e-9 relative
    // beyond what the joint start alone did, at up to 2.3× the inner solves (#2261), and
    // tightening every model's covariance reconvergence is what #486 declined.
    let warm = crate::estimation::outer_optimizer::joint_warm_starts(eta_hats, kappas);

    // The covariance OFV = −2·logL over an arbitrary subject set, reconverged at `xv`.
    //
    // One implementation for both consumers — the whole-population stencil and the per-subject
    // salvage of #1514 — so subject `i`'s term is the same number whichever route computes it,
    // and the base-OFV evaluation cannot drift from the stencil's (#298).
    //
    // Covariance OFV = −2·logL = 2·pop_nll for both FOCE and FOCEI.
    //
    // FOCE uses the Sheiner–Beal linearised marginal `(y−f₀)ᵀR̃⁻¹(y−f₀) + log|R̃|` with
    // R̃ = HΩHᵀ + R. By Woodbury that marginal *already* carries the Ω penalty (it equals the
    // conditional form including η̂ᵀΩ⁻¹η̂ + log|Ω|), so its Ω-curvature is complete. An earlier
    // version added the η̂ᵀΩ⁻¹η̂ + log|Ω| prior here for the FOCE branch, which double-counted Ω
    // and flattened the Ω-block curvature — the source of the ~31%-low FOCE omega SEs (issue
    // #243). FOCEI's Almquist–Laplace marginal likewise carries the prior internally. So
    // neither method needs an add-back.
    //
    // `parallel_subjects`: `true` only for a point evaluated on its own (see
    // `reconverge_population`); the stencil and probe call it `false` from inside their point
    // loops.
    let subset_cov_ofv =
        |xv: &[f64], pop: &Population, warm: &[DVector<f64>], parallel_subjects: bool| -> f64 {
            let (params, ehs, hms, kaps) = reconverge_population(
                xv,
                model,
                pop,
                template,
                warm,
                options,
                cov_inner_tol,
                parallel_subjects,
            );
            2.0 * pop_nll_opts(model, pop, &params, &ehs, &hms, &kaps, options)
        };

    // Covariance OFV = −2·logL at a reconverged point. For FOCEI the per-subject
    // marginal already carries ηᵀΩ⁻¹η + log|Ω|; for FOCE we add that prior here.
    //
    // Mixture (#983 Phase 6): the FD-of-OFV Hessian must be built on the K-fold
    // mixture objective (`mixture_ofv`), not the single-population marginal — the
    // per-class Ω/Σ overrides and mixing-logit thetas only enter through it.
    // `mixture_ofv.ofv` is already on the −2·logL scale (= −2 Σᵢ log Σₖ pᵢₖ e^{−nllᵢₖ}),
    // so it takes no ×2, and it reconverges every (subject × class) inner solve
    // internally (cold — the FD stencil favours correctness over a warm start).
    // Mixture: reconverge its per-class EBEs at `cov_inner_tol` too, not the fit's
    // `inner_tol` — otherwise `cov_inner_tol` is a silent no-op on the mixture path
    // (a loose fit + tight `cov_inner_tol` for trustworthy SEs would get bit-
    // identical contaminated numbers), exactly the trap the single-population
    // reconvergence above avoids. `mixture_ofv` reads its inner tolerance from the
    // options it is handed, so pass an override clone.
    let cov_options;
    let mixture_cov_options = if is_mixture && cov_inner_tol != options.inner_tol {
        cov_options = FitOptions {
            inner_tol: cov_inner_tol,
            ..options.clone()
        };
        &cov_options
    } else {
        options
    };
    let cov_ofv = |xv: &[f64]| -> f64 {
        if is_mixture {
            let params = unpack_params(xv, template);
            return crate::estimation::mixture::mixture_ofv(
                model,
                population,
                &params,
                mixture_cov_options,
                None,
            )
            .ofv;
        }
        subset_cov_ofv(xv, population, &warm, false)
    };

    // Reconverge once at `x_hat` and keep the result. Both consumers need it: the base OFV
    // for the FD stencil, and — as of #953 review finding 8 — the analytic path's η̂.
    //
    // The analytic assembly is derived under stationarity (`∂lᵢ/∂η|_η̂ = 0`): both the M2
    // envelope term `−M_ξᵀH⁻¹M_ζ` and the adjoint's mode response (`η′ = −H⁻¹∂ₓg`) assume
    // it. Feeding it the modes converged during the fit at `inner_tol` would make
    // `cov_inner_tol` a silent
    // no-op on this path, so a user who fits loose and then sets `cov_inner_tol = 1e-10`
    // for trustworthy SEs would get bit-identical contaminated numbers with no diagnostic.
    // Reconverging here costs one population inner solve — the thing the FD stencil pays
    // `~2·n_free²` times.
    // For a mixture the base OFV comes straight from `cov_ofv` (the K-fold
    // objective, reconverged per class internally); the single-population
    // reconvergence and its EBEs/H are unused (the analytic path is off), so feed
    // empty vecs. `base_params` is still needed for the non-finite diagnostic.
    let (base_params, base_eta_hats, base_h_matrices, base_kappas, base_ofv) = if is_mixture {
        (
            unpack_params(x_hat, template),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            cov_ofv(x_hat),
        )
    } else {
        let (p, e, h, k) = reconverge_population(
            x_hat,
            model,
            population,
            template,
            &warm,
            options,
            cov_inner_tol,
            true,
        );
        let o = 2.0 * pop_nll_opts(model, population, &p, &e, &h, &k, options);
        (p, e, h, k, o)
    };
    let _ = (&base_params, &base_h_matrices, &base_kappas);
    if !base_ofv.is_finite() {
        // Diagnose: check Omega conditioning to distinguish Omega collapse from
        // a model-evaluation overflow/underflow. For a mixture the base (class-1)
        // Omega can be well-conditioned while a per-class `omega(k)` override has
        // collapsed — the actual cause of the non-finite OFV — so inspect the
        // worst-conditioned Omega across all classes, not just the base.
        let params_at = unpack_params(x_hat, template);
        let reason = match extract_eigenvalues(diagnostic_omega(&params_at)) {
            Some(ref ev) if ev.last().copied().unwrap_or(1.0) <= 1e-8 => {
                let min_eig = ev.last().copied().unwrap_or(f64::NAN);
                // Distinguish truly negative eigenvalues from tiny-positive (near-singular).
                let descriptor = if min_eig < 0.0 {
                    "not positive definite"
                } else {
                    "near-singular"
                };
                format!(
                    "Covariance step failed: Omega matrix is {} at convergence \
                     (min eigenvalue = {}; eigenvalues: [{}]). \
                     SE estimates not available.",
                    descriptor,
                    fmt_eig(min_eig),
                    ev.iter()
                        .map(|&v| fmt_eig(v))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            _ => "Covariance step failed: base OFV is non-finite at convergence \
                  (likely numerical overflow or underflow in model evaluation). \
                  SE estimates not available."
                .to_string(),
        };
        if options.verbose {
            eprintln!("  {}", reason);
        }
        return Some(CovarianceStepResult::Unusable(reason));
    }

    // FIX parameters contribute no information — skip their FD stencils and,
    // after inverting the Hessian of the free block, leave their covariance
    // rows/cols at zero (→ SE = 0 downstream). `fixed_mask` came from the same
    // `pack_with_bounds` walk as `bounds`, at the top of this function.
    // It also holds the structural-zero Ω off-diagonals (the cross-block
    // elements of a mixed block+diagonal Ω, where `free_mask[(i,j)] == false`):
    // they are not estimated parameters and their Hessian diagonal is flat, so
    // without the exclusion the ill-conditioning guard below rejects the entire
    // covariance step (#243). Since #1018 the optimizer holds them through the
    // same mask, so there is one gate, not a second structural filter here.
    let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed_mask[i]).collect();
    let n_free = free_idx.len();
    // The free coordinates whose units the eigenvalue floor on `S` and `R` must not see (#2145).
    let unit_coords =
        free_coordinate_units(template, x_hat, &free_idx, &model.nn_weight_theta_ranges());
    if n_free == 0 {
        // Nothing to estimate — return an all-zero covariance so downstream
        // SE extraction reports zeros (all params FIX). Ahead of `S` and `R` alike: neither
        // has a coordinate to assemble.
        return Some(CovarianceStepResult::Success(CovarianceOutput {
            matrix: DMatrix::zeros(n, n),
            warnings: vec![],
            method: options.covariance_method,
        }));
    }

    // The score cross-product `S = Σᵢ gᵢgᵢᵀ` over the free block, at the step's modes — the
    // fit's own, or on a tightened step the modes reconverged at its tolerance
    // ([`tightened_covariance_step`]). `None` when a cancel landed during the per-subject loop,
    // whose placeholder rows are then never trusted.
    let score_cross_product = || {
        let s = assemble_score_cross_product(
            x_hat, template, model, population, eta_hats, h_matrices, kappas, &bounds, options,
            &free_idx,
        );
        (!crate::cancel::is_cancelled(&options.cancel)).then_some(s)
    };

    // #1803: `covariance_method = s` is `S⁻¹` and nothing else — the estimator reads no `R`
    // ([`needs_r`]), as NONMEM's `$COV MATRIX=S` needs only `S`. So `R` is not built: before
    // this the step ran the whole stencil below (`2·n²` reconverged population
    // objectives on the FD route), inverted it, discarded it, and let its vetoes refuse an `S⁻¹`
    // that existed. Nothing of `R` was published either way — no warning about it is emitted
    // under `s` — and a requested `s` never takes the #1781 fallback (`r_falls_back_to_s`), so
    // skipping `R` loses no decision. A parameter the objective does not depend on still fails:
    // its score column is zero, which `invert_cross_product` rejects.
    if !needs_r(options.covariance_method) {
        let Some(s_free) = score_cross_product() else {
            return Some(CovarianceStepResult::Unusable(
                COV_CANCELLED_MSG.to_string(),
            ));
        };
        let s_free = match s_free {
            Ok(s) => s,
            Err(reason) => return Some(CovarianceStepResult::Unusable(reason)),
        };
        let Some(cov_free) = invert_cross_product(&s_free, &unit_coords) else {
            return Some(CovarianceStepResult::Unusable(S_SINGULAR_MSG.to_string()));
        };
        if options.verbose {
            eprintln!("  Covariance step successful");
        }
        return Some(CovarianceStepResult::Success(CovarianceOutput {
            matrix: embed_free(&cov_free, &free_idx, n),
            warnings: nn_reg
                .is_some()
                .then(|| format_nn_penalty_cov_note(CovarianceMethod::CrossProduct))
                .into_iter()
                .collect(),
            method: CovarianceMethod::CrossProduct,
        }));
    }

    // ── The exact analytic R-matrix (#436), attempted before the FD stencil ─────────────
    //
    // Assembled per subject, and no longer all-or-nothing (#1514). The observed information is
    // `Σᵢ Rᵢ`; a subject outside the analytic scope contributes one term of that sum, and that
    // term can be second-differenced from the subject's *own* marginal — same objective, same
    // point, same warm start — while every other subject keeps its exact term. Before #1514 one
    // such subject dropped the whole population onto the `2·n_free²`-point population stencil:
    // on the motivating fit, one subject of 55 turning a 2.0 s covariance step into 23.4 s.
    //
    // This is not the stencil #639 removed. That one finite-differenced a gradient that held
    // `a = ∂f/∂η` fixed — an envelope approximation, which is why it biased weakly-identified
    // structural SEs. This is the exact second derivative of the same marginal the outer loop
    // minimises.
    //
    // The attempt is made **before** `select_fd_step`, not after. That step probes `2·n_free`
    // perturbed points, each of which reconverges every subject's inner loop, purely to size a
    // finite-difference step a fully-analytic route never uses. Running it first would have
    // left the analytic path paying `2·n_free` reconverged population objectives for nothing —
    // most of the cost it exists to remove, and a claim of "no inner re-solve" that the code
    // did not honour. On the salvage route the probe still runs, but over the declined subjects
    // only, which is the same reduction the stencil itself gets.
    let assembly = if options.analytic_cov_hessian && is_mixture {
        // The mixture objective is not a per-subject sum of single-population marginals, so
        // it has its own exact assembly, all or nothing (`mixture_cov_hessian`).
        if crate::sens::ode_provider::adjoint::recorder_in_scope(model)
            && !crate::types::GradientMethod::forced_fd(model)
        {
            match crate::estimation::mixture::mixture_cov_hessian(
                model,
                population,
                template,
                x_hat,
                options,
                cov_objective(options),
            ) {
                Some(h) => AnalyticCovAssembly::Full(h),
                None => AnalyticCovAssembly::Unavailable,
            }
        } else {
            AnalyticCovAssembly::Unavailable
        }
    } else if options.analytic_cov_hessian {
        // `base_eta_hats`, not `eta_hats` — the modes reconverged at `cov_inner_tol`, which
        // is what the stationarity assumption in the assembly needs (see above).
        analytic_cov_assembly(
            model,
            population,
            template,
            x_hat,
            &base_eta_hats,
            &base_kappas,
            options,
        )
    } else {
        AnalyticCovAssembly::Unavailable
    };
    // A cancel during the analytic loop surfaces as `Unavailable`; without this the fallback
    // would start the *more* expensive FD stencil instead of stopping.
    if matches!(assembly, AnalyticCovAssembly::Unavailable)
        && crate::cancel::is_cancelled(&options.cancel)
    {
        return Some(CovarianceStepResult::Unusable(
            COV_CANCELLED_MSG.to_string(),
        ));
    }

    // A step run at the fit's tolerance only because `R` was expected fully analytic stops
    // here when it is not: its `S` and its stencil belong to the tightened step instead
    // ([`compute_covariance_for`], #520 review round 6). Before `S`, so nothing is wasted
    // but the assembly.
    if require_analytic_r && !matches!(assembly, AnalyticCovAssembly::Full(_)) {
        return None;
    }

    // `rsr` reads `S` too, and `S` costs one pass where `R`'s stencil costs `2·n²` objectives,
    // so `S` is assembled ahead of the stencil (after the analytic assembly, whose route decides
    // the step's ODE tolerance — #520) and a failed one stops the step before the stencil starts. A
    // singular `S` does not stop `rsr` — `R⁻¹SR⁻¹` never inverts it — so this changes what fails
    // first, never whether the step succeeds, except where a failed `R` carries a SIR proposal.
    // Two gates keep `S` after `R` there:
    // - the estimate did not minimise this objective: a failed `S` is then the #1781 fallback's
    //   `FailedNonPd`, whose proposal is built from `R`;
    // - a SIR fallback is armed (`covariance_fallback = sir` or `sir = true`): an `R` with no
    //   positive eigenvalue returns `FailedNonPd` with the |λ|-rectified proposal, and
    //   `should_run_sir_fallback` turns that into SIR intervals. Stopping on `S` first would
    //   return `Unusable` with no proposal, and the fit would get no intervals at all.
    let sir_fallback_armed = options.covariance_fallback == CovarianceFallback::Sir || options.sir;
    let s_first = if options.covariance_method == CovarianceMethod::Sandwich
        && estimate_minimises_cov_objective(fit_method)
        && !sir_fallback_armed
    {
        let Some(s_free) = score_cross_product() else {
            return Some(CovarianceStepResult::Unusable(
                COV_CANCELLED_MSG.to_string(),
            ));
        };
        match s_free {
            Ok(s) => Some(s),
            Err(reason) => return Some(CovarianceStepResult::Unusable(reason)),
        }
    } else {
        None
    };

    // Which subjects the objective stencil still has to cover.
    enum FdScope {
        /// The whole population, evaluated through `cov_ofv` — the only scope a mixture can
        /// take, since `mixture_ofv` is not a per-subject sum.
        Whole,
        /// Only the subjects the analytic assembly declined, carried as their own `Population`
        /// so `pop_nll_opts` sums exactly those terms and nothing else.
        Subset {
            pop: Population,
            warm: Vec<DVector<f64>>,
        },
    }

    // Population indices of the salvaged subjects, for the informational note. Empty on both
    // pure routes, and its emptiness is what makes that note's presence a statement.
    let mut salvaged: Vec<usize> = Vec::new();
    let (mut hess, fd_scope): (DMatrix<f64>, Option<FdScope>) = match assembly {
        AnalyticCovAssembly::Full(h) => {
            if options.verbose {
                eprintln!("  [covariance] analytic R-matrix (third-order sensitivities, #436)");
            }
            (h, None)
        }
        AnalyticCovAssembly::Partial { hess, declined } => {
            if options.verbose {
                eprintln!(
                    "  [covariance] analytic R-matrix for {} of {} subjects; {} \
                     finite-differenced from their own marginal (#1514)",
                    population.subjects.len() - declined.len(),
                    population.subjects.len(),
                    declined.len(),
                );
            }
            // The **fit's** modes, exactly what the whole-population stencil warm-starts from
            // — not `base_eta_hats`, which the analytic assembly needs for its stationarity
            // assumption. The two agree wherever the EBE is start-independent, which is why
            // swapping them here kills no test in this PR's mutation sweep (cell `M16`), and
            // the choice is not made on a measured difference: it is made so that subject `i`
            // enters `find_ebe` with the same warm start on both routes *by construction*,
            // rather than by an argument about how close two starts are. A start-dependent
            // subject is a real thing here — `W_EBE_START_DEPENDENT` exists — and on one of
            // those the salvaged term would otherwise stop being the term the population
            // stencil computes for it.
            let (pop, warm) = subset_population(population, &warm, &declined);
            let scope = FdScope::Subset { pop, warm };
            salvaged = declined;
            (hess, Some(scope))
        }
        AnalyticCovAssembly::Unavailable => (DMatrix::zeros(n, n), Some(FdScope::Whole)),
    };

    // Track FD failures at source so diagnostics name the right cause (a NaN/Inf
    // stencil result is not a genuine zero curvature). HashSet for O(1) ops.
    let mut fd_diag_nan: HashSet<usize> = HashSet::new();
    let mut fd_offdiag_nan: HashSet<usize> = HashSet::new();
    // The tolerances the stencil below integrated at, for the regularization message: the
    // fit's own, or the plateau the stencil was tightened to (#520). Read where it ran.
    let mut stencil_ode: Option<OdeToleranceFacts> = None;

    if let Some(scope) = fd_scope.as_ref() {
        // Test-only: every objective the probe and the stencil evaluate, counted across the
        // rayon workers and handed to the calling thread's `FD_OBJECTIVE_EVALS` below.
        #[cfg(test)]
        let fd_evals = std::sync::atomic::AtomicUsize::new(0);
        // The objective this stencil differences: the whole population's covariance OFV, or —
        // on the salvage route — only the declined subjects' share of it.
        let fd_ofv = |xv: &[f64]| -> f64 {
            #[cfg(test)]
            fd_evals.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match scope {
                FdScope::Whole => cov_ofv(xv),
                FdScope::Subset { pop, warm } => subset_cov_ofv(xv, pop, warm, false),
            }
        };
        // The stencil's base point must be evaluated on the same subject set as its perturbed
        // points: `(f₊ − 2f₀ + f₋)/h²` is the declined subjects' second difference only when
        // all three points are theirs. On the subset route this costs one extra inner solve
        // over those subjects — `find_ebe` is deterministic, so the value is exactly their
        // share of `base_ofv`, recomputed rather than carried because `pop_nll` keeps no
        // per-subject breakdown.
        let f0_fd = match scope {
            FdScope::Whole => base_ofv,
            FdScope::Subset { pop, warm } => subset_cov_ofv(x_hat, pop, warm, true),
        };

        let fd_floor = fd_step_floors(template, &model.default_params);
        let stencil = fd_ofv_hessian(
            n,
            x_hat,
            &free_idx,
            &fd_floor,
            initial_eps,
            f0_fd,
            &fd_ofv,
            options,
        );
        // Read here, on the thread the stencil ran on: inside a tightened step it is the
        // plateau's tolerance, not the fit's (#520).
        stencil_ode = OdeToleranceFacts::from_route(model, population);
        #[cfg(test)]
        FD_OBJECTIVE_EVALS
            .with(|c| c.set(c.get() + fd_evals.load(std::sync::atomic::Ordering::Relaxed)));
        let stencil = match stencil {
            Some(stencil) => stencil,
            None => {
                return Some(CovarianceStepResult::Unusable(
                    COV_CANCELLED_MSG.to_string(),
                ))
            }
        };
        // `+=`, not `copy_from`: on the salvage route `hess` already holds the analytic terms.
        // On the `Whole` route it is the zero matrix, so this is the previous assignment.
        hess += stencil.hess;
        fd_diag_nan = stencil.diag_nan;
        fd_offdiag_nan = stencil.offdiag_nan;
    }

    // Which mechanism produced the matrix, for every message below. Three routes, three
    // labels: no stencil ran at all; a stencil ran over the whole population; a stencil ran
    // over the salvaged minority while the rest was exact. Derived from what actually happened
    // rather than from `options.analytic_cov_hessian`, which says only what was asked for.
    let source = if fd_scope.is_none() {
        CovHessianSource::AnalyticRMatrix
    } else if salvaged.is_empty() {
        CovHessianSource::FdStencil
    } else {
        CovHessianSource::HybridRMatrix
    };

    // ── Parameter-prior curvature (#254) ─────────────────────────────────────
    //
    // The reported SE is the curvature of the objective that was *minimised*, and
    // under a prior that objective is `OFV_data + Σ((x−m)/s)²`. Leaving the prior
    // out here would report the unpenalized curvature — wrong in exactly the
    // regime the feature exists for, since a sparse fit's whole reason for
    // carrying a prior is that the data alone does not identify the direction,
    // and that is also where the unpenalized Hessian goes flat (rejected just
    // below as "zero diagonal — flat objective") or non-PD.
    //
    // Added here rather than inside `cov_ofv` for two reasons: the penalty's
    // second derivative is the exact constant `2/s²` (no stencil, no extra
    // objective evaluations, no FD noise), and adding it post-assembly covers the
    // analytic R-matrix route and the FD stencil route with one line instead of
    // one each. It lands before the ill-conditioning diagnosis on purpose — a
    // coordinate the prior identifies must read as curved, not as flat.
    //
    // Name it in the SE report, not just here: these are penalized-ML / MAP
    // standard errors, not posterior SDs.
    let cov_priors = crate::estimation::outer_optimizer::build_prior_set(model, template);
    cov_priors.add_hessian(&mut |i, j, v| hess[(i, j)] += v);

    // ── Covariate-NN regularization curvature (#1352) ────────────────────────
    //
    // The same argument as the prior's, for `nn_l2` / `nn_smooth`: the optimizer minimised
    // `OFV_data + penalty(θ)`, and the penalty exists for exactly the weight directions the
    // data does not identify — where the unpenalized Hessian is flat or indefinite, so leaving
    // it out floors those directions or sends the step to SIR on the fits the feature is for.
    // Added post-assembly for the prior's reasons too (one line covers the analytic, stencil
    // and hybrid routes, with no extra objective evaluation), and with the penalty's *exact*
    // Hessian rather than the Gauss–Newton one the optimizers use: this is the curvature
    // being reported, not a search direction. NN weights are identity-packed, so a theta
    // index is its packed index.
    if let Some(reg) = nn_reg.as_ref() {
        let theta = unpack_params(x_hat, template).theta;
        reg.add_exact_packed_hessian(&theta, &mut |i, j, v| hess[(i, j)] += v);
    }

    // Diagnose fatal Hessian problems. Use the FD-failure trackers for accurate cause labels:
    // a non-finite stencil result is never stored, so the entry keeps whatever was already
    // there — the zero initialisation on the population-stencil route, the in-scope subjects'
    // analytic term on the hybrid one — and a post-hoc check on `hess` would read that as
    // genuine curvature (or, on the FD route, as a flat objective) either way.
    let mut problem_params: Vec<String> = Vec::new();
    for &i in &free_idx {
        let diag = hess[(i, i)];
        if fd_diag_nan.contains(&i) {
            // The diagonal stencil overflowed, so this coordinate's curvature is incomplete
            // whatever `hess` now reads. Fatal on both routes — reported here and turned into
            // an `Unusable` below — because a diagonal is what every SE divides through.
            // Adjust fd_hessian_step or check for model overflow.
            problem_params.push(format!(
                "{} (FD stencil non-finite; model may overflow at perturbation — \
                 try tuning fd_hessian_step)",
                packed_param_label(i, template)
            ));
        } else if diag.abs() < 1e-30 {
            // Genuine flat objective: the FD stencil succeeded but returned ~0 curvature.
            problem_params.push(format!(
                "{} (zero diagonal — flat objective)",
                packed_param_label(i, template)
            ));
        }
    }

    if !problem_params.is_empty() {
        let reason = ill_conditioned_entries_reason(&problem_params);
        if options.verbose {
            eprintln!("  {}", reason);
        }
        return Some(CovarianceStepResult::Unusable(reason));
    }

    // Build the reduced Hessian over free indices, invert, then embed back
    // into the full n×n covariance matrix (FIX rows/cols stay zero).
    let mut hess_free = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            hess_free[(a, b)] = hess[(i, j)];
        }
    }
    let hess_free_sym = (&hess_free + hess_free.transpose()) * 0.5;

    let inv = match invert_psd_with_floor_unit_free(&hess_free_sym, &unit_coords) {
        Some(inv) => inv,
        None => {
            // `invert_psd_with_floor` returns None in two distinct cases, and we
            // must not conflate them: (a) every eigenvalue is finite but the
            // spectrum has no positive curvature (a genuine non-PD Hessian — a
            // SIR fallback is meaningful here), or (b) the eigendecomposition
            // itself diverged and produced a non-finite eigenvalue (the Hessian
            // contains NaN/Inf — no usable proposal can be built).
            //
            // `extract_eigenvalues` returns None for exactly case (b). Building a
            // fallback proposal there would re-run the same divergent
            // decomposition and embed NaN eigenvectors into the proposal
            // covariance, which SIR would then silently turn into NaN samples.
            // So only build the proposal when the eigenvalues are finite.
            match extract_eigenvalues(&hess_free_sym) {
                Some(eigvals) => {
                    let fallback_proposal = build_non_pd_fallback_proposal(
                        &hess_free_sym,
                        &free_idx,
                        n,
                        4.0,
                        &unit_coords,
                    );
                    return Some(CovarianceStepResult::FailedNonPd {
                        reason: format_non_pd_warning(&eigvals),
                        fallback_proposal,
                    });
                }
                None => {
                    return Some(CovarianceStepResult::Unusable(
                        "Covariance step failed: could not compute eigenvalues of the \
                         FD Hessian (Hessian may contain NaN or Inf). \
                         SE estimates not available."
                            .to_string(),
                    ));
                }
            }
        }
    };
    // The FD Hessian is of the OFV = −2·logL. The asymptotic covariance is the
    // inverse observed Fisher information R = Hessian of −logL = ½·H_ofv, so
    // R⁻¹ = 2·H_ofv⁻¹. Without this factor every SE is 1/√2 too small.
    let r_inv = inv.inverse.clone() * 2.0;
    // #1781: an `R` the floor had to clip, at an estimate its objective never optimised
    // (SAEM / IMP / IMPMAP / VI), is not inverted at all — the estimator becomes `S⁻¹`. The
    // floored direction is the objective's slope at a non-stationary point, not a precision,
    // and `R⁻¹` / `R⁻¹SR⁻¹` would report `1/floor` / `1/floor²` along it. `estimator` is the
    // single value every downstream branch reads, so the eigen-floor warning, the salvage
    // note and the published label all describe the matrix that is actually returned.
    let fallback_to_s = r_falls_back_to_s(fit_method, inv.n_clipped, options.covariance_method);
    let estimator = if fallback_to_s {
        CovarianceMethod::CrossProduct
    } else {
        options.covariance_method
    };
    // Only built on the fallback, where both failure arms below need it.
    let non_stationary = |s_unavailable: Option<&str>| {
        format_non_stationary_warning(&NonStationaryFacts {
            method: fit_method,
            objective: cov_objective_method(options),
            requested: options.covariance_method,
            min_eigenvalue: inv.min_eigenvalue,
            max_eigenvalue: inv.max_eigenvalue,
            s_unavailable,
            // Only the #1781 route scores an estimate on another method's objective.
            quadrature_nodes: (fit_method != options.method)
                .then(|| options.agq_nodes())
                .flatten(),
            n_subjects: population.subjects.len(),
            n_free: free_idx.len(),
            sir_interaction: options.interaction,
            sir_armed: sir_fallback_armed,
        })
    };
    // The same `R⁻¹` with the floored directions dropped instead of floored — the covariance
    // the data alone supports. Carried through the *same* estimator and the *same* reported
    // -parameter transform as `r_inv`, so the inflation the message prints is the inflation of
    // the numbers on the page (#1508 review §1). Built only when the floor actually fired and
    // only when the warning can be emitted at all (the cross-product estimator never returns
    // `R⁻¹`, so nothing there is inflated by this floor).
    let r_inv_ref = (inv.n_clipped > 0 && estimator != CovarianceMethod::CrossProduct)
        .then(|| inv.unclipped_inverse.clone() * 2.0);

    // Select the covariance estimator (NONMEM `$COV MATRIX=`). `R⁻¹` is the
    // model-based default; `S⁻¹` and `R⁻¹SR⁻¹` additionally need the per-subject
    // score cross-product `S = Σᵢ gᵢgᵢᵀ`. `S` is on the −logL scale
    // (`gᵢ = ∂(−logLᵢ)/∂θ`, no factor of 2), matching `R = ½·H_ofv`.
    // Anchored against NONMEM `$COV MATRIX=S`/`RSR` for both FOCEI (#266) and
    // FOCE (no-INTER) (#250): all SEs within ~10% of NONMEM.
    //
    // On the #1781 fallback neither failure of `S` is `Unusable`: `R` is non-positive-definite
    // there, which is the `FailedNonPd` contract, so the SIR fallback still gets its
    // |λ|-rectified proposal — and the reason names both matrices instead of advising the
    // `covariance_method = r or rsr` this step has just declined to use.
    //
    // A requested `s` returned before `R` was built (#1803), so the only way to reach `S⁻¹`
    // here is that fallback: `combine_covariance_and_reference`'s one failure, a singular `S`
    // under `CrossProduct`, can only be the fallback's.
    let (cov_free, cov_free_ref) = if estimator == CovarianceMethod::Hessian {
        (r_inv, r_inv_ref)
    } else {
        let s_free = match s_first {
            Some(s) => Ok(s),
            None => match score_cross_product() {
                Some(s) => s,
                None => {
                    return Some(CovarianceStepResult::Unusable(
                        COV_CANCELLED_MSG.to_string(),
                    ))
                }
            },
        };
        let s_free = match s_free {
            Ok(s) => s,
            Err(reason) if fallback_to_s => {
                return Some(CovarianceStepResult::FailedNonPd {
                    reason: non_stationary(Some(&reason)),
                    fallback_proposal: build_non_pd_fallback_proposal(
                        &hess_free_sym,
                        &free_idx,
                        n,
                        4.0,
                        &unit_coords,
                    ),
                })
            }
            Err(reason) => return Some(CovarianceStepResult::Unusable(reason)),
        };
        match combine_covariance_and_reference(estimator, r_inv, r_inv_ref, &s_free, &unit_coords) {
            Some(pair) => pair,
            None => {
                debug_assert!(
                    fallback_to_s,
                    "only `CrossProduct` fails to combine, and a requested `s` returned before `R`"
                );
                return Some(CovarianceStepResult::FailedNonPd {
                    reason: non_stationary(Some(S_RANK_DEFICIENT_REASON)),
                    fallback_proposal: build_non_pd_fallback_proposal(
                        &hess_free_sym,
                        &free_idx,
                        n,
                        4.0,
                        &unit_coords,
                    ),
                });
            }
        }
    };

    let cov = embed_free(&cov_free, &free_idx, n);

    let mut cov_warnings: Vec<String> = Vec::new();

    // #1514: name the salvaged subjects, unconditionally — not only when the eigenvalue floor
    // fired. The route is a property of the numbers on the page, and a user comparing two runs
    // of the same model (or this model against its `analytic_cov_hessian = false` twin) has no
    // other way to see that one subject's term came off a different estimator. Emitted here,
    // after the estimator has been assembled, so a covariance step that failed earlier says
    // nothing about a route whose result was discarded.
    //
    // Read off `source` rather than off `salvaged` directly, so the note and the
    // regularization message's route label are **one** derived value with two consumers. That
    // is not decoration: the label is only *observable* on a fit whose eigenvalue floor fired,
    // so a mutation of the three-way derivation above went undetected by the whole suite until
    // the note was routed through it (measured — the `M5` cell of this PR's mutation sweep
    // killed nothing before this line, and kills `salvage_tests::one_declining_subject_reproduces_the_pure_
    // analytic_covariance` after). `format_salvage_note`'s own emptiness guard is what makes
    // it total for the two non-hybrid arms, not a second gate on the same fact.
    //
    // Gated on the estimator for the same reason the eigen-floor warning twelve lines below is
    // (#1516 review §1): where the returned covariance is `S⁻¹` from
    // `assemble_score_cross_product` alone and `R` is discarded, *nothing* about the route
    // that assembled `R` reaches the numbers on the page. The note's closing claim — that the
    // named subjects' terms use a different estimator than the rest — would then be a
    // statement about a matrix this step threw away. Since #1803 a requested `s` builds no `R`
    // and never gets here, so the gate's `S⁻¹` side is the #1781 fallback alone.
    let salvaged_ids: Vec<&str> = match source {
        CovHessianSource::HybridRMatrix if estimator != CovarianceMethod::CrossProduct => salvaged
            .iter()
            .map(|&i| population.subjects[i].id.as_str())
            .collect(),
        CovHessianSource::HybridRMatrix
        | CovHessianSource::AnalyticRMatrix
        | CovHessianSource::FdStencil => Vec::new(),
    };
    if let Some(note) = format_salvage_note(&salvaged_ids, population.subjects.len()) {
        if options.verbose {
            eprintln!("  {}", note);
        }
        cov_warnings.push(note);
    }

    // The Hessian eigenvalue-floor warning is about `R`. It is relevant only when
    // the returned covariance actually uses `R⁻¹` (Hessian and sandwich); the
    // cross-product estimator returns `S⁻¹` (with a full-rank `S` guaranteed above), so
    // a clipped `R` there would be a misleading note about a matrix it didn't use.
    // #1781: the fallback replaced `R⁻¹` by `S⁻¹`, so the floor warning just below — about a
    // matrix this step no longer returns — is skipped by the same `estimator` gate, and this
    // one says what happened instead.
    if fallback_to_s {
        let msg = non_stationary(None);
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    }
    if inv.n_clipped > 0 && estimator != CovarianceMethod::CrossProduct {
        // #520 C1/C2 and the label fix. Severity is graded on magnitude (`|min λ| / λ_max` and
        // the variance inflation the floor caused), never on the clipped count; the route is
        // named from what actually ran, so "FD Hessian" is no longer printed on the analytic
        // R-matrix route; and the stencil-only sentences (which gate clause declined the
        // analytic route, what tolerance the ODEs integrate at) are attached here, where all of
        // those facts are in scope. The prose itself lives in `cov_diagnostics` and is
        // unit-tested cell by cell without a fit.
        //
        // Resolved only where a stencil ran, and only here: the walk is one predicate sweep per
        // subject, which is nothing against the stencil that has just run, but it is not free
        // on a fit that had no complaint to make. On the hybrid route it is the *salvaged*
        // subjects' clauses it names — the same walk, since every other subject passes it.
        let declines = if source.emits_fd_guidance() {
            analytic_cov_declines(model, population, options, is_mixture)
        } else {
            Vec::new()
        };
        let facts = CovRegularizationFacts {
            source,
            n_clipped: inv.n_clipped,
            n_free,
            min_eigenvalue: inv.min_eigenvalue,
            max_eigenvalue: inv.max_eigenvalue,
            floor: inv.floor,
            variance_inflation: reported_variance_inflation(
                &cov,
                cov_free_ref.as_ref(),
                &free_idx,
                n,
                template,
            ),
            declines: &declines,
            ode: stencil_ode,
        };
        let msg = format_regularized_warning(&facts);
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    } else if options.verbose {
        eprintln!("  Covariance step successful");
    }

    // #1352: say which curvature the SEs are — the penalized objective's, or (on the #1781
    // `S⁻¹` fallback) one that cannot represent the penalty at all.
    if nn_reg.is_some() {
        cov_warnings.push(format_nn_penalty_cov_note(estimator));
    }

    // Soft warning: cross-partial stencils that returned NaN/Inf contributed nothing, so some
    // off-diagonal correlation is missing for these parameters and their SEs may be
    // over-optimistic. *How much* is missing depends on the route — the whole cross-partial on
    // the population stencil, only the declined subjects' share of it on the hybrid — so the
    // sentence is built from `source` rather than stated once (#1514 review §1).
    if let Some(msg) = offdiag_nan_warning(&fd_offdiag_nan, estimator, source, template) {
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    }

    Some(CovarianceStepResult::Success(CovarianceOutput {
        matrix: cov,
        warnings: cov_warnings,
        method: estimator,
    }))
}

/// The covariate-NN regularizer whose curvature the covariance step adds to `R` (#1352), or
/// `None` when there is none to add: no positive `nn_l2` / `nn_smooth` on a model with a
/// `[covariate_nn]` block, or a `fit_method` that does not apply the penalty
/// ([`crate::estimation::nn_reg::applies_nn_regularization`]) — an unregularized estimate's
/// SEs are the curvature of the objective it actually minimised.
pub(crate) fn cov_nn_regularizer(
    fit_method: EstimationMethod,
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
) -> Option<crate::estimation::nn_reg::NnRegularizer> {
    if !crate::estimation::nn_reg::applies_nn_regularization(fit_method) {
        return None;
    }
    let reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    reg.is_active().then_some(reg)
}

/// The note naming what a regularized DCM fit's standard errors are (#1352), by the estimator
/// that produced them. `R` carries the covariate-NN penalty's curvature, so `r` and `rsr` give
/// penalized-ML SEs; `S` is a sum of per-subject scores and the penalty has no per-subject
/// part, so `s` — requested, or the #1781 fallback — cannot represent it.
pub(crate) fn format_nn_penalty_cov_note(estimator: CovarianceMethod) -> String {
    let body = match estimator {
        CovarianceMethod::Hessian => {
            "covariate-NN regularization (nn_l2 / nn_smooth) is part of \
             the minimised objective, so the R matrix includes the penalty's curvature: these are \
             penalized-ML standard errors, not the data-only curvature of the reported \
             (unpenalized) OFV."
        }
        CovarianceMethod::Sandwich => {
            "covariate-NN regularization (nn_l2 / nn_smooth) is part of \
             the minimised objective: the sandwich's R includes the penalty's curvature, while S \
             (per-subject scores) has no penalty term — penalized-ML sandwich standard errors."
        }
        CovarianceMethod::CrossProduct => {
            "covariate-NN regularization (nn_l2 / nn_smooth) is not \
             represented in these standard errors: S⁻¹ is built from per-subject scores and the \
             penalty has no per-subject part, so they ignore it. Use covariance_method = r for \
             penalized-ML standard errors."
        }
    };
    format!("{NN_PENALTY_COV_TOKEN}: {body}")
}

/// The token every [`format_nn_penalty_cov_note`] starts with (#1352), so
/// [`crate::types::classify_warning`] files it as a covariance-step note and `run_covariance`
/// replaces it with the step it reruns, instead of keeping an `r` note beside an `s` one.
pub(crate) const NN_PENALTY_COV_TOKEN: &str = "W_COV_NN_PENALTY";

/// The soft warning for cross-partial stencils that came back non-finite, naming their
/// parameters in packed order (deterministic whatever the `HashSet` iteration order) — or
/// `None` when there were none, or when the returned matrix is `S⁻¹` (#1803).
///
/// The stencil is `R`'s, and the warning says that *these SEs* may be over-optimistic. Under
/// `S⁻¹` they are not built from `R` at all — that is the #1781 fallback, the one way a step
/// that built `R` returns `S⁻¹` — so the sentence would be about a matrix the step discarded.
/// The same gate as the eigen-floor warning and the salvage note.
fn offdiag_nan_warning(
    fd_offdiag_nan: &HashSet<usize>,
    estimator: CovarianceMethod,
    source: CovHessianSource,
    template: &ModelParameters,
) -> Option<String> {
    if fd_offdiag_nan.is_empty() || estimator == CovarianceMethod::CrossProduct {
        return None;
    }
    let mut sorted_idx: Vec<usize> = fd_offdiag_nan.iter().copied().collect();
    sorted_idx.sort_unstable();
    let names: Vec<String> = sorted_idx
        .iter()
        .map(|&i| packed_param_label(i, template))
        .collect();
    Some(format_offdiag_nan_warning(&names.join(", "), source))
}

/// Result of [`invert_psd_with_floor`].
pub(crate) struct RegularizedInverse {
    pub inverse: DMatrix<f64>,
    /// Smallest eigenvalue of the input matrix (before clipping). `f64::INFINITY`
    /// for 0×0 matrices.
    pub min_eigenvalue: f64,
    /// Largest eigenvalue of the input matrix. `f64::NEG_INFINITY` for 0×0 matrices.
    /// Paired with `min_eigenvalue` this gives the scale-free `|min λ| / λ_max` the
    /// regularization severity is graded on (#520).
    pub max_eigenvalue: f64,
    /// Floor used for clipping. Same shape rules as `min_eigenvalue`.
    pub floor: f64,
    /// How many eigenvalues fell below the floor and were clipped.
    pub n_clipped: usize,
    /// `Q Λ⁻¹ Qᵀ` restricted to the directions the floor left alone — the Moore–Penrose
    /// pseudo-inverse over the unclipped spectrum, with the clipped directions contributing
    /// **nothing** instead of `1/floor` (#520 C1).
    ///
    /// This is the *reference* the regularization diagnostic is measured against: it is the
    /// covariance the data alone supports, so running it through the same estimator and the
    /// same reported-parameter delta transform as the returned inverse gives the inflation the
    /// user's standard errors actually carry. Equal to `inverse` when `n_clipped == 0`.
    ///
    /// Kept as a matrix rather than collapsed to a per-coordinate ratio here because the two
    /// steps that follow — the sandwich `R⁻¹ S R⁻¹` and the block-Ω Jacobian — both mix
    /// coordinates, so a packed-space diagonal ratio is not the number a user reads (#1508
    /// review §1).
    pub unclipped_inverse: DMatrix<f64>,
}

/// Worst inflation of a variance caused by the eigenvalue floor: `max_k var[k] / var_ref[k]`,
/// never below `1.0` (#520 C1).
///
/// `var` and `var_ref` are the **same** quantities computed two ways — the returned covariance
/// and the same pipeline with the floored directions dropped instead of floored. Either
/// diagonals of a covariance or squared reported standard errors; the caller decides which,
/// and [`compute_covariance`] passes the reported ones.
///
/// Two guards, and both are load-bearing — each is killed by its own mutation, which is how
/// they were cut down from three (the third rejected exactly the inputs the first already did,
/// and deleting it left the suite green: AGENTS.md's redundant-gate hole):
///
/// * **`var[k]` not positive** — the `FIX`ed-parameter cell. A pinned coordinate reports
///   `SE = 0` on *both* sides, and `0 / 0` would fall into the `var_ref <= 0` arm below and
///   report every `FIX`ed parameter as `unbounded`. Skipping is the right answer: the floor
///   cannot have inflated a variance that is not there.
/// * **`var[k]` not finite** — a diverged solve. Without it an `inf` reported variance divides
///   to `inf` and the message reports an unbounded inflation for a covariance that is simply
///   broken, where the caller's own non-finite handling should speak instead.
///
/// `var_ref[k] <= 0` with a positive `var[k]` is `f64::INFINITY` — the meaningful answer (that
/// coordinate's entire variance came from floored directions), not an accident of division.
/// The fold compares with `>` rather than folding through `f64::max`; both let a `NaN` fall
/// through here, but the explicit comparison says so where a reader can see it.
/// Every standard error the fit will **report**, flattened: θ on its natural scale, the Ω
/// lower triangle through the multivariate Cholesky Jacobian, σ, Ω_IOV, and the `block_sigma`
/// ρ's. Exactly the numbers `FitResult` carries, produced by the one function that produces
/// them, so the diagnostic cannot measure a different transform than the user reads.
fn reported_standard_errors(cov: &DMatrix<f64>, template: &ModelParameters) -> Vec<f64> {
    let owned = Some(cov.clone());
    let (se_theta, se_omega, se_sigma, se_kappa) =
        crate::api::postfit::extract_standard_errors(&owned, template);
    let rho = crate::api::postfit::extract_residual_correlation_se(&owned, template);
    [se_theta, se_omega, se_sigma, se_kappa, rho]
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}

/// Worst inflation of a **reported** variance caused by the eigenvalue floor (#520 C1, reworked
/// for #1508 review §1).
///
/// `cov` is the covariance the fit will return, already embedded in the full packed space;
/// `cov_free_ref` is the same estimator's output with the floored directions dropped instead of
/// floored, over the free block. Both are run through [`reported_standard_errors`], i.e.
/// through the selected estimator **and** the reported-parameter delta transform, and compared
/// as variances.
///
/// That is what makes the number responsive to the two things a packed-space `R⁻¹` diagonal
/// could not see: the sandwich `R⁻¹ S R⁻¹`, where a different `S` concentrates or suppresses
/// the floored eigendirection for the same `R`, and the block-Ω Jacobian, which mixes packed
/// coordinates on its way to a reported `ω_ij`.
///
/// `1.0` when there is no reference, which is exactly the cells where nothing was floored into
/// the returned covariance (`n_clipped == 0`, or `covariance_method = s`, which never inverts
/// `R`) — and in those cells the caller does not emit the warning at all.
fn reported_variance_inflation(
    cov: &DMatrix<f64>,
    cov_free_ref: Option<&DMatrix<f64>>,
    free_idx: &[usize],
    n: usize,
    template: &ModelParameters,
) -> f64 {
    let Some(cov_free_ref) = cov_free_ref else {
        return 1.0;
    };
    let mut cov_ref = DMatrix::zeros(n, n);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            cov_ref[(i, j)] = cov_free_ref[(a, b)];
        }
    }
    let se = reported_standard_errors(cov, template);
    let se_ref = reported_standard_errors(&cov_ref, template);
    let var: Vec<f64> = se.iter().map(|s| s * s).collect();
    let var_ref: Vec<f64> = se_ref.iter().map(|s| s * s).collect();
    worst_variance_inflation(&var, &var_ref)
}

pub(crate) fn worst_variance_inflation(var: &[f64], var_ref: &[f64]) -> f64 {
    debug_assert_eq!(
        var.len(),
        var_ref.len(),
        "worst_variance_inflation compares the same coordinates two ways"
    );
    let mut worst: f64 = 1.0;
    for (&v, &v_ref) in var.iter().zip(var_ref.iter()) {
        if !(v > 0.0) || !v.is_finite() {
            continue;
        }
        let ratio = if v_ref > 0.0 {
            v / v_ref
        } else {
            f64::INFINITY
        };
        if ratio > worst {
            worst = ratio;
        }
    }
    worst
}

/// Invert a symmetric matrix by clipping eigenvalues to a small positive floor.
///
/// This is the regularised replacement for `try_inverse() + neg-diag check` on
/// the FD Hessian. The previous code rejected the entire covariance step on a
/// single negative diagonal of the raw inverse — which on a well-conditioned
/// surface (FOCE/FOCEI converges cleanly to the same OFV across optimizers) is
/// almost always an FD-noise artefact rather than real ill-conditioning. The
/// floor leaves PD inputs untouched (`n_clipped == 0`, exact inverse) and
/// recovers a PD inverse on near-singular or marginally-indefinite inputs.
///
/// Floor: `max(max_eig * 1e-10, 1e-12)`. Anchoring to `max_eig` keeps the
/// regularisation equivariant to multiplying the **whole** matrix by a constant; the
/// absolute floor handles the edge case where the whole spectrum is tiny. It is *not*
/// invariant to rescaling one coordinate, which is what a change of units does to an
/// identity-packed θ — the covariance step's callers go through
/// [`invert_psd_with_floor_unit_free`] for that (#2145).
///
/// Returns `None` only when the eigendecomposition fails or every eigenvalue
/// is non-finite or non-positive — i.e. the Hessian carries no usable
/// curvature information at all, in which case regularisation cannot help.
pub(crate) fn invert_psd_with_floor(sym: &DMatrix<f64>) -> Option<RegularizedInverse> {
    let n = sym.nrows();
    debug_assert_eq!(
        n,
        sym.ncols(),
        "invert_psd_with_floor requires square input"
    );
    if n == 0 {
        return Some(RegularizedInverse {
            inverse: DMatrix::zeros(0, 0),
            min_eigenvalue: f64::INFINITY,
            max_eigenvalue: f64::NEG_INFINITY,
            floor: f64::INFINITY,
            n_clipped: 0,
            unclipped_inverse: DMatrix::zeros(0, 0),
        });
    }

    // Symmetric eigendecomposition: H = Q Λ Qᵀ ⇒ H⁻¹ = Q Λ⁻¹ Qᵀ. Inverting via
    // the eigendecomposition lets us clip non-positive Λ entries before
    // forming Λ⁻¹, which is what `try_inverse` cannot do.
    let eig = SymmetricEigen::new(sym.clone());
    let q = &eig.eigenvectors;
    let lambdas = &eig.eigenvalues;

    let mut max_eig = f64::NEG_INFINITY;
    for i in 0..n {
        let l = lambdas[i];
        if !l.is_finite() {
            return None;
        }
        if l > max_eig {
            max_eig = l;
        }
    }
    if !max_eig.is_finite() || max_eig <= 0.0 {
        // Spectrum is entirely ≤ 0 — no positive curvature anywhere; this is
        // a genuinely degenerate Hessian, not FD noise. Flag as failure so the
        // caller can report "Covariance step failed" rather than silently
        // returning a meaningless matrix.
        return None;
    }

    let floor = (max_eig * 1e-10).max(1e-12);
    let mut min_eig = f64::INFINITY;
    let mut n_clipped = 0;
    let mut inv_lambdas = DVector::zeros(n);
    for i in 0..n {
        let l = lambdas[i];
        if l < min_eig {
            min_eig = l;
        }
        let l_clipped = if l < floor {
            n_clipped += 1;
            floor
        } else {
            l
        };
        inv_lambdas[i] = 1.0 / l_clipped;
    }

    // cov = Q diag(1/λ) Qᵀ — scale columns of Q by 1/λ, then multiply by Qᵀ.
    let mut q_scaled = q.clone();
    for j in 0..n {
        let s = inv_lambdas[j];
        for i in 0..n {
            q_scaled[(i, j)] *= s;
        }
    }
    let mut inverse = &q_scaled * q.transpose();
    // Eigendecomposition + reconstruction is symmetric in exact arithmetic but
    // not in floating point; symmetrise so downstream consumers (e.g. SIR
    // proposal Cholesky) see a numerically symmetric matrix.
    let inv_t = inverse.transpose();
    inverse = (&inverse + &inv_t) * 0.5;

    // The reference the regularization diagnostic is graded against (#520 C1): the same
    // spectral inverse with the floored directions contributing **nothing** rather than
    // `1/floor`. Built here because it needs the eigenvectors, which do not survive this
    // function; the ratio itself is taken downstream, after the estimator and the
    // reported-parameter delta transform have both had their say (#1508 review §1).
    //
    // Skipped when nothing was clipped: the two matrices are then equal by construction, and
    // the caller's `n_clipped > 0` gate means it is never read.
    let unclipped_inverse = if n_clipped == 0 {
        inverse.clone()
    } else {
        let mut q_unclipped = q.clone();
        for j in 0..n {
            let s = if lambdas[j] >= floor {
                1.0 / lambdas[j]
            } else {
                0.0
            };
            for i in 0..n {
                q_unclipped[(i, j)] *= s;
            }
        }
        let m = &q_unclipped * q.transpose();
        let m_t = m.transpose();
        (&m + &m_t) * 0.5
    };

    Some(RegularizedInverse {
        inverse,
        min_eigenvalue: min_eig,
        max_eigenvalue: max_eig,
        floor,
        n_clipped,
        unclipped_inverse,
    })
}

/// How a packed free coordinate carries units, as the eigenvalue floor of
/// [`invert_psd_with_floor_unit_free`] must treat it (#2145, #2274).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CoordUnits {
    /// Log-θ, the log-Cholesky Ω diagonal, log-σ, Fisher-z ρ: a tiny diagonal entry is a flat
    /// direction, and the floor sees the coordinate as it is.
    Dimensionless,
    /// An identity-packed θ: its units are unknown to ferx, so its diagonal is equilibrated
    /// against the largest dimensionless one (#2145). Not a `[covariate_nn]` weight, which is
    /// identity-packed but has no user units, and stays [`Dimensionless`](Self::Dimensionless)
    /// so a weight the data does not identify is still clipped and reported (#2273).
    IdentityTheta,
    /// A coordinate whose units are those of a known estimate `s`, so `x / s` is unit-free: an
    /// Ω Cholesky off-diagonal `L_ij`, scaled by its row's diagonal `L̂_ii` (#2274).
    Scaled(f64),
}

/// [`invert_psd_with_floor`] with the floor made independent of the units of the coordinates
/// `units` marks as carrying them (#2145, #2274).
///
/// Two kinds of packed coordinate carry units, and writing either in units `k` times larger
/// divides its row and column of `R` or `S` by `k`. The floor, taken from the raw spectrum,
/// then clips a well-determined fit purely because of those units — under
/// `covariance_method = s` that is a rejected `S`, under `r`/`rsr` an understated SE. So each
/// is rescaled first, `C = D⁻¹ M D⁻¹`; the floor is applied to `C`'s spectrum and the result
/// mapped back, `M⁻¹ = D⁻¹ C⁻¹ D⁻¹`.
///
/// - **An Ω Cholesky off-diagonal** ([`CoordUnits::Scaled`]). With `Ω = L·Lᵀ`, `L_ij` (`i > j`)
///   has the units of `η_i` — of `BASE` itself when `BASE = TVBASE + ETA_B` — and so has `L_ii`.
///   `dᵢ = 1 / L̂_ii` makes the coordinate exactly unit-free (#2274). The scale is the estimate,
///   not the matrix, so a genuinely flat off-diagonal direction is still clipped; ferx cannot
///   tell from the packed vector whether an η enters additively, so every off-diagonal is
///   scaled, which moves the floor on any block-Ω model by an O(1) factor (`L̂_ii` is ~0.1–1
///   for an exponential η). A non-positive or non-finite scale keeps `dᵢ = 1`.
/// - **An identity-packed θ** ([`CoordUnits::IdentityTheta`], a negative lower bound) has units
///   ferx knows nothing about. Each with a positive, finite diagonal gets `dᵢ = sqrt(Mᵢᵢ / g)`,
///   `g` the largest positive diagonal entry of the **other** coordinates after their own
///   scaling, so its diagonal entry in `C` is `g` whatever its units. With no usable other
///   entry (every free coordinate an identity-packed θ, e.g. Ω and σ FIX) `g = 1`, i.e. Jacobi
///   scaling of the flagged block: any anchor read off a flagged diagonal would move with the
///   units, and the absolute `1e-12` floor would clip on them again.
///
/// The largest rather than a mean: a mean is dragged down by exactly the flat dimensionless
/// direction the floor exists to catch, and would then pull a well-determined θ under the floor
/// with it. At the largest entry the flagged coordinate leaves `λ_max`, and so the floor every
/// other coordinate is tested against, where the dimensionless block alone puts it (within the
/// factor the off-diagonals allow).
///
/// Every [`CoordUnits::Dimensionless`] coordinate keeps `dᵢ = 1`. Log-θ, the log-Cholesky Ω
/// diagonal, log-σ and ρ are dimensionless, so there a tiny diagonal entry *is* a flat direction
/// and the floor must keep seeing it — which full Jacobi scaling of every coordinate would hide,
/// since a diagonal matrix equilibrates to the identity. A flagged θ with a non-positive or
/// non-finite diagonal also keeps `dᵢ = 1`: a parameter with no information is still clipped.
///
/// The returned `min_eigenvalue`, `max_eigenvalue` and `floor` are those of `C`, so the
/// regularization warning built from them no longer moves with these units either.
pub(crate) fn invert_psd_with_floor_unit_free(
    sym: &DMatrix<f64>,
    units: &[CoordUnits],
) -> Option<RegularizedInverse> {
    let n = sym.nrows();
    debug_assert_eq!(units.len(), n, "one flag per coordinate");
    let Some(d) = unit_coordinate_scales(sym, units) else {
        return invert_psd_with_floor(sym);
    };
    let mut c = sym.clone();
    for i in 0..n {
        for j in 0..n {
            c[(i, j)] /= d[i] * d[j];
        }
    }
    let mut inv = invert_psd_with_floor(&c)?;
    for i in 0..n {
        for j in 0..n {
            let s = d[i] * d[j];
            inv.inverse[(i, j)] /= s;
            inv.unclipped_inverse[(i, j)] /= s;
        }
    }
    Some(inv)
}

/// The diagonal scaling `d` of [`invert_psd_with_floor_unit_free`], or `None` when it is the
/// identity (no coordinate with a usable scale), so the unscaled path is taken unchanged.
pub(crate) fn unit_coordinate_scales(sym: &DMatrix<f64>, units: &[CoordUnits]) -> Option<Vec<f64>> {
    let usable = |m: f64| m.is_finite() && m > 0.0;
    let n = sym.nrows();
    // The known scales first: `dᵢ = 1 / sᵢ`, so `Cᵢᵢ = Mᵢᵢ·sᵢ²`.
    let mut d: Vec<f64> = units
        .iter()
        .map(|u| match *u {
            CoordUnits::Scaled(s) if usable(s) => 1.0 / s,
            _ => 1.0,
        })
        .collect();
    let is_theta = |i: usize| units[i] == CoordUnits::IdentityTheta && usable(sym[(i, i)]);
    if !(0..n).any(|i| is_theta(i) || d[i] != 1.0) {
        return None;
    }
    let g = (0..n)
        .filter(|&i| units[i] != CoordUnits::IdentityTheta)
        .map(|i| sym[(i, i)] / (d[i] * d[i]))
        .filter(|&m| usable(m))
        .reduce(f64::max)
        .unwrap_or(1.0);
    for (i, di) in d.iter_mut().enumerate() {
        if is_theta(i) {
            *di = (sym[(i, i)] / g).sqrt();
        }
    }
    Some(d)
}

/// How each free coordinate of the covariance step carries units, the mask
/// [`invert_psd_with_floor_unit_free`] reads: an identity-packed θ (`theta_packs_log` false,
/// #2145), an Ω or Ω_IOV Cholesky off-diagonal scaled by its row's diagonal `L̂_ii` read off
/// `x_hat` (#2274), and everything else dimensionless. The walk mirrors
/// `pack_params_with_moves`: θ, the Ω lower triangle, σ, the IOV lower triangle; the mixture
/// overrides and the ρ's that follow are log- and Fisher-z-packed.
///
/// `nn_weights` are the θ ranges of the `[covariate_nn]` weights
/// (`CompiledModel::nn_weight_theta_ranges`). A weight is identity-packed (its lower bound is
/// `-∞`) but is not a parameter in user units, so it stays dimensionless: equilibrated, a flat
/// weight — one feeding a saturated unit — would sit at the anchor and escape the floor and the
/// `covariance_regularized` warning (#2273).
pub(crate) fn free_coordinate_units(
    template: &ModelParameters,
    x_hat: &[f64],
    free_idx: &[usize],
    nn_weights: &[std::ops::Range<usize>],
) -> Vec<CoordUnits> {
    let mut units: Vec<CoordUnits> = template
        .theta_lower
        .iter()
        .enumerate()
        .map(|(i, &lo)| {
            if theta_packs_log(lo) || nn_weights.iter().any(|w| w.contains(&i)) {
                CoordUnits::Dimensionless
            } else {
                CoordUnits::IdentityTheta
            }
        })
        .collect();
    let at = unpack_params(x_hat, template);
    let push_omega = |chol: &DMatrix<f64>, diagonal: bool, units: &mut Vec<CoordUnits>| {
        for (i, j) in lower_tri_iter(chol.nrows(), diagonal) {
            units.push(if i == j {
                CoordUnits::Dimensionless
            } else {
                CoordUnits::Scaled(chol[(i, i)])
            });
        }
    };
    push_omega(&at.omega.chol, at.omega.diagonal, &mut units);
    units.extend(std::iter::repeat_n(
        CoordUnits::Dimensionless,
        template.sigma.values.len(),
    ));
    if let Some(ref iov) = at.omega_iov {
        push_omega(&iov.chol, iov.diagonal, &mut units);
    }
    free_idx
        .iter()
        .map(|&i| units.get(i).copied().unwrap_or(CoordUnits::Dimensionless))
        .collect()
}

/// Owned result of the gated covariance step, consumed by every estimator
/// finalizer. Mirrors exactly what the 8 inline blocks produced:
/// `(matrix, wall_time_secs)` is the old tuple; `warnings` is drained into the
/// caller's vec via `.extend` at the same program point; `sir_fallback_proposal`
/// carries the |λ|-rectified proposal on `FailedNonPd`.
pub(crate) struct CovStepOutcome {
    pub matrix: Option<DMatrix<f64>>,
    pub wall_time_secs: f64,
    pub warnings: Vec<String>,
    pub sir_fallback_proposal: Option<DMatrix<f64>>,
    /// Which estimator produced `matrix` (#1382) — the **routed** one, which is
    /// not always the requested one (see [`scale_routed_covariance_method`]).
    /// `None` exactly when `matrix` is `None`; see
    /// [`published_covariance_method`].
    pub method: Option<crate::types::CovarianceMethod>,
}

/// The estimator label to publish alongside a covariance matrix: `Some(routed)`
/// when a matrix was produced, `None` otherwise (#1382).
///
/// `routed` is the post-[`scale_routed_covariance_method`] choice — the one
/// `compute_covariance` was actually configured with — never
/// `FitOptions::covariance_method`, which is what was *asked for*. The two
/// differ whenever a defaulted `r` is routed onto the cross-product above
/// [`crate::types::COV_HESSIAN_MAX_DIM`] free parameters, and a label that
/// reported the request there would be wrong precisely on the fits where the
/// user could not have predicted the answer.
///
/// The `None`-without-a-matrix half is the other invariant: a failed, skipped or
/// SIR-fallback step publishes no matrix, no standard errors, no eigenvalues and
/// no condition number, so there is nothing for an estimator name to describe.
pub(crate) fn published_covariance_method(
    matrix: Option<&DMatrix<f64>>,
    routed: crate::types::CovarianceMethod,
) -> Option<crate::types::CovarianceMethod> {
    matrix.map(|_| routed)
}

/// Which covariance estimator to actually assemble at `n` free coordinates, and
/// the warning to record for the choice (#1064).
///
/// The `R` matrix is a finite-difference Hessian of the objective that
/// re-converges every subject's EBEs at each of `2·n²` objective
/// evaluations (`fd_r_objective_evals`) — ~1.3 million re-converged population
/// objectives at `n = 800`, which will not finish. Above [`COV_HESSIAN_MAX_DIM`]
/// a *defaulted* `Hessian` is therefore routed to the cross-product, which
/// builds no `R` (#1803) and needs one pass. An explicit
/// `covariance_method = r` is the user's call and is honoured, but is told what
/// it is about to cost.
///
/// `rsr` builds the same `R` and is told the same (#2055). It is never the
/// default, so it is never routed — the routing is gated on `Hessian`, the cost
/// warning on [`needs_r`]. Its warning does not read `explicitly_set`: a
/// `Sandwich` request is the user's whether or not the flag was raised.
///
/// Split out from [`run_covariance_step_inner`] so the three branches are
/// reachable from a unit test: exercising them through the real covariance step
/// would mean converging a fit with several hundred free parameters.
pub(crate) fn scale_routed_covariance_method(
    n: usize,
    requested: CovarianceMethod,
    explicitly_set: bool,
    has_priors: bool,
) -> (CovarianceMethod, Option<String>) {
    if n <= crate::types::COV_HESSIAN_MAX_DIM || !needs_r(requested) {
        return (requested, None);
    }
    // #254: `S` is a sum of per-*subject* score cross-products, and a parameter
    // prior has no subject decomposition — it contributes one score for the whole
    // population, not N of them — so `S` cannot carry the prior's information at
    // all. Auto-routing a priored fit onto it would silently report unpenalized
    // standard errors for a penalized fit, which is the one thing the covariance
    // wiring exists to prevent. Stay on `R` (which does carry the prior) and say
    // why it will be slow, rather than being quietly wrong and fast.
    // A packed dimension near `usize::MAX.sqrt()` is not a model anyone can write; the
    // saturated count is still a lower bound on the cost.
    let stencil = fd_r_objective_evals(n).unwrap_or(usize::MAX);
    if requested == CovarianceMethod::Sandwich {
        // `S` cannot carry a prior (above), so with one declared the cheap way out
        // is not `s` — it would drop the prior from the sandwich's `S` and its `R`
        // alike — and only switching the step off is left to name.
        let warning = if has_priors {
            format!(
                "covariance_method = rsr with {n} free parameters and parameter priors \
                 declared: the sandwich R⁻¹SR⁻¹ needs the R matrix, a finite-difference \
                 Hessian needing {stencil} re-converged objective evaluations. \
                 `covariance_method = s` builds no R but cannot represent a prior, so it is \
                 not a way out here. Set `covariance = false` if this does not finish."
            )
        } else {
            format!(
                "covariance_method = rsr with {n} free parameters: the sandwich R⁻¹SR⁻¹ \
                 needs the R matrix, a finite-difference Hessian that re-converges every \
                 subject's EBEs at each of {stencil} objective evaluations. Set \
                 `covariance_method = s` (the score cross-product, one pass) if this does \
                 not finish."
            )
        };
        return (requested, Some(warning));
    }
    if has_priors {
        return (
            requested,
            Some(format!(
                "covariance_method = r with {n} free parameters and parameter priors \
                 declared: the R matrix is a finite-difference Hessian needing {stencil} \
                 re-converged objective evaluations. The usual large-problem fallback \
                 (`covariance_method = s`) cannot represent a prior, so it was not used. \
                 Set `covariance = false` if this does not finish."
            )),
        );
    }
    if explicitly_set {
        (
            requested,
            Some(format!(
                "covariance_method = r with {n} free parameters: the R matrix is a \
                 finite-difference Hessian that re-converges every subject's EBEs at each \
                 of {stencil} objective evaluations. Set `covariance_method = s` (the score \
                 cross-product, one pass) if this does not finish."
            )),
        )
    } else {
        (
            CovarianceMethod::CrossProduct,
            Some(format!(
                "{n} free parameters: the default covariance step (R = a \
                 finite-difference Hessian) would need {stencil} re-converged objective \
                 evaluations, so the score cross-product (`covariance_method = s`) was \
                 used instead. Set `covariance_method = r` explicitly to force it."
            )),
        )
    }
}

/// [`scale_routed_covariance_method`] for a step whose `R` is the stencil; on the analytic
/// R-matrix (`analytic_r`, see [`analytic_r_expected`]) `R` costs one pass per subject, so the
/// request is kept and nothing is said about a stencil that will not run.
pub(crate) fn routed_covariance_method(
    n: usize,
    options: &FitOptions,
    has_priors: bool,
    analytic_r: bool,
) -> (CovarianceMethod, Option<String>) {
    if analytic_r {
        return (options.covariance_method, None);
    }
    scale_routed_covariance_method(
        n,
        options.covariance_method,
        options.covariance_method_set,
        has_priors,
    )
}

/// The count for `fit()`'s pre-fit covariance cost note, at `n` free coordinates: `None` when
/// no note is due — the step is off, `n ≤ 30`, `R` is expected on the analytic assembly for
/// every subject (`route`, see [`predicted_r_route`]; #2054), or the estimator the step will
/// actually run (after [`scale_routed_covariance_method`]) builds no `R` — else `Some` of
/// `fd_r_objective_evals`, itself `None` on overflow (#1803). On
/// [`PredictedRRoute::Salvage`] the count is the same `2·n²`, but each evaluation covers only
/// the declined subjects; [`pre_fit_covariance_cost_note`] says so. Before #1803 the note fired
/// under `covariance_method = s` too, and quoted `n²`; before #2054 it ignored the route.
pub(crate) fn pre_fit_covariance_cost(
    n: usize,
    options: &FitOptions,
    has_priors: bool,
    route: PredictedRRoute,
) -> Option<Option<usize>> {
    if !options.run_covariance_step || n <= 30 || route == PredictedRRoute::Analytic {
        return None;
    }
    let (routed, _) = routed_covariance_method(n, options, has_priors, false);
    needs_r(routed).then(|| fd_r_objective_evals(n))
}

/// The pre-fit cost note for a [`pre_fit_covariance_cost`] result at `n` free coordinates, or
/// `None` when no note is due. On overflow it still warns, without the (wrapped) number. On
/// [`PredictedRRoute::Salvage`] it names the subject subset the stencil is expected to cover.
/// Worded as an expectation: a subject the walk predicts in scope can still bail at run time.
pub(crate) fn pre_fit_covariance_cost_note(
    n: usize,
    cost: Option<Option<usize>>,
    route: PredictedRRoute,
) -> Option<String> {
    let count = match cost? {
        Some(n_evals) => n_evals.to_string(),
        None => "2n²".to_string(),
    };
    let scope = match route {
        PredictedRRoute::Salvage {
            declined,
            n_subjects,
        } => format!(
            " over {declined} of {n_subjects} subjects (finite-difference Hessian for the \
             subjects outside the analytic R-matrix; the rest are analytic)"
        ),
        PredictedRRoute::Analytic | PredictedRRoute::Stencil => {
            " (finite-difference Hessian)".to_string()
        }
    };
    let tail = match cost? {
        Some(_) => "This may take several minutes on complex models.",
        None => "Estimate exceeds usize range; expect this to be very slow.",
    };
    Some(format!(
        "Covariance step: {n} parameters → {count} OFV evaluations expected{scope}. {tail}"
    ))
}

/// The covariance step WITHOUT the `run_covariance_step` gate: timer + optional
/// verbose line + `Success/Unusable/FailedNonPd` match. Contains NO floating-point
/// arithmetic — it only wraps `compute_covariance`, so it cannot change any numeric
/// result. This is the single home of the `CovarianceStepResult` match; both the
/// gated estimator finalizers (via [`run_covariance_step`]) and the ungated
/// standalone API (`run_covariance`, which deliberately ignores the flag) call it,
/// so the match has exactly one copy. `pre_msg` is the verbose stderr line the
/// caller folds its own `verbose` flag into (`Some` prints, `None` stays silent).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_covariance_step_inner(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
    pre_msg: Option<&str>,
) -> CovStepOutcome {
    if let Some(m) = pre_msg {
        eprintln!("{m}");
    }
    let mut warnings = Vec::new();
    // #1064: route away from the FD-of-OFV `R` matrix when the problem is too
    // large for it. The decision is a pure function so it can be unit-tested
    // without standing up a several-hundred-parameter fit.
    let scaled_options;
    let (routed, scale_warning) = routed_covariance_method(
        model.free_packed_dim(),
        options,
        !model.priors.is_empty(),
        analytic_r_expected(model, population, options, template.mixture.is_some()),
    );
    if let Some(w) = scale_warning {
        warnings.push(w);
    }
    let options = if routed == options.covariance_method {
        options
    } else {
        scaled_options = FitOptions {
            covariance_method: routed,
            ..options.clone()
        };
        &scaled_options
    };
    // #1781: which objective `R` and `S` differentiate. A SAEM / IMP / IMPMAP estimate on a
    // model with a non-Gaussian endpoint is scored on the quadrature marginal — the exact
    // likelihood those methods estimate — rather than on the FOCEI objective, whose minimum
    // can sit far from it. `fit_method` keeps the estimator that produced the point, which
    // `compute_covariance_for` needs for its own #1781 gate.
    let fit_method = options.method;
    let quadrature_options;
    let options = match cov_objective_route(
        fit_method,
        model.has_non_gaussian(),
        model.n_eta,
        template.mixture.is_some(),
        model.n_kappa > 0,
    ) {
        CovObjectiveRoute::AsFitted => options,
        CovObjectiveRoute::Quadrature { n_agq } => {
            warnings.push(format_quadrature_objective_note(
                fit_method,
                n_agq,
                model.n_eta,
            ));
            quadrature_options = FitOptions {
                method: crate::types::EstimationMethod::Laplace,
                n_agq,
                ..options.clone()
            };
            &quadrature_options
        }
        CovObjectiveRoute::Declined(decline) => {
            warnings.push(format_quadrature_declined_warning(
                fit_method,
                cov_objective_method(options),
                decline,
            ));
            options
        }
    };
    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    // The estimator the returned matrix actually is: `routed`, unless `compute_covariance`
    // itself replaced it (#1781's `S⁻¹` fallback), which it reports on its output.
    let mut used = routed;
    let cov_timer = std::time::Instant::now();
    let matrix = match compute_covariance_for(
        fit_method, x_hat, template, model, population, eta_hats, h_matrices, kappas, options,
    ) {
        CovarianceStepResult::Success(out) => {
            warnings.extend(out.warnings);
            used = out.method;
            Some(out.matrix)
        }
        CovarianceStepResult::Unusable(msg) => {
            warnings.push(msg);
            None
        }
        CovarianceStepResult::FailedNonPd {
            reason,
            fallback_proposal,
        } => {
            warnings.push(reason);
            sir_fallback_proposal = Some(fallback_proposal);
            None
        }
    };
    CovStepOutcome {
        // `used`: the estimator `compute_covariance` was configured with (`routed`, never the
        // caller's request as it arrived, #1382) — or, where it fell back to `S⁻¹` on its own
        // (#1781), the one it returned. A label naming the request there would describe a
        // matrix that was never built.
        method: published_covariance_method(matrix.as_ref(), used),
        matrix,
        wall_time_secs: cov_timer.elapsed().as_secs_f64(),
        warnings,
        sir_fallback_proposal,
    }
}

/// Gated covariance-step orchestration used by the estimator finalizers: the
/// `run_covariance_step && !is_cancelled` gate around [`run_covariance_step_inner`].
/// When the gate is closed, returns an empty outcome (`matrix = None`,
/// `wall_time_secs = 0.0`, no warnings) — exactly what the old inline `else` arm
/// produced. `pre_msg` is only evaluated/printed when the gate is open.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_covariance_step(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
    pre_msg: Option<&str>,
) -> CovStepOutcome {
    if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
        run_covariance_step_inner(
            x_hat, template, model, population, eta_hats, h_matrices, kappas, options, pre_msg,
        )
    } else {
        CovStepOutcome {
            matrix: None,
            wall_time_secs: 0.0,
            warnings: Vec::new(),
            sir_fallback_proposal: None,
            // No step ran, so no estimator to name (#1382).
            method: None,
        }
    }
}

#[cfg(test)]
#[path = "covariance_salvage_tests.rs"]
mod salvage_tests;

#[cfg(all(test, feature = "nn"))]
#[path = "covariance_nn_reg_tests.rs"]
mod nn_reg_tests;

#[cfg(test)]
mod tests {
    // The moved cov-specific unit tests remain in `outer_optimizer`'s test module
    // (they reach the moved symbols via the cross-module import added there). The
    // `run_covariance_step` gate + match is exercised end-to-end by every
    // estimator finalizer's integration/lib tests.
    use super::{
        diagnostic_omega, packed_param_label, published_covariance_method,
        scale_routed_covariance_method,
    };
    use crate::types::{CovarianceMethod, COV_HESSIAN_MAX_DIM};
    use nalgebra::DMatrix;

    /// Copilot on #2093: a nine-effect Laplace fit with `analytic_cov_hessian = false` is
    /// told about the width clause too, so the remedy sentence (gated on the flag clearing
    /// *every* clause) no longer promises the analytic route. Three effects, the flag alone.
    #[test]
    fn analytic_cov_declines_names_a_subject_too_wide_for_the_exact_anchor() {
        use crate::estimation::cov_diagnostics::CovScopeDecline;
        use crate::types::{EstimationMethod, FitOptions, Population, Subject};
        let model = |n_extra: usize| {
            let extra: String = (0..n_extra)
                .map(|k| format!("  omega ETA_X{k} ~ 0.02\n"))
                .collect();
            let sum: String = (0..n_extra).map(|k| format!(" + ETA_X{k}")).collect();
            crate::parser::model_parser::parse_model_string(&format!(
                "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
                 theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n\
                 omega ETA_KA ~ 0.3\n{extra}  sigma PROP_ERR ~ 0.04\n\
                 [individual_parameters]\n  CL = TVCL * exp(ETA_CL{sum})\n\
                 V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n\
                 [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
                 [error_model]\n  DV ~ proportional(PROP_ERR)\n"
            ))
            .unwrap()
        };
        let population = Population {
            subjects: vec![Subject::default()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let options = FitOptions {
            method: EstimationMethod::Laplace,
            analytic_cov_hessian: false,
            ..FitOptions::default()
        };
        for (n_extra, want) in [
            (0, vec![CovScopeDecline::Disabled]),
            (
                6,
                vec![
                    CovScopeDecline::Disabled,
                    CovScopeDecline::TooManyRandomEffects,
                ],
            ),
        ] {
            let m = model(n_extra);
            assert_eq!(m.n_eta, 3 + n_extra);
            assert_eq!(
                super::analytic_cov_declines(&m, &population, &options, false),
                want,
                "{} η",
                m.n_eta
            );
        }
    }

    /// #2093 review round 2 (finding 7): the R-matrix scope walk reports **every** clause that
    /// fired, in gate order, and the routing predicate (`analytic_r_expected`) agrees with it —
    /// the pair base's `covariance_scope_declines` / `covariance_scope_decline` pinned. A model
    /// carrying three clauses at once against the in-scope control on the same model, whose
    /// walk can only name the per-subject bail.
    #[test]
    fn the_scope_walk_reports_every_clause_and_the_routing_predicate_agrees() {
        use crate::estimation::cov_diagnostics::CovScopeDecline;
        use crate::types::{FitOptions, GradientMethod, Population, Subject};
        let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
                   omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n\
                   CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n\
                   pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let population = Population {
            subjects: vec![Subject::default()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let in_scope = crate::parser::model_parser::parse_model_string(src).unwrap();
        let on = FitOptions::default();
        assert_eq!(
            super::analytic_cov_declines(&in_scope, &population, &on, false),
            vec![CovScopeDecline::PerSubjectBail]
        );
        assert!(super::analytic_r_expected(
            &in_scope,
            &population,
            &on,
            false
        ));

        let mut fd = crate::parser::model_parser::parse_model_string(src).unwrap();
        fd.gradient_method = GradientMethod::Fd;
        // Laplace, so the mixture clause (the quadrature's) fires too.
        let off = FitOptions {
            analytic_cov_hessian: false,
            method: crate::types::EstimationMethod::Laplace,
            ..FitOptions::default()
        };
        assert_eq!(
            super::analytic_cov_declines(&fd, &population, &off, true),
            vec![
                CovScopeDecline::Disabled,
                CovScopeDecline::Mixture,
                CovScopeDecline::GradientFd
            ],
            "every clause, in gate order"
        );
        assert!(!super::analytic_r_expected(&fd, &population, &off, true));
        // Each clause alone moves the predicate.
        assert!(
            !super::analytic_r_expected(&fd, &population, &on, false),
            "gradient = fd"
        );
        assert!(
            !super::analytic_r_expected(&in_scope, &population, &off, false),
            "disabled"
        );
    }

    /// #1613 G4 on the adjoint: the R-matrix scope reads the **call's** `gradient = fd`, not
    /// only the model's — `fit`'s guard (`arm_forced_fd`) and `run_covariance`'s scope
    /// (`with_fit_scope`) — against the unarmed control on the same model.
    #[test]
    fn the_scope_walk_reads_the_calls_gradient_fd() {
        use crate::estimation::cov_diagnostics::CovScopeDecline;
        use crate::types::{FitOptions, GradientMethod, Population, Subject};
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
             omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n\
             CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n\
             pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .unwrap();
        assert_eq!(model.gradient_method, GradientMethod::Auto);
        let population = Population {
            subjects: vec![Subject::default()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let opts = FitOptions::default();
        let has_fd = || {
            super::analytic_cov_declines(&model, &population, &opts, false)
                .contains(&CovScopeDecline::GradientFd)
        };
        assert!(!has_fd(), "control: unarmed, no gradient = fd clause");
        {
            let _fd = crate::types::arm_forced_fd(true);
            assert!(
                has_fd(),
                "fit's guard: the call's gradient = fd declines the R-matrix"
            );
            assert!(!super::analytic_r_expected(
                &model,
                &population,
                &opts,
                false
            ));
        }
        let fd_opts = FitOptions {
            gradient_method: GradientMethod::Fd,
            threads: Some(1),
            ..FitOptions::default()
        };
        let posthoc = crate::api::with_fit_scope(&fd_opts, has_fd).expect("scope");
        assert!(posthoc, "run_covariance's scope must carry gradient = fd");
        assert!(!has_fd(), "the guard and the scope disarm on exit");
    }

    /// On the analytic R-matrix a defaulted `r` above `COV_HESSIAN_MAX_DIM` is not rerouted to
    /// `S`, and neither it nor the pre-fit note quotes the stencil's `2·n²` (warfarin_dcm, 147
    /// free parameters, was told 43218 reconverged objectives for a ~15 s analytic step). The
    /// straddle is `analytic_r` alone; the predicate itself is the declines walk, straddled on
    /// a mixture and on `analytic_cov_hessian`.
    #[test]
    fn the_analytic_r_matrix_is_not_rerouted_or_costed_as_a_stencil() {
        use crate::types::{FitOptions, Population, Subject};
        let n = COV_HESSIAN_MAX_DIM + 1;
        let defaulted = FitOptions {
            run_covariance_step: true,
            covariance_method: CovarianceMethod::Hessian,
            covariance_method_set: false,
            ..FitOptions::default()
        };
        assert_eq!(
            super::routed_covariance_method(n, &defaulted, false, true),
            (CovarianceMethod::Hessian, None)
        );
        let (stencil, warning) = super::routed_covariance_method(n, &defaulted, false, false);
        assert_eq!(stencil, CovarianceMethod::CrossProduct);
        assert!(warning.is_some());
        let explicit = FitOptions {
            covariance_method_set: true,
            ..defaulted.clone()
        };
        assert_eq!(
            super::pre_fit_covariance_cost(n, &explicit, false, super::PredictedRRoute::Analytic),
            None
        );
        assert_eq!(
            super::pre_fit_covariance_cost(n, &explicit, false, super::PredictedRRoute::Stencil),
            Some(Some(20_402))
        );

        let m = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
             omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n\
             CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n\
             pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .unwrap();
        let population = Population {
            subjects: vec![Subject::default()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        assert!(super::analytic_r_expected(
            &m,
            &population,
            &defaulted,
            false
        ));
        // A FOCE/FOCEI mixture has its own exact R (`mixture::mixture_cov_hessian`); under
        // the quadrature it takes the stencil.
        assert!(
            super::analytic_r_expected(&m, &population, &defaulted, true),
            "a FOCE/FOCEI mixture is analytic"
        );
        let laplace = FitOptions {
            method: crate::types::EstimationMethod::Laplace,
            ..defaulted.clone()
        };
        assert!(
            !super::analytic_r_expected(&m, &population, &laplace, true),
            "a Laplace mixture takes the stencil"
        );
        let disabled = FitOptions {
            analytic_cov_hessian: false,
            ..defaulted
        };
        assert!(!super::analytic_r_expected(
            &m,
            &population,
            &disabled,
            false
        ));
    }

    // ── #520 C1: the magnitudes the severity grade is made from ──────────────

    /// The diagonal of a covariance, as the inflation metric consumes it.
    fn diag(m: &DMatrix<f64>) -> Vec<f64> {
        (0..m.nrows()).map(|k| m[(k, k)]).collect()
    }

    /// The #2145 mask from booleans: `true` an identity-packed θ, `false` dimensionless.
    fn theta_flags(flags: &[bool]) -> Vec<super::CoordUnits> {
        flags
            .iter()
            .map(|&f| {
                if f {
                    super::CoordUnits::IdentityTheta
                } else {
                    super::CoordUnits::Dimensionless
                }
            })
            .collect()
    }

    /// `M` with coordinate `j` written in units `k` times larger: row and column `j` divided
    /// by `k`, exactly what a change of units does to an identity-packed θ's block of `R` or `S`.
    fn rescale_coordinate(m: &DMatrix<f64>, j: usize, k: f64) -> DMatrix<f64> {
        let mut out = m.clone();
        for i in 0..m.nrows() {
            out[(i, j)] /= k;
            out[(j, i)] /= k;
        }
        out
    }

    /// Worst entrywise relative gap — per entry, since the entries span `k²` in magnitude and a
    /// gap normalised by the largest would not see the small ones.
    fn max_rel_gap(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let mut worst = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let gap = (x - y).abs() / y.abs();
            assert!(gap.is_finite(), "{x} vs {y}");
            if gap > worst {
                worst = gap;
            }
        }
        worst
    }

    /// #2145: an identity-packed θ in large or small units is not clipped once its coordinate
    /// is flagged, and its inverse is the rescaled inverse of the original — while the same
    /// rescale on an **unflagged** coordinate is still clipped, so an implementation that
    /// equilibrated every coordinate (full Jacobi) fails the second half.
    #[test]
    fn unit_free_floor_ignores_the_units_of_a_flagged_coordinate_only() {
        // Well conditioned: eigenvalues ≈ 0.6 … 4.6, correlations ≤ 0.5.
        let m = DMatrix::from_row_slice(3, 3, &[4.0, 0.6, 0.3, 0.6, 1.0, 0.2, 0.3, 0.2, 0.8]);
        let m_inv = m.clone().try_inverse().expect("PD");
        let flag_1 = [false, true, false];
        let at_k1 = super::invert_psd_with_floor_unit_free(&m, &theta_flags(&flag_1)).expect("PD");
        for k in [1e6, 1e-6] {
            let mk = rescale_coordinate(&m, 1, k);
            // The fixture straddles the floor: unscaled, the rescale alone is clipped.
            let raw = super::invert_psd_with_floor(&mk).expect("PD");
            assert!(
                raw.n_clipped > 0,
                "k = {k}: fixture must be under the raw floor"
            );

            let inv =
                super::invert_psd_with_floor_unit_free(&mk, &theta_flags(&flag_1)).expect("PD");
            assert_eq!(inv.n_clipped, 0, "k = {k}: flagged coordinate was clipped");
            // (Mₖ)⁻¹ = row/column 1 of M⁻¹ multiplied by k.
            let want = rescale_coordinate(&m_inv, 1, 1.0 / k);
            let gap = max_rel_gap(&inv.inverse, &want);
            let gap_ref = max_rel_gap(&inv.unclipped_inverse, &want);
            eprintln!("k = {k:e}: inverse gap {gap:.2e}, reference gap {gap_ref:.2e}");
            // Measured ≤ 6.4e-16 (macOS arm64) at both k: a few roundings per entry.
            assert!(gap < 1e-12 && gap_ref < 1e-12, "k = {k}: {gap} / {gap_ref}");
            // The spectrum the warning reports does not move with the units either.
            for (got, want) in [
                (inv.max_eigenvalue, at_k1.max_eigenvalue),
                (inv.min_eigenvalue, at_k1.min_eigenvalue),
                (inv.floor, at_k1.floor),
            ] {
                assert!(
                    ((got - want) / want).abs() < 1e-12,
                    "k = {k}: {got} vs k = 1 {want}"
                );
            }

            // The same rescale on a coordinate that is not flagged: still clipped.
            let unflagged =
                super::invert_psd_with_floor_unit_free(&mk, &theta_flags(&[true, false, false]))
                    .expect("PD");
            assert!(
                unflagged.n_clipped > 0,
                "k = {k}: an unflagged coordinate escaped the floor"
            );
        }
    }

    /// #2145: the equilibration does not switch the floor off. A flagged coordinate that is
    /// collinear with another (correlation `1 − 1e-14`) is still a flat direction, and so is a
    /// flagged coordinate with no information at all; an uncorrelated tiny diagonal entry on a
    /// dimensionless (unflagged) coordinate is still clipped beside a flagged one.
    #[test]
    fn unit_free_floor_still_clips_a_genuinely_flat_direction() {
        let rho = 1.0 - 1e-14;
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        for k in [1.0, 1e6, 1e-6] {
            let mk = rescale_coordinate(&corr, 1, k);
            let inv = super::invert_psd_with_floor_unit_free(&mk, &theta_flags(&[false, true]))
                .expect("PSD");
            assert_eq!(
                inv.n_clipped, 1,
                "k = {k}: collinear pair escaped the floor"
            );
            let both = super::invert_psd_with_floor_unit_free(&mk, &theta_flags(&[true, true]))
                .expect("PSD");
            assert_eq!(
                both.n_clipped, 1,
                "k = {k}: all-flagged collinear pair escaped"
            );
        }

        let no_info = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![2.0, 0.0]));
        let inv = super::invert_psd_with_floor_unit_free(&no_info, &theta_flags(&[false, true]))
            .expect("PSD");
        assert_eq!(
            inv.n_clipped, 1,
            "a flagged coordinate with no score must stay clipped"
        );

        // Exactly one clip: the flat entry. The flagged θ's anchor is the largest unflagged
        // diagonal; a geometric mean (1e-10 here) drags it onto the floor beside the flat one.
        let flat = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-20, 1.0, 3e8]));
        let inv =
            super::invert_psd_with_floor_unit_free(&flat, &theta_flags(&[false, false, true]))
                .expect("PD");
        assert_eq!(
            inv.n_clipped, 1,
            "a flat dimensionless direction must stay clipped"
        );
    }

    /// Review of #2145: with **every** coordinate flagged (Ω and σ FIX, only identity-packed θ's
    /// free) there is no dimensionless diagonal to anchor on. An anchor read off the flagged
    /// diagonals moves with the units, and `[1e-14]` — `[1]` in units 1e7 larger — fell under
    /// the absolute `1e-12` floor. The straddle: unscaled it is clipped, flagged it is not, and
    /// the inverse is exact; a two-coordinate block in mixed units likewise.
    #[test]
    fn unit_free_floor_is_unit_free_when_every_coordinate_is_flagged() {
        let one = DMatrix::from_element(1, 1, 1e-14);
        assert_eq!(super::invert_psd_with_floor(&one).expect("PD").n_clipped, 1);
        let inv = super::invert_psd_with_floor_unit_free(&one, &theta_flags(&[true])).expect("PD");
        assert_eq!(
            inv.n_clipped, 0,
            "an all-flagged θ in large units was clipped"
        );
        assert!(
            ((inv.inverse[(0, 0)] - 1e14) / 1e14).abs() < 1e-14,
            "{}",
            inv.inverse[(0, 0)]
        );

        let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let m_inv = m.clone().try_inverse().expect("PD");
        for (k0, k1) in [(1e7, 1e-7), (1e8, 1e8)] {
            let mk = rescale_coordinate(&rescale_coordinate(&m, 0, k0), 1, k1);
            let inv = super::invert_psd_with_floor_unit_free(&mk, &theta_flags(&[true, true]))
                .expect("PD");
            assert_eq!(inv.n_clipped, 0, "k = ({k0}, {k1}): clipped");
            let want = rescale_coordinate(&rescale_coordinate(&m_inv, 0, 1.0 / k0), 1, 1.0 / k1);
            let gap = max_rel_gap(&inv.inverse, &want);
            // Measured ≤ 4e-16 (macOS arm64).
            assert!(gap < 1e-12, "k = ({k0}, {k1}): gap {gap}");
        }
    }

    /// The mask the covariance step passes, indexed over the free coordinates: identity-packed
    /// θ's (a negative lower bound) flagged, the Ω Cholesky off-diagonal scaled by its row's
    /// estimated diagonal `L̂_ii` (#2274), everything else dimensionless.
    #[test]
    fn free_coordinate_units_flags_identity_thetas_and_scales_omega_off_diagonals() {
        use super::CoordUnits::{Dimensionless, IdentityTheta, Scaled};
        // Ω = [[0.09, 0.3k], [0.3k, (5k)²]] at k = 1e3: L₁₁ = 0.3, L₂₁ = k, L₂₂ = √(25k² − k²).
        let k = 1e3_f64;
        let omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.3 * k, 0.3 * k, 25.0 * k * k]);
        let t = crate::types::ModelParameters {
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
            theta: vec![5.0, 2.0],
            theta_names: vec!["SLOPE".into(), "TVCL".into()],
            // Identity-packed (negative lower bound), then log-packed (a lower bound of 0).
            theta_lower: vec![-100.0, 0.0],
            theta_upper: vec![100.0, 50.0],
            theta_fixed: vec![false; 2],
            omega: crate::types::OmegaMatrix::from_matrix(
                omega,
                vec!["E1".into(), "E2".into()],
                false,
            ),
            omega_fixed: vec![false; 3],
            sigma: crate::types::SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
            mixture: None,
        };
        let x = crate::estimation::parameterization::pack_params(&t);
        // Packed: θ₀, θ₁, L₁₁, L₂₁, L₂₂, σ.
        assert_eq!(x.len(), 6);
        let units = super::free_coordinate_units(&t, &x, &[0, 1, 2, 3, 4, 5], &[]);
        let want_l11 = 0.3_f64;
        let want_l22 = (25.0 * k * k - k * k).sqrt();
        match units[3] {
            Scaled(s) => assert!(((s - want_l22) / want_l22).abs() < 1e-12, "{s}"),
            u => panic!("L₂₁ is {u:?}"),
        }
        assert!(((t.omega.chol[(1, 0)] - k) / k).abs() < 1e-12);
        assert!((t.omega.chol[(0, 0)] - want_l11).abs() < 1e-12);
        assert_eq!(
            [units[0], units[1], units[2], units[4], units[5]],
            [
                IdentityTheta,
                Dimensionless,
                Dimensionless,
                Dimensionless,
                Dimensionless
            ]
        );
        // With θ₀ FIX the free block starts at θ₁, and the off-diagonal moves to index 2.
        let free = super::free_coordinate_units(&t, &x, &[1, 2, 3, 4, 5], &[]);
        assert_eq!(free[0], Dimensionless);
        assert!(matches!(free[2], Scaled(_)));
    }

    /// A template for the #2273 tests: a log-packed `TVCL`, then `n_weights` identity-packed
    /// θ's laid out as the parser lays out `[covariate_nn]` weights (bounds `±∞`), one Ω and
    /// one σ. Packed: θ…, `L₁₁`, σ.
    fn nn_weight_template(n_weights: usize) -> crate::types::ModelParameters {
        let n = 1 + n_weights;
        let mut theta_names = vec!["TVCL".to_string()];
        theta_names.extend((0..n_weights).map(|j| format!("W_NET_1_1_{}", j + 1)));
        let mut theta_lower = vec![0.001];
        theta_lower.extend(std::iter::repeat_n(f64::NEG_INFINITY, n_weights));
        crate::types::ModelParameters {
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
            theta: (0..n).map(|i| 0.3 + i as f64).collect(),
            theta_names,
            theta_lower,
            theta_upper: vec![f64::INFINITY; n],
            theta_fixed: vec![false; n],
            omega: crate::types::OmegaMatrix::from_matrix(
                DMatrix::from_element(1, 1, 0.09),
                vec!["ETA_CL".into()],
                true,
            ),
            omega_fixed: vec![false],
            sigma: crate::types::SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
            mixture: None,
        }
    }

    /// #2273: a `[covariate_nn]` weight is identity-packed but carries no user units, so the
    /// mask leaves it dimensionless. The straddle: the same θ's are flagged when no weight
    /// range is passed, so a mask that ignored `nn_weights` fails the first half.
    #[test]
    fn free_coordinate_units_leaves_nn_weights_dimensionless() {
        use super::CoordUnits::{Dimensionless, IdentityTheta};
        // θ₀ TVCL (log); θ₁..θ₃ identity-packed, of which only θ₂..θ₃ are passed as the
        // network's weights, so θ₁ stands in for a user θ with a negative lower bound.
        let mut t = nn_weight_template(3);
        let x = crate::estimation::parameterization::pack_params(&t);
        let all: Vec<usize> = (0..x.len()).collect();
        let units = super::free_coordinate_units(&t, &x, &all, &[2..4]);
        assert_eq!(
            units,
            [
                Dimensionless,
                IdentityTheta,
                Dimensionless,
                Dimensionless,
                Dimensionless,
                Dimensionless
            ]
        );
        let unmasked = super::free_coordinate_units(&t, &x, &all, &[]);
        assert_eq!(&unmasked[1..4], &[IdentityTheta; 3]);
        // Indexed over the free block: with θ₁ FIX the first weight moves to index 1.
        t.theta_fixed[1] = true;
        let free = super::free_coordinate_units(&t, &x, &[0, 2, 3, 4, 5], &[2..4]);
        assert_eq!(free[..3], [Dimensionless; 3]);
    }

    /// #2273: a flat `[covariate_nn]` weight — a tiny diagonal, nearly uncorrelated with the
    /// rest, as for a weight feeding a saturated unit — is clipped by the floor exactly as it
    /// was before #2145, so the fit still gets the regularization warning. The straddle:
    /// flagged as an identity θ, the same coordinate is equilibrated up to the anchor and
    /// escapes the floor.
    #[test]
    fn a_flat_nn_weight_is_still_clipped_by_the_floor() {
        let t = nn_weight_template(1);
        let x = crate::estimation::parameterization::pack_params(&t);
        // Packed: TVCL, W, L₁₁, σ. The weight's diagonal is 1e-11, under the floor
        // 1e-10·λmax ≈ 4e-10.
        #[rustfmt::skip]
        let m = DMatrix::from_row_slice(4, 4, &[
            4.0,   1e-12, 0.3, 0.2,
            1e-12, 1e-11, 0.0, 0.0,
            0.3,   0.0,   2.0, 0.1,
            0.2,   0.0,   0.1, 1.0,
        ]);
        let all = [0, 1, 2, 3];
        let raw = super::invert_psd_with_floor(&m).expect("PD");
        assert!(raw.n_clipped > 0, "fixture must be under the raw floor");

        let units = super::free_coordinate_units(&t, &x, &all, &[1..2]);
        let inv = super::invert_psd_with_floor_unit_free(&m, &units).expect("PD");
        assert_eq!(
            inv.n_clipped, raw.n_clipped,
            "flat weight escaped the floor"
        );
        assert_eq!(inv.inverse, raw.inverse);
        assert_eq!(inv.floor, raw.floor);

        let flagged = super::free_coordinate_units(&t, &x, &all, &[]);
        let eq = super::invert_psd_with_floor_unit_free(&m, &flagged).expect("PD");
        assert_eq!(
            eq.n_clipped, 0,
            "straddle: as an identity θ the weight is not clipped"
        );
    }

    /// #2273 end to end on the parser's layout: `CompiledModel::nn_weight_theta_ranges` covers
    /// exactly the network's weight θ's, so a user identity-packed covariate θ is flagged and
    /// every weight is not.
    #[cfg(feature = "nn")]
    #[test]
    fn covariate_nn_weights_are_not_flagged_as_identity_thetas() {
        use super::CoordUnits::{Dimensionless, IdentityTheta};
        let model = crate::parser::model_parser::parse_model_string(
            r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  theta KA_WT(0.1, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [3]
  activation = tanh
  output = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V
  KA = TVKA * exp(KA_WT * (WT - 70) / 70)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#,
        )
        .expect("nn model parses");
        let t = &model.default_params;
        // 2 declared θ + 17 weights (2→3→2: W_1 6 + b_1 3 + W_2 6 + b_2 2).
        assert_eq!(t.theta.len(), 19);
        let ranges = model.nn_weight_theta_ranges();
        assert_eq!(ranges, [2..19]);
        let x = crate::estimation::parameterization::pack_params(t);
        let all: Vec<usize> = (0..x.len()).collect();
        let units = super::free_coordinate_units(t, &x, &all, &ranges);
        assert_eq!(units[..2], [Dimensionless, IdentityTheta]);
        assert!(
            units[2..19].iter().all(|&u| u == Dimensionless),
            "{units:?}"
        );
        // Every weight is identity-packed, so without the ranges each would be flagged.
        let unmasked = super::free_coordinate_units(t, &x, &all, &[]);
        assert!(unmasked[2..19].iter().all(|&u| u == IdentityTheta));
    }

    /// #2274: an Ω Cholesky off-diagonal carries its row η's units, and so does that row's
    /// diagonal `L̂_ii`, so scaling the coordinate by `L̂_ii` makes the floor's decision
    /// unit-free. The straddle: the same rescale, unscaled, is clipped; scaled by `k·L̂_ii` it is
    /// not, and the inverse is the rescaled inverse of the original. A coordinate whose scale is
    /// genuinely tiny is still clipped — the scale is the estimate, not the matrix.
    #[test]
    fn unit_free_floor_scales_an_off_diagonal_by_its_row_diagonal() {
        use super::CoordUnits::{Dimensionless, Scaled};
        let m = DMatrix::from_row_slice(3, 3, &[4.0, 0.6, 0.3, 0.6, 1.0, 0.2, 0.3, 0.2, 0.8]);
        let m_inv = m.clone().try_inverse().expect("PD");
        let l_ii = 0.7;
        let at_k1 = super::invert_psd_with_floor_unit_free(
            &m,
            &[Dimensionless, Scaled(l_ii), Dimensionless],
        )
        .expect("PD");
        for k in [1e6, 1e-6] {
            let mk = rescale_coordinate(&m, 1, k);
            assert!(
                super::invert_psd_with_floor(&mk).expect("PD").n_clipped > 0,
                "k = {k}: fixture must be under the raw floor"
            );
            let units = [Dimensionless, Scaled(k * l_ii), Dimensionless];
            let inv = super::invert_psd_with_floor_unit_free(&mk, &units).expect("PD");
            assert_eq!(inv.n_clipped, 0, "k = {k}: scaled off-diagonal was clipped");
            let want = rescale_coordinate(&m_inv, 1, 1.0 / k);
            let gap = max_rel_gap(&inv.inverse, &want);
            eprintln!("k = {k:e}: inverse gap {gap:.2e}");
            assert!(gap < 1e-12, "k = {k}: {gap}");
            for (got, want) in [
                (inv.max_eigenvalue, at_k1.max_eigenvalue),
                (inv.min_eigenvalue, at_k1.min_eigenvalue),
                (inv.floor, at_k1.floor),
            ] {
                assert!(
                    ((got - want) / want).abs() < 1e-12,
                    "k = {k}: {got} vs k = 1 {want}"
                );
            }
            // An unscaled coordinate in the same units: still clipped.
            let unscaled = super::invert_psd_with_floor_unit_free(&mk, &[Dimensionless; 3]);
            assert!(unscaled.expect("PD").n_clipped > 0, "k = {k}");
        }

        // The θ anchor is read after the off-diagonal's scaling: a large-unit off-diagonal is the
        // largest raw diagonal here, and anchoring on it would drag the θ to a unit-dependent g.
        let three = DMatrix::from_row_slice(3, 3, &[1.0, 0.1, 0.0, 0.1, 2.0, 0.3, 0.0, 0.3, 0.5]);
        let base = super::invert_psd_with_floor_unit_free(
            &three,
            &[super::CoordUnits::IdentityTheta, Dimensionless, Scaled(1.0)],
        )
        .expect("PD");
        let small = rescale_coordinate(&three, 2, 1e-9);
        let moved = super::invert_psd_with_floor_unit_free(
            &small,
            &[
                super::CoordUnits::IdentityTheta,
                Dimensionless,
                Scaled(1e-9),
            ],
        )
        .expect("PD");
        assert!(((moved.max_eigenvalue - base.max_eigenvalue) / base.max_eigenvalue).abs() < 1e-12);

        // A row diagonal estimated at ~0 (ω_ii collapsed): the off-diagonal is a flat direction
        // in unit-free terms and the floor still clips it.
        let inv = super::invert_psd_with_floor_unit_free(
            &m,
            &[Dimensionless, Scaled(1e-8), Dimensionless],
        )
        .expect("PD");
        assert_eq!(
            inv.n_clipped, 1,
            "a collapsed row's off-diagonal escaped the floor"
        );
        // A non-positive or non-finite scale is ignored rather than trusted.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let inv = super::invert_psd_with_floor_unit_free(
                &m,
                &[Dimensionless, Scaled(bad), Dimensionless],
            )
            .expect("PD");
            assert_eq!(
                inv.inverse,
                super::invert_psd_with_floor(&m).expect("PD").inverse
            );
        }
    }

    #[test]
    fn invert_psd_with_floor_reports_the_spectrum_ends_and_no_inflation_when_clean() {
        // A positive-definite input is inverted exactly: nothing is clipped, so the floor
        // manufactured no variance and the reference equals the inverse — an inflation of
        // exactly 1.0, the value that grades `Minor` however many free parameters there are.
        let m = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 1.0]));
        let inv = super::invert_psd_with_floor(&m).expect("PD input inverts");
        assert_eq!(inv.n_clipped, 0);
        assert!(
            (inv.max_eigenvalue - 4.0).abs() < 1e-12,
            "{}",
            inv.max_eigenvalue
        );
        assert!(
            (inv.min_eigenvalue - 1.0).abs() < 1e-12,
            "{}",
            inv.min_eigenvalue
        );
        assert_eq!(
            super::worst_variance_inflation(&diag(&inv.inverse), &diag(&inv.unclipped_inverse)),
            1.0
        );
    }

    #[test]
    fn invert_psd_with_floor_measures_the_variance_the_floor_manufactured() {
        // The quantity #520 added, checked against a hand computation rather than against a
        // second implementation. Spectrum {100, -1e-3} rotated 45°, so BOTH coordinates load
        // half their mass on the negative direction — a fixture where every eigenvector
        // component is non-zero, so the ratio is finite and can be written down:
        //
        //   floor      = 100 * 1e-10 = 1e-8            (max_eig * 1e-10, above the 1e-12 floor)
        //   inv[k,k]   = 0.5/100 + 0.5/1e-8 = 5e7 + 0.005
        //   unclipped  = 0.5/100             = 0.005
        //   inflation  = (5e7 + 0.005) / 0.005 = 1e10 + 1
        //
        // Count grades this "1 of 2 clipped" — 50%, "moderate" under the old tiers. Magnitude
        // grades it severe, which is the whole point.
        let a = 0.5 * (100.0 + -1e-3);
        let b = 0.5 * (100.0 - -1e-3);
        let m = DMatrix::from_row_slice(2, 2, &[a, b, b, a]);
        let inv = super::invert_psd_with_floor(&m).expect("has positive curvature");
        assert_eq!(inv.n_clipped, 1);
        assert!(
            (inv.max_eigenvalue - 100.0).abs() < 1e-9,
            "{}",
            inv.max_eigenvalue
        );
        assert!(
            (inv.min_eigenvalue + 1e-3).abs() < 1e-9,
            "{}",
            inv.min_eigenvalue
        );
        assert!((inv.floor - 1e-8).abs() < 1e-18, "{}", inv.floor);
        // The unclipped reference drops the floored direction entirely, so its diagonal is the
        // hand-computed `0.5/100` and not `0.5/100 + 0.5/floor`.
        let unclipped = diag(&inv.unclipped_inverse);
        assert!(
            (unclipped[0] - 0.005).abs() / 0.005 < 1e-9,
            "unclipped reference {unclipped:?}"
        );
        let got = super::worst_variance_inflation(&diag(&inv.inverse), &unclipped);
        let expected = (0.5 / 100.0 + 0.5 / 1e-8) / (0.5 / 100.0);
        let rel = (got - expected).abs() / expected;
        assert!(rel < 1e-6, "inflation {got} vs hand-computed {expected}");
        // And the grade that number produces, end to end.
        assert_eq!(
            crate::estimation::cov_diagnostics::grade(
                inv.min_eigenvalue.abs() / inv.max_eigenvalue,
                got,
            )
            .severity,
            crate::estimation::cov_diagnostics::CovSeverity::Severe,
        );
    }

    #[test]
    fn a_coordinate_supported_only_by_floored_directions_reports_unbounded_inflation() {
        // The other reachable end: an axis-aligned spectrum, so coordinate 1 loads *entirely*
        // on the clipped direction and the unclipped part of the spectrum supports no variance
        // for it at all. The ratio is `+∞`, which `format_regularized_warning` words rather
        // than printing as `inf`.
        let m = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![100.0, -1e-3]));
        let inv = super::invert_psd_with_floor(&m).expect("has positive curvature");
        assert_eq!(inv.n_clipped, 1);
        let got =
            super::worst_variance_inflation(&diag(&inv.inverse), &diag(&inv.unclipped_inverse));
        assert!(got.is_infinite(), "{got}");
    }

    #[test]
    fn worst_variance_inflation_guards_each_kill_their_own_mutation() {
        // Both guards, each asserted where **only** it can produce the right answer, so
        // deleting either one reddens this test on its own line. The three-guard version this
        // replaced had a third condition that rejected exactly the inputs the first already
        // did — deleting it left the suite green, which is AGENTS.md's redundant-gate hole and
        // is why the guards were cut down rather than added to.
        let live = 9.0;

        // Guard 1 — the `FIX`ed-parameter cell. A pinned coordinate reports SE 0 on *both*
        // sides. Without `!(v > 0.0)` the pair `(0.0, 0.0)` takes the `v_ref <= 0` arm and
        // every fixed parameter in the model reports `unbounded`.
        assert_eq!(
            super::worst_variance_inflation(&[0.0, live], &[0.0, 3.0]),
            3.0,
            "a FIXed coordinate is zero on both sides and must not read as unbounded"
        );

        // Guard 2 — a diverged solve. Without `!v.is_finite()` this is `inf / 1.0 = inf`.
        assert_eq!(
            super::worst_variance_inflation(&[f64::INFINITY, live], &[1.0, 3.0]),
            3.0,
            "a non-finite returned variance is a broken covariance, not an inflation"
        );

        // What falls through on its own, with no guard: a NaN (every comparison against it is
        // false) and a negative reconstructed variance.
        assert_eq!(
            super::worst_variance_inflation(&[f64::NAN, -1.0, live], &[1.0, 1.0, 3.0]),
            3.0
        );

        // A zero reference against a live value is the meaningful unbounded answer — the cell
        // guard 1 must not swallow.
        assert!(super::worst_variance_inflation(&[1.0], &[0.0]).is_infinite());
        // Never below 1: a floor cannot shrink a variance, and a ratio under 1 is noise.
        assert_eq!(super::worst_variance_inflation(&[1.0], &[2.0]), 1.0);
    }

    #[test]
    fn the_inflation_metric_moves_with_s_under_the_sandwich_estimator() {
        // #1508 review §1. The metric used to be a diagonal of packed-space `R⁻¹`, so the same
        // regularized `R` printed the same inflation for every `S` — even though the returned
        // covariance under `covariance_method = rsr` is `R⁻¹ S R⁻¹` and `S` can suppress or
        // concentrate the floored eigendirection. One `R`, two `S`, and the number must differ.
        //
        // `R` is axis-aligned with a floored second direction, so `R⁻¹ = diag(1/100, 1/floor)`
        // and the floored direction is coordinate 1.
        let r = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![100.0, -1e-3]));
        let inv = super::invert_psd_with_floor(&r).expect("has positive curvature");
        assert_eq!(inv.n_clipped, 1);
        let r_inv = inv.inverse.clone() * 2.0;
        let r_inv_ref = inv.unclipped_inverse.clone() * 2.0;

        let inflation_for = |s: &DMatrix<f64>| {
            // Through `combine_covariance_and_reference`, which is the wiring
            // `compute_covariance` uses: both sides take the *selected* estimator. Calling
            // `combine_covariance` twice here instead would test a second spelling of the
            // thing under test rather than the thing itself.
            let (cov, cov_ref) = super::combine_covariance_and_reference(
                CovarianceMethod::Sandwich,
                r_inv.clone(),
                Some(r_inv_ref.clone()),
                s,
                &[super::CoordUnits::Dimensionless; 2],
            )
            .expect("sandwich never inverts S");
            let cov_ref = cov_ref.expect("a reference was supplied");
            super::worst_variance_inflation(&diag(&cov), &diag(&cov_ref))
        };

        // `s_blind` puts no score mass on the floored direction, so the sandwich's floored
        // column is annihilated and the reported variance is not inflated at all.
        let s_blind = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 0.0]));
        // `s_loaded` puts mass there, so the floor's `1/floor` is squared into the answer.
        let s_loaded = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 1.0]));

        let blind = inflation_for(&s_blind);
        let loaded = inflation_for(&s_loaded);
        assert_eq!(
            blind, 1.0,
            "an S with no mass on the floored direction leaves the returned variance untouched"
        );
        assert!(
            loaded > 1e12,
            "an S loaded on the floored direction squares 1/floor into the answer: {loaded}"
        );
        assert_ne!(
            blind, loaded,
            "the metric must be a function of the returned covariance, not of R alone"
        );

        // And the metric the review rejected, computed here so the difference is on the
        // record: the diagonals of packed `R⁻¹` against packed `R⁻¹_ref`. It does not take `S`
        // at all, so it reports the same large inflation for both — including for the `S` that
        // annihilates the floored direction entirely, where nothing the user reads moved.
        let packed_only = super::worst_variance_inflation(&diag(&r_inv), &diag(&r_inv_ref));
        assert!(packed_only.is_infinite(), "{packed_only}");
        assert_ne!(
            packed_only, blind,
            "the packed-R metric cannot see that this S left the reported variance untouched"
        );

        // The reference's *shape*, asserted against a hand-written `R⁻¹_ref S R⁻¹_ref`. Without
        // this, replacing the reference with a bare `R⁻¹_ref` passes everything above — the
        // ratios above happen to land on the same side of every bound — so this is the
        // assertion that actually pins "the reference goes through the same estimator".
        // `S` is deliberately not the identity, so `R⁻¹_ref S R⁻¹_ref` differs from both
        // `R⁻¹_ref` and `R⁻¹_ref R⁻¹_ref`.
        let s_asym = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0, 1.0]));
        let (_, reference) = super::combine_covariance_and_reference(
            CovarianceMethod::Sandwich,
            r_inv.clone(),
            Some(r_inv_ref.clone()),
            &s_asym,
            &[super::CoordUnits::Dimensionless; 2],
        )
        .expect("sandwich never inverts S");
        let reference = reference.expect("a reference was supplied");
        let want = &r_inv_ref * &s_asym * &r_inv_ref;
        assert!(
            (0..2).all(|k| (reference[(k, k)] - want[(k, k)]).abs() <= 1e-12 * want[(k, k)].abs()),
            "reference {:?} is not R^-1_ref S R^-1_ref {:?}",
            diag(&reference),
            diag(&want)
        );
        assert_ne!(
            diag(&reference),
            diag(&r_inv_ref),
            "a bare R^-1_ref reference is the spelling this test exists to reject"
        );

        // `None` reference means "nothing to compare", and must not invent one.
        let (_, none_ref) = super::combine_covariance_and_reference(
            CovarianceMethod::Sandwich,
            r_inv.clone(),
            None,
            &s_loaded,
            &[super::CoordUnits::Dimensionless; 2],
        )
        .expect("sandwich never inverts S");
        assert!(none_ref.is_none());
    }

    #[test]
    fn the_inflation_metric_passes_through_the_block_omega_jacobian() {
        // #1508 review §1's second half. The reported standard error of a block-Ω element is
        // `g^T C_ω g` over the packed Cholesky coordinates, so the transform **mixes**
        // coordinates — a packed-space diagonal ratio is not the number the user reads.
        //
        // The fixture is built so the two metrics must disagree: `cov` and `cov_ref` have
        // *identical* diagonals (a packed-diagonal metric reports exactly 1.0) and differ only
        // in the off-diagonal coupling between the two packed coordinates that `ω₂₂ = L₂₁² +
        // L₂₂²` loads on. Both halves are asserted in one test, so a formatter that drops the
        // delta transform reddens rather than quietly agreeing.
        let omega = crate::types::OmegaMatrix::from_matrix(
            DMatrix::from_row_slice(2, 2, &[0.09, 0.03, 0.03, 0.04]),
            vec!["E1".into(), "E2".into()],
            false,
        );
        let template = crate::types::ModelParameters {
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 2],
            sigma: crate::types::SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
            mixture: None,
        };
        // Packed layout: theta(1) | omega lower triangle L₁₁, L₂₁, L₂₂ (3) | sigma(1).
        let n = 5;
        let free_idx: Vec<usize> = (0..n).collect();
        let cov_ref = DMatrix::<f64>::identity(n, n);
        let mut cov = cov_ref.clone();
        // Couple the L₂₁ and L₂₂ coordinates, leaving every diagonal at 1.
        cov[(2, 3)] = 0.5;
        cov[(3, 2)] = 0.5;

        // A packed-space diagonal metric is blind to this by construction.
        assert_eq!(
            super::worst_variance_inflation(&diag(&cov), &diag(&cov_ref)),
            1.0,
            "the fixture's packed diagonals are equal on purpose"
        );

        // Hand computation, not a second implementation. Ω = L Lᵀ with L₁₁ = 0.3,
        // L₂₁ = 0.1, L₂₂ = √0.03; ω₂₂ = L₂₁² + L₂₂², and the packed coordinates are
        // (L₂₁, log L₂₂), so g = (2·L₂₁, 2·L₂₂²) = (0.2, 0.06).
        //   var_ref = g₁² + g₂²                    = 0.04 + 0.0036 = 0.0436
        //   var     = var_ref + 2·g₁·g₂·cov[2,3]   = 0.0436 + 0.012 = 0.0556
        let g1 = 2.0 * 0.1;
        let g2 = 2.0 * 0.03;
        let var_ref = g1 * g1 + g2 * g2;
        let expected = (var_ref + 2.0 * g1 * g2 * 0.5) / var_ref;
        let got = super::reported_variance_inflation(&cov, Some(&cov_ref), &free_idx, n, &template);
        assert!(
            (got - expected).abs() / expected < 1e-9,
            "reported inflation {got} vs hand-computed {expected}"
        );
        assert!(
            got > 1.0,
            "the Jacobian must be able to see a coupling the diagonals hide"
        );

        // And the `None` reference — no floor fired, or `covariance_method = s`, which never
        // inverts R — reports no inflation rather than dividing by nothing.
        assert_eq!(
            super::reported_variance_inflation(&cov, None, &free_idx, n, &template),
            1.0
        );
    }

    // ── #1382: the estimator label published alongside the matrix ────────────

    /// **L1 — the label names the estimator that ran, not the one requested.**
    ///
    /// This is the composition the whole field exists for. At scale a *defaulted*
    /// `covariance_method = r` is routed onto the cross-product (#1064), so a
    /// label read back off `FitOptions::covariance_method` would report `r` for
    /// SEs that came out of `S⁻¹`. The two are asserted to disagree first, so the
    /// case cannot quietly become one where both spellings are the same answer.
    ///
    /// Composed here rather than driven through `run_covariance_step_inner`, for
    /// the same reason the routing tests above are: reaching the branch needs a
    /// fit with several hundred free parameters.
    ///
    /// Mutation: label from `requested` instead of `routed` → the second
    /// assertion fires.
    #[test]
    fn the_published_label_names_the_routed_estimator_not_the_requested_one() {
        let requested = CovarianceMethod::Hessian;
        let (routed, _warning) = scale_routed_covariance_method(
            COV_HESSIAN_MAX_DIM + 1,
            requested,
            /* explicitly_set */ false,
            /* has_priors */ false,
        );
        assert_ne!(
            routed, requested,
            "the premise: this input must be one where the router actually swaps the \
             estimator, or the assertion below is satisfied by either spelling"
        );

        let matrix = DMatrix::<f64>::identity(2, 2);
        assert_eq!(
            published_covariance_method(Some(&matrix), routed),
            Some(CovarianceMethod::CrossProduct),
            "the label must describe the matrix that was produced (S⁻¹), not the \
             estimator the caller asked for (R⁻¹)"
        );
    }

    /// **L2 — no matrix, no estimator.** A failed, skipped or SIR-fallback step
    /// publishes no SEs, no eigenvalues and no condition number, so there is
    /// nothing for a name to describe; labelling one would invite a reader to
    /// compare a figure that was never computed.
    #[test]
    fn a_step_that_produced_no_matrix_publishes_no_estimator() {
        for m in [
            CovarianceMethod::Hessian,
            CovarianceMethod::CrossProduct,
            CovarianceMethod::Sandwich,
        ] {
            assert_eq!(published_covariance_method(None, m), None, "{m:?}");
        }
    }

    // ── #1064: routing the covariance step away from `R` at scale ───────────
    //
    // Exercised here rather than through `run_covariance_step_inner`, which
    // would need a converged fit with several hundred free parameters.

    #[test]
    fn ordinary_dimensions_keep_the_requested_covariance_method() {
        for method in [
            CovarianceMethod::Hessian,
            CovarianceMethod::CrossProduct,
            CovarianceMethod::Sandwich,
        ] {
            for explicit in [false, true] {
                let (routed, warning) =
                    scale_routed_covariance_method(COV_HESSIAN_MAX_DIM, method, explicit, false);
                assert_eq!(routed, method);
                assert!(
                    warning.is_none(),
                    "no warning at the threshold: {warning:?}"
                );
            }
        }
    }

    #[test]
    fn a_defaulted_hessian_routes_to_the_cross_product_at_scale() {
        let n = COV_HESSIAN_MAX_DIM + 1;
        assert_eq!(n, 101, "the counts below are 2·n² at n = 101");
        let (routed, warning) =
            scale_routed_covariance_method(n, CovarianceMethod::Hessian, false, false);
        assert_eq!(routed, CovarianceMethod::CrossProduct);
        let warning = warning.expect("the substitution must be reported");
        assert!(warning.contains("score cross-product"), "{warning}");
        // The stencil count is the whole argument for switching, so it has to
        // be in the message rather than left for the reader to work out.
        assert!(
            warning.contains("20402"),
            "message must name the stencil size: {warning}"
        );
    }

    #[test]
    fn an_explicit_hessian_is_honoured_at_scale_but_warned_about() {
        let n = COV_HESSIAN_MAX_DIM + 1;
        let (routed, warning) =
            scale_routed_covariance_method(n, CovarianceMethod::Hessian, true, false);
        assert_eq!(
            routed,
            CovarianceMethod::Hessian,
            "an explicit `covariance_method = r` is the user's call"
        );
        let warning = warning.expect("the cost must still be reported");
        assert!(warning.contains("covariance_method = r"), "{warning}");
        assert!(warning.contains("20402"), "2·101²: {warning}");
    }

    /// A priored fit at scale stays on `R` instead of being auto-routed to the
    /// cross-product (#254).
    ///
    /// `S` is a sum of per-*subject* scores and a prior has no subject
    /// decomposition, so the usual large-problem fallback would silently report
    /// unpenalized standard errors for a penalized fit. The straddle is the
    /// pair: identical `n` and identical `explicitly_set`, differing only in
    /// `has_priors`, so the two arms land on *different* methods. Without both
    /// sides this passes on an implementation that never routes at all.
    #[test]
    fn a_priored_fit_stays_on_the_hessian_at_scale() {
        let n = COV_HESSIAN_MAX_DIM + 1;

        let (routed, warning) =
            scale_routed_covariance_method(n, CovarianceMethod::Hessian, false, true);
        assert_eq!(
            routed,
            CovarianceMethod::Hessian,
            "a prior must keep the covariance step on R"
        );
        let warning = warning.expect("staying on the slow estimator must be reported");
        // The message has to say *why* the usual fallback was skipped, or the
        // user reads it as the plain large-problem warning and switches to `s`
        // by hand — which is the outcome this branch exists to prevent.
        assert!(warning.contains("parameter priors"), "{warning}");
        assert!(warning.contains("cannot represent a prior"), "{warning}");
        // And the cost, as in the other two arms.
        assert!(
            warning.contains("20402"),
            "message must name the stencil size: {warning}"
        );

        // The straddle: same n, same `explicitly_set`, no prior — routed away.
        let (unpriored, _) =
            scale_routed_covariance_method(n, CovarianceMethod::Hessian, false, false);
        assert_eq!(
            unpriored,
            CovarianceMethod::CrossProduct,
            "without a prior the same inputs must still route to S"
        );

        // Below the threshold the prior changes nothing: the guard is about
        // scale, and a small priored fit is not warned at all.
        let (small, small_warning) = scale_routed_covariance_method(
            COV_HESSIAN_MAX_DIM,
            CovarianceMethod::Hessian,
            false,
            true,
        );
        assert_eq!(small, CovarianceMethod::Hessian);
        assert!(small_warning.is_none(), "{small_warning:?}");
    }

    #[test]
    fn a_non_hessian_request_is_never_rerouted() {
        // Only a defaulted `r` is rerouted: `s` builds no `R`, and `rsr` is an explicit
        // request for the sandwich, which no cheaper estimator computes. `s` is not warned
        // about either; `rsr` is (#2055), in the test below.
        for method in [CovarianceMethod::CrossProduct, CovarianceMethod::Sandwich] {
            for explicit in [false, true] {
                for priors in [false, true] {
                    let (routed, _) = scale_routed_covariance_method(
                        COV_HESSIAN_MAX_DIM * 8,
                        method,
                        explicit,
                        priors,
                    );
                    assert_eq!(routed, method, "explicit {explicit}, priors {priors}");
                }
            }
        }
        let (_, warning) = scale_routed_covariance_method(
            COV_HESSIAN_MAX_DIM * 8,
            CovarianceMethod::CrossProduct,
            false,
            false,
        );
        assert!(warning.is_none(), "`s` builds no R: {warning:?}");
    }

    /// #2055: `rsr` builds the same `R` as `r`, so above the threshold it gets the
    /// explicit-`r` cost warning, naming `rsr` — before the fix it returned before any cost
    /// branch, because the gate read `requested == Hessian`. The straddle is `n`: 100 is
    /// silent, 101 warns. The warning does not depend on `explicitly_set`, since `rsr` is
    /// never the default. With priors the `s` advice is replaced, as for `r`: `S` cannot
    /// carry a prior.
    ///
    /// Mutations: gate on `Hessian` again → the 101 cell has no warning; route `rsr` to `s`
    /// → the `routed` assertion fires; give the priored arm the unpriored text → the
    /// `covariance = false` assertion fires.
    #[test]
    fn an_rsr_request_is_honoured_at_scale_but_warned_about() {
        let n = COV_HESSIAN_MAX_DIM + 1;
        for explicit in [false, true] {
            let (routed, warning) =
                scale_routed_covariance_method(n, CovarianceMethod::Sandwich, explicit, false);
            assert_eq!(routed, CovarianceMethod::Sandwich, "rsr is never rerouted");
            let warning = warning.expect("rsr builds R, so its cost must be reported");
            assert!(warning.contains("covariance_method = rsr"), "{warning}");
            assert!(warning.contains("20402"), "2·101²: {warning}");
            assert!(
                warning.contains("Set `covariance_method = s`"),
                "the way out must be named: {warning}"
            );
        }

        let (routed, warning) =
            scale_routed_covariance_method(n, CovarianceMethod::Sandwich, true, true);
        assert_eq!(routed, CovarianceMethod::Sandwich);
        let warning = warning.expect("priored rsr at scale must be reported");
        assert!(warning.contains("covariance_method = rsr"), "{warning}");
        assert!(warning.contains("parameter priors"), "{warning}");
        assert!(warning.contains("cannot represent a prior"), "{warning}");
        assert!(warning.contains("covariance = false"), "{warning}");
        assert!(
            !warning.contains("Set `covariance_method = s`"),
            "`s` drops the prior, so it must not be advised: {warning}"
        );
        assert!(warning.contains("20402"), "{warning}");

        // The straddle: at the threshold nothing is said.
        for priors in [false, true] {
            let (small, small_warning) = scale_routed_covariance_method(
                COV_HESSIAN_MAX_DIM,
                CovarianceMethod::Sandwich,
                true,
                priors,
            );
            assert_eq!(small, CovarianceMethod::Sandwich);
            assert!(small_warning.is_none(), "{small_warning:?}");
        }
    }

    /// #1803: `fit()`'s pre-fit cost note quotes `R`'s `2·n²` exactly when the step will
    /// build `R`. The straddles share every input but the one that moves the answer: the
    /// estimator at `n = 31` (`r` and `rsr` warn, `s` does not), the threshold (30 is silent),
    /// the step switch, and — above `COV_HESSIAN_MAX_DIM` — whether `r` was explicit or priored,
    /// which is what decides if the step runs `R` or the routed `S⁻¹`.
    #[test]
    fn the_pre_fit_cost_note_counts_r_only_when_the_step_builds_r() {
        let cost = |n, o: &crate::types::FitOptions, p| {
            super::pre_fit_covariance_cost(n, o, p, super::PredictedRRoute::Stencil)
        };
        let opts = |method: CovarianceMethod, set: bool| crate::types::FitOptions {
            run_covariance_step: true,
            covariance_method: method,
            covariance_method_set: set,
            ..Default::default()
        };
        // 2·31² = 1922.
        for method in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
            assert_eq!(
                cost(31, &opts(method, true), false),
                Some(Some(1922)),
                "{method:?}"
            );
        }
        assert_eq!(
            cost(31, &opts(CovarianceMethod::CrossProduct, true), false),
            None,
            "s builds no R"
        );
        assert_eq!(
            cost(30, &opts(CovarianceMethod::Hessian, true), false),
            None
        );
        let off = crate::types::FitOptions {
            run_covariance_step: false,
            ..opts(CovarianceMethod::Hessian, true)
        };
        assert_eq!(cost(31, &off, false), None, "no step, no note");

        let n = COV_HESSIAN_MAX_DIM + 1;
        assert_eq!(
            cost(n, &opts(CovarianceMethod::Hessian, false), false),
            None,
            "a defaulted r is routed to s at scale"
        );
        assert_eq!(
            cost(n, &opts(CovarianceMethod::Hessian, true), false),
            Some(Some(20_402))
        );
        assert_eq!(
            cost(n, &opts(CovarianceMethod::Hessian, false), true),
            Some(Some(20_402)),
            "a priored fit stays on R"
        );
        assert_eq!(
            cost(
                1 << (usize::BITS / 2),
                &opts(CovarianceMethod::Hessian, true),
                false
            ),
            Some(None),
            "an overflowed count still warns, without a number"
        );
    }

    /// #1803 / #2054: the note's text, pinned whole on every arm so deleting any sentence
    /// reddens it, and still classified as the `covariance_step` Info it was before the count
    /// changed. The salvage arm names the declined subset; the stencil arm does not.
    #[test]
    fn the_pre_fit_cost_note_quotes_the_r_count_or_says_it_overflowed() {
        use super::pre_fit_covariance_cost_note as note;
        use super::PredictedRRoute;
        use crate::types::{classify_warning, WarningCode, WarningSeverity};
        let salvage = PredictedRRoute::Salvage {
            declined: 2,
            n_subjects: 20,
        };
        let counted = note(31, Some(Some(1922)), PredictedRRoute::Stencil).expect("a note is due");
        assert_eq!(
            counted,
            "Covariance step: 31 parameters → 1922 OFV evaluations expected (finite-difference \
             Hessian). This may take several minutes on complex models."
        );
        let overflowed = note(31, Some(None), PredictedRRoute::Stencil).expect("a note is due");
        assert_eq!(
            overflowed,
            "Covariance step: 31 parameters → 2n² OFV evaluations expected (finite-difference \
             Hessian). Estimate exceeds usize range; expect this to be very slow."
        );
        let salvaged = note(31, Some(Some(1922)), salvage).expect("a note is due");
        assert_eq!(
            salvaged,
            "Covariance step: 31 parameters → 1922 OFV evaluations expected over 2 of 20 \
             subjects (finite-difference Hessian for the subjects outside the analytic \
             R-matrix; the rest are analytic). This may take several minutes on complex models."
        );
        let salvaged_overflow = note(31, Some(None), salvage).expect("a note is due");
        assert_eq!(
            salvaged_overflow,
            "Covariance step: 31 parameters → 2n² OFV evaluations expected over 2 of 20 \
             subjects (finite-difference Hessian for the subjects outside the analytic \
             R-matrix; the rest are analytic). Estimate exceeds usize range; expect this to be \
             very slow."
        );
        for msg in [&counted, &overflowed, &salvaged, &salvaged_overflow] {
            let entry = classify_warning(msg);
            assert_eq!(
                (entry.severity, entry.category),
                (WarningSeverity::Info, WarningCode::CovarianceStep),
                "{msg}"
            );
            // ferx-r's `ferx_get_warnings()` advice arm matches this exact token.
            assert!(msg.contains("OFV evaluations"), "{msg}");
        }
        assert_eq!(note(31, None, salvage), None, "no cost, no note");
    }

    /// A variance mixture (per-class Ω/Σ overrides) so the packed vector carries
    /// the override tail. Packed order: [TVCL, TVV, MIXL, BWT | ω(ETA_CL) |
    /// σ(EPS) | ω_MIX2 | σ_MIX2].
    const MIX_OVERRIDE_MODEL: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.2, -10.0, 10.0)
  theta BWT(0.05, -5.0, 5.0)
  omega ETA_CL ~ 0.06
  sigma EPS ~ 0.02

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)
  omega(2) ETA_CL ~ 0.15
  sigma(2) EPS ~ 0.03

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    #[test]
    fn packed_param_label_names_mixture_overrides() {
        // #983 Phase 6: the per-class Ω/Σ override coordinates must carry real
        // `omega[<eta>_MIX{k}]` / `sigma[<sigma>_MIX{k}]` names (matching
        // `coordinate_names`), not the old `packed[N]` fallback. The mixing-logit
        // coefficients (MIXL, BWT) are ordinary thetas and label via the theta
        // branch.
        let model = crate::parser::model_parser::parse_model_string(MIX_OVERRIDE_MODEL).unwrap();
        let t = &model.default_params;
        // Base segment: 4 theta + 1 omega + 1 sigma = indices 0..6.
        assert_eq!(packed_param_label(0, t), "theta[TVCL]");
        assert_eq!(packed_param_label(2, t), "theta[MIXL]");
        assert_eq!(packed_param_label(3, t), "theta[BWT]");
        assert_eq!(packed_param_label(4, t), "omega[ETA_CL, ETA_CL]");
        assert_eq!(packed_param_label(5, t), "sigma[1]");
        // Override tail: ω(2) then σ(2), class 2 (1-based).
        assert_eq!(packed_param_label(6, t), "omega[ETA_CL_MIX2]");
        assert_eq!(packed_param_label(7, t), "sigma[EPS_MIX2]");
        // Past the override tail → generic fallback (defensive).
        assert_eq!(packed_param_label(8, t), "packed[8]");
    }

    #[test]
    fn diagnostic_omega_picks_collapsed_class_override() {
        // #984 regression: when a per-class `omega(k)` override collapses but the
        // base (class-1) Omega is well-conditioned, the non-finite-OFV diagnostic
        // must inspect the collapsed class Omega — else it misreports "numerical
        // overflow" instead of "Omega not positive definite".
        let model = crate::parser::model_parser::parse_model_string(MIX_OVERRIDE_MODEL).unwrap();
        let mut params = model.default_params.clone();
        // Base Omega stays healthy (0.06); collapse the class-2 override to negative.
        let mix = params.mixture.as_mut().expect("mixture params");
        mix.omega[1].matrix[(0, 0)] = -1.0;
        let chosen = diagnostic_omega(&params);
        assert_eq!(
            chosen[(0, 0)],
            -1.0,
            "must select the collapsed class Omega"
        );
        assert!(
            params.omega.matrix[(0, 0)] > 0.0,
            "base Omega is healthy, so the base branch would have hidden the collapse"
        );

        // Non-mixture params fall back to the single base Omega unchanged.
        let mut plain = model.default_params.clone();
        plain.mixture = None;
        assert_eq!(diagnostic_omega(&plain)[(0, 0)], plain.omega.matrix[(0, 0)]);
    }

    // ── #1781: R at an estimate the covariance objective never optimised ──────────────────

    /// `examples/tte_exponential.ferx` with its initial estimates moved to `(λ, ω²)`, parsed,
    /// with the 30-subject example dataset read through the model's endpoint routing.
    #[cfg(feature = "survival")]
    fn tte_exponential_at(
        lambda: f64,
        omega: f64,
    ) -> (crate::types::CompiledModel, crate::types::Population) {
        let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
        let src = src
            .replace(
                "theta TVLAMBDA(0.05, 0.001, 10.0)",
                &format!("theta TVLAMBDA({lambda:?}, 0.001, 10.0)"),
            )
            .replace(
                "omega ETA_LAMBDA ~ 0.09",
                &format!("omega ETA_LAMBDA ~ {omega:?}"),
            );
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        let (pop, _) = crate::api::read_population_for(
            &model,
            &None,
            "data/tte_exponential.csv",
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        (model, pop)
    }

    /// The covariance-step inputs at the model's initial estimates: the packed point, the
    /// parameters, and the EBEs converged there (the score cross-product reads them as given).
    #[cfg(feature = "survival")]
    #[allow(clippy::type_complexity)]
    fn inputs_at_init(
        model: &crate::types::CompiledModel,
        pop: &crate::types::Population,
        opts: &crate::types::FitOptions,
    ) -> (
        Vec<f64>,
        crate::types::ModelParameters,
        Vec<nalgebra::DVector<f64>>,
        Vec<DMatrix<f64>>,
    ) {
        let p = model.default_params.clone();
        let x = crate::estimation::parameterization::pack_params(&p);
        let (etas, hs): (Vec<_>, Vec<_>) = pop
            .subjects
            .iter()
            .map(|s| {
                let e = crate::estimation::inner_optimizer::find_ebe_for_stage(
                    model, s, &p, 200, 1e-8, None, None, 0, opts,
                );
                (e.eta, e.h_matrix)
            })
            .unzip();
        (x, p, etas, hs)
    }

    #[cfg(feature = "survival")]
    fn opts_for(
        method: crate::types::EstimationMethod,
        covariance_method: CovarianceMethod,
    ) -> crate::types::FitOptions {
        crate::types::FitOptions {
            method,
            covariance_method,
            ..crate::types::FitOptions::default()
        }
    }

    /// `compute_covariance` itself at the model's initial estimates — the step on the
    /// objective `options` selects, with no #1781 routing in front of it.
    #[cfg(feature = "survival")]
    fn cov_at_init(
        model: &crate::types::CompiledModel,
        pop: &crate::types::Population,
        opts: &crate::types::FitOptions,
    ) -> super::CovarianceStepResult {
        let (x, p, etas, hs) = inputs_at_init(model, pop, opts);
        let kappas = vec![Vec::new(); pop.subjects.len()];
        super::compute_covariance(&x, &p, model, pop, &etas, &hs, &kappas, opts)
    }

    /// The whole step as a fit's finalizer runs it — routing, warnings, published label.
    #[cfg(feature = "survival")]
    fn step_at_init(
        model: &crate::types::CompiledModel,
        pop: &crate::types::Population,
        opts: &crate::types::FitOptions,
    ) -> super::CovStepOutcome {
        let (x, p, etas, hs) = inputs_at_init(model, pop, opts);
        let kappas = vec![Vec::new(); pop.subjects.len()];
        super::run_covariance_step_inner(&x, &p, model, pop, &etas, &hs, &kappas, opts, None)
    }

    #[cfg(feature = "survival")]
    fn success(r: super::CovarianceStepResult, label: &str) -> super::CovarianceOutput {
        use super::CovarianceStepResult as R;
        match r {
            R::Success(o) => o,
            R::Unusable(m) => panic!("{label}: unusable: {m}"),
            R::FailedNonPd { reason, .. } => panic!("{label}: non-PD: {reason}"),
        }
    }

    /// Messages in `w` opening with `token`.
    #[cfg(feature = "survival")]
    fn count_token(w: &[String], token: &str) -> usize {
        w.iter().filter(|w| w.starts_with(token)).count()
    }

    /// SAEM's estimate on the 30-subject example, as the release CLI prints it (Linux x86_64,
    /// 2026-10-12): θ = 0.038756, ω² = 0.054346. The FOCEI OFV is still falling in ω² there
    /// (its minimum is 0.033183 / 0.291718), so the FOCEI `R` is indefinite.
    #[cfg(feature = "survival")]
    const SAEM_POINT: (f64, f64) = (0.038756, 0.054346);
    /// The FOCEI minimum on the same data, as the CLI prints it. `R` is positive definite.
    #[cfg(feature = "survival")]
    const FOCEI_POINT: (f64, f64) = (0.033183, 0.291718);

    /// #1781, the fallback: on an objective the estimate never minimised, an `R` the floor had
    /// to clip is replaced by `S⁻¹` — for every estimator on that side of
    /// `estimate_minimises_cov_objective` and for both `r` and `rsr` — and the result *is*
    /// `S⁻¹`, bit for bit. Driven through `compute_covariance` on the FOCEI objective, which
    /// is what the step still differentiates wherever the quadrature route declines (VI, a
    /// mixture, IOV, > 3 random effects).
    ///
    /// Before the fix this point returned `R⁻¹` with SE(log λ) = 108 and `R⁻¹SR⁻¹` with 3.3e5,
    /// where `S⁻¹` gives 0.220 and `R⁻¹` at the FOCEI minimum 0.265 (measured here).
    ///
    /// Three straddles, each inside this one test so a gate stuck on either branch fails it:
    /// the method (FOCEI at the same point keeps the floored `R` and its `REGULARIZED_PREFIX`
    /// warning, and says nothing about stationarity); the clip (SAEM at the FOCEI minimum,
    /// where nothing is clipped, keeps `R⁻¹`, bit-identical to FOCEI's there); and the request
    /// (`s` asked for directly carries no fallback warning). The published label is checked
    /// through `run_covariance_step_inner` under VI, the one estimator on this side of the
    /// gate that the quadrature route leaves on FOCEI (`fit()` refuses VI on a non-Gaussian
    /// model, so the step is driven directly).
    #[cfg(feature = "survival")]
    #[test]
    fn a_floored_r_at_a_non_stationary_estimate_is_replaced_by_s_inverse() {
        use crate::estimation::cov_diagnostics::{NON_STATIONARY_TOKEN, REGULARIZED_PREFIX};
        use crate::types::EstimationMethod as M;
        let token = |w: &[String]| count_token(w, NON_STATIONARY_TOKEN);
        let floor = |w: &[String]| count_token(w, REGULARIZED_PREFIX) > 0;

        let (model, pop) = tte_exponential_at(SAEM_POINT.0, SAEM_POINT.1);
        let s_opts = opts_for(M::Saem, CovarianceMethod::CrossProduct);
        let s_ref = success(cov_at_init(&model, &pop, &s_opts), "saem s");
        assert_eq!(
            token(&s_ref.warnings),
            0,
            "s requested: no fallback to report"
        );
        assert_eq!(s_ref.method, CovarianceMethod::CrossProduct);
        let se_s = s_ref.matrix[(0, 0)].sqrt();
        assert!(
            se_s.is_finite() && se_s > 0.1 && se_s < 0.5,
            "SE(log λ) from S⁻¹ = {se_s}"
        );

        for method in [M::Saem, M::Imp, M::Impmap, M::Vi] {
            for requested in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
                let label = format!("{method:?} {requested:?}");
                let out = success(
                    cov_at_init(&model, &pop, &opts_for(method, requested)),
                    &label,
                );
                assert_eq!(out.method, CovarianceMethod::CrossProduct, "{label}");
                assert_eq!(
                    out.matrix, s_ref.matrix,
                    "{label}: the matrix is S⁻¹ itself"
                );
                assert_eq!(token(&out.warnings), 1, "{label}: {:?}", out.warnings);
                assert!(
                    !floor(&out.warnings),
                    "{label}: no floor warning for an unused R"
                );
            }
        }
        let vi = step_at_init(&model, &pop, &opts_for(M::Vi, CovarianceMethod::Hessian));
        assert_eq!(
            vi.method,
            Some(CovarianceMethod::CrossProduct),
            "the published label"
        );
        assert_eq!(vi.matrix.as_ref(), Some(&s_ref.matrix));

        // Straddle 1 — the method. FOCEI minimises this objective: same point, same R, and
        // the floored inverse is kept (and is the absurd number the issue reported).
        let focei = success(
            cov_at_init(&model, &pop, &opts_for(M::FoceI, CovarianceMethod::Hessian)),
            "focei r",
        );
        assert_eq!(focei.method, CovarianceMethod::Hessian);
        assert!(
            floor(&focei.warnings),
            "precondition: R is clipped here: {:?}",
            focei.warnings
        );
        assert_eq!(token(&focei.warnings), 0);
        assert!(
            focei.matrix[(0, 0)].sqrt() > 10.0,
            "{}",
            focei.matrix[(0, 0)].sqrt()
        );

        // Straddle 2 — the clip. At the FOCEI minimum nothing is clipped, so SAEM keeps R⁻¹.
        let (model, pop) = tte_exponential_at(FOCEI_POINT.0, FOCEI_POINT.1);
        let h = CovarianceMethod::Hessian;
        let saem = success(cov_at_init(&model, &pop, &opts_for(M::Saem, h)), "saem");
        let focei = success(cov_at_init(&model, &pop, &opts_for(M::FoceI, h)), "focei");
        assert_eq!(saem.method, CovarianceMethod::Hessian);
        assert_eq!((token(&saem.warnings), floor(&saem.warnings)), (0, false));
        assert_eq!(
            saem.matrix, focei.matrix,
            "the method matters only through the gate"
        );
    }

    /// #1781: when the fallback's `S` is singular as well — one subject, two free parameters —
    /// the step fails as `FailedNonPd` (the SIR fallback keeps its proposal), with the
    /// non-stationary reason rather than the cross-product message's advice to use `r` or
    /// `rsr`, which is what was just declined. The same subject under a FOCEI fit asking for
    /// `s` still gets that ordinary message; and a subject whose `R` is positive definite at
    /// the same point never reaches `S` at all.
    #[cfg(feature = "survival")]
    #[test]
    fn a_non_stationary_fallback_with_a_singular_s_fails_non_pd() {
        use super::CovarianceStepResult as R;
        use crate::estimation::cov_diagnostics::NON_STATIONARY_TOKEN;
        use crate::types::EstimationMethod as M;
        let (model, full) = tte_exponential_at(SAEM_POINT.0, SAEM_POINT.1);
        let one = |k: usize| crate::types::Population {
            subjects: vec![full.subjects[k].clone()],
            ..full.clone()
        };
        let saem_r = opts_for(M::Saem, CovarianceMethod::Hessian);
        // Subject 1 (index 0): an event at 7.23 h. R is indefinite at this point.
        match cov_at_init(&model, &one(0), &saem_r) {
            R::FailedNonPd {
                reason,
                fallback_proposal,
            } => {
                assert!(
                    reason.starts_with(&format!(
                        "{NON_STATIONARY_TOKEN}: Covariance step failed: the FOCEI R matrix"
                    )),
                    "{reason}"
                );
                // One subject, more free parameters: the rank sentence replaces the hedge
                // (#2536 review r1 §6).
                assert!(
                    reason.contains("unavailable (S is singular).")
                        && !reason.contains("typically fewer subjects"),
                    "{reason}"
                );
                assert!(
                    !reason.contains("Use covariance_method = r or rsr"),
                    "{reason}"
                );
                // #2536: the counts reach the message from the population and the free block,
                // so a one-subject fixture states its design, not a hedge.
                assert!(
                    reason.contains(" With 1 subject and ")
                        && reason
                            .contains("S has rank at most 1, so it is singular on this design"),
                    "{reason}"
                );
                // SIR is not armed here, so the message offers it (#2536 review r1 §2).
                assert!(
                    reason.contains("`sir = true` samples")
                        && !reason.contains("The SIR intervals"),
                    "{reason}"
                );
                let n = crate::estimation::parameterization::packed_len(&model.default_params);
                assert_eq!(fallback_proposal.shape(), (n, n));
                assert!(fallback_proposal.iter().all(|v| v.is_finite()));
            }
            R::Success(o) => panic!("expected FailedNonPd, got {:?}", o.method),
            R::Unusable(m) => panic!("expected FailedNonPd, got Unusable: {m}"),
        }
        // The same point with SIR armed: the step reads `sir`, so the message describes the
        // intervals that follow instead of telling the user to switch SIR on.
        let saem_r_sir = crate::types::FitOptions {
            sir: true,
            ..saem_r.clone()
        };
        match cov_at_init(&model, &one(0), &saem_r_sir) {
            R::FailedNonPd { reason, .. } => assert!(
                reason.contains("The SIR intervals reported with this fit")
                    && !reason.contains("sir = true"),
                "{reason}"
            ),
            R::Success(o) => panic!("expected FailedNonPd with SIR armed, got {:?}", o.method),
            R::Unusable(m) => panic!("expected FailedNonPd with SIR armed, got Unusable: {m}"),
        }
        let vi = step_at_init(&model, &one(0), &opts_for(M::Vi, CovarianceMethod::Hessian));
        assert_eq!(
            (vi.matrix.is_none(), vi.method),
            (true, None),
            "no matrix, no label"
        );
        assert!(
            vi.sir_fallback_proposal.is_some(),
            "FailedNonPd keeps the SIR proposal"
        );
        match cov_at_init(
            &model,
            &one(0),
            &opts_for(M::FoceI, CovarianceMethod::CrossProduct),
        ) {
            R::Unusable(m) => assert!(
                m.contains(
                    "Use covariance_method = r or rsr, which never invert S; both need R \
                     positive definite"
                ),
                "{m}"
            ),
            _ => panic!("FOCEI + s on one subject: the ordinary singular-S message"),
        }
        // Subject 4 (index 3): R is positive definite at the same point — no fallback.
        match cov_at_init(&model, &one(3), &saem_r) {
            R::Success(o) => assert_eq!(o.method, CovarianceMethod::Hessian),
            _ => panic!("subject 4: R is positive definite, so R⁻¹ is returned"),
        }
    }

    /// #1781, the fix: a SAEM / IMP / IMPMAP estimate on a non-Gaussian model is scored on the
    /// `COV_AGQ_NODES`-node quadrature marginal. Through `run_covariance_step_inner` — the
    /// route lives there — the matrix is **bit-identical** to `compute_covariance` asked for
    /// `method = laplace, n_agq = 7` directly, it is `R⁻¹` (positive definite, nothing
    /// floored, no fallback), and the route note is emitted exactly once.
    ///
    /// The straddle is the method, at the same point: FOCEI keeps its own objective, floored,
    /// with no route note. The value is banded against the exact marginal's `R⁻¹` at its own
    /// minimum, computed outside ferx (composite Simpson over η, 4001 points; Newton on
    /// log λ, log ω²): SE(log λ) = 0.008182 / 0.033776 = 0.2422. At SAEM's point the measured
    /// value is 0.22171 (Linux x86_64) — 8.5% under the reference, because the point is not
    /// the reference's minimum (θ is 15% higher). Before the fix the same call returned 108.
    #[cfg(feature = "survival")]
    #[test]
    fn a_sampler_estimate_on_a_non_gaussian_model_is_scored_on_the_quadrature_marginal() {
        use crate::estimation::cov_diagnostics::{
            COV_AGQ_NODES, NON_STATIONARY_TOKEN, QUADRATURE_OBJECTIVE_TOKEN, REGULARIZED_PREFIX,
        };
        use crate::types::EstimationMethod as M;
        let (model, pop) = tte_exponential_at(SAEM_POINT.0, SAEM_POINT.1);
        let h = CovarianceMethod::Hessian;
        let direct = success(
            cov_at_init(
                &model,
                &pop,
                &crate::types::FitOptions {
                    n_agq: COV_AGQ_NODES,
                    ..opts_for(M::Laplace, h)
                },
            ),
            "laplace n_agq = 7",
        );
        assert_eq!(direct.method, h);
        let se = direct.matrix[(0, 0)].sqrt();
        eprintln!("SE(log λ) on the 7-node marginal at SAEM's point: {se:.6}");
        assert!(
            se.is_finite() && (0.2422 - se).abs() < 0.03,
            "SE(log λ) = {se}"
        );

        for method in [M::Saem, M::Imp, M::Impmap] {
            let out = step_at_init(&model, &pop, &opts_for(method, h));
            let label = format!("{method:?}");
            assert_eq!(
                out.matrix.as_ref(),
                Some(&direct.matrix),
                "{label}: the 7-node R⁻¹"
            );
            assert_eq!(out.method, Some(h), "{label}");
            assert_eq!(
                count_token(&out.warnings, QUADRATURE_OBJECTIVE_TOKEN),
                1,
                "{label}"
            );
            assert_eq!(
                count_token(&out.warnings, NON_STATIONARY_TOKEN),
                0,
                "{label}"
            );
            assert_eq!(count_token(&out.warnings, REGULARIZED_PREFIX), 0, "{label}");
        }
        let focei = step_at_init(&model, &pop, &opts_for(M::FoceI, h));
        assert_eq!(count_token(&focei.warnings, QUADRATURE_OBJECTIVE_TOKEN), 0);
        assert_eq!(count_token(&focei.warnings, REGULARIZED_PREFIX), 1);
        let focei_se = focei.matrix.expect("floored R⁻¹")[(0, 0)].sqrt();
        assert!(focei_se > 10.0, "FOCEI keeps its own objective: {focei_se}");
    }

    /// #1781: a sampler fit the quadrature route cannot serve — here four random effects, a
    /// `7⁴ = 2401`-node grid — keeps the FOCEI objective and carries
    /// `W_COV_QUADRATURE_DECLINED` exactly once, through the same `run_covariance_step_inner`
    /// the finalizers call; FOCEI on the same model carries neither route message. Three
    /// subjects keep it a Tier-1 cost: only the warnings are asserted, not the matrix.
    #[cfg(feature = "survival")]
    #[test]
    fn a_sampler_fit_past_the_quadrature_grid_limit_is_warned_and_stays_on_focei() {
        use crate::estimation::cov_diagnostics::{
            QUADRATURE_DECLINED_TOKEN, QUADRATURE_OBJECTIVE_TOKEN,
        };
        use crate::types::EstimationMethod as M;
        let src = std::fs::read_to_string("examples/tte_exponential.ferx")
            .expect("model file")
            .replace(
                "omega ETA_LAMBDA ~ 0.09",
                "omega ETA_LAMBDA ~ 0.09\n  omega ETA_B ~ 0.05\n  omega ETA_C ~ 0.05\n  \
                 omega ETA_D ~ 0.05",
            )
            .replace(
                "scale  = TVLAMBDA * exp(ETA_LAMBDA)",
                "scale  = TVLAMBDA * exp(ETA_LAMBDA + ETA_B + ETA_C + ETA_D)",
            );
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        assert_eq!(model.n_eta, 4);
        let (full, _) = crate::api::read_population_for(
            &model,
            &None,
            "data/tte_exponential.csv",
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        let pop = crate::types::Population {
            subjects: full.subjects[..3].to_vec(),
            ..full.clone()
        };
        let h = CovarianceMethod::Hessian;
        let saem = step_at_init(&model, &pop, &opts_for(M::Saem, h));
        assert_eq!(
            count_token(&saem.warnings, QUADRATURE_DECLINED_TOKEN),
            1,
            "{:?}",
            saem.warnings
        );
        assert_eq!(count_token(&saem.warnings, QUADRATURE_OBJECTIVE_TOKEN), 0);
        let declined = saem
            .warnings
            .iter()
            .find(|w| w.starts_with(QUADRATURE_DECLINED_TOKEN))
            .expect("declined");
        assert!(
            declined.contains("4 random effects need a 7^4 = 2401-node grid"),
            "{declined}"
        );
        let focei = step_at_init(&model, &pop, &opts_for(M::FoceI, h));
        assert!(
            !focei
                .warnings
                .iter()
                .any(|w| w.contains("W_COV_QUADRATURE")),
            "{:?}",
            focei.warnings
        );
    }

    /// `(SE(λ), SE(ω²))` as the fit reports them, from a packed covariance.
    #[cfg(feature = "survival")]
    fn reported_lambda_omega(
        cov: &DMatrix<f64>,
        params: &crate::types::ModelParameters,
    ) -> (f64, f64) {
        let (th, om, _, _) =
            crate::api::postfit::extract_standard_errors(&Some(cov.clone()), params);
        (th.expect("se_theta")[0], om.expect("se_omega")[0])
    }

    /// #1781, the same-point external anchor (Codex review r1 §2). At each sampler's own
    /// estimate, the covariance the routed step returns under `r`, `s` and `rsr` against the
    /// **exact** marginal's at the *same* point, computed outside ferx by
    /// `nonmem_anchor/tte_exponential_exact.py --at λ ω²` (composite Simpson over η, 4001
    /// points; central-difference Hessian and per-subject scores). Comparing at one point
    /// separates covariance error from estimator location, which a fit-vs-MLE band conflates.
    ///
    /// Measured (Linux x86_64), relative error against the oracle, worst over the 18 numbers:
    /// 2.02e-3 (IMP `rsr`, SE(ω²)); `s` agrees to 6e-6, since `S` has no Hessian in it. The
    /// `R` error is the FD stencil's truncation at the default `fd_hessian_step = 1e-2`, not the
    /// quadrature: going from 7 to 15 nodes moves no SE by more than 3e-5, while the step moves
    /// it as h² — worst `r` / `rsr` error 3.6e-3 / 8.2e-3 at h = 2e-2 and 2.3e-4 / 5.3e-4 at
    /// h = 5e-3, at either node count. Three nodes, by contrast, is 11% low on SE(ω²), and the
    /// FOCEI objective this replaced is off by three orders of magnitude.
    #[cfg(feature = "survival")]
    #[test]
    fn the_sampler_points_match_the_exact_marginal_covariance() {
        use crate::types::EstimationMethod as M;
        // (method, point, [R, S, RSR] as (SE(λ), SE(ω²))) — the oracle's output, verbatim.
        let cases: [(M, (f64, f64), [(f64, f64); 3]); 3] = [
            (
                M::Saem,
                SAEM_POINT,
                [
                    (0.0085912, 0.6132721),
                    (0.0085592, 0.7800549),
                    (0.0086290, 0.4840343),
                ],
            ),
            (
                M::Imp,
                (0.033848, 0.081064),
                [
                    (0.0081875, 0.6417223),
                    (0.0081655, 0.8941711),
                    (0.0082157, 0.4629967),
                ],
            ),
            (
                M::Impmap,
                (0.033591, 0.082876),
                [
                    (0.0081921, 0.6555724),
                    (0.0081522, 0.9008471),
                    (0.0082376, 0.4801825),
                ],
            ),
        ];
        let mut worst: f64 = 0.0;
        for (method, (lambda, omega), oracle) in cases {
            let (model, pop) = tte_exponential_at(lambda, omega);
            for (cm, want) in [
                CovarianceMethod::Hessian,
                CovarianceMethod::CrossProduct,
                CovarianceMethod::Sandwich,
            ]
            .into_iter()
            .zip(oracle)
            {
                let out = step_at_init(&model, &pop, &opts_for(method, cm));
                assert_eq!(
                    out.method,
                    Some(cm),
                    "{method:?} {cm:?}: {:?}",
                    out.warnings
                );
                let got = reported_lambda_omega(
                    out.matrix.as_ref().expect("matrix"),
                    &model.default_params,
                );
                assert!(
                    got.0.is_finite() && got.1.is_finite(),
                    "{method:?} {cm:?}: {got:?}"
                );
                let err = ((got.0 / want.0 - 1.0).abs()).max((got.1 / want.1 - 1.0).abs());
                eprintln!(
                    "{method:?} {:>3}: SE(λ) {:.7} vs {:.7}, SE(ω²) {:.7} vs {:.7}, rel {err:.2e}",
                    cm.label(),
                    got.0,
                    want.0,
                    got.1,
                    want.1
                );
                worst = worst.max(err);
            }
        }
        eprintln!("worst relative error: {worst:.2e}");
        assert!(worst < SAME_POINT_TOL, "worst {worst:.3e}");
    }

    /// 2.5× the realised worst (2.02e-3) of `the_sampler_points_match_the_exact_marginal_covariance`.
    #[cfg(feature = "survival")]
    const SAME_POINT_TOL: f64 = 5e-3;

    /// `examples/tte_exponential.ferx` with a `FIX`ed κ on the (unused) dummy clearance, and the
    /// 30-subject dataset read with one occasion each — a reachable IOV model (`n_kappa = 1`)
    /// whose likelihood is the 1-η TTE one, so its FOCEI `R` at `SAEM_POINT` is the same
    /// indefinite matrix the plain model has there.
    #[cfg(feature = "survival")]
    fn tte_iov_at_saem_point() -> (crate::types::CompiledModel, crate::types::Population) {
        use std::io::Write;
        let src = std::fs::read_to_string("examples/tte_exponential.ferx")
            .expect("model file")
            .replace(
                "theta TVLAMBDA(0.05, 0.001, 10.0)",
                &format!("theta TVLAMBDA({:?}, 0.001, 10.0)", SAEM_POINT.0),
            )
            .replace(
                "omega ETA_LAMBDA ~ 0.09",
                &format!(
                    "omega ETA_LAMBDA ~ {:?}\n  kappa KAPPA_CL ~ 0.04 FIX",
                    SAEM_POINT.1
                ),
            )
            .replace("CL     = DUMMY_CL", "CL     = DUMMY_CL * exp(KAPPA_CL)");
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        assert_eq!(model.n_kappa, 1, "fixture must carry a κ");
        let csv = std::fs::read_to_string("data/tte_exponential.csv").expect("data");
        let mut lines = csv.lines();
        let mut out = format!("{},OCC\n", lines.next().expect("header"));
        for l in lines.filter(|l| !l.trim().is_empty()) {
            out.push_str(&format!("{l},1\n"));
        }
        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        f.write_all(out.as_bytes()).expect("write");
        let (pop, _) = crate::api::read_population_for(
            &model,
            &None,
            f.path().to_str().expect("path"),
            None,
            Some("OCC"),
            None,
            &[],
        )
        .expect("data");
        (model, pop)
    }

    /// #1781 (Codex review r1 §1, §4): a live IOV model declines the quadrature route through
    /// `run_covariance_step_inner` — the production call site supplies `model.n_kappa > 0` — and
    /// the step then runs on the objective it would have used without the route, bit for bit.
    /// The declined warning must hold for every outcome of that step, so all three are driven
    /// through the same declined route: `s` directly (S⁻¹), `r` with a clipped `R` (the
    /// `S⁻¹` fallback), and `r` on one subject, where `S` is singular too (the step fails).
    #[cfg(feature = "survival")]
    #[test]
    fn a_live_iov_model_declines_the_quadrature_route_for_every_step_outcome() {
        use crate::estimation::cov_diagnostics::{
            NON_STATIONARY_TOKEN, QUADRATURE_DECLINED_TOKEN, QUADRATURE_OBJECTIVE_TOKEN,
        };
        use crate::types::EstimationMethod as M;
        let (model, pop) = tte_iov_at_saem_point();
        let declined = |w: &[String]| {
            assert_eq!(count_token(w, QUADRATURE_OBJECTIVE_TOKEN), 0, "{w:?}");
            assert_eq!(count_token(w, QUADRATURE_DECLINED_TOKEN), 1, "{w:?}");
            let msg = w
                .iter()
                .find(|m| m.starts_with(QUADRATURE_DECLINED_TOKEN))
                .unwrap();
            assert!(msg.contains("inter-occasion variability"), "{msg}");
        };
        // Direct S: the same matrix the unrouted step produces.
        let s_opts = opts_for(M::Saem, CovarianceMethod::CrossProduct);
        let s = step_at_init(&model, &pop, &s_opts);
        declined(&s.warnings);
        assert_eq!(s.method, Some(CovarianceMethod::CrossProduct));
        let direct = success(cov_at_init(&model, &pop, &s_opts), "unrouted s");
        assert_eq!(
            s.matrix.as_ref(),
            Some(&direct.matrix),
            "the unchanged objective"
        );
        // R clipped → S⁻¹ fallback, still on the unrouted objective.
        let r = step_at_init(&model, &pop, &opts_for(M::Saem, CovarianceMethod::Hessian));
        declined(&r.warnings);
        assert_eq!(
            count_token(&r.warnings, NON_STATIONARY_TOKEN),
            1,
            "{:?}",
            r.warnings
        );
        assert_eq!(r.method, Some(CovarianceMethod::CrossProduct));
        assert_eq!(r.matrix.as_ref(), Some(&direct.matrix));
        // One subject: S singular as well → the step fails, and still says it was declined.
        let one = crate::types::Population {
            subjects: vec![pop.subjects[0].clone()],
            ..pop.clone()
        };
        let failed = step_at_init(&model, &one, &opts_for(M::Saem, CovarianceMethod::Hessian));
        declined(&failed.warnings);
        assert!(failed.matrix.is_none() && failed.method.is_none());
        assert!(
            failed
                .warnings
                .iter()
                .any(|w| w.starts_with(&format!("{NON_STATIONARY_TOKEN}: Covariance step failed"))),
            "{:?}",
            failed.warnings
        );
    }

    /// #1781 (Codex review r1 §1): a live mixture model declines the quadrature route through
    /// `run_covariance_step_inner` — the call site supplies `template.mixture.is_some()` — and
    /// the step runs on the K-fold mixture objective it would have used without the route,
    /// bit for bit. Two identical classes with a `FIX`ed mixing probability, so the mixture
    /// changes nothing but the gate.
    #[cfg(feature = "survival")]
    #[test]
    fn a_live_mixture_model_declines_the_quadrature_route() {
        use crate::estimation::cov_diagnostics::{
            QUADRATURE_DECLINED_TOKEN, QUADRATURE_OBJECTIVE_TOKEN,
        };
        use crate::types::EstimationMethod as M;
        let src = std::fs::read_to_string("examples/tte_exponential.ferx")
            .expect("model file")
            .replace(
                "theta TVLAMBDA(0.05, 0.001, 10.0)",
                &format!(
                    "theta TVLAMBDA({:?}, 0.001, 10.0)\n  theta P1(0.5, FIX)",
                    FOCEI_POINT.0
                ),
            )
            .replace(
                "omega ETA_LAMBDA ~ 0.09",
                &format!("omega ETA_LAMBDA ~ {:?}", FOCEI_POINT.1),
            )
            .replacen(
                "\n[individual_parameters]\n",
                "\n[mixture]\n  nsub = 2\n  p(1) = P1\n\n[individual_parameters]\n",
                1,
            )
            .replace(
                "CL     = DUMMY_CL",
                "CL     = if (MIXNUM == 1) DUMMY_CL else DUMMY_CL",
            );
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        assert!(
            model.default_params.mixture.is_some(),
            "fixture must be a mixture"
        );
        let (pop, _) = crate::api::read_population_for(
            &model,
            &None,
            "data/tte_exponential.csv",
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        let opts = opts_for(M::Saem, CovarianceMethod::Hessian);
        let out = step_at_init(&model, &pop, &opts);
        assert_eq!(
            count_token(&out.warnings, QUADRATURE_OBJECTIVE_TOKEN),
            0,
            "{:?}",
            out.warnings
        );
        assert_eq!(
            count_token(&out.warnings, QUADRATURE_DECLINED_TOKEN),
            1,
            "{:?}",
            out.warnings
        );
        let msg = out
            .warnings
            .iter()
            .find(|m| m.starts_with(QUADRATURE_DECLINED_TOKEN))
            .unwrap();
        assert!(msg.contains("the model is a mixture"), "{msg}");
        let unrouted = cov_at_init(&model, &pop, &opts);
        match unrouted {
            super::CovarianceStepResult::Success(o) => {
                assert_eq!(
                    out.matrix.as_ref(),
                    Some(&o.matrix),
                    "the mixture objective"
                )
            }
            other => {
                assert!(out.matrix.is_none(), "both routes must agree");
                let _ = other;
            }
        }
    }

    // ── #1803: `covariance_method = s` builds no `R` ───────────────────────────────────────

    /// Only `S⁻¹` is on the no-`R` side, and each estimator says which side it is on.
    #[test]
    fn only_the_cross_product_estimator_reads_no_r() {
        assert!(super::needs_r(CovarianceMethod::Hessian));
        assert!(super::needs_r(CovarianceMethod::Sandwich));
        assert!(!super::needs_r(CovarianceMethod::CrossProduct));
    }

    /// #1803: the soft off-diagonal warning names its parameters in packed order under the two
    /// estimators that read `R`, and is withheld under `S⁻¹` (the #1781 fallback), where the
    /// stencil it describes is `R`'s and `R` was discarded. An empty set says nothing on any
    /// estimator. Both sides of the gate in one test.
    #[test]
    fn the_offdiag_nan_warning_is_gated_on_an_estimator_that_reads_r() {
        use crate::estimation::cov_diagnostics::format_offdiag_nan_warning;
        let model = crate::parser::model_parser::parse_model_string(MIX_OVERRIDE_MODEL).unwrap();
        let t = &model.default_params;
        let set: super::HashSet<usize> = [4, 0].into_iter().collect();
        let fd = super::CovHessianSource::FdStencil;
        for estimator in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
            let msg = super::offdiag_nan_warning(&set, estimator, fd, t)
                .unwrap_or_else(|| panic!("{estimator:?}: the warning is about R"));
            assert_eq!(
                msg,
                format_offdiag_nan_warning("theta[TVCL], omega[ETA_CL, ETA_CL]", fd),
                "{estimator:?}"
            );
        }
        assert_eq!(
            super::offdiag_nan_warning(&set, CovarianceMethod::CrossProduct, fd, t),
            None,
            "S⁻¹ is not built from the stencil"
        );
        for estimator in [
            CovarianceMethod::Hessian,
            CovarianceMethod::Sandwich,
            CovarianceMethod::CrossProduct,
        ] {
            assert_eq!(
                super::offdiag_nan_warning(&super::HashSet::new(), estimator, fd, t),
                None
            );
        }
    }

    /// Runs `f` with `FD_OBJECTIVE_EVALS` reset, returning its result and the objectives the
    /// covariance step's FD probe and stencil evaluated meanwhile.
    #[cfg(feature = "survival")]
    fn counting_fd_evals<T>(f: impl FnOnce() -> T) -> (T, usize) {
        super::FD_OBJECTIVE_EVALS.with(|c| c.set(0));
        let out = f();
        (out, super::FD_OBJECTIVE_EVALS.with(|c| c.get()))
    }

    /// `opts_for` on the FD route: the analytic `R` assembly off, so `r` pays the stencil.
    #[cfg(feature = "survival")]
    fn fd_opts_for(
        method: crate::types::EstimationMethod,
        covariance_method: CovarianceMethod,
    ) -> crate::types::FitOptions {
        crate::types::FitOptions {
            analytic_cov_hessian: false,
            ..opts_for(method, covariance_method)
        }
    }

    /// The production count every cost warning quotes, so the counter tests below measure it
    /// against the objectives the FD route actually evaluates.
    #[cfg(feature = "survival")]
    fn fd_stencil_evals(n_free: usize) -> usize {
        super::fd_r_objective_evals(n_free).expect("small n")
    }

    /// `2·n²`: the `2·n` step probe, which is the stencil's diagonal (#1935), and four points
    /// per pair — pinned on values, and `None` rather than a wrapped count once it leaves
    /// `usize`. The counter tests below measure it on the route.
    #[test]
    fn the_fd_r_cost_is_two_n_squared() {
        assert_eq!(super::fd_r_objective_evals(0), Some(0));
        assert_eq!(super::fd_r_objective_evals(1), Some(2));
        assert_eq!(super::fd_r_objective_evals(2), Some(8));
        assert_eq!(super::fd_r_objective_evals(101), Some(20_402));
        assert_eq!(super::fd_r_objective_evals(800), Some(1_280_000));
        assert_eq!(super::fd_r_objective_evals(usize::MAX), None);
        assert_eq!(super::fd_r_objective_evals(1 << (usize::BITS / 2)), None);
    }

    /// #1803, the cost: `s` evaluates **no** FD objective, where `r` and `rsr` each pay the
    /// whole `2·n²` at the FOCEI minimum (two free parameters, so 8). The `r` count is
    /// what keeps the zero meaningful: a counter that is never incremented would read 0 on
    /// every estimator.
    ///
    /// `s` is the same matrix as `S⁻¹` reached the other way — through a built `R`, as the
    /// #1781 fallback — bit for bit, so skipping `R` changed no number, only the cost.
    #[cfg(feature = "survival")]
    #[test]
    fn s_evaluates_no_fd_objective_where_r_and_rsr_pay_the_whole_stencil() {
        use crate::types::EstimationMethod as M;
        let (model, pop) = tte_exponential_at(FOCEI_POINT.0, FOCEI_POINT.1);
        let (s, s_evals) = counting_fd_evals(|| {
            cov_at_init(
                &model,
                &pop,
                &fd_opts_for(M::FoceI, CovarianceMethod::CrossProduct),
            )
        });
        assert_eq!(success(s, "focei s").method, CovarianceMethod::CrossProduct);
        assert_eq!(s_evals, 0, "s builds no R");
        for cm in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
            let (out, evals) =
                counting_fd_evals(|| cov_at_init(&model, &pop, &fd_opts_for(M::FoceI, cm)));
            assert_eq!(success(out, "focei").method, cm);
            assert_eq!(evals, fd_stencil_evals(2), "{cm:?}");
        }

        // The other way to S⁻¹: SAEM's fallback at SAEM's point builds R, clips it, and
        // returns S⁻¹ — the same matrix the early return gives there.
        let (model, pop) = tte_exponential_at(SAEM_POINT.0, SAEM_POINT.1);
        let (fallback, fallback_evals) = counting_fd_evals(|| {
            cov_at_init(
                &model,
                &pop,
                &fd_opts_for(M::Saem, CovarianceMethod::Hessian),
            )
        });
        let fallback = success(fallback, "saem r → S⁻¹");
        assert_eq!(fallback.method, CovarianceMethod::CrossProduct);
        assert_eq!(
            fallback_evals,
            fd_stencil_evals(2),
            "the fallback did build R"
        );
        let direct = success(
            cov_at_init(
                &model,
                &pop,
                &fd_opts_for(M::Saem, CovarianceMethod::CrossProduct),
            ),
            "saem s",
        );
        assert_eq!(direct.matrix, fallback.matrix);
    }

    /// `examples/tte_exponential.ferx` at `SAEM_POINT` with λ `FIX`ed, so ω² is the one free
    /// parameter — and the FOCEI `R` along it is negative there (the objective is still falling
    /// in ω², see `SAEM_POINT`), so `R` has no positive eigenvalue at all.
    #[cfg(feature = "survival")]
    fn tte_omega_only_at_saem_point() -> (crate::types::CompiledModel, crate::types::Population) {
        let src = std::fs::read_to_string("examples/tte_exponential.ferx")
            .expect("model file")
            .replace(
                "theta TVLAMBDA(0.05, 0.001, 10.0)",
                &format!("theta TVLAMBDA({:?}, 0.001, 10.0) FIX", SAEM_POINT.0),
            )
            .replace(
                "omega ETA_LAMBDA ~ 0.09",
                &format!("omega ETA_LAMBDA ~ {:?}", SAEM_POINT.1),
            );
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        let (pop, _) = crate::api::read_population_for(
            &model,
            &None,
            "data/tte_exponential.csv",
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        (model, pop)
    }

    /// #1803, the veto: an `R` with no positive curvature no longer refuses an `S⁻¹` that
    /// exists. On the ω²-only fixture FOCEI `r` fails as `FailedNonPd` — the precondition, and
    /// before the fix `s` failed with it — while `s` now returns `S⁻¹`, bit-identical to
    /// inverting `assemble_score_cross_product` directly, and evaluates no FD objective. `rsr`
    /// reads `R`, so it still fails: the straddle is the estimator, at one point.
    #[cfg(feature = "survival")]
    #[test]
    fn an_r_with_no_positive_curvature_no_longer_vetoes_s() {
        use super::CovarianceStepResult as R;
        use crate::types::EstimationMethod as M;
        let (model, pop) = tte_omega_only_at_saem_point();
        for cm in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
            match cov_at_init(&model, &pop, &fd_opts_for(M::FoceI, cm)) {
                R::FailedNonPd { reason, .. } => eprintln!("{cm:?}: {reason}"),
                R::Success(o) => panic!("{cm:?}: precondition, R is non-PD: got {:?}", o.matrix),
                R::Unusable(m) => panic!("{cm:?}: precondition, R is non-PD: unusable {m}"),
            }
        }

        let opts = fd_opts_for(M::FoceI, CovarianceMethod::CrossProduct);
        let (out, evals) = counting_fd_evals(|| cov_at_init(&model, &pop, &opts));
        let out = success(out, "focei s");
        assert_eq!((out.method, evals), (CovarianceMethod::CrossProduct, 0));

        let (x, p, etas, hs) = inputs_at_init(&model, &pop, &opts);
        let kappas = vec![Vec::new(); pop.subjects.len()];
        let super::PackedStart { bounds, fixed, .. } = super::pack_with_bounds(&p);
        let free_idx: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
        assert_eq!(free_idx.len(), 1, "fixture: ω² alone is free");
        let s = super::assemble_score_cross_product(
            &x, &p, &model, &pop, &etas, &hs, &kappas, &bounds, &opts, &free_idx,
        )
        .expect("S");
        let want = super::embed_free(
            &super::invert_cross_product(&s, &vec![super::CoordUnits::Dimensionless; s.nrows()])
                .expect("S⁻¹ exists"),
            &free_idx,
            x.len(),
        );
        assert_eq!(out.matrix, want);
        let var = out.matrix[(free_idx[0], free_idx[0])];
        eprintln!("S⁻¹ variance of log ω² on the ω²-only fixture: {var:.6}");
        assert!(var.is_finite() && var > 0.0, "{var}");
    }

    /// #1803, `rsr` fails fast: where the estimate minimised the objective, a failed `S` stops
    /// `rsr` before `R`'s stencil starts — the same `Unusable` it returned after the stencil
    /// before. On the other side of `estimate_minimises_cov_objective` the order is kept,
    /// because there a failed `S` may be the #1781 fallback's `FailedNonPd`, whose SIR proposal
    /// is built from `R`. It is kept, too, when a SIR fallback is armed, by either option: a
    /// non-PD `R` then returns the proposal SIR runs from, which stopping on `S` would drop.
    ///
    /// Both sides in one test, at one point, on one objective — the 7-node quadrature marginal
    /// a #1781-routed SAEM estimate is scored on — differing only in the estimate's method, so
    /// the gate is the one thing that changes. `S` is made to fail by an exhausted inner budget
    /// (`inner_maxiter = 0` with per-evaluation reconvergence — the fixture of
    /// the forward provider's quadrature-score test, retired in #2093), which the quadrature
    /// score refuses to square, while `R`'s stencil differences the objective at the
    /// warm-started modes, so `r` still succeeds on the same inputs.
    #[cfg(feature = "survival")]
    #[test]
    fn rsr_stops_on_a_failed_s_before_the_stencil_where_the_estimate_is_stationary() {
        use super::CovarianceStepResult as R;
        use crate::estimation::cov_diagnostics::COV_AGQ_NODES;
        use crate::types::EstimationMethod as M;
        let (model, pop) = tte_exponential_at(SAEM_POINT.0, SAEM_POINT.1);
        let opts_on_quadrature = |cm: CovarianceMethod| crate::types::FitOptions {
            n_agq: COV_AGQ_NODES,
            inner_maxiter: 0,
            reconverge_gradient_interval: 1,
            ..fd_opts_for(M::Laplace, cm)
        };
        let (x, p, etas, hs) =
            inputs_at_init(&model, &pop, &opts_on_quadrature(CovarianceMethod::Hessian));
        let kappas = vec![Vec::new(); pop.subjects.len()];
        let step_with = |fit_method: M, opts: &crate::types::FitOptions| {
            counting_fd_evals(|| {
                super::compute_covariance_for(
                    fit_method, &x, &p, &model, &pop, &etas, &hs, &kappas, opts,
                )
            })
        };
        let step =
            |fit_method: M, cm: CovarianceMethod| step_with(fit_method, &opts_on_quadrature(cm));
        let score_failed = |r: R, label: &str| match r {
            R::Unusable(m) => assert!(m.contains("quadrature score"), "{label}: {m}"),
            R::Success(_) => panic!("{label}: S failed, so the step must"),
            R::FailedNonPd { reason, .. } => panic!("{label}: R is PD here: {reason}"),
        };

        let (r, r_evals) = step(M::Laplace, CovarianceMethod::Hessian);
        assert_eq!(
            success(r, "laplace r, no inner budget").method,
            CovarianceMethod::Hessian
        );
        assert_eq!(r_evals, fd_stencil_evals(2), "R does not need the score");

        let (rsr, rsr_evals) = step(M::Laplace, CovarianceMethod::Sandwich);
        score_failed(rsr, "laplace rsr");
        assert_eq!(rsr_evals, 0, "S failed first, so no stencil ran");

        let (rsr, rsr_evals) = step(M::Saem, CovarianceMethod::Sandwich);
        score_failed(rsr, "saem rsr");
        assert_eq!(
            rsr_evals,
            fd_stencil_evals(2),
            "a non-stationary estimate builds R before reading S"
        );

        // An armed SIR fallback keeps `R` first on the stationary side too, by either switch:
        // there a non-PD `R` carries the proposal SIR runs from, which `S` failing first would
        // drop. The unarmed Laplace `rsr` above, at 0 evaluations, is the other side of the gate.
        for (label, armed) in [
            (
                "covariance_fallback = sir",
                crate::types::FitOptions {
                    covariance_fallback: crate::types::CovarianceFallback::Sir,
                    ..opts_on_quadrature(CovarianceMethod::Sandwich)
                },
            ),
            (
                "sir = true",
                crate::types::FitOptions {
                    sir: true,
                    ..opts_on_quadrature(CovarianceMethod::Sandwich)
                },
            ),
        ] {
            let (rsr, rsr_evals) = step_with(M::Laplace, &armed);
            score_failed(rsr, &format!("laplace rsr, {label}"));
            assert_eq!(
                rsr_evals,
                fd_stencil_evals(2),
                "{label}: an armed SIR fallback builds R before reading S"
            );
        }
    }

    // ── #1935: the step probe hands its pass to the stencil ───────────────────────────────

    /// The FD Hessian route evaluates `2·n_free²` objectives when no halving is needed — the
    /// probe's `2·n_free` diagonal points are the stencil's, so they are evaluated once — plus
    /// `2·n_free` per failed halving pass. Before #1935 it was `2·n_free + 2·n_free²` with no
    /// halving: the stencil recomputed the probe's points.
    ///
    /// Each case also requires the matrix to be bit-identical to the stencil run on its own at
    /// the chosen step with nothing reused — the reuse must skip work, never change a number.
    /// Three cells: no halving; one failed pass (the reused values are the *second* pass's, at
    /// the halved step); every pass failing (nothing to reuse, so the stencil evaluates its own
    /// diagonal and reports it non-finite).
    ///
    /// Mutations, each killed here: dropping the reuse (`None` into `fd_ofv_stencil`) moves
    /// every count up by `2·n_free`; reusing a failed pass's values puts the first pass's `NaN`s
    /// into the one-halving diagonal, which the bit-identity then rejects; handing back the
    /// last failed pass when every halving fails — values at a step the stencil no longer uses
    /// — drops the all-fail count by `2·n_free`.
    #[test]
    fn the_fd_hessian_evaluates_the_probe_pass_once() {
        use super::{fd_ofv_hessian, fd_ofv_stencil, select_fd_step};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let opts = crate::types::FitOptions::default();
        // Coupled and non-quadratic, so every free entry of the Hessian is live; coordinate 2
        // is held fixed, so the free set is not a prefix and `free_idx` indexing is exercised.
        let x_hat = [0.3f64, -1.2, 2.0, 0.7];
        let free_idx = [0usize, 1, 3];
        let n = x_hat.len();
        let nf = free_idx.len();
        let ones = [1.0; 4];
        let smooth = |x: &[f64]| -> f64 {
            0.1 * (x[0] * x[1]).exp() + x[0] * x[0] + 3.0 * x[1] * x[3] + x[3].powi(4) + x[2]
        };
        let eps0 = 1e-2;
        let h0 = eps0 * (1.0 + x_hat[0].abs());

        // (label, objective, expected halvings, expected objective count)
        type Ofv<'a> = Box<dyn Fn(&[f64]) -> f64 + Sync + 'a>;
        let cases: [(&str, Ofv, usize, usize); 3] = [
            ("no halving", Box::new(smooth), 0, 2 * nf * nf),
            (
                // NaN once coordinate 0 moves by more than 0.75·h₀: the first pass fails on
                // that coordinate alone, the halved one passes, and so does every pair point.
                "one halving",
                Box::new(move |x: &[f64]| {
                    if (x[0] - x_hat[0]).abs() > 0.75 * h0 {
                        f64::NAN
                    } else {
                        smooth(x)
                    }
                }),
                1,
                2 * nf + 2 * nf * nf,
            ),
            (
                "every halving fails",
                Box::new(move |x: &[f64]| {
                    if x[0] != x_hat[0] {
                        f64::NAN
                    } else {
                        smooth(x)
                    }
                }),
                8,
                8 * 2 * nf + 2 * nf * nf,
            ),
        ];

        for (label, ofv, halvings, want_calls) in cases {
            let f0 = ofv(&x_hat);
            let calls = AtomicUsize::new(0);
            let counted = |x: &[f64]| {
                calls.fetch_add(1, Ordering::Relaxed);
                ofv(x)
            };
            let got = fd_ofv_hessian(n, &x_hat, &free_idx, &ones, eps0, f0, &counted, &opts)
                .expect("nothing cancels in a test");
            assert_eq!(
                calls.load(Ordering::Relaxed),
                want_calls,
                "{label}: objective evaluations on the FD Hessian route"
            );

            let step = select_fd_step(&x_hat, &free_idx, &ones, eps0, f0, &ofv, &opts);
            assert_eq!(step.n_halvings, halvings, "{label}: halvings");
            assert_eq!(
                step.diag.is_some(),
                halvings < 8,
                "{label}: a reusable pass exists exactly when one was accepted"
            );
            let alone =
                fd_ofv_stencil(n, &x_hat, &free_idx, &ones, step.eps, f0, None, &ofv, &opts)
                    .expect("nothing cancels in a test");
            assert_eq!(got.hess, alone.hess, "{label}: reuse changed the matrix");
            assert_eq!(got.diag_nan, alone.diag_nan, "{label}: diag_nan");
            assert_eq!(got.offdiag_nan, alone.offdiag_nan, "{label}: offdiag_nan");
            // The comparison must be against a live matrix, or bit-identity is vacuous.
            assert!(
                got.hess[(1, 1)] != 0.0 && got.hess[(1, 3)] != 0.0,
                "{label}: the fixture must produce live curvature: {}",
                got.hess
            );
            if halvings == 8 {
                assert!(
                    got.diag_nan.contains(&0),
                    "{label}: coordinate 0 stays non-finite"
                );
            } else {
                assert!(
                    got.diag_nan.is_empty(),
                    "{label}: diag_nan {:?}",
                    got.diag_nan
                );
                assert!(
                    got.hess[(0, 0)] != 0.0,
                    "{label}: coordinate 0 must be live"
                );
            }
        }
    }
    /// Witness that two calls of a closure were in flight at the same time (#1935).
    ///
    /// A latch, not a timing heuristic: the first caller to find nobody else in flight blocks
    /// on a condition variable until a second caller arrives, so a correct parallel loop
    /// proves overlap as soon as a second worker picks up any item, however late the scheduler
    /// runs it. Only a serial loop — where no second caller can ever arrive while the first
    /// waits — reaches the watchdog, which then releases every later caller so the failing
    /// test ends in one watchdog period rather than one per item.
    struct OverlapWitness {
        /// `(in flight, overlap seen, gave up)`.
        state: std::sync::Mutex<(usize, bool, bool)>,
        cv: std::sync::Condvar,
    }

    impl OverlapWitness {
        const WATCHDOG: std::time::Duration = std::time::Duration::from_secs(30);

        fn new() -> Self {
            Self {
                state: std::sync::Mutex::new((0, false, false)),
                cv: std::sync::Condvar::new(),
            }
        }

        /// Call at the start of every evaluation.
        fn enter(&self) {
            let mut s = self.state.lock().unwrap();
            s.0 += 1;
            if s.0 >= 2 {
                s.1 = true;
                self.cv.notify_all();
            } else if !s.1 && !s.2 {
                let (mut g, timeout) = self
                    .cv
                    .wait_timeout_while(s, Self::WATCHDOG, |st| !st.1)
                    .unwrap();
                if timeout.timed_out() {
                    g.2 = true;
                }
                s = g;
            }
            s.0 -= 1;
        }

        fn overlapped(&self) -> bool {
            self.state.lock().unwrap().1
        }
    }

    /// #1935: a probe pass evaluates its points concurrently. Nothing numerical can see this
    /// — a serial pass returns the same values — so the witness is overlap itself. A serial
    /// pass (`par_iter` → `iter`, the pre-#1935 loop) never has two points in flight, and the
    /// assertion fails after one watchdog period.
    #[test]
    fn the_step_probe_evaluates_its_pass_concurrently() {
        use super::select_fd_step;

        let opts = crate::types::FitOptions::default();
        let witness = OverlapWitness::new();
        let quad = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };
        let ofv = |x: &[f64]| -> f64 {
            witness.enter();
            quad(x)
        };
        let x_hat = [0.5f64, -1.0, 2.0];
        let step = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                select_fd_step(
                    &x_hat,
                    &[0, 1, 2],
                    &[1.0; 3],
                    1e-2,
                    quad(&x_hat),
                    &ofv,
                    &opts,
                )
            });
        assert_eq!(step.n_halvings, 0);
        assert!(
            witness.overlapped(),
            "no two probe points were ever in flight together: the pass ran serially"
        );
    }

    /// #2147: an identity-packed θ̂ = 0.0077 under `eps = 1e-2`. Its legacy step
    /// `eps·(1 + |x̂|) ≈ 0.0101` crosses zero, where the objective returns a *finite* penalty
    /// (`1e16`, as a negative volume does) — so the probe's non-finite guard accepts it and
    /// the diagonal comes out ~1e20. With the floor `|θ₀|` the step is `eps·2|x̂|`, the minus
    /// point stays positive and the stencil is the quadratic's. Coordinate 1 at 7.7 keeps
    /// floor 1 and must keep today's step exactly, so the change straddles the gate.
    #[test]
    fn a_small_identity_theta_is_stepped_relative_to_its_floor() {
        use super::{fd_ofv_hessian, fd_step, select_fd_step};
        use std::sync::atomic::{AtomicBool, Ordering};

        let opts = crate::types::FitOptions::default();
        let x_hat = [0.0077f64, 7.7];
        let quad = |x: &[f64]| 3.0 * x[0] * x[0] + 0.2 * x[0] * x[1] + 0.5 * x[1] * x[1];
        let crossed = AtomicBool::new(false);
        let ofv = |x: &[f64]| {
            if x[0] <= 0.0 {
                crossed.store(true, Ordering::Relaxed);
                1e16
            } else {
                quad(x)
            }
        };
        let f0 = ofv(&x_hat);
        let eps = 1e-2;

        let floor = [0.0077, 1.0];
        let step = select_fd_step(&x_hat, &[0, 1], &floor, eps, f0, &ofv, &opts);
        assert_eq!((step.eps, step.n_halvings), (eps, 0));
        let got = fd_ofv_hessian(2, &x_hat, &[0, 1], &floor, eps, f0, &ofv, &opts).unwrap();
        assert!(
            !crossed.load(Ordering::Relaxed),
            "a stencil point crossed zero"
        );
        // A quadratic's central second differences are exact up to rounding of f ≈ 30 at
        // h₀ = 1.54e-4: measured worst relative error 1.8e-8 (the (0,0) entry; 1.2e-10 and
        // 2e-14 on the others). The legacy step misses it by ~1e20.
        for (ij, want) in [((0, 0), 6.0), ((0, 1), 0.2), ((1, 1), 1.0)] {
            let rel = (got.hess[ij] - want).abs() / want;
            assert!(rel < 1e-6, "H{ij:?} = {} vs {want}", got.hess[ij]);
        }
        // Coordinate 1's step is the legacy one, bit for bit.
        assert_eq!(
            fd_step(eps, x_hat[1], floor[1]).to_bits(),
            (eps * (1.0 + x_hat[1].abs())).to_bits()
        );

        // The straddle: floor 1 on coordinate 0 (the pre-#2147 step) crosses zero, is
        // accepted anyway, and builds the stencil on the penalty.
        let legacy = fd_ofv_hessian(2, &x_hat, &[0, 1], &[1.0, 1.0], eps, f0, &ofv, &opts).unwrap();
        assert!(crossed.load(Ordering::Relaxed));
        assert!(
            legacy.diag_nan.is_empty(),
            "the penalty is finite, so nothing is flagged"
        );
        assert!(legacy.hess[(0, 0)] > 1e19, "H00 = {}", legacy.hess[(0, 0)]);
    }

    /// #1935: `map_subjects` with `parallel = true` really re-solves subjects concurrently,
    /// and with `false` really does not — both sides of the flag in one test, so a branch
    /// stuck on either arm fails it. Combined with the production test that reads the log
    /// `map_subjects` writes from the branch it takes, this pins that the base point and
    /// `f0_fd` run in parallel and the stencil's points do not.
    #[test]
    fn map_subjects_runs_concurrently_exactly_when_asked() {
        use super::map_subjects;

        let pop = crate::types::Population {
            subjects: (0..6)
                .map(|k| crate::types::Subject {
                    id: format!("t1935-map-subjects-{k}"),
                    ..Default::default()
                })
                .collect(),
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let parallel = OverlapWitness::new();
        let got = pool.install(|| {
            map_subjects(&pop, true, |i| {
                parallel.enter();
                i * 10
            })
        });
        assert_eq!(got, vec![0, 10, 20, 30, 40, 50], "subject order is kept");
        assert!(
            parallel.overlapped(),
            "parallel = true ran the subjects serially"
        );

        // The serial arm cannot overlap by construction, so the latch would wait out its
        // watchdog here; a flat in-flight counter is enough to see a parallel serial arm.
        let in_flight = std::sync::atomic::AtomicUsize::new(0);
        let peak = std::sync::atomic::AtomicUsize::new(0);
        let got = pool.install(|| {
            map_subjects(&pop, false, |i| {
                use std::sync::atomic::Ordering::SeqCst;
                let now = in_flight.fetch_add(1, SeqCst) + 1;
                peak.fetch_max(now, SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(5));
                in_flight.fetch_sub(1, SeqCst);
                i * 10
            })
        });
        assert_eq!(got, vec![0, 10, 20, 30, 40, 50]);
        assert_eq!(
            peak.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "parallel = false must re-solve one subject at a time"
        );
        assert_eq!(
            super::reconverge_log::take("t1935-map-subjects-"),
            vec![(6, true), (6, false)],
            "the log records the branch taken"
        );
    }

    /// A cancel stops the FD Hessian route from evaluating further points. The objective sets
    /// the token on its first call; points already past their cancel check finish, so at most
    /// one per worker is evaluated, against the `2·n_free = 8` of a full probe pass (or the
    /// `2·n_free² = 32` of the whole route) a missing per-point check would let through.
    #[test]
    fn a_cancelled_fd_hessian_stops_evaluating_points() {
        use super::fd_ofv_hessian;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const THREADS: usize = 2;
        let flag = crate::cancel::CancelFlag::new();
        let opts = crate::types::FitOptions {
            cancel: Some(flag.clone()),
            ..Default::default()
        };
        let calls = AtomicUsize::new(0);
        let ofv = |x: &[f64]| -> f64 {
            calls.fetch_add(1, Ordering::SeqCst);
            flag.cancel();
            x.iter().map(|v| v * v).sum()
        };
        let x_hat = [0.5f64, -1.0, 2.0, 0.25];
        let got = rayon::ThreadPoolBuilder::new()
            .num_threads(THREADS)
            .build()
            .unwrap()
            .install(|| {
                fd_ofv_hessian(4, &x_hat, &[0, 1, 2, 3], &[1.0; 4], 1e-2, 1.0, &ofv, &opts)
            });
        assert!(got.is_none(), "a cancelled step must not return a Hessian");
        let n = calls.load(Ordering::SeqCst);
        assert!(
            (1..=THREADS).contains(&n),
            "{n} objective evaluations after the first call cancelled (at most {THREADS})"
        );
    }

    // ── #2261: every reconverged point of the step starts at the fit's (η̂, κ̂) ──────

    /// The covariance step on `fixture`'s default-parameter modes, with the exact R-matrix
    /// off so the base reconvergence and the whole-population stencil run: every IOV start
    /// of the step begins at the modes it was handed, κ̂ included.
    fn assert_cov_points_start_at_mode(
        label: &str,
        (model, pop, opts): (
            crate::types::CompiledModel,
            crate::types::Population,
            crate::types::FitOptions,
        ),
    ) {
        use crate::estimation::inner_optimizer::record_inner_solves;
        use crate::estimation::inner_optimizer::test_helpers::{
            assert_starts_at_modes, on_one_thread,
        };
        use crate::types::FitOptions;
        let p = &model.default_params;
        let (etas, hms, _, kappas) = crate::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, p, 200, 1e-8, None, None, 0, 0,
        );
        let opts = FitOptions {
            analytic_cov_hessian: false,
            ..opts
        };
        let x = crate::estimation::parameterization::pack_params(p);
        let rec = on_one_thread(|| {
            record_inner_solves(|| {
                super::compute_covariance(&x, p, &model, &pop, &etas, &hms, &kappas, &opts)
            })
            .1
        });
        assert!(
            rec.passes.is_empty(),
            "{label}: the step solves per subject"
        );
        assert_starts_at_modes(label, &rec, |_| true, &pop, &etas, &kappas);
        // The base reconvergence and every stencil point keep `cov_inner_tol`, not the FD
        // gradients' stencil tolerance (#2261), so a user's `cov_inner_tol` is what runs.
        let cov_tol = opts.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());
        assert!(
            cov_tol > crate::estimation::inner_optimizer::STENCIL_INNER_TOL,
            "{label}: fixture: cov_inner_tol must differ from the stencil tolerance"
        );
        assert!(
            rec.solve_tols
                .iter()
                .all(|(_, t)| t.to_bits() == cov_tol.to_bits()),
            "{label}: the covariance step re-solved at {:?}, not cov_inner_tol {cov_tol:e}",
            rec.solve_tols
                .iter()
                .map(|(_, t)| format!("{t:e}"))
                .collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn covariance_points_start_at_iov_mode_one_level() {
        use crate::estimation::inner_optimizer::test_helpers::warfarin_iov_fixture;
        assert_cov_points_start_at_mode("covariance step warfarin_iov", warfarin_iov_fixture());
    }

    #[test]
    fn covariance_points_start_at_iov_mode_nested_levels() {
        use crate::estimation::inner_optimizer::test_helpers::nested_levels_fixture;
        assert_cov_points_start_at_mode("covariance step nested_levels_3", nested_levels_fixture());
    }

    #[test]
    fn covariance_points_start_at_iov_mode_kappa_correlation() {
        use crate::estimation::inner_optimizer::test_helpers::kappa_correlation_fixture;
        assert_cov_points_start_at_mode(
            "covariance step warfarin_iov ar1",
            kappa_correlation_fixture(),
        );
    }

    /// #2261: the quadrature S-matrix's FD score (forced here by
    /// `reconverge_gradient_interval = 1`) re-solves at `cov_inner_tol`, the covariance step's
    /// tolerance, not the outer gradient's stencil tolerance, and from the fit's (η̂, κ̂).
    /// Mutation: drop `with_fd_inner_tol` → the score re-solves at the stencil tolerance.
    #[test]
    fn quadrature_s_matrix_fd_score_keeps_cov_inner_tol() {
        use crate::estimation::inner_optimizer::record_inner_solves;
        use crate::estimation::inner_optimizer::test_helpers::{
            assert_starts_at_modes, on_one_thread, warfarin_iov_fixture,
        };
        let (model, pop, opts) = warfarin_iov_fixture();
        let opts = crate::types::FitOptions {
            method: crate::types::EstimationMethod::Laplace,
            reconverge_gradient_interval: 1,
            ..opts
        };
        let p = &model.default_params;
        let x = crate::estimation::parameterization::pack_params(p);
        let bounds = crate::estimation::parameterization::compute_bounds(p);
        let free: Vec<usize> = crate::estimation::parameterization::packed_fixed_mask(p)
            .iter()
            .enumerate()
            .filter(|(_, f)| !**f)
            .map(|(i, _)| i)
            .collect();
        let (etas, hms, _, kappas) = crate::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, p, 200, 1e-8, None, None, 0, 0,
        );
        let rec = on_one_thread(|| {
            record_inner_solves(|| {
                super::assemble_score_cross_product(
                    &x, p, &model, &pop, &etas, &hms, &kappas, &bounds, &opts, &free,
                )
                .expect("the S matrix")
            })
            .1
        });
        let cov_tol = opts.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());
        assert!(cov_tol > crate::estimation::inner_optimizer::STENCIL_INNER_TOL);
        assert!(
            !rec.solve_tols.is_empty(),
            "fixture: the FD score must re-solve"
        );
        assert!(
            rec.solve_tols
                .iter()
                .all(|(_, t)| t.to_bits() == cov_tol.to_bits()),
            "the S-matrix score re-solved at {:?}, not cov_inner_tol {cov_tol:e}",
            rec.solve_tols
                .iter()
                .map(|(_, t)| format!("{t:e}"))
                .collect::<std::collections::BTreeSet<_>>()
        );
        assert_starts_at_modes("S-matrix FD score", &rec, |_| true, &pop, &etas, &kappas);
    }
}
