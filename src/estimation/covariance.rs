//! Covariance / standard-error subsystem (moved verbatim from
//! `estimation::outer_optimizer` in refactor T4). The FD-of-OFV Hessian step,
//! the eigen-floor inverse, the score cross-product, the non-PD SIR fallback,
//! and the progress reporter all live here. `outer_optimizer` retains only the
//! population optimizers, the outer-gradient family, `OuterResult`, and
//! `pop_nll`/`pop_nll_opts` (imported below).

use crate::estimation::cov_diagnostics::{
    format_offdiag_nan_warning, format_regularized_warning, format_salvage_note, CovHessianSource,
    CovRegularizationFacts, CovScopeDecline, OdeToleranceFacts,
};
use crate::estimation::inner_optimizer::find_ebe;
use crate::estimation::outer_optimizer::pop_nll_opts;
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::types::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;
use std::collections::HashSet;

/// Outcome of the FD covariance step. `matrix` is the n×n covariance with FIX
/// rows/cols zeroed; `warnings` carries non-fatal notes (regularisation applied,
/// off-diagonal FD stencil failures, etc.). Empty when everything was clean.
pub(crate) struct CovarianceOutput {
    pub matrix: DMatrix<f64>,
    pub warnings: Vec<String>,
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
fn fmt_eig(v: f64) -> String {
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
/// The result is embedded into the full packed-parameter covariance (zeros for
/// FIX parameters) and explicitly symmetrised, since the eigen-reconstruction
/// `V·diag·Vᵀ` can leave sub-ULP asymmetry that a downstream Cholesky rejects.
pub(crate) fn build_non_pd_fallback_proposal(
    hess_free_sym: &DMatrix<f64>,
    free_idx: &[usize],
    n_full: usize,
    inflation: f64,
) -> DMatrix<f64> {
    let eig = SymmetricEigen::new(hess_free_sym.clone());
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
    let cov_free = (&cov_free_raw + cov_free_raw.transpose()) * 0.5;
    // Embed free block into full n×n (FIX rows/cols stay zero).
    let mut cov = DMatrix::zeros(n_full, n_full);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            cov[(i, j)] = cov_free[(a, b)];
        }
    }
    cov
}

/// Choose a finite-difference step that keeps all free-parameter diagonal
/// stencils finite, starting from `initial_eps` and halving up to
/// `MAX_HALVINGS` times.
///
/// Returns `(chosen_eps, n_halvings)`. If every halving fails (all stencils
/// still non-finite at `initial_eps / 2^MAX_HALVINGS`), returns the final
/// eps anyway — the FD loop will detect and report the remaining failures.
///
/// The probe is on the scalar-OFV second-difference stencil
/// `(f₊ − 2·f₀ + f₋)/h²`, which is the exact stencil the IOV Hessian path uses.
/// The non-IOV path instead assembles the Hessian from central differences of
/// the analytical population gradient, so the OFV probe is a deliberate *proxy*
/// there: it shares the same underlying model evaluations (an OFV overflow at a
/// perturbation implies the gradient overflows too), is far cheaper than probing
/// the gradient, and the gradient FD loop carries its own `is_finite()` guard as
/// a backstop for the rare case the two disagree.
pub(crate) fn select_fd_step<F: Fn(&[f64]) -> f64>(
    x_hat: &[f64],
    free_idx: &[usize],
    initial_eps: f64,
    f0: f64,
    ofv: &F,
) -> (f64, usize) {
    const MAX_HALVINGS: usize = 8;
    let mut eps = initial_eps;
    let mut x = x_hat.to_vec();
    for halvings in 0..MAX_HALVINGS {
        let all_ok = free_idx.iter().all(|&i| {
            let hi = eps * (1.0 + x_hat[i].abs());
            x[i] = x_hat[i] + hi;
            let fp = ofv(&x);
            x[i] = x_hat[i] - hi;
            let fm = ofv(&x);
            x[i] = x_hat[i]; // always restore before returning
                             // Mirror the diagonal stencil the FD loop actually computes —
                             // (fp - 2·f0 + fm) / hi² — including the division. A finite
                             // numerator can still overflow once divided by a tiny hi², and the
                             // FD loop rejects on the quotient, so accepting the step here on the
                             // numerator alone would hand back an eps the loop then rejects.
            let h_ii = (fp - 2.0 * f0 + fm) / (hi * hi);
            h_ii.is_finite()
        });
        if all_ok {
            return (eps, halvings);
        }
        eps *= 0.5;
    }
    (eps, MAX_HALVINGS)
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
pub(crate) fn combine_covariance(
    method: CovarianceMethod,
    r_inv: DMatrix<f64>,
    s: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    match method {
        CovarianceMethod::Hessian => Some(r_inv),
        CovarianceMethod::Sandwich => Some(&r_inv * s * &r_inv),
        // Accept S⁻¹ only when S is full-rank (no eigenvalues clipped); a
        // rank-deficient or indefinite S yields `None`.
        CovarianceMethod::CrossProduct => match invert_psd_with_floor(s) {
            Some(inv) if inv.n_clipped == 0 => Some(inv.inverse),
            _ => None,
        },
    }
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
) -> Option<(DMatrix<f64>, Option<DMatrix<f64>>)> {
    let reference = r_inv_ref.and_then(|ri| combine_covariance(method, ri, s));
    Some((combine_covariance(method, r_inv, s)?, reference))
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
const COV_CANCELLED_MSG: &str =
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
    });

    // Per-subject scores in parallel (mirrors `build_gn_system`).
    //
    // The score cross-product evaluates the per-subject gradient directly at x̂.
    // Unlike the FD-built R-matrix — which reconverges η̂ at every perturbed point
    // and so captures the `log|H̃|` EBE-response `½·∂log|H̃|/∂η̂·dη̂/dθ` — the raw
    // analytic gradient holds η̂ fixed and drops it. Add it back here (the #274
    // `tᵢ` term, in −logL units; in −2logL units this contributes `2·tᵢ` to the
    // gradient) so the score matches how NONMEM differences the individual objective with
    // its conditional estimate responding to θ. This is what makes the FOCEI
    // S/RSR match NONMEM (warfarin RSR ≈ 1.8% with it, ≈ 5% without); the
    // alternative `∂a/∂θ` "a-response" was tested and is NOT what NONMEM's S
    // carries (it holds the model sensitivities `a` fixed at the linearization).
    // FOCE (`!interaction`) uses the Sheiner–Beal gradient, which has no `log|H̃|`
    // term — applying this Laplace-form `tᵢ` to FOCE was tested and over-corrects
    // (warfarin FOCE RSR 1.3% → 9.8% vs NONMEM), so the correction is FOCEI-only.
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
            if let Some(context) = &quadrature {
                // The subject score has NLL units already and cannot contain the
                // optimizer's population-level EBE penalty. Never square a failed row.
                let score = context.score(&population.subjects[i], eta_hats[i].as_slice(), kap_i)
                    .ok_or_else(|| format!("Covariance step failed: could not obtain a converged, finite quadrature score for subject {}. SE estimates not available.", population.subjects[i].id));
                report();
                return score;
            }
            let (_, mut gi) = crate::estimation::gauss_newton::subject_nll_pop_grad(
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
            if options.interaction {
                if let Some(ti) = crate::estimation::gauss_newton::subject_eta_response_correction(
                    None,
                    x_hat,
                    template,
                    model,
                    population,
                    i,
                    &eta_hats[i],
                    &h_matrices[i],
                    bounds,
                    options,
                ) {
                    for (g, t) in gi.iter_mut().zip(ti.iter()) {
                        *g += *t;
                    }
                }
            }
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

/// **Every** clause that kept this fit off the exact analytic covariance R-matrix (#520 C2).
///
/// All of them, not the first: the remedy sentence promises that an action "moves the fit onto
/// the analytic route", and that is only true when the action clears every clause that
/// declined. Dropping `gradient = fd` does not move a non-Gaussian model, and enabling
/// `analytic_cov_hessian` does not move a mixture — both were promised unconditionally before
/// (#1508 review §3). The three model-level gates are therefore collected side by side rather
/// than short-circuited, and the per-subject walk uses
/// [`crate::sens::provider::covariance_scope_declines`], the exhaustive half of the same gate
/// the routing decision runs.
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
    let mut out: Vec<CovScopeDecline> = Vec::new();
    if !options.analytic_cov_hessian {
        out.push(CovScopeDecline::Disabled);
    }
    if is_mixture {
        out.push(CovScopeDecline::Mixture);
    }
    // Mirrors `analytic_cov_hessian`'s own first bail: an AGQ/Laplace fit anchored on the exact
    // `H` needs fourth-order sensitivities for `∂²H/∂x²`, which nothing computes.
    if options.agq_nodes().is_some() && options.hessian_anchor() != HessianAnchor::GaussNewton {
        out.push(CovScopeDecline::ExactHessianAnchor);
    }
    let iov = model.n_kappa > 0;
    for subject in &population.subjects {
        for decline in crate::sens::provider::covariance_scope_declines(model, subject, iov) {
            if !out.contains(&decline) {
                out.push(decline);
            }
        }
    }
    if out.is_empty() {
        out.push(CovScopeDecline::PerSubjectBail);
    }
    out
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
/// never reaches here — `compute_covariance` gates the whole analytic path on `!is_mixture`.
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

impl AnalyticCovAssembly {
    /// The complete analytic Hessian, or `None` if *anything* declined.
    ///
    /// Test-facing only. Production reads the variants, because collapsing `Partial` into
    /// `None` here is precisely the all-or-nothing behaviour #1514 removed — a helper that
    /// made it a one-character change to reintroduce would not survive the next refactor.
    #[cfg(test)]
    pub(super) fn full(self) -> Option<DMatrix<f64>> {
        match self {
            AnalyticCovAssembly::Full(h) => Some(h),
            _ => None,
        }
    }
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
    use crate::estimation::sens_cov_hessian::{
        subject_packed_cov_hessian, subject_packed_cov_hessian_foce,
    };
    // `eta_hats` indexes `population.subjects` positionally everywhere below, and the
    // `par_iter().zip()` that follows would silently *truncate* to the shorter of the two —
    // dropping the tail subjects from both the sum and the declined list, i.e. returning a
    // `Full` matrix missing terms. Under the old all-or-nothing shape that was a wrong matrix
    // too; here it would additionally be reported as complete, so it is refused outright.
    if eta_hats.len() != population.subjects.len() {
        return AnalyticCovAssembly::Unavailable;
    }
    // The objective this assembly differentiates is the FOCE/FOCEI marginal. When
    // `agq_nodes()` is `Some` — `method = laplace` at any node count, or `method = focei`
    // with `n_agq > 1` — the fit minimised the AGQ marginal instead, under an anchor that
    // may not even be H̃ (`HessianAnchor::Exact` for Laplace), and with a quadrature
    // correction this assembly has no term for. `pop_nll_opts` exists precisely so no
    // production site reports "standard errors for a likelihood it never optimised"; the
    // analytic path bypasses that helper, so it must repeat its dispatch condition here
    // (PR #953 review finding 1).
    //
    // #251 narrows this: `method = focei` with `n_agq > 1` **is** served, by
    // `agq_cov_hessian`, which differentiates the quadrature marginal itself. The split is on
    // the *anchor*, not on `agq_nodes()`:
    //
    //   * `HessianAnchor::GaussNewton` (FOCEI) — `H̃ = Ω⁻¹ + Σ pⱼaⱼaⱼᵀ` is built from first
    //     derivatives of `f`, so `∂²H̃/∂x²` needs third-order sensitivities, which
    //     `subject_sensitivities_cov` already provides.
    //   * `HessianAnchor::Exact` (Laplace, at any node count) — `H = ∂²nll/∂b²` already carries
    //     `∂²f/∂η²`, so its second derivative needs **fourth** order. Nothing computes those, so
    //     Laplace keeps the FD covariance. Keying this off `agq_nodes()` instead of the anchor
    //     would report `H̃`-derived SEs for a fit anchored on `H` — the same class of error the
    //     bail was added for.
    let agq = if options.agq_nodes().is_some() {
        if options.hessian_anchor() != HessianAnchor::GaussNewton {
            return AnalyticCovAssembly::Unavailable;
        }
        options.agq_nodes()
    } else {
        None
    };
    // The quadrature rule and parameter unpacking are shared by every subject.
    let agq = agq.map(|n_agq| {
        let (nodes, weights) = crate::estimation::agq::gauss_hermite(n_agq);
        let params = crate::estimation::parameterization::unpack_params(x_hat, template);
        (nodes, weights, params)
    });
    let n = x_hat.len();
    let per_subject: Vec<Option<DMatrix<f64>>> = population
        .subjects
        .par_iter()
        .zip(eta_hats.par_iter())
        .enumerate()
        .map(|(i, (subject, eta_hat))| {
            let b: std::borrow::Cow<'_, [f64]> = if model.n_kappa > 0 {
                let kap = kappas.get(i)?;
                if kap.len() != crate::stats::likelihood::iov_occasion_groups(subject).len()
                    || kap.iter().any(|k| k.len() != model.n_kappa)
                {
                    return None;
                }
                std::borrow::Cow::Owned(
                    eta_hat
                        .iter()
                        .copied()
                        .chain(kap.iter().flat_map(|k| k.iter().copied()))
                        .collect(),
                )
            } else {
                std::borrow::Cow::Borrowed(eta_hat.as_slice())
            };
            // Cooperative cancel. The FD stencil checks on every perturbed point; this loop is
            // `2·(n_theta+n_eta)+1` provider evaluations plus an O(n_eta³·n_obs) assembly per
            // subject, and for the default `covariance_method = r` there is no later checkpoint
            // — so without this, the Ctrl-C affordance `saem.rs` advertises before the
            // covariance step (#893) was inoperative. Returning `None` alone would drop into
            // the *more* expensive FD stencil; the caller re-checks the flag and reports
            // cancelled instead (PR #953 review finding 10).
            if crate::cancel::is_cancelled(&options.cancel) {
                return None;
            }
            let h = if let Some((nodes, weights, params)) = &agq {
                // FOCEI-anchored quadrature (#251). The grid is rebuilt here from the same
                // Gauss-Hermite rule the objective used, so the Hessian differentiates the grid the
                // fit actually evaluated — the same reason the proposal jitter is carried.
                let (grid, pi) = crate::estimation::agq::subject_grid_and_weights(
                    model, subject, params, &b, nodes, weights,
                )?;
                crate::estimation::agq_cov_hessian::subject_packed_agq_cov_hessian(
                    model, subject, template, params, &b, &grid, &pi,
                )
            } else if options.interaction {
                subject_packed_cov_hessian(model, subject, template, x_hat, &b)
            } else {
                subject_packed_cov_hessian_foce(model, subject, template, x_hat, &b)
            }?;
            if h.nrows() != n || h.ncols() != n || h.iter().any(|v| !v.is_finite()) {
                return None;
            }
            // ×2 — the OFV convention, and the one place this can be silently wrong.
            //
            // `subject_packed_cov_hessian` is the second derivative of `subject_packed_gradient`,
            // which is `∂Fᵢ/∂x` with `OFV = 2·Σᵢ Fᵢ` (see `population_gradient_sens`'s
            // `grad[k] += 2.0 * gi[k]`). The stencil this replaces differences `2·pop_nll`, so
            // `hess` here must be `∂²OFV/∂x²`, and the caller's `covariance = 2·H⁻¹` assumes it.
            // Summing the per-subject Hessians unscaled yields exactly half of that, which inflates
            // every standard error by √2 — with no other symptom, since the matrix stays symmetric,
            // positive-definite and plausibly sized.
            Some(2.0 * h)
        })
        .collect();
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
    if declined.len() as f64 >= SALVAGE_MAX_DECLINED_FRACTION * population.subjects.len() as f64 {
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
/// `pub(super)` so `sens_cov_hessian`'s tests can run the *production* stencil over an
/// arbitrary subject set and check that the population's equals the sum of the per-subject
/// ones — the premise #1514 rests on. A test that rebuilt the difference formulas to do that
/// would be comparing two copies rather than one formula on two inputs.
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
/// Returns `None` when the step was cancelled mid-stencil.
///
/// #256: flattened to one `par_iter` over all ~2·n_free² perturbed OFV points (subjects
/// iterated serially inside `ofv`) instead of a serial loop firing a per-subject `par_iter` at
/// every point — removing the fork/join overhead of a rayon barrier per point.
pub(super) fn fd_ofv_stencil<F: Fn(&[f64]) -> f64 + Sync>(
    n: usize,
    x_hat: &[f64],
    free_idx: &[usize],
    eps: f64,
    f0: f64,
    ofv: &F,
    options: &FitOptions,
) -> Option<FdStencil> {
    let nf = free_idx.len();
    let hsteps: Vec<f64> = free_idx
        .iter()
        .map(|&i| eps * (1.0 + x_hat[i].abs()))
        .collect();
    // Flat list of perturbation SPECS (not materialised x-vectors): 2 per diagonal (±hᵢ), then
    // 4 per (a<b) off-diagonal pair. Each par_iter task clones `x_hat` once and applies its
    // spec, so only ~n_threads vectors are live at a time instead of all ~2·nf² perturbed
    // points held resident for the whole reduction (the pre-#298 O(nf²·np) footprint) (#298).
    #[derive(Clone, Copy)]
    enum Pert {
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
    let mut specs: Vec<Pert> = Vec::with_capacity(2 * nf + 2 * nf * nf);
    for a in 0..nf {
        let (i, hi) = (free_idx[a], hsteps[a]);
        specs.push(Pert::Single { i, di: hi });
        specs.push(Pert::Single { i, di: -hi });
    }
    let n_diag = specs.len();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..nf {
        for b in (a + 1)..nf {
            let (i, j) = (free_idx[a], free_idx[b]);
            let (hi, hj) = (hsteps[a], hsteps[b]);
            for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)] {
                specs.push(Pert::Pair {
                    i,
                    di: si * hi,
                    j,
                    dj: sj * hj,
                });
            }
            pairs.push((a, b));
        }
    }
    let report = cov_progress("Hessian", specs.len(), options.verbose);
    let vals: Vec<f64> = specs
        .par_iter()
        .map(|p| {
            // Cooperative cancel: skip this point's EBE reconvergence and
            // return NaN so the queue drains; bailed on below.
            if crate::cancel::is_cancelled(&options.cancel) {
                report();
                return f64::NAN;
            }
            let mut xv = x_hat.to_vec();
            match *p {
                Pert::Single { i, di } => xv[i] += di,
                Pert::Pair { i, di, j, dj } => {
                    xv[i] += di;
                    xv[j] += dj;
                }
            }
            let v = ofv(&xv);
            report();
            v
        })
        .collect();
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

/// Re-solve the inner EBE loop for every subject of `pop` at the packed point `xv`,
/// warm-started from `warm[i]`, serially over subjects.
///
/// NONMEM reconverges the conditional estimates at every perturbed point in its covariance
/// step; holding η̂/H fixed gives a Hessian with the wrong curvature — indefinite even on
/// warfarin, which previously forced eigenvalue clipping (#129) and inflated the SEs.
///
/// Serial (not the parallel `run_inner_loop_warm`) because the covariance step parallelises
/// over perturbed POINTS, not subjects; nested parallelism is what #256 removed. `find_ebe` is
/// deterministic per subject, so the per-subject EBEs are bit-identical to the parallel loop.
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
) -> (
    ModelParameters,
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    Vec<Vec<DVector<f64>>>,
) {
    let params = unpack_params(xv, template);
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let n = pop.subjects.len();
    let mut ehs = Vec::with_capacity(n);
    let mut hms = Vec::with_capacity(n);
    let mut kaps = Vec::with_capacity(n);
    for i in 0..n {
        let ebe = find_ebe(
            model,
            &pop.subjects[i],
            &params,
            options.inner_maxiter,
            cov_inner_tol,
            Some(warm[i].as_slice()),
            Some(&mu_k),
            0,
        );
        ehs.push(ebe.eta);
        hms.push(ebe.h_matrix);
        kaps.push(ebe.kappas);
    }
    (params, ehs, hms, kaps)
}

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
    let n = x_hat.len();
    let initial_eps = options.fd_hessian_step;
    if initial_eps <= 0.0 || !initial_eps.is_finite() {
        return CovarianceStepResult::Unusable(format!(
            "Covariance step failed: fd_hessian_step must be positive and finite, got {}. \
             SE estimates not available.",
            initial_eps
        ));
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

    // `h_matrices` (the H from the fit) is intentionally unused: the covariance
    // step reconverges the EBEs at every perturbed point and recomputes H there.
    // It stays in the signature for symmetry with `eta_hats` (the reconvergence
    // warm-start) and with the other optimizers' call sites.
    let _ = h_matrices;

    let cov_inner_tol = options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());

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
    let subset_cov_ofv = |xv: &[f64], pop: &Population, warm: &[DVector<f64>]| -> f64 {
        let (params, ehs, hms, kaps) =
            reconverge_population(xv, model, pop, template, warm, options, cov_inner_tol);
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
        subset_cov_ofv(xv, population, eta_hats)
    };

    // Reconverge once at `x_hat` and keep the result. Both consumers need it: the base OFV
    // for the FD stencil, and — as of #953 review finding 8 — the analytic path's η̂.
    //
    // The analytic assembly is derived under stationarity (`∂lᵢ/∂η|_η̂ = 0`): both the M2
    // envelope term `−M_ξᵀH⁻¹M_ζ` and the `inner_eta_responses` chain assume it. Feeding it
    // the modes converged during the fit at `inner_tol` would make `cov_inner_tol` a silent
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
            eta_hats,
            options,
            cov_inner_tol,
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
        return CovarianceStepResult::Unusable(reason);
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
    let assembly = if options.analytic_cov_hessian && !is_mixture {
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
        return CovarianceStepResult::Unusable(COV_CANCELLED_MSG.to_string());
    }

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
            let (pop, warm) = subset_population(population, eta_hats, &declined);
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

    if let Some(scope) = fd_scope.as_ref() {
        // The objective this stencil differences: the whole population's covariance OFV, or —
        // on the salvage route — only the declined subjects' share of it.
        let fd_ofv = |xv: &[f64]| -> f64 {
            match scope {
                FdScope::Whole => cov_ofv(xv),
                FdScope::Subset { pop, warm } => subset_cov_ofv(xv, pop, warm),
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
            FdScope::Subset { pop, warm } => subset_cov_ofv(x_hat, pop, warm),
        };

        // Adaptively select the FD step: halve up to 8× until all free-parameter
        // diagonal stencils are finite. Most models use the initial step; halving
        // only kicks in when the OFV overflows at the default perturbation size.
        let (eps, n_halvings) = select_fd_step(x_hat, &free_idx, initial_eps, f0_fd, &fd_ofv);
        if options.verbose && n_halvings > 0 {
            eprintln!(
                "  [covariance] Adaptive FD step: reduced {:.3e} → {:.3e} ({} halving{})",
                initial_eps,
                eps,
                n_halvings,
                if n_halvings == 1 { "" } else { "s" }
            );
        }

        let stencil = match fd_ofv_stencil(n, x_hat, &free_idx, eps, f0_fd, &fd_ofv, options) {
            Some(stencil) => stencil,
            None => return CovarianceStepResult::Unusable(COV_CANCELLED_MSG.to_string()),
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
        let reason = format!(
            "Covariance step failed: Hessian has ill-conditioned entries for the following \
             parameter(s) — {}. SE estimates not available.",
            problem_params.join("; ")
        );
        if options.verbose {
            eprintln!("  {}", reason);
        }
        return CovarianceStepResult::Unusable(reason);
    }

    // Build the reduced Hessian over free indices, invert, then embed back
    // into the full n×n covariance matrix (FIX rows/cols stay zero).
    let n_free = free_idx.len();
    if n_free == 0 {
        // Nothing to estimate — return an all-zero covariance so downstream
        // SE extraction reports zeros (all params FIX).
        return CovarianceStepResult::Success(CovarianceOutput {
            matrix: DMatrix::zeros(n, n),
            warnings: vec![],
        });
    }
    let mut hess_free = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            hess_free[(a, b)] = hess[(i, j)];
        }
    }
    let hess_free_sym = (&hess_free + hess_free.transpose()) * 0.5;

    let inv = match invert_psd_with_floor(&hess_free_sym) {
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
                    let fallback_proposal =
                        build_non_pd_fallback_proposal(&hess_free_sym, &free_idx, n, 4.0);
                    return CovarianceStepResult::FailedNonPd {
                        reason: format_non_pd_warning(&eigvals),
                        fallback_proposal,
                    };
                }
                None => {
                    return CovarianceStepResult::Unusable(
                        "Covariance step failed: could not compute eigenvalues of the \
                         FD Hessian (Hessian may contain NaN or Inf). \
                         SE estimates not available."
                            .to_string(),
                    );
                }
            }
        }
    };
    // The FD Hessian is of the OFV = −2·logL. The asymptotic covariance is the
    // inverse observed Fisher information R = Hessian of −logL = ½·H_ofv, so
    // R⁻¹ = 2·H_ofv⁻¹. Without this factor every SE is 1/√2 too small.
    let r_inv = inv.inverse.clone() * 2.0;
    // The same `R⁻¹` with the floored directions dropped instead of floored — the covariance
    // the data alone supports. Carried through the *same* estimator and the *same* reported
    // -parameter transform as `r_inv`, so the inflation the message prints is the inflation of
    // the numbers on the page (#1508 review §1). Built only when the floor actually fired and
    // only when the warning can be emitted at all (the cross-product estimator never returns
    // `R⁻¹`, so nothing there is inflated by this floor).
    let r_inv_ref = (inv.n_clipped > 0
        && options.covariance_method != CovarianceMethod::CrossProduct)
        .then(|| inv.unclipped_inverse.clone() * 2.0);

    // Select the covariance estimator (NONMEM `$COV MATRIX=`). `R⁻¹` is the
    // model-based default; `S⁻¹` and `R⁻¹SR⁻¹` additionally need the per-subject
    // score cross-product `S = Σᵢ gᵢgᵢᵀ`. `S` is on the −logL scale
    // (`gᵢ = ∂(−logLᵢ)/∂θ`, no factor of 2), matching `R = ½·H_ofv`.
    // Anchored against NONMEM `$COV MATRIX=S`/`RSR` for both FOCEI (#266) and
    // FOCE (no-INTER) (#250): all SEs within ~10% of NONMEM.
    let (cov_free, cov_free_ref) = if options.covariance_method == CovarianceMethod::Hessian {
        (r_inv, r_inv_ref)
    } else {
        let s_free = assemble_score_cross_product(
            x_hat, template, model, population, eta_hats, h_matrices, kappas, &bounds, options,
            &free_idx,
        );
        if crate::cancel::is_cancelled(&options.cancel) {
            return CovarianceStepResult::Unusable(COV_CANCELLED_MSG.to_string());
        }
        let s_free = match s_free {
            Ok(s) => s,
            Err(reason) => return CovarianceStepResult::Unusable(reason),
        };
        match combine_covariance_and_reference(options.covariance_method, r_inv, r_inv_ref, &s_free)
        {
            Some(pair) => pair,
            None => {
                return CovarianceStepResult::Unusable(
                    "Covariance step failed: the score cross-product matrix S is singular or \
                     rank-deficient (covariance_method = s); typically fewer subjects than free \
                     parameters, or collinear per-subject scores. Use covariance_method = r or \
                     rsr. SE estimates not available."
                        .to_string(),
                );
            }
        }
    };

    let mut cov = DMatrix::zeros(n, n);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            cov[(i, j)] = cov_free[(a, b)];
        }
    }

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
    // killed nothing before this line, and kills `one_declining_subject_reproduces_the_pure_
    // analytic_covariance` after). `format_salvage_note`'s own emptiness guard is what makes
    // it total for the two non-hybrid arms, not a second gate on the same fact.
    //
    // Gated on the estimator for the same reason the eigen-floor warning twelve lines below is
    // (#1516 review §1): under `covariance_method = s` the returned covariance is `S⁻¹` from
    // `assemble_score_cross_product` alone and `R` is discarded, so *nothing* about the route
    // that assembled `R` reaches the numbers on the page. The note's closing claim — that the
    // named subjects' terms use a different estimator than the rest — would then be a
    // statement about a matrix this step threw away.
    let salvaged_ids: Vec<&str> = match source {
        CovHessianSource::HybridRMatrix
            if options.covariance_method != CovarianceMethod::CrossProduct =>
        {
            salvaged
                .iter()
                .map(|&i| population.subjects[i].id.as_str())
                .collect()
        }
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
    // cross-product path returns `S⁻¹` (with a full-rank `S` guaranteed above), so
    // a clipped `R` there would be a misleading note about a matrix it didn't use.
    if inv.n_clipped > 0 && options.covariance_method != CovarianceMethod::CrossProduct {
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
            ode: OdeToleranceFacts::from_route(model, population),
        };
        let msg = format_regularized_warning(&facts);
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    } else if options.verbose {
        eprintln!("  Covariance step successful");
    }

    // Soft warning: cross-partial stencils that returned NaN/Inf contributed nothing, so some
    // off-diagonal correlation is missing for these parameters and their SEs may be
    // over-optimistic. *How much* is missing depends on the route — the whole cross-partial on
    // the population stencil, only the declined subjects' share of it on the hybrid — so the
    // sentence is built from `source` rather than stated once (#1514 review §1).
    if !fd_offdiag_nan.is_empty() {
        // Sort by packed index so the warning message is deterministic regardless
        // of HashSet iteration order.
        let mut sorted_idx: Vec<usize> = fd_offdiag_nan.iter().cloned().collect();
        sorted_idx.sort_unstable();
        let names: Vec<String> = sorted_idx
            .iter()
            .map(|&i| packed_param_label(i, template))
            .collect();
        let msg = format_offdiag_nan_warning(&names.join(", "), source);
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    }

    CovarianceStepResult::Success(CovarianceOutput {
        matrix: cov,
        warnings: cov_warnings,
    })
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
/// regularisation scale-equivariant; the absolute floor handles the edge case
/// where the whole spectrum is tiny.
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
/// re-converges every subject's EBEs at each of `n(n+1)/2` stencil points —
/// ~320,000 re-converged population objectives at `n = 800`, which will not
/// finish. Above [`COV_HESSIAN_MAX_DIM`] a *defaulted* `Hessian` is therefore
/// routed to the cross-product, which needs one pass. An explicit
/// `covariance_method = r` is the user's call and is honoured, but is told what
/// it is about to cost.
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
    if n <= crate::types::COV_HESSIAN_MAX_DIM || requested != CovarianceMethod::Hessian {
        return (requested, None);
    }
    // #254: `S` is a sum of per-*subject* score cross-products, and a parameter
    // prior has no subject decomposition — it contributes one score for the whole
    // population, not N of them — so `S` cannot carry the prior's information at
    // all. Auto-routing a priored fit onto it would silently report unpenalized
    // standard errors for a penalized fit, which is the one thing the covariance
    // wiring exists to prevent. Stay on `R` (which does carry the prior) and say
    // why it will be slow, rather than being quietly wrong and fast.
    if has_priors {
        let stencil = n * (n + 1) / 2;
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
    let stencil = n * (n + 1) / 2;
    if explicitly_set {
        (
            requested,
            Some(format!(
                "covariance_method = r with {n} free parameters: the R matrix is a \
                 finite-difference Hessian that re-converges every subject's EBEs at each \
                 of {stencil} stencil points. Set `covariance_method = s` (the score \
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
    let (routed, scale_warning) = scale_routed_covariance_method(
        model.free_packed_dim(),
        options.covariance_method,
        options.covariance_method_set,
        !model.priors.is_empty(),
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
    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let cov_timer = std::time::Instant::now();
    let matrix = match compute_covariance(
        x_hat, template, model, population, eta_hats, h_matrices, kappas, options,
    ) {
        CovarianceStepResult::Success(out) => {
            warnings.extend(out.warnings);
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
        // `routed`, not `options.covariance_method` as it arrived: the two are the
        // same binding by construction here (`options` is rebound to
        // `scaled_options` exactly when they differ), but naming `routed` is what
        // makes it read as the estimator `compute_covariance` was configured with
        // rather than the one the caller asked for (#1382).
        method: published_covariance_method(matrix.as_ref(), routed),
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

    // ── #520 C1: the magnitudes the severity grade is made from ──────────────

    /// The diagonal of a covariance, as the inflation metric consumes it.
    fn diag(m: &DMatrix<f64>) -> Vec<f64> {
        (0..m.nrows()).map(|k| m[(k, k)]).collect()
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
        let (routed, warning) =
            scale_routed_covariance_method(n, CovarianceMethod::Hessian, false, false);
        assert_eq!(routed, CovarianceMethod::CrossProduct);
        let warning = warning.expect("the substitution must be reported");
        assert!(warning.contains("score cross-product"), "{warning}");
        // The stencil count is the whole argument for switching, so it has to
        // be in the message rather than left for the reader to work out.
        assert!(
            warning.contains(&(n * (n + 1) / 2).to_string()),
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
            warning.contains(&(n * (n + 1) / 2).to_string()),
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
        // `s` and `rsr` already cost one pass; the guard has nothing to say.
        for method in [CovarianceMethod::CrossProduct, CovarianceMethod::Sandwich] {
            let (routed, warning) =
                scale_routed_covariance_method(COV_HESSIAN_MAX_DIM * 8, method, false, false);
            assert_eq!(routed, method);
            assert!(warning.is_none());
        }
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
}
