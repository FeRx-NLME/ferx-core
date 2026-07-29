//! Covariance / standard-error subsystem (moved verbatim from
//! `estimation::outer_optimizer` in refactor T4). The FD-of-OFV Hessian step,
//! the eigen-floor inverse, the score cross-product, the non-PD SIR fallback,
//! and the progress reporter all live here. `outer_optimizer` retains only the
//! population optimizers, the outer-gradient family, `OuterResult`, and
//! `pop_nll`/`pop_nll_opts` (imported below).

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
    } else {
        format!("packed[{}]", packed_idx)
    }
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

/// Eigenvalues of `sym` sorted descending. Returns `None` if any eigenvalue is non-finite.
pub(crate) fn extract_eigenvalues(sym: &DMatrix<f64>) -> Option<Vec<f64>> {
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
) -> DMatrix<f64> {
    let n_free = free_idx.len();
    let n_subj = population.subjects.len();

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
    let scores: Vec<Vec<f64>> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            // Cooperative cancel: skip the per-subject gradient and return a
            // cheap zero score so the in-flight rayon queue drains fast. The
            // caller (`compute_covariance`) re-checks the flag and discards this
            // matrix before it is used, so the placeholder is never trusted.
            if crate::cancel::is_cancelled(&options.cancel) {
                report();
                return vec![0.0; x_hat.len()];
            }
            let kap_i = if i < kappas.len() {
                kappas[i].as_slice()
            } else {
                &[]
            };
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
            gi
        })
        .collect();

    let mut s = DMatrix::zeros(n_free, n_free);
    for gi in &scores {
        let gi_free = DVector::from_iterator(n_free, free_idx.iter().map(|&k| gi[k]));
        s.ger(1.0, &gi_free, &gi_free, 1.0); // s += gi_free * gi_freeᵀ (full outer product)
    }
    s
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
/// The exact analytic R-matrix (#436): `Σᵢ ∂²Fᵢ/∂x²` in packed coordinates, or `None` when
/// **any** subject is outside the analytic scope.
///
/// All-or-nothing on purpose. Mixing an analytic block for some subjects with a
/// finite-difference block for others would produce a matrix that is neither, and the
/// resulting SEs would be silently method-dependent per subject. The finite-difference
/// stencil is correct for everything, so it is the honest fallback.
///
/// Serial over subjects, reduced in subject order, so the result cannot depend on thread
/// count — matching how the FD stencil and the outer gradient reduce (#703). The covariance
/// step runs once per fit, so the per-subject assembly is not on any hot path.
fn analytic_cov_hessian(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x_hat: &[f64],
    eta_hats: &[DVector<f64>],
    options: &FitOptions,
) -> Option<DMatrix<f64>> {
    use crate::estimation::sens_cov_hessian::{
        subject_packed_cov_hessian, subject_packed_cov_hessian_foce,
    };
    // IOV is out of scope: the assembly is written over the η-only random-effect block, not
    // the stacked `[η, κ]` one.
    if population
        .subjects
        .iter()
        .any(|s| !s.dose_occasions.is_empty() && model.n_kappa > 0)
    {
        return None;
    }
    let n = x_hat.len();
    let mut acc = DMatrix::<f64>::zeros(n, n);
    for (subject, eta_hat) in population.subjects.iter().zip(eta_hats.iter()) {
        let h = if options.interaction {
            subject_packed_cov_hessian(model, subject, template, x_hat, eta_hat.as_slice())
        } else {
            subject_packed_cov_hessian_foce(model, subject, template, x_hat, eta_hat.as_slice())
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
        acc += 2.0 * h;
    }
    Some(acc)
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
    let bounds = compute_bounds(template);

    // `h_matrices` (the H from the fit) is intentionally unused: the covariance
    // step reconverges the EBEs at every perturbed point and recomputes H there.
    // It stays in the signature for symmetry with `eta_hats` (the reconvergence
    // warm-start) and with the other optimizers' call sites.
    let _ = h_matrices;
    let n_subj_cov = population.subjects.len();

    // Re-solve the inner EBE loop at a packed point, warm-started from the
    // converged EBEs, serially over subjects. NONMEM reconverges the conditional
    // estimates at every perturbed point in its covariance step; holding η̂/H
    // fixed gives a Hessian with the wrong curvature — indefinite even on
    // warfarin, which previously forced eigenvalue clipping (#129) and inflated
    // the SEs.
    //
    // This single helper is the reconvergence used by both covariance-OFV
    // evaluations — the base-OFV evaluation and the second-difference stencil's
    // `serial_ofv` (which now serves the non-IOV and IOV cases alike, since the
    // FD-of-OFV Hessian is the sole R stencil) — so they cannot drift apart
    // (#298). It is
    // serial (not the parallel `run_inner_loop_warm`) because the covariance step
    // parallelises over perturbed POINTS, not subjects; nested parallelism is
    // what #256 removed. `find_ebe` is deterministic per subject, so the
    // per-subject EBEs are bit-identical to the parallel loop.
    // The covariance step reconverges EBEs at its own tolerance (`cov_inner_tol`),
    // decoupled from the fit's `inner_tol`: the second-difference-of-OFV R-matrix is
    // far more sensitive to EBE precision than the fit, so LTBS tightens it by default
    // (the `g = ln(f)` Hessian needs it) and any model can opt in. Defaults to
    // `inner_tol` for non-LTBS (byte-identical). See `FitOptions::effective_cov_inner_tol`.
    let cov_inner_tol = options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());
    let reconverge_point = |xv: &[f64]| -> (
        ModelParameters,
        Vec<DVector<f64>>,
        Vec<DMatrix<f64>>,
        Vec<Vec<DVector<f64>>>,
    ) {
        let params = unpack_params(xv, template);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let mut ehs = Vec::with_capacity(n_subj_cov);
        let mut hms = Vec::with_capacity(n_subj_cov);
        let mut kaps = Vec::with_capacity(n_subj_cov);
        for i in 0..n_subj_cov {
            let ebe = find_ebe(
                model,
                &population.subjects[i],
                &params,
                options.inner_maxiter,
                cov_inner_tol,
                Some(eta_hats[i].as_slice()),
                Some(&mu_k),
                0,
            );
            ehs.push(ebe.eta);
            hms.push(ebe.h_matrix);
            kaps.push(ebe.kappas);
        }
        (params, ehs, hms, kaps)
    };

    // Covariance OFV = −2·logL at a reconverged point. For FOCEI the per-subject
    // marginal already carries ηᵀΩ⁻¹η + log|Ω|; for FOCE we add that prior here.
    let ofv = |xv: &[f64]| -> f64 {
        let (params, ehs, hms, kaps) = reconverge_point(xv);
        let foce_nll = pop_nll_opts(model, population, &params, &ehs, &hms, &kaps, options);
        // Covariance OFV = −2·logL = 2·pop_nll for both FOCE and FOCEI.
        //
        // FOCE uses the Sheiner–Beal linearised marginal `(y−f₀)ᵀR̃⁻¹(y−f₀) +
        // log|R̃|` with R̃ = HΩHᵀ + R. By Woodbury that marginal *already* carries
        // the Ω penalty (it equals the conditional form including η̂ᵀΩ⁻¹η̂ +
        // log|Ω|), so its Ω-curvature is complete. An earlier version added the
        // η̂ᵀΩ⁻¹η̂ + log|Ω| prior here for the FOCE branch, which double-counted Ω
        // and flattened the Ω-block curvature — the source of the ~31%-low FOCE
        // omega SEs (issue #243). FOCEI's Almquist–Laplace marginal likewise
        // carries the prior internally. So neither method needs an add-back.
        2.0 * foce_nll
    };

    let base_ofv = ofv(x_hat);
    if !base_ofv.is_finite() {
        // Diagnose: check Omega conditioning to distinguish Omega collapse from
        // a model-evaluation overflow/underflow.
        let params_at = unpack_params(x_hat, template);
        let reason = match extract_eigenvalues(&params_at.omega.matrix) {
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
    // rows/cols at zero (→ SE = 0 downstream).
    let fixed_mask = packed_fixed_mask(template);
    // Structural-zero Ω off-diagonals (the cross-block elements of a mixed
    // block+diagonal Ω, where `free_mask[(i,j)] == false`) are not estimated
    // parameters — the analytical population gradient zeroes them, so their
    // Hessian diagonal is flat. Exclude them from the free set exactly like FIX
    // parameters; otherwise the ill-conditioning guard below rejects the entire
    // covariance step. (Before #243 the omega-prior add-back iterated all
    // lower-triangle entries and gave these a spurious non-zero curvature, which
    // masked the issue for the FOCE path; FOCEI never had that mask.)
    let structural_zero = omega_structural_zero_mask(template);
    let free_idx: Vec<usize> = (0..n)
        .filter(|&i| !fixed_mask[i] && !structural_zero[i])
        .collect();

    let f0 = base_ofv;

    // Adaptively select the FD step: halve up to 8× until all free-parameter
    // diagonal stencils are finite. Most models use the initial step; halving
    // only kicks in when the OFV overflows at the default perturbation size.
    let (eps, n_halvings) = select_fd_step(x_hat, &free_idx, initial_eps, f0, &ofv);
    if options.verbose && n_halvings > 0 {
        eprintln!(
            "  [covariance] Adaptive FD step: reduced {:.3e} → {:.3e} ({} halving{})",
            initial_eps,
            eps,
            n_halvings,
            if n_halvings == 1 { "" } else { "s" }
        );
    }

    let mut hess = DMatrix::zeros(n, n);

    // Track FD failures at source so diagnostics name the right cause (a NaN/Inf
    // stencil result is not a genuine zero curvature). HashSet for O(1) ops.
    let mut fd_diag_nan: HashSet<usize> = HashSet::new();
    let mut fd_offdiag_nan: HashSet<usize> = HashSet::new();

    // ── Analytic R-matrix (#436), attempted before the FD stencil ────────────────
    //
    // The exact observed information from third-order sensitivities, assembled per
    // subject and summed. All-or-nothing: a single subject outside the analytic
    // scope drops the whole population back to the finite-difference stencil below,
    // because a Hessian half-assembled from two different approximations would be
    // neither.
    //
    // This is not the stencil #639 removed. That one finite-differenced a gradient
    // that held `a = ∂f/∂η` fixed — an envelope approximation, which is why it
    // biased weakly-identified structural SEs. This is the exact second derivative
    // of the same marginal the outer loop minimises.
    let analytic_hess: Option<DMatrix<f64>> = if options.analytic_cov_hessian {
        analytic_cov_hessian(model, population, template, x_hat, eta_hats, options)
    } else {
        None
    };
    if let Some(h) = analytic_hess.as_ref() {
        hess.copy_from(h);
        if options.verbose {
            eprintln!("  [covariance] analytic R-matrix (third-order sensitivities, #436)");
        }
    }
    if analytic_hess.is_none()
    // Reconverged-OFV second-difference Hessian (3-point diagonal, 4-point
    // off-diagonal), reconverging the EBEs at each perturbed point. The sole
    // covariance R stencil: it recomputes the marginal curvature end-to-end
    // (`a = ∂f/∂η` and the `log|H̃|` EBE-response included) at every perturbed
    // point, so it serves FOCE, FOCEI and IOV, and additive/proportional/combined
    // error uniformly — no envelope approximation, no held-fixed `a`.
    {
        // `pop_nll` dispatches on the kappa count, so this stencil is correct for
        // both the IOV (joint η, κ) and the non-IOV (η-only) cases.
        //
        // #256: flattened to one `par_iter` over all ~2·n_free² perturbed OFV
        // points (subjects iterated serially inside `serial_ofv`) instead of the
        // old serial loop that fired a per-subject `par_iter` at every point —
        // removing the fork/join overhead of firing a rayon barrier per point.
        // Bit-identical to the serial stencil: each point's OFV is the same
        // `2·pop_nll` at the same per-subject `find_ebe`, and the difference
        // formulas/assembly are unchanged; only the scheduling differs.
        let f0 = base_ofv;
        let serial_ofv = |xv: &[f64]| -> f64 {
            let (params, ehs, hms, kaps) = reconverge_point(xv);
            2.0 * pop_nll_opts(model, population, &params, &ehs, &hms, &kaps, options)
        };

        let nf = free_idx.len();
        let hsteps: Vec<f64> = free_idx
            .iter()
            .map(|&i| eps * (1.0 + x_hat[i].abs()))
            .collect();
        // Flat list of perturbation SPECS (not materialised x-vectors): 2 per
        // diagonal (±hᵢ), then 4 per (a<b) off-diagonal pair. Each par_iter task
        // clones `x_hat` once and applies its spec, so only ~n_threads vectors are
        // live at a time instead of all ~2·nf² perturbed points held resident for
        // the whole reduction (the pre-#298 O(nf²·np) footprint) (#298).
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
                let v = serial_ofv(&xv);
                report();
                v
            })
            .collect();
        if crate::cancel::is_cancelled(&options.cancel) {
            return CovarianceStepResult::Unusable(COV_CANCELLED_MSG.to_string());
        }
        // Diagonal: (f(x+h) − 2f(x) + f(x−h)) / h².
        for a in 0..nf {
            let i = free_idx[a];
            let hi = hsteps[a];
            let h_ii = (vals[2 * a] - 2.0 * f0 + vals[2 * a + 1]) / (hi * hi);
            if h_ii.is_finite() {
                hess[(i, i)] = h_ii;
            } else {
                fd_diag_nan.insert(i);
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
                fd_offdiag_nan.insert(i);
                fd_offdiag_nan.insert(j);
            }
        }
    }

    // Diagnose fatal Hessian problems. Use the FD-failure trackers for accurate
    // cause labels — post-hoc checks on `hess` would always read 0 (finite) because
    // non-finite FD results are never stored (only the zero initialisation remains).
    let mut problem_params: Vec<String> = Vec::new();
    for &i in &free_idx {
        let diag = hess[(i, i)];
        if fd_diag_nan.contains(&i) {
            // Diagonal FD stencil overflowed; zero stored value does not mean flat
            // objective. Adjust fd_hessian_step or check for model overflow.
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
    let r_inv = inv.inverse * 2.0;

    // Select the covariance estimator (NONMEM `$COV MATRIX=`). `R⁻¹` is the
    // model-based default; `S⁻¹` and `R⁻¹SR⁻¹` additionally need the per-subject
    // score cross-product `S = Σᵢ gᵢgᵢᵀ`. `S` is on the −logL scale
    // (`gᵢ = ∂(−logLᵢ)/∂θ`, no factor of 2), matching `R = ½·H_ofv`.
    // Anchored against NONMEM `$COV MATRIX=S`/`RSR` for both FOCEI (#266) and
    // FOCE (no-INTER) (#250): all SEs within ~10% of NONMEM.
    let cov_free = if options.covariance_method == CovarianceMethod::Hessian {
        r_inv
    } else {
        let s_free = assemble_score_cross_product(
            x_hat, template, model, population, eta_hats, h_matrices, kappas, &bounds, options,
            &free_idx,
        );
        if crate::cancel::is_cancelled(&options.cancel) {
            return CovarianceStepResult::Unusable(COV_CANCELLED_MSG.to_string());
        }
        match combine_covariance(options.covariance_method, r_inv, &s_free) {
            Some(c) => c,
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

    // The Hessian eigenvalue-floor warning is about `R`. It is relevant only when
    // the returned covariance actually uses `R⁻¹` (Hessian and sandwich); the
    // cross-product path returns `S⁻¹` (with a full-rank `S` guaranteed above), so
    // a clipped `R` there would be a misleading note about a matrix it didn't use.
    if inv.n_clipped > 0 && options.covariance_method != CovarianceMethod::CrossProduct {
        let pct = inv.n_clipped * 100 / n_free.max(1);
        // Informal thresholds: ≤33 % clipped → minor concern; 34–50 % → caution; >50 % → unreliable.
        // Note: integer truncation means the boundary moves in steps of 1/n_free; for small
        // n_free adjacent clipped counts can jump directly from "minor" to "severe".
        let (severity, interp) = match pct {
            0..=33 => ("minor", "Standard errors are likely reliable."),
            34..=50 => (
                "moderate",
                "Standard errors should be interpreted with caution; \
                 consider SIR-based confidence intervals.",
            ),
            _ => (
                "severe",
                "Standard errors are likely unreliable; \
                 SIR-based confidence intervals are recommended.",
            ),
        };
        let msg = format!(
            "Covariance step regularized: eigenvalue floor applied to FD Hessian \
             ({} of {} free-block eigenvalues clipped; min eig = {:.3e}, floor = {:.3e}; \
             severity: {}). {}",
            inv.n_clipped, n_free, inv.min_eigenvalue, inv.floor, severity, interp
        );
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    } else if options.verbose {
        eprintln!("  Covariance step successful");
    }

    // Soft warning: cross-partial FD stencils that returned NaN/Inf were stored as 0,
    // so off-diagonal correlation is missing for these parameters. SEs for the named
    // parameters may be over-optimistic (correlation with other parameters is absent).
    if !fd_offdiag_nan.is_empty() {
        // Sort by packed index so the warning message is deterministic regardless
        // of HashSet iteration order.
        let mut sorted_idx: Vec<usize> = fd_offdiag_nan.iter().cloned().collect();
        sorted_idx.sort_unstable();
        let names: Vec<String> = sorted_idx
            .iter()
            .map(|&i| packed_param_label(i, template))
            .collect();
        let msg = format!(
            "Covariance step: off-diagonal FD stencil(s) non-finite for {}. \
             Cross-partial correlation set to 0; SE for these parameter(s) \
             may be over-optimistic. Try tuning fd_hessian_step.",
            names.join(", ")
        );
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
    /// Floor used for clipping. Same shape rules as `min_eigenvalue`.
    pub floor: f64,
    /// How many eigenvalues fell below the floor and were clipped.
    pub n_clipped: usize,
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
            floor: f64::INFINITY,
            n_clipped: 0,
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

    Some(RegularizedInverse {
        inverse,
        min_eigenvalue: min_eig,
        floor,
        n_clipped,
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
        }
    }
}

#[cfg(test)]
mod tests {
    // The moved cov-specific unit tests remain in `outer_optimizer`'s test module
    // (they reach the moved symbols via the cross-module import added there). The
    // `run_covariance_step` gate + match is exercised end-to-end by every
    // estimator finalizer's integration/lib tests.
}
