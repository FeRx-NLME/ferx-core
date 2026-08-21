//! Sampling Importance Resampling (SIR) for parameter uncertainty estimation.
//!
//! Implements the SIR procedure described in Dosne et al. (2017):
//! "Improving the estimation of parameter uncertainty distributions in
//! nonlinear mixed effects models using sampling importance resampling"
//!
//! SIR provides a non-parametric estimate of parameter uncertainty that is
//! more robust than the asymptotic covariance matrix.

use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::pop_nll_opts;
use crate::estimation::parameterization::{
    compute_bounds, compute_mu_k, coordinate_names, pack_params, packed_fixed_mask, unpack_params,
    PackedBounds,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{weighted::WeightedIndex, ChiSquared, Distribution, StandardNormal};
use rayon::prelude::*;

/// Results from the SIR procedure.
#[derive(Debug, Clone)]
pub struct SirResult {
    /// 95% CI (2.5th, 97.5th percentile) for each theta on original scale
    pub ci_theta: Vec<(f64, f64)>,
    /// 95% CI for each omega diagonal element
    pub ci_omega: Vec<(f64, f64)>,
    /// 95% CI for each sigma
    pub ci_sigma: Vec<(f64, f64)>,
    /// Effective sample size (ESS = 1 / sum(w_k^2))
    pub effective_sample_size: f64,
    /// Resampled packed parameter vectors, retained when
    /// `FitOptions.sir_keep_samples = true`. `None` otherwise.
    /// Length equals `FitOptions.sir_resamples` when populated.
    pub resamples_packed: Option<Vec<Vec<f64>>>,
    /// Diagnostics about the proposal that callers should surface to the user
    /// — currently the rank deficiency and the bound-driven shrinkage the
    /// proposal needed (#1021). Empty on a clean run.
    pub warnings: Vec<String>,
}

/// How many proposal standard deviations must fit between the ML estimate and
/// the *nearer* of its packed bounds. A direction much wider than that is not
/// merely inefficient: every draw along it fails the bounds check in the weight
/// loop, and SIR degenerates to "All SIR samples had invalid weights" (#1021).
/// `4.0` (±2 sd of usable room on each side) is deliberately generous: the cap
/// is meant to catch a proposal direction that is *wider than the room the
/// parameter has to move in* — the signature of an eigenvalue-floored,
/// non-identified direction — not to trim a genuinely wide but legitimate one.
/// A tighter cap would silently narrow the CIs of a parameter whose real
/// uncertainty fills its user-declared bounds.
///
/// The guarantee is approximate, not exact, for two reasons that
/// [`proposal_sd_caps`] and [`CAP_TRIGGER_FACTOR`] between them keep small: the
/// cap is applied per eigen-direction while a coordinate's realised marginal sd
/// sums over directions (`Σ_i λ_i v_ki²`), and the proposal is Student-t rather
/// than Gaussian. The t inflation is compensated in `proposal_sd_caps`; the
/// per-direction accumulation is not, but only grossly-overshooting directions
/// are capped at all, so at most a handful contribute.
const PROPOSAL_BOUND_SIGMAS: f64 = 4.0;

/// How far a direction must overshoot a coordinate's usable room before the
/// bound cap touches it, expressed as a multiple of the per-coordinate sd cap.
///
/// Without this gate the cap fires on *legitimately* wide directions. The
/// packed bounds are narrow for the variance components — `compute_bounds`
/// gives an omega log-Cholesky diagonal `[-6, 6]` and a sigma `[-8, 5]`, so
/// their usable room is only a few units — and a collapsing omega genuinely has
/// log-scale uncertainty of that order. Shrinking it would silently narrow its
/// CI and mislabel it in the warning as "eigenvalue-floored (non-identified)
/// curvature". The failure this cap exists for overshoots by 1e3–1e4, so a
/// factor of ten separates the two cases cleanly (#1037).
const CAP_TRIGGER_FACTOR: f64 = 10.0;

/// Floor for a coordinate's usable room, as a fraction of its full packed bound
/// width. An ML estimate sitting *on* a bound has zero room, which would cap
/// every direction loading on it to a zero-variance proposal and abort SIR with
/// "no positive eigenvalue". Keeping a sliver of room keeps the proposal PD;
/// only directions that already overshoot by [`CAP_TRIGGER_FACTOR`] are shrunk
/// to it, and those are non-identified anyway.
const PROPOSAL_MIN_ROOM_FRAC: f64 = 0.01;

/// Relative floor for near-null proposal directions, mirroring the Hessian
/// floor in [`crate::estimation::covariance::invert_psd_with_floor`].
const PROPOSAL_EIG_FLOOR_REL: f64 = 1e-10;

/// Loadings below this magnitude are not reported when naming the parameters
/// that make up a degenerate or shrunk proposal direction.
const DIRECTION_LOADING_MIN: f64 = 0.15;

/// Why a proposal sample contributed no weight. Tallied so a run in which
/// *every* sample is rejected can say which check did the rejecting (#1021).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleOutcome {
    Accepted,
    /// Packed coordinate index that fell outside its bound.
    OutOfBounds(usize),
    NonPositiveParams,
    NonFiniteOfv,
    Cancelled,
}

/// A SIR proposal that has been made safe to sample from, plus a record of
/// what had to be done to the raw covariance to get there.
#[derive(Debug, Clone)]
pub(crate) struct ConditionedProposal {
    /// Lower-triangular Cholesky factor of the conditioned free-block covariance.
    pub chol: DMatrix<f64>,
    /// `log|C|` of the conditioned free block, taken from the (modified)
    /// eigenvalues rather than the Cholesky diagonal.
    pub log_det: f64,
    /// Directions with effectively zero variance — the rank deficiency that
    /// remains *after* FIX-ed parameters have been removed. Described as
    /// "PAR_A +0.71, PAR_B -0.70".
    pub null_dirs: Vec<String>,
    /// Directions shrunk to keep draws inside the packed bounds, worst first.
    pub capped_dirs: Vec<String>,
}

impl ConditionedProposal {
    /// User-facing notes about the conditioning, empty when the raw covariance
    /// needed no repair.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.null_dirs.is_empty() {
            out.push(format!(
                "proposal covariance is rank-deficient beyond the FIX-ed parameters: \
                 {} direction(s) carry no uncertainty [{}]. SIR holds those parameter \
                 combinations at their ML values, so their CIs are not explored — \
                 they are not identified by the data.",
                self.null_dirs.len(),
                self.null_dirs.join("; ")
            ));
        }
        if !self.capped_dirs.is_empty() {
            out.push(format!(
                "proposal was shrunk in {} direction(s) so draws mostly stay inside \
                 the parameter bounds [{}]. Those directions come from eigenvalue-floored \
                 (non-identified) curvature in the covariance step; the SIR CIs along \
                 them understate the true uncertainty.",
                self.capped_dirs.len(),
                self.capped_dirs.join("; ")
            ));
        }
        out
    }
}

/// Name the parameters that load on eigenvector column `col`, largest first.
///
/// Loadings below [`DIRECTION_LOADING_MIN`] are noise on a direction that has a
/// dominant parameter. On a model with many free coordinates, though, a
/// degenerate direction can be spread thinly enough that *no* loading clears the
/// threshold — and "no dominant parameter" leaves the user with nothing to act
/// on, while the accompanying advice ("fix or drop one parameter from each
/// listed combination") points at nothing. In that case fall back to the three
/// largest loadings whatever their magnitude (#1037).
fn describe_direction(eigenvectors: &DMatrix<f64>, col: usize, names: &[String]) -> String {
    let mut loadings: Vec<(usize, f64)> = (0..eigenvectors.nrows())
        .map(|k| (k, eigenvectors[(k, col)]))
        .collect();
    loadings.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if loadings.is_empty() {
        return "no dominant parameter".to_string();
    }
    let n_above = loadings
        .iter()
        .take_while(|(_, v)| v.abs() >= DIRECTION_LOADING_MIN)
        .count();
    let n_report = if n_above == 0 { 3 } else { n_above.min(3) };
    loadings.truncate(n_report);
    loadings
        .iter()
        .map(|(k, v)| {
            let name = names.get(*k).map(String::as_str).unwrap_or("?");
            format!("{name} {v:+.2}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Largest proposal standard deviation each free coordinate can carry.
///
/// The bound cap used to derive this from the *width* of the packed box,
/// `(upper - lower) / PROPOSAL_BOUND_SIGMAS`, which silently assumes the
/// proposal centre sits in the middle of that box. It does not: a theta declared
/// `(0, 100)` whose ML estimate is `0.5` packs to bounds
/// `[ln(1e-10), ln(100)] = [-23.0, 4.6]` around `x̂ = -0.69`, so it has 5.3 of
/// room above and 22.3 below. A width-derived cap would allow ±13.8 and leave
/// the large majority of draws above the upper bound — the very rejection the
/// cap exists to prevent, now with a warning claiming the proposal had been made
/// bound-safe (#1037).
///
/// So the room is the distance to the *nearer* bound, and the cap keeps
/// `PROPOSAL_BOUND_SIGMAS / 2` standard deviations inside it. Two corrections on
/// top of that:
///
///  * the proposal is Student-t with `nu = sir_df` degrees of freedom, whose
///    draws are `sqrt(nu / (nu - 2))` wider than its Cholesky scale, so the cap
///    is divided by that factor;
///  * room is floored at [`PROPOSAL_MIN_ROOM_FRAC`] of the box width, so an
///    estimate sitting exactly on its bound still yields a PD proposal instead
///    of aborting SIR with "no positive eigenvalue".
fn proposal_sd_caps(
    x_hat: &[f64],
    bounds: &PackedBounds,
    free_idx: &[usize],
    sir_df: f64,
) -> Vec<f64> {
    // Var(t_nu) = nu / (nu - 2), undefined at or below 2 df — there the raw
    // Cholesky scale is the only finite proxy available.
    let t_inflation = if sir_df > 2.0 {
        (sir_df / (sir_df - 2.0)).sqrt()
    } else {
        1.0
    };
    free_idx
        .iter()
        .map(|&i| {
            let room = (x_hat[i] - bounds.lower[i])
                .min(bounds.upper[i] - x_hat[i])
                .max(0.0);
            let floor = (bounds.upper[i] - bounds.lower[i]).max(0.0) * PROPOSAL_MIN_ROOM_FRAC;
            room.max(floor) * 2.0 / PROPOSAL_BOUND_SIGMAS / t_inflation
        })
        .collect()
}

/// Turn the free-block covariance into a proposal SIR can sample from.
///
/// Each eigen-direction's variance is
///  * floored at `λ_max · PROPOSAL_EIG_FLOOR_REL` when it is ≤ 0 or negligible
///    (a ridge / rank deficiency that survived the FIX exclusion), and
///  * capped so that `sqrt(λ)·|v_k| ≤ sd_caps[k]` for every coordinate `k` the
///    direction loads on — but *only* when it overshoots that cap by more than
///    [`CAP_TRIGGER_FACTOR`], so a legitimately wide direction is left alone
///    (#1037).
///
/// `sd_caps[k]` is the largest proposal standard deviation coordinate `k` can
/// carry and still keep `PROPOSAL_BOUND_SIGMAS / 2` sd inside the *nearer* of
/// its packed bounds; see [`proposal_sd_caps`].
///
/// The cap is what makes SIR survive a covariance matrix whose non-identified
/// directions were eigenvalue-floored during inversion: those come back with
/// variances around `1/floor` (1e7 and up in packed log-space), which without a
/// cap puts every single draw outside the bounds (#1021).
pub(crate) fn condition_free_proposal(
    sub_cov: &DMatrix<f64>,
    sd_caps: &[f64],
    names: &[String],
) -> Result<ConditionedProposal, String> {
    let n = sub_cov.nrows();
    debug_assert_eq!(
        n,
        sub_cov.ncols(),
        "condition_free_proposal needs a square block"
    );
    debug_assert_eq!(n, sd_caps.len(), "one sd cap per free coordinate");

    let sym = (sub_cov + sub_cov.transpose()) * 0.5;
    let eig = sym.clone().symmetric_eigen();
    if eig.eigenvalues.iter().any(|v| !v.is_finite()) {
        return Err(
            "SIR proposal covariance has non-finite eigenvalues — the covariance step \
             produced an unusable matrix; re-run the fit with `covariance = true` and \
             check the covariance warnings."
                .to_string(),
        );
    }
    // Pass 1 — cap each direction at the widest variance that keeps
    // ±(PROPOSAL_BOUND_SIGMAS / 2) standard deviations inside the usable room of
    // every coordinate it loads on.
    let caps: Vec<f64> = (0..n)
        .map(|i| {
            let mut cap = f64::INFINITY;
            for (k, &sd_cap) in sd_caps.iter().enumerate() {
                let load = eig.eigenvectors[(k, i)].abs();
                if load > 1e-8 {
                    cap = cap.min((sd_cap / load).powi(2));
                }
            }
            cap
        })
        .collect();
    // The cap only fires on a *gross* overshoot. `CAP_TRIGGER_FACTOR` is an sd
    // multiple, so the variance trigger is its square.
    let trigger = CAP_TRIGGER_FACTOR * CAP_TRIGGER_FACTOR;
    let capped_eigs: Vec<f64> = (0..n)
        .map(|i| {
            let lam = eig.eigenvalues[i];
            if lam > caps[i] * trigger {
                caps[i]
            } else {
                lam
            }
        })
        .collect();

    // The near-null floor is anchored on the *capped* spectrum, not the raw one.
    // An eigenvalue-floored direction can come back at 1e7+; anchoring on it
    // would put the relative floor at ~1e-3 and flag every genuinely
    // well-determined direction (variance ~1e-4) as null — inflating real,
    // informative variances by orders of magnitude.
    let max_eig = capped_eigs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max_eig <= 0.0 {
        return Err(
            "SIR proposal covariance has no positive eigenvalue — there is no uncertainty \
             direction to sample; the covariance step carries no usable information."
                .to_string(),
        );
    }
    let floor = (max_eig * PROPOSAL_EIG_FLOOR_REL).max(1e-12);

    // Pass 2 — apply the floor to the capped spectrum and record what changed.
    let mut lambdas = DVector::zeros(n);
    let mut null_dirs = Vec::new();
    let mut capped: Vec<(f64, String)> = Vec::new();
    for i in 0..n {
        let raw = eig.eigenvalues[i];
        let mut lam = capped_eigs[i];
        if capped_eigs[i] < raw && caps[i] >= floor {
            capped.push((
                (raw / caps[i]).sqrt(),
                format!(
                    "{} (sd {:.2e} → {:.2e})",
                    describe_direction(&eig.eigenvectors, i, names),
                    raw.sqrt(),
                    caps[i].sqrt()
                ),
            ));
        }
        if lam < floor {
            null_dirs.push(describe_direction(&eig.eigenvectors, i, names));
            lam = floor;
        }
        // A coordinate pinned to a zero-width bound would cap at exactly 0;
        // keep the proposal strictly PD so the Cholesky below succeeds.
        lambdas[i] = lam.max(1e-12);
    }
    capped.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // A healthy block must not be perturbed. Round-tripping it through `V Λ Vᵀ`
    // changes it at ~1e-16 relative, which changes every drawn sample and
    // therefore the SIR CIs — so when no eigenvalue was capped or floored,
    // factor the input directly and leave the draws bit-identical (#1037).
    let untouched = (0..n).all(|i| lambdas[i] == eig.eigenvalues[i]);
    let chol = untouched
        .then(|| sym.cholesky().map(|c| c.l()))
        .flatten()
        .map(Ok)
        .unwrap_or_else(|| {
            let cov_raw =
                &eig.eigenvectors * DMatrix::from_diagonal(&lambdas) * eig.eigenvectors.transpose();
            let cov = (&cov_raw + cov_raw.transpose()) * 0.5;
            cov.clone()
                .cholesky()
                .or_else(|| {
                    // Eigen-reconstruction can leave a sub-ULP indefinite
                    // remainder; a jitter proportional to the spectrum recovers it.
                    let jitter = DMatrix::identity(n, n) * (max_eig * 1e-12).max(1e-14);
                    (&cov + jitter).cholesky()
                })
                .map(|c| c.l())
                .ok_or_else(|| {
                    "SIR proposal covariance could not be made positive definite after \
                     eigenvalue conditioning."
                        .to_string()
                })
        })?;

    Ok(ConditionedProposal {
        chol,
        log_det: lambdas.iter().map(|l| l.ln()).sum::<f64>(),
        null_dirs,
        capped_dirs: capped.into_iter().map(|(_, d)| d).collect(),
    })
}

/// Build the error returned when no proposal sample earned a finite weight.
///
/// The bare "All SIR samples had invalid weights" gave no hint of *why*; this
/// reports the rejection tally, the coordinates whose bounds were overshot, and
/// the proposal's rank/shrinkage diagnostics (#1021).
fn all_invalid_weights_message(
    outcomes: &[SampleOutcome],
    coord_names: &[String],
    conditioned: &ConditionedProposal,
) -> String {
    let mut n_bounds = 0usize;
    let mut n_nonpos = 0usize;
    let mut n_ofv = 0usize;
    let mut n_cancelled = 0usize;
    let mut per_coord: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for o in outcomes {
        match *o {
            SampleOutcome::OutOfBounds(i) => {
                n_bounds += 1;
                *per_coord.entry(i).or_insert(0) += 1;
            }
            SampleOutcome::NonPositiveParams => n_nonpos += 1,
            SampleOutcome::NonFiniteOfv => n_ofv += 1,
            SampleOutcome::Cancelled => n_cancelled += 1,
            SampleOutcome::Accepted => {}
        }
    }
    let mut msg = format!(
        "All {} SIR samples had invalid weights (rejected: {} out of bounds, {} \
         non-positive theta/sigma/omega, {} non-finite OFV, {} cancelled).",
        outcomes.len(),
        n_bounds,
        n_nonpos,
        n_ofv,
        n_cancelled
    );
    if !per_coord.is_empty() {
        let mut top: Vec<(usize, usize)> = per_coord.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        top.truncate(3);
        let listed = top
            .iter()
            .map(|(i, c)| {
                let name = coord_names.get(*i).map(String::as_str).unwrap_or("?");
                format!("{name} ({c})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        msg.push_str(&format!(
            " Bound first overshot by: {listed} (count of samples, first offending coordinate only)."
        ));
    }
    for w in conditioned.warnings() {
        msg.push(' ');
        // Capitalise the fragment so it reads as its own sentence here.
        let mut chars = w.chars();
        if let Some(first) = chars.next() {
            msg.push_str(&first.to_uppercase().to_string());
            msg.push_str(chars.as_str());
        }
    }
    if !conditioned.null_dirs.is_empty() || !conditioned.capped_dirs.is_empty() {
        msg.push_str(
            " Fix or drop one parameter from each listed combination, or re-fit a model \
             the data can identify; the asymptotic covariance is not a usable SIR proposal \
             as it stands.",
        );
    }
    msg
}

/// Math kernel for the SIR procedure. Operates on pre-built parameter and
/// EBE arrays; users should typically call the higher-level
/// [`run_sir`](crate::run_sir) wrapper in `estimation::run_sir`, which takes
/// a `FitResult` and handles `ModelParameters` reconstruction, EBE
/// extraction, and source-file integrity checks.
///
/// # Arguments
/// * `model` - The compiled model
/// * `population` - The dataset
/// * `params` - ML parameter estimates
/// * `eta_hats` - ML EBE estimates (for warm-starting inner loop)
/// * `proposal_cov` - Covariance matrix in packed (log-transformed) parameter space
/// * `ofv_hat` - OFV at ML estimates
/// * `options` - Fit options containing SIR settings
pub fn run_sir_core(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    proposal_cov: &DMatrix<f64>,
    ofv_hat: f64,
    options: &FitOptions,
) -> Result<SirResult, String> {
    let n_samples = options.sir_samples;
    let n_resamples = options.sir_resamples;

    if n_resamples > n_samples {
        return Err("sir_resamples must be <= sir_samples".to_string());
    }

    // Pack ML estimates as the proposal center
    let x_hat = pack_params(params);
    let n_packed = x_hat.len();

    if proposal_cov.nrows() != n_packed || proposal_cov.ncols() != n_packed {
        return Err(format!(
            "Covariance matrix dimensions ({},{}) don't match packed parameters ({})",
            proposal_cov.nrows(),
            proposal_cov.ncols(),
            n_packed,
        ));
    }

    // Restrict the proposal to the free subspace. `compute_covariance` zeroes
    // the rows/cols of FIX-ed parameters, and `compute_bounds` pins their
    // bounds to `lower == upper == x_hat[i]`. Sampling on the full space would
    // (after regularising the singular covariance) perturb fixed indices by
    // ~sqrt(reg) ≈ 1e-4, which then fails the strict bounds check on every
    // sample — yielding "All SIR samples had invalid weights" for any model
    // with at least one FIX-ed parameter. Sampling on the free block instead
    // keeps fixed indices exactly at `x_hat`, and uses `d = n_free` as the
    // Student-t dimensionality so the importance weights are consistent.
    let fixed_mask = packed_fixed_mask(params);
    let free_idx: Vec<usize> = (0..n_packed).filter(|&i| !fixed_mask[i]).collect();
    let n_free = free_idx.len();
    if n_free == 0 {
        return Err("run_sir_core: every packed parameter is FIX — nothing to sample.".to_string());
    }

    // Symmetrize first, then extract the free block (rows/cols of non-FIX
    // indices) before Cholesky.
    let sym_cov_full = (proposal_cov + proposal_cov.transpose()) * 0.5;
    let mut sub_cov = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            sub_cov[(a, b)] = sym_cov_full[(i, j)];
        }
    }

    // Condition the free block into a proposal SIR can actually sample from
    // (#1021). Two failure modes are handled here, both of which used to
    // degenerate into "All SIR samples had invalid weights":
    //
    //  * near-null directions — a rank deficiency left over *after* FIX-ed
    //    parameters are removed (a likelihood ridge; two parameters that
    //    determine only their sum). These are floored to keep the proposal PD.
    //  * explosive directions — the covariance step floors the FD Hessian's
    //    eigenvalues before inverting it (`invert_psd_with_floor`), so a
    //    non-identified direction comes back with variance ≈ 1/floor ≈ 1e7+.
    //    In packed (log) space that is a proposal sd of thousands: every draw
    //    lands outside the parameter bounds and is rejected. Those directions
    //    are shrunk so ±2 sd still fits inside the room the ML estimate has.
    let bounds = compute_bounds(params);
    let coord_names = coordinate_names(params);
    let free_names: Vec<String> = free_idx
        .iter()
        .map(|&i| {
            coord_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("packed[{i}]"))
        })
        .collect();
    let sd_caps = proposal_sd_caps(&x_hat, &bounds, &free_idx, options.sir_df);
    let conditioned = condition_free_proposal(&sub_cov, &sd_caps, &free_names)?;
    if options.verbose {
        for w in conditioned.warnings() {
            eprintln!("  SIR: {w}");
        }
    }
    let proposal_chol = conditioned.chol.clone();

    // Log-determinant of the conditioned free-block proposal covariance (for
    // density computation). Uses n_free, matching the Student-t dimensionality.
    let log_det_proposal = conditioned.log_det;

    let mut rng = match options.sir_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(12345),
    };

    if options.verbose {
        eprintln!(
            "  SIR: drawing {} samples, resampling {}...",
            n_samples, n_resamples
        );
    }

    // Step 1: Pre-generate all samples (RNG is sequential).
    // Use a multivariate Student-t proposal with nu degrees of freedom.
    // Sampling: draw z ~ N(0,I), chi2 ~ chi2(nu), then scale z by sqrt(nu/chi2).
    // Heavier tails than MVN improve ESS for parameters near boundaries (e.g. omega variances).
    let nu = options.sir_df;
    let chi2_dist = ChiSquared::new(nu).map_err(|e| format!("sir_df invalid: {e}"))?;

    let d = n_free as f64;
    // Cache lgamma terms that are constant across all samples.
    let log_norm =
        lgamma((nu + d) / 2.0) - lgamma(nu / 2.0) - (d / 2.0) * (nu * std::f64::consts::PI).ln();
    // At the centre the quadratic form is 0, so log_q_hat = log_norm - 0.5*log_det.
    let log_q_hat = log_norm - 0.5 * log_det_proposal;

    let mut z_vectors: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let z_free: Vec<f64> = (0..n_free).map(|_| rng.sample(StandardNormal)).collect();
        let chi2: f64 = chi2_dist.sample(&mut rng);
        let scale = (nu / chi2).sqrt();
        let z_vec_free = DVector::from_column_slice(&z_free);
        let delta_free = &proposal_chol * &z_vec_free * scale;
        // Build the full packed sample: free indices get x_hat + delta_free,
        // fixed indices stay pinned at x_hat (so the strict bounds check
        // `lower == upper == x_hat[i]` passes).
        let mut x_k = x_hat.clone();
        for (a, &i) in free_idx.iter().enumerate() {
            x_k[i] += delta_free[a];
        }
        samples.push(x_k);
        // store L_free⁻¹(delta_free) = z_free * scale for the quadratic form
        // in log_q_k. Length = n_free.
        z_vectors.push(z_free.into_iter().map(|zi| zi * scale).collect());
    }

    // Step 2: Evaluate importance weights in parallel (warm-started inner loop)
    let inner_maxiter = options.inner_maxiter;
    let inner_tol = options.inner_tol;

    let (log_weights, outcomes): (Vec<f64>, Vec<SampleOutcome>) = samples
        .par_iter()
        .zip(z_vectors.par_iter())
        .map(|(x_k, z)| {
            if crate::cancel::is_cancelled(&options.cancel) {
                return (f64::NEG_INFINITY, SampleOutcome::Cancelled);
            }
            // Reject samples outside parameter bounds (avoids wasting inner-loop work).
            // The first offending coordinate is recorded so a total rejection can
            // name the parameter whose bound the proposal keeps overshooting (#1021).
            let out_of_bounds = x_k
                .iter()
                .zip(bounds.lower.iter().zip(bounds.upper.iter()))
                .position(|(&x, (&lo, &hi))| x < lo || x > hi);
            if let Some(i) = out_of_bounds {
                return (f64::NEG_INFINITY, SampleOutcome::OutOfBounds(i));
            }

            let params_k = unpack_params(x_k, params);

            // Check for invalid parameters: theta, sigma, and omega
            let theta_invalid = params_k.theta.iter().any(|&t| !t.is_finite() || t <= 0.0);
            let sigma_invalid = params_k
                .sigma
                .values
                .iter()
                .any(|&s| !s.is_finite() || s <= 0.0);
            let n_eta = params_k.omega.dim();
            let omega_invalid = (0..n_eta).any(|i| {
                let var = params_k.omega.matrix[(i, i)];
                let lii = params_k.omega.chol[(i, i)];
                !var.is_finite() || var <= 0.0 || !lii.is_finite() || lii <= 0.0
            });
            if theta_invalid || sigma_invalid || omega_invalid {
                return (f64::NEG_INFINITY, SampleOutcome::NonPositiveParams);
            }

            // Run inner loop warm-started from ML EBEs
            let sir_mu_k = compute_mu_k(model, &params_k.theta, options.mu_referencing);
            let (ehs, hms, _, _kappas) = run_inner_loop_warm(
                model,
                population,
                &params_k,
                inner_maxiter,
                inner_tol,
                Some(eta_hats),
                Some(&sir_mu_k),
                0, // SIR: no EBE convergence tracking
                0, // SIR: warm-started; no inner multi-start
            );

            // Compute OFV — through the method-aware seam, so an AGQ fit's SIR weights come
            // from the AGQ marginal it was actually optimised against, not the FOCE one.
            let nll_k = pop_nll_opts(model, population, &params_k, &ehs, &hms, &_kappas, options);
            let ofv_k = 2.0 * nll_k;
            if !ofv_k.is_finite() {
                return (f64::NEG_INFINITY, SampleOutcome::NonFiniteOfv);
            }

            let dofv = ofv_k - ofv_hat;

            // Log Student-t proposal density at x_k.
            // z holds the scaled standardised residual L^{-1}(x_k - x_hat), so
            // the quadratic form is z^T z (already in the scaled space).
            let quad_form: f64 = z.iter().map(|zi| zi * zi).sum();
            let log_q_k =
                log_norm - 0.5 * log_det_proposal - ((nu + d) / 2.0) * (1.0 + quad_form / nu).ln();

            // Importance weight: log w_k = -0.5 * dOFV_k - log_q_k + log_q_hat
            (-0.5 * dofv - log_q_k + log_q_hat, SampleOutcome::Accepted)
        })
        .unzip();

    // Step 2: Normalize weights using log-sum-exp trick
    let max_log_w = log_weights
        .iter()
        .cloned()
        .filter(|w| w.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    if max_log_w == f64::NEG_INFINITY {
        return Err(all_invalid_weights_message(
            &outcomes,
            &coord_names,
            &conditioned,
        ));
    }

    let weights: Vec<f64> = log_weights
        .iter()
        .map(|lw| (lw - max_log_w).exp())
        .collect();
    let sum_w: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();

    // Effective sample size — shared Kish ESS (`1/Σw̃²` with zero-guard). SIR's
    // log-sum-exp normalisation above stays local (its `Err`-on-all-invalid +
    // no-non-finite-filter contract differs from `stats::util::log_sum_exp_normalised`).
    let ess = crate::stats::util::ess_from_weights(&normalized_weights);

    if options.verbose {
        eprintln!("  SIR: effective sample size = {:.1}", ess);
    }

    // Step 3: Resample with replacement proportional to weights
    let weighted_dist = WeightedIndex::new(&weights)
        .map_err(|e| format!("Failed to build weighted sampler: {}", e))?;
    let resampled_indices: Vec<usize> = (0..n_resamples)
        .map(|_| weighted_dist.sample(&mut rng))
        .collect();

    // Step 4: Unpack resampled parameter vectors and compute CIs
    let n_theta = params.theta.len();
    let n_eta = params.omega.dim();
    let n_sigma = params.sigma.values.len();

    let mut theta_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_theta];
    let mut omega_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_eta];
    let mut sigma_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_sigma];

    for &idx in &resampled_indices {
        let p = unpack_params(&samples[idx], params);
        for (j, &th) in p.theta.iter().enumerate() {
            theta_samples[j].push(th);
        }
        for j in 0..n_eta {
            omega_samples[j].push(p.omega.matrix[(j, j)]);
        }
        for (j, &s) in p.sigma.values.iter().enumerate() {
            sigma_samples[j].push(s);
        }
    }

    let ci_theta: Vec<(f64, f64)> = theta_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_omega: Vec<(f64, f64)> = omega_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_sigma: Vec<(f64, f64)> = sigma_samples.iter().map(|s| percentile_ci(s)).collect();

    let resamples_packed = if options.sir_keep_samples {
        Some(
            resampled_indices
                .iter()
                .map(|&idx| samples[idx].clone())
                .collect(),
        )
    } else {
        None
    };

    Ok(SirResult {
        ci_theta,
        ci_omega,
        ci_sigma,
        effective_sample_size: ess,
        resamples_packed,
        warnings: conditioned.warnings(),
    })
}

/// Log-gamma function via the Lanczos approximation (g=7, n=9 coefficients).
/// Accurate to ~15 significant figures for x > 0.5.
fn lgamma(x: f64) -> f64 {
    // Lanczos coefficients (g=7)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let xm1 = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (xm1 + (i + 1) as f64);
    }
    let t = xm1 + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + sum.ln()
}

/// Compute 2.5th and 97.5th percentiles from a sample.
fn percentile_ci(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo_idx = ((n as f64) * 0.025).floor() as usize;
    let hi_idx = ((n as f64) * 0.975).ceil() as usize;
    let lo = sorted[lo_idx.min(n - 1)];
    let hi = sorted[hi_idx.min(n - 1)];
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("P{i}")).collect()
    }

    /// A well-conditioned block must come back untouched: no floor, no cap,
    /// and `L·Lᵀ` reproducing the input.
    #[test]
    fn condition_free_proposal_is_identity_on_a_healthy_block() {
        let cov = DMatrix::from_row_slice(2, 2, &[0.01, 0.002, 0.002, 0.04]);
        let c = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap();
        assert!(c.null_dirs.is_empty(), "{:?}", c.null_dirs);
        assert!(c.capped_dirs.is_empty(), "{:?}", c.capped_dirs);
        assert!(c.warnings().is_empty());
        let round = &c.chol * c.chol.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (round[(i, j)] - cov[(i, j)]).abs() < 1e-12,
                    "({i},{j}): {} vs {}",
                    round[(i, j)],
                    cov[(i, j)]
                );
            }
        }
        let expected_log_det = (0.01_f64 * 0.04 - 0.002 * 0.002).ln();
        assert!(
            (c.log_det - expected_log_det).abs() < 1e-12,
            "{}",
            c.log_det
        );
    }

    /// #1021: the covariance step floors non-identified Hessian eigenvalues
    /// before inverting, so their proposal variance comes back at ~1/floor.
    /// Sampling that direction unshrunk puts every draw outside the bounds.
    #[test]
    fn condition_free_proposal_caps_an_explosive_direction() {
        // Coordinate 0 carries variance 1e7 (sd ≈ 3162 in packed log-space);
        // its bound allows a proposal sd of at most 1.0.
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e7, 0.01]));
        let c = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap();
        assert_eq!(c.capped_dirs.len(), 1, "{:?}", c.capped_dirs);
        assert!(
            c.capped_dirs[0].contains("P0"),
            "the shrunk direction must name its parameter: {}",
            c.capped_dirs[0]
        );
        let round = &c.chol * c.chol.transpose();
        assert!(
            round[(0, 0)] <= 1.0 + 1e-9,
            "variance not capped: {}",
            round[(0, 0)]
        );
        // The identified direction is left alone.
        assert!((round[(1, 1)] - 0.01).abs() < 1e-12, "{}", round[(1, 1)]);
        assert!(c
            .warnings()
            .iter()
            .any(|w| w.contains("shrunk") && w.contains("P0")));
    }

    /// The near-null floor must be anchored on the *capped* spectrum. Anchored
    /// on the raw one, a single eigenvalue-floored direction (variance 1e8)
    /// would put the relative floor at ~1e-2 and inflate every genuinely
    /// well-determined direction to it — reporting real parameters as null and
    /// widening their CIs by orders of magnitude.
    #[test]
    fn condition_free_proposal_does_not_let_an_explosive_direction_swamp_the_floor() {
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e8, 1e-4]));
        let c = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap();
        assert_eq!(c.capped_dirs.len(), 1, "{:?}", c.capped_dirs);
        assert!(
            c.null_dirs.is_empty(),
            "a well-determined direction must not be reported null: {:?}",
            c.null_dirs
        );
        let round = &c.chol * c.chol.transpose();
        assert!(
            (round[(1, 1)] - 1e-4).abs() < 1e-12,
            "well-determined variance was altered: {}",
            round[(1, 1)]
        );
    }

    /// A likelihood ridge (two parameters determining only their sum) leaves an
    /// exactly-null direction after FIX-ed parameters are excluded. It must be
    /// floored — not rejected — and reported by name.
    #[test]
    fn condition_free_proposal_floors_and_names_a_null_direction() {
        // Eigenvectors (1,1)/√2 with λ=0.02 and (1,-1)/√2 with λ=0.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let v = DMatrix::from_row_slice(2, 2, &[h, h, h, -h]);
        let cov =
            &v * DMatrix::from_diagonal(&DVector::from_column_slice(&[0.02, 0.0])) * v.transpose();
        let c = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap();
        assert_eq!(c.null_dirs.len(), 1, "{:?}", c.null_dirs);
        assert!(
            c.null_dirs[0].contains("P0") && c.null_dirs[0].contains("P1"),
            "both ridge parameters must be named: {}",
            c.null_dirs[0]
        );
        // Still PD, so sampling works.
        let round = &c.chol * c.chol.transpose();
        assert!(round[(0, 0)] > 0.0 && round[(1, 1)] > 0.0);
        assert!(c.log_det.is_finite());
        assert!(c.warnings().iter().any(|w| w.contains("rank-deficient")));
    }

    /// Two null directions (the reported #1021 case: two FIX-like degeneracies)
    /// are just as survivable as one.
    #[test]
    fn condition_free_proposal_survives_two_null_directions() {
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[0.01, 0.0, 0.0]));
        let c = condition_free_proposal(&cov, &[1.0, 1.0, 1.0], &names(3)).unwrap();
        assert_eq!(c.null_dirs.len(), 2, "{:?}", c.null_dirs);
        assert!(c.chol.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn condition_free_proposal_rejects_a_covariance_with_no_positive_direction() {
        let cov = DMatrix::zeros(2, 2);
        let err = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap_err();
        assert!(err.contains("no positive eigenvalue"), "{err}");
    }

    #[test]
    fn condition_free_proposal_rejects_a_non_finite_covariance() {
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[f64::NAN, 0.01]));
        let err = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
    }

    /// #1021: the bare "All SIR samples had invalid weights" was a dead end.
    /// The replacement names the rejecting check, the coordinates whose bounds
    /// were overshot, and the proposal's rank deficiency.
    #[test]
    fn all_invalid_weights_message_reports_tally_and_offenders() {
        let outcomes = vec![
            SampleOutcome::OutOfBounds(1),
            SampleOutcome::OutOfBounds(1),
            SampleOutcome::OutOfBounds(0),
            SampleOutcome::NonFiniteOfv,
            SampleOutcome::NonPositiveParams,
        ];
        let coord_names = vec!["TVCL".to_string(), "PROP_ERR".to_string()];
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e7, 0.0]));
        let conditioned = condition_free_proposal(&cov, &[1.0, 1.0], &coord_names).unwrap();

        let msg = all_invalid_weights_message(&outcomes, &coord_names, &conditioned);
        assert!(msg.contains("All 5 SIR samples"), "{msg}");
        assert!(msg.contains("3 out of bounds"), "{msg}");
        assert!(msg.contains("1 non-positive"), "{msg}");
        assert!(msg.contains("1 non-finite OFV"), "{msg}");
        // Most-frequent offender first, with its hit count.
        assert!(msg.contains("PROP_ERR (2)"), "{msg}");
        assert!(msg.contains("TVCL (1)"), "{msg}");
        // Proposal diagnosis + what to do about it.
        assert!(msg.contains("rank-deficient"), "{msg}");
        assert!(msg.contains("Fix or drop one parameter"), "{msg}");
    }

    /// A clean run must not decorate the failure message with proposal
    /// diagnostics it doesn't have.
    #[test]
    fn all_invalid_weights_message_stays_bare_for_a_healthy_proposal() {
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[0.01, 0.04]));
        let conditioned = condition_free_proposal(&cov, &[1.0, 1.0], &names(2)).unwrap();
        let msg =
            all_invalid_weights_message(&[SampleOutcome::NonFiniteOfv], &names(2), &conditioned);
        assert!(msg.contains("1 non-finite OFV"), "{msg}");
        assert!(!msg.contains("rank-deficient"), "{msg}");
        assert!(!msg.contains("Fix or drop"), "{msg}");
    }

    /// #1037: the cap must be driven by the room the estimate actually has, not
    /// by the width of its box. A theta declared `(0, 100)` estimated at `0.5`
    /// packs to `[-23.0, 4.6]` around `x̂ = -0.69`: 5.3 above, 22.3 below. A
    /// width-derived cap would allow ±13.8 and still put most draws over the
    /// upper bound.
    #[test]
    fn proposal_sd_caps_use_the_nearer_bound() {
        let x_hat = vec![(0.5_f64).ln()];
        let bounds = PackedBounds {
            lower: vec![(1e-10_f64).ln()],
            upper: vec![(100.0_f64).ln()],
        };
        // 30 df: t inflation sqrt(30/28) = 1.035, small enough to read the
        // geometry through.
        let caps = proposal_sd_caps(&x_hat, &bounds, &[0], 30.0);
        let room = bounds.upper[0] - x_hat[0];
        // ±2 sd of the *realised* t draws must fit in the room above.
        let realised_sd = caps[0] * (30.0_f64 / 28.0).sqrt();
        assert!(
            2.0 * realised_sd <= room + 1e-9,
            "cap {} overshoots the {} of room above x_hat",
            caps[0],
            room
        );
        // And it must be materially tighter than the old width-derived cap.
        let width_cap = (bounds.upper[0] - bounds.lower[0]) / PROPOSAL_BOUND_SIGMAS;
        assert!(caps[0] < 0.5 * width_cap, "{} vs {}", caps[0], width_cap);
    }

    /// A centred estimate must be unaffected by the switch from box width to
    /// nearer-bound room — the two agree exactly there (before the t correction).
    #[test]
    fn proposal_sd_caps_match_the_box_half_width_when_centred() {
        let bounds = PackedBounds {
            lower: vec![-6.0],
            upper: vec![6.0],
        };
        // 1e12 df ⇒ t inflation ≈ 1, isolating the geometry.
        let caps = proposal_sd_caps(&[0.0], &bounds, &[0], 1e12);
        let width_cap = (bounds.upper[0] - bounds.lower[0]) / PROPOSAL_BOUND_SIGMAS;
        assert!(
            (caps[0] - width_cap).abs() < 1e-6,
            "{} vs {}",
            caps[0],
            width_cap
        );
    }

    /// The Student-t proposal draws are `sqrt(nu/(nu-2))` wider than the
    /// Cholesky scale; the cap compensates so "±2 sd inside the room" is a
    /// statement about the draws, not about the scale matrix.
    #[test]
    fn proposal_sd_caps_compensate_the_student_t_inflation() {
        let bounds = PackedBounds {
            lower: vec![-4.0],
            upper: vec![4.0],
        };
        let wide = proposal_sd_caps(&[0.0], &bounds, &[0], 1e12);
        let heavy = proposal_sd_caps(&[0.0], &bounds, &[0], 5.0);
        let ratio = wide[0] / heavy[0];
        assert!(
            (ratio - (5.0_f64 / 3.0).sqrt()).abs() < 1e-6,
            "t inflation not applied: {ratio}"
        );
        // Degrees of freedom at or below 2 have no finite variance; fall back to
        // the raw scale rather than dividing by a NaN.
        let df2 = proposal_sd_caps(&[0.0], &bounds, &[0], 2.0);
        assert!(df2[0].is_finite() && df2[0] > 0.0, "{}", df2[0]);
    }

    /// An estimate pinned exactly on its bound has zero room. Capping every
    /// direction that loads on it to zero variance would abort SIR with "no
    /// positive eigenvalue"; a sliver of room keeps the proposal PD.
    #[test]
    fn proposal_sd_caps_floor_the_room_for_an_estimate_on_its_bound() {
        let bounds = PackedBounds {
            lower: vec![-6.0],
            upper: vec![6.0],
        };
        let caps = proposal_sd_caps(&[6.0], &bounds, &[0], 5.0);
        assert!(caps[0] > 0.0, "a zero cap would abort SIR: {}", caps[0]);
        let expected =
            12.0 * PROPOSAL_MIN_ROOM_FRAC * 2.0 / PROPOSAL_BOUND_SIGMAS / (5.0_f64 / 3.0).sqrt();
        assert!((caps[0] - expected).abs() < 1e-12, "{}", caps[0]);
        // A proposal built on it is still PD, not an error.
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e7]));
        let c = condition_free_proposal(&cov, &caps, &names(1)).unwrap();
        assert!(c.chol[(0, 0)] > 0.0);
    }

    /// Only free coordinates get a cap, and they line up with `free_idx`.
    #[test]
    fn proposal_sd_caps_follow_the_free_index_map() {
        let bounds = PackedBounds {
            lower: vec![-6.0, -2.0, -8.0],
            upper: vec![6.0, 2.0, 5.0],
        };
        let caps = proposal_sd_caps(&[0.0, 0.0, 0.0], &bounds, &[0, 2], 1e12);
        assert_eq!(caps.len(), 2);
        assert!((caps[0] - 3.0).abs() < 1e-6, "{}", caps[0]);
        // Coordinate 2 is off-centre in [-8, 5]: nearer bound is 5.
        assert!((caps[1] - 2.5).abs() < 1e-6, "{}", caps[1]);
    }

    /// #1037: the packed bounds are narrow for the variance components (an omega
    /// log-Cholesky diagonal lives in `[-6, 6]`), so a *legitimately* imprecise
    /// omega exceeds its cap without being an eigenvalue-floored artifact.
    /// Shrinking it would narrow its CI and mislabel it as non-identified.
    #[test]
    fn condition_free_proposal_leaves_a_legitimately_wide_direction_alone() {
        // sd 3.0 against a cap of 1.5: over the cap, but nowhere near the
        // 1e3–1e4 overshoot the cap exists for.
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[9.0, 0.01]));
        let c = condition_free_proposal(&cov, &[1.5, 1.5], &names(2)).unwrap();
        assert!(
            c.capped_dirs.is_empty(),
            "a merely imprecise direction must not be shrunk: {:?}",
            c.capped_dirs
        );
        let round = &c.chol * c.chol.transpose();
        assert!((round[(0, 0)] - 9.0).abs() < 1e-9, "{}", round[(0, 0)]);
    }

    /// The gate is a factor, not a switch: past `CAP_TRIGGER_FACTOR` sd the
    /// direction is still shrunk all the way back to the cap.
    #[test]
    fn condition_free_proposal_still_caps_past_the_trigger() {
        let cap_sd = 1.5_f64;
        let over = (CAP_TRIGGER_FACTOR + 1.0) * cap_sd;
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[over * over, 0.01]));
        let c = condition_free_proposal(&cov, &[cap_sd, cap_sd], &names(2)).unwrap();
        assert_eq!(c.capped_dirs.len(), 1, "{:?}", c.capped_dirs);
        let round = &c.chol * c.chol.transpose();
        assert!(
            (round[(0, 0)] - cap_sd * cap_sd).abs() < 1e-9,
            "not shrunk to the cap: {}",
            round[(0, 0)]
        );
    }

    /// #1037: a healthy block must come back *bit*-identical, not merely close.
    /// Round-tripping through `V Λ Vᵀ` perturbs it at ~1e-16 relative, which
    /// changes every drawn sample and so the reported CIs.
    #[test]
    fn condition_free_proposal_is_bit_identical_on_a_healthy_block() {
        let cov = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.013, 0.0021, -0.0007, 0.0021, 0.041, 0.0033, -0.0007, 0.0033, 0.0089,
            ],
        );
        let c = condition_free_proposal(&cov, &[1.0, 1.0, 1.0], &names(3)).unwrap();
        let expected = cov.clone().cholesky().expect("input is PD").l();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    c.chol[(i, j)].to_bits(),
                    expected[(i, j)].to_bits(),
                    "({i},{j}) drifted: {} vs {}",
                    c.chol[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    /// #1037: a degenerate direction spread thinly over many coordinates has no
    /// loading above `DIRECTION_LOADING_MIN`. "No dominant parameter" leaves the
    /// user nothing to act on, so name the largest loadings anyway.
    #[test]
    fn describe_direction_falls_back_when_no_loading_dominates() {
        // 100 coordinates, each loading 0.1 — well under the 0.15 threshold.
        let n = 100;
        let mut v = DMatrix::zeros(n, n);
        for k in 0..n {
            v[(k, 0)] = 1.0 / (n as f64).sqrt();
        }
        let d = describe_direction(&v, 0, &names(n));
        assert!(!d.contains("no dominant parameter"), "{d}");
        assert_eq!(d.matches(',').count(), 2, "expected three loadings: {d}");
        assert!(d.contains("P0"), "{d}");
    }

    /// The fallback must not fire when a direction *does* have dominant
    /// loadings — reporting the noise floor alongside them would be worse.
    #[test]
    fn describe_direction_reports_only_dominant_loadings_when_present() {
        let mut v = DMatrix::zeros(3, 3);
        v[(0, 0)] = 0.99;
        v[(1, 0)] = 0.1;
        v[(2, 0)] = 0.05;
        let d = describe_direction(&v, 0, &names(3));
        assert!(d.contains("P0"), "{d}");
        assert!(!d.contains("P1"), "{d}");
        assert!(!d.contains("P2"), "{d}");
    }

    #[test]
    fn test_percentile_ci_sorted() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let (lo, hi) = percentile_ci(&values);
        assert!(lo >= 0.02 && lo <= 0.03, "lo={}", lo);
        assert!(hi >= 0.97 && hi <= 0.98, "hi={}", hi);
    }

    #[test]
    fn test_percentile_ci_single() {
        let (lo, hi) = percentile_ci(&[5.0]);
        assert_eq!(lo, 5.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn test_percentile_ci_empty() {
        let (lo, hi) = percentile_ci(&[]);
        assert!(lo.is_nan());
        assert!(hi.is_nan());
    }

    #[test]
    fn test_lgamma_known_values() {
        // lgamma(1) = 0, lgamma(2) = 0, lgamma(0.5) = ln(sqrt(pi))
        assert!((lgamma(1.0)).abs() < 1e-12);
        assert!((lgamma(2.0)).abs() < 1e-12);
        let expected_half = (std::f64::consts::PI.sqrt()).ln();
        assert!(
            (lgamma(0.5) - expected_half).abs() < 1e-10,
            "lgamma(0.5)={}",
            lgamma(0.5)
        );
        // lgamma(5) = ln(4!) = ln(24)
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
    }

    /// Student-t log-density at the centre must equal log_q_hat (quadratic form = 0).
    #[test]
    fn test_student_t_density_at_centre() {
        let nu = 5.0_f64;
        let d = 3.0_f64;
        let log_det = 0.0_f64; // identity covariance
        let log_q_hat = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * log_det;
        // At centre, quad_form = 0, so log_q_k should equal log_q_hat
        let quad_form = 0.0_f64;
        let log_q_k = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * log_det
            - ((nu + d) / 2.0) * (1.0 + quad_form / nu).ln();
        assert!((log_q_k - log_q_hat).abs() < 1e-12);
    }

    /// Large nu should recover near-normal proposal (lgamma ratio converges).
    #[test]
    fn test_large_nu_approaches_normal() {
        // For nu=1000, d=2, the Student-t log-density should be very close
        // to the MVN log-density at the same quadratic form.
        let nu = 1000.0_f64;
        let d = 2.0_f64;
        let log_det = 0.5_f64;
        let quad_form = 1.5_f64;

        let log_t = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * log_det
            - ((nu + d) / 2.0) * (1.0 + quad_form / nu).ln();

        let log_mvn = -0.5 * (d * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);

        assert!(
            (log_t - log_mvn).abs() < 0.01,
            "Student-t (nu=1000) vs MVN: diff = {:.4e}",
            (log_t - log_mvn).abs()
        );
    }
}
