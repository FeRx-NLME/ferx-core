//! Sampling Importance Resampling (SIR) for parameter uncertainty estimation.
//!
//! Implements the SIR procedure described in Dosne et al. (2017):
//! "Improving the estimation of parameter uncertainty distributions in
//! nonlinear mixed effects models using sampling importance resampling"
//!
//! SIR provides a non-parametric estimate of parameter uncertainty that is
//! more robust than the asymptotic covariance matrix.

use crate::estimation::inner_optimizer::{run_inner_loop_warm_seeded, InnerHessianSeed};
use crate::estimation::outer_optimizer::pop_nll_opts;
use crate::estimation::parameterization::{
    compute_mu_k, coordinate_names, coordinate_values, pack_with_bounds, packed_segments,
    unpack_params, PackedBounds, PackedStart,
};
use crate::estimation::uncertainty_samples::{
    admissible_values, bounds_to_draw_scale, from_draw_scale, log_abs_jacobian,
    log_abs_jacobian_natural, logit_theta_coords, to_draw_scale, LogitThetaCoord,
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
    /// 95% CI for each IOV kappa variance (the `omega_iov` diagonal, packed
    /// order). Empty when the model declares no kappa (#1705).
    pub ci_kappa: Vec<(f64, f64)>,
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

impl SirResult {
    /// `ci_kappa` as `FitResult.sir_ci_kappa` carries it: `None` for a model
    /// with no kappa, so a non-IOV fit's writers and `.fitrx` are unchanged
    /// (#1705). The one conversion both the in-fit and the standalone
    /// (`run_sir`) path use.
    pub(crate) fn kappa_ci(&self) -> Option<Vec<(f64, f64)>> {
        (!self.ci_kappa.is_empty()).then(|| self.ci_kappa.clone())
    }
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

/// The effective sample size below which SIR's intervals carry a warning
/// (#1723) — the threshold `docs/estimation/sir.qmd` already names as a poor
/// proposal. Measured margin: healthy fixtures at 1000 draws sit at 143 (the
/// MBMA placebo shape), 293, 405 (`warfarin_iov` FOCE, since #1722) and 497;
/// the degenerate `warfarin_iov` FOCEI run at 3.5.
pub(crate) const SIR_LOW_ESS: f64 = 100.0;

/// χ²₁(0.95). A free variance whose **conditional** ΔOFV at its packed lower
/// bound is below this has zero inside its likelihood-ratio interval: the
/// conditional ΔOFV is an upper bound on the profile one. The converse does not
/// hold, so a coordinate above it is never described as bounded away from zero.
const CHI2_1_95: f64 = 3.841_458_820_694_124;

/// The draw the SIR intervals lean on most, described for the low-ESS warning.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HeaviestDraw {
    /// Its share of the normalised importance weight, in `[0, 1]`.
    pub share: f64,
    /// The coordinate it moves furthest, in proposal standard deviations.
    pub name: String,
    /// That move, in marginal proposal standard deviations (signed).
    pub sd_units: f64,
    /// That coordinate's value at the draw, on the reported scale.
    pub value: f64,
}

/// One free Ω / κ diagonal moved alone to its packed lower bound (#1723).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FloorProbe {
    pub name: String,
    /// OFV there minus the OFV at the estimates, EBEs re-solved.
    pub dofv: f64,
    /// The variance there, on the reported scale.
    pub variance: f64,
    /// The probe point also set non-zero block covariances of this η to 0
    /// ([`FloorProbeCoord::zeroes_a_covariance`]); the warning says so.
    pub covariances_zeroed: bool,
}

/// The low-ESS warning (#1723), or `None` when `ess >= SIR_LOW_ESS`.
///
/// `probe` runs the floor probe and is called **only** below the threshold
/// and only under [`SirScale::Packed`] — a healthy run spends nothing on it,
/// and under `Natural` the target has no box-dependent shelf to report.
pub(crate) fn low_ess_warning(
    ess: f64,
    n_samples: usize,
    scale: SirScale,
    heaviest: &HeaviestDraw,
    probe: impl FnOnce() -> Vec<FloorProbe>,
) -> Option<String> {
    if ess >= SIR_LOW_ESS {
        return None;
    }
    // Rounded down, so an ESS just under the threshold never prints as "100.0".
    let shown = (ess * 10.0).floor() / 10.0;
    let mut msg = if (n_samples as f64) < SIR_LOW_ESS {
        format!(
            "effective sample size is {shown:.1} of {n_samples} draws; with fewer than \
             {SIR_LOW_ESS:.0} draws it cannot reach the {SIR_LOW_ESS:.0} at which the proposal \
             is adequate, so these intervals rest on few draws."
        )
    } else {
        format!(
            "effective sample size is {shown:.1} of {n_samples} draws, below the \
             {SIR_LOW_ESS:.0} at which the proposal is adequate, so these intervals rest on \
             few draws."
        )
    };
    msg.push_str(&format!(
        " The heaviest draw carries {:.1}% of the weight; it moves {} by {:+.2} proposal \
         standard deviations from the proposal centre, to {:.3e} on the reported scale.",
        100.0 * heaviest.share,
        heaviest.name,
        heaviest.sd_units,
        heaviest.value
    ));
    msg.push_str(" Increase `sir_samples` for more stable intervals.");
    match scale {
        SirScale::Packed => {
            let flagged: Vec<FloorProbe> =
                probe().into_iter().filter(|p| p.dofv < CHI2_1_95).collect();
            if !flagged.is_empty() {
                let listed = flagged
                    .iter()
                    .map(|p| {
                        format!(
                            "{} (ΔOFV {:.2} at variance {:.2e}{})",
                            p.name,
                            p.dofv,
                            p.variance,
                            if p.covariances_zeroed {
                                ", with its covariances in the block also set to 0"
                            } else {
                                ""
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                msg.push_str(&format!(
                    " The data do not bound {listed} away from zero: each, moved to its \
                     lower bound in the parameter box with the other parameters at the \
                     estimates, costs less than χ²₁(0.95) = 3.84 in \
                     OFV, so its SIR lower limit reflects the parameter box rather than the \
                     data."
                ));
                msg.push_str(
                    " `sir_scale = natural` makes these lower limits independent of the box.",
                );
            }
        }
        SirScale::Natural => {
            msg.push_str(
                " Under `sir_scale = natural` a variance informed by few groups has a heavy \
                 upper tail; `sir_scale = packed` may sample it better.",
            );
        }
    }
    Some(msg)
}

/// One variance the floor probe moves: the packed index of its Cholesky
/// diagonal `ln L_ii`, and those of the off-diagonals `L_ik` (k < i) in the
/// same row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FloorProbeCoord {
    pub diag: usize,
    pub row_off: Vec<usize>,
}

/// The free Ω / Ω_IOV variances the floor probe moves — never a FIX one, a θ,
/// a σ or a `[mixture]` override. In a `block_omega` / block κ an η's variance
/// is `Σ_ii = Σ_k L_ik²`, so flooring `ln L_ii` alone leaves `Σ_ii ≈ Σ_{k<i}
/// L_ik²` and only drives the η's correlation towards ±1; the probe point
/// therefore also zeroes the row's off-diagonals ([`floor_probe_point`]).
pub(crate) fn floor_probe_coords(params: &ModelParameters, fixed: &[bool]) -> Vec<FloorProbeCoord> {
    let seg = packed_segments(params);
    let mut out = Vec::new();
    let mut block = |start: usize, om: &OmegaMatrix| {
        let mut rows: Vec<FloorProbeCoord> = (0..om.dim())
            .map(|_| FloorProbeCoord {
                diag: usize::MAX,
                row_off: Vec::new(),
            })
            .collect();
        for (off, (r, c)) in
            crate::estimation::parameterization::lower_tri_iter(om.dim(), om.diagonal).enumerate()
        {
            if r == c {
                rows[r].diag = start + off;
            } else {
                rows[r].row_off.push(start + off);
            }
        }
        out.extend(rows.into_iter().filter(|c| !fixed[c.diag]));
    };
    block(seg.omega_start(), &params.omega);
    if let Some(ref iov) = params.omega_iov {
        block(seg.iov_start(), iov);
    }
    out
}

impl FloorProbeCoord {
    /// Whether [`floor_probe_point`] changes a covariance as well as the
    /// variance: some off-diagonal of the row is non-zero at `x_hat`. False for
    /// a diagonal η, a block's first η, and a row of structural zeros.
    pub(crate) fn zeroes_a_covariance(&self, x_hat: &[f64]) -> bool {
        self.row_off.iter().any(|&j| x_hat[j] != 0.0)
    }
}

/// The packed point a floor probe scores: the estimate with the variance's
/// Cholesky diagonal at its own packed lower bound and its row's off-diagonals
/// at 0, so `Σ_ii = e^{2·lower}`. Any point on that constraint is a valid
/// upper bound for the profile ΔOFV there. A held non-zero off-diagonal is set
/// to 0 too; the bounds screen then rejects the point and the variance is not
/// flagged — the conservative outcome.
pub(crate) fn floor_probe_point(
    x_hat: &[f64],
    bounds: &PackedBounds,
    c: &FloorProbeCoord,
) -> Vec<f64> {
    let mut x = x_hat.to_vec();
    x[c.diag] = bounds.lower[c.diag];
    for &j in &c.row_off {
        x[j] = 0.0;
    }
    x
}

/// The proposal centre under [`SirScale::Natural`], on the draw scale: the
/// estimate moved to the Laplace mode of the tilted target, `ŷ + C ∇ log|J|`,
/// with `C` the free-block proposal covariance and `∇ log|J|` the gradient of
/// [`log_abs_jacobian_natural`] there; clamped into the draw-scale box.
pub(crate) fn natural_centre(
    y_hat: &[f64],
    cov_free: &DMatrix<f64>,
    grad_free: &DVector<f64>,
    free_idx: &[usize],
    draw_bounds: &PackedBounds,
) -> Vec<f64> {
    let shift = cov_free * grad_free;
    let mut c = y_hat.to_vec();
    for (a, &i) in free_idx.iter().enumerate() {
        c[i] = (y_hat[i] + shift[a]).clamp(draw_bounds.lower[i], draw_bounds.upper[i]);
    }
    c
}

/// `∇_y log_abs_jacobian_natural` over the free coordinates, on the draw scale,
/// by central differences of the one implementation (a logit θ goes through
/// its `y → x` map, so the chain factor comes for free).
fn natural_log_jac_gradient(
    y_hat: &[f64],
    free_idx: &[usize],
    logit_coords: &[LogitThetaCoord],
    params: &ModelParameters,
    fixed: &[bool],
) -> DVector<f64> {
    let at = |y: &[f64]| {
        let mut x = y.to_vec();
        from_draw_scale(&mut x, logit_coords);
        log_abs_jacobian_natural(&x, params, fixed)
    };
    let h = 1e-5;
    DVector::from_iterator(
        free_idx.len(),
        free_idx.iter().map(|&i| {
            let mut yp = y_hat.to_vec();
            let mut ym = y_hat.to_vec();
            yp[i] += h;
            ym[i] -= h;
            (at(&yp) - at(&ym)) / (2.0 * h)
        }),
    )
}

/// Describe the draw with the largest normalised weight: its share and the free
/// coordinate it moves furthest in marginal proposal standard deviations.
fn heaviest_draw(
    normalized_weights: &[f64],
    z_vectors: &[Vec<f64>],
    samples: &[Vec<f64>],
    proposal_chol: &DMatrix<f64>,
    free_idx: &[usize],
    free_names: &[String],
    params: &ModelParameters,
) -> HeaviestDraw {
    let (k, &share) =
        normalized_weights
            .iter()
            .enumerate()
            .fold((0, &f64::NEG_INFINITY), |best, (i, w)| {
                if *w > *best.1 {
                    (i, w)
                } else {
                    best
                }
            });
    let delta = proposal_chol * DVector::from_column_slice(&z_vectors[k]);
    let (a, t) = (0..free_idx.len())
        .map(|a| (a, delta[a] / proposal_chol.row(a).norm()))
        .fold((0, 0.0f64), |best, (a, t)| {
            if t.abs() > best.1.abs() {
                (a, t)
            } else {
                best
            }
        });
    let values = coordinate_values(&unpack_params(&samples[k], params));
    HeaviestDraw {
        share,
        name: free_names[a].clone(),
        sd_units: t,
        value: values[free_idx[a]],
    }
}

/// Why a proposal sample contributed no weight. Tallied so a run in which
/// *every* sample is rejected can say which check did the rejecting (#1021).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleOutcome {
    Accepted,
    /// Packed coordinate index that fell outside its bound.
    OutOfBounds(usize),
    /// In bounds, but failed `admissible_values`: a non-finite θ, or a
    /// non-positive σ / Ω / κ variance.
    InadmissibleValues,
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
    let mut n_inadmissible = 0usize;
    let mut n_ofv = 0usize;
    let mut n_cancelled = 0usize;
    let mut per_coord: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for o in outcomes {
        match *o {
            SampleOutcome::OutOfBounds(i) => {
                n_bounds += 1;
                *per_coord.entry(i).or_insert(0) += 1;
            }
            SampleOutcome::InadmissibleValues => n_inadmissible += 1,
            SampleOutcome::NonFiniteOfv => n_ofv += 1,
            SampleOutcome::Cancelled => n_cancelled += 1,
            SampleOutcome::Accepted => {}
        }
    }
    let mut msg = format!(
        "All {} SIR samples had invalid weights (rejected: {} out of bounds, {} \
         non-finite theta or non-positive sigma/omega/kappa variance, {} non-finite OFV, \
         {} cancelled).",
        outcomes.len(),
        n_bounds,
        n_inadmissible,
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

/// Screen one packed draw before any inner-loop work is spent on it: the
/// packed bounds first, then `admissible_values` — finite θ, positive
/// variances, no sign condition on θ (#1701). The first out-of-bounds
/// coordinate is recorded so a total rejection can name it (#1021).
fn screen_draw(
    x_k: &[f64],
    bounds: &PackedBounds,
    template: &ModelParameters,
) -> Result<ModelParameters, SampleOutcome> {
    let out_of_bounds = x_k
        .iter()
        .zip(bounds.lower.iter().zip(bounds.upper.iter()))
        .position(|(&x, (&lo, &hi))| x < lo || x > hi);
    if let Some(i) = out_of_bounds {
        return Err(SampleOutcome::OutOfBounds(i));
    }
    let params_k = unpack_params(x_k, template);
    if !admissible_values(&params_k) {
        return Err(SampleOutcome::InadmissibleValues);
    }
    Ok(params_k)
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
/// * `ofv_hat` - the **data** OFV (−2 log L) at the estimates, i.e. what every
///   optimizer reports as `OuterResult::ofv`. Under parameter priors (#254) the
///   penalty is added internally, to this baseline and to every draw alike, so
///   callers pass the clean value and cannot get the two halves out of step.
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
    // #1212: the math kernel is public API in its own right, so it opens the fit-scoped ODE
    // scope too rather than relying on its caller having done so. Inside `fit()` the current
    // worker already carries that override, which the scope detects without leasing a second
    // pool; a direct caller gets the tolerances it passed instead of the spec's parse-time ones.
    crate::api::with_fit_ode_scope(options, || {
        run_sir_core_scoped(
            model,
            population,
            params,
            eta_hats,
            proposal_cov,
            ofv_hat,
            options,
        )
    })?
}

#[allow(clippy::too_many_arguments)]
fn run_sir_core_scoped(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    proposal_cov: &DMatrix<f64>,
    ofv_hat: f64,
    options: &FitOptions,
) -> Result<SirResult, String> {
    run_sir_in_box(
        model,
        population,
        params,
        eta_hats,
        proposal_cov,
        ofv_hat,
        options,
        |_| {},
    )
}

/// [`run_sir_core_scoped`] with a hook on the packed box before anything reads
/// it. Production passes a no-op; the #1723 box-dependence test moves the
/// variance floors with it, which no public input can do.
#[allow(clippy::too_many_arguments)]
fn run_sir_in_box(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    proposal_cov: &DMatrix<f64>,
    ofv_hat: f64,
    options: &FitOptions,
    adjust_box: impl FnOnce(&mut PackedBounds),
) -> Result<SirResult, String> {
    let n_samples = options.sir_samples;
    let n_resamples = options.sir_resamples;

    // `sir_draw_ofv` scores a `[mixture]` model with the K-class marginal, which needs
    // the per-class Ω/Σ (#1755); and a non-mixture model has no class to score. A
    // mismatch is a caller error, refused here rather than panicking in `mixture_ofv`.
    match (model.mixture.is_some(), params.mixture.is_some()) {
        (true, false) => {
            return Err(
                "run_sir_core: the model declares [mixture] but the parameters carry \
                 no per-class Ω/Σ (ModelParameters::mixture is None); build them with \
                 fitted_params_from_result or pass the fit's own parameters."
                    .to_string(),
            )
        }
        (false, true) => {
            return Err("run_sir_core: the parameters carry per-class Ω/Σ \
                 (ModelParameters::mixture) but the model declares no [mixture]."
                .to_string())
        }
        _ => {}
    }

    if n_resamples > n_samples {
        return Err("sir_resamples must be <= sir_samples".to_string());
    }

    // Pack ML estimates as the proposal center. The FIX mask and the box come
    // out of the same walk (#1252) — both are consulted below, and building the
    // box requires the packed vector anyway.
    let PackedStart {
        packed: x_hat,
        mut bounds,
        fixed: fixed_mask,
        // #1307's pack-move list is not this caller's object.
        moves: _,
    } = pack_with_bounds(params);
    adjust_box(&mut bounds);
    let n_packed = x_hat.len();

    // Parameter priors (#254). SIR approximates the posterior the fit targeted,
    // so under a prior the target is `L(data) · p(θ)` and the importance weight
    // must score both halves. Without this the proposal is centred on the MAP
    // estimate and shaped by the *penalized* curvature, while the weights score
    // the *unpenalized* likelihood — so the resampling actively pulls the
    // intervals back toward the MLE and a tight prior vanishes from the reported
    // CIs, which is worse than not supporting SIR at all.
    //
    // `ofv_hat` arrives as the data −2 log L (what every caller has to hand), so
    // the penalty is added here on **both** sides rather than being the caller's
    // job — one derivation, so the baseline and the samples cannot disagree
    // about whether the prior is in.
    let priors = crate::estimation::outer_optimizer::build_prior_set(model, params);
    let ofv_hat = ofv_hat + priors.penalty(&x_hat);
    // #1723: a declared prior is a density on the packed scale, so the target is
    // already `L · p(x)` with no flat prior left to move to the reported scale;
    // the natural Jacobian would tilt the prior the fit was estimated under.
    if options.sir_scale == SirScale::Natural && priors.is_active() {
        return Err(
            "sir_scale = natural is not defined for a model with prior(...): the SIR \
             target is already the likelihood times the declared prior on the packed scale, \
             so there is no flat prior to move to the reported scale. Use sir_scale = packed."
                .to_string(),
        );
    }

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
    let free_idx: Vec<usize> = (0..n_packed).filter(|&i| !fixed_mask[i]).collect();
    let n_free = free_idx.len();
    if n_free == 0 {
        return Err("run_sir_core: every packed parameter is FIX — nothing to sample.".to_string());
    }

    // A `LogitProbability` θ is proposed on the logit scale (#1548): a proposal
    // on its packed `ln θ` scale has no ceiling at 1, and when the declared
    // upper bound is above 1 a draw there is scored with the clamped-logit
    // likelihood and can be resampled. The proposal lives on the draw scale
    // `y`; the samples are stored, bounds-checked and scored on the packed
    // scale `x`, and the weights carry `|dx/dy|` so the target is unchanged.
    let logit_coords = logit_theta_coords(params, &model.theta_transform, &fixed_mask)?;
    let draw_bounds = bounds_to_draw_scale(&bounds, &logit_coords);
    let log_jac_hat = log_abs_jacobian(&x_hat, &logit_coords);

    // Symmetrize first, then extract the free block (rows/cols of non-FIX
    // indices) before Cholesky.
    let (y_hat, proposal_cov_y) = to_draw_scale(&x_hat, proposal_cov, &logit_coords);
    let sym_cov_full = (&proposal_cov_y + proposal_cov_y.transpose()) * 0.5;
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
    let sd_caps = proposal_sd_caps(&y_hat, &draw_bounds, &free_idx, options.sir_df);
    let conditioned = condition_free_proposal(&sub_cov, &sd_caps, &free_names)?;
    if options.verbose {
        for w in conditioned.warnings() {
            eprintln!("  SIR: {w}");
        }
    }
    let proposal_chol = conditioned.chol.clone();

    // `sir_scale = natural` (#1723): the target gains `log|∂n/∂x|`, and the
    // proposal is re-centred on that tilted target's Laplace mode so it does
    // not have to reach for it from the packed estimate. Under `Packed` the
    // centre is `ŷ` itself and nothing below differs from before #1723.
    let natural = options.sir_scale == SirScale::Natural;
    let centre = if natural {
        let grad = natural_log_jac_gradient(&y_hat, &free_idx, &logit_coords, params, &fixed_mask);
        let cov_free = &proposal_chol * proposal_chol.transpose();
        natural_centre(&y_hat, &cov_free, &grad, &free_idx, &draw_bounds)
    } else {
        y_hat.clone()
    };
    let log_jac_nat_hat = if natural {
        log_abs_jacobian_natural(&x_hat, params, &fixed_mask)
    } else {
        0.0
    };

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
        // Build the full packed sample: free indices get centre + delta_free,
        // fixed indices stay pinned at x_hat (so the strict bounds check
        // `lower == upper == x_hat[i]` passes; a fixed index is never a logit
        // coordinate, so `y_hat[i] == x_hat[i]` there, and the natural
        // re-centre moves free indices only).
        let mut x_k = centre.clone();
        for (a, &i) in free_idx.iter().enumerate() {
            x_k[i] += delta_free[a];
        }
        from_draw_scale(&mut x_k, &logit_coords);
        samples.push(x_k);
        // store L_free⁻¹(delta_free) = z_free * scale for the quadratic form
        // in log_q_k. Length = n_free.
        z_vectors.push(z_free.into_iter().map(|zi| zi * scale).collect());
    }

    // Step 2: Evaluate importance weights in parallel (warm-started inner loop).
    // OFV (prior included) at one admissible packed point. Shared by the draws and the
    // low-ESS floor probe, so the probe's ΔOFV is the very quantity the weights score.
    let ofv_at = |params_k: &ModelParameters, x_k: &[f64]| -> f64 {
        // The prior half, at this draw. `x_k` is already the packed vector the penalty
        // is defined on. A no-op (`+ 0.0`) for an unpriored fit, so those weights stay
        // bit-identical.
        sir_draw_ofv(model, population, params_k, eta_hats, options) + priors.penalty(x_k)
    };

    let (log_weights, outcomes): (Vec<f64>, Vec<SampleOutcome>) = samples
        .par_iter()
        .zip(z_vectors.par_iter())
        .map(|(x_k, z)| {
            if crate::cancel::is_cancelled(&options.cancel) {
                return (f64::NEG_INFINITY, SampleOutcome::Cancelled);
            }
            let params_k = match screen_draw(x_k, &bounds, params) {
                Ok(p) => p,
                Err(outcome) => return (f64::NEG_INFINITY, outcome),
            };

            let ofv_k = ofv_at(&params_k, x_k);
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

            // The proposal density is on the draw scale, the target on the
            // packed scale: `π_y(y) = π_x(x(y)) |dx/dy|`. Taken relative to the
            // centre, like `log_q_hat`; exactly 0 with no logit coordinate.
            let mut log_jac = log_abs_jacobian(x_k, &logit_coords) - log_jac_hat;
            if natural {
                log_jac += log_abs_jacobian_natural(x_k, params, &fixed_mask) - log_jac_nat_hat;
            }

            // Importance weight: log w_k = -0.5 * dOFV_k + log|dx/dy| - log_q_k + log_q_hat
            (
                -0.5 * dofv + log_jac - log_q_k + log_q_hat,
                SampleOutcome::Accepted,
            )
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

    // #1723: below the docs' own adequacy threshold the intervals are not
    // returned silently. The floor probe runs only here, and only under the
    // packed target, so a healthy run costs nothing extra.
    let mut warnings = conditioned.warnings();
    if ess < SIR_LOW_ESS {
        let heaviest = heaviest_draw(
            &normalized_weights,
            &z_vectors,
            &samples,
            &proposal_chol,
            &free_idx,
            &free_names,
            params,
        );
        let probe = || {
            // The baseline is `ofv_at` at the estimate, not the caller's
            // `ofv_hat`: an offset between the two cancels in the normalised
            // weights, but here it would shift an absolute ΔOFV compared against
            // 3.84 (e.g. a Laplace `fit.ofv` against FOCEI-scored draws).
            let base = match screen_draw(&x_hat, &bounds, params) {
                Ok(p) => ofv_at(&p, &x_hat),
                Err(_) => return Vec::new(),
            };
            floor_probe_coords(params, &fixed_mask)
                .into_iter()
                .filter_map(|c| {
                    let x = floor_probe_point(&x_hat, &bounds, &c);
                    let p = screen_draw(&x, &bounds, params).ok()?;
                    let i = c.diag;
                    let dofv = ofv_at(&p, &x) - base;
                    dofv.is_finite().then(|| FloorProbe {
                        name: coord_names[i].clone(),
                        dofv,
                        variance: coordinate_values(&p)[i],
                        covariances_zeroed: c.zeroes_a_covariance(&x_hat),
                    })
                })
                .collect()
        };
        if let Some(w) = low_ess_warning(ess, n_samples, options.sir_scale, &heaviest, probe) {
            if options.verbose {
                eprintln!("  SIR: {w}");
            }
            warnings.push(w);
        }
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
    let n_kappa = params.omega_iov.as_ref().map_or(0, |m| m.dim());

    let mut theta_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_theta];
    let mut omega_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_eta];
    let mut sigma_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_sigma];
    let mut kappa_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_kappa];

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
        for (ks, v) in kappa_samples.iter_mut().zip(kappa_variances(&p)) {
            ks.push(v);
        }
    }

    let ci_theta: Vec<(f64, f64)> = theta_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_omega: Vec<(f64, f64)> = omega_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_sigma: Vec<(f64, f64)> = sigma_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_kappa: Vec<(f64, f64)> = kappa_samples.iter().map(|s| percentile_ci(s)).collect();

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
        ci_kappa,
        effective_sample_size: ess,
        resamples_packed,
        warnings,
    })
}

/// The **data** objective (−2 log L, no prior) SIR scores one draw with: the objective
/// the fit minimised, re-evaluated at `params` (#1755).
///
/// * A `[mixture]` model scores the K-class marginal, [`mixture_ofv`], cold
///   (`warm = None`) — the objective a mixture fit reports, which the one-class
///   `pop_nll_opts` is not: on `tests/nonmem/mixture_iv.csv` with a class-2 Ω override
///   that was 937.968 against the fit's 298.328 at the estimates, a gap that moved by
///   hundreds across draws and left an ESS of 1.09 of 200.
/// * Every other model re-solves the EBEs warm from the fit's (`eta_hats`) and takes
///   [`pop_nll_opts`] under `options`, whose `method` / `interaction` the callers set to
///   the fit's own through `fit_inputs::scoring_options` — so a Laplace fit is scored
///   with its Laplace marginal, not the FOCEI one.
///
/// At the estimates this equals the fit's data OFV exactly (pinned by
/// `sir_draw_ofv_at_the_estimate_is_the_fits_objective`).
///
/// [`mixture_ofv`]: crate::estimation::mixture::mixture_ofv
pub(crate) fn sir_draw_ofv(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    options: &FitOptions,
) -> f64 {
    if model.mixture.is_some() {
        return crate::estimation::mixture::mixture_ofv(model, population, params, options, None)
            .ofv;
    }
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let (ehs, hms, _, kappas) = run_inner_loop_warm_seeded(
        model,
        population,
        params,
        options.inner_maxiter,
        options.inner_tol,
        Some(eta_hats),
        Some(&mu_k),
        0, // SIR: no EBE convergence tracking
        0, // SIR: warm-started; no inner multi-start
        InnerHessianSeed::None,
        // A FOCE fit's weights are FOCE marginals, so each draw's EBEs are the
        // frozen-variance mode that marginal linearises around (#1722).
        options,
    );
    // Through the method-aware seam, so an AGQ fit's SIR weights come from the AGQ
    // marginal it was actually optimised against, not the FOCE one.
    2.0 * pop_nll_opts(model, population, params, &ehs, &hms, &kappas, options)
}

/// The IOV kappa variances of one unpacked draw — the `omega_iov` diagonal, in
/// packed (`kappa_names`) order, as `ci_kappa` reports them (#1705). Empty for a
/// model with no kappa. Only the diagonal, like `ci_omega`: a `block_kappa`'s
/// covariances get no interval on either side.
fn kappa_variances(p: &ModelParameters) -> Vec<f64> {
    p.omega_iov.as_ref().map_or_else(Vec::new, |iov| {
        (0..iov.dim()).map(|j| iov.matrix[(j, j)]).collect()
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
#[path = "sir_low_ess_tests.rs"]
mod low_ess_tests;

#[cfg(test)]
#[path = "sir_scorer_tests.rs"]
mod scorer_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("P{i}")).collect()
    }

    /// #1705: `ci_kappa` reads each kappa's own variance, in packed order, and
    /// only the diagonal. Two kappas with distinct variances and a non-zero
    /// covariance, so reading `(0, 0)` for every kappa, transposing the index
    /// or reading an off-diagonal each give a different vector. A model with no
    /// kappa yields none, which is what keeps a non-IOV fit's `sir_ci_kappa`
    /// `None`.
    #[test]
    fn kappa_variances_are_the_iov_diagonal_in_packed_order() {
        let mut p =
            crate::types::test_helpers::analytical_model(GradientMethod::Auto).default_params;
        assert!(kappa_variances(&p).is_empty(), "no kappa → no κ variances");
        p.omega_iov = Some(OmegaMatrix::from_matrix(
            DMatrix::from_row_slice(2, 2, &[0.05, 0.01, 0.01, 0.03]),
            vec!["KAPPA_CL".into(), "KAPPA_V".into()],
            false,
        ));
        assert_eq!(kappa_variances(&p), vec![0.05, 0.03]);
    }

    /// #1705: what `FitResult.sir_ci_kappa` carries — `None` for no kappa, the
    /// intervals otherwise. Mutation: `Some(self.ci_kappa.clone())` regardless
    /// makes a non-IOV fit write an empty `ci_kappa:` key.
    #[test]
    fn kappa_ci_is_none_without_a_kappa() {
        let mut r = SirResult {
            ci_theta: vec![(1.0, 2.0)],
            ci_omega: vec![(0.1, 0.2)],
            ci_sigma: vec![(0.01, 0.02)],
            ci_kappa: Vec::new(),
            effective_sample_size: 10.0,
            resamples_packed: None,
            warnings: Vec::new(),
        };
        assert_eq!(r.kappa_ci(), None);
        r.ci_kappa = vec![(0.02, 0.12)];
        assert_eq!(r.kappa_ci(), Some(vec![(0.02, 0.12)]));
    }

    /// #1548: a `LogitProbability` θ is proposed on the logit scale, and the
    /// weights carry `|dx/dy|` so the SIR **target** is unchanged — flat in the
    /// packed `ln θ`, restricted to the declared box.
    ///
    /// The fixture makes that target exact: `P` is a `LogitProbability`
    /// individual parameter the structural model never reads and everything
    /// else is FIX, so the likelihood is constant in `TVP` and the target is
    /// uniform on `ln θ ∈ [ln 0.001, ln 0.999]`, `E[ln θ] = −3.454`. Dropping
    /// the Jacobian makes the target uniform in `logit θ` instead, with
    /// `E[ln θ] = −E[ln(1 + e^{−y})] = −1.846` (a 1.6-unit gap; that mutation
    /// measures −1.845).
    ///
    /// The mean alone cannot see the packed-scale proposal this replaced — it
    /// targets the same flat-in-`ln θ` box. The second run can: with the upper
    /// bound widened to 5, that proposal's draws above 1 score a finite
    /// clamped-logit likelihood and are resampled, so every resampled `ln θ`
    /// must stay below 0.
    #[test]
    fn sir_logit_probability_proposal_keeps_the_packed_scale_target() {
        let model_src = "
[parameters]
  theta TVCL(5.0, FIX)
  theta TVV(50.0, FIX)
  theta TVKA(1.5, FIX)
  theta TVP(0.5, 0.001, UPPER)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_P ~ 0.1 FIX
  sigma PROP_ERR ~ 0.15 (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  P  = inv_logit(logit(TVP) + ETA_P)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("d.csv");
        std::fs::write(
            &data,
            "ID,TIME,AMT,EVID,CMT,DV,MDV\n\
             1,0,100,1,1,.,1\n1,1,0,0,1,1.2,0\n1,4,0,0,1,1.4,0\n\
             2,0,100,1,1,.,1\n2,1,0,0,1,1.0,0\n2,4,0,0,1,1.6,0\n",
        )
        .unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(&data, None, None).unwrap();

        let run = |upper: &str, tvp: f64| -> Result<Vec<f64>, String> {
            let model =
                crate::parser::model_parser::parse_model_string(&model_src.replace("UPPER", upper))
                    .expect("parse");
            assert_eq!(model.theta_transform[3], ThetaTransform::LogitProbability);
            let mut params = model.default_params.clone();
            params.theta[3] = tvp;
            let n_packed = crate::estimation::parameterization::packed_len(&params);
            let mut cov = DMatrix::zeros(n_packed, n_packed);
            // sd(ln θ) = 1.5 at θ̂ = 0.5 ⇒ sd(logit θ) = 3: wide enough to
            // cover the whole box, well short of the bound cap's trigger.
            cov[(3, 3)] = 2.25;
            let etas = vec![DVector::zeros(2); pop.subjects.len()];
            let opts = FitOptions {
                sir_samples: 8000,
                sir_resamples: 4000,
                sir_keep_samples: true,
                sir_seed: Some(7),
                verbose: false,
                ..FitOptions::default()
            };
            let r = run_sir_core(&model, &pop, &params, &etas, &cov, 0.0, &opts)?;
            Ok(r.resamples_packed
                .expect("kept")
                .iter()
                .map(|x| x[3])
                .collect())
        };

        let x = run("0.999", 0.5).expect("sir");
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let want = (f64::ln(0.001) + f64::ln(0.999)) / 2.0;
        assert!(
            (mean - want).abs() < SIR_MEAN_TOL,
            "resampled E[ln θ] = {mean}, flat-in-ln-θ target {want} (no-Jacobian target −1.846)"
        );

        let x_wide = run("5.0", 0.5).expect("sir");
        assert!(
            x_wide.iter().all(|&xi| xi < 0.0),
            "a resampled θ reached 1 with the upper bound widened to 5"
        );

        // An estimate past 1 (reachable only with such a bound) has no logit:
        // SIR refuses rather than falling back to the packed-scale proposal.
        let err = run("5.0", 1.2).unwrap_err();
        assert!(err.contains("uncertainty for TVP"), "{err}");
    }
    // Target sd of ln θ is 6.9/sqrt(12) = 2.0, so 4000 resamples give SE >= 0.03
    // (more after weighting); measured error 0.005. 0.25 is ~5 SE and 6x short
    // of the 1.6 gap to the no-Jacobian target.
    const SIR_MEAN_TOL: f64 = 0.25;

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
            SampleOutcome::InadmissibleValues,
        ];
        let coord_names = vec!["TVCL".to_string(), "PROP_ERR".to_string()];
        let cov = DMatrix::from_diagonal(&DVector::from_column_slice(&[1e7, 0.0]));
        let conditioned = condition_free_proposal(&cov, &[1.0, 1.0], &coord_names).unwrap();

        let msg = all_invalid_weights_message(&outcomes, &coord_names, &conditioned);
        assert!(msg.contains("All 5 SIR samples"), "{msg}");
        assert!(msg.contains("3 out of bounds"), "{msg}");
        assert!(msg.contains("1 non-finite theta or non-positive"), "{msg}");
        assert!(msg.contains("1 non-finite OFV"), "{msg}");
        // Most-frequent offender first, with its hit count.
        assert!(msg.contains("PROP_ERR (2)"), "{msg}");
        assert!(msg.contains("TVCL (1)"), "{msg}");
        // Proposal diagnosis + what to do about it.
        assert!(msg.contains("rank-deficient"), "{msg}");
        assert!(msg.contains("Fix or drop one parameter"), "{msg}");
    }

    /// #1701: `screen_draw` sorts each draw into the counter the failure
    /// message reports. A negative θ inside its declared box is a draw, not a
    /// rejection. A θ outside its box is "out of bounds", named by coordinate.
    /// A σ that unpacks to 0 is "non-positive … variance". That draw needs a
    /// box wide enough to let `exp` underflow, since the production box keeps
    /// every log-packed coordinate far from it. (An Ω or κ Cholesky diagonal of
    /// 0 cannot be reached this way: `unpack_params` refuses the factor first.
    /// Those clauses are pinned on `admissible_values` directly.) Both outcomes
    /// are asserted together, so the fix cannot simply drop the variance check
    /// along with θ's.
    ///
    /// Mutations: restoring `|| t <= 0.0` fails the first `Ok`; deleting the σ
    /// clause of `admissible_values` turns its `Err` into `Ok`.
    #[test]
    fn screen_draw_lets_bounds_govern_theta_and_still_rejects_a_bad_variance() {
        let model = crate::parser::model_parser::parse_model_string(
            "
[parameters]
  theta TVCL(2.0, 0.01, 20.0)
  theta SLOPE(-1.0, -5.0, 5.0)
  theta TVV(10.0, FIX)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(SLOPE + ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
        )
        .expect("parse");
        let params = model.default_params.clone();
        let PackedStart {
            packed: x_hat,
            bounds,
            ..
        } = pack_with_bounds(&params);
        // SLOPE packs on its natural scale, so the packed value is θ itself.
        assert_eq!(x_hat[1], -1.0);
        let screen = |x: &[f64], b: &PackedBounds| screen_draw(x, b, &params).map(|p| p.theta);

        let theta = screen(&x_hat, &bounds).expect("a negative θ inside its box is a draw");
        assert_eq!(theta[1], -1.0);

        let mut outside = x_hat.clone();
        outside[1] = -6.0;
        assert_eq!(
            screen(&outside, &bounds),
            Err(SampleOutcome::OutOfBounds(1))
        );

        // Packed layout: TVCL, SLOPE, TVV, ln L(ETA_CL), ln σ.
        let wide = PackedBounds {
            lower: vec![f64::NEG_INFINITY; x_hat.len()],
            upper: vec![f64::INFINITY; x_hat.len()],
        };
        let mut sigma_zero = x_hat.clone();
        sigma_zero[4] = -800.0; // exp(-800) == 0.0
        let got = screen(&sigma_zero, &wide);
        assert_eq!(got, Err(SampleOutcome::InadmissibleValues));
        let outcomes = vec![screen(&outside, &bounds).unwrap_err(), got.unwrap_err()];

        let names = coordinate_names(&params);
        let cov = DMatrix::identity(1, 1);
        let conditioned = condition_free_proposal(&cov, &[1.0], &names[..1]).unwrap();
        let msg = all_invalid_weights_message(&outcomes, &names, &conditioned);
        assert!(msg.contains("1 out of bounds"), "{msg}");
        assert!(
            msg.contains("1 non-finite theta or non-positive sigma/omega/kappa variance"),
            "{msg}"
        );
        assert!(msg.contains("SLOPE (1)"), "{msg}");
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
