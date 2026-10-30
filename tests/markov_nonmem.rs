//! NONMEM 7.5.1 cross-check for the ferx CTMM (continuous-time Markov) endpoint
//! (`[markov_model]`, issue #759).
//!
//! Runs ferx on the 2-state CTMM `examples/ctmm_2state.ferx` +
//! `data/ctmm_2state.csv` and asserts the converged log-intensities and OFV match
//! a NONMEM reference on the **same** likelihood.
//!
//! ## Why this anchors to NONMEM (overriding the plan's `msm` choice)
//!
//! `plans/tte-survival-markov.md` proposed anchoring CTMM to R `msm`, on the
//! grounds that NONMEM's *dataset* CTMM mechanism (EVID=3 + A0_FLG occupancy
//! bookkeeping) is architecturally incompatible with ferx's reader. That
//! objection is about the dataset encoding, not the likelihood: NONMEM can score
//! the identical panel/snapshot CTMM likelihood through the general `F_FLAG=1`
//! (LIKELIHOOD) route, with the closed-form 2-state transition-probability matrix
//! written directly in `$PRED`. For a **fixed-effects** (`n_eta = 0`) CTMM both
//! engines minimise the same exact `−2 Σ log P(Δt)[sₘ,sₘ₊₁]`, so the OFV is
//! directly comparable and NONMEM is the stronger external anchor. See
//! `tests/nonmem/ctmm_2state.ctl` (and `.lst`/`.ext`).
//!
//! The generator is `Q = [[−q01, q01], [q10, −q10]]` with `q01 = exp(LQ01)`,
//! `q10 = exp(LQ10)`, `λ = q01 + q10`, and
//! `P(Δt) = [[ (q10+q01 e)/λ , q01(1−e)/λ ], [ q10(1−e)/λ , (q01+q10 e)/λ ]]`,
//! `e = exp(−λ Δt)`. The NONMEM data file `tests/nonmem/ctmm_2state.csv` carries
//! the previous state (`PSTATE`) and the gap (`DT`) per record; each subject's
//! first record is `MDV = 1` (the initial condition ferx conditions on).
//!
//! Gated behind `slow-tests` (fits to convergence + runs the covariance step);
//! skipped in the default PR job, run nightly.
//!
//! ## Reference values
//!
//! NONMEM 7.5.1 `$EST METHOD=0` (FO; exact for a fixed-effects `F_FLAG=1` model),
//! from `tests/nonmem/ctmm_2state.ext` (`-1000000000` row):
//!
//! | Parameter | NONMEM | ferx |
//! |-----------|-------:|-----:|
//! | LQ01 (log q₀→₁) | −0.540113 | −0.539011 |
//! | LQ10 (log q₁→₀) | −1.215660 | −1.215053 |
//! | OFV (−2 log L)  | 606.98181 | 606.9817 |
//!
//! The OFV agrees to 4 decimals; the log-intensities agree to ~0.001 (well under
//! 0.01 of the ~0.14 standard errors) — the likelihood is flat near the optimum,
//! so the two optimizers stop at sub-tolerance-different points on an identical
//! objective. Both start from the same initial estimates (LQ01 = −0.7,
//! LQ10 = −1.2).

#![cfg(feature = "markov")]

use ferx_core::{fit_from_files, FitOptions};

// NONMEM 7.5.1 METHOD=0 reference (tests/nonmem/ctmm_2state.ext, -1000000000 row).
const NM_LQ01: f64 = -0.540113;
const NM_LQ10: f64 = -1.215660;
const NM_OFV: f64 = 606.98181;

// `examples/ctmm_pd_2state.ferx` at its data-generating initial estimates, one evaluation.
const OFV_AT_TRUTH: f64 = 715.6644417808;
const OFV_AT_TRUTH_TOL: f64 = 1e-7;
// More than the 3.88 that refitting everything but LQ01 at its true value buys; 2.1 below
// the measured 6.59.
const MIN_IMPROVEMENT: f64 = 4.5;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_ctmm_matches_nonmem_2state() {
    let opts = FitOptions {
        outer_maxiter: 300,
        ..Default::default()
    };
    let res = fit_from_files(
        "examples/ctmm_2state.ferx",
        Some("data/ctmm_2state.csv"),
        None,
        Some(opts),
    )
    .expect("2-state CTMM fit must converge");

    let (lq01, lq10) = (res.theta[0], res.theta[1]);

    // Log-intensities: NONMEM and ferx agree to ~1e-3 on an identical likelihood
    // (< 0.01 of the ~0.14 SEs). Use an absolute tolerance — the values are near
    // zero, so a relative one would be needlessly strict.
    assert!(
        (lq01 - NM_LQ01).abs() < 5e-3,
        "LQ01 {lq01} vs NONMEM {NM_LQ01}"
    );
    assert!(
        (lq10 - NM_LQ10).abs() < 5e-3,
        "LQ10 {lq10} vs NONMEM {NM_LQ10}"
    );

    // OFV (−2 log L): exact same objective; agree to well under 0.01.
    assert!(
        (res.ofv - NM_OFV).abs() < 1e-2,
        "OFV {} vs NONMEM {NM_OFV}",
        res.ofv
    );
}

/// The objective of `examples/ctmm_pd_2state.ferx` at its own initial estimates, which are
/// the data-generating parameters (`outer_maxiter = 0`, so one evaluation and no optimizer).
fn drug_driven_ctmm_ofv_at_truth() -> f64 {
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    };
    let res = fit_from_files(
        "examples/ctmm_pd_2state.ferx",
        Some("data/ctmm_pd_2state.csv"),
        None,
        Some(opts),
    )
    .expect("drug-driven CTMM evaluates at its initial estimates");
    assert!(res.ofv.is_finite(), "OFV at truth {}", res.ofv);
    res.ofv
}

/// Drug-driven (time-inhomogeneous) CTMM, objective pin (#817, #2437).
///
/// One evaluation at the data-generating parameters, so it pins the occupancy integration
/// and the FD η-Hessian without depending on where an optimizer stops. Measured:
/// 715.6644417809 on Linux x86_64 and 715.6644417808 on macOS arm64. The bound is 1e-7
/// because a CTMM occupancy integration is not reproducible to the last digit across
/// platforms in general: on the previous step the same evaluation differed by 1.4e-9.
#[test]
fn drug_driven_ctmm_objective_at_truth() {
    let ofv = drug_driven_ctmm_ofv_at_truth();
    eprintln!("drug-driven CTMM OFV at truth: {ofv:.10}");
    assert!(
        (ofv - OFV_AT_TRUTH).abs() < OFV_AT_TRUTH_TOL,
        "OFV at truth {ofv} vs recorded {OFV_AT_TRUTH}"
    );
}

/// Drug-driven (time-inhomogeneous) CTMM recovery test (#817).
///
/// A NONMEM `$DES` anchor is impractical for a PK-coupled CTMM: the PK must integrate
/// continuously across the subject while the occupancy resets to the observed state at
/// every observation, and NONMEM's only per-record reset (EVID=3) zeros *all*
/// compartments — the PK included (plan §3.4 already deems NONMEM's CTMM mechanism not
/// feasible). The drug-driven path is instead validated by (a) an **exact closed-form
/// anchor** — a commuting generator `Q(t)=C(t)·Q₀` matches `expm(Q₀·∫C)`
/// (`tests/markov_smoke.rs::ctmm_drug_driven_matches_closed_form_commuting_generator`),
/// and (b) **reduction** — `SLOPE=0` reduces to the homogeneous CTMM, which *is*
/// NONMEM-anchored above. This test adds a **simulate → fit self-consistency** check on
/// what these data identify.
///
/// They identify the sign of the drug effect, **not** `LQ01`. No PK is observed, so
/// `LQ01`, `SLOPE`, `CL` and `V` trade off along a ridge in the concentration, and `LQ10`
/// rides it too. Profile (#2437, macOS arm64, `LQ01` FIX, the rest refitted with BOBYQA):
/// OFV 712.69 / 711.79 / 710.94 / 710.45 / 710.08 / 709.45 / 709.07 at `LQ01` = −1.0 /
/// −1.2 (truth) / −1.6 / −2.3 / −3.0 / −4.5 / −6.0 (the lower bound), with `SLOPE`
/// 0.11–0.99 and `LQ10` −0.51…−0.59 along it; the free fits land anywhere from `LQ01`
/// −1.6 to the bound, `LQ10` −0.48…−0.69. An `|LQ01 − truth| < 0.12` bound held only
/// while the optimizer stopped near its start, which is the truth.
///
/// What the test can assert is that the fit **descends the ridge**. Refitting everything
/// but `LQ01` at the truth's `LQ01` gets 3.88 below the start (715.66 → 711.79), so a fit
/// that only re-solves the other parameters cannot clear `MIN_IMPROVEMENT`. Measured: 6.59
/// on macOS arm64 and 6.59 on Linux x86_64 (BOBYQA to the bound, 709.07).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_drug_driven_ctmm_recovers_simulated_params() {
    let opts = FitOptions {
        outer_maxiter: 300,
        ..Default::default()
    };
    let res = fit_from_files(
        "examples/ctmm_pd_2state.ferx",
        Some("data/ctmm_pd_2state.csv"),
        None,
        Some(opts),
    )
    .expect("drug-driven CTMM fit must converge");

    // theta = [TVCL, TVV, LQ01, LQ10, SLOPE]. Data-generating: SLOPE = 0.15. The
    // concentration effect is resolved as clearly positive (not merely near 0) all along the
    // ridge (≥ 0.11 on the profile above); its size rides the ridge.
    let slope = res.theta[4];
    assert!(
        slope > 0.05,
        "SLOPE {slope} must be a clear positive drug effect"
    );
    assert!(
        res.converged,
        "drug-driven CTMM fit must report convergence"
    );
    // The start is the truth, so a fit that barely moves would pass the bound above. It must
    // descend the ridge, past what refitting at the truth's LQ01 reaches (see above).
    let at_truth = drug_driven_ctmm_ofv_at_truth();
    eprintln!("fitted OFV {:.6}, at truth {at_truth:.6}", res.ofv);
    assert!(
        res.ofv.is_finite() && at_truth - res.ofv > MIN_IMPROVEMENT,
        "fitted OFV {} must improve on the objective at its start (the truth, {at_truth}) by \
         more than {MIN_IMPROVEMENT}",
        res.ofv
    );
}
