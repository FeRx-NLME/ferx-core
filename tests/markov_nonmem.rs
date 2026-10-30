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
/// NONMEM-anchored above. This test adds a **simulate → fit self-consistency** check:
/// fitting the Gillespie-simulated example recovers the data-generating transition
/// structure (the drug effect `SLOPE`, and both baseline log-intensities).
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

    // theta = [TVCL, TVV, LQ01, LQ10, SLOPE]. Data-generating: LQ01=-1.2, LQ10=-0.7,
    // SLOPE=0.15. The transition structure (baseline intensities + concentration effect) is
    // well identified from 60 subjects; CL and its IIV are only weakly informed (no PK is
    // observed), so they are not asserted here.
    let (lq01, lq10, slope) = (res.theta[2], res.theta[3], res.theta[4]);
    assert!((lq01 - (-1.2)).abs() < 0.12, "LQ01 {lq01} vs truth -1.2");
    assert!((lq10 - (-0.7)).abs() < 0.15, "LQ10 {lq10} vs truth -0.7");
    assert!((slope - 0.15).abs() < 0.06, "SLOPE {slope} vs truth 0.15");
    // The concentration effect is resolved as clearly positive (not merely near 0).
    assert!(
        slope > 0.05,
        "SLOPE {slope} must be a clear positive drug effect"
    );

    // Regression pin on the objective (catches numerical drift in the occupancy integration).
    assert!(
        (res.ofv - 712.226).abs() < 0.5,
        "OFV {} vs recorded 712.226",
        res.ofv
    );
}

// ── Minimal CTMM (mCTMM, `type = mctmm`, #2474) ─────────────────────────────────────
//
// `examples/mctmm_3state.ferx` + `data/mctmm_3state.csv`: a 3-state mCTMM (Schindler &
// Karlsson 2017) with a proportional-odds steady state, two cut-points and a subject
// covariate on the linear predictor. The data were simulated by
// `tests/nonmem/mctmm_3state_sim.R` (LTAU = ln 2, C1 = −0.5, LD2 = ln 1.5, BX = 0.8).
//
// NONMEM scores the identical fixed-effects panel likelihood through `F_FLAG=1`, with the
// 3×3 `expm(Q·Δt)` written in closed form (Sylvester's formula on the eigenvalues
// 0, λ₁, λ₂ of the tridiagonal generator) — `tests/nonmem/mctmm_3state.ctl`. A three-state
// chain is the smallest that tests the construction: a two-state mCTMM has no interior
// state, so it cannot tell a pairwise rate from a single shared one.
//
// NONMEM 7.5.1 `METHOD=0` ends with ERROR=134 (rounding errors) but the objective is flat to
// 1e-9 over its last 20 iterations, and a third, independent minimisation — R `nlminb` on
// `expm::expm` (`tests/nonmem/mctmm_3state_refit.R`) — lands on
// (0.637929, −0.440238, 0.506227, 1.064568) at OFV 966.4469312412, agreeing with NONMEM's
// 966.4469312446 to 3e-9 and its estimates to 1e-5. So the reference is the optimum, not an
// artefact of the early stop.
//
// | Parameter | NONMEM | R nlminb |
// |-----------|-------:|---------:|
// | LTAU | 0.637932 | 0.637929 |
// | C1   | −0.440228 | −0.440238 |
// | LD2  | 0.506227 | 0.506227 |
// | BX   | 1.06458 | 1.064568 |
// | OFV  | 966.446931 | 966.446931 |

const NM_MCTMM_THETA: [f64; 4] = [0.637932, -0.440228, 0.506227, 1.06458];
const NM_MCTMM_OFV: f64 = 966.44693124;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_mctmm_matches_nonmem_3state() {
    let opts = FitOptions {
        outer_maxiter: 300,
        ..Default::default()
    };
    let res = fit_from_files(
        "examples/mctmm_3state.ferx",
        Some("data/mctmm_3state.csv"),
        None,
        Some(opts),
    )
    .expect("3-state mCTMM fit must converge");

    eprintln!("mCTMM ferx theta = {:?}, OFV = {:.10}", res.theta, res.ofv);
    for (i, (got, want)) in res.theta.iter().zip(NM_MCTMM_THETA).enumerate() {
        assert!(got.is_finite(), "theta[{i}] = {got}");
        assert!(
            (got - want).abs() < MCTMM_THETA_TOL,
            "theta[{i}] {got} vs NONMEM {want}"
        );
    }
    assert!(
        (res.ofv - NM_MCTMM_OFV).abs() < MCTMM_OFV_TOL,
        "OFV {} vs NONMEM {NM_MCTMM_OFV}",
        res.ofv
    );
}

// Measured on the Linux reference platform (`tools/linux-test.sh`, `ferx-platform:
// Linux/x86_64 … GLIBC 2.41`); native macOS arm64 prints the same digits. ferx θ =
// (0.6379674, −0.4401577, 0.5062020, 1.0647260), worst |Δθ| = 1.46e-4 (BX) against NONMEM —
// under 1e-3 of every SE (0.08–0.31) — and OFV 966.4469319355, 7.0e-7 above NONMEM. The
// likelihood is flat near the optimum (ferx sits 1e-4 off in θ yet within 7e-7 in OFV), so
// the optimizers stop at sub-tolerance-different points of one objective. Bounds: ~10× the
// θ gap, ~14× the OFV gap.
const MCTMM_THETA_TOL: f64 = 1.5e-3;
const MCTMM_OFV_TOL: f64 = 1e-5;
