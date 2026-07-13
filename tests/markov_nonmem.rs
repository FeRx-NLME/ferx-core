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
