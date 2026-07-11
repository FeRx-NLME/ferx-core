//! Tier-3 convergence anchor for the Phase 4 Binary / logistic endpoint (#760).
//!
//! Gated behind BOTH `survival` (the feature) and `slow-tests` (a full fit to
//! convergence), so it is skipped on the per-PR job and runs nightly — see the
//! test-tier rules in CLAUDE.md.
//!
//! **Exact anchor.** A fixed-effects (`n_eta = 0`) logistic fit *is* ordinary
//! logistic regression, so ferx must reproduce base-R
//! `glm(DV ~ X + TIME, family = binomial)` on the same committed data. The reference,
//! from `data/binary_logistic.csv` (generator seeded, 60 subjects × 3 times):
//!
//! ```text
//! (Intercept) = -0.775172   X = 0.870140   TIME = 0.827029   deviance (-2logL) = 213.5955
//! ```
//!
//! ferx's OFV is on the `-2logL` (deviance) scale, so it must equal the glm deviance —
//! the strongest correctness signal, pinning the Bernoulli likelihood constant *and*
//! form against an independent, license-free tool. (NONMEM `F_FLAG=1` logistic would
//! reproduce the same glm fit; glm is the canonical exact reference here.)

#![cfg(all(feature = "survival", feature = "slow-tests"))]

use ferx_core::{fit_from_files, FitOptions};

#[test]
fn binary_fixed_effects_matches_r_glm() {
    let r = fit_from_files(
        "examples/binary_logistic.ferx",
        Some("data/binary_logistic.csv"),
        Some(&["X"]),
        Some(FitOptions::default()),
    )
    .expect("fixed-effects binary fit must converge");

    // base-R glm(DV ~ X + TIME, family = binomial) on data/binary_logistic.csv.
    let (r_intercept, r_x, r_time, r_deviance) = (-0.775172, 0.870140, 0.827029, 213.595477);
    let (th0, thx, tht) = (r.theta[0], r.theta[1], r.theta[2]);

    // OFV == glm deviance pins the likelihood constant + form (exact to optimizer tol).
    assert!(
        (r.ofv - r_deviance).abs() < 0.05,
        "ferx OFV {} vs glm deviance {r_deviance}",
        r.ofv
    );
    // θ estimates match the glm MLE within the derivative-free (BOBYQA) outer tolerance.
    assert!(
        (th0 - r_intercept).abs() < 0.01,
        "TH0 {th0} vs glm {r_intercept}"
    );
    assert!((thx - r_x).abs() < 0.01, "THX {thx} vs glm {r_x}");
    assert!((tht - r_time).abs() < 0.01, "THT {tht} vs glm {r_time}");
}
