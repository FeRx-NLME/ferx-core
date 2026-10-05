//! Tier 3: SIR intervals for the IOV kappa variance (#1705), on a fit to
//! convergence.
//!
//! `mbma_placebo` is ferx-r's BTAV example (`tests/data/mbma_placebo/`, also
//! used by `theta_level_blocks.rs`): a compartment-free arm-level MBMA with a
//! `PLACEBO[STUDY, TIME]` level block and one between-arm kappa weighted by arm
//! size (`kappa KAPPA_ARM ~ … weight = NARM`). FOCEI, as the file declares it.
//!
//! The per-PR test of the plumbing is
//! `estimation::run_sir::tests::in_fit_and_standalone_sir_report_the_same_kappa_ci`
//! (`warfarin_iov`, Tier 1). It cannot pin bracketing or the Wald agreement:
//! that fixture's ESS is ~5 of 1000, so its interval is a handful of distinct
//! draws. This fixture's ESS is ~143.

use ferx_core::types::FitOptions;
use ferx_core::{fit, prepare_run, run_sir};
use std::path::PathBuf;

/// Measured worst `|ln(SIR endpoint / Wald endpoint)|` over both endpoints is
/// recorded on the test; this is the bound, with the headroom stated there.
const WALD_LOG_GAP_TOL: f64 = 0.35;

fn ci_bits(ci: &Option<Vec<(f64, f64)>>) -> Option<Vec<(u64, u64)>> {
    ci.as_ref()
        .map(|v| v.iter().map(|(a, b)| (a.to_bits(), b.to_bits())).collect())
}

/// One fit with `sir = true` and the covariance step, then `run_sir` on the
/// same fit with every SIR field cleared.
///
/// 1. **Identity.** Both paths report the same `sir_ci_kappa`, θ/Ω/σ CIs and
///    ESS, to the bit — on a weighted kappa this time, and with an ESS high
///    enough (asserted > 10 first) that a bit match compares a distribution,
///    not one draw. Mutation: drop either path's fill → that path's `expect`.
/// 2. **Bracketing.** The κ interval contains the fitted κ. Mutation: read
///    `omega` instead of `omega_iov` in `kappa_variances` → `ETA_E0`'s
///    interval (2.05–25.2 measured) cannot contain κ ≈ 156.
/// 3. **Agreement with the covariance step.** The SIR interval and the Wald
///    interval on the scale the proposal is built on (`ln L_kk`, i.e.
///    `κ·exp(±2·1.96·se_kappa/(2κ))`) agree endpoint by endpoint to
///    [`WALD_LOG_GAP_TOL`] on the log scale. They are not expected to be
///    equal — SIR exists to correct the asymptotic interval — but an interval
///    built from the wrong coordinate (an off-diagonal, a Cholesky factor, the
///    standard deviation instead of the variance) misses by far more: reading
///    `L_kk` instead of `L_kk²` halves `ln κ`, a gap of ~2.5 here.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn mbma_placebo_kappa_sir_interval_is_shared_brackets_and_matches_wald() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mbma_placebo");
    let prep = prepare_run(
        dir.join("mbma_placebo.ferx").to_str().unwrap(),
        Some(dir.join("mbma_placebo.csv").to_str().unwrap()),
    )
    .expect("prepare mbma_placebo");
    let opts = FitOptions {
        verbose: false,
        run_covariance_step: true,
        sir: true,
        sir_samples: 1000,
        sir_resamples: 250,
        sir_seed: Some(1705),
        ..prep.parsed.fit_options.clone()
    };
    let model = &prep.parsed.model;
    let pop = &prep.population;
    let f = fit(model, pop, &prep.init_params, &opts).expect("fit mbma_placebo");
    assert!(f.covariance_matrix.is_some(), "covariance step failed");

    let mut bare = f.clone();
    bare.sir_ci_theta = None;
    bare.sir_ci_omega = None;
    bare.sir_ci_sigma = None;
    bare.sir_ci_kappa = None;
    bare.sir_ess = None;
    let sa = run_sir(&bare, Some(model), Some(pop), &opts).expect("run_sir");

    // 1. Identity.
    let ess = f.sir_ess.expect("in-fit ESS");
    assert!(ess.is_finite() && ess > 10.0, "degenerate SIR, ESS {ess}");
    let k_fit = f
        .sir_ci_kappa
        .as_ref()
        .expect("in-fit SIR filled no sir_ci_kappa");
    sa.sir_ci_kappa
        .as_ref()
        .expect("standalone run_sir filled no sir_ci_kappa");
    assert_eq!(ci_bits(&sa.sir_ci_kappa), ci_bits(&f.sir_ci_kappa), "κ");
    assert_eq!(ci_bits(&sa.sir_ci_theta), ci_bits(&f.sir_ci_theta), "θ");
    assert_eq!(ci_bits(&sa.sir_ci_omega), ci_bits(&f.sir_ci_omega), "Ω");
    assert_eq!(ci_bits(&sa.sir_ci_sigma), ci_bits(&f.sir_ci_sigma), "σ");
    assert_eq!(sa.sir_ess.map(f64::to_bits), Some(ess.to_bits()), "ESS");

    // 2. Bracketing, and 3. Wald agreement.
    let iov = f.omega_iov.as_ref().expect("an IOV fit");
    let se = f
        .se_kappa
        .as_ref()
        .expect("se_kappa from the covariance step");
    assert_eq!(k_fit.len(), 1, "one kappa: {:?}", f.kappa_names);
    let mut worst = 0.0_f64;
    for (k, &(lo, hi)) in k_fit.iter().enumerate() {
        let kappa = iov[(k, k)];
        let half = 1.96 * se[k] / kappa; // 2·1.96·se(ln L_kk), se(ln L_kk) = se_κ/(2κ)
        let (wlo, whi) = (kappa * (-half).exp(), kappa * half.exp());
        for v in [kappa, se[k], lo, hi, wlo, whi] {
            assert!(
                v.is_finite() && v > 0.0,
                "{}: non-finite or non-positive {v}",
                f.kappa_names[k]
            );
        }
        assert!(
            lo < kappa && kappa < hi,
            "{}: SIR [{lo}, {hi}] does not bracket κ = {kappa}",
            f.kappa_names[k]
        );
        let gap = (lo / wlo).ln().abs().max((hi / whi).ln().abs());
        eprintln!(
            "{}: κ = {kappa:.6}, SIR [{lo:.6}, {hi:.6}], Wald(ln) [{wlo:.6}, {whi:.6}], \
             |ln| gap lo {:.4} hi {:.4}, ESS {ess:.1}",
            f.kappa_names[k],
            (lo / wlo).ln().abs(),
            (hi / whi).ln().abs()
        );
        worst = worst.max(gap);
    }
    assert!(
        worst < WALD_LOG_GAP_TOL,
        "SIR vs Wald worst |ln| gap {worst:.4} ≥ {WALD_LOG_GAP_TOL}"
    );
}
