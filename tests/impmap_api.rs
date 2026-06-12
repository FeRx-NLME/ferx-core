//! Integration tests for the IMPMAP estimator (`method = importance_sampling_map`).
//!
//! Tier 2 (fast, default PR job): wire-up and validation — IMPMAP runs standalone
//! and as a chain stage, returns finite estimates, and refuses IOV models. These
//! cap iterations aggressively; convergence *quality* is asserted in the Tier-3
//! slow suite below.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

fn warfarin_setup() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
    FitOptions,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    // Aggressively capped — Tier 2 tests wire-up, not convergence quality.
    opts.impmap_iterations = 12;
    opts.impmap_samples = 100;
    opts.impmap_averaging = 4;
    opts.impmap_seed = Some(7);
    opts.inner_maxiter = 30;
    (model, population, opts)
}

#[test]
fn impmap_standalone_produces_finite_estimates() {
    let (model, population, mut opts) = warfarin_setup();
    opts.method = EstimationMethod::Impmap;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("standalone impmap must produce a fit");

    assert_eq!(result.method, EstimationMethod::Impmap);
    assert_eq!(result.method_chain, vec![EstimationMethod::Impmap]);
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    for (name, v) in result.theta_names.iter().zip(result.theta.iter()) {
        assert!(
            v.is_finite() && *v > 0.0,
            "theta {name} must be finite > 0, got {v}"
        );
    }
    // Ω diagonals stay positive & finite (the diagonal floor guards this).
    for i in 0..model.n_eta {
        let w = result.omega[(i, i)];
        assert!(
            w.is_finite() && w > 0.0,
            "omega[{i},{i}] must be finite > 0, got {w}"
        );
    }
}

#[test]
fn focei_then_impmap_chain_runs() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Impmap];
    opts.outer_maxiter = 25; // bound the FOCEI warm-up stage too
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("focei → impmap chain must produce a fit");

    // IMPMAP is an estimator, so it is the final reported method.
    assert_eq!(result.method, EstimationMethod::Impmap);
    assert_eq!(
        result.method_chain,
        vec![EstimationMethod::FoceI, EstimationMethod::Impmap]
    );
    assert!(result.ofv.is_finite());
}

/// FOCEI → IMPMAP → IMP chain: the EONLY-equivalent workflow.
/// IMPMAP should compute covariance (it is the last *estimating* stage),
/// and IMP should produce an IS-likelihood evaluation at IMPMAP's parameters.
#[test]
fn focei_impmap_imp_chain_produces_covariance_and_is_result() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![
        EstimationMethod::FoceI,
        EstimationMethod::Impmap,
        EstimationMethod::Imp,
    ];
    opts.outer_maxiter = 25;
    opts.run_covariance_step = true;
    opts.is_samples = 200;
    opts.is_proposal_df = 5.0;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("focei → impmap → imp chain must produce a fit");

    // IMPMAP is the last estimator; IMP is evaluation-only.
    assert_eq!(result.method, EstimationMethod::Impmap);
    assert_eq!(
        result.method_chain,
        vec![
            EstimationMethod::FoceI,
            EstimationMethod::Impmap,
            EstimationMethod::Imp,
        ]
    );
    assert!(result.ofv.is_finite());

    // Covariance must be present (computed by IMPMAP, the last estimating stage).
    assert!(
        matches!(
            result.covariance_status,
            ferx_core::CovarianceStatus::Computed | ferx_core::CovarianceStatus::SirFallback
        ),
        "covariance should succeed when IMPMAP precedes IMP, got {:?}",
        result.covariance_status
    );

    // IS result must be populated (from the IMP evaluation stage).
    assert!(
        result.importance_sampling.is_some(),
        "importance_sampling result should be populated from IMP stage"
    );
    let is = result.importance_sampling.as_ref().unwrap();
    assert!(
        is.minus2_log_likelihood.is_finite(),
        "IS -2LL should be finite, got {}",
        is.minus2_log_likelihood
    );

    // SE should be present (extracted from covariance).
    assert!(
        result.se_theta.as_ref().map_or(false, |v| !v.is_empty()),
        "SE(theta) should be available from IMPMAP covariance"
    );
}

#[test]
fn impmap_rejects_iov_models() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, None)
        .expect("warfarin_iov data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.method = EstimationMethod::Impmap;
    opts.impmap_iterations = 3;

    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("impmap on an IOV model must be rejected (v1)");
    assert!(
        err.to_lowercase().contains("inter-occasion") || err.contains("IOV"),
        "expected IOV-not-supported error, got: {err}"
    );
}

#[test]
fn impmap_converges_with_mu_referencing_off() {
    // Regression: the closed-form log-mu-ref θ shift is the EM-correct typical-
    // value update and must apply regardless of `mu_referencing`. When it was
    // gated on `mu_referencing`, turning it off left θ stuck at its start (TVCL
    // ≈ 0.198, init 0.2) and Ω inflated (ETA_CL ≈ 0.19) — the θ/η-mean
    // confounding. With the fix, mu_referencing = false recovers the same well-
    // identified estimates (TVCL ≈ 0.13, ETA_CL ≈ 0.03).
    let (model, population, mut opts) = warfarin_setup();
    opts.method = EstimationMethod::Impmap;
    opts.mu_referencing = false;
    opts.impmap_iterations = 40;
    opts.impmap_samples = 150;
    opts.impmap_averaging = 15;
    let r = fit(&model, &population, &model.default_params, &opts)
        .expect("impmap with mu_referencing=false must fit");

    // The broken (confounded) result had TVCL ≈ 0.198 and ω²(CL) ≈ 0.19; these
    // thresholds cleanly separate the fixed result (≈0.13 / ≈0.03) from it.
    assert!(
        r.theta[0] < 0.18,
        "TVCL should move off its 0.2 start toward ~0.13, got {} (θ/η confounding regressed?)",
        r.theta[0]
    );
    assert!(
        r.omega[(0, 0)] < 0.10,
        "ω²(ETA_CL) should be ~0.03, not inflated ~0.19, got {}",
        r.omega[(0, 0)]
    );
}

#[test]
fn impmap_trace_collected_when_enabled() {
    let (model, population, mut opts) = warfarin_setup();
    opts.method = EstimationMethod::Impmap;
    opts.impmap_trace = true;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("impmap with trace must produce a fit");

    let trace = result
        .impmap_trace
        .as_ref()
        .expect("impmap_trace should be Some when impmap_trace = true");

    // 12 iteration rows + 1 final row (no covariance → no SE row).
    assert_eq!(
        trace.rows.len(),
        13,
        "expected 12 iter rows + 1 final row, got {}",
        trace.rows.len()
    );
    assert_eq!(trace.rows[0].iteration, 1);
    assert_eq!(trace.rows[11].iteration, 12);
    assert_eq!(trace.rows[12].iteration, -1_000_000_000);

    // Column name counts.
    assert_eq!(trace.theta_names.len(), 3); // TVCL, TVV, TVKA
    assert_eq!(trace.sigma_names.len(), 1); // proportional sigma
                                            // 3 etas → 6 lower-triangle elements: (1,1),(2,1),(2,2),(3,1),(3,2),(3,3)
    assert_eq!(trace.omega_names.len(), 6);

    // Every row has the right shape and finite values.
    for row in &trace.rows {
        assert!(
            row.ofv.is_finite(),
            "OFV must be finite at iter {}",
            row.iteration
        );
        assert_eq!(row.theta.len(), 3);
        assert_eq!(row.omega_lower_tri.len(), 6);
        assert_eq!(row.sigma.len(), 1);
    }
}

#[test]
fn impmap_trace_absent_when_disabled() {
    let (model, population, mut opts) = warfarin_setup();
    opts.method = EstimationMethod::Impmap;
    // impmap_trace defaults to false
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("impmap without trace must produce a fit");

    assert!(
        result.impmap_trace.is_none(),
        "impmap_trace should be None when impmap_trace = false"
    );
}

#[test]
fn impmap_rejects_invalid_proposal_df() {
    // A programmatic caller can set impmap_proposal_df directly, bypassing the
    // parser's range check. A finite df < 1 must return a clean Err, not panic
    // in the ChiSquared proposal sampler.
    let (model, population, mut opts) = warfarin_setup();
    opts.method = EstimationMethod::Impmap;
    opts.impmap_proposal_df = 0.0;
    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("impmap_proposal_df = 0 must be rejected");
    assert!(
        err.contains("impmap_proposal_df"),
        "expected impmap_proposal_df error, got: {err}"
    );
}

/// Tier 3 — full convergence. IMPMAP should recover the FOCEI solution on
/// warfarin (the Laplace approximation is accurate for this well-sampled model,
/// so the MCEM marginal and the FOCEI Laplace estimates agree). Gated behind
/// `slow-tests`; run nightly.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn impmap_converges_to_focei_on_warfarin() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    // Reference: FOCEI.
    let mut focei = FitOptions::default();
    focei.method = EstimationMethod::FoceI;
    focei.run_covariance_step = false;
    focei.outer_maxiter = 300;
    let r_focei = fit(&model, &population, &model.default_params, &focei)
        .expect("FOCEI reference fit must succeed");

    // IMPMAP.
    let mut imp = FitOptions::default();
    imp.method = EstimationMethod::Impmap;
    imp.run_covariance_step = false;
    imp.impmap_iterations = 150;
    imp.impmap_samples = 500;
    imp.impmap_averaging = 50;
    imp.impmap_seed = Some(12345);
    let r_imp =
        fit(&model, &population, &model.default_params, &imp).expect("IMPMAP fit must succeed");

    // Thetas within 10% (MCEM is stochastic; the band absorbs MC noise).
    for ((name, ti), tf) in r_imp
        .theta_names
        .iter()
        .zip(r_imp.theta.iter())
        .zip(r_focei.theta.iter())
    {
        let rel = (ti - tf).abs() / tf.abs().max(1e-8);
        assert!(
            rel < 0.10,
            "theta {name}: IMPMAP {ti} vs FOCEI {tf} (rel {rel:.3})"
        );
    }
    // Ω diagonals within 25% (variance components are noisier).
    for i in 0..model.n_eta {
        let wi = r_imp.omega[(i, i)];
        let wf = r_focei.omega[(i, i)];
        let rel = (wi - wf).abs() / wf.abs().max(1e-8);
        assert!(
            rel < 0.25,
            "omega[{i},{i}]: IMPMAP {wi} vs FOCEI {wf} (rel {rel:.3})"
        );
    }
    // OFV (both Laplace) within a few units.
    assert!(
        (r_imp.ofv - r_focei.ofv).abs() < 5.0,
        "OFV: IMPMAP {} vs FOCEI {}",
        r_imp.ofv,
        r_focei.ofv
    );
}
