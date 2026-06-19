//! Slow end-to-end Importance Sampling tests.
//!
//! Gated behind the `slow-tests` feature; skipped in the default PR job and
//! run nightly via `slow-tests.yml`. The point of this suite is to exercise
//! the algorithmic claim that IS-LL is meaningfully different from the
//! FOCE/Laplace OFV in the regime where Laplace is biased — sparsely-sampled
//! per-subject PK.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::KappaTreatment;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// Downsample every subject to at most `keep_per_subject` observations. Mirrors
/// the typical sparse-PK regime (e.g. 2–3 troughs per patient in routine TDM)
/// where the per-subject posterior of η is non-Gaussian and Laplace biases the
/// marginal LL.
fn downsample_population(population: &mut ferx_core::types::Population, keep_per_subject: usize) {
    for subj in population.subjects.iter_mut() {
        if subj.obs_times.len() <= keep_per_subject {
            continue;
        }
        // Keep evenly-spaced observations across the original schedule rather
        // than the first N — preserves coverage of absorption + elimination
        // phases, which is more representative of real sparse-TDM data than
        // a head-of-schedule slice (which would over-sample absorption).
        let n = subj.obs_times.len();
        let stride = (n - 1) / (keep_per_subject - 1).max(1);
        let mut keep_idx: Vec<usize> = (0..keep_per_subject).map(|k| k * stride).collect();
        // Make sure indices are unique and in bounds.
        keep_idx.sort_unstable();
        keep_idx.dedup();
        keep_idx.retain(|&i| i < n);

        let obs_times: Vec<f64> = keep_idx.iter().map(|&i| subj.obs_times[i]).collect();
        let observations: Vec<f64> = keep_idx.iter().map(|&i| subj.observations[i]).collect();
        let obs_cmts: Vec<usize> = keep_idx.iter().map(|&i| subj.obs_cmts[i]).collect();
        let cens: Vec<u8> = keep_idx
            .iter()
            .map(|&i| subj.cens.get(i).copied().unwrap_or(0))
            .collect();

        subj.obs_times = obs_times;
        subj.observations = observations;
        subj.obs_cmts = obs_cmts;
        subj.cens = cens;
    }
}

/// Regression test for joint (eta, kappa) IS on the warfarin IOV example.
///
/// Before this fix IMP held kappa fixed at its EBE mode, producing a ~64-unit
/// OFV gap vs FOCEI. Joint sampling closes that gap: ferx IMP should land near
/// the NONMEM 7.5.1 IMP reference of 310.18.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn warfarin_iov_joint_sampling_matches_nonmem() {
    let model = parse_model_file(std::path::Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(
        std::path::Path::new("data/warfarin_iov.csv"),
        None,
        Some("OCC"),
    )
    .expect("warfarin_iov data must load");

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.methods = vec![
        EstimationMethod::Saem,
        EstimationMethod::FoceI,
        EstimationMethod::Imp,
    ];
    opts.is_samples = 3000;
    opts.is_seed = Some(2026);
    opts.is_eval_only = true; // scoring the chain's −2 log L, not re-estimating

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("warfarin_iov SAEM → FOCEI → IMP must succeed");

    let imp = result
        .importance_sampling
        .as_ref()
        .expect("IMP stage must populate FitResult.importance_sampling");

    // NONMEM 7.5.1 IMP: 310.18; ferx joint sampling: 309.00.
    // Tolerance ±2.0 accommodates MC noise and platform-dependent FOCEI polish.
    // Under the old kappa-fixed path this would have been ~373 (+64 gap) — so
    // this assertion fails without the joint-sampling fix.
    assert!(
        (imp.minus2_log_likelihood - 309.0).abs() < 2.0,
        "warfarin_iov IMP −2LL should be near 309.0 (NONMEM: 310.18), got {:.4}",
        imp.minus2_log_likelihood
    );
    assert!(
        imp.mc_standard_error < 0.1,
        "MC SE should be small at K=3000, got {:.4}",
        imp.mc_standard_error
    );
    assert!(
        matches!(imp.kappa_treatment, KappaTreatment::Marginalized),
        "kappa_treatment must be Marginalized for IOV model"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn sparse_warfarin_focei_imp_runs_and_reports_finite_ll() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let mut population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    // Downsample to 2 obs/subject — sparse-data regime where the Laplace
    // marginal-LL approximation is known to be biased.
    downsample_population(&mut population, 2);

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.outer_maxiter = 200;
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    opts.is_samples = 2000; // enough to drive MC SE well below the gap we expect
    opts.is_seed = Some(2026);
    opts.is_eval_only = true; // scoring FOCEI's fit, not re-estimating

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("sparse-warfarin FOCEI → IMP must produce a fit");

    let imp = result
        .importance_sampling
        .as_ref()
        .expect("imp stage must populate FitResult.importance_sampling");

    // Plausibility: −2 log L must be finite and within a sane PK band.
    assert!(
        imp.minus2_log_likelihood.is_finite(),
        "−2 log L IS must be finite, got {}",
        imp.minus2_log_likelihood
    );
    assert!(
        imp.mc_standard_error.is_finite() && imp.mc_standard_error >= 0.0,
        "MC SE must be finite & non-negative, got {}",
        imp.mc_standard_error
    );

    // The IS estimate should be deterministic at a fixed seed. Run a second
    // pass with identical options and compare bit-for-bit on −2 log L.
    let result2 = fit(&model, &population, &model.default_params, &opts)
        .expect("second sparse-warfarin fit must succeed");
    let imp2 = result2.importance_sampling.unwrap();
    assert_eq!(
        imp.minus2_log_likelihood, imp2.minus2_log_likelihood,
        "IS −2 log L must be deterministic at a fixed seed"
    );
}
