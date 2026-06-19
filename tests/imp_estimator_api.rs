//! Integration tests for the **estimating** IMP path (default `method = imp`,
//! NONMEM `METHOD=IMP`).
//!
//! Tier 2 (fast, default PR job): wire-up and validation — estimating IMP runs,
//! moves parameters (vs the evaluation-only path which leaves them fixed), may
//! sit mid-chain, and refuses IOV models.
//!
//! Tier 3 (slow): warm-started end-to-end convergence on rich data. Plain IMP
//! uses a one-iteration-*lagged* proposal (mode/variance found only on the first
//! iteration), so on rich data — where the conditional posterior is razor-sharp
//! — a cold start that takes large early steps moves the posterior past the
//! lagged proposal and the ESS collapses. This is the documented rich-data
//! weakness of `METHOD=IMP` (the reason NONMEM offers IMPMAP). The estimator is
//! therefore exercised for convergence in its robust regime: sparse data, or
//! warm-started from FOCEI where the per-iteration steps are tiny.
//!
//! The evaluation-only path (`is_eval_only = true`, NONMEM `EONLY=1`) is covered
//! in `importance_sampling_api.rs`.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

fn warfarin() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

/// Keep `keep_per_subject` evenly-spaced observations per subject — the sparse-PK
/// regime where the conditional posterior is broad enough that plain IMP's lagged
/// proposal stays well-overlapped and the estimator is stable from a cold start.
fn downsample(population: &mut ferx_core::types::Population, keep_per_subject: usize) {
    for subj in population.subjects.iter_mut() {
        let n = subj.obs_times.len();
        if n <= keep_per_subject {
            continue;
        }
        let stride = (n - 1) / (keep_per_subject - 1).max(1);
        let mut idx: Vec<usize> = (0..keep_per_subject).map(|k| k * stride).collect();
        idx.sort_unstable();
        idx.dedup();
        idx.retain(|&i| i < n);
        subj.obs_times = idx.iter().map(|&i| subj.obs_times[i]).collect();
        subj.observations = idx.iter().map(|&i| subj.observations[i]).collect();
        subj.obs_cmts = idx.iter().map(|&i| subj.obs_cmts[i]).collect();
        subj.cens = idx
            .iter()
            .map(|&i| subj.cens.get(i).copied().unwrap_or(0))
            .collect();
    }
}

fn sparse_opts() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.is_iterations = 30;
    opts.is_samples = 300;
    opts.is_averaging = 10;
    opts.is_seed = Some(7);
    opts.inner_maxiter = 30;
    opts
}

#[test]
fn imp_standalone_is_an_estimator_and_moves_parameters() {
    // Default `imp` (no is_eval_only) is an MCEM estimator: it updates θ/Ω/σ.
    // On sparse data it is stable from a cold start, so the move off the initial
    // values is genuine estimation, not divergence.
    let (model, mut population) = warfarin();
    downsample(&mut population, 2);
    let mut opts = sparse_opts();
    opts.method = EstimationMethod::Imp;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("standalone estimating imp must produce a fit");

    // Estimating IMP is the canonical reported method (unlike the eval-only
    // stage, which is dropped from `method`).
    assert_eq!(result.method, EstimationMethod::Imp);
    assert_eq!(result.method_chain, vec![EstimationMethod::Imp]);
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );

    // At least one θ must move off its initial value — the defining contrast
    // with the evaluation-only path, which leaves every parameter fixed.
    let moved = result
        .theta
        .iter()
        .zip(model.default_params.theta.iter())
        .any(|(t, t0)| (t - t0).abs() > 1e-4);
    assert!(
        moved,
        "estimating imp must update at least one θ off its initial value"
    );
    for (name, v) in result.theta_names.iter().zip(result.theta.iter()) {
        assert!(
            v.is_finite() && *v > 0.0,
            "theta {name} must be finite > 0, got {v}"
        );
    }
    for i in 0..model.n_eta {
        let w = result.omega[(i, i)];
        assert!(
            w.is_finite() && w > 0.0,
            "omega[{i},{i}] must be finite > 0, got {w}"
        );
    }

    // Estimating IMP now surfaces the importance-sampling Monte-Carlo marginal
    // −2 log L (the NONMEM `METHOD=IMP` #OBJV), evaluated at the final estimates,
    // alongside the Laplace `ofv`. Previously populated only by the eval-only path.
    let is = result
        .importance_sampling
        .as_ref()
        .expect("estimating imp must surface the marginal −2 log L on importance_sampling");
    assert!(
        is.minus2_log_likelihood.is_finite(),
        "marginal −2 log L must be finite, got {}",
        is.minus2_log_likelihood
    );
    assert!(
        is.mc_standard_error.is_finite() && is.mc_standard_error >= 0.0,
        "marginal MC SE must be finite & non-negative, got {}",
        is.mc_standard_error
    );
}

#[test]
fn imp_eval_only_leaves_parameters_unchanged() {
    // The contrast case: `is_eval_only = true` evaluates −2 log L at the fixed
    // input parameters and must not move them (NONMEM `EONLY=1`).
    let (model, mut population) = warfarin();
    downsample(&mut population, 2);
    let mut opts = sparse_opts();
    opts.method = EstimationMethod::Imp;
    opts.is_eval_only = true;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("eval-only imp must produce a fit");
    for (i, (t, t0)) in result
        .theta
        .iter()
        .zip(model.default_params.theta.iter())
        .enumerate()
    {
        assert_eq!(t, t0, "eval-only imp must not move theta[{i}]");
    }
    assert!(
        result.importance_sampling.is_some(),
        "eval-only imp must populate the importance_sampling result"
    );
}

#[test]
fn imp_estimator_may_lead_a_chain() {
    // `[imp, focei]` is rejected for the evaluator but allowed for the estimator
    // — estimating IMP produces parameters the next stage can refine.
    let (model, mut population) = warfarin();
    downsample(&mut population, 2);
    let mut opts = sparse_opts();
    opts.methods = vec![EstimationMethod::Imp, EstimationMethod::FoceI];
    opts.outer_maxiter = 25;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("methods = [imp, focei] must be allowed for estimating imp");
    assert_eq!(
        result.method_chain,
        vec![EstimationMethod::Imp, EstimationMethod::FoceI]
    );
    assert_eq!(result.method, EstimationMethod::FoceI);
    assert!(result.ofv.is_finite());
}

#[test]
fn saem_then_imp_chain_runs() {
    let (model, mut population) = warfarin();
    downsample(&mut population, 2);
    let mut opts = sparse_opts();
    opts.saem_n_exploration = 5;
    opts.saem_n_convergence = 5;
    opts.saem_n_mh_steps = 2;
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::Imp];
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("saem → imp chain must produce a fit");
    // Estimating IMP is the final estimating stage.
    assert_eq!(result.method, EstimationMethod::Imp);
    assert_eq!(
        result.method_chain,
        vec![EstimationMethod::Saem, EstimationMethod::Imp]
    );
    assert!(result.ofv.is_finite());
}

#[test]
fn imp_estimator_rejects_iov_models() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, None)
        .expect("warfarin_iov data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.method = EstimationMethod::Imp;
    opts.is_iterations = 3;

    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("estimating imp on an IOV model must be rejected (v1)");
    assert!(
        err.to_lowercase().contains("inter-occasion") || err.contains("IOV"),
        "expected IOV-not-supported error, got: {err}"
    );
}

#[test]
fn imp_estimator_rejects_invalid_proposal_df() {
    // A programmatic caller can set is_proposal_df directly, bypassing the
    // parser's range check. A finite df < 1 must return a clean Err, not panic.
    let (model, population) = warfarin();
    let mut opts = sparse_opts();
    opts.method = EstimationMethod::Imp;
    opts.is_proposal_df = 0.0;
    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("is_proposal_df = 0 must be rejected");
    assert!(
        err.contains("proposal_df"),
        "expected proposal_df error, got: {err}"
    );
}

/// Tier 3 — warm-started convergence on rich data. `[focei, imp]` starts IMP at
/// the FOCEI solution where the per-iteration steps are tiny, so the lagged
/// proposal stays overlapped and the estimator is stable even on rich warfarin.
/// It must hold (refine) the FOCEI solution. Gated behind `slow-tests`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn imp_estimator_refines_focei_on_warfarin() {
    let (model, population) = warfarin();

    // Reference: FOCEI.
    let mut focei = FitOptions::default();
    focei.method = EstimationMethod::FoceI;
    focei.run_covariance_step = false;
    focei.outer_maxiter = 300;
    let r_focei = fit(&model, &population, &model.default_params, &focei)
        .expect("FOCEI reference fit must succeed");

    // FOCEI → estimating IMP.
    let mut imp = FitOptions::default();
    imp.run_covariance_step = false;
    imp.outer_maxiter = 300;
    imp.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    imp.is_iterations = 80;
    imp.is_samples = 1000;
    imp.is_averaging = 30;
    imp.is_seed = Some(12345);
    let r_imp = fit(&model, &population, &model.default_params, &imp)
        .expect("FOCEI → estimating IMP fit must succeed");

    assert_eq!(
        r_imp.method,
        EstimationMethod::Imp,
        "IMP is the final estimating stage"
    );

    // Thetas within 5% of FOCEI (IMP refines, not relocates).
    for ((name, ti), tf) in r_imp
        .theta_names
        .iter()
        .zip(r_imp.theta.iter())
        .zip(r_focei.theta.iter())
    {
        let rel = (ti - tf).abs() / tf.abs().max(1e-8);
        assert!(
            rel < 0.05,
            "theta {name}: IMP {ti} vs FOCEI {tf} (rel {rel:.3})"
        );
    }
    // Ω diagonals within 20% (variance components are noisier).
    for i in 0..model.n_eta {
        let wi = r_imp.omega[(i, i)];
        let wf = r_focei.omega[(i, i)];
        let rel = (wi - wf).abs() / wf.abs().max(1e-8);
        assert!(
            rel < 0.20,
            "omega[{i},{i}]: IMP {wi} vs FOCEI {wf} (rel {rel:.3})"
        );
    }
    // OFV (both Laplace) within a couple of units.
    assert!(
        (r_imp.ofv - r_focei.ofv).abs() < 3.0,
        "OFV: IMP {} vs FOCEI {}",
        r_imp.ofv,
        r_focei.ofv
    );
}
