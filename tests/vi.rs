//! Integration tests for variational inference (`method = vi`).
//!
//! Tier-2: every test calls the public `fit()` boundary but returns immediately —
//! a handful of VI iterations, or an expected `Err` from a compatibility guard.
//! No convergence loops, so these are compile-checked and run on every PR.
//!
//! The headline checks are the ones only the public boundary can make:
//!
//! * `method = vi` and the `vi_*` keys parse, dispatch, and reach `FitResult::vi`.
//! * `ofv` is `NaN` by default and finite under `vi_final_ofv = laplace` — the
//!   contract that keeps a lower bound from being mistaken for a `−2 log L`.
//! * VI composes with the other estimators in a chain, in both directions.
//!
//! Scope refusals, gradient parity and the optimizer's behaviour are unit-tested
//! in `src/estimation/vi/`. The full convergence fit and the NONMEM comparison
//! live in `docs/estimation/vi.qmd`.

use ferx_core::parser::model_parser::{parse_full_model, parse_model_string};
use ferx_core::{
    fit, read_nonmem_csv, CompiledModel, EstimationMethod, FitOptions, FitResult, Population,
    ViFamily, ViFinalOfv,
};
use std::path::Path;

/// Warfarin oral 1-cpt, three random effects, proportional error.
const WARFARIN_SRC: &str = r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

fn warfarin_data() -> Population {
    read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data must load")
}

fn warfarin_model() -> CompiledModel {
    parse_model_string(WARFARIN_SRC).expect("warfarin model parses")
}

/// A short VI run: enough iterations to exercise the loop, few enough for Tier-2.
fn vi_opts(iters: usize) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Vi,
        vi_iters: iters,
        vi_mc_samples: 1,
        vi_seed: Some(7),
        run_covariance_step: false,
        ..FitOptions::default()
    }
}

fn run(model: &CompiledModel, population: &Population, opts: &FitOptions) -> FitResult {
    fit(model, population, &model.default_params, opts).expect("VI fit runs")
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[test]
fn vi_options_parse_from_the_model_file() {
    let src = format!(
        "{WARFARIN_SRC}\n[fit_options]\n  method = vi\n  vi_iters = 5\n  \
         vi_mc_samples = 2\n  vi_lr = 0.02\n  vi_family = mean_field\n  \
         vi_omega_update = adam\n  vi_avg_last = 3\n  vi_eta_grad = fd\n  \
         vi_kl = mc\n  vi_final_ofv = laplace\n  vi_seed = 42\n"
    );
    let parsed = parse_full_model(&src).expect("the vi_* keys parse");
    let o = &parsed.fit_options;
    assert_eq!(o.method, EstimationMethod::Vi);
    assert_eq!(o.vi_iters, 5);
    assert_eq!(o.vi_mc_samples, 2);
    assert_eq!(o.vi_lr, 0.02);
    assert_eq!(o.vi_family, ViFamily::MeanField);
    assert_eq!(o.vi_omega_update, ferx_core::ViOmegaUpdate::Adam);
    assert_eq!(o.vi_avg_last, Some(3));
    assert_eq!(o.vi_eta_grad, ferx_core::ViEtaGrad::Fd);
    assert_eq!(o.vi_kl, ferx_core::ViKl::Mc);
    assert_eq!(o.vi_final_ofv, ViFinalOfv::Laplace);
    assert_eq!(o.vi_seed, Some(42));
}

#[test]
fn unknown_vi_option_values_are_rejected_with_a_useful_message() {
    // `ParsedModel` is not `Debug`, so match rather than `expect_err`.
    let parse_err = |src: &str| match parse_full_model(src) {
        Err(e) => e,
        Ok(_) => panic!("expected a parse error for: {src}"),
    };

    let err = parse_err(&format!(
        "{WARFARIN_SRC}\n[fit_options]\n  method = vi\n  vi_family = banana\n"
    ));
    assert!(
        err.contains("full_rank") && err.contains("mean_field"),
        "the error should name the valid values, got: {err}"
    );

    let err = parse_err(&format!(
        "{WARFARIN_SRC}\n[fit_options]\n  method = vi\n  vi_final_ofv = imp\n"
    ));
    assert!(
        err.contains("imp_eval_only"),
        "the error should point at the chain that does give an IS likelihood, got: {err}"
    );

    let err = parse_err(&format!(
        "{WARFARIN_SRC}\n[fit_options]\n  method = vi\n  vi_kl = laplace\n"
    ));
    assert!(
        err.contains("analytic") && err.contains("mc"),
        "the error should name the valid values, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

#[test]
fn vi_runs_and_populates_its_result() {
    let (model, population) = (warfarin_model(), warfarin_data());
    let result = run(&model, &population, &vi_opts(15));

    let vi = result.vi.as_ref().expect("FitResult::vi populated");
    assert_eq!(vi.n_iterations, 15);
    assert_eq!(vi.elbo_trace.len(), 15);
    assert_eq!(vi.family, "full_rank");
    assert!(vi.neg_two_elbo.is_finite());
    assert_eq!(vi.eta_means.len(), population.subjects.len());
    assert_eq!(vi.eta_means[0].len(), 3);
    assert_eq!(vi.eta_covs[0].len(), 3);
    assert_eq!(
        vi.n_fd_subjects, 0,
        "warfarin is inside the analytic eta-gradient scope"
    );

    // Estimates come back on the natural scale, positive, and finite.
    assert!(result.theta.iter().all(|t| t.is_finite() && *t > 0.0));
    for k in 0..3 {
        assert!(result.omega[(k, k)] > 0.0);
    }
}

/// The load-bearing contract: an ELBO is a lower bound, so it never lands in
/// `ofv` unless a genuine marginal likelihood was asked for.
#[test]
fn ofv_is_nan_by_default_and_finite_when_requested() {
    let (model, population) = (warfarin_model(), warfarin_data());

    let default = run(&model, &population, &vi_opts(10));
    assert!(
        default.ofv.is_nan(),
        "the ELBO must not be reported as an OFV"
    );
    assert!(
        default.warnings.iter().any(|w| w.contains("lower bound")),
        "a NaN OFV must be explained; warnings were {:?}",
        default.warnings
    );

    let mut opts = vi_opts(10);
    opts.vi_final_ofv = ViFinalOfv::Laplace;
    let with_ofv = run(&model, &population, &opts);
    assert!(
        with_ofv.ofv.is_finite(),
        "vi_final_ofv = laplace must produce an OFV"
    );
    assert!(
        (with_ofv.ofv - with_ofv.vi.as_ref().unwrap().neg_two_elbo).abs() > 1e-9,
        "the Laplace objective and the ELBO bound must be distinguishable"
    );
}

/// Both variational families are reachable through the public API.
#[test]
fn mean_field_family_runs_end_to_end() {
    let (model, population) = (warfarin_model(), warfarin_data());
    let mut opts = vi_opts(10);
    opts.vi_family = ViFamily::MeanField;
    let result = run(&model, &population, &opts);
    let vi = result.vi.as_ref().unwrap();
    assert_eq!(vi.family, "mean_field");
    // A diagonal posterior has no off-diagonal covariance to report.
    assert_eq!(vi.eta_covs[0][0][1], 0.0);
}

/// `vi_kl = mc` is reachable through the public API and reports the route it took.
///
/// Both routes estimate the same objective, so the fitted `θ` should be close; the
/// Monte-Carlo KL is only noisier. See `src/estimation/vi/elbo_tests.rs` for the
/// kernel-level convergence and gradient checks.
#[test]
fn mc_kl_route_runs_end_to_end() {
    let (model, population) = (warfarin_model(), warfarin_data());

    let analytic = run(&model, &population, &vi_opts(10));
    assert_eq!(analytic.vi.as_ref().unwrap().kl, "analytic");

    let mut opts = vi_opts(10);
    opts.vi_kl = ferx_core::ViKl::Mc;
    opts.vi_mc_samples = 8;
    let result = run(&model, &population, &opts);
    let vi = result.vi.as_ref().expect("FitResult::vi populated");

    assert_eq!(vi.kl, "mc");
    assert_eq!(vi.n_kl_fallback_subjects, 0);
    assert!(vi.neg_two_elbo.is_finite());
    assert!(result.theta.iter().all(|t| t.is_finite() && *t > 0.0));
    for k in 0..3 {
        assert!(result.omega[(k, k)] > 0.0);
    }
}

/// Combining `vi_kl = mc` with `vi_omega_update = adam` warns, and the warning reaches
/// `FitResult$warnings` rather than only `ferx check`.
///
/// That plumbing is the substance of this test: `fit_inner` consumes
/// `check_model_options` through `first_error`, which discards warning-severity
/// diagnostics, so a warning-only check needs the second collecting pass to be visible
/// at all. It must also *not* block the fit — this is the published configuration.
#[test]
fn mc_kl_with_adam_omega_warns_but_still_fits() {
    let (model, population) = (warfarin_model(), warfarin_data());
    let mut opts = vi_opts(10);
    opts.vi_kl = ferx_core::ViKl::Mc;
    opts.vi_omega_update = ferx_core::ViOmegaUpdate::Adam;

    let result = run(&model, &population, &opts);
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("vi_kl = mc") && w.contains("vi_omega_update = adam")),
        "the unanchored-Omega warning must reach FitResult; warnings were {:?}",
        result.warnings
    );
    // A warning, not a refusal: the fit ran and Omega is still usable.
    assert!(result.vi.is_some());
    for k in 0..3 {
        assert!(result.omega[(k, k)] > 0.0);
    }

    // The default pair is silent.
    let quiet = run(&model, &population, &vi_opts(10));
    assert!(
        !quiet
            .warnings
            .iter()
            .any(|w| w.contains("vi_omega_update = adam")),
        "the default configuration must not warn; warnings were {:?}",
        quiet.warnings
    );
}

/// A VI fit reports standard errors and a `Computed` covariance status, like every
/// other method — the covariance step runs at the VI estimate.
///
/// Only the public boundary can check this: `covariance_status` is resolved in
/// `fit_inner` from `run_covariance_step && matrix.is_some()`, so an estimator that
/// silently skipped the step would surface here as `Failed` with no SEs.
#[test]
fn covariance_step_produces_standard_errors() {
    let (model, population) = (warfarin_model(), warfarin_data());
    let mut opts = vi_opts(10);
    opts.run_covariance_step = true;

    let result = run(&model, &population, &opts);
    assert_eq!(
        result.covariance_status,
        ferx_core::CovarianceStatus::Computed,
        "VI must run the covariance step; warnings were {:?}",
        result.warnings
    );
    let se = result
        .se_theta
        .as_ref()
        .expect("a computed covariance must yield theta SEs");
    assert_eq!(se.len(), 3);
    assert!(
        se.iter().all(|s| s.is_finite() && *s > 0.0),
        "SEs must be finite and positive, got {se:?}"
    );
    // The ELBO contract is unaffected: SEs come from the Laplace covariance, so a
    // NaN OFV and a computed covariance coexist.
    assert!(result.ofv.is_nan());
}

/// Declared `θ` bounds hold through the public API. Adam is unconstrained, so
/// without the projection in `run_vi` the box would be silently ignored.
///
/// The comparison carries a ULP-scale tolerance because the box is enforced in
/// *packed* space, exactly as it is for FOCE/FOCEI: `x` is clamped to `ln(lower)`
/// and `unpack_params` reports `exp(ln(lower))`, which is not bit-identical to
/// `lower`. Snapping the natural-scale value instead would make VI behave
/// differently from every other estimator. The tolerance is ~4 orders tighter than
/// the pre-fix escape (3% of the bound), so it still catches an unenforced box.
#[test]
fn declared_theta_bounds_hold_through_the_api() {
    let population = warfarin_data();

    // The free estimate first, then a box that excludes it.
    let free = run(&warfarin_model(), &population, &vi_opts(40)).theta[0];
    let src = WARFARIN_SRC.replace(
        "theta TVCL(0.13, 0.001, 10.0)",
        &format!("theta TVCL(0.13, {:.10}, 10.0)", 0.5 * (free + 0.13)),
    );
    let bounded = parse_model_string(&src).expect("bounded model parses");
    let lower = bounded.default_params.theta_lower[0];
    assert!(
        free < lower,
        "test is vacuous: free TVCL {free} is not below the new lower bound {lower}"
    );

    let got = run(&bounded, &population, &vi_opts(40)).theta[0];
    assert!(
        got >= lower - 1e-12 * (1.0 + lower.abs()),
        "TVCL = {got} escaped its declared lower bound {lower}"
    );
}

// ---------------------------------------------------------------------------
// Chaining
// ---------------------------------------------------------------------------

/// VI as the first stage: FOCEI must pick up its parameters and report a real
/// OFV. This is the recommended way to finish a VI fit.
#[test]
fn vi_chains_into_focei() {
    let (model, population) = (warfarin_model(), warfarin_data());
    let opts = FitOptions {
        methods: vec![EstimationMethod::Vi, EstimationMethod::FoceI],
        vi_iters: 10,
        vi_mc_samples: 1,
        vi_seed: Some(7),
        outer_maxiter: 1,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("vi → focei chain runs");
    assert!(
        result.vi.is_some(),
        "the VI stage's result must survive the chain"
    );
    assert!(
        result.ofv.is_finite(),
        "the terminal FOCEI stage must report a real OFV"
    );
}

/// And as a later stage: VI must accept another estimator's parameters as its
/// starting point, and a terminal VI stage still reports no OFV by default.
#[test]
fn focei_chains_into_vi() {
    let (model, population) = (warfarin_model(), warfarin_data());
    let opts = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Vi],
        vi_iters: 10,
        vi_mc_samples: 1,
        vi_seed: Some(7),
        outer_maxiter: 1,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("focei → vi chain runs");
    assert!(result.vi.is_some());
    assert!(
        result.ofv.is_nan(),
        "a terminal VI stage reports no OFV by default"
    );
}
