use super::*;
use nalgebra::{DMatrix, DVector};

// ── default thread count cap (#707) ─────────────────────────────────────

#[test]
fn cap_default_threads_leaves_one_core_free() {
    assert_eq!(cap_default_threads(2), 1);
    assert_eq!(cap_default_threads(4), 3);
    assert_eq!(cap_default_threads(8), 7);
}

#[test]
fn cap_default_threads_caps_at_eight() {
    assert_eq!(cap_default_threads(9), 8);
    assert_eq!(cap_default_threads(16), 8);
    assert_eq!(cap_default_threads(128), 8);
}

#[test]
fn cap_default_threads_never_zero() {
    // A single-core report (or an `available_parallelism()` failure mapped to 1)
    // must still leave at least one worker thread.
    assert_eq!(cap_default_threads(1), 1);
    assert_eq!(cap_default_threads(0), 1);
}

#[test]
fn configure_global_thread_pool_rejects_zero() {
    // `0` must be rejected rather than silently recorded, which would otherwise read
    // as "pick automatically" and mask the caller's intent (and would incorrectly
    // mark the width explicit). Returns before touching the process-wide width, so
    // this is safe to run alongside other tests.
    assert!(configure_global_thread_pool(0).is_err());
}

#[test]
fn an_explicit_thread_count_beats_the_default_cap() {
    // The precedence `effective_default_threads` applies, without the process-global
    // read — the statics are shared with every other test in this binary, so the
    // resolution is tested here and the end-to-end path in `ferx-cli`'s
    // `a_threads_flag_sizes_one_pool_and_no_idle_set_beside_it` (#1460).
    assert_eq!(resolve_default_threads(3), 3);
    // An explicit count is a user's choice, not a suggestion: it is not re-capped at
    // DEFAULT_THREADS_CAP the way the automatic count is.
    assert_eq!(resolve_default_threads(64), 64);
}

#[test]
fn no_explicit_thread_count_falls_back_to_the_capped_default() {
    // `0` is "nothing was declared", which is why `configure_global_thread_pool`
    // refuses to store it.
    assert_eq!(resolve_default_threads(0), default_thread_count());
}

#[test]
fn reconfiguring_the_same_thread_count_is_accepted_and_a_different_one_is_not() {
    // `configure_global_thread_pool` is once-per-process, as the `build_global` it
    // replaced was: pools sized from the first value already exist, so a second,
    // differing count would be reported as in force while nothing used it — and
    // `PoolPlan::from_budget(0, …)` would then read a width no pool has (#1115/#1460).
    assert!(reconfiguration_result(2, 2).is_ok());
    let err = reconfiguration_result(2, 4).expect_err("4 after 2 must not be accepted");
    assert!(err.contains('2') && err.contains('4'), "{err}");
}

#[test]
fn default_thread_count_matches_cap_of_available_parallelism() {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    assert_eq!(default_thread_count(), cap_default_threads(available));
}

// ── resolve_data_path (#690) ────────────────────────────────────────────

#[test]
fn resolve_data_path_neither_given_is_error() {
    assert!(resolve_data_path(None, None).is_err());
}

#[test]
fn resolve_data_path_external_only_no_warning() {
    let (path, warning) = resolve_data_path(None, Some("cli.csv")).unwrap();
    assert_eq!(path, "cli.csv");
    assert!(warning.is_none());
}

#[test]
fn resolve_data_path_model_only_no_warning() {
    let (path, warning) = resolve_data_path(Some("model.csv"), None).unwrap();
    assert_eq!(path, "model.csv");
    assert!(warning.is_none());
}

#[test]
fn resolve_data_path_both_equal_no_warning() {
    let (path, warning) = resolve_data_path(Some("same.csv"), Some("same.csv")).unwrap();
    assert_eq!(path, "same.csv");
    assert!(warning.is_none());
}

#[test]
fn resolve_data_path_both_differ_external_wins_with_warning() {
    let (path, warning) = resolve_data_path(Some("model.csv"), Some("cli.csv")).unwrap();
    assert_eq!(path, "cli.csv");
    let warning = warning.expect("differing paths must warn");
    assert!(warning.contains("cli.csv"));
    assert!(warning.contains("model.csv"));
}

#[test]
fn resolve_data_path_same_file_different_text_no_warning() {
    // Same on-disk file reached via textually different paths (e.g. the
    // model's `[data] path` dir-joined vs. a raw `--data` value) must not
    // trigger a spurious override warning.
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("warfarin.csv");
    std::fs::write(&csv, "ID,TIME,DV,EVID,AMT,CMT\n").unwrap();
    let model_p = csv.to_str().unwrap().to_string();
    let external_p = format!("{}/./warfarin.csv", dir.path().to_str().unwrap());

    let (path, warning) = resolve_data_path(Some(&model_p), Some(&external_p)).unwrap();
    assert_eq!(path, external_p);
    assert!(warning.is_none(), "same file must not warn: {:?}", warning);
}

#[test]
fn fit_from_files_uses_model_data_block_when_data_path_none() {
    // #690 regression: fit_from_files must resolve the model's own
    // [data] block via resolve_data_path, like run_model_with_data_inits
    // does, rather than requiring an explicit data_path argument.
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("warfarin.csv");
    std::fs::copy("data/warfarin.csv", &data).unwrap();
    let model_content = std::fs::read_to_string("examples/warfarin.ferx").unwrap();
    let model_content = format!("{model_content}\n[data]\n  path = warfarin.csv\n");
    let model = dir.path().join("model.ferx");
    std::fs::write(&model, model_content).unwrap();

    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let result = fit_from_files(model.to_str().unwrap(), None, None, Some(opts))
        .expect("fit_from_files must honor the model's [data] block");
    assert_eq!(result.data_path.as_deref(), Some(data.to_str().unwrap()));
}

/// #658: an adaptive-dosing simulation on a covariate-selected error model
/// must be rejected — the assay keys residual variance by CMT, but a
/// `Selected` spec keys endpoints by covariate branch (a CMT-keyed lookup
/// would miss and draw NaN). A single-endpoint error model is accepted.
#[test]
fn adaptive_rejects_selected_error_model() {
    use crate::parser::model_parser::parse_model_string;
    let ode_selected = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_TOTAL   ~ 0.05 (sd)
  sigma PROP_UNBOUND ~ 0.30 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous
";
    let model = parse_model_string(ode_selected).expect("ODE+Selected model parses");
    let err = reject_selected_error_for_adaptive(&model)
        .expect_err("Selected error model must be rejected for adaptive dosing");
    assert_eq!(err.code(), Some("E_ADAPTIVE_UNSUPPORTED"), "{err}");
    assert_eq!(err.block(), Some("error_model"), "{err}");
    let err = err.to_string();
    assert!(
        err.to_lowercase().contains("adaptive") && err.contains("error_model"),
        "error should cite the adaptive/error_model restriction: {err}"
    );

    // A single-endpoint error model on the same structure is accepted.
    let ode_single = ode_selected.replace(
            "  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }",
            "  DV ~ proportional(PROP_TOTAL)",
        );
    let single = parse_model_string(&ode_single).expect("ODE+Single model parses");
    assert!(reject_selected_error_for_adaptive(&single).is_ok());
}

// --- Adaptive-dosing "no silent wrong answer" guards (#391) --------------
// The reactive driver now supports IOV (#701), time-varying covariates (#700),
// and system resets (#716); only an SDE `[diffusion]` model remains unsupported
// and must be a typed error, not a silent number. See `reject_unsupported_adaptive`.

const PLAIN_ADAPTIVE_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  DV ~ proportional(PROP)
";

// A dose-free base subject with one observation and no per-event covariate
// snapshots — the shape the reactive driver accepts. Tests mutate one field
// to trip a specific guard.
fn adaptive_base_subject() -> crate::types::Subject {
    crate::types::Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: Vec::new(),
        obs_times: vec![24.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0],
        obs_cmts: vec![1],
        covariates: std::collections::HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn adaptive_pop(subject: crate::types::Subject) -> Population {
    Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

#[test]
fn adaptive_accepts_iov_model() {
    // IOV is now supported (#701): a fresh κ is drawn per decision window and
    // threaded through the per-segment eta, so the guard no longer rejects it.
    use crate::parser::model_parser::parse_model_string;
    let iov = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(iov).expect("IOV model parses");
    assert!(model.n_kappa > 0, "test model must declare a kappa");
    assert!(
        reject_unsupported_adaptive(&model, &adaptive_pop(adaptive_base_subject())).is_ok(),
        "IOV model is now supported for adaptive dosing (#701)"
    );
}

#[test]
fn adaptive_rejects_sde_model() {
    use crate::parser::model_parser::parse_model_string;
    let sde = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[diffusion]
  central ~ 0.01

[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(sde).expect("SDE model parses");
    assert!(model.is_sde(), "test model must be an SDE model");
    let err = reject_unsupported_adaptive(&model, &adaptive_pop(adaptive_base_subject()))
        .expect_err("SDE model must be rejected for adaptive dosing");
    assert_eq!(err.code(), Some("E_ADAPTIVE_UNSUPPORTED"), "{err}");
    assert_eq!(err.block(), Some("diffusion"), "{err}");
    let err = err.to_string();
    assert!(
        err.to_lowercase().contains("diffusion")
            || err.to_lowercase().contains("sde")
            || err.to_lowercase().contains("stochastic"),
        "error should cite the SDE/diffusion restriction: {err}"
    );
}

/// #1081 review r1 #1: a `correlation` line is refused, naming the line and why;
/// the same model without it is accepted.
#[test]
fn adaptive_rejects_a_residual_correlation_line() {
    use crate::parser::model_parser::parse_model_string;
    let with_line = PLAIN_ADAPTIVE_MODEL
        .replace(
            "  sigma PROP ~ 0.05 (sd)",
            "  theta RHO(0.3, -0.4, 0.9)\n  sigma PROP ~ 0.05 (sd)",
        )
        .replace(
            "  DV ~ proportional(PROP)",
            "  DV ~ proportional(PROP)\n  correlation ar1(RHO) by ARM",
        );
    let model = parse_model_string(&with_line).expect("parses");
    assert!(model.residual_serial.is_some());
    let err = reject_unsupported_adaptive(&model, &adaptive_pop(adaptive_base_subject()))
        .expect_err("a correlation line is refused");
    assert_eq!(err.code(), Some("E_ADAPTIVE_UNSUPPORTED"), "{err}");
    assert_eq!(err.block(), Some("error_model"), "{err}");
    assert_eq!(
        err.to_string(),
        "adaptive-dosing simulation does not support `correlation ar1(RHO) by ARM`: the \
         controller draws its assay noise one reading at a time, so readings in one block \
         would be independent. Drop the line for the adaptive simulation."
    );
    let plain = parse_model_string(PLAIN_ADAPTIVE_MODEL).expect("parses");
    assert!(reject_unsupported_adaptive(&plain, &adaptive_pop(adaptive_base_subject())).is_ok());
}

#[test]
fn adaptive_accepts_time_varying_covariate_subject() {
    use crate::parser::model_parser::parse_model_string;
    // Time-varying covariates are now supported via per-event PK (#700); the
    // guard no longer rejects them. Correctness of the per-event PK itself is
    // pinned by the driver-level oracle
    // `adaptive_tv_covariate_controller_matches_static_event_driven`.
    let model = parse_model_string(PLAIN_ADAPTIVE_MODEL).expect("plain model parses");
    let mut subject = adaptive_base_subject();
    subject.obs_covariates = vec![std::collections::HashMap::from([("WT".to_string(), 70.0)])];
    assert!(subject.has_tv_covariates());
    assert!(
        reject_unsupported_adaptive(&model, &adaptive_pop(subject)).is_ok(),
        "a subject with time-varying covariates must now be accepted"
    );
}

#[test]
fn adaptive_accepts_pk_only_covariate_subject() {
    use crate::parser::model_parser::parse_model_string;
    // A covariate that varies only on EVID=2 rows lands in `pk_only_covariates`,
    // which `has_tv_covariates()` does NOT inspect. It is now supported via
    // per-event PK (#700) and no longer rejected; the `has_tv_covariates()`
    // assert proves this pk-only case is a distinct path from the obs/dose one
    // (`compute_event_pk_params` was broadened to materialise it too).
    let model = parse_model_string(PLAIN_ADAPTIVE_MODEL).expect("plain model parses");
    let mut subject = adaptive_base_subject();
    subject.obs_times = Vec::new();
    subject.observations = Vec::new();
    subject.obs_cmts = Vec::new();
    subject.cens = Vec::new();
    subject.pk_only_times = vec![24.0];
    subject.pk_only_covariates = vec![std::collections::HashMap::from([("WT".to_string(), 90.0)])];
    assert!(
        !subject.has_tv_covariates(),
        "pk_only snapshots are invisible to has_tv_covariates() — a distinct path"
    );
    assert!(
        reject_unsupported_adaptive(&model, &adaptive_pop(subject)).is_ok(),
        "a subject with EVID=2 time-varying covariates must now be accepted"
    );
}

#[test]
fn adaptive_accepts_reset_subject() {
    use crate::parser::model_parser::parse_model_string;
    // System resets (EVID=3) are now supported (#716): the reactive driver zeros
    // the compartments at each reset time and turns off infusions opened before it,
    // and the frozen-replay verifier is reset-aware, so the guard no longer rejects
    // them. Correctness of the reset itself is pinned by the driver-level oracle
    // `adaptive_reset_matches_static_predict` in adaptive_sim_tests.rs.
    let model = parse_model_string(PLAIN_ADAPTIVE_MODEL).expect("plain model parses");
    let mut subject = adaptive_base_subject();
    subject.reset_times = vec![48.0];
    assert!(subject.has_resets());
    assert!(
        reject_unsupported_adaptive(&model, &adaptive_pop(subject)).is_ok(),
        "a subject with system resets must now be accepted (#716)"
    );
}

#[test]
fn adaptive_accepts_plain_bsv_model() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(PLAIN_ADAPTIVE_MODEL).expect("plain model parses");
    // A BSV-only model with a dose-free, time-constant-covariate, reset-free
    // subject is exactly the supported cut — no guard should fire.
    assert!(reject_unsupported_adaptive(&model, &adaptive_pop(adaptive_base_subject())).is_ok());
}

fn make_subject(eta: Vec<f64>, iwres: Vec<f64>) -> SubjectResult {
    let n = iwres.len();
    SubjectResult {
        id: "1".to_string(),
        eta: DVector::from_vec(eta),
        ipred: vec![0.0; n],
        pred: vec![0.0; n],
        iwres,
        cwres: vec![0.0; n],
        npde: vec![],
        npd: vec![],
        ofv_contribution: 0.0,
        cens: vec![0; n],
        n_obs: n,
        pmix: None,
        mixest: None,
        extra_columns: vec![],
        per_obs_tad: vec![],
        compartment_states: vec![],
        #[cfg(feature = "survival")]
        discrete_rows: Vec::new(),
    }
}

#[test]
fn test_eta_shrinkage_zero_when_eta_rms_matches_omega_sd() {
    // NONMEM convention: shrinkage = 1 - sqrt(mean(eta^2)) / sqrt(omega).
    // omega = 1.0, eta = [+1, -1] => mean(eta^2) = 1 => shrinkage = 0.
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let subjects = vec![
        make_subject(vec![1.0], vec![0.0]),
        make_subject(vec![-1.0], vec![0.0]),
    ];
    let sh = compute_eta_shrinkage(&subjects, &omega);
    assert_eq!(sh.len(), 1);
    assert!(
        (sh[0]).abs() < 1e-10,
        "expected ~0 shrinkage, got {}",
        sh[0]
    );
}

#[test]
fn test_eta_shrinkage_positive_when_etas_shrunk() {
    // Etas close to zero → shrinkage > 0
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let subjects: Vec<SubjectResult> = (0..10)
        .map(|_| make_subject(vec![0.01], vec![0.0]))
        .collect();
    let sh = compute_eta_shrinkage(&subjects, &omega);
    assert!(sh[0] > 0.5, "expected high shrinkage, got {}", sh[0]);
}

#[test]
fn test_eta_shrinkage_uses_n_not_n_minus_1_divisor() {
    // Regression: with the old (centered, n-1) estimator, eta = [+a, -a] gave
    // SD = a*sqrt(2), so omega = 2.0 was needed to land at shrinkage 0.
    // With the NONMEM (uncentered, n) estimator, mean(eta^2) = a^2, so the
    // same eta values + omega = 2.0 must now give a clearly positive value:
    // shrinkage = 1 - 1/sqrt(2) ≈ 0.293.
    let omega = DMatrix::from_diagonal_element(1, 1, 2.0);
    let subjects = vec![
        make_subject(vec![1.0], vec![0.0]),
        make_subject(vec![-1.0], vec![0.0]),
    ];
    let sh = compute_eta_shrinkage(&subjects, &omega);
    let expected = 1.0 - 1.0 / 2.0_f64.sqrt();
    assert!(
        (sh[0] - expected).abs() < 1e-10,
        "expected {}, got {}",
        expected,
        sh[0]
    );
}

#[test]
fn test_eta_shrinkage_nan_when_omega_zero() {
    let omega = DMatrix::zeros(1, 1);
    let subjects = vec![
        make_subject(vec![0.1], vec![0.0]),
        make_subject(vec![-0.1], vec![0.0]),
    ];
    let sh = compute_eta_shrinkage(&subjects, &omega);
    assert!(sh[0].is_nan(), "expected NaN when omega=0");
}

#[test]
fn test_eta_shrinkage_nan_when_fewer_than_2_subjects() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let subjects = vec![make_subject(vec![0.5], vec![0.0])];
    let sh = compute_eta_shrinkage(&subjects, &omega);
    assert!(sh[0].is_nan(), "expected NaN with only 1 subject");
}

#[test]
fn test_eps_shrinkage_near_zero_for_unit_normal_iwres() {
    // NONMEM convention: shrinkage = 1 - sqrt(mean(IWRES^2)).
    // IWRES = [+1, -1] => mean(IWRES^2) = 1 => shrinkage = 0.
    let subjects = vec![
        make_subject(vec![0.0], vec![1.0]),
        make_subject(vec![0.0], vec![-1.0]),
    ];
    let sh = compute_eps_shrinkage(&subjects);
    assert!((sh).abs() < 1e-10, "expected ~0 eps shrinkage, got {}", sh);
}

#[test]
fn test_eps_shrinkage_nan_for_fewer_than_2_residuals() {
    let subjects = vec![make_subject(vec![0.0], vec![0.5])];
    assert!(compute_eps_shrinkage(&subjects).is_nan());
}

#[test]
fn test_eps_shrinkage_negative_when_iwres_inflated() {
    // IWRES with mean(IWRES^2) > 1 must produce a negative value, not be
    // clamped. Matches the SAEM repro on `nmdata_20230216_1.csv` where
    // mean(IWRES^2) ~ 2.45 -> shrinkage ~ -0.566.
    let subjects = vec![
        make_subject(vec![0.0], vec![2.0]),
        make_subject(vec![0.0], vec![-2.0]),
    ];
    let sh = compute_eps_shrinkage(&subjects);
    assert!(sh < 0.0, "expected negative shrinkage, got {}", sh);
    assert!((sh - (1.0 - 2.0)).abs() < 1e-10);
}

#[test]
fn test_eps_shrinkage_warning_emits_below_threshold() {
    let w = eps_shrinkage_warning(-0.10).expect("expected warning");
    assert!(w.contains("mean(IWRES^2) > 1"));
    assert!(w.contains("-10.0%"));
}

#[test]
fn test_eps_shrinkage_warning_silent_above_threshold() {
    // Tiny negatives are noise — no warning.
    assert!(eps_shrinkage_warning(-0.01).is_none());
    // Positive shrinkage — no warning.
    assert!(eps_shrinkage_warning(0.20).is_none());
    // Right at the boundary — no warning (uses `<`).
    assert!(eps_shrinkage_warning(-0.05).is_none());
    // NaN — no warning.
    assert!(eps_shrinkage_warning(f64::NAN).is_none());
}

#[test]
fn saem_non_mu_referenced_warning_lists_individual_params_with_unmapped_eta() {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
    model.indiv_param_names = vec!["CL".into(), "V".into(), "KA".into()];
    model.eta_names = vec!["ETA_CL".into(), "ETA_V".into()];
    model.eta_map = vec![0, 1, -1];
    model.mu_refs.insert(
        "ETA_CL".into(),
        MuRef {
            theta_name: "TVCL".into(),
            transform: MuTransform::Log,
        },
    );

    let warning =
        saem_non_mu_referenced_individual_params_warning(&model, &[]).expect("expected warning");
    assert!(warning.contains("not mu-referenced: V."));
    assert!(!warning.contains("not mu-referenced: CL"));
    assert!(!warning.contains("KA"));
}

#[test]
fn saem_non_mu_referenced_warning_is_none_when_all_eta_params_are_muref() {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
    model.indiv_param_names = vec!["CL".into(), "V".into(), "KA".into()];
    model.eta_names = vec!["ETA_CL".into(), "ETA_V".into()];
    model.eta_map = vec![0, 1, -1];
    model.mu_refs.insert(
        "ETA_CL".into(),
        MuRef {
            theta_name: "TVCL".into(),
            transform: MuTransform::Log,
        },
    );
    model.mu_refs.insert(
        "ETA_V".into(),
        MuRef {
            theta_name: "TVV".into(),
            transform: MuTransform::Log,
        },
    );

    assert!(saem_non_mu_referenced_individual_params_warning(&model, &[]).is_none());
}

/// A covariate mu-reference group (#619) mu-references its eta — but only in a
/// run that builds one. `run_saem` builds none under `mu_referencing = false`,
/// and none for a mixture, and those are exactly the runs where the warning has
/// to survive: reading the group off the model alone would silence it in the
/// case it exists for (#621).
#[test]
fn saem_non_mu_referenced_warning_returns_when_the_run_builds_no_covariate_group() {
    let src = r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let model = crate::parser::model_parser::parse_model_string(src).expect("model parses");
    assert_eq!(model.covariate_mu_refs.len(), 1, "CL carries a group");
    assert!(
        saem_non_mu_referenced_individual_params_warning(&model, &["ETA_CL"]).is_none(),
        "a run that builds the group does mu-reference CL"
    );
    let w = saem_non_mu_referenced_individual_params_warning(&model, &[])
        .expect("with no group built, nothing mu-references CL");
    assert!(w.contains("CL"), "{w}");
    assert!(
        !w.contains(" V"),
        "V keeps its own single-anchor mu-ref: {w}"
    );
}

// ── kappa shrinkage ──────────────────────────────────────────────────────

fn make_kappas(vals: Vec<Vec<f64>>) -> Vec<Vec<DVector<f64>>> {
    // vals[subj_idx][occ_idx] = single-kappa value
    vals.into_iter()
        .map(|occ_vals| {
            occ_vals
                .into_iter()
                .map(|v| DVector::from_vec(vec![v]))
                .collect()
        })
        .collect()
}

#[test]
fn test_kappa_shrinkage_pooled_zero_when_rms_matches_omega_sd() {
    // omega_iov = 1.0; kappas = [+1, -1] across 2 subjects × 1 occasion
    // mean(κ²) = 1 → shrinkage = 0
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![vec![1.0], vec![-1.0]]);
    let sh = compute_kappa_shrinkage(&kappas, &[], &omega);
    assert_eq!(sh.len(), 1);
    assert!((sh[0]).abs() < 1e-10, "expected ~0, got {}", sh[0]);
}

#[test]
fn test_kappa_shrinkage_pooled_positive_when_shrunk() {
    // kappas near zero → shrinkage > 0
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![
        vec![0.01, 0.02],
        vec![-0.01, -0.02],
        vec![0.01, 0.02],
        vec![-0.01, -0.02],
    ]);
    let sh = compute_kappa_shrinkage(&kappas, &[], &omega);
    assert!(sh[0] > 0.9, "expected high shrinkage, got {}", sh[0]);
}

#[test]
fn test_kappa_shrinkage_pooled_nan_when_omega_zero() {
    let omega = DMatrix::zeros(1, 1);
    let kappas = make_kappas(vec![vec![0.1], vec![-0.1]]);
    let sh = compute_kappa_shrinkage(&kappas, &[], &omega);
    assert!(sh[0].is_nan());
}

#[test]
fn test_kappa_shrinkage_pooled_nan_when_fewer_than_2_obs() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![vec![0.5]]);
    let sh = compute_kappa_shrinkage(&kappas, &[], &omega);
    assert!(sh[0].is_nan());
}

#[test]
fn test_kappa_shrinkage_by_occ_returns_empty_for_single_occasion() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![vec![0.5], vec![-0.5]]);
    let sh = compute_kappa_shrinkage_by_occ(&kappas, &[], &omega);
    assert!(sh.is_empty(), "expected empty for 1 occasion, got {:?}", sh);
}

#[test]
fn test_kappa_shrinkage_by_occ_values() {
    // 4 subjects, 2 occasions.
    // OCC 1: kappas = [+1, -1, +1, -1] → mean(κ²) = 1 → shrinkage = 0 with omega=1
    // OCC 2: kappas = [0.1, -0.1, 0.1, -0.1] → mean(κ²) = 0.01 → high shrinkage
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![
        vec![1.0, 0.1],
        vec![-1.0, -0.1],
        vec![1.0, 0.1],
        vec![-1.0, -0.1],
    ]);
    let sh = compute_kappa_shrinkage_by_occ(&kappas, &[], &omega);
    assert_eq!(sh.len(), 2, "expected 2 occasions");
    assert!(
        (sh[0][0]).abs() < 1e-10,
        "occ 1 shrinkage ~0, got {}",
        sh[0][0]
    );
    assert!(sh[1][0] > 0.8, "occ 2 shrinkage high, got {}", sh[1][0]);
}

#[test]
fn test_kappa_shrinkage_two_kappas_independent() {
    // n_kappa = 2: each kappa parameter should be computed independently.
    // kappa 0: RMS = 1.0 → shrinkage = 0 with omega_00 = 1.0
    // kappa 1: RMS = 0.1 → shrinkage = 1 - 0.1/1.0 = 0.9 with omega_11 = 1.0
    let omega = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
    // Each subject has 1 occasion; kappa vector is [k0_val, k1_val].
    let kappas: Vec<Vec<DVector<f64>>> = vec![
        vec![DVector::from_vec(vec![1.0, 0.1])],
        vec![DVector::from_vec(vec![-1.0, -0.1])],
    ];
    let sh = compute_kappa_shrinkage(&kappas, &[], &omega);
    assert_eq!(sh.len(), 2);
    assert!((sh[0]).abs() < 1e-10, "kappa 0 shrinkage ~0, got {}", sh[0]);
    assert!(
        (sh[1] - 0.9).abs() < 1e-10,
        "kappa 1 shrinkage ~0.9, got {}",
        sh[1]
    );
}

/// #1911: an uninformed group (κ̂ = 0, the prior mode) no longer moves the pooled
/// shrinkage — it equals the informed groups' value alone, bit for bit, while the same
/// κ̂ with the group counted reads higher (the straddle: without it the test would pass
/// on a filter that drops nothing, since an all-informed mask is the old behaviour).
#[test]
fn test_kappa_shrinkage_pooled_skips_uninformed_group() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let with_extra = make_kappas(vec![vec![0.6, 0.0], vec![-0.4, 0.0], vec![0.5, 0.0]]);
    let informed_only = make_kappas(vec![vec![0.6], vec![-0.4], vec![0.5]]);
    let mask = vec![vec![true, false]; 3];
    let filtered = compute_kappa_shrinkage(&with_extra, &mask, &omega);
    let alone = compute_kappa_shrinkage(&informed_only, &[], &omega);
    let unfiltered = compute_kappa_shrinkage(&with_extra, &[], &omega);
    assert_eq!(
        filtered[0].to_bits(),
        alone[0].to_bits(),
        "{filtered:?} vs {alone:?}"
    );
    // mean κ̂² over 6 pairs is half that over 3: 1 − (1 − s)/√2 > s.
    assert!(
        unfiltered[0] > filtered[0] + 0.1,
        "counting the uninformed group must read higher: {unfiltered:?} vs {filtered:?}"
    );
}

/// The all-informed control: a mask of all `true` is bit-identical to no mask, pooled
/// and per slot.
#[test]
fn test_kappa_shrinkage_all_informed_mask_is_identity() {
    let omega = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 1.3]));
    let kappas: Vec<Vec<DVector<f64>>> = (0..4)
        .map(|i| {
            (0..3)
                .map(|g| {
                    DVector::from_vec(vec![
                        0.1 * (i as f64) - 0.17 * g as f64,
                        0.3 - 0.07 * i as f64,
                    ])
                })
                .collect()
        })
        .collect();
    let mask = vec![vec![true; 3]; 4];
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&compute_kappa_shrinkage(&kappas, &mask, &omega)),
        bits(&compute_kappa_shrinkage(&kappas, &[], &omega))
    );
    let a = compute_kappa_shrinkage_by_occ(&kappas, &mask, &omega);
    let b = compute_kappa_shrinkage_by_occ(&kappas, &[], &omega);
    assert_eq!(a.len(), 3);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(bits(x), bits(y));
    }
}

/// A subject with no informed group contributes nothing: the result equals the one
/// without that subject, and a population where no group is informed is NaN (fewer
/// than two pairs), not a 0/0.
#[test]
fn test_kappa_shrinkage_subject_with_no_informed_group_excluded() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let with = make_kappas(vec![vec![0.6, 0.2], vec![-0.4, 0.3], vec![0.0, 0.0]]);
    let without = make_kappas(vec![vec![0.6, 0.2], vec![-0.4, 0.3]]);
    let mask = vec![vec![true, true], vec![true, true], vec![false, false]];
    let a = compute_kappa_shrinkage(&with, &mask, &omega);
    let b = compute_kappa_shrinkage(&without, &[], &omega);
    assert_eq!(a[0].to_bits(), b[0].to_bits(), "{a:?} vs {b:?}");
    let none = vec![vec![false, false]; 3];
    assert!(compute_kappa_shrinkage(&with, &none, &omega)[0].is_nan());
    let by_occ = compute_kappa_shrinkage_by_occ(&with, &none, &omega);
    assert_eq!(by_occ.len(), 2);
    assert!(by_occ.iter().flatten().all(|v| v.is_nan()), "{by_occ:?}");
}

/// Per slot: an uninformed group leaves its slot's value to the informed subjects, and
/// a slot where fewer than two subjects are informed is NaN — the issue's
/// `[[0.912], [1.0]]` (second slot a pk-only-only occasion after every observation)
/// reads `[[…], [NaN]]`.
///
/// Slot 3 is informed **after** the uninformed slot 2 — the reachable order
/// `[obs, obs, dose after the last obs, mid-course pk-only row]`, since groups run
/// observed → dose-only → pk-only. A filtered group keeps its position: slot 3 holds
/// the subjects' slot-3 κ̂ and slot 2 stays NaN. Renumbering after the filter would
/// shift those κ̂ into slot 2 (#1958 review r1 #1).
#[test]
fn test_kappa_shrinkage_by_occ_skips_uninformed_group() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![
        vec![0.6, 0.5, 0.0, 0.7],
        vec![-0.4, -0.3, 0.0, -0.2],
        vec![0.5, 0.0, 0.0, 0.0],
    ]);
    let mask = vec![
        vec![true, true, false, true],
        vec![true, true, false, true],
        vec![true, false, false, false],
    ];
    let sh = compute_kappa_shrinkage_by_occ(&kappas, &mask, &omega);
    assert_eq!(sh.len(), 4, "{sh:?}");
    let alone = |v: Vec<f64>| {
        compute_kappa_shrinkage(
            &make_kappas(v.into_iter().map(|x| vec![x]).collect()),
            &[],
            &omega,
        )[0]
    };
    assert_eq!(
        sh[1][0].to_bits(),
        alone(vec![0.5, -0.3]).to_bits(),
        "{sh:?}"
    );
    assert!(
        sh[2][0].is_nan(),
        "uninformed slot 2 must stay empty: {sh:?}"
    );
    assert_eq!(
        sh[3][0].to_bits(),
        alone(vec![0.7, -0.2]).to_bits(),
        "{sh:?}"
    );
    // Slot 0 is all informed: unchanged.
    let unmasked = compute_kappa_shrinkage_by_occ(&kappas, &[], &omega);
    assert_eq!(sh[0][0].to_bits(), unmasked[0][0].to_bits());
}

/// The helper `fit()` calls derives the mask from the model and the population. Three
/// subjects observe OCC 1 (t = 1) and OCC 2 (t = 2); a pk-only row adds OCC 3.
///
/// - Trailing (t = 5, the #1906 `iov_pk_only_*_AT` shape): uninformed on the PK model,
///   so the pooled and per-slot values equal the two observed groups' alone.
/// - Mid-course (t = 1.5): on the PK model the interval it closes reaches the t = 2
///   observation, so it counts (the unmasked value); on the compartment-free MBMA model
///   nothing carries over, so it does not (the observed-only value). One fixture, both
///   sides of `is_algebraic()`.
#[test]
fn test_kappa_shrinkages_reads_informed_rule_from_population() {
    use crate::parser::model_parser::parse_model_string;
    let pk = parse_model_string(include_str!("../../../examples/warfarin_iov.ferx")).unwrap();
    let pred = parse_model_string(include_str!(
        "../../../tests/data/mbma_placebo/mbma_placebo.ferx"
    ))
    .unwrap();
    assert!(!pk.is_algebraic() && pred.is_algebraic());
    let pop = |pk_only_at: Option<f64>| {
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            dose_occasions: vec![1],
            obs_times: vec![1.0, 2.0],
            observations: vec![1.0, 1.0],
            occasions: vec![1, 2],
            pk_only_times: pk_only_at.into_iter().collect(),
            pk_only_occasions: pk_only_at.map(|_| 3).into_iter().collect(),
            ..Subject::default()
        };
        Population {
            subjects: vec![subject; 3],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    };
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let observed = make_kappas(vec![vec![0.6, 0.2], vec![-0.4, 0.3], vec![0.5, -0.1]]);
    let with_pk = make_kappas(vec![
        vec![0.6, 0.2, 0.0],
        vec![-0.4, 0.3, 0.0],
        vec![0.5, -0.1, 0.0],
    ]);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (base, base_occ) = kappa_shrinkages(&pk, &pop(None), &observed, &omega);
    // Control: the all-observed population is the unmasked value.
    assert_eq!(
        bits(&base),
        bits(&compute_kappa_shrinkage(&observed, &[], &omega))
    );
    let unmasked = compute_kappa_shrinkage(&with_pk, &[], &omega);
    assert!(unmasked[0] > base[0] + 0.05, "{unmasked:?} vs {base:?}");

    let (at, at_occ) = kappa_shrinkages(&pk, &pop(Some(5.0)), &with_pk, &omega);
    assert_eq!(bits(&at), bits(&base), "trailing pk-only row, PK model");
    assert_eq!(at_occ.len(), 3);
    assert!(at_occ[2][0].is_nan(), "{at_occ:?}");
    for s in 0..2 {
        assert_eq!(bits(&at_occ[s]), bits(&base_occ[s]));
    }

    let (mid_pk, _) = kappa_shrinkages(&pk, &pop(Some(1.5)), &with_pk, &omega);
    assert_eq!(
        bits(&mid_pk),
        bits(&unmasked),
        "mid-course pk-only row, PK model"
    );
    let (mid_pred, _) = kappa_shrinkages(&pred, &pop(Some(1.5)), &with_pk, &omega);
    assert_eq!(
        bits(&mid_pred),
        bits(&base),
        "mid-course pk-only row, MBMA model"
    );
}

#[test]
fn test_eps_shrinkage_ignores_nan_iwres() {
    // Censored rows have NaN IWRES — they must be filtered out.
    // After filtering, two values with mean(IWRES^2)=1 remain => shrinkage = 0.
    let subjects = vec![
        make_subject(vec![0.0], vec![1.0, f64::NAN]),
        make_subject(vec![0.0], vec![-1.0, f64::NAN]),
    ];
    let sh = compute_eps_shrinkage(&subjects);
    assert!((sh).abs() < 1e-10, "NaN IWRES not filtered, got {}", sh);
}

// ── #619 groups the run actually builds (#918 review) ────────────────────

/// `TVCL + (CRCL−90)·TH_CRCL` on `ETA_CL`, with the IIV variance left as a
/// parameter so the weak-IIV drop can be exercised.
fn additive_group_model(eta_cl_var: f64) -> crate::types::CompiledModel {
    let src = format!(
        r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ {eta_cl_var}
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
    );
    crate::parser::model_parser::parse_model_string(&src).expect("model parses")
}

fn crcl_population() -> Population {
    use std::io::Write;
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,CRCL\n");
    for (i, c) in [60.0f64, 90.0, 120.0].iter().enumerate() {
        let id = i + 1;
        csv.push_str(&format!("{id},0,0,100,1,1,{c}\n{id},1,5.0,0,0,1,{c}\n"));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), Some(&["CRCL"]), None).unwrap()
}

/// The pre-run #621 warning has to ask what the estimator will *do*, not what
/// the parser found. `resolve_covariate_mu_groups` drops a group whose eta has
/// negligible IIV, and then nothing mu-references `CL` — so the warning must
/// fire, exactly as it does when `mu_referencing` is off.
///
/// The healthy-IIV arm is the straddle: same model, same code path, group
/// alive, no warning. Without it a fix that returned `&[]` unconditionally
/// would pass.
#[test]
fn saem_active_covariate_group_etas_drops_a_weak_iiv_group() {
    let pop = crcl_population();
    let options = FitOptions {
        mu_referencing: true,
        ..Default::default()
    };

    let healthy = additive_group_model(0.2);
    assert_eq!(
        healthy.covariate_mu_refs.len(),
        1,
        "premise: parsed a group"
    );
    let etas = saem_active_covariate_group_etas(&healthy, &pop, &healthy.default_params, &options);
    assert_eq!(etas, vec!["ETA_CL"], "a healthy group is built");
    assert!(
        saem_non_mu_referenced_individual_params_warning(&healthy, &etas).is_none(),
        "CL is mu-referenced through the group"
    );

    // ω² = 1e-4 is below WEAK_GROUP_IIV_VAR (1e-3).
    let weak = additive_group_model(0.0001);
    assert_eq!(weak.covariate_mu_refs.len(), 1, "premise: still parsed");
    let etas = saem_active_covariate_group_etas(&weak, &pop, &weak.default_params, &options);
    assert!(
        etas.is_empty(),
        "the resolver drops it for negligible IIV, got {etas:?}"
    );
    let w = saem_non_mu_referenced_individual_params_warning(&weak, &etas)
        .expect("with no group built, nothing mu-references CL");
    assert!(w.contains("CL"), "{w}");
    assert!(
        !w.contains(" V"),
        "V keeps its own single-anchor mu-ref: {w}"
    );
}

/// The two outer gates `run_saem` applies before it resolves anything.
#[test]
fn saem_active_covariate_group_etas_honours_the_two_outer_gates() {
    let pop = crcl_population();
    let m = additive_group_model(0.2);

    let off = FitOptions {
        mu_referencing: false,
        ..Default::default()
    };
    assert!(
        saem_active_covariate_group_etas(&m, &pop, &m.default_params, &off).is_empty(),
        "no group under mu_referencing = false"
    );

    let mut mixture = additive_group_model(0.2);
    mixture.mixture = Some(crate::types::MixtureSpec {
        n_classes: 2,
        mixing: Vec::new(),
        logit_covariates: Vec::new(),
        omega_overrides: Vec::new(),
        sigma_overrides: Vec::new(),
        mu_refs: Vec::new(),
    });
    let on = FitOptions {
        mu_referencing: true,
        ..Default::default()
    };
    assert!(
        saem_active_covariate_group_etas(&mixture, &pop, &mixture.default_params, &on).is_empty(),
        "no group under a mixture"
    );
}
