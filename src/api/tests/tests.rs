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
    // `0` must be rejected rather than silently forwarded to Rayon, which would
    // otherwise interpret it as "pick automatically" and mask the caller's intent
    // (and would incorrectly mark GLOBAL_THREADS_EXPLICIT). Returns before touching
    // the process-wide pool, so this is safe to run alongside other tests.
    assert!(configure_global_thread_pool(0).is_err());
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
        cens: vec![0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
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
    assert!(
        err.to_lowercase().contains("diffusion")
            || err.to_lowercase().contains("sde")
            || err.to_lowercase().contains("stochastic"),
        "error should cite the SDE/diffusion restriction: {err}"
    );
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
            log_transformed: true,
        },
    );

    let warning =
        saem_non_mu_referenced_individual_params_warning(&model).expect("expected warning");
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
            log_transformed: true,
        },
    );
    model.mu_refs.insert(
        "ETA_V".into(),
        MuRef {
            theta_name: "TVV".into(),
            log_transformed: true,
        },
    );

    assert!(saem_non_mu_referenced_individual_params_warning(&model).is_none());
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
    let sh = compute_kappa_shrinkage(&kappas, &omega);
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
    let sh = compute_kappa_shrinkage(&kappas, &omega);
    assert!(sh[0] > 0.9, "expected high shrinkage, got {}", sh[0]);
}

#[test]
fn test_kappa_shrinkage_pooled_nan_when_omega_zero() {
    let omega = DMatrix::zeros(1, 1);
    let kappas = make_kappas(vec![vec![0.1], vec![-0.1]]);
    let sh = compute_kappa_shrinkage(&kappas, &omega);
    assert!(sh[0].is_nan());
}

#[test]
fn test_kappa_shrinkage_pooled_nan_when_fewer_than_2_obs() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![vec![0.5]]);
    let sh = compute_kappa_shrinkage(&kappas, &omega);
    assert!(sh[0].is_nan());
}

#[test]
fn test_kappa_shrinkage_by_occ_returns_empty_for_single_occasion() {
    let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
    let kappas = make_kappas(vec![vec![0.5], vec![-0.5]]);
    let sh = compute_kappa_shrinkage_by_occ(&kappas, &omega);
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
    let sh = compute_kappa_shrinkage_by_occ(&kappas, &omega);
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
    let sh = compute_kappa_shrinkage(&kappas, &omega);
    assert_eq!(sh.len(), 2);
    assert!((sh[0]).abs() < 1e-10, "kappa 0 shrinkage ~0, got {}", sh[0]);
    assert!(
        (sh[1] - 0.9).abs() < 1e-10,
        "kappa 1 shrinkage ~0.9, got {}",
        sh[1]
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
