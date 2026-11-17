//! Tier-2: SAEM's exploration phase ends by a convergence test unless
//! `n_exploration` fixes it, and the fit reports the length it used.
//!
//! The rule itself (trend test, window, calibration) is unit-tested in
//! `estimation/saem_exploration_tests.rs`; these tests pin the wiring through
//! `fit()`: precedence of the explicit length, the minimum/maximum, the
//! `FitResult::saem_n_exploration_used` report, the maximum-reached warning, and
//! the smoothing exponent. Every fit returns after at most ~150 iterations of a
//! 40-subject model.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaMatrix, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions, FitResult};

mod common;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  1.0, 500.0)
  omega ETA_CL ~ 0.10
  omega ETA_V  ~ 0.10
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn population(model: &ferx_core::types::CompiledModel) -> Population {
    let times = vec![0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0];
    let subjects = (1..=40)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                times.clone(),
                vec![0.0; times.len()],
                vec![1; times.len()],
            )
        })
        .collect();
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let mut truth = model.default_params.clone();
    truth.theta = vec![5.0, 50.0];
    truth.omega = OmegaMatrix::from_diagonal(&[0.09, 0.09], vec!["ETA_CL".into(), "ETA_V".into()]);
    truth.sigma.values = vec![0.10];
    let sim = simulate_with_seed(model, &template, &truth, 1, 20240527).unwrap();
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
    }
    pop
}

fn run(edit: impl FnOnce(&mut FitOptions)) -> FitResult {
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = population(&model);
    let mut o = FitOptions::default();
    o.method = EstimationMethod::Saem;
    o.saem_n_convergence = 10;
    o.saem_seed = Some(11);
    o.run_covariance_step = false;
    o.verbose = false;
    edit(&mut o);
    fit(&model, &pop, &model.default_params, &o).expect("SAEM fit must succeed")
}

fn max_warning(r: &FitResult) -> bool {
    r.warnings
        .iter()
        .any(|w| w.contains("exploration phase reached its maximum"))
}

/// An explicit `n_exploration` is a fixed length: it is used as given, the
/// min/max are ignored (here deliberately set to contradict it), and no
/// maximum-reached warning is raised.
#[test]
fn explicit_n_exploration_is_a_fixed_length() {
    let r = run(|o| {
        o.saem_n_exploration = Some(12);
        o.saem_exploration_min = 60;
        o.saem_exploration_max = 90;
    });
    assert_eq!(r.saem_n_exploration_used, Some(12));
    assert!(!max_warning(&r), "{:?}", r.warnings);
}

/// The automatic stop never ends before the minimum and never exceeds the
/// maximum. With `max = min` the trend window (100 iterations) can never fill,
/// so the run is cut off by the maximum and says so.
#[test]
fn maximum_is_respected_and_reported() {
    let r = run(|o| {
        o.saem_exploration_min = 15;
        o.saem_exploration_max = 15;
    });
    assert_eq!(r.saem_n_exploration_used, Some(15));
    assert!(max_warning(&r), "{:?}", r.warnings);
}

/// Stationary, well-identified model: the rule needs one full window, so the
/// run ends between the window length and the maximum, and the maximum-reached
/// warning appears exactly when the length equals the maximum.
#[test]
fn automatic_stop_lies_between_the_bounds() {
    let r = run(|o| {
        o.saem_exploration_min = 20;
        o.saem_exploration_max = 160;
    });
    let used = r
        .saem_n_exploration_used
        .expect("SAEM reports its exploration length");
    // Measured: this stationary fixture stops at 100, the first iteration a full trend window
    // exists. Requiring a stop strictly before the maximum is what makes the test fail if the
    // automatic stop is never taken (which would also yield a length inside the bounds).
    assert!((100..160).contains(&used), "used {used}");
    assert!(!max_warning(&r), "{:?}", r.warnings);
}

/// The report is SAEM-only and survives to the result of the default
/// configuration (automatic, 150..500).
#[test]
fn default_is_automatic_and_non_saem_reports_nothing() {
    assert!(FitOptions::default().saem_n_exploration.is_none());
    let model = parse_model_string(MODEL).unwrap();
    let pop = population(&model);
    let mut o = FitOptions::default();
    o.method = EstimationMethod::FoceI;
    o.run_covariance_step = false;
    o.outer_maxiter = 3;
    let r = fit(&model, &pop, &model.default_params, &o).unwrap();
    assert_eq!(r.saem_n_exploration_used, None);
}

/// `smoothing_exponent = 1` is the classical schedule bit for bit; `0.7`
/// changes the convergence-phase averaging and therefore the estimates.
#[test]
fn smoothing_exponent_one_is_the_default_and_seven_tenths_differs() {
    let fixed = |o: &mut FitOptions| {
        o.saem_n_exploration = Some(10);
        o.saem_n_convergence = 15;
    };
    let base = run(fixed);
    let one = run(|o| {
        fixed(o);
        o.saem_smoothing_exponent = 1.0;
    });
    let sev = run(|o| {
        fixed(o);
        o.saem_smoothing_exponent = 0.7;
    });
    assert_eq!(base.theta, one.theta);
    assert_eq!(base.omega.diagonal(), one.omega.diagonal());
    assert_ne!(base.theta, sev.theta);
}
