//! End-to-end checks for θ level blocks (#1064).
//!
//! The declaration syntax, the gather evaluator, the identifiability
//! conventions, and the scale guards are unit-tested in `src/`. What only the
//! file entry points can exercise is the *binding*: a level block's
//! level count is a property of the dataset, discovered after the CSV is read
//! and folded back into the model by a re-parse. These tests run that whole
//! path and stop after a couple of outer iterations (Tier 2 — no convergence).

use ferx_core::run_model_with_data;
use std::io::Write;
use std::path::PathBuf;

/// Two studies × three timepoints, one subject per study, plus a `PLA_IDX`
/// column giving the same design in the explicit 1-based form.
const DATA: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY,PLA_IDX
1,0,.,1,100,1,0,1,1,1
1,1,8.1,0,.,1,0,0,1,1
1,4,6.2,0,.,1,0,0,1,2
1,12,3.1,0,.,1,0,0,1,3
2,0,.,1,100,1,0,1,2,4
2,1,7.4,0,.,1,0,0,2,4
2,4,5.5,0,.,1,0,0,2,5
2,12,2.8,0,.,1,0,0,2,6
";

/// `PLA_IDX` written 0-based — the single most likely user error, and one the
/// evaluator's NaN guard alone would report as "the fit diverged".
const DATA_ZERO_BASED: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY,PLA_IDX
1,0,.,1,100,1,0,1,1,0
1,1,8.1,0,.,1,0,0,1,0
1,4,6.2,0,.,1,0,0,1,1
1,12,3.1,0,.,1,0,0,1,2
2,0,.,1,100,1,0,1,2,3
2,1,7.4,0,.,1,0,0,2,3
2,4,5.5,0,.,1,0,0,2,4
2,12,2.8,0,.,1,0,0,2,5
";

const FIT_OPTIONS: &str = "
[fit_options]
  maxiter = 2
  inner_maxiter = 3
  covariance = false
";

/// A `[STUDY, TIME]` placebo effect with no random effect on the same
/// scale — global sum-to-zero.
fn level_block_model() -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.001, 20.0)
  theta PLACEBO[STUDY, TIME](0.0, -5.0, 5.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.05

[individual_parameters]
  CL = TVCL + PLACEBO
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
{FIT_OPTIONS}"#
    )
}

/// The same design written in the explicit form, indexed by a data column.
fn explicit_model(levels: usize) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.001, 20.0)
  theta PLACEBO[{levels}](0.0, -5.0, 5.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.05

[individual_parameters]
  CL = TVCL + PLACEBO[PLA_IDX]
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
{FIT_OPTIONS}"#
    )
}

/// Write `model` and `data` into a fresh temp dir and return their paths. The
/// `TempDir` is returned too — dropping it deletes the files.
fn write_case(model: &str, data: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = dir.path().join("m.ferx");
    let data_path = dir.path().join("d.csv");
    write!(std::fs::File::create(&model_path).unwrap(), "{model}").unwrap();
    write!(std::fs::File::create(&data_path).unwrap(), "{data}").unwrap();
    (dir, model_path, data_path)
}

#[test]
fn a_level_block_binds_against_the_dataset_end_to_end() {
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let (result, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("fit");

    // 2 studies x 3 timepoints = 6 observed levels; global sum-to-zero leaves 5.
    assert_eq!(result.theta_names.len(), 2 + 5, "{:?}", result.theta_names);
    assert_eq!(result.theta_names[0], "TVCL");
    assert_eq!(result.theta_names[1], "PLACEBO[STUDY=1,TIME=1]");
    assert_eq!(result.theta_names[5], "PLACEBO[STUDY=2,TIME=4]");
    assert_eq!(result.theta_names[6], "TVV");
    assert!(
        result.theta.iter().all(|t| t.is_finite()),
        "every estimate must be finite: {:?}",
        result.theta
    );
}

/// As [`DATA`], but study 2 is never sampled at TIME 1 — an unbalanced design,
/// which is the norm in a meta-analysis.
const DATA_SPARSE: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY,PLA_IDX
1,0,.,1,100,1,0,1,1,1
1,1,8.1,0,.,1,0,0,1,1
1,4,6.2,0,.,1,0,0,1,2
1,12,3.1,0,.,1,0,0,1,3
2,0,.,1,100,1,0,1,2,4
2,4,5.5,0,.,1,0,0,2,4
2,12,2.8,0,.,1,0,0,2,5
";

#[test]
fn a_level_the_data_never_shows_is_not_estimated() {
    // One θ per *observed* combination, not per cell of the full grid: study 2
    // is never sampled at TIME 1, so that cell must not become a parameter with
    // nothing to inform it.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA_SPARSE);
    let (result, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("fit");
    assert!(
        !result
            .theta_names
            .iter()
            .any(|n| n == "PLACEBO[STUDY=2,TIME=1]"),
        "{:?}",
        result.theta_names
    );
    // 5 observed levels, global sum-to-zero leaves 4.
    assert_eq!(result.theta_names.len(), 2 + 4, "{:?}", result.theta_names);
}

#[test]
fn the_explicit_gather_form_fits_through_the_same_path() {
    let (_dir, model_path, data_path) = write_case(&explicit_model(6), DATA);
    let (result, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("fit");
    assert_eq!(result.theta_names.len(), 2 + 6);
    assert_eq!(result.theta_names[1], "PLACEBO[1]");
    assert!(result.theta.iter().all(|t| t.is_finite()));
}

#[test]
fn a_zero_based_index_column_fails_loudly_before_the_fit() {
    let (_dir, model_path, data_path) = write_case(&explicit_model(6), DATA_ZERO_BASED);
    let err = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect_err("a 0-based index must be rejected");
    assert!(
        err.contains("1-based") && err.contains("PLA_IDX"),
        "the error must name the column and the convention, got: {err}"
    );
}

#[test]
fn an_index_past_the_declared_level_count_fails_loudly() {
    // The data reaches level 6, the model declares 4.
    let (_dir, model_path, data_path) = write_case(&explicit_model(4), DATA);
    let err = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect_err("an out-of-range index must be rejected");
    assert!(
        err.contains("has 4 levels"),
        "the error must name the declared count, got: {err}"
    );
}

#[test]
fn a_level_block_model_cannot_be_fit_without_binding() {
    // The in-memory `fit()` entry point never sees the data before the model is
    // compiled, so it must refuse rather than gather out of an empty level
    // table and predict NaN.
    let parsed =
        ferx_core::parser::model_parser::parse_full_model(&level_block_model()).expect("parse");
    let population =
        ferx_core::read_nonmem_csv(std::path::Path::new("data/warfarin.csv"), None, None)
            .expect("read warfarin");
    let err = ferx_core::fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &ferx_core::FitOptions::default(),
    )
    .expect_err("an unbound level block must not fit");
    assert!(
        err.contains("never bound to data") && err.contains("PLACEBO"),
        "unexpected error: {err}"
    );
}

#[test]
fn binding_is_deterministic_so_predict_rebuilds_the_same_levels() {
    // The level → index map is what `predict()` / `simulate()` must rebuild
    // identically after a fit. It is not stored: it is *derived*, from the
    // observed combinations in a fixed sort order. So the round-trip guarantee
    // is that binding the same data twice gives the same map — including the
    // synthesized index column, level for level.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let text = std::fs::read_to_string(&model_path).unwrap();

    let bind_once = || {
        let mut parsed = ferx_core::parser::model_parser::parse_full_model(&text).expect("parse");
        let mut population = ferx_core::read_nonmem_csv(&data_path, None, None).expect("read");
        ferx_core::bind_theta_levels(&mut parsed, &text, &mut population).expect("bind");
        (parsed.model, population)
    };

    let (model_a, pop_a) = bind_once();
    let (model_b, pop_b) = bind_once();

    assert_eq!(model_a.theta_names, model_b.theta_names);
    assert_eq!(
        ferx_core::theta_level_map(&model_a),
        ferx_core::theta_level_map(&model_b)
    );
    for (a, b) in pop_a.subjects.iter().zip(&pop_b.subjects) {
        let idx = |s: &ferx_core::Subject| -> Vec<f64> {
            s.obs_covariates
                .iter()
                .map(|m| m["__level_PLACEBO"])
                .collect()
        };
        assert_eq!(idx(a), idx(b), "subject {} index column drifted", a.id);
    }
}

#[test]
fn predict_runs_on_a_bound_level_block_model() {
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let text = std::fs::read_to_string(&model_path).unwrap();
    let mut parsed = ferx_core::parser::model_parser::parse_full_model(&text).expect("parse");
    let mut population = ferx_core::read_nonmem_csv(&data_path, None, None).expect("read");
    ferx_core::bind_theta_levels(&mut parsed, &text, &mut population).expect("bind");

    let preds = ferx_core::predict(&parsed.model, &population, &parsed.model.default_params);
    assert_eq!(preds.len(), 6, "one prediction per observation");
    assert!(
        preds.iter().all(|p| p.pred.is_finite() && p.pred > 0.0),
        "a bound gather must predict finite values: {:?}",
        preds.iter().map(|p| p.pred).collect::<Vec<_>>()
    );
}

#[test]
fn predict_refuses_a_population_that_was_never_bound() {
    // The synthesized index column is a required covariate of the bound model,
    // so a population read fresh — with no `__level_*` column — must fail
    // loudly rather than gather at index 0 and return NaN.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let text = std::fs::read_to_string(&model_path).unwrap();
    let mut parsed = ferx_core::parser::model_parser::parse_full_model(&text).expect("parse");
    let mut bound = ferx_core::read_nonmem_csv(&data_path, None, None).expect("read");
    ferx_core::bind_theta_levels(&mut parsed, &text, &mut bound).expect("bind");

    let unbound = ferx_core::read_nonmem_csv(&data_path, None, None).expect("read");
    let params = parsed.model.default_params.clone();
    let model = parsed.model;
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ferx_core::predict(&model, &unbound, &params)
    }))
    .is_err();
    assert!(
        panicked,
        "predict() must refuse a population missing the level index column"
    );
}

#[test]
fn a_level_block_binds_against_a_simulation_design() {
    // `--simulate` has no dataset, so the levels come from the `[simulation]`
    // covariates (#1083) and the observation grid instead. Every level takes
    // the declaration's broadcast init, since the DSL has no way to state
    // per-level simulation values.
    let model = r#"
[parameters]
  theta TVCL(2.0, 0.001, 20.0)
  theta PLACEBO[STUDY, TIME](0.0, -5.0, 5.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.05

[individual_parameters]
  CL = TVCL + PLACEBO
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[simulation]
  n_subjects = 3
  dose_amt   = 100
  dose_cmt   = 1
  seed       = 7
  times      = [1, 4, 12]
  covariate STUDY = [1, 2, 3]
"#;
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = dir.path().join("sim.ferx");
    write!(std::fs::File::create(&model_path).unwrap(), "{model}").unwrap();

    let (result, population) =
        ferx_core::run_model_simulate(model_path.to_str().unwrap()).expect("simulate");

    // 3 studies x 3 timepoints, global sum-to-zero leaves 8 free levels.
    assert_eq!(result.theta_names.len(), 2 + 8, "{:?}", result.theta_names);
    assert_eq!(result.theta_names[1], "PLACEBO[STUDY=1,TIME=1]");
    for subject in &population.subjects {
        assert!(subject.covariates.contains_key("__level_PLACEBO"));
        assert!(
            subject.observations.iter().all(|v| v.is_finite()),
            "a bound level block must simulate finite observations"
        );
    }
}
