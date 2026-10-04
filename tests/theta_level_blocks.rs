//! End-to-end checks for θ level blocks (#1064).
//!
//! The declaration syntax, the gather evaluator, the identifiability
//! conventions, and the scale guards are unit-tested in `src/`. What only the
//! file entry points can exercise is the *binding*: a level block's
//! level count is a property of the dataset, discovered after the CSV is read
//! and folded back into the model by a re-parse. These tests run that whole
//! path and stop after a couple of outer iterations (Tier 2 — no convergence).

use ferx_core::{run_model_with_data, validate_model_file};
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
    // One assertion per sentence of the message (#1384): each one dies when
    // its sentence is deleted.
    assert!(
        err.contains("`theta PLACEBO[...]` was never bound to data"),
        "the block and the cause: {err}"
    );
    assert!(
        err.contains("`bind_theta_levels(&mut parsed, &model_text, &mut population)`")
            && err.contains("(`read_population_for`) and before `fit`"),
        "the public binder, and when to call it: {err}"
    );
    assert!(
        err.contains("`prepare_run` and the file entry points") && err.contains("bind for you"),
        "the entry points that bind on their own: {err}"
    );
    assert!(
        err.contains("`theta PLACEBO[N](...)` and index it with your own column"),
        "the counted-form alternative: {err}"
    );
}

#[test]
fn the_unbound_refusal_names_every_unbound_block() {
    // Two level blocks: the message must name both, and its counted-form
    // example uses the first.
    let model = level_block_model()
        .replace(
            "theta TVV(10.0, 0.1, 500.0)",
            "theta TVV(10.0, 0.1, 500.0)\n  theta DRUG[STUDY](0.0, -5.0, 5.0)",
        )
        .replace("V  = TVV * exp(ETA_V)", "V  = TVV * exp(ETA_V) + DRUG");
    let parsed = ferx_core::parser::model_parser::parse_full_model(&model).expect("parse");
    assert_eq!(
        parsed.model.theta_blocks().unbound_level_blocks().len(),
        2,
        "the fixture must declare two unbound blocks"
    );
    let population =
        ferx_core::read_nonmem_csv(std::path::Path::new("data/warfarin.csv"), None, None)
            .expect("read warfarin");
    let err = ferx_core::fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &ferx_core::FitOptions::default(),
    )
    .expect_err("unbound level blocks must not fit");
    assert!(
        err.contains("`theta PLACEBO`, `theta DRUG[...]` was never bound to data"),
        "both block names: {err}"
    );
    assert!(
        err.contains("`theta PLACEBO[N](...)`"),
        "the counted example uses the first block: {err}"
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
fn check_binds_level_blocks_before_data_validation() {
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let report = validate_model_file(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    );
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.message.contains("__level_PLACEBO")),
        "the synthesized level index must exist before checks run: {:?}",
        report.diagnostics
    );
    assert!(
        report.valid,
        "valid bound model should pass check: {:?}",
        report.diagnostics
    );
}

#[test]
fn check_reports_level_binding_errors_directly() {
    let invalid = level_block_model().replace(
        "PLACEBO[STUDY, TIME](0.0, -5.0, 5.0)",
        "PLACEBO[STUDY, TIME](0.5, -5.0, 5.0)",
    );
    let (_dir, model_path, data_path) = write_case(&invalid, DATA);
    let report = validate_model_file(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    );
    assert!(!report.valid);
    assert!(report
        .diagnostics
        .iter()
        .any(|d| { d.code == "E_THETA_LEVEL_BINDING" && d.message.contains("requires init = 0") }));
}

#[test]
fn predict_runs_on_a_bound_level_block_model() {
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let text = std::fs::read_to_string(&model_path).unwrap();
    let mut parsed = ferx_core::parser::model_parser::parse_full_model(&text).expect("parse");
    let mut population = ferx_core::read_nonmem_csv(&data_path, None, None).expect("read");
    ferx_core::bind_theta_levels(&mut parsed, &text, &mut population).expect("bind");

    let preds =
        ferx_core::predict(&parsed.model, &population, &parsed.model.default_params).unwrap();
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
        ferx_core::predict(&model, &unbound, &params).unwrap()
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

/// Parse `model_path` and read `data_path` the way an API caller (the R glue,
/// a tool) does it — `parse_full_model_file` → `read_population_for` with the
/// file's IOV column and `[data_selection]` filter — then `bind_theta_levels`
/// unless `bind` is false (#1384), then the file's `gradient = ...` stamped onto
/// the model. Statement for statement the documented example
/// (`docs/api/fitting.qmd`, `docs/api/index.qmd`), imports included.
fn read_composable(
    model_path: &std::path::Path,
    data_path: &std::path::Path,
    bind: bool,
) -> (ferx_core::ParsedModel, ferx_core::Population) {
    use ferx_core::io::datareader::SelectionFilter;
    use ferx_core::GradientMethod;

    let mut parsed = ferx_core::parse_full_model_file(model_path).expect("parse");
    let opts = &parsed.fit_options;
    let filter = SelectionFilter::from_opts(
        &opts.ignore_exprs,
        &opts.accept_exprs,
        &opts.ignore_subjects,
    )
    .expect("selection filter");
    let (mut population, _) = ferx_core::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data_path.to_str().unwrap(),
        None,
        opts.iov_column.as_deref(),
        Some(&filter),
        &parsed.column_map,
    )
    .expect("read");
    if bind {
        let model_text = std::fs::read_to_string(model_path).unwrap();
        ferx_core::bind_theta_levels(&mut parsed, &model_text, &mut population).expect("bind");
        // A no-op on this model; called so the helper is the documented example.
        ferx_core::api::bind_covariate_stats(&mut parsed, &model_text, &population)
            .expect("bind covariate stats");
    }
    // `fit` reads the gradient method off the model, not the options, and an
    // SDE model is always FD. After the binds, which re-parse the model.
    parsed.model.gradient_method = if parsed.model.is_sde() {
        GradientMethod::Fd
    } else {
        parsed.fit_options.gradient_method
    };
    (parsed, population)
}

/// [`read_composable`], then `fit` with the parsed model's own inits and options.
fn fit_composable(
    model_path: &std::path::Path,
    data_path: &std::path::Path,
    bind: bool,
) -> Result<(ferx_core::FitResult, ferx_core::CompiledModel), String> {
    let (parsed, population) = read_composable(model_path, data_path, bind);
    let result = ferx_core::fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &parsed.fit_options,
    )?;
    Ok((result, parsed.model))
}

#[test]
fn the_composable_path_binds_exactly_like_the_file_entry_point() {
    // #1384: `bind_theta_levels` is public, so a caller that reads the data
    // itself can fit a level-block model without a file entry point. The file
    // path is the oracle: the same parameter vector, and an OFV equal to the bit.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let (file, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("file fit");
    let (composable, model) = fit_composable(&model_path, &data_path, true).expect("composable");

    assert_eq!(composable.theta_names, file.theta_names);
    assert_eq!(composable.n_parameters, file.n_parameters);
    assert_eq!(composable.n_parameters, 9, "{:?}", composable.theta_names);
    assert!(file.ofv.is_finite(), "file-path OFV must be finite");
    assert_eq!(
        composable.ofv.to_bits(),
        file.ofv.to_bits(),
        "composable OFV {:.17e} vs file OFV {:.17e}",
        composable.ofv,
        file.ofv
    );
    // `theta_names` omits the dependent level under sum-to-zero; the level map
    // does not — all 6 observed combinations.
    let map = ferx_core::theta_level_map(&model);
    let labels = map.get("PLACEBO").expect("PLACEBO in the level map");
    assert_eq!(labels.len(), 6, "{labels:?}");
    assert_eq!(labels[0], "STUDY=1,TIME=1");
    let dependent: Vec<_> = labels
        .iter()
        .filter(|l| !composable.theta_names.contains(&format!("PLACEBO[{l}]")))
        .collect();
    assert_eq!(dependent.len(), 1, "one dependent level: {labels:?}");
}

/// Fit `model` on [`DATA`] through `run_model_with_data` and through the
/// composable path, and return `(file, composable)` OFV and parameter count.
fn file_and_composable(model: &str) -> ((f64, usize), (f64, usize)) {
    let (_dir, model_path, data_path) = write_case(model, DATA);
    let (file, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("file fit");
    let (composable, _model) = fit_composable(&model_path, &data_path, true).expect("composable");
    (
        (file.ofv, file.n_parameters),
        (composable.ofv, composable.n_parameters),
    )
}

#[test]
fn the_composable_path_honours_the_files_gradient_setting() {
    // `fit` reads `gradient` off the model, which only the file entry points
    // stamp, so the documented path has to stamp it too. Before it did, the
    // composable fit here was bit-identical to the run *without* `gradient = fd`.
    let fd_model =
        level_block_model().replace("covariance = false", "covariance = false\n  gradient = fd");
    assert_ne!(fd_model, level_block_model(), "the replace must take");
    let ((file_ofv, file_n), (comp_ofv, comp_n)) = file_and_composable(&fd_model);
    let ((base_ofv, _), _) = file_and_composable(&level_block_model());
    // The straddle: `gradient = fd` must move the file path's objective, or the
    // equality below holds whether the composable path stamps it or not.
    assert!(file_ofv.is_finite() && base_ofv.is_finite());
    assert_ne!(
        file_ofv.to_bits(),
        base_ofv.to_bits(),
        "gradient = fd must change the file-path objective on this fixture"
    );
    assert_eq!(comp_n, file_n);
    assert_eq!(
        comp_ofv.to_bits(),
        file_ofv.to_bits(),
        "gradient = fd: composable OFV {comp_ofv:.17e} vs file OFV {file_ofv:.17e}"
    );
}

#[test]
fn the_composable_path_applies_the_files_data_selection() {
    // `ignore = TIME > 10` drops both 12 h records, so 4 observed combinations
    // remain: 3 free levels + TVCL + TVV + ω + σ = 7 parameters, against 9 on
    // the full data. The file path filters; the composable path must too.
    let model = format!(
        "{}\n[data_selection]\n  ignore = TIME > 10\n",
        level_block_model()
    );
    let ((file_ofv, file_n), (comp_ofv, comp_n)) = file_and_composable(&model);
    assert_eq!(file_n, 7, "the file path must apply [data_selection]");
    assert_eq!(comp_n, file_n, "composable parameter count vs file");
    assert!(file_ofv.is_finite());
    assert_eq!(
        comp_ofv.to_bits(),
        file_ofv.to_bits(),
        "[data_selection]: composable OFV {comp_ofv:.17e} vs file OFV {file_ofv:.17e}"
    );
}

#[test]
fn the_composable_path_without_binding_is_refused_by_name() {
    // The other side of the gate above: the same path minus the bind call.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let err = fit_composable(&model_path, &data_path, false)
        .map(|_| ())
        .expect_err("an unbound level block must not fit");
    assert!(
        err.contains("never bound to data") && err.contains("bind_theta_levels"),
        "unexpected error: {err}"
    );
}

#[test]
fn the_composable_path_writes_the_level_index_the_data_encodes() {
    // Every PLACEBO init is 0.0, so at the initial estimates any permutation of
    // the level index predicts identically and the OFV equality above cannot
    // see a wrong index. Check the synthesized column itself, record by record,
    // against `PLA_IDX` — the dataset's own hand-written encoding of the same
    // design, in the binder's (STUDY, TIME) sort order — a reference outside the
    // binder. `prepare_run` is compared too, but it shares the binder's index
    // writer, so that leg pins routing, not the index.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let prepared = ferx_core::prepare_run(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("prepare_run");
    let (_parsed, population) = read_composable(&model_path, &data_path, true);

    assert_eq!(population.subjects.len(), 2);
    assert_eq!(prepared.population.subjects.len(), 2);
    let mut n_records = 0;
    for (c, p) in population
        .subjects
        .iter()
        .zip(&prepared.population.subjects)
    {
        assert_eq!(c.id, p.id);
        for j in 0..c.obs_times.len() {
            let got = c.obs_cov(j)["__level_PLACEBO"];
            assert_eq!(
                got,
                c.obs_cov(j)["PLA_IDX"],
                "subject {} record {j}: composable index vs PLA_IDX",
                c.id
            );
            assert_eq!(
                p.obs_cov(j)["__level_PLACEBO"],
                got,
                "subject {} record {j}: prepare_run vs composable",
                c.id
            );
            n_records += 1;
        }
    }
    assert_eq!(n_records, 6);
}

// ── #1614: simulating a design with a fit's θ ────────────────────────────────
//
// A fit's θ vector is laid out by the levels of the *fit* data. Re-binding a
// simulation design with `bind_theta_levels` re-discovers the levels from the
// design, so any design whose combinations differ from the fit's reads the
// fitted values at the wrong positions — measured on `a1cd1b5b`: silently
// remapped at the same level count, all-zero predictions at a different one.
// `bind_theta_levels_from_fit` binds the design against the fit's own
// bindings instead. Every fixture here is the analytic one-compartment IV
// model of `level_block_model()`; no gradient path is reached.

/// [`DATA`] with study 2's subject listed first — the same cells, the same
/// count, a different listing order.
const DESIGN_REORDERED: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY,PLA_IDX
2,0,.,1,100,1,0,1,2,4
2,1,.,0,.,1,0,0,2,4
2,4,.,0,.,1,0,0,2,5
2,12,.,0,.,1,0,0,2,6
1,0,.,1,100,1,0,1,1,1
1,1,.,0,.,1,0,0,1,1
1,4,.,0,.,1,0,0,1,2
1,12,.,0,.,1,0,0,1,3
";

/// Study 2 only: a subset whose every label the fit saw, at positions 4, 5, 6
/// of the fit (`PLA_IDX`), not 1, 2, 3 as the design alone would number them.
/// Level 6 (`STUDY=2,TIME=12`) is the fit's dependent sum-to-zero level.
const DESIGN_STUDY2: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY,PLA_IDX
2,0,.,1,100,1,0,1,2,4
2,1,.,0,.,1,0,0,2,4
2,4,.,0,.,1,0,0,2,5
2,12,.,0,.,1,0,0,2,6
";

/// Case (a): TIME 12 → 24 on both studies. Same level count, two labels the
/// fit never saw.
const DESIGN_TIME24: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY
1,0,.,1,100,1,0,1,1
1,1,.,0,.,1,0,0,1
1,4,.,0,.,1,0,0,1
1,24,.,0,.,1,0,0,1
2,0,.,1,100,1,0,1,2
2,1,.,0,.,1,0,0,2
2,4,.,0,.,1,0,0,2
2,24,.,0,.,1,0,0,2
";

/// Case (g): only study 2's last time moves — only the fit's *dependent* level
/// is unseen, and the θ names of a re-bound design would be identical.
const DESIGN_DEPENDENT_ONLY: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY
1,0,.,1,100,1,0,1,1
1,1,.,0,.,1,0,0,1
1,4,.,0,.,1,0,0,1
1,12,.,0,.,1,0,0,1
2,0,.,1,100,1,0,1,2
2,1,.,0,.,1,0,0,2
2,4,.,0,.,1,0,0,2
2,24,.,0,.,1,0,0,2
";

/// Case (b): [`DATA`]'s cells plus a third study.
const DESIGN_STUDY3: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY
1,0,.,1,100,1,0,1,1
1,1,.,0,.,1,0,0,1
1,4,.,0,.,1,0,0,1
1,12,.,0,.,1,0,0,1
2,0,.,1,100,1,0,1,2
2,1,.,0,.,1,0,0,2
2,4,.,0,.,1,0,0,2
2,12,.,0,.,1,0,0,2
3,0,.,1,100,1,0,1,3
3,1,.,0,.,1,0,0,3
3,4,.,0,.,1,0,0,3
3,12,.,0,.,1,0,0,3
";

/// Read `design` as a simulation design for the model at `model_path` and bind
/// it against `fitted` — the documented simulate-after-fit sequence.
fn bind_design_from_fit(
    model_path: &std::path::Path,
    design: &str,
    fitted: &ferx_core::parser::model_parser::LevelBindings,
) -> Result<(ferx_core::ParsedModel, ferx_core::Population), String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let design_path = dir.path().join("design.csv");
    write!(std::fs::File::create(&design_path).unwrap(), "{design}").unwrap();
    let model_text = std::fs::read_to_string(model_path).unwrap();
    let mut parsed = ferx_core::parse_full_model_file(model_path).expect("parse");
    let (mut population, _) = ferx_core::api::read_population_for_simulation(
        &parsed.model,
        &parsed.covariate_decls,
        design_path.to_str().unwrap(),
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("read design");
    ferx_core::bind_theta_levels_from_fit(&mut parsed, &model_text, &mut population, fitted)?;
    Ok((parsed, population))
}

/// Bind the fit data with `bind_theta_levels` and return the fit's model and
/// level bindings — what a caller keeps after the fit.
fn fit_bindings() -> (
    tempfile::TempDir,
    PathBuf,
    ferx_core::CompiledModel,
    ferx_core::parser::model_parser::LevelBindings,
) {
    let (dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let (parsed, _population) = read_composable(&model_path, &data_path, true);
    let levels = parsed.bindings.levels.clone();
    (dir, model_path, parsed.model, levels)
}

/// Every observation record's synthesized index against the `PLA_IDX` the
/// design carries — the dataset's own encoding of the fit's positions, a
/// reference outside the binder. Returns the number of records checked.
fn assert_index_is_pla_idx(population: &ferx_core::Population, case: &str) -> usize {
    let mut n = 0;
    for s in &population.subjects {
        for j in 0..s.obs_times.len() {
            assert_eq!(
                s.obs_cov(j)["__level_PLACEBO"],
                s.obs_cov(j)["PLA_IDX"],
                "{case}: subject {} record {j} (TIME {})",
                s.id,
                s.obs_times[j]
            );
            n += 1;
        }
    }
    n
}

/// T1. Same cells, same count, study 2 listed first: each record carries its
/// fit position, and the design model has the fit's θ layout.
///
/// Mutation — index by the design's own first-seen order (the from-fit lookup
/// replaced by the design's position, with `discover_levels`' sort dropped):
/// study 2's records become 1, 2, 3 and this dies on the first record.
#[test]
fn a_reordered_design_binds_at_the_fits_positions() {
    let (_dir, model_path, fit_model, fitted) = fit_bindings();
    let (parsed, population) =
        bind_design_from_fit(&model_path, DESIGN_REORDERED, &fitted).expect("bind");
    assert_eq!(
        population.subjects[0].id, "2",
        "fixture lists study 2 first"
    );
    assert_eq!(assert_index_is_pla_idx(&population, "reordered"), 6);
    assert_eq!(parsed.model.n_theta, fit_model.n_theta);
    assert_eq!(parsed.model.theta_names, fit_model.theta_names);
}

/// T2. A subset design (study 2 only, every label seen by the fit) is indexed
/// 4, 5, 6 — its fit positions — and keeps the fit's θ count of 7, not the 4
/// (TVCL, TVV, two free levels) a re-bind of the design alone would produce.
///
/// Mutations — index by the design's own sorted position (today's
/// `bind_theta_levels` behaviour → 1, 2, 3) dies on the index; re-parse with
/// bindings derived from the design instead of `fitted` dies on `n_theta`.
#[test]
fn a_subset_design_binds_at_the_fits_positions_and_keeps_its_layout() {
    let (_dir, model_path, fit_model, fitted) = fit_bindings();
    assert_eq!(
        fit_model.n_theta, 7,
        "2 studies x 3 times, sum-to-zero, + TVCL, TVV"
    );
    let (parsed, population) =
        bind_design_from_fit(&model_path, DESIGN_STUDY2, &fitted).expect("bind");
    assert_eq!(assert_index_is_pla_idx(&population, "study 2 only"), 3);
    assert_eq!(parsed.model.n_theta, fit_model.n_theta);
    assert_eq!(parsed.model.theta_names, fit_model.theta_names);
    // The fit's bindings, verbatim (`LevelBinding` has no `PartialEq`).
    let (got, want) = (&parsed.bindings.levels["PLACEBO"], &fitted["PLACEBO"]);
    assert_eq!(parsed.bindings.levels.len(), 1);
    assert_eq!(got.labels, want.labels);
    assert_eq!(got.groups, want.groups);
    assert_eq!(
        format!("{:?}", got.contrast),
        format!("{:?}", want.contrast)
    );
}

/// The refusal for each unseen-label case, held to the message contract: the
/// block, every unseen label, "the fit estimated no theta", the TIME-grid
/// consequence (this block is keyed on TIME), both options — and not the
/// level-count claim, which is false at the same count, nor a Rust function name,
/// since the R wrapper passes the text through verbatim (#1623).
fn assert_unseen_refusal(err: &str, labels: &[&str], case: &str) {
    assert!(
        err.contains("theta PLACEBO[STUDY, TIME]"),
        "{case}: block not named: {err}"
    );
    assert!(
        err.contains(&format!(
            "the design has {} level(s) the fit estimated no theta for",
            labels.len()
        )),
        "{case}: count / reason missing: {err}"
    );
    assert!(
        err.contains("A level's theta exists only for a combination the fit's data observed."),
        "{case}: why an unseen level has no theta, missing: {err}"
    );
    for l in labels {
        assert!(
            err.contains(&format!("`{l}`")),
            "{case}: label {l} not named: {err}"
        );
    }
    assert!(
        err.contains(
            "can only be simulated at the fit's observation times; a denser or different \
             time grid has no fitted theta."
        ),
        "{case}: TIME-grid consequence missing: {err}"
    );
    assert!(
        err.contains("Either simulate only the fit's levels,"),
        "{case}: first option missing: {err}"
    );
    assert!(
        err.contains(
            "or simulate the design without the fit's theta, from a theta vector for the \
             design's own levels (the model's initial estimates, for example)."
        ),
        "{case}: second option missing: {err}"
    );
    assert!(
        !err.contains("bind_theta_levels"),
        "{case}: a Rust function in a message the R wrapper passes through (#1623): {err}"
    );
    assert!(
        !err.contains("number of levels"),
        "{case}: a level-count claim in: {err}"
    );
}

/// T3. Same level count, unseen labels: refused, naming every one. The
/// dependent-only variant (case g) is the one a θ-name comparison cannot see.
///
/// Mutations — skip unseen labels, or map them to index 1, or to the dependent
/// level: each turns the `Err` into an `Ok` (or the writer's internal "data
/// changed between passes" text) and these die on the expected message.
#[test]
fn a_same_count_design_with_unseen_levels_is_refused_by_label() {
    let (_dir, model_path, _fit_model, fitted) = fit_bindings();
    let err = bind_design_from_fit(&model_path, DESIGN_TIME24, &fitted)
        .map(|_| ())
        .expect_err("TIME 24 was never fitted");
    assert_unseen_refusal(
        &err,
        &["STUDY=1,TIME=24", "STUDY=2,TIME=24"],
        "TIME 12 -> 24",
    );

    let err = bind_design_from_fit(&model_path, DESIGN_DEPENDENT_ONLY, &fitted)
        .map(|_| ())
        .expect_err("STUDY=2,TIME=24 was never fitted");
    assert_unseen_refusal(&err, &["STUDY=2,TIME=24"], "dependent level only");
    assert!(
        !err.contains("STUDY=1,TIME="),
        "only the unseen label may be named: {err}"
    );
}

/// T4. A study the fit never saw: refused, naming all three of its labels.
/// Today's re-bind instead returned `Ok` with every ipred exactly 0.
#[test]
fn a_design_with_an_unseen_study_is_refused_by_label() {
    let (_dir, model_path, _fit_model, fitted) = fit_bindings();
    let err = bind_design_from_fit(&model_path, DESIGN_STUDY3, &fitted)
        .map(|_| ())
        .expect_err("STUDY 3 was never fitted");
    assert_unseen_refusal(
        &err,
        &["STUDY=3,TIME=1", "STUDY=3,TIME=4", "STUDY=3,TIME=12"],
        "STUDY 3",
    );
}

/// T6. The simulated prediction against the closed form, outside every engine:
/// a 100 mg bolus into one compartment with a clearance that moves with the
/// level, `C(t_j) = 100/V · exp(−Σ_{k≤j} (TVCL + P_k)/V · (t_k − t_{k−1}))`,
/// with `P_k` read from a hand-written table keyed by `PLA_IDX`. The fitted θ are
/// distinct per level, and the design (study 2 only) includes the fit's
/// dependent level, whose value is `−Σ` of the five free ones.
///
/// Ω is zeroed in the parameters handed to `simulate_with_seed`, so η = 0
/// exactly and `ipred` is the typical prediction.
///
/// Mutations — a design-position index reads levels 1, 2, 3 (P = 0.1, 0.2, 0.3
/// instead of 0.4, 0.5, −1.5): every record is off by more than 1e-6. A wrong
/// dependent value (`+Σ` for `−Σ`) moves the TIME 12 record by orders of
/// magnitude.
#[test]
fn a_subset_design_simulates_the_closed_form_with_the_fits_theta() {
    let (_dir, model_path, fit_model, fitted) = fit_bindings();
    let (parsed, population) =
        bind_design_from_fit(&model_path, DESIGN_STUDY2, &fitted).expect("bind");

    // θ layout: TVCL, PLACEBO[STUDY=1,TIME=1 .. STUDY=2,TIME=4], TVV.
    let (tvcl, tvv) = (2.0, 10.0);
    let free = [0.1, 0.2, 0.3, 0.4, 0.5];
    assert_eq!(fit_model.theta_names[1], "PLACEBO[STUDY=1,TIME=1]");
    assert_eq!(fit_model.theta_names[5], "PLACEBO[STUDY=2,TIME=4]");
    let mut params = parsed.model.default_params.clone();
    params.theta = vec![tvcl, free[0], free[1], free[2], free[3], free[4], tvv];
    // The draw reads the cached Cholesky factor, so both go to zero.
    params.omega.matrix.fill(0.0);
    params.omega.chol.fill(0.0);

    // PLA_IDX → P, written out by hand. 6 is the dependent level: −(0.1+…+0.5).
    let p_of = |idx: f64| -> f64 {
        match idx as i64 {
            1 => 0.1,
            2 => 0.2,
            3 => 0.3,
            4 => 0.4,
            5 => 0.5,
            6 => -1.5,
            other => panic!("no PLA_IDX {other} in the fit"),
        }
    };

    let rows = ferx_core::simulate_with_seed(&parsed.model, &population, &params, 1, 17)
        .expect("simulate");
    assert_eq!(rows.len(), 3);
    let s = &population.subjects[0];
    let mut worst = 0.0f64;
    // CL moves with the level, record by record, and each record's value drives the
    // interval that ends at it (the bolus is at 0): the exponent accumulates
    // Σ (TVCL + P_k)/V · (t_k − t_{k−1}).
    let mut exponent = 0.0;
    let mut t_prev = 0.0;
    for (j, row) in rows.iter().enumerate() {
        let t = s.obs_times[j];
        assert_eq!(row.time, t);
        let p = p_of(s.obs_cov(j)["PLA_IDX"]);
        exponent += (tvcl + p) / tvv * (t - t_prev);
        t_prev = t;
        let want = 100.0 / tvv * (-exponent).exp();
        assert!(row.ipred.is_finite(), "TIME {t}: ipred {}", row.ipred);
        // Measured worst: 1.8e-16 (one ULP-scale rounding of the same exponentials).
        // The bound leaves four orders of headroom and is still ~1e10 below the
        // smallest wrong-position error (P 0.4 → 0.1 at TIME 1: rel 3e-2).
        let rel = (row.ipred - want).abs() / want;
        assert!(
            rel <= 1e-12,
            "TIME {t}: ipred {} vs closed form {want} (rel {rel:e})",
            row.ipred
        );
        worst = worst.max(rel);
    }
    eprintln!("closed-form worst rel error: {worst:e}");
}

/// T7 (#1623). The level-reporting surface is reachable as an **external** crate —
/// how ferx-r consumes it: `theta_level_values` at the crate root, the fields of
/// `api::ThetaLevelValue`, and `io::output::{compact_theta_blocks,
/// THETA_BLOCK_COMPACT_MIN}`. On the file path's own binding of the 2 × 3 design
/// (global sum-to-zero, so the last level is dependent), at the model's inits.
///
/// Mutation — make any of them `pub(crate)`: this file stops compiling.
#[test]
fn the_level_reporting_surface_is_reachable_from_outside_the_crate() {
    use ferx_core::api::ThetaLevelValue;
    use ferx_core::io::output::{compact_theta_blocks, THETA_BLOCK_COMPACT_MIN};

    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let (parsed, _population) = read_composable(&model_path, &data_path, true);
    let model = &parsed.model;
    let theta: Vec<f64> = (0..model.n_theta).map(|k| 0.1 * (k + 1) as f64).collect();

    let map = ferx_core::theta_level_values(model, &theta).expect("level values");
    let levels: &Vec<ThetaLevelValue> = &map["PLACEBO"];
    let labels: Vec<&str> = levels.iter().map(|l| l.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "STUDY=1,TIME=1",
            "STUDY=1,TIME=4",
            "STUDY=1,TIME=12",
            "STUDY=2,TIME=1",
            "STUDY=2,TIME=4",
            "STUDY=2,TIME=12",
        ]
    );
    for (i, l) in levels[..5].iter().enumerate() {
        assert_eq!((l.value, l.theta_index), (theta[i + 1], Some(i + 1)));
    }
    let free: f64 = theta[1..6].iter().sum();
    assert_eq!((levels[5].value, levels[5].theta_index), (-free, None));

    assert_eq!(THETA_BLOCK_COMPACT_MIN, 20);
    assert!(
        compact_theta_blocks(&model.theta_names).is_empty(),
        "five free coefficients stay inline"
    );
}

// ── #1644: the synthesized `__level_` column stays engine plumbing ───────────

#[test]
fn a_fit_reports_only_the_datas_real_covariate_columns() {
    // `FitResult::covariate_names` (R's `fit$covariate_names`, the runlog's
    // "Covariates:" line, `.fitrx`) lists the data's columns. The binder's
    // synthesized `__level_PLACEBO` is not one of them.
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let (result, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("fit");
    assert_eq!(result.covariate_names, ["STUDY", "PLA_IDX"]);

    // Every real column declared: the bound re-parse raises no
    // undeclared-covariate warning, and no warning names the column.
    let declared = level_block_model().replace(
        "[individual_parameters]",
        "[covariates]\n  STUDY categorical\n  PLA_IDX categorical\n\n[individual_parameters]",
    );
    let (_dir, model_path, data_path) = write_case(&declared, DATA);
    let (result, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("fit");
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("__level_") || w.contains("not declared in [covariates]")),
        "{:#?}",
        result.warnings
    );
}

#[test]
fn listing_the_index_column_in_covariate_names_changes_no_bit_of_the_fit() {
    // #1644 stopped the binder adding `__level_PLACEBO` to
    // `Population::covariate_names`; the predictors and the covariate check read
    // the subjects' maps instead. The twin restores the old list on the same
    // bound population: any engine read of the list would part the two fits.
    // (Measured before the change: Q1/Q8 MBMA fits at `maxiter = 0` and to
    // convergence, 204 and 108 iterations, OFV/θ/Ω/σ identical to the bit.)
    let (_dir, model_path, data_path) = write_case(&level_block_model(), DATA);
    let (parsed, population) = read_composable(&model_path, &data_path, true);
    let mut old_contract = population.clone();
    old_contract
        .covariate_names
        .push("__level_PLACEBO".to_string());

    let fit = |pop: &ferx_core::Population| {
        ferx_core::fit(
            &parsed.model,
            pop,
            &parsed.model.default_params,
            &parsed.fit_options,
        )
        .expect("fit")
    };
    let (now, before) = (fit(&population), fit(&old_contract));
    assert!(now.ofv.is_finite(), "OFV must be finite: {}", now.ofv);
    assert_eq!(now.ofv.to_bits(), before.ofv.to_bits(), "OFV");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&now.theta), bits(&before.theta), "theta");
    assert_eq!(now.covariate_names, ["STUDY", "PLA_IDX"]);
}

/// #1621: a level block **and** a `center = median` covariate centre, so a fit
/// carries both halves of its data-derived bindings.
const BINDINGS_MODEL: &str = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[STUDY](0.0, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL + PLACEBO
  V  = TVV * exp(ETA_V)

[covariates]
  WT continuous
  STUDY categorical

[covariate_model]
  V ~ WT power(center = median) => THETA_V_WT(0.9, 0.01, 5.0)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Six subjects, three studies, two doses each (0 h, 12 h); WT median 71.475.
fn bindings_fit_data() -> String {
    let wts = [61.3, 70.85, 72.1, 80.4, 55.55, 90.2];
    bindings_rows(
        &wts.iter()
            .enumerate()
            .map(|(i, &w)| (i / 2 + 1, w))
            .collect::<Vec<_>>(),
    )
}

/// Three subjects in studies 2 and 3 only, all heavier than the fit's: on its
/// own data the design would bind two levels and a WT median of 95.0.
fn bindings_design_data() -> String {
    bindings_rows(&[(2, 95.0), (2, 101.3), (3, 88.8)])
}

fn bindings_rows(subjects: &[(usize, f64)]) -> String {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,MDV,CMT,STUDY,WT\n");
    for (i, &(study, wt)) in subjects.iter().enumerate() {
        let id = i + 1;
        for t in [0, 12] {
            csv.push_str(&format!("{id},{t},0,100,1,1,1,{study},{wt}\n"));
        }
        for t in [1, 2, 4, 13, 14] {
            csv.push_str(&format!("{id},{t},1.0,0,0,0,1,{study},{wt}\n"));
        }
    }
    csv
}

/// Simulate the design under the fit's bindings `b`, by the sequence
/// `docs/api/fitting.qmd#simulate-with-fit-theta` documents: the fit's
/// statistics go into the design's bindings, a re-parse compiles them in, then
/// the levels bind from the fit — the from-fit path a reloaded `.fitrx` takes.
fn simulate_design_from(
    model: &str,
    b: &ferx_core::parser::model_parser::DataBindings,
    design_path: &std::path::Path,
    theta: &[f64],
) -> (usize, Vec<f64>) {
    let mut pop =
        ferx_core::io::datareader::read_nonmem_csv(design_path, None, None).expect("design");
    let mut parsed = ferx_core::parser::model_parser::parse_full_model(model).unwrap();
    parsed.bindings.covariate_stats = b.covariate_stats.clone();
    parsed.model = ferx_core::parser::model_parser::parse_full_model_with(model, &parsed.bindings)
        .expect("re-parse with the fit's statistics")
        .model;
    ferx_core::api::bind_theta_levels_from_fit(&mut parsed, model, &mut pop, &b.levels)
        .expect("bind the design from the fit");
    let mut params = parsed.model.default_params.clone();
    params.theta = theta.to_vec();
    let rows =
        ferx_core::api::simulate_with_seed(&parsed.model, &pop, &params, 2, 7).expect("simulate");
    (parsed.model.n_theta, rows.iter().map(|r| r.ipred).collect())
}

/// #1621 T6. A fit through the file entry point carries its data-derived
/// bindings; they survive `save_fit` → `load_fit`; and the design simulated from
/// the reloaded bindings is bit-identical to the design simulated from the live
/// ones. One analytic 1-cpt predictor, value path only; no gradient is involved
/// (`outer_maxiter = 0`, no covariance step).
///
/// Straddle control, same test: the design bound on *its own* data gets a
/// different θ layout (2 levels, so 4 θ against the fit's 5) and a different WT
/// median (95.0 against 71.475), and its WT median alone moves the worst ipred
/// by 0.2829 relative (measured) — so bit-identity above is a claim about
/// bindings that matter. The bound is 0.2, ~30 % headroom.
///
/// Mutations — `fit.rs` copies `DataBindings::default()` into the result, or
/// the parse stamp drops `covariate_stats`: the reloaded bindings are empty (or
/// lack the statistics), the bindings assertions die, and the from-fit design
/// refuses to bind or to simulate.
#[test]
fn a_fits_data_bindings_survive_fitrx_and_drive_the_design_bit_for_bit() {
    let (dir, model_path, data_path) = write_case(BINDINGS_MODEL, &bindings_fit_data());
    let design_path = dir.path().join("design.csv");
    std::fs::write(&design_path, bindings_design_data()).unwrap();

    let opts = ferx_core::FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    };
    let result = ferx_core::api::fit_from_files(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
        None,
        Some(opts),
    )
    .expect("fit");
    let live = &result.data_bindings;
    assert_eq!(
        live.levels["PLACEBO"].labels,
        ["STUDY=1", "STUDY=2", "STUDY=3"]
    );
    assert_eq!(live.covariate_stats["WT"].median, 71.475);
    assert_eq!(result.theta.len(), 5);

    let fitrx = dir.path().join("fit.fitrx");
    let pop = ferx_core::io::datareader::read_nonmem_csv(&data_path, None, None).expect("fit data");
    ferx_core::io::fitrx::save_fit(
        &result,
        &pop,
        BINDINGS_MODEL,
        &fitrx,
        ferx_core::io::fitrx::SaveFitOptions::default(),
    )
    .expect("save");
    let loaded = ferx_core::io::fitrx::load_fit(&fitrx).expect("load");
    assert_eq!(loaded.fit.data_bindings, *live);

    // A distinctive θ, so every level and the WT exponent are live.
    let mut theta: Vec<f64> = (0..5).map(|i| 0.3 + 0.17 * i as f64).collect();
    theta[0] = 2.3;
    theta[3] = 12.0; // TVV
    assert_eq!(result.theta_names[3], "TVV");

    let (n_live, a) = simulate_design_from(BINDINGS_MODEL, live, &design_path, &theta);
    let (n_loaded, b) = simulate_design_from(
        BINDINGS_MODEL,
        &loaded.fit.data_bindings,
        &design_path,
        &theta,
    );
    assert_eq!((n_live, n_loaded), (5, 5));
    assert_eq!(a.len(), 30);
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x.is_finite() && *x > 0.0, "row {i}: {x}");
        assert_eq!(x.to_bits(), y.to_bits(), "row {i}: {x} vs {y}");
    }

    // Control: what the design resolves on its own.
    let mut own_pop =
        ferx_core::io::datareader::read_nonmem_csv(&design_path, None, None).expect("design");
    let mut own = ferx_core::parser::model_parser::parse_full_model(BINDINGS_MODEL).unwrap();
    ferx_core::api::bind_theta_levels(&mut own, BINDINGS_MODEL, &mut own_pop).unwrap();
    ferx_core::api::bind_covariate_stats(&mut own, BINDINGS_MODEL, &own_pop).unwrap();
    let own_b = own.model.data_bindings();
    assert_eq!(own.model.n_theta, 4);
    assert_eq!(own_b.covariate_stats["WT"].median, 95.0);
    assert_ne!(own_b, live);
    // Same fitted levels, the design's own WT statistics: the ipred moves.
    let mut mixed = live.clone();
    mixed.covariate_stats = own_b.covariate_stats.clone();
    let (_, c) = simulate_design_from(BINDINGS_MODEL, &mixed, &design_path, &theta);
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(&c) {
        assert!(y.is_finite(), "{y}");
        worst = worst.max(((x - y) / x).abs());
    }
    assert!(
        worst > 0.2,
        "design-own WT statistics moved ipred by only {worst}"
    );
}

/// #1621 review round 1, finding 1: the documented from-fit sequence must also
/// hold for a model with a symbolic `[covariate_model]` centre and **no** level
/// block. `bind_theta_levels_from_fit` returns before re-parsing on such a model,
/// so the fit's statistics reach it only through the documented re-parse.
///
/// Both sides of that step in one test: without the re-parse the model stays
/// statistics-unbound and `simulate` refuses (measured on `e8e27f51`); with it,
/// the design simulates bit-identically to the fitted model's own compilation
/// (`bind_covariate_stats` on the fit data), and not like the design's own
/// statistics (WT median 95.0 against the fit's 71.475).
///
/// Mutations — drop the re-parse from `simulate_design_from` and the documented
/// arm panics on `simulate`; `fit.rs` copying `DataBindings::default()` leaves no
/// statistics to re-parse with and the same arm dies.
#[test]
fn the_documented_from_fit_sequence_binds_statistics_without_a_level_block() {
    let model = BINDINGS_MODEL
        .replace("  theta PLACEBO[STUDY](0.0, -10.0, 10.0)\n", "")
        .replace("CL = TVCL + PLACEBO", "CL = TVCL");
    let (dir, model_path, data_path) = write_case(&model, &bindings_fit_data());
    let design_path = dir.path().join("design.csv");
    std::fs::write(&design_path, bindings_design_data()).unwrap();
    let opts = ferx_core::FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    };
    let result = ferx_core::api::fit_from_files(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
        None,
        Some(opts),
    )
    .expect("fit");
    assert!(result.data_bindings.levels.is_empty());
    assert_eq!(result.data_bindings.covariate_stats["WT"].median, 71.475);
    assert_eq!(result.theta_names, ["TVCL", "TVV", "THETA_V_WT"]);
    let theta = [2.3, 12.0, 0.65];

    // The other side: the sequence without the re-parse leaves the model unbound.
    let mut pop =
        ferx_core::io::datareader::read_nonmem_csv(&design_path, None, None).expect("design");
    let mut bare = ferx_core::parser::model_parser::parse_full_model(&model).unwrap();
    bare.bindings.covariate_stats = result.data_bindings.covariate_stats.clone();
    ferx_core::api::bind_theta_levels_from_fit(&mut bare, &model, &mut pop, &Default::default())
        .expect("no level block: a no-op");
    assert!(bare.model.data_bindings().covariate_stats.is_empty());
    let mut params = bare.model.default_params.clone();
    params.theta = theta.to_vec();
    let err = ferx_core::api::simulate_with_seed(&bare.model, &pop, &params, 2, 7).unwrap_err();
    assert!(err.contains("still need data-derived statistics"), "{err}");

    // The documented sequence.
    let (n, a) = simulate_design_from(&model, &result.data_bindings, &design_path, &theta);
    assert_eq!(n, 3);

    // Reference: the fitted model as the fit compiled it, on the design.
    let mut fit_pop =
        ferx_core::io::datareader::read_nonmem_csv(&data_path, None, None).expect("fit data");
    let mut fitted = ferx_core::parser::model_parser::parse_full_model(&model).unwrap();
    ferx_core::api::bind_theta_levels(&mut fitted, &model, &mut fit_pop).unwrap();
    ferx_core::api::bind_covariate_stats(&mut fitted, &model, &fit_pop).unwrap();
    let mut params = fitted.model.default_params.clone();
    params.theta = theta.to_vec();
    let want: Vec<f64> = ferx_core::api::simulate_with_seed(&fitted.model, &pop, &params, 2, 7)
        .expect("reference")
        .iter()
        .map(|r| r.ipred)
        .collect();
    assert_eq!(a.len(), 30);
    assert_eq!(a.len(), want.len());
    for (i, (x, y)) in a.iter().zip(&want).enumerate() {
        assert!(x.is_finite() && *x > 0.0, "row {i}: {x}");
        assert_eq!(x.to_bits(), y.to_bits(), "row {i}: {x} vs {y}");
    }
    // Not the design's own statistics: those move the ipred.
    let mut own = ferx_core::parser::model_parser::parse_full_model(&model).unwrap();
    ferx_core::api::bind_covariate_stats(&mut own, &model, &pop).unwrap();
    let own_b = own.model.data_bindings().clone();
    assert_eq!(own_b.covariate_stats["WT"].median, 95.0);
    let (_, c) = simulate_design_from(&model, &own_b, &design_path, &theta);
    assert!(
        a.iter().zip(&c).any(|(x, y)| x.to_bits() != y.to_bits()),
        "the design's own statistics must give different rows"
    );
}
