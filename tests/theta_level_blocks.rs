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
/// a tool) does it — `parse_full_model_file` → `read_population_for` — then
/// `bind_theta_levels` unless `bind` is false (#1384).
fn read_composable(
    model_path: &std::path::Path,
    data_path: &std::path::Path,
    bind: bool,
) -> (ferx_core::ParsedModel, ferx_core::Population) {
    let mut parsed = ferx_core::parse_full_model_file(model_path).expect("parse");
    let (mut population, _) = ferx_core::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data_path.to_str().unwrap(),
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("read");
    if bind {
        let model_text = std::fs::read_to_string(model_path).unwrap();
        ferx_core::bind_theta_levels(&mut parsed, &model_text, &mut population).expect("bind");
        // A no-op on this model; called so the helper is the documented example
        // (`docs/api/fitting.qmd`), imports included.
        ferx_core::api::bind_covariate_stats(&mut parsed, &model_text, &population)
            .expect("bind covariate stats");
    }
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
