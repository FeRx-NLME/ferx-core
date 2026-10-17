//! #1774 part b: every input refusal of `run_sir` / `run_covariance` carries a code,
//! one per cause. The post-hoc codes are assigned in `api/entry_refusals.rs`; the
//! re-reads of `fit.model_path` and `fit.data_path` are coded by `ferx check`'s own
//! read classifiers (`model_source_diagnostic`, `data_read_diagnostic`,
//! `selection_filter_diagnostic` in `validation.rs`), so a failure both can reach gets
//! the code `ferx check` gives it.
//!
//! One row per refusal **text**, through both entry points where the resolver raises
//! it. Each row asserts four things, and each has its own mutation:
//!
//! | Assertion | Regression it catches |
//! |---|---|
//! | `code()` | the site still converts its `String` (`code()` is `None`), or names another cause's constructor |
//! | `block()` | a classifier keyed on the wrong block, or a block dropped |
//! | `context()` | the refusal no longer attributed to the entry point (or newly attributed where it was not) |
//! | `to_string()`, the whole text | a sentence of the message deleted or reworded while it moved; every sentence of every refusal is pinned here once |
//!
//! The rows a `ferx check` run can reach (the model and data reads) also assert that
//! `ferx check` on the same files reports the same code, block and message, so the
//! two cannot drift: a second copy of a classifier dies there.
//!
//! The unrecorded-occasion-rule refusal (`E_FIT_INPUT_MISSING`) has its whole texts
//! pinned in `tests::iov_rules::unrecorded_rule_without_labels_is_refused`, which
//! asserts its code too.
//!
//! Fixture: the #1622 two-compartment oral fixture (`test_fixtures`), two fits (`Plain`,
//! `Median`). Every row refuses before an inner loop runs.

use std::path::{Path, PathBuf};

use super::test_fixtures::{case, design, Case, Kind};
use super::*;
use crate::estimation::run_covariance::run_covariance;
use crate::estimation::run_sir::run_sir;

const ENTRIES: [&str; 2] = ["run_sir", "run_covariance"];

/// One refusal: the `Err` an entry point returned, and what it must carry.
struct Row {
    name: String,
    got: Result<(), EngineError>,
    code: &'static str,
    block: Option<&'static str>,
    context: Option<&'static str>,
    text: String,
}

/// Every mismatch at once, so one run names every row that moved rather than the first.
fn check_rows(rows: Vec<Row>) {
    let mut bad = Vec::new();
    for r in rows {
        let Err(e) = r.got else {
            bad.push(format!("{}: accepted", r.name));
            continue;
        };
        if e.code() != Some(r.code) {
            bad.push(format!("{}: code {:?}, want {}", r.name, e.code(), r.code));
        }
        if e.block() != r.block {
            bad.push(format!(
                "{}: block {:?}, want {:?}",
                r.name,
                e.block(),
                r.block
            ));
        }
        if e.context() != r.context {
            bad.push(format!(
                "{}: context {:?}, want {:?}",
                r.name,
                e.context(),
                r.context
            ));
        }
        if e.to_string() != r.text {
            bad.push(format!("{}: text\n  got:  {e}\n  want: {}", r.name, r.text));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn run(
    entry: &str,
    c: &Case,
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
) -> Result<(), EngineError> {
    match entry {
        "run_sir" => run_sir(fit, model, population, &c.opts).map(|_| ()),
        _ => run_covariance(fit, model, population, &c.opts).map(|_| ()),
    }
}

/// A row of `entry`, the entry point's context, `text` after its `"{entry}: "`.
fn row(
    name: &str,
    entry: &'static str,
    got: Result<(), EngineError>,
    code: &'static str,
    block: Option<&'static str>,
    text: &str,
) -> Row {
    Row {
        name: format!("{entry}/{name}"),
        got,
        code,
        block,
        context: Some(entry),
        text: format!("{entry}: {text}"),
    }
}

/// A copy of the case's model and data files in a fresh directory, and the fit
/// pointed at them, so a row can edit or delete a file without touching the case.
struct Files {
    _dir: tempfile::TempDir,
    model: PathBuf,
    data: PathBuf,
    fit: FitResult,
}

fn files(c: &Case) -> Files {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.ferx");
    let data = dir.path().join("data.csv");
    std::fs::copy(&c.model_path, &model).unwrap();
    std::fs::copy(c.fit.data_path.as_deref().unwrap(), &data).unwrap();
    let mut fit = c.fit.clone();
    fit.model_path = Some(model.to_str().unwrap().to_string());
    fit.data_path = Some(data.to_str().unwrap().to_string());
    Files {
        _dir: dir,
        model,
        data,
        fit,
    }
}

fn show(p: &Path) -> String {
    p.display().to_string()
}

/// The data file with `edit` applied to each row's cells (the header is row 0).
fn edit_data(path: &Path, edit: impl Fn(usize, &mut Vec<String>)) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut out = String::new();
    for (i, line) in text.lines().enumerate() {
        let mut cols: Vec<String> = line.split(',').map(str::to_string).collect();
        edit(i, &mut cols);
        out.push_str(&cols.join(","));
        out.push('\n');
    }
    std::fs::write(path, out).unwrap();
}

/// `ferx check` on the same files reports the refusal's code, block and message: the
/// first error of `validate_model_file(model, data)`.
fn assert_check_agrees(what: &str, model: &Path, data: Option<&Path>, e: &EngineError) {
    let report =
        crate::api::validate_model_file(model.to_str().unwrap(), data.map(|d| d.to_str().unwrap()));
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.is_error())
        .unwrap_or_else(|| panic!("{what}: ferx check reports no error"));
    assert_eq!(Some(d.code.as_str()), e.code(), "{what}: code");
    assert_eq!(d.block.as_deref(), e.block(), "{what}: block");
    assert_eq!(d.message, e.message(), "{what}: message");
}

/// The reader's own refusal for `path` under `settings`: `read_population_for`, the
/// call beneath `read_population_with`, which no classifier touches. An expected text
/// taken from `read_population_with` would pass through `data_read_diagnostic`, the
/// code under test, so a classifier that rewrote the message would agree with itself
/// (review r1 #1).
fn raw_read(model: &CompiledModel, settings: &crate::api::ReaderSettings, path: &Path) -> String {
    assert!(
        settings.ignore_exprs.is_empty()
            && settings.accept_exprs.is_empty()
            && settings.ignore_subjects.is_empty(),
        "the fixture's recorded settings carry no row filter"
    );
    let fallback: Option<Vec<&str>> = settings
        .fallback_columns
        .as_ref()
        .map(|cols| cols.iter().map(String::as_str).collect());
    crate::api::read_population_for(
        model,
        &settings.covariate_decls,
        path.to_str().unwrap(),
        fallback.as_deref(),
        settings.iov_column.as_deref(),
        None,
        &settings.column_map,
    )
    .map(|_| ())
    .unwrap_err()
}

const BECAUSE: &str = " The population is re-read with this file's `[data]` renames and \
                       `[data_selection]`, which decide the rows the fit saw. Pass \
                       `population = Some(&pop)` as well, and the model file is not read.";

/// `E_FIT_INPUT_MISSING`, `E_FIT_INPUT_CHANGED`, `E_THETA_LENGTH` and
/// `E_FIT_INPUT_MODEL` on the η̂ width: the post-hoc codes with no model statistics.
///
/// Mutations — any site back on `EngineError::uncoded` (its rows' `code()` is `None`);
/// `fit_input_missing` / `fit_input_changed` / `fit_input_model` / `theta_length`
/// swapped at a site (its rows' code); `in_context` dropped from either η̂ check or
/// added to the covariance check (`context()`); a sentence deleted from any message
/// (`to_string()`).
#[test]
fn the_post_hoc_input_refusals_carry_their_codes() {
    let c = case(Kind::Plain);
    let model = &c.prep.parsed.model;
    let iov = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(40.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n  sigma PROP ~ 0.02\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(PROP)\n",
    )
    .expect("parse");
    assert!(iov.n_kappa > 0);

    let mut rows = Vec::new();
    for entry in ENTRIES {
        // E_FIT_INPUT_MISSING
        let mut f = c.fit.clone();
        f.model_path = None;
        rows.push(row(
            "no model_path",
            entry,
            run(entry, &c, &f, None, None),
            "E_FIT_INPUT_MISSING",
            None,
            "no model supplied and fit.model_path is None. Either pass `model = \
             Some(&model)` or re-fit via fit_from_files so the path is recorded.",
        ));
        let mut f = c.fit.clone();
        f.data_path = None;
        rows.push(row(
            "no data_path",
            entry,
            run(entry, &c, &f, Some(model), None),
            "E_FIT_INPUT_MISSING",
            None,
            "no population supplied and fit.data_path is None. Either pass `population = \
             Some(&pop)` or re-fit via fit_from_files so the path is recorded.",
        ));
        let mut f = c.fit.clone();
        f.reader_settings = None;
        rows.push(row(
            "IOV model without its population",
            entry,
            run(entry, &c, &f, Some(&iov), None),
            "E_FIT_INPUT_MISSING",
            None,
            "caller-supplied `model` for an IOV (n_kappa > 0) model requires `population` to \
             also be supplied — the model carries no `iov_column`, and per-occasion kappas \
             are parsed with it.",
        ));

        // E_FIT_INPUT_CHANGED, from each file.
        let fs = files(&c);
        let stored = fs.fit.data_hash.clone().unwrap();
        std::fs::write(
            &fs.data,
            format!("{}# edited\n", std::fs::read_to_string(&fs.data).unwrap()),
        )
        .unwrap();
        let current = crate::io::hash::sha256_file(&fs.data).unwrap();
        rows.push(row(
            "data file changed",
            entry,
            run(entry, &c, &fs.fit, None, None),
            "E_FIT_INPUT_CHANGED",
            None,
            &format!(
                "data hash mismatch for {}. Stored: {stored}, current: {current}. The dataset \
                 has changed since the fit was produced — refusing to run against stale data.",
                show(&fs.data)
            ),
        ));
        let fs = files(&c);
        let stored = fs.fit.model_hash.clone().unwrap();
        std::fs::write(
            &fs.model,
            format!(
                "{}\n# edited\n",
                std::fs::read_to_string(&fs.model).unwrap()
            ),
        )
        .unwrap();
        let current = crate::io::hash::sha256_file(&fs.model).unwrap();
        let changed = format!(
            "model hash mismatch for {}. Stored: {stored}, current: {current}. The .ferx file \
             has changed since the fit was produced — refusing to run against stale source.",
            show(&fs.model)
        );
        rows.push(row(
            "model file changed, rebuilt",
            entry,
            run(entry, &c, &fs.fit, None, None),
            "E_FIT_INPUT_CHANGED",
            None,
            &changed,
        ));
        let mut legacy = fs.fit.clone();
        legacy.reader_settings = None;
        rows.push(row(
            "model file changed, read for its reader settings",
            entry,
            run(entry, &c, &legacy, Some(model), None),
            "E_FIT_INPUT_CHANGED",
            None,
            &format!("{changed}{BECAUSE}"),
        ));

        // E_THETA_LENGTH: the fit's θ against the model rebuilt from its file.
        let mut f = c.fit.clone();
        f.theta.push(1.0);
        rows.push(row(
            "θ count",
            entry,
            run(entry, &c, &f, None, None),
            "E_THETA_LENGTH",
            None,
            "the model has n_theta = 6 but the fit has 7 θ. Verify you supplied the same model \
             the fit used, bound the way the fit was (`bind_from_fit` with the fit's \
             `data_bindings`).",
        ));

        // E_FIT_INPUT_MODEL: η̂ one wider than the model's η.
        let mut f = c.fit.clone();
        for s in &mut f.subjects {
            s.eta = s.eta.clone().insert_row(s.eta.len(), 0.0);
        }
        rows.push(row(
            "η̂ width",
            entry,
            run(entry, &c, &f, None, None),
            "E_FIT_INPUT_MODEL",
            None,
            "fit.subjects[0] has eta dim 4 but model has n_eta = 3. Subject EBEs are \
             inconsistent with the supplied model.",
        ));
    }
    // SIR's proposal: the one refusal not attributed to the entry point, as before.
    rows.push(Row {
        name: "run_sir/no covariance matrix".to_string(),
        got: run("run_sir", &c, &c.fit, None, None),
        code: "E_FIT_INPUT_MISSING",
        block: None,
        context: None,
        text: "run_sir requires fit.covariance_matrix; re-run the original fit with the \
               covariance step enabled (FitOptions.run_covariance_step = true, or `covariance \
               = true` in the model file's [fit_options])."
            .to_string(),
    });
    check_rows(rows);
}

/// The model and data re-reads: coded as `ferx check` codes the same failure, through
/// its own classifiers, and the rows `ferx check` can reach agree with it on the same
/// files. #1774's control: a re-read with a covariate column missing is
/// `E_MISSING_COVARIATE`, from `run_sir` as from `ferx check`.
///
/// Mutations — any read site back on a bare `String` (`code()` `None`); the
/// classifier's `Read` / `Parse` arms swapped or collapsed to one code (the unreadable,
/// not-UTF-8 and parse rows); `E_DATA` for every data failure (the covariate rows); the
/// selection classifier's block dropped; a second copy of a classifier that drifts
/// from `ferx check`'s (`assert_check_agrees`).
#[test]
fn the_re_reads_carry_ferx_checks_codes() {
    let c = case(Kind::Plain);
    let model = &c.prep.parsed.model;
    let mut rows = Vec::new();
    for entry in ENTRIES {
        // E_MODEL_READ: the file is gone, for each of its two uses.
        let fs = files(&c);
        std::fs::remove_file(&fs.model).unwrap();
        let os = std::fs::read(&fs.model).unwrap_err();
        let gone = format!("cannot read the model file {}: {os}.", show(&fs.model));
        let r = run(entry, &c, &fs.fit, None, None);
        assert_check_agrees("model gone", &fs.model, None, r.as_ref().unwrap_err());
        rows.push(row(
            "model file gone, rebuilt",
            entry,
            r,
            "E_MODEL_READ",
            None,
            &gone,
        ));
        let mut legacy = fs.fit.clone();
        legacy.reader_settings = None;
        rows.push(row(
            "model file gone, read for its reader settings",
            entry,
            run(entry, &c, &legacy, Some(model), None),
            "E_MODEL_READ",
            None,
            &format!("{gone}{BECAUSE}"),
        ));

        // E_MODEL_READ on bytes that are not UTF-8; the parse's code on a file that
        // does not parse. Neither has a hash to fail first.
        let fs = files(&c);
        let mut f = fs.fit.clone();
        f.model_hash = None;
        std::fs::write(&fs.model, [0xffu8, 0xfe]).unwrap();
        let utf8 = String::from_utf8(vec![0xffu8, 0xfe]).unwrap_err();
        let r = run(entry, &c, &f, None, None);
        assert_check_agrees("not UTF-8", &fs.model, None, r.as_ref().unwrap_err());
        rows.push(row(
            "model file not UTF-8",
            entry,
            r,
            "E_MODEL_READ",
            None,
            &format!("the model file {} is not UTF-8: {utf8}", show(&fs.model)),
        ));
        let text = std::fs::read_to_string(&c.model_path).unwrap();
        let cut = text.find("[error_model]").unwrap();
        let end = text[cut..].find("\n\n").map_or(text.len(), |i| cut + i);
        std::fs::write(&fs.model, format!("{}{}", &text[..cut], &text[end..])).unwrap();
        let r = run(entry, &c, &f, None, None);
        assert_check_agrees("parse", &fs.model, None, r.as_ref().unwrap_err());
        rows.push(row(
            "model file does not parse",
            entry,
            r,
            "E_MISSING_BLOCK",
            Some("error_model"),
            "Missing [error_model] block",
        ));

        // E_DATA: the data file is gone, at its hash and at each of the three readers.
        let fs = files(&c);
        std::fs::remove_file(&fs.data).unwrap();
        let os = std::fs::File::open(&fs.data).unwrap_err();
        rows.push(row(
            "data file gone, hashed",
            entry,
            run(entry, &c, &fs.fit, None, None),
            "E_DATA",
            None,
            &format!("opening {}: {os}", show(&fs.data)),
        ));
        let mut f = fs.fit.clone();
        f.data_hash = None;
        let settings = f.reader_settings.clone().unwrap();
        let read = raw_read(model, &settings, &fs.data);
        let r = run(entry, &c, &f, None, None);
        assert_eq!(
            r.as_ref().unwrap_err().code(),
            Some("E_DATA"),
            "ferx check's code for a data file it cannot read"
        );
        rows.push(row(
            "data file gone, recorded settings",
            entry,
            r,
            "E_DATA",
            None,
            &read,
        ));
        let mut file_settings = f.clone();
        file_settings.reader_settings = None;
        rows.push(row(
            "data file gone, the model file's settings",
            entry,
            run(entry, &c, &file_settings, None, None),
            "E_DATA",
            None,
            &read,
        ));
        let mut routed = file_settings.clone();
        routed.model_path = None;
        let read = crate::api::read_population_routed_by(model, &fs.data, None, &[])
            .map(|_| ())
            .unwrap_err();
        rows.push(row(
            "data file gone, routed",
            entry,
            run(entry, &c, &routed, Some(model), None),
            "E_DATA",
            None,
            &read,
        ));

        // E_MISSING_COVARIATE / E_COVARIATE_NOT_NUMERIC: the strict reader's refusals,
        // through the recorded settings and through the model file's.
        for (what, code, edit) in [
            (
                "WT column missing",
                "E_MISSING_COVARIATE",
                (|_: usize, cols: &mut Vec<String>| {
                    cols.remove(8);
                }) as fn(usize, &mut Vec<String>),
            ),
            ("WT not numeric", "E_COVARIATE_NOT_NUMERIC", |i, cols| {
                if i == 1 {
                    cols[8] = "heavy".to_string();
                }
            }),
        ] {
            let fs = files(&c);
            edit_data(&fs.data, edit);
            assert!(
                std::fs::read_to_string(&fs.data)
                    .unwrap()
                    .starts_with("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,"),
                "{what}"
            );
            let mut f = fs.fit.clone();
            f.data_hash = None;
            let read = raw_read(model, f.reader_settings.as_ref().unwrap(), &fs.data);
            let r = run(entry, &c, &f, None, None);
            assert_check_agrees(what, &fs.model, Some(&fs.data), r.as_ref().unwrap_err());
            rows.push(row(
                &format!("{what}, recorded settings"),
                entry,
                r,
                code,
                Some("covariates"),
                &read,
            ));
            f.reader_settings = None;
            rows.push(row(
                &format!("{what}, the model file's settings"),
                entry,
                run(entry, &c, &f, None, None),
                code,
                Some("covariates"),
                &read,
            ));
        }

        // E_PARSE on `data_selection`: a recorded clause edited into one that does not
        // compile.
        let mut f = c.fit.clone();
        let mut settings = f.reader_settings.clone().unwrap();
        settings.ignore_exprs = vec!["TIME >".to_string()];
        // The filter's own refusal, not `read_population_with`'s (which classifies it).
        let read =
            crate::io::datareader::SelectionFilter::from_opts(&settings.ignore_exprs, &[], &[])
                .map(|_| ())
                .unwrap_err();
        f.reader_settings = Some(settings);
        rows.push(row(
            "recorded selection clause",
            entry,
            run(entry, &c, &f, None, None),
            "E_PARSE",
            Some("data_selection"),
            &read,
        ));
    }
    check_rows(rows);
}

/// `E_FIT_INPUT_MODEL` on a lent model's bindings and statistics, and the
/// statistics' own `E_COVARIATE_STATS_BINDING` where the population cannot summarise
/// a covariate (the code `bind_covariate_stats` gives the same data).
///
/// Mutations — either `fit_input_model` site back on `String` / `EngineError::uncoded`
/// (`code()` `None`); `check_stats_on`'s summary failure left uncoded (the no-value row);
/// a sentence of the #1729 refusal deleted (`to_string()`).
#[test]
fn a_model_that_is_not_the_fits_carries_its_code() {
    let c = case(Kind::Median);
    let lent = design(&c);
    let mut legacy = c.fit.clone();
    legacy.data_bindings = Default::default();
    let pop = &c.prep.population;
    let stat = |m: &CompiledModel| m.data_bindings().covariate_stats["WT"].median.to_string();
    let (mv, dv) = (stat(&lent.parsed.model), stat(&c.prep.parsed.model));
    assert_ne!(mv, dv, "the design's WT median is not the fit's");
    let mut no_wt = pop.clone();
    for s in &mut no_wt.subjects {
        for m in std::iter::once(&mut s.covariates)
            .chain(&mut s.obs_covariates)
            .chain(&mut s.dose_covariates)
            .chain(&mut s.pk_only_covariates)
            .chain(&mut s.reset_covariates)
        {
            if let Some(v) = m.get_mut("WT") {
                *v = f64::NAN;
            }
        }
    }

    let mut rows = Vec::new();
    for entry in ENTRIES {
        rows.push(row(
            "bindings differ",
            entry,
            run(entry, &c, &c.fit, Some(&lent.parsed.model), Some(pop)),
            "E_FIT_INPUT_MODEL",
            None,
            "the supplied model is not bound with this fit's bindings: its data-derived \
             bindings (level layout, covariate statistics) differ from the fit's \
             `data_bindings`. Pass `model = None` to rebuild it from the fit, or bind it with \
             `ferx_core::api::bind_from_fit` and the fit's `data_bindings`.",
        ));
        rows.push(row(
            "statistics differ, no bindings recorded",
            entry,
            run(entry, &c, &legacy, Some(&lent.parsed.model), Some(pop)),
            "E_FIT_INPUT_MODEL",
            None,
            &format!(
                "this fit records no data-derived bindings (an older `.fitrx`), so the \
                 supplied model's covariate statistics were checked against the supplied \
                 population, and they differ: the median of `WT` is {mv} in the model but \
                 {dv} in the supplied population. The model was bound on other data, and \
                 scoring the fit's θ with it would centre the relations on that data, not on \
                 the data the θ was estimated from. Re-parse the model and bind it on the \
                 fit's data: `prepare_run` on the fit's model and data files, or \
                 `bind_covariate_stats` on a freshly parsed model with the fit's population."
            ),
        ));
        rows.push(row(
            "a covariate with no value, no bindings recorded",
            entry,
            run(entry, &c, &legacy, Some(&c.prep.parsed.model), Some(&no_wt)),
            "E_COVARIATE_STATS_BINDING",
            Some("covariate_model"),
            "[covariate_model] needs summary statistics for covariate `WT`, but the dataset \
             carries no non-missing value for it",
        ));
    }
    check_rows(rows);
}

/// #2044: ODE solver settings no solver can run with, on a hand-built `FitOptions`
/// (a model file's `[fit_options]` refuses them at parse time, as `E_PARSE`), are
/// `E_ODE_SOLVER_OPTIONS` on every entry point that opens the fit scope: `run_sir`,
/// `run_covariance` and `run_sir_core_coded`. No block, no context. Each of the three
/// texts, through each entry, with its whole text pinned.
///
/// Mutations — `with_fit_scope`'s validate back on `EngineError::uncoded` (every
/// row's `code()`); a constructor naming another code (code); a sentence of a text
/// deleted in `OdeSolverOverride::validate` (`to_string()`). `with_fit_scope` is the
/// one site, so a mutation there reddens all three entries at once.
#[test]
fn invalid_ode_solver_options_carry_their_code_on_every_entry() {
    let c = super::test_fixtures::sir_case(Kind::Plain);
    let model = &c.prep.parsed.model;
    let pop = &c.prep.population;
    let params = &c.prep.init_params;
    let n_packed = crate::estimation::parameterization::packed_len(params);
    let cov = nalgebra::DMatrix::identity(n_packed, n_packed);
    let etas = vec![nalgebra::DVector::zeros(model.n_eta); pop.subjects.len()];
    let bad = |set: &dyn Fn(&mut FitOptions)| {
        let mut o = c.opts.clone();
        set(&mut o);
        o
    };
    let cases: [(&str, FitOptions, &str); 3] = [
        (
            "ode_reltol NaN",
            bad(&|o| o.ode_reltol = f64::NAN),
            "ode_reltol must be finite and positive, got NaN",
        ),
        (
            "ode_abstol zero",
            bad(&|o| o.ode_abstol = 0.0),
            "ode_abstol must be finite and positive, got 0",
        ),
        (
            "ode_max_steps zero",
            bad(&|o| o.ode_max_steps = 0),
            "ode_max_steps must be positive, got 0",
        ),
    ];
    let mut rows = Vec::new();
    for (name, o, text) in &cases {
        let entries: [(&str, Result<(), EngineError>); 3] = [
            ("run_sir", run_sir(&c.fit, None, None, o).map(|_| ())),
            (
                "run_covariance",
                run_covariance(&c.fit, None, None, o).map(|_| ()),
            ),
            (
                "run_sir_core_coded",
                crate::estimation::sir::run_sir_core_coded(model, pop, params, &etas, &cov, 0.0, o)
                    .map(|_| ()),
            ),
        ];
        for (entry, got) in entries {
            rows.push(Row {
                name: format!("{entry}/{name}"),
                got,
                code: "E_ODE_SOLVER_OPTIONS",
                block: None,
                context: None,
                text: text.to_string(),
            });
        }
    }
    check_rows(rows);
}

/// SIR's own input refusals through `run_sir`, from the shared core
/// (`run_sir_core_coded`): `E_SIR_OPTIONS` for settings the resampler cannot run with,
/// `E_FIT_INPUT_MODEL` for a covariance matrix that is not the size of the model's
/// packed parameters, `E_SIR_NOTHING_TO_SAMPLE` on `[parameters]` for a fit whose
/// θ, Ω and σ are all FIX. Not attributed to the entry point, as before. The fourth option
/// refusal, `sir_scale = natural` under a prior, is pinned on the core in
/// `sir_low_ess_tests::natural_scale_is_refused_with_a_prior_only`.
///
/// Mutations — any of the four sites back on a bare `String` (`code()` `None`);
/// `sir_options` / `fit_input_model` / `sir_nothing_to_sample` swapped (code), the
/// last one's block dropped (block); `run_sir` calling the `String`
/// wrapper `run_sir_core` (every row's code); a sentence deleted (`to_string()`).
#[test]
fn sirs_own_input_refusals_carry_their_codes() {
    let c = super::test_fixtures::sir_case(Kind::Plain);
    let cov = c
        .fit
        .covariance_matrix
        .as_ref()
        .expect("the SIR case carries one");
    let n = cov.nrows();
    let mut wrong = c.fit.clone();
    wrong.covariance_matrix = Some(nalgebra::DMatrix::identity(2, 2));
    let mut resamples = c.opts.clone();
    resamples.sir_samples = 10;
    resamples.sir_resamples = 20;
    let mut df = c.opts.clone();
    df.sir_df = 0.0;
    let chi2 = rand_distr::ChiSquared::<f64>::new(0.0).unwrap_err();
    let mut all_fix = c.fit.clone();
    all_fix.theta_fixed.iter_mut().for_each(|b| *b = true);
    all_fix.omega_fixed.iter_mut().for_each(|b| *b = true);
    all_fix.sigma_fixed.iter_mut().for_each(|b| *b = true);
    // A FIX that does not reach every packed coordinate is a sampled run, not this
    // refusal: the row must not pass on a fixture with no Ω or σ to fix.
    assert!(!all_fix.omega_fixed.is_empty() && !all_fix.sigma_fixed.is_empty());
    let sir = |fit: &FitResult, o: &FitOptions| run_sir(fit, None, None, o).map(|_| ());
    let plain = |name: &str, got, code, text: String| Row {
        name: format!("run_sir/{name}"),
        got,
        code,
        block: None,
        context: None,
        text,
    };
    check_rows(vec![
        plain(
            "sir_resamples above sir_samples",
            sir(&c.fit, &resamples),
            "E_SIR_OPTIONS",
            "sir_resamples must be <= sir_samples".to_string(),
        ),
        plain(
            "sir_df",
            sir(&c.fit, &df),
            "E_SIR_OPTIONS",
            format!("sir_df invalid: {chi2}"),
        ),
        plain(
            "proposal dimension",
            sir(&wrong, &c.opts),
            "E_FIT_INPUT_MODEL",
            format!("Covariance matrix dimensions (2,2) don't match packed parameters ({n})"),
        ),
        Row {
            block: Some("parameters"),
            ..plain(
                "every packed parameter FIX",
                sir(&all_fix, &c.opts),
                "E_SIR_NOTHING_TO_SAMPLE",
                "run_sir_core: every packed parameter is FIX — nothing to sample.".to_string(),
            )
        },
    ]);
}
