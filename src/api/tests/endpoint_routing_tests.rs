//! `E_ENDPOINT_UNROUTED` / `E_ENDPOINT_NO_RECORDS` (#1199): a model with a
//! non-Gaussian endpoint fed a population that was not built for it.
//!
//! The defect these pin: `read_nonmem_csv` knows no model, so a joint PK-TTE
//! dataset read through it carries the event rows as *Gaussian* observations on
//! the event CMT and no `obs_records`. `fit()` then ran the Gaussian half only and
//! returned `Ok` with a plausible, finite, wrong objective (4606.6 against the
//! NONMEM-anchored 1912.0 on `pktte_tdep`), `predict()` returned a concentration
//! for the event row, and `simulate()` drew no events at all. Nothing said so.
//!
//! Every fixture here is a real file pair from the repo (`examples/` + `data/`),
//! read both ways, so the guard is exercised on the signature the readers
//! actually produce rather than on a hand-built `Population`.
#![cfg(feature = "survival")]

use crate::api::{
    check_model_data, check_simulation_data, fit, predict, read_population_for, validate_model_file,
};
use crate::diagnostics::{first_error, Diagnostic};
use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::{CompiledModel, FitOptions, Population};
use std::io::Write;
use std::path::Path;

const JOINT_MODEL: &str = "examples/pktte_joint.ferx";
const JOINT_DATA: &str = "data/pktte_joint.csv";
const BINARY_MODEL: &str = "examples/binary_logistic.ferx";
const BINARY_DATA: &str = "data/binary_logistic.csv";

fn model(path: &str) -> CompiledModel {
    let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    parse_full_model(&src).expect("example model parses").model
}

/// The model-aware read every fitting entry point is documented to use.
fn routed(m: &CompiledModel, data: &str) -> Population {
    read_population_for(m, &None, data, None, None, None, &[])
        .expect("routed read")
        .0
}

/// The model-blind public reader — the wrong loader for these models.
fn unrouted(data: &str) -> Population {
    read_nonmem_csv(Path::new(data), None, None).expect("unrouted read")
}

fn codes(diags: &[Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.code.as_str()).collect()
}

fn message_of(diags: &[Diagnostic], code: &str) -> String {
    diags
        .iter()
        .find(|d| d.code == code)
        .map(|d| d.message.clone())
        .unwrap_or_else(|| panic!("expected {code}, got {:?}", codes(diags)))
}

/// `data` with its `CMT` column removed, so every row reads as CMT 1 — the
/// routed-but-empty shape (measured: identical OFV to the unrouted read, and the
/// reader emits no warning for it).
fn without_cmt_column(data: &str) -> tempfile::NamedTempFile {
    let src = std::fs::read_to_string(data).unwrap();
    let mut lines = src.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let ci = header
        .iter()
        .position(|c| c.eq_ignore_ascii_case("CMT"))
        .expect("fixture has a CMT column");
    let strip = |l: &str| -> String {
        l.split(',')
            .enumerate()
            .filter(|(i, _)| *i != ci)
            .map(|(_, c)| c)
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", strip(&header.join(","))).unwrap();
    for l in lines {
        writeln!(f, "{}", strip(l)).unwrap();
    }
    f.flush().unwrap();
    f
}

fn total_events(pop: &Population) -> usize {
    pop.subjects.iter().map(|s| s.obs_records.len()).sum()
}

fn gaussian_rows_on(pop: &Population, cmt: usize) -> usize {
    pop.subjects
        .iter()
        .map(|s| s.obs_cmts.iter().filter(|&&c| c == cmt).count())
        .sum()
}

/// The two reads really do differ in the way the guard relies on — asserted here
/// so a later reader change that blurs the signature fails this file first.
#[test]
fn the_two_reads_of_the_joint_fixture_differ_by_the_guard_signature() {
    let m = model(JOINT_MODEL);
    let r = routed(&m, JOINT_DATA);
    let u = unrouted(JOINT_DATA);
    assert!(total_events(&r) > 0, "routed read must carry event records");
    assert_eq!(
        gaussian_rows_on(&r, 3),
        0,
        "routed read keeps CMT 3 out of the Gaussian grid"
    );
    assert_eq!(total_events(&u), 0, "unrouted read has no event records");
    assert_eq!(
        gaussian_rows_on(&u, 3),
        total_events(&r),
        "unrouted read carries every event row as a Gaussian CMT-3 observation"
    );
}

/// The defect itself: `fit()` on the unrouted population is a hard error, named
/// after the CMT and the loader to use — and it is the *first* error, so a per-CMT
/// check downstream cannot rename the cause.
#[test]
fn fit_rejects_an_unrouted_joint_population() {
    let m = model(JOINT_MODEL);
    let u = unrouted(JOINT_DATA);

    let diags = check_model_data(&m, &u);
    let first = first_error(&diags).expect_err("unrouted population must be an error");
    assert!(
        first.contains("E_ENDPOINT_UNROUTED"),
        "the first error must be the routing guard, got: {first}"
    );
    let msg = message_of(&diags, "E_ENDPOINT_UNROUTED");
    assert!(msg.contains("CMT 3"), "names the CMT: {msg}");
    assert!(
        msg.contains("time-to-event"),
        "names the endpoint kind: {msg}"
    );
    assert!(
        msg.contains("read_population_for"),
        "names the loader to use: {msg}"
    );
    assert!(
        msg.contains("subject 1)"),
        "names the first offending subject: {msg}"
    );

    let err = fit(&m, &u, &m.default_params, &FitOptions::default())
        .expect_err("fit() must refuse the unrouted population");
    assert!(err.contains("E_ENDPOINT_UNROUTED"), "fit() error: {err}");
}

/// The routed population — the one the docs prescribe — is silent under both
/// codes and fits.
#[test]
fn fit_accepts_the_routed_joint_population() {
    let m = model(JOINT_MODEL);
    let r = routed(&m, JOINT_DATA);
    let diags = check_model_data(&m, &r);
    assert!(
        !codes(&diags).iter().any(|c| c.starts_with("E_ENDPOINT_")),
        "routed population must pass the guard, got {:?}",
        codes(&diags)
    );
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let res = fit(&m, &r, &m.default_params, &opts).expect("routed joint fit runs");
    assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
}

/// Routed, but the `CMT` column is missing: every row reads as CMT 1, the event
/// CMT has no rows at all, and the reader says nothing. Only the second code can
/// see this shape — the first sees no Gaussian row on CMT 3 because there is none.
#[test]
fn fit_rejects_a_routed_population_whose_endpoint_has_no_rows() {
    let m = model(JOINT_MODEL);
    let f = without_cmt_column(JOINT_DATA);
    let r = routed(&m, f.path().to_str().unwrap());
    assert_eq!(
        total_events(&r),
        0,
        "the dropped column leaves no event records"
    );
    assert_eq!(
        gaussian_rows_on(&r, 3),
        0,
        "…and no Gaussian row on CMT 3 either"
    );

    let diags = check_model_data(&m, &r);
    assert!(
        !codes(&diags).contains(&"E_ENDPOINT_UNROUTED"),
        "the unrouted code must stay silent here — nothing is on CMT 3: {:?}",
        codes(&diags)
    );
    let msg = message_of(&diags, "E_ENDPOINT_NO_RECORDS");
    assert!(msg.contains("CMT=3"), "names the CMT: {msg}");
    assert!(
        msg.contains("`CMT` column"),
        "points at the missing column: {msg}"
    );

    let err = fit(&m, &r, &m.default_params, &FitOptions::default())
        .expect_err("fit() must refuse an endpoint with no rows");
    assert!(err.contains("E_ENDPOINT_NO_RECORDS"), "fit() error: {err}");
}

/// The guard is on the endpoint *family*, not on TTE: a `[binary_model]` gets the
/// same two codes.
#[test]
fn a_binary_endpoint_gets_the_same_two_codes() {
    let m = model(BINARY_MODEL);

    let u = unrouted(BINARY_DATA);
    let msg = message_of(&check_model_data(&m, &u), "E_ENDPOINT_UNROUTED");
    assert!(msg.contains("CMT 3") && msg.contains("binary"), "{msg}");

    let f = without_cmt_column(BINARY_DATA);
    let r = routed(&m, f.path().to_str().unwrap());
    let msg = message_of(&check_model_data(&m, &r), "E_ENDPOINT_NO_RECORDS");
    assert!(msg.contains("CMT=3") && msg.contains("binary"), "{msg}");

    let ok = routed(&m, BINARY_DATA);
    assert!(
        !codes(&check_model_data(&m, &ok))
            .iter()
            .any(|c| c.starts_with("E_ENDPOINT_")),
        "routed binary population must pass"
    );
}

/// `predict()` returned a concentration for the event row; it now panics, per its
/// precondition convention (#898), naming the loader.
#[test]
#[should_panic(expected = "E_ENDPOINT_UNROUTED")]
fn predict_panics_on_an_unrouted_population() {
    let m = model(JOINT_MODEL);
    let u = unrouted(JOINT_DATA);
    let _ = predict(&m, &u, &m.default_params);
}

/// `predict()` on the routed population is unchanged: Gaussian rows only, none of
/// them on the event CMT.
#[test]
fn predict_on_the_routed_population_returns_the_pk_rows_only() {
    let m = model(JOINT_MODEL);
    let r = routed(&m, JOINT_DATA);
    let rows = predict(&m, &r, &m.default_params);
    let n_pk: usize = r.subjects.iter().map(|s| s.obs_times.len()).sum();
    assert!(total_events(&r) > 0);
    assert_eq!(
        rows.len(),
        n_pk,
        "one prediction per Gaussian row and none for the {} event rows",
        total_events(&r)
    );
}

/// `simulate()` gets the unrouted half only. A design template has no event rows
/// by construction (they are what gets generated), so the no-records code must
/// stay silent there — asserted on a routed population with its event records
/// stripped, the exact shape of a template.
#[test]
fn simulate_rejects_an_unrouted_population_but_not_an_event_free_template() {
    let m = model(JOINT_MODEL);

    let u = unrouted(JOINT_DATA);
    let msg = message_of(&check_simulation_data(&m, &u), "E_ENDPOINT_UNROUTED");
    assert!(msg.contains("CMT 3"), "{msg}");
    let err = crate::api::simulate_with_options_diag(
        &m,
        &u,
        &m.default_params,
        1,
        &crate::api::SimulateOptions {
            seed: Some(1),
            // The ODE-hazard horizon precondition runs first; satisfy it so the
            // routing guard is what this call reaches.
            horizon: Some(100.0),
            ..Default::default()
        },
    )
    .expect_err("simulate() must refuse the unrouted population");
    assert!(
        err.contains("E_ENDPOINT_UNROUTED"),
        "simulate() error: {err}"
    );

    let mut template = routed(&m, JOINT_DATA);
    for s in &mut template.subjects {
        s.obs_records.clear();
    }
    assert_eq!(total_events(&template), 0);
    let diags = check_simulation_data(&m, &template);
    assert!(
        !codes(&diags).iter().any(|c| c.starts_with("E_ENDPOINT_")),
        "an event-free routed template must simulate, got {:?}",
        codes(&diags)
    );
}

/// `ferx check` reports the dropped-column case through the same validation path
/// as `fit()` — the CLI-visible half of the fix.
#[test]
fn ferx_check_reports_the_missing_cmt_column() {
    let f = without_cmt_column(JOINT_DATA);
    let report = validate_model_file(JOINT_MODEL, Some(f.path().to_str().unwrap()));
    assert!(!report.valid, "the report must be invalid");
    assert!(
        codes(&report.diagnostics).contains(&"E_ENDPOINT_NO_RECORDS"),
        "got {:?}",
        codes(&report.diagnostics)
    );

    let clean = validate_model_file(JOINT_MODEL, Some(JOINT_DATA));
    assert!(
        !codes(&clean.diagnostics)
            .iter()
            .any(|c| c.starts_with("E_ENDPOINT_")),
        "the intact fixture must pass `ferx check`: {:?}",
        codes(&clean.diagnostics)
    );
}

/// `read_population_routed_by` — the re-read `load_fit`, `run_covariance` and
/// `run_sir` share — is the routed read, not the model-blind one.
#[test]
fn the_shared_reread_is_routed() {
    let m = model(JOINT_MODEL);
    let want = routed(&m, JOINT_DATA);
    let got = crate::api::read_population_routed_by(&m, Path::new(JOINT_DATA), None, &[])
        .expect("re-read");
    assert!(total_events(&want) > 0);
    assert_eq!(total_events(&got), total_events(&want));
    assert_eq!(gaussian_rows_on(&got, 3), 0);
}

/// A `FitResult` for the joint model with its file paths and hashes recorded, as
/// `fit_from_files` produces — the input `run_covariance` / `run_sir` re-read from.
fn joint_fit_from_files() -> crate::types::FitResult {
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    crate::api::fit_from_files(JOINT_MODEL, Some(JOINT_DATA), None, Some(opts))
        .expect("joint fit at the initial estimates")
}

/// `run_covariance` / `run_sir` on a caller-supplied population that was read
/// model-blind: refused by name, before any inner-loop work (the same guard `fit()`
/// runs, since both re-run the inner loop).
#[test]
fn run_covariance_and_run_sir_reject_an_unrouted_population() {
    let m = model(JOINT_MODEL);
    let u = unrouted(JOINT_DATA);
    let fit = joint_fit_from_files();
    let opts = FitOptions::default();
    let err = crate::run_covariance(&fit, Some(&m), Some(&u), &opts)
        .expect_err("unrouted population is refused");
    assert!(err.contains("E_ENDPOINT_UNROUTED"), "{err}");
    let err = crate::run_sir(&fit, Some(&m), Some(&u), &opts)
        .expect_err("unrouted population is refused");
    assert!(err.contains("E_ENDPOINT_UNROUTED"), "{err}");
}

/// With `population = None` both re-read `fit.data_path` — routed, since #1199 (this
/// is the path the R wrapper takes). The re-read is checked by what fails *next*:
/// `run_sir` stops at the missing covariance matrix and `run_covariance` at a
/// deliberately mis-sized `omega`, both downstream of the routing guard, so an
/// unrouted re-read — the pre-fix behaviour — surfaces here as `E_ENDPOINT_UNROUTED`
/// instead (observed by mutation: re-reading through `read_nonmem_csv_mapped` turns
/// both arms red with that code).
#[test]
fn run_covariance_and_run_sir_reread_the_dataset_routed() {
    let fit = joint_fit_from_files();
    let opts = FitOptions::default();
    let err = crate::run_sir(&fit, None, None, &opts).expect_err("no covariance matrix to seed");
    assert!(!err.contains("E_ENDPOINT_"), "{err}");
    assert!(err.contains("covariance_matrix"), "{err}");

    let mut wrong = fit.clone();
    wrong.omega = nalgebra::DMatrix::zeros(2, 2);
    let err = crate::run_covariance(&wrong, None, None, &opts).expect_err("mis-sized omega");
    assert!(!err.contains("E_ENDPOINT_"), "{err}");
    assert!(err.contains("n_eta"), "{err}");
}

/// `predict_categorical` walks `obs_records`, so on a model-blind population it would
/// return nothing — indistinguishable from a model without a binary endpoint. Same
/// panic convention as `predict()`.
#[test]
#[should_panic(expected = "E_ENDPOINT_UNROUTED")]
fn predict_categorical_panics_on_an_unrouted_population() {
    let m = model(BINARY_MODEL);
    let u = unrouted(BINARY_DATA);
    let _ = crate::api::predict_categorical(&m, &u, &m.default_params);
}

/// …and on the routed population it returns one row per binary record.
#[test]
fn predict_categorical_on_the_routed_population_returns_one_row_per_record() {
    let m = model(BINARY_MODEL);
    let r = routed(&m, BINARY_DATA);
    let rows = crate::api::predict_categorical(&m, &r, &m.default_params);
    assert!(total_events(&r) > 0, "the fixture carries binary rows");
    assert_eq!(rows.len(), total_events(&r));
}
