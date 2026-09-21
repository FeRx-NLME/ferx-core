//! `ferx check --data` reads the dataset through the model file's own
//! `[data_selection]` clauses, so every data-dependent finding describes the
//! records the fit will score (#1465).
//!
//! Before this, `validate_model_file` passed `filter: None` to
//! `read_population_for` while the CLI fit built a `SelectionFilter` from the
//! *same* `parsed.fit_options`, two lines away. Both directions were measured on
//! the issue: `ferx check` rejected a model the fit accepts (a `CMT=2` the clause
//! deletes, reported as `E_PER_CMT_ERROR_MODEL` with exit code 1 — the damaging
//! one, since a script branches on that code), and it *missed* half of a finding
//! the fit reports, describing the unfiltered dataset in the message.
//!
//! **The oracle is the fit on the same two files**, never a restated expectation:
//! the property is that the two entry points read the same records, so anything
//! this file asserts about check is asserted against what `run_model_with_data`
//! did with the identical pair. `[fit_options] maxiter = 0` makes that one
//! objective evaluation — Tier 2, no convergence loop.
//!
//! An equality between the two messages pins *routing* and not content — both
//! sides come from the same producer, so two empty strings, or two identically
//! wrong ones, would satisfy it. Every equality here therefore also asserts a
//! literal from the message it expects.

use ferx_core::{run_model_with_data, validate_model_file, CheckReport, FitResult};
use std::io::Write;
use tempfile::NamedTempFile;

/// Fixture **A** — the rejecting direction. A one-state `[odes]` model whose
/// `[error_model]` declares `CMT=1:` only, against data observed on `{1, 2}`.
/// Without a clause both entry points reject it (`E_PER_CMT_ERROR_MODEL`); with
/// `ignore = CMT == 2` the fit scores only compartment 1, which the single entry
/// covers.
const A_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  CMT=1: DV ~ proportional(PROP)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
";

/// Only **subject 2** carries the CMT-2 rows, so `ignore_subjects = [2]` and
/// `accept = CMT == 1` clear the model–data mismatch by different routes and
/// leave different record sets behind (measured: OFV 245.8442 against 415.9892).
/// A dataset whose every subject carries the offending compartment would let the
/// `ignore_subjects` leg pass on the strength of the `accept` one.
const A_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                     1,0,.,1,100,1,1\n\
                     1,1,9.0,0,.,1,0\n\
                     1,2,8.2,0,.,1,0\n\
                     1,4,6.7,0,.,1,0\n\
                     1,8,4.5,0,.,1,0\n\
                     2,0,.,1,100,1,1\n\
                     2,1,9.4,0,.,1,0\n\
                     2,2,8.0,0,.,1,0\n\
                     2,1,1.2,0,.,2,0\n\
                     2,2,1.0,0,.,2,0\n";

/// Fixture **H** — the mirror. `pk one_cpt_iv` declaring three per-CMT scales
/// against data observed on `{1, 3}`: the finding (`W_PER_CMT_UNMATCHED`) is
/// present either way, and what the filter changes is *which* entries are named
/// and what the message says about them.
const H_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[scaling]
  obs_scale[CMT=1] = 1
  obs_scale[CMT=2] = 2
  obs_scale[CMT=3] = 3

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
";

const H_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                     1,0,.,1,100,1,1\n\
                     1,1,9.0,0,.,1,0\n\
                     1,2,8.2,0,.,1,0\n\
                     1,1,2.7,0,.,3,0\n\
                     1,2,2.4,0,.,3,0\n\
                     2,0,.,1,100,1,1\n\
                     2,1,9.4,0,.,1,0\n\
                     2,2,8.0,0,.,1,0\n\
                     2,1,2.9,0,.,3,0\n\
                     2,2,2.2,0,.,3,0\n";

/// One `EVID=0` row with a missing `DV` and no `MDV=1` — the reader skips it and
/// says so as `W_MISSING_DV`.
const H_MISSING_DV_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                                1,0,.,1,100,1,1\n\
                                1,1,9.0,0,.,1,0\n\
                                1,2,8.2,0,.,1,0\n\
                                1,4,.,0,.,1,0\n\
                                2,0,.,1,100,1,1\n\
                                2,1,9.4,0,.,1,0\n\
                                2,2,8.0,0,.,1,0\n";

/// One observation row whose `CENS` cell is `7` — neither `-1`, `0` nor `1` — and the
/// reader says so as `W_CENS_UNEXPECTED`. The warning is raised while reading,
/// whatever `bloq_method` is: `H_MODEL` sets none, so it runs the default `drop`,
/// which keeps a `CENS` row as an ordinary observation. Only under `m3` would the flag
/// be scored as censored, and then by its sign (`m3_logcdf`: positive is the lower
/// tail, negative the upper).
const H_BAD_CENS_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV,CENS\n\
                              1,0,.,1,100,1,1,0\n\
                              1,1,9.0,0,.,1,0,0\n\
                              1,2,8.2,0,.,1,0,7\n\
                              2,0,.,1,100,1,1,0\n\
                              2,1,9.4,0,.,1,0,0\n\
                              2,2,8.0,0,.,1,0,0\n";

/// No `CMT` column at all, against H's per-CMT scaling — the reader defaults every
/// row to compartment 1 and says so as `W_CMT_DEFAULTED`.
const H_NO_CMT_CSV: &str = "ID,TIME,DV,EVID,AMT,MDV\n\
                            1,0,.,1,100,1\n\
                            1,1,9.0,0,.,0\n\
                            1,2,8.2,0,.,0\n\
                            2,0,.,1,100,1\n\
                            2,1,9.4,0,.,0\n\
                            2,2,8.0,0,.,0\n";

/// A `[data_selection]` block, or nothing at all.
const NO_SELECTION: &str = "";
const IGNORE_CMT2: &str = "\n[data_selection]\n  ignore = CMT == 2\n";
const ACCEPT_CMT1: &str = "\n[data_selection]\n  accept = CMT == 1\n";
const IGNORE_SUBJECT_2: &str = "\n[data_selection]\n  ignore_subjects = [2]\n";
/// A clause that parses and compiles but matches no record in either dataset.
const IGNORE_NOTHING: &str = "\n[data_selection]\n  ignore = CMT == 9\n";
const IGNORE_CMT3: &str = "\n[data_selection]\n  ignore = CMT == 3\n";
/// A clause naming a column the dataset does not have. It compiles, matches
/// nothing, and makes the reader say so.
const IGNORE_ABSENT_COLUMN: &str = "\n[data_selection]\n  ignore = STUDY == 7\n";

fn temp(contents: &str, suffix: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .expect("temp file");
    write!(f, "{contents}").expect("write");
    f.flush().expect("flush");
    f
}

/// `ferx check` and the CLI fit, on the **same two files**.
///
/// `run_model_with_data` is the entry point `ferx model.ferx --data d.csv` runs,
/// so the clauses reach it from the model file alone — the same place
/// `validate_model_file` reads them from, and the only place it can read them
/// from, since it takes no options.
fn check_and_fit(model_src: &str, data: &str) -> (CheckReport, Result<FitResult, String>) {
    let m = temp(model_src, ".ferx");
    let d = temp(data, ".csv");
    let mp = m.path().to_str().expect("utf-8 temp path").to_string();
    let dp = d.path().to_str().expect("utf-8 temp path").to_string();
    let report = validate_model_file(&mp, Some(&dp));
    let fit = run_model_with_data(&mp, Some(&dp)).map(|(r, _)| r);
    (report, fit)
}

/// The error-severity codes of a report, in order.
fn error_codes(report: &CheckReport) -> Vec<String> {
    report
        .diagnostics
        .iter()
        .filter(|d| d.code.starts_with('E'))
        .map(|d| d.code.clone())
        .collect()
}

/// A diagnostic whose message opens with `code`, or a message naming what was
/// there instead — a bare `expect` on `find` says only "None".
fn diagnostic_for<'a>(report: &'a CheckReport, code: &str) -> &'a ferx_core::Diagnostic {
    report
        .diagnostics
        .iter()
        .find(|d| d.message.starts_with(code))
        .unwrap_or_else(|| {
            panic!(
                "no `{code}` in the check report; it carried {:?}",
                report
                    .diagnostics
                    .iter()
                    .map(|d| d.code.as_str())
                    .collect::<Vec<_>>()
            )
        })
}

/// A report serialized with the two fields that necessarily differ between two
/// temp files — the model name (the file stem) and the data path — replaced by
/// constants. Everything else is the report itself.
fn normalized_report(model_src: &str, data: &str) -> String {
    let m = temp(model_src, ".ferx");
    let d = temp(data, ".csv");
    let mut report = validate_model_file(
        m.path().to_str().expect("utf-8 temp path"),
        Some(d.path().to_str().expect("utf-8 temp path")),
    );
    report.model = "<model>".to_string();
    report.data = Some("<data>".to_string());
    serde_json::to_string_pretty(&report).expect("a check report serializes")
}

/// T1 — the issue's first measurement, and the straddle that makes it a test
/// rather than a one-sided table.
///
/// Mutation that must redden it: pass `None` for the filter again (either
/// literally, or as `FitOptions::default()`).
#[test]
fn a_clause_that_empties_an_uncovered_compartment_makes_check_agree_with_the_fit() {
    // With the clause the fit scores compartment 1 only, which the model's single
    // `CMT=1:` entry covers — so check must return a clean report and exit 0.
    let (report, fit) = check_and_fit(&format!("{A_MODEL}{IGNORE_CMT2}"), A_CSV);
    assert_eq!(
        error_codes(&report),
        Vec::<String>::new(),
        "the clause removes every observation the [error_model] does not cover, so \
         `ferx check` has nothing fatal to report: {:?}",
        report.diagnostics
    );
    assert!(
        report.valid,
        "and the report says so: {:?}",
        report.diagnostics
    );
    let ofv = fit
        .expect("the fit accepts the filtered dataset — that is what check must agree with")
        .ofv;
    assert!(
        ofv.is_finite(),
        "the oracle has to be a real evaluation, not a diverged one: OFV = {ofv}"
    );

    // The straddle, in the same test: delete the clause and *both* must reject.
    // Without this half, an implementation that accepts everything passes.
    let (bare, bare_fit) = check_and_fit(&format!("{A_MODEL}{NO_SELECTION}"), A_CSV);
    assert_eq!(
        error_codes(&bare),
        vec!["E_PER_CMT_ERROR_MODEL".to_string()],
        "with no clause the CMT-2 rows are scored and uncovered: {:?}",
        bare.diagnostics
    );
    assert!(
        bare_fit.is_err(),
        "and the fit rejects the same pair — the clause is the only variable"
    );
}

/// T2 — the mirror direction: check reported the finding, but computed it from
/// rows the fit never sees, so it named one dead entry where the fit names two
/// and printed `observed: 1, 3` where the fit prints `observed: 1`.
///
/// Mutation that must redden it: pass `None` for the filter again — independently
/// of T1, which is why both tests exist.
#[test]
fn the_unmatched_entry_warning_describes_the_records_the_fit_scores() {
    let (report, fit) = check_and_fit(&format!("{H_MODEL}{IGNORE_CMT3}"), H_CSV);
    let checked = diagnostic_for(&report, "W_PER_CMT_UNMATCHED")
        .message
        .clone();
    let fitted = fit
        .expect("the fit evaluates this pair")
        .warnings
        .iter()
        .find(|w| w.starts_with("W_PER_CMT_UNMATCHED"))
        .cloned()
        .expect("the fit reports the dead per-CMT entries too");
    assert_eq!(
        checked, fitted,
        "check and the fit must describe the same dataset"
    );

    // Both sides come from the same producer, so the equality above pins routing
    // and not content. These literals are the content: the unfiltered read named
    // `2` against `observed: 1, 3` and carried no `[data_selection]` note at all,
    // because `Population::exclusions` was `None` on that path.
    assert!(
        checked.contains("compartment(s) 2, 3"),
        "the CMT-3 entry is dead once the clause has run: {checked}"
    );
    assert!(
        checked.contains("(observed: 1)"),
        "and only compartment 1 is left observed: {checked}"
    );
    assert!(
        checked.contains("`[data_selection]` clause removed"),
        "with the exclusion attribution #1456 added, which check could never reach: {checked}"
    );
}

/// T3 — the twin. A clause that compiles and matches nothing must leave the
/// report exactly as it was, so what moved is record selection and nothing else.
///
/// Mutation that must redden it: any edit keying a message on the *presence* of a
/// filter or of `Population::exclusions` rather than on the records it removed.
#[test]
fn a_clause_that_matches_no_record_leaves_the_report_byte_identical() {
    for (name, model, csv, literal) in [
        ("A", A_MODEL, A_CSV, "E_PER_CMT_ERROR_MODEL"),
        ("H", H_MODEL, H_CSV, "W_PER_CMT_UNMATCHED"),
    ] {
        let with = normalized_report(&format!("{model}{IGNORE_NOTHING}"), csv);
        let without = normalized_report(&format!("{model}{NO_SELECTION}"), csv);
        assert_eq!(
            with, without,
            "{name}: a clause matching no record is not a change of dataset"
        );
        // Both reports say something, so this is not two empty documents agreeing.
        assert!(
            with.contains(literal),
            "{name}: the fixture must still produce its finding: {with}"
        );
    }
}

/// T4 — `accept` and `ignore_subjects` are separate legs of the builder, and a
/// filter built from `ignore_exprs` alone (the deliberately partial spelling at
/// `validation.rs`'s `data_selection_reads_cmt`) would drop both.
///
/// Mutation that must redden it: build the filter from `ignore_exprs` only; or,
/// for the first leg alone, pass `&[]` for `ignore_subjects`.
#[test]
fn every_clause_kind_reaches_the_check_read() {
    let (by_subject, subject_fit) = check_and_fit(&format!("{A_MODEL}{IGNORE_SUBJECT_2}"), A_CSV);
    assert_eq!(
        error_codes(&by_subject),
        Vec::<String>::new(),
        "dropping subject 2 removes every CMT-2 row: {:?}",
        by_subject.diagnostics
    );
    let subject_ofv = subject_fit
        .expect("the fit accepts the subject-filtered dataset")
        .ofv;

    let (by_accept, accept_fit) = check_and_fit(&format!("{A_MODEL}{ACCEPT_CMT1}"), A_CSV);
    assert_eq!(
        error_codes(&by_accept),
        Vec::<String>::new(),
        "and so does keeping compartment 1 only: {:?}",
        by_accept.diagnostics
    );
    let accept_ofv = accept_fit
        .expect("the fit accepts the accept-filtered dataset")
        .ofv;

    // The straddle: no clause at all and both entry points reject the pair.
    let (bare, bare_fit) = check_and_fit(&format!("{A_MODEL}{NO_SELECTION}"), A_CSV);
    assert_eq!(
        error_codes(&bare),
        vec!["E_PER_CMT_ERROR_MODEL".to_string()],
        "{:?}",
        bare.diagnostics
    );
    assert!(bare_fit.is_err(), "the fit rejects it too");

    // Non-degenerate: the two clause kinds leave *different* record sets behind,
    // so neither leg can be passing on the strength of the other. Measured at
    // `6cbf5dbd` + this fix: 245.8442 (subject 2 gone entirely, 4 observations)
    // against 415.9892 (both subjects, 6 observations).
    assert!(
        subject_ofv.is_finite() && accept_ofv.is_finite(),
        "both evaluations have to be real: {subject_ofv} / {accept_ofv}"
    );
    assert!(
        (subject_ofv - accept_ofv).abs() > 1.0,
        "the two clauses must select different records, or this test has one leg: \
         {subject_ofv} vs {accept_ofv}"
    );
}
/// T5 — a reader warning is reported under the code the reader itself wrote.
///
/// `W_FILTER_COLUMN_ABSENT` is unreachable from `ferx check` until the filter is
/// applied at all; the other two were reachable before, and `W_MISSING_DV` was
/// being reported as `warning[W_DATA]` because the code was chosen from a
/// hand-written list of three prefixes.
///
/// Mutation that must redden it: restore the three-arm `if`/`else if` chain, or
/// strip the prefix extraction's fallback.
#[test]
fn a_reader_warning_is_reported_under_the_code_the_reader_wrote() {
    // Only reachable once check applies the filter: the reader raises this while
    // compiling the clause against the dataset's columns.
    let (absent, absent_fit) = check_and_fit(&format!("{H_MODEL}{IGNORE_ABSENT_COLUMN}"), H_CSV);
    let d = diagnostic_for(&absent, "W_FILTER_COLUMN_ABSENT");
    assert_eq!(
        d.code, "W_FILTER_COLUMN_ABSENT",
        "not the generic `W_DATA`: {}",
        d.message
    );
    assert!(
        absent_fit
            .expect("the fit evaluates this pair")
            .warnings
            .iter()
            .any(|w| *w == d.message),
        "and it is the fit's own sentence, not a re-spelling: {}",
        d.message
    );

    // Reachable before this change, and miscoded: `warning[W_DATA]: W_MISSING_DV: …`.
    let (missing_dv, _) = check_and_fit(H_MODEL, H_MISSING_DV_CSV);
    let d = diagnostic_for(&missing_dv, "W_MISSING_DV");
    assert_eq!(
        d.code, "W_MISSING_DV",
        "the code `check-report.qmd` documents: {}",
        d.message
    );

    // The hand-written arm this replaces. `W_CMT_DEFAULTED` had one, so it is the
    // arm the prefix rule has to reproduce rather than merely not break.
    let (defaulted, _) = check_and_fit(H_MODEL, H_NO_CMT_CSV);
    let d = diagnostic_for(&defaulted, "W_CMT_DEFAULTED");
    assert_eq!(
        d.code, "W_CMT_DEFAULTED",
        "unchanged by the prefix rule: {}",
        d.message
    );

    // The one the first review round found: reachable from `ferx check --data` with
    // no feature gate and no clause, newly coded here (the three-arm chain relayed
    // it as `W_DATA`), and documented in neither code table until #1494's round 1.
    // A *hand-written* list of codes is what let it through 7 mutations — see
    // #1495, which proposes asserting the registries instead of enumerating them.
    let (censored, _) = check_and_fit(H_MODEL, H_BAD_CENS_CSV);
    let d = diagnostic_for(&censored, "W_CENS_UNEXPECTED");
    assert_eq!(
        d.code, "W_CENS_UNEXPECTED",
        "not the generic `W_DATA`: {}",
        d.message
    );
}
