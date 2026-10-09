//! Tier-1 tests for #1491: a population with no scored record is refused, once, with
//! `E_NO_SCORED_OBSERVATIONS`, by the shared `check_model_data_rule` — so `fit()` and
//! `ferx check --data` refuse the same populations.
//!
//! **The gap.** A dataset whose every row was a dose, or whose observations a
//! `[data_selection]` clause removed, fitted to OFV 0 with every parameter at its initial
//! estimate, and `ferx check` said the pair was valid. Nothing can move on such an
//! objective.
//!
//! **The message is three sentences, two of them alternatives behind one gate** — whether
//! `[data_selection]` removed anything. Its input space, and the cell each test holds:
//!
//! | cell | fires | S1 (count) | S2 (filter) | S3 (what scores) |
//! |---|---|---|---|---|
//! | R1 doses only, no filter | yes | yes | no | yes |
//! | R2 filter empties every subject | yes | `0 subject(s)` | yes | no |
//! | R3 filter removes the observations, subjects kept | yes | yes | yes | no |
//! | R5 only `DV = .` observation rows | yes | yes | no | yes |
//! | R7 filter removes dose-only subjects | yes | `0 subject(s)` | no | yes |
//! | R6 in-memory empty population | yes | `0 subject(s)` | no | yes |
//! | R4 / TTE-only / binary-only | no | | | |
//!
//! R1 and R3 sit in **one** test, so a gate stuck on either branch fails half of it.
//! Asserted by code at the check level and by message at `fit()` (a `String` until #1772).

use super::*;

const CODE: &str = "E_NO_SCORED_OBSERVATIONS";
/// S1: the count, which every refusal carries.
const S1: &str = "but nothing the likelihood can score";
/// S1's second sentence, also on every refusal: the consequence, which holds under a
/// `[priors]` / NN-regularization penalty too (#1829 review r1 row 1).
const S1_CONSEQUENCE: &str = "The data would contribute nothing to the objective, so no \
                              estimate could move off its initial value except under a prior \
                              or a regularization penalty.";
/// S2: the filter, behind the gate.
const S2: &str = "`[data_selection]` removed";
/// S2's tail: the action.
const S2_ACTION: &str = "check its `ignore` / `accept` clauses.";
/// S3: what makes a row scored, the other side of the gate.
const S3: &str = "A row is scored when it has `EVID = 0`, `MDV = 0` and a `DV`";
/// S3's tail: the endpoint alternative.
const S3_ENDPOINT: &str = "or when it is routed to a declared endpoint.";

const MODEL: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 0
  covariance = false
";

const DOSES_ONLY: &str = "ID,TIME,DV,AMT,EVID,MDV,CMT\n1,0,.,100,1,1,1\n2,0,.,100,1,1,1\n";
const WITH_OBS: &str = "ID,TIME,DV,AMT,EVID,MDV,CMT\n\
1,0,.,100,1,1,1\n1,1,2.1,.,0,0,1\n1,4,1.9,.,0,0,1\n1,8,1.6,.,0,0,1\n1,12,1.3,.,0,0,1\n\
2,0,.,100,1,1,1\n2,1,2.4,.,0,0,1\n2,4,2.0,.,0,0,1\n2,8,1.7,.,0,0,1\n2,12,1.5,.,0,0,1\n";
const MISSING_DV_ONLY: &str = "ID,TIME,DV,AMT,EVID,MDV,CMT\n\
1,0,.,100,1,1,1\n1,1,.,.,0,0,1\n1,4,.,.,0,0,1\n2,0,.,100,1,1,1\n2,1,.,.,0,0,1\n";
const ONE_OBS: &str = "ID,TIME,DV,AMT,EVID,MDV,CMT\n1,0,.,100,1,1,1\n1,1,2.1,.,0,0,1\n";

/// A model file (with an optional `[data_selection]` body) and a dataset, in a temp dir.
fn files(selection: Option<&str>, csv: &str) -> (tempfile::TempDir, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.ferx");
    let data = dir.path().join("d.csv");
    let text = match selection {
        Some(s) => format!("{MODEL}\n[data_selection]\n  {s}\n"),
        None => MODEL.to_string(),
    };
    std::fs::write(&model, text).unwrap();
    std::fs::write(&data, csv).unwrap();
    let (m, d) = (
        model.to_str().unwrap().to_string(),
        data.to_str().unwrap().to_string(),
    );
    (dir, m, d)
}

/// The check report's findings with this code.
fn coded(report: &CheckReport) -> Vec<&Diagnostic> {
    report
        .diagnostics
        .iter()
        .filter(|d| d.code == CODE)
        .collect()
}

/// `ferx check --data` and the file-path fit on one cell: returns the single diagnostic's
/// message and `fit()`'s error, after asserting the code fires exactly once and the fit is
/// refused with the same text.
fn refused(selection: Option<&str>, csv: &str) -> (String, String) {
    let (_dir, model, data) = files(selection, csv);
    let report = validate_model_file(&model, Some(&data));
    let hits = coded(&report);
    assert!(!report.valid, "check must fail: {:?}", report.diagnostics);
    assert_eq!(
        hits.len(),
        1,
        "{CODE} exactly once: {:?}",
        report.diagnostics
    );
    assert_eq!(hits[0].severity, crate::diagnostics::Severity::Error);
    let err = run_model_with_data(&model, Some(&data))
        .map(|(r, _)| r.ofv)
        .expect_err("fit must refuse a population with nothing to score");
    assert_eq!(err, hits[0].message, "fit() and check carry one message");
    assert!(err.contains(S1_CONSEQUENCE), "S1's consequence: {err}");
    (hits[0].message.clone(), err)
}

/// **Z1 — R1 and R3 side by side: the gate's two branches in one test.**
///
/// R1 (doses only, no filter) must carry S3 and not S2; R3 (`ignore = EVID == 0` removes
/// every observation, subjects kept) must carry S2 with its count phrased as rows "classed as
/// observations" and not S3. Mutations: force the gate true → R1 gains S2; force it false →
/// R3 loses S2; delete S1 / S2 / S3 from the message → the matching assert dies.
#[test]
fn doses_only_and_filtered_out_observations_are_refused_with_the_right_branch() {
    let (r1, _) = refused(None, DOSES_ONLY);
    assert!(
        r1.contains(S1) && r1.contains("has 2 subject(s)"),
        "R1 S1: {r1}"
    );
    assert!(r1.contains(S3), "R1 S3: {r1}");
    assert!(r1.contains(S3_ENDPOINT), "R1 S3's endpoint clause: {r1}");
    assert!(!r1.contains("[data_selection]"), "R1 has no filter: {r1}");

    let (r3, _) = refused(Some("ignore = EVID == 0"), WITH_OBS);
    assert!(
        r3.contains(S1) && r3.contains("has 2 subject(s)"),
        "R3 S1: {r3}"
    );
    assert!(r3.contains(S2), "R3 S2: {r3}");
    assert!(r3.contains(S2_ACTION), "R3 S2's action: {r3}");
    assert!(
        r3.contains("8 row(s) classed as observations"),
        "R3 S2 count, phrased as what the filter could see: {r3}"
    );
    assert!(
        !r3.contains("scored observation"),
        "the filter's count includes rows the reader would not have scored (#1405 r2): {r3}"
    );
    assert!(!r3.contains(S3), "R3 has a filter to blame: {r3}");
}

/// **Z1 — R2: the filter removes every subject.** S1 reads `0 subject(s)` and S2 gives the
/// excluded-subject count.
#[test]
fn a_filter_that_empties_every_subject_is_refused_naming_the_filter() {
    let (r2, _) = refused(Some("ignore = TIME >= 0"), WITH_OBS);
    assert!(
        r2.contains(S1) && r2.contains("has 0 subject(s)"),
        "R2 S1: {r2}"
    );
    assert!(r2.contains(S2), "R2 S2: {r2}");
    assert!(r2.contains("2 subject(s) entirely"), "R2 S2 subjects: {r2}");
    assert!(!r2.contains(S3), "R2: {r2}");
}

/// **Z1 — R7: a filter that removed only dose rows is not blamed.** `ignore = TIME >= 0` on
/// the dose-only dataset removes both subjects entirely but no row classed as an
/// observation, so the cause is the data, not the filter: S3, not S2. The straddle with R2,
/// whose same clause removed observations.
///
/// Mutation: gate on `excluded_subject_ids` as well → this gains S2.
#[test]
fn a_filter_that_removed_only_doses_is_not_named() {
    let (r7, _) = refused(Some("ignore = TIME >= 0"), DOSES_ONLY);
    assert!(
        r7.contains(S1) && r7.contains("has 0 subject(s)"),
        "R7 S1: {r7}"
    );
    assert!(r7.contains(S3), "R7 S3: {r7}");
    assert!(!r7.contains("[data_selection]"), "R7: {r7}");
}

/// **Z1 — R5: observation rows whose every `DV` is `.`.** The reader skips them, so there
/// is nothing to score and no filter to blame.
#[test]
fn observation_rows_with_no_dv_are_refused() {
    let (r5, _) = refused(None, MISSING_DV_ONLY);
    assert!(r5.contains(S1), "R5 S1: {r5}");
    assert!(r5.contains(S3), "R5 S3: {r5}");
    assert!(!r5.contains("[data_selection]"), "R5: {r5}");
}

/// **Z1 — R6: an in-memory population with no subjects, through `fit()` itself.**
#[test]
fn an_empty_in_memory_population_is_refused_by_fit() {
    let (_dir, model, data) = files(None, ONE_OBS);
    let prepared = prepare_run(&model, Some(&data)).expect("prepare");
    let mut population = prepared.population;
    population.subjects.clear();
    population.exclusions = None;
    let diags = check_model_data_rule(
        &prepared.parsed.model,
        &population,
        &IovOccasionRule::Column,
    );
    let hits: Vec<_> = diags.iter().filter(|d| d.code == CODE).collect();
    assert_eq!(hits.len(), 1, "{diags:?}");
    let err = fit(
        &prepared.parsed.model,
        &population,
        &prepared.init_params,
        &prepared.parsed.fit_options,
    )
    .map(|r| r.ofv)
    .expect_err("fit must refuse an empty population");
    assert_eq!(err, hits[0].message);
    assert!(err.contains(S1_CONSEQUENCE), "R6 S1's consequence: {err}");
    assert!(
        err.contains(S1) && err.contains("has 0 subject(s)"),
        "R6 S1: {err}"
    );
    assert!(
        err.contains(S3) && !err.contains("[data_selection]"),
        "R6: {err}"
    );
}

/// **Z2 — R4, one observation: the control.** Fits, and the code is absent; the straddle
/// with Z1 is one row.
#[test]
fn a_single_observation_fits() {
    let (_dir, model, data) = files(None, ONE_OBS);
    let report = validate_model_file(&model, Some(&data));
    assert!(coded(&report).is_empty(), "{:?}", report.diagnostics);
    let (r, _) = run_model_with_data(&model, Some(&data)).expect("one observation fits");
    assert!(r.ofv.is_finite() && r.ofv != 0.0, "ofv {}", r.ofv);
}

/// **Z2 — endpoint-only datasets score their records.** A TTE-only and a binary-only fit,
/// neither with a single Gaussian row: `obs_records` is what the likelihood reads there, so
/// counting `observations` alone would refuse both.
///
/// Mutation: count only `observations.len()` → both refused.
#[cfg(feature = "survival")]
#[test]
fn tte_only_and_binary_only_populations_are_not_refused() {
    for (model, data) in [
        ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
        ("examples/binary_logistic.ferx", "data/binary_logistic.csv"),
    ] {
        let report = validate_model_file(model, Some(data));
        assert!(
            coded(&report).is_empty(),
            "{model}: {:?}",
            report.diagnostics
        );
        let prepared = prepare_run(model, Some(data)).expect("prepare");
        assert!(
            prepared
                .population
                .subjects
                .iter()
                .all(|s| s.observations.is_empty()),
            "premise: {model} carries no Gaussian observation"
        );
        let opts = FitOptions {
            outer_maxiter: 1,
            run_covariance_step: false,
            verbose: false,
            ..prepared.parsed.fit_options.clone()
        };
        let r = fit(
            &prepared.parsed.model,
            &prepared.population,
            &prepared.init_params,
            &opts,
        )
        .unwrap_or_else(|e| panic!("{model} must fit: {e}"));
        assert!(r.ofv.is_finite(), "{model}: ofv {}", r.ofv);
    }
}

/// **Z3 — the more specific code wins `fit()`'s first error.** A TTE model whose endpoint
/// has no record and a dataset with no Gaussian row either: check reports both codes, and
/// `fit()` still errors with `E_ENDPOINT_NO_RECORDS`'s message, as it did before #1491.
///
/// Mutation: put the new rule ahead of `check_endpoint_routing` → `fit()` names the new code.
#[cfg(feature = "survival")]
#[test]
fn an_unrouted_endpoint_still_wins_the_first_error() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("d.csv");
    std::fs::write(
        &data,
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,0,1\n2,0,.,1,100,1,0,1\n",
    )
    .unwrap();
    let data = data.to_str().unwrap();
    let model = "examples/tte_exponential.ferx";
    let report = validate_model_file(model, Some(data));
    let codes: Vec<&str> = report.diagnostics.iter().map(|d| d.code.as_str()).collect();
    assert!(codes.contains(&"E_ENDPOINT_NO_RECORDS"), "{codes:?}");
    assert_eq!(codes.iter().filter(|c| **c == CODE).count(), 1, "{codes:?}");
    let err = run_model_with_data(model, Some(data))
        .map(|(r, _)| r.ofv)
        .expect_err("refused");
    assert!(err.contains("E_ENDPOINT_NO_RECORDS"), "{err}");
    assert!(!err.contains(S1), "{err}");
}

/// **Z4 — the rule is not on the simulate path.** A dose-only design with `DV = .` rows is
/// exactly what `simulate()` takes.
#[test]
fn simulate_still_takes_a_design_with_no_dv() {
    let (_dir, model, data) = files(None, MISSING_DV_ONLY);
    let parsed = crate::parser::model_parser::parse_full_model_file(Path::new(&model)).unwrap();
    let (design, _) = read_population_for_simulation(
        &parsed.model,
        &parsed.covariate_decls,
        &data,
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("design");
    let sims = simulate(&parsed.model, &design, &parsed.model.default_params, 1)
        .expect("simulate takes a design");
    assert!(!sims.is_empty());
}

/// **Z5 — no dataset, nothing to count.** `ferx check` without `--data` never reaches the
/// rule.
#[test]
fn check_without_data_does_not_report_it() {
    let (_dir, model, _data) = files(None, DOSES_ONLY);
    let report = validate_model_file(&model, None);
    assert!(coded(&report).is_empty(), "{:?}", report.diagnostics);
    assert!(report.valid, "{:?}", report.diagnostics);
}
