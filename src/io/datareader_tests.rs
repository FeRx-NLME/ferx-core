use super::*;
use crate::types::CovariateKind;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_csv(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f
}

// ── "no happy paths": malformed-input rejection ──────────────────────────
// The reader's validation surface — branches where a bug silently corrupts
// (or crashes on) real NONMEM datasets. All deterministic and fit-free:
// feed malformed CSV, assert the exact error or warning. These error paths
// are otherwise exercised only indirectly, if at all.

#[test]
fn missing_required_columns_are_rejected() {
    // ID / TIME / DV are mandatory; each missing one is a hard error that
    // names the absent column.
    let f = write_csv("TIME,DV,EVID,AMT\n0,.,1,100\n");
    let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
    assert!(err.contains("Missing ID column"), "{err}");

    let f = write_csv("ID,DV,EVID,AMT\n1,.,1,100\n");
    let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
    assert!(err.contains("Missing TIME column"), "{err}");

    let f = write_csv("ID,TIME,EVID,AMT\n1,0,1,100\n");
    let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
    assert!(err.contains("Missing DV column"), "{err}");
}

#[test]
fn dose_rows_without_amt_column_are_rejected() {
    // #753: EVID=1 dose rows but the amount column is named `DOSE`, not `AMT`.
    // Every dose parses to amt=0 → flat objective → fit pinned at init. This is
    // a hard error naming the fix, not a silent bad fit.
    let f = write_csv("ID,TIME,DV,EVID,DOSE\n1,0,.,1,100\n1,1,5.0,0,.\n");
    let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
    assert!(err.contains("E_DOSE_NO_AMT"), "{err}");
}

#[test]
fn dose_free_dataset_without_amt_column_is_ok() {
    // The E_DOSE_NO_AMT guard must be scoped to *dose rows present*: a dataset
    // with only observations (EVID=0) and no AMT column is a legitimate
    // dose-free fit (#262) and must not error.
    let f = write_csv("ID,TIME,DV,EVID\n1,0,1.0,0\n1,1,2.0,0\n");
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.subjects.iter().all(|s| s.doses.is_empty()));
}

#[test]
fn all_zero_amt_doses_warn() {
    // #753: an `AMT` column exists (so E_DOSE_NO_AMT does not fire) but every
    // dose amount is 0 → flat objective. Warn rather than error.
    let f = write_csv("ID,TIME,DV,EVID,AMT\n1,0,.,1,0\n1,1,5.0,0,.\n");
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(
        pop.warnings.iter().any(|w| w.contains("W_ALL_DOSES_ZERO")),
        "{:?}",
        pop.warnings
    );
}

#[test]
fn nonzero_amt_doses_do_not_warn_all_zero() {
    // Guard against a false-positive W_ALL_DOSES_ZERO on a normal dataset.
    let f = write_csv("ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n");
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(
        !pop.warnings.iter().any(|w| w.contains("W_ALL_DOSES_ZERO")),
        "{:?}",
        pop.warnings
    );
}

#[test]
fn ss2_dose_row_is_rejected() {
    // An `SS=2` dose (superimpose steady state without reset) must not load
    // silently as `SS=1` (reset) — reading the dataset fails end-to-end with a
    // message that identifies the row and points at the tracking issue. This
    // is the wiring test for `validate_ss` on the real parse path.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
                   1,0,.,1,100,1,0,1,2,24\n\
                   1,1,5.0,0,.,1,0,0,0,0\n";
    let f = write_csv(csv);
    let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
    assert!(err.contains("SS=2") && err.contains("#694"), "{err}");

    // SS=1 on the same layout still loads (the regimen ferx does support).
    let ok = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
                  1,0,.,1,100,1,0,1,1,24\n\
                  1,1,5.0,0,.,1,0,0,0,0\n";
    let f = write_csv(ok);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.subjects[0].doses[0].ss);
}

#[test]
fn unknown_iov_column_is_rejected() {
    // A requested IOV column that isn't in the header is a hard error, not a
    // silent "no occasions" — otherwise an IOV model would quietly collapse.
    let f = write_csv("ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n");
    let err = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap_err();
    assert!(
        err.contains("iov_column 'OCC'") && err.contains("not found"),
        "{err}"
    );
}

#[test]
fn declared_covariate_column_missing_is_rejected() {
    // `[covariates]` declares WT but the dataset has no WT column → hard
    // error (a silently-vanished covariate would evaluate to nothing).
    let f = write_csv("ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n");
    let decls = vec![CovariateDecl {
        levels: None,
        name: "WT".to_string(),
        kind: CovariateKind::Continuous,
    }];
    let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
    assert!(err.contains(ERR_COV_MISSING_COLUMNS), "{err}");
    assert!(err.contains("WT"), "missing column should be named: {err}");
}

#[test]
fn a_wholly_missing_covariate_reads_as_nan_not_zero() {
    // The column exists (so `check_covariates` passes) but subject 2 has `.`
    // in every row. Leaving the key absent made it resolve to the covariate
    // map's `0.0` default at every evaluation site, so `(WT/70)^0.75` silently
    // contributed `0` for that subject — and `present(WT)`, which is `!is_nan`,
    // read it as present. `NaN` is what makes both of those correct.
    let f = write_csv(
        "ID,TIME,DV,EVID,AMT,WT\n\
         1,0,.,1,100,70\n\
         1,1,5.0,0,.,70\n\
         2,0,.,1,100,.\n\
         2,1,4.0,0,.,.\n",
    );
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert_eq!(pop.subjects[0].covariates["WT"], 70.0);
    assert!(
        pop.subjects[1].covariates["WT"].is_nan(),
        "a subject with no finite value must not read as 0.0"
    );
    assert!(
        pop.warnings
            .iter()
            .any(|w| w.contains("covariate WT has no value for 1 of 2 subjects")),
        "the gap must be reported: {:?}",
        pop.warnings
    );
}

#[test]
fn declared_covariate_non_numeric_value_is_rejected() {
    // A declared covariate must be numerically coded; a text value is a hard
    // error rather than a silent 0.0 that would bias the fit.
    let f = write_csv("ID,TIME,DV,EVID,AMT,WT\n1,0,.,1,100,heavy\n1,1,5.0,0,.,heavy\n");
    let decls = vec![CovariateDecl {
        levels: None,
        name: "WT".to_string(),
        kind: CovariateKind::Continuous,
    }];
    let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
    assert!(err.contains(ERR_COV_NON_NUMERIC), "{err}");
    assert!(
        err.contains("WT"),
        "offending covariate should be named: {err}"
    );
}

#[test]
fn unparseable_iov_occasion_values_warn_not_fail() {
    // A non-numeric occasion value doesn't abort the read: the row is
    // assigned occ=0 and surfaced as a W_IOV_OCC_MISSING population warning
    // so the user can clean the data (a hard error here would be too brittle).
    let f = write_csv("ID,TIME,DV,EVID,AMT,OCC\n1,0,.,1,100,x\n1,1,5.0,0,.,x\n");
    let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
    assert!(
        pop.warnings.iter().any(|w| w.contains("W_IOV_OCC_MISSING")),
        "expected an IOV-occasion warning, got {:?}",
        pop.warnings
    );
}

#[test]
fn test_obs_cmt_dot_defaults_to_compartment_one() {
    // Regression: a "." (missing) CMT on an observation row must default to
    // compartment 1, not 0. `parse_usize(".")` yields 0 — an invalid
    // compartment — so the observation path must guard "." / blank exactly
    // like the dose path does.
    let csv = "ID,TIME,DV,EVID,AMT,CMT\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(
        subj.obs_cmts,
        vec![1],
        "obs CMT='.' must default to compartment 1, not 0"
    );
}

#[test]
fn test_occ_absent_gives_empty_occasions() {
    let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n1,2,3.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.subjects[0].occasions.is_empty());
    assert!(pop.subjects[0].dose_occasions.is_empty());
}

// ── #262: EVID-absent dose inference + dose-coverage warnings ─────────────

#[test]
fn no_evid_column_infers_dose_from_amt() {
    // No EVID column: NONMEM infers a dose from a nonzero AMT. Dose rows here
    // carry AMT>0 with MDV=1 (the #154 shape), which without inference are
    // neither dose (needs EVID 1/4) nor obs (needs EVID 0 & MDV 0) — silently
    // dropped. With inference they administer and the fit is non-degenerate.
    let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,.,1,100\n\
                   1,1,9.5,0,.\n\
                   1,2,7.3,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(
        subj.doses.len(),
        1,
        "AMT>0 row should be inferred as a dose"
    );
    assert_eq!(subj.doses[0].amt, 100.0);
    assert_eq!(subj.observations, vec![9.5, 7.3]);
    // The dataset "just works" — no dose-coverage warnings.
    assert!(
        !pop.warnings
            .iter()
            .any(|w| w.contains("W_AMT_NOT_DOSED") || w.contains("W_NO_DOSES")),
        "inferred-dose dataset must not warn, got {:?}",
        pop.warnings
    );
}

#[test]
fn no_evid_column_infers_multiple_doses_across_subjects() {
    // Two subjects, each with an AMT-coded dose and observations; no EVID.
    let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,.,1,10000\n\
                   1,1,4.2,0,.\n\
                   2,0,.,1,5000\n\
                   2,1,2.1,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert_eq!(pop.subjects.len(), 2);
    assert_eq!(pop.subjects[0].doses.len(), 1);
    assert_eq!(pop.subjects[0].doses[0].amt, 10000.0);
    assert_eq!(pop.subjects[1].doses[0].amt, 5000.0);
}

#[test]
fn no_evid_zero_amt_all_observations_warns_no_doses() {
    // No EVID column and no nonzero AMT anywhere: nothing to infer, so the
    // population parses zero doses. With scored observations present this is
    // almost always a data error — surface the generic W_NO_DOSES backstop.
    let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,1.0,0,.\n\
                   1,1,5.0,0,0\n\
                   1,2,3.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.subjects[0].doses.is_empty());
    assert_eq!(pop.subjects[0].observations.len(), 3);
    assert!(
        pop.warnings.iter().any(|w| w.contains("W_NO_DOSES")),
        "zero-dose population with observations should warn, got {:?}",
        pop.warnings
    );
    // Generic only — no AMT was ignored, so the specific warning stays silent.
    assert!(!pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")));
}

#[test]
fn evid_present_amt_on_nondose_row_warns_amt_not_dosed() {
    // EVID column present (so no inference), but a dose row is mistyped
    // EVID=0 with AMT=5000 and MDV=1 — dropped entirely (not dose, not obs).
    // Its AMT is silently ignored; W_AMT_NOT_DOSED must catch it. The real
    // EVID=1 dose still administers.
    let csv = "ID,TIME,DV,EVID,AMT,MDV\n\
                   1,0,.,1,100,1\n\
                   1,0,.,0,5000,1\n\
                   1,1,5.0,0,.,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(
        subj.doses.len(),
        1,
        "the mistyped AMT=5000 row is not a dose"
    );
    assert_eq!(subj.doses[0].amt, 100.0);
    assert!(
        pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
        "ignored-AMT row should warn, got {:?}",
        pop.warnings
    );
    // Specific wins — the generic backstop must not also fire.
    assert!(!pop.warnings.iter().any(|w| w.contains("W_NO_DOSES")));
}

#[test]
fn wellformed_evid_dataset_emits_no_dose_warnings() {
    // Regression: a normal EVID dataset (dose EVID=1, obs EVID=0) is wholly
    // unaffected — neither dose-coverage warning fires.
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,9.5,0,.\n\
                   1,2,7.3,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert_eq!(pop.subjects[0].doses.len(), 1);
    assert!(
        !pop.warnings
            .iter()
            .any(|w| w.contains("W_AMT_NOT_DOSED") || w.contains("W_NO_DOSES")),
        "well-formed EVID data must not warn, got {:?}",
        pop.warnings
    );
}

#[test]
fn no_evid_inference_mirrored_in_covariate_table() {
    // The covariate table's per-row EVID must agree with how parse_subject
    // classified the row, including AMT-based inference when EVID is absent.
    let csv = "ID,TIME,DV,AMT,MDV,WT\n\
                   1,0,.,100,1,70\n\
                   1,1,5.0,.,0,70\n";
    let f = write_csv(csv);
    let decls = vec![CovariateDecl {
        levels: None,
        name: "WT".to_string(),
        kind: CovariateKind::Continuous,
    }];
    let (pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
    assert_eq!(
        pop.subjects[0].doses.len(),
        1,
        "dose inferred on table path too"
    );
    assert_eq!(table.rows[0].evid, 1, "AMT>0 row's table EVID should be 1");
    assert_eq!(table.rows[1].evid, 0, "obs row's table EVID should be 0");
}

#[test]
fn amt_not_dosed_counted_after_data_selection_filter() {
    // The AMT-ignored count is taken post-filter: a mistyped AMT row that the
    // data-selection filter removes must NOT trip W_AMT_NOT_DOSED, while the
    // same dataset read unfiltered does trip it. Locks the post-filter
    // placement so deliberately excluded dose rows don't cause false alarms.
    let csv = "ID,TIME,DV,EVID,AMT,MDV,STUDY\n\
                   1,0,.,1,100,1,1\n\
                   1,0,.,0,5000,1,2\n\
                   1,1,5.0,0,.,0,1\n";
    let f = write_csv(csv);

    // Unfiltered: the EVID=0/AMT=5000 row is dropped and its AMT flagged.
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(
        pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
        "unfiltered read should flag the ignored AMT, got {:?}",
        pop.warnings
    );

    // Filtered to exclude that row (STUDY==2): nothing is silently dropped,
    // so no warning.
    let filter = SelectionFilter::from_opts(&["STUDY == 2".to_string()], &[], &[]).unwrap();
    let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
    assert!(
        !pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
        "a filter-excluded AMT row must not warn, got {:?}",
        pop.warnings
    );
    assert_eq!(pop.subjects[0].doses.len(), 1);
}

#[test]
fn scored_obs_carrying_amt_does_not_warn_amt_not_dosed() {
    // A *scored* observation (EVID=0, MDV=0) that carries a nonzero AMT —
    // e.g. a pipeline that forward-fills / LOCFs the AMT column across all
    // rows — must NOT trip W_AMT_NOT_DOSED: it is a real observation, not a
    // dropped dose (a NONMEM dose row is MDV=1). The EVID=1 dose administers
    // and both observations are recorded.
    let csv = "ID,TIME,DV,EVID,AMT,MDV\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,100,0\n\
                   1,2,3.0,0,100,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.doses.len(), 1);
    assert_eq!(subj.observations, vec![5.0, 3.0]);
    assert!(
        !pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
        "a scored obs carrying a forward-filled AMT must not warn, got {:?}",
        pop.warnings
    );
}

#[test]
fn nonfinite_amt_is_not_inferred_as_a_dose() {
    // Robustness: a stray non-finite AMT ('inf'/'nan') must not become an
    // infinite/NaN-amount dose. parse_f64 accepts 'inf' (Rust FromStr), so
    // without the is_dosing_amt finiteness guard `amt != 0.0` would be true
    // and the row would infer a bogus dose. With no EVID column it is
    // instead rejected, leaving zero doses.
    let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,.,1,inf\n\
                   1,1,5.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(
        pop.subjects[0].doses.is_empty(),
        "a non-finite AMT must not be inferred as a dose, got {:?}",
        pop.subjects[0].doses
    );
    // The non-finite AMT is also not counted as an ignored dose-like AMT.
    assert!(!pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")));
}

// ── NONMEM coded RATE values (#324) ──────────────────────────────────────
// `RATE` is overloaded: 0 = bolus, >0 = infusion rate, -1 = modeled rate
// (R{n} in $PK), -2 = modeled duration (D{n} in $PK). `-1`/`-2` are accepted
// as `ModeledRate`/`ModeledDuration` (the R{cmt}/D{cmt} existence + engine
// check happens later at the model+data join); other negatives and malformed
// values are still rejected loudly. `validate_dose_rate` is the unit under test.

#[test]
fn validate_dose_rate_classifies_coded_and_malformed_values() {
    // -2 → modeled duration, -1 → modeled rate: both accepted here; the
    // R{cmt}/D{cmt} existence + engine check happens later at the model+data
    // join, where the model is known.
    assert_eq!(
        validate_dose_rate(-2.0, "7", 0.0).unwrap(),
        RateMode::ModeledDuration
    );
    assert_eq!(
        validate_dose_rate(-1.0, "7", 0.0).unwrap(),
        RateMode::ModeledRate
    );

    // Other negatives are not recognised NONMEM codes; the message echoes
    // the offending value so the bad row is identifiable. `-1.5`/`-2.5` are
    // the regression guard for the integer-match: a non-integer must NOT be
    // rounded into the -1/-2 codes (it is rejected, not silently accepted as
    // modeled duration).
    for r in [-0.5, -1.5, -2.5, -3.0, -100.0] {
        let e = validate_dose_rate(r, "1", 0.0).unwrap_err();
        assert!(
            e.contains(&format!("RATE={r}")) && e.contains("negative value"),
            "r={r}: {e}"
        );
    }

    // Non-finite RATE on a dose row is malformed.
    for r in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let e = validate_dose_rate(r, "1", 0.0).unwrap_err();
        assert!(e.contains("not finite"), "r={r}: {e}");
    }

    // Ordinary data-driven rates classify as Fixed.
    assert_eq!(validate_dose_rate(0.0, "1", 0.0).unwrap(), RateMode::Fixed);
    assert_eq!(validate_dose_rate(50.0, "1", 0.0).unwrap(), RateMode::Fixed);
}

// ── NONMEM SS codes: only 0/1 supported ──────────────────────────────────
// The engine stores steady state as a single `DoseEvent.ss: bool`. `SS=1`
// (reset then equilibrate) and `SS=2` (superimpose without reset) are
// *different* regimens, but both used to collapse to `ss = true` (`SS >= 0.5`)
// and run with SS=1 semantics — so an SS=2 record silently produced a wrong
// (reset) profile. `validate_ss` is the unit under test: 0/1 pass, everything
// else (SS=2, other codes, non-integers, non-finite) is rejected loudly.
#[test]
fn validate_ss_accepts_0_and_1_rejects_others() {
    // The only supported codes: 0 = not steady state, 1 = reset then dose to
    // steady state.
    assert!(!validate_ss(0.0, "1", 0.0).unwrap());
    assert!(validate_ss(1.0, "1", 0.0).unwrap());

    // SS=2 (superimpose without reset) is not supported — rejected loudly,
    // not silently collapsed to SS=1. The message names the value, the row,
    // and the tracking issue so the offending record is identifiable.
    let e = validate_ss(2.0, "7", 12.0).unwrap_err();
    assert!(
        e.contains("SS=2") && e.contains("#694") && e.contains("subject 7"),
        "{e}"
    );

    // Other codes and non-integers are not rounded into 0/1 — a stray SS is
    // malformed, not a silent steady-state regimen.
    for s in [3.0, -1.0, 0.5, 1.5] {
        let e = validate_ss(s, "1", 0.0).unwrap_err();
        assert!(e.contains(&format!("SS={s}")), "s={s}: {e}");
    }

    // Non-finite SS on a dose row is malformed (old code: `inf >= 0.5` was a
    // silent steady-state dose).
    for s in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let e = validate_ss(s, "1", 0.0).unwrap_err();
        assert!(e.contains("not finite"), "s={s}: {e}");
    }
}

#[test]
fn coded_rate_minus_one_on_dose_row_loads_as_modeled_rate() {
    // A RATE=-1 dose loads as a modeled-rate dose (the R{cmt} existence +
    // engine check happens later at the model+data join, where the model is
    // known). It must NOT be silently treated as a bolus — `is_infusion()`
    // reports true from the mode even before `resolve_rate` fills the rate.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                   1,0,.,1,100,1,-1,1\n\
                   1,1,5.0,0,.,.,.,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let dose = &pop.subjects[0].doses[0];
    assert_eq!(dose.rate_mode, RateMode::ModeledRate);
    assert_eq!(dose.amt, 100.0);
    assert_eq!(dose.cmt_raw(), 1);
    assert!(
        dose.is_infusion() && !dose.is_fixed(),
        "modeled, not a bolus"
    );
}

#[test]
fn addl_expansion_preserves_coded_rate_mode() {
    // Regression (ADDL + coded RATE): every ADDL-expanded dose of a coded
    // `RATE=-2` (modeled-duration) row must stay modeled, not collapse to a
    // `Fixed` bolus. The additional doses used to be built via
    // `DoseEvent::new` with the raw `-2` sentinel → `rate_mode = Fixed`,
    // `duration = 0` (since `rate <= 0`), so `is_infusion()` was false and
    // each additional dose silently became an instantaneous bolus (and
    // `check_modeled_dose_rates` skips `Fixed` doses, so the missing
    // `D{cmt}` slot was never caught). One modeled infusion followed by N
    // boluses is a silently-wrong regimen on fit/predict/simulate.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,ADDL\n\
                   1,0,.,1,100,1,-2,1,24,2\n\
                   1,1,5.0,0,.,.,.,0,.,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let doses = &pop.subjects[0].doses;
    assert_eq!(doses.len(), 3, "primary + 2 ADDL doses");
    for (k, d) in doses.iter().enumerate() {
        assert_eq!(
            d.rate_mode,
            RateMode::ModeledDuration,
            "dose {k} must stay modeled-duration, not Fixed"
        );
        assert!(
            d.is_infusion() && !d.is_fixed(),
            "dose {k} modeled, not a bolus"
        );
        assert_eq!(d.amt, 100.0);
        assert_eq!(d.cmt_raw(), 1);
    }
    // Additional doses land at time + k*II.
    assert_eq!(doses[1].time, 24.0);
    assert_eq!(doses[2].time, 48.0);
}

#[test]
fn addl_expansion_preserves_modeled_rate() {
    // Mirror of `addl_expansion_preserves_coded_rate_mode` for the OTHER
    // coded mode, `RATE=-1` (modeled rate → `R{cmt}`). Both modes share the
    // single `DoseEvent::modeled(...)` ADDL arm, so this locks in that the
    // `ModeledRate` branch (which `modeled()` tags `InfusionDef::RateDefined`,
    // vs `DurationDefined` for `RATE=-2`) is preserved across ADDL expansion,
    // not just `ModeledDuration`. Without the fix these collapsed to `Fixed`
    // boluses via `DoseEvent::new` with the raw `-1` sentinel.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,ADDL\n\
                   1,0,.,1,100,1,-1,1,24,2\n\
                   1,1,5.0,0,.,.,.,0,.,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let doses = &pop.subjects[0].doses;
    assert_eq!(doses.len(), 3, "primary + 2 ADDL doses");
    for (k, d) in doses.iter().enumerate() {
        assert_eq!(
            d.rate_mode,
            RateMode::ModeledRate,
            "dose {k} must stay modeled-rate, not Fixed"
        );
        assert!(
            d.is_infusion() && !d.is_fixed(),
            "dose {k} modeled, not a bolus"
        );
        assert_eq!(d.amt, 100.0);
        assert_eq!(d.cmt_raw(), 1);
    }
    assert_eq!(doses[1].time, 24.0);
    assert_eq!(doses[2].time, 48.0);
}

#[test]
fn positive_and_zero_rate_doses_still_parse() {
    // Don't break normal infusions/boluses: RATE=50 → duration = amt/rate,
    // RATE=0 → bolus (duration 0).
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                   1,0,.,1,500,1,50,1\n\
                   2,0,.,1,500,1,0,1\n\
                   1,1,5.0,0,.,.,.,0\n\
                   2,1,5.0,0,.,.,.,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let inf = &pop.subjects[0].doses[0];
    assert!(inf.is_infusion() && (inf.duration - 10.0).abs() < 1e-12);
    let bolus = &pop.subjects[1].doses[0];
    assert!(!bolus.is_infusion() && bolus.duration == 0.0);
}

#[test]
fn coded_rate_on_observation_row_is_ignored() {
    // NONMEM only interprets RATE on dose records. A coded RATE on an EVID=0
    // observation row must not error (it is never administered).
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                   1,0,.,1,100,1,0,1\n\
                   1,1,5.0,0,.,.,-1,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert_eq!(pop.subjects[0].doses.len(), 1);
}

#[test]
fn coded_rate_on_filtered_out_dose_row_does_not_error() {
    // The RATE check runs in the dose arm, after the data-selection filter
    // (`continue` on an excluded row). A coded RATE on a row the user IGNOREs
    // must not error — only administered doses are validated.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,FLAG\n\
                   1,0,.,1,100,1,0,1,1\n\
                   1,0.5,.,1,100,1,-2,1,9\n\
                   1,1,5.0,0,.,.,.,0,1\n";
    let f = write_csv(csv);
    let filter = SelectionFilter::from_opts(&["FLAG == 9".to_string()], &[], &[]).unwrap();
    let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
    // The coded-RATE dose row was filtered out; the normal dose survives.
    assert_eq!(pop.subjects[0].doses.len(), 1);
    assert!(!pop.subjects[0].doses[0].is_infusion());
}

#[test]
fn filter_referencing_absent_column_warns() {
    // A filter on a column the data does not have (typo / bare-shorthand on a
    // missing column) can never match. It must surface a W_FILTER_COLUMN_ABSENT
    // warning rather than silently fitting unfiltered data.
    let csv = "ID,TIME,DV,EVID,AMT,FLAG\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n";
    let f = write_csv(csv);
    // `COMENT` (typo of a non-existent column) — no such column in the data.
    let filter = SelectionFilter::from_opts(&["COMENT == X".to_string()], &[], &[]).unwrap();
    let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
    assert!(
        pop.warnings
            .iter()
            .any(|w| w.contains("W_FILTER_COLUMN_ABSENT") && w.contains("coment")),
        "absent filter column must warn, got {:?}",
        pop.warnings
    );
    // No rows excluded (the clause never fires): both records survive.
    assert_eq!(pop.subjects[0].observations, vec![5.0]);
}

#[test]
fn filter_on_present_column_does_not_warn_absent() {
    // Control: a filter naming a real column must NOT emit the absent warning.
    let csv = "ID,TIME,DV,EVID,AMT,FLAG\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,9\n";
    let f = write_csv(csv);
    let filter = SelectionFilter::from_opts(&["FLAG == 9".to_string()], &[], &[]).unwrap();
    let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
    assert!(
        !pop.warnings
            .iter()
            .any(|w| w.contains("W_FILTER_COLUMN_ABSENT")),
        "a present filter column must not warn, got {:?}",
        pop.warnings
    );
}

#[test]
fn test_evid3_reset_recorded_not_dose_or_obs() {
    // EVID=3 is a pure system reset: it must land in `reset_times` and
    // must NOT be parsed as a dose or an observation.
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,5.0,0,.\n\
                   1,5,.,3,.\n\
                   1,6,2.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    // Also the no-shift guard for a *forward* reset (TIME=5 advances past
    // the prior obs at TIME=1): the occasion-segmentation shift must only
    // fire on a restarting clock, so reset_times and obs_times keep their
    // raw values here (a spurious shift would push the reset past 5 and
    // move the t=6 obs).
    assert_eq!(subj.reset_times, vec![5.0]);
    assert!(subj.has_resets());
    // One dose (t=0), two observations (t=1, t=6) — the reset row is neither.
    assert_eq!(subj.doses.len(), 1);
    assert_eq!(subj.doses[0].time, 0.0);
    assert_eq!(subj.obs_times, vec![1.0, 6.0]);
}

#[test]
fn test_filter_on_undeclared_covariate_via_declared_path() {
    // Regression: a `[data_selection]` condition on a covariate that the
    // `[covariates]` block did NOT declare must still fire on the declared
    // read path. Here only WT is declared; `ignore = STUDY == 2` references
    // the undeclared STUDY column. `referenced_covariate_columns()` must
    // pull STUDY into the read union so it lands in each subject's covariate
    // map — otherwise the condition would silently never match.
    // STUDY is a covariate independent of ID: subjects 1 & 2 are in STUDY 1,
    // subjects 3 & 4 in STUDY 2. Filtering on STUDY (not ID) must drop a whole
    // study (3 and 4) while keeping the other (1 and 2). The not-1:1 mapping
    // ensures the test exercises covariate filtering, not ID matching.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,WT,STUDY\n\
                   1,0,.,1,100,1,70,1\n\
                   1,1,5.0,0,.,1,70,1\n\
                   2,0,.,1,100,1,72,1\n\
                   2,1,4.5,0,.,1,72,1\n\
                   3,0,.,1,100,1,80,2\n\
                   3,1,4.0,0,.,1,80,2\n\
                   4,0,.,1,100,1,85,2\n\
                   4,1,3.5,0,.,1,85,2\n";
    let f = write_csv(csv);
    let decls = vec![CovariateDecl {
        levels: None,
        name: "WT".to_string(),
        kind: CovariateKind::Continuous,
    }];
    let filter = SelectionFilter::from_opts(&["STUDY == 2".to_string()], &[], &[]).unwrap();
    let (pop, _table) =
        read_nonmem_csv_with_covariates_filtered(f.path(), &decls, &[], None, &filter).unwrap();
    // Both STUDY==2 subjects (3 and 4) are excluded; STUDY==1 subjects remain.
    let ids: Vec<&str> = pop.subjects.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec!["1", "2"], "only STUDY==1 subjects should remain");
    let excl = pop.exclusions.as_ref().expect("exclusions present");
    assert_eq!(
        excl.excluded_subject_ids,
        vec!["3".to_string(), "4".to_string()]
    );
    assert!(excl.fired_ignore.iter().any(|s| s.contains("STUDY == 2")));
}

#[test]
fn test_no_filter_keeps_subject_with_only_other_events() {
    // Regression: without a [data_selection] filter, a subject made up solely
    // of EVID=2 (other-event) rows has empty doses+observations but must
    // still be retained — matching the pre-feature reader, which pushed every
    // subject unconditionally. The empty-subject skip must be gated on the
    // filter actually having excluded records.
    let csv = "ID,TIME,DV,EVID,AMT,CMT\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   2,0,.,2,.,1\n\
                   2,1,.,2,.,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let ids: Vec<&str> = pop.subjects.iter().map(|s| s.id.as_str()).collect();
    assert!(
        ids.contains(&"2"),
        "subject with only EVID=2 rows must be retained when no filter is active; got {ids:?}"
    );
    assert!(pop.exclusions.is_none(), "no filter → no exclusion summary");
}

#[test]
fn test_evid4_reset_plus_dose_recorded_as_both() {
    // EVID=4 is reset + dose: it records both a reset time and a dose.
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,5.0,0,.\n\
                   1,10,.,4,200\n\
                   1,11,3.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.reset_times, vec![10.0]);
    // Two doses: the t=0 dose and the EVID=4 dose at t=10.
    assert_eq!(subj.doses.len(), 2);
    assert!(subj.doses.iter().any(|d| d.time == 10.0 && d.amt == 200.0));
    assert_eq!(subj.obs_times, vec![1.0, 11.0]);
}

#[test]
fn test_obs_before_same_time_dose_is_pre_dose_trough() {
    // NONMEM record order: an observation listed BEFORE a dose at the same
    // TIME is a pre-dose trough and must sort strictly before that dose.
    // The reader nudges its engine-clock time one ULP below the dose time;
    // the raw user-clock time is preserved for diagnostics.
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,14,5.0,0,.\n\
                   1,14,.,1,100\n\
                   1,28,3.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert!(subj.doses.iter().any(|d| d.time == 14.0));
    // First obs nudged just below the coincident dose at t=14.
    assert_eq!(subj.obs_times[0], 14.0_f64.next_down());
    assert!(subj.obs_times[0] < 14.0);
    // Raw user-clock time is untouched.
    assert_eq!(subj.obs_raw_times[0], 14.0);
    // A non-coincident obs is left exactly as-is.
    assert_eq!(subj.obs_times[1], 28.0);
}

#[test]
fn test_dose_before_same_time_obs_stays_post_dose() {
    // Reverse record order: the dose row precedes the obs row at t=14, so the
    // observation is a post-dose sample and its time is NOT nudged.
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,14,.,1,100\n\
                   1,14,5.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.obs_times, vec![14.0]);
}

#[test]
fn test_obs_before_ss_dose_is_not_nudged() {
    // An SS dose carries its own periodic pre-arrival tail; a sub-dose-time
    // sample would read 0 instead of the SS trough, so SS doses are excluded
    // from the record-order nudge even when the obs precedes them in the file.
    let csv = "ID,TIME,DV,EVID,AMT,SS,II\n\
                   1,0,.,1,100,1,24\n\
                   1,24,5.0,0,.,.,.\n\
                   1,24,.,1,100,1,24\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.obs_times, vec![24.0]);
}

#[test]
fn test_no_resets_leaves_reset_times_empty() {
    let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.subjects[0].reset_times.is_empty());
    assert!(!pop.subjects[0].has_resets());
}

#[test]
fn test_l2_column_populates_obs_l2_and_is_not_a_covariate() {
    // Two paired draws: each L2 id tags a total (FREE=0) + unbound (FREE=1)
    // row of the same sample. obs_l2 is parallel to obs_times.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,FREE,L2\n\
                   1,0,.,1,1,100,0,0\n\
                   1,1,30.0,0,0,.,0,10\n\
                   1,1,3.0,0,0,.,1,10\n\
                   1,5,20.0,0,0,.,0,11\n\
                   1,5,2.0,0,0,.,1,11\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.obs_l2, vec![10, 10, 11, 11]);
    assert_eq!(subj.obs_l2.len(), subj.obs_times.len());
    // L2 is a recognized data item, never surfaced as a covariate.
    assert!(!pop.covariate_names.contains(&"L2".to_string()));
}

#[test]
fn test_no_l2_column_leaves_obs_l2_empty() {
    let csv = "ID,TIME,DV,EVID,MDV,AMT\n1,0,.,1,1,100\n1,1,5.0,0,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.subjects[0].obs_l2.is_empty());
}

#[test]
fn test_off_zero_time_origin_keeps_raw_clock() {
    // Regression #573: data may use calendar/clock TIME values that do not
    // start at zero for each subject. The data reader keeps every TIME on the
    // raw data clock (no per-subject origin shift); the off-zero start is
    // handled by the ODE drivers, which begin integration at the first event
    // (NONMEM semantics) rather than at an artificial t=0. So `obs_times`,
    // `doses[].time`, and `obs_raw_times` all carry the user's TIME values.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CR\n\
                   1,10,.,1,1,100,1.0\n\
                   1,11,5.0,0,0,.,1.0\n\
                   1,14,.,2,1,0,2.0\n\
                   1,15,3.0,0,0,.,2.0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    assert_eq!(subj.doses.len(), 1);
    assert_eq!(subj.doses[0].time, 10.0);
    assert_eq!(subj.obs_times, vec![11.0, 15.0]);
    assert_eq!(subj.obs_raw_times, vec![11.0, 15.0]);
    assert_eq!(subj.pk_only_times, vec![14.0]);
}

#[cfg(feature = "survival")]
#[test]
fn test_tte_entry_time_is_raw_not_origin_shifted() {
    // Regression #573 / left-truncation: a survival subject whose risk-set
    // entry (TENTRY) precedes its first observed record must keep the raw
    // entry time. An earlier subject-relative-origin shift subtracted the
    // first row's TIME and clamped to 0, which silently defeated left
    // truncation (entry is always before the event row, so it collapsed to
    // t=0 and the cumulative hazard integrated from 0 instead of TENTRY).
    use crate::types::{EventType, ObsRecord};
    let tte_cmts: std::collections::HashSet<usize> = [1].into_iter().collect();
    // TIME starts at 100; TENTRY=90 is before the first record.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,TENTRY\n\
                   1,100,1,0,0,.,1,90\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv_routed(
        f.path(),
        None,
        None,
        &[],
        None,
        None,
        &ObsRouting::tte_and_discrete(&tte_cmts, &HashSet::new()),
        &[],
    )
    .map(|(pop, _)| pop)
    .unwrap();
    let recs = &pop.subjects[0].obs_records;
    assert_eq!(recs.len(), 1);
    let ObsRecord::Event {
        time,
        event_type,
        entry_time,
        ..
    } = &recs[0]
    else {
        panic!("expected a TTE Event record");
    };
    assert_eq!(*time, 100.0, "event time stays on the raw data clock");
    assert_eq!(
        *entry_time, 90.0,
        "TENTRY must be the raw value, not shifted to first-row origin or clamped to 0"
    );
    assert!(matches!(event_type, EventType::Exact));
}

// ── Phase 4.0: discrete-state / count observation routing ────────────────
// The reader routes EVID=0 rows on a declared discrete/count CMT into
// `obs_records` as `DiscreteState` / `Count`, with an integer + non-negative
// guard (mirrors the TTE integer-code rule). No endpoint math yet — these
// exercise only the plumbing, via the `_routed` entry point (no parser
// populates the discrete/count sets in Phase 4.0). Default-features tests so
// the per-PR coverage gate measures them.

#[test]
fn discrete_state_cmt_routes_integer_dv_into_obs_records() {
    use crate::types::ObsRecord;
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,0,0,0,.,3\n\
                   1,1,2,0,0,.,3\n\
                   1,2,1,0,0,.,3\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    let subj = &pop.subjects[0];
    assert!(
        subj.observations.is_empty(),
        "discrete rows must route to obs_records, not the Gaussian observation vec"
    );
    let states: Vec<(f64, usize)> = subj
        .obs_records
        .iter()
        .map(|r| match r {
            ObsRecord::DiscreteState {
                time, state, cmt, ..
            } => {
                assert_eq!(*cmt, 3);
                (*time, *state)
            }
            other => panic!("expected DiscreteState, got {other:?}"),
        })
        .collect();
    assert_eq!(states, vec![(0.0, 0), (1.0, 2), (2.0, 1)]);
}

#[test]
fn count_cmt_routes_nonneg_integer_dv_into_obs_records() {
    use crate::types::ObsRecord;
    let routing = ObsRouting {
        count: [4].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,5,0,0,.,4\n\
                   1,1,0,0,0,.,4\n\
                   1,2,12,0,0,.,4\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    let counts: Vec<(f64, u32)> = pop.subjects[0]
        .obs_records
        .iter()
        .map(|r| match r {
            ObsRecord::Count {
                time, count, cmt, ..
            } => {
                assert_eq!(*cmt, 4);
                (*time, *count)
            }
            other => panic!("expected Count, got {other:?}"),
        })
        .collect();
    assert_eq!(counts, vec![(0.0, 5), (1.0, 0), (2.0, 12)]);
}

#[test]
fn discrete_state_endpoint_rejects_noninteger_dv() {
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,1.5,0,0,.,3\n";
    let f = write_csv(csv);
    let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
    assert!(err.contains("non-integer"), "got: {err}");
    assert!(err.contains("discrete-state"), "got: {err}");
}

#[test]
fn discrete_state_endpoint_rejects_negative_dv() {
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,-1,0,0,.,3\n";
    let f = write_csv(csv);
    let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
    assert!(err.contains("negative"), "got: {err}");
}

#[test]
fn count_endpoint_rejects_noninteger_dv() {
    let routing = ObsRouting {
        count: [4].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,2.5,0,0,.,4\n";
    let f = write_csv(csv);
    let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
    assert!(err.contains("non-integer"), "got: {err}");
    assert!(err.contains("count"), "got: {err}");
}

#[test]
fn count_endpoint_rejects_negative_or_overflow_dv() {
    let routing = ObsRouting {
        count: [4].into_iter().collect(),
        ..Default::default()
    };
    // Below 0 and above u32::MAX both fail the count range check.
    for bad in ["-3", "5000000000"] {
        let csv = format!("ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,{bad},0,0,.,4\n");
        let f = write_csv(&csv);
        let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
        assert!(err.contains("out-of-range"), "DV={bad} got: {err}");
    }
}

#[test]
fn obs_routing_rejects_cmt_in_two_endpoint_kinds() {
    // A CMT declared under two endpoint kinds is ambiguous; rejected before
    // any row is read (ObsRouting::validate). Cover all three pairings.
    let cases = [
        ObsRouting {
            tte: [3].into_iter().collect(),
            discrete: [3].into_iter().collect(),
            ..Default::default()
        },
        ObsRouting {
            tte: [3].into_iter().collect(),
            count: [3].into_iter().collect(),
            ..Default::default()
        },
        ObsRouting {
            discrete: [3].into_iter().collect(),
            count: [3].into_iter().collect(),
            ..Default::default()
        },
    ];
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,1,0,0,.,3\n";
    for routing in cases {
        let f = write_csv(csv);
        let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
        assert!(err.contains("only one endpoint type"), "got: {err}");
    }
}

// ── Phase 4.0 regression: DV guard gaps on discrete/count endpoints ───────
// A missing / non-finite DV must NOT silently become a phantom record. The
// integer guard `(dv - dv.round()).abs() > 1e-9` can't reject a non-finite
// `dv` (every NaN/inf comparison is false), and `parse_f64` coerces a missing
// cell to `0.0`; without the finiteness guard and the shared missing-DV skip,
// `.` → `state:0`/`count:0`, `inf` → saturated `usize::MAX`, `NaN` → `0`.

#[test]
fn discrete_and_count_endpoints_skip_missing_dv() {
    // A missing DV on a discrete/count row is treated as MDV=1 (#258): the row
    // is skipped (no phantom `state:0`/`count:0`) and folded into the single
    // W_MISSING_DV summary, exactly like the Gaussian path. `.`, `NA`, and
    // `NaN` (case-insensitive) are all missing tokens (`is_missing_cell`).
    use crate::types::ObsRecord;
    for set_name in ["discrete", "count"] {
        for missing in [".", "NA", "NaN", "na", "nan"] {
            let routing = match set_name {
                "discrete" => ObsRouting {
                    discrete: [3].into_iter().collect(),
                    ..Default::default()
                },
                _ => ObsRouting {
                    count: [3].into_iter().collect(),
                    ..Default::default()
                },
            };
            // Row at t=1 is missing; the t=0 and t=2 rows are valid.
            let csv = format!(
                "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                     1,0,2,0,0,.,3\n\
                     1,1,{missing},0,0,.,3\n\
                     1,2,1,0,0,.,3\n"
            );
            let f = write_csv(&csv);
            let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
            let subj = &pop.subjects[0];
            // Exactly the two valid rows survive — the missing one is dropped,
            // not recorded as a spurious zero at t=1.
            let times: Vec<f64> = subj
                .obs_records
                .iter()
                .map(|r| match r {
                    ObsRecord::DiscreteState { time, .. } | ObsRecord::Count { time, .. } => *time,
                    // Reachable only when non-default endpoint features add
                    // more `ObsRecord` variants; unreachable under default features.
                    #[allow(unreachable_patterns)]
                    other => panic!("{set_name}/{missing}: unexpected record {other:?}"),
                })
                .collect();
            assert_eq!(
                times,
                vec![0.0, 2.0],
                "{set_name}/{missing}: missing-DV row must be skipped, not a phantom zero"
            );
            let warns = pop
                .warnings
                .iter()
                .filter(|w| w.starts_with("W_MISSING_DV"))
                .count();
            assert_eq!(
                warns, 1,
                "{set_name}/{missing}: expected one W_MISSING_DV summary"
            );
        }
    }
}

#[test]
fn discrete_state_endpoint_rejects_nonfinite_dv() {
    // `±inf` (including an overflow like `1e999`) must be rejected, not cast:
    // `inf as usize` saturates to usize::MAX and `-inf as usize` is 0. The
    // integer guard alone misses these because comparisons against a
    // non-finite value are always false. (A literal `NaN`/`NA` cell is a
    // *missing* token — covered by the skip test above, not here.)
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    };
    for bad in ["inf", "-inf", "1e999"] {
        let csv = format!("ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,{bad},0,0,.,3\n");
        let f = write_csv(&csv);
        let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
        assert!(err.contains("non-finite"), "DV={bad} got: {err}");
        assert!(err.contains("discrete-state"), "DV={bad} got: {err}");
    }
}

#[test]
fn count_endpoint_rejects_nonfinite_dv() {
    let routing = ObsRouting {
        count: [4].into_iter().collect(),
        ..Default::default()
    };
    for bad in ["inf", "-inf", "1e999"] {
        let csv = format!("ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,{bad},0,0,.,4\n");
        let f = write_csv(&csv);
        let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
        assert!(err.contains("non-finite"), "DV={bad} got: {err}");
        assert!(err.contains("count"), "DV={bad} got: {err}");
    }
}

#[test]
fn discrete_state_endpoint_rejects_dv_above_usize_range() {
    // A finite but astronomically large integer-valued DV (`1e300`) is beyond
    // usize; without the upper-bound check `1e300 as usize` saturates to
    // usize::MAX silently. Count already had this guard (via `> u32::MAX`);
    // discrete now mirrors it (via the shared exclusive_max bound).
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,1e300,0,0,.,3\n";
    let f = write_csv(csv);
    let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
    assert!(err.contains("out-of-range"), "got: {err}");
}

#[test]
fn count_endpoint_accepts_u32_max() {
    // The upper bound is exclusive at `u32::MAX + 1`, so the full u32 range —
    // including `u32::MAX` itself — is accepted, not rejected. (Locks the
    // boundary against a future "tighten `>=`" that would drop `u32::MAX`.)
    use crate::types::ObsRecord;
    let routing = ObsRouting {
        count: [4].into_iter().collect(),
        ..Default::default()
    };
    let csv = format!("ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,{},0,0,.,4\n", u32::MAX);
    let f = write_csv(&csv);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    assert!(
        matches!(
            pop.subjects[0].obs_records.as_slice(),
            [ObsRecord::Count { count, .. }] if *count == u32::MAX
        ),
        "u32::MAX must be an accepted count, got {:?}",
        pop.subjects[0].obs_records
    );
}

#[test]
fn discrete_state_endpoint_accepts_large_valid_index() {
    // A large but in-`usize` state index (`1e18` < 2^64) is a legitimate value
    // and must be recorded exactly, not rejected by the overflow guard.
    use crate::types::ObsRecord;
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    };
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n1,0,1e18,0,0,.,3\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    assert!(
        matches!(
            pop.subjects[0].obs_records.as_slice(),
            [ObsRecord::DiscreteState { state, .. }] if *state == 1_000_000_000_000_000_000
        ),
        "1e18 must be an accepted state index, got {:?}",
        pop.subjects[0].obs_records
    );
}

#[test]
fn test_evid4_restart_shifts_second_occasion_onto_monotonic_timeline() {
    // Two dosing occasions stacked under one ID, each opened by an EVID=4
    // reset whose TIME column restarts at 0 (NONMEM processes records
    // sequentially, so this is a fresh occasion sharing the first's clock).
    // The reader must shift the second occasion past the first so the two
    // don't collide on the sorted absolute timeline — otherwise both doses
    // land at t=0 and the subject is double-dosed (issue #195).
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,4,100\n\
                   1,2,5.0,0,.\n\
                   1,8,2.0,0,.\n\
                   1,0,.,4,100\n\
                   1,2,4.0,0,.\n\
                   1,8,1.5,0,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    // Two distinct reset times (the second shifted past the first occasion).
    assert_eq!(subj.reset_times.len(), 2);
    assert_eq!(subj.reset_times[0], 0.0);
    assert!(
        subj.reset_times[1] > 8.0,
        "second reset must be shifted past the first occasion's last event (t=8), got {}",
        subj.reset_times[1]
    );

    // Two doses, no longer colliding at t=0.
    assert_eq!(subj.doses.len(), 2);
    assert_eq!(subj.doses[0].time, 0.0);
    assert_eq!(subj.doses[1].time, subj.reset_times[1]);

    // Observation times are strictly increasing across the occasion
    // boundary (the second occasion's relative spacing is preserved).
    assert_eq!(subj.obs_times.len(), 4);
    for w in subj.obs_times.windows(2) {
        assert!(
            w[1] > w[0],
            "obs times must be monotonic: {:?}",
            subj.obs_times
        );
    }
    // Within-occasion spacing is unchanged: second occasion's two obs are
    // still 6 time units apart (raw t=2 and t=8).
    let gap2 = subj.obs_times[3] - subj.obs_times[2];
    assert!(
        (gap2 - 6.0).abs() < 1e-9,
        "second occasion spacing preserved"
    );
}

#[test]
fn test_addl_train_advances_occasion_watermark() {
    // Regression for the issue #195 review: occasion 1 carries an ADDL dose
    // train (II=10, ADDL=3 → boluses at 0,10,20,30) but its only observation
    // is at t=5. A following EVID=4 occasion restarts at TIME=0. The
    // occasion shift must place the new reset past the *whole* ADDL train,
    // not just past the last observation — otherwise the later ADDL boluses
    // (which a reset does not cancel) would land after the reset and
    // contaminate occasion 2.
    let csv = "ID,TIME,DV,EVID,AMT,II,ADDL\n\
                   1,0,.,1,100,10,3\n\
                   1,5,5.0,0,.,.,.\n\
                   1,0,.,4,100,0,0\n\
                   1,5,4.0,0,.,.,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    // Occasion 1 doses: 0,10,20,30 (raw). The single reset (occasion 2)
    // must be shifted strictly past the last ADDL dose at t=30.
    assert_eq!(subj.reset_times.len(), 1);
    let reset = subj.reset_times[0];
    assert!(
        reset > 30.0,
        "occasion-2 reset must be shifted past the ADDL train (last dose t=30), got {reset}"
    );
    // No occasion-1 dose lands at or after the reset (which would inject it
    // into occasion 2). Exactly one dose — occasion 2's own — is >= reset.
    let after_reset = subj.doses.iter().filter(|d| d.time >= reset - 1e-9).count();
    assert_eq!(
        after_reset,
        1,
        "only occasion 2's own dose may sit at/after its reset; doses={:?}",
        subj.doses.iter().map(|d| d.time).collect::<Vec<_>>()
    );
}

#[test]
fn test_parse_subject_reads_occ_column() {
    let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,1\n\
                   1,7,.,1,100,2\n\
                   1,8,4.0,0,.,2\n\
                   1,9,2.5,0,.,2\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
    let subj = &pop.subjects[0];
    // Two obs in occ 1, two in occ 2 (dose rows are stripped from occasions)
    assert_eq!(subj.occasions, vec![1, 1, 2, 2]);
    assert_eq!(subj.dose_occasions, vec![1, 2]);
}

#[test]
fn test_occ_column_excluded_from_covariates() {
    let csv = "ID,TIME,DV,EVID,AMT,OCC,WT\n\
                   1,0,.,1,100,1,70\n\
                   1,1,5.0,0,.,1,70\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
    // OCC should NOT appear as a covariate; WT should
    assert!(!pop.covariate_names.contains(&"OCC".to_string()));
    assert!(pop.covariate_names.contains(&"WT".to_string()));
}

#[test]
fn test_missing_iov_column_errors() {
    let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n";
    let f = write_csv(csv);
    let result = read_nonmem_csv(f.path(), None, Some("OCC"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("iov_column"));
}

#[test]
fn test_parse_occ_recognizes_missing_sentinels() {
    // NONMEM-style "." plus blanks/NAs should parse as None.
    assert_eq!(parse_occ(""), None);
    assert_eq!(parse_occ("."), None);
    assert_eq!(parse_occ("  "), None);
    assert_eq!(parse_occ("NA"), None);
    assert_eq!(parse_occ("nan"), None);
    // Non-integer or signed values that u32 can't parse
    assert_eq!(parse_occ("1.5"), None);
    assert_eq!(parse_occ("-1"), None);
    // Valid u32 round-trips
    assert_eq!(parse_occ("1"), Some(1));
    assert_eq!(parse_occ("42"), Some(42));
}

#[test]
fn test_missing_occ_value_does_not_break_load_but_falls_back_to_zero() {
    // Row with OCC = "." gets occ=0; load still succeeds (warning is
    // emitted to stderr, not asserted here).
    let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
    let subj = &pop.subjects[0];
    // Two obs: first has OCC=1, second had "." → 0
    assert_eq!(subj.occasions, vec![1, 0]);
}

#[test]
fn test_no_tv_covariates_leaves_per_event_snapshots_empty() {
    // WT is constant — no per-event snapshots should be allocated.
    let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,0,.,70\n\
                   1,2,3.0,0,.,70\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert!(!subj.has_tv_covariates());
    assert!(subj.dose_covariates.is_empty());
    assert!(subj.obs_covariates.is_empty());
    // Static covariate map still populated.
    assert_eq!(subj.covariates["WT"], 70.0);
    // Fallback helpers return the static map.
    assert_eq!(subj.dose_cov(0)["WT"], 70.0);
    assert_eq!(subj.obs_cov(0)["WT"], 70.0);
}

#[test]
fn test_tv_covariate_locf_per_event_snapshot() {
    // CR changes mid-subject. Each event must see the LOCF value at its
    // own row's time (NONMEM $PK semantics).
    let csv = "ID,TIME,DV,EVID,AMT,WT,CR\n\
                   1,0,.,1,100,70,1.0\n\
                   1,1,5.0,0,.,70,1.0\n\
                   1,2,3.0,0,.,70,1.5\n\
                   1,3,.,1,100,70,1.5\n\
                   1,4,2.5,0,.,70,2.0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert!(subj.has_tv_covariates());
    // 2 doses, 3 obs.
    assert_eq!(subj.dose_covariates.len(), 2);
    assert_eq!(subj.obs_covariates.len(), 3);
    // Static covariate map is the *first* CR value (1.0), kept for
    // the AD fast-path / fallback.
    assert_eq!(subj.covariates["CR"], 1.0);
    // Dose 1 (t=0): CR=1.0; Dose 2 (t=3): CR=1.5
    assert_eq!(subj.dose_covariates[0]["CR"], 1.0);
    assert_eq!(subj.dose_covariates[1]["CR"], 1.5);
    // Obs at t=1: CR=1.0; t=2: CR=1.5; t=4: CR=2.0
    assert_eq!(subj.obs_covariates[0]["CR"], 1.0);
    assert_eq!(subj.obs_covariates[1]["CR"], 1.5);
    assert_eq!(subj.obs_covariates[2]["CR"], 2.0);
    // WT is constant — but appears in every snapshot at its constant value.
    assert_eq!(subj.dose_covariates[0]["WT"], 70.0);
    assert_eq!(subj.obs_covariates[2]["WT"], 70.0);
}

#[test]
fn test_tv_covariate_snapshot_keeps_dose_sort_alignment() {
    // Doses arrive in non-time order in the CSV. After sorting doses by
    // time, dose_covariates must follow the same permutation so each
    // dose still pairs with its own snapshot.
    let csv = "ID,TIME,DV,EVID,AMT,CR\n\
                   1,5,.,1,100,2.0\n\
                   1,0,.,1,100,1.0\n\
                   1,6,5.0,0,.,2.0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert!(subj.has_tv_covariates());
    // After sorting: dose 0 = t=0 (CR=1.0), dose 1 = t=5 (CR=2.0).
    assert_eq!(subj.doses[0].time, 0.0);
    assert_eq!(subj.doses[1].time, 5.0);
    assert_eq!(subj.dose_covariates[0]["CR"], 1.0);
    assert_eq!(subj.dose_covariates[1]["CR"], 2.0);
}

#[test]
fn test_evid2_rows_captured_with_locf_covariates() {
    // EVID=2 ("other event") rows with TV covariates should be
    // captured into pk_only_times / pk_only_covariates so the
    // event-driven propagator can refresh the rate matrix at the
    // EVID=2 time. NONMEM/nlmixr2 equivalent: $PK runs at every
    // record (including EVID=2), so a covariate change marker
    // should switch CL/V immediately at its time.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CR\n\
                   1,0,.,1,1,100,1.0\n\
                   1,5,.,2,1,0,1.5\n\
                   1,10,5.0,0,0,.,1.5\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    // 1 dose, 1 obs, 1 pk-only event.
    assert_eq!(subj.doses.len(), 1);
    assert_eq!(subj.obs_times.len(), 1);
    assert_eq!(subj.pk_only_times.len(), 1);
    assert_eq!(subj.pk_only_times[0], 5.0);
    // EVID=2 row carries CR=1.5 — must end up in the snapshot.
    assert_eq!(subj.pk_only_covariates[0]["CR"], 1.5);
    // Subsequent obs sees the LOCF value.
    assert_eq!(subj.obs_covariates[0]["CR"], 1.5);
}

#[test]
fn test_evid2_rows_skipped_when_no_tv_covariates() {
    // With time-constant covariates, EVID=2 rows are no-ops in
    // NONMEM ($PK gives the same values), so we don't bother
    // building snapshots for them — saves allocation. This test
    // locks in that optimization.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,WT\n\
                   1,0,.,1,1,100,70\n\
                   1,5,.,2,1,0,70\n\
                   1,10,5.0,0,0,.,70\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    assert!(!subj.has_tv_covariates());
    assert!(subj.pk_only_times.is_empty());
    assert!(subj.pk_only_covariates.is_empty());
}

#[test]
fn test_reset_rows_captured_with_their_own_covariates() {
    // #1133: an EVID=3/4 row is a data record — `$PK` runs at it — so its own covariate
    // values must be captured, not just the reset time. Two resets, each carrying a `CR`
    // that differs from both its predecessor and its successor, so a snapshot taken one
    // row early or one row late is visible in the assertion rather than coincidentally
    // equal.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CR\n\
                   1,0,.,1,1,100,1.0\n\
                   1,2,5.0,0,0,.,1.0\n\
                   1,4,.,3,1,0,2.5\n\
                   1,6,5.0,0,0,.,4.0\n\
                   1,8,.,4,1,50,7.5\n\
                   1,10,5.0,0,0,.,9.0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    assert_eq!(subj.reset_times, vec![4.0, 8.0]);
    assert_eq!(subj.reset_covariates.len(), 2);
    // Parallel to `reset_times`, so the accessor pairs each reset with its own row.
    assert_eq!(subj.reset_cov(0)["CR"], 2.5);
    assert_eq!(subj.reset_cov(1)["CR"], 7.5);
    // EVID=4 also records its dose; that dose row's snapshot is the same row's values.
    assert_eq!(subj.doses.len(), 2);
    assert_eq!(subj.doses[1].time, 8.0);
    assert_eq!(subj.dose_covariates[1]["CR"], 7.5);
}

#[test]
fn test_reset_covariates_skipped_when_no_tv_covariates() {
    // Mirrors `test_evid2_rows_skipped_when_no_tv_covariates`: with time-constant
    // covariates every snapshot is the subject-static map, so the reader builds none and
    // `reset_cov` falls back to it. Locks in that the allocation is skipped and that the
    // fallback — not an empty map — is what consumers see.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,WT\n\
                   1,0,.,1,1,100,70\n\
                   1,4,.,3,1,0,70\n\
                   1,10,5.0,0,0,.,70\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    assert!(!subj.has_tv_covariates());
    assert_eq!(subj.reset_times, vec![4.0]);
    assert!(subj.reset_covariates.is_empty());
    assert_eq!(subj.reset_cov(0)["WT"], 70.0);
}

#[test]
fn test_reset_rows_capture_their_own_occasion() {
    // #1133: an EVID=3/4 row is a data record, so NONMEM runs `$PK` at it under THAT
    // row's `OCC` — measured in `nonmem_anchor/reset_init_snapshot_J.ctl`, where a reset
    // carrying `OCC = 2` between `OCC = 1` records seeds `A_0` under occasion 2 (42.0)
    // and not under the preceding record's occasion 1 (14.0).
    //
    // The reset row's `OCC` differs from BOTH its predecessor (1) and its successor (3),
    // so a snapshot taken one row early or one row late is visible here rather than
    // coincidentally equal — the same non-degeneracy the covariate test above uses. The
    // second reset repeats an occasion already seen, which is the ordinary case.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,OCC\n\
                   1,0,.,1,1,100,1\n\
                   1,2,5.0,0,0,.,1\n\
                   1,4,.,3,1,0,2\n\
                   1,6,5.0,0,0,.,3\n\
                   1,8,.,4,1,50,3\n\
                   1,10,5.0,0,0,.,3\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
    let subj = &pop.subjects[0];

    assert_eq!(subj.reset_times, vec![4.0, 8.0]);
    // Parallel to `reset_times`, and each reset carries its OWN row's label.
    assert_eq!(subj.reset_occasions, vec![2, 3]);
    // The neighbours differ, so this is not the predecessor's or the successor's value:
    // the record before the first reset is OCC=1 and the one after is OCC=3.
    assert_eq!(subj.occasions[0], 1);
    assert_eq!(subj.occasions[1], 3);
    // EVID=4 records its dose from the same row, so the two labels agree there.
    assert_eq!(subj.dose_occasions.len(), 2);
    assert_eq!(subj.dose_occasions[1], 3);
}

#[test]
fn test_reset_occasions_skipped_without_an_iov_column() {
    // Gated exactly like `occasions` / `dose_occasions`: with no `iov_column` the reader
    // stores no occasion labels at all, and `reset_occasions` must be empty rather than a
    // vector of zeros — `pk::reset_row_occasion` distinguishes "no label stored" (fall
    // back to the neighbour scan) from "label is 0" by emptiness, so a zero-filled vector
    // would silently pin every reset to occasion 0.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,OCC\n\
                   1,0,.,1,1,100,1\n\
                   1,4,.,3,1,0,2\n\
                   1,10,5.0,0,0,.,3\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    assert_eq!(subj.reset_times, vec![4.0]);
    assert!(subj.reset_occasions.is_empty());
    assert!(subj.occasions.is_empty());
    assert!(subj.dose_occasions.is_empty());
}

#[test]
fn test_missing_dv_obs_skipped_and_warned() {
    // Issue #258: an EVID=0 row with a missing DV and no MDV=1 must be
    // skipped (not scored as DV=0), and a single W_MISSING_DV warning fires.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,5.0,0,0,.,1\n\
                   1,2,.,0,0,.,1\n\
                   1,3,7.0,0,0,.,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    // Only the two valid observations are scored; the missing-DV row at t=2
    // is skipped (no phantom 0.0 observation, no t=2 entry).
    assert_eq!(subj.observations, vec![5.0, 7.0]);
    assert_eq!(subj.obs_times, vec![1.0, 3.0]);

    // Exactly one summary warning, reporting a single skipped row.
    let warns: Vec<&String> = pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_MISSING_DV"))
        .collect();
    assert_eq!(warns.len(), 1, "expected one W_MISSING_DV summary warning");
    assert!(warns[0].contains("1 observation row"), "got: {}", warns[0]);
}

#[test]
fn test_missing_dv_with_mdv1_no_warning() {
    // The same missing-DV row marked MDV=1 is the documented convention and
    // must NOT trigger the W_MISSING_DV warning (it's already handled).
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,5.0,0,0,.,1\n\
                   1,2,.,0,1,.,1\n\
                   1,3,7.0,0,0,.,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.observations, vec![5.0, 7.0]);
    assert!(
        !pop.warnings.iter().any(|w| w.starts_with("W_MISSING_DV")),
        "MDV=1 missing-DV row should not warn"
    );
}

#[test]
fn test_cens_negative_one_preserved_and_missing_defaults_zero() {
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,CENS\n\
                   1,0,.,1,1,100,1,.\n\
                   1,1,5.0,0,0,.,1,-1\n\
                   1,2,7.0,0,0,.,1,\n\
                   1,3,9.0,0,0,.,1,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    assert_eq!(subj.observations, vec![5.0, 7.0, 9.0]);
    assert_eq!(subj.cens, vec![-1, 0, 1]);
    assert!(subj.has_censored_observation());
    assert!(
        !pop.warnings
            .iter()
            .any(|w| w.starts_with("W_CENS_UNEXPECTED")),
        "valid CENS values (-1/0/1) must not warn"
    );
}

#[test]
fn test_cens_unexpected_value_warns_once() {
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,CENS\n\
                   1,0,.,1,1,100,1,0\n\
                   1,1,5.0,0,0,.,1,2\n\
                   1,2,7.0,0,0,.,1,3\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();

    // Value is preserved verbatim (no silent coercion in the reader)...
    assert_eq!(pop.subjects[0].cens, vec![2, 3]);
    // ...but the out-of-range flag is reported exactly once per subject.
    let n = pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_CENS_UNEXPECTED"))
        .count();
    assert_eq!(n, 1, "expected one W_CENS_UNEXPECTED per subject");
}

#[test]
fn test_missing_dv_count_aggregates_across_subjects() {
    // Issue #258: the per-subject missing-DV counts are summed into ONE
    // population warning. Two subjects, one skipped row each → a single
    // W_MISSING_DV reporting two rows (plural), not two warnings.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,.,0,0,.,1\n\
                   1,2,7.0,0,0,.,1\n\
                   2,0,.,1,1,100,1\n\
                   2,1,5.0,0,0,.,1\n\
                   2,2,.,0,0,.,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();

    // Each subject keeps only its single valid observation.
    assert_eq!(pop.subjects[0].observations, vec![7.0]);
    assert_eq!(pop.subjects[1].observations, vec![5.0]);

    let warns: Vec<&String> = pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_MISSING_DV"))
        .collect();
    assert_eq!(warns.len(), 1, "expected one aggregated W_MISSING_DV");
    assert!(
        warns[0].contains("2 observation row"),
        "expected aggregated count of 2, got: {}",
        warns[0]
    );
}

#[test]
fn test_missing_dv_recognizes_na_nan_and_blank_sentinels() {
    // `is_missing_cell` treats `.`, `NA`/`na`, `NaN`/`nan`, and blank as
    // missing — all of these on a scored obs row must be skipped and counted,
    // not just the `.` sentinel exercised by the other tests.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,NA,0,0,.,1\n\
                   1,2,nan,0,0,.,1\n\
                   1,3,,0,0,.,1\n\
                   1,4,6.0,0,0,.,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];

    // Only the single numeric observation survives.
    assert_eq!(subj.observations, vec![6.0]);
    assert_eq!(subj.obs_times, vec![4.0]);

    let warns: Vec<&String> = pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_MISSING_DV"))
        .collect();
    assert_eq!(warns.len(), 1);
    assert!(
        warns[0].contains("3 observation row"),
        "expected 3 skipped (NA, nan, blank), got: {}",
        warns[0]
    );
}

#[test]
fn test_missing_dv_and_amt_not_dosed_warnings_coexist() {
    // The missing-DV summary (#258) and the dose-coverage summary (#262) are
    // independent population warnings and must both fire when a dataset trips
    // both: a missing-DV scored obs row AND a nonzero-AMT row that is not a
    // dose (EVID=2, MDV=1) so its AMT is ignored.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,.,0,0,.,1\n\
                   1,2,5.0,0,0,.,1\n\
                   1,3,.,2,1,5000,1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();

    assert_eq!(pop.subjects[0].observations, vec![5.0]);
    assert!(
        pop.warnings.iter().any(|w| w.starts_with("W_MISSING_DV")),
        "missing-DV warning should fire; warnings: {:?}",
        pop.warnings
    );
    assert!(
        pop.warnings
            .iter()
            .any(|w| w.starts_with("W_AMT_NOT_DOSED")),
        "AMT-not-dosed warning should fire; warnings: {:?}",
        pop.warnings
    );
}

fn decl(name: &str, kind: CovariateKind) -> CovariateDecl {
    CovariateDecl {
        levels: None,
        name: name.to_string(),
        kind,
    }
}

#[test]
fn test_covariate_table_one_row_per_input_record() {
    // 1 dose + 2 obs = 3 input rows → 3 table rows, including the dose row.
    let csv = "ID,TIME,DV,EVID,AMT,WT,SEX\n\
                   1,0,.,1,100,70,1\n\
                   1,1,5.0,0,.,70,1\n\
                   1,2,3.0,0,.,70,1\n";
    let f = write_csv(csv);
    let decls = vec![
        decl("WT", CovariateKind::Continuous),
        decl("SEX", CovariateKind::Categorical),
    ];
    let (_pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
    assert_eq!(table.names, vec!["WT", "SEX"]);
    assert_eq!(
        table.kinds,
        vec![CovariateKind::Continuous, CovariateKind::Categorical]
    );
    assert_eq!(table.rows.len(), 3);
    // Dose row preserved with EVID=1.
    assert_eq!(table.rows[0].evid, 1);
    assert_eq!(table.rows[0].time, 0.0);
    assert_eq!(table.rows[0].values, vec![70.0, 1.0]);
    assert_eq!(table.rows[1].evid, 0);
    assert_eq!(table.rows[2].time, 2.0);
}

#[test]
fn test_covariate_table_missing_value_is_nan() {
    let csv = "ID,TIME,DV,EVID,AMT,WT,SEX\n\
                   1,0,.,1,100,70,.\n\
                   1,1,5.0,0,.,,1\n";
    let f = write_csv(csv);
    let decls = vec![
        decl("WT", CovariateKind::Continuous),
        decl("SEX", CovariateKind::Categorical),
    ];
    let (_pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
    // Row 0: SEX is "." → NaN. Row 1: WT is blank → NaN.
    assert!(table.rows[0].values[1].is_nan());
    assert!(table.rows[1].values[0].is_nan());
    assert_eq!(table.rows[0].values[0], 70.0);
}

#[test]
fn test_covariate_strict_numeric_errors_on_non_numeric() {
    let csv = "ID,TIME,DV,EVID,AMT,WT,SEX\n\
                   1,0,.,1,100,70,M\n\
                   1,1,5.0,0,.,70,M\n";
    let f = write_csv(csv);
    let decls = vec![
        decl("WT", CovariateKind::Continuous),
        decl("SEX", CovariateKind::Categorical),
    ];
    let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
    assert!(err.contains("non-numeric"), "got: {err}");
    assert!(err.contains("SEX"), "got: {err}");
}

#[test]
fn test_covariate_declared_column_missing_errors() {
    let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n";
    let f = write_csv(csv);
    let decls = vec![
        decl("WT", CovariateKind::Continuous),
        decl("CRCL", CovariateKind::Continuous),
    ];
    let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
    assert!(err.contains("not found"), "got: {err}");
    assert!(err.contains("CRCL"), "got: {err}");
}

#[test]
fn test_parse_evid_defaults_to_observation() {
    // parse_usize defaults to 1; parse_evid must default to 0 (observation)
    // for blank / missing / unparseable cells.
    assert_eq!(parse_evid("1"), 1);
    assert_eq!(parse_evid("0"), 0);
    assert_eq!(parse_evid(""), 0);
    assert_eq!(parse_evid("."), 0);
    assert_eq!(parse_evid("NA"), 0);
    assert_eq!(parse_evid("x"), 0);
}

#[test]
fn test_parse_l2_id_accepts_integer_and_float_formats() {
    // Plain integer ids.
    assert_eq!(parse_l2_id("10"), Some(10));
    assert_eq!(parse_l2_id(" 11 "), Some(11));
    // #830: pandas/R exports float-format the whole column when any row is
    // blank ("10.0"); a strict i64 parse would ungroup everything.
    assert_eq!(parse_l2_id("10.0"), Some(10));
    assert_eq!(parse_l2_id("11.0"), Some(11));
    // A genuinely non-integer value is left ungrouped, NOT rounded into a
    // group — silently mis-grouping would change the correlation pairing.
    assert_eq!(parse_l2_id("2.4"), None);
    assert_eq!(parse_l2_id("10.5"), None);
    // Out-of-range magnitude is rejected rather than saturated.
    assert_eq!(parse_l2_id("1e30"), None);
    // Blank / missing / non-finite / unparseable → ungrouped.
    assert_eq!(parse_l2_id(""), None);
    assert_eq!(parse_l2_id("."), None);
    assert_eq!(parse_l2_id("NA"), None);
    assert_eq!(parse_l2_id("NaN"), None);
    assert_eq!(parse_l2_id("x"), None);
}

#[test]
fn test_covtab_blank_evid_is_observation_not_dose() {
    // A blank EVID cell on an observation row must be EVID=0 in the covtab,
    // not 1 (which parse_usize would have produced).
    let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,,.,70\n";
    let f = write_csv(csv);
    let decls = vec![decl("WT", CovariateKind::Continuous)];
    let (_pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
    assert_eq!(table.rows[0].evid, 1); // explicit dose row
    assert_eq!(table.rows[1].evid, 0); // blank EVID → observation
}

#[test]
fn test_absent_extra_covariate_excluded_from_covariate_names() {
    // A referenced-but-undeclared covariate passed in `extra` that is NOT a
    // real column must not appear in covariate_names — otherwise the fit's
    // E_MISSING_COVARIATE guard would be masked and it would silently read
    // as 0.0. (Regression test for the masking bug.)
    let csv = "ID,TIME,DV,EVID,AMT,WT,CRCL\n\
                   1,0,.,1,100,70,80\n\
                   1,1,5.0,0,.,70,80\n";
    let f = write_csv(csv);
    let decls = vec![decl("WT", CovariateKind::Continuous)];
    let extra = vec!["GHOST".to_string()]; // not a column in the CSV
    let (pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &extra, None).unwrap();
    assert!(!pop.covariate_names.contains(&"GHOST".to_string()));
    assert!(pop.covariate_names.contains(&"WT".to_string()));
    // The table still reflects only declared columns.
    assert_eq!(table.names, vec!["WT"]);
}

#[test]
fn test_column_map_renames_canonical_roles() {
    // `[data]` remapping (#730): a dataset using TAFD/CONC headers is read
    // by mapping them to TIME/DV. Values land on the right roles and the
    // mapped headers do NOT leak into covariate auto-detection.
    let csv = "ID,TAFD,CONC,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,0,.,70\n\
                   1,2,3.0,0,.,70\n";
    let f = write_csv(csv);
    let map = vec![
        ("time".to_string(), "TAFD".to_string()),
        ("dv".to_string(), "CONC".to_string()),
    ];
    let pop = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap();
    let subj = &pop.subjects[0];
    assert_eq!(subj.obs_times, vec![1.0, 2.0]);
    assert_eq!(subj.observations, vec![5.0, 3.0]);
    // WT is still a covariate; the mapped headers are not.
    assert!(pop.covariate_names.contains(&"WT".to_string()));
    assert!(!pop.covariate_names.contains(&"TAFD".to_string()));
    assert!(!pop.covariate_names.contains(&"CONC".to_string()));
}

#[test]
fn test_column_map_is_case_insensitive_on_header() {
    // The actual header is matched case-insensitively: `time = tafd` finds a
    // `TAFD` header.
    let csv = "ID,TAFD,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,7,2.0,0,.\n";
    let f = write_csv(csv);
    let map = vec![("time".to_string(), "tafd".to_string())];
    let pop = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap();
    assert_eq!(pop.subjects[0].obs_times, vec![7.0]);
}

#[test]
fn test_column_map_missing_header_errors() {
    // Mapping a role to a header the dataset lacks is a hard error, not a
    // silent no-op (TIME would otherwise still be reported missing).
    let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,5.0,0,.\n";
    let f = write_csv(csv);
    let map = vec![("dv".to_string(), "CONC".to_string())];
    let err = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap_err();
    assert!(err.contains("mapped column `CONC`"), "{err}");
    assert!(err.contains("renamed to `dv`"), "{err}");
}

#[test]
fn test_column_map_conflicts_with_existing_canonical_column_errors() {
    // Dataset has BOTH a real TIME and a TAFD column; mapping TIME=TAFD is
    // ambiguous and must error rather than silently pick one.
    let csv = "ID,TIME,TAFD,DV,EVID,AMT\n\
                   1,0,0,.,1,100\n\
                   1,1,2,5.0,0,.\n";
    let f = write_csv(csv);
    let map = vec![("TIME".to_string(), "TAFD".to_string())];
    let err = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap_err();
    assert!(err.contains("already has a `TIME` column"), "{err}");
}

#[test]
fn test_column_map_renames_arbitrary_column() {
    // #742: a `[data]` target need not be a canonical role — an arbitrary
    // column can be renamed (e.g. a covariate). `weight` → `WT` and the new
    // name is the one that surfaces as a covariate.
    let csv = "ID,TIME,DV,EVID,AMT,weight\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,0,.,70\n";
    let f = write_csv(csv);
    let map = vec![("WT".to_string(), "weight".to_string())];
    let pop = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap();
    assert!(pop.covariate_names.contains(&"WT".to_string()));
    assert!(!pop.covariate_names.contains(&"weight".to_string()));
    assert_eq!(pop.subjects[0].covariates["WT"], 70.0);
}

#[test]
fn test_column_map_frees_renamed_away_column_for_role() {
    // #742: the dataset's raw `dv` is renamed aside to `ODV`, freeing the DV
    // role for the log-DV column `lndv`. Both renames must succeed — the old
    // `dv` no longer collides with the `DV` target because it is being
    // renamed away in the same block.
    let csv = "ID,TIME,dv,lndv,EVID,AMT\n\
                   1,0,.,.,1,100\n\
                   1,1,5.0,1.609,0,.\n";
    let f = write_csv(csv);
    let map = vec![
        ("ODV".to_string(), "dv".to_string()),
        ("DV".to_string(), "lndv".to_string()),
    ];
    let pop = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap();
    let subj = &pop.subjects[0];
    // DV now carries the log-DV values.
    assert_eq!(subj.observations, vec![1.609]);
    // The raw DV survives as the `ODV` covariate.
    assert!(pop.covariate_names.contains(&"ODV".to_string()));
    assert_eq!(subj.covariates["ODV"], 5.0);
}

#[test]
fn test_column_map_target_collides_with_surviving_column_errors() {
    // #742 boundary: promoting `lndv` to DV without renaming the existing
    // `dv` away leaves two DV columns — ambiguous, so it must error.
    let csv = "ID,TIME,dv,lndv,EVID,AMT\n\
                   1,0,.,.,1,100\n\
                   1,1,5.0,1.609,0,.\n";
    let f = write_csv(csv);
    let map = vec![("DV".to_string(), "lndv".to_string())];
    let err = read_nonmem_csv_mapped(f.path(), None, None, &map).unwrap_err();
    assert!(err.contains("already has a `dv` column"), "{err}");
}

#[test]
fn test_column_map_conflicts_with_iov_column_errors() {
    // Mapping a role onto the header used as the IOV occasion column is a
    // clear error (renaming it would break the occasion lookup).
    let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n";
    let f = write_csv(csv);
    let map = vec![("tentry".to_string(), "OCC".to_string())];
    let err = read_nonmem_csv_mapped(f.path(), None, Some("OCC"), &map).unwrap_err();
    assert!(err.contains("iov_column `OCC`"), "{err}");
}

#[test]
fn test_legacy_read_still_succeeds_with_non_numeric_covariate() {
    // The legacy auto-detect path must remain unchanged (no strict numeric
    // check): a non-numeric covariate column loads without erroring.
    let csv = "ID,TIME,DV,EVID,AMT,SEX\n\
                   1,0,.,1,100,M\n\
                   1,1,5.0,0,.,M\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(pop.covariate_names.contains(&"SEX".to_string()));
}

#[test]
fn test_tv_covariate_locf_handles_missing_intermediate() {
    // Missing CR values are filled forward (LOCF), matching NONMEM.
    let csv = "ID,TIME,DV,EVID,AMT,CR\n\
                   1,0,.,1,100,1.0\n\
                   1,1,5.0,0,.,.\n\
                   1,2,3.0,0,.,2.0\n\
                   1,3,2.0,0,.,.\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let subj = &pop.subjects[0];
    // Obs 0 (t=1, CR missing) → LOCF → 1.0.
    assert_eq!(subj.obs_covariates[0]["CR"], 1.0);
    // Obs 1 (t=2, CR=2.0) → 2.0.
    assert_eq!(subj.obs_covariates[1]["CR"], 2.0);
    // Obs 2 (t=3, CR missing) → LOCF → 2.0.
    assert_eq!(subj.obs_covariates[2]["CR"], 2.0);
}

#[test]
fn test_input_columns_preserves_full_header_order() {
    // input_columns must carry every column in original order and case,
    // including standard columns (ID, TIME, DV, …) that are excluded from
    // covariate_names, and IOV columns.
    let csv = "ID,TIME,DV,EVID,AMT,OCC,WT\n\
                   1,0,.,1,100,1,70\n\
                   1,1,5.0,0,.,1,70\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
    assert_eq!(
        pop.input_columns,
        vec!["ID", "TIME", "DV", "EVID", "AMT", "OCC", "WT"]
    );
    // Standard and IOV columns must not appear in covariate_names.
    assert_eq!(pop.covariate_names, vec!["WT"]);
}

#[test]
fn tte_aware_readers_route_through_gaussian_path_with_empty_tte_cmts() {
    // `read_nonmem_csv_routed` (used by api::read_population_for) carries the
    // TTE routing for [event_model] models, but its column-augmentation / union
    // lines are shared with the Gaussian path. With an empty tte_cmts set it
    // reads exactly like the Gaussian reader; drive it directly with both
    // covariate shapes to cover the augmentation / union / delegation lines.
    // (The cfg(survival) row-routing inside the impl is exercised by the
    // survival job, not here.)
    let no_tte = std::collections::HashSet::new();
    let csv = "ID,TIME,DV,EVID,AMT,WT,STUDY,AGE\n\
                   1,0,.,1,100,70,1,30\n\
                   1,1,5.0,0,.,70,1,30\n\
                   2,0,.,1,100,80,2,40\n\
                   2,1,4.0,0,.,80,2,40\n";
    let f = write_csv(csv);

    // filtered_tte: explicit covariate list, augmented by a filter that
    // references an out-of-list column (STUDY) — exercises the augmentation
    // branch; the filter then drops STUDY==2 (subject 2).
    let cols: &[&str] = &["WT"];
    let filter = SelectionFilter::from_opts(&["STUDY == 2".to_string()], &[], &[]).unwrap();
    let (pop, table_none) = read_nonmem_csv_routed(
        f.path(),
        Some(cols),
        None,
        &[],
        None,
        Some(&filter),
        &ObsRouting::tte_and_discrete(&no_tte, &HashSet::new()),
        &[],
    )
    .unwrap();
    assert!(
        table_none.is_none(),
        "no [covariates] declaration ⇒ no covariate table"
    );
    assert_eq!(
        pop.subjects
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        vec!["1"],
        "STUDY==2 subject should be filtered out via the augmented column"
    );

    // with_covariates_tte: declared WT + an undeclared `extra` (STUDY) + a
    // filter referencing a *third* column (AGE) — exercises BOTH the
    // extra-columns dedup loop and the filter-referenced-column merge, and
    // *validates* the merge: AGE==40 can only drop subject 2 if AGE was
    // actually pulled into the read union, so the assertion fails if the
    // merge regresses.
    let decls = vec![CovariateDecl {
        levels: None,
        name: "WT".to_string(),
        kind: CovariateKind::Continuous,
    }];
    let extra = ["STUDY".to_string()];
    let drop_age40 = SelectionFilter::from_opts(&["AGE == 40".to_string()], &[], &[]).unwrap();
    let (pop2, table) = read_nonmem_csv_routed(
        f.path(),
        None,
        Some(&decls),
        &extra,
        None,
        Some(&drop_age40),
        &ObsRouting::tte_and_discrete(&no_tte, &HashSet::new()),
        &[],
    )
    .unwrap();
    assert!(
        table.is_some(),
        "a [covariates] declaration ⇒ a covariate table"
    );
    // Subject 2 (AGE=40) is dropped via the merged AGE column; subject 1 remains.
    assert_eq!(
        pop2.subjects
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        vec!["1"],
        "AGE==40 must drop subject 2 — proving AGE was pulled into the read union"
    );
}

// ── Missing DV on the simulation path (#957) ─────────────────────────────
// `MissingDvPolicy::KeepAsDesign` reads a `DV = .` row as a design point (a
// sampling time whose observation has not been generated yet) instead of as a
// forgotten `MDV=1` (#258). The default `Skip` behaviour is unchanged; the two
// policies are asserted against the same CSV so a regression in either shows up
// as a diff between them.

/// The canonical simulation template: dosing plus sampling times, `DV = .`
/// everywhere because the DV is what the run is about to produce.
const SIM_TEMPLATE_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                                1,0,.,1,1.0,1,1\n\
                                1,0.25,.,0,.,1,0\n\
                                1,1,.,0,.,1,0\n\
                                1,4,.,0,.,1,0\n";

#[test]
fn keep_as_design_retains_missing_dv_rows_as_sampling_times() {
    let f = write_csv(SIM_TEMPLATE_CSV);
    let routing = ObsRouting::default().with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    let subj = &pop.subjects[0];

    assert_eq!(
        subj.obs_times,
        vec![0.25, 1.0, 4.0],
        "every sampling time of the template must survive"
    );
    assert_eq!(subj.doses.len(), 1, "the dose row is unaffected");
    assert!(
        subj.observations.iter().all(|v| v.is_nan()),
        "a design point carries a NaN placeholder, never a measured 0.0: {:?}",
        subj.observations
    );
    assert!(
        !pop.warnings.iter().any(|w| w.starts_with("W_MISSING_DV")),
        "nothing was skipped, so the skip warning must not fire: {:?}",
        pop.warnings
    );
}

#[test]
fn keep_as_design_reports_the_rows_it_kept() {
    // The rows are read either way, just differently — and either reading
    // changes how many rows the dataset contributes. Reporting only the `Skip`
    // side left a simulation off an observed dataset silently carrying rows the
    // fit had excluded (#957 review).
    let f = write_csv(SIM_TEMPLATE_CSV);
    let routing = ObsRouting::default().with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    let design: Vec<_> = pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_DESIGN_DV"))
        .collect();
    assert_eq!(design.len(), 1, "exactly one summary: {:?}", pop.warnings);
    assert!(
        design[0].contains("3 observation row(s)"),
        "the count must be the rows kept as design points: {}",
        design[0]
    );
}

#[test]
fn keep_as_design_uses_the_registered_placeholder_state_code() {
    use crate::types::ObsRecord;
    // A `state_codes` table need not be 0-based — CTMM states are commonly coded
    // `1`/`2`. A hard `0` placeholder is then a code no `state_codes` lookup can
    // map to a generator index, so a mis-routed design population would score as
    // an out-of-range state instead of failing (#957 review). The reader writes
    // the endpoint's first declared code instead.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,1,.,0,0,.,3\n";
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        design_states: [(3usize, 1usize)].into_iter().collect(),
        ..Default::default()
    }
    .with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let f = write_csv(csv);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    assert!(
        matches!(
            pop.subjects[0].obs_records[0],
            ObsRecord::DiscreteState {
                state: 1,
                cmt: 3,
                ..
            }
        ),
        "got {:?}",
        pop.subjects[0].obs_records[0]
    );

    // An unregistered CMT keeps the `0` default (Binary is `{0,1}`; counts start
    // at 0), so the builder is opt-in per endpoint.
    let unregistered = ObsRouting {
        discrete: [3].into_iter().collect(),
        ..Default::default()
    }
    .with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &unregistered).unwrap();
    assert!(matches!(
        pop.subjects[0].obs_records[0],
        ObsRecord::DiscreteState { state: 0, .. }
    ));
}

#[test]
fn skip_policy_still_drops_every_row_of_the_same_template() {
    // The fitting reading of the exact CSV above — unchanged by #957.
    let f = write_csv(SIM_TEMPLATE_CSV);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &ObsRouting::default()).unwrap();
    assert!(
        pop.subjects[0].obs_times.is_empty(),
        "under Skip a DV-less template scores nothing"
    );
    assert_eq!(
        pop.warnings
            .iter()
            .filter(|w| w.starts_with("W_MISSING_DV"))
            .count(),
        1,
        "…and says so exactly once: {:?}",
        pop.warnings
    );
}

#[test]
fn keep_as_design_still_excludes_mdv1_rows() {
    // MDV=1 is the user explicitly saying "this record is not an observation",
    // which is unambiguous on either path; only the *forgotten* MDV is reread.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                   1,0,.,1,1.0,1,1\n\
                   1,0.25,.,0,.,1,1\n\
                   1,1,.,0,.,1,0\n";
    let f = write_csv(csv);
    let routing = ObsRouting::default().with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    assert_eq!(
        pop.subjects[0].obs_times,
        vec![1.0],
        "the MDV=1 sampling time stays excluded"
    );
}

#[test]
fn keep_as_design_leaves_present_dvs_alone() {
    // A template that already carries values (the placeholder-number workaround
    // users resorted to) must read identically under both policies.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                   1,0,.,1,1.0,1,1\n\
                   1,1,5.0,0,.,1,0\n\
                   1,4,7.0,0,.,1,0\n";
    let f = write_csv(csv);
    let keep = read_nonmem_csv_filtered_routed(
        f.path(),
        &ObsRouting::default().with_missing_dv(MissingDvPolicy::KeepAsDesign),
    )
    .unwrap();
    let skip = read_nonmem_csv_filtered_routed(f.path(), &ObsRouting::default()).unwrap();
    assert_eq!(keep.subjects[0].observations, vec![5.0, 7.0]);
    assert_eq!(keep.subjects[0].observations, skip.subjects[0].observations);
    assert_eq!(keep.subjects[0].obs_times, skip.subjects[0].obs_times);
}

#[test]
fn keep_as_design_places_integer_endpoint_rows_with_a_zero_placeholder() {
    use crate::types::ObsRecord;
    // Discrete-state and count endpoints route to `obs_records`; their design
    // rows need the same treatment, with an integer placeholder (`NaN` is not a
    // state index) that the simulated outcome replaces.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,1,.,0,0,.,3\n\
                   1,2,.,0,0,.,4\n";
    let routing = ObsRouting {
        discrete: [3].into_iter().collect(),
        count: [4].into_iter().collect(),
        ..Default::default()
    }
    .with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let f = write_csv(csv);
    let pop = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap();
    let recs = &pop.subjects[0].obs_records;
    assert_eq!(recs.len(), 2, "both design rows are kept: {recs:?}");
    assert!(
        matches!(
            recs[0],
            ObsRecord::DiscreteState {
                state: 0,
                cmt: 3,
                ..
            }
        ),
        "got {:?}",
        recs[0]
    );
    assert!(
        matches!(
            recs[1],
            ObsRecord::Count {
                count: 0,
                cmt: 4,
                ..
            }
        ),
        "got {:?}",
        recs[1]
    );

    // Under the fitting policy the same rows are still skipped and counted.
    let skipped = read_nonmem_csv_filtered_routed(
        f.path(),
        &ObsRouting {
            discrete: [3].into_iter().collect(),
            count: [4].into_iter().collect(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(skipped.subjects[0].obs_records.is_empty());
    assert!(skipped
        .warnings
        .iter()
        .any(|w| w.starts_with("W_MISSING_DV")));
}

#[test]
fn keep_as_design_still_rejects_an_out_of_range_integer_dv() {
    // Only a *missing* DV becomes a placeholder — a present but invalid integer
    // DV is still the user's data error, on either policy.
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,1,-2,0,0,.,4\n";
    let routing = ObsRouting {
        count: [4].into_iter().collect(),
        ..Default::default()
    }
    .with_missing_dv(MissingDvPolicy::KeepAsDesign);
    let f = write_csv(csv);
    let err = read_nonmem_csv_filtered_routed(f.path(), &routing).unwrap_err();
    assert!(err.contains("out-of-range DV"), "{err}");
}

/// The invariant `api::validation::check_endpoint_routing` (#1199) stands on: a
/// routed CMT never reaches the Gaussian grid. With no routing every event row is
/// a Gaussian observation on its CMT; with the CMT routed, every one of them is an
/// `obs_records` entry and none is in `obs_cmts`. If a reader change ever blurs
/// this, the `E_ENDPOINT_UNROUTED` guard goes blind — this is the test that says so.
#[cfg(feature = "survival")]
#[test]
fn routed_reader_never_places_an_endpoint_cmt_in_the_gaussian_grid() {
    use std::collections::HashSet;
    let path = std::path::Path::new("data/pktte_joint.csv");
    let read = |routing: &ObsRouting| -> (usize, usize) {
        let (pop, _) = read_nonmem_csv_routed(path, None, None, &[], None, None, routing, &[])
            .expect("fixture reads");
        let events: usize = pop.subjects.iter().map(|s| s.obs_records.len()).sum();
        let gaussian_on_3: usize = pop
            .subjects
            .iter()
            .map(|s| s.obs_cmts.iter().filter(|&&c| c == 3).count())
            .sum();
        (events, gaussian_on_3)
    };
    let (ev_unrouted, g3_unrouted) = read(&ObsRouting::default());
    let tte: HashSet<usize> = [3].into_iter().collect();
    let (ev_routed, g3_routed) = read(&ObsRouting::tte_and_discrete(&tte, &HashSet::new()));

    assert!(g3_unrouted > 0, "the fixture has event rows on CMT 3");
    assert_eq!(ev_unrouted, 0, "no routing ⇒ no event records");
    assert_eq!(
        g3_routed, 0,
        "routed ⇒ CMT 3 is never a Gaussian observation"
    );
    assert_eq!(
        ev_routed, g3_unrouted,
        "every CMT-3 row moves from the Gaussian grid to obs_records, none is lost"
    );
}

/// The discrete half of the same invariant: a CMT in the `discrete` routing set
/// (binary / CTMM) never reaches the Gaussian grid either — the `integer_kind` arm
/// precedes the Gaussian push, and `ObsRouting::validate` forbids a CMT in two sets.
#[cfg(feature = "survival")]
#[test]
fn routed_reader_never_places_a_discrete_endpoint_cmt_in_the_gaussian_grid() {
    use std::collections::HashSet;
    let path = std::path::Path::new("data/binary_logistic.csv");
    let read = |routing: &ObsRouting| -> (usize, usize) {
        let (pop, _) = read_nonmem_csv_routed(path, None, None, &[], None, None, routing, &[])
            .expect("fixture reads");
        let records: usize = pop.subjects.iter().map(|s| s.obs_records.len()).sum();
        let gaussian_on_3: usize = pop
            .subjects
            .iter()
            .map(|s| s.obs_cmts.iter().filter(|&&c| c == 3).count())
            .sum();
        (records, gaussian_on_3)
    };
    let (rec_unrouted, g3_unrouted) = read(&ObsRouting::default());
    let discrete: HashSet<usize> = [3].into_iter().collect();
    let (rec_routed, g3_routed) = read(&ObsRouting::tte_and_discrete(&HashSet::new(), &discrete));

    assert!(g3_unrouted > 0, "the fixture has binary rows on CMT 3");
    assert_eq!(rec_unrouted, 0, "no routing ⇒ no discrete records");
    assert_eq!(
        g3_routed, 0,
        "routed ⇒ CMT 3 is never a Gaussian observation"
    );
    assert_eq!(
        rec_routed, g3_unrouted,
        "every CMT-3 row moves from the Gaussian grid to obs_records, none is lost"
    );
}

// ── #1009: the compartment the reader chooses when the dataset does not say ──
// Three sites used to spell the fallback `.unwrap_or(1)` by hand, and a
// float-formatted cell (`"2.0"`) fell through all three and silently dosed
// compartment 1. `parse_cmt_cell` is now the single reader of the cell, and every
// row the reader has to choose for is counted into one `W_CMT_DEFAULTED` summary.

#[test]
fn float_formatted_cmt_cell_reads_as_its_integer() {
    // T1a. The #830 `L2` bug in a second column: pandas/R float-format a whole
    // integer column once any cell in it is blank, so `2.0` is how a real export
    // spells compartment 2. Before the fix `parse::<usize>()` failed on it and
    // the dose landed in compartment 1 with no warning at all.
    //
    // Neither cell may be compartment **1**, and they must differ from each other.
    // The first draft wrote `1.0` on the observation row and asserted
    // `obs_cmts == [1]` — which the *broken* reader also produces, because the
    // strict parse fails and `.unwrap_or(1)` lands on the same answer. Mutating
    // the observation site alone left that assertion green, so the obs half of
    // the fix was untested by the test written for it. Distinct non-1 values also
    // mean a site-swap (obs reading the dose's cell, or vice versa) cannot pass.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,2.0,1\n\
               1,1,5.0,0,.,3.0,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert_eq!(
        pop.subjects[0].doses[0].cmt_1based(),
        2,
        "`2.0` is compartment 2, not the compartment-1 fallback"
    );
    assert_eq!(pop.subjects[0].obs_cmts, vec![3], "`3.0` is compartment 3");
    // A cell the reader could read is not a cell it chose.
    assert!(
        !pop.warnings.iter().any(|w| w.contains("W_CMT_DEFAULTED")),
        "a readable float cell must not be reported as defaulted, got {:?}",
        pop.warnings
    );
}

#[test]
fn missing_and_unparseable_cmt_cells_default_to_1_and_are_counted() {
    // T1b. With the column present, the summary separates the two causes and
    // quotes the offending spellings — `-1` (NONMEM's observation off-switch),
    // `2.5` (genuinely fractional, which must NOT be rounded to 2) and `x`.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,.,1\n\
               1,1,.,1,100,,1\n\
               1,2,.,1,100,NA,1\n\
               1,3,.,1,100,-1,1\n\
               1,4,.,1,100,2.5,1\n\
               1,5,.,1,100,x,1\n\
               1,6,5.0,0,.,1,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    // All six defaulted to compartment 1 — `2.5` included, which a rounding parse
    // would have sent to 2.
    assert_eq!(
        pop.subjects[0]
            .doses
            .iter()
            .map(|d| d.cmt_1based())
            .collect::<Vec<_>>(),
        vec![1; 6]
    );
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(
        w.contains("3 row(s) had a missing CMT cell"),
        "missing-cell count must be 3 (`.`, blank, `NA`), got: {w}"
    );
    assert!(
        w.contains("3 row(s) had a CMT cell that is not a compartment index"),
        "unparseable count must be 3 (`-1`, `2.5`, `x`), got: {w}"
    );
    for quoted in ["\"-1\"", "\"2.5\"", "\"x\""] {
        assert!(w.contains(quoted), "summary must quote {quoted}, got: {w}");
    }
    assert!(
        w.contains("6 dose row(s) and 0 observation row(s)"),
        "row split must be 6 dose / 0 obs, got: {w}"
    );
    // The column IS present, so the absent-column wording must not appear: the
    // two causes are reported by different clauses and T1c pins the other one.
    assert!(
        !w.contains("no CMT column"),
        "a present column must not be reported as absent, got: {w}"
    );
}

#[test]
fn absent_cmt_column_warns_once_with_dose_and_obs_counts() {
    // T1c. The absent-column case, with a decoy `COMPT` header — the misspelling
    // the issue was filed from. The per-row cause counters cannot tell an absent
    // column from a column of missing cells (there is no cell to classify), so
    // the summary reads `cmt_col` directly; this pins that wording.
    let csv = "ID,TIME,DV,EVID,AMT,COMPT,MDV\n\
               1,0,.,1,100,2,1\n\
               1,1,5.0,0,.,1,0\n\
               1,2,4.0,0,.,1,0\n\
               2,0,.,1,100,2,1\n\
               2,1,6.0,0,.,1,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let hits: Vec<&String> = pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_CMT_DEFAULTED"))
        .collect();
    assert_eq!(hits.len(), 1, "one summary, not one per row: {hits:?}");
    let w = hits[0];
    assert!(
        w.contains("the dataset has no CMT column"),
        "must name the absent column, got: {w}"
    );
    assert!(
        w.contains("2 dose row(s) and 3 observation row(s)"),
        "counts must sum across subjects (2 doses, 3 obs), got: {w}"
    );
    // The remedy has to be actionable: both spellings of the fix, and the way to
    // silence it deliberately.
    assert!(w.contains("ordered like the model's compartments"), "{w}");
    assert!(w.contains("CMT = <header>"), "{w}");
    assert!(w.contains("CMT=1"), "{w}");
    // The reader is model-blind, so one message reaches every model class this
    // warning is shown to — and after review round 3 that includes analytical `pk`
    // models. It must therefore not name anything only an `[odes]` model has. The
    // earlier text said "ordered like the model's `states = [...]`" and blamed a
    // NONMEM `DEFDOSE` that is not the first state; measured on
    // `examples/schnider.ferx` (a `pk three_cpt_iv`, no `[odes]` block anywhere) that
    // advice named a block the model does not have, and the DEFDOSE rationale is
    // false there — ADVAN11's DEFDOSE *is* compartment 1.
    for odes_only in ["states = [...]", "DEFDOSE"] {
        assert!(
            !w.contains(odes_only),
            "the message reaches analytical models too, so it must not name \
             `{odes_only}`: {w}"
        );
    }
    // The decoy column is not a CMT column: its `2` must not have been read.
    assert_eq!(pop.subjects[0].doses[0].cmt_1based(), 1);
}

#[test]
fn explicit_integer_cmt_column_including_zero_warns_nothing() {
    // T1d, control. A literal `CMT=0` is NONMEM's own "default compartment"
    // spelling and `check_dose_compartments` accepts it for a bolus; counting it
    // as defaulted would warn on `nonmem_anchor/dose_cmt_ss_cmt0.csv`. The
    // reader reports what the *dataset* did not say, and this one said 0.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,0,1\n\
               1,1,.,1,100,2,1\n\
               1,2,5.0,0,.,1,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert!(
        !pop.warnings.iter().any(|w| w.contains("W_CMT_DEFAULTED")),
        "an explicit CMT column (0 included) is not a default, got {:?}",
        pop.warnings
    );
    // `0` still resolves to compartment 1 downstream (#912); that convention is
    // untouched here — only whether it is *reported* was at stake.
    assert_eq!(pop.subjects[0].doses[0].cmt_1based(), 1);
    assert_eq!(pop.subjects[0].doses[1].cmt_1based(), 2);
}

#[test]
fn filter_context_cmt_resolves_like_the_dose_site() {
    // T1e. The `[data]` selection filter built its `RowContext.cmt` with
    // `parse_usize`, which maps both `.` and `2.0` to **0** — so `ignore = CMT ==
    // 1` failed to drop a dotted row the dose arm assigns to compartment 1. One
    // resolver now serves both, so the filter sees the compartment the row is
    // actually given.
    // The ignore clause is narrowed to dose rows (`EVID == 1 && CMT == 1`) so a
    // dotted *observation* survives — without a surviving defaulted row the warning never
    // fires at all and any assertion about its wording passes by absence. The
    // first revision of this test had that shape: it asserted
    // `!contains("no CMT column")` on a population that emitted no warning
    // whatsoever, so it survived deleting the entire summary block.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,2.0,1\n\
               1,1,.,1,50,.,1\n\
               1,2,5.0,0,.,.,0\n";
    let f = write_csv(csv);
    let filter = SelectionFilter::from_opts(&["EVID == 1 && CMT == 1".to_string()], &[], &[])
        .unwrap_or_else(|e| panic!("filter: {e}"));
    let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
    let cmts: Vec<usize> = pop.subjects[0]
        .doses
        .iter()
        .map(|d| d.cmt_1based())
        .collect();
    assert_eq!(
        cmts,
        vec![2],
        "the dotted dose resolves to 1 and is ignored; the `2.0` dose is compartment 2 and stays"
    );
    // The surviving dotted *observation* is a defaulted row, so the summary does
    // fire — and because the column is present it must use the cell wording, not
    // the absent-column one. Both halves are now load-bearing.
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| {
            panic!(
                "the surviving dotted observation must be reported, else the wording \
                 assertion below is vacuous; got {:?}",
                pop.warnings
            )
        });
    assert!(
        w.contains("0 dose row(s) and 1 observation row(s)"),
        "the excluded dose row is not counted, the kept observation is: {w}"
    );
    assert!(
        !w.contains("no CMT column"),
        "the column is present, so the cause must be the cell one: {w}"
    );
}

#[test]
fn cmt_less_dataset_routes_addl_and_evid_3_4_consistently() {
    // Review follow-up: with no `CMT` column, do the *other* dose-bearing record
    // kinds still work — ADDL expansion, EVID=4 (reset + dose), EVID=3 (pure
    // reset)? They share one resolver with the plain dose row, so the answer
    // should be yes, but "should" is not a measurement and the counting site moved
    // during this review round.
    //
    // Two properties, and they are different: every dose the reader *produces*
    // lands in compartment 1 (routing), while the summary counts dose **rows**
    // (reporting). ADDL is exactly where those two numbers diverge.
    let csv = "ID,TIME,DV,EVID,AMT,MDV,II,ADDL\n\
               1,0,.,1,100,1,12,3\n\
               1,1,5.0,0,.,0,0,0\n\
               1,48,.,3,0,1,0,0\n\
               1,48,.,4,100,1,0,0\n\
               1,49,6.0,0,.,0,0,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let s = &pop.subjects[0];

    // Routing: 1 primary + 3 ADDL + 1 EVID=4 dose = 5 doses, all compartment 1.
    // A single mis-routed expansion would show up as a stray value here.
    assert_eq!(
        s.doses.iter().map(|d| d.cmt_1based()).collect::<Vec<_>>(),
        vec![1; 5],
        "every dose — ADDL expansions and the EVID=4 dose included — defaults alike"
    );
    // EVID=3 is a pure reset: it is neither dose nor observation, so it must not
    // be counted as a compartment the reader chose.
    assert_eq!(s.reset_times.len(), 2, "EVID=3 and EVID=4 both reset");

    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    // Reporting: 2 dose *rows* (the ADDL parent and the EVID=4 row), but 5 dose
    // *events*, because the parent carries `ADDL = 3`. Both numbers appear: the
    // row count is how many cells to fix, the event count how much drug reached
    // the guessed compartment. The EVID=3 reset is absent from both.
    assert!(
        w.contains("2 dose row(s) (5 doses after ADDL expansion) and 2 observation row(s)"),
        "the row count and the ADDL-expanded dose count must both appear: {w}"
    );
}

#[test]
fn addl_expansion_note_is_absent_when_no_row_expands() {
    // Control for the clause above: it must be a straddle, not decoration. With no
    // ADDL column the event count equals the row count, and the parenthetical is
    // suppressed entirely — otherwise every message would carry a redundant
    // "(1 doses after ADDL expansion)".
    let csv = "ID,TIME,DV,EVID,AMT,MDV\n\
               1,0,.,1,100,1\n\
               1,1,5.0,0,.,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(
        w.contains("1 dose row(s) and 1 observation row(s)"),
        "unexpanded rows read plainly: {w}"
    );
    assert!(
        !w.contains("ADDL"),
        "no ADDL note when nothing expanded: {w}"
    );
}

#[test]
fn negative_zero_is_reported_like_every_other_negative() {
    // Review finding. `(0.0..=usize::MAX as f64).contains(&-0.0)` is `true` (IEEE
    // `-0.0 == 0.0`) and `(-0.0f64) as usize` is `0`, so `-0` used to read as
    // compartment 0 — the *default dose compartment*, silently accepted — while
    // `-1` one row below was reported. The straddle is the point: all three
    // spellings are negative, so all three must be reported alike — while the
    // literal `0` on row four must **not** be, even though it doses the same
    // compartment they do. `CMT=0` is NONMEM's default dose compartment and
    // resolves to 1 (#899, `DoseEvent::cmt_1based`), so the two cases are
    // indistinguishable downstream and only the *report* separates them: the
    // reader says what it had to guess at, and `0` was authored. Without this row
    // the comment named a control the fixture did not contain (review round 2).
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,-0,1\n\
               1,1,.,1,100,-0.0,1\n\
               1,2,.,1,100,-1,1\n\
               1,3,.,1,100,0,1\n\
               1,4,5.0,0,.,1,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(
        w.contains("3 row(s) had a CMT cell that is not a compartment index"),
        "all three negatives are reported, `-0` included: {w}"
    );
    for quoted in ["\"-0\"", "\"-0.0\"", "\"-1\""] {
        assert!(w.contains(quoted), "must quote {quoted}: {w}");
    }
    assert!(
        !w.contains("\"0\""),
        "the authored `0` is NONMEM's default dose compartment, not a guess: {w}"
    );
    assert_eq!(
        pop.subjects[0]
            .doses
            .iter()
            .map(|d| d.cmt_1based())
            .collect::<Vec<_>>(),
        vec![1; 4],
        "all four dose compartment 1 — which is why only the report distinguishes them"
    );
}

#[test]
fn a_capped_example_list_says_it_was_capped() {
    // Five distinct unreadable spellings, three retained. Without the ellipsis the
    // message reads `5 row(s) … ("a", "b", "c")`, presenting three spellings as if
    // they were all five — a count and a list that contradict each other.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,aa,1\n\
               1,1,.,1,100,bb,1\n\
               1,2,.,1,100,cc,1\n\
               1,3,.,1,100,dd,1\n\
               1,4,.,1,100,ee,1\n\
               1,5,5.0,0,.,1,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(w.contains("5 row(s) had a CMT cell"), "{w}");
    assert!(
        w.contains("\"aa\", \"bb\", \"cc\", …"),
        "three examples then an ellipsis, so the list is not read as exhaustive: {w}"
    );
    // Straddle: the exactly-three case must NOT carry the ellipsis.
    let csv3 = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                1,0,.,1,100,aa,1\n\
                1,1,.,1,100,bb,1\n\
                1,2,.,1,100,cc,1\n\
                1,3,5.0,0,.,1,0\n";
    let f3 = write_csv(csv3);
    let pop3 = read_nonmem_csv(f3.path(), None, None).unwrap();
    let w3 = pop3
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap();
    assert!(
        w3.contains("\"aa\", \"bb\", \"cc\"") && !w3.contains('…'),
        "a complete list must not claim to be truncated: {w3}"
    );
}

#[test]
fn a_long_or_quoted_example_cell_is_truncated_and_escaped() {
    // The cell is arbitrary user text that reaches `FitResult.warnings`, the fit
    // YAML and the check report. A mis-mapped free-text column would otherwise put
    // a whole sentence in each, and a cell containing `"` would produce `"a"b"`.
    let long = "x".repeat(60);
    // Row three carries U+2028 LINE SEPARATOR. `char::is_control()` is the Cc
    // category only, so U+2028 passes that test — and this string lands in a
    // one-line warning and in the fit YAML, either of which a line break would
    // split (review round 2, nit 9).
    //
    // Exactly `MAX_CMT_EXAMPLES` distinct spellings: a fourth would be withheld and
    // the assertion about it would pass for the wrong reason.
    let csv = format!(
        "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
         1,0,.,1,100,{long},1\n\
         1,1,.,1,100,a\"b,1\n\
         1,2,.,1,100,p\u{2028}q,1\n\
         1,3,5.0,0,.,1,0\n"
    );
    let f = write_csv(&csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(
        !w.contains(&long),
        "the full 60-char cell must not reach the warning: {w}"
    );
    assert!(
        w.contains(&format!("{}…", "x".repeat(24))),
        "truncated at 24 chars with an ellipsis: {w}"
    );
    assert!(
        !w.contains("a\"b"),
        "an embedded quote must not survive verbatim: {w}"
    );
    // The standalone `!w.contains('\u{2028}')` that used to sit here was deleted: the
    // loop below already rejects that character, and two gates excluding the same
    // input are a test hole rather than belt-and-braces — deleting either left the
    // suite green, so neither could fail alone (CLAUDE.md, "two redundant gates cover
    // for each other").
    assert!(
        w.contains("p\u{fffd}q"),
        "it is replaced in place rather than dropped, so the cell stays recognisable: {w}"
    );
    // The property the escaping exists for: no character that breaks a line survives
    // into the message. Asserted over the character set rather than via
    // `w.lines().count()`, which cannot observe this fix at all — `str::lines` splits
    // on `\n` only, so a raw U+2028 leaves the count at 1 and that assertion is green
    // with or without the arm it was written to guard (review round 3).
    for bad in ['\n', '\r', '\u{2028}', '\u{2029}', '\u{85}'] {
        assert!(
            !w.contains(bad),
            "a line-breaking character U+{:04X} reached the warning: {w:?}",
            bad as u32
        );
    }
}

#[test]
fn a_row_shorter_than_its_header_is_a_missing_cell_not_an_absent_column() {
    // The reader is `.flexible(true)`, so a ragged row is accepted and never
    // padded. `row.get(cmt_col)` is then `None` on a dataset that *does* declare
    // CMT. Treating that as `NoColumn` left both cause counters at zero while the
    // summary picked its clause from `cmt_col.is_some()`, so the message came out
    // as `W_CMT_DEFAULTED: , so 1 dose row(s) …` — an empty cause clause. The two
    // sources of "why" must agree.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100\n\
               1,1,5.0,0,.,2,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(
        w.contains("1 row(s) had a missing CMT cell"),
        "a short row is a missing cell: {w}"
    );
    assert!(
        !w.contains("no CMT column"),
        "the header does declare CMT: {w}"
    );
    // The bug this pins is a *malformed* message, so assert its shape directly:
    // no empty cause clause between the code and the counts.
    assert!(!w.contains("W_CMT_DEFAULTED: ,"), "empty cause clause: {w}");
    assert_eq!(pop.subjects[0].doses[0].cmt_1based(), 1);
}

#[test]
fn emitted_cmt_defaulted_message_classifies_as_data_quality() {
    // T1g. Classified against the string the reader *actually builds*, not a
    // paraphrase: `classify_warning` is a long else-if chain of substring arms,
    // and the summary quotes user cell text, so the only honest guard is to feed
    // it the real message. Both spellings — absent column and bad cells — since
    // they are different sentences.
    let absent = write_csv(
        "ID,TIME,DV,EVID,AMT,MDV\n\
         1,0,.,1,100,1\n\
         1,1,5.0,0,.,0\n",
    );
    let cells = write_csv(
        "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
         1,0,.,1,100,.,1\n\
         1,1,.,1,100,-1,1\n\
         1,2,5.0,0,.,1,0\n",
    );
    for f in [&absent, &cells] {
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let w = pop
            .warnings
            .iter()
            .find(|w| w.starts_with("W_CMT_DEFAULTED"))
            .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
        let classified = crate::types::classify_warning(w);
        assert_eq!(
            classified.severity,
            crate::types::WarningSeverity::Warning,
            "message was: {w}"
        );
        assert_eq!(
            classified.category.as_str(),
            "data_quality",
            "the emitted message must reach the DataQuality arm, not an earlier \
             prose arm or the `general` fallback; message was: {w}"
        );
    }
}

/// Review round 2, finding 4: the observation count's *move* was untested.
///
/// The count used to sit before the endpoint branch behind
/// `!skip_missing_dv || routing.tte.contains(&cmt)`; it now sits after all four
/// arms, so "reaching this line" *is* "the row became an observation". The only
/// behavioural difference between the two placements is a TTE row that the
/// `TENTRY > TIME` guard drops as malformed — the old guard's `routing.tte`
/// disjunct deliberately kept such a row, the new placement cannot. No fixture had
/// a defaulted CMT on one, so the mutation that puts the count back survived the
/// whole suite while this PR's own mutation table claimed it killed five tests.
///
/// The straddle is the point: the same two rows, differing only in `TENTRY`, must
/// give different counts. Either half alone passes for an implementation that
/// counts every `EVID=0` row.
#[cfg(feature = "survival")]
#[test]
fn a_tte_row_dropped_for_entry_after_event_is_not_counted_as_an_observation() {
    use std::collections::HashSet;
    let tte: HashSet<usize> = [1].into_iter().collect();
    let routing = ObsRouting::tte_and_discrete(&tte, &HashSet::new());

    // `x` is unparseable, so every row defaults to compartment 1 *and* is reported.
    let read = |tentry_first: &str| -> String {
        let csv = format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV,TENTRY\n\
             1,20,1,0,.,x,0,{tentry_first}\n\
             1,30,0,0,.,x,0,0\n"
        );
        let f = write_csv(&csv);
        let (pop, _) = read_nonmem_csv_routed(f.path(), None, None, &[], None, None, &routing, &[])
            .expect("fixture reads");
        pop.warnings
            .iter()
            .find(|w| w.starts_with("W_CMT_DEFAULTED"))
            .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings))
            .clone()
    };

    // Control: both rows are well-formed TTE observations, so both count.
    let both_valid = read("0");
    assert!(
        both_valid.contains("0 dose row(s) and 2 observation row(s)"),
        "control: two well-formed TTE rows are two observations: {both_valid}"
    );

    // TENTRY=50 > TIME=20: the reader warns about the row and `continue`s before it
    // becomes an `obs_records` entry, so it is not a compartment the fit ever used.
    let one_dropped = read("50");
    assert!(
        one_dropped.contains("0 dose row(s) and 1 observation row(s)"),
        "a TENTRY > TIME row is dropped, so it is not counted: {one_dropped}"
    );
}

/// The other half of the same move, on the Gaussian arm: a missing-DV row skipped
/// under #258 routes nowhere and is already reported by `W_MISSING_DV`, so counting
/// it here would report one row as two findings.
///
/// Separate from the TTE case because the two take different `continue`s — this one
/// fires before the endpoint branch, that one inside it — so a placement that fixes
/// only one reddens only one.
#[test]
fn a_missing_dv_row_skipped_by_the_reader_is_not_counted_as_an_observation() {
    let csv = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
               1,0,.,1,100,x,1\n\
               1,1,.,0,.,x,0\n\
               1,2,5.0,0,.,x,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();

    assert!(
        pop.warnings.iter().any(|w| w.starts_with("W_MISSING_DV")),
        "control: the skipped row is reported by its own code: {:?}",
        pop.warnings
    );
    let w = pop
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings));
    assert!(
        w.contains("1 dose row(s) and 1 observation row(s)"),
        "the missing-DV row is skipped, so only the scored row counts: {w}"
    );
}

/// Review round 2, finding 5: the ellipsis compared **rows** against **spellings**.
///
/// `n_unparseable_cell > examples.len()` is true whenever one bad spelling repeats,
/// so five rows all saying `x` printed `("x", …)` — implying spellings it had not
/// withheld. The gate is now the cap itself, and these two halves differ in exactly
/// the variable the old predicate could not see: the number of *rows* is 5 in both,
/// which is why the earlier straddle (5 spellings vs 3) could not separate them.
#[test]
fn the_ellipsis_marks_a_withheld_spelling_not_a_repeated_one() {
    let read = |cells: [&str; 5]| -> String {
        let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV\n");
        for (i, c) in cells.iter().enumerate() {
            csv.push_str(&format!("1,{},{}.0,0,.,{},0\n", i + 1, i + 1, c));
        }
        let f = write_csv(&csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        pop.warnings
            .iter()
            .find(|w| w.starts_with("W_CMT_DEFAULTED"))
            .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", pop.warnings))
            .clone()
    };

    // Five rows, one spelling. Nothing was withheld, so nothing may be implied.
    let one_spelling = read(["x"; 5]);
    assert!(
        one_spelling.contains("(\"x\")"),
        "one spelling is listed in full: {one_spelling}"
    );
    assert!(
        !one_spelling.contains('…'),
        "five rows of one spelling withhold nothing, so no ellipsis: {one_spelling}"
    );

    // Five rows, five spellings, cap of three. Two were withheld.
    let five_spellings = read(["a", "b", "c", "d", "e"]);
    assert!(
        five_spellings.contains("(\"a\", \"b\", \"c\", …)"),
        "a capped list says so: {five_spellings}"
    );
}

// ── #1496: a float-formatted whole number in an integer column ──────────────
// pandas and R float-format a whole integer column once any cell in it is blank,
// and ferx's own sdtab writes `CENS` as `1.000000`. `L2` (#830) and `CMT` (#1009)
// were each taught that alone; seven other integer sites went on reading `"1.0"`
// as 0 — `ADDL` and `MDV` with no warning at all. They now share one
// classification, `parse_whole_number_cell`; each caller keeps its own range and
// its own fallback for a cell that is not a whole number.

/// A1. The cell table, through the shared classification and every cell-level
/// reader. Each row is a spelling an exporter or a hand edit produces.
#[test]
fn integer_columns_read_a_float_formatted_whole_number_as_that_number() {
    use WholeCell::{Missing, NotWhole, Value};
    // (cell, class, CENS, EVID, OCC, MDV/ADDL/filter-SS as usize, FREMTYPE as u16)
    #[allow(clippy::type_complexity)]
    let table: &[(
        &str,
        WholeCell,
        i8,
        u32,
        Option<u32>,
        Option<usize>,
        Option<u16>,
    )] = &[
        // Integer spellings: the old parse accepted these, and they read as before.
        ("1", Value(1.0), 1, 1, Some(1), Some(1), Some(1)),
        ("+1", Value(1.0), 1, 1, Some(1), Some(1), Some(1)),
        // The defect: each of these read as 0 / None at every site before #1496.
        ("1.0", Value(1.0), 1, 1, Some(1), Some(1), Some(1)),
        ("1e0", Value(1.0), 1, 1, Some(1), Some(1), Some(1)),
        ("0.0", Value(0.0), 0, 0, Some(0), Some(0), Some(0)),
        // A negative whole number is a value only for the signed CENS; every
        // unsigned column keeps its old fallback, `-0.0` included.
        ("-1.0", Value(-1.0), -1, 0, None, None, None),
        ("-0.0", Value(-0.0), 0, 0, None, None, None),
        // Outside `i8`, CENS saturates and keeps its sign — a cast through `i64`
        // wraps 200 to -56 and flips the tail. The unsigned columns read 200.
        (
            "200",
            Value(200.0),
            127,
            200,
            Some(200),
            Some(200),
            Some(200),
        ),
        ("-200", Value(-200.0), -128, 0, None, None, None),
        // Not a whole number: the old fallback at every site, untouched here.
        ("1.5", NotWhole, 0, 0, None, None, None),
        ("abc", NotWhole, 0, 0, None, None, None),
        ("inf", NotWhole, 0, 0, None, None, None),
        // The NONMEM missing spellings.
        ("", Missing, 0, 0, None, None, None),
        (".", Missing, 0, 0, None, None, None),
        ("NA", Missing, 0, 0, None, None, None),
        ("NaN", Missing, 0, 0, None, None, None),
    ];
    for &(cell, class, cens, evid, occ, count, fremtype) in table {
        assert_eq!(parse_whole_number_cell(cell), class, "class of {cell:?}");
        assert_eq!(parse_cens(cell), cens, "CENS {cell:?}");
        assert_eq!(parse_evid(cell), evid, "EVID {cell:?}");
        assert_eq!(parse_occ(cell), occ, "OCC {cell:?}");
        assert_eq!(parse_unsigned_cell::<usize>(cell), count, "usize {cell:?}");
        assert_eq!(parse_unsigned_cell::<u16>(cell), fremtype, "u16 {cell:?}");
    }
}

/// A1, the range edges of the unsigned read: one past a type's end is the
/// fallback, never a wrap or a saturation, and an integer literal never detours
/// through `f64`.
#[test]
fn unsigned_integer_cells_keep_their_type_range_and_exact_literals() {
    assert_eq!(parse_unsigned_cell::<u16>("65535.0"), Some(u16::MAX));
    assert_eq!(parse_unsigned_cell::<u16>("65536.0"), None);
    assert_eq!(parse_unsigned_cell::<u32>("4294967295.0"), Some(u32::MAX));
    assert_eq!(parse_unsigned_cell::<u32>("4294967296.0"), None);
    // 2^64 is the first `f64` past `u64`, and `f as u64` would saturate it to
    // `u64::MAX`.
    assert_eq!(parse_unsigned_cell::<u64>("18446744073709551616"), None);
    // Read by the integer parse, exactly. Through `f64` this rounds to
    // 9007199254740992.
    assert_eq!(
        parse_unsigned_cell::<u64>("9007199254740993"),
        Some(9_007_199_254_740_993)
    );
}

/// A2. The `CENS` observation site. A pandas-shaped export: the dose row's cell is
/// blank, so the exporter float-formats the whole column. It must read exactly as
/// its integer twin — both tails — and warn about nothing.
#[test]
fn a_float_formatted_cens_column_reads_like_its_integer_twin() {
    let read = |minus: &str, zero: &str, plus: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,MDV,AMT,CMT,CENS\n\
             1,0,.,1,1,100,1,\n\
             1,1,50.0,0,0,.,1,{minus}\n\
             1,2,7.0,0,0,.,1,{zero}\n\
             1,3,2.0,0,0,.,1,{plus}\n"
        ));
        read_nonmem_csv(f.path(), None, None).unwrap()
    };
    let integer = read("-1", "0", "1");
    let float = read("-1.0", "0.0", "1.0");
    // The integer leg is not trivial: both tails and a quantified row.
    assert_eq!(integer.subjects[0].cens, vec![-1, 0, 1]);
    assert_eq!(
        float.subjects[0].cens, integer.subjects[0].cens,
        "`-1.0` / `0.0` / `1.0` are the flags -1 / 0 / 1"
    );
    for (leg, pop) in [("integer", &integer), ("float", &float)] {
        assert!(
            pop.warnings.is_empty(),
            "{leg}: a well-formed dataset warns about nothing, got {:?}",
            pop.warnings
        );
    }
}

/// A3. The `CENS` read inside `[data_selection]`. A rule on `CENS` must remove the
/// rows the likelihood would score as censored, whichever way the flag is spelled.
/// Paired with A2 so each of the two sites is pinned by its own test: taking the
/// filter context off the shared resolver leaves A2 green and kills this one.
#[test]
fn a_cens_rule_in_data_selection_removes_the_same_rows_on_both_spellings() {
    let read = |one: &str, zero: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,MDV,AMT,CMT,CENS\n\
             1,0,.,1,1,100,1,\n\
             1,1,7.0,0,0,.,1,{zero}\n\
             1,2,2.0,0,0,.,1,{one}\n\
             1,3,2.0,0,0,.,1,{one}\n"
        ));
        let filter = SelectionFilter::from_opts(&["CENS == 1".to_string()], &[], &[])
            .unwrap_or_else(|e| panic!("filter: {e}"));
        read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap()
    };
    let integer = read("1", "0");
    let float = read("1.0", "0.0");
    // The integer leg removes something: two of its three observations.
    assert_eq!(integer.subjects[0].observations, vec![7.0]);
    assert_eq!(
        float.subjects[0].observations, integer.subjects[0].observations,
        "`ignore = CENS == 1` must remove the `1.0` rows too"
    );
}

/// A4. `EVID`. A float-formatted `1.0` is a dose. Before #1496 it read as 0, so
/// every dose became an unscored `MDV=1` observation and the fit ran without drug.
#[test]
fn a_float_formatted_evid_column_reads_like_its_integer_twin() {
    let read = |dose: &str, obs: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
             1,0,.,{dose},100,1,1\n\
             1,1,5.0,{obs},.,1,0\n\
             1,12,.,{dose},100,1,1\n\
             1,13,4.0,{obs},.,1,0\n"
        ));
        read_nonmem_csv(f.path(), None, None).unwrap()
    };
    let integer = read("1", "0");
    let float = read("1.0", "0.0");
    let times = |p: &Population| {
        p.subjects[0]
            .doses
            .iter()
            .map(|d| d.time)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        times(&integer),
        vec![0.0, 12.0],
        "the integer leg has two doses"
    );
    assert_eq!(times(&float), times(&integer), "`1.0` is EVID 1, a dose");
    assert_eq!(
        float.subjects[0].observations,
        integer.subjects[0].observations
    );
    assert!(
        float.warnings.is_empty(),
        "no dose may be reported as not dosed, got {:?}",
        float.warnings
    );
}

/// A5. `MDV`. The fixture needs an *observation* row carrying `MDV=1` and a real
/// `DV`: on a dose row the flag decides nothing, which is why float-formatting the
/// stock warfarin file's whole `MDV` column changes no number at all. The straddle
/// is asserted, so the row cannot quietly stop being the one the flag excludes.
#[test]
fn a_float_formatted_mdv_flag_excludes_the_row_its_integer_twin_excludes() {
    let read = |one: &str, zero: &str, flagged: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
             1,0,.,1,100,1,{one}\n\
             1,1,5.0,0,.,1,{zero}\n\
             1,2,4.0,0,.,1,{flagged}\n\
             1,3,3.0,0,.,1,{zero}\n"
        ));
        read_nonmem_csv(f.path(), None, None).unwrap()
    };
    let integer = read("1", "0", "1");
    let float = read("1.0", "0.0", "1.0");
    let unflagged = read("1", "0", "0");
    // The straddle: the flag on that row is what excludes it.
    assert_eq!(unflagged.subjects[0].observations, vec![5.0, 4.0, 3.0]);
    assert_eq!(integer.subjects[0].observations, vec![5.0, 3.0]);
    assert_eq!(
        float.subjects[0].observations, integer.subjects[0].observations,
        "`MDV=1.0` must exclude the row as `MDV=1` does"
    );
}

/// A6. `ADDL`. `ADDL=2.0` is two additional doses. Before #1496 it read as 0 and
/// the train collapsed to its first dose, with no warning.
#[test]
fn a_float_formatted_addl_expands_like_its_integer_twin() {
    let read = |addl: &str, zero: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV,II,ADDL\n\
             1,0,.,1,100,1,1,24,{addl}\n\
             1,1,5.0,0,.,1,0,0,{zero}\n\
             1,50,4.0,0,.,1,0,0,{zero}\n"
        ));
        read_nonmem_csv(f.path(), None, None).unwrap()
    };
    let integer = read("2", "0");
    let float = read("2.0", "0.0");
    let times = |p: &Population| {
        p.subjects[0]
            .doses
            .iter()
            .map(|d| d.time)
            .collect::<Vec<_>>()
    };
    // More doses than dose rows: the expansion is live on the integer leg.
    assert_eq!(times(&integer), vec![0.0, 24.0, 48.0]);
    assert_eq!(
        times(&float),
        times(&integer),
        "`ADDL=2.0` is two additional doses"
    );
}

/// A7. The occasion column. `1.0` / `2.0` are occasions 1 and 2, not two rows of
/// occasion 0 reported as `W_IOV_OCC_MISSING`.
#[test]
fn a_float_formatted_occasion_column_reads_like_its_integer_twin() {
    let read = |first: &str, second: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV,OCC\n\
             1,0,.,1,100,1,1,{first}\n\
             1,1,5.0,0,.,1,0,{first}\n\
             1,7,.,1,100,1,1,{second}\n\
             1,8,4.0,0,.,1,0,{second}\n"
        ));
        read_nonmem_csv(f.path(), None, Some("OCC")).unwrap()
    };
    let integer = read("1", "2");
    let float = read("1.0", "2.0");
    assert_eq!(integer.subjects[0].occasions, vec![1, 2]);
    assert_eq!(float.subjects[0].occasions, integer.subjects[0].occasions);
    assert_eq!(
        float.subjects[0].dose_occasions,
        integer.subjects[0].dose_occasions
    );
    assert!(
        !float
            .warnings
            .iter()
            .any(|w| w.contains("W_IOV_OCC_MISSING")),
        "a readable occasion is not a missing one, got {:?}",
        float.warnings
    );
}

/// A8. `SS` inside `[data_selection]`. On the dose row itself `SS=1.0` already read
/// as steady state (it goes through `validate_ss` as a float); the filter context
/// read it as `0`, so `ignore = SS == 1` removed nothing.
#[test]
fn an_ss_rule_in_data_selection_removes_the_same_doses_on_both_spellings() {
    let read = |one: &str, zero: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV,II,SS\n\
             1,0,.,1,100,1,1,24,{one}\n\
             1,1,5.0,0,.,1,0,0,{zero}\n\
             1,24,.,1,100,1,1,0,{zero}\n\
             1,25,4.0,0,.,1,0,0,{zero}\n"
        ));
        let filter = SelectionFilter::from_opts(&["SS == 1".to_string()], &[], &[])
            .unwrap_or_else(|e| panic!("filter: {e}"));
        read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap()
    };
    let integer = read("1", "0");
    let float = read("1.0", "0.0");
    let times = |p: &Population| {
        p.subjects[0]
            .doses
            .iter()
            .map(|d| d.time)
            .collect::<Vec<_>>()
    };
    // The integer leg removes the steady-state dose and keeps the other.
    assert_eq!(times(&integer), vec![24.0]);
    assert_eq!(
        times(&float),
        times(&integer),
        "`ignore = SS == 1` must remove `SS=1.0`"
    );
}

/// `FREMTYPE`, the last integer site: `1.0` / `2.0` are FREM covariate types 1
/// and 2, not two rows of type 0 (an ordinary observation).
#[test]
fn a_float_formatted_fremtype_column_reads_like_its_integer_twin() {
    let read = |pk: &str, cov1: &str, cov2: &str| {
        let f = write_csv(&format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV,FREMTYPE\n\
             1,0,.,1,100,1,1,{pk}\n\
             1,1,5.0,0,.,1,0,{pk}\n\
             1,1,70.0,0,.,1,0,{cov1}\n\
             1,1,30.0,0,.,1,0,{cov2}\n"
        ));
        read_nonmem_csv(f.path(), None, None).unwrap()
    };
    let integer = read("0", "1", "2");
    let float = read("0.0", "1.0", "2.0");
    assert_eq!(integer.subjects[0].fremtype, vec![0, 1, 2]);
    assert_eq!(float.subjects[0].fremtype, integer.subjects[0].fremtype);
}
