use super::*;
use crate::bootstrap::summary::{BootstrapSummary, ParameterSummary};
use crate::bootstrap::{Replicate, ReplicateResult};

fn replicate(index: usize, estimates: Vec<f64>, delta: Option<f64>) -> ReplicateResult {
    ReplicateResult {
        index,
        estimates,
        standard_errors: None,
        ofv: 100.0 + index as f64,
        converged: index != 2,
        estimate_near_boundary: false,
        covariance_step_successful: true,
        covariance_step_warnings: false,
        seconds: 0.5,
        error: None,
        delta_ofv: delta,
    }
}

fn result() -> BootstrapResult {
    BootstrapResult {
        parameter_names: vec!["CL".to_string(), "V".to_string()],
        original: Some(replicate(0, vec![1.0, 10.0], None)),
        replicates: vec![
            replicate(1, vec![1.1, 10.5], Some(0.25)),
            replicate(2, vec![0.9, 9.5], Some(1.5)),
        ],
        draws: vec![
            Replicate {
                index: 1,
                keys: vec![0, 0, 2],
            },
            Replicate {
                index: 2,
                keys: vec![1, 1, 2],
            },
        ],
        subject_ids: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        summary: BootstrapSummary {
            parameters: vec![ParameterSummary {
                name: "CL".to_string(),
                original: Some(1.0),
                mean: 1.0,
                bias: Some(0.0),
                standard_error: 0.1,
                median: 1.0,
                ci_percentile: None,
                ci_standard_error: Some((0.8, 1.2)),
            }],
            n_completed: 2,
            n_included: 1,
            excluded_by: vec![("minimization terminated".to_string(), 1)],
            confidence_level: 95.0,
            diagnostic_means: vec![("minimization_successful".to_string(), 0.5)],
        },
        n_estimated_parameters: 3,
    }
}

fn read(path: &std::path::Path) -> Vec<Vec<String>> {
    csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .expect("readable csv")
        .records()
        .map(|r| r.expect("record").iter().map(str::to_string).collect())
        .collect()
}

#[test]
fn write_all_produces_every_psn_artefact() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = BootstrapOptions {
        dofv: true,
        ..BootstrapOptions::default()
    };
    write_all(dir.path(), &result(), &options).expect("write");

    for name in [
        "raw_results.csv",
        "bootstrap_results.csv",
        "bootstrap_diagnostics.csv",
        "all_individuals1.csv",
        "included_individuals1.csv",
        "included_keys1.csv",
        "sample_keys1.csv",
        "delta_ofv.csv",
    ] {
        assert!(dir.path().join(name).exists(), "{name} was not written");
    }
}

#[test]
fn delta_ofv_is_only_written_when_asked_for() {
    let dir = tempfile::tempdir().expect("temp dir");
    write_all(dir.path(), &result(), &BootstrapOptions::default()).expect("write");
    assert!(!dir.path().join("delta_ofv.csv").exists());
}

#[test]
fn raw_results_puts_the_original_dataset_first() {
    // PsN: "The first row is for the original dataset." Downstream tools (PsN's
    // own `covmat -offset=1`) rely on it.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("raw.csv");
    write_raw_results(&path, &result(), &BootstrapOptions::default()).expect("write");
    let rows = read(&path);

    assert_eq!(rows[0][0], "sample");
    assert_eq!(rows[1][0], "0", "row 1 must be the original dataset");
    assert_eq!(rows[2][0], "1");
    assert_eq!(rows[3][0], "2");

    // Diagnostics travel with the estimates, which is what makes the exclusion
    // filters re-appliable without refitting.
    let header = &rows[0];
    for column in [
        "minimization_successful",
        "estimate_near_boundary",
        "covariance_step_successful",
        "covariance_step_warnings",
        "ofv",
        "CL",
        "V",
    ] {
        assert!(header.contains(&column.to_string()), "missing `{column}`");
    }
    let idx = header
        .iter()
        .position(|h| h == "minimization_successful")
        .unwrap();
    assert_eq!(rows[3][idx], "0", "replicate 2 did not converge");
}

#[test]
fn raw_results_gains_se_columns_only_with_keep_covariance() {
    let dir = tempfile::tempdir().expect("temp dir");
    let plain = dir.path().join("plain.csv");
    write_raw_results(&plain, &result(), &BootstrapOptions::default()).expect("write");
    assert!(!read(&plain)[0].contains(&"se_CL".to_string()));

    let with_cov = dir.path().join("cov.csv");
    let options = BootstrapOptions {
        keep_covariance: true,
        ..BootstrapOptions::default()
    };
    write_raw_results(&with_cov, &result(), &options).expect("write");
    assert!(read(&with_cov)[0].contains(&"se_CL".to_string()));
}

#[test]
fn the_replicate_files_agree_row_for_row() {
    // PsN: "The row order is consistent between the files raw_results,
    // included_individuals, included_keys and sample_keys so that row j ...
    // concerns the same bootstrapped dataset."
    let dir = tempfile::tempdir().expect("temp dir");
    let r = result();

    let individuals = dir.path().join("ind.csv");
    let keys = dir.path().join("keys.csv");
    let sample = dir.path().join("sample.csv");
    write_included_individuals(&individuals, &r).expect("write");
    write_included_keys(&keys, &r).expect("write");
    write_sample_keys(&sample, &r).expect("write");

    // Replicate 1 drew A twice and C once.
    assert_eq!(read(&individuals)[0], vec!["A", "A", "C"]);
    assert_eq!(read(&keys)[0], vec!["1", "1", "3"]);
    // sample_keys has a header row of subject IDs, then one count per subject.
    let sample_rows = read(&sample);
    assert_eq!(sample_rows[0], vec!["A", "B", "C"]);
    assert_eq!(sample_rows[1], vec!["2", "0", "1"]);
    assert_eq!(sample_rows[2], vec!["0", "2", "1"]);

    // Second row of each file is the same replicate.
    assert_eq!(read(&individuals)[1], vec!["B", "B", "C"]);
    assert_eq!(read(&keys)[1], vec!["2", "2", "3"]);
}

#[test]
fn included_individuals_uses_the_original_ids() {
    // The `#2` suffix that keeps duplicate draws independent inside the fit is
    // an implementation detail; writing it here would make the file disagree
    // with all_individuals1.csv.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("ind.csv");
    write_included_individuals(&path, &result()).expect("write");
    for row in read(&path) {
        assert!(row.iter().all(|id| !id.contains('#')), "{row:?}");
    }
}

#[test]
fn results_table_is_one_row_per_parameter() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("results.csv");
    write_bootstrap_results(&path, &result()).expect("write");
    let rows = read(&path);
    assert_eq!(rows[0][0], "parameter");
    assert!(rows[0].contains(&"percentile.2.5".to_string()));
    assert!(rows[0].contains(&"percentile.97.5".to_string()));
    assert_eq!(rows[1][0], "CL");
    // A missing percentile CI is an empty cell, not a NaN or a zero.
    let lo = rows[0].iter().position(|h| h == "percentile.2.5").unwrap();
    assert_eq!(rows[1][lo], "");
}

#[test]
fn diagnostics_report_counts_exclusions_and_means() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("diag.csv");
    write_diagnostics(&path, &result()).expect("write");
    let rows = read(&path);
    let find = |k: &str| {
        rows.iter()
            .find(|r| r[0] == k)
            .unwrap_or_else(|| panic!("no row `{k}`"))[1]
            .clone()
    };
    assert_eq!(find("samples_completed").parse::<f64>().unwrap(), 2.0);
    assert_eq!(find("samples_included").parse::<f64>().unwrap(), 1.0);
    assert_eq!(find("chi_square_df").parse::<f64>().unwrap(), 3.0);
    assert_eq!(
        find("excluded: minimization terminated")
            .parse::<f64>()
            .unwrap(),
        1.0
    );
    assert_eq!(
        find("mean: minimization_successful")
            .parse::<f64>()
            .unwrap(),
        0.5
    );
}

#[test]
fn delta_ofv_rows_are_sample_and_difference() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("dofv.csv");
    write_delta_ofv(&path, &result()).expect("write");
    let rows = read(&path);
    assert_eq!(rows[0], vec!["sample", "delta_ofv"]);
    assert_eq!(rows[1][0], "1");
    assert!((rows[1][1].parse::<f64>().unwrap() - 0.25).abs() < 1e-9);
}

// ── reading the raw results back ────────────────────────────────────────────

#[test]
fn raw_results_round_trip_preserves_estimates_and_diagnostics() {
    // `--summarize` reads this file back and re-applies the filters, so a lossy
    // round trip would silently change which replicates count.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("raw_results.csv");
    let options = BootstrapOptions {
        dofv: true,
        ..BootstrapOptions::default()
    };
    let written = result();
    write_raw_results(&path, &written, &options).expect("write");

    let (names, original, replicates) = read_raw_results(&path).expect("read");
    assert_eq!(names, written.parameter_names);

    let original = original.expect("the sample=0 row is the original fit");
    assert_eq!(original.index, 0);
    assert!((original.estimates[0] - 1.0).abs() < 1e-9);

    assert_eq!(replicates.len(), 2);
    assert!((replicates[0].estimates[1] - 10.5).abs() < 1e-9);
    assert!(replicates[0].converged);
    assert!(
        !replicates[1].converged,
        "the diagnostic must survive the round trip"
    );
    assert!((replicates[1].delta_ofv.unwrap() - 1.5).abs() < 1e-9);
}

#[test]
fn raw_results_round_trip_finds_the_se_columns() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("raw_results.csv");
    let options = BootstrapOptions {
        keep_covariance: true,
        ..BootstrapOptions::default()
    };
    let mut written = result();
    written.replicates[0].standard_errors = Some(vec![0.1, 0.2]);
    write_raw_results(&path, &written, &options).expect("write");

    let (names, _, replicates) = read_raw_results(&path).expect("read");
    // The parameter block must stop at `se_`, not swallow it.
    assert_eq!(names, vec!["CL", "V"]);
    let se = replicates[0].standard_errors.as_ref().expect("se columns");
    assert!((se[0] - 0.1).abs() < 1e-9);
    assert!((se[1] - 0.2).abs() < 1e-9);
}

#[test]
fn reading_a_file_that_is_not_raw_results_says_so() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("sdtab.csv");
    std::fs::write(&path, "ID,TIME,DV\n1,0,5\n").expect("write");
    let err = read_raw_results(&path).unwrap_err();
    assert!(
        err.contains("does not look like a bootstrap raw_results file"),
        "{err}"
    );
}

#[test]
fn a_failed_replicate_round_trips_as_failed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("raw_results.csv");
    let mut written = result();
    written.replicates[1].error = Some("inner loop diverged".to_string());
    written.replicates[1].estimates = Vec::new();
    write_raw_results(&path, &written, &BootstrapOptions::default()).expect("write");

    let (_, _, replicates) = read_raw_results(&path).expect("read");
    assert_eq!(replicates[1].error.as_deref(), Some("inner loop diverged"));
    assert!(replicates[1].estimates.is_empty());
}

#[test]
fn a_diagnostic_can_be_read_back_by_name() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("bootstrap_diagnostics.csv");
    write_diagnostics(&path, &result()).expect("write");
    assert_eq!(read_diagnostic(&path, "chi_square_df"), Some(3.0));
    assert_eq!(read_diagnostic(&path, "not_a_statistic"), None);
    assert_eq!(read_diagnostic(&dir.path().join("absent.csv"), "x"), None);
}
