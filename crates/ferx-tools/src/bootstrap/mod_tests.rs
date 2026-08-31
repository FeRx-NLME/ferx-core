use super::*;
use ferx_core::Subject;
use nalgebra::DMatrix;

/// A two-theta, two-eta, one-sigma template with a correlated Omega block.
///
/// The block matters: it is what makes `free_mask` non-trivial, so a
/// flatten/rebuild that quietly dropped the off-diagonal would show up.
fn template(diagonal: bool) -> ModelParameters {
    let mut m = DMatrix::from_element(2, 2, 0.0);
    m[(0, 0)] = 0.09;
    m[(1, 1)] = 0.16;
    if !diagonal {
        m[(0, 1)] = 0.02;
        m[(1, 0)] = 0.02;
    }
    let free = DMatrix::from_fn(2, 2, |i, j| diagonal == (i == j) || i == j);
    ModelParameters {
        theta: vec![1.0, 2.0],
        theta_names: vec!["CL".to_string(), "V".to_string()],
        theta_fixed: vec![false, false],
        theta_lower: vec![0.0, 0.0],
        theta_upper: vec![100.0, 100.0],
        omega: OmegaMatrix::from_matrix_with_mask(
            m,
            vec!["ETA_CL".to_string(), "ETA_V".to_string()],
            diagonal,
            free,
        ),
        omega_fixed: vec![false, false],
        omega_iov: None,
        sigma: SigmaVector {
            values: vec![0.04],
            names: vec!["PROP".to_string()],
        },
        sigma_fixed: vec![false],
        kappa_fixed: Vec::new(),
        mixture: None,
    }
}

#[test]
fn parameter_names_cover_theta_free_omega_and_sigma() {
    let names = parameter_names(&template(false));
    assert_eq!(
        names,
        vec![
            "CL",
            "V",
            "OMEGA(ETA_CL,ETA_CL)",
            "OMEGA(ETA_V,ETA_CL)",
            "OMEGA(ETA_V,ETA_V)",
            "PROP",
        ]
    );
}

#[test]
fn a_diagonal_omega_contributes_only_its_diagonal() {
    let names = parameter_names(&template(true));
    assert_eq!(
        names,
        vec![
            "CL",
            "V",
            "OMEGA(ETA_CL,ETA_CL)",
            "OMEGA(ETA_V,ETA_V)",
            "PROP",
        ]
    );
}

#[test]
fn flat_estimates_round_trip_through_model_parameters() {
    // `--update-inits` and `--dofv` both go through the flat vector rather than
    // through a `FitResult`, so that they work identically from a stored
    // `raw_results.csv`. That only holds if the round trip is exact.
    let t = template(false);
    let flat = vec![3.0, 4.0, 0.25, 0.05, 0.36, 0.01];
    let params = params_from_estimates(&t, &flat);

    assert_eq!(params.theta, vec![3.0, 4.0]);
    assert_eq!(params.omega.matrix[(0, 0)], 0.25);
    assert_eq!(params.omega.matrix[(1, 0)], 0.05);
    // Symmetry is restored, not just the lower triangle written.
    assert_eq!(params.omega.matrix[(0, 1)], 0.05);
    assert_eq!(params.omega.matrix[(1, 1)], 0.36);
    assert_eq!(params.sigma.values, vec![0.01]);

    // Structure is the template's, not the flat vector's.
    assert_eq!(params.theta_names, t.theta_names);
    assert_eq!(params.theta_lower, t.theta_lower);
    assert_eq!(params.theta_upper, t.theta_upper);
    assert_eq!(params.omega.eta_names, t.omega.eta_names);
    assert_eq!(params.omega.diagonal, t.omega.diagonal);
    assert_eq!(params.sigma.names, t.sigma.names);

    // And re-flattening reproduces the vector we started from.
    let names = parameter_names(&t);
    assert_eq!(names.len(), flat.len());
    let mut back = params.theta.clone();
    for i in 0..params.omega.dim() {
        for j in 0..=i {
            if t.omega.free_mask[(i, j)] {
                back.push(params.omega.matrix[(i, j)]);
            }
        }
    }
    back.extend(params.sigma.values.iter().copied());
    assert_eq!(back, flat);
}

#[test]
fn structural_zeros_are_not_overwritten_by_the_round_trip() {
    let t = template(true);
    let flat = vec![3.0, 4.0, 0.25, 0.36, 0.01];
    let params = params_from_estimates(&t, &flat);
    // A diagonal Omega has no off-diagonal parameter; the round trip must not
    // invent one by consuming the next slot.
    assert_eq!(params.omega.matrix[(0, 1)], 0.0);
    assert_eq!(params.omega.matrix[(1, 0)], 0.0);
    assert_eq!(params.sigma.values, vec![0.01]);
}

#[test]
fn estimated_parameter_count_skips_fixed_entries() {
    let mut t = template(true);
    assert_eq!(count_estimated(&t), 2 + 2 + 1);
    t.theta_fixed[1] = true;
    t.sigma_fixed[0] = true;
    assert_eq!(count_estimated(&t), 1 + 2);
}

// ── the ID guard ────────────────────────────────────────────────────────────

#[test]
fn a_model_using_id_as_a_covariate_is_refused() {
    // Resampling renames duplicate copies of a subject, so a model reading `ID`
    // would silently see different values than in the base fit. PsN documents
    // the same hazard and leaves it to the user; refuse instead.
    let err = reject_id_dependent(&["WT".to_string(), "ID".to_string()]).unwrap_err();
    assert!(err.contains("`ID` as a covariate"), "{err}");
    // Case-insensitive: the DSL does not force a spelling.
    assert!(reject_id_dependent(&["id".to_string()]).is_err());
    // A covariate that merely contains "id" is fine.
    assert!(reject_id_dependent(&["WT".to_string(), "MIDAZ".to_string()]).is_ok());
    assert!(reject_id_dependent(&[]).is_ok());
}

// ── strata read from the dataset ────────────────────────────────────────────

fn population(ids: &[&str]) -> Population {
    Population {
        subjects: ids
            .iter()
            .map(|id| Subject {
                id: (*id).to_string(),
                ..Subject::default()
            })
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn write_csv(body: &str) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp file");
    write!(f, "{body}").expect("write");
    f.flush().expect("flush");
    f
}

#[test]
fn strata_come_from_the_dataset_not_the_model_covariates() {
    // The stratification column is usually a study or group id the model never
    // declares, so it is read straight from the CSV.
    let csv = write_csv("ID,TIME,DV,STUD\n1,0,.,1001\n1,1,5,1001\n2,0,.,1002\n3,0,.,1001\n");
    let pop = population(&["1", "2", "3"]);
    let strata = strata_from_csv(csv.path().to_str().unwrap(), &[], &pop, "STUD").unwrap();
    assert_eq!(strata.groups["1001"], vec![0, 2]);
    assert_eq!(strata.groups["1002"], vec![1]);
}

#[test]
fn a_stratification_column_that_varies_within_a_subject_is_refused() {
    // PsN: "the algorithm requires that an individual can unambiguously be
    // categorized according to the stratification variable".
    let csv = write_csv("ID,DV,STUD\n1,5,1001\n1,6,1002\n2,7,1002\n");
    let pop = population(&["1", "2"]);
    let err = strata_from_csv(csv.path().to_str().unwrap(), &[], &pop, "STUD").unwrap_err();
    assert!(err.contains("subject `1`"), "{err}");
    assert!(err.contains("more than one value"), "{err}");
}

#[test]
fn a_missing_stratification_column_names_the_columns_that_are_there() {
    let csv = write_csv("ID,DV,WT\n1,5,70\n");
    let pop = population(&["1"]);
    let err = strata_from_csv(csv.path().to_str().unwrap(), &[], &pop, "STUD").unwrap_err();
    assert!(err.contains("STUD"), "{err}");
    assert!(err.contains("WT"), "{err}");
}

#[test]
fn a_remapped_id_column_is_honoured() {
    // `[data] ID = SUBJID` remaps the role; the strata reader must follow it
    // rather than insisting on a literal `ID` header.
    let csv = write_csv("SUBJID,DV,STUD\nA,5,1001\nB,6,1002\n");
    let pop = population(&["A", "B"]);
    let map = vec![("id".to_string(), "SUBJID".to_string())];
    let strata = strata_from_csv(csv.path().to_str().unwrap(), &map, &pop, "STUD").unwrap();
    assert_eq!(strata.groups["1001"], vec![0]);
    assert_eq!(strata.groups["1002"], vec![1]);
}

#[test]
fn a_subject_absent_from_the_dataset_is_reported() {
    let csv = write_csv("ID,DV,STUD\n1,5,1001\n");
    let pop = population(&["1", "9"]);
    let err = strata_from_csv(csv.path().to_str().unwrap(), &[], &pop, "STUD").unwrap_err();
    assert!(err.contains("`9`"), "{err}");
}

// ── replicate fit options ───────────────────────────────────────────────────

#[test]
fn replicate_fits_are_quiet_single_threaded_and_checkpoint_free() {
    // Each of these is load-bearing rather than tidy: 200 verbose fits are
    // unreadable, nested Rayon pools oversubscribe the machine, and 200 fits
    // sharing one `{model}.tmp` checkpoint would restore each other's state.
    let base = FitOptions {
        verbose: true,
        threads: Some(8),
        checkpoint: true,
        checkpoint_path: Some("model.tmp".to_string()),
        sir: true,
        run_covariance_step: true,
        ..FitOptions::default()
    };

    let o = replicate_options(&base, false);
    assert!(!o.verbose);
    assert_eq!(o.threads, Some(1));
    assert!(!o.checkpoint);
    assert_eq!(o.checkpoint_path, None);
    assert!(!o.sir);
    assert!(!o.run_covariance_step);

    // `--keep-covariance` is the one that turns the step back on.
    assert!(replicate_options(&base, true).run_covariance_step);
}

// ── --summarize ─────────────────────────────────────────────────────────────

fn stored_run(dir: &std::path::Path) {
    let mut replicates: Vec<ReplicateResult> = (1..=4)
        .map(|i| ReplicateResult {
            index: i,
            estimates: vec![10.0],
            standard_errors: None,
            ofv: 100.0,
            // Two of the four terminated.
            converged: i <= 2,
            estimate_near_boundary: false,
            covariance_step_successful: true,
            covariance_step_warnings: false,
            seconds: 0.1,
            error: None,
            delta_ofv: None,
        })
        .collect();
    replicates[2].estimates = vec![20.0];
    replicates[3].estimates = vec![20.0];

    let options = BootstrapOptions::default();
    let names = vec!["CL".to_string()];
    let summary = summary::summarize(&names, None, &replicates, &options);
    let result = BootstrapResult {
        parameter_names: names,
        original: None,
        replicates,
        draws: Vec::new(),
        subject_ids: Vec::new(),
        summary,
        n_estimated_parameters: 7,
    };
    output::write_all(dir, &result, &options).expect("write");
}

#[test]
fn resummarize_reapplies_the_filters_without_refitting() {
    // The PsN recovery case "if too many samples are filtered out": the
    // estimates and their diagnostics are already on disk, so relaxing a filter
    // is a re-read, not a re-run.
    let dir = tempfile::tempdir().expect("temp dir");
    stored_run(dir.path());

    let tight = resummarize(dir.path(), &BootstrapOptions::default()).expect("summarize");
    assert_eq!(tight.n_included, 2);
    assert!((tight.parameters[0].mean - 10.0).abs() < 1e-9);

    let relaxed = BootstrapOptions {
        skip_minimization_terminated: false,
        ..BootstrapOptions::default()
    };
    let loose = resummarize(dir.path(), &relaxed).expect("summarize");
    assert_eq!(loose.n_included, 4);
    assert!((loose.parameters[0].mean - 15.0).abs() < 1e-9);
}

#[test]
fn resummarize_carries_forward_the_model_level_diagnostics() {
    // The chi-square degrees of freedom describe the model, not the filtering,
    // and are not recoverable from raw_results.csv — so they must survive a
    // re-summary rather than being rewritten as 0.
    let dir = tempfile::tempdir().expect("temp dir");
    stored_run(dir.path());
    resummarize(dir.path(), &BootstrapOptions::default()).expect("summarize");
    assert_eq!(
        output::read_diagnostic(
            &dir.path().join("bootstrap_diagnostics.csv"),
            "chi_square_df"
        ),
        Some(7.0)
    );
}

#[test]
fn resummarize_needs_an_existing_run_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let err = resummarize(dir.path(), &BootstrapOptions::default()).unwrap_err();
    assert!(err.contains("raw_results.csv"), "{err}");
}
