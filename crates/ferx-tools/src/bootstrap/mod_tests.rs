use super::*;
use ferx_core::{CancelFlag, Subject};
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

/// [`template`] with a diagonal Omega plus a one-kappa IOV block.
fn template_with_iov() -> ModelParameters {
    let mut t = template(true);
    let iov = DMatrix::from_element(1, 1, 0.01);
    t.omega_iov = Some(OmegaMatrix::from_matrix_with_mask(
        iov,
        vec!["KAPPA_CL".to_string()],
        true,
        DMatrix::from_element(1, 1, true),
    ));
    t.kappa_fixed = vec![false];
    t
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

#[test]
fn the_estimated_count_matches_the_parameter_vector_for_a_block_omega() {
    // Regression for the second review finding: `count_estimated` used to count
    // the `omega_fixed` flags, which are per *eta*, so a free 2x2 block scored 2
    // where the parameter vector has 3. That understates the chi-square degrees
    // of freedom for `--dofv`, which makes the Δofv distribution look better
    // than it is — the one direction a diagnostic must not fail in.
    let t = template(false);
    assert_eq!(parameter_names(&t).len(), 6, "2 theta + 3 omega + 1 sigma");
    assert_eq!(
        count_estimated(&t),
        parameter_names(&t).len(),
        "with nothing fixed, every coordinate is an estimated parameter"
    );

    // Fixing an eta removes its variance *and* its covariances, since the flags
    // are per-eta and a covariance needs both of its etas free.
    let mut fixed = t.clone();
    fixed.omega_fixed[1] = true;
    assert_eq!(count_estimated(&fixed), 6 - 2);
}

#[test]
fn an_iov_block_is_part_of_the_parameter_vector() {
    let t = template_with_iov();
    let names = parameter_names(&t);
    assert_eq!(
        names,
        vec![
            "CL",
            "V",
            "OMEGA(ETA_CL,ETA_CL)",
            "OMEGA(ETA_V,ETA_V)",
            "PROP",
            "OMEGA_IOV(KAPPA_CL,KAPPA_CL)",
        ],
        "the IOV block goes last, so a model without one keeps its old columns"
    );
    assert_eq!(count_estimated(&t), names.len());

    // A fixed kappa drops out of the count but keeps its column.
    let mut fixed = t.clone();
    fixed.kappa_fixed[0] = true;
    assert_eq!(parameter_names(&fixed).len(), names.len());
    assert_eq!(count_estimated(&fixed), names.len() - 1);
}

#[test]
fn the_iov_variance_round_trips() {
    let t = template_with_iov();
    let flat = vec![3.0, 4.0, 0.25, 0.36, 0.01, 0.07];
    let params = params_from_estimates(&t, &flat);
    let iov = params.omega_iov.as_ref().expect("the IOV block survives");
    assert_eq!(iov.matrix[(0, 0)], 0.07);
    assert_eq!(iov.eta_names, vec!["KAPPA_CL"]);
    // And the rest is untouched by the extra coordinate.
    assert_eq!(params.theta, vec![3.0, 4.0]);
    assert_eq!(params.sigma.values, vec![0.01]);
}

#[test]
fn every_view_of_the_flat_vector_has_the_same_length() {
    // The invariant the `Coord` enum exists to hold: names, the estimated count
    // and the rebuild all walk one traversal, so they cannot disagree about how
    // many parameters there are or what order they are in.
    for t in [template(true), template(false), template_with_iov()] {
        let names = parameter_names(&t);
        assert_eq!(count_estimated(&t), names.len());

        // Scaled from the template's own values rather than made up: an
        // arbitrary vector can describe a non-positive-definite Omega, which
        // `OmegaMatrix::from_matrix_with_mask` repairs — correctly — by moving
        // the diagonal, and the round trip would then not be an identity for a
        // reason that has nothing to do with the traversal being tested.
        // Scaling a PD matrix by a positive constant keeps it PD.
        let flat: Vec<f64> = coordinates(&t)
            .into_iter()
            .map(|c| {
                1.5 * match c {
                    Coord::Theta(i) => t.theta[i],
                    Coord::Omega(i, j) => t.omega.matrix[(i, j)],
                    Coord::Sigma(i) => t.sigma.values[i],
                    Coord::OmegaIov(i, j) => t.omega_iov.as_ref().unwrap().matrix[(i, j)],
                }
            })
            .collect();
        let rebuilt = params_from_estimates(&t, &flat);
        assert_eq!(parameter_names(&rebuilt), names);
        // Re-flattening the rebuild returns the vector it was built from.
        let coords = coordinates(&rebuilt);
        assert_eq!(coords.len(), names.len());
        let back: Vec<f64> = coords
            .into_iter()
            .map(|c| match c {
                Coord::Theta(i) => rebuilt.theta[i],
                Coord::Omega(i, j) => rebuilt.omega.matrix[(i, j)],
                Coord::Sigma(i) => rebuilt.sigma.values[i],
                Coord::OmegaIov(i, j) => rebuilt.omega_iov.as_ref().unwrap().matrix[(i, j)],
            })
            .collect();
        assert_eq!(back, flat);
    }
}

// ── mixture models ──────────────────────────────────────────────────────────

#[test]
fn a_mixture_model_is_refused() {
    // Not a plumbing gap: a mixture's classes are identified only up to
    // relabelling, so averaging estimates across replicates would mix them.
    let plain = template(true);
    assert!(reject_mixture_model(&plain).is_ok());

    let mut mixture = template(true);
    mixture.mixture = Some(ferx_core::MixtureParams {
        omega: vec![plain.omega.clone(), plain.omega.clone()],
        sigma: vec![plain.sigma.clone(), plain.sigma.clone()],
        omega_override_addr: Vec::new(),
        omega_override_fixed: Vec::new(),
        sigma_override_addr: Vec::new(),
        sigma_override_fixed: Vec::new(),
    });
    let err = reject_mixture_model(&mixture).unwrap_err();
    assert!(err.contains("relabelling"), "{err}");
    assert!(err.contains("mixture"), "{err}");
    // #1145 closed this as decided, not deferred, so the message must name the
    // method that does work rather than read as a missing feature.
    assert!(err.contains("will not"), "{err}");
    assert!(err.contains("SIR"), "{err}");
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

    let o = replicate_options(&base, false, &None);
    assert!(!o.verbose);
    assert_eq!(o.threads, Some(1));
    assert!(!o.checkpoint);
    assert_eq!(o.checkpoint_path, None);
    assert!(!o.sir);
    assert!(!o.run_covariance_step);

    // `--keep-covariance` is the one that turns the step back on.
    assert!(replicate_options(&base, true, &None).run_covariance_step);
    // Nothing to cancel with, so nothing is installed: a replicate of an
    // uncancellable run polls no flag.
    assert!(o.cancel.is_none());
}

#[test]
fn the_run_wide_cancel_flag_reaches_every_replicate_fit() {
    // The propagation #1161 depends on: a replicate is an ordinary `fit`, and
    // the only way the flag it polls gets there is this clone.
    let flag = CancelFlag::new();
    let o = replicate_options(&FitOptions::default(), false, &Some(flag.clone()));
    let installed = o.cancel.expect("the replicate options carry the flag");
    assert!(!installed.is_cancelled());
    flag.cancel();
    assert!(
        installed.is_cancelled(),
        "the replicate got a copy of the flag rather than a share of it"
    );
}

#[test]
fn a_flag_on_the_model_s_own_fit_options_is_still_honoured() {
    // The pre-#1161 route: callers set `prepared.parsed.fit_options.cancel` and
    // relied on the replicate options cloning it. That has to keep working, or
    // this change silently stops honouring a flag that used to abort a run.
    let flag = CancelFlag::new();
    let base = FitOptions {
        cancel: Some(flag.clone()),
        ..FitOptions::default()
    };
    let options = BootstrapOptions::default();
    let effective = effective_cancel(&options, &base).expect("inherited from the fit options");
    flag.cancel();
    assert!(effective.is_cancelled());

    // And the run-wide flag wins when both are set, so a caller that passes one
    // to the tool is not quietly overridden by whatever the model file carried.
    let run_wide = CancelFlag::new();
    let effective = effective_cancel(
        &BootstrapOptions {
            cancel: Some(run_wide.clone()),
            ..BootstrapOptions::default()
        },
        &base,
    )
    .expect("the run-wide flag");
    assert!(
        !effective.is_cancelled(),
        "the already-cancelled fit-options flag took precedence over the run's own"
    );
    run_wide.cancel();
    assert!(effective.is_cancelled());
}

#[test]
fn a_cancellation_is_not_a_failure() {
    // What a caller reports with. `Cancelled` has to survive the trip through a
    // `String` error type readably, since that is what the CLI and the R glue
    // still use.
    let cancelled = BootstrapError::Cancelled;
    assert!(cancelled.is_cancelled());
    assert!(!BootstrapError::Failed("boom".to_string()).is_cancelled());

    assert_eq!(
        BootstrapError::Failed("boom".to_string()).to_string(),
        "boom",
        "a failure must not be dressed up on its way through Display"
    );
    assert!(String::from(cancelled).contains("cancel"));

    // `?` on the String-returning helpers inside the run depends on both.
    assert_eq!(
        BootstrapError::from("boom"),
        BootstrapError::Failed("boom".to_string())
    );
    assert_eq!(
        BootstrapError::from("boom".to_string()),
        BootstrapError::Failed("boom".to_string())
    );
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
        n_reused: 0,
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

#[test]
fn resummarize_validates_its_options_instead_of_panicking() {
    // Regression for the fourth review finding. `run_bootstrap` checked the
    // confidence level; `resummarize` went straight to `summarize`, so
    // `--summarize --ci 100` reached `normal_quantile`'s assertion and the
    // process died with 101 instead of reporting a bad argument.
    let dir = tempfile::tempdir().expect("temp dir");
    stored_run(dir.path());

    for level in [100.0, 0.0, -1.0, 137.0] {
        let err = resummarize(
            dir.path(),
            &BootstrapOptions {
                confidence_level: level,
                ..BootstrapOptions::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("--ci"), "level {level}: {err}");
    }

    // Checked before the directory is read, so a bad option is reported even
    // when there is nothing to summarize.
    let empty = tempfile::tempdir().expect("temp dir");
    let err = resummarize(
        empty.path(),
        &BootstrapOptions {
            confidence_level: 100.0,
            ..BootstrapOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.contains("--ci"), "{err}");
}

#[test]
fn the_covstep_filters_are_allowed_when_re_summarizing() {
    // The rule that a covstep filter needs `--keep-covariance` applies to a
    // fresh run, where the step would have to actually run. Under `--summarize`
    // the diagnostics are already in `raw_results.csv`, so filtering on them is
    // the entire point and requires no covariance step now.
    let dir = tempfile::tempdir().expect("temp dir");
    stored_run(dir.path());
    let options = BootstrapOptions {
        skip_covariance_step_terminated: true,
        ..BootstrapOptions::default()
    };
    assert!(options.validate().is_ok());
    assert!(
        options.validate_for_run().is_err(),
        "a fresh run must refuse it"
    );
    assert!(resummarize(dir.path(), &options).is_ok());
}

// ── --resume (#1143) ────────────────────────────────────────────────────────

fn resume_options(dir: &std::path::Path) -> BootstrapOptions {
    BootstrapOptions {
        samples: 4,
        resume: true,
        directory: Some(dir.to_path_buf()),
        ..BootstrapOptions::default()
    }
}

fn resume_manifest(options: &BootstrapOptions) -> RunManifest {
    RunManifest::new(
        options,
        Some("model".to_string()),
        Some("data".to_string()),
        &["CL".to_string()],
    )
}

/// A directory as an interrupted run would have left it: samples `1..=through`
/// recorded, the rest never fitted.
fn interrupted_run(dir: &std::path::Path, through: usize, options: &BootstrapOptions) {
    let one = |index: usize, estimate: f64| ReplicateResult {
        index,
        estimates: vec![estimate],
        standard_errors: None,
        ofv: 100.0 + index as f64,
        converged: true,
        estimate_near_boundary: false,
        covariance_step_successful: false,
        covariance_step_warnings: false,
        seconds: 0.1,
        error: None,
        delta_ofv: None,
    };
    let replicates: Vec<ReplicateResult> = (1..=through).map(|i| one(i, 10.0 + i as f64)).collect();
    let names = vec!["CL".to_string()];
    let summary = summary::summarize(&names, None, &replicates, options);
    let result = BootstrapResult {
        parameter_names: names,
        original: Some(one(0, 9.0)),
        replicates,
        draws: Vec::new(),
        subject_ids: Vec::new(),
        summary,
        n_estimated_parameters: 1,
        n_reused: 0,
    };
    std::fs::create_dir_all(dir).expect("dir");
    output::write_raw_results(&dir.join("raw_results.csv"), &result, options).expect("write");
    resume_manifest(options)
        .write(&journal::manifest_path(dir))
        .expect("write manifest");
}

#[test]
fn resume_reuses_the_completed_samples_and_the_base_fit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = resume_options(dir.path());
    interrupted_run(dir.path(), 2, &options);

    let (original, kept) =
        load_resumable(dir.path(), &resume_manifest(&options), &options).expect("resume loads");
    assert_eq!(
        kept.iter().map(|r| r.index).collect::<Vec<_>>(),
        vec![1, 2],
        "samples 3 and 4 were never fitted and must be refitted"
    );
    // The base fit is reused rather than repeated: `--update-inits` starts every
    // replicate from its estimates, so refitting it would start the new
    // replicates from a different point than the ones already on disk did.
    let original = original.expect("the sample=0 row is reused");
    assert!((original.estimates[0] - 9.0).abs() < 1e-9);
}

#[test]
fn resume_into_a_complete_run_refits_nothing() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = resume_options(dir.path());
    interrupted_run(dir.path(), 4, &options);

    let (_, kept) =
        load_resumable(dir.path(), &resume_manifest(&options), &options).expect("resume loads");
    assert_eq!(kept.len(), 4, "every sample is already on disk");
}

/// A recorded failure is kept by default — PsN's answer, and the right one when
/// the failure is deterministic. `--retry-failed` is the other half of the
/// choice, for a failure that was a transient resource problem instead.
#[test]
fn a_failed_replicate_is_kept_by_default_and_refitted_with_retry_failed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut options = resume_options(dir.path());
    interrupted_run(dir.path(), 3, &options);

    // Mark sample 2 as failed, the way `run_one` records a `fit` error.
    let path = dir.path().join("raw_results.csv");
    let text = std::fs::read_to_string(&path).expect("read");
    let patched: Vec<String> = text
        .lines()
        .map(|l| {
            if l.starts_with("2,") {
                format!("{l}the optimizer diverged")
            } else {
                l.to_string()
            }
        })
        .collect();
    std::fs::write(&path, patched.join("\n") + "\n").expect("write");

    let (_, kept) =
        load_resumable(dir.path(), &resume_manifest(&options), &options).expect("resume loads");
    assert_eq!(
        kept.iter().map(|r| r.index).collect::<Vec<_>>(),
        vec![1, 2, 3],
        "the failure is carried forward, not refitted"
    );
    assert!(kept[1].error.is_some());

    options.retry_failed = true;
    let (_, kept) =
        load_resumable(dir.path(), &resume_manifest(&options), &options).expect("resume loads");
    assert_eq!(
        kept.iter().map(|r| r.index).collect::<Vec<_>>(),
        vec![1, 3],
        "--retry-failed drops the failure so the sample is refitted"
    );
}

/// Resuming into a directory another run wrote would label one model's
/// estimates with another model's parameter names — a corruption nothing
/// downstream could detect, because the file would still parse.
#[test]
fn resuming_a_mismatched_directory_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = resume_options(dir.path());
    interrupted_run(dir.path(), 2, &options);

    let asked_for = RunManifest {
        seed: 999,
        ..resume_manifest(&options)
    };
    let err = load_resumable(dir.path(), &asked_for, &options).unwrap_err();
    assert!(err.contains("--seed"), "{err}");
}

/// The columns of `raw_results.csv` are checked against the manifest as well,
/// so a hand-edited or foreign file is caught even when neither input could be
/// hashed.
#[test]
fn resuming_a_directory_whose_columns_disagree_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = resume_options(dir.path());
    interrupted_run(dir.path(), 2, &options);
    let path = dir.path().join("raw_results.csv");
    let text = std::fs::read_to_string(&path).expect("read");
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    lines[0] = lines[0].replace("CL,error", "CL,V,error");
    // Every row must stay full width, or the tail tolerance would fire first.
    for line in lines.iter_mut().skip(1) {
        let mut fields: Vec<&str> = line.split(',').collect();
        fields.insert(fields.len() - 1, "1.0");
        *line = fields.join(",");
    }
    std::fs::write(&path, lines.join("\n") + "\n").expect("write");

    let err = load_resumable(dir.path(), &resume_manifest(&options), &options).unwrap_err();
    assert!(err.contains("different run"), "{err}");
}

#[test]
fn resuming_a_directory_with_no_run_in_it_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = resume_options(dir.path());
    let err = load_resumable(dir.path(), &resume_manifest(&options), &options).unwrap_err();
    assert!(err.contains("raw_results.csv"), "{err}");

    // A raw_results.csv with no manifest beside it is equally unusable: there is
    // nothing to check the directory against.
    interrupted_run(dir.path(), 1, &options);
    std::fs::remove_file(journal::manifest_path(dir.path())).expect("remove");
    let err = load_resumable(dir.path(), &resume_manifest(&options), &options).unwrap_err();
    assert!(err.contains("bootstrap_run.json"), "{err}");
}

/// Changing where the replicates *start* is the subtle mismatch, and the one
/// that would otherwise pass every other check.
///
/// Interrupt a default run and resume it with `--no-update-inits`: every reused
/// replicate was started from the base fit, every refitted one starts from the
/// model file's estimates. Both halves are valid fits; the file holding them is
/// a bootstrap of neither, and nothing about it looks wrong.
#[test]
fn a_resume_that_moves_the_replicate_starting_point_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut options = resume_options(dir.path());
    interrupted_run(dir.path(), 2, &options);

    options.update_inits = false;
    let err = load_resumable(dir.path(), &resume_manifest(&options), &options).unwrap_err();
    assert!(err.contains("--update-inits"), "{err}");

    // And the inverse: a run started from the model file, resumed asking for
    // the base fit's estimates.
    let other = tempfile::tempdir().expect("temp dir");
    let mut options = resume_options(other.path());
    options.update_inits = false;
    interrupted_run(other.path(), 2, &options);

    options.update_inits = true;
    let err = load_resumable(other.path(), &resume_manifest(&options), &options).unwrap_err();
    assert!(err.contains("--update-inits"), "{err}");
}

/// `--no-run-base-model` moves the starting point too — and changes whether a
/// `sample = 0` row belongs in the file at all — so it is pinned in its own
/// right rather than left to the initialization-mode check.
#[test]
fn a_resume_that_adds_or_drops_the_base_model_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut options = resume_options(dir.path());
    interrupted_run(dir.path(), 2, &options);

    options.run_base_model = false;
    options.update_inits = false;
    let err = load_resumable(dir.path(), &resume_manifest(&options), &options).unwrap_err();
    assert!(err.contains("--run-base-model"), "{err}");
}

/// When the directory really was written without a base model, a stored
/// `sample = 0` row is not carried forward — it would reappear as a row the
/// caller asked not to have. The manifest refuses that transition first, so this
/// covers a library caller that built its own manifest.
#[test]
fn a_run_without_a_base_model_drops_a_stored_base_fit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = BootstrapOptions {
        run_base_model: false,
        update_inits: false,
        ..resume_options(dir.path())
    };
    interrupted_run(dir.path(), 2, &options);

    let (original, kept) =
        load_resumable(dir.path(), &resume_manifest(&options), &options).expect("resume loads");
    assert!(original.is_none());
    assert_eq!(kept.len(), 2);
}

#[test]
fn resume_needs_a_directory_and_retry_failed_needs_resume() {
    let bare = BootstrapOptions {
        resume: true,
        ..BootstrapOptions::default()
    };
    let err = bare.validate_for_run().unwrap_err();
    assert!(err.contains("--directory"), "{err}");

    let dangling = BootstrapOptions {
        retry_failed: true,
        directory: Some(std::path::PathBuf::from("run")),
        ..BootstrapOptions::default()
    };
    let err = dangling.validate_for_run().unwrap_err();
    assert!(err.contains("--resume"), "{err}");
}

// ── progress events ─────────────────────────────────────────────────────────

/// One real run with a sink attached, on the smallest useful model.
///
/// The event contract is a sequence, not a set - a bar that is told its length
/// after it has already been stepped draws wrong - so this asserts the order as
/// well as the counts. Warfarin (10 subjects) at `--samples 1` with `--dofv` is
/// three fits, one of which is an evaluation: enough to see every variant.
#[test]
fn a_run_reports_its_fits_in_order() {
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let prepared = ferx_core::prepare_run(
        repo.join("examples/warfarin.ferx").to_str().unwrap(),
        Some(repo.join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin loads");
    let options = BootstrapOptions {
        samples: 1,
        seed: 5,
        threads: Some(1),
        dofv: true,
        ..BootstrapOptions::default()
    };

    let events = std::sync::Mutex::new(Vec::new());
    let sink = |event| events.lock().expect("not poisoned").push(event);
    run_bootstrap_with_progress(&prepared, &options, Some(&sink)).expect("the run succeeds");

    let events = events.into_inner().expect("not poisoned");
    assert_eq!(
        events,
        vec![
            BootstrapEvent::Started {
                replicates: 1,
                reused: 0,
                base_fit: true,
            },
            BootstrapEvent::BaseFitDone,
            BootstrapEvent::ReplicateDone {
                completed: 1,
                total: 1,
            },
            BootstrapEvent::DeltaOfvDone {
                completed: 1,
                total: 1,
            },
            BootstrapEvent::Finished,
        ],
        "{events:?}"
    );
}

/// A run that fails after it has announced itself still reports `Finished`.
///
/// A sink told that a run started and never told it ended leaves its bar on the
/// terminal, over the error the caller is about to print. `--dofv` without a
/// base fit is the cheapest way in: it passes `validate()` and fails at the
/// reference OFV, after `Started` and after every replicate is in.
#[test]
fn a_failed_run_still_reports_that_it_finished() {
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let prepared = ferx_core::prepare_run(
        repo.join("examples/warfarin.ferx").to_str().unwrap(),
        Some(repo.join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin loads");
    let options = BootstrapOptions {
        samples: 1,
        seed: 5,
        threads: Some(1),
        dofv: true,
        run_base_model: false,
        update_inits: false,
        ..BootstrapOptions::default()
    };

    let events = std::sync::Mutex::new(Vec::new());
    let sink = |event| events.lock().expect("not poisoned").push(event);
    let err = run_bootstrap_with_progress(&prepared, &options, Some(&sink))
        .expect_err("--dofv needs a base fit");
    assert!(err.to_string().contains("--dofv"), "{err}");

    let events = events.into_inner().expect("not poisoned");
    assert_eq!(
        events.first(),
        Some(&BootstrapEvent::Started {
            replicates: 1,
            reused: 0,
            base_fit: false,
        }),
        "{events:?}"
    );
    assert_eq!(events.last(), Some(&BootstrapEvent::Finished), "{events:?}");
    // Exactly one, whichever way the run leaves: the guard emits on the way
    // out, and the success path disarms it by emitting first.
    assert_eq!(
        events
            .iter()
            .filter(|e| **e == BootstrapEvent::Finished)
            .count(),
        1,
        "{events:?}"
    );
}

/// A run with no sink is the same run. Nothing about the draws or the fits may
/// depend on whether a bar was being drawn, so the two results have to agree
/// down to the estimate.
#[test]
fn watching_a_run_does_not_change_it() {
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let prepared = ferx_core::prepare_run(
        repo.join("examples/warfarin.ferx").to_str().unwrap(),
        Some(repo.join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin loads");
    let options = BootstrapOptions {
        samples: 1,
        seed: 7,
        threads: Some(1),
        run_base_model: false,
        update_inits: false,
        ..BootstrapOptions::default()
    };

    let quiet = run_bootstrap(&prepared, &options).expect("the run succeeds");
    let watched = {
        let seen = std::sync::atomic::AtomicUsize::new(0);
        let sink = |_| {
            seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        };
        let result = run_bootstrap_with_progress(&prepared, &options, Some(&sink))
            .expect("the run succeeds");
        // Started, one ReplicateDone, Finished - and no BaseFitDone, because
        // `run_base_model = false` means there was no base fit to report.
        assert_eq!(seen.load(std::sync::atomic::Ordering::Relaxed), 3);
        result
    };

    assert_eq!(
        quiet.replicates[0].estimates,
        watched.replicates[0].estimates
    );
}
