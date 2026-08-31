//! Tier-2 end-to-end checks for the bootstrap (#1140), on a real model and
//! dataset but never to convergence: every fit here is capped at a handful of
//! outer iterations, or is an evaluation (`outer_maxiter = 0`).
//!
//! The interesting one is the **degenerate oracle**: a "resample" that draws
//! every subject exactly once must be bit-for-bit indistinguishable from an
//! ordinary fit on the original population. That is what pins the replicate
//! population against the engine — a dropped covariate map, a lost occasion
//! column or a mishandled dose record would change the objective, and nothing
//! else in the bootstrap would notice, because there is no second copy of the
//! reconstruction to disagree with it.

use std::path::Path;

use ferx_core::{fit, prepare_run, FitOptions, PreparedRun};
use ferx_tools::bootstrap::{
    flatten_estimates, params_from_estimates, resample, run_bootstrap, BootstrapOptions, Replicate,
    SampleSize,
};

const MODEL: &str = "../../examples/warfarin.ferx";
const DATA: &str = "../../data/warfarin.csv";

fn prepared() -> PreparedRun {
    prepare_run(MODEL, Some(DATA)).expect("warfarin model + data load")
}

/// An evaluation, not a fit: `outer_maxiter = 0` is ferx's `MAXEVAL=0`.
fn eval_options(base: &FitOptions) -> FitOptions {
    let mut o = base.clone();
    o.outer_maxiter = 0;
    o.run_covariance_step = false;
    o.verbose = false;
    o.threads = Some(1);
    o.checkpoint = false;
    o
}

#[test]
fn an_identity_resample_reproduces_the_original_fit_exactly() {
    let p = prepared();
    let options = eval_options(&p.parsed.fit_options);

    let direct = fit(&p.parsed.model, &p.population, &p.init_params, &options)
        .expect("evaluation on the original population");

    let identity = Replicate {
        index: 1,
        keys: (0..p.population.subjects.len()).collect(),
    };
    let rebuilt = resample::build_population(&p.population, &identity);
    let via_resample = fit(&p.parsed.model, &rebuilt, &p.init_params, &options)
        .expect("evaluation on the rebuilt population");

    assert_eq!(
        rebuilt.subjects.len(),
        p.population.subjects.len(),
        "the identity replicate changed the subject count"
    );
    // Bit-for-bit. Anything less would let a reconstruction that is merely
    // close pass, and "merely close" is exactly what a lost covariate looks
    // like at this scale.
    assert_eq!(
        via_resample.ofv.to_bits(),
        direct.ofv.to_bits(),
        "identity resample gave OFV {} vs {}",
        via_resample.ofv,
        direct.ofv
    );
    assert_eq!(via_resample.n_obs, direct.n_obs);
    assert_eq!(via_resample.n_subjects, direct.n_subjects);
}

#[test]
fn a_duplicated_subject_contributes_twice() {
    // The other side of the oracle: drawing one subject twice must add its
    // likelihood contribution again, as an independent individual. If the
    // duplicate were silently merged (or dropped as a repeated ID), the
    // observation count would not move.
    let p = prepared();
    let options = eval_options(&p.parsed.fit_options);

    let mut keys: Vec<usize> = (0..p.population.subjects.len()).collect();
    keys.push(0);
    keys.sort_unstable();
    let doubled = resample::build_population(
        &p.population,
        &Replicate {
            index: 1,
            keys: keys.clone(),
        },
    );
    assert_eq!(doubled.subjects.len(), p.population.subjects.len() + 1);

    let base = fit(&p.parsed.model, &p.population, &p.init_params, &options).expect("base eval");
    let with_dup =
        fit(&p.parsed.model, &doubled, &p.init_params, &options).expect("duplicated eval");

    assert_eq!(
        with_dup.n_obs,
        base.n_obs + p.population.subjects[0].observations.len(),
        "the duplicate's observations did not enter the fit"
    );
    assert_eq!(with_dup.n_subjects, base.n_subjects + 1);

    // The two copies are the same data under different labels, so the engine
    // must give them identical individual results. If the duplicate had been
    // merged into one subject with twice the records, or dropped as a repeated
    // ID, this is where it shows.
    assert_eq!(with_dup.subjects[0].id, p.population.subjects[0].id);
    assert_eq!(
        with_dup.subjects[1].id,
        format!("{}#2", p.population.subjects[0].id)
    );
    assert_eq!(
        with_dup.subjects[0].ofv_contribution.to_bits(),
        with_dup.subjects[1].ofv_contribution.to_bits(),
    );
    assert_eq!(with_dup.subjects[0].eta, with_dup.subjects[1].eta);

    // Each additional identical copy must move the objective by the *same*
    // amount. At fixed parameters the individual contributions are independent,
    // so the increments are equal by construction — and comparing increments
    // sidesteps whatever constant terms `ofv_contribution` does or does not
    // carry, which is a convention this test has no business pinning.
    let tripled = resample::build_population(
        &p.population,
        &Replicate {
            index: 2,
            keys: {
                let mut k = keys.clone();
                k.push(0);
                k.sort_unstable();
                k
            },
        },
    );
    let with_two_dups =
        fit(&p.parsed.model, &tripled, &p.init_params, &options).expect("tripled eval");

    let first = with_dup.ofv - base.ofv;
    let second = with_two_dups.ofv - with_dup.ofv;
    assert!(
        first.abs() > 1.0,
        "duplicating a subject did not change the objective at all ({first})"
    );
    assert!(
        (first - second).abs() < 1e-9 * first.abs(),
        "the 2nd and 3rd copies of the same subject contributed {first} and {second}"
    );
}

#[test]
fn the_flat_parameter_vector_round_trips_through_a_real_model() {
    // `--update-inits` and `--dofv` both rebuild `ModelParameters` from the flat
    // vector rather than from a `FitResult`. Evaluating the rebuilt parameters
    // must give back exactly the OFV they came from — the Δofv of a fit against
    // itself is 0, which is the `--dofv` sanity check.
    let p = prepared();
    let options = eval_options(&p.parsed.fit_options);

    let evaluated = fit(&p.parsed.model, &p.population, &p.init_params, &options).expect("eval");
    let flat = flatten_estimates(&p.init_params, &evaluated);
    let rebuilt = params_from_estimates(&p.init_params, &flat);
    let again = fit(&p.parsed.model, &p.population, &rebuilt, &options).expect("re-eval");

    assert_eq!(
        again.ofv.to_bits(),
        evaluated.ofv.to_bits(),
        "delta-ofv against the same parameters must be exactly 0, got {}",
        again.ofv - evaluated.ofv
    );
}

#[test]
fn a_small_bootstrap_runs_end_to_end_and_writes_every_artefact() {
    let mut p = prepared();
    // Tier 2: a handful of outer iterations, no convergence loop.
    p.parsed.fit_options.outer_maxiter = 2;

    let dir = tempfile::tempdir().expect("temp dir");
    let options = BootstrapOptions {
        samples: 3,
        seed: 42,
        threads: Some(1),
        directory: Some(dir.path().to_path_buf()),
        ..BootstrapOptions::default()
    };
    let result = run_bootstrap(&p, &options).expect("bootstrap runs");

    assert_eq!(result.replicates.len(), 3);
    assert_eq!(result.draws.len(), 3);
    assert!(result.original.is_some(), "the base model should have run");
    assert_eq!(result.subject_ids.len(), p.population.subjects.len());

    // Warfarin: 3 thetas + 3 diagonal omegas + 1 sigma.
    assert_eq!(result.parameter_names.len(), 7);
    assert_eq!(result.parameter_names[0], "TVCL");
    assert!(result
        .parameter_names
        .iter()
        .any(|n| n == "OMEGA(ETA_CL,ETA_CL)"));
    assert_eq!(result.n_estimated_parameters, 7);

    for r in &result.replicates {
        assert!(
            r.error.is_none(),
            "replicate {} failed: {:?}",
            r.index,
            r.error
        );
        assert_eq!(r.estimates.len(), result.parameter_names.len());
        assert!(r.ofv.is_finite());
        // No covariance step by default, so no per-replicate SEs.
        assert!(r.standard_errors.is_none());
    }

    for name in [
        "raw_results.csv",
        "bootstrap_results.csv",
        "bootstrap_diagnostics.csv",
        "all_individuals1.csv",
        "included_individuals1.csv",
        "included_keys1.csv",
        "sample_keys1.csv",
    ] {
        let path = dir.path().join(name);
        assert!(path.exists(), "{name} was not written");
        assert!(
            std::fs::metadata(&path).unwrap().len() > 0,
            "{name} is empty"
        );
    }
    assert!(!dir.path().join("delta_ofv.csv").exists());
}

#[test]
fn the_drawn_datasets_do_not_depend_on_the_thread_count() {
    // The reproducibility claim, on the real model: the same seed must give the
    // same replicates whether the run is serial or parallel. PsN's guide
    // documents the opposite behaviour as a known wart.
    let mut p = prepared();
    p.parsed.fit_options.outer_maxiter = 1;

    let base = BootstrapOptions {
        samples: 4,
        seed: 7,
        directory: None,
        run_base_model: false,
        ..BootstrapOptions::default()
    };
    let serial = run_bootstrap(
        &p,
        &BootstrapOptions {
            threads: Some(1),
            ..base.clone()
        },
    )
    .expect("serial bootstrap");
    let parallel = run_bootstrap(
        &p,
        &BootstrapOptions {
            threads: Some(4),
            ..base
        },
    )
    .expect("parallel bootstrap");

    assert_eq!(serial.draws, parallel.draws);
    // And the results come back in replicate order regardless of who finished
    // first, so row j of raw_results is replicate j either way.
    let indices: Vec<usize> = parallel.replicates.iter().map(|r| r.index).collect();
    assert_eq!(indices, vec![1, 2, 3, 4]);
}

/// The shipped warfarin dataset carries no grouping column, so write a copy with
/// one: `STUDY` alternates by ID, and is constant within each subject.
///
/// Deriving it from the real file rather than hand-writing three rows is the
/// point — the strata have to be matched against the same IDs the reader
/// produced, through the same header conventions.
fn warfarin_with_study_column() -> tempfile::NamedTempFile {
    use std::io::Write;
    let src = std::fs::read_to_string(DATA).expect("warfarin data");
    let mut lines = src.lines();
    let header = lines.next().expect("header row");
    let id_col = header
        .split(',')
        .position(|h| h.trim().eq_ignore_ascii_case("ID"))
        .expect("an ID column");

    let mut out = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp file");
    writeln!(out, "{header},STUDY").expect("write header");
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let id: u64 = line
            .split(',')
            .nth(id_col)
            .and_then(|s| s.trim().parse().ok())
            .expect("numeric ID");
        writeln!(out, "{line},{}", 1001 + id % 2).expect("write row");
    }
    out.flush().expect("flush");
    out
}

#[test]
fn a_stratified_run_keeps_every_replicate_at_the_stratum_sizes() {
    let p = prepared();
    let data = warfarin_with_study_column();
    let strata = ferx_tools::bootstrap::strata_from_csv(
        data.path().to_str().expect("utf-8 path"),
        &p.parsed.column_map,
        &p.population,
        "STUDY",
    )
    .expect("STUDY is constant within each subject");

    // Non-vacuity: the split has to be a real one, or "composition preserved"
    // is trivially true.
    assert_eq!(strata.groups.len(), 2, "expected two strata");
    assert!(strata.groups.values().all(|g| !g.is_empty()));
    assert_eq!(
        strata.groups.values().map(|g| g.len()).sum::<usize>(),
        p.population.subjects.len()
    );

    let allocation = strata.allocation(&SampleSize::Original).unwrap();
    assert_eq!(
        allocation.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
        strata.groups.values().map(|g| g.len()).collect::<Vec<_>>()
    );

    for index in 1..=10 {
        let replicate = resample::draw(&strata, &allocation, 11, index);
        for (label, group) in &strata.groups {
            let n = replicate.keys.iter().filter(|k| group.contains(k)).count();
            assert_eq!(
                n,
                group.len(),
                "replicate {index}: stratum `{label}` drew {n}, expected {}",
                group.len()
            );
        }
    }
}

#[test]
fn a_stratification_column_that_varies_within_a_subject_is_refused_on_real_data() {
    use std::io::Write;
    // Same dataset, but the column now changes between records of one subject —
    // PsN's "at least one individual has multiple values" error.
    let src = std::fs::read_to_string(DATA).expect("warfarin data");
    let mut lines = src.lines();
    let header = lines.next().expect("header row");
    let mut file = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp file");
    writeln!(file, "{header},STUDY").expect("write header");
    for (i, line) in lines.filter(|l| !l.trim().is_empty()).enumerate() {
        writeln!(file, "{line},{}", 1001 + (i as u64 % 2)).expect("write row");
    }
    file.flush().expect("flush");

    let p = prepared();
    let err = ferx_tools::bootstrap::strata_from_csv(
        file.path().to_str().expect("utf-8 path"),
        &p.parsed.column_map,
        &p.population,
        "STUDY",
    )
    .unwrap_err();
    assert!(err.contains("more than one value"), "{err}");
}

#[test]
fn a_missing_model_file_is_an_error_not_a_panic() {
    assert!(prepare_run("does-not-exist.ferx", Some(DATA)).is_err());
    assert!(Path::new(MODEL).exists(), "the fixture moved");
}
