use super::*;
use std::collections::BTreeMap;
use std::path::Path;

fn names() -> Vec<String> {
    vec!["CL".to_string(), "V".to_string()]
}

fn options() -> BootstrapOptions {
    BootstrapOptions {
        samples: 20,
        seed: 7,
        ..BootstrapOptions::default()
    }
}

fn manifest() -> RunManifest {
    RunManifest::new(
        &options(),
        Some("model-hash".to_string()),
        Some("data-hash".to_string()),
        &names(),
    )
}

#[test]
fn the_manifest_records_the_run_inputs() {
    let m = manifest();
    assert_eq!(m.seed, 7);
    assert_eq!(m.samples, 20);
    assert_eq!(m.sample_size, "original");
    assert_eq!(m.stratify_on, None);
    assert_eq!(m.model_hash.as_deref(), Some("model-hash"));
    assert_eq!(m.parameter_names, names());
    assert!(!m.keep_covariance && !m.dofv);
    assert!(m.run_base_model);
    assert_eq!(m.replicate_inits, "base_fit");
}

/// The recorded mode is the *effective* one, not the raw `update_inits` flag.
///
/// `update_inits` does nothing without a base fit, so a caller that sets it
/// alongside `run_base_model = false` is asking for the model file's estimates
/// just as plainly as one that left it off. Comparing the raw flags would refuse
/// a resume that changes nothing.
#[test]
fn the_recorded_initialization_mode_is_the_effective_one() {
    let base = |run_base_model, update_inits| BootstrapOptions {
        run_base_model,
        update_inits,
        ..BootstrapOptions::default()
    };
    assert_eq!(describe_replicate_inits(&base(true, true)), "base_fit");
    assert_eq!(describe_replicate_inits(&base(true, false)), "model_file");
    assert_eq!(describe_replicate_inits(&base(false, false)), "model_file");
    // `update_inits` with nothing to take inits from resolves to the model file,
    // and so must not differ from the flag being off.
    assert_eq!(describe_replicate_inits(&base(false, true)), "model_file");

    let manifest_of = |o| RunManifest::new(&o, None, None, &names());
    assert!(manifest_of(base(false, true))
        .check_compatible(&manifest_of(base(false, false)), Path::new("run"))
        .is_ok());
}

#[test]
fn a_sample_size_round_trips_through_its_description() {
    assert_eq!(describe_sample_size(&SampleSize::Original), "original");
    assert_eq!(describe_sample_size(&SampleSize::Total(32)), "total:32");

    let mut map = BTreeMap::new();
    map.insert("1001".to_string(), 12usize);
    map.insert("1002".to_string(), 24usize);
    assert_eq!(
        describe_sample_size(&SampleSize::PerStratum(map)),
        "per_stratum:1001=>12,1002=>24"
    );
}

#[test]
fn the_manifest_survives_a_write_and_read() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("bootstrap_run.json");
    manifest().write(&path).expect("write");
    assert_eq!(RunManifest::read(&path).expect("read"), manifest());
}

#[test]
fn a_missing_or_unparseable_manifest_is_an_error_naming_the_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("bootstrap_run.json");
    let err = RunManifest::read(&path).unwrap_err();
    assert!(err.contains("bootstrap_run.json"), "{err}");

    std::fs::write(&path, "not json at all").expect("write");
    let err = RunManifest::read(&path).unwrap_err();
    assert!(err.contains("cannot parse"), "{err}");
}

#[test]
fn a_matching_manifest_is_compatible_with_itself() {
    let m = manifest();
    assert!(m.check_compatible(&m, Path::new("run")).is_ok());
}

/// Every field the manifest carries must be able to *refuse* a resume, and the
/// message must say which one differs. A generic "does not match" leaves the
/// user to diff two files by hand to find out what they changed.
#[test]
fn each_differing_field_refuses_the_resume_by_name() {
    let disk = manifest();
    let cases: Vec<(&str, RunManifest)> = vec![
        (
            "--seed",
            RunManifest {
                seed: 8,
                ..disk.clone()
            },
        ),
        (
            "--samples",
            RunManifest {
                samples: 21,
                ..disk.clone()
            },
        ),
        (
            "--sample-size",
            RunManifest {
                sample_size: "total:5".to_string(),
                ..disk.clone()
            },
        ),
        (
            "--stratify-on",
            RunManifest {
                stratify_on: Some("STUD".to_string()),
                ..disk.clone()
            },
        ),
        (
            "--run-base-model",
            RunManifest {
                run_base_model: false,
                ..disk.clone()
            },
        ),
        (
            "--update-inits",
            RunManifest {
                replicate_inits: "model_file".to_string(),
                ..disk.clone()
            },
        ),
        (
            "--keep-covariance",
            RunManifest {
                keep_covariance: true,
                ..disk.clone()
            },
        ),
        (
            "--dofv",
            RunManifest {
                dofv: true,
                ..disk.clone()
            },
        ),
        (
            "model file",
            RunManifest {
                model_hash: Some("other".to_string()),
                ..disk.clone()
            },
        ),
        (
            "dataset",
            RunManifest {
                data_hash: Some("other".to_string()),
                ..disk.clone()
            },
        ),
        (
            "parameter vector",
            RunManifest {
                parameter_names: vec!["CL".to_string()],
                ..disk.clone()
            },
        ),
    ];
    for (field, now) in cases {
        let err = now
            .check_compatible(&disk, Path::new("run"))
            .expect_err("a differing manifest must refuse the resume");
        assert!(err.contains(field), "expected `{field}` to be named: {err}");
        assert!(err.contains("run"), "the directory should be named: {err}");
    }
}

/// A hash is `None` when the file could not be hashed. Comparing `None` against
/// `Some` would turn a *missing* check into a hard refusal, which would make
/// `--resume` unusable on any platform where hashing failed once.
#[test]
fn an_absent_hash_does_not_refuse_the_resume() {
    let disk = manifest();
    let now = RunManifest {
        model_hash: None,
        data_hash: None,
        ..disk.clone()
    };
    assert!(now.check_compatible(&disk, Path::new("run")).is_ok());
    assert!(disk.check_compatible(&now, Path::new("run")).is_ok());
}
