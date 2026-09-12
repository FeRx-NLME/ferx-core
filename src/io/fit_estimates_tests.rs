//! Tier-1 tests for the `[priors] from_fit` fit-file readers (#254 phase 2).

use super::*;
use crate::types::test_helpers::minimal_fit_result;
use crate::types::Population;
use nalgebra::DMatrix;

/// Look one estimate up by kind and name.
fn find<'a>(v: &'a [SourceEstimate], kind: EstimateKind, name: &str) -> &'a SourceEstimate {
    v.iter()
        .find(|e| e.kind == kind && e.name == name)
        .unwrap_or_else(|| panic!("no {kind:?} named {name} in {v:#?}"))
}

/// A fit with an IOV block and a FIXed θ, so every branch of
/// [`from_fit_result`] has something to produce.
fn fit_with_iov() -> FitResult {
    let mut r = minimal_fit_result();
    r.theta_fixed = vec![false, true, false];
    r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[0.04]));
    r.kappa_names = vec!["kappa_CL".into()];
    r.kappa_fixed = vec![false];
    r.kappa_init_as_sd = vec![false];
    r.se_kappa = Some(vec![0.008]);
    r
}

#[test]
fn from_fit_result_reports_omega_as_a_variance_and_sigma_as_an_sd() {
    let est = from_fit_result(&fit_with_iov());

    // θ: the natural value and its SE, straight through.
    let cl = find(&est, EstimateKind::Theta, "CL");
    assert_eq!(cl.value, 1.0);
    assert_eq!(cl.se, Some(0.01));

    // Ω: the *variance* (0.1), not the SD — the single scale rule this whole
    // module is organised around.
    let eta_cl = find(&est, EstimateKind::Omega, "eta_CL");
    assert_eq!(eta_cl.value, 0.1);
    assert_eq!(eta_cl.se, Some(0.01));

    // Σ: the *SD* (0.05), because that is how `FitResult` stores it.
    let prop = find(&est, EstimateKind::Sigma, "prop");
    assert_eq!(prop.value, 0.05);
    assert_eq!(prop.se, Some(0.001));

    // κ: a variance, like Ω.
    let kappa = find(&est, EstimateKind::Kappa, "kappa_CL");
    assert_eq!(kappa.value, 0.04);
    assert_eq!(kappa.se, Some(0.008));
}

#[test]
fn a_fixed_parameter_reports_no_standard_error() {
    // `V` is FIXed in `fit_with_iov`, but `se_theta` still carries a number for
    // it. Reading that number through would give the importer a prior on a
    // parameter that cannot move, so it must come back as absent — and the
    // `.yaml` path cannot report one either (the writer emits `se: ~`), which is
    // the disagreement this pins shut.
    let est = from_fit_result(&fit_with_iov());
    assert_eq!(find(&est, EstimateKind::Theta, "V").se, None);
    assert_eq!(find(&est, EstimateKind::Theta, "V").value, 2.0);
}

#[test]
fn a_zero_or_missing_standard_error_reads_as_absent() {
    let mut r = minimal_fit_result();
    // 0.0 is what `compute_standard_errors` leaves for a coordinate the
    // covariance step could not reach. It means "no SE", not "an infinitely
    // sharp prior" — and `rse = 0` would give a prior SD of 0 and a division by
    // zero in the penalty.
    r.se_theta = Some(vec![0.0, f64::NAN, -1.0]);
    let est = from_fit_result(&r);
    for name in ["CL", "V", "KA"] {
        assert_eq!(find(&est, EstimateKind::Theta, name).se, None, "{name}");
    }

    let mut r = minimal_fit_result();
    r.se_theta = None;
    r.se_omega = None;
    r.se_sigma = None;
    let est = from_fit_result(&r);
    assert!(est.iter().all(|e| e.se.is_none()));
    // …but the estimates themselves are still there, so the caller's diagnostic
    // can name the parameters rather than reporting an empty source.
    assert_eq!(est.len(), 6);
}

#[test]
fn yaml_round_trip_matches_the_fit_result() {
    // The oracle for the hand-written YAML reader: the same fit, flattened two
    // ways. `from_fit_result` is the reference because it reads `FitResult`
    // directly; the YAML path goes out through `write_estimates_yaml`'s
    // formatting and back, so anything the reader mis-indents, mis-keys or
    // mis-sections shows up as a missing or wrong entry here.
    let fit = fit_with_iov();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run-fit.yaml");
    crate::io::output::write_estimates_yaml(&fit, path.to_str().unwrap()).unwrap();

    let want = from_fit_result(&fit);
    let got = read_fit_estimates(&path).unwrap();

    assert_eq!(got.len(), want.len(), "got {got:#?}\nwant {want:#?}");
    for w in &want {
        let g = find(&got, w.kind, &w.name);
        // The writer formats `{:.6}`, so compare at that resolution rather than
        // bit-for-bit; every value in the fixture is exactly representable there.
        assert!(
            (g.value - w.value).abs() < 5e-7,
            "{:?} {}: {} vs {}",
            w.kind,
            w.name,
            g.value,
            w.value
        );
        match (g.se, w.se) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 5e-7, "{} se {a} vs {b}", w.name),
            (None, None) => {}
            (a, b) => panic!("{} se {a:?} vs {b:?}", w.name),
        }
    }
}

#[test]
fn yaml_skips_block_omega_off_diagonal_entries() {
    // An off-diagonal is written as `NAME_I__NAME_J:` with a `covariance:` key
    // and no `variance:`. The reader must drop it — a covariance imported as a
    // variance prior would be a prior on the wrong quantity, with the wrong
    // magnitude, on a coordinate v1 rejects anyway.
    let mut r = minimal_fit_result();
    r.omega = DMatrix::from_row_slice(2, 2, &[0.1, 0.03, 0.03, 0.2]);
    r.se_omega = Some(vec![0.01, 0.005, 0.02]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blk-fit.yaml");
    crate::io::output::write_estimates_yaml(&r, path.to_str().unwrap()).unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("eta_V__eta_CL:"),
        "fixture lost its off-diagonal"
    );

    let got = parse_estimates_yaml(&text).unwrap();
    let omegas: Vec<&str> = got
        .iter()
        .filter(|e| e.kind == EstimateKind::Omega)
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(omegas, vec!["eta_CL", "eta_V"]);
}

#[test]
fn yaml_reads_the_commented_sigma_section_header() {
    // `write_estimates_yaml` writes `sigma:  # error model: proportional`. A
    // reader that matched the raw line would miss the whole σ section and
    // silently import no residual-error prior.
    let text = "\nsigma:  # error model: proportional\n  prop:\n    estimate: 0.050000\n    se: 0.001000\n";
    let got = parse_estimates_yaml(text).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].kind, EstimateKind::Sigma);
    assert_eq!(got[0].value, 0.05);
    assert_eq!(got[0].se, Some(0.001));
}

#[test]
fn yaml_honours_the_fixed_flag_and_the_null_se() {
    let text = "theta:\n  CL:\n    estimate: 1.000000\n    fixed: true\n    se: ~\n  V:\n    estimate: 2.000000\n    se: ~\n";
    let got = parse_estimates_yaml(text).unwrap();
    assert_eq!(got.len(), 2);
    assert!(got.iter().all(|e| e.se.is_none()));
}

#[test]
fn yaml_that_is_not_a_fit_report_is_an_error() {
    let err = parse_estimates_yaml("hello: world\n").unwrap_err();
    assert!(err.contains("theta:"), "{err}");
}

#[test]
fn json_round_trip_matches_the_fit_result() {
    let fit = fit_with_iov();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run-fit.json");
    crate::io::output::write_result_json(&fit, path.to_str().unwrap()).unwrap();

    let got = read_fit_estimates(&path).unwrap();
    let want = from_fit_result(&fit);
    assert_eq!(got.len(), want.len());
    for w in &want {
        let g = find(&got, w.kind, &w.name);
        assert_eq!(g.value, w.value, "{}", w.name);
        assert_eq!(g.se, w.se, "{}", w.name);
    }
}

#[test]
fn fitrx_round_trip_matches_the_fit_result() {
    let mut fit = fit_with_iov();
    // `save_fit` walks `subjects` against the population it is handed; only the
    // parameter block matters here, so give it neither.
    fit.subjects = Vec::new();
    fit.n_subjects = 0;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.fitrx");
    crate::io::fitrx::save_fit(
        &fit,
        &Population {
            subjects: Vec::new(),
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        },
        "model TEST\nend\n",
        &path,
        crate::io::fitrx::SaveFitOptions::default(),
    )
    .unwrap();

    let got = read_fit_estimates(&path).unwrap();
    let want = from_fit_result(&fit);
    assert_eq!(got.len(), want.len());
    for w in &want {
        let g = find(&got, w.kind, &w.name);
        assert_eq!(g.value, w.value, "{}", w.name);
        assert_eq!(g.se, w.se, "{}", w.name);
    }
}

#[test]
fn an_unsupported_extension_names_the_three_that_work() {
    let err = read_fit_estimates(std::path::Path::new("run-fit.csv")).unwrap_err();
    assert!(err.contains(".fitrx"), "{err}");
    assert!(err.contains(".json"), "{err}");
    assert!(err.contains(".yaml"), "{err}");
}

#[test]
fn a_missing_file_names_the_path() {
    let err = read_fit_estimates(std::path::Path::new("/no/such/run-fit.yaml")).unwrap_err();
    assert!(err.contains("/no/such/run-fit.yaml"), "{err}");
}

#[test]
fn json_that_is_not_a_fit_result_is_an_error_naming_the_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.json");
    std::fs::write(&path, "{\"not\": \"a fit\"}").unwrap();
    let err = read_fit_estimates(&path).unwrap_err();
    assert!(err.contains("--output-format json"), "{err}");
}

#[test]
fn a_covariate_nn_weight_is_not_offered_as_a_prior_target() {
    // The θ exclusions are shared with the YAML writer so `from_fit` means the
    // same thing whichever file the user kept. A large `theta NAME[LEVEL]` block
    // is summarized under `theta_blocks:` rather than listed per θ, so a `.json`
    // import must not see them either.
    let mut r = minimal_fit_result();
    r.theta = vec![1.0, 2.0, 0.5];
    r.theta_names = vec!["CL".into(), "SITE[A]".into(), "SITE[B]".into()];
    // Two levels is below the compaction threshold, so both stay in `theta:` and
    // both are offered — the gate is the writer's, not a name-shape rule.
    assert_eq!(from_fit_result(&r).len(), 3 + 2 + 1);

    let n = crate::io::output::THETA_BLOCK_COMPACT_MIN;
    r.theta = vec![1.0; n + 1];
    r.theta_names = std::iter::once("CL".to_string())
        .chain((0..n).map(|i| format!("SITE[{i}]")))
        .collect();
    r.theta_fixed = vec![false; n + 1];
    r.se_theta = Some(vec![0.01; n + 1]);
    let est = from_fit_result(&r);
    let thetas: Vec<&str> = est
        .iter()
        .filter(|e| e.kind == EstimateKind::Theta)
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(thetas, vec!["CL"]);
}

#[test]
fn a_nameless_omega_sigma_or_kappa_falls_back_to_the_writers_spelling() {
    // `write_estimates_yaml` keys an unnamed Ω/Σ/κ as `omega_1_1` / `sigma_1` /
    // `kappa_1`. Producing a different placeholder here would break the YAML
    // round-trip on exactly the fits (older artifacts, hand-built results) that
    // have no names to carry.
    let mut r = minimal_fit_result();
    r.eta_names = Vec::new();
    r.sigma_names = Vec::new();
    r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[0.04]));
    r.kappa_names = Vec::new();
    r.kappa_fixed = Vec::new();
    r.se_kappa = Some(vec![0.008]);

    let names: Vec<String> = from_fit_result(&r)
        .iter()
        .filter(|e| e.kind != EstimateKind::Theta)
        .map(|e| e.name.clone())
        .collect();
    assert_eq!(
        names,
        ["omega_1_1", "omega_2_2", "sigma_1", "kappa_1"],
        "must match write_estimates_yaml's fallback keys"
    );
}

#[test]
fn every_kind_names_the_keyword_it_is_declared_with() {
    // The note the importer emits is `"<keyword> <name>: <reason>"`, so a user
    // scanning the warnings can tell an Ω called `CL` from a θ called `CL`.
    assert_eq!(EstimateKind::Theta.keyword(), "theta");
    assert_eq!(EstimateKind::Omega.keyword(), "omega");
    assert_eq!(EstimateKind::Sigma.keyword(), "sigma");
    assert_eq!(EstimateKind::Kappa.keyword(), "kappa");
}
