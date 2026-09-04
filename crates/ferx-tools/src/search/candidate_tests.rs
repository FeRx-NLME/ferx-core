//! Candidate identity, feature vectors and ranking criteria (#1178).

use ferx_core::{bic, BicType, StrictnessVerdict};

use super::*;
use crate::search::test_support::{converged_fit, model_text, same_model_two_ways};

// ── FeatureVector ────────────────────────────────────────────────────────────

#[test]
fn features_render_in_key_order_whatever_order_they_were_set_in() {
    let a = FeatureVector::new()
        .with("PERIPHERALS", "1")
        .with("ABSORPTION", "FO");
    let b = FeatureVector::new()
        .with("ABSORPTION", "FO")
        .with("PERIPHERALS", "1");
    assert_eq!(a.render(), "ABSORPTION=FO;PERIPHERALS=1");
    assert_eq!(a, b, "insertion order leaked into the value");
}

#[test]
fn setting_a_feature_twice_replaces_it() {
    let mut features = FeatureVector::new().with("CL-WT", "lin");
    assert_eq!(features.set("CL-WT", "pow"), Some("lin".to_string()));
    assert_eq!(features.get("CL-WT"), Some("pow"));
    assert_eq!(features.len(), 1);
    assert!(!features.is_empty());
    assert!(FeatureVector::new().is_empty());
    assert_eq!(FeatureVector::new().render(), "");
}

#[test]
fn features_round_trip_through_serde() {
    // The journal stores them, so a resumed run's table must describe its rows
    // the same way the interrupted run's did.
    let features: FeatureVector = [("CL-WT", "pow"), ("V-SEX", "cat")].into_iter().collect();
    let json = serde_json::to_string(&features).expect("serialize");
    assert_eq!(json, r#"{"CL-WT":"pow","V-SEX":"cat"}"#);
    let back: FeatureVector = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, features);
    assert_eq!(
        back.iter().collect::<Vec<_>>(),
        vec![("CL-WT", "pow"), ("V-SEX", "cat")]
    );
}

// ── identity ─────────────────────────────────────────────────────────────────

#[test]
fn the_hash_ignores_comments_and_whitespace_but_not_tokens() {
    let (a, b) = same_model_two_ways();
    let plain = Candidate::new("a", a);
    let decorated = Candidate::new("b", b);
    assert_eq!(plain.hash(), decorated.hash());
    assert_eq!(plain.hash().len(), 64, "not a hex sha-256");
    assert!(plain.hash().chars().all(|c| c.is_ascii_hexdigit()));

    let edited = Candidate::new(
        "c",
        model_text("[parameters]\ntheta CL = 2\ntheta V = 10\n"),
    );
    assert_ne!(
        plain.hash(),
        edited.hash(),
        "a changed initial estimate left the identity unchanged"
    );
}

#[test]
fn the_builders_carry_provenance() {
    let candidate = Candidate::new("step2", model_text("[parameters]\ntheta CL = 1\n"))
        .parent("step1")
        .features(FeatureVector::new().with("CL-WT", "pow"));
    assert_eq!(candidate.id, "step2");
    assert_eq!(candidate.parent.as_deref(), Some("step1"));
    assert_eq!(candidate.features.get("CL-WT"), Some("pow"));

    let bare = Candidate::new("base", model_text("[parameters]\ntheta CL = 1\n"));
    assert!(bare.parent.is_none());
    assert!(bare.features.is_empty());
}

// ── Criterion ────────────────────────────────────────────────────────────────

#[test]
fn each_criterion_reads_the_field_it_names() {
    let fit = converged_fit(123.0);
    assert_eq!(Criterion::Ofv.of(&fit), fit.ofv);
    assert_eq!(Criterion::Aic.of(&fit), fit.aic);
    for kind in [
        BicType::Mixed,
        BicType::Iiv,
        BicType::Random,
        BicType::Fixed,
    ] {
        assert_eq!(Criterion::Bic(kind).of(&fit), bic(&fit, kind));
    }
    // The variants are genuinely different numbers on this fit, so a criterion
    // that silently fell back to another one would show here.
    assert_ne!(Criterion::Ofv.of(&fit), Criterion::Aic.of(&fit));
    assert_ne!(
        Criterion::Bic(BicType::Mixed).of(&fit),
        Criterion::Bic(BicType::Fixed).of(&fit)
    );
}

#[test]
fn criterion_labels_are_distinct_and_serde_round_trips() {
    let all = [
        Criterion::Ofv,
        Criterion::Aic,
        Criterion::Bic(BicType::Mixed),
        Criterion::Bic(BicType::Iiv),
        Criterion::Bic(BicType::Random),
        Criterion::Bic(BicType::Fixed),
    ];
    let labels: std::collections::HashSet<&str> = all.iter().map(|c| c.label()).collect();
    assert_eq!(
        labels.len(),
        all.len(),
        "two criteria share a label, so the manifest cannot tell them apart"
    );
    for criterion in all {
        let json = serde_json::to_string(&criterion).expect("serialize");
        assert_eq!(
            serde_json::from_str::<Criterion>(&json).expect("deserialize"),
            criterion
        );
    }
    assert_eq!(Criterion::default(), Criterion::Bic(BicType::Mixed));
}

#[test]
fn two_candidates_differing_only_inside_a_quoted_path_are_not_one_candidate() {
    // The hash is the dedup key *and* the fit-cache key, so a collision here is
    // not a wasted fit — it is candidate B reported with candidate A's
    // criterion and verdict under `duplicate_of`. A comment stripper that cut
    // at the first `//` regardless of quoting ended both of these at
    // `path = "s3:`.
    let a = Candidate::new("a", model_text("[data]\n  path = \"s3://bucket/a.csv\"\n"));
    let b = Candidate::new("b", model_text("[data]\n  path = \"s3://bucket/b.csv\"\n"));
    assert_ne!(a.hash(), b.hash(), "{}", a.model.canonical_form());
    // The spelling that *is* one candidate still is, so the fix did not simply
    // stop canonicalising.
    let reflowed = Candidate::new(
        "a2",
        model_text("[data]\npath=\"s3://bucket/a.csv\"  # x\n"),
    );
    assert_eq!(a.hash(), reflowed.hash());
}

// ── eligibility ──────────────────────────────────────────────────────────────

fn result_with(criterion: f64, passed: bool, error: Option<&str>) -> CandidateResult {
    CandidateResult {
        id: "c".into(),
        hash: "0".repeat(64),
        parent: None,
        features: FeatureVector::new(),
        fit: None,
        ofv: None,
        converged: None,
        verdict: StrictnessVerdict {
            passed,
            failures: vec![],
            skipped: vec![],
        },
        criterion,
        seconds: 0.0,
        error: error.map(str::to_string),
        duplicate_of: None,
        reused: false,
    }
}

#[test]
fn only_a_passing_finite_error_free_result_is_eligible() {
    assert!(result_with(10.0, true, None).eligible());
    assert!(!result_with(10.0, false, None).eligible());
    assert!(!result_with(f64::NAN, true, None).eligible());
    assert!(!result_with(f64::INFINITY, true, None).eligible());
    assert!(!result_with(10.0, true, Some("no fit")).eligible());
}
