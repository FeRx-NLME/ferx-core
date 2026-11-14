//! #2911: the fit YAML's estimate sections — top-level and under `estimates:` — and
//! `fit.json`'s `estimates` object are one table rendered three times.

use super::*;
use crate::types::{
    EtaParamInfo, EtaParamType, HeldAtBound, HeldReason, MixtureOverrideParam, ParameterKind,
    ResidualCorrelation, SigmaType,
};
use nalgebra::DMatrix;

/// Every shape the estimate sections render: a FIX θ, a θ held on its bound, a θ with no
/// SE entry; a 2×2 **block** Ω (one off-diagonal, full lower-triangle SEs) with a
/// log-normal and an additive η; a proportional and an additive σ (the latter FIX) and a
/// free ρ between them; a 2×2 block κ with a level and a weight; one Ω and one σ
/// `[mixture]` override, one of them held.
pub(super) fn rich_result() -> FitResult {
    let mut r = crate::types::test_helpers::minimal_fit_result();
    r.theta = vec![1.5, 20.0, 0.75, 1e-9];
    r.theta_names = vec!["CL".into(), "V".into(), "KA".into(), "TINY".into()];
    r.theta_fixed = vec![false, true, false, false];
    r.se_theta = Some(vec![0.15, 0.0, 0.0, 0.25]);
    r.held_at_bound = vec![
        HeldAtBound {
            name: "KA".into(),
            kind: ParameterKind::Theta,
            index: 2,
            packed_index: 2,
            reason: HeldReason::ProfiledBound,
            class: None,
        },
        HeldAtBound {
            name: "add_MIX2".into(),
            kind: ParameterKind::Sigma,
            index: 1,
            packed_index: 12,
            reason: HeldReason::CollapsedAtGuard,
            class: Some(2),
        },
    ];
    r.omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.03, 0.03, 0.16]);
    r.omega_init = r.omega.clone();
    r.omega_is_diagonal = Some(false);
    r.se_omega = Some(vec![0.01, 0.005, 0.02]);
    r.eta_param_info = vec![
        EtaParamInfo {
            eta_name: "eta_CL".into(),
            param_type: EtaParamType::LogNormal,
            linked_theta: Some("CL".into()),
            individual_param_name: "CL".into(),
        },
        EtaParamInfo {
            eta_name: "eta_V".into(),
            param_type: EtaParamType::Additive,
            linked_theta: Some("V".into()),
            individual_param_name: "V".into(),
        },
    ];
    r.sigma = vec![0.12, 0.5];
    r.sigma_names = vec!["prop".into(), "add".into()];
    r.sigma_types = vec![SigmaType::Proportional, SigmaType::Additive];
    r.sigma_fixed = vec![false, true];
    r.se_sigma = Some(vec![0.012, 0.0]);
    r.error_model = crate::types::ErrorModel::Combined;
    r.residual_correlations = vec![ResidualCorrelation {
        sigma_i: 1,
        sigma_j: 0,
        rho: 0.25,
    }];
    r.residual_correlation_fixed = vec![false];
    r.se_residual_correlations = Some(vec![0.05]);
    r.omega_iov = Some(DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.0625]));
    r.kappa_names = vec!["KAPPA_CL".into(), "KAPPA_V".into()];
    r.kappa_fixed = vec![false, false];
    r.kappa_is_diagonal = Some(false);
    r.se_kappa = Some(vec![0.004, 0.006]);
    r.kappa_levels = vec!["OCC".into(), "OCC".into()];
    r.kappa_param_types = vec![EtaParamType::LogNormal, EtaParamType::Additive];
    r.kappa_weights = vec![Some("NARM".into()), None];
    r.kappa_weight_typical = vec![Some(4.0), None];
    let param = |name: &str, kind, index, packed_index| {
        let mut p: MixtureOverrideParam = serde_json::from_value(serde_json::json!({
            "name": name, "kind": kind, "index": index, "class": 2,
            "packed_index": packed_index, "fixed": false,
        }))
        .unwrap();
        p.name = name.into();
        p
    };
    r.mixture_override_params = vec![
        param("eta_CL_MIX2", "Omega", 0, 11),
        param("add_MIX2", "Sigma", 1, 12),
    ];
    r.mixture_overrides = Some(vec![0.3, 0.7]);
    r.se_mixture_overrides = Some(vec![0.06, 0.0]);
    r
}

/// The YAML's estimate sections, read back as `(section, entry, field) → scalar text`:
/// the top-level copies (`nested = false`) or the `estimates:` block (`nested = true`).
/// Only the sections [`estimate_sections`] owns, at the writer's exact indentation.
fn yaml_estimate_fields(
    yaml: &str,
    nested: bool,
) -> std::collections::BTreeMap<(String, String, String), String> {
    const OWNED: &[&str] = &[
        "theta",
        "omega",
        "sigma",
        "block_sigma",
        "omega_iov",
        "mixture_overrides",
    ];
    let base = if nested { 2 } else { 0 };
    let mut out = std::collections::BTreeMap::new();
    let mut in_estimates = false;
    let (mut section, mut entry): (Option<String>, Option<String>) = (None, None);
    for line in yaml.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if indent == 0 {
            in_estimates = t == "estimates:";
            section = None;
            entry = None;
            if nested {
                continue;
            }
        }
        if nested && !in_estimates {
            continue;
        }
        if indent == base {
            let key = t.split(':').next().unwrap();
            section = OWNED.contains(&key).then(|| key.to_string());
            entry = None;
        } else if indent == base + 2 && section.is_some() {
            entry = Some(t.trim_end_matches(':').to_string());
        } else if indent == base + 4 {
            if let (Some(s), Some(e)) = (&section, &entry) {
                let (k, v) = t.split_once(": ").unwrap_or_else(|| panic!("{line}"));
                out.insert((s.clone(), e.clone(), k.to_string()), v.to_string());
            }
        }
    }
    out
}

/// The JSON `estimates` object flattened the same way.
fn json_estimate_fields(
    estimates: &serde_json::Value,
) -> std::collections::BTreeMap<(String, String, String), serde_json::Value> {
    let mut out = std::collections::BTreeMap::new();
    for (s, entries) in estimates.as_object().expect("estimates is an object") {
        for (e, fields) in entries.as_object().expect("a section is an object") {
            for (k, v) in fields.as_object().expect("an entry is an object") {
                out.insert((s.clone(), e.clone(), k.clone()), v.clone());
            }
        }
    }
    out
}

/// Assert the YAML file and the JSON `estimates` of `r` carry the same sections, entries
/// and fields, with the JSON value being the YAML's before rounding. Returns the
/// number of fields compared, so a caller can assert the fixture reached them.
fn assert_yaml_and_json_agree(r: &FitResult) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m-fit.yaml");
    write_estimates_yaml(r, path.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let yaml = yaml_estimate_fields(&text, false);
    // The `estimates:` block is the top-level sections again, field for field: one path
    // reads both files, and the old top-level one keeps working.
    assert_eq!(
        yaml_estimate_fields(&text, true),
        yaml,
        "the YAML's `estimates:` block differs from its top-level sections:\n{text}"
    );
    assert_eq!(text.matches("\nestimates:\n").count(), 1, "{text}");
    // Through the writer and `serde_json`'s text, as a reader of `{model}-fit.json` sees it.
    let json_path = dir.path().join("m-fit.json");
    write_result_json(r, json_path.to_str().unwrap()).unwrap();
    let full: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
    let json = json_estimate_fields(&full["estimates"]);

    let yaml_keys: Vec<_> = yaml.keys().collect();
    let json_keys: Vec<_> = json.keys().collect();
    assert_eq!(yaml_keys, json_keys, "YAML and JSON estimate fields differ");

    for (key, text) in &yaml {
        let v = &json[key];
        let ok = match v {
            serde_json::Value::Null => text == "~" || text == "NaN" || text.ends_with("inf"),
            serde_json::Value::Bool(b) => text == &b.to_string(),
            serde_json::Value::String(s) => text == s || text == &yaml_quote(s),
            serde_json::Value::Number(n) => {
                let x = n.as_f64().unwrap();
                assert!(x.is_finite(), "{key:?}");
                // The YAML's decimals, re-rendered from the full-precision JSON value.
                let decimals = text.split_once('.').map_or(0, |(_, d)| d.len());
                format!("{x:.decimals$}") == *text
            }
            other => panic!("{key:?}: unexpected JSON value {other}"),
        };
        assert!(ok, "{key:?}: YAML `{text}` vs JSON {v}");
    }
    yaml.len()
}

#[test]
fn json_estimates_are_the_yaml_estimate_sections() {
    let r = rich_result();
    let n = assert_yaml_and_json_agree(&r);
    // Every section the fixture exists to reach: a regression dropping one from either
    // side must not pass on the strength of the others.
    let json = r.to_json_value();
    let est = &json["estimates"];
    for s in [
        "theta",
        "omega",
        "sigma",
        "block_sigma",
        "omega_iov",
        "mixture_overrides",
    ] {
        assert!(est[s].is_object(), "`estimates.{s}` missing: {est}");
    }
    assert!(n >= 60, "only {n} fields compared");

    // The layout the issue asks for, spelled out: keyed by name, YAML field names.
    assert_eq!(est["theta"]["CL"]["estimate"], 1.5);
    assert_eq!(est["theta"]["CL"]["se"], 0.15);
    assert_eq!(est["theta"]["V"]["fixed"], true);
    assert_eq!(est["theta"]["V"]["se"], serde_json::Value::Null);
    assert_eq!(est["theta"]["KA"]["held"], "profiled_bound");
    assert_eq!(est["omega"]["eta_CL"]["variance"], 0.09);
    assert_eq!(est["omega"]["eta_V__eta_CL"]["covariance"], 0.03);
    assert_eq!(est["sigma"]["prop"]["estimate"], 0.12);
    assert_eq!(est["omega_iov"]["KAPPA_CL"]["weight"], "NARM");
    assert_eq!(
        est["mixture_overrides"]["add_MIX2"]["held"],
        "collapsed_at_guard"
    );
}

#[test]
fn json_estimates_keep_the_precision_the_yaml_rounds_away() {
    // `TINY` = 1e-9 is `0.000000` in the YAML; the JSON is the reason to read it instead.
    let r = rich_result();
    let est = r.to_json_value()["estimates"].clone();
    assert_eq!(est["theta"]["TINY"]["estimate"], 1e-9);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m-fit.yaml");
    write_estimates_yaml(&r, path.to_str().unwrap()).unwrap();
    let yaml = yaml_estimate_fields(&std::fs::read_to_string(&path).unwrap(), true);
    let key = |f: &str| ("theta".to_string(), "TINY".to_string(), f.to_string());
    assert_eq!(yaml[&key("estimate")], "0.000000");
    // A non-finite estimate, and the NaN %RSE it gives, are `null` in the JSON — where the
    // YAML writes `NaN` — never a string.
    let mut r = rich_result();
    r.theta[0] = f64::NAN;
    assert_yaml_and_json_agree(&r);
    let est = r.to_json_value()["estimates"].clone();
    assert_eq!(est["theta"]["CL"]["estimate"], serde_json::Value::Null);
    assert_eq!(est["theta"]["CL"]["rse_pct"], serde_json::Value::Null);
}

#[test]
fn json_estimates_agree_with_the_yaml_on_the_minimal_and_empty_fits() {
    assert!(assert_yaml_and_json_agree(&crate::types::test_helpers::minimal_fit_result()) > 0);
    // No θ, no η, no σ: the three base sections are present and empty on both sides, the
    // optional ones absent on both.
    let r = crate::types::test_helpers::empty_fit_result();
    assert_yaml_and_json_agree(&r);
    let est = r.to_json_value()["estimates"].clone();
    let keys: Vec<_> = est.as_object().unwrap().keys().cloned().collect();
    assert_eq!(keys, ["omega", "sigma", "theta"]);
    // In the YAML too, as empty maps (`{}`) — not `null` — under `estimates:`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m-fit.yaml");
    write_estimates_yaml(&r, path.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let block = &text[text.find("\nestimates:\n").unwrap()..];
    for s in ["theta", "omega"] {
        assert!(block.contains(&format!("\n  {s}: {{}}\n")), "{block}");
    }
    assert!(block.contains("\n  sigma: {}  # error model:"), "{block}");
}

#[test]
fn a_fit_json_with_estimates_still_reads_back() {
    // `estimates` is derived output: `from_json_str` ignores it, as it does `status`.
    let r = rich_result();
    let text = serde_json::to_string(&r.to_json_value()).unwrap();
    let back = FitResult::from_json_str(&text).unwrap();
    assert_eq!(back.theta_names, r.theta_names);
    assert_eq!(
        back.to_json_value()["estimates"],
        r.to_json_value()["estimates"]
    );
}

#[test]
fn an_unrenderable_block_sigma_gives_null_estimates_not_a_partial_table() {
    let mut r = rich_result();
    r.residual_correlations[0].sigma_i = 7;
    assert_eq!(r.to_json_value()["estimates"], serde_json::Value::Null);
}
