//! #2833: a `[mixture]` fit's per-class overrides in every renderer — the covariance
//! labels, the console (`print_results` / `format_summary` share
//! `format_mixture_override_rows`), the fit YAML and `parameter_table`.

use super::*;
use crate::estimation::parameterization::{
    coordinate_names, mixture_override_params, mixture_override_values,
};
use nalgebra::DMatrix;

/// θ ×3, a 2-η **diagonal** Ω, two σ (`prop`, `add`), one κ, an Ω override (class 2 on
/// `eta_CL`), a σ override (class 2 on `add`) and one ρ — every packed segment. Packed:
/// `[CL, V, KA, eta_CL, eta_V, prop, add, KAPPA_CL, eta_CL_MIX2, add_MIX2, ρ]`.
fn template() -> ModelParameters {
    let eta = vec!["eta_CL".to_string(), "eta_V".to_string()];
    let names = vec!["prop".to_string(), "add".to_string()];
    let omega = OmegaMatrix::from_diagonal(&[0.1, 0.2], eta.clone());
    let sigma = SigmaVector {
        values: vec![0.05, 0.4],
        names: names.clone(),
    };
    ModelParameters {
        residual_correlations: vec![ResidualCorrelation {
            sigma_i: 1,
            sigma_j: 0,
            rho: 0.5,
        }],
        residual_correlation_fixed: vec![false],
        theta: vec![1.0, 2.0, 0.5],
        theta_names: vec!["CL".into(), "V".into(), "KA".into()],
        theta_lower: vec![0.001; 3],
        theta_upper: vec![100.0; 3],
        theta_fixed: vec![false; 3],
        omega: omega.clone(),
        omega_fixed: vec![false; 2],
        sigma: sigma.clone(),
        sigma_fixed: vec![false; 2],
        omega_iov: Some(OmegaMatrix::from_diagonal(&[0.02], vec!["KAPPA_CL".into()])),
        kappa_fixed: vec![false],
        mixture: Some(MixtureParams {
            omega: vec![omega, OmegaMatrix::from_diagonal(&[0.3, 0.2], eta)],
            sigma: vec![
                sigma,
                SigmaVector {
                    values: vec![0.05, 0.7],
                    names,
                },
            ],
            omega_override_addr: vec![(1, 0)],
            omega_override_fixed: vec![false],
            sigma_override_addr: vec![(1, 1)],
            sigma_override_fixed: vec![false],
        }),
    }
}

const N_PACKED: usize = 11;

/// A `FitResult` of [`template`], with SEs 0.06 (Ω override, 20 %) and 0.035 (σ, 5 %).
fn result() -> FitResult {
    let t = template();
    let mut r = crate::types::test_helpers::minimal_fit_result();
    r.sigma = t.sigma.values.clone();
    r.sigma_names = t.sigma.names.clone();
    r.sigma_types = vec![SigmaType::Proportional, SigmaType::Additive];
    r.sigma_fixed = vec![false; 2];
    r.sigma_init_as_sd = vec![false; 2];
    r.se_sigma = Some(vec![0.002, 0.03]);
    r.omega_iov = Some(DMatrix::from_element(1, 1, 0.02));
    r.kappa_names = vec!["KAPPA_CL".into()];
    r.kappa_fixed = vec![false];
    r.se_kappa = Some(vec![0.001]);
    r.kappa_is_diagonal = Some(true);
    r.residual_correlations = t.residual_correlations.clone();
    r.residual_correlation_fixed = vec![false];
    r.se_residual_correlations = Some(vec![0.1]);
    r.mixture_overrides = mixture_override_values(t.mixture.as_ref().unwrap());
    r.mixture_override_params = mixture_override_params(&t);
    r.se_mixture_overrides = Some(vec![0.06, 0.035]);
    r.covariance_matrix = Some(DMatrix::identity(N_PACKED, N_PACKED));
    r
}

/// The line of `text` whose first cell is exactly `name`.
fn row<'a>(text: &'a str, name: &str) -> &'a str {
    text.lines()
        .find(|l| l.trim_start().split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no `{name}` row in:\n{text}"))
}

fn yaml_of(r: &FitResult) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fit.yaml");
    write_estimates_yaml(r, path.to_str().unwrap()).expect("yaml write");
    std::fs::read_to_string(&path).expect("yaml read")
}

/// T2 (#2833): the covariance labels name every packed row — the overrides by their
/// descriptors (`log_chol_` for Ω, bare for σ), then ρ — with or without a recorded Ω
/// layout, and each label is `coordinate_names`' entry for that row (with the
/// `log_chol_` prefix on a Cholesky diagonal), so the YAML and the optimizer trace agree.
///
/// Mutations: drop the override push (ρ lands in an override slot, `param_11` pads);
/// push ρ before the overrides (order); leave `packed_layout` counting the trailing rows
/// as Ω/κ (the `None` run reads Ω as a 2-η block: `chol_eta_V_eta_CL`).
#[test]
fn packed_labels_name_overrides_and_rho_for_either_layout_record() {
    let want = [
        "CL",
        "V",
        "KA",
        "log_chol_eta_CL",
        "log_chol_eta_V",
        "prop",
        "add",
        "log_chol_KAPPA_CL",
        "log_chol_eta_CL_MIX2",
        "add_MIX2",
        "add~prop",
    ];
    let coords = coordinate_names(&template());
    assert_eq!(coords.len(), N_PACKED);
    for recorded in [Some(true), None] {
        let mut r = result();
        r.omega_is_diagonal = recorded;
        r.kappa_is_diagonal = recorded;
        let names = packed_param_names(&r, N_PACKED);
        assert_eq!(names, want, "omega_is_diagonal = {recorded:?}");
        assert!(!names.iter().any(|n| n.starts_with("param_")), "{names:?}");
        for (label, coord) in names.iter().zip(&coords) {
            assert!(
                label == coord || *label == format!("log_chol_{coord}"),
                "{label} is not coordinate {coord}"
            );
        }
    }
    // Values without one descriptor each: the override and ρ rows cannot be told apart,
    // so they pad rather than mislabel.
    let mut bare = result();
    bare.mixture_override_params.clear();
    let names = packed_param_names(&bare, N_PACKED);
    assert_eq!(&names[..8], &want[..8]);
    assert_eq!(&names[8..], ["param_9", "param_10", "param_11"]);
}

/// Two Ω overrides on the base `eta_CL` (classes 2 and 3), for [`held_matching_is_by_class`].
fn two_class_result() -> FitResult {
    let mut r = crate::types::test_helpers::minimal_fit_result();
    let param = |class: usize, packed_index: usize| MixtureOverrideParam {
        name: format!("eta_CL_MIX{class}"),
        kind: ParameterKind::Omega,
        index: 0,
        class,
        packed_index,
        fixed: false,
    };
    r.mixture_overrides = Some(vec![0.3, 0.5]);
    r.mixture_override_params = vec![param(2, 6), param(3, 7)];
    r.se_mixture_overrides = Some(vec![0.06, 0.07]);
    r
}

fn held(class: Option<usize>, packed_index: usize) -> HeldAtBound {
    HeldAtBound {
        name: match class {
            Some(c) => format!("eta_CL_MIX{c}"),
            None => "eta_CL".into(),
        },
        kind: ParameterKind::Omega,
        index: 0,
        packed_index,
        reason: HeldReason::CollapsedAtGuard,
        class,
    }
}

/// T3 (#2833): an override is held by a `held_at_bound` entry of its own class, family
/// and base index, and by nothing else. Every arm in one test, both sides of each gate.
///
/// Mutations: match on `(kind, index)` alone (the base entry holds both overrides —
/// case A dies); match on `class.is_some()` (class 3 held by class 2's entry — case B
/// dies); drop the `kind` comparison (a σ entry of the same class/index holds the Ω
/// override — case C dies).
#[test]
fn held_matching_is_by_class() {
    let se_of = |r: &FitResult| {
        let text = format_mixture_override_rows(r);
        (
            row(&text, "eta_CL_MIX2").contains("SE = N/A"),
            row(&text, "eta_CL_MIX3").contains("SE = N/A"),
            se_held(r, ParameterKind::Omega, 0).is_some(),
        )
    };
    // A: base held, both overrides free.
    let mut a = two_class_result();
    a.held_at_bound = vec![held(None, 3)];
    assert_eq!(se_of(&a), (false, false, true), "base held");
    assert!(row(&format_mixture_override_rows(&a), "eta_CL_MIX2").contains("SE = 0.060000"));
    // B: class 2 held; class 3 and the base free.
    let mut b = two_class_result();
    b.held_at_bound = vec![held(Some(2), 6)];
    assert_eq!(se_of(&b), (true, false, false), "class 2 held");
    // B': class 3 held; class 2 and the base free.
    let mut b3 = two_class_result();
    b3.held_at_bound = vec![held(Some(3), 7)];
    assert_eq!(se_of(&b3), (false, true, false), "class 3 held");
    // C: a σ entry with the same class and index holds nothing here.
    let mut c = two_class_result();
    let mut sigma = held(Some(2), 6);
    sigma.kind = ParameterKind::Sigma;
    c.held_at_bound = vec![sigma];
    assert_eq!(se_of(&c), (false, false, false), "σ entry");
}

/// T4 (#2833): the rendered input space — console section, YAML `mixture_overrides:`,
/// `parameter_table` rows — for each override state. Each row type's arm, deleted, kills
/// only its own assertion block.
#[test]
fn rendered_override_rows_cover_the_input_space() {
    // Non-mixture fit, and a mixture fit with no override — both record no
    // `mixture_overrides` and no descriptor: no section, no key, no rows, no empty header.
    let r = crate::types::test_helpers::minimal_fit_result();
    assert!(r.mixture_overrides.is_none() && r.mixture_override_params.is_empty());
    assert!(!format_summary(&r).contains("Mixture per-class"));
    assert_eq!(format_mixture_override_rows(&r), "");
    assert!(!yaml_of(&r).contains("mixture_overrides"));
    assert!(!parameter_table(&r).contains("class"));

    // Free, covariance ran: value, CV% (Ω), SE, %RSE; the section follows SIGMA.
    let free = result();
    let summary = format_summary(&free);
    let at = summary
        .find("--- Mixture per-class overrides ---")
        .expect("section");
    assert!(summary.find("--- SIGMA").unwrap() < at, "{summary}");
    assert_eq!(
        row(&summary, "eta_CL_MIX2"),
        "  eta_CL_MIX2          = 0.300000  (CV% = 59.1)  SE = 0.060000"
    );
    assert_eq!(
        row(&summary, "add_MIX2"),
        "  add_MIX2             = 0.700000  SE = 0.035000"
    );
    let yaml = yaml_of(&free);
    assert!(
        yaml.contains(
            "\nmixture_overrides:\n  eta_CL_MIX2:\n    class: 2\n    base: eta_CL\n    \
             kind: omega\n    variance: 0.300000\n    cv_pct: 59.15\n    se: 0.060000\n  \
             add_MIX2:\n    class: 2\n    base: add\n    kind: sigma\n    \
             estimate: 0.700000\n    se: 0.035000\n"
        ),
        "{yaml}"
    );
    let table = parameter_table(&free);
    let cells = |name: &str| -> Vec<String> {
        row(&table, name)
            .split_whitespace()
            .skip(1)
            .map(String::from)
            .collect()
    };
    assert_eq!(
        cells("eta_CL_MIX2"),
        ["0.300000", "0.060000", "20.0", "OMEGA", "var", "(class", "2)"]
    );
    assert_eq!(
        cells("add_MIX2"),
        ["0.700000", "0.035000", "5.0", "SIGMA", "SD", "(class", "2)"]
    );

    // Free, no covariance: `N/A` (console), `se: ~` (YAML), `---` (table, as base rows);
    // never `0.000000`.
    let mut no_cov = result();
    no_cov.se_mixture_overrides = None;
    let text = format_mixture_override_rows(&no_cov);
    assert!(row(&text, "eta_CL_MIX2").ends_with("SE = N/A"), "{text}");
    assert!(yaml_of(&no_cov).contains("    variance: 0.300000\n    cv_pct: 59.15\n    se: ~\n"));
    let table = parameter_table(&no_cov);
    assert!(row(&table, "eta_CL_MIX2").contains(" --- "), "{table}");
    assert!(!row(&table, "eta_CL_MIX2").contains("0.000000"), "{table}");

    // Fixed: the Ω override by the user's FIX, the σ one as SAEM holds it (#985) — both
    // `fixed` in the descriptor, both `---`, an SE of 0 stored and never printed, and no
    // `held:` key (neither is in `held_at_bound`).
    let mut fixed = result();
    for p in &mut fixed.mixture_override_params {
        p.fixed = true;
    }
    fixed.se_mixture_overrides = Some(vec![0.0, 0.0]);
    let text = format_mixture_override_rows(&fixed);
    assert!(row(&text, "eta_CL_MIX2").ends_with("SE = ---"), "{text}");
    assert!(text.contains("eta_CL_MIX2 [FIX]"), "{text}");
    assert!(text.contains("add_MIX2 [FIX]"), "{text}");
    let yaml = yaml_of(&fixed);
    assert_eq!(
        yaml.matches("    fixed: true\n    se: ~\n").count(),
        2,
        "{yaml}"
    );
    assert!(!yaml.contains("held:"), "{yaml}");
    let table = parameter_table(&fixed);
    for name in ["eta_CL_MIX2", "add_MIX2"] {
        let line = row(&table, name);
        assert!(line.contains("[FIX]") && line.contains(" --- "), "{table}");
        assert!(!line.contains("0.000000"), "{table}");
    }

    // Guard-held (#2809): `N/A`, `se: ~` + `held:`, `N/A` — and the base row keeps its SE.
    let mut held_r = result();
    held_r.held_at_bound = vec![HeldAtBound {
        name: "eta_CL_MIX2".into(),
        kind: ParameterKind::Omega,
        index: 0,
        packed_index: 8,
        reason: HeldReason::CollapsedAtGuard,
        class: Some(2),
    }];
    let summary = format_summary(&held_r);
    assert!(
        row(&summary, "eta_CL_MIX2").ends_with("SE = N/A"),
        "{summary}"
    );
    assert!(
        row(&summary, "eta_CL").ends_with("SE = 0.010000"),
        "{summary}"
    );
    let yaml = yaml_of(&held_r);
    assert!(
        yaml.contains("    cv_pct: 59.15\n    se: ~\n    held: collapsed_at_guard\n"),
        "{yaml}"
    );
    assert_eq!(
        yaml.matches("held: collapsed_at_guard").count(),
        1,
        "{yaml}"
    );
    let table = parameter_table(&held_r);
    assert!(row(&table, "eta_CL_MIX2").contains(" N/A "), "{table}");
    assert!(row(&table, "eta_CL").contains("0.010000"), "{table}");
}

/// T6 (#2833): the `[priors] from_fit` reader reads only the top-level `theta:` /
/// `omega:` / `sigma:` / `omega_iov:` sections, so the overrides — written to their own
/// `mixture_overrides:` section — never come back as a base parameter. Mutation: nest the
/// rows under `omega:` / `sigma:` (both carry the section's value key, so the reader
/// takes them as base η / σ).
#[test]
fn from_fit_reader_ignores_the_override_section() {
    let r = result();
    let yaml = yaml_of(&r);
    assert!(yaml.contains("\nmixture_overrides:\n"), "{yaml}");
    let read = crate::io::fit_estimates::parse_estimates_yaml(&yaml).expect("parse");
    assert!(
        !read.iter().any(|e| e.name.contains("_MIX")),
        "{:?}",
        read.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    let names = |v: &[crate::io::fit_estimates::SourceEstimate]| {
        v.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
    };
    assert_eq!(
        names(&read),
        names(&crate::io::fit_estimates::from_fit_result(&r))
    );
}
