//! Nested variability levels (#1082 PR2): the `level` / `at` grammar, the data checks
//! and the refusals, and the gradient route.
//!
//! The numerical oracles live in `tests/`: the NONMEM anchor and nesting-by-key (O1,
//! O3, `nested_levels_nonmem_anchor.rs`) and the one-level degenerate oracle (O2,
//! `nested_levels_degenerate_oracle.rs`).

use crate::api::validation::{check_model_data, check_model_options, validate_model_file};
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::types::{
    CompiledModel, EstimationMethod, FitOptions, IovOccasionRule, Population, Subject,
};
use std::collections::HashMap;

/// A compartment-free arm-level model: η per sponsor (the subject), κ per study, a
/// weighted κ per arm. `extra` goes at the end of `[parameters]`; `emax` is the EMAX
/// line, so a test can move a κ onto another parameter.
fn src_with(extra: &str, emax: &str, tail: &str) -> String {
    format!(
        "[parameters]\n  theta TVE0(10.0)\n  theta TVEMAX(20.0)\n  omega ETA_SP ~ 0.09\n  \
         level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n  \
         kappa K_ST ~ 0.04 at STUDY\n  kappa K_ARM ~ 4.0 at ARM weight = NARM\n  \
         sigma ADD ~ 0.25\n{extra}[individual_parameters]\n  {emax}\n  \
         E0 = TVE0 * exp(K_ARM * 0)\n[structural_model]\n  y = E0 + EMAX * TIME / (TIME + 3)\n\
         [error_model]\n  DV ~ additive(ADD)\n{tail}"
    )
}

fn src() -> String {
    src_with("", "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)", "")
}

fn model() -> CompiledModel {
    parse_model_string(&src()).expect("parse")
}

/// One record's columns. `ARMID` numbers an arm uniquely across the population
/// (subject, study, arm); `ARM` repeats inside every study.
fn snap_of(subject: usize, study: f64, arm: f64) -> HashMap<String, f64> {
    [
        ("STUDY".to_string(), study),
        ("ARM".to_string(), arm),
        (
            "ARMID".to_string(),
            100.0 * subject as f64 + 10.0 * study + arm,
        ),
        ("STUDYID".to_string(), 10.0 * subject as f64 + study),
        ("NARM".to_string(), 50.0),
    ]
    .into_iter()
    .collect()
}

fn snap(study: f64, arm: f64) -> HashMap<String, f64> {
    snap_of(1, study, arm)
}

/// One subject per entry of `subjects`, each a list of `(STUDY, ARM)` rows.
fn population(subjects: &[&[(f64, f64)]]) -> Population {
    let subjects = subjects
        .iter()
        .enumerate()
        .map(|(i, rows)| Subject {
            id: (i + 1).to_string(),
            obs_times: (0..rows.len()).map(|j| j as f64 + 1.0).collect(),
            observations: vec![15.0; rows.len()],
            obs_cmts: vec![1; rows.len()],
            cens: vec![0; rows.len()],
            covariates: snap_of(i + 1, rows[0].0, rows[0].1),
            obs_covariates: rows.iter().map(|&(s, a)| snap_of(i + 1, s, a)).collect(),
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: ["STUDY", "ARM", "ARMID", "NARM"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Two subjects, two studies each, two arms per study.
fn full_design() -> Population {
    let rows: &[(f64, f64)] = &[(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)];
    population(&[rows, rows])
}

fn code_in(diags: &[crate::diagnostics::Diagnostic], code: &str) -> Option<String> {
    diags
        .iter()
        .find(|d| d.code == code)
        .map(|d| d.message.clone())
}

// ── grammar ──────────────────────────────────────────────────────────────────────

#[test]
fn level_lines_and_at_tails_make_a_nested_model() {
    let m = model();
    assert!(m.has_nested_levels());
    let levels = m.nested_levels().unwrap();
    let names: Vec<(&str, &str, &str)> = levels
        .levels
        .iter()
        .map(|l| (l.name.as_str(), l.column.as_str(), l.parent.as_str()))
        .collect();
    assert_eq!(
        names,
        vec![("STUDY", "STUDY", "ID"), ("ARM", "ARM", "STUDY")]
    );
    assert_eq!(levels.kappa_level, vec![0, 1]);
    // The weight survives the `at` tail.
    assert_eq!(m.kappa_weights.iter().filter(|w| w.is_some()).count(), 1);
    // The level columns are read with the data.
    for c in ["STUDY", "ARM"] {
        assert!(m.referenced_covariates.iter().any(|r| r == c), "{c}");
    }
    // The readers the single-member check compares, in [η, κ] order.
    assert_eq!(levels.readers[0], vec!["EMAX".to_string()]);
    assert_eq!(
        levels.readers[2],
        vec!["E0".to_string(), "EMAX".to_string()]
    );
}

#[test]
fn one_level_is_spelled_onto_iov_column_and_is_not_nested() {
    let one =
        "[parameters]\n  theta TVCL(0.2)\n  omega ETA_CL ~ 0.09\n  level OCC by OCC in ID\n  \
               kappa K ~ 0.01 at OCC\n  sigma P ~ 0.04\n[individual_parameters]\n  \
               CL = TVCL * exp(ETA_CL + K)\n  V = 10\n[structural_model]\n  \
               pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(P)\n";
    let parsed = parse_full_model(one).expect("parse");
    assert!(!parsed.model.has_nested_levels());
    assert_eq!(parsed.fit_options.iov_column.as_deref(), Some("OCC"));
    assert_eq!(parsed.model.n_kappa, 1);
}

#[test]
fn a_level_must_nest_in_the_one_declared_before_it() {
    let bad = src().replace("level ARM by ARM in STUDY", "level ARM by ARM in ID");
    let err = parse_model_string(&bad).err().expect("refused");
    assert!(err.contains("level ARM by ARM in ID"), "{err}");
    assert!(err.contains("must be `in STUDY`"), "{err}");
    assert!(
        err.contains("would be crossed, which is not supported"),
        "{err}"
    );
    let first = src().replace("level STUDY by STUDY in ID", "level STUDY by STUDY in ARM");
    let err = parse_model_string(&first).err().expect("refused");
    assert!(err.contains("must be `in ID`"), "{err}");
}

#[test]
fn a_level_name_is_used_once() {
    let dup = src().replace("level ARM by ARM in STUDY", "level STUDY by ARM in STUDY");
    let err = parse_model_string(&dup).err().expect("refused");
    assert!(
        err.contains("level STUDY: the name is already taken"),
        "{err}"
    );
    let id = src().replace("level STUDY by STUDY in ID", "level ID by STUDY in ID");
    assert!(parse_model_string(&id)
        .err()
        .unwrap()
        .contains("already taken"));
}

#[test]
fn at_names_a_declared_level() {
    let unknown = src().replace("kappa K_ST ~ 0.04 at STUDY", "kappa K_ST ~ 0.04 at SITE");
    let err = parse_model_string(&unknown).err().expect("refused");
    assert!(
        err.contains("`SITE` is not a declared level (declared: STUDY, ARM)"),
        "{err}"
    );
    let none = "[parameters]\n  theta T(1.0)\n  kappa K ~ 0.1 at ARM\n  sigma S ~ 0.1\n\
                [individual_parameters]\n  P = T * exp(K)\n[structural_model]\n  y = P\n\
                [error_model]\n  DV ~ additive(S)\n";
    let err = parse_model_string(none).err().expect("refused");
    assert!(
        err.contains("kappa K at ARM: no level is declared"),
        "{err}"
    );
    assert!(err.contains("`level ARM by COLUMN in ID`"), "{err}");
}

#[test]
fn a_level_takes_no_weight() {
    let w = src().replace(
        "level ARM by ARM in STUDY",
        "level ARM by ARM in STUDY weight = NARM",
    );
    let err = parse_model_string(&w).err().expect("refused");
    assert!(
        err.contains("a level takes no `prior(...)` or `weight =`"),
        "{err}"
    );
}

/// The mixing refusal, from both of its causes, through `ferx check`'s code mapping.
#[test]
fn levels_do_not_mix_with_the_occasion_options_or_an_unattached_kappa() {
    let check = |text: &str| {
        let path = std::env::temp_dir().join(format!(
            "ferx_nested_mixed_{}_{}.ferx",
            std::process::id(),
            text.len()
        ));
        std::fs::write(&path, text).unwrap();
        let report = validate_model_file(path.to_str().unwrap(), None);
        let _ = std::fs::remove_file(&path);
        report
            .diagnostics
            .iter()
            .find(|d| d.code == "E_LEVEL_MIXED_WITH_IOV_COLUMN")
            .map(|d| d.message.clone())
    };
    let column = check(&src_with(
        "",
        "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)",
        "[fit_options]\n  iov_column = OCC\n",
    ))
    .expect("iov_column refused");
    assert!(column.contains("declares `level STUDY`"), "{column}");
    assert!(column.contains("also sets `iov_column`"), "{column}");
    assert!(
        column.contains("A `level` declaration replaces the occasion option"),
        "{column}"
    );
    assert!(column.contains("remove it from [fit_options]"), "{column}");
    let derived = check(&src_with(
        "",
        "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)",
        "[fit_options]\n  iov_occasion = dose\n",
    ))
    .expect("iov_occasion refused");
    assert!(derived.contains("also sets `iov_occasion`"), "{derived}");
    let unattached = check(&src().replace("kappa K_ST ~ 0.04 at STUDY", "kappa K_ST ~ 0.04"))
        .expect("an unattached kappa refused");
    assert!(
        unattached.contains("kappa K_ST has no `at LEVEL`"),
        "{unattached}"
    );
    assert!(
        unattached
            .contains("Once a `level` is declared, every kappa says which level it lives at."),
        "{unattached}"
    );
    // The model itself is accepted.
    assert!(check(&src()).is_none());
}

// ── data checks ──────────────────────────────────────────────────────────────────

#[test]
fn a_record_without_a_level_label_is_refused() {
    let m = model();
    let mut pop = full_design();
    assert!(code_in(&check_model_data(&m, &pop), "E_LEVEL_MISSING_LABEL").is_none());
    pop.subjects[1].obs_covariates[2].insert("ARM".to_string(), f64::NAN);
    let msg = code_in(&check_model_data(&m, &pop), "E_LEVEL_MISSING_LABEL").expect("refused");
    assert!(
        msg.contains(
            "level `ARM` reads column `ARM`, which has no numeric value on the observation \
                      record of subject `2` at TIME 3"
        ),
        "{msg}"
    );
    assert!(
        msg.contains("Every record needs a label at every level"),
        "{msg}"
    );
    assert!(
        msg.contains("a label is a numeric code: write a text id such as `NCT01` as a number."),
        "{msg}"
    );
    // A pk-only row names its kind.
    let mut pop = full_design();
    pop.subjects[0].pk_only_times = vec![0.5];
    pop.subjects[0].pk_only_covariates = vec![snap(1.0, 1.0)];
    pop.subjects[0].pk_only_covariates[0].remove("STUDY");
    let msg = code_in(&check_model_data(&m, &pop), "E_LEVEL_MISSING_LABEL").expect("refused");
    assert!(msg.contains("level `STUDY` reads column `STUDY`"), "{msg}");
    assert!(
        msg.contains("non-observation (EVID=2 or MDV=1) record of subject `1`"),
        "{msg}"
    );
}

#[test]
fn the_occasion_column_check_does_not_fire_on_a_nested_model() {
    // A nested population carries no occasion labels until it is prepared for a fit.
    let diags = check_model_data(&model(), &full_design());
    assert!(code_in(&diags, "E_IOV_MISSING_OCC").is_none());
}

/// Both sides of the single-member gate, in one test: one arm in every study is refused
/// while κ_ARM and κ_ST read the same parameters, and accepted once either half of the
/// condition is false.
#[test]
fn a_level_with_one_member_per_parent_is_refused_only_when_its_kappa_has_a_twin() {
    let single_arm: &[(f64, f64)] = &[(1.0, 1.0), (1.0, 1.0), (2.0, 1.0)];
    let one_study_two_arms: &[(f64, f64)] = &[(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)];
    // κ_ARM reads E0 too; put K_ST on both so the readers match.
    let twin = parse_model_string(&src_with(
        "",
        "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)\n  E0 = TVE0 * exp(K_ST * 0)",
        "",
    ))
    .unwrap();
    let levels = twin.nested_levels().unwrap();
    assert_eq!(
        levels.readers[1], levels.readers[2],
        "the fixture's κs are twins"
    );
    let msg = code_in(
        &check_model_data(&twin, &population(&[single_arm, single_arm])),
        "E_LEVEL_SINGLE_MEMBER",
    )
    .expect("refused");
    assert!(
        msg.contains("level `ARM` has a single group inside every `STUDY` group"),
        "{msg}"
    );
    assert!(
        msg.contains("kappa `K_ARM` moves the same records as `K_ST` does"),
        "{msg}"
    );
    assert!(
        msg.contains("through the same parameters (E0, EMAX)"),
        "{msg}"
    );
    assert!(
        msg.contains("Drop one of the two random effects, or drop the level."),
        "{msg}"
    );
    // Some study with two arms: identified.
    assert!(code_in(
        &check_model_data(&twin, &population(&[single_arm, one_study_two_arms])),
        "E_LEVEL_SINGLE_MEMBER"
    )
    .is_none());
    // Different readers: identified through the parameters themselves.
    let apart = model();
    let l = apart.nested_levels().unwrap();
    assert_ne!(l.readers[1], l.readers[2]);
    assert!(code_in(
        &check_model_data(&apart, &population(&[single_arm, single_arm])),
        "E_LEVEL_SINGLE_MEMBER"
    )
    .is_none());
}

#[test]
fn a_single_study_per_subject_twins_the_study_kappa_with_the_subject_eta() {
    let m = parse_model_string(&src_with(
        "",
        "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)",
        "",
    ))
    .unwrap();
    let one_study: &[(f64, f64)] = &[(1.0, 1.0), (1.0, 2.0)];
    let msg = code_in(
        &check_model_data(&m, &population(&[one_study, one_study])),
        "E_LEVEL_SINGLE_MEMBER",
    )
    .expect("refused");
    assert!(
        msg.contains("level `STUDY` has a single group inside every subject"),
        "{msg}"
    );
    assert!(
        msg.contains("kappa `K_ST` moves the same records as `ETA_SP` does"),
        "{msg}"
    );
    // Two studies in one subject: identified.
    let two: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 1.0)];
    assert!(code_in(
        &check_model_data(&m, &population(&[one_study, two])),
        "E_LEVEL_SINGLE_MEMBER"
    )
    .is_none());
}

/// A θ level block whose levels each sit inside one arm could absorb κ_ARM, which the
/// binder cannot yet certify under nested levels; one keyed on `TIME` crosses the arms
/// and is left to the binder as before.
#[test]
fn a_theta_block_nested_in_a_kappa_level_is_refused_and_one_crossing_it_is_not() {
    let with_block = |cols: &str| {
        parse_model_string(&src_with(
            &format!("  theta PL[{cols}](0.0, -50.0, 50.0)\n"),
            "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM) + PL",
            "",
        ))
        .unwrap()
    };
    let nested = with_block("ARMID");
    let msg = code_in(
        &check_model_data(&nested, &full_design()),
        "E_NESTED_LEVEL_THETA_BLOCK_UNSUPPORTED",
    )
    .expect("refused");
    assert!(
        msg.contains("theta PL[ARMID]: each of its levels lies within one `ARM` group"),
        "{msg}"
    );
    assert!(msg.contains("so it could absorb a kappa at `ARM`"), "{msg}");
    assert!(
        msg.contains("refused rather than fitted unchecked"),
        "{msg}"
    );
    assert!(
        msg.contains("Key the block on columns whose levels cross the `ARM` groups"),
        "{msg}"
    );
    // A block nested in the studies but crossing their arms names the study level.
    let by_study = with_block("STUDYID");
    let msg = code_in(
        &check_model_data(&by_study, &full_design()),
        "E_NESTED_LEVEL_THETA_BLOCK_UNSUPPORTED",
    )
    .expect("refused");
    assert!(
        msg.contains("theta PL[STUDYID]: each of its levels lies within one `STUDY` group"),
        "{msg}"
    );
    assert!(msg.contains("could absorb a kappa at `STUDY`"), "{msg}");
    let crossing = with_block("TIME");
    assert!(code_in(
        &check_model_data(&crossing, &full_design()),
        "E_NESTED_LEVEL_THETA_BLOCK_UNSUPPORTED"
    )
    .is_none());
}

// ── refusals ─────────────────────────────────────────────────────────────────────

fn options(method: EstimationMethod) -> FitOptions {
    FitOptions {
        method,
        ..Default::default()
    }
}

#[test]
fn only_foce_and_focei_fit_a_nested_model() {
    let m = model();
    for ok in [EstimationMethod::Foce, EstimationMethod::FoceI] {
        assert!(
            code_in(
                &check_model_options(&m, &options(ok)),
                "E_NESTED_LEVEL_METHOD_UNSUPPORTED"
            )
            .is_none(),
            "{ok:?}"
        );
    }
    for (bad, label) in [
        (EstimationMethod::Saem, "method = saem"),
        (EstimationMethod::Imp, "method = imp"),
        (EstimationMethod::Laplace, "method = laplace"),
        (EstimationMethod::FoceGn, "method = foce-gn"),
        (EstimationMethod::Vi, "method = vi"),
        (EstimationMethod::Bayes, "method = bayes"),
    ] {
        let msg = code_in(
            &check_model_options(&m, &options(bad)),
            "E_NESTED_LEVEL_METHOD_UNSUPPORTED",
        )
        .unwrap_or_else(|| panic!("{bad:?} not refused"));
        assert!(
            msg.starts_with(&format!("{label} is not yet supported")),
            "{msg}"
        );
        assert!(
            msg.contains("on a model with nested levels (more than one `level`)"),
            "{msg}"
        );
        assert!(
            msg.contains("Fit it with method = focei (or foce) at n_agq = 1."),
            "{msg}"
        );
    }
    let agq = FitOptions {
        n_agq: 3,
        ..options(EstimationMethod::FoceI)
    };
    let msg = code_in(
        &check_model_options(&m, &agq),
        "E_NESTED_LEVEL_METHOD_UNSUPPORTED",
    )
    .unwrap();
    assert!(msg.starts_with("method = focei with n_agq = 3"), "{msg}");
    // The same method on a one-level model is not this refusal.
    let one = parse_model_string(&src().replace("  level ARM by ARM in STUDY\n", "").replace(
        "kappa K_ARM ~ 4.0 at ARM weight = NARM",
        "kappa K_ARM ~ 4.0 at STUDY weight = NARM",
    ))
    .unwrap();
    assert!(!one.has_nested_levels());
    assert!(code_in(
        &check_model_options(&one, &options(EstimationMethod::Saem)),
        "E_NESTED_LEVEL_METHOD_UNSUPPORTED"
    )
    .is_none());
}

#[test]
fn a_residual_correlation_is_refused_on_a_nested_model() {
    let m = parse_model_string(
        &src_with(
            "  theta RHO(0.3, -0.9, 0.9)\n",
            "EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)",
            "",
        )
        .replace(
            "DV ~ additive(ADD)\n",
            "DV ~ additive(ADD)\n  correlation cs(RHO) by ARM\n",
        ),
    )
    .expect("parse");
    let msg = code_in(
        &check_model_options(&m, &options(EstimationMethod::FoceI)),
        "E_NESTED_LEVEL_CORRELATION_UNSUPPORTED",
    )
    .expect("refused");
    assert!(
        msg.contains("a residual `correlation … by` line is not yet supported"),
        "{msg}"
    );
    assert!(
        msg.contains("Fit the levels without it, or the correlation with a single level."),
        "{msg}"
    );
    assert!(code_in(
        &check_model_options(&model(), &options(EstimationMethod::FoceI)),
        "E_NESTED_LEVEL_CORRELATION_UNSUPPORTED"
    )
    .is_none());
}

#[test]
fn simulate_refuses_a_nested_model() {
    let m = model();
    let err = crate::api::simulate(&m, &full_design(), &m.default_params, 1)
        .err()
        .expect("refused");
    assert_eq!(err.code(), Some("E_NESTED_LEVEL_SIMULATE_UNSUPPORTED"));
    assert!(err
        .to_string()
        .contains("simulation does not yet support nested levels"));
    assert!(err
        .to_string()
        .contains("a kappa at an outer level must be drawn once per group"));
    assert!(err
        .to_string()
        .contains("Fit the model; simulate from it in a later release."));
}

#[test]
fn fit_refuses_a_nested_model_under_saem_with_its_code() {
    let m = model();
    let opts = FitOptions {
        method: EstimationMethod::Saem,
        iov_occasion: IovOccasionRule::Column,
        ..Default::default()
    };
    let err = crate::api::fit(&m, &full_design(), &m.default_params, &opts)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        err.contains("method = saem is not yet supported on a model with nested levels"),
        "{err}"
    );
}

// ── route ────────────────────────────────────────────────────────────────────────

/// #1082 PR3: each parity fixture (`stats::nested_levels::test_helpers` — one per analytic
/// walk) fitted end to end reports the analytic route on both loops and no FD-fallback
/// warning: the engine a fixture runs on is read off the fit, not assumed from the model
/// class. A subject that dropped to FD would name itself in `iov_fd_fallback_warning`.
#[test]
fn every_nested_parity_fixture_fits_on_the_analytic_walks() {
    use crate::build_info::GradientMethodKind;
    use crate::stats::nested_levels::test_helpers as fx;
    let analytic = GradientMethodKind::Analytic.as_str();
    for (label, src) in [
        ("algebraic", fx::algebraic()),
        ("closed form", fx::closed_form()),
        ("ode", fx::ode()),
    ] {
        let (m, pop) = fx::prepared(&src, 2);
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            iov_occasion: IovOccasionRule::Column,
            outer_maxiter: 1,
            run_covariance_step: false,
            ..Default::default()
        };
        let r = crate::api::fit(&m, &pop, &m.default_params, &opts)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(r.ofv.is_finite(), "{label}");
        assert_eq!(r.gradient_method_inner, analytic, "{label}");
        assert_eq!(r.gradient_method_outer, analytic, "{label}");
        let fd: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("finite-difference"))
            .collect();
        assert!(fd.is_empty(), "{label}: {fd:?}");
    }
}

/// The closed-form transit model with a κ on each of two levels. Its IOV subjects run on
/// the ODE twin, which is re-parsed from emitted source.
const TRANSIT_NESTED: &str = "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  \
    theta TVV(50.0, 5.0, 500.0)\n  theta TVMTT(3.0, 0.05, 24.0)\n  theta TVN(3.0, 0.0, 30.0)\n  \
    omega ETA_V ~ 0.09\n  level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n  \
    kappa K_ST ~ 0.04 at STUDY\n  kappa K_ARM ~ 0.09 at ARM\n  sigma PROP ~ 0.15 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(K_ST)\n  V = TVV * exp(ETA_V)\n  \
    MTT = TVMTT * exp(K_ARM)\n  NTR = TVN\n[structural_model]\n  \
    pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n[error_model]\n  \
    DV ~ proportional(PROP)\n";

/// #1082 PR3: a nested model reports the analytic walks on every gradient, as its
/// one-level twin does — the walks seed κ on the compact axes. The one nested model left
/// on finite differences is a closed-form transit/IG one, whose IOV subjects run on an ODE
/// twin not known to carry the levels; its one-level twin stays analytic, so the decline
/// follows the nesting and not the model class.
#[test]
fn a_nested_model_reports_the_analytic_walks_unless_it_runs_on_a_transit_twin() {
    let one_level = |src: &str, inner: &str, outer: &str| {
        parse_model_string(
            &src.replace(&format!("  level {inner} by {inner} in {outer}\n"), "")
                .replace(&format!("at {inner}"), &format!("at {outer}")),
        )
        .unwrap()
    };
    let nested = model();
    let one = one_level(&src(), "ARM", "STUDY");
    let transit = parse_model_string(TRANSIT_NESTED).unwrap();
    let transit_one = one_level(TRANSIT_NESTED, "ARM", "STUDY");
    assert!(
        transit.absorption_ode_equivalent.is_some(),
        "the transit fixture must carry an ODE twin, or it cannot see the decline"
    );
    for (label, m, nested_levels, analytic) in [
        ("algebraic, nested", &nested, true, true),
        ("algebraic, one level", &one, false, true),
        ("transit, nested", &transit, true, false),
        ("transit, one level", &transit_one, false, true),
    ] {
        assert_eq!(m.has_nested_levels(), nested_levels, "{label}");
        assert_eq!(
            crate::sens::provider::nested_levels_outside_analytic_scope(m),
            !analytic,
            "{label}"
        );
        assert_eq!(
            crate::sens::provider::iov_sens_supported(m),
            analytic,
            "{label}"
        );
        assert_eq!(
            crate::sens::provider::iov_sens_eta_supported(m),
            analytic,
            "{label}"
        );
        assert_eq!(
            crate::estimation::inner_optimizer::inner_reports_analytic_model(m),
            analytic,
            "{label}"
        );
        assert_eq!(
            crate::sens::provider::analytic_outer_gradient_in_scope(m),
            analytic,
            "{label}"
        );
    }
}

/// The weight-variation warning groups by the κ's own level: a weight that changes
/// inside an arm warns for the arm kappa, and one that changes only between the arms of
/// a study does not — while it does for a weighted kappa at the study.
#[test]
fn a_weight_varying_inside_its_kappas_group_warns_per_level() {
    let warn = |m: &CompiledModel, pop: &Population| {
        crate::api::validation::check_model_data_warnings(m, pop, &m.default_params)
            .into_iter()
            .find(|d| d.code == "W_KAPPA_WEIGHT_VARIES_WITHIN_OCCASION")
            .map(|d| d.message)
    };
    let rows: &[(f64, f64)] = &[(1.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0)];
    let m = model();
    // NARM differs between the arms of study 1, not inside an arm.
    let mut between = population(&[rows]);
    between.subjects[0].obs_covariates[2].insert("NARM".to_string(), 25.0);
    assert!(warn(&m, &between).is_none());
    // ... and inside study 1's arm 1.
    let mut inside = population(&[rows]);
    inside.subjects[0].obs_covariates[1].insert("NARM".to_string(), 25.0);
    let msg = warn(&m, &inside).expect("warned");
    assert!(
        msg.contains(
            "weight `NARM` is not constant within group `STUDY 1 / ARM 1` of subject `1` (50 then 25)"
        ),
        "{msg}"
    );
    assert!(
        msg.contains("κ is drawn once per group of its level, so the group's"),
        "{msg}"
    );
    // A weighted kappa at the study sees the between-arm change.
    let at_study = parse_model_string(&src().replace(
        "kappa K_ST ~ 0.04 at STUDY",
        "kappa K_ST ~ 0.04 at STUDY weight = NARM",
    ))
    .unwrap();
    let msg = warn(&at_study, &between).expect("warned");
    assert!(
        msg.contains("kappa `K_ST` weight `NARM` is not constant within group `STUDY 1` of"),
        "{msg}"
    );
}

/// A blank level cell, read from a CSV, is a missing label: the level columns are not
/// carried forward like covariates (`ObsRouting::no_carry_forward`), so the record is
/// refused at its own row instead of silently joining the group of the record before
/// it. An ordinary covariate's blank cell on the same row still carries forward — the
/// two sides of the gate in one read.
#[test]
fn a_blank_level_cell_is_a_missing_label_and_a_blank_covariate_still_carries_forward() {
    let m = model();
    let read = |csv: &str| {
        let path = std::env::temp_dir().join(format!(
            "ferx_nested_blank_{}_{}.csv",
            std::process::id(),
            csv.len()
        ));
        std::fs::write(&path, csv).unwrap();
        let (pop, _) = crate::api::read_population_for(
            &m,
            &None,
            path.to_str().unwrap(),
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        let _ = std::fs::remove_file(&path);
        pop
    };
    // TIME 3 has a blank ARM and a blank NARM.
    let csv = "ID,TIME,DV,STUDY,ARM,NARM\n1,1,10,1,1,50\n1,2,11,1,2,40\n1,3,12,1,,\n\
               1,4,13,2,1,30\n";
    let pop = read(csv);
    let s = &pop.subjects[0];
    assert!(s.obs_cov(2)["ARM"].is_nan(), "the blank label is missing");
    assert_eq!(
        s.obs_cov(2)["NARM"],
        40.0,
        "the blank covariate carries forward"
    );
    let msg = code_in(&check_model_data(&m, &pop), "E_LEVEL_MISSING_LABEL").expect("refused");
    assert!(
        msg.contains(
            "level `ARM` reads column `ARM`, which has no numeric value on the observation \
                      record of subject `1` at TIME 3"
        ),
        "{msg}"
    );
    // A label blank on the first row is missing too, not the first value below it.
    let first = read("ID,TIME,DV,STUDY,ARM,NARM\n1,1,10,1,,50\n1,2,11,1,2,50\n");
    assert!(first.subjects[0].obs_cov(0)["ARM"].is_nan());
    // No ARM value on any record of a subject: refused as well.
    let none = read("ID,TIME,DV,STUDY,ARM,NARM\n1,1,10,1,1,50\n2,1,10,1,,50\n2,2,11,1,,50\n");
    let msg = code_in(&check_model_data(&m, &none), "E_LEVEL_MISSING_LABEL").expect("refused");
    assert!(msg.contains("subject `2`"), "{msg}");
    // Without blanks every record keeps its own label.
    let full = read("ID,TIME,DV,STUDY,ARM,NARM\n1,1,10,1,1,50\n1,2,11,1,2,50\n");
    assert!(code_in(&check_model_data(&m, &full), "E_LEVEL_MISSING_LABEL").is_none());
}

/// Review r1 #1: a fit-time occasion rule on a nested model. `[fit_options]` is refused at
/// parse time, but `FitOptions::iov_occasion` (and R's `settings`) reach `fit()` directly.
/// Both sides of the gate in one test: `Column` (the default) passes, a derived rule is
/// refused under the mixing code.
#[test]
fn a_fit_time_occasion_rule_is_refused_on_a_nested_model() {
    let m = model();
    let with = |rule: IovOccasionRule| FitOptions {
        method: EstimationMethod::FoceI,
        iov_occasion: rule,
        ..Default::default()
    };
    assert!(code_in(
        &check_model_options(&m, &with(IovOccasionRule::Column)),
        "E_LEVEL_MIXED_WITH_IOV_COLUMN"
    )
    .is_none());
    for rule in [
        IovOccasionRule::PerDose,
        IovOccasionRule::TimeWindows(vec![2.5]),
    ] {
        let msg = code_in(
            &check_model_options(&m, &with(rule.clone())),
            "E_LEVEL_MIXED_WITH_IOV_COLUMN",
        )
        .unwrap_or_else(|| panic!("{rule:?} not refused"));
        assert!(msg.contains("the fit options set `iov_occasion`"), "{msg}");
        assert!(
            msg.contains("on a model that declares `level STUDY`"),
            "{msg}"
        );
        assert!(
            msg.contains("The levels already define the groups"),
            "{msg}"
        );
        assert!(
            msg.contains("leave `iov_occasion` at its default (`column`)"),
            "{msg}"
        );
        let err = crate::api::fit(&m, &full_design(), &m.default_params, &with(rule))
            .err()
            .expect("fit refuses it")
            .to_string();
        assert!(err.contains("the fit options set `iov_occasion`"), "{err}");
    }
    // A one-level model keeps the rule.
    let one = parse_model_string(&src().replace("  level ARM by ARM in STUDY\n", "").replace(
        "kappa K_ARM ~ 4.0 at ARM weight = NARM",
        "kappa K_ARM ~ 4.0 at STUDY weight = NARM",
    ))
    .unwrap();
    assert!(code_in(
        &check_model_options(&one, &with(IovOccasionRule::PerDose)),
        "E_LEVEL_MIXED_WITH_IOV_COLUMN"
    )
    .is_none());
}

/// The post-hoc steps (`run_covariance`, `run_sir`) prepare the population with the rule
/// the fit recorded, through `fitted_population`, without `check_model_options`. On a
/// nested model the level cells must win there whatever the rule says, so a regrouping
/// cannot get in by that door either.
#[test]
fn fitted_population_keeps_the_level_cells_under_a_derived_rule() {
    let m = model();
    let pop = full_design();
    let mut warnings = Vec::new();
    for rule in [
        IovOccasionRule::Column,
        IovOccasionRule::TimeWindows(vec![2.5]),
    ] {
        let prepared = crate::api::fit::fitted_population(
            &m,
            &rule,
            false,
            std::borrow::Cow::Borrowed(&pop),
            &mut warnings,
        );
        for s in &prepared.subjects {
            assert_eq!(s.occasions, vec![1, 2, 3, 4], "{rule:?}");
        }
    }
}

/// Review r1 #2: a text study id is not a label the engine can read. It reaches the data
/// check as a missing numeric value, and the message says a label is a numeric code
/// rather than claiming the cell is empty.
#[test]
fn a_text_level_label_is_refused_as_a_non_numeric_value() {
    let m = model();
    let path = std::env::temp_dir().join(format!("ferx_nested_text_{}.csv", std::process::id()));
    std::fs::write(
        &path,
        "ID,TIME,DV,STUDY,ARM,NARM\n1,1,10,NCT01,1,50\n1,2,11,NCT01,2,50\n1,3,12,NCT02,1,50\n",
    )
    .unwrap();
    let (pop, _) =
        crate::api::read_population_for(&m, &None, path.to_str().unwrap(), None, None, None, &[])
            .unwrap();
    let _ = std::fs::remove_file(&path);
    let msg = code_in(&check_model_data(&m, &pop), "E_LEVEL_MISSING_LABEL").expect("refused");
    assert!(
        msg.contains("level `STUDY` reads column `STUDY`, which has no numeric value"),
        "{msg}"
    );
    assert!(
        msg.contains("write a text id such as `NCT01` as a number"),
        "{msg}"
    );
}

/// Review r1 #8: `at LEVEL` names a level as `in PARENT` and the uniqueness rule do,
/// ignoring ASCII case, and resolves to that level.
#[test]
fn at_names_a_level_ignoring_case_like_the_other_level_rules() {
    let m = parse_model_string(
        &src()
            .replace("level STUDY by STUDY in ID", "level Study by STUDY in ID")
            .replace("level ARM by ARM in STUDY", "level ARM by ARM in Study"),
    )
    .expect("`at STUDY` names `level Study`");
    assert_eq!(m.nested_levels().unwrap().kappa_level, vec![0, 1]);
}

/// Review r1 #6: a nested model handed a population that never went through
/// `fitted_population` (the lower-level entries `optimize_population` / `run_inner_loop`,
/// or an in-memory population) has no cells. Each estimator site refuses such a subject
/// with its sentinel instead of falling back to one-level IOV or dropping κ — and the
/// same subject, once prepared, is estimated (the straddle, per site).
#[test]
fn an_unprepared_nested_subject_is_refused_by_every_estimator_site() {
    use crate::estimation::inner_optimizer::find_ebe;
    use crate::stats::likelihood::{foce_subject_nll_iov, individual_nll_iov};
    let m = model();
    let raw = full_design();
    let mut prepared = raw.clone();
    crate::stats::nested_levels::assign_level_cells(&m, &mut prepared);
    let p = &m.default_params;
    let iov = p.omega_iov.as_ref().unwrap();
    let eta = nalgebra::DVector::zeros(m.n_eta);
    let h = |s: &Subject| nalgebra::DMatrix::zeros(s.obs_times.len(), m.n_eta);
    let cells =
        |s: &Subject| vec![vec![0.0; m.n_kappa]; crate::stats::stacked_layout::group_count(s)];
    let dcells = |s: &Subject| {
        cells(s)
            .into_iter()
            .map(nalgebra::DVector::from_vec)
            .collect::<Vec<_>>()
    };
    for (s_raw, s_ok) in raw.subjects.iter().zip(&prepared.subjects) {
        assert!(crate::stats::nested_levels::unprepared(&m, s_raw));
        assert!(!crate::stats::nested_levels::unprepared(&m, s_ok));

        let r = find_ebe(&m, s_raw, p, 50, 1e-6, None, None, 0);
        assert!(
            r.hard_reject && r.nll >= 1e20,
            "find_ebe took the subject: {}",
            r.nll
        );
        let r = find_ebe(&m, s_ok, p, 50, 1e-6, None, None, 0);
        assert!(
            !r.hard_reject && r.nll.is_finite() && r.nll < 1e10,
            "{}",
            r.nll
        );
        assert_eq!(r.kappas.len(), 4, "one κ̂ vector per cell");

        let f = |s: &Subject| {
            foce_subject_nll_iov(
                &m,
                s,
                &p.theta,
                &eta,
                &h(s),
                &p.omega,
                &p.sigma.values,
                true,
                &dcells(s),
                iov,
            )
        };
        assert_eq!(f(s_raw), 1e20);
        assert!(f(s_ok) < 1e10);

        let lin = |s: &Subject| {
            crate::stats::likelihood::iov_linearization(
                &m,
                s,
                &p.theta,
                &eta,
                &h(s),
                &p.omega,
                &dcells(s),
                iov,
            )
        };
        assert!(
            lin(s_raw).is_none(),
            "the linearization refuses the unprepared subject"
        );
        assert!(lin(s_ok).is_some());

        let i = |s: &Subject| {
            individual_nll_iov(
                &m,
                s,
                &p.theta,
                eta.as_slice(),
                &cells(s),
                &p.omega,
                Some(iov),
                &p.sigma.values,
            )
        };
        assert_eq!(i(s_raw), 1e20);
        assert!(i(s_ok) < 1e10);
    }
}
