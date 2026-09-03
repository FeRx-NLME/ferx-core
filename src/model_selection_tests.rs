use super::*;
use crate::estimation::parameterization::packed_held_mask;
use crate::parser::model_parser::parse_model_string;
use crate::types::test_helpers::minimal_fit_result;
use crate::types::{WarningEntry, WarningSeverity};
use nalgebra::DMatrix;

// ── BIC variants ────────────────────────────────────────────────────────────

/// Pharmpy's `pheno` example (`tests/testdata/nonmem/pheno_real.mod`): 59
/// subjects, 155 observations, three θ (all reaching an η-bearing parameter:
/// `TVCL = THETA(1)*WGT`, `TVV = THETA(2)*WGT*(1+THETA(3))`), two diagonal Ω,
/// one σ. OFV from `pheno_real.ext`; the four BIC values are the
/// `calculate_bic` docstring's own outputs on that fit.
fn pheno_like() -> FitResult {
    let mut r = minimal_fit_result();
    r.ofv = 586.27605628188053;
    r.n_subjects = 59;
    r.n_obs = 155;
    r.n_parameters = 6;
    r.bic_inputs = BicInputs {
        n_obs: 155,
        theta_random: 3,
        theta_fixed: 0,
        omega: 2,
        kappa: 0,
        sigma: 1,
        sigma_random: false,
    };
    r
}

#[test]
fn bic_variants_match_pharmpy_calculate_bic_on_pheno() {
    let r = pheno_like();
    let pharmpy = [
        (BicType::Mixed, 611.7071686216575),
        (BicType::Fixed, 616.5366069867251),
        (BicType::Random, 610.741280948644),
        (BicType::Iiv, 594.4311311730211),
    ];
    // All four sit a constant 3.3e-9 above `OFV + penalty` on the `.ext` OFV —
    // Pharmpy reads its OFV from a differently-rounded source — so the
    // penalties are anchored through the *differences* between variants,
    // which cancel that offset, and the absolute values to 1e-8.
    for (kind, want) in pharmpy {
        let got = bic(&r, kind);
        assert!((got - want).abs() < 1e-8, "{kind:?}: {got} vs {want}");
    }
    let offset = bic(&r, BicType::Fixed) - 616.5366069867251;
    for (kind, want) in pharmpy {
        let got = bic(&r, kind) - offset;
        assert!((got - want).abs() < 1e-12, "{kind:?}: {got} vs {want}");
    }
}

#[test]
fn bic_variants_match_hand_computed_penalties() {
    // ofv 100; 20 subjects, 200 obs; θ 2 random + 1 fixed-class, Ω 3 (block),
    // κ 1, σ 2 with iiv_on_ruv → σ joins the random class.
    let mut r = minimal_fit_result();
    r.ofv = 100.0;
    r.n_subjects = 20;
    r.n_parameters = 9;
    r.bic_inputs = BicInputs {
        n_obs: 200,
        theta_random: 2,
        theta_fixed: 1,
        omega: 3,
        kappa: 1,
        sigma: 2,
        sigma_random: true,
    };
    let ls = 20f64.ln();
    let lo = 200f64.ln();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(bic(&r, BicType::Mixed), 100.0 + 8.0 * ls + 1.0 * lo));
    assert!(close(bic(&r, BicType::Iiv), 100.0 + 3.0 * ls));
    assert!(close(bic(&r, BicType::Random), 100.0 + 9.0 * ls));
    assert!(close(bic(&r, BicType::Fixed), 100.0 + 9.0 * lo));

    // Without the RUV eta, σ moves back to the ln(n_obs) class.
    r.bic_inputs.sigma_random = false;
    assert!(close(bic(&r, BicType::Mixed), 100.0 + 6.0 * ls + 3.0 * lo));
}

#[test]
fn bic_fixed_reproduces_fit_result_bic_bit_for_bit() {
    // The same expression `fit()` evaluates, so `Fixed` is not merely close.
    let mut r = pheno_like();
    r.bic = r.ofv + r.n_parameters as f64 * (r.bic_inputs.n_obs as f64).ln();
    assert_eq!(bic(&r, BicType::Fixed).to_bits(), r.bic.to_bits());
}

#[test]
fn bic_is_nan_when_the_tally_is_missing_or_degenerate() {
    // A pre-#1177 bundle: all-zero tally against a non-zero parameter count.
    let mut r = pheno_like();
    r.bic_inputs = BicInputs::default();
    for kind in [
        BicType::Mixed,
        BicType::Iiv,
        BicType::Random,
        BicType::Fixed,
    ] {
        assert!(
            bic(&r, kind).is_nan(),
            "{kind:?} must be NaN without a tally"
        );
    }
    // A tally that does not partition n_parameters is refused too.
    let mut r = pheno_like();
    r.n_parameters = 7;
    assert!(bic(&r, BicType::Mixed).is_nan());
    // Zero subjects cannot take a log …
    let mut r = pheno_like();
    r.n_subjects = 0;
    assert!(bic(&r, BicType::Random).is_nan());
    assert!(bic(&r, BicType::Mixed).is_nan());
    // … but a convention that never needs that log still works.
    assert!(bic(&r, BicType::Fixed).is_finite());
}

#[test]
fn bic_mixed_needs_only_the_logs_its_classes_use() {
    // Every free parameter random-class and n_obs = 0 (a TTE-only fit whose
    // record count went unrecorded): ln(n_obs) is never taken.
    let mut r = minimal_fit_result();
    r.ofv = 50.0;
    r.n_subjects = 10;
    r.n_parameters = 2;
    r.bic_inputs = BicInputs {
        n_obs: 0,
        theta_random: 1,
        theta_fixed: 0,
        omega: 1,
        kappa: 0,
        sigma: 0,
        sigma_random: false,
    };
    assert!((bic(&r, BicType::Mixed) - (50.0 + 2.0 * 10f64.ln())).abs() < 1e-12);
    assert!(bic(&r, BicType::Fixed).is_nan());
}

#[test]
fn bic_type_serde_uses_snake_case_tokens() {
    assert_eq!(serde_json::to_string(&BicType::Mixed).unwrap(), "\"mixed\"");
    assert_eq!(
        serde_json::from_str::<BicType>("\"iiv\"").unwrap(),
        BicType::Iiv
    );
    assert_eq!(BicType::default(), BicType::Mixed);
}

// ── bic_inputs_for: the tally fit() records ─────────────────────────────────

const WARFARIN_TAIL: &str = r#"
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn model_free_dim(model_text: &str) -> usize {
    parse_model_string(model_text)
        .expect("model parses")
        .free_packed_dim()
}

fn tally(model_text: &str, n_obs: usize) -> BicInputs {
    let model = parse_model_string(model_text).expect("model parses");
    let template = &model.default_params;
    let mask = packed_held_mask(template);
    let out = bic_inputs_for(&model, template, &mask, n_obs);
    assert_eq!(
        out.n_free(),
        mask.iter().filter(|&&f| !f).count(),
        "tally must partition the free packed parameters"
    );
    out
}

#[test]
fn tally_splits_theta_by_eta_linkage_and_honours_fix() {
    let text = format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVF(0.8, 0.1, 1.0) FIX
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * TVF
{WARFARIN_TAIL}"#
    );
    let t = tally(&text, 123);
    assert_eq!(
        t,
        BicInputs {
            n_obs: 123,
            theta_random: 2,
            theta_fixed: 1, // TVKA: no eta reaches KA; TVF is FIX and uncounted
            omega: 2,
            kappa: 0,
            sigma: 1,
            sigma_random: false,
        }
    );
}

#[test]
fn tally_counts_block_omega_elements_and_iov_kappas() {
    let text = format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
{WARFARIN_TAIL}"#
    );
    let t = tally(&text, 10);
    assert_eq!(t.theta_random, 3);
    assert_eq!(t.theta_fixed, 0);
    // 2×2 block → 3 Cholesky elements, plus the diagonal ETA_KA; the two
    // cross-block structural zeros are not parameters.
    assert_eq!(t.omega, 4);
    assert_eq!(t.n_free(), model_free_dim(&text));
    assert_eq!(t.kappa, 1);
    assert_eq!(t.sigma, 1);
}

#[test]
fn tally_moves_sigma_to_the_random_class_under_iiv_on_ruv() {
    let text = r#"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
"#;
    let t = tally(text, 10);
    assert!(t.sigma_random);
    assert_eq!(t.sigma, 1);
    assert_eq!(t.omega, 4);
    assert_eq!(t.n_random_class(), 3 + 4 + 1);
    assert_eq!(t.n_fixed_class(), 0);
}

// ── Strictness ──────────────────────────────────────────────────────────────

fn clean_fit() -> FitResult {
    // `minimal_fit_result` has converged = true, a 3×3 identity covariance
    // (condition number 5.0 recorded), no warnings — and, so the init-stall
    // gate has something to compare, estimates that have moved off their inits.
    let mut r = minimal_fit_result();
    r.theta_init = r.theta.iter().map(|t| t * 0.5).collect();
    r
}

fn boundary_warning(with_details: bool) -> WarningEntry {
    WarningEntry {
        severity: WarningSeverity::Warning,
        category: WarningCode::BoundaryEstimate,
        message: "Parameter estimate(s) pinned to an optimizer bound: TVKA (50.0000 at upper bound)."
            .into(),
        source_method: None,
        details: with_details.then(|| {
            serde_json::json!({
                "parameters": [{"parameter": "TVKA", "estimate": 50.0, "bound": 50.0, "side": "upper"}]
            })
        }),
    }
}

#[test]
fn default_strictness_passes_a_clean_fit_with_no_skips() {
    let v = check_strictness(&clean_fit(), &Strictness::default());
    assert!(v.passed, "{v:?}");
    assert!(v.failures.is_empty());
    assert!(v.skipped.is_empty(), "{v:?}");
}

#[test]
fn strictness_none_passes_everything() {
    let mut r = clean_fit();
    r.converged = false;
    r.covariance_matrix = None;
    r.cov_condition_number = Some(f64::INFINITY);
    r.warnings_structured.push(boundary_warning(true));
    r.theta_init = r.theta.clone();
    let v = check_strictness(&r, &Strictness::none());
    assert!(v.passed);
    assert!(v.failures.is_empty() && v.skipped.is_empty());
}

#[test]
fn require_converged_fails_an_unconverged_fit() {
    let mut r = clean_fit();
    r.converged = false;
    let s = Strictness {
        require_converged: true,
        ..Strictness::none()
    };
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    assert_eq!(v.failures.len(), 1);
    assert!(v.failures[0].contains("did not converge"), "{v:?}");
    r.converged = true;
    assert!(check_strictness(&r, &s).passed);
}

#[test]
fn require_covariance_names_each_non_computed_status() {
    let s = Strictness {
        require_covariance: true,
        ..Strictness::none()
    };
    let mut r = clean_fit();
    assert!(check_strictness(&r, &s).passed);

    r.covariance_status = CovarianceStatus::NotRequested;
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    assert!(v.failures[0].contains("not requested"), "{v:?}");

    r.covariance_status = CovarianceStatus::Failed;
    assert!(check_strictness(&r, &s).failures[0].contains("failed"));

    r.covariance_status = CovarianceStatus::SirFallback;
    assert!(check_strictness(&r, &s).failures[0].contains("SIR"));

    r.covariance_status = CovarianceStatus::Computed;
    r.covariance_matrix = None;
    assert!(check_strictness(&r, &s).failures[0].contains("no matrix"));
}

#[test]
fn max_condition_number_fails_above_threshold_and_skips_without_one() {
    let s = Strictness {
        max_condition_number: Some(1000.0),
        ..Strictness::none()
    };
    let mut r = clean_fit();
    r.cov_condition_number = Some(999.0);
    assert!(check_strictness(&r, &s).passed);

    r.cov_condition_number = Some(1.4e6);
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    assert!(
        v.failures[0].starts_with("condition number 1.4000e6"),
        "{v:?}"
    );

    // Singular correlation matrix → +∞, which exceeds any finite threshold.
    r.cov_condition_number = Some(f64::INFINITY);
    assert!(!check_strictness(&r, &s).passed);

    r.cov_condition_number = Some(f64::NAN);
    assert!(check_strictness(&r, &s).failures[0].contains("NaN"));

    r.cov_condition_number = None;
    let v = check_strictness(&r, &s);
    assert!(v.passed, "an untestable gate is skipped, not failed");
    assert_eq!(v.skipped.len(), 1);
    assert!(v.skipped[0].starts_with("condition number"), "{v:?}");
}

/// 3×3 covariance whose (0,1) correlation is exactly `r01`, the rest 0.
fn cov_with_corr(r01: f64) -> DMatrix<f64> {
    // var = 4, 9, 1 → sd = 2, 3, 1 → cov01 = r·6
    DMatrix::from_row_slice(
        3,
        3,
        &[4.0, r01 * 6.0, 0.0, r01 * 6.0, 9.0, 0.0, 0.0, 0.0, 1.0],
    )
}

#[test]
fn max_abs_correlation_reads_the_whole_matrix() {
    let mut r = clean_fit();
    r.covariance_matrix = Some(cov_with_corr(-0.97));
    assert!((max_abs_correlation(&r).unwrap() - 0.97).abs() < 1e-12);

    // A coordinate with zero variance (a FIXed parameter's row) is skipped.
    let mut m = cov_with_corr(0.5);
    m[(1, 1)] = 0.0;
    m[(0, 1)] = 0.0;
    m[(1, 0)] = 0.0;
    r.covariance_matrix = Some(m);
    assert_eq!(max_abs_correlation(&r), Some(0.0));

    r.covariance_matrix = Some(DMatrix::from_element(1, 1, 2.0));
    assert_eq!(max_abs_correlation(&r), None);
    r.covariance_matrix = None;
    assert_eq!(max_abs_correlation(&r), None);
}

#[test]
fn max_correlation_fails_above_threshold_naming_the_pair() {
    let s = Strictness {
        max_correlation: Some(0.95),
        ..Strictness::none()
    };
    let mut r = clean_fit();
    r.covariance_matrix = Some(cov_with_corr(0.94));
    assert!(check_strictness(&r, &s).passed);

    r.covariance_matrix = Some(cov_with_corr(-0.96));
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    assert!(
        v.failures[0].contains("0.9600") && v.failures[0].contains("CL ~ V"),
        "{v:?}"
    );

    r.covariance_matrix = None;
    let v = check_strictness(&r, &s);
    assert!(v.passed);
    assert!(v.skipped[0].starts_with("parameter correlation"), "{v:?}");
}

#[test]
fn reject_on_boundary_is_the_bootstrap_predicate() {
    let s = Strictness {
        reject_on_boundary: true,
        ..Strictness::none()
    };
    let mut r = clean_fit();
    assert!(!estimate_near_boundary(&r));
    assert!(check_strictness(&r, &s).passed);

    r.warnings_structured.push(boundary_warning(true));
    assert!(estimate_near_boundary(&r));
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    assert_eq!(v.failures[0], "estimate pinned to a declared bound: TVKA");

    // Without the structured payload (string-classified warning) the message
    // itself is the reason.
    r.warnings_structured = vec![boundary_warning(false)];
    let v = check_strictness(&r, &s);
    assert!(
        v.failures[0].contains("TVKA (50.0000 at upper bound)"),
        "{v:?}"
    );

    // A runaway-guard hit is not a boundary estimate; `converged` covers it.
    r.warnings_structured = vec![WarningEntry {
        category: WarningCode::ParameterAtRunawayGuard,
        ..boundary_warning(false)
    }];
    assert!(!estimate_near_boundary(&r));
}

#[test]
fn stalled_at_init_reads_free_theta_omega_sigma_displacement() {
    let mut r = minimal_fit_result();
    // Everything on its initial value → stalled.
    r.theta_init = r.theta.clone();
    r.omega_init = r.omega.clone();
    r.sigma_init = r.sigma.clone();
    assert_eq!(stalled_at_init(&r), Some(true));

    // #751's own measurement: 1.4e-4 relative on one theta is still a stall.
    r.theta[0] = r.theta_init[0] * (1.0 + 1.4e-4);
    assert_eq!(stalled_at_init(&r), Some(true));

    // A 2% move on a single free coordinate is enough to have left init.
    r.theta[0] = r.theta_init[0] * 1.02;
    assert_eq!(stalled_at_init(&r), Some(false));
    r.theta[0] = r.theta_init[0];
    r.omega[(1, 1)] = r.omega_init[(1, 1)] * 1.02;
    assert_eq!(stalled_at_init(&r), Some(false));
    r.omega[(1, 1)] = r.omega_init[(1, 1)];
    r.sigma[0] = r.sigma_init[0] * 0.98;
    assert_eq!(stalled_at_init(&r), Some(false));
    r.sigma[0] = r.sigma_init[0];

    // A FIXed coordinate that "moved" does not count.
    r.theta_fixed[1] = true;
    r.theta[1] = r.theta_init[1] * 2.0;
    assert_eq!(stalled_at_init(&r), Some(true));
    r.omega_fixed[0] = true;
    r.omega[(0, 0)] = r.omega_init[(0, 0)] * 2.0;
    assert_eq!(stalled_at_init(&r), Some(true));
}

#[test]
fn stalled_at_init_uses_an_absolute_tolerance_for_a_zero_init() {
    let mut r = minimal_fit_result();
    r.theta_init = r.theta.clone();
    r.omega_init = r.omega.clone();
    r.sigma_init = r.sigma.clone();
    // An identity-packed theta (covariate exponent) initialised at exactly 0.
    r.theta_init[2] = 0.0;
    r.theta[2] = 0.005;
    assert_eq!(stalled_at_init(&r), Some(true));
    r.theta[2] = 0.02;
    assert_eq!(stalled_at_init(&r), Some(false));
}

#[test]
fn stalled_at_init_is_none_without_comparable_initials() {
    let mut r = minimal_fit_result();
    r.theta_init.clear();
    assert_eq!(stalled_at_init(&r), None);

    let mut r = minimal_fit_result();
    r.theta_init = r.theta.clone();
    r.sigma_init = r.sigma.clone();
    r.omega_init = DMatrix::zeros(1, 1);
    assert_eq!(stalled_at_init(&r), None);
}

#[test]
fn reject_init_stall_names_the_751_signature() {
    let s = Strictness {
        reject_init_stall: true,
        ..Strictness::none()
    };
    // A candidate that never left its initial estimates.
    let mut r = minimal_fit_result();
    r.converged = true; // #751 before the fix: reported converged at init
    r.theta_init = r.theta.clone();
    r.omega_init = r.omega.clone();
    r.sigma_init = r.sigma.clone();
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    assert_eq!(v.failures.len(), 1);
    assert!(
        v.failures[0].starts_with("stalled at the initial estimates")
            && v.failures[0].contains("#751"),
        "{v:?}"
    );

    r.theta[0] *= 1.5;
    assert!(check_strictness(&r, &s).passed);

    r.theta_init.clear();
    let v = check_strictness(&r, &s);
    assert!(v.passed);
    assert!(v.skipped[0].starts_with("init stall"), "{v:?}");
}

#[test]
fn verdict_lists_every_failed_gate_in_field_order() {
    let mut r = minimal_fit_result();
    r.converged = false;
    r.covariance_status = CovarianceStatus::Failed;
    r.covariance_matrix = Some(cov_with_corr(0.99));
    r.cov_condition_number = Some(5e4);
    r.warnings_structured.push(boundary_warning(true));
    r.theta_init = r.theta.clone();
    r.omega_init = r.omega.clone();
    r.sigma_init = r.sigma.clone();
    let s = Strictness {
        require_covariance: true,
        ..Strictness::default()
    };
    let v = check_strictness(&r, &s);
    assert!(!v.passed);
    let heads: Vec<&str> = v
        .failures
        .iter()
        .map(|f| f.split_whitespace().next().unwrap())
        .collect();
    assert_eq!(
        heads,
        [
            "did",
            "covariance",
            "condition",
            "parameter",
            "estimate",
            "stalled"
        ],
        "{v:?}"
    );
    assert!(v.skipped.is_empty());
}

#[test]
fn strictness_serde_fills_missing_keys_from_default() {
    // A `.ferxsearch` `[strictness]` table lists only what it changes.
    let s: Strictness = serde_json::from_str(r#"{"require_covariance": true}"#).unwrap();
    assert!(s.require_covariance);
    assert!(s.require_converged);
    assert_eq!(s.max_condition_number, Some(1000.0));
    assert_eq!(s.max_correlation, Some(0.95));
    let none: Strictness = serde_json::from_str("{}").unwrap();
    assert_eq!(none, Strictness::default());
}
