//! #1512: the strictness gate must exclude a fit whose covariance step had to floor a
//! Hessian eigenvalue.
//!
//! The fixture is the one-peripheral candidate `ferx modelsearch` builds from
//! `examples/warfarin.ferx` over `PERIPHERALS(0..1)`: warfarin carries no information about a
//! second compartment, so the fit collapses it (V2 → ~3e-5, Q free, OFV equal to the
//! one-compartment base). That leaves one flat Hessian direction, which the covariance step's
//! eigenvalue floor replaces with a finite one — and the condition number and correlations the
//! numeric gates read are then computed from the floored matrix, which no longer shows the
//! collapse. Measured before the fix: correlation-matrix condition number 2.98, TVQ RSE
//! 293519 %, and the candidate passed `Strictness::default()`.
//!
//! The test asserts the straddle first — the numeric gates alone *pass* this fit — so the
//! exclusion it then asserts can only come from the regularization check.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{
    check_strictness, fit, max_abs_correlation, read_nonmem_csv, EstimationMethod, FitOptions,
    Strictness, WarningCode,
};
use std::path::Path;

/// `ferx modelsearch`'s `run1` for `examples/warfarin_modelsearch.ferxsearch`, verbatim bar
/// the `[simulation]` block.
const WARFARIN_ONE_PERIPHERAL: &str = r"
[parameters]
  theta TVCL(0.132968912843207, 0.001, 10.0)
  theta TVV(7.73070027830032, 0.1, 500.0)
  theta TVKA(0.725208107987929, 0.01, 50.0)

  omega ETA_CL ~ 0.0285950800457939
  omega ETA_V  ~ 0.00957691316624095
  omega ETA_KA ~ 0.348964093203287

  sigma PROP_ERR ~ 0.0107485219888545 (sd)
  theta TVQ(0.132968912843207, 0.0, 1000000.0)
  theta TVV2(0.386535013915016, 0.0, 1000000.0)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  Q = TVQ
  V2 = TVV2

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_collapsed_peripheral_fails_strictness_on_its_regularized_covariance() {
    let model = parse_model_string(WARFARIN_ONE_PERIPHERAL).expect("model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");
    // `parse_model_string` drops `[fit_options]`, so the method is pinned here.
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        interaction: false,
        outer_maxiter: 300,
        run_covariance_step: true,
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");

    // The collapse the fixture exists for.
    let v2 = r.theta[r.theta_names.iter().position(|n| n == "TVV2").unwrap()];
    assert!(v2 < 1e-2, "TVV2 = {v2}: the peripheral did not collapse");
    assert!(
        r.warnings_structured
            .iter()
            .any(|w| w.category == WarningCode::CovarianceRegularized),
        "the eigenvalue floor did not fire: {:?}",
        r.warnings
    );

    // The straddle: the numbers the thresholds read pass them.
    let cn = r.cov_condition_number.expect("condition number");
    let max_r = max_abs_correlation(&r).expect("correlation");
    let s = Strictness::default();
    assert!(
        cn <= s.max_condition_number.unwrap(),
        "condition number {cn} already fails its threshold, so this fixture no longer isolates \
         the regularization check"
    );
    assert!(
        max_r <= s.max_correlation.unwrap(),
        "max |r| {max_r} already fails its threshold"
    );

    let v = check_strictness(&r, &s);
    assert!(!v.passed, "a collapsed peripheral passed strictness: {v:?}");
    assert!(
        v.failures
            .iter()
            .any(|f| f.starts_with("covariance matrix regularized:")),
        "{v:?}"
    );
}
