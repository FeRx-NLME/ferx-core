//! #2799: `FitResult::status()` end to end, on #2797's guard-hold fixture
//! (`tests/data/guard_hold/`, a one-compartment oral population simulated with no KA
//! variability and fitted with an ETA on KA that collapses onto the rail ω = e⁻¹²).
//!
//! Tier 2: both fits are evaluations (`maxiter = 0`) at a fixed point, which run the
//! covariance step once and return. `converged` is `false` at `maxiter = 0` by construction,
//! so no test asserts it; the status's reasons are what is under test.
//!
//! The step-failed arm (a collapse beside `CovarianceStatus::Failed`) lives in
//! `tests/profiled_levels.rs::a_profiled_fit_lists_a_collapse_even_when_its_step_fails`:
//! from this fixture's poor-σ point no input moved one at a time (σ to 0.001, each θ by
//! 4–10× either way, 20 points measured) made the analytic `R` fail rather than floor.

use ferx_core::run_model_with_data;
use ferx_core::types::{FitResult, FitStatus, StatusReason};
use std::path::PathBuf;

const DATA: &str = "tests/data/guard_hold/sim_oral_no_ka_iiv.csv";
/// ω = exp(−12), written so it packs back onto the log-Cholesky guard −6 exactly.
const RAIL: &str = "6.1442123533282098e-6";
/// The converged σ (SD scale): the stationary point, where `R` is clean.
const SIGMA_HAT: &str = "0.10069194732313741";
/// The poor-σ point (#2797 audit probe G): `R` is indefinite along ETA_KA and floored.
const SIGMA_POOR: &str = "0.02";

fn model_source(sigma: &str) -> String {
    format!(
        "[parameters]
  theta TVCL(0.12250099819205257, 0.001, 10.0)
  theta TVV(7.753670975867449, 0.1, 500.0)
  theta TVKA(0.9982755334099482, 0.01, 50.0)
  omega ETA_CL ~ 0.0792688562315328
  omega ETA_V  ~ 0.05944155243978235
  omega ETA_KA ~ {RAIL}
  sigma PROP_ERR ~ {sigma} (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = true
"
    )
}

fn evaluate(name: &str, sigma: &str) -> FitResult {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join(format!("{name}.ferx"));
    std::fs::write(&path, model_source(sigma)).unwrap();
    run_model_with_data(path.to_str().unwrap(), Some(DATA))
        .expect("evaluate")
        .0
}

fn floored(r: &FitResult) -> bool {
    r.warnings
        .iter()
        .any(|w| w.starts_with("Covariance step regularized"))
}

/// F and G in one test, so the pair cannot both land on one side of the hold:
///
/// - **F** (stationary): a clean `R` holds ETA_KA, so the collapse is a clean answer and the
///   only reason is `evaluation_only`;
/// - **G** (poor σ): a floored `R` does not hold it, so `collapsed_at_guard: ETA_KA` joins.
///
/// The straddle itself (no floor / a floor; held / not held) is asserted first. Mutations,
/// each named by the assertion that dies: the `EvaluationOnly` source off → both; the
/// `collapsed_at_guard` source off → G; the `cov_fixed` conjunct dropped → F.
#[test]
fn status_names_an_unheld_collapse_and_an_evaluation_only_run() {
    const ETA_KA_ROW: usize = 5;
    let f = evaluate("stationary", SIGMA_HAT);
    let g = evaluate("poor_sigma", SIGMA_POOR);

    // The straddle.
    assert!(!floored(&f), "F has a clean R: {:?}", f.warnings);
    assert!(floored(&g), "G floors R: {:?}", g.warnings);
    assert!(f.cov_fixed[ETA_KA_ROW], "F holds ETA_KA: {:?}", f.cov_fixed);
    assert!(
        !g.cov_fixed[ETA_KA_ROW],
        "G does not hold ETA_KA: {:?}",
        g.cov_fixed
    );
    for (name, r) in [("F", &f), ("G", &g)] {
        assert!(r.evaluation_only, "{name}: maxiter = 0 on FOCEI evaluates");
        assert!(
            !r.eval_budget_spent,
            "{name}: an evaluation spends no budget"
        );
        assert_eq!(r.held_at_bound.len(), 1, "{name}");
        assert_eq!(r.held_at_bound[0].name, "ETA_KA", "{name}");
    }

    assert_eq!(
        f.status(),
        FitStatus::NotConverged {
            reasons: vec![StatusReason::EvaluationOnly]
        },
        "F: {:?}",
        f.warnings
    );
    assert_eq!(
        g.status(),
        FitStatus::NotConverged {
            reasons: vec![
                StatusReason::EvaluationOnly,
                StatusReason::CollapsedAtGuard {
                    parameters: vec!["ETA_KA".into()]
                },
            ]
        },
        "G: {:?}",
        g.warnings
    );
    // The printed form a user sees, from the same labels every printer uses.
    let printed =
        |r: &FitResult| -> Vec<String> { r.status().reasons().iter().map(|x| x.label()).collect() };
    assert_eq!(printed(&f), ["evaluation_only"]);
    assert_eq!(
        printed(&g),
        ["evaluation_only", "collapsed_at_guard: ETA_KA"]
    );
}
