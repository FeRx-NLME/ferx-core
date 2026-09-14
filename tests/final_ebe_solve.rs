//! #833 — the OFV a fit reports must not be worse than the objective its optimizer
//! reached.
//!
//! After the outer loop restores its best-seen point (#59) the final inner loop
//! re-derives the empirical Bayes estimates there. It used to do that from a **cold**
//! start while every evaluation during the fit had warm-started them, so on an inner
//! problem where the two disagree the reported `ofv` — and the AIC/BIC, and the point
//! the covariance step runs at — came out above the value that made the point
//! best-seen. Measured at +3.5 OFV on a fluconazole 2-cpt binding model, +6.96 on the
//! FREM warfarin fixture, and +3513 on that same FREM fixture under glibc (#1349).
//!
//! The unit tests in `outer_optimizer_tests.rs` own the selection rule itself
//! (`warm_solve_wins`, `ebe_start_dependence_gap`); this file pins the wiring, on a
//! fixture built so the disagreement is a property of the *configuration* rather than
//! of one platform's libm.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::{WarningCode, WarningSeverity};
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// A tightened inner budget is the deterministic way to make a cold restart disagree
/// with the warm trajectory: `inner_maxiter` is a budget, so a warm start that begins at
/// the previous iteration's η̂ arrives on it and a cold start from η = 0 spending the
/// same budget does not. Nothing here depends on the individual objective being
/// multimodal, which is what makes it reproducible off one machine.
///
/// 10 is measured, not picked. Below it the *fit* starves too and never reaches the
/// optimum at all (OFV 242.5 at 1, 212.4 at 2, 196.5 at 3, 188.9 at 5 — and no gap,
/// because a trajectory that never converged has nothing a cold restart can fail to
/// reproduce). At 10 the warm trajectory reaches warfarin's usual FOCEI optimum while a
/// cold re-solve on the same budget lands 255 OFV units short.
fn starved_inner_opts() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.inner_maxiter = 10;
    opts.outer_maxiter = 200;
    opts
}

#[test]
fn a_fit_reports_the_objective_its_optimizer_reached() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let starved = fit(
        &model,
        &population,
        &model.default_params,
        &starved_inner_opts(),
    )
    .expect("a starved-inner fit still returns a result");

    // Precondition, and the reason this fixture exists: the two final solves must
    // actually disagree here. Without a gap the assertions below are satisfied by the
    // pre-#833 behaviour they exist to reject.
    let entry = starved
        .warnings_structured
        .iter()
        .find(|w| w.category == WarningCode::EbeStartDependent)
        .unwrap_or_else(|| {
            panic!(
                "expected an ebe_start_dependent warning — a cold restart on a 10-iteration \
                 inner budget cannot reproduce the warm trajectory's EBEs, so if this is \
                 absent the second solve is not running. warnings: {:?}",
                starved.warnings
            )
        });
    assert_eq!(entry.severity, WarningSeverity::Warning);

    // The message quotes `(<cold> vs the reported <ofv>)`. The reported OFV must be the
    // lower of the two, and must be the number the fit actually publishes.
    let (cold, reported) = parse_ebe_gap_message(&entry.message);
    assert!(
        cold > reported,
        "the reported OFV ({reported}) must be the lower of the two solves; cold = {cold}"
    );
    assert!(
        (reported - starved.ofv).abs() < 5e-4,
        "the warning's reported value ({reported}) must be the fit's own OFV ({})",
        starved.ofv
    );
    // The substantive half: what the fit publishes is warfarin's own FOCEI optimum, the
    // value the trajectory reached. Before #833 this same run reported −31.06 — the cold
    // re-solve — 255 OFV units worse, on a fit whose parameters were fine.
    assert!(
        (starved.ofv - (-286.0042)).abs() < 0.05,
        "expected the trajectory's own optimum (−286.0042), got {} (cold re-solve = {cold})",
        starved.ofv
    );
    assert!(
        cold - reported > 100.0,
        "the measured gap on this fixture is ~255 OFV units; got {}",
        cold - reported
    );
}

/// The control: with an inner budget that lets a cold restart reach the same mode, the
/// two solves agree, no warning fires, and the fit is what it always was. This is what
/// keeps the warning from becoming background noise on ordinary fits — and it is the
/// arm that fails if the gap threshold is ever dropped to zero.
#[test]
fn an_ordinary_fit_reports_no_start_dependence() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = starved_inner_opts();
    opts.inner_maxiter = FitOptions::default().inner_maxiter;

    let ordinary = fit(&model, &population, &model.default_params, &opts).expect("warfarin fit");

    assert!(
        !ordinary
            .warnings_structured
            .iter()
            .any(|w| w.category == WarningCode::EbeStartDependent),
        "no start dependence expected on warfarin at the default inner budget: {:?}",
        ordinary.warnings
    );
}

/// Pull `(cold, reported)` out of the `W_EBE_START_DEPENDENT` message, which formats
/// them as `(<cold> vs the reported <reported>)`.
fn parse_ebe_gap_message(msg: &str) -> (f64, f64) {
    let (before, after) = msg
        .split_once("vs the reported ")
        .unwrap_or_else(|| panic!("unexpected message shape: {msg}"));
    let reported: f64 = after
        .split(')')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("reported value not a number in {msg}: {e}"));
    let cold: f64 = before
        .rsplit('(')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("cold value not a number in {msg}: {e}"));
    (cold, reported)
}
