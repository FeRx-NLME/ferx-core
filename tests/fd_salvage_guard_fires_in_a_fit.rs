//! Tier 2 — the #1520 salvage guard, end to end through `fit()`.
//!
//! The unit tests in `outer_optimizer_tests.rs` pin the predicate, the assembly and the
//! per-optimizer bookkeeping with hand-built inputs; nothing there can see whether the
//! NLopt closure actually *wires* them — records each evaluation, asks the bookkeeping
//! for a reference, hands the assembly a real `OuterTrial`. This test does: a fit on the
//! bundled `warfarin_if` example, which the instrumented sweep behind #1520 found reaching
//! blown-up first trials at evaluations 2 and 4 under SLSQP (excess over the iterate
//! `2.2e5` and `3.7e6` on 320 observations) and under NLopt L-BFGS (`1.7e4`, `1.0e5`), so
//! eight evaluations are enough and the fit returns in well under a second. Asserted on
//! the `FitResult` warning the guard extends:
//!
//! - under **SLSQP** the sentence is present with a non-zero count — a closure that never
//!   records an evaluation, or hands the assembly `OuterTrial::unknown()`, leaves it out;
//! - under **NLopt L-BFGS** the sentence is absent — its policy is `Off` (PR #1525 review:
//!   no reference ferx can form is a rejected-trial guarantee for Luksan's line search),
//!   so a wiring that fired there anyway reddens this.
//!
//! Deliberately not asserted: the OFV, which the guard must not be able to move at an
//! accepted point, and which #1515 established is not evidence about a gradient anyway.

use ferx_core::{fit, parse_model_file, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

fn skipped_salvages_in(warnings: &[String]) -> Option<usize> {
    warnings.iter().find_map(|w| {
        let marker = " of their salvage gradients were skipped";
        let end = w.find(marker)?;
        let head = &w[..end];
        head.rsplit(' ').next()?.parse().ok()
    })
}

#[test]
fn a_blown_up_first_trial_skips_the_salvage_under_slsqp_and_never_under_lbfgs() {
    let model = parse_model_file(Path::new("examples/warfarin_if.ferx")).expect("bundled example");
    let pop = read_nonmem_csv(Path::new("data/warfarin_if.csv"), None, None).expect("bundled data");
    for (optimizer, expect_skips) in [(Optimizer::Slsqp, true), (Optimizer::NloptLbfgs, false)] {
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            interaction: true,
            optimizer,
            outer_maxiter: 8,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let result = fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{optimizer:?}: the short fit must return Ok: {e}"));
        let fd = result
            .warnings
            .iter()
            .find(|w| w.contains("could not be given the exact analytic outer gradient"))
            .unwrap_or_else(|| {
                panic!(
                    "{optimizer:?}: the fixture must still decline a subject on this path, \
                     or the guard has nothing to skip; warnings: {:?}",
                    result.warnings
                )
            });
        let skipped = skipped_salvages_in(&result.warnings);
        if expect_skips {
            assert!(
                skipped.is_some_and(|k| k >= 1),
                "{optimizer:?}: at least one salvage must be skipped within 8 evaluations; \
                 got: {fd}"
            );
        } else {
            assert!(
                skipped.is_none(),
                "{optimizer:?}: the guard is off for this optimizer and must skip nothing; \
                 got: {fd}"
            );
        }
    }
}
