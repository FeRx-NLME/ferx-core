//! Tier 2 — the #1520 salvage guard, end to end through `fit()`.
//!
//! The unit tests in `outer_optimizer_tests.rs` pin the predicate and the assembly with a
//! hand-built `OuterTrial`; nothing there can see whether the optimizer closures actually
//! *fill* one — the incumbent promoted on an improving evaluation, the trial's objective
//! and per-subject contributions handed to the gradient. This test does: a fit on the
//! bundled `warfarin_if` example, which the instrumented sweep behind #1520 found firing
//! the guard at evaluations 2 and 4 under both NLopt L-BFGS (excess over the incumbent
//! `1.7e4` and `1.0e5` on 320 observations) and SLSQP (`2.2e5` and `3.7e6`), so eight
//! evaluations are enough to reach a skipped salvage and the fit returns in well under a
//! second. Asserted on the `FitResult` warning the guard extends, both that it is present
//! and that its count is non-zero — a closure that never promotes an incumbent, or hands
//! the assembly `OuterTrial::unknown()`, leaves the sentence out and reddens this.
//!
//! Deliberately not asserted: the OFV, which the guard must not be able to move at an
//! accepted point, and which #1515 established is not evidence about a gradient anyway.

use ferx_core::{fit, parse_model_file, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

fn skipped_salvages_in(warnings: &[String]) -> Option<usize> {
    warnings.iter().find_map(|w| {
        let marker = " of their finite-difference salvages were skipped";
        let end = w.find(marker)?;
        let head = &w[..end];
        head.rsplit(' ').next()?.parse().ok()
    })
}

#[test]
fn a_blown_up_line_search_trial_skips_the_salvage_under_lbfgs_and_slsqp() {
    let model = parse_model_file(Path::new("examples/warfarin_if.ferx")).expect("bundled example");
    let pop = read_nonmem_csv(Path::new("data/warfarin_if.csv"), None, None).expect("bundled data");
    for optimizer in [Optimizer::NloptLbfgs, Optimizer::Slsqp] {
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
            .find(|w| w.contains("used finite-difference outer gradients"))
            .unwrap_or_else(|| {
                panic!(
                    "{optimizer:?}: the fixture must still decline a subject on this path, \
                     or the guard has nothing to skip; warnings: {:?}",
                    result.warnings
                )
            });
        let skipped = skipped_salvages_in(&result.warnings).unwrap_or_else(|| {
            panic!("{optimizer:?}: the FD-fallback warning must report skipped salvages; got: {fd}")
        });
        assert!(
            skipped >= 1,
            "{optimizer:?}: at least one salvage must be skipped within 8 evaluations; got: {fd}"
        );
    }
}
