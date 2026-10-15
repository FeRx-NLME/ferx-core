//! #1565 — a fit must not report `converged` at parameters it optimized under trapped EBEs.
//!
//! Every evaluation of the outer loop warm-starts the inner loop from the incumbent's
//! EBEs (#1297). When some subjects enter a secondary individual mode early — at the
//! poor initial estimates — that start carries them there for the whole trajectory, and
//! the optimizer minimises a surrogate lying *above* the FOCE(I) objective. The final cold
//! re-solve (#751 / #833) finds the lower modes, but before #1565 its consistency check
//! was one-sided: a cold objective materially *better* than best-seen passed as
//! "consistent", and the fit reported `converged = true` at the surrogate's optimum.
//!
//! The fixture is the #1564 benchmark kit's Michaelis–Menten ODE model, from its own
//! perturbed initial estimates (`tests/fixtures/mm_basin_shift/`). On the full
//! 200-subject dataset `main` reported OFV 784.33 converged; the optimum is 775.39
//! (8.94 lower, with materially different θ). NONMEM 7.6 FOCEI agrees on both numbers
//! when its inner search is given enough starts (`MCETA = 200`): 784.349 at the
//! reported point and 775.414 at the optimum, a gap of 8.935 against ferx's 8.938.
//! Without `MCETA`, NONMEM's own cold η search lands subject 54 in the higher mode at the
//! optimum (+13.78 there, every other subject within 0.012 of ferx), and its estimation
//! from these starts stops with rounding errors at 814.84. The control streams are in
//! `tests/nonmem/mm_basin_shift_*.ctl`.
//!
//! The two Tier-2 tests use a 10-subject subset (subjects 1–5 plus the five the issue
//! names as trapped) on which the same trap reproduces in about a second. Every number
//! below was measured on Linux x86_64, identically at 1, 2, 4 and 16 threads.
//!
//! The unit tests in `outer_optimizer_tests.rs` own the gate itself
//! (`cold_solve_beats_best_seen`, the restart resolver's tie rule, the message); this
//! file pins what a fit publishes.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitResult, WarningCode};
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const MODEL: &str = "tests/fixtures/mm_basin_shift/mm.ferx";

fn fit_fixture(data: &str, outer_maxiter: Option<usize>) -> FitResult {
    let text = std::fs::read_to_string(MODEL).expect("fixture model must exist");
    let parsed = parse_full_model(&text).expect("fixture model must parse");
    let population = read_nonmem_csv(Path::new(data), None, None).expect("fixture data must load");
    let mut opts = parsed.fit_options.clone();
    opts.verbose = false;
    if let Some(m) = outer_maxiter {
        opts.outer_maxiter = m;
    }
    fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &opts,
    )
    .expect("the fixture fit returns a result")
}

/// The OFV a `W_EBE_START_DEPENDENT` basin-shift message quotes as the cold re-solve:
/// `({cold:.4} vs {best_seen:.4})`.
fn quoted_pair(message: &str) -> (f64, f64) {
    let open = message
        .find(" there (")
        .expect("message quotes the two objectives")
        + 8;
    let close = open + message[open..].find(')').unwrap();
    let mut it = message[open..close].split(" vs ");
    let cold = it.next().unwrap().trim().parse().unwrap();
    let best = it.next().unwrap().trim().parse().unwrap();
    (cold, best)
}

/// The restart path. Before #1565 this fit reported OFV −143.4726 with `converged =
/// true`: best-seen −141.7318, cold re-solve at the same θ 1.74 *lower*, one subject in
/// a lower cold mode. Now the run is restarted from that point — its first inner solve
/// is cold — and reaches −148.6799, 5.2 below what was reported. A refit from those
/// estimates reproduces −148.6804, so this is the optimum and not another trap.
#[test]
fn a_trapped_fit_is_restarted_to_the_lower_objective() {
    let result = fit_fixture("tests/fixtures/mm_basin_shift/mm_basin10.csv", None);

    // The straddle: this fixture must actually have trapped, or the remaining
    // assertions test a fit that never needed the fix. The adopted restart says so.
    let notice = result
        .warnings_structured
        .iter()
        .find(|w| w.message.contains("higher individual mode"))
        .unwrap_or_else(|| {
            panic!(
                "expected the basin-shift restart notice — the fixture no longer traps, or \
                 the restart did not run (OFV = {}, warnings: {:?})",
                result.ofv, result.warnings
            )
        });
    assert_eq!(notice.category, WarningCode::OptimizerHealth);
    assert!(
        notice.message.contains("cold OFV = -143.472619"),
        "the notice names the trapped leg's cold objective: {}",
        notice.message
    );

    assert!(
        result.ofv < -148.6,
        "expected the restart to reach the optimum (−148.6799); pre-#1565 reported \
         −143.4726. Got {}",
        result.ofv
    );
    assert!(
        result.converged,
        "the restarted leg converged on its own terms"
    );
    // The gap was resolved, so it must not be reported as surviving.
    assert!(
        !result
            .warnings_structured
            .iter()
            .any(|w| w.category == WarningCode::EbeStartDependent),
        "no start-dependence warning once the restart resolved the gap: {:?}",
        result.warnings
    );
}

/// The no-restart path. At `maxiter = 5` (a 45-evaluation budget) the first leg ends
/// with NLopt `Success` after 46 evaluations — L-BFGS checks `maxeval` only between line
/// searches — at the same trapped point as above, so there is no budget left to restart
/// on. Before #1565 this fit reported `converged = true`; the gap must now surface as a
/// warning and a non-converged fit.
#[test]
fn a_basin_shift_with_no_budget_left_is_not_reported_converged() {
    let result = fit_fixture("tests/fixtures/mm_basin_shift/mm_basin10.csv", Some(5));

    let warning = result
        .warnings_structured
        .iter()
        .find(|w| w.category == WarningCode::EbeStartDependent)
        .unwrap_or_else(|| {
            panic!(
                "expected W_EBE_START_DEPENDENT for the surviving gap (OFV = {}, \
                 converged = {}, warnings: {:?})",
                result.ofv, result.converged, result.warnings
            )
        });
    assert!(
        warning.message.contains("OFV units *lower*"),
        "the basin-shift arm, not #833's worse-cold arm: {}",
        warning.message
    );
    assert!(
        warning.message.contains("1 subject's EBEs"),
        "one subject is trapped on this fixture: {}",
        warning.message
    );
    let (cold, best) = quoted_pair(&warning.message);
    assert!(
        (cold - result.ofv).abs() < 1e-4,
        "the quoted cold objective is the reported one ({cold} vs {})",
        result.ofv
    );
    assert!(
        (best - cold - 1.7408).abs() < 1e-3,
        "measured gap 1.7408 (best-seen −141.7318, cold −143.4726): got {best} vs {cold}"
    );
    assert!(
        !result.converged,
        "a fit optimized under trapped EBEs must not report converged"
    );
}

/// Tier 3: the benchmark the issue was found on, 200 subjects. `main` reported 784.33
/// converged; the optimum is 775.39 (NONMEM, `MCETA = 200`: 775.414).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn mm_benchmark_reaches_the_optimum() {
    let result = fit_fixture("tests/fixtures/mm_basin_shift/mm_n200.csv", None);
    assert!(
        result.ofv <= 775.40,
        "expected the FOCEI optimum 775.39 (pre-#1565: 784.33 reported converged), got {}",
        result.ofv
    );
    assert!(result.converged);
}
