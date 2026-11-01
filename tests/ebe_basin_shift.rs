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
//! The two Tier-2 tests use a 10-subject subset on which the same trap reproduces in a
//! few seconds. Every number below was measured on Linux x86_64.
//!
//! **The subset changed with #1527.** The original one — subjects 1–5 plus the five the
//! issue names as trapped — trapped only because the inner loop's Nelder–Mead fallback
//! certified two of its EBEs as converged at gradient norms of 57.8 and 19.0 early in the
//! fit, and the warm start carried them. With that fallback certified on stationarity the
//! same fit goes straight to its optimum (−148.6795) and never traps, so it can no longer
//! exercise the restart. Subjects 5, 20, 40, 74, 83, 106, 159, 167, 175 and 199 (one of
//! 3 traps in 60 random 10-subject draws from `mm_n200.csv`) trap on a genuine secondary
//! mode instead. The trapped leg and both tests' warnings are identical before and after
//! #1527 to every digit quoted below; the restarted leg ends at −79.1679 with #1527 and at
//! −79.1691 without it.
//!
//! **And again with #2093.** Since a curvature-free inner first step is bounded and fallback
//! EBEs meet `inner_tol`, that subset goes straight to its optimum (−79.1679) and never traps.
//! Subjects 5, 9, 49, 69, 106, 115, 130, 142, 156 and 174 — the one trap in 60 seeded
//! 10-subject draws from `mm_n200.csv` (`random.Random(1565)`) — do: a genuine secondary mode,
//! caught by the mid-leg basin probe at the trapped leg's 33rd evaluation.
//!
//! **And again with #2441.** With the absolute `1e-12` residual-variance floor gone, the #2093
//! subset no longer traps on either platform — no row sits on the floor at the initial
//! estimates, so it is a search-path effect, not a change to the trap the gate exists for.
//! The #2441 subset was subjects 7, 52, 59, 75, 86, 126, 129, 175, 176 and 193: draw 32 of
//! 80 seeded 10-subject draws from `mm_n200.csv` (`random.Random(2441)`, `sample(ids, 10)`
//! sorted). 8 of the 80 trapped on Linux x86_64 and 8 on native macOS arm64, 7 on both, and
//! of those this draw (with draw 48) agreed between the platforms in every printed digit:
//! restart notice cold OFV 68.083107 (macOS) / 68.083108 (Linux) at the 33rd evaluation,
//! restarted OFV 63.287990 on both; at `maxiter = 3` best-seen 75.1577, cold 68.0963, gap
//! 7.0614, one subject, on both. Measured at `72e1c6ef` + #2441, macOS in `release`, Linux
//! in `ci-test` via `tools/linux-test.sh`.
//!
//! **And again with #2549.** With every warm start predicted from the incumbent's exact mode
//! response, the inner modes track their branch closely and the #2441 subset no longer traps
//! (it reaches 63.2878 without a restart): 1 of 80 seeded draws traps, against 8 of 80 before.
//! The current subset is that one: subjects 28, 65, 70, 72, 85, 95, 98, 119, 132 and 199, draw
//! 20 of 80 from `mm_n200.csv` (`random.Random(2549)`, `sample(ids, 10)` sorted). Measured on
//! Linux x86_64: restart notice cold OFV −22.901309 at the 33rd evaluation, restarted OFV
//! −25.371534; at `maxiter = 3` best-seen −15.8526, cold −22.7681, gap 6.9154, one subject.
//!
//! **The #2093 subset trapped on macOS arm64 too (#2070).** The #1527 subset did not: native macOS
//! went straight to −79.1679 on its first leg (the #1688 libm split, not a defect in the
//! gate), so both tests were red there and the restart path ran only on Linux. The #2093
//! subset was re-measured natively on macOS arm64 at `3909a294`, debug and release: both
//! tests pass, and every quoted digit matches the Linux numbers — cold 17.064785 at the 33rd
//! evaluation, restarted OFV 11.5047986; at `maxiter = 3` best-seen 26.9989, cold 17.0007,
//! gap 9.9982. Both tests pass on Linux aarch64 (`tools/linux-test.sh`, glibc 2.41) at the
//! same commit. Linux stays the reference for the bands; if a future re-pick is needed,
//! intersect the trapping draws on both platforms rather than taking a Linux-only one.
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

/// The restart path. The trapped leg is stopped by the mid-leg basin probe after 33
/// evaluations, at a point whose cold re-solve scores −22.9013, below what the optimizer saw
/// there; the run is restarted from it — its first inner solve is cold — and converges at
/// −25.3715, 2.47 below the trapped leg's cold objective, with no gap left for the end-of-leg
/// check to find.
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
    // The trapped basin, not a digit string: the cold objective moves in its sixth decimal
    // with any rounding-level change to the gradient, while the basin it identifies sits 2.47
    // above the optimum (−25.3715) — over 100× the band.
    let cold = notice
        .message
        .split("cold OFV = ")
        .nth(1)
        .and_then(|t| t.split_whitespace().next())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("the notice quotes the cold objective: {}", notice.message));
    assert!(
        (cold - (-22.901)).abs() < 0.02,
        "the notice names the trapped leg's cold objective (≈ −22.901): {}",
        notice.message
    );

    assert!(
        result.ofv < -25.2,
        "expected the restart to reach the optimum (−25.3715); the trapped leg's cold \
         objective is ≈ −22.901. Got {}",
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

/// The no-restart path. At `maxiter = 3` (a 27-evaluation budget, under the probe's first
/// check at 32) the first leg spends its budget at a trapped point — one subject in the
/// higher mode — so there is no budget left to restart on. Before #1565 such a fit reported
/// `converged = true`; the gap must surface as a warning and a non-converged fit.
#[test]
fn a_basin_shift_with_no_budget_left_is_not_reported_converged() {
    let result = fit_fixture("tests/fixtures/mm_basin_shift/mm_basin10.csv", Some(3));

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
        // Measured 6.9154 (best-seen −15.8526, cold −22.7681); a band, since the sixth decimal
        // follows the rounding.
        (best - cold - 6.9154).abs() < 0.02,
        "measured gap ≈ 6.92 (best-seen ≈ −15.85, cold ≈ −22.77): got {best} vs {cold}"
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
