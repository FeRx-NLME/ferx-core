//! #2707 — a basin-shift restart that reaches the optimum is reported converged, even when
//! its own step in θ is small.
//!
//! A basin-shift restart (#1565) starts at the previous leg's estimates. Most of its gain is
//! the EBE modes the cold re-solve found, not a step in θ, so it often does not leave *its*
//! start by the #751 escape step. Before #2707 that alone rejected its stop, however sound:
//! both fixtures below ended at the optimum, −3392.235417, with a flat tail, a cold re-solve
//! that reproduced the best seen and no subject off its modes, and were reported
//! `converged: false` with `left init = false` the only failing check.
//!
//! Now such a restart is handed the #2506 start probe, which restarts from a perturbation of
//! the restart's point and asks whether the optimizer comes back. On these fixtures the
//! probe's own leg lands in another EBE basin (−3379.69, four subjects in a higher mode), so
//! it runs the fit's restart chain before its verdict is read; it then converges back to
//! −3392.235417 and verifies the point.
//!
//! The fixture (`tests/fixtures/flip_flop_pop/`) is a simulated one-compartment oral
//! population in which 24 of 80 subjects are flip-flop (`sim_flip_flop_pop.py`, plain Python,
//! seed 21, re-picked in #3153 from seed 2096, whose closed form stopped trapping). The closed
//! form (`flip_flop_pop.ferx`) stops through the #2462 verified-stop
//! branch and the ODE twin (`flip_flop_pop_ode.ferx`) through the plateau branch, so between
//! them they cover both acceptance rules that read the displacement.
//!
//! The two converged fits are Tier 3 and gated with `slow-tests`. They take 0.8 s together in
//! a local release build, but 15 s (closed form) and 81 s (ODE twin) in `Tests + coverage
//! (core)`'s instrumented `ci-cov` profile (PR #2821, run 42619572756), so the local figure
//! is not the cost a PR pays. What stays per-PR: the budget test below, which reaches the
//! probe and its restart chain through `restart_leg` on every PR, and the Tier-1 tests in
//! `outer_optimizer_tests.rs` for the gate (`start_probe_due`) and the verdict
//! (`settle_restart_probe`, the provenance wording). Numbers measured on Linux x86_64; the
//! macOS arm64 numbers are still to be measured on the #3153 seed.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitResult, OmegaMatrix, WarningCode};
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const DATA: &str = "tests/fixtures/flip_flop_pop/flip_flop_pop.csv";
const CLOSED_FORM: &str = "tests/fixtures/flip_flop_pop/flip_flop_pop.ferx";
const ODE_TWIN: &str = "tests/fixtures/flip_flop_pop/flip_flop_pop_ode.ferx";
/// The optimum both fixtures reach, before #2707 (unconverged) and after it.
const OPTIMUM: f64 = -3392.235417;

fn fit_example(model: &str) -> FitResult {
    let text = std::fs::read_to_string(model).expect("example model must exist");
    let parsed = parse_full_model(&text).expect("example model must parse");
    let population = read_nonmem_csv(Path::new(DATA), None, None).expect("example data must load");
    let mut opts = parsed.fit_options.clone();
    opts.verbose = false;
    fit(&parsed.model, &population, &parsed.model.default_params, &opts)
        .expect("the example fit returns a result")
}

/// The cold objective the basin-shift restart notice quotes (`cold OFV = …`).
fn quoted_cold(message: &str) -> f64 {
    message
        .split("cold OFV = ")
        .nth(1)
        .and_then(|t| t.split_whitespace().next())
        .and_then(|v| v.trim_end_matches(',').parse::<f64>().ok())
        .unwrap_or_else(|| panic!("the notice quotes the cold objective: {message}"))
}

/// `cold_ofv` is the trapped first leg's cold re-solve, which the fit's own restart notice
/// names; it identifies the restart the fit ran, as opposed to one of the probe's.
///
/// Which point the verification reports, the probe's or the restart's own, is not pinned
/// here: the two end within 1.7e-8 (closed form) and 1.5e-7 (ODE twin) of each other on an
/// OFV of −3051 (relative 1e-11, measured on macOS arm64 on the previous fixture seed), and `settle_restart_probe` takes
/// the lower. The winner flips with the platform and the build: on CI the closed form
/// reported the resumed run where Linux locally had reported the probe's. Both clauses and
/// both outcomes of that choice are pinned deterministically in
/// `outer_optimizer_tests.rs` (`settle_restart_probe`, the provenance wording); what this
/// test pins is that the notice names exactly one of them and the fit is reported
/// converged.
///
/// `probe_less_evals` is what the same fit spends with the restart's probe switched off; the
/// fit must spend more, or it was reported converged without the probe that this tests.
/// Measured with the probe gate forced shut: 73 (closed form) and 112 (ODE twin) on Linux
/// x86_64 (macOS arm64 not yet measured on the #3153 seed). With the probe the fits spend
/// 156 / 208, so the margin is 83 and 96 evaluations.
fn assert_restart_reported_converged(model: &str, cold_ofv: f64, probe_less_evals: usize) {
    let result = fit_example(model);
    assert!(result.ofv.is_finite(), "{model}: OFV {}", result.ofv);
    // The straddle: the fit must still go through a basin-shift restart, or this tests a
    // fit that never needed the fix.
    let notices: Vec<&str> = result
        .warnings_structured
        .iter()
        .filter(|w| w.message.contains("higher individual mode"))
        .map(|w| w.message.as_str())
        .collect();
    assert_eq!(
        notices.len(),
        1,
        "{model}: exactly one restart notice, the fit's own — the probe's restarts are not \
         announced: {notices:?}"
    );
    let cold = quoted_cold(notices[0]);
    let (probe_clause, resumed_clause) = (
        "the resumed run did not leave its start, and the reported estimates are from a \
         restart from a perturbation of that point, which reached a lower objective.",
        "the reported estimates are from the resumed run.",
    );
    let names_probe = notices[0].ends_with(probe_clause) && !notices[0].contains(resumed_clause);
    let names_resumed = notices[0].ends_with(resumed_clause) && !notices[0].contains(probe_clause);
    assert!(
        names_probe != names_resumed,
        "{model}: the notice names exactly one run the estimates come from: {}",
        notices[0]
    );
    eprintln!(
        "MEASURE #2707 {model}: cold {cold:.6} (|Δ| {:.3e}), OFV {:.9} (|Δ| {:.3e}), {} evaluations",
        (cold - cold_ofv).abs(),
        result.ofv,
        (result.ofv - OPTIMUM).abs(),
        result.n_iterations
    );
    // The leg, not a digit string. Realised |Δ| 0 on both fixtures on Linux x86_64 (the notice
    // prints 6 decimals; the trapped first leg runs before any restart, so the platforms'
    // choice of the winning point cannot reach it); the probe's own restarts quote −3386.45
    // and −3391.69, at least 0.4 away.
    assert!(
        cold.is_finite() && (cold - cold_ofv).abs() < 1e-3,
        "{model}: the notice names the fit's trapped leg (cold ≈ {cold_ofv}), not a probe \
         leg (≈ −3386.45 / −3391.69): {}",
        notices[0]
    );
    // The probe ran (#2821 review r1): a restart leg that moved past its own displacement
    // rule would converge without it and keep every other assertion here green.
    assert!(
        result.n_iterations > probe_less_evals,
        "{model}: {} evaluations, no more than the {probe_less_evals} the fit spends without \
         the restart's probe",
        result.n_iterations
    );
    // Realised 2.0e-7 (ODE twin) and 1.9e-7 (closed form) on Linux x86_64; the bound has 50×
    // headroom over the worst.
    // `main` ends at the same OFV, so this pins that the verification moved no estimate
    // materially, not that it found a better one.
    assert!(
        (result.ofv - OPTIMUM).abs() < 1e-5,
        "{model}: the optimum ({OPTIMUM}); got {}",
        result.ofv
    );
    assert!(
        result.converged,
        "{model}: a restart verified by the start probe is reported converged: {:?}",
        result.warnings
    );
    assert_eq!(result.left_init, Some(true), "{model}: the estimates left the user's start");
    assert!(!result.stationary_start, "{model}");
    assert!(
        !result.warnings_structured.iter().any(|w| w.category == WarningCode::StalledAtInit),
        "{model}: a recovered fit is not a stall at the inits: {:?}",
        result.warnings
    );
}

#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn a_verified_basin_shift_restart_is_converged_closed_form() {
    assert_restart_reported_converged(CLOSED_FORM, -3392.136568, 73);
}

#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn a_verified_basin_shift_restart_is_converged_ode_twin() {
    assert_restart_reported_converged(ODE_TWIN, -3392.136568, 112);
}

/// #2905: a refit from the fit's own estimates is a warm start at the optimum. Its first
/// attempt does not leave its start, and its #2506 probe lands in another EBE basin
/// (−3047.527166, six subjects), as a restart's probe does. With a single leg that probe
/// returned no verdict, so the refit carried `W_STALLED_AT_INIT`, and the ODE twin's was
/// `converged: false`. Now the probe runs the restart chain, comes back to the optimum and
/// verifies it (macOS arm64: 47 → 123 evaluations closed form, 63 → 137 ODE twin).
fn assert_warm_refit_verified(model: &str) {
    let text = std::fs::read_to_string(model).expect("model");
    let parsed = parse_full_model(&text).expect("model parses");
    let population = read_nonmem_csv(Path::new(DATA), None, None).expect("data loads");
    let mut opts = parsed.fit_options.clone();
    opts.verbose = false;
    opts.run_covariance_step = false;
    let cold = fit(&parsed.model, &population, &parsed.model.default_params, &opts).expect("cold");
    assert!((cold.ofv - OPTIMUM).abs() < 1e-5, "{model}: precondition, cold {}", cold.ofv);
    let mut warm = parsed.model.default_params.clone();
    warm.theta = cold.theta.clone();
    let (names, diagonal) = (warm.omega.eta_names.clone(), warm.omega.diagonal);
    warm.omega = OmegaMatrix::from_matrix(cold.omega.clone(), names, diagonal);
    warm.sigma.values = cold.sigma.clone();
    let r = fit(&parsed.model, &population, &warm, &opts).expect("warm");
    assert!(r.converged && r.stationary_start, "{model}: {:?}", r.warnings);
    assert!((r.ofv - OPTIMUM).abs() < 1e-5, "{model}: {}", r.ofv);
    // The probe's chain ran: without it the refits end at 47 / 63 evaluations.
    assert!(r.n_iterations > 90, "{model}: {} evaluations", r.n_iterations);
    assert!(
        !r.warnings_structured.iter().any(|w| w.category == WarningCode::StalledAtInit),
        "{model}: {:?}",
        r.warnings
    );
}

#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn a_warm_refit_whose_probe_shifts_basin_is_verified() {
    assert_warm_refit_verified(CLOSED_FORM);
    assert_warm_refit_verified(ODE_TWIN);
}

/// The probe and its restart chain run on what is left of the fit's `maxiter` budget, not on
/// a fresh one: every restart leg inside the probe counts the evaluations spent before the
/// probe began. `maxiter = 14` is a budget of 112 evaluations (L-BFGS, 7 free parameters:
/// `14 × (7 + 1)`). The fit's own legs end at 73, so the probe starts with 39 left and is cut
/// by the budget: the fit ends at exactly 112, unconverged. With the probe's legs counting
/// only the probe's own evaluations, the fit ran to its full 156 on the previous fixture's
/// seed (to 197 there, at every budget from 128 to 176; measured, Linux x86_64).
#[test]
fn the_restart_probe_spends_only_what_is_left_of_the_fits_budget() {
    let text = std::fs::read_to_string(CLOSED_FORM).expect("model");
    let parsed = parse_full_model(&text).expect("model parses");
    let population = read_nonmem_csv(Path::new(DATA), None, None).expect("data loads");
    let mut opts = parsed.fit_options.clone();
    opts.verbose = false;
    opts.outer_maxiter = 14;
    opts.run_covariance_step = false;
    let result = fit(&parsed.model, &population, &parsed.model.default_params, &opts)
        .expect("the example fit returns a result");
    let budget = 14 * (7 + 1);
    eprintln!(
        "MEASURE #2707 budget {budget}: {} evaluations, converged {}",
        result.n_iterations, result.converged
    );
    // The straddle: the fit must have reached the probe and been cut by the budget. Its own
    // legs end at 73 evaluations, so a fit without the probe stops far below the budget, and
    // reaching it can only be the probe's legs spending what was left. Realised 112 with
    // `converged: false` on Linux x86_64.
    assert!(!result.converged, "the probe was cut by the budget, so the fit is not converged");
    assert!(
        result.n_iterations >= budget,
        "the probe spent what was left of the budget ({} evaluations, budget {budget})",
        result.n_iterations
    );
    // L-BFGS checks `maxeval` only between line searches, so a run can end a few
    // evaluations past it (2 measured at `maxiter = 18`); one iteration's worth of slack.
    assert!(
        result.n_iterations <= budget + 8,
        "{} evaluations against a budget of {budget}",
        result.n_iterations
    );
    // The probe spent the budget the restart's own leg stopped short of, so the fit reports
    // it spent and gives the advice (#2821 review r1); before, it reported neither.
    assert!(
        result.eval_budget_spent,
        "a budget spent by the probe is the fit's: {:?}",
        result.warnings
    );
    let advice: Vec<&String> =
        result.warnings.iter().filter(|w| w.contains("increase maxiter")).collect();
    assert_eq!(advice.len(), 1, "the fit gives the budget advice once: {:?}", result.warnings);
    assert!(
        advice[0].contains("(maxiter = 14)"),
        "the advice names the fit's budget: {}",
        advice[0]
    );
}
