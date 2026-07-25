//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **pre-scheduled base
//! regimen** support — ferx-core's loading-dose + maintenance-titration path (issue
//! #702, epic #391).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor for
//! this feature family is **mrgsolve** (which does do feedback dosing). The reference
//! kit lives in `tests/reference/vanco_loading_mrgsolve/`:
//!
//!   - `vanco_loading_mrgsolve.R` builds the identical 1-cpt IV vancomycin model in
//!     mrgsolve 1.7.2, applies a 1500 mg LOADING bolus at t=0, and runs an R loop whose
//!     controller mirrors ferx's continuous-titration semantics exactly (read the latent
//!     pre-dose trough, first-matching rule wins, `increase 25%` / `decrease 25%` scale
//!     the running dose by 1.25 / 0.75 and clamp to `dose_bounds`, no match re-issues).
//!   - `vanco_loading_subject.csv` carries the loading bolus (the base regimen) + the
//!     daily decision grid; `expected.md` is the frozen realized maintenance ladder
//!     (regenerate with `Rscript vanco_loading_mrgsolve.R`).
//!
//! ## What this anchors — the base regimen drives every reactive decision
//!
//! The model under test is `examples/adaptive_vanco_loading.ferx`, driven from the
//! loading-dose subject. The controller's day-1 pre-dose trough (~5.6 mg/L) is the
//! DECAYED LOADING DOSE — nothing else is in the system yet (the first maintenance dose
//! is issued *at* that decision). So a dropped or mis-integrated base regimen (#702)
//! would move that trough to ~0 and every trough after it: the trough comparison below
//! is the base-regimen test. ferx (RK45) and mrgsolve (LSODA) integrate the same ODE, so
//! they reach the same troughs (to a small cross-solver tolerance) and — the titration
//! being exact f64 arithmetic (`750·1.25^k`) with every trough margins away from the rule
//! thresholds — the same maintenance doses bit-for-bit. The loading dose is a base dose,
//! so it never appears in the reactive `ledger` (which holds only controller doses).

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, simulate_adaptive_from_spec, AdaptiveSimulateOptions};
use std::path::Path;

/// mrgsolve 1.7.2 reference, frozen in `tests/reference/vanco_loading_mrgsolve/expected.md`.
/// Per daily MAINTENANCE decision: `(time_h, trough_mg_per_L, dose_mg)`. The doses are
/// exact f64 (`750 · 1.25^k` within `dose_bounds`), so ferx matches them bit-for-bit;
/// the day-1 trough is the decayed 1500 mg loading dose.
const REF_LADDER: [(f64, f64, f64); 6] = [
    (24.0, 5.647391, 937.5),
    (48.0, 5.230581, 1171.875),
    (72.0, 5.987445, 1464.84375),
    (96.0, 7.318415, 1831.0546875),
    (120.0, 9.098053, 2288.818359375),
    (144.0, 11.357516, 2288.818359375),
];

/// The first 5 decisions fire the `increase 25%` rung (troughs below 10); the last
/// re-issues the converged dose (trough in [10, 15]), recorded by route.
const N_INCREASE_RULE_FIRINGS: usize = 5;

/// Cross-solver trough tolerance (mg/L). The model is a pure 1-cpt IV bolus decay, so
/// RK45 and LSODA agree far tighter than this; 0.01 is a safe but meaningful guard.
const SIGNAL_TOL: f64 = 0.01;

/// The pre-scheduled loading dose (mg), carried on the subject, not in the ledger.
const LOADING_DOSE_MG: f64 = 1500.0;

/// Fast PR-time guard (NOT slow-gated). The cross-engine check below only runs nightly,
/// so without this a typo in `adaptive_vanco_loading.ferx` or the loading-dose subject
/// CSV would slip past the per-PR job. `parse_full_model_file` runs the full
/// `[adaptive_dosing]` `validate()`, and loading the CSV pins the base regimen.
#[test]
fn vanco_loading_example_parses_and_pins_the_scenario() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_loading.ferx"))
        .expect("vanco loading model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] block present");
    assert_eq!(spec.start_dose, 750.0, "empiric maintenance start dose");
    assert_eq!(spec.dose_bounds, (250.0, 4000.0));
    assert_eq!(
        spec.target_window,
        Some((10.0, 15.0)),
        "trough target band (point metric)"
    );
    assert_eq!(
        spec.at.len(),
        6,
        "6 daily maintenance decisions, t = 24..=144 h"
    );
    assert_eq!(spec.levels, None, "continuous (percentage) titration");
    assert_eq!(spec.rules.len(), 2, "increase / decrease rungs");

    // The base regimen rides on the subject: a single 1500 mg loading bolus at t=0.
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_loading_mrgsolve/vanco_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco loading subject must load");
    let s = &pop.subjects[0];
    assert_eq!(s.doses.len(), 1, "one pre-scheduled loading dose");
    assert_eq!(s.doses[0].time, 0.0, "loading dose at t=0");
    assert_eq!(s.doses[0].amt, LOADING_DOSE_MG, "1500 mg loading bolus");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored vanco loading-dose base regimen (#702): opt in with --features slow-tests"
)]
fn vanco_loading_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_loading.ferx"))
        .expect("vanco loading model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_loading_mrgsolve/vanco_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco loading subject data must load");

    // `AdaptiveSimulateOptions` is `#[non_exhaustive]`: build from `default()`.
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);

    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("adaptive vanco loading-dose sim runs and passes the base-aware verifier");

    // Six maintenance decisions, each dosed (no hold/stop here). The loading dose is a
    // BASE dose — it drives the trajectory but never enters the reactive ledger.
    assert_eq!(res.decisions.len(), 6, "one decision per day, t = 24..=144");
    assert_eq!(
        res.ledger.len(),
        6,
        "every decision issued a maintenance dose"
    );
    assert!(
        res.ledger.iter().all(|e| e.amt != LOADING_DOSE_MG),
        "the loading dose is a base dose, not a reactive ledger entry"
    );

    // The #702 assertion: the day-1 trough is the decayed loading dose (~5.6 mg/L), not
    // ~0. A dropped base regimen would fail this by a wide margin.
    let first_trough = res.decisions[0]
        .observed_signals
        .first()
        .map(|s| s.value)
        .unwrap_or(f64::NAN);
    assert!(
        first_trough > 5.0,
        "day-1 trough {first_trough:.6} must reflect the integrated loading dose (>5 mg/L); \
         a value near 0 means the base regimen was dropped"
    );

    // Per-decision: the trough matches mrgsolve within the cross-solver band, and the
    // realized maintenance dose matches the frozen ladder exactly (exact f64 titration).
    for (i, &(t_ref, trough_ref, dose_ref)) in REF_LADDER.iter().enumerate() {
        let d = &res.decisions[i];
        assert_eq!(d.time, t_ref, "decision {i} time");
        let sig = d
            .observed_signals
            .first()
            .map(|s| s.value)
            .unwrap_or(f64::NAN);
        assert!(
            (sig - trough_ref).abs() < SIGNAL_TOL,
            "decision {i}: ferx trough {sig:.6} vs mrgsolve {trough_ref:.6} \
             (|Δ| {:.6} >= {SIGNAL_TOL}); a trough mismatch this large signals a dropped \
             or mis-integrated base regimen",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // The first 5 decisions climb (the `increase` rung); the last re-issues the converged
    // dose (recorded by route — a bolus).
    for (i, e) in res.ledger.iter().enumerate() {
        if i < N_INCREASE_RULE_FIRINGS {
            assert!(
                e.rule_fired.contains("increase"),
                "decision {i} should fire the increase rung, got {:?}",
                e.rule_fired
            );
        } else {
            assert_eq!(
                e.rule_fired, "bolus",
                "decision {i} should re-issue (record the route), got {:?}",
                e.rule_fired
            );
        }
    }

    // Per-subject metrics: a faithful reduction of the realized maintenance run.
    let m = &res.metrics[0];
    let dose_sum: f64 = REF_LADDER.iter().map(|&(_, _, d)| d).sum();
    assert!(
        (m.cumulative_dose - dose_sum).abs() < 1e-6,
        "cumulative_dose {} vs ladder sum {dose_sum}",
        m.cumulative_dose
    );
    assert_eq!(m.n_doses, 6);
    assert_eq!(m.n_decreases, 0);
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
}
