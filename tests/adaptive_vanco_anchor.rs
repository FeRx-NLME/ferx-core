//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **continuous
//! (percentage) titration** *and* its metrics-only AUC machinery — ferx-core's
//! vancomycin AUC-guided TDM path (epic #391 S2.5b, parent #584).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor
//! for this feature family is **mrgsolve** (which does do feedback dosing). The
//! reference kit lives in `tests/reference/vanco_mrgsolve/`:
//!
//!   - `vanco_mrgsolve.R` builds the identical structural model (a 1-cpt IV
//!     vancomycin model, once-daily 1-h infusion) in mrgsolve 1.7.2 and runs an R
//!     loop whose controller mirrors ferx's continuous-titration semantics exactly
//!     (read the latent pre-dose trough, first-matching rule wins, `increase 25%`
//!     / `decrease 25%` scale the running dose by 1.25 / 0.75 and clamp to
//!     `dose_bounds`, no match re-issues the running dose). Each day's AUC24 is
//!     integrated by a dedicated AUC compartment.
//!   - `expected.md` is the frozen realized dose ladder + per-window AUC24 +
//!     attainment (regenerate with `Rscript vanco_mrgsolve.R`).
//!
//! ## What this anchors — the reactive decisions and the exposure metric
//!
//! The model under test is `examples/adaptive_vanco_auc.ferx`. With negligible IIV
//! it is effectively a typical-value patient, so ferx's reactive driver (RK45) and
//! mrgsolve's R loop (LSODA) integrate the *same* ODE system and the controller
//! sees the *same* trough trajectory — they must therefore reach the *same* dose
//! decisions. The empiric start is subtherapeutic, the controller titrates the
//! dose up to keep the trough in `[10, 15]` mg/L, and the daily AUC24 climbs
//! through the `[400, 600]` mg·h/L target band — so `auc_target_attainment` is a
//! non-trivial fraction (the early under-dosed days miss; the converged days hit).
//!
//! ferx reproduces every dose **exactly** (the titration is exact f64 arithmetic
//! shared with the R loop), every trough to a small cross-solver tolerance, and
//! the AUC-target attainment fraction exactly. ferx integrates the exposure by a
//! 128-panel trapezoid per window while mrgsolve uses an AUC compartment; the two
//! agree to ~1e-5 relative, far inside the margin from each day's AUC24 to the
//! band edges, so the in/out classification (hence attainment) is identical. The
//! exact AUC value accuracy is pinned separately by the analytic unit test
//! `adaptive_window_signal_aucs_matches_closed_form` in `src/ode/predictions.rs`.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, simulate_adaptive_from_spec, AdaptiveSimulateOptions};
use std::path::Path;

/// mrgsolve 1.7.2 reference, frozen in `tests/reference/vanco_mrgsolve/expected.md`.
/// Per daily decision: `(time_h, trough_mg_per_L, dose_mg)`. The doses are exact
/// f64 (`500 · 1.25^k`, within `dose_bounds`), so ferx matches them bit-for-bit.
const REF_LADDER: [(f64, f64, f64); 14] = [
    (0.0, 0.000000, 625.0),
    (24.0, 2.412900, 781.25),
    (48.0, 3.742876, 976.5625),
    (72.0, 4.897488, 1220.703125),
    (96.0, 6.187790, 1525.87890625),
    (120.0, 7.754595, 1907.3486328125),
    (144.0, 9.699224, 2384.185791015625),
    (168.0, 12.125832, 2384.185791015625),
    (192.0, 12.856712, 2384.185791015625),
    (216.0, 13.076849, 2384.185791015625),
    (240.0, 13.143153, 2384.185791015625),
    (264.0, 13.163123, 2384.185791015625),
    (288.0, 13.169138, 2384.185791015625),
    (312.0, 13.170950, 2384.185791015625),
];

/// The first 7 decisions (0..=6) fire the `increase 25%` rung; the rest re-issue
/// the converged dose (recorded by route).
const N_INCREASE_RULE_FIRINGS: usize = 7;

/// The `n_increases` *metric* counts realized dose step-ups, which is 6 — one
/// fewer than the rule firings: decision 0's increase steps off the un-realized
/// `start_dose` (500 → 625 mg), so among the 14 realized ledger doses there are
/// only 6 upward deltas (625 → 781 → … → 2384). This is the documented "by dose
/// change, not by which rule fired" semantics of `AdaptiveSubjectMetrics`.
const N_DOSE_INCREASES: usize = 6;

/// Cross-solver trough tolerance (mg/L); see the module note. The observed RK45
/// vs LSODA gap is ~1e-4, so this is a safe but meaningful guard.
const SIGNAL_TOL: f64 = 0.02;

/// Frozen AUC-target attainment: 8 of the 13 closed daily windows have AUC24 in
/// `[400, 600]` (windows 5..=12; the 5 early under-dosed days miss).
const REF_AUC_IN_BAND: usize = 8;
const REF_AUC_WINDOWS: usize = 13;

/// Fast PR-time guard (NOT slow-gated). The cross-engine check below only runs
/// nightly, so without this a typo in `adaptive_vanco_auc.ferx` — which exercises
/// new DSL surface, notably the `auc_target` key — would slip past the per-PR job.
/// `parse_full_model_file` runs the full `[adaptive_dosing]` `validate()`, so a
/// successful parse plus these spot-checks pin the scenario the anchor relies on.
#[test]
fn vanco_example_parses_and_pins_the_scenario() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_auc.ferx"))
        .expect("vanco model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] block present");
    assert_eq!(spec.start_dose, 500.0, "empiric start dose");
    assert_eq!(spec.dose_bounds, (250.0, 4000.0));
    assert_eq!(
        spec.target_window,
        Some((10.0, 15.0)),
        "trough target band (point metric)"
    );
    assert_eq!(
        spec.auc_target,
        Some((400.0, 600.0)),
        "AUC24 exposure band parsed (the `auc_target` key)"
    );
    assert_eq!(spec.at.len(), 14, "14 daily decisions, t = 0..=312 h");
    // Continuous titration ⇒ percentage steps, no discrete ladder.
    assert_eq!(spec.levels, None, "continuous (percentage) titration");
    assert_eq!(spec.rules.len(), 2, "increase / decrease rungs");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored vanco AUC-TDM titration (#391 S2.5b): opt in with --features slow-tests"
)]
fn vanco_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_auc.ferx"))
        .expect("vanco model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_mrgsolve/vanco_subject.csv"),
        None,
        None,
    )
    .expect("vanco subject data must load");

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
    .expect("adaptive vanco sim runs");

    // Every decision dosed (no hold/stop here), so the ledger is 14 long.
    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");

    // Per-decision: the trough matches mrgsolve within the cross-solver band, and
    // the realized dose matches the frozen ladder exactly (exact f64 titration).
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
             (|Δ| {:.6} >= {SIGNAL_TOL}); a trajectory mismatch this large signals \
             a structural/integration bug",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // The first 7 decisions climb (the `increase` rung); the rest re-issue the
    // converged dose (recorded by route — an infusion).
    for (i, e) in res.ledger.iter().enumerate() {
        if i < N_INCREASE_RULE_FIRINGS {
            assert!(
                e.rule_fired.contains("increase"),
                "decision {i} should fire the increase rung, got {:?}",
                e.rule_fired
            );
        } else {
            assert_eq!(
                e.rule_fired, "infuse",
                "decision {i} should re-issue (record the route), got {:?}",
                e.rule_fired
            );
        }
    }

    // Per-subject metrics: a faithful reduction of the realized run.
    let m = &res.metrics[0];
    let dose_sum: f64 = REF_LADDER.iter().map(|&(_, _, d)| d).sum();
    assert!(
        (m.cumulative_dose - dose_sum).abs() < 1e-6,
        "cumulative_dose {} vs ladder sum {dose_sum}",
        m.cumulative_dose
    );
    assert_eq!(m.n_doses, 14);
    assert_eq!(
        m.n_increases, N_DOSE_INCREASES,
        "six realized step-ups (the increase rule fired 7×, but the first steps off \
         the un-realized start_dose), then it holds"
    );
    assert_eq!(m.n_decreases, 0);
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.time_to_discontinuation, None);

    // Point metric: troughs in [10, 15] on 7 of 14 decisions (decisions 7..=13).
    assert!(
        (m.pct_time_in_window.expect("trough window declared") - 7.0 / 14.0).abs() < 1e-9,
        "pct_time_in_window {:?}",
        m.pct_time_in_window
    );

    // Exposure metric (the S2.5b headline): AUC24 in [400, 600] on 8 of the 13
    // closed daily windows. ferx's 128-panel trapezoid classifies the same windows
    // as mrgsolve's AUC compartment, so the attainment fraction is identical.
    let attain = m.auc_target_attainment.expect("auc_target declared");
    assert!(
        (attain - REF_AUC_IN_BAND as f64 / REF_AUC_WINDOWS as f64).abs() < 1e-9,
        "auc_target_attainment {attain} vs mrgsolve {REF_AUC_IN_BAND}/{REF_AUC_WINDOWS}"
    );
}
