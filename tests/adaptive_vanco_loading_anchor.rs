//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **pre-scheduled base
//! regimen** support — ferx-core's loading-dose + maintenance-titration path (issue
//! #702, epic #391).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor for
//! this feature family is **mrgsolve** (which does do feedback dosing). The reference
//! kit lives in `tests/reference/vanco_loading_mrgsolve/`:
//!
//!   - `vanco_loading_mrgsolve.R` builds the identical 1-cpt IV vancomycin model in
//!     mrgsolve 2.0.1, applies a 1500 mg LOADING bolus at t=0, and runs an R loop whose
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
//! is the base-regimen test. The run uses a **supplied η = 0**
//! (`common::simulate_from_spec_at_typical_value`), the typical-value patient mrgsolve
//! simulates: the model's `omega ~ 1e-10` still draws an η of order 1e-5 on CL, and that
//! drift — not either solver — was the whole of a one-sided 3.0e-4 mg/L trough gap when the
//! anchor drew it (#2580). At η = 0 ferx (RK45) and mrgsolve (LSODA) integrate the same ODE,
//! so they reach the same troughs (to [`SIGNAL_TOL`]) and — the titration
//! being exact f64 arithmetic (`750·1.25^k`) with every trough margins away from the rule
//! thresholds — the same maintenance doses bit-for-bit. The loading dose is a base dose,
//! so it never appears in the reactive `ledger` (which holds only controller doses).

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, AdaptiveSimulationResult};
use std::path::Path;

mod common;

/// mrgsolve reference, frozen to 9 decimals in `tests/reference/vanco_loading_mrgsolve/expected.md`.
/// Per daily MAINTENANCE decision: `(time_h, trough_mg_per_L, dose_mg)`. The doses are
/// exact f64 (`750 · 1.25^k` within `dose_bounds`), so ferx matches them bit-for-bit;
/// the day-1 trough is the decayed 1500 mg loading dose.
const REF_LADDER: [(f64, f64, f64); 6] = [
    (24.0, 5.647391473, 937.5),
    (48.0, 5.230581295, 1171.875),
    (72.0, 5.987445399, 1464.84375),
    (96.0, 7.318414634, 1831.0546875),
    (120.0, 9.098052548, 2288.818359375),
    (144.0, 11.357516291, 2288.818359375),
];

/// The first 5 decisions fire the `increase 25%` rung (troughs below 10); the last
/// re-issues the converged dose (trough in [10, 15]), recorded by route.
const N_INCREASE_RULE_FIRINGS: usize = 5;

/// Cross-solver trough band (mg/L), ferx (RK45 at the model's default tolerances) against
/// mrgsolve (LSODA at 1e-10) at η = 0. The worst gap over the 6 decisions, printed by the
/// test, is 6.004e-10 mg/L on Linux aarch64 (`tools/linux-test.sh`) and identical on macOS
/// arm64, so 5e-9 is 8.3× headroom; `expected.md`'s 9 decimals floor the comparison at
/// 5e-10, which is why the headroom is wider than the vanco anchor's. With the drawn η the
/// gap was 2.99e-4, all one sign (#2580). A dropped loading dose moves the day-1 trough by
/// ~5.6 mg/L.
const SIGNAL_TOL: f64 = 5e-9;

/// The pre-scheduled loading dose (mg), carried on the subject, not in the ledger.
const LOADING_DOSE_MG: f64 = 1500.0;

/// Fast guard: a typo in `adaptive_vanco_loading.ferx` or the loading-dose subject CSV fails
/// here, by name, rather than as a mismatch in the cross-engine check below.
/// `parse_full_model_file` runs the full `[adaptive_dosing]` `validate()`, and loading the CSV
/// pins the base regimen.
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
fn vanco_loading_titration_matches_mrgsolve() {
    let res = run(None);

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
    let mut worst: f64 = 0.0;
    for (i, &(t_ref, trough_ref, dose_ref)) in REF_LADDER.iter().enumerate() {
        let d = &res.decisions[i];
        assert_eq!(d.time, t_ref, "decision {i} time");
        let sig = d
            .observed_signals
            .first()
            .map(|s| s.value)
            .unwrap_or(f64::NAN);
        assert!(sig.is_finite(), "decision {i}: trough {sig}");
        let gap = (sig - trough_ref).abs();
        worst = worst.max(gap);
        assert!(
            gap < SIGNAL_TOL,
            "decision {i}: ferx trough {sig:.9} vs mrgsolve {trough_ref:.9} \
             (|Δ| {gap:.3e} >= {SIGNAL_TOL:e}); a trough mismatch this large signals a dropped \
             or mis-integrated base regimen"
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    eprintln!("loading: worst trough gap {worst:.3e} mg/L (band {SIGNAL_TOL:e})");

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

/// mrgsolve's AUC24 of each closed inter-decision window (#943), frozen in
/// `tests/reference/vanco_loading_mrgsolve/expected.md`: an AUC compartment `dxdt_AUC = CENT/V` reset at the
/// start of each window, integrated on the same piecewise-constant CL as the troughs above.
const REF_AUC24: [f64; 5] = [
    242.711203568,
    277.831467910,
    339.591552808,
    422.170913604,
    527.015314969,
];

/// Relative band of each ferx AUC24 against mrgsolve's (#943), at η = 0.
/// Measured on Linux aarch64 (`tools/linux-test.sh`), identical in
/// every printed digit on macOS arm64: every window reads high by 7.32e-6, so 2.5e-5 is 3.4×
/// headroom. That offset is the trapezoid's own error, not the solvers or η: each window is
/// a pure exponential (bolus doses, `k = CL/V = 0.05`/h), on which a 128-panel trapezoid over
/// 24 h over-reads by `(k·h)²/12 = 7.32e-6` with `h = 24/128` h. This is a cross-engine band (RK45 + a trapezoid on
/// soft samples vs LSODA + an AUC compartment), not ferx's own accuracy; the kit has a constant `V` and no
/// `[scaling]`, so it checks the exposure of the per-segment trajectory, and the readout
/// conversion under κ and covariates is pinned by the closed forms in
/// `src/api/tests/adaptive_auc_tv_tests.rs`.
const AUC_REL_TOL: f64 = 2.5e-5;

/// Each window's AUC24 matches mrgsolve's to [`AUC_REL_TOL`] (#943): the exposure integral
/// under the per-segment snapshot, checked window by window rather than through the
/// attainment fraction.
#[test]
fn vanco_loading_window_aucs_match_mrgsolve() {
    common::assert_window_aucs_match(&windows_in, &REF_AUC24, AUC_REL_TOL, "loading");
}

/// Run the anchor scenario with `auc_target = band` and count the windows in it.
fn windows_in(band: (f64, f64)) -> usize {
    common::windows_in_band(&run(Some(band)))
}

/// Run the anchor scenario at η = 0, with `auc_target` replaced by `band` when given.
fn run(band: Option<(f64, f64)>) -> AdaptiveSimulationResult {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_loading.ferx"))
        .expect("vanco loading model must parse");
    let mut spec = parsed
        .adaptive_dosing
        .clone()
        .expect("[adaptive_dosing] present");
    if let Some(band) = band {
        spec.auc_target = Some(band);
    }
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_loading_mrgsolve/vanco_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco loading subject data must load");
    common::simulate_from_spec_at_typical_value(&parsed.model, &pop, &spec)
}
