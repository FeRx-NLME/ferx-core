//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **continuous
//! (percentage) titration** *and* its metrics-only AUC machinery — ferx-core's
//! vancomycin AUC-guided TDM path (epic #391 S2.5b, parent #584).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor
//! for this feature family is **mrgsolve** (which does do feedback dosing). The
//! reference kit lives in `tests/reference/vanco_mrgsolve/`:
//!
//!   - `vanco_mrgsolve.R` builds the identical structural model (a 1-cpt IV
//!     vancomycin model, once-daily 1-h infusion) in mrgsolve 2.0.1 and runs an R
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
//! The model under test is `examples/adaptive_vanco_auc.ferx`, run at a **supplied
//! η = 0** (`common::simulate_from_spec_at_typical_value`): the typical-value patient
//! mrgsolve simulates. The model's `omega ~ 1e-10` still draws an η of order 1e-5 on CL,
//! and that drift — not either solver — was the whole of a one-sided 3.7e-4 mg/L trough
//! gap and a 1.6e-5 AUC24 under-read when the anchor drew it (#2580). At η = 0 ferx's
//! reactive driver (RK45) and mrgsolve's R loop (LSODA) integrate the *same* ODE system,
//! the controller sees the *same* trough trajectory, and they reach the *same* dose
//! decisions. The empiric start is subtherapeutic, the controller titrates the
//! dose up to keep the trough in `[10, 15]` mg/L, and the daily AUC24 climbs
//! through the `[400, 600]` mg·h/L target band — so `auc_target_attainment` is a
//! non-trivial fraction (the early under-dosed days miss; the converged days hit).
//!
//! ferx reproduces every dose **exactly** (the titration is exact f64 arithmetic
//! shared with the R loop), every trough to [`SIGNAL_TOL`], and the AUC-target
//! attainment fraction exactly. ferx integrates the exposure by a
//! 128-panel trapezoid per window on the driver's own trajectory (#943) while
//! mrgsolve uses an AUC compartment; `vanco_window_aucs_match_mrgsolve` pins each of
//! the 13 daily AUC24 values to [`AUC_REL_TOL`], far inside the margin from each to the
//! band edges, so the in/out classification (hence attainment) is identical. The
//! exact AUC accuracy is pinned separately by the analytic unit tests
//! `adaptive_window_auc_*` in `src/ode/predictions_tests.rs`.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, AdaptiveSimulationResult};
use std::path::Path;

mod common;

/// mrgsolve reference, frozen to 9 decimals in `tests/reference/vanco_mrgsolve/expected.md`.
/// Per daily decision: `(time_h, trough_mg_per_L, dose_mg)`. The doses are exact
/// f64 (`500 · 1.25^k`, within `dose_bounds`), so ferx matches them bit-for-bit.
const REF_LADDER: [(f64, f64, f64); 14] = [
    (0.0, 0.000000000, 625.0),
    (24.0, 2.412899604, 781.25),
    (48.0, 3.742875900, 976.5625),
    (72.0, 4.897488188, 1220.703125),
    (96.0, 6.187789635, 1525.87890625),
    (120.0, 7.754594597, 1907.3486328125),
    (144.0, 9.699224226, 2384.185791015625),
    (168.0, 12.125831719, 2384.185791015625),
    (192.0, 12.856711850, 2384.185791015625),
    (216.0, 13.076848715, 2384.185791015625),
    (240.0, 13.143152665, 2384.185791015625),
    (264.0, 13.163123031, 2384.185791015625),
    (288.0, 13.169137990, 2384.185791015625),
    (312.0, 13.170949660, 2384.185791015625),
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

/// Cross-solver trough band (mg/L), ferx (RK45 at the model's default tolerances) against
/// mrgsolve (LSODA at 1e-10) at η = 0. The worst gap over the 14 decisions, printed by the
/// test, is 1.045e-9 mg/L on Linux aarch64 (`tools/linux-test.sh`) and identical on macOS
/// arm64, so 5e-9 is 4.8× headroom; `expected.md`'s 9 decimals floor the comparison at
/// 5e-10. With the drawn η the gap was 3.71e-4, all one sign (#2580). A dosing defect worth
/// naming moves a trough by orders of magnitude more (one 25% titration step is ~1 mg/L at
/// steady state).
const SIGNAL_TOL: f64 = 5e-9;

/// Frozen AUC-target attainment: 8 of the 13 closed daily windows have AUC24 in
/// `[400, 600]` (windows 5..=12; the 5 early under-dosed days miss).
const REF_AUC_IN_BAND: usize = 8;
const REF_AUC_WINDOWS: usize = 13;

/// mrgsolve's per-window AUC24 (mg·h/L), frozen in `expected.md` to 9 decimals.
const REF_AUC24: [f64; REF_AUC_WINDOWS] = [
    107.992007917,
    168.712974085,
    221.048379232,
    279.369752324,
    350.133627321,
    437.944565617,
    547.514297892,
    581.428845133,
    591.643710453,
    594.720368765,
    595.647040435,
    595.926148572,
    596.010214337,
];

/// Relative band of each ferx AUC24 against mrgsolve's, at η = 0. Measured on Linux aarch64
/// (`tools/linux-test.sh`), identical in every printed digit on macOS arm64, by bisecting
/// each window's AUC through the public band: the worst window is window 0 at −2.75e-6 and
/// the rest lie within ±7.5e-7, so 1e-5 is 3.6× headroom. With the drawn η every window read
/// low by 1.07e-5 to 1.63e-5 (#2580). This is a cross-engine band, not ferx's own accuracy:
/// the engines differ in solver (RK45 vs LSODA) and in quadrature (a 128-panel trapezoid on
/// soft samples vs an AUC compartment), and the remainder is not split between the two.
/// Windows 11 and 12, the closest pair, differ by 1.4e-4 relative, so the band isolates
/// each window.
const AUC_REL_TOL: f64 = 1e-5;

/// Fast guard: a typo in `adaptive_vanco_auc.ferx` — which exercises new DSL surface,
/// notably the `auc_target` key — fails here, by name, rather than as a mismatch in the
/// cross-engine check below. `parse_full_model_file` runs the full `[adaptive_dosing]`
/// `validate()`, so a successful parse plus these spot-checks pin the scenario the
/// anchor relies on.
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
fn vanco_titration_matches_mrgsolve() {
    let res = run(None);

    // Every decision dosed (no hold/stop here), so the ledger is 14 long.
    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");

    // Per-decision: the trough matches mrgsolve within the cross-solver band, and
    // the realized dose matches the frozen ladder exactly (exact f64 titration).
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
             (|Δ| {gap:.3e} >= {SIGNAL_TOL:e}); a trajectory mismatch this large signals \
             a structural/integration bug"
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    eprintln!("vanco: worst trough gap {worst:.3e} mg/L (band {SIGNAL_TOL:e})");

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

/// Run the vanco scenario at η = 0, with `auc_target` replaced by `band` when given.
fn run(band: Option<(f64, f64)>) -> AdaptiveSimulationResult {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_auc.ferx"))
        .expect("vanco model must parse");
    let mut spec = parsed
        .adaptive_dosing
        .clone()
        .expect("[adaptive_dosing] present");
    if let Some(band) = band {
        spec.auc_target = Some(band);
    }
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_mrgsolve/vanco_subject.csv"),
        None,
        None,
    )
    .expect("vanco subject data must load");
    common::simulate_from_spec_at_typical_value(&parsed.model, &pop, &spec)
}

/// Run the vanco scenario with `auc_target = band` and return the number of the 13
/// daily windows whose AUC24 lies in it — the only view of the window AUCs the public
/// API gives.
fn windows_in(band: (f64, f64)) -> usize {
    let res = run(Some(band));
    let attain = res.metrics[0]
        .auc_target_attainment
        .expect("auc_target declared");
    let n = attain * REF_AUC_WINDOWS as f64;
    assert!((n - n.round()).abs() < 1e-9, "attainment {attain} is k/13");
    n.round() as usize
}

/// Each daily AUC24 matches mrgsolve's AUC compartment to [`AUC_REL_TOL`] (#943): a band of
/// that width around window `i`'s reference holds exactly one ferx window.
///
/// The public API reports only how many windows lie in a band, not which, so the test first
/// asserts that the bands are pairwise disjoint: the references rise strictly, and each
/// reference's band ends below the next one's. A ferx window inside window `i`'s band is
/// then window `i` and no other, and the count of 1 identifies it. Widening the band or
/// tightening the spacing past that point fails here, by name, instead of quietly letting
/// two windows swap.
///
/// The attainment count alone (`vanco_titration_matches_mrgsolve`) only sees the two
/// windows nearest the `[400, 600]` edges; a 5% under-estimate of every AUC24 keeps it at
/// 8/13. This test sees each window.
#[test]
fn vanco_window_aucs_match_mrgsolve() {
    let band_of = |r: f64| (r * (1.0 - AUC_REL_TOL), r * (1.0 + AUC_REL_TOL));
    for (i, pair) in REF_AUC24.windows(2).enumerate() {
        assert!(
            band_of(pair[0]).1 < band_of(pair[1]).0,
            "windows {i} and {}: references {} and {} are closer than the band \
             ({AUC_REL_TOL:e}), so a count of 1 no longer identifies the window",
            i + 1,
            pair[0],
            pair[1]
        );
    }
    for (i, &r) in REF_AUC24.iter().enumerate() {
        let band = band_of(r);
        let want = REF_AUC24
            .iter()
            .filter(|&&x| band.0 <= x && x <= band.1)
            .count();
        assert_eq!(want, 1, "window {i}: the bands are disjoint, checked above");
        assert_eq!(
            windows_in(band),
            want,
            "window {i}: ferx's AUC24 is not within {AUC_REL_TOL:e} of mrgsolve's {r}"
        );
    }
}
