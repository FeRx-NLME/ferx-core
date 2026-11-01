//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **continuous
//! (percentage) titration under a time-varying covariate** — ferx-core's
//! vancomycin trough-TDM path when a declining renal covariate drives clearance
//! (epic #391 / **#700**).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor
//! for this feature family is **mrgsolve** (which does do feedback dosing). This
//! is the time-varying-covariate analogue of the constant-covariate vanco anchor
//! (`adaptive_vanco_anchor.rs`); the reference kit lives in
//! `tests/reference/vanco_renal_mrgsolve/`:
//!
//!   - `vanco_renal_mrgsolve.R` builds the identical structural model (a 1-cpt IV
//!     vancomycin model, once-daily 1-h infusion, `CL = TVCL·CRCL/100`) in
//!     mrgsolve 1.7.2 and runs an R loop whose controller mirrors ferx's
//!     continuous-titration semantics exactly (read the latent pre-dose trough,
//!     first-matching rule wins, `increase 25%` / `decrease 25%` scale the running
//!     dose by 1.25 / 0.75 and clamp to `dose_bounds`, no match re-issues). CRCL
//!     declines 120 → 40 mL/min across the horizon, and — crucially — each day's
//!     window is integrated with CL fixed from the covariate at the window's *end*
//!     (the NONMEM end-of-interval convention ferx's per-segment PK uses). That
//!     includes the 1-h infusion sub-window: its terminating break is not a data
//!     record, so the next obs record governs it too (#1073).
//!   - `vanco_renal_subject.csv` is the dose-free subject: EVID=0 trough rows at
//!     t = 0, 24, …, 312 h carrying the declining `CRCL` column (auto-detected by
//!     `read_nonmem_csv` into each subject's `obs_covariates`).
//!   - `expected.md` is the frozen realized dose ladder (regenerate with
//!     `Rscript vanco_renal_mrgsolve.R`).
//!
//! ## What this anchors — per-event covariate handling under feedback dosing
//!
//! The model under test is `examples/adaptive_vanco_renal.ferx`. With negligible
//! IIV it is effectively a typical-value patient whose *only* source of PK
//! variation is the CRCL trajectory. ferx's reactive driver (RK45) recomputes PK
//! per integration segment from the covariate at the data record that GOVERNS that
//! segment (#700, #1073) — resolving CL for segment `(t_{k-1}, t_k]` from CRCL at
//! `t_k`, and for a segment ending at a non-record break (the infusion end) from
//! the next record ahead — and mrgsolve's R loop (LSODA) feeds the identical
//! piecewise-constant CL. So the two engines see
//! the *same* trough trajectory and reach the *same* dose decisions. As renal
//! function declines, CL falls and drug accumulates: the empiric start is
//! subtherapeutic so the controller titrates *up* early (decisions 0–4), the
//! trough climbs into `[10, 15]` mg/L, then overshoots and the controller titrates
//! *back down* (the `decrease` rung fires 3×) — a genuinely reactive ladder the
//! constant-covariate anchor never exercises.
//!
//! ferx reproduces every dose **exactly** (the titration is exact f64 arithmetic
//! shared with the R loop) and every trough to a small cross-solver tolerance.
//! This is the key result: it proves the per-event covariate handling under
//! feedback dosing matches an independent engine.
//!
//! The example declares no `auc_target`; `vanco_renal_window_aucs_match_mrgsolve` adds one
//! and checks each window's AUC24 against the kit's AUC compartment (#943), so the exposure
//! integral under the per-segment covariate snapshot is anchored too.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, simulate_adaptive_from_spec, AdaptiveSimulateOptions};
use std::path::Path;

mod common;

/// mrgsolve 1.7.2 reference, frozen in
/// `tests/reference/vanco_renal_mrgsolve/expected.md`. Per daily decision:
/// `(time_h, trough_mg_per_L, dose_mg)`. The doses are exact f64 (`500` scaled by
/// `1.25` / `0.75` steps within `dose_bounds`), so ferx matches them bit-for-bit;
/// the troughs are the mrgsolve LSODA values (ferx's RK45 agrees within
/// `SIGNAL_TOL`).
const REF_LADDER: [(f64, f64, f64); 14] = [
    (0.0, 0.000000, 625.0),
    (24.0, 2.023063, 781.25),
    (48.0, 3.299133, 976.5625),
    (72.0, 4.877604, 1220.703125),
    (96.0, 7.122832, 1525.87890625),
    (120.0, 10.421810, 1525.87890625),
    (144.0, 13.187750, 1525.87890625),
    (168.0, 15.843894, 1144.4091796875),
    (192.0, 15.872606, 858.306884765625),
    (216.0, 14.849027, 858.306884765625),
    (240.0, 14.976323, 858.306884765625),
    (264.0, 15.597261, 643.7301635742188),
    (288.0, 14.506703, 643.7301635742188),
    (312.0, 14.005751, 643.7301635742188),
];

/// The declining covariate makes the ladder reactive in *both* directions.
/// Decisions 0..=4 fire the `increase 25%` rung (subtherapeutic under high CRCL);
/// decisions 7, 8, 11 fire the `decrease 25%` rung (trough overshoots as CL falls
/// and drug accumulates); the rest re-issue the running dose (recorded by route).
const N_INCREASE_RULE_FIRINGS: usize = 5;
const N_DECREASE_RULE_FIRINGS: usize = 3;
/// The decision indices (0-based) at which the `decrease` rung fires.
const DECREASE_DECISIONS: [usize; 3] = [7, 8, 11];

/// The `n_increases` / `n_decreases` *metrics* count realized dose step-ups /
/// -downs by dose-delta. `n_increases` is 4 — one fewer than the 5 increase-rule
/// firings, because decision 0's increase steps off the un-realized `start_dose`
/// (500 → 625 mg). `n_decreases` is 3 (all three `decrease` firings are realized
/// step-downs off already-realized doses).
const N_DOSE_INCREASES: usize = 4;
const N_DOSE_DECREASES: usize = 3;

/// Cross-solver trough tolerance (mg/L); see the module note. ferx (RK45) vs
/// mrgsolve (LSODA) integrate the same piecewise-constant system, so the observed
/// gap is small (~1e-3); this is a safe but meaningful guard — a trajectory
/// mismatch from a broken covariate hand-off would blow far past it.
const SIGNAL_TOL: f64 = 0.02;

/// Fast guard: a typo in `adaptive_vanco_renal.ferx` — which exercises the #700
/// time-varying-covariate adaptive surface — fails here, by name, rather than as a mismatch in
/// the cross-engine check below. `parse_full_model_file` runs the full `[adaptive_dosing]`
/// `validate()`, so a successful parse plus these spot-checks pin the scenario the anchor
/// relies on. It pins that the example declares no `auc_target` (the window-AUC test adds
/// one, #943).
#[test]
fn vanco_renal_example_parses_and_pins_the_ladder() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal.ferx"))
        .expect("vanco renal model must parse");
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
        spec.auc_target, None,
        "the renal example declares no auc_target; the window-AUC test adds one (#943)"
    );
    assert_eq!(spec.at.len(), 14, "14 daily decisions, t = 0..=312 h");
    // Continuous titration ⇒ percentage steps, no discrete ladder.
    assert_eq!(spec.levels, None, "continuous (percentage) titration");
    assert_eq!(
        spec.rules.len(),
        2,
        "increase / decrease rungs (the covariate decline drives both)"
    );
}

#[test]
fn vanco_renal_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal.ferx"))
        .expect("vanco renal model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    // `read_nonmem_csv` auto-detects the (non-standard) `CRCL` column and, because
    // it varies across rows, populates each subject's `obs_covariates` — the
    // per-event snapshots the reactive driver resolves PK from (#700).
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_renal_mrgsolve/vanco_renal_subject.csv"),
        None,
        None,
    )
    .expect("vanco renal subject data must load");
    // Sanity: the covariate really is time-varying (else this test wouldn't bite —
    // a constant CRCL would take the frozen-snapshot path, not the per-event one).
    assert!(
        pop.subjects[0].has_tv_covariates(),
        "CRCL must be read as a time-varying covariate (per-event PK path)"
    );

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
    .expect("adaptive vanco renal sim runs (+ frozen-replay verifier)");

    // Every decision dosed (no hold/stop here), so the ledger is 14 long.
    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");

    // Per-decision: the trough matches mrgsolve within the cross-solver band, and
    // the realized dose matches the frozen ladder exactly (exact f64 titration).
    // This is the dose-for-dose match under a time-varying covariate.
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
             a broken time-varying-covariate hand-off",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // The rungs fire exactly where the declining covariate drives them: increases
    // early (0..=4), decreases as drug accumulates (7, 8, 11), re-issue otherwise.
    for (i, e) in res.ledger.iter().enumerate() {
        if i <= 4 {
            assert!(
                e.rule_fired.contains("increase"),
                "decision {i} should fire the increase rung, got {:?}",
                e.rule_fired
            );
        } else if DECREASE_DECISIONS.contains(&i) {
            assert!(
                e.rule_fired.contains("decrease"),
                "decision {i} should fire the decrease rung, got {:?}",
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
        "four realized step-ups (the increase rule fired {N_INCREASE_RULE_FIRINGS}×, \
         but the first steps off the un-realized start_dose)"
    );
    assert_eq!(
        m.n_decreases, N_DOSE_DECREASES,
        "three realized step-downs (the decrease rule fired {N_DECREASE_RULE_FIRINGS}×)"
    );
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.time_to_discontinuation, None);

    // Point metric: troughs in [10, 15] on 6 of 14 decisions (5, 6, 9, 10, 12, 13).
    assert!(
        (m.pct_time_in_window.expect("trough window declared") - 6.0 / 14.0).abs() < 1e-9,
        "pct_time_in_window {:?}",
        m.pct_time_in_window
    );

    // The example declares no `auc_target`, so no exposure metric is reported.
    assert_eq!(
        m.auc_target_attainment, None,
        "no auc_target on the renal (time-varying-covariate) example"
    );
}

/// mrgsolve's AUC24 of each closed inter-decision window (#943), frozen in
/// `tests/reference/vanco_renal_mrgsolve/expected.md`: an AUC compartment `dxdt_AUC = CENT/V` reset at the
/// start of each window, integrated on the same piecewise-constant CL as the troughs above.
const REF_AUC24: [f64; 13] = [
    100.685858, 157.213992, 216.909383, 295.762742, 404.474585, 479.633720, 547.244753, 528.755624,
    479.690442, 471.179557, 481.328460, 445.716362, 427.378953,
];

/// Relative band of each ferx AUC24 against mrgsolve's (#943), the band of the constant-path
/// vanco anchor. Measured on Linux aarch64 (`tools/linux-test.sh`), identical in every
/// printed digit on macOS arm64: the windows read −2.7e-6 to +2.3e-7 relative, so 5e-5 is at least
/// 19× headroom. This is a cross-engine band (RK45 + a trapezoid on soft samples vs
/// LSODA + an AUC compartment), not ferx's own accuracy; the kit has a constant `V` and no
/// `[scaling]`, so it checks the exposure of the per-segment trajectory, and the readout
/// conversion under κ and covariates is pinned by the closed forms in
/// `src/api/tests/adaptive_auc_tv_tests.rs`.
const AUC_REL_TOL: f64 = 5e-5;

/// Each window's AUC24 matches mrgsolve's to [`AUC_REL_TOL`] (#943): the exposure integral
/// under the per-segment snapshot, checked window by window rather than through the
/// attainment fraction.
#[test]
fn vanco_renal_window_aucs_match_mrgsolve() {
    common::assert_window_aucs_match(&windows_in, &REF_AUC24, AUC_REL_TOL, "renal");
}

/// Run the anchor scenario with `auc_target = band` and count the windows in it.
fn windows_in(band: (f64, f64)) -> usize {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal.ferx"))
        .expect("model must parse");
    let mut spec = parsed
        .adaptive_dosing
        .clone()
        .expect("[adaptive_dosing] present");
    spec.auc_target = Some(band);
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_renal_mrgsolve/vanco_renal_subject.csv"),
        None,
        None,
    )
    .expect("vanco renal subject data must load");
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        &spec,
        &opts,
    )
    .expect("adaptive run with auc_target");
    common::windows_in_band(&res)
}
