//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **pre-scheduled base regimen
//! under a time-varying covariate** — ferx-core's loading-dose + maintenance-titration path
//! when a declining renal covariate drives clearance (epic #391 / #702 / #930).
//!
//! This is the union of two shipped anchors: `adaptive_vanco_loading_anchor.rs` (a loading
//! dose rides on the subject, #702) and `adaptive_vanco_renal_anchor.rs` (a declining CRCL
//! drives per-segment clearance, #700). #702 rejected that combination — a base regimen with
//! a time-varying covariate — and #930 lifts it; this anchor is the external check that ferx
//! integrates the loading dose under the *declining* per-segment clearance and reaches the
//! same ladder as an independent engine. NONMEM has no native feedback dosing, so mrgsolve is
//! the comparator. The reference kit lives in `tests/reference/vanco_renal_loading_mrgsolve/`:
//!
//!   - `vanco_renal_loading_mrgsolve.R` builds the identical 1-cpt IV vancomycin model in
//!     mrgsolve 1.7.2, applies a 1500 mg LOADING bolus at t=0, declines CRCL 120 → 41 mL/min
//!     (`CL = TVCL·CRCL/100`), and runs an R loop mirroring ferx's continuous-titration
//!     semantics exactly. Every dose is a bolus, so each day is a single segment integrated
//!     under the NEXT record's CRCL (the NONMEM end-of-interval convention ferx's per-segment
//!     PK uses).
//!   - `vanco_renal_loading_subject.csv` carries the loading bolus (base regimen) + the
//!     declining `CRCL` column on daily trough rows; `expected.md` is the frozen realized
//!     ladder (regenerate with `Rscript vanco_renal_loading_mrgsolve.R`).
//!
//! ## What this anchors — a base dose integrated under a *changing* covariate
//!
//! The model under test is `examples/adaptive_vanco_renal_loading.ferx`. The controller's
//! day-1 pre-dose trough (~5.6 mg/L) is the loading dose DECAYED UNDER CRCL(24) — a base
//! regimen dropped, or frozen at the t=0 covariate, would move it (and every trough after).
//! As CRCL falls, clearance falls and drug accumulates, so the ladder is reactive in *both*
//! directions: increases early (troughs subtherapeutic), then decreases as the trough
//! overshoots. ferx (RK45) and mrgsolve (LSODA) feed the identical piecewise-constant CL, so
//! they reach the same troughs (to a small cross-solver tolerance) and — the titration being
//! exact f64 arithmetic (`750·1.25^k·0.75^m`) — the same maintenance doses bit-for-bit. The
//! loading dose is a base dose, so it never appears in the reactive `ledger`.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, simulate_adaptive_from_spec, AdaptiveSimulateOptions};
use std::path::Path;

/// mrgsolve 1.7.2 reference, frozen in
/// `tests/reference/vanco_renal_loading_mrgsolve/expected.md`. Per daily MAINTENANCE
/// decision: `(time_h, trough_mg_per_L, dose_mg)`. The doses are exact f64 (`750` scaled by
/// `1.25` / `0.75` steps within `dose_bounds`), so ferx matches them bit-for-bit; the troughs
/// are the mrgsolve LSODA values (ferx's RK45 agrees within `SIGNAL_TOL`). The day-1 trough is
/// the 1500 mg loading dose decayed under CRCL(24) = 100 mL/min.
const REF_LADDER: [(f64, f64, f64); 8] = [
    (24.0, 5.647391, 937.5),
    (48.0, 6.262143, 1171.875),
    (72.0, 8.813241, 1464.84375),
    (96.0, 12.889476, 1464.84375),
    (120.0, 16.320448, 1098.6328125),
    (144.0, 16.894268, 823.974609375),
    (168.0, 16.038540, 617.98095703125),
    (192.0, 14.528939, 617.98095703125),
];

/// The declining covariate makes the ladder reactive in *both* directions. Decisions 0..=2
/// fire the `increase 25%` rung (subtherapeutic as the loading dose washes out), decisions
/// 4..=6 fire the `decrease 25%` rung (the trough overshoots as CL falls and drug
/// accumulates), and decisions 3 & 7 re-issue the running dose (recorded by route).
const INCREASE_DECISIONS: [usize; 3] = [0, 1, 2];
const DECREASE_DECISIONS: [usize; 3] = [4, 5, 6];
const REISSUE_DECISIONS: [usize; 2] = [3, 7];

/// The `n_increases` / `n_decreases` metrics count realized dose step-ups / -downs by
/// dose-delta. `n_increases` is 2 — one fewer than the 3 increase-rule firings — because
/// decision 0's increase steps off the un-realized `start_dose` (750 → 937.5). `n_decreases`
/// is 3 (all three `decrease` firings are realized step-downs off already-realized doses).
const N_DOSE_INCREASES: usize = 2;
const N_DOSE_DECREASES: usize = 3;

/// Cross-solver trough tolerance (mg/L); ferx (RK45) vs mrgsolve (LSODA) integrate the same
/// piecewise-constant system, so the observed gap is small (~1e-3). A trajectory mismatch
/// from a broken base-under-covariate hand-off would blow far past it.
const SIGNAL_TOL: f64 = 0.02;

/// The pre-scheduled loading dose (mg), carried on the subject, not in the reactive ledger.
const LOADING_DOSE_MG: f64 = 1500.0;

/// Fast PR-time guard (NOT slow-gated). The cross-engine check below only runs nightly, so
/// without this a typo in `adaptive_vanco_renal_loading.ferx` or its subject CSV — which
/// exercise the #930 base-regimen × time-varying-covariate surface — would slip past the
/// per-PR job. `parse_full_model_file` runs the full `[adaptive_dosing]` `validate()`, and
/// loading the CSV pins the base regimen and the time-varying covariate.
#[test]
fn vanco_renal_loading_example_parses_and_pins_the_scenario() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal_loading.ferx"))
        .expect("vanco renal-loading model must parse");
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
        spec.auc_target, None,
        "auc_target must be absent — it is rejected for a time-varying-covariate subject (#700)"
    );
    assert_eq!(
        spec.at.len(),
        8,
        "8 daily maintenance decisions, t = 24..=192 h"
    );
    assert_eq!(spec.levels, None, "continuous (percentage) titration");
    assert_eq!(spec.rules.len(), 2, "increase / decrease rungs");

    // The base regimen rides on the subject: a single 1500 mg loading bolus at t=0, with the
    // declining CRCL covariate on every record (auto-detected as time-varying).
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_renal_loading_mrgsolve/vanco_renal_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco renal-loading subject must load");
    let s = &pop.subjects[0];
    assert_eq!(s.doses.len(), 1, "one pre-scheduled loading dose");
    assert_eq!(s.doses[0].time, 0.0, "loading dose at t=0");
    assert_eq!(s.doses[0].amt, LOADING_DOSE_MG, "1500 mg loading bolus");
    assert!(
        s.has_tv_covariates(),
        "CRCL must be read as a time-varying covariate (per-event PK path)"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored vanco renal-decline loading-dose base regimen (#930): opt in with --features slow-tests"
)]
fn vanco_renal_loading_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal_loading.ferx"))
        .expect("vanco renal-loading model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    // `read_nonmem_csv` auto-detects the `CRCL` column and, because it varies across rows,
    // populates each subject's per-record covariates — the snapshots the reactive driver
    // resolves PK from (#700), and the per-dose F snapshot for the base dose (#930).
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_renal_loading_mrgsolve/vanco_renal_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco renal-loading subject data must load");
    assert!(
        pop.subjects[0].has_tv_covariates(),
        "CRCL must be time-varying (per-event PK path) — else this test wouldn't bite"
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
    .expect("adaptive vanco renal-loading sim runs (+ base-aware frozen-replay verifier)");

    // Eight maintenance decisions, each dosed (no hold/stop here). The loading dose is a BASE
    // dose — it drives the trajectory but never enters the reactive ledger.
    assert_eq!(res.decisions.len(), 8, "one decision per day, t = 24..=192");
    assert_eq!(
        res.ledger.len(),
        8,
        "every decision issued a maintenance dose"
    );
    assert!(
        res.ledger.iter().all(|e| e.amt != LOADING_DOSE_MG),
        "the loading dose is a base dose, not a reactive ledger entry"
    );

    // The #930 assertion: the day-1 trough is the loading dose decayed under CRCL(24), not ~0
    // and not the t=0-covariate value. A dropped or covariate-frozen base regimen fails this.
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

    // Per-decision: the trough matches mrgsolve within the cross-solver band, and the realized
    // maintenance dose matches the frozen ladder exactly (exact f64 titration). This is the
    // dose-for-dose match of a base regimen under a time-varying covariate.
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
             (|Δ| {:.6} >= {SIGNAL_TOL}); a trough mismatch this large signals a dropped or \
             covariate-frozen base regimen",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // The rungs fire exactly where the declining covariate drives them: increases early
    // (0..=2), decreases as drug accumulates (4..=6), re-issue otherwise (3, 7 — by route).
    for (i, e) in res.ledger.iter().enumerate() {
        if INCREASE_DECISIONS.contains(&i) {
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
            assert!(
                REISSUE_DECISIONS.contains(&i),
                "decision {i} is neither increase/decrease nor an expected reissue"
            );
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
    assert_eq!(m.n_doses, 8);
    assert_eq!(
        m.n_increases, N_DOSE_INCREASES,
        "two realized step-ups (the increase rule fired 3×, but the first steps off the \
         un-realized start_dose)"
    );
    assert_eq!(
        m.n_decreases, N_DOSE_DECREASES,
        "three realized step-downs (the decrease rule fired 3×)"
    );
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.time_to_discontinuation, None);

    // Point metric: troughs in [10, 15] on 2 of 8 decisions (3 and 7).
    assert!(
        (m.pct_time_in_window.expect("trough window declared") - 2.0 / 8.0).abs() < 1e-9,
        "pct_time_in_window {:?}",
        m.pct_time_in_window
    );

    // `auc_target` was intentionally not declared (a typed error on a TV-covariate subject).
    assert_eq!(
        m.auc_target_attainment, None,
        "no auc_target on the renal-loading (time-varying-covariate) example"
    );
}
