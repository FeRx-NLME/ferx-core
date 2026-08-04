//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **pre-scheduled base regimen
//! under inter-occasion variability** — ferx-core's loading-dose + maintenance-titration path
//! when a fresh per-occasion κ drives clearance (epic #391 / #702 / #931).
//!
//! This is the union of two shipped anchors: `adaptive_vanco_loading_anchor.rs` (a loading
//! dose rides on the subject, #702) and `adaptive_vanco_iov_anchor.rs` (a per-occasion κ on CL
//! drives the trajectory, #701). #702/#930 rejected that combination — a base regimen with
//! IOV — and #931 lifts it; this anchor is the external check that ferx integrates the loading
//! dose under the per-occasion clearance and reaches the same ladder as an independent engine.
//! NONMEM has no native feedback dosing, so mrgsolve is the comparator. The reference kit lives
//! in `tests/reference/vanco_iov_loading_mrgsolve/`:
//!
//!   - `vanco_iov_loading_mrgsolve.R` builds the identical 1-cpt IV vancomycin model in
//!     mrgsolve 1.7.2, applies a 1500 mg LOADING bolus at t=0, **injects ferx's exact
//!     per-occasion κ** (reconstructed from the seeded substream), and runs an R loop mirroring
//!     ferx's continuous-titration semantics exactly. Every dose is a bolus, so each day is a
//!     single segment integrated under the NEXT record's occasion CL (the NONMEM
//!     end-of-interval convention ferx's per-segment IOV PK uses) — so the loading bolus decays
//!     over (0, 24] under occasion 0's CL, not the baseline κ=0 value.
//!   - `vanco_iov_loading_subject.csv` carries the loading bolus (base regimen) on daily trough
//!     rows (no covariate — the per-occasion κ is the sole source of PK variation); `expected.md`
//!     is the frozen realized ladder (regenerate with `Rscript vanco_iov_loading_mrgsolve.R`).
//!
//! ## What this anchors — a base dose integrated under per-occasion κ
//!
//! The model under test is `examples/adaptive_vanco_iov_loading.ferx`. The controller's day-1
//! pre-dose trough (~6.8 mg/L) is the loading dose DECAYED UNDER OCCASION 0's clearance — a base
//! regimen dropped, or frozen at κ=0, would move it (and every trough after). The per-occasion κ
//! swings CL window to window, so the ladder is reactive in *both* directions. ferx (RK45) and
//! mrgsolve (LSODA) feed the identical piecewise-constant CL, so they reach the same troughs (to
//! a small cross-solver tolerance) and — the titration being exact f64 arithmetic — the same
//! maintenance doses bit-for-bit. The loading dose is a base dose, so it never appears in the
//! reactive `ledger`.
//!
//! ## Regenerating the ferx-side inputs baked into the R kit
//!
//! The κ / eta constants in `vanco_iov_loading_mrgsolve.R` are ferx's, from the seed-20260726
//! run. Reconstruct them exactly with the same recipe the in-crate oracle uses (the
//! `reconstruct_kappas` / `reconstruct_eta_bsv` helpers in `src/api/tests/adaptive_sim_tests.rs`:
//! `subject_kappa_base_seed` + `kappa_standard_normal` + `chol(Ω_IOV)`, pinned by
//! `adaptive_iov_matches_predict_iov_with_reconstructed_kappa`). Bake them into the R script, run
//! `Rscript vanco_iov_loading_mrgsolve.R` to refreeze `expected.md`, and copy the ladder into
//! `REF_LADDER` above; the R loop recomputes the realized doses from κ, so they must match ferx.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, simulate_adaptive_from_spec, AdaptiveSimulateOptions};
use std::path::Path;

/// The anchor seed. Must match the seed the frozen κ / troughs in
/// `tests/reference/vanco_iov_loading_mrgsolve/` were generated at — the reconstruction is
/// seed-deterministic, so a different seed draws different κ and the frozen troughs no longer
/// apply.
const ANCHOR_SEED: u64 = 20260726;

/// The pre-scheduled loading dose (mg), carried on the subject, not in the reactive ledger.
const LOADING_DOSE_MG: f64 = 1500.0;

/// mrgsolve 1.7.2 reference, frozen in `tests/reference/vanco_iov_loading_mrgsolve/expected.md`.
/// Per daily MAINTENANCE decision: `(time_h, trough_mg_per_L, dose_mg)`. The troughs are the
/// mrgsolve LSODA values under ferx's reconstructed per-occasion κ; ferx's RK45 agrees within
/// `SIGNAL_TOL`. The doses are ferx's realized ladder (exact f64 titration: `750` scaled by
/// `1.25` / `0.75` within `dose_bounds`), so ferx reproduces them bit-for-bit. The day-1 trough
/// is the 1500 mg loading dose decayed under occasion 0's clearance.
const REF_LADDER: [(f64, f64, f64); 8] = [
    (24.0, 6.843985, 937.5),
    (48.0, 6.291058, 1171.875),
    (72.0, 4.348557, 1464.84375),
    (96.0, 9.507614, 1831.0546875),
    (120.0, 18.435378, 1373.291015625),
    (144.0, 9.197104, 1716.61376953125),
    (168.0, 12.786866, 1716.61376953125),
    (192.0, 11.958851, 1716.61376953125),
];

/// The per-occasion κ makes the ladder reactive in *both* directions. Decisions 0..=3 and 5 fire
/// `increase 25%` (subtherapeutic trough), decision 4 fires `decrease 25%` (the trough overshoots
/// when that occasion's CL drops), and decisions 6 & 7 re-issue the running dose (trough in band).
const INCREASE_DECISIONS: [usize; 5] = [0, 1, 2, 3, 5];
const DECREASE_DECISIONS: [usize; 1] = [4];
const REISSUE_DECISIONS: [usize; 2] = [6, 7];

/// `n_increases` / `n_decreases` count realized dose step-ups / -downs by dose-delta.
/// `n_increases` is 4 — one fewer than the 5 increase-rule firings — because decision 0's
/// increase steps off the un-realized `start_dose` (750 → 937.5). `n_decreases` is 1.
const N_DOSE_INCREASES: usize = 4;
const N_DOSE_DECREASES: usize = 1;

/// Cross-solver trough tolerance (mg/L). ferx (RK45) vs mrgsolve (LSODA) integrate the same
/// piecewise-constant per-occasion-CL system, so the observed gap is small (~3e-4). A trajectory
/// mismatch from a broken occasion → CL hand-off, or a dropped / κ=0-frozen base regimen, would
/// blow far past it.
const SIGNAL_TOL: f64 = 2e-3;

/// Fast PR-time guard (NOT slow-gated). The cross-engine check below only runs nightly, so
/// without this a typo in `adaptive_vanco_iov_loading.ferx` or its subject CSV — which exercise
/// the #931 base-regimen × IOV surface — would slip past the per-PR job. `parse_full_model_file`
/// runs the full `[adaptive_dosing]` `validate()`, and loading the CSV pins the base regimen; a
/// short seeded run then confirms the reactive driver is `Ok` (the default-on frozen-replay +
/// #748 snapshot verifiers validating the realized run) without waiting for the nightly check.
#[test]
fn vanco_iov_loading_example_parses_and_pins_the_scenario() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_iov_loading.ferx"))
        .expect("vanco IOV-loading model must parse");
    assert_eq!(
        parsed.model.n_kappa, 1,
        "exactly one IOV effect (κ on CL) — the #701/#931 surface under test"
    );
    assert_eq!(parsed.model.n_eta, 1, "one (negligible) BSV η on CL");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] block present");
    assert_eq!(spec.start_dose, 750.0, "empiric maintenance start dose");
    assert_eq!(spec.dose_bounds, (250.0, 4000.0));
    assert_eq!(
        spec.target_window,
        Some((10.0, 15.0)),
        "trough target band (per-occasion-aware point metric)"
    );
    assert_eq!(
        spec.auc_target, None,
        "auc_target must be absent — it is a typed error for an IOV (`kappa`) subject (#701)"
    );
    assert_eq!(
        spec.at.len(),
        8,
        "8 daily maintenance decisions, t = 24..=192 h"
    );
    assert_eq!(spec.levels, None, "continuous (percentage) titration");
    assert_eq!(spec.rules.len(), 2, "increase / decrease rungs");

    // The base regimen rides on the subject: a single 1500 mg loading bolus at t=0, no covariate.
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_iov_loading_mrgsolve/vanco_iov_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco IOV-loading subject must load");
    let s = &pop.subjects[0];
    assert_eq!(s.doses.len(), 1, "one pre-scheduled loading dose");
    assert_eq!(s.doses[0].time, 0.0, "loading dose at t=0");
    assert_eq!(s.doses[0].amt, LOADING_DOSE_MG, "1500 mg loading bolus");

    // A short seeded reactive run must pass the default frozen-replay + #748 snapshot verifiers.
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(ANCHOR_SEED);
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("adaptive IOV-loading sim runs and passes the default verifiers");
    assert_eq!(res.decisions.len(), 8, "one decision per day");
    assert_eq!(
        res.ledger.len(),
        8,
        "every decision issued a maintenance dose"
    );
    assert!(
        res.ledger.iter().all(|e| e.amt != LOADING_DOSE_MG),
        "the loading dose is a base dose, not a reactive ledger entry"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored vanco per-occasion-IOV loading-dose base regimen (#931): opt in with --features slow-tests"
)]
fn vanco_iov_loading_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_iov_loading.ferx"))
        .expect("vanco IOV-loading model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    let pop = read_nonmem_csv(
        Path::new("tests/reference/vanco_iov_loading_mrgsolve/vanco_iov_loading_subject.csv"),
        None,
        None,
    )
    .expect("vanco IOV-loading subject data must load");

    // `AdaptiveSimulateOptions` is `#[non_exhaustive]`: build from `default()`.
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(ANCHOR_SEED);

    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("adaptive vanco IOV-loading sim runs (+ base-aware frozen-replay verifier)");

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

    // The #931 assertion: the day-1 trough is the loading dose decayed under occasion 0's CL, not
    // ~0 and not the baseline-κ value. A dropped or κ=0-frozen base regimen fails this.
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
    // maintenance dose matches the frozen ladder (exact f64 titration). This is the dose-for-dose
    // match of a base regimen under per-occasion IOV κ.
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
             (|Δ| {:.6} >= {SIGNAL_TOL}); a trough mismatch this large signals a broken \
             per-occasion κ → CL hand-off or a dropped / κ=0-frozen base regimen",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < 1e-6,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // The rungs fire exactly where the per-occasion κ drives them: increases (0..=3, 5),
    // one decrease (4), re-issue otherwise (6, 7 — by route).
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
    assert_eq!(m.n_increases, N_DOSE_INCREASES);
    assert_eq!(m.n_decreases, N_DOSE_DECREASES);
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.time_to_discontinuation, None);

    // Point metric: troughs in [10, 15] on 2 of 8 decisions (6 and 7).
    assert!(
        (m.pct_time_in_window.expect("trough window declared") - 2.0 / 8.0).abs() < 1e-9,
        "pct_time_in_window {:?}",
        m.pct_time_in_window
    );

    // `auc_target` was intentionally not declared (a typed error on an IOV subject).
    assert_eq!(
        m.auc_target_attainment, None,
        "no auc_target on the IOV-loading example"
    );

    // The per-occasion κ genuinely swings the troughs: the trajectory is not the degenerate
    // constant-CL one, so the anchor actually exercises the IOV mechanism rather than trivially
    // passing.
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, tr, _) in REF_LADDER.iter() {
        lo = lo.min(tr);
        hi = hi.max(tr);
    }
    assert!(
        hi - lo > 5.0,
        "the trough trajectory must vary substantially under the per-occasion κ \
         (spread {:.3} mg/L) — else the anchor would not bite",
        hi - lo
    );
}
