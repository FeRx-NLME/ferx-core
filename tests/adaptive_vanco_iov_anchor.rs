//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **inter-occasion
//! variability** path — a fresh per-occasion κ on clearance drives a vancomycin
//! trough-TDM titration (epic #391 / **#701**).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor
//! for this feature family is **mrgsolve** (which does do feedback dosing). This is
//! the IOV analogue of the constant-covariate vanco anchor (`adaptive_vanco_anchor`)
//! and the time-varying-covariate one (`adaptive_vanco_renal_anchor`, #700); the
//! reference kit lives in `tests/reference/vanco_iov_mrgsolve/`:
//!
//!   - `vanco_iov_mrgsolve.R` builds the identical structural model (1-cpt IV
//!     vancomycin, once-daily 1-h infusion, `CL = TVCL·exp(η + κ)`) in mrgsolve
//!     1.7.2, **injects ferx's exact per-occasion κ** (reconstructed from the seeded
//!     substream), replays ferx's realized dose ladder, and records the pre-dose
//!     trough at each decision. Each day is integrated as two segments — the
//!     infusion on THIS occasion's CL, the decay on the NEXT occasion's CL
//!     (end-of-interval convention ferx's per-segment IOV PK uses).
//!   - `expected.md` is the frozen mrgsolve ladder (κ, CL, trough, dose per
//!     decision). Regenerate with `Rscript vanco_iov_mrgsolve.R`.
//!
//! ## Anchor form — deterministic dose-for-dose replay with reconstructed κ
//!
//! κ is *random* (drawn per decision window on a seeded RNG substream), so an
//! independent mrgsolve draw could never match. The anchor reconstructs ferx's
//! EXACT per-occasion κ (the Rust reconstruction is pinned by the unit test
//! `adaptive_iov_matches_predict_iov_with_reconstructed_kappa` in `src/api.rs`) and
//! injects the SAME per-occasion clearance into mrgsolve, replaying ferx's realized
//! doses. What is cross-validated is the occasion → CL → trajectory *mechanism*:
//! ferx's RK45 and mrgsolve's LSODA integrate the identical piecewise-constant
//! system and must reach the same trough trajectory. This test runs the ferx
//! reactive driver **live** at the anchor seed and asserts its troughs (and its
//! realized dose ladder) match the frozen mrgsolve ladder.
//!
//! ## Regenerating the ferx-side inputs baked into the R kit
//!
//! The κ / CL / dose constants in `vanco_iov_mrgsolve.R` are ferx's, from the
//! seed-20260708 run. To regenerate them, un-`#[ignore]` and run the in-crate
//! harness `temp_print_vanco_iov_anchor_coordination` (or reconstruct via the
//! api.rs unit test's recipe): it prints, per occasion, the reconstructed κ, the
//! resolved CL, the realized dose, and the live trough — the exact numbers frozen
//! into the R script and `expected.md`. The reconstruction (`subject_kappa_base_seed`
//! + `kappa_standard_normal` + `chol(Ω_IOV)`) is `pub(crate)`, so it lives in the
//! crate; this integration test consumes only public API (`simulate_adaptive_from_spec`)
//! and the frozen reference, which is the whole point of an *external* anchor.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{
    simulate_adaptive_from_spec, AdaptiveSimulateOptions, DoseEvent, Population, Subject,
};
use std::collections::HashMap;
use std::path::Path;

/// The anchor seed. Must match the seed the frozen κ / troughs in
/// `tests/reference/vanco_iov_mrgsolve/{vanco_iov_mrgsolve.R,expected.md}` were
/// generated at — the reconstruction is seed-deterministic, so a different seed
/// draws different κ and the frozen mrgsolve troughs would no longer apply.
const ANCHOR_SEED: u64 = 20260708;

/// mrgsolve 1.7.2 reference, frozen in `tests/reference/vanco_iov_mrgsolve/expected.md`.
/// Per daily decision: `(time_h, trough_mg_per_L, dose_mg)`. The troughs are the
/// mrgsolve LSODA values under ferx's reconstructed per-occasion κ; ferx's RK45
/// agrees within `SIGNAL_TOL`. The doses are ferx's realized ladder (exact f64
/// titration), so ferx reproduces them bit-for-bit.
const REF_LADDER: [(f64, f64, f64); 14] = [
    (0.0, 0.000000, 625.000000),
    (24.0, 2.960637, 781.250000),
    (48.0, 3.697443, 976.562500),
    (72.0, 3.174336, 1220.703125),
    (96.0, 5.578850, 1525.878906),
    (120.0, 10.761070, 1525.878906),
    (144.0, 5.932958, 1907.348633),
    (168.0, 11.376082, 1907.348633),
    (192.0, 6.447514, 2384.185791),
    (216.0, 13.606025, 2384.185791),
    (240.0, 12.646158, 2384.185791),
    (264.0, 16.481410, 1788.139343),
    (288.0, 16.646880, 1341.104507),
    (312.0, 12.439516, 1341.104507),
];

/// Cross-solver trough tolerance (mg/L). ferx (RK45) vs mrgsolve (LSODA) integrate
/// the SAME piecewise-constant per-occasion-CL system, so the observed gap is tiny
/// (max ~5e-4 over the horizon). `2e-3` is a safe but meaningful guard — a
/// trajectory mismatch from a broken occasion → CL hand-off (e.g. the wrong
/// occasion's κ on a segment, which the two other plausible conventions produce)
/// would blow past it by 1–3 orders of magnitude.
const SIGNAL_TOL: f64 = 2e-3;

/// The realized dose ladder is exact f64 (the mrgsolve `expected.md` doses are
/// rounded to 6 dp, but ferx's are the un-rounded titration), so compare doses to
/// a tolerance that admits the 6-dp rounding of the frozen table but nothing more.
const DOSE_TOL: f64 = 1e-3;

/// Build the dose-free anchor subject: an EVID=0 observation at each of the 14
/// daily decision times (t = 0, 24, …, 312 h). The reactive driver injects the
/// doses; the subject only carries the readout grid so the pre-dose trough is
/// recorded at each decision. No covariates — the per-occasion κ is the sole source
/// of PK variation.
fn anchor_subject(decision_times: &[f64]) -> Subject {
    let n = decision_times.len();
    Subject {
        id: "1".to_string(),
        doses: Vec::<DoseEvent>::new(),
        obs_times: decision_times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1u32; n],
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        #[cfg(feature = "survival")]
        obs_records: vec![],
    }
}

/// Fast PR-time guard (NOT slow-gated). The cross-engine check below only runs
/// nightly, so without this a typo in `adaptive_vanco_iov.ferx` — which exercises
/// the #701 IOV adaptive surface — would slip past the per-PR job.
/// `parse_full_model_file` runs the full `[adaptive_dosing]` `validate()`, so a
/// successful parse plus these spot-checks pin the scenario the anchor relies on:
/// exactly one κ (the IOV effect), the trough control law, and that `auc_target` is
/// **absent** (it is a typed error for an IOV subject, #701). A short seeded run of
/// the reactive driver then confirms it is `Ok` — the default frozen-replay verifier
/// validating the realized run — without waiting for the nightly cross-check.
#[test]
fn vanco_iov_example_parses_and_runs_the_verifier() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_iov.ferx"))
        .expect("vanco IOV model must parse");
    assert_eq!(
        parsed.model.n_kappa, 1,
        "exactly one IOV effect (κ on CL) — the #701 surface under test"
    );
    assert_eq!(parsed.model.n_eta, 1, "one (negligible) BSV η on CL");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] block present");
    assert_eq!(spec.start_dose, 500.0, "empiric start dose");
    assert_eq!(spec.dose_bounds, (250.0, 4000.0));
    assert_eq!(
        spec.target_window,
        Some((10.0, 15.0)),
        "trough target band (per-occasion-aware point metric)"
    );
    assert_eq!(
        spec.auc_target, None,
        "auc_target must be absent — it is a typed error for an IOV (`kappa`) \
         subject (#701), so the IOV example never declares it"
    );
    assert_eq!(spec.at.len(), 14, "14 daily decisions, t = 0..=312 h");
    assert_eq!(spec.levels, None, "continuous (percentage) titration");
    assert_eq!(
        spec.rules.len(),
        2,
        "increase / decrease rungs (the per-occasion κ drives both)"
    );

    // A short seeded reactive run must pass the default frozen-replay verifier.
    let pop = Population {
        subjects: vec![anchor_subject(&spec.at)],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
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
    .expect("adaptive IOV sim runs and passes the default verifier");
    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");
}

/// Slow + mrgsolve-anchored. Runs the ferx reactive driver live at `ANCHOR_SEED`
/// (which draws the exact per-occasion κ the R kit injects), then asserts every
/// per-decision trough matches the frozen mrgsolve trough within the cross-solver
/// band and every realized dose matches the frozen ladder — the dose-for-dose IOV
/// occasion → CL → trajectory cross-check against an independent engine.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored vanco IOV (per-occasion κ) TDM titration (#701): opt in with --features slow-tests"
)]
fn vanco_iov_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_iov.ferx"))
        .expect("vanco IOV model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");

    let pop = Population {
        subjects: vec![anchor_subject(&spec.at)],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

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
    .expect("adaptive vanco IOV sim runs (+ frozen-replay verifier)");

    // Every decision dosed (no hold/stop here), so the ledger is 14 long.
    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");

    // Per-decision: the live ferx trough matches mrgsolve within the cross-solver
    // band, and the realized dose matches the frozen ladder. This is the dose-for-
    // dose match under per-occasion IOV κ — both engines integrate the identical
    // piecewise-constant CL and reach the same trough trajectory.
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
             (|Δ| {:.6} >= {SIGNAL_TOL}); a mismatch this large signals a broken \
             per-occasion κ → CL hand-off",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < DOSE_TOL,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // The per-occasion κ is genuinely nonzero (it swings CL ~2.8 → 5.8 L/h across
    // the horizon): the frozen CL column in expected.md spans that range, so the
    // realized trough trajectory is *not* the degenerate constant-CL one — the
    // anchor actually exercises the IOV mechanism rather than trivially passing.
    let troughs: Vec<f64> = REF_LADDER.iter().map(|&(_, tr, _)| tr).collect();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &tr in troughs.iter().skip(1) {
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
