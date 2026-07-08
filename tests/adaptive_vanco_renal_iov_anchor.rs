//! mrgsolve cross-check for the **composition** of the two reactive
//! `[adaptive_dosing]` PK-variation paths — a declining renal covariate (**#700**)
//! AND a per-occasion κ on clearance (**#701**) driving CL together (epic #391).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor for
//! this feature family is **mrgsolve**. This is the composition of the two single-
//! effect anchors (`adaptive_vanco_renal_anchor`, #700; `adaptive_vanco_iov_anchor`,
//! #701): here `CL = TVCL·(CRCL/100)·exp(η + κ)`, so each day's clearance is set by
//! BOTH the renal function active in that segment and that window's occasion κ. The
//! reference kit lives in `tests/reference/vanco_renal_iov_mrgsolve/`:
//!
//!   - `vanco_renal_iov_mrgsolve.R` builds the identical structural model in mrgsolve
//!     1.7.2, feeds the SAME declining CRCL trajectory AND ferx's exact per-occasion κ
//!     (reconstructed from the seeded substream — identical to the #701 anchor's, as
//!     the κ stream is model-independent), replays ferx's realized dose ladder, and
//!     records the pre-dose trough at each decision. Each day is two segments — the
//!     infusion on THIS occasion's (CRCL_g, κ_g), the decay on the NEXT occasion's
//!     (CRCL_{g+1}, κ_{g+1}) — the end-of-interval convention ferx's per-segment PK
//!     uses, now with BOTH effects composed into the piecewise-constant CL.
//!   - `expected.md` is the frozen mrgsolve ladder. Regenerate with
//!     `Rscript vanco_renal_iov_mrgsolve.R`.
//!
//! ## What this adds over the two single-effect anchors
//!
//! The renal anchor proves a covariate-driven CL matches mrgsolve; the IOV anchor
//! proves a κ-driven CL matches mrgsolve. This anchor proves they are correct
//! **together** — the #700 per-event PK recompute and the #701 per-occasion κ compose
//! into one piecewise-constant CL that an independent LSODA engine reproduces
//! dose-for-dose. (The off-record-decision covariate LOCF edge is pinned separately,
//! and fail-without-fix, by `adaptive_iov_decision_pk_uses_locf_covariate_off_grid`
//! in `src/api.rs`.)

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{
    simulate_adaptive_from_spec, AdaptiveSimulateOptions, DoseEvent, Population, Subject,
};
use std::collections::HashMap;
use std::path::Path;

/// The anchor seed. Must match the seed the frozen κ / troughs in the R kit were
/// generated at (the κ reconstruction is seed-deterministic). Identical to the #701
/// IOV anchor's seed, so the per-occasion κ is the same model-independent stream.
const ANCHOR_SEED: u64 = 20260708;

/// Declining renal function over the horizon (one CRCL per daily decision/obs),
/// CRCL(0)=120 down to CRCL(312)=40 mL/min. MUST match the `crcl` vector in
/// `tests/reference/vanco_renal_iov_mrgsolve/vanco_renal_iov_mrgsolve.R` exactly.
const CRCL: [f64; 14] = [
    120.0, 115.0, 108.0, 98.0, 88.0, 78.0, 68.0, 60.0, 54.0, 49.0, 45.0, 42.0, 41.0, 40.0,
];

/// mrgsolve 1.7.2 reference, frozen in
/// `tests/reference/vanco_renal_iov_mrgsolve/expected.md`. Per daily decision:
/// `(time_h, trough_mg_per_L, dose_mg)`. Troughs are the mrgsolve LSODA values under
/// ferx's declining CRCL × reconstructed per-occasion κ; ferx's RK45 agrees within
/// `SIGNAL_TOL`. Doses are ferx's realized ladder (exact f64 titration), reproduced
/// bit-for-bit.
const REF_LADDER: [(f64, f64, f64); 14] = [
    (0.0, 0.000000, 625.000000),
    (24.0, 2.556064, 781.250000),
    (48.0, 3.239114, 976.562500),
    (72.0, 3.175663, 1220.703125),
    (96.0, 6.413905, 1525.878906),
    (120.0, 13.302792, 1525.878906),
    (144.0, 10.762650, 1525.878906),
    (168.0, 16.631220, 1144.409180),
    (192.0, 12.316018, 1144.409180),
    (216.0, 16.342389, 858.306885),
    (240.0, 15.479172, 643.730164),
    (264.0, 15.727982, 482.797623),
    (288.0, 15.334823, 362.098217),
    (312.0, 13.346793, 362.098217),
];

/// Cross-solver trough tolerance (mg/L). ferx (RK45) vs mrgsolve (LSODA) integrate
/// the SAME piecewise-constant composed-CL system, so the gap is tiny; `2e-3` is a
/// safe but meaningful guard.
const SIGNAL_TOL: f64 = 2e-3;

/// Realized doses are exact f64; the mrgsolve `expected.md` doses are 6-dp rounded,
/// so admit that rounding but nothing more.
const DOSE_TOL: f64 = 1e-3;

/// Build the dose-free anchor subject: an EVID=0 observation at each of the 14 daily
/// decision times (t = 0, 24, …, 312 h) carrying that day's CRCL. The reactive driver
/// injects the doses; the covariate declines over the horizon and — composed with the
/// per-occasion κ — drives the pre-dose trough at each decision.
fn anchor_subject(decision_times: &[f64]) -> Subject {
    let n = decision_times.len();
    let obs_covariates: Vec<HashMap<String, f64>> = CRCL
        .iter()
        .map(|&c| HashMap::from([("CRCL".to_string(), c)]))
        .collect();
    Subject {
        id: "1".to_string(),
        doses: Vec::<DoseEvent>::new(),
        obs_times: decision_times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::from([("CRCL".to_string(), CRCL[0])]),
        dose_covariates: Vec::new(),
        obs_covariates,
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

fn anchor_population(decision_times: &[f64]) -> Population {
    Population {
        subjects: vec![anchor_subject(decision_times)],
        covariate_names: vec!["CRCL".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Coordination harness (ignored): prints ferx's realized ladder (time, trough, dose)
/// at `ANCHOR_SEED`, the numbers baked into `vanco_renal_iov_mrgsolve.R` and
/// `REF_LADDER`. Run with `cargo test --test adaptive_vanco_renal_iov_anchor -- --ignored print_coordination --nocapture`.
#[test]
#[ignore = "coordination harness: prints the ferx ladder to bake into the R kit"]
fn print_coordination() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal_iov.ferx"))
        .expect("model parses");
    let spec = parsed.adaptive_dosing.as_ref().expect("[adaptive_dosing]");
    let pop = anchor_population(&spec.at);
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
    .expect("sim runs");
    println!("=== ferx renal x IOV ladder (seed {ANCHOR_SEED}) ===");
    for (i, d) in res.decisions.iter().enumerate() {
        let sig = d
            .observed_signals
            .first()
            .map(|s| s.value)
            .unwrap_or(f64::NAN);
        let dose = res.ledger[i].amt;
        println!(
            "g={i} t={} crcl={} trough={sig:.9} dose={dose:.9}",
            d.time, CRCL[i]
        );
    }
}

/// Fast PR-time guard (NOT slow-gated): a typo in the composed example must not slip
/// past the per-PR job. Parses the model (full `[adaptive_dosing]` validate), pins the
/// scenario (one κ, the CRCL covariate referenced, trough control law, `auc_target`
/// absent), and runs a short seeded reactive run through the default frozen-replay
/// verifier.
#[test]
fn vanco_renal_iov_example_parses_and_runs_the_verifier() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal_iov.ferx"))
        .expect("vanco renal×IOV model must parse");
    assert_eq!(parsed.model.n_kappa, 1, "one IOV effect (κ on CL)");
    assert_eq!(parsed.model.n_eta, 1, "one (negligible) BSV η on CL");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] block present");
    assert_eq!(
        spec.auc_target, None,
        "auc_target is a typed error here (#700/#701)"
    );
    assert_eq!(spec.at.len(), 14, "14 daily decisions");

    let pop = anchor_population(&spec.at);
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
    .expect("adaptive renal×IOV sim runs and passes the default verifier");
    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");
}

/// Slow + mrgsolve-anchored. Runs the ferx reactive driver live at `ANCHOR_SEED`
/// (declining CRCL × the exact per-occasion κ the R kit injects) and asserts every
/// per-decision trough matches the frozen mrgsolve trough within the cross-solver band
/// and every realized dose matches the frozen ladder — the dose-for-dose composition
/// cross-check against an independent engine.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored vanco renal×IOV (covariate × per-occasion κ) TDM titration (#700×#701): opt in with --features slow-tests"
)]
fn vanco_renal_iov_titration_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_vanco_renal_iov.ferx"))
        .expect("vanco renal×IOV model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    let pop = anchor_population(&spec.at);
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
    .expect("adaptive vanco renal×IOV sim runs (+ frozen-replay verifier)");

    assert_eq!(res.decisions.len(), 14, "one decision per day");
    assert_eq!(res.ledger.len(), 14, "every decision issued a dose");

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
             covariate × occasion CL composition",
            (sig - trough_ref).abs()
        );
        let e = &res.ledger[i];
        assert!(
            (e.amt - dose_ref).abs() < DOSE_TOL,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (titration divergence)",
            e.amt
        );
    }

    // Anti-triviality: both the CRCL decline AND the per-occasion κ move CL, so the
    // trough trajectory must vary substantially — else the anchor would not bite.
    let troughs: Vec<f64> = REF_LADDER.iter().map(|&(_, tr, _)| tr).collect();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &tr in troughs.iter().skip(1) {
        lo = lo.min(tr);
        hi = hi.max(tr);
    }
    assert!(
        hi - lo > 5.0,
        "the trough trajectory must vary substantially under CRCL × κ (spread {:.3} mg/L)",
        hi - lo
    );
}
