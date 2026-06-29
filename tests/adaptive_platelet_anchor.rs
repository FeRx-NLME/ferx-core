//! mrgsolve cross-check for the reactive `[adaptive_dosing]` **`levels` ladder**
//! — ferx-core's oncology dose-modification path (epic #391 S2.5a, parent #584).
//!
//! NONMEM has no native feedback dosing, so the apples-to-apples external anchor
//! for this feature family is **mrgsolve** (which does do feedback dosing). The
//! reference kit lives in `tests/reference/platelet_mrgsolve/`:
//!
//!   - `platelet_mrgsolve.R` builds the identical structural model (a Friberg
//!     semi-mechanistic myelosuppression model driven by a 1-cpt IV drug) in
//!     mrgsolve 1.7.2 and runs an R loop whose controller mirrors ferx's
//!     `levels`-ladder semantics exactly (read latent platelet count pre-dose,
//!     first-matching rule wins, bare `decrease` steps one rung down a
//!     strictly-increasing ladder, no match re-issues the current rung, `stop`
//!     discontinues).
//!   - `expected.md` is the frozen realized dose ladder (regenerate with
//!     `Rscript platelet_mrgsolve.R`).
//!
//! ## What this anchors — the reactive decisions, against a different engine
//!
//! The model under test is `examples/adaptive_platelet_ladder.ferx`. With
//! negligible IIV it is effectively a typical-value patient, so ferx's reactive
//! driver (RK45) and mrgsolve's R loop (LSODA) integrate the *same* ODE system
//! and the controller sees the *same* platelet trajectory — they must therefore
//! reach the *same* dose decisions. The platelets fall into thrombocytopenia
//! under the top dose, the controller steps the dose down two levels
//! (100 -> 75 -> 50), and platelets then recover and hold at 50 mg.
//!
//! The frozen mrgsolve ladder (`expected.md`, full precision in
//! `platelet_mrgsolve.R` stdout) is, per decision, `(time_h, platelet, dose_mg)`.
//! ferx reproduces every dose **exactly** and every platelet signal to
//! < 0.01 x10^9/L (a cross-solver difference ~3e-5 relative). The 0.5-unit signal
//! band below is ~80x that gap yet ~50x *below* the ~25-unit margin from either
//! decision signal to its rule threshold (120 / 30), so a real trajectory bug —
//! which would move platelets by tens — is caught while the legitimate
//! cross-solver run never false-positives.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, simulate_adaptive_from_spec, AdaptiveSimulateOptions};
use std::path::Path;

/// mrgsolve 1.7.2 reference, frozen in `tests/reference/platelet_mrgsolve/expected.md`.
/// Per weekly decision: `(time_h, platelet_x10^9/L, dose_mg)`.
const REF_LADDER: [(f64, f64, f64); 10] = [
    (0.0, 250.000000, 100.0),
    (168.0, 169.562700, 100.0),
    (336.0, 94.266960, 75.0),
    (504.0, 98.687490, 50.0),
    (672.0, 151.280660, 50.0),
    (840.0, 178.503940, 50.0),
    (1008.0, 166.771510, 50.0),
    (1176.0, 158.404870, 50.0),
    (1344.0, 160.123630, 50.0),
    (1512.0, 162.540060, 50.0),
];

/// Cross-solver platelet-signal band (x10^9/L); see the module note.
const SIGNAL_TOL: f64 = 0.5;

/// Fast PR-time guard (NOT slow-gated). The cross-engine check below only runs
/// nightly, so without this a typo in `adaptive_platelet_ladder.ferx` — which
/// exercises new DSL surface, notably the one-sided `target_window = [100, inf]`
/// — would slip past the per-PR job. `parse_full_model_file` runs the full
/// `[adaptive_dosing]` `validate()`, so a successful parse plus these spot-checks
/// pin the ladder the anchor relies on.
#[test]
fn platelet_example_parses_and_pins_the_ladder() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_platelet_ladder.ferx"))
        .expect("platelet model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] block present");
    assert_eq!(
        spec.levels.as_deref(),
        Some(&[25.0, 50.0, 75.0, 100.0][..]),
        "strictly-increasing discrete ladder"
    );
    assert_eq!(spec.start_dose, 100.0, "starts at the top rung");
    assert_eq!(
        spec.target_window,
        Some((100.0, f64::INFINITY)),
        "one-sided `PLT >= 100` window parsed (the `inf` upper bound)"
    );
    assert_eq!(spec.at.len(), 10, "ten weekly decisions, t = 0..=1512 h");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + mrgsolve-anchored platelet ladder (#391 S2.5a): opt in with --features slow-tests"
)]
fn platelet_ladder_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new("examples/adaptive_platelet_ladder.ferx"))
        .expect("platelet model must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    let pop = read_nonmem_csv(
        Path::new("tests/reference/platelet_mrgsolve/platelet_subject.csv"),
        None,
        None,
    )
    .expect("platelet subject data must load");

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
    .expect("adaptive platelet sim runs");

    // Every decision dosed (no hold/stop here), so all three artifacts are 10 long.
    assert_eq!(res.decisions.len(), 10, "one decision per weekly cycle");
    assert_eq!(res.ledger.len(), 10, "every decision issued a dose");

    // Per-decision: the platelet signal matches mrgsolve within the cross-solver
    // band, and the realized dose matches the frozen ladder exactly.
    for (i, &(t_ref, plt_ref, dose_ref)) in REF_LADDER.iter().enumerate() {
        let d = &res.decisions[i];
        assert_eq!(d.time, t_ref, "decision {i} time");
        let sig = d
            .observed_signals
            .first()
            .map(|s| s.value)
            .unwrap_or(f64::NAN);
        assert!(
            (sig - plt_ref).abs() < SIGNAL_TOL,
            "decision {i}: ferx platelet {sig:.6} vs mrgsolve {plt_ref:.6} \
             (|Δ| {:.6} >= {SIGNAL_TOL}); a trajectory mismatch this large signals \
             a structural/integration bug",
            (sig - plt_ref).abs()
        );
        let e = &res.ledger[i];
        assert_eq!(
            e.amt, dose_ref,
            "decision {i}: ferx dose {} != mrgsolve {dose_ref} (ladder divergence)",
            e.amt
        );
    }

    // The de-escalation happened exactly at decisions 2 and 3 (the two `decrease`
    // rungs); every other decision re-issued the running dose (recorded by route).
    for (i, e) in res.ledger.iter().enumerate() {
        if i == 2 || i == 3 {
            assert!(
                e.rule_fired.contains("decrease"),
                "decision {i} should fire the decrease rung, got {:?}",
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

    // Per-subject metrics: a faithful reduction of the realized ladder.
    let m = &res.metrics[0];
    assert_eq!(m.cumulative_dose, 625.0, "100+100+75+50*7");
    assert_eq!(m.n_doses, 10);
    assert_eq!(m.n_increases, 0);
    assert_eq!(m.n_decreases, 2, "100 -> 75 -> 50");
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.time_to_discontinuation, None);
    // pct of decisions with platelets >= 100 (one-sided target_window = [100, inf]):
    // 8 of 10 (only the two `decrease` decisions, ~94 and ~99, fall below).
    assert!((m.pct_time_in_window.expect("window declared") - 0.8).abs() < 1e-9);
    assert_eq!(
        m.signal_max.expect("max"),
        250.0,
        "baseline at the first decision"
    );
    assert!((m.signal_min.expect("min") - 94.266960).abs() < SIGNAL_TOL);
    assert!((m.signal_mean.expect("mean") - 159.014182).abs() < SIGNAL_TOL);
}
