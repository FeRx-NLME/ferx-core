//! NONMEM anchors for the two **edges** of the steady-state record seed (#1121).
//!
//! `dose_form_lag_nonmem_anchor` pins the ordinary regime: an `SS=1` dose whose
//! lagtime is short enough that the previous cycle has fully finished by the dose
//! record, so the seeded state is a bare decaying tail. Two geometries sit outside
//! that regime, and in both of them the seed-and-propagate convention and the old
//! equilibrate-at-the-arrival shortcut give *different numbers* — which is exactly
//! the condition under which a convention can be validated at all.
//!
//! Both were found by review of #1125 and reproduced before being fixed.
//!
//! # `ss_lag_infusion` — the previous cycle is still running at the record
//!
//! An SS **infusion** with `II − ALAG < T_inf`. The pulse before the record is at
//! `d.time − (II − ALAG)`, so its infusion window `[that, that + T_inf]` straddles
//! the record and keeps delivering into the pre-arrival window, stopping at
//! `d.time + (T_inf − phase)`. That window belongs to no `DoseEvent` — it is the
//! tail of the periodic fiction — so every engine has to carry it explicitly.
//!
//! NONMEM's own trajectory says so unambiguously: with `II = 12`, `T_inf = 6`,
//! `ALAG1 = 8` the concentration **rises** from `6.5468` at the record to `6.8754`
//! at `t = 0.5` and turns over at exactly `t = 2 = 8 − 12 + 6`.
//!
//! `dose_form_lag_ss_infusion` cannot see this: its `AMT 100 / RATE 100` gives
//! `T_inf = 1` against `II = 12` and `ALAG1 ≈ 0.7`, so the seed phase is `11.3` and
//! `ss_state_at_phase` only ever takes its `phase > T_inf` arm. Here `T_inf` is
//! half the interval, so the `phase ≤ T_inf` arm — real code with no other
//! coverage — is entered by two of the three subjects.
//!
//! # `ss_lag_ge_ii` — a lagtime of a full interval or more
//!
//! `ALAG ≥ II` has no phase `II − ALAG`. NONMEM **clamps** it to zero rather than
//! wrapping it into `[0, II)`: the pulse lands on the record itself, so the record
//! carries the steady-state *peak* and decays from there with no intervening pulse
//! before the real arrival. Measured, not assumed — `PRED(0) = 13.197 =
//! (D/V)/(1 − e^{−k·II})` for ID 2, where wrapping would give `2.1815`.
//!
//! Clamping is also the only *continuous* choice, which matters because `ALAG` is
//! routinely estimated: nothing in the objective steps as the outer optimizer walks
//! a lagtime across the interval.
//!
//! # Four routes, one oracle
//!
//! Each geometry ships twice. The `_flatwt` variant holds `WT` constant, so ferx
//! routes the subject to its **static** predictors (the dense ODE walk, the
//! analytical superposition); the varying-`WT` variant routes it to the two
//! **event-driven** walks. Same model, same lagtime, same NONMEM table — so a
//! defect that lives in only one router is visible as a disagreement with the
//! oracle rather than having to be inferred from engine-vs-engine agreement.
//!
//! That split is not hypothetical: `ALAG ≥ II` was wrong on *both* sides in
//! different ways. The event-driven walks read the pre-arrival window as `0`; the
//! static paths wrapped the phase, and then re-equilibrated at an arrival that is
//! no longer the periodic trough, reading `+4.3 %` high *after* it.
//!
//! # The four routes do not all probe the same mechanism
//!
//! Worth knowing before reading a failure. Three of them *walk*, so they need the
//! residual infusion carried as an explicit forcing window. The analytical
//! superposition does not walk at all — it evaluates the steady-state closed form
//! at a phase, and that form already has the infusion in it — so it needs only the
//! phase to be right.
//!
//! Each test below was mutation-checked against the specific defect it exists for:
//!
//! ```text
//!   wrap the seed phase instead of clamping   -> the four ss_lag_ge_ii tests
//!   drop the residual infusion window         -> the three walking engines
//!   re-equilibrate at every arrival           -> the two static ss_lag_ge_ii tests
//!   drop the seed phase in the superposition  -> the two superposition tests
//! ```
//!
//! No mutation turns all nine red, which is the point: a shared oracle over four
//! independent implementations localises a defect instead of just detecting one.
//!
//! # What these do **not** cover
//!
//! `ALAG ≥ II` **combined with an infusion** (ID 3 of `ss_lag_infusion`) is
//! deliberately not asserted against NONMEM. NONMEM infuses over
//! `[record, record + T_inf + (ALAG − II)]` there — 8 h at `RATE = 100` for an
//! `AMT = 600` dose, i.e. 800 mg — which is not a physical reading of the dose, so
//! there is nothing to anchor to. ferx delivers the dose's own `T_inf` from the
//! clamped phase, which is mass-balanced and continuous with the `ALAG < II` side;
//! `ferx_engines_agree_on_a_steady_state_infusion_with_a_lagtime_past_the_interval`
//! pins that the four ferx routes at least agree with each other, and
//! `docs/model-file/lagtime.qmd` states the divergence.
//!
//! Tier 1/2 by construction: these are `predict()` evaluations at fixed
//! parameters, ~0.5 s in total, so they run on every PR and carry the diff's
//! coverage. The objective anchors below are gated only for symmetry with the rest
//! of the suite.

use std::path::PathBuf;

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// NONMEM's `PRED` column keyed by `(ID, TIME)` — the η = 0 population
/// prediction, which is what `ferx_core::predict` computes, so no EBEs and no fit
/// are involved on either side.
fn nonmem_pred_table(tab: &str) -> std::collections::HashMap<(String, u64), f64> {
    let text = std::fs::read_to_string(anchor(tab)).expect("the NONMEM table is committed");
    let mut lines = text.lines();
    lines.next().expect("TABLE NO. banner");
    let header: Vec<&str> = lines
        .next()
        .expect("column header")
        .split_whitespace()
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("{name} column"))
    };
    let (id_col, time_col, pred_col) = (col("ID"), col("TIME"), col("PRED"));
    let mut out = std::collections::HashMap::new();
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() <= pred_col {
            continue;
        }
        let id: f64 = f[id_col].parse().expect("numeric ID");
        let time: f64 = f[time_col].parse().expect("numeric TIME");
        let pred: f64 = f[pred_col].parse().expect("numeric PRED");
        out.insert((format!("{}", id as i64), time.to_bits()), pred);
    }
    out
}

/// ferx's η = 0 predictions for `model`/`data`, as `(id, time, pred)`.
fn ferx_pred_table(model: &str, data: &str) -> Vec<(String, f64, f64)> {
    let parsed = ferx_core::parse_full_model_file(std::path::Path::new(&anchor(model)))
        .expect("the anchor model parses");
    let (population, _covariates) = ferx_core::read_nonmem_csv_with_covariates(
        std::path::Path::new(&anchor(data)),
        parsed.covariate_decls.as_deref().unwrap_or(&[]),
        &[],
        None,
    )
    .expect("the anchor dataset loads");
    ferx_core::predict(&parsed.model, &population, &parsed.model.default_params)
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

/// Compare every ferx η = 0 prediction against NONMEM's `PRED` at the same
/// `(ID, TIME)`, optionally restricted to `only_ids`.
fn assert_pred_matches_nonmem_for(
    model: &str,
    data: &str,
    tab: &str,
    max_relative: f64,
    only_ids: &[&str],
) {
    let nm = nonmem_pred_table(tab);
    let ferx = ferx_pred_table(model, data);
    assert!(!ferx.is_empty(), "predict() returned no rows for {data}");
    let mut compared = 0usize;
    for (id, time, pred) in &ferx {
        if !only_ids.is_empty() && !only_ids.contains(&id.as_str()) {
            continue;
        }
        let Some(&want) = nm.get(&(id.clone(), time.to_bits())) else {
            panic!("no NONMEM PRED row for ID {id} at TIME {time} in {tab}");
        };
        let rel = (pred - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < max_relative,
            "{model} on {data}: ID {id} t={time}: ferx PRED {pred:.6} vs NONMEM {want:.6} \
             (relative {rel:.3e} > {max_relative:.0e})"
        );
        compared += 1;
    }
    // A silent zero-comparison pass is the failure mode this helper exists to
    // avoid — a key mismatch, or an `only_ids` filter that matches nothing, must
    // not read as agreement.
    assert!(
        compared > 0,
        "no ferx prediction was matched against a NONMEM row for {data}"
    );
}

/// The tolerance every pointwise check here uses. NONMEM's `$TABLE` writes five
/// significant figures, so ~1e-5 is the floor a correct engine can reach and 1e-4
/// is a hair above it — tight enough that the defects these anchors were built for
/// (4.1e-2 and 100 %) are nowhere near passing.
const PRED_TOL: f64 = 1e-4;

/// Every subject except ID 3 of the infusion geometry, whose `ALAG ≥ II` +
/// infusion combination NONMEM answers non-physically (see the module docs).
const INFUSION_ANCHORED_IDS: &[&str] = &["1", "2"];

// ---------------------------------------------------------------------------
// The previous cycle's infusion crosses the dose record.
// ---------------------------------------------------------------------------

/// Event-driven route: a time-varying `WT` steps at `t = 1`, inside the residual
/// window, and again at `t = 9`, inside the real infusion.
#[test]
fn ss_infusion_residual_matches_nonmem_on_the_event_driven_walk() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_fit.ferx",
        "ss_lag_infusion.csv",
        "results/ss_lag_infusion.tab",
        PRED_TOL,
        INFUSION_ANCHORED_IDS,
    );
}

/// The same geometry through the **closed-form** event-driven walk, which has its
/// own copy of the residual-window logic in `pk/event_driven.rs`. Before the fix
/// this read `4.1e-2` low across the whole pre-arrival window, exactly like the
/// ODE walk.
#[test]
fn ss_infusion_residual_matches_nonmem_on_the_closed_form_walk() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_cf_fit.ferx",
        "ss_lag_infusion.csv",
        "results/ss_lag_infusion.tab",
        PRED_TOL,
        INFUSION_ANCHORED_IDS,
    );
}

/// Static route (flat `WT`): the dense ODE predictor rather than the event-driven
/// walk. Same convention, different code path.
#[test]
fn ss_infusion_residual_matches_nonmem_on_the_static_ode_predictor() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_fit.ferx",
        "ss_lag_infusion_flatwt.csv",
        "results/ss_lag_infusion_flatwt.tab",
        PRED_TOL,
        INFUSION_ANCHORED_IDS,
    );
}

/// Static route through the analytical superposition (`pk::predict_concentration`).
#[test]
fn ss_infusion_residual_matches_nonmem_on_the_analytical_superposition() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_cf_fit.ferx",
        "ss_lag_infusion_flatwt.csv",
        "results/ss_lag_infusion_flatwt.tab",
        PRED_TOL,
        INFUSION_ANCHORED_IDS,
    );
}

// ---------------------------------------------------------------------------
// A lagtime of a full dosing interval or more.
// ---------------------------------------------------------------------------

/// Event-driven route. Before the fix the whole pre-arrival window read exactly
/// `0.0` here — under a proportional error model, the loudest failure available.
#[test]
fn ss_lag_ge_ii_matches_nonmem_on_the_event_driven_walk() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_fit.ferx",
        "ss_lag_ge_ii.csv",
        "results/ss_lag_ge_ii.tab",
        PRED_TOL,
        &[],
    );
}

/// The closed-form event-driven walk on the same data.
#[test]
fn ss_lag_ge_ii_matches_nonmem_on_the_closed_form_walk() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_cf_fit.ferx",
        "ss_lag_ge_ii.csv",
        "results/ss_lag_ge_ii.tab",
        PRED_TOL,
        &[],
    );
}

/// Static ODE route. This one was wrong on *both* sides of the arrival: the seed
/// phase was the bare trough (a whole cycle of decay low) and the arrival then
/// re-equilibrated to a state the propagation no longer reaches.
#[test]
fn ss_lag_ge_ii_matches_nonmem_on_the_static_ode_predictor() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_fit.ferx",
        "ss_lag_ge_ii_flatwt.csv",
        "results/ss_lag_ge_ii_flatwt.tab",
        PRED_TOL,
        &[],
    );
}

/// Analytical superposition. Its pre-arrival branch *wrapped* the phase into
/// `[0, II)` — defensible on its face, and wrong: NONMEM clamps. Its post-arrival
/// branch then collapsed the tail and the arrival into one `C_ss(t − t_eff)`,
/// which is an identity only while the arrival lands on the trough.
#[test]
fn ss_lag_ge_ii_matches_nonmem_on_the_analytical_superposition() {
    assert_pred_matches_nonmem_for(
        "ss_lag_iv_cf_fit.ferx",
        "ss_lag_ge_ii_flatwt.csv",
        "results/ss_lag_ge_ii_flatwt.tab",
        PRED_TOL,
        &[],
    );
}

// ---------------------------------------------------------------------------
// The one cell with no external oracle.
// ---------------------------------------------------------------------------

/// `ALAG ≥ II` **with an infusion** — ID 3 of the infusion geometry, the cell the
/// module docs exclude from the NONMEM comparison.
///
/// There is no oracle to anchor to, so what is asserted instead is that ferx's
/// four routes agree with each other: an unanchored cell that at least cannot
/// depend on how the model happens to be written, or on whether the subject
/// happens to carry a time-varying covariate. Agreement to solver tolerance, not
/// to a fixture — a committed expected value here would only re-record whatever
/// ferx did on the day it was written.
#[test]
fn ferx_engines_agree_on_a_steady_state_infusion_with_a_lagtime_past_the_interval() {
    for data in ["ss_lag_infusion.csv", "ss_lag_infusion_flatwt.csv"] {
        let ode = ferx_pred_table("ss_lag_iv_fit.ferx", data);
        let cf = ferx_pred_table("ss_lag_iv_cf_fit.ferx", data);
        assert_eq!(
            ode.len(),
            cf.len(),
            "{data}: both engines predict every row"
        );
        let mut compared = 0usize;
        for ((id, t, a), (id2, t2, b)) in ode.iter().zip(cf.iter()) {
            assert_eq!((id, t), (id2, t2), "{data}: row alignment");
            if id != "3" {
                continue;
            }
            let rel = (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel < 1e-6,
                "{data} ID 3 t={t}: [odes] {a:.9} vs closed form {b:.9} (relative {rel:.3e})"
            );
            compared += 1;
        }
        assert!(compared > 0, "{data}: ID 3 must be present");
    }
}

// ---------------------------------------------------------------------------
// Objective-level anchors.
// ---------------------------------------------------------------------------

/// ferx FOCEI OFV at the same fixed parameters NONMEM evaluated (`maxiter = 0`,
/// covariance off — both set in the `.ferx` files).
fn ferx_ofv(model: &str, data: &str) -> f64 {
    let (result, _pop) = ferx_core::run_model_with_data(&anchor(model), Some(&anchor(data)))
        .expect("the anchor model and dataset load and evaluate");
    result.ofv
}

/// The repo's standard anchor tolerance: half an OFV unit.
const TOL: f64 = 0.5;

/// `nonmem_anchor/results/ss_lag_ge_ii.ext`.
const NM_OBJV_GE_II: f64 = 143.839_252_203_062_90;

/// `nonmem_anchor/results/ss_lag_ge_ii_flatwt.ext`.
const NM_OBJV_GE_II_FLATWT: f64 = -50.236_960_368_462_199;

/// The pointwise checks above compare η = 0 predictions; this closes the loop at
/// the objective, over the EBEs, which is what a fit actually optimises. Only the
/// bolus geometry is anchored this way — the infusion dataset's ID 3 has no
/// NONMEM-comparable answer, so its objective is not comparable either.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_objective_matches_nonmem_for_a_lagtime_past_the_dosing_interval() {
    let ofv = ferx_ofv("ss_lag_iv_fit.ferx", "ss_lag_ge_ii.csv");
    assert!(
        (ofv - NM_OBJV_GE_II).abs() < TOL,
        "ferx {ofv:.6} vs NONMEM {NM_OBJV_GE_II:.6}"
    );
}

/// The static-route twin of the above.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_objective_matches_nonmem_for_a_lagtime_past_the_interval_under_flat_covariates() {
    let ofv = ferx_ofv("ss_lag_iv_fit.ferx", "ss_lag_ge_ii_flatwt.csv");
    assert!(
        (ofv - NM_OBJV_GE_II_FLATWT).abs() < TOL,
        "ferx {ofv:.6} vs NONMEM {NM_OBJV_GE_II_FLATWT:.6}"
    );
}
