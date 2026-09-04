//! #1186 — a **derived break time inside the event-match tolerance** must not apply a
//! dose twice, through the real user path: model file → parser → `dose_attr_map` →
//! `read_nonmem_csv` → `predict()`, against committed NONMEM `PRED`.
//!
//! The unit tests in `src/ode/predictions_tests.rs` drive the four engines directly with
//! a hand-built `OdeSpec`, which is the sharpest form of the check but not the shape a
//! user meets. This file closes that gap: the collision here is produced by the *parser*
//! resolving `ALAG1` and a per-route `lag=` into the same `8.200000000000001` onset, so a
//! regression anywhere between the DSL and the engines is caught.
//!
//! # The geometry
//!
//! `depot ← first_order(ka=KA, lag=LAGR)` with a compartment lag `ALAG1` on the depot,
//! feeding a 1-cpt central; central boluses at `t = 0` and `t = 8.2`. The route onset is
//! `TDOS + ALAG1 + LAGR = (0 + 0.3) + 7.9`, which in double precision is
//! `8.200000000000001` — **1.78e-15** past the `t = 8.2` dose's own break. That is past
//! the timeline's `1e-15` dedup (so the two stay separate breaks) and well inside the
//! dose-arrival match (so a rescanning loop matches the same dose at both), which is
//! exactly the band #1186 is about.
//!
//! Two arms, because they fail on different engines:
//!
//! | arm | victim at `t = 8.2` | what doubled | which engines were wrong |
//! |---|---|---|---|
//! | `break_collision` | bolus | the `F·AMT` state jump | objective **and** both dense/gated engines |
//! | `break_collision_inf` | 1 h infusion (`RATE = 100`) | the `active_infusions` push, i.e. the *rate* for the whole window | the two `Gated` engines only — the objective's `Spanning` arm recomputes its active set per segment and cannot push twice |
//!
//! The second arm is the one no OFV could ever have caught: ferx's objective path agreed
//! with NONMEM to 8 digits while sdtab, the joint PK-TTE hazard, `[derived]` integrals
//! and `simulate()` read 148.30 against 99.52.
//!
//! # Why NONMEM is a sound oracle for a float coincidence
//!
//! It is not that NONMEM reproduces the coincidence — no control stream can *express*
//! "two break times 1.78e-15 apart". It is that NONMEM's event handling is **typed**: it
//! applies each dose record exactly once by construction, whatever the arithmetic does.
//! So `break_collision.ctl` is an independent statement of the right answer, and the
//! defect is visible as ferx disagreeing with it by a whole 100 mg dose.
//!
//! The control stream is an exact twin, not an approximation: it hand-writes
//! `RIN = PODO·KA·EXP(−KA·(T−ONSET))` with `ONSET = TDOS + LAGC + LAGR`, so the same
//! three-term float sum is formed on both sides. It needs one extra (inert, `F1 = 0`)
//! carrier compartment to capture `PODO`/`TDOS`, which is why the two datasets are keyed
//! differently — see the header of `break_collision_fit.ferx`.
//!
//! Fast (`MAXEVAL = 0` / `predict()`, no fit), so it runs on every PR.

use ferx_core::ode::predictions::{ode_dense_solve_states, ode_predictions_with_states};
use ferx_core::pk::compute_event_pk_params;
use std::collections::HashMap;

fn anchor(name: &str) -> String {
    format!("{}/nonmem_anchor/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// `(ID, TIME bits) → PRED` from a committed NONMEM `$TABLE`.
fn nonmem_pred_table(tab: &str) -> HashMap<(String, u64), f64> {
    let text = std::fs::read_to_string(anchor(tab)).expect("the committed NONMEM table reads");
    let mut lines = text.lines();
    lines.next(); // "TABLE NO.  1"
    let header: Vec<&str> = lines
        .next()
        .expect("the table has a header row")
        .split_whitespace()
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("{tab} has no {name} column"))
    };
    let (id_col, time_col, pred_col) = (col("ID"), col("TIME"), col("PRED"));
    let mut out = HashMap::new();
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() <= pred_col {
            continue;
        }
        let id: f64 = f[id_col].parse().expect("numeric ID");
        let time: f64 = f[time_col].parse().expect("numeric TIME");
        let pred: f64 = f[pred_col].parse().expect("numeric PRED");
        // IDs are written `1.0000E+00`; ferx keeps the raw CSV string `1`.
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

/// Every ferx η = 0 prediction against NONMEM's `PRED` at the same `(ID, TIME)`.
///
/// `compared` is pinned to the row count because a key mismatch (a `TIME` that rounds
/// differently, say) would otherwise make an empty comparison read as agreement — the
/// silent-pass this helper exists to rule out.
fn assert_pred_matches_nonmem(model: &str, data: &str, tab: &str, max_relative: f64) {
    let nm = nonmem_pred_table(tab);
    let ferx = ferx_pred_table(model, data);
    assert!(!ferx.is_empty(), "predict() returned no rows for {data}");
    let mut compared = 0usize;
    for (id, time, pred) in &ferx {
        let Some(&want) = nm.get(&(id.clone(), time.to_bits())) else {
            panic!("no NONMEM PRED row for ID {id} at TIME {time} in {tab}");
        };
        assert!(
            pred.is_finite(),
            "ID {id} t={time}: ferx PRED is not finite"
        );
        let rel = (pred - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < max_relative,
            "ID {id} t={time}: ferx PRED {pred:.6} vs NONMEM {want:.6} \
             (relative {rel:.3e} > {max_relative:.0e})"
        );
        compared += 1;
    }
    assert_eq!(
        compared,
        ferx.len(),
        "every ferx prediction must have been matched against a NONMEM row"
    );
}

/// The straddle the whole anchor rests on, restated in the units the *model file*
/// produces: `ALAG1 + LAGR` must land past the `1e-15` dedup and inside the event match
/// of the `t = 8.2` dose, or the collision does not occur and both arms below pass for
/// the wrong reason.
///
/// Asserted from the θ initial values in `break_collision_fit.ferx` itself, so editing
/// those numbers fails here rather than quietly turning the anchor into a control.
#[test]
fn the_anchor_model_still_produces_a_colliding_break() {
    let parsed =
        ferx_core::parse_full_model_file(std::path::Path::new(&anchor("break_collision_fit.ferx")))
            .expect("the anchor model parses");
    let theta = &parsed.model.default_params.theta;
    // TVCL, TVV, TVKA, TVLAGC, TVLAGR — the last two are the collision.
    assert_eq!(theta.len(), 5, "the anchor model declares five thetas");
    let (lag_c, lag_r) = (theta[3], theta[4]);
    let onset = (0.0 + lag_c) + lag_r;
    let sep = (onset - 8.2f64).abs();
    assert!(
        sep > 1e-15,
        "onset {onset:.17} is inside the 1e-15 timeline dedup of the t=8.2 dose \
         (sep {sep:.3e}): the two breaks would merge and nothing could double"
    );
    assert!(
        sep < 1e-12,
        "onset {onset:.17} is outside the event match of the t=8.2 dose \
         (sep {sep:.3e}): the scan would never match that dose twice"
    );
}

/// **The bolus arm.** A route onset landing 1.78e-15 past a *different* dose's break
/// must not apply that dose twice, through the parser and the public `predict()`.
///
/// Before the fix, every engine but the event-driven one read `244.04` at `t = 8.2001`
/// against NONMEM's `144.041725` — a whole extra 100 mg dose, and `239.11` against
/// `170.725636` at `t = 12`.
#[test]
fn ferx_predictions_match_nonmem_when_a_route_onset_collides_with_a_bolus() {
    assert_pred_matches_nonmem(
        "break_collision_fit.ferx",
        "break_collision_ferx.csv",
        "results/break_collision.tab",
        1e-6,
    );
}

/// **The infusion arm, through `predict()`.** This one is a **control, not a regression
/// test**, and saying so matters: `predict()` routes to `ode_predictions`, whose
/// `Spanning` arm rebuilds `active_infusions` per segment and therefore never doubled an
/// infusion. It passed before the fix and passes after — verified by mutation (removing
/// the objective mask kills the bolus arm above and leaves this one green). Its job is to
/// pin that the objective path stays exact on the infusion geometry (`99.52498572` against
/// NONMEM's `99.5249857`); the engines that *were* wrong are covered by the test below.
#[test]
fn ferx_predictions_match_nonmem_when_a_route_onset_collides_with_an_infusion() {
    assert_pred_matches_nonmem(
        "break_collision_fit.ferx",
        "break_collision_inf_ferx.csv",
        "results/break_collision_inf.tab",
        1e-6,
    );
}

/// **The two `Gated` engines on the infusion arm — the half no OFV could ever see.**
///
/// A colliding *infusion* was pushed into `active_infusions` twice, doubling its rate for
/// the whole window, but only on the engines that carry a pre-built active list:
/// `ode_predictions_with_states` (sdtab IPRED + compartment states) and
/// `ode_dense_solve_states` (the joint PK-TTE hazard, `[derived]` integrals, the Markov
/// endpoint, the adaptive AUC signal, and — through the shared `apply_segment_boundary` —
/// `simulate()`'s event times). They read `148.30 / 255.59 / 246.18` against NONMEM's
/// `99.5249857 / 160.432175 / 174.261827` while `predict()` above was exact to 8 digits.
///
/// Driven through the parser and `read_nonmem_csv` like the tests above, so the collision
/// is produced by the real `ALAG1` + per-route `lag=` resolution rather than a hand-built
/// `OdeSpec`. The anchor model reads `y = central`, so the state *is* NONMEM's
/// `IPRED = A(3)` and the compartment compares to the table directly.
#[test]
fn the_gated_engines_match_nonmem_when_a_route_onset_collides_with_an_infusion() {
    let parsed =
        ferx_core::parse_full_model_file(std::path::Path::new(&anchor("break_collision_fit.ferx")))
            .expect("the anchor model parses");
    let (population, _covariates) = ferx_core::read_nonmem_csv_with_covariates(
        std::path::Path::new(&anchor("break_collision_inf_ferx.csv")),
        parsed.covariate_decls.as_deref().unwrap_or(&[]),
        &[],
        None,
    )
    .expect("the anchor dataset loads");
    let model = &parsed.model;
    let ode = model
        .ode_spec
        .as_ref()
        .expect("the anchor model is an ODE model");
    let theta = &model.default_params.theta;
    let subject = &population.subjects[0];
    let eta = vec![0.0; model.n_eta];
    // The eta = 0 typical-value snapshot, the same quantity `predict()` evaluates at.
    let pk = compute_event_pk_params(model, subject, theta, &eta).obs[0];

    let nm = nonmem_pred_table("results/break_collision_inf.tab");
    // `central` is state index 1 (`ode(states=[depot, central])`).
    let central = 1usize;

    let (_ipred, states) = ode_predictions_with_states(ode, &pk.values, theta, &eta, subject);
    let dense = ode_dense_solve_states(ode, &pk.values, theta, &eta, subject, &subject.obs_times);
    assert_eq!(states.len(), subject.obs_times.len());
    assert_eq!(dense.len(), subject.obs_times.len());

    let mut compared = 0usize;
    for (i, &t) in subject.obs_times.iter().enumerate() {
        let want = *nm
            .get(&(subject.id.clone(), t.to_bits()))
            .unwrap_or_else(|| panic!("no NONMEM PRED row at TIME {t}"));
        for (engine, got) in [
            ("ode_predictions_with_states", states[i][central]),
            ("ode_dense_solve_states", dense[i][central]),
        ] {
            assert!(got.is_finite(), "{engine} at t={t} is not finite");
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 1e-6,
                "{engine} at t={t}: {got:.6} vs NONMEM {want:.6} (relative {rel:.3e}) \
                 — a doubled infusion rate reads about 1.5x this"
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        2 * subject.obs_times.len(),
        "every observation must have been compared on both gated engines"
    );
}
