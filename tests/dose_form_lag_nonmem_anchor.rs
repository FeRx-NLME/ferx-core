//! NONMEM anchors for #1073 across **dosing forms** — does the record convention
//! hold when the thing crossing the covariate change is not an oral bolus?
//!
//! `tvcov_lag_saltation_nonmem_anchor` pins the convention on one geometry: an oral
//! bolus whose lagged arrival lands inside a record interval carrying a different
//! covariate. The rule it pins — a segment is governed by the record that
//! *terminates* it, and a lagged arrival is not a record — is not specific to that
//! form, and a fix that happened to work only there would be worth catching. Each
//! dataset here moves one other dosing feature into the same cell and changes
//! nothing else.
//!
//! All three share the #1060 family's eight subjects, θ, Ω and Σ, so a reader
//! comparing them against anchor B is looking at one variable.
//!
//! # E2 — rate-defined infusion (`dose_form_lag_infusion`)
//!
//! 1-cpt IV, `ADVAN13` / ferx `[odes]`, two infusions with `ALAG1 = TVP·exp(ETA3)`.
//! The second runs 0.5 h from `6 + ALAG`, so its window **end** falls strictly
//! between the `t = 6` dose record (`WT = 150`) and the `t = 7` record (`WT = 75`).
//!
//! That end is a second non-record boundary governed by the same rule, and it is
//! where ferx's two production engines used to disagree with each other: the
//! closed-form walk already treated infusion edges as sub-interval bounds inside
//! one record interval (correct), while the ODE walk promoted them to parameter
//! breakpoints consuming the **previous** record. Measured against NONMEM 7.6.0 on
//! a dedicated probe, that put a 4.2 % error on the predictions after the boundary
//! and 23.5 OFV on the objective — a defect of the same family as #1073, in the
//! opposite direction, that no committed anchor covered.
//!
//! # E3 — steady state (`dose_form_lag_ss`), and a gap it uncovered
//!
//! Oral `depot → central`, `SS = 1`, `II = 12`, with a lagtime; a later SS-cycle
//! dose at `t = 12` has its arrival cross a `WT` change.
//!
//! The load-bearing distinction here is between a dose *attribute* and the
//! *interval* a dose opens. `equilibrate_ss_state` must keep reading the dose
//! **row**'s snapshot — the steady state is a property of that record — while the
//! interval from the row to the arrival is governed by the next record. Before
//! #1073 the two were the same object, so nothing distinguished them.
//!
//! Building this anchor surfaced a **separate, pre-existing** defect, so the
//! triple is not asserted. Two one-variable controls localise it:
//!
//! ```text
//!   SS + lagtime, FLAT WT        ferx  2968.202993  NONMEM  2968.2029933781705  exact
//!   SS + TV covariate, NO lag    ferx  1700.878933  NONMEM  1700.8789328578407  exact
//!   SS + lagtime + TV covariate  ferx  -382.381669  NONMEM  -390.0469773085871  Δ 7.67
//! ```
//!
//! Each ingredient is exact on its own; only the interaction is not. Measured
//! against the pre-#1073 baseline (`6c9af38b`) the same triple sat **10.91** out, so
//! this issue's fix *improves* it by 3.24 without closing it.
//!
//! The cause is structural and outside #1073: the **event-driven walk has no SS
//! pre-arrival seed**. The dense predictor has one (`ode/predictions.rs`, the
//! `ss_state_at_phase(… ii − lag)` call), and so does the analytic twin
//! (`K_SS_SEED` in `sens/ode_provider.rs`) — but the time-varying-covariate
//! production path has neither, so it equilibrates the periodic trough *at the
//! arrival* under the dose row's covariates instead of seeding it at the dose
//! record and propagating to the arrival under the next record's. With flat
//! covariates those are the same number, which is why the gap stayed invisible.
//!
//! Tracked as #1121. The two controls are asserted below; the triple ships as a committed
//! stream with its NONMEM result so whoever picks it up starts from the measurement.
//!
//! # E4 — EVID=4 reset + dose (`dose_form_lag_reset`)
//!
//! Oral, with an `EVID = 4` row at `t = 6` carrying a lagged dose. Reset, dose
//! record and arrival are then **three distinct instants** (6, 6, ≈6.7) whose
//! ordering is observable: the reset must zero the state before the dose record's
//! parameters take effect, and the dose must land at the arrival on the re-seeded
//! state.
//!
//! Ordering is exactly what this issue changed — a new `Kind::DoseRecord` had to be
//! given a rank against `Kind::Reset` and against a co-timed observation — so an
//! EVID=4 anchor is the one that fails loudly if that rank is wrong.
//!
//! # What these do **not** cover
//!
//! ferx's per-route absorption lag (`fn(..., lag = L)`, #859) composes a second
//! delay *inside one compartment*, which NONMEM cannot express: `ALAGn` is
//! per-compartment, so two composing lags need two compartments and stop being the
//! same object. That case is covered by the `Dual2`-vs-FD parity tests and the
//! ODE-twin equivalence in `sens/ode_provider_tests.rs` instead, with no external
//! anchor — see the note in `tvcov_route_lagged_zero_order_matches_production`.
//!
//! Tier 3: ODE evaluations over eight subjects; gated for **runtime**, not because
//! they run a convergence loop (`maxiter = 0`).

use std::path::PathBuf;

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// ferx FOCEI OFV at the same fixed parameters NONMEM evaluated (`maxiter = 0`,
/// covariance off — both set in the `.ferx` files).
fn ferx_ofv(model: &str, data: &str) -> f64 {
    let (result, _pop) = ferx_core::run_model_with_data(&anchor(model), Some(&anchor(data)))
        .expect("the anchor model and dataset load and evaluate");
    result.ofv
}

/// The repo's standard anchor tolerance: half an OFV unit.
const TOL: f64 = 0.5;

/// `nonmem_anchor/results/dose_form_lag_infusion.lst`.
const NM_OBJV_INFUSION: f64 = -436.381_586_377_188_11;

/// `nonmem_anchor/results/dose_form_lag_ss_flatwt.lst`.
const NM_OBJV_SS_FLATWT: f64 = 2_968.202_993_378_170_5;

/// `nonmem_anchor/results/dose_form_lag_ss_nolag.lst`.
const NM_OBJV_SS_NOLAG: f64 = 1_700.878_932_857_840_7;

/// `nonmem_anchor/results/dose_form_lag_reset.lst`.
const NM_OBJV_RESET: f64 = -455.383_750_284_223_65;

/// `nonmem_anchor/results/dose_form_lag_ss.ext` — the E3 triple itself.
const NM_OBJV_SS: f64 = -390.046_977_308_587_09;

/// `nonmem_anchor/results/dose_form_lag_ss_prearrival.ext`.
const NM_OBJV_SS_PREARRIVAL: f64 = -380.654_622_895_383_40;

/// `nonmem_anchor/results/dose_form_lag_ss_midstream.ext`.
const NM_OBJV_SS_MIDSTREAM: f64 = -476.729_556_897_566_06;

/// `nonmem_anchor/results/dose_form_lag_ss_infusion.ext`.
const NM_OBJV_SS_INFUSION: f64 = -458.896_244_660_178_47;

/// **E5 — a steady-state *infusion* with a lagtime, under a varying covariate
/// (#1121).**
///
/// Everything else in this file's SS family is an oral bolus, so the seed they
/// exercise is the plain `trough -> pulse -> decay` one. An SS infusion takes a
/// different branch of `ss_state_at_phase`: the phase `II - ALAG` the record is
/// seeded at lies past the end of the previous cycle's infusion window, so
/// reconstructing it means crossing the rate-off boundary — an active leg then a
/// quiet one — rather than decaying monotonically.
///
/// That branch had unit coverage under *flat* covariates only
/// (`ode_provider_ss_lagtime_infusion_matches_production`), and nothing anchored it
/// under varying ones — a gap in precisely the code this issue changed. 1-cpt IV,
/// `RATE`-defined (`AMT 100 / RATE 100`, a 1 h window), `II = 12`, with a sample at
/// `t = 0.5` inside every subject's pre-arrival window.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_steady_state_infusion_with_a_lagtime_under_tv_covariates() {
    let ofv = ferx_ofv(
        "dose_form_lag_infusion_fit.ferx",
        "dose_form_lag_ss_infusion.csv",
    );
    let delta = (ofv - NM_OBJV_SS_INFUSION).abs();
    assert!(
        delta < TOL,
        "SS infusion + lagtime: ferx {ofv:.6} vs NONMEM {NM_OBJV_SS_INFUSION:.6} \
         (Δ {delta:.3e})"
    );
}

/// **E3B — an `SS=1` record that is not the subject's first (#1121).**
///
/// E3 and E3P both open with the steady-state row, so the compartments are empty
/// before it and one half of what an `SS` record does stays invisible: it does not
/// merely *fill* the compartments, it **replaces** them — and *where* that
/// replacement happens is exactly what this issue moved, from the lagged arrival
/// back to the dose record.
///
/// Here a plain dose runs first and its drug is still present when an `SS=1` row
/// lands at `t = 12` with a lagtime. The sample at `t = 12.3` falls inside that
/// record's pre-arrival window for every subject, so it reads whichever state the
/// engine believes is there — the first dose's residual, if the replacement were
/// deferred to the arrival, or the steady-state tail, if it happens at the record.
/// Those differ by a large factor rather than a few percent, which makes this the
/// geometry where the two conventions separate loudly.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_steady_state_record_that_is_not_the_first_dose() {
    let ofv = ferx_ofv(
        "dose_form_lag_oral_fit.ferx",
        "dose_form_lag_ss_midstream.csv",
    );
    let delta = (ofv - NM_OBJV_SS_MIDSTREAM).abs();
    assert!(
        delta < TOL,
        "mid-stream SS record: ferx {ofv:.6} vs NONMEM {NM_OBJV_SS_MIDSTREAM:.6} \
         (Δ {delta:.3e})"
    );
}

/// **E3P — a sample *inside* the pre-arrival window (#1121).**
///
/// E3 pins the pre-arrival *propagation*: its first observation is at `t = 1`, past
/// every subject's arrival, so the seeded state is only ever read after it has
/// flowed through the arrival and the dose pulse. A seed wrong by an additive
/// constant, or taken at the wrong phase but compensated by the pulse, would still
/// pass there. This dataset adds one observation at `t = 0.5`, which lies strictly
/// inside `[dose row, arrival)` for all eight subjects (every lagtime is in
/// `[0.536, 0.951]`), and so reads the pre-arrival state directly.
///
/// It is also an invariance check of the #1073 kind. The new record carries
/// `WT = 140` — the value already governing the interval it splits — so adding it
/// must not move any other prediction. Taking a sample cannot change the
/// pharmacology, and if it does, the extra breakpoint is being given parameters it
/// should not have.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_sample_inside_a_steady_state_dose_pre_arrival_window() {
    let ofv = ferx_ofv(
        "dose_form_lag_oral_fit.ferx",
        "dose_form_lag_ss_prearrival.csv",
    );
    let delta = (ofv - NM_OBJV_SS_PREARRIVAL).abs();
    assert!(
        delta < TOL,
        "SS pre-arrival sample: ferx {ofv:.6} vs NONMEM {NM_OBJV_SS_PREARRIVAL:.6} \
         (Δ {delta:.3e})"
    );
}

/// **E3, the triple (#1121).** `SS = 1` with `II = 12` and an `ALAG1` whose
/// arrival lands inside a record interval carrying a different `WT`.
///
/// This was the case the two controls above were built to bracket: each
/// ingredient — steady state with a lagtime under flat covariates, and steady
/// state under a varying covariate with no lagtime — is exact on its own, and
/// only the interaction was not. It sat **7.67** OFV out (and **10.91** before
/// #1073's record convention improved it), because the periodic trough was
/// equilibrated at the *lagged arrival* under the dose row's snapshot rather than
/// seeded at the dose **record** and advanced to the arrival under the record
/// governing that interval.
///
/// The pointwise `PRED` test below localises the same defect without a fit and
/// runs on every PR; this asserts the objective the issue was measured on.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_steady_state_dose_whose_lagged_arrival_crosses_a_covariate_change() {
    let ofv = ferx_ofv("dose_form_lag_oral_fit.ferx", "dose_form_lag_ss.csv");
    let delta = (ofv - NM_OBJV_SS).abs();
    assert!(
        delta < TOL,
        "SS + lagtime + TV covariate: ferx {ofv:.6} vs NONMEM {NM_OBJV_SS:.6} (Δ {delta:.3e})"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_lagged_infusion_whose_window_end_crosses_a_covariate_change() {
    let ofv = ferx_ofv(
        "dose_form_lag_infusion_fit.ferx",
        "dose_form_lag_infusion.csv",
    );
    let delta = (ofv - NM_OBJV_INFUSION).abs();
    assert!(
        delta < TOL,
        "lagged-infusion anchor: ferx {ofv:.6} vs NONMEM {NM_OBJV_INFUSION:.6} (Δ {delta:.3e})"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_steady_state_dose_with_a_lagtime_under_flat_covariates() {
    // E3a. Isolates the lagtime: same SS geometry and the same data, every `WT`
    // set to 70. The SS trough, the pre-arrival window and the lagged pulse are all
    // exercised; only the covariate variation is removed.
    let ofv = ferx_ofv("dose_form_lag_oral_fit.ferx", "dose_form_lag_ss_flatwt.csv");
    let delta = (ofv - NM_OBJV_SS_FLATWT).abs();
    assert!(
        delta < TOL,
        "SS + lagtime under flat covariates: ferx {ofv:.6} vs NONMEM {NM_OBJV_SS_FLATWT:.6} \
         (Δ {delta:.3e})"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_a_steady_state_dose_under_time_varying_covariates_without_a_lagtime() {
    // E3b. The complementary isolation: the same SS dataset, the covariate varying
    // exactly as it does in the triple, but `ALAG1 = 0` on both sides. Together with
    // E3a this pins the residual disagreement to the *interaction* rather than to
    // steady state, to lagtimes, or to covariate handling on their own.
    let ofv = ferx_ofv("dose_form_lag_oral_nolag_fit.ferx", "dose_form_lag_ss.csv");
    let delta = (ofv - NM_OBJV_SS_NOLAG).abs();
    assert!(
        delta < TOL,
        "SS + TV covariate without a lagtime: ferx {ofv:.6} vs NONMEM {NM_OBJV_SS_NOLAG:.6} \
         (Δ {delta:.3e})"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_for_an_evid4_reset_carrying_a_lagged_dose() {
    let ofv = ferx_ofv("dose_form_lag_oral_fit.ferx", "dose_form_lag_reset.csv");
    let delta = (ofv - NM_OBJV_RESET).abs();
    assert!(
        delta < TOL,
        "EVID=4 + lagtime anchor: ferx {ofv:.6} vs NONMEM {NM_OBJV_RESET:.6} (Δ {delta:.3e}); \
         a wrong sort rank for the dose record against the reset or a co-timed \
         observation shows up here"
    );
}

// ---------------------------------------------------------------------------
// EBE-free `PRED` oracle (#1121)
// ---------------------------------------------------------------------------
//
// The OFV anchors above compare an *aggregate* at ferx's own EBEs against
// NONMEM's at its own — so a disagreement tells you there is one, not where it
// starts. NONMEM's `$TABLE` also carries **`PRED`**, the η = 0 population
// prediction, and these runs are `MAXEVAL=0` at exactly the `.ferx` initial
// estimates. That makes `PRED` a *pointwise* oracle with no empirical-Bayes step
// on either side: `predict()` builds `zero_eta = vec![0.0; n_eta + n_kappa]` and
// walks the same `compute_predictions_with_tv` dispatcher `fit()` uses, so the
// two sides evaluate the identical function at identical parameters.
//
// Three things follow, and all three are why these exist alongside the OFV
// anchors rather than instead of them:
//
//   * They **localise**. A per-observation comparison says *when* the trajectory
//     departs, which an objective cannot.
//   * They are **fast and ungated**. No fit, no convergence loop — so they run on
//     every PR and carry the diff's coverage. The `slow-tests` anchors above
//     never run on a PR and register none.
//   * They are **honest about magnitude**. Issue #1121's per-time table compares
//     ferx's `PRED` against NONMEM's `IPRED`, which conflates the state defect
//     with EBE drift and reads as a 2.9–3.4× error. Like for like at η = 0 the
//     defect is ≈ 2.1 % at `t = 1` and ≈ 0.8 % at the `t = 11` trough — small per
//     point, but worth 7.67 OFV once `ETA_LAG` absorbs it across 8 × 11
//     observations under a 5 % proportional error model.
//
// Tolerance: NONMEM writes `$TABLE` at five significant digits, i.e. ~1e-5
// relative, so 1e-4 is the tightest defensible bound — still 200× under the
// defect being measured.

/// `(ID, TIME) → PRED` from a NONMEM `$TABLE` file.
///
/// The table carries a row per *record*, including the `MDV=1` dose rows that
/// `predict()` returns nothing for, so callers look up by key rather than zipping
/// positionally.
fn nonmem_pred_table(tab: &str) -> std::collections::HashMap<(String, u64), f64> {
    let text = std::fs::read_to_string(anchor(tab)).expect("the NONMEM table is committed");
    let mut lines = text.lines();
    lines.next().expect("TABLE NO. banner");
    let header: Vec<&str> = lines
        .next()
        .expect("column header")
        .split_whitespace()
        .collect();
    let id_col = header.iter().position(|c| *c == "ID").expect("ID column");
    let time_col = header
        .iter()
        .position(|c| *c == "TIME")
        .expect("TIME column");
    let pred_col = header
        .iter()
        .position(|c| *c == "PRED")
        .expect("PRED column — the η=0 oracle these tests are built on");

    let mut out = std::collections::HashMap::new();
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

/// Compare every ferx η = 0 prediction against NONMEM's `PRED` at the same
/// `(ID, TIME)`.
fn assert_pred_matches_nonmem(model: &str, data: &str, tab: &str, max_relative: f64) {
    let nm = nonmem_pred_table(tab);
    let ferx = ferx_pred_table(model, data);
    assert!(!ferx.is_empty(), "predict() returned no rows for {data}");
    let mut compared = 0usize;
    for (id, time, pred) in &ferx {
        let Some(&want) = nm.get(&(id.clone(), time.to_bits())) else {
            panic!("no NONMEM PRED row for ID {id} at TIME {time} in {tab}");
        };
        let rel = (pred - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < max_relative,
            "ID {id} t={time}: ferx PRED {pred:.6} vs NONMEM {want:.6} \
             (relative {rel:.3e} > {max_relative:.0e})"
        );
        compared += 1;
    }
    // A silent zero-comparison pass is the failure mode this helper exists to
    // avoid; pin the count so a key mismatch cannot read as agreement.
    assert_eq!(
        compared,
        ferx.len(),
        "every ferx prediction must have been matched against a NONMEM row"
    );
}

/// **#1121, the pointwise form.** An `SS=1` dose whose lagged arrival crosses a
/// covariate change: the periodic trough must be seeded at the dose **record**
/// (phase `II − ALAG`) and propagated to the arrival under the record that
/// terminates that interval — not equilibrated *at* the arrival under the dose
/// row's snapshot.
///
/// Before the fix, subject 1 at `t = 1` reads `0.976020` against NONMEM's
/// `0.955530`.
#[test]
fn ferx_population_predictions_match_nonmem_for_a_lagged_ss_dose_under_tv_covariates() {
    assert_pred_matches_nonmem(
        "dose_form_lag_oral_fit.ferx",
        "dose_form_lag_ss.csv",
        "results/dose_form_lag_ss.tab",
        1e-4,
    );
}

/// **E3P pointwise — the pre-arrival state itself, read directly.**
///
/// The sharpest form of the #1121 check: `t = 0.5` sits inside every subject's
/// `[dose row, arrival)` window, so this compares the seeded steady-state tail
/// against NONMEM point by point *before* the arrival has had a chance to mask an
/// error in it. Fast and ungated, so it runs on every PR.
#[test]
fn ferx_population_predictions_match_nonmem_inside_a_steady_state_pre_arrival_window() {
    assert_pred_matches_nonmem(
        "dose_form_lag_oral_fit.ferx",
        "dose_form_lag_ss_prearrival.csv",
        "results/dose_form_lag_ss_prearrival.tab",
        1e-4,
    );
}

/// **E5 pointwise — the SS *infusion* seed, read inside its pre-arrival window.**
///
/// The rate-off-crossing branch of `ss_state_at_phase` under a varying covariate,
/// checked point by point rather than only through an objective.
#[test]
fn ferx_population_predictions_match_nonmem_for_a_lagged_steady_state_infusion() {
    assert_pred_matches_nonmem(
        "dose_form_lag_infusion_fit.ferx",
        "dose_form_lag_ss_infusion.csv",
        "results/dose_form_lag_ss_infusion.tab",
        1e-4,
    );
}

/// **E3B pointwise — a mid-stream `SS` record's pre-arrival window.**
///
/// The loudest of the pointwise checks: at `t = 12.3` the steady-state tail and the
/// preceding dose's residual are different by a large factor, so this separates
/// "the SS row replaces the compartments at the record" from "…at the arrival"
/// without needing tight tolerances to see it.
#[test]
fn ferx_population_predictions_match_nonmem_for_a_mid_stream_steady_state_record() {
    assert_pred_matches_nonmem(
        "dose_form_lag_oral_fit.ferx",
        "dose_form_lag_ss_midstream.csv",
        "results/dose_form_lag_ss_midstream.tab",
        1e-4,
    );
}

/// Control E3a, pointwise — the same SS + lagtime geometry with every `WT`
/// flattened to 70. Exact today, and it must **stay** exact after the fix, which
/// routes it through seed-and-propagate instead of a direct trough.
#[test]
fn ferx_population_predictions_match_nonmem_for_a_lagged_ss_dose_under_flat_covariates() {
    assert_pred_matches_nonmem(
        "dose_form_lag_oral_fit.ferx",
        "dose_form_lag_ss_flatwt.csv",
        "results/dose_form_lag_ss_flatwt.tab",
        1e-4,
    );
}

/// Control E3b, pointwise — the same TV covariates with `ALAG1 = 0`. Isolates
/// covariate handling from the lagtime, and must not move.
#[test]
fn ferx_population_predictions_match_nonmem_for_an_unlagged_ss_dose_under_tv_covariates() {
    assert_pred_matches_nonmem(
        "dose_form_lag_oral_nolag_fit.ferx",
        "dose_form_lag_ss.csv",
        "results/dose_form_lag_ss_nolag.tab",
        1e-4,
    );
}

/// **ferx's two production engines must agree with each other here (#1121).**
///
/// A time-varying-covariate subject routes to one of two event-driven walks
/// depending only on how the model is *written*: `[odes]` goes to
/// `ode::ode_predictions_event_driven`, an analytical `pk one_cpt_oral` to
/// `pk::event_driven::compute_predictions_event_driven`. Same system, same data,
/// same parameters — so the same numbers.
///
/// They were not. Both walks equilibrated a lagged SS dose's periodic trough at
/// the *arrival* rather than seeding it at the dose record, and the closed-form
/// walk additionally patched its pre-arrival predictions in a post-hoc pass over
/// `preds` that never touched the running state. Measured before the fix, the
/// closed-form model returned `-382.381669` on this dataset — bit-identical to
/// the ODE walk's pre-fix answer, both 7.665 OFV from NONMEM. That is what makes
/// this an anchor rather than a tautology: the `[odes]` side is externally pinned
/// to NONMEM 7.6.0 by
/// `ferx_matches_nonmem_for_a_steady_state_dose_whose_lagged_arrival_crosses_a_covariate_change`,
/// so agreement here transfers that anchor to the closed-form walk, which has no
/// NONMEM run of its own.
///
/// Tolerance is 1e-3 OFV, not the anchors' 0.5: this compares two ferx engines on
/// identical inputs, where the only legitimate difference is the closed form's
/// exact kernels against RK45 at `reltol 1e-10`. A gap large enough to matter
/// clinically would be thousands of times this.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_analytical_and_ode_engines_agree_on_a_lagged_steady_state_dose_under_tv_covariates() {
    let ode = ferx_ofv("dose_form_lag_oral_fit.ferx", "dose_form_lag_ss.csv");
    let closed_form = ferx_ofv("dose_form_lag_oral_cf_fit.ferx", "dose_form_lag_ss.csv");
    let delta = (ode - closed_form).abs();
    assert!(
        delta < 1e-3,
        "the [odes] and analytical engines disagree on the same model: \
         ODE {ode:.6} vs closed-form {closed_form:.6} (Δ {delta:.3e})"
    );
    // And both must still be the anchored answer — otherwise they could agree on a
    // shared wrong one, which is exactly the state this test was written to end.
    assert!(
        (ode - NM_OBJV_SS).abs() < TOL,
        "the ODE engine must remain NONMEM-anchored: {ode:.6} vs {NM_OBJV_SS:.6}"
    );
}
