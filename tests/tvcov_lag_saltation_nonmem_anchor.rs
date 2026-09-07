//! NONMEM anchor for #1060 — an ODE model with time-varying covariates and IIV on
//! an absorption lagtime, where the lagged dose arrival lands past a covariate
//! change.
//!
//! # Why an OFV comparison anchors a *gradient* fix
//!
//! #1060's predictions are bit-identical either side of the fix, so the natural
//! reading is that nothing NONMEM-comparable moves. That reading is wrong.
//! `estimation/inner_optimizer.rs` builds the FOCEI `h_matrix` — the `∂f/∂η` inside
//! `log|H'ΩH + R|` — from `sens::provider::subject_eta_jacobian`, the **analytic**
//! provider, whenever `analytic_inner_grad_supported`; `compute_jacobian_fd` is only
//! the fallback. A wrong analytic sensitivity therefore lands directly in the
//! reported objective, and `#OBJV` at fixed θ is a sharp anchor for it.
//!
//! So these runs are `$EST METHOD=1 INTERACTION MAXEVAL=0 POSTHOC` — one evaluation,
//! θ fixed at the values ferx also starts from, no convergence loop on either side.
//!
//! # The model
//!
//! Oral 1-cpt (`depot → central`), `ADVAN13 TOL=9` / ferx `[odes]`,
//! `ALAG1 = TVP·exp(ETA3)`, `WT` allometric on **both** `CL` and `KA`.
//!
//! The `KA` term is load-bearing, not decoration. With a plain `KA` the depot
//! compartment's own velocity carries no covariate, so the post-arrival field does
//! not jump across the boundary and the defect does not reproduce — the same reason
//! `sens/ode_provider_tests.rs` notes its oral variant "leaks only a few percent".
//! Putting a covariate where the dose lands is what makes the cell load-bearing.
//!
//! # A — single dose (`tvcov_lag_saltation.{ctl,csv}`)
//!
//! One dose at `t = 0` on a `WT = 70` row; the first record (`t = 1`) carries
//! `WT = 140`. All eight arrivals fall in `(0.52, 0.95)`, strictly inside the
//! crossed interval. Every dose is a *first* dose, so `g(x⁻) = 0` and only the
//! **post**-arrival snapshot is under test — #1060 in isolation.
//!
//! ```text
//! NONMEM 7.6.0  #OBJV   -302.91470812769074
//! ferx                  -302.914703            Δ = 5e-6
//! ```
//!
//! and the EBEs agree to every printed digit on all three etas, all eight subjects
//! (`nonmem_anchor/results/tvcov_lag_saltation.tab`, columns ETA1..ETA3):
//!
//! ```text
//!  ID     ETA1        ETA2        ETA3
//!   1   0.203521   -0.089843    0.279085
//!   2  -0.159938    0.139210   -0.330314
//!   3   0.052373    0.217718    0.132899
//!   4  -0.233000   -0.151794   -0.286676
//!   5   0.305743    0.023302    0.202363
//!   6  -0.029885   -0.194583   -0.123614
//!   7   0.118879    0.154685    0.294623
//!   8  -0.204741    0.032751   -0.073444
//! ```
//!
//! # D — control (`tvcov_lag_saltation_md_control.{ctl,csv}`)
//!
//! The multi-dose geometry with the second dose row carrying the **same** `WT` as
//! the record after it, so the two snapshots either side of the arrival agree.
//!
//! ```text
//! NONMEM 7.6.0  #OBJV   -415.64410902342507
//! ferx                  -415.644111            Δ = 2e-6
//! ```
//!
//! # B — the multi-dose crossing (#1073)
//!
//! `tvcov_lag_saltation_multidose.{ctl,csv}` is D with the dose row's `WT` set to
//! 150 while the next record keeps 75 — **one number different**. That one number
//! is what makes the interval `[dose record, lagged arrival]` discriminating, and
//! ferx used to disagree with NONMEM on `PRED` there by 14.89 OFV:
//!
//! ```text
//!                        before #1073   after #1073   NONMEM 7.6.0
//! #OBJV                  -474.660106    -459.772592   -459.77259187257408
//! Δ                      14.89          ~1e-5
//! ```
//!
//! The cause was a convention error, not a tolerance: ferx never broke the
//! timeline at the dose record's own time (`Kind::Dose` sat at `d.time + ALAG`),
//! so the dose row's covariate snapshot was stretched forward to the arrival.
//! NONMEM runs `$PK` at every record and ADVANs *to* it, so `(5, 6]` runs under
//! the `t = 6` dose record and the lagged dose is injected inside the **next**
//! record's advance, which runs at `WT = 75`.
//!
//! # C — the invariance that pins the direction
//!
//! `tvcov_lag_saltation_md_shrink.{ctl,csv}` is B plus a `WT = 75` record at
//! `t = 6.5`, i.e. **between** the dose row and the arrival (`≈ 6.7`). Under the
//! record convention that record changes nothing: `(6, 6.7]` already ran at the
//! next record's `WT = 75`, so splitting it in two at 6.5 — same `WT` on both
//! halves — must leave the answer alone. Subject 1's `PRED` at `t = 7`:
//!
//! ```text
//!          ferx before   ferx after    NONMEM
//!   B        0.816054     0.846892     0.846890
//!   C        0.886124     0.846892     0.846890
//!   D        0.896650     0.896650     0.896650
//! ```
//!
//! D isolates the disagreement to the interval `[dose record, lagged arrival]` — it
//! is B with the dose row's `WT` set to the next record's, so the two snapshots
//! agree and no convention can tell them apart. C pins the *direction*: NONMEM's
//! answer is invariant to inserting that record, and ferx's was not. Reproducing
//! that invariance is a stronger statement than matching B alone, because it holds
//! for a reason — the interval is governed by one record either way — rather than
//! by arithmetic coincidence.
//!
//! C's own `#OBJV` is not comparable: its injected `t = 6.5` row carries a
//! placeholder `DV`, so the stream exists for the `PRED` comparison, not a fit.
//!
//! Tier 3: an ODE evaluation over eight subjects; gated for **runtime**, not because
//! it runs a convergence loop (`maxiter = 0`).

use std::path::PathBuf;

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// ferx FOCEI OFV at the same fixed parameters NONMEM evaluated (`maxiter = 0`,
/// covariance off — both set in `tvcov_lag_saltation_fit.ferx`).
fn ferx_ofv(data: &str) -> f64 {
    let (result, _pop) = ferx_core::run_model_with_data(
        &anchor("tvcov_lag_saltation_fit.ferx"),
        Some(&anchor(data)),
    )
    .expect("the anchor model and dataset load and evaluate");
    result.ofv
}

/// The repo's standard anchor tolerance: half an OFV unit. The measured margins are
/// 5e-6 and 2e-6, so this leaves four orders of magnitude of headroom for solver and
/// platform drift while still catching the 14.89-unit class of defect #1060 was.
const TOL: f64 = 0.5;

/// `nonmem_anchor/results/tvcov_lag_saltation.lst`.
const NM_OBJV_SINGLE: f64 = -302.914_708_127_690_74;

/// `nonmem_anchor/results/tvcov_lag_saltation_md_control.lst`.
const NM_OBJV_CONTROL: f64 = -415.644_109_023_425_07;

/// `nonmem_anchor/results/tvcov_lag_saltation_multidose.lst`.
const NM_OBJV_MULTIDOSE: f64 = -459.772_591_872_574_08;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_when_a_lagged_arrival_crosses_a_covariate_change() {
    let ofv = ferx_ofv("tvcov_lag_saltation.csv");
    let delta = (ofv - NM_OBJV_SINGLE).abs();
    assert!(
        delta < TOL,
        "single-dose crossing anchor: ferx {ofv:.6} vs NONMEM {NM_OBJV_SINGLE:.6} \
         (Δ {delta:.3e}); before #1060's fix this sat 7.52 units out"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_when_the_dose_row_and_the_next_record_agree() {
    // The control for the B/C divergence documented above: same multi-dose geometry,
    // but the dose row carries the next record's covariate value, so the interval
    // `[dose record, lagged arrival]` is governed by the same parameters under either
    // convention. This must stay green through any fix to that divergence.
    let ofv = ferx_ofv("tvcov_lag_saltation_md_control.csv");
    let delta = (ofv - NM_OBJV_CONTROL).abs();
    assert!(
        delta < TOL,
        "multi-dose control anchor: ferx {ofv:.6} vs NONMEM {NM_OBJV_CONTROL:.6} \
         (Δ {delta:.3e})"
    );
}

/// Population `PRED` (η = 0) for one subject at one time, straight from
/// [`ferx_core::predict`] — no fit, so this costs one evaluation rather than a
/// posthoc pass, and it reads the *population* prediction the NONMEM `$TABLE`
/// `PRED` column reports.
fn ferx_pred_at(data: &str, subject_id: &str, time: f64) -> f64 {
    use std::path::Path;
    let parsed = ferx_core::parser::model_parser::parse_full_model_file(Path::new(&anchor(
        "tvcov_lag_saltation_fit.ferx",
    )))
    .expect("the anchor model parses");
    let pop = ferx_core::read_nonmem_csv(Path::new(&anchor(data)), None, None)
        .expect("the anchor dataset loads");
    let preds = ferx_core::predict(&parsed.model, &pop, &parsed.model.default_params);
    preds
        .iter()
        .find(|p| p.id == subject_id && (p.time - time).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no prediction for subject {subject_id} at t={time} in {data}"))
        .pred
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_matches_nonmem_when_a_lagged_arrival_crosses_a_covariate_change_mid_regimen() {
    // #1073: the discriminating cell. One number apart from the control above — the
    // `t = 6` dose row carries `WT = 150` while the record after it carries 75 — so
    // the interval `[dose record, lagged arrival]` is the only thing the two
    // conventions disagree about, and the second dose lands with residual drug
    // present so both sides of the arrival are live.
    //
    // Stretching the dose row's snapshot forward to the arrival put this 14.89 OFV
    // out. All four engines shared the error and it moves converged fits, so the
    // anchor is asserted rather than documented.
    let ofv = ferx_ofv("tvcov_lag_saltation_multidose.csv");
    let delta = (ofv - NM_OBJV_MULTIDOSE).abs();
    assert!(
        delta < TOL,
        "multi-dose crossing anchor: ferx {ofv:.6} vs NONMEM {NM_OBJV_MULTIDOSE:.6} \
         (Δ {delta:.3e}); before #1073's fix this sat 14.89 units out"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_record_inside_the_dose_to_arrival_window_does_not_move_the_prediction() {
    // C, asserted as an invariance rather than against its own `#OBJV` (the injected
    // `t = 6.5` row carries a placeholder `DV`, so that number is meaningless).
    //
    // Splitting `[dose record, arrival]` with a record carrying the SAME `WT` as the
    // one already governing it must change nothing. That is NONMEM's behaviour —
    // `PRED` at `t = 7` is 0.846890 for both B and C — and exactly what ferx got
    // wrong: stretching the dose row's snapshot made the inserted record shorten the
    // stretch, so B and C disagreed by 8.6 %.
    //
    // This is the stronger half of the pair. B pins one number; C pins the *reason*,
    // and would still fail a fix that landed on B's value by arithmetic coincidence
    // rather than by governing the interval with a single record.
    let b = ferx_pred_at("tvcov_lag_saltation_multidose.csv", "1", 7.0);
    let c = ferx_pred_at("tvcov_lag_saltation_md_shrink.csv", "1", 7.0);
    assert!(
        (b - c).abs() < 1e-6,
        "inserting a same-covariate record inside [dose row, arrival] moved PRED: \
         B {b:.6} vs C {c:.6} (NONMEM gives 0.846890 for both)"
    );
    // Non-degeneracy: two wrong-but-equal numbers would satisfy the invariance, so
    // pin the shared value to NONMEM's as well.
    assert!(
        (b - 0.846_890).abs() < 1e-4,
        "B and C agree at {b:.6}, but NONMEM gives 0.846890"
    );
}
