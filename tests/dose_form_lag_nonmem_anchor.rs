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
