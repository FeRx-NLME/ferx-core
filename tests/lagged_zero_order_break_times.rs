//! #1171 — a **lagged `zero_order` input rate** must reach every consumer of the
//! dense ODE engine, not only the objective's predictor.
//!
//! `zero_order(dur=D, lag=L)` is delivered as a *per-segment constant rate*: the
//! per-segment filter (`active_zero_order_inputs`) injects it only when the segment
//! is **fully contained** in the window `[t_dose + L, t_dose + L + D]`, so the
//! integrator must break the timeline at `w_start` as well as `w_end`. Two of the
//! four break-time builders in `src/ode/predictions.rs` pushed only `w_end`
//! (`push_zero_order_break_times`) and omitted the route-onset break
//! (`push_route_lag_break_times`), so the containment test failed for the segment
//! spanning `w_start` and the constant rate was dropped for the whole window. On a
//! model whose *only* input is a lagged `zero_order`, that leaves the compartment at
//! exactly `0.0` everywhere.
//!
//! The two defective builders feed six surfaces (the pure-Gaussian objective is
//! **not** one of them — `compute_predictions_with_tv` → `ode_predictions` always had
//! the call, which is why no existing test caught this):
//!
//! | builder | surface |
//! |---|---|
//! | `ode_predictions_with_states`' inline builder | sdtab IPRED + compartment states, plain subjects |
//! | `build_segment_break_times` → `ode_dense_solve_states` | sdtab states for TV-covariate / reset subjects |
//! | " | the **joint PK-TTE** `H`/`h` (`survival::ode_cumhaz_hazard`) |
//! | " | the Markov / CTMM endpoint NLL |
//! | " | the adaptive-dosing window AUC signal (#391) |
//! | " | `[derived]` grid integrals |
//! | `build_segment_break_times` → `ode_solve_until_chz_threshold` | `simulate()` event times |
//!
//! These are Tier-2 checks that run on every PR. The external oracle is
//! `tests/lagged_zero_order_nonmem_anchor.rs` (slow-gated); the checks here pin
//! cross-path consistency and the two objective/simulate surfaces the anchor cannot
//! reach. Every one of them returns `0.0` (or the drug-free baseline) without the fix
//! — see the per-test notes.

mod common;

use ferx_core::ode::predictions::ode_dense_solve_states;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::pk::{
    compute_event_pk_params, compute_predictions_with_states, compute_predictions_with_tv,
};
use ferx_core::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;

/// 1-cpt with a **single** input route: a lagged zero-order rate. Sole route on
/// purpose — a surviving `first_order` would mask the dropped window rather than
/// zero the compartment. `V = 50` so `central` is an amount and `central / V` the
/// concentration NONMEM's `PRED` column reports.
const LAGGED_ZO: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(2.0, 0.05, 24.0)
  theta TVLAG(1.5, 0.0, 12.0)

  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04

  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV * exp(ETA_V)
  DUR = TVDUR
  LAG = TVLAG

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  ode_reltol = 1e-9
  ode_abstol = 1e-9
";

/// The anchor dataset's observation times (`nonmem_anchor/lagged_zo.csv`). Both
/// zero-order windows — `[1.5, 3.5]` and `[13.5, 15.5]` — open strictly inside the
/// record range, and records sit on both sides of each.
const OBS_T: [f64; 10] = [2.3, 3.1, 4.2, 8.0, 11.5, 12.4, 13.8, 15.0, 18.0, 24.0];

/// Two doses. The second lands at `t = 12` with ~38 mg still in the compartment, so
/// the later window opens **with drug present** — a single-dose fixture would make
/// the pre-arrival side identically zero and could not tell a dropped window from a
/// correct one there (the non-degeneracy rule in `CLAUDE.md`).
fn two_doses() -> Vec<DoseEvent> {
    vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ]
}

fn lagged_zo_subject() -> Subject {
    common::subject(
        "1",
        two_doses(),
        OBS_T.to_vec(),
        vec![1.0; OBS_T.len()],
        vec![1; OBS_T.len()],
    )
}

fn single_subject_pop(subject: Subject) -> Population {
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject],
    }
}

/// Assert `got ≈ want` elementwise with a *relative* tolerance, and assert the
/// reference itself is non-degenerate — a comparison against an all-zero `want`
/// would pass vacuously in exactly the failure mode under test.
fn assert_rel_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        peak > 1e-3,
        "{what}: the reference is degenerate (peak {peak:.3e}) — the assertion would \
         pass vacuously"
    );
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(1e-12);
        assert!(
            rel <= tol,
            "{what}: element {i} is {g}, want {w} (rel {rel:.3e} > {tol:.1e})"
        );
    }
}

// ── The sdtab / prediction surfaces ──────────────────────────────────────────

/// The sdtab IPRED path (`compute_predictions_with_states`) must agree with the
/// objective's path (`compute_predictions_with_tv`) at the **same** η.
///
/// Both are public API and evaluate the same object; only the break-time builder
/// differs. (`predict()` is *not* a valid comparator here — it evaluates at
/// `zero_eta`, i.e. PRED, while sdtab IPRED is at η̂.) A non-zero η is used as well
/// as η = 0, so `CL`/`V` differ from their typical values and the window's decay is
/// not the one the population curve happens to have.
#[test]
fn with_states_ipred_matches_the_objective_path_at_the_same_eta() {
    let model = parse_model_string(LAGGED_ZO).expect("model must parse");
    let subject = lagged_zo_subject();
    let theta = &model.default_params.theta;

    for eta in [vec![0.0, 0.0], vec![0.3, -0.2]] {
        let objective = compute_predictions_with_tv(&model, &subject, theta, &eta);
        let (sdtab, _) = compute_predictions_with_states(&model, &subject, theta, &eta);
        assert_rel_close(
            &sdtab,
            &objective,
            1e-8,
            &format!("sdtab IPRED vs objective IPRED at eta={eta:?}"),
        );
    }
}

/// The compartment-state column returned alongside sdtab IPRED must be the same
/// solve: `states[j][central] / V == ipred[j]` (the model's `[scaling]`).
///
/// Without the fix the states are all `0.0` while — on this plain-subject path —
/// IPRED is zero too, so this pairs with the test above rather than replacing it.
#[test]
fn with_states_compartment_states_are_the_scaled_ipred() {
    let model = parse_model_string(LAGGED_ZO).expect("model must parse");
    let subject = lagged_zo_subject();
    let theta = &model.default_params.theta;
    let eta = vec![0.25, 0.15];

    let v = compute_event_pk_params(&model, &subject, theta, &eta).obs[0].values
        [ferx_core::types::PK_IDX_V];
    let (ipred, states) = compute_predictions_with_states(&model, &subject, theta, &eta);
    let scaled: Vec<f64> = states.iter().map(|u| u[0] / v).collect();
    assert_rel_close(&scaled, &ipred, 1e-12, "compartment states / V vs IPRED");
}

/// `ode_dense_solve_states` — the shared dense engine behind sdtab states for
/// TV-covariate subjects, the joint PK-TTE hazard, the Markov endpoint NLL, the
/// adaptive AUC signal and `[derived]` grid integrals — must reproduce the
/// objective path's concentrations at the observation times.
///
/// This is the *second* defective builder (`build_segment_break_times`); without
/// the fix every returned state is exactly `0.0`.
#[test]
fn dense_solve_states_matches_the_objective_path() {
    let model = parse_model_string(LAGGED_ZO).expect("model must parse");
    let subject = lagged_zo_subject();
    let theta = &model.default_params.theta;
    let eta = vec![0.1, -0.1];

    let ode = model.ode_spec.as_ref().expect("model is an [odes] model");
    let pk = compute_event_pk_params(&model, &subject, theta, &eta).obs[0];
    let v = pk.values[ferx_core::types::PK_IDX_V];

    let states = ode_dense_solve_states(ode, &pk.values, theta, &eta, &subject, &OBS_T);
    let conc: Vec<f64> = states.iter().map(|u| u[0] / v).collect();
    let objective = compute_predictions_with_tv(&model, &subject, theta, &eta);
    assert_rel_close(
        &conc,
        &objective,
        1e-8,
        "ode_dense_solve_states vs objective IPRED",
    );
}

/// A subject with **time-varying covariates** routes `compute_predictions_with_states`
/// to `ode_predictions_event_driven_with_states`, whose states half is
/// `ode_dense_solve_states` — the second defective builder. Its IPRED half comes from
/// `ode_predictions_event_driven` and was always correct, so on this path the
/// compartment columns were entirely zero while IPRED was right: the asymmetry that
/// hid the defect.
///
/// The covariates are **uniform-valued** on purpose. `Subject::has_tv_covariates` is
/// a structural `!dose_covariates.is_empty()` test, so the subject takes the
/// event-driven branch while the documented `W_DERIVED_CMT_TV_ODE` freezing error is
/// exactly zero — the states must equal the plain subject's. The precondition is
/// asserted below: a future value-aware predicate would silently destroy the
/// isolation rather than fail.
#[test]
fn tv_covariate_subject_states_match_the_plain_subject() {
    let model = parse_model_string(LAGGED_ZO).expect("model must parse");
    let theta = &model.default_params.theta;
    let eta = vec![0.0, 0.0];

    let plain = lagged_zo_subject();
    assert!(
        !plain.has_tv_covariates(),
        "the reference subject must take the plain (non-event-driven) branch"
    );
    let (_, want) = compute_predictions_with_states(&model, &plain, theta, &eta);

    let mut tv = lagged_zo_subject();
    let cov: HashMap<String, f64> = [("WT".to_string(), 70.0)].into_iter().collect();
    tv.covariates = cov.clone();
    tv.dose_covariates = vec![cov.clone(); tv.doses.len()];
    tv.obs_covariates = vec![cov; tv.obs_times.len()];
    assert!(
        tv.has_tv_covariates(),
        "isolation precondition: the probe subject must route to the event-driven \
         branch (a value-aware has_tv_covariates would break this test's premise)"
    );
    let (_, got) = compute_predictions_with_states(&model, &tv, theta, &eta);

    let flat = |s: &[Vec<f64>]| s.iter().map(|u| u[0]).collect::<Vec<_>>();
    assert_rel_close(
        &flat(&got),
        &flat(&want),
        1e-8,
        "TV-covariate subject states vs plain subject states",
    );
}

/// Metamorphic: `zero_order(dur=D, lag=L)` through the **states** path must equal
/// the compartment-lag twin `zero_order(dur=D)` + `LAGTIME = L`.
///
/// This is `tests/per_route_lag.rs`'s `route_lag_equals_compartment_lag_zero_order`
/// re-driven through `compute_predictions_with_states`. It is **not** an external
/// oracle: it discriminates only because the compartment-lag side pushes its onset
/// break from the dose loop (`dose.time + lag`), which every builder already had —
/// so the two sides disagree exactly when the route-lag side loses its break. The
/// external check is `tests/lagged_zero_order_nonmem_anchor.rs`.
#[test]
fn route_lag_equals_compartment_lag_through_the_states_path() {
    const COMP_LAG_ZO: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(2.0, 0.05, 24.0)
  theta TVLAG(1.5, 0.0, 12.0)

  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04

  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV * exp(ETA_V)
  DUR     = TVDUR
  LAGTIME = TVLAG

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  ode_reltol = 1e-9
  ode_abstol = 1e-9
";
    let route = parse_model_string(LAGGED_ZO).expect("route-lag model must parse");
    let comp = parse_model_string(COMP_LAG_ZO).expect("comp-lag twin must parse");
    let subject = lagged_zo_subject();
    let eta = vec![0.0, 0.0];

    let (route_ipred, _) =
        compute_predictions_with_states(&route, &subject, &route.default_params.theta, &eta);
    let (comp_ipred, _) =
        compute_predictions_with_states(&comp, &subject, &comp.default_params.theta, &eta);
    assert_rel_close(
        &route_ipred,
        &comp_ipred,
        1e-6,
        "route-lag vs compartment-lag zero_order through the states path",
    );
}

// ── The objective / simulate surfaces the anchor cannot reach ────────────────

/// A `[derived]` **grid integral** over `compartments[0]` goes through
/// `ode_dense_solve_states` on its own dense grid (`api/output_columns.rs`), not
/// through the per-observation states.
///
/// The reference is a closed-form mass balance, independent of both the integrator
/// and the trapezoid: for `dA/dt = R_in(t) − k·A` with `A(0) = 0`,
/// `∫₀ᵀ A dt = (∫₀ᵀ R_in dt − A(T)) / k`. Both windows close before `T = 24`, so
/// `∫R_in = 200` mg; `k` is the *individual* `CL/V` and `A(24)` the last compartment
/// state, both read back from the same fit — which makes the check independent of
/// wherever the inner loop happens to stop.
///
/// Both halves of the fix are load-bearing here, and neither can hide the other: the
/// grid integral comes from `build_segment_break_times` (it reads `0` without that
/// half) while `A(24)` comes from `ode_predictions_with_states` (it reads `0` without
/// *that* half, moving the reference instead). Measured: each mutation alone fails
/// this test.
#[test]
fn derived_grid_integral_over_a_lagged_zero_order_compartment() {
    let src = LAGGED_ZO.replace(
        "[error_model]",
        "[derived]\n  AUC = integral(compartments[0], from=0, to=24, step=0.005)\n\n[error_model]",
    );
    let parsed = ferx_core::parser::model_parser::parse_full_model(&src)
        .expect("model with [derived] must parse");
    let model = parsed.model;
    let pop = single_subject_pop(lagged_zo_subject());

    // The model's own `[fit_options]` — `maxiter = 0` and the tight ODE tolerances.
    // `FitOptions::default()` would let the outer loop move θ, and the mass-balance
    // reference below is written at the model's declared θ.
    let mut opts = parsed.fit_options;
    opts.verbose = false;
    opts.run_covariance_step = false;
    let result =
        ferx_core::fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");
    let sr = &result.subjects[0];

    let auc = sr
        .extra_columns
        .iter()
        .find(|(n, _)| n == "AUC")
        .expect("AUC column must exist")
        .1[0];

    // A(24) from the same run: the last compartment state, i.e. IPRED(24) · V.
    let a_end = *sr
        .compartment_states
        .last()
        .expect("per-obs compartment states")
        .first()
        .expect("central");
    // The elimination rate is the *individual* one — `fit` runs the inner loop even at
    // `maxiter = 0`, so `CL`/`V` carry η̂. Read it back from the reported η rather than
    // from the typical values.
    let theta = &model.default_params.theta;
    let (cl, v) = (theta[0] * sr.eta[0].exp(), theta[1] * sr.eta[1].exp());
    let want = (200.0 - a_end) / (cl / v);
    let rel = (auc - want).abs() / want;
    assert!(
        rel < 5e-4,
        "[derived] grid AUC {auc} vs the mass-balance reference {want} (rel {rel:.3e})"
    );
}

/// The **joint PK-TTE hazard**. `survival::ode_cumhaz_hazard` — reached here through
/// the public `predict_survival`, and by `stats::likelihood::tte_ode_nll` in the
/// objective — reads `H`/`h` out of `ode_dense_solve_states`.
///
/// Without the fix `central` is identically `0`, so `exp(BETA · central/V) = 1` and
/// the whole drug effect vanishes: `H(t)` collapses to the drug-free baseline
/// `H0 · t` and `h(t)` to a flat `H0`. That is not a diagnostic — it is half of the
/// objective, whenever `try_joint_pktte_shared_solve` declines.
///
/// The reference is the hazard's own closed form evaluated on the objective path's
/// (NONMEM-anchored) concentrations: `h(t) = H0 · exp(BETA · C(t))`. `H` is then
/// checked to be far from the drug-free `H0 · t` — the exact quantity the defect
/// returns.
#[cfg(feature = "survival")]
#[test]
fn joint_pk_tte_hazard_sees_the_lagged_zero_order_route() {
    use ferx_core::types::{EventType, ObsRecord};

    const H0: f64 = 0.01;
    const BETA: f64 = 0.5;
    const TTE: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(2.0, 0.05, 24.0)
  theta TVLAG(1.5, 0.0, 12.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)

  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  DUR  = TVDUR
  LAG  = TVLAG
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central

[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)

[simulation]
  horizon = 24

[fit_options]
  method     = focei
  maxiter    = 0
  ode_reltol = 1e-9
  ode_abstol = 1e-9
";
    let model = parse_model_string(TTE).expect("joint PK-TTE model must parse");

    // The PK twin — same disposition, no hazard — supplies the reference C(t) from
    // the objective path, which the NONMEM anchor pins.
    let pk_model = parse_model_string(LAGGED_ZO).expect("PK twin must parse");
    let pk_subject = {
        let mut s = lagged_zo_subject();
        s.obs_times = OBS_T.to_vec();
        s
    };
    let conc = compute_predictions_with_tv(
        &pk_model,
        &pk_subject,
        &pk_model.default_params.theta,
        &[0.0, 0.0],
    );

    let mut tte_subject = common::subject("1", two_doses(), vec![], vec![], vec![]);
    tte_subject.obs_records = vec![ObsRecord::Event {
        time: 24.0,
        event_type: EventType::RightCensored,
        entry_time: 0.0,
        cmt: 2,
    }];
    let pop = single_subject_pop(tte_subject);

    let rows = ferx_core::predict_survival(&model, &pop, &model.default_params, &OBS_T);
    assert_eq!(rows.len(), OBS_T.len(), "one survival row per grid point");

    let got_h: Vec<f64> = rows.iter().map(|r| r.hazard).collect();
    let want_h: Vec<f64> = conc.iter().map(|&c| H0 * (BETA * c).exp()).collect();
    assert_rel_close(
        &got_h,
        &want_h,
        1e-6,
        "h(t) vs H0·exp(BETA·C(t)) on the NONMEM-anchored concentrations",
    );

    // …and H must be nowhere near the drug-free baseline the defect returns. At
    // t = 24 the correct value is 0.4641 against a drug-free H0·24 = 0.24.
    let cum_at_24 = rows.last().expect("last grid point").cum_hazard;
    let drug_free = H0 * 24.0;
    assert!(
        cum_at_24 > 1.5 * drug_free,
        "H(24) = {cum_at_24} must exceed the drug-free baseline H0·24 = {drug_free} \
         by a wide margin; a value at the baseline means the PK compartment was zero"
    );
}

/// `simulate()` event times. `ode_solve_until_chz_threshold` walks the same
/// `build_segment_break_times` timeline, root-finding the first `t` where the
/// accumulated hazard reaches `−log u`.
///
/// The discriminator is the defect's own signature: with `central ≡ 0` the
/// drug-driven model's hazard is **exactly** the drug-free one, so `BETA = 0.5` and
/// `BETA = 0` produce bit-identical draws off the same seed. A subject who should
/// have an event at `t ≈ 9.4` is instead censored at the horizon.
#[cfg(feature = "survival")]
#[test]
fn simulated_event_times_see_the_lagged_zero_order_route() {
    use ferx_core::types::{EventType, ObsRecord};

    const TTE: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(2.0, 0.05, 24.0)
  theta TVLAG(1.5, 0.0, 12.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(BETA_VALUE, -10.0, 10.0)

  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  DUR  = TVDUR
  LAG  = TVLAG
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central

[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  ode_reltol = 1e-9
  ode_abstol = 1e-9
";
    let mut subject = common::subject("1", two_doses(), vec![], vec![], vec![]);
    subject.obs_records = vec![ObsRecord::Event {
        time: 24.0,
        event_type: EventType::RightCensored,
        entry_time: 0.0,
        cmt: 2,
    }];
    let pop = single_subject_pop(subject);

    let opts = ferx_core::SimulateOptions {
        horizon: Some(24.0),
        seed: Some(42),
        match_method: None,
    };
    let draw = |beta: &str| -> Vec<f64> {
        let model =
            parse_model_string(&TTE.replace("BETA_VALUE", beta)).expect("TTE model must parse");
        ferx_core::api::simulate_with_options(&model, &pop, &model.default_params, 20, &opts)
            .expect("simulation must run")
            .iter()
            .map(|s| s.time)
            .collect()
    };

    let drug_driven = draw("0.5");
    let drug_free = draw("0.0");
    assert_eq!(drug_driven.len(), 20, "20 draws");

    let n_events = drug_driven.iter().filter(|&&t| t < 24.0 - 1e-9).count();
    assert!(
        n_events > 0,
        "the drug-driven arm must fire at least one event before the horizon; \
         all-censored means the hazard never saw the absorbed drug"
    );
    assert!(
        drug_driven
            .iter()
            .zip(&drug_free)
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "BETA = 0.5 and BETA = 0 produced identical event times off the same seed — \
         the drug-driven hazard collapsed to the drug-free baseline, i.e. the PK \
         compartment was zero"
    );
}
