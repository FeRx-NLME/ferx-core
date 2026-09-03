use super::*;
use crate::types::DoseEvent;
use approx::assert_relative_eq;

/// 1-cpt IV bolus ODE: dA/dt = -ke·A. RHS reads CL,V from pk_params_flat.
fn one_cpt_ode_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

fn pk_one(cl: f64, v: f64) -> PkParams {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = cl;
    p.values[crate::types::PK_IDX_V] = v;
    p
}

fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
    let n_obs = obs_times.len();
    Subject {
        id: "1".into(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n_obs],
        obs_cmts: vec![1; n_obs],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n_obs],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// A Form C readout that returns nothing but the model-time thread-local, so a
/// prediction vector *is* the sequence of `TIME` values the readout saw (#1028).
fn time_readout_ode_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::Single(Box::new(|_state, _pk, _theta, _eta, _cov| {
            crate::parser::model_parser::current_model_time()
        })),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

/// The Form C readout's `TIME` is the **raw data-file clock** (`obs_raw_times`),
/// not the shifted internal timeline `obs_times` carries for subjects with stacked
/// reset occasions (#1028). That is the convention every other per-record object
/// already uses — sdtab/covtab `TIME`, `predict()`/`simulate()` `TIME`, `[derived]`
/// integral windows, and the custom residual-magnitude model
/// (`ModelParameters::ruv_obs_mult`, documented as matching what NONMEM's `$ERROR`
/// sees) — and a Form C readout is the `$ERROR` twin. Before this, a reset-stacked
/// dataset got the integrator clock in the readout and the data clock in the
/// residual-magnitude model, for the same record.
#[test]
fn form_c_readout_time_is_the_raw_data_clock() {
    // Two occasions of 0 h / 1 h, stacked onto a monotonic 0/1/2/3 integrator grid.
    let mut subj = make_subject(Vec::new(), vec![0.0, 1.0, 2.0, 3.0]);
    subj.obs_raw_times = vec![0.0, 1.0, 0.0, 1.0];

    let ode = time_readout_ode_spec();
    let pk = pk_one(5.0, 1.0);
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert_eq!(
        preds, subj.obs_raw_times,
        "the readout must see the raw data-file TIME, got {preds:?}"
    );

    // Fallback: an in-memory subject with no `obs_raw_times` keeps reading
    // `obs_times`, so nothing changes for the (overwhelmingly common) no-reset case.
    let plain = make_subject(Vec::new(), vec![0.0, 1.0, 2.0, 3.0]);
    let preds_plain = ode_predictions(&ode, &pk.values, &[], &[], &plain);
    assert_eq!(
        preds_plain, plain.obs_times,
        "with no raw times the readout falls back to obs_times, got {preds_plain:?}"
    );
}

/// 1-cpt IV bolus + a cumulative-hazard accumulator: state 0 = central
/// (`dC/dt = -ke·C`, `ke = CL/V`), state 1 = CHZ (`dCHZ/dt = 0.1·C`). With a
/// bolus `amt` at t=0 this has the closed form `C(t) = amt·e^{-ke t}`,
/// `CHZ(t) = 0.1·amt·(1 - e^{-ke t})/ke`.
#[cfg(feature = "survival")]
fn one_cpt_chz_ode_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
            dy[1] = 0.1 * y[0];
        }),
        n_states: 2,
        state_names: vec!["central".into(), "chz".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions {
            abstol: 1e-10,
            reltol: 1e-9,
            ..OdeSolverOptions::default()
        },
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

/// Bolus-driven crossing matches the closed form: `amt=100`, `CL=10`, `V=100`
/// ⇒ `ke=0.1`, `CHZ(t)=100(1-e^{-0.1t})`; solving `CHZ=50` gives `t = 10·ln2`.
#[cfg(feature = "survival")]
#[test]
fn until_chz_threshold_bolus_crossing_matches_closed_form() {
    let ode = one_cpt_chz_ode_spec();
    let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
    let pk = pk_one(10.0, 100.0);
    match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, 50.0, 1000.0) {
        ThresholdOutcome::Crossed(t) => {
            assert_relative_eq!(t, 10.0 * std::f64::consts::LN_2, epsilon = 1e-4)
        }
        other => panic!("expected Crossed, got {other:?}"),
    }
}

/// Parity pin against the fit-path orchestration: the crossing time the wrapper
/// returns, fed back through `ode_dense_solve_states`, must read a CHZ equal to
/// the threshold. If the restated segment loop ever drifts from the dense one,
/// `CHZ_dense(t_cross) ≠ threshold` and this fails.
#[cfg(feature = "survival")]
#[test]
fn until_chz_threshold_parity_with_dense_solve() {
    let ode = one_cpt_chz_ode_spec();
    let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
    let pk = pk_one(10.0, 100.0);
    let threshold = 37.0;
    let t = match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, threshold, 1000.0) {
        ThresholdOutcome::Crossed(t) => t,
        other => panic!("expected Crossed, got {other:?}"),
    };
    let states = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[t]);
    assert_relative_eq!(states[0][1], threshold, epsilon = 1e-4);
}

/// #570 driver-level share: `ode_predictions_and_chz` returns the Gaussian
/// predictions **bit-identical** to `ode_predictions` (the obs `saveat` / step
/// sequence is untouched) and the **full** ODE state at every event time equal to
/// the dedicated `ode_dense_solve_states` read it replaces — proving one
/// integration now serves both consumers.
///
/// The event times deliberately exercise the boundary cases the open-interval soft
/// filter alone gets wrong (regression for the two bugs in the #613 review):
///   - `0.0` = the integration start (a segment's *left* boundary). The open
///     `(t_start, t_end]` solve never reads it, so before the left-boundary handler
///     it stayed NaN → the TTE `1e20` sentinel (e.g. an interval-censored `left=0`).
///   - `24.0` = an *interior dose time*. The shared state must be **post-dose** (the
///     dedicated path overwrites with the post-dose state); reading the pre-dose
///     value would move the instantaneous hazard `h = dCHZ/dt` for an event there.
/// plus interior (`1/6/18`), on the obs grid (`6`), and **past** the last obs
/// (`33 > 30`, exercising the `t_last` extension).
#[cfg(feature = "survival")]
#[test]
fn ode_predictions_and_chz_shares_one_solve() {
    let ode = one_cpt_chz_ode_spec();
    let subject = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        vec![0.5, 2.0, 6.0, 12.0, 24.5, 30.0],
    );
    let pk = pk_one(10.0, 100.0);
    // Sorted, unique TTE times: integration start, interior, on-grid, interior dose
    // time, and past last obs.
    let chz_times = vec![0.0, 1.0, 6.0, 18.0, 24.0, 33.0];

    let (ipred, chz_states) =
        ode_predictions_and_chz(&ode, &pk.values, &[], &[], &subject, &chz_times);
    let ipred_ref = ode_predictions(&ode, &pk.values, &[], &[], &subject);
    let chz_ref = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &chz_times);

    // (1) Predictions bit-identical — the shared solve does not move the fit ipred,
    // even with a soft time at the integration start and on an interior dose break.
    assert_eq!(
        ipred, ipred_ref,
        "ode_predictions_and_chz must not change the predictions"
    );

    // (2) The full state (PK compartment *and* the CHZ accumulator) at each event
    // time matches the dedicated clamped solve to solver tolerance. Checking both
    // components is what catches the interior-dose case: CHZ (slot 1) is continuous
    // across a dose, but the PK compartment (slot 0) jumps, so a pre- vs post-dose
    // read only shows up there.
    assert_eq!(chz_states.len(), chz_times.len());
    for (i, st) in chz_states.iter().enumerate() {
        assert_eq!(st.len(), ode.n_states, "state {i} must be fully populated");
        for j in 0..ode.n_states {
            assert_relative_eq!(st[j], chz_ref[i][j], max_relative = 1e-5);
        }
    }
}

/// #731 / #570 parity: a dose landing **exactly on the terminal CHZ time**
/// (`t_last`) must read post-dose on *both* the shared solve
/// (`ode_predictions_and_chz`) and the dedicated two-solve reference
/// (`ode_dense_solve_states`). The shared solve was fixed by #731 to visit the
/// final break as a left boundary; this pins that the dense reference now does
/// too, so the one-solve == two-solve equivalence holds at a dose-on-`t_last`.
/// Before the dense-side fix this read the pre-dose state (`≈9.07`), diverging
/// from the shared solve's post-dose `≈109.07` — the gap that broke #570.
#[cfg(feature = "survival")]
#[test]
fn terminal_dose_on_chz_time_reads_post_dose_on_both_paths() {
    let ode = one_cpt_chz_ode_spec();
    // Boluses at t=0 and t=24; max obs = 6, so t_last = max(6, chz 24) = 24 — a
    // dose lands exactly on the terminal break and the terminal CHZ time.
    let subject = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        vec![6.0],
    );
    let pk = pk_one(10.0, 100.0); // ke = CL/V = 0.1
    let chz_times = vec![24.0];

    let (_ipred, chz_states) =
        ode_predictions_and_chz(&ode, &pk.values, &[], &[], &subject, &chz_times);
    let chz_ref = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &chz_times);

    // (1) The two paths agree at the dose-coincident terminal time — full state,
    // PK compartment (slot 0) and CHZ accumulator (slot 1).
    for j in 0..ode.n_states {
        assert_relative_eq!(chz_states[0][j], chz_ref[0][j], max_relative = 1e-5);
    }
    // (2) And the read is genuinely post-dose: the t=0 bolus decayed 24 h plus the
    // fresh 100 mg (≈109.07), not the pre-dose ≈9.07 the dropped-dose bug produced.
    let ke = 0.1_f64;
    assert_relative_eq!(
        chz_states[0][0],
        100.0 * (-ke * 24.0).exp() + 100.0,
        max_relative = 1e-4
    );
}

/// A threshold above the asymptotic cumulative hazard (`CHZ → 100`) is never
/// reached ⇒ the draw is censored at the horizon, not failed.
#[cfg(feature = "survival")]
#[test]
fn until_chz_threshold_censors_when_unreached() {
    let ode = one_cpt_chz_ode_spec();
    let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
    let pk = pk_one(10.0, 100.0);
    assert_eq!(
        ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, 200.0, 50.0),
        ThresholdOutcome::CensoredAtHorizon
    );
}

/// An infusion input is handled like the dense path: a 100-unit infusion over
/// 10 h (`rate = 10`) gives the same total exposure as the bolus, so the same
/// asymptotic `CHZ → 100`. The crossing time reads back (via the dense solve) a
/// CHZ equal to the threshold — the parity pin, now over the infusion branch.
#[cfg(feature = "survival")]
#[test]
fn until_chz_threshold_infusion_parity_with_dense_solve() {
    let ode = one_cpt_chz_ode_spec();
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)],
        vec![],
    );
    let pk = pk_one(10.0, 100.0);
    let threshold = 50.0;
    let t = match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, threshold, 1000.0) {
        ThresholdOutcome::Crossed(t) => t,
        other => panic!("expected Crossed, got {other:?}"),
    };
    let states = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[t]);
    assert_relative_eq!(states[0][1], threshold, epsilon = 1e-4);
}

/// 1-cpt with an *invalid* negative hazard (`dCHZ/dt = -0.1·C`): the accumulator
/// decreases, so a crossing can never be well-defined.
#[cfg(feature = "survival")]
fn one_cpt_neg_chz_ode_spec() -> OdeSpec {
    let mut ode = one_cpt_chz_ode_spec();
    ode.rhs = Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
        let cl = p[crate::types::PK_IDX_CL];
        let v = p[crate::types::PK_IDX_V];
        let ke = if v > 0.0 { cl / v } else { 0.0 };
        dy[0] = -ke * y[0];
        dy[1] = -0.1 * y[0];
    });
    ode
}

/// A negative hazard propagates the primitive's `Failed` to a `SolveFailed` —
/// never a silent censor (the wrapper's failure arm).
#[cfg(feature = "survival")]
#[test]
fn until_chz_threshold_negative_hazard_fails() {
    let ode = one_cpt_neg_chz_ode_spec();
    let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
    let pk = pk_one(10.0, 100.0);
    match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, 5.0, 1000.0) {
        ThresholdOutcome::SolveFailed(msg) => {
            assert!(msg.contains("non-monotone"), "msg: {msg}")
        }
        other => panic!("expected SolveFailed, got {other:?}"),
    }
}

/// A bolus dose-ledger row, for the signal-AUC pass test below.
fn ledger_bolus(time: f64, amt: f64, cmt: usize) -> DoseLedgerEntry {
    DoseLedgerEntry {
        subject: "1".into(),
        draw: 0,
        sim: 0,
        dose_idx: 0,
        time,
        amt,
        cmt,
        rate: 0.0,
        decision_idx: 0,
        rule_fired: "bolus".into(),
        observed_signals: Vec::new(),
        pre_state: None,
        post_state: None,
        f_applied: 1.0,
    }
}

/// Analytic accuracy check for the metrics-only signal-AUC pass
/// ([`adaptive_window_signal_aucs`], #391 S2.5b). For a 1-cpt model with a single
/// IV bolus `D` at t = 0, the amount readout is `A(t) = D·e^{-ke t}` (ke = CL/V),
/// so the exposure over a window `[a, b]` *after* the dose is the closed form
/// `∫ₐᵇ A dt = (D/ke)(e^{-ke a} − e^{-ke b})`. The window is placed strictly after
/// the bolus so every grid edge sits in the smooth decay phase (no dose
/// discontinuity to straddle), and the trapezoid converges to that closed form.
#[test]
fn adaptive_window_signal_aucs_matches_closed_form() {
    let ode = one_cpt_ode_spec(); // readout = ObsCmt(0): the central amount
    let pk = pk_one(10.0, 100.0); // ke = CL/V = 0.1
    let (ke, d) = (0.1_f64, 100.0_f64);
    let base = make_subject(vec![], vec![]); // dose-free; the pass uses its own grid
    let ledger = vec![ledger_bolus(0.0, d, 1)];
    let auc = |a: f64, b: f64| (d / ke) * ((-ke * a).exp() - (-ke * b).exp());

    // Single window [2, 10], entirely in the decay phase.
    let one = adaptive_window_signal_aucs(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[2.0, 10.0],
        &ledger,
        None,
        1,
    );
    assert_eq!(one.len(), 1);
    // 128-panel trapezoid of a smooth decay ⇒ ~3e-6 relative to the closed form.
    assert_relative_eq!(one[0], auc(2.0, 10.0), max_relative = 1e-4);

    // Two windows [2,6],[6,10]: exercises per-window splitting *and* state
    // continuity across the shared boundary (the second window must resume the
    // decay, not restart from the initial state).
    let two = adaptive_window_signal_aucs(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[2.0, 6.0, 10.0],
        &ledger,
        None,
        1,
    );
    assert_eq!(two.len(), 2);
    assert_relative_eq!(two[0], auc(2.0, 6.0), max_relative = 1e-4);
    assert_relative_eq!(two[1], auc(6.0, 10.0), max_relative = 1e-4);
    // The two windows partition [2, 10], so their exposures sum to the total.
    assert_relative_eq!(two[0] + two[1], auc(2.0, 10.0), max_relative = 1e-4);

    // Fewer than two decisions ⇒ no closed window ⇒ empty (the metric is `None`).
    let none =
        adaptive_window_signal_aucs(&ode, &pk.values, &[], &[], &base, &[2.0], &ledger, None, 1);
    assert!(none.is_empty());
}

/// Regression (#391 S2.5b): a dose landing exactly on a window's **right**
/// boundary belongs to the *next* window and must not inflate this one. Because
/// [`ode_dense_solve_states`] saves the post-dose state at a save point on a dose
/// time, an earlier single-grid implementation folded the next bolus's jump into
/// the preceding window's right endpoint (≈ ½·Δsignal·(window ⁄ panels)). With a
/// bolus at *every* decision time, each window must integrate only the doses at
/// or before its own left edge. This test fails on that earlier implementation.
#[test]
fn adaptive_window_signal_aucs_excludes_boundary_dose() {
    let ode = one_cpt_ode_spec(); // readout = central amount, RHS = -ke·y
    let pk = pk_one(10.0, 100.0); // ke = CL/V = 0.1
    let (ke, d) = (0.1_f64, 100.0_f64);
    let base = make_subject(vec![], vec![]);
    // A bolus at each decision time 0, 24, 48 ⇒ windows [0,24] and [24,48].
    let ledger = vec![
        ledger_bolus(0.0, d, 1),
        ledger_bolus(24.0, d, 1),
        ledger_bolus(48.0, d, 1),
    ];
    let aucs = adaptive_window_signal_aucs(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0, 24.0, 48.0],
        &ledger,
        None,
        1,
    );
    assert_eq!(aucs.len(), 2);

    // Window 0 = [0,24]: exposure from the t=0 bolus ONLY (the t=24 bolus is the
    // next window's). Post-dose A(0) = D, so ∫₀²⁴ = (D/ke)(1 − e^{−24ke}). The
    // buggy single-grid version reported ≈ +9.4 (≈ +1%) here from the t=24 jump.
    let w0 = (d / ke) * (1.0 - (-ke * 24.0).exp());
    assert_relative_eq!(aucs[0], w0, max_relative = 1e-4);

    // Window 1 = [24,48]: superposition of the t=0 and t=24 boluses, decaying
    // from the post-dose amount A(24⁺) = D·e^{−24ke} + D; the t=48 bolus excluded.
    let a24 = d * (-ke * 24.0).exp() + d;
    let w1 = (a24 / ke) * (1.0 - (-ke * 24.0).exp());
    assert_relative_eq!(aucs[1], w1, max_relative = 1e-4);
}

/// Regression: the metrics-only signal-AUC pass must integrate the pre-scheduled
/// **base regimen** (#702), not just the controller's realized ledger. The earlier
/// `sub.doses = ledger` overwrite dropped any loading / maintenance dose carried on
/// `base_subject.doses`, silently under-counting the exposure behind
/// `auc_target_attainment`. Here a base IV loading bolus at t = 0 precedes the first
/// decision window `[24, 48]`, so its decayed contribution to the post-dose amount at
/// t = 24 must appear in that window's AUC.
#[test]
fn adaptive_window_signal_aucs_includes_base_loading_dose() {
    let ode = one_cpt_ode_spec(); // readout = central amount, RHS = -ke·y
    let pk = pk_one(10.0, 100.0); // ke = CL/V = 0.1
    let (ke, d_base, d_ctrl) = (0.1_f64, 200.0_f64, 100.0_f64);

    // Base regimen: a loading bolus at t=0, carried on the SUBJECT (not the ledger).
    let with_base = make_subject(
        vec![DoseEvent::new(0.0, d_base, 1, 0.0, false, 0.0)],
        vec![],
    );
    // Controller decisions at 24, 48 ⇒ one window [24, 48]; a controller bolus at 24.
    let ledger = vec![ledger_bolus(24.0, d_ctrl, 1)];
    let decisions = [24.0, 48.0];

    let aucs = adaptive_window_signal_aucs(
        &ode,
        &pk.values,
        &[],
        &[],
        &with_base,
        &decisions,
        &ledger,
        None,
        1,
    );
    assert_eq!(aucs.len(), 1);

    // Degenerate oracle: post-dose amount at 24⁺ is the base bolus decayed from t=0
    // PLUS the fresh controller bolus, and the window then decays smoothly (no dose at
    // 48), so the closed form is exact. A(24⁺) = D_base·e^{−24ke} + D_ctrl.
    let a24 = d_base * (-ke * 24.0).exp() + d_ctrl;
    let want = (a24 / ke) * (1.0 - (-ke * 24.0).exp());
    assert_relative_eq!(aucs[0], want, max_relative = 1e-4);

    // Positive control (non-vacuous): a dose-free base — the pre-fix behaviour, where
    // only the ledger survived — leaves just the controller bolus and a materially
    // smaller AUC. The base loading dose adds ≈ 18% here; assert the two differ well
    // beyond any trapezoid / solver noise so the oracle above has real teeth.
    let dose_free = make_subject(vec![], vec![]);
    let without_base = adaptive_window_signal_aucs(
        &ode,
        &pk.values,
        &[],
        &[],
        &dose_free,
        &decisions,
        &ledger,
        None,
        1,
    );
    let want_ctrl_only = (d_ctrl / ke) * (1.0 - (-ke * 24.0).exp());
    assert_relative_eq!(without_base[0], want_ctrl_only, max_relative = 1e-4);
    assert!(
        aucs[0] > without_base[0] * 1.1,
        "base loading dose not reflected: with-base {} vs ledger-only {}",
        aucs[0],
        without_base[0]
    );
}

/// Regression: a base **maintenance** dose landing strictly INSIDE a decision window
/// `(a, b)` — the common MIPD pattern of dosing more often than TDM sampling — must
/// also be integrated (`time < b`, not `<= a`). Dropping it would under-count that
/// window entirely. Here a base bolus at t = 12 sits inside the single window
/// `[0, 24]`; the mutation `time <= a` (a = 0) would exclude it and collapse the
/// with-base AUC onto the ledger-only value, which this test forbids.
#[test]
fn adaptive_window_signal_aucs_includes_mid_window_base_dose() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(10.0, 100.0); // ke = 0.1
    let (ke, d_ctrl, d_mid) = (0.1_f64, 100.0_f64, 300.0_f64);

    // Controller bolus at t=0 (decision at 0); base maintenance bolus at t=12.
    let with_base = make_subject(
        vec![DoseEvent::new(12.0, d_mid, 1, 0.0, false, 0.0)],
        vec![],
    );
    let dose_free = make_subject(vec![], vec![]);
    let ledger = vec![ledger_bolus(0.0, d_ctrl, 1)];
    let decisions = [0.0, 24.0];

    let call = |sub: &Subject| {
        adaptive_window_signal_aucs(
            &ode,
            &pk.values,
            &[],
            &[],
            sub,
            &decisions,
            &ledger,
            None,
            1,
        )
    };
    let with_ = call(&with_base);
    let without = call(&dose_free);
    assert_eq!(with_.len(), 1);

    // Closed form, splitting at the mid-window bolus t=12:
    //   ∫₀¹² D_ctrl·e^{−ke t} dt  +  ∫₁²²⁴ A(12⁺)·e^{−ke(t−12)} dt,
    // with A(12⁺) = D_ctrl·e^{−12ke} + D_mid. The instantaneous bolus into the
    // monitored compartment jumps the signal at an off-node instant, so the uniform
    // 128-panel trapezoid carries the documented ≈ ½·Δ·(span⁄panels) node-placement
    // bias (measured ≈ +0.94% here) — hence the looser band vs the smooth-decay cases
    // above. The point is inclusion: dropping the base dose is a ~70% error, far
    // outside this.
    let a12_pre = d_ctrl * (-ke * 12.0).exp();
    let a12_post = a12_pre + d_mid;
    let seg = 1.0 - (-ke * 12.0).exp();
    let want = (d_ctrl / ke) * seg + (a12_post / ke) * seg;
    assert_relative_eq!(with_[0], want, max_relative = 2e-2);

    // The mid-window base dose roughly triples this window's AUC; a `<= a` filter
    // would drop it and make with == without. Demand a large, unambiguous gap.
    assert!(
        with_[0] > without[0] * 2.0,
        "mid-window base dose not integrated: with-base {} vs ledger-only {}",
        with_[0],
        without[0]
    );
}

/// A base **infusion** or **steady-state** dose must reach the AUC pass with its full
/// attributes — the fix keeps base doses via `clone`, never a `DoseEvent::new` rebuild
/// (which would flatten `ss`/`ii`/`rate` to a plain `Fixed` bolus). This verifies that
/// "correct by construction" claim directly, rather than trusting it: both are
/// integrated by the same `ode_dense_solve_states` machinery `predict()` uses, so the
/// window AUC matches the closed form, and — critically — differs from the same total
/// dose given as a plain t=0 bolus. Closes the coverage gap PR #942's bolus-only
/// oracles left.
#[test]
fn adaptive_window_signal_aucs_preserves_ss_and_infusion_base_attrs() {
    let ode = one_cpt_ode_spec(); // readout = central amount, RHS = -ke·y
    let pk = pk_one(10.0, 100.0); // ke = 0.1
    let ke = 0.1_f64;
    let decisions = [24.0, 48.0]; // one window [24, 48], well after the base dose at 0
    let ledger = vec![ledger_bolus(24.0, 100.0, 1)];
    let call = |sub: &Subject| {
        adaptive_window_signal_aucs(
            &ode,
            &pk.values,
            &[],
            &[],
            sub,
            &decisions,
            &ledger,
            None,
            1,
        )
    };

    // --- Infusion base dose: amt=200 delivered over 20 h (rate=10) at t=0. Because it
    // is spread over [0,20] rather than dumped at t=0, more drug survives to t=24 than
    // an equal t=0 bolus would leave. A(24) decays from the end-of-infusion amount
    // (R/ke)(1 − e^{−20ke}); the window [24,48] is then smooth decay past the ctrl bolus.
    let inf = make_subject(
        vec![DoseEvent::new(0.0, 200.0, 1, 10.0, false, 0.0)],
        vec![],
    );
    let a20 = (10.0 / ke) * (1.0 - (-ke * 20.0).exp());
    let a24_inf = a20 * (-ke * 4.0).exp() + 100.0;
    let want_inf = (a24_inf / ke) * (1.0 - (-ke * 24.0).exp());
    assert_relative_eq!(call(&inf)[0], want_inf, max_relative = 1e-3);
    // Non-vacuous: the SAME 200 units as a plain t=0 bolus give a materially smaller
    // AUC (≈ −25%). Running the function on both proves it honors the infusion (does
    // not flatten it), not just that two closed forms differ.
    let bolus200 = make_subject(vec![DoseEvent::new(0.0, 200.0, 1, 0.0, false, 0.0)], vec![]);
    assert!(
        call(&inf)[0] > call(&bolus200)[0] * 1.2,
        "infusion base dose flattened to a bolus: inf {} vs bolus {}",
        call(&inf)[0],
        call(&bolus200)[0]
    );

    // --- Steady-state bolus base dose: amt=100, II=6, ss=true at t=0. The post-dose
    // amount is the SS peak D/(1 − e^{−ke·II}) (equilibrate_ss_state trough + the
    // record's bolus); a single SS record does not re-pulse, so it then decays plainly.
    let ss = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 6.0)], vec![]);
    let peak = 100.0 / (1.0 - (-ke * 6.0).exp());
    let a24_ss = peak * (-ke * 24.0).exp() + 100.0;
    let want_ss = (a24_ss / ke) * (1.0 - (-ke * 24.0).exp());
    assert_relative_eq!(call(&ss)[0], want_ss, max_relative = 1e-3);
    // Non-vacuous: dropping `ss` (a plain 100 bolus at t=0) omits the SS priming and
    // gives a smaller AUC (≈ −9%), so the SS attribute is genuinely acted upon.
    let bolus100 = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
    assert!(
        call(&ss)[0] > call(&bolus100)[0] * 1.05,
        "SS base dose treated as a plain bolus: ss {} vs bolus {}",
        call(&ss)[0],
        call(&bolus100)[0]
    );
}

/// Build the per-segment `obs_time -> indices` map the integrator uses.
fn obs_index_map(obs_times: &[f64]) -> HashMap<u64, Vec<usize>> {
    let mut m: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in obs_times.iter().enumerate() {
        m.entry(t.to_bits()).or_default().push(i);
    }
    m
}

#[test]
fn integrate_segment_zero_length_is_a_noop() {
    // A degenerate `[t, t]` segment must skip integration and leave the carried
    // state and predictions untouched — the guard a reactive driver relies on
    // when a decision time coincides with another break (#391 S1.2).
    // `ode_predictions` never reaches it (break_times are deduped at the same
    // 1e-15), so it has to be exercised directly here.
    let ode = one_cpt_ode_spec();
    let subject = make_subject(vec![], vec![5.0]);
    let pk = pk_one(1.0, 10.0);
    let mut ext_params = [0.0f64; crate::types::MAX_PK_PARAMS + 2];
    let mut u = vec![10.0];
    let mut predictions = vec![f64::NAN; subject.obs_times.len()];
    let obs_map = obs_index_map(&subject.obs_times);

    integrate_segment(
        &ode,
        &mut u,
        5.0,
        5.0,
        &subject,
        &[],
        &[],
        f64::NEG_INFINITY,
        &mut ext_params,
        &pk.values,
        &[],
        &[],
        &obs_map,
        &mut predictions,
        None,
        &[],
    );

    assert_eq!(u, vec![10.0], "zero-length segment must not change state");
    assert!(
        predictions[0].is_nan(),
        "zero-length segment must record no observation"
    );
}

#[test]
fn integrate_segment_advances_state_and_records_obs() {
    // A normal segment integrates 1-cpt decay (ke = CL/V = 0.1) over [0, 10]
    // and writes the observation at t_end, advancing `u` in place.
    let ode = one_cpt_ode_spec();
    let subject = make_subject(vec![], vec![10.0]);
    let pk = pk_one(1.0, 10.0);
    let mut ext_params = [0.0f64; crate::types::MAX_PK_PARAMS + 2];
    ext_params[crate::types::PK_IDX_CL] = 1.0;
    ext_params[crate::types::PK_IDX_V] = 10.0;
    let mut u = vec![10.0];
    let mut predictions = vec![f64::NAN; subject.obs_times.len()];
    let obs_map = obs_index_map(&subject.obs_times);

    integrate_segment(
        &ode,
        &mut u,
        0.0,
        10.0,
        &subject,
        &[],
        &[],
        f64::NEG_INFINITY,
        &mut ext_params,
        &pk.values,
        &[],
        &[],
        &obs_map,
        &mut predictions,
        None,
        &[],
    );

    let expected = 10.0 * (-1.0f64).exp(); // 10·e^{-ke·10}, ke = 0.1
    assert_relative_eq!(u[0], expected, max_relative = 1e-4);
    assert_relative_eq!(predictions[0], expected, max_relative = 1e-4);
}

// ----- S1.3a reactive driver (#391) ---------------------------------

#[test]
fn adaptive_state_independent_controller_matches_static_ode() {
    // Certainty anchor (degenerate oracle): a controller that ignores state
    // and gives a fixed 100 mg bolus at every decision must reproduce
    // `ode_predictions` on the same realized doses — pinning the reactive
    // bookkeeping to the trusted static engine.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = CL/V = 0.1
    let decisions = [0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 54.0];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_subject = make_subject(static_doses, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    assert_eq!(run.ledger.len(), 3);
    for (i, &t) in decisions.iter().enumerate() {
        assert_eq!(run.ledger[i].time, t);
        assert_eq!(run.ledger[i].amt, 100.0);
        assert_eq!(run.ledger[i].cmt, 1);
        assert_eq!(run.ledger[i].decision_idx, i);
        assert_eq!(run.ledger[i].dose_idx, i);
    }
}

#[test]
fn adaptive_form_c_readout_time_matches_the_static_engine() {
    // Degenerate oracle for the readout clock (#1028). The adaptive driver used to
    // feed the readout `t_start` — the integrator break the observation was keyed to,
    // off `shadow.obs_times` — while the static predictor reads the raw data clock.
    // For a subject with stacked reset occasions the two are *different numbers*, so
    // a `TIME`-reading Form C readout returned one answer under `simulate_adaptive`
    // and another under `predict()`/`fit()` for the same record.
    //
    // The readout here returns nothing but the model-time thread-local, so each
    // prediction *is* the `TIME` the readout saw. A fixed-dose controller keeps the
    // dynamics identical to the static twin, isolating the clock.
    let ode = time_readout_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 24.0];
    let obs = vec![6.0, 30.0, 54.0, 78.0];
    // Two occasions whose data TIME restarts: the internal grid stays monotonic
    // while the user clock repeats 6/30.
    let raw = vec![6.0, 30.0, 6.0, 30.0];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let mut base = make_subject(vec![], obs.clone());
    base.obs_raw_times = raw.clone();
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    assert_eq!(
        run.predictions, raw,
        "the adaptive readout must see the raw data-file TIME, got {:?}",
        run.predictions
    );

    // …and agree with the static engine on the realized doses, record for record.
    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let mut static_subject = make_subject(static_doses, obs);
    static_subject.obs_raw_times = raw;
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
    assert_eq!(
        run.predictions, static_preds,
        "adaptive and static readouts must be on the same clock"
    );
}

#[test]
fn adaptive_tv_frozen_replay_readout_time_matches_the_driver() {
    // The same clock split in `adaptive_frozen_replay_tv` (#1028) — the replay
    // verifier, whose entire job is to prove bit-equality with the static engine, so
    // it is the one path that must not be on the other convention. Same TIME-returning
    // readout; the replay must reproduce the driver's `TIME` values exactly.
    let ode = time_readout_ode_spec();
    let v = 10.0;
    let obs_times = [0.0, 24.0, 48.0, 72.0];
    let raw = vec![0.0, 24.0, 0.0, 24.0];
    let cls = [1.0, 0.7, 0.5, 0.3];
    let obs_pk: Vec<PkParams> = cls.iter().map(|&cl| pk_one(cl, v)).collect();
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: obs_pk.clone(),
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    let decisions = [0.0, 24.0, 48.0];
    let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        rule: None,
    };
    let mut base = make_subject(vec![], obs_times.to_vec());
    base.obs_raw_times = raw.clone();
    let run = ode_predictions_adaptive_impl(
        &ode,
        &obs_pk[0].values,
        Some(&event_pk),
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.predictions, raw, "driver reads the raw data clock");

    let mut static_subject = base.clone();
    static_subject.doses = run
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let dose_f: Vec<f64> = run.ledger.iter().map(|e| e.f_applied).collect();
    let replay = adaptive_frozen_replay_tv(
        &ode,
        &event_pk,
        None,
        None,
        &[],
        &[],
        &static_subject,
        &dose_f,
        &decisions,
    );
    assert_eq!(
        replay, run.predictions,
        "the frozen replay must be on the driver's clock, bit for bit"
    );
}

#[test]
fn adaptive_tv_covariate_controller_matches_static_event_driven() {
    // Degenerate oracle on the time-varying-covariate path (#700): a controller
    // that ignores state and gives a fixed bolus at every decision must
    // reproduce the trusted static event-driven engine on the same realized
    // doses, when the per-event PK drifts across the horizon. CL declines here
    // (e.g. renal-function decline), so a frozen t=0 snapshot gives a visibly
    // wrong answer — the whole point of #700.
    let ode = one_cpt_ode_spec();
    let v = 10.0;
    // Decisions coincide with observation rows; CL declines across the horizon.
    let times = [0.0, 24.0, 48.0, 72.0];
    let cls = [1.0, 0.7, 0.5, 0.3];
    let obs_pk: Vec<PkParams> = cls.iter().map(|&cl| pk_one(cl, v)).collect();
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: obs_pk.clone(),
        pk_only: Vec::new(),
        reset: Vec::new(),
    };

    let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        rule: None,
    };
    let base = make_subject(vec![], times.to_vec());
    let run = ode_predictions_adaptive_impl(
        &ode,
        &obs_pk[0].values, // t=0 baseline — unused on the TV path
        Some(&event_pk),
        None,
        None,
        &[],
        &[],
        &base,
        &times,
        &[],
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");

    // Trusted static engine: same realized doses, same drifting per-event PK.
    // Each dose coincides with an obs row, so its LOCF snapshot is that row's PK.
    let static_doses: Vec<DoseEvent> = times
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_subject = make_subject(static_doses, times.to_vec());
    let static_preds = ode_predictions_event_driven(
        &ode,
        &static_subject,
        &[],
        &[],
        &obs_pk, // pk_at_dose: dose k coincides with obs k
        &obs_pk, // pk_at_obs
        &[],     // pk_at_pk_only
        &[],
    );

    assert_eq!(run.predictions.len(), static_preds.len());
    // Cross-integrator (dense driver vs `solve_ode` static) → solver-tolerance.
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    assert_eq!(run.ledger.len(), 4);

    // Guard against a silent regression to the frozen-t0 path: the same run with
    // PK frozen at CL=1.0 for the whole horizon must give a materially different
    // (lower — faster clearance) final prediction.
    let mut frozen = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        rule: None,
    };
    let frozen_run = ode_predictions_adaptive_impl(
        &ode,
        &obs_pk[0].values,
        None,
        None,
        None,
        &[],
        &[],
        &base,
        &times,
        &[],
        &mut frozen,
        100,
        None,
    )
    .expect("frozen driver runs");
    let tv_last = *run.predictions.last().unwrap();
    let frozen_last = *frozen_run.predictions.last().unwrap();
    assert!(
        (tv_last - frozen_last).abs() > 1e-6 * tv_last.abs().max(1.0),
        "TV path must differ from the frozen-t0 path: tv={tv_last}, frozen={frozen_last}"
    );
}

#[test]
fn adaptive_tv_frozen_replay_is_bit_exact() {
    // The TV frozen-replay engine must be BIT-aligned with the reactive driver
    // (not merely tolerance-close) — the same invariant the constant path pins.
    // Both share `segment_pk_at` + `integrate_segment` over identical segments.
    // A hold-mix controller and a decision that falls *between* obs rows (t=12,
    // exercising the LOCF carry and a decision-only break) make it a real test.
    let ode = one_cpt_ode_spec();
    let v = 10.0;
    let obs_times = [0.0, 24.0, 48.0, 72.0];
    let cls = [1.0, 0.7, 0.5, 0.3];
    let obs_pk: Vec<PkParams> = cls.iter().map(|&cl| pk_one(cl, v)).collect();
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: obs_pk.clone(),
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    let decisions = [0.0, 12.0, 24.0, 48.0, 72.0];
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let mut decide = |ctx: &ControllerCtx| ControllerDecision {
        actions: if ctx.signal("A").expect("monitor A declared") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        },
        rule: None,
    };
    let mons: Vec<AdaptiveMonitor> = monitors
        .iter()
        .map(|s| AdaptiveMonitor {
            spec: s,
            observe: None,
        })
        .collect();
    let base = make_subject(vec![], obs_times.to_vec());
    let run = ode_predictions_adaptive_impl(
        &ode,
        &obs_pk[0].values,
        Some(&event_pk),
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &mons,
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");

    // The default-tolerance verifier accepts it.
    verify_adaptive_frozen_replay(
        &ode,
        &obs_pk[0].values,
        Some(&event_pk),
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &run,
    )
    .expect("TV run matches the static replay");

    // And the agreement is bit-exact, not merely within the verifier's slack.
    let mut static_subject = base.clone();
    static_subject.doses = run
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let dose_f: Vec<f64> = run.ledger.iter().map(|e| e.f_applied).collect();
    let replay = adaptive_frozen_replay_tv(
        &ode,
        &event_pk,
        None,
        None,
        &[],
        &[],
        &static_subject,
        &dose_f,
        &decisions,
    );
    let mut max_rel = 0.0_f64;
    for (got, want) in run.predictions.iter().zip(replay.iter()) {
        if got.is_nan() && want.is_nan() {
            continue;
        }
        let d = (got - want).abs();
        let r = if want.abs() > 0.0 { d / want.abs() } else { d };
        max_rel = max_rel.max(r);
    }
    assert!(
        max_rel <= 1e-12,
        "TV frozen replay not bit-aligned: max_rel={max_rel}"
    );
    // The controller must actually hold at least once, so a decision-only break
    // (LOCF carry) is genuinely exercised.
    assert!(
        run.decisions
            .iter()
            .any(|d| matches!(d.outcome, DecisionOutcome::Hold)),
        "expected at least one hold to exercise a decision-only break"
    );
}

#[test]
fn earliest_record_pk_seeds_earliest_else_fallback() {
    // #700 seed helper: with records present, seed from the earliest record's
    // per-event PK (obs vs pk-only, whichever is earlier). With NO record, fall
    // back to the supplied baseline — never a zero-PK default (which would
    // integrate CL=V=0 → NaN for a record-free TIME-in-PK subject).
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: vec![pk_one(1.0, 10.0), pk_one(2.0, 10.0)],
        pk_only: vec![pk_one(9.0, 10.0)],
        reset: Vec::new(),
    };
    // obs at 24 & 48, pk-only at 6 (earliest) → seed = pk_only[0] (CL=9).
    let mut s = make_subject(vec![], vec![24.0, 48.0]);
    s.pk_only_times = vec![6.0];
    let seeded = earliest_record_pk(&s, &event_pk, pk_one(5.0, 5.0));
    assert_eq!(
        seeded.cl(),
        9.0,
        "earliest record is the pk-only row at t=6"
    );

    // No records at all → the baseline fallback, not PkParams::default().
    let empty = crate::pk::EventPkParams::default();
    let recordless = make_subject(vec![], vec![]);
    let fallback = pk_one(5.0, 5.0);
    let seeded = earliest_record_pk(&recordless, &empty, fallback);
    assert_eq!(
        seeded.cl(),
        5.0,
        "record-free subject seeds from the baseline"
    );
    assert_ne!(
        seeded.cl(),
        PkParams::default().cl(),
        "must NOT be the zero-PK default"
    );
}

#[test]
fn locf_decision_cov_carries_forward_prefers_obs_and_falls_back() {
    // #700: at a decision that lands between records, the covariate must be the
    // most-recent record's (LOCF), not the frozen t=0 baseline; an obs record
    // wins a tie with a pk-only record; before the first record, the baseline.
    let baseline = HashMap::from([("WT".to_string(), 70.0)]);
    let mut s = make_subject(vec![], vec![0.0, 24.0, 48.0]);
    s.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 50.0)]),
        HashMap::from([("WT".to_string(), 30.0)]),
    ];
    // A pk-only row at t=24 with a *different* WT proves obs wins the tie.
    s.pk_only_times = vec![24.0];
    s.pk_only_covariates = vec![HashMap::from([("WT".to_string(), 999.0)])];

    // Decision at t=36 → LOCF is the obs at t=24 (WT=50), not baseline 70.
    assert_eq!(locf_decision_cov(36.0, &s, &baseline)["WT"], 50.0);
    // At t=24 exactly, the obs record wins over the coincident pk-only row.
    assert_eq!(locf_decision_cov(24.0, &s, &baseline)["WT"], 50.0);
    // Before the first record (t=-1) → baseline.
    assert_eq!(locf_decision_cov(-1.0, &s, &baseline)["WT"], 70.0);
}

#[test]
fn adaptive_tv_decision_between_records_reads_locf_covariate() {
    // #700 regression: a decision that does NOT coincide with an observation
    // must read the LOCF covariate (the most-recent record), not the frozen t=0
    // baseline. Before the fix, `decision_cov` fell back to `shadow.covariates`
    // (t=0) while `pk_readout` used the LOCF PK — so an `observe` expression that
    // references a time-varying covariate directly saw a stale value and the
    // controller titrated off the wrong signal. The frozen-replay verifier is
    // blind to this (it never re-runs decisions), so it needs its own test.
    let ode = one_cpt_ode_spec();
    let mut base = make_subject(vec![], vec![0.0, 24.0, 48.0]);
    base.covariates = HashMap::from([("WT".to_string(), 70.0)]);
    base.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 50.0)]),
        HashMap::from([("WT".to_string(), 30.0)]),
    ];
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: vec![pk_one(1.0, 10.0); 3],
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    // `observe` reads the covariate WT directly (the case that regresses).
    let obs_fn: OdeOutputFn = Box::new(
        |_u: &[f64], _pk: &[f64], _th: &[f64], _eta: &[f64], cov: &HashMap<String, f64>| {
            *cov.get("WT").unwrap_or(&f64::NAN)
        },
    );
    let spec = MonitorSpec::new("A", 1, ObserveMode::Ipred);
    let mons = vec![AdaptiveMonitor {
        spec: &spec,
        observe: Some(&obs_fn),
    }];
    // Hold every decision — we only care about the monitored signal it recorded.
    let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Hold],
        rule: None,
    };
    // Decisions at 12 and 36 fall *between* obs rows.
    let decisions = [0.0, 12.0, 36.0];
    let run = ode_predictions_adaptive_impl(
        &ode,
        &event_pk.obs[0].values,
        Some(&event_pk),
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &mons,
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");

    let sig_at = |t: f64| {
        run.decisions
            .iter()
            .find(|d| d.time == t)
            .and_then(|d| d.observed_signals.first())
            .map(|s| s.value)
            .unwrap_or(f64::NAN)
    };
    // t=36 is after the t=24 obs (WT=50) → LOCF WT=50, NOT the t=0 baseline 70.
    assert_eq!(sig_at(36.0), 50.0, "decision at t=36 must read LOCF WT=50");
    // t=12 is after only the t=0 obs → WT=70 (which equals the baseline here).
    assert_eq!(
        sig_at(12.0),
        70.0,
        "decision at t=12 reads the t=0 record WT=70"
    );
}

#[test]
fn adaptive_tv_break_collision_within_tolerance_is_rejected() {
    // #700 guard: the 1e-15 `dedup_by` on `break_times` can merge a decision and
    // an observation that are within tolerance but not bit-identical, after which
    // the exact-`to_bits()` lookups in `segment_pk_at` / `decision_index_of` miss
    // the dropped one — silently losing a per-event PK snapshot or a dose. Rather
    // than a silent wrong answer, the driver rejects it loudly.
    let ode = one_cpt_ode_spec();
    // 0.1 + 0.2 == 0.30000000000000004 ≠ 0.3 (0.29999999999999998), |Δ| ≈ 5.5e-17.
    let drifted = 0.1_f64 + 0.2_f64;
    let base = make_subject(vec![], vec![0.0, 0.3]);
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: vec![pk_one(1.0, 10.0); 2],
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        rule: None,
    };
    let err = ode_predictions_adaptive_impl(
        &ode,
        &event_pk.obs[0].values,
        Some(&event_pk),
        None,
        None,
        &[],
        &[],
        &base,
        &[0.0, drifted],
        &[],
        &mut decide,
        100,
        None,
    )
    .expect_err("a sub-tolerance decision/obs collision must be rejected");
    assert!(
        err.contains("bit-identical"),
        "error should cite the collision: {err}"
    );
}

#[test]
fn adaptive_tv_pk_only_records_drive_per_event_pk() {
    // #700: an EVID=2 (pk-only) record carries a per-event PK snapshot that
    // `has_tv_covariates()` never inspects. The driver must resolve the segment
    // ending at the pk-only row from `event_pk.pk_only`, and the frozen-replay
    // engine must agree bit-for-bit. Every prior test used `pk_only: Vec::new()`,
    // so this pins the pk-only branch of `segment_pk_at` + the pk-only seed/break
    // loops in both engines.
    let ode = one_cpt_ode_spec();
    let v = 10.0;
    // obs at 0 & 48; a pk-only record at 24 governs the (0, 24] decay.
    let mut base = make_subject(vec![], vec![0.0, 48.0]);
    base.pk_only_times = vec![24.0];
    base.pk_only_covariates = vec![HashMap::from([("CRCL".to_string(), 40.0)])];
    base.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 100.0)]),
    ];

    let run_for = |pk_only_cl: f64| {
        let event_pk = crate::pk::EventPkParams {
            dose: Vec::new(),
            obs: vec![pk_one(1.0, v), pk_one(1.0, v)],
            pk_only: vec![pk_one(pk_only_cl, v)],
            reset: Vec::new(),
        };
        let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
            actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
            rule: None,
        };
        let run = ode_predictions_adaptive_impl(
            &ode,
            &event_pk.obs[0].values,
            Some(&event_pk),
            None,
            None,
            &[],
            &[],
            &base,
            &[0.0, 24.0, 48.0],
            &[],
            &mut decide,
            100,
            None,
        )
        .expect("driver runs");
        // The frozen-replay verifier exercises the pk-only branch of the static
        // replay engine and pins bit-alignment with the driver.
        verify_adaptive_frozen_replay(
            &ode,
            &event_pk.obs[0].values,
            Some(&event_pk),
            None,
            None,
            &[],
            &[],
            &base,
            &[0.0, 24.0, 48.0],
            &run,
        )
        .expect("pk-only TV run matches the static replay");
        *run.predictions.last().unwrap()
    };

    // A faster pk-only CL clears more drug over (0, 24], so the t=48 prediction
    // is materially lower — proving the pk-only snapshot is actually consumed.
    let slow = run_for(0.2);
    let fast = run_for(3.0);
    assert!(
        fast < slow - 1e-6,
        "pk-only per-event CL must drive the trajectory: slow={slow}, fast={fast}"
    );
}

#[test]
fn adaptive_iov_controller_matches_static_event_driven() {
    // Degenerate oracle for IOV (#701): a fixed-regimen controller under a
    // per-decision-window κ (occasion = decision index) must reproduce the trusted
    // static event-driven engine on the same realized doses with the same
    // per-occasion PK. Decisions coincide with observations here (every dose lands
    // on an obs row), so each event's occasion snapshot is unambiguous and the two
    // engines share the end-of-interval convention exactly.
    let ode = one_cpt_ode_spec();
    let v = 10.0;
    // Four decisions = four occasions; CL shifts each occasion (the κ effect).
    let times = [0.0, 24.0, 48.0, 72.0];
    let occ_cls = [1.0, 0.7, 0.5, 0.3];
    let occ_pk: Vec<PkParams> = occ_cls.iter().map(|&cl| pk_one(cl, v)).collect();
    // event_pk.obs[j]: obs j coincides with decision j → occasion j's PK.
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: occ_pk.clone(),
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    // decision_pk[g] = occasion g's PK; eta_occ activates the IOV path (the plain
    // ODE reads PK from the params array, so the eta *values* are immaterial here —
    // the eta *threading* is exercised end-to-end by the api-level IOV test).
    let decision_pk = occ_pk.clone();
    let eta_occ: Vec<Vec<f64>> = vec![Vec::new(); times.len()];

    let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        rule: None,
    };
    let base = make_subject(vec![], times.to_vec());
    let run = ode_predictions_adaptive_impl(
        &ode,
        &occ_pk[0].values,
        Some(&event_pk),
        Some(&decision_pk),
        Some(&eta_occ),
        &[],
        &[],
        &base,
        &times,
        &[],
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");

    // Trusted static engine: same realized doses, per-dose / per-obs occasion PK.
    let static_doses: Vec<DoseEvent> = times
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_subject = make_subject(static_doses, times.to_vec());
    let static_preds = ode_predictions_event_driven(
        &ode,
        &static_subject,
        &[],
        &[],
        &occ_pk, // pk_at_dose: dose g coincides with occasion g
        &occ_pk, // pk_at_obs
        &[],
        &[],
    );

    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    assert_eq!(run.ledger.len(), 4);

    // Guard against a silent regression to a single-occasion (κ-frozen) path: the
    // same run with every occasion pinned to CL=1.0 clears faster in the later
    // occasions, so its final concentration is materially lower.
    let frozen_pk: Vec<PkParams> = vec![pk_one(1.0, v); times.len()];
    let frozen_event = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: frozen_pk.clone(),
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    let frozen_eta: Vec<Vec<f64>> = vec![Vec::new(); times.len()];
    let mut decide2 = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        rule: None,
    };
    let frozen_run = ode_predictions_adaptive_impl(
        &ode,
        &frozen_pk[0].values,
        Some(&frozen_event),
        Some(&frozen_pk),
        Some(&frozen_eta),
        &[],
        &[],
        &base,
        &times,
        &[],
        &mut decide2,
        100,
        None,
    )
    .expect("frozen driver runs");
    let iov_last = *run.predictions.last().unwrap();
    let frozen_last = *frozen_run.predictions.last().unwrap();
    assert!(
            iov_last > frozen_last * (1.0 + 1e-6),
            "per-occasion κ must differ from a single-occasion path: iov={iov_last}, frozen={frozen_last}"
        );
}

#[test]
fn adaptive_iov_frozen_replay_is_bit_exact() {
    // The occasion-aware frozen replay (#701) must be BIT-aligned with the reactive
    // driver, exactly as the constant and #700 TV paths are. A decision that falls
    // *between* observations (t=12) exercises `decision_pk` + the occasion LOCF
    // carry — the case where a naive LOCF would read the previous occasion's PK.
    let ode = one_cpt_ode_spec();
    let v = 10.0;
    let obs_times = [6.0, 30.0, 54.0, 78.0];
    // Five decisions = five occasions; t=12 falls between obs rows. CL per occasion.
    let decisions = [0.0, 12.0, 24.0, 48.0, 72.0];
    let occ_cls = [1.2, 1.0, 0.8, 0.6, 0.4];
    let occ_pk: Vec<PkParams> = occ_cls.iter().map(|&cl| pk_one(cl, v)).collect();
    // obs occasion = decision window containing the obs time:
    //   6→occ0([0,12)), 30→occ2([24,48)), 54→occ3([48,72)), 78→occ4([72,∞)).
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: vec![occ_pk[0], occ_pk[2], occ_pk[3], occ_pk[4]],
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    let decision_pk = occ_pk.clone();
    let eta_occ: Vec<Vec<f64>> = vec![Vec::new(); decisions.len()];
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let mut decide = |ctx: &ControllerCtx| ControllerDecision {
        actions: if ctx.signal("A").expect("monitor A declared") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        },
        rule: None,
    };
    let mons: Vec<AdaptiveMonitor> = monitors
        .iter()
        .map(|s| AdaptiveMonitor {
            spec: s,
            observe: None,
        })
        .collect();
    let base = make_subject(vec![], obs_times.to_vec());
    let run = ode_predictions_adaptive_impl(
        &ode,
        &occ_pk[0].values,
        Some(&event_pk),
        Some(&decision_pk),
        Some(&eta_occ),
        &[],
        &[],
        &base,
        &decisions,
        &mons,
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");
    assert!(!run.ledger.is_empty(), "the run must dose at least once");

    // The default-tolerance verifier accepts it...
    verify_adaptive_frozen_replay(
        &ode,
        &occ_pk[0].values,
        Some(&event_pk),
        Some(&decision_pk),
        Some(&eta_occ),
        &[],
        &[],
        &base,
        &decisions,
        &run,
    )
    .expect("IOV run matches the occasion-aware static replay");

    // ...and the agreement is bit-exact, not merely within the verifier's slack.
    let mut static_subject = base.clone();
    static_subject.doses = run
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let dose_f: Vec<f64> = run.ledger.iter().map(|e| e.f_applied).collect();
    let replay = adaptive_frozen_replay_tv(
        &ode,
        &event_pk,
        Some(&decision_pk),
        Some(&eta_occ),
        &[],
        &[],
        &static_subject,
        &dose_f,
        &decisions,
    );
    let mut max_rel = 0.0_f64;
    for (got, want) in run.predictions.iter().zip(replay.iter()) {
        if got.is_nan() && want.is_nan() {
            continue;
        }
        let d = (got - want).abs();
        let r = if want.abs() > 0.0 { d / want.abs() } else { d };
        max_rel = max_rel.max(r);
    }
    assert!(
        max_rel <= 1e-12,
        "IOV frozen replay not bit-aligned: max_rel={max_rel}"
    );
}

#[test]
fn adaptive_iov_threads_occasion_eta_into_readout() {
    // #701 review (coverage): in every other IOV test the per-occasion κ reaches
    // the trajectory only through the precomputed PK array (event_pk / decision_pk),
    // so the *separate* `eta_for` threading of the occasion eta into
    // `read_observable` / `integrate_segment` — load-bearing only when an expression
    // references κ or η *directly* — was never exercised end to end. Here the
    // monitored `observe` returns the κ component (eta[1]) it is handed, so the
    // recorded decision signal must equal that occasion's κ. If `eta_for` /
    // `segment_occ_at` threaded the wrong occasion (e.g. the fixed baseline eta),
    // the signals would be the baseline value, not the per-occasion κ.
    let ode = one_cpt_ode_spec();
    let v = 10.0;
    let times = [0.0, 24.0, 48.0]; // 3 decisions = 3 occasions; obs coincide.
    let occ_pk: Vec<PkParams> = vec![pk_one(1.0, v); times.len()];
    let event_pk = crate::pk::EventPkParams {
        dose: Vec::new(),
        obs: occ_pk.clone(),
        pk_only: Vec::new(),
        reset: Vec::new(),
    };
    let decision_pk = occ_pk.clone();
    // Distinct κ per occasion in eta[1]; eta[0] is a dummy η_bsv.
    let eta_occ: Vec<Vec<f64>> = vec![vec![0.0, 10.0], vec![0.0, 20.0], vec![0.0, 30.0]];
    // A non-empty *baseline* eta with a sentinel κ=999: a wrong occasion (falling
    // back to baseline) would surface 999, not the per-occasion κ — a clean miss
    // rather than an index panic.
    let baseline_eta = [0.0, 999.0];

    // `observe` returns the κ component (eta[1]) it is handed — the occasion's κ.
    let obs_fn: OdeOutputFn = Box::new(
        |_u: &[f64], _pk: &[f64], _th: &[f64], eta: &[f64], _cov: &HashMap<String, f64>| eta[1],
    );
    let spec = MonitorSpec::new("K", 1, ObserveMode::Ipred);
    let mons = vec![AdaptiveMonitor {
        spec: &spec,
        observe: Some(&obs_fn),
    }];
    let mut decide = |_ctx: &ControllerCtx| ControllerDecision {
        actions: vec![DoseAction::Hold],
        rule: None,
    };
    let base = make_subject(vec![], times.to_vec());
    let run = ode_predictions_adaptive_impl(
        &ode,
        &occ_pk[0].values,
        Some(&event_pk),
        Some(&decision_pk),
        Some(&eta_occ),
        &[],
        &baseline_eta,
        &base,
        &times,
        &mons,
        &mut decide,
        100,
        None,
    )
    .expect("driver runs");

    let sig_at = |t: f64| {
        run.decisions
            .iter()
            .find(|d| d.time == t)
            .and_then(|d| d.observed_signals.first())
            .map(|s| s.value)
            .unwrap_or(f64::NAN)
    };
    // Each decision's readout must carry its own occasion's κ (eta[1]), threaded by
    // `eta_for(segment_occ_at(...))` — not the baseline sentinel 999.
    assert_eq!(sig_at(0.0), 10.0, "occasion 0 κ");
    assert_eq!(sig_at(24.0), 20.0, "occasion 1 κ");
    assert_eq!(sig_at(48.0), 30.0, "occasion 2 κ");
}

#[test]
fn frozen_replay_verifier_accepts_aligned_run_and_rejects_corruption() {
    // The verifier's Err branches aren't reachable from a faithful run (the
    // bookkeeping is correct), so exercise them directly: a faithful run
    // passes, a perturbed trajectory is a typed divergence error, and a
    // wrong-length prediction vector is a typed error rather than a panic.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 54.0];
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(vec![], obs);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    // A dose at every decision aligns the segment structure → exact match.
    verify_adaptive_frozen_replay(
        &ode,
        &pk.values,
        None,
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &run,
    )
    .expect("aligned run matches the static replay");

    let mut perturbed = run.clone();
    perturbed.predictions[0] += 10.0;
    let err = verify_adaptive_frozen_replay(
        &ode,
        &pk.values,
        None,
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &perturbed,
    )
    .expect_err("a perturbed trajectory must fail verification");
    assert!(err.contains("diverges"), "got: {err}");

    let mut short = run.clone();
    short.predictions.pop();
    let err = verify_adaptive_frozen_replay(
        &ode,
        &pk.values,
        None,
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &short,
    )
    .expect_err("a length mismatch must fail verification");
    assert!(err.contains("prediction"), "got: {err}");
}

#[test]
fn frozen_replay_aligns_break_structure_on_held_decisions() {
    // Regression for the held-decision tolerance fix: a run that holds at
    // some decisions used to only agree with the static replay within a wide
    // (×100·reltol) slack, because the driver breaks at every decision while a
    // naive static replay breaks only at realized doses. Feeding the decision
    // schedule back in as no-op breaks aligns the two engines' segments, so
    // the run now passes the *tight* default verifier. Dose only while the
    // central amount is below 50: at t=0 the trough is 0 → dose; the later
    // decisions see a decayed-but-still-high amount → hold.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 2.0, 4.0];
    let obs = vec![1.0, 3.0, 5.0];
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("monitor A declared") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    };
    let base = make_subject(vec![], obs);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &monitors,
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    // Exactly one realized dose (t=0); the t=2 / t=4 decisions held.
    assert_eq!(run.ledger.len(), 1, "only the t=0 decision should dose");

    // Passes the tight (aligned) verifier — the whole point of the fix.
    verify_adaptive_frozen_replay(
        &ode,
        &pk.values,
        None,
        None,
        None,
        &[],
        &[],
        &base,
        &decisions,
        &run,
    )
    .expect("held-decision run matches the aligned static replay");
}

#[test]
fn frozen_replay_residual_is_pinned_below_the_verifier_bound() {
    // Characterization of the residual that justifies the verifier's tolerance
    // factor (`REPLAY_TOL_FACTOR = 8`). On a held-decision run we measure the
    // max relative |reactive − static| both ways:
    //   * ALIGNED   (decision times fed back as no-op breaks): measured 0.0 —
    //     the reactive driver and the static engine, walking identical segments
    //     through the same `integrate_segment`, agree BIT-FOR-BIT.
    //   * UNALIGNED (naive replay, breaks only at realized doses): measured
    //     ~7.3e-8 here — a real held-decision perturbation that the alignment
    //     removes entirely.
    // Both sit far under the live verifier bound (×8·reltol = 8e-4), so it
    // never false-positives; the ×8 is the conservative margin that holds even
    // on stiffer models where the (pre-alignment) perturbation would be larger.
    // If the alignment ever regresses, `rel_aligned` jumps toward the unaligned
    // level and the bit-exact bound below fails loudly.
    let ode = one_cpt_ode_spec(); // reltol 1e-4 / abstol 1e-6 (defaults)
    let pk = pk_one(1.0, 10.0);
    // CL=1, V=10 → k=0.1/h. A 100-unit bolus only while the central amount is
    // below 50; it decays 100·e^{-0.1t}, crossing 50 near t≈6.9, so over this
    // schedule the t=0 and t=8 troughs dose and t∈{2,4,6} hold — a dose/hold mix.
    let decisions = [0.0, 2.0, 4.0, 6.0, 8.0];
    let obs = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("monitor A declared") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    };
    let base = make_subject(vec![], obs);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &monitors,
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert!(
        run.ledger.len() >= 2,
        "expected a dose/hold mix (≥2 realized doses), got {}",
        run.ledger.len()
    );

    // Rebuild the static subject from the realized ledger, exactly as the
    // verifier does.
    let mut static_subject = base.clone();
    static_subject.doses = run
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();

    let max_rel = |preds: &[f64]| -> f64 {
        run.predictions
            .iter()
            .zip(preds)
            .filter(|(g, w)| g.is_finite() && w.is_finite() && w.abs() > 0.0)
            .map(|(g, w)| (g - w).abs() / w.abs())
            .fold(0.0_f64, f64::max)
    };

    let aligned =
        ode_predictions_with_extra_breaks(&ode, &pk.values, &[], &[], &static_subject, &decisions);
    let unaligned = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

    let rel_aligned = max_rel(&aligned);
    let rel_unaligned = max_rel(&unaligned);

    // Aligned replay is bit-exact (measured 0.0). Allow a few ULP of headroom
    // for future legitimate reordering, but stay ~9 orders under the live ×8
    // bound: a held-break-mismatch regression pushes this toward the unaligned
    // level (~7e-8) and trips here long before the ×8 verifier would.
    assert!(
        rel_aligned <= 1e-12,
        "aligned replay should match the reactive driver bit-for-bit, got {rel_aligned:e} \
             (verifier bound is 8·reltol = 8e-4); the decision-time break alignment may have \
             regressed"
    );
    // And the alignment is genuinely doing the work: the naive (unaligned)
    // replay carries a real, measurable residual that the alignment eliminates.
    assert!(
        rel_unaligned > 1e-9,
        "expected a measurable unaligned residual (the perturbation alignment removes); \
             got {rel_unaligned:e} — if this is ~0 the scenario no longer holds any decisions, \
             so the characterization is vacuous"
    );
}

#[test]
fn adaptive_feedback_doses_only_below_threshold() {
    // State-dependent: dose 100 only when the monitored amount is below 50.
    // At t=0 amount is 0 (<50) -> dose; by t=2 it decayed to 100·e^{-0.2}
    // ≈ 81.9 (>50) -> hold. Exactly one realized dose.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let decisions = [0.0, 2.0];
    let obs = vec![1.0, 3.0];

    let mut controller = |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("monitor A is declared") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &monitors,
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    assert_eq!(run.ledger.len(), 1, "dose at t=0, hold at t=2");
    assert_eq!(run.ledger[0].time, 0.0);
    assert_eq!(run.ledger[0].decision_idx, 0);
    assert_eq!(run.ledger[0].observed_signals[0].name, "A");
    assert_eq!(run.ledger[0].observed_signals[0].value, 0.0);

    // Trajectory vs the exact 1-cpt closed form A(t) = 100·e^{-ke·t}, ke=0.1.
    // (An analytical oracle, not `ode_predictions`: with a hold at t=2 the
    // driver breaks there while the static engine wouldn't, so a static
    // comparison would confound the integrator restart with the dosing logic.)
    let ke = 0.1;
    for (i, t) in [1.0_f64, 3.0].into_iter().enumerate() {
        let exact = 100.0 * (-ke * t).exp();
        assert_relative_eq!(run.predictions[i], exact, max_relative = 1e-5);
    }
}

#[test]
fn adaptive_decision_monitor_uses_observation_covariates() {
    // Regression (#538): at a decision time the monitored Form-C readout
    // must see the covariate snapshot in effect at that time (the coincident
    // observation row), not the subject-level first-row covariate. The
    // readout is `state * FREE`; with no decay the state stays at the dose
    // amount, so the monitored signal is driven purely by FREE.
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::Single(Box::new(|state, _pk, _theta, _eta, covariates| {
            state[0] * covariates.get("FREE").copied().unwrap_or(0.0)
        })),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    };
    let pk = pk_one(0.0, 1.0); // ke = 0 -> state holds at the dose amount
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let decisions = [0.0, 10.0];

    // Single observation at the second decision time, carrying FREE=2.0;
    // the subject-static map carries the stale FREE=1.0.
    let mut base = make_subject(vec![], vec![10.0]);
    base.covariates.insert("FREE".into(), 1.0);
    base.obs_covariates = vec![HashMap::from([("FREE".to_string(), 2.0)])];

    // Dose 100 at t=0 (signal 0 < 150). At t=10 the pre-dose state is 100,
    // so the monitored signal is 100*FREE. With the observation snapshot
    // (FREE=2) the signal is 200 >= 150 -> hold; with the stale static
    // value (FREE=1) it would be 100 < 150 -> a second (wrong) dose.
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("monitor A is declared") < 150.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    };
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &monitors,
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    assert_eq!(
        run.ledger.len(),
        1,
        "dose only at t=0; the t=10 monitor must read FREE=2 (signal 200) and hold"
    );
    assert_eq!(run.ledger[0].time, 0.0);
    // The decision at t=10 logged the monitored signal computed with the
    // observation-row covariate: 100 * 2.0 = 200.
    let d10 = run
        .decisions
        .iter()
        .find(|d| d.time == 10.0)
        .expect("decision logged at t=10");
    assert_relative_eq!(d10.observed_signals[0].value, 200.0, epsilon = 1e-9);
}

#[test]
fn adaptive_stop_discontinues_further_dosing() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Stop];
    let base = make_subject(vec![], vec![12.0, 36.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0, 24.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert!(
        run.ledger.is_empty(),
        "Stop at decision 0 prevents all doses"
    );
    assert!(
        run.predictions.iter().all(|&p| p == 0.0),
        "no dose -> zero state"
    );
}

#[test]
fn adaptive_zero_amount_bolus_is_treated_as_hold() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 0.0, cmt: 1 }];
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert!(run.ledger.is_empty(), "zero-amount bolus records no dose");
    assert_eq!(run.predictions[0], 0.0);
}

#[test]
fn adaptive_infusion_state_independent_matches_static_ode() {
    // Degenerate oracle (infusion edition): a controller that ignores state
    // and issues the same fixed infusion at every decision must reproduce
    // `ode_predictions` on the equivalent static infusion schedule, bit-exact.
    // This pins the dynamic infusion-end timeline (every F-scaled end inserted
    // as a break) to the trusted static segmentation. The last observation is
    // the global maximum so neither engine breaks at an interior observation
    // (which would restart the integrator on only one side and diverge).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = 0.1
    let decisions = [0.0, 24.0, 48.0];
    // Each infusion: 100 mg at rate 25 -> 4 h. Ends 4, 28, 52 (between
    // decisions). Observations span during/post-infusion; the last (60) is
    // past every infusion end so it is the global maximum.
    let obs = vec![2.0, 6.0, 26.0, 30.0, 50.0, 60.0];

    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 25.0,
        }]
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 25.0, false, 0.0))
        .collect();
    let static_subject = make_subject(static_doses, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    assert_eq!(run.ledger.len(), 3);
    for (i, &t) in decisions.iter().enumerate() {
        assert_eq!(run.ledger[i].time, t);
        assert_eq!(run.ledger[i].rate, 25.0);
        assert_eq!(run.ledger[i].rule_fired, "infuse");
    }
}

// ----- S1.5: DV-mode (assay-noised) monitors (#391) -----------------

fn dv_monitor() -> [MonitorSpec; 1] {
    [MonitorSpec::new("A", 1, ObserveMode::Dv)]
}

#[test]
fn adaptive_dv_without_assay_capability_errors() {
    // A DV monitor on an Ipred-only run (assay = None) is a typed error, not a
    // silent fallback to the latent value.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &dv_monitor(),
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(
        err.contains("DV") && err.contains("capability"),
        "got: {err}"
    );
}

#[test]
fn adaptive_dv_no_error_model_errors() {
    // Edge (a): a DV monitor on a compartment with no residual error model is a
    // typed error (resid_var returns None), never a fabricated sigma.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(vec![], vec![1.0]);
    let no_model = |_cmt: usize, _ipred: f64| None;
    let assay = AssayNoise {
        resid_var: &no_model,
        base_seed: 7,
    };
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &dv_monitor(),
        &mut controller,
        100,
        Some(&assay),
    )
    .unwrap_err();
    assert!(err.contains("error_model"), "got: {err}");
}

#[test]
fn adaptive_dv_zero_variance_equals_ipred() {
    // sigma -> 0: the DV signal collapses to the latent IPRED. Compare the value
    // the controller saw under a zero-variance assay against an Ipred monitor on
    // the same realized run.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = 0.1
    let decisions = [0.0, 24.0]; // dose at t=0, observe pre-dose trough at t=24
    let base = make_subject(vec![], vec![24.0]);
    let dose = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];

    let mut ctrl_ref = dose;
    let ref_run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[MonitorSpec::new("A", 1, ObserveMode::Ipred)],
        &mut ctrl_ref,
        100,
        None,
    )
    .expect("ipred run");
    let ipred = ref_run.decisions[1].observed_signals[0].value;

    let zero_var = |_cmt: usize, _ipred: f64| Some(0.0);
    let assay = AssayNoise {
        resid_var: &zero_var,
        base_seed: 12345,
    };
    let mut ctrl_dv = dose;
    let dv_run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &dv_monitor(),
        &mut ctrl_dv,
        100,
        Some(&assay),
    )
    .expect("dv run");
    let dv = dv_run.decisions[1].observed_signals[0].value;

    assert!(ipred > 0.0, "expected a non-zero trough at t=24");
    assert_relative_eq!(dv, ipred, epsilon = 1e-12);
}

#[test]
fn adaptive_dv_noised_and_deterministic() {
    // Non-zero variance perturbs the latent IPRED, and the draw is reproducible:
    // the same base seed yields the same value, a different seed a different one.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 24.0];
    let base = make_subject(vec![], vec![24.0]);
    let var4 = |_cmt: usize, _ipred: f64| Some(4.0); // sd = 2

    let observe = |seed: u64| {
        let assay = AssayNoise {
            resid_var: &var4,
            base_seed: seed,
        };
        let mut ctrl = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &dv_monitor(),
            &mut ctrl,
            100,
            Some(&assay),
        )
        .expect("dv run")
        .decisions[1]
            .observed_signals[0]
            .value
    };

    let a = observe(999);
    let b = observe(999);
    let c = observe(1000);
    assert_eq!(a, b, "same base seed must reproduce the assay draw");
    assert_ne!(a, c, "a different base seed must change the assay draw");
    let latent = 100.0 * (-2.4f64).exp(); // trough at t=24
    assert!(
        (a - latent).abs() > 1e-9,
        "expected the assay to perturb the latent value"
    );
}

#[test]
fn adaptive_dv_clamps_negative_at_zero() {
    // Edge (b): on a *bare-state* readout (`OdeReadout::ObsCmt`, a compartment
    // amount / concentration) the noised value cannot read below zero. At t=0 the
    // pre-dose trough is 0, so a negative assay draw with a large sigma would push
    // it negative; assert it clamps to exactly 0. Form C readouts are exempt —
    // see `adaptive_dv_form_c_monitor_keeps_negative_sample` (#1039).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let neg_seed = (0u64..)
        .find(|&s| assay_standard_normal(s, 0, "A") < 0.0)
        .expect("some seed gives a negative draw");
    let big_var = |_cmt: usize, _ipred: f64| Some(1.0e6);
    let assay = AssayNoise {
        resid_var: &big_var,
        base_seed: neg_seed,
    };
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &dv_monitor(),
        &mut controller,
        100,
        Some(&assay),
    )
    .expect("dv run");
    assert_eq!(
        run.decisions[0].observed_signals[0].value, 0.0,
        "a negative assay reading must clamp at 0"
    );
}

/// A Form C `[scaling]` monitor read in `dv` mode must keep its negative
/// samples (#1039). The assay floor is a statement about a compartment amount,
/// not about an arbitrary user expression — a change-from-baseline signal that
/// a controller thresholds is legitimately negative, and flooring it at 0 made
/// the controller blind over exactly the region it reacts to.
#[test]
fn adaptive_dv_form_c_monitor_keeps_negative_sample() {
    // `clock - 5`: latent = -5 at t=0. sd = 0.1, so no draw of a standard normal
    // this test could see reaches 0 — the sample must stay negative *and* noised.
    let ode = signed_readout_clock_spec(OdeReadout::Single(Box::new(
        |s: &[f64], _pk: &[f64], _t: &[f64], _e: &[f64], _c: &HashMap<String, f64>| s[0] - 5.0,
    )));
    let pk = pk_one(1.0, 1.0);
    let small_var = |_cmt: usize, _ipred: f64| Some(0.01);
    let assay = AssayNoise {
        resid_var: &small_var,
        base_seed: 4242,
    };
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(
        vec![DoseEvent::new(0.0, 0.0, 1, 0.0, false, 0.0)],
        vec![1.0],
    );
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &dv_monitor(),
        &mut controller,
        100,
        Some(&assay),
    )
    .expect("dv run");
    let value = run.decisions[0].observed_signals[0].value;
    assert!(
        value < 0.0,
        "a Form C dv monitor must keep its negative sample, got {value}"
    );
    assert!(
        (value + 5.0).abs() > 1e-12,
        "expected the assay to perturb the latent value, got {value}"
    );
    assert!(
        (value + 5.0).abs() < 1.0,
        "expected a sd=0.1 perturbation around -5, got {value}"
    );
}

/// Degenerate oracle for #1039 (epic #391 validation rule — no NONMEM anchor):
/// with `sigma -> 0` a `dv` monitor on a Form C model must equal the `ipred`
/// monitor on the same model, decision for decision, *including* the negative
/// samples. That equality is exactly what the ungated assay clamp broke.
#[test]
fn adaptive_dv_zero_variance_equals_ipred_on_signed_form_c() {
    let ode = signed_readout_clock_spec(OdeReadout::Single(Box::new(
        |s: &[f64], _pk: &[f64], _t: &[f64], _e: &[f64], _c: &HashMap<String, f64>| s[0] - 5.0,
    )));
    let pk = pk_one(1.0, 1.0);
    // Straddle the sign change at t=5: two negative samples, then two positive.
    let decisions = [0.0, 2.0, 6.0, 10.0];
    let base = make_subject(
        vec![DoseEvent::new(0.0, 0.0, 1, 0.0, false, 0.0)],
        vec![1.0],
    );
    let hold = |_ctx: &ControllerCtx| vec![DoseAction::Hold];

    let sample = |monitors: &[MonitorSpec], assay: Option<&AssayNoise>| -> Vec<f64> {
        let mut ctrl = hold;
        ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            monitors,
            &mut ctrl,
            100,
            assay,
        )
        .expect("adaptive run")
        .decisions
        .iter()
        .map(|d| d.observed_signals[0].value)
        .collect()
    };

    let ipred = sample(&[MonitorSpec::new("A", 1, ObserveMode::Ipred)], None);
    let zero_var = |_cmt: usize, _ipred: f64| Some(0.0);
    let assay = AssayNoise {
        resid_var: &zero_var,
        base_seed: 7,
    };
    let dv = sample(&dv_monitor(), Some(&assay));

    for (got, want) in ipred.iter().zip([-5.0, -3.0, 1.0, 5.0]) {
        assert_relative_eq!(*got, want, epsilon = 1e-9);
    }
    assert_eq!(
        dv, ipred,
        "a zero-variance dv monitor must reproduce the ipred monitor sample for sample"
    );
}

#[test]
fn adaptive_dv_added_monitor_does_not_perturb_other_draw() {
    // Non-perturbing: adding a second DV monitor (a new analyte) must not change
    // the first analyte's draw — each is keyed by its own analyte name.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 24.0];
    let base = make_subject(vec![], vec![24.0]);
    let var4 = |_cmt: usize, _ipred: f64| Some(4.0);

    let signal_a = |monitors: &[MonitorSpec]| {
        let assay = AssayNoise {
            resid_var: &var4,
            base_seed: 555,
        };
        let mut ctrl = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            monitors,
            &mut ctrl,
            100,
            Some(&assay),
        )
        .expect("dv run");
        run.decisions[1]
            .observed_signals
            .iter()
            .find(|s| s.name == "A")
            .expect("analyte A present")
            .value
    };

    let one = [MonitorSpec::new("A", 1, ObserveMode::Dv)];
    let two = [
        MonitorSpec::new("A", 1, ObserveMode::Dv),
        MonitorSpec::new("B", 1, ObserveMode::Dv),
    ];
    assert_eq!(
        signal_a(&one),
        signal_a(&two),
        "adding analyte B must not perturb A's draw"
    );
}

#[test]
fn adaptive_base_regimen_matches_static_ode() {
    // #702 driver-level oracle: a base loading regimen with a Hold-all controller must
    // reproduce `ode_predictions` on that regimen — the reactive driver seeds and
    // integrates the pre-scheduled doses through the same static-engine helpers.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    // Decisions on the loading-dose grid so the reactive driver and `ode_predictions`
    // segment identically (bit-exact); an off-grid decision would add a break the static
    // engine lacks, diverging by RK45 step noise (the frozen-replay verifier is the
    // bit-exact check when decisions are off-grid).
    let decisions = [0.0, 24.0];
    let obs = vec![6.0, 30.0, 54.0];
    let loading = vec![
        DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 250.0, 1, 0.0, false, 0.0),
    ];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(loading.clone(), obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs with a base regimen");
    assert!(run.ledger.is_empty(), "Hold controller adds no doses");

    let static_subject = make_subject(loading, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
}

#[test]
fn adaptive_base_regimen_plus_titration_matches_static_ode() {
    // #702: base loading dose + a fixed controller ≡ `ode_predictions` on (base ∪ ledger).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 54.0];
    let loading = vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(loading.clone(), obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.ledger.len(), 3);

    let mut static_doses = loading;
    static_doses.extend(
        run.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let static_subject = make_subject(static_doses, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
}

/// 1-cpt IV amount ODE with a per-compartment dose lagtime (`ALAG1`) at `lag_slot`. Used
/// to exercise a base dose into a lagged compartment on the reactive path (#935).
fn one_cpt_lag_spec(lag_slot: usize) -> OdeSpec {
    let mut map = crate::types::DoseAttrMap::default();
    map.insert(crate::types::DoseAttr::Lag, 1, lag_slot);
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: map,
        init_fn: None,
    }
}

#[test]
fn adaptive_base_dose_into_input_rate_compartment_matches_static_ode() {
    // #935: a base dose into a built-in input-rate (first_order absorption) compartment is
    // delivered as `R_in` over time, NOT as a bolus jump — exercising the reactive
    // `input_rate_consumes_cmt` skip in `apply_prescheduled_boluses_at` on a genuine base
    // run (previously covered only via the static-engine code motion). A Hold controller
    // with the single decision on the integration start keeps the segmentation identical to
    // `ode_predictions`, so the match is bit-exact.
    let mut ode = first_order_one_cpt_spec();
    ode.solver_opts.reltol = 1e-10;
    ode.solver_opts.abstol = 1e-10;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 1.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.5; // ka
    pk.values[crate::types::PK_IDX_F] = 1.0;
    let decisions = [0.0];
    let obs = vec![2.0, 12.0, 30.0];
    let base = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base_subj = make_subject(base.clone(), obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base_subj,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs with an input-rate base dose");
    assert!(run.ledger.is_empty(), "Hold controller adds no doses");

    let static_subj = make_subject(base, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subj);
    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    // Non-vacuity: the input-rate base dose actually drives the compartment.
    assert!(run.predictions.iter().any(|&p| p > 1.0));
}

#[test]
fn adaptive_base_dose_with_lagtime_matches_static_ode() {
    // #935: a base dose into a LAGGED compartment — its bolus lands at `dose.time + lag`, and
    // the reactive `dose_lagtimes` / break placement / `apply_prescheduled_boluses_at` lag
    // filter must reproduce `ode_predictions`. `ALAG1 = 5 h`; the single decision on the
    // integration start keeps the segmentation identical (bit-exact). Base doses into lagged
    // compartments are supported (they run the exact static machinery `predict()` uses) —
    // unlike CONTROLLER doses into a lagged compartment, which are rejected at injection
    // (`reject_unsupported_dose_compartment`) because the TAD-anchor/double-count subtleties
    // bite only for a dose discovered mid-run, not a pre-resolved base dose.
    let lag_slot = 8usize;
    let ode = one_cpt_lag_spec(lag_slot);
    let mut pk = pk_one(1.0, 10.0);
    pk.values[lag_slot] = 5.0; // ALAG1 = 5 h
    let decisions = [0.0];
    let obs = vec![3.0, 8.0, 24.0]; // t=3 pre-lag (empty), t=8/24 post-lag
    let base = vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base_subj = make_subject(base.clone(), obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base_subj,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs with a lagged base dose");
    assert!(run.ledger.is_empty(), "Hold controller adds no doses");

    let static_subj = make_subject(base, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subj);
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    // The lag is real: nothing has arrived at t=3 (pre-lag), the dose is present by t=8.
    assert!(run.predictions[0].abs() < 1e-9, "pre-lag readout must be 0");
    assert!(run.predictions[1] > 100.0, "post-lag dose must be present");
}

#[test]
fn adaptive_base_regimen_with_reset_matches_static_ode_driver() {
    // #932 driver-level oracle (constant covariates, tv/iov off): a base loading regimen on a
    // reset-carrying subject now integrates — the reset zeros the state and lowers the reset
    // floor — and must reproduce `ode_predictions` (itself reset-aware) on the realized
    // (base ∪ ledger) regimen carrying the same reset. On-grid decisions keep the two engines'
    // segmentation identical, so the match is bit-tight. (Base × reset UNDER a time-varying
    // covariate / IOV — `event_pk`/`eta_occ` = Some — stays a typed error; that boundary is
    // covered by `adaptive_base_regimen_with_reset_under_iov_is_rejected`.)
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [24.0];
    let obs = vec![6.0, 18.0, 30.0];
    let reset_at = 12.0;
    let loading = vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)];

    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let mut base = make_subject(loading.clone(), obs.clone());
    base.reset_times = vec![reset_at];
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs with a base regimen across a reset");
    assert_eq!(run.ledger.len(), 1, "one controller bolus at t=24");

    let mut static_doses = loading;
    static_doses.extend(
        run.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let mut static_subject = make_subject(static_doses, obs);
    static_subject.reset_times = vec![reset_at];
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
}

#[test]
fn adaptive_max_decisions_runaway_guard() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0, 24.0, 48.0],
        &[],
        &mut controller,
        2,
        None,
    )
    .unwrap_err();
    assert!(err.contains("max_decisions"), "got: {err}");
}

#[test]
fn adaptive_rejects_zero_compartment_via_validate() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 0 }];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("compartment"), "got: {err}");
}

#[test]
fn ode_predictions_applies_a_dose_landing_on_the_last_observation() {
    // #731: a bolus coinciding with the LAST observation time must be applied
    // before that observation is read (post-dose) — exactly as every INTERIOR
    // dose already is, and as the reactive driver and analytical engine do. The
    // old `0..len-1` loop treated the final break as an integration endpoint
    // only, so a terminal dose was silently dropped and the last obs read
    // pre-dose. Pinned against the exact 1-cpt closed form.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = CL/V = 0.1
    let ke = 0.1_f64;
    // Boluses at t=0 and t=24; the second lands exactly on the last obs (t=24).
    let subject = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        vec![6.0, 24.0],
    );
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subject);
    // t=6: only the t=0 dose has been given.
    assert_relative_eq!(preds[0], 100.0 * (-ke * 6.0).exp(), max_relative = 1e-5);
    // t=24: the first dose decayed 24 h PLUS the fresh post-dose 100 mg bolus.
    // Before #731 this read the pre-dose 100·e^{-2.4} ≈ 9.07 (the dose dropped).
    assert_relative_eq!(
        preds[1],
        100.0 * (-ke * 24.0).exp() + 100.0,
        max_relative = 1e-5
    );
}

#[test]
fn ode_predictions_single_instant_dose_and_obs_not_double_applied() {
    // #731: a subject whose only records are a bolus and an observation at the
    // SAME instant (t=0) has a 1-element `[t0]` timeline. The `0..len` loop runs it
    // exactly once (k=0), applying the dose and reading the obs post-dose with no
    // trailing integration (`k + 1 < len` is false) — so the dose is applied once
    // (100), not twice (200). Guards against a regression that double-visits the
    // sole break.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        vec![0.0],
    );
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subject);
    // Exactly one 100 mg bolus (readout is the compartment amount, ke·t = 0 at t=0).
    assert_relative_eq!(preds[0], 100.0, max_relative = 1e-9);
}

#[test]
fn adaptive_final_decision_at_max_time_still_fires() {
    // Regression: the last decision must fire even when it lands on the
    // schedule's maximum time (i.e. at or after the last observation). Here
    // the second decision (t=24) coincides with the last observation, so it
    // is the maximum break time; it must still dose, reach the ledger, and
    // make the t=24 observation read the *post*-dose state.
    //
    // Checked against the exact 1-cpt closed form (ke = CL/V = 0.1). Since #731
    // the constant-parameter `ode_predictions` engine also applies a dose on its
    // terminal break, so a static comparison would no longer mask the bug; the
    // closed form remains the tightest oracle and is kept.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = 0.1
    let decisions = [0.0, 24.0];
    let obs = vec![6.0, 24.0]; // last obs coincides with the last decision
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(vec![], obs);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    // Both decisions dosed — including the one at the maximum time.
    assert_eq!(run.ledger.len(), 2, "final decision at t_max must dose");
    assert_eq!(run.ledger[1].time, 24.0);
    assert_eq!(run.ledger[1].decision_idx, 1);

    let ke = 0.1_f64;
    // t=6: only the t=0 dose has been given.
    assert_relative_eq!(
        run.predictions[0],
        100.0 * (-ke * 6.0).exp(),
        max_relative = 1e-5
    );
    // t=24: first dose decayed to 24 h, plus the fresh 100 mg bolus (post-dose).
    let expected_24 = 100.0 * (-ke * 24.0).exp() + 100.0;
    assert_relative_eq!(run.predictions[1], expected_24, max_relative = 1e-5);
}

#[test]
fn adaptive_rejects_out_of_range_bolus_compartment() {
    // `validate()` only catches cmt == 0; an out-of-range cmt (> n_states) is
    // caught by the driver's own guard. 1-state model, bolus into cmt 2.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 2 }];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("state"), "got: {err}");
}

#[test]
fn adaptive_rejects_out_of_range_monitor_compartment() {
    // A monitor on a compartment beyond the model is a precondition error.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let monitors = [MonitorSpec::new("A", 2, ObserveMode::Ipred)]; // n_states = 1
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &monitors,
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("state"), "got: {err}");
}

#[test]
fn adaptive_rejects_dosing_into_input_rate_compartment() {
    // A bolus into a compartment fed by a built-in input-rate function would
    // be double-counted (state jump *and* `R_in` forcing). Must be a typed
    // error, not a silent wrong answer.
    let mut ode = one_cpt_ode_spec();
    ode.input_rate = vec![crate::pk::absorption::InputRateForcing {
        cmt: 0, // 0-based -> consumes 1-based compartment 1
        kind: crate::pk::absorption::InputRateKind::Transit,
        arg_slots: vec![],
        frac_slot: None,
        lag_slot: None,
    }];
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("input-rate"), "got: {err}");
}

#[test]
fn adaptive_rejects_lagged_dose_compartment() {
    // A lag time on the dosed compartment would be applied with zero delay
    // here yet dropped from its own TAD anchor in `integrate_segment`. Reject.
    let ode = one_cpt_ode_spec();
    let mut pk = pk_one(1.0, 10.0);
    pk.values[crate::types::PK_IDX_LAGTIME] = 2.0; // bare-slot lag on cmt 1
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("lag time"), "got: {err}");
}

// ----- S1.3b reactive infusions (#391) ------------------------------

#[test]
fn insert_break_keeps_sorted_and_dedups_within_tolerance() {
    let mut breaks = vec![0.0, 10.0, 20.0];
    insert_break(&mut breaks, 5.0); // strictly between -> inserted
    assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0]);
    insert_break(&mut breaks, 10.0 + 1e-16); // within 1e-15 of existing -> dropped
    assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0]);
    insert_break(&mut breaks, 25.0); // past the end -> appended
    assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0, 25.0]);
    insert_break(&mut breaks, 0.0); // duplicate of the first -> dropped
    assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0, 25.0]);
}

#[test]
fn adaptive_infusion_matches_closed_form() {
    // Absolute oracle: a single zero-order infusion into a 1-cpt linear model
    // has the closed form A(t) = (R/ke)(1 - e^{-ke t}) while infusing and
    // A(t_inf)·e^{-ke (t - t_inf)} afterward. Pins magnitude against
    // mathematics, not just against the static engine.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = 0.1
    let ke = 0.1;
    let (rate, amt) = (10.0_f64, 100.0_f64);
    let t_inf = amt / rate; // 10 h (F = 1)
    let obs = vec![5.0, 10.0, 20.0]; // during, at end, after
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Infuse { amt, cmt: 1, rate }];
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    let a_inf = (rate / ke) * (1.0 - (-ke * t_inf).exp());
    let expected = [
        (rate / ke) * (1.0 - (-ke * 5.0_f64).exp()), // during
        a_inf,                                       // at end
        a_inf * (-ke * (20.0 - t_inf)).exp(),        // after
    ];
    // RK45-vs-analytical tolerance (the established 1e-4 in this file): the
    // bit-exact 1e-9 oracle below pins the integrator to the static engine;
    // this test pins *magnitude* against mathematics, where 0.01% is ample.
    for (i, e) in expected.iter().enumerate() {
        assert_relative_eq!(run.predictions[i], *e, max_relative = 1e-4);
    }
}

#[test]
fn adaptive_overlapping_infusions_match_static() {
    // The hard case: an infusion whose end falls *after* the next decision, so
    // two controller infusions overlap. `active_infusions` must sum both rates
    // over the overlap window, and the timeline must carry both ends as breaks.
    // Compared bit-exact to the equivalent two-infusion static schedule (dosing
    // at every decision, so there is no phantom break).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 5.0];
    // 100 mg @ rate 10 -> 10 h each: windows [0,10] and [5,15] overlap on [5,10].
    let obs = vec![2.0, 7.0, 12.0, 20.0]; // last (20) past both ends
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 10.0,
        }]
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    let static_doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0),
        DoseEvent::new(5.0, 100.0, 1, 10.0, false, 0.0),
    ];
    let static_subject = make_subject(static_doses, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    assert_eq!(run.ledger.len(), 2);
}

#[test]
fn adaptive_infusion_end_coincident_with_decision_dedups() {
    // An infusion that ends *exactly* at the next decision must not create a
    // second break: the end coincides with the decision break (which the static
    // engine also has, as the infusion end), so the timelines match bit-exact.
    // A hold at that decision is therefore safe to compare to the
    // single-infusion static schedule (no phantom break).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let decisions = [0.0, 4.0]; // 100@25 -> 4 h, ends exactly at decision 1
    let obs = vec![2.0, 4.0, 8.0];
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.decision_index == 0 {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 25.0,
            }]
        } else {
            vec![DoseAction::Hold]
        }
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.ledger.len(), 1, "only the first decision infuses");

    let static_subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)], obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
}

#[test]
fn adaptive_infusion_f_scaling_matches_static() {
    // The S1.3b invariant *under F != 1*: the F-scaled infusion end inserted as
    // a break (`t_start + dur_eff`, from `bioavailable_infusion`) must coincide
    // with the F-scaled window `active_infusions` re-derives inside
    // `integrate_segment`. At F = 1 the two are trivially equal (`dur_eff ==
    // amt/rate`), so every other oracle test leaves this seam unexercised. Here
    // a bare-slot F = 0.5 halves a rate-defined infusion's window to F·amt/rate;
    // the degenerate oracle must still reproduce the equivalent static infusion
    // schedule (carrying the same F) bit-exact.
    let ode = one_cpt_ode_spec();
    let mut pk = pk_one(1.0, 10.0); // ke = 0.1
    pk.values[crate::types::PK_IDX_F] = 0.5; // bare-slot F on all compartments
    let decisions = [0.0, 24.0, 48.0];
    // 100 mg @ rate 25 -> nominal 4 h window, F-scaled to 0.5*4 = 2 h. Ends at
    // 2, 26, 50 (between decisions). The last obs (60) is past every end, so it
    // is the global maximum and neither engine breaks at an interior obs.
    let obs = vec![1.0, 3.0, 25.0, 27.0, 49.0, 60.0];
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 25.0,
        }]
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 25.0, false, 0.0))
        .collect();
    let static_subject = make_subject(static_doses, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    // F is actually applied (window halved), recorded as f_applied on each row.
    assert_eq!(run.ledger.len(), 3);
    for entry in &run.ledger {
        assert_eq!(entry.f_applied, 0.5);
    }
}

#[test]
fn adaptive_bolus_f_scaling_matches_static() {
    // Coverage for the bolus emit-path multiply `u[cmt-1] += F*amt` under
    // F != 1. The infusion-F seam is covered above, but no test drove the
    // *bolus* F multiply with F != 1 (it shares the static engine's structure,
    // but was unexercised). A bare-slot F = 0.5 halves every controller-issued
    // bolus; the degenerate oracle (re-issue the same bolus at each decision)
    // must reproduce the equivalent static bolus schedule (carrying the same F)
    // bit-exact.
    let ode = one_cpt_ode_spec();
    let mut pk = pk_one(1.0, 10.0); // ke = 0.1
    pk.values[crate::types::PK_IDX_F] = 0.5; // bare-slot F on all compartments
    let decisions = [0.0, 24.0, 48.0];
    // A dose is realized at every decision and the last observation (60) is the
    // global maximum, so neither engine breaks at an interior observation — the
    // condition under which the degenerate oracle is bit-exact.
    let obs = vec![1.0, 12.0, 25.0, 36.0, 49.0, 60.0];
    let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_subject = make_subject(static_doses, obs);
    let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

    assert_eq!(run.predictions.len(), static_preds.len());
    for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
        assert_relative_eq!(*got, *want, max_relative = 1e-9);
    }
    // F is actually applied to the bolus (u += F*amt), recorded per row.
    assert_eq!(run.ledger.len(), 3);
    for entry in &run.ledger {
        assert_eq!(entry.f_applied, 0.5);
    }
}

#[test]
fn adaptive_reactive_infusion_titrates_against_closed_form() {
    // Genuine state-reactive infusion: infuse 100 mg @ 25 (4 h) only when the
    // monitored amount is below 50, else hold. Checked against the exact 1-cpt
    // infusion closed form — NOT the static engine: the hold at the second
    // decision makes the driver break where a dose-list replay would not, so a
    // static comparison would confound the integrator restart with the logic
    // (same reason as the bolus feedback test).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = 0.1
    let ke = 0.1;
    let (rate, amt) = (25.0_f64, 100.0_f64);
    let t_inf = amt / rate; // 4 h
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let decisions = [0.0, 6.0];
    let obs = vec![2.0, 5.0, 8.0];
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("A declared") < 50.0 {
            vec![DoseAction::Infuse { amt, cmt: 1, rate }]
        } else {
            vec![DoseAction::Hold]
        }
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &monitors,
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    // t=0: A=0 (<50) -> infuse [0,4]. By t=6 the amount has decayed back above
    // 50 (A(6) = a_inf·e^{-0.2} ≈ 67.5) -> hold. Exactly one realized dose.
    assert_eq!(run.ledger.len(), 1, "infuse once, then hold");
    assert_eq!(run.ledger[0].rule_fired, "infuse");

    let a_inf = (rate / ke) * (1.0 - (-ke * t_inf).exp()); // amount at end of infusion
    let expected = [
        (rate / ke) * (1.0 - (-ke * 2.0_f64).exp()), // t=2 during infusion
        a_inf * (-ke * (5.0 - t_inf)).exp(),         // t=5 post infusion
        a_inf * (-ke * (8.0 - t_inf)).exp(),         // t=8 post infusion
    ];
    for (i, e) in expected.iter().enumerate() {
        assert_relative_eq!(run.predictions[i], *e, max_relative = 1e-4);
    }
}

#[test]
fn adaptive_stop_lets_in_flight_infusion_complete() {
    // Contract: `Stop` discontinues *future* decisions, but an infusion already
    // issued is a committed dose and keeps delivering to its end. Infuse at t=0
    // over [0,20]; Stop at t=5. The infusion must still be active at t=10 (well
    // past the Stop) and finish at t=20 — verified against the closed form. A
    // true safety-halt that truncates delivery is a separate action (tracked as
    // a follow-up), deliberately not conflated with `Stop`.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = 0.1
    let ke = 0.1;
    let (rate, amt) = (5.0_f64, 100.0_f64);
    let t_inf = amt / rate; // 20 h
    let decisions = [0.0, 5.0];
    let obs = vec![10.0, 25.0];
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.decision_index == 0 {
            vec![DoseAction::Infuse { amt, cmt: 1, rate }]
        } else {
            vec![DoseAction::Stop]
        }
    };
    let base = make_subject(vec![], obs.clone());
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    assert_eq!(run.ledger.len(), 1, "Stop adds no dose; the infusion stays");
    let a_inf = (rate / ke) * (1.0 - (-ke * t_inf).exp());
    let expected = [
        (rate / ke) * (1.0 - (-ke * 10.0_f64).exp()), // t=10: still infusing despite Stop@5
        a_inf * (-ke * (25.0 - t_inf)).exp(),         // t=25: after the infusion finished @20
    ];
    for (i, e) in expected.iter().enumerate() {
        assert_relative_eq!(run.predictions[i], *e, max_relative = 1e-4);
    }
}

#[test]
fn adaptive_zero_amount_infusion_is_treated_as_hold() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 0.0,
            cmt: 1,
            rate: 10.0,
        }]
    };
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert!(
        run.ledger.is_empty(),
        "zero-amount infusion records no dose"
    );
    assert_eq!(run.predictions[0], 0.0);
}

#[test]
fn adaptive_rejects_nonpositive_infusion_rate() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 0.0,
        }]
    };
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("rate"), "got: {err}");
}

#[test]
fn adaptive_rejects_out_of_range_infusion_compartment() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 2,
            rate: 10.0,
        }]
    };
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("state"), "got: {err}");
}

#[test]
fn adaptive_rejects_infusion_into_input_rate_compartment() {
    let mut ode = one_cpt_ode_spec();
    ode.input_rate = vec![crate::pk::absorption::InputRateForcing {
        cmt: 0, // 0-based -> consumes 1-based compartment 1
        kind: crate::pk::absorption::InputRateKind::Transit,
        arg_slots: vec![],
        frac_slot: None,
        lag_slot: None,
    }];
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 10.0,
        }]
    };
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("input-rate"), "got: {err}");
}

#[test]
fn adaptive_rejects_lagged_infusion_compartment() {
    let ode = one_cpt_ode_spec();
    let mut pk = pk_one(1.0, 10.0);
    pk.values[crate::types::PK_IDX_LAGTIME] = 2.0; // bare-slot lag on cmt 1
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 10.0,
        }]
    };
    let base = make_subject(vec![], vec![1.0]);
    let err = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .unwrap_err();
    assert!(err.contains("lag time"), "got: {err}");
}

// ----- S1.4a decision log (#391) ------------------------------------

#[test]
fn adaptive_decision_log_records_dose_hold_and_stop() {
    // Every decision is logged — including the hold, which leaves no ledger
    // row — with the signal the controller observed and the outcome it chose.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let decisions = [0.0, 24.0, 48.0];
    let mut controller = |ctx: &ControllerCtx| match ctx.decision_index {
        0 => vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
        1 => vec![DoseAction::Hold],
        _ => vec![DoseAction::Stop],
    };
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &decisions,
        &monitors,
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");

    assert_eq!(run.decisions.len(), 3, "one log entry per decision");
    for (i, d) in run.decisions.iter().enumerate() {
        assert_eq!(d.decision_idx, i);
        assert_eq!(d.time, decisions[i]);
        assert_eq!(d.observed_signals.len(), 1);
        assert_eq!(d.observed_signals[0].name, "A");
    }
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Dosed { n: 1 });
    assert_eq!(run.decisions[1].outcome, DecisionOutcome::Hold);
    assert_eq!(run.decisions[2].outcome, DecisionOutcome::Stop { dosed: 0 });
    // The pre-dose signal at the first decision is the empty initial state.
    assert_eq!(run.decisions[0].observed_signals[0].value, 0.0);
}

#[test]
fn adaptive_decision_log_omits_decisions_after_stop() {
    // Once the controller stops, the driver issues no further decisions, so
    // the Stop entry is the last record (no phantom post-stop log rows).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |ctx: &ControllerCtx| {
        if ctx.decision_index == 0 {
            vec![DoseAction::Stop]
        } else {
            // The driver must never call the controller again after a Stop;
            // reaching here would be the bug this test guards against.
            unreachable!(
                "driver issued a decision after Stop (idx {})",
                ctx.decision_index
            )
        }
    };
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0, 10.0, 20.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.decisions.len(), 1, "only the stop decision is logged");
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Stop { dosed: 0 });
    assert!(run.ledger.is_empty());
}

#[test]
fn adaptive_decision_log_dose_then_stop_in_one_action_list() {
    // `[Bolus, Stop]` — a final dose, then discontinue — is logged as
    // `Stop { dosed: 1 }`, not a bare stop, and the dose reaches the ledger.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller =
        |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }, DoseAction::Stop];
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0, 24.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.decisions.len(), 1, "stop ends the schedule after one");
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Stop { dosed: 1 });
    assert_eq!(run.ledger.len(), 1);
}

#[test]
fn adaptive_decision_log_counts_multiple_doses_in_one_decision() {
    // A decision can issue more than one dose (e.g. a loading split); the log
    // records `Dosed { n }` with the realized count, and a zero-amount action
    // in the same list is excluded (it leaves no ledger row).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![
            DoseAction::Bolus { amt: 50.0, cmt: 1 },
            DoseAction::Bolus { amt: 0.0, cmt: 1 }, // normalized to Hold, not counted
            DoseAction::Bolus { amt: 50.0, cmt: 1 },
        ]
    };
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.decisions.len(), 1);
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Dosed { n: 2 });
    assert_eq!(
        run.ledger.len(),
        2,
        "two realized doses, the zero-amt excluded"
    );
}

#[test]
fn adaptive_decision_log_records_infusion_as_dosed() {
    // An infusion is a realized dose: its decision categorizes to `Dosed { n }`
    // exactly as a bolus does (the outcome doesn't distinguish route), and it
    // reaches the ledger.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![DoseAction::Infuse {
            amt: 100.0,
            cmt: 1,
            rate: 50.0,
        }]
    };
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.decisions.len(), 1);
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Dosed { n: 1 });
    assert_eq!(run.ledger.len(), 1);
}

#[test]
fn adaptive_decision_log_infusion_then_stop() {
    // `[Infuse, Stop]` mirrors the bolus dose-then-stop: a final infusion, then
    // discontinue, logged as `Stop { dosed: 1 }` with the infusion in the ledger.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| {
        vec![
            DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 50.0,
            },
            DoseAction::Stop,
        ]
    };
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0, 24.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.decisions.len(), 1, "stop ends the schedule after one");
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Stop { dosed: 1 });
    assert_eq!(run.ledger.len(), 1);
}

#[test]
fn adaptive_decision_log_empty_action_list_is_hold() {
    // An empty action list is a no-change decision: it categorizes to `Hold`
    // (no dose, not stopped) and leaves no ledger row — same as `[Hold]`.
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let mut controller = |_ctx: &ControllerCtx| Vec::<DoseAction>::new();
    let base = make_subject(vec![], vec![1.0]);
    let run = ode_predictions_adaptive(
        &ode,
        &pk.values,
        &[],
        &[],
        &base,
        &[0.0],
        &[],
        &mut controller,
        100,
        None,
    )
    .expect("driver runs");
    assert_eq!(run.decisions.len(), 1);
    assert_eq!(run.decisions[0].outcome, DecisionOutcome::Hold);
    assert!(run.ledger.is_empty());
}

#[test]
fn adaptive_driver_rejects_malformed_or_post_stop_actions() {
    // The whole action list is validated up front, before anything is applied:
    // a malformed action is a typed error wherever it sits, and `Stop` must be
    // the final action — a controller that issues actions after discontinuing is
    // rejected, not silently truncated, so the log can't disagree with the
    // ledger. Nothing is applied when the list is rejected (the ledger would be
    // discarded with the `Err` regardless).
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let base = make_subject(vec![], vec![1.0]);

    let cases: [(Vec<DoseAction>, &str); 3] = [
        // Malformed action (compartment 0) -> the up-front validate() error.
        (
            vec![DoseAction::Bolus { amt: 100.0, cmt: 0 }],
            "compartment is 0",
        ),
        // A well-formed action after a Stop -> Stop-must-be-final error.
        (
            vec![DoseAction::Stop, DoseAction::Bolus { amt: 100.0, cmt: 1 }],
            "Stop must be the final action",
        ),
        // A Stop in the middle of the list -> same rejection (not a silent drop
        // of the trailing dose).
        (
            vec![
                DoseAction::Bolus { amt: 50.0, cmt: 1 },
                DoseAction::Stop,
                DoseAction::Bolus { amt: 50.0, cmt: 1 },
            ],
            "Stop must be the final action",
        ),
    ];

    for (actions, needle) in cases {
        let mut controller = |_ctx: &ControllerCtx| actions.clone();
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect_err("malformed / post-stop action list is rejected");
        assert!(err.contains(needle), "expected {needle:?}, got: {err}");
    }
}

#[test]
fn integrate_segment_tad_anchor_set_when_prior_dose_exists() {
    // Covers the `last_dose_eff.is_finite()` branch: when a dose precedes the
    // segment the TAD anchor slot must hold that dose time (not NaN).
    let ode = one_cpt_ode_spec();
    let dose = crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let subject = make_subject(vec![dose], vec![10.0]);
    let pk = pk_one(1.0, 10.0);
    let mut ext_params = [0.0f64; crate::types::MAX_PK_PARAMS + 2];
    ext_params[crate::types::PK_IDX_CL] = 1.0;
    ext_params[crate::types::PK_IDX_V] = 10.0;
    let mut u = vec![100.0]; // pre-loaded with the bolus amount
    let mut predictions = vec![f64::NAN; subject.obs_times.len()];
    let obs_map = obs_index_map(&subject.obs_times);

    integrate_segment(
        &ode,
        &mut u,
        0.0,
        10.0,
        &subject,
        &[0.0],
        &[1.0],
        f64::NEG_INFINITY,
        &mut ext_params,
        &pk.values,
        &[],
        &[],
        &obs_map,
        &mut predictions,
        None,
        &[],
    );

    // TAD anchor must be the dose time (0.0), not NaN.
    assert_eq!(
        ext_params[crate::types::MAX_PK_PARAMS + 1],
        0.0,
        "TAD anchor must equal the prior dose time"
    );
    let expected = 100.0 * (-1.0f64).exp();
    assert_relative_eq!(predictions[0], expected, max_relative = 1e-4);
}

/// Two-compartment "accumulator": `d/dt = 0` for both states, so each state
/// holds exactly the bioavailable amount injected into it — letting a test
/// read `F·amt` (and lag timing) straight off the state. `readout_idx`
/// selects which compartment the observable reads.
fn two_cpt_accumulator(readout_idx: usize, map: crate::types::DoseAttrMap) -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
            dy[1] = 0.0;
        }),
        n_states: 2,
        state_names: vec!["c1".into(), "c2".into()],
        readout: OdeReadout::ObsCmt(readout_idx),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: map,
        init_fn: None,
    }
}

#[test]
fn ode_predictions_apply_per_compartment_bioavailability_and_lag() {
    // Issue #369. Dose 100 into cmt 1 and 100 into cmt 2. Bare F = 0.5
    // applies to every compartment; `F2` = 0.25 overrides compartment 2;
    // `ALAG2` = 5 h delays only the compartment-2 dose. Reading each state
    // off the accumulator must show the *per-compartment* attribute.
    let mut map = crate::types::DoseAttrMap::default();
    map.insert(crate::types::DoseAttr::F, 2, 9); // F2 -> spare slot 9
    map.insert(crate::types::DoseAttr::Lag, 2, 10); // ALAG2 -> spare slot 10

    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_F] = 0.5; // bare F (all compartments)
    p.values[9] = 0.25; // F2 overrides cmt 2
    p.values[10] = 5.0; // ALAG2 on cmt 2

    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0),
    ];
    // Observe at t = 1 (before ALAG2 = 5) and t = 10 (after).
    let subj = make_subject(doses, vec![1.0, 10.0]);

    // Compartment 1: bare F = 0.5, no lag -> 50 at both times.
    let c1 = ode_predictions(
        &two_cpt_accumulator(0, map.clone()),
        &p.values,
        &[],
        &[],
        &subj,
    );
    assert!((c1[0] - 50.0).abs() < 1e-9, "cmt1 @t=1: {}", c1[0]);
    assert!((c1[1] - 50.0).abs() < 1e-9, "cmt1 @t=10: {}", c1[1]);

    // Compartment 2: F2 = 0.25 and ALAG2 = 5 -> 0 before lag, 25 after.
    let c2 = ode_predictions(&two_cpt_accumulator(1, map), &p.values, &[], &[], &subj);
    assert!(c2[0].abs() < 1e-9, "cmt2 pre-lag: {}", c2[0]);
    assert!((c2[1] - 25.0).abs() < 1e-9, "cmt2 @t=10 (F2): {}", c2[1]);
}

#[test]
fn ode_predictions_event_driven_apply_per_compartment_bioavailability_and_lag() {
    // #369 review #3: the event-driven path is the actual fit path and
    // resolves F through a *distinct* inline form
    // (`dose_attr_map.f_bio(d.cmt, &pk_now.values)`), so per-compartment
    // correctness must be asserted here too — not only on `ode_predictions`.
    // Same 2-compartment accumulator and expectations as the no-TV test.
    let mut map = crate::types::DoseAttrMap::default();
    map.insert(crate::types::DoseAttr::F, 2, 9);
    map.insert(crate::types::DoseAttr::Lag, 2, 10);

    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_F] = 0.5;
    p.values[9] = 0.25; // F2
    p.values[10] = 5.0; // ALAG2

    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![1.0, 10.0]);
    let dose_pk = vec![p; subj.doses.len()];
    let obs_pk = vec![p; subj.obs_times.len()];

    // Compartment 1: bare F = 0.5, no lag.
    let c1 = ode_predictions_event_driven(
        &two_cpt_accumulator(0, map.clone()),
        &subj,
        &[],
        &[],
        &dose_pk,
        &obs_pk,
        &[],
        &[],
    );
    assert!((c1[0] - 50.0).abs() < 1e-9, "cmt1 @t=1: {}", c1[0]);
    assert!((c1[1] - 50.0).abs() < 1e-9, "cmt1 @t=10: {}", c1[1]);

    // Compartment 2: F2 = 0.25, ALAG2 = 5 -> 0 pre-lag, 25 after.
    let c2 = ode_predictions_event_driven(
        &two_cpt_accumulator(1, map),
        &subj,
        &[],
        &[],
        &dose_pk,
        &obs_pk,
        &[],
        &[],
    );
    assert!(c2[0].abs() < 1e-9, "cmt2 pre-lag: {}", c2[0]);
    assert!((c2[1] - 25.0).abs() < 1e-9, "cmt2 @t=10 (F2): {}", c2[1]);
}

/// Coverage: the steady-state branch of the event-driven TAD anchor in
/// `ode_predictions_event_driven` (`last_dose_eff` reckons from the most
/// recent SS cycle). Smoke-level — predictions must stay finite.
#[test]
fn event_driven_ss_dose_predictions_finite() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(5.0, 80.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)]; // SS bolus
    let subj = make_subject(doses, vec![6.0, 18.0]);
    let dose_pk = vec![pk; subj.doses.len()];
    let obs_pk = vec![pk; subj.obs_times.len()];
    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &dose_pk, &obs_pk, &[], &[]);
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "SS preds finite: {preds:?}"
    );
}

/// Coverage: the infusion break-time branch of `ode_predictions_with_states`.
#[test]
fn with_states_infusion_dose_runs() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(5.0, 80.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)]; // infusion, dur=10
    assert!(is_real_infusion(&doses[0]));
    let subj = make_subject(doses, vec![5.0, 20.0]);
    let (preds, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    assert_eq!(states.len(), 2);
    assert!(preds.iter().all(|p| p.is_finite()));
}

/// Coverage: `ode_dense_solve_states` with a steady-state, *lagged* infusion —
/// exercises the infusion break, the SS pre-seed at the dose record time, and
/// the SS branch of the dense TAD anchor in a single pass.
#[test]
fn dense_solve_ss_lagged_infusion_runs() {
    let ode = one_cpt_ode_spec();
    let mut pk = pk_one(5.0, 80.0);
    pk.values[crate::types::PK_IDX_LAGTIME] = 2.0; // lag > 0
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 10.0, true, 12.0)]; // SS infusion
    let subj = make_subject(doses, vec![6.0]);
    let states = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subj, &[6.0, 14.0]);
    assert_eq!(states.len(), 2);
    assert!(states.iter().all(|s| s.iter().all(|x| x.is_finite())));
}

/// Coverage: the `ode_predictions_ekf` wrapper (a 1-state `[diffusion]` spec);
/// elsewhere only `solve_ekf` is exercised directly.
#[test]
fn ode_predictions_ekf_wrapper_runs() {
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: vec![0.1],
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    };
    let pk = pk_one(5.0, 80.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![2.0, 8.0]);
    let (ipreds, p_obs) = ode_predictions_ekf(&ode, &pk.values, &subj, |_| 1.0);
    assert_eq!(ipreds.len(), 2);
    assert!(ipreds.iter().chain(p_obs.iter()).all(|x| x.is_finite()));
}

/// Turnover model with a baseline initial condition:
///   d/dt(R) = kin - kout*R,  init(R) = kin/kout
/// params: kin @ slot 0, kout @ slot 1. Observable reads R (state 0).
fn turnover_ode_spec_with_init() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = p[0] - p[1] * y[0];
        }),
        n_states: 1,
        state_names: vec!["R".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: Some(Box::new(|p: &[f64]| {
            let (kin, kout) = (p[0], p[1]);
            vec![if kout > 0.0 { kin / kout } else { 0.0 }]
        })),
    }
}

fn pk_kin_kout(kin: f64, kout: f64) -> PkParams {
    let mut p = PkParams::default();
    p.values[0] = kin;
    p.values[1] = kout;
    p
}

// ── Built-in absorption input-rate forcing (transit) ──────────────────
use crate::pk::absorption::{InputRateForcing, InputRateKind};

/// Single compartment that only *accumulates* the transit input (`dy = 0`),
/// so its amount at large `t` equals the total delivered mass `∫R_in = F·amt`
/// — a direct mass-balance probe of the forcing through the real integrator.
/// Transit args live at free slots: `n` @ 6, `mtt` @ 7.
fn transit_accumulator_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["depot".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7],
            frac_slot: None,
            lag_slot: None,
        }],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    }
}

fn pk_transit_vec(n: f64, mtt: f64, f: f64) -> Vec<f64> {
    let mut v = vec![0.0; crate::types::MAX_PK_PARAMS];
    v[6] = n;
    v[7] = mtt;
    v[crate::types::PK_IDX_F] = f;
    v
}

fn pk_transit_struct(n: f64, mtt: f64, f: f64) -> PkParams {
    let mut p = PkParams::default();
    p.values[6] = n;
    p.values[7] = mtt;
    p.values[crate::types::PK_IDX_F] = f;
    p
}

/// One-compartment disposition fed by a built-in `first_order(ka)` absorption
/// forcing into central: `d/dt(central) = first_order(ka) − (CL/V)·central`.
/// `ka` @ free slot 4; CL/V at the canonical slots. Used to check steady-state
/// dosing into a built-in absorption compartment against an explicit run-in (#719).
fn first_order_one_cpt_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![4],
            frac_slot: None,
            lag_slot: None,
        }],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    }
}

/// Steady-state (`SS=1`) dosing into a built-in `first_order()` absorption
/// compartment must reproduce a long explicit run-in of the same `q-II` schedule
/// (#719, gap 1). `ka` is deliberately slow relative to `II` so each dose's
/// absorption tail spills across more than one interval — that is exactly the
/// carryover the periodic-sum forward `R_in` and the pulse-train equilibration
/// trough must capture, so a bolus-only SS treatment would fail this.
#[test]
fn ss_into_first_order_absorption_matches_explicit_run_in() {
    let mut ode = first_order_one_cpt_spec();
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;

    let ka = 0.2; // t½,abs ≈ 3.5 h — absorption tail spans ~2 intervals
    let ii = 8.0;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 1.0;
    pk.values[crate::types::PK_IDX_V] = 20.0; // ke = 0.05/h ⇒ ~3× accumulation
    pk.values[4] = ka;
    pk.values[crate::types::PK_IDX_F] = 1.0;

    let obs_offsets = [0.5, 1.0, 2.0, 4.0, 6.0, 7.9];

    // SS subject: a single SS=1 record standing for the infinite pulse train.
    let ss_subj = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii)],
        obs_offsets.to_vec(),
    );
    let ss_preds = ode_predictions(&ode, &pk.values, &[], &[], &ss_subj);

    // Explicit run-in: 40 q-II doses; observe in the final interval at the same offsets.
    let n = 40usize;
    let runin_doses: Vec<DoseEvent> = (0..n)
        .map(|k| DoseEvent::new(k as f64 * ii, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let last = (n - 1) as f64 * ii;
    let runin_obs: Vec<f64> = obs_offsets.iter().map(|o| last + o).collect();
    let runin_subj = make_subject(runin_doses, runin_obs);
    let runin_preds = ode_predictions(&ode, &pk.values, &[], &[], &runin_subj);

    assert_eq!(ss_preds.len(), runin_preds.len());
    for (i, (&ss, &ri)) in ss_preds.iter().zip(&runin_preds).enumerate() {
        assert!(ss.is_finite() && ri > 0.0, "non-finite pred at offset {i}");
        let rel = (ss - ri).abs() / ri;
        assert!(
            rel < 1e-6,
            "offset {}: SS {ss:.8} vs explicit run-in {ri:.8} (rel {rel:.2e})",
            obs_offsets[i]
        );
    }
}

/// Transit absorption into an explicit **depot** state, then first-order `KA` into
/// central: `d/dt(depot) = transit(n,mtt) − KA·depot; d/dt(central) = KA·depot −
/// (CL/V)·central`. `n,mtt` @ slots 6,7; `KA` @ slot 4. Observable is central (state 1).
fn transit_one_cpt_oral_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ka = p[4];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ka * y[0];
            dy[1] = ka * y[0] - ke * y[1];
        }),
        n_states: 2,
        state_names: vec!["depot".into(), "central".into()],
        readout: OdeReadout::ObsCmt(1),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7],
            frac_slot: None,
            lag_slot: None,
        }],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    }
}

/// Steady-state into a `transit()` depot (the incomplete-gamma density kind, feeding a
/// real depot state that itself carries un-absorbed mass across intervals) must also
/// equal an explicit run-in (#719). `mtt` is a large fraction of `II` so the Erlang
/// absorption is still delivering into the depot when the next dose lands — carryover on
/// both the depot *and* the periodic R_in.
#[test]
fn ss_into_transit_depot_absorption_matches_explicit_run_in() {
    let mut ode = transit_one_cpt_oral_spec();
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;

    let ii = 6.0;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 3.0;
    pk.values[crate::types::PK_IDX_V] = 30.0; // ke = 0.1/h
    pk.values[4] = 1.0; // KA
    pk.values[6] = 3.0; // n transit compartments
    pk.values[7] = 4.0; // MTT = 4 h vs II = 6 h ⇒ tail spills into the next interval
    pk.values[crate::types::PK_IDX_F] = 1.0;

    let obs_offsets = [0.5, 1.5, 3.0, 5.0, 5.9];

    let ss_subj = make_subject(
        vec![DoseEvent::new(0.0, 50.0, 1, 0.0, true, ii)],
        obs_offsets.to_vec(),
    );
    let ss_preds = ode_predictions(&ode, &pk.values, &[], &[], &ss_subj);

    let n = 60usize;
    let runin_doses: Vec<DoseEvent> = (0..n)
        .map(|k| DoseEvent::new(k as f64 * ii, 50.0, 1, 0.0, false, 0.0))
        .collect();
    let last = (n - 1) as f64 * ii;
    let runin_obs: Vec<f64> = obs_offsets.iter().map(|o| last + o).collect();
    let runin_preds = ode_predictions(
        &ode,
        &pk.values,
        &[],
        &[],
        &make_subject(runin_doses, runin_obs),
    );

    assert_eq!(ss_preds.len(), runin_preds.len());
    for (i, (&ss, &ri)) in ss_preds.iter().zip(&runin_preds).enumerate() {
        assert!(ss.is_finite() && ri > 0.0, "non-finite pred at offset {i}");
        let rel = (ss - ri).abs() / ri;
        assert!(
            rel < 1e-6,
            "offset {}: SS {ss:.8} vs explicit run-in {ri:.8} (rel {rel:.2e})",
            obs_offsets[i]
        );
    }
}

/// An **infusion** (RATE>0) into a built-in `first_order()` absorption compartment must equal
/// a train of many tiny bolus sub-doses spread over the infusion window — the zero-order-
/// source-feeding-the-kernel semantics (#719 gap 2), integrated through the full ODE engine.
/// This validates `rate_infused` *and* the `+rate` double-count suppression end-to-end: the
/// infused dose is delivered only through the convolved `R_in_inf`, exactly as the sub-dose
/// train is delivered through the superposed bolus kernel.
#[test]
fn infusion_into_first_order_absorption_matches_subdose_train() {
    let mut ode = first_order_one_cpt_spec();
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 2.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.6; // ka
    pk.values[crate::types::PK_IDX_F] = 1.0;

    let (d, t_inf) = (100.0_f64, 3.0_f64);
    let obs = vec![0.5, 1.5, 3.0, 5.0, 9.0];

    // Infusion: RATE = D/T (rate-defined → window = amt/rate = T).
    let inf_subj = make_subject(
        vec![DoseEvent::new(0.0, d, 1, d / t_inf, false, 0.0)],
        obs.clone(),
    );
    assert!(is_real_infusion(&inf_subj.doses[0]));
    let inf_preds = ode_predictions(&ode, &pk.values, &[], &[], &inf_subj);

    // Sub-dose train: N boluses of D/N at the sub-interval midpoints of [0, T].
    let n = 300usize;
    let subdoses: Vec<DoseEvent> = (0..n)
        .map(|k| {
            let tk = (k as f64 + 0.5) * t_inf / n as f64;
            DoseEvent::new(tk, d / n as f64, 1, 0.0, false, 0.0)
        })
        .collect();
    let train_preds = ode_predictions(
        &ode,
        &pk.values,
        &[],
        &[],
        &make_subject(subdoses, obs.clone()),
    );

    assert_eq!(inf_preds.len(), train_preds.len());
    for (i, (&a, &b)) in inf_preds.iter().zip(&train_preds).enumerate() {
        assert!(a.is_finite() && b > 0.0);
        let rel = (a - b).abs() / b;
        assert!(
            rel < 1e-3,
            "obs {i} (t={}): infusion {a:.6} vs sub-dose train {b:.6} (rel {rel:.2e})",
            obs[i]
        );
    }
}

/// An infusion into a **transit** depot (the incomplete-gamma density feeding a real depot
/// state, then `KA` to central) must also equal a sub-dose train (#719 gap 2) — covering the
/// gamma-CDF `rate_infused` branch and the depot-state carryover through the ODE engine.
#[test]
fn infusion_into_transit_depot_absorption_matches_subdose_train() {
    let mut ode = transit_one_cpt_oral_spec();
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 3.0;
    pk.values[crate::types::PK_IDX_V] = 30.0;
    pk.values[4] = 1.2; // KA
    pk.values[6] = 3.0; // n transit compartments
    pk.values[7] = 1.5; // MTT
    pk.values[crate::types::PK_IDX_F] = 1.0;

    let (d, t_inf) = (50.0_f64, 2.5_f64);
    let obs = vec![0.5, 1.5, 2.5, 4.0, 8.0];
    let inf_subj = make_subject(
        vec![DoseEvent::new(0.0, d, 1, d / t_inf, false, 0.0)],
        obs.clone(),
    );
    let inf_preds = ode_predictions(&ode, &pk.values, &[], &[], &inf_subj);

    let n = 300usize;
    let subdoses: Vec<DoseEvent> = (0..n)
        .map(|k| {
            let tk = (k as f64 + 0.5) * t_inf / n as f64;
            DoseEvent::new(tk, d / n as f64, 1, 0.0, false, 0.0)
        })
        .collect();
    let train_preds = ode_predictions(
        &ode,
        &pk.values,
        &[],
        &[],
        &make_subject(subdoses, obs.clone()),
    );

    for (i, (&a, &b)) in inf_preds.iter().zip(&train_preds).enumerate() {
        assert!(a.is_finite() && b > 0.0);
        let rel = (a - b).abs() / b;
        assert!(
            rel < 1e-3,
            "obs {i} (t={}): infusion {a:.6} vs sub-dose train {b:.6} (rel {rel:.2e})",
            obs[i]
        );
    }
}

/// The infusion-into-kernel branch of `add_prepared_input_rate_forcing` must apply
/// bioavailability `F` through the **mode-aware** window reshaping (`bioavailable_infusion`,
/// #419), not merely the `F = 1` identity the sub-dose-train tests exercise. For the *same*
/// nominal `(rate, duration)` the two infusion definitions reshape differently under `F < 1`:
///   * **rate-defined** (`RATE>0`): the rate is held and the window *shrinks* to `F·(amt/rate)`
///     — `R_in_inf` = mass `F·amt` over `F·T`.
///   * **duration-defined** (`RATE=-2`): the window is *held* at `amt/rate` and the rate scales
///     to `F·rate` — `R_in_inf` = mass `F·amt` over `T`.
/// Both deliver `F·amt` total, but the differing windows give different appearance rates. A
/// swapped arm — or an `F` dropped from the mass or the window — is caught here, at the exact
/// seam (`predictions.rs`'s `d.bioavailable_infusion(f).1` / `dose_mass = F·amt`), against a
/// `rate_infused` evaluated with a *hand-literal* window (not `bioavailable_infusion` itself, so
/// the check is not circular).
#[test]
fn infusion_into_kernel_f_reshaping_is_mode_aware() {
    let ode = first_order_one_cpt_spec();
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 2.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.6; // ka
    let prepared = prepare_input_rates(&ode, &pk.values);

    let (amt, rate, f) = (100.0_f64, 25.0_f64, 0.6_f64);
    // The F = 1 infusion length, amt/rate = 4 h.
    let nominal_window = amt / rate;
    // tad = 3 sits *inside* the held duration-defined window (4 h) but *past* the shrunk
    // rate-defined window (F·4 = 2.4 h), so the two arms must give genuinely different rates.
    let tad = 3.0_f64;

    let rate_defined = DoseEvent::new(0.0, amt, 1, rate, false, 0.0); // InfusionDef::RateDefined
    let mut dur_defined = DoseEvent::new(0.0, amt, 1, rate, false, 0.0);
    dur_defined.infusion_def = crate::types::InfusionDef::DurationDefined;

    let lags = [0.0_f64];
    let f_bio = [f];
    let mut dy_rate = [0.0_f64];
    add_prepared_input_rate_forcing(
        &ode,
        &prepared,
        &pk.values,
        std::slice::from_ref(&rate_defined),
        &lags,
        &f_bio,
        f64::NEG_INFINITY,
        tad,
        &mut dy_rate,
    );
    let mut dy_dur = [0.0_f64];
    add_prepared_input_rate_forcing(
        &ode,
        &prepared,
        &pk.values,
        std::slice::from_ref(&dur_defined),
        &lags,
        &f_bio,
        f64::NEG_INFINITY,
        tad,
        &mut dy_dur,
    );

    // Rate-defined: mass F·amt = 60 over the *shrunk* window F·nominal = 2.4 h.
    let want_rate = prepared[0].rate_infused(tad, f * amt, f * nominal_window);
    // Duration-defined: mass F·amt = 60 over the *held* window nominal = 4 h.
    let want_dur = prepared[0].rate_infused(tad, f * amt, nominal_window);
    assert_relative_eq!(dy_rate[0], want_rate, max_relative = 1e-12);
    assert_relative_eq!(dy_dur[0], want_dur, max_relative = 1e-12);
    assert!(
        (dy_rate[0] - dy_dur[0]).abs() > 1e-6,
        "the two infusion definitions must reshape F into distinct windows: \
             rate-defined {:.6} vs duration-defined {:.6}",
        dy_rate[0],
        dy_dur[0]
    );
}

/// A **rate-defined** infusion (`RATE>0`) under `F < 1` into a `first_order()` kernel, through
/// the full ODE engine (#719 gap 2, #419). `F < 1` holds the rate and *shrinks* the window to
/// `F·(amt/rate)`, delivering the bioavailable mass `F·amt` over `[0, F·T]`. The oracle sub-dose
/// train spreads N nominal `amt/N` boluses (the engine applies the *same* `F`) over that shrunk
/// window — so it matches only if the rate-defined arm is selected at `predictions.rs`'s
/// `bioavailable_infusion(f).1`. Complements the `F = 1` train test above.
#[test]
fn infusion_into_first_order_absorption_with_f_below_one_matches_subdose_train() {
    let mut ode = first_order_one_cpt_spec();
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 2.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.6; // ka
    let f = 0.6_f64;
    pk.values[crate::types::PK_IDX_F] = f;

    let (d, t_inf) = (100.0_f64, 3.0_f64); // rate-defined: RATE = d/t_inf, nominal window t_inf
    let obs = vec![0.5, 1.5, 3.0, 5.0, 9.0];
    let inf_subj = make_subject(
        vec![DoseEvent::new(0.0, d, 1, d / t_inf, false, 0.0)],
        obs.clone(),
    );
    let inf_preds = ode_predictions(&ode, &pk.values, &[], &[], &inf_subj);

    // Sub-dose train over the *shrunk* window [0, F·t_inf].
    let window = f * t_inf;
    let n = 300usize;
    let subdoses: Vec<DoseEvent> = (0..n)
        .map(|k| {
            let tk = (k as f64 + 0.5) * window / n as f64;
            DoseEvent::new(tk, d / n as f64, 1, 0.0, false, 0.0)
        })
        .collect();
    let train_preds = ode_predictions(
        &ode,
        &pk.values,
        &[],
        &[],
        &make_subject(subdoses, obs.clone()),
    );

    assert_eq!(inf_preds.len(), train_preds.len());
    for (i, (&a, &b)) in inf_preds.iter().zip(&train_preds).enumerate() {
        assert!(a.is_finite() && b > 0.0);
        let rel = (a - b).abs() / b;
        assert!(
            rel < 1e-3,
            "obs {i} (t={}): F<1 infusion {a:.6} vs sub-dose train {b:.6} (rel {rel:.2e})",
            obs[i]
        );
    }
}

/// A **duration-defined** infusion (`RATE=-2` → `D{cmt}`) into a `first_order()` kernel, through
/// the full ODE engine (#719 gap 2, #419). Unlike the rate-defined case, `F < 1` *holds* the
/// window at `amt/rate` and scales the rate to `F·rate`, delivering `F·amt` over the full
/// `[0, T]`. The dose reaches the forcing seam already resolved (`rate_mode = Fixed`) but with
/// `infusion_def = DurationDefined` persisted (the tag survives `resolve_rate`, #419), so the
/// seam takes the held-window arm. The oracle train therefore spreads over the *unshrunk*
/// window — the opposite reshaping from the rate-defined test, which pins the arm selection.
#[test]
fn duration_defined_infusion_into_first_order_absorption_matches_subdose_train() {
    let mut ode = first_order_one_cpt_spec();
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 2.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.6; // ka
    let f = 0.6_f64;
    pk.values[crate::types::PK_IDX_F] = f;

    let (d, t_inf) = (100.0_f64, 3.0_f64);
    let obs = vec![0.5, 1.5, 3.0, 5.0, 9.0];
    // Resolved duration-defined infusion: concrete rate/duration (Fixed), DurationDefined tag.
    let mut dur_defined = DoseEvent::new(0.0, d, 1, d / t_inf, false, 0.0);
    dur_defined.infusion_def = crate::types::InfusionDef::DurationDefined;
    let inf_subj = make_subject(vec![dur_defined], obs.clone());
    let inf_preds = ode_predictions(&ode, &pk.values, &[], &[], &inf_subj);

    // Sub-dose train over the *held* window [0, t_inf] (F scales the rate, not the window).
    let n = 300usize;
    let subdoses: Vec<DoseEvent> = (0..n)
        .map(|k| {
            let tk = (k as f64 + 0.5) * t_inf / n as f64;
            DoseEvent::new(tk, d / n as f64, 1, 0.0, false, 0.0)
        })
        .collect();
    let train_preds = ode_predictions(
        &ode,
        &pk.values,
        &[],
        &[],
        &make_subject(subdoses, obs.clone()),
    );

    assert_eq!(inf_preds.len(), train_preds.len());
    for (i, (&a, &b)) in inf_preds.iter().zip(&train_preds).enumerate() {
        assert!(a.is_finite() && b > 0.0);
        let rel = (a - b).abs() / b;
        assert!(
            rel < 1e-3,
            "obs {i} (t={}): duration-defined infusion {a:.6} vs sub-dose train {b:.6} \
                 (rel {rel:.2e})",
            obs[i]
        );
    }
}

// ── #1187: `gated_infusions` must drop what `active_infusions` drops ─────────
//
// The `InfusionInput::Gated` arm feeds the same `wrap_rhs_with_forcings` seam as the
// `Spanning` arm, which adds the resolved `+rate` *alongside* the convolved `R_in_inf`.
// Without the input-rate suppression the mass is delivered twice on every gated engine
// (`ode_predictions_with_states`, `ode_dense_solve_states`, `ode_solve_until_chz_threshold`)
// while `ode_predictions` — which resolves through `active_infusions` — stays correct.

/// The helper itself: an infusion into a forcing compartment is dropped, one into any
/// other compartment is kept, and an empty forcing slice keeps everything (`ode/ekf.rs`
/// passes `&[]` because it never applies the forcing).
#[test]
fn gated_infusions_suppresses_infusion_into_input_rate_cmt() {
    // Two infusions: CMT 1 (≡ state 0, the forcing compartment) and CMT 2 (≡ state 1).
    let doses = vec![
        DoseEvent::new(0.0, 0.0, 1, 4.0, false, 0.0),
        DoseEvent::new(0.0, 0.0, 2, 7.0, false, 0.0),
    ];
    let f_bio = vec![1.0, 1.0];
    let active = vec![(0usize, 1.0, 3.0), (1usize, 1.0, 3.0)];
    let forcing = transit_one_cpt_oral_spec().input_rate; // forcing on state 0

    // No forcing applied → both plain `+rate` injections are kept.
    assert_eq!(
        gated_infusions(&[], &active, &doses, &f_bio, 2),
        vec![(0usize, 4.0, 1.0, 3.0), (1usize, 7.0, 1.0, 3.0)]
    );
    // Forcing on state 0 → only the CMT-2 infusion survives; the CMT-1 mass arrives
    // through `R_in_inf` instead.
    assert_eq!(
        gated_infusions(&forcing, &active, &doses, &f_bio, 2),
        vec![(1usize, 7.0, 1.0, 3.0)]
    );
}

/// Mass balance, with no integrator in the oracle: `transit_accumulator_spec` has
/// `dy = 0`, so its single state *is* `∫R_in dt`, and `R_in_inf` is mass-exact
/// (`∫₀^∞ R_in_inf = F·amt`, `PreparedInputRate::rate_infused`). Every engine must
/// therefore read `F·amt` for an infusion, exactly as it already does for a bolus.
/// Before the fix the two gated engines read `2·F·amt`.
#[test]
fn infusion_into_input_rate_cmt_is_mass_exact_on_every_engine() {
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let obs = vec![40.0]; // far past the kernel's tail

    // A duration-defined (RATE=-2) dose is *not* redundant with the rate-defined one:
    // under `F != 1` the two modes differ (#419 — `F` scales the window in one and the
    // rate in the other), and at `F = 1` they coincide, which is why `F` is free here.
    let mut dur_defined = DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0);
    dur_defined.infusion_def = crate::types::InfusionDef::DurationDefined;

    for (tag, doses, expected) in [
        (
            "bolus",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            100.0,
        ),
        (
            "infusion RATE>0",
            vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)],
            100.0,
        ),
        ("infusion RATE=-2", vec![dur_defined.clone()], 100.0),
        (
            // Non-degenerate: the second window opens while the first kernel is still
            // delivering, so the suppression has to hold per dose, not just for the first.
            "two infusions",
            vec![
                DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0),
                DoseEvent::new(2.0, 100.0, 1, 25.0, false, 0.0),
            ],
            200.0,
        ),
        (
            // Mixed: a bolus and an infusion into the same forcing compartment.
            "bolus + infusion",
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(3.0, 100.0, 1, 25.0, false, 0.0),
            ],
            200.0,
        ),
    ] {
        let subj = make_subject(doses, obs.clone());
        let plain = ode_predictions(&ode, &pk, &[], &[], &subj);
        let (with_states_pred, with_states) =
            ode_predictions_with_states(&ode, &pk, &[], &[], &subj);
        let dense = ode_dense_solve_states(&ode, &pk, &[], &[], &subj, &obs);
        for (engine, got) in [
            ("ode_predictions", plain[0]),
            ("ode_predictions_with_states", with_states_pred[0]),
            ("with_states[0][0]", with_states[0][0]),
            ("ode_dense_solve_states", dense[0][0]),
        ] {
            // One bound, not two: a `got < 1.5·expected` "double-count?" guard alongside
            // this would be dead code — 5e-3 relative already excludes 2·expected.
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 5e-3,
                "{tag}/{engine}: delivered {got:.6}, expected {expected} \
                 (rel {rel:.2e}; ~2x means the plain `+rate` was not suppressed)"
            );
        }
    }
}

/// The readout, not just the delivered mass: on real PK models every gated engine must
/// equal an explicit **sub-dose train** — N sub-doses over the same window, which is the
/// rectangle `rate_infused` claims to convolve with the kernel.
///
/// Be precise about what that does and does not pin. The train's sub-doses are *not*
/// ordinary boluses: a bolus into an input-rate compartment is suppressed
/// (`apply_prescheduled_boluses_at`) and delivered through the same
/// `add_prepared_input_rate_forcing` machinery, via `prep.rate` instead of
/// `prep.rate_infused`. So this pins **`rate_infused` against `rate`** — the convolution
/// against the impulse response it is built from — not the kernel itself, and not the
/// engines against anything outside `PreparedInputRate`. The kernel-independent checks
/// are the closed-form mass balance above and the NONMEM anchors in
/// `tests/infusion_absorption_nonmem_anchor.rs`.
///
/// It is nonetheless a sharp test of *this* defect: the stray `+rate` is injected by
/// `wrap_rhs_with_forcings`, downstream of everything the two paths share.
/// `ode_predictions` rides along as the already-anchored control — it resolves through
/// `active_infusions` and was correct throughout.
///
/// Before the fix the gated engines were 8.3× (`first_order`) and 214× (`transit`) the
/// train at the first observation: the stray `+rate` lands in the compartment *directly*,
/// bypassing the kernel, so the divergence is worst early and settles near 1.86× once
/// both copies have distributed.
#[test]
fn infusion_into_absorption_matches_subdose_train_on_gated_engines() {
    /// `doses` are infusions into CMT 1, each paired with the *held* window length its
    /// train must cover. Several are allowed on purpose: the gated engines assemble their
    /// timeline with `build_segment_break_times`, a different builder from the guarded
    /// engines' `collect_dose_break_times` — the pair that diverged in #1171 — and only a
    /// second window landing mid-kernel exercises it. (The mass-balance arms cannot: their
    /// spec is `dy = 0`, so the state is `∫R_in dt` and is invariant to *when* the mass
    /// arrives, hence to break placement.)
    fn check(
        tag: &str,
        ode: &OdeSpec,
        pk: &[f64],
        doses: Vec<(DoseEvent, f64)>,
        obs: &[f64],
        read: usize,
    ) {
        let n = 300usize;
        let train: Vec<DoseEvent> = doses
            .iter()
            .flat_map(|(d, t_inf)| {
                // N boluses of amt/N at the sub-interval midpoints of this dose's window.
                let (amt, t0, w) = (d.amt, d.time, *t_inf);
                (0..n).map(move |k| {
                    let tk = t0 + (k as f64 + 0.5) * w / n as f64;
                    DoseEvent::new(tk, amt / n as f64, 1, 0.0, false, 0.0)
                })
            })
            .collect();
        let subj = make_subject(doses.into_iter().map(|(d, _)| d).collect(), obs.to_vec());
        let reference = ode_predictions(ode, pk, &[], &[], &make_subject(train, obs.to_vec()));

        let plain = ode_predictions(ode, pk, &[], &[], &subj);
        let (ws_pred, ws) = ode_predictions_with_states(ode, pk, &[], &[], &subj);
        let dense = ode_dense_solve_states(ode, pk, &[], &[], &subj, obs);

        for i in 0..obs.len() {
            assert!(reference[i] > 0.0, "{tag}: degenerate reference at obs {i}");
            for (engine, got) in [
                ("ode_predictions", plain[i]),
                ("ode_predictions_with_states", ws_pred[i]),
                ("with_states.state", ws[i][read]),
                ("ode_dense_solve_states", dense[i][read]),
            ] {
                let rel = (got - reference[i]).abs() / reference[i];
                assert!(
                    rel < 1e-3,
                    "{tag}/{engine} obs {i} (t={}): {got:.6} vs sub-dose train {:.6} \
                     (rel {rel:.2e})",
                    obs[i],
                    reference[i]
                );
            }
        }
    }

    // `first_order` kernel, rate-defined infusion. Readout ≡ state 0.
    let mut fo = first_order_one_cpt_spec();
    fo.solver_opts.reltol = 1e-11;
    fo.solver_opts.abstol = 1e-11;
    let mut pk_fo = PkParams::default();
    pk_fo.values[crate::types::PK_IDX_CL] = 2.0;
    pk_fo.values[crate::types::PK_IDX_V] = 20.0;
    pk_fo.values[4] = 0.6; // ka
    pk_fo.values[crate::types::PK_IDX_F] = 1.0;
    let obs_fo = vec![0.5, 1.5, 3.0, 5.0, 9.0];
    check(
        "first_order RATE>0",
        &fo,
        &pk_fo.values,
        vec![(DoseEvent::new(0.0, 100.0, 1, 100.0 / 3.0, false, 0.0), 3.0)],
        &obs_fo,
        0,
    );

    // Duration-defined (RATE=-2) at **F != 1**: at F = 1 the two `InfusionDef` modes are
    // numerically identical, so this arm only discriminates away from unity (#419 — here
    // `F` scales the rate and the window is held, which is what the train mirrors).
    let mut pk_dd = pk_fo;
    pk_dd.values[crate::types::PK_IDX_F] = 0.6;
    let mut dur_defined = DoseEvent::new(0.0, 100.0, 1, 100.0 / 3.0, false, 0.0);
    dur_defined.infusion_def = crate::types::InfusionDef::DurationDefined;
    check(
        "first_order RATE=-2 (F=0.6)",
        &fo,
        &pk_dd.values,
        vec![(dur_defined, 3.0)],
        &obs_fo,
        0,
    );

    // **Multi-dose trajectory** (#1171 class). The second infusion opens at t = 2 while the
    // first window (`[0, 3]`) is still delivering and the kernel still has mass in flight, so
    // the two windows overlap and the timeline needs breaks at 0, 2, 3 and 5. This is the arm
    // that exercises `build_segment_break_times` against `collect_dose_break_times`; the
    // single-dose arms above and the `dy = 0` mass-balance arms both cannot.
    check(
        "first_order RATE>0, two overlapping windows",
        &fo,
        &pk_fo.values,
        vec![
            (DoseEvent::new(0.0, 100.0, 1, 100.0 / 3.0, false, 0.0), 3.0),
            (DoseEvent::new(2.0, 100.0, 1, 100.0 / 3.0, false, 0.0), 3.0),
        ],
        &[0.5, 2.5, 3.5, 5.5, 9.0, 14.0],
        0,
    );

    // `transit` kernel into a real depot state, readout ≡ state 1 (central) — a second
    // kernel family, and a forcing compartment that is not the observed one.
    let mut tr = transit_one_cpt_oral_spec();
    tr.solver_opts.reltol = 1e-11;
    tr.solver_opts.abstol = 1e-11;
    let mut pk_tr = PkParams::default();
    pk_tr.values[crate::types::PK_IDX_CL] = 3.0;
    pk_tr.values[crate::types::PK_IDX_V] = 30.0;
    pk_tr.values[4] = 1.2; // KA
    pk_tr.values[6] = 3.0; // n transit compartments
    pk_tr.values[7] = 1.5; // MTT
    pk_tr.values[crate::types::PK_IDX_F] = 1.0;
    check(
        "transit RATE>0",
        &tr,
        &pk_tr.values,
        vec![(DoseEvent::new(0.0, 50.0, 1, 20.0, false, 0.0), 2.5)],
        &[0.5, 1.5, 2.5, 4.0, 8.0],
        1,
    );

    // Multi-dose on the transit kernel too — a forcing compartment that is not the observed
    // one, so a break-placement error has to propagate through `KA` to show up.
    check(
        "transit RATE>0, two overlapping windows",
        &tr,
        &pk_tr.values,
        vec![
            (DoseEvent::new(0.0, 50.0, 1, 20.0, false, 0.0), 2.5),
            (DoseEvent::new(1.5, 50.0, 1, 20.0, false, 0.0), 2.5),
        ],
        &[0.5, 2.0, 3.0, 5.0, 9.0],
        1,
    );
}

/// The other side of the guard, and the arm that fails if it is written too wide: an
/// infusion into a compartment that is **not** an input-rate target keeps its plain
/// `+rate` on every engine. A guard spelled `!input_rate.is_empty()` — or one asking
/// [`input_rate_consumes_cmt`] about the spec rather than the applied forcing slice —
/// passes every other test here and deletes this dose's mass entirely.
#[test]
fn infusion_into_non_input_rate_cmt_is_untouched_on_gated_engines() {
    let mut ode = transit_one_cpt_oral_spec(); // forcing on state 0 (depot) only
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 3.0;
    pk.values[crate::types::PK_IDX_V] = 30.0;
    pk.values[4] = 1.2;
    pk.values[6] = 3.0;
    pk.values[7] = 1.5;
    pk.values[crate::types::PK_IDX_F] = 1.0;

    let obs = vec![0.5, 1.5, 2.5, 4.0, 8.0];
    // CMT 2 ≡ state 1 (central): a plain infusion, no kernel involved.
    let subj = make_subject(
        vec![DoseEvent::new(0.0, 50.0, 2, 20.0, false, 0.0)],
        obs.clone(),
    );
    let plain = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    let (ws_pred, ws) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    let dense = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subj, &obs);

    for i in 0..obs.len() {
        assert!(plain[i] > 0.0, "obs {i}: reference is degenerate (zero)");
        for (engine, got) in [
            ("ode_predictions_with_states", ws_pred[i]),
            ("with_states.state", ws[i][1]),
            ("ode_dense_solve_states", dense[i][1]),
        ] {
            assert_relative_eq!(got, plain[i], max_relative = 1e-9);
        }
    }
}

/// The sixth surface: `simulate()`'s event-time search
/// ([`ode_solve_until_chz_threshold`]) has its own `wrap_rhs_with_forcings` call site and
/// so resolved infusions through the unguarded `gated_infusions` too — a doubled exposure
/// inflates the hazard and the simulated event fires early.
///
/// **The obvious oracle here is vacuous and is deliberately not used.** Comparing the
/// returned crossing against [`ode_dense_solve_states`]' CHZ column compares two *gated*
/// engines: under the defect both inflate identically and they agree. The reference has to
/// come from the guarded side, so the crossing time is fed back through
/// [`ode_predictions`] — which resolves via `active_infusions` — and the cumulative hazard
/// it reports there must be the threshold that was asked for.
#[cfg(feature = "survival")]
#[test]
fn simulated_event_time_is_consistent_with_the_guarded_hazard() {
    // state 0: central, fed by the `first_order` kernel. state 1: the CHZ accumulator.
    // Readout is the CHZ state, so `ode_predictions` reads H(t) off the guarded path.
    let make_spec = || OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let (cl, v) = (p[crate::types::PK_IDX_CL], p[crate::types::PK_IDX_V]);
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            let (h0, beta) = (p[6], p[7]);
            dy[0] = -ke * y[0];
            dy[1] = h0 * (beta * y[0] / v).exp();
        }),
        n_states: 2,
        state_names: vec!["central".into(), "chz".into()],
        readout: OdeReadout::ObsCmt(1),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions {
            reltol: 1e-11,
            abstol: 1e-11,
            ..OdeSolverOptions::default()
        },
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![4],
            frac_slot: None,
            lag_slot: None,
        }],
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    };

    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 2.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.6; // ka
    pk.values[6] = 0.02; // H0
    pk.values[7] = 0.5; // BETA
    pk.values[crate::types::PK_IDX_F] = 1.0;

    // RATE>0 into CMT 1 — the input-rate compartment, i.e. #719 gap 2.
    let dose = DoseEvent::new(0.0, 100.0, 1, 100.0 / 3.0, false, 0.0);
    let threshold = 0.3_f64;
    let horizon = 48.0_f64;

    let outcome = ode_solve_until_chz_threshold(
        &make_spec(),
        &pk.values,
        &make_subject(vec![dose.clone()], vec![]),
        1,
        threshold,
        horizon,
    );
    let t_cross = match outcome {
        ThresholdOutcome::Crossed(t) => t,
        other => panic!("hazard must reach {threshold} before t={horizon}; got {other:?}"),
    };
    assert!(
        t_cross > 0.0 && t_cross < horizon,
        "crossing {t_cross} must be interior, or the arm is degenerate"
    );

    // The guarded reference: H(t_cross) read through `ode_predictions`.
    let h_at_cross = ode_predictions(
        &make_spec(),
        &pk.values,
        &[],
        &[],
        &make_subject(vec![dose], vec![t_cross]),
    )[0];
    let rel = (h_at_cross - threshold).abs() / threshold;
    assert!(
        rel < 1e-4,
        "simulate()'s crossing t={t_cross:.6} carries guarded H={h_at_cross:.6}, not the \
         requested threshold {threshold} (rel {rel:.2e}) — the event-time search and the \
         guarded hazard disagree about how much drug was delivered"
    );
}

/// The fifth surface, and the one with a symptom of its own: on a TV-covariate / reset
/// subject, `ode_predictions_event_driven_with_states` takes its `ipred` from the
/// event-driven walk (which resolves through the guarded `active_infusions`) but its
/// **states** from `ode_dense_solve_states` (gated). Under the defect the same sdtab row
/// therefore carried a correct `IPRED` beside a compartment column roughly twice as large —
/// internally inconsistent, and neither column announcing a problem.
///
/// This pins the two halves against each other. The spec's readout is the forcing
/// compartment itself, so `ipred[i]` and `states[i][0]` are the *same quantity* computed by
/// the two different engines, and any disagreement is the split.
#[test]
fn event_driven_with_states_ipred_and_states_agree_for_infusion_into_absorption() {
    let mut ode = first_order_one_cpt_spec(); // 1 state, readout ≡ state 0
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-11;

    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 2.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.6; // ka
    pk.values[crate::types::PK_IDX_F] = 1.0;

    let obs = vec![0.5, 1.5, 3.0, 5.0, 9.0];
    let subj = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 100.0 / 3.0, false, 0.0)],
        obs.clone(),
    );
    assert!(
        subj.doses[0].is_infusion(),
        "fixture must carry a real infusion or it tests nothing"
    );

    // Per-event PK snapshots, uniform here — this entry point is reached for TV-covariate
    // and reset subjects, but the engine split it exposes does not depend on them varying.
    let at = |n: usize| vec![pk; n];
    let (ipred, states) = ode_predictions_event_driven_with_states(
        &ode,
        &subj,
        &[],
        &[],
        &at(subj.doses.len()),
        &at(obs.len()),
        &[],
        &[],
    );

    for i in 0..obs.len() {
        assert!(
            ipred[i] > 0.0 && states[i][0].is_finite(),
            "obs {i}: degenerate ({}, {})",
            ipred[i],
            states[i][0]
        );
        let rel = (states[i][0] - ipred[i]).abs() / ipred[i];
        assert!(
            rel < 1e-6,
            "obs {i} (t={}): the event-driven ipred is {:.6} but the dense state column is \
             {:.6} (rel {rel:.2e}) — the same sdtab row would report a correct IPRED beside \
             a doubled compartment amount",
            obs[i],
            ipred[i],
            states[i][0]
        );
    }
}

#[test]
fn input_rate_consumes_cmt_matches_forcing_compartment() {
    let ode = transit_accumulator_spec(); // forcing on state 0 ≡ 1-based CMT 1
    assert!(input_rate_consumes_cmt(&ode, 1));
    assert!(!input_rate_consumes_cmt(&ode, 2));
    // A spec with no input-rate term never consumes a dose.
    assert!(!input_rate_consumes_cmt(&one_cpt_ode_spec(), 1));
}

/// Single accumulator compartment (`dy = 0`) fed by a `zero_order(dur)`
/// forcing, `dur` at free slot 4 — the zero-order analogue of
/// `transit_accumulator_spec`, so its amount at large `t` equals the delivered
/// mass `∫R_in = F·amt` and at an interior `t < dur` equals the linear partial
/// `(F·amt/dur)·t` (a direct probe that the cutoff break is placed correctly).
fn zero_order_accumulator_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["depot".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::ZeroOrder,
            arg_slots: vec![4],
            frac_slot: None,
            lag_slot: None,
        }],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    }
}

fn pk_zero_order_vec(dur: f64, f: f64) -> Vec<f64> {
    let mut v = vec![0.0; crate::types::MAX_PK_PARAMS];
    v[4] = dur;
    v[crate::types::PK_IDX_F] = f;
    v
}

/// Build the per-dose windows from a single subject snapshot (the dense-path
/// shape) — the test analogue of the `|_, d| zero_order_dur_and_frac_for_dose(...)`
/// closure the production callers pass.
fn zo_windows_for(
    ode: &OdeSpec,
    doses: &[DoseEvent],
    lags: &[f64],
    pk: &[f64],
) -> Vec<ZeroOrderWindow> {
    let f_bio: Vec<f64> = doses.iter().map(|_| 1.0).collect();
    zero_order_windows(doses, lags, &f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk)
    })
}

#[test]
fn zero_order_window_edges_rate_and_cutoff_break() {
    // A dose at t=2 into the zero-order compartment, lag 0.5, dur 4 ⇒ a window
    // [2.5, 6.5] with rate F·amt/dur = 100/4 = 25, and breaks at **both** edges. A
    // dose into a *different* compartment, and a zero-amount dose, contribute no
    // window (so no break).
    //
    // Both edges because `active_zero_order_inputs`' containment test is two-sided
    // (#1171): pinning only the cutoff is what let two builders segment a lagged
    // window's start away and drop the rate entirely.
    let ode = zero_order_accumulator_spec(); // cmt 0 ≡ 1-based CMT 1
    let pk = pk_zero_order_vec(4.0, 1.0);
    let doses = vec![
        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0), // feeds R_in
        DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0), // other cmt → no window
        DoseEvent::new(1.0, 0.0, 1, 0.0, false, 0.0),   // zero amt → no window
    ];
    let windows = zo_windows_for(&ode, &doses, &[0.5, 0.0, 0.0], &pk);
    assert_eq!(windows, vec![(0, 25.0, 2.5, 6.5)]);

    let mut breaks = Vec::new();
    push_zero_order_break_times(&mut breaks, &windows);
    assert_eq!(breaks, vec![2.5, 6.5]);
}

#[test]
fn push_route_lag_break_times_adds_route_onsets() {
    // Two first-order routes on cmt 0: one with a per-route lag (slot 6), one
    // without. A break is added at `d.time + lag_cmt + lag_route` for the LAGGED
    // route only over a positive-amount dose feeding cmt 0; the unlagged route adds
    // nothing, and doses into another compartment / with zero amount are skipped.
    let mk = |lag_slot| InputRateForcing {
        cmt: 0,
        kind: InputRateKind::FirstOrder,
        arg_slots: vec![4],
        frac_slot: None,
        lag_slot,
    };
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![mk(Some(6)), mk(None)],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    };
    let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
    params[6] = 1.5; // per-route lag
    let doses = vec![
        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0), // cmt 0, amt>0 → onset break
        DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0), // other cmt → skipped
        DoseEvent::new(5.0, 0.0, 1, 0.0, false, 0.0),   // zero amt → skipped
    ];
    let subj = make_subject(doses, vec![]);
    let mut breaks = Vec::new();
    push_route_lag_break_times(&mut breaks, &ode, &subj, &[0.5, 0.0, 0.0], |f| {
        f.route_lag(&params)
    });
    // Only the lagged forcing × the valid dose: 2.0 + 0.5 (cmt lag) + 1.5 (route lag).
    assert_eq!(breaks, vec![4.0]);
}

#[test]
fn active_zero_order_includes_only_fully_contained_segments() {
    // Window [2.5, 6.5], rate 25. A segment strictly inside is active; a segment
    // straddling the cutoff (right end past w_end) is excluded — the
    // full-containment rule that makes the post-cutoff mass exact. A reset_floor
    // after the window start turns it off.
    let windows: Vec<ZeroOrderWindow> = vec![(0, 25.0, 2.5, 6.5)];
    assert_eq!(
        active_zero_order_inputs(&windows, 3.0, 5.0, f64::NEG_INFINITY),
        vec![(0, 25.0)]
    );
    // [5, 7] ends past w_end=6.5 ⇒ not fully contained ⇒ excluded.
    assert!(active_zero_order_inputs(&windows, 5.0, 7.0, f64::NEG_INFINITY).is_empty());
    // [1, 2] precedes the window start ⇒ excluded.
    assert!(active_zero_order_inputs(&windows, 1.0, 2.0, f64::NEG_INFINITY).is_empty());
    // reset_floor past the window start (e.g. 3.0) turns the window off.
    assert!(active_zero_order_inputs(&windows, 3.0, 5.0, 3.0).is_empty());
}

#[test]
fn smooth_forcing_contributes_no_zero_order_window() {
    // transit/igd/weibull are smooth (no cutoff) — they yield no zero-order
    // window, so they keep their existing break structure and pointwise forcing.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(zo_windows_for(&ode, &doses, &[0.0], &pk).is_empty());
}

#[test]
fn zero_order_forcing_delivers_full_dose_mass() {
    // After the window closes (`t > dur`) the accumulator holds ∫R_in = F·amt
    // = 100 — not 200 (bolus double-count) and not 0 (no forcing). The cutoff
    // break stops the input cleanly at `dur`, so the plateau is exact.
    let ode = zero_order_accumulator_spec();
    let pk = pk_zero_order_vec(4.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![20.0]);
    let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
    assert_relative_eq!(preds[0], 100.0, max_relative = 1e-6);
}

#[test]
fn zero_order_partial_window_is_linear() {
    // Inside the window the accumulated mass is the rectangle's running area
    // `(F·amt/dur)·t`: at t = dur/2 = 2 it is exactly half the dose. This only
    // holds if the constant rate is delivered over `(0, dur]` and the cutoff
    // break does not truncate the window early.
    let ode = zero_order_accumulator_spec();
    let pk = pk_zero_order_vec(4.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![2.0]);
    let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
    assert_relative_eq!(preds[0], 50.0, max_relative = 1e-6);
}

#[test]
fn transit_forcing_delivers_full_dose_mass() {
    // The accumulator depot should hold ∫R_in = F·amt = 100 once absorption
    // is complete — NOT 200 (bolus would double-count) and NOT 0 (no forcing).
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![40.0]);
    let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
    assert_relative_eq!(preds[0], 100.0, max_relative = 5e-3);
}

#[test]
fn transit_dose_does_not_enter_as_bolus() {
    // An observation exactly at the dose time reads ~0: the transit dose is
    // delivered as R_in over time, never as an instantaneous bolus jump. (A
    // trailing obs keeps the break-time loop non-empty.) The late obs then
    // confirms the full mass still arrives.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![0.0, 40.0]);
    let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
    assert!(preds[0].abs() < 1e-9, "bolus not suppressed: {}", preds[0]);
    assert_relative_eq!(preds[1], 100.0, max_relative = 5e-3);
}

#[test]
fn transit_forcing_scales_with_bioavailability() {
    // F = 0.4 ⇒ delivered mass = 0.4·100 = 40.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 0.4);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![40.0]);
    let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
    assert_relative_eq!(preds[0], 40.0, max_relative = 5e-3);
}

#[test]
fn transit_forcing_superposes_over_doses() {
    // Two doses (100 @ t=0, 50 @ t=10) superpose: ∫R_in = F·(100+50) = 150.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 50.0, 1, 0.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![60.0]);
    let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
    assert_relative_eq!(preds[0], 150.0, max_relative = 5e-3);
}

#[test]
fn transit_forcing_respects_reset_floor() {
    // Event-driven path: an EVID=3 reset at t=1 zeros the depot AND turns off
    // the pre-reset dose's input rate. With no post-reset dose, the
    // accumulator stays at 0 — the t=0 dose's R_in must not resume.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_struct(3.0, 2.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let mut subj = make_subject(doses, vec![40.0]);
    subj.reset_times = vec![1.0];
    let dose_pk = vec![pk; subj.doses.len()];
    let obs_pk = vec![pk; subj.obs_times.len()];
    let reset_pk = vec![pk; subj.reset_times.len()];
    let preds =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &dose_pk, &obs_pk, &[], &reset_pk);
    assert!(
        preds[0].abs() < 1e-6,
        "pre-reset dose R_in leaked past the reset: got {}",
        preds[0]
    );
}

#[test]
fn transit_forcing_applied_in_with_states_path() {
    // The per-compartment states path (`ode_predictions_with_states`, used for
    // derived-output state extraction) must inject the transit forcing too —
    // the accumulator state holds ∫R_in = F·amt = 100.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![40.0]);
    let (preds, states) = ode_predictions_with_states(&ode, &pk, &[], &[], &subj);
    assert_relative_eq!(preds[0], 100.0, max_relative = 5e-3);
    assert_relative_eq!(states[0][0], 100.0, max_relative = 5e-3);
}

#[test]
fn transit_forcing_in_dense_solve_states_skips_other_cmt_dose() {
    // `ode_dense_solve_states` applies the forcing; a dose targeting a
    // *non-forcing* compartment is skipped by the superposition loop. State 0
    // (the forcing cmt ≡ CMT 1) holds only the CMT-1 dose's mass — not the
    // CMT-2 dose, which never feeds R_in.
    let ode = transit_accumulator_spec();
    let pk = pk_transit_vec(3.0, 2.0, 1.0);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // CMT 1: feeds R_in
        DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0),  // CMT 2: not the forcing cmt
    ];
    let subj = make_subject(doses, vec![40.0]);
    let states = ode_dense_solve_states(&ode, &pk, &[], &[], &subj, &[40.0]);
    assert_relative_eq!(states[0][0], 100.0, max_relative = 5e-3);
}

// ── Forcing-seam helpers (#353): the single RHS-wrapper seam + per-segment
//    prepare() hoist shared by all four ODE integration paths. ────────────

#[test]
fn prepare_input_rates_parallel_to_forcings_and_empty_without_them() {
    // Parallel to `ode.input_rate`; empty (non-allocating) when the model has
    // no built-in input-rate forcing.
    let ode = transit_accumulator_spec();
    let params = pk_transit_vec(3.0, 2.0, 1.0);
    let prepared = prepare_input_rates(&ode, &params);
    assert_eq!(prepared.len(), 1);
    // The hoisted constant must match a direct `prepare` on the same params —
    // the invariant that keeps the #7 hoist from drifting from the per-eval form.
    assert_eq!(
        prepared[0].rate(2.5, 100.0),
        ode.input_rate[0].prepare(&params).rate(2.5, 100.0)
    );
    assert!(prepare_input_rates(&one_cpt_ode_spec(), &params).is_empty());
}

#[test]
fn gated_infusions_resolves_rate_and_drops_unaddressable() {
    // (dose_idx, t_start, t_end) -> (cmt_idx, rate_eff, t_start, t_end) with
    // the mode-aware bioavailability rate (#419): a rate-defined infusion
    // holds its rate (F scales the duration, carried by the window), while a
    // duration-defined infusion (RATE=-2) gets F·rate. CMT=0 and compartments
    // beyond the state vector are dropped. The empty forcing slice is the
    // "model has no built-in absorption" case; the input-rate drop it also owns
    // (#1187) is covered by `gated_infusions_suppresses_infusion_into_input_rate_cmt`.
    let mut dur_defined = DoseEvent::new(0.0, 0.0, 1, 4.0, false, 0.0);
    dur_defined.infusion_def = crate::types::InfusionDef::DurationDefined;
    let doses = vec![
        DoseEvent::new(0.0, 0.0, 1, 4.0, false, 0.0), // rate-defined: rate held
        DoseEvent::new(0.0, 0.0, 0, 9.0, false, 0.0), // CMT 0 -> dropped
        DoseEvent::new(0.0, 0.0, 5, 9.0, false, 0.0), // CMT 5 -> state 4 >= n -> dropped
        dur_defined,                                  // duration-defined: F·rate
    ];
    let f_bio = vec![0.5, 1.0, 1.0, 0.5];
    let active = vec![
        (0usize, 1.0, 3.0),
        (1, 1.0, 3.0),
        (2, 1.0, 3.0),
        (3, 1.0, 3.0),
    ];
    let gated = gated_infusions(&[], &active, &doses, &f_bio, 1);
    assert_eq!(
        gated,
        vec![(0usize, 4.0, 1.0, 3.0), (0usize, 4.0 * 0.5, 1.0, 3.0)]
    );
}

#[test]
fn add_prepared_forcing_superposes_skips_other_cmt_and_respects_floor() {
    let ode = transit_accumulator_spec(); // forcing on state 0 ≡ CMT 1
    let params = pk_transit_vec(3.0, 2.0, 1.0);
    let prepared = prepare_input_rates(&ode, &params);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // feeds R_in
        DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0),  // other cmt → ignored
    ];
    let lags = vec![0.0, 0.0];
    let f_bio = vec![1.0, 1.0];
    let t = 1.5;

    // No reset: only the CMT-1 dose contributes its R_in(tad).
    let mut dy = vec![0.0];
    add_prepared_input_rate_forcing(
        &ode,
        &prepared,
        &params,
        &doses,
        &lags,
        &f_bio,
        f64::NEG_INFINITY,
        t,
        &mut dy,
    );
    let want = prepared[0].rate(t, 100.0);
    assert!(want > 0.0);
    assert_relative_eq!(dy[0], want, max_relative = 1e-12);

    // A reset_floor after the dose time turns its forcing off.
    let mut dy_off = vec![0.0];
    add_prepared_input_rate_forcing(
        &ode,
        &prepared,
        &params,
        &doses,
        &lags,
        &f_bio,
        1.0,
        t,
        &mut dy_off,
    );
    assert_eq!(dy_off[0], 0.0);
}

/// **`ss_periodic_forcing`'s truncation break is dose-local** (#719 review, finding 2).
/// The SS pulse-train sum truncates once a tail term is negligible relative to *its own*
/// running total. Extracting it into a pure helper makes the pre-refactor hazard —
/// comparing each SS tail term against a cross-dose accumulator inflated by an unrelated
/// run-in dose — structurally impossible. This pins both directions:
///   * the local-break sum equals a brute-force full sum (the break only drops negligible
///     terms), and
///   * a break taken against an externally inflated accumulator (the old shared-`acc`
///     scheme) truncates the pre-mode leading terms and materially under-counts — so the
///     two are *not* interchangeable, which is exactly why the extraction matters.
#[test]
fn ss_periodic_forcing_break_is_dose_local_not_cross_dose() {
    // transit(n = 3, mtt = 4) ⇒ density mode at tad = mtt·(n−1)/n ≈ 2.67. A small starting
    // `tad` with a sub-mode `II` means the first several pulse-train terms are pre-mode
    // (small and *rising* toward the mode) — precisely the terms an inflated cross-dose
    // threshold would wrongly discard.
    let prep = transit_accumulator_spec().input_rate[0].prepare(&pk_transit_vec(3.0, 4.0, 1.0));
    let tad = 0.1_f64;
    let ii = 0.5_f64;
    let dose_mass = 1.0_f64;

    let got = ss_periodic_forcing(&prep, tad, ii, dose_mass);

    // Brute-force reference: every in-range term, no early break.
    let mut full = 0.0_f64;
    for j in 0..SS_EQUILIBRATION_CYCLES {
        let tad_j = tad + ii * j as f64;
        if tad_j > 0.0 {
            full += prep.rate(tad_j, dose_mass);
        }
    }
    assert!(full > 0.0);
    // The dose-local break drops only negligible tail terms.
    assert_relative_eq!(got, full, max_relative = 1e-9);

    // Replicate the pre-refactor cross-dose scheme: seed the break accumulator with a large
    // unrelated run-in contribution, then break each SS term against that shared total.
    let external_runin = 1e12_f64; // a huge prior dose's R_in already summed into `acc`
    let mut shared = external_runin;
    let mut buggy = 0.0_f64;
    for j in 0..SS_EQUILIBRATION_CYCLES {
        let tad_j = tad + ii * j as f64;
        if tad_j > 0.0 {
            let term = prep.rate(tad_j, dose_mass);
            shared += term;
            buggy += term;
            if j >= 1 && shared.abs() > 0.0 && term.abs() <= SS_TAIL_REL_FLOOR * shared.abs() {
                break;
            }
        }
    }
    // The inflated threshold breaks on the small pre-mode terms, dropping the mode and the
    // bulk of the absorption — so the cross-dose sum is far below the true (local) sum.
    assert!(
        buggy < 0.5 * full,
        "cross-dose break should truncate the pre-mode train (buggy {buggy:.6}, full {full:.6})"
    );
    assert!(
        (got - buggy).abs() > 0.25 * full,
        "the dose-local helper must not reproduce the truncated cross-dose value"
    );
}

#[test]
fn add_prepared_forcing_applies_pathway_fraction_linear_in_frac() {
    // Biphasic IG (#388): two igd forcings on one compartment, split FR1/FR2.
    // The seam adds `FR1·R_in1 + FR2·R_in2`; because the fraction enters
    // linearly, the analytic dual derivative ∂(dy)/∂FR1 is exactly R_in1 (so
    // the FOCEI/Bayes gradient w.r.t. a pathway fraction is exact, no FD).
    use crate::pk::absorption::{InputRateForcing, InputRateKind, PreparedInputRate};
    use crate::sens::dual_mixed::DualMixed;
    use crate::sens::num::PkNum;
    use crate::types::{MAX_PK_PARAMS, PK_IDX_F};

    // Slots: FR1@0, FR2@1, MAT1@2, CV2_1@3, MAT2@4, CV2_2@5, F@PK_IDX_F.
    let mk = |frac_slot, arg_slots| InputRateForcing {
        cmt: 0,
        kind: InputRateKind::InverseGaussian,
        arg_slots,
        frac_slot: Some(frac_slot),
        lag_slot: None,
    };
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![mk(0, vec![2, 3]), mk(1, vec![4, 5])],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    };

    let (fr1, fr2) = (0.7_f64, 0.3_f64);
    let mut params = vec![0.0; MAX_PK_PARAMS];
    params[0] = fr1;
    params[1] = fr2;
    params[2] = 2.0; // MAT1
    params[3] = 0.3; // CV2_1
    params[4] = 5.0; // MAT2
    params[5] = 0.6; // CV2_2
    params[PK_IDX_F] = 1.0;

    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)]; // CMT 1 → cmt 0
    let f_bio = vec![1.0];
    let (t, tad) = (2.5_f64, 2.5_f64);

    // f64: dy = FR1·R_in1 + FR2·R_in2.
    let prepared = prepare_input_rates(&ode, &params);
    let (r1, r2) = (prepared[0].rate(tad, 100.0), prepared[1].rate(tad, 100.0));
    assert!(r1 > 0.0 && r2 > 0.0);
    let mut dy = vec![0.0];
    add_prepared_input_rate_forcing(
        &ode,
        &prepared,
        &params,
        &doses,
        &[0.0],
        &f_bio,
        f64::NEG_INFINITY,
        t,
        &mut dy,
    );
    assert_relative_eq!(dy[0], fr1 * r1 + fr2 * r2, max_relative = 1e-12);

    // Dual: seed FR1 as the variable ⇒ value matches f64 and ∂(dy)/∂FR1 = R_in1.
    type D = DualMixed<1, 1>;
    let mut dp = vec![D::constant(0.0); MAX_PK_PARAMS];
    dp[0] = D::var(fr1, 0);
    dp[1] = D::constant(fr2);
    dp[2] = D::constant(2.0);
    dp[3] = D::constant(0.3);
    dp[4] = D::constant(5.0);
    dp[5] = D::constant(0.6);
    dp[PK_IDX_F] = D::constant(1.0);
    let prepared_d: Vec<PreparedInputRate<D>> = ode
        .input_rate
        .iter()
        .map(|f| f.prepare_dual::<D>(&dp).unwrap())
        .collect();
    let f_bio_d = vec![D::constant(1.0)];
    let mut dyd = vec![D::constant(0.0)];
    add_prepared_input_rate_forcing(
        &ode,
        &prepared_d,
        &dp,
        &doses,
        &[],
        &f_bio_d,
        f64::NEG_INFINITY,
        t,
        &mut dyd,
    );
    assert_relative_eq!(dyd[0].val(), fr1 * r1 + fr2 * r2, max_relative = 1e-12);
    assert_relative_eq!(dyd[0].grad[0], r1, max_relative = 1e-9);
}

#[test]
fn seam_spanning_adds_base_rhs_and_infusion() {
    // Spanning infusion is added unconditionally on top of the user RHS; with
    // no input_rate forcing the forcing branch is skipped.
    let ode = one_cpt_ode_spec();
    let params = pk_one(1.0, 1.0).values; // ke = cl/v = 1
    let prepared: Vec<PreparedInputRate> = Vec::new();
    let rhs = wrap_rhs_with_forcings(
        &ode,
        &[],
        &[],
        &[],
        f64::NEG_INFINITY,
        &prepared,
        InfusionInput::Spanning(vec![(0, 7.0)]),
        &[],
    );
    let mut dy = vec![0.0];
    rhs(&[2.0], &params, 0.0, &mut dy); // base −ke·y = −2, +7 infusion = 5
    assert_relative_eq!(dy[0], 5.0, max_relative = 1e-12);
}

#[test]
fn seam_gated_infusion_active_only_inside_window() {
    let ode = one_cpt_ode_spec();
    let params = pk_one(0.0, 1.0).values; // ke = 0 ⇒ base RHS = 0
    let prepared: Vec<PreparedInputRate> = Vec::new();
    let rhs = wrap_rhs_with_forcings(
        &ode,
        &[],
        &[],
        &[],
        f64::NEG_INFINITY,
        &prepared,
        InfusionInput::Gated(vec![(0, 3.0, 2.0, 5.0)]),
        &[],
    );
    let mut before = vec![0.0];
    rhs(&[0.0], &params, 1.0, &mut before); // before [2,5)
    assert_eq!(before[0], 0.0);
    let mut inside = vec![0.0];
    rhs(&[0.0], &params, 3.0, &mut inside); // inside
    assert_relative_eq!(inside[0], 3.0, max_relative = 1e-12);
    let mut after = vec![0.0];
    rhs(&[0.0], &params, 6.0, &mut after); // past t_end
    assert_eq!(after[0], 0.0);
}

#[test]
fn seam_applies_input_rate_forcing_on_top_of_base_rhs() {
    // With an input_rate forcing and no infusions, the seam adds R_in(tad)
    // into the forcing compartment — matching the hoisted prepared constant.
    let ode = transit_accumulator_spec(); // rhs sets dy[0] = 0
    let params = pk_transit_vec(3.0, 2.0, 1.0);
    let prepared = prepare_input_rates(&ode, &params);
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let lags = vec![0.0];
    let f_bio = vec![1.0];
    let rhs = wrap_rhs_with_forcings(
        &ode,
        &doses,
        &lags,
        &f_bio,
        f64::NEG_INFINITY,
        &prepared,
        InfusionInput::Spanning(Vec::new()),
        &[],
    );
    let t = 1.5;
    let mut dy = vec![0.0];
    rhs(&[0.0], &params, t, &mut dy);
    assert_relative_eq!(dy[0], prepared[0].rate(t, 100.0), max_relative = 1e-12);
}

#[test]
fn ode_init_state_seeds_plain_path() {
    // No doses; the system starts at baseline kin/kout = 5 and stays there
    // (dR/dt = 0). Without init it would start at 0 and climb.
    let ode = turnover_ode_spec_with_init();
    let pk = pk_kin_kout(10.0, 2.0);
    let subj = make_subject(Vec::new(), vec![0.0, 1.0, 5.0, 20.0]);
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (i, &p) in preds.iter().enumerate() {
        assert_relative_eq!(p, 5.0, epsilon = 1e-5);
        let _ = i;
    }
}

#[test]
fn ode_init_state_then_dose_and_reset_reapplies_init() {
    // Exercises all three: (1) init seeds the start at baseline=5, (2) a
    // bolus at t=0 lands on top of the seeded state (5 + 20 = 25), and
    // (3) an EVID=3 reset at t=5 re-applies init (back to 5, NOT zero).
    let ode = turnover_ode_spec_with_init();
    let pk = pk_kin_kout(10.0, 2.0); // baseline 5
    let doses = vec![DoseEvent::new(0.0, 20.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0, 5.0];
    let mut subj = make_subject(doses, obs_times.clone());
    subj.reset_times = vec![5.0];
    let pk_dose = vec![pk; subj.doses.len()];
    let pk_obs = vec![pk; obs_times.len()];
    // The reset row's OWN snapshot, deliberately different from every other record's
    // (#1133): `KIN = 40` gives baseline 20, against the dose/obs rows' 5. A uniform
    // `vec![pk; …]` here would make `pk_at_reset[idx]` bit-identical to `last_pk` and the
    // assertion below could not tell the two conventions apart — which is exactly how this
    // test passed before the fix.
    let pk_reset = vec![pk_kin_kout(40.0, 2.0)]; // baseline 20

    let preds =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &pk_reset);
    // t=0: init(5) + bolus(20) = 25.
    assert_relative_eq!(preds[0], 25.0, epsilon = 1e-6);
    // t=5: the reset re-applies init at the RESET ROW's snapshot → 20. A zeroing reset
    // gives 0; carrying the previous record forward (the pre-#1133 bug) gives 5.
    assert_relative_eq!(preds[1], 20.0, epsilon = 1e-6);
}

#[test]
fn ode_reset_reseeds_each_reset_from_its_own_snapshot() {
    // #1133, the per-reset index. Every other reset test in the repo has exactly ONE
    // reset, so `pk_at_reset[idx]` is always `pk_at_reset[0]` and swapping one for the
    // other passes the whole suite. Two resets with different baselines separate them.
    let ode = turnover_ode_spec_with_init();
    let pk = pk_kin_kout(10.0, 2.0); // baseline 5
    let obs_times = vec![0.0, 5.0, 9.0];
    let mut subj = make_subject(Vec::new(), obs_times.clone());
    subj.reset_times = vec![5.0, 9.0];
    let pk_obs = vec![pk; obs_times.len()];
    // Reset 0 -> baseline 20, reset 1 -> baseline 35.
    let pk_reset = vec![pk_kin_kout(40.0, 2.0), pk_kin_kout(70.0, 2.0)];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &[], &pk_obs, &[], &pk_reset);
    assert_relative_eq!(preds[0], 5.0, epsilon = 1e-6);
    // Indexing `pk_at_reset[0]` for both would read 20.0 here.
    assert_relative_eq!(preds[1], 20.0, epsilon = 1e-6);
    assert_relative_eq!(preds[2], 35.0, epsilon = 1e-6);
}

#[test]
fn ode_init_uses_chronologically_first_record_not_first_dose() {
    // Regression (Copilot review #1): with time-varying covariates the init
    // snapshot must come from the earliest record by TIME, not the first
    // dose. Here a pre-dose observation at t=0 carries KIN=10 (baseline 5)
    // while a later dose at t=5 carries KIN=100 (baseline 50). Seeding must
    // use the t=0 obs → prediction at t=0 is 5, not 50.
    let ode = turnover_ode_spec_with_init();
    let doses = vec![DoseEvent::new(5.0, 0.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0];
    let subj = make_subject(doses, obs_times.clone());
    let pk_dose = vec![pk_kin_kout(100.0, 2.0)]; // baseline 50 (must NOT be used)
    let pk_obs = vec![pk_kin_kout(10.0, 2.0)]; // baseline 5 (first record)

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);
    assert_relative_eq!(preds[0], 5.0, epsilon = 1e-9);
}

#[test]
fn ode_init_reapplied_when_reset_is_first_event() {
    // Regression (Copilot review #2): an EVID=4 reset+dose at t=0 re-applies
    // init *before* the same-time dose. last_pk must be seeded from the
    // first record's params (not zeroed defaults), or the re-applied
    // baseline would evaluate KIN/KOUT with zero params and collapse to 0.
    // Expected: init(5) re-applied at reset, then bolus 20 → 25.
    let ode = turnover_ode_spec_with_init();
    let doses = vec![DoseEvent::new(0.0, 20.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0];
    let mut subj = make_subject(doses, obs_times.clone());
    subj.reset_times = vec![0.0];
    let pk = pk_kin_kout(10.0, 2.0); // baseline 5
    let pk_dose = vec![pk];
    let pk_obs = vec![pk];
    let pk_reset = vec![pk];

    let preds =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &pk_reset);
    // Re-applied baseline (5) + bolus (20) = 25. A zero-param re-seed would
    // give 0 + 20 = 20.
    assert_relative_eq!(preds[0], 25.0, epsilon = 1e-6);
}

#[test]
fn ode_event_driven_reset_evid3_zeros_state() {
    // EVID=3 reset at t=5 must zero the ODE state: obs after the reset
    // read ~0 when no later dose exists.
    let ode = one_cpt_ode_spec();
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![1.0, 6.0, 10.0];
    let mut subj = make_subject(doses, obs_times.clone());
    subj.reset_times = vec![5.0];
    let pk = pk_one(10.0, 100.0);
    let pk_dose = vec![pk; subj.doses.len()];
    let pk_obs = vec![pk; obs_times.len()];
    let pk_reset = vec![pk; subj.reset_times.len()];

    let preds =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &pk_reset);
    assert!(preds[0] > 0.0, "pre-reset obs should be positive");
    assert_relative_eq!(preds[1], 0.0, epsilon = 1e-6);
    assert_relative_eq!(preds[2], 0.0, epsilon = 1e-6);
}

#[test]
fn ode_event_driven_reset_evid4_matches_fresh_dose() {
    // EVID=4 (reset + dose) at t=10 must match a single fresh dose at t=10.
    let ode = one_cpt_ode_spec();
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0),
    ];
    let obs_times = vec![10.0, 12.0, 15.0];
    let mut subj = make_subject(doses, obs_times.clone());
    subj.reset_times = vec![10.0];
    let pk = pk_one(8.0, 50.0);
    let pk_dose = vec![pk; subj.doses.len()];
    let pk_obs = vec![pk; obs_times.len()];
    let pk_reset = vec![pk; subj.reset_times.len()];

    let preds =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &pk_reset);

    // Reference: lone 500 mg dose at t=10 through the same ODE path.
    let fresh = make_subject(
        vec![DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0)],
        obs_times.clone(),
    );
    let fresh_pk_dose = vec![pk; fresh.doses.len()];
    let fresh_pk_obs = vec![pk; obs_times.len()];
    let expected = ode_predictions_event_driven(
        &ode,
        &fresh,
        &[],
        &[],
        &fresh_pk_dose,
        &fresh_pk_obs,
        &[],
        &[],
    );
    for (a, e) in preds.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *e, epsilon = 1e-6, max_relative = 1e-6);
    }
}

#[test]
fn ode_event_driven_matches_constant_path_when_pk_constant() {
    // Equivalence: when the per-event PK params are all the same, the
    // event-driven ODE path must agree with the existing single-snapshot
    // path. This is the "no TV covariates" sanity check.
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(8.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    let obs_times = vec![1.0, 4.0, 8.5, 12.0, 24.0];
    let subj = make_subject(doses, obs_times.clone());
    let pk = pk_one(5.0, 80.0);
    let pk_dose = vec![pk; subj.doses.len()];
    let pk_obs = vec![pk; obs_times.len()];
    let ode = one_cpt_ode_spec();

    let baseline = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    let event_driven =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);
    assert_eq!(baseline.len(), event_driven.len());
    for (b, e) in baseline.iter().zip(event_driven.iter()) {
        // ODE solver tolerance is ~1e-4 relative — a tighter equality
        // would over-constrain RK45.
        assert_relative_eq!(*b, *e, epsilon = 1e-6, max_relative = 1e-4);
    }
}

#[test]
fn ode_event_driven_picks_up_changing_cl() {
    // Same shape as the analytical TV test: CL doubles between two doses.
    // End-of-interval / NONMEM convention — each segment uses the PK
    // params at the record being arrived at:
    //   [0, t_obs1=5]: uses pk at obs1 = pk_low → ke = 0.05
    //   [5, t_dose2=10]: uses pk at dose2 = pk_high → ke = 0.10
    //   [10, t_obs2=12]: uses pk at obs2 = pk_high → ke = 0.10
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    let obs_times = vec![5.0, 12.0];
    let subj = make_subject(doses, obs_times);
    let pk_low = pk_one(5.0, 100.0); // ke = 0.05
    let pk_high = pk_one(10.0, 100.0); // ke = 0.10
    let pk_dose = vec![pk_low, pk_high];
    let pk_obs = vec![pk_low, pk_high];
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);

    // [0, 5] uses pk_low (pk at obs1): A(5) = 1000 * exp(-0.05*5) ≈ 778.80
    let a5 = 1000.0 * (-0.05f64 * 5.0).exp();
    assert_relative_eq!(preds[0], a5, epsilon = 1e-3, max_relative = 1e-4);

    // [5, 10] uses pk_high (pk at dose2): ke=0.10 for 5h.
    //   A(10⁻) = A(5) * exp(-0.10*5) = 778.80 * 0.6065 ≈ 472.37
    // After dose2: A(10⁺) = 472.37 + 1000 = 1472.37.
    // [10, 12] uses pk_high (pk at obs2): A(12) = 1472.37 * exp(-0.20) ≈ 1205.49
    let a10_minus = a5 * (-0.10f64 * 5.0).exp();
    let a10_plus = a10_minus + 1000.0;
    let a12 = a10_plus * (-0.20f64).exp();
    assert_relative_eq!(preds[1], a12, epsilon = 1e-2, max_relative = 1e-4);
}

/// 1-cpt oral ODE: dA1/dt = -ka·A1, dA2/dt = ka·A1 - ke·A2.
/// Used to test infusion into the depot compartment (cmt=1).
fn one_cpt_oral_ode_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ka = p[crate::types::PK_IDX_KA];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ka * y[0];
            dy[1] = ka * y[0] - ke * y[1];
        }),
        n_states: 2,
        state_names: vec!["depot".into(), "central".into()],
        readout: OdeReadout::ObsCmt(1),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

#[test]
fn ode_infusion_one_cpt_iv_matches_closed_form() {
    // 1-cpt IV infusion. Closed form during infusion:
    //   A(t) = (R/ke) · (1 - exp(-ke·t))
    // and after end-of-infusion T:
    //   A(t) = A(T) · exp(-ke·(t-T))
    // Verifies that the wrapped-RHS path produces the right shape.
    let rate = 100.0;
    let amt = 1000.0; // duration = 10 h
    let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
    let obs_times = vec![5.0, 10.0, 15.0, 20.0];
    let subj = make_subject(doses, obs_times);
    let pk = pk_one(5.0, 80.0); // ke = 0.0625
    let ke = 5.0_f64 / 80.0;
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    // During infusion [0, 10]
    let a5 = (rate / ke) * (1.0 - (-ke * 5.0).exp());
    let a10 = (rate / ke) * (1.0 - (-ke * 10.0).exp());
    // After end-of-infusion
    let a15 = a10 * (-ke * 5.0).exp();
    let a20 = a10 * (-ke * 10.0).exp();

    assert_relative_eq!(preds[0], a5, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[1], a10, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[2], a15, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[3], a20, epsilon = 1e-2, max_relative = 1e-4);
}

#[test]
fn ode_event_driven_infusion_matches_constant_pk_path() {
    // Same infusion-only subject, run through both paths with identical
    // per-event PK params. Verifies the event-driven path's
    // InfusionEnd handling agrees with the simple-timeline path.
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0)];
    let obs_times = vec![3.0, 7.0, 10.0, 14.0, 20.0];
    let subj = make_subject(doses, obs_times.clone());
    let pk = pk_one(5.0, 80.0);
    let pk_dose = vec![pk; subj.doses.len()];
    let pk_obs = vec![pk; obs_times.len()];
    let ode = one_cpt_ode_spec();

    let baseline = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    let event_driven =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);
    assert_eq!(baseline.len(), event_driven.len());
    for (b, e) in baseline.iter().zip(event_driven.iter()) {
        assert_relative_eq!(*b, *e, epsilon = 1e-3, max_relative = 1e-4);
    }
}

#[test]
fn ode_event_driven_form_c_uses_observation_covariates() {
    // Regression for a NONMEM translation with paired total/free assays:
    // the dose row carried FREE=3, while same-time observation rows carried
    // FREE=0 and FREE=1. Form C must see the observation snapshot, not the
    // subject-level first-row covariate.
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 0.0;
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::Single(Box::new(|state, _pk, _theta, _eta, covariates| {
            state[0] * covariates.get("FREE").copied().unwrap_or(0.0)
        })),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    };
    let mut subj = make_subject(
        vec![DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0)],
        vec![1.0, 1.0],
    );
    subj.covariates.insert("FREE".into(), 3.0);
    subj.dose_covariates = vec![HashMap::from([("FREE".to_string(), 3.0)])];
    subj.obs_covariates = vec![
        HashMap::from([("FREE".to_string(), 0.0)]),
        HashMap::from([("FREE".to_string(), 1.0)]),
    ];
    let pk = pk_one(0.0, 1.0);
    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &[pk], &[pk, pk], &[], &[]);

    assert_relative_eq!(preds[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(preds[1], 10.0, epsilon = 1e-12);
}

#[test]
fn ode_overlapping_infusions_sum_rates() {
    // Two infusions overlap on [2, 6] for a combined rate of 200,
    // then both end at t=6. After t=6, plain elimination.
    //   inf1: t∈[0,6], rate=100
    //   inf2: t∈[2,6], rate=100
    let doses = vec![
        DoseEvent::new(0.0, 600.0, 1, 100.0, false, 0.0),
        DoseEvent::new(2.0, 400.0, 1, 100.0, false, 0.0),
    ];
    let obs_times = vec![2.0, 4.0, 6.0, 12.0];
    let subj = make_subject(doses, obs_times);
    let pk = pk_one(5.0, 80.0);
    let ke = 5.0_f64 / 80.0;
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    // [0, 2]: rate=100, A(0)=0 → A(t) = (100/ke)·(1 - exp(-ke·t))
    let a2 = (100.0_f64 / ke) * (1.0 - (-ke * 2.0).exp());
    // [2, 6]: rate=200, A0=a2
    //   A(t) = (200/ke) + (A0 - 200/ke) · exp(-ke·(t-2))
    let r_over_ke = 200.0_f64 / ke;
    let a4 = r_over_ke + (a2 - r_over_ke) * (-ke * 2.0).exp();
    let a6 = r_over_ke + (a2 - r_over_ke) * (-ke * 4.0).exp();
    // [6, ∞]: rate=0
    let a12 = a6 * (-ke * 6.0).exp();

    assert_relative_eq!(preds[0], a2, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[1], a4, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[2], a6, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[3], a12, epsilon = 1e-2, max_relative = 1e-4);
}

#[test]
fn ode_infusion_then_bolus() {
    // Infusion [0, 10] followed by a bolus at t=15. Observation at
    // the bolus time should record state AFTER the bolus is applied,
    // matching the existing bolus convention.
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0), // infusion, ends at 10
        DoseEvent::new(15.0, 500.0, 1, 0.0, false, 0.0),   // bolus
    ];
    let obs_times = vec![10.0, 15.0, 20.0];
    let subj = make_subject(doses, obs_times);
    let pk = pk_one(5.0, 80.0);
    let ke = 5.0_f64 / 80.0;
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    let a10 = (100.0_f64 / ke) * (1.0 - (-ke * 10.0).exp());
    let a15_pre = a10 * (-ke * 5.0).exp();
    let a15_post = a15_pre + 500.0;
    let a20 = a15_post * (-ke * 5.0).exp();

    assert_relative_eq!(preds[0], a10, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[1], a15_post, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[2], a20, epsilon = 1e-2, max_relative = 1e-4);
}

#[test]
fn ode_infusion_into_oral_depot() {
    // Infusion into depot (cmt=1) of a 1-cpt oral model. Verifies
    // that the wrapped RHS adds `+rate` to the correct compartment
    // (depot index 0), not central (index 1). For the depot alone
    // the closed form is decoupled from ke:
    //   A1(t) during infusion = (R/ka)·(1 - exp(-ka·t))
    //   A1(t) after end T     = A1(T) · exp(-ka·(t-T))
    // Re-use the oral ODE spec but observe the depot.
    let mut ode = one_cpt_oral_ode_spec();
    ode.readout = OdeReadout::ObsCmt(0);

    let rate = 50.0;
    let amt = 200.0; // duration = 4 h
    let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
    let obs_times = vec![2.0, 4.0, 8.0];
    let subj = make_subject(doses, obs_times);
    let mut pk = pk_one(5.0, 80.0);
    pk.values[crate::types::PK_IDX_KA] = 1.0;
    let ka = 1.0_f64;

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    let depot_2 = (rate / ka) * (1.0 - (-ka * 2.0).exp());
    let depot_4 = (rate / ka) * (1.0 - (-ka * 4.0).exp());
    let depot_8 = depot_4 * (-ka * 4.0).exp();

    assert_relative_eq!(preds[0], depot_2, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[1], depot_4, epsilon = 1e-2, max_relative = 1e-4);
    assert_relative_eq!(preds[2], depot_8, epsilon = 1e-2, max_relative = 1e-4);
}

// Degenerate input guards: `rate > 0` alone is insufficient to mark a
// dose as an infusion — `duration = amt/rate` must also be > 0 and
// finite. Otherwise:
//   - `amt < 0` would push an infusion-end break time *before* the
//     dose, scrambling the segmented integration order.
//   - `amt = NaN` would make `partial_cmp` return None and panic
//     the break-time sort.
//   - In both cases, the bolus branch is skipped (because
//     `is_infusion()` is true on rate alone), so the dose silently
//     disappears from the prediction.
// `is_real_infusion` falls back to the bolus path for these rows.

#[test]
fn ode_degenerate_zero_amt_with_positive_rate_falls_back_to_bolus() {
    // amt=0, rate>0 → duration=0. Treated as a (no-op) bolus.
    // Result must match "no dose at all".
    let doses = vec![DoseEvent::new(0.0, 0.0, 1, 100.0, false, 0.0)];
    let obs_times = vec![1.0, 5.0, 10.0];
    let subj = make_subject(doses, obs_times);
    let pk = pk_one(5.0, 80.0);
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    assert_eq!(preds, vec![0.0, 0.0, 0.0]);
}

#[test]
fn ode_degenerate_negative_amt_with_positive_rate_does_not_break_ordering() {
    // amt<0, rate>0 → duration<0. Pre-fix, the infusion-end break time
    // would sort *before* the dose, producing nonsense segments and
    // (silently) zero output because the bolus branch was skipped.
    // Post-fix this is treated as a bolus with negative amt — at least
    // visible to the caller.
    let doses = vec![DoseEvent::new(0.0, -10.0, 1, 100.0, false, 0.0)];
    let obs_times = vec![0.0, 1.0];
    let subj = make_subject(doses, obs_times);
    let pk = pk_one(5.0, 80.0);
    let ode = one_cpt_ode_spec();

    // Must not panic; the negative bolus update is clamped to 0 by
    // the negative-prediction guard in `ode_predictions`.
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert_eq!(preds.len(), 2);
}

#[test]
fn ode_degenerate_nan_amt_with_positive_rate_does_not_panic() {
    // amt=NaN, rate>0 → duration=NaN. Pre-fix, sort_by(partial_cmp).unwrap()
    // would panic on the break-time vec. Post-fix the row falls through
    // to the bolus branch and the panic is avoided.
    let doses = vec![DoseEvent::new(0.0, f64::NAN, 1, 100.0, false, 0.0)];
    let obs_times = vec![1.0];
    let subj = make_subject(doses, obs_times);
    let pk = pk_one(5.0, 80.0);
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert_eq!(preds.len(), 1);
}

#[test]
fn ode_iv_bolus_with_lagtime_shifts_curve() {
    // 1-cpt IV bolus integrated via ODE: with lagtime=2.0 and dose at
    // t_dose=0, the central-amount state should be 0 until t=2 (the
    // lagged dose arrival), then decay as if dose-time were 2.
    // (`one_cpt_ode_spec` observes the amount A(t), not A/V.)
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![1.0, 3.0, 6.0];
    let subj = make_subject(doses, obs_times);
    let mut pk = pk_one(5.0, 80.0);
    pk.values[crate::types::PK_IDX_LAGTIME] = 2.0;
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    // At t=1, dose has not yet arrived (lagtime=2). State stays 0.
    assert_relative_eq!(preds[0], 0.0, epsilon = 1e-10);

    // At t=3, effective elapsed time since dose is 1.0.
    // A(1) = Amt * exp(-ke * 1)
    let ke = 5.0_f64 / 80.0;
    let expected_3 = 1000.0_f64 * (-ke * 1.0).exp();
    assert_relative_eq!(preds[1], expected_3, epsilon = 1e-4, max_relative = 1e-4);

    // At t=6, effective elapsed time is 4.0.
    let expected_6 = 1000.0_f64 * (-ke * 4.0).exp();
    assert_relative_eq!(preds[2], expected_6, epsilon = 1e-4, max_relative = 1e-4);
}

#[test]
fn ode_infusion_with_lagtime_shifts_break_times_and_active_window() {
    // Direct test of the ODE infusion + lagtime path that the analytical
    // superposition test alone doesn't cover. Amt=100, rate=100 ⇒
    // duration=1.0; with lagtime=0.5, the active-infusion window runs
    // [2.5, 3.5] rather than [2.0, 3.0]. Compare against an equivalent
    // unlagged dose starting at 2.5 — predictions at matched observation
    // offsets should agree to ODE tolerance.
    let dose_lag = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
    assert!(dose_lag.is_infusion() && dose_lag.duration > 0.0);
    let subj_lag = make_subject(vec![dose_lag], vec![2.0, 3.0, 4.0]);
    let mut pk_lag = pk_one(5.0, 80.0);
    pk_lag.values[crate::types::PK_IDX_LAGTIME] = 0.5;

    // Reference: dose shifted at the data level, no lagtime applied.
    let dose_ref = DoseEvent::new(2.5, 100.0, 1, 100.0, false, 0.0);
    let subj_ref = make_subject(vec![dose_ref], vec![2.0, 3.0, 4.0]);
    let pk_ref = pk_one(5.0, 80.0);

    let ode = one_cpt_ode_spec();
    let preds_lag = ode_predictions(&ode, &pk_lag.values, &[], &[], &subj_lag);
    let preds_ref = ode_predictions(&ode, &pk_ref.values, &[], &[], &subj_ref);

    // Observation before lagged infusion start: zero.
    assert_relative_eq!(preds_lag[0], 0.0, epsilon = 1e-10);

    // Observations during and after the lagged infusion: must match the
    // reference where the dose was shifted at the dataset level.
    assert_relative_eq!(
        preds_lag[1],
        preds_ref[1],
        epsilon = 1e-4,
        max_relative = 1e-4
    );
    assert_relative_eq!(
        preds_lag[2],
        preds_ref[2],
        epsilon = 1e-4,
        max_relative = 1e-4
    );
}

// --- Steady-state (SS=1) tests ---
//
// The ODE SS path is verified against the corresponding analytical
// 1-cpt SS closed forms (PR #75): a 1-cpt IV-bolus ODE with SS dose
// must match `one_cpt_iv_bolus_ss` to RK45 tolerance, and similarly
// for infusion. This cross-checks the per-cycle pulse-expansion
// equilibration loop in `equilibrate_ss_state`.

#[test]
fn ss_cycle_converged_is_mixed_atol_rtol_on_increment() {
    // Referenced directly from `crate::dosing` (test-only here — not re-exported by
    // the private facade above).
    use crate::dosing::{ss_cycle_converged, SS_EQUILIBRATION_TOL};
    // Increment below tol: a 1e-13 move on a magnitude-100 state is ≪ tol·(|a| + max) →
    // converged.
    assert!(ss_cycle_converged(
        &[100.0, 50.0],
        &[100.0 + 1e-13, 50.0],
        SS_EQUILIBRATION_TOL
    ));
    // Increment above tol: a 1e-4 move ≫ tol·(|a| + max) → not converged.
    assert!(!ss_cycle_converged(
        &[100.0, 50.0],
        &[100.0001, 50.0],
        SS_EQUILIBRATION_TOL
    ));
    // Scale-invariant: the same *relative* increment at a tiny magnitude → same verdict.
    assert!(ss_cycle_converged(
        &[1e-6],
        &[1e-6 + 1e-19],
        SS_EQUILIBRATION_TOL
    ));
    assert!(!ss_cycle_converged(
        &[1e-6],
        &[1e-6 + 1e-15],
        SS_EQUILIBRATION_TOL
    ));
    // A genuinely zero state (no dose effect) is trivially converged.
    assert!(ss_cycle_converged(
        &[0.0, 0.0],
        &[0.0, 0.0],
        SS_EQUILIBRATION_TOL
    ));
    // Non-finite compartment (blown-up integration) is never "converged" → no early exit.
    assert!(!ss_cycle_converged(
        &[f64::NAN, 1.0],
        &[f64::NAN, 1.0],
        SS_EQUILIBRATION_TOL
    ));
    assert!(!ss_cycle_converged(
        &[f64::INFINITY],
        &[1.0],
        SS_EQUILIBRATION_TOL
    ));
    // Per-compartment: a small compartment moving 1% (relative to itself), by an amount well
    // above the system-scale atol, blocks the stop even though the dominant compartment is
    // steady.
    assert!(!ss_cycle_converged(
        &[100.0, 1e-3],
        &[100.0, 1e-3 * 1.01],
        SS_EQUILIBRATION_TOL
    ));
    // #532 review #1 — the footgun the increment test fixes: a compartment whose *magnitude*
    // (5e-11) is below the old `tol·max_mag` floor (1e-10) but which is still *moving* by
    // more than that floor (Δ = 1.5e-10). The old magnitude-floor declared it converged; the
    // increment test correctly keeps the loop running.
    assert!(!ss_cycle_converged(
        &[100.0, 5e-11],
        &[100.0, 2e-10],
        SS_EQUILIBRATION_TOL
    ));
}

#[test]
fn ss_linear_disposition_uses_exact_fixed_point() {
    // #914: a LINEAR disposition now equilibrates via the exact closed-form fixed point
    // `u_ss = (I − M)⁻¹·b` (one recorded cycle), for both fast and slow PK, rather than the
    // up-to-50-cycle pulse train. The slow case is the payoff: the old train ran the full budget
    // and truncated its geometric tail (~30% low here), while the exact solve nails the
    // analytical steady state. (The #519 pulse-train early stop is now reachable only on the
    // nonlinear fallback — see `ss_nonlinear_bolus_with_steady_state_uses_fallback`.)
    let mut ode = one_cpt_ode_spec();
    ode.solver_opts.reltol = 1e-10;
    ode.solver_opts.abstol = 1e-12;
    let ii = 12.0_f64;
    let amt = 1000.0_f64;
    let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);

    // (cl, v) — "fast" (ke·II = 6, the old early stop fired) and "slow" (ke·II ≈ 0.024, the old
    // full-budget truncation was ~30% low). The fast case is deliberately not *extreme*
    // (ke·II ≫ 6): a near-total between-dose decay drives the trough toward the RK45 `abstol`
    // floor, where relative precision is lost — an integration-noise artifact, not the fixed
    // point being wrong (`ode_provider_ss_linear_bolus_uses_exact_solve` checks the gradient too).
    for (cl, v, label) in [(5.0_f64, 10.0_f64, "fast"), (0.1_f64, 50.0_f64, "slow")] {
        let pk = pk_one(cl, v);
        let trough = equilibrate_ss_state(&ode, &pk.values, &dose, &ode.solver_opts, &[]);
        assert_eq!(
            crate::dosing::last_ss_equilibration_cycles(),
            1,
            "{label} linear PK must equilibrate via the exact fixed point (one cycle)"
        );
        // Pre-pulse SS amount for a 1-cpt bolus: `F·amt·e^{−ke·II}/(1 − e^{−ke·II})` (F = 1).
        // The state stores amount, so compare directly.
        let ke = cl / v;
        let expected = amt * (-ke * ii).exp() / (1.0 - (-ke * ii).exp());
        assert_relative_eq!(trough[0], expected, max_relative = 1e-6);
    }
}

/// 1-cpt Michaelis–Menten **disposition** (no absorption kernel), amount state:
/// `dA/dt = −Vmax·A/(Km + A)`, Vmax in the CL slot, Km in the V slot. A plain bolus into this
/// compartment is genuinely nonlinear, so the #914 exact fixed point's linearity self-check
/// declines and the SS equilibration falls back to the capped pulse train (with the #867
/// non-convergence warning when it can't converge). Distinct from `mm_ss_absorption_spec`, which
/// also carries a `first_order` input rate and so exercises the input-rate branch instead.
fn mm_disposition_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let vmax = p[crate::types::PK_IDX_CL];
            let km = p[crate::types::PK_IDX_V];
            dy[0] = -vmax * y[0] / (km + y[0]);
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    }
}

/// Independent SS reference for a plain **bolus** into a (possibly nonlinear) disposition: run the
/// pulse train `(add F·amt; integrate II)` from a zero state for `max_cycles` and return the
/// pre-next-pulse trough. Uncapped (`max_cycles ≫ 50`), so for a disposition that admits a
/// periodic steady state it is fully converged — the genuine SS the 50-cycle production fallback
/// approximates. `F = 1`.
fn bolus_ss_reference(ode: &OdeSpec, pk: &[f64], dose: &DoseEvent, max_cycles: usize) -> Vec<f64> {
    let eq = ss_equilibration_opts(&ode.solver_opts);
    let cmt = dose.cmt_idx();
    let mut u = vec![0.0; ode.n_states];
    for _ in 0..max_cycles {
        u[cmt] += dose.amt;
        if let Some(last) = solve_ode(&ode.rhs, &u, (0.0, dose.ii), pk, &[dose.ii], &eq).last() {
            u.copy_from_slice(&last.u);
        }
    }
    u
}

/// #914 gap 1 (nonlinear branch preserved): a Michaelis–Menten **disposition** that *admits* a
/// periodic steady state (mean input `100/8 = 12.5 ≪ Vmax = 50`) fails the exact fixed point's
/// linearity self-check and falls back to the capped pulse train — which, since it converges well
/// inside the 50-cycle budget, reaches the true SS (matching an uncapped run-in) with **no**
/// warning. Proves the exact-solve short-circuit does not swallow a genuinely nonlinear model.
#[test]
fn ss_nonlinear_bolus_with_steady_state_uses_fallback() {
    let _guard = crate::dosing::SS_WARN_SINK_READER_GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    crate::dosing::clear_ss_nonconvergence_warnings();

    let mut ode = mm_disposition_spec();
    ode.solver_opts.reltol = 1e-10;
    ode.solver_opts.abstol = 1e-12;
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 50.0; // Vmax
    pk.values[crate::types::PK_IDX_V] = 30.0; // Km — the peak amount ≈ 100 ≫ Km, so genuinely nonlinear
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 8.0);

    let trough = equilibrate_ss_state(&ode, &pk.values, &dose, &ode.solver_opts, &[]);
    let cycles = crate::dosing::last_ss_equilibration_cycles();
    assert!(
        (2..SS_EQUILIBRATION_CYCLES).contains(&cycles),
        "a nonlinear disposition must fall back to the pulse train and converge inside the budget, \
         ran {cycles} cycles"
    );
    let reference = bolus_ss_reference(&ode, &pk.values, &dose, 500);
    assert_relative_eq!(trough[0], reference[0], max_relative = 1e-6);
    assert!(
        crate::dosing::take_ss_nonconvergence_warnings().is_empty(),
        "a converging nonlinear SS must not warn"
    );
}

/// #914 gap 2 (the newly-wired warning): a Michaelis–Menten **disposition** dosed *above capacity*
/// (mean input `50/8 = 6.25 > Vmax = 5`) has no periodic steady state, so the bolus fallback runs
/// the full 50-cycle budget without converging and now surfaces the #867 non-convergence warning
/// — previously silent for the ordinary bolus/infusion path (it was wired only into the
/// input-rate branch). Predictions stay finite.
#[test]
fn ss_nonlinear_over_capacity_bolus_caps_and_warns() {
    let _guard = crate::dosing::SS_WARN_SINK_READER_GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    crate::dosing::clear_ss_nonconvergence_warnings();

    let ode = mm_disposition_spec();
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 5.0; // Vmax
    pk.values[crate::types::PK_IDX_V] = 8.0; // Km
    let ss = DoseEvent::new(0.0, 50.0, 1, 0.0, true, 8.0); // mean input 6.25 > Vmax 5

    let trough = equilibrate_ss_state(&ode, &pk.values, &ss, &ode.solver_opts, &[]);
    assert_eq!(
        crate::dosing::last_ss_equilibration_cycles(),
        SS_EQUILIBRATION_CYCLES,
        "an over-capacity (no-SS) nonlinear disposition must run the full capped budget"
    );
    assert!(trough.iter().all(|x| x.is_finite()));

    let subj = make_subject(vec![ss], vec![1.0, 4.0, 7.9]);
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "predictions must stay finite: {preds:?}"
    );
    let warnings = crate::dosing::take_ss_nonconvergence_warnings();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("Steady-state (SS=1) equilibration")),
        "an over-capacity bolus must surface a non-convergence warning; got: {warnings:?}"
    );
}

#[test]
fn ode_ss_iv_bolus_matches_analytical_ss() {
    // The test ODE stores compartment AMOUNT (dA/dt = -ke·A), while the
    // analytical formula returns CONCENTRATION = amount/V. Divide
    // before comparing.
    use crate::pk::one_cpt_iv_bolus_ss;
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let amt = 1000.0_f64;
    let ii = 12.0_f64;
    // Sample times within and beyond one dosing interval.
    let obs_times = vec![1.0, 4.0, 8.0, 11.0, 14.0, 24.0];
    let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
    let subj = make_subject(vec![dose.clone()], obs_times.clone());
    let pk = pk_one(cl, v);
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert_eq!(preds.len(), obs_times.len());

    for (j, &t) in obs_times.iter().enumerate() {
        let expected = one_cpt_iv_bolus_ss(&dose, t, cl, v);
        // The RK45 reltol/abstol set in `[fit_options]` dominate the error here; the SS
        // equilibration now stops on the `SS_EQUILIBRATION_TOL` (1e-12) early-stop
        // (#519) rather than a fixed N=50 truncation, so its own tail is negligible.
        // 1e-4 is the safe headroom across the population.
        assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
    }
}

#[test]
fn ode_ss_infusion_matches_analytical_ss() {
    use crate::pk::one_cpt_infusion_ss;
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let amt = 1000.0_f64;
    let rate = 250.0_f64; // T_inf = 4 h
    let ii = 24.0_f64;
    // Cover during-infusion, post-infusion, and beyond one interval.
    let obs_times = vec![1.0, 3.5, 4.0, 8.0, 12.0, 23.0, 48.0];
    let dose = DoseEvent::new(0.0, amt, 1, rate, true, ii);
    let subj = make_subject(vec![dose.clone()], obs_times.clone());
    let pk = pk_one(cl, v);
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, &t) in obs_times.iter().enumerate() {
        let expected = one_cpt_infusion_ss(&dose, t, cl, v);
        assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
    }
}

/// #914 regression, **infusion** side: the exact solve on a SLOW disposition. At `ke·II ≈ 0.03`
/// the old 50-cycle pulse train truncated the SS trough ~22% low (`exp(−50·ke·II) ≈ 0.22`); the
/// exact `(I − M)⁻¹·b` fixed point matches the analytical infusion SS closed form. The fast
/// `ode_ss_infusion_matches_analytical_ss` above sits at `ke·II = 1.5`, where a 50-cycle residual
/// is `exp(−75) ≈ 1e-33` — it passes on *both* old and new code and so cannot detect an
/// infusion truncation-tail bug. This is the infusion analogue of the slow-bolus case in
/// `ss_linear_disposition_uses_exact_fixed_point`, and it fails against the pre-#914 truncated
/// train (the only PR-level truncation-sensitive infusion oracle — the `tests/` twin runs nightly).
#[test]
fn ode_ss_slow_infusion_matches_analytical_ss() {
    use crate::pk::one_cpt_infusion_ss;
    let cl = 0.1_f64;
    let v = 80.0_f64; // ke = CL/V = 1.25e-3, ke·II = 0.03 → the pre-#914 train was ~22% low
    let amt = 1000.0_f64;
    let rate = 100.0_f64; // T_inf = 10 h < II
    let ii = 24.0_f64;
    // During-infusion, at the window end, post-infusion within the interval, and beyond it.
    let obs_times = vec![2.0, 10.0, 12.0, 20.0, 30.0, 48.0];
    let dose = DoseEvent::new(0.0, amt, 1, rate, true, ii);
    let subj = make_subject(vec![dose.clone()], obs_times.clone());
    let pk = pk_one(cl, v);
    let mut ode = one_cpt_ode_spec();
    // Tight solver tol so the forward walk tracks the exact analytical SS (the equilibration
    // itself already runs at ss_equilibration_opts); the 22% truncation gap dwarfs this regardless.
    ode.solver_opts.reltol = 1e-11;
    ode.solver_opts.abstol = 1e-13;

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, &t) in obs_times.iter().enumerate() {
        let expected = one_cpt_infusion_ss(&dose, t, cl, v);
        assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-7, max_relative = 1e-6);
    }
}

#[test]
fn ode_ss_resets_prior_state() {
    // SS=1 semantics: at the SS dose time, prior compartment state is
    // discarded and reset to the SS-train value. Build a subject with
    // a non-SS dose at t=0 (which would normally contribute decay
    // through to t=10) and an SS=1 dose at t=10. The post-SS-dose
    // observation must match the SS analytical formula evaluated at
    // tau = obs_time - 10, independent of the t=0 dose.
    use crate::pk::one_cpt_iv_bolus_ss;
    let cl = 5.0;
    let v = 80.0;
    let amt = 1000.0;
    let ii = 12.0;
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, amt, 1, 0.0, true, ii),
    ];
    let obs_times = vec![11.0, 14.0, 20.0];
    let subj = make_subject(doses.clone(), obs_times.clone());
    let pk = pk_one(cl, v);
    let ode = one_cpt_ode_spec();

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, &t) in obs_times.iter().enumerate() {
        let expected = one_cpt_iv_bolus_ss(&doses[1], t - 10.0, cl, v);
        assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
    }
}

#[test]
fn ode_ss_iv_bolus_with_lagtime_matches_nonmem() {
    // ODE-path coverage of SS + ALAG1 (issue #15). Reference PRED from
    // NONMEM 7.5.1 (ADVAN1 TRANS2, MAXEVAL=0): CL=5, V=80, ALAG1=2.0,
    // single SS=1 II=12 AMT=1000 IV bolus into the central compartment
    // (S1=V). Control file + dataset in tests/ss_lagtime_nonmem.rs.
    //
    // The first three samples (t=0.5,1.0,1.5 < ALAG1=2.0) exercise the
    // previous-interval steady-state tail seeded by `ss_state_at_phase`;
    // without the seed the ODE state would still be empty there (≈0).
    let cl = 5.0;
    let v = 80.0;
    let amt = 1000.0;
    let ii = 12.0;
    let lagtime = 2.0;
    let nonmem: &[(f64, f64)] = &[
        (0.5, 12.291),
        (1.0, 11.912),
        (1.5, 11.546),
        (2.0, 23.691),
        (3.0, 22.255),
        (6.0, 18.450),
        (11.0, 13.499),
        (13.0, 11.912),
        (18.0, 8.7153),
    ];
    let obs_times: Vec<f64> = nonmem.iter().map(|&(t, _)| t).collect();
    let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
    let subj = make_subject(vec![dose], obs_times);
    let mut pk = pk_one(cl, v);
    pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
    let ode = one_cpt_ode_spec();

    // one_cpt_ode_spec stores amount; divide by V for concentration.
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, &(_t, pred)) in nonmem.iter().enumerate() {
        assert_relative_eq!(preds[j] / v, pred, max_relative = 1e-4);
    }
}

// ── [scaling] Form C: output_fn replaces obs_cmt readout ────────────────

/// Same shape as `one_cpt_ode_spec` but the state holds `amount` (not
/// concentration), and `output_fn` produces concentration via `A/V`.
/// V is passed in via `pk_params_flat[PK_IDX_V]`.
fn one_cpt_ode_spec_amount_form() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            // dA/dt = -CL/V * A   (state is amount; same exp decay as
            // the concentration-baked spec)
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::Single(Box::new(
            |state: &[f64], pk: &[f64], _theta: &[f64], _eta: &[f64], _cov| {
                let v = pk[crate::types::PK_IDX_V];
                if v > 0.0 {
                    state[0] / v
                } else {
                    0.0
                }
            },
        )),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

#[test]
fn test_ode_output_fn_form_c_matches_concentration_form() {
    // Build two equivalent 1-cpt IV bolus ODE models:
    //   Reference: state = concentration; obs_cmt_idx = Some(0).
    //   Form C:    state = amount;        output_fn = state/V.
    //
    // The dose adds amt to state in both cases. In the reference, that
    // means state = amt directly equals the initial concentration AMT/V
    // ONLY if AMT/V already matches. To make the two truly equivalent
    // we have to scale the dose differently. Easier: pick V = 1.0 so
    // amount equals concentration numerically, and run an analytical
    // sanity check instead.
    let pk = pk_one(5.0, 1.0); // CL=5, V=1 → ke = 5
    let doses = vec![DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0, 0.5, 1.0, 2.0];
    let subj = make_subject(doses, obs_times.clone());

    let ode_ref = one_cpt_ode_spec();
    let ode_form_c = one_cpt_ode_spec_amount_form();

    let preds_ref = ode_predictions(&ode_ref, &pk.values, &[], &[], &subj);
    let preds_c = ode_predictions(&ode_form_c, &pk.values, &[], &[], &subj);

    // V = 1 makes amount/V numerically equal to amount, so both must agree.
    for (a, b) in preds_ref.iter().zip(preds_c.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-6, max_relative = 1e-6);
    }

    // And — crucially — Form C must produce a different numeric answer
    // when V differs from 1, demonstrating the readout actually divides
    // by V rather than ignoring it.
    let pk_v_5 = pk_one(5.0, 5.0); // CL=5, V=5 → ke = 1
    let preds_c_v5 = ode_predictions(&ode_form_c, &pk_v_5.values, &[], &[], &subj);
    // At t=0 just after the bolus, state = 10, V = 5 → conc = 2.
    assert_relative_eq!(preds_c_v5[0], 2.0, epsilon = 1e-9);
    // Reference (concentration-baked) with same params: state = 10
    // ⇒ conc = 10. Different from Form C, confirming output_fn ran.
    let preds_ref_v5 = ode_predictions(&ode_ref, &pk_v_5.values, &[], &[], &subj);
    assert!(
        (preds_ref_v5[0] - preds_c_v5[0]).abs() > 1.0,
        "output_fn must change the readout (ref={} c={})",
        preds_ref_v5[0],
        preds_c_v5[0]
    );
}

/// Regression for the co-temporal multi-CMT recorder bug: two observations
/// at the SAME time but different CMTs (simultaneous PK/PD sampling) must
/// BOTH be recorded. Before the fix, `obs_map` keyed by time alone kept
/// only one index per time and left the other observation at its initial
/// NaN.
#[test]
fn test_ode_predictions_records_cotemporal_multi_cmt() {
    // CMT=1 reads the compartment amount; CMT=2 reads twice that — two
    // distinct, finite readouts of the same single-state system, so we can
    // confirm each observation got its own value (not one overwriting the
    // other).
    let mut map: HashMap<usize, PerCmtReadout> = HashMap::new();
    map.insert(
        1,
        PerCmtReadout {
            out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| s[0]),
            program: None,
        },
    );
    map.insert(
        2,
        PerCmtReadout {
            out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| 2.0 * s[0]),
            program: None,
        },
    );
    let mut ode = one_cpt_ode_spec();
    ode.readout = OdeReadout::PerCmt(map);

    let pk = pk_one(5.0, 80.0);
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
    // Two obs at t=1 (CMT 1 and 2) and two at t=4 (CMT 1 and 2).
    let mut subj = make_subject(doses, vec![1.0, 1.0, 4.0, 4.0]);
    subj.obs_cmts = vec![1, 2, 1, 2];

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

    assert!(
        preds.iter().all(|p| p.is_finite()),
        "all co-temporal obs must be recorded (finite), got {preds:?}"
    );
    // CMT=2 readout is exactly twice CMT=1 at the same time.
    assert!((preds[1] - 2.0 * preds[0]).abs() < 1e-9);
    assert!((preds[3] - 2.0 * preds[2]).abs() < 1e-9);
}

/// Regression for Copilot review on PR #84: pre-Phase-2 the ODE paths
/// clamped NaN predictions to 0 at the end of `ode_predictions` (and
/// at the Obs branch of `ode_predictions_event_driven`). That defeated
/// the "loud failure" semantic for `OdeReadout::PerCmt` missing
/// entries (and for any other genuine NaN). The clamp now only
/// touches negatives; NaN propagates.
#[test]
fn test_ode_predictions_propagates_nan_from_readout() {
    // Build an OdeReadout::PerCmt that DELIBERATELY returns NaN for
    // CMT=1 — emulating a missing-CMT lookup that bypassed pre-fit
    // validation. The resulting prediction must be NaN, not 0.
    let mut map: HashMap<usize, PerCmtReadout> = HashMap::new();
    map.insert(
        1,
        PerCmtReadout {
            out_fn: Box::new(|_state: &[f64], _pk: &[f64], _theta, _eta, _cov| f64::NAN),
            program: None,
        },
    );
    let mut ode = one_cpt_ode_spec();
    ode.readout = OdeReadout::PerCmt(map);

    let pk = pk_one(5.0, 80.0);
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![1.0, 4.0];
    let subj = make_subject(doses, obs_times);

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, p) in preds.iter().enumerate() {
        assert!(
            p.is_nan(),
            "obs {} from a NaN-returning readout must be NaN, got {}",
            j,
            p
        );
    }
}

#[test]
fn test_ode_predictions_still_clamps_negatives() {
    // Sanity: dropping the NaN clamp must not change the negative
    // clamp behavior (ODE solver overshoot guard).
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        // dA/dt = -1 → state goes negative quickly with starting amount 1
        rhs: Box::new(|_y, _p, _t, dy| {
            dy[0] = -1.0;
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    };
    let pk = pk_one(1.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 1.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![10.0]; // dose=1, after 10s of -1/s → state = -9
    let subj = make_subject(doses, obs_times);

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert!(
        !preds[0].is_nan(),
        "negative readout must be clamped to 0, not NaN"
    );
    assert!(
        preds[0] >= 0.0,
        "negative readout must be clamped to 0, got {}",
        preds[0]
    );
}

/// A `clock` system (`d/dt clock = 1`, no doses) whose Form C readout is the
/// straight line `clock - 5`: exactly zero at t=5, legitimately negative before
/// it. The reprex of #1020, reduced to a Tier-1 spec.
fn signed_readout_clock_spec(readout: OdeReadout) -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
            dy[0] = 1.0;
        }),
        n_states: 1,
        state_names: vec!["clock".into()],
        readout,
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

/// The clamp is a property of the readout, not of the driver: `ObsCmt` (a bare
/// compartment amount, physically non-negative) clamps, both Form C `[scaling]`
/// variants (arbitrary user expressions) do not.
#[test]
fn only_bare_state_readouts_clamp_negatives() {
    assert!(OdeReadout::ObsCmt(0).clamps_negative());
    assert!(
        !OdeReadout::Single(Box::new(|s: &[f64], _p: &[f64], _t, _e, _c| s[0])).clamps_negative()
    );
    let mut map: HashMap<usize, PerCmtReadout> = HashMap::new();
    map.insert(
        1,
        PerCmtReadout {
            out_fn: Box::new(|s: &[f64], _p: &[f64], _t, _e, _c| s[0]),
            program: None,
        },
    );
    assert!(!OdeReadout::PerCmt(map).clamps_negative());
}

/// Regression for #1020: a Form C `[scaling] y[CMT=N] = <expr>` readout is an
/// arbitrary user expression (change from baseline, z-score, `sqrt(N)*logit(p)`)
/// and its negative part must reach the caller, not be silently zeroed by the
/// ODE overshoot guard.
#[test]
fn form_c_per_cmt_readout_keeps_negative_predictions() {
    let mut map: HashMap<usize, PerCmtReadout> = HashMap::new();
    map.insert(
        1,
        PerCmtReadout {
            out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| s[0] - 5.0),
            program: None,
        },
    );
    let ode = signed_readout_clock_spec(OdeReadout::PerCmt(map));
    let pk = pk_one(1.0, 1.0);
    // Zero-amount bolus at t=0, as in the issue's reprex: it anchors the
    // timeline at 0 (a dose-free subject starts integrating at its first
    // observation) without perturbing the clock state.
    let doses = vec![DoseEvent::new(0.0, 0.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![2.0, 4.0, 6.0, 10.0]);

    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (got, want) in preds.iter().zip([-3.0, -1.0, 1.0, 5.0]) {
        assert!(
            (got - want).abs() < 1e-6,
            "signed Form C readout must not be clamped: got {preds:?}"
        );
    }
}

/// The uniform Form C variant (`[scaling] y = <expr>`) behaves the same, on
/// every driver that shares the clamp epilogue: the dense predictor, the
/// dense-with-states predictor, and the event-driven walk (whose clamp is
/// inline, per observation).
#[test]
fn form_c_single_readout_keeps_negative_predictions_on_all_drivers() {
    let ode = signed_readout_clock_spec(OdeReadout::Single(Box::new(
        |s: &[f64], _pk: &[f64], _t: &[f64], _e: &[f64], _c: &HashMap<String, f64>| s[0] - 5.0,
    )));
    let pk = pk_one(1.0, 1.0);
    let doses = vec![DoseEvent::new(0.0, 0.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![2.0, 10.0]);
    let want = [-3.0, 5.0];

    let dense = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    let (with_states, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    let dose_pk = vec![pk; subj.doses.len()];
    let obs_pk = vec![pk; subj.obs_times.len()];
    let event_driven =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &dose_pk, &obs_pk, &[], &[]);

    for (name, preds) in [
        ("dense", &dense),
        ("with_states", &with_states),
        ("event_driven", &event_driven),
    ] {
        for (got, w) in preds.iter().zip(want) {
            assert!(
                (got - w).abs() < 1e-6,
                "{name}: signed Form C readout must not be clamped, got {preds:?}"
            );
        }
    }
}

/// Helper: oral PK params with clearance, volume, ka, and bioavailability.
fn pk_oral_f(cl: f64, v: f64, ka: f64, f: f64) -> PkParams {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = cl;
    p.values[crate::types::PK_IDX_V] = v;
    p.values[crate::types::PK_IDX_KA] = ka;
    p.values[crate::types::PK_IDX_F] = f;
    p
}

#[test]
fn ode_applies_f_bio_to_bolus_dose() {
    // Issue #122: the ODE engine must load the depot with F·AMT (NONMEM
    // convention), not the full AMT. For this linear oral system the
    // central readout is exactly proportional to the depot load, so a
    // bioavailability of F = 0.5 must halve every prediction relative to
    // F = 1.0. Covers both the plain and event-driven paths.
    let ode = one_cpt_oral_ode_spec();
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    let subj = make_subject(doses, obs_times.clone());

    let pk_full = pk_oral_f(5.0, 50.0, 1.5, 1.0);
    let pk_half = pk_oral_f(5.0, 50.0, 1.5, 0.5);

    // Plain (non-TV) path.
    let full = ode_predictions(&ode, &pk_full.values, &[], &[], &subj);
    let half = ode_predictions(&ode, &pk_half.values, &[], &[], &subj);
    for (f, h) in full.iter().zip(half.iter()) {
        assert!(*f > 0.0, "expected positive prediction");
        assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
    }

    // Event-driven path.
    let pk_dose_full = vec![pk_full; subj.doses.len()];
    let pk_obs_full = vec![pk_full; obs_times.len()];
    let pk_dose_half = vec![pk_half; subj.doses.len()];
    let pk_obs_half = vec![pk_half; obs_times.len()];
    let ed_full =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose_full, &pk_obs_full, &[], &[]);
    let ed_half =
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose_half, &pk_obs_half, &[], &[]);
    for (f, h) in ed_full.iter().zip(ed_half.iter()) {
        assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
    }
}

#[test]
fn ode_applies_f_bio_to_infusion() {
    // A rate-defined infusion under F holds the rate and scales the duration
    // (#419): F=0.5 on (AMT=100, rate=50, T=2h) delivers rate 50 over 1h -
    // identical to a full-F infusion of F·AMT=50 at rate 50, NOT 0.5x the F=1
    // curve.
    let ode = one_cpt_oral_ode_spec();
    let rate = 50.0;
    let obs_times = vec![1.0, 2.0, 4.0, 8.0];
    let preds = |amt: f64, f: f64| {
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let subj = make_subject(doses, obs_times.clone());
        ode_predictions(&ode, &pk_oral_f(5.0, 50.0, 1.5, f).values, &[], &[], &subj)
    };
    let full = preds(100.0, 1.0);
    let half_f = preds(100.0, 0.5);
    let equiv = preds(50.0, 1.0); // F=1, F·AMT delivered at the same rate
    for ((f, hf), e) in full.iter().zip(half_f.iter()).zip(equiv.iter()) {
        assert!(*f > 0.0, "expected positive prediction");
        assert_relative_eq!(*hf, *e, epsilon = 1e-9, max_relative = 1e-6);
    }
    assert!(
        half_f
            .iter()
            .zip(full.iter())
            .any(|(h, f)| (*h - 0.5 * *f).abs() > 1e-6),
        "rate-defined infusion under F must reshape, not scale"
    );
}

#[test]
fn ode_applies_f_bio_to_ss_dose() {
    // Steady-state pre-equilibration must also load F·AMT each cycle, so a
    // halved F halves the steady-state predictions.
    let ode = one_cpt_oral_ode_spec();
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    let obs_times = vec![1.0, 4.0, 8.0, 11.0];
    let subj = make_subject(doses, obs_times);

    let full = ode_predictions(
        &ode,
        &pk_oral_f(5.0, 50.0, 1.5, 1.0).values,
        &[],
        &[],
        &subj,
    );
    let half = ode_predictions(
        &ode,
        &pk_oral_f(5.0, 50.0, 1.5, 0.5).values,
        &[],
        &[],
        &subj,
    );
    for (f, h) in full.iter().zip(half.iter()) {
        assert!(*f > 0.0, "expected positive SS prediction");
        assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
    }
}

// -----------------------------------------------------------------------
// Regression tests for ode_predictions_with_states / ode_dense_solve_states
// -----------------------------------------------------------------------

/// Bug regression: state must be advanced through segments that contain no
/// observations (the t_end push).  Before the fix, `sol.last()` returned
/// `None` for an empty saveat and `u` was not updated, so all subsequent
/// compartment states were wrong.
#[test]
fn ode_with_states_advances_through_empty_segment() {
    // Two doses, observations only after the second.  The segment [0, 12)
    // has no obs — the state must still decay correctly through it.
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let ode = one_cpt_ode_spec();
    let pk = pk_one(cl, v);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 50.0, 1, 0.0, false, 0.0),
    ];
    let obs_times = vec![24.0];
    let subj = make_subject(doses, obs_times);
    let (preds, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    // Compare against the full ode_predictions path.
    let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert!(
        approx::relative_eq!(preds[0], preds_ref[0], max_relative = 1e-6),
        "ipred diverges — state was not advanced through the empty segment"
    );
    // State[0] must be positive (non-zero drug remaining).
    assert!(
        states[0][0] > 0.0 && states[0][0].is_finite(),
        "compartment state is wrong after empty inter-dose segment: {}",
        states[0][0]
    );
}

/// #731 regression: a dose landing exactly on the LAST observation must be applied
/// before that observation is read (post-dose) on the states path too — the doc on
/// `ode_predictions_with_states` requires it to mirror `ode_predictions`, and its
/// `windows(2)` loop used to treat the final break as an endpoint only, dropping a
/// terminal dose. Both the returned ipred and the compartment state must match the
/// (fixed) primary path, i.e. read the fresh post-dose bolus.
#[test]
fn ode_with_states_applies_dose_on_last_observation() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0); // ke = CL/V = 0.1
    let ke = 0.1_f64;
    let subj = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0), // lands on the last obs
        ],
        vec![6.0, 24.0],
    );
    let (preds, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    // ipred must match the fixed primary path at both observations.
    for i in 0..2 {
        assert_relative_eq!(preds[i], preds_ref[i], max_relative = 1e-6);
    }
    // And t=24 is genuinely post-dose: decayed t=0 bolus + fresh 100 mg (≈109.07),
    // not the pre-dose ≈9.07 the dropped-dose bug produced. State[0] carries it.
    assert_relative_eq!(
        states[1][0],
        100.0 * (-ke * 24.0).exp() + 100.0,
        max_relative = 1e-5
    );
}

/// Bug regression: a CMT past the end of the state vector must be ignored by
/// both new functions, matching `ode_predictions` behaviour. Before the original
/// fix, `saturating_sub(1).min(n-1)` applied the dose to the *last* compartment
/// instead. (Rejecting it up front is `check_dose_compartments`' job since #899;
/// this pins the engine-level fallback, which these `pub fn`s still need because
/// they are reachable from hand-built `OdeSpec`s that run no validation.)
#[test]
fn ode_with_states_ignores_out_of_range_cmt() {
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let ode = one_cpt_ode_spec();
    let pk = pk_one(cl, v);
    let dose_valid = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    // CMT=2 — past the end of a 1-state ODE (its only state is CMT 1).
    let dose_oor = DoseEvent::new(0.0, 999.0, 2, 0.0, false, 0.0);
    let obs_times = vec![4.0, 12.0];

    let subj_ref = make_subject(vec![dose_valid.clone()], obs_times.clone());
    let subj_oor = make_subject(vec![dose_valid.clone(), dose_oor], obs_times.clone());

    let (preds_ref, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_ref);
    let (preds_oor, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_oor);
    for j in 0..obs_times.len() {
        assert!(
            approx::relative_eq!(preds_ref[j], preds_oor[j], max_relative = 1e-9),
            "obs {j}: out-of-range dose was applied (got {}) instead of being ignored \
             (expected {})",
            preds_oor[j],
            preds_ref[j]
        );
    }
}

/// `CMT=0` is **not** out of range — it is NONMEM's default dose compartment and
/// resolves to compartment 1 (#899). This function used to skip it outright,
/// while `ode_predictions_event_driven` on the identical dataset applied it to
/// compartment 1 and the plain `ode_predictions` segment loop underflowed
/// (debug panic / release silent drop). All three now agree, so a `CMT=0` dose
/// must be indistinguishable from the same dose written `CMT=1`.
#[test]
fn ode_with_states_applies_cmt_zero_to_the_default_compartment() {
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let ode = one_cpt_ode_spec();
    let pk = pk_one(cl, v);
    let obs_times = vec![4.0, 12.0];

    let subj_one = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times.clone(),
    );
    let subj_zero = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 0, 0.0, false, 0.0)],
        obs_times.clone(),
    );

    let (preds_one, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_one);
    let (preds_zero, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_zero);
    for j in 0..obs_times.len() {
        assert!(
            approx::relative_eq!(preds_one[j], preds_zero[j], max_relative = 1e-12),
            "obs {j}: CMT=0 should dose the default compartment like CMT=1 \
             (got {} vs {})",
            preds_zero[j],
            preds_one[j]
        );
    }
    // Guard against the assertion passing vacuously on an all-zero curve.
    assert!(preds_one[0] > 0.0, "reference curve should be non-trivial");
}

/// Bug regression: TAD for SS doses must be computed with rem_euclid so it
/// stays within [0, II).  Before the fix the raw elapsed time was used,
/// injecting a growing TAD into the ODE RHS.  This test uses an ODE that
/// writes TAD into its output so we can verify it.
#[test]
fn ode_with_states_tad_stays_within_dosing_interval_for_ss() {
    // ODE: dA/dt = -ke*A; but we read TAD = t - ext_params[TAD_SLOT] back
    // as the compartment state for the diagnostic.  Use a second-state ODE:
    //   dA/dt = -ke*A
    //   dT/dt = 0  (T is just a placeholder; we seed it externally via the
    //               TAD anchor update, which is non-state, so we use ipred)
    // Actually simplest: verify ode_predictions and ode_predictions_with_states
    // agree on ipred for an SS dose observed beyond one II, because the TAD
    // error only shows up when TAD modulates the ODE.
    //
    // For a pure 1-cpt IV where the RHS does NOT use TAD, both paths must
    // agree with the closed-form SS regardless of the TAD anchor.
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let ii = 24.0_f64;
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii);
    // Observations beyond one dosing interval.
    let obs_times = vec![0.5, 6.0, 24.0, 30.0, 48.0, 53.0];
    let subj = make_subject(vec![dose.clone()], obs_times.clone());
    let pk = pk_one(cl, v);
    let ode = one_cpt_ode_spec();

    let (preds_ws, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, &t) in obs_times.iter().enumerate() {
        // ipred must agree with ode_predictions (which uses rem_euclid for TAD).
        assert!(
            approx::relative_eq!(preds_ws[j], preds_ref[j], max_relative = 1e-6),
            "ipred diverges at t={t} — TAD anchor mismatch for SS dose"
        );
        // For ObsCmt(0) readout, ipred == u[0] == state[0], so they must agree.
        assert!(
            approx::relative_eq!(states[j][0], preds_ws[j], max_relative = 1e-9),
            "state != ipred at t={t} — state not self-consistent with ipred"
        );
    }
}

/// Bug regression: for an SS dose with lagtime > 0, the pre-lag break
/// point at dose.time must seed ss_state_at_phase so observations before
/// the lagged pulse see the correct pre-lag SS tail rather than zero.
/// Before the fix the merged dose loop fired only at dose.time + lagtime,
/// leaving the pre-lag segment with an all-zero initial state.
#[test]
fn ode_with_states_ss_lagtime_preseed_is_correct() {
    let cl = 5.0_f64;
    let v = 80.0_f64;
    let ii = 24.0_f64;
    let lagtime = 2.0_f64;
    // SS dose at t=0 with lagtime=2; observations at t=0.5 and t=1.5
    // (both before the lagged arrival at t=2) should see the SS tail
    // from the prior cycle.
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii);
    let mut pk = pk_one(cl, v);
    pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
    let obs_times = vec![0.5, 1.5, 3.0, 12.0];
    let subj = make_subject(vec![dose.clone()], obs_times.clone());
    let ode = one_cpt_ode_spec();

    let (preds_ws, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
    let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    for (j, (&t, &p_ws)) in obs_times.iter().zip(preds_ws.iter()).enumerate() {
        assert!(
            approx::relative_eq!(p_ws, preds_ref[j], max_relative = 1e-6),
            "ipred diverges at t={t} — SS+lagtime pre-lag seeding missing"
        );
        // Pre-lag obs (t < lagtime) must be > 0 (from prior SS cycle).
        if t < lagtime {
            assert!(
                states[j][0] > 0.0,
                "state is zero at t={t} (before lag) — SS tail was not pre-seeded"
            );
        }
    }
}

#[test]
fn adaptive_observe_expression_flows_through_driver() {
    // The model readout is the raw `central` amount (`[scaling] y = central`),
    // but the declarative controller observes `central / V` (concentration).
    // The driver must feed the controller the compiled expression's value, not
    // the cmt readout — this exercises the S2.2 `observe_exprs` path.
    const M: &str = r#"
[parameters]
  theta TVCL(1.0)
  theta TVV(50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [24, 48]
  start_dose = 100
  route = bolus(cmt = 1)
  dose_bounds = [0, 400]
  when signal > 1000 : decrease 25%
"#;
    let parsed = crate::parser::model_parser::parse_full_model(M).expect("parses");
    let model = parsed.model;
    let spec = parsed.adaptive_dosing.expect("has adaptive block");
    let compiled = crate::sim::adaptive_control::compile_adaptive(&model, &spec).expect("compiles");
    let theta = model.default_params.theta.clone();
    let eta = vec![0.0; model.n_eta + model.n_kappa];
    let pk = (model.pk_param_fn)(&theta, &eta, &HashMap::new(), 0.0);
    let subject = make_subject(vec![], spec.at.clone());
    let mut controller = (compiled.make_controller)();
    let monitors = vec![crate::sim::adaptive::AdaptiveMonitor {
        spec: &compiled.monitors[0],
        observe: compiled.observe.as_ref(),
    }];
    let run = ode_predictions_adaptive_impl(
        model.ode_spec.as_ref().unwrap(),
        &pk.values,
        None,
        None,
        None,
        &theta,
        &eta,
        &subject,
        &spec.at,
        &monitors,
        &mut controller,
        spec.at.len() + 1,
        None,
    )
    .expect("driver runs");

    // Decision 0 is the pre-dose trough (central = 0 ⇒ concentration 0).
    assert_eq!(run.decisions[0].observed_signals[0].value, 0.0);
    // Decision 1: one bolus of start_dose decayed over Δt, divided by V — the
    // CONCENTRATION. Reading the cmt amount instead would be ~50× larger (the
    // raw `central`), so this pins the expression path.
    let ke = theta[0] / theta[1]; // CL/V at eta = 0
    let dt = spec.at[1] - spec.at[0];
    let expected_conc = spec.start_dose * (-ke * dt).exp() / theta[1];
    let got = run.decisions[1].observed_signals[0].value;
    assert!(
        (got - expected_conc).abs() < 1e-3,
        "observed {got}, expected concentration {expected_conc} (raw amount would be ~{})",
        expected_conc * theta[1]
    );
}

/// A 1-cpt Michaelis–Menten SS-absorption `OdeSpec` (`first_order(ka)` into MM elimination),
/// with `Vmax`/`Km`/`ka` in the CL/V/slot-4 positions the tests set. Shared by the #867
/// nonlinear-SS tests.
fn mm_ss_absorption_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        // Vmax reuses the CL slot, Km the V slot; `first_order` ka at slot 4.
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let vmax = p[crate::types::PK_IDX_CL];
            let km = p[crate::types::PK_IDX_V];
            dy[0] = -vmax * y[0] / (km + y[0]);
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![4],
            frac_slot: None,
            lag_slot: None,
        }],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    }
}

/// Explicit ground-truth trough: integrate a run-in of past pulses (non-SS, at 0, II, 2II, …)
/// from a zero state until the pre-next-pulse trough stops moving — the #867 evidence table's
/// methodology, a genuine periodic-SS reference independent of the solver under test. Returns
/// `(trough, cycles_used)`; `cycles_used == max_cycles` means it did **not** converge (the
/// caller asserts otherwise, so the reference is trustworthy). Each pulse's absorption tail is
/// re-evaluated by absolute age, so a tail longer than `II` keeps contributing across cycles.
fn explicit_ss_run_in(
    ode: &OdeSpec,
    pk: &[f64],
    dose: &DoseEvent,
    max_cycles: usize,
) -> (Vec<f64>, usize) {
    let prepared = prepare_input_rates(ode, pk);
    let doses: Vec<DoseEvent> = (0..max_cycles)
        .map(|m| {
            DoseEvent::new(
                m as f64 * dose.ii,
                dose.amt,
                dose.cmt_raw(),
                0.0,
                false,
                0.0,
            )
        })
        .collect();
    let fbios = vec![1.0; max_cycles];
    let no_lag: [f64; 0] = [];
    let no_zero: [(usize, f64); 0] = [];
    let train = wrap_rhs_with_forcings(
        ode,
        &doses,
        &no_lag,
        &fbios,
        f64::NEG_INFINITY,
        &prepared,
        InfusionInput::Spanning(Vec::new()),
        &no_zero,
    );
    let eq_opts = ss_equilibration_opts(&ode.solver_opts);
    let mut u = vec![0.0; ode.n_states];
    for m in 0..max_cycles {
        let prev = u.clone();
        let seg = (m as f64 * dose.ii, (m + 1) as f64 * dose.ii);
        let sol = solve_ode(&train, &u, seg, pk, &[seg.1], &eq_opts);
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        // Converged once the pre-pulse trough stops moving (relative-L∞ increment tiny).
        let mag = u.iter().fold(0.0_f64, |a, &x| a.max(x.abs())).max(1e-300);
        let inc = u
            .iter()
            .zip(&prev)
            .fold(0.0_f64, |a, (&c, &p)| a.max((c - p).abs()));
        if m > 0 && inc / mag < 1e-9 {
            return (u, m + 1);
        }
    }
    (u, max_cycles)
}

/// #867 fix: a slowly-accumulating saturable (Michaelis–Menten) SS-absorption model whose plain
/// pulse-train needs **more than the 50-cycle cap** to converge is nonetheless solved to the
/// true periodic steady state by the Anderson-accelerated fixed point. The trough must (a) match
/// a self-certified explicit run-in (the genuine SS), (b) be reached in fewer cycles than that
/// run-in — and specifically inside the cap that would have left the plain iteration short — and
/// (c) emit **no** non-convergence warning.
#[test]
fn ss_input_rate_heavy_accumulation_converges_via_anderson() {
    // Proves convergence via the thread-local cycle counter (`last_ss_equilibration_cycles`),
    // not the process-global warning sink, so it needs no cross-test serialization.
    let ode = mm_ss_absorption_spec();
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 20.0; // Vmax
    pk.values[crate::types::PK_IDX_V] = 100.0; // Km
    pk.values[4] = 2.0; // ka
    pk.values[crate::types::PK_IDX_F] = 1.0;
    // Mean absorbed input 130/8 ≈ 16.3 < Vmax = 20 but deep into saturation (C_ss ≫ Km): the
    // per-cycle carryover ρ ≈ 0.94, so the plain iteration needs a few hundred cycles — far past
    // the 50-cycle cap that silently under-converged pre-#867.
    let ss = DoseEvent::new(0.0, 130.0, 1, 0.0, true, 8.0);

    // Self-certified reference: a plain run-in iterated until the trough stops moving.
    let (truth, runin_cycles) = explicit_ss_run_in(&ode, &pk.values, &ss, 3000);
    assert!(
        runin_cycles < 3000,
        "run-in reference must converge to be trustworthy (used {runin_cycles})"
    );
    assert!(
        runin_cycles > SS_EQUILIBRATION_CYCLES,
        "test model must genuinely exceed the {SS_EQUILIBRATION_CYCLES}-cycle cap (run-in used \
         {runin_cycles}) — otherwise it would not exercise the pre-#867 under-convergence"
    );

    let prepared = prepare_input_rates(&ode, &pk.values);
    let aa = equilibrate_ss_input_rate(&ode, &pk.values, &ss, 1.0, &ode.solver_opts, &prepared)
        .expect("Anderson must converge for a disposition that admits a steady state");
    // Anderson reached the fixed point in far fewer cycles than the plain run-in, and inside the
    // cap that left the plain iteration short.
    let aa_cycles = crate::dosing::last_ss_equilibration_cycles();
    assert!(
        (2..runin_cycles).contains(&aa_cycles),
        "Anderson ({aa_cycles} cycles) must beat the plain run-in ({runin_cycles})"
    );
    // Matches the true periodic SS to solver precision (the 50-cap value would be materially low).
    let rel = (aa[0] - truth[0]).abs() / truth[0];
    assert!(
        rel < 1e-2,
        "Anderson trough {:.3} vs run-in {:.3} (rel {rel:.2e})",
        aa[0],
        truth[0]
    );
}

/// #867 regression: a **deeply-saturated** over-capacity model (mean input just above `Vmax`,
/// `Km` huge) has no periodic steady state, but the per-cycle surplus `Δ = (mean input − Vmax)·II`
/// is tiny next to the single-period seed — so Anderson can extrapolate the divergent map to a
/// huge (even negative) iterate where `Δ` hides beneath the magnitude-relative tolerance and
/// false-trips convergence. Before the seed-scale residual + non-negativity guards these returned
/// `Some(huge_or_negative)` with no warning; now every one must decline (`None`) so the caller
/// warns. (The shallow `Km` case in `ss_input_rate_no_steady_state_warns` declined even without
/// the guards — these are the cases that did not.)
#[test]
fn ss_input_rate_over_capacity_deep_saturation_declines() {
    // (vmax, km, ka, amt, ii): mean = amt/ii > vmax (no SS); Km huge (deep saturation) so the
    // per-cycle surplus Δ ≪ seed — the false-convergence regime.
    let configs = [
        (10.0, 5000.0, 2.0, 82.0, 8.0),
        (10.0, 20000.0, 2.0, 85.0, 8.0),
        (10.0, 50000.0, 3.0, 81.0, 8.0),
        (20.0, 50000.0, 2.0, 165.0, 8.0),
        (10.0, 200000.0, 3.0, 80.5, 8.0),
    ];
    for (vmax, km, ka, amt, ii) in configs {
        let ode = mm_ss_absorption_spec();
        let mut pk = PkParams::default();
        pk.values[crate::types::PK_IDX_CL] = vmax;
        pk.values[crate::types::PK_IDX_V] = km;
        pk.values[4] = ka;
        pk.values[crate::types::PK_IDX_F] = 1.0;
        let ss = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let prepared = prepare_input_rates(&ode, &pk.values);
        let res =
            equilibrate_ss_input_rate(&ode, &pk.values, &ss, 1.0, &ode.solver_opts, &prepared);
        assert!(
            res.is_none(),
            "over-capacity vmax={vmax} km={km} amt={amt} (mean {:.2} > vmax) has no SS: the \
             solve must decline, not return a spurious trough {res:?}",
            amt / ii
        );
    }
}

/// #867 boundary: when the mean input rate meets or exceeds the maximum elimination rate
/// (`amt/II = 6.25 > Vmax = 5`), *no* periodic steady state exists — the per-cycle map is not a
/// contraction. The Anderson solve must decline (not manufacture a false fixed point or hang),
/// the caller falls to the capped pulse train, and the non-convergence warning fires.
#[test]
fn ss_input_rate_no_steady_state_warns() {
    let _guard = crate::dosing::SS_WARN_SINK_READER_GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    crate::dosing::clear_ss_nonconvergence_warnings();
    let ode = mm_ss_absorption_spec();
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 5.0; // Vmax
    pk.values[crate::types::PK_IDX_V] = 8.0; // Km
    pk.values[4] = 0.5; // ka
    pk.values[crate::types::PK_IDX_F] = 1.0;
    let ss = DoseEvent::new(0.0, 50.0, 1, 0.0, true, 8.0); // mean input 6.25 > Vmax 5

    let prepared = prepare_input_rates(&ode, &pk.values);
    assert!(
        equilibrate_ss_input_rate(&ode, &pk.values, &ss, 1.0, &ode.solver_opts, &prepared)
            .is_none(),
        "no steady state exists (input ≥ Vmax): the solve must decline"
    );

    let subj = make_subject(vec![ss], vec![1.0, 4.0, 7.9]);
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "predictions must stay finite: {preds:?}"
    );
    // A warning must fire (both the ρ ≥ 1 "no steady state" text and the near-ρ = 1 "below the
    // true periodic steady state" text are valid here — the capped drift can estimate ρ either
    // side of 1 — so assert on the shared prefix rather than one branch).
    let warnings = crate::dosing::take_ss_nonconvergence_warnings();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("Steady-state (SS=1) equilibration")),
        "an input-above-capacity model must surface a non-convergence warning; got: {warnings:?}"
    );
}

/// The Anderson solve carries analytic sensitivities: run over a dual it must return the
/// *implicit-function derivative* `∂u*/∂θ` of the fixed point, not merely the value (#867). This
/// isolates that mechanism on a scalar nonlinear contraction `P(u; θ) = θ/(1 + u)` whose fixed
/// point and derivative are known in closed form — `u* = (−1 + √(1+4θ))/2`,
/// `∂u*/∂θ = 1/√(1+4θ)` — and cross-checks against a finite difference of the f64 solve. If the
/// AA mixing coefficients (chosen on the value residual) were applied incorrectly to the dual
/// jets, the derivative would be wrong even though the value converged.
#[test]
fn ss_input_rate_nonlinear_dual_gradient_matches_fd() {
    use crate::sens::dual1::Dual1;
    let theta_val = 2.0_f64;
    // P(u; θ) = θ / (1 + u), with θ seeded as the single differentiation variable.
    let advance_dual = |u: &[Dual1<1>]| -> Option<Vec<Dual1<1>>> {
        let theta = Dual1::<1>::var(theta_val, 0);
        Some(vec![theta / (Dual1::<1>::constant(1.0) + u[0])])
    };
    let u_star = anderson_ss_fixed_point_g::<Dual1<1>, _>(1, 1.0, 1e-12, 1e-14, &advance_dual)
        .expect("a scalar contraction converges");
    let analytic_deriv = 1.0 / (1.0 + 4.0 * theta_val).sqrt(); // 1/3
    assert!(
        (u_star[0].value - 1.0).abs() < 1e-9,
        "fixed-point value {} != 1",
        u_star[0].value
    );
    assert!(
        (u_star[0].grad[0] - analytic_deriv).abs() < 1e-5,
        "dual ∂u*/∂θ {} vs analytic {analytic_deriv}",
        u_star[0].grad[0]
    );
    // Cross-check the dual derivative against a central FD of the f64 fixed point.
    let f64_fp = |th: f64| {
        let adv = |u: &[f64]| -> Option<Vec<f64>> { Some(vec![th / (1.0 + u[0])]) };
        anderson_ss_fixed_point_g::<f64, _>(1, 1.0, 1e-12, 1e-14, &adv).unwrap()[0]
    };
    let h = 1e-6;
    let fd = (f64_fp(theta_val + h) - f64_fp(theta_val - h)) / (2.0 * h);
    assert!(
        (u_star[0].grad[0] - fd).abs() < 1e-4,
        "dual ∂u*/∂θ {} vs FD {fd}",
        u_star[0].grad[0]
    );
}

/// #867 unit guards on `anderson_ss_fixed_point_g` itself (no ODE): a divergent map must be
/// declined via the divergence ceiling / non-contraction, and the degenerate arguments return
/// `None`. Uses synthetic scalar maps so it stays in the fast-test budget.
#[test]
fn anderson_declines_divergent_and_degenerate_maps() {
    // Expanding affine map P(u) = 2u + 1: no fixed point above the seed, the iterate blows up
    // past `diverged_ceiling` (or is caught as non-contracting) → None, never a false SS.
    let diverging = |u: &[f64]| -> Option<Vec<f64>> { Some(vec![2.0 * u[0] + 1.0]) };
    assert!(
        anderson_ss_fixed_point_g::<f64, _>(1, 1.0, 1e-9, 1e-12, &diverging).is_none(),
        "an expanding map has no periodic SS and must decline"
    );
    // A constant-surplus saturated map P(u) = u + 5: the value residual is a fixed 5 that never
    // clears the seed-scale bound, so it must not false-converge to a huge iterate.
    let surplus = |u: &[f64]| -> Option<Vec<f64>> { Some(vec![u[0] + 5.0]) };
    assert!(
        anderson_ss_fixed_point_g::<f64, _>(1, 1.0, 1e-9, 1e-12, &surplus).is_none(),
        "a constant per-cycle surplus (no SS) must decline, not inflate to a false trough"
    );
    // Degenerate arguments.
    let ok = |u: &[f64]| -> Option<Vec<f64>> { Some(vec![u[0]]) };
    assert!(anderson_ss_fixed_point_g::<f64, _>(1, 0.0, 1e-9, 1e-12, &ok).is_none());
    assert!(anderson_ss_fixed_point_g::<f64, _>(0, 1.0, 1e-9, 1e-12, &ok).is_none());
}

/// Calibration guard for the `256·reltol` verification bound (#835): a `first_order(ka)`
/// absorption into a **linear** 1-cpt disposition MUST accept the closed-form fixed point —
/// and the full equilibration record exactly one cycle — even at a tight `ode_reltol` (1e-10).
/// The linear one-cycle residual is a solver-noise floor (≈ 45·reltol, from the two forced
/// solves taking different adaptive step sequences); the earlier `32·reltol` bound sat *below*
/// that floor, so the fast path was silently abandoned to the 50-cycle iteration at every
/// realistic `reltol`. Without the fix this fails (`is_none()` / 50 cycles).
#[test]
fn ss_input_rate_linear_disposition_uses_fixed_point() {
    let mut solver_opts = OdeSolverOptions::default();
    solver_opts.reltol = 1e-10;
    solver_opts.abstol = 1e-10;
    let ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            dy[0] = -(cl / v) * y[0];
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts,
        input_rate: vec![InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![4],
            frac_slot: None,
            lag_slot: None,
        }],
        init_fn: None,
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
    };
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 1.0;
    pk.values[crate::types::PK_IDX_V] = 20.0;
    pk.values[4] = 0.15; // slow ka: t½,abs ≈ II, so the fixed point must handle carryover
    pk.values[crate::types::PK_IDX_F] = 1.0;
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 8.0);

    let prepared = prepare_input_rates(&ode, &pk.values);
    assert!(
        equilibrate_ss_input_rate(&ode, &pk.values, &ss, 1.0, &ode.solver_opts, &prepared)
            .is_some(),
        "a linear disposition must accept the closed-form SS fixed point at tight reltol"
    );
    // End-to-end: the full equilibration takes the fast path (one recorded cycle).
    let subj = make_subject(vec![ss], vec![1.0, 4.0, 7.9]);
    let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
    assert_eq!(
        crate::dosing::last_ss_equilibration_cycles(),
        1,
        "linear SS-absorption must equilibrate via the closed-form fixed point"
    );
    assert!(preds.iter().all(|p| p.is_finite() && *p >= 0.0));
}

// ---------------------------------------------------------------------------
// #1073 — which record's parameters govern each segment.
//
// NONMEM runs `$PK` at every data record and then ADVANs *to* that record, so a
// segment is governed by the record that TERMINATES it. Events that are not
// records — a lagged dose arrival, an infusion end, a zero-order cutoff, a
// per-route onset — subdivide the interval their enclosing record terminates and
// carry no parameters of their own.
//
// These assert the *segment parameters* through a hand-computed closed form
// rather than an aggregated objective: an OFV moves for either half of the fix
// and isolates neither. Each geometry gives a different clearance to the interval
// under test than to its neighbours, so a segment taking the wrong record's
// snapshot lands on a different number, not merely a different digit.
// ---------------------------------------------------------------------------

/// 1-cpt IV with a bare lagtime on the dose row's own snapshot.
fn pk_one_lagged(cl: f64, v: f64, lag: f64) -> PkParams {
    let mut p = pk_one(cl, v);
    p.values[crate::types::PK_IDX_LAGTIME] = lag;
    p
}

/// `one_cpt_ode_spec` at tolerances tight enough for the hand-computed closed
/// forms below to be a genuine oracle rather than a two-digit sanity check: the
/// default `1e-6`/`1e-8` leaves ~2e-6 of relative solver error, which is larger
/// than several of the quantities under test deserve. `Rk45` is pinned so the
/// `auto` stepper probe cannot silently change what these assert.
fn one_cpt_ode_spec_tight() -> OdeSpec {
    let mut ode = one_cpt_ode_spec();
    ode.solver_opts.reltol = 1e-12;
    ode.solver_opts.abstol = 1e-14;
    ode.solver_opts.method = crate::ode::OdeMethod::Rk45;
    ode
}

#[test]
fn lagged_arrival_segment_runs_on_the_next_record_not_the_dose_row() {
    // Two doses so the arrival lands with residual drug present: a single dose
    // cannot test this at all, because the state is zero before a first arrival
    // and the interval `[dose record, arrival]` propagates nothing.
    //
    //   t=0   dose (CL=20)      t=5  obs (CL=20)
    //   t=10  dose row (CL=20, ALAG=2)  ->  arrival at t=12
    //   t=20  obs (CL=5)
    //
    // `(10, 12]` is terminated by the t=20 observation, so it runs at CL=5. The
    // dose row's CL=20 governs the advance INTO t=10 and nothing after it.
    let ode = one_cpt_ode_spec_tight();
    let (v, cl_a, cl_b) = (100.0, 20.0, 5.0);
    let (ka, kb) = (cl_a / v, cl_b / v);

    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![5.0, 20.0]);
    let pk_dose = vec![pk_one_lagged(cl_a, v, 0.0), pk_one_lagged(cl_a, v, 2.0)];
    let pk_obs = vec![pk_one(cl_a, v), pk_one(cl_b, v)];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);

    // (0,10] at CL=20; (10,12] at CL=5 (the t=20 record); bolus; (12,20] at CL=5.
    let a12 = 1000.0 * (-10.0 * ka).exp() * (-2.0 * kb).exp();
    let expected = (a12 + 1000.0) * (-8.0 * kb).exp();
    // What stretching the dose row's snapshot to the arrival would have given.
    let stale = (1000.0 * (-12.0 * ka).exp() + 1000.0) * (-8.0 * kb).exp();

    assert_relative_eq!(preds[1], expected, epsilon = 1e-7, max_relative = 1e-7);
    assert!(
        (expected - stale).abs() > 20.0,
        "the fixture must separate the two conventions: {expected} vs {stale}"
    );
}

#[test]
fn a_dose_record_governs_the_advance_into_its_own_time() {
    // The other half of the fix, and the only geometry that isolates it: a dose
    // row co-timed with an observation carrying a DIFFERENT snapshot. The dose
    // record sorts first, so it — not the observation — terminates the incoming
    // segment. (`io::datareader` nudges a co-timed observation one ULP earlier
    // when the dose comes later in the file, which is how the opposite file order
    // reaches the opposite answer; here the times are equal, i.e. dose-first.)
    //
    //   t=0   dose (CL=20)              t=5  obs (CL=20)
    //   t=10  dose row (CL=5, ALAG=2)  +  obs (CL=40)   -> arrival at t=12
    //   t=20  obs (CL=40)
    let ode = one_cpt_ode_spec_tight();
    let (v, cl_a, cl_dose, cl_obs) = (100.0, 20.0, 5.0, 40.0);
    let (ka, k_dose) = (cl_a / v, cl_dose / v);

    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![5.0, 10.0, 20.0]);
    let pk_dose = vec![pk_one_lagged(cl_a, v, 0.0), pk_one_lagged(cl_dose, v, 2.0)];
    let pk_obs = vec![pk_one(cl_a, v), pk_one(cl_obs, v), pk_one(cl_obs, v)];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);

    // (0,5] and (5,10] at CL=20 and CL=5 respectively — the second terminated by
    // the DOSE record, not the co-timed observation.
    let expected = 1000.0 * (-5.0 * ka).exp() * (-5.0 * k_dose).exp();
    let stale = 1000.0 * (-5.0 * ka).exp() * (-5.0 * cl_obs / v).exp();

    assert_relative_eq!(preds[1], expected, epsilon = 1e-7, max_relative = 1e-7);
    assert!(
        (expected - stale).abs() > 50.0,
        "the fixture must separate the two conventions: {expected} vs {stale}"
    );
}

#[test]
fn a_zero_lagtime_dose_still_takes_its_own_row_for_the_incoming_segment() {
    // Trap: with `ALAG = 0` the dose row IS the record terminating the incoming
    // interval, so `(5, 10]` must take the dose row's snapshot. Skipping the
    // dose-record break when the lagtime is zero — the obvious "optimisation" —
    // is wrong in the opposite direction: the arrival is no longer a parameter
    // source, so the segment would look forward past the dose to the t=20 record.
    let ode = one_cpt_ode_spec_tight();
    let (v, cl_a, cl_dose, cl_late) = (100.0, 20.0, 5.0, 40.0);
    let (ka, k_dose, k_late) = (cl_a / v, cl_dose / v, cl_late / v);

    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![5.0, 20.0]);
    let pk_dose = vec![pk_one(cl_a, v), pk_one(cl_dose, v)]; // both unlagged
    let pk_obs = vec![pk_one(cl_a, v), pk_one(cl_late, v)];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);

    let a10 = 1000.0 * (-5.0 * ka).exp() * (-5.0 * k_dose).exp();
    let expected = (a10 + 1000.0) * (-10.0 * k_late).exp();
    // Looking forward past the dose row would run (5,10] at the t=20 record's CL.
    let stale =
        (1000.0 * (-5.0 * ka).exp() * (-5.0 * k_late).exp() + 1000.0) * (-10.0 * k_late).exp();

    assert_relative_eq!(preds[1], expected, epsilon = 1e-7, max_relative = 1e-7);
    assert!(
        (expected - stale).abs() > 2.0,
        "the fixture must separate the two conventions: {expected} vs {stale}"
    );
}

#[test]
fn an_infusion_end_between_records_runs_on_the_terminating_record() {
    // The same rule at the other non-record kind. An infusion end is a rate
    // change, not a `$PK` evaluation, so the whole record interval containing it
    // runs on that interval's terminating record — measured against NONMEM 7.6.0,
    // which puts a 4.2 % prediction error on the previous-record reading this
    // replaced. The closed-form engine (`pk/event_driven.rs`) already treats
    // infusion edges as sub-interval bounds, so this also brings the two
    // production engines into agreement.
    //
    //   t=0   bolus (CL=20)        t=5  obs (CL=20)
    //   t=6   infusion 50 @ rate 100 (CL=20)  -> ends at t=6.5, no record there
    //   t=7   obs (CL=5)
    let ode = one_cpt_ode_spec_tight();
    let (v, cl_a, cl_c) = (100.0, 20.0, 5.0);
    let (ka, kc) = (cl_a / v, cl_c / v);

    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 50.0, 1, 100.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![5.0, 7.0]);
    let pk_dose = vec![pk_one(cl_a, v), pk_one(cl_a, v)];
    let pk_obs = vec![pk_one(cl_a, v), pk_one(cl_c, v)];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);

    // (0,6] at CL=20; (6, 6.5] and (6.5, 7] both at CL=5 (the t=7 record), the
    // infusion delivering rate=100 over the first of the two.
    let a6 = 1000.0 * (-6.0 * ka).exp();
    let a65 = a6 * (-0.5 * kc).exp() + (100.0 / kc) * (1.0 - (-0.5 * kc).exp());
    let expected = a65 * (-0.5 * kc).exp();
    // The previous-record reading: (6, 6.5] at CL=20.
    let stale_a65 = a6 * (-0.5 * ka).exp() + (100.0 / ka) * (1.0 - (-0.5 * ka).exp());
    let stale = stale_a65 * (-0.5 * kc).exp();

    assert_relative_eq!(preds[1], expected, epsilon = 1e-6, max_relative = 1e-6);
    assert!(
        (expected - stale).abs() > 1.0,
        "the fixture must separate the two conventions: {expected} vs {stale}"
    );
}

/// The adaptive walk's PK and occasion resolvers must land on the **same record**.
///
/// `seg_pk` is written into the RHS parameter slots and `seg_occ` picks the `eta`
/// (`[η_bsv | κ_g]`) threaded alongside it. `event_pk.obs[j]` already embeds occasion
/// `j`'s κ in its PK values, so if the two resolvers disagree the segment integrates PK
/// carrying one window's κ against an `eta` carrying another's — incoherent whichever
/// window is "right", and invisible to any prediction test that does not put a record
/// and a non-record break in different decision windows.
///
/// This is asserted at the resolvers rather than through a simulation because the
/// occasion half is otherwise unpinned: reverting `governing_segment_occ_at` to the
/// pre-#1073 LOCF `segment_occ_at` passes all 220 adaptive tests.
///
/// Geometry: records at `t = 0` and `t = 30`, a non-record break at `t = 24`, and
/// decisions at 0 and 24 — so `t = 24` opens a new occasion window while the record
/// that *terminates* its interval (`t = 30`) sits in that later window and the LOCF
/// carry still holds the earlier one. Every resolution below is therefore a genuine
/// three-way choice.
#[test]
fn the_adaptive_segment_pk_and_occasion_resolve_to_the_same_record() {
    use super::{governing_segment_occ_at, governing_segment_pk_at, AdaptiveRecordIndex};

    let obs_times = [0.0_f64, 30.0];
    let records = AdaptiveRecordIndex::new(&[], &[], &obs_times);

    // Distinct PK per record so the resolved snapshot is identifiable, and occasions
    // that differ across them (window 0 at t=0, window 1 at t=30).
    let event_pk = crate::pk::EventPkParams {
        dose: vec![],
        obs: vec![pk_one(11.0, 100.0), pk_one(22.0, 100.0)],
        pk_only: vec![],
        reset: Vec::new(),
    };
    let obs_occ = [Some(0_usize), Some(1_usize)];
    // The LOCF carry the walk holds when it reaches the t=24 break: the earlier window.
    let (last_pk, last_occ) = (pk_one(99.0, 100.0), Some(0_usize));

    for &t_end in &[24.0_f64, 30.0] {
        let pk = governing_segment_pk_at(t_end, &records, &event_pk, last_pk);
        let occ = governing_segment_occ_at(t_end, &records, &[], &[], &obs_occ, last_occ);
        // Both must resolve to the t=30 record: for `t_end = 30` it is the record
        // itself, and for the non-record break at 24 it is the record that terminates
        // the interval containing it.
        assert_eq!(
            pk.values[crate::types::PK_IDX_CL],
            22.0,
            "t_end={t_end}: segment PK must come from the t=30 record"
        );
        assert_eq!(
            occ,
            Some(1),
            "t_end={t_end}: segment occasion must come from the SAME record as its PK; \
             Some(0) means the occasion fell back to the LOCF carry while the PK looked \
             ahead, so the segment would integrate one window's kappa against another's eta"
        );
    }

    // Past the final record both fall back, and they must fall back together.
    let pk = governing_segment_pk_at(31.0, &records, &event_pk, last_pk);
    let occ = governing_segment_occ_at(31.0, &records, &[], &[], &obs_occ, last_occ);
    assert_eq!(pk.values[crate::types::PK_IDX_CL], 99.0);
    assert_eq!(occ, last_occ);
}

#[test]
fn a_non_record_event_past_the_final_record_does_not_poison_the_walk() {
    // Trap: the forward lookahead is undefined past the last record. An infusion
    // whose end falls after the final observation has no terminating record, so the
    // walk must not panic, read a default-initialised snapshot, or emit NaN.
    //
    //   t=0  bolus (CL=20)   t=5 obs (CL=20)   t=6 infusion 50 @ 100 -> ends 6.5
    //   observations at t=5 and t=6.2, both BEFORE the infusion end.
    //
    // Deliberately NOT an assertion about *which* record the trailing segment picks:
    // that choice is unobservable through predictions, because a segment ending after
    // the final record precedes no observation by construction. The rule itself —
    // "keep the last record that ran" — is pinned where it is observable, on the
    // resolver, by `dosing::tests::governing_record_indices_*`; what matters here is
    // that all four engines take it from that one helper. Do not "strengthen" this by
    // asserting a number that no fallback can move.
    //
    // The *observable* half of the lookahead is made non-degenerate: the two records
    // carry different `CL`, so the `(5, 6]` and `(6, 6.2]` segments are resolved to
    // genuinely different snapshots and the exact value below still pins them.
    let ode = one_cpt_ode_spec_tight();
    let (v, cl_a, cl_c) = (100.0, 20.0, 5.0);
    let (ka, kc) = (cl_a / v, cl_c / v);

    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 50.0, 1, 100.0, false, 0.0),
    ];
    let subj = make_subject(doses, vec![5.0, 6.2]);
    let pk_dose = vec![pk_one(cl_a, v), pk_one(cl_a, v)];
    let pk_obs = vec![pk_one(cl_a, v), pk_one(cl_c, v)];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);
    assert!(
        preds.iter().all(|p| p.is_finite() && *p > 0.0),
        "trailing non-record events must not produce NaN or a zeroed snapshot: {preds:?}"
    );

    // (0, 5] and (5, 6] on CL_A (the t=5 obs, then the t=6 dose record); (6, 6.2] on
    // CL_C (the t=6.2 obs), with the infusion delivering rate=100 throughout it.
    let a6 = 1000.0 * (-6.0 * ka).exp();
    let expected = a6 * (-0.2 * kc).exp() + (100.0 / kc) * (1.0 - (-0.2 * kc).exp());
    assert_relative_eq!(preds[1], expected, epsilon = 1e-6, max_relative = 1e-6);

    // The previous-record reading of the final segment, so the fixture is live.
    let stale = a6 * (-0.2 * ka).exp() + (100.0 / ka) * (1.0 - (-0.2 * ka).exp());
    assert!(
        (expected - stale).abs() > 1.0,
        "the fixture must separate the two conventions: {expected} vs {stale}"
    );
}

/// 1-cpt IV whose RHS reads the `TAD` builtin, so a NaN time-since-dose anchor
/// propagates into the state instead of staying inert.
fn one_cpt_tad_ode_spec() -> OdeSpec {
    let mut ode = OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            // `ext_params[MAX_PK_PARAMS + 1]` is the TAD anchor; the RHS reads
            // `TAD = t − anchor` exactly as the compiled programs do.
            let tad = t - p[crate::types::MAX_PK_PARAMS + 1];
            dy[0] = -ke * y[0] * (1.0 + 0.3 * tad);
        }),
        n_states: 1,
        state_names: vec!["central".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions::default(),
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    };
    ode.solver_opts.method = crate::ode::OdeMethod::Rk45;
    ode.solver_opts.reltol = 1e-12;
    ode.solver_opts.abstol = 1e-14;
    ode
}

#[test]
fn a_tad_reading_rhs_survives_the_pre_arrival_segment_of_a_lagged_first_dose() {
    // #1073 opened a real `(d.time, arrival]` segment ahead of every lagged dose.
    // For a FIRST dose no dose has arrived when that segment integrates, so the
    // TAD anchor has nothing to point at. Answering NaN there multiplies into the
    // state (`0.0 * NaN = NaN`) and poisons every later prediction — turning a
    // finite fit into the 1e20 objective sentinel. The anchor must stay finite.
    //
    // The pre-arrival state is zero here, so the anchor's *value* cannot change
    // the trajectory: the assertion is the closed form for a plain lagged bolus,
    // where TAD is measured from the arrival over the whole observed range.
    let ode = one_cpt_tad_ode_spec();
    let (v, cl, lag) = (100.0, 20.0, 0.5);

    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![2.0, 4.0, 8.0]);
    let pk_dose = vec![pk_one_lagged(cl, v, lag)];
    let pk_obs = vec![pk_one(cl, v); 3];

    let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[]);
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "a NaN TAD anchor in the pre-arrival segment poisoned the trajectory: {preds:?}"
    );

    // `dA/dt = -k·A·(1 + 0.3·(t − τ))` with `τ = 0.5` integrates to
    // `A(t) = A(τ)·exp(−k[(t−τ) + 0.15(t−τ)²])`.
    let k = cl / v;
    for (j, &t) in [2.0_f64, 4.0, 8.0].iter().enumerate() {
        let tad = t - lag;
        let expected = 1000.0 * (-k * (tad + 0.15 * tad * tad)).exp();
        assert_relative_eq!(preds[j], expected, epsilon = 1e-6, max_relative = 1e-5);
    }
}

#[test]
fn both_production_ode_predictors_agree_on_tad_before_the_first_arrival() {
    // `compute_predictions_ode` picks between the two production ODE predictors on
    // `subject.has_resets()` (`pk/mod.rs`). So if they answer `TAD` differently in the
    // pre-arrival window, two subjects of the SAME model and the SAME data shape behave
    // differently — one finite, its neighbour carrying an EVID=3/4 row poisoned into the
    // `1e20` objective sentinel, or the reverse. That split is not a tolerance question;
    // it is a user-visible fork, and it is what this pins.
    //
    // Both are called directly on one reset-free subject, so the comparison is of the
    // predictors and nothing else. The observation at `t = 0.2` sits inside the
    // `(dose row, arrival]` window that #1073 opened; the two after it check that
    // agreeing in the window did not cost agreement outside it.
    let ode = one_cpt_tad_ode_spec();
    let (v, cl, lag) = (100.0, 20.0, 0.5);

    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
    let subj = make_subject(doses, vec![0.2, 2.0, 4.0]);
    let pk_lagged = pk_one_lagged(cl, v, lag);

    let dense = ode_predictions(&ode, &pk_lagged.values, &[], &[], &subj);
    let walk = ode_predictions_event_driven(
        &ode,
        &subj,
        &[],
        &[],
        &[pk_lagged],
        &vec![pk_lagged; 3],
        &[],
        &[],
    );

    assert!(
        dense.iter().all(|p| p.is_finite()),
        "the dense predictor answered NaN for a TAD-reading RHS in the pre-arrival \
         window, poisoning the trajectory: {dense:?}"
    );
    assert!(
        walk.iter().all(|p| p.is_finite()),
        "the event-driven walk answered NaN in the pre-arrival window: {walk:?}"
    );
    for (j, (d, w)) in dense.iter().zip(walk.iter()).enumerate() {
        assert_relative_eq!(d, w, epsilon = 1e-6, max_relative = 1e-5);
        let _ = j;
    }

    // And the value, so this is not merely "the two agree on something". Before the
    // arrival the compartment is empty; after it,
    // `dA/dt = -k·A·(1 + 0.3·(t − τ))` integrates to
    // `A(t) = A(τ)·exp(−k[(t−τ) + 0.15(t−τ)²])`.
    let k = cl / v;
    assert_relative_eq!(dense[0], 0.0, epsilon = 1e-9);
    for (j, &t) in [2.0_f64, 4.0].iter().enumerate() {
        let tad = t - lag;
        let expected = 1000.0 * (-k * (tad + 0.15 * tad * tad)).exp();
        assert_relative_eq!(dense[j + 1], expected, epsilon = 1e-6, max_relative = 1e-5);
    }
}

#[test]
fn the_pre_arrival_tad_anchor_does_not_depend_on_the_sampling_mesh() {
    // The anchor is recomputed per segment, so a fallback that keys off the segment
    // START restarts `TAD` at zero at every record inside the pre-arrival window —
    // a sawtooth whose shape is set by where samples happen to fall. It has to key
    // off something invariant: the subject's first arrival.
    //
    // `init(central)` is what makes this observable. Without it the pre-arrival state
    // is zero, so no anchor can change the trajectory; with it the baseline decays
    // across the window under a `TAD`-reading RHS and the anchor is live.
    //
    // Two physically identical subjects — same dose, same lagtime, same baseline —
    // differing ONLY by one extra observation at t=1, inside the pre-arrival window
    // `(0, 1.5)`. Every shared observation must agree exactly. Measured before the
    // fix: they diverged by 4.2e-4 at t=2, t=4 AND t=8 — the same relative offset at
    // every later time, injected once inside the window and then carried
    // multiplicatively.
    let mut ode = one_cpt_tad_ode_spec();
    ode.init_fn = Some(Box::new(|p: &[f64]| {
        // Baseline is the (otherwise unused) KA slot, so the spec's CL/V keep their
        // meaning in the RHS.
        vec![p[crate::types::PK_IDX_KA]]
    }));

    let (v, cl, lag, base) = (100.0, 20.0, 1.5, 70.0);
    let pk = {
        let mut p = pk_one_lagged(cl, v, lag);
        p.values[crate::types::PK_IDX_KA] = base;
        p
    };

    let run = |obs: Vec<f64>| -> Vec<f64> {
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, obs.clone());
        let pk_dose = vec![pk];
        let pk_obs = vec![pk; obs.len()];
        ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[], &[])
    };

    let coarse = run(vec![0.5, 2.0, 4.0, 8.0]);
    let fine = run(vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    assert!(
        coarse.iter().chain(fine.iter()).all(|p| p.is_finite()),
        "a non-finite TAD anchor poisoned the pre-arrival window: {coarse:?} / {fine:?}"
    );

    // Shared times: coarse[0] ↔ fine[0] (t=0.5), then coarse[1..] ↔ fine[2..].
    assert_relative_eq!(coarse[0], fine[0], epsilon = 1e-12, max_relative = 1e-12);
    for (c, f) in coarse[1..].iter().zip(fine[2..].iter()) {
        assert_relative_eq!(*c, *f, epsilon = 1e-12, max_relative = 1e-12);
    }
    // Guard the guard: the sample at t=0.5 sits INSIDE the pre-arrival window, and it
    // must show the baseline genuinely decaying — strictly below `base`, strictly
    // above zero. Without that the agreement above would hold for a trajectory the
    // anchor cannot reach, and the test would pass for the wrong reason.
    assert!(
        coarse[0] > 1e-3 && coarse[0] < base,
        "fixture must keep live init state inside the pre-arrival window: {coarse:?}"
    );
}

/// The `dose_form_lag_ss` anchor's `[odes]`: oral `depot → central`, reading
/// `CL`/`V`/`KA` from the PK snapshot. Mirrors the `$DES` in
/// `nonmem_anchor/dose_form_lag_ss.ctl` term for term.
fn oral_depot_central_ode_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: Vec::new(),
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ka = p[crate::types::PK_IDX_KA];
            dy[0] = -ka * y[0];
            dy[1] = ka * y[0] - if v > 0.0 { cl / v } else { 0.0 } * y[1];
        }),
        n_states: 2,
        state_names: vec!["depot".into(), "central".into()],
        readout: OdeReadout::ObsCmt(1),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        },
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

/// **#1121 — the SS + lagtime pre-arrival seed, pinned against NONMEM directly.**
///
/// `nonmem_anchor/results/dose_form_lag_ss.tab` tabulates `PRED` — the η = 0
/// population prediction — at the SS dose *record itself* (`TIME = 0`, an
/// `MDV = 1` row): `2.8687E-01`. Nothing downstream has happened at that instant:
/// the lagged pulse has not arrived and no propagation has run, so that number is
/// `A(central)/V` for the periodic trough at phase `II − ALAG`, evaluated under
/// the dose row's own covariates (`WT = 70`, giving `CL = 10`, `KA = 1`). It is a
/// readout of `ss_state_at_phase(…, II − lag)` and of nothing else.
///
/// Confirmed independently of *both* codebases by hand, from the 1-cpt oral
/// steady-state closed form:
///
/// ```text
///   A2(τ) = F·D·KA/(KA−ke)·[ e^(−ke·τ)/(1−e^(−ke·II)) − e^(−KA·τ)/(1−e^(−KA·II)) ]
///   ke = 0.2, KA = 1, II = 12, τ = II − ALAG = 11.3, D = 100, F = 1
///         = 125·(0.1147595 − 0.0000124) = 14.34339   →   /V = 0.2868678
/// ```
///
/// against NONMEM's `0.28687` — agreement to six significant figures, which is
/// all the five-digit `$TABLE` format can express.
///
/// Why this is worth a test of its own, separate from the pointwise `PRED`
/// comparison in `tests/dose_form_lag_nonmem_anchor.rs`: that one cannot reach
/// `t = 0` at all, because the row is `MDV = 1` and `predict()` returns nothing
/// for it. So the seed — the single quantity #1121 is about — is only ever
/// observable there *through* the propagation and the bolus. Isolating it means a
/// failure here says "the phase or the snapshot is wrong" and a failure there
/// says "the propagation is", instead of one ambiguous red.
#[test]
fn ss_state_at_phase_matches_nonmem_at_the_dose_record() {
    // η = 0 at the anchor's initial estimates, evaluated at the dose row's own
    // WT = 70: CL = 10·(70/70)^0.75, V = 50, KA = 1·(70/70)^0.75, ALAG1 = 0.7.
    let (cl, v, ka, lag, ii, amt) = (10.0, 50.0, 1.0, 0.7, 12.0, 100.0);
    let mut pk = pk_one(cl, v);
    pk.values[crate::types::PK_IDX_KA] = ka;
    pk.values[crate::types::PK_IDX_LAGTIME] = lag;

    let ode = oral_depot_central_ode_spec();
    // CMT = 1 (1-based) is the depot — the compartment the dose lands in.
    let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
    assert!(dose.ss && dose.ii > 0.0, "precondition: this is an SS dose");

    let u = ss_state_at_phase(&ode, &pk.values, &dose, ii - lag, &ode.solver_opts, &[]);

    // `PRED` at `TIME = 0`, `nonmem_anchor/results/dose_form_lag_ss.tab`.
    const NM_PRED_AT_RECORD: f64 = 2.8687e-01;
    assert_relative_eq!(u[1] / v, NM_PRED_AT_RECORD, max_relative = 1e-4);

    // Non-degeneracy: the depot leg must be live too, or this pins only the
    // central compartment and a wrong `KA` inside the seed would slip through.
    // NONMEM tabulates no depot amount, but the same steady-state algebra gives it
    // in closed form — the prior pulse's undecayed remainder at phase `II − lag`:
    //
    //     A1(τ) = F·D·e^(−KA·τ)/(1 − e^(−KA·II))
    //           = 100·e^(−11.3)/(1 − e^(−12)) = 1.23730e-3
    //
    // Tiny in absolute terms, but pinned rather than merely bounded: a loose
    // `> 0` would pass for any `KA`, which is the whole point of checking it.
    assert_relative_eq!(u[0], 1.237_30e-3, max_relative = 1e-6);

    // And the seed must genuinely differ from the bare trough, or the whole
    // distinction #1121 rests on is untested by this fixture: at phase `II` the
    // pulse has decayed a further `lag`, so the trough is strictly lower.
    let trough = equilibrate_ss_state(&ode, &pk.values, &dose, &ode.solver_opts, &[]);
    assert!(
        trough[1] < u[1] * 0.95,
        "phase II−lag must be materially above the phase-II trough \
         (seed {}, trough {}) — otherwise this fixture cannot tell the two apart",
        u[1],
        trough[1]
    );
}

/// **Which steady-state doses get the record-time seed, and at what phase (#1121).**
///
/// The predicate and the phase are shared by five paths — the two production
/// walks, their two dual twins, and the dense/superposition static paths — so
/// their boundaries are worth pinning directly rather than inferring them from
/// whichever path a later test happens to exercise. A seed and an arrival-time
/// equilibration must be exactly complementary: both firing loads the trough
/// twice, neither firing leaves the compartments empty.
///
/// `lag == 0` is the one exclusion, and it is deliberate: record and arrival are
/// the same instant, so there is nothing to propagate, and routing it through
/// `ss_state_at_phase(…, II)` would integrate a full extra cycle and move every
/// existing non-lagged SS result by solver error.
///
/// `lag >= II` **is** seeded, at a phase CLAMPED to zero rather than wrapped into
/// `[0, II)`. That is NONMEM 7.6.0's convention, measured rather than assumed
/// (`nonmem_anchor/results/ss_lag_ge_ii.tab`), and it is the only continuous
/// choice: `ALAG` is routinely estimated, so a phase that jumped by a factor
/// `e^{−k·II}` as the outer optimizer walked the lagtime across `II` would put a
/// step in the objective. `ss_lag_ge_ii_pre_arrival_matches_nonmem` pins the
/// resulting *prediction*; this pins the arithmetic it comes from.
#[test]
fn ss_record_seed_declines_a_zero_lag_only() {
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);

    assert!(
        ss_seeded_at_record(&dose, 0.7),
        "an ordinary lagged SS dose"
    );
    assert!(
        !ss_seeded_at_record(&dose, 0.0),
        "no lag: record and arrival coincide, so the arrival equilibrates as before"
    );
    for lag in [12.0, 20.0] {
        assert!(
            ss_seeded_at_record(&dose, lag),
            "a lag of a full interval or more is still seeded — at a clamped phase"
        );
    }

    // The phase itself: `II − lag` while that is positive, then pinned at 0.
    assert_relative_eq!(ss_seed_phase(&dose, 0.7), 11.3, max_relative = 1e-12);
    assert_relative_eq!(ss_seed_phase(&dose, 11.999), 0.001, max_relative = 1e-9);
    assert_eq!(ss_seed_phase(&dose, 12.0), 0.0);
    assert_eq!(
        ss_seed_phase(&dose, 20.0),
        0.0,
        "clamped, NOT wrapped to 4.0"
    );

    // Complementarity of the *arrival-side* shortcut: the paths that re-equilibrate
    // at the arrival rather than propagating there are exact only while the flowed
    // state is the trough, which is exactly `lag <= II`.
    assert!(ss_arrival_is_trough(&dose, 0.7));
    assert!(ss_arrival_is_trough(&dose, 12.0));
    assert!(!ss_arrival_is_trough(&dose, 12.001));

    // A non-SS dose is never seeded however it is lagged, and an SS record with no
    // dosing interval is not a steady state at all.
    let plain = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    assert!(!ss_seeded_at_record(&plain, 0.7));
    let ss_no_ii = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 0.0);
    assert!(!ss_seeded_at_record(&ss_no_ii, 0.7));
}

/// **The previous cycle's infusion when it is still running at the dose record
/// (#1121).**
///
/// `ss_state_at_phase` hands back a state with the infusion mid-flight whenever
/// the seed phase falls inside the window; the walk must then carry `+rate` for
/// the rest of it. The window is what every engine breaks and gates on, so pin
/// its edge here rather than only through a prediction.
///
/// Measured against NONMEM 7.6.0 (`nonmem_anchor/results/ss_lag_infusion.tab`):
/// with `II = 12`, `T_inf = 6`, `ALAG1 = 8`, the concentration *rises* off the
/// record and turns over at exactly `t = 2 = 8 − 12 + 6`.
#[test]
fn ss_residual_infusion_window_ends_where_the_previous_cycle_does() {
    // AMT 600 at RATE 100 => `DoseEvent::new` derives T_inf = 6, against II = 12.
    let inf = DoseEvent::new(0.0, 600.0, 1, 100.0, true, 12.0);
    assert_relative_eq!(inf.duration, 6.0, max_relative = 1e-12);

    // lag <= II − T_inf: the previous cycle finished before the record.
    assert_eq!(ss_residual_infusion_end(&inf, 0.2, 1.0), None);
    assert_eq!(
        ss_residual_infusion_end(&inf, 6.0, 1.0),
        None,
        "phase == T_inf"
    );

    // lag > II − T_inf: it is still running, and stops at `d.time + (T_inf − phase)`.
    let end = ss_residual_infusion_end(&inf, 8.0, 1.0).expect("still running at the record");
    assert_relative_eq!(end, 2.0, max_relative = 1e-12);
    // Continuous at the boundary rather than switching on with a finite window.
    let just_past = ss_residual_infusion_end(&inf, 6.0 + 1e-9, 1.0).expect("just inside");
    assert!(just_past > 0.0 && just_past < 1e-8);
    // Under the phase clamp, `lag >= II` puts the whole window on the record.
    assert_relative_eq!(
        ss_residual_infusion_end(&inf, 14.0, 1.0).expect("clamped phase 0"),
        6.0,
        max_relative = 1e-12
    );

    // A bolus has no window however it is lagged, and neither does an unseeded dose.
    let bolus = DoseEvent::new(0.0, 600.0, 1, 0.0, true, 12.0);
    assert_eq!(ss_residual_infusion_end(&bolus, 8.0, 1.0), None);
    assert_eq!(
        ss_residual_infusion_end(&inf, 0.0, 1.0),
        None,
        "lag 0 is not seeded"
    );

    // `F` reshapes a rate-defined window (#419), so the residual edge moves with it —
    // and the caller must pass the same `f_bio` it built the dose's own window from.
    // At F = 0.5 the window is only 3 h, so a phase of 4 is already past its end and
    // there is no residual at all: the same lag that HAS one at F = 1 does not here.
    assert_eq!(ss_residual_infusion_end(&inf, 8.0, 0.5), None);
    // At F = 0.9 the window is 5.4 h, so a phase of 4 is still inside it.
    assert_relative_eq!(
        ss_residual_infusion_end(&inf, 8.0, 0.9).expect("F·T_inf = 5.4 > phase 4"),
        1.4,
        max_relative = 1e-12
    );
}

// ── #1171: the route-onset break must reach every break-time builder ─────────

/// [`zero_order_accumulator_spec`] with a **per-route lag** at free slot 6 — the
/// #1171 fixture. `dy = 0`, so the compartment holds exactly the delivered mass and
/// the truth at any `t` is a clamped ramp, computable in closed form without an
/// integrator (see [`route_lagged_zero_order_truth`]).
fn route_lagged_zero_order_accumulator_spec() -> OdeSpec {
    let mut ode = zero_order_accumulator_spec();
    ode.input_rate[0].lag_slot = Some(6);
    ode
}

/// Two 100 mg doses at `t = 0` and `t = 12` into the zero-order route. The second
/// window opens with the first dose's mass already in the compartment, so the
/// *incoming* side of the later window is live — a single-dose fixture would leave
/// `g(x⁻) = 0` there and could not tell a dropped window from a correct one.
fn route_lagged_zero_order_doses() -> Vec<DoseEvent> {
    vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ]
}

/// Exact accumulated mass for [`route_lagged_zero_order_accumulator_spec`]: each
/// dose contributes a clamped linear ramp over `[t_dose + lag, t_dose + lag + dur]`.
/// Independent of every ODE engine — the oracle, not a second implementation.
fn route_lagged_zero_order_truth(t: f64, dur: f64, lag: f64) -> f64 {
    route_lagged_zero_order_doses()
        .iter()
        .map(|d| d.amt * ((t - d.time - lag) / dur).clamp(0.0, 1.0))
        .sum()
}

/// The two break-time builders that segment a dense solve must agree on a
/// route-lagged subject. `build_segment_break_times` (the dense/simulate engines)
/// omitted [`push_route_lag_break_times`] while `collect_dose_break_times` (the
/// prediction engines) had it — #1171. `ode_predictions_with_states`' inline builder
/// has no unit-callable entry point; it is pinned end-to-end by
/// `route_lagged_zero_order_reaches_every_dense_engine` below.
#[test]
fn route_lagged_zero_order_break_builders_agree() {
    const DUR: f64 = 2.0;
    const LAG: f64 = 1.5;
    let ode = route_lagged_zero_order_accumulator_spec();
    let mut pk = pk_zero_order_vec(DUR, 1.0);
    pk[6] = LAG;
    let doses = route_lagged_zero_order_doses();
    let lags = vec![0.0; doses.len()];
    let f_bio = vec![1.0; doses.len()];
    let subject = make_subject(doses, vec![24.0]);
    let windows = zo_windows_for(&ode, &subject.doses, &lags, &pk);

    let normalize = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        v
    };
    // The dense builder additionally brackets the timeline with the integration
    // start and the terminal, so add both to the prediction side before comparing.
    let mut from_predictions = vec![subject_integration_start(&subject), 24.0];
    collect_dose_break_times(&mut from_predictions, &ode, &subject, &lags, &f_bio, &pk);
    let from_predictions = normalize(from_predictions);

    let from_dense = build_segment_break_times(&ode, &pk, &subject, &lags, &f_bio, &windows, 24.0);

    assert_eq!(
        from_predictions, from_dense,
        "the dense builder must segment a route-lagged subject exactly as the \
         prediction builder does"
    );
    // Non-degeneracy: both windows' onsets are actually in the list, so the
    // comparison above is not two empty vectors agreeing.
    for want in [LAG, LAG + DUR, 12.0 + LAG, 12.0 + LAG + DUR] {
        assert!(
            from_predictions.iter().any(|&t| (t - want).abs() < 1e-12),
            "break at {want} missing from {from_predictions:?}"
        );
    }
}

/// End-to-end #1171: all three dense engines must deliver the lagged zero-order
/// mass, checked against the closed-form ramp rather than against each other.
///
/// - `ode_predictions` (the objective's engine) was always correct.
/// - `ode_predictions_with_states` (sdtab IPRED + compartment states) uses its own
///   inline break-time builder — the near-copy the `KEEP-IN-SYNC` note on it warns
///   about, and the copy that drifted.
/// - `ode_dense_solve_states` (sdtab states under TV covariates, the joint PK-TTE
///   hazard, the Markov endpoint NLL, the adaptive AUC signal, `[derived]` grid
///   integrals) goes through `build_segment_break_times`.
///
/// Sample times straddle each window on both sides and sit *inside* the first one,
/// where the truth is a strict partial (`(t − 1.5)/2 · 100`) that neither an
/// all-delivered nor an all-dropped window can produce.
#[test]
fn route_lagged_zero_order_reaches_every_dense_engine() {
    const DUR: f64 = 2.0;
    const LAG: f64 = 1.5;
    const AMT: f64 = 100.0;
    let ode = route_lagged_zero_order_accumulator_spec();
    let mut pk = pk_zero_order_vec(DUR, 1.0);
    pk[6] = LAG;
    // 1.0 pre-onset, 2.3/3.1 inside and just past the first window, 11.5 the trough,
    // 13.8/15.0 inside and just past the second, 24.0 the tail.
    let obs = vec![1.0, 2.3, 3.1, 4.2, 11.5, 12.4, 13.8, 15.0, 18.0, 24.0];
    let subject = make_subject(route_lagged_zero_order_doses(), obs.clone());
    let want: Vec<f64> = obs
        .iter()
        .map(|&t| route_lagged_zero_order_truth(t, DUR, LAG))
        .collect();
    assert!(
        want.iter().any(|&m| m > 1.0 && m < AMT - 1.0),
        "the oracle must include a partially-delivered window, else a dropped \
         window is indistinguishable from a completed one"
    );

    let check = |got: &[f64], what: &str| {
        // `zip` stops at the shorter side, so without this a short or empty result —
        // the #1171 signature itself — would make every assertion below vacuous.
        assert_eq!(got.len(), want.len(), "{what}: wrong number of values");
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6 * (1.0 + w.abs()),
                "{what}: t={} gives {g}, want {w}",
                obs[i]
            );
        }
    };

    let preds = ode_predictions(&ode, &pk, &[], &[], &subject);
    check(&preds, "ode_predictions");

    let (states_ipred, states) = ode_predictions_with_states(&ode, &pk, &[], &[], &subject);
    check(&states_ipred, "ode_predictions_with_states ipred");
    check(
        &states.iter().map(|u| u[0]).collect::<Vec<_>>(),
        "ode_predictions_with_states states",
    );

    let dense = ode_dense_solve_states(&ode, &pk, &[], &[], &subject, &obs);
    check(
        &dense.iter().map(|u| u[0]).collect::<Vec<_>>(),
        "ode_dense_solve_states",
    );
}

/// The route-onset break must reach both dense builders for a **smooth** lagged
/// kernel too — the half of #1171's fix that `push_zero_order_break_times`' own
/// `w_start` push does *not* cover.
///
/// This is the discriminating case. Every other fixture here uses `zero_order`,
/// whose onset coincides exactly with `w_start`, so all of them stay green if the
/// route-onset push is deleted and the window pusher is left to bracket the start.
/// A `first_order(ka=KA, lag=L)` route produces **no `ZeroOrderWindow` at all**, so
/// its onset has no other source — delete
/// [`push_route_lag_break_times`](super::push_route_lag_break_times) from either
/// builder and this test is the one that fails.
///
/// It is not a mere nicety either: `first_order`'s kernel is `dose·ka·exp(-ka·tad)`
/// with a hard `0` for `tad <= 0`, i.e. a **step** at the onset, and
/// `sens/ode_provider.rs` emits `K_ROUTE_ONSET` for every lagged kind pinned
/// bit-identical to this break (#859).
///
/// The compartment lag is **non-zero** on purpose. The onset is
/// `d.time + lag_cmt + lag_route`; with `dose_lagtimes` all-zero — as every other
/// fixture here has it — that middle term is degenerate, and passing an empty slice
/// instead would be bit-identical.
#[test]
fn route_lagged_first_order_onset_reaches_both_dense_builders() {
    const LAG_CMT: f64 = 2.0;
    const LAG_ROUTE: f64 = 1.5;
    // `first_order` on cmt 0 with `ka` at slot PK_IDX_KA and the route lag at slot 6.
    let mut ode = zero_order_accumulator_spec();
    ode.input_rate = vec![InputRateForcing {
        cmt: 0,
        kind: InputRateKind::FirstOrder,
        arg_slots: vec![crate::types::PK_IDX_KA],
        frac_slot: None,
        lag_slot: Some(6),
    }];
    let mut pk = vec![0.0; crate::types::MAX_PK_PARAMS];
    pk[crate::types::PK_IDX_KA] = 1.2;
    pk[crate::types::PK_IDX_F] = 1.0;
    pk[6] = LAG_ROUTE;

    let doses = route_lagged_zero_order_doses();
    // A LIVE compartment lag, not zeros: the onset is `d.time + lag_cmt + lag_route`,
    // and with `dose_lagtimes` all-zero that middle term is degenerate — dropping the
    // slice entirely would be bit-identical and no assertion could tell.
    let lags = vec![LAG_CMT; doses.len()];
    let f_bio = vec![1.0; doses.len()];
    let subject = make_subject(doses, vec![24.0]);

    // Non-degeneracy: a smooth route yields no zero-order window, so the window
    // pusher contributes nothing and the onset can only come from the route-lag push.
    let windows = zo_windows_for(&ode, &subject.doses, &lags, &pk);
    assert!(
        windows.is_empty(),
        "a first_order route must produce no ZeroOrderWindow, else this fixture \
         cannot discriminate the route-onset push from the window pusher"
    );

    let breaks = build_segment_break_times(&ode, &pk, &subject, &lags, &f_bio, &windows, 24.0);
    for want in [LAG_CMT + LAG_ROUTE, 12.0 + LAG_CMT + LAG_ROUTE] {
        assert!(
            breaks.iter().any(|&t| (t - want).abs() < 1e-12),
            "build_segment_break_times dropped the first_order route onset at {want}: \
             {breaks:?}"
        );
    }

    // …and the prediction builder, which always had it, must agree.
    let mut from_predictions = Vec::new();
    collect_dose_break_times(&mut from_predictions, &ode, &subject, &lags, &f_bio, &pk);
    for want in [LAG_CMT + LAG_ROUTE, 12.0 + LAG_CMT + LAG_ROUTE] {
        assert!(
            from_predictions.iter().any(|&t| (t - want).abs() < 1e-12),
            "collect_dose_break_times dropped the first_order route onset at {want}: \
             {from_predictions:?}"
        );
    }
}

// ── #1210 — an SS dose must not carry the equilibration run-in into H(0) ─────────────
//
// The whole family is anchored on a **constant** hazard, `dCHZ/dt = SS_CHZ_H0`. Then
// `H(t) = H0·t` exactly, whatever the PK does, so the reference is arithmetic rather than a
// second integration — a disagreement here is the engine's and cannot be the oracle's.
// `nonmem_anchor/ss_chz_r2_*.ctl` carries the drug-driven and time-dependent arms, which no
// closed form reaches; `tests/ss_chz_nonmem_anchor.rs` compares against them.

/// #1210's constant hazard. `H(t) = SS_CHZ_H0 · t`.
#[cfg(feature = "survival")]
const SS_CHZ_H0: f64 = 0.02;

/// #1210's dosing interval, and the SS run-in's own displacement: with
/// `SS_EQUILIBRATION_CYCLES = 50` cycles of `II = 12`, an unmasked accumulator banks
/// `0.02 × 50 × 12 = 12.0` before the record starts.
#[cfg(feature = "survival")]
const SS_CHZ_II: f64 = 12.0;

/// 1-cpt IV bolus + a **declared** constant-rate cumulative-hazard accumulator: state 0 =
/// central (`dC/dt = -ke·C`), state 1 = the injected `d/dt(__chz_<cmt>)` row.
///
/// The declaration is the point: `chz_state_slots = [1]` is what `build_ode_spec` produces
/// for a model with an `[event_model]`, and it is what tells the steady-state equilibration
/// that row is not a compartment. A fixture that carried the accumulator *without* declaring
/// it would exercise the pre-#1210 code path and pass whatever the engine did.
#[cfg(feature = "survival")]
fn one_cpt_const_chz_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: vec![1],
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
            dy[1] = SS_CHZ_H0;
        }),
        n_states: 2,
        state_names: vec!["central".into(), "__chz_3".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions {
            abstol: 1e-11,
            reltol: 1e-9,
            ..OdeSolverOptions::default()
        },
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

/// Cumulative hazard (state 1) at each of `times`, through the dense states path — the same
/// walk `predict_survival`, `ode_cumhaz_hazard` and TTE simulation read.
#[cfg(feature = "survival")]
fn chz_at(ode: &OdeSpec, pk: &PkParams, subject: &Subject, times: &[f64]) -> Vec<f64> {
    // Read the slot the spec declares, not a literal: a fixture that puts the accumulator
    // anywhere but state 1 would otherwise have its drug amounts compared against `H0 * t`.
    let slot = ode.chz_state_slots[0];
    ode_dense_solve_states(ode, &pk.values, &[], &[], subject, times)
        .iter()
        .map(|s| s[slot])
        .collect()
}

/// `H(t) = H0·t` at every requested time, and `H` non-decreasing along the grid.
///
/// **Length first.** `zip` truncates to the shorter side, so without this every assertion below
/// is a no-op on an empty `got` and the arm passes having computed nothing. That is not
/// hypothetical: `chz_at` hands back whatever `ode_dense_solve_states` returned, and a walk
/// that bails — or the zero-max-grid path in #1218 — yields fewer rows than `times`. Every arm
/// in this file routes its whole claim through this helper.
///
/// **Monotonicity is checked before the closed form**, so it is the assertion that fires on a
/// backwards accumulator rather than dead weight behind a tighter bound. #1210's `ALAG1 = 2`
/// arm read `H(1) = 12.22 > H(4) = 12.04` — impossible for a cumulative hazard, and the shape a
/// phase advance through an unmasked RHS produces. Ordered the other way the closed-form bound
/// panics on `H(1)` first and the monotonicity loop can never fail on its own.
#[cfg(feature = "survival")]
fn assert_hazard_is_the_closed_form(got: &[f64], times: &[f64], arm: &str) {
    assert_eq!(
        got.len(),
        times.len(),
        "{arm}: the walk returned {} hazard values for {} requested times; `zip` below would \
         silently truncate and assert nothing",
        got.len(),
        times.len()
    );
    for (&h, &t) in got.iter().zip(times) {
        assert!(
            h.is_finite(),
            "{arm}: H({t}) = {h} is not finite — a NaN here would be silently absorbed by any \
             `max`/`min` fold downstream"
        );
    }
    for w in got.windows(2) {
        assert!(
            w[1] >= w[0],
            "{arm}: H is non-monotone: {got:?} over {times:?}"
        );
    }
    for (&h, &t) in got.iter().zip(times) {
        assert!(
            (h - SS_CHZ_H0 * t).abs() < 1e-9,
            "{arm}: H({t}) = {h}, closed form {}",
            SS_CHZ_H0 * t
        );
    }
}

/// An `SS=1` bolus starts the hazard clock at the record: `H(0) = 0`, not the 600 hours of
/// run-in the equilibration integrates through (#1210).
///
/// **No single-piece mutation kills this arm, and that is a property of the fix, not a gap in
/// the test.** Measured: masking the accumulator, projecting it out of the fixed-point solve,
/// and restoring `chz_before` each independently leave `H(0) = 0` for a *first* SS dose, whose
/// pre-record value is zero either way. Removing any one of them keeps this green. It dies
/// under the composite — all three removed, i.e. the pre-#1210 state — where `H(0)` reads
/// `12.0 = H0 × SS_EQUILIBRATION_CYCLES × II` and every later record is displaced by the same
/// amount. That is the number this arm exists to catch, and it is what ferx returned at
/// `1187062a`.
///
/// The shape stays right under that defect — it is an origin error — so an assertion on the
/// *increments* alone would pass in both directions; the absolute values are the test.
#[cfg(feature = "survival")]
#[test]
fn ss_bolus_starts_the_hazard_clock_at_the_record() {
    let ode = one_cpt_const_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );
    let times = [0.0, 1.0, 4.0, 12.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert_hazard_is_the_closed_form(&got, &times, "SS bolus");
}

/// The same for an `SS=1` **infusion**, which equilibrates through a different branch (an
/// active-rate window plus a quiet window per cycle) and so cannot be inferred from the bolus
/// arm. #1210's before-table read `12.0` here too.
///
/// Same mutation profile as the bolus arm: no single-piece removal reaches it, and the
/// composite does. See [`ss_bolus_starts_the_hazard_clock_at_the_record`].
#[cfg(feature = "survival")]
#[test]
fn ss_infusion_starts_the_hazard_clock_at_the_record() {
    let ode = one_cpt_const_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 25.0, true, SS_CHZ_II)],
        vec![],
    );
    let times = [0.0, 1.0, 4.0, 12.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert_hazard_is_the_closed_form(&got, &times, "SS infusion");
}

/// An `SS=1` dose whose arrival is lagged by less than `II` is seeded at the *record*
/// (#1121): `ss_state_at_phase` equilibrates and then advances the trough through
/// `phase = II − lag`. That advance is still run-in — it reconstructs the previous interval's
/// tail, which happened before the record began — so it must not accrue hazard either.
///
/// Killed by dropping `restore_chz` — or by zeroing instead of preserving — and it dies
/// through the **arrival**, not the record seed: with `lag < II` the lagged arrival is still
/// the trough, so it re-equilibrates, and by then `0.02 × 2 = 0.04` of hazard has accrued
/// since the record. Discarding it puts every later value 0.04 low. Measured, not inferred:
/// masking `ss_state_at_phase_pk`'s own advance kills nothing on its own, because the wrapper
/// restores that row afterwards regardless (see [`mask_chz`] for what the mask is actually
/// load-bearing for).
///
/// At `1187062a` this arm read `H(0) = 12.2` — the equilibration's `12.0` plus the phase
/// advance's `0.02 × 10` — with `H` non-monotone across the record (`H(1) = 12.22`,
/// `H(4) = 12.04`), which is why the shared helper asserts monotonicity too.
#[cfg(feature = "survival")]
#[test]
fn ss_lagged_dose_does_not_bank_its_phase_advance() {
    let lag_slot = 6;
    let mut ode = one_cpt_const_chz_spec();
    ode.dose_attr_map
        .insert(crate::types::DoseAttr::Lag, 1, lag_slot);
    let mut pk = pk_one(1.0, 10.0);
    pk.values[lag_slot] = 2.0; // < II, so the trough is seeded at the record and flows
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );
    let times = [0.0, 1.0, 4.0, 12.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert_hazard_is_the_closed_form(&got, &times, "SS bolus + ALAG1 = 2");
}

/// A lag **past** `II` takes the other branch (#1121: the arrival is no longer the trough, so
/// the record-time seed flows to it instead of being recomputed). Its own arm, because the two
/// branches reach the accumulator by different routes.
#[cfg(feature = "survival")]
#[test]
fn ss_lag_beyond_the_interval_starts_the_hazard_clock_at_the_record() {
    let lag_slot = 6;
    let mut ode = one_cpt_const_chz_spec();
    ode.dose_attr_map
        .insert(crate::types::DoseAttr::Lag, 1, lag_slot);
    let mut pk = pk_one(1.0, 10.0);
    pk.values[lag_slot] = 14.0; // > II
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );
    let times = [0.0, 1.0, 4.0, 12.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert_hazard_is_the_closed_form(&got, &times, "SS bolus + ALAG1 = 14");
}

/// **The rule is preserve, not zero.** A second `SS=1` dose in the middle of a record
/// re-equilibrates the compartments — that is what `SS=1` means — but the cumulative hazard
/// accrued over `[0, 48)` is a fact about the subject's record and survives it.
///
/// This is the arm that separates the fix from the issue's original suggestion. Zeroing the
/// accumulator rows (rather than restoring `chz_before`) passes every single-dose arm above,
/// because a first SS dose's pre-record value *is* zero; here it would discard `H(48⁻) ≈ 0.96`
/// and restart the clock, and `H(60)` would read `0.24` instead of `1.2`.
///
/// The straddle is asserted, not assumed: `H` just before the second dose must be materially
/// non-zero, or "preserve" and "zero" would agree and the test could not tell them apart.
#[cfg(feature = "survival")]
#[test]
fn a_second_ss_dose_keeps_the_hazard_accrued_so_far() {
    let ode = one_cpt_const_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II),
            DoseEvent::new(48.0, 100.0, 1, 0.0, true, SS_CHZ_II),
        ],
        vec![],
    );
    let times = [0.0, 12.0, 47.9, 48.0, 48.1, 60.0];
    let got = chz_at(&ode, &pk, &subject, &times);

    // Without a live hazard on the incoming side of the second dose record, "preserve" and
    // "zero" are the same edit and this test would be vacuous.
    let h_before = got[2];
    assert!(
        h_before > 0.9,
        "the pre-dose hazard must be materially non-zero for this arm to discriminate; got \
         H(47.9) = {h_before}"
    );

    assert_hazard_is_the_closed_form(&got, &times, "SS@0 + SS@48");

    // Continuity across the dose record itself, stated separately from the closed form: the
    // accumulator is a state, so the only thing that may move it over `[47.9, 48]` is the
    // 0.1 h of elapsed time — the dose event contributes no jump of its own.
    let elapsed = times[3] - times[2];
    assert!(
        (got[3] - got[2] - SS_CHZ_H0 * elapsed).abs() < 1e-9,
        "H jumped across the second SS dose beyond the {elapsed} h of accrual: H(47.9) = {}, \
         H(48) = {}",
        got[2],
        got[3]
    );
}

/// The same mid-record rule when the *first* dose is an ordinary bolus, so the accumulated
/// hazard reaching the SS record came from a plain forward integration rather than from an
/// earlier equilibration.
#[cfg(feature = "survival")]
#[test]
fn a_plain_dose_followed_by_an_ss_dose_keeps_the_hazard_accrued_so_far() {
    let ode = one_cpt_const_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(48.0, 100.0, 1, 0.0, true, SS_CHZ_II),
        ],
        vec![],
    );
    let times = [0.0, 12.0, 47.9, 48.0, 48.1, 60.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert!(got[2] > 0.9, "pre-dose hazard must be live; got {}", got[2]);
    assert_hazard_is_the_closed_form(&got, &times, "single@0 + SS@48");
}

/// The mid-record rule when the SS dose is **also lagged past the interval** — the only
/// combination that reaches [`ss_state_at_phase`]'s own [`restore_chz`].
///
/// `ss_state_at_phase_pk` returns the accumulator at zero unconditionally: it starts from
/// `equilibrate_ss_pk_state`'s zero vector and every integration inside it runs under
/// [`mask_chz`]. So the wrapper's restore only has an effect when `chz_before != 0` — a dose
/// that is *both* mid-record and lagged. Every other lag arm in this file doses once at
/// `t = 0`, where restoring and zeroing are the same edit; measured, dropping that restore
/// killed nothing at all before this arm existed. It is also why
/// `ss_lag_beyond_the_interval_starts_the_hazard_clock_at_the_record` survived the composite
/// pre-fix mutation: that arm's `chz_before` is zero, so the removal was invisible to it.
///
/// `ALAG1 > II` on purpose, and that is the half that isolates the restore. With a lag inside
/// the interval the arrival is still the trough, so it re-equilibrates through
/// [`equilibrate_ss_state`] and *that* function's restore would put the row back — the arm
/// would then die from the wrong edit. Past `II` the phase clamps to zero and no arrival
/// re-equilibration happens, leaving `ss_state_at_phase` the only thing carrying `H` across
/// the record.
///
/// Killed by dropping `restore_chz` from [`ss_state_at_phase`]: the record seed zeroes the
/// accumulator, so `H(48)` restarts from the phase advance's own accrual instead of
/// continuing `H(47.9) ≈ 0.958`, and every later value is short by that much.
#[cfg(feature = "survival")]
#[test]
fn a_mid_record_ss_dose_lagged_past_the_interval_keeps_the_hazard_accrued_so_far() {
    let lag_slot = 6;
    let mut ode = one_cpt_const_chz_spec();
    ode.dose_attr_map
        .insert(crate::types::DoseAttr::Lag, 1, lag_slot);
    let mut pk = pk_one(1.0, 10.0);
    pk.values[lag_slot] = 14.0; // > II, so the phase clamps and the record seed is what flows
    let subject = make_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(48.0, 100.0, 1, 0.0, true, SS_CHZ_II),
        ],
        vec![],
    );
    let times = [0.0, 12.0, 47.9, 48.0, 48.1, 60.0];
    let got = chz_at(&ode, &pk, &subject, &times);

    // The straddle, asserted rather than assumed: with no live hazard arriving at the second
    // record, "preserve" and "zero" agree and this arm cannot discriminate.
    let h_before = got[2];
    assert!(
        h_before > 0.9,
        "the pre-dose hazard must be materially non-zero for this arm to discriminate; got \
         H(47.9) = {h_before}"
    );

    assert_hazard_is_the_closed_form(&got, &times, "single@0 + SS@48 + ALAG1 = 14");

    let elapsed = times[3] - times[2];
    assert!(
        (got[3] - got[2] - SS_CHZ_H0 * elapsed).abs() < 1e-9,
        "H jumped across the lagged SS dose record beyond the {elapsed} h of accrual: \
         H(47.9) = {}, H(48) = {}",
        got[2],
        got[3]
    );
}

/// `SS=1` into a built-in absorption compartment (#719) equilibrates through
/// `equilibrate_ss_input_rate` — a third branch, with its own exact solve and its own
/// Anderson fallback. It reaches the accumulator by a different route than either bolus
/// branch, so it gets its own arm.
#[cfg(feature = "survival")]
#[test]
fn ss_into_an_input_rate_compartment_starts_the_hazard_clock_at_the_record() {
    let mut ode = one_cpt_const_chz_spec();
    ode.input_rate = vec![InputRateForcing {
        cmt: 0,
        kind: InputRateKind::FirstOrder,
        arg_slots: vec![4],
        frac_slot: None,
        lag_slot: None,
    }];
    let mut pk = pk_one(1.0, 10.0);
    pk.values[4] = 0.15; // slow ka: the absorption tail spans more than one II
    pk.values[crate::types::PK_IDX_F] = 1.0;
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );
    let times = [0.0, 1.0, 4.0, 12.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert_hazard_is_the_closed_form(&got, &times, "SS into first_order()");

    // The hazard assertion above is carried by the *fallback* path: with the joint branch left
    // unmasked and unreduced, the exact solve is singular, the Anderson iteration cannot
    // converge a row that grows every cycle, and `equilibrate_ss_input_rate` returns `None` —
    // at which point the masked pulse train downstream produces the right `H` anyway. So the
    // cycle count is what actually pins this branch: measured `1` (one linear solve), versus
    // the 50-cycle cap the pre-#1210 code reached.
    let cycles = crate::dosing::last_ss_equilibration_cycles();
    assert_eq!(
        cycles, 1,
        "the input-rate branch fell back to the pulse train ({cycles} cycles); its joint path \
         is not taking the masked, projected solve"
    );
}

/// The equilibration hands the accumulator rows back **exactly** as it received them — the
/// unit statement of #1210's rule, one level below the walks above.
///
/// A non-zero, non-round `chz_before` catches a restore that writes the wrong slot or that
/// silently substitutes zero, neither of which the `H(0) = 0` arms can see.
#[cfg(feature = "survival")]
#[test]
fn ss_equilibration_returns_the_accumulator_untouched() {
    let ode = one_cpt_const_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II);
    let u = equilibrate_ss_state(&ode, &pk.values, &dose, &ode.solver_opts, &[7.5]);
    assert_eq!(u[1], 7.5, "the accumulator row was not preserved: {u:?}");
    assert!(
        u[0] > 0.0,
        "the PK row must still be equilibrated — a fixture that returns a zero state would \
         satisfy the accumulator assertion vacuously; got {u:?}"
    );
}

/// Masking alone would leave `I − M` singular: an accumulator's one-cycle map is the
/// identity, so its row of `I − M` is all zeros and the exact solve declines for **every**
/// joint model, whatever the PK block looks like. That is why #1210's fixtures ran the full
/// 50-cycle pulse train — and why some of them fired spurious #867 non-convergence warnings.
/// Only some: the warning rides a geometric-tail test, so it is not a reliable signature of
/// the defect, which is why this arm asserts the cycle count instead.
///
/// The straddle is the point: the *same* RHS, differing only in whether the accumulator row is
/// declared, must take different paths — one solve when it is (the row is projected out), the
/// capped train when it is not (the pre-#1210 behaviour, kept here as the contrast rather
/// than asserted as desirable).
#[cfg(feature = "survival")]
#[test]
fn a_joint_linear_model_takes_the_exact_ss_solve() {
    let pk = pk_one(1.0, 10.0);
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II);

    let declared = one_cpt_const_chz_spec();
    let _ = equilibrate_ss_state(&declared, &pk.values, &dose, &declared.solver_opts, &[0.0]);
    let cycles_declared = crate::dosing::last_ss_equilibration_cycles();

    let mut undeclared = one_cpt_const_chz_spec();
    undeclared.chz_state_slots.clear();
    // The snapshot is parallel to `chz_state_slots`, so with none declared it is empty. In
    // production that is guaranteed — every caller passes a `chz_snapshot` of this same spec —
    // and `restore_chz` debug-asserts the pairing.
    let _ = equilibrate_ss_state(&undeclared, &pk.values, &dose, &undeclared.solver_opts, &[]);
    let cycles_undeclared = crate::dosing::last_ss_equilibration_cycles();

    assert_eq!(
        cycles_declared, 1,
        "a joint model with a linear PK block must take the exact fixed point, not the pulse \
         train; ran {cycles_declared} cycles"
    );
    assert!(
        cycles_undeclared > 1,
        "the contrast has collapsed: with the accumulator left undeclared the solve is \
         singular and the capped train is expected, but it ran {cycles_undeclared} cycles — \
         this test can no longer tell the projection from the fallback"
    );
}

/// A reset **is** a reset: `EVID=3` zeroes the accumulator like every other state, and an
/// `EVID=4` row (reset + dose at the same time) does too, because the reset runs first and
/// the SS equilibration then preserves what it finds — which is zero.
///
/// This pins the boundary of #1210's rule from the other side. "Preserve across an SS dose"
/// and "zero on a reset" are two different statements about the same row, and the second is
/// unchanged by the fix: before it, `EVID=3` already zeroed and `EVID=4` read `12.0` (the
/// run-in, not the pre-reset hazard). Both now read `0`, so a later decision to change reset
/// semantics for the hazard shows up here as a deliberate diff rather than as drift.
#[cfg(feature = "survival")]
#[test]
fn a_reset_zeroes_the_accumulator_with_or_without_an_ss_dose_on_the_row() {
    let ode = one_cpt_const_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let times = [4.0, 5.9, 6.0, 12.0];

    for (label, doses) in [
        (
            "EVID=3 reset@6",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        ),
        (
            "EVID=4 reset+SS@6",
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II),
                DoseEvent::new(6.0, 100.0, 1, 0.0, true, SS_CHZ_II),
            ],
        ),
    ] {
        let mut subject = make_subject(doses, vec![]);
        subject.reset_times = vec![6.0];
        subject.reset_covariates = vec![HashMap::new()];
        let got = chz_at(&ode, &pk, &subject, &times);

        // The hazard must be live going into the reset, or "zero" and "preserve" agree at
        // t = 6 and the test cannot tell them apart.
        assert!(
            got[1] > 0.1,
            "{label}: pre-reset hazard must be materially non-zero; H(5.9) = {}",
            got[1]
        );
        for (&h, &t) in got.iter().zip(&times) {
            assert!(h.is_finite(), "{label}: H({t}) = {h} is not finite");
        }
        // Before the reset the clock reads record time; after it, time since the reset.
        assert!(
            (got[0] - SS_CHZ_H0 * 4.0).abs() < 1e-9,
            "{label}: H(4) = {}, expected {}",
            got[0],
            SS_CHZ_H0 * 4.0
        );
        assert!(
            got[2].abs() < 1e-9,
            "{label}: the reset did not zero the accumulator; H(6) = {}",
            got[2]
        );
        assert!(
            (got[3] - SS_CHZ_H0 * 6.0).abs() < 1e-9,
            "{label}: H(12) = {}, expected {} (6 h after the reset)",
            got[3],
            SS_CHZ_H0 * 6.0
        );
    }
}

/// 1-cpt **Michaelis–Menten** disposition + a declared constant-rate accumulator.
///
/// Nonlinear on purpose: `periodic_ss_fixed_point_g`'s linearity self-check declines a
/// saturable disposition, so this model cannot take the exact solve and *must* run the capped
/// pulse train — which is the only path on which [`mask_chz`] is observable. `Vmax` and `Km`
/// ride the `CL`/`V` slots, as in `mm_disposition_spec`.
#[cfg(feature = "survival")]
fn mm_const_chz_spec() -> OdeSpec {
    let mut ode = one_cpt_const_chz_spec();
    ode.rhs = Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
        let vmax = p[crate::types::PK_IDX_CL];
        let km = p[crate::types::PK_IDX_V];
        dy[0] = -vmax * y[0] / (km + y[0]);
        dy[1] = SS_CHZ_H0;
    });
    ode
}

/// On a **nonlinear** joint model the equilibration must still judge convergence on the PK
/// compartments — the accumulator has none to reach.
///
/// This is the arm that makes the masking observable at all. Everywhere else `restore_chz`
/// overwrites the accumulator row on the way out and `[odes]` may not read `__chz_*`, so a
/// mask-less run reaches the same `H` and the same predictions; measured, dropping the mask
/// from either equilibration kills no other test in this file. Here it is load-bearing:
/// a saturable disposition fails the exact solve's linearity check, so the capped pulse train
/// runs, and `SsStopTracker` watches the **whole** state vector. An unmasked accumulator grows
/// by `H0 · II` every cycle and never stops moving, so the tracker can never early-stop — the
/// run burns all `SS_EQUILIBRATION_CYCLES` and then reports a non-convergence that did not
/// happen (#867's warning, fired on a PK block that settled long before).
///
/// Killed by dropping `mask_chz` from `equilibrate_ss_pk_state`'s `base_rhs`: the cycle count
/// goes to the 50-cycle cap. The `H` assertion alone does **not** die — that is the point of
/// asserting the cycle count here.
#[cfg(feature = "survival")]
#[test]
fn a_nonlinear_joint_model_judges_convergence_on_the_pk_rows() {
    let ode = mm_const_chz_spec();
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = 20.0; // Vmax
    pk.values[crate::types::PK_IDX_V] = 15.0; // Km — saturating at these amounts
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );

    let times = [0.0, 1.0, 4.0, 12.0];
    let got = chz_at(&ode, &pk, &subject, &times);
    assert_hazard_is_the_closed_form(&got, &times, "MM disposition, SS bolus");

    // The equilibration this model actually ran. It must be the capped pulse train (the
    // straddle: if a future change made this disposition take the exact solve, the arm would
    // silently stop testing the train), and it must have early-stopped well inside the cap.
    let cycles = crate::dosing::last_ss_equilibration_cycles();
    assert!(
        cycles > 1,
        "this fixture is meant to be nonlinear enough to decline the exact solve, but it took \
         it ({cycles} cycle) — the pulse-train path is no longer under test here"
    );
    assert!(
        cycles < crate::dosing::SS_EQUILIBRATION_CYCLES,
        "the pulse train hit the {}-cycle cap ({cycles} cycles): the stop tracker is watching a \
         row that never settles, which is what masking the accumulator prevents",
        crate::dosing::SS_EQUILIBRATION_CYCLES
    );
}

// ── #1218 — a single-instant `saveat` reads the post-dose state, never the `NaN` prefill ──
//
// `ode_dense_solve_states` takes its horizon as the largest `saveat`, so a grid with no point
// past the first event builds a one-break timeline. The `windows(2)` loop never runs on it,
// and until #1218 the post-loop `#731` left-boundary visit was skipped for exactly that
// timeline — "a single-instant `saveat` keeps its prior behaviour", and the prior behaviour was
// the `f64::NAN` prefill. `predict_survival(&[0.0])` returned `NaN, NaN` while `[0.0, 1.0]`
// returned a finite `t = 0` row; `ode_predictions_event_driven_with_states` returned a finite
// `ipred` over an all-`NaN` state row for a subject whose only observation sits on its dose.
//
// The wrong fix is as quiet as the bug: filling the node from the *seeded* initial state
// (widening the pre-first-event prefill from `<` to `<=`) gives `H(0) = 0` and a pre-dose
// `h(0)`. On a drug-driven hazard that is the hazard at zero drug — `0.02` where the
// multi-point grid reads `0.219` on #1210's fixture. A constant hazard cannot tell the two
// apart, so the arms below use a drug-driven accumulator and compare the single-instant row
// against the multi-point row on **every** state, bit for bit.

/// Exposure slope of the drug-driven accumulator below. Small enough that `h` stays O(1) at
/// an SS peak of ~143 amount units, large enough that "post-dose" and "seeded" differ by far
/// more than a rounding error.
#[cfg(feature = "survival")]
const SS_CHZ_BETA: f64 = 0.01;

/// 1-cpt IV bolus + a **drug-driven** accumulator: `d/dt(__chz) = H0 · exp(BETA · central)`.
/// Same declaration as [`one_cpt_const_chz_spec`]; the hazard reads the state, so a row read
/// from the wrong state is visible in `h` as well as in the compartments.
#[cfg(feature = "survival")]
fn one_cpt_drug_chz_spec() -> OdeSpec {
    OdeSpec {
        chz_state_slots: vec![1],
        rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
            dy[1] = SS_CHZ_H0 * (SS_CHZ_BETA * y[0]).exp();
        }),
        n_states: 2,
        state_names: vec!["central".into(), "__chz_3".into()],
        readout: OdeReadout::ObsCmt(0),
        diffusion_var: Vec::new(),
        solver_opts: OdeSolverOptions {
            abstol: 1e-11,
            reltol: 1e-9,
            ..OdeSolverOptions::default()
        },
        input_rate: Vec::new(),
        rhs_program: None,
        readout_program: None,
        indiv_param_program: None,
        dose_attr_map: Default::default(),
        init_fn: None,
    }
}

/// Every state of `got` is finite and bit-identical to `want`.
///
/// Finiteness first and on its own: `NaN == NaN` is `false`, so the equality would fail on a
/// `NaN` row too, but it would say "bits differ" about a row that was never computed.
#[cfg(feature = "survival")]
fn assert_states_bit_identical(got: &[f64], want: &[f64], arm: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{arm}: state vectors differ in length"
    );
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{arm}: state {k} is not finite on both sides: got {g}, want {w}"
        );
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{arm}: state {k} differs: got {g}, want {w}"
        );
    }
}

/// `[0.0]` on an SS dose at `t = 0` must equal the `t = 0` row of `[0.0, 1.0]` — every state,
/// bit for bit — and that row must be the *post-dose* state.
///
/// Killed by: the `len >= 2` guard restored (the row is the `NaN` prefill); the `<=` prefill
/// variant (central reads the seeded `0.0` instead of trough + pulse — `H` still reads `0.0`
/// under that variant, which is why the comparison is on every state and not on `H`); a
/// second visit of the sole break (another pulse moves central by 100).
#[cfg(feature = "survival")]
#[test]
fn a_single_instant_grid_on_an_ss_dose_reads_the_post_dose_state() {
    let ode = one_cpt_drug_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );
    let single = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[0.0]);
    let multi = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[0.0, 1.0]);
    assert_eq!(single.len(), 1, "one row per saveat");
    assert_eq!(multi.len(), 2, "one row per saveat");
    assert_states_bit_identical(&single[0], &multi[0], "[0.0] vs [0.0, 1.0]");

    // The straddle, asserted: the row is the post-SS-dose state, not the seeded one. An SS
    // bolus lands on its own trough, so central exceeds the bare 100 mg pulse, and the
    // hazard read off it is not the drug-free `H0` the seeded state would give.
    let got = &single[0];
    assert!(
        got[0] > 100.0,
        "central at t = 0 is {} — the seeded pre-dose state, not trough + pulse",
        got[0]
    );
    assert_eq!(got[1], 0.0, "H(0) must be exactly 0.0 at the first record");
    let mut du = vec![0.0; ode.n_states];
    (ode.rhs)(got, &pk.values, 0.0, &mut du);
    assert!(
        du[1] > 2.0 * SS_CHZ_H0,
        "h(0) = {} reads as the drug-free hazard {SS_CHZ_H0}; the row is not post-dose",
        du[1]
    );
}

/// Duplicate nodes at the instant are all written, and a node *before* the first event keeps
/// the seeded initial state (nothing has acted on the system yet) while the node on the event
/// reads post-dose.
///
/// The `-1.0` row is the pre-first-event prefill's territory and must not move: widening that
/// prefill to cover the instant is the wrong fix, and this is the arm that would see the two
/// rows collapse onto the same (seeded) state.
#[cfg(feature = "survival")]
#[test]
fn a_single_instant_grid_writes_every_duplicate_and_leaves_pre_event_nodes_seeded() {
    let ode = one_cpt_drug_chz_spec();
    let pk = pk_one(1.0, 10.0);
    let subject = make_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, SS_CHZ_II)],
        vec![],
    );
    let want = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[0.0, 1.0]);

    let dup = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[0.0, 0.0]);
    assert_eq!(dup.len(), 2, "one row per saveat, duplicates included");
    assert_states_bit_identical(&dup[0], &want[0], "[0.0, 0.0] row 0");
    assert_states_bit_identical(&dup[1], &want[0], "[0.0, 0.0] row 1");

    let pre = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[-1.0, 0.0]);
    assert_eq!(pre.len(), 2);
    let seeded = ode.initial_state(&pk.values);
    assert_states_bit_identical(&pre[0], &seeded, "[-1.0, 0.0] row 0 (pre-event)");
    assert_states_bit_identical(&pre[1], &want[0], "[-1.0, 0.0] row 1");
    assert!(
        pre[0][0] < pre[1][0],
        "the pre-event row and the on-event row collapsed onto one state: {pre:?}"
    );
}

/// `ode_predictions_event_driven_with_states` hands `subject.obs_times` to the dense solve, so
/// a subject whose only observation sits on its dose returned `ipred = [100.0]` over
/// `states = [[NaN]]` — a finite prediction with a `NaN` state behind it, which is what a
/// `[derived]` output column reads. Both a plain bolus and an SS dose, since they take
/// different arms of `apply_segment_boundary` at the sole break.
///
/// The readout is the compartment amount, so the state must also equal `ipred` exactly — the
/// invariant the two-pass engine promises.
#[test]
fn with_states_on_a_single_instant_timeline_returns_the_post_dose_state() {
    let ode = one_cpt_ode_spec();
    let pk = pk_one(1.0, 10.0);
    let at = |n: usize| vec![pk; n];
    for (arm, ss, ii) in [("bolus", false, 0.0), ("ss", true, 12.0)] {
        let one = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, ss, ii)], vec![0.0]);
        let two = make_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, ss, ii)],
            vec![0.0, 1.0],
        );
        let (ipred1, states1) = ode_predictions_event_driven_with_states(
            &ode,
            &one,
            &[],
            &[],
            &at(1),
            &at(1),
            &[],
            &[],
        );
        let (ipred2, states2) = ode_predictions_event_driven_with_states(
            &ode,
            &two,
            &[],
            &[],
            &at(1),
            &at(2),
            &[],
            &[],
        );
        assert_eq!(states1.len(), 1, "{arm}: one state row per observation");
        assert!(
            states1[0][0].is_finite(),
            "{arm}: the single-instant state row is {:?}",
            states1[0]
        );
        assert_eq!(
            states1[0][0].to_bits(),
            states2[0][0].to_bits(),
            "{arm}: single-observation state {} differs from the two-observation row {}",
            states1[0][0],
            states2[0][0]
        );
        assert_eq!(
            states1[0][0].to_bits(),
            ipred1[0].to_bits(),
            "{arm}: state {} and ipred {} disagree on an amount readout",
            states1[0][0],
            ipred1[0]
        );
        assert_eq!(
            ipred1[0].to_bits(),
            ipred2[0].to_bits(),
            "{arm}: ipred moved"
        );
        assert!(
            ipred1[0] >= 100.0,
            "{arm}: the post-dose amount is {}",
            ipred1[0]
        );
    }
}
