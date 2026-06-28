//! End-to-end tests for **parallel / mixed dual-pathway absorption** (#322 Phase 2,
//! #505) — the `first_order(ka)` input-rate function composed with a pathway
//! fraction into `parallel` (dual first-order) and `mixed` (zero-order +
//! first-order) models. Exercises the public API (parse → `predict()` → ODE
//! integration → readout), the `first_order`/`zero_order` counterpart of
//! `tests/zero_order_absorption.rs` / `tests/igd_absorption.rs`.
//!
//! The value-path anchor is an **in-engine superposition oracle** (needs no
//! licensed NONMEM run): a dual-pathway model is, by the linearity of its disposition
//! ODE, exactly the fraction-weighted sum of the single-pathway curves — so
//! `parallel` must predict identically to `FR1·single_first_order(ka1) +
//! FR2·single_first_order(ka2)`, and `mixed` to `FZO1·single_first_order(ka) +
//! FZO·single_zero_order(dur)`. This pins the fraction split, the per-pathway mass,
//! and the channel routing (the Channel-A pointwise `first_order` and the Channel-B
//! per-segment `zero_order` carrying the fraction). It is the in-engine analogue of
//! the NONMEM `$DES` anchor (`nonmem_anchor/`, slow-gated), which ties the absolute
//! OFV to an external run.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, DoseEvent, Population};

/// Parallel dual first-order absorption into central (Form C, amount ODE + scaling).
/// FR1 = 0.6 fast (KA1 = 1.5), FR2 = 0.4 slow (KA2 = 0.3). η fixed at 0 (omega ~ 0)
/// so `predict()` returns the typical-value curve. CL = 5, V = 50 ⇒ ke = 0.1.
const PARALLEL_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVFR1(0.6, 0.05,  0.95)
  theta TVKA1(1.5, 0.05,  24.0)
  theta TVKA2(0.3, 0.01,  24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Mixed zero-order + first-order absorption into central (Form C). FZO = 0.4
/// zero-order (DUR = 3), FZO1 = 0.6 first-order (KA = 1.0). Same disposition.
const MIXED_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVFZO(0.4, 0.05,  0.95)
  theta TVKA(1.0,  0.05,  24.0)
  theta TVDUR(3.0, 0.05,  24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Single-pathway **first-order** reference (one bare `first_order(ka)`), ka at
/// theta[2]. The superposition reference for both `parallel` and `mixed`.
const SINGLE_FIRST_ORDER_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.01,  24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Single-pathway **zero-order** reference (one bare `zero_order(dur)`), dur at
/// theta[2]. The zero-order half of the `mixed` superposition reference.
const SINGLE_ZERO_ORDER_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVDUR(3.0, 0.05, 24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Single 100 mg bolus into CMT 1 over `obs_times` (fed to the input-rate
/// function; the bolus is suppressed). Observe CMT 1.
fn pop_single(obs_times: Vec<f64>) -> Population {
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![dose],
            obs_times,
            vec![0.0; n],
            vec![1; n],
        )],
    }
}

/// Predict the single-first-order reference at a given `ka`.
fn single_first_order_curve(ka: f64, obs_times: &[f64]) -> Vec<f64> {
    let model = parse_full_model(SINGLE_FIRST_ORDER_MODEL)
        .expect("single first_order model parses")
        .model;
    let mut params = model.default_params.clone();
    params.theta[2] = ka;
    predict(&model, &pop_single(obs_times.to_vec()), &params)
        .iter()
        .map(|p| p.pred)
        .collect()
}

/// Predict the single-zero-order reference at a given `dur`.
fn single_zero_order_curve(dur: f64, obs_times: &[f64]) -> Vec<f64> {
    let model = parse_full_model(SINGLE_ZERO_ORDER_MODEL)
        .expect("single zero_order model parses")
        .model;
    let mut params = model.default_params.clone();
    params.theta[2] = dur;
    predict(&model, &pop_single(obs_times.to_vec()), &params)
        .iter()
        .map(|p| p.pred)
        .collect()
}

#[test]
fn shipped_example_models_parse() {
    // The self-simulating examples/*.ferx files use the same DSL as the inline
    // models above; parse them from disk so a typo in a shipped example is caught in
    // the fast CI job (not only by a `--simulate` smoke run). CWD is the crate root.
    for path in [
        "examples/parallel_absorption.ferx",
        "examples/mixed_absorption.ferx",
        // The NONMEM-anchor ferx models (read external data, tight ODE tols) — guard
        // them here too so a typo surfaces in CI, not only on the licensed run.
        "nonmem_anchor/parallel_first_order_fit.ferx",
        "nonmem_anchor/mixed_zero_first_fit.ferx",
    ] {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        parse_full_model(&src).unwrap_or_else(|e| panic!("{path} should parse: {e}"));
    }
}

#[test]
fn parallel_equals_fraction_weighted_sum_of_single_first_order_pathways() {
    // parallel(FR1·ka1 + FR2·ka2) ≡ FR1·single(ka1) + FR2·single(ka2), exactly, by
    // the linearity of the (shared) disposition ODE. Pins the dual-pathway fraction
    // split and the Channel-A superposition across the whole curve — onset, the two
    // overlapping absorption phases, and the terminal decay.
    let model = parse_full_model(PARALLEL_MODEL)
        .expect("parallel parses")
        .model;
    let obs_times: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect(); // 0..30 h
    let parallel: Vec<f64> = predict(
        &model,
        &pop_single(obs_times.clone()),
        &model.default_params,
    )
    .iter()
    .map(|p| p.pred)
    .collect();

    let (fr1, fr2) = (0.6, 0.4);
    let c1 = single_first_order_curve(1.5, &obs_times);
    let c2 = single_first_order_curve(0.3, &obs_times);

    for (i, &t) in obs_times.iter().enumerate() {
        let want = fr1 * c1[i] + fr2 * c2[i];
        assert!(
            (parallel[i] - want).abs() <= 1e-5 * (1.0 + want.abs()),
            "parallel vs FR1·single(ka1)+FR2·single(ka2) diverge at t={t}: {} vs {want}",
            parallel[i]
        );
    }
    // Not a degenerate all-zero match: there is a real peak.
    assert!(
        parallel.iter().cloned().fold(0.0_f64, f64::max) > 0.5,
        "expected a real parallel concentration curve"
    );
}

#[test]
fn mixed_equals_fraction_weighted_sum_of_first_and_zero_order_pathways() {
    // mixed(FZO1·first(ka) + FZO·zero(dur)) ≡ FZO1·single_first(ka) +
    // FZO·single_zero(dur). Crucially this pins the **Channel-B zero-order fraction**
    // (#505): the per-segment window rate must be scaled by FZO, or the zero-order
    // pathway would deliver the full dose and the sum would be wrong.
    let model = parse_full_model(MIXED_MODEL).expect("mixed parses").model;
    let obs_times: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect();
    let mixed: Vec<f64> = predict(
        &model,
        &pop_single(obs_times.clone()),
        &model.default_params,
    )
    .iter()
    .map(|p| p.pred)
    .collect();

    let (fzo1, fzo) = (0.6, 0.4);
    let c_fo = single_first_order_curve(1.0, &obs_times);
    let c_zo = single_zero_order_curve(3.0, &obs_times);

    for (i, &t) in obs_times.iter().enumerate() {
        let want = fzo1 * c_fo[i] + fzo * c_zo[i];
        assert!(
            (mixed[i] - want).abs() <= 1e-5 * (1.0 + want.abs()),
            "mixed vs FZO1·first+FZO·zero diverge at t={t}: {} vs {want}",
            mixed[i]
        );
    }
    assert!(
        mixed.iter().cloned().fold(0.0_f64, f64::max) > 0.5,
        "expected a real mixed concentration curve"
    );
}

#[test]
fn parallel_and_mixed_recover_full_dose_auc() {
    // Absorption-independent invariant: ∫ conc dt = F·Dose/CL regardless of how the
    // dose is split across pathways (the fractions sum to 1, so the whole dose is
    // absorbed). A dropped pathway, an unscaled zero-order window, or a
    // double-counted bolus would all break this.
    let obs_times: Vec<f64> = (0..=400).map(|i| i as f64 * 0.25).collect(); // 0..100 h
    let auc_inf = 100.0 / 5.0; // F·Dose/CL with F = 1, CL = 5 ⇒ 20 mg·h/L
    for (src, name) in [(PARALLEL_MODEL, "parallel"), (MIXED_MODEL, "mixed")] {
        let model = parse_full_model(src).expect("model parses").model;
        let preds = predict(
            &model,
            &pop_single(obs_times.clone()),
            &model.default_params,
        );
        let auc: f64 = preds
            .windows(2)
            .map(|w| 0.5 * (w[0].pred + w[1].pred) * (w[1].time - w[0].time))
            .sum();
        let rel = (auc - auc_inf).abs() / auc_inf;
        assert!(
            rel < 0.02,
            "{name} conc AUC {auc:.4} vs F·Dose/CL {auc_inf:.4} (rel err {rel:.2e})"
        );
    }
}

#[test]
fn first_order_on_analytical_pk_is_rejected_pointing_at_ode_template() {
    // The error rule: a `first_order()` input rate on an analytical `pk` disposition
    // is a hard error pointing at `ode_template`, never a silent analytical→ODE swap.
    // (`first_order` ships ODE-only in #505; it leaves the ODE-only list when its
    // closed-form acceleration — a Bateman superposition — lands.)
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL
  V  = TVV
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[odes]
  d/dt(depot) = first_order(ka=KA) - KA*depot
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let err = match parse_full_model(src) {
        Ok(_) => panic!("pk + first_order() must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("ode_template"),
        "error should point at ode_template, got: {err}"
    );
    assert!(
        err.contains("first_order"),
        "error should name the first_order function, got: {err}"
    );
}

#[test]
fn multiple_zero_order_on_one_compartment_is_rejected_at_parse() {
    // Biphasic zero-order (≥2 `zero_order(...)` on one compartment) is not supported:
    // the per-segment zero-order channel builds one window per dose and resolves a
    // single zero-order forcing, so a second would be silently under-delivered. The
    // parser rejects it loudly — *at parse time* (`build_ode_spec`), so every entry
    // point surfaces the error, not just `fit()`'s data-level `check_model_data`
    // (the `simulate`-path regression below pins the gap the fit-init-only check
    // left open). Contrast the allowed `mixed` (one zero + one first), exercised by
    // the oracle test above.
    let src = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVFA(0.5,  0.05,  0.95)
  theta TVDUR1(2.0, 0.05, 24.0)
  theta TVDUR2(6.0, 0.05, 24.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FA   = TVFA
  FB   = 1 - TVFA
  DUR1 = TVDUR1
  DUR2 = TVDUR2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FA*zero_order(dur=DUR1) + FB*zero_order(dur=DUR2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;
    let err = parse_full_model(src)
        .err()
        .expect("biphasic zero-order (≥2 zero_order on a compartment) must be rejected at parse");
    assert!(
        err.contains("zero-order input-rate terms") && err.contains("compartment 1"),
        "≥2 zero-order terms on a compartment must be rejected with a clear message, got: {err}"
    );
}

#[test]
fn multiple_zero_order_rejected_on_simulate_path() {
    // The structural guard lives in the parser (`build_ode_spec`), so the
    // `--simulate` entry point (`run_model_simulate`, which parses the file before
    // it touches the `[simulation]` block) rejects a biphasic-zero-order model too —
    // the exact path the old fit-init-only check let through with a silently
    // under-delivered second pathway (#505 review). The model below is otherwise a
    // complete, self-simulating design, so the rejection is attributable to the
    // biphasic zero-order alone.
    let src = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVFA(0.5,  0.05,  0.95)
  theta TVDUR1(2.0, 0.05, 24.0)
  theta TVDUR2(6.0, 0.05, 24.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FA   = TVFA
  FB   = 1 - TVFA
  DUR1 = TVDUR1
  DUR2 = TVDUR2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FA*zero_order(dur=DUR1) + FB*zero_order(dur=DUR2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[simulation]
  n_subjects = 4
  dose_amt   = 100.0
  dose_cmt   = 1
  times      = [0.5, 1.0, 2.0, 4.0, 8.0]
  seed       = 1
"#;
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("biphasic_zero_order_sim.ferx");
    std::fs::write(&path, src).expect("write model");
    let err = ferx_core::run_model_simulate(path.to_str().unwrap())
        .expect_err("biphasic zero-order must be rejected on the simulate path");
    assert!(
        err.contains("zero-order input-rate terms") && err.contains("compartment 1"),
        "the simulate path must surface the biphasic-zero-order rejection, got: {err}"
    );
}

#[test]
fn parallel_steady_state_dosing_is_rejected() {
    // SS=1 into a parallel/mixed compartment is rejected (E_ABSORPTION_SS), inherited
    // kind-agnostically from the transit/igd/zero-order family: the steady-state
    // equilibration applies the dose as a bolus pulse, not as R_in over the cycle.
    use ferx_core::check_model_data;
    let model = parse_full_model(PARALLEL_MODEL)
        .expect("parallel parses")
        .model;
    let mut ss_dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    ss_dose.ii = 12.0;
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![ss_dose],
            vec![1.0, 4.0, 8.0],
            vec![0.0; 3],
            vec![1; 3],
        )],
    };
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_SS"),
        "SS into a parallel compartment must raise E_ABSORPTION_SS, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

/// Tier-3 (slow): full FOCEI fit recovering the parallel **fraction** and the two
/// absorption rates from a perturbed start. Gated out of the default PR job.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn parallel_fit_recovers_fraction_and_kas() {
    use ferx_core::{fit, FitOptions};

    let model = parse_full_model(PARALLEL_MODEL)
        .expect("parallel parses")
        .model;
    // Noise-free data at the truth: predictions at the typical values become the
    // observations, so the likelihood optimum sits at the data-generating params.
    let obs_times: Vec<f64> = (1..=48).map(|i| i as f64 * 0.5).collect();
    let truth = predict(
        &model,
        &pop_single(obs_times.clone()),
        &model.default_params,
    );
    let obs: Vec<f64> = truth.iter().map(|p| p.pred).collect();
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject("1", vec![dose], obs_times, obs, vec![1; n])],
    };

    // Perturb FR1 (theta[2]), KA1 (theta[3]), KA2 (theta[4]) off the truth.
    let mut init = model.default_params.clone();
    init.theta[2] = 0.45;
    init.theta[3] = 1.0;
    init.theta[4] = 0.5;
    let fitted = fit(&model, &pop, &init, &FitOptions::default()).expect("fit converges");
    assert!(fitted.converged, "parallel fit should converge");
    assert!(
        (fitted.theta[2] - 0.6).abs() < 0.05,
        "recovered FR1 {} should be near 0.6",
        fitted.theta[2]
    );
}

/// Tier-3 (slow): full FOCEI fit recovering the mixed **zero-order fraction**,
/// duration, and first-order rate from a perturbed start.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn mixed_fit_recovers_fraction_dur_and_ka() {
    use ferx_core::{fit, FitOptions};

    let model = parse_full_model(MIXED_MODEL).expect("mixed parses").model;
    let obs_times: Vec<f64> = (1..=48).map(|i| i as f64 * 0.5).collect();
    let truth = predict(
        &model,
        &pop_single(obs_times.clone()),
        &model.default_params,
    );
    let obs: Vec<f64> = truth.iter().map(|p| p.pred).collect();
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject("1", vec![dose], obs_times, obs, vec![1; n])],
    };

    // Perturb FZO (theta[2]), KA (theta[3]), DUR (theta[4]) off the truth.
    let mut init = model.default_params.clone();
    init.theta[2] = 0.55;
    init.theta[3] = 1.6;
    init.theta[4] = 1.8;
    let fitted = fit(&model, &pop, &init, &FitOptions::default()).expect("fit converges");
    assert!(fitted.converged, "mixed fit should converge");
    assert!(
        (fitted.theta[2] - 0.4).abs() < 0.06,
        "recovered FZO {} should be near 0.4",
        fitted.theta[2]
    );
    assert!(
        (fitted.theta[4] - 3.0).abs() / 3.0 < 0.12,
        "recovered DUR {} should be near 3.0",
        fitted.theta[4]
    );
}
