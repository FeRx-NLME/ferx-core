//! `[scaling]` Form A/B on the adaptive path (#2129).
//!
//! `simulate_adaptive` returned the bare readout — the amount — wherever the model
//! divided it by `obs_scale`, in the trajectory rows *and* in the signal an
//! observe-less monitor hands the controller, so a controller titrated against a
//! value `V`× too high. Validation follows the adaptive-dosing exception in AGENTS.md
//! (no NONMEM): the degenerate oracle (a holding controller must equal `simulate()`
//! row for row) and the Form C twin (`obs_scale = V` must equal `y = central / V`
//! under the same seed — Form C was always honoured, so it is the reference that sees
//! the monitor, the IOV window and the AUC pass, which `simulate()` has no analogue of).
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, Comparison, ControllerCtx,
    DoseAction, DoseStep, MonitorSpec, ObserveMode,
};
use std::collections::HashMap;

/// 1-cpt ODE plus an independent second state, so a per-CMT Form B has two observed
/// compartments. `V` and the error model are spliced in per case; a `KAPPA_V` model
/// declares the κ (and so is an IOV model) only when `V` reads it.
fn model_src(readout: &str, v: &str, error: &str) -> String {
    let kappa = if v.contains("KAPPA_V") {
        "  kappa KAPPA_V ~ 0.04"
    } else {
        ""
    };
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_V ~ 0.09
{kappa}
  sigma S ~ 0.04
[individual_parameters]
  CL = TVCL * WT / 70
  V  = {v}
[structural_model]
  ode(obs_cmt=central, states=[central, periph])
[odes]
  d/dt(central) = -(CL / V) * central
  d/dt(periph) = -0.2 * periph
[scaling]
{readout}
[error_model]
{error}
"#
    )
}

const PROP: &str = "  DV ~ proportional(S)";
const ADD: &str = "  DV ~ additive(S)";

fn model(readout: &str, v: &str, error: &str) -> CompiledModel {
    parse_model_string(&model_src(readout, v, error)).expect("parse")
}

const OBS: [f64; 4] = [1.0, 4.0, 13.0, 20.0];
const DECISIONS: [f64; 2] = [4.0, 12.0];

/// One subject: a base bolus into each state at 0, observations at `OBS` on `cmts`,
/// `WT = 70` — or, with `tv_wt`, `WT` moving 70 → 90 across the observation rows (the
/// issue's time-varying table).
fn population(cmts: [usize; 4], tv_wt: bool) -> Population {
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(0.0, 80.0, 2, 0.0, false, 0.0),
    ];
    let n = OBS.len();
    let s = Subject {
        obs_rgroup: Vec::new(),
        id: "1".to_string(),
        dose_occasions: vec![1u32; doses.len()],
        doses,
        obs_times: OBS.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: cmts.to_vec(),
        covariates: wt(70.0),
        dose_covariates: Vec::new(),
        obs_covariates: if tv_wt {
            vec![wt(70.0), wt(90.0), wt(90.0), wt(90.0)]
        } else {
            Vec::new()
        },
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1u32; n],
        obs_l2: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    };
    Population {
        subjects: vec![s],
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// A holding controller with one monitor on CMT 1 — the degenerate controller: the
/// realized regimen is the base regimen, so the run must equal `simulate()` on it.
fn run_hold(m: &CompiledModel, pop: &Population, mode: ObserveMode) -> AdaptiveSimulationResult {
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: DECISIONS.to_vec(),
        monitors: vec![MonitorSpec::new("A", 1, mode)],
        ..Default::default()
    };
    let hold = || |_c: &ControllerCtx| vec![DoseAction::Hold];
    simulate_adaptive(m, pop, &m.default_params, 1, hold, &opts).expect("adaptive run")
}

fn ipreds(r: &AdaptiveSimulationResult) -> Vec<f64> {
    r.trajectories.iter().map(|t| t.ipred).collect()
}

fn values(r: &AdaptiveSimulationResult) -> Vec<f64> {
    r.trajectories
        .iter()
        .map(|t| match t.outcome {
            SimOutcome::Continuous { value } => value,
            ref o => panic!("continuous trajectory row expected, got {o:?}"),
        })
        .collect()
}

fn signals(r: &AdaptiveSimulationResult) -> Vec<f64> {
    r.decisions
        .iter()
        .map(|d| d.observed_signals[0].value)
        .collect()
}

/// Worst relative error of `got` against `want`, refusing a non-finite entry rather than
/// folding it away (`f64::max` discards `NaN`).
fn worst_rel(got: &[f64], want: &[f64], label: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: row count");
    got.iter().zip(want).fold(0.0, |w, (g, x)| {
        assert!(g.is_finite() && x.is_finite(), "{label}: {g} vs {x}");
        w.max((g - x).abs() / x.abs().max(1e-300))
    })
}

/// Realised worst relative error across the cases below is 4.1e-16 (two ulps of a
/// different association between the divided readout and Form C's `central / V`);
/// 1e-12 leaves four orders of headroom and is still nine orders inside the smallest
/// defect the cases can show (the ×2 of `obs_scale = TVV / 25`).
const TOL: f64 = 1e-12;

/// The degenerate oracle (#2129's table): a holding controller reproduces `simulate()`
/// row for row, `ipred` and the trajectory value, under Form A (constant, θ-, an
/// individual parameter, η-dependent), Form B (per-CMT, observations on both CMTs) and
/// Form C, on a constant and on a time-varying covariate.
///
/// Same seed ⇒ same η: both entry points draw η first from one seeded stream, which the
/// η-dependent case would otherwise expose as a different trajectory, not a tolerance.
///
/// Regression: drop the `apply_scaling` call on the adaptive trajectory rows and every
/// Form A/B case comes back as the amount — ×50, ×1000, ×2, ×50·e^η — while the Form C
/// rows stay green. The straddle assertion keeps each Form A/B case live: its scaled
/// rows must sit at least 1% away from the bare amount, so a fixture whose divisor
/// collapses to 1 fails here instead of passing vacuously.
#[test]
fn holding_controller_matches_simulate_under_every_scaling_form() {
    let per_cmt_b = "  obs_scale[CMT=1] = V\n  obs_scale[CMT=2] = 10";
    let per_cmt_c = "  y[CMT=1] = central / V\n  y[CMT=2] = periph / 10";
    let per_cmt_err = "  CMT=1: DV ~ proportional(S)\n  CMT=2: DV ~ proportional(S)";
    let cases: [(&str, &str, &str, [usize; 4], bool); 7] = [
        ("A: obs_scale = V", "  obs_scale = V", "TVV", [1; 4], true),
        (
            "A: obs_scale = 1000",
            "  obs_scale = 1000",
            "TVV",
            [1; 4],
            true,
        ),
        (
            "A: obs_scale = TVV / 25",
            "  obs_scale = TVV / 25",
            "TVV",
            [1; 4],
            true,
        ),
        (
            "A: obs_scale = V, V = TVV·exp(ETA_V)",
            "  obs_scale = V",
            "TVV * exp(ETA_V)",
            [1; 4],
            true,
        ),
        ("B: obs_scale[CMT=n]", per_cmt_b, "TVV", [1, 2, 1, 2], true),
        (
            "C: y = central / V",
            "  y = central / V",
            "TVV",
            [1; 4],
            false,
        ),
        ("C: y[CMT=n]", per_cmt_c, "TVV", [1, 2, 1, 2], false),
    ];
    for tv_wt in [false, true] {
        for (label, readout, v, cmts, divides) in cases {
            let label = format!("{label} (time-varying WT: {tv_wt})");
            let err = if cmts == [1; 4] { PROP } else { per_cmt_err };
            let m = model(readout, v, err);
            let pop = population(cmts, tv_wt);
            let adaptive = run_hold(&m, &pop, ObserveMode::Ipred);
            let reference = simulate_with_seed(&m, &pop, &m.default_params, 1, 7).expect("sim");
            let want: Vec<f64> = reference.iter().map(|r| r.ipred).collect();
            let w = worst_rel(&ipreds(&adaptive), &want, &label);
            assert!(w <= TOL, "{label}: adaptive ipred vs simulate() {w:e}");
            // The trajectory value is the individual prediction (no residual draw), so
            // it must carry the same scale — the issue's "DV" column.
            let w = worst_rel(&values(&adaptive), &want, &label);
            assert!(w <= TOL, "{label}: adaptive value vs simulate() {w:e}");

            if divides {
                // The bare readout of the same run: the identical model with no divisor.
                let bare = model("  y = central", v, PROP);
                let raw = ipreds(&run_hold(
                    &bare,
                    &population([1; 4], tv_wt),
                    ObserveMode::Ipred,
                ));
                for (j, (s, r)) in ipreds(&adaptive).iter().zip(&raw).enumerate() {
                    assert!(
                        (s - r).abs() > 0.01 * r.abs(),
                        "{label}: row {j} scaled {s} is within 1% of the bare amount {r}"
                    );
                }
            }
        }
    }
}

/// The controller's signal is the model's readout (#2129's unmeasured half): an
/// observe-less monitor — the programmatic `MonitorSpec`, and `with_assay_error`'s Dv —
/// must read `obs_scale = V` exactly as it reads Form C `y = central / V`, and the Dv σ
/// must be drawn on that scaled value. Additive error is the discriminating case: a
/// proportional σ scales with the readout, so a ×V latent noised proportionally is still
/// ×V of the right reading, while an additive σ on the amount is `V`× too small.
///
/// Regression: leave the monitor's divisor unset (`obs_scale: None` in
/// `run_adaptive_population`) → every Form A signal is ×50 (Ipred: 67.03 vs 1.3406 at
/// t = 4). Scaling the latent *after* the assay draw instead → the additive Dv case
/// carries σ/50 of noise and its signal leaves the Form C twin by up to 30% (measured);
/// the proportional cases stay green under that mutation, as argued above.
#[test]
fn monitor_signal_reads_the_scaled_readout() {
    let pop = population([1; 4], false);
    for (error, elabel) in [(ADD, "additive"), (PROP, "proportional")] {
        for (mode, mlabel) in [(ObserveMode::Ipred, "Ipred"), (ObserveMode::Dv, "Dv")] {
            let label = format!("{mlabel} monitor, {elabel} error");
            let a = run_hold(&model("  obs_scale = V", "TVV", error), &pop, mode);
            let c = run_hold(&model("  y = central / V", "TVV", error), &pop, mode);
            let w = worst_rel(&signals(&a), &signals(&c), &label);
            assert!(w <= TOL, "{label}: Form A signal vs Form C twin {w:e}");
            if mode == ObserveMode::Ipred {
                // The decision at t = 4 coincides with an observation: the controller
                // reads what that trajectory row reports.
                let row = ipreds(&a)[1];
                assert!(
                    (signals(&a)[0] - row).abs() <= TOL * row,
                    "{label}: signal {} at t=4 != trajectory row {row}",
                    signals(&a)[0]
                );
            }
        }
    }
}

/// An `observe` expression names its own signal and is not divided again: under
/// `obs_scale = V`, `observe = central / V` is the concentration, not concentration / V.
/// This is the other side of the monitor gate above — the divisor applies to the
/// model's readout, never to a user's expression.
///
/// Regression: attach the divisor to every monitor (drop `am.observe.is_none()`) → the
/// signal is 1/50 of the concentration (0.02681 vs 1.3406 at t = 4).
#[test]
fn observe_expression_is_not_divided_by_obs_scale() {
    let m = model("  obs_scale = V", "TVV", PROP);
    let spec = from_spec_hold(Some("central / V"), false);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let pop = population([1; 4], false);
    let r = simulate_adaptive_from_spec(&m, &pop, &m.default_params, 1, &spec, &opts)
        .expect("observe run");
    // V = 50 and the 100 bolus decays at CL/V = 0.1 /h: 2·e^{-0.4} at the t = 4 decision.
    let want = 2.0 * (-0.4f64).exp();
    let got = signals(&r)[0];
    assert!(
        (got - want).abs() <= 1e-3 * want,
        "observe = central / V read {got}, want the concentration {want}"
    );
}

/// A from-spec block that re-issues a zero dose at each decision (the dose bounds pin
/// it, and its one rule never fires), so the realized regimen is the base regimen.
fn from_spec_hold(observe: Option<&str>, with_assay_error: bool) -> AdaptiveDosingSpec {
    AdaptiveDosingSpec {
        observe: observe.map(str::to_string),
        observe_declared_covariates: Vec::new(),
        with_assay_error,
        assay_cmt: with_assay_error.then_some(1),
        at: DECISIONS.to_vec(),
        start_dose: 0.0,
        route: AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: (0.0, 0.0),
        confirm: 1,
        levels: None,
        target_window: None,
        auc_target: None,
        // Required by the block; it never fires (no reading is below −1e9).
        rules: vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: -1e9,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
        }],
    }
}

/// `with_assay_error` reads the assay endpoint's `[scaling]` readout — the docs promise
/// value and σ come from that one output — and the `auc_target` pass integrates the same
/// signal. The band `(1, 10)` holds the concentration AUC of the window `[4, 12]`
/// (`∫ 2·e^{-0.1t} dt = 20·(e^{-0.4} − e^{-1.2}) ≈ 7.38`) and excludes the amount AUC
/// (×50 ≈ 369), which the bare-readout control confirms.
///
/// Regression: drop the divisor on the window AUC → Form A attains 0.0 where the Form C
/// twin attains 1.0; drop it on the monitor → the first Dv signal is ×50.
#[test]
fn assay_signal_and_auc_target_use_the_scaled_readout() {
    let pop = population([1; 4], false);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let mut spec = from_spec_hold(None, true);
    spec.auc_target = Some((1.0, 10.0));
    let run = |readout: &str| {
        let m = model(readout, "TVV", ADD);
        simulate_adaptive_from_spec(&m, &pop, &m.default_params, 1, &spec, &opts)
            .expect("assay run")
    };
    let (a, c, bare) = (
        run("  obs_scale = V"),
        run("  y = central / V"),
        run("  y = central"),
    );
    let w = worst_rel(&signals(&a), &signals(&c), "with_assay_error");
    assert!(w <= TOL, "Form A assay signal vs Form C twin {w:e}");
    let attain = |r: &AdaptiveSimulationResult| r.metrics[0].auc_target_attainment;
    assert_eq!(
        attain(&c),
        Some(1.0),
        "the band holds the concentration AUC"
    );
    assert_eq!(attain(&bare), Some(0.0), "the band excludes the amount AUC");
    assert_eq!(attain(&a), attain(&c), "Form A AUC is on the Form C scale");
}

/// Under IOV the divisor follows the decision window: `obs_scale = V` with
/// `V = TVV·exp(KAPPA_V)` must equal the Form C twin row for row and signal for signal,
/// as `predict_iov` scales each occasion under its own κ.
///
/// Regression: divide the IOV rows — or the driver's signal — under the baseline η
/// (κ = 0) instead of their window's, and the rows / signals after the first decision
/// leave the twin by 36% (measured). The straddle assertion shows the κ draw actually moves the divisor: the
/// rows' implied `V` differs across windows, so a κ-blind divisor cannot pass.
#[test]
fn iov_divisor_follows_the_decision_window() {
    let pop = population([1; 4], false);
    let v = "TVV * exp(KAPPA_V)";
    for (mode, mlabel) in [(ObserveMode::Ipred, "Ipred"), (ObserveMode::Dv, "Dv")] {
        let a = run_hold(&model("  obs_scale = V", v, PROP), &pop, mode);
        let c = run_hold(&model("  y = central / V", v, PROP), &pop, mode);
        let w = worst_rel(&ipreds(&a), &ipreds(&c), mlabel);
        assert!(w <= TOL, "{mlabel}: IOV Form A rows vs Form C twin {w:e}");
        let w = worst_rel(&signals(&a), &signals(&c), mlabel);
        assert!(
            w <= TOL,
            "{mlabel}: IOV Form A signals vs Form C twin {w:e}"
        );

        let bare = ipreds(&run_hold(&model("  y = central", v, PROP), &pop, mode));
        let implied_v: Vec<f64> = bare.iter().zip(ipreds(&a)).map(|(r, s)| r / s).collect();
        assert!(
            (implied_v[0] - 50.0).abs() < 1e-9,
            "{mlabel}: the row before the first decision divides by the κ = 0 V"
        );
        assert!(
            (implied_v[1] - implied_v[0]).abs() > 0.01 * implied_v[0]
                && (implied_v[2] - implied_v[1]).abs() > 0.01 * implied_v[1],
            "{mlabel}: κ must move the divisor across windows: {implied_v:?}"
        );
    }
}

/// LTBS is refused, not returned on the linear scale: `simulate()` reports `ln(IPRED)`
/// while the adaptive path reads, noises and returns the linear readout (#2129).
///
/// Regression: delete the `log_transform` refusal → the run returns `Ok` with
/// trajectories `e^{ipred}`-off from `simulate()` (1.8097 vs 0.5931 at t = 1).
#[test]
fn log_transformed_error_model_is_refused() {
    let m = model("  y = central / V", "TVV", "  log(DV) ~ additive(S)");
    assert!(m.log_transform, "fixture is an LTBS model");
    let pop = population([1; 4], false);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: DECISIONS.to_vec(),
        ..Default::default()
    };
    let hold = || |_c: &ControllerCtx| vec![DoseAction::Hold];
    let err = simulate_adaptive(&m, &pop, &m.default_params, 1, hold, &opts)
        .expect_err("LTBS is refused");
    assert_eq!(err.code(), Some("E_ADAPTIVE_UNSUPPORTED"), "{err}");
    assert!(
        err.to_string().contains("log-transformed error model"),
        "{err}"
    );
    let err = simulate_adaptive_from_spec(
        &m,
        &pop,
        &m.default_params,
        1,
        &from_spec_hold(Some("central / V"), false),
        &AdaptiveSimulateOptions::default(),
    )
    .expect_err("LTBS is refused on the declarative path too");
    assert_eq!(err.code(), Some("E_ADAPTIVE_UNSUPPORTED"), "{err}");
}
