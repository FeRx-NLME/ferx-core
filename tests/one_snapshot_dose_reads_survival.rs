//! #1575 — an ODE-accumulated (drug-driven) hazard reads every dose-record quantity at its
//! dose record.
//!
//! `survival::ode_cumhaz_hazard` (the TTE objective's dedicated engine and
//! `predict_survival`) and `draw_ode_tte_latent` (`simulate()`'s event times) integrate the
//! augmented ODE under one disposition snapshot, `$PK` at `TIME = 0` (#610). Until #1575
//! they read every dose-record quantity there too — absorption kernel, compartment lag, the
//! `TAD` anchor — so a `$PK` that differed at `TIME = 0` from the dose records moved `H`,
//! `h`, the objective and the event times by up to the whole dose (measured: `H(30)` −97 %,
//! OFV −36.5), while every dose record was in domain and no `E_` fired.
//!
//! The reachable axis is a `TIME`-reading `$PK`: an ODE hazard already refuses every
//! time-varying covariate (#741, `check_survival_tv_covariates`). Such a subject declines
//! the #570 one-solve share (`model_uses_time_anywhere`) and lands on the dedicated engine
//! this file exercises.
//!
//! **Engine.** Every fixture here is the dense f64 walk (`ode_dense_solve_states_reading`)
//! or the threshold walk (`ode_solve_until_chz_threshold`). A subject carrying a TTE record
//! resolves its inner η-gradient to finite differences (`subject_has_survival_records`):
//! there is **no** `Dual2` twin of this path, so there is no parity test to add. That is
//! pinned at Tier 1 (`inner_optimizer::tests::an_ode_hazard_subject_resolves_its_inner_
//! gradient_to_fd`) rather than here, because `FitResult::gradient_method_inner` is a
//! model-level label and reads `analytic (Dual2)` on this model — the run banner's
//! per-subject route says FD.
//!
//! **The fixture.** `DROP` is a literal spliced into `$PK` that only ever acts at
//! `TIME < 0.5` (a literal, not a theta: a `0 FIX` theta is packed on the log scale and
//! floored to 1e-10). The bad arm sets it to 10, the control to 0; doses land at 1 and 13
//! (the second on residual drug),
//! so every dose record reads `MTT = 1`, `ALAG1 = 0.5` in both arms, while the `TIME = 0`
//! disposition snapshot reads `MTT = −9`, `ALAG1 = 10.5` in the bad one. `DROP` reaches
//! nothing else — not CL, V, KA, the hazard or `init()` — so the two arms differ only in a
//! snapshot no dose read takes, and must agree bit for bit. The hazard reads `TAD`, so the
//! readout `h(t)` also sees which lags its `TAD` anchor was built from.

#![cfg(feature = "survival")]

mod common;

use ferx_core::api::{check_model_data, read_population_for};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, Population};
use ferx_core::{fit, EstimationMethod, FitOptions};
use std::collections::HashMap;
use std::io::Write;

const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKA(1.0, FIX)
  theta TVMTT(1.0, FIX)
  theta TVN(3.0, FIX)
  theta TVLAG(0.5, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  MTT   = TVMTT - DROP_VALUE * (if (TIME < 0.5) 1.0 else 0.0)
  NTR   = TVN
  ALAG1 = TVLAG + DROP_VALUE * (if (TIME < 0.5) 1.0 else 0.0)
  H0    = TVH0
  BETA  = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA * depot
  d/dt(central) = KA * depot - (CL/V) * central

[scaling]
  obs_scale = V

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V)) + 0.002 * TAD

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  ode_reltol = 1e-9
  ode_abstol = 1e-11
";

/// Doses at 1 and 13 into the transit compartment (CMT 1), a PK observation at 8 (CMT 2),
/// and an exact event at 20 (CMT 3).
const DATA: &str = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
                    1,1,.,1,100,1,0,1,0,0\n\
                    1,8,5,0,.,2,0,0,0,0\n\
                    1,13,.,1,100,1,0,1,0,0\n\
                    1,20,1,0,0,3,0,0,0,0\n";

const GRID: [f64; 8] = [0.5, 2.0, 5.0, 10.0, 14.0, 17.5, 20.0, 30.0];

fn model(drop: f64) -> CompiledModel {
    parse_model_string(&MODEL.replace("DROP_VALUE", &format!("{drop:?}")))
        .expect("the TIME-reading joint PK-TTE model parses")
}

fn population(model: &CompiledModel) -> Population {
    let mut f = tempfile::NamedTempFile::new().expect("create temp csv");
    f.write_all(DATA.as_bytes()).expect("write temp csv");
    let path = f.path().to_str().expect("temp path is utf-8");
    let (pop, _) = read_population_for(model, &None, path, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    assert_eq!(pop.subjects.len(), 1, "fixture is a single subject");
    pop
}

/// `param`'s value in the model's own `$PK` at `TIME = t`.
fn pk_at(model: &CompiledModel, param: &str, t: f64) -> f64 {
    let slot = model.pk_indices[model
        .indiv_param_names
        .iter()
        .position(|n| n == param)
        .unwrap_or_else(|| panic!("no individual parameter {param}"))];
    let eta = vec![0.0; model.n_eta];
    (model.pk_param_fn)(&model.default_params.theta, &eta, &HashMap::new(), t).values[slot]
}

/// The straddle, through `pk_param_fn`: at `TIME = 0` — the snapshot the hazard pass
/// integrates under — `MTT` and `ALAG1` differ between the arms (out of domain in the bad
/// one), while at both dose records they are the control's in-domain values. Without it the
/// arms could silently stop differing, and the bit-identity below would be a tautology.
fn assert_straddle(bad: &CompiledModel, ctl: &CompiledModel) {
    assert_eq!(pk_at(bad, "MTT", 0.0), -9.0, "bad MTT at TIME = 0");
    assert_eq!(pk_at(ctl, "MTT", 0.0), 1.0, "control MTT at TIME = 0");
    assert_eq!(pk_at(bad, "ALAG1", 0.0), 10.5, "bad ALAG1 at TIME = 0");
    assert_eq!(pk_at(ctl, "ALAG1", 0.0), 0.5, "control ALAG1 at TIME = 0");
    for t in [1.0, 13.0] {
        for m in [bad, ctl] {
            assert_eq!(pk_at(m, "MTT", t), 1.0, "MTT at the dose record t = {t}");
            assert_eq!(
                pk_at(m, "ALAG1", t),
                0.5,
                "ALAG1 at the dose record t = {t}"
            );
        }
    }
}

fn bits(v: impl IntoIterator<Item = f64>) -> Vec<u64> {
    v.into_iter().map(f64::to_bits).collect()
}

/// H1, `predict_survival`: `(H, h)` on a grid bit-identical to the control. Dies if
/// `ode_cumhaz_hazard` hands the dense walk `Shared` reads (the clamped `MTT = −9` kernel
/// delivers no drug and the 10.5 lag moves every arrival), and — separately — if the `h(t)`
/// readout's `TAD` anchor is built from the `TIME = 0` lags while the walk read per dose:
/// only `h` moves then, and only because the hazard reads `TAD`.
#[test]
fn predict_survival_reads_dose_quantities_at_the_dose_record() {
    let (bad, ctl) = (model(10.0), model(0.0));
    assert_straddle(&bad, &ctl);
    let pop = population(&bad);
    let diags = check_model_data(&bad, &pop);
    assert!(
        !diags.iter().any(|d| d.code.starts_with("E_")),
        "every dose record is in domain, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );

    let rows = |m: &CompiledModel| {
        ferx_core::predict_survival(m, &population(m), &m.default_params, &GRID)
            .expect("predict_survival runs")
    };
    let (b, c) = (rows(&bad), rows(&ctl));
    assert_eq!(c.len(), GRID.len(), "one row per grid point");
    for r in &c {
        assert!(
            r.cum_hazard.is_finite() && r.hazard.is_finite(),
            "control row at t = {}: H {} h {}",
            r.time,
            r.cum_hazard,
            r.hazard
        );
    }
    assert_eq!(
        bits(b.iter().map(|r| r.cum_hazard)),
        bits(c.iter().map(|r| r.cum_hazard)),
        "H(t) read a dose quantity at TIME = 0: {:?} vs {:?}",
        b.iter().map(|r| r.cum_hazard).collect::<Vec<_>>(),
        c.iter().map(|r| r.cum_hazard).collect::<Vec<_>>()
    );
    assert_eq!(
        bits(b.iter().map(|r| r.hazard)),
        bits(c.iter().map(|r| r.hazard)),
        "h(t) read a dose quantity (or its TAD anchor's lag) at TIME = 0: {:?} vs {:?}",
        b.iter().map(|r| r.hazard).collect::<Vec<_>>(),
        c.iter().map(|r| r.hazard).collect::<Vec<_>>()
    );

    // The drug is live in the control: a pass that lost the dose in both arms would agree on
    // the drug-free value and pass the bit-identity above. Measured: `H(30)` = 1.48498 against
    // the drug-free (`BETA = 0`) 0.99600, a gap of 0.489; bound at 0.3.
    let h30 = c.last().expect("t = 30 row").cum_hazard;
    let drug_free = live_margin();
    assert!(
        h30 - drug_free > 0.3,
        "control H(30) = {h30} must clear the drug-free {drug_free} by the measured margin"
    );
}

/// `H(30)` with `BETA = 0`: the same model and data with the drug's effect switched off —
/// the value both arms would agree on if the dose were lost in both.
fn live_margin() -> f64 {
    let m = parse_model_string(
        &MODEL
            .replace("DROP_VALUE", "0.0")
            .replace("TVBETA(0.1, FIX)", "TVBETA(0.0, FIX)"),
    )
    .expect("drug-free twin parses");
    ferx_core::predict_survival(&m, &population(&m), &m.default_params, &[30.0])
        .expect("predict_survival runs")[0]
        .cum_hazard
}

/// H1, the objective: `fit(maxiter = 0)` OFV bit-identical to the control. Dies with the
/// same `ode_cumhaz_hazard` mutation as the arm above, through the TTE likelihood rather
/// than `predict_survival`.
#[test]
fn the_objective_reads_dose_quantities_at_the_dose_record() {
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ode_reltol: 1e-9,
        ode_abstol: 1e-11,
        ..FitOptions::default()
    };
    let run = |m: &CompiledModel| {
        fit(m, &population(m), &m.default_params, &opts).expect("maxiter-0 fit runs")
    };
    let (bad, ctl) = (model(10.0), model(0.0));
    assert_straddle(&bad, &ctl);
    let (b, c) = (run(&bad), run(&ctl));
    // `is_finite()` cannot see a repelled subject: `tte_nll_from_curves` maps a NaN `H` to
    // `1e20`, a finite `f64`. The real objective is O(10), so bound the magnitude.
    assert!(
        c.ofv.is_finite() && c.ofv.abs() < 1e6,
        "control objective {} must be a real objective, not the 1e20 repel sentinel",
        c.ofv
    );
    assert_eq!(
        b.ofv.to_bits(),
        c.ofv.to_bits(),
        "the TTE objective read a dose quantity at TIME = 0: {} vs {}",
        b.ofv,
        c.ofv
    );
}

/// H2, `simulate()`: event times off a fixed seed, finite horizon, bit-identical to the
/// control. Dies if `draw_ode_tte_latent` hands the threshold walk `Shared` reads, or if
/// that walk's input-rate forcings stay on the shared snapshot.
#[test]
fn simulated_event_times_read_dose_quantities_at_the_dose_record() {
    let opts = ferx_core::SimulateOptions {
        horizon: Some(30.0),
        seed: Some(1575),
        match_method: None,
    };
    let draw = |m: &CompiledModel| -> Vec<f64> {
        ferx_core::api::simulate_with_options(m, &population(m), &m.default_params, 40, &opts)
            .expect("simulation runs")
            .iter()
            .filter(|s| s.cmt == 3)
            .map(|s| s.time)
            .collect()
    };
    let (bad, ctl) = (model(10.0), model(0.0));
    assert_straddle(&bad, &ctl);
    let (b, c) = (draw(&bad), draw(&ctl));
    assert_eq!(c.len(), 40, "one event row per replicate");
    let n_events = c.iter().filter(|&&t| t < 30.0 - 1e-9).count();
    assert!(
        n_events > 0 && n_events < c.len(),
        "the control must both fire and censor across 40 draws ({n_events} events), or the \
         times cannot tell a hazard that saw the drug from one that did not"
    );
    assert_eq!(
        bits(b.iter().copied()),
        bits(c.iter().copied()),
        "simulated event times read a dose quantity at TIME = 0: {b:?} vs {c:?}"
    );
}
