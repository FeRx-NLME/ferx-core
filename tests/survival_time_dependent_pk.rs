//! #1853 — every public entry point refuses a `TIME`-dependent disposition on an
//! ODE-accumulated (drug-driven) hazard.
//!
//! `survival::ode_cumhaz_hazard` (the TTE objective's dedicated engine and
//! `predict_survival`) and `survival::draw_ode_tte_latent` (`simulate()`'s event times)
//! take one `$PK` snapshot — `(model.pk_param_fn)(…, 0.0)` — and integrate the whole
//! timeline under it. Since #1575 every dose-record quantity is read at its own dose
//! record instead, so what is still frozen is the disposition the `[odes]` RHS integrates
//! (CL, V, …), the parameters the hazard expression reads, and `init()`. The **covariate**
//! axis of that freeze is refused by #741 (`check_survival_tv_covariates`); this is the
//! `TIME` axis, which was neither refused nor warned about.
//!
//! ## What was measured, on this file's model
//!
//! `CL = TVCL·(1 + KT·TIME)`, `TVCL = 1`, `V = 10`, a 100 mg bolus at `t = 0`, records at
//! 1, 8 and an exact event at 20, `hazard = H0·exp(BETA·central/V)`. Values from a Bateman
//! / Simpson computation in Python — outside both engines — against ferx at `KT = 0.05`,
//! before the refusal below was added:
//!
//! | quantity | Gaussian half (per-record) | hazard half (frozen at `TIME = 0`) |
//! |---|---|---|
//! | `central(1)`  | 9.00324522586 (ferx IPRED 9.0032452259)  | 9.04837418036 |
//! | `central(8)`  | 3.37901785895 (ferx IPRED 3.3790178590)  | 4.49328964117 |
//! | `central(20)` | 0.30653758429                            | 1.35335283237 |
//! | `H(20)`       | 0.577304587464                           | 0.635569335705 (ferx 0.635569335704) |
//! | `−log L` (exact event @20) | 4.458673834463              | 4.412257057897 |
//!
//! So ferx's Gaussian predictions are the per-record (NONMEM `$PK`-at-every-record)
//! convention to 10 significant figures, while its hazard pass is the frozen-at-`TIME = 0`
//! closed form to 1.6e-12 relative: one objective, two different concentration curves,
//! 4.4× apart at the event time. `H(20)` is 10.1 % high and the exact-event `−log L` is
//! off by 0.046 per subject. The hazard half is also **bit-identical** between `KT = 0`
//! and `KT = 0.05` — the `TIME` coefficient has no influence at all on the survival term
//! it was written for, which is why nothing downstream could notice.
//!
//! ## Why there is no NONMEM anchor here
//!
//! The change produces no numerical result to anchor: it refuses a model class. The
//! measurement above is what the refusal rests on, and its oracle is a closed form
//! computed outside both engines rather than a second engine, because the quantity in
//! dispute is *which* `$PK` snapshot an integration runs under — the thing the two ferx
//! engines disagree about, so neither can referee it.

#![cfg(feature = "survival")]

mod common;

use ferx_core::api::{read_population_for, simulate_with_options};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, Population};
use ferx_core::{fit, predict, predict_survival, EstimationMethod, FitOptions, SimulateOptions};
use std::io::Write;

/// `CL_EXPR` is spliced into `CL`; everything else is held fixed so the arms differ in
/// nothing but that one line.
const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKT(0.05, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01

[individual_parameters]
  CL   = CL_EXPR
  V    = TVV
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  obs_scale = V

[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  ode_reltol = 1e-10
  ode_abstol = 1e-12
";

/// A 100 mg bolus at 0, PK observations at 1 and 8 (CMT 1), an exact event at 20 (CMT 2).
const DATA: &str = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
                    1,0,.,1,100,1,0,1,0,0\n\
                    1,1,9,0,.,1,0,0,0,0\n\
                    1,8,4,0,.,1,0,0,0,0\n\
                    1,20,1,0,0,2,0,0,0,0\n";

/// The refusing arm: `CL` reads `TIME`.
fn time_dependent() -> CompiledModel {
    model("TVCL * (1.0 + TVKT * TIME) * exp(ETA_CL)")
}

/// The control: the same model with `TIME` gone from `CL` and nothing else changed.
fn time_free() -> CompiledModel {
    model("TVCL * exp(ETA_CL)")
}

fn model(cl: &str) -> CompiledModel {
    parse_model_string(&MODEL.replace("CL_EXPR", cl)).expect("the joint PK-TTE model parses")
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

fn fit_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        threads: Some(2),
        ode_reltol: 1e-10,
        ode_abstol: 1e-12,
        ..FitOptions::default()
    }
}

/// The message every entry point must carry, by its load-bearing fragments.
fn assert_is_the_1853_refusal(entry: &str, msg: &str) {
    for want in [
        "CMT=2",
        "`CL` reads the `TIME` built-in and the [odes] RHS reads it",
        "#1853",
        "Hold `CL` constant in TIME",
    ] {
        assert!(msg.contains(want), "{entry}: missing {want:?} in: {msg}");
    }
}

/// `fit()` — the objective. Returns `Err` before any iteration.
#[test]
fn fit_refuses_a_time_dependent_disposition_on_an_ode_hazard() {
    let m = time_dependent();
    let err = fit(&m, &population(&m), &m.default_params, &fit_opts())
        .expect_err("fit must refuse a TIME-dependent disposition on an ODE hazard");
    assert_is_the_1853_refusal("fit", &err.to_string());
}

/// `predict_survival()` — the entry point that *reports* the frozen `H`/`h`.
#[test]
fn predict_survival_refuses_a_time_dependent_disposition_on_an_ode_hazard() {
    let m = time_dependent();
    let err = predict_survival(&m, &population(&m), &m.default_params, &[1.0, 8.0, 20.0])
        .expect_err("predict_survival must refuse");
    assert_is_the_1853_refusal("predict_survival", &err.to_string());
}

/// `predict()` — the Gaussian half alone is correct on such a model, but it shares the
/// refusal so one model is not half-accepted across the public API (as #741 does).
#[test]
fn predict_refuses_a_time_dependent_disposition_on_an_ode_hazard() {
    let m = time_dependent();
    let err = predict(&m, &population(&m), &m.default_params).expect_err("predict must refuse");
    assert_is_the_1853_refusal("predict", &err.to_string());
}

/// `simulate()` — `draw_ode_tte_latent` root-finds the same frozen snapshot for its event
/// times, so the simulate chokepoint refuses too. A finite horizon is supplied, since
/// without one the ODE-TTE simulate guard would refuse first for a different reason and
/// this test would assert nothing about #1853.
#[test]
fn simulate_refuses_a_time_dependent_disposition_on_an_ode_hazard() {
    let m = time_dependent();
    let opts = SimulateOptions {
        horizon: Some(30.0),
        seed: Some(1853),
        match_method: None,
        iov_occasion: None,
    };
    let err = simulate_with_options(&m, &population(&m), &m.default_params, 1, &opts)
        .expect_err("simulate must refuse");
    assert_is_the_1853_refusal("simulate", &err.to_string());
}

/// **The straddle.** The control differs from the refusing arm in one line of `$PK` and is
/// accepted by all four entry points, each producing a live answer — so the refusals above
/// are the `TIME` read, not the fixture, the horizon, or the endpoint routing.
///
/// `H(20) > 0` is the non-degenerate part: a drug-free hazard would still give `H > 0`
/// (`h = H0` at `central = 0`), so the bound is against the drug-free `H0·20 = 0.4`, which
/// only a live drug term clears.
#[test]
fn the_time_free_control_is_accepted_by_every_entry_point() {
    let m = time_free();
    let pop = population(&m);

    let res = fit(&m, &pop, &m.default_params, &fit_opts()).expect("the control must fit");
    // `is_finite()` cannot see a repelled subject: `tte_nll_from_curves` maps a NaN `H` to
    // the finite `1e20` sentinel, so bound the magnitude instead. The real objective here
    // is O(10).
    assert!(
        res.ofv.is_finite() && res.ofv.abs() < 1e6,
        "control objective {} must be a real objective, not the 1e20 repel sentinel",
        res.ofv
    );

    let rows = predict_survival(&m, &pop, &m.default_params, &[20.0])
        .expect("the control must predict survival");
    let row = rows
        .iter()
        .find(|r| r.cmt == 2 && r.time == 20.0)
        .expect("a CMT 2 row at t = 20");
    assert!(
        row.cum_hazard.is_finite() && row.hazard.is_finite(),
        "control H(20) = {}, h(20) = {} must be finite",
        row.cum_hazard,
        row.hazard
    );
    assert!(
        row.cum_hazard > 0.4,
        "control H(20) = {} must clear the drug-free H0·20 = 0.4, or the hazard's drug \
         term is not live and the refusing arm's defect would have nothing to move",
        row.cum_hazard
    );

    let preds = predict(&m, &pop, &m.default_params).expect("the control must predict");
    assert!(
        preds.iter().all(|p| p.pred.is_finite()),
        "control predictions must be finite: {preds:?}"
    );

    let opts = SimulateOptions {
        horizon: Some(30.0),
        seed: Some(1853),
        match_method: None,
        iov_occasion: None,
    };
    let sim = simulate_with_options(&m, &pop, &m.default_params, 1, &opts)
        .expect("the control must simulate");
    assert!(
        sim.iter().any(|s| s.cmt == 2),
        "the control simulation must emit the TTE endpoint's rows"
    );
}
