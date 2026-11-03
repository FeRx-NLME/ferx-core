//! #1861 — a joint PK-TTE fit whose every subject carries a TTE record the inner gradient
//! cannot differentiate runs the FD inner gradient, and `fit()` must say so:
//! `FitResult::warnings` carries the "N of M subjects use finite-difference inner
//! gradients" warning naming the time-to-event record as the cause. (`gradient_method_inner`
//! read `finite differences` while no hazard had an η-channel; it is the model-level
//! best case, so since #2482 it reads analytic and the warning carries the decline.)
//!
//! Since #2482 the drug-driven hazard has an exact η-channel (the dual walk) except beside
//! a steady-state dose, which the walk declines (its value path is not the walk's). So the
//! warning arms dose to steady state, and a fourth arm pins the exact route without one.
//!
//! Before the fix the label read `analytic (Dual2)` (it is model-level and the recorder
//! serves the PK rows), and the warning, when it fired, listed only causes that did not
//! apply ("an SDE, a per-CMT `obs_scale`, …").
//!
//! Controls. `fit()` refuses a TTE endpoint no row reaches (`E_ENDPOINT_NO_RECORDS`), so
//! "the same population without its event records" is run on the same model minus its
//! `[event_model]`: every subject then runs the analytic recorder on the same PK rows, the
//! label reads analytic and no warning fires — without this a warning that fired
//! unconditionally would pass. A mixed arm (event rows on 2 of 4 subjects) pins that the
//! count is per subject, not "every subject of a TTE model".
//!
//! Tier 2: `outer_maxiter = 0`, the evaluation-only path; the warning is computed before
//! any iteration runs.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, FitOptions, FitResult};
use std::io::Write;

/// The drug-driven hazard of #1575's fixture, without its `TIME`-reading `$PK`.
const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central

[scaling]
  obs_scale = V

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
";

const WARNING: &str = "use finite-difference inner gradients";

/// Four subjects with two PK observations each; subjects `1..=n_events` also carry a TTE
/// record (an event on odd IDs, censored on even). `ss` doses every subject to steady state
/// (`SS = 1`, `II = 12`), the case the hazard's dual walk declines.
fn csv(n_events: usize, ss: bool) -> String {
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,SS,II\n");
    let (ss, ii) = if ss { (1, 12) } else { (0, 0) };
    for id in 1..=4usize {
        let (c2, c8) = (6.0 + id as f64 * 0.2, 3.0 + id as f64 * 0.1);
        csv += &format!(
            "{id},0,.,1,100,1,1,{ss},{ii}\n{id},2,{c2},0,.,2,0,0,0\n{id},8,{c8},0,.,2,0,0,0\n"
        );
        if id <= n_events {
            csv += &format!("{id},{},{},0,.,3,0,0,0\n", 15 + id, id % 2);
        }
    }
    csv
}

fn run(model: &str, n_events: usize, ss: bool) -> FitResult {
    let m = parse_model_string(model).expect("the model parses");
    let mut f = tempfile::NamedTempFile::new().expect("temp csv");
    f.write_all(csv(n_events, ss).as_bytes())
        .expect("write csv");
    let (pop, _) =
        read_population_for(&m, &None, f.path().to_str().unwrap(), None, None, None, &[])
            .expect("data loads");
    assert_eq!(pop.subjects.len(), 4);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    };
    fit(&m, &pop, &m.default_params, &opts).expect("maxiter-0 fit runs")
}

#[test]
fn an_all_tte_population_warns_and_names_the_survival_cause() {
    let r = run(MODEL, 4, true);
    // The label is the best-case, model-level route (`build_info::gradient_method_inner`):
    // this hazard has an exact channel, so it reads analytic, and the per-subject decline is
    // the warning's to report — as for any out-of-scope subject of an in-scope model.
    assert_eq!(r.gradient_method_inner, "analytic (Dual2)");
    let w = r
        .warnings
        .iter()
        .find(|w| w.contains(WARNING))
        .unwrap_or_else(|| panic!("every TTE subject runs FD: {:?}", r.warnings));
    assert!(
        w.starts_with(
            "4 of 4 subjects use finite-difference inner gradients (4 carry a time-to-event \
             (or other non-Gaussian) record whose likelihood term has no analytic η-gradient)"
        ),
        "{w}"
    );
    assert!(!w.contains("shape no analytic recorder"), "{w}");
}

#[test]
fn a_mixed_population_counts_only_its_tte_subjects() {
    let r = run(MODEL, 2, true);
    let w = r
        .warnings
        .iter()
        .find(|w| w.contains(WARNING))
        .unwrap_or_else(|| panic!("two TTE subjects run FD: {:?}", r.warnings));
    assert!(
        w.starts_with(
            "2 of 4 subjects use finite-difference inner gradients (2 carry a time-to-event"
        ),
        "{w}"
    );
}

#[test]
fn the_same_pk_rows_without_an_event_model_emit_no_fd_warning() {
    let pk_only = MODEL.replace(
        "[event_model]\n  cmt    = 3\n  hazard = H0 * exp(BETA * (central / V))\n\n",
        "",
    );
    assert!(
        !pk_only.contains("[event_model]"),
        "fixture: the endpoint is removed"
    );
    let r = run(&pk_only, 0, true);
    assert_eq!(r.gradient_method_inner, "analytic (Dual2)");
    assert!(
        !r.warnings.iter().any(|w| w.contains(WARNING)),
        "PK-only subjects run the analytic recorder: {:?}",
        r.warnings
    );
}

/// The same TTE population without the steady-state dose: the dual walk serves every
/// subject's hazard, so the inner gradient is exact and nothing warns. The pair with
/// `an_all_tte_population_warns_and_names_the_survival_cause` straddles the walk's one
/// remaining decline, so a scope that served SS (or declined everything) reddens one arm.
#[test]
fn a_drug_driven_hazard_without_steady_state_runs_the_exact_inner_gradient() {
    let r = run(MODEL, 4, false);
    assert_eq!(r.gradient_method_inner, "analytic (Dual2)");
    assert!(
        !r.warnings.iter().any(|w| w.contains(WARNING)),
        "every subject's hazard is served: {:?}",
        r.warnings
    );
}
