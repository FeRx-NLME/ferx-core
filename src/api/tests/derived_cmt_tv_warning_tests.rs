//! `W_DERIVED_CMT_TV_ODE` / `W_DERIVED_CMT_TV_ANALYTICAL` over their input space
//! (#1855).
//!
//! Before #1855 both warnings asked only `has_tv_covariates()`, while the state routing
//! in `pk::compute_predictions_with_states` also sends a subject of a `TIME`-reading
//! model down the same path: the ODE engine froze its disposition at the first
//! observation, the analytical engine returned NaN states, and neither said a word.
//!
//! The input space is engine × TV covariate × `TIME` in the individual parameters ×
//! reset × `TIME` in the `[odes]` RHS. The first test measures which cells actually
//! give approximate states, so the table the warning is gated on is a measurement
//! rather than a reading of the code; the rest pin, cell by cell, the exact message
//! (string equality, so deleting or rewording any clause reddens a test).

use super::*;
use crate::parser::model_parser::parse_model_string;

/// A 1-cpt IV ODE model. `cl` is the CL line, `rhs_factor` multiplies the elimination
/// rate. The elimination is saturable with a `Km` far above any amount in the data, so
/// the RHS is structurally nonlinear and integrated, not propagated by the matrix
/// exponential.
fn ode_model(cl: &str, rhs_factor: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01
[individual_parameters]
  {cl}
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central * {rhs_factor} / (1 + central / 1e6)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[derived]
  c0 = compartments[0]
"#
    ))
    .expect("the ODE test model must parse")
}

fn analytical_model(cl: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01
[individual_parameters]
  {cl}
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
[derived]
  c0 = compartments[0]
"#
    ))
    .expect("the analytical test model must parse")
}

const CL_PLAIN: &str = "CL = TVCL * (WT / 70) * exp(ETA_CL)";
const CL_TIME: &str = "CL = TVCL * (WT / 70) * (1 + 0.05 * TIME) * exp(ETA_CL)";

#[derive(Clone, Copy)]
struct Cell {
    tv_cov: bool,
    reset: bool,
}

/// One subject, three IV doses (the second lands with residual drug, so a frozen
/// disposition shows on the incoming side too; the third follows the optional reset at
/// 26 h, so every observation stays positive) and observations out to 36 h. `WT` is
/// 70 at baseline; with `tv_cov` it changes on every observation row.
fn subject(cell: Cell) -> Subject {
    let obs_times = vec![1.0, 4.0, 12.0, 14.0, 24.0, 36.0];
    let n = obs_times.len();
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(30.0, 100.0, 1, 0.0, false, 0.0),
        ],
        observations: vec![4.0; n],
        obs_cmts: vec![1; n],
        covariates: wt(70.0),
        dose_covariates: if cell.tv_cov {
            vec![wt(70.0), wt(100.0), wt(120.0)]
        } else {
            Vec::new()
        },
        obs_covariates: if cell.tv_cov {
            (0..n).map(|i| wt(70.0 + 10.0 * i as f64)).collect()
        } else {
            Vec::new()
        },
        obs_times,
        obs_raw_times: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: if cell.reset { vec![26.0] } else { Vec::new() },
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

fn population(cells: &[Cell]) -> Population {
    Population {
        subjects: cells.iter().map(|&c| subject(c)).collect(),
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Worst `|states/V − ipred| / ipred` over the observations, asserting every term is
/// finite first (`f64::max` would discard a NaN).
fn worst_state_vs_ipred(model: &CompiledModel, cell: Cell) -> f64 {
    let s = subject(cell);
    let theta = &model.default_params.theta;
    let (ipred, states) = crate::pk::compute_predictions_with_states(model, &s, theta, &[0.0]);
    assert_eq!(
        states.len(),
        ipred.len(),
        "one state vector per observation"
    );
    let v = theta[1];
    ipred
        .iter()
        .zip(&states)
        .map(|(&ip, st)| {
            let rel = (st[0] / v - ip).abs() / ip.abs();
            assert!(rel.is_finite(), "non-finite state/ipred: {} vs {ip}", st[0]);
            rel
        })
        .fold(0.0, f64::max)
}

/// The measurement behind the table: which cells give approximate ODE states.
///
/// Realised worst relative gap between `central / V` and `ipred` (macOS arm64; both
/// sides are deterministic integrations at the spec tolerances):
///
/// | cell | worst gap |
/// |---|---|
/// | plain (single-pass engine) | 0 |
/// | reset only | 4.5e-6 |
/// | `[odes]` RHS reading `TIME` | 5.6e-5 |
/// | TV `WT` | 1.08 |
/// | `CL` reading `TIME` (the #1855 cell) | 2.93 |
///
/// The first three are the two engines' integrator tolerances (one snapshot for the
/// whole timeline); the last two are the disposition frozen at the first observation's
/// snapshot. The bounds (1e-3, 0.3) sit ≥ 18× and ≥ 3.6× from the nearest measured
/// value, so the test separates the classes rather than pinning digits.
#[test]
fn which_cells_freeze_the_disposition_is_measured() {
    let plain = ode_model(CL_PLAIN, "1");
    let rhs_time = ode_model(CL_PLAIN, "(1 + 0.05 * TIME)");
    let pk_time = ode_model(CL_TIME, "1");
    let none = Cell {
        tv_cov: false,
        reset: false,
    };
    let reset = Cell {
        tv_cov: false,
        reset: true,
    };
    let tv = Cell {
        tv_cov: true,
        reset: false,
    };

    let exact = [
        ("plain", worst_state_vs_ipred(&plain, none)),
        ("reset only", worst_state_vs_ipred(&plain, reset)),
        ("RHS reads TIME", worst_state_vs_ipred(&rhs_time, none)),
    ];
    let approx = [
        ("CL reads TIME", worst_state_vs_ipred(&pk_time, none)),
        ("TV covariate", worst_state_vs_ipred(&plain, tv)),
    ];
    eprintln!("exact cells: {exact:?}\napproximate cells: {approx:?}");
    for (name, gap) in exact {
        assert!(
            gap < 1e-3,
            "{name}: states should be exact, worst gap {gap:e}"
        );
    }
    for (name, gap) in approx {
        assert!(
            gap > 0.3,
            "{name}: states should be approximate, worst gap {gap:e}"
        );
    }
    // The warning's predicate agrees with the measurement, cell by cell.
    assert!(!crate::pk::ode_cmt_states_approximate(
        &plain,
        &subject(none)
    ));
    assert!(!crate::pk::ode_cmt_states_approximate(
        &plain,
        &subject(reset)
    ));
    assert!(!crate::pk::ode_cmt_states_approximate(
        &rhs_time,
        &subject(none)
    ));
    assert!(crate::pk::ode_cmt_states_approximate(
        &pk_time,
        &subject(none)
    ));
    assert!(crate::pk::ode_cmt_states_approximate(&plain, &subject(tv)));
}

fn warnings_for(model: &CompiledModel, cells: &[Cell]) -> Vec<String> {
    let pop = population(cells);
    let n = pop.subjects.len();
    derived_cmt_time_varying_warnings(
        model,
        &pop,
        &model.default_params.theta,
        &vec![DVector::zeros(model.n_eta); n],
        &vec![Vec::new(); n],
    )
}

const NONE: Cell = Cell {
    tv_cov: false,
    reset: false,
};
const RESET: Cell = Cell {
    tv_cov: false,
    reset: true,
};
const TV: Cell = Cell {
    tv_cov: true,
    reset: false,
};

fn ode_msg(cause: &str) -> String {
    format!(
        "W_DERIVED_CMT_TV_ODE: ODE model with {cause} — compartment states for those \
         subjects are approximate (the disposition parameters at the first observation \
         are held for the whole state pass; ipred is exact). Use compartments[i] results \
         with care for those subjects."
    )
}

fn analytical_msg(cause: &str) -> String {
    format!(
        "W_DERIVED_CMT_TV_ANALYTICAL: analytical model with {cause} — compartment states \
         are not available for those subjects; [derived] expressions that reference \
         compartments[i] evaluate to NaN for them."
    )
}

/// The ODE cells with exact states warn about nothing: a reset alone, and an `[odes]`
/// RHS that reads time, route to the event-driven pass but under one snapshot.
#[test]
fn ode_exact_cells_are_silent() {
    let plain = ode_model(CL_PLAIN, "1");
    let rhs_time = ode_model(CL_PLAIN, "(1 + 0.05 * TIME)");
    assert!(warnings_for(&plain, &[NONE]).is_empty());
    assert!(warnings_for(&plain, &[RESET]).is_empty());
    assert!(warnings_for(&rhs_time, &[NONE, RESET]).is_empty());
}

/// The #1855 cell: a `TIME`-reading `CL` with no covariate change. Silent before the
/// fix. The message names `TIME` and not covariates the data does not have.
#[test]
fn ode_time_reading_individual_params_warn() {
    let m = ode_model(CL_TIME, "1");
    assert_eq!(
        warnings_for(&m, &[NONE]),
        vec![ode_msg("individual parameters that read TIME")]
    );
    // A reset does not change the cause.
    assert_eq!(
        warnings_for(&m, &[RESET]),
        vec![ode_msg("individual parameters that read TIME")]
    );
}

/// TV covariates alone name covariates and not `TIME`; with both present, both.
/// Asserting the three cause variants together kills a cause selector stuck on any one
/// arm.
#[test]
fn ode_cause_names_what_is_present() {
    let plain = ode_model(CL_PLAIN, "1");
    let time = ode_model(CL_TIME, "1");
    assert_eq!(
        warnings_for(&plain, &[TV]),
        vec![ode_msg("time-varying covariates")]
    );
    assert_eq!(
        warnings_for(&plain, &[NONE, TV, RESET]),
        vec![ode_msg("time-varying covariates")]
    );
    assert_eq!(
        warnings_for(&time, &[TV]),
        vec![ode_msg(
            "time-varying covariates or individual parameters that read TIME"
        )]
    );
}

/// The analytical engine's twin of the gap: a `TIME`-reading `CL` empties the states
/// (`compute_predictions_with_states` asks `model_uses_time_anywhere`) and used not to
/// warn either.
#[test]
fn analytical_cells() {
    let plain = analytical_model(CL_PLAIN);
    let time = analytical_model(CL_TIME);
    assert!(warnings_for(&plain, &[NONE]).is_empty());
    assert_eq!(
        warnings_for(&plain, &[TV]),
        vec![analytical_msg("time-varying covariates")]
    );
    assert_eq!(
        warnings_for(&time, &[NONE]),
        vec![analytical_msg("individual parameters that read TIME")]
    );
    assert_eq!(
        warnings_for(&time, &[TV]),
        vec![analytical_msg(
            "time-varying covariates or individual parameters that read TIME"
        )]
    );
    // The analytical NaN claim is the state path's actual behaviour on the TIME cell.
    let s = subject(NONE);
    let (_, states) =
        crate::pk::compute_predictions_with_states(&time, &s, &time.default_params.theta, &[0.0]);
    assert!(
        states.is_empty(),
        "TIME-reading analytical states must be empty"
    );
}

/// An IOV subject is skipped — its states are empty under `W_DERIVED_CMT_IOV_UNSUPPORTED`.
#[test]
fn iov_subjects_are_skipped() {
    let m = ode_model(CL_TIME, "1");
    let pop = population(&[NONE]);
    let out = derived_cmt_time_varying_warnings(
        &m,
        &pop,
        &m.default_params.theta,
        &[DVector::zeros(m.n_eta)],
        &[vec![DVector::zeros(1)]],
    );
    assert!(out.is_empty(), "{out:?}");
}

/// What `fit()` pushes for the #1855 cell is the helper's message, exactly once — the
/// helper tests above cannot see the call site.
#[test]
fn fit_pushes_the_time_warning() {
    let m = ode_model(CL_TIME, "1");
    let pop = population(&[NONE, NONE]);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    }
    .quiet();
    let result = fit(&m, &pop, &m.default_params, &opts).expect("the evaluation must succeed");
    let hits: Vec<_> = result
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_DERIVED_CMT_TV"))
        .collect();
    assert_eq!(
        hits,
        vec![&ode_msg("individual parameters that read TIME")],
        "{:?}",
        result.warnings
    );
}
