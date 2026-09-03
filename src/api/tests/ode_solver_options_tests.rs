//! Tier-1 tests for #1212: a caller's `FitOptions::ode_*` must reach the integrator.
//!
//! `OdeSpec::solver_opts` is stamped at parse time and every integration entry point reads it
//! off the spec, so before #1212 a `FitOptions { ode_reltol: 1e-10, .. }` handed to [`fit`]
//! changed nothing at all — the fit ran at the parse-time value and reported success. The
//! failure mode was a silent wrong answer, which is why the assertions here are on the OFV
//! rather than on a warning or a returned setting: only the objective can say whether the
//! value actually reached the solver.
//!
//! The three cases are the three directions the merge has to get right — the caller's value
//! wins where they set one, the model file's value survives where they did not, and neither
//! outlives the fit.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// A hand-written ODE 1-cpt IV model, so the objective goes through the integrator rather than
/// a closed form. `tail` carries any `[fit_options]` the case needs baked onto the spec.
fn ode_model(tail: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
{tail}"#
    ))
    .expect("the ODE test model must parse")
}

fn subject(id: &str, scale: f64) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times
        .iter()
        .map(|t| scale * 10.0 * (-0.1f64 * t).exp())
        .collect();
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn population() -> Population {
    Population {
        subjects: vec![subject("1", 1.0), subject("2", 1.2), subject("3", 0.9)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// `outer_maxiter: 0` evaluates the objective at the initial vector and returns — enough to see
/// which tolerance the integrator ran at, without a convergence loop.
fn eval_opts() -> FitOptions {
    FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    }
    .quiet()
}

fn ofv(model: &CompiledModel, opts: &FitOptions) -> f64 {
    let pop = population();
    fit(model, &pop, &model.default_params, opts)
        .expect("the evaluation must succeed")
        .ofv
}

/// The regression this issue was filed for. Six orders of magnitude of tolerance have to move
/// the objective; before #1212 the two calls were bit-identical because neither value reached
/// the solver.
#[test]
fn a_caller_supplied_ode_tolerance_reaches_the_integrator() {
    let model = ode_model("");
    let loose = ofv(&model, &eval_opts());
    let tight = ofv(
        &model,
        &FitOptions {
            ode_reltol: 1e-10,
            ode_abstol: 1e-12,
            ..eval_opts()
        },
    );
    assert_ne!(
        loose.to_bits(),
        tight.to_bits(),
        "#1212: `ode_reltol` passed to `fit()` must reach the integrator, but the objective \
         is bit-identical at 1e-4 and 1e-10 ({loose} vs {tight})"
    );
    // ...and by more than last-bit noise, so this cannot pass on an incidental rounding
    // difference while the tolerance is still being ignored.
    let rel = (loose - tight).abs() / tight.abs().max(1.0);
    assert!(
        rel > 1e-9,
        "the two objectives differ by only {rel:e} relative, which is not an integration \
         accuracy change ({loose} vs {tight})"
    );
}

/// The other direction, and the reason the merge is per-field rather than wholesale: a caller
/// who hand-builds a `FitOptions` and never touches `ode_reltol` must not silently *loosen* a
/// model file that pinned it. Their untouched `1e-4` is indistinguishable from "no opinion", so
/// the file wins — the fit stays at the accuracy the model asked for.
#[test]
fn an_untouched_ode_tolerance_keeps_the_model_files_pinned_value() {
    let pinned = ode_model("[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n");
    let plain = ode_model("");

    // Default `ode_*` in the options, tight values on the spec: the spec's must win.
    let from_file = ofv(&pinned, &eval_opts());
    // Same effective accuracy reached the other way round — default spec, tight options.
    let from_caller = ofv(
        &plain,
        &FitOptions {
            ode_reltol: 1e-10,
            ode_abstol: 1e-12,
            ..eval_opts()
        },
    );
    assert_eq!(
        from_file.to_bits(),
        from_caller.to_bits(),
        "the same effective tolerance must give the same objective whether it came from the \
         model file or the caller ({from_file} vs {from_caller})"
    );

    // And it is genuinely the tight number, not the default one that happens to agree.
    let default_accuracy = ofv(&plain, &eval_opts());
    assert_ne!(
        from_file.to_bits(),
        default_accuracy.to_bits(),
        "a `[fit_options] ode_reltol = 1e-10` model evaluated with default `FitOptions` was \
         loosened back to 1e-4 ({from_file} vs {default_accuracy})"
    );
}

/// The override is armed for the duration of one `fit` and no longer. A leak would be the same
/// class of bug pointed the other way: a `predict` after a tight fit would silently run at the
/// fit's tolerance instead of the spec's.
#[test]
fn a_fits_ode_options_do_not_outlive_it() {
    let model = ode_model("");
    let baked = model
        .ode_spec
        .as_ref()
        .expect("an ODE model has a spec")
        .solver_opts;
    let _ = ofv(
        &model,
        &FitOptions {
            ode_reltol: 1e-10,
            ode_abstol: 1e-12,
            ..eval_opts()
        },
    );
    let after = model
        .ode_spec
        .as_ref()
        .expect("an ODE model has a spec")
        .effective_solver_opts();
    assert_eq!(after.reltol, baked.reltol);
    assert_eq!(after.abstol, baked.abstol);
}
