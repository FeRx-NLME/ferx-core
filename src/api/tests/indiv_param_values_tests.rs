//! `CompiledModel::indiv_param_values` / `indiv_param_value_map` — the name-based
//! individual-parameter value lookup that replaced the `PkParams.values[pk_indices[i]]`
//! slot read in the `[output]` / `[derived]` / EBE paths (#1356).
//!
//! # What these fixtures have to do that the old ones did not
//!
//! The defect is invisible under the idiomatic `CL = TVCL * exp(ETA_CL)`: at η = 0 the
//! intermediate equals the parameter, so reading `CL`'s slot for `TVCL` gives the right
//! number anyway. Every fixture here keeps the two apart (`CL = TVCL * 2 * …`, so
//! `TVCL = 6` while `CL = 12`) and each one asserts the placeholder is actually present —
//! `pk_indices[TVCL] == 0` next to `pk.values[0] == 12.0` — so a fixture that stops
//! reaching the defect fails rather than passing vacuously.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, GradientMethod, PkParams, Population, Subject, SubjectResult};
use nalgebra::DVector;
use std::collections::HashMap;

/// The issue's fixture: an analytical model whose `TVCL` is a top-level
/// `[individual_parameters]` name that the `[structural_model]` line does not bind.
/// `THCL = 2` ⇒ `TVCL = 6`, `CL = 12` at η = 0 — hand-computed closed forms, not
/// tolerances.
const ANALYTIC_INTERMEDIATE: &str = "
[parameters]
  theta THCL(2.0, 0.01, 50.0)
  theta THV(10.0, 0.1, 500.0)
  theta THKA(1.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  TVCL = THCL * 3
  CL   = TVCL * 2 * exp(ETA_CL)
  V    = THV
  KA   = THKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
";

/// Position of `name` in `indiv_param_names`.
fn idx_of(model: &CompiledModel, name: &str) -> usize {
    model
        .indiv_param_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| {
            panic!(
                "`{name}` must be an individual parameter; have {:?}",
                model.indiv_param_names
            )
        })
}

/// Assert the fixture still *reaches* the defect: `name` has the placeholder slot 0 and
/// slot 0 holds a different number, so a slot read would return the wrong value. Without
/// this the value assertions below could pass on a degenerate model that never had a bug.
fn assert_placeholder_is_live(model: &CompiledModel, name: &str, wrong_value: f64) {
    let i = idx_of(model, name);
    assert_eq!(
        model.pk_indices[i],
        crate::types::PK_IDX_CL,
        "fixture must keep `{name}` on the placeholder slot — otherwise it cannot \
         observe #1356"
    );
    let pk = (model.pk_param_fn)(&model.default_params.theta, &[0.0], &HashMap::new(), 0.0);
    assert_eq!(
        pk.values[crate::types::PK_IDX_CL],
        wrong_value,
        "fixture must be non-degenerate: the placeholder slot has to hold a value \
         different from `{name}`'s own"
    );
}

/// A single observation at `t = 1`, no dose — enough for the post-fit column pass.
fn one_obs_subject() -> Subject {
    Subject {
        id: "S1".into(),
        doses: Vec::new(),
        obs_times: vec![1.0],
        obs_raw_times: vec![1.0],
        observations: vec![1.0],
        obs_cmts: vec![1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0],
        occasions: vec![1],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn one_obs_population() -> Population {
    Population {
        subjects: vec![one_obs_subject()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn sr_for(n_eta: usize) -> SubjectResult {
    SubjectResult {
        id: "S1".into(),
        eta: DVector::from_vec(vec![0.0; n_eta]),
        ipred: vec![1.0],
        pred: vec![1.0],
        iwres: vec![0.0],
        cwres: vec![0.0],
        npde: vec![],
        npd: vec![],
        ofv_contribution: 0.0,
        cens: vec![0],
        n_obs: 1,
        pmix: None,
        mixest: None,
        extra_columns: Vec::new(),
        per_obs_tad: Vec::new(),
        compartment_states: Vec::new(),
        #[cfg(feature = "survival")]
        discrete_rows: Vec::new(),
    }
}

/// Run the post-fit `[derived]` / `[output]` pass on a one-observation subject at η = 0
/// and return `column_name -> value at the single observation`.
fn extra_columns_at_zero_eta(model: &CompiledModel) -> HashMap<String, f64> {
    let population = one_obs_population();
    let mut results = vec![sr_for(model.n_eta)];
    compute_extra_output_columns(
        model,
        &population,
        &model.default_params.theta,
        &[],
        &mut results,
        None,
    );
    results[0]
        .extra_columns
        .iter()
        .map(|(n, v)| (n.clone(), v[0]))
        .collect()
}

// ── the lookup itself ────────────────────────────────────────────────────────

/// An unbound intermediate reports its own value, not `CL`'s. On `main` the slot read
/// returned 12.0 for `TVCL` (#1356).
#[test]
fn analytical_unbound_intermediate_reports_its_own_value() {
    let model = parse_model_string(ANALYTIC_INTERMEDIATE).expect("model must parse");
    assert_placeholder_is_live(&model, "TVCL", 12.0);

    let vals =
        model.indiv_param_value_map(&model.default_params.theta, &[0.0], &HashMap::new(), 0.0);
    // Hand-computed: TVCL = THCL·3 = 6, CL = TVCL·2·exp(0) = 12.
    assert_eq!(vals["TVCL"], 6.0, "TVCL must be THCL*3, not CL");
    assert_eq!(vals["CL"], 12.0);
    assert_eq!(vals["V"], 10.0);
    assert_eq!(vals["KA"], 1.5);
}

/// The η the caller passes has to reach the values — a lookup that quietly evaluated at
/// η = 0 would agree with the test above and differ everywhere it matters.
#[test]
fn indiv_param_values_honour_eta() {
    let model = parse_model_string(ANALYTIC_INTERMEDIATE).expect("model must parse");
    let eta = [0.25f64];
    let vals = model.indiv_param_value_map(&model.default_params.theta, &eta, &HashMap::new(), 0.0);
    assert_eq!(vals["TVCL"], 6.0, "TVCL carries no eta");
    assert_eq!(
        vals["CL"],
        12.0 * eta[0].exp(),
        "CL must be evaluated at the eta passed in"
    );
}

/// A *bound* name must come back bit-identical on both routes. The values are produced by
/// the same statements through the same evaluator, so this is an equality, not a
/// tolerance — and it is what keeps the new lookup from becoming a second implementation
/// of the individual-parameter block.
#[test]
fn indiv_param_values_match_pk_slot_reads_for_bound_names() {
    let model = parse_model_string(ANALYTIC_INTERMEDIATE).expect("model must parse");
    let eta = [0.31f64];
    let theta = &model.default_params.theta;
    let pk = (model.pk_param_fn)(theta, &eta, &HashMap::new(), 0.0);
    let vals = model.indiv_param_values(theta, &eta, &HashMap::new(), 0.0);
    for name in ["CL", "V", "KA"] {
        let i = idx_of(&model, name);
        assert_eq!(
            vals[i].to_bits(),
            pk.values[model.pk_indices[i]].to_bits(),
            "`{name}` is bound on the [structural_model] line, so the two routes must \
             agree bit-for-bit"
        );
    }
}

/// `D{n}` is the case the issue text got wrong in the other direction: its value *is*
/// written, to a spare slot above `PK_IDX_LAGTIME`, but `pk_indices` records the
/// placeholder 0 — so the old read still returned `CL`.
#[test]
fn analytical_modeled_duration_reports_its_own_value() {
    const MODEL: &str = "
[parameters]
  theta THCL(2.0, 0.01, 50.0)
  theta THV(10.0, 0.1, 500.0)
  theta THD(3.0, 0.01, 50.0)
  sigma PROP ~ 0.01

[individual_parameters]
  CL = THCL * 6
  V  = THV
  D1 = THD

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(MODEL).expect("model must parse");
    assert_placeholder_is_live(&model, "D1", 12.0);
    let vals = model.indiv_param_value_map(&model.default_params.theta, &[], &HashMap::new(), 0.0);
    assert_eq!(vals["D1"], 3.0, "D1 must report THD, not CL");
    assert_eq!(vals["CL"], 12.0);
}

/// ODE control: every name already had a real slot there, so the values must be exactly
/// what the slot read gave. `KE` and `LAGTIME` are the interesting ones — `ode_param_slots`
/// sends `KE` to slot 0 and `LAGTIME` to slot 8, so a lookup that used the declaration
/// index instead of the slot would break here and nowhere else.
#[test]
fn ode_model_values_unchanged() {
    const MODEL: &str = "
[parameters]
  theta THV(10.0, 0.1, 500.0)
  theta THKA(1.5, 0.01, 10.0)
  theta THKE(0.2, 0.001, 10.0)
  theta THLAG(0.4, 0.001, 10.0)
  sigma PROP ~ 0.01

[individual_parameters]
  V       = THV
  KA      = THKA
  KE      = THKE
  LAGTIME = THLAG

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - KE * central

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let theta = &model.default_params.theta;
    let pk = (model.pk_param_fn)(theta, &[], &HashMap::new(), 0.0);
    let vals = model.indiv_param_values(theta, &[], &HashMap::new(), 0.0);
    for (i, name) in model.indiv_param_names.iter().enumerate() {
        assert_eq!(
            vals[i].to_bits(),
            pk.values[model.pk_indices[i]].to_bits(),
            "ODE `{name}` must be unchanged by #1356"
        );
    }
    let map = model.indiv_param_value_map(theta, &[], &HashMap::new(), 0.0);
    assert_eq!(map["V"], 10.0);
    assert_eq!(map["KA"], 1.5);
    assert_eq!(map["KE"], 0.2);
    assert_eq!(map["LAGTIME"], 0.4);
}

/// The `TIME` built-in must be evaluated at the time the caller asks for, the same
/// per-row rule the sdtab columns follow (#610).
#[test]
fn indiv_param_values_honour_the_time_builtin() {
    const MODEL: &str = "
[parameters]
  theta THCL(2.0, 0.01, 50.0)
  theta THV(10.0, 0.1, 500.0)
  sigma PROP ~ 0.01

[individual_parameters]
  TSCALE = 1 + TIME
  CL     = THCL * TSCALE
  V      = THV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let theta = &model.default_params.theta;
    for t in [0.0, 3.0] {
        let vals = model.indiv_param_value_map(theta, &[], &HashMap::new(), t);
        assert_eq!(vals["TSCALE"], 1.0 + t, "TSCALE at t={t}");
        assert_eq!(vals["CL"], 2.0 * (1.0 + t), "CL at t={t}");
    }
}

/// #486's synthetic parameters are internal: they are positional in
/// `indiv_param_values` (which is parallel to `indiv_param_names`) and absent from the
/// user-facing map, so they never reach an sdtab / EBE column.
#[test]
fn synthetic_readout_params_hidden_from_the_value_map() {
    const MODEL: &str = "
[parameters]
  theta THV(10.0, 0.1, 500.0)
  theta THKE(0.2, 0.001, 10.0)
  theta THOFF(0.5, 0.01, 10.0)
  omega ETA_V ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  V  = THV * exp(ETA_V)
  KE = THKE

[odes]
  d/dt(central) = -KE * central

[structural_model]
  ode(states=[central])

[scaling]
  y = central / V + THOFF

[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(MODEL).expect("model must parse");
    assert!(
        model
            .indiv_param_names
            .iter()
            .any(|n| crate::parser::model_parser::is_synthetic_readout_param(n)),
        "fixture must actually synthesize a readout parameter, or it cannot observe the \
         filter"
    );
    let theta = &model.default_params.theta;
    let vals = model.indiv_param_values(theta, &[0.0], &HashMap::new(), 0.0);
    assert_eq!(
        vals.len(),
        model.indiv_param_names.len(),
        "the positional form stays parallel to indiv_param_names"
    );
    let map = model.indiv_param_value_map(theta, &[0.0], &HashMap::new(), 0.0);
    assert!(
        !map.keys()
            .any(|k| crate::parser::model_parser::is_synthetic_readout_param(k)),
        "synthetic readout params must be hidden from the user-facing map, got {:?}",
        map.keys().collect::<Vec<_>>()
    );
    assert_eq!(map["V"], 10.0);
    assert_eq!(map["KE"], 0.2);
}

/// A hand-built `CompiledModel` carries no compiled program; the documented fallback is
/// the old slot read, so such a fixture keeps working rather than panicking or returning
/// an empty vector.
#[test]
fn hand_built_model_without_a_program_falls_back_to_the_slot_read() {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
    assert!(
        model.indiv_param_partials.indiv_param_program.is_none(),
        "the shared fixture must stay program-free, or this tests the wrong branch"
    );
    model.indiv_param_names = vec!["CL".into(), "V".into()];
    model.pk_indices = vec![crate::types::PK_IDX_CL, crate::types::PK_IDX_V];
    model.pk_param_fn = Box::new(|_, _, _, _t: f64| {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 1.25;
        p.values[crate::types::PK_IDX_V] = 7.5;
        p
    });
    let map = model.indiv_param_value_map(&[], &[], &HashMap::new(), 0.0);
    assert_eq!(map["CL"], 1.25);
    assert_eq!(map["V"], 7.5);
}

// ── the consumers ────────────────────────────────────────────────────────────

/// End-to-end through the post-fit column pass: an `[output]` column naming an unbound
/// intermediate held `CL`'s value on `main` (#1356).
#[test]
fn output_column_for_an_unbound_intermediate_is_its_own_value() {
    let model = parse_model_string(&format!(
        "{ANALYTIC_INTERMEDIATE}\n[output]\n  CL\n  TVCL\n"
    ))
    .expect("model must parse");
    assert_placeholder_is_live(&model, "TVCL", 12.0);
    let cols = extra_columns_at_zero_eta(&model);
    assert_eq!(
        cols["TVCL"], 6.0,
        "[output] TVCL held CL's value before #1356"
    );
    assert_eq!(cols["CL"], 12.0);
}

/// Same read through `[derived]`, which binds individual parameters by name into the
/// expression context.
#[test]
fn derived_expression_reading_an_unbound_intermediate_uses_its_own_value() {
    let model = parse_model_string(&format!(
        "{ANALYTIC_INTERMEDIATE}\n[derived]\n  TENX = TVCL * 10\n"
    ))
    .expect("model must parse");
    assert_placeholder_is_live(&model, "TVCL", 12.0);
    let cols = extra_columns_at_zero_eta(&model);
    assert_eq!(
        cols["TENX"], 60.0,
        "[derived] computed with CL (120) before #1356"
    );
}

/// The bundled example the issue reports as reproducing this through ferx-r: an
/// analytical TTE model whose `LAMBDA` is not bound on the `[structural_model]` line, so
/// the slot read returned `DUMMY_CL` (1.0) for it (#1356). It needs no contrived
/// fixture — it is what a user running this example actually saw.
#[test]
fn bundled_tte_exponential_example_reports_lambda_not_dummy_cl() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/tte_exponential.ferx"
    ))
    .expect("bundled example must be readable");
    let model = parse_model_string(&src).expect("bundled example must parse");
    assert_placeholder_is_live(&model, "LAMBDA", 1.0);
    let map =
        model.indiv_param_value_map(&model.default_params.theta, &[0.0], &HashMap::new(), 0.0);
    // TVLAMBDA = 0.05, η = 0 ⇒ LAMBDA = 0.05. `DUMMY_CL` is FIXed at 1.0.
    assert_eq!(map["LAMBDA"], 0.05, "LAMBDA reported DUMMY_CL before #1356");
    assert_eq!(map["CL"], 1.0);
}
