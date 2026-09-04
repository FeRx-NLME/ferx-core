//! #1189 item 1 — the **fit-init front door** for a non-finite dose attribute, and the
//! bound on #1196's membership change.
//!
//! A `NaN`/infinite `ALAG` or `F` makes `dose.time + lag` — and every break derived from
//! it — non-finite, so the subject's integration timeline cannot be ordered. Until #1189
//! that *panicked* the objective path and both dense builders inside
//! `break_times.sort_by(|a, b| a.partial_cmp(b).unwrap())`; the engines now return a
//! typed non-finite subject instead, which the estimation guards absorb as a diverged
//! solve. That is right for a mid-fit θ/η excursion, but wasteful and opaque when the
//! model is already broken at its starting point — so `check_model_data` rejects it up
//! front, naming the subject.
//!
//! **How a lag actually goes non-finite matters for whether this can fire at all.** The
//! DSL's arithmetic is domain-guarded in the obvious places: division by ~0 returns 0,
//! and `ln`/`sqrt` floor their argument — measured, not assumed. `exp` is *not* clamped,
//! so an exponential covariate model on an unscaled covariate (`ALAG1 = TVLAG*exp(WT)`
//! with `WT` in the hundreds — a real modelling mistake, not a contrivance) overflows to
//! `+inf`. That is the fixture below.

mod common;

use ferx_core::api::check_model_data;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, Population};

/// Oral 1-cpt ODE whose depot lag is `TVLAG · exp(WT)` — finite for a small `WT`,
/// `+inf` once `WT` is large enough to overflow the exponential.
const EXP_LAG_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.3, 0.0, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  ALAG1 = TVLAG * exp(WT)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn population_with_wt(wt: f64) -> Population {
    let mut subject = common::subject(
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        vec![1.0, 4.0, 8.0],
        vec![0.0; 3],
        vec![1; 3],
    );
    subject.covariates.insert("WT".to_string(), wt);
    Population {
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject],
    }
}

/// A non-finite `ALAG` at typical values is rejected before the fit starts, naming the
/// subject — instead of the panic it used to be, or the all-`NaN` fit the engine guard
/// alone would produce.
///
/// Mutation: delete the `check_dose_attr_finiteness` call from `check_model_data` and no
/// diagnostic is raised.
#[test]
fn a_non_finite_lagtime_at_typical_values_is_rejected_at_fit_init() {
    let model = parse_full_model(EXP_LAG_MODEL)
        .expect("the exp-lag model parses")
        .model;
    let pop = population_with_wt(1000.0);
    // Sanity: the fixture really is non-finite, so this cannot pass on a mis-built model.
    assert!(
        (0.3f64 * 1000.0f64.exp()).is_infinite(),
        "the fixture must overflow, or there is nothing to reject"
    );
    let diags = check_model_data(&model, &pop);
    let hit = diags
        .iter()
        .find(|d| d.code == "E_DOSE_ATTR_NONFINITE")
        .unwrap_or_else(|| {
            panic!(
                "expected E_DOSE_ATTR_NONFINITE, got {:?}",
                diags.iter().map(|d| &d.code).collect::<Vec<_>>()
            )
        });
    assert!(
        hit.message.contains("subject 1"),
        "the diagnostic must name the offending subject: {}",
        hit.message
    );
    assert!(
        hit.message.contains("lag time"),
        "the diagnostic must name which attribute went non-finite: {}",
        hit.message
    );
}

/// The control: the *same model* with an ordinary covariate is accepted. Without this the
/// test above would also pass if the check fired on every model, which would break every
/// lagged fit in the repo.
#[test]
fn a_finite_lagtime_is_not_rejected() {
    let model = parse_full_model(EXP_LAG_MODEL)
        .expect("the exp-lag model parses")
        .model;
    let pop = population_with_wt(0.5);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code == "E_DOSE_ATTR_NONFINITE"),
        "a finite lag must not be rejected, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

/// The bound on #1196's behaviour change. Unifying the two infusion resolvers on one
/// membership predicate changes what a `CMT=0` or out-of-range **infusion** contributes —
/// `active_infusions` used to admit both and `gated_infusions` dropped them. That cannot
/// affect any `fit()` / `predict()` result because validation rejects both compartments
/// (#899); the change is confined to hand-built `OdeSpec`s. If this ever stops holding,
/// the unification needs a user-facing note it does not currently carry.
#[test]
fn validation_still_rejects_the_infusion_compartments_the_resolvers_used_to_differ_on() {
    let model = parse_full_model(EXP_LAG_MODEL)
        .expect("the exp-lag model parses")
        .model;
    for (cmt, why) in [(0usize, "CMT=0"), (7usize, "cmt > n_states")] {
        let mut pop = population_with_wt(0.5);
        pop.subjects[0].doses = vec![DoseEvent::new(0.0, 100.0, cmt, 25.0, false, 0.0)];
        let diags = check_model_data(&model, &pop);
        assert!(
            diags
                .iter()
                .any(|d| d.severity == ferx_core::Severity::Error),
            "an infusion with {why} must still be rejected, got {:?}",
            diags.iter().map(|d| &d.code).collect::<Vec<_>>()
        );
    }
}
