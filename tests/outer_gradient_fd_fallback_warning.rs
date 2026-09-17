//! #1154 — a runtime analytic-sensitivity decline must reach `FitResult::warnings`.
//!
//! A subject whose data shape falls out of the outer sensitivity provider's scope is
//! salvaged onto a per-subject reconverged-FD gradient. That is correct, several times
//! slower, and — before this — invisible: no warning code, no count, and
//! `FitResult::gradient_method_outer` keeps reporting `analytic (Dual2)` because it reads a
//! **model**-level predicate. The Tier-1 tests in
//! `estimation::outer_optimizer::tests::outer_fd_fallback` pin the counting and the reason
//! the probe stops at the provider; this pins the wiring — that `fit()` actually pushes
//! the warning, and that it does so while the reported outer gradient method still says
//! analytic, which is the mismatch the warning exists to reconcile.
//!
//! Tier 2: `outer_maxiter = 0` returns after one objective evaluation, no convergence loop.

use ferx_core::types::{
    CompiledModel, DoseEvent, EstimationMethod, FitOptions, Optimizer, Population, RateMode,
};
use std::path::Path;

/// Warfarin with a bioavailability `F1 < 1`.
///
/// Under `F ≠ 1` a **rate-defined** (`RATE > 0`) infusion reshapes the dosing window —
/// the rate is held and the window scaled to `F·dur` — which the closed-form walk cannot
/// express, so `subject_sensitivities` declines any subject carrying one (#419). Every
/// bolus-dosed subject is unaffected, and the model as a whole stays inside the analytic
/// outer scope. That combination is exactly the shape this warning exists for: valid data,
/// a model the report calls analytic, and one subject that quietly runs FD.
const WARFARIN_F1: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVF(0.7, 0.05, 1.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  F = TVF

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The FD-fallback sentence `outer_fd_fallback_warning` emits. Matched on the phrase that
/// carries the signal rather than the whole sentence, so a wording change does not redden
/// this while a dropped call does.
fn outer_fd_warning(warnings: &[String]) -> Option<&String> {
    warnings
        .iter()
        .find(|w| w.contains("finite-difference outer gradients"))
}

fn fixture() -> (CompiledModel, Population) {
    let model = ferx_core::parse_model_string(WARFARIN_F1).expect("model parses");
    let population = ferx_core::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    (model, population)
}

fn eval_only_options() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        // The analytic outer gradient is only dispatched for a gradient-driven optimizer;
        // derivative-free BOBYQA has no outer gradient to fall back from.
        optimizer: Optimizer::NloptLbfgs,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    }
}

/// Control: an unmodified population is entirely inside the provider's scope, so no
/// warning. Without this a warning that fired unconditionally would pass the test below.
#[test]
fn in_scope_population_emits_no_outer_fd_warning() {
    let (model, population) = fixture();
    let result = ferx_core::fit(
        &model,
        &population,
        &model.default_params,
        &eval_only_options(),
    )
    .expect("eval-only fit succeeds");
    assert_eq!(
        result.gradient_method_outer, "analytic (Dual2)",
        "fixture precondition: the model must report the analytic outer route"
    );
    assert!(
        outer_fd_warning(&result.warnings).is_none(),
        "in-scope population must not warn; got {:?}",
        result.warnings
    );
}

/// One subject whose dose is a rate-defined infusion falls out of the closed-form
/// provider's scope under `F ≠ 1`. The fit still reports the model-level `analytic (Dual2)`
/// route — correctly, for the model — and the warning is what reconciles that with what
/// the subject actually ran.
#[test]
fn out_of_scope_subject_warns_while_the_report_still_says_analytic() {
    let (model, mut population) = fixture();
    let n_total = population.subjects.len();
    assert!(n_total > 1, "fixture precondition: need a mixed population");

    let declining = &mut population.subjects[0];
    let declining_id = declining.id.clone();
    // Replace the subject's bolus with the same amount delivered as a rate-defined
    // infusion, rather than adding a second dose: the point is a data *shape* the
    // provider declines, not a different amount of drug.
    let mut dose = DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0);
    dose.rate_mode = RateMode::Fixed;
    declining.doses = vec![dose];

    let result = ferx_core::fit(
        &model,
        &population,
        &model.default_params,
        &eval_only_options(),
    )
    .expect("eval-only fit succeeds");

    let w = outer_fd_warning(&result.warnings).unwrap_or_else(|| {
        panic!(
            "an out-of-scope subject must warn; got {:?}",
            result.warnings
        )
    });
    assert!(
        w.contains(&format!("1 of {n_total}")),
        "warning must count exactly the declining subject; got: {w}"
    );
    assert!(
        w.contains(&declining_id),
        "warning must name the declining subject ({declining_id}); got: {w}"
    );
    // The whole point: the report and the run disagree, and only the warning says so.
    assert_eq!(
        result.gradient_method_outer, "analytic (Dual2)",
        "the reported outer method is model-level and must still read analytic — if it \
         ever reports the per-subject route instead, this warning's framing needs updating"
    );
}
