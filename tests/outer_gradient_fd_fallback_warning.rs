//! #1154 — a runtime analytic-sensitivity decline must reach `FitResult::warnings`.
//!
//! A subject whose data shape falls out of the outer sensitivity provider's scope is
//! salvaged onto a per-subject gradient (held-EBE since #1529; reconverged FD under IOV).
//! Before #1154 that was invisible: no warning code, no count, and
//! `FitResult::gradient_method_outer` keeps reporting `analytic (Dual2)` because it reads a
//! **model**-level predicate. The Tier-1 tests in
//! `estimation::outer_optimizer::tests::outer_fd_fallback` pin the recording; this pins the
//! wiring — that the warning reaches `FitResult::warnings`, that it does so while the
//! reported outer gradient method still says analytic, and that a fit which never
//! evaluates an analytic outer gradient stays silent.
//!
//! Tier 2: `outer_maxiter = 1` runs one gradient evaluation and stops. It must not be `0`
//! — that is the evaluation-only path, which computes no outer gradient at all, so there
//! would be nothing to report and the test would pass for the wrong reason.

use ferx_core::types::{
    CompiledModel, DoseEvent, EstimationMethod, FitOptions, Optimizer, Population, RateMode,
};
use std::path::Path;

/// Warfarin with a bioavailability `F < 1`.
///
/// Under `F ≠ 1` a **rate-defined** (`RATE > 0`) infusion reshapes the dosing window —
/// the rate is held and the window scaled to `F·dur` — which the closed-form walk cannot
/// express, so `subject_sensitivities` declines any subject carrying one (#419). Every
/// bolus-dosed subject is unaffected, and the model as a whole stays inside the analytic
/// outer scope. That combination is exactly the shape this warning exists for: valid data,
/// a model the report calls analytic, and one subject that quietly runs FD.
const WARFARIN_F: &str = r#"
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
  F  = TVF

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// [`WARFARIN_F`] with a two-class `[mixture]`, so `optimizer = auto` resolves to BOBYQA
/// via `resolve_outer_optimizer` while `gradient_method_outer` still reports analytic.
const MIXTURE_F: &str = r#"
[parameters]
  theta TVCL1(0.15, 0.001, 10.0)
  theta TVCL2(0.30, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  theta MIXL(0.0, -10.0, 10.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  F  = TVF

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The fallback sentence `outer_fd_fallback_warning` emits. Matched on the phrase that
/// carries the signal rather than the whole sentence, so a wording change does not redden
/// this while a dropped call does. (Not on the salvage's name: since #1529 that differs
/// between the non-IOV held-EBE and the IOV reconverged routes.)
fn outer_fd_warning(warnings: &[String]) -> Option<&String> {
    warnings
        .iter()
        .find(|w| w.contains("fell outside the analytic sensitivity provider's scope"))
}

/// The same population with subject 0's bolus replaced by a rate-defined infusion — the
/// dose shape the provider declines under `F ≠ 1`. Replaced rather than added: the point
/// is a data *shape* the provider declines, not a different amount of drug.
fn make_subject_0_decline(population: &mut Population) -> String {
    let declining = &mut population.subjects[0];
    let mut dose = DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0);
    dose.rate_mode = RateMode::Fixed;
    declining.doses = vec![dose];
    declining.id.clone()
}

fn fixture() -> (CompiledModel, Population) {
    let model = ferx_core::parse_model_string(WARFARIN_F).expect("model parses");
    let population = ferx_core::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    (model, population)
}

fn one_gradient_eval_options() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        // The analytic outer gradient is only dispatched for a gradient-driven optimizer;
        // derivative-free BOBYQA has no outer gradient to fall back from.
        optimizer: Optimizer::NloptLbfgs,
        // One real gradient evaluation. NOT `0` — see the module docs.
        outer_maxiter: 1,
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
        &one_gradient_eval_options(),
    )
    .expect("fit succeeds");
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
    let declining_id = make_subject_0_decline(&mut population);

    let result = ferx_core::fit(
        &model,
        &population,
        &model.default_params,
        &one_gradient_eval_options(),
    )
    .expect("fit succeeds");

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

/// PR #1418 review, finding 2. A mixture model with `optimizer = auto` runs **BOBYQA**:
/// `resolve_outer_optimizer` downgrades `Auto` silently for `[mixture]`, a rule
/// `build_info::gradient_method_outer` does not model — it still classifies the model as
/// analytic. A derivative-free fit requests no outer gradient at all, so no subject can
/// have taken an FD *outer* gradient and the warning must stay silent even with a subject
/// the provider would decline.
///
/// This passes because the warning reads a runtime log rather than a model-level
/// predicate; a gate written against `gradient_method_outer` would have emitted
/// "1 of 2 ... use finite-difference outer gradients" here.
#[test]
fn a_derivative_free_mixture_auto_fit_does_not_warn() {
    let model = ferx_core::parse_model_string(MIXTURE_F).expect("mixture model parses");
    let mut population = ferx_core::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    make_subject_0_decline(&mut population);

    let options = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        // The knob under test: `auto` on a mixture model resolves to BOBYQA.
        optimizer: Optimizer::Auto,
        outer_maxiter: 1,
        run_covariance_step: false,
        ..Default::default()
    };
    let result = ferx_core::fit(&model, &population, &model.default_params, &options)
        .expect("mixture fit succeeds");

    assert_eq!(
        result.gradient_method_outer, "analytic (Dual2)",
        "fixture precondition: the model-level label must claim analytic, or this test \
         does not reproduce the reported mismatch"
    );
    assert!(
        outer_fd_warning(&result.warnings).is_none(),
        "a derivative-free fit computes no outer gradient, so no subject can have fallen \
         back to an FD one; got {:?}",
        result.warnings
    );
}

/// An evaluation-only run (`outer_maxiter = 0`) short-circuits before any optimizer is
/// built and computes no outer gradient, so it must not report outer FD fallbacks either.
/// Same population and model as the warning test above, so the only difference is the
/// number of iterations.
#[test]
fn an_evaluation_only_fit_does_not_warn() {
    let (model, mut population) = fixture();
    make_subject_0_decline(&mut population);

    let options = FitOptions {
        outer_maxiter: 0,
        ..one_gradient_eval_options()
    };
    let result =
        ferx_core::fit(&model, &population, &model.default_params, &options).expect("fit succeeds");
    assert!(
        outer_fd_warning(&result.warnings).is_none(),
        "an evaluation-only fit runs no outer gradient; got {:?}",
        result.warnings
    );
}

/// PR #1418 review, finding 1. `reconverge_gradient_interval = 1` is the documented
/// escape hatch that forces the reconverged-FD outer gradient on **every** evaluation, so
/// the analytic branch is never selected and no subject "fell outside the provider's
/// scope". Attributing that fit's FD gradients to a scope gap would be a false cause even
/// though the fit really is on FD throughout.
#[test]
fn forcing_the_reconverged_gradient_does_not_warn_about_scope() {
    let (model, mut population) = fixture();
    make_subject_0_decline(&mut population);

    let options = FitOptions {
        reconverge_gradient_interval: 1,
        ..one_gradient_eval_options()
    };
    let result =
        ferx_core::fit(&model, &population, &model.default_params, &options).expect("fit succeeds");
    assert!(
        outer_fd_warning(&result.warnings).is_none(),
        "`reconverge_gradient_interval = 1` bypasses the analytic branch for every \
         subject, which is not a provider scope gap; got {:?}",
        result.warnings
    );
}
