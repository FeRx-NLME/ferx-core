//! What `FitResult::optimizer` reports for `optimizer = auto` (#490, #1540).
//!
//! `auto (<resolved>)` is a claim about which outer optimizer ran. Before #1540 the
//! label was built from `Optimizer::resolve_auto`, which has no `[mixture]` branch,
//! while the outer loop dispatches on
//! `estimation::outer_optimizer::resolve_outer_optimizer`. The label now comes from
//! `resolve_fit_optimizer`, the rule the outer loop runs, which knows the two things
//! `resolve_auto` does not:
//!
//! - under a mixture, an explicit optimizer that cannot carry the mixture objective runs
//!   as BOBYQA;
//! - a mixture whose classes start indistinguishable runs `auto` as BOBYQA, not L-BFGS.
//!
//! Every test here is a pair straddling one of those gates on an analytic-scope model,
//! and asserts the analytic scope itself: without it `auto` resolves to BOBYQA on both
//! sides and the pair cannot tell the rules apart.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// A 2-class closed-form mixture model (class-specific CL). No covariate in the
/// mixing logit, so the fixture subjects need no covariate columns.
const MIXTURE: &str = r"
[parameters]
  theta TVCL1(1.0, 0.001, 100.0)
  theta TVCL2(3.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// The same structural model with no `[mixture]` block — the other side of the gate.
const NO_MIXTURE: &str = r"
[parameters]
  theta TVCL1(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL1 * exp(ETA_CL)
  V = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

fn parse(src: &str) -> CompiledModel {
    parse_model_string(src).expect("fixture parses")
}

/// The premise both tests rest on: the model is in analytic outer-gradient scope,
/// so `Optimizer::resolve_auto` alone picks the gradient-based L-BFGS. If this ever
/// stops holding, the mixture assertions below turn tautological.
fn assert_analytic_scope(model: &CompiledModel, interaction: bool) {
    assert!(
        crate::sens::provider::analytic_outer_gradient_for_interaction(model, interaction),
        "fixture must be in analytic outer-gradient scope (interaction = {interaction})"
    );
    assert_eq!(
        Optimizer::Auto.resolve_auto(model, interaction),
        Optimizer::NloptLbfgs,
        "the mixture-blind resolver must pick L-BFGS here, or the pair does not straddle"
    );
}

/// The helper, both sides of the symmetric-start gate, for both FOCE and FOCEI: a mixture in
/// analytic scope left on `auto` runs L-BFGS, and from a start whose classes are
/// indistinguishable it runs BOBYQA. The flag means nothing without a mixture.
///
/// Mutations that must redden this: building the label from
/// `options.optimizer.resolve_auto(model, options.interaction)` again (the pre-#1540
/// code), or from `resolve_outer_optimizer` without the symmetric-start fallback — the
/// symmetric arm then reads `auto (nlopt_lbfgs)` for a fit that ran BOBYQA.
#[test]
fn auto_label_follows_the_mixture_rule() {
    let model = parse(MIXTURE);
    assert!(model.mixture.is_some());
    for (method, interaction) in [
        (EstimationMethod::Foce, false),
        (EstimationMethod::FoceI, true),
    ] {
        assert_analytic_scope(&model, interaction);
        let opts = FitOptions {
            method,
            interaction,
            optimizer: Optimizer::Auto,
            ..Default::default()
        };
        assert_eq!(
            reported_optimizer_label(method, &opts, &model, true, || false),
            "auto (nlopt_lbfgs)",
            "{method:?}: a mixture in analytic scope left on `auto` runs L-BFGS"
        );
        assert_eq!(
            reported_optimizer_label(method, &opts, &model, true, || true),
            "auto (bobyqa)",
            "{method:?}: from a symmetric start it runs BOBYQA"
        );
        assert_eq!(
            reported_optimizer_label(method, &opts, &model, false, || true),
            "auto (nlopt_lbfgs)",
            "{method:?}: without a mixture the symmetric-start flag changes nothing"
        );
    }
}

/// An explicit optimizer is reported as the one that ran. Under a mixture an NLopt
/// gradient optimizer is honoured, while one that cannot carry the mixture objective
/// is replaced by BOBYQA and must be reported as `bobyqa`. Each replaced choice is
/// paired with the same choice on the non-mixture side of the gate, where it runs as
/// requested.
///
/// Mutation that must redden this: reporting `options.optimizer.label()` for an
/// explicit choice (the pre-#1540 code), which reads `bfgs` for a fit that ran BOBYQA.
#[test]
fn explicit_optimizer_label_reports_the_mixture_replacement() {
    let model = parse(MIXTURE);
    // The symmetric-start flag set: it overrides only `auto`, never an explicit choice.
    let label = |optimizer, has_mixture| {
        let opts = FitOptions {
            optimizer,
            ..Default::default()
        };
        reported_optimizer_label(EstimationMethod::FoceI, &opts, &model, has_mixture, || true)
    };
    for (optimizer, own) in [
        (Optimizer::Bfgs, "bfgs"),
        (Optimizer::Lbfgs, "lbfgs"),
        (Optimizer::TrustRegion, "trust_region"),
    ] {
        assert_eq!(
            label(optimizer, true),
            "bobyqa",
            "{optimizer:?} under a mixture"
        );
        assert_eq!(
            label(optimizer, false),
            own,
            "{optimizer:?} without a mixture"
        );
    }
    for (optimizer, own) in [
        (Optimizer::NloptLbfgs, "nlopt_lbfgs"),
        (Optimizer::Slsqp, "slsqp"),
        (Optimizer::Mma, "mma"),
        (Optimizer::Bobyqa, "bobyqa"),
    ] {
        assert_eq!(
            label(optimizer, true),
            own,
            "{optimizer:?} is honoured under a mixture"
        );
    }
}

fn subject(id: &str, cl: f64) -> Subject {
    let obs_times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
    let observations: Vec<f64> = obs_times
        .iter()
        .map(|t| 100.0 / 10.0 * (-cl / 10.0 * t).exp())
        .collect();
    let n = obs_times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cmt_toggles: Vec::new(),
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

/// What `fit()` actually writes onto `FitResult::optimizer`, both sides of the
/// symmetric-start gate, and the warning that says why.
///
/// The helper test above cannot see the call site: passing `false` for `symmetric_start`
/// (or checking another start than the stage's) inside `fit_inner` leaves it green. The
/// symmetric fixture is `MIXTURE` with `TVCL2` set to `TVCL1`'s 1.0 and the mixing logit at
/// 0: the classes are then the same model with probability ½ each. One outer iteration — the
/// label does not depend on convergence.
#[test]
fn fit_reports_the_optimizer_a_mixture_fit_ran() {
    let pop = Population {
        subjects: vec![
            subject("1", 1.0),
            subject("2", 1.2),
            subject("3", 3.0),
            subject("4", 2.8),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        optimizer: Optimizer::Auto,
        outer_maxiter: 1,
        run_covariance_step: false,
        ..Default::default()
    };
    let symmetric = MIXTURE.replace(
        "theta TVCL2(3.0, 0.001, 100.0)",
        "theta TVCL2(1.0, 0.001, 100.0)",
    );
    assert_ne!(symmetric, MIXTURE, "the symmetric edit must apply");
    for (label, src, want, warns) in [
        ("distinct classes", MIXTURE, "auto (nlopt_lbfgs)", false),
        ("symmetric start", symmetric.as_str(), "auto (bobyqa)", true),
        ("no mixture", NO_MIXTURE, "auto (nlopt_lbfgs)", false),
    ] {
        let model = parse(src);
        assert_analytic_scope(&model, true);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
        assert_eq!(result.optimizer, want, "{label}");
        let warned = result
            .warnings
            .iter()
            .any(|w| w.starts_with("W_MIXTURE_SYMMETRIC_START"));
        assert_eq!(warned, warns, "{label}: {:#?}", result.warnings);
    }
}
