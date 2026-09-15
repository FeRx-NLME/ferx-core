//! `W_AUTO_OPTIMIZER_FOLLOWS_GRADIENT` — the `gradient = fd` ⇒ `optimizer = auto`
//! coupling (#1381).
//!
//! `optimizer = auto` resolves off the availability of the exact analytic outer
//! gradient, so `gradient = fd` moves the optimizer as well as the gradient. The
//! default is right (#490); what the warning adds is that the user is told, at the
//! point the coupling bites, that a one-line change moved two factors.
//!
//! Each test below names the regression it exists to catch, because three of the
//! four gates in the check reject overlapping inputs and a green suite is not
//! evidence any of them is load-bearing.

use super::*;
use crate::diagnostics::Severity;
use crate::types::test_helpers::{analytical_model, ode_model};
use crate::types::{EstimationMethod, GradientMethod, Optimizer};

const CODE: &str = "W_AUTO_OPTIMIZER_FOLLOWS_GRADIENT";

/// `fit()` / `run_*` mirror `options.gradient_method` onto `model.gradient_method`
/// before the fit; the check reads the option for "did the user ask?" and the model
/// for "what did `auto` resolve to?", so a test that sets only one of the two is
/// testing a state no fit reaches. This mirrors both, as the engine does.
fn coupled_case(
    gradient: GradientMethod,
    optimizer: Optimizer,
    method: EstimationMethod,
) -> Vec<Diagnostic> {
    let model = analytical_model(gradient);
    let opts = FitOptions {
        method,
        optimizer,
        gradient_method: gradient,
        interaction: matches!(method, EstimationMethod::FoceI),
        ..Default::default()
    };
    check_model_options(&model, &opts)
        .into_iter()
        .filter(|d| d.code == CODE)
        .collect()
}

/// The reported case: an analytic-scope model, `gradient = fd`, `optimizer` left at
/// `auto`. Both concrete optimizers must be named — the message is useless for a
/// controlled experiment if it does not say what the arm actually ran and what the
/// other arm will run.
#[test]
fn fd_on_an_analytic_model_warns_and_names_both_optimizers() {
    let diags = coupled_case(GradientMethod::Fd, Optimizer::Auto, EstimationMethod::FoceI);
    assert_eq!(diags.len(), 1, "expected exactly one {CODE}, got {diags:?}");

    let d = &diags[0];
    assert_eq!(
        d.severity,
        Severity::Warning,
        "`gradient = fd` is a legitimate request — this must not block the fit"
    );
    assert_eq!(d.block.as_deref(), Some("fit_options"));
    // The optimizer that ran, and the one the unforced arm will run. Asserted by
    // the labels the rest of the output uses (`FitResult.optimizer` prints
    // `auto (bobyqa)`), so the user can match the two up without the docs.
    assert!(
        d.message.contains("bobyqa"),
        "must name what auto resolved to: {}",
        d.message
    );
    assert!(
        d.message.contains("nlopt_lbfgs"),
        "must name what auto would have resolved to under gradient = auto: {}",
        d.message
    );
    assert!(
        d.message.contains("gradient = fd"),
        "must name the setting that caused it: {}",
        d.message
    );
}

/// Catches: the check firing off the optimizer alone. `gradient = auto` on the same
/// model resolves to `nlopt_lbfgs` and nothing moved, so there is nothing to say.
#[test]
fn auto_gradient_is_silent() {
    assert!(
        coupled_case(
            GradientMethod::Auto,
            Optimizer::Auto,
            EstimationMethod::FoceI
        )
        .is_empty(),
        "no gradient override, no coupling"
    );
}

/// Catches: dropping the `optimizer == Auto` gate. A user who pinned the optimizer
/// explicitly moved exactly one factor, which is the whole point of pinning it — and
/// is the remedy the message recommends, so warning here would nag at the fix.
#[test]
fn an_explicit_optimizer_is_silent_under_fd() {
    for opt in [Optimizer::Bobyqa, Optimizer::NloptLbfgs] {
        assert!(
            coupled_case(GradientMethod::Fd, opt, EstimationMethod::FoceI).is_empty(),
            "optimizer = {} was pinned by the user; nothing was coupled to the gradient",
            opt.label()
        );
    }
}

/// Catches: dropping the counterfactual and warning on `gradient = fd` alone. On a
/// model already outside the analytic scope, `auto` resolves to `bobyqa` either way —
/// `fd` changed nothing, and a warning claiming otherwise would be false.
///
/// The straddle is asserted rather than assumed: if `ode_model` ever comes into
/// analytic outer scope this test would silently become a second copy of the
/// analytic case and stop covering the counterfactual.
#[test]
fn an_out_of_scope_model_is_silent_because_fd_changed_nothing() {
    let model = ode_model(GradientMethod::Auto);
    assert!(
        !crate::sens::provider::analytic_outer_gradient_in_scope(&model),
        "fixture must be OUT of analytic outer scope for this test to test anything"
    );
    let in_scope = analytical_model(GradientMethod::Auto);
    assert!(
        crate::sens::provider::analytic_outer_gradient_in_scope(&in_scope),
        "the sibling fixture must be IN scope, or the pair does not straddle the gate"
    );

    let model = ode_model(GradientMethod::Fd);
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        optimizer: Optimizer::Auto,
        gradient_method: GradientMethod::Fd,
        interaction: true,
        ..Default::default()
    };
    assert!(
        !check_model_options(&model, &opts)
            .iter()
            .any(|d| d.code == CODE),
        "auto resolves to bobyqa on this model with or without the override"
    );
}

/// Catches: dropping the `options.gradient_method == Fd` gate, which is *not*
/// redundant with the counterfactual. An SDE model has `model.gradient_method`
/// overwritten to `Fd` by `fit()` whatever the user asked for; the user moved no
/// lines, so there is no user-side coupling to report. Reproduced here in the
/// general form — the model forced to FD while the options are not — since that is
/// the only state in which the two gates disagree.
#[test]
fn an_engine_forced_fd_is_silent_because_the_user_set_nothing() {
    let model = analytical_model(GradientMethod::Fd);
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        optimizer: Optimizer::Auto,
        // The user's request, untouched: `auto`.
        gradient_method: GradientMethod::Auto,
        interaction: true,
        ..Default::default()
    };
    // The counterfactual alone *would* fire here — that is what makes the options
    // gate load-bearing rather than belt-and-braces.
    assert_ne!(
        Optimizer::Auto.resolve_auto(&model, true),
        Optimizer::Auto.resolve_auto_given_analytic(
            &model,
            crate::sens::provider::analytic_outer_gradient_in_scope(&model)
        ),
        "fixture must be one the counterfactual gate alone would flag"
    );
    assert!(
        !check_model_options(&model, &opts)
            .iter()
            .any(|d| d.code == CODE),
        "the user did not write `gradient = fd`; do not report a coupling they did not cause"
    );
}

/// Catches: dropping the method gate. `saem`, `imp`, `impmap`, `bayes` and `vi`
/// never run the outer optimizer, and `laplace`'s `auto` is overridden to
/// `nlopt_lbfgs` by `fit()` before the outer loop sees it (#317) — so on none of
/// them does `gradient = fd` move the optimizer.
#[test]
fn methods_that_never_consult_resolve_auto_are_silent() {
    for method in [
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::Bayes,
        EstimationMethod::Vi,
        EstimationMethod::Laplace,
    ] {
        assert!(
            coupled_case(GradientMethod::Fd, Optimizer::Auto, method).is_empty(),
            "{method:?} does not resolve `auto` through the outer optimizer"
        );
    }
    // …and the three that do.
    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGnHybrid,
    ] {
        assert_eq!(
            coupled_case(GradientMethod::Fd, Optimizer::Auto, method).len(),
            1,
            "{method:?} reaches the outer optimizer with `auto` still live"
        );
    }
}

/// Catches: the counterfactual being re-spelled instead of taken off
/// `resolve_auto_given_analytic`. Above `BOBYQA_MAX_DIM` the resolver takes L-BFGS
/// on an FD gradient, so `fd` moves nothing and the warning must stay quiet — an arm
/// a hand-written `if in_scope { lbfgs } else { bobyqa }` counterfactual would get
/// wrong, because it would report a switch to `bobyqa` that never happened.
#[test]
fn above_the_bobyqa_dimension_threshold_nothing_is_coupled() {
    let mut model = analytical_model(GradientMethod::Fd);
    let n = crate::types::BOBYQA_MAX_DIM + 1;
    model.n_theta = n;
    model.theta_names = (0..n).map(|i| format!("T{i}")).collect();
    model.default_params.theta = vec![1.0; n];
    model.default_params.theta_names = model.theta_names.clone();
    model.default_params.theta_lower = vec![0.0; n];
    model.default_params.theta_upper = vec![f64::INFINITY; n];
    model.default_params.theta_fixed = vec![false; n];
    assert!(
        model.free_packed_dim() > crate::types::BOBYQA_MAX_DIM,
        "fixture must clear the threshold, else this tests the ordinary arm"
    );
    assert_eq!(
        Optimizer::Auto.resolve_auto(&model, true),
        Optimizer::NloptLbfgs,
        "above the threshold `auto` takes L-BFGS even on an FD gradient"
    );

    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        optimizer: Optimizer::Auto,
        gradient_method: GradientMethod::Fd,
        interaction: true,
        ..Default::default()
    };
    assert!(
        !check_model_options(&model, &opts)
            .iter()
            .any(|d| d.code == CODE),
        "the optimizer did not move, so there is no coupling to report"
    );
}

/// The warning reaches `FitResult.warnings`, not just `ferx check`: `fit()` drains
/// the warning-severity half of `check_model_options` into `accumulated_warnings`,
/// and a diagnostic that only `ferx check` prints is exactly the "a field the user
/// has to think to read" failure the issue is about.
#[test]
fn the_warning_is_warning_severity_so_fit_forwards_it() {
    let diags = coupled_case(GradientMethod::Fd, Optimizer::Auto, EstimationMethod::FoceI);
    assert_eq!(diags.len(), 1);
    assert!(
        !diags[0].is_error(),
        "an error would refuse a legitimate fit; `fit()` also only forwards non-errors"
    );
}

/// `resolve_auto` must keep resolving what it always resolved — the split into
/// `resolve_auto_given_analytic` is a refactor, and a refactor that changed the
/// pick would change every default fit in the repo.
#[test]
fn resolve_auto_is_unchanged_by_the_split() {
    for (model, expected) in [
        (
            analytical_model(GradientMethod::Auto),
            Optimizer::NloptLbfgs,
        ),
        (analytical_model(GradientMethod::Fd), Optimizer::Bobyqa),
        (ode_model(GradientMethod::Auto), Optimizer::Bobyqa),
        (ode_model(GradientMethod::Fd), Optimizer::Bobyqa),
    ] {
        assert_eq!(Optimizer::Auto.resolve_auto(&model, true), expected);
        // Idempotent for every non-`Auto` variant, as before.
        assert_eq!(
            Optimizer::Slsqp.resolve_auto(&model, true),
            Optimizer::Slsqp
        );
    }
}
