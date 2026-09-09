//! `check_model_options` checks specific to `method = vi`.

use super::*;
use crate::diagnostics::Severity;
use crate::types::test_helpers::analytical_model;
use crate::types::{EstimationMethod, GradientMethod, ViKl, ViOmegaUpdate};

fn vi_opts(kl: ViKl, omega: ViOmegaUpdate) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Vi,
        vi_kl: kl,
        vi_omega_update: omega,
        ..Default::default()
    }
}

fn omega_warnings(opts: &FitOptions) -> Vec<Diagnostic> {
    let model = analytical_model(GradientMethod::Auto);
    check_model_options(&model, opts)
        .into_iter()
        .filter(|d| d.code == "W_VI_OMEGA_UNANCHORED")
        .collect()
}

/// `vi_kl = mc` + `vi_omega_update = adam` is the one combination that leaves `Ω`
/// with nothing exact behind it, and it warns.
///
/// A *warning*, not an error: it is exactly the published configuration, so asking
/// for it is legitimate — the point is that the user should know what they asked for.
#[test]
fn mc_kl_plus_adam_omega_warns_that_omega_is_unanchored() {
    let diags = omega_warnings(&vi_opts(ViKl::Mc, ViOmegaUpdate::Adam));
    assert_eq!(
        diags.len(),
        1,
        "expected exactly one warning, got {diags:?}"
    );

    let d = &diags[0];
    assert_eq!(
        d.severity,
        Severity::Warning,
        "this must not block a fit — reproducing the published behaviour is legitimate"
    );
    assert_eq!(d.block.as_deref(), Some("fit_options"));
    // Both culprits are named, so the message is actionable without the docs.
    assert!(d.message.contains("vi_kl = mc"), "message: {}", d.message);
    assert!(
        d.message.contains("vi_omega_update = adam"),
        "message: {}",
        d.message
    );
    assert!(
        d.message.contains("closed_form") || d.message.contains("vi_mc_samples"),
        "the warning should say what to do about it: {}",
        d.message
    );
}

/// Neither option alone is a problem, and the default pair is silent.
///
/// This is the assertion that makes the warning worth having: with `adam` under the
/// analytic KL the `Ω` gradient is exact, and with `mc` under the closed form `Ω`
/// never enters the stochastic optimization at all. Only the combination is unanchored,
/// so warning on either alone would be noise.
#[test]
fn neither_option_alone_warns_about_omega() {
    for (kl, omega) in [
        (ViKl::Analytic, ViOmegaUpdate::ClosedForm),
        (ViKl::Analytic, ViOmegaUpdate::Adam),
        (ViKl::Mc, ViOmegaUpdate::ClosedForm),
    ] {
        let diags = omega_warnings(&vi_opts(kl, omega));
        assert!(
            diags.is_empty(),
            "{kl:?} + {omega:?} must not warn, got {diags:?}"
        );
    }
}

/// And the check is scoped to VI: the same option values on another method are inert,
/// since `vi_*` keys are ignored there.
#[test]
fn the_omega_warning_is_scoped_to_vi() {
    let mut opts = vi_opts(ViKl::Mc, ViOmegaUpdate::Adam);
    opts.method = EstimationMethod::FoceI;
    assert!(omega_warnings(&opts).is_empty());

    // But a chain *containing* VI is still checked.
    let mut chained = vi_opts(ViKl::Mc, ViOmegaUpdate::Adam);
    chained.methods = vec![EstimationMethod::Vi, EstimationMethod::FoceI];
    assert_eq!(omega_warnings(&chained).len(), 1);
}

fn nagq_diags(opts: &FitOptions) -> Vec<Diagnostic> {
    let model = analytical_model(GradientMethod::Auto);
    check_model_options(&model, opts)
        .into_iter()
        .filter(|d| d.code == "E_VI_NAGQ_UNSUPPORTED")
        .collect()
}

/// `n_agq` is a chain-wide option, so a chain that pairs VI with a quadrature stage
/// legitimately carries `n_agq > 1` — the grid belongs to that stage, and VI ignores it.
///
/// This is the documented way to finish a VI fit: `methods = [vi, laplace]` with
/// `agq_eval_only = true` turns the ELBO, which is only a lower bound, into a real
/// `−2 log L`. Rejecting `n_agq` for the whole chain made that path impossible to ask
/// for. The check now fires only when *no* stage consumes the option.
#[test]
fn n_agq_is_accepted_when_a_quadrature_stage_consumes_it() {
    // VI alone: nothing consumes `n_agq`, so setting it is a mistake worth reporting.
    let mut solo = vi_opts(ViKl::Analytic, ViOmegaUpdate::ClosedForm);
    solo.n_agq = 21;
    let diags = nagq_diags(&solo);
    assert_eq!(
        diags.len(),
        1,
        "expected the VI-only rejection, got {diags:?}"
    );
    assert_eq!(diags[0].severity, Severity::Error);

    // The documented readout chain: the Laplace stage consumes the grid.
    let mut readout = vi_opts(ViKl::Analytic, ViOmegaUpdate::ClosedForm);
    readout.methods = vec![EstimationMethod::Vi, EstimationMethod::Laplace];
    readout.n_agq = 21;
    readout.agq_eval_only = true;
    assert!(
        nagq_diags(&readout).is_empty(),
        "methods = [vi, laplace] with agq_eval_only must accept n_agq"
    );

    // An *estimating* Laplace stage consumes it just the same — `agq_eval_only` decides
    // whether the stage moves the parameters, not whether it reads `n_agq`.
    let mut estimating = readout.clone();
    estimating.agq_eval_only = false;
    assert!(nagq_diags(&estimating).is_empty());

    // FOCEI consumes `n_agq > 1` as the Gauss-Newton-anchored quadrature.
    let mut focei = vi_opts(ViKl::Analytic, ViOmegaUpdate::ClosedForm);
    focei.methods = vec![EstimationMethod::Vi, EstimationMethod::FoceI];
    focei.n_agq = 9;
    assert!(nagq_diags(&focei).is_empty());

    // `n_agq = 1` is the default and never rejected, chain or no chain.
    let mut one = vi_opts(ViKl::Analytic, ViOmegaUpdate::ClosedForm);
    one.n_agq = 1;
    assert!(nagq_diags(&one).is_empty());
}
