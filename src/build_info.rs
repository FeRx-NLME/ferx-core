use crate::types::{CompiledModel, EstimationMethod, Optimizer};

/// Compile-time build metadata embedded by `build.rs`.
pub struct BuildInfo {
    pub variant: &'static str,
    pub profile: &'static str,
    pub ferx_version: &'static str,
    pub rustc_version: &'static str,
    pub build_timestamp: u64,
}

pub const BUILD_INFO: BuildInfo = BuildInfo {
    variant: env!("FERX_BUILD_VARIANT"),
    profile: env!("FERX_BUILD_PROFILE"),
    ferx_version: env!("CARGO_PKG_VERSION"),
    rustc_version: env!("FERX_RUSTC_VERSION"),
    build_timestamp: {
        let s = env!("FERX_BUILD_TIMESTAMP");
        let mut n: u64 = 0;
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            n = n * 10 + (b[i] - b'0') as u64;
            i += 1;
        }
        n
    },
};

/// Reported gradient method for a fit loop (inner or outer).
///
/// Distinct from [`crate::types::GradientMethod`] which controls _selection_;
/// this enum describes what actually runs and is used only for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientMethodKind {
    /// Exact analytic `Dual2` η-gradient from the sensitivity provider — one
    /// provider evaluation per inner step, independent of n_eta.
    Analytic,
    /// Central finite differences — cost scales as 2×n_eta per gradient call.
    FiniteDifferences,
    /// Not applicable: derivative-free optimizer or sampling-based step.
    NotApplicable,
}

impl GradientMethodKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Analytic => "analytic (Dual2)",
            Self::FiniteDifferences => "finite differences",
            Self::NotApplicable => "N/A",
        }
    }
}

/// Gradient method used in the inner (per-subject EBE) loop.
///
/// Coarse model-level mirror of the per-subject route in
/// [`estimation::inner_optimizer::resolve_gradient_method`]: the analytic `Dual2`
/// gradient runs when the model has an analytical PK path (`tv_fn` populated —
/// ODE models have none) and is in the provider's scope (not SDE; plain LTBS is
/// served, but LTBS + η-dependent `ExpressionScale` is not), and the user did not
/// force `gradient_method = Fd`. Otherwise FD. This is best-case, model-level:
/// per-subject fallbacks — including **TV-cov + LTBS**, where the event-driven
/// inner walk declines LTBS so every such subject actually runs FD — are reported
/// by `gradient_route_summary` (banner) and `fd_fallback_warning`.
pub fn gradient_method_inner(_build: &BuildInfo, model: &CompiledModel) -> GradientMethodKind {
    // Report off the *same* model-level predicates `find_ebe` / `find_ebe_iov` consult, so
    // the two can't diverge as scope grows (PR #381 review #9). Per-subject FD fallbacks
    // (time-varying covariates, survival obs) are reported separately by
    // `gradient_route_summary` / `fd_fallback_warning`. The IOV inner loop runs the exact
    // analytic stacked-η gradient when the model is in IOV scope and clears the shared
    // model-level bails (`find_ebe_iov`'s `analytic_iov_inner` gate) — `analytic_inner_grad
    // _supported_model` returns `false` for every IOV model (it requires `n_kappa == 0`),
    // so the IOV branch must be reported explicitly (#466 review round 4 #1).
    let analytic = crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(model)
        || (crate::sens::provider::iov_sens_supported(model)
            && !crate::estimation::inner_optimizer::analytic_inner_common_bail(model));
    if analytic {
        GradientMethodKind::Analytic
    } else {
        GradientMethodKind::FiniteDifferences
    }
}

/// Gradient method used in the outer (population parameter) loop.
///
/// For a gradient-driven FOCE/FOCEI fit the outer optimiser uses the **exact
/// analytic** packed gradient when the model is in the sensitivity provider's
/// scope (`sens_supported` / `iov_sens_supported`) and the user did not
/// force FD — mirroring the live dispatch in `outer_optimizer` (PR #381 review
/// #4), so the report tracks the headline feature instead of always reading
/// "finite differences". Otherwise:
/// - NLopt-based: NLopt uses its own internal FD.
/// - BOBYQA: derivative-free — no outer gradient at all.
/// - GN/GnHybrid: BHHH outer approximation — always FD.
/// - SAEM/IMP/Bayes: no outer gradient step.
///
/// (A per-fit `reconverge_gradient_interval` override can still force FD; this
/// reports the default in-scope route.)
pub fn gradient_method_outer(
    _build: &BuildInfo,
    method: EstimationMethod,
    optimizer: Optimizer,
    model: &CompiledModel,
) -> GradientMethodKind {
    match method {
        EstimationMethod::Saem
        | EstimationMethod::Imp
        | EstimationMethod::Impmap
        | EstimationMethod::Bayes => GradientMethodKind::NotApplicable,
        EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => {
            GradientMethodKind::FiniteDifferences
        }
        EstimationMethod::Foce | EstimationMethod::FoceI => {
            // `interaction` derives from `method` (the parser sets
            // `opts.interaction = method == FoceI`, so the two never disagree),
            // not a separate `FitOptions` field this function doesn't receive.
            let interaction = method == EstimationMethod::FoceI;
            match optimizer.resolve_auto(model, interaction) {
                Optimizer::Bobyqa => GradientMethodKind::NotApplicable,
                // `Auto` is resolved above; only its concrete results reach here.
                Optimizer::Auto
                | Optimizer::Bfgs
                | Optimizer::Lbfgs
                | Optimizer::Slsqp
                | Optimizer::NloptLbfgs
                | Optimizer::Mma
                | Optimizer::TrustRegion => {
                    // Shared predicate (#490) — now IOV-aware via `iov_sens_supported`, which
                    // admits ODE IOV models too, so the reported method tracks the live outer
                    // dispatch (`outer_optimizer.rs`) for IOV as well (#466 review #4 / #439 IOV).
                    // Shared with `resolve_auto` so the reported method tracks the live outer
                    // dispatch; a custom-magnitude model is analytic on both FOCE and FOCEI now
                    // (#486 σ-magnitude FOCE port), so this no longer narrows by interaction.
                    if crate::sens::provider::analytic_outer_gradient_for_interaction(
                        model,
                        interaction,
                    ) {
                        GradientMethodKind::Analytic
                    } else {
                        GradientMethodKind::FiniteDifferences
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers;
    use crate::types::GradientMethod;

    fn ad_build() -> BuildInfo {
        BuildInfo {
            variant: "default",
            profile: "debug",
            ferx_version: "0.1.0",
            rustc_version: "rustc 1.0.0",
            build_timestamp: 0,
        }
    }

    fn ci_build() -> BuildInfo {
        BuildInfo {
            variant: "ci",
            profile: "release",
            ferx_version: "0.1.0",
            rustc_version: "rustc 1.0.0",
            build_timestamp: 0,
        }
    }

    #[test]
    fn inner_analytical_auto_returns_analytic() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_inner(&ci_build(), &m),
            GradientMethodKind::Analytic
        );
    }

    #[test]
    fn inner_plain_ltbs_returns_analytic() {
        // Closed-form LTBS takes the analytic inner gradient (PR #665; the Tier-1 follow-up
        // extends it to × `ExpressionScale` and × TV-cov). The covariance step reconverges
        // its EBEs at the tighter `cov_inner_tol`. LTBS × IOV still routes to FD (via
        // `iov_analytical_supported`'s own gate).
        let mut m = test_helpers::analytical_model(GradientMethod::Auto);
        m.log_transform = true;
        assert_eq!(
            gradient_method_inner(&ci_build(), &m),
            GradientMethodKind::Analytic
        );
    }

    #[test]
    fn inner_ode_model_returns_fd() {
        let m = test_helpers::ode_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_inner(&ad_build(), &m),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn inner_user_forces_fd_returns_fd() {
        let m = test_helpers::analytical_model(GradientMethod::Fd);
        assert_eq!(
            gradient_method_inner(&ad_build(), &m),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn outer_gradient_optimizers_report_analytic_in_scope() {
        // An in-scope analytical FOCE/FOCEI fit drives every gradient optimizer
        // (built-in BFGS/LBFGS *and* the NLopt SLSQP/LBFGS/MMA, which receive the
        // gradient from `population_gradient`) with the exact analytic packed
        // gradient (PR #381 review #4).
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        for optimizer in [
            Optimizer::Bfgs,
            Optimizer::Lbfgs,
            Optimizer::Slsqp,
            Optimizer::NloptLbfgs,
            Optimizer::Mma,
        ] {
            for &build in &[&ad_build(), &ci_build()] {
                assert_eq!(
                    gradient_method_outer(build, EstimationMethod::FoceI, optimizer, &m),
                    GradientMethodKind::Analytic,
                    "expected analytic for in-scope gradient optimizer {optimizer:?}"
                );
            }
        }
    }

    #[test]
    fn outer_reports_fd_when_out_of_scope_or_forced() {
        // `gradient = fd` forces FD; an ODE model is outside the analytic scope.
        let forced_fd = test_helpers::analytical_model(GradientMethod::Fd);
        let ode = test_helpers::ode_model(GradientMethod::Auto);
        for m in [&forced_fd, &ode] {
            assert_eq!(
                gradient_method_outer(&ci_build(), EstimationMethod::FoceI, Optimizer::Bfgs, m),
                GradientMethodKind::FiniteDifferences
            );
        }
    }

    #[test]
    fn outer_bobyqa_not_applicable() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Foce, Optimizer::Bobyqa, &m),
            GradientMethodKind::NotApplicable
        );
    }

    #[test]
    fn outer_saem_not_applicable() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Saem, Optimizer::Bobyqa, &m),
            GradientMethodKind::NotApplicable
        );
    }

    #[test]
    fn outer_gn_always_fd() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::FoceGn, Optimizer::Bobyqa, &m),
            GradientMethodKind::FiniteDifferences
        );
    }

    /// #486 σ-magnitude FOCE port: a custom / time-varying residual-magnitude model
    /// is now analytic on **both** loops (the Sheiner–Beal FOCE assembly threads
    /// `mult(θ)` through its marginal `R⁰`), so `gradient_method_outer` reports
    /// `Analytic` under plain `method = foce` as well as `method = focei`. Regression
    /// test that FOCE no longer routes a magnitude model to FD.
    #[test]
    fn outer_custom_magnitude_reports_analytic_under_foce_and_focei() {
        let content = "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  theta RUV_LATE(1.5, 0.0, 10.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n";
        let m = crate::parser::model_parser::parse_model_string(content).expect("parse");
        assert!(m.has_custom_ruv_magnitude());
        // `Optimizer::Auto` now resolves to a gradient-based optimizer under FOCE too,
        // so the reported method is `Analytic` (not the old derivative-free bobyqa).
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Foce, Optimizer::Auto, &m),
            GradientMethodKind::Analytic,
            "FOCE + custom magnitude now has the analytic outer gradient"
        );
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::FoceI, Optimizer::Auto, &m),
            GradientMethodKind::Analytic,
            "FOCEI + custom magnitude has the analytic outer gradient"
        );
        // A user-forced concrete gradient optimizer under FOCE also reports Analytic.
        assert_eq!(
            gradient_method_outer(
                &ad_build(),
                EstimationMethod::Foce,
                Optimizer::NloptLbfgs,
                &m
            ),
            GradientMethodKind::Analytic,
            "FOCE + magnitude with a forced gradient optimizer reports the analytic method"
        );
    }
}
