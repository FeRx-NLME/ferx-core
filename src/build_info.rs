use crate::types::{CompiledModel, EstimationMethod, GradientMethod, Optimizer};

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
/// ODE models have none) and is in the provider's scope (not LTBS / expression
/// scaling / SDE), and the user did not force `gradient_method = Fd`. Otherwise
/// FD. (The exact per-subject split is reported by `gradient_route_summary`.)
pub fn gradient_method_inner(_build: &BuildInfo, model: &CompiledModel) -> GradientMethodKind {
    // Report off the *same* model-level predicate `find_ebe` consults, so the two
    // can't diverge as scope grows (PR #381 review #9). Per-subject FD fallbacks
    // (time-varying covariates, survival obs) are reported separately by
    // `gradient_route_summary` / `fd_fallback_warning`.
    if crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(model) {
        GradientMethodKind::Analytic
    } else {
        GradientMethodKind::FiniteDifferences
    }
}

/// Gradient method used in the outer (population parameter) loop.
///
/// For a gradient-driven FOCE/FOCEI fit the outer optimiser uses the **exact
/// analytic** packed gradient when the model is in the sensitivity provider's
/// scope (`sens_supported` / `iov_analytical_supported`) and the user did not
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
        EstimationMethod::Foce | EstimationMethod::FoceI => match optimizer {
            Optimizer::Bobyqa => GradientMethodKind::NotApplicable,
            Optimizer::Bfgs
            | Optimizer::Lbfgs
            | Optimizer::Slsqp
            | Optimizer::NloptLbfgs
            | Optimizer::Mma
            | Optimizer::TrustRegion => {
                let user_forces_fd = matches!(model.gradient_method, GradientMethod::Fd);
                let analytic = !user_forces_fd
                    && (crate::sens::provider::sens_supported(model)
                        || crate::sens::provider::iov_analytical_supported(model));
                if analytic {
                    GradientMethodKind::Analytic
                } else {
                    GradientMethodKind::FiniteDifferences
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers;

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
    fn inner_ltbs_model_returns_fd() {
        let mut m = test_helpers::analytical_model(GradientMethod::Auto);
        m.log_transform = true; // LTBS keeps the FD inner gradient
        assert_eq!(
            gradient_method_inner(&ci_build(), &m),
            GradientMethodKind::FiniteDifferences
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
}
