use crate::types::{CompiledModel, EstimationMethod, GradientMethod, Optimizer};

/// Compile-time build metadata embedded by `build.rs`.
pub struct BuildInfo {
    pub variant: &'static str,
    pub profile: &'static str,
    pub ferx_version: &'static str,
    pub rustc_version: &'static str,
    pub build_timestamp: u64,
    pub has_autodiff: bool,
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
    has_autodiff: cfg!(feature = "autodiff"),
};

/// Reported gradient method for a fit loop (inner or outer).
///
/// Distinct from [`crate::types::GradientMethod`] which controls _selection_;
/// this enum describes what actually runs and is used only for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientMethodKind {
    /// Enzyme automatic differentiation — exact, constant cost in n_eta.
    EnzymeAD,
    /// Central finite differences — cost scales as 2×n_eta per gradient call.
    FiniteDifferences,
    /// Not applicable: derivative-free optimizer or sampling-based step.
    NotApplicable,
}

impl GradientMethodKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EnzymeAD => "Enzyme AD",
            Self::FiniteDifferences => "finite differences",
            Self::NotApplicable => "N/A",
        }
    }
}

/// Gradient method used in the inner (per-subject EBE) loop.
///
/// Mirrors the runtime resolution in
/// [`estimation::inner_optimizer::resolve_gradient_method`]: AD is used iff
/// (a) the crate is compiled with the `autodiff` feature, (b) the model has
/// an analytical PK path (`tv_fn` populated — ODE models have no AD path),
/// and (c) the user did not force `gradient_method = Fd`. Otherwise FD.
pub fn gradient_method_inner(build: &BuildInfo, model: &CompiledModel) -> GradientMethodKind {
    let user_forces_fd = matches!(model.gradient_method, GradientMethod::Fd);
    if build.has_autodiff && model.tv_fn.is_some() && !user_forces_fd {
        GradientMethodKind::EnzymeAD
    } else {
        GradientMethodKind::FiniteDifferences
    }
}

/// Gradient method used in the outer (population parameter) loop.
///
/// Depends on the chosen estimation method/optimizer:
/// - NLopt-based: NLopt uses its own internal FD.
/// - BOBYQA: derivative-free — no outer gradient at all.
/// - Built-in BFGS/LBFGS: always central finite differences (no outer AD path).
/// - TrustRegion: FD Hessian via the argmin crate.
/// - GN/GnHybrid: BHHH outer approximation — always FD.
/// - SAEM: MH E-step has no gradient; M-step uses NLopt internally.
///
/// `_build` is accepted for API symmetry with [`gradient_method_inner`] and
/// to leave room for a future outer-AD path; the current implementation does
/// not consult it.
pub fn gradient_method_outer(
    _build: &BuildInfo,
    method: EstimationMethod,
    optimizer: Optimizer,
) -> GradientMethodKind {
    match method {
        EstimationMethod::Saem | EstimationMethod::Imp | EstimationMethod::Impmap => {
            GradientMethodKind::NotApplicable
        }
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
            | Optimizer::TrustRegion => GradientMethodKind::FiniteDifferences,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers;

    fn ad_build() -> BuildInfo {
        BuildInfo {
            variant: "autodiff",
            profile: "debug",
            ferx_version: "0.1.0",
            rustc_version: "rustc 1.0.0",
            build_timestamp: 0,
            has_autodiff: true,
        }
    }

    fn ci_build() -> BuildInfo {
        BuildInfo {
            variant: "ci",
            profile: "release",
            ferx_version: "0.1.0",
            rustc_version: "rustc 1.0.0",
            build_timestamp: 0,
            has_autodiff: false,
        }
    }

    #[test]
    fn inner_ad_build_analytical_auto_returns_enzyme() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_inner(&ad_build(), &m),
            GradientMethodKind::EnzymeAD
        );
    }

    #[test]
    fn inner_ci_build_returns_fd() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_inner(&ci_build(), &m),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn inner_ode_model_returns_fd_even_with_ad_build() {
        let m = test_helpers::ode_model(GradientMethod::Auto);
        assert_eq!(
            gradient_method_inner(&ad_build(), &m),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn inner_user_forces_fd_returns_fd_even_with_ad_build() {
        let m = test_helpers::analytical_model(GradientMethod::Fd);
        assert_eq!(
            gradient_method_inner(&ad_build(), &m),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn outer_nlopt_always_fd() {
        for optimizer in [Optimizer::Slsqp, Optimizer::NloptLbfgs, Optimizer::Mma] {
            for &build in &[&ad_build(), &ci_build()] {
                assert_eq!(
                    gradient_method_outer(build, EstimationMethod::Foce, optimizer),
                    GradientMethodKind::FiniteDifferences,
                    "expected FD for NLopt optimizer {:?}",
                    optimizer
                );
            }
        }
    }

    #[test]
    fn outer_bobyqa_not_applicable() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Foce, Optimizer::Bobyqa),
            GradientMethodKind::NotApplicable
        );
    }

    #[test]
    fn outer_bfgs_always_fd() {
        // Built-in outer BFGS/LBFGS use central finite differences regardless
        // of the build variant — there is no outer-AD path today.
        for &build in &[&ad_build(), &ci_build()] {
            for optimizer in [Optimizer::Bfgs, Optimizer::Lbfgs] {
                assert_eq!(
                    gradient_method_outer(build, EstimationMethod::Foce, optimizer),
                    GradientMethodKind::FiniteDifferences,
                    "expected FD for outer optimizer {:?}",
                    optimizer
                );
            }
        }
    }

    #[test]
    fn outer_saem_not_applicable() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Saem, Optimizer::Bobyqa),
            GradientMethodKind::NotApplicable
        );
    }

    #[test]
    fn outer_gn_always_fd() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::FoceGn, Optimizer::Bobyqa),
            GradientMethodKind::FiniteDifferences
        );
    }
}
