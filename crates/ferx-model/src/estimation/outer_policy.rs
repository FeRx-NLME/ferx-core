//! The outer optimizer's resolution rule and the objective-validity test. Below the estimators
//! so `types` and `build_info` can call them (#3041).
use crate::types::*;

/// The outer optimizer a fit actually runs, and the warning owed when an explicit
/// choice could not be honoured. **Single source of truth for `auto`**:
/// `Optimizer::resolve_auto` alone is wrong on a mixture model (#1381).
/// `analytic_outer_gradient` is a parameter so the same rule answers the
/// counterfactual "what would `auto` pick had the gradient not been forced?".
pub fn resolve_outer_optimizer(
    requested: Optimizer, model: &CompiledModel, has_mixture: bool, analytic_outer_gradient: bool,
) -> (Optimizer, Option<String>) {
    // Mixture models (#977): BOBYQA is the default (robust to label switching). An
    // explicit NLopt gradient optimizer is honoured (its closure branches to the mixture
    // gradient), as is explicit `bobyqa` with no warning (#2852); everything else,
    // including `auto` and BFGS / trust region, falls back to BOBYQA.
    if has_mixture {
        // Exhaustive, no wildcard: a new variant must say here whether it carries
        // the mixture objective.
        let resolved = match requested {
            Optimizer::Bobyqa | Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma => {
                requested
            }
            // `Auto` is the mixture default; the built-in BFGS / L-BFGS and
            // trust-region paths do not carry the mixture objective.
            Optimizer::Auto | Optimizer::Bfgs | Optimizer::Lbfgs | Optimizer::TrustRegion => {
                Optimizer::Bobyqa
            }
        };
        // The warning is owed exactly when an *explicit* choice did not run — keyed
        // on the outcome, not on a second list of variants, so "replaced" can never
        // be said of an optimizer that ran as requested (#2852). `Auto` asked for
        // no optimizer in particular and resolves silently by design.
        let replaced = requested != Optimizer::Auto && resolved != requested;
        let warning = replaced.then(|| {
            format!(
                "Mixture models are optimized with BOBYQA or an NLopt gradient method \
                 (SLSQP / L-BFGS / MMA); the requested `{}` optimizer does not carry \
                 the mixture objective and was replaced by BOBYQA.",
                requested.label()
            )
        });
        return (resolved, warning);
    }
    (requested.resolve_auto_given_analytic(model, analytic_outer_gradient), None)
}

/// Resolve the derivative-free `bobyqa` outer optimizer's `ftol_rel` stop tolerance.
///
/// `override_ftol` (`[fit_options] outer_ftol`) wins when set. Otherwise auto-select:
/// `1e-8` for a **pure non-Gaussian** model (TTE or the Phase-4 categorical family,
/// #760) — its data objective is evaluated *exactly*, so the looser historical `1e-6`
/// stopped BOBYQA short of the optimum on the near-flat frailty/random-effect-ω² ridge
/// (#469: a Weibull shape-frailty read 0.204 vs the NONMEM/nlmixr2 0.175 consensus;
/// `1e-8` lands 0.176) — and `1e-6` for everything else. The floor is deliberate: on a
/// **noisy** objective (ODE solver error, or an FD-inner FOCE model such as LTBS) `1e-8`
/// is unreachable, so BOBYQA would grind toward its maxeval budget instead of converging
/// (≈3× the evaluations on an ODE fit). A non-Gaussian endpoint carried on an ODE
/// disposition (`is_ode` true) therefore keeps `1e-6`.
/// OFVs at or above this are treated as diverged/invalid.
///
/// The inner objective clamps a blown-up value to a `~1e20` sentinel, which is
/// *finite* — so `is_finite()` alone does not separate a diverged run from a real
/// one. This cutoff sits well below the sentinel and far above any legitimate
/// population OFV, so a real fit never trips it.
pub const DIVERGENCE_OFV: f64 = 1e14;

/// Whether an OFV is a real population objective rather than a clamped sentinel or `NaN`.
/// One rule for multi-start ranking (`api::fit::multistart_prefers`) and the convergence
/// verdict.
pub fn ofv_is_valid(ofv: f64) -> bool {
    ofv.is_finite() && ofv < DIVERGENCE_OFV
}
