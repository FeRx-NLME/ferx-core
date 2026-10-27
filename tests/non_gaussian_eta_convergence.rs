//! Tier-3 (#1778): on a non-Gaussian model with η, every gradient-based outer optimizer
//! reaches the optimum derivative-free BOBYQA finds on the same platform.
//!
//! The fixed-EBE score these routes used to descend on drops `∂_η log|H̃| · dη̂/dθ`.
//! With a Gaussian data term that term is carried by the adjoint (or is zero with additive
//! error); with a TTE hazard or a Bernoulli likelihood `H_data(η)` depends strongly on η,
//! so the score was biased and each route stopped at a non-stationary point. All but
//! trust-region (on the joint model) said they had converged. The fix routes these models
//! to the reconverged FD score (`gauss_newton::fixed_ebe_score_drops_ebe_response`); the
//! Tier-1 tests in `outer_optimizer_tests.rs`, `gauss_newton.rs` and `trust_region.rs` pin
//! each call site, and this file pins the end-to-end effect, which only a fit to
//! convergence can show.
//!
//! The reference is a BOBYQA fit in the same test run, so no optimum is hard-coded and the
//! comparison stays within one platform. The gap is one-sided, `route − BOBYQA`: a route
//! that ends *below* BOBYQA passes (the joint model's routes do, by ~3e-4).
//!
//! Bounds, measured as `route − BOBYQA` with the fix and with the scope predicate forced to
//! `false` (the pre-fix routing). Every route fails under that mutation. Platform: MACOS_PENDING_LINUX.
//!
//! | model | bound | worst gap, fix | smallest gap, predicate off |
//! |---|---|---|---|
//! | `tte_exponential` (lbfgs, slsqp, mma, trust_region, gn) | 5e-3 | +1.16e-4 (mma) | +5.05e-2 (mma) |
//! | `pktte_joint` (lbfgs, slsqp, mma) | 1e-3 | −3.12e-4 (mma) | +3.32e-3 (lbfgs) |
//! | binary with η (lbfgs, mma, trust_region) | 2e-3 | +4.80e-4 (mma) | +5.06e-3 (lbfgs) |
//!
//! Each bound sits at least 2.5× under the smallest pre-fix gap, and at least 4× over the
//! worst gap with the fix.
//!
//! Left out on purpose:
//! - `method = gn` and `optimizer = trust_region` on the joint PK + TTE model. Neither
//!   converges there, before or after the fix (GN 71.92 → 72.63, trust-region 70.68 →
//!   68.42, optimum 66.975), and both say so. That is a separate GN/BHHH limitation.
//! - `gn_hybrid`, which reached the optimum before the fix too (its FOCEI polish runs
//!   BOBYQA), so it cannot discriminate.
//! - SLSQP on the binary model: before the fix its gap was 2.9e-3 against 1.4e-3 after,
//!   a 2× separation too small to hold a band across platforms.
//! - GN on the binary model: `method = gn` is rejected for `[binary_model]` endpoints.
//! - CTMM, which this fix does not change (see `fixed_ebe_score_drops_ebe_response`).

#[cfg(feature = "survival")]
mod non_gaussian_eta_convergence {
    use ferx_core::api::read_population_for;
    use ferx_core::parser::model_parser::{parse_full_model, parse_full_model_file};
    use ferx_core::types::{EstimationMethod, Optimizer, ParsedModel};
    use ferx_core::{fit, FitOptions};
    use std::path::Path;

    const LBFGS: (EstimationMethod, Optimizer) = (EstimationMethod::FoceI, Optimizer::NloptLbfgs);
    const SLSQP: (EstimationMethod, Optimizer) = (EstimationMethod::FoceI, Optimizer::Slsqp);
    const MMA: (EstimationMethod, Optimizer) = (EstimationMethod::FoceI, Optimizer::Mma);
    const TR: (EstimationMethod, Optimizer) = (EstimationMethod::FoceI, Optimizer::TrustRegion);
    const GN: (EstimationMethod, Optimizer) = (EstimationMethod::FoceGn, Optimizer::Auto);

    /// `examples/binary_logistic.ferx` with a per-subject random intercept.
    fn binary_with_eta() -> ParsedModel {
        let src = std::fs::read_to_string("examples/binary_logistic.ferx")
            .expect("model file")
            .replace("  theta THT(", "  omega ETA_I ~ 0.25\n  theta THT(")
            .replace(
                "logit = TH0 + THX * X + THT * TIME",
                "logit = TH0 + THX * X + THT * TIME + ETA_I",
            );
        parse_full_model(&src).expect("parse")
    }

    fn fit_ofv(
        label: &str,
        parsed: &ParsedModel,
        data: &str,
        (method, optimizer): (EstimationMethod, Optimizer),
    ) -> f64 {
        let (pop, _) = read_population_for(
            &parsed.model,
            &parsed.covariate_decls,
            data,
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        let mut user_set_keys = parsed.fit_options.user_set_keys.clone();
        user_set_keys.extend(["method".to_string(), "optimizer".to_string()]);
        let options = FitOptions {
            method,
            optimizer,
            user_set_keys,
            run_covariance_step: false,
            threads: Some(1),
            ..parsed.fit_options.clone()
        };
        let res = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
            .unwrap_or_else(|e| panic!("{label} {method:?}/{optimizer:?}: {e}"));
        assert!(
            res.ofv.is_finite(),
            "{label} {method:?}/{optimizer:?}: OFV is not finite"
        );
        res.ofv
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn gradient_optimizers_reach_the_bobyqa_optimum_on_non_gaussian_models_with_eta() {
        let tte = parse_full_model_file(Path::new("examples/tte_exponential.ferx")).expect("parse");
        let joint = parse_full_model_file(Path::new("examples/pktte_joint.ferx")).expect("parse");
        let binary = binary_with_eta();
        let cases: [(&str, &ParsedModel, &str, Vec<_>, f64); 3] = [
            (
                "tte_exponential",
                &tte,
                "data/tte_exponential.csv",
                vec![LBFGS, SLSQP, MMA, TR, GN],
                5e-3,
            ),
            (
                "pktte_joint",
                &joint,
                "data/pktte_joint.csv",
                vec![LBFGS, SLSQP, MMA],
                1e-3,
            ),
            (
                "binary with η",
                &binary,
                "data/binary_logistic.csv",
                vec![LBFGS, MMA, TR],
                2e-3,
            ),
        ];
        let mut failures = Vec::new();
        for (label, parsed, data, routes, bound) in cases {
            let reference = fit_ofv(
                label,
                parsed,
                data,
                (EstimationMethod::FoceI, Optimizer::Bobyqa),
            );
            for route in routes {
                let gap = fit_ofv(label, parsed, data, route) - reference;
                eprintln!(
                    "[#1778] {label} {:?}/{:?}: OFV − BOBYQA = {gap:+.4e} (bound {bound:e})",
                    route.0, route.1
                );
                if gap >= bound {
                    failures.push(format!(
                        "{label} {:?}/{:?}: {gap:+.4e} above BOBYQA's {reference:.6} (bound {bound:e})",
                        route.0, route.1
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "gradient optimizers stopped above the optimum:\n{}",
            failures.join("\n")
        );
    }
}
