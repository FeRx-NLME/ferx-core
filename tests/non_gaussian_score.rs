//! Tier-2: a non-Gaussian model's per-subject score reaches the S / R⁻¹SR⁻¹
//! covariance and the gradient-based outer optimizers.
//!
//! The fixed-EBE per-subject gradient (`subject_nll_pop_grad`) used to route
//! `[event_model]` / `[binary_model]` models to closed forms that score only the
//! Gaussian rows. On a TTE-only model every subject's score came back `0`, so the
//! S matrix was singular, `covariance_method = rsr` reported SE = 0 with no
//! warning, and L-BFGS / SLSQP / trust-region / GN stopped at the initial
//! estimates — all but trust-region reporting convergence.
//!
//! The oracle is the exponential proportional-hazards likelihood itself, scored by
//! hand below — no second engine. Neither test runs a fit to convergence: the
//! covariance test evaluates at the closed-form MLE (`outer_maxiter = 0`) and the
//! optimizer test takes a handful of outer iterations, so both stay ungated.
//!
//! Everything is behind `#[cfg(feature = "survival")]`, so the file compiles to no
//! tests without the feature.

mod common;

#[cfg(feature = "survival")]
mod non_gaussian_score {
    use crate::common;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{CovarianceMethod, EstimationMethod, FitResult, Optimizer, Population};
    use ferx_core::{fit, FitOptions};
    use nalgebra::{Matrix2, Vector2};

    /// `h(t) = LAM · exp(B·X)`, no random effects — the shape of the covariate TTE
    /// model that surfaced #1744.
    const MODEL: &str = r"
[parameters]
  theta LAM(0.05, 0.0001, 10.0)
  theta B(0.3, -10.0, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = LAM
  loghr  = B * X
";

    /// `(time, event, X)`: 24 subjects, 19 events, the X = 1 arm at higher hazard.
    const DATA: &[(f64, u8, f64)] = &[
        (12.0, 1, 0.0),
        (30.0, 0, 0.0),
        (5.5, 1, 0.0),
        (18.0, 1, 0.0),
        (40.0, 0, 0.0),
        (9.0, 1, 0.0),
        (25.0, 1, 0.0),
        (40.0, 0, 0.0),
        (14.0, 1, 0.0),
        (3.0, 1, 0.0),
        (33.0, 1, 0.0),
        (40.0, 0, 0.0),
        (4.0, 1, 1.0),
        (8.5, 1, 1.0),
        (2.0, 1, 1.0),
        (40.0, 0, 1.0),
        (11.0, 1, 1.0),
        (6.0, 1, 1.0),
        (1.5, 1, 1.0),
        (19.0, 0, 1.0),
        (7.5, 1, 1.0),
        (3.5, 1, 1.0),
        (22.0, 1, 1.0),
        (10.0, 1, 1.0),
    ];

    fn population() -> Population {
        let pairs: Vec<(f64, u8)> = DATA.iter().map(|&(t, d, _)| (t, d)).collect();
        let mut pop = common::tte_pop_from_pairs(&pairs);
        for (s, &(_, _, x)) in pop.subjects.iter_mut().zip(DATA) {
            s.covariates.insert("X".to_string(), x);
        }
        pop.covariate_names = vec!["X".to_string()];
        pop
    }

    /// Per-subject score of the log-likelihood `δ(log λ + βx) − λ e^{βx} t`
    /// w.r.t. `(λ, β)`.
    fn score(lam: f64, b: f64, (t, d, x): (f64, u8, f64)) -> Vector2<f64> {
        let (d, e) = (f64::from(d), (b * x).exp());
        Vector2::new(d / lam - e * t, d * x - lam * e * t * x)
    }

    /// Observed information `−∂²ℓ/∂(λ, β)²`, summed over subjects.
    fn information(lam: f64, b: f64) -> Matrix2<f64> {
        DATA.iter().fold(Matrix2::zeros(), |acc, &(t, d, x)| {
            let (d, et) = (f64::from(d), (b * x).exp() * t);
            acc + Matrix2::new(d / (lam * lam), et * x, et * x, lam * et * x * x)
        })
    }

    fn score_cross_product(lam: f64, b: f64) -> Matrix2<f64> {
        DATA.iter().fold(Matrix2::zeros(), |acc, &row| {
            let s = score(lam, b, row);
            acc + s * s.transpose()
        })
    }

    /// The MLE by Newton on the closed-form score and information.
    fn mle() -> (f64, f64) {
        let (mut lam, mut b) = (0.05, 0.0);
        for _ in 0..50 {
            let g: Vector2<f64> = DATA.iter().map(|&row| score(lam, b, row)).sum();
            let step = information(lam, b)
                .try_inverse()
                .expect("information is PD")
                * g;
            lam += step[0];
            b += step[1];
            if step.norm() < 1e-14 {
                break;
            }
        }
        let g: Vector2<f64> = DATA.iter().map(|&row| score(lam, b, row)).sum();
        assert!(g.norm() < 1e-9, "Newton did not reach the MLE: score {g:?}");
        (lam, b)
    }

    fn closed_form_se(cov: Matrix2<f64>) -> [f64; 2] {
        [cov[(0, 0)].sqrt(), cov[(1, 1)].sqrt()]
    }

    fn options() -> FitOptions {
        FitOptions {
            method: EstimationMethod::FoceI,
            user_set_keys: vec!["method".to_string()],
            run_covariance_step: false,
            ..FitOptions::default()
        }
    }

    /// An evaluation, not a fit: `outer_maxiter = 0` is ferx's `MAXEVAL=0`.
    fn evaluate(theta: [f64; 2], options: &FitOptions) -> FitResult {
        let model = parse_model_string(MODEL).expect("MODEL parses");
        let mut params = model.default_params.clone();
        params.theta = theta.to_vec();
        let options = FitOptions {
            outer_maxiter: 0,
            ..options.clone()
        };
        fit(&model, &population(), &params, &options).expect("the evaluation runs")
    }

    fn se_at_mle(method: CovarianceMethod) -> Vec<f64> {
        let (lam, b) = mle();
        let res = evaluate(
            [lam, b],
            &FitOptions {
                run_covariance_step: true,
                covariance_method: method,
                covariance_method_set: true,
                ..options()
            },
        );
        res.se_theta
            .unwrap_or_else(|| panic!("{method:?}: no SEs; warnings: {:?}", res.warnings))
    }

    /// `MATRIX=S` and `MATRIX=RSR` are the two estimators that read the per-subject
    /// score; `MATRIX=R` does not and is the control — it was right before the fix.
    ///
    /// Measured worst relative SE error against the closed form, on Linux x86_64
    /// (glibc 2.28), identical in every printed digit on macOS arm64 — the
    /// quantities are a 24-row sum, not a fit: S 1.5e-8, R 3.6e-4, RSR 6.8e-4. The
    /// `eprintln!` below prints them, so `--nocapture` re-measures. S is that tight
    /// because the score is a central difference of a smooth objective; R is the FD
    /// Hessian of the OFV, and RSR inherits R's error. Bounds: S 1e-6 (65×), R 2e-3
    /// (5.5×), RSR 3e-3 (4.4×). Before the fix S was singular (no SEs, so
    /// `se_at_mle` panics) and RSR returned SE = 0 (relative error 1).
    #[test]
    fn s_and_sandwich_covariance_match_the_closed_form_score() {
        let (lam, b) = mle();
        let r_inv = information(lam, b)
            .try_inverse()
            .expect("information is PD");
        let s = score_cross_product(lam, b);
        let cases = [
            (CovarianceMethod::Hessian, closed_form_se(r_inv), 2e-3),
            (
                CovarianceMethod::CrossProduct,
                closed_form_se(s.try_inverse().expect("S is PD")),
                1e-6,
            ),
            (
                CovarianceMethod::Sandwich,
                closed_form_se(r_inv * s * r_inv),
                3e-3,
            ),
        ];
        for (method, want, tol) in cases {
            let got = se_at_mle(method);
            assert_eq!(got.len(), 2, "{method:?}: one SE per θ");
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(g.is_finite(), "{method:?} SE[{k}] is not finite: {g}");
                let rel = (g - w).abs() / w;
                eprintln!("{method:?} SE[{k}]: rel {rel:.3e} (bound {tol:e})");
                assert!(
                    rel < tol,
                    "{method:?} SE[{k}]: ferx {g:.8e}, closed form {w:.8e}, rel {rel:.3e}"
                );
            }
        }
    }

    /// Every outer optimizer that consumes the per-subject score: the FOCEI
    /// gradient optimizers (`ad_population_gradient`), pure GN (`build_gn_system`)
    /// and `gn_hybrid` with a gradient-based polish.
    ///
    /// From `(LAM, B) = (0.05, 0.3)` the OFV gap to the MLE is 2.692. Measured OFV
    /// left after `outer_maxiter = 5`, on Linux x86_64 (glibc 2.28) and identical in
    /// every printed digit on macOS arm64: L-BFGS 3e-14, SLSQP 8.1e-6, trust-region
    /// 1.1e-2 (0.41% of the gap), GN 1.1e-2, `gn_hybrid` + L-BFGS 3e-14 (its polish
    /// has its own iteration budget). The bound — 10% of the gap left — is 24× the
    /// worst. Trust-region and GN agree bit-for-bit here: both solve the same BHHH
    /// model `4 Σ gᵢgᵢᵀ` in a trust region, so on an η-free model they are two
    /// callers of one geometry, not two checks. Before the fix every case stayed at
    /// the initial estimates — 100% of the gap left, each case killed on its own —
    /// while L-BFGS, SLSQP and GN reported convergence.
    #[test]
    fn gradient_optimizers_move_off_the_initial_estimates() {
        let model = parse_model_string(MODEL).expect("MODEL parses");
        let pop = population();
        let (lam, b) = mle();
        let ofv_mle = evaluate([lam, b], &options()).ofv;
        let ofv_start = evaluate([0.05, 0.3], &options()).ofv;
        let gap = ofv_start - ofv_mle;
        assert!(
            ofv_mle.is_finite() && gap > 1.0,
            "fixture precondition: the start is well off the MLE (gap {gap})"
        );
        let cases = [
            (EstimationMethod::FoceI, Optimizer::NloptLbfgs),
            (EstimationMethod::FoceI, Optimizer::Slsqp),
            (EstimationMethod::FoceI, Optimizer::TrustRegion),
            (EstimationMethod::FoceGn, Optimizer::Auto),
            (EstimationMethod::FoceGnHybrid, Optimizer::NloptLbfgs),
        ];
        for (method, optimizer) in cases {
            let res = fit(
                &model,
                &pop,
                &model.default_params,
                &FitOptions {
                    method,
                    optimizer,
                    outer_maxiter: 5,
                    ..options()
                },
            )
            .expect("the fit runs");
            assert!(
                res.ofv.is_finite(),
                "{method:?}/{optimizer:?}: OFV is not finite"
            );
            let left = (res.ofv - ofv_mle) / gap;
            eprintln!(
                "{method:?}/{optimizer:?}: OFV left {:.3e} ({:.3e} of the gap)",
                res.ofv - ofv_mle,
                left
            );
            assert!(
                left < 0.1,
                "{method:?}/{optimizer:?}: {:.1}% of the OFV gap left after 5 outer iterations \
                 (OFV {:.6}, start {ofv_start:.6}, MLE {ofv_mle:.6}, θ {:?})",
                100.0 * left,
                res.ofv,
                res.theta
            );
        }
    }
}
