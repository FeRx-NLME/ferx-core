use super::*;
use crate::sens::dual2::Dual2;

/// Reference values computed in Python outside ferx (`math.expm1`, and the
/// Petersson / Pharmpy formulas written out literally), so the kernel is
/// checked against a second implementation, not against itself.
/// Dies under: any coefficient of the t-dist series, the John-Draper sign
/// handling, or the Box-Cox closed form changing.
#[test]
fn each_transform_matches_an_external_reference() {
    use ShapeKind::*;
    let cases = [
        (BoxCox, 0.7, 0.5, 0.8381350971865145),
        (BoxCox, -1.3, -1.0, -2.6692966676192444),
        (BoxCox, 0.3, 1.0, 0.3498588075760031),
        (TDist, 0.7, 5.0, 0.7556362165187498),
        (TDist, -1.3, 3.0, -1.6684439646122684),
        (TDist, 0.2, 80.0, 0.2006511729640625),
        (JohnDraper, 0.7, 0.5, 0.6076809620810595),
        (JohnDraper, -1.3, 2.0, -2.1449999999999996),
        (JohnDraper, 0.4, -1.0, 0.2857142857142857),
    ];
    for (k, eta, s, want) in cases {
        let got = shape_g(k, eta, s);
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
            "{}({eta}, {s}) = {got}, want {want}",
            k.name()
        );
    }
}

/// `h(0) = 0` for every kind and shape, so the population prediction (η = 0)
/// is the untransformed one. Dies under: a constant term in any kernel.
#[test]
fn every_transform_is_zero_at_zero() {
    for k in ShapeKind::ALL {
        for s in [-2.0, -0.5, 0.0, 0.01, 1.0, 3.0, 80.0] {
            if k == ShapeKind::TDist && s == 0.0 {
                continue; // ν = 0 is outside the domain
            }
            assert_eq!(shape_g(k, 0.0, s), 0.0, "{}(0, {s})", k.name());
        }
    }
}

/// Box-Cox is exact through λ = 0, where the hand-written `(exp(η)^λ−1)/λ` is
/// `0/0` (evaluated as 0, #1714): `η·exprel(λη)` gives `η`, and on both sides of
/// `exprel`'s own series/closed-form switch (`|λη| = 2`) it agrees with libm's
/// `expm1`. Measured worst relative error over the grid below: under 2 ulp; bound 4 ulp.
/// Dies under: the closed form `expm1(λη)/λ` (λ = 0 → NaN).
#[test]
fn box_cox_is_exact_through_lambda_zero() {
    assert_eq!(shape_g(ShapeKind::BoxCox, 0.3, 0.0), 0.3);
    assert_eq!(shape_g(ShapeKind::BoxCox, -1.7, 0.0), -1.7);
    let mut worst = 0.0f64;
    for eta in [-2.0, -0.3, 0.05, 0.4, 1.5] {
        for x in [1e-9, 1e-6, 1e-3, 0.05, 0.2, 1.0, 1.999, 2.001, 3.0] {
            for sign in [-1.0, 1.0] {
                let lambda: f64 = sign * x / eta;
                let want = (lambda * eta).exp_m1() / lambda;
                let got = shape_g(ShapeKind::BoxCox, eta, lambda);
                assert!(
                    got.is_finite() && want.is_finite(),
                    "boxcox({eta}, {lambda}) = {got} vs {want}"
                );
                // `want` is `expm1(λη)/λ` with `λη ≠ 0`, so never 0.
                worst = worst.max((got - want).abs() / want.abs().max(1e-300));
            }
        }
    }
    eprintln!("box-cox worst relative error vs exp_m1: {worst:e}");
    assert!(worst <= 4.0 * f64::EPSILON, "worst relative error {worst:e}");
}

/// The identity points: Box-Cox λ → 0, t-dist ν → ∞, John-Draper λ = 1 are η;
/// John-Draper λ → 0 is `sign(η)·ln(1+|η|)`.
#[test]
fn each_transform_reduces_to_its_identity() {
    for eta in [-1.4, -0.2, 0.0, 0.3, 2.1] {
        assert_eq!(shape_g(ShapeKind::BoxCox, eta, 0.0), eta);
        let jd1 = shape_g(ShapeKind::JohnDraper, eta, 1.0);
        assert!((jd1 - eta).abs() < 1e-15, "johndraper({eta}, 1) = {jd1}");
        let td = shape_g(ShapeKind::TDist, eta, 1e12);
        assert!((td - eta).abs() < 1e-11, "tdist({eta}, 1e12) = {td}");
        let jd0 = shape_g(ShapeKind::JohnDraper, eta, 0.0);
        let want = eta.signum() * eta.abs().ln_1p();
        assert!((jd0 - want).abs() < 1e-15, "johndraper({eta}, 0) = {jd0}");
    }
}

/// `Dual2` vs central finite differences of the `f64` kernel, for the gradient
/// and Hessian in (η, shape), on both sides of 0 and of `exprel`'s series/closed-form
/// switch at `|λη| = 2`. AGENTS.md: every analytic sensitivity needs this parity.
/// Dies under: any kernel step that is not differentiable over `PkNum` (a
/// value-only branch, a `from_f64(x.val())` lift).
#[test]
fn shape_g_dual_matches_fd() {
    use ShapeKind::*;
    let points = [
        (BoxCox, 0.7, 0.5),
        (BoxCox, -1.1, -0.8),
        (BoxCox, 0.4, 0.002),  // λη = 8e-4
        (BoxCox, 0.4, 4.9975), // λη = 1.999: `exprel`'s series, at its switch
        (BoxCox, 0.4, 5.0025), // λη = 2.001: its closed form
        (BoxCox, 0.4, 0.0),
        (TDist, 0.7, 5.0),
        (TDist, -1.3, 3.5),
        (JohnDraper, 0.7, 0.5),
        (JohnDraper, -0.9, 2.0),
        (JohnDraper, 0.3, 0.0),
    ];
    for (k, eta, s) in points {
        let d = shape_g(k, Dual2::<2>::var(eta, 0), Dual2::<2>::var(s, 1));
        let v = |a: f64, b: f64| shape_g(k, a, b);
        assert_eq!(d.value, v(eta, s), "{} value", k.name());
        let h = 1e-6;
        let g = [
            (v(eta + h, s) - v(eta - h, s)) / (2.0 * h),
            (v(eta, s + h) - v(eta, s - h)) / (2.0 * h),
        ];
        let hh = 1e-4;
        let hxx = (v(eta + hh, s) - 2.0 * v(eta, s) + v(eta - hh, s)) / (hh * hh);
        let hyy = (v(eta, s + hh) - 2.0 * v(eta, s) + v(eta, s - hh)) / (hh * hh);
        let hxy = (v(eta + hh, s + hh) - v(eta + hh, s - hh) - v(eta - hh, s + hh)
            + v(eta - hh, s - hh))
            / (4.0 * hh * hh);
        let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * (1.0 + b.abs());
        for i in 0..2 {
            assert!(
                close(d.grad[i], g[i], 1e-8),
                "{}({eta},{s}) grad[{i}] {} vs {}",
                k.name(),
                d.grad[i],
                g[i]
            );
        }
        assert!(close(d.hess[0][0], hxx, 1e-5), "{} hess ηη {} vs {hxx}", k.name(), d.hess[0][0]);
        assert!(close(d.hess[1][1], hyy, 1e-5), "{} hess ss {} vs {hyy}", k.name(), d.hess[1][1]);
        assert!(close(d.hess[0][1], hxy, 1e-5), "{} hess ηs {} vs {hxy}", k.name(), d.hess[0][1]);
        assert_eq!(d.hess[0][1], d.hess[1][0], "{} hess symmetry", k.name());
    }
}

/// John-Draper at η = 0 — the inner optimizer's warm start — has `∂h/∂η = 1`,
/// not 0. Dies under: writing the odd extension as `sign(η) · h(|η|)`, whose
/// dual `sign` is flat.
#[test]
fn john_draper_slope_at_zero_is_one() {
    for s in [0.0, 0.5, 1.0, 2.5] {
        let d = shape_g(ShapeKind::JohnDraper, Dual2::<2>::var(0.0, 0), Dual2::<2>::var(s, 1));
        assert!((d.grad[0] - 1.0).abs() < 1e-15, "λ = {s}: ∂h/∂η = {}", d.grad[0]);
    }
}

/// The symbolic partial nodes evaluate to the kernel's own `Dual2` gradient,
/// and that matches finite differences.
#[test]
fn shape_partials_match_fd() {
    for (k, eta, s) in [
        (ShapeKind::BoxCox, 0.6, -0.4),
        (ShapeKind::TDist, -0.8, 4.0),
        (ShapeKind::JohnDraper, 1.2, 0.3),
    ] {
        let h = 1e-6;
        let fe = (shape_g(k, eta + h, s) - shape_g(k, eta - h, s)) / (2.0 * h);
        let fs = (shape_g(k, eta, s + h) - shape_g(k, eta, s - h)) / (2.0 * h);
        assert!((shape_f64(k, ShapeOut::DEta, eta, s) - fe).abs() < 1e-8, "{} ∂η", k.name());
        assert!((shape_f64(k, ShapeOut::DShape, eta, s) - fs).abs() < 1e-8, "{} ∂s", k.name());
        assert_eq!(shape_f64(k, ShapeOut::Value, eta, s), shape_g(k, eta, s));
    }
}

// ── Parsing: the inline functions and the `[eta_shape]` block ──────────────

use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, EtaParamType};

/// A 1-cpt IV model with ETAs `ETA_CL`, `ETA_V`, a kappa `K`, θs `TVCL`,
/// `TVV`, `L` and `NU`, the given `[individual_parameters]`, and `extra`
/// blocks appended (an `[eta_shape]`, say).
fn shaped(indiv: &str, extra: &str) -> Result<CompiledModel, String> {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         theta L(0.3, -3.0, 3.0)\n  theta NU(10.0, 3.0, 100.0)\n  \
         omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  kappa K ~ 0.04\n  sigma EPS ~ 0.01\n\n\
         [individual_parameters]\n{indiv}\n\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n  DV ~ proportional(EPS)\n\n{extra}"
    ))
}

/// The individual parameters at `(theta, eta)` as their `Debug` text — a
/// bijective rendering of every `f64`, so string equality is bit equality.
fn params_at(m: &CompiledModel, theta: &[f64], eta: &[f64]) -> String {
    let cov = std::collections::HashMap::new();
    format!("{:?}", (m.pk_param_fn)(theta, eta, &cov, 0.0))
}

/// The θ vector of `m` with `L` set to `l`, everything else at its init.
fn theta_with(m: &CompiledModel, l: f64) -> Vec<f64> {
    let mut th = m.default_params.theta.clone();
    if let Some(i) = m.theta_names.iter().position(|n| n == "L") {
        th[i] = l;
    }
    th
}

/// The `[eta_shape]` block desugars to exactly the inline spelling: the same θ
/// layout and bit-identical individual parameters at several (θ, η, κ),
/// including λ = 0 and both signs of η.
/// Dies under: the rewrite producing anything but `kind(ETA, SHAPE)` (argument
/// order swapped, the wrong kind), or a read of the ETA left untouched.
#[test]
fn the_block_and_the_inline_form_are_identical() {
    let inline =
        shaped("  CL = TVCL * exp(boxcox(ETA_CL, L) + K)\n  V = TVV * exp(tdist(ETA_V, NU))", "")
            .unwrap();
    let block = shaped(
        "  CL = TVCL * exp(ETA_CL + K)\n  V = TVV * exp(ETA_V)",
        "[eta_shape]\n  ETA_CL ~ boxcox(L)\n  ETA_V ~ tdist(NU)\n",
    )
    .unwrap();
    assert_eq!(inline.theta_names, block.theta_names);
    for l in [0.0, 0.3, -1.2] {
        for eta in [[0.4, -0.2, 0.1], [-0.7, 0.5, -0.3]] {
            let (a, b) = (theta_with(&inline, l), theta_with(&block, l));
            let pa = params_at(&inline, &a, &eta);
            assert_eq!(pa, params_at(&block, &b, &eta), "λ = {l}, η = {eta:?}");
            // Not vacuous: the shape is live, so CL differs from the
            // untransformed `TVCL * exp(ETA_CL + K)` away from λ = 0.
            let plain =
                shaped("  CL = TVCL * exp(ETA_CL + K)\n  V = TVV * exp(tdist(ETA_V, NU))", "")
                    .unwrap();
            let same = pa == params_at(&plain, &theta_with(&plain, l), &eta);
            assert_eq!(same, l == 0.0, "λ = {l}: shape live iff λ ≠ 0");
        }
    }
}

/// Box-Cox at λ = 0 is the log-normal model **bit for bit** (`η · exprel(0)` is
/// exactly `η · 1`), the degenerate oracle the issue asks for.
#[test]
fn box_cox_at_zero_is_the_log_normal_model() {
    let shaped_m =
        shaped("  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)", "").unwrap();
    let plain = shaped("  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)", "").unwrap();
    for eta in [[0.0, 0.0, 0.0], [0.9, -0.4, 0.0], [-1.3, 0.2, 0.0]] {
        assert_eq!(
            params_at(&shaped_m, &theta_with(&shaped_m, 0.0), &eta),
            params_at(&plain, &theta_with(&plain, 0.0), &eta),
        );
    }
}

/// Empty parentheses declare the shape θ at `ShapeKind::default_theta`, named
/// `LAMBDA_<ETA>` (Box-Cox, John-Draper) or `NU_<ETA>` (t-dist).
#[test]
fn empty_parentheses_declare_the_shape_theta() {
    let m = shaped(
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        "[eta_shape]\n  ETA_CL ~ boxcox()\n  ETA_V ~ tdist()\n",
    )
    .unwrap();
    let theta = |n: &str| {
        let i = m.theta_names.iter().position(|t| t == n).unwrap_or_else(|| panic!("{n} declared"));
        (
            m.default_params.theta[i],
            m.default_params.theta_lower[i],
            m.default_params.theta_upper[i],
        )
    };
    assert_eq!(theta("LAMBDA_ETA_CL"), (0.01, -3.0, 3.0));
    assert_eq!(theta("NU_ETA_V"), (80.0, 3.0, 100.0));
    let jd = shaped(
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        "[eta_shape]\n  ETA_V ~ johndraper()\n",
    )
    .unwrap();
    let i = jd.theta_names.iter().position(|t| t == "LAMBDA_ETA_V").unwrap();
    assert_eq!(
        (
            jd.default_params.theta[i],
            jd.default_params.theta_lower[i],
            jd.default_params.theta_upper[i]
        ),
        (1.0, -3.0, 3.0)
    );
}

/// A shaped ETA is labelled `Custom` (#1714): `exp(h(η))` is not log-normal,
/// so no CV% is printed for it until the shape-aware report lands. Both
/// spellings, and the unshaped ETA beside it keeps its label.
#[test]
fn a_shaped_eta_is_labelled_custom() {
    for m in [
        shaped("  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)", "").unwrap(),
        shaped(
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
        )
        .unwrap(),
    ] {
        let t = |e: &str| m.eta_param_info.iter().find(|i| i.eta_name == e).unwrap().param_type;
        assert_eq!(t("ETA_CL"), EtaParamType::Custom);
        assert_eq!(t("ETA_V"), EtaParamType::LogNormal);
    }
}

/// Every refusal, one row each, with the phrase that names the fix. Each row
/// asserts the model parses without its one offending part, so a row cannot
/// pass on an unrelated parse error.
#[test]
fn eta_shape_refusals() {
    let ok_indiv = "  CL = TVCL * exp(ETA_CL + K)\n  V = TVV * exp(ETA_V)";
    // Each row names both halves of its message: what is wrong, and the fix.
    let cases: [(&str, &str, &str, &[&str]); 6] = [
        (
            "kappa",
            ok_indiv,
            "[eta_shape]\n  K ~ boxcox(L)\n",
            &["#1717", "shape the ETA it is summed with"],
        ),
        (
            "unknown eta",
            ok_indiv,
            "[eta_shape]\n  ETA_X ~ boxcox(L)\n",
            &["not a declared ETA", "declared: ETA_CL, ETA_V"],
        ),
        (
            "twice",
            ok_indiv,
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n  ETA_CL ~ tdist(NU)\n",
            &["shaped twice", "takes one shape"],
        ),
        (
            "unknown shape",
            ok_indiv,
            "[eta_shape]\n  ETA_CL ~ gamma(L)\n",
            &["unknown shape `gamma`", "`boxcox`, `tdist` and `johndraper`"],
        ),
        (
            "malformed",
            ok_indiv,
            "[eta_shape]\n  ETA_CL boxcox(L)\n",
            &["cannot read", "leave the parentheses empty"],
        ),
        (
            "both inline and in the block",
            "  CL = TVCL * exp(boxcox(ETA_CL, L) + K)\n  V = TVV * exp(ETA_V)",
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
            &["already shape-transformed", "not both"],
        ),
    ];
    for (label, indiv, extra, phrases) in cases {
        let err = match shaped(indiv, extra) {
            Ok(_) => panic!("{label}: parsed"),
            Err(e) => e,
        };
        for phrase in phrases {
            assert!(err.contains(phrase), "{label}: `{err}` lacks `{phrase}`");
        }
    }
    // Without its offending block every row's model parses.
    assert!(shaped(ok_indiv, "").is_ok());
    assert!(shaped("  CL = TVCL * exp(boxcox(ETA_CL, L) + K)\n  V = TVV * exp(ETA_V)", "").is_ok());
}

/// An auto-declared θ whose name is taken is refused, naming the spelling that
/// avoids it; the same block with the name given parses.
#[test]
fn an_auto_theta_name_that_is_taken_is_refused() {
    let src = |eta_shape: &str| {
        parse_model_string(&format!(
            "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta LAMBDA_ETA_CL(0.2, -3.0, 3.0)\n  \
             omega ETA_CL ~ 0.1\n  sigma EPS ~ 0.01\n\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 10\n\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
             [error_model]\n  DV ~ proportional(EPS)\n\n[eta_shape]\n  {eta_shape}\n"
        ))
    };
    let err = src("ETA_CL ~ boxcox()").err().expect("refused");
    assert!(err.contains("already declares") && err.contains("boxcox(LAMBDA_ETA_CL)"), "{err}");
    assert!(src("ETA_CL ~ boxcox(LAMBDA_ETA_CL)").is_ok());
}

/// A shaped ETA read directly in a prediction block is refused — there it
/// would be η where `[individual_parameters]` sees `h(η)`. The same model with
/// the ETA unshaped parses.
#[test]
fn a_shaped_eta_read_raw_in_a_prediction_block_is_refused() {
    let src = |eta_shape: &str| {
        parse_model_string(&format!(
            "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta L(0.3, -3.0, 3.0)\n  \
             omega ETA_CL ~ 0.1\n  omega ETA_E ~ 0.1\n  sigma EPS ~ 0.01\n\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 10\n\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
             [error_model]\n  DV ~ proportional(EPS)\n  iiv_on_ruv = ETA_E\n\n{eta_shape}"
        ))
    };
    let base = src("");
    assert!(base.is_ok(), "the unshaped model parses: {:?}", base.err());
    let err = src("[eta_shape]\n  ETA_E ~ boxcox(L)\n").err().expect("refused");
    // `iiv_on_ruv` takes an ETA by name, so the advice is its own, not the
    // individual-parameter one, which cannot be followed there.
    for phrase in ["[error_model] reads it directly", "`iiv_on_ruv` takes an unshaped ETA"] {
        assert!(err.contains(phrase), "`{err}` lacks `{phrase}`");
    }
    assert!(!err.contains("individual parameter"), "{err}");
}

/// A `[mixture]` mixing expression reads no ETA (#977), so its refusal does not
/// tell the user to read one through an individual parameter.
#[test]
fn a_shaped_eta_read_in_a_mixing_expression_is_refused_without_the_parameter_advice() {
    let err = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta MIXL(0.0, -10.0, 10.0)\n  \
         omega ETA_CL ~ 0.1\n  sigma EPS ~ 0.01\n\n\
         [mixture]\n  nsub = 2\n  logit(1) = MIXL + ETA_CL\n\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 10\n\n\
         [eta_shape]\n  ETA_CL ~ boxcox()\n\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n  DV ~ proportional(EPS)\n",
    )
    .err()
    .expect("refused");
    for phrase in ["[mixture] reads it directly", "a mixing expression cannot read an ETA"] {
        assert!(err.contains(phrase), "`{err}` lacks `{phrase}`");
    }
    assert!(!err.contains("individual parameter") && !err.contains("iiv_on_ruv"), "{err}");
}

/// The refusal reads the blocks that carry an η into the objective, and a
/// `[mixture]` class-variance line names an ETA without reading it: a shaped
/// ETA with its own class variance parses. Dies under: the mixture filter
/// dropped from the reader scan (`is_mixture_class_variance_line`).
#[test]
fn a_mixture_class_variance_on_a_shaped_eta_is_not_a_read() {
    let src = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta MIXL(0.0, -10.0, 10.0)\n  \
               omega ETA_CL ~ 0.1\n  sigma EPS ~ 0.01\n\n\
               [mixture]\n  nsub = 2\n  logit(1) = MIXL\n  omega(2) ETA_CL ~ 0.3\n\n\
               [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 10\n\n\
               [eta_shape]\n  ETA_CL ~ boxcox()\n\n\
               [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
               [error_model]\n  DV ~ proportional(EPS)\n";
    parse_model_string(src).expect("a class variance is not a read");
}

/// The inline functions take exactly two arguments.
#[test]
fn shape_functions_take_two_arguments() {
    for (line, phrase) in [
        ("CL = TVCL * exp(boxcox(ETA_CL))", "`boxcox(eta, shape)`"),
        ("CL = TVCL * exp(tdist(ETA_CL, NU, 3))", "`tdist(eta, shape)`"),
        ("CL = TVCL * exp(boxcox(boxcox(ETA_CL, L), L))", "already shape-transformed"),
    ] {
        let err = shaped(&format!("  {line}\n  V = TVV * exp(ETA_V)"), "").err().expect(line);
        assert!(err.contains(phrase), "{line}: {err}");
    }
}

/// The rewrite matches whole identifiers only: `ETA_CL` inside `ETA_CL2`,
/// `MY_ETA_CL` or `LAMBDA_ETA_CL` is not a read of `ETA_CL`.
#[test]
fn the_rewrite_matches_whole_identifiers() {
    let line = "X = ETA_CL + ETA_CL2 + MY_ETA_CL + LAMBDA_ETA_CL*(ETA_CL)";
    let spans = word_spans(line, "ETA_CL");
    let words: Vec<&str> = spans.iter().map(|&(s, e)| &line[s..e]).collect();
    assert_eq!(words, ["ETA_CL", "ETA_CL"]);
    assert_eq!(spans[0].0, 4);
    assert_eq!(spans[1].0, line.len() - "ETA_CL)".len());
}

/// A shaped ETA is not mu-referenced (`log CL = log TVCL + h(η)` has no
/// closed-form typical-value shift), so the EM estimators update its θ and the
/// shape numerically; the plain ETA beside it keeps its mu-reference. Both
/// spellings. Dies under: a mu-ref detector that looked through the shape node.
#[test]
fn a_shaped_eta_is_not_mu_referenced() {
    for m in [
        shaped("  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)", "").unwrap(),
        shaped(
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
        )
        .unwrap(),
    ] {
        assert!(!m.mu_refs.contains_key("ETA_CL"), "{:?}", m.mu_refs.keys());
        assert_eq!(m.mu_refs.get("ETA_V").map(|r| r.theta_name.as_str()), Some("TVV"));
    }
}

/// A **named** prediction block (`[event_model cause_b]`) reading a shaped ETA
/// directly is refused like an unnamed one. Named blocks are stored apart from
/// unnamed ones, so a check over the unnamed map alone would let this through.
/// The same model without `[eta_shape]` parses. Dies under: scanning only the
/// unnamed blocks.
#[cfg(feature = "survival")]
#[test]
fn a_shaped_eta_read_raw_in_a_named_block_is_refused() {
    let src = |eta_shape: &str| {
        parse_model_string(&format!(
            "[parameters]\n  theta TVLAMBDA_A(0.10, 0.001, 10.0)\n  \
             theta TVLAMBDA_B(0.06, 0.001, 10.0)\n  theta L(0.3, -3.0, 3.0)\n  \
             omega ETA_F ~ 0.25\n  sigma EPS ~ 0.01\n\n\
             [individual_parameters]\n  LA = TVLAMBDA_A * exp(ETA_F)\n  CL = 2.0 * LA\n  V = 20.0\n\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n[error_model]\n  DV ~ proportional(EPS)\n\n\
             [event_model cause_a]\n  cmt = 2\n  family = exponential\n  scale = LA\n\n\
             [event_model cause_b]\n  cmt = 3\n  family = exponential\n  \
             scale = TVLAMBDA_B * exp(ETA_F)\n\n{eta_shape}"
        ))
    };
    let base = src("");
    assert!(base.is_ok(), "the unshaped model parses: {:?}", base.err());
    let err = src("[eta_shape]\n  ETA_F ~ boxcox(L)\n").err().expect("refused");
    // The block is named as its header writes it, and a hazard can read an
    // individual parameter, so that is the advice.
    for phrase in
        ["[event_model cause_b] reads it directly", "read it through an individual parameter"]
    {
        assert!(err.contains(phrase), "`{err}` lacks `{phrase}`");
    }
    assert!(!err.contains("iiv_on_ruv"), "{err}");
}

/// With two blocks reading shaped ETAs, the refusal names the first in a fixed
/// order, whatever the hash order of the blocks: one message over repeated parses
/// (each parse builds its own `HashMap`, with its own seed). Both blocks are
/// unnamed, so they share that map. Dies under: the sort of the readers removed.
#[test]
fn the_refusal_names_the_same_block_every_time() {
    let src = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta MIXL(0.0, -10.0, 10.0)\n  \
               theta L(0.3, -3.0, 3.0)\n  omega ETA_CL ~ 0.1\n  omega ETA_E ~ 0.1\n  \
               sigma EPS ~ 0.01\n\n\
               [mixture]\n  nsub = 2\n  logit(1) = MIXL + ETA_CL\n\n\
               [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 10\n\n\
               [eta_shape]\n  ETA_CL ~ boxcox(L)\n  ETA_E ~ boxcox(L)\n\n\
               [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
               [error_model]\n  DV ~ proportional(EPS)\n  iiv_on_ruv = ETA_E\n";
    let first = parse_model_string(src).err().expect("refused");
    assert!(first.contains("[error_model]"), "{first}");
    for _ in 0..16 {
        assert_eq!(parse_model_string(src).err().expect("refused"), first);
    }
}

/// #1721 review r1, finding 1: the absorption ODE twin carries the shape.
/// Subjects with time-varying covariates route to the twin, so an unshaped twin
/// would silently predict with the unshaped CL. The twin's CL must equal the
/// primary's, and differ from the unshaped one (measured: 7.785347 vs 7.459123
/// at η_CL = 0.4, λ = 0.5).
///
/// Two mechanisms deliver the shape today, and each alone suffices (measured):
/// the desugar runs before the twin source is built, and the twin source
/// re-emits every unnamed block — a not-yet-desugared `[eta_shape]` included —
/// so the twin's own parse desugars it again. Moving the call alone, or dropping
/// `eta_shape` from the re-emit alone, kills nothing.
/// Dies under: both at once (call moved after `absorption_ode_equivalent_source`
/// **and** `"eta_shape"` added to the re-emit's skip list).
#[test]
fn the_absorption_twin_carries_the_shape() {
    let m = parse_model_string(
        "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  theta TVV(50.0, 5.0, 500.0)\n  \
         theta TVMTT(1.0, 0.05, 24.0)\n  theta TVN(3.0, 0.0, 30.0)\n  theta L(0.5, -3.0, 3.0)\n  \
         omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.15 (sd)\n\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
         MTT = TVMTT\n  NTR = TVN\n\n\
         [eta_shape]\n  ETA_CL ~ boxcox(L)\n\n\
         [structural_model]\n  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n",
    )
    .expect("transit model parses");
    let twin = m
        .absorption_ode_equivalent
        .as_ref()
        .expect("a plain transit model carries its ODE twin")
        .built();
    assert_eq!(twin.theta_names, m.theta_names, "same θ layout, L included");
    let th = m.default_params.theta.clone();
    let eta = [0.4, -0.2];
    let none = std::collections::HashMap::new();
    let cl = |model: &CompiledModel| (model.pk_param_fn)(&th, &eta, &none, 0.0).values[0];
    let unshaped = 5.0 * 0.4f64.exp();
    assert_eq!(cl(twin), cl(&m), "twin CL == primary CL");
    assert!((cl(&m) - unshaped).abs() > 0.1, "the shape is live: {} vs {unshaped}", cl(&m));
}

/// #1721 review r1, finding 1: `[eta_shape]` composes with `[covariate_model]`,
/// giving the same CL as the model written out inline,
/// `TVCL * (WT/70)^THETA_CL_WT * exp(boxcox(ETA_CL, L))` — and not the unshaped
/// covariate model. The order of the two desugars does not matter (measured:
/// running `[eta_shape]` first passes too), so this pins the composition, not an
/// order.
/// Dies under: the rewrite's argument order swapped.
#[test]
fn eta_shape_composes_with_the_covariate_model() {
    let src = |indiv: &str, covmodel: &str, eta_shape: &str| {
        parse_model_string(&format!(
            "[parameters]\n  theta TVCL(2.0, 0.01, 100.0)\n  theta TVV(20.0, 0.1, 1000.0)\n  \
             theta L(0.5, -3.0, 3.0)\n{}  omega ETA_CL ~ 0.2\n  omega ETA_V ~ 0.1\n  \
             sigma EPS ~ 0.01\n\n[covariates]\n  WT continuous\n\n\
             [individual_parameters]\n{indiv}\n  V = TVV * exp(ETA_V)\n\n{covmodel}{eta_shape}\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n[error_model]\n  \
             DV ~ proportional(EPS)\n",
            if covmodel.is_empty() { "  theta THETA_CL_WT(0.75, 0.01, 5.0)\n" } else { "" }
        ))
        .unwrap_or_else(|e| panic!("parse: {e}"))
    };
    let covmodel =
        "[covariate_model]\n  CL ~ WT power(center = 70) => THETA_CL_WT(0.75, 0.01, 5.0)\n\n";
    let block = src("  CL = TVCL * exp(ETA_CL)", covmodel, "[eta_shape]\n  ETA_CL ~ boxcox(L)\n\n");
    let unshaped = src("  CL = TVCL * exp(ETA_CL)", covmodel, "");
    let inline = src("  CL = TVCL * (WT / 70)^THETA_CL_WT * exp(boxcox(ETA_CL, L))", "", "");
    let cov = std::collections::HashMap::from([("WT".to_string(), 85.0)]);
    let cl = |m: &CompiledModel, eta: &[f64]| {
        let mut th = vec![0.0; m.theta_names.len()];
        for (i, n) in m.theta_names.iter().enumerate() {
            th[i] = match n.as_str() {
                "TVCL" => 2.0,
                "TVV" => 20.0,
                "L" => 0.5,
                "THETA_CL_WT" => 0.75,
                other => panic!("unexpected θ {other}"),
            };
        }
        (m.pk_param_fn)(&th, eta, &cov, 0.0).values[0]
    };
    for eta in [[0.4, -0.2], [-0.6, 0.3]] {
        let (b, i, u) = (cl(&block, &eta), cl(&inline, &eta), cl(&unshaped, &eta));
        assert!((b - i).abs() <= 1e-14 * i.abs(), "η {eta:?}: block {b} vs inline {i}");
        assert!((b - u).abs() > 1e-3, "η {eta:?}: the shape is live ({b} vs unshaped {u})");
    }
}

/// Past `exp`'s overflow Box-Cox is `+inf`, the limit, not `NaN` (#1721 review r1,
/// finding 5). A `NaN`
/// would poison an objective that an `inf` lets a line search back away from.
/// Dies under: Box-Cox as `expm1(x)` with its value and jets split (`inf − inf`).
#[test]
fn box_cox_overflows_to_infinity_not_nan() {
    let v = shape_g(ShapeKind::BoxCox, 800.0, 1.0);
    assert!(v.is_infinite() && v > 0.0, "boxcox(800, 1) = {v}");
    let d = shape_g(ShapeKind::BoxCox, Dual2::<2>::var(800.0, 0), Dual2::<2>::var(1.0, 1));
    assert!(d.value.is_infinite() && d.value > 0.0, "dual value {}", d.value);
    // Just below the overflow it is still finite and exact.
    let x: f64 = 700.0;
    assert_eq!(shape_g(ShapeKind::BoxCox, x, 1.0), x.exp_m1());
}

// ── #1716: `density` / `transform` on the omega line ────────────────────────

/// A 1-cpt IV model with θs `TVCL`, `TVV`, `L`, `NU`, `MU`, the given omega lines, the
/// given `[individual_parameters]` and `extra` blocks.
fn densed(omegas: &str, indiv: &str, extra: &str) -> Result<CompiledModel, String> {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         theta L(0.3, -3.0, 3.0)\n  theta NU(10.0, 3.0, 100.0)\n  theta MU(0.4, 0.01, 0.99)\n\
         {omegas}\n  sigma EPS ~ 0.01\n\n\
         [individual_parameters]\n{indiv}\n\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n  DV ~ proportional(EPS)\n\n{extra}"
    ))
}

/// The model reads a constrained density's random effect through its domain map: under
/// `density gamma`, `CL = TVCL * ETA_CL` is `TVCL · exp(z)` at latent `z`; under
/// `density beta(MU)` a fraction read as `ETA_V` is `inv_logit(z)`; a `t` reads `z`.
/// Bit-identical to the hand-written mapped model at several latents.
/// Dies under: the AST rewrite skipped (CL = TVCL · z), the wrong map for a kind, or a
/// map applied to an unconstrained kind.
#[test]
fn a_density_reads_the_random_effect_through_its_domain_map() {
    let dens = densed(
        "  omega ETA_CL ~ 0.1 density gamma\n  omega ETA_V ~ 0.02 density beta(MU)",
        "  CL = TVCL * ETA_CL\n  V = TVV * (0.5 + ETA_V)",
        "",
    )
    .unwrap();
    let hand = densed(
        "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.02",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * (0.5 + inv_logit(ETA_V))",
        "",
    )
    .unwrap();
    let t = densed(
        "  omega ETA_CL ~ 0.1 density t(NU)\n  omega ETA_V ~ 0.02",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    let t_hand = densed(
        "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.02",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    let th = dens.default_params.theta.clone();
    for z in [[0.3, -0.8], [-1.1, 1.4], [0.0, 0.0]] {
        assert_eq!(params_at(&dens, &th, &z), params_at(&hand, &th, &z), "z = {z:?}");
        assert_eq!(params_at(&t, &th, &z), params_at(&t_hand, &th, &z), "t, z = {z:?}");
    }
    use crate::stats::eta_density::DensityKind;
    let kinds: Vec<_> = dens.eta_densities.iter().map(|d| (d.kind, d.etas.clone())).collect();
    assert_eq!(kinds, vec![(DensityKind::Gamma, vec![0]), (DensityKind::Beta, vec![1])]);
    assert_eq!(dens.eta_densities[1].shape_thetas, vec![4], "μ is θ `MU`");
}

/// A density ETA is not mu-referenced — `TVCL * exp(ETA)` after the gamma's map would
/// otherwise read as a Gaussian location shift — and is labelled `custom`, so no CV% is
/// printed from a variance that is not on a log-normal scale.
/// Dies under: the mu-ref filter or the label override removed.
#[test]
fn a_density_eta_is_neither_mu_referenced_nor_log_normal() {
    let m = densed(
        "  omega ETA_CL ~ 0.1 density gamma\n  omega ETA_V ~ 0.1",
        "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    assert!(!m.mu_refs.contains_key("ETA_CL"), "{:?}", m.mu_refs.keys());
    assert!(m.mu_refs.contains_key("ETA_V"), "the Gaussian ETA keeps its mu-ref");
    let label = |n: &str| m.eta_param_info.iter().find(|i| i.eta_name == n).map(|i| i.param_type);
    assert_eq!(label("ETA_CL"), Some(EtaParamType::Custom));
    assert_eq!(label("ETA_V"), Some(EtaParamType::LogNormal));
}

/// `density t()` / `beta()` declare their shape θ (`NU_<ETA>`, `MU_<ETA>`) at the
/// documented defaults, and the clause is rewritten to name it, so a re-parse of the
/// text (the absorption twin's) reads the same model.
#[test]
fn empty_parentheses_declare_the_density_theta() {
    let m = densed(
        "  omega ETA_CL ~ 0.1 density t()\n  omega ETA_V ~ 0.02 density beta()",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * (0.5 + ETA_V)",
        "",
    )
    .unwrap();
    let at = |n: &str| m.theta_names.iter().position(|t| t == n).expect(n);
    let (nu, mu) = (at("NU_ETA_CL"), at("MU_ETA_V"));
    assert_eq!(m.default_params.theta[nu], 80.0);
    assert_eq!(m.default_params.theta_lower[nu], 3.0);
    assert_eq!(m.default_params.theta[mu], 0.5);
    assert_eq!(m.eta_densities[0].shape_thetas, vec![nu]);
    assert_eq!(m.eta_densities[1].shape_thetas, vec![mu]);
}

/// A shape transform has one declarative home, `[eta_shape]` (#1721 review): the
/// omega-line `transform KIND(…)` and the `density boxcox(λ)` alias are refused with the
/// `[eta_shape]` spelling to write instead, which itself still parses.
/// Dies under: either omega-line spelling accepted again, or a refusal that does not name
/// the `[eta_shape]` line.
#[test]
fn a_transform_is_declared_in_eta_shape_only() {
    let indiv = "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)";
    assert!(densed(
        "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1",
        indiv,
        "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
    )
    .is_ok());
    for spelling in ["transform boxcox(L)", "density boxcox(L)"] {
        let e = densed(&format!("  omega ETA_CL ~ 0.1 {spelling}\n  omega ETA_V ~ 0.1"), indiv, "")
            .unwrap_err();
        assert!(
            e.contains("declared in [eta_shape]") && e.contains("`ETA_CL ~ boxcox(L)`"),
            "{spelling}: {e}"
        );
    }
}

/// Each refusal names its cause: an omega-line transform on a kappa or on a block (both
/// pointed at [eta_shape]), a
/// non-t kind on a block, an unknown kind, a wrong shape-θ count, a t whose ν may reach 0,
/// a μ bounded outside [0, 1], a clause on a θ, trailing text after the clause, a mapped
/// ETA read raw outside `[individual_parameters]`, and a kind without a free spread whose
/// ω is not `FIX`.
/// Dies under: any one refusal removed (each asserts its own message).
#[test]
fn omega_line_distribution_refusals() {
    // The direct read, so a positive kind passes the second-transform check (#2669) and
    // reaches the refusal under test.
    let indiv = "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)";
    let err = |omegas: &str, extra: &str| densed(omegas, indiv, extra).unwrap_err();
    let cases: [(&str, &str, &str); 10] = [
        (
            "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  kappa K ~ 0.04 transform boxcox(L)",
            "",
            "declared in [eta_shape]",
        ),
        (
            "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.1] transform boxcox(L)",
            "",
            "declared in [eta_shape]",
        ),
        ("  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.1] density gamma", "", "univariate"),
        ("  omega ETA_CL ~ 0.1 density banana\n  omega ETA_V ~ 0.1", "", "unknown density"),
        ("  omega ETA_CL ~ 0.1 density gamma(L)\n  omega ETA_V ~ 0.1", "", "no shape θ"),
        ("  omega ETA_CL ~ 0.1 density t(L)\n  omega ETA_V ~ 0.1", "", "bounded inside (0, inf)"),
        ("  omega ETA_CL ~ 0.1 density beta(NU)\n  omega ETA_V ~ 0.1", "", "inside [0, 1]"),
        ("  omega ETA_CL ~ 0.1 density t(NU) banana\n  omega ETA_V ~ 0.1", "", "must end"),
        (
            "  omega ETA_CL ~ 0.1 density gamma\n  omega ETA_V ~ 0.1",
            "[derived]\n  RAW = ETA_CL\n",
            "unconstrained latent",
        ),
        ("  omega ETA_CL ~ 1 density exponential\n  omega ETA_V ~ 0.1", "", "declare it `FIX`"),
    ];
    for (omegas, extra, says) in cases {
        let e = err(omegas, extra);
        assert!(e.contains(says), "`{omegas}` should say `{says}`: {e}");
    }
}

/// The omega-line forms beyond a single kind: `truncated(…)` reads the effect through its
/// bounded map, a `block_omega … density copula(…)` gives each ETA its own margin (and
/// `beta()` declares its μ), `lognormal_mean` on a block is MvLogNormal, a κ takes a
/// `density` read through its map, and `density normal` is the plain Gaussian.
/// Dies under: any of the forms resolved to the wrong kind, map or shape layout.
#[test]
fn the_omega_line_forms_resolve() {
    use crate::stats::eta_density::{Bound, DensityKind};
    let tr = densed(
        "  omega ETA_CL ~ 0.1 density truncated(gamma, 0.5, 2)\n  omega ETA_V ~ 0.1 density normal",
        "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    assert_eq!(tr.eta_densities.len(), 1, "`density normal` makes no group");
    let t = tr.eta_densities[0].trunc.expect("truncated");
    assert_eq!((t.lo, t.hi), (Bound::Value(0.5), Bound::Value(2.0)));
    let hand = densed(
        "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1",
        "  CL = TVCL * (0.5 + (2 - 0.5) * inv_logit(ETA_CL))\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    let th = tr.default_params.theta.clone();
    for z in [[0.3, -0.2], [-1.4, 0.6]] {
        assert_eq!(params_at(&tr, &th, &z), params_at(&hand, &th, &z), "truncated map");
    }
    let cop = densed(
        "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02] density copula(ETA_CL = gamma, ETA_V = beta())",
        "  CL = TVCL * ETA_CL\n  V = TVV * (0.5 + ETA_V)",
        "",
    )
    .unwrap();
    let g = &cop.eta_densities[0];
    assert_eq!(g.kind, DensityKind::Copula);
    assert_eq!(g.margins, vec![DensityKind::Gamma, DensityKind::Beta]);
    assert_eq!(cop.theta_names[g.shape_thetas[0]], "MU_ETA_V");
    let cop_hand = densed(
        "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02]\n  theta MU_ETA_V(0.5, 0.001, 0.999)",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * (0.5 + inv_logit(ETA_V))",
        "",
    )
    .unwrap();
    let th = cop.default_params.theta.clone();
    assert_eq!(params_at(&cop, &th, &[0.2, -0.4]), params_at(&cop_hand, &th, &[0.2, -0.4]));
    let mvln = densed(
        "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02] density lognormal_mean",
        "  CL = TVCL * ETA_CL\n  V = TVV * ETA_V",
        "",
    )
    .unwrap();
    assert_eq!(mvln.eta_densities[0].kind, DensityKind::LogNormalMean);
    assert_eq!(mvln.eta_densities[0].etas, vec![0, 1]);
    // A κ density: the model reads `exp(K)` where it wrote `K`.
    let kmodel = |kappa_line: &str, cl: &str| {
        parse_model_string(&format!(
            "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
             omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  {kappa_line}\n  sigma EPS ~ 0.01\n\n\
             [individual_parameters]\n  CL = {cl}\n  V = TVV * exp(ETA_V)\n\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
             [error_model]\n  DV ~ proportional(EPS)\n\n[fit_options]\n  iov_column = OCC\n"
        ))
        .unwrap()
    };
    let kd = kmodel("kappa K ~ 0.04 density gamma", "TVCL * exp(ETA_CL) * K");
    let kh = kmodel("kappa K ~ 0.04", "TVCL * exp(ETA_CL) * exp(K)");
    assert_eq!(kd.kappa_densities.len(), 1);
    assert_eq!(kd.kappa_densities[0].kind, DensityKind::Gamma);
    assert!(kd.eta_densities.is_empty());
    let th = kd.default_params.theta.clone();
    assert_eq!(params_at(&kd, &th, &[0.1, -0.2, 0.3]), params_at(&kh, &th, &[0.1, -0.2, 0.3]));
}

/// What each non-Gaussian ETA follows reaches `eta_param_info` for reporting (#1716): a
/// density's kind and shape θ names, each copula margin's own kind and shapes, a
/// truncation's bounds, and a transform's kind, with `None` on a Gaussian ETA.
/// Dies under: the metadata left unset, a copula ETA given the group's kind or another
/// margin's shapes, the truncation dropped, or a transform reported as a density.
#[test]
fn eta_param_info_carries_each_distribution() {
    let dist = |m: &CompiledModel, eta: &str| {
        m.eta_param_info
            .iter()
            .find(|i| i.eta_name == eta)
            .and_then(|i| i.distribution.clone())
            .map(|d| (d.form, d.kind, d.shape_thetas, d.truncation))
    };
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let m = densed(
        "  omega ETA_CL ~ 0.1 density truncated(t(NU), -1, inf)\n  omega ETA_V ~ 0.1",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        "[eta_shape]\n  ETA_V ~ boxcox(L)\n",
    )
    .unwrap();
    assert_eq!(
        dist(&m, "ETA_CL"),
        Some(("density".into(), "t".into(), s(&["NU"]), Some((Some(-1.0), None))))
    );
    assert_eq!(dist(&m, "ETA_V"), Some(("transform".into(), "boxcox".into(), s(&["L"]), None)));
    let cop = densed(
        "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02] density copula(ETA_V = beta(MU), ETA_CL = t(NU))",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * (0.5 + ETA_V)",
        "",
    )
    .unwrap();
    assert_eq!(dist(&cop, "ETA_CL"), Some(("density".into(), "t".into(), s(&["NU"]), None)));
    assert_eq!(dist(&cop, "ETA_V"), Some(("density".into(), "beta".into(), s(&["MU"]), None)));
    let gauss = densed(
        "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    assert_eq!(dist(&gauss, "ETA_CL"), None);
}

/// A `[mixture]` class's `omega(2) ETA_CL ~ v` override names a mapped density ETA's
/// variance, not its value, so it parses; a mixing-probability expression that reads the
/// ETA is still a read of the latent and is refused.
/// Dies under: the override line counted as a read, or the exemption widened to every
/// `[mixture]` line.
#[test]
fn a_mixture_override_of_a_density_eta_is_no_read() {
    let indiv =
        "  CL = if (MIXNUM == 1) TVCL * ETA_CL else 2 * TVCL * ETA_CL\n  V = TVV * exp(ETA_V)";
    let omegas = "  omega ETA_CL ~ 0.1 density gamma\n  omega ETA_V ~ 0.1";
    let m =
        densed(omegas, indiv, "[mixture]\n  nsub = 2\n  logit(1) = 0\n  omega(2) ETA_CL ~ 0.2\n")
            .expect("an override parses");
    assert_eq!(m.eta_densities.len(), 1);
    let e = densed(omegas, indiv, "[mixture]\n  nsub = 2\n  logit(1) = ETA_CL\n").unwrap_err();
    assert!(e.contains("[mixture] reads `ETA_CL`"), "{e}");
}

/// Copula margins are named (#1721 review), so reordering the clause cannot re-bind them:
/// the same margins written in the other order resolve to the same groups, and a
/// positional margin, a name outside the block, a name given twice and an ETA left
/// without a margin are each refused with their own message.
/// Dies under: margins bound by position, or any of the four refusals removed.
#[test]
fn copula_margins_are_named() {
    use crate::stats::eta_density::DensityKind;
    let indiv = "  CL = TVCL * ETA_CL\n  V = TVV * (0.5 + ETA_V)";
    let block = |clause: &str| {
        densed(
            &format!("  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02] density copula({clause})"),
            indiv,
            "",
        )
    };
    let a = block("ETA_CL = gamma, ETA_V = beta(MU)").unwrap();
    let b = block("ETA_V = beta(MU), ETA_CL = gamma").unwrap();
    assert_eq!(a.eta_densities[0].margins, b.eta_densities[0].margins);
    assert_eq!(a.eta_densities[0].margins, vec![DensityKind::Gamma, DensityKind::Beta]);
    assert_eq!(a.eta_densities[0].shape_thetas, b.eta_densities[0].shape_thetas);
    for (clause, want) in [
        ("gamma, beta(MU)", "copula margins are named"),
        ("ETA_CL = gamma, ETA_KA = beta(MU)", "not an ETA of this block"),
        ("ETA_CL = gamma, ETA_CL = beta(MU)", "names `ETA_CL` twice"),
        ("ETA_CL = gamma", "gives `ETA_V` no margin"),
    ] {
        let e = block(clause).unwrap_err();
        assert!(e.contains(want), "{clause}: {e}");
    }
}

/// #1731: an `[eta_shape]` `tdist` shape θ whose bounds let ν fall below 3 is refused at
/// parse time, naming the bound and the fix; both sides of the gate (3 and 2.9), an
/// unbounded θ, and a FIXed θ judged by its value.
/// Dies under: the check removed, the bound moved, or a FIXed value ignored.
#[test]
fn a_tdist_shape_theta_must_keep_nu_at_least_3() {
    let model = |nu_line: &str| {
        parse_model_string(&format!(
            "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
             {nu_line}\n  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  sigma EPS ~ 0.01\n\n\
             [eta_shape]\n  ETA_V ~ tdist(NU)\n\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
             [error_model]\n  DV ~ proportional(EPS)\n"
        ))
    };
    assert!(model("theta NU(10, 3, 100)").is_ok());
    assert!(model("theta NU(4) FIX").is_ok());
    for (line, reach) in
        [("theta NU(10, 2.9, 100)", "2.9"), ("theta NU(10)", "-inf"), ("theta NU(2) FIX", "2")]
    {
        let e = model(line).unwrap_err();
        assert!(
            e.contains("needs ν ≥ 3") && e.contains(&format!("may reach {reach}")),
            "{line}: {e}"
        );
    }
}

/// `density lognormal` is median 1, `ln η ~ N(0, Ω)` (#2669 review): the classic
/// `TV·exp(ETA)` random effect. The parser compiles it as exactly that Gaussian ETA, so no
/// density group remains, the log mu-reference is kept, and the individual parameters and
/// the objective's prior are those of the classic spelling. Only the reported distribution
/// says `lognormal`. On a block it is the Gaussian block under `exp`.
/// Dies under: the median kind left as a density group, the mu-reference dropped, or the
/// `exp` map not applied to its reads.
#[test]
fn the_median_lognormal_is_the_classic_exp_eta() {
    for (omegas, classic_omegas, indiv, classic_indiv) in [
        (
            "  omega ETA_CL ~ 0.09 density lognormal\n  omega ETA_V ~ 0.04",
            "  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04",
            "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)",
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        ),
        (
            "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02] density lognormal",
            "  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.02]",
            "  CL = TVCL * ETA_CL\n  V = TVV * ETA_V",
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
        ),
    ] {
        let ln = densed(omegas, indiv, "").unwrap();
        let classic = densed(classic_omegas, classic_indiv, "").unwrap();
        assert!(ln.eta_densities.is_empty(), "{omegas}: {:?}", ln.eta_densities);
        let th = ln.default_params.theta.clone();
        for eta in [[0.0, 0.0], [0.3, -0.2], [-1.1, 0.7]] {
            assert_eq!(params_at(&ln, &th, &eta), params_at(&classic, &th, &eta), "{omegas}");
        }
        let mut refs: Vec<_> = ln.mu_refs.keys().cloned().collect();
        refs.sort();
        let mut classic_refs: Vec<_> = classic.mu_refs.keys().cloned().collect();
        classic_refs.sort();
        assert_eq!(refs, classic_refs, "{omegas}");
        let dist = ln
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_CL")
            .and_then(|i| i.distribution.as_ref())
            .expect("the declared distribution is reported");
        assert_eq!((dist.form.as_str(), dist.kind.as_str()), ("density", "lognormal"));
    }
}

/// `(sd)` / `(var)` read the same on either side of a `density` clause (#2669 review,
/// finding 13): `0.3 (sd)` is ω = 0.09 for a bare kind, a kind with a shape θ and a κ, and
/// `(var)` keeps the number. `density t (sd)`, where the parentheses follow a kind that
/// takes a shape, stays the shape list. `levy`, whose ω is its scale, refuses `(sd)`.
/// Dies under: the trailing word read as a shape θ, dropped, or moved for a shape kind.
#[test]
fn sd_and_var_read_the_same_on_either_side_of_the_density_clause() {
    let w = |omegas: &str| {
        densed(omegas, "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)", "")
            .unwrap_or_else(|e| panic!("{omegas}: {e}"))
            .default_params
            .omega
            .matrix[(0, 0)]
    };
    for (line, want) in [
        ("  omega ETA_CL ~ 0.3 (sd) density gamma", 0.09),
        ("  omega ETA_CL ~ 0.3 density gamma (sd)", 0.09),
        ("  omega ETA_CL ~ 0.3 density gamma ( SD )", 0.09),
        ("  omega ETA_CL ~ 0.09 density gamma (var)", 0.09),
        ("  omega ETA_CL ~ 0.3 density truncated(gamma, 0.2, 5) (sd)", 0.09),
    ] {
        let got = w(&format!("{line}\n  omega ETA_V ~ 0.04"));
        assert!((got - want).abs() < 1e-15, "{line}: ω = {got}");
    }
    // A shape kind on ℝ: `CL = TVCL * exp(ETA_CL)`.
    let t = |omegas: &str| {
        densed(omegas, "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)", "")
            .map(|m| m.default_params.omega.matrix[(0, 0)])
    };
    for line in [
        "  omega ETA_CL ~ 0.3 (sd) density t(NU)\n  omega ETA_V ~ 0.04",
        "  omega ETA_CL ~ 0.3 density t(NU) (sd)\n  omega ETA_V ~ 0.04",
    ] {
        let got = t(line).unwrap_or_else(|e| panic!("{line}: {e}"));
        assert!((got - 0.09).abs() < 1e-15, "{line}: ω = {got}");
    }
    // `t (sd)`: the parentheses are t's shape list, so `sd` must be a declared θ.
    let e = t("  omega ETA_CL ~ 0.3 density t (sd)\n  omega ETA_V ~ 0.04").unwrap_err();
    assert!(e.contains("`sd`") && e.contains("not a declared θ"), "{e}");
    for line in [
        "  omega ETA_CL ~ 0.3 (sd) density levy\n  omega ETA_V ~ 0.04",
        "  omega ETA_CL ~ 0.3 density levy (sd)\n  omega ETA_V ~ 0.04",
    ] {
        let e = densed(line, "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)", "").unwrap_err();
        assert!(e.contains("density levy") && e.contains("`(sd)` does not apply"), "{e}");
    }
    // A κ line takes the same order.
    let k = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         omega ETA_CL ~ 0.09\n  kappa K_CL ~ 0.2 density gamma (sd)\n  sigma EPS ~ 0.01\n\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL) * K_CL\n  V = TVV\n\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n  DV ~ proportional(EPS)\n",
    )
    .unwrap();
    let kw = k.default_params.omega_iov.as_ref().unwrap().matrix[(0, 0)];
    assert!((kw - 0.04).abs() < 1e-15, "κ ω = {kw}");
}

/// A positive or (0, 1) `density` ETA read under `exp`, `log`, `inv_logit` or `logit` is
/// refused (#2669 review, finding 2): it already is the effect, so `TVCL * exp(ETA_CL)`
/// under `density gamma` would fit `TVCL·exp(Gamma)`. The message names the line to write.
/// A kind on ℝ keeps `exp`, and a positive ETA added to a term is a `W_DENSITY_ADDITIVE`
/// warning, not an error.
/// Dies under: the check removed, applied to an ℝ kind, or blind to a nested read.
#[test]
fn a_bounded_density_eta_under_a_second_transform_is_refused() {
    for (omegas, indiv, want) in [
        ("  omega ETA_CL ~ 0.09 density gamma", "CL = TVCL * exp(ETA_CL)", "CL = TVCL * ETA_CL"),
        (
            "  omega ETA_CL ~ 0.09 density lognormal",
            "CL = TVCL * exp(ETA_CL)",
            "CL = TVCL * ETA_CL",
        ),
        (
            "  omega ETA_CL ~ 0.09 density gamma",
            "CL = TVCL * exp(0.1 + 2 * ETA_CL)",
            "CL = TVCL * ETA_CL",
        ),
        (
            "  omega ETA_CL ~ 0.09 density gamma",
            "CL = TVCL * exp(log(ETA_CL))",
            "CL = TVCL * ETA_CL",
        ),
        (
            "  omega ETA_CL ~ 0.09 density truncated(gamma, 0.2, 5)",
            "CL = TVCL * exp(ETA_CL)",
            "CL = TVCL * ETA_CL",
        ),
        ("  omega ETA_CL ~ 0.02 density beta(MU)", "CL = TVCL * inv_logit(ETA_CL)", "CL = ETA_CL"),
    ] {
        let e = densed(
            &format!("{omegas}\n  omega ETA_V ~ 0.04"),
            &format!("  {indiv}\n  V = TVV * exp(ETA_V)"),
            "",
        )
        .expect_err(omegas);
        assert!(
            e.contains("`ETA_CL` follows `density") && e.contains(want),
            "{omegas} / {indiv}: {e}"
        );
    }
    // ℝ kinds keep `exp`.
    for omegas in ["  omega ETA_CL ~ 0.09 density t(NU)", "  omega ETA_CL ~ 0.09 density logistic"]
    {
        densed(
            &format!("{omegas}\n  omega ETA_V ~ 0.04"),
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
            "",
        )
        .unwrap_or_else(|e| panic!("{omegas}: {e}"));
    }
    // Additive: a warning naming the proportional read.
    let m = densed(
        "  omega ETA_CL ~ 0.09 density gamma\n  omega ETA_V ~ 0.04",
        "  CL = TVCL + ETA_CL\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    let w: Vec<_> = m.parse_warnings.iter().filter(|w| w.contains("W_DENSITY_ADDITIVE")).collect();
    assert_eq!(w.len(), 1, "{:?}", m.parse_warnings);
    assert!(w[0].contains("CL = TVCL * ETA_CL"), "{}", w[0]);
    // The proportional read warns about nothing.
    let m = densed(
        "  omega ETA_CL ~ 0.09 density gamma\n  omega ETA_V ~ 0.04",
        "  CL = TVCL * ETA_CL\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
    assert!(!m.parse_warnings.iter().any(|w| w.contains("W_DENSITY")), "{:?}", m.parse_warnings);
}

/// An ETA with both a `density` and an `[eta_shape]` transform (or an inline one) is
/// refused, naming both (#2669 review, finding 3): the pair would fit `boxcox(exp(z), λ)`,
/// which neither line describes, and the report would show only the density.
/// Dies under: the pair accepted, in either spelling.
#[test]
fn a_density_eta_with_a_shape_transform_is_refused() {
    let omegas = "  omega ETA_CL ~ 0.09 density t(NU)\n  omega ETA_V ~ 0.04";
    for (indiv, extra) in [
        (
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
        ),
        ("  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)", ""),
    ] {
        let e = densed(omegas, indiv, extra).expect_err(indiv);
        assert!(e.contains("`ETA_CL` follows `density t`") && e.contains("boxcox"), "{indiv}: {e}");
    }
}

/// The examples on the random-effect distributions page parse as written (#2669 review,
/// finding 10: the intro example read an undeclared `TVV`), with a structural model and
/// an error model added.
#[test]
fn the_distribution_page_examples_parse() {
    let tail = "\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
                [error_model]\n  DV ~ proportional(EPS)\n";
    let intro = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
                 theta TVF(0.6, 0.01, 0.99)\n  theta NU(10.0, 2.5, 100.0)\n  \
                 omega ETA_CL ~ 0.09 density gamma\n  omega ETA_F  ~ 0.02 density beta(TVF)\n  \
                 omega ETA_V  ~ 0.06 density t(NU)\n  theta TVKA(1.0, 0.01, 10.0)\n  sigma EPS ~ 0.01\n\n\
                 [individual_parameters]\n  CL = TVCL * ETA_CL\n  F  = ETA_F\n  \
                 V  = TVV * exp(ETA_V)\n  KA = TVKA\n";
    parse_model_string(&format!("{intro}{tail}")).unwrap_or_else(|e| panic!("intro: {e}"));
    let mixture = "[parameters]\n  theta TVCL1(2.0, 0.1, 100.0)\n  theta TVCL2(8.0, 0.1, 100.0)\n  \
                   theta MIXL(0.4, -5.0, 5.0)\n  omega ETA_CL ~ 0.09 density gamma\n  \
                   theta TVV(30.0, 1.0, 500.0)\n  theta TVKA(1.0, 0.01, 10.0)\n  sigma EPS ~ 0.01\n\n\
                   [individual_parameters]\n  CL = if (MIXNUM == 1) TVCL1 * ETA_CL else TVCL2 * ETA_CL\n  \
                   V = TVV\n  KA = TVKA\n\n\
                   [mixture]\n  nsub = 2\n  logit(1) = MIXL\n  omega(2) ETA_CL ~ 0.04\n";
    let m =
        parse_model_string(&format!("{mixture}{tail}")).unwrap_or_else(|e| panic!("mixture: {e}"));
    assert_eq!(m.eta_densities.len(), 1);
}
