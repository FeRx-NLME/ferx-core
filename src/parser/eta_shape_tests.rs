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
/// `0/0` (evaluated as 0, #1714): the series gives `η`, and on both sides of
/// the series threshold it agrees with libm's `expm1`. Measured worst relative
/// error over the grid below: under 2 ulp; bound 4 ulp.
/// Dies under: removing the series arm (λ = 0 → NaN), or a wrong series term.
#[test]
fn box_cox_is_exact_through_lambda_zero() {
    assert_eq!(shape_g(ShapeKind::BoxCox, 0.3, 0.0), 0.3);
    assert_eq!(shape_g(ShapeKind::BoxCox, -1.7, 0.0), -1.7);
    let mut worst = 0.0f64;
    for eta in [-2.0, -0.3, 0.05, 0.4, 1.5] {
        for x in [1e-9, 1e-6, 1e-3, 0.05, 0.0999, 0.1001, 0.2, 1.0] {
            for sign in [-1.0, 1.0] {
                let lambda: f64 = sign * x / eta;
                let want = (lambda * eta).exp_m1() / lambda;
                let got = shape_g(ShapeKind::BoxCox, eta, lambda);
                assert!(got.is_finite(), "boxcox({eta}, {lambda}) = {got}");
                worst = worst.max(((got - want) / want).abs());
            }
        }
    }
    eprintln!("box-cox worst relative error vs exp_m1: {worst:e}");
    assert!(
        worst <= 4.0 * f64::EPSILON,
        "worst relative error {worst:e}"
    );
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
/// and Hessian in (η, shape), on both sides of 0 and of the Box-Cox series
/// threshold. AGENTS.md: every analytic sensitivity needs this parity.
/// Dies under: any kernel step that is not differentiable over `PkNum` (a
/// value-only branch, a `from_f64(x.val())` lift).
#[test]
fn shape_g_dual_matches_fd() {
    use ShapeKind::*;
    let points = [
        (BoxCox, 0.7, 0.5),
        (BoxCox, -1.1, -0.8),
        (BoxCox, 0.4, 0.002),  // x = 8e-4: series arm
        (BoxCox, 0.4, 0.2495), // x = 0.0998: series arm, at the threshold
        (BoxCox, 0.4, 0.2505), // x = 0.1002: closed form, at the threshold
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
        assert!(
            close(d.hess[0][0], hxx, 1e-5),
            "{} hess ηη {} vs {hxx}",
            k.name(),
            d.hess[0][0]
        );
        assert!(
            close(d.hess[1][1], hyy, 1e-5),
            "{} hess ss {} vs {hyy}",
            k.name(),
            d.hess[1][1]
        );
        assert!(
            close(d.hess[0][1], hxy, 1e-5),
            "{} hess ηs {} vs {hxy}",
            k.name(),
            d.hess[0][1]
        );
        assert_eq!(d.hess[0][1], d.hess[1][0], "{} hess symmetry", k.name());
    }
}

/// John-Draper at η = 0 — the inner optimizer's warm start — has `∂h/∂η = 1`,
/// not 0. Dies under: writing the odd extension as `sign(η) · h(|η|)`, whose
/// dual `sign` is flat.
#[test]
fn john_draper_slope_at_zero_is_one() {
    for s in [0.0, 0.5, 1.0, 2.5] {
        let d = shape_g(
            ShapeKind::JohnDraper,
            Dual2::<2>::var(0.0, 0),
            Dual2::<2>::var(s, 1),
        );
        assert!(
            (d.grad[0] - 1.0).abs() < 1e-15,
            "λ = {s}: ∂h/∂η = {}",
            d.grad[0]
        );
    }
}

/// The symbolic partial nodes evaluate to the kernel's own `Dual1` gradient,
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
        assert!(
            (shape_f64(k, ShapeOut::DEta, eta, s) - fe).abs() < 1e-8,
            "{} ∂η",
            k.name()
        );
        assert!(
            (shape_f64(k, ShapeOut::DShape, eta, s) - fs).abs() < 1e-8,
            "{} ∂s",
            k.name()
        );
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
    let inline = shaped(
        "  CL = TVCL * exp(boxcox(ETA_CL, L) + K)\n  V = TVV * exp(tdist(ETA_V, NU))",
        "",
    )
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
            let plain = shaped(
                "  CL = TVCL * exp(ETA_CL + K)\n  V = TVV * exp(tdist(ETA_V, NU))",
                "",
            )
            .unwrap();
            let same = pa == params_at(&plain, &theta_with(&plain, l), &eta);
            assert_eq!(same, l == 0.0, "λ = {l}: shape live iff λ ≠ 0");
        }
    }
}

/// Box-Cox at λ = 0 is the log-normal model **bit for bit** (the series is
/// exactly `η · 1`), the degenerate oracle the issue asks for.
#[test]
fn box_cox_at_zero_is_the_log_normal_model() {
    let shaped_m = shaped(
        "  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)",
        "",
    )
    .unwrap();
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
        let i = m
            .theta_names
            .iter()
            .position(|t| t == n)
            .unwrap_or_else(|| panic!("{n} declared"));
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
    let i = jd
        .theta_names
        .iter()
        .position(|t| t == "LAMBDA_ETA_V")
        .unwrap();
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
        shaped(
            "  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)",
            "",
        )
        .unwrap(),
        shaped(
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
        )
        .unwrap(),
    ] {
        let t = |e: &str| {
            m.eta_param_info
                .iter()
                .find(|i| i.eta_name == e)
                .unwrap()
                .param_type
        };
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
            &[
                "unknown shape `gamma`",
                "`boxcox`, `tdist` and `johndraper`",
            ],
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
    assert!(shaped(
        "  CL = TVCL * exp(boxcox(ETA_CL, L) + K)\n  V = TVV * exp(ETA_V)",
        ""
    )
    .is_ok());
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
    assert!(
        err.contains("already declares") && err.contains("boxcox(LAMBDA_ETA_CL)"),
        "{err}"
    );
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
    let err = src("[eta_shape]\n  ETA_E ~ boxcox(L)\n")
        .err()
        .expect("refused");
    for phrase in [
        "[error_model] reads it directly",
        "read it through an individual parameter",
    ] {
        assert!(err.contains(phrase), "`{err}` lacks `{phrase}`");
    }
}

/// The inline functions take exactly two arguments.
#[test]
fn shape_functions_take_two_arguments() {
    for (line, phrase) in [
        ("CL = TVCL * exp(boxcox(ETA_CL))", "`boxcox(eta, shape)`"),
        (
            "CL = TVCL * exp(tdist(ETA_CL, NU, 3))",
            "`tdist(eta, shape)`",
        ),
        (
            "CL = TVCL * exp(boxcox(boxcox(ETA_CL, L), L))",
            "already shape-transformed",
        ),
    ] {
        let err = shaped(&format!("  {line}\n  V = TVV * exp(ETA_V)"), "")
            .err()
            .expect(line);
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
        shaped(
            "  CL = TVCL * exp(boxcox(ETA_CL, L))\n  V = TVV * exp(ETA_V)",
            "",
        )
        .unwrap(),
        shaped(
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)",
            "[eta_shape]\n  ETA_CL ~ boxcox(L)\n",
        )
        .unwrap(),
    ] {
        assert!(!m.mu_refs.contains_key("ETA_CL"), "{:?}", m.mu_refs.keys());
        assert_eq!(
            m.mu_refs.get("ETA_V").map(|r| r.theta_name.as_str()),
            Some("TVV")
        );
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
    let err = src("[eta_shape]\n  ETA_F ~ boxcox(L)\n")
        .err()
        .expect("refused");
    assert!(err.contains("[event_model] reads it directly"), "{err}");
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
    assert!(
        (cl(&m) - unshaped).abs() > 0.1,
        "the shape is live: {} vs {unshaped}",
        cl(&m)
    );
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
            if covmodel.is_empty() {
                "  theta THETA_CL_WT(0.75, 0.01, 5.0)\n"
            } else {
                ""
            }
        ))
        .unwrap_or_else(|e| panic!("parse: {e}"))
    };
    let covmodel =
        "[covariate_model]\n  CL ~ WT power(center = 70) => THETA_CL_WT(0.75, 0.01, 5.0)\n\n";
    let block = src(
        "  CL = TVCL * exp(ETA_CL)",
        covmodel,
        "[eta_shape]\n  ETA_CL ~ boxcox(L)\n\n",
    );
    let unshaped = src("  CL = TVCL * exp(ETA_CL)", covmodel, "");
    let inline = src(
        "  CL = TVCL * (WT / 70)^THETA_CL_WT * exp(boxcox(ETA_CL, L))",
        "",
        "",
    );
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
        assert!(
            (b - i).abs() <= 1e-14 * i.abs(),
            "η {eta:?}: block {b} vs inline {i}"
        );
        assert!(
            (b - u).abs() > 1e-3,
            "η {eta:?}: the shape is live ({b} vs unshaped {u})"
        );
    }
}

/// Past `exp`'s overflow Box-Cox is `+inf`, the limit, not `NaN` from
/// `inf − inf` in the value/jet split (#1721 review r1, finding 5). A `NaN`
/// would poison an objective that an `inf` lets a line search back away from.
/// Dies under: removing the overflow arm of `expm1_g`.
#[test]
fn box_cox_overflows_to_infinity_not_nan() {
    let v = shape_g(ShapeKind::BoxCox, 800.0, 1.0);
    assert!(v.is_infinite() && v > 0.0, "boxcox(800, 1) = {v}");
    let d = shape_g(
        ShapeKind::BoxCox,
        Dual2::<2>::var(800.0, 0),
        Dual2::<2>::var(1.0, 1),
    );
    assert!(
        d.value.is_infinite() && d.value > 0.0,
        "dual value {}",
        d.value
    );
    // Just below the overflow the split still holds: finite and exact.
    let x: f64 = 700.0;
    assert_eq!(shape_g(ShapeKind::BoxCox, x, 1.0), x.exp_m1());
}
