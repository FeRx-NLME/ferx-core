//! Integration tests for adaptive Gauss–Hermite quadrature (`method = agq`, #251).
//!
//! Tier-2: every test calls the public `fit()` boundary but returns immediately — either
//! eval-only (`outer_maxiter = 0`, so the OFV is evaluated at the initial parameters with
//! no outer minimisation) or with an expected `Err` from a compatibility guard. No
//! convergence loops, so these are compile-checked and run on every PR.
//!
//! The headline checks:
//!
//! * `n_agq = 1` reproduces the Laplace OFV — the identity the method rests on.
//! * More nodes move the OFV monotonically and *converge*, i.e. the extra work buys a
//!   settled answer rather than drifting.
//! * The objective is deterministic run to run (no Monte-Carlo noise), which is AGQ's
//!   headline advantage over SAEM/IMP.
//!
//! The full convergence fit and the NONMEM comparison live in
//! `docs/estimation/agq.qmd`.

use ferx_core::parser::model_parser::{parse_full_model, parse_model_string};
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult, Population};
use std::path::Path;

/// Warfarin oral 1-cpt, 3 random effects, proportional error. With `n_agq = 3` this is a
/// 3³ = 27-node grid per subject — small enough to stay a Tier-2 test.
const WARFARIN_SRC: &str = r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
";

fn warfarin() -> Population {
    read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data must load")
}

/// OFV at the initial parameters (no outer minimisation) under `method`.
fn eval_only_ofv(population: &Population, method: EstimationMethod, n_agq: usize) -> f64 {
    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    let opts = FitOptions {
        method,
        methods: vec![],
        interaction: method == EstimationMethod::FoceI,
        n_agq,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let result =
        fit(&model, population, &model.default_params, &opts).expect("eval-only fit must succeed");
    assert!(result.ofv.is_finite(), "OFV must be finite");
    result.ofv
}

/// Parse `src` (model + its `[fit_options]`) and fit it eval-only, honouring the file's
/// own fit options — the path a real `.ferx` user takes.
fn fit_from_src(src: &str) -> Result<FitResult, String> {
    let parsed = parse_full_model(src)?;
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options
    };
    fit(
        &parsed.model,
        &warfarin(),
        &parsed.model.default_params,
        &opts,
    )
}

/// The `[fit_options]` a model source parses to.
fn options_of(src: &str) -> FitOptions {
    parse_full_model(src).expect("model must parse").fit_options
}

/// `WARFARIN_SRC` with its `method = focei` line replaced by `body`.
fn with_fit_options(body: &str) -> String {
    WARFARIN_SRC.replace("  method = focei", body)
}

#[test]
fn agq_method_parses_with_aliases() {
    for token in [
        "agq",
        "aghq",
        "gauss_hermite",
        "adaptive_gaussian_quadrature",
    ] {
        let opts = options_of(&with_fit_options(&format!("  method = {token}")));
        assert_eq!(
            opts.method,
            EstimationMethod::Agq,
            "`{token}` must select AGQ — note `gauss_hermite` is a near-miss for the \
             Gauss-*Newton* alias, which matches on `contains(\"gauss\")`, so the AGQ arm \
             has to sit above it in the parser"
        );
    }
    // …and the Gauss-Newton aliases must still resolve to Gauss-Newton, i.e. the AGQ arm
    // sitting above them did not over-capture.
    for token in ["gn", "gauss_newton"] {
        assert_eq!(
            options_of(&with_fit_options(&format!("  method = {token}"))).method,
            EstimationMethod::FoceGn,
            "`{token}` must still select Gauss-Newton"
        );
    }
}

#[test]
fn n_agq_is_validated() {
    // Rejected at parse time (the `[fit_options]` grammar), so `parse_full_model` errors.
    // `ParsedModel` isn't `Debug`, so unwrap the Result by hand rather than `expect_err`.
    for bad in [
        "  method = agq\n  n_agq = 0",
        "  method = agq\n  n_agq = 99",
    ] {
        match parse_full_model(&with_fit_options(bad)) {
            Ok(_) => panic!("out-of-range n_agq must be rejected: {bad:?}"),
            Err(err) => assert!(err.contains("n_agq"), "unhelpful error: {err}"),
        }
    }
    // The default (3) applies when the key is omitted.
    assert_eq!(options_of(&with_fit_options("  method = agq")).n_agq, 3);
}

/// **The identity.** One Gauss–Hermite node sits at the mode with weight `√π`, so the
/// quadrature sum collapses to `(2π)^(d/2)·|H|^(−½)·exp(l(η̂))` — the Laplace approximation.
/// (The arithmetic of that collapse is pinned exactly in `agq::tests::one_node_agq_equals_laplace`;
/// this test pins the *wiring* — that the engine feeds AGQ the right EBEs, parameters and
/// likelihood constant.)
///
/// AGQ(1) is therefore Laplace **with the exact Hessian** — NONMEM's `LAPLACIAN`. It is not
/// expected to equal ferx's FOCEI bit-for-bit, because FOCEI is Laplace with the
/// *Gauss-Newton* Hessian `C'C + Ω⁻¹`, which drops `∂²f/∂η²`. That is a real, documented
/// difference between the two estimators, not implementation drift — so what this test pins
/// is that AGQ(1) lands in the Laplace family at all:
///
/// * it sits within ~1 OFV unit of FOCEI (same family, differing only in `½log|H|`), and
/// * it is nowhere near FOCE, which uses the Sheiner–Beal *linearised* marginal — a
///   genuinely different objective (16.47 vs ~7.5 on this data).
///
/// A wrong likelihood constant (a dropped `log|Ω|`, a stray `log(2π)`) would show up here as
/// tens of OFV units, not tenths.
#[test]
fn one_node_agq_is_the_laplace_approximation() {
    let pop = warfarin();
    let foce = eval_only_ofv(&pop, EstimationMethod::Foce, 1);
    let focei = eval_only_ofv(&pop, EstimationMethod::FoceI, 1);
    let agq1 = eval_only_ofv(&pop, EstimationMethod::Agq, 1);

    let to_focei = (agq1 - focei).abs();
    let to_foce = (agq1 - foce).abs();
    assert!(
        to_focei < 1.0,
        "1-node AGQ must be Laplace, i.e. within a Hessian-approximation's distance of \
         FOCEI: agq1={agq1}, focei={focei} (gap {to_focei:.4} OFV units)"
    );
    assert!(
        to_focei < 0.25 * to_foce,
        "1-node AGQ must be much closer to the Laplace marginal (FOCEI, gap {to_focei:.4}) \
         than to the Sheiner-Beal linearised one (FOCE, gap {to_foce:.4})"
    );
}

/// **The payoff.** Adding nodes must drive the OFV to the *true* marginal, and the truth
/// here is supplied independently: ferx's importance sampler evaluated at the same
/// parameters (`imp_eval_only`) is an unbiased Monte-Carlo estimate of the very same
/// integral, computed by a completely different route (Student-t proposal + self-normalised
/// weights — no quadrature, no Hessian, no Laplace).
///
/// Two estimators of one integral, sharing no code path, must agree. They do, to ~0.02 OFV
/// units. This is what certifies AGQ's marginal *and* its constant convention: any error in
/// either would move the two apart by orders of magnitude more than the MC noise.
#[test]
fn agq_converges_to_the_importance_sampling_marginal() {
    let pop = warfarin();
    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");

    // Independent truth: IS marginal −2logL at the same (initial) parameters.
    let is_opts = FitOptions {
        method: EstimationMethod::Imp,
        methods: vec![],
        interaction: true,
        imp_eval_only: true,
        imp_samples: 20_000,
        imp_seed: Some(7),
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let truth = fit(&model, &pop, &model.default_params, &is_opts)
        .expect("IS evaluation must succeed")
        .importance_sampling
        .expect("IMP eval-only must report a marginal")
        .minus2_log_likelihood;

    let ofvs: Vec<f64> = [1usize, 3, 5, 7, 9, 11]
        .iter()
        .map(|&n| eval_only_ofv(&pop, EstimationMethod::Agq, n))
        .collect();
    let errs: Vec<f64> = ofvs.iter().map(|o| (o - truth).abs()).collect();

    // The refined grid agrees with the Monte-Carlo marginal. The residual is dominated by
    // IS's own MC error at 20k samples, not by quadrature error.
    let refined = *errs.last().expect("non-empty");
    assert!(
        refined < 0.1,
        "AGQ must converge to the IS marginal: truth={truth:.6}, AGQ OFVs={ofvs:?} \
         (final gap {refined:.4})"
    );
    // …and the nodes are what got us there: Laplace alone is markedly further off.
    assert!(
        refined < 0.25 * errs[0],
        "adding nodes must close the gap to the true marginal: Laplace was off by {:.4}, \
         the refined grid by {refined:.4}",
        errs[0]
    );
}

/// Refining the grid must *converge*, not wander: successive refinements move the OFV by
/// less and less. This is what makes a node count meaningful — if the OFV kept drifting with
/// `n_agq` there would be no answer to report.
#[test]
fn ofv_settles_as_nodes_are_added() {
    let pop = warfarin();
    let ofvs: Vec<f64> = [1usize, 3, 5, 7]
        .iter()
        .map(|&n| eval_only_ofv(&pop, EstimationMethod::Agq, n))
        .collect();

    let steps: Vec<f64> = ofvs.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    assert!(
        steps.last().unwrap() < steps.first().unwrap(),
        "AGQ must settle as nodes are added; OFVs {ofvs:?} moved by {steps:?}"
    );
    // The 5→7 refinement is small in absolute terms — the grid is converged.
    assert!(
        *steps.last().unwrap() < 0.1,
        "5→7-node refinement moved the OFV by {:.4} — grid is not converged. OFVs: {ofvs:?}",
        steps.last().unwrap()
    );
}

/// AGQ is deterministic: a fixed grid, no sampling. Re-evaluating at the same parameters
/// must return a **bit-identical** OFV. (SAEM/IMP cannot promise this, and it is why AGQ's
/// outer optimizer needs no common-random-numbers or step-size tricks.)
#[test]
fn ofv_is_bit_identical_across_runs() {
    let pop = warfarin();
    let a = eval_only_ofv(&pop, EstimationMethod::Agq, 3);
    let b = eval_only_ofv(&pop, EstimationMethod::Agq, 3);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "AGQ OFV must be bit-identical run to run: {a} vs {b}"
    );
}

/// The quadrature integrates over the BSV η only. Under IOV the marginal is a joint
/// integral over η *and* every occasion's κ; silently integrating the η-only marginal
/// would report a confidently wrong OFV, so the fit must be rejected instead.
#[test]
fn agq_rejects_iov() {
    let src = WARFARIN_SRC
        .replace("  method = focei", "  method = agq")
        .replace(
            "  sigma PROP_ERR ~ 0.1 (sd)",
            "  sigma PROP_ERR ~ 0.1 (sd)\n  kappa KAPPA_CL ~ 0.05",
        )
        .replace(
            "  CL = TVCL * exp(ETA_CL)",
            "  CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        );
    // If the IOV DSL spelling drifts, fail loudly rather than let a parse error make this
    // test vacuously "pass" without ever reaching the guard.
    let err = fit_from_src(&src).expect_err("AGQ + IOV must be rejected");
    assert!(
        !err.contains("parse") && !err.contains("unknown"),
        "IOV warfarin model must parse (test needs updating): {err}"
    );
    assert!(
        err.contains("iov") || err.contains("IOV") || err.contains("occasion"),
        "the AGQ+IOV rejection must say why: {err}"
    );
}

/// The tensor grid is `n_agq^n_eta`. With 3 random effects, `n_agq = 21` is 9261 nodes
/// (fine); the cap exists to stop a grid that would never finish. Check the guard fires
/// with an actionable message rather than launching the fit.
#[test]
fn agq_rejects_an_intractable_grid() {
    // 8 random effects at 21 nodes = 21^8 ≈ 3.8e10 nodes — far past MAX_AGQ_GRID.
    let mut params = String::new();
    let mut ind = String::new();
    for i in 0..8 {
        params.push_str(&format!("  omega ETA_{i} ~ 0.05\n"));
    }
    // Fold every eta into CL so they are all real random effects on the model.
    let sum: Vec<String> = (0..8).map(|i| format!("ETA_{i}")).collect();
    ind.push_str(&format!("  CL = TVCL * exp({})\n", sum.join(" + ")));

    let src = format!(
        r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
{params}
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
{ind}  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = agq
  n_agq = 21
"
    );
    let err = fit_from_src(&src).expect_err("an intractable AGQ grid must be rejected");
    assert!(
        err.contains("n_agq") && err.contains("grid"),
        "the grid-cap rejection must name the lever to turn: {err}"
    );
}
