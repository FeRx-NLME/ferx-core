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
use ferx_core::{
    fit, read_nonmem_csv, CompiledModel, EstimationMethod, FitOptions, FitResult, Population,
};
use std::path::Path;

mod common;

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

/// OFV at the initial parameters (no outer minimisation) of an already-parsed `model`.
fn eval_only_ofv_model(
    model: &CompiledModel,
    population: &Population,
    method: EstimationMethod,
    n_agq: usize,
) -> f64 {
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
        fit(model, population, &model.default_params, &opts).expect("eval-only fit must succeed");
    assert!(result.ofv.is_finite(), "OFV must be finite");
    result.ofv
}

/// OFV at the initial parameters (no outer minimisation) of `WARFARIN_SRC` under `method`.
fn eval_only_ofv(population: &Population, method: EstimationMethod, n_agq: usize) -> f64 {
    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    eval_only_ofv_model(&model, population, method, n_agq)
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
fn agq_method_token_is_rejected_with_migration_note() {
    // `method = agq` was removed (#251): adaptive quadrature is `method = laplace` with
    // `n_agq > 1`. The old tokens must error, and the message must name the replacement.
    for token in [
        "agq",
        "aghq",
        "gauss_hermite",
        "adaptive_gaussian_quadrature",
    ] {
        match parse_full_model(&with_fit_options(&format!("  method = {token}"))) {
            Ok(_) => panic!("`{token}` must be rejected now that agq is removed"),
            Err(err) => assert!(
                err.contains("laplace") && err.contains("n_agq"),
                "`{token}` error must point to `method = laplace` + `n_agq`: {err}"
            ),
        }
    }
    // The Gauss-Newton aliases must still resolve to Gauss-Newton (the removed `agq` arm
    // sat above them and must not have taken `gn` / `gauss_newton` down with it).
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
        "  method = laplace\n  n_agq = 0",
        "  method = laplace\n  n_agq = 99",
    ] {
        match parse_full_model(&with_fit_options(bad)) {
            Ok(_) => panic!("out-of-range n_agq must be rejected: {bad:?}"),
            Err(err) => assert!(err.contains("n_agq"), "unhelpful error: {err}"),
        }
    }
    // The default (1 = Laplace) applies when the key is omitted.
    assert_eq!(options_of(&with_fit_options("  method = laplace")).n_agq, 1);
}

/// **The identity.** One Gauss–Hermite node sits at the mode with weight `√π`, so the
/// quadrature sum collapses to `(2π)^(d/2)·|H|^(−½)·exp(l(η̂))` — the Laplace approximation.
/// (The arithmetic of that collapse is pinned exactly in `agq::tests::one_node_agq_equals_laplace`;
/// this test pins the *wiring* — that the engine feeds AGQ the right EBEs, parameters and
/// likelihood constant.)
///
/// AGQ(1) is therefore Laplace **with the exact Hessian** — NONMEM's `LAPLACIAN`. It is not
/// expected to equal ferx's FOCEI bit-for-bit, because FOCEI is Laplace with the
/// *Gauss-Newton* Hessian `C'C + Ω⁻¹`, which drops `∂²f/∂η²`.
///
/// That difference is real and externally confirmed, not implementation drift: on this very
/// model NONMEM `$EST METHOD=1 LAPLACIAN INTER` returns 7.8994 (matching AGQ(1) to five
/// significant figures) while NONMEM `METHOD=1 INTER` returns 7.4967 (matching ferx FOCEI)
/// — the reference implementation shows the same 0.40-unit gap. See `docs/estimation/agq.qmd`.
///
/// So what this test pins is that AGQ(1) lands in the Laplace family at all:
///
/// * it sits within ~1 OFV unit of FOCEI (same family, differing only in `½log|H|`), and
/// * it is nowhere near FOCE, which uses the Sheiner–Beal *linearised* marginal — a
///   genuinely different objective (16.47 vs ~7.5 on this data).
///
/// A wrong likelihood constant (a dropped `log|Ω|`, a stray `log(2π)`) would show up here as
/// tens of OFV units, not tenths.
/// **`focei` with `n_agq > 1` is the Gauss-Newton-anchored quadrature** (#251): the same
/// adaptive-GH machinery as `laplace`, but scaling the grid with FOCEI's Gauss-Newton `H̃`
/// instead of the exact Hessian. So `focei, n_agq = 3` must
///
/// * be finite and in FOCEI's basin (a few OFV units of `focei, n_agq = 1`), and
/// * differ from `laplace, n_agq = 3` — same nodes-count, *different anchor* — which is the
///   whole point of exposing the anchor through the method name.
#[test]
fn focei_n_agq_gt_1_is_the_gauss_newton_anchored_quadrature() {
    let pop = warfarin();
    let focei1 = eval_only_ofv(&pop, EstimationMethod::FoceI, 1);
    let focei3 = eval_only_ofv(&pop, EstimationMethod::FoceI, 3);
    let laplace3 = eval_only_ofv(&pop, EstimationMethod::Laplace, 3);

    assert!(
        focei3.is_finite() && (focei3 - focei1).abs() < 5.0,
        "focei n=3 (GN-anchored quadrature) must refine FOCEI, not diverge: \
         focei1={focei1}, focei3={focei3}"
    );
    assert_ne!(
        focei3.to_bits(),
        laplace3.to_bits(),
        "the anchor must matter: focei (Gauss-Newton) and laplace (exact) at the same n_agq = 3 \
         must differ — focei3={focei3}, laplace3={laplace3}"
    );
}

#[test]
fn one_node_agq_is_the_laplace_approximation() {
    let pop = warfarin();
    let foce = eval_only_ofv(&pop, EstimationMethod::Foce, 1);
    let focei = eval_only_ofv(&pop, EstimationMethod::FoceI, 1);
    let agq1 = eval_only_ofv(&pop, EstimationMethod::Laplace, 1);

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
        .map(|&n| eval_only_ofv(&pop, EstimationMethod::Laplace, n))
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

/// **The unification's mathematical heart** (#251): the exact anchor (`laplace`) and the
/// Gauss-Newton anchor (`focei`) place their grids differently, so they disagree at one node
/// — but the quadrature identity holds for *any* positive-definite scaling, so both converge
/// to the **same** marginal integral as `n_agq` grows. This pins exactly that: the two anchors
/// differ at n = 1 (FOCEI ≠ Laplace) yet both land on the IS marginal by n = 11.
#[test]
fn both_anchors_converge_to_the_same_marginal() {
    let pop = warfarin();
    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");

    // Independent truth: the IS marginal at the initial parameters.
    let is_opts = FitOptions {
        method: EstimationMethod::Imp,
        interaction: true,
        imp_eval_only: true,
        imp_samples: 20_000,
        imp_seed: Some(7),
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let truth = fit(&model, &pop, &model.default_params, &is_opts)
        .expect("IS eval must succeed")
        .importance_sampling
        .expect("IMP eval-only reports a marginal")
        .minus2_log_likelihood;

    let laplace1 = eval_only_ofv(&pop, EstimationMethod::Laplace, 1);
    let focei1 = eval_only_ofv(&pop, EstimationMethod::FoceI, 1);
    let laplace11 = eval_only_ofv(&pop, EstimationMethod::Laplace, 11);
    let focei11 = eval_only_ofv(&pop, EstimationMethod::FoceI, 11);

    // Different anchor ⇒ different value at one node.
    assert!(
        (laplace1 - focei1).abs() > 1e-6,
        "the anchors must differ at n = 1: laplace={laplace1}, focei={focei1}"
    );
    // Same integral in the limit ⇒ both close on the IS marginal (residual is IS's own MC
    // error at 20k samples, not quadrature error).
    assert!(
        (laplace11 - truth).abs() < 0.15 && (focei11 - truth).abs() < 0.15,
        "both anchors must converge to the same marginal (IS truth {truth:.4}): \
         laplace(11)={laplace11:.4}, focei(11)={focei11:.4}"
    );
    // And to each other.
    assert!(
        (laplace11 - focei11).abs() < 0.1,
        "the anchors must agree once refined: laplace(11)={laplace11:.4}, focei(11)={focei11:.4}"
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
        .map(|&n| eval_only_ofv(&pop, EstimationMethod::Laplace, n))
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
    let a = eval_only_ofv(&pop, EstimationMethod::Laplace, 3);
    let b = eval_only_ofv(&pop, EstimationMethod::Laplace, 3);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "AGQ OFV must be bit-identical run to run: {a} vs {b}"
    );
}

/// **The analytic outer gradient must be the exact gradient of the objective — at every
/// node count, `n_agq = 1` included.**
///
/// AGQ's gradient has two parts (see `estimation::agq`):
///
/// 1. the posterior-weighted average of fixed-η scores over the nodes (the Fisher identity),
///    which is exact only in the exact-quadrature limit; and
/// 2. the **grid-response** term `∂Φ/∂H · dH/dx`, which supplies exactly what part 1 omits.
///
/// Part 1 alone degrades as the quadrature coarsens — measured on this model it is ~26%
/// wrong at `n_agq = 1`, where the rule *is* Laplace and the `½·log|H|` derivative no longer
/// cancels against the node movement. Part 2 is what makes the total exact anyway, and
/// `n_agq = 1` is therefore the **strongest** test of it, not the weakest: it is the case
/// where the correction is carrying the entire missing term.
///
/// The FD reference re-solves the inner loop and rebuilds the grid at each perturbed point,
/// so it is the *total* derivative — node movement and EBE response included — not a
/// fixed-node echo of the analytic formula.
#[test]
fn analytic_gradient_matches_fd_at_every_node_count() {
    use ferx_core::estimation::agq;
    use ferx_core::estimation::parameterization::{pack_params, unpack_params};

    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    let pop = warfarin();
    assert!(
        agq::analytic_gradient_available(&model),
        "warfarin must be in the analytic-gradient scope, else this test proves nothing"
    );

    let template = &model.default_params;
    let x0 = pack_params(template);
    let np = x0.len();

    // Objective at a packed point: re-solve EBEs (so the grid re-centres) then evaluate the
    // AGQ OFV. This is the function the outer optimizer actually minimises.
    let ofv = |x: &[f64], n: usize| -> f64 {
        let params = unpack_params(x, template);
        let (ehs, _hms, _stats, _k) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, &params, 500, 1e-12, None, None, 0, 0,
        );
        2.0 * agq::agq_population_nll(
            &model,
            &pop,
            &params,
            &ehs,
            &[],
            n,
            ferx_core::types::HessianAnchor::Exact,
        )
    };

    let mut rel_errs = Vec::new();
    for &n in &[1usize, 3, 5, 7] {
        // Analytic gradient at x0.
        let params = unpack_params(&x0, template);
        let (ehs, _h, _s, _k) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, &params, 500, 1e-12, None, None, 0, 0,
        );
        let g = agq::agq_population_gradient(
            &model,
            &pop,
            &params,
            template,
            &x0,
            &ehs,
            &[],
            n,
            ferx_core::types::HessianAnchor::Exact,
        )
        .expect("analytic gradient must be available in scope");

        // Central FD of the true objective. Step 1e-4, not 1e-5: this references a
        // **reconverged** objective, so a step below the inner solver's noise floor amplifies
        // that noise rather than cutting truncation, and the reference — not the gradient —
        // becomes the inaccurate side. See the IOV twin of this test.
        let mut fd = vec![0.0f64; np];
        for i in 0..np {
            let h = 1e-4 * (1.0 + x0[i].abs());
            let (mut xp, mut xm) = (x0.clone(), x0.clone());
            xp[i] += h;
            xm[i] -= h;
            fd[i] = (ofv(&xp, n) - ofv(&xm, n)) / (2.0 * h);
        }

        let scale = fd
            .iter()
            .chain(g.iter())
            .fold(1e-6f64, |m, v| m.max(v.abs()));
        let max_diff = g
            .iter()
            .zip(fd.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        rel_errs.push(max_diff / scale);
    }

    // The gradient must be exact at EVERY node count — n = 1 included. There is no node
    // gate: `grid_response_correction` supplies the `∂Φ/∂H·dH/dx` term that the fixed-node
    // score omits, and at n = 1 that term is the entire difference (a 26% error without it).
    //
    // Measured: 2.0e-5 at n = 1, 1.5e-6 at n = 3, 2.1e-7 at n = 7 — the gradient is exact to
    // the reference's own precision. (An earlier revision of this test used a 1e-5 FD step and
    // read 1.6e-3; that was the *reference's* noise, not the gradient's error — see the step
    // comment above. Do not "restore" the smaller step.)
    //
    // The 1e-4 bound also pins `AGQ_GRID_FD_STEP`: at 1e-2 this test fails outright, because
    // the grid-response term is truncation-dominated and a "safe" large step puts the θ
    // gradient 5x off.
    for (k, &n) in [1usize, 3, 5, 7].iter().enumerate() {
        assert!(
            rel_errs[k] < 1e-4,
            "n_agq={n}: analytic gradient must match FD of the objective, got {:.3e} \
             relative error (all: {rel_errs:?})",
            rel_errs[k]
        );
    }
    // …and it sharpens as the quadrature does — the signature of the grid-response term doing
    // its job rather than agreeing by coincidence.
    assert!(
        rel_errs[3] < rel_errs[0],
        "accuracy should improve with node count: {rel_errs:?}"
    );
}

/// The **Gauss-Newton-anchored** quadrature (`focei, n_agq > 1`) has its own analytic
/// gradient, and it must match a finite difference of its own objective (#251). Same
/// machinery as the exact-anchor test above — the fixed-node score is anchor-independent, the
/// grid-response term differences the Gauss-Newton `H̃` instead of the exact Hessian, and the
/// mode response `dη̂/dx` stays the exact one (η̂ minimises the true conditional NLL regardless
/// of which Hessian scales the grid).
#[test]
fn gauss_newton_anchored_gradient_matches_fd() {
    use ferx_core::estimation::agq;
    use ferx_core::estimation::parameterization::{pack_params, unpack_params};
    use ferx_core::types::HessianAnchor::GaussNewton;

    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    let pop = warfarin();
    let template = &model.default_params;
    let x0 = pack_params(template);
    let np = x0.len();

    let ofv = |x: &[f64], n: usize| -> f64 {
        let params = unpack_params(x, template);
        let (ehs, _h, _s, _k) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, &params, 500, 1e-12, None, None, 0, 0,
        );
        2.0 * agq::agq_population_nll(&model, &pop, &params, &ehs, &[], n, GaussNewton)
    };

    let mut rel_errs = Vec::new();
    for &n in &[1usize, 3, 5] {
        let params = unpack_params(&x0, template);
        let (ehs, _h, _s, _k) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, &params, 500, 1e-12, None, None, 0, 0,
        );
        let g = agq::agq_population_gradient(
            &model,
            &pop,
            &params,
            template,
            &x0,
            &ehs,
            &[],
            n,
            GaussNewton,
        )
        .expect("GN-anchored analytic gradient must be available in scope");

        let mut fd = vec![0.0f64; np];
        for i in 0..np {
            let h = 1e-4 * (1.0 + x0[i].abs());
            let (mut xp, mut xm) = (x0.clone(), x0.clone());
            xp[i] += h;
            xm[i] -= h;
            fd[i] = (ofv(&xp, n) - ofv(&xm, n)) / (2.0 * h);
        }

        let scale = fd
            .iter()
            .chain(g.iter())
            .fold(1e-6f64, |m, v| m.max(v.abs()));
        let max_diff = g
            .iter()
            .zip(fd.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        rel_errs.push(max_diff / scale);
    }

    for (k, &n) in [1usize, 3, 5].iter().enumerate() {
        assert!(
            rel_errs[k] < 2e-3,
            "n_agq={n}: GN-anchored analytic gradient must match FD of its objective, got \
             {:.3e} (all: {rel_errs:?})",
            rel_errs[k]
        );
    }
}

/// **AGQ has its own gradient for every model — nothing falls back to the inner-re-solving
/// FD path.**
///
/// The θ/σ score is analytic where the `Dual2` provider reaches, and finite-differenced *at
/// fixed η* everywhere else (TTE, categorical, M3, LTBS, `block_sigma`, FREM, ODE). Both
/// avoid the inner re-solve, which is the expensive part — so the endpoints AGQ actually
/// exists for are not the slow case.
///
/// This pins that the two paths compute the *same* gradient. `gradient_method = fd` takes
/// warfarin off the analytic score (it is the documented FD escape hatch and
/// `analytic_outer_gradient_available` honours it), so the same model can be driven down both
/// routes and the results compared directly.
#[test]
fn fd_score_path_agrees_with_the_analytic_one() {
    use ferx_core::estimation::agq;
    use ferx_core::estimation::parameterization::{pack_params, unpack_params};
    use ferx_core::types::GradientMethod;

    let pop = warfarin();
    let analytic_model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    let mut fd_model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    fd_model.gradient_method = GradientMethod::Fd;

    assert!(
        agq::analytic_gradient_available(&analytic_model)
            && agq::analytic_gradient_available(&fd_model),
        "AGQ must supply its own gradient for BOTH models — there is no scope gate"
    );

    let template = &analytic_model.default_params;
    let x = pack_params(template);
    let params = unpack_params(&x, template);
    let (ehs, _h, _s, _k) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
        &analytic_model,
        &pop,
        &params,
        300,
        1e-10,
        None,
        None,
        0,
        0,
    );

    for n in [1usize, 3] {
        let g_an = agq::agq_population_gradient(
            &analytic_model,
            &pop,
            &params,
            template,
            &x,
            &ehs,
            &[],
            n,
            ferx_core::types::HessianAnchor::Exact,
        )
        .expect("analytic-score gradient");
        let g_fd = agq::agq_population_gradient(
            &fd_model,
            &pop,
            &params,
            template,
            &x,
            &ehs,
            &[],
            n,
            ferx_core::types::HessianAnchor::Exact,
        )
        .expect("FD-score gradient");

        let scale = g_an
            .iter()
            .chain(g_fd.iter())
            .fold(1e-6f64, |m, v| m.max(v.abs()));
        let max_diff = g_an
            .iter()
            .zip(g_fd.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_diff / scale < 1e-3,
            "n_agq={n}: the FD score must reproduce the analytic one — they are the same \
             quantity. rel diff {:.3e}\n  analytic: {g_an:?}\n  fd:       {g_fd:?}",
            max_diff / scale
        );
    }
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
  method = laplace
  n_agq = 21
"
    );
    let err = fit_from_src(&src).expect_err("an intractable AGQ grid must be rejected");
    assert!(
        err.contains("n_agq") && err.contains("grid"),
        "the grid-cap rejection must name the lever to turn: {err}"
    );
}

/// `n_agq` past the per-dimension node cap is rejected at model-check time even when the
/// resulting *grid* fits under `MAX_AGQ_GRID` — the case a direct-Rust-API caller can reach,
/// since the parser guard that protects file/FFI fits is bypassed (review #821, point 2).
/// With a single random effect, `n_agq = 100` is a 100-node grid (well under the grid cap)
/// yet drives `gauss_hermite(100)` past where Golub–Welsch stays accurate.
#[test]
fn agq_rejects_out_of_range_node_count_built_via_the_rust_api() {
    use ferx_core::api::check_model_options;

    // One random effect: grid = n_agq^1 = n_agq, so the grid cap cannot catch this — only the
    // per-dimension node cap can.
    let src = WARFARIN_SRC
        .replace("  omega ETA_V  ~ 0.04\n", "")
        .replace("  omega ETA_KA ~ 0.30\n", "")
        .replace("  V  = TVV  * exp(ETA_V)", "  V  = TVV")
        .replace("  KA = TVKA * exp(ETA_KA)", "  KA = TVKA");
    let model = parse_model_string(&src).expect("single-eta warfarin must parse");
    assert_eq!(
        model.n_eta, 1,
        "fixture must have exactly one random effect"
    );

    let n = ferx_core::estimation::agq::MAX_AGQ_NODES + 79; // 100, safely under MAX_AGQ_GRID at n_eta = 1
    assert!(
        ferx_core::estimation::agq::grid_size(n, model.n_eta)
            < ferx_core::estimation::agq::MAX_AGQ_GRID,
        "the grid must stay under MAX_AGQ_GRID so only the node cap can reject this"
    );
    let opts = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: n,
        ..FitOptions::default()
    };
    let diags = check_model_options(&model, &opts);
    assert!(
        diags.iter().any(|d| d.code == "E_AGQ_NODES_TOO_LARGE"),
        "an out-of-range n_agq must be rejected by check_model_options even under the grid cap; \
         got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
    // The in-range boundary (exactly MAX_AGQ_NODES) must NOT be rejected.
    let ok_opts = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: ferx_core::estimation::agq::MAX_AGQ_NODES,
        ..FitOptions::default()
    };
    assert!(
        !check_model_options(&model, &ok_opts)
            .iter()
            .any(|d| d.code == "E_AGQ_NODES_TOO_LARGE"),
        "n_agq = MAX_AGQ_NODES is in range and must be accepted"
    );
}

/// The `n_agq` caps must fire for the **FOCEI** quadrature too, not only Laplace (#251
/// review #1). A non-IOV closed-form model with `method = focei, n_agq` over the cap would
/// otherwise skip every check-time guard (the runtime cap only fires under IOV) and build a
/// 100k+-node grid unchecked — a hang / OOM. Plain FOCEI (`n_agq = 1`) must stay uncapped.
#[test]
fn focei_quadrature_hits_the_node_cap_like_laplace() {
    use ferx_core::api::check_model_options;

    let model = parse_model_string(WARFARIN_SRC).expect("warfarin must parse");
    let over = ferx_core::estimation::agq::MAX_AGQ_NODES + 79; // 100

    // focei + n_agq > MAX_AGQ_NODES → rejected, exactly as laplace is.
    let focei = FitOptions {
        method: EstimationMethod::FoceI,
        n_agq: over,
        ..FitOptions::default()
    };
    assert!(
        check_model_options(&model, &focei)
            .iter()
            .any(|d| d.code == "E_AGQ_NODES_TOO_LARGE" || d.code == "E_AGQ_GRID_TOO_LARGE"),
        "focei with an over-cap n_agq must be rejected at check time, not left to hang"
    );

    // Plain FOCEI (n_agq = 1) is not a quadrature — the caps must NOT fire.
    let plain = FitOptions {
        method: EstimationMethod::FoceI,
        n_agq: 1,
        ..FitOptions::default()
    };
    assert!(
        !check_model_options(&model, &plain)
            .iter()
            .any(|d| d.code.starts_with("E_AGQ_")),
        "plain FOCEI (n_agq = 1) must not trip the quadrature caps"
    );
}

/// `focei` + `n_agq > 1` is rejected outside the analytic sensitivity scope (the GN anchor's
/// `H̃` comes from the provider). Forcing `gradient_method = fd` takes the model out of scope,
/// which must surface `E_FOCEI_NAGQ_UNSUPPORTED` rather than a deep runtime failure. `laplace`
/// (exact anchor, no provider dependency) has no such restriction.
#[test]
fn focei_quadrature_out_of_analytic_scope_is_rejected() {
    use ferx_core::api::check_model_options;
    use ferx_core::types::GradientMethod;

    let mut model = parse_model_string(WARFARIN_SRC).expect("warfarin must parse");
    model.gradient_method = GradientMethod::Fd; // forces analytic_score_supported == false

    let focei = FitOptions {
        method: EstimationMethod::FoceI,
        n_agq: 3,
        ..FitOptions::default()
    };
    assert!(
        check_model_options(&model, &focei)
            .iter()
            .any(|d| d.code == "E_FOCEI_NAGQ_UNSUPPORTED"),
        "focei + n_agq > 1 out of analytic scope must be rejected"
    );
    // Laplace is unaffected — the exact anchor needs no provider.
    let laplace = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: 3,
        ..FitOptions::default()
    };
    assert!(
        !check_model_options(&model, &laplace)
            .iter()
            .any(|d| d.code == "E_FOCEI_NAGQ_UNSUPPORTED"),
        "laplace + n_agq > 1 must not be gated on analytic scope"
    );
}

/// The runtime IOV grid cap fires for the FOCEI quadrature too, and names `focei` (not
/// `laplace`) in the message (#251 review #4). `21^5` over the stacked (η, κ) dimension is
/// far past `MAX_AGQ_GRID`.
#[test]
fn focei_iov_grid_cap_reports_focei() {
    let parsed = parse_full_model(WARFARIN_IOV_SRC).expect("IOV model must parse");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        n_agq: 21,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options
    };
    let err = fit(
        &parsed.model,
        &warfarin_iov(),
        &parsed.model.default_params,
        &opts,
    )
    .expect_err("an intractable FOCEI-IOV grid must be rejected");
    assert!(
        err.contains("method = focei") && err.contains("n_agq"),
        "the FOCEI-IOV grid-cap error must name focei and the lever: {err}"
    );
}

/// **The AGQ covariance stencil.** Every other AGQ test sets `run_covariance_step = false`,
/// so the AGQ-marginal FD covariance (differenced through the `pop_nll_opts` seam in
/// `compute_covariance`, so the standard errors difference the objective AGQ actually
/// optimised) has no regression test. Run a short AGQ fit *with* covariance on and assert it
/// produces finite, positive standard errors (review #821, point 4).
#[test]
fn agq_covariance_step_produces_finite_standard_errors() {
    let pop = warfarin();
    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    let opts = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: 1,
        // A handful of outer iterations so covariance is evaluated near the optimum where the
        // marginal Hessian is PD, while keeping this a Tier-2 (fast, no convergence loop) test.
        outer_maxiter: 8,
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let res = fit(&model, &pop, &model.default_params, &opts).expect("AGQ fit with covariance");
    let se = res
        .se_theta
        .expect("AGQ must report theta standard errors when covariance is on");
    assert_eq!(se.len(), 3, "one SE per theta");
    assert!(
        se.iter().all(|s| s.is_finite() && *s > 0.0),
        "AGQ theta standard errors must be finite and positive: {se:?}"
    );
    assert!(
        res.covariance_matrix.is_some(),
        "AGQ covariance matrix must be reported"
    );
}

/// AGQ over a **non-Gaussian endpoint** — the capability the method exists for and which every
/// other test in this file (all Gaussian warfarin PK) leaves unexercised (review #821, point 3).
///
/// The likelihood integrated here is a mixed-effects **binary / logistic** model, which sits
/// entirely outside the `Dual2` sensitivity provider — so it drives AGQ's fixed-η FD score path
/// (the one that also serves TTE, categorical, ODE, M3), not the analytic-provider path the
/// Gaussian tests take. The independent truth is ferx's importance sampler evaluating the same
/// marginal by a completely different route (Student-t / prior proposal, self-normalised
/// weights — no quadrature, no Hessian), exactly as `agq_converges_to_the_importance_sampling_marginal`
/// anchors the Gaussian node sweep.
#[cfg(feature = "survival")]
mod non_gaussian {
    use ferx_core::estimation::agq;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::{fit, CompiledModel, EstimationMethod, FitOptions, Population};

    /// Mixed-effects logistic: population intercept + slope on `X`, plus a per-subject random
    /// intercept `ETA_I`. One binary observation per row; the logit reads the per-record `TIME`
    /// via the covariate `X` only, so the marginal is a genuine 1-D integral over `ETA_I`.
    const BINARY_MIXED: &str = r"
[parameters]
  theta TH0(0.2, -10.0, 10.0)
  theta THX(0.8, -10.0, 10.0)
  omega ETA_I ~ 0.4

[binary_model]
  cmt   = 3
  logit = TH0 + THX * X + ETA_I
";

    /// The same model with the random intercept removed — ordinary (fixed-effects) logistic
    /// regression, `n_eta = 0`. Exercises AGQ's `d == 0` short-circuits.
    const BINARY_FIXED: &str = r"
[parameters]
  theta TH0(0.2, -10.0, 10.0)
  theta THX(0.8, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = TH0 + THX * X
";

    /// A deterministic binary dataset: 16 subjects, 4 observations each, `X` spread across a
    /// range and the 0/1 states loosely tracking `TH0 + THX·X` so the model is identifiable.
    fn binary_population() -> Population {
        let subjects: Vec<(f64, Vec<(f64, u8)>)> = vec![
            (-2.0, vec![(0.0, 0), (1.0, 0), (2.0, 0), (3.0, 0)]),
            (-1.5, vec![(0.0, 0), (1.0, 0), (2.0, 1), (3.0, 0)]),
            (-1.0, vec![(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 0)]),
            (-0.8, vec![(0.0, 1), (1.0, 0), (2.0, 0), (3.0, 0)]),
            (-0.5, vec![(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 1)]),
            (-0.3, vec![(0.0, 1), (1.0, 0), (2.0, 1), (3.0, 0)]),
            (-0.1, vec![(0.0, 0), (1.0, 1), (2.0, 1), (3.0, 0)]),
            (0.1, vec![(0.0, 1), (1.0, 0), (2.0, 1), (3.0, 1)]),
            (0.3, vec![(0.0, 1), (1.0, 1), (2.0, 0), (3.0, 1)]),
            (0.5, vec![(0.0, 0), (1.0, 1), (2.0, 1), (3.0, 1)]),
            (0.8, vec![(0.0, 1), (1.0, 1), (2.0, 1), (3.0, 0)]),
            (1.0, vec![(0.0, 1), (1.0, 0), (2.0, 1), (3.0, 1)]),
            (1.3, vec![(0.0, 1), (1.0, 1), (2.0, 1), (3.0, 1)]),
            (1.6, vec![(0.0, 1), (1.0, 1), (2.0, 0), (3.0, 1)]),
            (2.0, vec![(0.0, 1), (1.0, 1), (2.0, 1), (3.0, 1)]),
            (2.5, vec![(0.0, 0), (1.0, 1), (2.0, 1), (3.0, 1)]),
        ];
        crate::common::binary_pop(&subjects, 3)
    }

    /// Eval-only AGQ OFV of `model` on the binary population at the initial parameters.
    fn agq_ofv(model: &CompiledModel, pop: &Population, n_agq: usize) -> f64 {
        crate::eval_only_ofv_model(model, pop, EstimationMethod::Laplace, n_agq)
    }

    /// **Point 3: the binary marginal is integrated correctly, cross-checked against IS.**
    #[test]
    fn agq_integrates_a_binary_endpoint_and_agrees_with_importance_sampling() {
        let model = parse_model_string(BINARY_MIXED).expect("mixed binary model must parse");
        assert_eq!(
            model.n_eta, 1,
            "fixture must have exactly one random effect"
        );
        assert!(
            agq::analytic_gradient_available(&model),
            "AGQ supplies its own gradient for the binary endpoint too (no scope gate)"
        );
        let pop = binary_population();

        // AGQ node sweep at the initial parameters — every OFV must be finite.
        let agq: Vec<f64> = [1usize, 3, 7, 15]
            .iter()
            .map(|&n| agq_ofv(&model, &pop, n))
            .collect();
        assert!(
            agq.iter().all(|o| o.is_finite()),
            "AGQ binary OFVs must be finite: {agq:?}"
        );

        // Independent truth: the IS marginal at the same parameters. IS estimates the same
        // integral by sampling η and averaging the exact conditional likelihood (which includes
        // the logit term via `accumulate_non_gaussian_nll`), so it shares no code path with the
        // quadrature. Deterministic under a fixed seed.
        let is_opts = FitOptions {
            method: EstimationMethod::Imp,
            interaction: true,
            imp_eval_only: true,
            imp_samples: 100_000,
            imp_seed: Some(11),
            outer_maxiter: 0,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let truth = fit(&model, &pop, &model.default_params, &is_opts)
            .expect("IS evaluation must succeed on the binary model")
            .importance_sampling
            .expect("IMP eval-only must report a marginal")
            .minus2_log_likelihood;

        let refined = (agq.last().unwrap() - truth).abs();
        assert!(
            refined < 0.5,
            "the refined AGQ grid must agree with the IS marginal on the binary endpoint: \
             truth={truth:.6}, AGQ OFVs={agq:?} (final gap {refined:.4})"
        );
    }

    /// Point 3 (gradient side): a short *optimising* AGQ fit over the binary endpoint drives the
    /// fixed-η FD score path end-to-end (the analytic-provider path never fires here), returning
    /// a finite OFV. This is the headline capability — a non-Gaussian likelihood optimised by
    /// AGQ — exercised through the public `fit()` boundary rather than only the eval-only OFV.
    #[test]
    fn agq_optimises_a_binary_endpoint() {
        let model = parse_model_string(BINARY_MIXED).expect("mixed binary model must parse");
        let pop = binary_population();
        let opts = FitOptions {
            method: EstimationMethod::Laplace,
            n_agq: 3,
            outer_maxiter: 3, // Tier-2: a few steps, no convergence loop
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let res = fit(&model, &pop, &model.default_params, &opts)
            .expect("a short AGQ fit over a binary endpoint must return Ok");
        assert!(
            res.ofv.is_finite(),
            "AGQ binary fit OFV must be finite: {}",
            res.ofv
        );
    }

    /// **Point 4: the `d == 0` short-circuit** (`agq_subject_nll`, `agq_subject_packed_gradient`).
    /// A fixed-effects binary model has no random effect, so the marginal is a point mass and AGQ
    /// degenerates to the conditional likelihood — the node count cannot matter, and the OFV must
    /// equal FOCEI's exactly. The short optimising fit then drives the gradient's own `d == 0`
    /// branch (`accumulate_fixed_eta_packed_gradient`).
    #[test]
    fn agq_with_no_random_effects_is_the_conditional_likelihood() {
        let model =
            parse_model_string(BINARY_FIXED).expect("fixed-effects binary model must parse");
        assert_eq!(model.n_eta, 0, "fixture must have no random effects");
        let pop = binary_population();

        let agq1 = agq_ofv(&model, &pop, 1);
        let agq7 = agq_ofv(&model, &pop, 7);
        let focei = crate::eval_only_ofv_model(&model, &pop, EstimationMethod::FoceI, 1);

        // Nothing to integrate over ⇒ the node count cannot move the OFV, bit-for-bit.
        assert_eq!(
            agq1.to_bits(),
            agq7.to_bits(),
            "with n_eta = 0 the node count is irrelevant: n=1 {agq1} vs n=7 {agq7}"
        );
        // …and the marginal *is* the conditional likelihood, so AGQ = FOCEI exactly.
        assert!(
            (agq1 - focei).abs() < 1e-9,
            "with no random effects AGQ must equal FOCEI exactly: agq={agq1}, focei={focei}"
        );

        // Drive the gradient's d == 0 branch through a short optimising fit.
        let opts = FitOptions {
            method: EstimationMethod::Laplace,
            n_agq: 3,
            outer_maxiter: 3,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let res = fit(&model, &pop, &model.default_params, &opts)
            .expect("a fixed-effects AGQ fit must return Ok");
        assert!(
            res.ofv.is_finite(),
            "fixed-effects AGQ OFV must be finite: {}",
            res.ofv
        );
    }
}

// ---------------------------------------------------------------------------
// `method = laplace` — AGQ at one node, exposed as a first-class method (#251)
// ---------------------------------------------------------------------------

/// **Laplace is the exact-anchor quadrature at one node, and a genuinely different estimator
/// from FOCEI.** `method = laplace` with `n_agq = 1` uses the exact Hessian; FOCEI uses the
/// Gauss-Newton one. If they ever coincide, the exact Hessian has been lost.
#[test]
fn laplace_one_node_is_not_focei() {
    let pop = warfarin();
    let laplace = eval_only_ofv(&pop, EstimationMethod::Laplace, 1);
    let focei = eval_only_ofv(&pop, EstimationMethod::FoceI, 1);
    assert!(
        (laplace - focei).abs() > 1e-6,
        "LAPLACE (exact Hessian) must not collapse onto FOCEI (Gauss-Newton Hessian): \
         laplace={laplace}, focei={focei}"
    );
}

/// `n_agq` is now an **argument** to Laplace, not pinned (#251): `n_agq = 1` is Laplace,
/// `n_agq > 1` is adaptive Gauss–Hermite quadrature over the same integral. So varying it must
/// change the OFV (the quadrature refines the marginal the one-point rule approximates), and
/// `n_agq` must be one of the method's keys.
#[test]
fn laplace_respects_n_agq() {
    let pop = warfarin();
    let a = eval_only_ofv(&pop, EstimationMethod::Laplace, 1);
    let b = eval_only_ofv(&pop, EstimationMethod::Laplace, 7);
    assert_ne!(
        a.to_bits(),
        b.to_bits(),
        "method = laplace must now vary with n_agq (1 = Laplace, 7 = a 7-node quadrature of \
         the same marginal): {a} vs {b}"
    );
    assert!(
        ferx_core::types::method_specific_keys(EstimationMethod::Laplace).contains(&"n_agq"),
        "`n_agq` must be an option of method = laplace"
    );
}

#[test]
fn laplace_method_parses() {
    for token in ["laplace", "LAPLACE", "laplacian"] {
        let opts = options_of(&with_fit_options(&format!("  method = {token}")));
        assert_eq!(
            opts.method,
            EstimationMethod::Laplace,
            "`{token}` must select LAPLACE"
        );
    }
    assert_eq!(EstimationMethod::Laplace.label(), "LAPLACE");
}

// ---------------------------------------------------------------------------
// IOV — AGQ and Laplace integrate the joint (η, κ) marginal (#251)
// ---------------------------------------------------------------------------

/// Warfarin with a per-occasion κ on CL. 3 η + 1 κ; the dataset has 2 occasions, so the
/// stacked integration variable is `b = [η_CL, η_V, η_KA, κ¹, κ²]` — dimension 5.
const WARFARIN_IOV_SRC: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = laplace
  iov_column = OCC
  covariance = false
";

fn warfarin_iov() -> Population {
    read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load")
}

/// Eval-only OFV on the IOV model under `method`.
fn iov_ofv(method: EstimationMethod, n_agq: usize) -> f64 {
    let parsed = parse_full_model(WARFARIN_IOV_SRC).expect("IOV model must parse");
    let opts = FitOptions {
        method,
        methods: vec![],
        interaction: method == EstimationMethod::FoceI,
        n_agq,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options
    };
    let r = fit(
        &parsed.model,
        &warfarin_iov(),
        &parsed.model.default_params,
        &opts,
    )
    .expect("IOV eval-only fit must succeed");
    assert!(r.ofv.is_finite(), "IOV OFV must be finite, got {}", r.ofv);
    r.ofv
}

/// **AGQ integrates the *joint* (η, κ) marginal under IOV — it does not silently drop κ.**
///
/// The integration variable becomes the stacked `b = [η, κ₁ … κ_K]`, whose prior is the
/// block-diagonal `Ω ⊕ Ω_iov^⊕K`. That is exactly what `individual_nll_iov` already scores,
/// so every AGQ formula carries over with `d` the stacked dimension.
///
/// The check that this is really happening: the κ prior contributes `K·log|Ω_iov|` and the
/// κ's own quadratic form to the integrand, so an η-only marginal would give a *different*
/// number. Ω_iov here is 0.01 (a tight prior), so `log|Ω_iov|` is large and negative — an
/// implementation that ignored κ could not land anywhere near the IOV-aware objective.
#[test]
fn agq_and_laplace_integrate_the_joint_iov_marginal() {
    let laplace = iov_ofv(EstimationMethod::Laplace, 1);
    let agq3 = iov_ofv(EstimationMethod::Laplace, 3);
    let focei = iov_ofv(EstimationMethod::FoceI, 1);
    // The Gauss-Newton-anchored quadrature under IOV: exercises `anchor_hessian`'s stacked
    // (η, κ) `subject_sensitivities_iov` branch.
    let focei3 = iov_ofv(EstimationMethod::FoceI, 3);

    // Laplace-family: within a Hessian approximation of FOCEI-IOV, which uses the same joint
    // mode but the Gauss-Newton curvature. Nowhere near it would mean κ was mishandled.
    assert!(
        (laplace - focei).abs() < 5.0,
        "Laplace-IOV must be in the same family as FOCEI-IOV: laplace={laplace}, focei={focei}"
    );

    // Refining the grid over the *stacked* vector must move the answer (the joint posterior
    // is not exactly Gaussian) but stay in the same basin — for both anchors.
    assert!(
        agq3.is_finite() && (agq3 - laplace).abs() < 5.0,
        "Laplace-IOV n=3 must refine, not diverge: n1={laplace}, n3={agq3}"
    );
    assert!(
        focei3.is_finite() && (focei3 - focei).abs() < 5.0,
        "FOCEI-IOV n=3 (GN-anchored quadrature) must refine, not diverge: n1={focei}, n3={focei3}"
    );
}

/// The grid is `n_agq^d` with `d = n_eta + K·n_kappa`, and `K` lives in the *data* — so the
/// cap has to be enforced where the population is visible, not at model-check time. A node
/// count that is fine without IOV can be intractable with it.
///
/// `method = laplace` is exempt by construction: one node, whatever `d` is.
#[test]
fn agq_iov_grid_cap_counts_the_stacked_dimension() {
    // 3 η + 1 κ × 2 occasions = d 5. At n_agq = 21 that is 21^5 ≈ 4.1M nodes — over the cap.
    let parsed = parse_full_model(WARFARIN_IOV_SRC).expect("parses");
    let opts = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: 21,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let err = fit(
        &parsed.model,
        &warfarin_iov(),
        &parsed.model.default_params,
        &opts,
    )
    .expect_err("an intractable IOV grid must be rejected");
    assert!(
        err.contains("stacked") || err.contains("occasions"),
        "the rejection must explain that IOV raised the dimension: {err}"
    );

    // …and Laplace sails through the same model: one node regardless of the dimension.
    assert!(
        iov_ofv(EstimationMethod::Laplace, 1).is_finite(),
        "method = laplace must always be tractable under IOV — its grid is a single point"
    );
}

/// **The analytic gradient must be exact under IOV too.**
///
/// Under IOV the θ/σ score comes from the *stacked* sensitivity provider
/// (`subject_sensitivities_iov`, which walks (θ, η, κ) duals), the Ω block from the η-prior,
/// and — the only genuinely new piece — the **Ω_iov block** from the κ prior
/// `½(Σ_k κ_kᵀΩ_iov⁻¹κ_k + K·log|Ω_iov|)`, whose derivative is
/// `½(−Σ_k z_k z_kᵀ + K·Ω_iov⁻¹)` summed over the K occasions that *share* one Ω_iov.
///
/// That shared-block sum is easy to get wrong (forget the `K·Ω_iov⁻¹`, or sum over the wrong
/// axis, and the ω_iov gradient is silently off). Differencing the real objective catches it.
#[test]
fn analytic_gradient_matches_fd_under_iov() {
    use ferx_core::estimation::agq;
    use ferx_core::estimation::parameterization::{pack_params, unpack_params};

    let parsed = parse_full_model(WARFARIN_IOV_SRC).expect("IOV model must parse");
    let model = &parsed.model;
    let pop = warfarin_iov();
    assert!(model.n_kappa > 0, "test model must carry kappa");

    let template = &model.default_params;
    let x0 = pack_params(template);
    let np = x0.len();

    // Objective at a packed point: re-solve the joint (η, κ) EBEs, then the AGQ OFV.
    //
    // The inner solve must be *tightly* converged (1e-12, not the 1e-10 default). This is the
    // FD reference, and its noise floor is set by how precisely the joint (η, κ) mode is found
    // — a loosely-solved mode makes the reference, not the gradient, the inaccurate side.
    let ofv = |x: &[f64], n: usize| -> f64 {
        let params = unpack_params(x, template);
        let (ehs, _h, _s, kaps) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
            model, &pop, &params, 500, 1e-12, None, None, 0, 0,
        );
        2.0 * agq::agq_population_nll(
            model,
            &pop,
            &params,
            &ehs,
            &kaps,
            n,
            ferx_core::types::HessianAnchor::Exact,
        )
    };

    for n in [1usize, 3] {
        let params = unpack_params(&x0, template);
        let (ehs, _h, _s, kaps) = ferx_core::estimation::inner_optimizer::run_inner_loop_warm(
            model, &pop, &params, 500, 1e-12, None, None, 0, 0,
        );
        assert!(
            kaps.iter().any(|k| !k.is_empty()),
            "the inner loop must return per-occasion kappas, else this proves nothing"
        );

        let g = agq::agq_population_gradient(
            model,
            &pop,
            &params,
            template,
            &x0,
            &ehs,
            &kaps,
            n,
            ferx_core::types::HessianAnchor::Exact,
        )
        .expect("IOV gradient must be available");

        // FD step 1e-4, NOT the 1e-5 that *looks* more accurate. The reference differences a
        // **reconverged** objective, so shrinking the step past the inner solver's own noise
        // floor amplifies that noise instead of reducing truncation. Measured on this model:
        // 1.4e-3 at h = 1e-5, settling to 2.4e-4 at 1e-4 and staying there through 1e-3 — i.e.
        // the *reference* was the inaccurate side, not the gradient. Mistaking one for the
        // other made this gradient look 10x worse than it is; hence this comment.
        let mut fd = vec![0.0f64; np];
        for i in 0..np {
            let h = 1e-4 * (1.0 + x0[i].abs());
            let (mut xp, mut xm) = (x0.clone(), x0.clone());
            xp[i] += h;
            xm[i] -= h;
            fd[i] = (ofv(&xp, n) - ofv(&xm, n)) / (2.0 * h);
        }

        let scale = fd
            .iter()
            .chain(g.iter())
            .fold(1e-6f64, |m, v| m.max(v.abs()));
        let max_diff = g
            .iter()
            .zip(fd.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        // Measured: 2.4e-4 — the same order as the non-IOV case, not a degraded one. The Ω_iov
        // block (the genuinely new algebra here) is exact; a wrong one — a dropped
        // `K·Ω_iov⁻¹`, a mis-summed occasion axis — misses by tens of percent.
        assert!(
            max_diff / scale < 1e-3,
            "n_agq={n} (IOV): analytic gradient must match FD of the objective, got {:.3e} \
             relative error\n  analytic: {g:?}\n  fd:       {fd:?}",
            max_diff / scale
        );
    }
}

/// **AGQ must be able to *declare* convergence, not just reach it.**
///
/// AGQ's gradient is exact but **finite-difference-limited** — the grid-response term and the
/// posterior Hessian are both central differences, so it carries a noise floor around 1e-4
/// relative. The gradient-optimizer path historically set NLopt's stops to `1e-12`, which is
/// *unreachable* for such a gradient: L-BFGS keeps stepping until the true gradient drops
/// under the floor, at which point the search direction is noise, the line search cannot find
/// a decrease, and NLopt returns a bare `NLOPT_FAILURE`.
///
/// The fit was fine (the engine restores the best-seen point) but was reported as **not
/// converged** — a lie about a result that had been flat to 8 significant figures for a dozen
/// evaluations, and one paid for with ~40% wasted wall-clock grinding past the plateau.
///
/// So AGQ stops on the reachable objective-change criterion instead. This pins that a plain,
/// well-behaved AGQ fit actually reports `converged`.
#[test]
fn agq_reports_convergence_rather_than_grinding_into_nlopt_failure() {
    let model = parse_model_string(WARFARIN_SRC).expect("model must parse");
    let pop = warfarin();

    for (method, n) in [
        (EstimationMethod::Laplace, 1usize),
        (EstimationMethod::Laplace, 3),
    ] {
        let opts = FitOptions {
            method,
            methods: vec![],
            n_agq: n,
            outer_maxiter: 500,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let r = fit(&model, &pop, &model.default_params, &opts).expect("fit must succeed");
        assert!(
            r.converged,
            "{method:?} (n_agq={n}) must report convergence, not grind into NLOPT_FAILURE \
             against an unreachable 1e-12 gradient stop. OFV = {}",
            r.ofv
        );
    }
}
