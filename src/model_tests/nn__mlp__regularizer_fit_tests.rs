// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::nn::Activation;
use ferx_model::types::PkParams;
use nalgebra::DMatrix;
use nalgebra::DVector;
use std::collections::HashMap;

use approx::assert_relative_eq;

use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::{CompiledModel, FitOptions, Population};

use ferx_model::nn::mlp::{CovariateMapper, NnRegularizer};

/// Two-cpt oral DCM with a small MLP mapping (WT, CRCL) → the five PK params.
/// One file shared with `tests/nn_regularization.rs` and the trust-region
/// end-to-end check (`include_str!` is visible from both `tests/` and
/// `#[cfg(test)]` code), so the fixtures cannot drift. The notes in it on why
/// `center` / `scale` (and `init`) are load-bearing apply to the λ = 0 baseline
/// of [`l2_shrinks_weights_and_modulator_variation`] in particular. Only that
/// slow fit reads the declared start; every per-PR test here uses [`load_bare`].
#[cfg(feature = "slow-tests")]
const MODEL: &str = include_str!("../../tests/fixtures/two_cpt_dcm_regularized.ferx");

#[cfg(feature = "slow-tests")]
fn load() -> (CompiledModel, FitOptions, Population) {
    load_text(MODEL)
}

/// The fixture without its `init` line; see [`super::dcm_fixture_without_init`].
fn load_bare() -> (CompiledModel, FitOptions, Population) {
    load_text(&ferx_model::nn::mlp::dcm_fixture_without_init())
}

fn load_text(text: &str) -> (CompiledModel, FitOptions, Population) {
    let parsed = parse_full_model(text).expect("model parses with --features nn");
    let population = read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        Some(&["WT", "CRCL"]),
        None,
    )
    .expect("dataset loads");
    (parsed.model, parsed.fit_options, population)
}

/// Variance of the NN's CL modulator (output 0) across subjects, evaluated at
/// a given theta. Higher = more covariate-driven spread in the learned curve.
#[cfg(feature = "slow-tests")]
fn cl_modulator_variance(model: &CompiledModel, population: &Population, theta: &[f64]) -> f64 {
    let nn = &model.covariate_nns[0];
    let n_w = nn.mapper.n_weights();
    let w = &theta[nn.weights_offset..nn.weights_offset + n_w];
    let cls: Vec<f64> = population
        .subjects
        .iter()
        .map(|s| nn.mapper.forward_raw(w, &s.covariates).expect("forward")[0])
        .collect();
    let n = cls.len() as f64;
    let mean = cls.iter().sum::<f64>() / n;
    cls.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / n
}

/// L2 norm of the NN's *weight matrices* (biases excluded) at a given theta —
/// the quantity `nn_l2` directly shrinks.
#[cfg(feature = "slow-tests")]
fn weight_block_sq_norm(model: &CompiledModel, theta: &[f64]) -> f64 {
    let nn = &model.covariate_nns[0];
    let w = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    nn.mapper.mlp().weight_param_indices().iter().map(|&i| w[i] * w[i]).sum()
}

/// λ = 0 must leave the regularizer inactive, its penalty exactly 0 and its
/// gradient contribution exactly 0 — the no-op guarantee that keeps
/// unregularized fits byte-identical.
#[test]
fn regularizer_lambda_zero_is_noop() {
    let (model, mut options, population) = load_bare();
    options.nn_l2_lambda = 0.0;
    options.nn_smooth_lambda = 0.0;
    let reg = NnRegularizer::build(&model, &population, &options);
    assert!(!reg.is_active(), "λ = 0 regularizer must be inactive");

    let theta = model.default_params.theta.clone();
    assert_eq!(reg.penalty_value(&theta), 0.0, "λ = 0 penalty must be 0");

    let mut grad = vec![0.0; theta.len()];
    reg.add_packed_gradient(&theta, &mut grad);
    assert!(grad.iter().all(|&g| g == 0.0), "λ = 0 gradient contribution must be exactly 0");
}

/// The analytic penalty gradient must match central finite differences of
/// `penalty_value` at λ > 0, and must touch only the NN-weight coordinates.
#[test]
fn regularizer_penalty_gradient_matches_fd() {
    let (model, mut options, population) = load_bare();
    options.nn_l2_lambda = 1e-2;
    options.nn_smooth_lambda = 1e-1;
    let reg = NnRegularizer::build(&model, &population, &options);
    assert!(reg.is_active());

    let theta = model.default_params.theta.clone();
    let mut grad = vec![0.0; theta.len()];
    reg.add_packed_gradient(&theta, &mut grad);

    // Only the NN-weight coordinates carry a penalty gradient; everything else
    // (omegas, sigma) must be untouched.
    let nn = &model.covariate_nns[0];
    let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
    for (k, &g) in grad.iter().enumerate() {
        if k < w_lo || k >= w_hi {
            assert_eq!(g, 0.0, "non-NN coordinate {k} must have zero penalty grad");
        }
    }

    // Central FD of penalty_value w.r.t. each NN weight theta.
    let eps = 1e-6;
    let mut p = theta.clone();
    for k in w_lo..w_hi {
        let saved = p[k];
        p[k] = saved + eps;
        let vp = reg.penalty_value(&p);
        p[k] = saved - eps;
        let vm = reg.penalty_value(&p);
        p[k] = saved;
        let fd = (vp - vm) / (2.0 * eps);
        let tol = 1e-6 + 1e-5 * fd.abs();
        assert!(
            (grad[k] - fd).abs() <= tol,
            "penalty grad mismatch at weight {k}: analytic {}, fd {}",
            grad[k],
            fd
        );
    }
}

/// The single-pass `penalty_and_gradient` the optimizers call must equal
/// `penalty_value` + `add_packed_gradient` exactly, and the packed Hessian
/// adder must touch NN-weight coordinates only (with the exact `2λ` L2
/// diagonal on weight entries).
#[test]
fn single_pass_and_hessian_agree_with_the_separate_calls() {
    let (model, mut options, population) = load_bare();
    options.nn_l2_lambda = 2e-2;
    options.nn_smooth_lambda = 3e-1;
    let reg = NnRegularizer::build(&model, &population, &options);
    let theta = model.default_params.theta.clone();
    let n = theta.len();

    let mut g_sep = vec![0.0; n];
    reg.add_packed_gradient(&theta, &mut g_sep);
    let v_sep = reg.penalty_value(&theta);
    let mut g_one = vec![0.0; n];
    let v_one = reg.penalty_and_gradient(&theta, &mut g_one);
    assert_relative_eq!(v_one, v_sep, max_relative = 1e-12);
    for (a, b) in g_one.iter().zip(&g_sep) {
        assert_relative_eq!(a, b, max_relative = 1e-12);
    }

    let nn = &model.covariate_nns[0];
    let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
    let mut h = vec![vec![0.0; n]; n];
    reg.add_packed_hessian(&theta, &mut |i, j, v| h[i][j] += v);
    for i in 0..n {
        for j in 0..n {
            let in_block = (w_lo..w_hi).contains(&i) && (w_lo..w_hi).contains(&j);
            if !in_block {
                assert_eq!(h[i][j], 0.0, "penalty Hessian leaked to ({i}, {j})");
            }
            assert_relative_eq!(h[i][j], h[j][i], epsilon = 1e-12);
        }
    }
    // L2 alone: exact diagonal on weight entries, zero on biases.
    let mut l2 = options.clone();
    l2.nn_smooth_lambda = 0.0;
    let reg_l2 = NnRegularizer::build(&model, &population, &l2);
    let mut h = vec![vec![0.0; n]; n];
    reg_l2.add_packed_hessian(&theta, &mut |i, j, v| h[i][j] += v);
    let weight_idx: std::collections::HashSet<usize> =
        nn.mapper.mlp().weight_param_indices().into_iter().map(|i| w_lo + i).collect();
    for i in 0..n {
        for j in 0..n {
            let expect = if i == j && weight_idx.contains(&i) { 2.0 * 2e-2 } else { 0.0 };
            assert_relative_eq!(h[i][j], expect, epsilon = 1e-15);
        }
    }
}

/// `add_exact_packed_hessian` — the curvature the covariance step reports (#1352) — is
/// the penalty's exact Hessian: bit-equal to `add_packed_hessian` under L2 alone, and under
/// smoothness equal to the second difference of `penalty_value` itself (an oracle outside
/// the analytic gradient it is built from), while the Gauss–Newton `add_packed_hessian`
/// is not. The straddle is asserted, so the fixture cannot drift to a point with `C ≈ 0`,
/// where the two forms agree and a revert to Gauss–Newton would pass.
#[test]
fn exact_penalty_hessian_matches_the_second_difference_of_the_penalty() {
    let (model, mut options, population) = load_bare();
    let theta = model.default_params.theta.clone();
    let n = theta.len();
    let nn = &model.covariate_nns[0];
    let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
    let dense = |f: &dyn Fn(&mut dyn FnMut(usize, usize, f64))| {
        let mut h = vec![vec![0.0; n]; n];
        f(&mut |i, j, v| h[i][j] += v);
        h
    };

    options.nn_l2_lambda = 2e-2;
    options.nn_smooth_lambda = 0.0;
    let reg = NnRegularizer::build(&model, &population, &options);
    assert!(reg.hessian_is_exact());
    assert_eq!(
        dense(&|a| reg.add_exact_packed_hessian(&theta, a)),
        dense(&|a| reg.add_packed_hessian(&theta, a)),
        "L2 only: the exact Hessian is the analytic one"
    );

    options.nn_smooth_lambda = 3e-1;
    let reg = NnRegularizer::build(&model, &population, &options);
    assert!(!reg.hessian_is_exact());
    let exact = dense(&|a| reg.add_exact_packed_hessian(&theta, a));
    let gn = dense(&|a| reg.add_packed_hessian(&theta, a));
    let eps = 1e-4;
    let f = |p: &[f64]| reg.penalty_value(p);
    let mut worst = 0.0f64;
    let mut gn_gap = 0.0f64;
    let mut p = theta.clone();
    for i in 0..n {
        for j in 0..n {
            let in_block = (w_lo..w_hi).contains(&i) && (w_lo..w_hi).contains(&j);
            if !in_block {
                assert_eq!(exact[i][j], 0.0, "exact Hessian leaked to ({i}, {j})");
                continue;
            }
            assert_eq!(exact[i][j], exact[j][i], "not symmetric at ({i}, {j})");
            let mut at = |di: f64, dj: f64| {
                p[i] += di;
                p[j] += dj;
                let v = f(&p);
                p[i] -= di;
                p[j] -= dj;
                v
            };
            let fd =
                (at(eps, eps) - at(eps, -eps) - at(-eps, eps) + at(-eps, -eps)) / (4.0 * eps * eps);
            let err = (exact[i][j] - fd).abs();
            assert!(err.is_finite(), "non-finite Hessian entry at ({i}, {j})");
            worst = worst.max(err);
            gn_gap = gn_gap.max((gn[i][j] - fd).abs());
        }
    }
    // Measured on this fixture: worst |exact − FD| = 2.6e-9 (the FD oracle's own
    // `O(eps²)` + rounding), against a Gauss–Newton gap of 5.9e-3 — the bound sits ~400×
    // above the first and ~6,000× below the second.
    assert!(worst < 1e-6, "exact penalty Hessian vs FD: worst {worst:e}");
    assert!(
        gn_gap > 100.0 * 1e-6,
        "the fixture must sit where Gauss–Newton is not exact (gap {gn_gap:e})"
    );
}

/// Growing the L2 strength shrinks the NN weights, flattening the learned covariate→modulator curve.
///
/// **The start is part of the fixture (#1561).** Without its `init` line every output starts at
/// `softplus(0) = 0.693` (true CL 5, V1 50, Q 10, V2 100, KA 1.2): the λ = 0 fit reported
/// `converged = true` at data OFV −234.14 and the λ = 1000 fit stopped at a covariate-free −1173.56,
/// both poor optima (the cross-optimality check below failed on the λ = 0 fit). With `init` declared,
/// as the `[covariate_nn]` docs say to do on every DCM, the same fits measure (macOS/arm64,
/// `ci-test`, one thread):
///
/// | λ | data OFV | ‖W‖² |
/// |---|---|---|
/// | 0 | −1252.35 | 9.7e4 |
/// | 5 | −1196.52 | 1.373 |
/// | 1000 | −1185.46 | 7.4e-12 |
///
/// The λ = 5 point is the same −1196.517 from three declared starts (this one, the simulation's true
/// values, and 40% of them). The bare start is also warned about at fit time (`W_NN_BARE_OUTPUT_START`).
///
/// `two_cpt_oral_cov` is **not** a null-covariate dataset: it was simulated with `(WT/70)^θ` on CL and
/// V1 and `(CRCL/100)^θ` on CL (`src/bin/generate_data.rs`), and this NN maps (WT, CRCL) → the typical
/// PK parameters. With every weight at zero each hidden unit is constant and the network collapses to a
/// covariate-free typical value, also a stationary point of the data objective (constant hidden units
/// zero the weight gradient), so a fit can stop there.
///
/// The test once asserted that λ = 5 *and* λ = 100 both reach a ~1e-4 norm floor, which tested which
/// basin each fit stopped in, not what L2 does, and flipped with unrelated changes (#1389, #1474) and
/// the toolchain. What is asserted is what a correct optimizer must produce:
///
/// 1. every fit converges;
/// 2. the λ = 0 fit has weights and CL-modulator spread for L2 to remove;
/// 3. **cross-optimality**: under its own λ, each fit's penalized objective is no worse than any other
///    fitted point's (a necessary condition of a minimum). Computed from the three fits alone, because
///    the reported `ofv` excludes the penalty and so does not depend on λ (verified by re-scoring one
///    θ at λ = 0, 5 and 100);
/// 4. the mid λ shrinks the weights by orders of magnitude and reduces the modulator spread, while the
///    heavy λ flattens both to the floor. `LAMBDA_BIG = 1000` is far above the λ ≈ 8 at which the
///    zero-weight point first beats the λ = 5 point ((−1185.46 + 1196.52) / 1.373).
///
/// Gated with `#[cfg]` rather than the usual Tier-3 `#[cfg_attr(not(feature = "slow-tests"), ignore =
/// "…")]`: the test lives in `src/` (measured by codecov) rather than `tests/` (ignored wholesale), and
/// an `#[ignore]`d body is compiled but never run, so on a PR its ~50 lines would read as uncovered
/// and drag the patch gate under its 90% floor. Compiling it out makes it a measurement gap, as for
/// the rest of the feature-gated surface (#293). Bit-rot is caught on every PR by
/// `cargo check --tests --no-default-features --features ci,nn,slow-tests`, and the body *runs*
/// nightly: `slow-tests.yml` passes `ferx-core/nn` alongside `ferx-core/slow-tests` (once it passed no
/// `nn`, leaving every DCM Tier-3 fit running in no CI job).
#[test]
#[cfg(feature = "slow-tests")]
fn l2_shrinks_weights_and_modulator_variation() {
    let (model, options, population) = load();

    // Every λ must run to convergence for its fitted weight norm to mean
    // anything: #1277 was three fits whose weight norms were compared without
    // anyone asking whether the optimizer had finished, and two of them had
    // not — L-BFGS quit at eval 12 with the trace still falling ~2600 OFV per
    // eval, so the "fitted" norms were wherever the stall happened to land.
    let fit_at = |lambda: f64| -> (Vec<f64>, f64) {
        let mut o = options.clone();
        o.nn_l2_lambda = lambda;
        let r = crate::fit(&model, &population, &model.default_params, &o)
            .unwrap_or_else(|e| panic!("fit at λ={lambda} failed: {e}"));
        assert!(
            r.converged,
            "fit at λ={lambda} must converge for its weight norm to be a fitted \
             quantity at all (OFV {:.4}); warnings: {:?}",
            r.ofv, r.warnings
        );
        assert!(r.ofv.is_finite(), "fit at λ={lambda}: OFV {}", r.ofv);
        (r.theta, r.ofv)
    };

    // Zero weights are the λ-optimum among the fitted points only above
    // λ ≈ 8 (see the doc comment); 1000 leaves two orders on that threshold.
    const LAMBDA_MID: f64 = 5.0;
    const LAMBDA_BIG: f64 = 1000.0;
    let lambdas = [0.0, LAMBDA_MID, LAMBDA_BIG];
    let fits: Vec<(Vec<f64>, f64)> = lambdas.iter().map(|&l| fit_at(l)).collect();
    let norms: Vec<f64> = fits.iter().map(|(t, _)| weight_block_sq_norm(&model, t)).collect();
    let vars: Vec<f64> =
        fits.iter().map(|(t, _)| cl_modulator_variance(&model, &population, t)).collect();
    let ofvs: Vec<f64> = fits.iter().map(|(_, o)| *o).collect();
    let (n0, n_mid, n_big) = (norms[0], norms[1], norms[2]);
    let (v0, v_mid, v_big) = (vars[0], vars[1], vars[2]);
    // Scientific notation: at ~1e-9 a fixed-point `{:.5}` renders every
    // regularized quantity as `0.00000`, which is how #1277's failure message
    // came to hide that the two numbers being ordered were both at the floor.
    eprintln!(
        "λ: {lambdas:?}\ndata OFV: {ofvs:?}\nweight ‖W‖²: {norms:?}\n\
         CL modulator var: {vars:?}"
    );

    // (2) The baseline must have weights and spread worth removing, or every
    // check below is vacuous. Measured: ‖W‖² 9.7e4, var 1.91.
    assert!(n0 > 1.0, "the λ=0 fit must leave a weight block for L2 to shrink (‖W‖² {n0:e})");
    const FLAT_VAR_MAX: f64 = 1e-3;
    assert!(
        v0 > 100.0 * FLAT_VAR_MAX,
        "the λ=0 fit must carry CL modulator spread for L2 to remove (var {v0:e})"
    );

    // (3) Cross-optimality. `ofv` excludes the penalty, so the penalized
    // objective of fitted point j under λ_i is `ofv_j + λ_i·‖W_j‖²`.
    // From the bare start (#1561) this failed first at i = λ0, j = λ1000: J_0 was
    // −234.14 at the λ=0 point vs −1173.56 at the λ=1000 point. A converged λ=0
    // fit can never lose to a point in its own space, so the check cannot pass on
    // a stuck baseline. Measured from the declared start, the tightest pair is
    // i = λ5, j = λ1000: −1189.65 against −1185.46, a margin of 4.2.
    let penalized = |i: usize, j: usize| ofvs[j] + lambdas[i] * norms[j];
    for i in 0..lambdas.len() {
        for j in 0..lambdas.len() {
            assert!(
                penalized(i, i) <= penalized(i, j) + 1e-3,
                "the λ={} fit is not a minimum of its own penalized objective: \
                 {:.4} at its point vs {:.4} at the λ={} point (data OFV {ofvs:?}, \
                 ‖W‖² {norms:?})",
                lambdas[i],
                penalized(i, i),
                penalized(i, j),
                lambdas[j]
            );
        }
    }

    // (4) The effect. The mid λ is asserted to shrink, not to vanish; measured
    // ‖W‖² 1.373 (7e4x below the baseline) and var 0.375 against 1.91. The heavy
    // λ flattens the network; measured ‖W‖² 7.4e-12 and var 3.5e-24, against a
    // 1e-4 / 1e-3 floor. Measured with the fixture's `init` (#1561).
    assert!(
        n_mid < 1e-2 * n0,
        "L2 at λ={LAMBDA_MID} must shrink the weights by orders of magnitude \
         (‖W‖²: {n0:e} → {n_mid:e})"
    );
    assert!(
        v_mid < v0,
        "L2 at λ={LAMBDA_MID} must reduce the CL modulator spread ({v0:e} → {v_mid:e})"
    );
    const REGULARIZED_NORM_FLOOR: f64 = 1e-4;
    assert!(
        n_big < REGULARIZED_NORM_FLOOR && v_big < FLAT_VAR_MAX,
        "L2 at λ={LAMBDA_BIG} must flatten the network (‖W‖² {n_big:e}, \
         CL modulator var {v_big:e})"
    );
}
/// Each λ must act alone. `nn_l2` on its own is the likeliest real
/// configuration, and it must not drag the curvature machinery in with it:
/// with `nn_smooth = 0` the regularizer skips building a grid entirely, and
/// the penalty and gradient must equal the pure-L2 ones. Symmetrically for
/// `nn_smooth` alone.
///
/// Without this, every test drove both λ together or both at zero, so the
/// single-λ branches in `penalty_value` / `add_packed_gradient` — and the
/// `CurvatureGrid::default()` arm of `build` — were never taken.
#[test]
fn each_lambda_acts_alone() {
    let (model, options, population) = load_bare();
    let theta = model.default_params.theta.clone();
    let nn = &model.covariate_nns[0];
    let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
    let w = &theta[w_lo..w_hi];
    let mlp = nn.mapper.mlp();

    let build = |l2: f64, smooth: f64| {
        let mut o = options.clone();
        o.nn_l2_lambda = l2;
        o.nn_smooth_lambda = smooth;
        NnRegularizer::build(&model, &population, &o)
    };
    let grad_of = |reg: &NnRegularizer| {
        let mut g = vec![0.0; theta.len()];
        reg.add_packed_gradient(&theta, &mut g);
        g
    };

    // --- L2 only: no grid is built, and the penalty is exactly the L2 one.
    let l2_only = build(1e-2, 0.0);
    assert!(l2_only.is_active());
    let expected_l2 = mlp.l2_weight_penalty_value(w, 1e-2);
    assert_relative_eq!(l2_only.penalty_value(&theta), expected_l2, max_relative = 1e-12);
    let (_v, expected_l2_g) = mlp.l2_weight_penalty(w, 1e-2);
    let g_l2 = grad_of(&l2_only);
    for (i, e) in expected_l2_g.iter().enumerate() {
        assert_relative_eq!(g_l2[w_lo + i], e, max_relative = 1e-12);
    }

    // --- Smoothness only: the penalty carries no L2 term at all.
    let smooth_only = build(0.0, 1e-1);
    assert!(smooth_only.is_active());
    let p_smooth = smooth_only.penalty_value(&theta);
    assert!(p_smooth > 0.0, "the fixture's network must have some curvature to penalize");
    let g_smooth = grad_of(&smooth_only);

    // --- Both together must equal the sum of the two acting alone: the
    // terms are additive and independent, which is what lets a caller reason
    // about one λ at a time.
    let both = build(1e-2, 1e-1);
    assert_relative_eq!(both.penalty_value(&theta), expected_l2 + p_smooth, max_relative = 1e-12);
    let g_both = grad_of(&both);
    for k in w_lo..w_hi {
        assert_relative_eq!(g_both[k], g_l2[k] + g_smooth[k], max_relative = 1e-9);
    }
}

/// The abort behind #1277's reopen, caught on a 44-eval budget at PR time (#1428; the test is #1411's,
/// ported onto the #1414 restart).
///
/// Run on the fixture **without** its `init` line ([`load_bare`]), the start this was measured from.
/// That start's λ = 0 fit begins with a scaled gradient of norm 1.7e5, so the identity-Hessian cap
/// shrinks L-BFGS's opening step by that factor and its first line search extrapolates (eleven
/// improving evaluations, 18759 → 7975) until Luksan's ten-extrapolation cap returns `Failure` on
/// eval 12. Before the restart in `outer_optimizer::resolve_mid_descent_restart` that was the whole
/// fit: OFV 7902 on Linux (3209.81 on macOS), `converged = false`, with
/// `l2_shrinks_weights_and_modulator_variation` failing on the ‖W‖² it produced. The restarted run, on
/// the 32 evals left, reaches **−229.56 at eval 44**, exactly the budget, with the restart notice and
/// the "increase maxiter" warning on the result (−540.59 with #1411's in-launch resume; −271.34 in 56
/// evals with #1414's fresh-budget restart).
///
/// `maxiter = 1` is a 44-eval budget (`outer_maxiter × (n + 1)`, n = 43), enough to recover and descend
/// past the old abort point. The Hessian-seeded inner solve may now avoid that exact line-search abort,
/// so the invariant is the outcome rather than the historical warning: the fit gets below zero and
/// stays within the requested budget. `resolve_mid_descent_restart` unit tests pin the restart dispatch.
#[test]
fn first_line_search_abort_is_avoided_or_resumed_not_reported() {
    let (model, options, population) = load_bare();
    let mut o = options.clone();
    o.nn_l2_lambda = 0.0;
    o.outer_maxiter = 1;
    let fit = crate::fit(&model, &population, &model.default_params, &o)
        .unwrap_or_else(|e| panic!("fit failed: {e}"));
    // Abort point measured at 7974.86 (best-seen) / 7901.99 (reported) on
    // Linux; zero sits hundreds below the restarted measurement and
    // thousands below the failure it exists to catch, on both platforms.
    assert!(
        fit.ofv < 0.0,
        "a restarted fit must get well below the abort point it was restarted \
         from (reported {:.3}, abort 7901.99 / 3209.81, measured restarted −540.59)",
        fit.ofv
    );
    // Both legs together are bounded by `maxiter` the way a single run is:
    // 12 spent by the stalled leg plus a restart on the remaining 32
    // (measured: exactly 44). NLopt's `maxeval` is a *soft* bound for L-BFGS
    // — `luksan/plis.c` checks it between line searches, so a leg can
    // overshoot by up to one line search (≤ ~20 evals) — and the macOS
    // trajectory is unmeasured, so the bound is set against the pre-fix
    // number rather than at the budget: #1414 as merged spent 12 + a fresh
    // 44 = 56 here, and a fresh-budget restart cannot come in under it.
    let n = crate::estimation::parameterization::pack_params(&model.default_params).len();
    let budget = o.outer_maxiter * (n + 1);
    let fresh_budget_total = 12 + budget;
    assert!(
        fit.n_iterations < fresh_budget_total,
        "the restart must run on the remaining budget, not a fresh one: {} evaluations \
         against a maxiter budget of {budget} (a fresh-budget restart spends \
         {fresh_budget_total})",
        fit.n_iterations
    );
}
