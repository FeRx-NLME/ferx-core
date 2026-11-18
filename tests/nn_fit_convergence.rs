//! Fit-to-convergence validation for `[covariate_nn]` models.
//!
//! Unlike `tests/nn_fit_smoke.rs` (which runs `maxiter = 5` and just
//! asserts the pipeline doesn't crash), this test runs FOCEI long enough
//! to see real progress and asserts that:
//!
//! 1. The OFV decreases substantially from its initial value.
//! 2. Per-subject predicted concentrations stay within a generous
//!    multiplicative tolerance of the observed values (a fitted NN
//!    shouldn't be wildly off).
//! 3. The NN weights moved from their Glorot init.
//!
//! It's NOT a recovery test (simulate from known weights, fit, recover) —
//! NN weights are non-identifiable in general, so recovery would require
//! either parameter-space distance metrics under permutation symmetry
//! (out of scope) or output-space comparison (less informative than
//! comparing predictions to real observations, which is what we do here).
//!
//! Gated with `slow-tests` per AGENTS.md. Wall time ~30–60s for `maxiter
//! = 50` on the warfarin two-cpt covariate dataset.
//!
//! Run via:
//!   RUSTFLAGS="-Z autodiff=Enable" cargo test --release \
//!     --features nn,slow-tests --test nn_fit_convergence

#![cfg(feature = "nn")]

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, read_nonmem_csv};

/// Same shape as `examples/warfarin_dcm.ferx` but with `maxiter = 50` so
/// the test stays under a minute on a modern laptop. A smaller hidden
/// width too (3 instead of 8 each layer) to keep the optimizer's job
/// tractable in 50 iterations.
const MODEL: &str = r#"
[parameters]
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.08
  omega ETA_V2 ~ 0.08
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs     = [WT, CRCL]
  outputs    = [CL, V1, Q, V2, KA]
  layers     = [3]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V1 = TYPICAL_PK.V1 * exp(ETA_V1)
  Q  = TYPICAL_PK.Q  * exp(ETA_Q)
  V2 = TYPICAL_PK.V2 * exp(ETA_V2)
  KA = TYPICAL_PK.KA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  maxiter = 50
  covariance = false
"#;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn focei_makes_substantial_progress_on_nn_model() {
    let parsed = parse_full_model(MODEL).expect("model parses with --features nn");
    let model = parsed.model;
    let options = parsed.fit_options;

    let population = read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        Some(&["WT", "CRCL"]),
        None,
    )
    .expect("dataset loads");

    let initial_theta = model.default_params.theta.clone();

    // Run a 1-iteration "fit" to capture the OFV at the Glorot-initialised
    // weights. This is the cleanest way to get a baseline using only the
    // public API; `foce_subject_nll` is internal and its signature shifts
    // with refactors.
    let mut baseline_opts: FitOptions = options.clone();
    baseline_opts.outer_maxiter = 1;
    let baseline = fit(&model, &population, &model.default_params, &baseline_opts)
        .expect("baseline fit (maxiter=1) runs");
    let baseline_ofv = baseline.ofv;
    assert!(
        baseline_ofv.is_finite(),
        "baseline OFV at Glorot init must be finite (got {baseline_ofv})"
    );

    let result =
        fit(&model, &population, &model.default_params, &options).expect("fit runs to completion");

    // ── Assertion 1: OFV decreased substantially.
    //
    // The Glorot-init NN starts somewhere reasonable but far from the
    // data, so 50 FOCEI iterations should knock a meaningful chunk off
    // the OFV. "Substantially" = at least 10% better than the maxiter=1
    // baseline (loose tolerance because NN fits don't always converge
    // in 50 iterations; we just want to catch "no progress at all").
    let improvement_pct = (baseline_ofv - result.ofv) / baseline_ofv.abs() * 100.0;
    assert!(
        result.ofv < baseline_ofv,
        "OFV must decrease from baseline {:.2} (got {:.2})",
        baseline_ofv,
        result.ofv
    );
    assert!(
        improvement_pct > 10.0,
        "expected OFV to improve by >10% from baseline {:.2} to <{:.2}, got {:.2} ({:.1}% improvement)",
        baseline_ofv,
        baseline_ofv * 0.9,
        result.ofv,
        improvement_pct
    );

    // ── Assertion 2: NN weights moved meaningfully.
    //
    // A "meaningful" move = the L2 distance from init to final, divided by
    // n_weights, exceeds a small threshold. If all weights stayed at init,
    // this number would be 0; even a few light gradient steps push it well
    // above 1e-3.
    let nn = &model.covariate_nns[0];
    use ferx_core::nn::CovariateMapper;
    let n_w = nn.mapper.n_weights();
    let mut sq_dist = 0.0_f64;
    for i in 0..n_w {
        let idx = nn.weights_offset + i;
        let d = result.theta[idx] - initial_theta[idx];
        sq_dist += d * d;
    }
    let rms_move = (sq_dist / n_w as f64).sqrt();
    assert!(
        rms_move > 1e-3,
        "NN weights barely moved (RMS step = {rms_move:.6}); optimizer may not be \
         touching them as live parameters"
    );

    // ── Assertion 3: Predictions stay within a generous tolerance of
    // observations. Loose because NNs find non-unique weight sets that
    // can produce similar predictions; we just want to catch
    // "concentration predicted as 1000× observed" garbage.
    let mut n_within = 0_usize;
    let mut n_total = 0_usize;
    for sr in &result.subjects {
        for (j, &ipred) in sr.ipred.iter().enumerate() {
            if !ipred.is_finite() {
                continue;
            }
            let obs = population
                .subjects
                .iter()
                .find(|s| s.id == sr.id)
                .and_then(|s| s.observations.get(j))
                .copied();
            let Some(obs) = obs else {
                continue;
            };
            if obs <= 0.0 {
                continue;
            }
            n_total += 1;
            // Within a factor of 5 (very loose). Tight enough to flag a
            // catastrophic fit, loose enough to tolerate the slow-convergence
            // reality of a 50-iteration NN fit.
            let ratio = ipred / obs;
            if (0.2..=5.0).contains(&ratio) {
                n_within += 1;
            }
        }
    }
    let frac = n_within as f64 / n_total as f64;
    assert!(
        frac >= 0.6,
        "only {:.0}% of predictions are within 0.2×–5× of obs ({}/{}); fit looks broken",
        frac * 100.0,
        n_within,
        n_total
    );

    // Print a one-line summary so a passing run is informative in the
    // test log (cargo test --no-capture).
    eprintln!(
        "\nNN fit summary: baseline OFV {:.2} -> {:.2} ({:.1}% better) | \
         RMS weight step {:.4} | pred-vs-obs in [0.2x, 5x]: {:.0}% ({}/{})",
        baseline_ofv,
        result.ofv,
        improvement_pct,
        rms_move,
        frac * 100.0,
        n_within,
        n_total,
    );
}
