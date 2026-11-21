//! End-to-end smoke test for the NN module that doubles as a usage example.
//!
//! Shows how a future `pk_param_fn` will use [`NamedMlpMapper`] to translate
//! `(weights, covariates) -> PkParams`. The parser hookup in subsequent PRs
//! will build the same closure automatically from a `[covariate_nn]` block;
//! this test exercises the trait-level API directly so reviewers can see
//! exactly what the parser will be producing.
//!
//! Run with: `RUSTFLAGS="-Z autodiff=Enable" cargo test --release \
//!            --features nn --test nn_dcm_smoke`.

#![cfg(feature = "nn")]

use std::collections::HashMap;

use ferx_core::nn::{Activation, CovariateMapper, MlpMapper, NamedMlpMapper};
use ferx_core::types::PkParams;

/// Build a tiny DCM-style mapper: 2 covariates -> 4 hidden (tanh) -> 5 PK
/// params (softplus head). Matches the shape sketched in
/// `examples/drafts/warfarin_dcm.ferx`.
fn build_mapper() -> NamedMlpMapper {
    let mlp = MlpMapper::new(vec![2, 4, 5], Activation::Tanh, Activation::Softplus)
        .expect("layer shape valid");
    NamedMlpMapper::new(
        mlp,
        vec!["WT".into(), "CRCL".into()],
        vec!["CL".into(), "V1".into(), "Q".into(), "V2".into(), "KA".into()],
    )
    .expect("output names map to PK slots")
}

/// Approximate Glorot-uniform-ish initial weights — deterministic, no rand
/// dep needed for a test. Small-magnitude weights keep tanh active and
/// softplus output bounded.
fn deterministic_init(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let s = ((i * 31 + 7) as f64).sin();
            let c = ((i * 17 + 3) as f64).cos();
            0.15 * (s + 0.5 * c)
        })
        .collect()
}

fn warfarin_covariates() -> HashMap<String, f64> {
    let mut cov = HashMap::new();
    cov.insert("WT".into(), 70.0);
    cov.insert("CRCL".into(), 95.0);
    cov
}

#[test]
fn mapper_produces_positive_pk_params_for_realistic_covariates() {
    let mapper = build_mapper();
    let weights = deterministic_init(mapper.n_weights());
    let cov = warfarin_covariates();

    let mut out = PkParams::default();
    mapper.forward(&weights, &cov, &mut out).unwrap();

    // The softplus head guarantees strictly positive outputs — the central
    // physical-plausibility constraint for any PK parameter.
    assert!(out.cl() > 0.0, "CL must be > 0, got {}", out.cl());
    assert!(out.v() > 0.0, "V1 must be > 0, got {}", out.v());
    assert!(out.q() > 0.0, "Q must be > 0, got {}", out.q());
    assert!(out.v2() > 0.0, "V2 must be > 0, got {}", out.v2());
    assert!(out.ka() > 0.0, "KA must be > 0, got {}", out.ka());

    // F was not in the output list — it must stay at its PkParams::default value of 1.0.
    assert_eq!(out.f_bio(), 1.0);
}

#[test]
fn mapper_jacobian_consistent_with_forward_under_perturbation() {
    // For each weight, perturb by eps and check that the linearised forward
    // change matches the Jacobian column. This is the "live" version of the
    // unit test inside src/nn/mod.rs; the value is that it exercises the
    // full trait API through ferx-core's public surface.
    let mapper = build_mapper();
    let weights = deterministic_init(mapper.n_weights());
    let cov = warfarin_covariates();

    let mut base = PkParams::default();
    mapper.forward(&weights, &cov, &mut base).unwrap();
    let jac = mapper.jacobian(&weights, &cov).unwrap();
    assert_eq!(jac.nrows(), 5);
    assert_eq!(jac.ncols(), mapper.n_weights());

    let eps = 1e-7;
    // Spot-check the first few weights — the unit test covers exhaustive FD.
    for j in 0..std::cmp::min(10, mapper.n_weights()) {
        let mut w_plus = weights.clone();
        let mut w_minus = weights.clone();
        w_plus[j] += eps;
        w_minus[j] -= eps;
        let mut out_plus = PkParams::default();
        let mut out_minus = PkParams::default();
        mapper.forward(&w_plus, &cov, &mut out_plus).unwrap();
        mapper.forward(&w_minus, &cov, &mut out_minus).unwrap();

        for (i, pk_idx) in [
            ferx_core::types::PK_IDX_CL,
            ferx_core::types::PK_IDX_V,
            ferx_core::types::PK_IDX_Q,
            ferx_core::types::PK_IDX_V2,
            ferx_core::types::PK_IDX_KA,
        ]
        .into_iter()
        .enumerate()
        {
            let fd = (out_plus.values[pk_idx] - out_minus.values[pk_idx]) / (2.0 * eps);
            let analytic = jac[(i, j)];
            // Looser tolerance than the inline unit test because Softplus
            // and tanh compose into a more numerically sensitive path.
            let abs_diff = (fd - analytic).abs();
            let rel_diff = abs_diff / (fd.abs().max(1e-12));
            assert!(
                abs_diff < 1e-5 || rel_diff < 1e-4,
                "jac[{},{}] = {} vs FD {} (abs {}, rel {})",
                i,
                j,
                analytic,
                fd,
                abs_diff,
                rel_diff
            );
        }
    }
}

/// Demonstrate the M2 mu-ref composition shape: NN output × exp(eta). This
/// is the call pattern the parser will eventually generate from
/// `CL = TYPICAL_PK.CL * exp(ETA_CL)` in `[individual_parameters]`.
#[test]
fn composition_with_eta_gives_expected_individual_parameters() {
    let mapper = build_mapper();
    let weights = deterministic_init(mapper.n_weights());
    let cov = warfarin_covariates();

    let mut tv = PkParams::default();
    mapper.forward(&weights, &cov, &mut tv).unwrap();

    // Per-output etas. For mu-referenced lognormal IIV the individual
    // parameter is tv * exp(eta); at eta=0 it equals the typical value.
    let etas: [f64; 5] = [0.1, -0.2, 0.0, 0.05, -0.1];

    let mut indiv = tv;
    for (i, pk_idx) in [
        ferx_core::types::PK_IDX_CL,
        ferx_core::types::PK_IDX_V,
        ferx_core::types::PK_IDX_Q,
        ferx_core::types::PK_IDX_V2,
        ferx_core::types::PK_IDX_KA,
    ]
    .into_iter()
    .enumerate()
    {
        indiv.values[pk_idx] = tv.values[pk_idx] * etas[i].exp();
    }

    // Sanity: positive eta increases CL, negative eta decreases V1.
    assert!(indiv.cl() > tv.cl());
    assert!(indiv.v() < tv.v());
    // Q has eta=0 -> individual == typical value.
    assert_eq!(indiv.q(), tv.q());
}
