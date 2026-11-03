//! #2531: the per-subject non-Gaussian sums iterate `model.endpoints` in ascending CMT
//! order on every compile, so the objective is bit-reproducible.
//!
//! Each test pins the sum to the ascending-CMT fold of the per-endpoint terms (each from a
//! compile holding that endpoint alone, through the same code path), recompiling the full
//! model `COMPILES` times, and asserts the straddle: some subject's terms give other bits
//! in another order, so a walk in map order cannot pass.

use super::*;
use crate::survival::endpoint_order_test_helpers::{
    fold, order_sensitive, EndpointOrderFixture, COMPILES, ETA,
};

/// `accumulate_non_gaussian_nll` adds one `factor · raw` per TTE endpoint into `acc`, in
/// ascending CMT order, on every compile — for the three-cause model from `acc = 0`, and
/// for the two-cause model from a non-zero `acc` (a Gaussian term already in `data_ll`).
///
/// Two causes from `acc = 0` would be a tautology — `(0 + a) + b == (0 + b) + a` — which
/// is why the two-cause leg starts at `0.123456789` and asserts its own straddle.
///
/// Mutation: iterate a per-call `std::collections::HashMap` copy of `model.endpoints` in
/// `accumulate_non_gaussian_nll`.
#[test]
fn accumulate_non_gaussian_nll_adds_endpoints_in_ascending_cmt_order() {
    const FACTOR: f64 = 2.0;
    for (label, fx, acc0) in [
        ("three-cause", EndpointOrderFixture::three_cause(), 0.0),
        ("two-cause", EndpointOrderFixture::two_cause(), 0.123456789),
    ] {
        let cmts = fx.cmts();
        let (m, pop) = fx.compile();
        let theta = m.default_params.theta.clone();
        let eta = &ETA[..m.n_eta];
        let singles: Vec<_> = cmts.iter().map(|&c| fx.compile_only(c)).collect();
        // Each endpoint's `factor · raw`: from `acc = 0`, `0 + factor·raw` is exact.
        let terms: Vec<Vec<f64>> = pop
            .subjects
            .iter()
            .map(|s| {
                singles
                    .iter()
                    .map(|ms| {
                        let mut a = 0.0;
                        accumulate_non_gaussian_nll(ms, s, &theta, eta, None, FACTOR, &mut a);
                        a
                    })
                    .collect()
            })
            .collect();
        assert!(
            terms.iter().flatten().all(|t| t.is_finite()),
            "{label}: {terms:?}"
        );
        assert!(
            terms.iter().any(|t| order_sensitive(acc0, t)),
            "{label}: no subject's sum depends on the order: the pin cannot fail"
        );

        for k in 0..COMPILES {
            let (m, pop) = fx.compile();
            for (s, t) in pop.subjects.iter().zip(&terms) {
                let want = fold(acc0, t);
                let mut got = acc0;
                accumulate_non_gaussian_nll(&m, s, &theta, eta, None, FACTOR, &mut got);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{label}, compile {k}, subject {}: {got:e} is not the ascending-CMT sum {want:e}",
                    s.id
                );
            }
        }
    }
}

/// `non_gaussian_mode_terms` — the FOCE/FOCEI marginal's data term `q` and curvature
/// `∇²q` at the mode — sums its TTE endpoints in ascending CMT order, both the NLL and
/// the η-Hessian, on every compile of the three-cause model.
///
/// Mutation: iterate a per-call `std::collections::HashMap` copy of `model.endpoints` in
/// `non_gaussian_mode_terms`.
#[test]
fn non_gaussian_mode_terms_sum_endpoints_in_ascending_cmt_order() {
    let fx = EndpointOrderFixture::three_cause();
    let cmts = fx.cmts();
    let (m, pop) = fx.compile();
    let theta = m.default_params.theta.clone();
    let omega = m.default_params.omega.matrix.clone();
    let eta_hat = DVector::from_column_slice(&ETA);
    let singles: Vec<_> = cmts.iter().map(|&c| fx.compile_only(c)).collect();
    // Per subject, per endpoint: (nll, H[0,0]) of that endpoint alone.
    let terms: Vec<(Vec<f64>, Vec<f64>)> = pop
        .subjects
        .iter()
        .map(|s| {
            singles
                .iter()
                .map(|ms| {
                    let (q, h) =
                        non_gaussian_mode_terms(ms, s, &theta, &eta_hat, None, &omega, &mut None);
                    assert_eq!(h.shape(), (1, 1));
                    (q, h[(0, 0)])
                })
                .unzip()
        })
        .collect();
    assert!(
        terms
            .iter()
            .flat_map(|(q, h)| q.iter().chain(h))
            .all(|x| x.is_finite()),
        "{terms:?}"
    );
    assert!(
        terms.iter().any(|(q, _)| order_sensitive(0.0, q)),
        "no subject's NLL depends on the order: the pin cannot fail"
    );
    assert!(
        terms.iter().any(|(_, h)| order_sensitive(0.0, h)),
        "no subject's Hessian depends on the order: the pin cannot fail"
    );

    for k in 0..COMPILES {
        let (m, pop) = fx.compile();
        for (s, (tq, th)) in pop.subjects.iter().zip(&terms) {
            let (q, h) = non_gaussian_mode_terms(&m, s, &theta, &eta_hat, None, &omega, &mut None);
            let (want_q, want_h) = (fold(0.0, tq), fold(0.0, th));
            assert_eq!(
                (q.to_bits(), h[(0, 0)].to_bits()),
                (want_q.to_bits(), want_h.to_bits()),
                "compile {k}, subject {}: (q, H) = ({q:e}, {:e}), ascending-CMT sums ({want_q:e}, {want_h:e})",
                s.id,
                h[(0, 0)]
            );
        }
    }
}
