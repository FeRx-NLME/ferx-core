//! #2531: [`TteTerm`] sums its endpoints in ascending CMT order on every compile.

use super::endpoint_order_test_helpers::{
    fold, order_sensitive, EndpointOrderFixture, COMPILES, ETA,
};
use super::TteTerm;

/// `TteTerm::new(..).eval_shared::<f64>` — the dual walk's per-subject TTE sum — is the
/// ascending-CMT sum of the per-endpoint terms, bit for bit, on every one of `COMPILES`
/// compiles of the three-cause model.
///
/// Straddle: some subject's three terms give other bits in another order, so a walk in
/// map order cannot pass. Mutation: build `endpoints` from a per-call
/// `std::collections::HashMap` copy of `model.endpoints` in `TteTerm::new`.
#[test]
fn tte_term_sums_endpoints_in_ascending_cmt_order() {
    let fx = EndpointOrderFixture::three_cause();
    let cmts = fx.cmts();
    assert_eq!(cmts, vec![2, 3, 4]);
    let (m, pop) = fx.compile();
    let theta = m.default_params.theta.clone();

    // Per subject, each endpoint's term alone (ascending CMT).
    let singles: Vec<_> = cmts.iter().map(|&c| fx.compile_only(c)).collect();
    let terms: Vec<Vec<f64>> = pop
        .subjects
        .iter()
        .map(|s| {
            singles
                .iter()
                .map(|ms| {
                    TteTerm::new(ms, s)
                        .eval_shared::<f64>(&theta, &ETA, None)
                        .expect("term")
                })
                .collect()
        })
        .collect();
    assert!(terms.iter().flatten().all(|t| t.is_finite()), "{terms:?}");
    assert!(
        terms.iter().any(|t| order_sensitive(0.0, t)),
        "no subject's sum depends on the order: the pin cannot fail"
    );

    for k in 0..COMPILES {
        let (m, pop) = fx.compile();
        for (s, t) in pop.subjects.iter().zip(&terms) {
            let want = fold(0.0, t);
            let got = TteTerm::new(&m, s)
                .eval_shared::<f64>(&theta, &ETA, None)
                .expect("term");
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "compile {k}, subject {}: {got:e} is not the ascending-CMT sum {want:e}",
                s.id
            );
        }
    }
}
