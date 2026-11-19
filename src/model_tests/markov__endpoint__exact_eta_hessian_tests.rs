// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::markov::ctmm_data_term;
use ferx_model::markov::StateObs;
use ferx_model::sens::num::NonRecRead;
use ferx_model::stats::likelihood::SUBJECT_SENTINEL_NLL;
use ferx_model::types::CompiledModel;
use ferx_model::types::EndpointLikelihood;
use ferx_model::types::GeneratorFn;
use ferx_model::types::ObsRecord;
use ferx_model::types::Subject;
use std::collections::HashMap;

use ferx_model::markov::endpoint::*;

/// The homogeneous CTMM η-Hessian (second derivative of `expm` on a dual) against the
/// stencil it replaces, to the stencil's accuracy; and the inhomogeneous (drug-driven)
/// endpoint, which has no generator program, declines to it.
#[test]
fn the_ctmm_eta_hessian_is_exact_on_a_homogeneous_generator() {
    let prep = crate::model_tests::categorical::exact_eta_hessian_tests::load(
        r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  omega ETA_A ~ 0.2
  omega ETA_B ~ 0.3
[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + ETA_A)
  transition asleep -> awake  = exp(LQ10 + 0.5 * ETA_A + ETA_B)
",
        "data/ctmm_2state.csv",
    );
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta;
    let eta = [0.09, -0.12];
    let mut worst = 0.0f64;
    let mut n = 0;
    for s in prep.population.subjects.iter().take(6) {
        if s.obs_records.is_empty() {
            continue;
        }
        let (_, h) = ctmm_subject_eta_hessian(model, s, theta, &eta).expect("in scope");
        // Each column by a Richardson difference of the exact gradient (`O(h⁴)`).
        let grad = |e: &[f64]| ctmm_subject_eta_grad(model, s, theta, e).unwrap().1;
        let fd = nalgebra::DMatrix::from_fn(2, 2, |l, k| {
            let d = |step: f64| {
                let (mut a, mut b) = (eta.to_vec(), eta.to_vec());
                a[k] += step;
                b[k] -= step;
                (grad(&a)[l] - grad(&b)[l]) / (2.0 * step)
            };
            (4.0 * d(5e-4) - d(1e-3)) / 3.0
        });
        let scale = fd.iter().fold(1e-300f64, |a, &v| a.max(v.abs()));
        for (a, b) in h.iter().zip(fd.iter()) {
            assert!(a.is_finite());
            worst = worst.max((a - b).abs() / scale);
        }
        n += 1;
    }
    assert!(n > 2, "too few subjects ({n})");
    eprintln!("CTMM η-Hessian vs Richardson of the exact gradient: worst {worst:.2e}");
    // Bound to be set from the measured value (see the eprintln). A dropped clamp or gap
    // misses by O(1).
    assert!(worst < 1e-6, "{worst:.2e}");

    let root = env!("CARGO_MANIFEST_DIR");
    let pd = crate::api::prepare_run(
        &format!("{root}/examples/ctmm_pd_2state.ferx"),
        Some(&format!("{root}/data/ctmm_pd_2state.csv")),
    )
    .unwrap();
    let m = &pd.parsed.model;
    if m.n_eta > 0 {
        let s = pd.population.subjects.iter().find(|s| !s.obs_records.is_empty()).unwrap();
        let e = vec![0.0; m.n_eta];
        // The drug-driven example is served too (`a_drug_driven_ctmm_is_differentiated_exactly`
        // pins the values).
        let (_, h) = ctmm_subject_eta_hessian(m, s, &m.default_params.theta, &e)
            .expect("a drug-driven generator is served exactly");
        assert!(h.iter().all(|v| v.is_finite()));
    }
}
