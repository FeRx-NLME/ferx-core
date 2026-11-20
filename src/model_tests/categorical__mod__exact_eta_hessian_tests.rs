// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::categorical::count::count_diagnostics;
use ferx_model::categorical::count::count_pearson_residual;
use ferx_model::categorical::count::predict_count;
use ferx_model::categorical::count::simulate_count;
use ferx_model::parser::model_parser::with_model_time;
use ferx_model::sens::num::NonRecRead;
use ferx_model::stats::likelihood::SUBJECT_SENTINEL_NLL;
use ferx_model::types::CompiledModel;
use ferx_model::types::DiscreteObsDiagnostic;
use ferx_model::types::EndpointLikelihood;
use ferx_model::types::EndpointPredictionResult;
use ferx_model::types::LinearPredictorFn;
use ferx_model::types::LinkFn;
use ferx_model::types::ObsRecord;
use ferx_model::types::Prediction;
use ferx_model::types::Subject;
use rand::RngExt;
use std::collections::HashMap;

use ferx_model::categorical::*;

/// A model string against an example dataset, through the production loader.
pub fn load(src: &str, data: &str) -> crate::api::PreparedRun {
    let root = env!("CARGO_MANIFEST_DIR");
    let f = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
    std::fs::write(f.path(), src).unwrap();
    crate::api::prepare_run(f.path().to_str().unwrap(), Some(&format!("{root}/{data}"))).unwrap()
}

/// The binary η-Hessian is exact. With `lp = TH0 + ETA_0 + THX·X + THT·TIME·exp(ETA_1)`
/// it is `Σ p(1−p)∇lp∇lpᵀ + (p − y)∇²lp`, `∇lp = (1, THT·TIME·e^{η₁})`, `∇²lp` nonzero
/// only at `(1,1)` (`= ∇lp₁`): built here by hand and matched to rounding; and the
/// production route takes it (the stencil agrees to its own accuracy).
#[test]
fn the_binary_eta_hessian_is_exact() {
    let prep = load(
        r"
[parameters]
  theta TH0(0.2, -10.0, 10.0)
  theta THX(0.4, -10.0, 10.0)
  theta THT(0.05, -10.0, 10.0)
  omega ETA_0 ~ 0.3
  omega ETA_1 ~ 0.2
[covariates]
  X continuous
[binary_model]
  cmt   = 3
  logit = TH0 + ETA_0 + THX * X + THT * TIME * exp(ETA_1)
",
        "data/binary_logistic.csv",
    );
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta;
    let eta = [0.11, -0.07];
    let mut worst_closed = 0.0f64;
    let mut n = 0;
    for s in prep.population.subjects.iter().take(6) {
        if s.obs_records.is_empty() {
            continue;
        }
        let (_, h) = discrete_subject_eta_hessian(model, s, theta, &eta).expect("in scope");
        // By hand.
        let mut want = nalgebra::DMatrix::<f64>::zeros(2, 2);
        for r in &s.obs_records {
            let ObsRecord::DiscreteState { time, state, .. } = r else {
                continue;
            };
            let x = s.covariates.get("X").copied().unwrap_or(0.0);
            let tt = theta[2] * time * eta[1].exp();
            let lp = theta[0] + eta[0] + theta[1] * x + tt;
            let p = inv_logit(lp);
            let y = (*state as f64).min(1.0);
            let g = [1.0, tt];
            for k in 0..2 {
                for l in 0..2 {
                    want[(k, l)] += p * (1.0 - p) * g[k] * g[l];
                }
            }
            want[(1, 1)] += (p - y) * tt;
        }
        let scale = want.iter().fold(1e-300f64, |a, &v| a.max(v.abs()));
        for (a, b) in h.iter().zip(want.iter()) {
            worst_closed = worst_closed.max((a - b).abs() / scale);
        }
        n += 1;
    }
    assert!(n > 2, "too few subjects ({n})");
    eprintln!("binary η-Hessian: vs closed form {worst_closed:.2e}");
    // Measured 1.4e-17 against the closed form (`non_gaussian_eta_hessian_tests` pins the
    // same Hessian against a Richardson stencil of the value path).
    assert!(worst_closed < 1e-12, "{worst_closed:.2e}");
}
