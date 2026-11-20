// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::sens::num::PkNum;
use ferx_model::stats::likelihood::SUBJECT_SENTINEL_NLL;
use ferx_model::survival::parametric::cum_hazard;
use ferx_model::survival::parametric::hazard_and_cum_hazard;
use ferx_model::survival::parametric::hazard_and_cum_hazard_g;
use ferx_model::survival::parametric::mean_survival;
use ferx_model::survival::parametric::median_survival;
use ferx_model::survival::parametric::sample_conditional_event_time;
use ferx_model::survival::parametric::sample_event_time;
use ferx_model::types::CompiledModel;
use ferx_model::types::EventType;
use ferx_model::types::HazardFamily;
use ferx_model::types::ObsRecord;
use ferx_model::types::RtteClock;
use ferx_model::types::SimOutcome;
use ferx_model::types::Subject;
use ferx_model::types::TteRecurrence;
use nalgebra::DMatrix;
use rand::RngExt;
use std::collections::HashMap;

use crate::types::{EndpointLikelihood, HazardSpec};
use ferx_model::survival::*;

/// The exponential family in closed form: with `λ = θ·e^η`, `−log L = Σ λ t_i − Σ_exact
/// (log λ)` (no truncation in the fixture), so `∂²/∂η² = λ Σ t_i` — the exact Hessian
/// must match it to rounding.
#[test]
fn the_exact_eta_hessian_is_the_closed_form_on_an_exponential_hazard() {
    let root = env!("CARGO_MANIFEST_DIR");
    let prep = crate::api::prepare_run(
        &format!("{root}/examples/tte_exponential.ferx"),
        Some(&format!("{root}/data/tte_exponential.csv")),
    )
    .unwrap();
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta;
    let (cmt, ep) = model.endpoints.iter().next().unwrap();
    assert!(
        matches!(ep, EndpointLikelihood::Tte { hazard: HazardSpec::Analytic { .. }, .. }),
        "an analytic exponential endpoint"
    );
    let mut checked = 0;
    for s in prep.population.subjects.iter().take(8) {
        let recs: Vec<ObsRecord> = s
            .obs_records
            .iter()
            .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt))
            .cloned()
            .collect();
        if recs.is_empty() {
            continue;
        }
        let eta = [0.13];
        let term = TteTerm::new(model, s);
        let kind = crate::sens::ode_provider::adjoint::terms::TermKind::Tte;
        let (_, h) = term_eta_hessian(model, s, kind, &term, theta, &eta).unwrap();
        let lambda = theta[0] * eta[0].exp();
        let t_sum: f64 = recs
            .iter()
            .map(|r| match r {
                ObsRecord::Event { time, entry_time, .. } => {
                    assert_eq!(*entry_time, 0.0, "the closed form assumes no truncation");
                    *time
                }
                _ => 0.0,
            })
            .sum();
        let want = lambda * t_sum;
        assert!(((h[(0, 0)] - want) / want).abs() < 1e-14, "{} vs {want}", h[(0, 0)]);
        checked += 1;
    }
    assert!(checked > 0);
}

/// The inner gradient's TTE block in closed form on the exponential hazard:
/// `∂/∂η [λ Σ t_i − Σ_exact log λ] = λ Σ t_i − n_exact`, matched to rounding, with the
/// value equal to the production data term's.
#[test]
fn the_tte_eta_gradient_is_the_closed_form_on_an_exponential_hazard() {
    let root = env!("CARGO_MANIFEST_DIR");
    let prep = crate::api::prepare_run(
        &format!("{root}/examples/tte_exponential.ferx"),
        Some(&format!("{root}/data/tte_exponential.csv")),
    )
    .unwrap();
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta;
    let mut checked = 0;
    for s in prep.population.subjects.iter().take(8) {
        if s.obs_records.is_empty() {
            continue;
        }
        let eta = [-0.21];
        let (v, g) = tte_subject_eta_grad(model, s, theta, &eta).expect("in scope");
        let lambda = theta[0] * eta[0].exp();
        let (mut t_sum, mut n_exact) = (0.0, 0.0);
        for r in &s.obs_records {
            if let ObsRecord::Event { time, event_type, .. } = r {
                t_sum += time;
                if matches!(event_type, EventType::Exact) {
                    n_exact += 1.0;
                }
            }
        }
        let want = lambda * t_sum - n_exact;
        assert!((g[0] - want).abs() <= 1e-13 * (1.0 + want.abs()), "{} vs {want}", g[0]);
        let (cmt, ep) = model.endpoints.iter().next().unwrap();
        let crate::types::EndpointLikelihood::Tte { hazard, recurrence, .. } = ep else {
            unreachable!()
        };
        let recs: Vec<ObsRecord> = s
            .obs_records
            .iter()
            .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt))
            .cloned()
            .collect();
        let f = tte_data_term(&recs, hazard, *recurrence, theta, &eta, &s.covariates);
        assert!((v - f).abs() <= 1e-12 * (1.0 + f.abs()), "value {v} vs {f}");
        checked += 1;
    }
    assert!(checked > 0);
}
