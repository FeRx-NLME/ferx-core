//! `predict_survival_at_etas_diag` from a fit, end to end (#1885, test T9).
//!
//! A two-iteration fit of `examples/tte_exponential.ferx` feeds its EBEs through
//! `SuppliedEtas::from_fit`; each subject's predicted `H` at its own record time must be
//! the closed form `λ·exp(η̂ᵢ)·Tᵢ` at the fit's λ and that subject's η̂. The population
//! handed to the predictor is the fitted one **reversed**, so a lookup by position
//! instead of by ID gives each subject another's η̂ and fails.
//!
//! Tier 2: two outer iterations; the oracle needs η̂ to be the fit's, not optimal.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::estimation::uncertainty_samples::fitted_params_from_result;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::FitOptions;
use ferx_core::{fit, predict_survival_at_etas_diag, SuppliedEtas};

#[test]
fn individual_curves_from_a_fit_are_at_each_subjects_ebe() {
    let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
    let m = parse_model_string(&src).expect("parse");
    let (pop, _) =
        read_population_for(&m, &None, "data/tte_exponential.csv", None, None, None, &[])
            .expect("data");
    let opts = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&m, &pop, &m.default_params, &opts).expect("quick fit");
    eprintln!("T9 fit warnings: {:#?}", r.warnings);
    let params = fitted_params_from_result(&r, &m).expect("fitted params");
    let etas = SuppliedEtas::from_fit(&r, &m).expect("from_fit");

    let mut rev = pop.clone();
    rev.subjects.reverse();
    let times: Vec<(String, f64)> = pop
        .subjects
        .iter()
        .map(|s| match s.obs_records.first() {
            Some(ferx_core::types::ObsRecord::Event { time, .. }) => (s.id.clone(), *time),
            _ => panic!("subject {} has one TTE record", s.id),
        })
        .collect();
    let mut grid: Vec<f64> = times.iter().map(|t| t.1).collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let out = predict_survival_at_etas_diag(&m, &rev, &params, &etas, &grid).expect("curves");

    let lambda = params.theta[0];
    let mut distinct = std::collections::HashSet::new();
    for (id, t) in &times {
        let s = r
            .subjects
            .iter()
            .find(|s| &s.id == id)
            .expect("fitted subject");
        let eta = s.eta[0];
        assert!(eta.is_finite() && eta != 0.0, "subject {id}: η̂ {eta}");
        distinct.insert(eta.to_bits());
        let want = lambda * eta.exp() * t;
        let row = out
            .results
            .iter()
            .find(|row| &row.id == id && row.time == *t)
            .expect("record time on the grid");
        assert!(
            row.cum_hazard.is_finite(),
            "subject {id}: H {}",
            row.cum_hazard
        );
        let rel = ((row.cum_hazard - want) / want).abs();
        assert!(
            rel <= 4.0 * f64::EPSILON,
            "subject {id} at t = {t}: H {} vs λ·exp(η̂)·T {want} (rel {rel:e})",
            row.cum_hazard
        );
    }
    // Every subject censored at T = 30 has the same data and so the same η̂; the exact
    // events each have their own. Enough distinct values that a reversal cannot pass.
    assert!(
        distinct.len() >= 10,
        "the fit's η̂ must differ across subjects: {} distinct",
        distinct.len()
    );
}
