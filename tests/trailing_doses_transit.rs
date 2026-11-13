//! Tier 2 for #2778: a transit-absorption ODE fit with trailing doses equals, bit for bit,
//! the fit of the same data with those rows deleted.
//!
//! The Tier-1 tests in `src/api/scoring_horizon_tests.rs` compare the other routes as whole
//! fits, and this route at one point (`trailing_doses_do_not_change_a_transit_subject`):
//! scoring every record of a smooth-input model is the cost #2778 removes, and it is too slow
//! for a per-PR test. Here both fits run on the cut records, so a few iterations stay quick.

use ferx_core::{fit, prepare_run, EstimationMethod, FitOptions, FitResult, Optimizer};
use std::io::Write;

const MODEL: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.2 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  MTT = 1.5
  NTR = 3.0

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central

[error_model]
  DV ~ proportional(PROP)
";

/// Daily doses sampled over two days. `addl = 2` stops at the last dose before the last
/// observation; a larger value, the `ADDL` train crossing it.
fn dataset(addl: u32) -> String {
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,ADDL\n");
    for id in 1..=4 {
        csv += &format!("{id},0,.,1,100,1,0,1,24,{addl}\n");
        for (j, t) in [1.0, 4.0, 25.0, 30.0, 50.0].iter().enumerate() {
            let dv = 4.0 + (id as f64 * 1.7 + j as f64).sin() + 0.01 * t;
            csv += &format!("{id},{t},{dv:.5},0,.,1,0,0,0,0\n");
        }
    }
    csv
}

fn fitted(csv: &str) -> FitResult {
    let mut model = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
    model.write_all(MODEL.as_bytes()).unwrap();
    let mut data = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    data.write_all(csv.as_bytes()).unwrap();
    let run = prepare_run(
        model.path().to_str().unwrap(),
        Some(data.path().to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("prepare_run: {e}"));
    let options = FitOptions {
        method: EstimationMethod::Foce,
        interaction: true,
        optimizer: Optimizer::NloptLbfgs,
        outer_maxiter: 3,
        run_covariance_step: false,
        threads: Some(1),
        ..FitOptions::default()
    };
    fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        &options,
    )
    .unwrap_or_else(|e| panic!("fit: {e}"))
}

fn bits(r: &FitResult) -> Vec<u64> {
    let mut v = vec![r.ofv.to_bits()];
    v.extend(r.theta.iter().map(|x| x.to_bits()));
    v.extend(r.sigma.iter().map(|x| x.to_bits()));
    v.extend(r.omega.iter().map(|x| x.to_bits()));
    for s in &r.subjects {
        v.extend(s.eta.iter().map(|x| x.to_bits()));
        v.extend(s.ipred.iter().map(|x| x.to_bits()));
    }
    v
}

#[test]
fn trailing_doses_do_not_change_a_transit_fit() {
    let with_tail = fitted(&dataset(29));
    let without = fitted(&dataset(2));
    assert_eq!(with_tail.gradient_method_inner, "analytic (Dual2)");
    assert_eq!(with_tail.gradient_method_outer, "analytic (Dual2)");
    assert!(with_tail.ofv.is_finite());
    assert_eq!(bits(&with_tail), bits(&without));
}
