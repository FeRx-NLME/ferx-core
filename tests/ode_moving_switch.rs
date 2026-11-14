//! A θ that reaches `[odes]` only through an `if` condition must move (#2649).
//!
//! The objective depends on `TLAG`, but its analytic gradient was exactly zero, so the
//! default route never moved it; the model was then routed to finite differences, and is now
//! served by the exact gradient, the switch located and its sensitivity jump applied. Tier-2: a handful of outer iterations on a small dataset
//! generated with the switch at 2, started at 1.13. Asserting that `TLAG` left its start is
//! the whole claim: before the fix it came back bit-identical.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{EstimationMethod, FitOptions};
use ferx_core::{fit, read_nonmem_csv};
use std::io::Write;

const MODEL: &str = "
[parameters]
  theta TLAG(1.13, 0.1, 6.0)
  theta TVKE(0.2, 0.01, 5.0)
  omega ETA_KE ~ 0.2
  sigma ADD ~ 0.5 (sd)
[individual_parameters]
  KE = TVKE * exp(ETA_KE)
  LAGT = TLAG
[structural_model]
  ode(obs_cmt=a, states=[a, tau])
[odes]
  d/dt(a) = if (tau > LAGT) -KE * a else 0.0
  d/dt(tau) = 1.0
[error_model]
  DV ~ additive(ADD)
";

#[test]
fn a_theta_only_in_an_odes_condition_is_estimated() {
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV\n");
    for id in 1..=8 {
        let ke = 0.3 * (0.2 * (f64::from(id) - 4.5) / 3.5).exp();
        csv.push_str(&format!("{id},0,0,1,10,1,1\n"));
        for t in [0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0] {
            let wobble = 0.1 * (7.0 * t + f64::from(id)).sin();
            let dv = 10.0 * (-ke * (t - 2.0).max(0.0)).exp() + wobble;
            csv.push_str(&format!("{id},{t},{dv:.4},0,0,1,0\n"));
        }
    }
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(csv.as_bytes()).expect("write data");
    let population = read_nonmem_csv(file.path(), None, None).expect("data loads");

    let model = parse_model_string(MODEL).expect("model parses");
    let options = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 5,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let result = fit(&model, &population, &model.default_params, &options).expect("fit runs");

    let tlag = result.theta[0];
    assert!(tlag.is_finite(), "TLAG = {tlag}");
    assert!(
        (tlag - 1.13).abs() > 1e-3,
        "TLAG never left its start: {tlag}"
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("finite-difference")),
        "the exact gradient serves it: {:?}",
        result.warnings
    );
}
