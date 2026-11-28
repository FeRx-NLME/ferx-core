//! Tier 2 (#3025): a Laplace / AGQ mixture fit takes the exact routes — the exact outer
//! gradient and the exact covariance R-matrix — so the fit reports no finite-difference
//! fallback. `fit()` with a covariance step, three outer iterations: returns immediately.
//! The unit-level routing (`predicted_r_route`, the exact gradient and R-matrix at a point)
//! is `mixture::tests::a_laplace_mixture_takes_the_exact_routes`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::io::Write;

const MODEL: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.2, -10.0, 10.0)
  theta BWT(0.05, -5.0, 5.0)
  omega ETA_CL ~ 0.06
  sigma EPS ~ 0.02

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)
  omega(2) ETA_CL ~ 0.15
  sigma(2) EPS ~ 0.03

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Two clearance groups (1.0 and 3.0), five subjects each, deterministic ±3 % ripple.
fn bimodal_csv() -> String {
    let (v, dose) = (10.0_f64, 100.0_f64);
    let times = [0.5_f64, 1.0, 2.0, 4.0, 8.0];
    let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
    let mut sid = 0;
    for (grp, &cl) in [1.0_f64, 3.0].iter().enumerate() {
        let wt = if grp == 0 { 60.0 } else { 90.0 };
        for _ in 0..5 {
            sid += 1;
            s.push_str(&format!("{sid},0,0,{dose},1,1,{wt}\n"));
            for (ti, &t) in times.iter().enumerate() {
                let c = (dose / v) * (-(cl / v) * t).exp();
                let ripple = 1.0 + 0.03 * (((sid + ti) as f64) * 1.3).sin();
                s.push_str(&format!("{sid},{t},{:.5},0,0,1,{wt}\n", c * ripple));
            }
        }
    }
    s
}

#[test]
fn a_laplace_mixture_fit_takes_the_exact_routes() {
    let model = parse_model_string(MODEL).unwrap();
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(bimodal_csv().as_bytes()).unwrap();
    let pop = read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap();
    for (kind, n_agq) in [("laplace", 1usize), ("agq3", 3)] {
        let opts = FitOptions {
            method: if n_agq == 1 { EstimationMethod::Laplace } else { EstimationMethod::FoceI },
            interaction: true,
            n_agq,
            outer_maxiter: 3,
            run_covariance_step: true,
            ..FitOptions::default()
        };
        let fit = fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert!(fit.ofv.is_finite(), "{kind}");
        let fd_warning = fit
            .warnings
            .iter()
            .find(|w| w.contains("finite-difference") || w.contains("finite difference"));
        assert!(fd_warning.is_none(), "{kind}: {fd_warning:?}");
    }
}
