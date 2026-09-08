//! Analytic IOV covariance at the stored NONMEM 7.6.0 beta 1 FOCEI MATRIX=R estimates.
//! No outer optimization: compare covariance at the same parameter point.
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const MODEL: &str = r"
[parameters]
theta TVCL(0.172762, 0.001, 10)
theta TVV(8.62783, 0.1, 500)
theta TVKA(1.17857, 0.01, 50)
omega ETA_CL ~ 0.0399164
omega ETA_V ~ 0.0107732
omega ETA_KA ~ 0.0254212
kappa KAPPA_CL ~ 0.0357135
sigma PROP_ERR ~ 0.035386 (var)
[individual_parameters]
CL = TVCL * exp(ETA_CL + KAPPA_CL)
V = TVV * exp(ETA_V)
KA = TVKA * exp(ETA_KA)
[structural_model]
pk one_cpt_oral(cl=CL,v=V,ka=KA)
[error_model]
DV ~ proportional(PROP_ERR)
";

#[test]
fn iov_analytic_covariance_matches_nonmem() {
    let model = ferx_core::parser::model_parser::parse_model_string(MODEL).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC")).unwrap();
    let run = |analytic| {
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            interaction: true,
            outer_maxiter: 0,
            run_covariance_step: true,
            analytic_cov_hessian: analytic,
            inner_maxiter: 500,
            inner_tol: 1e-10,
            cov_inner_tol: Some(1e-10),
            verbose: true,
            ..FitOptions::default()
        };
        fit(&model, &pop, &model.default_params, &opts).unwrap()
    };
    let an = run(true);

    let values = |r: &ferx_core::FitResult| {
        let mut v = r.se_theta.clone().unwrap();
        v.extend(r.se_sigma.as_ref().unwrap());
        for i in 0..3 {
            v.push(ferx_core::types::omega_se_at(&r.se_omega, 3, i, i).unwrap());
        }
        v.extend(r.se_kappa.as_ref().unwrap());
        v
    };
    let a = values(&an);

    // Sigma is an SD in ferx. Convert NONMEM's variance SE by the delta method.
    let nm = [
        0.0133628,
        0.341219,
        0.0932369,
        0.00381392 / (2.0 * 0.035386_f64.sqrt()),
        0.0280561,
        0.00643777,
        0.0279915,
        0.0176156,
    ];
    assert_eq!(a.len(), nm.len());
    eprintln!("OFV analytic={} NONMEM=308.83046873982664", an.ofv);
    for i in 0..a.len() {
        eprintln!(
            "SE[{i}]: analytic={} NONMEM={} rel_NM={}",
            a[i],
            nm[i],
            (a[i] - nm[i]).abs() / nm[i]
        );
        assert!(
            (a[i] - nm[i]).abs() / nm[i] < 0.01,
            "analytic/NONMEM SE mismatch at {i}"
        );
    }
    assert!((an.ofv - 308.83046873982664).abs() < 0.01);
}
