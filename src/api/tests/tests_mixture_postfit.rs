use super::*;
use nalgebra::{DMatrix, DVector};

/// Two-class model: class 1 clears at `TVCL1 = 1`, class 2 at `TVCL2 = 3`, so
/// the per-class IPRED curves are far apart and a class mix-up is unmistakable.
const MIX_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.5, 0.001, 0.999)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

const TIMES: [f64; 4] = [0.5, 1.0, 2.0, 4.0];

fn one_subject_pop() -> crate::types::Population {
    use std::io::Write;
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT\n");
    csv.push_str("1,0,0,100,1,1\n");
    for t in TIMES {
        // DV value is irrelevant to the IPRED assertions below.
        csv.push_str(&format!("1,{t},1.0,0,0,1\n"));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), None, None).unwrap()
}

/// Closed-form 1-cpt IV bolus concentration at `eta = 0`.
fn analytic_ipred(cl: f64) -> Vec<f64> {
    TIMES
        .iter()
        .map(|t| (100.0 / 10.0) * (-(cl / 10.0) * t).exp())
        .collect()
}

/// Post-fit diagnostics must be computed under each subject's **fitted** class
/// (#985). `compute_subject_results` receives the MIXEST-class EBEs, so leaving
/// `MIXNUM` at its class-1 default would pair a class-2 η̂ with class-1 typical
/// values — IPRED/PRED/IWRES/CWRES for every subject the fit assigned to class 2
/// would be silently wrong.
#[test]
fn subject_results_use_the_fitted_mixture_class() {
    let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
    let population = one_subject_pop();
    let params = &model.default_params;
    let eta_hats = vec![DVector::zeros(1)];
    let h_matrices = vec![DMatrix::zeros(TIMES.len(), 1)];
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]];

    let run = |mixest: Option<&[usize]>| {
        compute_subject_results(
            &model,
            &population,
            params,
            &eta_hats,
            &h_matrices,
            &kappas,
            true,
            mixest,
        )
    };

    // No mixture posteriors (non-mixture caller): the class-1 default stands.
    let class1 = analytic_ipred(1.0);
    let default_run = run(None);
    for (got, want) in default_run[0].ipred.iter().zip(&class1) {
        assert!(
            (got - want).abs() < 1e-9,
            "default IPRED {got} vs class-1 {want}"
        );
    }

    // MIXEST = 2 (0-based 1): every prediction must follow TVCL2, not TVCL1.
    let class2 = analytic_ipred(3.0);
    let mixest_run = run(Some(&[1]));
    for (got, want) in mixest_run[0].ipred.iter().zip(&class2) {
        assert!(
            (got - want).abs() < 1e-9,
            "MIXEST=2 IPRED {got} vs class-2 {want}"
        );
    }
    // PRED (population prediction at eta = 0) follows the same class.
    for (got, want) in mixest_run[0].pred.iter().zip(&class2) {
        assert!(
            (got - want).abs() < 1e-9,
            "MIXEST=2 PRED {got} vs class-2 {want}"
        );
    }
    // Sanity: the two classes are genuinely distinguishable here, so the test
    // cannot pass by both runs collapsing onto the same curve.
    assert!(
        default_run[0]
            .ipred
            .iter()
            .zip(&mixest_run[0].ipred)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "class-1 and class-2 IPRED must differ"
    );
}

/// The same guard covers `[derived]` columns, which can branch on `MIXNUM`.
#[test]
fn derived_columns_use_the_fitted_mixture_class() {
    let src = MIX_MODEL.to_string() + "\n[derived]\n  CLD = CL\n";
    let model = crate::parser::model_parser::parse_model_string(&src).unwrap();
    let population = one_subject_pop();
    let params = &model.default_params;
    let eta_hats = vec![DVector::zeros(1)];
    let h_matrices = vec![DMatrix::zeros(TIMES.len(), 1)];
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]];

    let mut subjects = compute_subject_results(
        &model,
        &population,
        params,
        &eta_hats,
        &h_matrices,
        &kappas,
        true,
        Some(&[1]),
    );
    compute_extra_output_columns(
        &model,
        &population,
        &params.theta,
        &kappas,
        &mut subjects,
        Some(&[1]),
    );
    let (_, cld) = subjects[0]
        .extra_columns
        .iter()
        .find(|(name, _)| name == "CLD")
        .expect("[derived] CLD computed");
    assert!(
        cld.iter().all(|v| (v - 3.0).abs() < 1e-9),
        "derived CL must use the MIXEST class (TVCL2 = 3), got {cld:?}"
    );
}
