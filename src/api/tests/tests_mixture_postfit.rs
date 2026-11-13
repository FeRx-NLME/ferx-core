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

    // `false`: a closed-form mixture model integrates nothing, so there are no
    // solver statistics to collect and the second half of the pair is empty.
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
            false,
        )
        .0
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

    let (mut subjects, _no_ode_stats) = compute_subject_results(
        &model,
        &population,
        params,
        &eta_hats,
        &h_matrices,
        &kappas,
        true,
        Some(&[1]),
        false,
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

/// One subject whose data carry a `MIXNUM` column (value 7, no class of the model).
fn one_subject_pop_with_mixnum_column() -> crate::types::Population {
    use std::io::Write;
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,MIXNUM\n");
    csv.push_str("1,0,0,100,1,1,7\n");
    for t in TIMES {
        csv.push_str(&format!("1,{t},1.0,0,0,1,7\n"));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), None, None).unwrap()
}

/// `[output] MIXNUM`'s column on `population`, computed under fitted class `mixest`
/// (0-based, as `compute_extra_output_columns` takes it).
fn output_mixnum_column(
    model: &CompiledModel,
    population: &crate::types::Population,
    mixest: Option<&[usize]>,
) -> Option<Vec<f64>> {
    let params = &model.default_params;
    let eta_hats = vec![DVector::zeros(1)];
    let h_matrices = vec![DMatrix::zeros(TIMES.len(), 1)];
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]];
    let (mut subjects, _) = compute_subject_results(
        model,
        population,
        params,
        &eta_hats,
        &h_matrices,
        &kappas,
        true,
        mixest,
        false,
    );
    compute_extra_output_columns(
        model,
        population,
        &params.theta,
        &kappas,
        &mut subjects,
        mixest,
    );
    let hits: Vec<_> = subjects[0]
        .extra_columns
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("MIXNUM"))
        .collect();
    assert!(hits.len() <= 1, "one MIXNUM column at most: {hits:?}");
    hits.first().map(|(_, v)| v.clone())
}

/// #2784: on a `[mixture]` model `[output] MIXNUM` is the class, as `MIXNUM` is in every
/// expression — the fitted MIXEST, or the simulated class where the data were drawn
/// under one — and never the data column of that name (here 7, no class at all).
///
/// Dies on: the echo reading the covariate (7), the class column not written, the
/// simulated class not preferred over the fitted one.
#[test]
fn output_mixnum_is_the_class_not_the_data_column() {
    let src = MIX_MODEL.to_string() + "\n[output]\n  mixnum\n";
    let model = crate::parser::model_parser::parse_model_string(&src).unwrap();
    let mut population = one_subject_pop_with_mixnum_column();
    assert!(
        population.covariate_names.iter().any(|c| c == "MIXNUM"),
        "fixture: the data carry a MIXNUM column"
    );
    let n = TIMES.len();
    assert_eq!(
        output_mixnum_column(&model, &population, Some(&[1])),
        Some(vec![2.0; n]),
        "the fitted class (MIXEST = 2), not the data column"
    );
    population.subjects[0].simulated_mixnum = Some(1);
    assert_eq!(
        output_mixnum_column(&model, &population, Some(&[1])),
        Some(vec![1.0; n]),
        "the simulated class wins over the fitted one"
    );
}

/// Without `[mixture]` there is no class, and the parser refuses `MIXNUM` in every
/// block, `[output]` included — so a data column named `MIXNUM` is never echoed by
/// `[output]` on any model (#2784): the only model that accepts the name reads it as
/// the class.
#[test]
fn output_mixnum_is_refused_without_a_mixture() {
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)

[output]
  MIXNUM
";
    let err = crate::parser::model_parser::parse_model_string(src).unwrap_err();
    assert!(err.contains("MIXNUM") && err.contains("[mixture]"), "{err}");
}

/// #2784 validation: `[output] MIXNUM` on a `[mixture]` model is accepted without a data
/// column (it used to be `E_OUTPUT_UNKNOWN_COLUMN`) and, beside a data column of that
/// name, warns that the column written is the class and not that data — the silent
/// substitution the issue reports.
///
/// Dies on: the mixture gate dropped from validation, the warning not raised, or raised
/// with no data column.
#[test]
fn output_mixnum_validation_on_a_mixture_model() {
    let src = MIX_MODEL.to_string() + "\n[output]\n  MIXNUM\n";
    let model = crate::parser::model_parser::parse_model_string(&src).unwrap();

    let no_column = validate_output_columns(&model, &one_subject_pop());
    assert!(no_column.is_empty(), "class column accepted: {no_column:?}");

    let with_column = validate_output_columns(&model, &one_subject_pop_with_mixnum_column());
    let codes: Vec<&str> = with_column.iter().map(|d| d.code.as_str()).collect();
    assert_eq!(codes, vec!["W_OUTPUT_MIXNUM_IS_CLASS"], "{with_column:?}");
    let msg = &with_column[0].message;
    assert!(
        msg.contains("mixture class") && msg.contains("not the data column `MIXNUM`"),
        "{msg}"
    );
    assert!(
        msg.contains("fitted MIXEST") && msg.contains("Rename"),
        "{msg}"
    );
}
