//! #3150: an `omega NAME ~ 0 FIX` η is a placeholder. NONMEM holds it at 0 and the objective is
//! the same with and without it; ferx used to give it the variance `1e-8` and carry it through
//! the inner problem, which moved the OFV by about 3e-6 on this fixture and cost run time.
//!
//! Anchored on the **mixed** case, four live ηs next to six `$OMEGA 0 FIX` (the all-zero case is
//! anchored by `count_pois_fixed` / `one_cpt_iv_pooled`). `nonmem_anchor/zero_fix_eta.ctl` and
//! `zero_fix_eta_null.ctl` (NONMEM 7.5.1, `ADVAN3 TRANS4`, `METHOD=1 INTERACTION MAXEVAL=0
//! POSTHOC`, `data/three_cpt_iv.csv` as `zero_fix_eta.csv`) give the same objective,
//! 744.72835307525645, to every digit. ferx runs the closed-form twin of the same model.
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

const NULL_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.15
  omega ETA_V2 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  maxiter = 0
"#;

/// The same model with six zero-`FIX` placeholders, used as in `zero_fix_eta.ctl`
/// (`CL` and `V1` read two each, `Q` and `V2` one).
fn with_placeholders() -> String {
    let decl: String = (1..=6).map(|i| format!("  omega ETA_Z{i} ~ 0.0 FIX\n")).collect();
    NULL_MODEL
        .replace("  sigma PROP_ERR", &format!("{decl}  sigma PROP_ERR"))
        .replace("exp(ETA_CL)", "exp(ETA_CL + ETA_Z1 + ETA_Z5)")
        .replace("exp(ETA_V1)", "exp(ETA_V1 + ETA_Z2 + ETA_Z6)")
        .replace("exp(ETA_Q)", "exp(ETA_Q + ETA_Z3)")
        .replace("exp(ETA_V2)", "exp(ETA_V2 + ETA_Z4)")
}

fn ofv(model: &str) -> f64 {
    let parsed = parse_full_model(model).expect("the model parses");
    let pop = read_nonmem_csv(Path::new("nonmem_anchor/zero_fix_eta.csv"), None, None)
        .expect("the data load");
    let options =
        FitOptions { outer_maxiter: 0, run_covariance_step: false, ..parsed.fit_options.clone() }
            .quiet();
    fit(&parsed.model, &pop, &parsed.model.default_params, &options).expect("evaluates").ofv
}

/// The `OBJ` of the `ITERATION = -1000000000` row of a NONMEM `.ext`.
fn nonmem_ofv(run: &str) -> f64 {
    let text = std::fs::read_to_string(format!("nonmem_anchor/results/{run}.ext")).unwrap();
    let last = text.lines().find(|l| l.split_whitespace().next() == Some("-1000000000"));
    last.and_then(|l| l.split_whitespace().last()).expect("final row").parse().unwrap()
}

/// ferx's OFV with the six placeholders equals NONMEM's, and equals ferx's own without them.
///
/// Measured (Linux x86_64, `ci-test`): ferx 744.7283530727 with and without the placeholders
/// (identical), NONMEM 744.7283530753 (2.6e-9 apart: the closed form's own rounding). On `main`
/// the placeholders moved the OFV to 744.7283495325, 3.5e-6 below.
#[test]
fn zero_fix_placeholders_do_not_move_the_objective_and_match_nonmem() {
    let (nm_with, nm_null) = (nonmem_ofv("zero_fix_eta"), nonmem_ofv("zero_fix_eta_null"));
    assert_eq!(nm_with, nm_null, "NONMEM: a `0 FIX` placeholder changes nothing");
    let (with, null) = (ofv(&with_placeholders()), ofv(NULL_MODEL));
    assert!((with - null).abs() < SAME_MODEL_BOUND, "with {with} vs without {null}");
    assert!((with - nm_with).abs() < EVAL_BOUND, "ferx {with} vs NONMEM {nm_with}");
}

/// The two ferx models are one model once the placeholders are gone, so they agree exactly
/// (measured 0); 1e-10 is far under the 3.5e-6 defect.
const SAME_MODEL_BOUND: f64 = 1e-10;
/// ferx against NONMEM: 38× over the measured 2.6e-9 and 35× under the defect.
const EVAL_BOUND: f64 = 1e-7;

/// The ηs that remain are the reduced model's, bit for bit: the placeholders are not in the inner
/// problem, so the objective, every subject's EBEs and the η count are those of the model
/// written without them (the issue's Tier-1 check).
#[test]
fn the_remaining_etas_are_the_reduced_models_bit_for_bit() {
    let run = |model: &str| {
        let parsed = parse_full_model(model).expect("the model parses");
        let pop = read_nonmem_csv(Path::new("nonmem_anchor/zero_fix_eta.csv"), None, None).unwrap();
        let o = FitOptions {
            outer_maxiter: 0,
            run_covariance_step: false,
            ..parsed.fit_options.clone()
        }
        .quiet();
        let r = fit(&parsed.model, &pop, &parsed.model.default_params, &o).expect("evaluates");
        (parsed.model.n_eta, r)
    };
    let ((n_with, with), (n_null, null)) = (run(&with_placeholders()), run(NULL_MODEL));
    assert_eq!((n_with, n_null), (4, 4), "the inner problem's dimension");
    assert_eq!(with.ofv.to_bits(), null.ofv.to_bits(), "OFV");
    assert_eq!(with.eta_names, null.eta_names);
    assert!(!with.subjects.is_empty());
    for (a, b) in with.subjects.iter().zip(&null.subjects) {
        let bits = |s: &ferx_core::types::SubjectResult| -> Vec<u64> {
            s.eta.iter().map(|e| e.to_bits()).collect()
        };
        assert_eq!(bits(a), bits(b), "EBEs of subject {}", a.id);
    }
}
