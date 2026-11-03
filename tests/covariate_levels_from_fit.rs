//! #1740, end to end: a design recoded to a categorical level the fit never saw is
//! refused by `predict` after `bind_from_fit`, through the real model file, the real
//! CSV reader and a real `fit_from_files` — instead of being scored as the
//! reference level.
//!
//! Tier 2: the fit stops at `outer_maxiter = 0`; nothing converges.

use std::io::Write;
use std::path::PathBuf;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;

const MODEL: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[covariates]
  GRP categorical(levels = auto)

[covariate_model]
  CL ~ GRP categorical(ref = mode)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Twelve subjects, `GRP` cycling 1, 2, 2, 3 (levels [1, 2, 3], mode 2), with
/// `last` in place of the twelfth subject's 3.
fn data(last: u32) -> String {
    let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,GRP\n");
    for id in 1..=12u32 {
        let g = if id == 12 {
            last
        } else {
            [1, 2, 2, 3][(id as usize - 1) % 4]
        };
        let dv = 2.0 + 0.1 * f64::from(id);
        s.push_str(&format!("{id},0,.,1,100,1,1,{g}\n"));
        s.push_str(&format!("{id},1,{dv},0,.,1,0,{g}\n"));
        s.push_str(&format!("{id},6,{},0,.,1,0,{g}\n", dv / 3.0));
    }
    s
}

fn write(dir: &tempfile::TempDir, name: &str, text: &str) -> PathBuf {
    let p = dir.path().join(name);
    write!(std::fs::File::create(&p).unwrap(), "{text}").unwrap();
    p
}

#[test]
fn predict_after_bind_from_fit_refuses_a_level_the_fit_never_saw() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = write(&dir, "m.ferx", MODEL);
    let fit_data = write(&dir, "fit.csv", &data(3));
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let fit = ferx_core::fit_from_files(
        model_path.to_str().unwrap(),
        Some(fit_data.to_str().unwrap()),
        None,
        Some(opts),
    )
    .expect("fit at the initial estimates");

    let predict_on = |last: u32| -> Result<Vec<f64>, ferx_core::EngineError> {
        let design_path = write(&dir, &format!("design{last}.csv"), &data(last));
        let mut design = ferx_core::read_nonmem_csv(&design_path, None, None).expect("read design");
        let mut parsed = parse_full_model(MODEL).expect("parse");
        ferx_core::bind_from_fit(&mut parsed, MODEL, &mut design, &fit.data_bindings)
            .expect("bind from fit");
        let mut params = parsed.model.default_params.clone();
        params.theta = fit.theta.clone();
        ferx_core::predict(&parsed.model, &design, &params)
            .map(|r| r.iter().map(|r| r.pred).collect())
    };

    // The twin recodes the twelfth subject to 2, the reference: it predicts.
    let twin = predict_on(2).expect("a listed level predicts");
    assert_eq!(twin.len(), 24);
    assert!(twin.iter().all(|p| p.is_finite()), "{twin:?}");

    // Recoded to 4, a level the fit has no θ for: refused, with the from-fit advice.
    let err = predict_on(4).expect_err("an unseen level must be refused");
    // The refusal carries `ferx check`'s code, so a caller (ferx-r#498) can match on it
    // rather than on this text (#1746).
    assert_eq!(err.code(), Some("E_COV_LEVEL_UNKNOWN"), "{err}");
    let e = err.to_string();
    assert!(
        e.contains("has the fit's levels [1.0, 2.0, 3.0] (reference 2)"),
        "{e}"
    );
    assert!(e.contains("`GRP` takes [4.0] in this data"), "{e}");
    assert!(e.contains("The fit estimated no θ for these values"), "{e}");
    assert!(!e.contains("levels = auto"), "{e}");
}
