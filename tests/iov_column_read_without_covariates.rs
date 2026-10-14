//! #1936 end to end: a model that reads its `iov_column` in
//! `[individual_parameters]` with no `[covariates]` block evaluates, and its
//! objective is `to_bits()`-equal to the `[covariates] OCC categorical` twin.
//!
//! No NONMEM anchor: the fix creates no new numeric object. It makes the no-block
//! read take the same columns the declared read already took, so the oracle is
//! the declared twin. Mutation: the whole reader fix (`read_population_for_policy`
//! back to `None ⇒ Vec::new()`) — the no-block run returns `Err`
//! (`E_MISSING_COVARIATE` on `OCC`).
//!
//! `maxiter = 0`: one evaluation, no convergence loop (Tier 2). There is no
//! tolerance to measure: the assertion is bit-equality. Realised, both arms on
//! `gradient: analytic (Dual2)`: OFV 312.860485 on Linux aarch64
//! (`tools/linux-test.sh`) and on macOS arm64 — the issue's 312.8605 for the
//! declared run on Linux x86_64.

use std::path::Path;

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta OCC2_CL(0.5, -0.9, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  OCCF = if (OCC == 2) 1 + OCC2_CL else 1
  CL = TVCL * OCCF * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
  covariance = false
  maxiter    = 0
  checkpoint = false
"#;

fn ofv(dir: &Path, name: &str, text: &str) -> Result<ferx_core::FitResult, String> {
    let model_path = dir.join(name);
    std::fs::write(&model_path, text).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/warfarin_iov.csv");
    ferx_core::run_model_with_data(model_path.to_str().unwrap(), Some(data.to_str().unwrap()))
        .map(|(fit, _)| fit)
}

#[test]
fn iov_column_read_with_no_block_matches_the_declared_twin() {
    let dir = tempfile::tempdir().unwrap();
    let bare = ofv(dir.path(), "bare.ferx", MODEL).expect("no-block model evaluates");
    let declared_text = format!("{MODEL}\n[covariates]\n  OCC categorical\n");
    let declared =
        ofv(dir.path(), "declared.ferx", &declared_text).expect("declared twin evaluates");

    assert!(bare.ofv.is_finite(), "OFV {}", bare.ofv);
    eprintln!(
        "#1936 OFV no-block {:.6} declared {:.6}; warnings: {:?}",
        bare.ofv, declared.ofv, bare.warnings
    );
    assert_eq!(bare.ofv.to_bits(), declared.ofv.to_bits());
}
