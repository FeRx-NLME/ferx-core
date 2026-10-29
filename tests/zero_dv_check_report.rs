//! Tier 2 (#2459): `ferx check` reports `W_ZERO_DV_MAGNITUDE_ONLY` from the model file's own
//! `[fit_options]`, as `fit()` does — including its `bloq_method`, which decides whether a
//! `CENS = 1` zero is scored at its DV (`drop`, counted) or by its CDF (`m3`, skipped).

use ferx_core::{validate_model_file, CheckReport};
use std::io::Write;

const MODEL: &str =
    "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
     omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  \
     CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
     [error_model]\n  DV ~ proportional(PROP)\n";

/// One subject: a dose, then a `DV = 0` row flagged `CENS = 1`, then two positive rows.
const DATA: &str = "ID,TIME,DV,EVID,AMT,CMT,CENS\n\
                    1,0,.,1,100,1,0\n\
                    1,0.5,0,0,.,1,1\n\
                    1,2,4.1,0,.,1,0\n\
                    1,8,2.2,0,.,1,0\n";

fn temp(contents: &str, suffix: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .expect("temp file");
    f.write_all(contents.as_bytes()).expect("write temp file");
    f
}

fn check(bloq: &str) -> CheckReport {
    let m = temp(
        &format!("{MODEL}[fit_options]\n  method = focei\n  bloq_method = {bloq}\n"),
        ".ferx",
    );
    let d = temp(DATA, ".csv");
    validate_model_file(m.path().to_str().unwrap(), Some(d.path().to_str().unwrap()))
}

fn has_code(r: &CheckReport) -> bool {
    r.diagnostics
        .iter()
        .any(|d| d.code == "W_ZERO_DV_MAGNITUDE_ONLY")
}

#[test]
fn check_report_follows_the_model_files_bloq_method() {
    let drop = check("drop");
    assert!(has_code(&drop), "drop: {:?}", drop.diagnostics);
    let m3 = check("m3");
    assert!(!has_code(&m3), "m3: {:?}", m3.diagnostics);
    // The m3 silence must come from the skip, not from a check that stopped before the
    // data warnings: neither report may carry an error.
    for r in [&drop, &m3] {
        assert!(
            !r.diagnostics.iter().any(|d| d.is_error()),
            "{:?}",
            r.diagnostics
        );
    }
}
