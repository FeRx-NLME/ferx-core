//! `validate_model_file` reports the name the fit will carry (#1395).
//!
//! The check report used the file stem unconditionally, so `ferx check --json`
//! showed the stem for `model NAME`, `model = NAME` and no `model` line alike —
//! it could not surface a declaration the parser had dropped. Now the report
//! goes through the same `set_model_name` the fit entry points use: declared
//! name when there is one, the stem otherwise.

use std::io::Write;

fn model(preamble: &str) -> String {
    format!(
        "{preamble}\n\
         [parameters]\n  theta TVCL(1.0, 0.1, 10.0)\n  theta TVV(10.0, 1.0, 100.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma ADD ~ 1.0\n\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n  DV ~ additive(ADD)\n"
    )
}

/// A temp file whose stem is known, so the two sources of the name are told apart.
fn temp_model(src: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("from_the_stem.ferx");
    std::fs::File::create(&path)
        .expect("create temp model")
        .write_all(src.as_bytes())
        .expect("write temp model");
    let p = path.to_str().expect("utf-8 temp path").to_string();
    (dir, p)
}

#[test]
fn the_check_report_carries_the_declared_name_in_either_spelling() {
    for preamble in ["model declared_name", "model = declared_name"] {
        let (_dir, path) = temp_model(&model(preamble));
        let report = crate::api::validate_model_file(&path, None);
        assert!(report.valid, "{preamble:?}: {:?}", report.diagnostics);
        assert_eq!(report.model, "declared_name", "{preamble:?}");
    }
}

#[test]
fn the_check_report_falls_back_to_the_stem_without_a_declaration() {
    let (_dir, path) = temp_model(&model(""));
    let report = crate::api::validate_model_file(&path, None);
    assert!(report.valid, "{:?}", report.diagnostics);
    assert_eq!(report.model, "from_the_stem");
}

/// A malformed `model` line is a parse error in the report — under the stem,
/// since nothing was parsed — rather than a `valid: true` report under the stem.
#[test]
fn a_malformed_model_line_is_a_parse_error_in_the_report() {
    let (_dir, path) = temp_model(&model("model = two words"));
    let report = crate::api::validate_model_file(&path, None);
    assert!(!report.valid);
    assert_eq!(report.model, "from_the_stem");
    assert_eq!(report.diagnostics.len(), 1);
    assert!(
        report.diagnostics[0]
            .message
            .contains("Malformed model name declaration"),
        "{}",
        report.diagnostics[0].message
    );
}
