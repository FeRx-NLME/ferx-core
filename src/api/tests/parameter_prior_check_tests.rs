//! `check_parameter_priors` reached through `validate_model_file` — the
//! `ferx check` half of the #254 gate.
//!
//! The unit-level rejections (`PriorSet::build`'s messages, the scale
//! conversions) live in `estimation/priors_tests.rs`, and the `fit()` half in
//! `tests/parameter_priors.rs`. What is pinned *here* is the wiring: `ferx
//! check` and `fit()` must refuse the same models. Before this the check ran
//! only from `fit_inner`, so a model whose prior named a FIXed or unknown
//! parameter reported "no errors" and then failed at fit time — the worst
//! ordering, because the check step is what a user runs precisely to avoid
//! discovering it there.
//!
//! Each test states the regression it catches, and each has a straddle: an
//! otherwise-identical model that must stay clean, so none of them is satisfied
//! by a gate that simply rejects every priored model.

/// Write `src` to a `.ferx` temp file — `validate_model_file` is reached
/// through a path, and the `.ferx` suffix is what `parse_full_model_file` sees.
fn temp_model(src: &str) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(".ferx")
        .tempfile()
        .expect("create temp model");
    f.write_all(src.as_bytes()).expect("write temp model");
    f.flush().expect("flush temp model");
    f
}

/// A one-compartment IV model whose `[parameters]` and `[fit_options]` blocks
/// the caller supplies, so each test differs by exactly the declaration under
/// test.
fn model_src(parameters: &str, fit_options: &str) -> String {
    format!(
        "[parameters]\n{parameters}\n\
         [individual_parameters]\n\
         \x20 CL = TVCL * exp(ETA_CL)\n\
         \x20 V  = TVV\n\n\
         [structural_model]\n\
         \x20 pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP_ERR)\n\n\
         [fit_options]\n{fit_options}"
    )
}

/// `[parameters]` with the `TVV` line supplied by the caller — the one
/// declaration every test below varies.
fn params_with_tvv(tvv_line: &str) -> String {
    format!(
        "  theta TVCL(0.2, 0.001, 10.0)\n  \
         {tvv_line}\n  \
         omega ETA_CL ~ 0.09\n  \
         sigma PROP_ERR ~ 0.04\n\n"
    )
}

fn report_for(src: &str) -> crate::api::CheckReport {
    let f = temp_model(src);
    crate::api::validate_model_file(f.path().to_str().expect("utf-8 temp path"), None)
}

fn codes(report: &crate::api::CheckReport) -> Vec<&str> {
    report.diagnostics.iter().map(|d| d.code.as_str()).collect()
}

/// A prior that cannot resolve must fail `ferx check`, not only `fit()`.
///
/// The straddle is the same prior on a *free* θ: that model is valid, so the
/// check is discriminating on resolvability rather than on the presence of a
/// prior. Deleting the `check_parameter_priors` call from `validate_model_file`
/// reddens the first half and leaves the second green.
#[test]
fn check_rejects_a_prior_on_a_fixed_parameter_without_data() {
    let report = report_for(&model_src(
        &params_with_tvv("theta TVV(10.0, 0.1, 500.0, FIX) prior(10.0, rse = 20%)"),
        "  method = focei\n",
    ));
    assert!(
        !report.valid,
        "a prior on a FIXed parameter must invalidate the report: {:?}",
        report.diagnostics
    );
    assert!(
        codes(&report).contains(&"E_PRIOR_UNRESOLVED"),
        "{:?}",
        report.diagnostics
    );

    // Straddle: identical model, θ free — valid.
    let ok = report_for(&model_src(
        &params_with_tvv("theta TVV(10.0, 0.1, 500.0) prior(10.0, rse = 20%)"),
        "  method = focei\n",
    ));
    assert!(
        ok.valid,
        "a prior on a free theta must still check clean: {:?}",
        ok.diagnostics
    );
}

/// A method that does not apply priors is a `ferx check` error too — this is
/// the failure that is *invisible* at fit time (the fit converges and reports
/// the unpenalized MLE), so catching it before the run is the whole point.
#[test]
fn check_rejects_a_prior_the_final_method_cannot_apply() {
    let priors = params_with_tvv("theta TVV(10.0, 0.1, 500.0) prior(10.0, rse = 20%)");
    let report = report_for(&model_src(&priors, "  method = saem\n"));
    assert!(
        codes(&report).contains(&"E_PRIOR_METHOD_UNSUPPORTED"),
        "{:?}",
        report.diagnostics
    );

    // Straddle: a chain whose *last* stage applies priors is fine, so the check
    // reads the final method rather than any method.
    let ok = report_for(&model_src(&priors, "  methods = [saem, focei]\n"));
    assert!(
        !codes(&ok).contains(&"E_PRIOR_METHOD_UNSUPPORTED"),
        "{:?}",
        ok.diagnostics
    );
}

/// `covariance_method = rsr` under a prior is rejected, and so is `s`.
///
/// `rsr` is the one this test exists for: `R⁻¹ S R⁻¹` wraps a prior-free `S` in
/// two prior-shrunk `R⁻¹` factors, so it silently under-states the SE on every
/// priored coordinate. It was accepted at first, and the `s` diagnostic
/// recommended it by name — hence the third assertion, which pins that the
/// suggestion no longer sends users to the other broken estimator.
///
/// The straddle is `r` (the default), which carries the prior's curvature and
/// must stay clean.
#[test]
fn check_rejects_both_s_and_rsr_covariance_under_a_prior() {
    let priors = params_with_tvv("theta TVV(10.0, 0.1, 500.0) prior(10.0, rse = 20%)");
    for method in ["s", "rsr"] {
        let report = report_for(&model_src(
            &priors,
            &format!("  method = focei\n  covariance = true\n  covariance_method = {method}\n"),
        ));
        let hit = report
            .diagnostics
            .iter()
            .find(|d| d.code == "E_PRIOR_COV_METHOD_UNSUPPORTED")
            .unwrap_or_else(|| {
                panic!("`covariance_method = {method}` must be rejected: {report:?}")
            });
        assert!(
            hit.message.contains(method),
            "the message must name the rejected method: {}",
            hit.message
        );
        assert!(
            !hit.suggestion.as_deref().is_some_and(|s| s.contains("rsr")),
            "the suggestion must not recommend `rsr`, which is rejected too: {:?}",
            hit.suggestion
        );
    }

    // Straddle: `r` carries the prior's curvature and is the recommended answer.
    let ok = report_for(&model_src(
        &priors,
        "  method = focei\n  covariance = true\n  covariance_method = r\n",
    ));
    assert!(
        !codes(&ok).contains(&"E_PRIOR_COV_METHOD_UNSUPPORTED"),
        "{:?}",
        ok.diagnostics
    );

    // And with no prior declared, `rsr` is untouched — the rejection is about
    // the prior, not about the estimator.
    let unpriored = report_for(&model_src(
        &params_with_tvv("theta TVV(10.0, 0.1, 500.0)"),
        "  method = focei\n  covariance = true\n  covariance_method = rsr\n",
    ));
    assert!(
        !codes(&unpriored).contains(&"E_PRIOR_COV_METHOD_UNSUPPORTED"),
        "{:?}",
        unpriored.diagnostics
    );
}
