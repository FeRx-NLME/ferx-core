//! `parse_method_token` and `apply_method_option` are the engine's one reading of
//! a `method` value, public so a front end with its own `method` argument
//! (ferx-r's `ferx_fit(method = ...)`, ferx-r #506) takes its tokens from here
//! instead of keeping a copy. These pin the crate-root paths that front end uses.

use ferx_core::{apply_method_option, parse_method_token, EstimationMethod, FitOptions};

#[test]
fn parse_method_token_reads_vi_and_rejects_unknown_tokens() {
    assert_eq!(parse_method_token("vi"), Ok(EstimationMethod::Vi));
    assert_eq!(parse_method_token(" VI "), Ok(EstimationMethod::Vi));
    assert_eq!(
        parse_method_token("importance_sampling"),
        Ok(EstimationMethod::Imp)
    );
    let err = parse_method_token("advi").unwrap_err();
    assert!(err.contains("unknown estimation method"), "{err}");
    let err = parse_method_token("agq").unwrap_err();
    assert!(err.contains("method = laplace with n_agq"), "{err}");
}

#[test]
fn apply_method_option_sets_a_chain_like_a_fit_options_line() {
    let mut opts = FitOptions::default();
    apply_method_option(&mut opts, "[vi, laplace]").unwrap();
    assert_eq!(opts.method, EstimationMethod::Laplace);
    assert_eq!(
        opts.methods,
        vec![EstimationMethod::Vi, EstimationMethod::Laplace]
    );
    assert!(opts.user_set_keys.iter().any(|k| k == "method"));

    let mut opts = FitOptions::default();
    apply_method_option(&mut opts, "foce").unwrap();
    assert_eq!(opts.method, EstimationMethod::Foce);
    assert!(opts.methods.is_empty());
    assert!(!opts.interaction, "a final `foce` clears interaction");

    let mut opts = FitOptions::default();
    let err = apply_method_option(&mut opts, "[focei, nope]").unwrap_err();
    assert!(err.contains("`nope`"), "{err}");
}
