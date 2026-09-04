//! Fixtures shared by the `search` unit tests.
//!
//! The one thing these tests cannot fake is a [`FitResult`]: it has 130-odd
//! fields and no constructor, so there is no way to build one field by field.
//! [`fixture_fit`] is therefore one genuine `outer_maxiter = 0` **evaluation**
//! of `examples/warfarin.ferx` — ferx's `MAXEVAL=0`, the same trick
//! `tests/bootstrap_end_to_end.rs` uses — computed once per test binary and
//! cloned. Tests that need a different fit mutate the clone.

use std::sync::OnceLock;

use ferx_core::edit::ModelText;
use ferx_core::{fit, prepare_run, FitResult, Population, Subject};

use super::candidate::Candidate;

pub(crate) const MODEL: &str = "../../examples/warfarin.ferx";
pub(crate) const DATA: &str = "../../data/warfarin.csv";

/// One real evaluation of the warfarin model, memoized.
///
/// The initialisation runs on a **dedicated thread the caller joins**, and that
/// is load-bearing rather than tidy. The tests call this from inside the
/// runner's Rayon pool, and the evaluation itself enters a nested pool; a Rayon
/// worker blocked on a nested `install` keeps stealing work, so it can re-enter
/// this function *in the middle of its own* `get_or_init` and deadlock against
/// the `OnceLock` it is already initialising. Observed, not theorised: a
/// two-candidate run on a one-worker pool hung there. Blocking on a plain
/// `JoinHandle` is not a Rayon latch, so the waiting worker steals nothing and
/// the initialisation cannot be re-entered.
pub(crate) fn fixture_fit() -> FitResult {
    static FIT: OnceLock<FitResult> = OnceLock::new();
    FIT.get_or_init(|| {
        std::thread::spawn(|| {
            let prepared = prepare_run(MODEL, Some(DATA)).expect("warfarin model + data load");
            let mut options = prepared.parsed.fit_options.clone();
            options.outer_maxiter = 0;
            options.run_covariance_step = false;
            options.verbose = false;
            options.threads = Some(1);
            options.checkpoint = false;
            fit(
                &prepared.parsed.model,
                &prepared.population,
                &prepared.init_params,
                &options,
            )
            .expect("warfarin evaluation")
        })
        .join()
        .expect("the fixture thread panicked")
    })
    .clone()
}

/// The fixture with `converged` set and a chosen OFV / AIC / BIC.
pub(crate) fn converged_fit(ofv: f64) -> FitResult {
    let mut result = fixture_fit();
    result.converged = true;
    result.ofv = ofv;
    result.aic = ofv + 2.0;
    result.bic = ofv + 5.0;
    result
}

/// The smallest population the runner will accept. It is only ever read for the
/// resume manifest's data fingerprint, since these tests inject the fitter.
pub(crate) fn population(ids: &[&str]) -> Population {
    Population {
        subjects: ids
            .iter()
            .map(|id| Subject {
                id: (*id).to_string(),
                ..Default::default()
            })
            .collect(),
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

pub(crate) fn model_text(body: &str) -> ModelText {
    ModelText::parse(body).expect("fixture model text")
}

/// Two spellings of one model: different bytes, identical canonical form.
pub(crate) fn same_model_two_ways() -> (ModelText, ModelText) {
    (
        model_text("[parameters]\ntheta CL = 1\ntheta V = 10\n"),
        model_text("[parameters]   # a comment\n\ntheta CL = 1\n   theta V  =  10\n"),
    )
}

pub(crate) fn candidate(id: &str, body: &str) -> Candidate {
    Candidate::new(id, model_text(body))
}
