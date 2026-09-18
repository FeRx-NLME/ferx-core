//! An endpoint model is reported because its rows route by CMT — not because its
//! error-model dispatch table happens to be empty (#1409).
//!
//! Round 4 of #1404's review measured the gap this closes. That PR's predicate used
//! `ErrorSpec::PerCmt(_)` as a proxy for "this model routes rows by CMT", and the
//! proxy only matches endpoint-**only** models: the parser produces an empty per-CMT
//! error map solely from its `is_endpoint_only` default, so an endpoint model that
//! *also* declares an `[error_model]` gets `ErrorSpec::Single` and stayed suppressed.
//! Measured there on `examples/tte_exponential.ferx` — `[event_model] cmt = 2`
//! alongside a `pk` block and an `[error_model]` — with one event row's `CMT` cell
//! spelled `x`: no `W_CMT_DEFAULTED` on either spelling, while the row moved out of
//! the TTE endpoint into the Gaussian grid for an OFV of 254.5871 against 166.4395.
//!
//! #1409 keys the channel on `api::run::obs_routing_for` — the one place a routing
//! set is derived from a model — so the declared endpoints answer the question and
//! the error map is not consulted at all. This file pins that on the very example
//! the gap was measured on, so the fix is tied to the measurement rather than to a
//! hand-built fixture that shares its shape.
//!
//! The unit-level counterpart is
//! `api::reader_warning_suppression_tests::an_endpoint_model_reports_because_its_rows_route_by_cmt`,
//! which asserts the same thing on a fixture whose `ErrorSpec` is explicitly `Single`
//! so the error-model arm is provably inert. This one is the end-to-end check that a
//! *shipped* model file takes that path.
//!
//! Tier 2: `validate_model_file` only — no `fit()`.

use ferx_core::validate_model_file;
use std::io::Write;
use tempfile::NamedTempFile;

fn temp(contents: &str, suffix: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .expect("temp file");
    write!(f, "{contents}").expect("write");
    f.flush().expect("flush");
    f
}

/// Two subjects on the `tte_exponential` example's endpoint (`cmt = 2`): one event,
/// one right-censored. `spelling` is the first subject's event-row `CMT` cell.
fn csv(spelling: &str) -> String {
    format!(
        "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
         1,0,.,1,100,1,1\n\
         1,12,1,0,.,{spelling},0\n\
         2,0,.,1,100,1,1\n\
         2,30,0,0,.,2,0\n"
    )
}

fn warns(src: &str, data: &str) -> bool {
    let m = temp(src, ".ferx");
    let d = temp(data, ".csv");
    let report = validate_model_file(m.path().to_str().unwrap(), Some(d.path().to_str().unwrap()));
    report
        .diagnostics
        .iter()
        .any(|x| x.code == "W_CMT_DEFAULTED")
}

#[test]
fn an_endpoint_model_that_also_declares_an_error_model_is_reported() {
    let src =
        std::fs::read_to_string("examples/tte_exponential.ferx").expect("the example still exists");
    // The fixture has to be the shape the gap was about, not merely an endpoint
    // model: an `[error_model]` block is what gives it `ErrorSpec::Single` and so
    // what made the old `ErrorSpec::PerCmt(_)` proxy miss it. Asserted on the source
    // rather than assumed, so a future edit to the example that drops the block
    // turns this into a red test instead of a silently weaker one.
    assert!(
        src.contains("[error_model]") && src.contains("[event_model]"),
        "the example must carry BOTH blocks, or it is not the case #1404 round 4 \
         measured and this test proves nothing"
    );

    assert!(
        warns(&src, &csv("x")),
        "an unreadable CMT cell on an endpoint model's event row must be reported — \
         the row silently leaves the TTE endpoint for the Gaussian grid"
    );
    // The other side of the straddle: the warning is about the *defaulted* cell, not
    // about the model being an endpoint model. A readable dataset stays quiet, so a
    // predicate hard-wired to report every endpoint model fails here.
    assert!(
        !warns(&src, &csv("2")),
        "with every cell readable there is no compartment to have guessed"
    );
}
