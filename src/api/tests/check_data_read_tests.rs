//! The two private helpers of `ferx check`'s data read (#1465):
//! [`check_selection_filter`], which compiles the model file's `[data_selection]`
//! clauses, and [`reader_warning_code`], which relays a reader warning under the
//! code the reader itself wrote.
//!
//! Tier 1, and against hand-built inputs rather than model files, because both have
//! arms no file reaches. `check_selection_filter`'s `Err` is **unreachable through
//! a file**: the parser compiles every clause with the same `FilterClause::parse`
//! the builder calls, so a clause that will not compile is already an `E_PARSE` and
//! `validate_model_file` has returned. Unreachable is not the same as absent — the
//! arm still has to exist, still has to not panic, and `codecov/patch` still counts
//! its lines against the diff. `reader_warning_code`'s `W_DATA` fallback is the
//! same shape: every warning the reader writes today states a code, so the arm that
//! handles one that does not is only reachable from here.
//!
//! The end-to-end half — that these are the codes `ferx check` actually prints on
//! a real dataset — is `tests/check_applies_data_selection.rs`.

use super::{check_selection_filter, reader_warning_code};
use crate::FitOptions;

/// A clause the builder cannot compile is a diagnostic, not a panic and not a
/// silent fallback to "no filter".
///
/// Mutation that must redden this: `unwrap` / `expect` in place of the `map_err`,
/// or an `Err` arm that returns `Ok(None)` — which is exactly the unfiltered read
/// #1465 removed, reintroduced on the path nothing else covers.
#[test]
fn a_clause_that_does_not_compile_is_an_e_parse_diagnostic() {
    let opts = FitOptions {
        // `===` is not an operator the clause grammar has.
        ignore_exprs: vec!["CMT ===".to_string()],
        ..FitOptions::default()
    };
    let err = check_selection_filter(&opts)
        .err()
        .expect("a clause that does not compile cannot yield a filter");
    assert_eq!(err.code, "E_PARSE", "{}", err.message);
    assert_eq!(
        err.block.as_deref(),
        Some("data_selection"),
        "the block is what `validate_model_file` attaches the line number to: {:?}",
        err
    );
    assert!(
        !err.message.is_empty(),
        "the builder's own sentence has to survive the wrap"
    );
}

/// No clauses at all is `Ok(None)` — the same "read everything" the check did
/// before #1465, and the state of every model file that declares no
/// `[data_selection]` block.
#[test]
fn no_clauses_is_no_filter() {
    let opts = FitOptions::default();
    assert!(
        opts.ignore_exprs.is_empty()
            && opts.accept_exprs.is_empty()
            && opts.ignore_subjects.is_empty(),
        "the default must carry no clauses, or this passes for the wrong reason"
    );
    let filter = check_selection_filter(&opts).expect("no clauses cannot fail to compile");
    assert!(
        filter.is_none(),
        "an empty filter is `None`, not a filter that matches nothing"
    );
}

/// Each clause kind on its own reaches the builder. `ignore_subjects` is the one
/// the partial builder next door (`data_selection_reads_cmt`) deliberately drops,
/// so a filter built from `ignore_exprs` alone would answer `None` here.
#[test]
fn every_clause_kind_on_its_own_produces_a_filter() {
    for (label, opts) in [
        (
            "ignore",
            FitOptions {
                ignore_exprs: vec!["CMT == 2".to_string()],
                ..FitOptions::default()
            },
        ),
        (
            "accept",
            FitOptions {
                accept_exprs: vec!["CMT == 1".to_string()],
                ..FitOptions::default()
            },
        ),
        (
            "ignore_subjects",
            FitOptions {
                ignore_subjects: vec!["2".to_string()],
                ..FitOptions::default()
            },
        ),
    ] {
        let filter = check_selection_filter(&opts)
            .unwrap_or_else(|e| panic!("{label}: {} {}", e.code, e.message));
        assert!(
            filter.is_some(),
            "{label}: the clause must reach the compiled filter"
        );
    }
}

/// [`reader_warning_code`] reads the code the reader wrote, in both spellings the
/// reader uses — a colon straight after the code (`W_MISSING_DV: …`) and a code
/// followed by prose (`W_ADDL_MISSING_II subject 3: …`). The second is why this is
/// a leading-`[A-Z0-9_]`-run rule and not `split_once(':')`, which would have
/// answered `W_ADDL_MISSING_II subject 3`.
///
/// Mutation that must redden this: restore the three-arm `if`/`else if` chain —
/// every row below but `W_ADDL_MISSING_II` and `W_CMT_DEFAULTED` then reads
/// `W_DATA`.
#[test]
fn a_reader_warning_is_relayed_under_the_code_it_states() {
    for (message, want) in [
        (
            "W_MISSING_DV: 1 observation row(s) (EVID=0) had a missing DV",
            "W_MISSING_DV",
        ),
        (
            "W_ADDL_MISSING_II subject 3: ADDL > 0 but II is zero",
            "W_ADDL_MISSING_II",
        ),
        (
            "W_FILTER_COLUMN_ABSENT: data-selection filter references column(s)",
            "W_FILTER_COLUMN_ABSENT",
        ),
        (
            "W_CMT_DEFAULTED: the dataset has no CMT column",
            "W_CMT_DEFAULTED",
        ),
        ("W_NO_DOSES: parsed zero dose events", "W_NO_DOSES"),
        (
            "W_ALL_DOSES_ZERO: every dose record has AMT = 0",
            "W_ALL_DOSES_ZERO",
        ),
        (
            "W_AMT_NOT_DOSED: 2 record(s) across 1 subject(s)",
            "W_AMT_NOT_DOSED",
        ),
    ] {
        assert_eq!(reader_warning_code(message), want, "on {message:?}");
    }
}

/// A message stating no code of its own is still the generic `W_DATA` — the
/// fallback the hand-written chain had, kept.
///
/// Mutation that must redden this: drop the `starts_with("W_")` guard, or the
/// length guard, and `"WARNING: …"` / `"W_ …"` become codes.
#[test]
fn a_message_with_no_code_of_its_own_is_relayed_as_w_data() {
    for message in [
        "plain prose with no code at all",
        "WARNING: an upper-case word that is not a code",
        "W_ is a prefix, not a code",
        "",
        "  leading space, so the run is empty",
        "w_missing_dv: lower case is not the reader's spelling",
    ] {
        assert_eq!(reader_warning_code(message), "W_DATA", "on {message:?}");
    }
}
