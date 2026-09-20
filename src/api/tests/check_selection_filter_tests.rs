//! [`check_selection_filter`] — the `[data_selection]` filter `ferx check`'s data
//! read applies (#1465).
//!
//! Tier 1, and hand-built `FitOptions` rather than a model file, because the `Err`
//! arm is **unreachable through a file**: the parser compiles every clause with the
//! same `FilterClause::parse` the builder calls, so a clause that will not compile
//! is already an `E_PARSE` and `validate_model_file` has returned. Unreachable is
//! not the same as absent — the arm still has to exist, still has to not panic, and
//! `codecov/patch` still counts its lines against the diff.

use super::check_selection_filter;
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
