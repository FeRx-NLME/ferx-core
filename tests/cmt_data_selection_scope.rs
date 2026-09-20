//! `W_CMT_DEFAULTED` covers the `[data_selection]` channel too (#1409).
//!
//! `resolve_row_cmt`'s own doc says it is shared by "the dose row, the observation
//! row **and the `[data]` selection filter's `RowContext`**". The filter is handed
//! the compartment the reader *resolved*, so on a dataset whose `CMT` cell the
//! reader had to invent, `ignore = CMT == 2` selects rows on a value nobody wrote —
//! and until #1409 the suppression predicate never asked whether a clause named
//! `CMT` at all. It was the third channel found missing in three consecutive review
//! rounds of #1404, each round having widened a *list* of the model classes someone
//! had thought of.
//!
//! This file pins the property rather than the arm, in the shape
//! `cmt_defaulting_scope.rs` uses for the dose channel:
//!
//! > A `[data_selection]` clause naming `CMT` makes the set of scored records
//! > depend on the compartment the reader chose **if and only if** `fit()` raises
//! > `W_CMT_DEFAULTED` on that dataset.
//!
//! The left side is *measured* — the records are read both ways and counted, and the
//! objective is evaluated on what survived — so the expectation cannot drift into a
//! restatement of the predicate. The straddle (a clause on another column; no clause
//! at all) is asserted in the same loop, because a one-sided table agrees with a
//! predicate stuck at `true`.
//!
//! **Both directions of the filter are covered**, and the Codex review of #1423 is
//! why. A defaulted compartment can make a clause *stop* matching, so the row is
//! wrongly kept — and a kept row is counted by the reader's observation arm like any
//! other. Or it can make a clause *start* matching, so the row is wrongly deleted,
//! and a deleted row hits the filter's `continue` long before that arm runs. The
//! first version of this file tested only the first direction and used `ferx check`
//! as its oracle; since `validate_model_file` reads with `filter: None`, that oracle
//! could not observe the second direction at all, and the mirror case passed green
//! against a reader that counted nothing (CLAUDE.md, "a green test is not evidence
//! that it can fail"). The oracle is now `fit()`.
//!
//! Tier 2: `fit()` only at `outer_maxiter = 0` — one objective evaluation, no
//! convergence loop — plus `stats::likelihood::individual_nll` summed over the
//! subjects at the model's own initial estimates, which is the quantity the issue's
//! OFV measurements moved.

use ferx_core::api::read_population_for;
use ferx_core::io::datareader::SelectionFilter;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::stats::likelihood::individual_nll;
use ferx_core::OmegaMatrix;
use ferx_core::{fit, validate_model_file, FitOptions};
use std::io::Write;
use tempfile::NamedTempFile;

/// The model the issue measured on: one compartment, so **no model-side channel is
/// live** — the dose channel has a single target and there is no per-CMT scaling,
/// error model or readout. Any warning raised here is the filter's doing, which is
/// what makes this file a test of that channel rather than of the predicate at large.
fn model_src(selection: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
{selection}
"#
    )
}

/// One subject, one IV bolus, four observations. The `CMT` cell of the observation
/// at `TIME=4` is written as `spelling` — `2` in one arm, an unreadable `x` in the
/// other — and nothing else differs between the two datasets.
///
/// A *cell* rather than a dropped column on purpose: with the column absent,
/// `E_ENDPOINT_NO_RECORDS` and friends can sometimes speak instead. An unreadable
/// cell is the case where every other diagnostic stays quiet and only
/// `W_CMT_DEFAULTED` is left to say the compartment was chosen.
fn csv(spelling: &str) -> String {
    format!(
        "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
         1,0,.,1,100,1,1\n\
         1,1,1.60,0,.,1,0\n\
         1,4,1.20,0,.,{spelling},0\n\
         1,8,0.85,0,.,1,0\n\
         1,12,0.60,0,.,1,0\n"
    )
}

fn temp(contents: &str, suffix: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .expect("temp file");
    write!(f, "{contents}").expect("write");
    f.flush().expect("flush");
    f
}

/// `(records scored, −log L at the initial estimates)` for this model + dataset,
/// read through the **same** filtered path `fit()` uses.
///
/// The filter is built from the parsed `[data_selection]` block exactly as
/// `api::run` builds it, so what is counted here is what a fit would score.
fn scored(src: &str, data: &str) -> (usize, f64) {
    let m = temp(src, ".ferx");
    let d = temp(data, ".csv");
    let parsed = parse_full_model(src).expect("model parses");
    let opts = &parsed.fit_options;
    let filter = SelectionFilter::from_opts(&opts.ignore_exprs, &opts.accept_exprs, &[])
        .expect("the fixtures' clauses all parse");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        d.path().to_str().unwrap(),
        None,
        None,
        Some(&filter).filter(|f| !f.is_empty()),
        &parsed.column_map,
    )
    .expect("dataset loads");
    drop(m);

    let params = &parsed.model.default_params;
    let n: usize = population.subjects.iter().map(|s| s.obs_times.len()).sum();
    let zero_eta = vec![0.0; parsed.model.n_eta];
    let omega: &OmegaMatrix = &params.omega;
    let mut nll = 0.0;
    for s in &population.subjects {
        let v = individual_nll(
            &parsed.model,
            s,
            &params.theta,
            &zero_eta,
            omega,
            &params.sigma.values,
        );
        // Folded with an explicit finiteness guard rather than through a running
        // `max`/`+` that would absorb a `NaN`: a non-finite objective is the
        // likeliest way to break what this measures, and a silent absorption would
        // let the comparison below pass on the rows that worked (CLAUDE.md).
        assert!(
            v.is_finite(),
            "subject {} has a non-finite objective ({v}) — the comparison below \
             cannot see a difference it does not have",
            s.id
        );
        nll += v;
    }
    (n, nll)
}

/// Whether the **fitting** path raises `W_CMT_DEFAULTED` — the population read
/// *through the filter*, then passed through the same suppression predicate
/// `api::fit` applies to `population.warnings`.
///
/// This, not [`warns`], is the oracle for anything the filter decides. `ferx check`
/// reads with no filter at all (`validation.rs` passes `filter: None`), so it counts
/// a row the fit never sees and reports the warning whether or not the fit does —
/// which made the first version of the mirror case below pass green against a reader
/// that counted nothing (CLAUDE.md: "a green test is not evidence that it can fail").
/// The two entry points genuinely differ here; the last test in this file pins where.
///
/// Through `fit()` itself, at `outer_maxiter = 0` — one objective evaluation, no
/// convergence loop — rather than through a re-spelling of the suppression predicate
/// here. `reader_warning_suppressed` is `pub(crate)`, and a second copy of it in this
/// file would agree with a wrong answer by construction; `fit()` applies the real one
/// to the real filtered population, which is the thing under test.
fn warns_on_read(src: &str, data: &str) -> bool {
    let d = temp(data, ".csv");
    let parsed = parse_full_model(src).expect("model parses");
    let opts = &parsed.fit_options;
    let filter = SelectionFilter::from_opts(&opts.ignore_exprs, &opts.accept_exprs, &[])
        .expect("the fixtures' clauses all parse");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        d.path().to_str().unwrap(),
        None,
        None,
        Some(&filter).filter(|f| !f.is_empty()),
        &parsed.column_map,
    )
    .expect("dataset loads");
    let fit_opts = FitOptions {
        outer_maxiter: 0,
        ..opts.clone()
    };
    let result = fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &fit_opts,
    )
    .expect("a 0-iteration evaluation returns");
    result
        .warnings
        .iter()
        .any(|w| w.starts_with("W_CMT_DEFAULTED"))
}

/// Whether `ferx check` raises `W_CMT_DEFAULTED`. Through the public entry point, so
/// this exercises the filter `fit()` applies rather than a private predicate.
fn warns(src: &str, data: &str) -> bool {
    let m = temp(src, ".ferx");
    let d = temp(data, ".csv");
    let report = validate_model_file(m.path().to_str().unwrap(), Some(d.path().to_str().unwrap()));
    assert!(
        report.diagnostics.iter().all(|x| !x.code.starts_with('E')),
        "fixture must be a valid model + dataset, else the warning could be missing \
         because the check stopped early: {:?}",
        report.diagnostics
    );
    report
        .diagnostics
        .iter()
        .any(|x| x.code == "W_CMT_DEFAULTED")
}

/// A `[data_selection]` block, or nothing at all.
const NO_SELECTION: &str = "";
const IGNORE_CMT: &str = "\n[data_selection]\n  ignore = CMT == 2\n";
const ACCEPT_CMT: &str = "\n[data_selection]\n  accept = CMT == 1\n";
const IGNORE_DV: &str = "\n[data_selection]\n  ignore = DV < 0.001\n";
/// The **mirror** case, and the one the `ignore = CMT == 2` row cannot reach: here
/// the defaulted compartment makes the filter *remove* the row rather than keep it.
/// `TIME == 4` narrows the clause to the one record whose cell differs between the
/// two datasets, so the other three are untouched and the measurement isolates it.
const IGNORE_CMT1_AT_T4: &str = "\n[data_selection]\n  ignore = CMT == 1 && TIME == 4\n";

#[test]
fn the_warning_fires_exactly_where_the_filter_reads_the_chosen_compartment() {
    // `(label, block, expected records kept on the readable dataset)`. The counts are
    // stated so a clause that silently stops firing — the way `ignore = CMT == 2`
    // does when the cell is unreadable — is visible as a number rather than inferred.
    //
    // Both directions of the filter are present on purpose. A defaulted compartment
    // can either make a clause *stop* matching (the row is wrongly kept) or *start*
    // matching (the row is wrongly dropped), and the two reach the reader's
    // defaulting counters completely differently: a kept row goes on to be counted
    // by `record_dose`/`record_obs`, while a dropped one hits the filter's `continue`
    // long before either. A table with only the first kind agrees with an
    // implementation that counts nothing on the second (Codex review of #1423).
    let cases: [(&str, &str, usize); 5] = [
        ("no [data_selection]", NO_SELECTION, 4),
        ("ignore = CMT == 2", IGNORE_CMT, 3),
        ("accept = CMT == 1", ACCEPT_CMT, 3),
        ("ignore = CMT == 1 && TIME == 4", IGNORE_CMT1_AT_T4, 4),
        ("ignore = DV < 0.001", IGNORE_DV, 4),
    ];

    let mut table: Vec<String> = Vec::new();
    for (label, block, want_kept) in cases {
        let src = model_src(block);
        let (n_readable, nll_readable) = scored(&src, &csv("2"));
        let (n_defaulted, nll_defaulted) = scored(&src, &csv("x"));
        assert_eq!(
            n_readable, want_kept,
            "{label}: the clause must actually fire on the readable dataset, or this \
             row measures nothing"
        );

        // Measured: does losing the compartment change what gets scored, or what the
        // objective is on what got scored? Both are asked — a clause could in
        // principle keep the same number of records and still re-weight them.
        let observable = n_readable != n_defaulted
            || (nll_readable - nll_defaulted).abs() > 1e-9 * nll_readable.abs().max(1.0);
        let warned = warns_on_read(&src, &csv("x"));
        table.push(format!(
            "  {label}: kept {n_readable} -> {n_defaulted}, −logL {nll_readable:.4} -> \
             {nll_defaulted:.4}, observable {observable}, warned {warned}"
        ));
        assert_eq!(
            warned,
            observable,
            "{label}: W_CMT_DEFAULTED must fire exactly when the defaulted compartment \
             changes what is scored. Records kept went {n_readable} -> {n_defaulted} and \
             −logL {nll_readable} -> {nll_defaulted}, so `observable` is {observable}, but \
             `ferx check` {}. Either `api::validation::CmtConsumer::DataSelectionFilter` \
             disagrees with what the filter reads, or this case's clause is wrong.\n\n\
             Full table:\n{}",
            if warned { "warned" } else { "did not warn" },
            table.join("\n")
        );
    }

    // The straddle, asserted so the table cannot quietly become one-sided: both
    // outcomes must occur. Without this, a predicate stuck at either value passes
    // every row above on a table that happens to agree with it.
    assert!(
        table.iter().any(|r| r.ends_with("warned true"))
            && table.iter().any(|r| r.ends_with("warned false")),
        "the cases must cover both outcomes, else this test cannot fail:\n{}",
        table.join("\n")
    );
    eprintln!("cmt data-selection scope, realised:\n{}", table.join("\n"));
}

#[test]
fn the_filtered_row_is_silently_kept_and_scored_when_its_compartment_is_defaulted() {
    // The defect itself, stated as its own property rather than left inside the
    // agreement above: the row the user told ferx to drop *is* dropped when its cell
    // is readable and *is not* when the reader had to choose the compartment. This is
    // what the warning is about, and it is measured here rather than described.
    let src = model_src(IGNORE_CMT);
    let (kept_readable, nll_readable) = scored(&src, &csv("2"));
    let (kept_defaulted, nll_defaulted) = scored(&src, &csv("x"));
    assert_eq!(
        (kept_readable, kept_defaulted),
        (3, 4),
        "the CMT=2 row must be dropped when spelled and kept when unreadable"
    );
    assert!(
        (nll_readable - nll_defaulted).abs() > 1e-6,
        "an extra scored record must move the objective: {nll_readable} vs {nll_defaulted}"
    );
    assert!(
        warns_on_read(&src, &csv("x")),
        "and ferx must say so — this is the `ok — 0 warning(s)` the issue measured"
    );
}

/// The message text `fit()` reports, or `None` when it reports no `W_CMT_DEFAULTED`.
fn warning_text(src: &str, data: &str) -> Option<String> {
    let d = temp(data, ".csv");
    let parsed = parse_full_model(src).expect("model parses");
    let opts = &parsed.fit_options;
    let filter = SelectionFilter::from_opts(&opts.ignore_exprs, &opts.accept_exprs, &[])
        .expect("the fixtures' clauses all parse");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        d.path().to_str().unwrap(),
        None,
        None,
        Some(&filter).filter(|f| !f.is_empty()),
        &parsed.column_map,
    )
    .expect("dataset loads");
    let fit_opts = FitOptions {
        outer_maxiter: 0,
        ..opts.clone()
    };
    fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &fit_opts,
    )
    .expect("a 0-iteration evaluation returns")
    .warnings
    .into_iter()
    .find(|w| w.starts_with("W_CMT_DEFAULTED"))
}

#[test]
fn a_defaulted_row_the_filter_deletes_is_still_reported() {
    // The Codex review of #1423, and the mirror of the test above. There the
    // defaulted compartment made a clause *stop* matching, so the row was wrongly
    // kept — and being kept, it went on to be counted by the observation arm like
    // any other row. Here it makes a clause *start* matching, so the row is wrongly
    // deleted, and it hits the filter's `continue` long before `record_obs` can see
    // it. Measured before the fix: records kept 4 -> 3, and `W_CMT_DEFAULTED` absent
    // from `fit()` entirely.
    //
    // The whole summary is checked, not just its presence: the count has to be the
    // deleted row, and the message has to say the row left the fit rather than
    // repeat "assigned compartment 1", which would describe the opposite outcome.
    let src = model_src(IGNORE_CMT1_AT_T4);
    let (kept_readable, _) = scored(&src, &csv("2"));
    let (kept_defaulted, _) = scored(&src, &csv("x"));
    assert_eq!(
        (kept_readable, kept_defaulted),
        (4, 3),
        "the t=4 row must survive when its cell is readable and be deleted when it is not"
    );
    let msg = warning_text(&src, &csv("x")).expect("the deleted row must be reported");
    assert!(
        msg.contains("1 row(s) were removed from the fit by a [data_selection] condition"),
        "the summary must name the deleted row as deleted: {msg}"
    );
    assert!(
        msg.contains("0 dose row(s) and 0 observation row(s)"),
        "…and must not double-count it as a scored row: {msg}"
    );
    assert!(
        msg.contains("not a compartment index (\"x\")"),
        "…and must still name the offending spelling: {msg}"
    );
}

#[test]
fn an_exclusion_decided_by_another_column_stays_silent() {
    // The scoping half, and the reason the count is keyed on the rule that actually
    // fired rather than on "this block mentions CMT somewhere". Both clauses are
    // present; the `DV` one fires first and deletes the row, and the row's defaulted
    // compartment decided nothing — so there is nothing to report.
    //
    // Without this, the natural cheap implementation (`sel.references_column("cmt")`
    // on the whole filter) passes every other test in this file and reports a row
    // whose compartment was irrelevant.
    //
    // The CMT clause is written so it matches *neither* spelling (`CMT == 7`), which
    // removes the ordering question entirely: only the `TIME` rule ever fires, so any
    // count here can only have come from the mere presence of a CMT clause.
    let src = model_src("\n[data_selection]\n  ignore = TIME == 4\n  ignore = CMT == 7\n");
    // The t=4 row is the one deleted, and it is also the defaulted one.
    let (kept_readable, _) = scored(&src, &csv("2"));
    let (kept_defaulted, _) = scored(&src, &csv("x"));
    assert_eq!(
        (kept_readable, kept_defaulted),
        (3, 3),
        "the TIME rule must delete the t=4 row under both spellings, so the compartment \
         changed nothing"
    );
    assert_eq!(
        warning_text(&src, &csv("x")),
        None,
        "a row deleted for a DV reason must not be reported as a compartment guess"
    );
}

#[test]
fn a_deleted_evid_2_row_is_reported_even_though_a_kept_one_is_not() {
    // `n_dose` / `n_obs` deliberately count only rows the fit uses, so an `EVID=2`
    // covariate-change marker with a defaulted CMT is *not* reported when it is kept
    // — it changes no compartment. Deleting it is different: the marker is gone from
    // the fit, and a guessed compartment is why. Asserted in both directions on the
    // same row, so the asymmetry is a decision rather than an accident of where the
    // counter sits.
    let rows = |spelling: &str| {
        format!(
            "ID,TIME,DV,EVID,AMT,CMT,MDV,WT\n\
             1,0,.,1,100,1,1,70\n\
             1,1,1.60,0,.,1,0,70\n\
             1,2,.,2,.,{spelling},1,80\n\
             1,8,0.85,0,.,1,0,80\n\
             1,12,0.60,0,.,1,0,80\n"
        )
    };
    let with_cov = |selection: &str| {
        format!(
            r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[covariates]
  WT continuous

[individual_parameters]
  CL = TVCL * (WT/70) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
{selection}
"#
        )
    };

    // Kept: the EVID=2 row survives, and its guessed compartment selects nothing.
    let kept = with_cov("");
    assert_eq!(
        warning_text(&kept, &rows("x")),
        None,
        "a kept EVID=2 row's compartment addresses nothing, so it is not a guess \
         worth reporting"
    );

    // Deleted by a CMT rule: the marker leaves the fit because of the guess.
    let deleted = with_cov("\n[data_selection]\n  ignore = CMT == 1 && EVID == 2\n");
    let msg = warning_text(&deleted, &rows("x"))
        .expect("an EVID=2 row deleted by a CMT rule must be reported");
    assert!(
        msg.contains("1 row(s) were removed from the fit by a [data_selection] condition"),
        "the deleted marker must be counted: {msg}"
    );
}

#[test]
fn ferx_check_reads_unfiltered_so_it_answers_a_different_question() {
    // Recorded rather than asserted away. `validate_model_file` reads with
    // `filter: None` (`api/validation.rs`), so on a dataset whose defaulted row the
    // filter deletes, `ferx check` counts that row as a plain observation while
    // `fit()` counts it as a deletion. Both report `W_CMT_DEFAULTED` — they agree on
    // the finding — but not on the sentence, and a reader comparing the two should
    // know why.
    //
    // This is also why `warns_on_read` exists: the first version of the mirror case
    // above used `ferx check` as its oracle and passed green against a reader that
    // counted nothing at all, because check never applied the filter that caused the
    // defect.
    let src = model_src(IGNORE_CMT1_AT_T4);
    assert!(
        warns(&src, &csv("x")),
        "ferx check reports it (having counted the row as a kept observation)"
    );
    let fit_msg = warning_text(&src, &csv("x")).expect("and fit() reports it too");
    assert!(
        fit_msg.contains("0 dose row(s) and 0 observation row(s)")
            && fit_msg.contains("removed from the fit"),
        "but fit()'s sentence is about a deletion: {fit_msg}"
    );
}

#[test]
fn fit_from_files_reports_the_filter_its_own_read_applied() {
    // The #1423 review's second finding, and the sharper half of the defect: this
    // entry point — the one ferx-r calls — reads its population through
    // `build_selection_filter_merged`, so the model file's `[data_selection]` DOES
    // filter the fit, but it used to hand `fit()` only the *caller's* options, whose
    // clause lists are empty. `CmtConsumer::DataSelectionFilter` was therefore asked
    // about a filter that was not the one that ran, and withheld the warning on a fit
    // the guessed compartment had genuinely changed.
    //
    // Measured on the same fixture as the rest of this file: 4 records scored against
    // 3, and before the fix no `W_CMT_DEFAULTED` in `FitResult.warnings` either way.
    // `run_model_with_data` (the CLI path) passes `parsed.fit_options` and was always
    // correct, which is why every other test here missed it.
    let src = model_src(IGNORE_CMT);
    let m = temp(&src, ".ferx");
    let d = temp(&csv("x"), ".csv");
    // No `[data_selection]` in the caller's options at all — the whole point is that
    // the clause reaches the predicate from the *model file*.
    let opts = FitOptions {
        outer_maxiter: 0,
        ..FitOptions::default()
    };
    assert!(
        opts.ignore_exprs.is_empty() && opts.accept_exprs.is_empty(),
        "the caller must supply no clauses, or this passes for the wrong reason"
    );
    let result = ferx_core::fit_from_files(
        m.path().to_str().unwrap(),
        Some(d.path().to_str().unwrap()),
        None,
        Some(opts),
    )
    .expect("a 0-iteration evaluation returns");

    // The filter really did run on this read — otherwise there is nothing to warn
    // about and the assertion below would be vacuous.
    let n: usize = result.n_obs;
    let unfiltered = temp(&model_src(NO_SELECTION), ".ferx");
    let baseline = ferx_core::fit_from_files(
        unfiltered.path().to_str().unwrap(),
        Some(d.path().to_str().unwrap()),
        None,
        Some(FitOptions {
            outer_maxiter: 0,
            ..FitOptions::default()
        }),
    )
    .expect("baseline returns");
    assert_eq!(
        (n, baseline.n_obs),
        (4, 4),
        "with the cell unreadable the clause matches nothing, so both score every row \
         — the difference this warning is about is against the *readable* spelling"
    );

    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.starts_with("W_CMT_DEFAULTED")),
        "fit_from_files must report the compartment it guessed, since its own read \
         filtered on it: {:?}",
        result.warnings
    );
}

#[test]
fn a_filter_deleted_row_is_reported_by_every_entry_point_including_predict() {
    // The #1423 review's sixth finding. `predict()` / `simulate()` / the adaptive
    // driver are handed a `Population` the caller already read, so they have no
    // `&FitOptions` and cannot answer "does a `[data_selection]` clause read CMT?" —
    // they passed `FitOptions::default()` and therefore suppressed a warning whose
    // entire content was rows the filter had deleted on a guessed compartment.
    // `read_population_for_simulation` is public and takes a `SelectionFilter`, so
    // ferx-r can reach exactly that state.
    //
    // The fix does not thread options through four public entry points: the reader
    // only counts a deleted row when the rule that removed it actually read `CMT`, so
    // the finding carries its own proof and is never suppressed. This pins that the
    // proof survives on the entry point that has nothing else to go on.
    let src = model_src(IGNORE_CMT1_AT_T4);
    let d = temp(&csv("x"), ".csv");
    let parsed = parse_full_model(&src).expect("model parses");
    let opts = &parsed.fit_options;
    let filter = SelectionFilter::from_opts(&opts.ignore_exprs, &opts.accept_exprs, &[])
        .expect("the clause parses");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        d.path().to_str().unwrap(),
        None,
        None,
        Some(&filter),
        &parsed.column_map,
    )
    .expect("dataset loads");

    // The reader really did delete a row on the guessed compartment, else there is
    // nothing for the entry point to report and the assertion below is vacuous.
    assert!(
        population
            .warnings
            .iter()
            .any(|w| w.contains("removed from the fit by a [data_selection] condition")),
        "the reader must report the deletion: {:?}",
        population.warnings
    );

    // `predict_diag` carries the same bundle as `simulate`/`simulate_adaptive`
    // (`non_fit_diagnostics`), so one of the three is enough to pin the shared filter.
    let out =
        ferx_core::predict_diag(&parsed.model, &population, &parsed.model.default_params).unwrap();
    assert!(
        out.warnings
            .iter()
            .any(|w| w.starts_with("W_CMT_DEFAULTED")),
        "predict() must report a compartment guess that deleted rows from the data it \
         was handed: {:?}",
        out.warnings
    );
}

#[test]
fn predict_still_suppresses_a_cmt_guess_that_addresses_nothing() {
    // The control, and the reason the arm above is keyed on the deletion rather than
    // on "this is a W_CMT_DEFAULTED". A one-compartment model with no per-CMT
    // anything and no filter reads `CMT` for nothing, so `predict()` must stay quiet —
    // otherwise the change above would have widened the warning to every model.
    let src = model_src(NO_SELECTION);
    let d = temp(&csv("x"), ".csv");
    let parsed = parse_full_model(&src).expect("model parses");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        d.path().to_str().unwrap(),
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("dataset loads");
    assert!(
        population
            .warnings
            .iter()
            .any(|w| w.starts_with("W_CMT_DEFAULTED")),
        "the reader still counts the guess — it is the model-aware filter that drops it"
    );
    let out =
        ferx_core::predict_diag(&parsed.model, &population, &parsed.model.default_params).unwrap();
    assert!(
        !out.warnings
            .iter()
            .any(|w| w.starts_with("W_CMT_DEFAULTED")),
        "nothing on this model reads CMT, so the guess changed no number: {:?}",
        out.warnings
    );
}

#[cfg(feature = "survival")]
/// A TTE-only model whose single endpoint sits at `cmt`, with no `[error_model]`
/// block — so the parser leaves its per-CMT error map empty and there is no Gaussian
/// grid for a row to fall into.
fn tte_only_src(cmt: usize) -> String {
    format!(
        r#"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = {cmt}
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
"#
    )
}

#[cfg(feature = "survival")]
/// Event rows on `cmt`, with the first subject's cell written as `spelling`.
fn tte_csv(spelling: &str, other: usize) -> String {
    format!(
        "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
         1,12,1,0,.,{spelling},0\n\
         2,30,0,0,.,{other},0\n\
         3,22,1,0,.,{other},0\n"
    )
}

#[cfg(feature = "survival")]
#[test]
fn a_lone_endpoint_at_the_default_compartment_is_not_reported() {
    // The endpoint channel's documented false positive, closed. A TTE-only model whose
    // only endpoint is at `cmt = 1` has exactly one place a row can go: the reader's
    // fallback IS the endpoint, and a row explicitly assigned any other compartment
    // lands in the Gaussian vectors, where the empty per-CMT error map stops it with
    // `E_PER_CMT_ERROR_MODEL`. So a defaulted cell changes nothing, getting it wrong
    // on purpose is loud, and the warning was pure noise.
    //
    // That error code is asserted below rather than described. The first version of
    // this comment named `E_ENDPOINT_UNROUTED`, which is a different condition — it
    // detects Gaussian observations sitting on a *declared endpoint* CMT because the
    // population was read without endpoint routing (#1199) — and nothing here would
    // have caught the wrong claim (Codex review of #1440).
    //
    // Measured, not asserted from the predicate: the objective is identical under both
    // spellings, which is what "changes nothing" means here.
    let src = tte_only_src(1);
    let (n_readable, nll_readable) = scored(&src, &tte_csv("1", 1));
    let (n_defaulted, nll_defaulted) = scored(&src, &tte_csv("x", 1));
    assert_eq!(
        (n_readable, n_defaulted),
        (0, 0),
        "TTE rows are event records, not Gaussian observations"
    );
    assert_eq!(
        nll_readable.to_bits(),
        nll_defaulted.to_bits(),
        "the guessed compartment must change nothing on this model: \
         {nll_readable} vs {nll_defaulted}"
    );
    assert!(
        !warns_on_read(&src, &tte_csv("x", 1)),
        "a lone endpoint at the default compartment leaves nothing to guess wrong"
    );

    // The "loud, not silent" half of the argument for suppressing it: a row that
    // really is on another compartment is refused, with the code named above.
    let m = temp(&src, ".ferx");
    let off_endpoint = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                        1,12,1,0,.,1,0\n\
                        2,20,0,0,.,1,0\n\
                        3,15,5.0,0,.,2,0\n";
    let d = temp(off_endpoint, ".csv");
    let report = validate_model_file(m.path().to_str().unwrap(), Some(d.path().to_str().unwrap()));
    let codes: Vec<&str> = report
        .diagnostics
        .iter()
        .filter(|x| x.code.starts_with('E'))
        .map(|x| x.code.as_str())
        .collect();
    assert_eq!(
        codes,
        vec!["E_PER_CMT_ERROR_MODEL"],
        "an observation outside the lone endpoint must be refused, and by this code — \
         the suppression above is only safe because getting it wrong is loud"
    );
}

#[cfg(feature = "survival")]
/// Competing risks: two TTE endpoints, at `cmt = 1` and `cmt = 2`. No
/// `[error_model]`, so still endpoint-only — but the routing names more than the
/// default compartment.
fn competing_risks_src() -> String {
    r#"
[parameters]
  theta TVA(0.05, 0.001, 10.0)
  theta TVB(0.03, 0.001, 10.0)
  omega ETA_A ~ 0.09

[event_model cause_a]
  cmt    = 1
  family = exponential
  scale  = TVA * exp(ETA_A)

[event_model cause_b]
  cmt    = 2
  family = exponential
  scale  = TVB
"#
    .to_string()
}

#[cfg(feature = "survival")]
#[test]
fn an_endpoint_model_that_routes_more_than_the_default_is_still_reported() {
    // The straddle, and the half that keeps the fix above from being a blanket
    // silence for endpoint-only models. With endpoints at `cmt = 1` AND `cmt = 2`, a
    // defaulted row keys to 1 — a real endpoint — so it is silently scored against
    // `cause_a` when the dataset meant `cause_b`. That is the #1404 measurement this
    // channel exists for (OFV 27.8497 against 28.3610 on its fixture), and
    // `routes_only(DEFAULT_CMT)` is false here, so the suppression must not apply.
    //
    // A single endpoint at `cmt = 2` is deliberately NOT the straddle used: there the
    // defaulted row leaves the endpoint for a Gaussian grid the model has no error
    // model for, and ferx stops with `E_PER_CMT_ERROR_MODEL` — loud, not silent, so it
    // is not what this warning is for. The end-to-end version of that shape, on a
    // model that *does* carry an `[error_model]` and so takes the row silently, is
    // `tests/cmt_endpoint_scope.rs`.
    let src = competing_risks_src();
    assert!(
        warns_on_read(&src, &tte_csv("x", 2)),
        "a defaulted row lands on cause_a although the dataset named cause_b"
    );
    assert!(
        !warns_on_read(&src, &tte_csv("1", 2)),
        "control: no defaulted cell, no finding"
    );
}
