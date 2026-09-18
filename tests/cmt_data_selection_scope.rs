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
//! > depend on the compartment the reader chose **if and only if** `ferx check`
//! > raises `W_CMT_DEFAULTED` on that dataset.
//!
//! The left side is *measured* — the records are read both ways and counted, and the
//! objective is evaluated on what survived — so the expectation cannot drift into a
//! restatement of the predicate. The straddle (a clause on another column; no clause
//! at all) is asserted in the same loop, because a one-sided table agrees with a
//! predicate stuck at `true`.
//!
//! Tier 2: no `fit()`. The objective is `stats::likelihood::individual_nll` summed
//! over the subjects at the model's own initial estimates — the same quantity the
//! issue's OFV measurements moved, without an optimizer loop.

use ferx_core::api::read_population_for;
use ferx_core::io::datareader::SelectionFilter;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::stats::likelihood::individual_nll;
use ferx_core::validate_model_file;
use ferx_core::OmegaMatrix;
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

#[test]
fn the_warning_fires_exactly_where_the_filter_reads_the_chosen_compartment() {
    // `(label, block, expected records kept on the readable dataset)`. The counts are
    // stated so a clause that silently stops firing — the way `ignore = CMT == 2`
    // does when the cell is unreadable — is visible as a number rather than inferred.
    let cases: [(&str, &str, usize); 4] = [
        ("no [data_selection]", NO_SELECTION, 4),
        ("ignore = CMT == 2", IGNORE_CMT, 3),
        ("accept = CMT == 1", ACCEPT_CMT, 3),
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
        let warned = warns(&src, &csv("x"));
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
        warns(&src, &csv("x")),
        "and ferx must say so — this is the `ok — 0 warning(s)` the issue measured"
    );
}
