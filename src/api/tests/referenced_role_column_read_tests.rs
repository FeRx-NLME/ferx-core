//! #1936: a model-aware read takes every column the model references, whether or
//! not the model has a `[covariates]` block.
//!
//! Before the fix the no-block path auto-detected only non-role columns, so a
//! model reading its `iov_column` (`if (OCC == 2) …`, NONMEM's `$PK` spelling) or
//! a role column (`CMT`) failed with `E_MISSING_COVARIATE` — while an unrelated
//! `[covariates] WT` line, which routes the read through the declared path and its
//! `extra` columns, made the same model fit.
//!
//! Each test names the mutation that must turn it red. The two reader paths
//! (`read_population_for` and `read_population_routed_by`) are separate mutation
//! targets; `routed_by_reads_what_read_population_for_reads` covers the second.

use std::io::Write;
use std::path::Path;

use crate::api::validation::check_covariates;
use crate::api::{read_population_for, read_population_routed_by};
use crate::parser::model_parser::parse_full_model;
use crate::types::{ParsedModel, Population};

fn data_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}

/// `data/warfarin_iov.csv` with a per-subject `WT` column appended
/// (`WT = 60 + ID`), the `iov_wt.csv` of the #1936 plan.
fn iov_wt_csv() -> tempfile::NamedTempFile {
    let src = std::fs::read_to_string(data_path("warfarin_iov.csv")).unwrap();
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    for (i, line) in src.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if i == 0 {
            writeln!(f, "{line},WT").unwrap();
        } else {
            let id: f64 = line.split(',').next().unwrap().trim().parse().unwrap();
            writeln!(f, "{line},{}", 60.0 + id).unwrap();
        }
    }
    f.flush().unwrap();
    f
}

/// The issue's model, reading `expr` in the occasion factor. `extra_blocks` is
/// appended verbatim (e.g. a `[covariates]` block).
fn model(expr: &str, extra_blocks: &str) -> ParsedModel {
    let text = format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta OCC2_CL(0.5, -0.9, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  OCCF = if ({expr} == 2) 1 + OCC2_CL else 1
  CL = TVCL * OCCF * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
  covariance = false
  maxiter    = 0
{extra_blocks}
"#
    );
    parse_full_model(&text).expect("model parses")
}

/// `read_population_for` with the model's own declarations, `iov_column = OCC`,
/// auto-detect (or the explicit `fallback`) and no filter.
fn read(parsed: &ParsedModel, path: &str, fallback: Option<&[&str]>) -> Population {
    read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        path,
        fallback,
        Some("OCC"),
        None,
        &[],
    )
    .expect("read succeeds")
    .0
}

fn missing_codes(parsed: &ParsedModel, pop: &Population) -> Vec<String> {
    check_covariates(&parsed.model, pop)
        .into_iter()
        .map(|d| d.message)
        .collect()
}

/// T1 — the issue: the model reads its `iov_column` with no `[covariates]` block.
/// Mutation: revert `read_population_for_policy`'s `None ⇒ referenced_columns`
/// arm to `None ⇒ Vec::new()` — `OCC` is no longer a covariate and the check fires.
#[test]
fn no_block_model_reads_its_iov_column() {
    let parsed = model("OCC", "");
    assert!(
        parsed.covariate_decls.is_none(),
        "fixture must have no block"
    );
    let pop = read(&parsed, &data_path("warfarin_iov.csv"), None);
    assert!(
        pop.covariate_names.iter().any(|c| c == "OCC"),
        "OCC must be read as a covariate, got {:?}",
        pop.covariate_names
    );
    assert_eq!(missing_codes(&parsed, &pop), Vec::<String>::new());
}

/// Every record-level value of `name` a subject carries, in record order, as bits.
fn per_record_bits(pop: &Population, name: &str) -> Vec<(String, Vec<Option<u64>>)> {
    pop.subjects
        .iter()
        .map(|s| {
            let get = |m: &std::collections::HashMap<String, f64>| m.get(name).map(|v| v.to_bits());
            let mut v = vec![get(&s.covariates)];
            v.extend(s.dose_covariates.iter().map(get));
            v.extend(s.obs_covariates.iter().map(get));
            v.extend(s.pk_only_covariates.iter().map(get));
            v.extend(s.reset_covariates.iter().map(get));
            (s.id.clone(), v)
        })
        .collect()
}

/// T2 — the twin. The same `OCC`-reading model on the same file, once with an
/// unrelated `[covariates] WT continuous` (the declared path, which already read
/// `OCC` as an undeclared extra) and once with no block. **Straddle:** under the
/// old behaviour the no-block read leaves `OCC` unread, so the pair sits on
/// opposite sides of the gate; asserted below so it cannot become a tautology.
/// Mutation: drop the extras on either path (`None ⇒ Vec::new()` in
/// `read_population_for_policy`, or `undeclared_referenced` returning `[]`).
#[test]
fn no_block_read_is_bit_identical_to_the_declared_read() {
    let csv = iov_wt_csv();
    let path = csv.path().to_str().unwrap();
    let declared = model("OCC", "[covariates]\n  WT continuous\n");
    let bare = model("OCC", "");
    assert!(declared.covariate_decls.is_some() && bare.covariate_decls.is_none());

    let pop_d = read(&declared, path, None);
    let pop_b = read(&bare, path, None);
    assert_eq!(
        missing_codes(&declared, &pop_d),
        Vec::<String>::new(),
        "declared side"
    );
    assert_eq!(
        missing_codes(&bare, &pop_b),
        Vec::<String>::new(),
        "no-block side"
    );

    let d = per_record_bits(&pop_d, "OCC");
    let b = per_record_bits(&pop_b, "OCC");
    // Non-degenerate: both occasions are present on the read records.
    let flat: Vec<u64> = d
        .iter()
        .flat_map(|(_, v)| v.iter().flatten().copied())
        .collect();
    assert!(flat.contains(&1.0f64.to_bits()) && flat.contains(&2.0f64.to_bits()));
    assert_eq!(
        d, b,
        "per-record OCC must be bit-identical across the two reads"
    );
}

/// T3 — the "every working fit unchanged" guard. A model that reads `WT` only, on
/// the auto path: `WT` is auto-detected *and* referenced, and must appear once,
/// in header order. Mutation: a naive `auto ++ extras` union (`["WT", "WT"]`).
#[test]
fn auto_detected_reference_is_not_duplicated() {
    let csv = iov_wt_csv();
    // The occasion factor reads `WT` instead of `OCC`: `referenced_columns == ["WT"]`.
    let parsed = model("WT", "");
    let pop = read(&parsed, csv.path().to_str().unwrap(), None);
    assert_eq!(pop.covariate_names, vec!["WT".to_string()]);
}

/// T4 — a `STANDARD_COLS` role column is read too, not only the `iov_column`.
/// Mutation: an exclusion that special-cases only the `iov_column` (the extras
/// match restricted to `iov_column`).
#[test]
fn no_block_model_reads_a_role_column() {
    let parsed = model("CMT", "");
    let pop = read(&parsed, &data_path("warfarin_iov.csv"), None);
    assert!(
        pop.covariate_names.iter().any(|c| c == "CMT"),
        "CMT must be read, got {:?}",
        pop.covariate_names
    );
    assert_eq!(missing_codes(&parsed, &pop), Vec::<String>::new());
}

/// T5 — the extras match is exact-case. A model reading `Occ` against an `OCC`
/// header stays unread, so the check still reports it (as on the declared path,
/// whose population key is the header's own spelling). Mutation: a
/// case-insensitive extras match (`e.eq_ignore_ascii_case(h)`), which binds the
/// header under `OCC` *and* changes `covariate_names`.
#[test]
fn case_differing_reference_stays_unread() {
    let parsed = model("Occ", "");
    let pop = read(&parsed, &data_path("warfarin_iov.csv"), None);
    assert_eq!(pop.covariate_names, Vec::<String>::new());
    assert_eq!(check_covariates(&parsed.model, &pop).len(), 1);
}

/// T6 — an explicit `covariate_columns` list is read as list ∪ extras. Mutation:
/// drop the union in `read_nonmem_csv_routed`'s `None` arm (read the list alone).
#[test]
fn explicit_list_is_unioned_with_references() {
    let csv = iov_wt_csv();
    let parsed = model("OCC", "");
    let pop = read(&parsed, csv.path().to_str().unwrap(), Some(&["WT"]));
    assert_eq!(
        pop.covariate_names,
        vec!["WT".to_string(), "OCC".to_string()]
    );
    assert_eq!(missing_codes(&parsed, &pop), Vec::<String>::new());
}

/// T7 — the reload / post-hoc reader reads what the fit's reader read, so a
/// `.fitrx` of an `OCC`-reading model is not refused by `PopulationFingerprint`.
/// Mutation: leave `read_population_routed_by` at `&[]`.
#[test]
fn routed_by_reads_what_read_population_for_reads() {
    let parsed = model("OCC", "");
    let path = data_path("warfarin_iov.csv");
    let fit_read = read(&parsed, &path, None);
    let reload = read_population_routed_by(&parsed.model, Path::new(&path), Some("OCC"), &[])
        .expect("reload read succeeds");
    assert!(fit_read.covariate_names.iter().any(|c| c == "OCC"));
    assert_eq!(reload.covariate_names, fit_read.covariate_names);
}

// ── `check_covariates` tells an unread header from an absent one ──
//
// After the reader fix the only way to reach `E_MISSING_COVARIATE` on a name the
// data has is a population built by the model-free `read_nonmem_csv*` and handed
// to `fit()`. Those names get their own message; every other name keeps the
// historical text byte-for-byte.

/// The historical message, spelled out so a drift in it is a red test.
fn historical(names: &str, available: &str) -> String {
    format!(
        "Model references covariate(s) not found in data (case-sensitive): {names}. \
         Available covariate columns: {available}."
    )
}

/// A model-free read of `warfarin_iov.csv` with `OCC` as the `iov_column`: `OCC`
/// and `CMT` are headers it does not take as covariates.
fn model_free_population() -> Population {
    crate::io::datareader::read_nonmem_csv(
        Path::new(&data_path("warfarin_iov.csv")),
        None,
        Some("OCC"),
    )
    .expect("model-free read succeeds")
}

/// Both sides of the gate in one test: `OCC` (an exact header, unread) gets the
/// new text; `OCCX` (a typo, not a header) gets the historical text byte-for-byte,
/// and the two are separate diagnostics. Forcing the partition to either side
/// reddens this test. Each sentence of the new text is pinned by its own
/// assertion, so deleting any one of them names the assertion that dies:
///   S1 "…in the data but were not read as covariates: OCC." → "names the column"
///   S2 "A read that does not know the model takes only…"    → "explains the reader"
///   S3 "Read the data with the model (…) … `covariate_columns`." → "remedy"
#[test]
fn unread_header_and_typo_get_their_own_messages() {
    let parsed = model("OCC + 0 * OCCX", "");
    let pop = model_free_population();
    assert!(pop.input_columns.iter().any(|h| h == "OCC"));
    let diags = check_covariates(&parsed.model, &pop);
    assert_eq!(diags.len(), 2, "{diags:#?}");
    assert!(diags.iter().all(|d| d.code == "E_MISSING_COVARIATE"));

    // The typo side: historical, byte-identical, and naming only the typo.
    assert_eq!(diags[0].message, historical("OCCX", "(none)"));

    // The unread side.
    let m = &diags[1].message;
    // S1 names the column and says it *is* in the data.
    assert!(
        m.contains("in the data but were not read as covariates: OCC."),
        "names the column: {m}"
    );
    // S2 explains why the reader skipped it.
    assert!(
        m.contains(
            "A read that does not know the model takes only the covariate columns it is given"
        ) && m.contains("record columns (ID, TIME, DV, AMT, CMT, ...) and the `iov_column`"),
        "explains the reader: {m}"
    );
    // S3 the remedy, both halves.
    assert!(
        m.contains("Read the data with the model (`read_population_with` / `read_population_for`), which reads every column the model references")
            && m.contains("add the column(s) to `covariate_columns`"),
        "remedy: {m}"
    );
    // Must not say it is missing, nor offer a model-file remedy to a model-free read.
    assert!(
        !m.contains("not found in data") && !m.contains("[covariates]"),
        "{m}"
    );
    assert!(
        !m.contains("OCCX"),
        "the typo belongs to the other diagnostic: {m}"
    );
    assert_eq!(
        diags[1].suggestion.as_deref(),
        Some("read the data with `read_population_with`, or add OCC to `covariate_columns`")
    );
}

/// A role column (`CMT`) alone: one diagnostic, the new one — and a model-aware
/// read of the same file and model reports nothing.
#[test]
fn unread_role_column_alone_gets_only_the_new_message() {
    let parsed = model("CMT", "");
    let diags = check_covariates(&parsed.model, &model_free_population());
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert!(diags[0]
        .message
        .contains("in the data but were not read as covariates: CMT."));
    let aware = read(&parsed, &data_path("warfarin_iov.csv"), None);
    assert!(check_covariates(&parsed.model, &aware).is_empty());
}

/// The cells that must keep the historical text: an in-memory population
/// (`input_columns` empty, so whether the name is a header cannot be known) and a
/// case-differing spelling (`Occ` against the `OCC` header — not an exact header).
/// Mutation: a case-insensitive header match, or treating an empty
/// `input_columns` as "every name is a header".
#[test]
fn in_memory_and_case_differing_keep_the_historical_text() {
    let parsed = model("OCC", "");
    let mut in_memory = model_free_population();
    in_memory.input_columns.clear();
    let d = check_covariates(&parsed.model, &in_memory);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].message, historical("OCC", "(none)"));

    let parsed = model("Occ", "");
    let d = check_covariates(&parsed.model, &model_free_population());
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].message, historical("Occ", "(none)"));
}
