//! `W_COVARIATE_MISSING` is raised only for a covariate a relation of the model
//! reads (#2599).
//!
//! The defect: the reader warned for **every** dataset column with a missing cell,
//! including columns no relation reads, with a message claiming "a relation that reads
//! it is then NaN". One benchmark fit printed eight of them for lab values and IDs the
//! model never touched, burying the warnings that matter.
//!
//! One fixture holds both sides of the gate. Read, with gaps, must warn:
//! `WT` (`[individual_parameters]`), `CRCL` (`[scaling]` only), `LAB` (`[derived]`
//! only — a name `referenced_covariates` deliberately does not hold, since a
//! `[derived]` read is not a required column) and `OUT` (`[output]` only, echoed
//! into the sdtab). Unread, with gaps, must not: `UNUSED_ALL` (every cell `.`),
//! `UNUSED_SOME` (filled for subject 1 only) — not even when the caller declares or
//! lists them, since no relation reads them either way — and `LAB_RATIO`, which
//! `[output]` names but the `[derived]` column of that name fills, so the echo never
//! reads the data column. The model-blind reader on the same file warns for all of
//! them, which is what
//! proves the unread columns are read as covariate columns with gaps: the silence on
//! the model-aware read is the gate, not a fixture that never had a gap.
//!
//! Mutations that must redden this file: dropping the `warns_missing_covariate` filter
//! in the reader (unread columns warn again); making it return `false` (read
//! covariates go silent); not registering `[scaling]` covariates (`CRCL` goes silent);
//! dropping the `[derived]` reads from `columns_read_by_relations` (`LAB` goes silent);
//! dropping the `[output]` echo columns (`OUT` goes silent); dropping the filter
//! check (`KEEP` goes silent); counting a string comparison as a read (`FLAG`
//! warns); matching `referenced_columns` ignoring case (`wt` warns); matching
//! `[derived]` reads ignoring case (`mX` warns) or verbatim only (`ALB` goes silent); dropping the `[derived]`
//! shadow from `output_echo_columns` (`LAB_RATIO` warns); re-adding the declared / `covariate_columns` widening (the last two tests).

use crate::api::read_population_for;
use crate::io::datareader::SelectionFilter;
use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::Population;
use std::io::Write;

/// `WT` is read in `[individual_parameters]`, `CRCL` only in `[scaling]`, `LAB` only
/// in `[derived]`, `OUT` only in `[output]`; `[output]`'s `LAB_RATIO` is the derived
/// column, not the data column of that name. `[derived]` resolves a column by its
/// verbatim, uppercase or lowercase spelling only: `alb` reads the `ALB` column,
/// `Mx` reads nothing — not the `mX` column beside it.
const MODEL: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.02\n\
    [individual_parameters]\n  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  ode(states=[central])\n\
    [odes]\n  d/dt(central) = -(CL / V) * central\n\
    [scaling]\n  y = central / V * CRCL / 100\n\
    [error_model]\n  DV ~ proportional(PROP)\n\
    [derived]\n  LAB_RATIO = LAB / IPRED\n  ALB_RATIO = alb / IPRED\n  MX_RATIO = Mx / IPRED\n\
    [output]\n  OUT LAB_RATIO\n";

/// The same model with the never-read `UNUSED_SOME` declared in `[covariates]`.
fn model_declaring_unused() -> String {
    MODEL.replace(
        "[error_model]",
        "[covariates]\n  WT continuous\n  CRCL continuous\n  LAB continuous\n  OUT continuous\n  UNUSED_SOME continuous\n\
         [error_model]",
    )
}

/// Subject 2 has no value in any covariate column; `UNUSED_ALL` is `.` everywhere,
/// the rest are filled for subject 1.
///
/// `FLAG` is a label column, `KEEP` in every row: no cell is missing, though the
/// numeric map holds the text as `NaN`, so the model-blind read warns for it.
///
/// `KEEP` is read only by the `[data_selection]` filter of
/// [`an_unread_column_read_by_the_row_filter_warns`]; `wt` is an unread column whose
/// name differs from the read `WT` only in case — a model covariate is matched
/// case-sensitively, so reading `WT` must not make `wt` warn.
const DATA: &str =
    "ID,TIME,DV,EVID,AMT,CMT,WT,CRCL,LAB,OUT,LAB_RATIO,KEEP,FLAG,ALB,mX,wt,UNUSED_ALL,UNUSED_SOME\n\
    1,0,.,1,100,1,70,100,5,1,9,0,KEEP,40,2,70,.,3\n\
    1,1,5.0,0,.,1,70,100,5,1,9,0,KEEP,40,2,70,.,3\n\
    1,4,3.0,0,.,1,70,100,5,1,9,0,KEEP,40,2,70,.,3\n\
    2,0,.,1,100,1,.,.,.,.,.,.,KEEP,.,.,.,.,.\n\
    2,1,4.0,0,.,1,.,.,.,.,.,.,KEEP,.,.,.,.,.\n\
    2,4,2.5,0,.,1,.,.,.,.,.,.,KEEP,.,.,.,.,.\n";

fn data_file() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(DATA.as_bytes()).unwrap();
    f
}

/// The `W_COVARIATE_MISSING` covariate names in `pop.warnings`, sorted.
fn warned(pop: &Population) -> Vec<String> {
    let mut names: Vec<String> = pop
        .warnings
        .iter()
        .filter_map(|w| w.strip_prefix("W_COVARIATE_MISSING: covariate "))
        .map(|rest| rest.split_whitespace().next().unwrap().to_string())
        .collect();
    names.sort();
    names
}

fn read(model_src: &str, fallback: Option<&[&str]>) -> Population {
    read_filtered(model_src, fallback, None)
}

fn read_filtered(
    model_src: &str,
    fallback: Option<&[&str]>,
    filter: Option<&SelectionFilter>,
) -> Population {
    let parsed = parse_full_model(model_src).expect("model parses");
    let f = data_file();
    read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        f.path().to_str().unwrap(),
        fallback,
        None,
        filter,
        &[],
    )
    .expect("model-aware read")
    .0
}

/// The control: a model-blind read cannot tell which columns are read, so it warns for
/// every covariate column with a gap — including the two the model never reads.
#[test]
fn the_model_blind_reader_warns_for_every_gapped_column() {
    let f = data_file();
    let pop = read_nonmem_csv(f.path(), None, None).unwrap();
    assert_eq!(
        warned(&pop),
        [
            "ALB",
            "CRCL",
            "FLAG",
            "KEEP",
            "LAB",
            "LAB_RATIO",
            "OUT",
            "UNUSED_ALL",
            "UNUSED_SOME",
            "WT",
            "mX",
            "wt"
        ],
        "{:?}",
        pop.warnings
    );
}

/// Both sides of the gate in one read: the three covariates a relation reads warn,
/// the two columns none reads stay quiet — although they are still read, with gaps.
#[test]
fn an_unread_column_with_gaps_does_not_warn() {
    let pop = read(MODEL, None);
    assert_eq!(
        warned(&pop),
        ["ALB", "CRCL", "LAB", "OUT", "WT"],
        "{:?}",
        pop.warnings
    );
    // The unread columns are still in the population with their gaps, so the silence
    // is the gate and not a column the reader dropped.
    assert!(pop.subjects[1].covariates["UNUSED_SOME"].is_nan());
    assert_eq!(pop.subjects[0].covariates["UNUSED_SOME"], 3.0);
    assert!(pop
        .subjects
        .iter()
        .all(|s| s.covariates["UNUSED_ALL"].is_nan()));
    assert!(pop.subjects[1].covariates["LAB_RATIO"].is_nan());
    assert!(
        !pop.warnings.iter().any(|w| w.contains("UNUSED")
            || w.contains("LAB_RATIO")
            || w.contains("KEEP")
            || w.contains("FLAG")
            || w.contains("covariate mX ")
            || w.contains("covariate wt ")),
        "no warning may name an unread column: {:?}",
        pop.warnings
    );
}

/// The parser records `[derived]`'s reads, minus its built-ins — the input the
/// reader's gate takes them from, since `referenced_covariates` does not hold them.
/// Hand-built fixtures (`IndivParamPartials::empty()`) record none.
#[test]
fn a_derived_expression_records_the_names_it_reads() {
    let parsed = parse_full_model(MODEL).expect("model parses");
    assert_eq!(
        parsed.model.derived_reads(),
        ["LAB", "alb", "Mx"],
        "IPRED is a built-in, not a column"
    );
    assert!(
        !parsed
            .model
            .referenced_covariates
            .iter()
            .any(|c| c == "LAB"),
        "a [derived]-only read is not a required column, so the reader needs `derived_reads`"
    );
    assert!(crate::types::IndivParamPartials::empty()
        .derived_reads
        .is_empty());
}

/// Listing an unread column in `covariate_columns` does not make a relation read it,
/// so its gap is still not reported.
#[test]
fn a_caller_listed_but_unread_column_does_not_warn() {
    let pop = read(MODEL, Some(&["WT", "CRCL", "LAB", "OUT", "UNUSED_SOME"]));
    assert_eq!(
        warned(&pop),
        ["CRCL", "LAB", "OUT", "WT"],
        "{:?}",
        pop.warnings
    );
    assert!(pop.subjects[1].covariates["UNUSED_SOME"].is_nan());
}

/// Likewise for a column declared in `[covariates]` that no relation reads.
#[test]
fn a_declared_but_unread_column_does_not_warn() {
    let pop = read(&model_declaring_unused(), None);
    assert_eq!(
        warned(&pop),
        ["CRCL", "LAB", "OUT", "WT"],
        "{:?}",
        pop.warnings
    );
    assert!(pop.subjects[1].covariates["UNUSED_SOME"].is_nan());
}

/// A `[data_selection]` clause reads its column too: `KEEP == 1` is false for a
/// `NaN` cell, so subject 2 is silently kept. Both sides in one fixture — the same
/// read without the filter (every other test here) leaves `KEEP` quiet.
#[test]
fn an_unread_column_read_by_the_row_filter_warns() {
    let filter = SelectionFilter::from_opts(&["KEEP == 1".to_string()], &[], &[]).unwrap();
    let pop = read_filtered(MODEL, None, Some(&filter));
    assert_eq!(
        warned(&pop),
        ["ALB", "CRCL", "KEEP", "LAB", "OUT", "WT"],
        "{:?}",
        pop.warnings
    );
    assert_eq!(pop.subjects.len(), 2, "the NaN cell keeps subject 2");
}

/// A string comparison reads the raw cell, not the parsed one: `FLAG == DROP` on a
/// `KEEP` label is no missing value, although the numeric map holds it as `NaN`. Both
/// sides in one filter — the numeric `KEEP == 1` beside it still warns.
#[test]
fn a_column_read_only_by_a_string_comparison_does_not_warn() {
    let filter = SelectionFilter::from_opts(
        &["KEEP == 1".to_string(), "FLAG == DROP".to_string()],
        &[],
        &[],
    )
    .unwrap();
    assert!(
        filter.needs_str_covariates(),
        "FLAG == DROP is a string clause"
    );
    let pop = read_filtered(MODEL, None, Some(&filter));
    assert_eq!(
        warned(&pop),
        ["ALB", "CRCL", "KEEP", "LAB", "OUT", "WT"],
        "{:?}",
        pop.warnings
    );
    assert_eq!(pop.subjects.len(), 2, "no row holds FLAG = DROP");
}
