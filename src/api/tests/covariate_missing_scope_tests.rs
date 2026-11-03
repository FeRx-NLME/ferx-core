//! `W_COVARIATE_MISSING` is raised only for a covariate a relation of the model
//! reads (#2599).
//!
//! The defect: the reader warned for **every** dataset column with a missing cell,
//! including columns no relation reads, with a message claiming "a relation that reads
//! it is then NaN". One benchmark fit printed eight of them for lab values and IDs the
//! model never touched, burying the warnings that matter.
//!
//! One fixture holds both sides of the gate. Read, with gaps, must warn:
//! `WT` (`[individual_parameters]`), `CRCL` (`[scaling]` only) and `LAB` (`[derived]`
//! only — a name `referenced_covariates` deliberately does not hold, since a
//! `[derived]` read is not a required column). Unread, with gaps, must not:
//! `UNUSED_ALL` (every cell `.`) and `UNUSED_SOME` (filled for subject 1 only) — not
//! even when the caller declares or lists them, since no relation reads them either
//! way. The model-blind reader on the same file warns for all five, which is what
//! proves the unread columns are read as covariate columns with gaps: the silence on
//! the model-aware read is the gate, not a fixture that never had a gap.
//!
//! Mutations that must redden this file: dropping the `warns_missing_covariate` filter
//! in the reader (unread columns warn again); making it return `false` (read
//! covariates go silent); not registering `[scaling]` covariates (`CRCL` goes silent);
//! dropping the `[derived]` reads from `columns_read_by_relations` (`LAB` goes silent);
//! re-adding the declared / `covariate_columns` widening (the last two tests).

use crate::api::read_population_for;
use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::Population;
use std::io::Write;

/// `WT` is read in `[individual_parameters]`, `CRCL` only in `[scaling]`, `LAB` only
/// in `[derived]`.
const MODEL: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.02\n\
    [individual_parameters]\n  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  ode(states=[central])\n\
    [odes]\n  d/dt(central) = -(CL / V) * central\n\
    [scaling]\n  y = central / V * CRCL / 100\n\
    [error_model]\n  DV ~ proportional(PROP)\n\
    [derived]\n  LAB_RATIO = LAB / IPRED\n";

/// The same model with the never-read `UNUSED_SOME` declared in `[covariates]`.
fn model_declaring_unused() -> String {
    MODEL.replace(
        "[error_model]",
        "[covariates]\n  WT continuous\n  CRCL continuous\n  LAB continuous\n  UNUSED_SOME continuous\n\
         [error_model]",
    )
}

/// Subject 2 has no `WT`, `CRCL` or `LAB`; `UNUSED_ALL` is `.` everywhere,
/// `UNUSED_SOME` is filled for subject 1 only.
const DATA: &str = "ID,TIME,DV,EVID,AMT,CMT,WT,CRCL,LAB,UNUSED_ALL,UNUSED_SOME\n\
    1,0,.,1,100,1,70,100,5,.,3\n\
    1,1,5.0,0,.,1,70,100,5,.,3\n\
    1,4,3.0,0,.,1,70,100,5,.,3\n\
    2,0,.,1,100,1,.,.,.,.,.\n\
    2,1,4.0,0,.,1,.,.,.,.,.\n\
    2,4,2.5,0,.,1,.,.,.,.,.\n";

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
    let parsed = parse_full_model(model_src).expect("model parses");
    let f = data_file();
    read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        f.path().to_str().unwrap(),
        fallback,
        None,
        None,
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
        ["CRCL", "LAB", "UNUSED_ALL", "UNUSED_SOME", "WT"],
        "{:?}",
        pop.warnings
    );
}

/// Both sides of the gate in one read: the three covariates a relation reads warn,
/// the two columns none reads stay quiet — although they are still read, with gaps.
#[test]
fn an_unread_column_with_gaps_does_not_warn() {
    let pop = read(MODEL, None);
    assert_eq!(warned(&pop), ["CRCL", "LAB", "WT"], "{:?}", pop.warnings);
    // The unread columns are still in the population with their gaps, so the silence
    // is the gate and not a column the reader dropped.
    assert!(pop.subjects[1].covariates["UNUSED_SOME"].is_nan());
    assert_eq!(pop.subjects[0].covariates["UNUSED_SOME"], 3.0);
    assert!(pop
        .subjects
        .iter()
        .all(|s| s.covariates["UNUSED_ALL"].is_nan()));
    assert!(
        !pop.warnings.iter().any(|w| w.contains("UNUSED")),
        "no warning may name an unread column: {:?}",
        pop.warnings
    );
}

/// The parser records `[derived]`'s reads, minus its built-ins — the input the
/// reader's gate takes them from, since `referenced_covariates` does not hold them.
#[test]
fn a_derived_expression_records_the_names_it_reads() {
    let parsed = parse_full_model(MODEL).expect("model parses");
    let spec = &parsed.model.derived_exprs[0];
    assert_eq!(spec.reads, ["LAB"], "IPRED is a built-in, not a column");
    assert!(
        !parsed
            .model
            .referenced_covariates
            .iter()
            .any(|c| c == "LAB"),
        "a [derived]-only read is not a required column, so the reader needs `reads`"
    );
}

/// Listing an unread column in `covariate_columns` does not make a relation read it,
/// so its gap is still not reported.
#[test]
fn a_caller_listed_but_unread_column_does_not_warn() {
    let pop = read(MODEL, Some(&["WT", "CRCL", "LAB", "UNUSED_SOME"]));
    assert_eq!(warned(&pop), ["CRCL", "LAB", "WT"], "{:?}", pop.warnings);
    assert!(pop.subjects[1].covariates["UNUSED_SOME"].is_nan());
}

/// Likewise for a column declared in `[covariates]` that no relation reads.
#[test]
fn a_declared_but_unread_column_does_not_warn() {
    let pop = read(&model_declaring_unused(), None);
    assert_eq!(warned(&pop), ["CRCL", "LAB", "WT"], "{:?}", pop.warnings);
    assert!(pop.subjects[1].covariates["UNUSED_SOME"].is_nan());
}
