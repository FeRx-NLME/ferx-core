//! `W_COVARIATE_MISSING` is raised only for a covariate the model reads (#2599).
//!
//! The defect: the reader warned for **every** dataset column with a missing cell,
//! including columns no relation reads, with a message claiming "a relation that reads
//! it is then NaN". One benchmark fit printed eight of them for lab values and IDs the
//! model never touched, burying the warnings that matter.
//!
//! One fixture holds both sides of the gate: `WT` (read in `[individual_parameters]`)
//! and `CRCL` (read only in `[scaling]`) have gaps and must warn; `UNUSED_ALL` (every cell
//! `.`) and `UNUSED_SOME` (filled for subject 1 only) have gaps and must not. The
//! model-blind reader on the same file warns for all four, which is what proves the
//! unread columns are read as covariate columns with gaps — the silence on the
//! model-aware read is the gate, not a fixture that never had a gap.
//!
//! Mutations that must redden this file: dropping the `warns_missing_covariate` filter
//! in the reader (unread columns warn again — `an_unread_column_with_gaps_does_not_warn`);
//! making it return `false` (read covariates go silent — both model-aware tests); not
//! registering `[scaling]` covariates (`CRCL` goes silent); dropping the `fallback_columns`
//! or `[covariates]` widening in `read_population_for_policy` (the two caller-named tests).

use crate::api::read_population_for;
use crate::parser::model_parser::parse_full_model;
use crate::read_nonmem_csv;
use crate::types::Population;
use std::io::Write;

/// `WT` is read in `[individual_parameters]`, `CRCL` only in `[scaling]`.
const MODEL: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.02\n\
    [individual_parameters]\n  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  ode(states=[central])\n\
    [odes]\n  d/dt(central) = -(CL / V) * central\n\
    [scaling]\n  y = central / V * CRCL / 100\n\
    [error_model]\n  DV ~ proportional(PROP)\n";

/// The same model with the never-read `UNUSED_SOME` declared in `[covariates]`.
fn model_declaring_unused() -> String {
    MODEL.replace(
        "[error_model]",
        "[covariates]\n  WT continuous\n  CRCL continuous\n  UNUSED_SOME continuous\n\
         [error_model]",
    )
}

/// Subject 2 has no `WT` and no `CRCL`; `UNUSED_ALL` is `.` everywhere, `UNUSED_SOME`
/// is filled for subject 1 only.
const DATA: &str = "ID,TIME,DV,EVID,AMT,CMT,WT,CRCL,UNUSED_ALL,UNUSED_SOME\n\
    1,0,.,1,100,1,70,100,.,3\n\
    1,1,5.0,0,.,1,70,100,.,3\n\
    1,4,3.0,0,.,1,70,100,.,3\n\
    2,0,.,1,100,1,.,.,.,.\n\
    2,1,4.0,0,.,1,.,.,.,.\n\
    2,4,2.5,0,.,1,.,.,.,.\n";

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
        ["CRCL", "UNUSED_ALL", "UNUSED_SOME", "WT"],
        "{:?}",
        pop.warnings
    );
}

/// Both sides of the gate in one read: the two covariates the model reads warn, the
/// two columns it does not read stay quiet — although they are still read, with gaps.
#[test]
fn an_unread_column_with_gaps_does_not_warn() {
    let pop = read(MODEL, None);
    assert_eq!(warned(&pop), ["CRCL", "WT"], "{:?}", pop.warnings);
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

/// A column the caller lists in `fallback_columns` is one they asked about, so its gap
/// is reported even though no relation reads it; the unlisted one stays quiet.
#[test]
fn a_caller_listed_column_warns_even_when_unread() {
    let pop = read(MODEL, Some(&["WT", "CRCL", "UNUSED_SOME"]));
    assert_eq!(
        warned(&pop),
        ["CRCL", "UNUSED_SOME", "WT"],
        "{:?}",
        pop.warnings
    );
}

/// A column declared in `[covariates]` is likewise reported; the undeclared, unread
/// `UNUSED_ALL` is not.
#[test]
fn a_declared_column_warns_even_when_unread() {
    let pop = read(&model_declaring_unused(), None);
    assert_eq!(
        warned(&pop),
        ["CRCL", "UNUSED_SOME", "WT"],
        "{:?}",
        pop.warnings
    );
}
