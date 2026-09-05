use super::*;
use std::path::Path;

use ferx_core::BicType;

const EXAMPLES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples");
const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data");

fn issue_example() -> String {
    format!(
        r#"
base = "{EXAMPLES}/two_cpt_oral_cov.ferx"
data = "{DATA}/two_cpt_oral_cov.csv"

[space]
mfl = """
ABSORPTION([INST,FO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])
COVARIATE?(@IIV, @CONTINUOUS, [pow,lin])
"""

[rank]
type   = "bic"
cutoff = 3.84

[strictness]
require_converged    = true
max_condition_number = 1000.0
max_correlation      = 0.95
reject_on_boundary   = true

[run]
threads   = 8
retries   = 3
cache_dir = ".ferx-search"
"#
    )
}

fn load(text: &str) -> SearchConfig {
    SearchConfig::from_str(text, Path::new("/cfg")).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

fn load_err(text: &str) -> String {
    match SearchConfig::from_str(text, Path::new("/cfg")) {
        Ok(_) => panic!("should not load:\n{text}"),
        Err(e) => e,
    }
}

#[test]
fn the_issue_example_loads_and_maps_onto_the_runner() {
    let cfg = load(&issue_example());
    assert_eq!(cfg.dir, Path::new("/cfg"));
    assert!(
        cfg.base.ends_with("two_cpt_oral_cov.ferx"),
        "{:?}",
        cfg.base
    );
    assert!(cfg.data.as_ref().unwrap().ends_with("two_cpt_oral_cov.csv"));
    assert_eq!(cfg.mfl.features().count(), 4);
    assert!(cfg.mfl_source.contains("COVARIATE?(@IIV"));
    assert_eq!(cfg.rank.kind, RankType::Bic);
    assert_eq!(cfg.rank.cutoff, Some(3.84));
    assert_eq!(cfg.run.threads, Some(8));
    assert_eq!(cfg.run.retries, 3);
    assert_eq!(
        cfg.run.cache_dir.as_deref(),
        Some(Path::new(".ferx-search"))
    );
    assert!(cfg.tools.is_empty());

    let opts = cfg.run_options();
    assert_eq!(opts.criterion, Criterion::Bic(BicType::Mixed));
    // Pharmpy's `retries = 3` is three perturbed starts on top of the exact
    // one, not three starts in total.
    assert_eq!(opts.n_starts, 4);
    assert!(!opts.resume);
    assert!(opts.fit_options.is_none());
    assert!(opts.strictness.require_converged);
    assert_eq!(opts.strictness.max_condition_number, Some(1000.0));
    assert_eq!(opts.strictness.max_correlation, Some(0.95));
    assert!(opts.strictness.reject_on_boundary);
    // Keys the file did not state keep ferx-core's defaults.
    let d = Strictness::default();
    assert_eq!(opts.strictness.require_covariance, d.require_covariance);
    assert_eq!(opts.strictness.reject_init_stall, d.reject_init_stall);
    // The runner builds without touching the filesystem.
    let _ = cfg.runner();
}

#[test]
fn relative_paths_are_taken_against_the_config_directory() {
    let cfg = load(
        r#"
base = "m.ferx"
data = "d.csv"
[space]
mfl = "ABSORPTION(FO)"
[run]
cache_dir = "cache"
"#,
    );
    assert_eq!(cfg.base, Path::new("/cfg/m.ferx"));
    assert_eq!(cfg.data.as_deref(), Some(Path::new("/cfg/d.csv")));
    let cfg = load("base = \"/abs/m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n");
    assert_eq!(cfg.base, Path::new("/abs/m.ferx"));
    assert!(cfg.data.is_none());
}

#[test]
fn defaults_when_sections_are_omitted() {
    let cfg = load("base = \"m.ferx\"\n[space]\nmfl = \"LAGTIME(*)\"\n");
    assert_eq!(cfg.rank, RankConfig::default());
    assert_eq!(cfg.strictness, StrictnessConfig::default());
    assert_eq!(cfg.run, RunConfig::default());
    let opts = cfg.run_options();
    let defaults = RunOptions::default();
    assert_eq!(opts.criterion, defaults.criterion);
    assert_eq!(opts.n_starts, defaults.n_starts);
    assert_eq!(
        opts.strictness.require_converged,
        Strictness::default().require_converged
    );
}

#[test]
fn every_rank_type_maps_to_a_criterion_except_penalized() {
    for (name, want) in [
        ("ofv", Criterion::Ofv),
        ("aic", Criterion::Aic),
        ("bic", Criterion::Bic(BicType::Mixed)),
        ("bic_mixed", Criterion::Bic(BicType::Mixed)),
        ("bic_iiv", Criterion::Bic(BicType::Iiv)),
        ("bic_random", Criterion::Bic(BicType::Random)),
        ("bic_fixed", Criterion::Bic(BicType::Fixed)),
    ] {
        let cfg = load(&format!(
            "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[rank]\ntype = \"{name}\"\n"
        ));
        assert_eq!(cfg.run_options().criterion, want, "{name}");
    }
    let e = load_err(
        "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[rank]\ntype = \"penalized\"\n",
    );
    assert!(e.contains("penalized\" is not implemented yet"), "{e}");
    let e =
        load_err("base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[rank]\ntype = \"lrt\"\n");
    assert!(e.contains("unknown variant `lrt`"), "{e}");
}

#[test]
fn unsupported_feature_is_a_load_error_naming_it() {
    let e = load_err("base = \"m.ferx\"\n[space]\nmfl = \"ELIMINATION([FO,MM])\"\n");
    assert!(
        e.starts_with("[space] mfl: the search space asks for 1 feature"),
        "{e}"
    );
    assert!(e.contains("ELIMINATION(MM)"), "{e}");
    assert!(e.contains(super::super::coverage::COVERAGE_DOCS), "{e}");
}

#[test]
fn mfl_syntax_error_is_prefixed_with_the_key() {
    let e = load_err("base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FAST)\"\n");
    assert!(
        e.starts_with("[space] mfl: MFL: in ABSORPTION: `FAST` is not a known"),
        "{e}"
    );
    let e = load_err("base = \"m.ferx\"\n[space]\nmfl = \"LET(X,[A])\"\n");
    assert!(e.contains("the search space is empty"), "{e}");
}

#[test]
fn typos_in_known_sections_are_rejected() {
    let e = load_err("base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[run]\nthread = 8\n");
    assert!(e.contains("unknown field `thread`"), "{e}");
    let e = load_err(
        "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[strictness]\nconverged = true\n",
    );
    assert!(e.contains("unknown field `converged`"), "{e}");
    let e = load_err("base = \"m.ferx\"\n[space]\nmlf = \"ABSORPTION(FO)\"\n");
    assert!(e.contains("unknown field `mlf`"), "{e}");
    let e = load_err("[space]\nmfl = \"ABSORPTION(FO)\"\n");
    assert!(e.contains("missing field `base`"), "{e}");
}

#[test]
fn tool_sections_pass_through_and_stray_scalars_do_not() {
    let cfg = load(
        r#"
base = "m.ferx"
[space]
mfl = "ABSORPTION(FO)"
[covsearch]
algorithm = "scm-forward-then-backward"
p_forward = 0.01
"#,
    );
    let covsearch = cfg.tools.get("covsearch").expect("kept for the tool");
    assert_eq!(
        covsearch.get("algorithm").and_then(|v| v.as_str()),
        Some("scm-forward-then-backward")
    );
    let e = load_err("base = \"m.ferx\"\nbsae = \"x\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n");
    assert!(e.contains("unknown top-level key `bsae`"), "{e}");
}

#[test]
fn a_misspelt_core_section_is_an_error_not_a_tool_section() {
    // `[strictnes]` filed under `tools` would leave the gate at its defaults
    // with nothing to say so.
    let e = load_err(
        "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[strictnes]\n\
         require_converged = true\n",
    );
    assert!(e.contains("unknown section `[strictnes]`"), "{e}");
    assert!(e.contains("[strictness]"), "{e}");
    assert!(e.contains("[covsearch]"), "{e}");
    for tool in TOOL_SECTIONS {
        load(&format!(
            "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[{tool}]\nx = 1\n"
        ));
    }
}

#[test]
fn retries_are_pharmpys_perturbed_starts_on_top_of_the_exact_one() {
    let starts = |retries: usize| {
        load(&format!(
            "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n[run]\nretries = {retries}\n"
        ))
        .run_options()
        .n_starts
    };
    assert_eq!(starts(0), 1, "no retries is a single fit from the initials");
    assert_eq!(starts(1), 2);
    assert_eq!(starts(3), 4);
    // The default stays the runner's default: three starts.
    assert_eq!(RunConfig::default().retries, 2);
    assert_eq!(
        load("base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n")
            .run_options()
            .n_starts,
        RunOptions::default().n_starts
    );
}

#[test]
fn load_reads_the_file_and_names_it_in_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("warfarin.ferxsearch");
    std::fs::write(
        &path,
        "base = \"m.ferx\"\n[space]\nmfl = \"ELIMINATION(MM)\"\n",
    )
    .unwrap();
    let e = SearchConfig::load(&path).expect_err("coverage gap");
    assert!(e.starts_with(&path.display().to_string()), "{e}");
    let e = SearchConfig::load(dir.path().join("missing.ferxsearch")).expect_err("no file");
    assert!(e.contains("cannot read"), "{e}");
    std::fs::write(
        &path,
        "base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n",
    )
    .unwrap();
    let cfg = SearchConfig::load(&path).expect("loads");
    assert_eq!(cfg.dir, dir.path());
    assert_eq!(cfg.base, dir.path().join("m.ferx"));
    assert_eq!(EXTENSION, "ferxsearch");
}

/// The whole path on a shipped example: load the file, read the base model
/// and dataset, resolve the symbols.
#[test]
fn end_to_end_on_the_shipped_covariate_example() {
    let cfg = load(&issue_example());
    let base = cfg.load_base().expect("base model + data");
    assert_eq!(base.prepared.population.covariate_names, vec!["WT", "CRCL"]);
    let resolved = cfg.resolve_space(&base).expect("resolves");
    assert!(resolved.notes.is_empty(), "{:?}", resolved.notes);
    assert_eq!(
        resolved.mfl.render(),
        "ABSORPTION([INST,FO]);PERIPHERALS(0..1);LAGTIME([OFF,ON]);\
         COVARIATE?(CL,WT,[pow,lin]);COVARIATE?(CL,CRCL,[pow,lin]);\
         COVARIATE?(V1,WT,[pow,lin]);COVARIATE?(V1,CRCL,[pow,lin]);\
         COVARIATE?(Q,WT,[pow,lin]);COVARIATE?(Q,CRCL,[pow,lin]);\
         COVARIATE?(V2,WT,[pow,lin]);COVARIATE?(V2,CRCL,[pow,lin]);\
         COVARIATE?(KA,WT,[pow,lin]);COVARIATE?(KA,CRCL,[pow,lin])"
    );
    assert_eq!(resolved.covariate_effects.len(), 5 * 2 * 2);
}

#[test]
fn end_to_end_categorical_symbol_on_a_model_without_the_block_is_loud() {
    // warfarin.ferx has no [covariates] block and its data no covariate columns.
    let text = format!(
        "base = \"{EXAMPLES}/warfarin.ferx\"\ndata = \"{DATA}/warfarin.csv\"\n\
         [space]\nmfl = \"COVARIATE?(@IIV, @CATEGORICAL, cat)\"\n"
    );
    let cfg = load(&text);
    let base = cfg.load_base().expect("base model + data");
    let e = cfg.resolve_space(&base).expect_err("no block");
    assert!(
        e.contains("`@CATEGORICAL` needs a `[covariates]` block"),
        "{e}"
    );
}

/// The shipped example file loads and resolves; it is what the docs page
/// quotes, so a format change has to update both.
#[test]
fn the_shipped_example_file_loads_and_resolves() {
    let path = Path::new(EXAMPLES).join("two_cpt_oral_cov.ferxsearch");
    let cfg = SearchConfig::load(&path).expect("shipped example loads");
    assert_eq!(cfg.base, Path::new(EXAMPLES).join("two_cpt_oral_cov.ferx"));
    assert_eq!(cfg.run.threads, Some(4));
    let base = cfg.load_base().expect("base model + data");
    let resolved = cfg.resolve_space(&base).expect("resolves");
    assert!(resolved.notes.is_empty(), "{:?}", resolved.notes);
    assert_eq!(resolved.covariate_effects.len(), 5 * 2 * 2);
    assert!(resolved
        .mfl
        .render()
        .starts_with("PERIPHERALS(0..1);LAGTIME([OFF,ON]);COVARIATE?(CL,WT,[pow,lin])"));
}
