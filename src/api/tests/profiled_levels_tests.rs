//! `profile`d θ level blocks (#1065): the per-subject active sets, the route and
//! scope refusals, the parameter count and the output fields.

use super::*;
use crate::parser::model_parser::parse_full_model;
use crate::types::{CompiledModel, Population, Subject};

/// An algebraic MBMA model: `BASE = TVE0 + <read> + ETA_E0`, additive error. `decl`
/// is the block's declaration line.
fn mbma(decl: &str, read: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVE0(45.0, 0.0, 200.0)
  {decl}
  omega ETA_E0 ~ 4.0
  sigma ADD_ERR ~ 1.0 FIX

[covariates]
  STUDY categorical

[individual_parameters]
  BASE = TVE0 + {read} + ETA_E0

[structural_model]
  y = BASE

[error_model]
  DV ~ additive(ADD_ERR)
"#
    )
}

/// Two arms (subjects) per study, `n_studies` studies, each arm observed at
/// `TIME = 1..=n_times`; `STUDY` a subject-level covariate. Two subjects share each
/// study so a study-level block and a per-subject η are told apart. No doses: the
/// model is compartment-free.
fn population(n_studies: usize, n_times: usize) -> Population {
    let subjects = (0..2 * n_studies)
        .map(|i| {
            let mut covariates = HashMap::new();
            covariates.insert("STUDY".to_string(), (i / 2 + 1) as f64);
            Subject {
                id: format!("{}", i + 1),
                obs_times: (1..=n_times).map(|t| t as f64).collect(),
                observations: vec![45.0; n_times],
                obs_cmts: vec![1; n_times],
                cens: vec![0; n_times],
                covariates,
                ..Subject::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["STUDY".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Bind `text` against `pop` (writing the level columns), returning the model.
fn bind(text: &str, pop: &mut Population) -> CompiledModel {
    let mut parsed = parse_full_model(text).unwrap_or_else(|e| panic!("{e}"));
    crate::api::bind_theta_levels(&mut parsed, text, pop).unwrap_or_else(|e| panic!("{e}"));
    parsed.model
}

// ── ActiveLevels ────────────────────────────────────────────────────────────

/// A block keyed on a subject-constant column: each subject reads one level, its
/// own study's, from the baseline map (no per-row snapshots are written).
#[test]
fn a_subject_constant_index_gives_each_subject_its_one_level() {
    let mut pop = population(3, 4);
    let model = bind(
        &mbma("theta PLACEBO[STUDY](0.0) profile", "PLACEBO"),
        &mut pop,
    );
    assert!(
        pop.subjects.iter().all(|s| s.obs_covariates.is_empty()),
        "the fixture must stay on the baseline-only path"
    );
    let active = active_levels(&model, &pop);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].block, "PLACEBO");
    assert_eq!(
        active[0].per_subject,
        [vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]]
    );
}

/// A block keyed on `[STUDY, TIME]`: the index moves row by row, so each subject
/// reads its own four cells. Reading the baseline alone sees one cell per subject.
#[test]
fn a_per_row_index_gives_each_subject_every_cell_it_observes() {
    let mut pop = population(3, 4);
    let model = bind(
        &mbma("theta PLACEBO[STUDY, TIME](0.0) profile", "PLACEBO"),
        &mut pop,
    );
    assert!(
        pop.subjects.iter().all(|s| !s.obs_covariates.is_empty()),
        "the fixture must take the per-row path"
    );
    let active = active_levels(&model, &pop);
    let study = |s: usize| (4 * s..4 * s + 4).collect::<Vec<usize>>();
    assert_eq!(
        active[0].per_subject,
        [study(0), study(0), study(1), study(1), study(2), study(2)]
    );
}

/// The counted form, read through the user's own plain column, gives the same sets
/// as the column form; an unprofiled block gets no entry at all.
#[test]
fn the_counted_form_reads_its_plain_index_column_and_unprofiled_blocks_are_skipped() {
    let mut pop = population(3, 2);
    let model = bind(
        &mbma("theta PLACEBO[3](0.0) profile", "PLACEBO[STUDY]"),
        &mut pop,
    );
    let active = active_levels(&model, &pop);
    assert_eq!(
        active[0].per_subject,
        [vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]]
    );

    let mut pop = population(3, 2);
    let plain = bind(&mbma("theta PLACEBO[3](0.0)", "PLACEBO[STUDY]"), &mut pop);
    assert!(active_levels(&plain, &pop).is_empty());
}
