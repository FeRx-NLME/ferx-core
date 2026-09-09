//! Tier-2 coverage for `prepare_run` (#1140): the "load a model and its data,
//! but do not fit" entry point.
//!
//! It exists because a tool that runs many fits needs everything
//! `run_model_with_data` does *up to* the fit — data-path resolution against the
//! model's `[data]` block, the `[data_selection]` filter, the covariate-aware
//! read, `theta NAME[...]` level binding, and the initial `ModelParameters` —
//! and previously the only way to get there was to run a fit.
//!
//! No fit happens here, so these return immediately.

use ferx_core::{prepare_run, run_model_with_data};

const MODEL: &str = "examples/warfarin.ferx";
const DATA: &str = "data/warfarin.csv";

#[test]
fn prepare_run_loads_the_model_the_data_and_the_initial_estimates() {
    let p = prepare_run(MODEL, Some(DATA)).expect("warfarin loads");

    assert_eq!(p.parsed.model.name, "warfarin");
    assert!(!p.population.subjects.is_empty());
    assert!(p.population.n_obs() > 0);
    assert_eq!(p.data_path, DATA);
    assert!(p.data_path_warning.is_none());

    // The initial parameters are the model file's, ready to hand to `fit`.
    assert_eq!(p.init_params.theta_names, vec!["TVCL", "TVV", "TVKA"]);
    assert_eq!(p.init_params.theta, vec![0.2, 10.0, 1.5]);
    assert_eq!(p.init_params.theta_lower, vec![0.001, 0.1, 0.01]);
    assert_eq!(p.init_params.omega.dim(), 3);
    assert_eq!(p.init_params.sigma.names, vec!["PROP_ERR"]);

    // Hashes are computed once, up front, for the checkpoint integrity check.
    assert!(p.model_hash.is_some());
    assert!(p.data_hash.is_some());
}

#[test]
fn prepare_run_and_run_model_with_data_read_the_same_population() {
    // `run_model_with_data` is implemented on top of `prepare_run`, and this is
    // what pins that: a tool loading a model must see exactly what the CLI sees,
    // or every fit it runs is quietly against a different dataset.
    //
    // This is the one test here that does fit — there is no other way to reach
    // the entry point's own population. Warfarin is 10 subjects and converges in
    // well under a second, so it stays in the PR job rather than the slow tier.
    let prepared = prepare_run(MODEL, Some(DATA)).expect("prepare");
    let (_, population) = run_model_with_data(MODEL, Some(DATA)).expect("fit");

    assert_eq!(
        prepared.population.subjects.len(),
        population.subjects.len()
    );
    assert_eq!(prepared.population.n_obs(), population.n_obs());
    let prepared_ids: Vec<&str> = prepared
        .population
        .subjects
        .iter()
        .map(|s| s.id.as_str())
        .collect();
    let fitted_ids: Vec<&str> = population.subjects.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(prepared_ids, fitted_ids);
    assert_eq!(prepared.population.dv_column, population.dv_column);
    assert_eq!(
        prepared.population.covariate_names,
        population.covariate_names
    );
}

/// `PreparedRun` holds a `CompiledModel` of closures and so cannot be `Debug`,
/// which rules out `unwrap_err`.
fn error_of(result: Result<ferx_core::PreparedRun, String>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

#[test]
fn a_missing_model_or_dataset_is_an_error() {
    assert!(prepare_run("no-such-model.ferx", Some(DATA)).is_err());
    assert!(prepare_run(MODEL, Some("no-such-data.csv")).is_err());
}

#[test]
fn a_model_without_a_data_block_needs_an_explicit_dataset() {
    // warfarin.ferx has no `[data]` block, so `None` has nothing to fall back
    // on and must say so rather than reading an empty population.
    let err = error_of(prepare_run(MODEL, None));
    assert!(
        err.contains("no dataset specified"),
        "expected the missing-dataset message, got: {err}"
    );
}
