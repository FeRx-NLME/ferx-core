//! Tier-1 tests for #1824: `fit()` honours `FitOptions::bloq_method` per call.
//!
//! **The gap.** Every censored-likelihood, gradient, Hessian and diagnostic path read
//! `model.bloq_method`, which only the parser and the file entry points stamped from the
//! options. `fit` takes `&CompiledModel` (not `Clone`), so it could not stamp, and a direct
//! `fit()` with `bloq_method = M3` in the options on a `drop` model scored the censored rows
//! as ordinary observations, reported `drop`, and warned nothing. Measured at `d43afca9`: the
//! M3 request and the `Drop` control were bit-identical.
//!
//! **The rule.** `FitOptions::bloq_method` is an `Option`: `Some(m)` runs this call under `m`,
//! `None` under the model's own method. `fit` arms it (`types::arm_bloq_override`, carried to
//! the workers by `api::pool::FitScope::bloq`) and every reader asks
//! `CompiledModel::bloq_in_force` (pinned by `tests/bloq_in_force_is_the_one_reader.rs`).
//!
//! Every case reads **which method the fit reports** (`FitResult::bloq_method`, built from
//! `bloq_in_force`) and the objective, against a reference run where the model itself carries
//! the method and the options set no override — the pre-#1824 stamped path.

use super::*;
use crate::types::BloqMethod;

const MODEL: &str = "examples/warfarin_bloq.ferx";
const DATA: &str = "data/warfarin_bloq.csv";

/// `examples/warfarin_bloq.ferx` on `data/warfarin_bloq.csv` (an `m3` file with `CENS = 1`
/// rows), with the model's own method set to `own`.
fn warfarin_bloq(own: BloqMethod) -> (CompiledModel, Population) {
    let mut parsed = crate::parser::model_parser::parse_full_model_file(Path::new(MODEL))
        .expect("parse warfarin_bloq");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        DATA,
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("read warfarin_bloq");
    parsed.model.bloq_method = own;
    (parsed.model, population)
}

/// An evaluation at the initial estimates: `outer_maxiter = 0`, FOCEI (the file's method, so
/// M3 runs with interaction and raises no FOCE-M3 note), no covariance step.
fn opts(bloq: Option<BloqMethod>, threads: usize) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        bloq_method: bloq,
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        threads: Some(threads),
        ..Default::default()
    }
}

fn run_on(own: BloqMethod, bloq: Option<BloqMethod>, threads: usize) -> FitResult {
    let (model, population) = warfarin_bloq(own);
    fit(
        &model,
        &population,
        &model.default_params,
        &opts(bloq, threads),
    )
    .expect("fit")
}

fn run(own: BloqMethod, bloq: Option<BloqMethod>) -> FitResult {
    run_on(own, bloq, 1)
}

/// The fixture must carry censored rows, and the two methods must reach different
/// objectives on it, or every bit-equality below holds whether the override is read or not.
fn straddle(m3: &FitResult, drop: &FitResult) {
    let (_, population) = warfarin_bloq(BloqMethod::Drop);
    assert!(
        population
            .subjects
            .iter()
            .any(|s| s.has_censored_observation()),
        "the fixture carries no censored rows"
    );
    assert!(m3.ofv.is_finite() && drop.ofv.is_finite());
    assert_eq!(m3.bloq_method, "m3", "reference M3 run reports m3");
    assert_eq!(drop.bloq_method, "drop", "reference drop run reports drop");
    assert_ne!(
        m3.ofv.to_bits(),
        drop.ofv.to_bits(),
        "M3 and drop agree on this fixture ({:.17e})",
        m3.ofv
    );
}

/// **B1 — the reported defect: `Some(M3)` on a `drop` model scores M3.** At `d43afca9` this
/// reported `drop` with the drop objective bit for bit.
///
/// Mutations: arm `None` in `fit_unstamped` and drop `FitScope::bloq` (the override never
/// reaches a reader) → label and OFV; revert one likelihood reader to the field → OFV only
/// (and the reader guard).
#[test]
fn a_direct_fit_with_m3_in_the_options_scores_m3_on_a_drop_model() {
    let m3 = run(BloqMethod::M3, None);
    let drop = run(BloqMethod::Drop, None);
    straddle(&m3, &drop);

    let direct = run(BloqMethod::Drop, Some(BloqMethod::M3));
    assert_eq!(direct.bloq_method, "m3", "reported method");
    assert_eq!(
        direct.ofv.to_bits(),
        m3.ofv.to_bits(),
        "direct fit() OFV {:.17e} vs the M3 path {:.17e} (drop {:.17e})",
        direct.ofv,
        m3.ofv,
        drop.ofv
    );
}

/// **B2 — the other direction, which a union rule (#1613's) could not express: an explicit
/// `Some(Drop)` turns M3 off on an `m3` model.**
///
/// Mutation: `bloq_in_force` as "M3 from either side wins" → this test, and only this one.
#[test]
fn an_explicit_drop_in_the_options_overrides_an_m3_model() {
    let m3 = run(BloqMethod::M3, None);
    let drop = run(BloqMethod::Drop, None);
    straddle(&m3, &drop);

    let direct = run(BloqMethod::M3, Some(BloqMethod::Drop));
    assert_eq!(direct.bloq_method, "drop", "reported method");
    assert_eq!(direct.ofv.to_bits(), drop.ofv.to_bits(), "drop objective");
}

/// **B3 — `None` defers to the model, both ways: a caller who parses an `m3` file and passes
/// `FitOptions::default()` stays on M3** (the failure #1824 warned "options win" would cause),
/// **and agreeing options change nothing** (what every file entry point passes).
///
/// Mutation: `bloq_in_force` as "the options win, `None` is `Drop`" → the `None` half.
#[test]
fn no_override_defers_to_the_model_and_agreeing_options_change_nothing() {
    let m3 = run(BloqMethod::M3, None);
    let drop = run(BloqMethod::Drop, None);
    straddle(&m3, &drop);

    let (model, population) = warfarin_bloq(BloqMethod::M3);
    let defaulted = fit(
        &model,
        &population,
        &model.default_params,
        &FitOptions {
            bloq_method: None,
            ..opts(None, 1)
        },
    )
    .expect("fit");
    assert_eq!(defaulted.bloq_method, "m3", "None on an m3 model");

    for (own, want) in [(BloqMethod::M3, &m3), (BloqMethod::Drop, &drop)] {
        let agreeing = run(own, Some(own));
        assert_eq!(agreeing.bloq_method, want.bloq_method, "{own:?}: label");
        assert_eq!(
            agreeing.ofv.to_bits(),
            want.ofv.to_bits(),
            "{own:?}: objective"
        );
    }
}

/// **B4 — the pool's workers carry the override.** On four threads the per-subject solves run
/// on workers that never saw `fit_unstamped`'s guard; only `FitScope` reaches them.
///
/// Mutation: delete the `bloq` install in `FitScope::install_on_worker` and the arm in
/// `FitScope::armed` → the workers score drop and the objective moves.
#[test]
fn the_fit_pool_workers_score_the_override() {
    let m3 = run_on(BloqMethod::M3, None, 4);
    let drop = run_on(BloqMethod::Drop, None, 4);
    straddle(&m3, &drop);

    let direct = run_on(BloqMethod::Drop, Some(BloqMethod::M3), 4);
    assert_eq!(direct.bloq_method, "m3", "reported method");
    assert_eq!(
        direct.ofv.to_bits(),
        m3.ofv.to_bits(),
        "four-thread direct fit() OFV {:.17e} vs M3 {:.17e}",
        direct.ofv,
        m3.ofv
    );
}

/// **B5 — the override ends with the call.** Neither the calling thread nor a later default
/// `fit()` on the same model sees it; and a nested `None` reads its own model, not the
/// enclosing override.
///
/// Mutation: make `BloqOverrideGuard::drop` a no-op → the thread still reads M3 afterwards.
#[test]
fn the_override_does_not_outlive_the_call() {
    let (model, population) = warfarin_bloq(BloqMethod::Drop);
    let _ = fit(
        &model,
        &population,
        &model.default_params,
        &opts(Some(BloqMethod::M3), 1),
    )
    .expect("fit");
    assert_eq!(
        model.bloq_in_force(),
        BloqMethod::Drop,
        "calling thread after the fit"
    );
    let after = fit(&model, &population, &model.default_params, &opts(None, 1)).expect("fit");
    assert_eq!(after.bloq_method, "drop", "a later default fit");

    let outer = crate::types::arm_bloq_override(Some(BloqMethod::M3));
    assert_eq!(model.bloq_in_force(), BloqMethod::M3, "armed");
    {
        let _inner = crate::types::arm_bloq_override(None);
        assert_eq!(
            model.bloq_in_force(),
            BloqMethod::Drop,
            "nested None reads the model"
        );
    }
    assert_eq!(
        model.bloq_in_force(),
        BloqMethod::M3,
        "restored after the nested call"
    );
    drop(outer);
    assert_eq!(
        model.bloq_in_force(),
        BloqMethod::Drop,
        "restored after the outer call"
    );
}

/// **B6 — the post-hoc steps score the fit's recorded method.** `run_covariance` is handed the
/// `drop` model the direct fit ran on with an M3 override, and default options: it must
/// differentiate the M3 objective the estimates minimise, as it does for the stamped fit.
///
/// Mutation: delete the `bloq_method` lines in `fitted_marginal_options` → the step runs under
/// drop and the standard errors move.
#[test]
fn run_covariance_scores_the_method_the_fit_recorded() {
    let (drop_model, population) = warfarin_bloq(BloqMethod::Drop);
    let (m3_model, _) = warfarin_bloq(BloqMethod::M3);
    let direct = fit(
        &drop_model,
        &population,
        &drop_model.default_params,
        &opts(Some(BloqMethod::M3), 1),
    )
    .expect("direct fit");
    let stamped = fit(
        &m3_model,
        &population,
        &m3_model.default_params,
        &opts(None, 1),
    )
    .expect("stamped fit");
    assert_eq!(
        direct.ofv.to_bits(),
        stamped.ofv.to_bits(),
        "same estimates"
    );

    let defaults = FitOptions {
        verbose: false,
        threads: Some(1),
        ..Default::default()
    };
    let se = |fit: &FitResult, model: &CompiledModel| -> Vec<f64> {
        crate::run_covariance(fit, Some(model), Some(&population), &defaults)
            .expect("run_covariance")
            .se_theta
            .expect("standard errors")
    };
    let want = se(&stamped, &m3_model);
    let got = se(&direct, &drop_model);
    // The straddle: the same fit's step under drop lands elsewhere.
    let mut as_drop = direct.clone();
    as_drop.bloq_method = "drop".to_string();
    let under_drop = se(&as_drop, &drop_model);
    assert!(want
        .iter()
        .chain(&got)
        .chain(&under_drop)
        .all(|v| v.is_finite()));
    assert_ne!(
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        under_drop.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the drop and M3 covariance steps agree on this fixture"
    );
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "SE(θ) of the direct fit {got:?} vs the stamped fit {want:?}"
    );
}

/// The resolver alone, both sides of its gate: a recorded label wins over the caller's
/// override; an empty label (a result that recorded none) leaves the caller's.
#[test]
fn fitted_marginal_options_takes_the_recorded_method_when_there_is_one() {
    let mut fit = FitResult {
        bloq_method: "m3".to_string(),
        ..crate::types::test_helpers::minimal_fit_result()
    };
    let caller = FitOptions {
        bloq_method: Some(BloqMethod::Drop),
        ..Default::default()
    };
    let resolved = crate::estimation::fit_inputs::fitted_marginal_options(&fit, &caller);
    assert_eq!(resolved.bloq_method, Some(BloqMethod::M3), "recorded m3");
    fit.bloq_method = String::new();
    let resolved = crate::estimation::fit_inputs::fitted_marginal_options(&fit, &caller);
    assert_eq!(
        resolved.bloq_method,
        Some(BloqMethod::Drop),
        "nothing recorded"
    );
}

/// **B7 — `fit_from_files` is bit-identical to before #1824.** It ignores the file's
/// `[fit_options]` by design, so on an `m3` file a caller with no `bloq_method` opinion gets
/// `drop` (the old `model.bloq_method = opts.bloq_method` stamp of a `Drop` default), and one
/// asking for M3 gets M3.
///
/// Mutation: delete the options-side pin in `fit_from_files` → the `None` call scores M3.
#[test]
fn fit_from_files_keeps_ignoring_the_files_bloq_method() {
    let eval = |bloq: Option<BloqMethod>| {
        crate::fit_from_files(MODEL, Some(DATA), None, Some(opts(bloq, 1))).expect("fit")
    };
    let m3 = run(BloqMethod::M3, None);
    let drop = run(BloqMethod::Drop, None);
    straddle(&m3, &drop);

    let none = eval(None);
    assert_eq!(none.bloq_method, "drop", "no opinion on an m3 file");
    assert_eq!(
        none.ofv.to_bits(),
        drop.ofv.to_bits(),
        "no opinion: drop objective"
    );
    let asked = eval(Some(BloqMethod::M3));
    assert_eq!(asked.bloq_method, "m3", "asked for m3");
    assert_eq!(asked.ofv.to_bits(), m3.ofv.to_bits(), "asked: M3 objective");
}
