//! Integration tests for per-parameter priors / penalized-ML (MAP) estimation
//! (#254).
//!
//! Tier-2 here: the validation rejects return an `Err` immediately, and the
//! objective-split test runs a couple of outer iterations rather than a
//! convergence loop. The behavioural claims that need a converged fit — a weak
//! prior reproducing the MLE, a strong one shrinking toward it, and the prior's
//! curvature reaching the reported SE — are Tier-3 and gated behind
//! `slow-tests` at the bottom.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::Population;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// Warfarin oral 1-cpt model with the `[parameters]` block supplied by the
/// caller, so each test differs by exactly the prior under test.
fn warfarin_with(parameters: &str) -> ferx_core::types::CompiledModel {
    let src = format!(
        "[parameters]\n{parameters}\n\
         [individual_parameters]\n\
         \x20 CL = TVCL * exp(ETA_CL)\n\
         \x20 V  = TVV  * exp(ETA_V)\n\
         \x20 KA = TVKA\n\n\
         [structural_model]\n\
         \x20 pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP_ERR)\n"
    );
    parse_model_string(&src).expect("model must parse")
}

const BASE_PARAMS: &str = "  theta TVCL(0.13, 0.001, 10.0)\n  \
                           theta TVV(8.0, 0.1, 500.0)\n  \
                           theta TVKA(1.0, 0.01, 50.0)\n  \
                           omega ETA_CL ~ 0.09\n  \
                           omega ETA_V  ~ 0.04\n  \
                           sigma PROP_ERR ~ 0.01\n\n";

/// `BASE_PARAMS` with a prior on `TVCL`, at the given central value and RSE.
fn params_with_cl_prior(value: f64, rse_pct: f64) -> String {
    format!(
        "  theta TVCL(0.13, 0.001, 10.0) prior({value}, rse = {rse_pct}%)\n  \
         theta TVV(8.0, 0.1, 500.0)\n  \
         theta TVKA(1.0, 0.01, 50.0)\n  \
         omega ETA_CL ~ 0.09\n  \
         omega ETA_V  ~ 0.04\n  \
         sigma PROP_ERR ~ 0.01\n\n"
    )
}

fn warfarin_population() -> Population {
    read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin dataset must load")
}

/// `fit()` at the model's own declared initial estimates — which is what a
/// `.ferx` file means and what every test here wants.
fn fit_model(
    model: &ferx_core::types::CompiledModel,
    pop: &Population,
    opts: &FitOptions,
) -> Result<ferx_core::types::FitResult, String> {
    fit(model, pop, &model.default_params, opts)
}

/// A couple of outer iterations — Tier-2, no convergence loop.
///
/// The method is set **here**, not in the model file's `[fit_options]`: these
/// tests hand `fit()` a `FitOptions` of their own, and a model file's
/// `[fit_options]` only reaches `fit()` through `ParsedModel::fit_options` (see
/// #1212). Setting it in both places would let a test pass on the default while
/// reading as though it had pinned a method.
fn short_fit_options(method: EstimationMethod) -> FitOptions {
    let mut opts = FitOptions {
        method,
        ..FitOptions::default()
    };
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    opts
}

fn short_focei() -> FitOptions {
    short_fit_options(EstimationMethod::FoceI)
}

// ── The gate: an inapplicable prior stops the fit ────────────────────────────

/// A prior naming a parameter that does not exist must stop the fit, not be
/// dropped. This is the gate `estimation::outer_optimizer::build_prior_set`
/// relies on when it falls back to an empty set.
#[test]
fn fit_refuses_an_unresolvable_prior() {
    let model = warfarin_with(
        "  theta TVCL(0.13, 0.001, 10.0)\n  \
         theta TVV(8.0, 0.1, 500.0)\n  \
         theta TVKA(1.0, 0.01, 50.0) prior(1.0, rse = 20%)\n  \
         omega ETA_CL ~ 0.09\n  \
         omega ETA_V  ~ 0.04\n  \
         sigma PROP_ERR ~ 0.01\n\n",
    );
    // The declaration itself is fine; break the *resolution* by renaming the
    // parameter it points at, which is what a typo produces.
    let mut model = model;
    model.priors[0].name = "NO_SUCH_PARAM".into();

    let err = fit_model(&model, &warfarin_population(), &short_focei())
        .expect_err("a prior on an unknown parameter must stop the fit");
    assert!(err.contains("NO_SUCH_PARAM"), "{err}");
}

/// A prior on a `FIX`ed parameter can never move it, so it is a typo rather
/// than a no-op.
#[test]
fn fit_refuses_a_prior_on_a_fixed_parameter() {
    let model = warfarin_with(
        "  theta TVCL(0.13, 0.001, 10.0)\n  \
         theta TVV(8.0, 0.1, 500.0)\n  \
         theta TVKA(1.0, 0.01, 50.0, FIX) prior(1.0, rse = 20%)\n  \
         omega ETA_CL ~ 0.09\n  \
         omega ETA_V  ~ 0.04\n  \
         sigma PROP_ERR ~ 0.01\n\n",
    );
    let err = fit_model(&model, &warfarin_population(), &short_focei())
        .expect_err("a prior on a FIXed parameter must stop the fit");
    assert!(err.contains("FIX"), "{err}");
}

/// A method that does not apply priors must refuse, not quietly return the
/// unpenalized MLE. An unapplied prior is invisible in the output — the fit
/// converges, the estimates look reasonable, and nothing says the prior was
/// dropped.
#[test]
fn fit_refuses_a_prior_the_final_method_cannot_apply() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let pop = warfarin_population();

    let err = fit_model(&model, &pop, &short_fit_options(EstimationMethod::Saem))
        .expect_err("saem does not apply priors and must say so");
    // The message must name the stage that dropped it, or the user has to guess
    // which of a chain's methods is the problem.
    assert!(err.to_lowercase().contains("saem"), "{err}");
    assert!(err.contains("prior"), "{err}");

    // The straddle: the *identical model* under a FOCE-family method is
    // accepted, so the gate rejects the unsupported method rather than every
    // prior. Same model value, one differing input — the method.
    assert!(fit_model(&model, &pop, &short_focei()).is_ok());
}

/// The penalty has to reach every optimizer that claims to apply priors, not
/// only the default one.
///
/// `foce`/`focei` route through NLopt by default, so the built-in BFGS outer
/// loop and the Gauss-Newton optimizer each assemble the penalized objective in
/// their *own* code — `run_bfgs_outer`'s `f_only`/`fdfg`/`clean_ofv_at`, and
/// `gauss_newton`'s BHHH assembly. A prior dropped in either is invisible: the
/// fit converges, the report recomputes `ofv_prior` at the final estimate and
/// shows a plausible non-zero penalty, and only the estimate is wrong.
///
/// So the assertion is the *identity* `ofv == ofv_data + ofv_prior` with
/// `ofv_prior > 0` — `ofv_prior > 0` alone passes on an engine that never saw
/// the penalty, since the report computes it independently.
#[test]
fn every_prior_applying_optimizer_carries_the_penalty() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let pop = warfarin_population();

    let cases: [(&str, FitOptions); 3] = [
        ("builtin bfgs", {
            let mut o = short_focei();
            o.optimizer = ferx_core::types::Optimizer::Bfgs;
            o
        }),
        ("gn", short_fit_options(EstimationMethod::FoceGn)),
        (
            "gn_hybrid",
            short_fit_options(EstimationMethod::FoceGnHybrid),
        ),
    ];

    for (label, opts) in cases {
        let r = fit_model(&model, &pop, &opts)
            .unwrap_or_else(|e| panic!("{label} must accept a prior: {e}"));
        assert!(
            r.ofv_prior > 0.0,
            "{label}: the prior half must be reported"
        );
        assert!(
            (r.ofv - (r.ofv_data + r.ofv_prior)).abs() < 1e-9,
            "{label}: ofv {} != ofv_data {} + ofv_prior {}",
            r.ofv,
            r.ofv_data,
            r.ofv_prior
        );
        assert!(r.ofv.is_finite() && r.ofv_data.is_finite(), "{label}");
        assert_eq!(r.prior_summary.len(), 1, "{label}");
        assert_eq!(r.prior_summary[0].name, "TVCL", "{label}");
    }

    // The control: the unpriored twin under the same three engines reports a
    // zero prior half, so the identity above is not satisfied by an engine that
    // reports `ofv_data == ofv` regardless.
    let unpriored = warfarin_with(BASE_PARAMS);
    let mut bfgs = short_focei();
    bfgs.optimizer = ferx_core::types::Optimizer::Bfgs;
    for (label, opts) in [
        ("builtin bfgs", bfgs),
        ("gn", short_fit_options(EstimationMethod::FoceGn)),
    ] {
        let r = fit_model(&unpriored, &pop, &opts)
            .unwrap_or_else(|e| panic!("{label} unpriored must fit: {e}"));
        assert_eq!(r.ofv_prior, 0.0, "{label}");
        assert_eq!(r.ofv_data, r.ofv, "{label}");
        assert!(r.prior_summary.is_empty(), "{label}");
    }
}

/// A chain whose *final* stage applies priors is fine even when an earlier one
/// does not — the last stage is the one whose estimates are reported.
#[test]
fn a_chain_ending_in_focei_accepts_a_prior() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let mut opts = short_focei();
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
    opts.saem_n_exploration = 1;
    opts.saem_n_convergence = 1;

    let r = fit_model(&model, &warfarin_population(), &opts);
    assert!(
        r.is_ok(),
        "a chain ending in focei must be accepted: {:?}",
        r.err()
    );

    // The straddle: reverse the chain so the *last* stage is the one that
    // cannot apply priors, and it must be refused. Without this the test would
    // pass on a gate that looked at any stage, or at none.
    let mut reversed = short_focei();
    reversed.methods = vec![EstimationMethod::FoceI, EstimationMethod::Saem];
    reversed.saem_n_exploration = 1;
    reversed.saem_n_convergence = 1;
    assert!(fit_model(&model, &warfarin_population(), &reversed).is_err());
}

// ── The objective split ──────────────────────────────────────────────────────

/// The reported OFV is the penalized total, the two halves are published
/// separately, and AIC/BIC come from the data half.
///
/// Every assertion here would pass vacuously on an implementation that ignored
/// the prior entirely *except* `ofv_prior > 0`, which is why that one is first:
/// it is the check that the penalty actually reached the objective.
#[test]
fn a_priored_fit_splits_the_objective_and_keeps_the_ics_on_the_data_half() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let r = fit_model(&model, &warfarin_population(), &short_focei())
        .expect("short priored fit should run");

    assert!(
        r.ofv_prior > 0.0,
        "the prior must contribute to the objective; got {}",
        r.ofv_prior
    );
    assert!((r.ofv - (r.ofv_data + r.ofv_prior)).abs() < 1e-9);
    // AIC/BIC are computed from the data half: a penalized objective is not a
    // log-likelihood, and an IC built from one would reward a tighter prior.
    assert!((r.aic - (r.ofv_data + 2.0 * r.n_parameters as f64)).abs() < 1e-9);
    assert!(r.aic < r.ofv + 2.0 * r.n_parameters as f64);

    // The per-parameter report reconciles with the total, or the split shown to
    // the user does not add up to the number next to it.
    assert_eq!(r.prior_summary.len(), 1);
    let row = &r.prior_summary[0];
    assert_eq!(row.name, "TVCL");
    assert_eq!(row.prior_value, 0.15);
    assert_eq!(row.family, "lognormal");
    let total: f64 = r.prior_summary.iter().map(|p| p.penalty).sum();
    assert!((total - r.ofv_prior).abs() < 1e-9);
    // The reported estimate is the θ that was estimated, on its declared scale.
    let i = r.theta_names.iter().position(|n| n == "TVCL").unwrap();
    assert!((row.estimate - r.theta[i]).abs() < 1e-9);
}

/// The control: the identical model with no prior reports a zero prior half and
/// an `ofv` that still equals `ofv_data`, so nothing about an unpriored fit
/// moved.
#[test]
fn an_unpriored_fit_reports_a_zero_prior_half() {
    let model = warfarin_with(BASE_PARAMS);
    let r = fit_model(&model, &warfarin_population(), &short_focei())
        .expect("short unpriored fit should run");
    assert_eq!(r.ofv_prior, 0.0);
    assert_eq!(r.ofv, r.ofv_data);
    assert!(r.prior_summary.is_empty());
}

/// Two priors centred at different places must give different objectives at the
/// same starting point — a penalty wired to the wrong coordinate, or dropped,
/// would give the same one twice.
#[test]
fn moving_the_prior_moves_the_objective() {
    let pop = warfarin_population();
    let near = fit_model(
        &warfarin_with(&params_with_cl_prior(0.13, 25.0)),
        &pop,
        &short_focei(),
    )
    .expect("near-prior fit should run");
    let far = fit_model(
        &warfarin_with(&params_with_cl_prior(0.50, 25.0)),
        &pop,
        &short_focei(),
    )
    .expect("far-prior fit should run");

    // The start is TVCL = 0.13, so the near prior sits on it and the far one is
    // ~4× away: the penalties must differ by a lot, and in the right direction.
    assert!(
        far.ofv_prior > near.ofv_prior + 1.0,
        "a prior 4x away from the start must penalize more: near {} vs far {}",
        near.ofv_prior,
        far.ofv_prior
    );
}

/// The prior must reach the **optimizer**, not only the report.
///
/// This exists because every other Tier-2 test here can pass without it:
/// `ofv_prior` and `prior_summary` are recomputed in `fit()` at the final
/// estimate, so they are non-zero whether or not the objective the optimizer
/// minimised ever saw the penalty. Verified by mutation — deleting
/// `priors.penalty(..)` from `outer_optimizer`'s objective leaves every other
/// test in this file green and reddens only this one (and the Tier-3 straddle,
/// which does not run on a PR).
///
/// Two tight priors on opposite sides of the starting value, a couple of outer
/// iterations each: the estimates must separate, and in the right order.
#[test]
fn a_prior_moves_the_estimates_and_not_only_the_report() {
    let pop = warfarin_population();
    // Start is TVCL = 0.13. RSE 1% makes each prior far stronger than the data.
    let low = fit_model(
        &warfarin_with(&params_with_cl_prior(0.065, 1.0)),
        &pop,
        &short_focei(),
    )
    .expect("low-prior fit should run");
    let high = fit_model(
        &warfarin_with(&params_with_cl_prior(0.26, 1.0)),
        &pop,
        &short_focei(),
    )
    .expect("high-prior fit should run");

    let i = low.theta_names.iter().position(|n| n == "TVCL").unwrap();
    let (cl_low, cl_high) = (low.theta[i], high.theta[i]);
    assert!(
        cl_low < cl_high,
        "the lower prior must pull TVCL below the higher one: {cl_low} vs {cl_high}"
    );
    // Not merely ordered — separated. Two runs of the same unpriored fit would
    // land on the same number, so any separation at all is the prior's doing;
    // the margin keeps the assertion above optimizer noise.
    assert!(
        (cl_high - cl_low) / cl_low > 0.1,
        "the two priors must pull the estimate materially apart: {cl_low} vs {cl_high}"
    );
}

// ── `[priors] from_fit` — the model-updating import (#254 phase 2) ──────────

/// A minimal `{model}-fit.yaml` for the warfarin parameter names, with `TVCL`
/// at `cl` and a 1% relative standard error on every parameter.
///
/// Hand-written rather than produced by a first `fit()`: the shape is what
/// `write_estimates_yaml` emits (pinned against the writer by the Tier-1
/// round-trip), and writing it here keeps this test at Tier 2 — no convergence
/// loop, and no dependence on a covariance step that a two-iteration fit does
/// not run.
fn source_fit_yaml(cl: f64) -> String {
    format!(
        "model:\n  converged: true\n\n\
         theta:\n\
         \x20 TVCL:\n    estimate: {cl:.6}\n    se: {:.6}\n\
         \x20 TVV:\n    estimate: 8.000000\n    se: 0.080000\n\
         \x20 TVKA:\n    estimate: 1.000000\n    se: 0.010000\n\n\
         omega:\n\
         \x20 ETA_CL:\n    variance: 0.090000\n    cv_pct: 30.00\n    se: 0.000900\n\
         \x20 ETA_V:\n    variance: 0.040000\n    cv_pct: 20.00\n    se: 0.000400\n\n\
         sigma:  # error model: proportional\n\
         \x20 PROP_ERR:\n    estimate: 0.100000\n    variance: 0.010000\n    type: proportional\n    se: 0.001000\n",
        cl * 0.01
    )
}

/// The warfarin model source with a `[priors] from_fit` block pointing at
/// `path`, and no inline prior at all — so everything penalizing this fit came
/// from the import.
fn warfarin_from_fit_src(path: &std::path::Path) -> String {
    format!(
        "[parameters]\n{BASE_PARAMS}\
         [priors]\n  from_fit = \"{}\"\n\n\
         [individual_parameters]\n\
         \x20 CL = TVCL * exp(ETA_CL)\n\
         \x20 V  = TVV  * exp(ETA_V)\n\
         \x20 KA = TVKA\n\n\
         [structural_model]\n\
         \x20 pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP_ERR)\n",
        path.display()
    )
}

/// …parsed. The import is expanded **by the parser**, so a `from_fit` that
/// cannot be honoured fails here rather than at `fit()`.
fn warfarin_from_fit(path: &std::path::Path) -> ferx_core::types::CompiledModel {
    parse_model_string(&warfarin_from_fit_src(path)).expect("model must parse")
}

/// Write `source_fit_yaml(cl)` into `dir` and return its path.
fn write_source_fit(dir: &std::path::Path, cl: f64) -> std::path::PathBuf {
    let path = dir.join("parent-fit.yaml");
    std::fs::write(&path, source_fit_yaml(cl)).unwrap();
    path
}

/// An imported prior must reach the **optimizer**, not only the report — the
/// `from_fit` counterpart of `a_prior_moves_the_estimates_and_not_only_the_report`
/// above, and for the same reason: `ofv_prior` and `prior_summary` are
/// recomputed in `fit()` at the final estimate, so they read non-zero whether or
/// not the objective ever saw the penalty.
///
/// Two source fits differing only in `TVCL`, each with a 1% RSE that makes the
/// prior far stronger than the data. The estimates must separate, and in the
/// right order.
#[test]
fn an_imported_prior_moves_the_estimates() {
    let pop = warfarin_population();
    let dir = tempfile::tempdir().unwrap();
    let low_path = dir.path().join("low-fit.yaml");
    std::fs::write(&low_path, source_fit_yaml(0.065)).unwrap();
    let high_path = dir.path().join("high-fit.yaml");
    std::fs::write(&high_path, source_fit_yaml(0.26)).unwrap();

    let low = fit_model(&warfarin_from_fit(&low_path), &pop, &short_focei())
        .expect("low-prior fit should run");
    let high = fit_model(&warfarin_from_fit(&high_path), &pop, &short_focei())
        .expect("high-prior fit should run");

    // Every priorable parameter was imported: 3 θ, 2 Ω, 1 σ.
    assert_eq!(low.prior_summary.len(), 6, "{:?}", low.prior_summary);
    assert!(low.ofv_prior > 0.0);
    // The penalty is in the objective, not only the report.
    assert!((low.ofv - (low.ofv_data + low.ofv_prior)).abs() < 1e-8);

    let i = low.theta_names.iter().position(|n| n == "TVCL").unwrap();
    let (cl_low, cl_high) = (low.theta[i], high.theta[i]);
    assert!(
        cl_low < cl_high,
        "the lower imported prior must pull TVCL below the higher one: {cl_low} vs {cl_high}"
    );
    assert!(
        (cl_high - cl_low) / cl_low > 0.1,
        "the two imported priors must pull the estimate materially apart: {cl_low} vs {cl_high}"
    );
}

/// The prior centre lands on the source estimate, on this model's declared
/// scale — a variance for `omega ETA_CL ~ 0.09`, and `σ²` for
/// `sigma PROP_ERR ~ 0.01` (the source reports σ as an SD of 0.1).
///
/// Reads the whole conversion table off one fit, so a scale flipped on any one
/// family shows up here rather than as a slightly-off estimate.
#[test]
fn an_imported_prior_reports_the_source_estimate_on_the_declared_scale() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source_fit(dir.path(), 0.13);
    let fit = fit_model(
        &warfarin_from_fit(&path),
        &warfarin_population(),
        &short_focei(),
    )
    .expect("fit should run");

    let centre = |name: &str| {
        fit.prior_summary
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no prior on {name}: {:?}", fit.prior_summary))
            .prior_value
    };
    assert!((centre("TVCL") - 0.13).abs() < 1e-9);
    assert!((centre("TVV") - 8.0).abs() < 1e-9);
    assert!(
        (centre("ETA_CL") - 0.09).abs() < 1e-9,
        "{}",
        centre("ETA_CL")
    );
    // σ: the source's SD of 0.1, squared onto the declared variance scale.
    assert!(
        (centre("PROP_ERR") - 0.01).abs() < 1e-9,
        "{}",
        centre("PROP_ERR")
    );
}

/// A relative `from_fit` resolves against the model file, end to end through
/// `parse_model_file` — the path a user actually takes, where the model and the
/// fit it updates from sit together in a run directory.
#[test]
fn a_relative_from_fit_works_end_to_end_from_a_model_file() {
    let dir = tempfile::tempdir().unwrap();
    write_source_fit(dir.path(), 0.13);
    let model_path = dir.path().join("update.ferx");
    // The block names the file by its bare name, with no directory at all.
    let src = format!(
        "[parameters]\n{BASE_PARAMS}\
         [priors]\n  from_fit = parent-fit.yaml\n\n\
         [individual_parameters]\n\
         \x20 CL = TVCL * exp(ETA_CL)\n\
         \x20 V  = TVV  * exp(ETA_V)\n\
         \x20 KA = TVKA\n\n\
         [structural_model]\n\
         \x20 pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP_ERR)\n"
    );
    std::fs::write(&model_path, src).unwrap();

    let model = ferx_core::parser::model_parser::parse_model_file(&model_path)
        .expect("model file must parse");
    let fit = fit_model(&model, &warfarin_population(), &short_focei())
        .expect("a model-file-relative from_fit must resolve");
    assert_eq!(fit.prior_summary.len(), 6);

    // The straddle: the same model text parsed from a *string* has no directory
    // to resolve against, so the bare name does not exist and the parse is
    // refused. Without it this test passes on an implementation that resolves
    // against the working directory and happens to find nothing either way.
    let err = parse_model_string(&std::fs::read_to_string(&model_path).unwrap())
        .map(|_| ())
        .expect_err("a bare name has nothing to resolve against from a string");
    assert!(err.contains("from_fit"), "{err}");
}

/// A `from_fit` that lands nothing must stop the run, for the same reason an
/// unresolvable typed prior does: the alternative is an unpenalized fit that is
/// indistinguishable from a penalized one.
///
/// It stops it at **parse** time, because that is where the source fit is read
/// (once) — so `ferx check` refuses the model too, and no post-fit entry point
/// can be handed a model whose priors quietly failed to materialise.
#[test]
fn a_from_fit_that_lands_nothing_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unrelated-fit.yaml");
    std::fs::write(
        &path,
        "theta:\n  SOMETHING_ELSE:\n    estimate: 1.000000\n    se: 0.100000\n",
    )
    .unwrap();
    let err = parse_model_string(&warfarin_from_fit_src(&path))
        .map(|_| ())
        .expect_err("an import that lands nothing must be refused");
    assert!(err.contains("no prior could be imported"), "{err}");
}

/// A missing `from_fit` file is refused at the same gate, naming the path.
#[test]
fn a_missing_from_fit_file_is_refused() {
    let err = parse_model_string(&warfarin_from_fit_src(std::path::Path::new(
        "/no/such/parent-fit.yaml",
    )))
    .map(|_| ())
    .expect_err("a missing from_fit file must be refused");
    assert!(err.contains("/no/such/parent-fit.yaml"), "{err}");
}

/// A `CompiledModel` assembled by hand with an unexpanded `from_fit` is refused
/// by `fit()` rather than fit unpenalized.
///
/// The parser is the only thing that reads the source fit now, so a model that
/// never went through it has no priors — and `build_prior_set` turns a build
/// error into an *empty* set, which is exactly how an unpenalized fit would have
/// passed for a penalized one. This is the guard that stops that, checked at the
/// public boundary rather than only in the unit tests.
#[test]
fn fit_refuses_a_model_whose_from_fit_was_never_expanded() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_source_fit(dir.path(), 0.13);

    // Parse a model with no `[priors]` block, then bolt the path on afterwards —
    // which is what a caller constructing a `CompiledModel` field by field does.
    let mut model = warfarin_with(BASE_PARAMS);
    model.prior_from_fit = Some(path.to_string_lossy().into_owned());
    let err = fit_model(&model, &warfarin_population(), &short_focei())
        .expect_err("an unexpanded from_fit must not fit unpenalized");
    assert!(err.contains("has not been expanded"), "{err}");

    // The straddle: expanding it makes the very same model fit, with the priors
    // in force — so the guard rejects the unexpanded state, not the feature.
    ferx_core::parser::model_parser::expand_prior_from_fit(&mut model)
        .expect("the file is right there");
    let fit = fit_model(&model, &warfarin_population(), &short_focei()).expect("fit should run");
    assert_eq!(fit.prior_summary.len(), 6);
}

/// `examples/warfarin_update.ferx` resolves against the file the documented
/// commands actually produce.
///
/// The example is a two-command flow across two different mechanisms, and they
/// disagreed on the first attempt: the CLI writes `{model}-fit.yaml` into the
/// **working directory** (`main.rs`, from the model path's file stem), while
/// `from_fit` resolves against the **model file's** directory. Run from the repo
/// root, the parent wrote `./warfarin-fit.yaml` and the update looked for
/// `./examples/warfarin-fit.yaml`.
///
/// Two things are asserted, because either alone permits the mismatch:
///
/// - the example names its parent by a **bare filename**, so "the file sits next
///   to the model" is what the model file claims;
/// - the parent example is a **sibling** of the update example, so the directory
///   the documented `cd` names is the one the CLI will write into.
///
/// Together they are the invariant that makes `cd examples` the right
/// instruction, and they are what a future edit to either file has to keep.
#[test]
fn the_update_example_names_a_file_the_documented_commands_produce() {
    let update = Path::new("examples/warfarin_update.ferx");
    let src = std::fs::read_to_string(update).expect("the example must exist");

    let from_fit = src
        .lines()
        .find_map(|l| l.trim().strip_prefix("from_fit"))
        .and_then(|r| r.split_once('='))
        .map(|(_, v)| v.trim().trim_matches(|c| c == '"' || c == '\'').to_string())
        .expect("the example must declare `from_fit`");

    assert_eq!(
        Path::new(&from_fit).parent(),
        Some(Path::new("")),
        "`from_fit = {from_fit}` must be a bare filename: it resolves against the \
         model file's directory, while the CLI writes into the working directory, \
         so any directory component makes the two disagree"
    );

    // The CLI names the output from the *parent* model's file stem.
    let parent = update.with_file_name("warfarin.ferx");
    assert!(
        parent.exists(),
        "the parent example must sit beside the update example, or the documented \
         `cd examples` is not where the CLI would write the fit"
    );
    assert_eq!(
        from_fit,
        format!("{}-fit.yaml", parent.file_stem().unwrap().to_string_lossy()),
        "the example reads a different file from the one `ferx {}` writes",
        parent.display()
    );

    // …and the resolution really lands there. Stage the two files the way the
    // documented commands leave them and parse the update for real, so this is
    // not only a string comparison.
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(update, dir.path().join("warfarin_update.ferx")).unwrap();
    std::fs::write(dir.path().join(&from_fit), source_fit_yaml(0.13)).unwrap();
    let model =
        ferx_core::parser::model_parser::parse_model_file(&dir.path().join("warfarin_update.ferx"))
            .expect("the example must parse with its parent fit beside it");
    assert!(
        !model.priors.is_empty(),
        "the example's from_fit resolved but imported nothing"
    );
}

/// Parameters the import skipped are reported in `warnings`.
///
/// A skip is only visible as an absence otherwise: the fit converges, the
/// remaining priors look fine, and nothing says the one the user cared about was
/// dropped.
#[test]
fn skipped_imports_are_reported_in_the_warnings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial-fit.yaml");
    // TVKA carries no standard error, so it cannot supply a prior spread.
    std::fs::write(
        &path,
        source_fit_yaml(0.13).replace(
            " TVKA:\n    estimate: 1.000000\n    se: 0.010000",
            " TVKA:\n    estimate: 1.000000\n    se: ~",
        ),
    )
    .unwrap();
    let fit = fit_model(
        &warfarin_from_fit(&path),
        &warfarin_population(),
        &short_focei(),
    )
    .expect("the remaining five imports must still land");

    assert_eq!(fit.prior_summary.len(), 5, "{:?}", fit.prior_summary);
    assert!(
        fit.warnings
            .iter()
            .any(|w| w.contains("from_fit") && w.contains("TVKA")),
        "{:?}",
        fit.warnings
    );
}

// ── Tier 3: behaviour that needs a converged fit ─────────────────────────────

/// The two halves of the issue's acceptance criteria, asserted **together** as
/// a straddle.
///
/// Each half alone is satisfied by an implementation that ignores the prior: a
/// weak prior reproducing the MLE is what "no prior at all" also does, and a
/// shrunk estimate could be chance. They only test anything as a pair, with a
/// stated minimum shift on the strong side.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_weak_prior_reproduces_the_mle_and_a_strong_one_shrinks_toward_it() {
    let pop = warfarin_population();
    let mut opts = FitOptions::default();
    opts.run_covariance_step = false;

    let mle = fit_model(&warfarin_with(BASE_PARAMS), &pop, &opts).expect("unpriored fit");
    let i = mle.theta_names.iter().position(|n| n == "TVCL").unwrap();
    let cl_mle = mle.theta[i];

    // A prior centred far from the MLE, so "shrinks toward the prior" has a
    // direction to be right about.
    let prior_at = cl_mle * 2.0;

    // RSE 10000% → a log-scale SD of ~4.8, i.e. a 95% interval spanning eight
    // orders of magnitude. Flat enough to be no prior at all.
    let weak = fit_model(
        &warfarin_with(&params_with_cl_prior(prior_at, 10000.0)),
        &pop,
        &opts,
    )
    .expect("weak-prior fit");
    let cl_weak = weak.theta[i];

    // RSE 1% → a log-scale SD of 0.01, which pins TVCL to the prior.
    let strong = fit_model(
        &warfarin_with(&params_with_cl_prior(prior_at, 1.0)),
        &pop,
        &opts,
    )
    .expect("strong-prior fit");
    let cl_strong = strong.theta[i];

    assert!(
        ((cl_weak - cl_mle) / cl_mle).abs() < 0.02,
        "a near-flat prior must reproduce the MLE: {cl_mle} vs {cl_weak}"
    );
    assert!(
        ((cl_strong - prior_at) / prior_at).abs() < 0.05,
        "a tight prior must pin the estimate to it: prior {prior_at}, got {cl_strong}"
    );
    // And the straddle itself, so the pair cannot silently become a tautology:
    // the two fits must actually disagree.
    assert!(
        (cl_strong - cl_weak).abs() > 0.3 * cl_mle,
        "weak and strong must land in different places: {cl_weak} vs {cl_strong}"
    );

    // Deliberately *not* asserted: `weak.ofv_prior < strong.ofv_prior`. It looks
    // like it should hold and does not — both are ≈ 0, for opposite reasons. The
    // weak prior is too flat to penalize anything, and the strong one pulls the
    // estimate onto its own centre, where the penalty it charges is zero. The
    // prior half of the OFV measures *disagreement* between data and prior, not
    // prior strength, so a well-obeyed tight prior costs nothing.
}

/// The prior's curvature must reach the reported standard error.
///
/// With a tight prior on a parameter, the penalized Hessian is dominated by the
/// prior's `2/s²`, so the SE collapses toward the prior SD. Without the
/// covariance-step wiring the SE would keep the (much larger) data-only value —
/// the specific failure that makes MAP SEs meaningless, and the one the
/// covariance step's own `add_hessian` call is the only thing preventing.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_tight_prior_tightens_the_reported_standard_error() {
    let pop = warfarin_population();
    let mut opts = FitOptions::default();
    opts.run_covariance_step = true;

    let loose = fit_model(
        &warfarin_with(&params_with_cl_prior(0.13, 200.0)),
        &pop,
        &opts,
    )
    .expect("loose-prior fit");
    let tight = fit_model(
        &warfarin_with(&params_with_cl_prior(0.13, 2.0)),
        &pop,
        &opts,
    )
    .expect("tight-prior fit");

    let i = loose.theta_names.iter().position(|n| n == "TVCL").unwrap();
    let se_loose = loose.se_theta.as_ref().expect("loose SEs")[i];
    let se_tight = tight.se_theta.as_ref().expect("tight SEs")[i];

    // The log-scale prior SD at RSE 2% is sqrt(ln(1+0.02²)) ≈ 0.02, so on the
    // natural scale the prior alone would give SE ≈ 0.02·TVCL. The penalized SE
    // must be at or below that — the data can only add information.
    let cl = tight.theta[i];
    assert!(
        se_tight < 0.021 * cl,
        "a 2% prior must pin the SE: got {se_tight} at TVCL = {cl}"
    );
    // And it must be a *change*, not just a small number. Measured on this
    // fixture: SE 7.959e-3 at RSE 200% falls to 2.486e-3 at RSE 2%, a ratio of
    // 0.312. The bound is set at 0.5 — loose enough to survive optimizer noise
    // in the loose arm, tight enough that dropping the covariance step's
    // `add_hessian` (which leaves both arms at the data-only SE, ratio 1.0)
    // fails it.
    assert!(
        se_tight < 0.5 * se_loose,
        "tightening the prior must tighten the SE: {se_loose} -> {se_tight}"
    );
}

// ── Review findings on 762f6aa (#1341) ──────────────────────────────────────

/// **Finding 1.** A trailing *evaluation-only* stage is not an estimator, so it
/// decides nothing about whether the prior was applied.
///
/// Both directions, because each alone passes under a different wrong predicate:
/// `chain.last()` accepts the first and rejects the second, while "any stage
/// applies priors" accepts both.
#[test]
fn the_prior_method_gate_looks_past_evaluation_only_stages() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let pop = warfarin_population();

    // [saem, laplace] with agq_eval_only: Laplace is a supported method, but here
    // it only evaluates SAEM's estimates. Nothing applied the prior → reject, and
    // the message must name SAEM rather than the evaluator.
    let mut eval_tail = short_focei();
    eval_tail.methods = vec![EstimationMethod::Saem, EstimationMethod::Laplace];
    eval_tail.agq_eval_only = true;
    eval_tail.saem_n_exploration = 1;
    eval_tail.saem_n_convergence = 1;
    let err = fit_model(&model, &pop, &eval_tail)
        .expect_err("an eval-only tail must not launder an unsupported estimator");
    assert!(err.to_lowercase().contains("saem"), "{err}");

    // [focei, imp] with imp_eval_only: IMP is unsupported as an *estimator*, but
    // here it only scores FOCEI's estimates, which did apply the prior → accept.
    let mut eval_imp = short_focei();
    eval_imp.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    eval_imp.imp_eval_only = true;
    eval_imp.imp_samples = 20;
    let r = fit_model(&model, &pop, &eval_imp);
    assert!(
        r.is_ok(),
        "an eval-only IMP tail must not reject a prior FOCEI applied: {:?}",
        r.err()
    );
    let r = r.unwrap();
    assert!(
        r.ofv_prior > 0.0,
        "the prior must still be in the objective"
    );
}

/// **Finding 2.** A priored fit must survive a `.fitrx` round trip.
///
/// Storing only the penalized `ofv` and reconstructing `ofv_data` from it on load
/// relabels the penalized objective as the data likelihood, drops every prior
/// row, and breaks the `aic == ofv_data + 2k` invariant `aic` was computed under.
#[test]
fn a_priored_fit_round_trips_through_fitrx() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let saved =
        fit_model(&model, &warfarin_population(), &short_focei()).expect("priored fit should run");
    assert!(saved.ofv_prior > 0.0, "fixture must actually carry a prior");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("priored.fitrx");
    ferx_core::io::fitrx::save_fit(
        &saved,
        &warfarin_population(),
        "",
        &path,
        ferx_core::io::fitrx::SaveFitOptions { include_data: None },
    )
    .expect("write");
    let loaded = ferx_core::io::fitrx::load_fit(&path).expect("read").fit;

    // The halves come back as written, not reconstructed from the total.
    assert_eq!(loaded.ofv, saved.ofv);
    assert_eq!(loaded.ofv_data, saved.ofv_data);
    assert_eq!(loaded.ofv_prior, saved.ofv_prior);
    assert_ne!(
        loaded.ofv_data, loaded.ofv,
        "ofv_data must not be the penalized total"
    );
    // The invariant `aic` was computed under survives the trip.
    assert!((loaded.aic - (loaded.ofv_data + 2.0 * loaded.n_parameters as f64)).abs() < 1e-9);
    // And the per-parameter rows survive.
    assert_eq!(loaded.prior_summary.len(), saved.prior_summary.len());
    assert_eq!(loaded.prior_summary[0].name, "TVCL");
    assert_eq!(
        loaded.prior_summary[0].shift_in_prior_sds,
        saved.prior_summary[0].shift_in_prior_sds
    );
}

/// The control for the round trip: an **unpriored** fit still loads, and its
/// `ofv_data` falls back to `ofv` (which for it is the data likelihood).
#[test]
fn an_unpriored_fit_still_round_trips_through_fitrx() {
    let model = warfarin_with(BASE_PARAMS);
    let saved = fit_model(&model, &warfarin_population(), &short_focei()).expect("unpriored fit");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("plain.fitrx");
    ferx_core::io::fitrx::save_fit(
        &saved,
        &warfarin_population(),
        "",
        &path,
        ferx_core::io::fitrx::SaveFitOptions { include_data: None },
    )
    .expect("write");
    let loaded = ferx_core::io::fitrx::load_fit(&path).expect("read").fit;

    assert_eq!(loaded.ofv, saved.ofv);
    assert_eq!(loaded.ofv_data, saved.ofv);
    assert_eq!(loaded.ofv_prior, 0.0);
    assert!(loaded.prior_summary.is_empty());
}

/// **Finding 5.** Neither covariance estimator built on `S` can carry a prior,
/// so `s` and `rsr` are both refused rather than reported as MAP standard
/// errors.
///
/// `s` is the obvious half: `S⁻¹` is the unpenalized information outright.
/// `rsr` is the half that was missed when this gate first landed — `R⁻¹ S R⁻¹`
/// *looks* safe because both `R⁻¹` factors carry the prior's curvature, but the
/// `S` between them does not, so the sandwich is neither the penalized
/// estimator nor the unpenalized one and under-states the SE on every priored
/// coordinate. The old `s` diagnostic recommended `rsr` by name, which is why
/// the suggestion is asserted here too.
#[test]
fn a_prior_with_an_s_based_covariance_is_refused() {
    let model = warfarin_with(&params_with_cl_prior(0.15, 25.0));
    let pop = warfarin_population();

    let with_cov = |m: ferx_core::types::CovarianceMethod| {
        let mut o = short_focei();
        o.run_covariance_step = true;
        o.covariance_method = m;
        o
    };

    for (label, method) in [
        ("s", ferx_core::types::CovarianceMethod::CrossProduct),
        ("rsr", ferx_core::types::CovarianceMethod::Sandwich),
    ] {
        let err = fit_model(&model, &pop, &with_cov(method))
            .expect_err("an S-based covariance method cannot represent a prior");
        assert!(
            err.contains(&format!("covariance_method = {label}")),
            "the message must name the rejected method: {err}"
        );
        // The suggestion must not send the user to the other broken estimator.
        assert!(
            !err.contains("`rsr`"),
            "`rsr` is rejected too and must not be recommended: {err}"
        );

        // And an unpriored fit may still use it, so the rejection is about the
        // prior and not about the method.
        let plain = warfarin_with(BASE_PARAMS);
        assert!(
            fit_model(&plain, &pop, &with_cov(method)).is_ok(),
            "`{label}` without a prior must be untouched"
        );
    }

    // The straddle: the same model with the default `r` is accepted, so the gate
    // rejects the estimator rather than every priored covariance step.
    assert!(fit_model(
        &model,
        &pop,
        &with_cov(ferx_core::types::CovarianceMethod::Hessian)
    )
    .is_ok());
}

/// **Finding 3.** SIR must resample against the *penalized* objective.
///
/// SIR approximates the posterior the fit targeted. Under a prior that target is
/// `L(data)·p(θ)`, so the importance weights have to score both halves. If they
/// score the data alone while the proposal is centred on the MAP estimate and
/// shaped by the penalized curvature, the resampling actively pulls the interval
/// back toward the MLE — a prior then shows up in the point estimate but not in
/// the reported CI, which is worse than not supporting SIR at all.
///
/// # Why RSE = 10%, specifically
///
/// This test is only meaningful in the regime where the *weights* decide
/// something, and that regime had to be measured rather than assumed — the first
/// version of it used RSE = 2% and was **vacuous**: the penalized proposal is
/// then so tight that every draw lands within a hair of the MAP, both weightings
/// agree, and the assertion passed with the prior deleted from the sampler.
/// Measured on this fixture (MLE 0.135276, prior 0.202914, seed 20254):
///
/// | RSE | CI with the prior in the weights | CI without |
/// |-----|----------------------------------|------------|
/// | 2%  | (0.1931, 0.2096) | (0.1863, 0.2191) — both at the prior, no signal |
/// | **10%** | **(0.1384, 0.1927)** | **(0.1251, 0.1572)** — separated |
/// | 20% | (0.1245, 0.1595) | (0.1215, 0.1595) — prior too weak to matter |
/// | 40% | (0.1210, 0.1555) | (0.1197, 0.1555) — likewise |
///
/// At 10% the point estimate is identical either way (0.156190 — the *fit* is
/// unaffected, only the resampling is), so anything that moves is the weights.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn sir_intervals_respect_the_prior() {
    let pop = warfarin_population();
    let mut opts = FitOptions::default();
    opts.run_covariance_step = true;
    opts.sir = true;
    opts.sir_samples = 600;
    opts.sir_resamples = 300;
    opts.sir_seed = Some(20254);

    // The unpriored MLE, so the prior can be placed somewhere the data disagrees
    // with it and "pulled back toward the MLE" has a direction.
    let mut mle_opts = opts.clone();
    mle_opts.sir = false;
    let mle = fit_model(&warfarin_with(BASE_PARAMS), &pop, &mle_opts).expect("mle fit");
    let i = mle.theta_names.iter().position(|n| n == "TVCL").unwrap();
    let cl_mle = mle.theta[i];
    let prior_at = cl_mle * 1.5;

    let r = fit_model(
        &warfarin_with(&params_with_cl_prior(prior_at, 10.0)),
        &pop,
        &opts,
    )
    .expect("priored SIR fit");
    assert!(r.ofv_prior > 0.0, "the fixture must actually carry a prior");

    let ci = r.sir_ci_theta.as_ref().expect("SIR theta CIs")[i];
    let est = r.theta[i];
    let mid = 0.5 * (ci.0 + ci.1);

    // Upper bound: 0.1927 correct vs 0.1572 with the prior dropped. Bound placed
    // between them, nearer the wrong value so noise in the right arm cannot
    // reach it.
    assert!(
        ci.1 > 0.175,
        "SIR upper bound collapsed toward the MLE — the weights are scoring the \
         data alone: CI {ci:?}, est {est}, MLE {cl_mle}, prior {prior_at}"
    );
    // Direction: correct puts the resampled mass *above* the MAP point estimate
    // (toward the prior), the bug puts it below (toward the MLE). Measured
    // `mid - est` is +0.0093 vs -0.0150, so the sign alone separates them.
    assert!(
        mid > est,
        "the SIR distribution must sit toward the prior, not the MLE: \
         mid {mid}, est {est}, CI {ci:?}"
    );
}

// ── NONMEM anchor for the prior penalty ─────────────────────────────────────

/// The prior's contribution to the objective, anchored against **NONMEM
/// `$PRIOR NWPRI`**.
///
/// # Why this is an anchor and not a story
///
/// ferx and NONMEM do not agree on a prior objective by construction — they
/// disagree about the parameterisation (`CL` vs `log CL`), about which
/// normalisation constants the objective carries, and about the optimizer path.
/// The design removes all three:
///
/// 1. **Same coordinate.** The model is written on the log scale on both sides
///    (`CL = EXP(THETA(1) + ETA(1))`), and the ferx θ is declared with a
///    *negative* lower bound so ferx packs it as the identity. Both engines
///    therefore put a normal prior on the same number, with the same mean and
///    the same SD — nothing depends on ferx's packing conventions.
/// 2. **Same point.** Both run `MAXEVAL=0` / `maxiter = 0`. The quantity under
///    test is the objective, not a minimizer's path to it, so pinning the point
///    removes any chance the two engines' optimizers land somewhere different
///    and confound the comparison.
/// 3. **A measured null control first.** `prior_theta_null.ctl` and
///    `prior_theta_null.ferx` are the same model with **no** prior. They agree
///    (NONMEM 177.82632978040530, ferx 177.826330), which is what makes the
///    priored comparison meaningful: without it, an agreeing prior term could be
///    hiding two compensating disagreements.
///
/// A third NONMEM run, `prior_theta_nwpri_centered.ctl`, puts the prior mean
/// *exactly on* `THETA(1)` and reproduces the null objective to printed
/// precision — so NWPRI, like ferx, drops the prior's normalisation constant.
/// That is why the absolute values below can be compared directly, rather than
/// only as a difference.
///
/// # Measured
///
/// | | NONMEM 7.5.1 | ferx |
/// |---|---|---|
/// | null (no prior) | 177.82632978040530 | 177.826330 |
/// | prior centred on θ | 177.826 (== null) | — |
/// | prior offset from θ | 177.94393305151641 | 177.943933 |
/// | **prior contribution** | **0.11760327111** | **0.117603** |
///
/// Worst realised disagreement **5.2e-8** on the priored objective and 2.2e-7 on
/// the null — both at the limit of the six decimals ferx's fit YAML prints, not
/// a real gap. The closed form is `((-2.0 + 1.89712) / 0.30)² = 0.117603266`,
/// which both engines reproduce, so this pins ferx against NONMEM *and* against
/// arithmetic. Bound set at 1e-6: two orders above the realised error, and far
/// below the 1e-3 that would let the ½-scale or a dropped constant through.
#[test]
fn the_prior_penalty_matches_nonmem_nwpri() {
    let pop = warfarin_population();
    let mut opts = FitOptions {
        method: EstimationMethod::FoceI,
        ..FitOptions::default()
    };
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;

    // NONMEM 7.5.1, `nonmem_anchor/prior_theta_null.ctl` (MAXEVAL=0).
    const NM_NULL: f64 = 177.826_329_780_405_3;
    // NONMEM 7.5.1, `nonmem_anchor/prior_theta_nwpri.ctl` (MAXEVAL=0).
    const NM_PRIORED: f64 = 177.943_933_051_516_41;

    let null = parse_model_string(
        &std::fs::read_to_string("nonmem_anchor/prior_theta_null.ferx").expect("null model"),
    )
    .expect("null model parses");
    let priored = parse_model_string(
        &std::fs::read_to_string("nonmem_anchor/prior_theta_nwpri.ferx").expect("priored model"),
    )
    .expect("priored model parses");

    let r_null = fit_model(&null, &pop, &opts).expect("null run");
    let r_prior = fit_model(&priored, &pop, &opts).expect("priored run");

    // The null control. If this fails the priored comparison below means nothing,
    // so it is asserted first and separately.
    assert!(
        (r_null.ofv - NM_NULL).abs() < 1e-6,
        "null control disagrees with NONMEM: ferx {} vs NONMEM {NM_NULL}",
        r_null.ofv
    );
    assert_eq!(r_null.ofv_prior, 0.0, "the null run must carry no prior");

    // The priored objective, absolutely — valid because the centered NONMEM run
    // showed NWPRI drops the same constant ferx does.
    assert!(
        (r_prior.ofv - NM_PRIORED).abs() < 1e-6,
        "priored objective disagrees with NONMEM: ferx {} vs NONMEM {NM_PRIORED}",
        r_prior.ofv
    );

    // And the prior's own contribution, as a difference — the form that would
    // still hold if either engine changed its constants.
    let nm_delta = NM_PRIORED - NM_NULL;
    assert!(
        (r_prior.ofv_prior - nm_delta).abs() < 1e-6,
        "prior contribution disagrees with NONMEM: ferx {} vs NONMEM {nm_delta}",
        r_prior.ofv_prior
    );

    // The data half must be untouched by the prior: same model, same point.
    assert!(
        (r_prior.ofv_data - r_null.ofv).abs() < 1e-9,
        "the prior moved the data likelihood: {} vs {}",
        r_prior.ofv_data,
        r_null.ofv
    );

    // Non-degeneracy: the prior term has to be doing something, or every
    // assertion above is satisfied by an engine that ignores priors entirely.
    assert!(
        r_prior.ofv_prior > 0.1,
        "the anchor fixture must exercise a non-zero penalty, got {}",
        r_prior.ofv_prior
    );
}

/// The issue's model-updating acceptance criterion, end to end and in one call
/// (#254 phase 2): fit a parent model, write the run's own `{model}-fit.yaml`,
/// and refit on **sparse** data with a second model that names that file as its
/// prior.
///
/// The oracle is the parent fit's own estimates, and the whole test is the
/// three-way comparison — parent, sparse-with-prior, sparse-without-prior:
///
/// - the sparse **control** (no prior) has to drift away from the parent, or
///   nothing below tests anything: a prior is only observable where the data
///   alone would land somewhere else;
/// - the sparse **update** has to sit closer to the parent than the control
///   does, on the parameter the control drifted on.
///
/// Refitting the *same* data with a prior centred on its own MLE cannot be this
/// test: the likelihood and the prior are then maximised in the same place and
/// an engine that ignores priors entirely passes.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_sparse_fit_can_be_updated_from_a_parent_runs_written_estimates() {
    let full = warfarin_population();
    let mut parent_opts = FitOptions::default();
    // The import reads standard errors, so the parent run must do its
    // covariance step. That is what makes this Tier 3 rather than Tier 2.
    parent_opts.run_covariance_step = true;

    let parent = fit_model(&warfarin_with(BASE_PARAMS), &full, &parent_opts).expect("parent fit");
    assert!(
        parent.se_theta.is_some(),
        "the parent run must report standard errors for the import to read"
    );

    let dir = tempfile::tempdir().unwrap();
    let fit_yaml = dir.path().join("parent-fit.yaml");
    ferx_core::io::output::write_estimates_yaml(&parent, fit_yaml.to_str().unwrap())
        .expect("parent estimates must write");

    // The local dataset: two subjects out of ten, whose concentrations run 60%
    // high. The scaling is what makes the fixture non-degenerate — warfarin is
    // informative enough that two untouched subjects already reproduce the
    // parent's θ to ~2% (measured), leaving the prior nothing to pull back. A
    // uniform ×1.6 on DV moves the local MLE's `TVV` by construction, which is
    // exactly the situation the feature exists for: a small local dataset that
    // disagrees with the published model.
    let mut sparse = warfarin_population();
    sparse.subjects.truncate(2);
    for s in sparse.subjects.iter_mut() {
        for dv in s.observations.iter_mut() {
            *dv *= 1.6;
        }
    }

    let update_src = format!(
        "[parameters]\n{BASE_PARAMS}\
         [priors]\n  from_fit = \"{}\"\n\n\
         [individual_parameters]\n\
         \x20 CL = TVCL * exp(ETA_CL)\n\
         \x20 V  = TVV  * exp(ETA_V)\n\
         \x20 KA = TVKA\n\n\
         [structural_model]\n\
         \x20 pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP_ERR)\n",
        fit_yaml.display()
    );
    let updated = parse_model_string(&update_src).expect("the update model must parse");

    let mut opts = FitOptions::default();
    opts.run_covariance_step = false;
    let child = fit_model(&updated, &sparse, &opts).expect("update fit");
    let control =
        fit_model(&warfarin_with(BASE_PARAMS), &sparse, &opts).expect("sparse control fit");

    assert_eq!(child.prior_summary.len(), 6, "{:?}", child.prior_summary);

    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    let idx = |name: &str| parent.theta_names.iter().position(|n| n == name).unwrap();

    // Non-degeneracy first: the sparse control must actually drift, or the
    // closeness assertion below is satisfied by any engine at all. Realised
    // drift is recorded in the failure message so a dataset change that quietly
    // makes the fixture degenerate says so.
    let drifted: Vec<&str> = ["TVCL", "TVV", "TVKA"]
        .into_iter()
        .filter(|n| rel(control.theta[idx(n)], parent.theta[idx(n)]) > 0.10)
        .collect();
    assert!(
        !drifted.is_empty(),
        "degenerate fixture: two subjects already reproduce the parent, so there is \
         nothing for the prior to pull back — control {:?}, parent {:?}",
        control.theta,
        parent.theta
    );

    // …and on every parameter the control drifted on, the prior pulled the
    // update back toward the parent.
    for name in drifted {
        let i = idx(name);
        let (d_child, d_control) = (
            rel(child.theta[i], parent.theta[i]),
            rel(control.theta[i], parent.theta[i]),
        );
        assert!(
            // Measured: TVCL child 5.2% vs control 36.5% off the parent, TVV
            // 5.0% vs 38.0% — a factor of ~7 either way. The bound asks only for
            // a factor of 2, so it fails on a prior that is applied weakly as
            // well as on one that is not applied at all.
            d_child < d_control * 0.5,
            "{name}: the imported prior must pull the sparse fit back toward the parent — \
             parent {}, updated {} ({:.1}% off), control {} ({:.1}% off)",
            parent.theta[i],
            child.theta[i],
            d_child * 100.0,
            control.theta[i],
            d_control * 100.0
        );
    }
}
