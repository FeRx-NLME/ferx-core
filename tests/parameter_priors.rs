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
