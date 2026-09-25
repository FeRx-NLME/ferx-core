//! #1512: the strictness gate must exclude a fit whose covariance step had to floor a
//! Hessian eigenvalue.
//!
//! The fixture is the collapsed one-peripheral candidate `ferx modelsearch` builds from
//! `examples/warfarin.ferx` over `PERIPHERALS(0..1)`: warfarin carries no information about a
//! second compartment, so the fit collapses it (V2 → ~3e-5, Q free, OFV equal to the
//! one-compartment base). That leaves one flat Hessian direction, which the covariance step's
//! eigenvalue floor replaces with a finite one — and the condition number and correlations the
//! numeric gates read are then computed from the floored matrix, which no longer shows the
//! collapse. Measured on the unpinned candidate before #1543: correlation-matrix condition
//! number 2.98, TVQ RSE 293519 %, and the candidate passed `Strictness::default()`.
//!
//! # Why `TVV2` is fixed (#1553)
//!
//! The first version of this file let the fit find the collapse itself, and the straddle then
//! depended on *where along the collapse* the outer optimizer parked — a flat direction has no
//! preferred point. Measured on the same fixture: `main@293ac058` parked at TVV2 ≈ 2e-4 with a
//! floored matrix of condition number 1.6e7 (the numeric gate already failed, so the fixture
//! no longer isolated the regularization check); with `FERX_NO_INNER_HESSIAN_SEED=1`, 5.7e4;
//! on PR #1504's inner-solve trajectory nothing floored at all (condition number 12.5).
//!
//! So the collapse is now *given*, not found: `TVV2` is fixed at 5e-5 and `Q` starts at 6.
//! With V2 that small the peripheral equilibrates in microseconds, `Q` moves the objective only
//! through a term of order `V2²/Q`, and the outer optimizer leaves it where it started (it
//! moves by ~3e-8 relative). The other three θ start cold (`examples/warfarin.ferx`'s
//! initials) so the fit genuinely moves and converges, which keeps the init-stall and
//! convergence gates out of the verdict. The landing point is therefore pinned by
//! construction: every trajectory reaches the same well-identified (CL, V, KA) optimum with
//! the same `Q`, and the covariance step sees the same matrix.
//!
//! The pin sits in a measured window, and both of its edges are asserted below:
//!
//! * **Too flat, and `Q` is frozen before the fit starts.** The outer pre-flight freezes a θ
//!   whose initial gradient is below `1e-8` (`[parameters] … has no effect on the objective`),
//!   and a frozen `Q` leaves no flat direction to floor. Measured: frozen whenever
//!   `V2²/Q ≲ 1.4e-10` (V2 = 5e-5 frozen at Q₀ = 20, not at 16; V2 = 1e-4 frozen at Q₀ = 80,
//!   not at 60). Here `V2²/Q₀ = 4.2e-10`, 3× inside.
//! * **Not flat enough, and the floor does not fire.** The floor clips an eigenvalue below
//!   `1e-10·λ_max = 2.285e-7`. Measured here: min eigenvalue 2.50e-8 (2.43e-8 with the inner
//!   seed killed), 9× inside. Across the window (V2 ∈ [2e-5, 1e-4]) it was 1e-8 to 1.1e-7.
//!
//! The test asserts the straddle first — the same fit with the floor warning removed *passes*
//! `Strictness::default()` outright — so the exclusion it then asserts can only come from the
//! regularization check.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{
    check_strictness, fit, max_abs_correlation, read_nonmem_csv, run_covariance, CovarianceMethod,
    EstimationMethod, FitOptions, FitResult, Strictness, WarningCode,
};
use std::path::Path;

/// `ferx modelsearch`'s `run1` for `examples/warfarin_modelsearch.ferxsearch` with the
/// collapse pinned: `TVV2` fixed at a collapsed value, `TVQ` started inside the window
/// described in the module docs, `TVCL`/`TVV`/`TVKA` at `examples/warfarin.ferx`'s initials.
/// Ω and σ keep the candidate's values.
const WARFARIN_COLLAPSED_PERIPHERAL: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.0285950800457939
  omega ETA_V  ~ 0.00957691316624095
  omega ETA_KA ~ 0.348964093203287

  sigma PROP_ERR ~ 0.0107485219888545 (sd)
  theta TVQ(6.0, 0.0, 1000000.0)
  theta TVV2(5e-5, FIX)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  Q = TVQ
  V2 = TVV2

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

const REGULARIZED_FAILURE: &str = "covariance matrix regularized:";

fn options(m: CovarianceMethod) -> FitOptions {
    // `parse_model_string` drops `[fit_options]`, so the method is pinned here.
    FitOptions {
        method: EstimationMethod::Foce,
        interaction: false,
        outer_maxiter: 300,
        run_covariance_step: true,
        covariance_method: m,
        verbose: false,
        ..FitOptions::default()
    }
}

fn fit_under(m: CovarianceMethod) -> (ferx_core::CompiledModel, ferx_core::Population, FitResult) {
    let model = parse_model_string(WARFARIN_COLLAPSED_PERIPHERAL).expect("model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");
    let r = fit(&model, &pop, &model.default_params, &options(m)).expect("fit runs");
    (model, pop, r)
}

/// The eigenvalue-floor message, if the covariance step emitted one.
fn floor_message(r: &FitResult) -> Option<&str> {
    r.warnings_structured
        .iter()
        .find(|w| {
            w.category == WarningCode::CovarianceRegularized
                && w.message.contains("eigenvalue floor applied")
        })
        .map(|w| w.message.as_str())
}

/// `<key> = <number>` out of the floor message.
fn field(msg: &str, key: &str) -> f64 {
    let tail = &msg[msg
        .find(key)
        .unwrap_or_else(|| panic!("no `{key}` in {msg}"))
        + key.len()..];
    let end = tail.find([',', ';']).unwrap_or(tail.len());
    tail[..end]
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("`{key}` in {msg}: {e}"))
}

/// `r` with the eigenvalue-floor warning removed: the fit as the numeric gates alone see it.
fn without_floor_warning(r: &FitResult) -> FitResult {
    let mut bare = r.clone();
    bare.warnings_structured
        .retain(|w| w.category != WarningCode::CovarianceRegularized);
    bare
}

fn cites_regularization(r: &FitResult) -> bool {
    check_strictness(r, &Strictness::default())
        .failures
        .iter()
        .any(|f| f.starts_with(REGULARIZED_FAILURE))
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_collapsed_peripheral_fails_strictness_on_its_regularized_covariance() {
    let (_, _, r) = fit_under(CovarianceMethod::Hessian);

    // The pin held: Q is still free (the pre-flight did not freeze it, which would leave no
    // flat direction), it did not wander, and the fit converged from its cold start.
    assert!(
        !r.warnings.iter().any(|w| w.contains("has no effect")),
        "TVQ was frozen before the fit — the fixture left its window (V2²/Q too small): {:?}",
        r.warnings
    );
    let q = r.theta[r.theta_names.iter().position(|n| n == "TVQ").unwrap()];
    assert!(
        (q / 6.0 - 1.0).abs() < 1e-3,
        "TVQ = {q}: the flat direction moved, so the landing point is no longer pinned"
    );
    assert!(r.converged, "{:?}", r.warnings);

    // The collapse the fixture exists for: the covariance step floored the flat direction.
    // The margin is a canary — measured 2.50e-8 against a 2.285e-7 floor, so a drift that
    // halves it (toward the edge where nothing floors) reddens here, with the numbers, rather
    // than silently flipping the straddle below.
    let msg = floor_message(&r)
        .unwrap_or_else(|| panic!("the eigenvalue floor did not fire: {:?}", r.warnings));
    let (min_eig, floor) = (field(msg, "min eig = "), field(msg, "floor = "));
    assert!(
        min_eig.is_finite() && floor.is_finite() && min_eig < floor / 4.0,
        "min eig {min_eig:e} vs floor {floor:e}: the floored eigenvalue is within 4× of the \
         floor, so the fixture is at the edge of its window"
    );

    // The straddle, both sides from the one fit. Without the floor warning — the matrix as the
    // numeric gates see it — the fit passes `Strictness::default()` outright: every gate,
    // not just the two thresholds, so nothing but the regularization check can fail it.
    let s = Strictness::default();
    let cn = r.cov_condition_number.expect("condition number");
    let max_r = max_abs_correlation(&r).expect("correlation");
    eprintln!(
        "TVQ {q}, min eig {min_eig:e}, floor {floor:e}, condition number {cn:e}, max |r| \
         {max_r}, ofv {}",
        r.ofv
    );
    let bare = check_strictness(&without_floor_warning(&r), &s);
    assert!(
        bare.passed,
        "without the floor warning the fit already fails (condition number {cn:e}, max |r| \
         {max_r}), so this fixture no longer isolates the regularization check: {bare:?}"
    );
    // With it, the fit fails, for exactly that one reason.
    let v = check_strictness(&r, &s);
    assert!(!v.passed, "a collapsed peripheral passed strictness: {v:?}");
    assert_eq!(v.failures.len(), 1, "{v:?}");
    assert!(v.failures[0].starts_with(REGULARIZED_FAILURE), "{v:?}");
}

/// The gate follows the covariance step that produced the fit's matrix.
///
/// `rsr` reports `R⁻¹ S R⁻¹`, which uses the floored `R`, so it must be cited as `r` is.
///
/// The second leg is `run_covariance` re-running the `r` fit under `s`: the incoming fit
/// carries the floor warning, and a re-run that kept it would have the gate exclude the fit
/// over a step that no longer exists. On this fixture the `s` step itself *fails* ("the score
/// cross-product matrix S is singular or rank-deficient", measured), so the re-run stores no
/// matrix. That is why there is no
/// fresh-`s`-fit leg here: it would pass by skipping (no matrix, nothing to read), whatever
/// the estimator gating did — measured, making the floor warning fire under `s` too left such
/// a leg green. The re-run leg is live regardless, since the stale warning is carried in on
/// `r`: dropping `CovarianceRegularized` from `run_covariance`'s superseded set reddens it.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_regularization_failure_follows_the_covariance_step() {
    let (model, pop, r) = fit_under(CovarianceMethod::Hessian);
    assert!(cites_regularization(&r), "r: {:?}", r.warnings);
    let (_, _, rsr) = fit_under(CovarianceMethod::Sandwich);
    assert!(cites_regularization(&rsr), "rsr: {:?}", rsr.warnings);

    let rerun = run_covariance(
        &r,
        Some(&model),
        Some(&pop),
        &options(CovarianceMethod::CrossProduct),
    )
    .expect("s re-run");
    assert!(
        rerun.covariance_matrix.is_none(),
        "the premise changed: `s` now succeeds on this fixture, so add a fresh-`s`-fit leg \
         and update the doc comment"
    );
    assert!(
        !cites_regularization(&rerun),
        "r -> s re-run kept a stale floor warning: {:?}",
        rerun.warnings
    );
}
