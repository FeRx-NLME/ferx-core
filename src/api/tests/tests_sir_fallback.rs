use super::*;
use std::path::Path;

// ── should_run_sir_fallback (pure gate, #264) ────────────────────────────

#[test]
fn sir_fallback_gate_fires_only_when_all_conditions_hold() {
    // Opted in, no real covariance, no normal SIR, proposal present.
    assert!(should_run_sir_fallback(true, false, false, false, true));
}

#[test]
fn sir_fallback_gate_blocked_by_each_condition() {
    // Each single deviation from the firing case blocks the fallback.
    assert!(!should_run_sir_fallback(false, false, false, false, true)); // neither trigger set
    assert!(!should_run_sir_fallback(true, false, true, false, true)); // a real H⁻¹ covariance exists
    assert!(!should_run_sir_fallback(true, false, false, true, true)); // a normal sir=true run already produced CIs
    assert!(!should_run_sir_fallback(true, false, false, false, false)); // compute_covariance produced no proposal
}

/// #972: `sir = true` alone arms the non-PD fallback, with no separate
/// `covariance_fallback = sir` needed — the option the user naturally reaches
/// for now reaches the capability built for exactly this case.
#[test]
fn sir_requested_alone_arms_the_non_pd_fallback() {
    assert!(should_run_sir_fallback(false, true, false, false, true));
    // …and the other three conditions still gate it exactly as before.
    assert!(!should_run_sir_fallback(false, true, true, false, true)); // real covariance exists
    assert!(!should_run_sir_fallback(false, true, false, true, true)); // normal SIR already ran
    assert!(!should_run_sir_fallback(false, true, false, false, false)); // no proposal to run off
}

// ── sir_unavailable_warning (#972) ───────────────────────────────────────

#[test]
fn sir_unavailable_warning_is_silent_when_sir_is_not_stranded() {
    // Not requested at all.
    assert!(sir_unavailable_warning(false, true, false, false, false, false).is_none());
    // A covariance exists: the standard path reports its own failures.
    assert!(sir_unavailable_warning(true, true, false, true, false, false).is_none());
    // SIR actually ran (via either path).
    assert!(sir_unavailable_warning(true, true, false, false, true, true).is_none());
    // The fallback fired off a proposal and failed — that path already warned,
    // so this must not stack a second message on the same failure.
    assert!(sir_unavailable_warning(true, true, false, false, true, false).is_none());
}

#[test]
fn sir_unavailable_warning_points_at_covariance_when_the_step_never_ran() {
    let msg = sir_unavailable_warning(true, false, false, false, false, false)
        .expect("stranded SIR with no covariance step must warn");
    assert!(
        msg.contains("covariance = true"),
        "warning should point at the covariance option: {msg}"
    );
}

/// The pre-#972 warning sent every stranded user to `covariance = true`, which
/// is useless advice when the covariance step *did* run and failed. With no
/// proposal buildable the message must say SIR cannot run rather than
/// suggesting an option that is already on — and, per review #975, must not
/// assert one specific cause (a divergent eigendecomposition) when a flat FD
/// stencil, a non-finite base OFV or a singular `S` produce the same state.
#[test]
fn sir_unavailable_warning_reports_a_failed_covariance_step_without_guessing_the_cause() {
    let msg = sir_unavailable_warning(true, true, false, false, false, false)
        .expect("stranded SIR after a failed covariance step must warn");
    assert!(
        !msg.contains("covariance = true"),
        "must not tell the user to enable an option that is already on: {msg}"
    );
    assert!(
        msg.contains("no usable SIR proposal") && msg.contains("could not run"),
        "warning should explain that no proposal could be built: {msg}"
    );
    assert!(
        !msg.contains("eigen"),
        "must not blame the eigendecomposition — it is only one of several \
         covariance-step failures that leave no proposal: {msg}"
    );
}

/// A Bayesian fit never runs the covariance step (it reports posterior
/// credible intervals), so `covariance = true` + `sir = true` must not be told
/// to enable an option that is both already on and irrelevant (review #975).
#[test]
fn sir_unavailable_warning_explains_a_bayesian_fit_instead_of_blaming_covariance() {
    let msg = sir_unavailable_warning(true, true, true, false, false, false)
        .expect("stranded SIR on a Bayesian fit must warn");
    assert!(
        !msg.contains("covariance = true"),
        "must not tell a Bayesian user to enable covariance: {msg}"
    );
    assert!(
        msg.contains("Bayesian"),
        "warning should name the actual reason: {msg}"
    );
    // Silent for a Bayesian fit that did not ask for SIR.
    assert!(sir_unavailable_warning(false, true, true, false, false, false).is_none());
}

// ── resolve_sir_fallback (gate + run_sir_core + status, #264) ─────────────

fn warfarin_fixture() -> (
    CompiledModel,
    Population,
    ModelParameters,
    Vec<DVector<f64>>,
    DMatrix<f64>,
) {
    let model = crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
        .expect("warfarin model parses");
    let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let params = model.default_params.clone();
    let eta_hats: Vec<DVector<f64>> = (0..pop.subjects.len())
        .map(|_| DVector::zeros(params.omega.dim()))
        .collect();
    // Tame fallback-style proposal: small PD diagonal in packed space, so
    // draws stay near valid parameters (positive θ/σ, PD Ω) and SIR yields
    // finite weights. A real non-PD fixture risks a wide proposal whose draws
    // overflow `exp(...)` → "all invalid weights" → status `Failed`.
    let n_packed = crate::estimation::parameterization::pack_params(&params).len();
    let proposal = DMatrix::from_diagonal(&DVector::from_element(n_packed, 0.01));
    (model, pop, params, eta_hats, proposal)
}

/// `resolve_sir_fallback` short-circuits to `None` (without touching the SIR
/// machinery) when the gate declines — here because `covariance_fallback`
/// defaults to `none`. No warning is emitted for a simple decline.
#[test]
fn resolve_sir_fallback_is_none_when_option_off() {
    let (model, pop, params, eta_hats, proposal) = warfarin_fixture();
    let opts = FitOptions::default(); // covariance_fallback = None
    let mut warnings = Vec::new();
    let result = resolve_sir_fallback(
        &opts,
        false,
        false,
        Some(&proposal),
        &model,
        &pop,
        &params,
        &eta_hats,
        0.0,
        &mut warnings,
    );
    assert!(
        result.is_none(),
        "fallback must not fire when covariance_fallback = none"
    );
    assert!(
        warnings.is_empty(),
        "no warning when the gate simply declines: {warnings:?}"
    );
}

/// End-to-end fallback wiring (#264): with `covariance_fallback = sir`, no
/// real covariance, and a tame PD proposal (the part a real non-PD fit can't
/// reliably deliver), `resolve_sir_fallback` runs SIR and returns a result
/// whose θ/Ω/σ credible intervals are populated and finite — and the status
/// the caller derives from it is `SirFallback`. Slow: a full SIR pass
/// (sampling + per-draw population likelihood).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full SIR pass; opt in with --features slow-tests"
)]
fn resolve_sir_fallback_fires_and_yields_finite_cis() {
    let (model, pop, params, eta_hats, proposal) = warfarin_fixture();
    let mut opts = FitOptions::default();
    opts.covariance_fallback = CovarianceFallback::Sir;
    opts.verbose = false;
    opts.sir_samples = 400;
    opts.sir_resamples = 200;
    // Own the determinism explicitly rather than leaning on run_sir_core's
    // `None => fixed seed` fallback, so a future change to that fallback can't
    // silently make this sampling test flaky.
    opts.sir_seed = Some(20240612);

    let mut warnings = Vec::new();
    let result = resolve_sir_fallback(
        &opts,
        false,
        false,
        Some(&proposal),
        &model,
        &pop,
        &params,
        &eta_hats,
        // ofv_hat cancels in the SIR log-sum-exp weight normalisation, so any
        // finite value yields identical CIs — 0.0 keeps the fixture simple.
        0.0,
        &mut warnings,
    );

    // Derive the reported status from the actual outcome, *before* unwrapping,
    // so this checks the real fire→status mapping rather than a constant.
    assert_eq!(
        resolve_covariance_status(true, false, result.is_some()),
        CovarianceStatus::SirFallback
    );
    let sir = result.expect("fallback should fire and SIR should succeed with a tame proposal");

    assert!(!sir.ci_theta.is_empty(), "theta CIs must be populated");
    for (lo, hi) in sir
        .ci_theta
        .iter()
        .chain(&sir.ci_omega)
        .chain(&sir.ci_sigma)
    {
        assert!(
            lo.is_finite() && hi.is_finite() && lo <= hi,
            "SIR-fallback CI must be finite and ordered, got ({lo}, {hi})"
        );
    }
    assert!(
        sir.effective_sample_size.is_finite() && sir.effective_sample_size > 0.0,
        "ESS must be finite and positive, got {}",
        sir.effective_sample_size
    );
    assert!(
        !warnings.iter().any(|w| w.contains("SIR fallback failed")),
        "no failure warning expected on the success path: {warnings:?}"
    );
}
