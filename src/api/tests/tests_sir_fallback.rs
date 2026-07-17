use super::*;
use std::path::Path;

// ── should_run_sir_fallback (pure gate, #264) ────────────────────────────

#[test]
fn sir_fallback_gate_fires_only_when_all_conditions_hold() {
    // Opted in, no real covariance, no normal SIR, proposal present.
    assert!(should_run_sir_fallback(true, false, false, true));
}

#[test]
fn sir_fallback_gate_blocked_by_each_condition() {
    // Each single deviation from the firing case blocks the fallback.
    assert!(!should_run_sir_fallback(false, false, false, true)); // covariance_fallback != sir
    assert!(!should_run_sir_fallback(true, true, false, true)); // a real H⁻¹ covariance exists
    assert!(!should_run_sir_fallback(true, false, true, true)); // a normal sir=true run already produced CIs
    assert!(!should_run_sir_fallback(true, false, false, false)); // compute_covariance produced no proposal
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
