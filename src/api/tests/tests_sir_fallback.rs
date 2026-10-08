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

// ── apply_sir_result: the one SirResult → FitResult mapping (#1713, #1758) ──

/// A synthetic `SirResult` whose every reported field is distinguishable by
/// `tag`, so a field copied from the wrong run (or not copied) shows up.
fn tagged_sir(tag: f64) -> crate::estimation::sir::SirResult {
    let settings = crate::estimation::sir::SirSettings {
        samples: 100 + tag as usize,
        seed: 1000 + tag as u64,
        df: 2.0 + tag,
        scale: SirScale::Natural,
        ..Default::default()
    };
    crate::estimation::sir::SirResult {
        ci_theta: vec![(tag, tag + 1.0)],
        ci_omega: vec![(tag + 0.1, tag + 0.2)],
        ci_sigma: vec![(tag + 0.01, tag + 0.02)],
        ci_kappa: vec![(tag + 0.03, tag + 0.04)],
        effective_sample_size: 10.0 * tag,
        resamples_packed: Some(vec![vec![tag; 3]]),
        warnings: Vec::new(),
        settings,
    }
}

/// Every SIR field of `out` equals the one `sir` should have produced.
fn assert_sir_fields_from(out: &FitResult, sir: &crate::estimation::sir::SirResult, who: &str) {
    assert_eq!(out.sir_ci_theta.as_ref(), Some(&sir.ci_theta), "{who}: θ");
    assert_eq!(out.sir_ci_omega.as_ref(), Some(&sir.ci_omega), "{who}: Ω");
    assert_eq!(out.sir_ci_sigma.as_ref(), Some(&sir.ci_sigma), "{who}: σ");
    assert_eq!(out.sir_ci_kappa.as_ref(), Some(&sir.ci_kappa), "{who}: κ");
    assert_eq!(out.sir_ess, Some(sir.effective_sample_size), "{who}: ESS");
    assert_eq!(
        out.sir_resamples_packed, sir.resamples_packed,
        "{who}: resamples"
    );
    assert_eq!(out.sir_seed, Some(sir.settings.seed), "{who}: seed");
    assert_eq!(
        out.sir_settings.as_ref(),
        Some(&sir.settings),
        "{who}: settings"
    );
}

/// T1 (#1713): with only the non-PD fallback run, all eight SIR fields come
/// from it — κ included, the field #1713 found unpinned. Mutations: drop the
/// helper's `.or(fallback)` (every assertion dies), or drop any one field
/// assignment (that field's assertion dies, its `who` naming it).
#[test]
fn apply_sir_result_fills_every_field_from_the_fallback() {
    let fb = tagged_sir(3.0);
    let mut out = crate::types::test_helpers::minimal_fit_result();
    apply_sir_result(&mut out, None, Some(&fb));
    let k = out.sir_ci_kappa.as_ref().expect("fallback κ CI");
    assert_eq!(k.len(), 1);
    assert!(k[0].0.is_finite() && k[0].1.is_finite(), "{k:?}");
    assert_sir_fields_from(&out, &fb, "fallback");
}

/// T2: with both runs present the normal SIR wins, field by field. Mutation:
/// `fallback.or(normal)` reports the fallback's values.
#[test]
fn apply_sir_result_prefers_the_normal_run() {
    let normal = tagged_sir(1.0);
    let fb = tagged_sir(3.0);
    let mut out = crate::types::test_helpers::minimal_fit_result();
    apply_sir_result(&mut out, Some(&normal), Some(&fb));
    assert_sir_fields_from(&out, &normal, "normal");
}

/// T3 (#1758): with neither run, every SIR field is cleared — including a
/// `sir_seed` the fit was handed but never used. Mutation: a helper that only
/// writes on `Some`, or that leaves `sir_seed` alone, keeps the stale values.
#[test]
fn apply_sir_result_clears_every_field_when_sir_did_not_run() {
    let mut out = crate::types::test_helpers::minimal_fit_result();
    apply_sir_result(&mut out, Some(&tagged_sir(1.0)), None);
    assert!(out.sir_seed.is_some() && out.sir_settings.is_some());
    apply_sir_result(&mut out, None, None);
    assert_eq!(out.sir_ci_theta, None);
    assert_eq!(out.sir_ci_omega, None);
    assert_eq!(out.sir_ci_sigma, None);
    assert_eq!(out.sir_ci_kappa, None);
    assert_eq!(out.sir_ess, None);
    assert_eq!(out.sir_resamples_packed, None);
    assert_eq!(out.sir_seed, None);
    assert_eq!(out.sir_settings, None);
}

/// T4 (#1713, #1758): the fallback path through `run_sir_core` on a real IOV
/// model returns a κ interval, and records the settings it scored under.
/// `warfarin_iov` (one kappa), a tame PD proposal as in `warfarin_fixture`,
/// FOCE as the file declares it — FD inner gradients. Every recorded setting
/// is off its default, `inner_optimizer` and `ebe_warm_start` included: since
/// #426 they are per call, so a non-default value here reaches only this
/// test's own draws. So a `run_sir_core` that stamped `Default` fails the
/// equality with `from_options`; that equality cannot see a field
/// `from_options` itself drops, so the two inner settings are also asserted
/// against the options directly. Mutations: stamp `SirSettings::default()` in
/// `run_sir_core`; read `inner_optimizer` or `ebe_warm_start` from the defaults
/// in `SirSettings::from_options`; drop κ from the SIR.
#[test]
fn resolve_sir_fallback_records_its_settings_and_kappa() {
    let prep = crate::api::prepare_run("examples/warfarin_iov.ferx", Some("data/warfarin_iov.csv"))
        .expect("prepare warfarin_iov");
    let model = &prep.parsed.model;
    let pop = &prep.population;
    let params = prep.init_params.clone();
    let eta_hats: Vec<DVector<f64>> = (0..pop.subjects.len())
        .map(|_| DVector::zeros(model.n_eta))
        .collect();
    let n_packed = crate::estimation::parameterization::pack_params(&params).len();
    let proposal = DMatrix::from_diagonal(&DVector::from_element(n_packed, 0.01));
    let opts = FitOptions {
        verbose: false,
        covariance_fallback: CovarianceFallback::Sir,
        sir_samples: 40,
        sir_resamples: 20,
        sir_seed: Some(1713),
        sir_df: 7.0,
        sir_scale: SirScale::Natural,
        sir_keep_samples: true,
        inner_maxiter: 150,
        inner_tol: 2e-5,
        mu_referencing: false,
        n_agq: 3,
        inner_optimizer: crate::types::InnerOptimizer::Lbfgs,
        ebe_warm_start: true,
        ode_reltol: 2e-4,
        ode_abstol: 2e-6,
        ode_max_steps: 9_000,
        ode_method: crate::ode::OdeMethod::Rodas4,
        ode_stiff_abort_after: Some(17),
        ode_auto_switch: false,
        ..prep.parsed.fit_options.clone()
    };
    let mut warnings = Vec::new();
    let sir = resolve_sir_fallback(
        &opts,
        false,
        false,
        Some(&proposal),
        model,
        pop,
        &params,
        &eta_hats,
        0.0,
        &mut warnings,
    )
    .unwrap_or_else(|| panic!("the fallback must run: {warnings:?}"));
    assert_eq!(sir.ci_kappa.len(), 1, "one κ interval: {:?}", sir.ci_kappa);
    let (lo, hi) = sir.ci_kappa[0];
    assert!(
        lo.is_finite() && hi.is_finite() && lo <= hi,
        "κ [{lo}, {hi}]"
    );
    let want = crate::estimation::sir::SirSettings::from_options(&opts);
    assert_ne!(
        want,
        crate::estimation::sir::SirSettings::default(),
        "the fixture must be off-default, or the stamp check is a tautology"
    );
    assert_eq!(sir.settings, want);
    assert_eq!(sir.settings.seed, 1713);
    // `want` comes from `from_options` itself, so the equality above cannot see a field
    // `from_options` drops; the per-call inner settings are pinned against the options directly.
    assert_eq!(
        (
            sir.settings.scoring.inner_optimizer,
            sir.settings.scoring.ebe_warm_start
        ),
        (crate::types::InnerOptimizer::Lbfgs, true),
        "the recorded inner settings must be the ones passed"
    );
}

/// T8 (#1758): a fit whose SIR did not run reports neither a seed nor
/// settings, even when it was handed a `sir_seed` (before #1758 it echoed the
/// option). warfarin, `sir = false`, no covariance step. Mutation: the
/// pre-#1758 pair — the literal echoes `options.sir_seed` *and* the helper
/// leaves `sir_seed` alone when no SIR ran. Either half alone is inert, since
/// the helper clears what the literal set (T3 pins that half).
#[test]
fn a_fit_without_sir_reports_no_sir_seed_or_settings() {
    let prep = crate::api::prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .expect("prepare warfarin");
    let opts = FitOptions {
        verbose: false,
        sir: false,
        run_covariance_step: false,
        sir_seed: Some(9),
        outer_maxiter: 2,
        ..prep.parsed.fit_options.clone()
    };
    let fit = crate::api::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("fit");
    assert_eq!(fit.sir_ess, None, "SIR must not have run");
    assert_eq!(fit.sir_seed, None);
    assert_eq!(fit.sir_settings, None);
}
