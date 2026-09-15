use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const MODEL: &str = "\
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
";

#[test]
fn fit_result_carries_fixed_block_sigma_correlations() {
    let model = parse_model_string(MODEL).expect("block_sigma model must parse");
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("correlated residual data must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };

    let result = fit(&model, &population, &model.default_params, &options)
        .expect("initial correlated-residual evaluation must succeed");

    // Structural equality only. Since #847 the reported rho comes off the packed
    // parameter vector, so even a `FIX`ed one round-trips through
    // `tanh(atanh(rho))` — exact `assert_eq!` on the f64 is a last-ULP coin flip
    // that depends on the codegen (it held locally and failed under the coverage
    // build). The value itself is checked to tolerance just below.
    assert_eq!(
        result.residual_correlations.len(),
        model.residual_correlations.len()
    );
    let corr = result.residual_correlations[0];
    let declared = model.residual_correlations[0];
    assert_eq!(corr.sigma_i, declared.sigma_i);
    assert_eq!(corr.sigma_j, declared.sigma_j);
    assert!((corr.rho - declared.rho).abs() < 1e-12);
    assert_eq!(result.sigma_names[corr.sigma_i], "ADD_ERR");
    assert_eq!(result.sigma_names[corr.sigma_j], "PROP_ERR");
    assert!((corr.rho - 0.5).abs() < 1e-12);
    let covariance = corr.rho * result.sigma[corr.sigma_i] * result.sigma[corr.sigma_j];
    assert!((covariance - 0.1).abs() < 1e-12);
}

/// #847: a bare `block_sigma` (no `FIX`) **estimates** its off-diagonal — NONMEM
/// `$SIGMA BLOCK(n)` semantics — so the fit must report it as a free coordinate.
///
/// `outer_maxiter: 0` keeps this a Tier-2 test (one evaluation, no convergence
/// loop): it pins the *plumbing* — the flag reaches `FitResult`, rho is packed as
/// a free coordinate, and the reported value is the parameter vector's, not the
/// model's declaration. That the optimizer actually moves rho is the slow test
/// `dense_residual_free_rho_beats_fixed_rho`.
#[test]
fn fit_result_marks_a_bare_block_sigma_correlation_free() {
    use ferx_core::estimation::parameterization::{compute_bounds, packed_fixed_mask};

    let src = MODEL.replace("1.00] FIX", "1.00]");
    let model = parse_model_string(&src).expect("free block_sigma model must parse");
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("correlated residual data must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };

    let params = &model.default_params;
    assert_eq!(params.residual_correlation_fixed, vec![false]);
    // The rho coordinate is packed last, free (lower < upper) rather than pinned.
    let mask = packed_fixed_mask(params);
    let bounds = compute_bounds(params);
    let rho_idx = mask.len() - 1;
    assert!(!mask[rho_idx]);
    assert!(bounds.lower[rho_idx] < bounds.upper[rho_idx]);
    // The sigma SDs stay free too — `FIX` is what pinned them before.
    assert_eq!(params.sigma_fixed, vec![false, false]);

    let result = fit(&model, &population, params, &options)
        .expect("free correlated-residual evaluation must succeed");
    assert_eq!(result.residual_correlation_fixed, vec![false]);
    assert!((result.residual_correlations[0].rho - 0.5).abs() < 1e-12);
}

/// The `FIX` companion to the test above: the whole block is pinned, so no
/// optimizer that respects box bounds can move either the SDs or rho (#847).
#[test]
fn fit_result_marks_a_fixed_block_sigma_correlation_pinned() {
    use ferx_core::estimation::parameterization::{compute_bounds, packed_fixed_mask};

    let model = parse_model_string(MODEL).expect("block_sigma model must parse");
    let params = &model.default_params;
    assert_eq!(params.residual_correlation_fixed, vec![true]);
    let mask = packed_fixed_mask(params);
    let bounds = compute_bounds(params);
    let rho_idx = mask.len() - 1;
    assert!(mask[rho_idx]);
    assert_eq!(bounds.lower[rho_idx], bounds.upper[rho_idx]);
}

/// #847: only FOCE/FOCEI estimate the `block_sigma` off-diagonal, and the method
/// chain forwards fitted parameters to the next stage. A stage that cannot read
/// the estimated rho would score the *declared* one while the result reported the
/// estimate, so the configuration is refused rather than mis-scored.
#[test]
fn block_sigma_chain_after_focei_is_rejected() {
    let src = MODEL.replace("1.00] FIX", "1.00]");
    let model = parse_model_string(&src).expect("free block_sigma model must parse");
    let options = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Imp],
        ..FitOptions::default()
    };
    let diags = ferx_core::check_model_options(&model, &options);
    let hit = diags
        .iter()
        .find(|d| d.code == "E_BLOCK_SIGMA_CHAIN_UNSUPPORTED")
        .expect("a rho-estimating stage followed by IMP must be rejected");
    assert!(hit.message.contains("FIX"), "{}", hit.message);
}

/// The same chain is fine when the block is `FIX`ed — rho never moves, so every
/// stage scores the declaration and the reported value agrees with it (#847).
#[test]
fn block_sigma_chain_after_focei_is_allowed_when_fixed() {
    let model = parse_model_string(MODEL).expect("block_sigma model must parse");
    let options = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Imp],
        ..FitOptions::default()
    };
    let diags = ferx_core::check_model_options(&model, &options);
    assert!(!diags
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_CHAIN_UNSUPPORTED"));
}

/// A non-estimating stage placed **first** is fine: rho has not moved yet, so it
/// scores the declaration, which is still the live value (#847).
#[test]
fn block_sigma_chain_before_focei_is_allowed() {
    let src = MODEL.replace("1.00] FIX", "1.00]");
    let model = parse_model_string(&src).expect("free block_sigma model must parse");
    let options = FitOptions {
        methods: vec![EstimationMethod::Saem, EstimationMethod::FoceI],
        ..FitOptions::default()
    };
    let diags = ferx_core::check_model_options(&model, &options);
    assert!(!diags
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_CHAIN_UNSUPPORTED"));
}

/// #847 / review finding: AGQ's score assembles theta/omega/sigma/omega_iov and
/// never writes the trailing rho slot, so a model with a **free** block_sigma must
/// decline the analytic gradient and fall to `reconverged_fd_gradient`, which
/// differences every free packed coordinate. A hard zero there would leave the
/// optimizer no reason to move rho while the objective genuinely depends on it.
#[test]
fn agq_declines_the_analytic_gradient_for_a_free_block_sigma() {
    let src = MODEL.replace("1.00] FIX", "1.00]");
    let model = parse_model_string(&src).expect("free block_sigma model must parse");
    assert_eq!(model.default_params.residual_correlation_fixed, vec![false]);
    assert!(
        !ferx_core::estimation::agq::analytic_gradient_available(&model),
        "a free block_sigma must route AGQ/Laplace to the reconverged-FD gradient"
    );
}

/// A `FIX`ed block carries no free rho coordinate for the score to miss, so the
/// analytic AGQ gradient stays available — the fallback above must not become a
/// blanket opt-out for every correlated model.
#[test]
fn agq_keeps_the_analytic_gradient_for_a_fixed_block_sigma() {
    let model = parse_model_string(MODEL).expect("block_sigma model must parse");
    assert_eq!(model.default_params.residual_correlation_fixed, vec![true]);
    assert!(ferx_core::estimation::agq::analytic_gradient_available(
        &model
    ));
}

/// The chain guard keys on "can this stage move rho", which is not the same set as
/// "has an analytic rho gradient": `laplace` moves rho through AGQ's FD fallback,
/// so `[laplace, imp]` must be rejected exactly like `[focei, imp]` (#847).
#[test]
fn block_sigma_chain_after_laplace_is_rejected() {
    let src = MODEL.replace("1.00] FIX", "1.00]");
    let model = parse_model_string(&src).expect("free block_sigma model must parse");
    let options = FitOptions {
        methods: vec![EstimationMethod::Laplace, EstimationMethod::Imp],
        ..FitOptions::default()
    };
    let diags = ferx_core::check_model_options(&model, &options);
    assert!(diags
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_CHAIN_UNSUPPORTED"));
}

/// #1307. A `FIX`-ed correlation **above the Fisher-z estimation rail** reaches
/// the fit at the declared value — the contract `src/types.rs` documents for
/// `block_sigma ... FIX`, which `pack_rho`'s `clamp` used to break silently.
///
/// The fixture above is ρ = 0.5, four rail-widths inside, which is why
/// `fit_result_carries_fixed_block_sigma_correlations` could not see this: every
/// declared ρ above `tanh(3) = 0.995_055` collapsed onto that one number, so
/// 0.999 and 0.9999 both fit as 0.995055 and the covariance the model declared
/// was not the covariance scored.
///
/// The covariance is the assertion that makes it end-to-end rather than a
/// packing round-trip: `ρ · σ_i · σ_j` must reproduce the off-diagonal written
/// in the model file, and both σ are `FIX`ed here too, so every factor is the
/// user's own number. Under the old behaviour it came back as 0.0597 against a
/// declared 0.05994 — a 0.9% error in R, silently, on every evaluation.
///
/// `outer_maxiter: 0` keeps this Tier 2: one evaluation, no convergence loop.
#[test]
fn a_fixed_block_sigma_correlation_above_the_rail_is_held_at_the_declared_value() {
    // σ_prop = 0.2, σ_add = 0.3, ρ = 0.999 ⇒ cov = 0.999 · 0.2 · 0.3 = 0.05994.
    const DECLARED_COV: f64 = 0.05994;
    const DECLARED_RHO: f64 = 0.999;
    let src = MODEL.replace("[0.04, 0.10, 1.00] FIX", "[0.04, 0.05994, 0.09] FIX");
    let model = parse_model_string(&src).expect("block_sigma model must parse");
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("correlated residual data must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };

    // The premise: this ρ is outside the rail a *free* one would carry, so the
    // test is capable of observing the fix. Without it the whole thing passes
    // whether the rail is consulted or not.
    let rail = 3.0_f64.tanh();
    assert!(
        DECLARED_RHO > rail,
        "the fixture must declare a ρ above the {rail} rail"
    );
    assert!((model.residual_correlations[0].rho - DECLARED_RHO).abs() < 1e-12);

    let result = fit(&model, &population, &model.default_params, &options)
        .expect("initial correlated-residual evaluation must succeed");

    let corr = result.residual_correlations[0];
    assert!(
        (corr.rho - DECLARED_RHO).abs() < 1e-9,
        "a FIX-ed ρ must reach the fit as declared; got {} (the pre-#1307 value \
         was {rail}, the rail itself)",
        corr.rho
    );
    // The covariance the residual block is actually built from — the number in
    // the model file, reassembled from what the fit reports.
    let covariance = corr.rho * result.sigma[corr.sigma_i] * result.sigma[corr.sigma_j];
    assert!(
        (covariance - DECLARED_COV).abs() < 1e-9,
        "the scored covariance must be the declared one; got {covariance}"
    );
    // …and the warning the free spelling earns is absent here, because there is
    // nothing to warn about: the declared value survived.
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("W_INIT_NOT_REPRESENTABLE")),
        "{:#?}",
        result.warnings
    );
}

/// The straddle for the test above: the identical declaration **without** `FIX`
/// still takes the rail — the rail is an argument about what may be *searched*,
/// and #1307 changed only the held case — and now says so in
/// `FitResult.warnings`, where before it was silent.
///
/// Same data, same numbers, one keyword different. If this arm ever stops
/// railing, the test above is passing for the wrong reason.
#[test]
fn a_free_block_sigma_correlation_above_the_rail_is_railed_and_reported() {
    let src = MODEL.replace("[0.04, 0.10, 1.00] FIX", "[0.04, 0.05994, 0.09]");
    let model = parse_model_string(&src).expect("free block_sigma model must parse");
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("correlated residual data must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };

    let result = fit(&model, &population, &model.default_params, &options)
        .expect("initial correlated-residual evaluation must succeed");

    assert!(
        (result.residual_correlations[0].rho - 3.0_f64.tanh()).abs() < 1e-9,
        "a free ρ must still be railed; got {}",
        result.residual_correlations[0].rho
    );
    let hit = result
        .warnings
        .iter()
        .find(|w| w.contains("W_INIT_NOT_REPRESENTABLE"))
        .unwrap_or_else(|| panic!("the rail must be reported: {:#?}", result.warnings));
    assert!(hit.contains("9.99e-1"), "{hit}");
}
