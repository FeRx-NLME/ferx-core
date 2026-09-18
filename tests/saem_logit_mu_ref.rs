//! Logit-normal mu-referencing under SAEM (issue #918), anchored to NONMEM.
//!
//! A bounded `(0,1)` parameter written as `P = 1/(1 + exp(-(THETA + ETA)))` (or
//! the equivalent `inv_logit(THETA + ETA)`) is a mu-reference: the individual
//! parameter is `g⁻¹(g(θ) + η)` with `g = logit`. Before #918 ferx recognised
//! only the lognormal and additive forms, so such a parameter was reported as
//! "not mu-referenced" and its typical value went through SAEM's numeric M-step
//! instead of the closed-form EM step — biased low on models where the
//! conditional samples are noisy.
//!
//! ## Fixture
//!
//! Parallel dual first-order absorption (fast `KA1` / slow `KA2`) into a 1-cpt
//! central compartment, with the pathway fraction `FR1` carrying logit-scale
//! IIV. The fraction is identified by the *shape* of the curve, not by the
//! exposure magnitude — unlike a bioavailability `F`, which is confounded with
//! `CL`/`V` in oral-only data, so this design tests the estimator rather than
//! the design.
//!
//! `data/logit_fraction_oral.csv` (60 subjects × 16 samples) is simulated from
//! the model itself by `nonmem_anchor/simulate_logit_fraction_data.py`, so the
//! fit is matched (well-specified) and the truths are recoverable.
//!
//! ## NONMEM anchor
//!
//! `nonmem_anchor/logit_fraction_saem.ctl` is the same model with NONMEM's
//! explicit mu syntax (`MU_3 = THETA(3)`, `FR1 = 1/(1+EXP(-(MU_3+ETA(3))))`),
//! run with `METHOD=SAEM`. See `nonmem_anchor/README.md` for the run command and
//! the committed outputs.
//!
//! Run the slow tests with:
//!
//!   cargo test --features slow-tests --test saem_logit_mu_ref

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::MuTransform;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

mod common;

/// The #918 fixture: `FR1` is logit-normal, `CL`/`V` are lognormal.
/// `LOGIT_FR1` starts at the mirror image of the truth (FR1 = 0.4 vs 0.6) so the
/// M-step has to move it. Mirrors `nonmem_anchor/logit_fraction_saem_fit.ferx`.
const LOGIT_FRACTION_MODEL: &str = r"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta LOGIT_FR1(-0.405465, -10.0, 10.0)
  theta TVKA1(2.0,  0.5,  24.0)
  theta TVKA2(0.2,  0.01,  0.5)

  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.09
  omega ETA_FR1 ~ 0.25

  sigma PROP_ERR ~ 0.08 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  FR1 = 1.0 / (1.0 + exp(-(LOGIT_FR1 + ETA_FR1)))
  FR2 = 1 - FR1
  KA1 = TVKA1
  KA2 = TVKA2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Same fixture with the logit fraction as the **only** random effect: `CL` and
/// `V` have no ETA, so the SAEM closed-form M-step has work to do if and only if
/// the logit form is recognised. This is what makes the eval-saving assertion
/// below discriminating rather than incidental.
const LOGIT_ONLY_MODEL: &str = r"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta LOGIT_FR1(-0.405465, -10.0, 10.0)
  theta TVKA1(2.0,  0.5,  24.0)
  theta TVKA2(0.2,  0.01,  0.5)

  omega ETA_FR1 ~ 0.25

  sigma PROP_ERR ~ 0.08 (sd)

[individual_parameters]
  CL  = TVCL
  V   = TVV
  FR1 = inv_logit(LOGIT_FR1 + ETA_FR1)
  FR2 = 1 - FR1
  KA1 = TVKA1
  KA2 = TVKA2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Data-generating truth for the fraction (see `simulate_logit_fraction_data.py`).
const TRUE_FR1: f64 = 0.6;

/// NONMEM 7.5.1 `METHOD=SAEM` final estimates on the same dataset, from
/// `nonmem_anchor/results/logit_fraction_saem.ext` (the last `-1000000000` row of
/// the SAEM table). `THETA(3)` is the logit-scale typical fraction and
/// `OMEGA(3,3)` its logit-scale IIV variance.
const NONMEM_SAEM_LOGIT_FR1: f64 = 0.421_941; // inv_logit → FR1 = 0.6039
const NONMEM_SAEM_OMEGA_FR1: f64 = 0.182_790;

fn inv_logit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn theta_by_name(result_names: &[String], theta: &[f64], name: &str) -> f64 {
    let idx = result_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("theta `{name}` must exist, have {result_names:?}"));
    theta[idx]
}

/// Parse-level check: the hand-written inv_logit is recorded as a logit
/// mu-reference against `LOGIT_FR1`, and no parameter is reported as
/// non-mu-referenced.
#[test]
fn logit_fraction_is_detected_as_a_mu_reference() {
    let model = parse_full_model(LOGIT_FRACTION_MODEL)
        .expect("fixture must parse")
        .model;

    let fr1 = model
        .mu_refs
        .get("ETA_FR1")
        .expect("ETA_FR1 must be mu-referenced (#918)");
    assert_eq!(fr1.theta_name, "LOGIT_FR1");
    assert_eq!(fr1.transform, MuTransform::Logit);
    assert!(
        !fr1.log_transformed(),
        "a logit mu-ref is not lognormal; omega is variance on the logit scale"
    );

    // The lognormal siblings keep their existing classification.
    assert_eq!(model.mu_refs["ETA_CL"].transform, MuTransform::Log);
    assert_eq!(model.mu_refs["ETA_V"].transform, MuTransform::Log);
}

/// Tier-2: a short SAEM run on the real fixture must take the closed-form
/// M-step for the logit theta. `LOGIT_ONLY_MODEL` has no other random effect,
/// so `saem_mu_ref_m_step_evals_saved > 0` is only possible when the logit
/// mu-ref is both detected *and* judged eligible for the closed form.
#[test]
fn logit_mu_ref_drives_the_saem_closed_form_m_step() {
    let model = parse_full_model(LOGIT_ONLY_MODEL)
        .expect("fixture must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/logit_fraction_oral.csv"), None, None)
        .expect("fixture data must load");

    let opts = FitOptions {
        method: EstimationMethod::Saem,
        // Minimal budget: this asserts which M-step branch runs, not convergence.
        saem_n_exploration: 2,
        saem_n_convergence: 1,
        run_covariance_step: false,
        verbose: false,
        saem_seed: Some(918),
        ..FitOptions::default()
    };

    let result = fit(&model, &pop, &model.default_params, &opts).expect("short SAEM must run");

    let saved = result
        .saem_mu_ref_m_step_evals_saved
        .expect("SAEM + mu_referencing must report the saved-eval count");
    assert!(
        saved > 0,
        "the logit mu-ref is the only random effect, so a non-zero saved-eval count is the \
         signature of the closed-form M-step running for LOGIT_FR1 (#918); got {saved}"
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("packed on the log scale")),
        "a negative lower bound keeps the theta identity-packed; no packing advisory expected, \
         got {:?}",
        result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("individual parameter(s) not mu-referenced")),
        "no parameter should be flagged as non-mu-referenced, got {:?}",
        result.warnings
    );
}

/// `LOGIT_ONLY_MODEL` with **two** logit etas anchored to the same theta — the
/// shape a user writes when two bounded quantities share one typical value
/// (`F1 = inv_logit(LOGIT_FR + ETA_F1)`, `FX = inv_logit(LOGIT_FR + ETA_F2)`).
/// Each eta is a valid mu-reference on its own, but the closed form shifts the
/// packed theta by *one* eta mean and then pins it, so there is no single
/// well-defined update: the codex review of PR #1375 caught the classifier
/// emitting both pairs, which made `run_saem` apply `θ += γ·mean(η_1)` and then
/// `θ += γ·mean(η_2)` in the same iteration.
const SHARED_LOGIT_ANCHOR_MODEL: &str = r"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta LOGIT_FR(-0.405465, -10.0, 10.0)
  theta TVKA1(2.0,  0.5,  24.0)
  theta TVKA2(0.2,  0.01,  0.5)

  omega ETA_FR1 ~ 0.25
  omega ETA_FR2 ~ 0.25

  sigma PROP_ERR ~ 0.08 (sd)

[individual_parameters]
  FR1 = inv_logit(LOGIT_FR + ETA_FR1)
  FR2 = 1 - FR1
  FRX = inv_logit(LOGIT_FR + ETA_FR2)
  CL  = TVCL * FRX
  V   = TVV
  KA1 = TVKA1
  KA2 = TVKA2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Two logit etas on one theta must take the numerical M-step, not a double
/// shift: no closed-form eval saving at all (these two are the model's only
/// random effects) and an advisory naming the shared anchor.
///
/// The control is `logit_mu_ref_drives_the_saem_closed_form_m_step` above: the
/// same fixture with a single eta on the anchor reports `saved > 0`.
#[test]
fn saem_two_logit_etas_on_one_theta_route_to_the_numerical_mstep() {
    let model = parse_full_model(SHARED_LOGIT_ANCHOR_MODEL)
        .expect("fixture must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/logit_fraction_oral.csv"), None, None)
        .expect("fixture data must load");
    let opts = FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 2,
        saem_n_convergence: 1,
        run_covariance_step: false,
        verbose: false,
        saem_seed: Some(918),
        ..FitOptions::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("short SAEM must run");
    assert_eq!(
        result.saem_mu_ref_m_step_evals_saved.unwrap_or(0),
        0,
        "a shared anchor has no closed-form pair, so no theta is pinned and nothing is saved"
    );
    let hit = result
        .warnings
        .iter()
        .find(|w| w.contains("mu-reference anchor of more than one ETA"))
        .unwrap_or_else(|| {
            panic!(
                "expected the shared-anchor advisory, got {:?}",
                result.warnings
            )
        });
    assert!(
        hit.contains("LOGIT_FR"),
        "advisory must name the shared anchor: {hit}"
    );
}

/// `LOGIT_ONLY_MODEL` with the logit-scale theta declared with a **non-negative**
/// lower bound. ferx then packs it as `log θ`, which is not its mu scale, so the
/// closed form cannot apply — the mirror image of the #996 identity-packed
/// lognormal case.
fn log_packed_logit_model() -> &'static str {
    // `LOGIT_FR1` starts on the positive side so the declaration is admissible.
    Box::leak(
        LOGIT_ONLY_MODEL
            .replace(
                "theta LOGIT_FR1(-0.405465, -10.0, 10.0)",
                "theta LOGIT_FR1(0.405465, 0.0, 10.0)",
            )
            .into_boxed_str(),
    )
}

/// A logit mu-ref whose theta is log-packed must fall through to the numerical
/// M-step *and say so*: no closed-form branch runs (nothing saved) and the
/// #918 packing advisory names the theta. The control is the test above, where
/// the same model with a negative lower bound takes the closed form.
#[test]
fn saem_log_packed_logit_theta_routes_to_numerical_mstep_with_advisory() {
    let model = parse_full_model(log_packed_logit_model())
        .expect("fixture must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/logit_fraction_oral.csv"), None, None)
        .expect("fixture data must load");
    let opts = FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 2,
        saem_n_convergence: 1,
        run_covariance_step: false,
        verbose: false,
        saem_seed: Some(918),
        ..FitOptions::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("short SAEM must run");
    assert_eq!(
        result.saem_mu_ref_m_step_evals_saved.unwrap_or(0),
        0,
        "a log-packed logit theta has no closed-form pair, so nothing is saved"
    );
    let hit = result
        .warnings
        .iter()
        .find(|w| w.contains("packed on the log scale"))
        .unwrap_or_else(|| {
            panic!(
                "expected the #918 packing advisory, got {:?}",
                result.warnings
            )
        });
    assert!(
        hit.contains("LOGIT_FR1"),
        "advisory must name the theta: {hit}"
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("individual parameter(s) not mu-referenced")),
        "the parameter is still mu-referenced (detection is independent of packing), got {:?}",
        result.warnings
    );
}

/// Tier-3: a full SAEM fit must recover the logit-normal fraction, and land where
/// NONMEM's `METHOD=SAEM` with explicit `MU_3 = THETA(3)` lands on the same data.
///
/// The band is stated on the natural `(0,1)` scale, where it is interpretable:
/// the fraction must come back within 0.05 of the data-generating 0.6. Before
/// #918 this typical value went through the numeric M-step and did not.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored logit mu-referencing (#918): opt in with --features slow-tests"
)]
fn saem_recovers_the_logit_normal_fraction() {
    // NONMEM 7.5.1 SAEM (NBURN=2000 NITER=1000 ISAMPLE=10 SEED=918) on the same
    // dataset, `nonmem_anchor/results/logit_fraction_saem.ext`. See
    // `nonmem_anchor/README.md` for the run command.
    const NONMEM_LOGIT_FR1: f64 = NONMEM_SAEM_LOGIT_FR1;
    const FR1_TOLERANCE: f64 = 0.05;

    let model = parse_full_model(LOGIT_FRACTION_MODEL)
        .expect("fixture must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/logit_fraction_oral.csv"), None, None)
        .expect("fixture data must load");

    let opts = FitOptions {
        method: EstimationMethod::Saem,
        run_covariance_step: false,
        verbose: false,
        saem_seed: Some(918),
        ..FitOptions::default()
    };

    let result = fit(&model, &pop, &model.default_params, &opts).expect("SAEM fit must converge");

    let logit_fr1 = theta_by_name(&result.theta_names, &result.theta, "LOGIT_FR1");
    let fr1 = inv_logit(logit_fr1);

    assert!(
        (fr1 - TRUE_FR1).abs() < FR1_TOLERANCE,
        "SAEM must recover the logit-normal fraction: got FR1 = {fr1:.4} \
         (LOGIT_FR1 = {logit_fr1:.4}), truth {TRUE_FR1} ± {FR1_TOLERANCE} (#918)"
    );
    assert!(
        (fr1 - inv_logit(NONMEM_LOGIT_FR1)).abs() < FR1_TOLERANCE,
        "SAEM must land where NONMEM's METHOD=SAEM with MU_3 = THETA(3) lands: \
         ferx FR1 = {fr1:.4} vs NONMEM {:.4} (± {FR1_TOLERANCE})",
        inv_logit(NONMEM_LOGIT_FR1)
    );

    // The IIV is on the logit scale; recovering its order of magnitude is what
    // separates "the fraction is estimated" from "the fraction collapsed".
    let eta_idx = result
        .eta_names
        .iter()
        .position(|n| n == "ETA_FR1")
        .expect("ETA_FR1 must be in omega");
    let omega_fr1 = result.omega[(eta_idx, eta_idx)];
    assert!(
        (0.08..0.45).contains(&omega_fr1),
        "omega^2(ETA_FR1) = {omega_fr1:.4} is outside the plausible band around the \
         data-generating 0.25 (NONMEM SAEM reports {NONMEM_SAEM_OMEGA_FR1:.4})"
    );
}
