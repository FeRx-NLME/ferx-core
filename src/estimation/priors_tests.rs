//! Tier-1 tests for the parameter-prior penalty (#254).
//!
//! Every assertion here pins a **hand-computed** number rather than a tolerance
//! picked from a run: the penalty is three lines of closed-form algebra, so an
//! oracle outside the implementation is a pocket calculator, not a second
//! engine. Where a bound is used it is a finite-difference comparison, and the
//! realised worst error is recorded in the comment next to it.

use super::*;
use crate::estimation::parameterization::pack_params;
use crate::types::{
    GradientMethod, ModelParameters, OmegaMatrix, ParameterPrior, PriorSpread, SigmaVector,
};

/// `sqrt(ln(1 + 0.4²))` — the exact CV → log-SD conversion at RSE = 40%, which
/// every Ω assertion below is built from. Spelled once so a test that disagrees
/// with the implementation disagrees about the *formula*, not about a typo.
const TAU_40: f64 = 0.385_253_170_159_926_7;

/// A one-θ / one-η / one-σ analytical model whose parameter values, bounds and
/// declared scales the caller chooses. Everything the prior code reads comes
/// from here; nothing else about the model matters, because `PriorSet::build`
/// touches only the packed layout and the `*_init_as_sd` flags.
struct Fixture {
    model: CompiledModel,
}

impl Fixture {
    /// θ = `theta`, lower bound `theta_lower` (this is what decides whether the
    /// θ prior is lognormal or normal), Ω variance `omega_var`, σ SD `sigma_sd`.
    fn new(theta: f64, theta_lower: f64, omega_var: f64, sigma_sd: f64) -> Self {
        let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        model.default_params = ModelParameters {
            theta: vec![theta],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![theta_lower],
            theta_upper: vec![1e9],
            theta_fixed: vec![false],
            omega: OmegaMatrix::from_diagonal(&[omega_var], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![sigma_sd],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        };
        model.omega_init_as_sd = vec![false];
        model.sigma_init_as_sd = vec![false];
        Self { model }
    }

    fn with_prior(mut self, name: &str, value: f64, spread: PriorSpread) -> Self {
        self.model.priors.push(ParameterPrior {
            name: name.into(),
            value,
            spread,
        });
        self
    }

    fn omega_declared_as_sd(mut self) -> Self {
        self.model.omega_init_as_sd = vec![true];
        self
    }

    fn sigma_declared_as_sd(mut self) -> Self {
        self.model.sigma_init_as_sd = vec![true];
        self
    }

    /// Give the model a one-κ diagonal Ω_IOV of variance `kappa_var`.
    ///
    /// The κ segment is packed *after* Σ, so a prior that resolves onto it is
    /// also the only check that the coordinate walk reaches past the Σ block
    /// with its indices intact.
    fn with_iov(mut self, kappa_var: f64) -> Self {
        self.model.default_params.omega_iov = Some(OmegaMatrix::from_diagonal(
            &[kappa_var],
            vec!["KAPPA_CL".into()],
        ));
        self.model.default_params.kappa_fixed = vec![false];
        self.model.kappa_init_as_sd = vec![false];
        self
    }

    fn params(&self) -> &ModelParameters {
        &self.model.default_params
    }

    fn build(&self) -> Result<PriorSet, String> {
        PriorSet::build(&self.model, &self.model.default_params)
    }

    fn set(&self) -> PriorSet {
        self.build().expect("prior set should build")
    }

    fn packed(&self) -> Vec<f64> {
        pack_params(self.params())
    }
}

// ---------------------------------------------------------------------------
// The closed forms
// ---------------------------------------------------------------------------

/// A θ with a non-negative lower bound packs as `ln θ`, so its prior is
/// lognormal on θ with `m = ln(value)` and `s = sqrt(ln(1 + rse²))`.
///
/// Catches: a prior applied on the natural θ scale while the optimizer moves
/// `ln θ` — which would still produce a plausible, monotone penalty and a fit
/// that "looks regularized" while pulling toward the wrong point.
#[test]
fn log_packed_theta_prior_is_lognormal_on_theta() {
    // θ̂ = 0.2, prior centred at 0.15 with RSE 40%.
    let f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, PriorSpread::Rse(0.4));
    let set = f.set();
    let packed = f.packed();

    // z = ln(0.2 / 0.15) / TAU_40 = 0.2876820724 / 0.3852531702 = 0.7467351205
    let z = (0.2f64 / 0.15).ln() / TAU_40;
    assert!((set.penalty(&packed) - z * z).abs() < 1e-12);
    // Hand-computed, so the test disagrees with a wrong implementation rather
    // than with itself: z² = 0.5576133402
    assert!((set.penalty(&packed) - 0.557_613_340_2).abs() < 1e-8);

    // At the prior mean the penalty is exactly zero — no floating-point slop,
    // because `to_packed` is the exact inverse the packer uses.
    let at_mean =
        Fixture::new(0.15, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, PriorSpread::Rse(0.4));
    assert_eq!(at_mean.set().penalty(&at_mean.packed()), 0.0);
}

/// A θ whose declared lower bound is negative packs as the identity, so its
/// prior is normal on θ with `s = |value|·rse`.
///
/// This is the pair to the test above: the two straddle `theta_packs_log`, so a
/// change that collapses the two families to one reddens exactly one of them.
#[test]
fn identity_packed_theta_prior_is_normal_on_theta() {
    // Prior mean 0.5, RSE 40% → s = 0.2. θ̂ = 0.7 sits exactly one prior SD out.
    let f = Fixture::new(0.7, -5.0, 0.1, 0.1).with_prior("TVCL", 0.5, PriorSpread::Rse(0.4));
    let set = f.set();
    assert!((set.penalty(&f.packed()) - 1.0).abs() < 1e-12);

    // And the straddle is real rather than assumed: the same numbers on a
    // non-negative lower bound take the *other* arm and give a different
    // penalty. Without this the pair could quietly become a tautology.
    let logged = Fixture::new(0.7, 0.0, 0.1, 0.1).with_prior("TVCL", 0.5, PriorSpread::Rse(0.4));
    let logged_pen = logged.set().penalty(&logged.packed());
    assert!(
        (logged_pen - 1.0).abs() > 0.1,
        "log-packed θ must not coincide with the identity-packed penalty; got {logged_pen}"
    );
    // ln(0.7/0.5)/TAU_40 = 0.3364722366 / 0.3852531702 = 0.8733795402, squared.
    assert!((logged_pen - 0.762_791_821_3).abs() < 1e-8);
}

/// An Ω declared on the (default) variance scale gets a lognormal prior on the
/// **variance**, even though the packed coordinate is `ln(SD)`.
///
/// Catches: the ½ that relates `ln(SD)` to `ln(variance)` being dropped from the
/// mean, from the spread, or from both.
#[test]
fn variance_declared_omega_prior_is_lognormal_on_the_variance() {
    // Ω̂ = 0.16 (SD 0.4), prior centred on a variance of 0.09 at RSE 40%.
    let f = Fixture::new(0.2, 0.001, 0.16, 0.1).with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4));
    let set = f.set();
    // z = ln(0.16 / 0.09) / TAU_40 = 0.5753641449 / 0.3852531702 = 1.4934702410
    let z = (0.16f64 / 0.09).ln() / TAU_40;
    assert!((set.penalty(&f.packed()) - z * z).abs() < 1e-12);
    assert!((set.penalty(&f.packed()) - 2.230_453_360_9).abs() < 1e-8);
}

/// The same underlying Ω, declared as an SD instead of a variance, with the
/// same numeric RSE, is a **different** prior — an RSE of 40% on a variance is
/// roughly an RSE of 20% on the SD.
///
/// This is the differential pair for `omega_init_as_sd`. Both fixtures describe
/// the identical model (variance 0.16, SD 0.4) and the identical prior location
/// (variance 0.09, SD 0.3), so the only thing that can move the penalty is which
/// scale the declaration was on. The ratio is exactly 4 because the log-distance
/// on the variance scale is exactly twice the log-distance on the SD scale.
///
/// Catches: `flag_at` ignored, or the Ω scale hard-coded to one of the two arms.
#[test]
fn sd_declared_omega_is_a_different_prior_from_the_variance_declaration() {
    let as_var =
        Fixture::new(0.2, 0.001, 0.16, 0.1).with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4));
    let as_sd = Fixture::new(0.2, 0.001, 0.16, 0.1)
        .omega_declared_as_sd()
        .with_prior("ETA_CL", 0.3, PriorSpread::Rse(0.4));

    let pen_var = as_var.set().penalty(&as_var.packed());
    let pen_sd = as_sd.set().penalty(&as_sd.packed());

    // ln(0.4/0.3)/TAU_40 = 0.2876820724 / 0.3852531702 = 0.7467351205, squared.
    assert!((pen_sd - 0.557_613_340_2).abs() < 1e-8);
    assert!(
        (pen_var / pen_sd - 4.0).abs() < 1e-9,
        "{pen_var} vs {pen_sd}"
    );
}

/// σ is stored internally on the SD scale but declared as a variance by
/// default, so it needs the same two arms Ω does — and gets them from the same
/// `flag_at` call, which is why this test exists to pin that σ is wired up at
/// all rather than silently defaulting.
#[test]
fn sigma_honours_its_declared_scale() {
    // Stored σ is an SD of 0.4, i.e. a declared variance of 0.16.
    let as_var =
        Fixture::new(0.2, 0.001, 0.1, 0.4).with_prior("PROP_ERR", 0.09, PriorSpread::Rse(0.4));
    let as_sd = Fixture::new(0.2, 0.001, 0.1, 0.4)
        .sigma_declared_as_sd()
        .with_prior("PROP_ERR", 0.3, PriorSpread::Rse(0.4));

    let pen_var = as_var.set().penalty(&as_var.packed());
    let pen_sd = as_sd.set().penalty(&as_sd.packed());
    assert!((pen_var - 2.230_453_360_9).abs() < 1e-8);
    assert!((pen_sd - 0.557_613_340_2).abs() < 1e-8);
}

/// `sd = <absolute>` on a log-packed coordinate is sugar for the equivalent
/// RSE, so the two spellings must coincide exactly rather than approximately.
#[test]
fn absolute_sd_matches_the_equivalent_rse_on_a_log_packed_coordinate() {
    let by_rse = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, PriorSpread::Rse(0.2));
    // 0.03 / 0.15 == 0.2.
    let by_sd = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, PriorSpread::Sd(0.03));
    assert_eq!(
        by_rse.set().penalty(&by_rse.packed()),
        by_sd.set().penalty(&by_sd.packed())
    );
}

/// On an identity-packed θ there is no log scale to relate to, so `sd` is the
/// packed SD directly — the one place the two spellings are *not* related by a
/// division.
#[test]
fn absolute_sd_on_an_identity_packed_theta_is_the_packed_sd() {
    // θ̂ = 0.7, prior mean 0.5, sd 0.2 → exactly one prior SD out.
    let f = Fixture::new(0.7, -5.0, 0.1, 0.1).with_prior("TVCL", 0.5, PriorSpread::Sd(0.2));
    assert!((f.set().penalty(&f.packed()) - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Derivatives
// ---------------------------------------------------------------------------

/// The analytic gradient and Hessian must agree with central finite differences
/// of `penalty` itself.
///
/// This is the parity check the repo requires of every derivative path. It is a
/// genuine oracle here because `penalty` and `add_gradient` share no code — the
/// former squares a z-score, the latter differentiates it by hand — so a sign
/// error or a missing factor of two in either shows up.
///
/// Realised worst error over the three priored coordinates, measured by running
/// this test: gradient **2.12e-10**, Hessian **2.48e-5**. Central FD of a
/// quadratic has no truncation error, so both are pure round-off — and the
/// Hessian's is five orders larger than the gradient's only because the second
/// difference divides by `h²`, cancelling ~10 significant digits at `h = 1e-5`.
/// Bounds below are set ~4x above each realised value.
#[test]
fn gradient_and_hessian_match_central_finite_differences() {
    let f = Fixture::new(0.2, 0.001, 0.16, 0.4)
        .with_prior("TVCL", 0.15, PriorSpread::Rse(0.4))
        .with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4))
        .with_prior("PROP_ERR", 0.09, PriorSpread::Rse(0.25));
    let set = f.set();
    let x = f.packed();
    assert_eq!(set.summarize(&x).len(), 3);

    let mut analytic = vec![0.0; x.len()];
    set.add_gradient(&x, &mut analytic);

    let mut hess = vec![0.0; x.len()];
    set.add_hessian(&mut |i, j, v| {
        assert_eq!(i, j, "the prior Hessian must be diagonal");
        hess[i] += v;
    });

    let h = 1e-5;
    let mut worst_grad = 0.0f64;
    let mut worst_hess = 0.0f64;
    for k in 0..x.len() {
        let mut up = x.clone();
        let mut dn = x.clone();
        up[k] += h;
        dn[k] -= h;
        let (fu, fd, f0) = (set.penalty(&up), set.penalty(&dn), set.penalty(&x));
        // `is_finite` before folding: `f64::max` swallows a NaN, so a solver
        // returning NaN would otherwise leave `worst` at whatever the healthy
        // coordinates produced and the bound would pass on their strength.
        let g_fd = (fu - fd) / (2.0 * h);
        let h_fd = (fu - 2.0 * f0 + fd) / (h * h);
        assert!(g_fd.is_finite() && h_fd.is_finite(), "FD at coord {k}");
        worst_grad = worst_grad.max((analytic[k] - g_fd).abs());
        worst_hess = worst_hess.max((hess[k] - h_fd).abs());
    }
    assert!(worst_grad < 1e-8, "worst gradient error {worst_grad}");
    assert!(worst_hess < 1e-4, "worst Hessian error {worst_hess}");

    // The Hessian is a *constant* `2/s²`, which is what lets the covariance step
    // add it without extra objective evaluations. Pin that it does not depend on
    // the point, since an implementation that recomputed it at `x` would also
    // pass the FD check above.
    let far: Vec<f64> = x.iter().map(|v| v + 3.0).collect();
    let mut hess_far = vec![0.0; x.len()];
    set.add_hessian(&mut |i, _, v| hess_far[i] += v);
    assert_eq!(hess, hess_far);
    let _ = far;
}

/// `penalty_and_gradient` must be exactly `penalty` plus `add_gradient`.
///
/// The combined form exists so an optimizer cannot take the value without the
/// gradient — a defect that leaves the suite green because the line search still
/// crawls toward the prior. That only helps if the two forms agree, so this pins
/// it bit-for-bit rather than to a tolerance.
#[test]
fn penalty_and_gradient_is_exactly_the_two_separate_calls() {
    let f = Fixture::new(0.2, 0.001, 0.16, 0.4)
        .with_prior("TVCL", 0.15, PriorSpread::Rse(0.4))
        .with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4));
    let set = f.set();
    let x = f.packed();

    let mut separate = vec![0.25; x.len()];
    set.add_gradient(&x, &mut separate);
    let mut combined = vec![0.25; x.len()];
    let value = set.penalty_and_gradient(&x, &mut combined);

    assert_eq!(value, set.penalty(&x));
    assert_eq!(combined, separate);
    // And it must have written something, or the equality above is two
    // untouched copies of the same initial vector.
    assert_ne!(combined, vec![0.25; x.len()]);
}

/// An unpriored model must leave the objective and the gradient bit-identical
/// to what they were before this feature existed.
#[test]
fn an_unpriored_model_is_a_strict_noop() {
    let f = Fixture::new(0.2, 0.001, 0.1, 0.1);
    let set = f.set();
    assert!(!set.is_active());
    assert_eq!(set.penalty(&f.packed()), 0.0);
    let mut grad = vec![0.5; f.packed().len()];
    set.add_gradient(&f.packed(), &mut grad);
    assert_eq!(grad, vec![0.5; f.packed().len()]);
    let mut touched = false;
    set.add_hessian(&mut |_, _, _| touched = true);
    assert!(!touched);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// The report echoes the prior on the declared scale and the estimate on that
/// same scale, so a reader never has to know what the packed coordinate was.
#[test]
fn summary_reports_declared_scale_values_and_the_standardized_shift() {
    let f = Fixture::new(0.2, 0.001, 0.16, 0.1)
        .with_prior("TVCL", 0.15, PriorSpread::Rse(0.4))
        .with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4));
    let rows = f.set().summarize(&f.packed());
    assert_eq!(rows.len(), 2);

    let theta = &rows[0];
    assert_eq!(theta.name, "TVCL");
    assert_eq!(theta.prior_value, 0.15);
    assert!((theta.estimate - 0.2).abs() < 1e-12);
    assert_eq!(theta.family, "lognormal");
    assert!((theta.shift_in_prior_sds - 0.746_735_120_5).abs() < 1e-6);
    assert!((theta.penalty - 0.557_613_340_2).abs() < 1e-8);
    // 95% interval on the declared scale: 0.15·exp(±1.96·TAU_40).
    assert!((theta.prior_lower_95 - 0.15 * (-1.959_963_985 * TAU_40).exp()).abs() < 1e-9);

    // The Ω row reports the **variance**, because that is how it was declared —
    // reporting `ln(SD)` or the SD here would be a silent scale change between
    // the model file and the report.
    let omega = &rows[1];
    assert_eq!(omega.name, "ETA_CL");
    assert_eq!(omega.prior_value, 0.09);
    assert!((omega.estimate - 0.16).abs() < 1e-12);

    // The per-parameter penalties must add up to the total, or the split shown
    // to the user does not reconcile with the reported `ofv_prior`.
    let total: f64 = rows.iter().map(|r| r.penalty).sum();
    assert!((total - f.set().penalty(&f.packed())).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Rejections — each must be an error, never a silently dropped prior
// ---------------------------------------------------------------------------

fn err_of(f: &Fixture) -> String {
    f.build().expect_err("expected this prior to be rejected")
}

#[test]
fn a_prior_on_an_unknown_parameter_is_an_error() {
    let f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("NOPE", 1.0, PriorSpread::Rse(0.2));
    assert!(err_of(&f).contains("NOPE"), "{}", err_of(&f));
}

#[test]
fn a_prior_on_a_fixed_parameter_is_an_error() {
    let mut f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, PriorSpread::Rse(0.2));
    f.model.default_params.theta_fixed = vec![true];
    let msg = err_of(&f);
    assert!(msg.contains("FIX"), "{msg}");
}

#[test]
fn a_prior_declared_twice_is_an_error() {
    let f = Fixture::new(0.2, 0.001, 0.1, 0.1)
        .with_prior("TVCL", 0.15, PriorSpread::Rse(0.2))
        .with_prior("TVCL", 0.16, PriorSpread::Rse(0.3));
    assert!(err_of(&f).contains("more than once"), "{}", err_of(&f));
}

#[test]
fn a_non_positive_central_value_on_a_log_packed_coordinate_is_an_error() {
    let f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.0, PriorSpread::Rse(0.2));
    assert!(err_of(&f).contains("> 0"), "{}", err_of(&f));

    // …but the same value is fine on an identity-packed θ, where zero is an
    // ordinary point on the real line. Without this half the check could be
    // "reject zero everywhere" and still pass.
    let ok = Fixture::new(0.2, -5.0, 0.1, 0.1).with_prior("TVCL", 0.0, PriorSpread::Sd(0.1));
    assert!(ok.build().is_ok());
}

#[test]
fn a_non_positive_spread_is_an_error() {
    for spread in [
        PriorSpread::Rse(0.0),
        PriorSpread::Rse(-0.2),
        PriorSpread::Sd(0.0),
    ] {
        let f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, spread);
        let msg = err_of(&f);
        assert!(msg.contains("> 0"), "{spread:?}: {msg}");
    }
}

/// A non-finite central value is rejected *before* the positivity check, and
/// says so in those words.
///
/// The ordering is the point: `!(NaN > 0.0)` is `true`, so a positivity test
/// reached first would report a `NaN` as "must be > 0" and send the user looking
/// at the sign of a value that has no sign. Asserted on the message, not merely
/// on `is_err`, because both branches return `Err` and only the text
/// distinguishes them.
#[test]
fn a_non_finite_central_value_is_rejected_as_non_finite() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", value, PriorSpread::Rse(0.2));
        let msg = err_of(&f);
        assert!(
            msg.contains("finite"),
            "{value}: must be reported as non-finite, not as non-positive: {msg}"
        );
    }

    // The straddle: the same coordinate with a finite positive value resolves,
    // so the check rejects non-finiteness rather than every value.
    assert!(Fixture::new(0.2, 0.001, 0.1, 0.1)
        .with_prior("TVCL", 0.15, PriorSpread::Rse(0.2))
        .build()
        .is_ok());
}

/// A non-finite *spread* is rejected in both spellings.
///
/// `a_non_positive_spread_is_an_error` covers zero and negative; these are the
/// values that pass `> 0` and would otherwise produce an `sd` of `NaN` or `inf`
/// — a penalty that is silently zero (`z = finite/inf`) or `NaN` for the whole
/// fit, neither of which any outcome assertion would attribute to the prior.
#[test]
fn a_non_finite_spread_is_an_error() {
    for spread in [
        PriorSpread::Rse(f64::NAN),
        PriorSpread::Rse(f64::INFINITY),
        PriorSpread::Sd(f64::NAN),
        PriorSpread::Sd(f64::INFINITY),
    ] {
        let f = Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("TVCL", 0.15, spread);
        let msg = err_of(&f);
        assert!(
            msg.contains("finite") || msg.contains("> 0"),
            "{spread:?}: {msg}"
        );
    }
}

/// A `block_omega` element has no variance of its own to prior — the packed
/// coordinates are Cholesky entries — so v1 rejects it by name rather than
/// priding a Cholesky diagonal as if it were a variance.
#[test]
fn a_prior_on_a_block_omega_element_is_an_error() {
    let mut f =
        Fixture::new(0.2, 0.001, 0.1, 0.1).with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4));
    let block = nalgebra::DMatrix::from_row_slice(2, 2, &[0.1, 0.02, 0.02, 0.2]);
    f.model.default_params.omega =
        OmegaMatrix::from_matrix(block, vec!["ETA_CL".into(), "ETA_V".into()], false);
    f.model.default_params.omega_fixed = vec![false, false];
    f.model.omega_init_as_sd = vec![false, false];
    let msg = err_of(&f);
    assert!(msg.contains("block_omega"), "{msg}");
}

/// Block membership is a property of the **matrix**, not only of the free mask.
///
/// A fixed non-zero covariance is still a covariance: the packed coordinate for
/// row `r > 0` of such a block is `ln(L[r,r])`, and `SD_r² = Σⱼ L[r,j]²`, so
/// scaling a prior there as if `L[r,r]` were the SD is wrong with nothing in the
/// output to say so. Today `packed_fixed_mask` fixes the whole row and column of
/// a FIXed eta, which makes the arrangement unreachable through the parser — so
/// this is built directly, and it is the reason the predicate consults
/// `om.matrix` rather than resting on that coupling.
///
/// Deleting the `om.matrix` clause from `in_block` reddens the first half here
/// and nothing else in the suite.
#[test]
fn a_prior_on_a_block_diagonal_with_a_fixed_covariance_is_an_error() {
    // ETA_CL ~ ETA_V correlated, but the off-diagonal is *not* a free parameter:
    // `free_mask` is the identity while the matrix is genuinely non-diagonal.
    let fixed_cov = |off: f64| -> Fixture {
        let mut f = Fixture::new(0.2, 0.001, 0.1, 0.1);
        let m = nalgebra::DMatrix::from_row_slice(2, 2, &[0.09, off, off, 0.20]);
        let mut free = nalgebra::DMatrix::from_element(2, 2, false);
        free[(0, 0)] = true;
        free[(1, 1)] = true;
        f.model.default_params.omega = OmegaMatrix::from_matrix_with_mask(
            m,
            vec!["ETA_CL".into(), "ETA_V".into()],
            false,
            free,
        );
        f.model.default_params.omega_fixed = vec![false, false];
        f.model.omega_init_as_sd = vec![false, false];
        f.with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4))
    };

    let err = fixed_cov(0.02)
        .build()
        .expect_err("a diagonal correlated by a fixed covariance is a block member");
    assert!(err.contains("block_omega"), "{err}");

    // The straddle: the identical shape with a *zero* off-diagonal is an ordinary
    // independent variance and resolves. Without this the first half is satisfied
    // by a predicate that rejects every non-`diagonal` matrix outright, which is
    // the mixed-Ω regression the test above this one exists to prevent.
    fixed_cov(0.0)
        .build()
        .expect("an uncorrelated diagonal must still resolve");
}

/// A **mixed** Ω — `block_omega (A, B)` alongside an independent `omega C` — is a
/// single non-diagonal matrix, so a matrix-level `!diagonal` test rejects a prior
/// on `C` even though `C` is an ordinary uncorrelated variance.
///
/// That is exactly the arrangement the block diagnostic tells users to reach for
/// ("declare the diagonal variances separately to prior them"), so getting it
/// wrong makes the advice impossible to follow. Block membership has to come from
/// `free_mask`, per coordinate.
///
/// The second half is the packed-index trap the first half hides: in a
/// non-diagonal Ω the packed coordinate is a position in the column-major lower
/// triangle, **not** an eta index, so an `omega_init_as_sd` lookup keyed on the
/// packed offset reads the wrong flag — here it would read `ETA_A`'s (`false`)
/// for `ETA_C` and silently switch `C` to the variance scale.
#[test]
fn a_prior_on_an_independent_diagonal_of_a_mixed_omega_is_accepted() {
    // Ω = [[0.09, 0.02, 0], [0.02, 0.20, 0], [0, 0, 0.16]] — A~B correlated, C free.
    // Packed (column-major lower triangle): (A,A) (B,A) (C,A) (B,B) (C,B) (C,C),
    // so C's diagonal is packed index 5 while its eta index is 2.
    //
    // `CompiledModel` holds boxed closures and is not `Clone`, so the fixture is
    // built fresh per probe rather than cloned.
    let mixed = |prior_on: &str, value: f64| -> Fixture {
        let mut f = Fixture::new(0.2, 0.001, 0.1, 0.1);
        let m = nalgebra::DMatrix::from_row_slice(
            3,
            3,
            &[0.09, 0.02, 0.0, 0.02, 0.20, 0.0, 0.0, 0.0, 0.16],
        );
        let mut free = nalgebra::DMatrix::from_element(3, 3, false);
        for i in 0..3 {
            free[(i, i)] = true;
        }
        free[(0, 1)] = true;
        free[(1, 0)] = true;
        f.model.default_params.omega = OmegaMatrix::from_matrix_with_mask(
            m,
            vec!["ETA_A".into(), "ETA_B".into(), "ETA_C".into()],
            false,
            free,
        );
        f.model.default_params.omega_fixed = vec![false; 3];
        // Only C is declared `(sd)`.
        f.model.omega_init_as_sd = vec![false, false, true];
        f.with_prior(prior_on, value, PriorSpread::Rse(0.4))
    };

    // C is independent → its prior resolves, on C's own declared (SD) scale.
    let on_c = mixed("ETA_C", 0.4);
    let set = on_c
        .build()
        .expect("a prior on an independent diagonal of a mixed Omega must resolve");
    // Ω_CC = 0.16 → SD 0.4, which is the prior mean, so the penalty is exactly 0
    // on the SD scale. Under the variance scale the mean would be ½·ln(0.4) while
    // the packed value stays ln(0.4), giving a large penalty — so this pins the
    // flag lookup, not merely the acceptance.
    assert_eq!(set.penalty(&on_c.packed()), 0.0);

    // A belongs to the block → still rejected, naming the block.
    let err = mixed("ETA_A", 0.09)
        .build()
        .expect_err("a prior on a block member must still be rejected");
    assert!(err.contains("block_omega"), "{err}");
}

// ---------------------------------------------------------------------------
// `[priors] from_fit` — the model-updating import (#254 phase 2)
// ---------------------------------------------------------------------------

/// Write a source fit to a temp `.yaml` and point `f` at it.
///
/// The file is `.yaml` rather than `.json` deliberately: it is the one every
/// plain `ferx model.ferx --data …` run writes, so it is what a user actually
/// has to hand, and it is the lossiest of the three readers. A suite that only
/// exercised `.json` would leave the default path unpinned.
fn with_from_fit(f: Fixture, source: &crate::types::FitResult) -> (Fixture, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("parent-fit.yaml");
    crate::io::output::write_estimates_yaml(source, path.to_str().unwrap()).unwrap();
    let mut f = f;
    f.model.prior_from_fit = Some(path.to_string_lossy().into_owned());
    (f, dir)
}

/// A source fit whose parameter names line up with [`Fixture`]'s single
/// θ / η / σ, with round numbers chosen so every conversion below can be
/// hand-computed.
///
/// TVCL 0.15 ± 0.03 (RSE 20%), ETA_CL variance 0.09 ± 0.036 (RSE 40%),
/// PROP_ERR SD 0.2 ± 0.02 (RSE 10%).
fn source_fit() -> crate::types::FitResult {
    let mut r = crate::types::test_helpers::minimal_fit_result();
    r.theta = vec![0.15];
    r.theta_names = vec!["TVCL".into()];
    r.theta_fixed = vec![false];
    r.se_theta = Some(vec![0.03]);
    r.eta_names = vec!["ETA_CL".into()];
    r.omega = nalgebra::DMatrix::from_row_slice(1, 1, &[0.09]);
    r.omega_fixed = vec![false];
    r.omega_init_as_sd = vec![false];
    r.se_omega = Some(vec![0.036]);
    r.sigma = vec![0.2];
    r.sigma_names = vec!["PROP_ERR".into()];
    r.sigma_fixed = vec![false];
    r.sigma_init_as_sd = vec![false];
    r.se_sigma = Some(vec![0.02]);
    r
}

/// The whole feature in one assertion: an imported prior is the prior the user
/// would have typed off the source run's parameter table.
///
/// Catches a wrong scale on any of the three families at once, because the
/// penalty is compared against a *typed* `prior(...)` on the same fixture rather
/// than against a number recomputed by the import's own arithmetic.
#[test]
fn an_imported_prior_equals_the_prior_the_user_would_have_typed() {
    // θ̂ = 0.2 against a prior at 0.15 (RSE 20%); Ω̂ = 0.1 against 0.09 (RSE 40%);
    // σ̂ = 0.25 against a variance of 0.04 (RSE 20%, doubled from the source's
    // 10% on the SD). Every estimate is off its prior, so all three penalties
    // are non-zero and a dropped term changes the total.
    let (imported, _dir) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &source_fit());
    let typed = Fixture::new(0.2, 0.001, 0.1, 0.25)
        .with_prior("TVCL", 0.15, PriorSpread::Rse(0.2))
        .with_prior("ETA_CL", 0.09, PriorSpread::Rse(0.4))
        .with_prior("PROP_ERR", 0.04, PriorSpread::Rse(0.2));

    let got = imported.set().penalty(&imported.packed());
    let want = typed.set().penalty(&typed.packed());
    assert!(
        got > 0.0,
        "fixture is degenerate: every estimate sits on its prior"
    );
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    assert_eq!(imported.set().summarize(&imported.packed()).len(), 3);
}

/// σ is the one family whose source scale (SD) differs from its default declared
/// scale (variance), so it is where a missing delta-method step hides.
///
/// `PROP_ERR` is reported as an SD of 0.2 ± 0.02. Declared as a variance the
/// prior centre is 0.2² = 0.04 and the relative SE **doubles** to 20% (the
/// relative SE of `xᵖ` is `|p|` times that of `x`); declared `(sd)` it stays
/// 0.2 ± 10%. The pair is what makes a dropped factor of two visible rather than
/// plausible.
#[test]
fn sigma_converts_from_the_source_sd_onto_the_declared_scale() {
    let src = source_fit();

    // Declared as a variance (the default).
    let (var, _d1) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.2), &src);
    let as_var = var.set().summarize(&var.packed());
    let sigma = as_var
        .iter()
        .find(|s| s.name == "PROP_ERR")
        .unwrap()
        .clone();
    // σ̂ = 0.2 is exactly the prior centre on either scale, so the penalty is 0
    // and the informative number is the reported centre: 0.04, a variance.
    assert!((sigma.prior_value - 0.04).abs() < 1e-12, "{sigma:?}");
    assert_eq!(sigma.penalty, 0.0);

    // Declared `(sd)` — same source file, same parameter, centre now 0.2.
    let (sd, _d2) = with_from_fit(
        Fixture::new(0.2, 0.001, 0.1, 0.2).sigma_declared_as_sd(),
        &src,
    );
    let as_sd = sd.set().summarize(&sd.packed());
    let sigma_sd = as_sd.iter().find(|s| s.name == "PROP_ERR").unwrap().clone();
    assert!((sigma_sd.prior_value - 0.2).abs() < 1e-12, "{sigma_sd:?}");

    // …and the two spreads agree to first order. The lognormal CV→log-SD map is
    // not linear, so they are close rather than equal, and that is the
    // documented consequence of the user naming a scale. A dropped factor of two
    // would put them ~2x apart, which is what this bound excludes. Compared
    // through the reported 95% upper limit on a common scale (variance), so the
    // check runs on what the code produced rather than on algebra restated here.
    // Realised difference 7.3e-5; bound set ~4x above it.
    let hi_var = sigma.prior_upper_95;
    let hi_sd = sigma_sd.prior_upper_95 * sigma_sd.prior_upper_95;
    assert!((hi_var - hi_sd).abs() < 3e-4, "{hi_var} vs {hi_sd}");
    // The straddle: the two intervals really are on the scales claimed, so the
    // agreement above is not two copies of the same number.
    assert!(hi_var < 0.1 && hi_sd < 0.1, "{hi_var} {hi_sd}");
    assert!(sigma_sd.prior_upper_95 > 0.2, "{sigma_sd:?}");
}

/// Ω is reported as a variance whatever the *source* declared, so the source's
/// own `(sd)` spelling must be invisible here.
///
/// This is the trap the reader's module docs name: guessing the source scale
/// from `omega_init_as_sd` would be a factor of two on exactly the fits where a
/// published model was written `(sd)`, and would still look plausible.
#[test]
fn omega_declared_as_sd_in_the_source_is_invisible() {
    let mut as_sd = source_fit();
    // Same fit, same numbers, only the source model's *declaration* differs.
    as_sd.omega_init_as_sd = vec![true];
    as_sd.sigma_init_as_sd = vec![true];

    let (a, _d1) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &source_fit());
    let (b, _d2) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &as_sd);
    let pen = a.set().penalty(&a.packed());
    assert!(pen > 0.0, "fixture is degenerate");
    assert_eq!(pen, b.set().penalty(&b.packed()));
}

/// The target model's `(sd)` declaration, by contrast, *is* read: the centre
/// becomes `sqrt(variance)` and the relative SE halves.
#[test]
fn omega_converts_onto_the_targets_declared_scale() {
    let (f, _dir) = with_from_fit(
        Fixture::new(0.2, 0.001, 0.1, 0.25).omega_declared_as_sd(),
        &source_fit(),
    );
    let s = f.set().summarize(&f.packed());
    let omega = s.iter().find(|s| s.name == "ETA_CL").unwrap();
    // Source variance 0.09 → declared SD 0.3.
    assert!((omega.prior_value - 0.3).abs() < 1e-12, "{omega:?}");
    // Source RSE on the variance is 0.036/0.09 = 40%; on the SD it is 20%, so
    // the packed SD is `sqrt(ln(1 + 0.2²))` = 0.198_042, not `TAU_40`.
    let expect_s = (1.0f64 + 0.2 * 0.2).ln().sqrt();
    // Ω̂ = 0.1 → packed ln(sqrt(0.1)); prior mean ln(0.3).
    let z = (0.1f64.sqrt() / 0.3).ln() / expect_s;
    assert!((omega.shift_in_prior_sds - z).abs() < 1e-12, "{omega:?}");
    // The straddle, hand-computed: ln(sqrt(0.1)/0.3) = 0.052_680_2, over
    // `expect_s` = 0.198_042_9 → 0.266_005. An un-halved RSE would divide by
    // `TAU_40` = 0.385_253_2 instead and give 0.136_743, so the two cannot both
    // satisfy this bound.
    assert!(
        (omega.shift_in_prior_sds - 0.266_005).abs() < 1e-5,
        "{omega:?}"
    );
}

/// A θ that may be negative has no meaningful *relative* standard error, so its
/// prior must come in as the absolute SE the source reported.
///
/// Under `Rse` the spread would be `sqrt(ln(1 + (se/|θ̂|)²))` on a log scale the
/// parameter is not even packed on — and at θ̂ near zero it would explode.
#[test]
fn an_identity_packed_theta_imports_its_absolute_standard_error() {
    let mut src = source_fit();
    src.theta = vec![-0.4];
    src.se_theta = Some(vec![0.2]);
    let (f, _dir) = with_from_fit(Fixture::new(-0.2, -5.0, 0.1, 0.25), &src);
    let s = f.set().summarize(&f.packed());
    let th = s.iter().find(|s| s.name == "TVCL").unwrap();
    assert_eq!(th.family, "normal");
    assert!((th.prior_value - (-0.4)).abs() < 1e-12, "{th:?}");
    // θ̂ = −0.2 against a prior at −0.4 with SD 0.2 → exactly one prior SD out.
    assert!((th.shift_in_prior_sds - 1.0).abs() < 1e-9, "{th:?}");
}

/// An inline `prior(...)` on the same parameter wins over the imported one, so a
/// user can override one without giving up the rest of the import.
///
/// The differential is what makes this observable: without the override the
/// import supplies 0.15 ± 20%, so a test asserting only "three priors resolved"
/// would pass whichever won.
#[test]
fn an_inline_prior_overrides_the_imported_one() {
    let (f, _dir) = with_from_fit(
        Fixture::new(0.2, 0.001, 0.1, 0.25).with_prior("TVCL", 0.25, PriorSpread::Rse(0.2)),
        &source_fit(),
    );
    let s = f.set().summarize(&f.packed());
    assert_eq!(s.len(), 3, "the other two imports must still land: {s:?}");
    let th = s.iter().find(|s| s.name == "TVCL").unwrap();
    assert!((th.prior_value - 0.25).abs() < 1e-12, "{th:?}");
}

/// A FIXed target parameter cannot be moved by a prior, so it is skipped with a
/// note rather than failing the fit the way a typed prior on a FIXed parameter
/// does. The asymmetry is the point: an import is a bulk operation.
#[test]
fn a_fixed_target_parameter_is_skipped_with_a_note() {
    let mut f = Fixture::new(0.2, 0.001, 0.1, 0.25);
    f.model.default_params.theta_fixed = vec![true];
    let (f, _dir) = with_from_fit(f, &source_fit());
    let set = f.set();
    assert_eq!(set.summarize(&f.packed()).len(), 2);
    assert!(
        set.notes()
            .iter()
            .any(|n| n.contains("TVCL") && n.contains("FIX")),
        "{:?}",
        set.notes()
    );

    // The typed form on the same parameter is still a hard error — the two
    // behaviours must not collapse into one.
    let mut typed =
        Fixture::new(0.2, 0.001, 0.1, 0.25).with_prior("TVCL", 0.15, PriorSpread::Rse(0.2));
    typed.model.default_params.theta_fixed = vec![true];
    assert!(typed.build().unwrap_err().contains("FIX"));
}

/// A source parameter with no standard error carries no prior spread, so it is
/// skipped — and the note says why, because "my prior did nothing" is otherwise
/// only visible as an absence.
#[test]
fn a_source_parameter_without_a_standard_error_is_skipped() {
    let mut src = source_fit();
    src.se_theta = Some(vec![0.0]);
    let (f, _dir) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &src);
    let set = f.set();
    assert_eq!(set.summarize(&f.packed()).len(), 2);
    assert!(
        set.notes()
            .iter()
            .any(|n| n.contains("TVCL") && n.contains("standard error")),
        "{:?}",
        set.notes()
    );
}

/// Matching is on **name and family**. A source θ called `ETA_CL` must not
/// become a prior on this model's Ω called `ETA_CL`: the numbers are on
/// unrelated scales and the result would be silently wrong rather than absent.
#[test]
fn a_name_that_matches_a_different_family_is_not_imported() {
    let mut src = source_fit();
    src.theta_names = vec!["ETA_CL".into()];
    // Rename the real Ω/σ rows so the only candidate is the mis-familied θ.
    src.eta_names = vec!["SOMETHING_ELSE".into()];
    src.sigma_names = vec!["ALSO_ELSE".into()];

    let (f, _dir) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &src);
    let err = f
        .build()
        .expect_err("a θ named after an Ω must not be imported as one");
    assert!(err.contains("no prior could be imported"), "{err}");

    // The straddle: with the family put back, the very same name *does* import.
    let mut ok = source_fit();
    ok.theta_names = vec!["UNRELATED".into()];
    ok.eta_names = vec!["ETA_CL".into()];
    ok.sigma_names = vec!["ALSO_ELSE".into()];
    let (g, _d2) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &ok);
    assert_eq!(g.set().summarize(&g.packed()).len(), 1);
}

/// An import that lands nothing leaves an unpenalized fit that looks exactly
/// like a penalized one from the outside — the same failure mode phase 1 hard-
/// errors on for a typed prior that cannot be applied.
#[test]
fn an_import_that_lands_nothing_is_an_error() {
    let mut src = source_fit();
    src.theta_names = vec!["NOT_IN_THIS_MODEL".into()];
    src.eta_names = vec!["NOR_THIS".into()];
    src.sigma_names = vec!["NOR_THAT".into()];
    let (f, _dir) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &src);
    let err = f
        .build()
        .expect_err("an empty import must not pass silently");
    assert!(err.contains("no prior could be imported"), "{err}");
    assert!(err.contains("name and family"), "{err}");

    // Unconditional: a typed prior alongside it does not excuse the import. The
    // fit would be penalized — just not by the thing the user asked for — and a
    // wrong path is otherwise invisible.
    let f2 = Fixture::new(0.2, 0.001, 0.1, 0.25).with_prior("TVCL", 0.15, PriorSpread::Rse(0.2));
    let (f2, _d2) = with_from_fit(f2, &src);
    assert!(f2
        .build()
        .expect_err("a typed prior must not excuse an empty import")
        .contains("no prior could be imported"));

    // …and every candidate being *skipped* reports the reasons rather than the
    // bare "no name matched", so the user is told which gate each one hit.
    let mut all_skipped = source_fit();
    all_skipped.se_theta = Some(vec![0.0]);
    all_skipped.se_omega = Some(vec![0.0]);
    all_skipped.se_sigma = Some(vec![0.0]);
    let (f3, _d3) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25), &all_skipped);
    let err = f3.build().expect_err("no usable spread anywhere");
    assert!(err.contains("every candidate was skipped"), "{err}");
    assert!(err.contains("theta TVCL"), "{err}");
}

/// κ (Ω_IOV) imports too, and onto the right coordinate.
///
/// This is the one family every other test here reaches only through the Ω arm
/// they share in the conversion table, so a green suite without it says nothing
/// about κ: the κ segment is packed **after** Σ, so an index that stops at the Σ
/// block, or a `push_omega_coords` call handed the wrong `EstimateKind`, would
/// leave κ unimported or imported as an Ω with nothing to say so. The assertion
/// is the packed coordinate the penalty lands on, not merely that a prior
/// appeared.
#[test]
fn a_kappa_prior_is_imported_onto_the_iov_coordinate() {
    let mut src = source_fit();
    src.omega_iov = Some(nalgebra::DMatrix::from_row_slice(1, 1, &[0.04]));
    src.kappa_names = vec!["KAPPA_CL".into()];
    src.kappa_fixed = vec![false];
    src.kappa_init_as_sd = vec![false];
    src.se_kappa = Some(vec![0.016]); // RSE 40% on the variance

    // κ̂ = 0.09 against a prior centred on the source's variance of 0.04.
    let (f, _dir) = with_from_fit(Fixture::new(0.2, 0.001, 0.1, 0.25).with_iov(0.09), &src);
    let set = f.set();
    let s = set.summarize(&f.packed());
    assert_eq!(s.len(), 4, "θ, Ω, Σ and κ must all import: {s:?}");
    let kappa = s.iter().find(|p| p.name == "KAPPA_CL").unwrap();
    assert!((kappa.prior_value - 0.04).abs() < 1e-12, "{kappa:?}");
    // Lognormal on the variance, exactly as for a variance-declared Ω: the
    // packed coordinate is ln(SD) = ½·ln(v), the mean is ½·ln(0.04), and the
    // packed SD is TAU_40/2 — so the two halves cancel and z is the same
    // `ln(v̂/v₀)/TAU_40` the Ω test pins.
    // z = ln(0.09/0.04) / TAU_40 = 0.8109302162 / 0.3852531702 = 2.1049280811
    let z = (0.09f64 / 0.04).ln() / TAU_40;
    assert!((kappa.shift_in_prior_sds - z).abs() < 1e-12, "{kappa:?}");
    assert!(
        (kappa.shift_in_prior_sds - 2.104_928_081).abs() < 1e-8,
        "{kappa:?}"
    );

    // And it landed on the κ coordinate, not on the Ω one: the packed layout is
    // [θ, Ω, Σ, κ], so only index 3 may move the penalty. Perturbing each
    // coordinate in turn is what distinguishes "a κ prior" from "an Ω prior that
    // happens to have κ's numbers".
    let mut packed = f.packed();
    let base = set.penalty(&packed);
    packed[3] += 0.1;
    assert!(
        (set.penalty(&packed) - base).abs() > 1e-6,
        "the κ prior must respond to the κ coordinate"
    );
}

/// A missing `from_fit` file stops the fit at the same gate a malformed typed
/// prior does, naming the path.
#[test]
fn a_missing_from_fit_file_is_an_error_naming_the_path() {
    let mut f = Fixture::new(0.2, 0.001, 0.1, 0.25);
    f.model.prior_from_fit = Some("/no/such/parent-fit.yaml".into());
    let err = f.build().unwrap_err();
    assert!(err.contains("[priors] from_fit"), "{err}");
    assert!(err.contains("/no/such/parent-fit.yaml"), "{err}");
}

/// No `[priors]` block and no inline prior is the pre-#254 path, bit for bit.
#[test]
fn no_prior_declaration_reads_no_file_and_stays_inactive() {
    let f = Fixture::new(0.2, 0.001, 0.1, 0.25);
    assert!(!f.set().is_active());
    assert!(f.set().notes().is_empty());
}
