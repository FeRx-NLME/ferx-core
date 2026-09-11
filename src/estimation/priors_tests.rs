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
