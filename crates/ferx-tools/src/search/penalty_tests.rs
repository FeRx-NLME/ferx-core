//! Hand-computed penalized fitness (#1185): one fixture per penalty term,
//! the score checked against the arithmetic written out beside it, and the
//! two schedule-level guarantees — a partial `[rank.penalties]` table
//! overlays the defaults, and a schedule is validated at load.

use ferx_core::{BicInputs, CovarianceStatus, FitResult};
use nalgebra::DMatrix;

use super::*;
use crate::search::test_support::converged_fit;
use crate::search::Criterion;

/// A converged fit at OFV 500 with a clean covariance step: 3 θ, 2 Ω, 1 σ,
/// a 2×2 covariance matrix with |r| = 0.5 and condition number 10. Every
/// charge that can apply is off, so the score is the parameter charges
/// alone and each test below flips exactly one thing.
fn clean() -> FitResult {
    let mut fit = converged_fit(500.0);
    fit.n_parameters = 6;
    fit.bic_inputs = BicInputs {
        n_obs: 100,
        theta_random: 2,
        theta_fixed: 1,
        omega: 2,
        kappa: 0,
        sigma: 1,
        sigma_random: false,
    };
    fit.covariance_status = CovarianceStatus::Computed;
    // Read as stored, packed scale and all, when the layout is unknown —
    // which is what keeps this fixture's correlation exactly the number
    // written here.
    fit.omega_is_diagonal = None;
    fit.covariance_matrix = Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]));
    fit.cov_condition_number = Some(10.0);
    fit
}

const PARAMETER_CHARGE: f64 = 3.0 * 10.0 + 2.0 * 10.0 + 1.0 * 10.0;

#[test]
fn a_clean_fit_is_charged_for_its_parameters_only() {
    let p = Penalties::default();
    let fit = clean();
    assert_eq!(
        p.terms(&fit),
        vec![("theta", 30.0), ("omega", 20.0), ("sigma", 10.0)]
    );
    assert_eq!(p.score(&fit), 500.0 + PARAMETER_CHARGE);
    // The criterion is the score, and the search reads the schedule back
    // off it.
    let c = Criterion::Penalized(p);
    assert_eq!(c.of(&fit), 500.0 + PARAMETER_CHARGE);
    assert_eq!(c.label(), "penalized");
    assert_eq!(c.penalties(), Some(p));
    assert_eq!(Criterion::Ofv.penalties(), None);
}

#[test]
fn every_failure_term_is_charged_on_its_own_input() {
    let p = Penalties::default();
    let base = 500.0 + PARAMETER_CHARGE;

    let mut fit = clean();
    fit.converged = false;
    assert_eq!(p.score(&fit), base + 100.0, "convergence");
    assert!(p.terms(&fit).contains(&("convergence", 100.0)));

    for status in [
        CovarianceStatus::Failed,
        CovarianceStatus::NotRequested,
        CovarianceStatus::SirFallback,
    ] {
        let mut fit = clean();
        fit.covariance_status = status.clone();
        fit.covariance_matrix = None;
        fit.cov_condition_number = None;
        assert_eq!(p.score(&fit), base + 100.0, "{status:?}");
        // Without a matrix there is no correlation or condition number to
        // charge for: one term, not three.
        assert_eq!(p.terms(&fit).len(), 4, "{status:?}");
    }
    // A step that reports computed but stored nothing is a failed step.
    let mut fit = clean();
    fit.covariance_matrix = None;
    assert!(p.terms(&fit).contains(&("covariance", 100.0)));

    let mut fit = clean();
    fit.covariance_matrix = Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.96, 0.96, 1.0]));
    assert_eq!(p.score(&fit), base + 100.0, "correlation");
    assert!(p.terms(&fit).contains(&("correlation", 100.0)));
    // Exactly at the threshold is not over it.
    let mut fit = clean();
    fit.covariance_matrix = Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.95, 0.95, 1.0]));
    assert_eq!(p.score(&fit), base, "correlation at the threshold");

    let mut fit = clean();
    fit.cov_condition_number = Some(1000.5);
    assert_eq!(p.score(&fit), base + 100.0, "condition number");
    let mut fit = clean();
    fit.cov_condition_number = Some(f64::NAN);
    assert_eq!(p.score(&fit), base + 100.0, "NaN condition number");

    // Everything at once: pyDarwin's worst fitted model.
    let mut fit = clean();
    fit.converged = false;
    fit.covariance_matrix = Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.99, 0.99, 1.0]));
    fit.cov_condition_number = Some(1e5);
    // Convergence, correlation and condition number — the matrix is present,
    // so the covariance charge does not apply.
    assert_eq!(p.score(&fit), base + 300.0);
}

#[test]
fn the_schedule_scales_each_term() {
    let p = Penalties {
        theta: 1.0,
        omega: 2.0,
        sigma: 4.0,
        convergence: 8.0,
        covariance: 16.0,
        correlation: 32.0,
        max_correlation: 0.4,
        condition_number: 64.0,
        max_condition_number: 5.0,
        non_influential: 0.5,
        crash: 1e9,
        gate: 7.0,
    };
    let mut fit = clean();
    fit.converged = false;
    // |r| = 0.5 exceeds the tighter 0.4; CN 10 exceeds 5.
    assert_eq!(p.score(&fit), 500.0 + 3.0 + 4.0 + 4.0 + 8.0 + 32.0 + 64.0);
    assert_eq!(p.non_influential_charge(3), 1.5);
    assert_eq!(p.non_influential_charge(0), 0.0);
}

#[test]
fn a_kappa_is_an_omega_element_and_an_old_tally_charges_at_the_theta_rate() {
    let p = Penalties::default();
    let mut fit = clean();
    fit.bic_inputs.kappa = 3;
    fit.n_parameters = 9;
    assert!(p.terms(&fit).contains(&("omega", 50.0)));

    // A bundle saved before the tally existed: all-zero counts against a
    // non-zero `n_parameters`.
    let mut fit = clean();
    fit.bic_inputs = BicInputs::default();
    fit.n_parameters = 4;
    assert_eq!(
        p.terms(&fit),
        vec![("theta", 40.0)],
        "an unreadable tally is charged as θ"
    );
}

#[test]
fn a_schedule_is_validated() {
    assert!(Penalties::default().validate().is_ok());
    for (field, bad) in [
        (
            "theta",
            Penalties {
                theta: -1.0,
                ..Penalties::default()
            },
        ),
        (
            "crash",
            Penalties {
                crash: f64::INFINITY,
                ..Penalties::default()
            },
        ),
        (
            "max_correlation",
            Penalties {
                max_correlation: 0.0,
                ..Penalties::default()
            },
        ),
        (
            "max_condition_number",
            Penalties {
                max_condition_number: f64::NAN,
                ..Penalties::default()
            },
        ),
    ] {
        let e = bad.validate().unwrap_err();
        assert!(e.contains(&format!("[rank.penalties] {field} = ")), "{e}");
    }
}

#[test]
fn a_partial_table_overlays_the_defaults() {
    let p: Penalties = toml::from_str("theta = 1\nconvergence = 250").unwrap();
    assert_eq!(
        p,
        Penalties {
            theta: 1.0,
            convergence: 250.0,
            ..Penalties::default()
        }
    );
    let e = toml::from_str::<Penalties>("sigma = 1\nsigmas = 2").unwrap_err();
    assert!(e.to_string().contains("unknown field `sigmas`"), "{e}");
}

#[test]
fn the_manifest_key_carries_the_whole_schedule() {
    // Two penalized runs with different charges must not share a journal:
    // the label alone would let one reuse the other's rows.
    let a = Criterion::Penalized(Penalties::default());
    let b = Criterion::Penalized(Penalties {
        theta: 11.0,
        ..Penalties::default()
    });
    assert_ne!(a.manifest_key(), b.manifest_key());
    assert!(
        a.manifest_key().starts_with("penalized{theta=10,"),
        "{}",
        a.manifest_key()
    );
    assert!(
        a.manifest_key().ends_with("crash=99999999,gate=100}"),
        "{}",
        a.manifest_key()
    );
    // Every other criterion keys on its label, as before — an existing
    // journal still resumes.
    assert_eq!(Criterion::Ofv.manifest_key(), "ofv");
    assert_eq!(
        Criterion::Bic(ferx_core::BicType::Mixed).manifest_key(),
        "bic_mixed"
    );
}
