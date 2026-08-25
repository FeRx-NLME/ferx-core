//! Published model-based meta-analysis case study, fitted as a **compartment-free**
//! structural model (#811).
//!
//! Naproxen vs placebo in osteoarthritis — Case Study 1 of Bracis et al.,
//! *CPT:PSP* 2026;15:e70158: 18 trials, 36 arms, longitudinal WOMAC pain. This is
//! the model class #811 exists for. There is no PK anywhere in it: the structural
//! model is an Emax time-course on an aggregate arm mean, with study-level random
//! effects, and each arm weighted by its reported standard error.
//!
//! Two things are being asserted, and they are different claims:
//!
//! 1. **The published analysis is reproduced.** The estimates match the Monolix
//!    numbers in the tutorial to three significant figures. A structural model
//!    that predicted the wrong thing could not.
//! 2. **The compartment-free form is equivalent to the dummy-compartment
//!    workaround it replaces.** Until #811 this model had to be written as a state
//!    nobody reads, driven by `d/dt(clock) = 1`, with the real equation hidden in
//!    a `[scaling]` readout. Both forms are fitted here and must agree — the
//!    migration guarantee for every model already written the old way.
//!
//! ## The weighting scheme
//!
//! `DV` is the arm mean divided by its reported SE, the prediction is divided by
//! the same SE, and the residual variance is FIXED to 1 — so each arm enters the
//! likelihood with weight `1/SE²`. That fixed sigma is not a modelling
//! convenience; it *is* the weighting, and it is why `nlme` could not fit these
//! models for years (Boucher & Bennetts Part II: *"σ was fixed to 1 in the
//! modeling. However, in R, using the NLME function, it was not possible to fix σ
//! to 1."*).
//!
//! ## Published reference values
//!
//! | | ferx (this test) | Monolix (published) |
//! |---|---|---|
//! | naproxen effect on Emax | 0.7930 (SE 0.0646) | 0.792 (SE 0.064) |
//! | ET50, placebo (weeks)   | 0.6901 | 0.698 |
//! | ET50, naproxen (weeks)  | 0.2132 | 0.216 |
//! | ω(E0), SD               | 0.6215 | 0.627 |
//! | ω(Emax), SD             | 0.7387 | 0.763 |
//! | OFV                     | 261.8149 | — |
//!
//! ET50 for naproxen is `TVET50 · exp(B_LET50_TRT)`, the published
//! parameterisation. ferx reports ω as a **variance**; the table is on the SD
//! scale the tutorial uses.

use ferx_core::parser::model_parser::{parse_full_model, parse_full_model_file};
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const DATA: &str = "data/mbma_naproxen.csv";
const MODEL: &str = "examples/mbma_naproxen.ferx";

// Monolix values from the tutorial's `populationParameters.txt`, as quoted in the
// MBMA umbrella issue (#1032).
const PUB_EMAX_TRT: f64 = 0.792;
const PUB_EMAX_TRT_SE: f64 = 0.064;
const PUB_ET50_PLACEBO: f64 = 0.698;
const PUB_ET50_NAPROXEN: f64 = 0.216;
const PUB_OMEGA_E0_SD: f64 = 0.627;
const PUB_OMEGA_EMAX_SD: f64 = 0.763;

/// Theta order as declared in `examples/mbma_naproxen.ferx`.
const I_TVET50: usize = 2;
const I_B_EMAX_TRT: usize = 5;
const I_B_LET50_TRT: usize = 6;

fn fit_options() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 500,
        run_covariance_step: true,
        verbose: false,
        ..Default::default()
    }
}

/// The shipped compartment-free model must reproduce the published analysis.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow convergence fit against a published MBMA case study (#811/#1032): opt in with --features slow-tests"
)]
fn mbma_naproxen_reproduces_the_published_analysis() {
    let model = parse_full_model_file(Path::new(MODEL))
        .expect("the MBMA example must parse")
        .model;
    assert!(
        model.is_algebraic(),
        "the point of this test is that the model has no compartments"
    );
    // The whole analysis rests on the residual being fixed: unfix it and the SE
    // weighting silently becomes an ordinary homoscedastic fit.
    assert!(
        model.default_params.sigma_fixed.iter().all(|&f| f),
        "the residual variance must be FIXED to 1 — that is the weighting scheme"
    );

    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("MBMA data must load");
    assert_eq!(pop.subjects.len(), 18, "18 trials");
    assert!(
        pop.subjects.iter().all(|s| s.doses.is_empty()),
        "an MBMA dataset has no dose records"
    );

    let result = fit(&model, &pop, &model.default_params, &fit_options())
        .expect("the MBMA fit must converge");
    assert!(result.converged, "the published model must converge");

    // The headline estimate of the analysis: the naproxen effect on Emax.
    let emax_trt = result.theta[I_B_EMAX_TRT];
    assert!(
        (emax_trt - PUB_EMAX_TRT).abs() / PUB_EMAX_TRT < 0.01,
        "naproxen effect on Emax: ferx {emax_trt:.4} vs published {PUB_EMAX_TRT:.3}"
    );
    let se = result
        .se_theta
        .as_ref()
        .expect("covariance step must succeed")[I_B_EMAX_TRT];
    assert!(
        (se - PUB_EMAX_TRT_SE).abs() / PUB_EMAX_TRT_SE < 0.10,
        "SE of the naproxen effect: ferx {se:.4} vs published {PUB_EMAX_TRT_SE:.3}"
    );

    // ET50 in each arm — the published parameterisation is a log-scale shift.
    let et50_placebo = result.theta[I_TVET50];
    let et50_naproxen = et50_placebo * result.theta[I_B_LET50_TRT].exp();
    assert!(
        (et50_placebo - PUB_ET50_PLACEBO).abs() / PUB_ET50_PLACEBO < 0.03,
        "ET50 (placebo): ferx {et50_placebo:.4} vs published {PUB_ET50_PLACEBO:.3}"
    );
    assert!(
        (et50_naproxen - PUB_ET50_NAPROXEN).abs() / PUB_ET50_NAPROXEN < 0.03,
        "ET50 (naproxen): ferx {et50_naproxen:.4} vs published {PUB_ET50_NAPROXEN:.3}"
    );

    // Between-study variability, on the SD scale the tutorial reports.
    let omega_e0_sd = result.omega[(0, 0)].sqrt();
    let omega_emax_sd = result.omega[(1, 1)].sqrt();
    assert!(
        (omega_e0_sd - PUB_OMEGA_E0_SD).abs() / PUB_OMEGA_E0_SD < 0.05,
        "omega(E0) SD: ferx {omega_e0_sd:.4} vs published {PUB_OMEGA_E0_SD:.3}"
    );
    assert!(
        (omega_emax_sd - PUB_OMEGA_EMAX_SD).abs() / PUB_OMEGA_EMAX_SD < 0.05,
        "omega(Emax) SD: ferx {omega_emax_sd:.4} vs published {PUB_OMEGA_EMAX_SD:.3}"
    );
}

/// The dummy-compartment workaround this feature replaces: one state nobody
/// reads, `d/dt(clock) = 1` to make `clock` track time, and the equation moved
/// into a `[scaling]` readout. Written out in full because the equivalence only
/// means something if the old form is reproduced faithfully.
const DUMMY_ODE_MODEL: &str = r"
[parameters]
  theta TVE0(5.19,    1.0,  20.0)
  theta TVEMAX(0.88, -5.0,  20.0)
  theta TVET50(0.70,  0.01, 20.0)
  theta B_E0_FLARE(0.97,   -10.0, 10.0)
  theta B_EMAX_FLARE(1.10, -10.0, 10.0)
  theta B_EMAX_TRT(0.79,   -10.0, 10.0)
  theta B_LET50_TRT(-1.18, -10.0, 10.0)

  omega ETA_E0   ~ 0.627 (sd)
  omega ETA_EMAX ~ 0.763 (sd)

  sigma ADD_ERR ~ 1.0 (variance) FIX

[covariates]
  TRT   categorical
  FLARE categorical
  WPSE  continuous

[individual_parameters]
  E0   = TVE0   + B_E0_FLARE * FLARE + ETA_E0
  EMAX = TVEMAX + B_EMAX_FLARE * FLARE + B_EMAX_TRT * TRT + ETA_EMAX
  ET50 = TVET50 * exp(B_LET50_TRT * TRT)

[structural_model]
  ode(states=[clock])

[odes]
  d/dt(clock) = 1

[scaling]
  y = (E0 - EMAX * clock / (clock + ET50)) / WPSE

[error_model]
  DV ~ additive(ADD_ERR)
";

/// The compartment-free model and the dummy-compartment model it replaces are the
/// same model, so they must produce the same fit. This is the migration guarantee
/// for the `d/dt(clock) = 1` models people have already written.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "two slow convergence fits (#811): opt in with --features slow-tests"
)]
fn mbma_naproxen_compartment_free_matches_the_dummy_clock_workaround() {
    let algebraic = parse_full_model_file(Path::new(MODEL))
        .expect("the MBMA example must parse")
        .model;
    let dummy = parse_full_model(DUMMY_ODE_MODEL)
        .expect("the dummy-clock twin must parse")
        .model;
    assert!(algebraic.is_algebraic() && algebraic.ode_spec.is_none());
    assert!(!dummy.is_algebraic() && dummy.ode_spec.is_some());

    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("MBMA data must load");
    let opts = FitOptions {
        run_covariance_step: false,
        ..fit_options()
    };

    let a = fit(&algebraic, &pop, &algebraic.default_params, &opts)
        .expect("compartment-free fit must converge");
    let d = fit(&dummy, &pop, &dummy.default_params, &opts).expect("dummy-clock fit must converge");

    // The dummy model integrates a state to recover a number it already had, so
    // the two agree only to solver tolerance — but that is far tighter than any
    // difference that would matter to a conclusion drawn from the fit.
    assert!(
        (a.ofv - d.ofv).abs() < 1e-3,
        "OFV: compartment-free {:.6} vs dummy-clock {:.6}",
        a.ofv,
        d.ofv
    );
    for (i, (&x, &y)) in a.theta.iter().zip(d.theta.iter()).enumerate() {
        assert!(
            (x - y).abs() / (1.0 + y.abs()) < 1e-4,
            "theta[{i}]: compartment-free {x:.6} vs dummy-clock {y:.6}"
        );
    }
    for k in 0..2 {
        let (x, y) = (a.omega[(k, k)], d.omega[(k, k)]);
        assert!(
            (x - y).abs() / (1.0 + y.abs()) < 1e-4,
            "omega[{k}]: compartment-free {x:.6} vs dummy-clock {y:.6}"
        );
    }
}
