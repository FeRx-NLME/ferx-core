//! Tier-2 checks for the `power(σ, P)` residual-error form (#1182): the public
//! `fit()` / `simulate()` boundary, never to convergence.
//!
//! Two things a unit test on the variance math cannot see:
//!
//! * **The degenerate oracle.** `power(σ, P)` with `P = 1 FIX` *is* the
//!   proportional model, so an evaluation of the two must agree **bit for bit**
//!   — the exponent rides the magnitude channel, and a channel that changed the
//!   arithmetic on its neutral value would move every base fit a search
//!   started from. The same evaluation at `P = 1.3` must differ, or the pair
//!   is a tautology (the CLAUDE.md straddle rule).
//! * **Every consumer reads the exponent.** IWRES, CWRES, the simulated draw
//!   and each estimator route the variance through code the exponent has to
//!   reach; a path that fell back to the unscaled variance would show as a
//!   proportional IWRES on a power fit.

use std::path::Path;

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, simulate_with_seed, EstimationMethod, FitOptions};

const DATA: &str = "data/warfarin.csv";

fn model(error_model: &str, exponent_decl: &str) -> ferx_core::types::CompiledModel {
    let src = format!(
        "[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
{exponent_decl}
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  {error_model}

[fit_options]
  method = focei
"
    );
    parse_model_string(&src).expect("model must parse")
}

fn evaluate(m: &ferx_core::types::CompiledModel, method: EstimationMethod) -> ferx_core::FitResult {
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let opts = FitOptions {
        method,
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        threads: Some(1),
        ..FitOptions::default()
    };
    fit(m, &pop, &m.default_params, &opts).expect("evaluation must not error")
}

#[test]
fn a_unit_exponent_is_the_proportional_model_bit_for_bit_and_another_is_not() {
    let prop = model("DV ~ proportional(PROP_ERR)", "");
    let unit = model(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "  theta RUV_POW(1.0, 0.01, 10.0) FIX",
    );
    let other = model(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "  theta RUV_POW(1.3, 0.01, 10.0) FIX",
    );
    // The exponent rides the residual-magnitude channel, whose variance is
    // built as `((f·f)·σ)·σ` where the bare proportional path builds
    // `(f·σ)·(f·σ)` — a documented ~1-ULP reassociation (`residual_error.rs`).
    // So the bit-for-bit twin of `power(σ, 1)` is the magnitude-active
    // proportional model `proportional(σ * 1.0)`, and the plain proportional
    // model agrees to rounding.
    let scaled = model("DV ~ proportional(PROP_ERR * 1.0)", "");
    for method in [EstimationMethod::FoceI, EstimationMethod::Foce] {
        let a = evaluate(&prop, method);
        let s = evaluate(&scaled, method);
        let b = evaluate(&unit, method);
        let c = evaluate(&other, method);
        assert_eq!(
            s.ofv.to_bits(),
            b.ofv.to_bits(),
            "{method:?}: P = 1 must be the magnitude-scaled proportional OFV bit for bit              ({} vs {})",
            s.ofv,
            b.ofv
        );
        assert!(
            (a.ofv - b.ofv).abs() <= 1e-9 * a.ofv.abs(),
            "{method:?}: P = 1 must be the proportional OFV to rounding ({} vs {})",
            a.ofv,
            b.ofv
        );
        assert_ne!(
            a.ofv, c.ofv,
            "{method:?}: P = 1.3 must be a different model"
        );
        for (ss, sb) in s.subjects.iter().zip(&b.subjects) {
            assert_eq!(ss.iwres, sb.iwres);
            assert_eq!(ss.cwres, sb.cwres);
        }
        // The exponent reaches IWRES: on the power fit IWRES is
        // (y − f)/(σ·f^p), which differs from the proportional (y − f)/(σ·f).
        let some_differ = a
            .subjects
            .iter()
            .zip(&c.subjects)
            .any(|(sa, sc)| sa.iwres.iter().zip(&sc.iwres).any(|(x, y)| x != y));
        assert!(some_differ, "{method:?}: IWRES did not see the exponent");
        assert!(c
            .subjects
            .iter()
            .all(|s| s.iwres.iter().chain(&s.cwres).all(|v| v.is_finite())));
    }
}

#[test]
fn a_free_exponent_is_estimated_by_every_estimator() {
    let m = model(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "  theta RUV_POW(1.0, 0.01, 10.0)",
    );
    assert!(m.has_ruv_exponent());
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    // FOCEI and FOCE take the analytic exponent-aware gradients; SAEM
    // differences the magnitude-aware data term (the Gauss-Newton and IMP
    // paths are covered at the unit level and in the slow tier — their
    // finite-difference fallbacks are minutes in a debug build).
    for method in [
        EstimationMethod::FoceI,
        EstimationMethod::Foce,
        EstimationMethod::Saem,
    ] {
        // Tier-2: a handful of outer iterations, and the Monte-Carlo
        // estimators on a few dozen samples — enough to exercise their
        // exponent-aware paths, never a convergence run.
        let opts = FitOptions {
            method,
            outer_maxiter: 2,
            saem_n_exploration: 4,
            saem_n_convergence: 4,
            imp_samples: 40,
            run_covariance_step: false,
            verbose: false,
            checkpoint: false,
            threads: Some(2),
            ..FitOptions::default()
        };
        let r =
            fit(&m, &pop, &m.default_params, &opts).unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(r.ofv.is_finite(), "{method:?}");
        let p = r.theta[3];
        assert!(p.is_finite() && p > 0.0, "{method:?}: exponent {p}");
        assert_eq!(r.theta_names[3], "RUV_POW");
        assert_eq!(r.n_parameters, 8, "{method:?}: 4 θ + 3 ω + 1 σ");
    }
}

#[test]
fn simulate_draws_with_the_power_variance() {
    // σ = 0.1, P = 2: at f ≈ 10 the residual SD is 0.1·100 = 10, an order of
    // magnitude above the proportional 1.0 — the two draws cannot be confused.
    let power = model(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "  theta RUV_POW(2.0, 0.01, 10.0) FIX",
    );
    let prop = model("DV ~ proportional(PROP_ERR)", "");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let spread = |m: &ferx_core::types::CompiledModel| -> f64 {
        let sim = simulate_with_seed(m, &pop, &m.default_params, 1, 7);
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for row in &sim {
            if row.ipred > 5.0 {
                num += ((row.outcome.continuous_value() - row.ipred) / row.ipred).powi(2);
                den += 1.0;
            }
        }
        assert!(den > 0.0);
        (num / den).sqrt()
    };
    let cv_power = spread(&power);
    let cv_prop = spread(&prop);
    assert!(
        cv_power > 3.0 * cv_prop,
        "power CV {cv_power} should dwarf proportional CV {cv_prop}"
    );
}

#[test]
fn a_non_positive_exponent_is_refused_before_fitting() {
    let m = model(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "  theta RUV_POW(0.0, -1.0, 10.0)",
    );
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let err = fit(&m, &pop, &m.default_params, &FitOptions::default())
        .err()
        .expect("a zero exponent must be rejected");
    assert!(err.contains("power exponent"), "{err}");
}
