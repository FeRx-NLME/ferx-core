//! End-to-end check of `[fit_options] ode_method`: the stiff Rosenbrock steppers must reach
//! the integrator through the ordinary parse → `sync_ode_solver_opts` → `predict()` pipeline
//! and produce the same trajectory the explicit method produces when it is given enough
//! budget to be right.
//!
//! The model is a fast reversible-binding system (the TMDD / quasi-equilibrium shape): with
//! `KON = 1e4` and `KOFF = 1e3` alongside an hours-scale elimination, the Jacobian carries an
//! eigenvalue near `−1.1e4`, so an explicit method is **stability**-limited — it needs
//! ~10⁵ steps over the 24 h span regardless of the requested tolerance. That is what makes
//! this a solver test and not another transcription test.
//!
//! Fast (`predict()` only, no `fit()`), so it is not `slow-tests`-gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population, Subject};

mod common;

/// Observation times spanning the fast binding transient (0.25 h) and the slow elimination
/// tail (24 h) — a method that is only stable, not accurate, still fails the early points.
fn obs_times() -> Vec<f64> {
    vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
}

// The model's constants live here rather than as literals in the `.ferx` source below, so the
// closed form and the model that is actually integrated cannot drift apart.
const CL: f64 = 3.0;
const V: f64 = 20.0;
const KON: f64 = 1.0e4;
const KOFF: f64 = 1.0e3;
const DOSE: f64 = 100.0;

/// The exact scaled prediction `central(t) / V`, in closed form.
///
/// The system is linear with a constant coefficient matrix, so `u(t) = exp(A·t)·u₀` with
///
/// ```text
///     ⎡ −CL/V − KON   KOFF ⎤
/// A = ⎢                    ⎥,   u₀ = (DOSE, 0)
///     ⎣  KON         −KOFF ⎦
/// ```
///
/// For a 2×2 matrix `exp(A·t)` is Sylvester's formula
/// `(e^{λ₁t}(A − λ₂I) − e^{λ₂t}(A − λ₁I)) / (λ₁ − λ₂)`; `u₀` has only a first component, so
/// only the `(0,0)` entry is ever needed.
///
/// The two eigenvalues are deliberately **not** both taken from the quadratic formula. Here
/// `tr = −11000.15` and `det = 150`, so the small root computed as `(tr + √(tr² − 4·det))/2`
/// subtracts two numbers agreeing in their first five digits — and that root is precisely the
/// one governing the elimination tail this test measures out to 24 h. The large root is
/// cancellation-free (both terms carry `tr`'s sign), and the small one then follows from
/// `λ₁·λ₂ = det`, which is exact.
fn exact_conc(t: f64) -> f64 {
    let (a, b) = (-(CL / V) - KON, KOFF);
    let (c, d) = (KON, -KOFF);
    let (tr, det) = (a + d, a * d - b * c);

    let lambda_big = 0.5 * (tr - (tr * tr - 4.0 * det).sqrt());
    let lambda_small = det / lambda_big;

    let (l1, l2) = (lambda_small, lambda_big);
    let e00 = ((l1 * t).exp() * (a - l2) - (l2 * t).exp() * (a - l1)) / (l1 - l2);
    DOSE * e00 / V
}

fn stiff_population() -> Population {
    let times = obs_times();
    let n = times.len();
    let subject: Subject = common::subject(
        "1",
        vec![DoseEvent::new(0.0, DOSE, 1, 0.0, false, 0.0)],
        times,
        vec![0.0; n],
        vec![1; n],
    );
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject],
    }
}

/// Free drug in `central` binds reversibly to a target (`bound`) far faster than it is
/// eliminated. `[fit_options]` is appended by the caller.
fn stiff_model_src(fit_options: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL({CL:?}, 0.01, 100.0)
  theta TVV({V:?}, 0.1, 1000.0)
  theta TVKON({KON:?}, 1.0, 1.0e7)
  theta TVKOFF({KOFF:?}, 1.0, 1.0e7)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KON  = TVKON
  KOFF = TVKOFF
[structural_model]
  ode(obs_cmt=central, states=[central, bound])
[odes]
  d/dt(central) = -(CL/V) * central - KON * central + KOFF * bound
  d/dt(bound)   =  KON * central - KOFF * bound
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
{fit_options}
"#
    )
}

fn predictions(fit_options: &str, pop: &Population) -> Vec<(f64, f64)> {
    let src = stiff_model_src(fit_options);
    let model = parse_full_model(&src)
        .unwrap_or_else(|e| panic!("model did not parse: {e}"))
        .model;
    predict(&model, pop, &model.default_params)
        .iter()
        .map(|p| (p.time, p.pred))
        .collect()
}

/// The reference: the *default* explicit stepper, run at a tight tolerance with a step budget
/// large enough to absorb the stability limit. This is the same integrator every existing
/// (NONMEM-anchored) ODE result comes from — the stiff methods are checked against it rather
/// than against a golden value.
const RK45_REFERENCE_OPTIONS: &str = "  ode_method = rk45\n  ode_reltol = 1e-10\n  \
                                      ode_abstol = 1e-12\n  ode_max_steps = 5000000\n";

#[test]
fn rosenbrock_methods_reproduce_the_brute_force_rk45_trajectory() {
    let pop = stiff_population();
    let reference = predictions(RK45_REFERENCE_OPTIONS, &pop);
    assert_eq!(reference.len(), obs_times().len());
    assert!(
        reference.iter().all(|(_, p)| p.is_finite() && *p > 0.0),
        "reference predictions must be finite and positive: {reference:?}"
    );

    // First: is the brute-force RK45 reference itself right? Checked against the closed form,
    // because if it is not, every comparison below is meaningless.
    for (t, p) in &reference {
        let e = exact_conc(*t);
        assert!(
            (p - e).abs() <= 1e-8 + 1e-6 * e.abs(),
            "RK45 reference at t={t}: {p:.10} vs closed form {e:.10}"
        );
    }

    // Each stiff method at a tolerance appropriate to its order, on the *default* step
    // budget — the point being that they need no budget inflation at all.
    for (method, reltol, abstol, rtol) in [
        ("rodas5p", "1e-9", "1e-11", 1e-6),
        ("rodas4", "1e-9", "1e-11", 1e-6),
        ("rosenbrock23", "1e-7", "1e-9", 1e-3),
    ] {
        let opts =
            format!("  ode_method = {method}\n  ode_reltol = {reltol}\n  ode_abstol = {abstol}\n");
        let got = predictions(&opts, &pop);
        assert_eq!(got.len(), reference.len(), "[{method}] prediction count");
        println!("\n=== {method} vs closed form ===");
        for ((t, p), (_, r)) in got.iter().zip(reference.iter()) {
            let e = exact_conc(*t);
            println!(
                "  t={t:6.2}  ferx {p:.10}  exact {e:.10}  rel {:.2e}",
                (p - e).abs() / e.abs()
            );
            // Against the closed form — mathematics, not a sibling integrator.
            let tol_exact = 1e-9 + rtol * e.abs();
            assert!(
                (p - e).abs() <= tol_exact,
                "[{method}] t={t:.2}: PRED {p:.9} vs closed form {e:.9} \
                 (|diff| {:.2e} > tol {tol_exact:.2e})",
                (p - e).abs()
            );
            // …and against the brute-force RK45 run, so a regression in either surfaces.
            let tol = 1e-9 + rtol * r.abs();
            assert!(
                (p - r).abs() <= tol,
                "[{method}] t={t:.2}: PRED {p:.9} vs RK45 reference {r:.9} \
                 (|diff| {:.2e} > tol {tol:.2e})",
                (p - r).abs()
            );
        }
    }
}

/// The premise of the test above: on this system the explicit method at the *default* step
/// budget cannot reach the end of the span, so its late predictions are materially wrong.
/// If this ever stops holding, the model is no longer stiff and the comparison above has
/// quietly stopped testing anything.
#[test]
fn the_explicit_method_is_stability_limited_on_this_system() {
    let pop = stiff_population();
    let reference = predictions(RK45_REFERENCE_OPTIONS, &pop);
    let default_budget = predictions("  ode_reltol = 1e-8\n  ode_abstol = 1e-10\n", &pop);

    let (t_last, p_last) = *default_budget.last().unwrap();
    let (_, r_last) = *reference.last().unwrap();
    let rel = (p_last - r_last).abs() / r_last.abs();
    assert!(
        rel > 1e-3,
        "RK45 on the default budget landed within 0.1% at t={t_last} (rel {rel:.2e}) — this \
         system is no longer stiff enough for the ode_method comparison to mean anything"
    );
}
