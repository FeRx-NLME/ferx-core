//! `sin(...)` / `cos(...)` in an `[odes]` right-hand side (#2720): a circadian
//! forcing term on an indirect-response baseline, against the closed-form
//! solution of the linear ODE.
//!
//! The model is `dR/dt = KIN·(1 + AMP·cos(ω(t + SHIFT) − PHS)) + B·sin(2ω(t + SHIFT)) − KOUT·R`
//! with `R(0) = KIN/KOUT` and no doses. It is linear in `R` with a sinusoidal
//! input, so the solution is exact: the steady-state harmonic responses plus the
//! homogeneous decay that makes `R(0)` hold,
//!
//! ```text
//! R(t) = KIN/KOUT + P(t) + C·e^{−KOUT·t},   C = −P(0),
//! P(t) = KIN·AMP·(k·cos(u) + ω·sin(u))/(k² + ω²) + B·(k·sin(v) − 2ω·cos(v))/(k² + 4ω²),
//! u = ω(t + SHIFT) − PHS,  v = 2ω(t + SHIFT),  k = KOUT.
//! ```
//!
//! Non-degenerate: `SHIFT` and `PHS` are nonzero, so neither argument is a
//! multiple of `ωt`; both harmonics are live (dropping either moves `R` by more
//! than 1 at most times); the samples span two days, past the transient. An
//! `[odes]` RHS that reads `sin`/`cos` runs on the bytecode interpreter (the
//! register compiler declines both, see `rhs_regcode`), so this pins that route's
//! values end to end through `predict()`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::Population;

mod common;

const SRC: &str = r"
[parameters]
  theta KIN(10.0, 0.1, 100.0)
  theta KOUT(0.4, 0.01, 10.0)
  theta AMP(0.3, 0.0, 1.0)
  theta PHS(0.8, -5.0, 5.0)
  theta B(1.5, 0.0, 10.0)
  theta SHIFT(1.7, -12.0, 12.0)

  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  KINI  = KIN
  KOUTI = KOUT
  AMPI  = AMP
  PHSI  = PHS
  BI    = B
  SHIFTI = SHIFT

[structural_model]
  ode(obs_cmt=resp, states=[resp])

[odes]
  init(resp) = KINI / KOUTI
  d/dt(resp) = KINI * (1 + AMPI * cos(2 * 3.141592653589793 * (TIME + SHIFTI) / 24 - PHSI)) + BI * sin(4 * 3.141592653589793 * (TIME + SHIFTI) / 24) - KOUTI * resp

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-12
";

fn closed_form(t: f64) -> f64 {
    let (kin, k, amp, phs, b, shift) = (10.0, 0.4, 0.3, 0.8, 1.5, 1.7);
    let w = 2.0 * std::f64::consts::PI / 24.0;
    let p = |t: f64| {
        let u = w * (t + shift) - phs;
        let v = 2.0 * w * (t + shift);
        kin * amp * (k * u.cos() + w * u.sin()) / (k * k + w * w)
            + b * (k * v.sin() - 2.0 * w * v.cos()) / (k * k + 4.0 * w * w)
    };
    kin / k + p(t) - p(0.0) * (-k * t).exp()
}

#[test]
fn sin_cos_forcing_in_odes_matches_the_closed_form() {
    let model = parse_full_model(SRC)
        .expect("an [odes] RHS with sin/cos parses")
        .model;
    assert!(model.ode_spec.is_some(), "the model must integrate");

    let obs_times: Vec<f64> = (0..=16).map(|k| 3.0 * k as f64).collect();
    let n = obs_times.len();
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![],
            obs_times.clone(),
            vec![0.0; n],
            vec![1; n],
        )],
    };
    let rows = predict(&model, &pop, &model.default_params).expect("predict");
    assert_eq!(rows.len(), n);

    // The fixture is live: the harmonic part moves R well away from its baseline.
    let swing = obs_times
        .iter()
        .map(|&t| (closed_form(t) - 25.0).abs())
        .fold(0.0, f64::max); // nan-ok: closed_form is finite by construction, asserted below
    assert!(swing > 2.0, "the forcing must move R, swing {swing}");

    let mut worst = 0.0f64;
    for (row, &t) in rows.iter().zip(&obs_times) {
        let want = closed_form(t);
        assert!(
            row.pred.is_finite() && want.is_finite(),
            "t={t}: PRED {} vs {want}",
            row.pred
        );
        let rel = (row.pred - want).abs() / want.abs();
        assert!(rel.is_finite(), "t={t}: relative error {rel}");
        worst = worst.max(rel);
    }
    eprintln!("sin/cos forcing: worst relative error vs closed form {worst:.3e}");
    // Measured 7.5e-13 (Linux, at the fixture's 1e-12 solver tolerances); ~13× headroom.
    // Swapping `cos`/`sin` in either harmonic (a quarter-period phase error) moves R
    // by up to 42% (first) / 12% (second), measured by RK4 outside ferx.
    assert!(worst < 1e-11, "worst relative error {worst:.3e}");
}
