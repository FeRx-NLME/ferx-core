//! #3126: the EKF's Kalman `R` is the row's scored residual variance
//! (`row_residual_variance`) — magnitude row, `weight =`, `power` exponent, `block_sigma`'s ρ
//! and `exp(2η_r)` — not the bare `residual_variance(error_model, f, σ)`.
//!
//! NONMEM has no EKF, so there is no equivalent run to anchor against. The oracles are
//! (a) degenerate pairs whose two spellings are the same row variance by algebra (T1–T3, T9a),
//! each with a straddling control that the pre-#3126 filter cannot pass, (b) an exact closed
//! form outside both engines (T4: the scalar Riccati recursion of a 1-state linear SDE), and
//! (c) bit-identity of the plain path with the pre-#3126 formula (T5).
//!
//! Every assertion here runs at fixed η on `ekf_p_obs` / `individual_nll` (the value path); the
//! dual twin is pinned by `sens::ode_adjoint_tests` (`sde_row_variance_*`).

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::DoseEvent;

/// The two-state SDE fixture of `sens::ode_adjoint_tests::SDE_ONE_CPT`, with the error line,
/// extra `[parameters]` lines and an optional trailing block substituted.
fn sde_src(params: &str, error: &str) -> String {
    format!(
        "[parameters]\n  theta TVCL(0.134, 0.001, 10.0)\n  theta TVV(8.1, 0.1, 500.0)\n  \
         theta TVKA(1.0, 0.01, 50.0)\n  omega ETA_CL ~ 0.07\n  omega ETA_V ~ 0.02\n  \
         omega ETA_KA ~ 0.40\n{params}[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  \
         ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot / V - (CL / V) * central\n[diffusion]\n  central ~ 0.02\n\
         [error_model]\n  {error}\n"
    )
}

fn model(params: &str, error: &str) -> CompiledModel {
    let m = parse_model_string(&sde_src(params, error)).expect("parse");
    assert!(m.is_sde(), "premise: an SDE model");
    m
}

/// Two doses, the second on residual drug, six records — so every record after the first
/// carries a posterior built by a Kalman update whose `R` is under test.
fn subject_for(m: &CompiledModel) -> Subject {
    let times = [0.5, 1.0, 2.0, 6.0, 13.0, 18.0];
    let n = times.len();
    let mut s = Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: times.to_vec(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        ..Default::default()
    };
    s.covariates.insert("W".to_string(), 2.0);
    let eta = vec![0.0; m.n_eta];
    let preds = crate::pk::compute_predictions_with_tv(m, &s, &m.default_params.theta, &eta);
    s.observations =
        preds.iter().enumerate().map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin())).collect();
    s
}

/// `ekf_p_obs` as every production caller invokes it: the caller's own magnitude matrix and
/// the parameter set's live correlations.
fn p_obs(m: &CompiledModel, s: &Subject, eta: &[f64], sigma: &[f64]) -> Vec<f64> {
    let p = &m.default_params;
    let mult = m.ruv_obs_mult(s, &p.theta);
    ekf_p_obs(m, s, &p.theta, eta, sigma, &p.residual_correlations, mult.as_deref())
}

/// Worst relative gap between two `p_obs` vectors, finite-asserted before the fold.
fn worst_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite(), "{x} vs {y}");
            (x - y).abs() / y.abs()
        })
        .fold(0.0, f64::max) // nan-ok: finite asserted in the map
}

const ETA: [f64; 3] = [0.08, -0.06, 0.1];

/// T1: `proportional(PROP_ERR * 2)` at σ is the same row variance as `proportional(PROP_ERR)`
/// at 2σ, so the filter's `p_obs` must agree at every record. Straddle: the magnitude model at
/// σ against the plain one at σ — which is what the pre-#3126 filter computed for it — must
/// differ after the first record and be bit-equal at it (the first record is pre-update, so no
/// `R` has reached it). Dies when the row is dropped from the value closure.
#[test]
fn ekf_p_obs_reads_the_magnitude_row() {
    let sig = "  sigma PROP_ERR ~ 0.1 (sd)\n";
    let mag = model(sig, "DV ~ proportional(PROP_ERR * 2)");
    let plain = model(sig, "DV ~ proportional(PROP_ERR)");
    assert!(mag.ruv_obs_mult(&subject_for(&mag), &mag.default_params.theta).is_some());
    let s = subject_for(&plain);
    let sigma = plain.default_params.sigma.values.clone();
    let doubled: Vec<f64> = sigma.iter().map(|v| 2.0 * v).collect();
    let a = p_obs(&mag, &s, &ETA, &sigma);
    let want = p_obs(&plain, &s, &ETA, &doubled);
    let unscaled = p_obs(&plain, &s, &ETA, &sigma);
    let gap = worst_rel(&a, &want);
    let control = worst_rel(&a[1..], &unscaled[1..]);
    eprintln!("T1 magnitude: gap {gap:.3e}, straddle control {control:.3e}");
    // Measured 0 (doubling σ is exact in binary); bound at 1e-12.
    assert!(gap <= 1e-12, "magnitude row: {gap:.3e}");
    assert_eq!(a[0], unscaled[0], "the first record is pre-update");
    assert!(control > 1e-3, "straddle: the row must move p_obs ({control:.3e})");
}

/// T2: `additive(ADD_ERR) weight = W` with `W ≡ 2` is `additive` at 2σ. Dies when the row
/// (which here carries only the weight) is dropped.
#[test]
fn ekf_p_obs_reads_the_weight() {
    let sig = "  sigma ADD_ERR ~ 0.5 (sd)\n";
    let w = model(sig, "DV ~ additive(ADD_ERR) weight = W\n[covariates]\n  W continuous");
    let plain = model(sig, "DV ~ additive(ADD_ERR)");
    let s = subject_for(&plain);
    let sigma = plain.default_params.sigma.values.clone();
    let doubled: Vec<f64> = sigma.iter().map(|v| 2.0 * v).collect();
    let a = p_obs(&w, &s, &ETA, &sigma);
    let want = p_obs(&plain, &s, &ETA, &doubled);
    let control = worst_rel(&a[1..], &p_obs(&plain, &s, &ETA, &sigma)[1..]);
    let gap = worst_rel(&a, &want);
    eprintln!("T2 weight: gap {gap:.3e}, straddle control {control:.3e}");
    assert!(gap <= 1e-12, "weight: {gap:.3e}");
    assert!(control > 1e-3, "straddle: the weight must move p_obs ({control:.3e})");
}

/// T3: under `iiv_on_ruv = ETA_RUV`, `p_obs` at `η_r = 0.5` equals `p_obs` at `η_r = 0` with σ
/// scaled by `e^0.5` (an SD: the variance scales by `e^{2η_r}`). Straddle: at `η_r = 0.5` vs
/// `η_r = 0` at the same σ they must differ. Dies when `residual_var_scale` is dropped from the
/// closure.
#[test]
fn ekf_p_obs_reads_the_iiv_on_ruv_scale() {
    let params = "  omega ETA_RUV ~ 0.1\n  sigma PROP_ERR ~ 0.1 (sd)\n";
    let m = model(params, "DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV");
    assert_eq!(m.residual_error_eta, Some(3), "premise: η_r is the fourth BSV η");
    let s = subject_for(&m);
    let sigma = m.default_params.sigma.values.clone();
    let scaled: Vec<f64> = sigma.iter().map(|v| v * 0.5f64.exp()).collect();
    let at = |er: f64| [ETA[0], ETA[1], ETA[2], er];
    let a = p_obs(&m, &s, &at(0.5), &sigma);
    let want = p_obs(&m, &s, &at(0.0), &scaled);
    let control = worst_rel(&a[1..], &p_obs(&m, &s, &at(0.0), &sigma)[1..]);
    let gap = worst_rel(&a, &want);
    eprintln!("T3 iiv_on_ruv: gap {gap:.3e}, straddle control {control:.3e}");
    // `exp(2·0.5)` against `(e^0.5)²`: a few ULP apart, not bits.
    assert!(gap <= 1e-12, "iiv_on_ruv: {gap:.3e}");
    assert!(control > 1e-3, "straddle: η_r must move p_obs ({control:.3e})");
}

/// A 1-state linear SDE (`d/dt(central) = −ke·central`, bolus 100 at 0, diffusion `q`), whose
/// filter is exact in closed form: between records `P ← P·e^{−2keΔ} + q/(2ke)·(1 − e^{−2keΔ})`
/// (the scalar Riccati solution), and at a record `P⁺ = P·R/(P + R)` with `R = r(f(t))`,
/// `f(t) = 100·e^{−ke·t}`. Returns the pre-update `P` at each record — computed here, outside
/// both engines.
fn riccati_closed_form(ke: f64, q: f64, times: &[f64], r: impl Fn(f64) -> f64) -> Vec<f64> {
    let (mut p, mut t0, mut out) = (0.0f64, 0.0f64, Vec::new());
    for &t in times {
        let d = (-2.0 * ke * (t - t0)).exp();
        p = p * d + q / (2.0 * ke) * (1.0 - d);
        out.push(p);
        let rv = r(100.0 * (-ke * t).exp());
        p = p * rv / (p + rv);
        t0 = t;
    }
    out
}

fn linear_sde(params: &str, error: &str) -> (CompiledModel, Subject) {
    let src = format!(
        "[parameters]\n  theta TVCL(5.0, 0.1, 50.0)\n  theta TVV(100.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.09\n{params}[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V = TVV\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
         d/dt(central) = -(CL/V) * central\n[diffusion]\n  central ~ 20.0\n[error_model]\n  \
         {error}\n"
    );
    let mut m = parse_model_string(&src).expect("parse");
    m.ode_spec.as_mut().unwrap().solver_opts.reltol = 1e-12;
    m.ode_spec.as_mut().unwrap().solver_opts.abstol = 1e-12;
    assert!(m.is_sde());
    let times = [1.0, 4.0, 8.0];
    let s = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        observations: vec![90.0, 80.0, 65.0],
        obs_cmts: vec![1; 3],
        cens: vec![0; 3],
        ..Default::default()
    };
    (m, s)
}

/// T4: the filter against the closed-form recursion, for the three row terms the plain
/// `residual_variance` dropped — a `power` exponent (`R = σ²·f`), `block_sigma`'s ρ
/// (`R = σp²f² + 2ρσpσa·f + σa²`), and a θ-dependent magnitude (`R = (1.7·σ·f)²`). Each case's
/// control is the closed form with that term removed, which is what the pre-#3126 filter
/// computed: it must be farther from the filter than the bound. Dies when the exponent, ρ or
/// the row is dropped, each separately.
#[test]
fn ekf_p_obs_matches_the_closed_form_riccati_per_row_term() {
    let ke = 0.05;
    let times = [1.0, 4.0, 8.0];
    let cases: [(&str, &str, &str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>); 3] = [
        (
            "power",
            "  theta POW(0.5, 0.1, 2.0) FIX\n  sigma PROP_ERR ~ 0.1 (sd)\n",
            "DV ~ power(PROP_ERR, POW)",
            Box::new(|f| 0.01 * f),
            Box::new(|f| 0.01 * f * f),
        ),
        (
            "block_sigma",
            "  block_sigma (PROP_ERR, ADD_ERR) = [0.01, 0.008, 0.04]\n",
            "DV ~ combined(PROP_ERR, ADD_ERR)",
            Box::new(|f| 0.01 * f * f + 2.0 * 0.008 * f + 0.04),
            Box::new(|f| 0.01 * f * f + 0.04),
        ),
        (
            "magnitude",
            "  theta THM(1.7, 0.1, 5.0) FIX\n  sigma PROP_ERR ~ 0.1 (sd)\n",
            "DV ~ proportional(PROP_ERR * THM)",
            Box::new(|f| (1.7 * 0.1 * f) * (1.7 * 0.1 * f)),
            Box::new(|f| (0.1 * f) * (0.1 * f)),
        ),
    ];
    for (name, params, error, r, r_old) in cases {
        let (m, s) = linear_sde(params, error);
        let got = p_obs(&m, &s, &[0.0], &m.default_params.sigma.values);
        let want = riccati_closed_form(ke, 20.0, &times, &r);
        let old = riccati_closed_form(ke, 20.0, &times, &r_old);
        let gap = worst_rel(&got, &want);
        let control = worst_rel(&got, &old);
        eprintln!("T4 {name}: gap {gap:.3e}, pre-#3126 control {control:.3e}, {got:?}");
        // The filter's Crank–Nicolson Riccati step is the floor (`ekf.rs`'s free-drift test
        // measured 2.7e-5..5e-5 at these times). Measured: see the printed gaps; bound 2e-4.
        assert!(gap < 2e-4, "{name}: {gap:.3e}");
        assert!(control > 10.0 * 2e-4, "{name}: the term must be live ({control:.3e})");
    }
}

/// T5: with no row, no ρ and no `iiv_on_ruv`, the filter's `R` and `p_obs` are bit-identical to
/// the pre-#3126 formula — the public `ode_predictions_ekf_with_diffusion` fed
/// `residual_variance(error_model, f, σ)`, the closure `ekf_p_obs` used to pass. Proportional
/// and diagonal combined.
///
/// `R` is asserted directly, over a sweep of `f`: a 1-ULP change in `R` does not reach `p_obs`
/// here (`P ≪ R`, so `P·R/(P + R)` rounds it away), which is how a plain path routed through
/// `variance_at_scaled` with an all-ones row (`((f·f)·σ)·σ` against the legacy `(f·σ)·(f·σ)`)
/// survived the `p_obs`-only version of this test in the mutation sweep. The straddle asserts
/// that spelling really differs at some swept `f`, so the bit assertion can fail.
#[test]
fn ekf_p_obs_plain_path_is_bit_identical_to_the_pre_3126_formula() {
    for (params, error) in [
        ("  sigma PROP_ERR ~ 0.1 (sd)\n", "DV ~ proportional(PROP_ERR)"),
        (
            "  sigma PROP_ERR ~ 0.1 (sd)\n  sigma ADD_ERR ~ 0.3 (sd)\n",
            "DV ~ combined(PROP_ERR, ADD_ERR)",
        ),
    ] {
        let m = model(params, error);
        let s = subject_for(&m);
        let p = &m.default_params;
        let got = p_obs(&m, &s, &ETA, &p.sigma.values);
        let ode = m.ode_spec.as_ref().unwrap();
        let mut diffusion = vec![0.0; ode.n_states];
        let start = m.diffusion_theta_start.unwrap();
        for (k, &i) in m.diffusion_state_indices.iter().enumerate() {
            diffusion[i] = p.theta[start + k].max(0.0);
        }
        let pk = (m.pk_param_fn)(&p.theta, &ETA, &s.covariates, 0.0);
        let em = m.error_model;
        let (_, want) =
            crate::ode::ode_predictions_ekf_with_diffusion(ode, &pk.values, &s, &diffusion, |f| {
                crate::stats::residual_error::residual_variance(em, f, &p.sigma.values)
            });
        assert_eq!(got, want, "{error}");
        assert!(got[1..].iter().all(|v| *v > 0.0));
        let sig = &p.sigma.values;
        let mut differs = false;
        for k in 0..200 {
            let f = 0.37 + 0.913 * k as f64;
            let r = row_residual_variance(&m, 1, f, sig, &[], None, 1.0);
            let legacy = crate::stats::residual_error::residual_variance(em, f, sig);
            assert_eq!(r.to_bits(), legacy.to_bits(), "{error}: R at f = {f}");
            let all_ones = m.error_spec.variance_at_scaled(1, f, sig, &[], &[]);
            differs |= all_ones.to_bits() != legacy.to_bits();
        }
        assert!(differs, "{error}: straddle — the all-ones spelling must differ somewhere");
    }
}

/// T9a: the whole subject objective (`individual_nll`, prior + data term with `V = v_resid +
/// p_obs`) at fixed η agrees on both degenerate pairs of the issue — `proportional(S*2)` at σ
/// vs `proportional(S)` at 2σ, and `additive(S) weight = W` (W ≡ 2) vs `additive` at 2S. The
/// scoring already applied the row before #3126, so only the filter's `R` moved these; at the
/// CLI the same pairs were 3.60 and 1.04 OFV apart. Dies when the row is dropped from the value
/// closure.
#[test]
fn individual_nll_agrees_on_the_degenerate_row_pairs() {
    for (sig, a_err, b_err) in [
        (
            "  sigma PROP_ERR ~ 0.1 (sd)\n",
            "DV ~ proportional(PROP_ERR * 2)",
            "DV ~ proportional(PROP_ERR)",
        ),
        (
            "  sigma ADD_ERR ~ 0.5 (sd)\n",
            "DV ~ additive(ADD_ERR) weight = W\n[covariates]\n  W continuous",
            "DV ~ additive(ADD_ERR)",
        ),
    ] {
        let a = model(sig, a_err);
        let b = model(sig, b_err);
        let s = subject_for(&b);
        let p = &a.default_params;
        let doubled: Vec<f64> = p.sigma.values.iter().map(|v| 2.0 * v).collect();
        let la = individual_nll(&a, &s, &p.theta, &ETA, &p.omega, &p.sigma.values);
        let lb = individual_nll(&b, &s, &p.theta, &ETA, &p.omega, &doubled);
        assert!(la.is_finite() && lb.is_finite());
        eprintln!("T9a {a_err}: {la} vs {lb} (|Δ| {:.3e})", (la - lb).abs());
        assert!((la - lb).abs() <= 1e-12 * lb.abs().max(1.0), "{a_err}: {la} vs {lb}");
    }
}
