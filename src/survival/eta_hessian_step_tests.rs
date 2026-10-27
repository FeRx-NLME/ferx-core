//! #2437: the step of the data-term η-Hessian stencil, measured against closed forms.
//!
//! Three endpoints at η̂ = 0.3, each through its production evaluator where one exists
//! (`tte_data_term` for the Weibull hazard, `markov::ctmm_data_term`'s Padé `expm` for the
//! two-state CTMM) and against an exact second derivative from a second-order jet.
//! Absolute error of `data_term_hessian_fd` vs the exact value (≈ 1.77 / 0.79 / 1.04),
//! and its roundoff noise (sd over 64 η-jitters of 1e-7), macOS arm64:
//!
//! | step | binary err / noise | Weibull err / noise | CTMM err / noise |
//! |---|---|---|---|
//! | Shi (≈ 5e-6) | 5.4e-6 / 1.1e-5 | 2.6e-6 / 7.0e-6 | 1.2e-5 / 9.9e-6 |
//! | 1e-4 | 4.3e-8 / 2.8e-8 | 6.3e-9 / 9.6e-9 | 7.9e-8 / 4.1e-8 |
//! | 3e-4 | 1.9e-8 / 3.0e-9 | 2.3e-8 / 9.8e-10 | 1.7e-8 / 4.5e-9 |
//! | **1e-3** | 2.0e-7 / 2.8e-10 | 2.6e-7 / 9.9e-11 | 1.5e-7 / 3.9e-10 |
//! | 3e-3 | 1.8e-6 / 3.7e-11 | 2.4e-6 / 1.3e-11 | 1.3e-6 / 4.6e-11 |
//!
//! The bias column grows as h² and is smooth in θ; the noise column falls as 1/h² and is
//! what makes the objective rough. See `DATA_TERM_HESSIAN_STEP_PER_SD` for the choice.

use super::*;

/// Second-order forward jet in one variable: value, first and second derivative.
#[derive(Clone, Copy, Debug)]
struct J {
    v: f64,
    d: f64,
    dd: f64,
}

impl J {
    fn var(x: f64) -> Self {
        J {
            v: x,
            d: 1.0,
            dd: 0.0,
        }
    }
    fn c(x: f64) -> Self {
        J {
            v: x,
            d: 0.0,
            dd: 0.0,
        }
    }
    fn add(self, o: J) -> J {
        J {
            v: self.v + o.v,
            d: self.d + o.d,
            dd: self.dd + o.dd,
        }
    }
    fn sub(self, o: J) -> J {
        J {
            v: self.v - o.v,
            d: self.d - o.d,
            dd: self.dd - o.dd,
        }
    }
    fn mul(self, o: J) -> J {
        J {
            v: self.v * o.v,
            d: self.d * o.v + self.v * o.d,
            dd: self.dd * o.v + 2.0 * self.d * o.d + self.v * o.dd,
        }
    }
    fn recip(self) -> J {
        let v = self.v;
        J {
            v: 1.0 / v,
            d: -self.d / (v * v),
            dd: -self.dd / (v * v) + 2.0 * self.d * self.d / (v * v * v),
        }
    }
    fn div(self, o: J) -> J {
        self.mul(o.recip())
    }
    fn exp(self) -> J {
        let e = self.v.exp();
        J {
            v: e,
            d: e * self.d,
            dd: e * (self.dd + self.d * self.d),
        }
    }
    fn ln(self) -> J {
        let v = self.v;
        J {
            v: v.ln(),
            d: self.d / v,
            dd: self.dd / v - self.d * self.d / (v * v),
        }
    }
}

// Binary endpoint: logit p = th + eta, observations ys.
fn binary_nll_j(eta: J, th: f64, ys: &[u8]) -> J {
    let p = J::c(1.0).div(J::c(1.0).add(J::c(-th).sub(eta).exp()));
    let q = J::c(1.0).sub(p);
    let mut s = J::c(0.0);
    for &y in ys {
        s = s.sub(if y == 1 { p.ln() } else { q.ln() });
    }
    s
}

// Two-state homogeneous CTMM: q01 = a·exp(eta), q10 = b.
#[cfg(feature = "markov")]
fn ctmm_nll_j(eta: J, a: f64, b: f64, obs: &[(f64, usize)]) -> J {
    let u = J::c(a).mul(eta.exp());
    let bj = J::c(b);
    let s = u.add(bj);
    let mut nll = J::c(0.0);
    for w in obs.windows(2) {
        let dt = w[1].0 - w[0].0;
        let e = J::c(0.0).sub(s.mul(J::c(dt))).exp();
        let one_m = J::c(1.0).sub(e);
        let p = match (w[0].1, w[1].1) {
            (0, 0) => bj.add(u.mul(e)).div(s),
            (0, 1) => u.mul(one_m).div(s),
            (1, 0) => bj.mul(one_m).div(s),
            _ => u.add(bj.mul(e)).div(s),
        };
        nll = nll.sub(p.ln());
    }
    nll
}

const ETA_HAT: f64 = 0.3;

/// One endpoint: the data term `f(η)` and its exact `f''(η)`.
struct Case {
    name: &'static str,
    f: Box<dyn Fn(&[f64]) -> f64>,
    exact: Box<dyn Fn(f64) -> f64>,
    /// The closed form's value, when `f` is a production evaluator rather than the jet.
    value: Option<Box<dyn Fn(f64) -> f64>>,
}

fn cases() -> Vec<Case> {
    const YS: [u8; 8] = [1, 0, 1, 1, 0, 1, 1, 1];
    const TH_B: f64 = 0.4;
    let binary = Case {
        name: "binary",
        f: Box::new(|e: &[f64]| binary_nll_j(J::c(e[0]), TH_B, &YS).v),
        exact: Box::new(|e0: f64| binary_nll_j(J::var(e0), TH_B, &YS).dd),
        value: None,
    };

    // Weibull, scale 10, shape 1.5, η on the log hazard ratio, one exact event at t = 7:
    // −log L = (t/s)^k·e^η − η − const, so f''(η) = (t/s)^k·e^η.
    let recs = vec![ObsRecord::Event {
        time: 7.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 2,
    }];
    let param_fn: crate::types::HazardParamFn =
        Box::new(|th: &[f64], eta: &[f64], _c: &HashMap<String, f64>| vec![th[0], th[1], eta[0]]);
    let hazard = HazardSpec::Analytic {
        family: HazardFamily::Weibull,
        param_fn,
    };
    let weibull = Case {
        name: "tte_weibull",
        f: Box::new(move |e: &[f64]| {
            tte_data_term(
                &recs,
                &hazard,
                TteRecurrence::Single,
                &[10.0, 1.5],
                e,
                &HashMap::new(),
            )
        }),
        exact: Box::new(|e0: f64| (7.0f64 / 10.0).powf(1.5) * e0.exp()),
        // −log h(t) = −log(k/s·(t/s)^(k−1)) − η, + H(t) = (t/s)^k·e^η.
        value: Some(Box::new(|e0: f64| {
            let (k, t, s) = (1.5f64, 7.0f64, 10.0f64);
            (t / s).powf(k) * e0.exp() - e0 - ((k / s) * (t / s).powf(k - 1.0)).ln()
        })),
    };

    #[allow(unused_mut)]
    let mut out = vec![binary, weibull];

    #[cfg(feature = "markov")]
    {
        // Two states, q01 = 0.3·e^η, q10 = 0.5; every transition kind and a stay.
        const OBS: [(f64, usize); 8] = [
            (0.0, 0),
            (1.0, 0),
            (2.5, 1),
            (3.0, 1),
            (5.0, 0),
            (6.0, 1),
            (8.0, 1),
            (10.0, 0),
        ];
        let so: Vec<crate::markov::StateObs> = OBS
            .iter()
            .map(|&(time, state)| crate::markov::StateObs { time, state })
            .collect();
        out.push(Case {
            name: "ctmm",
            f: Box::new(move |e: &[f64]| {
                let u = 0.3 * e[0].exp();
                let q = DMatrix::from_row_slice(2, 2, &[-u, u, 0.5, -0.5]);
                crate::markov::ctmm_data_term(&q, &so).expect("valid generator")
            }),
            exact: Box::new(|e0: f64| ctmm_nll_j(J::var(e0), 0.3, 0.5, &OBS).dd),
            value: Some(Box::new(|e0: f64| ctmm_nll_j(J::c(e0), 0.3, 0.5, &OBS).v)),
        });
    }
    out
}

/// The jet is the oracle here, so it has to be the same function the production
/// evaluator scores: pin the value side before trusting the second derivative.
#[test]
fn closed_forms_match_the_production_evaluators() {
    for c in cases() {
        let Some(value) = &c.value else { continue };
        let (prod, jet) = ((c.f)(&[ETA_HAT]), value(ETA_HAT));
        assert!(
            (prod - jet).abs() < 1e-12 * prod.abs().max(1.0),
            "{}: production {prod} vs closed form {jet}",
            c.name
        );
    }
}

/// The production step resolves each endpoint's η-Hessian to its truncation bias, and
/// that bias is smooth: across 64 η-jitters of 1e-7 the error barely moves. Measured at
/// the step of `ω = 0.25`, 1e-3 (table above): worst relative bias 3.3e-7 (Weibull),
/// worst noise sd 3.9e-10 (CTMM). The Shi step fails both bounds by more than an order of
/// magnitude — its error is 1e-5 and almost all noise.
#[test]
fn production_step_resolves_closed_form_eta_hessians_smoothly() {
    let steps = data_term_hessian_steps(&DMatrix::from_element(1, 1, 0.25), 1);
    assert_eq!(steps, vec![DATA_TERM_HESSIAN_STEP]);
    for c in cases() {
        let err_at = |e0: f64| data_term_hessian_fd(&c.f, &[e0], &steps)[(0, 0)] - (c.exact)(e0);
        let exact = (c.exact)(ETA_HAT);
        let errs: Vec<f64> = (0..64).map(|j| err_at(ETA_HAT + j as f64 * 1e-7)).collect();
        assert!(errs.iter().all(|e| e.is_finite()), "{}: {errs:?}", c.name);
        let rel = errs[0].abs() / exact.abs();
        let lo = errs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = errs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "{}: rel bias {rel:.3e}, noise spread {:.3e}",
            c.name,
            hi - lo
        );
        assert!(rel < 1e-6, "{}: |H − exact|/exact = {rel:.3e}", c.name);
        assert!(
            hi - lo < 1e-8,
            "{}: the η-Hessian moves by {:.3e} across 1e-7 η-jitters — the stencil is \
             roundoff, not curvature",
            c.name,
            hi - lo
        );
    }
}

/// #2437 review: η's scale is the model's choice. `logit = TH + K·ETA` with `ω = ω₀/K²`
/// is `logit = TH + ETA` with `ω₀`, and its η-Hessian is K² times the unit one. The step
/// per prior SD keeps the stencil the same on the linear predictor, so the scaled
/// Hessian's relative error is the unit one's at every K. An absolute 1e-3 step fails both
/// ends: at K = 1000 it samples the logit ±2 and comes out 13% low; at K = 1e-3 the
/// curvature (~1.8e-6) drowns in roundoff ε·|f|/h².
#[test]
fn step_per_sd_is_invariant_to_the_eta_loading() {
    const YS: [u8; 8] = [1, 0, 1, 1, 0, 1, 1, 1];
    let (th, omega0, z) = (0.4, 0.25, 0.3);
    let mut rels = Vec::new();
    for k in [1e-3, 1.0, 1e3] {
        let f = |e: &[f64]| binary_nll_j(J::c(k * e[0]), th, &YS).v;
        // d²/dη² of g(K·η) = K²·g''(K·η).
        let exact = k * k * binary_nll_j(J::var(k * (z / k)), th, &YS).dd;
        let omega = DMatrix::from_element(1, 1, omega0 / (k * k));
        let steps = data_term_hessian_steps(&omega, 1);
        let rel = (data_term_hessian_fd(f, &[z / k], &steps)[(0, 0)] - exact).abs() / exact;
        let abs_rel =
            (data_term_hessian_fd(f, &[z / k], &[DATA_TERM_HESSIAN_STEP])[(0, 0)] - exact).abs()
                / exact;
        eprintln!("K = {k:e}: per-SD rel err {rel:.3e}, absolute-step rel err {abs_rel:.3e}");
        rels.push((k, rel, abs_rel));
    }
    let unit = rels[1].1;
    for &(k, rel, abs_rel) in &rels {
        assert!(rel < 1e-6, "K = {k:e}: per-SD step rel err {rel:.3e}");
        assert!(
            (rel - unit).abs() < 1e-8,
            "K = {k:e}: per-SD rel err {rel:.3e} vs unit loading {unit:.3e}"
        );
        // The straddle: the absolute step this replaces fails the same bound off K = 1.
        if k != 1.0 {
            assert!(
                abs_rel > 1e-4,
                "K = {k:e}: absolute step rel err {abs_rel:.3e}"
            );
        }
    }
}

/// One step per prior standard deviation, from ω's diagonal alone — not from the data term
/// or η̂, so it cannot pick up their noise — with the documented fallback.
#[test]
fn steps_scale_with_the_prior_sd() {
    let omega = DMatrix::from_row_slice(
        4,
        4,
        &[
            0.25, 0.1, 0.0, 0.0, //
            0.1, 0.01, 0.0, 0.0, //
            0.0, 0.0, 1e4, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ],
    );
    let steps = data_term_hessian_steps(&omega, 5);
    assert_eq!(steps[0], DATA_TERM_HESSIAN_STEP);
    assert!((steps[1] - DATA_TERM_HESSIAN_STEP_PER_SD * 0.1).abs() < 1e-18);
    assert!(
        (steps[2] - DATA_TERM_HESSIAN_STEP_PER_SD * 100.0).abs() < 1e-15,
        "ω = 1e4"
    );
    assert_eq!(
        steps[3], DATA_TERM_HESSIAN_STEP,
        "ω = 0: no scale, fallback"
    );
    assert_eq!(steps[4], DATA_TERM_HESSIAN_STEP, "axis past ω: fallback");
    let bad = DMatrix::from_element(1, 1, f64::NAN);
    assert_eq!(
        data_term_hessian_steps(&bad, 1),
        vec![DATA_TERM_HESSIAN_STEP]
    );
    assert!(data_term_hessian_steps(&omega, 0).is_empty());
}
