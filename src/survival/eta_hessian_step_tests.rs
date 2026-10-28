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

/// The production step resolves each endpoint's η-Hessian to its truncation bias at every
/// supported prior width, and that bias is smooth: across 64 η-jitters of 1e-7 the error
/// barely moves. ω runs from tight (1e-4) through the measured 0.25 to the packed rail
/// (`OMEGA_CHOL_PACKED_UPPER = 6`, ω = e¹² ≈ 1.63e5), where a prior-SD step put the stencil
/// at η ± 1.6 and the Bernoulli η-Hessian 6.7% low.
///
/// Errors are relative to `H̃ = 1/ω + H`, the matrix whose log-determinant the objective
/// reads: an error δ in `H` moves `½ log|H̃|` by `½ δ/H̃`. At a tight prior the step shrinks
/// with the posterior SD and `H`'s own noise grows (2.5e-6 at ω = 1e-4), but `1/ω = 1e4`
/// swamps it. The Shi step fails both bounds by more than an order of magnitude.
#[test]
fn production_step_resolves_closed_form_eta_hessians_at_every_prior_width() {
    let rail = (2.0 * crate::estimation::parameterization::OMEGA_CHOL_PACKED_UPPER).exp();
    let mut worst = (0.0_f64, 0.0_f64);
    for c in cases() {
        for omega in [1e-4, 0.25, 1e2, rail] {
            let om = DMatrix::from_element(1, 1, omega);
            let steps_at = |e0: f64| {
                data_term_hessian_steps(
                    &c.f,
                    &[e0],
                    (c.f)(&[e0]),
                    &om,
                    DATA_TERM_HESSIAN_STEP_PER_SD,
                )
            };
            let err_at =
                |e0: f64| data_term_hessian_fd(&c.f, &[e0], &steps_at(e0))[(0, 0)] - (c.exact)(e0);
            let h_tilde = 1.0 / omega + (c.exact)(ETA_HAT);
            let errs: Vec<f64> = (0..64)
                .map(|j| err_at(ETA_HAT + j as f64 * 1e-7) / h_tilde)
                .collect();
            assert!(errs.iter().all(|e| e.is_finite()), "{}: {errs:?}", c.name);
            let rel = errs[0].abs();
            let lo = errs.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = errs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "{} ω = {omega:.3e}: step {:.3e}, bias/H̃ {rel:.3e}, noise spread/H̃ {:.3e}",
                c.name,
                steps_at(ETA_HAT)[0],
                hi - lo
            );
            worst = (worst.0.max(rel), worst.1.max(hi - lo));
            assert!(
                rel < 1e-5,
                "{} ω = {omega:e}: |H − exact|/H̃ = {rel:.3e}",
                c.name
            );
            assert!(
                hi - lo < 1e-8,
                "{} ω = {omega:e}: H̃ moves by {:.3e} (relative) across 1e-7 η-jitters — the \
                 stencil is roundoff, not curvature",
                c.name,
                hi - lo
            );
        }
    }
    eprintln!(
        "worst rel bias {:.3e}, worst noise spread {:.3e}",
        worst.0, worst.1
    );
}

/// #2437 review: η's scale is the model's choice. `logit = TH + K·ETA` with `ω = ω₀/K²`
/// is `logit = TH + ETA` with `ω₀`, and its η-Hessian is K² times the unit one. The
/// posterior-SD step keeps the stencil the same on the linear predictor, so the scaled
/// Hessian's relative error is the unit one's at every K, at a tight and at a wide prior.
/// An absolute 1e-3 step fails both ends: at K = 1000 it samples the logit ±2 and comes out
/// ~10% low; at K = 1e-3 the curvature (~1.8e-6) drowns in roundoff ε·|f|/h².
#[test]
fn step_is_invariant_to_the_eta_loading() {
    const YS: [u8; 8] = [1, 0, 1, 1, 0, 1, 1, 1];
    let (th, z) = (0.4, 0.3);
    for omega0 in [0.25, 1e4] {
        let mut rels = Vec::new();
        for k in [1e-3, 1.0, 1e3] {
            let f = |e: &[f64]| binary_nll_j(J::c(k * e[0]), th, &YS).v;
            // d²/dη² of g(K·η) = K²·g''(K·η).
            let exact = k * k * binary_nll_j(J::var(z), th, &YS).dd;
            let omega = DMatrix::from_element(1, 1, omega0 / (k * k));
            let eta = [z / k];
            let steps =
                data_term_hessian_steps(f, &eta, f(&eta), &omega, DATA_TERM_HESSIAN_STEP_PER_SD);
            let rel = (data_term_hessian_fd(f, &eta, &steps)[(0, 0)] - exact).abs() / exact;
            let abs_rel =
                (data_term_hessian_fd(f, &eta, &[DATA_TERM_HESSIAN_STEP])[(0, 0)] - exact).abs()
                    / exact;
            eprintln!(
                "ω₀ = {omega0:e}, K = {k:e}: step·K {:.4e}, rel err {rel:.3e}, absolute-step \
                 rel err {abs_rel:.3e}",
                steps[0] * k
            );
            rels.push((k, rel, abs_rel));
        }
        let unit = rels[1].1;
        for &(k, rel, abs_rel) in &rels {
            assert!(rel < 1e-6, "ω₀ = {omega0:e}, K = {k:e}: rel err {rel:.3e}");
            assert!(
                (rel - unit).abs() < 1e-8,
                "ω₀ = {omega0:e}, K = {k:e}: rel err {rel:.3e} vs unit loading {unit:.3e}"
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
}

/// The rule on a quadratic data term `a·η²/2`, whose pilot is exact: the step is
/// `c / √(1/ω + a)`. The data's curvature takes over from a wide prior, the prior from a
/// flat data term; a concave pilot counts as flat, a sentinel-poisoned pilot is ignored,
/// and with neither scale the step is the absolute fallback.
#[test]
fn step_is_the_posterior_sd_rule() {
    let c = DATA_TERM_HESSIAN_STEP_PER_SD;
    let quad = |a: f64| move |e: &[f64]| 0.5 * a * e[0] * e[0];
    let step = |a: f64, omega: f64| {
        let f = quad(a);
        data_term_hessian_steps(
            f,
            &[0.3],
            f(&[0.3]),
            &DMatrix::from_element(1, 1, omega),
            DATA_TERM_HESSIAN_STEP_PER_SD,
        )[0]
    };
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-9 * want;
    assert!(close(step(2.0, 0.25), c / (4.0f64 + 2.0).sqrt()));
    assert!(
        close(step(2.0, 1e5), c / (1e-5f64 + 2.0).sqrt()),
        "wide prior: data's scale"
    );
    assert!(close(step(0.0, 0.25), c * 0.5), "flat data: prior SD");
    assert!(
        close(step(-3.0, 0.25), c * 0.5),
        "concave pilot counts as flat"
    );
    // No prior: no scale to start the pilot ladder from, so the absolute fallback it starts
    // at caps the step; a data term sharp enough to want less gets its own scale.
    assert!(
        close(step(2.0, 0.0), DATA_TERM_HESSIAN_STEP),
        "no prior: capped at the start"
    );
    assert!(close(step(1e6, 0.0), c / 1e3), "no prior, sharp data term");
    assert_eq!(step(0.0, 0.0), DATA_TERM_HESSIAN_STEP, "no scale at all");
    assert_eq!(step(0.0, f64::NAN), DATA_TERM_HESSIAN_STEP);
    let sentinel = |e: &[f64]| if e[0] == 0.3 { 1.0 } else { 1e20 };
    let om = DMatrix::from_element(1, 1, 0.25);
    assert!(close(
        data_term_hessian_steps(sentinel, &[0.3], 1.0, &om, DATA_TERM_HESSIAN_STEP_PER_SD)[0],
        c * 0.5
    ));
    // Per axis: the second axis has no ω entry and a flat data term.
    let two = data_term_hessian_steps(
        |e: &[f64]| e[0] * e[0],
        &[0.3, 0.0],
        0.09,
        &om,
        DATA_TERM_HESSIAN_STEP_PER_SD,
    );
    assert!(close(two[0], c / (4.0f64 + 2.0).sqrt()));
    assert_eq!(two[1], DATA_TERM_HESSIAN_STEP);
    assert!(
        data_term_hessian_steps(quad(1.0), &[], 0.0, &om, DATA_TERM_HESSIAN_STEP_PER_SD).is_empty()
    );
}

/// The time-inhomogeneous CTMM takes `DATA_TERM_HESSIAN_STEP_PER_SD_INTEGRATED`: on
/// `examples/ctmm_pd_2state.ferx` the η-Hessian at that step holds when the step is halved
/// or doubled, so the stencil is reading curvature, not the occupancy integration's error.
/// The exact evaluators' step on the same subjects lands off that plateau — the straddle,
/// asserted, so the wider constant cannot silently become unnecessary.
#[cfg(feature = "markov")]
#[test]
fn inhomogeneous_ctmm_step_sits_on_the_plateau() {
    let run = crate::prepare_run(
        "examples/ctmm_pd_2state.ferx",
        Some("data/ctmm_pd_2state.csv"),
    )
    .expect("example loads");
    let model = &run.parsed.model;
    let theta = &run.init_params.theta;
    let omega = &run.init_params.omega.matrix;
    let (mut worst_plateau, mut worst_exact) = (0.0_f64, 0.0_f64);
    for si in [0usize, 7, 23, 41] {
        let s = &run.population.subjects[si];
        let f = |e: &[f64]| crate::markov::endpoint::ctmm_subject_nll(model, s, theta, e);
        let eta = [0.1];
        let f0 = f(&eta);
        let h = |steps: &[f64]| data_term_hessian_fd(f, &eta, steps)[(0, 0)];
        let prod =
            data_term_hessian_steps(f, &eta, f0, omega, DATA_TERM_HESSIAN_STEP_PER_SD_INTEGRATED);
        let exact_class =
            data_term_hessian_steps(f, &eta, f0, omega, DATA_TERM_HESSIAN_STEP_PER_SD);
        let at = h(&prod);
        assert!(at.is_finite() && at > 0.0, "subject {si}: H {at}");
        let plateau = [0.5, 2.0]
            .iter()
            .map(|&m| (h(&[prod[0] * m]) - at).abs() / at)
            .fold(0.0_f64, f64::max);
        let off = (h(&exact_class) - at).abs() / at;
        eprintln!(
            "subject {si}: step {:.3e} H {at:.6} plateau {plateau:.3e}; exact-class step \
             {:.3e} off by {off:.3e}",
            prod[0], exact_class[0]
        );
        worst_plateau = worst_plateau.max(plateau);
        worst_exact = worst_exact.max(off);
    }
    assert!(
        worst_plateau < 1e-2,
        "H moves {worst_plateau:.3e} across ½×–2× the step"
    );
    assert!(
        worst_exact > 1e-2,
        "the exact evaluators' step is already on the plateau ({worst_exact:.3e})"
    );
}

/// #2437 review round 4: a pilot whose stencil meets a sentinel must not read it as
/// curvature. A quadratic data term `a·η²/2` that repels (`1e20`) beyond |Δη| = 0.5, at
/// the packed ω rail on the integrated constant: the first pilot is 40 wide, both of its
/// points are sentinels, and `1e20/(4·40²) ≈ 1.6e16` passes a guard on the second
/// difference, which made the step ~2e-9. Checked on the raw values, the pilot halves
/// until its points are real and recovers `a`. Also on the exact constant, and with a
/// sentinel on one side only.
#[test]
fn pilot_retries_past_sentinels_instead_of_reading_them_as_curvature() {
    let rail = (2.0 * crate::estimation::parameterization::OMEGA_CHOL_PACKED_UPPER).exp();
    let om = DMatrix::from_element(1, 1, rail);
    let a = 2.0;
    let eta0 = 0.3;
    for per_sd in [
        DATA_TERM_HESSIAN_STEP_PER_SD_INTEGRATED,
        DATA_TERM_HESSIAN_STEP_PER_SD,
    ] {
        let want = per_sd / (1.0 / rail + a).sqrt();
        for one_sided in [false, true] {
            let f = |e: &[f64]| {
                let d = e[0] - eta0;
                if d > 0.5 || (!one_sided && d < -0.5) {
                    1e20
                } else {
                    0.5 * a * e[0] * e[0]
                }
            };
            let got = data_term_hessian_steps(f, &[eta0], f(&[eta0]), &om, per_sd)[0];
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "per_sd {per_sd}, one-sided {one_sided}: step {got:.3e}, want {want:.3e}"
            );
        }
    }
    // A data term that is a sentinel at every perturbed point has no curvature to read:
    // the prior's scale is all there is.
    let wall = |e: &[f64]| if e[0] == eta0 { 1.0 } else { 1e20 };
    let got = data_term_hessian_steps(wall, &[eta0], 1.0, &om, DATA_TERM_HESSIAN_STEP_PER_SD)[0];
    assert!((got - DATA_TERM_HESSIAN_STEP_PER_SD * rail.sqrt()).abs() <= 1e-9 * got);
}

/// The integrated CTMM across the supported ω range, 0.01 to the packed rail (≈ 1.63e5), on
/// eight subjects of `examples/ctmm_pd_2state.ferx` including the weakly informative ones
/// (H ≈ 0.03–0.09). Measured against the plateau value at 3e-2, relative to `H̃ = 1/ω + H`:
/// worst 8.5e-3, which is the plateau's own spread on those subjects (subject 12 reads
/// 0.03366 / 0.03345 / 0.03417 at 1.5e-2 / 3e-2 / 6e-2). The smallest step was 6.5e-2 of
/// the posterior-SD step: the ladder's cap at a weak subject under a wide prior, whose
/// posterior SD (~5) is far wider than the span its curvature holds over.
/// A vague prior's first pilot spans η ± 80. Before the pilot ladder, its saturated (even
/// concave) curvature set steps of 1.5–40 and errors of ~100% of H̃. Before the raw-value
/// sentinel check, a sentinel at that width read as curvature and set a ~1e-9 step.
#[cfg(feature = "markov")]
#[test]
fn inhomogeneous_ctmm_step_holds_across_omega_widths() {
    let run = crate::prepare_run(
        "examples/ctmm_pd_2state.ferx",
        Some("data/ctmm_pd_2state.csv"),
    )
    .expect("example loads");
    let model = &run.parsed.model;
    let theta = &run.init_params.theta;
    let rail = (2.0 * crate::estimation::parameterization::OMEGA_CHOL_PACKED_UPPER).exp();
    // Smallest step as a fraction of the posterior-SD step the plateau value implies: a
    // collapse (the ~1e-9 step of a sentinel read as curvature) is ~1e-7 of it.
    let (mut worst, mut smallest) = (0.0_f64, f64::INFINITY);
    for si in [0usize, 3, 7, 12, 23, 30, 41, 50] {
        let s = &run.population.subjects[si];
        let f = |e: &[f64]| crate::markov::endpoint::ctmm_subject_nll(model, s, theta, e);
        let eta = [0.1];
        let f0 = f(&eta);
        let reference = data_term_hessian_fd(f, &eta, &[3e-2])[(0, 0)];
        for omega in [0.01, 0.09, 1.0, 1e2, rail] {
            let om = DMatrix::from_element(1, 1, omega);
            let step =
                data_term_hessian_steps(f, &eta, f0, &om, DATA_TERM_HESSIAN_STEP_PER_SD_INTEGRATED)
                    [0];
            let hess = data_term_hessian_fd(f, &eta, &[step])[(0, 0)];
            let rel = (hess - reference).abs() / (1.0 / omega + reference);
            eprintln!(
                "subject {si} ω = {omega:.3e}: step {step:.3e}, H {hess:.6}, off plateau \
                 {rel:.3e} of H̃"
            );
            assert!(rel.is_finite(), "subject {si}, ω = {omega:e}: H {hess}");
            worst = worst.max(rel);
            let posterior_step =
                DATA_TERM_HESSIAN_STEP_PER_SD_INTEGRATED / (1.0 / omega + reference).sqrt();
            smallest = smallest.min(step / posterior_step);
        }
    }
    assert!(
        worst < 1.5e-2,
        "η-Hessian {worst:.3e} of H̃ off its plateau value"
    );
    eprintln!("worst {worst:.3e} of H̃; smallest step {smallest:.3e} of the posterior-SD step");
    assert!(
        smallest >= 1e-2,
        "step collapsed to {smallest:.3e} of the posterior-SD step"
    );
}
