//! #2440: the fixed-η estimator family (SAEM, IMP, IMPMAP, VI) must score a model
//! whose prediction is legitimately negative as written.
//!
//! The fixed-η objective (`obs_nll_subject_into`) used to clamp every prediction to
//! `≥ 1e-12`, which FOCE/FOCEI/Laplace/AGQ never did. On an additive
//! baseline-minus-Emax model, where most predictions are negative, the four
//! fixed-η estimators were therefore optimizing a different objective: SAEM 445.9,
//! IMP 9119, IMPMAP 2025, VI drifting to the bounds, against FOCE's −10.72.
//!
//! The model is linear in η, so the marginal likelihood is Gaussian and closed form
//! (`exact_m2ll`). Each estimator's fitted point is scored by that oracle and compared
//! with FOCE's — a reference outside every engine under test, so a wrong OFV report
//! cannot mask a wrong fit and vice versa. Data are synthetic (seeded LCG +
//! Box–Muller), 30 subjects × 7 observations, 172 of 210 DV negative.
//!
//! Tier 3 (fits to convergence): gated on `slow-tests`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, Population, Subject};
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult};

const MODEL_SRC: &str = r"
[parameters]
  theta TVE0(1.0, -50.0, 100.0)
  theta TVEMAX(5.0, 0.1, 100.0)
  theta TVET50(1.0, 0.01, 100.0)
  omega ETA_E0 ~ 0.25
  sigma ADD ~ 0.25 (variance)

[individual_parameters]
  E0   = TVE0 + ETA_E0
  EMAX = TVEMAX
  ET50 = TVET50

[structural_model]
  EFF = EMAX * TIME / (ET50 + TIME)
  y   = E0 - EFF

[error_model]
  DV ~ additive(ADD)
";

const TIMES: [f64; 7] = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

/// The 64-bit LCG + Box–Muller of #2440's reproduction.
struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn pred(theta: &[f64], t: f64) -> f64 {
    theta[0] - theta[1] * t / (theta[2] + t)
}

fn population() -> Population {
    let mut rng = Lcg(12345);
    let truth = [1.0, 5.0, 1.0];
    let subjects = (0..30)
        .map(|i| {
            let eta = 0.5 * rng.normal();
            let observations = TIMES
                .iter()
                .map(|&t| pred(&truth, t) + eta + 0.5 * rng.normal())
                .collect();
            Subject {
                id: format!("{}", i + 1),
                obs_times: TIMES.to_vec(),
                observations,
                obs_cmts: vec![1; TIMES.len()],
                cens: vec![0; TIMES.len()],
                ..Default::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Exact marginal −2 log L (without `n·ln 2π`) at `(θ, ω, σ²)`. With `V = s2·I + ω·11ᵀ`:
/// `n ln s2 + ln(1 + nω/s2) + (Σr² − ω(Σr)²/(s2 + nω))/s2`.
fn exact_m2ll(pop: &Population, theta: &[f64], omega: f64, s2: f64) -> f64 {
    pop.subjects
        .iter()
        .map(|s| {
            let n = s.observations.len() as f64;
            let r: Vec<f64> = s
                .obs_times
                .iter()
                .zip(&s.observations)
                .map(|(&t, &y)| y - pred(theta, t))
                .collect();
            let sr: f64 = r.iter().sum();
            let srr: f64 = r.iter().map(|x| x * x).sum();
            n * s2.ln()
                + (1.0 + n * omega / s2).ln()
                + (srr - omega * sr * sr / (s2 + n * omega)) / s2
        })
        .sum()
}

fn exact_at_fit(pop: &Population, r: &FitResult) -> f64 {
    // `FitResult.sigma` is a standard deviation; square it.
    exact_m2ll(pop, &r.theta, r.omega[(0, 0)], r.sigma[0] * r.sigma[0])
}

fn model() -> CompiledModel {
    parse_model_string(MODEL_SRC).expect("#2440 model parses")
}

fn run(method: EstimationMethod, tweak: impl FnOnce(&mut FitOptions)) -> FitResult {
    let model = model();
    let pop = population();
    let mut opts = FitOptions {
        method,
        outer_maxiter: 300,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    tweak(&mut opts);
    fit(&model, &pop, &model.default_params, &opts)
        .unwrap_or_else(|e| panic!("{method:?} fit runs: {e}"))
}

/// FOCE's fitted point scored by the oracle: the reference every fixed-η estimator
/// must reach.
fn foce_exact() -> f64 {
    let pop = population();
    let r = run(EstimationMethod::Foce, |_| {});
    let exact = exact_at_fit(&pop, &r);
    assert!(
        (r.ofv - exact).abs() < 1e-3,
        "FOCE OFV {} must equal the exact marginal at its own fit {exact} (linear-in-η model)",
        r.ofv
    );
    exact
}

/// Common body: fit with `method`, score its point by the closed-form oracle, compare
/// with FOCE's. Under the old prediction floor the gap was hundreds to thousands.
fn assert_reaches_foce(method: EstimationMethod, tol: f64, tweak: impl FnOnce(&mut FitOptions)) {
    let pop = population();
    let negatives = pop
        .subjects
        .iter()
        .flat_map(|s| &s.observations)
        .filter(|&&y| y < 0.0)
        .count();
    assert!(
        negatives > 150,
        "fixture must be dominated by negative DV ({negatives})"
    );
    let reference = foce_exact();
    let r = run(method, tweak);
    let got = exact_at_fit(&pop, &r);
    eprintln!(
        "#2440 {method:?}: exact −2LL at fit {got:.6}, FOCE {reference:.6}, gap {:.3e}, \
         reported OFV {}, θ {:?}, ω {}, σ {}",
        got - reference,
        r.ofv,
        r.theta,
        r.omega[(0, 0)],
        r.sigma[0]
    );
    assert!(
        got.is_finite(),
        "{method:?}: exact −2LL at fit is not finite"
    );
    // FOCE is the exact MLE here, so a fixed-η fit can only sit at or above it.
    assert!(
        got - reference < tol && got - reference > -1e-3,
        "{method:?}: exact −2LL at its fit {got} vs FOCE {reference} (tol {tol})"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_reaches_the_foce_objective_on_negative_predictions() {
    assert_reaches_foce(EstimationMethod::Saem, TOL_SAEM, |o| o.saem_seed = Some(7));
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn imp_reaches_the_foce_objective_on_negative_predictions() {
    assert_reaches_foce(EstimationMethod::Imp, TOL_IMP, |o| o.imp_seed = Some(7));
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn impmap_reaches_the_foce_objective_on_negative_predictions() {
    assert_reaches_foce(EstimationMethod::Impmap, TOL_IMPMAP, |o| {
        o.imp_seed = Some(7)
    });
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn vi_reaches_the_foce_objective_on_negative_predictions() {
    assert_reaches_foce(EstimationMethod::Vi, TOL_VI, |o| o.vi_seed = Some(7));
}

// Gap above FOCE's exact −2LL, measured on the reference platform — Linux x86_64,
// glibc 2.41, `tools/linux-test.sh`, `--profile ci-fast`, seed 7 throughout: SAEM
// 9.60e-2, IMP 3.42e-4, IMPMAP 1.48e-3, VI 2.94e-4. Headroom: SAEM 5.2×, IMP 29×,
// IMPMAP 6.8×, VI 34×. (macOS arm64 differs in the stochastic estimators' digits —
// SAEM 7.66e-2, IMPMAP 3.76e-4 — and is not the basis for these bounds.) Under the
// restored 1e-12 prediction floor the same fits land hundreds to thousands of OFV
// units away (#2440: SAEM 468, IMP 9020, IMPMAP 2037, VI 1864, measured on macOS), so
// neither side is near the bound.
const TOL_SAEM: f64 = 0.5;
const TOL_IMP: f64 = 1e-2;
const TOL_IMPMAP: f64 = 1e-2;
const TOL_VI: f64 = 1e-2;
