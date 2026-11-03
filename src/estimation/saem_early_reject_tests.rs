//! The MH early-rejection gate (#2608): `log_u` is drawn before the proposal is
//! evaluated, and a proposal whose prior plus a prediction-free lower bound on its
//! data term already clears the acceptance threshold is rejected without a
//! prediction.
//!
//! The gate's whole claim is that it changes **no** decision, so the tests that
//! matter are A/B runs of the real kernels with the gate on and off, compared on
//! raw bits — a tolerance would be a test that has stopped testing, since SAEM's
//! per-iteration state feeds the next iteration. The gate-off arm is exactly the
//! pre-#2608 kernel: the same `log_u` draw at the same position of the stream,
//! then the full evaluation.
//!
//! An equality A/B is blind to the gate never firing (both arms then take the full
//! path), so every A/B also asserts the gate **did** fire on a material share of
//! proposals. And because the A/B can only catch a too-loose bound when the
//! loosened gate rejects a proposal the full evaluation would have accepted, the
//! bound itself is also pinned directly against a brute-force minimum.
//!
//! Mutations run against this file (each reddened the tests named):
//! * `gaussian_row_term_lower_bound` returning `best + 0.3` —
//!   `row_bound_never_exceeds_the_term_it_bounds` and the combined-error A/B.
//! * `mh_early_rejects` dropping `log_det_omega` from `lb` (`log|Ω| < 0` here, so
//!   the bound is too high by `−½·log|Ω|`) — all three kernel A/Bs.
//! * `prepared_data_term_lower_bound` without the `iiv_on_ruv` or the dense-`R`
//!   opt-out — `the_data_bound_declines_what_it_cannot_bound`.
//! * the split loop bypassed (the proportional bound collapses to the useless
//!   `ln MIN_POSITIVE` floor) — `row_bound_is_tight` and the proportional A/B's
//!   firing assertion.
//! * the rounding margin removed — `the_gate_declines_the_rail_and_respects_its_margin`.
//!
//! One known survivor: zeroing the componentwise kernel's prior term makes its
//! bound *weaker*, never wrong, so no decision changes and nothing reddens. That
//! is a lost speed-up, not a correctness defect, and the A/Bs are not built to see
//! it.

use super::{mh_early_rejects, mh_steps, mh_steps_componentwise, MhScratch};
use crate::parser::model_parser::parse_model_string;
use crate::stats::likelihood::{
    gaussian_row_term_lower_bound, individual_nll, individual_nll_prepared,
    prepared_data_term_lower_bound, quad_form, DataTermLowerBound,
};
use crate::stats::residual_error::guard_variance;
use crate::types::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ─── Fixtures ─────────────────────────────────────────────────────────────

fn model(error_line: &str, sigmas: &str) -> CompiledModel {
    parse_model_string(&format!(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
{sigmas}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  {error_line}
"
    ))
    .expect("fixture model parses")
}

fn additive_model() -> CompiledModel {
    model("DV ~ additive(ADD)", "  sigma ADD ~ 0.4 (sd)\n")
}

fn proportional_model() -> CompiledModel {
    model("DV ~ proportional(PROP)", "  sigma PROP ~ 0.15 (sd)\n")
}

fn combined_model() -> CompiledModel {
    model(
        "DV ~ combined(PROP, ADD)",
        "  sigma PROP ~ 0.12 (sd)\n  sigma ADD ~ 0.2 (sd)\n",
    )
}

/// Sparse subjects (three samples each), observations scattered ±25 % around the
/// typical profile `5·exp(−0.15 t)` with a different pattern per subject — sparse,
/// so the prior carries enough of the objective for tail proposals to clear the
/// threshold on the prior alone, which is the regime the gate exists for.
fn sparse_population(n: usize) -> Population {
    let times = [0.5, 4.0, 12.0];
    let subjects = (0..n)
        .map(|i| {
            let f = i as f64;
            let obs = times
                .iter()
                .enumerate()
                .map(|(k, &t)| {
                    let wiggle = ((f * 1.7 + k as f64 * 2.3).sin()) * 0.25;
                    5.0 * (-0.15_f64 * t).exp() * (1.0 + wiggle)
                })
                .collect();
            Subject {
                id: format!("{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: times.to_vec(),
                observations: obs,
                obs_cmts: vec![1; times.len()],
                cens: vec![0; times.len()],
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

/// One subject's chain: a block run then a componentwise sweep, from η = 0, at a
/// step scale broad enough that a large share of proposals land in the prior's
/// tail. Returns the raw bits of everything the chain produces, plus the number of
/// gate rejections.
fn run_chain(
    model: &CompiledModel,
    subject: &Subject,
    seed: u64,
    gate_off: bool,
) -> (Vec<u64>, usize) {
    let p = &model.default_params;
    let (theta, omega, sigma) = (&p.theta, &p.omega, &p.sigma.values);
    let n_eta = model.n_eta;
    let mut scratch = MhScratch::default();
    scratch.begin_subject(model, subject, theta, n_eta);
    scratch.early_reject_off = gate_off;
    let mut eta = vec![0.0; n_eta];
    let nll0 = individual_nll(model, subject, theta, &eta, omega, sigma);
    let mut rng = StdRng::seed_from_u64(seed);
    let (acc_b, nll_b) = mh_steps(
        &mut eta,
        nll0,
        subject,
        model,
        theta,
        omega,
        sigma,
        3.0,
        None,
        &mut rng,
        40,
        &mut scratch,
        None,
        None,
    );
    let cw_sd: Vec<f64> = (0..n_eta).map(|j| omega.matrix[(j, j)].sqrt()).collect();
    let (acc_c, n_prop_c, nll_c) = mh_steps_componentwise(
        &mut eta,
        nll_b,
        subject,
        model,
        theta,
        omega,
        sigma,
        &[3.5, 2.5],
        &cw_sd,
        &mut rng,
        15,
        &mut scratch,
        None,
        None,
    );
    let mut bits: Vec<u64> = eta.iter().map(|x| x.to_bits()).collect();
    bits.extend([
        acc_b as u64,
        nll_b.to_bits(),
        n_prop_c as u64,
        nll_c.to_bits(),
    ]);
    bits.extend(acc_c.iter().map(|&a| a as u64));
    // The next draw too: a gate that consumed (or skipped) a uniform would leave the
    // stream at a different position even if every decision so far agreed.
    bits.push(rng.random::<u64>());
    (bits, scratch.early_rejections)
}

/// The A/B every kernel test runs: the gate must change no bit of the chain, and
/// must have fired on at least `min_share` of all proposals (55 per subject).
fn assert_gate_is_bit_identical(model: &CompiledModel, min_share: f64) {
    let pop = sparse_population(16);
    let mut fired = 0usize;
    let mut total = 0usize;
    for (i, subject) in pop.subjects.iter().enumerate() {
        let seed = 2608 + i as u64;
        let (on, n_fired) = run_chain(model, subject, seed, false);
        let (off, n_off) = run_chain(model, subject, seed, true);
        assert_eq!(n_off, 0, "the gate fired with the gate switched off");
        assert_eq!(
            on, off,
            "subject {}: the early-rejection gate changed the chain (gate fired {n_fired} times)",
            subject.id
        );
        fired += n_fired;
        total += 40 + 15 * model.n_eta;
    }
    let share = fired as f64 / total as f64;
    assert!(
        share >= min_share,
        "the gate fired on {fired} of {total} proposals ({:.1} %) — below the {:.0} % \
         this fixture is built to reach, so the A/B above is not seeing the gate",
        100.0 * share,
        100.0 * min_share
    );
}

// ─── The kernels ──────────────────────────────────────────────────────────

#[test]
fn early_rejection_is_bit_identical_under_additive_error() {
    assert_gate_is_bit_identical(&additive_model(), 0.10);
}

#[test]
fn early_rejection_is_bit_identical_under_proportional_error() {
    assert_gate_is_bit_identical(&proportional_model(), 0.10);
}

#[test]
fn early_rejection_is_bit_identical_under_combined_error() {
    assert_gate_is_bit_identical(&combined_model(), 0.10);
}

// ─── The bounds ───────────────────────────────────────────────────────────

/// The scored term at prediction `f`, written the way the scoring loop scores it.
fn row_term(y: f64, f: f64, a: f64, b: f64) -> f64 {
    let v = guard_variance(f * f * b + a);
    (y - f) * (y - f) / v + v.ln()
}

/// Brute-force minimum over a grid that covers both signs, the guarded region near
/// `f = 0`, and a dense band around `y`.
fn grid_min(y: f64, a: f64, b: f64) -> f64 {
    let mut fs = vec![0.0];
    for e in -320..=40 {
        let mag = 10f64.powf(e as f64 / 4.0);
        fs.push(mag);
        fs.push(-mag);
    }
    for k in 0..=4000 {
        let f = y * (k as f64 / 1000.0);
        fs.push(f);
    }
    fs.iter()
        .map(|&f| row_term(y, f, a, b))
        .fold(f64::INFINITY, |m, t| {
            assert!(!t.is_nan(), "grid term NaN at y={y} a={a} b={b}");
            m.min(t)
        })
}

const ROW_CASES: [(f64, f64, f64); 12] = [
    (3.0, 0.16, 0.0),
    (3.0, 0.0, 0.0225),
    (0.002, 0.0, 0.0225),
    (450.0, 0.0, 0.09),
    (-2.0, 0.0, 0.04),
    (3.0, 0.04, 0.0144),
    (0.05, 0.04, 0.0144),
    (800.0, 0.04, 0.0144),
    (3.0, 1e-30, 0.0144),
    (1e-3, 1e-14, 1e-14),
    (0.0, 0.25, 0.01),
    (7.0, 0.0, 1e-300),
];

/// The bound is a bound: never above the term at any prediction. This is what a
/// too-loose bound breaks, and it is checked here directly rather than only
/// through the kernels, which see a loosening only when it flips a decision.
#[test]
fn row_bound_never_exceeds_the_term_it_bounds() {
    for (y, a, b) in ROW_CASES {
        let lb = gaussian_row_term_lower_bound(y, a, b);
        let min = grid_min(y, a, b);
        assert!(
            lb <= min + 1e-12 * min.abs().max(1.0),
            "y={y} a={a} b={b}: bound {lb} exceeds the grid minimum {min}"
        );
    }
}

/// …and it is useful: exact under additive error, and within 0.6 of the minimum
/// under proportional / combined error with a non-tiny observation. A bound that
/// collapsed to its `ln MIN_POSITIVE` floor would still be valid and would switch
/// the gate off silently.
#[test]
fn row_bound_is_tight() {
    assert_eq!(gaussian_row_term_lower_bound(3.0, 0.16, 0.0), 0.16f64.ln());
    for (y, a, b) in [(3.0, 0.0, 0.0225), (450.0, 0.0, 0.09), (3.0, 0.04, 0.0144)] {
        let lb = gaussian_row_term_lower_bound(y, a, b);
        let min = grid_min(y, a, b);
        assert!(
            min - lb < 0.6,
            "y={y} a={a} b={b}: bound {lb} is {} below the minimum {min}",
            min - lb
        );
    }
}

/// The subject-level bound holds at arbitrary η, on every error model the gate
/// covers, against the data term `individual_nll_prepared` actually scores.
#[test]
fn the_data_bound_never_exceeds_the_scored_data_term() {
    let pop = sparse_population(6);
    for model in [additive_model(), proportional_model(), combined_model()] {
        let p = &model.default_params;
        let (theta, omega, sigma) = (&p.theta, &p.omega, &p.sigma.values);
        for subject in &pop.subjects {
            let mut scratch = MhScratch::default();
            scratch.begin_subject(&model, subject, theta, model.n_eta);
            let bound =
                prepared_data_term_lower_bound(&model, subject, theta, sigma, &scratch.prep)
                    .expect("these error models are bounded");
            assert_eq!(bound.n_rows, subject.observations.len());
            for k in 0..60 {
                let t = k as f64;
                let eta = [2.5 * (t * 0.37).sin(), 2.0 * (t * 0.91).cos()];
                let nll = individual_nll_prepared(
                    &model,
                    subject,
                    theta,
                    &eta,
                    omega,
                    sigma,
                    &scratch.prep,
                    None,
                    &mut scratch.nll,
                );
                let data = 2.0 * nll - quad_form(&omega.inv, &eta) - omega.log_det;
                assert!(
                    bound.lb <= data + 1e-9 * data.abs().max(1.0),
                    "subject {} η={eta:?}: bound {} above the data term {data}",
                    subject.id,
                    bound.lb
                );
            }
        }
    }
}

/// Every opt-out is load-bearing: `iiv_on_ruv` scales the variance by a factor of η,
/// and a `block_sigma` correlation can lower it below the uncorrelated `a + b·f²`.
#[test]
fn the_data_bound_declines_what_it_cannot_bound() {
    let pop = sparse_population(1);
    let subject = &pop.subjects[0];
    let iiv_ruv = parse_model_string(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  omega ETA_RUV ~ 0.1
  sigma ADD ~ 0.4 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)
  iiv_on_ruv = ETA_RUV
",
    )
    .expect("iiv_on_ruv fixture parses");
    let block = parse_model_string(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  block_sigma (PROP, ADD) = [0.0144, -0.01, 0.04]

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP, ADD)
",
    )
    .expect("block_sigma fixture parses");
    for (name, m) in [("iiv_on_ruv", &iiv_ruv), ("block_sigma", &block)] {
        let p = &m.default_params;
        let mut scratch = MhScratch::default();
        scratch.begin_subject(m, subject, &p.theta, m.n_eta);
        assert_eq!(
            prepared_data_term_lower_bound(m, subject, &p.theta, &p.sigma.values, &scratch.prep),
            None,
            "{name}: the data bound must decline this model"
        );
    }
    // Control: the same subject on a covered model is bounded, so the `None`s above
    // are the opt-outs and not a fixture that never reaches the row loop.
    let m = additive_model();
    let p = &m.default_params;
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&m, subject, &p.theta, m.n_eta);
    assert!(
        prepared_data_term_lower_bound(&m, subject, &p.theta, &p.sigma.values, &scratch.prep)
            .is_some()
    );
}

/// The gate's guards: no firing on a threshold that is not finite or sits on the
/// sentinel rail (a current state at `1e20` accepts a sentinel proposal through
/// `1e20 − 1e20 = 0 > log_u`), nor with a non-finite `log|Ω|` (the full evaluation
/// then returns the sentinel, not the sum the bound bounds); and the rounding
/// margin: a bound a hair above the threshold does not fire, a clear one does.
#[test]
fn the_gate_declines_the_rail_and_respects_its_margin() {
    let bound = DataTermLowerBound {
        lb: 10.0,
        abs: 10.0,
        n_rows: 3,
    };
    // lb_nll = ½(eta_prior + log_det + 10).
    let lb_nll = |eta_prior: f64| 0.5 * (eta_prior + 0.0 + 10.0);
    // Clear: 20 ≫ threshold 5 − (−1) = 6.
    assert!(mh_early_rejects(&bound, 30.0, 0.0, 5.0, -1.0));
    // A hair above the threshold, inside the margin: no.
    let thr = 5.0 - (-1.0);
    let eta_prior = 2.0 * thr - 10.0 + 1e-13;
    assert!(lb_nll(eta_prior) > thr);
    assert!(!mh_early_rejects(&bound, eta_prior, 0.0, 5.0, -1.0));
    // Below the threshold: no.
    assert!(!mh_early_rejects(&bound, 1.0, 0.0, 5.0, -1.0));
    // log_u = −∞ (a uniform draw of exactly 0): threshold +∞, never fires.
    assert!(!mh_early_rejects(
        &bound,
        1e300,
        0.0,
        5.0,
        f64::NEG_INFINITY
    ));
    // Current state on the sentinel rail: never fires, however large the bound.
    assert!(!mh_early_rejects(&bound, 1e300, 0.0, 1e20, -1.0));
    // Non-finite log|Ω|.
    assert!(!mh_early_rejects(&bound, 30.0, f64::INFINITY, 5.0, -1.0));
    assert!(!mh_early_rejects(&bound, 30.0, f64::NAN, 5.0, -1.0));
}
