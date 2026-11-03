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

use super::{
    early_reject_hooks, mh_early_rejects, mh_steps, mh_steps_componentwise, run_saem, MhGate,
    MhScratch,
};
use crate::parser::model_parser::parse_model_string;
use crate::stats::likelihood::{
    gaussian_row_term_lower_bound, individual_nll, individual_nll_prepared,
    prepared_data_term_lower_bound, quad_form, DataTermLowerBound, PropLoading,
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
    run_chain_scaled(model, subject, seed, gate_off, 1.0)
}

/// [`run_chain`] with every step scale multiplied by `scale`.
fn run_chain_scaled(
    model: &CompiledModel,
    subject: &Subject,
    seed: u64,
    gate_off: bool,
    scale: f64,
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
        3.0 * scale,
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
        &[3.5 * scale, 2.5 * scale],
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
    let _shared = early_reject_hooks::LOCK
        .read()
        .unwrap_or_else(|e| e.into_inner());
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

/// One residual row: an observation, its additive and proportional SDs (either may
/// be absent), and — for the `#484` magnitude path — the `(additive, proportional)`
/// magnitude multipliers. `mult: None` is the bare-σ path.
#[derive(Clone, Copy, Debug)]
struct Row {
    y: f64,
    sd_add: Option<f64>,
    sd_prop: Option<f64>,
    mult: Option<(f64, f64)>,
}

const fn row(y: f64, sd_add: Option<f64>, sd_prop: Option<f64>) -> Row {
    Row {
        y,
        sd_add,
        sd_prop,
        mult: None,
    }
}

impl Row {
    /// The additive variance in the scorer's arithmetic.
    fn a(&self) -> f64 {
        match (self.sd_add, self.mult) {
            (None, _) => 0.0,
            (Some(s), None) => s * s,
            (Some(s), Some((m, _))) => m * m * s * s,
        }
    }

    fn prop(&self) -> Option<PropLoading> {
        self.sd_prop.map(|s| PropLoading {
            sigma: s,
            mult: self.mult.map(|(_, m)| m),
        })
    }

    fn bound(&self) -> f64 {
        gaussian_row_term_lower_bound(self.y, self.a(), self.prop())
    }

    /// The scored term at prediction `f`, written the way the scoring loop scores it
    /// on each path (`residual_variance` / `loadings_variance_sum`), so the
    /// brute-force minimum sees the scorer's rounding, not the algebra.
    fn term(&self, f: f64) -> f64 {
        let p = match (self.sd_prop, self.mult) {
            (None, _) => 0.0,
            (Some(s), None) => {
                let fs = f * s;
                fs * fs
            }
            (Some(s), Some((_, m))) => {
                let c = f * m;
                c * c * s * s
            }
        };
        let raw = match (self.sd_prop.is_some(), self.sd_add.is_some()) {
            (true, true) => p + self.a(),
            (true, false) => p,
            (false, _) => self.a(),
        };
        let v = guard_variance(raw);
        (self.y - f) * (self.y - f) / v + v.ln()
    }

    /// Brute-force minimum over a grid that covers both signs, the guarded region
    /// near `f = 0`, and a dense band around `y`.
    fn grid_min(&self) -> f64 {
        let y = self.y;
        let mut fs = vec![0.0];
        for e in -320..=40 {
            let mag = 10f64.powf(e as f64 / 4.0);
            fs.push(mag);
            fs.push(-mag);
        }
        for k in 0..=4000 {
            fs.push(y * (k as f64 / 1000.0));
        }
        fs.iter()
            .map(|&f| self.term(f))
            .fold(f64::INFINITY, |m, t| {
                assert!(!t.is_nan(), "grid term NaN for {self:?}");
                m.min(t)
            })
    }
}

const ROW_CASES: [Row; 15] = [
    row(3.0, Some(0.4), None),
    row(3.0, None, Some(0.15)),
    row(0.002, None, Some(0.15)),
    row(450.0, None, Some(0.3)),
    row(-2.0, None, Some(0.2)),
    row(3.0, Some(0.2), Some(0.12)),
    row(0.05, Some(0.2), Some(0.12)),
    row(800.0, Some(0.2), Some(0.12)),
    row(3.0, Some(1e-15), Some(0.12)),
    row(1e-3, Some(1e-7), Some(1e-7)),
    row(0.0, Some(0.5), Some(0.1)),
    row(7.0, None, Some(1e-150)),
    // The #484 magnitude path, including the review-round-2 case: a magnitude whose
    // square underflows (`1e-200 · 1e-200 = 0`) against a σ that would cancel it
    // algebraically (`(1e-200 · 1e200)² = 1`). The scorer sees a guarded `0`.
    Row {
        y: 3.0,
        sd_add: Some(1e200),
        sd_prop: None,
        mult: Some((1e-200, 1.0)),
    },
    Row {
        y: 3.0,
        sd_add: Some(1e200),
        sd_prop: Some(1e200),
        mult: Some((1e-200, 1e-200)),
    },
    Row {
        y: 3.0,
        sd_add: Some(0.2),
        sd_prop: Some(0.1),
        mult: Some((1.5, 0.7)),
    },
];

/// The bound is a bound: never above the term at any prediction. This is what a
/// too-loose bound breaks, and it is checked here directly rather than only
/// through the kernels, which see a loosening only when it flips a decision.
#[test]
fn row_bound_never_exceeds_the_term_it_bounds() {
    for r in ROW_CASES {
        let lb = r.bound();
        let min = r.grid_min();
        assert!(
            lb <= min + 1e-12 * min.abs().max(1.0),
            "{r:?}: bound {lb} exceeds the grid minimum {min}"
        );
    }
}

/// …and it is useful: exact under additive error, and within 0.6 of the minimum
/// under proportional / combined error with a non-tiny observation. A bound that
/// collapsed to its `ln MIN_POSITIVE` floor would still be valid and would switch
/// the gate off silently.
#[test]
fn row_bound_is_tight() {
    assert_eq!(row(3.0, Some(0.4), None).bound(), (0.4f64 * 0.4).ln());
    for r in [
        row(3.0, None, Some(0.15)),
        row(450.0, None, Some(0.3)),
        row(3.0, Some(0.2), Some(0.12)),
        ROW_CASES[14],
    ] {
        let lb = r.bound();
        let min = r.grid_min();
        assert!(
            min - lb < 0.6,
            "{r:?}: bound {lb} is {} below the minimum {min}",
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
    let gate = |offset: f64| MhGate {
        bound: DataTermLowerBound {
            lb: 10.0,
            abs: 10.0,
            n_rows: 3,
        },
        offset,
        offset_abs: offset.abs(),
    };
    let g = gate(0.0);
    // lb_nll = ½(eta_prior + offset + 10).
    let lb_nll = |eta_prior: f64| 0.5 * (eta_prior + 0.0 + 10.0);
    // Clear: 20 ≫ threshold 5 − (−1) = 6.
    assert!(mh_early_rejects(&g, 30.0, 5.0, -1.0));
    // A hair above the threshold, inside the margin: no.
    let thr = 5.0 - (-1.0);
    let eta_prior = 2.0 * thr - 10.0 + 1e-13;
    assert!(lb_nll(eta_prior) > thr);
    assert!(!mh_early_rejects(&g, eta_prior, 5.0, -1.0));
    // Below the threshold: no.
    assert!(!mh_early_rejects(&g, 1.0, 5.0, -1.0));
    // log_u = −∞ (a uniform draw of exactly 0): threshold +∞, never fires.
    assert!(!mh_early_rejects(&g, 1e300, 5.0, f64::NEG_INFINITY));
    // Current state on the sentinel rail: never fires, however large the bound.
    assert!(!mh_early_rejects(&g, 1e300, 1e20, -1.0));
    // Non-finite offset (log|Ω|).
    assert!(!mh_early_rejects(&gate(f64::INFINITY), 30.0, 5.0, -1.0));
    assert!(!mh_early_rejects(&gate(f64::NAN), 30.0, 5.0, -1.0));
    // The offset counts: a negative one (log|Ω| < 0) pulls the bound under.
    assert!(
        mh_early_rejects(&gate(0.0), 4.0, 5.0, -1.0)
            != mh_early_rejects(&gate(-10.0), 4.0, 5.0, -1.0)
    );
}

/// Review of #2608: a finite observation large enough that `y²` overflows made the
/// row bound `∞`, and the gate then read `∞ ≥ ∞` and rejected **every** proposal —
/// while the full evaluation, whose `b·f²` stays finite for a tiny proportional
/// `b`, accepted them. The fixture is that regime: a dose of `2e156` puts the
/// predictions near `1e155`, the observations sit exactly on the η = 0 profile, the
/// proportional SD is `3.16e-4` (`b = 1e-7`, so `v ≈ 1e303` is finite) and the
/// steps are tiny, so the gate-off chain accepts most proposals. Without the
/// overflow decline in `prepared_data_term_lower_bound` the gate-on arm accepts
/// none and the A/B reddens; the acceptance assertion keeps the fixture from
/// degenerating into "both arms reject everything", where it could not see that.
#[test]
fn an_overflowing_bound_declines_instead_of_rejecting_everything() {
    let model = model("DV ~ proportional(PROP)", "  sigma PROP ~ 0.000316 (sd)\n");
    let mut subject = sparse_population(1).subjects.remove(0);
    subject.doses = vec![DoseEvent::new(0.0, 2e156, 1, 0.0, false, 0.0)];
    let p = &model.default_params;
    let zeros = vec![0.0; model.n_eta];
    subject.observations =
        crate::pk::compute_predictions_with_tv(&model, &subject, &p.theta, &zeros);
    let y_max = subject.observations.iter().cloned().fold(0.0, f64::max);
    assert!(
        (y_max * y_max).is_infinite() && y_max.is_finite(),
        "fixture must overflow y² with a finite y (y_max = {y_max})"
    );
    let nll0 = individual_nll(
        &model,
        &subject,
        &p.theta,
        &zeros,
        &p.omega,
        &p.sigma.values,
    );
    assert!(
        nll0.is_finite() && nll0 < 1e19,
        "the full objective must be finite here (got {nll0})"
    );

    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &subject, &p.theta, model.n_eta);
    assert_eq!(
        prepared_data_term_lower_bound(&model, &subject, &p.theta, &p.sigma.values, &scratch.prep),
        None,
        "an overflowing bound must decline the subject"
    );

    let (on, _) = run_chain_scaled(&model, &subject, 77, false, 1e-4);
    let (off, _) = run_chain_scaled(&model, &subject, 77, true, 1e-4);
    // bits = [η…, acc_block, nll_block, …]: the gate-off chain must actually accept.
    let acc_block = off[model.n_eta];
    assert!(
        acc_block > 0,
        "fixture degenerate: the full kernel accepted nothing"
    );
    assert_eq!(on, off, "the gate changed a chain whose bound overflows");
}

/// Review of #2608, round 2: the bound used to compute the `#484` magnitude path's
/// variance as `(m·σ)²` while the scorer computes `c·c·σ·σ` with `c = m`. With
/// `m = 1e-200` and `σ = 1e200` the scorer's `m·m` underflows to `0`, the variance
/// is guarded to `MIN_VARIANCE` and a zero-residual row scores `ln 1e-12 ≈ −27.6`;
/// the reassociated bound said `ln 1 = 0` — above the scored term, so the gate
/// could reject a proposal the full kernel accepts. The σ is passed directly (a
/// file cannot declare `1e200` as a variance), the observations sit on the η = 0
/// profile and the steps are tiny, so the gate-off chain accepts.
#[test]
fn the_bound_replays_the_magnitude_paths_arithmetic() {
    let model = model("DV ~ additive(ADD * 1e-200)", "  sigma ADD ~ 0.4 (sd)\n");
    let p = &model.default_params;
    let sigma = vec![1e200];
    let zeros = vec![0.0; model.n_eta];
    let mut subject = sparse_population(1).subjects.remove(0);
    subject.observations =
        crate::pk::compute_predictions_with_tv(&model, &subject, &p.theta, &zeros);

    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &subject, &p.theta, model.n_eta);
    assert!(
        model.ruv_obs_mult(&subject, &p.theta).is_some(),
        "fixture must take the #484 magnitude path"
    );
    let bound = prepared_data_term_lower_bound(&model, &subject, &p.theta, &sigma, &scratch.prep)
        .expect("an additive magnitude model is bounded");
    let nll = individual_nll_prepared(
        &model,
        &subject,
        &p.theta,
        &zeros,
        &p.omega,
        &sigma,
        &scratch.prep,
        None,
        &mut scratch.nll,
    );
    let data = 2.0 * nll - quad_form(&p.omega.inv, &zeros) - p.omega.log_det;
    // The scored rows sit on the guard: 3 × ln 1e-12.
    assert!(
        (data - 3.0 * 1e-12f64.ln()).abs() < 1e-9,
        "fixture must score the guarded variance (data = {data})"
    );
    assert!(
        bound.lb <= data,
        "bound {} above the scored data term {data}",
        bound.lb
    );

    // Kernel A/B at that σ: the gate must not change the chain.
    let chain = |gate_off: bool| {
        let mut scratch = MhScratch::default();
        scratch.begin_subject(&model, &subject, &p.theta, model.n_eta);
        scratch.early_reject_off = gate_off;
        let mut eta = zeros.clone();
        let mut rng = StdRng::seed_from_u64(31);
        let (acc, nll_out) = mh_steps(
            &mut eta,
            nll,
            &subject,
            &model,
            &p.theta,
            &p.omega,
            &sigma,
            1e-12,
            None,
            &mut rng,
            30,
            &mut scratch,
            None,
            None,
        );
        let mut bits: Vec<u64> = eta.iter().map(|x| x.to_bits()).collect();
        bits.extend([acc as u64, nll_out.to_bits()]);
        bits
    };
    let off = chain(true);
    assert!(
        off[model.n_eta] > 0,
        "fixture degenerate: the full kernel accepted nothing"
    );
    assert_eq!(
        chain(false),
        off,
        "the gate changed the chain on the magnitude path"
    );
}

// ─── Whole fits ───────────────────────────────────────────────────────────
//
// The kernel A/Bs above pin each call; these pin the property the issue asks for —
// a whole fixed-seed trajectory, gate on vs gate off, equal on raw bits — on every
// path that reaches the gate: analytic, event-driven (time-varying covariates),
// ODE, a mixture, BLOQ M3 rows, IOV (κ in the gate's offset), and the Bayes chain.
// The gate is switched off process-wide (`early_reject_hooks::OFF`) because the
// kernels run on rayon workers; each A/B also asserts the gate fired during the
// gate-on arm, so an A/B whose fixture never reached it cannot pass.

/// Restores `OFF` even when an assertion panics mid-A/B.
struct GateOffGuard;
impl Drop for GateOffGuard {
    fn drop(&mut self) {
        early_reject_hooks::OFF.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

fn fit_ab(name: &str, run: impl Fn() -> Vec<u64>) {
    use std::sync::atomic::Ordering::SeqCst;
    let _exclusive = early_reject_hooks::LOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
    let _guard = GateOffGuard;
    early_reject_hooks::OFF.store(false, SeqCst);
    let before = early_reject_hooks::FIRED.load(SeqCst);
    let on = run();
    let fired = early_reject_hooks::FIRED.load(SeqCst) - before;
    early_reject_hooks::OFF.store(true, SeqCst);
    let off = run();
    early_reject_hooks::OFF.store(false, SeqCst);
    assert!(
        fired > 0,
        "{name}: the gate never fired during the gate-on fit, so this A/B cannot see it"
    );
    assert_eq!(on, off, "{name}: the early-rejection gate changed the fit");
}

fn fit_opts() -> FitOptions {
    FitOptions {
        saem_n_exploration: 6,
        saem_n_convergence: 4,
        saem_n_mh_steps: 6,
        saem_omega_burnin: 0,
        saem_seed: Some(2_608),
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    }
}

/// Every bit of the result the E-step feeds: estimates, objective, iteration
/// count, posterior summaries and κs.
fn result_bits(r: &crate::estimation::outer_optimizer::OuterResult) -> Vec<u64> {
    let mut out = vec![r.ofv.to_bits(), r.n_iterations as u64];
    out.extend(r.params.theta.iter().map(|v| v.to_bits()));
    out.extend(r.params.omega.matrix.iter().map(|v| v.to_bits()));
    out.extend(r.params.sigma.values.iter().map(|v| v.to_bits()));
    for e in &r.eta_hats {
        out.extend(e.iter().map(|v| v.to_bits()));
    }
    for subj in &r.kappas {
        for k in subj {
            out.extend(k.iter().map(|v| v.to_bits()));
        }
    }
    out
}

fn saem_bits(model: &CompiledModel, pop: &Population) -> Vec<u64> {
    let r = run_saem(model, pop, &model.default_params, &fit_opts()).expect("SAEM run");
    result_bits(&r)
}

fn src_with(params: &str, indiv: &str, structural: &str, error: &str, extra: &str) -> String {
    format!(
        "[parameters]\n{params}\n[individual_parameters]\n{indiv}\n[structural_model]\n{structural}\n{extra}\n[error_model]\n  {error}\n"
    )
}

const PK_PARAMS: &str = "  theta TVCL(3.0, 0.5, 30.0)\n  theta TVV(20.0, 2.0, 200.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.05\n  sigma PROP ~ 0.12 (sd)\n  sigma ADD ~ 0.2 (sd)\n";

#[test]
fn whole_saem_fit_is_bit_identical_analytic() {
    let m = combined_model();
    let pop = sparse_population(24);
    fit_ab("analytic", || saem_bits(&m, &pop));
}

#[test]
fn whole_saem_fit_is_bit_identical_ode() {
    let m = parse_model_string(&src_with(
        PK_PARAMS,
        "  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n",
        "  ode(states=[central])\n",
        "DV ~ combined(PROP, ADD)",
        "[odes]\n  d/dt(central) = -(CL/V) * central\n\n[scaling]\n  y = central / V\n",
    ))
    .expect("ODE fixture parses");
    assert!(m.ode_spec.is_some(), "fixture must take the ODE engine");
    let pop = sparse_population(16);
    fit_ab("ode", || saem_bits(&m, &pop));
}

/// Time-varying covariates route every subject onto the event-driven walk.
#[test]
fn whole_saem_fit_is_bit_identical_event_driven() {
    use std::collections::HashMap;
    let m = parse_model_string(&src_with(
        PK_PARAMS,
        "  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n",
        "  pk one_cpt_iv(cl=CL, v=V)\n",
        "DV ~ combined(PROP, ADD)",
        "",
    ))
    .expect("event-driven fixture parses");
    let mut pop = sparse_population(16);
    pop.covariate_names = vec!["WT".into()];
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let w0 = 55.0 + 3.0 * i as f64;
        s.dose_covariates = vec![wt(w0)];
        s.obs_covariates = vec![wt(w0), wt(w0 + 2.0), wt(w0 + 5.0)];
        s.covariates = wt(w0);
    }
    assert!(
        pop.subjects.iter().all(|s| s.has_tv_covariates()),
        "fixture must route onto the event-driven walk"
    );
    fit_ab("event-driven", || saem_bits(&m, &pop));
}

#[test]
fn whole_saem_fit_is_bit_identical_mixture() {
    let m = parse_model_string(
        r"
[parameters]
  theta TVCL1(2.0, 0.01, 100.0)
  theta TVCL2(5.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.12 (sd)
  sigma ADD ~ 0.2 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP, ADD)
",
    )
    .expect("mixture fixture parses");
    let pop = sparse_population(16);
    fit_ab("mixture", || saem_bits(&m, &pop));
}

/// M3: the last sample of every other subject is censored at 0.8, so the gate's
/// bound carries censored rows (bounded by 0) next to scored ones.
#[test]
fn whole_saem_fit_is_bit_identical_bloq_m3() {
    let mut m = combined_model();
    m.bloq_method = BloqMethod::M3;
    let mut pop = sparse_population(16);
    for s in pop.subjects.iter_mut().step_by(2) {
        let last = s.observations.len() - 1;
        s.observations[last] = 0.8;
        s.cens[last] = 1;
    }
    assert!(m.bloq_in_force().has_censored_row(&pop.subjects[0].cens));
    fit_ab("bloq-m3", || saem_bits(&m, &pop));
}

/// IOV: two occasions per subject; the η steps carry the κ prior in the gate's
/// offset.
#[test]
fn whole_saem_fit_is_bit_identical_iov() {
    let m = parse_model_string(&src_with(
        &format!("{PK_PARAMS}  kappa KAPPA_CL ~ 0.04\n"),
        "  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV * exp(ETA_V)\n",
        "  pk one_cpt_iv(cl=CL, v=V)\n",
        "DV ~ combined(PROP, ADD)",
        "",
    ))
    .expect("IOV fixture parses");
    assert!(m.n_kappa > 0);
    let mut pop = sparse_population(16);
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        let f = i as f64;
        s.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ];
        s.dose_occasions = vec![1, 2];
        s.obs_times = vec![1.0, 6.0, 25.0, 30.0];
        s.observations = s
            .obs_times
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                let tad = if t >= 24.0 { t - 24.0 } else { t };
                let carry = if t >= 24.0 {
                    (-0.15_f64 * t).exp()
                } else {
                    0.0
                };
                5.0 * ((-0.15_f64 * tad).exp() + carry)
                    * (1.0 + 0.25 * (f * 1.3 + k as f64 * 2.1).sin())
            })
            .collect();
        s.obs_cmts = vec![1; 4];
        s.cens = vec![0; 4];
        s.occasions = vec![1, 1, 2, 2];
    }
    fit_ab("iov", || saem_bits(&m, &pop));
}

#[test]
fn whole_bayes_chain_is_bit_identical() {
    let m = combined_model();
    let pop = sparse_population(12);
    let opts = FitOptions {
        bayes_warmup: 30,
        bayes_iters: 30,
        bayes_chains: 1,
        bayes_thin: 1,
        bayes_seed: Some(2_608),
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    fit_ab("bayes", || {
        let r = crate::estimation::bayes::run_bayes(&m, &pop, &m.default_params, &opts)
            .expect("Bayes run");
        result_bits(&r)
    });
}
