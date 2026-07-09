//! Tier-3 convergence + simulation-estimation (SSE) tests for Phase 1 TTE.
//!
//! Gated behind BOTH `survival` (the TTE feature) and `slow-tests` (these run a
//! full fit to convergence, so they are skipped on the per-PR `Test` job and run
//! nightly via `slow-tests.yml` — see the test-tier rules in CLAUDE.md).
//!
//! Two kinds of guard:
//!
//!   * **SSE** (`tte_sse_*`): generate a dataset from known `(theta, Omega)` with
//!     ferx's *own* `simulate()`, refit with ferx, and assert parameter recovery.
//!     This is the primary guard on the generative path (plan §14.11): a wrong
//!     event-time sampler or a wrong likelihood constant shows up here as
//!     non-recovery, which a fit-only test cannot detect.
//!
//!   * **Cross-tool** (`tte_convergence_*`): fit the committed reference dataset
//!     `tests/reference/tte_exponential/tte_exp.csv` — the *same* file NONMEM and
//!     nlmixr2 fit — through the real datareader (`read_population_for`, so the
//!     DV→`obs_records` routing is exercised too) and assert the estimates land in
//!     the documented bands. The fixed-effects (`n_eta = 0`) fit is checked
//!     *exactly* against the base-R `survival::survreg` closed-form MLE
//!     (`lambda = events / sum(time)`); the mixed-effects fit must instead recover
//!     the data-generating `lambda_pop = 0.1`, `omega^2 = 0.25` (the pooled
//!     fixed-effects rate is biased low by the ignored between-subject variance —
//!     which is the whole point of fitting the mixed model).

#![cfg(all(feature = "survival", feature = "slow-tests"))]

mod common;

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::Population;
use ferx_core::{fit, simulate_with_seed, FitOptions, SimOutcome, SimulationResult};

// ── Model strings ────────────────────────────────────────────────────────────

/// Data-generating ("truth") model used by the SSE test: lambda_pop = 0.1,
/// omega^2 = 0.25 on log(lambda). `scale` is the Exponential rate.
const EXP_TRUTH: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.25

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Mixed-effects fit model, initialised *away* from the truth so recovery is a
/// real test (rate 2x low, variance ~3x low).
const EXP_FIT: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Fixed-effects (n_eta = 0) fit model — ordinary parametric Exponential PH, the
/// plain-likelihood special case (plan D7). Anchored against `survreg`.
const EXP_FIT_FIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";

/// RTTE (Slice 3.3) **data-generating ("truth")** model for the SSE round-trip:
/// clock-forward exponential recurrent events, `lambda_pop = 0.15`, `omega^2 = 0.09`
/// on log(lambda). Event-rich (~3 events/subject at horizon 20) ⇒ FOCEI-Laplace ω²
/// bias is mild (plan §3.3; matches the `tests/reference/rtte_exponential` anchor).
const RTTE_EXP_TRUTH: &str = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// RTTE SSE **fit** model, initialised away from the truth (rate ~half, variance
/// ~third) so recovery is a real test.
const RTTE_EXP_FIT: &str = r"
[parameters]
  theta TVLAMBDA(0.08, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.03

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// RTTE **sampler goodness-of-fit** truth: fixed-effects clock-forward Weibull
/// (no frailty, no residual error). A single shared intensity `h(t)` means the
/// closed-form cumulative hazard `H(t) = (t/scale)^shape` is one curve, so the
/// probability-integral transform of the rescaled inter-event increments needs no
/// per-subject random effect. Shape > 1 ⇒ the hazard genuinely varies with
/// absolute time, so the test exercises *conditional* (not memoryless) sampling.
const RTTE_WEIBULL_FWD_PIT_TRUTH: &str = r"
[parameters]
  theta TVSCALE(8.0, 0.1, 1000.0)
  theta TVSHAPE(1.5, 0.1, 10.0)

[event_model]
  cmt    = 2
  type   = rtte
  clock  = forward
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";

/// Competing-risks "truth" model: two exponential causes (CMT 2, CMT 3) linked
/// by a shared log-frailty `ETA_F` (ω²=0.25). The cause-specific *rates* are
/// well-identified and recover tightly; the shared frailty ω² is weakly
/// identified and reads high under FOCEI-Laplace (#440/#469) — see the test's
/// band comment.
const COMPETING_TRUTH: &str = r"
[parameters]
  theta TVLAMBDA_A(0.10, 0.001, 10.0)
  theta TVLAMBDA_B(0.06, 0.001, 10.0)
  omega ETA_F ~ 0.25

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
";

/// Competing-risks fit model, initialised away from the truth.
const COMPETING_FIT: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)
  omega ETA_F ~ 0.09

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
";

/// Fixed-effects (no frailty) competing-risks model — anchors the cause-specific
/// likelihood against the per-cause closed-form / `survreg` MLE.
const COMPETING_FIXED: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B
";

/// Joint PK-TTE **data-generating ("truth")** model (Slice 2.2 SSE): 1-cpt oral
/// PK with a drug-driven (ODE-accumulated) hazard rising with central
/// concentration, sharing the CL random effect. PK on CMT 2, event on CMT 3.
const ODE_TTE_TRUTH: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.30, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Joint PK-TTE **fit** model, initialised away from the truth (CL/KA high, V/H0
/// low, BETA at 0, variance ~halved) so recovery is a real test.
const ODE_TTE_FIT: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(7.0, 0.1, 500.0)
  theta TVKA(0.7, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.03 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Joint PK-TTE truth for the **sampler goodness-of-fit** test — identical PK +
/// drug-driven hazard as `ODE_TTE_TRUTH` but **fixed-effects** (no `ETA_CL`, no
/// residual error: there are no continuous observations). With `n_eta = 0` every
/// subject shares one deterministic concentration trajectory, so the closed-form
/// cumulative-hazard oracle (`cumhaz_grid`) is a single curve and the
/// probability-integral transform needs no per-subject random effect.
const ODE_TTE_PIT_TRUTH: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.30, -10.0, 10.0)
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL   = TVCL
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)
";

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fit_opts() -> FitOptions {
    // Default is already FOCEI / outer_maxiter = 500 / interaction = true — i.e.
    // run-to-convergence. We only quiet it and request the covariance step so the
    // comparison table can report SEs.
    FitOptions {
        verbose: false,
        run_covariance_step: true,
        ..FitOptions::default()
    }
}

/// `n` bare TTE subjects (one placeholder Event each on CMT 2) used only as a
/// `simulate()` template. Each placeholder carries the administrative censoring
/// window `censor` in its record `time`; `simulate_tte` right-censors any draw
/// that reaches it (the drawn event time otherwise replaces the placeholder).
/// This is exactly [`common::tte_pop_from_pairs`] over `n` right-censored rows at
/// `t = censor`.
fn tte_sim_template(n: usize, censor: f64) -> Population {
    common::tte_pop_from_pairs(&vec![(censor, 0); n])
}

/// Map ferx `simulate()` outcomes to `(time, dv)` pairs. `simulate_tte` already
/// applies administrative right-censoring at each subject's observation window
/// (set by [`tte_sim_template`]), so this just reads the `observed` flag: an
/// observed draw is an event row `(time, 1)`, a censored draw a `(window, 0)` row.
/// Panics on any non-`Event` outcome — that would mean a Gaussian model/template
/// was simulated by mistake.
fn sims_to_pairs(sims: &[SimulationResult]) -> Vec<(f64, u8)> {
    sims.iter()
        .map(|r| match r.outcome {
            SimOutcome::Event { time, observed } => (time, observed as u8),
            _ => panic!("expected an Event outcome for a TTE simulation"),
        })
        .collect()
}

/// Rebuild a competing-risks fit population from `simulate()` outcomes: group the
/// per-cause rows by subject and turn each into an `ObsRecord::Event` (exact when
/// observed, right-censored otherwise) on its CMT. `BTreeMap` keeps subject order
/// deterministic.
fn competing_pop_from_sims(sims: &[SimulationResult]) -> Population {
    use ferx_core::types::{EventType, ObsRecord};
    use std::collections::BTreeMap;

    let mut by_id: BTreeMap<String, Vec<(usize, f64, bool)>> = BTreeMap::new();
    for r in sims {
        match &r.outcome {
            SimOutcome::Event { time, observed } => {
                by_id
                    .entry(r.id.clone())
                    .or_default()
                    .push((r.cmt, *time, *observed));
            }
            _ => panic!("expected an Event outcome for a competing-risks simulation"),
        }
    }

    let subjects = by_id
        .into_iter()
        .map(|(id, recs)| {
            let mut s = common::subject(&id, vec![], vec![], vec![], vec![]);
            s.obs_records = recs
                .into_iter()
                .map(|(cmt, time, observed)| ObsRecord::Event {
                    time,
                    event_type: if observed {
                        EventType::Exact
                    } else {
                        EventType::RightCensored
                    },
                    entry_time: 0.0,
                    cmt,
                })
                .collect();
            s
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Joint PK-TTE `simulate()` template: `n` dosed subjects, each with PK sampling
/// times on CMT 2 (central) and a right-censored TTE placeholder on CMT 3 at the
/// `horizon`. The dose level is rotated across {20, 40, 80} so the central
/// concentration spans a range across subjects — this informs the *PK* fit, but it
/// does NOT make the hazard parameters separately identifiable: `H0·exp(BETA·Cc)` is
/// collinear in (H0, BETA) at any single-occasion design (Slice 2.1 `expected.md`:
/// corr ≈ −0.91 even across NONMEM / nlmixr2, off-truth at a shared optimum). The
/// sampler itself is validated distribution-free by the PIT/KS goodness-of-fit test
/// (`joint_pktte_event_times_match_model_survival`), not by H0/BETA recovery here.
fn joint_pktte_sim_template(n: usize, horizon: f64) -> Population {
    use ferx_core::types::{DoseEvent, EventType, ObsRecord};
    let pk_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
    let dose_levels = [20.0, 40.0, 80.0];
    let subjects = (0..n)
        .map(|i| {
            let amt = dose_levels[i % dose_levels.len()];
            let mut s = common::subject(
                &i.to_string(),
                vec![DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)],
                pk_times.clone(),
                vec![0.0; pk_times.len()], // placeholder DVs, overwritten by sim
                vec![2; pk_times.len()],   // PK observed on central (CMT 2)
            );
            s.obs_records = vec![ObsRecord::Event {
                time: horizon,
                event_type: EventType::RightCensored,
                entry_time: 0.0,
                cmt: 3,
            }];
            s
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Rebuild a fittable joint PK-TTE population from `simulate()` output: keep the
/// template's doses / sampling design and fill in the simulated continuous PK
/// observations (CMT 2, in `obs_times` order) and the simulated event outcome
/// (CMT 3 → `ObsRecord::Event`, exact when observed, right-censored otherwise).
fn joint_pktte_pop_from_sims(template: &Population, sims: &[SimulationResult]) -> Population {
    use ferx_core::types::{EventType, ObsRecord};
    use std::collections::BTreeMap;
    let mut pk_by_id: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut ev_by_id: BTreeMap<String, (f64, bool, usize)> = BTreeMap::new();
    for r in sims {
        match r.outcome {
            SimOutcome::Continuous { value } => {
                pk_by_id.entry(r.id.clone()).or_default().push(value)
            }
            SimOutcome::Event { time, observed } => {
                ev_by_id.insert(r.id.clone(), (time, observed, r.cmt));
            }
        }
    }
    let mut pop = template.clone();
    for s in pop.subjects.iter_mut() {
        if let Some(vals) = pk_by_id.get(&s.id) {
            assert_eq!(
                vals.len(),
                s.observations.len(),
                "simulated PK obs count must match the template sampling design"
            );
            s.observations = vals.clone();
        }
        if let Some((time, observed, cmt)) = ev_by_id.get(&s.id) {
            s.obs_records = vec![ObsRecord::Event {
                time: *time,
                event_type: if *observed {
                    EventType::Exact
                } else {
                    EventType::RightCensored
                },
                entry_time: 0.0,
                cmt: *cmt,
            }];
        }
    }
    pop
}

/// Grid spacing for the cumulative-hazard oracle (`cumhaz_grid` / `interp_at`). At
/// 0.01 the trapezoid error on the smooth hazard is ~1e-6, negligible against the
/// KS statistic.
const PIT_GRID_STEP: f64 = 0.01;

/// Independent cumulative-hazard oracle for the drug-driven hazard, on the uniform
/// grid `t_k = k·PIT_GRID_STEP`, `k = 0..=⌈horizon/step⌉`. The concentration is the
/// **closed-form** 1-cpt oral solution `Cc(t) = dose·KA/(V(KA−ke))·(e^{−ke t} −
/// e^{−KA t})`, `ke = CL/V`, F = 1 — the same solution `tests/reference/
/// pktte_joint/simulate.R` uses — and `H(t) = ∫₀ᵗ H0·exp(BETA·Cc)` is accumulated by
/// trapezoid. This deliberately shares **no code** with the production augmented-ODE
/// root-finder, so feeding the simulated event times back through it is a genuine
/// cross-check of the sampler (not a tautology).
fn cumhaz_grid(dose: f64, cl: f64, v: f64, ka: f64, h0: f64, beta: f64, horizon: f64) -> Vec<f64> {
    let step = PIT_GRID_STEP;
    let ke = cl / v;
    let haz = |t: f64| {
        let cc = dose * ka / (v * (ka - ke)) * ((-ke * t).exp() - (-ka * t).exp());
        h0 * (beta * cc).exp()
    };
    let n = (horizon / step).ceil() as usize;
    let mut hgrid = Vec::with_capacity(n + 1);
    hgrid.push(0.0);
    let (mut acc, mut h_prev) = (0.0_f64, haz(0.0));
    for k in 1..=n {
        let h = haz(k as f64 * step);
        acc += 0.5 * (h + h_prev) * step;
        hgrid.push(acc);
        h_prev = h;
    }
    hgrid
}

/// Linear interpolation of a `cumhaz_grid` at an arbitrary `t` (clamped to the grid).
fn interp_at(hgrid: &[f64], t: f64) -> f64 {
    let x = (t / PIT_GRID_STEP).max(0.0);
    let k = x.floor() as usize;
    if k + 1 >= hgrid.len() {
        return *hgrid.last().unwrap();
    }
    let frac = x - k as f64;
    hgrid[k] * (1.0 - frac) + hgrid[k + 1] * frac
}

/// **Slice 2.2 sampler goodness-of-fit** — the decisive, estimation-free validator
/// for drug-driven event-time simulation. For a **fixed-effects** joint PK-TTE truth
/// (every subject shares one survival curve), simulate `N` event times with the
/// production augmented-ODE root-finder, then probability-integral-transform each:
/// `V_i = exp(−Ĥ(T_i))`, where `Ĥ` is the **independent** closed-form-PK + trapezoid
/// oracle (`cumhaz_grid`, no shared code with the sampler). If the root-finder
/// correctly inverts `CHZ(T) = −log U`, the `V_i` are Uniform(0,1); a
/// Kolmogorov–Smirnov test against Uniform must pass at the 5% level.
///
/// This is what actually validates the new Slice 2.2 code: unlike the SSE fit below,
/// it is **immune to the (H0, BETA) collinear ridge** (no estimation) and to any
/// FOCEI-Laplace bias — a wrong crossing, a dropped dose in the segmentation, or a
/// mis-scaled hazard shifts the `V_i` off Uniform and fails here directly.
#[test]
fn joint_pktte_event_times_match_model_survival() {
    use ferx_core::types::{DoseEvent, EventType, ObsRecord};
    use ferx_core::{simulate_with_options, SimulateOptions};
    const N: usize = 500;
    const DOSE: f64 = 100.0;
    // S(HORIZON) = exp(−H(600)) ≈ e^{−12}: no subject censors, so the PIT reference
    // is the full (untruncated) Uniform(0,1) and the standard KS critical applies.
    const HORIZON: f64 = 600.0;
    const SEED: u64 = 20260629;
    // Must match ODE_TTE_PIT_TRUTH (the oracle re-derives the same hazard).
    const CL: f64 = 1.0;
    const V: f64 = 10.0;
    const KA: f64 = 1.0;
    const H0: f64 = 0.02;
    const BETA: f64 = 0.30;

    let truth = parse_model_string(ODE_TTE_PIT_TRUTH).expect("PIT truth model must parse");

    // Minimal template: one depot dose (CMT 1) + a TTE placeholder censored at the
    // horizon (CMT 3). No continuous observations — this isolates the event sampler.
    let mut template = common::tte_pop_from_pairs(&vec![(HORIZON, 0u8); N]);
    for s in template.subjects.iter_mut() {
        s.doses = vec![DoseEvent::new(0.0, DOSE, 1, 0.0, false, 0.0)];
        s.obs_records = vec![ObsRecord::Event {
            time: HORIZON,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 3,
        }];
    }
    let opts = SimulateOptions {
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&truth, &template, &truth.default_params, 1, &opts)
        .expect("PIT simulation must succeed");

    // Every draw must be an observed event at this horizon (a censor would truncate
    // the PIT reference and silently weaken the GOF).
    let times: Vec<f64> = sims
        .iter()
        .map(|r| match r.outcome {
            SimOutcome::Event { time, observed } => {
                assert!(
                    observed,
                    "HORIZON={HORIZON} must admit no censoring; got a censor at {time}"
                );
                time
            }
            ref o => panic!("expected an Event outcome, got {o:?}"),
        })
        .collect();
    assert_eq!(times.len(), N, "one event per template subject");

    // PIT: V_i = exp(−Ĥ(T_i)) via the independent oracle; sorted for the KS sweep.
    let hgrid = cumhaz_grid(DOSE, CL, V, KA, H0, BETA, HORIZON);
    let mut u: Vec<f64> = times
        .iter()
        .map(|&t| (-interp_at(&hgrid, t)).exp())
        .collect();
    u.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));

    // Kolmogorov–Smirnov statistic against the FULLY-SPECIFIED Uniform(0,1) (truth
    // parameters, not estimated ⇒ the standard 1.36/√N critical is exact; no
    // Lilliefors correction).
    let n = u.len() as f64;
    let mut d = 0.0_f64;
    for (i, &ui) in u.iter().enumerate() {
        let upper = (i as f64 + 1.0) / n - ui;
        let lower = ui - i as f64 / n;
        d = d.max(upper).max(lower);
    }
    let crit = 1.36 / n.sqrt(); // two-sided KS, α = 0.05
    eprintln!("[PIT-ODE] N={N} KS D={d:.4} vs 5% critical {crit:.4}");
    assert!(
        d < crit,
        "simulated event times fail the KS goodness-of-fit against the model survival: \
         D={d:.4} ≥ {crit:.4} — the ODE event-time sampler does not reproduce S(t)"
    );
}

/// **Slice 2.2 SSE** — license-free round-trip validator for the joint PK-TTE
/// generative path. Simulate a dataset from known (θ, Ω) with ferx's own ODE
/// event-time sampler, rebuild a fit population, and refit from a perturbed start.
///
/// Scope: this asserts recovery of the **identifiable** parameters — the PK fixed
/// effects (CL, V, KA) and ω²(CL), which the continuous observations pin sharply —
/// and that the whole pipeline (root-finder → `simulate_tte` ODE branch →
/// sim→Population → joint FOCEI fit) runs to a finite OFV. It deliberately does
/// **not** assert recovery of (H0, BETA): they are collinear at a single-occasion
/// design (Slice 2.1 `expected.md`; this seed lands at H0≈0.010 / BETA≈0.43, off the
/// 0.02 / 0.30 truth along the ridge, max |ΔS(t)|≈0.16). Hazard-sampler correctness
/// is validated distribution-free by `joint_pktte_event_times_match_model_survival`
/// (and cross-tool by the NONMEM `$SIM` / nlmixr2 anchors), NOT by an H0/BETA band
/// here — a wide "sane-range" band on a non-identified parameter is exactly the kind
/// of pass-for-the-wrong-reason guard this suite avoids (cf. the Weibull ω² note).
#[test]
fn joint_pktte_sse_recovers_pk_and_omega() {
    use ferx_core::{simulate_with_options, SimulateOptions};
    const N: usize = 300;
    const HORIZON: f64 = 24.0;
    const SEED: u64 = 20260629;

    let truth = parse_model_string(ODE_TTE_TRUTH).expect("truth model must parse");
    let template = joint_pktte_sim_template(N, HORIZON);

    let opts = SimulateOptions {
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&truth, &template, &truth.default_params, 1, &opts)
        .expect("joint PK-TTE simulation must succeed");

    // The drug-driven hazard must produce a non-degenerate event/censor mix.
    let n_events = sims
        .iter()
        .filter(|r| matches!(r.outcome, SimOutcome::Event { observed: true, .. }))
        .count();
    eprintln!("[SSE-ODE] events = {n_events}/{N}");
    assert!(
        n_events > N / 10 && n_events < N - N / 10,
        "expected a mix of events and censors; got {n_events}/{N}"
    );

    let fit_pop = joint_pktte_pop_from_sims(&template, &sims);
    let model = parse_model_string(ODE_TTE_FIT).expect("fit model must parse");
    let r =
        fit(&model, &fit_pop, &model.default_params, &fit_opts()).expect("SSE fit must succeed");

    let (cl, v, ka, h0, beta) = (r.theta[0], r.theta[1], r.theta[2], r.theta[3], r.theta[4]);
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE-ODE] CL={cl:.4}(1.0) V={v:.4}(10) KA={ka:.4}(1) H0={h0:.5}(0.02; collinear) \
         BETA={beta:.4}(0.30; collinear) omega^2={omega2:.4}(0.09) OFV={:.3}",
        r.ofv
    );

    // PK fixed effects + ω²(CL) are sharply identified by the continuous observations
    // and recover tightly (truth recovery; bands bracket the seed-fixed deterministic
    // estimate with the truth inside). A gross sim→fit pipeline break (wrong PK
    // forcing, scrambled obs write-back) breaks these.
    assert!(
        (0.93..1.06).contains(&cl),
        "CL not recovered: {cl} (truth 1.0)"
    );
    assert!((9.0..10.2).contains(&v), "V not recovered: {v} (truth 10)");
    assert!(
        (0.92..1.08).contains(&ka),
        "KA not recovered: {ka} (truth 1.0)"
    );
    assert!(
        (0.05..0.13).contains(&omega2),
        "omega^2(CL) not recovered: {omega2} (truth 0.09)"
    );
    // (H0, BETA) are intentionally not band-checked here — see the doc comment and
    // `joint_pktte_event_times_match_model_survival`. Only require they stayed in the
    // feasible region (a NaN / sign-blown hazard would surface as a non-finite OFV).
    assert!(
        h0 > 0.0 && beta.is_finite(),
        "hazard params must stay feasible"
    );
    assert!(r.ofv.is_finite(), "joint PK-TTE SSE OFV must be finite");
}

// ── RTTE simulation (Slice 3.3) ──────────────────────────────────────────────

/// Rebuild an RTTE fit population from `simulate()` outcomes: group each subject's
/// rows (K observed events + 1 administrative censor) into `ObsRecord::Event`s on
/// the endpoint CMT — `Exact` for an observed event, `RightCensored` for the
/// trailing censor. This is exactly the layout the data reader produces from RTTE
/// rows (`DV = 1` events then a `DV = 0` censor), so the round-trip refit sees the
/// same records the sampler emitted.
fn rtte_pop_from_sims(sims: &[SimulationResult]) -> Population {
    use ferx_core::types::{EventType, ObsRecord};
    use std::collections::BTreeMap;

    let mut by_id: BTreeMap<String, Vec<(usize, f64, bool)>> = BTreeMap::new();
    for r in sims {
        match r.outcome {
            SimOutcome::Event { time, observed } => {
                by_id
                    .entry(r.id.clone())
                    .or_default()
                    .push((r.cmt, time, observed));
            }
            ref o => panic!("expected an Event outcome for an RTTE simulation, got {o:?}"),
        }
    }

    let subjects = by_id
        .into_iter()
        .map(|(id, recs)| {
            let mut s = common::subject(&id, vec![], vec![], vec![], vec![]);
            s.obs_records = recs
                .into_iter()
                .map(|(cmt, time, observed)| ObsRecord::Event {
                    time,
                    event_type: if observed {
                        EventType::Exact
                    } else {
                        EventType::RightCensored
                    },
                    entry_time: 0.0,
                    cmt,
                })
                .collect();
            s
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// **Slice 3.3 sampler goodness-of-fit** — the decisive, estimation-free validator
/// for the clock-forward RTTE stream. For a fixed-effects Weibull truth (one shared
/// intensity), the **time-rescaling theorem** says the inter-event increments
/// `ΔH_k = H(t_k) − H(t_{k−1})`, with `H(t) = (t/scale)^shape` (an inline closed
/// form, no shared code with the sampler), are iid `Exp(1)` under correct sampling,
/// so `V_k = exp(−ΔH_k) ~ Uniform(0,1)`; a Kolmogorov–Smirnov test against Uniform
/// must pass at 5%.
///
/// **Why only the first `K` gaps, with the horizon set far away.** Pooling *every*
/// completed gap from a fixed-horizon renewal process is biased: a gap is kept only
/// if its event lands before the horizon (`u > exp(−(H(horizon) − H(t_{k−1})))`), so
/// gaps adjacent to the boundary are selected toward small values — a property of the
/// truncated data, not the sampler. Taking the first `K` gaps per subject with
/// `horizon ≫ K·E[gap]` makes that keep-threshold `≈ exp(−H(horizon)) ≈ 0`, so those
/// `K` gaps are *unconditioned* draws and the KS reference is the exact Uniform(0,1).
///
/// What this pins that the SSE below cannot: the *loop* itself — conditioning each
/// draw on survival past the previous event (subtracting `H(t_{k−1})`), the forward
/// clock (a reset bug leaves the `V_k` non-uniform), and no RNG reordering. It is
/// immune to any estimator bias (no fit). Shape > 1 makes the hazard time-varying, so
/// a memoryless short-cut fails here (gaps 2..K are genuine conditional draws).
#[test]
fn rtte_forward_event_times_match_model_survival() {
    use ferx_core::{simulate_with_options, SimulateOptions};
    use std::collections::BTreeMap;
    const N: usize = 600;
    // horizon ≫ K·E[gap] (E[gap] = scale·Γ(1+1/shape) ≈ 7.2, K·E[gap] ≈ 43 ≪ 120)
    // ⇒ H(120) ≈ 58 expected events/subject, so every subject reaches ≥ K and the
    // first K gaps are unconditioned.
    const HORIZON: f64 = 120.0;
    const K: usize = 6;
    const SEED: u64 = 20260709;
    // Must match RTTE_WEIBULL_FWD_PIT_TRUTH (the oracle re-derives the same H).
    const SCALE: f64 = 8.0;
    const SHAPE: f64 = 1.5;

    let truth = parse_model_string(RTTE_WEIBULL_FWD_PIT_TRUTH).expect("PIT truth model must parse");
    // One right-censored template row per subject on CMT 2; `simulate_rtte_stream`
    // regenerates each subject's stream to the horizon.
    let template = common::tte_pop_from_pairs(&vec![(HORIZON, 0u8); N]);
    let opts = SimulateOptions {
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&truth, &template, &truth.default_params, 1, &opts)
        .expect("RTTE PIT simulation must succeed");

    // Independent closed-form cumulative hazard (no shared code with the conditional
    // sampler): H(t) = (t/scale)^shape.
    let cumhaz = |t: f64| (t / SCALE).powf(SHAPE);
    let mut by_id: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in &sims {
        if let SimOutcome::Event {
            time,
            observed: true,
        } = r.outcome
        {
            by_id.entry(r.id.clone()).or_default().push(time);
        }
    }
    // Rescaled increments for the first K events per subject (unconditioned; see the
    // doc comment). Assert every subject reached ≥ K events so no subject is dropped —
    // a dropped subject would reintroduce selection.
    let mut v: Vec<f64> = Vec::new();
    let mut min_events = usize::MAX;
    for times in by_id.values() {
        min_events = min_events.min(times.len());
        let mut h_prev = 0.0_f64;
        for &t in times.iter().take(K) {
            let h = cumhaz(t);
            v.push((-(h - h_prev)).exp());
            h_prev = h;
        }
    }
    assert_eq!(by_id.len(), N, "every subject must appear");
    assert!(
        min_events >= K,
        "every subject must reach ≥ K={K} events so the first-K gaps are unconditioned; \
         min was {min_events} — raise the horizon"
    );
    assert_eq!(v.len(), N * K, "N·K unconditioned increments");
    v.sort_by(|a, b| a.partial_cmp(b).expect("increments are finite"));

    // Kolmogorov–Smirnov against the FULLY-SPECIFIED Uniform(0,1) (truth parameters,
    // not estimated ⇒ the standard 1.36/√N critical is exact).
    let n = v.len() as f64;
    let mut d = 0.0_f64;
    for (i, &vi) in v.iter().enumerate() {
        d = d.max((i as f64 + 1.0) / n - vi).max(vi - i as f64 / n);
    }
    let crit = 1.36 / n.sqrt();
    eprintln!(
        "[PIT-RTTE] increments={} KS D={d:.4} vs 5% critical {crit:.4}",
        v.len()
    );
    assert!(
        d < crit,
        "RTTE forward sampler fails the KS goodness-of-fit against the model survival: \
         D={d:.4} ≥ {crit:.4} — the recurrent sampler does not reproduce H(t)"
    );
}

/// Per-subject observed-event counts from an RTTE `simulate()`, INCLUDING zero-event
/// subjects — every subject emits at least its censor row, so grouping on all rows and
/// counting the observed ones yields a 0 for the event-free subjects (dropping them
/// would bias the variance). Panics unless every one of `n` subjects appears.
fn rtte_event_counts(sims: &[SimulationResult], n: usize) -> Vec<f64> {
    let mut by_id = std::collections::BTreeMap::<String, usize>::new();
    for r in sims {
        let c = by_id.entry(r.id.clone()).or_default();
        if matches!(r.outcome, SimOutcome::Event { observed: true, .. }) {
            *c += 1;
        }
    }
    assert_eq!(
        by_id.len(),
        n,
        "every subject must appear (via its censor row)"
    );
    by_id.values().map(|&c| c as f64).collect()
}

/// **Slice 3.3 — the sampler injects the correct frailty variance.** For exponential
/// RTTE the per-subject count is `N_i ~ Poisson(λ_i·T)`, `λ_i = λ·exp(η_i)`, so
/// `E[N] = μ = λT·e^{ω²/2}` and `Var[N] = μ + μ²(e^{ω²} − 1)`, giving the
/// method-of-moments estimate `ω² = ln(1 + (Var − μ)/μ²)` straight from the simulated
/// counts — **no estimator (no fit) in the loop**.
///
/// The catch is that MoM is noisy when events/subject is small (~3): the count
/// over-dispersion is then a poor read on the frailty, and a single realization can sit
/// ±0.03 off (that noise, not a bias, is why the SSE below does not assert `ω²` at the
/// truth). Here the horizon is large (≈31 events/subject), so each subject's rate is
/// pinned precisely and the MoM read is tight — a clean, estimator-free confirmation
/// that `simulate()` injects `Var(log λ_i) = ω² = 0.09`. A sampler that over- or
/// under-dispersed the frailty would miss this band regardless of any fit.
#[test]
fn rtte_sampler_injects_correct_frailty_variance() {
    use ferx_core::{simulate_with_options, SimulateOptions};
    const N: usize = 4000;
    const HORIZON: f64 = 200.0; // ≈ λT = 30 events/subject ⇒ the frailty is well-identified
    const SEED: u64 = 3;

    let truth = parse_model_string(RTTE_EXP_TRUTH).expect("RTTE truth model must parse");
    let template = common::tte_pop_from_pairs(&vec![(HORIZON, 0u8); N]);
    let opts = SimulateOptions {
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&truth, &template, &truth.default_params, 1, &opts)
        .expect("RTTE simulation must succeed");

    let counts = rtte_event_counts(&sims, N);
    let mean_c = counts.iter().sum::<f64>() / N as f64;
    let var_c = counts.iter().map(|c| (c - mean_c).powi(2)).sum::<f64>() / (N as f64 - 1.0);
    let w2_mom = (1.0 + (var_c - mean_c) / (mean_c * mean_c)).ln();
    eprintln!("[FRAILTY-RTTE] mean={mean_c:.2} var={var_c:.2} → MoM ω²={w2_mom:.4} (truth 0.09)");
    // mean = λT·e^{ω²/2} = 0.15·200·e^0.045 ≈ 31.4.
    assert!(
        (30.0..33.0).contains(&mean_c),
        "mean events/subject {mean_c:.2} off λT·e^(ω²/2) ≈ 31.4"
    );
    assert!(
        (0.082..0.098).contains(&w2_mom),
        "method-of-moments ω² {w2_mom:.4} off truth 0.09 — the sampler injects the wrong \
         between-subject variance (no estimator here, so this isolates a sampler bug)"
    );
}

/// **Slice 3.3 SSE** — license-free round-trip validator for the RTTE generative path.
/// Simulate a clock-forward exponential RTTE dataset from known (θ, Ω) with ferx's own
/// `simulate()`, rebuild a fit population, and refit from a perturbed start. A wrong
/// event-time sampler, a dropped/duplicated event, or a broken sim→Population→fit
/// round-trip (wrong DV/censor layout) shows up here as non-recovery of the sharply
/// identified population rate `λ`.
///
/// **Scope.** `λ` is pinned by the total event count and recovers tightly. The frailty
/// `ω²` is only weakly identified at this realistic sparse design (~3 events/subject):
/// a single realization's empirical over-dispersion fluctuates around 0.09 by ±0.03, so
/// FOCEI lands high here (~0.13) simply because this seed's data happens to be
/// over-dispersed (its own MoM reads ~0.125). That is Monte-Carlo noise, **not** a
/// sampler or estimator bias — `rtte_sampler_injects_correct_frailty_variance` confirms
/// the true injected variance is 0.09 at high events/subject. So `ω²` is bracketed
/// around the seed-fixed estimate (guarding a gross fit regression), not asserted at the
/// truth — the same policy the joint PK-TTE SSE uses for its weakly-identified params.
#[test]
fn rtte_sse_forward_exponential_recovers_rate() {
    use ferx_core::{simulate_with_options, SimulateOptions};
    const N: usize = 1500;
    const HORIZON: f64 = 20.0;
    const SEED: u64 = 20260709;

    let truth = parse_model_string(RTTE_EXP_TRUTH).expect("RTTE truth model must parse");
    let template = common::tte_pop_from_pairs(&vec![(HORIZON, 0u8); N]);
    let opts = SimulateOptions {
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&truth, &template, &truth.default_params, 1, &opts)
        .expect("RTTE simulation must succeed");

    let counts = rtte_event_counts(&sims, N);
    let mean_c = counts.iter().sum::<f64>() / N as f64;
    let max_events = counts.iter().cloned().fold(0.0_f64, f64::max);
    eprintln!("[SSE-RTTE] mean events/subject={mean_c:.3} (~3.14) max={max_events}");
    assert!(
        (2.9..3.4).contains(&mean_c),
        "mean events/subject {mean_c:.3} off λT·e^(ω²/2) ≈ 3.14 — a 2x sampler rate error is out of band"
    );
    assert!(
        max_events >= 5.0,
        "a recurrent stream should reach ≥5 events for some subject; got {max_events}"
    );

    let fit_pop = rtte_pop_from_sims(&sims);
    let model = parse_model_string(RTTE_EXP_FIT).expect("RTTE fit model must parse");
    let r = fit(&model, &fit_pop, &model.default_params, &fit_opts())
        .expect("RTTE SSE fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE-RTTE] FOCEI λ={lambda:.5} (truth 0.15) ω²={omega2:.5} (weakly identified, see doc) OFV={:.4}",
        r.ofv
    );
    // λ is sharply identified ⇒ tight truth recovery (a factor-of-2 sampler error breaks it).
    assert!(
        (0.140..0.160).contains(&lambda),
        "lambda_pop not recovered: got {lambda:.5}, expected ~0.15"
    );
    // ω² brackets the seed-fixed FOCEI estimate (~0.129) — see the doc comment: this
    // realization is over-dispersed, and the truth-level variance check lives in
    // `rtte_sampler_injects_correct_frailty_variance`.
    assert!(
        (0.10..0.16).contains(&omega2),
        "FOCEI ω² {omega2:.5} outside the expected bracket for this realization"
    );
    assert!(r.ofv.is_finite(), "RTTE SSE OFV must be finite");
}

const RTTE_SIM_ANCHOR_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/rtte_exponential_sim/rtte_sim.csv"
);

/// **Slice 3.3 cross-tool simulation anchor** (the external, CLAUDE.md-required leg).
/// ferx *simulated* `tests/reference/rtte_exponential_sim/rtte_sim.csv` (300 subjects,
/// truth TVLAMBDA = 0.15, ω² = 0.09, horizon = 20; via `cargo run --bin rtte_sim_anchor
/// --features survival`). Here ferx **and NONMEM** both *fit* that ferx-simulated file:
/// an INDEPENDENT engine recovering the parameters from ferx-simulated data is the
/// external corroboration that the RTTE **simulator** is correct — a biased sampler
/// would move both engines off the data-generating values.
///
/// NONMEM LAPLACE (`nonmem.lst`, telescoping-AG `$PRED`): TVLAMBDA = 0.15617,
/// ω² = 0.12801, OFV = 5557.266. ferx FOCEI fits the identical file through the real
/// datareader and must reproduce them to a few significant figures (this reuses the
/// Slice 3.1 finding that ferx FOCEI ≡ NONMEM LAPLACE for constant-hazard RTTE). The ω²
/// sits above the data-generating 0.09 for *both* engines — this N = 300 realization is
/// over-dispersed; the sampler's true injected variance is pinned at 0.09, estimator-free,
/// by `rtte_sampler_injects_correct_frailty_variance`.
#[test]
fn rtte_sim_anchor_ferx_matches_nonmem() {
    // NONMEM LAPLACE on the same ferx-simulated file (see nonmem.lst / nonmem.ext).
    const NM_LAMBDA: f64 = 0.15617;
    const NM_OMEGA2: f64 = 0.12801;
    const NM_OFV: f64 = 5557.266;

    let model = parse_model_string(RTTE_EXP_FIT).expect("RTTE fit model must parse");
    let (pop, _cov) =
        read_population_for(&model, &None, RTTE_SIM_ANCHOR_CSV, None, None, None, &[])
            .expect("read ferx-simulated RTTE anchor CSV");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("ferx anchor fit");

    let (lambda, omega2) = (r.theta[0], r.omega[(0, 0)]);
    eprintln!(
        "[SIM-ANCHOR] ferx λ={lambda:.5} ω²={omega2:.5} OFV={:.3}  (NONMEM {NM_LAMBDA} / {NM_OMEGA2} / {NM_OFV})",
        r.ofv
    );
    assert!(
        (lambda - NM_LAMBDA).abs() < 0.002,
        "ferx λ={lambda:.5} should reproduce NONMEM {NM_LAMBDA} on the same ferx-simulated data"
    );
    assert!(
        (omega2 - NM_OMEGA2).abs() < 0.004,
        "ferx ω²={omega2:.5} should reproduce NONMEM {NM_OMEGA2} on the same ferx-simulated data"
    );
    assert!(
        (r.ofv - NM_OFV).abs() < 1.0,
        "ferx OFV={:.3} should reproduce NONMEM {NM_OFV} (same likelihood objective)",
        r.ofv
    );
}

const REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_exponential/tte_exp.csv"
);

/// `survival::survreg(Surv(TIME,DV) ~ 1, dist="exponential")` on `tte_exp.csv`.
/// Closed form `lambda = events / sum(time) = 82 / 1100.6`. Regenerate via
/// `tests/reference/tte_exponential/survreg.R`.
const SURVREG_LAMBDA: f64 = 0.074506;

/// `-2 logLik` reported by that same `survreg` exponential fit (printed by
/// `survreg.R`, tabulated in `expected.md`). The MLE argmax is invariant to a
/// dropped or doubled censored-likelihood normalizing constant, so matching
/// `lambda` alone cannot catch such a regression — the OFV must be anchored too
/// (the headline "likelihood constants are correct" rests on this number).
const SURVREG_LAMBDA_M2LL: f64 = 589.888;

// ── Tests ────────────────────────────────────────────────────────────────────

/// SSE: simulate from (lambda_pop=0.1, omega^2=0.25) with ferx's own `simulate()`,
/// apply administrative censoring at t=24 (matching the reference design), refit,
/// and assert both parameters are recovered. Guards the event-time sampler and the
/// censored-likelihood together.
#[test]
fn tte_sse_exponential_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 24.0;
    const SEED: u64 = 20260621;

    let truth = parse_model_string(EXP_TRUTH).expect("truth model must parse");
    let template = tte_sim_template(N, T_CENSOR);

    let sims = simulate_with_seed(&truth, &template, &truth.default_params, 1, SEED);
    assert_eq!(sims.len(), N, "one simulated event per template subject");

    let pairs = sims_to_pairs(&sims);

    let event_frac = pairs.iter().filter(|(_, dv)| *dv == 1).count() as f64 / N as f64;
    eprintln!("[SSE] event fraction = {event_frac:.4} (expected ~0.88 at lambda_pop=0.1, omega^2=0.25, censor t=24)");
    // At the truth (lambda_pop=0.1, omega^2=0.25, censor t=24) this N=2000 draw yields
    // ~0.88 events (the noisier 100-subject reference file lands at 0.82). The band
    // brackets that yet *excludes* a factor-of-2 sampler error, which the old
    // (0.5..0.98) did not: halving the rate (lambda~0.05) drops the fraction to ~0.66
    // (below 0.82) and doubling it pushes ~0.97 (above 0.93). The simulated data is
    // RNG-seed-deterministic, so this stays tight without being flaky.
    assert!(
        (0.82..0.93).contains(&event_frac),
        "simulated event fraction {event_frac:.4} off the expected ~0.88 — a 2x sampler error (rate halved → ~0.66, doubled → ~0.97) lands outside this band"
    );

    let model = parse_model_string(EXP_FIT).expect("fit model must parse");
    let pop = common::tte_pop_from_pairs(&pairs);
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("SSE fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!("[SSE] lambda_pop = {lambda:.5} (truth 0.1), omega^2 = {omega2:.5} (truth 0.25), OFV = {:.4}", r.ofv);

    // N is large enough that single-replicate Monte-Carlo noise is small, so these
    // bands test the *estimator* (near-unbiased recovery), not luck. Observed
    // (seed-fixed, deterministic): lambda ~= 0.099, omega^2 ~= 0.232 — the ~7%
    // omega^2 shortfall is the expected mild FOCEI-Laplace bias for TTE at this
    // event rate (plan §3.3), and grows at lower event rates. Bands are wide
    // enough to absorb that bias yet catch a gross sampler/likelihood error
    // (wrong constant, sign flip, factor-of-2).
    assert!(
        (0.090..0.110).contains(&lambda),
        "lambda_pop not recovered: got {lambda:.5}, expected ~0.1"
    );
    assert!(
        (0.200..0.275).contains(&omega2),
        "omega^2 not recovered: got {omega2:.5}, expected ~0.25 (mild Laplace bias allowed)"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// SSE for **competing risks**: simulate two cause-specific exponential hazards
/// (CMT 2, CMT 3) with a shared frailty via ferx's own `simulate()` (earliest
/// cause wins, the other censored), refit, and assert both cause-specific rates
/// and the frailty are recovered. Guards the competing-risks generative path
/// (earliest-of-causes + per-cause censoring) and the cause-specific likelihood
/// together. Linear (rate) frailty ⇒ FOCEI is near-unbiased (cf. the nonlinear
/// shape-frailty bias tracked in #469/#440).
#[test]
fn tte_sse_competing_risks_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 14.0;
    const SEED: u64 = 20260624;

    let truth = parse_model_string(COMPETING_TRUTH).expect("competing truth model must parse");
    let template = common::tte_competing_pop(&vec![(T_CENSOR, 0u8); N]);

    let sims = simulate_with_seed(&truth, &template, &truth.default_params, 1, SEED);
    assert_eq!(
        sims.len(),
        2 * N,
        "two rows per subject (one per cause CMT)"
    );

    // Cause-specific observed-event fractions (a subject contributes to at most one).
    let (mut f2, mut f3) = (0usize, 0usize);
    for r in &sims {
        if let SimOutcome::Event { observed: true, .. } = r.outcome {
            if r.cmt == 2 {
                f2 += 1;
            } else {
                f3 += 1;
            }
        }
    }
    let (frac2, frac3) = (f2 as f64 / N as f64, f3 as f64 / N as f64);
    eprintln!("[SSE-CR] cause-2 event frac = {frac2:.4}, cause-3 event frac = {frac3:.4}");
    // λ_A:λ_B = 0.10:0.06 ⇒ cause 2 (higher rate) produces more events; with τ=14 the
    // all-cause event rate is high, so both causes are well-populated and some subjects
    // are administratively censored.
    assert!(
        frac2 > frac3,
        "cause 2 (higher rate) must produce more events"
    );
    assert!(frac2 > 0.05 && frac3 > 0.05, "both causes must fire");

    let model = parse_model_string(COMPETING_FIT).expect("competing fit model must parse");
    let pop = competing_pop_from_sims(&sims);
    assert_eq!(
        pop.subjects.len(),
        N,
        "one fit subject per simulated subject"
    );

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("SSE fit must succeed");
    let (la, lb, om) = (r.theta[0], r.theta[1], r.omega[(0, 0)]);
    eprintln!(
        "[SSE-CR] lambda_A = {la:.5} (0.10), lambda_B = {lb:.5} (0.06), omega^2 = {om:.5} (0.25), OFV = {:.4}",
        r.ofv
    );

    // The cause-specific RATES are well-identified — assert tight recovery. These
    // are the headline competing-risks guard: a wrong cause→CMT routing, a dropped
    // censoring row, or a factor-of-2 in either hazard breaks them. (Deterministic
    // under RAYON_NUM_THREADS=1, as CI pins; seed-fixed values λ_A≈0.098, λ_B≈0.061.)
    assert!(
        (0.090..0.110).contains(&la),
        "lambda_A not recovered: {la:.5}"
    );
    assert!(
        (0.052..0.070).contains(&lb),
        "lambda_B not recovered: {lb:.5}"
    );

    // The SHARED frailty ω² is weakly identified (the −2LL is flat over a wide ω²
    // range — #469) AND over-estimated by FOCEI-Laplace at this all-cause event
    // rate (#440): a seed sweep gives 0.28–0.41 (truth 0.25), seed-20260624 ≈ 0.41.
    // This mirrors the single-cause exp(λ_tot) frailty problem to which competing
    // risks reduces, so it is NOT a competing-risks bug — the clean λ recovery
    // above is the proof. The band brackets the observed value with margin and
    // EXCLUDES the truth, so it characterises the bias and would fail (prompting a
    // re-baseline) if the Laplace frailty estimate is ever fixed to ~0.25.
    assert!(
        (0.32..0.50).contains(&om),
        "omega^2 {om:.5}: expected the FOCEI over-estimate ~0.41 (truth 0.25, weakly identified). \
         Below 0.32 may mean #440/#469 improved (FOCEI now nearer 0.25) — re-baseline this band."
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool: mixed-effects FOCEI fit of the committed reference dataset. Must
/// recover the data-generating parameters (this is the row NONMEM/nlmixr2 fill).
#[test]
fn tte_convergence_exponential_mixed() {
    let model = parse_model_string(EXP_FIT).expect("fit model must parse");
    let (pop, _cov) = read_population_for(&model, &None, REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 100, "reference dataset is 100 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[mixed] lambda_pop = {lambda:.5} (truth 0.1), omega^2 = {omega2:.5} (truth 0.25), OFV = {:.4}, converged = {}",
        r.ofv, r.converged
    );

    // Single 100-subject realisation (seed 42) is noisy, so the exact estimates
    // (deterministic: lambda ~= 0.0768, omega^2 ~= 0.290, OFV ~= 588.93) are what
    // gets tabulated next to NONMEM/nlmixr2 in
    // `tests/reference/tte_exponential/expected.md` — every tool fits *this* file,
    // so they should agree with each other regardless of the 0.1/0.25 truth. The
    // bands here are only a gross-failure guard.
    assert!(
        (0.06..0.14).contains(&lambda),
        "lambda_pop off: got {lambda:.5}, expected ~0.1"
    );
    assert!(
        (0.12..0.50).contains(&omega2),
        "omega^2 off: got {omega2:.5}, expected ~0.25"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool, exact: fixed-effects (n_eta=0) Exponential MLE must match the
/// base-R `survreg` closed-form rate on the same dataset to a tight tolerance.
#[test]
fn tte_convergence_exponential_fixed_matches_survreg() {
    let model = parse_model_string(EXP_FIT_FIXED).expect("fixed-effects model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) = read_population_for(&model, &None, REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let lambda = r.theta[0];
    eprintln!(
        "[fixed] lambda = {lambda:.6} (survreg {SURVREG_LAMBDA:.6}), OFV = {:.4}",
        r.ofv
    );

    let rel_err = (lambda - SURVREG_LAMBDA).abs() / SURVREG_LAMBDA;
    assert!(
        rel_err < 0.01,
        "fixed-effects rate {lambda:.6} must match survreg {SURVREG_LAMBDA:.6} within 1% (rel_err {rel_err:.4})"
    );
    // Anchor the OFV to survreg's -2logLik too (not just `is_finite`): this is the
    // number that pins the likelihood *constants*, which `lambda` cannot (#441 #2).
    assert!(
        (r.ofv - SURVREG_LAMBDA_M2LL).abs() < 1e-3,
        "fixed-effects OFV {:.4} must match survreg -2logLik {SURVREG_LAMBDA_M2LL} within 1e-3",
        r.ofv
    );
}

// ── RTTE (repeated TTE, clock-forward) — tests/reference/rtte_exponential ──────

const RTTE_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/rtte_exponential/rtte_exp.csv"
);

const RTTE_FIT_FIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.10, 0.001, 10.0)

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA
";

const RTTE_FIT_MIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Cross-tool, exact: a fixed-effects (n_eta=0) constant-hazard RTTE fit must recover the
/// analytic pooled Poisson-process MLE `lambda = D / Σ_i T_i = 305 / 2000 = 0.15250` on
/// the committed dataset (the RTTE analogue of the survreg exponential anchor), and the
/// OFV must equal the closed-form `-2 logL`. Both are exact and license-free. Critically,
/// this fails on the pre-RTTE per-record accumulation: summing independent single-event
/// terms over-counts the cumulative hazard by `Σ_k H(t_k)`, so both `lambda` and the OFV
/// would be wrong.
#[test]
fn rtte_convergence_fixed_matches_mle() {
    let model = parse_model_string(RTTE_FIT_FIXED).expect("fixed RTTE model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");
    let (pop, _cov) = read_population_for(&model, &None, RTTE_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    // rtte_exp.csv: D = 305 events; 100 subjects each observed over [0, 20] ⇒ exposure 2000.
    let lambda_mle: f64 = 305.0 / 2000.0; // 0.15250
    let m2ll_mle: f64 = -2.0 * (305.0 * lambda_mle.ln() - lambda_mle * 2000.0);
    let lambda = r.theta[0];
    eprintln!(
        "[rtte fixed] lambda = {lambda:.6} (MLE {lambda_mle:.6}), OFV = {:.4} (analytic {m2ll_mle:.4})",
        r.ofv
    );

    assert!(
        (lambda - lambda_mle).abs() / lambda_mle < 1e-3,
        "fixed RTTE rate {lambda:.6} must match analytic MLE {lambda_mle:.6}"
    );
    assert!(
        (r.ofv - m2ll_mle).abs() < 1e-2,
        "fixed RTTE OFV {:.4} must match analytic -2logL {m2ll_mle:.4}",
        r.ofv
    );
}

/// Cross-tool: a mixed-effects (frailty) RTTE FOCEI fit must reproduce NONMEM LAPLACE and
/// the exact Poisson-lognormal GLMM MLE on the committed dataset — TVLAMBDA ≈ 0.1406,
/// omega^2 ≈ 0.1645, OFV ≈ 1748.49 (see `tests/reference/rtte_exponential/expected.md`).
/// FOCEI is deterministic, so the bands are tight. The frailty omega^2 is well-identified
/// here because the data is event-rich; the Karlsson Laplace bias is a low-event-rate
/// effect, so FOCEI, SAEM, NONMEM and the GLMM all coincide on this design.
#[test]
fn rtte_convergence_mixed_matches_nonmem() {
    let model = parse_model_string(RTTE_FIT_MIXED).expect("mixed RTTE model must parse");
    let (pop, _cov) = read_population_for(&model, &None, RTTE_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[rtte mixed] TVLAMBDA = {lambda:.5} omega^2 = {omega2:.5} OFV = {:.4}  (NONMEM 0.14062 / 0.16450 / 1748.49)",
        r.ofv
    );

    // NONMEM LAPLACE: 0.140618 / 0.164496 / 1748.493. Poisson-LN GLMM MLE: 0.14062 / 0.16455.
    assert!(
        (0.132..0.150).contains(&lambda),
        "TVLAMBDA {lambda:.5} off the NONMEM/GLMM value ~0.1406"
    );
    assert!(
        (0.140..0.190).contains(&omega2),
        "omega^2 {omega2:.5} off the NONMEM/GLMM value ~0.1645"
    );
    assert!(
        (r.ofv - 1748.493).abs() < 0.2,
        "OFV {:.4} must match NONMEM LAPLACE 1748.493",
        r.ofv
    );
}

// ── RTTE clock-reset (gap time) — tests/reference/rtte_weibull_reset ───────────

const RTTE_RESET_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/rtte_weibull_reset/rtte_weibull_reset.csv"
);

const RTTE_RESET_FIT_FIXED: &str = r"
[parameters]
  theta TVSCALE(3.0, 0.1, 100.0)
  theta TVSHAPE(1.0, 0.1, 10.0)

[event_model]
  cmt    = 2
  type   = rtte
  clock  = reset
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";

const RTTE_RESET_FIT_MIXED: &str = r"
[parameters]
  theta TVSCALE(5.0, 0.1, 100.0)
  theta TVSHAPE(1.5, 0.1, 10.0)
  omega ETA_SCALE ~ 0.09

[event_model]
  cmt    = 2
  type   = rtte
  clock  = reset
  family = weibull
  scale  = TVSCALE * exp(ETA_SCALE)
  shape  = TVSHAPE
";

/// Cross-tool, exact: a fixed-effects (n_eta=0) clock-reset Weibull RTTE fit must match
/// `survreg(Surv(gap, event) ~ 1, dist="weibull")` on the inter-event gap durations —
/// under clock-reset the gaps are independent Weibull observations, so the reset RTTE
/// likelihood reduces exactly to that regression (see `survreg.R`). This also pins that
/// the gap bookkeeping is right: a clock-forward accumulation on the same data would give
/// a different scale/shape/OFV.
#[test]
fn rtte_reset_convergence_fixed_matches_survreg() {
    let model = parse_model_string(RTTE_RESET_FIT_FIXED).expect("fixed reset model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");
    let (pop, _cov) = read_population_for(&model, &None, RTTE_RESET_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let (scale, shape) = (r.theta[0], r.theta[1]);
    eprintln!(
        "[rtte reset fixed] scale = {scale:.5} shape = {shape:.5} OFV = {:.4}  (survreg 4.78920 / 1.32415 / 3243.856)",
        r.ofv
    );

    // survreg on the gap durations: scale 4.78920, shape 1.32415, -2logL 3243.856.
    assert!(
        (scale - 4.78920).abs() / 4.78920 < 2e-3,
        "reset scale {scale:.5} must match survreg 4.78920"
    );
    assert!(
        (shape - 1.32415).abs() / 1.32415 < 2e-3,
        "reset shape {shape:.5} must match survreg 1.32415"
    );
    assert!(
        (r.ofv - 3243.856).abs() < 1e-2,
        "reset OFV {:.4} must match survreg -2logL 3243.856",
        r.ofv
    );
}

/// Cross-tool: a mixed-effects (frailty) clock-reset Weibull RTTE FOCEI fit must
/// reproduce NONMEM LAPLACE on the committed dataset — TVSCALE ~ 5.16, TVSHAPE ~ 1.53,
/// omega^2 ~ 0.132, OFV ~ 3175.86 (see `tests/reference/rtte_weibull_reset/expected.md`).
#[test]
fn rtte_reset_convergence_mixed_matches_nonmem() {
    let model = parse_model_string(RTTE_RESET_FIT_MIXED).expect("mixed reset model must parse");
    let (pop, _cov) = read_population_for(&model, &None, RTTE_RESET_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let (scale, shape, omega2) = (r.theta[0], r.theta[1], r.omega[(0, 0)]);
    eprintln!(
        "[rtte reset mixed] scale = {scale:.5} shape = {shape:.5} omega^2 = {omega2:.5} OFV = {:.4}  (NONMEM 5.1594 / 1.5299 / 0.1322 / 3175.86)",
        r.ofv
    );

    // NONMEM LAPLACE: 5.15939 / 1.52987 / 0.13225 / 3175.863.
    assert!(
        (5.0..5.35).contains(&scale),
        "scale {scale:.5} off the NONMEM value ~5.16"
    );
    assert!(
        (1.45..1.60).contains(&shape),
        "shape {shape:.5} off the NONMEM value ~1.53"
    );
    assert!(
        (0.10..0.17).contains(&omega2),
        "omega^2 {omega2:.5} off the NONMEM value ~0.132"
    );
    assert!(
        (r.ofv - 3175.863).abs() < 0.3,
        "OFV {:.4} must match NONMEM LAPLACE 3175.863",
        r.ofv
    );
}

/// Cross-tool, exact: a FIXED-EFFECTS (n_eta=0) competing-risks exponential fit
/// must recover each cause's closed-form MLE `λ̂_k = d_k / Σ_i t_i` — identical to
/// base-R `survreg(dist="exponential")` fitted per cause (other-cause events as
/// censoring). Without a shared frailty the cause-specific likelihood factorises
/// into independent per-cause exponential regressions, so this anchors the
/// cause-specific hazard likelihood against an external reference (the
/// competing-risks analogue of `tte_convergence_exponential_fixed_matches_survreg`).
#[test]
fn tte_competing_fixed_matches_per_cause_mle() {
    // data/tte_competing_risks.csv: N=40, d_A(CMT2)=23, d_B(CMT3)=15, Σt=201.6552
    //   ⇒ λ̂_A = 23/201.6552 = 0.114056,  λ̂_B = 15/201.6552 = 0.074384
    //   (verified equal to survreg(Surv(time, cause==k) ~ 1, dist="exponential")).
    const MLE_A: f64 = 0.114056;
    const MLE_B: f64 = 0.074384;
    const COMPETING_CSV: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/data/tte_competing_risks.csv");

    let model = parse_model_string(COMPETING_FIXED).expect("fixed competing model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) = read_population_for(&model, &None, COMPETING_CSV, None, None, None, &[])
        .expect("competing CSV reads");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fixed competing fit");
    let (la, lb) = (r.theta[0], r.theta[1]);
    eprintln!(
        "[CR-fixed] lambda_A = {la:.6} (MLE {MLE_A:.6}), lambda_B = {lb:.6} (MLE {MLE_B:.6}), OFV = {:.4}",
        r.ofv
    );

    let rel_a = (la - MLE_A).abs() / MLE_A;
    let rel_b = (lb - MLE_B).abs() / MLE_B;
    assert!(
        rel_a < 0.01,
        "lambda_A {la:.6} must match per-cause MLE {MLE_A:.6} within 1% (rel {rel_a:.4})"
    );
    assert!(
        rel_b < 0.01,
        "lambda_B {lb:.6} must match per-cause MLE {MLE_B:.6} within 1% (rel {rel_b:.4})"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ═══════════════════════════ Weibull ═══════════════════════════════════════════
//
// Validation dataset puts between-subject variability on the *shape* (scale fixed):
// scale_pop = 20, shape_pop = 2, omega^2(log shape) = 0.20, censored at t = 30.

const WEIBULL_TRUTH: &str = r"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(2.0,  0.1, 10.0)
  omega ETA_SHAPE ~ 0.20

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE * exp(ETA_SHAPE)
";

const WEIBULL_FIT: &str = r"
[parameters]
  theta TVSCALE(15.0, 0.1, 500.0)
  theta TVSHAPE(1.2,  0.1, 10.0)
  omega ETA_SHAPE ~ 0.05

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE * exp(ETA_SHAPE)
";

const WEIBULL_FIT_FIXED: &str = r"
[parameters]
  theta TVSCALE(15.0, 0.1, 500.0)
  theta TVSHAPE(1.2,  0.1, 10.0)

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";

const WEIBULL_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_weibull/tte_weibull.csv"
);

// survreg(Surv(TIME,DV)~1, dist="weibull") on tte_weibull.csv, mapped to ferx's
// (scale, shape) via shape = 1/scale_sr, scale = exp(intercept). See survreg.R.
const SURVREG_WEIBULL_SHAPE: f64 = 2.119250;
const SURVREG_WEIBULL_SCALE: f64 = 22.176599;
/// `-2 logLik` of the same `survreg` weibull fit (see `survreg.R` / `expected.md`).
/// Anchors the fixed-effects OFV so the likelihood constants are pinned, not just
/// the (scale, shape) argmax.
const SURVREG_WEIBULL_M2LL: f64 = 640.261;

/// SSE: recover (scale=20, shape=2, omega^2=0.20) from a large simulated dataset.
#[test]
fn tte_sse_weibull_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 30.0;
    const SEED: u64 = 20260622;

    let truth = parse_model_string(WEIBULL_TRUTH).expect("truth model must parse");
    let sims = simulate_with_seed(
        &truth,
        &tte_sim_template(N, T_CENSOR),
        &truth.default_params,
        1,
        SEED,
    );

    let pairs = sims_to_pairs(&sims);

    let model = parse_model_string(WEIBULL_FIT).expect("fit model must parse");
    let r = fit(
        &model,
        &common::tte_pop_from_pairs(&pairs),
        &model.default_params,
        &fit_opts(),
    )
    .expect("SSE fit must succeed");

    let scale = r.theta[0];
    let shape = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE weibull] scale = {scale:.4} (truth 20), shape = {shape:.4} (truth 2), omega^2 = {omega2:.4} (truth 0.20)"
    );

    // Structural parameters recover tightly under FOCEI.
    assert!(
        (18.5..21.5).contains(&scale),
        "scale not recovered: got {scale:.4}, expected ~20"
    );
    assert!(
        (1.85..2.15).contains(&shape),
        "shape not recovered: got {shape:.4}, expected ~2"
    );
    // omega^2 (frailty on the *shape* — a nonlinear hazard parameter) is materially
    // OVER-estimated by FOCEI-Laplace: this seed deterministically gives ~0.344 vs a
    // true 0.20 (+72%), and the bias does NOT vanish as the true omega^2 shrinks
    // (truth 0.05 → ~0.16). A SAEM fit of the same data reads ~0.13 — the estimators
    // straddle the truth, confirming a FOCEI *approximation* limitation for
    // nonlinear-parameter frailty (plan §3.3/§13: SAEM/IMP preferred for TTE), NOT a
    // likelihood bug (the fixed-effects fit matches survreg exactly). Tracked in #440.
    //
    // This band CHARACTERIZES that bias on purpose: it brackets the current biased
    // value (~0.344) with platform margin but EXCLUDES both the truth (0.20) and the
    // SAEM value (0.13). So when #440 is resolved and FOCEI recovers omega^2 ~ 0.20,
    // this assertion will FAIL — the intended signal to revisit it (assert recovery to
    // the truth and drop the over-estimation note). A wide "sane-range" band that
    // admitted both the broken and the fixed value could never flip, letting #440 close
    // by attrition with no test ever signalling (#441 review #4).
    assert!(
        (0.29..0.40).contains(&omega2),
        "omega^2 {omega2:.4}: expected the documented FOCEI over-estimate ~0.34 (truth 0.20). \
         Below 0.29 likely means #440 is fixed (FOCEI now recovers ~0.20) — update this test; \
         above 0.40 means the shape frailty exploded."
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool, exact: fixed-effects (n_eta=0) Weibull MLE must match `survreg`
/// (both scale and shape) on the same dataset.
#[test]
fn tte_convergence_weibull_fixed_matches_survreg() {
    let model = parse_model_string(WEIBULL_FIT_FIXED).expect("fixed-effects model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) = read_population_for(&model, &None, WEIBULL_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    let scale = r.theta[0];
    let shape = r.theta[1];
    eprintln!(
        "[fixed weibull] scale = {scale:.4} (survreg {SURVREG_WEIBULL_SCALE:.4}), shape = {shape:.4} (survreg {SURVREG_WEIBULL_SHAPE:.4}), OFV = {:.4}",
        r.ofv
    );

    let scale_err = (scale - SURVREG_WEIBULL_SCALE).abs() / SURVREG_WEIBULL_SCALE;
    let shape_err = (shape - SURVREG_WEIBULL_SHAPE).abs() / SURVREG_WEIBULL_SHAPE;
    assert!(
        scale_err < 0.01,
        "scale {scale:.4} must match survreg {SURVREG_WEIBULL_SCALE:.4} within 1% (rel_err {scale_err:.4})"
    );
    assert!(
        shape_err < 0.01,
        "shape {shape:.4} must match survreg {SURVREG_WEIBULL_SHAPE:.4} within 1% (rel_err {shape_err:.4})"
    );
    // Pin the OFV to survreg's -2logLik (the likelihood-constants anchor, #441 #2).
    assert!(
        (r.ofv - SURVREG_WEIBULL_M2LL).abs() < 1e-3,
        "fixed-effects OFV {:.4} must match survreg -2logLik {SURVREG_WEIBULL_M2LL} within 1e-3",
        r.ofv
    );
}

/// Cross-tool: mixed-effects (frailty-on-shape) FOCEI fit of the committed Weibull
/// dataset — the row NONMEM/nlmixr2 fill. Records ferx's estimates; the FOCEI
/// over-estimation of the shape-frailty omega^2 (#440) shows up here on real data too.
#[test]
fn tte_convergence_weibull_mixed() {
    let model = parse_model_string(WEIBULL_FIT).expect("fit model must parse");
    let (pop, _cov) = read_population_for(&model, &None, WEIBULL_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 100, "reference dataset is 100 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let scale = r.theta[0];
    let shape = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[weibull mixed] scale = {scale:.4} (truth 20), shape = {shape:.4} (truth 2), omega^2 = {omega2:.4} (truth 0.20), OFV = {:.4}",
        r.ofv
    );

    // Gross-failure guard; exact estimates are tabulated in expected.md.
    assert!(
        (15.0..26.0).contains(&scale),
        "scale off: got {scale:.4}, expected ~20"
    );
    assert!(
        (1.5..2.8).contains(&shape),
        "shape off: got {shape:.4}, expected ~2"
    );
    // On this single n=100 realisation ferx's shape-frailty omega^2 is ~0.176 — on the
    // NONMEM LAPLACIAN (0.175) / nlmixr2 FOCEI (0.173) consensus. Before #469 ferx read
    // 0.204 here: the derivative-free BOBYQA outer optimizer false-converged on the
    // near-flat ω² ridge (the whole 0.175→0.204 span is <0.01 OFV) because its ftol_rel
    // default (1e-6) stopped it short of ferx's *own* profile minimum (which already sat
    // at ~0.175). Tightening the TTE ftol to 1e-8 (#469; auto-selected for pure-TTE
    // models, whose hazard objective is evaluated exactly) walks it down to the
    // consensus. This is a pure optimizer-convergence fix, distinct from the FOCEI-
    // Laplace *method* bias on nonlinear frailty (the large-N SSE above still reads
    // ~0.34 — that bias is unchanged, and tracked by #440). Band brackets the
    // deterministic 0.176 and EXCLUDES the pre-#469 0.204, so a regression in the outer
    // ftol (or its TTE auto-selection) fails here.
    assert!(
        (0.16..0.19).contains(&omega2),
        "omega^2 {omega2:.4} off ferx's post-#469 ~0.176 (NONMEM 0.175 / nlmixr2 0.173); \
         a value near 0.204 means the BOBYQA TTE ftol tightening regressed. See expected.md / #469"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ═══════════════════════════ Gompertz ══════════════════════════════════════════
//
// Validation dataset is a fixed-effects 2-arm RCT: h = alpha*exp(gamma*t)*exp(loghr*TRT),
// alpha = exp(-6) ≈ 0.00248, gamma = exp(-5.4) ≈ 0.00450, loghr = -0.8, censored at 365.
// No survreg/base-R anchor exists for Gompertz, so recovery is the guard (NONMEM/nlmixr2
// are the cross-tool hand-off). The fit exercises the [event_model] covariate + loghr path.

const GOMPERTZ_FIT: &str = r"
[parameters]
  theta TVALPHA(0.001, 1e-6, 1.0)
  theta TVGAMMA(0.003, 1e-5, 5.0)
  theta LHR(-0.3, -5.0, 5.0)

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA
  loghr  = LHR * TRT
";

const GOMPERTZ_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_gompertz/tte_gompertz.csv"
);

/// Frailty-Gompertz SSE truth (BSV on gamma) — recover alpha, gamma, omega^2.
const GOMPERTZ_TRUTH_FRAILTY: &str = r"
[parameters]
  theta TVALPHA(0.002, 1e-6, 1.0)
  theta TVGAMMA(0.05,  1e-5, 5.0)
  omega ETA_GAMMA ~ 0.05

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)
";

const GOMPERTZ_FIT_FRAILTY: &str = r"
[parameters]
  theta TVALPHA(0.001, 1e-6, 1.0)
  theta TVGAMMA(0.03,  1e-5, 5.0)
  omega ETA_GAMMA ~ 0.02

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)
";

/// Cross-tool: fixed-effects Gompertz RCT fit on the committed dataset. Recovers
/// the data-generating alpha/gamma/loghr and exercises the TRT covariate path.
/// (NONMEM/nlmixr2 columns are the hand-off.)
#[test]
fn tte_convergence_gompertz_rct_recovers() {
    let model = parse_model_string(GOMPERTZ_FIT).expect("gompertz model must parse");
    assert_eq!(model.n_eta, 0, "RCT model is fixed-effects");
    assert!(
        model.referenced_covariates.contains(&"TRT".to_string()),
        "TRT must be picked up from the loghr expression"
    );

    let (pop, _cov) = read_population_for(&model, &None, GOMPERTZ_REF_CSV, None, None, None, &[])
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 300, "reference dataset is 300 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let alpha = r.theta[0];
    let gamma = r.theta[1];
    let loghr = r.theta[2];
    eprintln!(
        "[gompertz RCT] alpha = {alpha:.6} (truth 0.00248), gamma = {gamma:.6} (truth 0.00450), loghr = {loghr:.4} (truth -0.8), OFV = {:.4}",
        r.ofv
    );

    assert!(
        (0.0012..0.0045).contains(&alpha),
        "alpha off: got {alpha:.6}, expected ~0.00248"
    );
    assert!(
        (0.0025..0.0070).contains(&gamma),
        "gamma off: got {gamma:.6}, expected ~0.00450"
    );
    assert!(
        (-1.3..-0.3).contains(&loghr),
        "loghr off: got {loghr:.4}, expected ~-0.8"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// SSE: frailty-Gompertz (BSV on gamma) recovery of alpha/gamma/omega^2.
#[test]
fn tte_sse_gompertz_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 80.0;
    const SEED: u64 = 20260623;

    let truth = parse_model_string(GOMPERTZ_TRUTH_FRAILTY).expect("truth model must parse");
    let sims = simulate_with_seed(
        &truth,
        &tte_sim_template(N, T_CENSOR),
        &truth.default_params,
        1,
        SEED,
    );

    let pairs = sims_to_pairs(&sims);

    let model = parse_model_string(GOMPERTZ_FIT_FRAILTY).expect("fit model must parse");
    let r = fit(
        &model,
        &common::tte_pop_from_pairs(&pairs),
        &model.default_params,
        &fit_opts(),
    )
    .expect("SSE fit must succeed");

    let alpha = r.theta[0];
    let gamma = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE gompertz] alpha = {alpha:.5} (truth 0.002), gamma = {gamma:.5} (truth 0.05), omega^2 = {omega2:.4} (truth 0.05)"
    );

    // alpha and gamma trade off (the Gompertz baseline/growth collinearity), so
    // allow a wider alpha band; gamma is the better-determined of the two.
    assert!(
        (0.0012..0.0028).contains(&alpha),
        "alpha not recovered: got {alpha:.5}, expected ~0.002"
    );
    assert!(
        (0.043..0.060).contains(&gamma),
        "gamma not recovered: got {gamma:.5}, expected ~0.05"
    );
    // omega^2 (frailty on *gamma*, a nonlinear hazard parameter) is over-estimated by
    // FOCEI-Laplace — this seed deterministically gives ~0.081 vs truth 0.05 (+62%),
    // the same nonlinear-frailty limitation seen for the Weibull shape (#440). As for
    // the Weibull SSE, this is a CHARACTERIZATION band: it brackets the biased value
    // but EXCLUDES the truth (0.05), so it flips (fails) once #440 lets FOCEI recover
    // ~0.05 — the signal to revisit. Documented in expected.md.
    assert!(
        (0.065..0.105).contains(&omega2),
        "omega^2 {omega2:.4}: expected the documented FOCEI over-estimate ~0.08 (truth 0.05). \
         Below 0.065 likely means #440 is fixed (recovers ~0.05) — update this test; \
         above 0.105 means the gamma frailty exploded."
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ── Joint PK-TTE (Phase 2, Slice 2.1 — #564) ─────────────────────────────────

/// Oral 1-cpt PK + drug-driven (ODE-accumulated) hazard. The hazard reads the
/// `central` ODE state, so it is integrated as a cumulative-hazard compartment and
/// estimated jointly with the PK.
const JOINT_PKTTE_FIT: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method  = focei
  maxiter = 3
";

/// End-to-end joint PK-TTE FOCEI fit must complete with a finite OFV. Nightly
/// Tier-3 guard: the augmented PK+CHZ integration (re-solved per inner eval and per
/// FD-Hessian perturbation) makes a full fit slow, so per-PR coverage of the
/// ODE-hazard likelihood lives in the fast `joint_pktte_ode_hazard_nll_paths_finite`
/// unit test instead. A simulation-estimation (SSE) recovery check awaits the
/// Slice 2.2 event-time root-finder. See #564.
#[test]
fn joint_pktte_focei_fit_completes() {
    use ferx_core::types::{DoseEvent, EventType, ObsRecord};

    let model = parse_model_string(JOINT_PKTTE_FIT).expect("joint PK-TTE model must parse");

    // 6 oral-dose subjects: 2 PK obs (central amount) + one TTE record on CMT 2,
    // mixing exact events and a window censor so both hazard arms are exercised.
    let event_times = [5.0, 12.0, 30.0, 8.0, 20.0, 3.0];
    let subjects = (0..event_times.len())
        .map(|i| {
            let mut s = common::subject(
                &format!("{}", i + 1),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                vec![2.0, 8.0],
                vec![30.0, 20.0],
                vec![1, 1],
            );
            let censored = event_times[i] >= 30.0;
            s.obs_records = vec![ObsRecord::Event {
                time: event_times[i],
                event_type: if censored {
                    EventType::RightCensored
                } else {
                    EventType::Exact
                },
                entry_time: 0.0,
                cmt: 2,
            }];
            s
        })
        .collect();
    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    };

    let mut opts = FitOptions::default();
    opts.verbose = false;
    match fit(&model, &pop, &model.default_params, &opts) {
        Ok(r) => assert!(
            r.ofv.is_finite(),
            "joint PK-TTE FOCEI OFV must be finite; got {}",
            r.ofv
        ),
        Err(e) => panic!("joint PK-TTE FOCEI fit must not error: {e}"),
    }
}
