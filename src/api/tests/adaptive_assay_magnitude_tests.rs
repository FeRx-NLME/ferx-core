//! The adaptive controller's assay applies the residual-error magnitude (#2741).
//!
//! A `Dv` reading's variance is `simulate()`'s variance for an observation row at the
//! decision: the same magnitude evaluator (`RuvMagnitude::eval_key`), the same variance
//! tail (`residual_variance_for_key`), read at the decision's covariate row, written
//! `TIME` and causal `TAD`. Before #2741 the assay drew from the bare σ and the run
//! reported success; #2746 refused the combination, and this lift removes the refusal.
//!
//! Adaptive-dosing exception (AGENTS.md): no NONMEM. The oracles are the bare-model bits
//! pinned before the lift (T1), `simulate()`'s own variance function on the replayed
//! subject (T3), the reading of a magnitude-free twin (T2, T7, T9) and hand-computed
//! `TAD` / `TIME` values (T4, T5, T8). Every run here is the ODE reactive driver
//! (`ode_predictions_adaptive_impl`); no fit, so no gradient engine is involved.
//!
//! How a test reads the assay's variance: a `Dv` monitor `A` and an `Ipred` monitor `L`
//! on the same compartment. `A − L = √v · ε`, where `ε` is the run's own draw on the
//! controller-assay substream, keyed by (seed, subject, replicate, decision, "A"), so it
//! does not depend on the model and cancels in a ratio between twins.
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    assay_standard_normal, subject_assay_base_seed, ControllerCtx, DoseAction, MonitorSpec,
    ObserveMode,
};

const SEED: u64 = 7;

/// The structural core every fixture shares: 1-cpt ODE with a second, independent state
/// (so a `CMT=2:` line has an endpoint), Form C readout `y = central`, which does not
/// clamp: a reading may be negative, and a deviation ratio is exact on either sign (no
/// `0` floor to break it). `{theta}`, `{omega}`, `{sigma}`, `{error}` and `{tail}` are the
/// per-form parts.
fn model_src(theta: &str, omega: &str, sigma: &str, error: &str, tail: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
{theta}
  omega ETA_CL ~ 0.09
{omega}
{sigma}
{tail}
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central, periph])
[odes]
  d/dt(central) = -(CL / V) * central
  d/dt(periph) = -0.2 * periph
[scaling]
  y = central
[error_model]
{error}
[covariates]
  MAG continuous
"#
    )
}

fn model(theta: &str, omega: &str, sigma: &str, error: &str, tail: &str) -> CompiledModel {
    parse_model_string(&model_src(theta, omega, sigma, error, tail)).expect("parse")
}

/// `rows`: one CSV line per record after the header `ID,TIME,DV,EVID,AMT,CMT,MDV,MAG`
/// (plus whatever `extra_cols` names), all for subject 1.
fn read(extra_cols: &str, rows: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let csv = format!("ID,TIME,DV,EVID,AMT,CMT,MDV,MAG{extra_cols}\n{rows}");
    std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap()
}

/// Observation records at each decision (0, 4, 8) and one after (10), `MAG` 2 → 3 → 5,
/// so the magnitude is live, differs across decisions, and is never `1`.
const OBS_AT_DECISIONS: &str = "1,0,1,0,0,1,0,2\n1,4,1,0,0,1,0,3\n1,8,1,0,0,1,0,5\n\
                                1,10,1,0,0,1,0,5\n";
const DECISIONS: [f64; 3] = [0.0, 4.0, 8.0];

fn two_monitors(decisions: &[f64]) -> AdaptiveSimulateOptions {
    AdaptiveSimulateOptions {
        seed: Some(SEED),
        decision_times: decisions.to_vec(),
        monitors: vec![
            MonitorSpec::new("A", 1, ObserveMode::Dv),
            MonitorSpec::new("L", 1, ObserveMode::Ipred),
        ],
        ..Default::default()
    }
}

/// A fixed 100 mg bolus at every decision: the ledger cannot depend on the reading.
fn fixed_bolus() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
}

/// One run's readings per decision: `(A, L, ε)`.
struct Readings {
    a: Vec<f64>,
    l: Vec<f64>,
    eps: Vec<f64>,
}

impl Readings {
    fn of(res: &AdaptiveSimulationResult) -> Self {
        let base = subject_assay_base_seed(SEED, "1", 1);
        let pick = |name: &str| -> Vec<f64> {
            res.decisions
                .iter()
                .map(|d| {
                    d.observed_signals
                        .iter()
                        .find(|s| s.name == name)
                        .expect("monitor read")
                        .value
                })
                .collect()
        };
        let (a, l) = (pick("A"), pick("L"));
        let eps = (0..a.len())
            .map(|k| assay_standard_normal(base, k, "A"))
            .collect();
        Readings { a, l, eps }
    }

    /// The deviation `A − L` at decision `k`, asserted finite and nonzero: a `0` would be
    /// a clamped or zero-variance reading, which no ratio can see through.
    fn dev(&self, k: usize) -> f64 {
        let d = self.a[k] - self.l[k];
        assert!(d.is_finite() && d != 0.0, "decision {k}: A − L = {d}");
        d
    }

    /// The assay's variance at decision `k`, recovered from the draw: `((A − L) / ε)²`.
    fn var(&self, k: usize) -> f64 {
        let s = self.dev(k) / self.eps[k];
        s * s
    }
}

fn relerr(got: f64, want: f64) -> f64 {
    assert!(got.is_finite() && want.is_finite(), "{got} vs {want}");
    ((got - want) / want).abs()
}

// ── T1: the bare gate ────────────────────────────────────────────────────────────────

/// `ODE_NO_IIV` of `adaptive_sim_tests.rs`, verbatim: the fixture the bare bits were
/// measured on.
const ODE_NO_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

/// The `Dv` readings of a bare model, bit for bit as before the lift: `ODE_NO_IIV`,
/// observations 2/6/10, decisions 0/4/8, the `dv_threshold` controller, seed 7. Measured
/// at `c5dac24b` (plan P1) and again at `3cd15863`, the base of this change.
///
/// Through `simulate_adaptive_from_spec`, the same model under an assay-error spec: each
/// reading must be `L + √v·ε` with `v` the legacy form (`variance_at_with_correlations`),
/// recomputed here, `L` read by an `Ipred` monitor on a programmatic replay of the spec's
/// own doses (which draws the same `ε`, under the spec's monitor name).
///
/// Regression: a bare model handed `Some(row of ones)` (`_scaled`), which differs from the
/// legacy form in the last bit on 55% of predictions (plan P2), or any change to the draw.
#[test]
fn bare_model_readings_keep_their_bits() {
    let m = parse_model_string(ODE_NO_IIV).expect("parse");
    assert!(!m.has_custom_ruv_magnitude(), "the fixture is bare");
    let pop = read("", "1,2,1,0,0,1,0,1\n1,6,1,0,0,1,0,1\n1,10,1,0,0,1,0,1\n");
    let opts = AdaptiveSimulateOptions {
        seed: Some(SEED),
        decision_times: DECISIONS.to_vec(),
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
        ..Default::default()
    };
    let dv_threshold = || {
        |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A declared") < 50.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        }
    };
    let res =
        simulate_adaptive(&m, &pop, &m.default_params, 1, dv_threshold, &opts).expect("bare run");
    let got: Vec<f64> = res
        .decisions
        .iter()
        .map(|d| d.observed_signals[0].value)
        .collect();
    let want: [f64; 3] = [1.036675500319689e-6, 57.583656434580604, 35.79057937490115];
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "decision {k}: {g} vs {w}");
    }

    // The spec entry point, on records at each decision.
    let pop = read("", OBS_AT_DECISIONS);
    let spec = crate::sim::adaptive::AdaptiveDosingSpec {
        observe: None,
        observe_declared_covariates: Vec::new(),
        with_assay_error: true,
        assay_cmt: Some(1),
        at: DECISIONS.to_vec(),
        start_dose: (100.0).into(),
        route: crate::sim::adaptive::AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: ((0.0).into(), (400.0).into()),
        confirm: 1,
        round_to: None,
        dose_knowledge: Default::default(),
        levels: None,
        target_window: None,
        auc_target: None,
        rules: vec![crate::sim::adaptive::AdaptiveRule {
            op: crate::sim::adaptive::Comparison::Lt,
            threshold: 50.0,
            action: crate::sim::adaptive::AdaptiveAction::Increase(
                crate::sim::adaptive::DoseStep::Percent(25.0),
            ),
            interval: None,
        }],
        intervals: None,
        start_interval: None,
    };
    let spec_opts = AdaptiveSimulateOptions {
        seed: Some(SEED),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(&m, &pop, &m.default_params, 1, &spec, &spec_opts)
        .expect("bare spec run");
    assert_eq!(res.decisions.len(), DECISIONS.len());
    // The same doses through the programmatic entry point, with the spec's monitor name, so
    // the draw is the same, plus an `Ipred` monitor for the latent value.
    let name = crate::sim::adaptive_control::ADAPTIVE_SIGNAL;
    let doses: Vec<(f64, f64)> = res.ledger.iter().map(|e| (e.time, e.amt)).collect();
    let replay = || {
        let doses = doses.clone();
        move |ctx: &ControllerCtx| match doses.iter().find(|d| (d.0 - ctx.t).abs() < 1e-9) {
            Some(&(_, amt)) => vec![DoseAction::Bolus { amt, cmt: 1 }],
            None => vec![DoseAction::Hold],
        }
    };
    let opts = AdaptiveSimulateOptions {
        seed: Some(SEED),
        decision_times: DECISIONS.to_vec(),
        monitors: vec![
            MonitorSpec::new(name, 1, ObserveMode::Dv),
            MonitorSpec::new("L", 1, ObserveMode::Ipred),
        ],
        ..Default::default()
    };
    let prog = simulate_adaptive(&m, &pop, &m.default_params, 1, replay, &opts).expect("replay");
    assert_eq!(
        prog.ledger.len(),
        res.ledger.len(),
        "the replay doses as the spec did"
    );
    let base = subject_assay_base_seed(SEED, "1", 1);
    for (k, (d, p)) in res.decisions.iter().zip(&prog.decisions).enumerate() {
        let latent = p.observed_signals[1].value;
        let v = m.error_spec.variance_at_with_correlations(
            1,
            latent,
            &m.default_params.sigma.values,
            &m.residual_correlations,
        ) * 1.0;
        let want = latent + v.sqrt() * assay_standard_normal(base, k, name);
        let got = d.observed_signals[0].value;
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "spec decision {k}: {got} vs {want}"
        );
        let prog_got = p.observed_signals[0].value;
        assert_eq!(prog_got.to_bits(), want.to_bits(), "replay decision {k}");
    }
}

// ── T2 / T9: the magnitude-free twin ────────────────────────────────────────────────────

/// `fixed_bolus` under `Dv` on `PROP · MAG` (MAG = 10 on every row) and on its bare twin:
/// the ledger, the trajectories' predictions and the latent readings are bit-equal, and
/// only the reading moved — by exactly the magnitude.
///
/// Regressions: the magnitude leaking into the PK path or the ledger (a bits assertion
/// dies); `mult = None` (the ratio is 1, not 10).
#[test]
fn a_magnitude_moves_the_reading_and_nothing_else() {
    let bare = model(
        "",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP)",
        "",
    );
    let mag = model(
        "",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP * MAG)",
        "",
    );
    let pop = read(
        "",
        "1,0,1,0,0,1,0,10\n1,4,1,0,0,1,0,10\n1,8,1,0,0,1,0,10\n1,10,1,0,0,1,0,10\n",
    );
    let opts = two_monitors(&DECISIONS);
    let run = |m: &CompiledModel| {
        simulate_adaptive(m, &pop, &m.default_params, 1, fixed_bolus, &opts).expect("runs")
    };
    let (b, g) = (run(&bare), run(&mag));
    let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let ledger = |r: &AdaptiveSimulationResult| {
        r.ledger
            .iter()
            .map(|e| (e.time.to_bits(), e.amt.to_bits(), e.cmt, e.rate.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(ledger(&b), ledger(&g), "ledger");
    assert_eq!(
        bits(b.trajectories.iter().map(|t| t.ipred).collect()),
        bits(g.trajectories.iter().map(|t| t.ipred).collect()),
        "trajectory predictions"
    );
    let (rb, rg) = (Readings::of(&b), Readings::of(&g));
    assert_eq!(bits(rb.l.clone()), bits(rg.l.clone()), "latent readings");
    // Decision 0 reads an empty system: a proportional variance is the floor on both.
    for k in 1..DECISIONS.len() {
        assert_ne!(
            rb.a[k].to_bits(),
            rg.a[k].to_bits(),
            "decision {k}: A moved"
        );
        let ratio = rg.dev(k) / rb.dev(k);
        assert!(
            relerr(ratio, 10.0) < 1e-12,
            "decision {k}: ratio {ratio}, want 10"
        );
    }
}

/// #2741's table, re-run on this fixture: `power(PROP, P)` at `P = 3` scales the
/// reading's deviation by `|L|²` against the proportional twin (`P = 1` is the
/// proportional loading).
///
/// Regression: the exponent half dropped from the row (the ratio is 1, not `L²`).
#[test]
fn a_power_exponent_moves_the_reading_by_the_prediction_power() {
    let prop = model(
        "",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP)",
        "",
    );
    let pow = model(
        "  theta POW(3.0, 0.1, 5.0)",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ power(PROP, POW)",
        "",
    );
    let pop = read("", OBS_AT_DECISIONS);
    let opts = two_monitors(&DECISIONS);
    let run = |m: &CompiledModel| {
        let r = simulate_adaptive(m, &pop, &m.default_params, 1, fixed_bolus, &opts);
        Readings::of(&r.expect("runs"))
    };
    let (p1, p3) = (run(&prop), run(&pow));
    // Decision 0 reads an empty system (L = 0): nothing to scale. From decision 1 on, the
    // prediction is far from 1, so `|L|²` is a large factor.
    for k in 1..DECISIONS.len() {
        let want = p1.l[k] * p1.l[k];
        assert!(want > 100.0, "decision {k}: the factor {want} is live");
        let ratio = p3.dev(k) / p1.dev(k);
        assert!(
            relerr(ratio, want) < 1e-9,
            "decision {k}: ratio {ratio}, want {want}"
        );
    }
}

// ── T3: parity with `simulate()` ────────────────────────────────────────────────────────

/// One parity cell: a model, its data, the parameter set and the supplied η (for
/// IIV-on-RUV).
struct Cell {
    name: &'static str,
    model: CompiledModel,
    pop: Population,
    params: ModelParameters,
    etas: Option<SuppliedEtas>,
}

fn cell(name: &'static str, theta: &str, omega: &str, sigma: &str, error: &str) -> Cell {
    let model = model(theta, omega, sigma, error, "");
    let params = model.default_params.clone();
    Cell {
        name,
        model,
        pop: read("", OBS_AT_DECISIONS),
        params,
        etas: None,
    }
}

fn parity_cells() -> Vec<Cell> {
    let mut cells = vec![
        cell(
            "M",
            "",
            "",
            "  sigma PROP ~ 0.04",
            "  DV ~ proportional(PROP * MAG)",
        ),
        cell(
            "W",
            "",
            "",
            "  sigma PROP ~ 0.04\n  sigma ADD ~ 1.0",
            "  DV ~ combined(PROP, ADD) weight = MAG",
        ),
        cell(
            "P",
            "  theta POW(1.5, 0.1, 5.0)",
            "",
            "  sigma PROP ~ 0.04",
            "  DV ~ power(PROP, POW)",
        ),
        cell(
            "E",
            "",
            "",
            "  sigma PROP ~ 0.04\n  sigma PROP2 ~ 0.09",
            "  CMT=1: DV ~ proportional(PROP * MAG)\n  CMT=2: DV ~ proportional(PROP2)",
        ),
        cell(
            "TAD",
            "",
            "",
            "  sigma PROP ~ 0.04",
            "  DV ~ proportional(PROP * (1.0 + TAD))",
        ),
    ];

    // M × ruv_scale: a supplied η_RUV makes `ruv_scale = exp(2·0.4)`.
    let mut ruv = cell(
        "M×ruv_scale",
        "",
        "  omega ETA_RUV ~ 0.09",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP * MAG)\n  iiv_on_ruv = ETA_RUV",
    );
    ruv.etas = Some(
        SuppliedEtas::from_table(
            vec!["1".to_string()],
            vec!["ETA_CL".to_string(), "ETA_RUV".to_string()],
            vec![vec![0.0, 0.4]],
        )
        .expect("valid table"),
    );
    cells.push(ruv);

    // M × block_sigma: the parameter set's ρ (−0.6) differs from the declared one (0.25).
    let mut block = cell(
        "M×block_sigma",
        "",
        "",
        "  block_sigma (PROP, ADD) = [0.04, 0.05, 1.0]",
        "  DV ~ combined(PROP * MAG, ADD)",
    );
    assert_eq!(block.model.residual_correlations.len(), 1);
    block.params.residual_correlations = vec![crate::types::ResidualCorrelation {
        rho: -0.6,
        ..block.model.residual_correlations[0].clone()
    }];
    cells.push(block);

    // M × FREM: a `FREMTYPE = 100` covariate row at 1, between decisions.
    let frem = model(
        "  theta TV_WT(70.0) FIX",
        "  omega ETA_WT ~ 100.0",
        "  sigma PROP ~ 0.04\n  sigma EPSCOV ~ 0.0001 FIX",
        "  DV ~ proportional(PROP * MAG)",
        "[fit_options]\n  frem_predictions = TV_WT/ETA_WT:100\n  frem_sigma = EPSCOV",
    );
    assert!(frem.frem_config.is_some(), "the FREM cell is a FREM model");
    let params = frem.default_params.clone();
    cells.push(Cell {
        name: "M×FREM",
        model: frem,
        pop: read(
            ",FREMTYPE",
            "1,0,1,0,0,1,0,2,0\n1,1,70,0,0,1,0,2,100\n1,4,1,0,0,1,0,3,0\n\
             1,8,1,0,0,1,0,5,0\n1,10,1,0,0,1,0,5,0\n",
        ),
        params,
        etas: None,
    });

    // M × MIXNUM: class 1 has probability 1 − 2e-9 at `MIXL = 20`; class 1 triples σ.
    let mix = model(
        "  theta MIXL(20.0, -30.0, 30.0)",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP * MAG * (if (MIXNUM == 1) 3.0 else 1.0))",
        "[mixture]\n  nsub = 2\n  logit(1) = MIXL",
    );
    assert!(mix.mixture.is_some(), "the MIXNUM cell is a mixture model");
    let params = mix.default_params.clone();
    cells.push(Cell {
        name: "M×MIXNUM",
        model: mix,
        pop: read("", OBS_AT_DECISIONS),
        params,
        etas: None,
    });
    cells
}

/// Each form's `Dv` variance at each decision equals `simulate()`'s variance function
/// (`sim_residual_variance`, with `ruv_obs_mult`'s row and `draw_correlations`) on the
/// observation row at that decision, on the replayed subject (base + realized ledger),
/// at the same latent value. The decisions sit on data records, and every magnitude is
/// live and differs from 1 there (`MAG` 2/3/5; `1 + TAD` = 1/5/5; `|f|^0.5`).
///
/// Regressions, each a different cell: `mult = None` (all); the endpoint lookup ignoring
/// `per_endpoint` (E); the exponent half dropped (P); `ruv_scale` dropped
/// (M×ruv_scale); the declared ρ instead of `draw_correlations` (M×block_sigma); the
/// FREM branch reached by the assay (M×FREM, whose row 1 is FREM); `MIXNUM` not held
/// (M×MIXNUM); the assay reading the subject's baseline covariates or the engine clock
/// (M). The message names each side's number.
#[test]
fn every_magnitude_form_draws_simulates_variance() {
    let mut worst = 0.0f64;
    for c in parity_cells() {
        let opts = two_monitors(&DECISIONS);
        let res = match &c.etas {
            Some(e) => {
                let inputs = AdaptiveInputs::new(&c.params).with_etas(e);
                simulate_adaptive_with_inputs(&c.model, &c.pop, &inputs, 1, fixed_bolus, &opts)
            }
            None => simulate_adaptive(&c.model, &c.pop, &c.params, 1, fixed_bolus, &opts),
        }
        .unwrap_or_else(|e| panic!("{}: {e}", c.name));
        assert!(
            c.model.has_custom_ruv_magnitude(),
            "{}: a magnitude form",
            c.name
        );
        let r = Readings::of(&res);

        // The replayed subject: the data subject plus the realized controller doses.
        let mut replay = c.pop.subjects[0].clone();
        replay.doses = res
            .ledger
            .iter()
            .map(|e| crate::types::DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
            .collect();
        let eta: Vec<f64> = match &c.etas {
            Some(_) => vec![0.0, 0.4],
            None => vec![0.0; c.model.n_eta],
        };
        let ruv_scale = c.model.residual_var_scale(&eta);
        let class = res.trajectories[0].mixnum;
        if c.model.mixture.is_some() {
            assert_eq!(class, Some(1), "{}: the drawn class", c.name);
        }
        let _guard = class.map(crate::parser::model_parser::MixtureClassGuard::enter);
        let mult = c.model.ruv_obs_mult(&replay, &c.params.theta);
        let corr = super::simulate::draw_correlations(&c.model, &c.params);

        for (k, &t) in DECISIONS.iter().enumerate() {
            let j = replay
                .obs_times
                .iter()
                .position(|&o| o == t)
                .expect("a record at the decision");
            assert_eq!(replay.fremtype.get(j).copied().unwrap_or(0), 0);
            let sim = c.model.sim_residual_variance(
                &replay,
                j,
                r.l[k],
                &c.params.sigma.values,
                corr.pairs,
                ruv_scale,
                mult.as_ref().map(|m| m[j].as_slice()),
            );
            let assay = r.var(k);
            let e = relerr(assay, sim);
            assert!(e.is_finite(), "{}: {e}", c.name);
            worst = worst.max(e);
            assert!(
                // Measured worst over all cells and decisions: 3.8e-16 (the recovery
                // `((A − L) / ε)²` costs a few ulps). 1e-12 is ~2600× headroom.
                e < 1e-12,
                "{} decision {k} (t = {t}): the assay draws variance {assay}, simulate() \
                 gives {sim} for the row there",
                c.name
            );
        }
    }
    eprintln!("worst assay-vs-simulate relative error: {worst:e}");
}

// ── T4 / T5: `TAD` at a decision ────────────────────────────────────────────────────────

/// `additive(ADD · (1 + TAD))` against `additive(ADD)`: the reading's deviation ratio is
/// `1 + TAD`, so `TAD` at the reading is `ratio − 1`. Additive, so a reading at an empty
/// system (`L = 0`) still carries noise.
fn tad_at_decisions(
    rows: &str,
    decisions: &[f64],
    dose_at: f64,
) -> (Vec<f64>, AdaptiveSimulationResult) {
    let bare = model("", "", "  sigma ADD ~ 1.0", "  DV ~ additive(ADD)", "");
    let tad = model(
        "",
        "",
        "  sigma ADD ~ 1.0",
        "  DV ~ additive(ADD * (1.0 + TAD))",
        "",
    );
    let pop = read("", rows);
    let opts = two_monitors(decisions);
    let make = || {
        move |ctx: &ControllerCtx| {
            if (ctx.t - dose_at).abs() < 1e-9 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        }
    };
    let run = |m: &CompiledModel| {
        simulate_adaptive(m, &pop, &m.default_params, 1, make, &opts).expect("runs")
    };
    let (b, t) = (run(&bare), run(&tad));
    let (rb, rt) = (Readings::of(&b), Readings::of(&t));
    let tads = (0..decisions.len())
        .map(|k| rt.dev(k) / rb.dev(k) - 1.0)
        .collect();
    (tads, t)
}

/// T4, the divergent cell (plan §2, P3a). A: a record at 0, `Hold` at 0, the first dose
/// decided at 4: the reading at 4 sees no dose yet, so `TAD` is the time since the first
/// record, 4 − 0 = 4 (by hand). The frozen replay, which holds the dose at 4, would read 0.
/// B: no record before 4: the reading at 4 reads 0 on either convention. One test holds
/// both, and the straddle (4 vs 0) is asserted.
///
/// Regressions: `TAD` taken from the replay convention, or forced to 0 (A reads 0).
#[test]
fn tad_before_the_first_dose_is_the_time_since_the_first_record() {
    let (a, ra) = tad_at_decisions(
        "1,0,1,0,0,1,0,1\n1,4,1,0,0,1,0,1\n1,8,1,0,0,1,0,1\n",
        &[0.0, 4.0],
        4.0,
    );
    let (b, _) = tad_at_decisions("1,4,1,0,0,1,0,1\n1,8,1,0,0,1,0,1\n", &[0.0, 4.0], 4.0);
    // The premise: the first dose is the controller's, at 4.
    assert_eq!(ra.ledger.len(), 1);
    assert_eq!(ra.ledger[0].time, 4.0);
    let want_a = [0.0, 4.0];
    let want_b = [0.0, 0.0];
    for k in 0..2 {
        assert!(
            (a[k] - want_a[k]).abs() < 1e-9,
            "A decision {k}: TAD {}",
            a[k]
        );
        assert!(
            (b[k] - want_b[k]).abs() < 1e-9,
            "B decision {k}: TAD {}",
            b[k]
        );
    }
    assert!(
        (a[1] - b[1]).abs() > 1.0,
        "the straddle: A and B differ at 4"
    );
}

/// T5: one controller dose, decided at 4 (not at 0), read at 8 → `TAD = 4` on the causal
/// record. The data subject alone (no dose) reads 8 − 0 = 8 there, and that is asserted,
/// so the straddle cannot silently close.
///
/// Regression: `TAD` read from `subject` instead of `shadow` (reads 8).
#[test]
fn tad_after_a_controller_dose_reads_that_dose() {
    let rows = "1,0,1,0,0,1,0,1\n1,4,1,0,0,1,0,1\n1,8,1,0,0,1,0,1\n";
    let (tad, res) = tad_at_decisions(rows, &[0.0, 4.0, 8.0], 4.0);
    assert_eq!(res.ledger.len(), 1, "one controller dose");
    assert!((tad[2] - 4.0).abs() < 1e-9, "TAD at 8: {}", tad[2]);
    let data_only = read("", rows).subjects[0].time_after_dose_at(8.0);
    assert_eq!(data_only, 8.0, "the straddle: without the dose, 8");
}

// ── T7: a time-varying covariate ────────────────────────────────────────────────────────

/// `PROP · MAG` with `MAG` 1 → 10 at 4, against the bare twin: the deviation ratio reads
/// 10, 10 at decisions 4 and 8, the row in force at each (decision 0 reads an empty
/// system, nothing to scale).
///
/// Regression: the assay reading the subject's baseline covariates (1 at every decision).
#[test]
fn a_time_varying_magnitude_reads_the_row_in_force() {
    let bare = model(
        "",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP)",
        "",
    );
    let mag = model(
        "",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP * MAG)",
        "",
    );
    let pop = read("", "1,0,1,0,0,1,0,1\n1,4,1,0,0,1,0,10\n1,8,1,0,0,1,0,10\n");
    let opts = two_monitors(&DECISIONS);
    let run = |m: &CompiledModel| {
        let r = simulate_adaptive(m, &pop, &m.default_params, 1, fixed_bolus, &opts);
        Readings::of(&r.expect("runs"))
    };
    let (rb, rm) = (run(&bare), run(&mag));
    // Decision 0 reads an empty system under a proportional error: nothing to compare.
    for (k, want) in [(1, 10.0), (2, 10.0)] {
        let ratio = rm.dev(k) / rb.dev(k);
        assert!(
            relerr(ratio, want) < 1e-12,
            "decision {k}: ratio {ratio}, want {want}"
        );
    }
    assert_eq!(
        read("", "1,0,1,0,0,1,0,1\n1,4,1,0,0,1,0,10\n").subjects[0].covariates["MAG"],
        1.0,
        "the straddle: the baseline row reads 1"
    );
}

// ── T8: the written `TIME` ──────────────────────────────────────────────────────────────

/// A dose at 0 and records at 6 and 24, then `EVID=3` at a written 0 that restarts `TIME`,
/// a dose there, and a record at a written 6. The reader moves the reset to engine 25, so
/// the second period runs 25 ahead of its written `TIME` (as `adaptive_written_time_tests`).
const RESTART_ROWS: &str = "1,0,0,1,100,1,1,1\n1,6,1,0,0,1,0,1\n1,24,1,0,0,1,0,1\n\
                            1,0,0,3,0,1,1,1\n1,0,0,1,100,1,1,1\n1,6,1,0,0,1,0,1\n";

/// The readout is `TIME` itself, so the `Ipred` monitor reads the clock the monitor sees,
/// and `additive(ADD · (1 + TIME))` against `additive(ADD)` gives the clock the σ sees:
/// `ratio − 1`. At decisions 6 (first period) and 31 (written 6) both must read 6 — the
/// same number for the monitor and the σ (#2756) — while the engine clock reads 31.
///
/// Regressions: `AssayRow::time` from engine `t_start` (the σ reads 31 at the second
/// decision); the monitor fed a different clock than the σ (the same-clock assertion).
#[test]
fn a_time_magnitude_reads_the_written_time_the_monitor_reads() {
    let src = |error: &str| {
        model_src("", "", "  sigma ADD ~ 1.0", error, "")
            .replace("  y = central", "  y = central * 0.0 + TIME")
    };
    let bare = parse_model_string(&src("  DV ~ additive(ADD)")).expect("parse");
    let time = parse_model_string(&src("  DV ~ additive(ADD * (1.0 + TIME))")).expect("parse");
    let pop = read("", RESTART_ROWS);
    let s = &pop.subjects[0];
    assert_eq!(s.obs_times, [6.0, 24.0, 31.0], "engine times");
    assert_eq!(s.obs_raw_times, [6.0, 24.0, 6.0], "written times");
    let decisions = [6.0, 31.0];
    let opts = two_monitors(&decisions);
    let hold = || |_c: &ControllerCtx| vec![DoseAction::Hold];
    let run = |m: &CompiledModel| {
        let r = simulate_adaptive(m, &pop, &m.default_params, 1, hold, &opts);
        Readings::of(&r.expect("runs"))
    };
    let (rb, rt) = (run(&bare), run(&time));
    for (k, &t) in decisions.iter().enumerate() {
        let sigma_time = rt.dev(k) / rb.dev(k) - 1.0;
        let monitor_time = rt.l[k];
        assert!(
            (sigma_time - 6.0).abs() < 1e-9,
            "decision {k} (engine {t}): the σ read TIME {sigma_time}, written 6"
        );
        assert!(
            (sigma_time - monitor_time).abs() < 1e-9,
            "decision {k}: the σ read TIME {sigma_time}, the monitor {monitor_time}"
        );
    }
    assert_eq!(
        decisions[1] - 6.0,
        25.0,
        "the straddle: engine and written differ"
    );
}

// ── T10: a non-positive magnitude at a decision ─────────────────────────────────────────

/// The decision at 4 sits on an `EVID=2` row whose `MAG` makes the magnitude non-positive;
/// every observation row is positive, so `check_residual_magnitude` passes and the assay's
/// own check is the one that fires. Three fixtures: a single-endpoint σ (`MAG = 0`, names no
/// `CMT=`), a `CMT=1:` line (`MAG = −2`, names `CMT=1` and not the other endpoint), and a
/// `power(PROP, MAG)` exponent (`MAG = 0`: the upper half of the row, named a power
/// exponent of `PROP`, not a magnitude of a slot past the σs). Each message is asserted
/// whole.
///
/// Regressions: the guard deleted (the reading would be `L`, silently); the guard scanning
/// only the multiplier half (an exponent of 0 drawn silently); the exponent half reported
/// as a magnitude; any sentence of the message deleted or changed; the endpoint named on a
/// single-endpoint model, or the wrong one on a `CMT=N:` block.
#[test]
fn a_non_positive_magnitude_at_a_decision_is_refused() {
    let rows = |mag: &str| {
        format!("1,0,1,0,0,1,0,1\n1,4,0,2,0,1,1,{mag}\n1,8,1,0,0,1,0,1\n1,10,1,0,0,1,0,1\n")
    };
    let single = model(
        "",
        "",
        "  sigma PROP ~ 0.04",
        "  DV ~ proportional(PROP * MAG)",
        "",
    );
    let cmt = model(
        "",
        "",
        "  sigma PROP ~ 0.04\n  sigma PROP2 ~ 0.09",
        "  CMT=1: DV ~ proportional(PROP2)\n  CMT=2: DV ~ proportional(PROP * MAG)",
        "",
    );
    let exponent = model("", "", "  sigma PROP ~ 0.04", "  DV ~ power(PROP, MAG)", "");
    // `CMT=1:` reads PROP2; the magnitude sits on the endpoint the monitor does not read
    // in this fixture, so it runs. The refused one reads it on its own line.
    let cmt_refused = model(
        "",
        "",
        "  sigma PROP ~ 0.04\n  sigma PROP2 ~ 0.09",
        "  CMT=1: DV ~ proportional(PROP * MAG)\n  CMT=2: DV ~ proportional(PROP2)",
        "",
    );
    let opts = two_monitors(&DECISIONS);
    let run = |m: &CompiledModel, mag: &str| {
        let pop = read("", &rows(mag));
        assert!(
            super::validation::check_residual_magnitude(m, &pop).is_empty(),
            "the data rows pass the magnitude check"
        );
        simulate_adaptive(m, &pop, &m.default_params, 1, fixed_bolus, &opts)
    };
    run(&cmt, "-2").expect("a magnitude on the endpoint the monitor does not read");

    let tail = "as written — it must be strictly positive and finite. The assay draws the \
                reading's residual error from it, and the check on the data's observation rows \
                does not see a decision. Check θ and the covariates its expression reads in the \
                row in force at this decision.";
    for (name, m, mag, want) in [
        (
            "single",
            &single,
            "0",
            format!(
                "subject '1' (sim 1): decision 1 at t=4: monitor 'A': [error_model] the \
                 residual-error magnitude of sigma `PROP` evaluates to 0 at this `Dv` reading, \
                 TIME 4 {tail}"
            ),
        ),
        (
            "CMT line",
            &cmt_refused,
            "-2",
            format!(
                "subject '1' (sim 1): decision 1 at t=4: monitor 'A': [error_model] the \
                 residual-error magnitude of sigma `PROP` on the `CMT=1:` line evaluates to -2 \
                 at this `Dv` reading, TIME 4 {tail}"
            ),
        ),
        (
            "exponent",
            &exponent,
            "0",
            format!(
                "subject '1' (sim 1): decision 1 at t=4: monitor 'A': [error_model] the \
                 residual-error power exponent of sigma `PROP` evaluates to 0 at this `Dv` \
                 reading, TIME 4 {tail}"
            ),
        ),
    ] {
        let err = run(m, mag).expect_err(name);
        assert_eq!(
            err.code(),
            Some("E_RUV_MAGNITUDE_NONPOSITIVE"),
            "{name}: {err}"
        );
        assert_eq!(err.message(), want, "{name}");
        let msg = err.message();
        if name != "CMT line" {
            assert!(!msg.contains("CMT="), "{name}: names no endpoint");
        } else {
            assert!(msg.contains("`CMT=1:`") && !msg.contains("CMT=2"), "{name}");
        }
        assert!(
            !msg.contains("missing"),
            "{name}: a decision row is not a data cell"
        );
    }
}

/// T1, the variance half. The readings above cannot see a bare model handed a row of
/// ones: the `_scaled` form differs from the legacy one in the last bit of the variance on
/// about half of all predictions, but `L + √v·ε` rounds that ulp away (measured: the
/// mutation left every reading of `bare_model_readings_keep_their_bits` unchanged). So the
/// assay's variance itself is pinned against the legacy form, bit for bit, on a grid of
/// 20 000 predictions — and the straddle is asserted: on the same grid the `_scaled` form
/// at a row of ones differs on a large share (measured 54% on 200 000).
///
/// Regression: `assay_resid_var` passing `Some(row of ones)` for a bare model.
#[test]
fn bare_model_assay_variance_is_the_legacy_form_bit_for_bit() {
    let m = parse_model_string(ODE_NO_IIV).expect("parse");
    let params = &m.default_params;
    let (s, c) = (&params.sigma.values, &m.residual_correlations);
    let cov = std::collections::HashMap::new();
    let (mut n, mut scaled_differs) = (0, 0);
    for i in 0..20_000 {
        let f = 0.001 + (i as f64) * 0.007_31;
        let row = crate::sim::adaptive::AssayRow {
            cmt: 1,
            ipred: f,
            time: 0.0,
            tad: 0.0,
            cov: &cov,
        };
        let got = super::adaptive::assay_resid_var(&m, params, &row, 1.0)
            .expect("bare: nothing to refuse")
            .expect("cmt 1 has an error model");
        let legacy = m.error_spec.variance_at_with_correlations(1, f, s, c) * 1.0;
        assert_eq!(
            got.to_bits(),
            legacy.to_bits(),
            "f = {f}: {got} vs {legacy}"
        );
        let scaled = m.error_spec.variance_at_scaled(1, f, s, c, &[1.0]);
        scaled_differs += usize::from(scaled.to_bits() != legacy.to_bits());
        n += 1;
    }
    assert!(
        scaled_differs * 4 > n,
        "the straddle: the `_scaled` form differs on {scaled_differs} of {n}"
    );
}
