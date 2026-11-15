//! #946 PR1: parameter-uncertainty draws on the adaptive path
//! (`AdaptiveInputs::with_uncertainty`).
//!
//! The adaptive exception (AGENTS.md, epic #391): there is no NONMEM run to anchor
//! against. The oracles are the degenerate oracle (a draw equal to the point estimate,
//! through the test seam, reproduces today's run bit for bit) and a cross-check against
//! `simulate_with_uncertainty` on the same injected draws. The structural tests pin each
//! keying fix by the smallest edit that removes it (the audit's C-table on #946).

use super::adaptive_sim_tests::{fixed_bolus, population, subj};
use super::*;
use crate::diagnostics::EngineError;
use crate::estimation::uncertainty_samples::UncertaintyMethod;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DoseAction, DoseLedgerEntry,
    MonitorSpec, ObserveMode,
};

/// A 1-cpt IV ODE whose `[parameters]` lines are spliced in: `omega` (and an optional
/// `kappa`) line, and the CL expression.
fn model(omega_kappa: &str, cl: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
{omega_kappa}
  sigma PROP ~ 0.04
[individual_parameters]
  CL = {cl}
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#
    ))
    .expect("parse")
}

/// Between-subject variability on CL, no κ.
fn iiv() -> CompiledModel {
    model("  omega ETA_CL ~ 0.09", "TVCL * exp(ETA_CL)")
}

/// η and κ on CL.
fn iiv_iov() -> CompiledModel {
    model(
        "  omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.09",
        "TVCL * exp(ETA_CL + KAPPA_CL)",
    )
}

/// Ω FIX 0 (η exactly 0, #1605), with or without a κ on CL.
fn zero_omega(kappa: bool) -> CompiledModel {
    if kappa {
        model(
            "  omega ETA_CL ~ 0.0 FIX\n  kappa KAPPA_CL ~ 0.09",
            "TVCL * exp(ETA_CL + KAPPA_CL)",
        )
    } else {
        model("  omega ETA_CL ~ 0.0 FIX", "TVCL * exp(ETA_CL)")
    }
}

const DECISIONS: [f64; 3] = [0.0, 24.0, 48.0];
const OBS: [f64; 4] = [6.0, 24.0, 30.0, 54.0];
const SEED: u64 = 946;

fn pop(n: usize) -> Population {
    population(
        (1..=n)
            .map(|i| subj(&i.to_string(), OBS.to_vec(), vec![]))
            .collect(),
    )
}

fn opts(mode: Option<ObserveMode>) -> AdaptiveSimulateOptions {
    AdaptiveSimulateOptions {
        seed: Some(SEED),
        decision_times: DECISIONS.to_vec(),
        monitors: mode
            .map(|m| vec![MonitorSpec::new("A", 1, m)])
            .unwrap_or_default(),
        ..Default::default()
    }
}

/// Doses 100 when the monitor reads below 50, else holds: decisions follow the signal,
/// so a different κ or assay draw shows in the decision log.
fn threshold() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("monitor A") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    }
}

/// Everything a run returns for one draw, with the `draw` tag cleared so two draws (or a
/// draw and a run without uncertainty) compare directly. Floats by their bits.
#[derive(Debug, PartialEq)]
struct DrawRows {
    traj: Vec<(usize, String, u64, usize, u64, Option<usize>)>,
    ledger: Vec<DoseLedgerEntry>,
    decisions: Vec<DecisionLogEntry>,
    metrics: Vec<AdaptiveSubjectMetrics>,
}

fn draw_rows(r: &AdaptiveSimulationResult, draw: usize) -> DrawRows {
    DrawRows {
        traj: r
            .trajectories
            .iter()
            .filter(|t| t.draw == draw)
            .map(|t| {
                assert!(t.ipred.is_finite(), "non-finite IPRED {t:?}");
                (
                    t.sim,
                    t.id.clone(),
                    t.time.to_bits(),
                    t.cmt,
                    t.ipred.to_bits(),
                    t.mixnum,
                )
            })
            .collect(),
        ledger: r
            .ledger
            .iter()
            .filter(|e| e.draw == draw)
            .cloned()
            .map(|mut e| {
                e.draw = 0;
                e
            })
            .collect(),
        decisions: r
            .decisions
            .iter()
            .filter(|d| d.draw == draw)
            .cloned()
            .map(|mut d| {
                d.draw = 0;
                d
            })
            .collect(),
        metrics: r
            .metrics
            .iter()
            .filter(|m| m.draw == draw)
            .cloned()
            .map(|mut m| {
                m.draw = 0;
                m
            })
            .collect(),
    }
}

fn refusal(r: Result<AdaptiveSimulationResult, EngineError>) -> (String, String) {
    let e = r.expect_err("refused");
    let code = e.diagnostic().map(|d| d.code.clone()).unwrap_or_default();
    (code, e.to_string())
}

fn draws_of(r: &AdaptiveSimulationResult) -> std::collections::BTreeSet<usize> {
    r.trajectories
        .iter()
        .map(|t| t.draw)
        .chain(r.ledger.iter().map(|e| e.draw))
        .chain(r.decisions.iter().map(|d| d.draw))
        .chain(r.metrics.iter().map(|m| m.draw))
        .collect()
}

/// A fit at the model's initial parameters with a small packed covariance: the real
/// asymptotic sampler, without `fit()`.
fn fit_of(model: &CompiledModel) -> FitResult {
    super::simulate_with_uncertainty_tests::synthetic_fit(&model.default_params)
}

// ── replicate tags in messages ─────────────────────────────────────────────────────────

/// The draw is named only under uncertainty (#946, audit A9): `sim_label` and
/// `replicate_label` keep their old text without it.
///
/// Mutations: always printing the draw (the off side reddens); dropping it (the on side).
#[test]
fn replicate_tag_names_the_draw_only_under_uncertainty() {
    use crate::sim::adaptive::ReplicateTag;
    let off = ReplicateTag { draw: None, sim: 3 };
    let on = ReplicateTag {
        draw: Some(2),
        sim: 3,
    };
    assert_eq!(off.sim_label(), "sim 3");
    assert_eq!(on.sim_label(), "draw 2, sim 3");
    assert_eq!(off.replicate_label(), "replicate 3");
    assert_eq!(on.replicate_label(), "draw 2, replicate 3");
}

/// `W_ADAPTIVE_STEP_ABSORBED` under uncertainty counts over `n_draws × n_sim` replicates
/// and names the first by its draw and sim (§5 row 13). The run-level count is the
/// caller's; the first replicate is the earliest `(draw, sim)`, not the earliest sim.
///
/// Mutations: naming the first replicate by its sim alone; sorting by sim before draw.
#[test]
fn absorbed_step_warning_names_the_draw_of_the_first_replicate() {
    use super::adaptive::{absorbed_step_warning, AbsorbedPlan};
    use crate::sim::adaptive::{AbsorbedIncrease::FromStart, ReplicateTag};
    let p = crate::parser::model_parser::parse_full_model(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [0, 24]
  start_dose = 500
  route = bolus(cmt=1)
  dose_bounds = [250, 4000]
  round_to = 250
  when signal < 0 : increase 10%
"#,
    )
    .expect("parse");
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let population = pop(1);
    let plan = |draw, sim| AbsorbedPlan {
        subject: 0,
        replicate: Some(ReplicateTag {
            draw: Some(draw),
            sim,
        }),
        found: FromStart(500.0),
    };
    let w =
        absorbed_step_warning(spec, &population, 6, vec![plan(2, 1), plan(1, 3)]).expect("a line");
    assert!(
        w.contains(
            "From the start dose on: '1' in 2 of 6 replicate(s), first in draw 1 sim 3 at 500."
        ),
        "{w}"
    );
}

// ── the degenerate oracle and the keying ───────────────────────────────────────────────

/// The degenerate oracle (plan §Validation; audit A8): handed draws at the point
/// estimate through the seam, draw 1's trajectories, ledger, decisions and metrics are
/// bit-identical to the run without uncertainty. η, κ and a `Dv` assay are all live, so
/// the κ and assay keys at `(draw, rep) = (1, 1)` and the loop order are all on the
/// line. Two draws, so a draw loop placed inside the sim loop would interleave the η
/// stream and move draw 1's second replicate.
///
/// Mutations: the draw loop reordered (draw 1's η move); θ drawn from the η stream;
/// the verifier's κ key not the build loop's. Measured: `replicate_key` folding at
/// `(1, 1)` too does **not** redden this test — both sides of the oracle run the same
/// helper, so the base run moves with the seam run. That edit dies in
/// `replicate_key_is_sim_at_draw_one_rep_one_and_distinct_elsewhere` and in every test
/// that rebuilds κ or the assay from `subject_*_base_seed(seed, id, sim)` (35 measured).
#[test]
fn draws_at_the_point_estimate_reproduce_the_run_without_uncertainty() {
    let model = iiv_iov();
    let p = &model.default_params;
    let population = pop(3);
    let o = opts(Some(ObserveMode::Dv));
    let base = simulate_adaptive(&model, &population, p, 2, threshold, &o).expect("base");
    let draws = [p.clone(), p.clone()];
    let seam = simulate_adaptive_with_inputs(
        &model,
        &population,
        &AdaptiveInputs::new(p).with_injected_draws(&draws),
        2,
        threshold,
        &o,
    )
    .expect("seam");
    assert_eq!(draws_of(&base), [1].into());
    assert_eq!(draws_of(&seam), [1, 2].into());
    assert!(base.parameters.is_empty(), "no uncertainty, no draws");
    assert_eq!(seam.parameters.len(), 2);
    assert_eq!(
        seam.parameters.iter().map(|d| d.draw).collect::<Vec<_>>(),
        [1, 2]
    );
    let (want, got) = (draw_rows(&base, 1), draw_rows(&seam, 1));
    assert_eq!(got.traj.len(), 3 * 2 * OBS.len(), "every row present");
    assert!(!got.ledger.is_empty(), "the controller dosed");
    assert_eq!(got, want);
}

/// C4: under L1 the η stream runs on into the next draw, so two draws at the same
/// parameters simulate different patients. No κ and no assay, so η is the only thing
/// that can tell the draws apart.
///
/// Mutation: re-seeding the η stream per draw (draw 2 = draw 1).
#[test]
fn eta_stream_continues_across_draws() {
    let model = iiv();
    let p = &model.default_params;
    let draws = [p.clone(), p.clone()];
    let r = simulate_adaptive_with_inputs(
        &model,
        &pop(2),
        &AdaptiveInputs::new(p).with_injected_draws(&draws),
        1,
        fixed_bolus,
        &opts(None),
    )
    .expect("runs");
    let (d1, d2) = (draw_rows(&r, 1), draw_rows(&r, 2));
    assert_eq!(d1.traj.len(), d2.traj.len());
    assert_eq!(d1.ledger, d2.ledger, "a fixed regimen: same doses");
    assert_ne!(d1.traj, d2.traj, "new patients in draw 2");
}

/// C2: the κ and assay keys fold the draw. Ω FIX 0, so η is exactly 0 in every draw,
/// and two draws at the same parameters can differ only through a κ or an assay key.
/// Three cells: neither (the straddle: the draws are bit-equal), κ only (trajectories
/// differ), a `Dv` assay only (the readings the decisions record differ).
///
/// Mutations: dropping `draw` from `replicate_key` (both live cells go equal); keying κ
/// alone, or the assay alone, by the bare `sim` (that cell goes equal).
#[test]
fn kappa_and_assay_keys_fold_the_draw() {
    let run = |model: &CompiledModel, mode| {
        let p = &model.default_params;
        let draws = [p.clone(), p.clone()];
        let r = simulate_adaptive_with_inputs(
            model,
            &pop(2),
            &AdaptiveInputs::new(p).with_injected_draws(&draws),
            1,
            threshold,
            &opts(Some(mode)),
        )
        .expect("runs");
        (draw_rows(&r, 1), draw_rows(&r, 2))
    };
    let (a, b) = run(&zero_omega(false), ObserveMode::Ipred);
    assert_eq!(
        a, b,
        "the straddle: with no κ and no assay, the draws are equal"
    );
    let (a, b) = run(&zero_omega(true), ObserveMode::Ipred);
    assert_ne!(a.traj, b.traj, "κ: draw 2 has its own occasion draws");
    let (a, b) = run(&zero_omega(false), ObserveMode::Dv);
    assert_eq!(a.traj.len(), b.traj.len());
    assert_ne!(
        a.decisions, b.decisions,
        "assay: draw 2 reads its own assay noise"
    );
}

/// C3: the mixture class is a between-subject draw keyed by `(draw, sim)`, so two draws at
/// the same parameters put some subject in different classes. Ten subjects at 50:50, so
/// the chance that none differs is 2⁻¹⁰ — and a seeded run is deterministic.
///
/// Mutation: keeping the hard-coded draw `1` in `classes.enter` (every subject keeps its
/// class).
#[test]
fn mixture_class_is_redrawn_per_draw() {
    let model = parse_model_string(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta MIXL(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse");
    let p = &model.default_params;
    let draws = [p.clone(), p.clone()];
    let r = simulate_adaptive_with_inputs(
        &model,
        &pop(10),
        &AdaptiveInputs::new(p).with_injected_draws(&draws),
        1,
        fixed_bolus,
        &opts(None),
    )
    .expect("runs");
    let class = |draw: usize| -> Vec<(String, Option<usize>)> {
        let mut v: Vec<_> = r
            .trajectories
            .iter()
            .filter(|t| t.draw == draw)
            .map(|t| (t.id.clone(), t.mixnum))
            .collect();
        v.dedup();
        v
    };
    let (c1, c2) = (class(1), class(2));
    assert_eq!(c1.len(), 10, "one class per subject: {c1:?}");
    assert!(c1.iter().all(|(_, m)| m.is_some()));
    assert_ne!(c1, c2, "some subject changes class between the draws");
}

/// C1: the parameter draws come from a stream of their own (§2), so asking for more
/// draws leaves draw 1 — its parameters and every row — bit-identical. On the real
/// asymptotic sampler: the seam bypasses it and cannot see this (audit A8). The draws are
/// also exactly `draw_parameter_samples_coded` on the parameter stream's seed.
///
/// Mutations: drawing θ from the η stream (draw 1's rows move with `n_draws`); any other
/// seed for the parameter stream (the parameters differ from the reference).
#[test]
fn more_draws_leave_draw_one_unchanged_on_the_real_sampler() {
    use rand::SeedableRng;
    let model = iiv();
    let fit = fit_of(&model);
    let params =
        crate::estimation::uncertainty_samples::fitted_params_from_result(&fit, &model).unwrap();
    let run = |n_draws| {
        simulate_adaptive_with_inputs(
            &model,
            &pop(2),
            &AdaptiveInputs::new(&params).with_uncertainty(
                &fit,
                UncertaintySpec {
                    method: UncertaintyMethod::Asymptotic,
                    n_draws,
                },
            ),
            2,
            fixed_bolus,
            &opts(None),
        )
        .expect("runs")
    };
    let (one, three) = (run(1), run(3));
    assert_eq!(draws_of(&one), [1].into());
    assert_eq!(draws_of(&three), [1, 2, 3].into());
    assert_eq!(draw_rows(&one, 1), draw_rows(&three, 1));
    assert_ne!(
        draw_rows(&three, 1).traj,
        draw_rows(&three, 2).traj,
        "the draws differ"
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(crate::sim::adaptive::combine_seed(
        SEED ^ super::adaptive_inputs::PARAM_PURPOSE_SALT,
        0,
    ));
    let want = crate::estimation::uncertainty_samples::draw_parameter_samples_coded(
        &fit,
        &params,
        3,
        UncertaintyMethod::Asymptotic,
        &mut rng,
    )
    .unwrap();
    for (got, want) in three.parameters.iter().zip(&want) {
        let bits = |p: &ModelParameters| {
            p.theta
                .iter()
                .chain(p.omega.matrix.iter())
                .chain(&p.sigma.values)
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&got.parameters), bits(want), "draw {}", got.draw);
    }
    assert_eq!(three.parameters.len(), 3);
    assert_ne!(
        three.parameters[0].parameters.theta, params.theta,
        "a real draw, not the point estimate"
    );
}

/// The cross-check against `simulate_with_uncertainty` (plan §Validation): the same
/// injected draws, Ω FIX 0 (so IPRED depends on θ_d alone) and a fixed regimen. Every
/// `(draw, id, time)` IPRED agrees **to the bit**, including the read on a dosing decision
/// (t = 24). Measured: worst relative gap 0 (macOS arm64; Linux CI runs it per PR). The
/// plan proposed #1151's 1e-9 band, measured on a `TAD`-reading RHS; this RHS reads no
/// time, so the two engines take identical steps and the band would be 10⁹× looser than
/// what is realised. Draw 2's CL is 40% higher, so a draw tag or parameter mix-up moves
/// every later row.
///
/// Mutations: simulating every draw at the first draw's parameters; tagging rows with the
/// wrong draw.
#[test]
fn cross_check_against_simulate_with_uncertainty_on_the_same_draws() {
    use rand::SeedableRng;
    let model = zero_omega(false);
    let p0 = model.default_params.clone();
    let mut p1 = p0.clone();
    p1.theta[0] = 7.0;
    let draws = [p0.clone(), p1];
    let adaptive = simulate_adaptive_with_inputs(
        &model,
        &pop(2),
        &AdaptiveInputs::new(&p0).with_injected_draws(&draws),
        1,
        fixed_bolus,
        &opts(None),
    )
    .expect("adaptive");
    let doses: Vec<DoseEvent> = DECISIONS
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_pop = population(
        ["1", "2"]
            .iter()
            .map(|id| subj(id, OBS.to_vec(), doses.clone()))
            .collect(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let (mut rows, mut warnings) = (Vec::new(), Vec::new());
    super::simulate::simulate_uncertainty_draws(
        &model,
        &static_pop,
        &draws,
        &SimulateUncertaintyOptions {
            n_uncertainty_draws: 2,
            n_sim_per_draw: 1,
            ..Default::default()
        },
        SEED,
        &mut rng,
        &mut rows,
        &mut warnings,
        None,
    )
    .expect("static");
    assert_eq!(rows.len(), adaptive.trajectories.len());
    for s in &rows {
        let a = adaptive
            .trajectories
            .iter()
            .find(|a| a.draw == s.draw && a.id == s.id && a.time == s.time)
            .unwrap_or_else(|| panic!("no adaptive row for {s:?}"));
        assert!(a.ipred.is_finite(), "{a:?}");
        assert_eq!(
            a.ipred.to_bits(),
            s.ipred.to_bits(),
            "adaptive {a:?} vs static {s:?}"
        );
    }
    // Draw 2 really is at its own θ: its IPRED is not draw 1's.
    let at = |draw: usize| {
        adaptive
            .trajectories
            .iter()
            .find(|t| t.draw == draw && t.id == "1" && t.time == 54.0)
            .unwrap()
            .ipred
    };
    assert!(at(2) < 0.9 * at(1), "draw 1 {} vs draw 2 {}", at(1), at(2));
}

// ── refusals ───────────────────────────────────────────────────────────────────────────

fn with_spec(
    model: &CompiledModel,
    params: &ModelParameters,
    fit: &FitResult,
    method: UncertaintyMethod,
    n_draws: usize,
) -> Result<AdaptiveSimulationResult, EngineError> {
    simulate_adaptive_with_inputs(
        model,
        &pop(1),
        &AdaptiveInputs::new(params).with_uncertainty(fit, UncertaintySpec { method, n_draws }),
        1,
        fixed_bolus,
        &opts(None),
    )
}

/// §5 row 7: `n_draws = 0` is refused, whole text; `n_draws = 1` runs.
///
/// Mutations: the check deleted (the run returns empty rows); any sentence changed.
#[test]
fn zero_draws_are_refused() {
    let model = iiv();
    let fit = fit_of(&model);
    let p =
        crate::estimation::uncertainty_samples::fitted_params_from_result(&fit, &model).unwrap();
    assert_eq!(
        refusal(with_spec(
            &model,
            &p,
            &fit,
            UncertaintyMethod::Asymptotic,
            0
        )),
        (
            "E_ADAPTIVE_OPTIONS".to_string(),
            "`UncertaintySpec::n_draws` must be at least 1; omit `with_uncertainty` to \
             simulate at the point estimate"
                .to_string()
        )
    );
    assert!(with_spec(&model, &p, &fit, UncertaintyMethod::Asymptotic, 1).is_ok());
}

/// §5 row 8 (audit A2): `params` must carry the fit's values, compared value by value.
/// ferx-r's construction — `OmegaMatrix::from_matrix` (no free mask) and the template's
/// FIX flags — passes; θ one ulp off is refused, naming the θ, and so is Ω, σ.
///
/// Mutations: a whole-struct comparison (the R-shaped params are refused); the θ, Ω or σ
/// leg dropped (that cell runs); the message naming another parameter.
#[test]
fn params_must_carry_the_fits_values() {
    let model = iiv();
    let fit = fit_of(&model);
    let t = &model.default_params;
    let mut r_shaped = t.clone();
    r_shaped.omega =
        OmegaMatrix::from_matrix(fit.omega.clone(), fit.eta_names.clone(), t.omega.diagonal);
    assert!(
        with_spec(&model, &r_shaped, &fit, UncertaintyMethod::Asymptotic, 1).is_ok(),
        "the R-shaped construction passes"
    );
    let mut theta = r_shaped.clone();
    theta.theta[0] = f64::from_bits(theta.theta[0].to_bits() + 1);
    let (code, text) = refusal(with_spec(
        &model,
        &theta,
        &fit,
        UncertaintyMethod::Asymptotic,
        1,
    ));
    assert_eq!(code, "E_ADAPTIVE_OPTIONS");
    assert_eq!(
        text,
        format!(
            "`with_uncertainty` draws around the fit's point estimate, but the run's \
             parameters differ from it: θ TVCL is {}, the fit's is 5. Pass the fit's \
             parameters (for example `fitted_params_from_result(&fit, &model)`) as the \
             run's `params`",
            theta.theta[0]
        )
    );
    let mut omega = r_shaped.clone();
    omega.omega = OmegaMatrix::from_matrix(
        fit.omega.clone() * 2.0,
        fit.eta_names.clone(),
        t.omega.diagonal,
    );
    let (_, text) = refusal(with_spec(
        &model,
        &omega,
        &fit,
        UncertaintyMethod::Asymptotic,
        1,
    ));
    assert!(
        text.contains("Ω[1, 1] is 0.18, the fit's is 0.09"),
        "{text}"
    );
    let mut sigma = r_shaped;
    sigma.sigma.values[0] = 0.25;
    let (_, text) = refusal(with_spec(
        &model,
        &sigma,
        &fit,
        UncertaintyMethod::Asymptotic,
        1,
    ));
    assert!(text.contains("σ[0] is 0.25, the fit's is 0.2"), "{text}");
}

/// `first_value_difference` covers Ω_IOV and the residual ρ too, and ignores what it must:
/// FIX flags and bounds.
///
/// Mutations: the Ω_IOV or ρ leg dropped; a flag or a bound compared.
#[test]
fn value_difference_covers_iov_and_rho_and_ignores_flags() {
    use super::adaptive_inputs::first_value_difference;
    let model = iiv_iov();
    let t = model.default_params.clone();
    let mut flags = t.clone();
    flags.theta_fixed[0] = !flags.theta_fixed[0];
    flags.theta_upper[0] = 99.0;
    assert_eq!(first_value_difference(&flags, &t), None);
    let mut iov = t.clone();
    let m = iov.omega_iov.as_ref().unwrap().matrix.clone() * 3.0;
    iov.omega_iov = Some(OmegaMatrix::from_matrix(m, vec!["KAPPA_CL".into()], true));
    let d = first_value_difference(&iov, &t).expect("Ω_IOV differs");
    assert!(d.starts_with("Ω_IOV[1, 1] is"), "{d}");
    let mut rho = t.clone();
    rho.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 0,
        rho: 0.5,
    }];
    let d = first_value_difference(&rho, &t).expect("ρ differs");
    assert_eq!(d, "residual ρ has 1 value(s), the fit's 0");
}

/// §5 rows 2–4: no silent switch between methods (Q1). An asymptotic draw from a fit
/// with no covariance matrix names the SIR pool when the fit kept one, the refit that
/// would keep one when it did not; a SIR draw without a pool keeps the sampler's own
/// refusal. Each whole text.
///
/// Mutations: the remedy's two arms swapped; either sentence changed; falling back to
/// the pool silently (row 2 runs).
#[test]
fn a_fit_without_a_covariance_matrix_is_refused_with_its_remedy() {
    let model = iiv();
    let mut fit = fit_of(&model);
    let p =
        crate::estimation::uncertainty_samples::fitted_params_from_result(&fit, &model).unwrap();
    fit.covariance_matrix = None;
    let head = "Asymptotic uncertainty requires FitResult.covariance_matrix, and this fit has \
                none (its covariance step did not run or did not succeed).";
    let code = "E_FIT_INPUT_MISSING".to_string();
    assert_eq!(
        refusal(with_spec(
            &model,
            &p,
            &fit,
            UncertaintyMethod::Asymptotic,
            1
        )),
        (
            code.clone(),
            format!(
                "{head} Refit with `covariance_fallback = sir` and `sir_keep_samples = true` \
                 to draw from the SIR pool instead (#223)."
            )
        )
    );
    assert_eq!(
        refusal(with_spec(&model, &p, &fit, UncertaintyMethod::Sir, 1)),
        (
            code.clone(),
            "SIR uncertainty requires FitResult.sir_resamples_packed; run the fit with `sir = \
             true` and `sir_keep_samples = true`."
                .to_string()
        )
    );
    let packed = crate::estimation::parameterization::pack_params(&p);
    fit.sir_resamples_packed = Some(vec![packed]);
    assert_eq!(
        refusal(with_spec(
            &model,
            &p,
            &fit,
            UncertaintyMethod::Asymptotic,
            1
        )),
        (
            code,
            format!(
                "{head} This fit carries a SIR resample pool: set `method` to \
                 `UncertaintyMethod::Sir` to draw from it."
            )
        )
    );
    assert!(
        with_spec(&model, &p, &fit, UncertaintyMethod::Sir, 1).is_ok(),
        "the pool the remedy names draws"
    );
}

/// C6 (§5 row 1, Q4): supplied η and uncertainty are refused under **both** unmatched
/// policies, before any other check of the η — through the sampler and through the seam.
/// Without uncertainty the same η run.
///
/// Mutations: refusing under `Refuse` only (the `Draw` cell runs); checking after
/// `resolve_supplied_etas` (the `Refuse` cell with an unmatched subject gets the
/// unmatched-ID refusal instead); any sentence changed.
#[test]
fn supplied_etas_are_refused_under_uncertainty_for_both_policies() {
    let model = iiv();
    let fit = fit_of(&model);
    let p =
        crate::estimation::uncertainty_samples::fitted_params_from_result(&fit, &model).unwrap();
    // Subject 2 has no row: under `Refuse` alone it would be refused for that.
    let etas =
        SuppliedEtas::from_table(vec!["1".into()], vec!["ETA_CL".into()], vec![vec![0.1]]).unwrap();
    let want = (
        "E_SUPPLIED_ETAS".to_string(),
        "parameter uncertainty (`with_uncertainty`) cannot be combined with supplied random \
         effects (`with_etas`): each supplied η was estimated at the fit's θ, so pairing it \
         with a drawn θ does not describe the same patient. Simulate the supplied subjects \
         without uncertainty, or drop the supplied η; #2154 tracks drawing η from its \
         posterior under each draw"
            .to_string(),
    );
    let draws = [p.clone()];
    for policy in [UnmatchedSubjects::Refuse, UnmatchedSubjects::Draw] {
        let base = AdaptiveInputs::new(&p)
            .with_etas(&etas)
            .with_unmatched(policy);
        for (how, inputs) in [
            (
                "sampler",
                base.with_uncertainty(
                    &fit,
                    UncertaintySpec {
                        method: UncertaintyMethod::Asymptotic,
                        n_draws: 1,
                    },
                ),
            ),
            ("seam", base.with_injected_draws(&draws)),
        ] {
            let r = simulate_adaptive_with_inputs(
                &model,
                &pop(2),
                &inputs,
                1,
                fixed_bolus,
                &opts(None),
            );
            assert_eq!(refusal(r), want, "{policy:?}, {how}");
        }
    }
    let ok = AdaptiveInputs::new(&p)
        .with_etas(&etas)
        .with_unmatched(UnmatchedSubjects::Draw);
    assert!(
        simulate_adaptive_with_inputs(&model, &pop(2), &ok, 1, fixed_bolus, &opts(None)).is_ok(),
        "without uncertainty the supplied η run"
    );
}

/// C8 (audit A9, §5 row 12): a driver refusal names `(sim s)` without uncertainty —
/// the exact old text, as the assay-magnitude anchors pin — and `(draw d, sim s)` with
/// it. The controller refuses on its second run only: subject 1's sim 2 without
/// uncertainty, its draw 2 with.
///
/// Mutations: always printing the draw (the off side); never printing it (the on side).
#[test]
fn driver_refusals_name_the_draw_only_under_uncertainty() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let model = iiv();
    let p = &model.default_params;
    let run = |inputs: &AdaptiveInputs<'_>, n_sim| {
        let minted = AtomicUsize::new(0);
        let make = || {
            let k = minted.fetch_add(1, Ordering::SeqCst);
            move |_ctx: &ControllerCtx| {
                let amt = if k == 1 { -1.0 } else { 100.0 };
                vec![DoseAction::Bolus { amt, cmt: 1 }]
            }
        };
        simulate_adaptive_with_inputs(&model, &pop(1), inputs, n_sim, make, &opts(None))
    };
    let (_, off) = refusal(run(&AdaptiveInputs::new(p), 2));
    assert!(off.starts_with("subject '1' (sim 2): "), "{off}");
    let draws = [p.clone(), p.clone()];
    let (_, on) = refusal(run(&AdaptiveInputs::new(p).with_injected_draws(&draws), 1));
    assert!(on.starts_with("subject '1' (draw 2, sim 1): "), "{on}");
    assert_eq!(
        off.trim_start_matches("subject '1' (sim 2): "),
        on.trim_start_matches("subject '1' (draw 2, sim 1): "),
        "only the replicate tag differs"
    );
}

/// Q5 (θ̂): a dose expression that reads only θ is the prescriber's rule, evaluated once at
/// the point estimate, so every draw starts on the same dose although draw 2's TVCL is 40%
/// higher. The straddle: the same draws do change the trajectories. Also the spec entry's
/// run under uncertainty: two draws, both tagged.
///
/// Regression it pins: a θ-only plan re-evaluated at each draw's θ (draw 2 would start at
/// 70, not 50).
#[test]
fn a_theta_only_dose_expression_stays_at_the_point_estimate_in_every_draw() {
    let p = crate::parser::model_parser::parse_full_model(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.0 FIX
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [0, 24]
  start_dose = 10 * TVCL
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal > 100000 : hold
"#,
    )
    .expect("parse");
    let spec = p.adaptive_dosing.as_ref().expect("block");
    let p0 = p.model.default_params.clone();
    let mut p1 = p0.clone();
    p1.theta[0] = 7.0;
    let draws = [p0.clone(), p1];
    let mut o = AdaptiveSimulateOptions::default();
    o.seed = Some(SEED);
    let r = simulate_adaptive_from_spec_with_inputs(
        &p.model,
        &pop(1),
        &AdaptiveInputs::new(&p0).with_injected_draws(&draws),
        1,
        spec,
        &o,
    )
    .expect("runs");
    assert_eq!(draws_of(&r), [1, 2].into());
    for draw in [1, 2] {
        let first = r
            .ledger
            .iter()
            .find(|e| e.draw == draw)
            .unwrap_or_else(|| panic!("draw {draw} doses"));
        assert_eq!(first.amt, 50.0, "draw {draw}: 10 × θ̂ = 50");
    }
    assert_ne!(
        draw_rows(&r, 1).traj,
        draw_rows(&r, 2).traj,
        "θ_d moves the run"
    );
}
