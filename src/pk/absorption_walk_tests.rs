//! Tier-1 tests for the transit/IG absorption walk (#1560).
//!
//! Two independent references per fixture: the committed pure-stdlib quadratures
//! (`tests/data/absorption_walk_1cpt_quadrature.{py,csv}` for 1-cpt, and #1569's
//! `absorption_fixed_at_dose_quadrature.{py,csv}` for 2-cpt), and the ODE twin at
//! `ode_reltol = 1e-12`. The walk shares no code with either: the quadrature integrates the
//! kernel density numerically, and the twin integrates the `transit()`/`igd()` forcing.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::DoseEvent;
use std::collections::HashMap;

/// `(dose, obs, pk_only)` per-event parameters for an IOV subject, built as `predict_iov`
/// builds them: each record under its own occasion's κ and covariate snapshot.
pub(super) fn iov_event_params(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
) -> (Vec<PkParams>, Vec<PkParams>, Vec<PkParams>) {
    let groups = crate::stats::likelihood::iov_occasion_groups(subject);
    let combined = |occ: u32| -> Vec<f64> {
        let mut c = eta.to_vec();
        match groups.iter().position(|(o, _)| *o == occ) {
            Some(g) => c.extend_from_slice(&kappas[g]),
            None => c.extend(std::iter::repeat_n(0.0, model.n_kappa)),
        }
        c
    };
    let dose = (0..subject.doses.len())
        .map(|k| {
            (model.pk_param_fn)(
                theta,
                &combined(subject.dose_occasions[k]),
                subject.dose_cov(k),
                subject.doses[k].time,
            )
        })
        .collect();
    let obs = (0..subject.obs_times.len())
        .map(|j| {
            (model.pk_param_fn)(
                theta,
                &combined(subject.occasions[j]),
                subject.obs_cov(j),
                subject.obs_times[j],
            )
        })
        .collect();
    (dose, obs, Vec::new())
}

/// The ODE twin of a closed-form absorption model — the reference engine.
pub(super) fn twin(model: &CompiledModel) -> &CompiledModel {
    model
        .absorption_ode_equivalent
        .as_ref()
        .expect("the fixture model has an ODE twin")
        .built()
}

/// `scenario → [(time, fixed_at_dose, current_interval)]` from a committed quadrature table.
fn table(path: &str) -> HashMap<String, Vec<(f64, f64, f64)>> {
    let text = std::fs::read_to_string(path).expect("the frozen quadrature table loads");
    let mut out: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 5, "malformed row: {line}");
        let parse = |s: &str| -> f64 { s.parse().expect("a float") };
        out.entry(f[0].to_string())
            .or_default()
            .push((parse(f[2]), parse(f[3]), parse(f[4])));
    }
    out
}

fn wt(w: f64) -> HashMap<String, f64> {
    HashMap::from([("WT".to_string(), w)])
}

/// A subject from `(time, occasion, WT)` doses (100 each, into the depot) and observations.
fn subject_from(doses: &[(f64, u32, f64)], obs: &[(f64, u32, f64)]) -> Subject {
    let n = obs.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: doses
            .iter()
            .map(|&(t, _, _)| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect(),
        obs_times: obs.iter().map(|&(t, _, _)| t).collect(),
        obs_raw_times: vec![],
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: wt(70.0),
        dose_covariates: doses.iter().map(|&(_, _, w)| wt(w)).collect(),
        obs_covariates: obs.iter().map(|&(_, _, w)| wt(w)).collect(),
        pk_only_times: vec![],
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: vec![],
        reset_times: vec![],
        reset_covariates: vec![],
        cens: vec![0; n],
        occasions: obs.iter().map(|&(_, o, _)| o).collect(),
        obs_l2: vec![],
        dose_occasions: doses.iter().map(|&(_, o, _)| o).collect(),
        reset_occasions: vec![],
        pk_only_occasions: Vec::new(),
        fremtype: vec![],
        obs_records: vec![],
    }
}

// ── 1-cpt fixture: tests/data/absorption_walk_1cpt_quadrature.py ─────────────────────────

const DOSES_1: [(f64, u32, f64); 3] = [(0.0, 1, 70.0), (6.0, 2, 72.0), (12.0, 3, 74.0)];
const OBS_1: [(f64, u32, f64); 14] = [
    (1.0, 1, 70.0),
    (2.5, 1, 70.0),
    (4.0, 1, 71.0),
    (5.0, 2, 72.0),
    (5.5, 2, 72.0),
    (7.0, 2, 73.0),
    (9.0, 2, 73.0),
    (11.0, 2, 74.0),
    (13.0, 3, 74.0),
    (15.0, 3, 76.0),
    (18.0, 3, 78.0),
    (24.0, 3, 80.0),
    (36.0, 3, 82.0),
    (48.0, 3, 84.0),
];
/// Index of the observation at `t = 5`, the record opening occasion 2.
const OCC2_OBS: usize = 3;

fn model_1cpt(kernel: &str) -> CompiledModel {
    let (thetas, params, pk) = match kernel {
        "transit" => (
            "  theta TVA(2.0, 0.0, 30.0)\n  theta TVB(4.0, 0.05, 24.0)",
            "  NTR = TVA * exp(KAPPA_A)\n  MTT = TVB * exp(KAPPA_B)",
            "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT, f=FB, lagtime=LAG)",
        ),
        "ig" => (
            "  theta TVA(4.0, 0.05, 24.0)\n  theta TVB(0.3, 0.01, 5.0)",
            "  MAT = TVA * exp(KAPPA_A)\n  CV2 = TVB * exp(KAPPA_B)",
            "pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2, f=FB, lagtime=LAG)",
        ),
        other => panic!("unknown kernel {other}"),
    };
    let src = format!(
        r#"
[parameters]
  theta TVCL(3.0, 0.1, 100.0)
  theta TVV(25.0, 1.0, 500.0)
{thetas}
  theta TVF(0.8, 0.01, 1.0)
  theta TVLAG(0.5, 0.0, 5.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.04
  kappa KAPPA_A ~ 0.04
  kappa KAPPA_B ~ 0.04
  kappa KAPPA_F ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70)^0.75 * exp(KAPPA_CL)
  V   = TVV * exp(ETA_V)
{params}
  FB  = TVF * exp(KAPPA_F)
  LAG = TVLAG
[structural_model]
  {pk}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#
    );
    parse_model_string(&src).expect("the 1-cpt fixture model parses")
}

/// κ per occasion, `[KAPPA_CL, KAPPA_A, KAPPA_B, KAPPA_F]` — the generator's `FULL` / `F_ONLY`.
fn kappas_1cpt(scenario: &str) -> Vec<Vec<f64>> {
    let (kcl, ka, kb, kf) = match scenario {
        "full" => (
            [0.10, -0.15, 0.20],
            [0.00, 0.25, -0.20],
            [0.20, -0.30, 0.35],
            [0.00, -0.10, 0.10],
        ),
        "f_only" => ([0.0; 3], [0.0; 3], [0.0; 3], [0.25, -0.30, 0.20]),
        other => panic!("unknown scenario {other}"),
    };
    (0..3).map(|o| vec![kcl[o], ka[o], kb[o], kf[o]]).collect()
}

const ETA_V: f64 = 0.1;

/// The walk's predictions and states for one fixture, plus the twin's predictions.
struct Run {
    walk: Vec<f64>,
    states: Vec<Vec<f64>>,
    twin: Vec<f64>,
    dose: Vec<PkParams>,
    obs: Vec<PkParams>,
}

fn run(model: &CompiledModel, subject: &Subject, kappas: &[Vec<f64>]) -> Run {
    let theta = model.default_params.theta.clone();
    let (dose, obs, only) = iov_event_params(model, subject, &theta, &[ETA_V], kappas);
    let mut states = Vec::new();
    let walk = absorption_walk_predictions(
        model.pk_model,
        subject,
        &dose,
        &obs,
        &only,
        Some(&mut states),
    )
    .expect("the fixture sits inside the tilting domain");
    let twin = crate::pk::predict_iov(twin(model), subject, &theta, &[ETA_V], kappas);
    Run {
        walk,
        states,
        twin,
        dose,
        obs,
    }
}

/// Worst relative error of `got` against `want`, after asserting every value finite (a
/// `NaN` folded through `f64::max` would be discarded).
fn worst_rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            assert!(
                g.is_finite() && w.is_finite(),
                "non-finite value: {g} vs {w}"
            );
            (g - w).abs() / w.abs()
        })
        .fold(0.0, f64::max)
}

/// T3 + T10 (#1560): the 1-cpt walk under IOV and a time-varying covariate reproduces the
/// fixed-at-dose quadrature and the ODE twin — on a fixture whose later occasion opens
/// while dose 1 is **still absorbing and still in central**, so both the carried amount and
/// the windowed tail are live across a boundary where `ke` genuinely changes. Occasion 2
/// opens on an observation record (`t = 5`), occasion 3 on a dose record (`t = 12`), so a
/// walk that took an interval's parameters from the wrong side of either record is off.
///
/// The straddle is asserted, not assumed: the rules `fixed_at_dose` and `current_interval`
/// differ by > 10 % on `full` (so the table can tell a dose-time kernel from a current one),
/// and at `t = 5` dose 1 still holds > 10 % of its mass in the kernel while central holds
/// > 20 % of its eventual peak. `transit_f_only` (T10) carries IOV on `F` alone, so the two
/// kernel rules coincide there and only the per-dose `F` separates right from wrong.
#[test]
fn walk_1cpt_matches_quadrature_and_twin_across_a_live_boundary() {
    let reference = table("tests/data/absorption_walk_1cpt_quadrature.csv");
    let subject = subject_from(&DOSES_1, &OBS_1);
    for (kernel, scenario) in [("transit", "full"), ("transit", "f_only"), ("ig", "full")] {
        let name = format!("{kernel}_{scenario}");
        let model = model_1cpt(kernel);
        assert!(
            walk_eligible(&model, &subject),
            "{name}: the fixture must be walk-eligible"
        );
        let rows = &reference[&name];
        assert_eq!(rows.len(), OBS_1.len());
        for (j, row) in rows.iter().enumerate() {
            assert_eq!(row.0, OBS_1[j].0, "{name}: table time {j}");
        }
        let r = run(&model, &subject, &kappas_1cpt(scenario));
        let fixed: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let current: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let vs_table = worst_rel(&r.walk, &fixed);
        let vs_twin = worst_rel(&r.walk, &r.twin);
        let split = worst_rel(&current, &fixed);
        println!(
            "#1560 1-cpt {name}: walk vs quadrature {vs_table:.3e}, walk vs twin {vs_twin:.3e}, \
             rules split {split:.3e}"
        );
        // The rule straddle on the committed file itself.
        if scenario == "full" {
            assert!(
                split > 0.1,
                "{name}: the table no longer separates the rules"
            );
        } else {
            assert!(
                split == 0.0,
                "{name}: kernel rules must coincide with IOV on F only"
            );
        }
        // The carry-over straddle at the record opening occasion 2.
        let dose1_mass = r.dose[0].f_bio() * 100.0;
        let peak = r.states.iter().map(|s| s[1]).fold(0.0, f64::max);
        let s5 = &r.states[OCC2_OBS];
        assert!(
            s5[0] > 0.10 * dose1_mass && s5[1] > 0.2 * peak,
            "{name}: at t = 5 dose 1 must still be absorbing (kernel {:.3} of {dose1_mass:.3}) \
             and in central ({:.3} of peak {peak:.3})",
            s5[0],
            s5[1]
        );
        let ke = |p: &PkParams| p.cl() / p.v();
        if scenario == "full" {
            assert!(
                (ke(&r.obs[OCC2_OBS]) / ke(&r.obs[OCC2_OBS - 1]) - 1.0).abs() > 0.1,
                "{name}: ke must genuinely change at the occasion boundary"
            );
        } else {
            // T10: dose 1's F differs from the F in force while it is still absorbing.
            assert!(
                (r.dose[0].f_bio() / r.obs[OCC2_OBS].f_bio() - 1.0).abs() > 0.2,
                "{name}: F must differ across the boundary dose 1 absorbs through"
            );
        }
        // Realised at 31f6abaa: ≤ 7.2e-15 vs the quadrature (IG), ≤ 8.2e-12 vs the twin
        // (transit, the twin's own `ode_reltol = 1e-12` error). Bounds carry ~12x headroom
        // and sit eleven orders inside the smallest rules split (19 %).
        assert!(
            vs_table < 1e-13,
            "{name}: walk vs quadrature {vs_table:.3e}"
        );
        assert!(vs_twin < 1e-10, "{name}: walk vs twin {vs_twin:.3e}");
    }
}

// ── 2-cpt fixture: #1569's tests/data/absorption_fixed_at_dose_quadrature.py ─────────────

const DOSES_2: [(f64, u32, f64); 4] = [
    (0.0, 1, 70.0),
    (6.0, 1, 72.0),
    (12.0, 2, 74.0),
    (18.0, 3, 80.0),
];
const OBS_2: [(f64, u32, f64); 15] = [
    (1.0, 1, 70.0),
    (2.5, 1, 70.0),
    (4.0, 1, 71.0),
    (5.5, 1, 72.0),
    (7.5, 1, 72.0),
    (9.0, 1, 73.0),
    (11.0, 1, 73.0),
    (13.5, 2, 74.0),
    (15.0, 2, 76.0),
    (16.5, 3, 77.0),
    (19.5, 3, 80.0),
    (22.0, 3, 82.0),
    (26.0, 3, 84.0),
    (36.0, 3, 86.0),
    (48.0, 3, 88.0),
];

fn model_2cpt(kernel: &str) -> CompiledModel {
    let (thetas, params, pk) = match kernel {
        "transit" => (
            "  theta TVA(3.0, 0.0, 30.0)\n  theta TVB(2.5, 0.05, 24.0)",
            "  NTR = TVA * exp(KAPPA_A)\n  MTT = TVB * exp(KAPPA_B)",
            "pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT, f=FB, lagtime=LAG)",
        ),
        "ig" => (
            "  theta TVA(2.5, 0.05, 24.0)\n  theta TVB(0.5, 0.01, 5.0)",
            "  MAT = TVA * exp(KAPPA_A)\n  CV2 = TVB * exp(KAPPA_B)",
            "pk two_cpt_ig(cl=CL, v1=V1, q=Q, v2=V2, mat=MAT, cv2=CV2, f=FB, lagtime=LAG)",
        ),
        other => panic!("unknown kernel {other}"),
    };
    let src = format!(
        r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(30.0, 1.0, 500.0)
  theta TVQ(3.0, 0.1, 100.0)
  theta TVV2(60.0, 1.0, 1000.0)
{thetas}
  theta TVF(0.8, 0.01, 1.0)
  theta TVLAG(0.7, 0.0, 5.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.04
  kappa KAPPA_A ~ 0.04
  kappa KAPPA_B ~ 0.04
  kappa KAPPA_F ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70)^0.75 * exp(KAPPA_CL)
  V1  = TVV1 * exp(ETA_V)
  Q   = TVQ
  V2  = TVV2
{params}
  FB  = TVF * exp(KAPPA_F)
  LAG = TVLAG
[structural_model]
  {pk}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#
    );
    parse_model_string(&src).expect("the 2-cpt fixture model parses")
}

/// κ per occasion — #1569's generator's `FULL` / `CL_ONLY`.
fn kappas_2cpt(scenario: &str) -> Vec<Vec<f64>> {
    let (kcl, ka, kb, kf) = match scenario {
        "full" => (
            [0.10, -0.15, 0.20],
            [0.00, 0.25, -0.20],
            [0.20, -0.30, 0.35],
            [0.00, -0.10, 0.10],
        ),
        "cl_only" => ([0.10, -0.15, 0.20], [0.0; 3], [0.0; 3], [0.0; 3]),
        other => panic!("unknown scenario {other}"),
    };
    (0..3).map(|o| vec![kcl[o], ka[o], kb[o], kf[o]]).collect()
}

/// T4 (#1560): the 2-cpt walk against #1569's independent fixed-at-dose quadrature and
/// the ODE twin — four overlapping doses, IOV on `CL`, both kernel parameters and `F`, a
/// lag, `WT` on `CL` changing mid-occasion; occasion 2 opens on a dose record and
/// occasion 3 on an observation record while dose 3 is still absorbing. The peripheral half
/// of each window (`k12/(α−β)·(W(β) − W(α))`) reaches this `IPRED` through `k21` over the
/// 48 h profile; its state is pinned directly by
/// [`walk_reduces_to_the_static_superposition_including_states`].
#[test]
fn walk_2cpt_matches_quadrature_and_twin() {
    let reference = table("tests/data/absorption_fixed_at_dose_quadrature.csv");
    let subject = subject_from(&DOSES_2, &OBS_2);
    for kernel in ["transit", "ig"] {
        let model = model_2cpt(kernel);
        let theta = model.default_params.theta.clone();
        for scenario in ["full", "cl_only"] {
            let name = format!("{kernel}_{scenario}");
            let kappas = kappas_2cpt(scenario);
            let fixed: Vec<f64> = reference[&name].iter().map(|r| r.1).collect();
            let (dose, obs, only) = iov_event_params(&model, &subject, &theta, &[ETA_V], &kappas);
            let walk =
                absorption_walk_predictions(model.pk_model, &subject, &dose, &obs, &only, None)
                    .unwrap_or_else(|| {
                        panic!("{name}: the fixture sits inside the tilting domain")
                    });
            let twin = crate::pk::predict_iov(twin(&model), &subject, &theta, &[ETA_V], &kappas);
            let vs_table = worst_rel(&walk, &fixed);
            let vs_twin = worst_rel(&walk, &twin);
            println!(
                "#1560 2-cpt {name}: walk vs quadrature {vs_table:.3e}, vs twin {vs_twin:.3e}"
            );
            // Realised at 31f6abaa: ≤ 7.6e-16 vs the quadrature, ≤ 3.6e-12 vs the twin.
            assert!(
                vs_table < 1e-13,
                "{name}: walk vs quadrature {vs_table:.3e}"
            );
            assert!(vs_twin < 1e-10, "{name}: walk vs twin {vs_twin:.3e}");
        }
    }
}

/// With nothing varying — one parameter set, no IOV, no covariate change — the walk must
/// reduce to the static closed-form superposition (`compute_predictions`,
/// `predict_all_states`), which is a separate implementation: per-dose `one_cpt_transit_g`
/// / `convolve_2cpt` / `convolve_2cpt_peripheral` and the `*_depot` amounts, evaluated once
/// from each arrival with no windows and no carried state. This pins the walk's **states**
/// — the kernel `depot`, and the 2-cpt peripheral that the IPRED references only see
/// through `k21` — which no other reference here can: the twin's states path holds the first
/// observation's parameters for the whole timeline on a TV/IOV subject (documented, and
/// warned as `W_DERIVED_CMT_TV_ODE`), so it is not an oracle for them.
#[test]
fn walk_reduces_to_the_static_superposition_including_states() {
    let static_subject = |doses: &[(f64, u32, f64)], obs: &[(f64, u32, f64)]| {
        let mut s = subject_from(doses, obs);
        s.dose_covariates.clear();
        s.obs_covariates.clear();
        s
    };
    let cases = [
        (model_1cpt("transit"), static_subject(&DOSES_1, &OBS_1)),
        (model_1cpt("ig"), static_subject(&DOSES_1, &OBS_1)),
        (model_2cpt("transit"), static_subject(&DOSES_2, &OBS_2)),
        (model_2cpt("ig"), static_subject(&DOSES_2, &OBS_2)),
    ];
    for (model, subject) in &cases {
        let name = format!("{:?}", model.pk_model);
        let theta = model.default_params.theta.clone();
        let eta = [ETA_V, 0.0, 0.0, 0.0, 0.0];
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
        let dose = vec![pk; subject.doses.len()];
        let obs = vec![pk; subject.obs_times.len()];
        let mut states = Vec::new();
        let walk = absorption_walk_predictions(
            model.pk_model,
            subject,
            &dose,
            &obs,
            &[],
            Some(&mut states),
        )
        .unwrap_or_else(|| panic!("{name}: inside the domain"));
        let want = crate::pk::compute_predictions(model.pk_model, subject, &pk);
        let want_states = crate::pk::predict_all_states(model.pk_model, subject, &pk);
        let ipred_err = worst_rel(&walk, &want);
        let mut state_err = 0.0_f64;
        for (j, (got, exp)) in states.iter().zip(&want_states).enumerate() {
            assert_eq!(got.len(), exp.len(), "{name} obs {j}: state layout");
            for (c, (&g, &e)) in got.iter().zip(exp).enumerate() {
                assert!(g.is_finite() && e.is_finite(), "{name} obs {j} state {c}");
                // Relative to the observation's total amount: the depot decays to ~0.
                let scale = exp.iter().map(|x| x.abs()).sum::<f64>();
                state_err = state_err.max((g - e).abs() / scale);
            }
        }
        println!("#1560 static reduction {name}: ipred {ipred_err:.3e}, states {state_err:.3e}");
        // Every compartment is live somewhere on the profile, so each is actually compared.
        for c in 0..want_states[0].len() {
            assert!(
                want_states.iter().any(|s| s[c] > 1.0),
                "{name}: compartment {c} never carries drug"
            );
        }
        // Realised ≤ 1.2e-15 (ipred) and ≤ 3.2e-16 (states); ~10x headroom.
        assert!(ipred_err < 1e-14, "{name}: ipred {ipred_err:.3e}");
        assert!(state_err < 1e-14, "{name}: states {state_err:.3e}");
    }
}

/// The domain is decided per **(interval, open dose) pair** (#1560 plan amendment 1): under
/// fixed-at-dose absorption, an interval's `ke` must sit below the MGF abscissa of every
/// dose that has arrived by its start, each at that dose's own `KTR`. One subject, three
/// configurations that differ in a single number each:
///
/// * a slow dose 1 (`KTR₁ = 0.3`) is still open when occasion 2 raises `ke` to `0.4` →
///   **twin**, although occasion 2's own dose (`KTR₂ = 3`) alone would pass — the pair, not
///   the interval, decides;
/// * the same with occasion 2's `ke` lowered to `0.2` → **walk** (the straddle);
/// * the same fast `ke = 0.4`, but occasion 2 now ends before dose 1 arrives → **walk**:
///   a dose that has not arrived is not in any pair yet.
#[test]
fn walk_domain_is_decided_per_interval_and_open_dose_pair() {
    let model = model_1cpt("transit");
    let disp = |ke: f64| PkDual {
        cl: ke * 10.0,
        v: 10.0,
        q: 0.0,
        v2: 0.0,
        ka: 0.0,
        q3: 0.0,
        v3: 0.0,
        f: 1.0,
    };
    let dose = |arrival: f64, ktr: f64| AbsDose {
        arrival,
        mass: 100.0,
        a: 2.0,
        b: 3.0 / ktr,
    };
    // Dose 1 at t = 0 (slow), dose 2 at t = 6 (fast). Obs at 3 (occasion 1) and 8, 12
    // (occasion 2, the fast-`ke` records).
    let subject = subject_from(
        &[(0.0, 1, 70.0), (6.0, 2, 70.0)],
        &[(3.0, 1, 70.0), (8.0, 2, 70.0), (12.0, 2, 70.0)],
    );
    let domain = |ke2: f64, arrival1: f64| {
        let doses = [dose(arrival1, 0.3), dose(6.0, 3.0)];
        let lags = [arrival1, 0.0];
        let schedule = EventSchedule::for_subject(&subject, model.pk_model, &subject.doses, &lags);
        let at_dose = [disp(0.1), disp(ke2)];
        let at_obs = [disp(0.1), disp(ke2), disp(ke2)];
        walk_domain(model.pk_model, &schedule, &doses, &at_dose, &at_obs, &[])
    };
    assert!(
        0.4 < 3.0,
        "occasion 2's own dose admits the fast ke on its own"
    );
    assert_eq!(
        domain(0.4, 0.0),
        WalkDomain::Twin,
        "slow open dose × fast ke"
    );
    assert_eq!(
        domain(0.2, 0.0),
        WalkDomain::Walk,
        "the straddle: ke below KTR₁"
    );
    assert_eq!(
        domain(0.4, 20.0),
        WalkDomain::Walk,
        "dose 1 arriving after the fast interval forms no pair with it"
    );
    // A dose arriving exactly at the start of the subject's **final** interval pairs with it:
    // no later interval re-checks the pair, so this is the only case that decides whether
    // the filter is `arrival ≤ start` (as it must be — the dose delivers over `[8, 12]`) or
    // `<`. Dose 2 (slow, `KTR = 0.3`) lags to `t = 8`, the last observation is at `12`.
    let final_interval = |ke2: f64| {
        let doses = [dose(0.0, 3.0), dose(8.0, 0.3)];
        let schedule =
            EventSchedule::for_subject(&subject, model.pk_model, &subject.doses, &[0.0, 2.0]);
        let at_dose = [disp(0.1), disp(ke2)];
        let at_obs = [disp(0.1), disp(ke2), disp(ke2)];
        walk_domain(model.pk_model, &schedule, &doses, &at_dose, &at_obs, &[])
    };
    assert_eq!(
        final_interval(0.4),
        WalkDomain::Twin,
        "a dose arriving at the final interval's start × fast ke"
    );
    assert_eq!(
        final_interval(0.2),
        WalkDomain::Walk,
        "the straddle: ke below the late dose's KTR"
    );
    // And the walk itself honours the verdict: `None` sends the subject to the twin.
    let schedule =
        EventSchedule::for_subject(&subject, model.pk_model, &subject.doses, &[0.0, 0.0]);
    let doses = [dose(0.0, 0.3), dose(6.0, 3.0)];
    let at_obs = [disp(0.1), disp(0.4), disp(0.4)];
    assert!(absorption_walk_g(
        model.pk_model,
        &schedule,
        &doses,
        &[disp(0.1), disp(0.4)],
        &at_obs,
        &[],
        3,
        None
    )
    .is_none());
}

// ── Routing (T6) ──────────────────────────────────────────────────────────────────────

/// Whitespace-split rows of a NONMEM output file.
fn nm_rows(path: &str) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// `THETA1..n` of the final-estimate row of `nonmem_anchor/results/<name>.ext`.
fn nm_thetas(name: &str, n: usize) -> Vec<f64> {
    let rows = nm_rows(&format!("nonmem_anchor/results/{name}.ext"));
    let header = rows
        .iter()
        .find(|r| r[0] == "ITERATION")
        .expect(".ext header");
    let last = rows
        .iter()
        .find(|r| r[0] == "-1000000000")
        .expect(".ext final row");
    (1..=n)
        .map(|k| {
            let i = header
                .iter()
                .position(|h| *h == format!("THETA{k}"))
                .expect("THETA");
            last[i].parse().expect("a float")
        })
        .collect()
}

/// Per subject ID, `ETA(1..n)` of `nonmem_anchor/results/<name>.phi`.
fn nm_etas(name: &str, n: usize) -> HashMap<String, Vec<f64>> {
    let rows = nm_rows(&format!("nonmem_anchor/results/{name}.phi"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("SUBJECT_NO"))
        .expect(".phi header");
    let at = |l: &str| header.iter().position(|h| h == l).expect(l);
    let id_col = at("ID");
    rows.iter()
        .filter(|r| r.len() == header.len() && r[0] != "SUBJECT_NO")
        .map(|r| {
            let id: f64 = r[id_col].parse().expect("ID");
            let eta = (1..=n)
                .map(|k| r[at(&format!("ETA({k})"))].parse().expect("ETA"))
                .collect();
            (format!("{}", id as i64), eta)
        })
        .collect()
}

/// The route on the committed NONMEM anchor populations (#1560 plan, Step 4: measure the
/// fraction the walk hands back to the twin): every subject of `transit_iov` (IOV on `CL`),
/// `transit_iov_mtt` and `ig_iov_mat` (IOV on `CL` **and** the kernel, doses absorbing across
/// the occasion boundary), each at NONMEM's final θ and POSTHOC η/κ — the point the IPRED
/// anchors (`tests/absorption_iov_carryover_nonmem_anchor.rs`) evaluate — is served by the
/// walk unless a pair leaves the tilting domain, and `predict_iov` on a walk subject equals
/// the walk's own output bit for bit. Measured at 0507337a: `transit_iov` 24 / 24 and
/// `transit_iov_mtt` 24 / 24 on the walk, `ig_iov_mat` **13 / 24** — the other 11 have an
/// occasion whose `ke` reaches `1/(2·MAT·CV²)` of a dose still in the walk (its `CV² = 0.5`
/// puts the IG abscissa near `ke`), and go to the twin. Pinned exactly, so a router that
/// admits more subjects (or fewer) than the domain allows changes a count.
#[test]
fn nonmem_anchor_populations_route_to_the_walk_at_nonmem_ebes() {
    // (name, pk line, kernel lines, κ names, ETA count, ETA → ([η], [κ per occasion]))
    type Split = fn(&[f64]) -> (Vec<f64>, Vec<Vec<f64>>);
    let expected = [24usize, 24, 13];
    let cases: [(&str, &str, &str, &str, usize, Split); 3] = [
        (
            "transit_iov",
            "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)",
            "  CL = TVCL * exp(KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  MTT = TVA\n  NTR = TVB",
            "  omega ETA_V ~ 0.1\n  kappa KAPPA_CL ~ 0.05",
            4,
            |e| (vec![e[0]], (0..3).map(|g| vec![e[1 + g]]).collect()),
        ),
        (
            "transit_iov_mtt",
            "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)",
            "  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  \
             MTT = TVA * exp(KAPPA_ABS)\n  NTR = TVB",
            "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  kappa KAPPA_CL ~ 0.05\n  \
             kappa KAPPA_ABS ~ 0.05",
            8,
            |e| {
                let k = (0..3).map(|g| vec![e[2 + g], e[5 + g]]).collect();
                (e[..2].to_vec(), k)
            },
        ),
        (
            "ig_iov_mat",
            "pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)",
            "  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  \
             MAT = TVA * exp(KAPPA_ABS)\n  CV2 = TVB",
            "  omega ETA_CL ~ 0.1\n  omega ETA_V ~ 0.1\n  kappa KAPPA_CL ~ 0.05\n  \
             kappa KAPPA_ABS ~ 0.05",
            8,
            |e| {
                let k = (0..3).map(|g| vec![e[2 + g], e[5 + g]]).collect();
                (e[..2].to_vec(), k)
            },
        ),
    ];
    for ((name, pk, kernel, randoms, n_eta, split), want_walked) in cases.into_iter().zip(expected)
    {
        let th = nm_thetas(name, 4);
        let src = format!(
            "[parameters]\n  theta TVCL({}, FIX)\n  theta TVV({}, FIX)\n  theta TVA({}, FIX)\n  \
             theta TVB({}, FIX)\n{randoms}\n  sigma PROP_ERR ~ 0.1 (sd)\n\
             [individual_parameters]\n{kernel}\n[structural_model]\n  {pk}\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n",
            th[0], th[1], th[2], th[3]
        );
        let model = parse_model_string(&src).expect("the anchor model parses");
        let pop = crate::read_nonmem_csv(
            std::path::Path::new(&format!("nonmem_anchor/{name}.csv")),
            None,
            Some("OCC"),
        )
        .expect("the anchor data loads");
        let etas = nm_etas(name, n_eta);
        let theta = model.default_params.theta.clone();
        let mut walked = 0usize;
        for s in &pop.subjects {
            let (bsv, kappas) = split(&etas[&s.id]);
            let stacked: Vec<f64> = bsv.iter().chain(kappas.iter().flatten()).copied().collect();
            let route = crate::pk::effective_model_for_eval_iov(&model, s, &theta, &stacked);
            if std::ptr::eq(route, &model) {
                walked += 1;
                let got = crate::pk::predict_iov(&model, s, &theta, &bsv, &kappas);
                let (d, o, p) = iov_event_params(&model, s, &theta, &bsv, &kappas);
                let want = absorption_walk_predictions(model.pk_model, s, &d, &o, &p, None)
                    .expect("the router admitted it");
                assert!(
                    got.iter()
                        .zip(&want)
                        .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
                    "{name} subject {}: predict_iov is not the walk",
                    s.id
                );
            }
        }
        println!(
            "#1560 route on {name}: {walked} of {} subjects on the walk at NONMEM's EBEs",
            pop.subjects.len()
        );
        assert_eq!(pop.subjects.len(), 24, "{name}: subject count");
        assert_eq!(walked, want_walked, "{name}: subjects on the walk");
    }
}

/// 1-cpt transit, IOV on `CL` only (`KTR = 3/4 = 0.75`, `ke = 0.12·e^κ`). WT on `CL` too, so
/// the non-IOV route can be pushed across the same boundary by the covariate.
const TRANSIT_IOV_CL: &str = r#"
[parameters]
  theta TVCL(3.0, 0.1, 100.0)
  theta TVV(25.0, 1.0, 500.0)
  theta TVN(2.0, 0.0, 30.0)
  theta TVMTT(4.0, 0.05, 24.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70)^3 * exp(KAPPA_CL)
  V   = TVV * exp(ETA_V)
  NTR = TVN
  MTT = TVMTT
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
"#;

/// T6 (#1560): the route is decided per subject **at its parameters**, on both sides of the
/// flip-flop gate, and every consumer takes the same one.
///
/// Occasion 2 opens at `t = 5` while dose 1 (`KTR = 0.75`) is still absorbing. `κ_CL(occ 2)`
/// sets that occasion's `ke`: at `ln(0.9/0.12)` it is `0.9 ≥ KTR` → the **twin**; at
/// `ln(0.5/0.12)` it is `0.5 < KTR` → the **walk**. Occasion 1's `ke` is `0.12` in both, so a
/// router that looked only at `t = 0` / η (the pre-#1560 `absorption_flip_flop_at`) or ignored
/// κ would call both "walk". For each side, three consumers must agree: the router
/// (`effective_model_for_eval_iov`), the value path (`predict_iov`, bit-identical to the
/// chosen engine) and the adjoint's recording (walk counter). The non-IOV route is then pushed
/// across the same boundary by a covariate instead: `WT` on `CL` to the third power.
/// Finally the structural exclusions — SS, infusion — never reach the walk.
#[test]
fn route_is_decided_per_subject_at_its_parameters_on_both_sides_of_flip_flop() {
    let model = parse_model_string(TRANSIT_IOV_CL).expect("parse");
    let tw = twin(&model);
    let theta = model.default_params.theta.clone();
    let subject = subject_from(
        &[(0.0, 1, 70.0), (6.0, 2, 70.0)],
        &[
            (1.0, 1, 70.0),
            (3.0, 1, 70.0),
            (5.0, 2, 70.0),
            (7.0, 2, 70.0),
            (9.0, 2, 70.0),
        ],
    );
    assert!(walk_eligible(&model, &subject));
    for (ke2, want_walk) in [(0.9, false), (0.5, true)] {
        let kappas = vec![vec![0.0], vec![(ke2 / 0.12_f64).ln()]];
        let stacked = [ETA_V, kappas[0][0], kappas[1][0]];
        let route = crate::pk::effective_model_for_eval_iov(&model, &subject, &theta, &stacked);
        assert_eq!(
            std::ptr::eq(route, &model),
            want_walk,
            "ke₂ = {ke2}: router says {}",
            if std::ptr::eq(route, &model) {
                "walk"
            } else {
                "twin"
            }
        );
        let got = crate::pk::predict_iov(&model, &subject, &theta, &[ETA_V], &kappas);
        let engine = if want_walk {
            let (d, o, p) = iov_event_params(&model, &subject, &theta, &[ETA_V], &kappas);
            absorption_walk_predictions(model.pk_model, &subject, &d, &o, &p, None)
                .expect("in domain")
        } else {
            crate::pk::predict_iov(tw, &subject, &theta, &[ETA_V], &kappas)
        };
        for (a, b) in got.iter().zip(&engine) {
            assert!(
                a.is_finite() && a.to_bits() == b.to_bits(),
                "ke₂ = {ke2}: {a} vs {b}"
            );
        }
        use crate::pk::absorption_walk::ABSORPTION_SENS_WALK_RUNS;
        ABSORPTION_SENS_WALK_RUNS.with(|c| c.set(0));
        crate::sens::ode_provider::adjoint::exact_eta_jacobian(&model, &subject, &theta, &stacked)
            .expect("the adjoint's recording is analytic on either route");
        let runs = ABSORPTION_SENS_WALK_RUNS.with(|c| c.get());
        assert_eq!(
            runs > 0,
            want_walk,
            "ke₂ = {ke2}: sens walk ran {runs} times"
        );
    }

    // Non-IOV: the same boundary crossed by WT (CL ∝ WT³), κ = 0.
    for (wt2, want_walk) in [(137.0, false), (112.0, true)] {
        let mut tv = subject_from(
            &[(0.0, 1, 70.0), (6.0, 2, wt2)],
            &[
                (1.0, 1, 70.0),
                (3.0, 1, 70.0),
                (5.0, 2, wt2),
                (7.0, 2, wt2),
                (9.0, 2, wt2),
            ],
        );
        tv.occasions.clear();
        tv.dose_occasions.clear();
        let eta = [ETA_V, 0.0];
        // ke = 0.12·(WT/70)³: 0.90 at 137 kg, 0.52 at 112 kg, against KTR = 0.75.
        let route = crate::pk::effective_model_for_eval(&model, &tv, &theta, &eta);
        assert_eq!(std::ptr::eq(route, &model), want_walk, "WT₂ = {wt2}");
        let got = crate::pk::compute_predictions_with_tv(&model, &tv, &theta, &eta);
        let want = if want_walk {
            let mut ev = crate::pk::EventPkParams::default();
            crate::pk::compute_event_pk_params_into(&model, &tv, &theta, &eta, &mut ev);
            absorption_walk_predictions(model.pk_model, &tv, &ev.dose, &ev.obs, &ev.pk_only, None)
                .expect("in domain")
        } else {
            crate::pk::compute_predictions_with_tv(tw, &tv, &theta, &eta)
        };
        for (a, b) in got.iter().zip(&want) {
            assert!(a.is_finite() && a.to_bits() == b.to_bits(), "WT₂ = {wt2}");
        }
    }

    // Structural exclusions: an SS dose or an infusion is never walk-eligible.
    let mut ss = subject.clone();
    ss.doses[1] = DoseEvent::new(6.0, 100.0, 1, 0.0, true, 12.0);
    let mut inf = subject.clone();
    inf.doses[1] = DoseEvent::new(6.0, 100.0, 1, 50.0, false, 0.0);
    for (name, s) in [("SS", &ss), ("infusion", &inf)] {
        assert!(!walk_eligible(&model, s), "{name}");
        assert!(
            std::ptr::eq(
                crate::pk::effective_model_for_eval_iov(&model, s, &theta, &[ETA_V, 0.0, 0.0]),
                tw
            ),
            "{name}: the twin"
        );
    }
}

// ── A NaN parameter (#2134) ─────────────────────────────────────────────────────────────

/// The four closed-form absorption models on a static subject, and the observation index
/// of the first record after dose 2 (`t = 6`) has arrived: the record a `NaN` disposition
/// starts on in [`a_nan_parameter_from_one_record_on_is_served_as_nan_by_the_walk`], and the
/// first one a `NaN` kernel on dose 2 reaches.
fn nan_cases() -> Vec<(CompiledModel, Subject, usize)> {
    let static_subject = |doses: &[(f64, u32, f64)], obs: &[(f64, u32, f64)]| {
        let mut s = subject_from(doses, obs);
        s.dose_covariates.clear();
        s.obs_covariates.clear();
        s
    };
    vec![
        (model_1cpt("transit"), static_subject(&DOSES_1, &OBS_1), 5),
        (model_1cpt("ig"), static_subject(&DOSES_1, &OBS_1), 5),
        (model_2cpt("transit"), static_subject(&DOSES_2, &OBS_2), 4),
        (model_2cpt("ig"), static_subject(&DOSES_2, &OBS_2), 4),
    ]
}

/// #2134: a `NaN` `CL`, `V` or `V2` from one observation record on, or a `NaN` kernel
/// parameter on one dose record, is served `NaN` from the record it reaches on, and every
/// record before it keeps the walk's ordinary value — bit for bit, states included. The
/// walk used to answer any `NaN` with `WalkDomain::Invalid` and an all-zero vector for the
/// whole subject, so the records before the `NaN`, which had valid parameters, read 0 too.
///
/// Mutations (run): restore `return Some(preds)` for a `NaN` (send it to `Invalid` in
/// `walk_domain`) → every row dies on its finite-before half; drop the `poisoned` state
/// fill → the `NaN` rows read finite values from a stale state; drop
/// `|| vol.val().is_nan()` from `per_volume` → the `V` rows read 0.
#[test]
fn a_nan_parameter_from_one_record_on_is_served_as_nan_by_the_walk() {
    use crate::types::PK_IDX_V2;
    use crate::types::{PK_IDX_CL, PK_IDX_CV2, PK_IDX_MAT, PK_IDX_MTT, PK_IDX_N, PK_IDX_V};
    for (model, subject, bad_from) in nan_cases() {
        let name = format!("{:?}", model.pk_model);
        let two_cpt = matches!(model.pk_model, PkModel::TwoCptTransit | PkModel::TwoCptIg);
        let (slot_a, slot_b) = match model.pk_model {
            PkModel::OneCptIg | PkModel::TwoCptIg => (PK_IDX_MAT, PK_IDX_CV2),
            _ => (PK_IDX_N, PK_IDX_MTT),
        };
        let theta = model.default_params.theta.clone();
        let pk = (model.pk_param_fn)(
            &theta,
            &[ETA_V, 0.0, 0.0, 0.0, 0.0],
            &subject.covariates,
            0.0,
        );
        let dose = vec![pk; subject.doses.len()];
        let obs = vec![pk; subject.obs_times.len()];
        let mut clean_states = Vec::new();
        let clean = absorption_walk_predictions(
            model.pk_model,
            &subject,
            &dose,
            &obs,
            &[],
            Some(&mut clean_states),
        )
        .expect("inside the domain");
        assert!(
            subject.doses[1].time + pk.lagtime() < subject.obs_times[bad_from]
                && subject.doses[1].time + pk.lagtime() > subject.obs_times[bad_from - 1],
            "{name}: record {bad_from} is not the first after dose 2 arrives"
        );
        let mut rows: Vec<(String, Vec<PkParams>, Vec<PkParams>)> = Vec::new();
        let mut disp = vec![("CL", PK_IDX_CL), ("V", PK_IDX_V)];
        if two_cpt {
            disp.push(("V2", PK_IDX_V2));
        }
        for (label, idx) in disp {
            let mut bad = pk;
            bad.values[idx] = f64::NAN;
            let at_obs = (0..obs.len())
                .map(|j| if j < bad_from { pk } else { bad })
                .collect();
            rows.push((format!("{label} on the observations"), dose.clone(), at_obs));
        }
        for (label, idx) in [("kernel a", slot_a), ("kernel b", slot_b)] {
            let mut at_dose = dose.clone();
            at_dose[1].values[idx] = f64::NAN;
            rows.push((format!("{label} on dose 2"), at_dose, obs.clone()));
        }
        for (label, at_dose, at_obs) in rows {
            let mut states = Vec::new();
            let preds = absorption_walk_predictions(
                model.pk_model,
                &subject,
                &at_dose,
                &at_obs,
                &[],
                Some(&mut states),
            )
            .unwrap_or_else(|| panic!("{name} {label}: the walk serves a NaN, not the twin"));
            for j in 0..preds.len() {
                if j < bad_from {
                    assert!(
                        clean[j] > 0.0 && preds[j].to_bits() == clean[j].to_bits(),
                        "{name} {label}: record {j} precedes the NaN and must keep its \
                         walked value {}, not {}",
                        clean[j],
                        preds[j]
                    );
                    assert_eq!(states[j], clean_states[j], "{name} {label}: states {j}");
                } else {
                    assert!(
                        preds[j].is_nan() && states[j].iter().all(|s| s.is_nan()),
                        "{name} {label}: record {j} is at or after the NaN and must be NaN, \
                         not {} / {:?}",
                        preds[j],
                        states[j]
                    );
                }
            }
        }
    }
}

/// #2134: with `CL` `NaN` on every record, the walk and the static closed form
/// (`compute_predictions`) both serve `NaN` at every observation, so whether a subject's
/// unusable covariate is reported no longer depends on whether it happens to vary over
/// time — the absorption-model row of #2039's event-driven parity. Both used to serve 0.
///
/// Mutation (run): restore `return T::from_f64(0.0)` for a `NaN` `ke` in
/// `one_cpt_transit_amt_g` → the `OneCptTransit` superposition premise dies.
#[test]
fn walk_and_static_closed_form_agree_on_a_constant_nan_clearance() {
    for (model, subject, _) in nan_cases() {
        let name = format!("{:?}", model.pk_model);
        let theta = model.default_params.theta.clone();
        let mut pk = (model.pk_param_fn)(
            &theta,
            &[ETA_V, 0.0, 0.0, 0.0, 0.0],
            &subject.covariates,
            0.0,
        );
        pk.values[crate::types::PK_IDX_CL] = f64::NAN;
        let superposition = crate::pk::compute_predictions(model.pk_model, &subject, &pk);
        assert!(
            superposition.iter().all(|p| p.is_nan()),
            "{name}: the static closed form must serve NaN: {superposition:?}"
        );
        let walk = absorption_walk_predictions(
            model.pk_model,
            &subject,
            &vec![pk; subject.doses.len()],
            &vec![pk; subject.obs_times.len()],
            &[],
            None,
        )
        .expect("the walk serves a NaN");
        assert!(
            walk.iter().all(|p| p.is_nan()),
            "{name}: the walk must serve the NaN the static closed form does: {walk:?}"
        );
    }
}

/// #2134 review: a `NaN` disposition on a record that governs no elapsed time — here the
/// first dose record, whose interval has zero length — never acts, so it must poison
/// nothing. The event-driven and ODE engines advance to the next record on that record's
/// own (finite) snapshot; the walk once set its poison flag from any record's snapshot
/// and served the whole profile `NaN`. Every prediction and state must equal the clean
/// walk bit for bit.
///
/// Mutation (run): check `disposition_is_nan(pk_now)` before the `ev.time > cur_t` test
/// → every row dies with an all-`NaN` profile.
#[test]
fn a_nan_disposition_on_a_record_governing_no_interval_poisons_nothing() {
    use crate::types::{PK_IDX_CL, PK_IDX_V, PK_IDX_V2};
    for (model, subject, _) in nan_cases() {
        let name = format!("{:?}", model.pk_model);
        let theta = model.default_params.theta.clone();
        let pk = (model.pk_param_fn)(
            &theta,
            &[ETA_V, 0.0, 0.0, 0.0, 0.0],
            &subject.covariates,
            0.0,
        );
        assert!(
            subject.doses[0].time < subject.obs_times[0],
            "{name}: premise — the first dose record opens the timeline"
        );
        let dose = vec![pk; subject.doses.len()];
        let obs = vec![pk; subject.obs_times.len()];
        let mut clean_states = Vec::new();
        let clean = absorption_walk_predictions(
            model.pk_model,
            &subject,
            &dose,
            &obs,
            &[],
            Some(&mut clean_states),
        )
        .expect("inside the domain");
        let two_cpt = matches!(model.pk_model, PkModel::TwoCptTransit | PkModel::TwoCptIg);
        let mut slots = vec![("CL", PK_IDX_CL), ("V", PK_IDX_V)];
        if two_cpt {
            slots.push(("V2", PK_IDX_V2));
        }
        for (label, idx) in slots {
            let mut at_dose = dose.clone();
            at_dose[0].values[idx] = f64::NAN;
            let mut states = Vec::new();
            let preds = absorption_walk_predictions(
                model.pk_model,
                &subject,
                &at_dose,
                &obs,
                &[],
                Some(&mut states),
            )
            .unwrap_or_else(|| panic!("{name} {label}: the walk serves this subject"));
            for (j, (p, c)) in preds.iter().zip(&clean).enumerate() {
                assert!(
                    *c > 0.0 && p.to_bits() == c.to_bits(),
                    "{name} {label} = NaN on the first dose record only: record {j} must keep \
                     {c}, not {p}"
                );
                assert_eq!(states[j], clean_states[j], "{name} {label}: states {j}");
            }
        }
    }
}
