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
        pk_only_covariates: vec![],
        reset_times: vec![],
        reset_covariates: vec![],
        cens: vec![0; n],
        occasions: obs.iter().map(|&(_, o, _)| o).collect(),
        obs_l2: vec![],
        dose_occasions: doses.iter().map(|&(_, o, _)| o).collect(),
        reset_occasions: vec![],
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
