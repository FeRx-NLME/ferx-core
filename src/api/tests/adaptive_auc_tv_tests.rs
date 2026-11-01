//! `auc_target` under a time-varying covariate, `TIME` in the PK and IOV (#943, PR 2).
//!
//! The window AUC is integrated on the reactive driver's own walk, each segment under the
//! snapshot the state was integrated with: the PK, κ and covariate row of the record that
//! governs it (the record that terminates it, #1073), and the `[scaling]` divisor under
//! that κ at the subject's t = 0 covariates (the divisor `simulate()` uses, #2129).
//!
//! Validation follows the adaptive-dosing exception in AGENTS.md (no NONMEM): exact closed
//! forms for a 1-cpt bolus model, whose elimination rate and readout factor are constant on
//! each segment, so `∫ factor·A(t) dt` is a sum of exponentials. The mrgsolve AUC24
//! references of the vancomycin kits are the external check (`tests/adaptive_vanco_*`).
//!
//! The public API reports only how many windows fall in the `auc_target` band, so
//! [`assert_window_aucs`] runs once per window with a band around that window's closed
//! form, after checking the bands are pairwise disjoint (the #943 PR 1 vanco-anchor idiom):
//! a count of one then identifies the window.
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, Comparison, DoseStep,
};

/// Relative band of a window AUC against its closed form — the driver's own band
/// (`AUC_TOL` in `ode/predictions_tests.rs`). Measured on Linux aarch64
/// (`tools/linux-test.sh`) across every closed form in this file and the two
/// `adaptive_auc_target_follows_*` tests, identical in every printed digit on macOS arm64:
/// every window reads high, by 2.3e-6 (the EVID=2 tie) to 5.1e-5 (`TIME`, records on the
/// edges), so 2e-4 is 3.9× headroom. (The IOV implied-dose fixture, added after that run,
/// reads 1.6e-5..3.6e-5 on macOS.) Every alternative convention these tests reject is
/// asserted more than 10× the band away ([`assert_distinct`]).
pub(super) const AUC_TOL: f64 = 2e-4;

/// One segment of a closed form: on `(t0, t1]` the amount decays at `k` and the signal is
/// `factor · A`.
#[derive(Clone, Copy)]
pub(super) struct Seg {
    pub t0: f64,
    pub t1: f64,
    pub k: f64,
    pub factor: f64,
}

/// The exact window AUCs of `factor · A(t)` for a 1-cpt bolus model over `segs` (contiguous,
/// covering the decisions), with `boluses` `(time, amount)` and `resets` applied at the start
/// of the segment beginning at their time — reset first (Reset < Dose). A dose at a decision
/// opens that decision's window.
pub(super) fn exact_window_aucs(
    segs: &[Seg],
    boluses: &[(f64, f64)],
    resets: &[f64],
    decisions: &[f64],
) -> Vec<f64> {
    for &(t, _) in boluses {
        assert!(
            segs.iter().any(|s| s.t0 == t) || t >= decisions[decisions.len() - 1],
            "bolus at {t} is not a segment start"
        );
    }
    for &t in resets {
        assert!(
            segs.iter().any(|s| s.t0 == t),
            "reset at {t} is not a segment start"
        );
    }
    for w in segs.windows(2) {
        assert_eq!(w[0].t1, w[1].t0, "segments are contiguous");
    }
    let mut a = 0.0;
    let mut out = vec![0.0; decisions.len() - 1];
    for s in segs {
        if resets.contains(&s.t0) {
            a = 0.0;
        }
        a += boluses
            .iter()
            .filter(|b| b.0 == s.t0)
            .map(|b| b.1)
            .sum::<f64>();
        let w = decisions.partition_point(|&d| d <= s.t0) - 1;
        let decay = (-s.k * (s.t1 - s.t0)).exp();
        out[w] += s.factor * a / s.k * (1.0 - decay);
        a *= decay;
    }
    out
}

/// A declarative block that re-issues `dose` at every decision of `at` (the bounds pin it
/// and the one rule never fires), reading `observe` — or, with `None`, the assay endpoint
/// CMT 1 (`with_assay_error`), whose readout carries the `[scaling]` divisor.
pub(super) fn fixed_dose_spec(observe: Option<&str>, at: &[f64], dose: f64) -> AdaptiveDosingSpec {
    AdaptiveDosingSpec {
        observe: observe.map(str::to_string),
        observe_declared_covariates: Vec::new(),
        with_assay_error: observe.is_none(),
        assay_cmt: observe.is_none().then_some(1),
        at: at.to_vec(),
        start_dose: dose,
        route: AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: (dose, dose),
        confirm: 1,
        levels: None,
        target_window: None,
        auc_target: None,
        rules: vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: -1e9,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
        }],
    }
}

/// How many realized windows fall in `band`.
fn windows_in(
    m: &CompiledModel,
    pop: &Population,
    spec: &AdaptiveDosingSpec,
    seed: u64,
    band: (f64, f64),
) -> usize {
    let mut spec = spec.clone();
    spec.auc_target = Some(band);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(m, pop, &m.default_params, 1, &spec, &opts)
        .expect("adaptive run with auc_target");
    let n_windows = res.decisions.len() - 1;
    let attain = res.metrics[0]
        .auc_target_attainment
        .expect("auc_target declared");
    let n = attain * n_windows as f64;
    assert!(
        (n - n.round()).abs() < 1e-9,
        "attainment {attain} is k/{n_windows}"
    );
    n.round() as usize
}

/// Every realized window AUC of the run matches `want[i]` to `tol` relative. The bands are
/// first checked pairwise disjoint, so a count of one in band `i` is window `i`.
pub(super) fn assert_window_aucs(
    m: &CompiledModel,
    pop: &Population,
    spec: &AdaptiveDosingSpec,
    seed: u64,
    want: &[f64],
    tol: f64,
    label: &str,
) {
    let band_of = |r: f64| (r * (1.0 - tol), r * (1.0 + tol));
    for (i, &a) in want.iter().enumerate() {
        assert!(a.is_finite() && a > 0.0, "{label}: closed form {i} is {a}");
        for (j, &b) in want.iter().enumerate().skip(i + 1) {
            let (ba, bb) = (band_of(a), band_of(b));
            assert!(
                ba.1 < bb.0 || bb.1 < ba.0,
                "{label}: windows {i} and {j} ({a}, {b}) are closer than the band ({tol:e}), \
                 so a count of 1 no longer identifies the window"
            );
        }
    }
    assert_eq!(
        windows_in(m, pop, spec, seed, (0.0, f64::MAX)),
        want.len(),
        "{label}: number of realized windows"
    );
    for (i, &r) in want.iter().enumerate() {
        assert_eq!(
            windows_in(m, pop, spec, seed, band_of(r)),
            1,
            "{label}: window {i} is not within {tol:e} of its closed form {r}"
        );
    }
}

/// `want` is outside every window's `tol` band around `got` — a non-degeneracy check: the
/// alternative convention is far enough from the right one for the fixture to tell them
/// apart.
pub(super) fn assert_distinct(got: &[f64], alt: &[f64], tol: f64, label: &str) {
    let worst = got
        .iter()
        .zip(alt)
        .map(|(g, a)| ((a - g) / g).abs())
        .fold(0.0, f64::max);
    assert!(
        got.iter().chain(alt).all(|x| x.is_finite()),
        "{label}: non-finite closed form"
    );
    assert!(
        worst > 10.0 * tol,
        "{label}: the alternative is only {worst:e} from the closed form, inside 10× the band"
    );
}

/// The one subject of `csv`, read with covariates `covs`.
pub(super) fn read_pop(csv: &str, covs: &[&str]) -> Population {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), (!covs.is_empty()).then_some(covs), None)
        .expect("read the fixture")
}

// ── Time-varying covariates ─────────────────────────────────────────────────────────────

/// CL follows `CRCL`, V follows `WT`, and the monitored signal names a third covariate
/// directly: the unbound concentration `FU · central / V`. η is declared but unread, so the
/// run is deterministic.
const MODEL_TV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * CRCL / 100.0
  V  = TVV * WT / 70.0
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

const TV_OBSERVE: &str = "FU * central / V";
const TV_COVS: [&str; 3] = ["CRCL", "WT", "FU"];
const DECISIONS: [f64; 3] = [0.0, 24.0, 48.0];

/// The closed-form segment of `MODEL_TV` on `(t0, t1]` under the row `(CRCL, WT, FU)`.
fn tv_seg(t0: f64, t1: f64, (crcl, wt, fu): (f64, f64, f64)) -> Seg {
    let (cl, v) = (5.0 * crcl / 100.0, 50.0 * wt / 70.0);
    Seg {
        t0,
        t1,
        k: cl / v,
        factor: fu / v,
    }
}

/// `MODEL_TV` with a 100 bolus at each decision; `rows` are `(TIME, EVID, AMT, II, ADDL,
/// CRCL, WT, FU)`.
fn tv_csv(rows: &[(f64, u8, f64, f64, u32, f64, f64, f64)]) -> String {
    let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,II,ADDL,CRCL,WT,FU\n");
    for &(t, evid, amt, ii, addl, crcl, wt, fu) in rows {
        let mdv = u8::from(evid != 0);
        s.push_str(&format!(
            "1,{t},0,{evid},{amt},1,{mdv},{ii},{addl},{crcl},{wt},{fu}\n"
        ));
    }
    s
}

const R0: (f64, f64, f64) = (100.0, 70.0, 0.5);
const R12: (f64, f64, f64) = (60.0, 50.0, 0.3);
const R24: (f64, f64, f64) = (90.0, 80.0, 0.6);
const R36: (f64, f64, f64) = (50.0, 60.0, 0.2);
const R48: (f64, f64, f64) = (80.0, 90.0, 0.4);

fn obs_row(t: f64, (crcl, wt, fu): (f64, f64, f64)) -> (f64, u8, f64, f64, u32, f64, f64, f64) {
    (t, 0, 0.0, 0.0, 0, crcl, wt, fu)
}

fn run_tv(csv: &str, want: &[f64], label: &str) {
    let m = parse_model_string(MODEL_TV).expect("parse");
    let pop = read_pop(csv, &TV_COVS);
    let spec = fixed_dose_spec(Some(TV_OBSERVE), &DECISIONS, 100.0);
    assert_window_aucs(&m, &pop, &spec, 1, want, AUC_TOL, label);
}

/// A covariate that changes **inside** a window (#943): the records at 12 and 36 split each
/// window into two segments, each under the row of the record that terminates it — CL, V
/// and the `FU` the signal names all step there. The straddle control moves the change onto
/// the window edges (records only at 0, 24, 48): each window is then one segment, and its
/// AUC moves by 30% or more, so the interior record is what the first fixture measures.
///
/// A third geometry ends the records at 36, before the last decision: the trailing segment
/// has no governing record and reads the row in force at its end.
///
/// Mutations, each killing this test (#943 PR 2): read the grid points under the t = 0 PK
/// (`pk_params_flat`, `V = 50`) instead of `seg_pk`; read them under the subject's static
/// covariates (`FU = 0.5`) instead of the governing record's row; past the final record,
/// read the baseline row instead of the one in force.
#[test]
fn auc_target_follows_a_covariate_that_changes_inside_a_window() {
    let rows = [
        obs_row(0.0, R0),
        obs_row(12.0, R12),
        obs_row(24.0, R24),
        obs_row(36.0, R36),
        obs_row(48.0, R48),
    ];
    let boluses = [(0.0, 100.0), (24.0, 100.0)];
    let inside = exact_window_aucs(
        &[
            tv_seg(0.0, 12.0, R12),
            tv_seg(12.0, 24.0, R24),
            tv_seg(24.0, 36.0, R36),
            tv_seg(36.0, 48.0, R48),
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    run_tv(&tv_csv(&rows), &inside, "covariate inside the window");

    // Straddle control: the same rows at the edges only.
    let edges = [obs_row(0.0, R0), obs_row(24.0, R24), obs_row(48.0, R48)];
    let at_edges = exact_window_aucs(
        &[tv_seg(0.0, 24.0, R24), tv_seg(24.0, 48.0, R48)],
        &boluses,
        &[],
        &DECISIONS,
    );
    run_tv(&tv_csv(&edges), &at_edges, "covariate on the window edges");
    assert_distinct(&inside, &at_edges, AUC_TOL, "interior record vs edges only");

    // Records stop at 36, before the last decision: `(36, 48]` has no record ahead, so it
    // runs under the last record that ran (the PK carry) and reads the row in force there —
    // the record at 36, not the subject's baseline row.
    let short = [
        obs_row(0.0, R0),
        obs_row(12.0, R12),
        obs_row(24.0, R24),
        obs_row(36.0, R36),
    ];
    let trailing = |last| {
        exact_window_aucs(
            &[
                tv_seg(0.0, 12.0, R12),
                tv_seg(12.0, 24.0, R24),
                tv_seg(24.0, 36.0, R36),
                tv_seg(36.0, 48.0, last),
            ],
            &boluses,
            &[],
            &DECISIONS,
        )
    };
    let past_last = trailing(R36);
    run_tv(&tv_csv(&short), &past_last, "past the final record");
    // The baseline row's FU under the carried PK would move window 1.
    let (cl, v) = (5.0 * R36.0 / 100.0, 50.0 * R36.1 / 70.0);
    let baseline_fu = exact_window_aucs(
        &[
            tv_seg(0.0, 12.0, R12),
            tv_seg(12.0, 24.0, R24),
            tv_seg(24.0, 36.0, R36),
            Seg {
                t0: 36.0,
                t1: 48.0,
                k: cl / v,
                factor: R0.2 / v,
            },
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    assert_distinct(
        &past_last,
        &baseline_fu,
        AUC_TOL,
        "baseline row past the final record",
    );

    // The alternatives the mutations produce are far from the closed form.
    let static_fu: Vec<Seg> = [
        tv_seg(0.0, 12.0, R12),
        tv_seg(12.0, 24.0, R24),
        tv_seg(24.0, 36.0, R36),
        tv_seg(36.0, 48.0, R48),
    ]
    .iter()
    .map(|s| Seg {
        factor: s.factor / [R12, R24, R36, R48][(s.t0 / 12.0) as usize].2 * R0.2,
        ..*s
    })
    .collect();
    let static_fu = exact_window_aucs(&static_fu, &boluses, &[], &DECISIONS);
    assert_distinct(&inside, &static_fu, AUC_TOL, "static FU at the grid points");
}

// ── TIME in the PK ──────────────────────────────────────────────────────────────────────

/// CL and V read the `TIME` built-in, which `$PK` evaluates at each record.
const MODEL_TIME: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * (1.0 + 0.02 * TIME)
  V  = TVV * (1.0 + 0.01 * TIME)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

fn time_seg(t0: f64, t1: f64) -> Seg {
    // The record at `t1` governs `(t0, t1]`, so `$PK` ran at `TIME = t1`.
    let (cl, v) = (5.0 * (1.0 + 0.02 * t1), 50.0 * (1.0 + 0.01 * t1));
    Seg {
        t0,
        t1,
        k: cl / v,
        factor: 1.0 / v,
    }
}

/// `TIME` in the PK (#943): each segment runs under `$PK` evaluated at the record that
/// terminates it, and the concentration reads that record's `V`. Records at 0, 12, …, 48
/// put a PK change inside each window; the straddle control (records at the edges only)
/// makes each window one segment and moves the AUC.
///
/// Mutation: read the grid points under the t = 0 PK (`V = 50` throughout) — this test
/// dies.
#[test]
fn auc_target_follows_time_in_the_pk() {
    let m = parse_model_string(MODEL_TIME).expect("parse");
    let spec = fixed_dose_spec(Some("central / V"), &DECISIONS, 100.0);
    let boluses = [(0.0, 100.0), (24.0, 100.0)];
    let csv = |times: &[f64]| {
        let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV\n");
        for t in times {
            s.push_str(&format!("1,{t},0,0,0,1,0\n"));
        }
        s
    };

    let inside = exact_window_aucs(
        &[
            time_seg(0.0, 12.0),
            time_seg(12.0, 24.0),
            time_seg(24.0, 36.0),
            time_seg(36.0, 48.0),
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    let pop = read_pop(&csv(&[0.0, 12.0, 24.0, 36.0, 48.0]), &[]);
    assert_window_aucs(&m, &pop, &spec, 1, &inside, AUC_TOL, "TIME, records inside");

    let edges = exact_window_aucs(
        &[time_seg(0.0, 24.0), time_seg(24.0, 48.0)],
        &boluses,
        &[],
        &DECISIONS,
    );
    let pop = read_pop(&csv(&[0.0, 24.0, 48.0]), &[]);
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        1,
        &edges,
        AUC_TOL,
        "TIME, records on the edges",
    );
    assert_distinct(
        &inside,
        &edges,
        AUC_TOL,
        "TIME: interior record vs edges only",
    );
}

// ── IOV ─────────────────────────────────────────────────────────────────────────────────

/// κ on V, and Form B `[scaling] obs_scale = V`: the observe-less assay monitor reads
/// `central / V(κ)`, with V both in the ODE (through `seg_pk`) and in the divisor (through
/// `seg_eta`). CL carries no κ, so `k = CL / V(κ)`.
const MODEL_IOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_V ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV * exp(KAPPA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP)
"#;

const IOV_SEED: u64 = 20260708;
const IOV_DECISIONS: [f64; 4] = [0.0, 24.0, 48.0, 72.0];

/// The per-decision-window κ the run draws (κ_g = chol(Ω_IOV)·z, the substream
/// `adaptive_iov_matches_predict_iov_with_reconstructed_kappa` pins).
pub(super) fn reconstructed_kappa(m: &CompiledModel, seed: u64, n: usize) -> Vec<f64> {
    let chol = &m.default_params.omega_iov.as_ref().expect("omega_iov").chol;
    let base = crate::sim::adaptive::subject_kappa_base_seed(seed, "1", 1);
    (0..n)
        .map(|g| chol[(0, 0)] * crate::sim::adaptive::kappa_standard_normal(base, g, 0))
        .collect()
}

fn iov_seg(t0: f64, t1: f64, kappa: f64) -> Seg {
    let v = 50.0 * kappa.exp();
    Seg {
        t0,
        t1,
        k: 5.0 / v,
        factor: 1.0 / v,
    }
}

fn iov_csv(resets: &[f64]) -> String {
    let mut rows: Vec<(f64, String)> = IOV_DECISIONS
        .iter()
        .map(|&t| (t, format!("1,{t},0,0,0,1,0\n")))
        .collect();
    rows.extend(resets.iter().map(|&t| (t, format!("1,{t},0,3,0,1,1\n"))));
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    std::iter::once("ID,TIME,DV,EVID,AMT,CMT,MDV\n".to_string())
        .chain(rows.into_iter().map(|r| r.1))
        .collect()
}

/// IOV with κ differing across adjacent windows (#943). The only records sit at the
/// decisions, so each window `(d_w, d_{w+1}]` is governed by the observation at `d_{w+1}`
/// and runs — state, readout `V` and `[scaling]` divisor alike — under κ_{w+1}, the next
/// window's (the mrgsolve kits' "day g on occasion g+1").
///
/// Mutations, each killing this test (#943 PR 2): divide the grid points by the κ = 0
/// divisor (the run's baseline η instead of `seg_eta`); read them under the decision
/// window's κ_w (`occasion_of(t)`) instead of the governing record's. Both alternatives are
/// asserted far from the closed form below, so neither can pass by coincidence.
#[test]
fn auc_target_reads_each_window_under_its_segments_kappa() {
    let m = parse_model_string(MODEL_IOV).expect("parse");
    let kappa = reconstructed_kappa(&m, IOV_SEED, IOV_DECISIONS.len());
    for w in 0..kappa.len() - 1 {
        assert!(
            (kappa[w + 1] - kappa[w]).abs() > 0.05,
            "κ must differ across adjacent windows for the fixture to see it: {kappa:?}"
        );
    }
    let spec = fixed_dose_spec(None, &IOV_DECISIONS, 100.0);
    let boluses = [(0.0, 100.0), (24.0, 100.0), (48.0, 100.0)];
    let segs: Vec<Seg> = (0..3)
        .map(|w| iov_seg(IOV_DECISIONS[w], IOV_DECISIONS[w + 1], kappa[w + 1]))
        .collect();
    let want = exact_window_aucs(&segs, &boluses, &[], &IOV_DECISIONS);
    let pop = read_pop(&iov_csv(&[]), &[]);
    assert_window_aucs(&m, &pop, &spec, IOV_SEED, &want, AUC_TOL, "IOV");

    // The κ = 0 divisor: the state still integrated under κ_{w+1}, divided by V(0).
    let zero_div: Vec<Seg> = segs
        .iter()
        .map(|s| Seg {
            factor: 1.0 / 50.0,
            ..*s
        })
        .collect();
    let zero_div = exact_window_aucs(&zero_div, &boluses, &[], &IOV_DECISIONS);
    assert_distinct(&want, &zero_div, AUC_TOL, "κ = 0 divisor");
    // The decision window's κ_w in the readout and divisor (state under κ_{w+1}).
    let window_kappa: Vec<Seg> = segs
        .iter()
        .zip(&kappa)
        .map(|(s, &kw)| Seg {
            factor: 1.0 / (50.0 * kw.exp()),
            ..*s
        })
        .collect();
    let window_kappa = exact_window_aucs(&window_kappa, &boluses, &[], &IOV_DECISIONS);
    assert_distinct(&want, &window_kappa, AUC_TOL, "decision-window κ");
}

/// An occasion carried only by a reset row has its own κ (#1153). A reset at 12 is the only
/// record inside window 0, so it governs `(0, 12]` under κ_0 — not the observation at 24
/// (κ_1) — and the state is zero after it until the next decision. The straddle control
/// moves the reset past the decision at 24: window 0 is then wholly κ_1, and window 1's
/// pre-reset segment `(24, 30]` runs under the reset's κ_1 instead of the observation at
/// 48's κ_2.
#[test]
fn auc_target_reads_a_reset_rows_own_kappa() {
    let m = parse_model_string(MODEL_IOV).expect("parse");
    let kappa = reconstructed_kappa(&m, IOV_SEED, IOV_DECISIONS.len());
    let spec = fixed_dose_spec(None, &IOV_DECISIONS, 100.0);
    let boluses = [(0.0, 100.0), (24.0, 100.0), (48.0, 100.0)];

    let in_window0 = exact_window_aucs(
        &[
            iov_seg(0.0, 12.0, kappa[0]),
            iov_seg(12.0, 24.0, kappa[1]),
            iov_seg(24.0, 48.0, kappa[2]),
            iov_seg(48.0, 72.0, kappa[3]),
        ],
        &boluses,
        &[12.0],
        &IOV_DECISIONS,
    );
    let pop = read_pop(&iov_csv(&[12.0]), &[]);
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        IOV_SEED,
        &in_window0,
        AUC_TOL,
        "reset at 12",
    );
    // Had the reset taken the next record's κ_1, window 0 would move.
    let next_kappa = exact_window_aucs(
        &[
            iov_seg(0.0, 12.0, kappa[1]),
            iov_seg(12.0, 24.0, kappa[1]),
            iov_seg(24.0, 48.0, kappa[2]),
            iov_seg(48.0, 72.0, kappa[3]),
        ],
        &boluses,
        &[12.0],
        &IOV_DECISIONS,
    );
    assert_distinct(&in_window0, &next_kappa, AUC_TOL, "reset row's own κ");

    let in_window1 = exact_window_aucs(
        &[
            iov_seg(0.0, 24.0, kappa[1]),
            iov_seg(24.0, 30.0, kappa[1]),
            iov_seg(30.0, 48.0, kappa[2]),
            iov_seg(48.0, 72.0, kappa[3]),
        ],
        &boluses,
        &[30.0],
        &IOV_DECISIONS,
    );
    let pop = read_pop(&iov_csv(&[30.0]), &[]);
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        IOV_SEED,
        &in_window1,
        AUC_TOL,
        "reset at 30",
    );
}

// ── Merged-PR geometries at a window boundary ───────────────────────────────────────────

/// An `ADDL`-implied dose inside a window takes its governing record's snapshot (#1909,
/// #2530): the dose at 0 implies one at 12, whose governing record is the observation at 24,
/// so both `(0, 12]` and `(12, 24]` run under the row at 24 — not under the dose row at 0,
/// which an implied dose is not (#2528).
#[test]
fn auc_target_reads_an_implied_dose_under_its_governing_record() {
    let rows = [
        (0.0, 1, 100.0, 12.0, 1, R0.0, R0.1, R0.2),
        obs_row(24.0, R24),
        obs_row(48.0, R48),
    ];
    // Base 100 at 0 and its implied 100 at 12, plus the controller's 100 at each decision.
    let boluses = [(0.0, 200.0), (12.0, 100.0), (24.0, 100.0)];
    let want = exact_window_aucs(
        &[
            tv_seg(0.0, 12.0, R24),
            tv_seg(12.0, 24.0, R24),
            tv_seg(24.0, 48.0, R48),
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    run_tv(&tv_csv(&rows), &want, "ADDL-implied dose");
    let dose_row = exact_window_aucs(
        &[
            tv_seg(0.0, 12.0, R0),
            tv_seg(12.0, 24.0, R24),
            tv_seg(24.0, 48.0, R48),
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    assert_distinct(&want, &dose_row, AUC_TOL, "implied dose read as a record");
}

/// Under IOV an `ADDL`-implied dose takes its governing record's occasion (#2371): the base
/// dose at 12 implies one at 36, inside window 1, whose governing record is the observation
/// at 48 (occasion 2). The segment `(24, 36]` ends at the implied dose, so it runs — and its
/// grid points are divided — under κ_2, not under κ_1 of the window the dose time falls in.
/// The `[scaling]` divisor is the only place that occasion reaches: the RHS cannot name a κ,
/// and the dose's PK snapshot is built separately.
///
/// Mutation: resolve the implied dose's occasion from its own time (`doses[k].time`) in
/// `AdaptiveRecordOcc::new` — this test dies.
#[test]
fn auc_target_reads_an_implied_doses_occasion_under_iov() {
    let m = parse_model_string(MODEL_IOV).expect("parse");
    let kappa = reconstructed_kappa(&m, IOV_SEED, IOV_DECISIONS.len());
    let spec = fixed_dose_spec(None, &IOV_DECISIONS, 100.0);
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,II,ADDL\n");
    for t in [0.0, 12.0, 24.0, 48.0, 72.0] {
        csv.push_str(&if t == 12.0 {
            "1,12,0,1,100,1,1,24,1\n".to_string()
        } else {
            format!("1,{t},0,0,0,1,0,0,0\n")
        });
    }
    let pop = read_pop(&csv, &[]);
    let boluses = [
        (0.0, 100.0),
        (12.0, 100.0),
        (24.0, 100.0),
        (36.0, 100.0),
        (48.0, 100.0),
    ];
    let with = |k_24_36: f64| {
        exact_window_aucs(
            &[
                iov_seg(0.0, 12.0, kappa[0]),
                iov_seg(12.0, 24.0, kappa[1]),
                iov_seg(24.0, 36.0, kappa[2]),
                iov_seg(36.0, 48.0, kappa[2]),
                iov_seg(48.0, 72.0, kappa[3]),
            ]
            .map(|s| {
                if s.t0 == 24.0 {
                    Seg {
                        factor: 1.0 / (50.0 * k_24_36.exp()),
                        ..s
                    }
                } else {
                    s
                }
            }),
            &boluses,
            &[],
            &IOV_DECISIONS,
        )
    };
    let want = with(kappa[2]);
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        IOV_SEED,
        &want,
        AUC_TOL,
        "IOV implied dose",
    );
    assert_distinct(
        &want,
        &with(kappa[1]),
        AUC_TOL,
        "implied dose under its own time's κ",
    );
}

/// Two co-timed records on a window edge (#2485 / #2533 tie order): at 24 an observation and
/// an EVID=2 row with different covariates. The record that comes first terminates
/// `(0, 24]`, so window 0 runs under its row, while window 1 runs under the record at 48.
///
/// "First" is the file order, carried by the reader: listed before the EVID=2 row, the
/// observation is moved one ULP below it (#614) and terminates the segment itself; listed
/// after it, the two share the instant and the processing order (`EVID=2 < Obs`) puts the
/// EVID=2 row first. Each order is the other's control — window 0 moves by about 2× between
/// them.
#[test]
fn auc_target_reads_the_first_of_two_cotimed_records_on_a_window_edge() {
    let evid2 = (40.0, 100.0, 0.9);
    let evid2_row = (24.0, 2, 0.0, 0.0, 0, evid2.0, evid2.1, evid2.2);
    let boluses = [(0.0, 100.0), (24.0, 100.0)];
    let under = |row| {
        exact_window_aucs(
            &[tv_seg(0.0, 24.0, row), tv_seg(24.0, 48.0, R48)],
            &boluses,
            &[],
            &DECISIONS,
        )
    };
    let (obs_first, evid2_first) = (under(R24), under(evid2));
    run_tv(
        &tv_csv(&[
            obs_row(0.0, R0),
            obs_row(24.0, R24),
            evid2_row,
            obs_row(48.0, R48),
        ]),
        &obs_first,
        "observation listed first",
    );
    run_tv(
        &tv_csv(&[
            obs_row(0.0, R0),
            evid2_row,
            obs_row(24.0, R24),
            obs_row(48.0, R48),
        ]),
        &evid2_first,
        "EVID=2 row listed first",
    );
    assert_distinct(&obs_first, &evid2_first, AUC_TOL, "file order of the tie");
}

/// A trough listed before a co-timed base dose inside a window (#2532 / #2533). The reader
/// moves the trough one ULP below the dose, so the observation terminates `(0, 12⁻]` and the
/// window's first segment runs under its row, not the dose row's; the dose then lands and
/// `(12, 24]` runs under the observation at 24.
#[test]
fn auc_target_reads_a_trough_tied_to_a_mid_window_dose() {
    let dose_row = (30.0, 100.0, 0.8);
    let rows = [
        obs_row(0.0, R0),
        obs_row(12.0, R12),
        (12.0, 1, 100.0, 0.0, 0, dose_row.0, dose_row.1, dose_row.2),
        obs_row(24.0, R24),
        obs_row(48.0, R48),
    ];
    let pop = read_pop(&tv_csv(&rows), &TV_COVS);
    assert_eq!(
        pop.subjects[0].obs_times[1],
        12.0f64.next_down(),
        "the reader moved the trough below the dose"
    );
    let boluses = [(0.0, 100.0), (12.0, 100.0), (24.0, 100.0)];
    let want = exact_window_aucs(
        &[
            tv_seg(0.0, 12.0, R12),
            tv_seg(12.0, 24.0, R24),
            tv_seg(24.0, 48.0, R48),
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    run_tv(&tv_csv(&rows), &want, "trough tied to a dose");
    let under_dose_row = exact_window_aucs(
        &[
            tv_seg(0.0, 12.0, dose_row),
            tv_seg(12.0, 24.0, R24),
            tv_seg(24.0, 48.0, R48),
        ],
        &boluses,
        &[],
        &DECISIONS,
    );
    assert_distinct(
        &want,
        &under_dose_row,
        AUC_TOL,
        "dose row governs the trough",
    );
}
