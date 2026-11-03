//! #2125: an adaptive decision placed before the subject's first data record.
//!
//! Nothing evolves before a subject's first record, so a decision there that issues no
//! dose leaves the state at `init` until that record (NONMEM deposits `A_0` there, and the
//! reactive driver holds it, #936). Both frozen-schedule replays used to add every decision
//! as a break and start integrating at the earliest one, so a run with a decision at 0.5 and
//! a first record at 1 was refused: "prediction 0 diverges … reactive=14,
//! static=13.317…", one segment of decay `14·e^{−0.1·0.5}`. Measured at `aa5f3b6e` on both
//! replays — the constant-covariate one (`ode_predictions_with_extra_breaks`) and the
//! time-varying one (`adaptive_frozen_replay_tv`) — and on a subject whose first record is a
//! base dose.
//!
//! The reference is NONMEM (`nonmem_anchor/adaptive_prestart_decision.ctl`): the dataset is
//! each subject's **realized** regimen, the controller's doses written as dose rows. A
//! controller dose has no row of its own on the ferx side, so each such row carries the
//! first record's WT and NONMEM seeds `A_0` from the same value ferx does (#2128). The ferx
//! base dataset is the same file with those rows removed, and the controller re-emits them.
//!
//! The replay verifier is on in every run, so a replay that integrates from a pre-start
//! decision fails the run. It cannot see the converse: a driver that integrated from the
//! decision too would agree with the replay and pass. That half is held by NONMEM's values.
//!
//! Tier 2: one no-op or re-emitting adaptive run and one `predict()` per subject.

use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::{
    parse_full_model_file, predict, read_nonmem_csv, simulate_adaptive, AdaptiveSimulateOptions,
    CompiledModel, Population,
};
use std::path::{Path, PathBuf};

const ANCHOR: &str = "adaptive_prestart_decision";

/// Each subject's decision schedule and the decisions at which the controller doses 100 into
/// cmt 1. A dosing decision's time is a dose row in the NONMEM dataset; every other decision
/// is a hold. ID 26's dose at 1 is a base dose (a data record), not a controller one.
struct Schedule {
    id: &'static str,
    decisions: &'static [f64],
    doses: &'static [f64],
}

/// Constant WT: the constant-covariate replay.
const CONSTANT: &[Schedule] = &[
    // Two holds before the first record.
    Schedule {
        id: "21",
        decisions: &[0.25, 0.5],
        doses: &[],
    },
    // A hold before the first record, a dose after it with residual drug present.
    Schedule {
        id: "22",
        decisions: &[0.5, 2.0],
        doses: &[2.0],
    },
    // A hold, then a dose, both before the first record (ID 24's constant twin, #2630 r1).
    Schedule {
        id: "28",
        decisions: &[0.25, 0.5],
        doses: &[0.5],
    },
];

/// WT 70 → 140 within the first instant: the time-varying replay.
const TIME_VARYING: &[Schedule] = &[
    // A dose at a decision before the first record, then a second with residual drug.
    Schedule {
        id: "23",
        decisions: &[0.5, 2.0],
        doses: &[0.5, 2.0],
    },
    // A hold, then a dose, both before the first record.
    Schedule {
        id: "24",
        decisions: &[0.25, 0.5],
        doses: &[0.5],
    },
    // A hold before the first record, a dose after it.
    Schedule {
        id: "25",
        decisions: &[0.5, 2.0],
        doses: &[2.0],
    },
    // The first record is a base dose; a hold before it.
    Schedule {
        id: "26",
        decisions: &[0.5],
        doses: &[],
    },
    // #2125's own subject (`init_seed_tie` ID 11 with an observation at 4).
    Schedule {
        id: "27",
        decisions: &[0.5],
        doses: &[],
    },
];

// Measured worst |ferx − NONMEM| on macOS arm64: 3.1e-8 (constant, 12 observations) and
// 4.5e-8 (time-varying, 19), so 22× headroom. NONMEM's `.tab` prints 11 significant figures
// at values ≈ 15; the rest is the two ODE solves. The smallest defect guarded is one segment
// of decay, 0.33 (ID 24).
const IPRED_BOUND: f64 = 1e-6;

fn model() -> CompiledModel {
    parse_full_model_file(Path::new("nonmem_anchor/init_seed_tie_fit.ferx"))
        .expect("init_seed_tie_fit")
        .model
}

/// `(ID, TIME, IPRED)` of every observation row in NONMEM's table.
fn nonmem_ipred() -> Vec<(String, f64, f64)> {
    let path = format!("nonmem_anchor/results/{ANCHOR}.tab");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |c: &str| header.iter().position(|h| *h == c).expect(c);
    let (id, time, evid, ipred) = (col("ID"), col("TIME"), col("EVID"), col("IPRED"));
    lines
        .map(|l| {
            l.split_whitespace()
                .map(|v| v.parse::<f64>().unwrap_or_else(|e| panic!("{v}: {e}")))
                .collect::<Vec<f64>>()
        })
        .filter(|r| r[evid] == 0.0)
        .map(|r| (format!("{}", r[id] as i64), r[time], r[ipred]))
        .collect()
}

/// The anchor dataset with every controller dose row removed — the subjects the adaptive
/// runs start from. Written per process so parallel test binaries do not share a file.
fn base_population() -> Population {
    let src = format!("nonmem_anchor/{ANCHOR}.csv");
    let text = std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let mut lines = text.lines();
    let header = lines.next().expect("header");
    let mut out = vec![header.to_string()];
    let mut removed = 0;
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        let (id, time, evid) = (f[0], f[1].parse::<f64>().unwrap(), f[4]);
        let controller = evid == "1"
            && CONSTANT
                .iter()
                .chain(TIME_VARYING)
                .any(|s| s.id == id && s.doses.contains(&time));
        if controller {
            removed += 1;
        } else {
            out.push(l.to_string());
        }
    }
    assert_eq!(removed, 6, "controller dose rows in the anchor dataset");
    let path: PathBuf =
        std::env::temp_dir().join(format!("ferx_2125_base_{}.csv", std::process::id()));
    std::fs::write(&path, out.join("\n") + "\n").expect("write base csv");
    let pop = read_nonmem_csv(&path, None, None).expect("base csv");
    let _ = std::fs::remove_file(&path);
    pop
}

/// Run each schedule on its own subject with the replay verifier on, the controller
/// re-emitting the schedule's doses. Returns `(ID, TIME, IPRED)` per observation.
fn adaptive(
    m: &CompiledModel,
    pop: &Population,
    schedules: &[Schedule],
) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();
    for s in schedules {
        let subject = pop
            .subjects
            .iter()
            .find(|x| x.id == s.id)
            .unwrap_or_else(|| panic!("ID {} in the base dataset", s.id));
        let first_record = subject
            .obs_times
            .iter()
            .chain(subject.doses.iter().map(|d| &d.time))
            .cloned()
            .fold(f64::INFINITY, f64::min);
        // The straddle this file exists for: every schedule's first decision precedes the
        // subject's first record, or the fixture tests nothing.
        assert!(
            s.decisions[0] < first_record,
            "ID {}: first decision {} is not before the first record {first_record}",
            s.id,
            s.decisions[0]
        );
        let one = Population {
            subjects: vec![subject.clone()],
            ..pop.clone()
        };
        let mut opts = AdaptiveSimulateOptions::default();
        opts.seed = Some(1);
        opts.decision_times = s.decisions.to_vec();
        opts.verify = true;
        let doses = s.doses;
        let res = simulate_adaptive(
            m,
            &one,
            &m.default_params,
            1,
            move || {
                move |c: &ControllerCtx| -> Vec<DoseAction> {
                    if doses.contains(&c.t) {
                        vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
                    } else {
                        Vec::new()
                    }
                }
            },
            &opts,
        )
        .unwrap_or_else(|e| panic!("ID {}: adaptive run refused: {e}", s.id));
        assert_eq!(
            res.ledger.len(),
            s.doses.len(),
            "ID {}: realized doses",
            s.id
        );
        out.extend(
            res.trajectories
                .into_iter()
                .map(|r| (r.id, r.time, r.ipred)),
        );
    }
    out
}

/// Compare `got` with NONMEM's rows for the same subjects, row for row. Each value is
/// checked finite before it is compared (`f64::max` drops a NaN).
fn assert_matches_nonmem(leg: &str, got: &[(String, f64, f64)], schedules: &[Schedule]) {
    let nm: Vec<_> = nonmem_ipred()
        .into_iter()
        .filter(|r| schedules.iter().any(|s| s.id == r.0))
        .collect();
    assert_eq!(got.len(), nm.len(), "{leg}: observation count");
    let mut worst = 0.0_f64;
    for ((id, t, v), (nid, nt, want)) in got.iter().zip(&nm) {
        assert!(
            id == nid && (t - nt).abs() < 1e-9,
            "{leg}: ferx row ({id}, {t}) is NONMEM's ({nid}, {nt})"
        );
        assert!(v.is_finite(), "{leg} ID {id} TIME {t}: {v}");
        let err = (v - want).abs();
        assert!(
            err < IPRED_BOUND,
            "{leg} ID {id} TIME {t}: ferx {v} vs NONMEM {want} (|error| {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!(
        "{leg}: worst |ferx − NONMEM| {worst:.3e} over {} observations",
        nm.len()
    );
}

/// The constant-covariate replay (`ode_predictions_with_extra_breaks`), IDs 21, 22 and 28.
#[test]
fn constant_path_decision_before_the_first_record_verifies_and_matches_nonmem() {
    let m = model();
    let got = adaptive(&m, &base_population(), CONSTANT);
    assert_matches_nonmem("adaptive, constant WT", &got, CONSTANT);
}

/// The time-varying replay (`adaptive_frozen_replay_tv`), IDs 23–27.
#[test]
fn time_varying_path_decision_before_the_first_record_verifies_and_matches_nonmem() {
    let m = model();
    let got = adaptive(&m, &base_population(), TIME_VARYING);
    assert_matches_nonmem("adaptive, time-varying WT", &got, TIME_VARYING);
}

/// The degenerate oracle: re-emitting a fixed regimen must equal `predict()` on that regimen
/// written as data, bit for bit (measured: every one of the 31 observations is bit-identical).
/// `predict()` reads the full anchor dataset, the controller's doses as rows.
#[test]
fn re_emitting_the_regimen_equals_predict_on_it_as_data() {
    let m = model();
    let full = read_nonmem_csv(
        Path::new(&format!("nonmem_anchor/{ANCHOR}.csv")),
        None,
        None,
    )
    .expect("anchor csv");
    let want: Vec<(String, f64, f64)> = predict(&m, &full, &m.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect();
    let all: Vec<&Schedule> = CONSTANT.iter().chain(TIME_VARYING).collect();
    let base = base_population();
    let mut n = 0;
    for s in all {
        let got = adaptive(&m, &base, std::slice::from_ref(s));
        let w: Vec<_> = want.iter().filter(|r| r.0 == s.id).collect();
        assert_eq!(got.len(), w.len(), "ID {}: observation count", s.id);
        for ((_, t, v), (_, wt, wv)) in got.iter().zip(w) {
            assert!((t - wt).abs() < 1e-9, "ID {}: time {t} vs {wt}", s.id);
            assert!(v.is_finite(), "ID {} TIME {t}: {v}", s.id);
            assert_eq!(
                v.to_bits(),
                wv.to_bits(),
                "ID {} TIME {t}: adaptive {v} vs predict {wv}",
                s.id
            );
            n += 1;
        }
    }
    assert_eq!(n, 31, "observations compared");
}
