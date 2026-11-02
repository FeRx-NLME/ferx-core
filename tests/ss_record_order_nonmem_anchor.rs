//! #2510: a Gaussian observation listed **before** a same-`TIME` `SS=1` dose reads the
//! amounts carried from the previous record, as NONMEM does. One listed after it reads the
//! steady state.
//!
//! The reader's #614 record-order pass moved an observation listed before a co-timed dose
//! one ULP below it, but it skipped steady-state doses. Its comment said a sub-dose-time
//! sample "would read 0 instead of the periodic pre-arrival trough". NONMEM 7.5.1 settles it
//! the other way (`nonmem_anchor/dtmm_ss_record_order.ctl`, run during #2473 on `5b42209e`):
//!
//! | ID | records at the tie | NONMEM before the SS dose | `main` before #2510 |
//! |---|---|---:|---:|
//! | 1 | first SS dose at `TIME 0` | 0 | 14.3101276 |
//! | 1 | SS dose at 12 on its own prior steady state | 4.3101276 | 14.3101276 |
//! | 2 | SS dose at 6 on a non-SS history | 5.4881164 | 14.3101276 |
//!
//! Each tie has one observation before the dose and one after it, so the pairs straddle the
//! rule: an engine ignoring record order at an SS dose reads both as 14.31. The fixture
//! covers a first dose (nothing before it) and a later dose with residual drug, so both
//! sides of the dose are live.
//!
//! `predict()` and the dual walk run on the closed form (`ss_record_order_fit.ferx`) and on
//! its `[odes]` twin (`ss_record_order_ode_fit.ferx`); the state pass on the twin. Tier 2: one `predict()`, one dual walk and one state
//! pass per subject, and a read of the committed NONMEM table.

use ferx_core::sens::provider::subject_eta_grad;
use ferx_core::{parse_full_model_file, predict, read_nonmem_csv, CompiledModel, Population};
use std::path::Path;

const MODELS: [&str; 2] = [
    "nonmem_anchor/ss_record_order_fit.ferx",
    "nonmem_anchor/ss_record_order_ode_fit.ferx",
];

// Measured worst |ferx − NONMEM|: 1.776e-15 (closed form) and 2.665e-15 (ODE twin, reltol
// 1e-10, abstol 1e-12) across predict, the dual walk and the state pass — the same digits on
// Linux x86_64 (`tools/linux-test.sh`, glibc 2.41) and macOS arm64. NONMEM's table is ADVAN1
// at 17 significant digits. The closed-form leg is analytic against analytic; the ODE leg is a
// numerical solve against it, and its error sits at the `1e-10` reltol's floor only because
// one-compartment decay is smooth. The bound leaves ~3,750× headroom over the worst; the
// smallest defect it guards (a pre-dose read taking the steady state) is 14.31 − 5.49 = 8.8.
const BOUND: f64 = 1e-11;

fn setup(model: &str) -> (CompiledModel, Population) {
    let m = parse_full_model_file(Path::new(model))
        .expect("model")
        .model;
    let pop = read_nonmem_csv(
        Path::new("nonmem_anchor/dtmm_ss_record_order.csv"),
        None,
        None,
    )
    .expect("data");
    (m, pop)
}

/// `(ID, TIME, IPRED)` of every observation (`EVID = 0`) row of NONMEM's table, in file order.
fn nonmem_ipred() -> Vec<(String, f64, f64)> {
    let path = "nonmem_anchor/results/dtmm_ss_record_order.tab";
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |c: &str| header.iter().position(|h| *h == c).expect(c);
    let (id, time, evid, ipred) = (col("ID"), col("TIME"), col("EVID"), col("IPRED"));
    lines
        .map(|l| {
            l.split_whitespace()
                .map(|v| {
                    v.parse::<f64>()
                        .unwrap_or_else(|e| panic!("{path}: {v}: {e}"))
                })
                .collect::<Vec<f64>>()
        })
        .filter(|r| r[evid] == 0.0)
        .map(|r| (format!("{}", r[id] as i64), r[time], r[ipred]))
        .collect()
}

/// Compare one engine's rows with NONMEM's, row by row; returns the worst error.
fn assert_nonmem(leg: &str, got: &[(String, f64, f64)]) -> f64 {
    let want = nonmem_ipred();
    assert_eq!(got.len(), want.len(), "{leg}: observation count");
    let mut worst = 0.0_f64;
    for ((id, t, v), (wid, wt, w)) in got.iter().zip(&want) {
        assert!(
            id == wid && (t - wt).abs() < 1e-9,
            "{leg}: row ({id}, {t}) is expected ({wid}, {wt})"
        );
        assert!(v.is_finite(), "{leg} ID {id} TIME {t}: {v}");
        let err = (v - w).abs();
        assert!(
            err < BOUND,
            "{leg} ID {id} TIME {t}: ferx {v} vs NONMEM {w} (|error| {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!(
        "{leg}: worst |ferx − NONMEM| {worst:.3e} over {} observations",
        want.len()
    );
    worst
}

fn via_predict(m: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    predict(m, pop, &m.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

/// Label one value per observation with `predict()`'s `(ID, TIME)`.
fn label(
    m: &CompiledModel,
    pop: &Population,
    values: Vec<(String, f64)>,
) -> Vec<(String, f64, f64)> {
    let labels = via_predict(m, pop);
    assert_eq!(values.len(), labels.len(), "row count against predict");
    values
        .into_iter()
        .zip(labels)
        .map(|((id, f), (pid, t, _))| {
            assert_eq!(id, pid, "rows are in one subject order");
            (id, t, f)
        })
        .collect()
}

/// The fixture straddles the rule on NONMEM's side: each tie's pre-dose and post-dose rows
/// differ by at least 8.8, and the three pre-dose reads are the three distinct histories.
/// Without this the anchor could pass on a table whose pairs agree.
#[test]
fn nonmem_table_straddles_every_ss_tie() {
    let nm = nonmem_ipred();
    let ss = 14.310127606933330;
    let pre: Vec<f64> = nm
        .windows(2)
        .filter(|w| w[0].0 == w[1].0 && w[0].1 == w[1].1)
        .map(|w| {
            assert!(
                (w[1].2 - ss).abs() < 1e-12,
                "post-dose row reads the steady state"
            );
            assert!(
                ss - w[0].2 > 8.8,
                "pre-dose row {w:?} reads below the steady state"
            );
            w[0].2
        })
        .collect();
    assert_eq!(pre.len(), 3, "three SS ties, {nm:?}");
    let want = [0.0, 4.3101276069333299, 5.4881163609402641];
    for (p, w) in pre.iter().zip(want) {
        assert!(
            (p - w).abs() < 1e-12,
            "pre-dose reads {pre:?}, want {want:?}"
        );
    }
}

#[test]
fn predict_reads_an_observation_before_an_ss_dose_pre_dose() {
    for model in MODELS {
        let (m, pop) = setup(model);
        assert_nonmem(&format!("predict {model}"), &via_predict(&m, &pop));
    }
}

/// The adjoint's recorder, the walk `fit()` differentiates.
#[test]
fn dual_walk_reads_an_observation_before_an_ss_dose_pre_dose() {
    for model in MODELS {
        let (m, pop) = setup(model);
        let eta = vec![0.0; m.n_eta];
        let mut values = Vec::new();
        for s in &pop.subjects {
            let obs = subject_eta_grad(&m, s, &m.default_params.theta, &eta).unwrap_or_else(|| {
                panic!(
                    "{model} ID {}: routed to FD, the dual walk is not exercised",
                    s.id
                )
            });
            assert_eq!(
                obs.len(),
                s.obs_times.len(),
                "{model} ID {}: dual rows",
                s.id
            );
            values.extend(obs.iter().map(|o| (s.id.clone(), o.f)));
        }
        assert_nonmem(&format!("dual {model}"), &label(&m, &pop, values));
    }
}

/// The ODE state pass behind the sdtab compartment columns and `[derived]`, read from the
/// states (`central / V`), not from its `ipred`. ODE twin only: the closed form's state
/// vector is not in amounts.
#[test]
fn state_pass_reads_an_observation_before_an_ss_dose_pre_dose() {
    for model in [MODELS[1]] {
        let (m, pop) = setup(model);
        let theta = &m.default_params.theta;
        let eta = vec![0.0; m.n_eta];
        let v = theta[m.theta_names.iter().position(|n| n == "TVV").expect("TVV")];
        let mut values = Vec::new();
        for s in &pop.subjects {
            let (_, states) = ferx_core::pk::compute_predictions_with_states(&m, s, theta, &eta);
            assert_eq!(
                states.len(),
                s.obs_times.len(),
                "{model} ID {}: state rows",
                s.id
            );
            values.extend(states.iter().map(|u| (s.id.clone(), u[0] / v)));
        }
        assert_nonmem(&format!("states {model}"), &label(&m, &pop, values));
    }
}
