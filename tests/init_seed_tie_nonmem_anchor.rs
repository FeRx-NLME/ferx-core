//! NONMEM anchors for #2121: which record's `$PK` seeds `init(state) = expr` when the
//! subject's first instant holds several records.
//!
//! NONMEM evaluates `A_0` from the `$PK` of the subject's **first record in file order, of
//! any kind** — an observation, an EVID=2 row or a dose. Every ferx engine used to scan
//! observations ahead of EVID=2 rows, and the adaptive driver skipped doses entirely, so an
//! EVID=2-then-observation tie (a covariate-marker row written before the baseline sample)
//! seeded from the second record.
//!
//! * `init_seed_tie` (14 subjects): `A_0(1) = 10·WT`, CL/V fixed, η = 0. WT reaches the
//!   prediction only through the seed, so `IPRED(0)` reads it directly: 14 for WT = 70, 28
//!   for WT = 140. Every pair of kinds at the first instant, both file orders; a dose at 0
//!   with no tie (ID 7); time-varying WT later (IDs 9, 10); co-timed resets (IDs 12–14).
//! * `init_seed_tie_cells`: an EVID=3 row tied with the first observation (IDs 16, 17) and
//!   an SS=1 dose at the first instant (ID 18).
//! * `init_seed_tie_ofv_{a,b}`: 20 subjects in the EVID=2-then-observation shape with η on
//!   the seed, `METHOD=1 INTER MAXEVAL=0 POSTHOC`. Arm b is the control: both rows carry the
//!   observation's WT, so it reads the same under either rule.
//!
//! On the merge-base `925f3b02` the engines read: `predict` 28 on ID 3 (NONMEM 14); adaptive
//! 28 / 30 / 27.145 on IDs 3 / 5 / 7 (NONMEM 14 / 16 / 14.477), with the frozen-replay
//! verifier passing on all three; `fit()` OFV 222.41002862 on arm a (NONMEM 172.84328734),
//! and 200.17 with only the value-path seed fixed, which is why the dual leg is separate.
//!
//! Tier 2: one `predict()`, one dual walk per subject, one no-op adaptive run per subject,
//! and two `fit()` evaluations at `maxiter = 0`.

use ferx_core::sens::ode_provider::ode_subject_sensitivities;
use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::{
    fit, parse_full_model_file, predict, read_nonmem_csv, simulate_adaptive,
    AdaptiveSimulateOptions, CompiledModel, FitOptions, Population,
};
use std::path::Path;

/// `(ID, TIME, IPRED)` of every observation (EVID = 0) row in `nonmem_anchor/results/<name>.tab`.
fn nonmem_ipred(name: &str) -> Vec<(String, f64, f64)> {
    let path = format!("nonmem_anchor/results/{name}.tab");
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

/// The final `#OBJV` in `nonmem_anchor/results/<name>.ext` (the `-1000000000` row).
fn nonmem_objv(name: &str) -> f64 {
    let path = format!("nonmem_anchor/results/{name}.ext");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let row = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some("-1000000000"))
        .expect("final-estimate row");
    row.split_whitespace().last().unwrap().parse().unwrap()
}

fn model(name: &str) -> (CompiledModel, FitOptions) {
    let parsed = parse_full_model_file(Path::new(&format!("nonmem_anchor/{name}.ferx")))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    (parsed.model, parsed.fit_options)
}

fn data(name: &str) -> Population {
    read_nonmem_csv(Path::new(&format!("nonmem_anchor/{name}.csv")), None, None)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Compare `got` with NONMEM's table row for row, skipping the IDs in `skip`. Returns the
/// worst absolute error. Every compared value is checked finite first (`f64::max` would
/// drop a NaN) and against `bound` individually, so the message names the row.
fn assert_matches(
    leg: &str,
    got: &[(String, f64, f64)],
    nm: &[(String, f64, f64)],
    skip: &[&str],
    bound: f64,
) -> f64 {
    let nm: Vec<_> = nm
        .iter()
        .filter(|r| !skip.contains(&r.0.as_str()))
        .collect();
    let got: Vec<_> = got
        .iter()
        .filter(|r| !skip.contains(&r.0.as_str()))
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
            err < bound,
            "{leg} ID {id} TIME {t}: ferx {v} vs NONMEM {want} (|error| {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!(
        "{leg}: worst |ferx − NONMEM| {worst:.3e} over {} observations",
        nm.len()
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

/// The dual ODE walk's value at η = 0. Panics if a subject routes to FD, which would make
/// this leg the f64 path again and blind to the dual seed. Rows are labelled with
/// `predict()`'s times: `obs_times` is the engine clock, which the reader shifts past an
/// EVID=3 row that does not advance the time (IDs 12–14), while the `.tab` has the file's.
fn via_dual(m: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; m.n_eta];
    let labels = via_predict(m, pop);
    let mut values = Vec::new();
    for s in &pop.subjects {
        let sens = ode_subject_sensitivities(m, s, &m.default_params.theta, &eta)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        assert_eq!(sens.obs.len(), s.obs_times.len(), "ID {}: dual rows", s.id);
        values.extend(sens.obs.iter().map(|o| (s.id.clone(), o.f)));
    }
    assert_eq!(values.len(), labels.len(), "dual vs predict row count");
    values
        .into_iter()
        .zip(labels)
        .map(|((id, f), (pid, t, _))| {
            assert_eq!(id, pid, "dual and predict rows are in one subject order");
            (id, t, f)
        })
        .collect()
}

/// `simulate_adaptive` with a controller that never doses, one subject at a time, the
/// frozen-replay verifier on, one decision at 2.5 (after every subject's first instant).
/// The verifier cannot see a wrong seed — the replay seeds through the same function — so
/// this leg is judged against NONMEM, not against the replay.
fn via_adaptive(m: &CompiledModel, pop: &Population, skip: &[&str]) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();
    for s in pop
        .subjects
        .iter()
        .filter(|s| !skip.contains(&s.id.as_str()))
    {
        let one = Population {
            subjects: vec![s.clone()],
            ..pop.clone()
        };
        let mut opts = AdaptiveSimulateOptions::default();
        opts.seed = Some(1);
        opts.decision_times = vec![2.5];
        opts.verify = true;
        let res = simulate_adaptive(
            m,
            &one,
            &m.default_params,
            1,
            || |_: &ControllerCtx| -> Vec<DoseAction> { Vec::new() },
            &opts,
        )
        .unwrap_or_else(|e| panic!("ID {}: adaptive run refused: {e}", s.id));
        assert!(
            res.ledger.is_empty(),
            "ID {}: the no-op controller dosed",
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

// Bound for the closed-form seed cells: the measured worst is printed by each leg; see the
// constant's value note at the bottom of the file.
const IPRED_BOUND: f64 = 1e-6;

/// #2126: an observation listed before a co-timed SS=1 dose reads post-SS in ferx (2.862)
/// and the pre-SS seed in NONMEM (28). A reader ordering question, not the seed's.
const CELLS_SKIP: &[&str] = &["15"];

#[test]
fn predict_seeds_init_from_the_first_record_in_file_order_matches_nonmem() {
    let m = model("init_seed_tie_fit").0;
    assert_matches(
        "predict, init_seed_tie",
        &via_predict(&m, &data("init_seed_tie")),
        &nonmem_ipred("init_seed_tie"),
        &[],
        IPRED_BOUND,
    );
    assert_matches(
        "predict, init_seed_tie_cells",
        &via_predict(&m, &data("init_seed_tie_cells")),
        &nonmem_ipred("init_seed_tie_cells"),
        CELLS_SKIP,
        IPRED_BOUND,
    );
}

#[test]
fn dual_walk_seeds_init_from_the_first_record_in_file_order_matches_nonmem() {
    let m = model("init_seed_tie_fit").0;
    assert_matches(
        "dual, init_seed_tie",
        &via_dual(&m, &data("init_seed_tie")),
        &nonmem_ipred("init_seed_tie"),
        &[],
        IPRED_BOUND,
    );
}

/// #2123: ID 4 (an observation listed before a co-timed EVID=2 row) is refused by the
/// driver's break-alignment guard. #2124: ID 6 (a trough listed before a co-timed dose)
/// reads post-dose at t = 0; its seed is right, as its t = 2 row would show, but the whole
/// subject is left out rather than half of it.
const ADAPTIVE_SKIP: &[&str] = &["4", "6"];

#[test]
fn adaptive_driver_seeds_init_from_the_first_record_in_file_order_matches_nonmem() {
    let m = model("init_seed_tie_fit").0;
    let got = via_adaptive(&m, &data("init_seed_tie"), ADAPTIVE_SKIP);
    assert_matches(
        "adaptive, init_seed_tie",
        &got,
        &nonmem_ipred("init_seed_tie"),
        ADAPTIVE_SKIP,
        IPRED_BOUND,
    );
}

/// `fit()` at `maxiter = 0` against NONMEM's `#OBJV` on both arms. Arm a needs the EVID=2
/// row's WT in the seed on the value path **and** on the dual path; arm b is the control,
/// unchanged by either rule. No subject may route to finite-difference inner gradients, or
/// the dual seed would not be exercised.
#[test]
fn fit_objective_at_an_evid2_obs_tie_matches_nonmem() {
    let (m, opts) = model("init_seed_tie_ofv_fit");
    for arm in ["init_seed_tie_ofv_a", "init_seed_tie_ofv_b"] {
        let pop = data(arm);
        let r = fit(&m, &pop, &m.default_params, &opts).unwrap_or_else(|e| panic!("{arm}: {e}"));
        let want = nonmem_objv(arm);
        assert!(r.ofv.is_finite(), "{arm}: OFV {}", r.ofv);
        let err = (r.ofv - want).abs();
        println!(
            "{arm}: ferx OFV {:.8}, NONMEM {want:.8}, |Δ| {err:.3e}",
            r.ofv
        );
        assert!(
            err < OFV_BOUND,
            "{arm}: ferx OFV {} vs NONMEM #OBJV {want} (|Δ| {err:.3e})",
            r.ofv
        );
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains("finite-difference inner gradients")),
            "{arm}: a subject routed to FD, so the dual seed was not exercised: {:?}",
            r.warnings
        );
    }
}

// The smallest defect each bound guards: a wrong seed moves IPRED(0) by 14 (a factor 2 in
// WT) and arm a's OFV by 27.3 (dual seed alone) — both orders of magnitude above the bound.
const OFV_BOUND: f64 = 1e-5;
