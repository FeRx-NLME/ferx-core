//! NONMEM cross-check for #1275: **a lagged steady-state dose's arrival adds only the
//! pulse; it does not overwrite the state flowed from its record.**
//!
//! A lagged `SS=1` dose loads its periodic state at its **record** and flows it to the
//! lagged arrival (#1121). The dense ODE walks — the static walker and the adaptive driver
//! (`reseed_prescheduled_states_at`), `ode_predictions_with_states`, and
//! `ode_dense_solve_states` (`apply_segment_boundary`) — then *re-equilibrated* at the
//! arrival whenever `lag ≤ II`, replacing the whole state vector with the periodic trough.
//! That is exact for the `SS` dose's own contribution, and it erased every other dose that
//! landed inside the pre-arrival window. The event-driven walks already flowed, and were right.
//!
//! The reference is NONMEM 7.6.0 (`nm3`, `anchor` build), **`ADVAN13 TOL=12`**, `MAXEVAL=0`,
//! `FORMAT=s1PE23.16`, `CL = 2, V = 20, KA = 0.15`, `II = 12`, `AMT = 100`, the lag (`ALAG1` on
//! the depot; `ALAG2` on central for ID 6 only) the one input varied across
//! `nonmem_anchor/ss_arrival_flow_lag{11,12,13}.ctl`.
//! NONMEM equals an independent closed form (`nonmem_anchor/ss_arrival_flow_closed_form.py`,
//! a Python pulse-train sum under the #1121 clamp) to **≤ 7.1e-12** on every asserted cell.
//! ADVAN13, not ADVAN2: at `ALAG1 = 13` ADVAN2 is 17 % off both ADVAN13 and the closed form
//! for this depot `SS` dose (#1604).
//!
//! | ID | doses | |
//! |---|---|---|
//! | 1 | `SS=1` depot at 10; central bolus 100 at 11 | the window dose |
//! | 2 | `SS=1` depot at 10 alone | the control ID 1 is compared with |
//! | 3 | `SS=1` depot at 10; central bolus 50 at 15 | a later window dose |
//! | 4 | depot bolus at 0 (wiped by the `SS=1` record), then as ID 1 | |
//! | 5 | `SS=1` depot **infusion** (`RATE = 50`, 2 h) at 10; central bolus 100 at 14 | at lags 11 and 12 the previous cycle's infusion is still running at the record |
//! | 6 | `SS=1` **central** bolus at 10, lagged by `ALAG2`; unlagged depot bolus 100 at 11 | window mass flows **into the `SS` dose's own compartment** |
//!
//! ID 6 answers PR #1607's review F2. In IDs 1–5 the `SS` dose goes to the depot and every
//! window dose to central, downstream of it, so a half-fix that re-equilibrated only the
//! `SS` dose's *own* compartment at the arrival passed every one of them. ID 6 puts the
//! window mass there: that half-fix erases the depot bolus's central amount at the arrival
//! (≥ 40 mg of 100 at every lag, asserted from the closed form below). ID 6 runs on its own
//! ODE model (`ALAG2 = TVLAG`, no `ALAG1`), which is how NONMEM's `IF (ID.EQ.6)` reads.
//!
//! **The lags straddle the old gate.** At 11 and 12 (`lag ≤ II`) the dense walks overwrote
//! the arrival — measured before the fix at **−32 %** (lag 11) and **−30 %** (lag 12). At 13
//! they already flowed (`lag > II`, #1121), and every engine matched. So lag 13 is the control:
//! it stays green with any one of the three arrival sites reverted, while 11 and 12 go red on
//! the engine that site serves. The straddle's premise — that ID 1 genuinely differs from
//! ID 2 after the arrival, so "the window dose is gone" cannot read as agreement — is asserted
//! on the NONMEM table itself.
//!
//! **Engines.** The static walker (`predict`), the ODE event-driven walk (an `EVID=3` row at
//! t = 0 routes a subject there and changes nothing else), `ode_predictions_with_states`,
//! `ode_dense_solve_states` (called directly: `central / V`), the adaptive driver, and the dual
//! ODE walk's value. The analytic superposition and the analytic event walk are asserted on
//! ID 2 only: the analytic `lagtime=` slot lags every dose, so it cannot express a depot-only
//! `ALAG1` next to a central bolus. The dual walk skips ID 5: `SS` infusion × lag declines to
//! finite differences (#1128), so ID 5 would not reach the dual walk at all.
//!
//! **ID 5 is skipped at lag 13 on every engine**, by name: with `lag > II` NONMEM infuses the
//! record-time cycle for `T_inf + (lag − II)` hours (it matches that reading to 4.0e-12), which
//! ferx deliberately does not reproduce (`dosing::ss_residual_infusion_end`,
//! `docs/model-file/lagtime.qmd`). That divergence is not this issue's. At lags 11 and 12 — the
//! `lag ≤ II` cell this issue changes — ID 5 is asserted everywhere it can be.
//!
//! The oracle is the committed NONMEM table, read at run time. Evaluation only — not gated.

use ferx_core::ode::predictions::ode_dense_solve_states;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::{compute_event_pk_params, compute_predictions_with_states};
use ferx_core::sens::ode_provider::ode_subject_sensitivities;
use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::{
    predict, read_nonmem_csv, simulate_adaptive, AdaptiveSimulateOptions, CompiledModel, Population,
};

const DATA: &str = "nonmem_anchor/ss_arrival_flow.csv";
const LAGS: [f64; 3] = [11.0, 12.0, 13.0];

/// Measured worst 8.8e-12 (ODE, dual, and the adaptive driver at η = 0, which equals the
/// static walker to four figures) against ADVAN13's own `TOL=12` floor. The adaptive leg sat
/// at 3.6e-4 only while it drew η from the regularised `omega ~ 0.0` (#1603).
const ODE_BOUND: f64 = 1e-10;
/// Measured 5.8e-12: NONMEM's ADVAN13 solver error, **not** the 1e-13 the ADVAN2 anchors use.
const ANALYTIC_BOUND: f64 = 1e-10;

/// IDs other than 2, which the analytic single `lagtime` slot cannot express.
const NOT_ANALYTIC: &[&str] = &["1", "3", "4", "5", "6"];
/// The subject that runs on `ode_central` (see the module doc); every other ID runs on
/// `ode_depot`.
const CENTRAL_ID: &str = "6";
const DEPOT_IDS: &[&str] = &["1", "2", "3", "4", "5"];

fn table(lag: f64) -> String {
    format!(
        "nonmem_anchor/results/ss_arrival_flow_lag{}.sdtab",
        lag as i64
    )
}

/// ID 5 at lag 13 (`lag > II` infusion) is NONMEM's `T_inf + (lag − II)` reading, which
/// ferx does not reproduce — see the module doc.
fn lag_skip(lag: f64) -> &'static [&'static str] {
    if lag > 12.0 {
        &["5"]
    } else {
        &[]
    }
}

fn ode_depot(lag: f64) -> CompiledModel {
    ode_lagged(lag, "ALAG1")
}

/// ID 6's model: the lag is on central (`ALAG2`), the depot is unlagged.
fn ode_central(lag: f64) -> CompiledModel {
    ode_lagged(lag, "ALAG2")
}

fn ode_lagged(lag: f64, slot: &str) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta TVLAG({lag}, 0.0, 30.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  {slot} = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#
    );
    parse_full_model(&src).expect("ODE model parses").model
}

fn analytic(lag: f64) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta TVLAG({lag}, 0.0, 30.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  LAG = TVLAG
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAG)
[error_model]
  DV ~ proportional(PROP_ERR)
"#
    );
    parse_full_model(&src).expect("analytic model parses").model
}

fn load(path: &std::path::Path) -> Population {
    read_nonmem_csv(path, None, None).expect("dataset loads")
}

/// The committed dataset, as is: every subject takes the static walk.
fn static_pop() -> Population {
    let pop = load(std::path::Path::new(DATA));
    assert!(pop.subjects.iter().all(|s| s.reset_times.is_empty()));
    pop
}

/// The committed dataset with an `EVID=3` row at t = 0 ahead of each subject, which routes
/// it to the event-driven walk and changes nothing else. Written to a per-call temp file
/// (the tests run on concurrent threads).
fn event_driven_pop() -> Population {
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let text = std::fs::read_to_string(DATA).expect("csv");
    let mut out = String::new();
    let mut last_id = String::new();
    for (i, line) in text.lines().enumerate() {
        if i > 0 {
            let id = line.split(',').next().expect("ID column").to_string();
            if id != last_id {
                out.push_str(&format!("{id},0,0,3,0,1,1,0,0,0\n"));
                last_id = id;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    let path = std::env::temp_dir().join(format!(
        "ferx_1275_ss_arrival_flow_evid3_{}_{}.csv",
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, out).expect("write temp csv");
    let pop = load(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        pop.subjects.iter().all(|s| s.reset_times == vec![0.0]),
        "every subject must carry the leading reset, or it is not on the event-driven walk"
    );
    pop
}

/// `(ID, TIME, PRED)` for every observation record of a NONMEM `$TABLE ... NOAPPEND` file
/// whose columns are `ID TIME EVID PRED`.
fn nonmem_obs(path: &str) -> Vec<(String, f64, f64)> {
    let text = std::fs::read_to_string(path).expect("NONMEM table");
    text.lines()
        .skip(2)
        .map(|l| {
            let c: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            (c[0], c[1], c[2], c[3])
        })
        .filter(|&(_, _, evid, _)| evid == 0.0)
        .map(|(id, t, _, p)| (format!("{}", id as i64), t, p))
        .collect()
}

/// Check the engine's `(ID, TIME, value)` rows reproduce `table` to `bound` (relative) on
/// every ID outside `skip`, returning every miss. `got` may omit whole IDs only if `skip`
/// names them. Each value is asserted finite before it is folded (`f64::max` discards `NaN`),
/// and a skip list naming an ID the table lacks, or a check of nothing, panics outright.
fn check_matches(
    label: &str,
    got: &[(String, f64, f64)],
    table: &str,
    bound: f64,
    skip: &[&str],
) -> Result<(), String> {
    let want: Vec<_> = nonmem_obs(table)
        .into_iter()
        .filter(|(id, _, _)| !skip.contains(&id.as_str()))
        .collect();
    let got: Vec<_> = got
        .iter()
        .filter(|(id, _, _)| !skip.contains(&id.as_str()))
        .collect();
    for s in skip {
        assert!(
            nonmem_obs(table).iter().any(|(id, _, _)| id == s),
            "{label}: skipped ID {s} is not in the data"
        );
    }
    assert!(!want.is_empty(), "{label}: nothing asserted");
    assert_eq!(got.len(), want.len(), "{label}: observation count");
    let mut worst = 0.0_f64;
    let mut bad = Vec::new();
    for ((gid, gt, g), (id, t, w)) in got.into_iter().zip(&want) {
        assert!(
            gid == id && (gt - t).abs() < 1e-12,
            "{label}: row order ({gid}, {gt}) vs NONMEM ({id}, {t})"
        );
        assert!(g.is_finite(), "{label}: ID {id} t={t} non-finite ({g})");
        let rel = (g - w).abs() / w.abs();
        if !(rel < bound) {
            bad.push(format!(
                "ID {id} t={t}: ferx {g} vs NONMEM {w} (rel {:+.3e})",
                (g - w) / w
            ));
        }
        worst = worst.max(rel);
    }
    println!(
        "#1275 {label}: worst rel vs NONMEM {worst:.3e} (bound {bound:.0e}, skipped {skip:?})"
    );
    if bad.is_empty() {
        return Ok(());
    }
    Err(format!(
        "#1275 {label}: {} cells off NONMEM ADVAN13 — a dense walk re-equilibrating at the \
         lagged SS arrival erases the window dose:\n  {}",
        bad.len(),
        bad.join("\n  ")
    ))
}

/// Run `check` at every lag, then fail once naming every red cell: a failure at lag 11 must
/// not hide whether lag 12 is red too and the lag-13 control green.
fn every_lag(mut check: impl FnMut(f64) -> Vec<Result<(), String>>) {
    let fails: Vec<String> = LAGS
        .iter()
        .flat_map(|&lag| check(lag))
        .filter_map(Result::err)
        .collect();
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

fn via_predict(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    predict(model, pop, &model.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

/// `compute_predictions_with_states` on a static subject is `ode_predictions_with_states`.
fn via_with_states(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let (ipred, _) =
            compute_predictions_with_states(model, s, &model.default_params.theta, &eta);
        for (t, v) in s.obs_times.iter().zip(ipred) {
            out.push((s.id.clone(), *t, v));
        }
    }
    out
}

/// `ode_dense_solve_states` itself (`apply_segment_boundary`), read as `central / V`. Called
/// directly: `compute_predictions_with_states` reaches it only on the event-driven route,
/// which would not exercise the static arrival. The model has no covariates, so the first
/// observation's snapshot is every snapshot.
fn via_dense_states(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let ode = model.ode_spec.as_ref().expect("an ODE model");
    let theta = &model.default_params.theta;
    let eta = vec![0.0; model.n_eta];
    let central = 1usize; // `states=[depot, central]`
    let v = 20.0;
    let mut out = Vec::new();
    for s in &pop.subjects {
        let pk = compute_event_pk_params(model, s, theta, &eta).obs[0];
        let dense = ode_dense_solve_states(ode, &pk.values, theta, &eta, s, &s.obs_times);
        assert_eq!(dense.len(), s.obs_times.len());
        for (t, st) in s.obs_times.iter().zip(dense) {
            out.push((s.id.clone(), *t, st[central] / v));
        }
    }
    out
}

/// The dual ODE walk's value (`integrate_tvcov_g`) on every subject but ID 5, asserting each
/// one stays on the analytic provider: an FD-routed subject would make this leg FD-of-f64 and
/// blind to the dual walk. ID 5 (`SS` infusion × lag) is declined to FD by design (#1128).
fn via_dual(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    let mut out = Vec::new();
    for s in pop.subjects.iter().filter(|s| s.id != "5") {
        let sens = ode_subject_sensitivities(model, s, &model.default_params.theta, &eta)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        for (t, o) in s.obs_times.iter().zip(sens.obs) {
            out.push((s.id.clone(), *t, o.f));
        }
    }
    out
}

/// `simulate_adaptive` with a controller that never doses: the base regimen through the
/// adaptive driver, frozen-replay verifier on.
/// It runs at η = 0 exactly, as the NONMEM `PRED` it is judged against does: `omega ~ 0.0`
/// is PD-regularised to Ω = 1e-8 (`OmegaMatrix`), so the default parameters would draw η
/// with sd 1e-4 and put the driver ~3e-4 off the table (#1603). η threading is pinned
/// elsewhere (`adaptive_iov_matches_predict_iov_with_reconstructed_kappa`).
fn via_adaptive(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);
    opts.decision_times = vec![25.0];
    let mut params = model.default_params.clone();
    params.omega.chol.fill(0.0);
    params.omega.matrix.fill(0.0);
    let res = simulate_adaptive(
        model,
        pop,
        &params,
        1,
        || |_: &ControllerCtx| -> Vec<DoseAction> { Vec::new() },
        &opts,
    )
    .expect("adaptive sim runs and passes the frozen-replay verifier");
    assert!(res.ledger.is_empty(), "the no-op controller dosed");
    res.trajectories
        .into_iter()
        .map(|r| (r.id, r.time, r.ipred))
        .collect()
}

type OdeCase = (&'static str, CompiledModel, Population, Vec<&'static str>);

/// Each ODE model with the subjects it serves: IDs 1–5 on `ode_depot`, ID 6 on
/// `ode_central`. The population is restricted, and the rest named in `skip`, so a skip list
/// never hides a subject that ran.
fn ode_cases(lag: f64, pop: Population) -> Vec<OdeCase> {
    let only = |keep: &dyn Fn(&str) -> bool| {
        let mut p = pop.clone();
        p.subjects.retain(|s| keep(&s.id));
        assert!(!p.subjects.is_empty());
        p
    };
    vec![
        (
            "",
            ode_depot(lag),
            only(&|id| id != CENTRAL_ID),
            skip_plus(lag, &[CENTRAL_ID]),
        ),
        (
            " [ID 6, central SS]",
            ode_central(lag),
            only(&|id| id == CENTRAL_ID),
            DEPOT_IDS.to_vec(),
        ),
    ]
}

fn skip_plus(lag: f64, extra: &[&'static str]) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = lag_skip(lag).to_vec();
    for e in extra {
        if !v.contains(e) {
            v.push(e);
        }
    }
    v
}

// ---- the straddle's premise, on the reference itself ----

/// After the arrival, ID 1 (window bolus) must differ from ID 2 (`SS=1` alone) in NONMEM,
/// at every lag. Measured least +11.3 % (t = 40, lag 13); the dense walks' old reading of
/// ID 1 equalled ID 2 exactly. Without this, a fixture whose window dose had no effect
/// would pass under both behaviours.
#[test]
fn the_window_dose_is_live_after_the_arrival_in_nonmem() {
    for lag in LAGS {
        let rows = nonmem_obs(&table(lag));
        let at = |id: &str, t: f64| {
            rows.iter()
                .find(|(i, tt, _)| i == id && (tt - t).abs() < 1e-12)
                .map(|r| r.2)
                .unwrap_or_else(|| panic!("no ID {id} t={t}"))
        };
        let post: Vec<f64> = rows
            .iter()
            .filter(|(i, t, _)| i == "1" && *t > 10.0 + lag)
            .map(|r| r.1)
            .collect();
        assert!(post.len() >= 3, "lag {lag}: need post-arrival samples");
        for t in post {
            let rel = at("1", t) / at("2", t) - 1.0;
            assert!(rel > 0.1, "lag {lag}, t={t}: ID 1 vs ID 2 only {rel:+.3e}");
        }
        // ID 6: the depot bolus at 11 has put this much into central — the `SS` dose's own
        // compartment — by the arrival at 10 + lag (closed form, 1-cpt oral amount). Measured
        // 43.4 / 42.2 / 40.8 mg of the 100 mg pulse at lags 11 / 12 / 13.
        let (ka, k) = (0.15_f64, 0.1_f64);
        let tau = 10.0 + lag - 11.0;
        let in_central = 100.0 * ka / (ka - k) * ((-k * tau).exp() - (-ka * tau).exp());
        assert!(
            in_central > 40.0,
            "lag {lag}: only {in_central:.1} mg of window mass in central at the arrival"
        );
        assert!(rows.iter().any(|(i, _, _)| i == CENTRAL_ID));
    }
}

// ---- dense walks: the three sites #1275 changes ----

/// `reseed_prescheduled_states_at`, static caller.
#[test]
fn ode_static_walker_flows_a_lagged_ss_dose_to_its_arrival() {
    every_lag(|lag| {
        ode_cases(lag, static_pop())
            .into_iter()
            .map(|(tag, m, pop, skip)| {
                let got = via_predict(&m, &pop);
                let label = format!("ODE static{tag}, lag {lag}");
                check_matches(&label, &got, &table(lag), ODE_BOUND, &skip)
            })
            .collect()
    });
}

/// `reseed_prescheduled_states_at`, adaptive caller.
#[test]
fn adaptive_driver_flows_a_lagged_ss_dose_to_its_arrival() {
    every_lag(|lag| {
        ode_cases(lag, static_pop())
            .into_iter()
            .map(|(tag, m, pop, skip)| {
                let got = via_adaptive(&m, &pop);
                let label = format!("adaptive{tag}, lag {lag}");
                check_matches(&label, &got, &table(lag), ODE_BOUND, &skip)
            })
            .collect()
    });
}

/// `ode_predictions_with_states`' own arrival.
#[test]
fn ode_with_states_flows_a_lagged_ss_dose_to_its_arrival() {
    every_lag(|lag| {
        ode_cases(lag, static_pop())
            .into_iter()
            .map(|(tag, m, pop, skip)| {
                let got = via_with_states(&m, &pop);
                let label = format!("ODE with_states{tag}, lag {lag}");
                check_matches(&label, &got, &table(lag), ODE_BOUND, &skip)
            })
            .collect()
    });
}

/// `apply_segment_boundary` (`ode_dense_solve_states`; also `ode_solve_until_chz_threshold`).
#[test]
fn ode_dense_solve_states_flows_a_lagged_ss_dose_to_its_arrival() {
    every_lag(|lag| {
        ode_cases(lag, static_pop())
            .into_iter()
            .map(|(tag, m, pop, skip)| {
                let got = via_dense_states(&m, &pop);
                let label = format!("ode_dense_solve_states{tag}, lag {lag}");
                check_matches(&label, &got, &table(lag), ODE_BOUND, &skip)
            })
            .collect()
    });
}

// ---- twins that already flowed: nothing else moved ----

#[test]
fn ode_event_driven_walks_flow_a_lagged_ss_dose_to_its_arrival() {
    every_lag(|lag| {
        let mut out = Vec::new();
        for (tag, m, pop, skip) in ode_cases(lag, event_driven_pop()) {
            let got = via_predict(&m, &pop);
            let label = format!("ODE event-driven{tag}, lag {lag}");
            out.push(check_matches(&label, &got, &table(lag), ODE_BOUND, &skip));
            let got = via_with_states(&m, &pop);
            let label = format!("ODE event-driven with_states{tag}, lag {lag}");
            out.push(check_matches(&label, &got, &table(lag), ODE_BOUND, &skip));
        }
        out
    });
}

#[test]
fn dual_ode_walk_value_flows_a_lagged_ss_dose_to_its_arrival() {
    every_lag(|lag| {
        let mut out = Vec::new();
        for (kind, pop) in [("static", static_pop()), ("EVID=3", event_driven_pop())] {
            for (tag, m, pop, mut skip) in ode_cases(lag, pop) {
                if !skip.contains(&"5") {
                    skip.push("5"); // `via_dual` leaves ID 5 out (#1128)
                }
                let got = via_dual(&m, &pop);
                let label = format!("dual ODE ({kind} subjects){tag}, lag {lag}");
                out.push(check_matches(&label, &got, &table(lag), ODE_BOUND, &skip));
            }
        }
        out
    });
}

#[test]
fn analytic_engines_match_the_ss_dose_alone() {
    every_lag(|lag| {
        let a = analytic(lag);
        let skip = skip_plus(lag, NOT_ANALYTIC);
        let got = via_predict(&a, &static_pop());
        let label = format!("analytic superposition, lag {lag}");
        let sup = check_matches(&label, &got, &table(lag), ANALYTIC_BOUND, &skip);
        let got = via_predict(&a, &event_driven_pop());
        let label = format!("analytic event-driven, lag {lag}");
        let walk = check_matches(&label, &got, &table(lag), ANALYTIC_BOUND, &skip);
        vec![sup, walk]
    });
}
