//! `W_TTE_DEGENERATE_HAZARD` appears once per (subject, CMT) on the diagnostics-carrying
//! simulate entry points (#1648), end to end. The text of each cell is pinned on a
//! hand-built tally in `survival::degenerate_tally_tests`; here the tally is filled by a
//! real run and checked against the parameter table of that run.

use super::simulate_with_uncertainty_tests::{
    parse_fixture, synthetic_fit, tiny_population, INDOMAIN_TWINLESS_TRANSIT_SRC,
};
use super::*;
use crate::types::{EventType, ObsRecord};
use std::collections::BTreeMap;

const CODE: &str = "W_TTE_DEGENERATE_HAZARD";

/// The #1648 measurement fixture: an exponential hazard whose rate `LAM = 0.1 + η`,
/// `η ~ N(0, 0.01)`, is non-positive in roughly one replicate in six. `LAM` is an
/// individual parameter so the run's parameter table reports it per replicate; the PK block
/// is there only because `[individual_parameters]` needs a structural model (no subject
/// carries a dose or a concentration).
const SOMETIMES_DEGENERATE: &str = "
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_L ~ 0.01
  sigma PROP ~ 0.1

[individual_parameters]
  CL  = TVCL
  V   = TVV
  LAM = TVLAMBDA + ETA_L

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[event_model]
  cmt    = 2
  family = exponential
  scale  = LAM
";

/// The same rate on two competing causes: both degenerate in exactly the same replicates.
const SOMETIMES_DEGENERATE_TWO_CAUSES: &str = "
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_L ~ 0.01
  sigma PROP ~ 0.1

[individual_parameters]
  CL  = TVCL
  V   = TVV
  LAM = TVLAMBDA + ETA_L

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = LAM

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = LAM
";

fn model(src: &str) -> CompiledModel {
    crate::parser::model_parser::parse_model_string(src).expect("parse")
}

/// Three subjects, one right-censored record at 20 on each of `cmts`.
fn censored_population(cmts: &[usize]) -> Population {
    let subjects = (1..=3)
        .map(|i| Subject {
            id: i.to_string(),
            obs_records: cmts
                .iter()
                .map(|cmt| ObsRecord::Event {
                    time: 20.0,
                    event_type: EventType::RightCensored,
                    entry_time: 0.0,
                    cmt: *cmt,
                })
                .collect(),
            ..Default::default()
        })
        .collect();
    Population {
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

fn degenerate_lines(warnings: &[String]) -> Vec<&String> {
    warnings.iter().filter(|w| w.starts_with(CODE)).collect()
}

/// Per subject, the `(draw, sim)` replicates whose drawn `LAM` is non-positive: what the
/// sampler calls degenerate, read off the run's parameter table and not off the tally.
fn expected_from_parameters(out: &SimulationOutput) -> BTreeMap<String, Vec<(usize, usize)>> {
    let lam = out
        .parameters
        .individual_names
        .iter()
        .position(|n| n == "LAM")
        .expect("LAM is reported");
    let mut by_subject: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for row in &out.parameters.rows {
        let rate = row.individual[lam];
        assert!(rate.is_finite(), "LAM of {}: {rate}", row.id);
        if rate <= 0.0 {
            by_subject
                .entry(row.id.clone())
                .or_default()
                .push((row.draw, row.sim));
        }
    }
    by_subject
}

/// Per subject **position**, the `(draw, sim)` replicates whose drawn `LAM` is non-positive.
/// The parameter table holds one row per subject per replicate, in population order, so
/// each replicate is a chunk of `n_subjects` rows; this reads a subject by where it is, not
/// by its id, which two subjects may share.
fn expected_by_position(out: &SimulationOutput, ids: &[&str]) -> Vec<Vec<(usize, usize)>> {
    let lam = out
        .parameters
        .individual_names
        .iter()
        .position(|n| n == "LAM")
        .expect("LAM is reported");
    let mut by_position = vec![Vec::new(); ids.len()];
    for replicate in out.parameters.rows.chunks(ids.len()) {
        let got: Vec<&str> = replicate.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(got, ids, "one row per subject, in population order");
        for (position, row) in replicate.iter().enumerate() {
            let rate = row.individual[lam];
            assert!(rate.is_finite(), "LAM of {}: {rate}", row.id);
            if rate <= 0.0 {
                by_position[position].push((row.draw, row.sim));
            }
        }
    }
    by_position
}

/// `"sim 3, 17"`, with the 10-index cap, spelled here independently of the renderer.
fn listed(label: &str, indices: &[usize]) -> String {
    let shown: Vec<String> = indices.iter().take(10).map(|i| i.to_string()).collect();
    let more = match indices.len().saturating_sub(10) {
        0 => String::new(),
        m => format!(" and {m} more"),
    };
    format!("{label} {}{more}", shown.join(", "))
}

/// The whole line, with `censored_in` after the observation window: the count sits on the
/// censoring claim (#1648 review r1 row 2).
fn expected_line(id: &str, cmt: usize, censored_in: &str) -> String {
    format!(
        "W_TTE_DEGENERATE_HAZARD: subject '{id}' (CMT={cmt}) drew a non-positive / non-finite \
         effective hazard rate; no event was generated and the subject is censored at the \
         observation window{censored_in} (#763). Check the hazard parameters / covariate \
         values."
    )
}

fn row_bits(rows: &[SimulationResult]) -> Vec<String> {
    rows.iter()
        .map(|r| {
            format!(
                "{} {} {} {} {} {:?}",
                r.draw,
                r.sim,
                r.id,
                r.cmt,
                r.time.to_bits(),
                r.outcome
            )
        })
        .collect()
}

fn options() -> SimulateOptions {
    SimulateOptions {
        seed: Some(1),
        horizon: Some(20.0),
        ..Default::default()
    }
}

fn uncertainty(draws: usize, per_draw: usize) -> SimulateUncertaintyOptions {
    SimulateUncertaintyOptions {
        n_uncertainty_draws: draws,
        n_sim_per_draw: per_draw,
        seed: Some(1),
        horizon: Some(20.0),
        ..Default::default()
    }
}

/// U2: `simulate_with_options_diag`, 50 replicates. Three lines where `main` gave 28
/// (9, 9 and 10 per subject, measured at `23688c51c`), each naming exactly the `sim`
/// indices whose drawn rate is non-positive in the parameter table, in population order.
///
/// Mutations: record only a subject's first replicate; drop a replicate; record the draw
/// index in place of `sim`; render before the last replicate; render in first-seen order.
#[test]
fn options_run_reports_each_subject_once_with_its_replicates() {
    let m = model(SOMETIMES_DEGENERATE);
    let pop = censored_population(&[2]);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 50, &options()).expect("sim");
    let expected = expected_from_parameters(&out);

    // The fixture is the partial case for every subject, never all-or-none, and the total
    // is the number of lines `main` printed.
    assert_eq!(expected.len(), 3, "every subject degenerates somewhere");
    let counts: Vec<usize> = expected.values().map(Vec::len).collect();
    assert_eq!(counts, vec![9, 9, 10], "measured at 23688c51c");
    assert!(counts.iter().all(|k| 0 < *k && *k < 50));
    // Subject 2 degenerates first (sim 3), so the order of first hits would be 2, 3, 1: the lines
    // come in population order regardless (#1648 review r1 row 3).
    let first: Vec<(usize, &String)> = expected.iter().map(|(id, at)| (at[0].1, id)).collect();
    assert!(
        first.iter().min().map(|(_, id)| id.as_str()) == Some("2"),
        "the fixture must not hit subject 1 first: {first:?}"
    );

    let lines = degenerate_lines(&out.warnings);
    let want: Vec<String> = expected
        .iter()
        .map(|(id, at)| {
            let sims: Vec<usize> = at.iter().map(|(_, sim)| *sim).collect();
            expected_line(
                id,
                2,
                &format!(
                    " in {} of 50 replicates ({})",
                    at.len(),
                    listed("sim", &sims)
                ),
            )
        })
        .collect();
    assert_eq!(
        lines.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
        want.iter().map(String::as_str).collect::<Vec<_>>()
    );

    // Each affected replicate is censored at the window. U6: the `_diag` form returns the
    // row-only wrapper's rows bit for bit. Both go through the same chokepoint, so this pins
    // that the tally adds nothing to the rows, not that the rows equal `main`'s (that is
    // `tests/tte_smoke.rs` and the tally reading no random number).
    for (id, at) in &expected {
        for (_, sim) in at {
            let row = out
                .results
                .iter()
                .find(|r| &r.id == id && r.sim == *sim)
                .expect("row");
            assert_eq!(
                format!("{:?}", row.outcome),
                "Event { time: 20.0, observed: false }",
                "subject {id} sim {sim}"
            );
        }
    }
    let rows = simulate_with_options(&m, &pop, &m.default_params, 50, &options()).expect("rows");
    assert_eq!(row_bits(&rows), row_bits(&out.results));
}

/// U3: `simulate_with_uncertainty_diag`, 20 draws of 5. Three lines where `main` gave 51,
/// each naming the affected **draws** (a `sim` index repeats in every draw), with the
/// counts against the 100 replicates and 20 draws that ran.
///
/// Mutations: name `sim` in place of `draw`; count replicates as draws; lose a draw; read
/// the tally by replicate on this entry point.
#[test]
fn uncertainty_run_reports_each_subject_once_with_its_draws() {
    let m = model(SOMETIMES_DEGENERATE);
    let pop = censored_population(&[2]);
    let fit = synthetic_fit(&m.default_params);
    let opts = uncertainty(20, 5);
    let out = simulate_with_uncertainty_diag(&m, &pop, &fit, &opts).expect("sim");
    let expected = expected_from_parameters(&out);

    assert_eq!(expected.len(), 3);
    let counts: Vec<usize> = expected.values().map(Vec::len).collect();
    assert_eq!(counts, vec![18, 15, 15], "measured on this fixture");
    assert!(counts.iter().all(|k| 0 < *k && *k < 100));
    // At least one subject is hit twice within one draw, so "replicates" and "draws" are
    // different numbers and a line that confuses them is wrong.
    let distinct_draws = |at: &[(usize, usize)]| {
        let mut d: Vec<usize> = at.iter().map(|(draw, _)| *draw).collect();
        d.dedup();
        d
    };
    assert!(
        expected
            .values()
            .any(|at| distinct_draws(at).len() < at.len()),
        "the fixture must hit one draw twice: {expected:?}"
    );

    let want: Vec<String> = expected
        .iter()
        .map(|(id, at)| {
            let draws = distinct_draws(at);
            expected_line(
                id,
                2,
                &format!(
                    " in {} of 100 replicates, in {} of 20 simulated uncertainty draws ({})",
                    at.len(),
                    draws.len(),
                    listed("draw", &draws)
                ),
            )
        })
        .collect();
    let lines = degenerate_lines(&out.warnings);
    assert_eq!(
        lines.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
        want.iter().map(String::as_str).collect::<Vec<_>>()
    );

    // U6: the `_diag` form returns the row-only wrapper's rows.
    let rows = simulate_with_uncertainty(&m, &pop, &fit, &opts).expect("rows");
    assert_eq!(row_bits(&rows), row_bits(&out.results));
}

/// U4: competing risks. Two analytic causes that degenerate together give two lines per
/// subject, one per CMT, with the same replicates.
///
/// Mutation: key the tally on the subject alone (one merged line per subject).
#[test]
fn competing_causes_are_reported_per_cmt() {
    let m = model(SOMETIMES_DEGENERATE_TWO_CAUSES);
    let pop = censored_population(&[2, 3]);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 50, &options()).expect("sim");
    let expected = expected_from_parameters(&out);
    assert!(
        !expected.is_empty() && expected.values().all(|at| !at.is_empty() && at.len() < 50),
        "{expected:?}"
    );

    // Population order, and within a subject CMT order.
    let mut want = Vec::new();
    for (id, at) in &expected {
        let sims: Vec<usize> = at.iter().map(|(_, sim)| *sim).collect();
        let extent = format!(
            " in {} of 50 replicates ({})",
            at.len(),
            listed("sim", &sims)
        );
        for cmt in [2, 3] {
            want.push(expected_line(id, cmt, &extent));
        }
    }
    let lines = degenerate_lines(&out.warnings);
    assert_eq!(
        lines.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
        want.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

/// U7: a skipped uncertainty draw simulates nothing, so it is in neither `n` nor `D`. A
/// twin-less transit model whose inflated covariance skips some draws (#786), carrying a
/// hazard whose rate is always zero, reports "all n replicates of all D draws" with `D`
/// the draws that ran, not the 30 requested.
///
/// Mutation: count the draw at the top of the draw loop, ahead of the skip.
#[test]
fn skipped_uncertainty_draws_are_not_counted() {
    let src = format!(
        "{INDOMAIN_TWINLESS_TRANSIT_SRC}\n[event_model]\n  cmt    = 2\n  family = exponential\n  \
         scale  = TVCL - TVCL\n"
    );
    let m = parse_fixture(&src);
    let mut pop = tiny_population();
    for subject in &mut pop.subjects {
        subject.obs_records = vec![ObsRecord::Event {
            time: 20.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        }];
    }
    let mut fit = synthetic_fit(&m.default_params);
    if let Some(cov) = fit.covariance_matrix.as_mut() {
        cov[(0, 0)] = 4.0;
    }
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 30,
        n_sim_per_draw: 2,
        seed: Some(7),
        ..Default::default()
    };
    let out = simulate_with_uncertainty_diag(&m, &pop, &fit, &opts).expect("sim");
    let mut ran: Vec<usize> = out.results.iter().map(|r| r.draw).collect();
    ran.sort_unstable();
    ran.dedup();
    let d = ran.len();
    assert!(
        1 < d && d < 30,
        "some draws must be skipped and several must run: {d} of 30"
    );
    assert!(
        out.warnings
            .iter()
            .any(|w| w.contains("W_UNCERTAINTY_DRAWS_SKIPPED")),
        "{:#?}",
        out.warnings
    );

    let lines = degenerate_lines(&out.warnings);
    let want: Vec<String> = ["S1", "S2"]
        .iter()
        .map(|id| {
            expected_line(
                id,
                2,
                &format!(
                    " in all {} replicates of all {d} simulated uncertainty draws",
                    2 * d
                ),
            )
        })
        .collect();
    assert_eq!(
        lines.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
        want.iter().map(String::as_str).collect::<Vec<_>>()
    );
    // Order: after the per-subject lines, before the two run-level findings.
    let at = |needle: &str| out.warnings.iter().position(|w| w.contains(needle));
    let last_line = out
        .warnings
        .iter()
        .rposition(|w| w.starts_with(CODE))
        .expect("a line");
    assert!(
        Some(last_line) < at("W_UNCERTAINTY_DRAWS_SKIPPED"),
        "{:#?}",
        out.warnings
    );
}

/// A single replicate keeps the per-call text, with no count, on both entry points: the
/// line ferx-r holds a canned copy of.
#[test]
fn one_replicate_keeps_the_text_without_a_count() {
    let src = SOMETIMES_DEGENERATE.replace("scale  = LAM", "scale  = LAM - LAM");
    let m = model(&src);
    let pop = censored_population(&[2]);
    let today = |id: &str| {
        format!(
            "W_TTE_DEGENERATE_HAZARD: subject '{id}' (CMT=2) drew a non-positive / non-finite \
             effective hazard rate; no event was generated and the subject is censored at the \
             observation window (#763). Check the hazard parameters / covariate values."
        )
    };
    let want: Vec<String> = ["1", "2", "3"].iter().map(|id| today(id)).collect();

    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 1, &options()).expect("sim");
    assert_eq!(
        degenerate_lines(&out.warnings)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>(),
        want
    );
    let fit = synthetic_fit(&m.default_params);
    let out = simulate_with_uncertainty_diag(&m, &pop, &fit, &uncertainty(1, 1)).expect("sim");
    assert_eq!(
        degenerate_lines(&out.warnings)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>(),
        want
    );
}

/// Two causes on one subject where only cause A can degenerate; cause B fires at a fixed
/// rate. A replicate where A degenerates and B fires leaves the subject with an event, so
/// it is not counted: the count is of replicates censored because of the degenerate draw,
/// and the line says so (#1648 review r1 row 2).
const ONE_CAUSE_DEGENERATE: &str = "
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_L ~ 0.01
  sigma PROP ~ 0.1

[individual_parameters]
  CL  = TVCL
  V   = TVV
  LAM = TVLAMBDA + ETA_L

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = LAM

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = 0.02
";

/// U4b: under competing risks the count is of replicates in which the subject ended
/// censored with cause A degenerate, recomputed from the parameter table and the rows, and
/// the fixture holds replicates where A degenerated and B fired, which are not counted.
///
/// Mutations: record every degenerate draw (drop the `!event` gate); move the count back
/// onto "drew … rate".
#[test]
fn a_degenerate_cause_beside_a_live_one_counts_only_censored_replicates() {
    let m = model(ONE_CAUSE_DEGENERATE);
    let pop = censored_population(&[2, 3]);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 50, &options()).expect("sim");
    let drew = expected_from_parameters(&out);

    let censored = |id: &str, sim: usize| {
        out.results
            .iter()
            .filter(|r| r.id == id && r.sim == sim)
            .all(|r| {
                matches!(
                    r.outcome,
                    crate::types::SimOutcome::Event {
                        observed: false,
                        ..
                    }
                )
            })
    };
    let mut fired_anyway = 0;
    let mut want = Vec::new();
    for (id, at) in &drew {
        let sims: Vec<usize> = at
            .iter()
            .map(|(_, sim)| *sim)
            .filter(|sim| censored(id, *sim))
            .collect();
        fired_anyway += at.len() - sims.len();
        if !sims.is_empty() {
            want.push(expected_line(
                id,
                2,
                &format!(
                    " in {} of 50 replicates ({})",
                    sims.len(),
                    listed("sim", &sims)
                ),
            ));
        }
    }
    // The straddle: some degenerate draws were followed by cause B firing (not counted),
    // and some left the subject censored (counted). Either half alone cannot tell the two
    // rules apart.
    assert!(
        fired_anyway > 0,
        "the fixture must let cause B fire after A degenerates"
    );
    assert!(
        !want.is_empty(),
        "the fixture must leave some subject censored: {drew:?}"
    );

    let lines = degenerate_lines(&out.warnings);
    assert_eq!(
        lines.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
        want.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

/// Two subjects with the same id (the data reader makes `ID = 1, 2, 1` two subjects named
/// `1`). Each gets its own line, named `1`, with its own count; their hits never merge, so
/// no `k` exceeds what that subject drew and no `sim` index is listed twice.
///
/// Mutation: key the tally on the id (the two `1`s merge into one line).
#[test]
fn subjects_sharing_an_id_keep_their_own_lines() {
    let m = model(SOMETIMES_DEGENERATE);
    let mut pop = censored_population(&[2]);
    pop.subjects[2].id = "1".to_string();
    let ids = ["1", "2", "1"];
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 50, &options()).expect("sim");
    let by_position = expected_by_position(&out, &ids);
    assert!(
        !by_position[0].is_empty() && !by_position[2].is_empty(),
        "both subjects named 1 must degenerate somewhere: {by_position:?}"
    );

    let want: Vec<String> = by_position
        .iter()
        .zip(ids)
        .filter(|(at, _)| !at.is_empty())
        .map(|(at, id)| {
            let sims: Vec<usize> = at.iter().map(|(_, sim)| *sim).collect();
            expected_line(
                id,
                2,
                &format!(
                    " in {} of 50 replicates ({})",
                    at.len(),
                    listed("sim", &sims)
                ),
            )
        })
        .collect();
    let lines = degenerate_lines(&out.warnings);
    assert_eq!(
        lines.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
        want.iter().map(String::as_str).collect::<Vec<_>>()
    );
}
