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

/// The subjects in the order the run first saw them degenerate: by first affected
/// replicate, then by position in the population (ids `1`, `2`, `3`). The order of the lines.
fn first_seen(
    expected: &BTreeMap<String, Vec<(usize, usize)>>,
) -> Vec<(&String, &Vec<(usize, usize)>)> {
    let mut order: Vec<(&String, &Vec<(usize, usize)>)> = expected.iter().collect();
    order.sort_by_key(|(id, at)| (at[0], id.parse::<usize>().expect("numeric id")));
    order
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

fn expected_line(id: &str, cmt: usize, extent: &str) -> String {
    format!(
        "W_TTE_DEGENERATE_HAZARD: subject '{id}' (CMT={cmt}) drew a non-positive / non-finite \
         effective hazard rate{extent}; no event was generated and the subject is censored at \
         the observation window in those replicates (#763). Check the hazard parameters / \
         covariate values."
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
/// indices whose drawn rate is non-positive in the parameter table.
///
/// Mutations: record only a subject's first replicate; drop a replicate; record the draw
/// index in place of `sim`; render before the last replicate.
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
    // Subject 2 degenerates first (sim 3), so first-seen order is not id order here.
    let order: Vec<&str> = first_seen(&expected)
        .iter()
        .map(|(id, _)| id.as_str())
        .collect();
    assert_eq!(order, ["2", "3", "1"]);

    let lines = degenerate_lines(&out.warnings);
    let want: Vec<String> = first_seen(&expected)
        .into_iter()
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

    // Each affected replicate is censored at the window, and no other row moved: the
    // row-only wrapper, which drops the tally, returns the same rows bit for bit (U6).
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

    let want: Vec<String> = first_seen(&expected)
        .into_iter()
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

    // U6: the row-only wrapper returns the same rows.
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

    // First-seen order, and within a subject its record order (CMT 2, then 3).
    let mut want = Vec::new();
    for (id, at) in first_seen(&expected) {
        let sims: Vec<usize> = at.iter().map(|(_, sim)| *sim).collect();
        let k = at.len();
        let extent = format!(" in {k} of 50 replicates ({})", listed("sim", &sims));
        for cmt in [2, 3] {
            let line = expected_line(id, cmt, &extent);
            want.push(if k == 1 {
                line.replace("in those replicates", "in that replicate")
            } else {
                line
            });
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
            .replace("in those replicates", "in every replicate")
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
