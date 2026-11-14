//! `W_TTE_DEGENERATE_HAZARD` is reported once per (subject, CMT) for a simulation run
//! (#1648): the renderer's text in every cell of its input space, and the two forms of
//! `simulate_tte` that feed it.

use super::*;
use crate::types::{EventType, ObsRecord, Subject};
use rand::SeedableRng;

const HEAD: &str = "W_TTE_DEGENERATE_HAZARD: subject 'A' (CMT=2) drew a non-positive / \
                    non-finite effective hazard rate";
const TAIL: &str = " (#763). Check the hazard parameters / covariate values.";

/// The whole line for subject `A`, CMT 2: `extent` after the rate, `censored_where` after
/// the window. Spelled here, not taken from `degenerate_hazard_text`, so the test owns the
/// expected text.
fn line(extent: &str, censored_where: &str) -> String {
    format!(
        "{HEAD}{extent}; no event was generated and the subject is censored at the \
         observation window{censored_where}{TAIL}"
    )
}

/// A tally of `draws` draws of `n_sim` replicates with subject `A`, CMT 2 degenerate at `at`.
fn tally(draws: usize, n_sim: usize, at: &[(usize, usize)]) -> DegenerateHazardTally {
    let mut t = DegenerateHazardTally::default();
    for _ in 0..draws {
        t.count_draw(n_sim);
    }
    for (draw, sim) in at {
        t.record("A", 2, *draw, *sim);
    }
    t
}

/// U1: one exact-text assertion per cell of the message's input space. Each clause of the
/// text is stated by at least one cell, so deleting a clause from the renderer fails the
/// cell named beside it.
#[test]
fn every_cell_of_the_degenerate_hazard_text() {
    // Cell 1: a single replicate keeps the per-call text byte for byte, on both forms.
    // Dies if: a count or an index is added when `n = 1`.
    let today = line("", "");
    assert_eq!(
        today,
        "W_TTE_DEGENERATE_HAZARD: subject 'A' (CMT=2) drew a non-positive / non-finite \
         effective hazard rate; no event was generated and the subject is censored at the \
         observation window (#763). Check the hazard parameters / covariate values.",
        "the helper spells today's text"
    );
    let one = tally(1, 1, &[(1, 1)]);
    assert_eq!(one.warnings_by_replicate(), vec![today.clone()]);
    assert_eq!(one.warnings_by_draw(), vec![today.clone()]);
    assert_eq!(degenerate_hazard_warning("A", 2), today);

    // Cell 2: some replicates, at most 10. Dies if: "k of n", the `sim` list, or "in those
    // replicates" is dropped.
    assert_eq!(
        tally(1, 50, &[(1, 3), (1, 17), (1, 41)]).warnings_by_replicate(),
        vec![line(
            " in 3 of 50 replicates (sim 3, 17, 41)",
            " in those replicates"
        )]
    );
    // One affected replicate reads in the singular.
    assert_eq!(
        tally(1, 50, &[(1, 7)]).warnings_by_replicate(),
        vec![line(" in 1 of 50 replicates (sim 7)", " in that replicate")]
    );
    // Exactly 10 indices are all named: the cap is "more than 10".
    let ten: Vec<(usize, usize)> = (1..=10).map(|s| (1, 2 * s)).collect();
    assert_eq!(
        tally(1, 50, &ten).warnings_by_replicate(),
        vec![line(
            " in 10 of 50 replicates (sim 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)",
            " in those replicates"
        )]
    );

    // Cell 3: more than 10. Dies if: the cap or "and m more" is dropped, or `k` becomes the
    // number of indices shown.
    let twelve: Vec<(usize, usize)> = (1..=12).map(|s| (1, s)).collect();
    assert_eq!(
        tally(1, 50, &twelve).warnings_by_replicate(),
        vec![line(
            " in 12 of 50 replicates (sim 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 and 2 more)",
            " in those replicates"
        )]
    );

    // Cell 4: every replicate. Dies if: "all" falls back to "k of n" or an index list.
    let all: Vec<(usize, usize)> = (1..=4).map(|s| (1, s)).collect();
    assert_eq!(
        tally(1, 4, &all).warnings_by_replicate(),
        vec![line(" in all 4 replicates", " in every replicate")]
    );

    // Cell 5: uncertainty draws, some replicates. The affected **draws** are named, each
    // once, never a bare `sim` list. Dies if: the draws clause is dropped, `sim` is named,
    // or a draw hit twice is counted twice.
    let some = [(2, 1), (2, 3), (5, 2), (11, 1)];
    assert_eq!(
        tally(20, 5, &some).warnings_by_draw(),
        vec![line(
            " in 4 of 100 replicates, in 3 of 20 simulated uncertainty draws (draw 2, 5, 11)",
            " in those replicates"
        )]
    );
    // The same tally read by replicate names the `sim` indices: the two forms differ.
    assert_eq!(
        tally(20, 5, &some).warnings_by_replicate(),
        vec![line(
            " in 4 of 100 replicates (sim 1, 3, 2, 1)",
            " in those replicates"
        )]
    );
    // More than 10 affected draws: the same cap.
    let many: Vec<(usize, usize)> = (1..=13).map(|d| (d, 1)).collect();
    assert_eq!(
        tally(20, 5, &many).warnings_by_draw(),
        vec![line(
            " in 13 of 100 replicates, in 13 of 20 simulated uncertainty draws (draw 1, 2, 3, \
             4, 5, 6, 7, 8, 9, 10 and 3 more)",
            " in those replicates"
        )]
    );

    // Cell 6: every replicate of every draw.
    let every: Vec<(usize, usize)> = (1..=3).flat_map(|d| (1..=2).map(move |s| (d, s))).collect();
    assert_eq!(
        tally(3, 2, &every).warnings_by_draw(),
        vec![line(
            " in all 6 replicates of all 3 simulated uncertainty draws",
            " in every replicate"
        )]
    );

    // Cell 7: `n` and `D` are what `count_draw` counted, so a skipped draw (never counted)
    // is in neither. 18 draws ran of a requested 20.
    assert_eq!(
        tally(18, 5, &[(4, 2)]).warnings_by_draw(),
        vec![line(
            " in 1 of 90 replicates, in 1 of 18 simulated uncertainty draws (draw 4)",
            " in that replicate"
        )]
    );

    // An uncertainty run of a single draw has no ambiguous `sim`: it reads by replicate.
    assert_eq!(
        tally(1, 5, &[(1, 2), (1, 4)]).warnings_by_draw(),
        vec![line(
            " in 2 of 5 replicates (sim 2, 4)",
            " in those replicates"
        )]
    );

    // Cell 8: two CMTs of one subject, and a second subject: one line each, first seen first.
    let mut t = DegenerateHazardTally::default();
    t.count_draw(3);
    t.record("A", 2, 1, 1);
    t.record("A", 3, 1, 1);
    t.record("B", 2, 1, 2);
    t.record("A", 2, 1, 3);
    let lines = t.warnings_by_replicate();
    assert_eq!(lines.len(), 3, "{lines:#?}");
    assert_eq!(
        lines[0],
        line(" in 2 of 3 replicates (sim 1, 3)", " in those replicates")
    );
    assert!(
        lines[1].starts_with("W_TTE_DEGENERATE_HAZARD: subject 'A' (CMT=3) ")
            && lines[1].contains(" in 1 of 3 replicates (sim 1)"),
        "{}",
        lines[1]
    );
    assert!(
        lines[2].starts_with("W_TTE_DEGENERATE_HAZARD: subject 'B' (CMT=2) ")
            && lines[2].contains(" in 1 of 3 replicates (sim 2)"),
        "{}",
        lines[2]
    );

    // Nothing degenerate: no line.
    assert!(tally(4, 5, &[]).warnings_by_replicate().is_empty());
    assert!(tally(4, 5, &[]).warnings_by_draw().is_empty());
}

const ZERO_RATE: &str = "
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA_L ~ 0.01

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA + ETA_L
";

const ZERO_RATE_TWO_CAUSES: &str = "
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA_L ~ 0.01

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA + ETA_L

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA + ETA_L
";

fn censored_subject(cmts: &[usize]) -> Subject {
    Subject {
        id: "A".to_string(),
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
    }
}

/// Both forms of `simulate_tte` at `eta` on `model`: `(rows, warnings)` of the public one,
/// then `(rows, warnings, degenerate CMTs)` of the reporting one, from the same seed.
#[allow(clippy::type_complexity)]
fn both_forms(
    src: &str,
    cmts: &[usize],
    eta: f64,
) -> (
    (Vec<String>, Vec<String>),
    (Vec<String>, Vec<String>, Vec<usize>),
) {
    let model = crate::parser::model_parser::parse_model_string(src).expect("parse");
    let subject = censored_subject(cmts);
    let theta = model.default_params.theta.clone();
    let rows_of = |rows: &[crate::api::SimulationResult]| {
        rows.iter()
            .map(|r| format!("{} {} {:?}", r.cmt, r.time.to_bits(), r.outcome))
            .collect::<Vec<_>>()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(1648);
    let (mut rows, mut warnings) = (Vec::new(), Vec::new());
    simulate_tte(
        &model,
        &subject,
        &theta,
        &[eta],
        1,
        1,
        None,
        &mut rng,
        &mut rows,
        &mut warnings,
    );
    let public = (rows_of(&rows), warnings);

    let mut rng = rand::rngs::StdRng::seed_from_u64(1648);
    let (mut rows, mut warnings, mut degenerate) = (Vec::new(), Vec::new(), Vec::new());
    simulate_tte_reporting(
        &model,
        &subject,
        &theta,
        &[eta],
        1,
        1,
        None,
        &mut rng,
        &mut rows,
        &mut warnings,
        &mut degenerate,
    );
    (public, (rows_of(&rows), warnings, degenerate))
}

/// U5: the public `simulate_tte`, called directly, still gives one line per degenerate
/// cause of the call, in today's text; the reporting form gives the same rows and the CMT
/// as a value, with no line.
///
/// Mutations: the wrapper drops its line or rewords it (first block); the reporting form
/// pushes a line as well, or reports nothing (second block).
#[test]
fn the_public_simulate_tte_keeps_its_per_call_line() {
    // η = −0.2 gives a rate of −0.1: degenerate.
    let (public, reporting) = both_forms(ZERO_RATE, &[2], -0.2);
    assert_eq!(public.1, vec![line("", "")], "one line, today's text");
    assert_eq!(public.0.len(), 1, "one censor row: {:?}", public.0);
    assert_eq!(reporting.0, public.0, "the same rows from the same seed");
    assert!(reporting.1.is_empty(), "no line: {:?}", reporting.1);
    assert_eq!(reporting.2, vec![2]);

    // η = 0 gives a rate of 0.1: nothing to report on either form.
    let (public, reporting) = both_forms(ZERO_RATE, &[2], 0.0);
    assert!(public.1.is_empty(), "{:?}", public.1);
    assert_eq!(reporting.0, public.0);
    assert!(reporting.1.is_empty() && reporting.2.is_empty());

    // Competing risks, both causes degenerate: one line per CMT, in record order.
    let (public, reporting) = both_forms(ZERO_RATE_TWO_CAUSES, &[2, 3], -0.2);
    assert_eq!(
        public.1,
        vec![line("", ""), line("", "").replace("(CMT=2)", "(CMT=3)"),]
    );
    assert_eq!(reporting.0, public.0);
    assert!(reporting.1.is_empty());
    assert_eq!(reporting.2, vec![2, 3]);
}
