//! A level `TIME` key reads the time as written, not the engines' ordering clock (#1959,
//! #1796). The reader moves an observation one ULP below a co-timed `MDV=1` / `EVID=2`
//! row listed after it (#1810), and a stacked-reset subject's later periods run on a
//! shifted clock. Neither may change which level a record is, or what a message calls
//! its time.

use super::*;
use crate::api::validation::{check_level_index_columns, LevelDataEntry};
use crate::api::{bind_from_fit_on, bind_theta_levels};
use crate::parser::model_parser::DataBindings;
use crate::types::ParsedModel;

/// Where a co-timed row with no observation sits in the file, if anywhere.
#[derive(Clone, Copy, Debug)]
enum Extra {
    None,
    /// After the observation at `TIME = 2`: the reader moves the observation (#1810).
    After(&'static str),
    /// Before it: nothing moves.
    Before(&'static str),
}

/// The co-timed rows, as `EVID,AMT,CMT,MDV`: an `MDV=1` observation row and an
/// `EVID=2` other-type event.
const KINDS: [&str; 2] = ["0,.,1,1", "2,.,1,1"];

const HEADER: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV,STUDY\n";

/// Studies 1–2, one subject each: a dose, observations at 1, 2 and 4, and `extra`
/// at `TIME = 2`.
fn csv(extra: Extra) -> String {
    let mut rows = String::from(HEADER);
    for id in 1..=2u32 {
        rows.push_str(&format!("{id},0,.,1,100,1,1,{id}\n"));
        for (t, dv) in [(1, 4.6), (2, 4.1), (4, 3.3)] {
            let row = |kind: &str| format!("{id},{t},.,{kind},{id}\n");
            if let (2, Extra::Before(kind)) = (t, extra) {
                rows.push_str(&row(kind));
            }
            rows.push_str(&format!("{id},{t},{dv},0,.,1,0,{id}\n"));
            if let (2, Extra::After(kind)) = (t, extra) {
                rows.push_str(&row(kind));
            }
        }
    }
    rows
}

/// Parse `text` and read `csv` for it through the file reader, which is what moves an
/// observation.
fn read(text: &str, csv: &str) -> (ParsedModel, Population) {
    use std::io::Write as _;
    let parsed = parse_full_model(text).unwrap();
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let pop = crate::api::read_population_for(&parsed.model, &None, &path, None, None, None, &[])
        .expect("fixture data reads")
        .0;
    (parsed, pop)
}

/// Whether any observation of `pop` sits off its written time — the straddle.
fn moved(pop: &Population) -> bool {
    pop.subjects.iter().any(|s| {
        (0..s.obs_times.len()).any(|j| s.obs_times[j].to_bits() != s.readout_time(j).to_bits())
    })
}

/// Per subject, the level index each observation carries.
fn obs_indices(pop: &Population) -> Vec<Vec<f64>> {
    let col = level_index_column("PLACEBO");
    pop.subjects
        .iter()
        .map(|s| s.obs_covariates.iter().map(|m| m[&col]).collect())
        .collect()
}

fn decl(parsed: &ParsedModel) -> LevelBlockDecl {
    parsed.model.theta_blocks().level_blocks()[0].clone()
}

/// The cells of a population with a co-timed `MDV=1` / `EVID=2` row are the cells of
/// the population without it, whether the row is listed after the observation (which
/// the reader moves a ULP below `TIME = 2`) or before it (which it does not) — under
/// global sum-to-zero (`no_eta_model`) and within (`mbma_model`), where the count of
/// free θ is the parametrisation, not only the labels.
///
/// Mutation — `column_value` reads `obs_times` for `TIME` (the pre-fix code): the
/// `After` arms grow a `TIME=1.9999999999999998` cell per study and die on the names.
/// The `Before` arms are the control: they bind the same either way.
#[test]
fn a_co_timed_row_without_an_observation_adds_no_level_cell() {
    for text in [no_eta_model(), mbma_model("")] {
        let (mut base, mut base_pop) = read(&text, &csv(Extra::None));
        bind_theta_levels(&mut base, &text, &mut base_pop).expect("base binds");
        assert!(!moved(&base_pop));
        let names = theta_names(&base.model);
        assert!(
            names.contains(&"PLACEBO[STUDY=1,TIME=1]".to_string()),
            "{names:?}"
        );
        for kind in KINDS {
            for extra in [Extra::After(kind), Extra::Before(kind)] {
                let (mut parsed, mut pop) = read(&text, &csv(extra));
                // The straddle: only a row listed after the observation moves it.
                assert_eq!(moved(&pop), matches!(extra, Extra::After(_)), "{extra:?}");
                bind_theta_levels(&mut parsed, &text, &mut pop).expect("binds");
                assert_eq!(theta_names(&parsed.model), names, "{extra:?}");
                assert_eq!(parsed.model.n_theta, base.model.n_theta, "{extra:?}");
                assert_eq!(obs_indices(&pop), obs_indices(&base_pop), "{extra:?}");
                assert_eq!(level_index_finding(&decl(&parsed), &pop), None, "{extra:?}");
            }
        }
    }
}

/// The from-fit binder finds the fit's cells on a population whose observations were
/// moved, against the bindings of a fit on the unmoved data. Every from-fit path binds
/// through `bind_from_fit_on`: `bind_from_fit` (simulate / predict on a fit, the R
/// glue), SIR and the standalone covariance step (`fit_inputs`), and a reloaded
/// `.fitrx` — so this one call stands for them all.
///
/// Mutation — `obs_times` in `column_value`: the moved population shows a level the
/// fit never saw, and the binder refuses it as unseen.
#[test]
fn the_from_fit_binder_binds_the_fits_cells_on_moved_observations() {
    for text in [no_eta_model(), mbma_model("")] {
        let (mut fit, mut fit_pop) = read(&text, &csv(Extra::None));
        bind_theta_levels(&mut fit, &text, &mut fit_pop).expect("fit binds");
        let fitted = DataBindings {
            levels: fit.bindings.levels.clone(),
            ..Default::default()
        };
        for kind in KINDS {
            let (mut sim, mut sim_pop) = read(&text, &csv(Extra::After(kind)));
            assert!(moved(&sim_pop));
            bind_from_fit_on(&mut sim, &text, Some(&mut sim_pop), &fitted)
                .expect("the from-fit binder");
            assert_eq!(sim.model.theta_names, fit.model.theta_names);
            assert_eq!(obs_indices(&sim_pop), obs_indices(&fit_pop));
        }
    }
}

/// Studies 1–2; subject 1 runs a second period after an `EVID=4` reset that restarts
/// `TIME`, so period 2's internal clock is shifted while its written times are 1 and 2.
fn reset_csv() -> String {
    let mut rows = String::from(HEADER);
    for id in 1..=2u32 {
        rows.push_str(&format!("{id},0,.,1,100,1,1,{id}\n"));
        rows.push_str(&format!("{id},1,4.6,0,.,1,0,{id}\n"));
        rows.push_str(&format!("{id},2,4.1,0,.,1,0,{id}\n"));
        if id == 1 {
            rows.push_str(&format!("{id},0,.,4,100,1,1,{id}\n"));
            rows.push_str(&format!("{id},1,4.5,0,.,1,0,{id}\n"));
            rows.push_str(&format!("{id},2,4.0,0,.,1,0,{id}\n"));
        }
    }
    rows
}

/// #1796. On a stacked-reset subject the misindexed-record message names the time as
/// written, and so does the label's `TIME=`: "at time 2 is level `STUDY=1,TIME=2`",
/// where the label used to read the shifted clock. The two periods share their
/// written-time cells, as NONMEM's `$PRED` `TIME` does.
///
/// Mutations — `obs_times` in `column_value`: the bind grows period 2's shifted cells
/// and the label names one. `obs_times[j]` for the message's `time`: "at time" names
/// the shifted value.
#[test]
fn the_misindex_message_reads_the_written_time_on_a_stacked_reset_subject() {
    let text = no_eta_model();
    let (mut parsed, mut pop) = read(&text, &reset_csv());
    bind_theta_levels(&mut parsed, &text, &mut pop).expect("binds");
    let levels = theta_names(&parsed.model)
        .iter()
        .filter(|n| n.starts_with("PLACEBO["))
        .count();
    assert_eq!(
        levels, 3,
        "2 studies × 2 written times, one derived: {:?}",
        parsed.model.theta_names
    );
    // The straddle: period 2's observation at written TIME 2 is on a shifted clock.
    let s1 = &pop.subjects[0];
    let j = (0..s1.obs_times.len())
        .rev()
        .find(|&j| s1.readout_time(j) == 2.0)
        .unwrap();
    assert!(s1.obs_times[j] > 2.0, "{:?}", s1.obs_times);
    assert_eq!(
        level_index_finding(&decl(&parsed), &pop),
        None,
        "bound as written"
    );

    let col = level_index_column("PLACEBO");
    let want = pop.subjects[0].obs_covariates[j][&col];
    let wrong = if want == 1.0 { 2.0 } else { 1.0 };
    pop.subjects[0].obs_covariates[j].insert(col, wrong);
    let d = check_level_index_columns(&parsed.model, &pop, LevelDataEntry::Run);
    assert_eq!(d.len(), 1, "{d:?}");
    let record = format!(
        "subject 1 at time 2 is level `STUDY=1,TIME=2`, level {want} of the model's block, \
         and its index says {wrong}."
    );
    assert!(d[0].message.contains(&record), "{}", d[0].message);
}

/// `one_instant`: two records are one instant when their written times agree and the
/// engine clock separates them by at most the reader's one-ULP move — not when only the
/// written times agree (a reset period that restarts `TIME`). Both sides on
/// `resolves_observations`, which asks it of every pair sharing a level.
///
/// Mutations — compare `obs_times` exactly (the pre-fix test): the moved replicate
/// stops resolving. Drop the engine-clock clause: the reset subject resolves.
#[test]
fn one_instant_tolerates_the_move_and_not_a_reset() {
    let text = no_eta_model();
    // Two replicates at TIME 2 with an MDV=1 row between them: the first is moved.
    let mut rows = String::from(HEADER);
    for id in 1..=2u32 {
        rows.push_str(&format!("{id},0,.,1,100,1,1,{id}\n"));
        rows.push_str(&format!("{id},1,4.6,0,.,1,0,{id}\n"));
        rows.push_str(&format!("{id},2,4.1,0,.,1,0,{id}\n"));
        rows.push_str(&format!("{id},2,.,0,.,1,1,{id}\n"));
        rows.push_str(&format!("{id},2,4.2,0,.,1,0,{id}\n"));
    }
    let (mut parsed, mut pop) = read(&text, &rows);
    bind_theta_levels(&mut parsed, &text, &mut pop).expect("binds");
    let s = &pop.subjects[0];
    assert_eq!(s.obs_times[1], 2.0f64.next_down(), "{:?}", s.obs_times);
    assert_eq!(s.obs_times[2], 2.0, "{:?}", s.obs_times);
    assert!(one_instant(s, 1, 2));
    assert!(
        resolves_observations(&decl(&parsed), &pop),
        "the moved replicate"
    );

    let (mut parsed, mut pop) = read(&text, &reset_csv());
    bind_theta_levels(&mut parsed, &text, &mut pop).expect("binds");
    let s = &pop.subjects[0];
    assert_eq!(
        s.readout_time(1),
        s.readout_time(3),
        "{:?}",
        s.obs_raw_times
    );
    assert_ne!(s.obs_times[1], s.obs_times[3]);
    assert!(!one_instant(s, 1, 3));
    assert!(
        !resolves_observations(&decl(&parsed), &pop),
        "the reset periods"
    );
}

/// The dead-level advice's "only records at `TIME = 0`" reads the written time: a
/// `TIME = 0` record moved a ULP below 0 (an observation listed before a co-timed
/// dose, #614) is still at `TIME = 0`. The twin is the same population unmoved.
///
/// Mutation — `obs_times[j] == 0.0`: the moved twin loses the sentence.
#[test]
fn the_time_zero_advice_reads_the_written_time() {
    use super::readout_share::{cf_model, cf_pop, BASE, EMAXY, T6};
    const AT_ZERO: &str = " Every one of them holds only records at `TIME = 0`.";
    let text = cf_model(
        "sum_to_zero",
        "STUDY, TIME",
        &format!("{BASE}  E0 = TVE0 + PLACEBO * TIME\n  Z = TVE0 * exp(ETA_E0)"),
        &format!("E0 + {EMAXY}"),
    );
    let still = cf_pop(3, 1, &T6);
    let mut moved = still.clone();
    for s in &mut moved.subjects {
        s.obs_raw_times = s.obs_times.clone();
        s.obs_times[0] = 0.0f64.next_down();
    }
    for (tag, mut pop) in [("still", still), ("moved", moved)] {
        let mut parsed = parse_full_model(&text).unwrap();
        let err = bind_theta_levels(&mut parsed, &text, &mut pop)
            .expect_err(tag)
            .to_string();
        assert!(err.contains(AT_ZERO), "{tag}: {err}");
    }
}
