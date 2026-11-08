//! A θ level block read by a discrete-state endpoint (#2562): the records of every
//! endpoint whose expression reads the block define its levels, as the Gaussian
//! observations do, and each carries its own level index in its covariate snapshot.

use super::*;
use crate::api::levels::{level_index_finding, LevelIndexFinding};
use crate::types::{ObsRecord, ParsedModel};

/// The binary MBMA shape of #2111's B1, small: one subject per study, an arm covariate
/// `TRT` that changes within the subject, the block on the log-odds.
const BINARY: &str = r#"
[parameters]
  theta TVL(-0.3, -5.0, 5.0)
  theta EMAX(0.5, -5.0, 5.0)
  theta PLACEBO[STUDY, TIME, contrast = sum_to_zero_within](0.0)
  omega ETA_S ~ 0.25
[covariates]
  STUDY categorical
  TRT   continuous
[binary_model]
  cmt   = 1
  logit = TVL + PLACEBO + ETA_S + EMAX * TRT
"#;

/// [`BINARY`] without the arm term, so no covariate varies within a subject and the
/// reader writes no per-record snapshot: the binder has to make them.
fn no_arm() -> String {
    BINARY
        .replace(" + EMAX * TRT", "")
        .replace("  theta EMAX(0.5, -5.0, 5.0)\n", "")
}

const HEADER: &str = "ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT\n";

/// Three studies (ID = study), visits at `times`, two arms, four Bernoulli rows per
/// arm-visit with a pattern that varies by study, visit and arm, so no level and no arm
/// is degenerate. Rows are written visit-major in `times` order, not sorted. Without
/// `arms` every row carries `TRT = 0`, so no covariate varies within a subject.
fn csv(times: &[u32], arms: bool) -> String {
    let mut rows = String::from(HEADER);
    for s in 1..=3u32 {
        for (v, &t) in times.iter().enumerate() {
            for trt in 0..=1u32 {
                for k in 0..4u32 {
                    let dv = u32::from((s + v as u32 + trt + k) % 3 == 0 || k == trt);
                    let x = if arms { trt } else { 0 };
                    rows.push_str(&format!("{s},{s},{t},{dv},1,0,0,{x}\n"));
                }
            }
        }
    }
    rows
}

/// Parse `text` and read `csv` for it through the file reader, which is what writes the
/// per-record snapshots and routes the rows to `obs_records`.
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

fn bound(text: &str, csv: &str) -> Result<(ParsedModel, Population), String> {
    let (mut parsed, mut pop) = read(text, csv);
    crate::api::bind_theta_levels(&mut parsed, text, &mut pop).map_err(|e| e.to_string())?;
    Ok((parsed, pop))
}

/// Per record of `subject`, `(STUDY, TIME as written, index)`.
fn record_levels(subject: &Subject) -> Vec<(f64, f64, f64)> {
    let col = level_index_column("PLACEBO");
    let snaps = subject.record_covariate_snapshots();
    subject
        .obs_records
        .iter()
        .enumerate()
        .map(|(r, rec)| {
            let ObsRecord::DiscreteState { raw_time, .. } = rec else {
                panic!("binary fixture: discrete records only");
            };
            let cov = subject.record_cov(snaps, r);
            (cov["STUDY"], *raw_time, cov[&col])
        })
        .collect()
}

/// L1 + L2. On binary-only data with no time-varying covariate, the block's levels are
/// the studies × visits the binary records show, and every record carries the index of
/// its own level in a snapshot the binder made (the reader made none). Each label's
/// index is its position in the bound labels.
///
/// Mutations: drop the discrete records from `level_rows` and the binding is refused
/// as read by no endpoint through its implicit index (#2623); write the index into the
/// baseline only and the visit-4 records read visit 0's index; skip materialising the snapshots and none
/// is aligned.
#[test]
fn binary_records_define_the_levels_and_each_carries_its_own() {
    let text = no_arm();
    let data = csv(&[0, 4], false);
    let (_, unbound) = read(&text, &data);
    assert!(
        unbound.subjects[0].endpoint_covariates.is_empty(),
        "fixture precondition: the reader writes no snapshot without a time-varying covariate"
    );
    let (parsed, pop) = bound(&text, &data).expect("binary records define the levels");
    let labels = parsed.model.theta_blocks().level_blocks()[0]
        .labels()
        .to_vec();
    assert_eq!(
        labels,
        [1, 2, 3]
            .iter()
            .flat_map(|s| [0, 4].map(|t| format!("STUDY={s},TIME={t}")))
            .collect::<Vec<_>>()
    );
    for subject in &pop.subjects {
        assert!(
            subject.record_covariate_snapshots().is_some(),
            "subject {}: the binder materialises aligned snapshots",
            subject.id
        );
        let levels = record_levels(subject);
        assert_eq!(levels.len(), 16, "subject {}", subject.id);
        for (study, time, index) in levels {
            let want = labels
                .iter()
                .position(|l| *l == format!("STUDY={study},TIME={time}"))
                .unwrap()
                + 1;
            assert_eq!(index, want as f64, "subject {} TIME {time}", subject.id);
        }
    }
}

/// L1 with a time-varying covariate: the reader's own snapshots carry the index, and
/// `TRT` survives in them. This is B1's shape, which needed both halves of #2562.
#[test]
fn binary_records_carry_the_index_beside_their_own_covariates() {
    let (_, pop) = bound(BINARY, &csv(&[0, 4, 8], true)).expect("B1's shape binds");
    let s = &pop.subjects[1];
    let snaps = s.record_covariate_snapshots().expect("aligned");
    let trts: Vec<f64> = snaps.iter().map(|m| m["TRT"]).collect();
    assert!(
        trts.contains(&0.0) && trts.contains(&1.0),
        "TRT kept: {trts:?}"
    );
    let indices: Vec<f64> = record_levels(s).iter().map(|l| l.2).collect();
    assert_eq!(
        {
            let mut u = indices.clone();
            u.dedup();
            u
        },
        vec![4.0, 5.0, 6.0],
        "study 2's three visits, in file order"
    );
}

/// L3. A discrete record's `TIME` key is the time as written. After an `EVID=3` reset
/// the reader runs the records on a shifted clock; their level must still be the written
/// visit, so both periods share the visit-0 and visit-4 levels.
///
/// Mutation: key a record on its engine `time` and the second period's visits become
/// new levels (`TIME=` a shifted value).
#[test]
fn a_record_after_a_reset_keys_on_its_written_time() {
    let mut data = String::from(HEADER);
    for s in 1..=3u32 {
        for period in 0..2u32 {
            if period == 1 {
                data.push_str(&format!("{s},{s},0,.,1,3,1,0\n"));
            }
            for t in [0u32, 4] {
                for k in 0..4u32 {
                    let dv = u32::from((s + t + k + period) % 2 == 0);
                    data.push_str(&format!("{s},{s},{t},{dv},1,0,0,0\n"));
                }
            }
        }
    }
    let text = no_arm();
    let (_, raw) = read(&text, &data);
    let shifted = raw.subjects[0]
        .obs_records
        .iter()
        .any(|r| matches!(r, ObsRecord::DiscreteState { time, raw_time, .. } if time != raw_time));
    assert!(
        shifted,
        "fixture precondition: the reset shifts the engine clock"
    );
    let (parsed, _) = bound(&text, &data).expect("binds");
    let labels = parsed.model.theta_blocks().level_blocks()[0]
        .labels()
        .to_vec();
    assert_eq!(labels.len(), 6, "3 studies × 2 written visits: {labels:?}");
    assert!(labels
        .iter()
        .all(|l| l.ends_with("TIME=0") || l.ends_with("TIME=4")));
}

/// F1. The index check re-derives each binary record's index from its own level and
/// compares the one the predictor reads, naming the record's written time.
///
/// Mutation: let the check skip discrete records and the edited index goes unseen.
#[test]
fn the_index_check_reads_each_binary_record() {
    let (parsed, mut pop) = bound(&no_arm(), &csv(&[0, 4], false)).expect("binds");
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert_eq!(level_index_finding(decl, &pop), None, "as bound");
    let col = level_index_column("PLACEBO");
    let s = &mut pop.subjects[2];
    let r = s.obs_records.len() - 1;
    s.endpoint_covariates[r].insert(col, 1.0);
    match level_index_finding(decl, &pop) {
        Some(LevelIndexFinding::Misindexed {
            id,
            time,
            label,
            want,
            got,
        }) => {
            assert_eq!(id, "3");
            assert_eq!(time, Some(4.0));
            assert_eq!(label, "STUDY=3,TIME=4");
            assert_eq!((want, got), (6, Some(1.0)));
        }
        other => panic!("an edited binary index is a finding: {other:?}"),
    }
}

/// `contrast = auto` sees the study η the predictor reads beside the block (#1649): the
/// block absorbs it, so auto resolves to the within-study contrast, as on a Gaussian
/// `y`. The η walk has to see the `[binary_model]` predictor for that.
///
/// Mutations: leave the binary predictor out of the coupling walk's readouts and auto
/// resolves to a global sum-to-zero (one free level per study and visit but one); leave
/// it out of the scale-share walk and the block shares no expression with the η.
#[test]
fn auto_sees_the_eta_the_binary_predictor_reads_with_the_block() {
    let text = BINARY.replace(", contrast = sum_to_zero_within", "");
    let (parsed, _) = bound(&text, &csv(&[0, 4, 8], true)).expect("binds");
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert_eq!(decl.contrast(), LevelContrast::SumToZeroWithin);
    assert!(
        decl.shares_scale_with_eta(),
        "the predictor reads the block and the η in one expression"
    );
}

/// A subject that also carries a time-to-event record cannot have one snapshot per
/// record, so its binary records cannot each carry their own level: refused, naming
/// both levels, rather than scored at the first. Both sides of the gate in one test:
/// subjects with one level each on all their binary records bind, each at its own.
///
/// Mutations: drop the refusal and the first binding succeeds; refuse every unaligned
/// subject and the second is refused.
#[test]
fn a_subject_with_tte_records_is_refused_only_when_its_levels_differ() {
    let text = r#"
[parameters]
  theta LAM(0.1, 0.001, 10.0)
  theta PLACEBO[STUDY, TIME, contrast = none](0.0)
[covariates]
  STUDY categorical
[binary_model]
  cmt   = 1
  logit = PLACEBO
[event_model]
  cmt    = 2
  family = exponential
  scale  = LAM
"#;
    let state = |time: f64, state: usize| ObsRecord::DiscreteState {
        time,
        raw_time: time,
        state,
        cmt: 1,
    };
    let subject = |id: u32, study: f64, times: [f64; 2]| {
        let mut s = Subject {
            id: id.to_string(),
            ..Default::default()
        };
        s.covariates.insert("STUDY".into(), study);
        s.obs_records = vec![
            state(times[0], 1),
            state(times[0], 0),
            state(times[1], 1),
            state(times[1], 0),
            ObsRecord::Event {
                time: 9.0,
                event_type: crate::types::EventType::RightCensored,
                entry_time: 0.0,
                cmt: 2,
            },
        ];
        s
    };
    let population = |subjects: Vec<Subject>| Population {
        subjects,
        covariate_names: vec!["STUDY".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut p = parse_full_model(text).unwrap();
    let mut pop = population(vec![subject(1, 1.0, [0.0, 4.0])]);
    let err = crate::api::bind_theta_levels(&mut p, text, &mut pop)
        .expect_err("one subject, two levels, no aligned snapshot")
        .to_string();
    assert_eq!(
        err,
        "theta PLACEBO[...]: subject 1 has records its per-record covariate snapshots cannot \
         be lined up with (it also carries time-to-event records, or was built in memory with \
         snapshots of another length), so its records cannot each carry their own level of the \
         block, and they are at more than one (STUDY=1,TIME=0 and, at record 2, \
         STUDY=1,TIME=4). Key the block on columns that are constant within the subject, or \
         fit the time-to-event endpoint separately."
    );
    let mut p = parse_full_model(text).unwrap();
    let mut pop = population(vec![
        subject(1, 1.0, [0.0, 0.0]),
        subject(2, 2.0, [4.0, 4.0]),
    ]);
    crate::api::bind_theta_levels(&mut p, text, &mut pop).expect("one level per subject binds");
    let col = level_index_column("PLACEBO");
    let got: Vec<f64> = pop.subjects.iter().map(|s| s.covariates[&col]).collect();
    assert_eq!(
        got,
        vec![1.0, 2.0],
        "each at its own level, in the baseline"
    );
}

/// Three studies (one subject each) of state records on CMT 5 at visits 0, 1, 2, 3,
/// alternating states so every transition is observed.
#[cfg(feature = "markov")]
fn state_population() -> Population {
    let subjects = (1..=3u32)
        .map(|s| {
            let mut subject = Subject {
                id: s.to_string(),
                ..Default::default()
            };
            subject.covariates.insert("STUDY".into(), f64::from(s));
            subject.obs_records = (0..8u32)
                .map(|k| {
                    let t = f64::from(k / 2);
                    ObsRecord::DiscreteState {
                        time: t,
                        raw_time: t,
                        state: ((k + s) % 2) as usize,
                        cmt: 5,
                    }
                })
                .collect();
            subject
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["STUDY".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// M. A DTMM transition probability reads each record's own snapshot (#2473), so its
/// records define the levels of a block it reads, each carrying its own index, as a
/// binary predictor's do.
///
/// Mutation: leave the DTMM arm out of `level_record_cmts` and the binding is refused
/// as read by no endpoint through its implicit index.
#[cfg(feature = "markov")]
#[test]
fn dtmm_records_define_the_levels_of_a_block_their_probabilities_read() {
    let text = r"
[parameters]
  theta A(-0.8, -10.0, 10.0)
  theta PLACEBO[STUDY, TIME, contrast = sum_to_zero_within](0.0)
[covariates]
  STUDY categorical
[markov_model]
  type   = dtmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = inv_logit(A + PLACEBO)
  transition asleep -> awake  = inv_logit(A)
";
    let mut parsed = parse_full_model(text).unwrap();
    let mut pop = state_population();
    crate::api::bind_theta_levels(&mut parsed, text, &mut pop)
        .unwrap_or_else(|e| panic!("DTMM records define the levels: {e}"));
    let labels = parsed.model.theta_blocks().level_blocks()[0]
        .labels()
        .to_vec();
    assert_eq!(labels.len(), 12, "3 studies × 4 visits: {labels:?}");
    for subject in &pop.subjects {
        let levels = record_levels(subject);
        for (study, time, index) in levels {
            let want = labels
                .iter()
                .position(|l| *l == format!("STUDY={study},TIME={time}"))
                .unwrap()
                + 1;
            assert_eq!(index, want as f64, "subject {} TIME {time}", subject.id);
        }
    }
}

/// The levels a subject's objective reads skip a DTMM chain's first record (#2111): the
/// likelihood scores each transition into a record from the one before, so the first
/// record names a level (it defines one, above) that no term reads. Each study's chain
/// here starts with the only record at visit 0, and a Gaussian `y` reads the block at
/// visit 3. So the visit-0 level is defined but read by no subject, and the `rsr`
/// single-subject count names the other nine levels, each read by its own study alone.
/// The Gaussian reader is what makes the count reachable: a block read only by a DTMM has
/// no plain index column (`read_index`), so every subject is given every level.
///
/// Mutations: count the chain's first record and all twelve are named; leave DTMM records
/// out of the count and only the three visit-3 levels are.
#[cfg(feature = "markov")]
#[test]
fn a_dtmm_chains_first_record_reads_no_level() {
    let text = r"
[parameters]
  theta A(-0.8, -10.0, 10.0)
  theta B(1.0, -10.0, 10.0)
  theta PLACEBO[STUDY, TIME, contrast = sum_to_zero_within](0.0)
  sigma ADD ~ 1.0
[covariates]
  STUDY categorical
[individual_parameters]
  BASE = B + PLACEBO
[structural_model]
  y = BASE
[error_model]
  DV ~ additive(ADD)
[markov_model]
  type   = dtmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = inv_logit(A + PLACEBO)
  transition asleep -> awake  = inv_logit(A)
";
    let mut parsed = parse_full_model(text).unwrap();
    let mut pop = state_population();
    for s in &mut pop.subjects {
        s.obs_records.remove(0);
        s.obs_times = vec![3.0];
        s.observations = vec![1.0];
        s.obs_cmts = vec![1];
        s.cens = vec![0];
    }
    crate::api::bind_theta_levels(&mut parsed, text, &mut pop)
        .unwrap_or_else(|e| panic!("DTMM records define the levels: {e}"));
    let found = crate::api::levels::one_subject_levels(&parsed.model, &pop);
    assert_eq!(found.len(), 1, "{found:?}");
    let (block, single, n) = &found[0];
    assert_eq!((block.as_str(), *n), ("PLACEBO", 12));
    assert_eq!(single.len(), 9, "{single:?}");
    assert!(single.iter().all(|l| !l.ends_with("TIME=0")), "{single:?}");
}

/// A CTMM generator is built once per subject from the baseline, so its records define
/// no levels (#2562), and a block whose index changes within a subject (here between two
/// Gaussian observations) is refused by `E_ENDPOINT_TV_COVARIATE`, naming the block
/// rather than its synthesized `__level_` column. The control: an index constant within
/// the subject passes.
///
/// Mutation: name the input as a covariate and the message names `__level_PLACEBO`.
#[cfg(feature = "markov")]
#[test]
fn a_ctmm_generator_reading_a_changing_level_is_refused_naming_the_block() {
    let model = parse_full_model(
        r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta PLACEBO[STUDY, TIME](0.0)
[covariates]
  STUDY categorical
[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + PLACEBO)
  transition asleep -> awake  = exp(LQ10)
",
    )
    .unwrap()
    .model;
    assert!(
        model.theta_blocks().level_blocks()[0]
            .record_cmts
            .is_empty(),
        "a CTMM's records define no levels"
    );
    let col = level_index_column("PLACEBO");
    let population = |second: f64| {
        let mut s = Subject {
            id: "1".into(),
            obs_times: vec![0.0, 1.0],
            observations: vec![0.0; 2],
            obs_cmts: vec![1; 2],
            cens: vec![0; 2],
            ..Default::default()
        };
        s.covariates.insert(col.clone(), 1.0);
        s.obs_covariates = vec![
            HashMap::from([(col.clone(), 1.0)]),
            HashMap::from([(col.clone(), second)]),
        ];
        Population {
            subjects: vec![s],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    };
    let refused = crate::api::check_survival_tv_covariates(&model, &population(2.0));
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(
        refused[0].message,
        "[markov_model] on CMT=5: a transition intensity references θ level block `PLACEBO`, \
         whose level changes between records, which is time-varying for subject `1`. \
         Time-varying covariates on the generator are not yet supported — the generator is \
         evaluated at the baseline covariate value (time-homogeneous CTMM), so the fit would \
         silently use the wrong intensity (#741). A drug-driven Q(t) is Phase 6; use a \
         time-constant (baseline) covariate for now."
    );
    assert!(
        crate::api::check_survival_tv_covariates(&model, &population(1.0)).is_empty(),
        "an index constant within the subject passes"
    );
}

/// The binary arm of `E_ENDPOINT_TV_COVARIATE` names a level block by its name, and its
/// remedy is to re-key the block, not to "hold `__level_PLACEBO` constant". Reached by
/// an in-memory subject whose records carry a changing index but whose snapshots cannot
/// align (it also has a time-to-event record); the binder refuses such a subject first.
///
/// Mutations: name the input as a covariate, or give the covariate remedy, and the text
/// dies.
#[test]
fn the_binary_refusal_names_a_changing_level_block_and_how_to_re_key_it() {
    let model = parse_full_model(&no_arm()).unwrap().model;
    let col = level_index_column("PLACEBO");
    let mut s = Subject {
        id: "7".into(),
        ..Default::default()
    };
    s.covariates.insert(col.clone(), 1.0);
    s.obs_records = vec![
        ObsRecord::DiscreteState {
            time: 0.0,
            raw_time: 0.0,
            state: 1,
            cmt: 1,
        },
        ObsRecord::Event {
            time: 9.0,
            event_type: crate::types::EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        },
    ];
    s.endpoint_covariates = vec![
        HashMap::from([(col.clone(), 1.0)]),
        HashMap::from([(col.clone(), 2.0)]),
    ];
    let pop = Population {
        subjects: vec![s],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let refused = crate::api::check_survival_tv_covariates(&model, &pop);
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(
        refused[0].message,
        "[binary_model] on CMT=1: the linear predictor references θ level block `PLACEBO`, \
         whose level changes between records, which is time-varying for subject `7`, but that \
         subject carries no per-record covariate snapshot aligned with its binary records (it \
         also has time-to-event records, or was built in memory without \
         `endpoint_covariates`), so the predictor could only read the baseline value (#741). \
         Key `PLACEBO` on columns that are constant within the subject, or fit the binary \
         endpoint without the time-to-event endpoint."
    );
}

/// #1797's refusal for a subject scored on something that reads the block at a level no
/// record defines says which records define the levels: here the binary records on CMT 1
/// as well as the Gaussian ones, since the TTE-only subject's `STUDY=3` appears on
/// neither. With no reading discrete endpoint the sentence keeps its Gaussian-only form
/// (pinned in `theta_levels_tests`).
///
/// Mutation: drop the `records` clause and the sentence claims the Gaussian records alone.
#[test]
fn the_unindexed_refusal_names_the_records_that_define_levels() {
    let text = r#"
[parameters]
  theta LAM(0.1, 0.001, 10.0)
  theta PLACEBO[STUDY, contrast = none](0.0)
[covariates]
  STUDY categorical
[binary_model]
  cmt   = 1
  logit = PLACEBO
[event_model]
  cmt    = 2
  family = exponential
  scale  = LAM * exp(PLACEBO)
"#;
    let subject = |id: u32, records: Vec<ObsRecord>| {
        let mut s = Subject {
            id: id.to_string(),
            ..Default::default()
        };
        s.covariates.insert("STUDY".into(), f64::from(id));
        s.obs_records = records;
        s
    };
    let state = |state: usize| ObsRecord::DiscreteState {
        time: 0.0,
        raw_time: 0.0,
        state,
        cmt: 1,
    };
    let tte = ObsRecord::Event {
        time: 9.0,
        event_type: crate::types::EventType::Exact,
        entry_time: 0.0,
        cmt: 2,
    };
    let mut pop = Population {
        subjects: vec![
            subject(1, vec![state(1), state(0)]),
            subject(2, vec![state(0), state(0), state(1)]),
            subject(3, vec![tte]),
        ],
        covariate_names: vec!["STUDY".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut parsed = parse_full_model(text).unwrap();
    let err = crate::api::bind_theta_levels(&mut parsed, text, &mut pop)
        .expect_err("subject 3's hazard reads a level no record defines")
        .to_string();
    assert!(
        err.contains(
            " The block's levels are discovered from Gaussian observation records and the \
             records on CMT 1 only, so a level that only such subjects show is not one of them."
        ),
        "{err}"
    );

    // The fit's levels, re-read with subject 3: the same claim about the fit.
    let fit_pop = |subjects: Vec<Subject>| Population {
        subjects,
        ..pop.clone()
    };
    let mut parsed = parse_full_model(text).unwrap();
    // Subject 1 also carries an event, so `LAM` is identified on the fit's data.
    let mut with_event = pop.subjects[..2].to_vec();
    with_event[0]
        .obs_records
        .push(pop.subjects[2].obs_records[0].clone());
    let mut fitted = fit_pop(with_event);
    crate::api::bind_theta_levels(&mut parsed, text, &mut fitted).expect("studies 1 and 2 bind");
    let mut reread = fit_pop(pop.subjects.clone());
    let err = crate::api::levels::write_fitted_level_columns(
        &parsed.model,
        &mut reread,
        &parsed.model.data_bindings().levels,
    )
    .expect_err("the fit has no theta for STUDY=3")
    .message;
    assert!(
        err.contains(
            " A level's theta exists only for a combination the fit's Gaussian observations \
             and the records on CMT 1 showed."
        ),
        "{err}"
    );
}

/// The data-side half of absorption reads the binary records too. A study η that enters
/// only through the arm — on the drug effect, or scaled by `TRT` — moves the two arms of
/// one study-visit level differently, so the block cannot carry it: the records at one
/// level are not replicates (`resolves_observations`), and `TRT` is not constant within
/// a study (`constant_within_units`). Auto then resolves as for a block with no η, to the
/// global sum-to-zero. The control is B1, where the η sits beside the block and is
/// absorbed (`auto_sees_the_eta_the_binary_predictor_reads_with_the_block`).
///
/// Mutations: walk only the Gaussian observations in `resolves_observations` or in
/// `constant_within_units` (both then hold vacuously on binary-only data) and the η reads
/// as absorbed.
#[test]
fn an_eta_on_the_arm_is_not_absorbed_by_a_visit_block() {
    for logit in [
        "TVL + PLACEBO + EMAX * TRT * exp(ETA_S)",
        "TVL + PLACEBO + EMAX * TRT + ETA_S * TRT",
    ] {
        let text = BINARY
            .replace(", contrast = sum_to_zero_within", "")
            .replace("TVL + PLACEBO + ETA_S + EMAX * TRT", logit);
        let (parsed, _) =
            bound(&text, &csv(&[0, 4, 8], true)).unwrap_or_else(|e| panic!("{logit}: {e}"));
        let decl = &parsed.model.theta_blocks().level_blocks()[0];
        assert_eq!(decl.contrast(), LevelContrast::SumToZero, "{logit}");
    }
}

/// A block keyed on the visit alone is shared by every study, so its levels do not nest
/// in the study η's units (`levels_nest_in_units`, on the binary records) and the η beside
/// it is not absorbed: auto resolves to the global sum-to-zero.
///
/// Mutation: walk only the Gaussian observations in `levels_nest_in_units` (it then
/// holds vacuously) and the η reads as absorbed.
#[test]
fn a_visit_block_shared_by_studies_does_not_absorb_the_study_eta() {
    let text = BINARY.replace(
        "PLACEBO[STUDY, TIME, contrast = sum_to_zero_within]",
        "PLACEBO[TIME]",
    );
    let (parsed, _) = bound(&text, &csv(&[0, 4, 8], true)).unwrap_or_else(|e| panic!("{e}"));
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert_eq!(decl.contrast(), LevelContrast::SumToZero);
}

/// A block constant within each subject (keyed on `STUDY`) beside a covariate that does
/// vary (`TRT`): the reader writes a snapshot for every record, and each one must carry
/// the block's index, since that snapshot is what the predictor reads. Before the PR the
/// predictor read the baseline, which has it (PR #2590 review, finding 1).
///
/// Mutation: return before writing the index into existing snapshots when every record
/// is at one level, and the records read index 0 (`NaN`): the binder refuses the block
/// as aliased.
#[test]
fn a_subject_constant_block_reaches_snapshots_the_reader_wrote() {
    let text = BINARY
        .replace(
            "PLACEBO[STUDY, TIME, contrast = sum_to_zero_within]",
            "PLACEBO[STUDY]",
        )
        .replace(" + ETA_S", "")
        .replace("  omega ETA_S ~ 0.25\n", "");
    let (parsed, pop) = bound(&text, &csv(&[0, 4, 8], true)).unwrap_or_else(|e| panic!("{e}"));
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert_eq!(level_index_finding(decl, &pop), None);
    for subject in &pop.subjects {
        assert!(subject.record_covariate_snapshots().is_some(), "TRT varies");
        for (study, _, index) in record_levels(subject) {
            assert_eq!(index, study, "subject {}", subject.id);
        }
    }
}

/// The identifiability refusal names the `[binary_model]` predictor as the site where
/// the block meets the random effect, not `y` (PR #2590 review).
///
/// Mutation: leave the share's `binary` flag false and the message says "the `y`
/// readout".
#[test]
fn the_share_site_names_the_binary_predictor() {
    let text = BINARY.replace(
        "PLACEBO[STUDY, TIME, contrast = sum_to_zero_within]",
        "PLACEBO[STUDY]",
    );
    let err = bound(&text, &csv(&[0, 4], true))
        .err()
        .expect("not identified");
    assert!(
        err.contains("the `[binary_model]` predictor reads this block and a random effect"),
        "{err}"
    );
    assert!(!err.contains("the `y` readout"), "{err}");
}

/// Two records at one written time in two reset periods are not replicates: the reset
/// shifts the engine clock the predictor's `TIME` reads, so an η on `TIME` moves them
/// differently and the block keyed on the written time cannot carry it. Auto resolves as
/// for a block with no η (PR #2590 review, finding 4).
///
/// Mutation: compare only the written time in `records_replicate` and the η reads as
/// absorbed.
#[test]
fn records_in_two_reset_periods_are_not_replicates() {
    let mut data = String::from(HEADER);
    for s in 1..=3u32 {
        for period in 0..2u32 {
            if period == 1 {
                data.push_str(&format!("{s},{s},0,.,1,3,1,0\n"));
            }
            for t in [0u32, 4] {
                for k in 0..4u32 {
                    let dv = u32::from((s + t + k + period) % 2 == 0);
                    data.push_str(&format!("{s},{s},{t},{dv},1,0,0,0\n"));
                }
            }
        }
    }
    let text = no_arm()
        .replace(", contrast = sum_to_zero_within", "")
        .replace("TVL + PLACEBO + ETA_S", "TVL + PLACEBO + ETA_S * TIME");
    let (parsed, _) = bound(&text, &data).unwrap_or_else(|e| panic!("{e}"));
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert_eq!(decl.contrast(), LevelContrast::SumToZero);
}

/// The pre-fit index range check reads each binary record's own snapshot, where the
/// predictor reads the index (PR #2590 review, finding 3): a record at index 3 of a
/// two-level block is refused at its own time, while the baseline (1) is in range. The
/// control: the same subject with every index in range passes.
///
/// Mutation: walk only Gaussian and dose rows in `check_theta_gather_indices` and the
/// out-of-range record passes.
#[test]
fn the_gather_index_check_reads_each_binary_record() {
    let model = parse_full_model(
        r"
[parameters]
  theta TVL(-0.3, -5.0, 5.0)
  theta PLACEBO[2](0.0)
[binary_model]
  cmt   = 1
  logit = TVL + PLACEBO[IDX]
",
    )
    .unwrap()
    .model;
    let population = |second: f64| {
        let idx = |v: f64| HashMap::from([("IDX".to_string(), v)]);
        let mut s = Subject {
            id: "1".into(),
            covariates: idx(1.0),
            endpoint_covariates: vec![idx(1.0), idx(second)],
            ..Default::default()
        };
        s.obs_records = [0.0, 4.0]
            .map(|t| ObsRecord::DiscreteState {
                time: t,
                raw_time: t,
                state: 1,
                cmt: 1,
            })
            .to_vec();
        Population {
            subjects: vec![s],
            covariate_names: vec!["IDX".into()],
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    };
    let codes = |second: f64| -> Vec<(String, String)> {
        crate::api::check_model_data(&model, &population(second))
            .into_iter()
            .filter(|d| d.code.starts_with("E_THETA_GATHER_INDEX"))
            .map(|d| (d.code.to_string(), d.message))
            .collect()
    };
    let refused = codes(3.0);
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0].0, "E_THETA_GATHER_INDEX_RANGE");
    assert!(
        refused[0].1.contains("is 3 on subject `1` at TIME 4"),
        "{}",
        refused[0].1
    );
    assert!(codes(2.0).is_empty(), "in range");
}

/// How the #2623 fixtures read the block: bare (the documented form), as a subscripted
/// `PLACEBO[STUDY]` (refused at parse since #2744), or not at all.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Read {
    Bare,
    Indexed,
    Unread,
}

/// Which observations the #2623 fixtures carry, and so which endpoints the model declares.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Rows {
    Binary,
    Gaussian,
    Mixed,
}

/// A model for `rows` whose every predictor reads `PLACEBO[STUDY]` as `read` says. No
/// random effect, so the one-subject-per-study layout is not refused for identifiability
/// before the binder gets to the levels.
fn read_model(rows: Rows, read: Read) -> String {
    let term = match read {
        Read::Bare => " + PLACEBO",
        Read::Indexed => " + PLACEBO[STUDY]",
        Read::Unread => "",
    };
    let mut text = String::from("[parameters]\n");
    if rows != Rows::Gaussian {
        text.push_str("  theta TVL(-0.3, -5.0, 5.0)\n");
    }
    if rows != Rows::Binary {
        text.push_str("  theta B(1.0, -10.0, 10.0)\n");
    }
    text.push_str("  theta PLACEBO[STUDY](0.0, -5.0, 5.0)\n");
    if rows != Rows::Binary {
        text.push_str("  sigma ADD ~ 1.0\n");
    }
    text.push_str("[covariates]\n  STUDY categorical\n");
    if rows != Rows::Binary {
        text.push_str(&format!(
            "[individual_parameters]\n  BASE = B{term}\n[structural_model]\n  y = BASE\n\
             [error_model]\n  DV ~ additive(ADD)\n"
        ));
    }
    if rows != Rows::Gaussian {
        let cmt = if rows == Rows::Mixed { 2 } else { 1 };
        text.push_str(&format!(
            "[binary_model]\n  cmt   = {cmt}\n  logit = TVL{term}\n"
        ));
    }
    text
}

/// Three studies, one subject each (`STUDY` = ID). Binary rows: six Bernoulli records on
/// the binary endpoint's CMT at TIME 0..5, with a pattern that differs by study. Gaussian
/// rows: three observations on CMT 1. With `records = false` on [`Rows::Binary`] the
/// population is dose-only: no row of either kind.
fn read_population(rows: Rows, records: bool) -> Population {
    let binary_cmt = if rows == Rows::Mixed { 2 } else { 1 };
    let subjects = (1..=3u32)
        .map(|s| {
            let mut subject = Subject {
                id: s.to_string(),
                ..Default::default()
            };
            subject.covariates.insert("STUDY".into(), f64::from(s));
            if records && rows != Rows::Gaussian {
                subject.obs_records = (0..6u32)
                    .map(|k| ObsRecord::DiscreteState {
                        time: f64::from(k),
                        raw_time: f64::from(k),
                        state: usize::from((k + s) % 3 == 0),
                        cmt: binary_cmt,
                    })
                    .collect();
            }
            if rows != Rows::Binary {
                subject.obs_times = vec![1.0, 2.0, 3.0];
                subject.observations = vec![1.0 + f64::from(s), 1.5, 0.5 * f64::from(s)];
                subject.obs_cmts = vec![1; 3];
                subject.cens = vec![0; 3];
            }
            subject
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["STUDY".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Bind [`read_model`] on [`read_population`], returning the refusal as the diagnostic
/// `ferx check --data` reports.
fn bind_read(rows: Rows, read: Read, records: bool) -> Result<(), crate::diagnostics::Diagnostic> {
    let text = read_model(rows, read);
    let mut parsed = parse_full_model(&text).unwrap_or_else(|e| panic!("{rows:?}/{read:?}: {e}"));
    let mut pop = read_population(rows, records);
    crate::api::levels::bind_theta_levels_diag(&mut parsed, &text, &mut pop)
}

/// The sentences of the #2623 refusal, each asserted on its own so deleting any one
/// reddens [`no_reading_endpoint_cells`].
const NO_LEVELS: &str = "theta PLACEBO[STUDY]: the block has no levels.";
const NO_GAUSSIAN: &str = "The data carries no Gaussian observation rows";
const NO_READER: &str = "none of its discrete-state records is on an endpoint whose expression \
                         reads the block through its implicit index";
/// Conditional on no bare read, which since #2744 means nothing reads the block at all.
const UNREAD: &str = "Nothing reads it:";
/// The per-level check's wording for an all-dead block on Gaussian data, a different arm
/// whose block may well be read (`0 * PLACEBO`), so it keeps pointing at the expression.
const CHECK: &str = "Check the expression that reads it";
const READ_BARE: &str = "a data-driven block is read bare, as `PLACEBO`; its index is implicit.";
/// Conditional on a bare read whose only readers define no level (review r1 #1): a
/// Gaussian prediction or a CTMM generator …
const BARE_NO_DEFINER: &str = "The block is read bare, but only by a Gaussian prediction, a \
                               `[markov_model]` CTMM generator or an `[event_model]` hazard, and \
                               none of them defines a level on this data";
const BARE_WHY: &str = "it has no Gaussian rows, a CTMM generator is built once per subject from \
                        the baseline, and one time-to-event row can yield zero or two records, \
                        so neither kind of record defines a level.";
const BARE_ADVICE: &str =
    "Read the block in a `[binary_model]` predictor or a DTMM `[markov_model]` transition.";
/// … or a level-defining endpoint the data has no record for.
const BARE_NO_RECORDS: &str = "The block is read bare by an endpoint whose records define levels \
                               (CMT 1), but the data has no record there.";
/// #2751: the bare-read advice when the data has no discrete-state record, so no reader
/// the model could add defines a level on it.
const BARE_NO_DEFINER_DATA: &str = "Only Gaussian rows and the discrete-state records of a \
                                    `[binary_model]` or DTMM `[markov_model]` endpoint define a \
                                    level, and this data has neither, so it cannot estimate the \
                                    block.";
/// Every bare-read sentence, for the cells where none belongs.
const BARE_ANY: [&str; 5] = [
    BARE_NO_DEFINER,
    BARE_WHY,
    BARE_ADVICE,
    BARE_NO_DEFINER_DATA,
    BARE_NO_RECORDS,
];
/// The pre-#2623 wording, still right when the data has no row at all.
const NO_ROWS: &str = "the data carries no observation rows";

/// #2623. The refusal's input space is data (binary-only, Gaussian-only, mixed) × how
/// the predictors read the block (bare, or not at all — a subscripted read is refused
/// at parse, #2744, see `a_subscripted_read_never_reaches_the_binder`). Only the
/// binary-only data with no bare read has no row that defines a level, and only that
/// cell carries the new refusal. Every other cell is unchanged: bare reads bind, and an
/// unread block on data with Gaussian rows is refused by the per-level checks, which
/// already point at the expression.
///
/// Mutations: drop the discrete-record test and the binary cell says "no observation
/// rows"; delete any sentence of the message and its constant's assertion fails.
#[test]
fn no_reading_endpoint_cells() {
    for rows in [Rows::Binary, Rows::Gaussian, Rows::Mixed] {
        for read in [Read::Bare, Read::Unread] {
            let cell = format!("{rows:?} × {read:?}");
            let got = bind_read(rows, read, true);
            match (rows, read) {
                (Rows::Binary, Read::Unread) => {
                    let d = got.expect_err(&cell);
                    assert_eq!(d.code, "E_THETA_LEVEL_BINDING", "{cell}");
                    let m = &d.message;
                    for sentence in [NO_LEVELS, NO_GAUSSIAN, NO_READER, UNREAD, READ_BARE] {
                        assert!(m.contains(sentence), "{cell}: missing {sentence:?} in {m}");
                    }
                    assert!(!m.contains(NO_ROWS), "{cell}: blames the data: {m}");
                    for sentence in BARE_ANY {
                        assert!(!m.contains(sentence), "{cell}: not read bare: {m}");
                    }
                }
                (_, Read::Indexed) => unreachable!("refused at parse; not iterated"),
                (_, Read::Bare) => {
                    assert!(got.is_ok(), "{cell}: {got:?}");
                }
                (Rows::Gaussian | Rows::Mixed, Read::Unread) => {
                    let m = got.expect_err(&cell).message;
                    assert!(
                        !m.contains("implicit index") && !m.contains(NO_ROWS),
                        "{cell}: Gaussian rows define levels, so the per-level checks refuse: {m}"
                    );
                }
            }
        }
    }
    // The per-level check keeps its own wording on Gaussian data, and not the binder's.
    let m = bind_read(Rows::Gaussian, Read::Unread, true)
        .expect_err("unread on Gaussian data")
        .message;
    assert!(
        m.contains("no level of the block affects the likelihood") && m.contains(CHECK),
        "{m}"
    );
}

/// #2623, the other side of the discrete-record gate: with no row of any kind the data
/// really carries no observation rows, and the refusal says so without blaming the
/// expression.
///
/// Mutation: drop the gate (always blame the expression) and this fails.
#[test]
fn a_population_with_no_rows_still_says_so() {
    let d = bind_read(Rows::Binary, Read::Unread, false).expect_err("no rows");
    assert_eq!(d.code, "E_THETA_LEVEL_BINDING");
    assert!(d.message.contains(NO_ROWS), "{}", d.message);
    for sentence in [NO_GAUSSIAN, NO_READER, UNREAD, READ_BARE]
        .into_iter()
        .chain(BARE_ANY)
    {
        assert!(
            !d.message.contains(sentence),
            "{sentence:?} in {}",
            d.message
        );
    }
}

/// Assert the bare-read refusal of review r1 #1 on `m`: the shared facts, then exactly
/// one bare-read arm — `no_records` picks which — and never the "read it bare" advice,
/// which would tell the user to do what they already did.
fn assert_bare_read_refusal(cell: &str, m: &str, no_records: bool) {
    for sentence in [NO_LEVELS, NO_GAUSSIAN, NO_READER] {
        assert!(m.contains(sentence), "{cell}: missing {sentence:?} in {m}");
    }
    for sentence in [UNREAD, CHECK, READ_BARE, NO_ROWS] {
        assert!(!m.contains(sentence), "{cell}: already read bare: {m}");
    }
    let (want, not): (&[&str], &[&str]) = if no_records {
        (
            &[BARE_NO_RECORDS],
            &[BARE_NO_DEFINER, BARE_WHY, BARE_ADVICE, BARE_NO_DEFINER_DATA],
        )
    } else {
        (
            &[BARE_NO_DEFINER, BARE_WHY, BARE_ADVICE],
            &[BARE_NO_RECORDS, BARE_NO_DEFINER_DATA],
        )
    };
    for sentence in want {
        assert!(m.contains(sentence), "{cell}: missing {sentence:?} in {m}");
    }
    for sentence in not {
        assert!(
            !m.contains(sentence),
            "{cell}: wrong arm, {sentence:?} in {m}"
        );
    }
}

/// Review r1 #1: a block read bare by a Gaussian `y` on binary-only data is read
/// through its implicit index, only not by a reader whose rows define levels — so the
/// refusal says that, not "check the expression". The binary predictor does not read it.
///
/// Mutations: drop `read_bare` (always false) and the cell gets `UNREAD`; route every
/// bare read to the no-records arm and it gets `BARE_NO_RECORDS`.
#[test]
fn a_gaussian_bare_read_on_binary_rows_is_not_blamed_on_the_expression() {
    let text = read_model(Rows::Mixed, Read::Unread).replace("BASE = B\n", "BASE = B + PLACEBO\n");
    let mut parsed = parse_full_model(&text).unwrap();
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert!(
        decl.read_bare && decl.record_cmts.is_empty(),
        "fixture precondition: read bare by `y` alone"
    );
    let mut pop = read_population(Rows::Binary, true);
    for s in &mut pop.subjects {
        for r in &mut s.obs_records {
            if let ObsRecord::DiscreteState { cmt, .. } = r {
                *cmt = 2;
            }
        }
    }
    let d = crate::api::levels::bind_theta_levels_diag(&mut parsed, &text, &mut pop)
        .expect_err("no Gaussian rows");
    assert_eq!(d.code, "E_THETA_LEVEL_BINDING");
    assert_bare_read_refusal("y bare", &d.message, false);
}

/// Review r2 #5: a block read bare by an `[event_model]` hazard alone is read through its
/// implicit index too, but a time-to-event row defines no level (#2562), so the refusal
/// is the bare-read arm that names the hazard, not "check the expression".
///
/// Mutation: drop `read_bare` and the cell gets `UNREAD`; drop the hazard from the arm's
/// reader list and `BARE_NO_DEFINER` is missing.
#[test]
fn a_hazard_bare_read_on_binary_rows_is_not_blamed_on_the_expression() {
    let text = read_model(Rows::Binary, Read::Unread).replace(
        "theta TVL(-0.3, -5.0, 5.0)\n",
        "theta TVL(-0.3, -5.0, 5.0)\n  theta LAM(0.1, 0.001, 10.0)\n",
    )
        + "[event_model]\n  cmt    = 2\n  family = exponential\n  scale  = LAM * exp(PLACEBO)\n";
    let mut parsed = parse_full_model(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let decl = &parsed.model.theta_blocks().level_blocks()[0];
    assert!(
        decl.read_bare && decl.record_cmts.is_empty(),
        "fixture precondition: read bare by the hazard alone"
    );
    let mut pop = read_population(Rows::Binary, true);
    let d = crate::api::levels::bind_theta_levels_diag(&mut parsed, &text, &mut pop)
        .expect_err("a hazard defines no levels");
    assert_eq!(d.code, "E_THETA_LEVEL_BINDING");
    assert_bare_read_refusal("hazard bare", &d.message, false);
}

/// Review r1 #1, the Markov cells: a block read bare by a CTMM generator alone (whose
/// records define no level, #2562), and one read bare by a DTMM on CMT 1 when the data's
/// records are all on another endpoint (CMT 5). Each names its own arm.
///
/// Mutations: drop `read_bare` and both get `UNREAD`; drop the `record_cmts` test and the
/// DTMM cell gets the CTMM arm; swap the arms and each gets the other's sentence.
#[cfg(feature = "markov")]
#[test]
fn a_markov_bare_read_names_why_it_defines_no_level() {
    let ctmm = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta PLACEBO[STUDY](0.0)
[covariates]
  STUDY categorical
[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + PLACEBO)
  transition asleep -> awake  = exp(LQ10)
";
    let mut parsed = parse_full_model(ctmm).unwrap();
    let mut pop = state_population();
    let d = crate::api::levels::bind_theta_levels_diag(&mut parsed, ctmm, &mut pop)
        .expect_err("a CTMM defines no levels");
    assert_eq!(d.code, "E_THETA_LEVEL_BINDING");
    assert_bare_read_refusal("ctmm bare", &d.message, false);

    let dtmm = r"
[parameters]
  theta A(-0.8, -10.0, 10.0)
  theta PLACEBO[STUDY](0.0)
[covariates]
  STUDY categorical
[markov_model reads]
  type   = dtmm
  cmt    = 1
  states = [awake=0, asleep=1]
  transition awake  -> asleep = inv_logit(A + PLACEBO)
  transition asleep -> awake  = inv_logit(A)
[markov_model other]
  type   = dtmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = inv_logit(A)
  transition asleep -> awake  = inv_logit(A)
";
    let mut parsed = parse_full_model(dtmm).unwrap();
    assert_eq!(
        parsed.model.theta_blocks().level_blocks()[0].record_cmts,
        [1],
        "fixture precondition: the CMT-1 DTMM reads the block"
    );
    let mut pop = state_population();
    let d = crate::api::levels::bind_theta_levels_diag(&mut parsed, dtmm, &mut pop)
        .expect_err("no record on CMT 1");
    assert_bare_read_refusal("dtmm bare, records elsewhere", &d.message, true);
}

/// #2744: the subscripted read #2623 named in the binder's no-levels refusal is refused
/// at parse now, on every data shape, so no binder cell can see it. The parse-side
/// message is pinned sentence by sentence in the parser's own tests; this pins its
/// code. A parse refusal, not the binder's `E_THETA_LEVEL_BINDING`: it fires before any
/// data is read.
///
/// Mutation: drop the parse refusal and every row parses.
#[test]
fn a_subscripted_read_never_reaches_the_binder() {
    for rows in [Rows::Binary, Rows::Gaussian, Rows::Mixed] {
        let Err(err) = parse_full_model(&read_model(rows, Read::Indexed)) else {
            panic!("{rows:?}: a subscripted read must not parse");
        };
        assert!(
            err.contains("`PLACEBO[STUDY]`: `PLACEBO` is a data-driven θ level block"),
            "{rows:?}: {err}"
        );
        assert_eq!(
            crate::api::validation::parse_error_to_diagnostic(&err).code,
            "E_PARSE",
            "{rows:?}"
        );
    }
}

/// #2751: the clause naming the data's time-to-event records.
const EVENT_CLAUSE: &str = ", and its time-to-event records define no level";
/// #2751: the clause naming the data's count records.
const COUNT_CLAUSE: &str =
    ", and its count records define no level, since no endpoint scores a count record";

/// The record kinds a #2751 cell's data carries, none of them on an endpoint that reads
/// the block (CMT 3 and up; the reading endpoints are on CMT 1).
#[derive(Debug, Clone, Copy)]
struct Kinds {
    discrete: bool,
    events: bool,
    counts: bool,
}

/// [`read_population`]'s three studies with each subject's records replaced by one of each
/// kind in `kinds`: a discrete-state record on CMT 3, an exact TTE event on CMT 2 and a
/// count on CMT 4. No Gaussian row, so no row defines a level.
fn records_population(kinds: Kinds) -> Population {
    let mut pop = read_population(Rows::Binary, false);
    for s in &mut pop.subjects {
        if kinds.discrete {
            s.obs_records.push(ObsRecord::DiscreteState {
                time: 1.0,
                raw_time: 1.0,
                state: 1,
                cmt: 3,
            });
        }
        if kinds.events {
            s.obs_records.push(ObsRecord::Event {
                time: 2.0,
                event_type: crate::types::EventType::Exact,
                entry_time: 0.0,
                cmt: 2,
            });
        }
        if kinds.counts {
            s.obs_records.push(ObsRecord::Count {
                time: 3.0,
                raw_time: 3.0,
                count: 4,
                cmt: 4,
            });
        }
    }
    pop
}

/// How a #2751 cell's model reads the block: not at all, bare by a level-defining endpoint
/// (a `[binary_model]` on CMT 1, whose CMT the data has no record on), or bare by a reader
/// that defines no level (a Gaussian prediction).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Reader {
    Unread,
    Definer,
    NonDefiner,
}

fn reader_model(reader: Reader) -> String {
    match reader {
        Reader::Unread => read_model(Rows::Binary, Read::Unread),
        Reader::Definer => read_model(Rows::Binary, Read::Bare),
        Reader::NonDefiner => read_model(Rows::Gaussian, Read::Bare),
    }
}

/// #2751. The refusal's input space is the record kinds the data carries (none; or any
/// non-empty set of discrete-state, time-to-event and count, none of them defining a
/// level) × how the block is read (unread, bare by a level-defining endpoint, bare only by
/// a reader that defines none). Gaussian rows, or a discrete-state record on a reading
/// endpoint, define a level and never reach the refusal. Each cell asserts both sides of
/// the data-vs-expression gate: the population with records blames the expression and
/// names exactly the kinds present, and the same model on the population stripped of its
/// records says the data has no rows and blames nothing else.
///
/// Mutations: gate on discrete-state records alone (the #2623 gate) and every event-only
/// and count-only cell says `NO_ROWS`; drop a kind's clause, or emit it unconditionally,
/// and that kind's present or absent cells fail; swap the non-definer advice and the
/// discrete and non-discrete cells each get the other's sentence.
#[test]
fn record_kind_cells() {
    let sets = [
        (false, true, false),
        (false, false, true),
        (false, true, true),
        (true, false, false),
        (true, true, true),
    ];
    for reader in [Reader::Unread, Reader::Definer, Reader::NonDefiner] {
        let text = reader_model(reader);
        for (discrete, events, counts) in sets {
            let kinds = Kinds {
                discrete,
                events,
                counts,
            };
            let cell = format!("{kinds:?} × {reader:?}");
            let bind = |pop: &mut Population| {
                let mut parsed = parse_full_model(&text).unwrap_or_else(|e| panic!("{cell}: {e}"));
                crate::api::levels::bind_theta_levels_diag(&mut parsed, &text, pop)
            };
            let d = bind(&mut records_population(kinds)).expect_err(&cell);
            assert_eq!(d.code, "E_THETA_LEVEL_BINDING", "{cell}");
            let m = &d.message;
            // Data side: the data has rows, so it is not blamed.
            assert!(!m.contains(NO_ROWS), "{cell}: blames the data: {m}");
            for sentence in [NO_LEVELS, NO_GAUSSIAN] {
                assert!(m.contains(sentence), "{cell}: missing {sentence:?} in {m}");
            }
            for (present, clause) in [
                (discrete, NO_READER),
                (events, EVENT_CLAUSE),
                (counts, COUNT_CLAUSE),
            ] {
                assert_eq!(
                    m.contains(clause),
                    present,
                    "{cell}: {clause:?} present = {present} in {m}"
                );
            }
            // Expression side: exactly one read arm.
            let want: &[&str] = match reader {
                Reader::Unread => &[UNREAD, READ_BARE],
                Reader::Definer => &[BARE_NO_RECORDS],
                Reader::NonDefiner if discrete => &[BARE_NO_DEFINER, BARE_WHY, BARE_ADVICE],
                Reader::NonDefiner => &[BARE_NO_DEFINER, BARE_WHY, BARE_NO_DEFINER_DATA],
            };
            for sentence in [UNREAD, READ_BARE].into_iter().chain(BARE_ANY) {
                assert_eq!(
                    m.contains(sentence),
                    want.contains(&sentence),
                    "{cell}: {sentence:?} in {m}"
                );
            }
            // The other side of the gate, same model: no record of any kind.
            let none = bind(&mut records_population(Kinds {
                discrete: false,
                events: false,
                counts: false,
            }))
            .expect_err(&cell)
            .message;
            assert!(none.contains(NO_ROWS), "{cell}, no rows: {none}");
            for sentence in [
                NO_GAUSSIAN,
                NO_READER,
                EVENT_CLAUSE,
                COUNT_CLAUSE,
                UNREAD,
                READ_BARE,
            ]
            .into_iter()
            .chain(BARE_ANY)
            {
                assert!(
                    !none.contains(sentence),
                    "{cell}, no rows: {sentence:?} in {none}"
                );
            }
        }
    }
}

/// #2751, end to end through the CSV reader `ferx check --data` uses: a Weibull TTE model
/// on TTE-only data, with the block read bare by the hazard or not at all. The reader
/// routes every row to an `Event` record, so the data has observation rows and the refusal
/// names them, never `NO_ROWS`. This is the measured repro of the issue: before the fix
/// both cells said "the data carries no observation rows".
///
/// Mutation: gate on discrete-state records alone and both cells say `NO_ROWS`.
#[test]
fn tte_only_csv_is_not_called_rowless() {
    use std::io::Write;
    let bare =
        "[parameters]\n  theta TVSCALE(20.0, 0.1, 500.0)\n  theta TVSHAPE(1.5, 0.1, 10.0)\n  \
                theta PLACEBO[STUDY](0.0, -5.0, 5.0)\n[covariates]\n  STUDY categorical\n\
                [event_model]\n  cmt    = 2\n  family = weibull\n  \
                scale  = TVSCALE * exp(PLACEBO)\n  shape  = TVSHAPE\n";
    let mut data = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(data, "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY").unwrap();
    for i in 1..=6u32 {
        writeln!(
            data,
            "{i},{},{},0,0,2,0,0,{}",
            5 + i,
            i % 2,
            1 + (i - 1) % 3
        )
        .unwrap();
    }
    data.flush().unwrap();
    for (label, text, unread) in [
        ("hazard bare", bare.to_string(), false),
        ("unread", bare.replace(" * exp(PLACEBO)", ""), true),
    ] {
        let mut model = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
        model.write_all(text.as_bytes()).unwrap();
        model.flush().unwrap();
        let report = crate::api::validate_model_file(
            model.path().to_str().unwrap(),
            Some(data.path().to_str().unwrap()),
        );
        let d = report
            .diagnostics
            .iter()
            .find(|d| d.code == "E_THETA_LEVEL_BINDING")
            .unwrap_or_else(|| panic!("{label}: {:?}", report.diagnostics));
        let m = &d.message;
        assert!(!m.contains(NO_ROWS), "{label}: blames the data: {m}");
        for sentence in [NO_LEVELS, NO_GAUSSIAN, EVENT_CLAUSE] {
            assert!(m.contains(sentence), "{label}: missing {sentence:?} in {m}");
        }
        for sentence in [NO_READER, COUNT_CLAUSE] {
            assert!(!m.contains(sentence), "{label}: {sentence:?} in {m}");
        }
        assert_eq!(m.contains(UNREAD), unread, "{label}: {m}");
        assert_eq!(m.contains(BARE_NO_DEFINER_DATA), !unread, "{label}: {m}");
    }
}
