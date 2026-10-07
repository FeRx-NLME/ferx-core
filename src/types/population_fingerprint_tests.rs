//! T11 (#1685): every field of `Subject`, `DoseEvent` and `ObsRecord` is in the
//! fingerprint, each lands in the digest the refusal text is chosen by, and map
//! order does not enter.

use std::collections::HashMap;

use super::*;

fn cov(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

/// A subject with every field populated, so a perturbation of any one of them has
/// something to change.
fn subject() -> Subject {
    let mut modeled = DoseEvent::modeled(12.0, 50.0, 2, true, 24.0, RateMode::ModeledRate);
    modeled.rate = 0.0;
    Subject {
        id: "7".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0), modeled],
        obs_times: vec![1.0, 2.0],
        obs_raw_times: vec![1.0, 2.0],
        observations: vec![3.5, 2.5],
        obs_cmts: vec![2, 2],
        covariates: cov(&[("WT", 70.0), ("AGE", 40.0)]),
        dose_covariates: vec![cov(&[("WT", 70.0)]), cov(&[("WT", 71.0)])],
        obs_covariates: vec![cov(&[("WT", 70.0)]), cov(&[("WT", 72.0)])],
        pk_only_times: vec![5.0],
        pk_only_covariates: vec![cov(&[("WT", 73.0)])],
        reset_times: vec![30.0],
        reset_covariates: vec![cov(&[("WT", 74.0)])],
        cens: vec![0, 1],
        occasions: vec![1, 2],
        obs_l2: vec![0, 3],
        dose_occasions: vec![1, 2],
        reset_occasions: vec![2],
        fremtype: vec![0, 100],
        obs_records: vec![
            ObsRecord::DiscreteState {
                time: 4.0,
                raw_time: 4.0,
                state: 1,
                cmt: 3,
            },
            ObsRecord::Count {
                time: 6.0,
                raw_time: 6.0,
                count: 2,
                cmt: 4,
            },
            #[cfg(feature = "survival")]
            ObsRecord::Event {
                time: 8.0,
                event_type: EventType::IntervalCensored {
                    left: 7.0,
                    right: 8.0,
                },
                entry_time: 0.5,
                cmt: 5,
            },
        ],
    }
}

fn population(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: vec!["WT".to_string(), "AGE".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Which digest a perturbation must land in.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Row {
    Id,
    Records,
    Doses,
    Covariates,
}

fn row_of(d: &Difference) -> Row {
    match d {
        Difference::SubjectId { .. } => Row::Id,
        Difference::Records { .. } => Row::Records,
        Difference::Doses { .. } => Row::Doses,
        Difference::Covariates { .. } => Row::Covariates,
        other => panic!("not a per-subject difference: {other:?}"),
    }
}

type Perturb = fn(&mut Subject);

/// One row per field. Each mutation that drops a field from the encoding leaves
/// exactly its own row's digest equal, and that row fails.
fn perturbations() -> Vec<(&'static str, Perturb, Row)> {
    #[cfg_attr(not(feature = "survival"), allow(unused_mut))]
    let mut rows: Vec<(&'static str, Perturb, Row)> = vec![
        ("id", |s| s.id = "8".into(), Row::Id),
        ("doses (count)", |s| s.doses.truncate(1), Row::Doses),
        ("obs_times", |s| s.obs_times[1] = 2.5, Row::Records),
        ("obs_raw_times", |s| s.obs_raw_times[1] = 2.5, Row::Records),
        ("observations", |s| s.observations[0] = 3.6, Row::Records),
        ("obs_cmts", |s| s.obs_cmts[0] = 1, Row::Records),
        (
            "covariates",
            |s| {
                s.covariates.insert("WT".into(), 70.5);
            },
            Row::Covariates,
        ),
        (
            "dose_covariates",
            |s| {
                s.dose_covariates[1].insert("WT".into(), 75.0);
            },
            Row::Covariates,
        ),
        (
            "obs_covariates",
            |s| {
                s.obs_covariates[1].insert("WT".into(), 75.0);
            },
            Row::Covariates,
        ),
        ("pk_only_times", |s| s.pk_only_times[0] = 5.5, Row::Records),
        (
            "pk_only_covariates",
            |s| {
                s.pk_only_covariates[0].insert("WT".into(), 75.0);
            },
            Row::Covariates,
        ),
        ("reset_times", |s| s.reset_times[0] = 31.0, Row::Records),
        (
            "reset_covariates",
            |s| {
                s.reset_covariates[0].insert("WT".into(), 75.0);
            },
            Row::Covariates,
        ),
        ("cens", |s| s.cens[1] = -1, Row::Records),
        ("occasions", |s| s.occasions[1] = 3, Row::Records),
        ("obs_l2", |s| s.obs_l2[1] = 4, Row::Records),
        ("dose_occasions", |s| s.dose_occasions[1] = 3, Row::Doses),
        (
            "reset_occasions",
            |s| s.reset_occasions[0] = 3,
            Row::Records,
        ),
        ("fremtype", |s| s.fremtype[1] = 200, Row::Records),
        (
            "obs_records (count)",
            |s| s.obs_records.truncate(1),
            Row::Records,
        ),
        // DoseEvent, one row per field, on the second (modeled) dose.
        ("DoseEvent.time", |s| s.doses[1].time = 13.0, Row::Doses),
        ("DoseEvent.amt", |s| s.doses[1].amt = 51.0, Row::Doses),
        ("DoseEvent.cmt", |s| s.doses[1].cmt = 1, Row::Doses),
        ("DoseEvent.rate", |s| s.doses[1].rate = 1.0, Row::Doses),
        (
            "DoseEvent.duration",
            |s| s.doses[1].duration = 1.0,
            Row::Doses,
        ),
        ("DoseEvent.ss", |s| s.doses[1].ss = false, Row::Doses),
        ("DoseEvent.ii", |s| s.doses[1].ii = 12.0, Row::Doses),
        (
            "DoseEvent.rate_mode",
            |s| s.doses[1].rate_mode = RateMode::ModeledDuration,
            Row::Doses,
        ),
        (
            "DoseEvent.infusion_def",
            |s| s.doses[1].infusion_def = InfusionDef::DurationDefined,
            Row::Doses,
        ),
        // ObsRecord, one row per field of each variant.
        (
            "DiscreteState.time",
            |s| set_discrete(s, |t, _, _, _| *t = 4.5),
            Row::Records,
        ),
        (
            "DiscreteState.raw_time",
            |s| set_discrete(s, |_, r, _, _| *r = 4.5),
            Row::Records,
        ),
        (
            "DiscreteState.state",
            |s| set_discrete(s, |_, _, x, _| *x = 0),
            Row::Records,
        ),
        (
            "DiscreteState.cmt",
            |s| set_discrete(s, |_, _, _, c| *c = 9),
            Row::Records,
        ),
        (
            "Count.time",
            |s| set_count(s, |t, _, _, _| *t = 6.5),
            Row::Records,
        ),
        (
            "Count.raw_time",
            |s| set_count(s, |_, r, _, _| *r = 6.5),
            Row::Records,
        ),
        (
            "Count.count",
            |s| set_count(s, |_, _, x, _| *x = 3),
            Row::Records,
        ),
        (
            "Count.cmt",
            |s| set_count(s, |_, _, _, c| *c = 9),
            Row::Records,
        ),
        (
            "DiscreteState → Count (variant tag)",
            |s| {
                s.obs_records[0] = ObsRecord::Count {
                    time: 4.0,
                    raw_time: 4.0,
                    count: 1,
                    cmt: 3,
                }
            },
            Row::Records,
        ),
    ];
    #[cfg(feature = "survival")]
    rows.extend::<[(&'static str, Perturb, Row); 6]>([
        (
            "Event.time",
            |s| set_event(s, |t, _, _, _| *t = 8.5),
            Row::Records,
        ),
        (
            "Event.event_type (variant)",
            |s| set_event(s, |_, e, _, _| *e = EventType::Exact),
            Row::Records,
        ),
        (
            "Event.event_type (left)",
            |s| {
                set_event(s, |_, e, _, _| {
                    *e = EventType::IntervalCensored {
                        left: 6.0,
                        right: 8.0,
                    }
                })
            },
            Row::Records,
        ),
        (
            "Event.event_type (right)",
            |s| {
                set_event(s, |_, e, _, _| {
                    *e = EventType::IntervalCensored {
                        left: 7.0,
                        right: 9.0,
                    }
                })
            },
            Row::Records,
        ),
        (
            "Event.entry_time",
            |s| set_event(s, |_, _, en, _| *en = 0.25),
            Row::Records,
        ),
        (
            "Event.cmt",
            |s| set_event(s, |_, _, _, c| *c = 9),
            Row::Records,
        ),
    ]);
    rows
}

fn set_discrete(s: &mut Subject, f: impl FnOnce(&mut f64, &mut f64, &mut usize, &mut usize)) {
    for o in &mut s.obs_records {
        if let ObsRecord::DiscreteState {
            time,
            raw_time,
            state,
            cmt,
        } = o
        {
            return f(time, raw_time, state, cmt);
        }
    }
    unreachable!("the fixture has a DiscreteState record");
}

fn set_count(s: &mut Subject, f: impl FnOnce(&mut f64, &mut f64, &mut u32, &mut usize)) {
    for o in &mut s.obs_records {
        if let ObsRecord::Count {
            time,
            raw_time,
            count,
            cmt,
        } = o
        {
            return f(time, raw_time, count, cmt);
        }
    }
    unreachable!("the fixture has a Count record");
}

#[cfg(feature = "survival")]
fn set_event(s: &mut Subject, f: impl FnOnce(&mut f64, &mut EventType, &mut f64, &mut usize)) {
    for o in &mut s.obs_records {
        if let ObsRecord::Event {
            time,
            event_type,
            entry_time,
            cmt,
        } = o
        {
            return f(time, event_type, entry_time, cmt);
        }
    }
    unreachable!("the fixture has an Event record");
}

/// T11. Each field, perturbed on its own, changes the fingerprint and is classified
/// into the row its refusal text is chosen by. The second subject is left alone,
/// so the difference must also name the right one.
///
/// Mutations — drop any field from the encoding: that field's row stays equal and
/// the assertion naming it fails; hash a dose field into the records digest: the
/// classification fails.
#[test]
fn every_field_is_hashed_into_its_own_digest() {
    let mut other = subject();
    other.id = "9".to_string();
    let base = population(vec![subject(), other.clone()]);
    let fp = PopulationFingerprint::of(&base);
    assert_eq!(fp.first_difference(&base), None, "a population is its own");
    let rows = perturbations();
    // 20 `Subject` fields + 9 `DoseEvent` + the `ObsRecord` fields (+ `Event`'s).
    assert_eq!(rows.len(), if cfg!(feature = "survival") { 44 } else { 38 });
    for (field, perturb, want) in rows {
        let mut s = subject();
        perturb(&mut s);
        let changed = population(vec![s, other.clone()]);
        let got = fp
            .first_difference(&changed)
            .unwrap_or_else(|| panic!("{field}: not in the fingerprint"));
        assert_eq!(row_of(&got), want, "{field}: {got:?}");
        if want != Row::Id {
            let named = match &got {
                Difference::Records { id, .. }
                | Difference::Doses { id, .. }
                | Difference::Covariates { id } => id.as_str(),
                _ => unreachable!(),
            };
            assert_eq!(named, "7", "{field}: names the subject that differs");
        }
    }
}

/// The counts each per-subject row reports, and the population-level rows.
#[test]
fn differences_report_their_counts() {
    let base = population(vec![subject()]);
    let fp = PopulationFingerprint::of(&base);

    let mut s = subject();
    s.doses
        .push(DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0));
    assert_eq!(
        fp.first_difference(&population(vec![s])),
        Some(Difference::Doses {
            id: "7".into(),
            population: 3,
            fit: 2
        })
    );

    let mut s = subject();
    s.observations.push(1.0);
    s.obs_times.push(3.0);
    // 2 Gaussian observations plus the non-Gaussian records.
    let n = 2 + subject().obs_records.len();
    assert_eq!(
        fp.first_difference(&population(vec![s])),
        Some(Difference::Records {
            id: "7".into(),
            population: n + 1,
            fit: n
        })
    );

    assert_eq!(
        fp.first_difference(&population(vec![subject(), subject()])),
        Some(Difference::SubjectCount {
            population: 2,
            fit: 1
        })
    );

    let mut renamed = population(vec![subject()]);
    renamed.covariate_names = vec!["WT".into(), "__level_SHIFT".into()];
    assert_eq!(
        fp.first_difference(&renamed),
        Some(Difference::CovariateNames {
            missing: vec!["AGE".into()],
            extra: vec!["__level_SHIFT".into()],
        })
    );

    // Records come before doses, doses before covariates: a subject that differs in
    // all three hears about its records.
    let mut s = subject();
    s.observations[0] = 9.0;
    s.doses[0].amt = 1.0;
    s.covariates.insert("WT".into(), 1.0);
    assert!(matches!(
        fp.first_difference(&population(vec![s])),
        Some(Difference::Records { .. })
    ));
    let mut s = subject();
    s.doses[0].amt = 1.0;
    s.covariates.insert("WT".into(), 1.0);
    assert!(matches!(
        fp.first_difference(&population(vec![s])),
        Some(Difference::Doses { .. })
    ));
}

/// Neither `HashMap` insertion order nor capacity enters the digest, nor the
/// order of `covariate_names`; the fields deliberately left out do not either.
#[test]
fn the_fingerprint_ignores_map_order_and_read_metadata() {
    let base = population(vec![subject()]);
    let fp = PopulationFingerprint::of(&base);

    let mut s = subject();
    let mut reordered: HashMap<String, f64> = HashMap::with_capacity(64);
    reordered.insert("AGE".into(), 40.0);
    reordered.insert("WT".into(), 70.0);
    s.covariates = reordered;
    let mut p = population(vec![s]);
    p.covariate_names.reverse();
    p.dv_column = "CONC".into();
    p.input_columns = vec!["ID".into()];
    p.warnings = vec!["a reader warning".into()];
    assert_eq!(fp.first_difference(&p), None);
    assert_eq!(PopulationFingerprint::of(&p), fp);
}

/// T12's unit half: a fingerprint from another scheme is not current; this one is.
#[test]
fn only_the_current_scheme_is_comparable() {
    let fp = PopulationFingerprint::of(&population(vec![subject()]));
    assert!(fp.is_current());
    assert!(!fp.clone().with_scheme(SCHEME + 1).is_current());
}

/// The fingerprint survives JSON (how `.fitrx` and the R fit object carry it).
#[test]
fn the_fingerprint_round_trips_through_json() {
    let fp = PopulationFingerprint::of(&population(vec![subject()]));
    let json = serde_json::to_string(&fp).unwrap();
    let back: PopulationFingerprint = serde_json::from_str(&json).unwrap();
    assert_eq!(back, fp);
}
