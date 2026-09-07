use super::*;
use std::collections::BTreeMap;

fn labels(spec: &[&str]) -> Vec<String> {
    spec.iter().map(|s| s.to_string()).collect()
}

#[test]
fn sample_size_parses_both_psn_forms() {
    assert_eq!(SampleSize::parse("32").unwrap(), SampleSize::Total(32));
    let per = SampleSize::parse("1001=>12,1002=>24,1003=>10").unwrap();
    let SampleSize::PerStratum(map) = per else {
        panic!("expected a per-stratum map");
    };
    assert_eq!(map["1001"], 12);
    assert_eq!(map["1002"], 24);
    assert_eq!(map["1003"], 10);
    // Whitespace around the pairs is what a shell-quoted PsN argument looks like
    // after a copy-paste.
    let spaced = SampleSize::parse(" 1001 => 12 , 1002 => 24 ").unwrap();
    let SampleSize::PerStratum(map) = spaced else {
        panic!("expected a per-stratum map");
    };
    assert_eq!(map["1001"], 12);
}

#[test]
fn sample_size_rejects_malformed_specs() {
    assert!(SampleSize::parse("").is_err());
    assert!(SampleSize::parse("abc").is_err());
    // A bare number mixed into a per-stratum list: silently dropping it would
    // change the design without saying so.
    assert!(SampleSize::parse("1001=>12,24").is_err());
    assert!(SampleSize::parse("1001=>x").is_err());
    assert!(SampleSize::parse("1001=>12,1001=>24").is_err());
}

#[test]
fn strata_group_subjects_in_dataset_order() {
    let strata = Strata::from_labels(&labels(&["a", "b", "a", "b", "a"]), "GRP");
    assert_eq!(strata.groups["a"], vec![0, 2, 4]);
    assert_eq!(strata.groups["b"], vec![1, 3]);
    assert_eq!(strata.column.as_deref(), Some("GRP"));
}

#[test]
fn default_allocation_is_each_stratum_at_its_own_size() {
    let strata = Strata::from_labels(&labels(&["a", "b", "a"]), "GRP");
    let alloc = strata.allocation(&SampleSize::Original).unwrap();
    assert_eq!(alloc, vec![("a".to_string(), 2), ("b".to_string(), 1)]);
}

#[test]
fn a_single_sample_size_splits_strata_in_proportion() {
    // 10 rich + 90 sparse, the guide's own example. A request for 50 must keep
    // the 1:9 composition rather than splitting evenly.
    let mut spec = Vec::new();
    spec.extend(std::iter::repeat_n("rich", 10));
    spec.extend(std::iter::repeat_n("sparse", 90));
    let strata = Strata::from_labels(&labels(&spec), "GRP");
    let alloc: BTreeMap<String, usize> = strata
        .allocation(&SampleSize::Total(50))
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(alloc["rich"], 5);
    assert_eq!(alloc["sparse"], 45);
}

#[test]
fn unstratified_total_is_taken_literally() {
    let strata = Strata::unstratified(32);
    let alloc = strata.allocation(&SampleSize::Total(7)).unwrap();
    assert_eq!(alloc, vec![(String::new(), 7)]);
}

#[test]
fn per_stratum_sample_size_requires_stratification_and_every_stratum() {
    let mut map = BTreeMap::new();
    map.insert("a".to_string(), 3);

    // Per-stratum counts with no strata at all.
    let flat = Strata::unstratified(4);
    assert!(flat
        .allocation(&SampleSize::PerStratum(map.clone()))
        .unwrap_err()
        .contains("--stratify-on"));

    // A stratum present in the data but missing from the list: far more likely a
    // typo than an intent to include it whole.
    let strata = Strata::from_labels(&labels(&["a", "b"]), "GRP");
    let err = strata
        .allocation(&SampleSize::PerStratum(map.clone()))
        .unwrap_err();
    assert!(err.contains("omits"), "{err}");

    // A stratum named in the list but absent from the data.
    let mut unknown = map.clone();
    unknown.insert("zz".to_string(), 1);
    unknown.insert("b".to_string(), 1);
    let err = strata
        .allocation(&SampleSize::PerStratum(unknown))
        .unwrap_err();
    assert!(err.contains("zz"), "{err}");
}

#[test]
fn replicate_seeds_are_distinct_and_depend_only_on_seed_and_index() {
    let a: Vec<u64> = (1..=100).map(|i| replicate_seed(7, i)).collect();
    let mut sorted = a.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), a.len(), "replicate seeds collided");
    // Pure function: recomputing gives the same answer, and a different master
    // seed gives a different stream.
    assert_eq!(replicate_seed(7, 42), replicate_seed(7, 42));
    assert_ne!(replicate_seed(7, 42), replicate_seed(8, 42));
}

#[test]
fn draws_are_reproducible_and_order_independent() {
    let strata = Strata::unstratified(20);
    let alloc = strata.allocation(&SampleSize::Original).unwrap();

    // The whole point of per-replicate seeding: replicate 5 is the same draw
    // whether it is computed first, last, or on another thread. PsN's guide
    // documents the opposite as a known wart.
    let forward: Vec<Replicate> = (1..=10).map(|i| draw(&strata, &alloc, 99, i)).collect();
    let backward: Vec<Replicate> = (1..=10)
        .rev()
        .map(|i| draw(&strata, &alloc, 99, i))
        .collect();
    for r in &forward {
        assert!(
            backward.contains(r),
            "replicate {} was not reproduced",
            r.index
        );
    }

    // Distinct replicates are actually distinct draws.
    assert_ne!(forward[0].keys, forward[1].keys);
}

#[test]
fn a_draw_is_sorted_and_the_right_size() {
    let strata = Strata::unstratified(12);
    let alloc = strata.allocation(&SampleSize::Original).unwrap();
    let r = draw(&strata, &alloc, 3, 1);
    assert_eq!(r.keys.len(), 12);
    assert!(
        r.keys.windows(2).all(|w| w[0] <= w[1]),
        "keys must be ascending"
    );
    assert!(r.keys.iter().all(|&k| k < 12));
    // With replacement: over 12 draws from 12 subjects, some subject is drawn
    // twice with overwhelming probability, so the multiset is not a permutation.
    let counts = r.counts(12);
    assert_eq!(counts.iter().sum::<usize>(), 12);
}

#[test]
fn stratified_draws_have_exactly_the_requested_composition() {
    let mut spec = Vec::new();
    spec.extend(std::iter::repeat_n("rich", 10));
    spec.extend(std::iter::repeat_n("sparse", 90));
    let strata = Strata::from_labels(&labels(&spec), "GRP");
    let alloc = strata.allocation(&SampleSize::Original).unwrap();

    for index in 1..=25 {
        let r = draw(&strata, &alloc, 5, index);
        let rich = r.keys.iter().filter(|&&k| k < 10).count();
        let sparse = r.keys.iter().filter(|&&k| k >= 10).count();
        assert_eq!(rich, 10, "replicate {index} lost the rich stratum's size");
        assert_eq!(
            sparse, 90,
            "replicate {index} lost the sparse stratum's size"
        );
    }
}

// ── building the replicate population ───────────────────────────────────────

fn population(ids: &[&str]) -> ferx_core::Population {
    let subjects = ids
        .iter()
        .map(|id| ferx_core::Subject {
            id: (*id).to_string(),
            obs_times: vec![1.0],
            observations: vec![2.0],
            ..Default::default()
        })
        .collect();
    ferx_core::Population {
        subjects,
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec!["ID".to_string(), "DV".to_string()],
        exclusions: None,
        warnings: vec!["a warning from reading the original data".to_string()],
    }
}

#[test]
fn an_identity_replicate_rebuilds_the_original_subjects() {
    // The degenerate oracle: a "resample" that draws every subject exactly once,
    // in order, must be indistinguishable from the original population. This is
    // what pins the reconstruction — if `build_population` dropped a field, the
    // fits below it would silently be fitting something else.
    let original = population(&["A", "B", "C"]);
    let replicate = Replicate {
        index: 1,
        keys: vec![0, 1, 2],
    };
    let rebuilt = build_population(&original, &replicate);

    assert_eq!(rebuilt.subjects.len(), 3);
    for (a, b) in original.subjects.iter().zip(&rebuilt.subjects) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.obs_times, b.obs_times);
        assert_eq!(a.observations, b.observations);
    }
    assert_eq!(rebuilt.covariate_names, original.covariate_names);
    assert_eq!(rebuilt.dv_column, original.dv_column);
    assert_eq!(rebuilt.input_columns, original.input_columns);
    // Read warnings belong to the original read, not to each of 200 replicates.
    assert!(rebuilt.warnings.is_empty());
}

#[test]
fn duplicate_draws_become_independent_subjects() {
    let original = population(&["A", "B"]);
    let replicate = Replicate {
        index: 1,
        keys: vec![0, 0, 0, 1],
    };
    let rebuilt = build_population(&original, &replicate);

    let ids: Vec<&str> = rebuilt.subjects.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec!["A", "A#2", "A#3", "B"]);

    // Distinct IDs are the point: three copies of A must contribute three
    // independent eta draws, not one subject with three times the data.
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), ids.len());

    // The data itself is untouched — only the label changes.
    for s in &rebuilt.subjects[..3] {
        assert_eq!(s.observations, original.subjects[0].observations);
    }
}

#[test]
fn sample_keys_count_every_original_subject() {
    let replicate = Replicate {
        index: 1,
        keys: vec![0, 0, 2],
    };
    assert_eq!(replicate.counts(4), vec![2, 0, 1, 0]);
}
