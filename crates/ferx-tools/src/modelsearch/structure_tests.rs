//! Tier-1 tests for the structural coordinates, Pharmpy's move rules and
//! the structure → `pk` line derivation (#1181).

use std::collections::HashMap;

use ferx_core::{Population, Subject};

use super::*;
use crate::search::mfl::Mfl;

fn template(line: &str) -> PkTemplate {
    PkTemplate::parse_line(line)
        .expect("a pk line")
        .expect("parses")
}

fn fo(peripherals: u32) -> Structure {
    Structure {
        absorption: Absorption::Fo,
        peripherals,
        transits: None,
        lagtime: false,
    }
}

fn keys(mfl: &str) -> Vec<FeatureKey> {
    space_features(&Mfl::parse(mfl).unwrap()).unwrap()
}

// ── coordinates ─────────────────────────────────────────────────────────────

#[test]
fn structure_is_read_off_the_pk_line() {
    let cases = [
        (
            "pk one_cpt_iv(cl=CL, v=V)",
            Absorption::Inst,
            0,
            None,
            false,
        ),
        (
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)",
            Absorption::Fo,
            0,
            None,
            true,
        ),
        (
            "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, alag=TLAG)",
            Absorption::Fo,
            1,
            None,
            true,
        ),
        (
            "pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)",
            Absorption::Inst,
            2,
            None,
            false,
        ),
        (
            "pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT)",
            Absorption::Fo,
            1,
            Some(TransitCount::N),
            false,
        ),
        (
            "pk one_compartment_oral(cl=CL, v=V, ka=KA)",
            Absorption::Fo,
            0,
            None,
            false,
        ),
    ];
    for (line, absorption, peripherals, transits, lagtime) in cases {
        let s = Structure::from_template(&template(line)).unwrap_or_else(|e| panic!("{line}: {e}"));
        assert_eq!(
            s,
            Structure {
                absorption,
                peripherals,
                transits,
                lagtime
            },
            "{line}"
        );
    }
    let err = Structure::from_template(&template("pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)"))
        .unwrap_err();
    assert!(err.contains("inverse-Gaussian"), "{err}");
    let err = Structure::from_template(&template("pk mystery(cl=CL)")).unwrap_err();
    assert!(err.contains("not a `<n>_cpt_<route>` template"), "{err}");
}

#[test]
fn every_buildable_structure_names_the_parser_template_and_roles() {
    // The role lists here are `PkModel::required_pk_params` (types.rs)
    // plus `lagtime`; a drift from the parser's table shows up in the
    // end-to-end test as a candidate that fails to compile, but this pins
    // the names so the failure is local.
    let t = fo(0).template().unwrap();
    assert_eq!(t.name, "one_cpt_oral");
    assert_eq!(t.roles, vec!["cl", "v", "ka"]);
    let t = fo(2).template().unwrap();
    assert_eq!(t.name, "three_cpt_oral");
    assert_eq!(t.roles, vec!["cl", "v1", "q2", "v2", "q3", "v3", "ka"]);
    let t = Structure {
        absorption: Absorption::Inst,
        peripherals: 1,
        transits: None,
        lagtime: false,
    }
    .template()
    .unwrap();
    assert_eq!(t.name, "two_cpt_iv");
    assert_eq!(t.roles, vec!["cl", "v1", "q", "v2"]);
    let t = Structure {
        transits: Some(TransitCount::Count(3)),
        ..fo(1)
    }
    .template()
    .unwrap();
    assert_eq!(t.name, "two_cpt_transit");
    assert_eq!(t.roles, vec!["cl", "v1", "q", "v2", "n", "mtt"]);
    let t = Structure {
        lagtime: true,
        ..fo(0)
    }
    .template()
    .unwrap();
    assert_eq!(t.name, "one_cpt_oral");
    assert_eq!(t.roles, vec!["cl", "v", "ka", "lagtime"]);
}

#[test]
fn unbuildable_structures_name_the_reason() {
    let iv = Structure {
        absorption: Absorption::Inst,
        ..fo(0)
    };
    assert!(iv.unbuildable().is_none());
    let why = Structure {
        transits: Some(TransitCount::N),
        ..iv
    }
    .unbuildable()
    .unwrap();
    assert!(why.contains("first-order absorption"), "{why}");
    let why = Structure {
        lagtime: true,
        ..iv
    }
    .unbuildable()
    .unwrap();
    assert!(why.contains("bolus"), "{why}");
    let why = Structure {
        lagtime: true,
        transits: Some(TransitCount::Count(2)),
        ..fo(0)
    }
    .unbuildable()
    .unwrap();
    assert!(why.contains("lag time and a transit chain"), "{why}");
    let why = Structure {
        transits: Some(TransitCount::N),
        ..fo(2)
    }
    .unbuildable()
    .unwrap();
    assert!(why.contains("three_cpt_transit"), "{why}");
    assert!(fo(3).unbuildable().unwrap().contains("three_cpt_*"));
    assert!(fo(3).template().is_err());
}

#[test]
fn apply_moves_one_coordinate_and_features_lists_all_four() {
    let s = fo(0);
    assert_eq!(s.apply(&FeatureKey::Peripherals(2)).peripherals, 2);
    assert!(s.apply(&FeatureKey::Lagtime(true)).lagtime);
    assert_eq!(
        s.apply(&FeatureKey::Transits(TransitCount::N)).transits,
        Some(TransitCount::N)
    );
    // `TRANSITS(0)` is "no transits", the same point as the base.
    assert_eq!(s.apply(&FeatureKey::Transits(TransitCount::Count(0))), s);
    assert_eq!(
        s.apply(&FeatureKey::Absorption(Absorption::Inst))
            .absorption,
        Absorption::Inst
    );
    assert_eq!(
        s.features(),
        vec![
            FeatureKey::Absorption(Absorption::Fo),
            FeatureKey::Peripherals(0),
            FeatureKey::Transits(TransitCount::Count(0)),
            FeatureKey::Lagtime(false),
        ]
    );
    assert_eq!(
        s.feature_vector().render(),
        "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=0;TRANSITS=0"
    );
}

#[test]
fn feature_keys_print_and_sort_as_pharmpy_does() {
    assert_eq!(FeatureKey::Peripherals(1).to_string(), "PERIPHERALS(1)");
    assert_eq!(
        FeatureKey::Transits(TransitCount::Count(3)).to_string(),
        "TRANSITS(3, NODEPOT)"
    );
    assert_eq!(
        FeatureKey::Transits(TransitCount::N).to_string(),
        "TRANSITS(N, NODEPOT)"
    );
    assert_eq!(FeatureKey::Lagtime(true).to_string(), "LAGTIME(ON)");
    assert_eq!(
        FeatureKey::Absorption(Absorption::Inst).to_string(),
        "ABSORPTION(INST)"
    );
    // Pharmpy sorts its feature dictionary by `(category, str(arg))`, so
    // LAGTIME precedes PERIPHERALS and `N` follows the counts.
    let k = keys(
        "TRANSITS(N); TRANSITS(1..2, NODEPOT); PERIPHERALS(0..1); LAGTIME([OFF,ON]); \
         ABSORPTION([INST,FO])",
    );
    let printed: Vec<String> = k.iter().map(|k| k.to_string()).collect();
    assert_eq!(
        printed,
        vec![
            "ABSORPTION(FO)",
            "ABSORPTION(INST)",
            "LAGTIME(OFF)",
            "LAGTIME(ON)",
            "PERIPHERALS(0)",
            "PERIPHERALS(1)",
            "TRANSITS(1, NODEPOT)",
            "TRANSITS(2, NODEPOT)",
            "TRANSITS(N, NODEPOT)",
        ]
    );
}

#[test]
fn space_features_accepts_elimination_fo_and_refuses_non_structural_statements() {
    assert_eq!(
        keys("ELIMINATION(FO); PERIPHERALS(1)"),
        vec![FeatureKey::Peripherals(1)]
    );
    let err = space_features(&Mfl::parse("PERIPHERALS(1); COVARIATE?(CL, WT, pow)").unwrap())
        .unwrap_err();
    assert!(err.contains("COVARIATE?"), "{err}");
    assert!(err.contains("not a structural statement"), "{err}");
    let err = space_features(&Mfl::parse("ALLOMETRY(WT, 70)").unwrap()).unwrap_err();
    assert!(err.contains("ALLOMETRY"), "{err}");
    // A depot chain reaching here means the coverage check was bypassed.
    let err = space_features(&Mfl::parse("TRANSITS(2, DEPOT)").unwrap()).unwrap_err();
    assert!(err.contains("DEPOT"), "{err}");
    let err = space_features(&Mfl::parse("ABSORPTION(ZO)").unwrap()).unwrap_err();
    assert!(err.contains("ABSORPTION(ZO)"), "{err}");
}

// ── onto the space ──────────────────────────────────────────────────────────

#[test]
fn onto_space_is_the_least_number_of_transformations() {
    // Already on the space: nothing to do.
    assert!(onto_space(&fo(0), &keys("PERIPHERALS(0..1); LAGTIME([OFF,ON])")).is_empty());
    // A space that lists only `ON` moves the base onto it — Pharmpy's
    // reading; a category the space does not name is left alone.
    assert_eq!(
        onto_space(&fo(0), &keys("LAGTIME(ON)")),
        vec![FeatureKey::Lagtime(true)]
    );
    // Peripherals: the smallest count in the space.
    assert_eq!(
        onto_space(&fo(0), &keys("PERIPHERALS(1..2)")),
        vec![FeatureKey::Peripherals(1)]
    );
    // Absorption: the first listed mode.
    let iv = Structure {
        absorption: Absorption::Inst,
        ..fo(0)
    };
    assert_eq!(
        onto_space(&iv, &keys("ABSORPTION([FO]); PERIPHERALS(0..1)")),
        vec![FeatureKey::Absorption(Absorption::Fo)]
    );
    // Two categories off at once: one move each.
    assert_eq!(
        onto_space(&iv, &keys("ABSORPTION(FO); TRANSITS([1,3], NODEPOT)")),
        vec![
            FeatureKey::Absorption(Absorption::Fo),
            FeatureKey::Transits(TransitCount::Count(1)),
        ]
    );
}

// ── Pharmpy's `_is_allowed` ─────────────────────────────────────────────────

#[test]
fn a_category_is_moved_once_per_path_and_repeats_are_refused() {
    let funcs = keys("ABSORPTION([FO,INST]); LAGTIME(ON); PERIPHERALS(1..2)");
    let inst = FeatureKey::Absorption(Absorption::Inst);
    let fo_ = FeatureKey::Absorption(Absorption::Fo);
    let lag = FeatureKey::Lagtime(true);
    assert!(allowed(&lag, &[], &funcs, &fo(0)));
    assert!(!allowed(&lag, &[lag], &funcs, &fo(0)));
    assert!(!allowed(&fo_, &[inst], &funcs, &fo(0)));
    assert!(!allowed(&inst, &[fo_], &funcs, &fo(0)));
    assert!(allowed(&lag, &[fo_], &funcs, &fo(0)));
}

#[test]
fn peripherals_start_at_the_smallest_count_then_any_other() {
    let funcs = keys("PERIPHERALS(1..2); LAGTIME(ON)");
    let p1 = FeatureKey::Peripherals(1);
    let p2 = FeatureKey::Peripherals(2);
    assert!(allowed(&p1, &[], &funcs, &fo(0)));
    assert!(
        !allowed(&p2, &[], &funcs, &fo(0)),
        "the first peripheral move must be the smallest"
    );
    assert!(allowed(&p2, &[p1], &funcs, &fo(0)));
    assert!(!allowed(&p1, &[p1], &funcs, &fo(0)));
    // Pharmpy's rule verbatim: after a non-minimal count, any count other
    // than the smallest is still allowed — `0 → 2 → 1` is a path.
    let funcs = keys("PERIPHERALS(0..2)");
    let p0 = FeatureKey::Peripherals(0);
    assert!(allowed(&p1, &[p0, p2], &funcs, &fo(0)));
    assert!(!allowed(&p0, &[p2], &funcs, &fo(0)));
    // A peripheral move is exempt from the same-category rule.
    assert!(allowed(&p2, &[p0], &funcs, &fo(0)));
}

#[test]
fn transits_zero_is_a_move_only_off_a_chain() {
    let funcs = keys("TRANSITS(0..2, NODEPOT)");
    let t0 = FeatureKey::Transits(TransitCount::Count(0));
    let t2 = FeatureKey::Transits(TransitCount::Count(2));
    // On a first-order model TRANSITS(0) is the model itself (Pharmpy's rule).
    assert!(!allowed(&t0, &[], &funcs, &fo(0)));
    assert!(allowed(&t2, &[], &funcs, &fo(0)));
    // From a chain, dropping it is a real move — one Pharmpy cannot make,
    // since its base never carries a chain into the space.
    let chain = Structure {
        transits: Some(TransitCount::Count(3)),
        ..fo(0)
    };
    assert!(allowed(&t0, &[], &funcs, &chain));
    // …and still one move per category: not after another transit step.
    assert!(!allowed(&t0, &[t2], &funcs, &chain));
    assert!(!combination_allowed(&[t0], &fo(0)));
    assert!(combination_allowed(&[t0], &chain));
}

#[test]
fn pharmpys_unsupported_pairs_are_refused_in_either_order() {
    let funcs = keys("ABSORPTION([FO,INST]); LAGTIME(ON); TRANSITS([1,3], NODEPOT); TRANSITS(N)");
    let inst = FeatureKey::Absorption(Absorption::Inst);
    let fo_ = FeatureKey::Absorption(Absorption::Fo);
    let lag = FeatureKey::Lagtime(true);
    let t1 = FeatureKey::Transits(TransitCount::Count(1));
    let t3 = FeatureKey::Transits(TransitCount::Count(3));
    let tn = FeatureKey::Transits(TransitCount::N);
    for (a, b) in [(inst, lag), (inst, t3), (lag, t3), (lag, tn), (fo_, t1)] {
        assert!(!allowed(&a, &[b], &funcs, &fo(0)), "{a} after {b}");
        assert!(!allowed(&b, &[a], &funcs, &fo(0)), "{b} after {a}");
    }
    // …and the pairs Pharmpy allows: FO with three transits, in both orders.
    assert!(allowed(&t3, &[fo_], &funcs, &fo(0)));
    assert!(allowed(&fo_, &[t3], &funcs, &fo(0)));
    // The exhaustive filter applies the same table to a combination.
    assert!(!combination_allowed(&[inst, lag], &fo(0)));
    assert!(!combination_allowed(&[t3, lag], &fo(0)));
    assert!(!combination_allowed(
        &[FeatureKey::Transits(TransitCount::Count(0))],
        &fo(0)
    ));
    assert!(combination_allowed(
        &[fo_, t3, FeatureKey::Peripherals(1)],
        &fo(0)
    ));
}

// ── from a structure to an edit ─────────────────────────────────────────────

fn population(times: &[f64]) -> Population {
    Population {
        subjects: vec![Subject {
            id: "1".into(),
            obs_times: times.to_vec(),
            ..Default::default()
        }],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn defaults() -> Defaults {
    Defaults::new(
        vec!["CL".into(), "V".into(), "KA".into()],
        vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
        vec![0.2, 10.0, 1.5],
        vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
        &population(&[0.0, 0.5, 1.0, 2.0]),
    )
}

const LINES: &[&str] = &[
    "CL = TVCL * exp(ETA_CL)",
    "V = TVV * exp(ETA_V)",
    "KA = TVKA * exp(ETA_KA)",
];

fn lines() -> Vec<String> {
    LINES.iter().map(|s| s.to_string()).collect()
}

#[test]
fn defaults_read_the_first_positive_observation_time() {
    assert_eq!(defaults().t_first, 0.5);
    let d = Defaults::new(vec![], vec![], vec![], vec![], &population(&[0.0]));
    assert_eq!(d.t_first, 1.0, "no positive time: Pharmpy's fallback of 1");
    let d = Defaults::new(
        vec![],
        vec![],
        vec![],
        vec![],
        &population(&[f64::NAN, 3.0]),
    );
    assert_eq!(d.t_first, 3.0);
}

#[test]
fn widening_keeps_bound_variables_by_slot_and_declares_the_rest_from_pharmpy_rules() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let spec = structural_spec(
        &fo(1),
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::AbsorptionDelay,
    )
    .unwrap();
    assert_eq!(spec.template, "two_cpt_oral");
    let bindings: Vec<(&str, &str)> = spec
        .bindings
        .iter()
        .map(|(r, v)| (r.as_str(), v.as_str()))
        .collect();
    assert_eq!(
        bindings,
        vec![
            ("cl", "CL"),
            ("v1", "V"),
            ("q", "Q"),
            ("v2", "V2"),
            ("ka", "KA")
        ]
    );
    let by_name: HashMap<&str, &NewParameter> = spec
        .new_parameters
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect();
    // Q = CL, V2 = 0.05·Vc; no η on a peripheral under absorption_delay.
    let q = by_name["Q"];
    assert_eq!(
        (q.theta.as_str(), q.init, q.lower, q.upper),
        ("TVQ", 0.2, 0.0, 1e6)
    );
    assert!(q.iiv.is_none() && !q.fixed);
    let v2 = by_name["V2"];
    assert_eq!((v2.theta.as_str(), v2.init), ("TVV2", 0.5));
    assert_eq!(spec.new_parameters.len(), 2);

    // Two peripherals at once: the first Q is a tenth of CL, the second
    // nine tenths, both volumes 0.05·Vc.
    let spec = structural_spec(
        &fo(2),
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::NoAdd,
    )
    .unwrap();
    assert_eq!(spec.template, "three_cpt_oral");
    let by_name: HashMap<&str, &NewParameter> = spec
        .new_parameters
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect();
    assert!((by_name["Q"].init - 0.02).abs() < 1e-12);
    assert!((by_name["Q3"].init - 0.18).abs() < 1e-12);
    assert_eq!(by_name["V3"].init, 0.5);
    assert_eq!(
        spec.bindings
            .iter()
            .map(|(r, _)| r.as_str())
            .collect::<Vec<_>>(),
        vec!["cl", "v1", "q2", "v2", "q3", "v3", "ka"]
    );
}

#[test]
fn a_second_peripheral_on_a_two_compartment_parent_reuses_q_and_v2_by_slot() {
    let parent = template("pk two_cpt_oral(cl=CL, v1=VC, q=QP, v2=VP, ka=KA)");
    let spec = structural_spec(
        &fo(2),
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::NoAdd,
    )
    .unwrap();
    let bindings: Vec<(&str, &str)> = spec
        .bindings
        .iter()
        .map(|(r, v)| (r.as_str(), v.as_str()))
        .collect();
    assert_eq!(
        bindings,
        vec![
            ("cl", "CL"),
            ("v1", "VC"),
            ("q2", "QP"),
            ("v2", "VP"),
            ("q3", "Q3"),
            ("v3", "V3"),
            ("ka", "KA")
        ]
    );
    let names: Vec<&str> = spec
        .new_parameters
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(names, vec!["Q3", "V3"]);
}

#[test]
fn narrowing_binds_only_what_the_smaller_template_reads() {
    let parent = template("pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)");
    let spec = structural_spec(
        &fo(0),
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::NoAdd,
    )
    .unwrap();
    assert_eq!(spec.template, "one_cpt_oral");
    assert_eq!(
        spec.bindings,
        vec![
            ("cl".to_string(), "CL".to_string()),
            ("v".to_string(), "V".to_string()),
            ("ka".to_string(), "KA".to_string())
        ]
    );
    assert!(spec.new_parameters.is_empty());
}

#[test]
fn a_lag_time_and_a_transit_chain_take_the_absorption_delay_init_and_eta() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let lag = Structure {
        lagtime: true,
        ..fo(0)
    };
    let spec = structural_spec(
        &lag,
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::AbsorptionDelay,
    )
    .unwrap();
    assert_eq!(
        spec.bindings.last().unwrap(),
        &("lagtime".to_string(), "ALAG".to_string())
    );
    let alag = &spec.new_parameters[0];
    assert_eq!(alag.name, "ALAG");
    assert_eq!(alag.init, 0.25, "half the first observation time");
    assert_eq!(alag.iiv, Some(("ETA_ALAG".to_string(), NEW_IIV_VARIANCE)));

    let transit = Structure {
        transits: Some(TransitCount::Count(3)),
        ..fo(0)
    };
    let spec = structural_spec(
        &transit,
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::AbsorptionDelay,
    )
    .unwrap();
    assert_eq!(spec.template, "one_cpt_transit");
    let by_name: HashMap<&str, &NewParameter> = spec
        .new_parameters
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect();
    let n = by_name["NTR"];
    assert!(n.fixed, "a counted chain is a FIXed θ");
    assert_eq!((n.init, n.lower, n.upper), (3.0, 0.0, 64.0));
    assert!(n.iiv.is_none());
    let mtt = by_name["MTT"];
    assert_eq!(mtt.init, 0.25);
    assert_eq!(mtt.iiv.as_ref().map(|(e, _)| e.as_str()), Some("ETA_MTT"));
    // KA is not bound by a transit template: the edit layer prunes it.
    assert!(!spec.bindings.iter().any(|(r, _)| r == "ka"));

    let estimated = Structure {
        transits: Some(TransitCount::N),
        ..fo(0)
    };
    let spec = structural_spec(
        &estimated,
        &fo(0),
        &parent,
        &lines(),
        &defaults(),
        IivStrategy::NoAdd,
    )
    .unwrap();
    let n = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "NTR")
        .unwrap();
    assert!(!n.fixed);
    assert_eq!(n.init, 2.0, "Pharmpy's N init");
}

#[test]
fn iiv_strategies_decide_which_new_parameters_get_an_eta() {
    let parent = template("pk one_cpt_iv(cl=CL, v=V)");
    let d = Defaults::new(
        vec!["CL".into(), "V".into()],
        vec!["TVCL".into(), "TVV".into()],
        vec![0.2, 10.0],
        vec!["ETA_CL".into(), "ETA_V".into()],
        &population(&[0.0, 0.5]),
    );
    let target = Structure {
        lagtime: true,
        ..fo(1)
    };
    let lines = vec![
        "CL = TVCL * exp(ETA_CL)".to_string(),
        "V = TVV * exp(ETA_V)".to_string(),
    ];
    let etas = |iiv: IivStrategy| -> Vec<String> {
        structural_spec(&target, &fo(0), &parent, &lines, &d, iiv)
            .unwrap()
            .new_parameters
            .iter()
            .filter_map(|p| p.iiv.as_ref().map(|(e, _)| e.clone()))
            .collect()
    };
    assert!(etas(IivStrategy::NoAdd).is_empty());
    assert_eq!(etas(IivStrategy::AbsorptionDelay), vec!["ETA_ALAG"]);
    assert_eq!(
        etas(IivStrategy::AddDiagonal),
        vec!["ETA_Q", "ETA_V2", "ETA_KA", "ETA_ALAG"]
    );
    let err =
        structural_spec(&target, &fo(0), &parent, &lines, &d, IivStrategy::Fullblock).unwrap_err();
    assert!(err.contains("fullblock"), "{err}");
    // KA on an IV parent: 1 / (2 · t_first).
    let spec = structural_spec(&fo(0), &fo(0), &parent, &lines, &d, IivStrategy::NoAdd).unwrap();
    assert_eq!(spec.new_parameters[0].name, "KA");
    assert_eq!(spec.new_parameters[0].init, 1.0);
}

#[test]
fn generated_names_avoid_the_base_models_own() {
    // The base already has `TVQ` and `ETA_Q` under other uses, so the new
    // peripheral's θ and η take a suffix rather than a duplicate declaration.
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let d = Defaults::new(
        vec!["CL".into(), "V".into(), "KA".into()],
        vec!["TVCL".into(), "TVV".into(), "TVKA".into(), "TVQ".into()],
        vec![0.2, 10.0, 1.5, 3.0],
        vec!["ETA_CL".into(), "ETA_Q".into()],
        &population(&[0.0, 0.5]),
    );
    let spec = structural_spec(
        &fo(1),
        &fo(0),
        &parent,
        &lines(),
        &d,
        IivStrategy::AddDiagonal,
    )
    .unwrap();
    let q = spec.new_parameters.iter().find(|p| p.name == "Q").unwrap();
    assert_eq!(q.theta, "TVQ_2");
    assert_eq!(q.iiv.as_ref().unwrap().0, "ETA_Q_2");
    // A parameter the base declares but does not bind is reused as is.
    let d = Defaults {
        parameters: vec!["CL".into(), "V".into(), "KA".into(), "Q".into()],
        ..d
    };
    let spec = structural_spec(&fo(1), &fo(0), &parent, &lines(), &d, IivStrategy::NoAdd).unwrap();
    assert!(spec.new_parameters.iter().all(|p| p.name != "Q"));
    assert!(spec.bindings.contains(&("q".to_string(), "Q".to_string())));
}

#[test]
fn inits_scale_from_the_parents_own_clearance_and_volume_lines() {
    // The parent binds `cl=CLR` whose line reads `TVCLR`; the init behind it
    // is what Q scales from — not a name guessed from the role.
    let parent = template("pk one_cpt_oral(cl=CLR, v=VC, ka=KA)");
    let d = Defaults::new(
        vec!["CLR".into(), "VC".into(), "KA".into()],
        vec!["TVCLR".into(), "TVVC".into(), "TVKA".into()],
        vec![4.0, 40.0, 1.0],
        vec![],
        &population(&[0.0, 0.5]),
    );
    let lines = vec![
        "CLR = TVCLR * (WT / 70)^0.75".to_string(),
        "VC = TVVC * exp(ETA_V)".to_string(),
        "KA = TVKA".to_string(),
    ];
    let spec = structural_spec(&fo(1), &fo(0), &parent, &lines, &d, IivStrategy::NoAdd).unwrap();
    let q = spec.new_parameters.iter().find(|p| p.name == "Q").unwrap();
    assert_eq!(q.init, 4.0);
    let v2 = spec.new_parameters.iter().find(|p| p.name == "V2").unwrap();
    assert_eq!(v2.init, 2.0);
    // No θ behind the line: Pharmpy's fallbacks, Q = 0.1 and V = 0.1.
    let lines = vec!["CLR = 4.0".to_string(), "VC = 40.0".to_string()];
    let spec = structural_spec(&fo(1), &fo(0), &parent, &lines, &d, IivStrategy::NoAdd).unwrap();
    let q = spec.new_parameters.iter().find(|p| p.name == "Q").unwrap();
    assert_eq!(q.init, 0.1);
    let v2 = spec.new_parameters.iter().find(|p| p.name == "V2").unwrap();
    assert!((v2.init - 0.1).abs() < 1e-12);
}

#[test]
fn theta_inits_are_read_off_the_text_so_a_seeded_parent_scales_its_children() {
    let text = ferx_core::edit::ModelText::parse(
        "[parameters]\n  theta TVCL(0.132695, 0.001, 10.0)\n  theta TVKA(1.5, FIX)\n  \
         theta PL[3](0.0, -1.0, 1.0)\n  omega ETA_CL ~ 0.09\n",
    )
    .unwrap();
    let inits = theta_inits_of(&text);
    assert_eq!(inits.get("TVCL"), Some(&0.132695));
    assert_eq!(inits.get("TVKA"), Some(&1.5));
    assert!(!inits.contains_key("PL"), "a vector θ has no single init");
    assert_eq!(inits.len(), 2);
}

// ── the review on #1256 ─────────────────────────────────────────────────────

fn text(src: &str) -> ferx_core::edit::ModelText {
    ferx_core::edit::ModelText::parse(src).unwrap()
}

#[test]
fn a_transit_bases_count_is_read_off_the_declaration_its_n_binding_points_at() {
    // A literal.
    let t = template("pk one_cpt_transit(cl=CL, v=V, n=3, mtt=MTT)");
    let m = text(
        "[parameters]\n  theta TVMTT(1.0, 0.0, 10.0)\n[individual_parameters]\n  MTT = TVMTT\n",
    );
    assert_eq!(
        Structure::from_model(&t, Some(&m)).unwrap().transits,
        Some(TransitCount::Count(3))
    );
    // A parameter behind a FIXed θ — what the search itself writes.
    let t = template("pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)");
    let m = text(
        "[parameters]\n  theta TVNTR(3.0, 0.0, 64.0) FIX\n  theta TVMTT(1.0, 0.0, 10.0)\n\
         [individual_parameters]\n  NTR = TVNTR\n  MTT = TVMTT\n",
    );
    assert_eq!(
        Structure::from_model(&t, Some(&m)).unwrap().transits,
        Some(TransitCount::Count(3))
    );
    // The same parameter behind a free θ.
    let m = text(
        "[parameters]\n  theta TVNTR(3.0, 0.0, 64.0)\n  theta TVMTT(1.0, 0.0, 10.0)\n\
         [individual_parameters]\n  NTR = TVNTR\n  MTT = TVMTT\n",
    );
    assert_eq!(
        Structure::from_model(&t, Some(&m)).unwrap().transits,
        Some(TransitCount::N)
    );
    // A parameter set from a number.
    let m = text("[parameters]\n  theta TVMTT(1.0, 0.0, 10.0)\n[individual_parameters]\n  NTR = 2\n  MTT = TVMTT\n");
    assert_eq!(
        Structure::from_model(&t, Some(&m)).unwrap().transits,
        Some(TransitCount::Count(2))
    );
    // A non-integral fixed count is refused by name.
    let m = text(
        "[parameters]\n  theta TVNTR(2.5, 0.0, 64.0) FIX\n[individual_parameters]\n  NTR = TVNTR\n",
    );
    let err = Structure::from_model(&t, Some(&m)).unwrap_err();
    assert!(err.contains("2.5") && err.contains("whole number"), "{err}");
    // Without the text, a chain reads as estimated.
    assert_eq!(
        Structure::from_template(&t).unwrap().transits,
        Some(TransitCount::N)
    );
    // The `pk` line alone is not enough for the tests that only need it.
    assert_eq!(
        Structure::from_model(&template("pk one_cpt_oral(cl=CL, v=V, ka=KA)"), Some(&m)).unwrap(),
        fo(0)
    );
}

#[test]
fn a_parents_bioavailability_binding_is_carried_across_a_swap() {
    // `f` is not a coordinate, and every template reads it: dropping it
    // would reset F to 1 and prune its θ while the row claims only the
    // compartment count changed.
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)");
    let d = Defaults {
        parameters: vec!["CL".into(), "V".into(), "KA".into(), "F".into()],
        ..defaults()
    };
    let spec = structural_spec(&fo(1), &fo(0), &parent, &lines(), &d, IivStrategy::NoAdd).unwrap();
    assert_eq!(
        spec.bindings.last().unwrap(),
        &("f".to_string(), "F".to_string())
    );
    assert!(spec.new_parameters.iter().all(|p| p.name != "F"));
    // …and to a bolus, where there is no absorption role at all.
    let iv = Structure {
        absorption: Absorption::Inst,
        ..fo(0)
    };
    let spec = structural_spec(&iv, &fo(0), &parent, &lines(), &d, IivStrategy::NoAdd).unwrap();
    assert_eq!(
        spec.bindings,
        vec![
            ("cl".to_string(), "CL".to_string()),
            ("v".to_string(), "V".to_string()),
            ("f".to_string(), "F".to_string())
        ]
    );
}

#[test]
fn a_different_transit_coordinate_rebinds_n_to_a_fresh_parameter() {
    // The parent's `n=NTR` is behind `theta TVNTR(3) FIX`; `TRANSITS(1)`
    // must not reuse that declaration — the edit layer binds an existing
    // name as it is — so `n` goes to a fresh `NTR2` and the old one is
    // left unreferenced for the pruner.
    let parent_t = template("pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)");
    let d = Defaults {
        parameters: vec!["CL".into(), "V".into(), "NTR".into(), "MTT".into()],
        theta_names: vec!["TVCL".into(), "TVV".into(), "TVNTR".into(), "TVMTT".into()],
        ..defaults()
    };
    let three = Structure {
        transits: Some(TransitCount::Count(3)),
        ..fo(0)
    };
    let one = Structure {
        transits: Some(TransitCount::Count(1)),
        ..fo(0)
    };
    let spec = structural_spec(&one, &three, &parent_t, &lines(), &d, IivStrategy::NoAdd).unwrap();
    assert!(
        spec.bindings
            .contains(&("n".to_string(), "NTR2".to_string())),
        "{:?}",
        spec.bindings
    );
    assert!(spec
        .bindings
        .contains(&("mtt".to_string(), "MTT".to_string())));
    let n = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "NTR2")
        .unwrap();
    assert!(n.fixed);
    assert_eq!((n.theta.as_str(), n.init), ("TVNTR2", 1.0));
    // Fixed → estimated: a fresh free θ at Pharmpy's init.
    let estimated = Structure {
        transits: Some(TransitCount::N),
        ..fo(0)
    };
    let spec = structural_spec(
        &estimated,
        &three,
        &parent_t,
        &lines(),
        &d,
        IivStrategy::NoAdd,
    )
    .unwrap();
    let n = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "NTR2")
        .unwrap();
    assert!(!n.fixed);
    assert_eq!(n.init, 2.0);
    // Estimated → fixed likewise.
    let spec = structural_spec(
        &three,
        &estimated,
        &parent_t,
        &lines(),
        &d,
        IivStrategy::NoAdd,
    )
    .unwrap();
    let n = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "NTR2")
        .unwrap();
    assert!(n.fixed && n.init == 3.0);
    // The same coordinate keeps the parent's declaration untouched.
    let spec =
        structural_spec(&three, &three, &parent_t, &lines(), &d, IivStrategy::NoAdd).unwrap();
    assert!(spec
        .bindings
        .contains(&("n".to_string(), "NTR".to_string())));
    assert!(spec.new_parameters.is_empty());
    // A chain to first-order: no `n`, a new KA; the chain's parameters are
    // the pruner's.
    let spec =
        structural_spec(&fo(0), &three, &parent_t, &lines(), &d, IivStrategy::NoAdd).unwrap();
    assert_eq!(spec.template, "one_cpt_oral");
    assert!(!spec.bindings.iter().any(|(r, _)| r == "n" || r == "mtt"));
    assert_eq!(spec.new_parameters[0].name, "KA");
}

#[test]
fn defaults_of_text_read_the_parents_own_declarations() {
    let m = text(
        "[parameters]\n  theta TVCL(0.5, 0.0, 10.0)\n  theta TVQ(0.2, 0.0, 1.0) FIX\n  \
         omega ETA_CL ~ 0.1\n  block_omega (ETA_V, ETA_KA) = [0.1, 0.01, 0.2]\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 10\n  if (WT > 70) {\n  Q = TVQ\n  }\n",
    );
    let d = Defaults::of_text(&m, 0.5);
    assert_eq!(d.parameters, vec!["CL", "V", "Q"]);
    assert_eq!(d.theta_names, vec!["TVCL", "TVQ"]);
    assert_eq!(d.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
    assert_eq!(d.theta_init["TVQ"], 0.2);
    assert_eq!(d.t_first, 0.5);
    let decls = theta_decls_of(&m);
    assert!(decls[1].fixed && !decls[0].fixed);
}
