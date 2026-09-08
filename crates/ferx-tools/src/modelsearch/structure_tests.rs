//! Tier-1 tests for the structural coordinates, Pharmpy's move rules and
//! the structure → `pk` line derivation (#1181).

use std::collections::HashMap;

use ferx_core::edit::{EliminationForm, InputForm, StructuralEngine};
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
        elimination: Elimination::Fo,
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
                elimination: Elimination::Fo,
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
    // An `ode_template` line parses to the same shape and names a template
    // the search could otherwise read a structure off — but `SetStructural`
    // writes a `pk` line and refuses anything else, so it is refused here,
    // before the base model is fitted (#1256).
    let err = Structure::from_template(&template(
        "ode_template two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, \
                                            ka=KA)",
    ))
    .unwrap_err();
    assert!(
        err.contains("`ode_template two_cpt_oral(...)`")
            && err.contains("cannot recover them from an equation"),
        "{err}"
    );
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
        elimination: Elimination::Fo,
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
        elimination: Elimination::Fo,
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
fn apply_moves_one_coordinate_and_features_lists_all_five() {
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
        s.apply(&FeatureKey::Elimination(Elimination::Mm))
            .elimination,
        Elimination::Mm
    );
    assert_eq!(
        s.features(),
        vec![
            FeatureKey::Absorption(Absorption::Fo),
            FeatureKey::Elimination(Elimination::Fo),
            FeatureKey::Peripherals(0),
            FeatureKey::Transits(TransitCount::Count(0)),
            FeatureKey::Lagtime(false),
        ]
    );
    assert_eq!(
        s.feature_vector().render(),
        "ABSORPTION=FO;ELIMINATION=FO;LAGTIME=OFF;PERIPHERALS=0;TRANSITS=0"
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
fn space_features_reads_elimination_and_refuses_non_structural_statements() {
    // `ELIMINATION(FO)` is a value like any other now (#1257): a space that
    // names it can move a saturable candidate back to first-order.
    assert_eq!(
        keys("ELIMINATION(FO); PERIPHERALS(1)"),
        vec![
            FeatureKey::Elimination(Elimination::Fo),
            FeatureKey::Peripherals(1)
        ]
    );
    assert_eq!(
        keys("ELIMINATION([MM,MIX-FO-MM,ZO])"),
        vec![
            FeatureKey::Elimination(Elimination::MixFoMm),
            FeatureKey::Elimination(Elimination::Mm),
            FeatureKey::Elimination(Elimination::Zo),
        ],
        "sorted by (category, str(argument)) as Pharmpy sorts its dictionary"
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
    // `ZO` and `WEIBULL` are ODE candidates now (#1257); `SEQ-ZO-FO` is the
    // one absorption mode with no spelling at all, so it is what must still
    // error here if the coverage check is ever bypassed.
    assert_eq!(
        keys("ABSORPTION([ZO,WEIBULL])"),
        vec![
            FeatureKey::Absorption(Absorption::Weibull),
            FeatureKey::Absorption(Absorption::Zo),
        ]
    );
    let err = space_features(&Mfl::parse("ABSORPTION(SEQ-ZO-FO)").unwrap()).unwrap_err();
    assert!(err.contains("SEQ-ZO-FO"), "{err}");
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
        elimination: Elimination::Fo,
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
    observed(times, &[])
}

/// A one-subject population with observation times and, when given,
/// observations — the range the saturable-elimination inits read (#1257).
fn observed(times: &[f64], values: &[f64]) -> Population {
    Population {
        subjects: vec![Subject {
            id: "1".into(),
            obs_times: times.to_vec(),
            observations: values.to_vec(),
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
        elimination: Elimination::Fo,
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
    let d = defaults().of_text(&m);
    assert_eq!(d.parameters, vec!["CL", "V", "Q"]);
    assert_eq!(d.theta_names, vec!["TVCL", "TVQ"]);
    assert_eq!(d.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
    assert_eq!(d.theta_init["TVQ"], 0.2);
    assert_eq!(d.t_first, 0.5);
    let decls = theta_decls_of(&m);
    assert!(decls[1].fixed && !decls[0].fixed);
}

// ── the ODE candidate family (#1257) ────────────────────────────────────────

/// `fo(peripherals)` with one coordinate moved off its analytic value.
fn with_absorption(a: Absorption) -> Structure {
    Structure {
        absorption: a,
        ..fo(0)
    }
}

fn with_elimination(e: Elimination) -> Structure {
    Structure {
        elimination: e,
        ..fo(0)
    }
}

#[test]
fn the_engine_is_ode_exactly_when_a_coordinate_has_no_analytic_template() {
    assert_eq!(fo(0).engine(), Engine::Pk);
    assert_eq!(with_absorption(Absorption::Inst).engine(), Engine::Pk);
    assert_eq!(with_elimination(Elimination::Fo).engine(), Engine::Pk);
    for a in [Absorption::Zo, Absorption::Weibull] {
        assert_eq!(with_absorption(a).engine(), Engine::Ode, "{}", a.label());
    }
    for e in [Elimination::Zo, Elimination::Mm, Elimination::MixFoMm] {
        assert_eq!(with_elimination(e).engine(), Engine::Ode, "{}", e.label());
    }
    // The template carries the engine, so a caller reading `template()` alone
    // cannot pick the wrong one.
    assert_eq!(fo(0).template().unwrap().engine, Engine::Pk);
    assert_eq!(
        with_elimination(Elimination::Mm).template().unwrap().engine,
        Engine::Ode
    );
}

#[test]
fn an_ode_candidate_carries_its_own_cost_and_start_budget() {
    // Ordering is the property the runner needs: the saturable eliminations
    // are the long poles, then the ODE absorptions, then everything analytic.
    assert!(with_elimination(Elimination::Mm).cost() > with_absorption(Absorption::Zo).cost());
    assert!(with_absorption(Absorption::Zo).cost() > fo(0).cost());
    assert_eq!(fo(0).cost(), 1.0);
    // Only a saturable elimination asks for extra starts — a zero-order or
    // Weibull absorption is not the multimodal case.
    assert_eq!(fo(0).starts(), None);
    assert_eq!(with_absorption(Absorption::Weibull).starts(), None);
    for e in [Elimination::Zo, Elimination::Mm, Elimination::MixFoMm] {
        assert_eq!(
            with_elimination(e).starts(),
            Some(MM_STARTS),
            "{}",
            e.label()
        );
    }
}

#[test]
fn zero_order_and_weibull_absorption_render_the_bolus_template() {
    // Both feed `central` directly, so the depot and its `ka` are gone —
    // Pharmpy's `set_zero_order_absorption` removes the same compartment.
    let t = with_absorption(Absorption::Zo).template().unwrap();
    assert_eq!(t.name, "one_cpt_iv");
    assert_eq!(t.roles, vec!["cl", "v", "dur"]);
    let t = Structure {
        absorption: Absorption::Weibull,
        ..fo(1)
    }
    .template()
    .unwrap();
    assert_eq!(t.name, "two_cpt_iv");
    assert_eq!(t.roles, vec!["cl", "v1", "q", "v2", "td", "beta"]);
    // …and every one of the added roles is read by the `[odes]` override
    // rather than bound on the line.
    for role in ["dur", "td", "beta"] {
        assert!(ODE_ONLY_ROLES.contains(&role), "{role}");
    }
}

#[test]
fn saturable_elimination_adds_its_parameters_to_any_template() {
    // Elimination is orthogonal: it composes with every absorption and every
    // compartment count, because the override replaces only `central`.
    let mm = Structure {
        elimination: Elimination::Mm,
        ..fo(2)
    }
    .template()
    .unwrap();
    assert_eq!(mm.name, "three_cpt_oral");
    assert_eq!(
        mm.roles,
        vec!["cl", "v1", "q2", "v2", "q3", "v3", "ka", "km"]
    );
    // `MIX-FO-MM` keeps `CL` first-order and adds a saturable clearance.
    let mix = with_elimination(Elimination::MixFoMm).template().unwrap();
    assert_eq!(mix.roles, vec!["cl", "v", "ka", "clmm", "km"]);
    // Zero-order elimination takes the same parameters as Michaelis-Menten;
    // only the `FIX` differs.
    assert_eq!(
        with_elimination(Elimination::Zo).template().unwrap().roles,
        with_elimination(Elimination::Mm).template().unwrap().roles
    );
    // A transit chain and a saturable elimination are an allowed pair.
    let transit_mm = Structure {
        elimination: Elimination::Mm,
        transits: Some(TransitCount::N),
        ..fo(0)
    }
    .template()
    .unwrap();
    assert_eq!(transit_mm.name, "one_cpt_transit");
    assert_eq!(transit_mm.roles, vec!["cl", "v", "n", "mtt", "km"]);
}

#[test]
fn the_new_absorptions_take_pharmpys_unsupported_pairs() {
    // ZO and WEIBULL each model the whole delay themselves, so a transit
    // chain in front of either is refused; WEIBULL additionally refuses a
    // lag time (Pharmpy's `not_supported_combo`), while ZO takes one.
    for a in [Absorption::Zo, Absorption::Weibull] {
        let why = Structure {
            absorption: a,
            transits: Some(TransitCount::N),
            ..fo(0)
        }
        .unbuildable()
        .unwrap_or_else(|| panic!("{} with transits must be refused", a.label()));
        assert!(why.contains("transit chain"), "{why}");
        assert!(why.contains(a.label()), "the reason names the mode: {why}");
    }
    let why = Structure {
        absorption: Absorption::Weibull,
        lagtime: true,
        ..fo(0)
    }
    .unbuildable()
    .expect("Weibull with a lag time is refused");
    assert!(why.contains("lag time"), "{why}");
    assert!(
        Structure {
            absorption: Absorption::Zo,
            lagtime: true,
            ..fo(0)
        }
        .unbuildable()
        .is_none(),
        "Pharmpy lags a zero-order infusion, and so does ferx"
    );
    // The move table agrees with the structure check, in both orders.
    let funcs = keys("ABSORPTION([ZO,WEIBULL]); LAGTIME(ON); TRANSITS(N)");
    let zo = FeatureKey::Absorption(Absorption::Zo);
    let wb = FeatureKey::Absorption(Absorption::Weibull);
    let lag = FeatureKey::Lagtime(true);
    let tn = FeatureKey::Transits(TransitCount::N);
    for (a, b) in [(zo, tn), (wb, tn), (wb, lag)] {
        assert!(!allowed(&a, &[b], &funcs, &fo(0)), "{a} after {b}");
        assert!(!allowed(&b, &[a], &funcs, &fo(0)), "{b} after {a}");
        assert!(!combination_allowed(&[a, b], &fo(0)), "{a} with {b}");
    }
    assert!(combination_allowed(&[zo, lag], &fo(0)));
    // Elimination is in none of the pairs.
    let mm = FeatureKey::Elimination(Elimination::Mm);
    for other in [zo, wb, lag, tn] {
        assert!(combination_allowed(&[mm, other], &fo(0)), "MM with {other}");
    }
}

#[test]
fn onto_space_moves_elimination_in_pharmpys_position() {
    // Pharmpy's `least_number_of_transformations` walks absorption,
    // elimination, transits, peripherals, lagtime — so a base outside a
    // space that names only saturable eliminations is moved onto it, and the
    // elimination move comes second.
    assert_eq!(
        onto_space(&fo(0), &keys("ELIMINATION(MM)")),
        vec![FeatureKey::Elimination(Elimination::Mm)]
    );
    assert_eq!(
        onto_space(&fo(0), &keys("ABSORPTION(INST); ELIMINATION(MM)")),
        vec![
            FeatureKey::Absorption(Absorption::Inst),
            FeatureKey::Elimination(Elimination::Mm),
        ]
    );
    // A space that lists the base's own elimination leaves it alone.
    assert!(onto_space(&fo(0), &keys("ELIMINATION([FO,MM])")).is_empty());
}

/// The Weibull scale init is `MAT / Γ(1 + 1/k)`, and the divisor is stored as
/// a constant because `k` is fixed at 1.5. Pinned against ferx's own
/// `ln_gamma` — a different implementation from the literal — so a mistyped
/// digit or a wrong argument (Γ(1.5) = 0.8862 is 1.8% away) fails.
#[test]
fn weibull_scale_matches_the_gamma_identity() {
    let want = ferx_core::stats::special::ln_gamma(1.0 + 1.0 / WEIBULL_SHAPE_INIT).exp();
    // Realised error 1.1e-16 — one ULP of the exp round-trip, not a
    // tolerance for disagreement.
    assert!(
        (GAMMA_1_PLUS_1_OVER_SHAPE - want).abs() < 1e-12,
        "{GAMMA_1_PLUS_1_OVER_SHAPE} vs {want}"
    );
    assert!(want.is_finite());
}

/// A one-subject population whose observations span a known range, for the
/// saturable-elimination inits.
fn dv_defaults() -> Defaults {
    Defaults::new(
        vec!["CL".into(), "V".into(), "KA".into()],
        vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
        vec![0.2, 10.0, 1.5],
        vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
        &observed(&[0.0, 0.5, 1.0, 2.0], &[2.0, 8.0, 20.0, 5.0]),
    )
}

#[test]
fn michaelis_menten_reuses_the_parents_clearance_and_declares_km_from_the_data() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let d = dv_defaults();
    assert_eq!((d.dv_min, d.dv_max), (2.0, 20.0));
    let spec = structural_spec(
        &with_elimination(Elimination::Mm),
        &fo(0),
        &parent,
        &lines(),
        &d,
        IivStrategy::AbsorptionDelay,
    )
    .unwrap();
    // The disposition line is unchanged — the clearance *is* the
    // Michaelis-Menten clearance, which is what carries its estimate and its
    // η across the move (Pharmpy renames `CL` to `CLMM`; ferx keeps the name).
    assert_eq!(
        spec.bindings,
        vec![
            ("cl".to_string(), "CL".to_string()),
            ("v".to_string(), "V".to_string()),
            ("ka".to_string(), "KA".to_string()),
        ]
    );
    assert!(
        !spec.bindings.iter().any(|(r, _)| r == "km"),
        "`km` is read by the override, not bound on the line: {:?}",
        spec.bindings
    );
    let km = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "KM")
        .expect("KM is declared");
    assert_eq!(km.init, 10.0, "max(DV)/2");
    assert_eq!(km.upper, 30.0, "1.5 * max(DV)");
    assert!(!km.fixed);
    assert!(km.iiv.is_none(), "absorption_delay adds η to MDT only");
    assert_eq!(
        spec.engine,
        StructuralEngine::Ode {
            input: InputForm::Template,
            elimination: EliminationForm::MichaelisMenten { km: "KM".into() },
        }
    );
}

#[test]
fn zero_order_elimination_is_michaelis_menten_with_km_fixed_low() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let spec = structural_spec(
        &with_elimination(Elimination::Zo),
        &fo(0),
        &parent,
        &lines(),
        &dv_defaults(),
        IivStrategy::NoAdd,
    )
    .unwrap();
    let km = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "KM")
        .expect("KM is declared");
    assert_eq!(km.init, 0.02, "min(DV)/100");
    assert!(km.fixed, "a free KM here would be Michaelis-Menten, not ZO");
    // The *term* is the same one Michaelis-Menten writes: the two differ by
    // the `FIX`, which is a declaration rather than an equation.
    assert_eq!(
        spec.engine,
        StructuralEngine::Ode {
            input: InputForm::Template,
            elimination: EliminationForm::MichaelisMenten { km: "KM".into() },
        }
    );
}

#[test]
fn mixed_elimination_adds_a_saturable_clearance_at_half_the_first_order_one() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let spec = structural_spec(
        &with_elimination(Elimination::MixFoMm),
        &fo(0),
        &parent,
        &lines(),
        &dv_defaults(),
        IivStrategy::NoAdd,
    )
    .unwrap();
    let clmm = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "CLMM")
        .expect("CLMM is declared");
    // `lines()` puts `CL = TVCL * exp(ETA_CL)` behind `TVCL = 0.2`.
    assert_eq!(clmm.init, 0.1, "CL/2");
    assert_eq!(
        spec.engine,
        StructuralEngine::Ode {
            input: InputForm::Template,
            elimination: EliminationForm::MixedFoMm {
                clmm: "CLMM".into(),
                km: "KM".into(),
            },
        }
    );
}

#[test]
fn zero_order_absorption_declares_pharmpys_duration_and_drops_the_depot() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let d = dv_defaults();
    assert_eq!(d.t_first, 0.5);
    let spec = structural_spec(
        &with_absorption(Absorption::Zo),
        &fo(0),
        &parent,
        &lines(),
        &d,
        IivStrategy::AbsorptionDelay,
    )
    .unwrap();
    assert_eq!(spec.template, "one_cpt_iv");
    assert_eq!(
        spec.bindings,
        vec![
            ("cl".to_string(), "CL".to_string()),
            ("v".to_string(), "V".to_string()),
        ],
        "`ka` is not a role of the bolus template, so the depot is gone"
    );
    let dur = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "DUR")
        .expect("DUR is declared");
    // Pharmpy: `D1 = 2·MAT`, `MAT = 2·t_first`.
    assert_eq!(dur.init, 2.0 * (2.0 * 0.5));
    assert_eq!(
        spec.engine,
        StructuralEngine::Ode {
            input: InputForm::ZeroOrder { dur: "DUR".into() },
            elimination: EliminationForm::FirstOrder,
        }
    );
}

#[test]
fn weibull_absorption_declares_the_scale_and_shape() {
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let spec = structural_spec(
        &with_absorption(Absorption::Weibull),
        &fo(0),
        &parent,
        &lines(),
        &dv_defaults(),
        IivStrategy::AbsorptionDelay,
    )
    .unwrap();
    let td = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "TD")
        .expect("TD is declared");
    let beta = spec
        .new_parameters
        .iter()
        .find(|p| p.name == "BETA")
        .expect("BETA is declared");
    assert_eq!(beta.init, WEIBULL_SHAPE_INIT);
    assert_eq!(td.init, 2.0 * 0.5 / GAMMA_1_PLUS_1_OVER_SHAPE);
    assert_eq!(
        spec.engine,
        StructuralEngine::Ode {
            input: InputForm::Weibull {
                td: "TD".into(),
                beta: "BETA".into(),
            },
            elimination: EliminationForm::FirstOrder,
        }
    );
}

#[test]
fn an_ode_parameter_the_parent_already_declares_is_reused_not_redeclared() {
    // The step that moves MM → MIX-FO-MM must keep the `KM` its parent
    // converged on rather than re-deriving one from the data — the seeding
    // step has already written the parent's estimate into that θ.
    let parent = template("pk one_cpt_oral(cl=CL, v=V, ka=KA)");
    let d = Defaults {
        parameters: vec!["CL".into(), "V".into(), "KA".into(), "KM".into()],
        ..dv_defaults()
    };
    let spec = structural_spec(
        &with_elimination(Elimination::MixFoMm),
        &with_elimination(Elimination::Mm),
        &parent,
        &lines(),
        &d,
        IivStrategy::NoAdd,
    )
    .unwrap();
    assert!(
        spec.new_parameters.iter().all(|p| p.name != "KM"),
        "KM is the parent's: {:?}",
        spec.new_parameters
    );
    assert!(spec.new_parameters.iter().any(|p| p.name == "CLMM"));
    assert_eq!(
        spec.engine,
        StructuralEngine::Ode {
            input: InputForm::Template,
            elimination: EliminationForm::MixedFoMm {
                clmm: "CLMM".into(),
                km: "KM".into(),
            },
        }
    );
}

#[test]
fn a_move_back_to_first_order_elimination_writes_an_analytic_candidate() {
    // The reverse move exists — `ELIMINATION([FO,MM])` names both — and it
    // must produce a `pk` line with no override, or the candidate would carry
    // a saturable term it no longer declares a `KM` for.
    let parent = template("ode_template one_cpt_oral(cl=CL, v=V, ka=KA)");
    let d = Defaults {
        parameters: vec!["CL".into(), "V".into(), "KA".into(), "KM".into()],
        ..dv_defaults()
    };
    let spec = structural_spec(
        &fo(0),
        &with_elimination(Elimination::Mm),
        &parent,
        &lines(),
        &d,
        IivStrategy::NoAdd,
    )
    .unwrap();
    assert_eq!(spec.engine, StructuralEngine::Pk);
    assert_eq!(spec.template, "one_cpt_oral");
    assert!(spec.new_parameters.is_empty());
}
