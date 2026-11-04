//! #2531: the CTMM sums and the CTMM record check walk `model.endpoints` in ascending CMT
//! order on every compile.
//!
//! A model can carry several CTMM endpoints: the parser reads the unnamed `[markov_model]`
//! and every named `[markov_model LABEL]`, each keyed by its CMT. Each test recompiles the
//! model inside its loop — one compile reuses one map and cannot fail.

use super::{ctmm_subject_nll, CtmmTerm};
use crate::parser::model_parser::parse_model_string;
use crate::survival::endpoint_order_test_helpers::{fold, order_sensitive, COMPILES, ETA};
use crate::types::{CompiledModel, ObsRecord, Population, Subject};

/// Three two-state CTMM endpoints on CMT 5, 6, 7, declared in descending CMT order, with
/// different intensities and one shared η.
fn three_ctmm_src() -> String {
    let block = |name: &str, cmt: usize, a: &str, b: &str| {
        format!(
            "[markov_model {name}]\n  type   = ctmm\n  cmt    = {cmt}\n  states = [off=0, on=1]\n  \
             transition off -> on = exp({a} + ETA_Q)\n  transition on -> off = exp({b})\n"
        )
    };
    format!(
        "[parameters]\n  theta A1(-0.7, -6.0, 3.0)\n  theta B1(-1.2, -6.0, 3.0)\n  \
         theta A2(-2.3, -6.0, 3.0)\n  theta B2(0.4, -6.0, 3.0)\n  theta A3(-4.1, -6.0, 3.0)\n  \
         theta B3(-0.05, -6.0, 3.0)\n  omega ETA_Q ~ 0.2\n{}{}{}",
        block("c", 7, "A3", "B3"),
        block("b", 6, "A2", "B2"),
        block("a", 5, "A1", "B1"),
    )
}

fn state(time: f64, state: usize, cmt: usize) -> ObsRecord {
    ObsRecord::DiscreteState {
        time,
        raw_time: time,
        state,
        cmt,
    }
}

/// Six subjects, each with an irregular state sequence on every one of the three CMTs.
fn three_ctmm_subjects() -> Vec<Subject> {
    let times = [0.0, 0.37, 1.9, 2.25, 5.1, 7.77];
    (0..6)
        .map(|i| {
            let obs_records = [5usize, 6, 7]
                .iter()
                .flat_map(|&cmt| {
                    times.iter().enumerate().map(move |(j, &t)| {
                        state(t * (1.0 + 0.1 * i as f64), (i + j + cmt) % 2, cmt)
                    })
                })
                .collect();
            Subject {
                id: format!("{}", i + 1),
                obs_records,
                ..Default::default()
            }
        })
        .collect()
}

/// A fresh compile keeping only the endpoint on `cmt`.
fn compile_only(src: &str, cmt: usize) -> CompiledModel {
    let mut m = parse_model_string(src).expect("parse");
    m.endpoints.retain(|&k, _| k == cmt);
    assert_eq!(m.endpoints.len(), 1, "no endpoint on CMT {cmt}");
    m
}

/// `ctmm_subject_nll` (the value path) and `CtmmTerm::eval_shared` (the dual walk's term) are
/// each the ascending-CMT sum of the per-endpoint terms, bit for bit, on every one of
/// `COMPILES` compiles of a three-endpoint CTMM model.
///
/// Straddle: some subject's three terms give other bits in another order, asserted per
/// path. Mutations: a per-call `std::collections::HashMap` walk in `ctmm_subject_nll`
/// reddens the `value` leg; in `CtmmTerm::eval_shared`, the `dual term` leg.
#[test]
fn ctmm_sums_endpoints_in_ascending_cmt_order() {
    let src = three_ctmm_src();
    let m = parse_model_string(&src).expect("parse");
    let mut cmts: Vec<usize> = m.endpoints.keys().copied().collect();
    cmts.sort_unstable();
    assert_eq!(cmts, vec![5, 6, 7], "three CTMM endpoints parse");
    let theta = m.default_params.theta.clone();
    let subjects = three_ctmm_subjects();
    let singles: Vec<_> = cmts.iter().map(|&c| compile_only(&src, c)).collect();

    type Path = fn(&CompiledModel, &Subject, &[f64]) -> f64;
    let value: Path = |m, s, theta| ctmm_subject_nll(m, s, theta, &ETA);
    let dual: Path = |m, s, theta| {
        CtmmTerm {
            model: m,
            subject: s,
        }
        .eval_shared::<f64>(theta, &ETA, None)
        .expect("term")
    };
    for (label, path) in [("value", value), ("dual term", dual)] {
        let terms: Vec<Vec<f64>> = subjects
            .iter()
            .map(|s| singles.iter().map(|ms| path(ms, s, &theta)).collect())
            .collect();
        assert!(
            terms.iter().flatten().all(|t| t.is_finite() && *t < 1e10),
            "{label}: {terms:?}"
        );
        assert!(
            terms.iter().any(|t| order_sensitive(0.0, t)),
            "{label}: no subject's sum depends on the order: the pin cannot fail"
        );
        for k in 0..COMPILES {
            let m = parse_model_string(&src).expect("parse");
            for (s, t) in subjects.iter().zip(&terms) {
                let (got, want) = (path(&m, s, &theta), fold(0.0, t));
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{label}, compile {k}, subject {}: {got:e} is not the ascending-CMT sum {want:e}",
                    s.id
                );
            }
        }
    }
}

/// With two CTMM endpoints that both have an undeclared state code, `check_ctmm_records`
/// pushes one `E_CTMM_RECORDS` per endpoint, in ascending CMT order, on every compile.
///
/// Straddle: each endpoint alone is refused. Mutation: a per-call
/// `std::collections::HashMap` walk in `check_ctmm_records`.
#[test]
fn ctmm_record_refusals_come_in_ascending_cmt_order() {
    let src = three_ctmm_src();
    // State code 9 is declared by no endpoint; CMT 7 is clean.
    let pop = Population {
        subjects: vec![Subject {
            id: "1".into(),
            obs_records: vec![
                state(0.0, 0, 5),
                state(1.0, 9, 5),
                state(0.0, 0, 6),
                state(1.0, 9, 6),
                state(0.0, 0, 7),
                state(1.0, 1, 7),
            ],
            ..Default::default()
        }],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let refused_cmts = |m: &CompiledModel| -> Vec<usize> {
        crate::api::check_model_data(m, &pop)
            .iter()
            .filter(|d| d.code == "E_CTMM_RECORDS")
            .map(|d| {
                [5usize, 6, 7]
                    .into_iter()
                    .filter(|c| d.message.contains(&format!("cmt = {c}")))
                    .collect::<Vec<_>>()
            })
            .map(|named| {
                assert_eq!(named.len(), 1, "one CMT per message: {named:?}");
                named[0]
            })
            .collect()
    };
    for c in [5, 6] {
        assert_eq!(
            refused_cmts(&compile_only(&src, c)),
            vec![c],
            "CMT {c} alone"
        );
    }
    for k in 0..COMPILES {
        let m = parse_model_string(&src).expect("parse");
        assert_eq!(refused_cmts(&m), vec![5, 6], "compile {k}");
    }
}
