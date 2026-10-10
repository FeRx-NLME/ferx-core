//! Co-timed records follow NONMEM's file order when an `EVID=2` row is in the tie (#1810).
//!
//! NONMEM advances the system into a shared `TIME` at the first record there in file order,
//! under that record's `$PK` values. ferx's engines order co-timed records by kind
//! (`DoseRecord < Dose < PkOnly < Obs`), so before #1810 an observation listed before a
//! co-timed `EVID=2` row was advanced under the `EVID=2` row's covariates, and an `EVID=2`
//! row listed before a dose under the dose row's. The reader now moves the first record one
//! ULP below the tie.
//!
//! The seven cases are #1810's. Every value is checked against a **closed form** of the
//! NONMEM convention, computed here outside every engine: records in file order, the
//! interval ending at a record runs on that record's `X`. The closed form is in turn pinned
//! to the values NONMEM 7.6.0 printed in the #1809/#1810 triage (`MAXEVAL=0`, `PRED`), so
//! the reference itself cannot drift. No test here compares two engines: the f64
//! predictors, the analytic sensitivity providers and their FD reference all read the order
//! from the same `Subject`, so a parity test agreed with the wrong order before the fix.
//!
//! Before the fix, cases 1, 4 and 6 failed on both models. Cases 2, 3, 5 and 7 are the
//! must-stay-green side: they guard against a nudge that moves too much (an `EVID=2` listed
//! first must not move the observation, a dose listed first must not move anything, two
//! observations keep file order).

use super::predict;
use crate::io::datareader::read_nonmem_csv;
use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, Population};
use std::io::Write;

/// #1810's ODE model: `X` drives the ODE field through `RATE` and the readout through `RO`.
const ODE_MODEL: &str = "[parameters]\n  theta TVR(1.0, 0.01, 100.0)\n  \
     theta SLOPE(10.0, 0.0, 100.0)\n  omega ETA_R ~ 0.01\n  sigma ADD ~ 1.0 (sd)\n\
     [individual_parameters]\n  RATE = TVR * X * exp(ETA_R)\n  RO = SLOPE * X\n\
     [structural_model]\n  ode(states=[A])\n[odes]\n  d/dt(A) = RATE\n\
     [scaling]\n  y = A + RO\n[covariates]\n  X continuous\n\
     [error_model]\n  DV ~ additive(ADD)\n";

/// #1810's analytical model: `CL = TVCL·X`, `V` constant, so the readout `A/V` cannot
/// depend on the tie and only the state can.
const PK_MODEL: &str = "[parameters]\n  theta TVCL(2.0, 0.01, 100.0)\n  \
     theta TVV(10.0, 0.1, 1000.0)\n  omega ETA_CL ~ 0.01\n  sigma PROP ~ 0.01\n\
     [individual_parameters]\n  CL = TVCL * X * exp(ETA_CL)\n  V = TVV\n\
     [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[covariates]\n  X continuous\n\
     [error_model]\n  DV ~ proportional(PROP)\n";

/// One record: `(TIME, EVID, X, AMT)`.
type Rec = (f64, u32, f64, f64);

/// #1810's seven cases, each a subject, with every time shifted by `t0`.
fn cases(t0: f64) -> Vec<(&'static str, Vec<Rec>)> {
    let c = |v: Vec<Rec>| {
        v.into_iter()
            .map(|(t, e, x, a)| (t + t0, e, x, a))
            .collect()
    };
    vec![
        (
            "1 obs, EVID=2",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 0, 1.0, 0.0),
                (4.0, 2, 2.0, 0.0),
                (8.0, 0, 2.0, 0.0),
            ]),
        ),
        (
            "2 EVID=2, obs",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 2, 2.0, 0.0),
                (4.0, 0, 1.0, 0.0),
                (8.0, 0, 2.0, 0.0),
            ]),
        ),
        (
            "3 obs, EVID=2 at +1e-6",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 0, 1.0, 0.0),
                (4.0 + 1e-6, 2, 2.0, 0.0),
                (8.0, 0, 2.0, 0.0),
            ]),
        ),
        (
            "4 obs1, EVID=2, obs2",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 0, 1.0, 0.0),
                (4.0, 2, 2.0, 0.0),
                (4.0, 0, 3.0, 0.0),
            ]),
        ),
        (
            "5 dose, EVID=2",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 1, 3.0, 50.0),
                (4.0, 2, 2.0, 0.0),
                (8.0, 0, 3.0, 0.0),
            ]),
        ),
        (
            "6 EVID=2, dose",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 2, 2.0, 0.0),
                (4.0, 1, 3.0, 50.0),
                (8.0, 0, 3.0, 0.0),
            ]),
        ),
        (
            "7 obs1, obs2 (control)",
            c(vec![
                (0.0, 1, 1.0, 100.0),
                (4.0, 0, 1.0, 0.0),
                (4.0, 0, 3.0, 0.0),
            ]),
        ),
    ]
}

/// The NONMEM convention in closed form, at the typical values: walk the records in file
/// order; the interval ending at a record runs on that record's `X`; a dose adds its
/// amount; an observation reads the state. `advance(a, x, dt)` is the model's exact
/// propagator, `read(a, x)` its readout.
fn nonmem_reference(
    recs: &[Rec],
    advance: impl Fn(f64, f64, f64) -> f64,
    read: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let (mut a, mut t_prev, mut out) = (0.0, recs[0].0, Vec::new());
    for &(t, evid, x, amt) in recs {
        a = advance(a, x, t - t_prev);
        t_prev = t;
        match evid {
            1 => a += amt,
            0 => out.push(read(a, x)),
            _ => {}
        }
    }
    out
}

/// ODE: `dA/dt = TVR·X` with TVR = 1, readout `A + 10·X`.
fn ode_reference(recs: &[Rec]) -> Vec<f64> {
    nonmem_reference(recs, |a, x, dt| a + x * dt, |a, x| a + 10.0 * x)
}

/// Analytical: `dA/dt = −(2·X/10)·A`, readout `A / 10`.
fn pk_reference(recs: &[Rec]) -> Vec<f64> {
    nonmem_reference(
        recs,
        |a, x, dt| a * (-(0.2 * x) * dt).exp(),
        |a, _| a / 10.0,
    )
}

fn csv_of(cases: &[(&str, Vec<Rec>)]) -> String {
    let mut s = String::from("ID,TIME,DV,EVID,MDV,AMT,CMT,X\n");
    for (i, (_, recs)) in cases.iter().enumerate() {
        for &(t, evid, x, amt) in recs {
            // `{:?}` keeps every digit of `4.000001` and `17524.0`, so the tie is exact.
            let amt = if evid == 1 {
                format!("{amt:?}")
            } else {
                ".".to_string()
            };
            let mdv = u32::from(evid != 0);
            s.push_str(&format!("{},{t:?},0,{evid},{mdv},{amt},1,{x:?}\n", i + 1));
        }
    }
    s
}

fn population(cases: &[(&str, Vec<Rec>)]) -> Population {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv_of(cases).as_bytes()).unwrap();
    read_nonmem_csv(f.path(), None, None).expect("read the tie fixture")
}

fn preds_by_subject(model: &CompiledModel, pop: &Population) -> Vec<Vec<f64>> {
    let rows = predict(model, pop, &model.default_params).expect("predict");
    pop.subjects
        .iter()
        .map(|s| {
            rows.iter()
                .filter(|r| r.id == s.id)
                .map(|r| r.pred)
                .collect()
        })
        .collect()
}

/// Finite first, then within `tol` of `want`, with the worst error in the message.
fn assert_close(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: row count");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{label}: non-finite {got:?}"
    );
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < tol,
        "{label}: got {got:?}, NONMEM convention {want:?} (worst |error| {worst:.3e})"
    );
}

/// NONMEM 7.6.0's printed `PRED`s for the seven cases (#1809 triage), ODE then ADVAN1.
const NONMEM_ODE: [&[f64]; 7] = [
    &[114.0, 132.0],
    &[118.0, 136.0],
    &[114.0, 132.0],
    &[114.0, 134.0],
    &[204.0],
    &[200.0],
    &[114.0, 134.0],
];
const NONMEM_PK: [&[f64]; 7] = [
    &[4.49329, 0.90718],
    &[2.01897, 0.40762],
    &[4.49329, 0.90718],
    &[4.49329, 4.49329],
    &[0.53589],
    &[0.63675],
    &[4.49329, 4.49329],
];

#[test]
fn the_closed_form_reference_is_nonmems() {
    // The reference must be NONMEM's answer, not just a plausible one: pinned to the
    // triage table at its 5 printed significant figures. Case 3's `+1e-6` moves the ODE
    // values by O(1e-6) and the ADVAN1 ones by less, inside the print.
    for (i, (label, recs)) in cases(0.0).iter().enumerate() {
        assert_close(
            &format!("ODE {label}"),
            &ode_reference(recs),
            NONMEM_ODE[i],
            1e-5,
        );
        assert_close(
            &format!("ADVAN1 {label}"),
            &pk_reference(recs),
            NONMEM_PK[i],
            6e-6,
        );
    }
}

/// The seven cases on one model, at time offset `t0`, against the closed form.
fn check_all_cases(model_src: &str, reference: fn(&[Rec]) -> Vec<f64>, t0: f64, tol: f64) {
    let model = parse_model_string(model_src).expect("parse");
    let cs = cases(t0);
    let pop = population(&cs);
    for (s, ((label, recs), got)) in pop
        .subjects
        .iter()
        .zip(cs.iter().zip(preds_by_subject(&model, &pop)))
    {
        assert!(
            s.has_tv_covariates(),
            "{label}: X must vary, so the EVID=2 rows carry snapshots"
        );
        assert_close(&format!("{label} (t0 = {t0})"), &got, &reference(recs), tol);
    }
}

// Measured worst |PRED − closed form| over the seven cases, Linux x86_64. ODE: 0 at
// t0 = 0, 1.4e-11 at t0 = 17520. ADVAN1: 8.9e-16 at t0 = 0, 3.3e-12 at t0 = 17520. That
// last one is the moved record's own sub-ULP interval, k·A·ulp/V = 0.2 · 44.93 · 3.64e-12 /
// 10, on exactly the two cases whose observation moved (1 and 4). Bounds: 1e-9, except
// 1e-12 for ADVAN1 at t0 = 0. The defect they catch is 4 on the ODE model and 2.5 on
// ADVAN1 (case 1).
#[test]
fn ode_ties_follow_file_order() {
    check_all_cases(ODE_MODEL, ode_reference, 0.0, 1e-9);
}

#[test]
fn analytical_ties_follow_file_order() {
    check_all_cases(PK_MODEL, pk_reference, 0.0, 1e-12);
}

#[test]
fn ties_follow_file_order_at_large_time() {
    // The #1226 regime: at t = 17520 one ULP is 3.6e-12, above `EVENT_MATCH_TOL` and the
    // 1e-15 dedup bands, so a moved record is a distinct instant on every engine and its
    // sub-ULP interval is integrated explicitly rather than merged away.
    check_all_cases(ODE_MODEL, ode_reference, 17520.0, 1e-9);
    check_all_cases(PK_MODEL, pk_reference, 17520.0, 1e-9);
}

#[test]
fn the_analytic_providers_value_follows_file_order() {
    // The `Dual2` walks carry their own copy of the kind order (`sens/propagate`,
    // `sens/ode_provider`'s `K_PKONLY < K_OBS`). The reader fix reaches them through the
    // same `Subject`, and their *value* must match the closed form. A Dual2-vs-FD parity
    // test cannot see the order, since the FD reference reads the same Subject.
    let cs = cases(0.0);
    let pop = population(&cs);
    let ode = parse_model_string(ODE_MODEL).unwrap();
    let pk = parse_model_string(PK_MODEL).unwrap();
    for (s, (label, recs)) in pop.subjects.iter().zip(&cs) {
        let o = crate::sens::ode_provider::ode_subject_sensitivities(
            &ode,
            s,
            &ode.default_params.theta,
            &[0.0],
        )
        .unwrap_or_else(|| panic!("{label}: ODE model in analytic scope"));
        let f: Vec<f64> = o.obs.iter().map(|x| x.f).collect();
        assert_close(
            &format!("ODE provider {label}"),
            &f,
            &ode_reference(recs),
            1e-9,
        );
        let a =
            crate::sens::provider::subject_sensitivities(&pk, s, &pk.default_params.theta, &[0.0])
                .unwrap_or_else(|| panic!("{label}: analytical model in analytic scope"));
        let f: Vec<f64> = a.obs.iter().map(|x| x.f).collect();
        assert_close(
            &format!("analytic provider {label}"),
            &f,
            &pk_reference(recs),
            1e-12,
        );
    }
}
