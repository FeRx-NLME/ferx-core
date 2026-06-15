//! Equivalence between an analytical `pk NAME(...)` model and the same model
//! written as `ode_template NAME(...)` (#322 Phase 0b).
//!
//! `ode_template` is lowering sugar: it generates the standard disposition ODE
//! for the named model (the transforms codified in
//! `tests/analytical_ode_equivalence.rs`) and runs it through the ODE pipeline.
//! So `ode_template NAME(...)` must `predict()` identically to the analytical
//! `pk NAME(...)` it desugars from — to the same solver tolerance as the
//! hand-written ODE twin in `analytical_ode_equivalence.rs`. That hand-written
//! file already pins `pk == ode(...)`; this file pins `ode_template == pk`, so
//! together they certify `ode_template == ode(...)` (what a user would write by
//! hand) without re-asserting it.
//!
//! Runs on the default/CI feature set (no autodiff), fast (no `fit()`), so it
//! is not gated behind `slow-tests`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population, Subject};

mod common;

// Same tolerance rationale as `analytical_ode_equivalence.rs`: the RK45 solver
// reproduces the closed form only to ~reltol, and long multi-dose / SS
// trajectories accumulate per-step error.
const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;
const ACCUM_RTOL: f64 = 1e-3;

/// One standard model: its `[parameters]`/`[individual_parameters]` bodies and
/// the disposition-role mapping shared by the analytical and `ode_template`
/// forms (`cl=CL, v=V, ka=KA, ...` — exactly the `pk NAME(...)` argument list).
struct Model {
    name: &'static str,
    thetas: &'static str,
    indiv: &'static str,
    /// Disposition role→variable mapping, e.g. `"cl=CL, v=V, ka=KA"`.
    roles: &'static str,
    is_oral: bool,
}

impl Model {
    fn obs_cmt(&self) -> usize {
        if self.is_oral {
            2
        } else {
            1
        }
    }
}

fn models() -> Vec<Model> {
    vec![
        Model {
            name: "one_cpt_iv",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV(20.0, 1.0, 500.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n",
            roles: "cl=CL, v=V",
            is_oral: false,
        },
        Model {
            name: "one_cpt_oral",
            thetas: "  theta TVCL(0.13, 0.001, 10.0)\n  theta TVV(8.0, 0.1, 500.0)\n  theta TVKA(1.2, 0.01, 50.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  KA = TVKA\n",
            roles: "cl=CL, v=V, ka=KA",
            is_oral: true,
        },
        Model {
            name: "two_cpt_iv",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  theta TVQ(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q  = TVQ\n  V2 = TVV2\n",
            roles: "cl=CL, v1=V1, q=Q, v2=V2",
            is_oral: false,
        },
        Model {
            name: "two_cpt_oral",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  theta TVQ(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n  theta TVKA(1.1, 0.01, 50.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q  = TVQ\n  V2 = TVV2\n  KA = TVKA\n",
            roles: "cl=CL, v1=V1, q=Q, v2=V2, ka=KA",
            is_oral: true,
        },
        Model {
            name: "three_cpt_iv",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  theta TVQ2(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n  theta TVQ3(1.5, 0.01, 100.0)\n  theta TVV3(60.0, 1.0, 999.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q2 = TVQ2\n  V2 = TVV2\n  Q3 = TVQ3\n  V3 = TVV3\n",
            roles: "cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3",
            is_oral: false,
        },
        Model {
            name: "three_cpt_oral",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  theta TVQ2(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n  theta TVQ3(1.5, 0.01, 100.0)\n  theta TVV3(60.0, 1.0, 999.0)\n  theta TVKA(1.1, 0.01, 50.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q2 = TVQ2\n  V2 = TVV2\n  Q3 = TVQ3\n  V3 = TVV3\n  KA = TVKA\n",
            roles: "cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA",
            is_oral: true,
        },
    ]
}

/// Build the analytical and `ode_template` source for a model, optionally adding
/// lag time / bioavailability. Lag/F enter the analytical form via the `pk(...)`
/// mapping; the `ode_template` form declares them only as individual parameters
/// (the engine applies them at the dose, never inside the generated RHS).
fn build_pair(m: &Model, lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(m.thetas);
    let mut indiv = String::from(m.indiv);
    let mut pk_extra = String::new();
    if lag {
        thetas.push_str("  theta TVLAG(0.5, 0.0, 5.0)\n");
        indiv.push_str("  LAGTIME = TVLAG\n");
        pk_extra.push_str(", lagtime=LAGTIME");
    }
    if fbio {
        thetas.push_str("  theta TVF(0.7, 0.01, 1.0)\n");
        indiv.push_str("  F = TVF\n");
        pk_extra.push_str(", f=F");
    }

    let header = format!(
        "[parameters]\n{thetas}  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\n\
         [individual_parameters]\n{indiv}\n"
    );

    let analytical = format!(
        "{header}[structural_model]\n  pk {name}({roles}{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n",
        name = m.name,
        roles = m.roles,
    );
    // ode_template takes only disposition roles; lag/F are individual params.
    let template = format!(
        "{header}[structural_model]\n  ode_template {name}({roles})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n",
        name = m.name,
        roles = m.roles,
    );
    (analytical, template)
}

fn population(doses: Vec<DoseEvent>, obs_times: Vec<f64>, obs_cmt: usize) -> Population {
    let n = obs_times.len();
    let s: Subject = common::subject("1", doses, obs_times, vec![0.0; n], vec![obs_cmt; n]);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![s],
    }
}

fn assert_equiv(
    label: &str,
    analytical_src: &str,
    template_src: &str,
    pop: &Population,
    rtol: f64,
) {
    let an = parse_full_model(analytical_src)
        .unwrap_or_else(|e| panic!("[{label}] analytical model did not parse: {e}"))
        .model;
    let tm = parse_full_model(template_src)
        .unwrap_or_else(|e| panic!("[{label}] ode_template model did not parse: {e}"))
        .model;

    let pa = predict(&an, pop, &an.default_params);
    let pt = predict(&tm, pop, &tm.default_params);
    assert_eq!(pa.len(), pt.len(), "[{label}] prediction count mismatch");
    assert!(!pa.is_empty(), "[{label}] produced no predictions");

    for (x, y) in pa.iter().zip(pt.iter()) {
        let tol = ATOL + rtol * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "[{label}] t={:.3}: analytical PRED {:.6} vs ode_template PRED {:.6} \
             (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

/// `ode_template NAME(...)` predicts identically to `pk NAME(...)` across every
/// dosing mode, for all six standard models.
#[test]
fn ode_template_matches_analytical_all_models() {
    let obs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    for m in models() {
        let oc = m.obs_cmt();
        let (an, tm) = build_pair(&m, false, false);

        // single bolus
        assert_equiv(
            &format!("{}/bolus", m.name),
            &an,
            &tm,
            &population(
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs.clone(),
                oc,
            ),
            RTOL,
        );

        // multiple doses (q24h x3)
        assert_equiv(
            &format!("{}/multidose", m.name),
            &an,
            &tm,
            &population(
                vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(48.0, 100.0, 1, 0.0, false, 0.0),
                ],
                vec![1.0, 6.0, 12.0, 25.0, 30.0, 49.0, 54.0, 72.0],
                oc,
            ),
            ACCUM_RTOL,
        );

        // steady state (II = 24)
        assert_equiv(
            &format!("{}/steady_state", m.name),
            &an,
            &tm,
            &population(
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
                vec![1.0, 4.0, 8.0, 12.0, 23.0],
                oc,
            ),
            ACCUM_RTOL,
        );

        // infusion (IV only)
        if !m.is_oral {
            assert_equiv(
                &format!("{}/infusion", m.name),
                &an,
                &tm,
                &population(
                    vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)],
                    obs.clone(),
                    oc,
                ),
                RTOL,
            );
        }

        // lag time
        let (an_l, tm_l) = build_pair(&m, true, false);
        assert_equiv(
            &format!("{}/lagtime", m.name),
            &an_l,
            &tm_l,
            &population(
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                vec![0.75, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
                oc,
            ),
            RTOL,
        );

        // bioavailability (oral only)
        if m.is_oral {
            let (an_f, tm_f) = build_pair(&m, false, true);
            assert_equiv(
                &format!("{}/bioavailability", m.name),
                &an_f,
                &tm_f,
                &population(
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    obs.clone(),
                    oc,
                ),
                RTOL,
            );
        }
    }
}
