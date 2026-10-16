//! The exact linear-ODE path (#852) against the analytical closed forms.
//!
//! An `[odes]` block that the parser certifies as linear time-invariant is propagated by the
//! matrix exponential instead of integrated (`ode_method = auto` resolves to `linear`). The
//! `ode_template NAME(...)` form lowers each standard model to exactly such a block, so
//! `ode_template` vs `pk NAME(...)` is a comparison between two engines that share **no**
//! numerical code — Padé scaling-and-squaring on one side, eigenvalue closed forms with their
//! own L'Hôpital/Vieta branches on the other — over the whole dosing layer: multiple doses
//! landing on residual drug, steady state (bolus and infusion), infusions, lag time and
//! bioavailability.
//!
//! The tolerance is the point. `ode_template_equivalence.rs` runs the same pairs at 1e-4/1e-3
//! because an integrator only reproduces a closed form to `ode_reltol`; here the bound is
//! measured at roundoff, several orders below anything RK45 can reach at the default
//! tolerance. So the tight bound doubles as the witness that the exact path actually ran: the
//! `rk45` twin below — the same model with the integrator named, which is how a user opts out —
//! fails it by orders of magnitude.
//!
//! Tier 2: `predict()` only, no fit, returns in milliseconds; not gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population};

mod common;

struct Model {
    name: &'static str,
    thetas: &'static str,
    indiv: &'static str,
    roles: &'static str,
    is_oral: bool,
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
            thetas: "  theta TVCL(0.9, 0.001, 10.0)\n  theta TVV(8.0, 0.1, 500.0)\n  theta TVKA(1.2, 0.01, 50.0)\n",
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

/// `(analytical, ode_template)` sources, optionally with lag / bioavailability and a
/// `[fit_options]` tail on the `ode_template` side.
fn build_pair(m: &Model, lag: bool, fbio: bool, template_tail: &str) -> (String, String) {
    let mut thetas = String::from(m.thetas);
    let mut indiv = String::from(m.indiv);
    let mut pk_extra = String::new();
    if lag {
        thetas.push_str("  theta TVLAG(0.6, 0.0, 5.0)\n");
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
    let err = "[error_model]\n  DV ~ proportional(PROP)\n";
    let analytical = format!(
        "{header}[structural_model]\n  pk {name}({roles}{pk_extra})\n\n{err}",
        name = m.name,
        roles = m.roles,
    );
    let template = format!(
        "{header}[structural_model]\n  ode_template {name}({roles})\n\n{err}{template_tail}",
        name = m.name,
        roles = m.roles,
    );
    (analytical, template)
}

fn population(doses: Vec<DoseEvent>, obs_times: Vec<f64>, obs_cmt: usize) -> Population {
    let n = obs_times.len();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            doses,
            obs_times,
            vec![0.0; n],
            vec![obs_cmt; n],
        )],
    }
}

/// One dosing scenario: a label, the subject's doses and observation times.
struct Scenario {
    label: &'static str,
    doses: Vec<DoseEvent>,
    obs: Vec<f64>,
}

/// Every scenario keeps the quantity under test live on both sides of each dose: later doses
/// land on residual drug, and steady state is sampled within and past the first interval.
fn scenarios(m: &Model) -> Vec<(Scenario, bool, bool)> {
    let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
    let mut v = vec![
        (
            Scenario {
                label: "bolus",
                doses: vec![bolus(0.0)],
                obs: vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            },
            false,
            false,
        ),
        (
            Scenario {
                label: "multidose_q12h",
                doses: vec![bolus(0.0), bolus(12.0), bolus(24.0)],
                obs: vec![1.0, 6.0, 11.9, 12.5, 18.0, 24.5, 30.0, 48.0],
            },
            false,
            false,
        ),
        (
            Scenario {
                label: "steady_state_bolus",
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
                obs: vec![0.5, 2.0, 6.0, 11.0, 18.0, 30.0],
            },
            false,
            false,
        ),
        (
            Scenario {
                label: "lag_multidose",
                doses: vec![bolus(0.0), bolus(12.0)],
                obs: vec![0.3, 0.75, 2.0, 12.3, 12.9, 20.0],
            },
            true,
            false,
        ),
    ];
    if m.is_oral {
        v.push((
            Scenario {
                label: "bioavailability_multidose",
                doses: vec![bolus(0.0), bolus(12.0)],
                obs: vec![0.5, 2.0, 12.5, 14.0, 24.0],
            },
            false,
            true,
        ));
    } else {
        v.push((
            Scenario {
                label: "infusion_multidose",
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0),
                    DoseEvent::new(12.0, 100.0, 1, 25.0, false, 0.0),
                ],
                obs: vec![1.0, 3.9, 4.0, 8.0, 13.0, 16.0, 30.0],
            },
            false,
            false,
        ));
        v.push((
            Scenario {
                label: "steady_state_infusion",
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 25.0, true, 12.0)],
                obs: vec![1.0, 4.0, 6.0, 11.5, 14.0],
            },
            false,
            false,
        ));
    }
    v
}

/// Worst relative difference between the two models' PRED over one scenario.
fn worst_rel(analytical: &str, template: &str, pop: &Population, label: &str) -> f64 {
    let an = parse_full_model(analytical)
        .unwrap_or_else(|e| panic!("[{label}] analytical did not parse: {e}"))
        .model;
    let tm = parse_full_model(template)
        .unwrap_or_else(|e| panic!("[{label}] ode_template did not parse: {e}"))
        .model;
    let pa = predict(&an, pop, &an.default_params).unwrap();
    let pt = predict(&tm, pop, &tm.default_params).unwrap();
    assert_eq!(pa.len(), pt.len(), "[{label}] prediction count");
    assert!(!pa.is_empty(), "[{label}] no predictions");
    let scale = pa.iter().fold(0.0_f64, |s, p| s.max(p.pred.abs()));
    assert!(scale > 0.0, "[{label}] all-zero analytical predictions");
    pa.iter().zip(&pt).fold(0.0_f64, |worst, (a, t)| {
        // `f64::max` discards NaN, so a non-finite prediction must fail here, not vanish.
        assert!(
            a.pred.is_finite() && t.pred.is_finite(),
            "[{label}] non-finite PRED at t={}: analytical {} vs ode_template {}",
            a.time,
            a.pred,
            t.pred
        );
        // Relative to the scenario's peak, so a trough near zero is not judged on its own
        // relative digits (the closed forms themselves cancel there).
        worst.max((a.pred - t.pred).abs() / scale)
    })
}

/// Measured worst over every model × scenario below: 1.2e-14 (three-cpt steady state). The
/// bound leaves three orders of headroom for platform libm differences and is still six orders
/// below the 1.0e-5 the integrator reaches at the default `ode_reltol` on the two-cpt oral
/// twin (`opting_out_integrates_again`).
const EXACT_TOL: f64 = 1e-11;

#[test]
fn linear_odes_match_the_closed_forms_to_roundoff() {
    for m in models() {
        let obs_cmt = if m.is_oral { 2 } else { 1 };
        for (sc, lag, fbio) in scenarios(&m) {
            let label = format!("{}/{}", m.name, sc.label);
            let (an, tm) = build_pair(&m, lag, fbio, "");
            let pop = population(sc.doses, sc.obs, obs_cmt);
            let worst = worst_rel(&an, &tm, &pop, &label);
            assert!(
                worst < EXACT_TOL,
                "[{label}] exact linear path vs closed form: worst {worst:e} ≥ {EXACT_TOL:e}"
            );
        }
    }
}

/// Naming an integrator opts a linear model out of the exact path — and the gap that opens
/// is what makes `EXACT_TOL` a witness for the route rather than a coincidence.
#[test]
fn opting_out_integrates_again() {
    let m = &models()[3]; // two_cpt_oral
    let (an, tm) = build_pair(m, false, false, "\n[fit_options]\n  ode_method = rk45\n");
    let pop = population(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        vec![0.5, 2.0, 6.0, 12.5, 18.0, 30.0],
        2,
    );
    let worst = worst_rel(&an, &tm, &pop, "two_cpt_oral/rk45");
    // Measured 1.0e-5 at the default ode_reltol = 1e-4: an integrator error, not roundoff.
    assert!(
        worst > 1e3 * EXACT_TOL,
        "the rk45 twin should carry integrator error well above EXACT_TOL, got {worst:e} — \
         is `ode_method = rk45` still reaching the solver?"
    );
}
