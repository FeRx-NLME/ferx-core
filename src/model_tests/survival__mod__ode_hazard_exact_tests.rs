// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::sens::num::PkNum;
use ferx_model::stats::likelihood::SUBJECT_SENTINEL_NLL;
use ferx_model::survival::parametric::cum_hazard;
use ferx_model::survival::parametric::hazard_and_cum_hazard;
use ferx_model::survival::parametric::hazard_and_cum_hazard_g;
use ferx_model::survival::parametric::mean_survival;
use ferx_model::survival::parametric::median_survival;
use ferx_model::survival::parametric::sample_conditional_event_time;
use ferx_model::survival::parametric::sample_event_time;
use ferx_model::types::CompiledModel;
use ferx_model::types::EventType;
use ferx_model::types::HazardFamily;
use ferx_model::types::ObsRecord;
use ferx_model::types::RtteClock;
use ferx_model::types::SimOutcome;
use ferx_model::types::Subject;
use ferx_model::types::TteRecurrence;
use nalgebra::DMatrix;
use rand::RngExt;
use std::collections::HashMap;

use crate::types::{EndpointLikelihood, HazardSpec};
use ferx_model::survival::*;

/// The joint PK-TTE (drug-driven, ODE-accumulated) hazard's η-Hessian is exact: the dual
/// walk's curves through the shared scoring, against the stencil it replaces, to the
/// stencil's accuracy; and its value against the production `tte_endpoint_nll`, to the
/// two engines' solver agreement.
#[test]
fn the_ode_hazard_eta_hessian_is_exact() {
    let root = env!("CARGO_MANIFEST_DIR");
    let prep = crate::api::prepare_run(
        &format!("{root}/examples/pktte_joint.ferx"),
        Some(&format!("{root}/data/pktte_joint.csv")),
    )
    .unwrap();
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta;
    let (cmt, chz) = model
        .endpoints
        .iter()
        .find_map(|(c, ep)| match ep {
            EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { chz_state }, ..
            } => Some((*c, *chz_state)),
            _ => None,
        })
        .expect("an ODE-accumulated endpoint");
    assert!(tte_endpoint_served(model, cmt));
    let (mut worst_h, mut worst_v, mut n) = (0.0f64, 0.0f64, 0);
    for s in prep.population.subjects.iter().take(5) {
        let recs: Vec<ObsRecord> = s
            .obs_records
            .iter()
            .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if *c == cmt))
            .cloned()
            .collect();
        if recs.is_empty() {
            continue;
        }
        let eta = [0.12];
        let term = TteTerm::new(model, s);
        let kind = crate::sens::ode_provider::adjoint::terms::TermKind::Tte;
        let (_, exact) = term_eta_hessian(model, s, kind, &term, theta, &eta).expect("in scope");
        let f = |e: &[f64]| {
            crate::stats::likelihood::tte_endpoint_nll_for_tests(model, s, cmt, &recs, theta, e)
        };
        // Richardson of the exact η-gradient (`O(h⁴)`), not a stencil of the value.
        let g = |e: f64| tte_subject_eta_grad(model, s, theta, &[e]).unwrap().1[0];
        let d = |h: f64| (g(eta[0] + h) - g(eta[0] - h)) / (2.0 * h);
        let fd = (4.0 * d(5e-4) - d(1e-3)) / 3.0;
        worst_h = worst_h.max((exact[(0, 0)] - fd).abs() / fd.abs().max(1e-12));
        let v = ode_tte_nll_g::<f64>(model, s, chz, &recs, theta, &eta).unwrap();
        worst_v = worst_v.max((v - f(&eta)).abs() / (1.0 + f(&eta).abs()));
        n += 1;
    }
    assert!(n > 2, "too few subjects ({n})");
    eprintln!("ODE hazard: Hessian vs Richardson {worst_h:.2e}, value vs production {worst_v:.2e}");
    // Measured: the value bit-identical to production's (0.0), the Hessian 3.65e-6 off
    // Richardson of the exact gradient — the frozen-step derivative against the adaptive
    // solve's slope at default tolerances. 2e-5 is 5× that; a dropped `h` term or CHZ
    // readout misses by O(1).
    assert!(worst_v < 1e-10, "value {worst_v:.2e}");
    assert!(worst_h < 2e-5, "Hessian {worst_h:.2e}");
}

/// A drug-driven hazard that also reads `TIME` (Gompertz-shaped, `H0·e^{GAM·TIME}·e^{BETA·C}`)
/// is exact too: the dual walk integrates the clock, and `h(t)` reads the value path's
/// anchors. The value matches production's non-Gaussian term; the η-gradient and the
/// θ-score (`GAM` among them, the clock's own channel) match Richardson differences of it.
#[test]
fn a_time_dependent_ode_hazard_is_exact() {
    let root = env!("CARGO_MANIFEST_DIR");
    // The anchor's model with a random effect on CL (the anchor itself has none), so the
    // hazard's drug channel carries an η.
    let src = std::fs::read_to_string(format!("{root}/nonmem_anchor/pktte_tdep_fit.ferx"))
        .unwrap()
        .replace("  sigma PROP_ERR", "  omega ETA_CL ~ 0.09\n  sigma PROP_ERR")
        .replace("CL   = TVCL\n", "CL   = TVCL * exp(ETA_CL)\n");
    let mut model = crate::parser::model_parser::parse_model_string(&src).unwrap();
    assert_eq!(model.n_eta, 1);
    if let Some(ode) = model.ode_spec.as_mut() {
        ode.solver_opts.reltol = 1e-12;
        ode.solver_opts.abstol = 1e-14;
    }
    let (population, _) = crate::api::read_population_for(
        &model,
        &None,
        &format!("{root}/nonmem_anchor/pktte_tdep.csv"),
        None,
        None,
        None,
        &[],
    )
    .unwrap();
    let model = &model;
    let theta = model.default_params.theta.clone();
    let cmt = *model
        .endpoints
        .iter()
        .find(|(_, ep)| matches!(ep, EndpointLikelihood::Tte { .. }))
        .unwrap()
        .0;
    assert!(tte_endpoint_served(model, cmt), "a TIME-reading hazard is served");
    let rich = |f: &dyn Fn(f64) -> f64, x: f64| {
        let d = |h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
        // Relative to `x` itself: a small parameter (a baseline hazard of 0.02) under an
        // absolute step is the oracle's own error, not the derivative's.
        let h = 1e-3 * x.abs().max(1e-2);
        (4.0 * d(h / 2.0) - d(h)) / 3.0
    };
    let nll = |s: &Subject, th: &[f64], e: &[f64]| {
        let mut acc = 0.0;
        crate::stats::likelihood::accumulate_non_gaussian_nll(model, s, th, e, None, 1.0, &mut acc);
        acc
    };
    let (mut worst_v, mut worst_g, mut n) = (0.0f64, 0.0f64, 0);
    for s in population.subjects.iter().filter(|s| !s.obs_records.is_empty()).take(4) {
        assert!(tte_record_served(model, s, cmt), "subject {} is served", s.id);
        let eta: Vec<f64> = (0..model.n_eta).map(|k| 0.1 - 0.05 * k as f64).collect();
        let (v, g) = tte_subject_eta_grad(model, s, &theta, &eta).expect("exact");
        let v_ref = nll(s, &theta, &eta);
        worst_v = worst_v.max((v - v_ref).abs() / (1.0 + v_ref.abs()));
        for k in 0..eta.len() {
            let fd = rich(
                &|x| {
                    let mut e = eta.clone();
                    e[k] = x;
                    nll(s, &theta, &e)
                },
                eta[k],
            );
            worst_g = worst_g.max((g[k] - fd).abs() / (1.0 + fd.abs()));
        }
        let (_, gt) =
            crate::sens::ode_provider::adjoint::non_gaussian_theta_score(model, s, &theta, &eta)
                .expect("exact");
        for i in 0..theta.len() {
            let fd = rich(
                &|x| {
                    let mut th = theta.clone();
                    th[i] = x;
                    nll(s, &th, &eta)
                },
                theta[i],
            );
            worst_g = worst_g.max((gt[i] - fd).abs() / (1.0 + fd.abs()));
        }
        n += 1;
    }
    assert!(n >= 3, "too few subjects ({n})");
    eprintln!("TIME-reading ODE hazard: value {worst_v:.2e}, gradients {worst_g:.2e}");
    assert!(worst_v < 1e-8, "value {worst_v:.2e}");
    assert!(worst_g < 1e-6, "gradient {worst_g:.2e}");
}

/// #1575's fixture (`tests/one_snapshot_dose_reads_survival.rs`) on the exact route: a
/// `TIME`-reading `$PK` whose `TIME = 0` disposition snapshot differs from every dose
/// record (`MTT = −9`, `ALAG1 = 10.5` there, `1` and `0.5` at the records), a transit
/// kernel and a compartment lag read per dose, a `TAD`-reading hazard, doses at 1 and 13
/// (the second on residual drug), η on `CL`, the lag and the kernel, and a `CL` the
/// disposition reads at `TIME = 0` (1.5×) but no dose record does; and a steady-state
/// twin (`SS=1`, the plain drug-driven hazard). Both take the event-driven dual walk with
/// every segment on the disposition (`ode_states_at_g`), which the static walk declined.
///
/// Value: the walk's `(H, h)` against production's dense single-pass solve
/// (`ode_cumhaz_hazard`, a different engine) — dies if a dose quantity is read at the
/// disposition (the kernel then delivers nothing, the lag moves every arrival), if a
/// segment is integrated on a dose record's snapshot instead of the disposition (its `CL`),
/// or if the `TAD` anchor drops the lag. Gradients: the exact η-gradient and θ-score
/// against Richardson differences of the production objective.
#[test]
fn a_time_reading_pk_with_an_ode_hazard_is_exact() {
    const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKA(1.0, FIX)
  theta TVMTT(1.0, FIX)
  theta TVN(3.0, FIX)
  theta TVLAG(0.5, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.09
  omega ETA_MTT ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL    = TVCL * exp(ETA_CL) * (if (TIME < 0.5) 1.5 else 1.0)
  V     = TVV
  KA    = TVKA
  MTT   = TVMTT * exp(ETA_MTT) - 10.0 * (if (TIME < 0.5) 1.0 else 0.0)
  NTR   = TVN
  ALAG1 = TVLAG * exp(ETA_LAG) + 10.0 * (if (TIME < 0.5) 1.0 else 0.0)
  H0    = TVH0
  BETA  = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  obs_scale = V
[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V)) + 0.002 * TAD
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
";
    use std::io::Write;
    // The steady-state twin: #1575's lag and transit kernel are rejected under `SS=1`
    // (`E_ABSORPTION_SS_LAG`), so it is the plain drug-driven hazard.
    let ss_model = MODEL
        .replace("  omega ETA_LAG ~ 0.09\n  omega ETA_MTT ~ 0.09\n", "")
        .replace("transit(n=NTR, mtt=MTT) - KA * depot", "-KA * depot")
        .replace("  MTT   = TVMTT * exp(ETA_MTT) - 10.0 * (if (TIME < 0.5) 1.0 else 0.0)\n", "")
        .replace("  ALAG1 = TVLAG * exp(ETA_LAG) + 10.0 * (if (TIME < 0.5) 1.0 else 0.0)\n", "")
        .replace(" + 0.002 * TAD", "")
        .replace(" * (if (TIME < 0.5) 1.5 else 1.0)", "");
    let (mut worst_v, mut worst_g) = (0.0f64, 0.0f64);
    for (src, ss) in [(MODEL, "0,0"), (ss_model.as_str(), "1,12")] {
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let (cmt, chz) = model
            .endpoints
            .iter()
            .find_map(|(c, ep)| match ep {
                EndpointLikelihood::Tte {
                    hazard: HazardSpec::OdeAccumulated { chz_state },
                    ..
                } => Some((*c, *chz_state)),
                _ => None,
            })
            .unwrap();
        let theta = model.default_params.theta.clone();
        let eta = [0.1, -0.2, 0.15][..model.n_eta].to_vec();
        let rich = |f: &dyn Fn(f64) -> f64, x: f64| {
            let d = |h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
            let h = 1e-3 * x.abs().max(1e-1);
            (4.0 * d(h / 2.0) - d(h)) / 3.0
        };
        let nll = |s: &Subject, th: &[f64], e: &[f64]| {
            let mut acc = 0.0;
            crate::stats::likelihood::accumulate_non_gaussian_nll(
                &model, s, th, e, None, 1.0, &mut acc,
            );
            acc
        };
        {
            let data = format!(
                "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
             1,1,.,1,100,1,0,1,{ss}\n\
             1,8,5,0,.,2,0,0,0,0\n\
             1,13,.,1,100,1,0,1,0,0\n\
             1,20,1,0,0,3,0,0,0,0\n"
            );
            let mut f = tempfile::NamedTempFile::new().unwrap();
            f.write_all(data.as_bytes()).unwrap();
            let (pop, _) = crate::api::read_population_for(
                &model,
                &None,
                f.path().to_str().unwrap(),
                None,
                None,
                None,
                &[],
            )
            .unwrap();
            let s = &pop.subjects[0];
            assert!(tte_record_served(&model, s, cmt), "SS={ss}: served exactly");
            let times = [20.0];
            let walk = crate::sens::ode_provider::adjoint::ode_states_at_g::<f64>(
                &model, s, &theta, &eta, &times,
            )
            .expect("the event walk serves it");
            let (cum, haz) = crate::sens::ode_provider::adjoint::cumhaz_hazard_from_states(
                &model, s, chz, &times, &walk,
            )
            .unwrap();
            let (cum_d, haz_d) = ode_cumhaz_hazard(&model, s, chz, &theta, &eta, &times);
            assert!(
                cum[0].is_finite()
                    && haz[0].is_finite()
                    && cum_d[0].is_finite()
                    && haz_d[0].is_finite()
            );
            worst_v = worst_v // nan-ok: all four values asserted finite above; the divisors are the dense cumulative hazard and hazard
                .max((cum[0] - cum_d[0]).abs() / cum_d[0].abs())
                .max((haz[0] - haz_d[0]).abs() / haz_d[0].abs());
            let (_, g) = tte_subject_eta_grad(&model, s, &theta, &eta).expect("exact");
            for k in 0..eta.len() {
                let fd = rich(
                    &|x| {
                        let mut e = eta.clone();
                        e[k] = x;
                        nll(s, &theta, &e)
                    },
                    eta[k],
                );
                assert!(fd.is_finite() && g[k].is_finite());
                // Every η is live: the lag and the kernel move the hazard at 20.
                assert!(fd.abs() > 1e-4, "SS={ss}: η[{k}] is not live ({fd:.2e})");
                worst_g = worst_g.max((g[k] - fd).abs() / (1.0 + fd.abs()));
            }
            let (_, gt) = crate::sens::ode_provider::adjoint::non_gaussian_theta_score(
                &model, s, &theta, &eta,
            )
            .expect("exact");
            for i in 0..theta.len() {
                let fd = rich(
                    &|x| {
                        let mut th = theta.clone();
                        th[i] = x;
                        nll(s, &th, &eta)
                    },
                    theta[i],
                );
                assert!(fd.is_finite() && gt[i].is_finite());
                worst_g = worst_g.max((gt[i] - fd).abs() / (1.0 + fd.abs()));
            }
        }
    }
    eprintln!("TIME-reading $PK ODE hazard: value vs dense {worst_v:.2e}, gradients {worst_g:.2e}");
    // Measured on Linux: value 2.0e-15 against the dense engine (the same arithmetic on
    // another walk), gradients 2.2e-10 (Richardson's own error at these tolerances).
    assert!(worst_v < 1e-12, "value {worst_v:.2e}");
    assert!(worst_g < 5e-9, "gradient {worst_g:.2e}");
}
