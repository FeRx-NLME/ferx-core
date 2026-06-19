//! Integration tests for issue #205: compartment state vector in DerivedContext.
//!
//! Tier 2: fast-returning tests (max 2 outer iterations or immediate parse/check).
//! Tier 3 slow tests are gated with `#[cfg_attr(not(feature = "slow-tests"), ignore)]`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::read_nonmem_csv;
use ferx_core::types::{DoseEvent, Population};
use ferx_core::{fit, FitOptions};
use std::collections::HashMap;
use std::path::Path;

mod common;

fn simple_iv_population() -> Population {
    let obs_times = vec![1.0, 4.0, 12.0, 24.0];
    let n = obs_times.len();
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            vec![5.0, 3.0, 1.5, 0.7],
            vec![1; n],
        )],
    }
}

/// 1-cpt IV model: `compartments[0]` (central concentration) must equal IPRED
/// for every observation (no scaling applied).
#[test]
fn analytical_1cpt_iv_compartments_equals_ipred() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_central = compartments[0]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let c_central = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_central")
            .expect("C_central column must exist");
        assert_eq!(
            c_central.1.len(),
            sr.ipred.len(),
            "C_central must have same length as IPRED"
        );
        for (j, (&cmpt, &ip)) in c_central.1.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (cmpt - ip).abs() < 1e-10,
                "subject {}: compartments[0] at obs {j}: {cmpt:.6} != ipred {ip:.6}",
                sr.id
            );
        }
    }
}

/// 1-cpt IV model: named compartment access `central` must equal IPRED.
#[test]
fn analytical_1cpt_iv_named_access_equals_ipred() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_named = central

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_named")
            .expect("C_named column must exist");
        for (j, (&v, &ip)) in col.1.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (v - ip).abs() < 1e-10,
                "subject {}: central at obs {j}: {v:.6} != ipred {ip:.6}",
                sr.id
            );
        }
    }
}

/// 2-cpt IV model: `compartments[1]` (peripheral concentration) must be
/// finite and positive at every observation time.
#[test]
fn analytical_2cpt_iv_peripheral_is_finite_and_positive() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q(1.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_periph = compartments[1]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_periph")
            .expect("C_periph column must exist");
        for (j, &v) in col.1.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 0.0,
                "subject {}: C_periph at obs {j} should be finite/non-negative, got {v}",
                sr.id
            );
        }
    }
}

/// 1-cpt oral model: `compartments[0]` is depot amount (>= 0),
/// `compartments[1]` is central concentration (= IPRED for no-scaling model).
#[test]
fn analytical_1cpt_oral_depot_and_central() {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let n = obs_times.len();
    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            vec![1.0, 2.0, 3.5, 4.0, 3.0, 2.0, 0.5],
            vec![1; n],
        )],
    };

    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V
  KA = KA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot   = compartments[0]
  C_central = compartments[1]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let depot_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "A_depot")
            .expect("A_depot must exist");
        let central_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_central")
            .expect("C_central must exist");
        for (j, &d) in depot_col.1.iter().enumerate() {
            assert!(
                d >= 0.0,
                "depot amount at obs {j} must be non-negative, got {d}"
            );
        }
        for (j, (&c, &ip)) in central_col.1.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "C_central at obs {j}: {c:.6} != ipred {ip:.6}"
            );
        }
    }
}

/// Parse-time check: `compartments[0]` in an integral sets `uses_compartments=true`.
/// This is a fast Tier 2 test — no fit required.
#[test]
fn parse_integral_compartment_subscript_sets_uses_compartments() {
    const MODEL: &str = "
[parameters]
  theta CL(1.0, 0, 100)
  theta V(10.0, 0, 1000)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  AUC_C0 = integral(compartments[0], from=0, to=24)
  AUC_IP = integral(IPRED, from=0, to=24)
";
    let result = parse_model_string(MODEL).expect("model must parse");
    let derived = &result.derived_exprs;

    let auc_c0 = derived
        .iter()
        .find(|s| s.name == "AUC_C0")
        .expect("AUC_C0 must be present");
    if let ferx_core::types::DerivedKind::Integral {
        uses_compartments, ..
    } = &auc_c0.kind
    {
        assert!(uses_compartments, "AUC_C0 must have uses_compartments=true");
    } else {
        panic!("AUC_C0 must be Integral kind");
    }

    let auc_ip = derived
        .iter()
        .find(|s| s.name == "AUC_IP")
        .expect("AUC_IP must be present");
    if let ferx_core::types::DerivedKind::Integral {
        uses_compartments, ..
    } = &auc_ip.kind
    {
        assert!(
            !uses_compartments,
            "AUC_IP must have uses_compartments=false"
        );
    } else {
        panic!("AUC_IP must be Integral kind");
    }
}

// ── ODE model tests ───────────────────────────────────────────────────────────
//
// These cover code paths that the analytical tests above cannot reach:
//   • `ode_predictions_with_states`   (src/ode/predictions.rs)
//   • `ode_dense_solve_states`        (src/ode/predictions.rs)
//   • `compute_predictions_with_states` ODE branch (src/pk/mod.rs)
//   • `compute_extra_output_columns`  with `uses_compartments=true` (src/api.rs)
//
// Model: 2-cpt IV + effect compartment (3 ODE states).
//   State 0: central    (mg, amount)
//   State 1: peripheral (mg, amount)
//   State 2: effect     (mg/L, concentration, as written in the ODE)
// Dose lands in CMT=1 → central. Observations are from central (scaled by V1).

const ODE_2CPT_EFFECT: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q(1.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta KE0(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL  = CL * exp(ETA_CL)
  V1  = V1
  Q   = Q
  V2  = V2
  KE0 = KE0

[structural_model]
  ode(states=[central, peripheral, effect])

[odes]
  d/dt(central)    = -(CL/V1 + Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) = (Q/V1) * central - (Q/V2) * peripheral
  d/dt(effect)     = KE0 * (central/V1 - effect)

[scaling]
  y = central / V1

[error_model]
  DV ~ proportional(PROP)
";

/// ODE 2-cpt + effect compartment: `compartments[2]` (subscript) and named
/// state `effect` must yield identical, finite, positive values at every
/// observation. Exercises `ode_predictions_with_states` end-to-end.
#[test]
fn ode_2cpt_effect_compartment_derived() {
    let model_str = format!(
        "{ODE_2CPT_EFFECT}
[derived]
  Ce       = compartments[2]
  Ce_named = effect

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
"
    );
    let model = parse_model_string(&model_str).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let ce_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "Ce")
            .expect("Ce column must exist");
        let ce_named_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "Ce_named")
            .expect("Ce_named column must exist");

        assert_eq!(
            ce_col.1.len(),
            sr.ipred.len(),
            "Ce must have same length as IPRED"
        );

        for (j, (&by_idx, &by_name)) in ce_col.1.iter().zip(ce_named_col.1.iter()).enumerate() {
            assert!(
                by_idx.is_finite() && by_idx > 0.0,
                "subject {}: Ce (compartments[2]) at obs {j} must be finite+positive, \
                 got {by_idx}",
                sr.id
            );
            assert!(
                (by_idx - by_name).abs() < 1e-10,
                "subject {}: subscript compartments[2]={by_idx:.8} != named \
                 effect={by_name:.8} at obs {j}",
                sr.id
            );
        }
    }
}

/// ODE effect compartment: `integral(compartments[2], ...)` and
/// `integral(effect, ...)` both trigger the `uses_compartments=true`
/// grid-integral re-solve path. Both AUCs must be finite, positive, and
/// identical. Exercises `ode_dense_solve_states` via the grid-integral path.
#[test]
fn ode_integral_over_compartment() {
    let model_str = format!(
        "{ODE_2CPT_EFFECT}
[derived]
  AUC_Ce    = integral(compartments[2], from=0, to=24, step=1.0)
  AUC_named = integral(effect, from=0, to=24, step=1.0)

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
"
    );
    let model = parse_model_string(&model_str).expect("model must parse");

    // Parse-time check: both integrands reference a compartment state, so
    // uses_compartments must be true for each.
    for spec in &model.derived_exprs {
        if let ferx_core::types::DerivedKind::Integral {
            uses_compartments, ..
        } = &spec.kind
        {
            assert!(
                uses_compartments,
                "{} integral must have uses_compartments=true (references ODE state)",
                spec.name
            );
        }
    }

    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let auc_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_Ce")
            .expect("AUC_Ce column must exist");
        let auc_named_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_named")
            .expect("AUC_named column must exist");

        for (j, &auc) in auc_col.1.iter().enumerate() {
            assert!(
                auc.is_finite() && auc > 0.0,
                "subject {}: AUC_Ce at obs {j} must be finite+positive, got {auc}",
                sr.id
            );
        }
        // Named-state integral and subscript integral must agree.
        for (j, (&by_idx, &by_name)) in auc_col.1.iter().zip(auc_named_col.1.iter()).enumerate() {
            assert!(
                (by_idx - by_name).abs() < 1e-8,
                "subject {}: AUC_Ce={by_idx:.6} != AUC_named={by_name:.6} at obs {j}",
                sr.id
            );
        }
    }
}

// ── Analytical peripheral-state tests ──────────────────────────────────────────
//
// These cover every multi-compartment analytical model variant:
//   two_cpt_oral  → [depot, central, peripheral]
//   three_cpt_iv  → [central, peripheral1, peripheral2]
//   three_cpt_oral → [depot, central, peripheral1, peripheral2]
//
// Shared oral population (100 mg oral dose, 7 observation times):

fn simple_oral_population() -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let n = obs_times.len();
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            vec![1.0, 2.0, 3.5, 4.0, 3.0, 2.0, 0.5],
            vec![1; n],
        )],
    }
}

/// 2-cpt oral: compartments[0] = depot (amount >= 0), compartments[1] = central
/// (= IPRED), compartments[2] = peripheral (>= 0).  Named access `depot`,
/// `central`, `peripheral` must each match the corresponding subscript form.
#[test]
fn analytical_2cpt_oral_peripheral_states() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q(1.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2
  KA = KA

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot      = compartments[0]
  C_central    = compartments[1]
  C_periph     = compartments[2]
  depot_named  = depot
  cent_named   = central
  periph_named = peripheral

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_oral_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let depot = col!("A_depot");
        let central = col!("C_central");
        let periph = col!("C_periph");
        let depot_n = col!("depot_named");
        let cent_n = col!("cent_named");
        let periph_n = col!("periph_named");

        for (j, &d) in depot.iter().enumerate() {
            assert!(
                d >= 0.0,
                "subject {}: depot at obs {j} must be >= 0, got {d}",
                sr.id
            );
        }
        for (j, (&c, &ip)) in central.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "subject {}: compartments[1] = {c:.8} but ipred = {ip:.8} at obs {j}",
                sr.id
            );
        }
        for (j, &p) in periph.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        // Named access must equal subscript access.
        for (j, (&d0, &dn)) in depot.iter().zip(depot_n.iter()).enumerate() {
            assert!(
                (d0 - dn).abs() < 1e-12,
                "subject {}: compartments[0]={d0:.8} != depot={dn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&c0, &cn)) in central.iter().zip(cent_n.iter()).enumerate() {
            assert!(
                (c0 - cn).abs() < 1e-12,
                "subject {}: compartments[1]={c0:.8} != central={cn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph.iter().zip(periph_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[2]={p0:.8} != peripheral={pn:.8} at obs {j}",
                sr.id
            );
        }
    }
}

/// 3-cpt IV: compartments[0] = central (= IPRED), compartments[1] = peripheral1
/// (>= 0), compartments[2] = peripheral2 (>= 0).  Named access `central`,
/// `peripheral1`, `peripheral2` must each match the corresponding subscript form.
#[test]
fn analytical_3cpt_iv_peripheral_states() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.1, 100.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q2(2.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta Q3(1.5, 0.01, 50.0)
  theta V3(30.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.04 (sd)

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q2 = Q2
  V2 = V2
  Q3 = Q3
  V3 = V3

[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_central  = compartments[0]
  C_periph1  = compartments[1]
  C_periph2  = compartments[2]
  cent_named = central
  p1_named   = peripheral1
  p2_named   = peripheral2

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let central = col!("C_central");
        let periph1 = col!("C_periph1");
        let periph2 = col!("C_periph2");
        let cent_n = col!("cent_named");
        let p1_n = col!("p1_named");
        let p2_n = col!("p2_named");

        for (j, (&c, &ip)) in central.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "subject {}: compartments[0] = {c:.8} but ipred = {ip:.8} at obs {j}",
                sr.id
            );
        }
        for (j, &p) in periph1.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral1 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        for (j, &p) in periph2.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral2 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        // Named access must equal subscript access.
        for (j, (&c0, &cn)) in central.iter().zip(cent_n.iter()).enumerate() {
            assert!(
                (c0 - cn).abs() < 1e-12,
                "subject {}: compartments[0]={c0:.8} != central={cn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph1.iter().zip(p1_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[1]={p0:.8} != peripheral1={pn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph2.iter().zip(p2_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[2]={p0:.8} != peripheral2={pn:.8} at obs {j}",
                sr.id
            );
        }
    }
}

/// 3-cpt oral: compartments[0] = depot (amount >= 0), compartments[1] = central
/// (= IPRED), compartments[2] = peripheral1 (>= 0), compartments[3] = peripheral2
/// (>= 0).  Named access matches subscript for all four compartments.
#[test]
fn analytical_3cpt_oral_peripheral_states() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.1, 100.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q2(2.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta Q3(1.5, 0.01, 50.0)
  theta V3(30.0, 0.1, 1000.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.04 (sd)

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q2 = Q2
  V2 = V2
  Q3 = Q3
  V3 = V3
  KA = KA

[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot    = compartments[0]
  C_central  = compartments[1]
  C_periph1  = compartments[2]
  C_periph2  = compartments[3]
  depot_n    = depot
  cent_n     = central
  p1_n       = peripheral1
  p2_n       = peripheral2

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_oral_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let depot = col!("A_depot");
        let central = col!("C_central");
        let periph1 = col!("C_periph1");
        let periph2 = col!("C_periph2");
        let depot_n = col!("depot_n");
        let cent_n = col!("cent_n");
        let p1_n = col!("p1_n");
        let p2_n = col!("p2_n");

        for (j, &d) in depot.iter().enumerate() {
            assert!(
                d >= 0.0,
                "subject {}: depot at obs {j} must be >= 0, got {d}",
                sr.id
            );
        }
        for (j, (&c, &ip)) in central.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "subject {}: compartments[1] = {c:.8} but ipred = {ip:.8} at obs {j}",
                sr.id
            );
        }
        for (j, &p) in periph1.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral1 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        for (j, &p) in periph2.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral2 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        // Named access must equal subscript access.
        for (j, (&d0, &dn)) in depot.iter().zip(depot_n.iter()).enumerate() {
            assert!(
                (d0 - dn).abs() < 1e-12,
                "subject {}: compartments[0]={d0:.8} != depot={dn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&c0, &cn)) in central.iter().zip(cent_n.iter()).enumerate() {
            assert!(
                (c0 - cn).abs() < 1e-12,
                "subject {}: compartments[1]={c0:.8} != central={cn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph1.iter().zip(p1_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[2]={p0:.8} != peripheral1={pn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph2.iter().zip(p2_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[3]={p0:.8} != peripheral2={pn:.8} at obs {j}",
                sr.id
            );
        }
    }
}

// ── Tier 2: AUC_Ce analytical reference check ─────────────────────────────────
//
// Verifies that `ode_dense_solve_states` (the grid-integral code path, activated
// when an integrand references a compartment state) produces AUC_Ce values that
// agree with the closed-form integral to within the expected trapezoidal error.
//
// Model: 1-cpt IV ODE + effect compartment (2 ODE states):
//   d/dt(central) = -(CL/V) * central
//   d/dt(effect)  = KE0 * (central/V − effect)
//
// Closed-form for a single IV dose D, ke = CL/V, c0 = D/V:
//   Ce(t) = c0 · KE0/(KE0−ke) · (exp(−ke·t) − exp(−KE0·t))
//
//   AUC_Ce(0→T) = c0·KE0/(KE0−ke) · [(1−e^{−ke·T})/ke − (1−e^{−KE0·T})/KE0]
//
// The test extracts the individual-level CL_i, V_i, and KE0 from the [derived]
// block (post-EBE values), computes the analytical AUC_Ce at those parameters,
// and compares with the ferx grid-integral result (step=0.5 h).  Agreement to
// within 3% is expected (trapezoidal error at step=0.5 plus small EBE movement).

fn analytical_auc_ce_1cpt_iv(dose: f64, cl: f64, v: f64, ke0: f64, t_end: f64) -> f64 {
    let c0 = dose / v;
    let ke = cl / v;
    if (ke - ke0).abs() < 1e-6 {
        // L'Hôpital limit: AUC_Ce = c0·KE0·[t·e^{−ke·t}/ke + (e^{−ke·t}−1)/ke²] ... +∞ integrands
        // For the finite-window case: AUC_Ce ≈ AUC_C (degenerate; return AUC_C as upper bound)
        return c0 / ke * (1.0 - (-ke * t_end).exp());
    }
    let factor = c0 * ke0 / (ke0 - ke);
    let integral_ke = (1.0 - (-ke * t_end).exp()) / ke;
    let integral_ke0 = (1.0 - (-ke0 * t_end).exp()) / ke0;
    factor * (integral_ke - integral_ke0)
}

/// Tier 2 analytical reference: 1-cpt IV ODE + effect compartment.
///
/// Extracts post-fit individual CL and V from `[derived]`, computes the
/// closed-form AUC_Ce, and verifies the ferx grid integral (step=0.5) agrees
/// to within 3% trapezoidal error.
#[test]
fn ode_auc_ce_matches_analytical_reference() {
    // Tight KE0 bounds keep it near 0.5 regardless of what theta CL/V do.
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  theta KE0(0.5, 0.45, 0.55)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL  = CL * exp(ETA_CL)
  V   = V
  KE0 = KE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -(CL/V) * central
  d/dt(effect)  = KE0 * (central/V - effect)

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)

[derived]
  AUC_Ce  = integral(effect, from=0, to=24, step=0.5)
  CL_indv = CL
  V_indv  = V
  KE0_val = KE0

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population(); // dose=100 at t=0, obs at 1/4/12/24
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1[0]
            };
        }
        // Individual-level PK at last observation row (they're per-row constants, take any).
        let cl_i = col!("CL_indv");
        let v_i = col!("V_indv");
        let ke0 = col!("KE0_val");

        // Grid-integral AUC_Ce from ferx.
        let auc_ce_ferx = col!("AUC_Ce");

        // Closed-form reference at the same individual parameters.
        let auc_ce_ref = analytical_auc_ce_1cpt_iv(100.0, cl_i, v_i, ke0, 24.0);

        assert!(
            auc_ce_ferx.is_finite() && auc_ce_ferx > 0.0,
            "subject {}: AUC_Ce must be finite+positive, got {auc_ce_ferx}",
            sr.id
        );
        let rel_err = (auc_ce_ferx - auc_ce_ref).abs() / auc_ce_ref;
        assert!(
            rel_err < 0.03,
            "subject {}: ferx AUC_Ce={auc_ce_ferx:.4} vs analytical={auc_ce_ref:.4} \
             (relative error {:.2}% > 3%)",
            sr.id,
            rel_err * 100.0
        );
    }
}

// ── Tier 3 slow: full convergence + AUC_Ce analytical reference ───────────────
//
// Fits a 2-cpt IV ODE model extended with an effect compartment to the
// `data/two_cpt_iv.csv` dataset (18 subjects, single IV bolus, bi-exponential
// decline).  The effect compartment is not observed; it is estimated purely for
// its compartment-state AUC:
//
//   AUC_Ce = integral(effect, from=0, to=24, step=0.5)
//
// **Analytical reference** (population-average parameters near the generating
// values CL=4 L/h, V1=12 L, Q=2 L/h, V2=25 L, KE0=0.5 h⁻¹, dose=100 mg):
//
//   Eigenvalues:
//     α = (k₁₀+k₁₂+k₂₁ + √[(k₁₀+k₁₂+k₂₁)²−4·k₁₀·k₂₁]) / 2  ≈ 0.530 h⁻¹
//     β = (k₁₀+k₁₂+k₂₁ − √[(k₁₀+k₁₂+k₂₁)²−4·k₁₀·k₂₁]) / 2  ≈ 0.050 h⁻¹
//   (k₁₀ = CL/V₁ = 0.333, k₁₂ = Q/V₁ = 0.167, k₂₁ = Q/V₂ = 0.080)
//
//   AUC_Ce(0→24) ≈ 21.6 mg/L·h    (ferx grid step=0.5, trapezoidal error <1%)
//   AUC_central(0→24) ≈ 21.9 mg/L·h
//
// Note: NONMEM cannot directly evaluate an integral over an effect-compartment
// state without custom $DES/$PK code; the closed-form formula above is the
// appropriate analytical reference for this derived variable.
//
// The test asserts:
//   1. Fit converges (finite OFV).
//   2. Each subject's AUC_Ce and AUC_central are finite and positive.
//   3. AUC_Ce ≤ AUC_central × 1.05 — the effect compartment can briefly exceed
//      the central concentration, but the cumulative area under the curve should
//      remain close to the central AUC over [0, 24] for any KE0 > 0 (because at
//      t = 24 h all drug is nearly gone and ∫₀^∞ Ce = ∫₀^∞ C for linear PK).
//   4. AUC_Ce / AUC_central is in [0.70, 1.05] for each subject (the lower
//      bound corresponds to very slow KE0 ≈ 0.01 with the fast 2-cpt elimination;
//      the upper bound is 1.05 to accommodate brief Ce > C_central periods).

const TWO_CPT_IV_ODE_EFFECT_MODEL: &str = "
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  theta TVKE0(0.5, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V1  = TVV1 * exp(ETA_V1)
  Q   = TVQ
  V2  = TVV2
  KE0 = TVKE0

[structural_model]
  ode(states=[central, peripheral, effect])

[odes]
  d/dt(central)    = -(CL/V1 + Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) = (Q/V1) * central - (Q/V2) * peripheral
  d/dt(effect)     = KE0 * (central/V1 - effect)

[scaling]
  y = central / V1

[error_model]
  DV ~ proportional(PROP_ERR)

[derived]
  AUC_Ce      = integral(effect, from=0, to=24, step=0.5)
  AUC_central = integral(IPRED, from=0, to=24)

[fit_options]
  method   = focei
  maxiter  = 300
  gradient = fd
";

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn auc_ce_2cpt_iv_effect_convergence_and_analytical_ref() {
    let model = parse_model_string(TWO_CPT_IV_ODE_EFFECT_MODEL).expect("model must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_iv.csv"), None, None)
        .expect("two_cpt_iv data must load");

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("fit must not error");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );

    let mut auc_ce_vec: Vec<f64> = Vec::new();
    let mut auc_central_vec: Vec<f64> = Vec::new();

    for sr in &result.subjects {
        // Each row holds the same aggregate AUC value; take the first.
        let auc_ce = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_Ce")
            .expect("AUC_Ce column must exist")
            .1[0];
        let auc_central = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_central")
            .expect("AUC_central column must exist")
            .1[0];

        assert!(
            auc_ce.is_finite() && auc_ce > 0.0,
            "subject {}: AUC_Ce must be finite+positive, got {auc_ce}",
            sr.id
        );
        assert!(
            auc_central.is_finite() && auc_central > 0.0,
            "subject {}: AUC_central must be finite+positive, got {auc_central}",
            sr.id
        );

        // For linear PK: ∫₀^∞ Ce = ∫₀^∞ C_central.  By t=24 h, nearly all
        // drug is eliminated (exp(-β·24) ≈ 0.30 for β≈0.05), so AUC(0→24)
        // captures ≥ 88% of AUC(0→∞).  Consequently AUC_Ce and AUC_central
        // should be within 5% of each other for any KE0 ≥ 0.01 at T=24.
        // We allow slightly wider bounds to accommodate subjects with high ETA_V1
        // (slower β, more drug remaining at t=24) and any KE0 value.
        let ratio = auc_ce / auc_central;
        assert!(
            ratio >= 0.70 && ratio <= 1.05,
            "subject {}: AUC_Ce/AUC_central = {ratio:.4} (expected [0.70, 1.05]); \
             AUC_Ce={auc_ce:.3}, AUC_central={auc_central:.3}",
            sr.id
        );

        auc_ce_vec.push(auc_ce);
        auc_central_vec.push(auc_central);
    }

    // Median AUC_Ce across subjects should be near the analytical reference
    // of ≈21.6 mg/L·h for population-average parameters; allow a wide range
    // [12, 35] to accommodate IIV in CL/V1 and the unidentified KE0.
    auc_ce_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_auc_ce = if auc_ce_vec.len() % 2 == 0 {
        (auc_ce_vec[auc_ce_vec.len() / 2 - 1] + auc_ce_vec[auc_ce_vec.len() / 2]) / 2.0
    } else {
        auc_ce_vec[auc_ce_vec.len() / 2]
    };
    assert!(
        median_auc_ce >= 12.0 && median_auc_ce <= 35.0,
        "median AUC_Ce = {median_auc_ce:.3} mg/L·h is outside the expected range \
         [12, 35] for the two_cpt_iv dataset; analytical reference ≈ 21.6 mg/L·h \
         at population-average parameters (CL=4, V1=12, Q=2, V2=25, KE0=0.5)"
    );

    // Summarise for --nocapture inspection.
    auc_central_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_auc_c = auc_central_vec[auc_central_vec.len() / 2];
    println!(
        "[auc_ce_2cpt_iv_effect] OFV={:.2}  n_subj={}  \
         median AUC_Ce={median_auc_ce:.2}  median AUC_central={median_auc_c:.2}",
        result.ofv,
        result.subjects.len(),
    );
}

// ── Regression: EVID=3 reset in ode_dense_solve_states (fdbbc95) ─────────────
//
// Fix: `ode_dense_solve_states` must add `subject.reset_times` to its
// `break_times` so the ODE state is re-seeded at the reset boundary when
// computing grid-integral derived variables.
//
// Without the fix the break-time list has no entry at t=12.  The solver
// runs in a single segment [0, t_last] and never applies the re-seed, so
// the integral captures the ongoing exponential decay instead of zero.
//
// Model: 1-cpt IV ODE (amount-tracking), 100 mg bolus at t=0, EVID=3 reset
// at t=12.  Observations at [1,4,8,12,16,20,24] h.
//
// Derived variables:
//   AUC_pre  = integral(compartments[0], from=0,  to=12, step=1.0)
//              → session 0 observations (t ≤ 12); should be ≈452 mg·h
//   AUC_post = integral(compartments[0], from=12, to=24, step=1.0)
//              → session 1 observations (t > 12); should be ≈0
//   C_cmt0   = compartments[0]  per-row  → ~0 for t > 12 (event-driven path)
//
// Regression check: `AUC_post < 1.0`.  Without fdbbc95 the value is ≈41 mg·h.

fn evid3_reset_population() -> Population {
    // Session 0: t = 1, 4, 8, 12  (at and before the EVID=3 reset)
    // Session 1: t = 16, 20, 24   (after the reset; model predicts ≈0)
    let obs_times = vec![1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
    let n = obs_times.len();
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![{
            // DV ≈ model prediction: monoexponential up to reset, near-zero after.
            // Additive error on the model side avoids proportional-error issues
            // when IPRED ≈ 0 post-reset.
            let mut s = common::subject(
                "1",
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times,
                vec![8.2, 4.5, 2.0, 0.9, 0.05, 0.05, 0.05],
                vec![1; n],
            );
            s.reset_times = vec![12.0]; // EVID=3 reset at t=12
            s
        }],
    }
}

/// Regression for fdbbc95 — `ode_dense_solve_states` with EVID=3 reset.
///
/// An EVID=3 reset at t=12 zeros all ODE compartments.  The trapezoidal
/// integral of `compartments[0]` over [12, 24] must therefore return ≈0.
///
/// At initial parameters (CL=2 L/h, V=10 L, ke=0.2 h⁻¹, dose=100 mg):
///   AUC_post (grid [12..24], step=1)  = 0       (states zeroed after reset)
///   AUC_pre  (grid [0..12],  step=1)  ≈ 452 mg·h (pre-reset accumulation)
///
/// Without fdbbc95 the reset is silently ignored and AUC_post ≈ 41 mg·h.
#[test]
fn ode_integral_over_compartment_with_evid3_reset() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.25

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ additive(ADD)

[derived]
  AUC_pre  = integral(compartments[0], from=0,  to=12, step=1.0)
  AUC_post = integral(compartments[0], from=12, to=24, step=1.0)
  C_cmt0   = compartments[0]

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = evid3_reset_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let auc_pre_vals = col!("AUC_pre");
        let auc_post_vals = col!("AUC_post");
        let c_cmt0_vals = col!("C_cmt0");

        let n_obs = sr.ipred.len();
        assert_eq!(auc_pre_vals.len(), n_obs, "AUC_pre must have n_obs values");
        assert_eq!(
            auc_post_vals.len(),
            n_obs,
            "AUC_post must have n_obs values"
        );
        assert_eq!(c_cmt0_vals.len(), n_obs, "C_cmt0 must have n_obs values");

        // ── Session 0: observations at t ≤ 12 (indices 0..=3 in obs_times) ──
        // AUC_pre is finite here (window [0,12] is inside session 0).
        // AUC_post is NaN (window [12,24] starts at the session boundary).
        //
        // At initial theta (CL=2, V=10) the expected trapezoidal AUC_pre ≈ 452
        // (see module-level comment). We allow a wide range to accommodate any
        // parameter movement over 2 outer iterations.
        let auc_pre_finite: Vec<f64> = auc_pre_vals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(
            !auc_pre_finite.is_empty(),
            "subject {}: at least one finite AUC_pre expected (session 0 observations)",
            sr.id
        );
        for &v in &auc_pre_finite {
            assert!(
                v > 50.0,
                "subject {}: AUC_pre = {v:.2} — unexpectedly small; \
                 expected >50 mg·h for integral(compartments[0], 0→12) with a 100 mg dose",
                sr.id
            );
        }

        // ── Session 1: observations at t > 12 (indices 4..=6 in obs_times) ──
        // AUC_post is finite here (window [12,24] is inside session 1).
        // AUC_pre is NaN (window [0,12] ends at the session boundary).
        //
        // Key regression assertion: after an EVID=3 reset at t=12 the ODE
        // state is zeroed, so the integral over [12,24] must be ≈0.
        // Without fdbbc95 the integral returns ≈41 mg·h (reset ignored).
        let auc_post_finite: Vec<f64> = auc_post_vals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(
            !auc_post_finite.is_empty(),
            "subject {}: at least one finite AUC_post expected (session 1 observations)",
            sr.id
        );
        for &v in &auc_post_finite {
            assert!(
                v < 1.0,
                "subject {}: AUC_post = {v:.4} — EVID=3 reset at t=12 not applied in \
                 ode_dense_solve_states; expected ≈0 (with fix), pre-fix value ≈41 mg·h. \
                 Regression: fdbbc95 (add reset_times to break_times)",
                sr.id
            );
        }

        // ── Per-row compartment states (event-driven path) ───────────────────
        // C_cmt0 = compartments[0] for session 1 obs (t > 12) should be ≈0
        // because the reset zeroes the central compartment and no new dose fires.
        // Values at the last 3 observations (t=16,20,24) are checked.
        let n = c_cmt0_vals.len();
        for (j, &v) in c_cmt0_vals.iter().enumerate().skip(n.saturating_sub(3)) {
            assert!(
                v.is_finite() && v < 0.5,
                "subject {}: C_cmt0 at obs {} (ipred={:.6}) = {v:.6} — \
                 expected ≈0 after EVID=3 reset at t=12",
                sr.id,
                j,
                sr.ipred.get(j).copied().unwrap_or(f64::NAN)
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EVID=4 (reset + re-dose) regression
// ─────────────────────────────────────────────────────────────────────────────

fn evid4_reset_population() -> Population {
    // Session 0: 100 mg IV dose at t=0; obs at t=1,4,8,12.
    // Session 1: 50 mg IV re-dose at t=12 (EVID=4: reset + dose); obs at t=16,20,24.
    let obs_times = vec![1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
    let n = obs_times.len();
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![{
            // Session 0 DV ≈ monoexponential decay from 100 mg.
            // Session 1 DV > 0 because a 50 mg dose fires at t=12.
            let mut s = common::subject(
                "1",
                vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(12.0, 50.0, 1, 0.0, false, 0.0), // re-dose
                ],
                obs_times,
                vec![8.2, 4.5, 2.0, 0.9, 4.0, 2.2, 1.1],
                vec![1; n],
            );
            s.reset_times = vec![12.0]; // EVID=4 reset at t=12 (re-dose also fires)
            s
        }],
    }
}

/// Regression for fdbbc95 — `ode_dense_solve_states` with EVID=4 (reset + re-dose).
///
/// An EVID=4 at t=12 zeros all ODE compartments *and* fires a 50 mg dose.
/// Unlike EVID=3, `AUC_post` must be substantially > 0 because the new dose
/// contributes.  `AUC_post` should also be < `AUC_pre` because the re-dose
/// is only 50 mg (half the initial 100 mg).
///
/// At initial parameters (CL=2 L/h, V=10 L, ke=0.2 h⁻¹):
///   AUC_pre  (grid [0..12],  step=1)  ≈ 452 mg·h
///   AUC_post (grid [12..24], step=1)  ≈ 226 mg·h  (≈ AUC_pre / 2, dose halved)
#[test]
fn ode_integral_over_compartment_with_evid4_reset() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.25

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ additive(ADD)

[derived]
  AUC_pre  = integral(compartments[0], from=0,  to=12, step=1.0)
  AUC_post = integral(compartments[0], from=12, to=24, step=1.0)

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = evid4_reset_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let auc_pre_vals = col!("AUC_pre");
        let auc_post_vals = col!("AUC_post");

        // ── Session 0 (t ≤ 12) — AUC_pre is finite ───────────────────────
        let auc_pre_finite: Vec<f64> = auc_pre_vals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(
            !auc_pre_finite.is_empty(),
            "subject {}: at least one finite AUC_pre expected",
            sr.id
        );
        for &v in &auc_pre_finite {
            assert!(
                v > 50.0,
                "subject {}: AUC_pre = {v:.2} — unexpectedly small; \
                 expected >50 mg·h for 100 mg dose integral over [0,12]",
                sr.id
            );
        }

        // ── Session 1 (t > 12) — AUC_post must be > 0 (new 50 mg dose) ──
        let auc_post_finite: Vec<f64> = auc_post_vals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(
            !auc_post_finite.is_empty(),
            "subject {}: at least one finite AUC_post expected (EVID=4 re-dose should \
             give positive integral)",
            sr.id
        );
        for &v in &auc_post_finite {
            assert!(
                v > 10.0,
                "subject {}: AUC_post = {v:.4} — after EVID=4 reset a 50 mg re-dose \
                 fires; expected AUC_post > 10 mg·h, got ≈0 (reset not applied?)",
                sr.id
            );
            // AUC_post must be < AUC_pre (50 mg vs 100 mg dose, same PK)
            for &pre in &auc_pre_finite {
                assert!(
                    v < pre,
                    "subject {}: AUC_post ({v:.2}) ≥ AUC_pre ({pre:.2}) — re-dose is \
                     half the initial dose so post-reset AUC should be smaller",
                    sr.id
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug regression: out-of-range compartments[i] must return NaN, not 0.0
// Before the fix, `build_derived_vars` only pre-seeded 0..n_expected with NaN;
// indices beyond n_expected fell through to eval_expression's unwrap_or(0.0).
// ─────────────────────────────────────────────────────────────────────────────

/// `compartments[8]` on a 1-cpt model (which has only 1 compartment) must
/// produce NaN — not 0.0 (which looks like an empty compartment and is wrong).
#[test]
fn out_of_range_compartment_index_returns_nan() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_valid = compartments[0]
  C_oor   = compartments[8]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let c_valid = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_valid")
            .expect("C_valid must exist");
        let c_oor = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_oor")
            .expect("C_oor must exist");

        // C_valid should equal IPRED (not NaN)
        for &v in &c_valid.1 {
            assert!(
                v.is_finite(),
                "subject {}: C_valid (compartments[0]) should be finite, got {v}",
                sr.id
            );
        }
        // C_oor (out-of-range) must be NaN, not 0.0
        for (j, &v) in c_oor.1.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: C_oor (compartments[8]) at obs {j} = {v:.6} — \
                 expected NaN for out-of-range index; before the fix this returned 0.0",
                sr.id
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug regression: analytical model + EVID=3 reset + integral(compartments[i])
// Before the fix, analytical_state_at_times was called even for reset subjects,
// returning wrong superposition values instead of NaN.
// ─────────────────────────────────────────────────────────────────────────────

/// Regression: analytical model + EVID=3 reset + `integral(compartments[0])`.
///
/// Before the fix the grid-integral path called `analytical_state_at_times`
/// on the reset subject, which does plain superposition and returns ≈41 mg·h
/// for the post-reset window [12, 24] — identical to the pre-fdbbc95 ODE bug.
///
/// After the fix, the analytical branch returns empty states for reset subjects,
/// so the integral evaluates to NaN (< 2 grid-points inside the session window).
///
/// Also verifies that `W_DERIVED_CMT_RESET_ANALYTICAL` is emitted.
#[test]
fn analytical_integral_over_compartment_with_evid3_reset_returns_nan() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD    ~ 0.25

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)

[derived]
  AUC_pre  = integral(compartments[0], from=0,  to=12, step=1.0)
  AUC_post = integral(compartments[0], from=12, to=24, step=1.0)
  C_cmt0   = compartments[0]

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    // Reuse the same EVID=3 population as the ODE regression test
    let pop = {
        let obs_times = vec![1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
        let n = obs_times.len();
        Population {
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects: vec![{
                let mut s = common::subject(
                    "1",
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    obs_times,
                    vec![8.2, 4.5, 2.0, 0.9, 0.05, 0.05, 0.05],
                    vec![1; n],
                );
                s.reset_times = vec![12.0];
                s
            }],
        }
    };
    let mut opts = FitOptions::default();
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    // W_DERIVED_CMT_RESET_ANALYTICAL must be emitted
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_RESET_ANALYTICAL")),
        "expected W_DERIVED_CMT_RESET_ANALYTICAL warning; got: {:?}",
        result.warnings
    );

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let auc_pre_vals = col!("AUC_pre");
        let auc_post_vals = col!("AUC_post");
        let c_cmt0_vals = col!("C_cmt0");

        // All per-obs compartment states must be NaN (analytical+reset → empty states)
        for (j, &v) in c_cmt0_vals.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: C_cmt0[{j}] = {v:.6} — expected NaN for \
                 analytical+reset subject (compartment states unsupported)",
                sr.id
            );
        }

        // AUC_pre and AUC_post must also be NaN (grid integral on empty states)
        for (j, &v) in auc_pre_vals.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: AUC_pre[{j}] = {v:.6} — expected NaN; before the fix \
                 this returned wrong superposition values (≈452 mg·h pre-reset)",
                sr.id
            );
        }
        for (j, &v) in auc_post_vals.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: AUC_post[{j}] = {v:.6} — expected NaN; before the fix \
                 this returned ≈41 mg·h (reset ignored in analytical superposition)",
                sr.id
            );
        }
    }
}

// ── TV-covariate warning tests (review finding #5) ──────────────────────────

/// Population whose single subject has time-varying WT supplied via
/// `obs_covariates`. Used to trigger `has_tv_covariates() == true` for
/// both the analytical and ODE TV-covariate warning tests.
fn tv_covariate_iv_population() -> Population {
    let obs_times = vec![1.0, 4.0, 12.0, 24.0];
    let n = obs_times.len();
    // Per-observation covariate maps: WT changes over the study.
    let obs_covariates: Vec<HashMap<String, f64>> = obs_times
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut m = HashMap::new();
            m.insert("WT".into(), 60.0 + i as f64 * 5.0);
            m
        })
        .collect();
    Population {
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![{
            let mut s = common::subject(
                "1",
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times,
                vec![5.0, 3.0, 1.5, 0.7],
                vec![1; n],
            );
            s.covariates = {
                let mut m = HashMap::new();
                m.insert("WT".into(), 70.0);
                m
            };
            s.obs_covariates = obs_covariates;
            s
        }],
    }
}

/// Regression for review finding #5 (analytical TV-covariate + compartments[i]).
///
/// For an analytical model whose individual CL depends on a time-varying
/// covariate (WT), `compartment_states` cannot be computed by superposition
/// because the baseline PK params would disagree with the TV ipred.
/// `fit()` must:
///   1. Emit `W_DERIVED_CMT_TV_ANALYTICAL`.
///   2. Return empty inner slices for each observation (so that
///      `compartments[i]` in `[derived]` evaluates to NaN, not a silently
///      wrong value derived from baseline params).
#[test]
fn analytical_tv_covariate_with_compartments_derived_emits_warning() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.01, 50.0)
  theta V(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * (WT/70)^0.75 * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  c0 = compartments[0]

[fit_options]
  method   = foce
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = tv_covariate_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    // Warning must be emitted.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_TV_ANALYTICAL")),
        "expected W_DERIVED_CMT_TV_ANALYTICAL warning; got: {:?}",
        result.warnings
    );
    // compartment_states must be outer-empty (vec![]) — the same convention as IOV
    // and reset subjects — so that downstream .get(j).unwrap_or(&[]) gives &[] and
    // compartments[i] evaluates to NaN rather than a silently-wrong baseline-PK value.
    for sr in &result.subjects {
        assert!(
            sr.compartment_states.is_empty(),
            "TV-covariate analytical subject {} must have outer-empty compartment_states \
             (len={}), got len={}",
            sr.id,
            0,
            sr.compartment_states.len()
        );
    }
}

/// Regression for review finding #5 (ODE TV-covariate + compartments[i]).
///
/// For an ODE model with a time-varying covariate the event-driven path
/// (`ode_predictions_event_driven_with_states`) is used, which holds PK
/// parameters constant at the first observation value for the state trajectory
/// (exact for ipred, approximate for states). `fit()` must emit
/// `W_DERIVED_CMT_TV_ODE` to inform the user that states may be approximate.
/// Unlike the analytical case, states are populated (not empty) because the
/// approximation is documented and useful; only ipred is exact.
#[test]
fn ode_tv_covariate_with_compartments_derived_emits_warning() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.01, 50.0)
  theta V(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * (WT/70)^0.75 * exp(ETA_CL)
  V  = V

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)

[derived]
  c0 = compartments[0]

[fit_options]
  method   = foce
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = tv_covariate_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    // Warning must be emitted.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_TV_ODE")),
        "expected W_DERIVED_CMT_TV_ODE warning; got: {:?}",
        result.warnings
    );
    // For ODE TV-covariate subjects, states are populated (approximate via
    // first-obs PK params) — they must be non-empty and finite.
    for sr in &result.subjects {
        assert_eq!(
            sr.compartment_states.len(),
            sr.ipred.len(),
            "compartment_states must have {} outer entries for TV-covariate ODE subject {}",
            sr.ipred.len(),
            sr.id
        );
        for (j, row) in sr.compartment_states.iter().enumerate() {
            assert!(
                !row.is_empty(),
                "ODE TV-covariate subject {}: compartment_states[{j}] must not be empty",
                sr.id
            );
            for (k, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "ODE TV-covariate subject {}: compartment_states[{j}][{k}] = {v:.6} is not finite",
                    sr.id
                );
            }
        }
    }
}

/// Regression: integral grid path for analytical TV-covariate subjects must
/// return NaN (not a silently-wrong finite approximation).
///
/// Before the fix, `eval_integral_grid` in `api.rs` had a `has_resets()` guard
/// but no `has_tv_covariates()` guard. For an analytical TV-covariate subject
/// with no resets it fell through to `analytical_state_at_times` using baseline
/// covariates, producing a finite but wrong approximate AUC while the per-obs
/// `compartments[i]` correctly returned NaN.
///
/// After the fix, `integral(compartments[0])` must also be NaN for such subjects,
/// consistent with the per-observation path and W_DERIVED_CMT_TV_ANALYTICAL.
#[test]
fn analytical_tv_covariate_integral_of_compartment_is_nan() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.01, 50.0)
  theta V(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * (WT/70)^0.75 * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  cmt0     = compartments[0]
  AUC_cmt0 = integral(compartments[0], from=0, to=24)

[fit_options]
  method   = foce
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = tv_covariate_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_TV_ANALYTICAL")),
        "expected W_DERIVED_CMT_TV_ANALYTICAL; got: {:?}",
        result.warnings
    );

    for sr in &result.subjects {
        // Per-observation column: compartments[0] → NaN.
        let cmt0_col = sr
            .extra_columns
            .iter()
            .find(|(name, _)| name == "cmt0")
            .map(|(_, vals)| vals.as_slice())
            .unwrap_or(&[]);
        for (j, &v) in cmt0_col.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: cmt0[{j}] = {v} — expected NaN for TV-covariate analytical",
                sr.id
            );
        }

        // Integral column: integral(compartments[0]) must also be NaN — not a
        // finite approximate value computed with baseline PK params.
        let auc_col = sr
            .extra_columns
            .iter()
            .find(|(name, _)| name == "AUC_cmt0")
            .map(|(_, vals)| vals.as_slice())
            .unwrap_or(&[]);
        for (j, &v) in auc_col.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: AUC_cmt0[{j}] = {v} — integral(compartments[0]) must be \
                 NaN for TV-covariate analytical subject (was a finite wrong approximation \
                 before the has_tv_covariates() guard was added to eval_integral_grid)",
                sr.id
            );
        }
    }
}

/// Regression for #400: integral grid path for an analytical oral model with a
/// zero-order input into the depot must return NaN (not a silently-wrong finite
/// approximation).
///
/// The superposition state helper (`single_dose_states`) models an oral infusion
/// as a depot-bypassing central infusion, so without a guard the grid path in
/// `eval_integral_grid` (api.rs) fell through to `analytical_state_at_times` and
/// produced finite-but-wrong compartment amounts — contradicting both the
/// per-observation `compartments[i]` (correctly NaN) and the
/// `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL` warning's "evaluate to NaN".
#[test]
fn analytical_oral_depot_infusion_integral_of_compartment_is_nan() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.01, 50.0)
  theta V(50.0, 0.1, 500.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V
  KA = KA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  cmt0     = compartments[0]
  AUC_cmt0 = integral(compartments[0], from=0, to=24)

[fit_options]
  method   = foce
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    // One subject with an explicit zero-order infusion into the depot (cmt 1):
    // rate 25 over AMT/rate = 4 h, then first-order KA absorption.
    let obs_times = vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let n = obs_times.len();
    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)],
            obs_times,
            vec![0.8, 1.4, 1.6, 0.9, 0.5, 0.2],
            vec![2; n],
        )],
    };
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL")),
        "expected W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL; got: {:?}",
        result.warnings
    );

    for sr in &result.subjects {
        // Per-observation column: compartments[0] → NaN.
        let cmt0_col = sr
            .extra_columns
            .iter()
            .find(|(name, _)| name == "cmt0")
            .map(|(_, vals)| vals.as_slice())
            .unwrap_or(&[]);
        for (j, &v) in cmt0_col.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: cmt0[{j}] = {v} — expected NaN for oral depot-infusion analytical",
                sr.id
            );
        }
        // Integral grid path: integral(compartments[0]) must also be NaN — this is
        // the path that previously returned a finite wrong AUC (the #400 grid guard).
        let auc_col = sr
            .extra_columns
            .iter()
            .find(|(name, _)| name == "AUC_cmt0")
            .map(|(_, vals)| vals.as_slice())
            .unwrap_or(&[]);
        for (j, &v) in auc_col.iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: AUC_cmt0[{j}] = {v} — integral(compartments[0]) must be NaN \
                 for an oral depot-infusion analytical subject (the #400 grid guard)",
                sr.id
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug regression: IOV model + integral(compartments[i]) → NaN, not finite wrong
// Before Fix 3 (eval_integral_grid IOV guard), analytical IOV subjects without
// resets or TV-covariates fell through to `analytical_state_at_times` with
// BSV-only eta (kappa=0), returning finite but wrong integral values silently.
// ─────────────────────────────────────────────────────────────────────────────

/// Regression for Fix 3 (eval_integral_grid IOV guard): analytical IOV model
/// + `integral(compartments[0])`.
///
/// Before the fix, `eval_integral_grid` had no IOV guard. Subjects with
/// IOV (kappa) parameters and no resets / no TV-covariates fell through to
/// `analytical_state_at_times` with BSV-only eta (kappa=0), producing finite
/// but wrong integral values (the per-occasion kappa contribution was silently
/// zeroed out).
///
/// After the fix, `model.n_kappa > 0` returns `vec![]` immediately, so every
/// grid point evaluates to NaN — consistent with per-obs `compartment_states`
/// being empty for IOV subjects.
///
/// Also verifies that `W_DERIVED_CMT_IOV_UNSUPPORTED` is emitted and that
/// per-obs `compartment_states` is outer-empty for IOV subjects.
#[test]
fn iov_analytical_integral_of_compartment_is_nan() {
    const MODEL: &str = "
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma ADD ~ 0.25

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)

[derived]
  cmt0     = compartments[0]
  AUC_cmt0 = integral(compartments[0], from=0, to=24, step=1.0)

[fit_options]
  method  = foce
  maxiter = 5
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    assert!(
        model.n_kappa > 0,
        "model must have n_kappa > 0 for this IOV regression test to be meaningful"
    );

    // Two-occasion population: occ 1 at t=[1,4,12], occ 2 at t=[25,28,36].
    // Two doses: 100 mg at t=0 (occ 1) and 100 mg at t=24 (occ 2).
    // Observations are rough IV-bolus concentrations (CL=2,V=10,dose=100):
    //   C(t) = 10*exp(-0.2*t) for each occasion.
    let obs_times = vec![1.0, 4.0, 12.0, 25.0, 28.0, 36.0];
    let n = obs_times.len();
    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![{
            let mut s = common::subject(
                "1",
                vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // occ 1
                    DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0), // occ 2
                ],
                obs_times,
                vec![7.0, 4.0, 1.5, 7.5, 4.5, 2.0],
                vec![1; n],
            );
            s.occasions = vec![1, 1, 1, 2, 2, 2];
            s.dose_occasions = vec![1, 2];
            s
        }],
    };

    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    // W_DERIVED_CMT_IOV_UNSUPPORTED must be emitted when kappas are present.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_IOV_UNSUPPORTED")),
        "expected W_DERIVED_CMT_IOV_UNSUPPORTED warning; got: {:?}",
        result.warnings
    );

    for sr in &result.subjects {
        // IOV subjects: compartment_states must be outer-empty (vec![]).
        assert!(
            sr.compartment_states.is_empty(),
            "subject {}: compartment_states must be outer-empty for IOV subjects, \
             got {} inner vecs",
            sr.id,
            sr.compartment_states.len()
        );

        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }

        // Per-obs cmt0 must be NaN (IOV subjects have no compartment states).
        for (j, &v) in col!("cmt0").iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: cmt0[{j}] = {v} — expected NaN for IOV subject",
                sr.id
            );
        }

        // integral must also be NaN.
        // Fix 3 regression: before the eval_integral_grid IOV guard was added,
        // this returned a finite approximation computed with BSV-only eta
        // (kappa=0), silently ignoring the per-occasion kappa contribution.
        for (j, &v) in col!("AUC_cmt0").iter().enumerate() {
            assert!(
                v.is_nan(),
                "subject {}: AUC_cmt0[{j}] = {v} — expected NaN; before Fix 3 this \
                 returned a finite wrong value (kappa zeroed in analytical_state_at_times)",
                sr.id
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug regression: ODE model with [scaling] + compartments[i] in [derived]
// Before Fix 1 (apply_scaling in compute_predictions_with_states ODE branch),
// SubjectResult.ipred was NOT divided by obs_scale when any [derived] expression
// referenced compartments[i] — because uses_compartments=true caused
// compute_predictions_with_states to be called instead of
// compute_predictions_with_tv, and the new function's ODE branch missed the
// apply_scaling / apply_log_transform insertion point.
// ─────────────────────────────────────────────────────────────────────────────

/// Regression for Fix 1 (apply_scaling in compute_predictions_with_states ODE
/// branch): ODE model with `[scaling] obs_scale = 1000` + `[derived] cmt0 =
/// compartments[0]`.
///
/// `cmt0 = compartments[0]` sets `uses_compartments = true`, which causes
/// `compute_predictions_with_states` (the new function introduced by PR #207)
/// to be called for this subject.  Before Fix 1, the ODE branch of that
/// function returned ipred straight from the ODE readout without calling
/// `apply_scaling`.  After Fix 1 the same `apply_scaling` / `apply_log_transform`
/// insertion point used by `compute_predictions_with_tv_into_with_schedule` is
/// applied, so `ipred = raw_ode_state / 1000`.
///
/// `compartments[0]` intentionally returns the raw unscaled state (documented
/// design).  After Fix 1: `cmt0[j] == ipred[j] * 1000` exactly (both values
/// originate from the same ODE solve; the only difference is the division).
/// Before Fix 1: `cmt0[j] ≈ ipred[j]` (no division applied), so the ratio
/// would be ~1 instead of ~1000 and this check would fail.
#[test]
fn ode_with_scaling_ipred_is_correctly_scaled() {
    // Amount-based 1-cpt IV ODE.  DV = central_amount / 1000 (trivial scaling
    // chosen to make the regression numerically obvious).
    // With CL=2 L/h, V=10 L, dose=100 mg: central(t) = 100*exp(-0.2*t) mg.
    // ipred(t) = central(t)/1000 ≈ [0.082, 0.045, 0.011, 0.001] at t=[1,4,12,24].
    const MODEL: &str = "
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  obs_scale = 1000

[derived]
  cmt0 = compartments[0]

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");

    // Observations roughly consistent with central_amount/1000.
    let obs_times = vec![1.0, 4.0, 12.0, 24.0];
    let n = obs_times.len();
    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            vec![0.082, 0.045, 0.011, 0.001],
            vec![1; n],
        )],
    };

    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let cmt0_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "cmt0")
            .expect("cmt0 column must exist")
            .1
            .as_slice();

        assert_eq!(
            cmt0_col.len(),
            sr.ipred.len(),
            "cmt0 and ipred must have the same length"
        );

        for (j, (&raw, &ip)) in cmt0_col.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                raw.is_finite() && raw > 0.0,
                "subject {}: cmt0[{j}] = {raw} — raw ODE state must be finite and positive",
                sr.id
            );
            assert!(
                ip.is_finite() && ip > 0.0,
                "subject {}: ipred[{j}] = {ip} — scaled ipred must be finite and positive",
                sr.id
            );
            // After Fix 1: ipred = raw_state / 1000, so raw_state = ipred * 1000.
            // Both originate from the same ODE solve; only a single division
            // separates them, so relative error must be negligible (< 1e-9).
            // Before Fix 1: apply_scaling was not called in the ODE branch of
            // compute_predictions_with_states, so ipred ≈ raw_state and
            // raw_state / (ipred * 1000) ≈ 0.001 — this assertion would fail.
            let ratio = raw / (ip * 1000.0);
            let scaled = ip * 1000.0;
            assert!(
                (ratio - 1.0).abs() < 1e-9,
                "subject {}: obs {j}: cmt0={raw:.6e} should equal ipred*1000={scaled:.6e} \
                 (ratio={ratio:.6}); before Fix 1 apply_scaling was skipped in the \
                 ODE branch of compute_predictions_with_states",
                sr.id
            );
        }
    }
}
