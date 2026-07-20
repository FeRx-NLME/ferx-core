//! Exact oracle for **infusion into an oral model's peripheral compartment**
//! (#375) — the closed form versus a tight-tolerance RK45 solve of the *same*
//! ODE system.
//!
//! This is the primary correctness test for that feature, and it is deliberately
//! independent of NONMEM: NONMEM's `ADVAN11`/`ADVAN12` forced response carries
//! ~2e-6 relative error on these cases (see
//! `tests/nonmem_dose_compartment_anchor.rs`, where that is measured), so it
//! cannot pin the closed form tighter than that. Writing the same system out as
//! explicit `[odes]` and integrating it to `ode_reltol = 1e-12` can, and does —
//! to ~1e-11.
//!
//! Why the oral peripheral needs no new closed form: nothing flows back into the
//! depot, so a constant rate into a peripheral drives exactly the
//! central/peripheral sub-system that the IV model has. The oral propagator
//! superposes that IV forced response onto its own homogeneous evolution. These
//! tests are what verify that reduction rather than assuming it.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv, Population};
use std::io::Write;
use tempfile::NamedTempFile;

fn pop_of(csv: &str) -> Population {
    let mut f = NamedTempFile::new().expect("temp file");
    write!(f, "{csv}").expect("write");
    f.flush().expect("flush");
    read_nonmem_csv(f.path(), None, None).expect("dataset loads")
}

fn preds(src: &str, csv: &str) -> Vec<f64> {
    let m = parse_full_model(src).expect("model parses").model;
    predict(&m, &pop_of(csv), &m.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect()
}

const TWO_CPT_ORAL_CF: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(80.0, 5.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;

const TWO_CPT_ORAL_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(80.0, 5.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  ode(states=[depot, central, periph])

[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - (CL/V)*central - (Q/V)*central + (Q/V2)*periph
  d/dt(periph)  = (Q/V)*central - (Q/V2)*periph

[scaling]
  y = central / V

[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
  ode_max_steps = 10000000

[error_model]
  DV ~ proportional(PROP)
"#;

const THREE_CPT_ORAL_CF: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(80.0, 5.0, 500.0)
  theta TVQ3(1.0, 0.1, 50.0)
  theta TVV3(120.0, 5.0, 900.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA

[structural_model]
  pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;

const THREE_CPT_ORAL_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(80.0, 5.0, 500.0)
  theta TVQ3(1.0, 0.1, 50.0)
  theta TVV3(120.0, 5.0, 900.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA

[structural_model]
  ode(states=[depot, central, periph1, periph2])

[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - (CL/V)*central - (Q/V)*central + (Q/V2)*periph1 - (Q3/V)*central + (Q3/V3)*periph2
  d/dt(periph1) = (Q/V)*central - (Q/V2)*periph1
  d/dt(periph2) = (Q3/V)*central - (Q3/V3)*periph2

[scaling]
  y = central / V

[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
  ode_max_steps = 10000000

[error_model]
  DV ~ proportional(PROP)
"#;

fn assert_agree(cf: &[f64], ode: &[f64], tol: f64, case: &str) {
    assert_eq!(cf.len(), ode.len(), "{case}: row count");
    assert!(
        ode.iter().all(|x| x.is_finite() && *x > 0.0),
        "{case}: ODE reference must be non-trivial, got {ode:?}"
    );
    for (j, (&c, &o)) in cf.iter().zip(ode).enumerate() {
        let rel = (c - o).abs() / o.abs();
        assert!(
            rel < tol,
            "{case} obs {j}: closed form {c:.14} vs ODE {o:.14} (rel {rel:.2e})"
        );
    }
}

/// `two_cpt_oral`, infusion into CMT=3 (the peripheral), observed in central.
#[test]
fn two_cpt_oral_peripheral_infusion_matches_the_ode_twin() {
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,3,20,1\n\
               1,1,5.0,0,.,2,.,0\n1,4,4.0,0,.,2,.,0\n1,8,3.0,0,.,2,.,0\n1,12,2.0,0,.,2,.,0\n";
    assert_agree(
        &preds(TWO_CPT_ORAL_CF, csv),
        &preds(TWO_CPT_ORAL_ODE, csv),
        1e-9,
        "two_cpt_oral infusion CMT=3",
    );
}

/// `three_cpt_oral`, infusion into each peripheral in turn.
#[test]
fn three_cpt_oral_peripheral_infusion_matches_the_ode_twin() {
    for cmt in [3usize, 4] {
        let csv = format!(
            "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,{cmt},20,1\n\
             1,1,5.0,0,.,2,.,0\n1,4,4.0,0,.,2,.,0\n1,8,3.0,0,.,2,.,0\n1,12,2.0,0,.,2,.,0\n"
        );
        assert_agree(
            &preds(THREE_CPT_ORAL_CF, &csv),
            &preds(THREE_CPT_ORAL_ODE, &csv),
            1e-9,
            &format!("three_cpt_oral infusion CMT={cmt}"),
        );
    }
}

/// The interesting combination: a peripheral infusion **and** an oral depot dose
/// on the same subject, so the depot's homogeneous evolution and the peripheral
/// forced response superpose. A reduction that only held with an empty depot
/// would pass the tests above and fail here.
#[test]
fn peripheral_infusion_plus_oral_dose_matches_the_ode_twin() {
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
               1,0,.,1,100,3,20,1\n\
               1,2,.,1,200,1,0,1\n\
               1,1,5.0,0,.,2,.,0\n1,4,4.0,0,.,2,.,0\n1,8,3.0,0,.,2,.,0\n1,12,2.0,0,.,2,.,0\n";
    assert_agree(
        &preds(TWO_CPT_ORAL_CF, csv),
        &preds(TWO_CPT_ORAL_ODE, csv),
        1e-9,
        "two_cpt_oral peripheral infusion + depot dose",
    );
}

/// …and the 3-cpt version, with a depot dose landing *during* the infusion
/// window so both forcings are simultaneously active.
#[test]
fn three_cpt_peripheral_infusion_overlapping_oral_dose_matches_the_ode_twin() {
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
               1,0,.,1,100,3,20,1\n\
               1,2,.,1,200,1,0,1\n\
               1,1,5.0,0,.,2,.,0\n1,3,4.5,0,.,2,.,0\n1,4,4.0,0,.,2,.,0\n\
               1,8,3.0,0,.,2,.,0\n1,12,2.0,0,.,2,.,0\n";
    assert_agree(
        &preds(THREE_CPT_ORAL_CF, csv),
        &preds(THREE_CPT_ORAL_ODE, csv),
        1e-9,
        "three_cpt_oral peripheral infusion + overlapping depot dose",
    );
}
