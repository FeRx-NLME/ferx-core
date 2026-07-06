//! NONMEM cross-check for the **ODE** non-IOV time-varying-covariate path combined
//! with an `EVID=2` covariate breakpoint **and** an `ExpressionScale` `obs_scale`
//! divisor — the two cells closed in #486. This is a value-level (`predict`) check, so
//! the divisor here is θ-only (`V1 = TVV1`, `obs_scale = V1`): it exercises the
//! production `ExpressionScale` apply path against real NONMEM PREDs. The η-dependence of
//! that quotient — the actual #486 gradient claim — is validated separately by the
//! `sens::ode_provider` unit tests, which drive `obs_scale = expr(θ,η)` with non-zero η.
//!
//! This reuses the committed NONMEM reference from `tvcov_intermediate_nonmem.rs`
//! (issue #455): the same dataset (`tests/nonmem/tvcov_intermediate_evid2.csv`, an
//! `EVID=2` record at t=3 changing `WT` 70→95) and the same NONMEM 7.5.1 PREDs from
//! `tests/nonmem/tvcov_intermediate_evid2.ctl` (ADVAN3 TRANS4, fixed
//! `CL=10*(WT/70)^0.75`, `V1=50`, `Q=15`, `V2=100`, `S1=V1` so `IPRED = F = A1/V1`).
//!
//! Here the structural model is expressed as a **2-cpt `[odes]`** with `obs_cmt=central`
//! and the concentration produced by an `[scaling] obs_scale = V1` divisor (`A1/V1`),
//! rather than the analytical `two_cpt_iv` of the #455 test or a Form-C `y = central/V1`
//! readout. So a match against the same NONMEM PREDs confirms the production ODE
//! predictor carries the `EVID=2` covariate breakpoint **and** applies the subject-static
//! `obs_scale` quotient correctly — the prediction the #486 analytic gradient is built on
//! (the gradient itself is validated against this predictor by the `sens::ode_provider`
//! unit tests). `obs_scale = V1` exercises the `ExpressionScale` apply path even though
//! `V1` is θ-only here, so `η = 0` and the divisor is constant — the production
//! `apply_scaling` quotient still runs.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::path::Path;

const MODEL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  WTREL = WT / 70
  WTCL  = WTREL ^ THETA_WT
  CL = TVCL * WTCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2

[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])

[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral

[scaling]
  obs_scale = V1

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

// NONMEM 7.5.1 PREDs (observation rows), from tests/nonmem/tvcov_intermediate_evid2.ctl.
const NONMEM: &[(f64, f64)] = &[
    (0.5, 1.5669061473),
    (2.0, 0.82189504423),
    (6.0, 0.27541521381),
    (12.0, 0.16633647993),
];

#[test]
fn ode_tvcov_evid2_obs_scale_matches_nonmem_pred() {
    let model = parse_full_model(MODEL).expect("model parses").model;
    let population = read_nonmem_csv(
        Path::new("tests/nonmem/tvcov_intermediate_evid2.csv"),
        None,
        None,
    )
    .expect("dataset loads");

    let preds = predict(&model, &population, &model.default_params);
    assert_eq!(preds.len(), NONMEM.len());

    for (p, &(t, expected)) in preds.iter().zip(NONMEM) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "prediction time {} != expected {t}",
            p.time
        );
        // ODE (RK45) vs NONMEM's closed-form ADVAN3: tight ODE tolerances bring the
        // event-driven dual walk to within a few ulps-scaled units of the analytical
        // reference, so a 1e-6 relative band is comfortable while still catching a
        // dropped breakpoint or a mis-applied scale quotient.
        let rel = (p.pred - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel < 1e-6,
            "t={t}: ferx ODE PRED {:.10} vs NONMEM {:.10} (rel err {:.2e})",
            p.pred,
            expected,
            rel
        );
    }
}
