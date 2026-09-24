//! NONMEM cross-check for analytical time-varying covariates with intermediate
//! individual-parameter assignments and an `EVID=2` covariate update.
//!
//! This is the small reproducer for issue #455. The Ferx model has a
//! `two_cpt_iv` structural model, but `[individual_parameters]` contains extra
//! intermediate assignments before the four structural PK outputs. Those
//! intermediate rows make `model.pk_indices` longer than the compiled program's
//! actual PK-output slots; the TV-cov Dual2 path must seed/scatter using the
//! compiled outputs.
//!
//! ## NONMEM reference
//!
//! The committed control stream at
//! `tests/nonmem/tvcov_intermediate_evid2.ctl` was run with NONMEM 7.5.1:
//!
//! ```text
//! docker exec pmx bash -lc 'cd /tmp/ferx_tvcov_intermediate && \
//!   /opt/NONMEM/nm751/run/nmfe75 tvcov_intermediate_evid2.ctl tvcov_intermediate_evid2.lst'
//! ```
//!
//! It uses ADVAN3 TRANS4 with fixed `CL=10*(WT/70)^0.75`, `V1=50`, `Q=15`,
//! `V2=100`, and an `EVID=2` record at `t=3` changing `WT` from 70 to 95. The
//! NONMEM table PREDs for observation rows are transcribed below.

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
  BASECL = TVCL * WTCL
  CL = BASECL * exp(ETA_CL)
  V1 = TVV1
  QBASE = TVQ
  Q  = QBASE
  V2BASE = TVV2
  V2 = V2BASE

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const NONMEM: &[(f64, f64)] = &[
    (0.5, 1.5669061473),
    (2.0, 0.82189504423),
    (6.0, 0.27541521381),
    (12.0, 0.16633647993),
];

#[test]
fn analytical_tvcov_intermediate_evid2_matches_nonmem_pred() {
    let model = parse_full_model(MODEL).expect("model parses").model;
    let population = read_nonmem_csv(
        Path::new("tests/nonmem/tvcov_intermediate_evid2.csv"),
        None,
        None,
    )
    .expect("dataset loads");

    let preds = predict(&model, &population, &model.default_params).unwrap();
    assert_eq!(preds.len(), NONMEM.len());

    for (p, &(t, expected)) in preds.iter().zip(NONMEM) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "prediction time {} != expected {t}",
            p.time
        );
        let rel = (p.pred - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel < 1e-8,
            "t={t}: ferx PRED {:.10} vs NONMEM {:.10} (rel err {:.2e})",
            p.pred,
            expected,
            rel
        );
    }
}
