//! #2104: the closed-form *recording* path (`analytic_static_predictions`, which every
//! adjoint recording of a static subject goes through) applies the same degenerate-`CL`/`V`
//! rule as the value path it differentiates.
use super::*;
use crate::sens::dual1::Dual1;
use crate::types::{
    DoseEvent, PK_IDX_CL, PK_IDX_KA, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
};

fn model_for(pk_model: PkModel) -> CompiledModel {
    let pk_line = match pk_model {
        PkModel::OneCptIv => "pk one_cpt_iv(cl=CL, v=V)",
        PkModel::OneCptOral => "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
        PkModel::TwoCptIv => "pk two_cpt_iv(cl=CL, v1=V, q=Q, v2=V2)",
        PkModel::TwoCptOral => "pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)",
        PkModel::ThreeCptIv => "pk three_cpt_iv(cl=CL, v1=V, q2=Q, v2=V2, q3=Q3, v3=V3)",
        PkModel::ThreeCptOral => "pk three_cpt_oral(cl=CL, v1=V, q2=Q, v2=V2, q3=Q3, v3=V3, ka=KA)",
        other => unreachable!("{other:?} is not in this fixture"),
    };
    let src = format!(
        "
[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV(80.0, 1.0, 1000.0)
  theta TVQ(3.0, 0.01, 100.0)
  theta TVV2(40.0, 1.0, 1000.0)
  theta TVQ3(2.0, 0.01, 100.0)
  theta TVV3(60.0, 1.0, 1000.0)
  theta TVKA(1.2, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  Q = TVQ
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA

[structural_model]
  {pk_line}

[error_model]
  DV ~ proportional(PROP)
"
    );
    let model = crate::parser::model_parser::parse_model_string(&src).expect("fixture parses");
    assert_eq!(model.pk_model, pk_model);
    model
}

fn subject() -> Subject {
    let obs_times = vec![0.0, 1.0, 10.0, 12.0, 13.0, 24.0];
    let n = obs_times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 500.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: std::collections::HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Every closed-form model × every degenerate `CL` / `V`: the recording's value equals the
/// prediction it differentiates (`pk::predict_concentration`) — the same number, or both
/// `NaN` — and on a degenerate case its jet is a constant 0, with no `NaN` derivative.
///
/// Mutation (run): drop the `d.cl_or_v_degenerate()` arm of `superpose_doses`' ladder → the
/// `CL = +inf` rows die (first at `OneCptIv`, `t = 0`: the kernels' `≤ 0` guards let it
/// through and the recording is `NaN` where the prediction is 0). The matching skip in
/// `with_disposition` is not observable — with the arm in place the disposition is never
/// read — and only keeps `NaN` macro rates from being built.
#[test]
fn static_recording_matches_the_prediction_on_a_degenerate_clearance_or_volume() {
    let subj = subject();
    let cases: [(&str, usize, f64); 8] = [
        ("control", PK_IDX_CL, 5.0),
        ("CL=0", PK_IDX_CL, 0.0),
        ("CL=-1", PK_IDX_CL, -1.0),
        ("CL=+inf", PK_IDX_CL, f64::INFINITY),
        ("CL=NaN", PK_IDX_CL, f64::NAN),
        ("V=-80", PK_IDX_V, -80.0),
        ("V=+inf", PK_IDX_V, f64::INFINITY),
        ("V=NaN", PK_IDX_V, f64::NAN),
    ];
    for pk_model in [
        PkModel::OneCptIv,
        PkModel::OneCptOral,
        PkModel::TwoCptIv,
        PkModel::TwoCptOral,
        PkModel::ThreeCptIv,
        PkModel::ThreeCptOral,
    ] {
        let model = model_for(pk_model);
        for (label, idx, val) in cases {
            let mut pk = crate::types::PkParams::default();
            pk.values[PK_IDX_CL] = 5.0;
            pk.values[PK_IDX_V] = 80.0;
            pk.values[PK_IDX_Q] = 3.0;
            pk.values[PK_IDX_V2] = 40.0;
            pk.values[PK_IDX_Q3] = 2.0;
            pk.values[PK_IDX_V3] = 60.0;
            pk.values[PK_IDX_KA] = 1.2;
            pk.values[crate::types::PK_IDX_F] = 1.0;
            pk.values[idx] = val;
            let at = format!("{pk_model:?} {label}");

            let want: Vec<f64> = subj
                .obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(pk_model, &subj.doses, t, &pk))
                .collect();
            let got = analytic_static_predictions::<f64>(&model, &subj, &pk, |_| None, &[], &[])
                .unwrap_or_else(|| panic!("{at}: the static recording must serve this subject"));
            // CL and V seeded on two η-tangent axes.
            let jets = analytic_static_predictions::<Dual1<2>>(
                &model,
                &subj,
                &pk,
                |slot| match slot {
                    PK_IDX_CL => Some(Dual1::var(pk.values[PK_IDX_CL], 0)),
                    PK_IDX_V => Some(Dual1::var(pk.values[PK_IDX_V], 1)),
                    _ => None,
                },
                &[],
                &[],
            )
            .unwrap();
            for (j, ((&w, &g), jet)) in want.iter().zip(&got).zip(&jets).enumerate() {
                let t = subj.obs_times[j];
                if w.is_nan() || g.is_nan() {
                    assert!(
                        w.is_nan() && g.is_nan() && jet.value.is_nan(),
                        "{at} t={t}: prediction {w}, recording {g}, jet {}",
                        jet.value
                    );
                    continue;
                }
                let tol = 1e-12 * w.abs().max(1.0);
                assert!(
                    (w - g).abs() <= tol && (w - jet.value).abs() <= tol,
                    "{at} t={t}: prediction {w}, recording {g}, jet {}",
                    jet.value
                );
                if label != "control" {
                    assert_eq!(
                        (jet.value, jet.grad),
                        (0.0, [0.0, 0.0]),
                        "{at} t={t}: a dropped dose records a constant 0"
                    );
                }
            }
            // Non-degeneracy: the control is live and its jet moves with CL.
            if label == "control" {
                assert!(got[2] > 0.1, "{at}: control is inert ({got:?})");
                assert!(
                    jets[2].grad[0].is_finite() && jets[2].grad[0] < 0.0,
                    "{at}: ∂C/∂CL at t = 10 must be finite and negative ({:?})",
                    jets[2].grad
                );
            }
        }
    }
}
