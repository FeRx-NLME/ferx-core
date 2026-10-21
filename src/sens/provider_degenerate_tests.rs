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

fn subject(first_rate: f64) -> Subject {
    let obs_times = vec![0.0, 1.0, 10.0, 12.0, 13.0, 24.0];
    let n = obs_times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        // First (IV models) a 5 h infusion observed at its start, `t = 0`, with nothing
        // else arrived — the 2-cpt closed form's own arithmetic returns 0 there under a
        // `NaN` CL while the value path's explicit rule says `NaN` — then a bolus on the
        // residual. An oral model takes a bolus first instead: a central infusion on it is
        // outside this recording's scope.
        doses: vec![
            DoseEvent::new(0.0, 500.0, 1, first_rate, false, 0.0),
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
/// `NaN` — and on a degenerate case its jet is a constant 0, with no `NaN` derivative, while
/// a `NaN` one carries a `NaN` jet.
///
/// Mutation (run): drop the `d.cl_or_v_degenerate()` arm of `superpose_doses`' ladder → the
/// `CL = +inf` rows die (first at `OneCptIv`, `t = 0`: the kernels' `≤ 0` guards let it
/// through and the recording is `NaN` where the prediction is 0). The matching skip in
/// `with_disposition` is not observable — with the arm in place the disposition is never
/// read — and only keeps `NaN` macro rates from being built. Drop the `NaN` arm after it →
/// the `TwoCptIv` `CL = NaN` row at the infusion's start (`t = 0`) dies.
#[test]
fn static_recording_matches_the_prediction_on_a_degenerate_clearance_or_volume() {
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
        let oral = matches!(
            pk_model,
            PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
        );
        let subj = subject(if oral { 0.0 } else { 100.0 });
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
                    // A `NaN` value with a finite jet lets FOCEI's `H̃` factor and the
                    // subject's `NaN` objective escape the `1e20` repel (#2134): measured
                    // on a 1-cpt oral fit with one subject's covariate `NaN`, OFV `NaN`
                    // with a bare `NaN`, 2e20 with `nan_jet`.
                    assert!(
                        jet.grad.iter().all(|x| x.is_nan()),
                        "{at} t={t}: a NaN recording needs a NaN jet ({:?})",
                        jet.grad
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

/// The degenerate rule lives in the closed forms themselves too, not only in the private
/// dispatchers, so a caller of the exported kernels sees the same convention as
/// `predict_concentration` (#2104 review). Every 1-/2-/3-cpt dispatcher × every dose kind ×
/// every degenerate case returns exactly 0, and on `Dual1` a constant 0 — before, `CL = +inf`
/// passed the kernels' `≤ 0` guards and came back `NaN` (`exp(−inf · 0)`). The f64 public
/// entry points and the superposition state helpers (the peripheral forms) are checked too.
///
/// Mutations (run): restore any one kernel's `cl.val() <= 0.0` term in place of
/// `cl_or_v_degenerate` → the `CL=+inf` rows reaching that kernel die; delete the
/// override ahead of the `F` post-multiply in one dispatcher (`one_cpt_conc_g`,
/// `two_cpt_conc_d`, `three_cpt_conc_d`) → its `F=NaN` / `F=inf` IV and infusion rows die.
#[test]
fn the_exported_kernels_drop_a_degenerate_dose_themselves() {
    use crate::sens::{
        one_cpt::one_cpt_conc_g, three_cpt::three_cpt_conc_g, two_cpt::two_cpt_conc_g,
    };
    let mk = |rate: f64, ss: bool, ii: f64| DoseEvent::new(0.0, 500.0, 1, rate, ss, ii);
    // (dose, oral) — bolus, infusion, oral bolus, and the SS form of each.
    let doses: [(DoseEvent, bool); 6] = [
        (mk(0.0, false, 0.0), false),
        (mk(100.0, false, 0.0), false),
        (mk(0.0, false, 0.0), true),
        (mk(0.0, true, 12.0), false),
        (mk(100.0, true, 12.0), false),
        (mk(0.0, true, 12.0), true),
    ];
    let cases: [(&str, f64, f64); 5] = [
        ("CL=+inf", f64::INFINITY, 80.0),
        ("V=+inf", 5.0, f64::INFINITY),
        ("CL=0", 0.0, 80.0),
        ("CL=-1", -1.0, 80.0),
        ("V=-80", 5.0, -80.0),
    ];
    let (q, v2, q3, v3, ka, f) = (3.0, 40.0, 2.0, 60.0, 1.2, 1.0);
    // `F` too: a dropped dose stays 0 under a `NaN` or infinite `F`, which the dispatchers'
    // post-multiply of the IV and infusion forms would otherwise turn `NaN` — as
    // `predict_concentration` serves 0 there.
    for f_bio in [f, f64::NAN, f64::INFINITY] {
        for (dose, oral) in &doses {
            for (label, cl, v) in cases {
                for t in [0.0, 1.0, 10.0] {
                    let at = format!(
                        "{label} F={f_bio} rate={} ss={} oral={oral} t={t}",
                        dose.rate, dose.ss
                    );
                    let c1 = one_cpt_conc_g::<f64>(dose, t, cl, v, ka, f_bio, *oral);
                    let c2 = two_cpt_conc_g::<f64>(dose, t, cl, v, q, v2, ka, f_bio, *oral);
                    let c3 =
                        three_cpt_conc_g::<f64>(dose, t, cl, v, q, v2, q3, v3, ka, f_bio, *oral);
                    assert_eq!((c1, c2, c3), (0.0, 0.0, 0.0), "{at}: 1/2/3-cpt kernels");
                    // The production predictor is the reference the dispatchers follow.
                    let mut p = crate::types::PkParams::default();
                    p.values[crate::types::PK_IDX_CL] = cl;
                    p.values[crate::types::PK_IDX_V] = v;
                    p.values[crate::types::PK_IDX_KA] = ka;
                    p.values[crate::types::PK_IDX_F] = f_bio;
                    let model = if *oral {
                        crate::types::PkModel::OneCptOral
                    } else {
                        crate::types::PkModel::OneCptIv
                    };
                    let want =
                        crate::pk::predict_concentration(model, std::slice::from_ref(dose), t, &p);
                    assert_eq!(want, 0.0, "{at}: predict_concentration");
                    let d = |x: f64, i: usize| Dual1::<2>::var(x, i);
                    let j2 = two_cpt_conc_g::<Dual1<2>>(
                        dose,
                        Dual1::constant(t),
                        d(cl, 0),
                        d(v, 1),
                        Dual1::constant(q),
                        Dual1::constant(v2),
                        Dual1::constant(ka),
                        Dual1::constant(f_bio),
                        *oral,
                    );
                    assert_eq!((j2.value, j2.grad), (0.0, [0.0; 2]), "{at}: 2-cpt jet");
                }
            }
        }
    }
    // The f64 public wrappers and the state helpers.
    let bolus = mk(0.0, false, 0.0);
    let inf = f64::INFINITY;
    assert_eq!(crate::pk::one_cpt_iv_bolus(&bolus, 0.0, inf, 80.0), 0.0);
    assert_eq!(
        crate::pk::one_cpt_predict(&bolus, 0.0, inf, 80.0, None, None),
        0.0
    );
    assert_eq!(
        crate::pk::two_compartment::two_cpt_iv_peripheral(&bolus, 1.0, inf, 80.0, q, v2),
        0.0
    );
    assert_eq!(
        crate::pk::two_compartment::two_cpt_oral_peripheral(&bolus, 1.0, inf, 80.0, q, v2, ka, f),
        0.0
    );
    assert_eq!(
        crate::pk::three_compartment::three_cpt_iv_peripherals(
            &bolus, 1.0, inf, 80.0, q, v2, q3, v3
        ),
        [0.0; 2]
    );
    assert_eq!(
        crate::pk::three_compartment::three_cpt_oral_peripherals(
            &bolus, 1.0, inf, 80.0, q, v2, q3, v3, ka, f
        ),
        [0.0; 2]
    );
}

/// The `NaN` half of the rule in the exported kernels: a `NaN` `CL` or `V` on an arrived
/// dose is `NaN` on every 1-/2-/3-cpt dispatcher × dose kind, as `predict_concentration`
/// and the event walk serve it — and 0 before the dose arrives. Before, the 2-cpt infusion
/// forms read the `NaN` disposition (`TwoCpt::new` maps it to `β = 0`) through their
/// numerical-degeneracy guard and returned 0, so `two_cpt_infusion`, `two_cpt_infusion_ss`
/// and `two_cpt_predict` disagreed with the bolus forms and the engines (#2104 review).
///
/// Mutation (run): delete the `superposed_dose_override` check in `two_cpt_infusion_d` →
/// the 2-cpt infusion `NaN` rows die (0 returned).
#[test]
fn the_exported_kernels_serve_a_nan_clearance_or_volume_as_nan() {
    use crate::sens::{
        one_cpt::one_cpt_conc_g, three_cpt::three_cpt_conc_g, two_cpt::two_cpt_conc_g,
    };
    let mk = |rate: f64, ss: bool, ii: f64| DoseEvent::new(0.0, 500.0, 1, rate, ss, ii);
    let doses: [(DoseEvent, bool); 6] = [
        (mk(0.0, false, 0.0), false),
        (mk(100.0, false, 0.0), false),
        (mk(0.0, false, 0.0), true),
        (mk(0.0, true, 12.0), false),
        (mk(100.0, true, 12.0), false),
        (mk(0.0, true, 12.0), true),
    ];
    let (q, v2, q3, v3, ka, f) = (3.0, 40.0, 2.0, 60.0, 1.2, 1.0);
    let nan = f64::NAN;
    for (dose, oral) in &doses {
        for (label, cl, v) in [("CL=NaN", nan, 80.0), ("V=NaN", 5.0, nan)] {
            for t in [0.0, 1.0, 10.0] {
                let at = format!(
                    "{label} rate={} ss={} oral={oral} t={t}",
                    dose.rate, dose.ss
                );
                let c1 = one_cpt_conc_g::<f64>(dose, t, cl, v, ka, f, *oral);
                let c2 = two_cpt_conc_g::<f64>(dose, t, cl, v, q, v2, ka, f, *oral);
                let c3 = three_cpt_conc_g::<f64>(dose, t, cl, v, q, v2, q3, v3, ka, f, *oral);
                assert!(
                    c1.is_nan() && c2.is_nan() && c3.is_nan(),
                    "{at}: 1/2/3-cpt kernels gave {c1}, {c2}, {c3}"
                );
                // And a `NaN` jet, not a zero one (see the static-recording test).
                let d = |x: f64, i: usize| Dual1::<2>::var(x, i);
                let k = Dual1::<2>::constant;
                let jets = [
                    one_cpt_conc_g(dose, k(t), d(cl, 0), d(v, 1), k(ka), k(f), *oral),
                    two_cpt_conc_g(
                        dose,
                        k(t),
                        d(cl, 0),
                        d(v, 1),
                        k(q),
                        k(v2),
                        k(ka),
                        k(f),
                        *oral,
                    ),
                    three_cpt_conc_g(
                        dose,
                        k(t),
                        d(cl, 0),
                        d(v, 1),
                        k(q),
                        k(v2),
                        k(q3),
                        k(v3),
                        k(ka),
                        k(f),
                        *oral,
                    ),
                ];
                for (n, j) in jets.iter().enumerate() {
                    assert!(
                        j.grad.iter().all(|x| x.is_nan()),
                        "{at}: {}-cpt jet {:?}",
                        n + 1,
                        j.grad
                    );
                }
            }
            // Before the arrival nothing has been given: 0, not `NaN`.
            let before = one_cpt_conc_g::<f64>(dose, -1.0, cl, v, ka, f, *oral)
                + two_cpt_conc_g::<f64>(dose, -1.0, cl, v, q, v2, ka, f, *oral)
                + three_cpt_conc_g::<f64>(dose, -1.0, cl, v, q, v2, q3, v3, ka, f, *oral);
            assert_eq!(before, 0.0, "{label}: before the arrival");
        }
    }
    // The public f64 entry points the review named, and the state helpers.
    let infusion = mk(100.0, false, 0.0);
    let infusion_ss = mk(100.0, true, 12.0);
    let bolus = mk(0.0, false, 0.0);
    for (cl, v) in [(nan, 80.0), (5.0, nan)] {
        for t in [0.0, 1.0, 10.0] {
            assert!(crate::pk::two_cpt_infusion(&infusion, t, cl, v, q, v2).is_nan());
            assert!(crate::pk::two_cpt_infusion_ss(&infusion_ss, t, cl, v, q, v2).is_nan());
            assert!(crate::pk::two_cpt_predict(&infusion, t, cl, v, q, v2, None, None).is_nan());
            assert!(crate::pk::one_cpt_predict(&bolus, t, cl, v, None, None).is_nan());
        }
        assert!(
            crate::pk::two_compartment::two_cpt_iv_peripheral(&bolus, 1.0, cl, v, q, v2).is_nan()
        );
        assert!(crate::pk::three_compartment::three_cpt_iv_peripherals(
            &bolus, 1.0, cl, v, q, v2, q3, v3
        )
        .iter()
        .all(|x| x.is_nan()));
    }
}

/// The transit / inverse-Gaussian closed forms reach the same values as the six IV / oral
/// models through their own kernels: since #2221 (#2134) a `NaN` `CL` or `V` serves `NaN`
/// there, and a non-positive or infinite one the non-physical penalty, 0. #2104's override
/// stays gated to the six IV / oral models (`cl_v_rule_governs`), since #2134 owns these
/// kernels' handling (their walk, `nan_onset`, and their `NaN` jets) — so no value here can
/// tell the gate apart; this pins that the two conventions agree.
#[test]
fn transit_and_ig_agree_with_the_rule_through_their_own_kernels() {
    let doses = [DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)];
    for model in [
        PkModel::OneCptTransit,
        PkModel::TwoCptTransit,
        PkModel::OneCptIg,
        PkModel::TwoCptIg,
    ] {
        for (label, idx) in [("CL", PK_IDX_CL), ("V", PK_IDX_V)] {
            let mut pk = crate::types::PkParams::default();
            pk.values[PK_IDX_CL] = 5.0;
            pk.values[PK_IDX_V] = 80.0;
            pk.values[PK_IDX_Q] = 3.0;
            pk.values[PK_IDX_V2] = 40.0;
            pk.values[crate::types::PK_IDX_F] = 1.0;
            pk.values[crate::types::PK_IDX_N] = 3.0;
            pk.values[crate::types::PK_IDX_MTT] = 2.0;
            pk.values[crate::types::PK_IDX_MAT] = 2.0;
            pk.values[crate::types::PK_IDX_CV2] = 0.3;
            let live = crate::pk::predict_concentration(model, &doses, 4.0, &pk);
            assert!(live > 0.0, "{model:?}: fixture is live ({live})");
            for x in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
                pk.values[idx] = x;
                let c = crate::pk::predict_concentration(model, &doses, 4.0, &pk);
                if x.is_nan() {
                    assert!(c.is_nan(), "{model:?} {label}={x}: {c}, not NaN");
                } else {
                    assert_eq!(c, 0.0, "{model:?} {label}={x}");
                }
            }
        }
    }
}
