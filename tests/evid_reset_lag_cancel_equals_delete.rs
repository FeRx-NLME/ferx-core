//! #1587: **a dose recorded before an EVID=3/4 reset is cancelled — which is the same as
//! deleting it — on every absorption form, value and derivative.**
//!
//! `evid_reset_lag_nonmem_anchor.rs` pins the rule against NONMEM on a depot bolus and a
//! lagged infusion. This file carries it to every other way a lagged dose reaches the
//! state, with an oracle that needs no second engine: after the reset, a subject whose
//! pending lagged dose was recorded before the reset must read exactly like the same
//! subject with that dose deleted — the prediction, and every `∂f/∂η` / `∂f/∂θ` of the dual
//! walk, since a cancelled dose has no derivative either. Every form puts the cancelled
//! dose's lagged arrival *after* the reset (asserted), and two live doses around it, the
//! second landing on residual drug from the first, so neither side is degenerate.
//!
//! Per form, four assertions:
//! 1. **cancel ≡ delete** on the production value (`predict`), the with-states engine and
//!    the dual walk's value;
//! 2. **the derivatives agree** between the two subjects on the dual walk — first order and
//!    both second-order blocks;
//! 3. **Dual2 vs FD**: the dual `∂f/∂η` of the cancelling subject against central FD of
//!    the production predictor;
//! 4. **the straddle**: the same subject with the reset moved *before* the dose record
//!    (the dose now live, its arrival unchanged) must differ from the deleted twin, so
//!    (1) cannot hold because the dose was lost for some other reason.
//!
//! Every form runs in three variants. **Flat** and **TV WT** (a time-varying `WT` on `CL`)
//! route the analytic forms to superposition and to the event-driven dual walk
//! (`sens/propagate.rs`) respectively; the ODE forms always take the event-driven dual walk
//! (a lag is estimated). In both, the cancelled dose's would-be arrival lands on an empty
//! system — every live dose arrives later — so a leftover jump or saltation there would
//! multiply a zero state. **Residual** removes that: a per-record lag factor `LAGF` (0.2
//! from the reset on) makes the first post-reset dose arrive *before* the cancelled one, so
//! drug is present when the cancelled arrival would have landed (asserted). **At obs** is
//! `Residual` with `η_LAG = 0`, so the cancelled arrival lands *exactly* on the 11.0
//! observation — the one place an observation-boundary correction for that arrival can act.
//! A third live dose (record 11.15) arrives, in `Residual` / `At obs`, *inside* the cancelled
//! dose's would-be absorption or infusion window, where the dual walk's saltation velocities
//! would carry the cancelled window if it leaked — a second-order effect, which is why (2)
//! compares the second-order blocks too.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::compute_predictions_with_states;
use ferx_core::sens::ode_provider::ode_subject_sensitivities;
use ferx_core::sens::provider::{subject_sensitivities, SubjectSens};
use ferx_core::types::Subject;
use ferx_core::{predict, read_nonmem_csv, CompiledModel, Population};

/// One absorption form: its model, whether it runs on the analytic engine, and the dose
/// row's `CMT` / `RATE`.
struct Form {
    name: &'static str,
    model: &'static str,
    analytic: bool,
    cmt: u32,
    rate: f64,
}

const PARAMS: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.6, 0.005, 20.0)
  theta TVLAG(2.0, 0.01, 10.0)
  theta TVDUR(1.5, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[covariates]
  WT continuous
  LAGF continuous
"#;

const IND: &str = r#"
[individual_parameters]
  CL  = TVCL * (WT/70)^0.75 * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  DUR = TVDUR
  LAG = TVLAG * exp(ETA_LAG) * LAGF
  NTR = 2
"#;

const TAIL: &str = r#"
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-12
"#;

const FORMS: &[Form] = &[
    Form {
        name: "ODE depot bolus",
        model: "  ALAG1 = LAG\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA*depot\n  d/dt(central) = KA*depot - CL/V*central\n[scaling]\n  y = central / V\n",
        analytic: false,
        cmt: 1,
        rate: 0.0,
    },
    Form {
        name: "ODE first_order forcing",
        model: "  ALAG1 = LAG\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = first_order(ka=KA) - CL/V*central\n[scaling]\n  y = central / V\n",
        analytic: false,
        cmt: 1,
        rate: 0.0,
    },
    Form {
        name: "ODE zero_order forcing",
        model: "  ALAG1 = LAG\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = zero_order(dur=DUR) - CL/V*central\n[scaling]\n  y = central / V\n",
        analytic: false,
        cmt: 1,
        rate: 0.0,
    },
    Form {
        name: "ODE transit forcing",
        model: "  ALAG1 = LAG\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = transit(n=NTR, mtt=DUR) - CL/V*central\n[scaling]\n  y = central / V\n",
        analytic: false,
        cmt: 1,
        rate: 0.0,
    },
    Form {
        name: "ODE first_order route lag",
        model: "[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central\n[scaling]\n  y = central / V\n",
        analytic: false,
        cmt: 1,
        rate: 0.0,
    },
    Form {
        name: "ODE lagged infusion",
        model: "  ALAG1 = LAG\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -CL/V*central\n[scaling]\n  y = central / V\n",
        analytic: false,
        cmt: 1,
        rate: 125.0,
    },
    Form {
        name: "analytic oral",
        model: "[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAG)\n",
        analytic: true,
        cmt: 1,
        rate: 0.0,
    },
    Form {
        name: "analytic lagged infusion",
        model: "[structural_model]\n  pk one_cpt_iv(cl=CL, v=V, alag=LAG)\n",
        analytic: true,
        cmt: 1,
        rate: 125.0,
    },
];

fn model(form: &Form) -> CompiledModel {
    let src = format!("{PARAMS}{IND}{}{TAIL}", form.model);
    parse_full_model(&src)
        .unwrap_or_else(|e| panic!("{}: model parses: {e:?}", form.name))
        .model
}

const OBS: &[f64] = &[11.0, 11.5, 12.5, 14.0, 16.0, 20.0];
/// The pending dose (record 9, lag 2.1 ⇒ arrival ≈ 11.1; 2.0 ⇒ 11.0 in `At obs`) and the
/// three live ones. An infusion is 0.8 h (`RATE = 125`), so the cancelled one's record-time
/// window `[9, 9.8]` ends before the reset and the reader does not shift it.
const CANCELLED: f64 = 9.0;
// 11.15, not 11.1: at `η_LAG = 0` a record at 11.1 arrives exactly on the 11.5 read, a kink
// that FD cannot difference across.
const LIVE: &[f64] = &[10.5, 11.15, 13.0];

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Flat,
    TvWt,
    Residual,
    AtObs,
}

/// `η = [ETA_CL, ETA_LAG]`; `At obs` zeroes `η_LAG` so the lag is exactly `TVLAG = 2`.
fn eta(v: Variant) -> [f64; 2] {
    if v == Variant::AtObs {
        [0.1, 0.0]
    } else {
        [0.1, 0.05]
    }
}

/// A one-subject population: doses at `doses`, one EVID=3 row at `reset`, the observations
/// `OBS`. Every variant but `Flat` changes `WT` on every row, so the covariate is live across
/// each arrival; `Residual` and `AtObs` also set `LAGF = 0.2` on every row from t = 10 on.
fn subject(form: &Form, doses: &[f64], reset: f64, v: Variant) -> Population {
    let wt = |t: f64| {
        if v == Variant::Flat {
            70.0
        } else {
            60.0 + 2.0 * t
        }
    };
    let lagf = |t: f64| {
        if matches!(v, Variant::Residual | Variant::AtObs) && t >= 10.0 {
            0.2
        } else {
            1.0
        }
    };
    // (time, row order within a time, ID,TIME,DV,AMT,EVID,CMT,MDV,RATE)
    let mut rows: Vec<(f64, u8, String)> = vec![(0.0, 0, "0,.,0,0,2,1,0".to_string())];
    for &d in doses {
        rows.push((d, 1, format!("{d},.,100,1,{},1,{}", form.cmt, form.rate)));
    }
    rows.push((reset, 0, format!("{reset},.,0,3,1,1,0")));
    for &t in OBS {
        rows.push((t, 2, format!("{t},1,0,0,1,0,0")));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,RATE,WT,LAGF\n");
    for (t, _, r) in rows {
        csv.push_str(&format!("1,{r},{},{}\n", wt(t), lagf(t)));
    }
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferx_1587_cancel_delete_{}_{}.csv",
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, csv).expect("write temp csv");
    let pop = read_nonmem_csv(&path, None, None).expect("csv loads");
    let _ = std::fs::remove_file(&path);
    assert_eq!(pop.subjects.len(), 1);
    pop
}

fn sens(m: &CompiledModel, form: &Form, s: &Subject, eta: &[f64]) -> SubjectSens {
    let theta = &m.default_params.theta;
    let r = if form.analytic {
        subject_sensitivities(m, s, theta, eta)
    } else {
        ode_subject_sensitivities(m, s, theta, eta)
    };
    r.unwrap_or_else(|| {
        panic!(
            "{}: routed to FD, the dual walk is not exercised",
            form.name
        )
    })
}

fn production(m: &CompiledModel, s: &Subject, eta: &[f64]) -> Vec<f64> {
    compute_predictions_with_states(m, s, &m.default_params.theta, eta).0
}

fn via_predict(m: &CompiledModel, pop: &Population) -> Vec<f64> {
    let mut p = m.default_params.clone();
    // `predict` evaluates at η = 0; the value checks read it there.
    p.theta = m.default_params.theta.clone();
    predict(m, pop, &p)
        .expect("predict")
        .into_iter()
        .map(|r| r.pred)
        .collect()
}

/// `|a − b| ≤ tol · (1 + |b|)`, every element, both finite.
fn close(label: &str, a: &[f64], b: &[f64], tol: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "{label}: length");
    let mut worst = 0.0_f64;
    for (j, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite(), "{label}[{j}]: {x} vs {y}");
        let e = (x - y).abs() / (1.0 + y.abs());
        assert!(
            e <= tol,
            "{label}[{j}]: {x} vs {y} (err {e:.3e} > {tol:.0e})"
        );
        worst = worst.max(e);
    }
    worst
}

// Measured at the fix, over 8 forms × 4 variants (printed per case), err = |a−b|/(1+|b|):
// cancel vs delete is bit-identical on `flat` / `TV WT` (the cancelled arrival lands on an
// empty system, so the extra break integrates zeros exactly); on `residual` / `at obs` it is
// ≤ 1.9e-14 on values and ≤ 1.2e-11 on derivatives (transit, a second-order block), because
// the cancelled arrival's break changes the integrator's steps. SAME_TOL is ~85× the worst.
// Dual vs FD ≤ 1.8e-9 (h = 1e-5): FD_TOL is ~55×. Straddle ≥ 3.26 against a bound of 1.0.
// The defect guarded — a kept dose — moves f by ≥ 3.
const SAME_TOL: f64 = 1e-9;
const FD_TOL: f64 = 1e-7;
const STRADDLE_MIN: f64 = 1.0;

fn check(form: &Form, v: Variant) {
    let label = format!(
        "{} ({})",
        form.name,
        match v {
            Variant::Flat => "flat",
            Variant::TvWt => "TV WT",
            Variant::Residual => "residual at the cancelled arrival",
            Variant::AtObs => "cancelled arrival on an observation",
        }
    );
    let e = eta(v);
    let m = model(form);
    let mut all = vec![CANCELLED];
    all.extend_from_slice(LIVE);
    let cancel = subject(form, &all, 10.0, v);
    let delete = subject(form, LIVE, 10.0, v);
    let live = subject(form, &all, 8.5, v);
    let (sc, sd) = (&cancel.subjects[0], &delete.subjects[0]);

    // The fixture: the cancelled dose's record precedes the reset and its arrival follows it
    // (its `LAGF` is 1 in every variant: its record is before t = 10).
    let lag = m.default_params.theta[3] * e[1].exp();
    let arrival = CANCELLED + lag;
    assert!(
        CANCELLED < 10.0 && arrival > 10.0,
        "{label}: arrival {arrival}"
    );
    if v == Variant::AtObs {
        assert_eq!(
            arrival, OBS[0],
            "{label}: the cancelled arrival must sit on an observation"
        );
    }
    if matches!(v, Variant::Residual | Variant::AtObs) {
        // The first live dose arrives (10.5 + 0.2·lag ≈ 10.92) before the cancelled one, so
        // drug is present there: the 11.0 read, just before `arrival`, is non-zero (least:
        // transit, whose onset is smooth, 3.0e-3).
        let first_live = LIVE[0] + 0.2 * lag;
        assert!(
            first_live < OBS[0] && OBS[0] <= arrival,
            "{label}: {first_live} / {arrival}"
        );
        // The third live dose arrives inside the cancelled dose's window (it opens at
        // `arrival`; the shortest, the 0.8 h infusion, closes at `arrival + 0.8`).
        let inside = LIVE[1] + 0.2 * lag;
        assert!(
            arrival < inside && inside < arrival + 0.8,
            "{label}: {inside}"
        );
        let f = production(&m, sd, &e);
        assert!(
            f[0] > 1e-3,
            "{label}: no residual drug at the cancelled arrival: {f:?}"
        );
    }

    // (1) cancel ≡ delete on every value path.
    let w_pred = close(
        &format!("{label}: predict"),
        &via_predict(&m, &cancel),
        &via_predict(&m, &delete),
        SAME_TOL,
    );
    let w_states = close(
        &format!("{label}: with-states"),
        &production(&m, sc, &e),
        &production(&m, sd, &e),
        SAME_TOL,
    );
    let (a, b) = (sens(&m, form, sc, &e), sens(&m, form, sd, &e));
    let fa: Vec<f64> = a.obs.iter().map(|o| o.f).collect();
    let fb: Vec<f64> = b.obs.iter().map(|o| o.f).collect();
    let w_dual = close(&format!("{label}: dual value"), &fa, &fb, SAME_TOL);
    assert!(
        fb.iter().any(|&v| v > 0.05),
        "{label}: the live doses must be seen: {fb:?}"
    );

    // (2) the derivatives of a cancelled dose are those of a deleted one.
    let mut w_grad = 0.0_f64;
    for (j, (x, y)) in a.obs.iter().zip(&b.obs).enumerate() {
        w_grad = w_grad.max(close(
            &format!("{label}: ∂f/∂η obs {j}"),
            &x.df_deta,
            &y.df_deta,
            SAME_TOL,
        ));
        w_grad = w_grad.max(close(
            &format!("{label}: ∂f/∂θ obs {j}"),
            &x.df_dtheta,
            &y.df_dtheta,
            SAME_TOL,
        ));
        w_grad = w_grad.max(close(
            &format!("{label}: ∂²f/∂η² obs {j}"),
            &x.d2f_deta2,
            &y.d2f_deta2,
            SAME_TOL,
        ));
        w_grad = w_grad.max(close(
            &format!("{label}: ∂²f/∂η∂θ obs {j}"),
            &x.d2f_deta_dtheta,
            &y.d2f_deta_dtheta,
            SAME_TOL,
        ));
    }
    // The lag axis is live on the live doses, so (2) is not comparing zeros.
    assert!(
        b.obs.iter().any(|o| o.df_deta[1].abs() > 1e-3),
        "{label}: ∂f/∂η_LAG must be live"
    );

    // (3) Dual2 vs central FD of the production predictor, on the cancelling subject.
    let h = 1e-5;
    let mut w_fd = 0.0_f64;
    for k in 0..e.len() {
        let (mut ep, mut em) = (e.to_vec(), e.to_vec());
        ep[k] += h;
        em[k] -= h;
        let (fp, fm) = (production(&m, sc, &ep), production(&m, sc, &em));
        let fd: Vec<f64> = fp
            .iter()
            .zip(&fm)
            .map(|(p, q)| (p - q) / (2.0 * h))
            .collect();
        let dual: Vec<f64> = a.obs.iter().map(|o| o.df_deta[k]).collect();
        w_fd = w_fd.max(close(
            &format!("{label}: dual vs FD ∂f/∂η[{k}]"),
            &dual,
            &fd,
            FD_TOL,
        ));
    }

    // (4) the straddle: the dose live (reset before its record), everything else equal.
    let fl = production(&m, &live.subjects[0], &e);
    let fdel = production(&m, sd, &e);
    let gap = fl
        .iter()
        .zip(&fdel)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        fl.iter().all(|v| v.is_finite()) && gap > STRADDLE_MIN,
        "{label}: with the reset before the record the dose must be live (gap {gap})"
    );
    println!(
        "#1587 {label}: cancel-vs-delete predict {w_pred:.2e} states {w_states:.2e} dual {w_dual:.2e} \
         grad {w_grad:.2e}; dual-vs-FD {w_fd:.2e}; straddle {gap:.3}"
    );
}

#[test]
fn a_dose_recorded_before_the_reset_is_deleted_on_every_absorption_form() {
    for form in FORMS {
        for v in [
            Variant::Flat,
            Variant::TvWt,
            Variant::Residual,
            Variant::AtObs,
        ] {
            check(form, v);
        }
    }
}
