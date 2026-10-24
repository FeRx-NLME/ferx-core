#![allow(unexpected_cfgs)]

use super::*;

/// The objective-stall predicate for a test that exercises an *exact* objective: the
/// historical `enable_stall = false`, spelled as the predicate the cores now take. A test
/// that wants the stop on passes its own closure.
fn never_stall(_: &[f64]) -> bool {
    false
}
use std::collections::HashMap;
#[cfg(profiling_allocations)]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(profiling_allocations)]
struct CountingAllocator;

#[cfg(profiling_allocations)]
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(profiling_allocations)]
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(profiling_allocations)]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
#[cfg(profiling_allocations)]
static TEST_ALLOCATOR: CountingAllocator = CountingAllocator;

/// An endpoint-only mixed-effects CTMM (#759): no `[structural_model]`, so no Gaussian
/// data term and no PK provider — the inner objective is `½(η'Ω⁻¹η + log|Ω|) + D_ctmm(η)`.
#[cfg(feature = "markov")]
mod ctmm_inner {
    use crate::types::{ObsRecord, Subject};

    const MIXED_CTMM: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  omega ETA_Q ~ 0.1

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + ETA_Q)
  transition asleep -> awake  = exp(LQ10)
";

    fn subject() -> Subject {
        Subject {
            id: "1".into(),
            obs_records: [(0.0, 0), (1.0, 0), (2.3, 1), (3.1, 1), (5.0, 0)]
                .iter()
                .map(|&(time, state)| ObsRecord::DiscreteState {
                    time,
                    raw_time: time,
                    state,
                    cmt: 5,
                })
                .collect(),
            ..Default::default()
        }
    }

    /// The route must actually change. Before #759 every subject with `obs_records`
    /// declined the analytic inner gradient outright, so this predicate was `false` and
    /// the EBE search finite-differenced the whole objective.
    #[test]
    fn endpoint_only_ctmm_reports_and_takes_the_analytic_inner_route() {
        let model = crate::parser::model_parser::parse_model_string(MIXED_CTMM).unwrap();
        // Model-level: what `build_info::gradient_method_inner` reports.
        assert!(
            super::super::analytic_inner_grad_supported_model(&model),
            "an endpoint-only CTMM must report an analytic inner gradient"
        );
        // Subject-level: what `find_ebe` actually runs.
        assert!(
            super::super::analytic_inner_grad_supported(&model, &subject()),
            "a CTMM subject must now take the analytic inner route"
        );
    }

    /// An endpoint-only CTMM whose generator has **no** dual-evaluable program — here by
    /// pushing the `(θ, η)` width past `MAX_CTMM_AXES` — must stay on FD, on *both* the
    /// model-level predicate `build_info` reports and the per-subject gate `find_ebe`
    /// runs. The two must agree, or the fit reports a gradient method it does not use.
    ///
    /// This pins the ordering inside `analytic_inner_grad_supported`: the survival-record
    /// guard runs first and rejects a program-less CTMM subject, so it never reaches the
    /// `obs_times.is_empty()` branch (which would otherwise wave it through on the
    /// strength of having no Gaussian block). Swap those two and this test fails.
    #[test]
    fn program_less_endpoint_only_ctmm_stays_fd() {
        let n_th = 25; // + 1 η = 26 axes > MAX_CTMM_AXES (24)
        let mut src = String::from("[parameters]\n");
        for i in 1..=n_th {
            src += &format!("  theta T{i}(0.1, -6.0, 3.0)\n");
        }
        src += "  omega ETA_Q ~ 0.1\n[markov_model]\n  type   = ctmm\n  cmt    = 5\n  \
                    states = [awake=0, asleep=1]\n  transition awake  -> asleep = exp(ETA_Q + ";
        src += &(1..=n_th)
            .map(|i| format!("T{i}"))
            .collect::<Vec<_>>()
            .join(" + ");
        src += ")\n  transition asleep -> awake  = exp(T1)\n";

        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        assert_eq!(
            model.n_theta + model.n_eta,
            26,
            "fixture must exceed the cap"
        );
        assert!(
            !super::super::analytic_inner_grad_supported_model(&model),
            "a CTMM past the dual dispatch cap must report FD"
        );
        assert!(
            !super::super::analytic_inner_grad_supported(&model, &subject()),
            "…and must actually take FD, not fall through the no-Gaussian-block branch"
        );
    }

    /// The whole inner gradient — prior + CTMM data term — against a central difference
    /// of `individual_nll`, the exact objective the EBE search minimizes. This is the
    /// contract that matters: the two must agree, or BFGS converges on one function
    /// while descending another.
    #[test]
    fn analytic_inner_gradient_matches_fd_of_individual_nll() {
        let model = crate::parser::model_parser::parse_model_string(MIXED_CTMM).unwrap();
        let subject = subject();
        let params = &model.default_params;
        let theta = &params.theta;

        for &eta0 in &[-0.4_f64, 0.0, 0.55] {
            let err_keys = model.error_spec.obs_keys(&subject);
            let g = super::super::analytic_eta_nll_gradient_with_schedule(
                &model,
                &subject,
                theta,
                &[eta0],
                &params.omega,
                &params.sigma.values,
                model.residual_cov_at(params),
                None,
                None,
                err_keys.as_ref(),
                &mut Vec::new(),
                &mut nalgebra::DVector::zeros(model.n_eta),
                &mut nalgebra::DVector::zeros(model.n_eta),
                None,
            )
            .expect("endpoint-only CTMM is in analytic scope");

            let nll = |e: f64| {
                crate::stats::likelihood::individual_nll(
                    &model,
                    &subject,
                    theta,
                    &[e],
                    &params.omega,
                    &params.sigma.values,
                )
            };
            let h = 1e-6;
            let fd = (nll(eta0 + h) - nll(eta0 - h)) / (2.0 * h);
            assert!(
                (g[0] - fd).abs() < 1e-6,
                "η = {eta0}: analytic {}, FD {fd}",
                g[0]
            );
        }
    }
}

/// The M3 censored coefficient `∂/∂f[−logΦ((y−f)/√v)]` must equal a central
/// finite difference of that data term — across additive (`dv_df = 0`) and
/// f-dependent (`dv_df ≠ 0`, e.g. proportional/combined) variance, and across
/// the regimes `f < LLOQ`, `f ≈ LLOQ`, and `f ≫ LLOQ` (deep tail, where the
/// inverse Mills ratio's log-domain evaluation matters).
#[test]
fn m3_censored_dterm_df_matches_fd() {
    // Per-row censored data term −logΦ(z), z = (y−f)/√v(f), with v(f) a
    // generic affine-in-f² residual variance: v = sig_add² + (sig_prop·f)².
    let term = |y: f64, f: f64, sig_add: f64, sig_prop: f64| -> f64 {
        let v = sig_add * sig_add + sig_prop * sig_prop * f * f;
        let z = (y - f) / v.sqrt();
        -crate::stats::special::log_normal_cdf(z)
    };
    let lloq = 1.0_f64;
    let cases = [
        // (f, sig_add, sig_prop)
        (0.6, 0.2, 0.0),  // additive, f below LLOQ
        (1.0, 0.2, 0.0),  // additive, f at LLOQ
        (0.8, 0.0, 0.25), // proportional, dv_df ≠ 0
        (0.7, 0.15, 0.2), // combined, dv_df ≠ 0
        (3.0, 0.2, 0.0),  // f ≫ LLOQ: deep tail (Φ(z)→0)
    ];
    for (f, sig_add, sig_prop) in cases {
        let v = sig_add * sig_add + sig_prop * sig_prop * f * f;
        let dv_df = 2.0 * sig_prop * sig_prop * f; // ∂v/∂f
        let analytic = m3_censored_dterm_df(lloq, f, v, dv_df, 1);
        // `normal_cdf` is a rational approximation (~1.5e-7 abs error); a tiny
        // FD step amplifies that noise (noise/h), so use a moderate step where
        // truncation and approximation error both sit well under the band.
        let h = 1e-3;
        let fd = (term(lloq, f + h, sig_add, sig_prop) - term(lloq, f - h, sig_add, sig_prop))
            / (2.0 * h);
        assert!(
            (analytic - fd).abs() < 1e-3 * (1.0 + fd.abs()),
            "f={f}, sig_add={sig_add}, sig_prop={sig_prop}: analytic {analytic} vs FD {fd}"
        );
    }
}

#[test]
fn find_ebe_uses_fd_h_matrix_when_inner_gradient_forced_fd() {
    use crate::parser::model_parser::parse_model_string;

    let mut model = parse_model_string(
        r#"
[parameters]
  theta TVCL(0.15, 0.01, 10.0)
  theta TVV(5.0, 0.1, 100.0)
  theta TVIMAX(-0.3, -10.0, 10.0)
  theta TVTI50(100.0, 1.0, 700.0)
  theta TVHILL(3.0, 0.1, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.01
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  IMAX = TVIMAX
  TI50 = TVTI50
  HILL = TVHILL

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL * exp(IMAX * TIME^HILL / (TI50^HILL + TIME^HILL)) / V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  gradient = fd
"#,
    )
    .expect("parse");
    model.gradient_method = GradientMethod::Fd;

    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 200.0, 1, 9600.0, false, 0.0)],
        obs_times: vec![20.0],
        obs_raw_times: Vec::new(),
        observations: vec![12.0],
        obs_cmts: vec![1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };

    let result = find_ebe(
        &model,
        &subject,
        &model.default_params,
        50,
        1e-5,
        None,
        None,
        0,
    );

    assert!(
        result.h_matrix.iter().all(|v| v.is_finite()),
        "forced-FD inner route must not consume a non-finite analytic h_matrix: {:?}",
        result.h_matrix
    );
}

/// End-to-end: the analytic M3 inner η-gradient must match a central finite
/// difference of the inner objective (`individual_nll_into_with_schedule`,
/// which carries the `−2·logΦ(z)` censored term) on the real warfarin BLOQ
/// model + data — exercising the full wiring (provider, cens lookup, coef
/// dispatch), not just the isolated coefficient.
#[test]
fn analytic_inner_gradient_m3_matches_fd_on_warfarin_bloq() {
    use std::cell::RefCell;
    use std::path::Path;
    let model =
        crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin_bloq.ferx"))
            .expect("warfarin BLOQ model parses");
    assert!(
        matches!(model.bloq_method, crate::types::BloqMethod::M3),
        "model must be M3"
    );
    let pop =
        crate::io::datareader::read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
            .expect("warfarin BLOQ data loads");
    let subject = pop
        .subjects
        .iter()
        .find(|s| s.cens.iter().any(|&c| c != 0))
        .expect("at least one subject with a censored row");

    let theta = &model.default_params.theta;
    let omega = &model.default_params.omega;
    let sigma = &model.default_params.sigma.values;
    let eta = vec![0.12, -0.05, 0.2];

    let analytic = analytic_eta_nll_gradient(&model, subject, theta, &eta, omega, sigma)
        .expect("analytic M3 inner gradient must be supported");

    let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(subject));
    let obj = |e: &[f64]| -> f64 {
        let mut s = scratch.borrow_mut();
        individual_nll_into_with_schedule(
            &model,
            subject,
            theta,
            e,
            omega,
            sigma,
            &model.residual_correlations,
            &mut s,
            None,
        )
    };
    let fd = gradient_fd(&obj, &eta, model.n_eta);

    for k in 0..model.n_eta {
        assert!(
            (analytic[k] - fd[k]).abs() < 1e-4 * (1.0 + fd[k].abs()),
            "η[{k}]: analytic {} vs FD {}",
            analytic[k],
            fd[k]
        );
    }
}

/// #958, as reported: a two-state target-binding ODE with a **parameter-
/// dependent initial condition** (`init(RTOT) = RBASE`) and a `sqrt` binding
/// quadratic in the RHS, observed at two endpoints under proportional error.
/// The analytic ODE inner η-gradient must match a central FD of `individual_nll`
/// at a non-zero η that pushes the drug endpoint's late sample into the
/// variance floor. This is the exact scenario the issue bisected to; the raw
/// four-cell design isolates it to `init(θ)` + `sqrt`, but the mechanism is the
/// floor-unaware `∂R/∂f`, so the reproduction is the analytic-vs-FD inner
/// gradient here (the prediction sensitivities themselves were never wrong).
#[test]
fn analytic_inner_gradient_ode_init_theta_sqrt_matches_fd() {
    use std::cell::RefCell;
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta KEL(0.05,0.001,10)\n  theta VC(5,0.5,100)\n  theta RBASE(20,1,400)\n  theta KINT(0.2,0.001,20)\n  theta KDEG(0.02,0.0001,5)\n  theta KM(10,0.1,1000)\n  omega IIV_KEL   ~ 0.09\n  omega IIV_RBASE ~ 0.09\n  sigma PROP_DRUG ~ 0.15 (sd)\n  sigma PROP_TGT  ~ 0.15 (sd)\n[individual_parameters]\n  KEL   = KEL * exp(IIV_KEL)\n  RBASE = RBASE * exp(IIV_RBASE)\n  VC    = VC\n  KINT  = KINT\n  KDEG  = KDEG\n  KM    = KM\n  KSYN  = RBASE * KDEG\n[structural_model]\n  ode(states=[CENT, RTOT])\n[odes]\n  init(RTOT) = RBASE\n  ct = CENT / VC\n  bb = ct - RTOT - KM\n  cf = 0.5 * (bb + sqrt(bb*bb + 4*KM*ct))\n  fb = cf / (KM + cf)\n  d/dt(CENT) = -KEL * cf * VC - KINT * fb * RTOT * VC\n  d/dt(RTOT) =  KSYN - KDEG * RTOT - KINT * fb * RTOT\n[scaling]\n  y[CMT=1] = CENT / VC\n  y[CMT=2] = RTOT\n[error_model]\n  CMT=1: DV ~ proportional(PROP_DRUG)\n  CMT=2: DV ~ proportional(PROP_TGT)\n[fit_options]\n  method     = focei\n  ode_reltol = 1e-9\n  ode_abstol = 1e-9\n",
    )
    .expect("parse binding-quadratic ODE with parameter-dependent init");

    // Drug (CMT 1) at 4 times incl. a far-tail sample where drug ~ 0 (floored
    // proportional variance), then total target (CMT 2) at the same times.
    let tms = [1.0, 7.0, 28.0, 120.0];
    let mut obs_times = Vec::new();
    let mut obs_cmts = Vec::new();
    for &t in &tms {
        obs_times.push(t);
        obs_cmts.push(1usize);
    }
    for &t in &tms {
        obs_times.push(t);
        obs_cmts.push(2usize);
    }
    let n = obs_times.len();
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 300.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        // Rough noise-free-ish values; exact magnitudes are immaterial — the
        // analytic gradient is checked against FD of the objective on the SAME
        // data, so the assertion is a self-consistency (gradient-path) check.
        observations: vec![
            30.0, 5.0, 0.5, 0.0, // drug
            16.0, 10.0, 6.0, 15.0, // target
        ],
        obs_cmts,
        covariates: HashMap::new(),
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
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };

    let theta = &model.default_params.theta;
    let omega = &model.default_params.omega;
    let sigma = &model.default_params.sigma.values;
    let eta = vec![0.5, -0.15];

    // Precondition: the drug tail sample really does hit the variance floor.
    let preds = crate::pk::compute_predictions_with_tv(&model, &subject, theta, &eta);
    assert!(
        model.residual_variance_at(1, preds[3], sigma) <= 1e-12,
        "drug tail prediction {} must drive its proportional variance to the floor",
        preds[3]
    );

    let analytic = analytic_eta_nll_gradient(&model, &subject, theta, &eta, omega, sigma)
        .expect("analytic ODE inner gradient must be supported");
    let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(&subject));
    let obj = |e: &[f64]| -> f64 {
        let mut s = scratch.borrow_mut();
        individual_nll_into_with_schedule(
            &model,
            &subject,
            theta,
            e,
            omega,
            sigma,
            &model.residual_correlations,
            &mut s,
            None,
        )
    };
    let fd = gradient_fd(&obj, &eta, model.n_eta);
    for k in 0..model.n_eta {
        assert!(
            (analytic[k] - fd[k]).abs() < 1e-3 * (1.0 + fd[k].abs()),
            "η[{k}]: analytic {} vs FD {}",
            analytic[k],
            fd[k]
        );
    }
}

/// Closed-form `iiv_on_ruv` + M3 BLOQ (#4c): the analytic non-IOV inner
/// η-gradient must match central FD of `individual_nll`, exercising the censored
/// `η_ruv` data column `h·z` and the `exp(2·η_ruv)` variance scaling on the
/// censored rows (which previously forced FD).
#[test]
fn analytic_inner_gradient_iiv_on_ruv_m3_matches_fd() {
    use std::cell::RefCell;
    use std::collections::HashMap;
    let mut model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n",
        )
        .expect("parse closed-form iiv_on_ruv");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert_eq!(model.residual_error_eta, Some(3));

    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        obs_raw_times: Vec::new(),
        // The last two rows are below the LLOQ = 2.0 (carried in `observations`).
        observations: vec![8.0, 7.0, 5.0, 3.0, 2.0, 2.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0, 0, 1, 1],
        occasions: vec![1; 6],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };

    let theta = &model.default_params.theta;
    let omega = &model.default_params.omega;
    let sigma = &model.default_params.sigma.values;
    let eta = vec![0.12, -0.05, 0.2, 0.15]; // non-zero η_ruv

    let analytic = analytic_eta_nll_gradient(&model, &subject, theta, &eta, omega, sigma)
        .expect("analytic closed-form M3 + iiv_on_ruv inner gradient");

    let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(&subject));
    let obj = |e: &[f64]| -> f64 {
        let mut s = scratch.borrow_mut();
        individual_nll_into_with_schedule(
            &model,
            &subject,
            theta,
            e,
            omega,
            sigma,
            &model.residual_correlations,
            &mut s,
            None,
        )
    };
    let fd = gradient_fd(&obj, &eta, model.n_eta);
    for k in 0..model.n_eta {
        assert!(
            (analytic[k] - fd[k]).abs() < 1e-4 * (1.0 + fd[k].abs()),
            "η[{k}]: analytic {} vs FD {}",
            analytic[k],
            fd[k]
        );
    }
}

/// ODE counterpart of [`analytic_inner_gradient_m3_matches_fd_on_warfarin_bloq`]:
/// the analytic M3 inner η-gradient produced via the **event-driven ODE
/// sensitivity walk** (not the closed-form provider) must match a central FD of
/// the inner objective on the warfarin BLOQ data — confirming non-IOV ODE+M3 is
/// served analytically on the inner loop (the censored `−logΦ` coefficient rides
/// the same provider-agnostic `apply_*_inner` path as the closed-form engine).
#[test]
fn analytic_inner_gradient_m3_matches_fd_on_warfarin_ode_bloq() {
    use std::cell::RefCell;
    use std::path::Path;
    let model =
        crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin_ode_bloq.ferx"))
            .expect("warfarin ODE BLOQ model parses");
    assert!(
        matches!(model.bloq_method, crate::types::BloqMethod::M3),
        "model must be M3"
    );
    assert!(
        model.is_ode_based(),
        "model must be on the ODE path for this probe"
    );
    let pop =
        crate::io::datareader::read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
            .expect("warfarin BLOQ data loads");
    let subject = pop
        .subjects
        .iter()
        .find(|s| s.cens.iter().any(|&c| c != 0))
        .expect("at least one subject with a censored row");

    let theta = &model.default_params.theta;
    let omega = &model.default_params.omega;
    let sigma = &model.default_params.sigma.values;
    let eta = vec![0.12, -0.05, 0.2];

    let analytic = analytic_eta_nll_gradient(&model, subject, theta, &eta, omega, sigma)
        .expect("analytic M3 inner gradient must be supported on ODE path");

    let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(subject));
    let obj = |e: &[f64]| -> f64 {
        let mut s = scratch.borrow_mut();
        individual_nll_into_with_schedule(
            &model,
            subject,
            theta,
            e,
            omega,
            sigma,
            &model.residual_correlations,
            &mut s,
            None,
        )
    };
    let fd = gradient_fd(&obj, &eta, model.n_eta);

    for k in 0..model.n_eta {
        assert!(
            (analytic[k] - fd[k]).abs() < 1e-4 * (1.0 + fd[k].abs()),
            "η[{k}]: analytic {} vs FD {}",
            analytic[k],
            fd[k]
        );
    }
}

/// **Non-IOV ODE** M3 BLOQ + `iiv_on_ruv` (#486 — the last `iiv_on_ruv` holdout):
/// the ODE counterpart of [`analytic_inner_gradient_iiv_on_ruv_m3_matches_fd`]. The
/// censored residual-eta data column `h·z` and the `exp(2·η_ruv)` variance scaling are
/// applied by the provider-agnostic `residual_inner_obs` over the **event-driven ODE
/// walk's** `ObsSens` (not the closed-form provider), so the analytic inner η-gradient
/// must match central FD of `individual_nll` — the non-IOV ODE M3 + `iiv_on_ruv` combo
/// the inner loop now admits (#623).
#[test]
fn analytic_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode() {
    use std::cell::RefCell;
    use std::collections::HashMap;
    let mut model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot / V - (CL/V) * central\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE iiv_on_ruv");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(model.is_ode_based(), "model must be on the ODE path");

    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        obs_raw_times: Vec::new(),
        // The last two rows are below the LLOQ (carried in `cens`).
        observations: vec![8.0, 7.0, 5.0, 3.0, 2.0, 2.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0, 0, 1, 1],
        occasions: vec![1; 6],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };

    let theta = &model.default_params.theta;
    let omega = &model.default_params.omega;
    let sigma = &model.default_params.sigma.values;
    let eta = vec![0.12, -0.05, 0.2, 0.15]; // non-zero η_ruv

    let analytic = analytic_eta_nll_gradient(&model, &subject, theta, &eta, omega, sigma)
        .expect("analytic non-IOV ODE M3 + iiv_on_ruv inner gradient");

    let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(&subject));
    let obj = |e: &[f64]| -> f64 {
        let mut s = scratch.borrow_mut();
        individual_nll_into_with_schedule(
            &model,
            &subject,
            theta,
            e,
            omega,
            sigma,
            &model.residual_correlations,
            &mut s,
            None,
        )
    };
    let fd = gradient_fd(&obj, &eta, model.n_eta);
    for k in 0..model.n_eta {
        assert!(
            (analytic[k] - fd[k]).abs() < 1e-4 * (1.0 + fd[k].abs()),
            "η[{k}]: analytic {} vs FD {}",
            analytic[k],
            fd[k]
        );
    }
}

/// Dense BFGS vs L-BFGS scaling with inner dimension `n`, on an
/// ill-conditioned 1-D-Laplacian quadratic `½xᵀLx − 1ᵀx` (cond ≈ (n/π)², so
/// the solve needs ~O(n) curvature updates — representative of a curved inner
/// NLL). Both use the **analytic** gradient `Lx − 1` so the per-iteration cost
/// is dominated by the solver's linear algebra, not the gradient: dense is
/// `O(n²)`/step (matvec + rank-2 update), L-BFGS `O(m·n)`/step. Isolates the
/// solver, unlike a real fit where the prediction/FD cost dominates.
#[test]
#[ignore = "bench: cargo test --release ... -- --ignored --nocapture inner_solver_scaling_bench"]
fn inner_solver_scaling_bench() {
    use std::time::Instant;
    eprintln!("inner-solver scaling (analytic-gradient Laplacian quadratic):");
    for &n in &[4usize, 8, 16, 32, 64, 128, 256] {
        // f(x) = ½ Σ_i (x_i − x_{i-1})² + ½ x_0²  −  Σ_i x_i   (x_{-1}=0).
        let obj = move |x: &[f64]| -> f64 {
            let mut f = 0.5 * x[0] * x[0];
            for i in 1..n {
                let d = x[i] - x[i - 1];
                f += 0.5 * d * d;
            }
            f - x.iter().sum::<f64>()
        };
        // grad = L x − 1, L the Dirichlet 1-D Laplacian (tridiag 2,−1).
        let grad = move |x: &[f64]| -> Vec<f64> {
            let mut g = vec![0.0; n];
            for i in 0..n {
                let mut v = 2.0 * x[i];
                if i > 0 {
                    v -= x[i - 1];
                }
                if i + 1 < n {
                    v -= x[i + 1];
                }
                g[i] = v - 1.0;
            }
            g
        };
        let runs = 50;
        let time_it = |solver: &dyn Fn(&mut [f64]) -> bool| -> f64 {
            let t0 = Instant::now();
            for _ in 0..runs {
                let mut x = vec![0.0; n];
                std::hint::black_box(solver(&mut x));
            }
            t0.elapsed().as_secs_f64() * 1e3 / runs as f64
        };
        let t_dense = time_it(&|x| {
            dense_bfgs_core(
                &obj,
                &grad,
                x,
                n,
                2000,
                1e-8,
                None,
                None,
                None,
                &never_stall,
                false,
            )
        });
        let t_lbfgs = time_it(&|x| {
            lbfgs_core(
                &obj,
                &grad,
                x,
                n,
                2000,
                1e-8,
                None,
                None,
                &never_stall,
                false,
            )
        });
        eprintln!(
            "  n={n:4}  dense={t_dense:8.3} ms  lbfgs={t_lbfgs:8.3} ms  dense/lbfgs={:.2}x",
            t_dense / t_lbfgs
        );
    }
}

/// Pre-scratch-reuse dense BFGS, retained only as a bitwise regression oracle.
fn legacy_dense_bfgs(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let mut h_inv = DMatrix::identity(n, n);
    let mut g = grad(x);
    let mut f_cur = obj(x);
    let mut first_step = true;

    for _ in 0..max_iter {
        let gnorm = grad_norm_metric(&g, None);
        if first_step && gnorm > 1.0 {
            h_inv *= 1.0 / gnorm;
            first_step = false;
        }
        if gnorm < tol {
            return true;
        }

        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let d: Vec<f64> = d_vec.iter().copied().collect();
        if d.iter().zip(&g).map(|(di, gi)| di * gi).sum::<f64>() >= 0.0 {
            h_inv = DMatrix::identity(n, n);
            let d: Vec<f64> = (-&h_inv * &g_vec).iter().copied().collect();
            let (alpha, f_new, verified_g) = legacy_step(obj, grad, x, &d, &g, n, f_cur);
            if alpha == 0.0 {
                return false;
            }
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            f_cur = f_new;
            g = verified_g.unwrap_or_else(|| grad(x));
            continue;
        }

        let (alpha, f_new, verified_g) = legacy_step(obj, grad, x, &d, &g, n, f_cur);
        if alpha == 0.0 {
            return false;
        }
        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }
        f_cur = f_new;

        let g_new = verified_g.unwrap_or_else(|| grad(x));
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();
        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let sy = s_vec.dot(&y_vec);
        if curvature_pair_accepted(sy, s_vec.norm(), y_vec.norm(), false) {
            let rho = 1.0 / sy;
            let eye = DMatrix::identity(n, n);
            let s_yt = rho * &s_vec * y_vec.transpose();
            let y_st = rho * &y_vec * s_vec.transpose();
            let s_st = rho * &s_vec * s_vec.transpose();
            h_inv = (&eye - &s_yt) * &h_inv * (&eye - &y_st) + s_st;
        }
        g = g_new;
    }
    false
}

/// The step rule both drivers apply since #2270: a gradient-verified step below the objective's
/// resolution first, else the line search. A verified step hands back the gradient it took at
/// its end point, which the driver uses instead of evaluating it again.
fn legacy_step(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    d: &[f64],
    g: &[f64],
    n: usize,
    f0: f64,
) -> (f64, f64, Option<Vec<f64>>) {
    match exact_step_verified_by_gradient(obj, grad, x, d, f0, g, None, false) {
        Verdict::Verified(fc, gc) => (1.0, fc, Some(gc)),
        Verdict::Rejected(g_full) => {
            let (alpha, f) = legacy_line_search(obj, x, d, g, n, f0);
            (alpha, f, if alpha == 1.0 { g_full } else { None })
        }
    }
}

fn legacy_line_search(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],
    g: &[f64],
    n: usize,
    f0: f64,
) -> (f64, f64) {
    let c1 = 1e-4;
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
    if !(dg < 0.0) || !dg.is_finite() {
        return (0.0, f0);
    }
    let mut alpha = 1.0;
    let mut x_new = vec![0.0; n];
    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }
        let f_new = obj(&x_new);
        if f_new.is_finite() && f_new <= f0 + c1 * alpha * dg {
            return (alpha, f_new);
        }
        let denom = 2.0 * (f_new - f0 - dg * alpha);
        let alpha_quad = if f_new.is_finite() && denom > 0.0 {
            -dg * alpha * alpha / denom
        } else {
            0.5 * alpha
        };
        alpha = alpha_quad.clamp(0.1 * alpha, 0.5 * alpha);
        if alpha < 1e-16 {
            break;
        }
    }
    (0.0, f0)
}

fn legacy_nelder_mead(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let mut simplex = Vec::with_capacity(n + 1);
    simplex.push(x.to_vec());
    for i in 0..n {
        let mut point = x.to_vec();
        let delta = if point[i].abs() > 1e-8 {
            0.05 * point[i].abs()
        } else {
            0.00025
        };
        point[i] += delta;
        simplex.push(point);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|p| obj(p)).collect();

    for _ in 0..max_iter {
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Greater)
        });
        let best = indices[0];
        let worst = indices[n];
        let second_worst = indices[n - 1];
        if fvals[worst] - fvals[best] < tol {
            x.copy_from_slice(&simplex[best]);
            return true;
        }

        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for value in &mut centroid {
            *value /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + (centroid[j] - simplex[worst][j]))
            .collect();
        let fr = obj(&reflected);
        if fr < fvals[second_worst] && fr >= fvals[best] {
            simplex[worst] = reflected;
            fvals[worst] = fr;
            continue;
        }
        if fr < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (reflected[j] - centroid[j]))
                .collect();
            let fe = obj(&expanded);
            if fe < fr {
                simplex[worst] = expanded;
                fvals[worst] = fe;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = fr;
            }
            continue;
        }

        let contracted: Vec<f64> = (0..n)
            .map(|j| centroid[j] + 0.5 * (simplex[worst][j] - centroid[j]))
            .collect();
        let fc = obj(&contracted);
        if fc < fvals[worst] {
            simplex[worst] = contracted;
            fvals[worst] = fc;
            continue;
        }

        let best_point = simplex[best].clone();
        for i in 0..=n {
            if i != best {
                for j in 0..n {
                    simplex[i][j] = best_point[j] + 0.5 * (simplex[i][j] - best_point[j]);
                }
                fvals[i] = obj(&simplex[i]);
            }
        }
    }
    let best = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
        .map(|(i, _)| i)
        .unwrap();
    x.copy_from_slice(&simplex[best]);
    false
}

fn run_dense_scratch_fixture(n: usize, legacy: bool) -> (bool, Vec<u64>, u64, usize, usize) {
    let obj_evals = std::cell::Cell::new(0usize);
    let grad_evals = std::cell::Cell::new(0usize);
    let obj = |x: &[f64]| {
        obj_evals.set(obj_evals.get() + 1);
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let target = (i + 1) as f64 / n as f64;
                let residual = xi - target;
                0.5 * (i + 1) as f64 * residual * residual + 0.01 * residual.powi(4)
            })
            .sum::<f64>()
    };
    let grad = |x: &[f64]| {
        grad_evals.set(grad_evals.get() + 1);
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let target = (i + 1) as f64 / n as f64;
                let residual = xi - target;
                (i + 1) as f64 * residual + 0.04 * residual.powi(3)
            })
            .collect::<Vec<_>>()
    };
    let mut x = vec![1.5; n];
    let converged = if legacy {
        legacy_dense_bfgs(&obj, &grad, &mut x, n, 200, 1e-10)
    } else {
        dense_bfgs_core(
            &obj,
            &grad,
            &mut x,
            n,
            200,
            1e-10,
            None,
            None,
            None,
            &never_stall,
            false,
        )
    };
    let final_objective = obj(&x).to_bits();
    (
        converged,
        x.iter().map(|v| v.to_bits()).collect(),
        final_objective,
        obj_evals.get(),
        grad_evals.get(),
    )
}

#[test]
fn dense_bfgs_scratch_reuse_is_bitwise_identical_to_legacy() {
    for n in [2, 4, 8] {
        assert_eq!(
            run_dense_scratch_fixture(n, false),
            run_dense_scratch_fixture(n, true),
            "dense BFGS changed its path at eta dimension {n}"
        );
    }
}

#[test]
fn nelder_mead_scratch_reuse_is_bitwise_identical_to_legacy() {
    for n in [2, 4, 8] {
        let run = |legacy: bool| {
            let evals = std::cell::Cell::new(0usize);
            let obj = |x: &[f64]| {
                evals.set(evals.get() + 1);
                x.iter()
                    .enumerate()
                    .map(|(i, &xi)| {
                        let residual = xi - (i + 1) as f64 / n as f64;
                        (i + 1) as f64 * residual * residual + 0.01 * residual.powi(4)
                    })
                    .sum::<f64>()
            };
            let mut x = vec![1.5; n];
            let converged = if legacy {
                legacy_nelder_mead(&obj, &mut x, n, 2_000, 1e-12)
            } else {
                nelder_mead_minimize(&obj, &mut x, n, 2_000, 1e-12)
            };
            let final_objective = obj(&x).to_bits();
            (
                converged,
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                final_objective,
                evals.get(),
            )
        };
        assert_eq!(
            run(false),
            run(true),
            "Nelder-Mead changed its path at eta dimension {n}"
        );
    }
}

#[test]
#[cfg(profiling_allocations)]
#[ignore = "allocation microbenchmark: run alone with --ignored --nocapture"]
fn dense_bfgs_scratch_allocation_bench() {
    for n in [2, 4, 8] {
        ALLOCATION_CALLS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        std::hint::black_box(run_dense_scratch_fixture(n, false));
        let current = (
            ALLOCATION_CALLS.load(Ordering::Relaxed),
            ALLOCATED_BYTES.load(Ordering::Relaxed),
        );

        ALLOCATION_CALLS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        std::hint::black_box(run_dense_scratch_fixture(n, true));
        let legacy = (
            ALLOCATION_CALLS.load(Ordering::Relaxed),
            ALLOCATED_BYTES.load(Ordering::Relaxed),
        );
        eprintln!(
            "n={n}: scratch={} allocs/{} bytes; legacy={} allocs/{} bytes",
            current.0, current.1, legacy.0, legacy.1
        );
        assert!(
            current.0 < legacy.0,
            "scratch reuse must reduce allocations"
        );
        assert!(current.1 < legacy.1, "scratch reuse must reduce bytes");
    }
}

/// The interpolating backtracking line search returns a step that satisfies
/// the Armijo sufficient-decrease test and strictly lowers the objective,
/// using only a handful of trial evaluations (the property the FOCEI inner
/// loop relies on — fixed halving used ~20 here and frequently hit the cap).
#[test]
fn line_search_finds_armijo_step_quickly() {
    // f(x) = (x − 3)²; at x = 0 the unit Newton-less step −g overshoots the
    // minimiser, so a fixed-halving search would backtrack repeatedly.
    let obj = |x: &[f64]| -> f64 { (x[0] - 3.0) * (x[0] - 3.0) };
    let x = [0.0];
    let g = [2.0 * (x[0] - 3.0)]; // = −6
    let d = [-g[0]]; // steepest descent, dg = −36 < 0
    let f0 = obj(&x);
    let evals = std::cell::Cell::new(0usize);
    let counting = |xx: &[f64]| {
        evals.set(evals.get() + 1);
        obj(xx)
    };
    let mut trial = [0.0];
    let (alpha, f_new) =
        backtracking_line_search(&counting, &x, &d, &g, f0, &mut trial, false, false);
    let evals = evals.get();
    assert!(alpha > 0.0, "a descent step must be found");
    let c1 = 1e-4;
    let dg: f64 = d.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
    assert!(
        f_new <= f0 + c1 * alpha * dg,
        "returned step must satisfy Armijo"
    );
    assert!(f_new < f0, "objective must strictly decrease");
    assert!(
        evals <= 5,
        "interpolation should converge in a few evals, got {evals}"
    );
}

#[test]
fn short_line_search_is_an_explicit_policy() {
    assert_eq!(inner_line_search_trials(true), 10);
    assert_eq!(inner_line_search_trials(false), MAX_LINE_SEARCH_TRIALS);
}

/// A non-descent direction (dg ≥ 0) yields `alpha == 0` and leaves the
/// objective baseline untouched — the signal the inner BFGS uses to stop /
/// fall back rather than step uphill.
#[test]
fn line_search_rejects_non_descent_direction() {
    let obj = |x: &[f64]| -> f64 { (x[0] - 3.0) * (x[0] - 3.0) };
    let x = [0.0];
    let g = [2.0 * (x[0] - 3.0)]; // = −6
    let d = [g[0]]; // SAME sign as g → dg = +36 ≥ 0 (ascent)
    let f0 = obj(&x);
    let mut trial = [0.0];
    let (alpha, f_new) = backtracking_line_search(&obj, &x, &d, &g, f0, &mut trial, false, false);
    assert_eq!(alpha, 0.0);
    assert_eq!(f_new, f0);
}

/// A trial point where the objective goes **non-finite** — an out-of-domain η
/// where an absorption closed form leaves its convergence region and returns
/// ±inf/NaN — must not crash the line search. The quadratic safeguard has no
/// finite sample to interpolate, so it falls back to plain halving and
/// eventually reports "no step" (`alpha == 0`). Regression for the
/// `clamp(NaN, NaN)` SIGABRT surfaced by the transit multi-dose + covariate
/// anchor (#719 close-out): a non-finite `f_new` used to poison `alpha`, and
/// the next trial's `clamp(0.1·α, 0.5·α)` — bounds both NaN — aborted.
#[test]
fn line_search_survives_non_finite_objective() {
    let x = [0.0];
    let g = [-6.0];
    let d = [6.0]; // dg = −36 < 0 (a genuine descent direction)
    let f0 = 10.0;
    // Every trial step returns NaN — must not panic, must report no step.
    let nan_obj = |_: &[f64]| -> f64 { f64::NAN };
    let mut trial = [0.0];
    let (alpha, f_new) =
        backtracking_line_search(&nan_obj, &x, &d, &g, f0, &mut trial, false, false);
    assert_eq!(alpha, 0.0, "a never-finite objective yields no step");
    assert_eq!(f_new, f0, "baseline objective is returned unchanged");
    // +inf trials behave identically (never accepted, never a panic).
    let inf_obj = |_: &[f64]| -> f64 { f64::INFINITY };
    let (alpha, f_new) =
        backtracking_line_search(&inf_obj, &x, &d, &g, f0, &mut trial, false, false);
    assert_eq!(alpha, 0.0);
    assert_eq!(f_new, f0);
}

/// A **non-finite search direction** (`dg = ±inf`, from a blown-up BFGS
/// update) is rejected up front rather than driving `−dg·α²/denom` to
/// `inf/inf = NaN`. Companion regression to the clamp-panic fix.
#[test]
fn line_search_rejects_non_finite_direction() {
    let obj = |x: &[f64]| -> f64 { (x[0] - 3.0) * (x[0] - 3.0) };
    let x = [0.0];
    let g = [-6.0];
    let d = [f64::INFINITY]; // dg = −inf: a non-finite "descent" direction
    let f0 = obj(&x);
    let mut trial = [0.0];
    let (alpha, f_new) = backtracking_line_search(&obj, &x, &d, &g, f0, &mut trial, false, false);
    assert_eq!(alpha, 0.0);
    assert_eq!(f_new, f0);
}

/// The refactored dense BFGS (objective-tracked line search) still drives a
/// well-conditioned quadratic to its analytic minimiser.
#[test]
fn dense_bfgs_converges_on_quadratic() {
    // f(x) = (x0−1)² + 4(x1+2)², minimiser (1, −2).
    let obj =
        |x: &[f64]| -> f64 { (x[0] - 1.0) * (x[0] - 1.0) + 4.0 * (x[1] + 2.0) * (x[1] + 2.0) };
    let grad = |x: &[f64]| -> Vec<f64> { vec![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)] };
    let mut x = vec![0.0, 0.0];
    let ok = dense_bfgs_core(
        &obj,
        &grad,
        &mut x,
        2,
        200,
        1e-10,
        None,
        None,
        None,
        &never_stall,
        false,
    );
    assert!(ok, "BFGS should report convergence");
    assert!((x[0] - 1.0).abs() < 1e-6, "x0 = {}", x[0]);
    assert!((x[1] + 2.0).abs() < 1e-6, "x1 = {}", x[1]);
}

#[test]
fn test_inner_loop_stats_default() {
    let s = InnerLoopStats::default();
    assert_eq!(s.n_unconverged, 0);
    assert_eq!(s.n_fallback, 0);
}

// ── FREM inner-loop preconditioner (issue #406) ──────────────────────────

#[test]
fn preconditioner_scales_each_dim_by_its_own_curvature() {
    // 4 etas: 2 PK (dims 0,1; no FREM pseudo-obs) and 2 covariate (dims 2,3;
    // FREMTYPE 100→eta2, 200→eta3). The covariate pseudo-obs precision is
    // 1/R = 1/(EPSCOV²) = 1e6; PK dims have no data term and fall back to the
    // prior conditional scale 1/Ω⁻¹ᵢᵢ.
    let mut fremtype_to_indices = std::collections::HashMap::new();
    fremtype_to_indices.insert(100u16, (5usize, 2usize));
    fremtype_to_indices.insert(200u16, (6usize, 3usize));
    let fc = FremConfig {
        fremtype_to_indices,
        covariate_sigma_index: 1,
    };
    // Ω⁻¹: PK precisions 10 and 4; covariate prior precisions tiny (0.01).
    let omega_inv = DMatrix::from_diagonal(&DVector::from_column_slice(&[10.0, 4.0, 0.01, 0.01]));
    // sigma[1] = EPSCOV = 1e-3 (SD) → R = 1e-6 → data precision 1e6.
    let sigma = [0.3, 1e-3];
    // One PK obs row (ft=0) plus one pseudo-obs row per covariate.
    let fremtype = [0u16, 100, 200];

    let p = preconditioner_from_parts(&fc, &fremtype, &omega_inv, &sigma, 4)
        .expect("Some for n_eta > 0");

    // PK dims: 1/Ω⁻¹ᵢᵢ.
    assert!((p[0] - 0.1).abs() < 1e-9, "p0 = {}", p[0]);
    assert!((p[1] - 0.25).abs() < 1e-9, "p1 = {}", p[1]);
    // Covariate dims: 1/(0.01 + 1e6) ≈ 1e-6 — sharply smaller than PK.
    assert!(p[2] < 1.1e-6 && p[2] > 0.9e-6, "p2 = {}", p[2]);
    assert!(p[3] < 1.1e-6 && p[3] > 0.9e-6, "p3 = {}", p[3]);
    // The whole point: covariate dims get a step scale ~1e5× tighter than PK,
    // so a single preconditioned BFGS step is near-Newton for them.
    assert!(p[0] / p[2] > 1e4);
}

#[test]
fn preconditioner_is_none_for_zero_eta() {
    let fc = FremConfig {
        fremtype_to_indices: std::collections::HashMap::new(),
        covariate_sigma_index: 0,
    };
    let omega_inv = DMatrix::<f64>::zeros(0, 0);
    assert!(preconditioner_from_parts(&fc, &[], &omega_inv, &[1e-3], 0).is_none());
}

/// The general (non-FREM) inner preconditioner inverts the Ω⁻¹ diagonal so
/// each BFGS dimension is scaled by its prior conditional variance, giving a
/// well-scaled H0 for multi-scale / correlated Ω.
#[test]
fn inner_precond_from_omega_inverts_diagonal() {
    // Diagonal Ω⁻¹ = diag(10, 2, 0.5) → precond = diag(0.1, 0.5, 2.0).
    let omega_inv = DMatrix::from_diagonal(&DVector::from_column_slice(&[10.0, 2.0, 0.5]));
    let p = inner_preconditioner_from_omega(&omega_inv, 3).expect("usable diagonal");
    assert!((p[0] - 0.1).abs() < 1e-12);
    assert!((p[1] - 0.5).abs() < 1e-12);
    assert!((p[2] - 2.0).abs() < 1e-12);
    // n_eta == 0 → None (identity H0).
    assert!(inner_preconditioner_from_omega(&DMatrix::<f64>::zeros(0, 0), 0).is_none());
    // A non-positive diagonal entry is skipped but a usable one still yields Some.
    let mixed = DMatrix::from_diagonal(&DVector::from_column_slice(&[0.0, 4.0]));
    let pm = inner_preconditioner_from_omega(&mixed, 2).expect("one usable entry");
    assert_eq!(pm[0], 1.0); // untouched default for the zero diagonal
    assert!((pm[1] - 0.25).abs() < 1e-12);
}

#[test]
fn test_ebe_result_converged_flag() {
    // Verify EbeResult struct has the expected fields.
    let r = EbeResult {
        eta: nalgebra::DVector::zeros(2),
        h_matrix: nalgebra::DMatrix::identity(2, 2),
        terminal_hessian: None,
        converged: true,
        used_fallback: false,
        grad_norm: 0.0,
        nll: 1.5,
        kappas: Vec::new(),
        hard_reject: false,
        screen_moved: false,
    };
    assert!(r.converged);
    assert!(!r.used_fallback);
    assert_eq!(r.grad_norm, 0.0);
}

#[test]
fn analytic_partial_certification_honors_requested_tolerance() {
    assert!(certifies_analytic_partial(1e-8, 1e-8));
    assert!(!certifies_analytic_partial(1e-7, 1e-8));
    assert!(!certifies_analytic_partial(0.05, 1e-5));
    assert!(!certifies_analytic_partial(f64::NAN, 1e-5));
}

#[test]
fn hessian_seed_is_applied_through_a_cholesky_solve() {
    let h = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    let got = seed_h_inv(2, Some(&h)).expect("SPD seed is accepted");
    let identity = &h * &got;
    assert!((identity - DMatrix::identity(2, 2)).norm() < 1e-12);
    assert_eq!(init_h_inv(2, None, Some(&got)), got);
    // An accepted seed outranks the diagonal preconditioner.
    assert_eq!(init_h_inv(2, Some(&[10.0, 20.0]), Some(&got)), got);

    // An indefinite analytical seed is *rejected* — `None`, so the caller can tell — and the
    // metric falls back to the legacy identity / preconditioner.
    let bad = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, -1.0]));
    assert_eq!(seed_h_inv(2, Some(&bad)), None);
    assert_eq!(init_h_inv(2, None, None), DMatrix::identity(2, 2));
    assert_eq!(
        init_h_inv(2, Some(&[10.0, 20.0]), None),
        DMatrix::from_diagonal(&DVector::from_column_slice(&[10.0, 20.0]))
    );
    // Wrong shape and a non-finite entry are rejected the same way.
    let wrong_shape = DMatrix::identity(3, 3);
    assert_eq!(seed_h_inv(2, Some(&wrong_shape)), None);
    let nan = DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
    assert_eq!(seed_h_inv(2, Some(&nan)), None);
}

/// A seed that fails the SPD gate must reproduce the **unseeded** dense BFGS exactly — same
/// iterates, same number of objective evaluations. The gate lives in `seed_h_inv`, but the
/// `1/gnorm` first-step scaling is keyed separately; when it was keyed on `hessian_seed.
/// is_some()` rather than on the seed being *accepted*, a rejected seed produced an
/// unscaled identity first step (`−g` at full length) and a different trajectory from the
/// baseline it claims to preserve. Start with `‖g‖ ≈ 16 > 1` so the scaling is live.
#[test]
fn rejected_hessian_seed_reproduces_the_unseeded_solve() {
    use std::cell::Cell;
    let n_obj = Cell::new(0usize);
    let obj = |x: &[f64]| -> f64 {
        n_obj.set(n_obj.get() + 1);
        (x[0] - 1.0) * (x[0] - 1.0) + 4.0 * (x[1] + 2.0) * (x[1] + 2.0)
    };
    let grad = |x: &[f64]| -> Vec<f64> { vec![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)] };
    let run = |seed: Option<&DMatrix<f64>>| {
        n_obj.set(0);
        let mut x = vec![0.0, 0.0];
        let ok = dense_bfgs_core(
            &obj,
            &grad,
            &mut x,
            2,
            200,
            1e-10,
            None,
            seed,
            None,
            &never_stall,
            false,
        );
        assert!(ok);
        (x, n_obj.get())
    };
    let (x_unseeded, evals_unseeded) = run(None);
    let bad = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, -1.0]));
    let (x_rejected, evals_rejected) = run(Some(&bad));
    assert_eq!(
        x_rejected, x_unseeded,
        "rejected seed changed the iterate path"
    );
    assert_eq!(
        evals_rejected, evals_unseeded,
        "rejected seed changed the objective-evaluation count"
    );

    // Control: an *accepted* exact seed (`H = diag(2, 8)`) is a different — one-step — solve,
    // so the equality above is not vacuous.
    let exact = DMatrix::from_diagonal(&DVector::from_column_slice(&[2.0, 8.0]));
    let (x_seeded, evals_seeded) = run(Some(&exact));
    assert!((x_seeded[0] - 1.0).abs() < 1e-9 && (x_seeded[1] + 2.0).abs() < 1e-9);
    assert!(
        evals_seeded < evals_unseeded,
        "accepted seed {evals_seeded} vs unseeded {evals_unseeded} objective evaluations"
    );
}

/// Under the default `bloq_method = drop` a `CENS` flag on a row is *not* a censored kernel —
/// `analytic_eta_nll_gradient_with_schedule` scores the row Gaussian (`cens = if m3 {..} else
/// {0}`). The fused first gradient must apply the same rule; passing the raw flag through
/// hands BFGS `−log Φ` rows against an `obj` that scores them Gaussian, so a warm start at the
/// mode fails `gnorm < tol` and takes a non-descent first step. Both arms are asserted: the
/// Gaussian rule under `drop`, and the censored kernel under `m3`, each bit-identical to the
/// ordinary gradient at the same point.
#[test]
fn light_seed_gradient_applies_cens_only_under_m3() {
    let (mut model, mut subject) = wellidentified_oral_fixture();
    // Row 5 (the 1.5 at t = 12) becomes a BLOQ row with LLOQ = 1.5 held in `observations`.
    subject.cens[5] = 1;
    let params = model.default_params.clone();
    let err_keys = model.error_spec.obs_keys(&subject);
    let eta = vec![0.05, -0.02, 0.1];
    for bloq in [crate::types::BloqMethod::Drop, crate::types::BloqMethod::M3] {
        model.bloq_method = bloq;
        let (_, fused) = analytic_inner_seed_hessian(
            &model,
            &subject,
            &params,
            &eta,
            None,
            None,
            &err_keys,
            &mut Vec::new(),
            false,
            None,
        )
        .expect("fixture supports the light seed");
        let ordinary = analytic_eta_nll_gradient(
            &model,
            &subject,
            &params.theta,
            &eta,
            &params.omega,
            &params.sigma.values,
        )
        .expect("fixture supports the ordinary gradient");
        assert_eq!(
            fused.expect("light path fuses the gradient"),
            ordinary,
            "{}: fused first gradient diverged from the ordinary inner gradient",
            bloq.label()
        );
    }
    // The two arms genuinely differ on the CENS row, so agreement in each is not a tautology.
    model.bloq_method = crate::types::BloqMethod::Drop;
    let g_drop = analytic_eta_nll_gradient(
        &model,
        &subject,
        &params.theta,
        &eta,
        &params.omega,
        &params.sigma.values,
    )
    .unwrap();
    model.bloq_method = crate::types::BloqMethod::M3;
    let g_m3 = analytic_eta_nll_gradient(
        &model,
        &subject,
        &params.theta,
        &eta,
        &params.omega,
        &params.sigma.values,
    )
    .unwrap();
    assert_ne!(
        g_drop, g_m3,
        "test premise: the CENS row must be live under m3"
    );
}

/// `gradient = fd` (like `FERX_NO_ANALYTIC_INNER`) promises the historical FD inner solve.
/// The seed is a provider product too, so it must decline under that hatch: without the gate
/// the fused gradient was correctly dropped (`use_analytic` false) but the provider's `H̃` was
/// still accepted as the BFGS metric whenever it was SPD, and the FD solve no longer
/// reproduced the unseeded path. Asserted on the seed helper (both the light and the exact
/// arm) and end to end: under `fd` a seeded policy must be bit-identical to the unseeded one,
/// while under the default `analytic` routing the two solves genuinely differ.
#[test]
fn hessian_seed_declines_under_the_fd_inner_gradient_hatch() {
    let (mut model, subject) = wellidentified_oral_fixture();
    let params = model.default_params.clone();
    let err_keys = model.error_spec.obs_keys(&subject);
    let eta = vec![0.05, -0.02, 0.1];
    let seed = |model: &CompiledModel, exact: bool| {
        analytic_inner_seed_hessian(
            model,
            &subject,
            &params,
            &eta,
            None,
            None,
            &err_keys,
            &mut Vec::new(),
            exact,
            None,
        )
    };
    assert!(
        seed(&model, false).is_some(),
        "premise: light seed in scope"
    );
    assert!(seed(&model, true).is_some(), "premise: exact seed in scope");

    // Warm-started, off-mode: the seed is gated to genuinely warm-started solves since
    // #1504, and a start already at the mode would converge identically under any metric,
    // so the premise is exercised from a nonzero start away from the solution.
    let warm = vec![0.5, -0.4, 0.6];
    let solve = |model: &CompiledModel, seed: InnerHessianSeed| {
        let r = find_ebe_cached(
            model,
            &subject,
            &params,
            100,
            1e-8,
            Some(&warm),
            None,
            0,
            None,
            InnerSolvePolicy {
                seed,
                ebe_variance: EbeVariance::Conditional,
                capture_terminal_hessian: false,
                accelerate_exact_outer: false,
                screen: 0,
                init_is_mode: false,
                keep_certified_start: false,
            },
        );
        (r.eta, r.h_matrix, r.grad_norm, r.nll)
    };
    let analytic_off = solve(&model, InnerHessianSeed::None);
    let analytic_on = solve(&model, InnerHessianSeed::GaussNewton);
    assert_ne!(
        analytic_off, analytic_on,
        "premise: the seed must change the analytic solve, or the fd identity is vacuous"
    );

    model.gradient_method = crate::types::GradientMethod::Fd;
    assert!(
        seed(&model, false).is_none(),
        "light seed ignored `gradient = fd`"
    );
    assert!(
        seed(&model, true).is_none(),
        "exact seed ignored `gradient = fd`"
    );
    let fd_off = solve(&model, InnerHessianSeed::None);
    for kind in [InnerHessianSeed::GaussNewton, InnerHessianSeed::Exact] {
        let fd_on = solve(&model, kind);
        assert_eq!(
            fd_on, fd_off,
            "{kind:?} seed changed the `gradient = fd` inner solve"
        );
    }
}

/// The seed is a *local* model of the individual objective: it accelerates a solve
/// started near the mode it describes, but from a cold start its Newton-scaled first
/// step selects the eta basin (#1474, #1504). A cold start under a seeded policy
/// therefore runs both metrics and keeps the lower objective, with ties going to the
/// unseeded solve — so on this unimodal subject, where both reach the same mode, the
/// cold solve must reproduce the historical unseeded one bit-for-bit, while a nonzero
/// warm start under the same policy is seeded and takes a different (faster) path.
/// Zero-vector "warm" starts are cold placeholders (`freeze_flat_thetas` passes zeros
/// as `prev_etas`). The bimodal cases live in
/// `cold_seeded_solve_keeps_the_lower_of_both_metrics`.
#[test]
fn hessian_seed_applies_only_to_genuinely_warm_started_solves() {
    let (model, subject) = wellidentified_oral_fixture();
    let params = model.default_params.clone();
    let solve = |eta_init: Option<&[f64]>, seed: InnerHessianSeed| {
        let r = find_ebe_cached(
            &model,
            &subject,
            &params,
            100,
            1e-8,
            eta_init,
            None,
            0,
            None,
            InnerSolvePolicy {
                seed,
                ebe_variance: EbeVariance::Conditional,
                capture_terminal_hessian: false,
                accelerate_exact_outer: false,
                screen: 0,
                init_is_mode: false,
                keep_certified_start: false,
            },
        );
        (r.eta, r.h_matrix, r.grad_norm, r.nll)
    };
    let cold_unseeded = solve(None, InnerHessianSeed::None);
    for kind in [InnerHessianSeed::GaussNewton, InnerHessianSeed::Exact] {
        // Premise: the forced-seed cold solve really ran and took another path to the
        // same mode — otherwise the tie arm below is never exercised.
        let schedule = cacheable_schedule(&model, &subject);
        let forced = find_ebe_solve(
            &model,
            &subject,
            &params,
            100,
            1e-8,
            None,
            None,
            0,
            schedule.as_ref(),
            InnerSolvePolicy {
                seed: kind,
                ebe_variance: EbeVariance::Conditional,
                capture_terminal_hessian: false,
                accelerate_exact_outer: false,
                screen: 0,
                init_is_mode: false,
                keep_certified_start: false,
            },
            true,
        )
        .expect("premise: the seed is in scope on this fixture");
        assert_ne!(
            (&forced.eta, forced.nll),
            (&cold_unseeded.0, cold_unseeded.3),
            "premise: the {kind:?}-seeded cold solve must differ in its last bits"
        );
        assert!(
            (forced.nll - cold_unseeded.3).abs() < 1e-9,
            "premise: both metrics reach the same mode ({} vs {})",
            forced.nll,
            cold_unseeded.3
        );
        assert_eq!(
            solve(None, kind),
            cold_unseeded,
            "{kind:?} seed changed a unimodal COLD solve — a tie must keep the unseeded one"
        );
        assert_eq!(
            solve(Some(&[0.0; 3]), kind),
            cold_unseeded,
            "{kind:?} seed changed a ZERO-placeholder warm start"
        );
    }
    // Premise: a genuine warm start is seeded, so the trajectory (and its terminal
    // point at this tolerance) differs from the unseeded one.
    let warm = [0.5, -0.4, 0.6];
    assert_ne!(
        solve(Some(&warm), InnerHessianSeed::GaussNewton),
        solve(Some(&warm), InnerHessianSeed::None),
        "premise: the seed must change a warm-started solve, or the gate is vacuous"
    );
}

/// A cold start under a seeded policy keeps whichever of the unseeded and seeded solves
/// reaches the lower objective (#1504). Neither metric picks the right η̂ basin everywhere,
/// and `examples/warfarin_if.ferx` has a subject on each side (FOCEI's Gauss–Newton seed,
/// cold start, realised −log L):
///
/// | θ | subject | unseeded | seeded | kept |
/// |---|---|---|---|---|
/// | `TVCL × 5` | ID 9 | −18.608 | −27.055 | seeded (8.45 lower) |
/// | `TVV2 × 5` | ID 11 | −24.772 | −12.653 | unseeded (12.1 lower) |
///
/// The case this was found on was ID 6 at the initial θ (unseeded −29.134, seeded −35.915).
/// Under the warm-only gate this replaced, ID 6 took the worse mode on the first outer
/// evaluation and every warm solve stayed there: the L-BFGS fit converged at −1185.250,
/// 3.3 OFV above the seeded-everywhere −1188.527. Since #2093 bounds a curvature-free first
/// step (`cap_curvature_free_step`) the *unseeded* solve reaches −35.915 there too, so ID 6
/// no longer straddles; ID 9 at `TVCL × 5` does, with the same values on both sides of that
/// change. Each case asserts its own straddle, so a policy that always keeps one side
/// reddens exactly one of them.
#[test]
fn cold_seeded_solve_keeps_the_lower_of_both_metrics() {
    use std::path::Path;
    let model =
        crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin_if.ferx"))
            .expect("warfarin_if parses");
    let pop = crate::io::datareader::read_nonmem_csv(
        Path::new("data/warfarin_if.csv"),
        Some(&["WT"]),
        None,
    )
    .expect("warfarin_if data loads");
    let policy = |seed| InnerSolvePolicy {
        seed,
        ebe_variance: EbeVariance::Conditional,
        capture_terminal_hessian: false,
        accelerate_exact_outer: true,
        screen: 0,
        init_is_mode: false,
        keep_certified_start: false,
    };
    let cases = [
        // (subject id, theta index to scale, factor, seeded side should win)
        ("9", 0, 5.0, true),
        ("11", 3, 5.0, false),
    ];
    for (id, k, factor, seeded_wins) in cases {
        let subject = pop.subjects.iter().find(|s| s.id == id).expect("subject");
        let mut params = model.default_params.clone();
        params.theta[k] *= factor;
        let schedule = cacheable_schedule(&model, subject);
        let run = |eta_init: Option<&[f64]>, seed| {
            find_ebe_cached(
                &model,
                subject,
                &params,
                500,
                1e-6,
                eta_init,
                None,
                0,
                schedule.as_ref(),
                policy(seed),
            )
        };
        let unseeded = run(None, InnerHessianSeed::None);
        let forced = find_ebe_solve(
            &model,
            subject,
            &params,
            500,
            1e-6,
            None,
            None,
            0,
            schedule.as_ref(),
            policy(InnerHessianSeed::GaussNewton),
            true,
        )
        .expect("premise: the seed is in scope");
        assert!(
            unseeded.nll.is_finite() && forced.nll.is_finite(),
            "ID {id}: non-finite cold solve ({} / {})",
            unseeded.nll,
            forced.nll
        );
        // Straddle: the two metrics land in different modes, on the side this case pins.
        let gap = forced.nll - unseeded.nll;
        assert!(
            if seeded_wins { gap < -1.0 } else { gap > 1.0 },
            "premise, ID {id}: seeded − unseeded cold −log L = {gap:.4} \
             (realised −8.447 for ID 9, +12.119 for ID 11)"
        );
        let expected = if seeded_wins { &forced } else { &unseeded };
        for (label, eta_init) in [("None", None), ("zero placeholder", Some(&[0.0; 5][..]))] {
            let got = run(eta_init, InnerHessianSeed::GaussNewton);
            assert_eq!(
                (&got.eta, got.nll),
                (&expected.eta, expected.nll),
                "ID {id}, cold start {label}: kept the {} solve, expected the {} one",
                if seeded_wins { "unseeded" } else { "seeded" },
                if seeded_wins { "seeded" } else { "unseeded" },
            );
        }
    }
}

/// The stage seed is a pure function of the stage's options — no process state, so two
/// concurrent fits cannot flip each other's seed and a post-fit `predict()` cannot inherit one.
/// Pins the mapping: one-node Laplace → exact; any multi-node quadrature (either anchor) →
/// Gauss–Newton; the FO family → Gauss–Newton; everything else (SAEM, Bayes, VI, …) → none.
#[test]
fn inner_hessian_seed_is_a_pure_function_of_the_stage_options() {
    use crate::types::{EstimationMethod, FitOptions};
    let seed = |method: EstimationMethod, n_agq: usize| {
        InnerHessianSeed::for_options(&FitOptions {
            method,
            n_agq,
            ..FitOptions::default()
        })
    };
    assert_eq!(seed(EstimationMethod::Laplace, 1), InnerHessianSeed::Exact);
    assert_eq!(
        seed(EstimationMethod::Laplace, 3),
        InnerHessianSeed::GaussNewton
    );
    assert_eq!(
        seed(EstimationMethod::FoceI, 3),
        InnerHessianSeed::GaussNewton
    );
    for m in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGn,
        EstimationMethod::FoceGnHybrid,
    ] {
        assert_eq!(seed(m, 1), InnerHessianSeed::GaussNewton, "{m:?}");
    }
    assert_eq!(seed(EstimationMethod::Saem, 1), InnerHessianSeed::None);
    assert_eq!(InnerSolvePolicy::default().seed, InnerHessianSeed::None);
    assert!(!InnerSolvePolicy::default().capture_terminal_hessian);
}

#[test]
fn test_inner_loop_stats_min_obs_filter() {
    // min_obs filter: subjects with fewer obs than min_obs are excluded
    // from n_unconverged count. We exercise this logic by constructing
    // InnerLoopStats manually (simulating what run_inner_loop_warm does).
    let results = vec![
        EbeResult {
            eta: nalgebra::DVector::zeros(1),
            h_matrix: nalgebra::DMatrix::identity(1, 1),
            terminal_hessian: None,
            converged: false, // unconverged
            used_fallback: false,
            grad_norm: 0.0,
            nll: 1.0,
            kappas: Vec::new(),
            hard_reject: false,
            screen_moved: false,
        },
        EbeResult {
            eta: nalgebra::DVector::zeros(1),
            h_matrix: nalgebra::DMatrix::identity(1, 1),
            terminal_hessian: None,
            converged: false, // also unconverged
            used_fallback: true,
            grad_norm: 0.0,
            nll: 2.0,
            kappas: Vec::new(),
            hard_reject: false,
            screen_moved: false,
        },
    ];
    // Simulate filter: first subject has 1 obs (below min_obs=2), second has 3 obs.
    let obs_counts = [1_usize, 3_usize];
    let min_obs = 2_usize;
    let n_unconverged = results
        .iter()
        .zip(obs_counts.iter())
        .filter(|(r, &n_obs)| !r.converged && n_obs >= min_obs.max(1))
        .count();
    let n_fallback = results.iter().filter(|r| r.used_fallback).count();
    // Only second subject counts (3 obs >= 2); first is filtered out.
    assert_eq!(n_unconverged, 1);
    // Both fallback counts regardless of min_obs.
    assert_eq!(n_fallback, 1);
}

/// #603 review #1/#2: a hard-rejected subject must be counted even with a short record,
/// so a single one forces the outer guard to reject the trial. Mirrors the `n_start_rejected`
/// derivation in `run_inner_loop_warm` (no `min_obs` filter, unlike `n_unconverged`).
#[test]
fn test_inner_loop_stats_counts_hard_reject_regardless_of_obs() {
    let make = |hard_reject: bool| EbeResult {
        eta: nalgebra::DVector::zeros(1),
        h_matrix: nalgebra::DMatrix::zeros(1, 1),
        terminal_hessian: None,
        converged: false,
        used_fallback: false,
        grad_norm: 0.0,
        nll: 1.0,
        kappas: Vec::new(),
        hard_reject,
        screen_moved: false,
    };
    // One hard-rejected subject with a single observation, one normal subject.
    let results = [make(true), make(false)];
    let obs_counts = [1_usize, 5_usize];
    let min_obs = 3_usize;

    // The `min_obs` filter would drop the 1-obs subject from `n_unconverged` …
    let n_unconverged = results
        .iter()
        .zip(obs_counts.iter())
        .filter(|(r, &n_obs)| !r.converged && n_obs >= min_obs.max(1))
        .count();
    assert_eq!(n_unconverged, 1); // only the 5-obs subject

    // … but `n_start_rejected` counts the hard reject regardless of obs count.
    let n_start_rejected = results.iter().filter(|r| r.hard_reject).count();
    assert_eq!(n_start_rejected, 1);
}

/// A minimal FREM model + subject: 3 etas (CL, V, COV_WT), and a subject with two PK
/// observations plus one covariate pseudo-observation (`FREMTYPE = 100`, value 90,
/// mapping to `theta[2]` = `TV_WT` and `eta[2]` = `ETA_WT_FREM`).
///
/// `default_params.theta[2]` is 90, i.e. the same value as the pseudo-observation, so a
/// caller testing anything that reads `cov_obs − TV` must move one of the two first —
/// at the default they are equal and a wrong answer is indistinguishable from a right one.
///
/// Shared by the Jacobian-override test and the cold-seed tests so the FREM shape is
/// declared once: a `CompiledModel` literal is ~50 fields wide and a second copy would
/// have to be edited on every field addition.
fn frem_model_and_subject() -> (CompiledModel, Subject) {
    use crate::types::{
        DoseEvent, ErrorModel, GradientMethod, OmegaMatrix, PkModel, PkParams, SigmaVector,
    };
    use std::collections::HashMap;

    // Build a minimal model with 3 etas: CL, V, COV_WT(FREM)
    let omega = OmegaMatrix::from_diagonal(
        &[0.09, 0.09, 100.0],
        vec!["ETA_CL".into(), "ETA_V".into(), "ETA_WT_FREM".into()],
    );
    let default_params = crate::types::ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![10.0, 100.0, 90.0],
        theta_names: vec!["TVCL".into(), "TVV".into(), "TV_WT".into()],
        theta_lower: vec![0.01, 1.0, 0.0],
        theta_upper: vec![100.0, 500.0, 200.0],
        theta_fixed: vec![false, false, true],
        omega,
        omega_fixed: vec![false, false, false],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["RUV".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: vec![],
        mixture: None,
    };
    let model = CompiledModel {
        residual_serial: None,
        priors: Vec::new(),
        prior_from_fit: None,
        covariate_model: None,
        has_conditional_eta_params: false,
        name: "frem_jac_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Additive,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Additive),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp(); // CL
                p.values[1] = theta[1] * eta[1].exp(); // V
                p
            },
        ),
        n_theta: 3,
        n_eta: 3,
        n_epsilon: 1,
        n_kappa: 0,
        kappa_names: vec![],
        theta_names: vec!["TVCL".into(), "TVV".into(), "TV_WT".into()],
        eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_WT_FREM".into()],
        indiv_param_names: vec!["CL".into(), "V".into(), "COV_WT".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        default_params,
        omega_init_as_sd: vec![false; 3],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: vec![],
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        covariate_mu_refs: Vec::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: None,
        pk_indices: vec![0, 1],
        eta_map: vec![0, 1, 2],
        pk_idx_f64: vec![0.0, 1.0],
        sel_flat: vec![1.0, 0.0],
        ode_spec: None,
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: crate::types::BloqMethod::Drop,
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(),
        eta_param_info: Vec::new(),
        kappa_param_types: Vec::new(),
        theta_transform: Vec::new(),
        theta_eta_linked: Vec::new(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: crate::types::ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: vec![],
        output_columns: vec![],
        dose_attr_map: Default::default(),
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
        frem_config: Some(crate::types::FremConfig {
            fremtype_to_indices: {
                let mut m = std::collections::HashMap::new();
                m.insert(100u16, (2usize, 2usize)); // TV_WT / ETA_WT_FREM
                m
            },
            covariate_sigma_index: 0,
        }),
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    };

    // Subject: 2 PK obs + 1 FREM obs
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 0.0],
        obs_raw_times: Vec::new(),
        observations: vec![5.0, 3.0, 90.0],
        obs_cmts: vec![1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: vec![0, 0, 100], // last obs is FREM
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };

    (model, subject)
}

#[test]
fn test_frem_jacobian_overrides_fd_with_exact_values() {
    let (model, subject) = frem_model_and_subject();

    let theta = [10.0, 100.0, 90.0];
    let eta = [0.1, -0.05, 2.5];

    let mut scratch = pk::EventPkParams::default();
    let jac = compute_jacobian_fd(&model, &subject, &theta, &eta, &mut scratch, None);

    // Row 2 (FREM obs) must be exactly [0, 0, 1]
    assert_eq!(jac[(2, 0)], 0.0, "FREM row: ∂Y/∂η_CL must be exactly 0");
    assert_eq!(jac[(2, 1)], 0.0, "FREM row: ∂Y/∂η_V must be exactly 0");
    assert_eq!(jac[(2, 2)], 1.0, "FREM row: ∂Y/∂η_COV must be exactly 1");

    // PK rows should be non-zero for at least CL (row 0, col 0)
    assert!(
        jac[(0, 0)].abs() > 1e-10,
        "PK row: ∂Y/∂η_CL should be nonzero"
    );
}

// ── FREM-aware cold seed (#406 / #1349) ────────────────────────────

/// The cold seed places each FREM covariate eta at its data-implied mode `cov_obs − TV`
/// and leaves every other eta at zero.
#[test]
fn cold_seed_places_frem_covariate_etas_at_their_data_implied_mode() {
    let (model, subject) = frem_model_and_subject();
    let mut params = model.default_params.clone();
    // The fixture ships `TV_WT` equal to the pseudo-observation, where the seed is 0 and
    // indistinguishable from the plain zero vector this test exists to reject. Move it.
    params.theta[2] = 72.0;

    let seed = cold_eta_seed(&model, &subject, &params, 3);

    // Non-degeneracy first: the quantity under test must not be zero, or the assertion
    // below passes against the very behaviour it is meant to catch.
    assert_eq!(subject.observations[2], 90.0);
    assert!(
        seed[2].abs() > 1.0,
        "the seeded value must be far from 0, or this test passes under the \
         pre-#1349 plain zero vector"
    );
    assert!(
        (seed[2] - 18.0).abs() < 1e-12,
        "covariate eta seeded at cov_obs − TV = 90 − 72 = 18, got {}",
        seed[2]
    );
    // PK etas carry no pseudo-observation and stay at 0.
    assert_eq!(seed[0], 0.0);
    assert_eq!(seed[1], 0.0);
}

/// #1349 — the *fallback* has to use the same seed, not a plain zero vector.
///
/// This is the call site the fix is about: when the inner BFGS does not certify
/// convergence, the FREM arm re-centres with a Nelder–Mead restart, and that restart used
/// to start at η = 0. NM's initial simplex step at zero is 0.00025 per coordinate, so it
/// cannot travel the 18 units this subject's covariate eta needs — it returned a point
/// with the covariate eta still at ~0 and the PK etas carrying the block-Ω⁻¹ force that
/// produces.
///
/// `max_iter = 1` is what forces the fallback: one BFGS iteration cannot reach a gradient
/// norm below `tol`, so `bfgs_converged` is false and the restart runs. The assertion is
/// on the covariate eta, which is the coordinate the two seeds disagree about.
#[test]
fn the_frem_nelder_mead_fallback_restarts_from_the_data_implied_seed() {
    let (model, subject) = frem_model_and_subject();
    let mut params = model.default_params.clone();
    params.theta[2] = 72.0; // pseudo-obs is 90 → the covariate eta's mode is 18

    let result = find_ebe(&model, &subject, &params, 1, 1e-12, None, None, 0);

    // Precondition: the fallback must actually have run, or this test is an assertion
    // about the ordinary BFGS path and the seed it exercises is the cold-start one.
    assert!(
        result.used_fallback,
        "max_iter = 1 must leave the inner BFGS uncertified so the NM restart runs"
    );
    assert!(
        (result.eta[2] - 18.0).abs() < 1.0,
        "covariate eta should stay at its data-implied mode 18, got {} — a restart from \
         a plain zero vector cannot travel there (#1349)",
        result.eta[2]
    );
}

/// A model with no `[frem]` config gets the plain zero vector, so every non-FREM caller
/// (the cold start *and* the Nelder–Mead fallback seed) is bit-identical to the
/// pre-#1349 `vec![0.0; n_eta]`.
#[test]
fn cold_seed_is_all_zeros_without_frem() {
    let (mut model, subject) = frem_model_and_subject();
    model.frem_config = None;
    let params = model.default_params.clone();

    let seed = cold_eta_seed(&model, &subject, &params, 3);

    assert_eq!(seed, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_nelder_mead_nan_objective_does_not_panic() {
    // Regression for issue #97: when a simplex vertex evaluates to a NaN
    // objective (e.g. an ODE prediction blowing up during the EBE search),
    // the `partial_cmp().unwrap()` sort used to panic — and, unwinding
    // through the non-unwinding optimizer callback, abort the whole fit.
    // NaN must now sort as worst and get reflected away instead.
    let obj = |x: &[f64]| -> f64 {
        if x[0] < 0.0 {
            // The "blow-up" region: objective is non-finite here.
            f64::NAN
        } else {
            (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
        }
    };
    // Seed the simplex entirely inside the NaN region so the very first
    // sort encounters only NaN vertices.
    let mut x = vec![-1.0, -1.0];
    // The contract under test is "does not panic"; the return flag and
    // final point are secondary. Coordinates must stay finite.
    let _converged = nelder_mead_minimize(&obj, &mut x, 2, 200, 1e-8);
    assert!(
        x.iter().all(|v| v.is_finite()),
        "Nelder-Mead must leave the point finite, got {x:?}"
    );
}

// ── #891: weakly-identified-coordinate detector for the guarded multi-start ──

/// The flatness probe flags a coordinate whose individual objective is flat
/// (data adds no curvature beyond the prior — the #891 weakly-identified case)
/// and leaves a sharply-curved, well-informed coordinate unscanned. Both share
/// the same prior (Ω = I ⇒ prior curvature 1.0), so the only difference is the
/// data curvature the probe measures.
#[test]
fn weakly_identified_coords_flags_flat_not_sharp() {
    use crate::types::OmegaMatrix;
    let omega = OmegaMatrix::from_diagonal(&[1.0, 1.0], vec!["SHARP".into(), "FLAT".into()]);
    // obj already carries the prior term, so these coefficients are the *total*
    // posterior curvature: coord 0 is sharply informed (10 ≫ 2·prior), coord 1
    // carries prior curvature only (1.0 < 2·prior ⇒ weakly identified).
    let obj = |e: &[f64]| -> f64 { 0.5 * 10.0 * e[0] * e[0] + 0.5 * 1.0 * e[1] * e[1] };
    let eta = vec![0.0, 0.0];
    let nll = obj(&eta);
    let flags = weakly_identified_coords(&obj, &eta, nll, &omega, 2);
    assert_eq!(
        flags,
        vec![false, true],
        "sharp coordinate must be skipped, flat coordinate must be scanned"
    );
}

/// A fixed (zero-variance) effect can't move, so it is never scanned even when
/// flat; and a non-finite objective at the mode disables the probe entirely
/// (returns all-false) rather than dividing by a bogus curvature.
#[test]
fn weakly_identified_coords_skips_fixed_and_nonfinite() {
    use crate::types::OmegaMatrix;
    let omega = OmegaMatrix::from_diagonal(&[0.0, 1.0], vec!["FIXED".into(), "FLAT".into()]);
    let flat = |e: &[f64]| -> f64 { 0.5 * e[1] * e[1] };
    let eta = vec![0.0, 0.0];
    // Fixed coord 0 skipped despite flat objective; flat free coord 1 flagged.
    let flags = weakly_identified_coords(&flat, &eta, flat(&eta), &omega, 2);
    assert_eq!(flags, vec![false, true]);
    // Non-finite objective at the mode → probe declines (all-false), no scan.
    let flags_bad = weakly_identified_coords(&flat, &eta, f64::INFINITY, &omega, 2);
    assert_eq!(flags_bad, vec![false, false]);
}

/// One-compartment oral fixture with six informative observations across the profile, so
/// every η is well identified and the light seed path (`analytic_inner_seed_hessian`) is in
/// scope. Shared by the restart, seed and fused-gradient tests below.
fn wellidentified_oral_fixture() -> (CompiledModel, Subject) {
    use crate::types::{DoseEvent, Subject};
    use std::collections::HashMap;
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n",
    )
    .expect("parse one_cpt_oral model");

    // Six informative observations across the profile ⇒ all etas well identified.
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 12.0, 10.0, 6.0, 3.0, 1.5],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };
    assert!(
        !subject.has_resets() && !subject.has_tv_covariates(),
        "test premise: subject must exercise the #891 (non reset/TV) probe path"
    );

    (model, subject)
}

/// No-regression: a well-identified, unimodal subject (no resets / TV-covariates)
/// must return a bit-identical EBE with the guarded multi-start on
/// (`inner_restarts = 3`) and off (`inner_restarts = 0`). The #891 probe may scan
/// weakly-informed coordinates, but every alternate seed reconverges to the same
/// basin and is rejected by the `+1e-9` improvement guard, so the returned η̂ and
/// its objective are unchanged — the added cost buys no spurious mode change.
#[test]
fn inner_restarts_bit_identical_on_wellidentified_subject() {
    let (model, subject) = wellidentified_oral_fixture();
    let params = &model.default_params;
    let off = find_ebe(&model, &subject, params, 100, 1e-8, None, None, 0);
    let on = find_ebe(&model, &subject, params, 100, 1e-8, None, None, 3);

    // Bit-identical is the exact contract, not mere closeness: for a unimodal
    // subject no alternate seed beats the improvement guard (`cand_nll + 1e-9 <
    // nll`), so `eta`/`nll` are never reassigned — `on` returns the very same
    // f64s the base solve produced. The two solves are deterministic (pure
    // objective, no RNG), so assert exact equality.
    assert!(off.nll.is_finite() && on.nll.is_finite());
    assert_eq!(
        on.nll, off.nll,
        "objective must be bit-identical on a unimodal subject: on {} vs off {}",
        on.nll, off.nll
    );
    for k in 0..model.n_eta {
        assert_eq!(
            on.eta[k], off.eta[k],
            "η[{k}] must be bit-identical on a unimodal subject: on {} vs off {}",
            on.eta[k], off.eta[k]
        );
    }
}

/// `ebe_prior_maha` is the runaway guard's detector: the squared Mahalanobis distance
/// `ηᵀΩ⁻¹η` under the prior. Pinned on a diagonal Ω where the value is hand-computable,
/// and on the non-finite η a diverged search can return — which must report `INFINITY`
/// so it *trips* the guard rather than comparing false against the threshold.
#[test]
fn ebe_prior_maha_is_the_omega_inverse_quadratic_form() {
    let omega =
        crate::types::OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["A".into(), "B".into()]);
    // (0.4²/0.04) + (0.9²/0.09) = 4 + 9 = 13 — i.e. 2 SD and 3 SD.
    assert!((ebe_prior_maha(&[0.4, 0.9], &omega) - 13.0).abs() < 1e-9);
    assert_eq!(ebe_prior_maha(&[0.0, 0.0], &omega), 0.0);
    assert!(ebe_prior_maha(&[f64::NAN, 0.0], &omega).is_infinite());
    assert!(ebe_prior_maha(&[f64::INFINITY, 0.0], &omega).is_infinite());
}

/// Subject 7 of the #958 gist reproducer (`synthetic_tmdd_init_sqrt`): a quasi-steady-state
/// target-binding ODE whose terminal drug sample is `4.7e-5` while the prediction at η = 0 is
/// ~`7.5e-9`. Under proportional error that row's variance is clamped to `MIN_VARIANCE`
/// (1e-12), so its residual term `(y−f)²/R` amplifies the ODE solver's local error
/// (`ode_abstol = 1e-9`) by ~1e8 — and the **finite-difference** inner gradient, which
/// differences that objective, comes back with the wrong *sign*.
///
/// The inner BFGS then marched to η ≈ `[2.49, 16.20, 14.52]` — `ηᵀΩ⁻¹η ≈ 9000`, ~80 prior SDs
/// on two axes — and *certified* it, because a noise-driven search satisfies the
/// objective-stall stop exactly like a converged one (the objective stops improving and the
/// gradient norm plateaus). Nothing downstream re-checked it, so `gradient = fd` and
/// `gradient = auto` reported first-evaluation objectives of `6082.24` and `−5.48` for the
/// same model at the same estimates, making ΔOFV model selection gradient-path-dependent.
///
/// This EBE is **not** multimodal — eight seeds under the analytic gradient (itself verified
/// against FD of the predictor to 2.6e-7) all reach the same mode — so the two routes must
/// return it. Fails without the runaway guard: the FD route returns the runaway instead.
#[test]
fn fd_inner_ebe_runaway_on_floored_row_is_recovered() {
    let model_src = "[parameters]\n  theta KEL(0.05, 0.001, 5.0)\n  theta VC(3, 0.1, 100.0)\n  theta KSS(10, 0.1, 500.0)\n  theta KINT(0.5, 0.01, 50.0)\n  theta KDEG(0.05, 0.001, 5.0)\n  theta RBASE(20, 0.5, 500.0)\n  omega IIV_KEL   ~ 0.09\n  omega IIV_VC    ~ 0.04\n  omega IIV_RBASE ~ 0.09\n  sigma PROP_DRUG ~ 0.15 (sd)\n  sigma PROP_TGT  ~ 0.20 (sd)\n[individual_parameters]\n  KEL   = KEL * exp(IIV_KEL)\n  VC    = VC * exp(IIV_VC)\n  RBASE = RBASE * exp(IIV_RBASE)\n  KSS   = KSS\n  KINT  = KINT\n  KDEG  = KDEG\n  KSYN  = RBASE * KDEG\n[structural_model]\n  ode(states=[CENT, RTOT])\n[odes]\n  init(RTOT) = RBASE\n  CT = CENT / VC\n  RT = RTOT\n  BB = CT - RT - KSS\n  CF = 0.5 * (BB + sqrt(BB*BB + 4*KSS*CT))\n  FB = CF / (KSS + CF)\n  d/dt(CENT) = -KEL * CF * VC - RT * KINT * FB * VC\n  d/dt(RTOT) = KSYN - KDEG * RT - (KINT - KDEG) * RT * FB\n[scaling]\n  y[CMT=1] = CENT / VC\n  y[CMT=3] = RTOT\n[error_model]\n  CMT=1: DV ~ proportional(PROP_DRUG)\n  CMT=3: DV ~ proportional(PROP_TGT)\n[fit_options]\n  method     = focei\n  ode_reltol = 1e-9\n  ode_abstol = 1e-9\n";

    // (time, cmt, DV) exactly as the gist's subject 7 — the `112 / cmt 1 / 4.7e-05` row is
    // the one whose proportional variance floors.
    let rows: [(f64, usize, f64); 23] = [
        (0.083, 1, 183.756742),
        (0.083, 3, 15.161013),
        (0.25, 1, 204.703415),
        (0.25, 3, 14.124375),
        (1.0, 1, 194.836064),
        (1.0, 3, 11.059003),
        (3.0, 1, 189.142239),
        (3.0, 3, 5.061678),
        (7.0, 1, 112.589072),
        (7.0, 3, 2.117357),
        (14.0, 1, 121.725206),
        (14.0, 3, 1.917394),
        (28.0, 1, 72.126391),
        (28.0, 3, 2.068264),
        (42.0, 1, 25.918106),
        (42.0, 3, 2.43914),
        (56.0, 1, 15.509434),
        (56.0, 3, 3.649646),
        (84.0, 1, 0.161923),
        (84.0, 3, 12.591933),
        (112.0, 1, 4.7e-05),
        (112.0, 3, 18.294563),
        (126.0, 3, 17.544296),
    ];
    let n = rows.len();
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "7".into(),
        doses: vec![DoseEvent::new(0.0, 600.0, 1, 0.0, false, 0.0)],
        obs_times: rows.iter().map(|r| r.0).collect(),
        obs_raw_times: Vec::new(),
        observations: rows.iter().map(|r| r.2).collect(),
        obs_cmts: rows.iter().map(|r| r.1).collect(),
        covariates: HashMap::new(),
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
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };

    let base = crate::parser::model_parser::parse_model_string(model_src)
        .expect("QSS TMDD model with a parameter-dependent init parses");
    let params = base.default_params.clone();

    // Precondition: the terminal drug row really is at the variance floor at η = 0. Without
    // this the test would pass vacuously if the model or data ever drifted out of the regime
    // that produces the runaway.
    let preds = crate::pk::compute_predictions_with_tv(&base, &subject, &params.theta, &[0.0; 3]);
    assert!(
        base.residual_variance_at(1, preds[20], &params.sigma.values) <= 1e-12,
        "terminal drug prediction {} must drive its proportional variance to the floor",
        preds[20]
    );

    let solve = |gm: GradientMethod| -> Vec<f64> {
        let mut m = crate::parser::model_parser::parse_model_string(model_src).unwrap();
        m.gradient_method = gm;
        find_ebe(&m, &subject, &params, 50, 1e-5, None, None, 0)
            .eta
            .iter()
            .copied()
            .collect()
    };
    let eta_auto = solve(GradientMethod::Auto);
    let eta_fd = solve(GradientMethod::Fd);

    // Both routes must land in the prior-plausible region — the runaway sat at ~9000.
    for (label, eta) in [("auto", &eta_auto), ("fd", &eta_fd)] {
        let maha = ebe_prior_maha(eta, &params.omega);
        assert!(
            maha < RUNAWAY_EBE_MAHA_PER_ETA * base.n_eta as f64,
            "{label} EBE {eta:?} is a prior runaway (ηᵀΩ⁻¹η = {maha})"
        );
    }
    // …and on the *same* mode, since this subject's individual objective is unimodal.
    for k in 0..base.n_eta {
        assert!(
            (eta_auto[k] - eta_fd[k]).abs() < 1e-3,
            "η[{k}]: analytic {} vs FD {} — the FOCEI objective must not depend on the \
             gradient route",
            eta_auto[k],
            eta_fd[k]
        );
    }
}

/// A subject with an `EVID 3/4` reset makes `cacheable_schedule` return `Some`; a plain
/// subject with neither a reset nor a time-varying covariate makes it return `None` (see
/// `cacheable_schedule`'s guard). `run_inner_loop_warm_map_cached` must produce results
/// bit-identical to the uncached `run_inner_loop_warm` in both cases — including when
/// `build_schedule_cache`'s per-subject `Some`/`None` entries are mixed in the same
/// population, which is the scenario that would expose an index misalignment between the
/// cache and `population.subjects`.
#[test]
fn cached_schedule_inner_loop_matches_uncached() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
    )
    .expect("parse one_cpt_iv model");
    let params = &model.default_params;

    let obs_times = vec![1.0, 4.0, 8.0, 24.0];
    let n = obs_times.len();
    let no_reset_subject = Subject {
        obs_rgroup: Vec::new(),
        id: "no-reset".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: obs_times.clone(),
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 5.0, 3.0, 1.0],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
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
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };
    let reset_subject = Subject {
        obs_rgroup: Vec::new(),
        id: "reset".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 4.0, 8.0, 13.0, 16.0, 20.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 5.0, 3.0, 8.0, 5.0, 3.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: vec![12.0],
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1; 6],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };
    assert!(!no_reset_subject.has_resets());
    assert!(reset_subject.has_resets());

    let population = Population {
        subjects: vec![
            no_reset_subject.clone(),
            reset_subject.clone(),
            no_reset_subject,
            reset_subject,
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let schedules = build_schedule_cache(&model, &population);
    assert!(schedules[0].is_none());
    assert!(schedules[1].is_some());
    assert!(schedules[2].is_none());
    assert!(schedules[3].is_some());

    let uncached = run_inner_loop_warm(&model, &population, params, 50, 1e-8, None, None, 0, 0);
    let cached = run_inner_loop_warm_map_cached(
        &model,
        &population,
        params,
        50,
        1e-8,
        None,
        None,
        0,
        0,
        &schedules,
        InnerSolvePolicy::default(),
        None,
        |_, _| (),
    );

    for i in 0..population.subjects.len() {
        assert_eq!(
            uncached.0[i].as_slice(),
            cached.0[i].as_slice(),
            "subject {i}: η̂ diverged between the cached and uncached inner loop"
        );
        assert_eq!(
            uncached.1[i].as_slice(),
            cached.1[i].as_slice(),
            "subject {i}: Jacobian diverged between the cached and uncached inner loop"
        );
    }
    assert_eq!(uncached.2.n_unconverged, cached.2.n_unconverged);
    assert_eq!(uncached.2.n_fallback, cached.2.n_fallback);
    assert_eq!(uncached.2.n_start_rejected, cached.2.n_start_rejected);
}

// ── #1519: the flip-flop reroute must switch the inner stall stop on ────────────────
//
// A transit closed form is rerouted to its ODE twin whenever `ke = CL/V` reaches the
// tilting abscissa `KTR = (n+1)/MTT` (`crate::pk::effective_model_for_eval`). That reroute
// is decided per `(θ, η)`, so `CompiledModel::effective_for` — which sees only the
// subject-static reroutes — cannot see it, and the objective-stall stop was left off for
// exactly the subjects whose objective had become an RK45 integration.

/// A transit model whose `[parameters]` block is spelled by the caller, so the only thing
/// that differs between the two sides of the flip-flop gate is `(TVCL, TVV, TVMTT, TVN)`.
fn flip_flop_transit_model(cl: f64, v: f64, mtt: f64, n: f64) -> CompiledModel {
    let src = format!(
        "[parameters]
  theta TVCL({cl}, 0.1, 100.0)
  theta TVV({v}, 5.0, 500.0)
  theta TVMTT({mtt}, 0.05, 24.0)
  theta TVN({n}, 0.0, 30.0)
  omega ETA_CL ~ 0.307804
  omega ETA_V  ~ 0.00001
  sigma PROP_ERR ~ 1.161107 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"
    );
    crate::parser::model_parser::parse_model_string(&src)
        .expect("1-cpt transit model with a proportional error parses")
}

/// Subject 2 of `data/datsim_oral.csv`, verbatim — one of the 16 subjects that #1519
/// measured declining to the FD outer-gradient salvage on `examples/one_cpt_transit.ferx`.
fn flip_flop_transit_subject() -> Subject {
    let rows: &[(f64, f64)] = &[
        (1.1, 4.843084),
        (4.29, 3.051025),
        (15.06, -3.107715),
        (27.78, -12.149598),
    ];
    let n = rows.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: "2".into(),
        doses: vec![DoseEvent::new(0.0, 10000.0, 1, 0.0, false, 0.0)],
        obs_times: rows.iter().map(|r| r.0).collect(),
        obs_raw_times: Vec::new(),
        observations: rows.iter().map(|r| r.1).collect(),
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
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
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    }
}

/// The gate itself, asserted on **both** sides in one test so forcing either branch reddens
/// it — and with the straddle asserted on the underlying predicate, so the pair cannot
/// silently drift onto one side of the abscissa and become a tautology.
///
/// The regression: `inner_stall_enabled` keys on `CompiledModel::effective_for`, which is
/// blind to the parameter-dependent flip-flop reroute. That blindness is asserted directly
/// (it reads `false` on *both* sides), so a reader cannot mistake the old spelling for one
/// that merely needed different parameters.
#[test]
fn inner_stall_enabled_at_straddles_the_flip_flop_abscissa() {
    // ke = 5/50 = 0.1 against KTR = (3+1)/1 = 4 — in domain, closed form.
    let in_domain = flip_flop_transit_model(5.0, 50.0, 1.0, 3.0);
    // ke = 80/110 = 0.727 against KTR = (0+1)/24 = 0.0417 — flip-flop, ODE twin.
    let flipped = flip_flop_transit_model(80.0, 110.0, 24.0, 0.0);
    let subject = flip_flop_transit_subject();
    let eta = [0.0, 0.0];

    // The straddle, on the predicate the reroute is built from.
    assert!(
        !crate::pk::absorption_flip_flop_at(
            &in_domain,
            &subject,
            &in_domain.default_params.theta,
            &eta
        ),
        "fixture drift: the in-domain arm must satisfy ke < KTR"
    );
    assert!(
        crate::pk::absorption_flip_flop_at(&flipped, &subject, &flipped.default_params.theta, &eta),
        "fixture drift: the flipped arm must violate ke < KTR"
    );

    // Neither model carries an `[odes]` block, and the subject has no TV covariate and no
    // occasion, so the subject-static reroute reads `false` on both sides — it is not that it
    // needed better parameters, it cannot see this reroute at all.
    assert!(!inner_stall_enabled(&in_domain, &subject));
    assert!(!inner_stall_enabled(&flipped, &subject));

    // Both sides of the gate under test.
    assert!(
        !inner_stall_enabled_at(&in_domain, &subject, &in_domain.default_params.theta, &eta),
        "an in-domain closed-form subject has an exact objective and must keep `gnorm < tol`"
    );
    assert!(
        inner_stall_enabled_at(&flipped, &subject, &flipped.default_params.theta, &eta),
        "a flip-flop subject's objective is an RK45 solve and must get the stall stop"
    );
}

/// The behavioural regression, on the same subject and parameters the measurement used: a
/// flip-flop-rerouted subject's inner solve must terminate on the objective-stall stop
/// rather than exhausting its budget and buying the Nelder-Mead recovery -- which does not
/// merely cost time, it *moves* the returned eta_hat to a worse point.
///
/// Measured on this fixture, cold start, `inner_tol = 1e-6`:
///
/// | | `used_fallback` | gradient L2 norm at the returned eta_hat |
/// |---|---|---|
/// | with the flip-flop-aware gate | `false` | **1.106e-4** |
/// | gate reverted to `effective_for` | `true` | **2.368e-1** (2100x worse) |
///
/// so the bound below is set at `1e-3` -- 9x headroom over the realised 1.106e-4, and 236x
/// below the 2.368e-1 the mutation produces. `used_fallback` is asserted as well because it
/// names the mechanism; the norm is asserted because a future recovery that happened not to
/// set that flag would still have to land somewhere sane.
///
/// The in-domain arm is asserted in the same test so a mutation that switches the stall stop
/// on *unconditionally* -- which would silence the flipped arm just as well -- reddens it
/// too: an exact closed-form objective must still stop on `gnorm < tol`, not on a plateau.
#[test]
fn a_flip_flop_subject_stops_instead_of_falling_back_to_nelder_mead() {
    let subject = flip_flop_transit_subject();
    let gnorm_at = |m: &CompiledModel, p: &crate::types::ModelParameters, eta: &[f64]| -> f64 {
        let g = analytic_eta_nll_gradient(m, &subject, &p.theta, eta, &p.omega, &p.sigma.values)
            .expect("the analytic eta-gradient is in scope for a 1-cpt transit subject");
        let n = grad_norm_metric(&g, None);
        assert!(
            n.is_finite(),
            "the gradient at the returned eta_hat must be finite"
        );
        n
    };

    let flipped = flip_flop_transit_model(80.0, 110.0, 24.0, 0.0);
    let flipped_params = flipped.default_params.clone();
    let res = find_ebe(
        &flipped,
        &subject,
        &flipped_params,
        200,
        1e-6,
        None,
        None,
        0,
    );
    assert!(
        !res.used_fallback,
        "the flip-flop subject's inner solve ran out of budget and recovered with \
         Nelder-Mead (eta_hat = {:?}); the stall stop did not fire",
        res.eta.as_slice()
    );
    assert!(
        res.converged,
        "the stall stop must report a terminated solve"
    );
    let flipped_gnorm = gnorm_at(&flipped, &flipped_params, res.eta.as_slice());
    assert!(
        flipped_gnorm < 1e-3,
        "the stall stop must return the BFGS descent point, not a Nelder-Mead excursion \
         (norm = {flipped_gnorm:e}, realised with the fix: 1.106e-4)"
    );

    // The other side of the gate: an in-domain closed-form subject is exact, reaches
    // `gnorm < tol`, and must not be handed the stall stop by this change.
    let in_domain = flip_flop_transit_model(5.0, 50.0, 1.0, 3.0);
    let in_params = in_domain.default_params.clone();
    let exact = find_ebe(&in_domain, &subject, &in_params, 200, 1e-6, None, None, 0);
    assert!(
        exact.converged && !exact.used_fallback,
        "an in-domain closed-form subject must still converge on the exact `gnorm < tol` test"
    );
    let exact_gnorm = gnorm_at(&in_domain, &in_params, exact.eta.as_slice());
    assert!(
        exact_gnorm < 1e-6,
        "the in-domain arm must still stop on stationarity, not on a plateau \
         (norm = {exact_gnorm:e})"
    );
}

/// The crossing case, found in review of PR #1521: a solve that **starts in domain and
/// enters the flip-flop regime partway through the search**.
///
/// The first cut of this fix read the reroute once, at the solve's start point, so a
/// subject whose `eta` walks across the abscissa mid-search kept the exact `gnorm < tol`
/// criterion against an objective that had already become an RK45 integration — the very
/// thing the fix exists to stop. Both of the earlier tests use cold starts that stay on
/// their initial side of the boundary, so neither could see it.
///
/// Fixture: `KTR = (0+1)/24 = 0.041667`. At `TVCL = 4.0` the start point sits at
/// `ke = 4.0/110 = 0.0364` — in domain — and the search walks `ETA_CL` to `+0.255`, where
/// `ke = 0.0364 e^{0.255} = 0.0470 > KTR`. `TVCL = 4.5` (`ke_0 = 0.0409`, within 2% of the
/// abscissa) is the tighter of the pair. Both were the reviewer's own reproduction.
///
/// | | `used_fallback` | returned `ETA_CL` |
/// |---|---|---|
/// | latched per-iterate gate (`TVCL = 4.0` / `4.5`) | `false` / `false` | 0.25499040 / 0.27412893 |
/// | start-point gate only | **`true`** / **`true`** | 0.25499040 / 0.27412893 |
///
/// **`used_fallback` is the assertion that dies under that mutation; the `eta` bound does
/// not** — the Nelder-Mead recovery returns the same point here, to 2e-16. The `eta` bound
/// is therefore not a second kill but a guard on the cheaper exit: it pins that stopping on
/// the stall instead of grinding to the budget did not move the answer, measured against a
/// forced-stall-on reference. That reference is the same production `find_ebe` on the same
/// model and theta, warm-started at `eta = [0.40, 0]` — already past the abscissa, asserted
/// below — so its start-point gate is on from iteration 0 whatever the latch does.
///
/// Realised max-abs agreement against that reference: **3.708e-7** (`TVCL = 4.0`) and
/// **2.660e-6** (`TVCL = 4.5`). The bound is `1e-5`: 3.8x headroom over the worse of the
/// two. The residual is the two solves' different start points, not the stall policy.
#[test]
fn a_solve_that_crosses_the_flip_flop_abscissa_mid_search_still_stalls() {
    let subject = flip_flop_transit_subject();
    // Already past the abscissa at every `TVCL` below, so the reference solve has the stall
    // stop on from its own start point.
    let past_the_boundary = [0.40_f64, 0.0];

    for tvcl in [4.0_f64, 4.5] {
        let m = flip_flop_transit_model(tvcl, 110.0, 24.0, 0.0);
        let p = m.default_params.clone();
        let cold = vec![0.0; m.n_eta];

        let res = find_ebe(&m, &subject, &p, 200, 1e-6, None, None, 0);

        // The straddle, asserted so the fixture cannot silently stop crossing.
        assert!(
            !crate::pk::absorption_flip_flop_at(&m, &subject, &p.theta, &cold),
            "TVCL = {tvcl}: fixture drift, the start point must be *in* the closed form's \
             domain or this is not the crossing case"
        );
        assert!(
            crate::pk::absorption_flip_flop_at(&m, &subject, &p.theta, res.eta.as_slice()),
            "TVCL = {tvcl}: fixture drift, the solve must end *past* the abscissa \
             (eta_hat = {:?})",
            res.eta.as_slice()
        );

        assert!(
            !res.used_fallback,
            "TVCL = {tvcl}: a solve that entered the ODE regime mid-search still ran out of \
             budget and recovered with Nelder-Mead; the stall policy did not follow the \
             iterate (eta_hat = {:?})",
            res.eta.as_slice()
        );
        assert!(
            res.converged,
            "TVCL = {tvcl}: the stall stop must terminate the solve"
        );

        // Forced-stall-on reference: same model, same theta, start already rerouted.
        assert!(
            crate::pk::absorption_flip_flop_at(&m, &subject, &p.theta, &past_the_boundary),
            "TVCL = {tvcl}: fixture drift, the reference start must itself be rerouted or it \
             is not a forced-stall-on run"
        );
        let forced = find_ebe(
            &m,
            &subject,
            &p,
            200,
            1e-6,
            Some(&past_the_boundary),
            None,
            0,
        );
        assert!(
            !forced.used_fallback,
            "TVCL = {tvcl}: the reference run is supposed to have the stall stop on throughout"
        );
        let worst = res
            .eta
            .iter()
            .zip(forced.eta.iter())
            .fold(0.0f64, |acc, (a, b)| {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "TVCL = {tvcl}: a non-finite eta cannot be folded into a max"
                );
                acc.max((a - b).abs())
            });
        assert!(
            worst < 1e-5,
            "TVCL = {tvcl}: the crossing solve landed {worst:e} from the forced-stall-on \
             reference (realised on this fixture: 3.708e-7 and 2.660e-6)"
        );
    }
}

/// Non-interaction FOCE EBE search: the residual variance held at `f(η = 0)` (#1722).
mod frozen_ebe_variance {
    use super::*;
    use crate::stats::likelihood::{
        individual_nll_into_prepared_with_schedule_frozen, individual_nll_iov_frozen,
        population_variance_preds, population_variance_preds_iov,
    };
    use crate::types::EstimationMethod;

    /// `warfarin_iov` at NONMEM's FOCE estimates (σ ≈ 19%, `nonmem_anchor/foce_ebe_freeze_iov`).
    pub(crate) fn iov_fixture() -> (CompiledModel, Population, ModelParameters) {
        let prep = crate::prepare_run("examples/warfarin_iov.ferx", Some("data/warfarin_iov.csv"))
            .expect("prepare");
        let mut p = prep.init_params.clone();
        p.theta = vec![0.315410, 8.38248, 2.64431];
        p.omega =
            OmegaMatrix::from_diagonal(&[0.446654, 0.0125169, 1.07769], p.omega.eta_names.clone());
        let names = p.omega_iov.as_ref().unwrap().eta_names.clone();
        p.omega_iov = Some(OmegaMatrix::from_diagonal(&[0.04258], names));
        p.sigma.values = vec![0.0370648_f64.sqrt()];
        (prep.parsed.model, prep.population, p)
    }

    /// The non-IOV twin (`nonmem_anchor/foce_ebe_freeze_noiov_fit.ferx`), at NONMEM's
    /// estimates (σ ≈ 21%), optionally with the error model swapped.
    pub(crate) fn noiov_fixture(
        error: Option<&str>,
    ) -> (CompiledModel, Population, ModelParameters) {
        let mut src = std::fs::read_to_string("nonmem_anchor/foce_ebe_freeze_noiov_fit.ferx")
            .expect("model file");
        if let Some(e) = error {
            let sigmas = if e.contains("additive") {
                "sigma ADD_ERR ~ 0.8 (sd)"
            } else {
                "sigma PROP_ERR ~ 0.2 (sd)\n  sigma ADD_ERR ~ 0.3 (sd)"
            };
            src = src
                .replace("DV ~ proportional(PROP_ERR)", e)
                .replace("sigma PROP_ERR ~ 0.2 (sd)", sigmas);
        }
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("data/warfarin_iov.csv"),
            None,
            None,
        )
        .expect("data");
        let mut p = model.default_params.clone();
        p.theta = vec![0.143657, 8.83827, 1.22791];
        p.omega = OmegaMatrix::from_diagonal(
            &[0.0831907, 0.00812141, 0.0233209],
            p.omega.eta_names.clone(),
        );
        p.sigma.values[0] = 0.0450339_f64.sqrt();
        (model, pop, p)
    }

    fn nll(
        model: &CompiledModel,
        s: &Subject,
        p: &ModelParameters,
        eta: &[f64],
        frozen: Option<&[f64]>,
    ) -> f64 {
        let keys = model.error_spec.obs_keys(s);
        let mult = model.ruv_obs_mult(s, &p.theta);
        individual_nll_into_prepared_with_schedule_frozen(
            model,
            s,
            &p.theta,
            eta,
            &p.omega,
            &p.sigma.values,
            model.residual_cov_at(p),
            &mut pk::EventPkParams::default(),
            None,
            keys.as_ref(),
            mult.as_deref(),
            &mut Vec::new(),
            &mut DVector::zeros(eta.len()),
            &mut DVector::zeros(eta.len()),
            frozen,
        )
    }

    /// Stacked `[η, κ₁, κ₂]` for an IOV subject.
    fn nll_iov(
        model: &CompiledModel,
        s: &Subject,
        p: &ModelParameters,
        b: &[f64],
        frozen: Option<&[f64]>,
    ) -> f64 {
        let (n_eta, n_k) = (model.n_eta, model.n_kappa);
        let kappas: Vec<Vec<f64>> = b[n_eta..].chunks(n_k).map(|c| c.to_vec()).collect();
        individual_nll_iov_frozen(
            model,
            s,
            &p.theta,
            &b[..n_eta],
            &kappas,
            &p.omega,
            p.omega_iov.as_ref(),
            &p.sigma.values,
            &mut pk::EventPkParams::default(),
            None,
            frozen,
        )
    }

    fn fd(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|k| {
                let h = 1e-6 * (1.0 + x[k].abs());
                let (mut a, mut b) = (x.to_vec(), x.to_vec());
                a[k] += h;
                b[k] -= h;
                (f(&a) - f(&b)) / (2.0 * h)
            })
            .collect()
    }

    fn assert_close(label: &str, got: &[f64], want: &[f64], rel: f64) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite() && w.is_finite(), "{label}[{k}]: {g} / {w}");
            assert!(
                (g - w).abs() <= rel * (1.0 + w.abs()),
                "{label}[{k}]: analytic {g} vs FD {w}"
            );
        }
    }

    /// T3 (objective): the frozen objective is the closed form
    /// `½[ηᵀΩ⁻¹η + log|Ω| + Σ (y − f(η))²/V⁰ + log V⁰]` with `V⁰ = (f(0)·σ)²`; it equals
    /// the conditional objective at `η = 0` (where `f(η) = f(0)`) and differs from it away
    /// from zero. Both halves in one test, so neither "ignore the frozen point" nor "freeze
    /// unconditionally" can pass it.
    #[test]
    fn frozen_objective_is_the_closed_form_and_straddles_the_conditional() {
        let (model, pop, p) = noiov_fixture(None);
        let s = &pop.subjects[9];
        let f0 = population_variance_preds(&model, s, &p.theta, 3).expect("proportional ⇒ Some");
        let zero = [0.0; 3];
        let at0 = (
            nll(&model, s, &p, &zero, Some(&f0)),
            nll(&model, s, &p, &zero, None),
        );
        assert!(
            (at0.0 - at0.1).abs() <= 1e-12 * at0.1.abs(),
            "η = 0: frozen {} vs conditional {}",
            at0.0,
            at0.1
        );
        let eta = [0.3, -0.2, 0.4];
        let frozen = nll(&model, s, &p, &eta, Some(&f0));
        let cond = nll(&model, s, &p, &eta, None);
        let preds = crate::pk::compute_predictions_with_tv(&model, s, &p.theta, &eta);
        let sig = p.sigma.values[0];
        let mut data = 0.0;
        for ((y, f), f0j) in s.observations.iter().zip(&preds).zip(&f0) {
            let v0 = (f0j * sig).powi(2);
            data += (y - f).powi(2) / v0 + v0.ln();
        }
        let prior: f64 = (0..3)
            .map(|k| eta[k] * eta[k] / p.omega.matrix[(k, k)])
            .sum();
        let want = 0.5 * (prior + p.omega.log_det + data);
        assert!(
            (frozen - want).abs() <= 1e-10 * want.abs(),
            "frozen objective {frozen} vs closed form {want}"
        );
        assert!(
            (frozen - cond).abs() > 1e-2,
            "fixture must separate the conventions at η ≠ 0 (frozen {frozen}, conditional {cond})"
        );
        // IOV twin: same straddle on the stacked [η, κ₁, κ₂].
        let (model, pop, p) = iov_fixture();
        let s = &pop.subjects[9];
        let f0 = population_variance_preds_iov(&model, s, &p.theta, 3, 1, 2).expect("Some");
        let zero = [0.0; 5];
        let (a, b) = (
            nll_iov(&model, s, &p, &zero, Some(&f0)),
            nll_iov(&model, s, &p, &zero, None),
        );
        assert!((a - b).abs() <= 1e-12 * b.abs(), "IOV η = 0: {a} vs {b}");
        let bvec = [0.3, -0.05, -0.6, 0.15, -0.1];
        let (a, b) = (
            nll_iov(&model, s, &p, &bvec, Some(&f0)),
            nll_iov(&model, s, &p, &bvec, None),
        );
        assert!(
            (a - b).abs() > 1e-2,
            "IOV fixture must separate the conventions ({a}, {b})"
        );
    }

    /// T3 (solve): under `Population` every subject's EBE is a stationary point of the
    /// *frozen* objective, and the policy moves it against the conditional solve. A policy
    /// that never reaches the solve leaves the two EBE sets identical; a solve that ignores
    /// the frozen point stops at the conditional mode, where the frozen gradient is O(1).
    #[test]
    fn population_policy_solves_the_frozen_objective() {
        for (label, (model, pop, p)) in [("noiov", noiov_fixture(None)), ("iov", iov_fixture())] {
            let mut moved = 0.0_f64;
            for s in &pop.subjects {
                let solve = |v| find_ebe_with_variance(&model, s, &p, 500, 1e-10, None, None, 0, v);
                let (pop_e, cond_e) = (
                    solve(EbeVariance::Population),
                    solve(EbeVariance::Conditional),
                );
                assert!(pop_e.converged, "{label} {}: frozen solve converged", s.id);
                let stack = |e: &EbeResult| {
                    let mut b = e.eta.as_slice().to_vec();
                    for k in &e.kappas {
                        b.extend_from_slice(k.as_slice());
                    }
                    b
                };
                let (bp, bc) = (stack(&pop_e), stack(&cond_e));
                for (x, y) in bp.iter().zip(&bc) {
                    assert!(x.is_finite() && y.is_finite());
                    moved = moved.max((x - y).abs());
                }
                let g = if model.n_kappa > 0 {
                    let f0 = population_variance_preds_iov(&model, s, &p.theta, 3, 1, 2).unwrap();
                    fd(|b| nll_iov(&model, s, &p, b, Some(&f0)), &bp)
                } else {
                    let f0 = population_variance_preds(&model, s, &p.theta, 3).unwrap();
                    fd(|b| nll(&model, s, &p, b, Some(&f0)), &bp)
                };
                let gmax = g.iter().fold(0.0_f64, |m, v| {
                    assert!(v.is_finite());
                    m.max(v.abs())
                });
                eprintln!("MEASURE {label} {}: frozen |g|max at EBE {gmax:.3e}", s.id);
                assert!(
                    gmax < 1e-4,
                    "{label} {}: frozen-objective gradient {gmax:e} at the EBE",
                    s.id
                );
            }
            eprintln!("MEASURE {label}: max |EBE Population − Conditional| {moved:.3e}");
            assert!(
                moved > 1e-2,
                "{label}: Population EBEs must differ from Conditional ({moved:e})"
            );
        }
    }

    /// T4: the frozen analytic η-gradient against central FD of the frozen objective —
    /// proportional, combined, and IOV — at a point where it differs from the conditional
    /// gradient, so dropping the `∂V/∂η` freeze (or the frozen point) cannot pass.
    #[test]
    fn frozen_analytic_inner_gradient_matches_fd() {
        for (label, err) in [
            ("proportional", None),
            ("combined", Some("DV ~ combined(PROP_ERR, ADD_ERR)")),
        ] {
            let (model, pop, p) = noiov_fixture(err);
            assert!(model.error_spec.has_f_dependent_variance());
            for s in pop.subjects.iter().take(4) {
                let f0 = population_variance_preds(&model, s, &p.theta, 3).unwrap();
                let eta = [0.25, -0.15, 0.35];
                let keys = model.error_spec.obs_keys(s);
                let grad = |frozen: Option<&[f64]>| {
                    analytic_eta_nll_gradient_with_schedule(
                        &model,
                        s,
                        &p.theta,
                        &eta,
                        &p.omega,
                        &p.sigma.values,
                        model.residual_cov_at(&p),
                        None,
                        None,
                        keys.as_ref(),
                        &mut Vec::new(),
                        &mut DVector::zeros(3),
                        &mut DVector::zeros(3),
                        frozen,
                    )
                    .expect("in scope")
                };
                let (g, gc) = (grad(Some(&f0)), grad(None));
                assert_close(
                    label,
                    &g,
                    &fd(|e| nll(&model, s, &p, e, Some(&f0)), &eta),
                    1e-5,
                );
                let sep = g
                    .iter()
                    .zip(&gc)
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    sep > 1e-2,
                    "{label} {}: frozen vs conditional gradient {sep:e}",
                    s.id
                );
            }
        }
        let (model, pop, p) = iov_fixture();
        let omega_iov = p.omega_iov.as_ref().unwrap();
        for s in pop.subjects.iter().take(4) {
            let f0 = population_variance_preds_iov(&model, s, &p.theta, 3, 1, 2).unwrap();
            let b = [0.25, -0.05, -0.4, 0.12, -0.08];
            let grad = |frozen: Option<&[f64]>| {
                analytic_eta_nll_gradient_iov_frozen(
                    &model,
                    s,
                    &crate::stats::stacked_layout::StackedLayout::sens_of(&model, s).unwrap(),
                    &p.theta,
                    &b,
                    &p.omega,
                    omega_iov,
                    &p.sigma.values,
                    None,
                    frozen,
                )
                .expect("IOV in scope")
            };
            let (g, gc) = (grad(Some(&f0)), grad(None));
            assert_close(
                "iov",
                &g,
                &fd(|x| nll_iov(&model, s, &p, x, Some(&f0)), &b),
                1e-5,
            );
            let sep = g
                .iter()
                .zip(&gc)
                .fold(0.0_f64, |m, (a, c)| m.max((a - c).abs()));
            assert!(
                sep > 1e-2,
                "iov {}: frozen vs conditional gradient {sep:e}",
                s.id
            );
        }
    }

    /// T4, dense-`R` branch (`block_sigma`, #627): under a frozen `R` the gradient is
    /// `−aₖᵀR⁻¹r + Ω⁻¹η` — no `∂R/∂η` trace term.
    #[test]
    fn frozen_dense_residual_inner_gradient_matches_fd() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = foce\n",
        )
        .expect("parse");
        assert!(!model.residual_correlations.is_empty());
        let mut s = Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 5],
            obs_cmts: vec![1; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 5],
            occasions: vec![1; 5],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
            endpoint_covariates: Vec::new(),
            filled_cells: Vec::new(),
        };
        let p = model.default_params.clone();
        let preds = crate::pk::compute_predictions_with_tv(&model, &s, &p.theta, &[0.1]);
        s.observations = preds.iter().map(|v| v * 1.15 + 0.3).collect();
        let f0 = population_variance_preds(&model, &s, &p.theta, 1).unwrap();
        let eta = [-0.2];
        let keys = model.error_spec.obs_keys(&s);
        let grad = |frozen: Option<&[f64]>| {
            analytic_eta_nll_gradient_with_schedule(
                &model,
                &s,
                &p.theta,
                &eta,
                &p.omega,
                &p.sigma.values,
                model.residual_cov_at(&p),
                None,
                None,
                keys.as_ref(),
                &mut Vec::new(),
                &mut DVector::zeros(1),
                &mut DVector::zeros(1),
                frozen,
            )
            .expect("dense in scope")
        };
        let (g, gc) = (grad(Some(&f0)), grad(None));
        assert_close(
            "dense",
            &g,
            &fd(|e| nll(&model, &s, &p, e, Some(&f0)), &eta),
            1e-5,
        );
        assert!(
            (g[0] - gc[0]).abs() > 1e-3,
            "dense: frozen {} vs conditional {}",
            g[0],
            gc[0]
        );
    }

    /// #1081 T6: the inner η-gradient on the dense path under a `correlation` line,
    /// against central FD of the individual NLL — conditional (the `∂R/∂η` trace
    /// term, through the serial `∂R/∂f`) and frozen. Combined error, so the serial
    /// off-diagonal moves with `f`; two arms of 3 and 2 rows, interleaved, at time
    /// gaps 1.5, 6 and 3, so `car1`'s `ρ^Δt` differs from `ar1`'s `ρ^lag`.
    #[test]
    fn serial_residual_inner_gradient_matches_fd() {
        for kind in ["cs", "ar1", "car1"] {
            let model = crate::parser::model_parser::parse_model_string(&format!(
                "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  theta RHO(0.6, -0.4, 0.95) FIX\n  omega ETA_CL ~ 0.09 FIX\n  sigma PROP_ERR ~ 0.2 (sd) FIX\n  sigma ADD_ERR ~ 1.0 (sd) FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation {kind}(RHO) by ARM\n[fit_options]\n  method = focei\n",
            ))
            .expect("parse");
            assert!(model.residual_serial.is_some());
            let arms = [1.0, 2.0, 1.0, 2.0, 1.0];
            let mut s = Subject {
                id: "1".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
                observations: vec![0.0; 5],
                obs_cmts: vec![1; 5],
                obs_covariates: arms
                    .iter()
                    .map(|a| HashMap::from([("ARM".to_string(), *a)]))
                    .collect(),
                cens: vec![0; 5],
                ..Default::default()
            };
            s.obs_rgroup = crate::stats::residual_error::residual_groups(&model, &s);
            assert_eq!(s.obs_rgroup, vec![1, 2, 1, 2, 1]);
            let p = model.default_params.clone();
            let preds = crate::pk::compute_predictions_with_tv(&model, &s, &p.theta, &[0.1]);
            s.observations = preds.iter().map(|v| v * 1.15 + 0.3).collect();
            let f0 = population_variance_preds(&model, &s, &p.theta, 1).unwrap();
            let eta = [-0.2];
            let keys = model.error_spec.obs_keys(&s);
            let grad = |frozen: Option<&[f64]>| {
                analytic_eta_nll_gradient_with_schedule(
                    &model,
                    &s,
                    &p.theta,
                    &eta,
                    &p.omega,
                    &p.sigma.values,
                    model.residual_cov_at(&p),
                    None,
                    None,
                    keys.as_ref(),
                    &mut Vec::new(),
                    &mut DVector::zeros(1),
                    &mut DVector::zeros(1),
                    frozen,
                )
                .expect("dense in scope")
            };
            for frozen in [None, Some(f0.as_slice())] {
                assert_close(
                    kind,
                    &grad(frozen),
                    &fd(|e| nll(&model, &s, &p, e, frozen), &eta),
                    1e-6,
                );
            }
            // ρ is live in the objective: the gradient at ρ = 0.6 is not the one an
            // independent `R` gives.
            let mut p0 = p.clone();
            p0.theta[2] = 0.0;
            let indep = fd(|e| nll(&model, &s, &p0, e, None), &eta);
            assert!(
                (grad(None)[0] - indep[0]).abs() > 1e-3,
                "{kind}: ρ moves the gradient"
            );
        }
    }

    /// T4, M3-censored rows: the censored kernel at `dv_df = 0` and `V = V⁰`.
    #[test]
    fn frozen_m3_inner_gradient_matches_fd() {
        let src = std::fs::read_to_string("examples/warfarin_bloq.ferx")
            .unwrap()
            .replace("sigma PROP_ERR ~ 0.02 (sd)", "sigma PROP_ERR ~ 0.2 (sd)");
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("data/warfarin_bloq.csv"),
            None,
            None,
        )
        .unwrap();
        let s = pop
            .subjects
            .iter()
            .find(|s| s.cens.iter().any(|&c| c != 0))
            .unwrap();
        let p = model.default_params.clone();
        let f0 = population_variance_preds(&model, s, &p.theta, 3).unwrap();
        let eta = [0.12, -0.05, 0.2];
        let keys = model.error_spec.obs_keys(s);
        let grad = |frozen: Option<&[f64]>| {
            analytic_eta_nll_gradient_with_schedule(
                &model,
                s,
                &p.theta,
                &eta,
                &p.omega,
                &p.sigma.values,
                model.residual_cov_at(&p),
                None,
                None,
                keys.as_ref(),
                &mut Vec::new(),
                &mut DVector::zeros(3),
                &mut DVector::zeros(3),
                frozen,
            )
            .expect("M3 in scope")
        };
        let (g, gc) = (grad(Some(&f0)), grad(None));
        assert_close(
            "m3",
            &g,
            &fd(|e| nll(&model, s, &p, e, Some(&f0)), &eta),
            1e-4,
        );
        let sep = g
            .iter()
            .zip(&gc)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(sep > 1e-3, "m3: frozen vs conditional gradient {sep:e}");
    }

    /// `R⁰` is the variance at every random effect zero, `η_ruv` included (`fit` refuses
    /// `iiv_on_ruv` under a non-interaction stage, so this guards the convention, not a
    /// reachable fit): under a frozen variance `η_ruv` moves only the prior — objective
    /// and analytic gradient alike — while the conditional objective's data term moves.
    #[test]
    fn frozen_variance_carries_no_ruv_eta_data_term() {
        let src = std::fs::read_to_string("nonmem_anchor/foce_ebe_freeze_noiov_fit.ferx")
            .unwrap()
            .replace(
                "omega ETA_KA ~ 0.30",
                "omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.10",
            )
            .replace(
                "DV ~ proportional(PROP_ERR)",
                "DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV",
            );
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        assert_eq!(
            model.residual_error_eta,
            Some(3),
            "fixture: η_ruv is the 4th eta"
        );
        let (_, pop, base) = noiov_fixture(None);
        let mut p = model.default_params.clone();
        p.theta = base.theta.clone();
        p.sigma.values = base.sigma.values.clone();
        let s = &pop.subjects[2];
        let f0 = population_variance_preds(&model, s, &p.theta, 4).unwrap();
        let (lo, hi) = ([0.1, -0.1, 0.2, 0.0], [0.1, -0.1, 0.2, 0.3]);
        let prior = 0.5 * 0.3 * 0.3 / p.omega.matrix[(3, 3)];
        let frozen = nll(&model, s, &p, &hi, Some(&f0)) - nll(&model, s, &p, &lo, Some(&f0));
        assert!(
            (frozen - prior).abs() < 1e-9,
            "frozen: Δ {frozen} vs prior-only {prior}"
        );
        let cond = nll(&model, s, &p, &hi, None) - nll(&model, s, &p, &lo, None);
        assert!(
            (cond - prior).abs() > 1e-2,
            "conditional data term must move ({cond})"
        );
        let keys = model.error_spec.obs_keys(s);
        let g = analytic_eta_nll_gradient_with_schedule(
            &model,
            s,
            &p.theta,
            &hi,
            &p.omega,
            &p.sigma.values,
            model.residual_cov_at(&p),
            None,
            None,
            keys.as_ref(),
            &mut Vec::new(),
            &mut DVector::zeros(4),
            &mut DVector::zeros(4),
            Some(&f0),
        )
        .expect("in scope");
        assert_close(
            "ruv",
            &g,
            &fd(|e| nll(&model, s, &p, e, Some(&f0)), &hi),
            1e-5,
        );
    }

    /// The production FOCE path is the **seeded** one (`InnerHessianSeed::GaussNewton`), and
    /// its light seed fuses BFGS's first gradient. That gradient must be the frozen
    /// objective's: handing BFGS the conditional one sent its first step the wrong way, and
    /// on `warfarin_iov` at a mid-fit point 4–8 of 10 subjects then failed BFGS and fell back
    /// to Nelder–Mead at every outer evaluation — noise the outer L-BFGS stalled on, 1.15
    /// OFV above NONMEM's optimum. The unseeded solve (the tests above) cannot see this.
    #[test]
    fn seeded_population_solve_fuses_the_frozen_gradient() {
        let (model, pop, _) = noiov_fixture(None);
        // ferx's stall point before the fix (θ = 0.13736, 8.73095, 1.64160; ω² = 0.099555,
        // 0.006699, 0.233339; σ = 0.193088).
        let mut p = model.default_params.clone();
        p.theta = vec![0.137360, 8.730946, 1.641600];
        p.omega =
            OmegaMatrix::from_diagonal(&[0.099555, 0.006699, 0.233339], p.omega.eta_names.clone());
        p.sigma.values = vec![0.193088];
        let s = &pop.subjects[0];
        let f0 = population_variance_preds(&model, s, &p.theta, 3).unwrap();
        let keys = model.error_spec.obs_keys(s);
        let eta = [0.2, -0.1, 0.3];
        let fused = |frozen: Option<&[f64]>| {
            analytic_inner_seed_hessian(
                &model,
                s,
                &p,
                &eta,
                None,
                None,
                keys.as_ref(),
                &mut Vec::new(),
                false,
                frozen,
            )
            .expect("light seed in scope")
            .1
            .expect("light seed fuses a gradient")
        };
        let want = analytic_eta_nll_gradient_with_schedule(
            &model,
            s,
            &p.theta,
            &eta,
            &p.omega,
            &p.sigma.values,
            model.residual_cov_at(&p),
            None,
            None,
            keys.as_ref(),
            &mut Vec::new(),
            &mut DVector::zeros(3),
            &mut DVector::zeros(3),
            Some(&f0),
        )
        .unwrap();
        let (gf, gc) = (fused(Some(&f0)), fused(None));
        assert_close("fused", &gf, &want, 1e-12);
        let sep = gf
            .iter()
            .zip(&gc)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(sep > 1e-2, "frozen vs conditional fused gradient {sep:e}");

        // The production warm, seeded solve: warm-started from the conditional modes, as an
        // outer evaluation after a step is. Every subject converges on the frozen mode, and
        // none falls back.
        let warm: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| {
                find_ebe_with_variance(
                    &model,
                    s,
                    &p,
                    200,
                    1e-8,
                    None,
                    None,
                    0,
                    EbeVariance::Conditional,
                )
                .eta
            })
            .collect();
        let (etas, _, stats, _) = run_inner_loop_warm_seeded(
            &model,
            &pop,
            &p,
            200,
            1e-6,
            Some(&warm),
            None,
            0,
            0,
            InnerHessianSeed::GaussNewton,
            &crate::types::FitOptions {
                method: EstimationMethod::Foce,
                interaction: false,
                ..Default::default()
            },
        );
        assert_eq!(
            stats.n_fallback, 0,
            "no frozen solve may fall back to Nelder–Mead"
        );
        assert_eq!(stats.n_unconverged, 0, "every frozen solve converges");
        for (s, e) in pop.subjects.iter().zip(&etas) {
            let f0 = population_variance_preds(&model, s, &p.theta, 3).unwrap();
            let g = fd(|b| nll(&model, s, &p, b, Some(&f0)), e.as_slice());
            let gmax = g.iter().fold(0.0_f64, |m, v| {
                assert!(v.is_finite());
                m.max(v.abs())
            });
            assert!(
                gmax < 1e-4,
                "{}: frozen gradient {gmax:e} at the seeded EBE",
                s.id
            );
        }
    }

    /// T7: the EBEs every estimator *reports* under `interaction = false` are modes of the
    /// frozen objective — the Sheiner–Beal marginal they are scored under — and under
    /// `interaction = true` modes of the conditional one. One fixture, one fit per stage, each
    /// asserted separately so a failure names the path: the FOCE final inner loop, the
    /// Gauss–Newton driver, SAEM's final OFV pass, VI's final EBE pass, and the standalone
    /// IMP evaluation stage. Both sides of the gate per method where the method takes both,
    /// so a path that ignores the convention fails one half whichever way it is wired.
    #[test]
    fn reported_ebes_follow_the_marginal_of_every_stage() {
        let (model, pop, p) = noiov_fixture(None);
        let base = crate::types::FitOptions {
            outer_maxiter: 0,
            run_covariance_step: false,
            verbose: false,
            saem_n_exploration: 2,
            saem_n_convergence: 0,
            saem_n_mh_steps: 1,
            saem_seed: Some(7),
            vi_iters: 5,
            ..Default::default()
        };
        let cases: Vec<(&str, crate::types::FitOptions)> = vec![
            (
                "foce",
                crate::types::FitOptions {
                    method: EstimationMethod::Foce,
                    interaction: false,
                    ..base.clone()
                },
            ),
            (
                "focei",
                crate::types::FitOptions {
                    method: EstimationMethod::FoceI,
                    interaction: true,
                    ..base.clone()
                },
            ),
            (
                "gn",
                crate::types::FitOptions {
                    method: EstimationMethod::FoceGn,
                    interaction: false,
                    ..base.clone()
                },
            ),
            (
                "saem",
                crate::types::FitOptions {
                    method: EstimationMethod::Saem,
                    interaction: false,
                    ..base.clone()
                },
            ),
            (
                "saem+inter",
                crate::types::FitOptions {
                    method: EstimationMethod::Saem,
                    interaction: true,
                    ..base.clone()
                },
            ),
            (
                "vi",
                crate::types::FitOptions {
                    method: EstimationMethod::Vi,
                    interaction: false,
                    ..base.clone()
                },
            ),
            (
                "vi+inter",
                crate::types::FitOptions {
                    method: EstimationMethod::Vi,
                    interaction: true,
                    ..base.clone()
                },
            ),
            (
                "imp-eval",
                crate::types::FitOptions {
                    method: EstimationMethod::Imp,
                    imp_eval_only: true,
                    interaction: false,
                    ..base.clone()
                },
            ),
            (
                "imp-eval+inter",
                crate::types::FitOptions {
                    method: EstimationMethod::Imp,
                    imp_eval_only: true,
                    interaction: true,
                    ..base.clone()
                },
            ),
        ];
        for (label, opts) in cases {
            let r = crate::fit(&model, &pop, &p, &opts).unwrap_or_else(|e| panic!("{label}: {e}"));
            let fp = r_params(&model, &r);
            let frozen_expected = !opts.interaction;
            let (mut worst_own, mut worst_other) = (0.0_f64, f64::INFINITY);
            for (s, sr) in pop.subjects.iter().zip(&r.subjects) {
                let eta = sr.eta.as_slice();
                let f0 = population_variance_preds(&model, s, &fp.theta, 3).unwrap();
                let gmax = |frozen: Option<&[f64]>| {
                    fd(|e| nll(&model, s, &fp, e, frozen), eta)
                        .iter()
                        .fold(0.0_f64, |m, v| {
                            assert!(v.is_finite(), "{label}: non-finite gradient");
                            m.max(v.abs())
                        })
                };
                let (gf, gc) = (gmax(Some(&f0)), gmax(None));
                let (own, other) = if frozen_expected { (gf, gc) } else { (gc, gf) };
                worst_own = worst_own.max(own);
                worst_other = worst_other.min(other);
            }
            eprintln!(
                "MEASURE {label}: own-objective |g| {worst_own:.2e}, other {worst_other:.2e}"
            );
            assert!(
                worst_own < 1e-3,
                "{label}: reported EBEs are not modes of the {} objective (|g| {worst_own:e})",
                if frozen_expected {
                    "frozen"
                } else {
                    "conditional"
                }
            );
            assert!(
                worst_other > 1e-2,
                "{label}: the two objectives must separate here (|g| {worst_other:e})"
            );
        }
    }

    /// `find_ebe_for_stage` — the covariance stencil's, the per-subject FD salvage's and the
    /// mixture class solve's entry point — solves under its stage's convention: bit-identical
    /// to the explicit `Population` solve for a FOCE stage and to the `Conditional` one for a
    /// FOCEI stage, which differ on this fixture.
    #[test]
    fn find_ebe_for_stage_reads_the_stage() {
        let (model, pop, p) = noiov_fixture(None);
        let s = &pop.subjects[9];
        let stage = |interaction: bool| crate::types::FitOptions {
            method: if interaction {
                EstimationMethod::FoceI
            } else {
                EstimationMethod::Foce
            },
            interaction,
            ..Default::default()
        };
        let solve = |v| find_ebe_with_variance(&model, s, &p, 200, 1e-10, None, None, 0, v);
        let (pop_e, cond_e) = (
            solve(EbeVariance::Population),
            solve(EbeVariance::Conditional),
        );
        assert!(
            (&pop_e.eta - &cond_e.eta).amax() > 1e-2,
            "fixture must separate the modes"
        );
        for (interaction, want) in [(false, &pop_e), (true, &cond_e)] {
            let got = find_ebe_for_stage(
                &model,
                s,
                &p,
                200,
                1e-10,
                None,
                None,
                0,
                &stage(interaction),
            );
            for (a, b) in got.eta.iter().zip(want.eta.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "interaction = {interaction}");
            }
        }
    }

    /// T7, covariance FD stencil (#1722 review finding 1): the EBEs `reconverge_population`
    /// re-solves at every stencil point are the stage's convention — bit-identical to the
    /// frozen solve under a FOCE stage and to the conditional one under FOCEI, which differ
    /// here. Every FOCE + proportional/combined covariance step now runs this stencil.
    #[test]
    fn covariance_stencil_reconverges_the_stage_convention() {
        let (model, pop, p) = noiov_fixture(None);
        let x = crate::estimation::parameterization::pack_params(&p);
        // The stencil solves at the unpacked point, which round-trips θ/Ω by ~1 ULP.
        let p = crate::estimation::parameterization::unpack_params(&x, &p);
        let warm: Vec<DVector<f64>> = pop.subjects.iter().map(|_| DVector::zeros(3)).collect();
        let mut sep = 0.0_f64;
        for interaction in [false, true] {
            let stage = crate::types::FitOptions {
                method: if interaction {
                    EstimationMethod::FoceI
                } else {
                    EstimationMethod::Foce
                },
                interaction,
                ..Default::default()
            };
            let (_, ehs, _, _) = crate::estimation::covariance::reconverge_population(
                &x, &model, &pop, &p, &warm, &stage, 1e-8, false,
            );
            let want = if interaction {
                EbeVariance::Conditional
            } else {
                EbeVariance::Population
            };
            let other = if interaction {
                EbeVariance::Population
            } else {
                EbeVariance::Conditional
            };
            let mu = crate::estimation::parameterization::compute_mu_k(
                &model,
                &p.theta,
                stage.mu_referencing,
            );
            for (i, s) in pop.subjects.iter().enumerate() {
                let solve = |v| {
                    find_ebe_with_variance(
                        &model,
                        s,
                        &p,
                        stage.inner_maxiter,
                        1e-8,
                        Some(warm[i].as_slice()),
                        Some(&mu),
                        0,
                        v,
                    )
                    .eta
                };
                let (w, o) = (solve(want), solve(other));
                for (a, b) in ehs[i].iter().zip(w.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "interaction {interaction}, subject {}",
                        s.id
                    );
                }
                sep = sep.max((&w - &o).amax());
            }
        }
        assert!(
            sep > 1e-2,
            "the conventions must separate on this fixture ({sep:e})"
        );
    }

    /// The fitted parameters of `r` on `model`'s template.
    fn r_params(model: &CompiledModel, r: &crate::types::FitResult) -> ModelParameters {
        let mut p = model.default_params.clone();
        p.theta = r.theta.clone();
        p.omega =
            OmegaMatrix::from_matrix(r.omega.clone(), p.omega.eta_names.clone(), p.omega.diagonal);
        p.sigma.values = r.sigma.clone();
        p
    }

    /// T6: the one decision. `Population` exactly for a Sheiner–Beal stage (FOCE family,
    /// interaction off, no quadrature) with a prediction-dependent error model.
    #[test]
    fn ebe_variance_truth_table() {
        use EstimationMethod::*;
        let (prop, _, _) = noiov_fixture(None);
        let (comb, _, _) = noiov_fixture(Some("DV ~ combined(PROP_ERR, ADD_ERR)"));
        let (add, _, _) = noiov_fixture(Some("DV ~ additive(ADD_ERR)"));
        assert!(
            !add.error_spec.has_f_dependent_variance(),
            "fixture: additive"
        );
        // Keyed on the marginal, not the estimator: every method whose EBEs feed the
        // Sheiner–Beal marginal (interaction off, no quadrature) — including a
        // non-interaction SAEM/IMP/Bayes/VI fit's OFV, covariance and SIR evaluations —
        // takes the frozen variance; interaction on, or a quadrature anchor, never does.
        let all = [
            Foce,
            FoceI,
            FoceGn,
            FoceGnHybrid,
            Saem,
            Imp,
            Impmap,
            Bayes,
            Vi,
        ];
        for m in [&prop, &comb] {
            for method in all {
                assert_eq!(
                    EbeVariance::for_stage(method, false, 1, m),
                    EbeVariance::Population,
                    "{method:?}"
                );
                assert_eq!(
                    EbeVariance::for_stage(method, true, 1, m),
                    EbeVariance::Conditional,
                    "{method:?} + interaction"
                );
            }
            // Quadrature (Laplace at any node count, FOCEI with n_agq > 1) scores its own
            // marginal at the conditional mode, interaction flag or not.
            for (method, n_agq) in [(Laplace, 1), (Laplace, 3), (FoceI, 3)] {
                for interaction in [false, true] {
                    assert_eq!(
                        EbeVariance::for_stage(method, interaction, n_agq, m),
                        EbeVariance::Conditional,
                        "{method:?} n_agq {n_agq} interaction {interaction}"
                    );
                }
            }
        }
        for method in all {
            assert_eq!(
                EbeVariance::for_stage(method, false, 1, &add),
                EbeVariance::Conditional,
                "additive {method:?}"
            );
        }
        let opts = crate::types::FitOptions {
            method: Foce,
            interaction: false,
            ..Default::default()
        };
        assert_eq!(
            EbeVariance::for_options(&opts, &prop),
            EbeVariance::Population
        );
        assert_eq!(
            EbeVariance::for_options(
                &crate::types::FitOptions {
                    interaction: true,
                    ..opts
                },
                &prop
            ),
            EbeVariance::Conditional
        );
    }

    /// T6: additive error has no frozen point (`population_variance_preds` is `None`), so a
    /// `Population` solve is the conditional solve bit for bit.
    #[test]
    fn additive_population_solve_is_bit_identical() {
        let (model, pop, mut p) = noiov_fixture(Some("DV ~ additive(ADD_ERR)"));
        p.sigma.values = vec![0.8];
        for s in &pop.subjects {
            assert!(population_variance_preds(&model, s, &p.theta, 3).is_none());
            let a = find_ebe_with_variance(
                &model,
                s,
                &p,
                200,
                1e-8,
                None,
                None,
                0,
                EbeVariance::Population,
            );
            let b = find_ebe_with_variance(
                &model,
                s,
                &p,
                200,
                1e-8,
                None,
                None,
                0,
                EbeVariance::Conditional,
            );
            assert_eq!(a.nll.to_bits(), b.nll.to_bits(), "{}", s.id);
            for (x, y) in a.eta.iter().zip(b.eta.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "{}", s.id);
            }
        }
    }
}

/// #1575: a subject carrying a TTE record on a drug-driven (ODE-accumulated) hazard resolves
/// its inner η-gradient to **FD** — `subject_has_survival_records` declines it, since the
/// hazard term has no analytic η-channel. So `ode_cumhaz_hazard`'s per-dose reads have no
/// `Dual2` twin to keep in parity, which is what `tests/one_snapshot_dose_reads_survival.rs`
/// states. Pinned here at the per-subject route; the model-level `gradient_method_inner`
/// label reads FD on a TTE model since #1861
/// (`fd_fallback_warning_names_the_survival_cause_and_the_label_reports_fd`).
///
/// The control is the same subject without its event record, which the analytic recorder
/// does serve — so the FD half is the survival record's doing, not the model's.
#[cfg(feature = "survival")]
#[test]
fn an_ode_hazard_subject_resolves_its_inner_gradient_to_fd() {
    use crate::types::{DoseEvent, EventType, ObsRecord};
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKA(1.0, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  obs_scale = V
[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("parse the drug-driven joint PK-TTE model");
    let pk_only = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![8.0],
        observations: vec![5.0],
        obs_cmts: vec![2],
        cens: vec![0],
        ..Default::default()
    };
    let mut with_event = pk_only.clone();
    with_event.obs_records = vec![ObsRecord::Event {
        time: 20.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    assert_eq!(
        resolve_gradient_method(&model, &pk_only),
        InnerGradientMethod::Analytic,
        "control: the PK-only subject is in the analytic provider's scope"
    );
    assert_eq!(
        resolve_gradient_method(&model, &with_event),
        InnerGradientMethod::Fd,
        "a TTE record on a drug-driven hazard has no analytic η-channel"
    );
}

/// #1861: the FD-fallback warning names each cause with its own count, and the model-level
/// `gradient_method_inner` label stops claiming analytic on a TTE model.
///
/// One model, three subject shapes: `a` (PK only, bolus) runs analytic; `b` (`a` plus an
/// event record) declines on its survival record; `c` (`a` with its bolus replaced by a
/// steady-state infusion into the lagged depot, which the event walk does not serve) declines
/// on its shape. Each population straddles one sentence of the message, so deleting either
/// sentence — or crediting a subject to the wrong one — reddens an arm here; `[a, a]` is the
/// silent control. `[b, b]` must warn although the label already reads FD (the survival
/// cause lifts the all-FD guard), while `[c, c]` under the same FD label stays silent. The label pair is the same model with and without `[event_model]`.
#[cfg(feature = "survival")]
#[test]
fn fd_fallback_warning_names_the_survival_cause_and_the_label_reports_fd() {
    use crate::types::{DoseEvent, EventType, ObsRecord, RateMode};
    const PK: &str = "[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  theta TVLAMBDA(0.01, 0.0001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = TVF
  ALAG = 0.5
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F, lagtime=ALAG)
[error_model]
  DV ~ proportional(PROP_ERR)
";
    let tte_src = format!(
        "{PK}[event_model]\n  cmt    = 3\n  family = exponential\n  scale  = TVLAMBDA * exp(ETA_CL)\n"
    );
    let parse = |src: &str| crate::parser::model_parser::parse_model_string(src).expect("parses");
    let (pk_model, tte_model) = (parse(PK), parse(&tte_src));

    // Label: the TTE endpoint alone flips it.
    assert!(
        inner_reports_analytic_model(&pk_model),
        "control: PK-only is analytic"
    );
    assert!(
        !inner_reports_analytic_model(&tte_model),
        "a TTE model's subjects with event records all run FD"
    );
    assert_eq!(
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, &tte_model),
        crate::build_info::GradientMethodKind::FiniteDifferences
    );

    let a = Subject {
        id: "a".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![2.0, 8.0],
        observations: vec![5.0, 3.0],
        obs_cmts: vec![2, 2],
        cens: vec![0, 0],
        ..Default::default()
    };
    let mut b = a.clone();
    b.id = "b".into();
    b.obs_records = vec![ObsRecord::Event {
        time: 20.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    let mut c = a.clone();
    c.id = "c".into();
    let mut ss = DoseEvent::new(0.0, 100.0, 1, 50.0, true, 24.0);
    ss.rate_mode = RateMode::Fixed;
    c.doses = vec![ss];
    let theta = &tte_model.default_params.theta;
    let zeros = [0.0];
    assert_eq!(subject_inner_fd_reason(&tte_model, &a, theta, &zeros), None);
    assert_eq!(
        subject_inner_fd_reason(&tte_model, &b, theta, &zeros),
        Some(InnerFdReason::Survival)
    );
    assert_eq!(
        subject_inner_fd_reason(&tte_model, &c, theta, &zeros),
        Some(InnerFdReason::Shape),
        "fixture precondition: the SS infusion under a lag is a shape decline"
    );

    let warn = |subjects: Vec<Subject>| {
        fd_fallback_warning(
            &tte_model,
            &Population {
                subjects,
                covariate_names: Vec::new(),
                dv_column: "DV".into(),
                input_columns: vec![],
                exclusions: None,
                warnings: vec![],
            },
            theta,
        )
    };
    const TTE: &str = "carry a time-to-event (or other non-Gaussian) record whose likelihood \
                       term has no analytic η-gradient";
    const SHAPE: &str = "have a shape no analytic recorder serves";

    assert_eq!(
        warn(vec![a.clone(), a.clone()]),
        None,
        "all-analytic is silent"
    );

    let w = warn(vec![a.clone(), b.clone()]).expect("a mixed TTE population warns");
    assert!(
        w.starts_with("1 of 2 subjects use finite-difference inner gradients (1 "),
        "{w}"
    );
    assert!(w.contains(TTE) && !w.contains(SHAPE), "{w}");

    // All-TTE: the label already reads FD, but the warning still fires to name the cause.
    let w = warn(vec![b.clone(), b.clone()]).expect("an all-TTE population warns");
    assert!(
        w.starts_with(&format!(
            "2 of 2 subjects use finite-difference inner gradients (2 {TTE})"
        )),
        "{w}"
    );
    // All-shape under the FD label stays silent: only the survival cause lifts that guard.
    assert_eq!(
        warn(vec![c.clone(), c.clone()]),
        None,
        "all-shape under an FD label"
    );

    let w = warn(vec![a.clone(), c.clone()]).expect("a mixed shape population warns");
    assert!(
        w.contains(&format!("(1 {SHAPE}")) && !w.contains(TTE),
        "{w}"
    );

    let w = warn(vec![a, b, c]).expect("both causes warn");
    assert!(
        w.contains(&format!(
            "2 of 3 subjects use finite-difference inner gradients (1 {TTE}; 1 {SHAPE}"
        )),
        "{w}"
    );
}

/// Flip-flop one-compartment oral subject (#2013, #2014). The data are noiseless from
/// CL = 1, V = 10, ka = 0.02 — absorption slower than elimination (k = 0.1) — and swapping
/// ka and k with V' = V·k/ka reproduces the curve exactly, so the individual objective has
/// two modes that differ only in the Ω penalty: the deeper one at η ≈ (0, 0, −3.90) and a
/// twin at η ≈ (0, +1.58, −2.34) (ka = 0.1, V = 50). BFGS from η = 0 lands in the twin. The
/// twin is well curved, so `weakly_identified_coords` does not flag it (#2013); `reset`
/// adds an inert reset before the dose so the subject takes the scan-every-coordinate arm.
/// NONMEM 7.5.1 at the same estimates agrees on both modes (#2013's comparison).
fn flipflop_fixture(reset: bool) -> (CompiledModel, Subject) {
    use crate::types::{DoseEvent, Subject};
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(1.0,0.01,100)\n  theta TVV(10.0,0.1,1000)\n  theta TVKA(1.0,0.001,100)\n  omega ETA_CL ~ 0.04\n  omega ETA_V ~ 0.1\n  omega ETA_KA ~ 1.0\n  sigma PROP_ERR ~ 0.05 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n",
    )
    .expect("parse the flip-flop model");
    let obs_times = vec![
        0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 48.0, 72.0, 96.0,
    ];
    let (cl, v, ka) = (1.0_f64, 10.0_f64, 0.02_f64);
    let k = cl / v;
    let observations: Vec<f64> = obs_times
        .iter()
        .map(|&t| 100.0 * ka / (v * (ka - k)) * ((-k * t).exp() - (-ka * t).exp()))
        .collect();
    let n = obs_times.len();
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: if reset { vec![0.0] } else { Vec::new() },
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };
    assert_eq!(subject.has_resets(), reset, "test premise: reset arm");
    assert!(!subject.has_tv_covariates());
    (model, subject)
}

/// #2014: every estimating fit hands the inner loop `Some(zeros)` on its first evaluation
/// (`new_nlopt_state`'s cache, the pre-flight solve), and the guarded multi-start tested
/// `eta_init.is_none()`, so it never ran before the final cold re-solve. An all-zero start
/// must multi-start exactly as `None` does, and a genuine warm start must not.
#[test]
fn an_all_zero_start_runs_the_guarded_multistart_like_none() {
    assert!(!is_warm_start(None, false));
    assert!(!is_warm_start(Some(&[0.0, 0.0, 0.0]), false));
    assert!(is_warm_start(Some(&[0.0, 0.5, 0.0]), false));
    // #2076: a certified mode is warm even when it is exactly zero, and `None` stays cold.
    assert!(is_warm_start(Some(&[0.0, 0.0, 0.0]), true));
    assert!(!is_warm_start(None, true));

    let (model, subject) = flipflop_fixture(true);
    let params = &model.default_params;
    let zeros = vec![0.0; model.n_eta];
    let trapped = find_ebe(&model, &subject, params, 200, 1e-8, None, None, 0);
    let from_none = find_ebe(&model, &subject, params, 200, 1e-8, None, None, 1);
    let from_zeros = find_ebe(&model, &subject, params, 200, 1e-8, Some(&zeros), None, 1);

    // The straddle: on this fixture the multi-start moves the mode, so the equality below
    // cannot hold merely because the multi-start is a no-op.
    assert!(
        trapped.eta[2] > -3.0 && from_none.eta[2] < -3.5,
        "premise: restarts=0 must sit in the twin and the cold multi-start in the deeper \
         mode; got η_KA {} and {}",
        trapped.eta[2],
        from_none.eta[2]
    );
    assert!(
        from_none.nll + 5.0 < trapped.nll,
        "premise: the deeper mode is lower ({} vs {})",
        from_none.nll,
        trapped.nll
    );

    assert_eq!(
        from_zeros.eta.as_slice(),
        from_none.eta.as_slice(),
        "an all-zero start skipped the guarded multi-start (#2014)"
    );
    assert_eq!(from_zeros.nll.to_bits(), from_none.nll.to_bits());

    // A genuine warm start at the twin is the previous evaluation's mode carried forward,
    // and is not re-scanned: the multi-start runs once per optimizer leg, not per evaluation.
    let warm = find_ebe(
        &model,
        &subject,
        params,
        200,
        1e-8,
        Some(trapped.eta.as_slice()),
        None,
        1,
    );
    assert!(
        warm.eta[2] > -3.0,
        "a genuine warm start re-ran the multi-start: η_KA {}",
        warm.eta[2]
    );
}

/// #2026: `ebe_h_matrices_at` rebuilds the `H` the inner loop reports from the EBEs alone,
/// bit for bit, on every route `H` takes — the closed-form and ODE analytic Jacobians, the
/// FD fallback, and the IOV solve's BSV block (analytic and FD) — and to one ULP the
/// Jacobian an `analytic_terminal_work` capture supplies instead. `run_covariance` hands it to the score cross-product for a
/// fit whose reported EBEs are not a cold solve, so any drift from the solve's own `H` is a
/// wrong `s` / `rsr` covariance. The init parameters put every fixture's EBEs off zero
/// (asserted), and a second call at moved EBEs must change `H`, so a rebuild that ignored
/// its `etas` (or `kappas`) could not pass. Each row also asserts the route it names:
/// mutation — drop `GradientMethod::forced_fd` from `analytic_inner_common_bail` → the
/// `warfarin fd` row dies on its premise instead of passing as a second analytic row.
#[test]
fn ebe_h_matrices_at_rebuilds_the_inner_loops_h() {
    let cases = [
        (
            "warfarin closed form",
            "examples/warfarin.ferx",
            "data/warfarin.csv",
            false,
            false,
        ),
        (
            "warfarin captured",
            "examples/warfarin.ferx",
            "data/warfarin.csv",
            false,
            true,
        ),
        (
            "warfarin fd",
            "examples/warfarin.ferx",
            "data/warfarin.csv",
            true,
            false,
        ),
        (
            "warfarin_ode",
            "examples/warfarin_ode.ferx",
            "data/warfarin.csv",
            false,
            false,
        ),
        (
            "warfarin_iov",
            "examples/warfarin_iov.ferx",
            "data/warfarin_iov.csv",
            false,
            false,
        ),
        (
            "warfarin_iov fd",
            "examples/warfarin_iov.ferx",
            "data/warfarin_iov.csv",
            true,
            false,
        ),
    ];
    for (what, model_path, data, fd, capture) in cases {
        let prep = crate::prepare_run(model_path, Some(data)).expect("prepare");
        let (mut model, population, params) =
            (prep.parsed.model, prep.population, prep.init_params);
        if fd {
            model.gradient_method = crate::types::GradientMethod::Fd;
        }
        let policy = InnerSolvePolicy {
            capture_terminal_hessian: capture,
            ..InnerSolvePolicy::default()
        };
        let (etas, hs, _, kappas, captured) = run_inner_loop_warm_map(
            &model,
            &population,
            &params,
            200,
            1e-8,
            None,
            None,
            0,
            0,
            policy,
            None,
            |_, r| r.terminal_hessian.is_some(),
        );
        assert!(
            etas.iter().any(|e| e.iter().any(|v| *v != 0.0)),
            "{what}: premise — the EBEs must be off zero"
        );
        assert_eq!(
            captured.iter().all(|c| *c),
            capture,
            "{what}: premise — the terminal capture (and its Jacobian) ran exactly when asked"
        );
        let is_iov = model.n_kappa > 0;
        assert_eq!(
            is_iov,
            kappas.iter().any(|k| !k.is_empty()),
            "{what}: premise — κ̂ present exactly on the IOV fixtures"
        );
        // The predicates `ebe_h_matrix` / `iov_ebe_h_matrix` branch on, so an `fd` row
        // that stopped reaching the FD Jacobian would fail here, not pass as a second
        // analytic row.
        for subject in &population.subjects {
            let analytic = if is_iov {
                analytic_iov_inner_supported(&model, subject, &params)
            } else {
                analytic_inner_grad_supported(&model, subject)
            };
            assert_eq!(
                analytic,
                !fd,
                "{what}: premise — subject {} takes the {} H route",
                subject.id,
                if fd { "FD" } else { "analytic" }
            );
        }
        let rebuilt = ebe_h_matrices_at(&model, &population, &params, &etas, &kappas);
        if capture {
            // The captured Jacobian comes out of the terminal `exact_eta_jacobian` recording,
            // the rebuild out of `exact_obs_grads`: the same `∂f/∂η`, not the
            // same rounding. Measured on macOS arm64: 4.8e-16 relative (5.3e-15 absolute),
            // one ULP; the bound leaves 20× headroom.
            let mut worst = 0.0f64;
            for (a, b) in rebuilt.iter().zip(&hs) {
                assert!(
                    a.iter().chain(b.iter()).all(|v| v.is_finite()),
                    "{what}: non-finite H"
                );
                // An all-zero `b` makes the ratio `0/0`, which `f64::max` would drop.
                let rel = (a - b).abs().max() / b.abs().max();
                assert!(rel.is_finite(), "{what}: relative H error {rel}");
                worst = worst.max(rel);
            }
            assert!(
                worst <= 1e-14,
                "{what}: rebuilt H off by {worst:e} relative"
            );
        } else {
            assert!(rebuilt == hs, "{what}: rebuilt H differs from the solve's");
        }

        let moved: Vec<DVector<f64>> = etas.iter().map(|e| e.add_scalar(0.1)).collect();
        assert!(
            ebe_h_matrices_at(&model, &population, &params, &moved, &kappas) != hs,
            "{what}: H must depend on the EBEs it is rebuilt at"
        );
        if is_iov {
            let moved_k: Vec<Vec<DVector<f64>>> = kappas
                .iter()
                .map(|ks| ks.iter().map(|k| k.add_scalar(0.1)).collect())
                .collect();
            assert!(
                ebe_h_matrices_at(&model, &population, &params, &etas, &moved_k) != hs,
                "{what}: the IOV H must depend on the κ̂ it is rebuilt at"
            );
        }
    }
}

/// #2093 review (round 2): `gradient = fd` keeps the inner search and the `H` matrix off
/// the adjoint kernel. `find_ebe`'s non-analytic arm returned `exact_inner_gradient`
/// whenever it served the subject, and `ebe_h_matrix` fell through to the exact
/// η-Jacobian, so a forced-FD fit ran analytic derivatives under an FD label. The
/// mutation this catches is dropping `inner_analytic_opted_out` from either route: the
/// `fd` arm's exact count goes non-zero, or its `H` equals the exact Jacobian's bits.
#[test]
fn gradient_fd_keeps_the_inner_search_and_h_off_the_exact_routes() {
    use std::path::Path;
    let mut model =
        crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
            .expect("warfarin parses");
    let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let subj = &pop.subjects[0];
    let params = model.default_params.clone();
    model.gradient_method = GradientMethod::Fd;
    INNER_AGRAD_ROUTE.with(|c| c.set((0, 0, 0)));
    let ebe = find_ebe(&model, subj, &params, 200, 1e-8, None, None, 0);
    let (analytic, exact, fd) = INNER_AGRAD_ROUTE.with(|c| c.get());
    assert!(ebe.nll.is_finite(), "{}", ebe.nll);
    assert!(
        analytic == 0 && exact == 0 && fd > 0,
        "gradient = fd: {analytic} analytic / {exact} exact / {fd} FD steps"
    );
    // H at the same EBE: the FD Jacobian, not the exact one bit for bit.
    let exact_j = crate::sens::ode_provider::adjoint::exact_eta_jacobian(
        &model,
        subj,
        &params.theta,
        ebe.eta.as_slice(),
    )
    .expect("the adjoint serves warfarin");
    let h = &ebe.h_matrix;
    assert_eq!((h.nrows(), h.ncols()), (exact_j.nrows(), exact_j.ncols()));
    assert!(
        h.iter()
            .zip(exact_j.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "gradient = fd must build H by finite differences"
    );
    let gap = h
        .iter()
        .zip(exact_j.iter())
        .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
        .fold(0.0, f64::max);
    assert!(
        gap < 1e-4,
        "the FD H must still be the same matrix: gap {gap:.2e}"
    );
}

/// A cold start on a steep objective takes a bounded first step (`cap_curvature_free_step`).
///
/// `examples/warfarin_ode_lagtime.ferx`, subject 3: the 0.5 h observation coincides with the
/// lagged arrival at the default `TVLAG = 0.5`, so at η = 0 its prediction is 0, its
/// proportional variance sits on the floor, the objective is 1.2e12 and ‖g‖ is 9.5e12.
/// Unbounded, the Ω-preconditioned first step `−Ω·g` put η_LAG at −800; the exact gradient
/// then followed a genuine descent to `KA = e^42`, where every ODE solve is stiff, and the
/// runaway guard's Nelder–Mead found the mode at the end anyway (0.96 s for this subject).
/// Measured in ODE stepper attempts across the whole solve: 311 bounded, with no fallback;
/// 2,038,257 with the cap removed (run), through the Nelder–Mead fallback. The mode is the
/// same either way, so this pins the cost: the bound of 20,000 is ~64× over the fix and
/// ~100× under the mutant.
#[test]
fn a_cold_start_on_a_variance_floor_takes_a_bounded_first_step() {
    use std::path::Path;
    let parsed = crate::parse_full_model_file(Path::new("examples/warfarin_ode_lagtime.ferx"))
        .expect("warfarin_ode_lagtime parses");
    let mut model = parsed.model;
    model.sync_ode_solver_opts(&parsed.fit_options);
    let pop = crate::io::datareader::read_nonmem_csv(
        Path::new("data/warfarin_ode_lagtime.csv"),
        None,
        None,
    )
    .expect("warfarin_ode_lagtime data loads");
    let subject = pop
        .subjects
        .iter()
        .find(|s| s.id == "3")
        .expect("subject 3");
    let params = model.default_params.clone();

    // Precondition: at η = 0 the first observation is on the variance floor.
    let preds = crate::pk::compute_predictions_with_tv(&model, subject, &params.theta, &[0.0; 4]);
    assert_eq!(
        preds[0], 0.0,
        "the 0.5 h observation must sit at the lagged arrival"
    );

    let scope = crate::ode::solver::SolverStatsScope::enter();
    let r = find_ebe(
        &model,
        subject,
        &params,
        parsed.fit_options.inner_maxiter,
        parsed.fit_options.inner_tol,
        None,
        None,
        0,
    );
    let attempts = scope.collected().attempted_steps;
    drop(scope);
    assert!(
        (r.nll - (-16.751285)).abs() < 1e-5,
        "EBE objective {} is not the mode",
        r.nll
    );
    assert!(
        ebe_prior_maha(r.eta.as_slice(), &params.omega) < RUNAWAY_EBE_MAHA_PER_ETA * 4.0,
        "EBE {:?} is a runaway",
        r.eta.as_slice()
    );
    assert!(
        !r.used_fallback,
        "the bounded first step should reach the mode without the Nelder–Mead fallback"
    );
    assert!(
        attempts < 20_000,
        "{attempts} ODE step attempts: the first step left the plausible region and the \
         search marched into a stiff one"
    );
}

/// `cap_curvature_free_step` leaves a step inside the runaway radius bit-identical and
/// shortens one outside it to exactly that radius, along the same direction. Radius for
/// `n = 2`: `√(RUNAWAY_EBE_MAHA_PER_ETA·2)` in the `diag(p)` metric.
#[test]
fn the_curvature_free_step_cap_keeps_shape_and_bounds_length() {
    let p = [0.09, 1e-4];
    let r2 = RUNAWAY_EBE_MAHA_PER_ETA * 2.0;
    let metric = |d: &[f64]| d.iter().zip(&p).map(|(di, pi)| di * di / pi).sum::<f64>();

    let inside = vec![0.5, 0.01];
    assert!(metric(&inside) < r2);
    let mut d = inside.clone();
    cap_curvature_free_step(&mut d, Some(&p));
    assert_eq!(d, inside, "a step inside the radius is untouched");

    let outside = vec![-0.09 * 9.5e12, 1e-4 * 2.7];
    let mut d = outside.clone();
    cap_curvature_free_step(&mut d, Some(&p));
    assert!(
        (metric(&d) - r2).abs() < 1e-9 * r2,
        "capped to the radius: {}",
        metric(&d)
    );
    assert!(
        ((d[0] / d[1]) - (outside[0] / outside[1])).abs() < 1e-9 * (outside[0] / outside[1]).abs(),
        "the direction is kept"
    );

    let mut d = outside.clone();
    cap_curvature_free_step(&mut d, None);
    assert_eq!(
        d, outside,
        "the identity path is bounded by `h0_step_scale`, not here"
    );
}

/// Both drivers' first trial point from a steep, Ω-preconditioned start lies within the
/// runaway radius (#2080, #2099). Fixture: `f = ½·1e12·(x₀ − 1)² + ½x₁²` from 0, so
/// ‖g‖ ≈ 1e12 and the unbounded first step `−diag(p)·g` is ~3e12 prior SDs long.
/// Mutation: removing the cap from `dense_bfgs_core`'s first iterate reddens the dense half;
/// removing it from `lbfgs_core`'s empty-history direction reddens the L-BFGS half.
#[test]
fn a_steep_preconditioned_start_takes_a_bounded_first_trial() {
    use std::cell::RefCell;
    let p = [0.09, 0.04];
    let r2 = RUNAWAY_EBE_MAHA_PER_ETA * 2.0;
    let never = |_: &[f64]| false;
    for dense in [true, false] {
        let first: RefCell<Option<Vec<f64>>> = RefCell::new(None);
        let calls = RefCell::new(0usize);
        let obj = |x: &[f64]| {
            let mut c = calls.borrow_mut();
            *c += 1;
            if *c == 2 {
                *first.borrow_mut() = Some(x.to_vec());
            }
            0.5e12 * (x[0] - 1.0).powi(2) + 0.5 * x[1] * x[1]
        };
        let grad = |x: &[f64]| vec![1e12 * (x[0] - 1.0), x[1]];
        let mut x = vec![0.0, 0.0];
        if dense {
            dense_bfgs_core(
                &obj,
                &grad,
                &mut x,
                2,
                3,
                1e-8,
                Some(&p),
                None,
                None,
                &never,
                false,
            );
        } else {
            lbfgs_core(
                &obj,
                &grad,
                &mut x,
                2,
                3,
                1e-8,
                Some(&p),
                None,
                &never,
                false,
            );
        }
        let t = first.borrow().clone().expect("a first trial was evaluated");
        let m: f64 = t.iter().zip(&p).map(|(ti, pi)| ti * ti / pi).sum();
        assert!(
            m <= r2 * (1.0 + 1e-9),
            "{}: first trial {t:?} is {m:.3e} in the prior metric, radius² {r2}",
            if dense { "dense BFGS" } else { "L-BFGS" }
        );
    }
}

// ---- #2080 / #2091 outcome fixtures, ported from @carldelfin's #2091 ---------------------
//
// #2091 fixes the far-seed case with a rescaled retry on a failed first line search; #2093
// bounds every curvature-free first step instead (`cap_curvature_free_step`). These are
// #2091's outcome assertions — the deeper mode is reached from the seed, by BFGS, and the
// cold multi-start keeps it — without its retry counter, which pins #2091's mechanism.

fn flip_flop_oral_fixture(v: f64, omega: [f64; 3]) -> (CompiledModel, Subject) {
    use crate::types::{DoseEvent, Subject};
    let times = [
        0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 48.0, 72.0, 96.0,
    ];
    let (cl, ka) = (1.0f64, 0.02f64);
    let k = cl / v;
    let observations: Vec<f64> = times
        .iter()
        .map(|&t| {
            let c = 100.0 * ka / (v * (ka - k)) * ((-k * t).exp() - (-ka * t).exp());
            format!("{c:.10e}").parse().unwrap()
        })
        .collect();
    let model = crate::parser::model_parser::parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 100)\n  theta TVV(10.0, 0.1, 1000)\n  theta TVKA(1.0, 0.001, 100)\n  omega ETA_CL ~ {}\n  omega ETA_V ~ {}\n  omega ETA_KA ~ {}\n  sigma PROP_ERR ~ 0.05 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n",
        omega[0], omega[1], omega[2]
    ))
    .expect("parse the flip-flop model");
    let n = times.len();
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    };
    (model, subject)
}

/// Solve from η = 0, then again from that η̂ with η_V moved to −2 sd (the seed
/// `inner_restarts = 1` tries). Returns `(from η = 0, from the seed)`.
fn flip_flop_eta_v_minus_2sd(v: f64, omega: [f64; 3]) -> (EbeResult, EbeResult) {
    let (model, subject) = flip_flop_oral_fixture(v, omega);
    let params = &model.default_params;
    let trapped = find_ebe(
        &model,
        &subject,
        params,
        200,
        1e-5,
        Some(&[0.0; 3]),
        None,
        0,
    );
    let mut seed = trapped.eta.as_slice().to_vec();
    seed[1] = -2.0 * omega[1].sqrt();
    let from_seed = find_ebe(&model, &subject, params, 200, 1e-5, Some(&seed), None, 0);
    (trapped, from_seed)
}

/// #2080. On config 2 (V = 5, Ω = diag(0.04, 0.04, 2.0)) the η_V −2 sd seed starts at an
/// objective of ~1.3e10 with ‖g‖ ~ 3.8e11. The Ω-preconditioned path skips the first-step
/// scaling, so the first step is ~1.6e11 long and 30 halvings never bring a trial back
/// below the start: BFGS failed, the fallback restarted from η = 0, and the solve returned
/// the trapped mode. With the first step bounded, BFGS reaches the deeper mode from the seed.
#[test]
fn far_seed_reaches_the_deeper_mode_from_a_bounded_first_step() {
    let (trapped, r) = flip_flop_eta_v_minus_2sd(5.0, [0.04, 0.04, 2.0]);
    eprintln!(
        "config 2: trapped {:.6}, from seed {:.6}, fallback {}",
        trapped.nll, r.nll, r.used_fallback
    );
    assert!(trapped.nll.is_finite() && r.nll.is_finite());
    assert!(
        !r.used_fallback,
        "BFGS must converge from the seed itself, not through the Nelder–Mead fallback"
    );
    assert!(
        r.nll < trapped.nll - 1.0,
        "the seed must reach the deeper mode: {} vs trapped {}",
        r.nll,
        trapped.nll
    );
}

/// The symptom #2080 was filed for: the cold multi-start drops that seed. The multi-start has
/// no Nelder–Mead fallback of its own, so on `main` config 2 at `inner_restarts = 1` kept the
/// twin at −21.073865. A per-row covariate the model never reads puts the subject on the
/// scan-all arm (`has_tv_covariates`), which tries the η_V −2 sd seed; the predictions do
/// not depend on it.
#[test]
fn cold_multi_start_keeps_the_far_seed() {
    let (model, mut subject) = flip_flop_oral_fixture(5.0, [0.04, 0.04, 2.0]);
    subject.obs_covariates = (0..subject.obs_times.len())
        .map(|i| HashMap::from([("WT".to_string(), 70.0 + i as f64)]))
        .collect();
    assert!(
        subject.has_tv_covariates() && !subject.has_resets(),
        "test premise: the subject is on the scan-all arm through its covariates"
    );
    let params = &model.default_params;
    let single = find_ebe(&model, &subject, params, 200, 1e-5, None, None, 0);
    let multi = find_ebe(&model, &subject, params, 200, 1e-5, None, None, 1);
    eprintln!(
        "config 2 multi-start: r=0 {:.6}, r=1 {:.6}",
        single.nll, multi.nll
    );
    assert!(single.nll.is_finite() && multi.nll.is_finite());
    assert!(
        multi.nll < single.nll - 1.0,
        "inner_restarts = 1 must keep the deeper mode: {} vs r = 0 {}",
        multi.nll,
        single.nll
    );
}

/// The control: on config 1 (V = 10, Ω = diag(0.04, 0.1, 1.0)) the same seed's first line
/// search already succeeded before #2080, and the solve still reaches the deeper mode.
#[test]
fn far_seed_on_config_1_still_reaches_the_deeper_mode() {
    let (trapped, r) = flip_flop_eta_v_minus_2sd(10.0, [0.04, 0.1, 1.0]);
    eprintln!(
        "config 1: trapped {:.6}, from seed {:.6}, fallback {}",
        trapped.nll, r.nll, r.used_fallback
    );
    assert!(trapped.nll.is_finite() && r.nll.is_finite());
    assert!(!r.used_fallback);
    assert!(
        r.nll < trapped.nll - 1.0,
        "{} vs trapped {}",
        r.nll,
        trapped.nll
    );
}

/// The identity path's first-step scaling is decided once, at the first iterate (#2093).
/// Fixture: `f = −eˣ` from `x₀ = −0.2`, so ‖g₀‖ = 0.82 (below 1: no scaling) and after the
/// unit first step ‖g₁‖ = 1.86. The function is concave, so `sᵀy < 0` and BFGS skips its
/// update: `h_inv` is still the identity at iteration 2 and its first trial is `x₁ + 1.86`.
/// Before, the flag cleared only when the scaling fired, so iteration 2 rescaled by 1/‖g₁‖
/// and tried `x₁ + 1` (mutation, run: restoring `first_step = false` inside the `if`).
#[test]
fn the_identity_first_step_scaling_is_decided_once() {
    use std::cell::RefCell;
    let trials = RefCell::new(Vec::new());
    let obj = |x: &[f64]| {
        trials.borrow_mut().push(x[0]);
        -x[0].exp()
    };
    let grad = |x: &[f64]| vec![-x[0].exp()];
    let never = |_: &[f64]| false;
    let mut x = vec![-0.2];
    dense_bfgs_core(
        &obj, &grad, &mut x, 1, 2, 1e-12, None, None, None, &never, false,
    );
    let t = trials.borrow();
    // trials[0] is the start's own objective; [1] the first iteration's α = 1 trial.
    let x0 = -0.2f64;
    // ‖g₀‖ < 1, so the identity step is −g₀ = +e^{x₀}, accepted at α = 1 (f is decreasing).
    let x1 = x0 + x0.exp();
    assert!((t[1] - x1).abs() < 1e-12, "first trial {} vs {x1}", t[1]);
    let g1 = x1.exp();
    assert!(g1 > 1.0, "fixture premise: ‖g₁‖ = {g1} must exceed 1");
    let second = t[2];
    assert!(
        (second - (x1 + g1)).abs() < 1e-12,
        "iteration 2's first trial is {second}: expected the unscaled x₁ + ‖g₁‖ = {}, not \
         the rescaled x₁ + 1 = {}",
        x1 + g1,
        x1 + 1.0
    );
}

/// A BFGS started next to the mode of an ill-conditioned objective learns its curvature (#2270).
/// Fixture: `f = ½(1e6·x₀² + x₁²)` from the steepest-descent worst case, `g ∝ (1, 1)` with
/// ‖g‖ = 25·`tol`. Every accepted step is short there, so `sᵀy = sᵀHs` is ~1e-13: under the
/// absolute `sᵀy > 1e-12` gate alone no pair updated the identity metric, and the search
/// zigzagged (each step shrinking ‖g‖ by about `1 − 2/κ`) to `max_iter` — in both drivers.
#[test]
fn bfgs_started_next_to_the_mode_learns_its_curvature() {
    let lambda = [1e6, 1.0];
    let tol = 1e-5;
    let obj = |x: &[f64]| 0.5 * x.iter().zip(&lambda).map(|(v, l)| l * v * v).sum::<f64>();
    let grad = |x: &[f64]| {
        x.iter()
            .zip(&lambda)
            .map(|(v, l)| l * v)
            .collect::<Vec<_>>()
    };
    let never = |_: &[f64]| false;
    let start: Vec<f64> = lambda
        .iter()
        .map(|l| 25.0 * tol / 2f64.sqrt() / l)
        .collect();
    let gn = |x: &[f64]| grad(x).iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((gn(&start) / tol - 25.0).abs() < 1e-9, "fixture premise");
    for (label, lbfgs) in [("dense", false), ("lbfgs", true)] {
        let calls = std::cell::Cell::new(0usize);
        let counted = |x: &[f64]| {
            calls.set(calls.get() + 1);
            obj(x)
        };
        let mut x = start.clone();
        let ok = if lbfgs {
            lbfgs_core(
                &counted, &grad, &mut x, 2, 200, tol, None, None, &never, false,
            )
        } else {
            dense_bfgs_core(
                &counted, &grad, &mut x, 2, 200, tol, None, None, None, &never, false,
            )
        };
        assert!(
            ok && gn(&x) < tol,
            "{label}: not converged from 25·tol (‖g‖ = {:.3e} after {} objective calls)",
            gn(&x),
            calls.get()
        );
    }
}

/// On an exact objective, a step whose gain is below what the objective resolves is taken on
/// the gradient's word (#2270). Fixture: `f = 10 + ½(1e5·x₀² + 1e3·x₁²) + e(x)`, where `e` is a
/// deterministic ±5e-11 evaluation error (a hash of `x`'s bits) and the gradient is exact, as
/// the analytic inner gradient is. From eight starts at ‖g‖ = 25·`tol` the seeded Newton step
/// gains ~1.6e-11, under the error, so the Armijo test reads noise. Without the gradient
/// verification the line search collapsed and the solve ran to `max_iter` from some of them.
#[test]
fn a_step_below_the_objective_resolution_is_verified_by_the_gradient() {
    let lambda = [1e5, 1e3];
    let tol = 1e-5;
    let error = |x: &[f64]| {
        let mut h = x[0].to_bits() ^ x[1].to_bits().rotate_left(32);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        1e-10 * ((h >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
    };
    let obj = |x: &[f64]| {
        10.0 + 0.5 * x.iter().zip(&lambda).map(|(v, l)| l * v * v).sum::<f64>() + error(x)
    };
    let grad = |x: &[f64]| {
        x.iter()
            .zip(&lambda)
            .map(|(v, l)| l * v)
            .collect::<Vec<_>>()
    };
    let gn = |x: &[f64]| grad(x).iter().map(|v| v * v).sum::<f64>().sqrt();
    let never = |_: &[f64]| false;
    let seed = DMatrix::from_diagonal(&DVector::from_column_slice(&lambda));
    let mut stuck = Vec::new();
    for k in 0..8 {
        let angle = std::f64::consts::PI * (k as f64 + 0.5) / 8.0;
        let g0 = [angle.cos(), angle.sin()].map(|c| 25.0 * tol * c);
        let start: Vec<f64> = g0.iter().zip(&lambda).map(|(g, l)| g / l).collect();
        // The premise: the Newton step's gain is below the evaluation error's spread.
        let gain = 0.5 * g0.iter().zip(&lambda).map(|(g, l)| g * g / l).sum::<f64>();
        assert!(
            gain < 1e-10 && gain > 0.0,
            "fixture premise: gain {gain:.3e}"
        );
        for driver in ["dense", "lbfgs"] {
            let mut x = start.clone();
            let points = std::cell::RefCell::new(Vec::new());
            let logged = |x: &[f64]| {
                points.borrow_mut().push([x[0].to_bits(), x[1].to_bits()]);
                grad(x)
            };
            let ok = if driver == "dense" {
                dense_bfgs_core(
                    &obj,
                    &logged,
                    &mut x,
                    2,
                    200,
                    tol,
                    None,
                    Some(&seed),
                    None,
                    &never,
                    false,
                )
            } else {
                lbfgs_core(
                    &obj, &logged, &mut x, 2, 200, tol, None, None, &never, false,
                )
            };
            // Convergence is asserted for the seeded dense driver, whose first step is Newton's.
            // Unseeded L-BFGS starts from the identity metric, whose first steps gain less than
            // this fixture's (deliberately large) evaluation error, and from start 2 it stops
            // unverified at ‖g‖ = 2.3e-4 (measured): a property of the fixture's noise, not of
            // the rule. It is here for the reuse below.
            if driver == "dense" && !(ok && gn(&x) < tol) {
                stuck.push((k, gn(&x)));
            }
            // A gradient the driver already took at a point is reused there, whether a verified
            // step took it or a rejected verification did and the line search then accepted the
            // full step: no point has its gradient evaluated twice. Before the rejected case was
            // reused, L-BFGS evaluated 6 gradients at 5 points from start 0, 9 at 7 from start 1.
            let mut seen = points.borrow().clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                points.borrow().len(),
                "{driver}, start {k}: a gradient was evaluated twice at one point"
            );
        }
    }
    assert!(
        stuck.is_empty(),
        "dense: starts that did not converge (index, final ‖g‖): {stuck:?}"
    );
}

/// A step that lands on an ODE objective (`stall_at` turns true there, a transit/IG subject
/// crossing onto its ODE twin) has its curvature pair judged as an ODE pair: the latch is updated
/// at the accepted point before the pair test (#2280 review, finding 6). Fixture: the
/// `bfgs_started_next_to_the_mode_learns_its_curvature` quadratic, where every pair's `sᵀy` is
/// ~1e-13, so the relative arm alone decides whether it updates the metric. Run A is an ODE
/// objective everywhere; run B everywhere but the start point. From the first accepted point on
/// both are ODE objectives, so their pairs and paths must agree bit for bit. With the latch read
/// only before each step, B's first pair was judged exact, taken by the relative arm, and B's
/// path left A's.
#[test]
fn a_step_onto_an_ode_objective_has_its_pair_judged_as_ode() {
    let lambda = [1e6, 1.0];
    let tol = 1e-5;
    let obj = |x: &[f64]| 0.5 * x.iter().zip(&lambda).map(|(v, l)| l * v * v).sum::<f64>();
    let grad = |x: &[f64]| {
        x.iter()
            .zip(&lambda)
            .map(|(v, l)| l * v)
            .collect::<Vec<_>>()
    };
    let start: Vec<f64> = lambda
        .iter()
        .map(|l| 25.0 * tol / 2f64.sqrt() / l)
        .collect();
    let start_bits: Vec<u64> = start.iter().map(|v| v.to_bits()).collect();
    for (label, lbfgs) in [("dense", false), ("lbfgs", true)] {
        let run = |stall: &dyn Fn(&[f64]) -> bool| {
            let trials = std::cell::RefCell::new(Vec::new());
            let logged = |x: &[f64]| {
                trials
                    .borrow_mut()
                    .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                obj(x)
            };
            let mut x = start.clone();
            if lbfgs {
                lbfgs_core(&logged, &grad, &mut x, 2, 4, tol, None, None, stall, false);
            } else {
                dense_bfgs_core(
                    &logged, &grad, &mut x, 2, 4, tol, None, None, None, stall, false,
                );
            }
            trials.into_inner()
        };
        let a = run(&|_| true);
        let b = run(&|x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>() != start_bits);
        assert!(
            a.len() > 3,
            "{label}: fixture premise, the search takes several steps"
        );
        assert_eq!(
            a, b,
            "{label}: the pair of a step onto an ODE objective was judged as an exact one"
        );
    }
}

/// Every arm of [`choose_step`], the one step rule of both BFGS drivers and the dense
/// driver's steepest-descent reset (#2280 review), on one 1-D fixture:
/// `f = 10 + ½·256·x² + e(x)` from `x₀ = 2⁻²³`, where `e` adds 5e-11 (inside the resolution
/// `1e-11·(1 + |f|)`) at the end point of the step under test. The full Newton step `d = −x₀`
/// lands on `0`, where ∇ = 0; the short step `d = −x₀/4` lands on `¾x₀`, where ∇ falls by a
/// quarter only.
///
/// | objective | gradient at the end point | arm |
/// |---|---|---|
/// | exact | 0 | gradient-verified step, its gradient reused |
/// | ODE at its noise floor (not improving) | 0 | noise-floor verified step, its gradient reused |
/// | ODE, gradient still improving | 0 | ordinary line search, shorter step, nothing reused |
/// | ODE at its noise floor | ¾ of the start's | converged at the floor |
/// | exact, ascent direction | — | failed |
#[test]
fn choose_step_takes_each_arm() {
    let lambda = 256.0;
    let x0 = 2f64.powi(-23);
    let g0 = [lambda * x0];
    let f0 = 10.0 + 0.5 * lambda * x0 * x0;
    let grad = |x: &[f64]| vec![lambda * x[0]];
    let run = |end: f64, d: f64, stall: bool, improving: bool| {
        let obj = move |x: &[f64]| {
            10.0 + 0.5 * lambda * x[0] * x[0] + if x[0] == end { 5e-11 } else { 0.0 }
        };
        let mut buf = [0.0];
        choose_step(
            &obj,
            &grad,
            &[x0],
            &[d],
            &g0,
            f0,
            None,
            stall,
            improving,
            &mut buf,
            false,
        )
    };
    let newton = -x0;
    match run(0.0, newton, false, false) {
        StepChoice::Step { alpha, g_new, .. } => {
            assert_eq!(alpha, 1.0, "exact: the verified full step");
            assert_eq!(
                g_new,
                Some(vec![0.0]),
                "exact: the end point's gradient is handed back"
            );
        }
        _ => panic!("exact: the verified step must be taken"),
    }
    match run(0.0, newton, true, false) {
        StepChoice::Step { alpha, g_new, .. } => {
            assert_eq!(alpha, 1.0, "ODE floor: the verified full step");
            assert_eq!(
                g_new,
                Some(vec![0.0]),
                "ODE floor: the end point's gradient is handed back"
            );
        }
        _ => panic!("ODE floor: the verified step must be taken"),
    }
    match run(0.0, newton, true, true) {
        StepChoice::Step { alpha, g_new, .. } => {
            assert!(
                alpha > 0.0 && alpha < 1.0,
                "ODE, improving: a shorter step, α = {alpha}"
            );
            assert_eq!(g_new, None, "ODE, improving: no gradient to reuse");
        }
        _ => panic!("ODE, improving: the line search must take a step"),
    }
    let short = -0.25 * x0;
    assert!(
        matches!(
            run(0.75 * x0, short, true, false),
            StepChoice::ConvergedAtFloor
        ),
        "ODE floor, ∇ not halved: converged at the floor"
    );
    assert!(
        matches!(run(0.0, x0, false, false), StepChoice::Failed),
        "an ascent direction: failed"
    );
}

/// A sub-resolution step the objective cannot verify is taken only when the gradient norm at
/// least **halves** there (#2270, PR #2280 review). The same geometry as
/// `only_an_exact_objective_takes_the_gradient_verified_step` with the seed's curvature four times
/// too high (1024 = 32², so it inverts exactly): the full step lands on `x₀·3/4`, where `f` is
/// higher by ~4.9e-11 (inside the resolution) and the gradient is ¾ of the start's, lower but not
/// halved. The verification must refuse it, and the line search, which rejects the rise, takes a
/// shorter step instead. With the halving test weakened to "any decrease", the step was taken.
#[test]
fn a_verified_step_must_halve_the_gradient() {
    let lambda = 256.0;
    let x0 = 2f64.powi(-23);
    let cand = 0.75 * x0;
    let obj =
        |x: &[f64]| 10.0 + 0.5 * lambda * x[0] * x[0] + if x[0] == cand { 5e-11 } else { 0.0 };
    let grad = |x: &[f64]| vec![lambda * x[0]];
    let seed = DMatrix::from_element(1, 1, 4.0 * lambda);
    // The premise: the step's gain is below the resolution, its end point lies within the
    // resolution above the start, and the gradient there falls by a quarter, not a half.
    let f0 = obj(&[x0]);
    let resolution = INNER_FTOL_REL * (1.0 + f0.abs());
    assert!(
        obj(&[cand]) > f0 && obj(&[cand]) <= f0 + resolution,
        "fixture premise: f"
    );
    assert_eq!(
        grad(&[cand])[0] / grad(&[x0])[0],
        0.75,
        "fixture premise: ‖∇‖"
    );
    let mut x = vec![x0];
    dense_bfgs_core(
        &obj,
        &grad,
        &mut x,
        1,
        1,
        1e-14,
        None,
        Some(&seed),
        None,
        &|_| false,
        false,
    );
    assert!(
        x[0] != cand && x[0] > cand && x[0] < x0,
        "a step that lowers ‖∇‖ by a quarter must go to the line search, which shortens it: \
         x = {:e}, the unverified end point {cand:e}",
        x[0]
    );
}

/// The gradient-verified step of #2270 is for exact objectives only; an ODE objective
/// (`enable_stall`) keeps its line search and its own noise-floor exit. A differential pair
/// across the gate: `f = 10 + ½·256·x² + e(x)` with `e(0) = 5e-11` and `e = 0` elsewhere,
/// started at `x₀ = 2⁻²³`, Newton-seeded (256 = 16², so the seed inverts exactly) so the full
/// step lands exactly on `0`. There `f` is
/// higher by ~4.8e-11, inside the resolution `1e-11·(1 + |f|)`, while the gradient is 0. One
/// iteration: the exact objective takes the step (x = 0), the line search rejects it and takes a
/// shorter one (x ≠ 0).
#[test]
fn only_an_exact_objective_takes_the_gradient_verified_step() {
    let lambda = 256.0;
    let obj = |x: &[f64]| 10.0 + 0.5 * lambda * x[0] * x[0] + if x[0] == 0.0 { 5e-11 } else { 0.0 };
    let grad = |x: &[f64]| vec![lambda * x[0]];
    let seed = DMatrix::from_element(1, 1, lambda);
    let x0 = 2f64.powi(-23);
    let run = |stall: &dyn Fn(&[f64]) -> bool| {
        let mut x = vec![x0];
        dense_bfgs_core(
            &obj,
            &grad,
            &mut x,
            1,
            1,
            1e-14,
            None,
            Some(&seed),
            None,
            stall,
            false,
        );
        x[0]
    };
    let exact = run(&|_| false);
    let ode = run(&|_| true);
    assert_eq!(
        exact, 0.0,
        "the exact objective must take the gradient-verified step"
    );
    assert!(
        ode != 0.0 && ode.abs() < x0,
        "the ODE objective must leave the step to its line search, which shortens it: x = {ode:e}"
    );
}

/// L-BFGS's empty-history direction on the identity path takes the same 1/‖g‖ first-step
/// scaling dense BFGS always had (#2099): from a steep start the first trial point is one unit
/// along `−g/‖g‖`.
#[test]
fn lbfgs_scales_its_identity_first_step_like_dense_bfgs() {
    use std::cell::RefCell;
    let trials = RefCell::new(Vec::new());
    let obj = |x: &[f64]| {
        trials.borrow_mut().push(x.to_vec());
        0.5e6 * (x[0] - 1.0).powi(2) + 0.5e6 * (x[1] + 2.0).powi(2)
    };
    let grad = |x: &[f64]| vec![1e6 * (x[0] - 1.0), 1e6 * (x[1] + 2.0)];
    let never = |_: &[f64]| false;
    let mut x = vec![0.0, 0.0];
    lbfgs_core(&obj, &grad, &mut x, 2, 1, 1e-8, None, None, &never, false);
    let t = &trials.borrow()[1];
    let step = (t[0].powi(2) + t[1].powi(2)).sqrt();
    assert!(
        (step - 1.0).abs() < 1e-12,
        "first trial {t:?} is {step} from the start, not 1"
    );
    let g0 = [-1e6f64, 2e6];
    let gn = (g0[0] * g0[0] + g0[1] * g0[1]).sqrt();
    assert!((t[0] + g0[0] / gn).abs() < 1e-12 && (t[1] + g0[1] / gn).abs() < 1e-12);
}

/// The two #2080 outcome fixtures above, under `inner_optimizer = lbfgs` (#2099). Three
/// random effects take dense BFGS under `auto`, so those tests never reach `lbfgs_core`;
/// on the #2091 branch, before L-BFGS's empty-history direction was bounded, this seed
/// went through the Nelder–Mead fallback and the multi-start kept the twin at −21.073865.
/// Mutation: deleting the `s_hist.is_empty()` bound in `lbfgs_core` reddens both.
fn armed_lbfgs() -> InnerSettingsGuard {
    arm_inner_settings(InnerSettings {
        mode: crate::types::InnerOptimizer::Lbfgs,
        warm: false,
    })
}

#[test]
fn lbfgs_far_seed_reaches_the_deeper_mode_from_a_bounded_first_step() {
    let _lbfgs = armed_lbfgs();
    let (trapped, r) = flip_flop_eta_v_minus_2sd(5.0, [0.04, 0.04, 2.0]);
    eprintln!(
        "config 2, lbfgs: trapped {:.6}, from seed {:.6}, fallback {}",
        trapped.nll, r.nll, r.used_fallback
    );
    assert!(trapped.nll.is_finite() && r.nll.is_finite());
    assert!(
        !r.used_fallback,
        "L-BFGS must converge from the seed itself, not through the Nelder–Mead fallback"
    );
    assert!(
        r.nll < trapped.nll - 1.0,
        "the seed must reach the deeper mode: {} vs trapped {}",
        r.nll,
        trapped.nll
    );
}

#[test]
fn lbfgs_cold_multi_start_keeps_the_far_seed() {
    let _lbfgs = armed_lbfgs();
    let (model, mut subject) = flip_flop_oral_fixture(5.0, [0.04, 0.04, 2.0]);
    subject.obs_covariates = (0..subject.obs_times.len())
        .map(|i| HashMap::from([("WT".to_string(), 70.0 + i as f64)]))
        .collect();
    assert!(
        subject.has_tv_covariates() && !subject.has_resets(),
        "test premise: the subject is on the scan-all arm through its covariates"
    );
    let params = &model.default_params;
    let single = find_ebe(&model, &subject, params, 200, 1e-5, None, None, 0);
    let multi = find_ebe(&model, &subject, params, 200, 1e-5, None, None, 1);
    eprintln!(
        "config 2 multi-start, lbfgs: r=0 {:.6}, r=1 {:.6}",
        single.nll, multi.nll
    );
    assert!(single.nll.is_finite() && multi.nll.is_finite());
    assert!(
        multi.nll < single.nll - 1.0,
        "inner_restarts = 1 must keep the deeper mode: {} vs r = 0 {}",
        multi.nll,
        single.nll
    );
}

/// Dense BFGS's steepest-descent reset is bounded like its first step (#2099). The reset
/// needs `dᵀg ≥ 0` from a BFGS-updated `h_inv`, which in exact arithmetic is PD and never
/// produces one; this fixture gets it from rounding. A fake gradient returns
/// `g₀ = (2⁻⁵, 2⁻⁵)` at the start and `g₁ ≈ (2.1e7, −4.8e7)` everywhere else, so the
/// accepted unit first step `s = −diag(p)·g₀` and `y = g₁ − g₀` are nearly orthogonal
/// (`sᵀy ≈ 5.7e-3` against `‖s‖‖y‖ ≈ 2e5`) and `g₁` lies almost along `y`, where the true
/// quadratic form `g₁ᵀH₁g₁` is tiny; the computed one comes out negative. Found by a
/// float-for-float search of the update; the premise is asserted below rather than assumed.
/// The reset direction `−diag(p)·g₁` is ~2e8 prior SDs long, so its first trial must sit on
/// the runaway radius, and so must the step after it: the reset arm skips the BFGS update,
/// so the third iteration's direction comes from the same unshaped metric. Mutations:
/// deleting the reset's `cap_curvature_free_step` call reddens the first bound; dropping
/// `h_unshaped` (capping only the first iterate) reddens the second. (L-BFGS's non-descent fallback is bounded the same way; the same
/// search over its two-loop recursion found no such fixture.)
#[test]
fn dense_bfgs_steepest_descent_reset_takes_a_bounded_first_trial() {
    use std::cell::RefCell;
    let p = [0.09, 0.04];
    let g0 = [0.03125, 0.03125];
    let g1 = [21474834.48, -48318382.064375];
    let r2 = RUNAWAY_EBE_MAHA_PER_ETA * 2.0;
    let never = |_: &[f64]| false;
    let trials: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::new());
    // Steep enough that the first step and the reset step both pass Armijo, so the solve
    // reaches a third iteration from the still-unshaped reset metric.
    let obj = |x: &[f64]| {
        trials.borrow_mut().push(x.to_vec());
        -1e9 * (x[0] * x[0] + x[1] * x[1]).sqrt()
    };
    let grad = |x: &[f64]| {
        if x == [0.0, 0.0] {
            g0.to_vec()
        } else {
            g1.to_vec()
        }
    };
    let mut x = vec![0.0, 0.0];
    dense_bfgs_core(
        &obj,
        &grad,
        &mut x,
        2,
        3,
        1e-8,
        Some(&p),
        None,
        None,
        &never,
        false,
    );
    let t = trials.borrow();
    // trials[0] is the start; [1] the first iteration's accepted unit step.
    let x1 = [-p[0] * g0[0], -p[1] * g0[1]];
    assert_eq!(
        t[1], x1,
        "fixture premise: the first step is −diag(p)·g₀ at α = 1"
    );
    // Premise: the BFGS direction at x₁ is not a descent direction, so the solve resets.
    // The same update `dense_bfgs_core` applies, in the same operation order.
    let s = DVector::from_column_slice(&x1);
    let y = DVector::from_column_slice(&[g1[0] - g0[0], g1[1] - g0[1]]);
    let rho = 1.0 / s.dot(&y);
    let eye = DMatrix::<f64>::identity(2, 2);
    let h0 = DMatrix::from_diagonal(&DVector::from_column_slice(&p));
    let h1 = (&eye - rho * &s * y.transpose()) * &h0 * (&eye - rho * &y * s.transpose())
        + rho * &s * s.transpose();
    let g1v = DVector::from_column_slice(&g1);
    let d_bfgs = -(&h1 * &g1v);
    assert!(
        d_bfgs.dot(&g1v) >= 0.0,
        "fixture premise: the BFGS direction must be non-descent, dᵀg = {}",
        d_bfgs.dot(&g1v)
    );
    let reset = &t[2];
    let step = [reset[0] - x1[0], reset[1] - x1[1]];
    let m: f64 = step.iter().zip(&p).map(|(si, pi)| si * si / pi).sum();
    assert!(
        m.is_finite() && m <= r2 * (1.0 + 1e-9),
        "the reset's first trial {reset:?} is {m:.3e} from x₁ in the prior metric, radius² {r2}"
    );
    // The direction is still steepest descent in the preconditioned metric.
    let sd = [-p[0] * g1[0], -p[1] * g1[1]];
    assert!(
        (step[0] / step[1] - sd[0] / sd[1]).abs() < 1e-9 * (sd[0] / sd[1]).abs(),
        "the reset step {step:?} must keep −diag(p)·g₁'s direction {sd:?}"
    );
    // The reset step is accepted with no BFGS update (the reset arm skips it), so the third
    // iteration's `−h_inv·g₁` is the same curvature-free direction and must be bounded too.
    assert!(
        t.len() >= 4,
        "fixture premise: the reset step is accepted and a third iteration runs"
    );
    let next = [t[3][0] - t[2][0], t[3][1] - t[2][1]];
    let m: f64 = next.iter().zip(&p).map(|(si, pi)| si * si / pi).sum();
    assert!(
        m.is_finite() && m <= r2 * (1.0 + 1e-9),
        "the step after the reset, {next:?}, is {m:.3e} in the prior metric, radius² {r2}"
    );
}

/// The identity-metric twin of the test above: the IOV inner solve passes `precond = None`,
/// where `cap_curvature_free_step` is a no-op and the reset must take `h0_step_scale`
/// instead (#2099 review). Fixture, from the same float-for-float search with `H₀ = I`:
/// `g₀ = (−2⁻¹⁰, 2⁻¹⁰)` (‖g₀‖ < 1, so the first step is the unscaled `−g₀`) and
/// `g₁ = (2²³, 2²³)` everywhere else, for which the computed `H₁·g₁` gives `dᵀg = 0`. The
/// reset's first trial must then be exactly one unit along `−g₁/‖g₁‖`, not the
/// 1.2e7-long `−g₁`, and so must the next iteration's, which steps from the same reset
/// metric. Mutations: dropping the reset's scaling reddens the first; scaling only the
/// reset's direction, not `h_inv` (#2099 review 2), reddens the second.
#[test]
fn dense_bfgs_identity_steepest_descent_reset_takes_a_scaled_first_trial() {
    use std::cell::RefCell;
    let g0 = [-0.0009765625, 0.0009765625];
    let g1 = [8388608.0, 8388608.0];
    let never = |_: &[f64]| false;
    let trials: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::new());
    // Steep enough that the first step and the reset step both pass Armijo, so the solve
    // reaches a third iteration from the still-unshaped reset metric.
    let obj = |x: &[f64]| {
        trials.borrow_mut().push(x.to_vec());
        -1e9 * (x[0] * x[0] + x[1] * x[1]).sqrt()
    };
    let grad = |x: &[f64]| {
        if x == [0.0, 0.0] {
            g0.to_vec()
        } else {
            g1.to_vec()
        }
    };
    let mut x = vec![0.0, 0.0];
    dense_bfgs_core(
        &obj, &grad, &mut x, 2, 3, 1e-8, None, None, None, &never, false,
    );
    let t = trials.borrow();
    let x1 = [-g0[0], -g0[1]];
    assert_eq!(t[1], x1, "fixture premise: the first step is −g₀ at α = 1");
    // Premise: the BFGS direction at x₁ is not a descent direction, so the solve resets.
    let s = DVector::from_column_slice(&x1);
    let y = DVector::from_column_slice(&[g1[0] - g0[0], g1[1] - g0[1]]);
    let rho = 1.0 / s.dot(&y);
    let eye = DMatrix::<f64>::identity(2, 2);
    let h1 = (&eye - rho * &s * y.transpose()) * &eye * (&eye - rho * &y * s.transpose())
        + rho * &s * s.transpose();
    let g1v = DVector::from_column_slice(&g1);
    let d_bfgs = -(&h1 * &g1v);
    assert!(
        d_bfgs.dot(&g1v) >= 0.0,
        "fixture premise: the BFGS direction must be non-descent, dᵀg = {}",
        d_bfgs.dot(&g1v)
    );
    let gn = (g1[0] * g1[0] + g1[1] * g1[1]).sqrt();
    let want = [x1[0] - g1[0] / gn, x1[1] - g1[1] / gn];
    let reset = &t[2];
    assert!(
        (reset[0] - want[0]).abs() < 1e-12 && (reset[1] - want[1]).abs() < 1e-12,
        "the reset's first trial {reset:?} must be x₁ − g₁/‖g₁‖ = {want:?}"
    );
    // The reset step is accepted with no BFGS update, so the third iteration steps from the
    // same reset metric: the scaling must live in `h_inv`, not only in the reset's direction.
    assert!(
        t.len() >= 4,
        "fixture premise: the reset step is accepted and a third iteration runs"
    );
    let x2 = &t[2];
    let want = [x2[0] - g1[0] / gn, x2[1] - g1[1] / gn];
    assert!(
        (t[3][0] - want[0]).abs() < 1e-12 && (t[3][1] - want[1]).abs() < 1e-12,
        "the trial after the reset {:?} must be x₂ − g₁/‖g₁‖ = {want:?}",
        t[3]
    );
}

/// #2209: a residual magnitude (`weight =`, or a θ in the magnitude) combined with a dense
/// `R` (`correlation`, `block_sigma`) is served exactly by the dense-`R` inner kernel,
/// which has threaded the per-observation multiplier since #627. The model-level label,
/// the banner and the route all read one predicate, so they must agree on it.
mod weighted_dense_residual_inner {
    use super::*;
    use crate::build_info::{gradient_method_inner, GradientMethodKind, BUILD_INFO};
    use crate::types::{DoseEvent, Population, Subject};

    const BASE: &str = "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  theta RHO(0.5, -0.4, 0.95) FIX\n  theta TH_M(1.7, 0.1, 10.0) FIX\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.2 (sd)\n  sigma ADD_ERR ~ 0.5 (sd)\n[covariates]\n  W continuous\n  ARM continuous\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n";

    /// `(cell, error-model lines, gradient = fd)`. The first three are the weighted dense
    /// cells #2209 is about; then the two single-feature controls; then the forced-FD twin.
    const CELLS: &[(&str, &str, bool)] = &[
        (
            "weight x correlation",
            "  DV ~ combined(PROP_ERR, ADD_ERR) weight = W\n  correlation ar1(RHO) by ARM\n",
            false,
        ),
        (
            "weight x block_sigma",
            "  DV ~ combined(PROP_ERR, ADD_ERR) weight = W\n",
            false,
        ),
        (
            "theta-magnitude x correlation",
            "  DV ~ combined(PROP_ERR * TH_M, ADD_ERR)\n  correlation ar1(RHO) by ARM\n",
            false,
        ),
        (
            "correlation alone",
            "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation ar1(RHO) by ARM\n",
            false,
        ),
        (
            "weight alone",
            "  DV ~ combined(PROP_ERR, ADD_ERR) weight = W\n",
            false,
        ),
        (
            "weight x correlation, gradient = fd",
            "  DV ~ combined(PROP_ERR, ADD_ERR) weight = W\n  correlation ar1(RHO) by ARM\n",
            true,
        ),
    ];

    fn model_for(cell: &str, error: &str, fd: bool) -> CompiledModel {
        let mut src = BASE.replace("  DV ~ combined(PROP_ERR, ADD_ERR)\n", error);
        if cell.contains("block_sigma") {
            src = src.replace(
                "  sigma PROP_ERR ~ 0.2 (sd)\n  sigma ADD_ERR ~ 0.5 (sd)\n",
                "  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 0.25] FIX\n",
            );
        }
        let mut model = crate::parser::model_parser::parse_model_string(&src)
            .unwrap_or_else(|e| panic!("{cell}: parse: {e}"));
        if fd {
            // What `fit()` sets from `gradient = fd` before the inner loop runs.
            model.gradient_method = GradientMethod::Fd;
        }
        model
    }

    /// One-cpt IV, doses at 0 and 12 (residual drug present at the second), two
    /// interleaved arms, and a per-row weight that is live in the objective.
    fn subject_for(model: &CompiledModel) -> Subject {
        let times = [0.5, 2.0, 6.0, 12.5, 14.0, 18.0, 24.0];
        let arms = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        let ws = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 1.5];
        let n = times.len();
        let mut s = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: times.to_vec(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            obs_covariates: arms
                .iter()
                .zip(ws)
                .map(|(a, w)| HashMap::from([("ARM".to_string(), *a), ("W".to_string(), w)]))
                .collect(),
            cens: vec![0; n],
            ..Default::default()
        };
        s.obs_rgroup = crate::stats::residual_error::residual_groups(model, &s);
        let theta = &model.default_params.theta;
        let preds = crate::pk::compute_predictions_with_tv(model, &s, theta, &[0.1, -0.05]);
        s.observations = preds
            .iter()
            .enumerate()
            .map(|(i, v)| v * 1.15 + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        s
    }

    /// T1: label, banner and route per cell, in one test so a gate stuck on either
    /// value fails. Kills: re-adding the magnitude × dense clause to
    /// `analytic_inner_common_bail` (label FD, `exact > 0`); `inner_reports_analytic_model`
    /// forced `true` (the `gradient = fd` cell); dropping the analytic-arm increment
    /// (`analytic == 0`).
    #[test]
    fn weighted_dense_residual_reports_and_runs_the_analytic_inner_gradient() {
        for &(cell, error, fd) in CELLS {
            let model = model_for(cell, error, fd);
            let s = subject_for(&model);
            if cell.contains("weight") || cell.contains("magnitude") {
                assert!(model.has_custom_ruv_magnitude(), "{cell}: magnitude live");
            }
            if cell.contains("correlation") || cell.contains("block_sigma") {
                assert!(model.residual_is_dense(), "{cell}: dense R");
            }
            let label = gradient_method_inner(&BUILD_INFO, &model);
            let pop = Population {
                subjects: vec![s.clone()],
                covariate_names: Vec::new(),
                dv_column: "DV".to_string(),
                input_columns: vec![],
                exclusions: None,
                warnings: Vec::new(),
            };
            let requested = if fd {
                GradientMethod::Fd
            } else {
                GradientMethod::Auto
            };
            let banner = gradient_route_summary(&model, &pop, requested);
            INNER_AGRAD_ROUTE.with(|c| c.set((0, 0, 0)));
            let ebe = find_ebe_for_stage(
                &model,
                &s,
                &model.default_params,
                200,
                1e-8,
                None,
                None,
                0,
                &crate::types::FitOptions::default(),
            );
            let (analytic, exact, fdn) = INNER_AGRAD_ROUTE.with(|c| c.get());
            assert!(ebe.nll.is_finite(), "{cell}: nll {}", ebe.nll);
            if fd {
                assert_eq!(label, GradientMethodKind::FiniteDifferences, "{cell}");
                assert!(
                    banner.starts_with("FD  [requested: FD]"),
                    "{cell}: {banner}"
                );
                assert!(
                    analytic == 0 && exact == 0 && fdn > 0,
                    "{cell}: {analytic} analytic / {exact} exact / {fdn} FD steps"
                );
            } else {
                assert_eq!(label, GradientMethodKind::Analytic, "{cell}");
                assert!(
                    banner.starts_with("analytic (Dual2)  [requested: auto]"),
                    "{cell}: {banner}"
                );
                assert!(
                    analytic > 0 && exact == 0 && fdn == 0,
                    "{cell}: {analytic} analytic / {exact} exact / {fdn} FD steps"
                );
            }
        }
    }

    /// T2: the dense-`R` inner kernel against central FD of `individual_nll` at a
    /// nonzero η, on the three weighted dense cells. Kills: building the kernel's `R`
    /// without the magnitude multiplier (`mult → None`).
    #[test]
    fn weighted_dense_residual_inner_gradient_matches_fd() {
        let eta = [0.17, -0.11];
        let mut worst = 0.0_f64;
        for &(cell, error, fd) in &CELLS[..3] {
            assert!(!fd);
            let model = model_for(cell, error, false);
            let s = subject_for(&model);
            let p = &model.default_params;
            let g =
                analytic_eta_nll_gradient(&model, &s, &p.theta, &eta, &p.omega, &p.sigma.values)
                    .unwrap_or_else(|| panic!("{cell}: the dense kernel serves the subject"));
            let nll = |e: &[f64]| {
                crate::stats::likelihood::individual_nll(
                    &model,
                    &s,
                    &p.theta,
                    e,
                    &p.omega,
                    &p.sigma.values,
                )
            };
            for k in 0..eta.len() {
                let h = 1e-5;
                let (mut a, mut b) = (eta.to_vec(), eta.to_vec());
                a[k] += h;
                b[k] -= h;
                let want = (nll(&a) - nll(&b)) / (2.0 * h);
                // Finite first: `f64::max` would discard a NaN and keep the rows that worked.
                assert!(g[k].is_finite() && want.is_finite(), "{cell}[{k}]");
                let rel = (g[k] - want).abs() / (1.0 + want.abs());
                eprintln!("{cell}[{k}]: kernel {} FD {want} rel {rel:.3e}", g[k]);
                worst = worst.max(rel);
            }
        }
        // Measured worst 6.1e-10 (θ-magnitude × correlation, macOS arm64); 1e-7 is
        // ~160× headroom. With `mult` dropped from the kernel's `R` it fails.
        eprintln!("worst relative gap {worst:.3e}");
        assert!(worst < 1e-7, "worst relative gap {worst:.3e}");
    }

    /// T3 (#2220): T2 under non-interaction FOCE. With `frozen = Some(f(η = 0))` the
    /// dense kernel takes its frozen branch (#1722) — `R⁰` built once at the population
    /// predictions, `∂R/∂η = 0` — and is checked against central FD of the frozen
    /// objective (`individual_nll_into_prepared_with_schedule_frozen`, which applies the
    /// same `mult`) at a nonzero η, on the three weighted dense cells. A straddle per
    /// cell asserts the frozen gradient is not the conditional one, so the FD half
    /// cannot pass on a kernel that ignores `frozen`. Kills: dropping `mult` from the
    /// frozen-branch `R⁰` build only — the conditional branch, and so T2, stays green.
    #[test]
    fn weighted_dense_residual_frozen_inner_gradient_matches_fd() {
        use crate::stats::likelihood::{
            individual_nll_into_prepared_with_schedule_frozen, population_variance_preds,
        };
        let eta = [0.17, -0.11];
        let (mut worst, mut min_sep) = (0.0_f64, f64::INFINITY);
        for &(cell, error, fd) in &CELLS[..3] {
            assert!(!fd);
            let model = model_for(cell, error, false);
            let s = subject_for(&model);
            let p = &model.default_params;
            let mult = model.ruv_obs_mult(&s, &p.theta);
            assert!(mult.is_some(), "{cell}: magnitude multiplier live");
            let keys = model.error_spec.obs_keys(&s);
            let f0 = population_variance_preds(&model, &s, &p.theta, model.n_eta)
                .unwrap_or_else(|| panic!("{cell}: combined error has an f-dependent R"));
            let grad = |frozen: Option<&[f64]>| {
                analytic_eta_nll_gradient_with_schedule(
                    &model,
                    &s,
                    &p.theta,
                    &eta,
                    &p.omega,
                    &p.sigma.values,
                    model.residual_cov_at(p),
                    None,
                    mult.as_deref(),
                    keys.as_ref(),
                    &mut Vec::new(),
                    &mut DVector::zeros(model.n_eta),
                    &mut DVector::zeros(model.n_eta),
                    frozen,
                )
                .unwrap_or_else(|| panic!("{cell}: the dense kernel serves the subject"))
            };
            let nll = |e: &[f64]| {
                individual_nll_into_prepared_with_schedule_frozen(
                    &model,
                    &s,
                    &p.theta,
                    e,
                    &p.omega,
                    &p.sigma.values,
                    model.residual_cov_at(p),
                    &mut pk::EventPkParams::default(),
                    None,
                    keys.as_ref(),
                    mult.as_deref(),
                    &mut Vec::new(),
                    &mut DVector::zeros(e.len()),
                    &mut DVector::zeros(e.len()),
                    Some(&f0),
                )
            };
            let (g, gc) = (grad(Some(&f0)), grad(None));
            for k in 0..eta.len() {
                let h = 1e-5;
                let (mut a, mut b) = (eta.to_vec(), eta.to_vec());
                a[k] += h;
                b[k] -= h;
                let want = (nll(&a) - nll(&b)) / (2.0 * h);
                // Finite first: `f64::max` would discard a NaN and keep the rows that worked.
                assert!(
                    g[k].is_finite() && gc[k].is_finite() && want.is_finite(),
                    "{cell}[{k}]"
                );
                let rel = (g[k] - want).abs() / (1.0 + want.abs());
                eprintln!(
                    "{cell}[{k}]: frozen {} conditional {} FD {want} rel {rel:.3e}",
                    g[k], gc[k]
                );
                worst = worst.max(rel);
            }
            let sep = g
                .iter()
                .zip(&gc)
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(sep > 1e-2, "{cell}: frozen vs conditional gradient {sep:e}");
            min_sep = min_sep.min(sep);
        }
        // Measured worst 8.6e-10 (weight × block_sigma, macOS arm64); 1e-7 is ~115×
        // headroom. With `mult` dropped from the frozen `R⁰` only, the worst gap is 1.2.
        // Smallest frozen/conditional separation measured 2.7, so the 1e-2 straddle
        // bound above sits ~270× below it.
        eprintln!("worst relative gap {worst:.3e}, smallest frozen/conditional gap {min_sep:.3e}");
        assert!(worst < 1e-7, "worst relative gap {worst:.3e}");
    }
}

/// `[binary_model]` on the analytic inner route (#2024 phase 1): the logit data term's
/// exact η-gradient `(p − y)·∂lp/∂η`, its routing, and its fold into the full inner gradient.
#[cfg(feature = "survival")]
mod binary_inner {
    use crate::types::{CompiledModel, EventType, ObsRecord, Population, Subject};

    /// Endpoint-only mixed-effects logistic with **two** live η: an intercept on the log-odds
    /// scale (`ETA_I`) and one inside a log-normal slope on `TIME` (`ETA_S`), so `∂lp/∂η_S`
    /// differs per record and the per-record model-time scoping is on the path. `X` is a
    /// baseline covariate (the predictor takes no time-varying one). An endpoint-only model
    /// takes no `[individual_parameters]`; the joint fixture below covers that route.
    const ENDPOINT_ONLY: &str = r"
[parameters]
  theta TH0(-0.3, -10.0, 10.0)
  theta THX(0.6, -10.0, 10.0)
  theta THT(0.4, -10.0, 10.0)
  omega ETA_I ~ 0.3
  omega ETA_S ~ 0.2

[binary_model]
  cmt   = 3
  logit = TH0 + THX * X + THT * exp(ETA_S) * TIME + ETA_I
";

    fn parse(src: &str) -> CompiledModel {
        crate::parser::model_parser::parse_model_string(src).expect("parse")
    }

    fn binary_records() -> Vec<ObsRecord> {
        [(0.5, 0), (1.0, 1), (2.0, 1), (3.0, 0), (4.5, 1)]
            .iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                raw_time: time,
                state,
                cmt: 3,
            })
            .collect()
    }

    fn subject() -> Subject {
        Subject {
            id: "1".into(),
            obs_records: binary_records(),
            covariates: [("X".to_string(), 1.3)].into_iter().collect(),
            ..Default::default()
        }
    }

    fn inner_grad(model: &CompiledModel, subject: &Subject, eta: &[f64]) -> Option<Vec<f64>> {
        let params = &model.default_params;
        let err_keys = model.error_spec.obs_keys(subject);
        super::super::analytic_eta_nll_gradient_with_schedule(
            model,
            subject,
            &params.theta,
            eta,
            &params.omega,
            &params.sigma.values,
            model.residual_cov_at(params),
            None,
            None,
            err_keys.as_ref(),
            &mut Vec::new(),
            &mut nalgebra::DVector::zeros(model.n_eta),
            &mut nalgebra::DVector::zeros(model.n_eta),
            None,
        )
    }

    /// Central FD of `individual_nll` — the exact objective the EBE search minimizes —
    /// asserted coordinate by coordinate against the analytic inner gradient. The worst
    /// realised gap is printed so the bound below stays measured.
    fn assert_inner_grad_matches_fd(model: &CompiledModel, subject: &Subject, eta: &[f64]) {
        let params = &model.default_params;
        let g = inner_grad(model, subject, eta).expect("subject is in analytic scope");
        let nll = |e: &[f64]| {
            crate::stats::likelihood::individual_nll(
                model,
                subject,
                &params.theta,
                e,
                &params.omega,
                &params.sigma.values,
            )
        };
        let h = 1e-6;
        let mut worst = 0.0_f64;
        for k in 0..eta.len() {
            let mut ep = eta.to_vec();
            let mut em = eta.to_vec();
            ep[k] += h;
            em[k] -= h;
            let fd = (nll(&ep) - nll(&em)) / (2.0 * h);
            assert!(
                g[k].is_finite() && fd.is_finite(),
                "η[{k}]: {} vs {fd}",
                g[k]
            );
            let gap = (g[k] - fd).abs();
            worst = worst.max(gap);
            // FD truncation + cancellation at h = 1e-6 on an O(1) objective is ~1e-9;
            // measured worst 5.1e-9 (joint fixture), so 1e-6 is ~200× headroom.
            assert!(
                gap < 1e-6,
                "η = {eta:?}, axis {k}: analytic {}, FD {fd}",
                g[k]
            );
        }
        eprintln!("binary inner grad worst |analytic − FD| at η = {eta:?}: {worst:.3e}");
    }

    /// The endpoint kernel alone: value equals `discrete_subject_nll` (same sum, other
    /// evaluator) and the η-gradient matches a central FD of it, on both η.
    #[test]
    fn discrete_subject_eta_grad_matches_fd() {
        let model = parse(ENDPOINT_ONLY);
        let s = subject();
        let theta = &model.default_params.theta;
        for eta in [[0.0, 0.0], [-0.4, 0.35], [0.7, -0.5]] {
            let (v, g) = crate::categorical::discrete_subject_eta_grad(&model, &s, theta, &eta)
                .expect("binary endpoint has a dual program");
            let want = crate::categorical::discrete_subject_nll(&model, &s, theta, &eta);
            assert!(
                (v - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{v} vs {want}"
            );
            let h = 1e-6;
            for k in 0..2 {
                let mut ep = eta;
                let mut em = eta;
                ep[k] += h;
                em[k] -= h;
                let fd = (crate::categorical::discrete_subject_nll(&model, &s, theta, &ep)
                    - crate::categorical::discrete_subject_nll(&model, &s, theta, &em))
                    / (2.0 * h);
                assert!(fd.abs() > 1e-3, "η[{k}] must be live on the term: FD {fd}");
                assert!(
                    (g[k] - fd).abs() < 1e-7,
                    "η = {eta:?}, axis {k}: analytic {}, FD {fd}",
                    g[k]
                );
            }
        }
    }

    /// An overflowed predictor scores the flat `1e20` repellent, which has no derivative:
    /// the kernel must decline (→ FD for that point) rather than return the score of an
    /// infinite `lp`. The finite twin at the same η is served, so the decline is the
    /// predictor's doing.
    #[test]
    fn an_overflowed_predictor_declines_instead_of_scoring() {
        let model = parse(ENDPOINT_ONLY);
        let s = subject();
        let mut theta = model.default_params.theta.clone();
        assert!(
            crate::categorical::discrete_subject_eta_grad(&model, &s, &theta, &[0.0, 0.0])
                .is_some()
        );
        theta[2] = 1e308; // THT · TIME overflows to +∞
        assert!(
            crate::categorical::discrete_subject_nll(&model, &s, &theta, &[0.0, 0.0]) >= 1e20,
            "fixture precondition: the value path repels"
        );
        assert!(
            crate::categorical::discrete_subject_eta_grad(&model, &s, &theta, &[0.0, 0.0])
                .is_none()
        );
    }

    /// The route must actually change: before #2024 every subject with a binary record
    /// declined the analytic inner gradient, on both the model-level report and the
    /// per-subject gate.
    #[test]
    fn endpoint_only_binary_takes_the_analytic_inner_route() {
        let model = parse(ENDPOINT_ONLY);
        assert!(super::super::analytic_inner_grad_supported_model(&model));
        assert!(super::super::analytic_inner_grad_supported(
            &model,
            &subject()
        ));
        assert_eq!(
            super::super::resolve_gradient_method(&model, &subject()),
            super::super::InnerGradientMethod::Analytic
        );
    }

    #[test]
    fn endpoint_only_inner_gradient_matches_fd_of_individual_nll() {
        let model = parse(ENDPOINT_ONLY);
        let s = subject();
        for eta in [[0.0, 0.0], [-0.4, 0.35], [0.7, -0.5]] {
            assert_inner_grad_matches_fd(&model, &s, &eta);
        }
    }

    /// Joint PK + binary: the same η (`ETA_CL`) is live on the Gaussian concentration rows
    /// *and* on the log-odds (through `CL`), and a second η sits only on the binary term.
    /// The full inner gradient — prior + Gaussian + binary — must match FD of the joint
    /// objective, which is what fails if the binary block is dropped or double-counted.
    #[test]
    fn joint_pk_binary_inner_gradient_matches_fd_of_individual_nll() {
        let src = "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
            theta TH0(-0.4, -10.0, 10.0)\n  theta THC(0.8, -10.0, 10.0)\n  \
            omega ETA_CL ~ 0.09\n  omega ETA_B ~ 0.25\n  sigma PROP ~ 0.1 (sd)\n\
            [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
            [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[binary_model]\n  cmt   = 3\n  \
            logit = TH0 + THC * log(CL) + ETA_B\n[error_model]\n  DV ~ proportional(PROP)\n";
        let model = parse(src);
        let pop = joint_population(&model);
        let s = &pop.subjects[0];
        assert!(!s.obs_times.is_empty() && !s.obs_records.is_empty());
        assert!(super::super::analytic_inner_grad_supported(&model, s));
        for eta in [[0.0, 0.0], [0.25, -0.3], [-0.3, 0.6]] {
            assert_inner_grad_matches_fd(&model, s, &eta);
        }
    }

    fn joint_population(model: &CompiledModel) -> Population {
        let dir = tempfile::tempdir().expect("tempdir");
        let csv = dir.path().join("joint.csv");
        let mut body = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV\n");
        body.push_str("1,0,.,1,100,1,1\n");
        for (t, y) in [(1.0, 1), (2.0, 0), (4.0, 1), (8.0, 0)] {
            let conc = 10.0 * (-0.1_f64 * t).exp() * 1.05;
            body.push_str(&format!("1,{t},{conc},0,.,1,0\n"));
            body.push_str(&format!("1,{t},{y},0,.,3,0\n"));
        }
        std::fs::write(&csv, body).expect("write csv");
        crate::api::read_population_for(model, &None, csv.to_str().unwrap(), None, None, None, &[])
            .expect("routed read")
            .0
    }

    /// A model carrying both a `[binary_model]` and an `[event_model]`. Its binary term is
    /// served; its TTE term is not.
    fn binary_plus_tte() -> CompiledModel {
        parse(
            "[parameters]\n  theta TH0(-0.3, -10.0, 10.0)\n  theta TVLAMBDA(0.05, 0.001, 10.0)\n  \
             omega ETA_I ~ 0.3\n[binary_model]\n  cmt   = 3\n  logit = TH0 + ETA_I\n\
             [event_model]\n  cmt    = 4\n  family = exponential\n  \
             scale  = TVLAMBDA * exp(ETA_I)\n",
        )
    }

    fn event_record() -> ObsRecord {
        ObsRecord::Event {
            time: 6.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 4,
        }
    }

    /// "Every term served" is per record: binary-only rows are admitted on a model that
    /// also declares a TTE endpoint, while a subject carrying one event record — alone or
    /// beside binary rows — declines to FD. Mutating the predicate to admit any
    /// `DiscreteState`-or-other record turns the mixed arm red; admitting every record
    /// turns the event-only arm red too.
    #[test]
    fn a_subject_with_an_unserved_term_declines_to_fd() {
        let model = binary_plus_tte();
        let served = |recs: Vec<ObsRecord>| {
            let s = Subject {
                id: "1".into(),
                obs_records: recs,
                ..Default::default()
            };
            (
                super::super::non_gaussian_eta_grad_served(&model, &s),
                super::super::analytic_inner_grad_supported(&model, &s),
            )
        };
        assert_eq!(served(binary_records()), (true, true), "binary-only");
        let mut mixed = binary_records();
        mixed.push(event_record());
        assert_eq!(served(mixed), (false, false), "binary + event");
        assert_eq!(served(vec![event_record()]), (false, false), "event-only");
    }

    /// HMC and VI call `analytic_eta_nll_gradient` directly, without the inner loop's up-front
    /// routing check. For an endpoint-only subject carrying a TTE record, the function used to
    /// return the Ω prior alone (no Gaussian rows, so nothing for the provider to compute) — a
    /// gradient missing the whole hazard term. It must decline instead. The binary-only twin
    /// on the same model straddles the gate.
    #[test]
    fn direct_callers_cannot_get_a_gradient_missing_the_tte_term() {
        let model = binary_plus_tte();
        let params = &model.default_params;
        let mk = |recs: Vec<ObsRecord>| Subject {
            id: "1".into(),
            obs_records: recs,
            ..Default::default()
        };
        let direct = |s: &Subject| {
            super::super::analytic_eta_nll_gradient(
                &model,
                s,
                &params.theta,
                &[0.1],
                &params.omega,
                &params.sigma.values,
            )
        };
        assert!(
            direct(&mk(binary_records())).is_some(),
            "binary-only is served"
        );
        assert!(
            direct(&mk(vec![event_record()])).is_none(),
            "TTE must decline"
        );
    }

    /// The FD-fallback warning counts the route `find_ebe` takes, not the Gaussian
    /// recorder's verdict alone. On this endpoint-only model the recorder serves no
    /// subject (there is no structural model), so the old probe reported "2 of 2" — both
    /// the binary-only subject that runs analytic and the TTE subject that runs FD. It must
    /// count only the TTE subject, and stay silent for an all-binary population.
    #[test]
    fn fd_fallback_warning_counts_the_route_that_runs() {
        let model = binary_plus_tte();
        let theta = &model.default_params.theta;
        let mk = |recs: Vec<ObsRecord>| Subject {
            id: "1".into(),
            obs_records: recs,
            ..Default::default()
        };
        let pop = |subjects| Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let w = super::super::fd_fallback_warning(
            &model,
            &pop(vec![mk(binary_records()), mk(vec![event_record()])]),
            theta,
        )
        .expect("a population with one TTE subject warns");
        assert!(w.contains("1 of 2"), "got: {w}");
        assert!(super::super::fd_fallback_warning(
            &model,
            &pop(vec![mk(binary_records()), mk(binary_records())]),
            theta
        )
        .is_none());
    }

    /// A binary predictor past the dual dispatch cap has no program: it reports FD on the
    /// model-level predicate and takes FD on the per-subject gate, so the reported route
    /// cannot drift from the one that runs.
    #[test]
    fn program_less_binary_stays_fd() {
        let n_th = 24; // + 1 η = 25 axes > MAX_ENDPOINT_AXES (24)
        let mut src = String::from("[parameters]\n");
        for i in 1..=n_th {
            src += &format!("  theta T{i}(0.01, -6.0, 3.0)\n");
        }
        src += "  omega ETA_I ~ 0.1\n[binary_model]\n  cmt   = 3\n  logit = ETA_I + ";
        src += &(1..=n_th)
            .map(|i| format!("T{i}"))
            .collect::<Vec<_>>()
            .join(" + ");
        src += "\n";
        let model = parse(&src);
        assert_eq!(
            model.n_theta + model.n_eta,
            25,
            "fixture must exceed the cap"
        );
        assert!(!crate::categorical::all_binary_endpoints_analytic(&model));
        assert!(!super::super::analytic_inner_grad_supported_model(&model));
        assert!(!super::super::analytic_inner_grad_supported(
            &model,
            &subject()
        ));
        assert!(inner_grad(&model, &subject(), &[0.0]).is_none());
    }
}
