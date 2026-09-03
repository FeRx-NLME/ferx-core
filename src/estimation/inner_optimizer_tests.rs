use super::*;
use std::collections::HashMap;

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
            let g = super::super::analytic_eta_nll_gradient_with_schedule(
                &model,
                &subject,
                theta,
                &[eta0],
                &params.omega,
                &params.sigma.values,
                &params.residual_correlations,
                None,
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

/// #378 task B — the model-level report `build_info::gradient_method_inner` must match the
/// per-subject route `find_ebe` actually runs for an **in-scope ODE** model. The live inner
/// takes the light `Dual1` ODE η-gradient (`analytic_inner_grad_supported` → the `ode_spec`
/// branch → `ode_inner_grad_supported`), but the report used to read the closed-form-only
/// `analytic_inner_grad_supported_model` (false for every ODE model — no `tv_fn`) and
/// mislabel it "finite differences". Pin report == route, exactly as the CTMM tests above
/// pin theirs. (Mutation: reverting the ODE disjunct in `gradient_method_inner` makes the
/// report `FiniteDifferences` while the route stays analytic, so the final assert fails.)
#[test]
fn in_scope_ode_reports_and_takes_the_analytic_inner_route() {
    const ONECPT_IV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = crate::parser::model_parser::parse_model_string(ONECPT_IV_ODE).expect("parse");
    // Fixture self-check: genuinely in ODE analytic scope (else the asserts pass vacuously).
    assert!(
        crate::sens::provider::ode_inner_grad_supported_model(&model),
        "fixture must be an in-scope ODE model"
    );

    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![9.0, 8.0, 6.0, 3.0, 1.0],
        obs_cmts: vec![1, 1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0, 0, 0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    // Subject-level: what `find_ebe` actually runs.
    assert!(
        super::analytic_inner_grad_supported(&model, &subject),
        "an in-scope plain-bolus ODE subject must take the analytic inner route"
    );
    // Model-level: what `build_info::gradient_method_inner` reports — must match the route.
    assert_eq!(
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, &model),
        crate::build_info::GradientMethodKind::Analytic,
        "the report must match the live analytic ODE inner route"
    );
}

/// #926 (review follow-up to #378 task B) — an in-scope ODE model whose ENTIRE population runs
/// the FD inner must still emit the FD-fallback warning. `gradient_method_inner` reports
/// "analytic" at the model level for such a model (best-case), so without the warning the
/// persisted `fit.yaml` label would contradict the per-subject reality with nothing to
/// reconcile it — exactly the closed-form all-FD case (TV-cov + LTBS) the warning already
/// covers. This pins the shared `inner_reports_analytic_model` coupling between the report and
/// the warning. Every subject here carries a modeled-duration dose with no `D{cmt}` slot, which
/// the ODE provider declines to FD (the same trick `fd_fallback_warning_fires_only_for_mixed_
/// population` uses). Mutation: reverting `fd_fallback_warning`'s `model_reports_analytic` to the
/// old `analytic_inner_grad_supported_model` (false for ODE) makes this return `None` and fail.
#[test]
fn fd_fallback_warning_fires_for_all_fd_in_scope_ode_population() {
    const IN_SCOPE_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = crate::parser::model_parser::parse_model_string(IN_SCOPE_ODE).expect("parse");
    // In-scope at the model level, so the headline reports analytic …
    assert!(crate::sens::provider::ode_inner_grad_supported_model(
        &model
    ));
    assert_eq!(
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, &model),
        crate::build_info::GradientMethodKind::Analytic,
    );
    let theta = &model.default_params.theta;
    let zeros = vec![0.0; model.n_eta];
    // … but a modeled-duration dose with no `D{cmt}` slot puts every subject on the FD inner.
    let fd_subject = || {
        let mut d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        d.rate_mode = crate::types::RateMode::ModeledDuration;
        Subject {
            id: "1".into(),
            doses: vec![d],
            obs_times: vec![1.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 4.0, 1.0],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    };
    // Confirm the FD-ness the warning counts (not a vacuous population).
    assert!(
        crate::sens::provider::subject_eta_grad(&model, &fd_subject(), theta, &zeros).is_none(),
        "the modeled-duration-no-slot subject must run the FD inner"
    );
    let pop = Population {
        subjects: vec![fd_subject(), fd_subject()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let w = super::fd_fallback_warning(&model, &pop, theta)
        .expect("all-FD in-scope ODE population must warn (report says analytic)");
    assert!(w.contains("2 of 2"), "got: {w}");
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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0, 0, 1, 1],
        occasions: vec![1; 6],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0, 0, 1, 1],
        occasions: vec![1; 6],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
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
        let t_dense =
            time_it(&|x| dense_bfgs_core(&obj, &grad, x, n, 2000, 1e-8, None, None, false));
        let t_lbfgs = time_it(&|x| lbfgs_core(&obj, &grad, x, n, 2000, 1e-8, None, None, false));
        eprintln!(
            "  n={n:4}  dense={t_dense:8.3} ms  lbfgs={t_lbfgs:8.3} ms  dense/lbfgs={:.2}x",
            t_dense / t_lbfgs
        );
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
    let (alpha, f_new) = backtracking_line_search(&counting, &x, &d, &g, 1, f0);
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
    let (alpha, f_new) = backtracking_line_search(&obj, &x, &d, &g, 1, f0);
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
    let (alpha, f_new) = backtracking_line_search(&nan_obj, &x, &d, &g, 1, f0);
    assert_eq!(alpha, 0.0, "a never-finite objective yields no step");
    assert_eq!(f_new, f0, "baseline objective is returned unchanged");
    // +inf trials behave identically (never accepted, never a panic).
    let inf_obj = |_: &[f64]| -> f64 { f64::INFINITY };
    let (alpha, f_new) = backtracking_line_search(&inf_obj, &x, &d, &g, 1, f0);
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
    let (alpha, f_new) = backtracking_line_search(&obj, &x, &d, &g, 1, f0);
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
    let ok = dense_bfgs_core(&obj, &grad, &mut x, 2, 200, 1e-10, None, None, false);
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
        converged: true,
        used_fallback: false,
        grad_norm: 0.0,
        nll: 1.5,
        kappas: Vec::new(),
        hard_reject: false,
    };
    assert!(r.converged);
    assert!(!r.used_fallback);
    assert_eq!(r.grad_norm, 0.0);
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
            converged: false, // unconverged
            used_fallback: false,
            grad_norm: 0.0,
            nll: 1.0,
            kappas: Vec::new(),
            hard_reject: false,
        },
        EbeResult {
            eta: nalgebra::DVector::zeros(1),
            h_matrix: nalgebra::DMatrix::identity(1, 1),
            converged: false, // also unconverged
            used_fallback: true,
            grad_norm: 0.0,
            nll: 2.0,
            kappas: Vec::new(),
            hard_reject: false,
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
        converged: false,
        used_fallback: false,
        grad_norm: 0.0,
        nll: 1.0,
        kappas: Vec::new(),
        hard_reject,
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

#[test]
fn test_frem_jacobian_overrides_fd_with_exact_values() {
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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: vec![0, 0, 100], // last obs is FREM
        obs_records: vec![],
    };

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

/// No-regression: a well-identified, unimodal subject (no resets / TV-covariates)
/// must return a bit-identical EBE with the guarded multi-start on
/// (`inner_restarts = 3`) and off (`inner_restarts = 0`). The #891 probe may scan
/// weakly-informed coordinates, but every alternate seed reconverges to the same
/// basin and is rejected by the `+1e-9` improvement guard, so the returned η̂ and
/// its objective are unchanged — the added cost buys no spurious mode change.
#[test]
fn inner_restarts_bit_identical_on_wellidentified_subject() {
    use crate::types::{DoseEvent, Subject};
    use std::collections::HashMap;
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n",
    )
    .expect("parse one_cpt_oral model");

    // Six informative observations across the profile ⇒ all etas well identified.
    let subject = Subject {
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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        !subject.has_resets() && !subject.has_tv_covariates(),
        "test premise: subject must exercise the #891 (non reset/TV) probe path"
    );

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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
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
