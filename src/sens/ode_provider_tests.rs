use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::pk::compute_predictions_with_tv;

/// Hardening for issue #410: an observation time and the segment break / solver
/// save time it coincides with are produced by different arithmetic, so they can
/// be value-equal yet bit-different. `0.1 + 0.2 ≠ 0.3` (IEEE-754) is the canonical
/// case the old `f64::to_bits` keying would silently miss — leaving the
/// observation's state (and its sensitivity) at zero. `obs_time_matches` must
/// match these while still separating genuinely distinct observation times.
#[test]
fn obs_time_matches_tolerates_ulp_differences_not_distinct_times() {
    let a = 0.3_f64;
    let b = 0.1_f64 + 0.2; // 0.30000000000000004
    assert_ne!(a.to_bits(), b.to_bits(), "precondition: bit-different");
    assert!(
        obs_time_matches(a, b),
        "ULP-apart times must match — bit-exact keying would drop this observation"
    );
    // Tolerance scales with magnitude (a late, large dosing time).
    let big = 168.0_f64;
    assert!(obs_time_matches(big, big + big * 1e-12));
    // Genuinely distinct observation times must NOT be conflated.
    assert!(!obs_time_matches(24.0, 24.001));
    assert!(!obs_time_matches(0.0, 0.5));
    assert!(!obs_time_matches(big, big + 0.01));
}
use crate::types::DoseEvent;
use std::collections::HashMap;

// 2-cpt IV bolus as a user ODE, with a Form C concentration readout
// (`y = central / V1`). CL/V1 carry IIV; Q/V2 are fixed individual params.
const TWOCPT_ODE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

fn bolus_subject(times: &[f64]) -> Subject {
    let n = times.len();
    Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
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
    }
}

// 2-cpt ODE with an allometric weight covariate on CL and V1 — exercises the
// covariate path: typical values (and their θ-Jacobian) must fold WT.
const TWOCPT_ODE_COV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
  V1 = TVV1 * (WT / 70) * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

fn bolus_subject_wt(times: &[f64], wt: f64) -> Subject {
    let mut s = bolus_subject(times);
    s.covariates.insert("WT".to_string(), wt);
    s
}

/// The ODE provider's `f`, `∂f/∂η`, `∂f/∂θ` must match the production
/// predictor (`compute_predictions_with_tv`) and its finite differences.
#[test]
fn ode_provider_2cpt_matches_production() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "2-cpt ODE with Form C readout should be supported"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];

    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(&model, &subject, th, e)[j]
    };
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let he = 1e-6;

    for (j, obs) in sens.obs.iter().enumerate() {
        // Value matches the production prediction.
        approx::assert_relative_eq!(
            obs.f,
            pred(&eta, &theta, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        // ∂f/∂η vs central FD.
        for k in 0..n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
        }
        // ∂f/∂θ vs central FD.
        for m in 0..n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 1e-3, epsilon = 1e-6);
        }
    }
}

// 2-cpt ODE whose Form C readout references a **covariate** (`FREE`) — a
// free/total protein-binding readout (#540, the fluconazole_radboudumc shape).
// The saturable bound term is gated off for free assays (FREE==1) and on for
// total assays (FREE==0). `BMAX`/`KD` are individual parameters; the covariate
// threads into the dual readout as a constant from the per-observation snapshot,
// so the analytic gradient must still match production + FD.
const TWOCPT_ODE_READOUT_COV: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBMAX(3.0, 0.0, 100.0)
  theta TVKD(5.0,   0.01, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V1   = TVV1 * exp(ETA_V1)
  Q    = TVQ
  V2   = TVV2
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 + (1.0 - FREE) * BMAX * (central / V1) / (KD + central / V1)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// 2-cpt ODE whose Form C readout references a θ and an η **directly** (#486):
// a multiplicative `ETA_CL` on the concentration plus an additive baseline
// `TVBASE`. The parser desugars each bare θ/η into a synthetic individual
// parameter (`__ferx_ro_*`), so the readout's ∂y/∂θ, ∂y/∂η ride the
// individual-parameter sensitivity chain. `ETA_CL` is multiplicative AND also
// drives `CL`, so the analytic gradient must capture both the explicit term and
// the cross-coupling through the state — the 2nd-order blocks are non-trivial.
const TWOCPT_ODE_READOUT_DIRECT_THETA_ETA: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBASE(0.5, 0.0, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 * (1.0 + ETA_CL) + TVBASE
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// #486: a Form C readout referencing a θ/η directly is now analytic — the parser
/// desugars each into a synthetic individual parameter, so the provider gradient
/// (value, ∂f/∂η, ∂f/∂θ) and the 2nd-order blocks must match production + FD.
#[test]
fn ode_provider_form_c_direct_theta_eta_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_READOUT_DIRECT_THETA_ETA).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "Form C readout referencing θ/η directly should be analytic (#486)"
    );
    // The two synthetic readout parameters extend the individual-parameter set.
    assert_eq!(
        model.pk_indices.len(),
        6,
        "4 real + 2 synthetic (__ferx_ro_th4, __ferx_ro_eta0) individual parameters"
    );
    let theta = vec![4.0, 12.0, 2.0, 25.0, 0.5];
    let eta = vec![0.12, -0.08];
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

    let subj = bolus_subject(&times);
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

// Direct θ/η readout (#486) combined with a time-varying covariate (`FREE`), so
// the subject routes through the (θ,η)-basis TV-cov walk (`run_subject_tvcov`)
// rather than the individual-parameter-basis static walk. The synthetic readout
// parameters must serve the analytic gradient on that path too.
const TWOCPT_ODE_READOUT_DIRECT_TVCOV: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBASE(0.5, 0.0, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 * (1.0 + ETA_CL) + (1.0 - FREE) * TVBASE
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// #486 + #540: a direct θ/η readout whose subject also carries a time-varying
/// covariate routes through the TV-cov `(θ,η)`-basis walk. The desugared synthetic
/// parameters must match production + FD there as well.
#[test]
fn ode_provider_form_c_direct_theta_eta_tvcov_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_READOUT_DIRECT_TVCOV).expect("parse");
    let theta = vec![4.0, 12.0, 2.0, 25.0, 0.5];
    let eta = vec![0.12, -0.08];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

    let mut subj = bolus_subject(&times);
    subj.obs_covariates = (0..times.len())
        .map(|i| HashMap::from([("FREE".to_string(), (i % 2) as f64)]))
        .collect();
    assert!(
        subj.has_tv_covariates(),
        "alternating FREE must register as time-varying"
    );
    assert!(
        ode_tvcov_supported(&model, &subj),
        "TV-cov direct-θ/η Form C readout should be analytic (#486)"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

/// #540: a Form C readout referencing a covariate is now analytic. With the
/// covariate held constant per subject (`obs_covariates` empty → the static
/// walk), the provider gradient must match production + FD for both the total
/// assay (bound term active) and the free assay (bound term zeroed).
#[test]
fn ode_provider_form_c_static_covariate_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_READOUT_COV).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "Form C readout referencing a covariate should be analytic (#540)"
    );
    let theta = vec![4.0, 12.0, 2.0, 25.0, 3.0, 5.0];
    let eta = vec![0.12, -0.08];
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

    let mut total = bolus_subject(&times);
    total.covariates.insert("FREE".to_string(), 0.0);
    check_vs_production(&model, &total, &theta, &eta);

    let mut free = bolus_subject(&times);
    free.covariates.insert("FREE".to_string(), 1.0);
    check_vs_production(&model, &free, &theta, &eta);
}

/// #540: the readout covariate read per observation. `FREE` alternates row to
/// row (paired free/total assays on one subject), so the bound term switches on
/// and off per observation — the TV-cov walk's `obs_cov(j)` snapshot path. The
/// analytic gradient must still match the production predictor + FD.
#[test]
fn ode_provider_form_c_per_obs_covariate_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_READOUT_COV).expect("parse");
    let theta = vec![4.0, 12.0, 2.0, 25.0, 3.0, 5.0];
    let eta = vec![0.12, -0.08];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

    let mut subj = bolus_subject(&times);
    subj.obs_covariates = (0..times.len())
        .map(|i| HashMap::from([("FREE".to_string(), (i % 2) as f64)]))
        .collect();
    assert!(
        subj.has_tv_covariates(),
        "alternating FREE must register as time-varying"
    );
    assert!(
        ode_tvcov_supported(&model, &subj),
        "TV-cov Form C readout referencing a covariate should be analytic (#540)"
    );
    check_vs_production(&model, &subj, &theta, &eta);
}

// 2-cpt ODE whose Form C readout references the `TIME` built-in directly
// (`y = central/V1 + BETA*TIME`, the response-versus-time shape from #1028). `TIME`
// compiles to `Op::PushTime`, which resolves from the model-time thread-local — the
// integrator's own guard is dropped before the readout runs, so without an explicit
// per-observation guard at the readout seam every `TIME` read the `0.0` default and
// the whole time-dependent term vanished silently. `BETA = TVBETA` is a
// non-structural readout parameter, so `∂y/∂TVBETA = TIME` — a direct probe that the
// observation's own time (not a stale 0) reaches the readout. Mirrors the analytic
// Form C twin (`iov_form_c_time_readout_evaluated_at_obs_time`, provider_tests.rs).
const TWOCPT_ODE_READOUT_TIME: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBETA(0.05, -10.0, 10.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V1   = TVV1 * exp(ETA_V1)
  Q    = TVQ
  V2   = TVV2
  BETA = TVBETA
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 + BETA * TIME
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// #1046: an ODE `init(...)` seed that reads a **dose attribute**. Before #1046 the
// parser rejected these, so no model could reach the provider with `∂init/∂F ≠ 0`.
// `F` carries IIV here deliberately (`ETA_F`), so the seed depends on both θ and η and
// the two `∂/∂F` contributions — the F-scaled dose bolus and the F-scaled initial
// amount — have to compose in the same gradient.
//
// The tight `ode_reltol`/`ode_abstol` are load-bearing, not decoration: the FD side of
// the parity harness differences the *production* ODE predictor, so at the default
// tolerances the reference is under-resolved and a correct model can miss the
// harness's `2e-4` bound on integrator noise alone. `TVCOV_INIT_ODE` below tightens
// them for the same reason.
const ONECPT_ODE_INIT_READS_F: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVF(0.5, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF * exp(ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = F * 100.0
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

// The lag twin. `LAGTIME` shifts the dose event *and* scales the seed, so its two
// contributions compose through different mechanisms than `F`'s: an event-time
// saltation on the dose side, a plain multiplier on the seed side.
const ONECPT_ODE_INIT_READS_LAGTIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.7, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = LAGTIME * 100.0
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

// Control: the identical model seeding from an ordinary parameter. Pins that the
// parity above is a property of the seed arithmetic, not of the name.
const ONECPT_ODE_INIT_READS_ORDINARY: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVF(0.5, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  FSEED = TVF * exp(ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = FSEED * 100.0
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// #1046 made `init(state) = F * …` / `= LAGTIME * …` writable for the first time, so
/// this is a **newly reachable gradient class**: the provider now has to carry
/// `∂init/∂F` (and `∂init/∂lag`) alongside the `∂/∂F` it already carried on the dose
/// bolus and the `∂/∂lag` on the event-time saltation. Those contributions had never
/// had to compose, because the parser rejected every model that would make them.
///
/// Per CLAUDE.md, a newly reachable analytic-sensitivity path needs a `Dual2`-vs-FD
/// parity test — the code was not edited, but the set of models that can reach it was
/// widened, and a wrong composition here would compile, run, and silently return a bad
/// gradient. Both fixtures put IIV on the attribute so the seed varies with η as well
/// as θ.
#[test]
fn ode_provider_init_reading_a_dose_attribute_matches_fd() {
    for (label, src, theta, eta) in [
        (
            "init reads F",
            ONECPT_ODE_INIT_READS_F,
            vec![0.2, 10.0, 0.5],
            vec![0.12, -0.08],
        ),
        (
            "init reads LAGTIME",
            ONECPT_ODE_INIT_READS_LAGTIME,
            vec![0.2, 10.0, 0.7],
            vec![0.12, -0.08],
        ),
        (
            "init reads an ordinary parameter (control)",
            ONECPT_ODE_INIT_READS_ORDINARY,
            vec![0.2, 10.0, 0.5],
            vec![0.12, -0.08],
        ),
    ] {
        let model = parse_model_string(src).unwrap_or_else(|e| panic!("{label}: parse: {e}"));
        // The routing half of the CLAUDE.md rule: if one of these ever falls out of
        // analytic scope it must do so loudly here rather than quietly returning a
        // FOCE-shaped gradient from a path that no longer models the seed.
        assert!(
            ode_analytical_supported(&model),
            "{label}: must be served analytically, not silently dropped to FD"
        );
        // Observations start before the lag (0.25 < 0.7) so the lag fixture is sampled
        // on both sides of its dose arrival — the seed alone, then seed + lagged dose.
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);

        // Non-vacuity: the attribute's θ (index 2) and its η (index 1) must actually
        // move the prediction, or the parity checks above would agree on a derivative
        // that is identically zero and would pin nothing about the newly reachable
        // composition. `∂f/∂TVF` is non-zero only because the seed reads it — a bolus
        // at t=0 into the observed compartment contributes through `F` too, so this
        // asserts the pair is live rather than isolating the seed term.
        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let max_dtheta = sens
            .obs
            .iter()
            .map(|o| o.df_dtheta[2].abs())
            .fold(0.0_f64, f64::max);
        let max_deta = sens
            .obs
            .iter()
            .map(|o| o.df_deta[1].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dtheta > 1e-6,
            "{label}: max |∂f/∂θ_attr| = {max_dtheta:.3e} — the attribute does not \
             reach the prediction, so the parity check above is vacuous"
        );
        assert!(
            max_deta > 1e-6,
            "{label}: max |∂f/∂η_attr| = {max_deta:.3e} — the attribute's IIV does not \
             reach the prediction, so the parity check above is vacuous"
        );
    }
}

/// #1028: a `TIME`-referencing ODE Form C readout must evaluate `TIME` at each
/// observation. `∂y/∂TVBETA` equals that observation's time — it would be 0 at every
/// row if the readout read a stale model-time of 0 — and `check_vs_production` ties
/// the analytic dual walk to FD of the production predictor, so both sides are pinned
/// to the same (guarded) expression.
#[test]
fn ode_provider_form_c_time_readout_evaluated_at_obs_time() {
    let model = parse_model_string(TWOCPT_ODE_READOUT_TIME).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "TIME-referencing Form C readout should stay analytic (#1028)"
    );
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    let subject = bolus_subject(&times);
    let theta = vec![4.0, 12.0, 2.0, 25.0, 0.05];
    let eta = vec![0.12, -0.08];

    check_vs_production(&model, &subject, &theta, &eta);

    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    for (j, obs) in sens.obs.iter().enumerate() {
        // ∂(central/V1 + TVBETA·TIME)/∂TVBETA = TIME at observation j.
        approx::assert_relative_eq!(
            obs.df_dtheta[4],
            times[j],
            max_relative = 1e-6,
            epsilon = 1e-9
        );
    }
}

/// #1028 production half: the f64 ODE predictor must read `TIME` at each observation
/// too — the sensitivity parity above would still pass if *both* sides read a stale
/// 0. Differencing the same model at `TVBETA = 0.05` and `TVBETA = 0` isolates the
/// readout's time term from the disposition, which is identical in the two runs, so
/// the gap must be exactly `BETA · t`.
#[test]
fn ode_form_c_time_readout_prediction_uses_obs_time() {
    let model = parse_model_string(TWOCPT_ODE_READOUT_TIME).expect("parse");
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    let subject = bolus_subject(&times);
    let eta = vec![0.0, 0.0];
    let beta = 0.05;
    let with_beta =
        compute_predictions_with_tv(&model, &subject, &[4.0, 12.0, 2.0, 25.0, beta], &eta);
    let no_beta = compute_predictions_with_tv(&model, &subject, &[4.0, 12.0, 2.0, 25.0, 0.0], &eta);
    for (j, t) in times.iter().enumerate() {
        approx::assert_relative_eq!(
            with_beta[j] - no_beta[j],
            beta * t,
            max_relative = 1e-8,
            epsilon = 1e-12
        );
    }
}

/// Shared check: provider `f`/`∂f/∂η`/`∂f/∂θ` vs production predictor + FD.
fn check_vs_production(model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64]) {
    let sens = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(model, subject, th, e)[j]
    };
    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            obs.f,
            pred(eta, theta, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        for k in 0..model.n_eta {
            let mut ep = eta.to_vec();
            ep[k] += he;
            let mut em = eta.to_vec();
            em[k] -= he;
            let g = (pred(&ep, theta, j) - pred(&em, theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
        }
        for m in 0..model.n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.to_vec();
            tp[m] += s;
            let mut tm = theta.to_vec();
            tm[m] -= s;
            let g = (pred(eta, &tp, j) - pred(eta, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-3, epsilon = 1e-6);
        }
    }
}

/// Inner/outer parity guardrail: the light `Dual1` inner provider
/// (`ode_subject_eta_grad`) must reproduce the full `Dual2` outer provider's value
/// and `∂f/∂η` (`ode_subject_sensitivities`) exactly — both are exact analytic, only
/// the dual order differs (and `solve_ode_g` uses value-based step control, so the
/// trajectories match). This is what keeps the inner EBE loop's η-gradient consistent
/// with the outer gradient across every ODE variant (Form-C, per-CMT, TV-cov, EVID=2
/// pk-only, `ExpressionScale`); factored out so a tolerance change touches one site,
/// not every parity test.
fn check_inner_outer_eta_parity(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let full = ode_subject_sensitivities(model, subject, theta, eta).expect("full outer sens");
    let light = ode_subject_eta_grad(model, subject, theta, eta).expect("light inner grad");
    assert_eq!(full.obs.len(), light.len());
    for (a, b) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-12);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                a.df_deta[k],
                b.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-10
            );
        }
    }
}

/// Validate the analytic 2nd-order blocks (`d2f_deta2`, `d2f_deta_dtheta`) against
/// central finite differences of the analytic *first*-order gradient `df_deta` from the
/// same provider — the Hessian must equal the derivative of the gradient.
/// `check_vs_production` only FD-checks the value / first order against the values-only
/// production predictor, so the δlag² saltation coefficients (the rate-boundary `coef2`
/// and the bolus `jg_cross` cross term) get no independent 2nd-order check otherwise
/// (#472 review round 2 #3/#4).
/// Validate the analytic 2nd-order blocks (`d2f_deta2`, `d2f_deta_dtheta`) against
/// **double central finite differences of the production predictor** — the ground-truth
/// Hessian, not the self-FD of the analytic gradient. This is the check the #486
/// init-composition scope needs: a small Hessian element sitting between large neighbours
/// (e.g. `∂²f/∂η_V²` crossing zero mid-window for an `init = BASE/V` decay) makes the
/// he=1e-6 self-FD of `check_hessian_vs_fd_of_grad` catastrophically cancel, even though
/// the analytic Hessian is exact — double-FD of the predictor with an absolute floor is
/// robust there. The `epsilon` floor (`5e-3`) only relaxes elements whose magnitude is
/// below it (where a 4-point stencil is dominated by its own truncation/solver noise); a
/// genuinely wrong large element is still caught by `max_relative`.
fn check_hessian_vs_production_fd(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let base = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(model, subject, th, e)[j]
    };
    let h = 1e-4;
    // ∂²f/∂η_k∂η_p via a 4-point cross stencil.
    for p in 0..n_eta {
        for (j, o) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let mut epp = eta.to_vec();
                epp[k] += h;
                epp[p] += h;
                let mut epm = eta.to_vec();
                epm[k] += h;
                epm[p] -= h;
                let mut emp = eta.to_vec();
                emp[k] -= h;
                emp[p] += h;
                let mut emm = eta.to_vec();
                emm[k] -= h;
                emm[p] -= h;
                let fd = (pred(&epp, theta, j) - pred(&epm, theta, j) - pred(&emp, theta, j)
                    + pred(&emm, theta, j))
                    / (4.0 * h * h);
                approx::assert_relative_eq!(
                    o.d2f_deta2[k * n_eta + p],
                    fd,
                    max_relative = 5e-3,
                    epsilon = 5e-3
                );
            }
        }
    }
    // ∂²f/∂η_k∂θ_m via a 4-point cross stencil.
    for m in 0..n_theta {
        let sm = h * (1.0 + theta[m].abs());
        for (j, o) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let mut epp = eta.to_vec();
                epp[k] += h;
                let mut tp = theta.to_vec();
                tp[m] += sm;
                let mut epm = eta.to_vec();
                epm[k] += h;
                let mut tm = theta.to_vec();
                tm[m] -= sm;
                let fd =
                    (pred(&epp, &tp, j) - pred(&epp, &tm, j) - pred(&emm_eta(eta, k, h), &tp, j)
                        + pred(&emm_eta(eta, k, h), &tm, j))
                        / (4.0 * h * sm);
                approx::assert_relative_eq!(
                    o.d2f_deta_dtheta[k * n_theta + m],
                    fd,
                    max_relative = 5e-3,
                    epsilon = 5e-3
                );
            }
        }
    }
}

/// `eta` with `eta[k] -= h` (helper for the mixed η×θ stencil above).
fn emm_eta(eta: &[f64], k: usize, h: f64) -> Vec<f64> {
    let mut e = eta.to_vec();
    e[k] -= h;
    e
}

fn check_hessian_vs_fd_of_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let base = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
    let he = 1e-6;
    // ∂(∂f/∂η_k)/∂η_p == d2f_deta2[k, p].
    for p in 0..n_eta {
        let mut ep = eta.to_vec();
        ep[p] += he;
        let mut em = eta.to_vec();
        em[p] -= he;
        let sp = ode_subject_sensitivities(model, subject, theta, &ep).expect("supported");
        let sm = ode_subject_sensitivities(model, subject, theta, &em).expect("supported");
        for (j, o) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                approx::assert_relative_eq!(
                    o.d2f_deta2[k * n_eta + p],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
    // ∂(∂f/∂η_k)/∂θ_m == d2f_deta_dtheta[k, m].
    for m in 0..n_theta {
        let s = he * (1.0 + theta[m].abs());
        let mut tp = theta.to_vec();
        tp[m] += s;
        let mut tm = theta.to_vec();
        tm[m] -= s;
        let sp = ode_subject_sensitivities(model, subject, &tp, eta).expect("supported");
        let sm = ode_subject_sensitivities(model, subject, &tm, eta).expect("supported");
        for (j, o) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                approx::assert_relative_eq!(
                    o.d2f_deta_dtheta[k * n_theta + m],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}

// ---- #835: steady-state dosing into a built-in absorption compartment ----
// Gap 1 (#719/#834) made these SS predictions closed-form (`u_ss = (I − M)⁻¹·b`) but left
// their sensitivities on FD. The dual walk now carries that fixed point over `Dual1`/`Dual2`
// (`equilibrate_ss_input_rate_state_g`), so the analytic gradient/Hessian must match the
// production predictor + FD, exactly as every other analytic ODE variant does.

const ONECPT_SS_FIRST_ORDER: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-10
"#;

const ONECPT_SS_TRANSIT: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVN(5.0, 1.0, 30.0)
  theta TVMTT(2.0, 0.1, 20.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

const ONECPT_SS_IGD: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVMAT(2.0, 0.1, 20.0)
  theta TVCV2(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MAT = TVMAT
  CV2 = TVCV2
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// first_order absorption into a 2-cpt disposition: exercises the multi-state fixed point
// (the `I − M` solve is genuinely 2×2, not scalar), with η on both CL and V1.
const TWOCPT_SS_FIRST_ORDER: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV1(20.0, 0.5, 200.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(30.0, 1.0, 500.0)
  theta TVKA(0.2, 0.005, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = first_order(ka=KA) - (CL/V1)*central - (Q/V1)*central + (Q/V2)*peripheral
  d/dt(peripheral) =  (Q/V1)*central - (Q/V2)*peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// weibull() is the fourth admitted kernel (gate + CHANGELOG) and rides the same generic SS
// fixed point / periodic forcing; its log-domain (`ln`/`exp`) Dual2 forcing under an SS pulse
// is otherwise unexercised (#835 review). β = 1.5 (> 1) so the density is smooth at the dose
// and analytic ≡ central-FD is clean.
const ONECPT_SS_WEIBULL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVTD(2.0, 0.05, 24.0)
  theta TVBETA(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// Estimated bioavailability F on an SS-absorption dose: every other SS fixture defaults F = 1,
// so the `f_bio: T` jet threaded into `equilibrate_ss_input_rate_state_g` is never
// differentiated. Here F rides both a θ (THETA_F) and an η (ETA_F) via `inv_logit`, so
// `∂u_ss/∂F` (through the SS trough) must match production FD (#835 review).
const ONECPT_SS_FIRST_ORDER_F: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta THETA_F(0.7, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = inv_logit(logit(THETA_F) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// first_order absorption into a **Michaelis–Menten** (nonlinear) disposition: the closed-form
// fixed point self-declines, so the dual walk falls back to the pulse-train iteration — its
// gradient must still match production's (which uses the same fallback). η on Vmax.
const MM_SS_FIRST_ORDER: &str = r#"
[parameters]
  theta TVVM(5.0, 0.1, 100.0)
  theta TVKM(8.0, 0.1, 200.0)
  theta TVKA(0.5, 0.005, 20.0)
  omega ETA_VM ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  VM = TVVM * exp(ETA_VM)
  KM = TVKM
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - VM*central/(KM+central)
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// ── `CMT=0`, the default dose compartment, on the gradient path (#899) ──────────────
//
// `CMT=0` is NONMEM's default dose compartment and resolves to compartment 1 on both
// engines. The gradient twins have to move with the value path or FOCEI differentiates a
// different dosing history than it predicts — the exact failure #375 removed on the
// analytical side. Before #899 the dual dose walk skipped `CMT=0` outright (`if d.cmt >= 1`,
// on the false premise that the datareader rejected it upstream) and `equilibrate_ss_state_g`
// bailed out of the SS equilibration, so the analytic gradient saw an undosed / unaccumulated
// subject while production dosed it.
//
// Each test carries two teeth: `check_vs_production` is the standard `Dual2`-vs-FD parity
// oracle on the `CMT=0` subject, and bit-equality with the `CMT=1` twin pins the *convention*
// rather than merely the self-consistency of a wrong one.

/// Every observation's value and both derivative blocks must be bit-identical between a
/// dose written `CMT=0` and the same dose written `CMT=1`.
fn assert_sens_identical(
    model: &CompiledModel,
    a: &Subject,
    b: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let sa = ode_subject_sensitivities(model, a, theta, eta).expect("supported");
    let sb = ode_subject_sensitivities(model, b, theta, eta).expect("supported");
    assert_eq!(sa.obs.len(), sb.obs.len(), "observation count differs");
    assert!(!sa.obs.is_empty(), "no observations to compare");
    for (j, (x, y)) in sa.obs.iter().zip(sb.obs.iter()).enumerate() {
        assert_eq!(x.f, y.f, "obs {j}: value differs between CMT=0 and CMT=1");
        assert_eq!(x.df_deta, y.df_deta, "obs {j}: ∂f/∂η differs");
        assert_eq!(x.df_dtheta, y.df_dtheta, "obs {j}: ∂f/∂θ differs");
    }
    // Guard against the comparison passing vacuously on an all-zero curve.
    assert!(
        sa.obs.iter().any(|o| o.f.abs() > 0.0),
        "reference curve should be non-trivial"
    );
}

#[test]
fn ode_provider_cmt_zero_bolus_matches_the_default_compartment_and_fd() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];

    let subj_one = bolus_subject(&times);
    let mut subj_zero = bolus_subject(&times);
    subj_zero.doses = vec![DoseEvent::new(0.0, 100.0, 0, 0.0, false, 0.0)];

    check_vs_production(&model, &subj_zero, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj_zero, &theta, &eta);
    assert_sens_identical(&model, &subj_zero, &subj_one, &theta, &eta);
}

/// The steady-state twin: `equilibrate_ss_state_g` must equilibrate the default compartment
/// like any other. Its `dose.cmt == 0` bail-out returned an unequilibrated all-zero trough,
/// so an `SS=1` dose written `CMT=0` was differentiated as a *single* dose.
#[test]
fn ode_provider_cmt_zero_steady_state_matches_the_default_compartment_and_fd() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    let times = [1.0, 4.0, 8.0, 12.0, 23.0];
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];

    let mut subj_one = bolus_subject(&times);
    subj_one.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)];
    let mut subj_zero = bolus_subject(&times);
    subj_zero.doses = vec![DoseEvent::new(0.0, 100.0, 0, 0.0, true, 24.0)];

    check_vs_production(&model, &subj_zero, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj_zero, &theta, &eta);
    assert_sens_identical(&model, &subj_zero, &subj_one, &theta, &eta);
}

// SS bolus into compartment 1 with interval `ii`; slow KA (t½,abs ≈ II) so the absorption
// tail spills across the interval — the carryover the SS trough must capture.
fn ss_absorption_subject(times: &[f64], ii: f64) -> Subject {
    let mut s = bolus_subject(times);
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii)];
    s
}

#[test]
fn ode_provider_ss_first_order_1cpt_matches_production() {
    let model = parse_model_string(ONECPT_SS_FIRST_ORDER).expect("parse SS first_order");
    let theta = vec![1.0, 20.0, 0.15];
    let eta = vec![0.15];
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: SS into a first_order absorption compartment must now be analytic"
    );
    // Value + ∂/∂η + ∂/∂θ vs the production predictor, analytic Hessian two ways, and
    // inner(Dual1)/outer(Dual2) parity.
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subj, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
    // The linear disposition takes the closed-form fixed point (one recorded "cycle"), not
    // the up-to-50-cycle iteration — the fast path #835 restores for sensitivities.
    let _ = ode_subject_sensitivities(&model, &subj, &theta, &eta).expect("supported");
    assert_eq!(
        crate::dosing::last_ss_equilibration_cycles(),
        1,
        "a linear disposition must equilibrate via the closed-form fixed point"
    );
}

/// Regression for the #913 review: an `SS=1` bolus written `CMT=0` into a **built-in absorption**
/// compartment. The plain-disposition SS twin (`equilibrate_ss_state_g`) was already fixed for
/// `CMT=0`, but the *absorption* SS twin (`equilibrate_ss_input_rate_state_g`) still bailed on
/// `dose.cmt == 0` and returned the zero (unequilibrated) trough, while the f64 value path
/// equilibrated it — a silent value≠gradient FOCEI error that the existing SS tests all missed by
/// dosing `CMT=1`. `check_vs_production` is the Dual2-vs-FD teeth on the `CMT=0` subject;
/// `assert_sens_identical` pins it bit-for-bit to the `CMT=1` twin (the convention, not just
/// self-consistency).
#[test]
fn ode_provider_ss_first_order_cmt_zero_matches_the_default_compartment_and_fd() {
    let model = parse_model_string(ONECPT_SS_FIRST_ORDER).expect("parse SS first_order");
    let theta = vec![1.0, 20.0, 0.15];
    let eta = vec![0.15];
    let times = [0.5, 1.0, 2.0, 4.0, 6.0, 7.9];

    let subj_one = ss_absorption_subject(&times, 8.0); // CMT=1
    let mut subj_zero = ss_absorption_subject(&times, 8.0);
    subj_zero.doses = vec![DoseEvent::new(0.0, 100.0, 0, 0.0, true, 8.0)]; // CMT=0

    assert!(
        ode_tvcov_supported(&model, &subj_zero),
        "SS into a first_order absorption compartment is analytic (#835), CMT=0 included"
    );
    check_vs_production(&model, &subj_zero, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj_zero, &theta, &eta);
    assert_sens_identical(&model, &subj_zero, &subj_one, &theta, &eta);
}

#[test]
fn ode_provider_ss_transit_1cpt_matches_production() {
    let model = parse_model_string(ONECPT_SS_TRANSIT).expect("parse SS transit");
    let theta = vec![1.0, 20.0, 5.0, 2.0];
    let eta = vec![0.12];
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: SS + transit analytic"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

#[test]
fn ode_provider_ss_igd_1cpt_matches_production() {
    let model = parse_model_string(ONECPT_SS_IGD).expect("parse SS igd");
    let theta = vec![1.0, 20.0, 2.0, 0.5];
    let eta = vec![0.12];
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: SS + igd analytic"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

#[test]
fn ode_provider_ss_first_order_2cpt_matches_production() {
    let model = parse_model_string(TWOCPT_SS_FIRST_ORDER).expect("parse SS 2-cpt first_order");
    let theta = vec![1.0, 20.0, 2.0, 30.0, 0.2];
    let eta = vec![0.15, -0.1];
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: SS + first_order into a 2-cpt disposition analytic"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

#[test]
fn ode_provider_ss_weibull_1cpt_matches_production() {
    let model = parse_model_string(ONECPT_SS_WEIBULL).expect("parse SS weibull");
    let theta = vec![1.0, 20.0, 2.0, 1.5];
    let eta = vec![0.12];
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: SS + weibull analytic"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

#[test]
fn ode_provider_ss_first_order_bioavailability_matches_production() {
    let model = parse_model_string(ONECPT_SS_FIRST_ORDER_F).expect("parse SS first_order under F");
    assert!(model.has_bioavailability());
    let theta = vec![1.0, 20.0, 0.15, 0.7];
    let eta = vec![0.12, 0.1]; // [ETA_CL, ETA_F] — F on IIV exercises ∂u_ss/∂F through the trough
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: SS + first_order under an estimated F stays analytic"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

#[test]
fn ode_provider_ss_absorption_nonlinear_disposition_matches_production() {
    // Nonlinear (MM) disposition: the fixed point self-declines and the dual walk falls back
    // to the pulse-train iteration — value + gradient must still match production (same
    // fallback). This is the analytic-parity teeth on the `equilibrate_ss_input_rate_state_g`
    // fallback branch.
    let model = parse_model_string(MM_SS_FIRST_ORDER).expect("parse MM SS first_order");
    let theta = vec![5.0, 8.0, 0.5];
    let eta = vec![0.1];
    let subj = ss_absorption_subject(&[0.5, 1.0, 2.0, 4.0, 6.0, 7.9], 8.0);
    assert!(
        ode_tvcov_supported(&model, &subj),
        "#835: the gate admits SS + smooth kernel regardless of disposition linearity"
    );
    check_vs_production(&model, &subj, &theta, &eta);
    // Prove the fallback fired: a nonlinear disposition runs the iteration (> 1 cycle), not the
    // single-cycle closed-form fixed point.
    let _ = ode_subject_sensitivities(&model, &subj, &theta, &eta).expect("supported");
    assert!(
        crate::dosing::last_ss_equilibration_cycles() > 1,
        "a nonlinear disposition must fall back to the pulse-train iteration"
    );
}

// SS + first_order under IOV: the SS-dosed occasion's κ enters the equilibrated trough, and
// later occasions' κ enter the forward walk — the whole stacked `[η, κ_g0, κ_g1]` gradient
// must match FD of the production `predict_iov`. κ is just another param axis to
// `equilibrate_ss_input_rate_state_g`, so this is the IOV teeth on the same helper.
const ONECPT_SS_FIRST_ORDER_IOV: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

#[test]
fn ode_provider_ss_first_order_iov_matches_predict_iov() {
    let model = parse_model_string(ONECPT_SS_FIRST_ORDER_IOV).expect("parse SS first_order IOV");
    assert_eq!(model.n_kappa, 1);
    let theta = vec![1.0, 20.0, 0.15];
    // One SS dose in occasion 1; observations span occasions 1 and 2, so there are two
    // κ groups and the stacked vector is [η_CL, κ_g0, κ_g1].
    let mut subj = ss_absorption_subject(&[1.0, 3.0, 5.0, 7.0], 8.0);
    subj.occasions = vec![1, 1, 2, 2];
    subj.dose_occasions = vec![1];
    let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(groups.len(), 2, "fixture must have two κ occasion groups");
    let stacked = vec![0.12, 0.05, -0.08];
    assert_eq!(stacked.len(), model.n_eta + groups.len() * model.n_kappa);

    let sens = ode_subject_sensitivities_iov(&model, &subj, &theta, &stacked)
        .expect("#835: SS + first_order under IOV must be analytic");

    // FD reference: production `predict_iov`, unpacking stacked → (η_bsv, per-group κ).
    let pred = |th: &[f64], st: &[f64], j: usize| -> f64 {
        let eta_bsv = st[..model.n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..groups.len())
            .map(|g| {
                st[model.n_eta + g * model.n_kappa..model.n_eta + (g + 1) * model.n_kappa].to_vec()
            })
            .collect();
        crate::pk::predict_iov(&model, &subj, th, &eta_bsv, &kappas)[j]
    };

    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            obs.f,
            pred(&theta, &stacked, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        for k in 0..stacked.len() {
            let mut sp = stacked.clone();
            sp[k] += he;
            let mut sm = stacked.clone();
            sm[k] -= he;
            let g = (pred(&theta, &sp, j) - pred(&theta, &sm, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
        }
        for m in 0..model.n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&tp, &stacked, j) - pred(&tm, &stacked, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-3, epsilon = 1e-6);
        }
    }
}

/// 2nd-order saltation validation (#472 review round 2 #3/#4): the δlag² coefficients
/// for the **rate-boundary** (infusion) and **bolus `jg_cross`** (multi-dose) cases are
/// FD-checked against the analytic gradient. All use `ETA_LAG` (lag-on-IIV) so the
/// lagtime η-jet — hence the saltation η-Hessian rows — is non-zero.
#[test]
fn ode_provider_lagtime_infusion_hessian_matches_fd_of_grad() {
    // Infusion + lag-on-IIV → exercises the rate-on/off `coef2 = −s·½·J·(Δr·e_cmt)`.
    let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0, 12.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
    check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
}

#[test]
fn ode_provider_multidose_bolus_lagtime_hessian_matches_fd_of_grad() {
    // ≥2 bolus doses + lag-on-IIV → the 2nd dose has a non-zero pre-dose state, so the
    // `jg_cross` (post-dose Jacobian on pre-dose velocity) term fires (#472 review #4).
    let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag ODE");
    let mut subject = bolus_subject(&[1.0, 3.0, 7.0, 10.0, 14.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
    ];
    check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
}

/// Reset + lag-on-IIV: the rate-off saltation is skipped after the reset, and the bolus
/// saltation's 2nd order across the reset boundary must still match (#472 review #3).
#[test]
fn ode_provider_lagtime_reset_hessian_matches_fd_of_grad() {
    let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag ODE");
    let mut subject = bolus_subject(&[2.0, 5.0, 9.0, 13.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![8.0];
    check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
}

// 1-cpt **Michaelis–Menten (strongly nonlinear)** elimination with an estimated lagtime
// carrying IIV. The MM curvature `∂(ẋ)/∂central = −VM·KM/(KM+central)²` is large when
// `central ~ KM`, so a concurrent high-rate infusion forcing into the same compartment
// contributes a dominant `J·(rate·e_central)` to the bolus saltation's δlag² term — the
// term the bare-user-RHS velocity drops (#472 review [1]).
const MM_LAG_INF_ODE: &str = r#"
[parameters]
  theta TVVM(30.0, 1.0, 300.0)
  theta TVKM(10.0, 0.5, 100.0)
  theta TVV(10.0,  1.0, 200.0)
  theta TVLAG(0.5, 0.01,  5.0)

  omega ETA_VM  ~ 0.09
  omega ETA_LAG ~ 0.04

  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  VM      = TVVM * exp(ETA_VM)
  KM      = TVKM
  V       = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -VM * central / (KM + central)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

/// **Concurrent bolus + infusion under lagtime, nonlinear RHS** (the coverage gap from
/// #472 review [2]). A bolus co-timed with a finite-duration infusion into the same
/// MM-eliminated compartment, both shifted by an estimated lagtime with IIV. Exercises the
/// bolus saltation's full δlag² Hessian (`d²f/dη_LAG²`, `d²f/dη_LAG dθ`) in the presence of
/// a concurrent infusion forcing on a strongly nonlinear RHS — validated vs FD of the
/// analytic gradient. (Review finding [1] proposed adding the infusion forcing to the
/// saltation velocity `J·ẋ`; this test refutes it — the forcing is continuous across the
/// bolus event and does not shift with the bolus lag, so adding it to the event-time
/// saltation makes `d²f/dη_LAG²` diverge from FD here. The bare user-RHS velocity is
/// correct.)
#[test]
fn ode_provider_bolus_concurrent_infusion_lagtime_hessian_matches_fd() {
    let model = parse_model_string(MM_LAG_INF_ODE).expect("parse MM lag+inf ODE");
    assert!(model.has_lagtime());
    let mut subject = bolus_subject(&[0.75, 1.0, 1.5, 2.0, 3.0, 4.5]);
    // Co-timed bolus + high-rate finite-duration infusion into the MM compartment, both
    // shifted by LAGTIME. Infusion rate 60, amt 240 → 4 h window; obs sit in the steep
    // MM region just after the lagged arrival.
    subject.doses = vec![
        DoseEvent::new(0.0, 30.0, 1, 0.0, false, 0.0),
        DoseEvent::new(0.0, 240.0, 1, 60.0, false, 0.0),
    ];
    assert!(subject.doses[1].is_infusion() && model.has_lagtime());
    assert!(ode_tvcov_supported(&model, &subject));
    // η = [ETA_VM, ETA_LAG]; θ = [TVVM, TVKM, TVV, TVLAG].
    // First validate the 1st-order gradient vs FD of the predictions (independent of the
    // analytic Hessian internals): if df is correct, FD-of-df is the true d²f.
    check_vs_production(&model, &subject, &[30.0, 10.0, 10.0, 0.5], &[0.1, 0.05]);
    check_hessian_vs_fd_of_grad(&model, &subject, &[30.0, 10.0, 10.0, 0.5], &[0.1, 0.05]);
}

// 1-cpt oral ODE with an **estimated lagtime** on the depot dose. The dose arrives
// at `t + LAGTIME`; the lagtime sensitivity (`∂f/∂TVLAG`, and `∂f/∂η` if lag carries
// IIV) comes from the event-time saltation injected at the dose. Tier 2 (#439).
const ONECPT_ORAL_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(1.0,  0.01, 100.0)
  theta TVV(10.0,  1.0, 500.0)
  theta TVKA(1.0,  0.01, 50.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
  LAGTIME = TVLAG
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **Per-compartment lagtime `ALAG1`.** Lagtime declared as a compartment-indexed
/// `ALAG1` (not the bare `LAGTIME` slot) — each dose reads its lag from its own
/// compartment's slot (`indexed_slot(Lag, cmt)`), so per-dose differing lags are
/// exact. Validates `f`/`∂f/∂η`/`∂f/∂θ` (incl. the `θ_ALAG` column) against the
/// production predictor + FD (#439 / #369).
const ONECPT_ORAL_ALAG1_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVALAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = TVALAG
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_provider_alag1_matches_production() {
    let model = parse_model_string(ONECPT_ORAL_ALAG1_ODE).expect("parse ALAG1 ODE");
    assert!(model.has_lagtime());
    assert!(
        model
            .active_dose_attr_map()
            .has_indexed_attr(crate::types::DoseAttr::Lag),
        "ALAG1 must be a compartment-indexed lag"
    );
    assert!(ode_analytical_supported(&model));
    let subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
    // θ = [TVCL, TVV, TVKA, TVALAG]; the TVALAG column is the per-compartment lag.
    check_vs_production(&model, &subject, &[1.0, 10.0, 1.2, 0.4], &[0.1]);
}

#[test]
fn ode_provider_lagtime_matches_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_ODE).expect("parse oral lag ODE");
    assert!(model.has_lagtime(), "model must declare a lagtime");
    assert!(
        ode_analytical_supported(&model),
        "bare-lagtime oral ODE must be analytic-supported"
    );
    // Single bolus into the depot at t=0; observations span the lagged onset.
    let subject = bolus_subject(&[0.25, 0.75, 1.5, 3.0, 6.0, 10.0]);
    // θ = [TVCL, TVV, TVKA, TVLAG]; the TVLAG column is driven entirely by the
    // event-time saltation, so it is the key check.
    check_vs_production(&model, &subject, &[1.0, 10.0, 1.0, 0.5], &[0.12, -0.08]);
}

// 2-cpt IV ODE under LTBS (`log(DV) ~ additive`): the readout is log-transformed
// (`p = ln f`), so the provider's f/∂f/∂η/∂f/∂θ must match the (also
// log-transformed) production predictor. Tier 1 output transform (#410).
const TWOCPT_ODE_LTBS: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma ADD_LOG ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  log(DV) ~ additive(ADD_LOG)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

#[test]
fn ode_provider_ltbs_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_LTBS).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "LTBS ODE should be supported (Tier 1)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

// 2-cpt IV ODE with a constant `ScalarScale` output divisor (`obs_scale = 50`)
// over the central-amount readout: `f = central / 50`. Tier 1 output transform.
const TWOCPT_ODE_SCALARSCALE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = 50
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

#[test]
fn ode_provider_scalar_scale_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_SCALARSCALE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "constant ScalarScale ODE should be supported (Tier 1)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

// 2-cpt IV ODE with an η-dependent `ExpressionScale` divisor `obs_scale = 1000 / V1`
// (`V1` carries `ETA_V1`) over the central-amount (`ObsCmt`) readout. The scale is
// applied as the subject-static quotient on the final `(θ,η)`-space jet (#486),
// reusing the closed-form provider's `apply_expression_scale_outer`.
const TWOCPT_ODE_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = 1000 / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// η-dependent `ExpressionScale` `obs_scale = 1000 / V1` on the ODE path (#486): the
/// outer provider's scaled `f` / `∂f/∂η` / `∂f/∂θ` must match FD of the production
/// predictor (which divides by the same scale through `apply_scaling`), and the
/// 2nd-order blocks must match FD of the analytic gradient — exercising the
/// `apply_expression_scale_outer` quotient rule layered onto the ODE jet.
#[test]
fn ode_provider_expression_scale_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_EXPRSCALE).expect("parse");
    assert!(
        matches!(
            model.scaling,
            ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ),
        "model must carry a differentiable scale program"
    );
    assert!(
        ode_analytical_supported(&model),
        "η-dependent ExpressionScale ODE must be supported (#486)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    check_hessian_vs_fd_of_grad(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

/// `ExpressionScale` + **LTBS** must route to FD (`None`): the walk applies the LTBS
/// log in PK-param space *before* the η/θ chain, so the production scale-then-log order
/// can't be reproduced by a post-walk quotient — `expression_scale_axes_admissible`
/// gates it on `!log_transform`, so the analytic scope declines it (#486).
#[test]
fn ode_provider_expression_scale_combos_fall_back_to_fd() {
    // + LTBS → out of analytic scope (the walk applies LTBS pre-chain, so the
    // production scale-then-log order can't be reproduced by a post-walk quotient).
    // This is the only `ExpressionScale` combination that still declines: since #486
    // the event-driven walk applies the subject-static scale quotient, so
    // `ExpressionScale` + time-varying covariate / `TIME` is analytic (validated in
    // `ode_expression_scale_on_event_walk_matches_production`).
    let mut ltbs = parse_model_string(TWOCPT_ODE_EXPRSCALE).expect("parse");
    ltbs.log_transform = true;
    assert!(
        !ode_analytical_supported(&ltbs),
        "ExpressionScale + LTBS must fall back to FD"
    );
}

/// A 2-cpt IV ODE with **time-varying covariates** (`WT` on `V1`) **and** an η-dependent
/// `ExpressionScale` divisor `obs_scale = 1000 / V1`. The scale is **subject-static even
/// under TV covariates** (production `apply_scaling` reads `subject.covariates`), so the
/// non-IOV TV-cov walk applies it as a single subject-static post-walk quotient (#486 —
/// the IOV per-occasion-group machinery of #590 collapses to one jet, no κ). The walk
/// itself runs the per-event TV-cov PK params; only the divisor is subject-static.
const TWOCPT_ODE_EXPRSCALE_TVCOV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)
  V1 = TVV1 * (WT/70) * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = 1000 / V1
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// Build a TV-cov subject (WT on dose + per-obs) for the 2-cpt ExpressionScale model,
/// with `subject.covariates` carrying the subject-static WT the scale divisor reads.
fn exprscale_tvcov_subject() -> Subject {
    let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.covariates = wt(70.0);
    subject.dose_covariates = vec![wt(62.0)];
    subject.obs_covariates = vec![wt(62.0), wt(66.0), wt(70.0), wt(74.0), wt(80.0), wt(88.0)];
    subject
}

/// TV-cov + `ExpressionScale` on the **non-IOV** ODE walk (#486): the outer provider's
/// scaled `f` / `∂f/∂η` / `∂f/∂θ` must match FD of the production TV-cov predictor (which
/// divides by the subject-static `obs_scale` via `apply_scaling`), and the 2nd-order
/// blocks must match FD of the analytic gradient — exercising the subject-static
/// `apply_expression_scale_outer` quotient layered onto the per-event TV-cov jet.
#[test]
fn ode_provider_tvcov_expression_scale_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_EXPRSCALE_TVCOV).expect("parse");
    assert!(
        matches!(
            model.scaling,
            ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ),
        "model must carry a differentiable scale program"
    );
    let subject = exprscale_tvcov_subject();
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TV-cov + ExpressionScale must be analytic on the non-IOV ODE walk (#486)"
    );
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// Inner/outer parity for TV-cov + `ExpressionScale` (#486): the light `Dual1` inner
/// η-gradient (`run_subject_tvcov_eta` + `apply_expression_scale_inner_dispatch`) must
/// equal the full `Dual2` outer `df_deta` (`run_subject_tvcov` + the outer quotient).
/// This is the guardrail: both walks share `ode_tvcov_supported`, so the inner quotient
/// must move in lockstep with the outer or the EBE gradient silently diverges.
#[test]
fn ode_provider_tvcov_expression_scale_inner_eta_grad_matches_outer() {
    let model = parse_model_string(TWOCPT_ODE_EXPRSCALE_TVCOV).expect("parse");
    let subject = exprscale_tvcov_subject();
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// Build the TV-cov subject for the 1-cpt `ONECPT_ODE_TVCOV` model with an EVID=2
/// covariate-only breakpoint (`pk_only`) at t=3 carrying its own WT snapshot.
fn tvcov_pkonly_subject() -> Subject {
    let mut subject = tvcov_subject();
    subject.pk_only_times = vec![3.0];
    subject.pk_only_covariates = vec![HashMap::from([("WT".to_string(), 75.0)])];
    subject
}

/// TV-cov + EVID=2 pk-only covariate breakpoint on the **non-IOV** ODE walk (#486): the
/// analytic outer `SubjectSens` (value, `∂f/∂η`, `∂f/∂θ`, + 2nd order) must match FD of
/// the production TV-cov predictor, which carries the same `K_PKONLY` breakpoint. With no
/// κ there is no `iov_combined_pk_only` analogue to build — pk-only events seed at their
/// own η-snapshot exactly like obs/dose (the IOV path got this in #590).
#[test]
fn ode_provider_tvcov_pkonly_matches_production() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let subject = tvcov_pkonly_subject();
    assert!(
        !subject.pk_only_times.is_empty(),
        "fixture must carry an EVID=2 covariate breakpoint"
    );
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TV-cov + EVID=2 pk-only must be analytic on the non-IOV ODE walk (#486)"
    );
    let theta = vec![1.0, 20.0, 0.75];
    let eta = vec![0.1];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// Inner/outer parity for TV-cov + EVID=2 pk-only (#486): the light inner η-gradient must
/// equal the full outer `df_deta`. Both walks share `ode_tvcov_supported`, so opening the
/// pk-only gate enables both — this confirms the inner walk consumes the breakpoint too.
#[test]
fn ode_provider_tvcov_pkonly_inner_eta_grad_matches_outer() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let subject = tvcov_pkonly_subject();
    let theta = vec![1.0, 20.0, 0.75];
    let eta = vec![0.1];
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// The light `Dual1` inner provider's `f` / `∂f/∂η` must equal the full `Dual2`
/// outer provider's `f` / `df_deta` exactly — both are exact analytic, only the
/// dual order differs (and `solve_ode_g` uses value-based step control, so the
/// trajectories match). This is what makes the inner EBE loop's analytic
/// η-gradient correct (#410). Run across the readout/dose variants so the light
/// driver's branches are exercised: Form-C readout (`TWOCPT_ODE`), an `ObsCmt`
/// model with non-zero `init(...)` (exercises `dual1_init_state`), estimated
/// bioavailability `F` (`BIOAV_ODE` — the `f_bio` path + `ObsCmt` arm), and LTBS.
#[test]
fn ode_light_inner_eta_grad_matches_full_provider() {
    // Form-C readout, IV bolus.
    let m = parse_model_string(TWOCPT_ODE).expect("parse");
    check_inner_outer_eta_parity(
        &m,
        &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
        &[4.0, 12.0, 2.0, 25.0],
        &[0.12, -0.08],
    );

    // ObsCmt readout + non-zero init(...) → exercises `dual1_init_state`.
    let m = parse_model_string(INIT_ODE).expect("parse");
    let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    s.doses = vec![];
    check_inner_outer_eta_parity(&m, &s, &[1.0, 20.0], &[0.1, -0.05]);

    // Estimated bioavailability F + ObsCmt readout, oral depot → `f_bio` path.
    let m = parse_model_string(BIOAV_ODE).expect("parse");
    let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    check_inner_outer_eta_parity(&m, &s, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

    // Compartment-indexed F1 (with IIV) → per-compartment `f_bio_slot` path (#486).
    let m = parse_model_string(F1_ODE).expect("parse");
    let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    check_inner_outer_eta_parity(&m, &s, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

    // LTBS output transform over the Dual1 readout.
    let m = parse_model_string(TWOCPT_ODE_LTBS).expect("parse");
    check_inner_outer_eta_parity(
        &m,
        &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
        &[4.0, 12.0, 2.0, 25.0],
        &[0.12, -0.08],
    );

    // η-dependent `ExpressionScale` divisor (#486) → the inner η-only quotient
    // (`apply_expression_scale_inner_dispatch`) must equal the full provider's
    // scaled `df_deta`.
    let m = parse_model_string(TWOCPT_ODE_EXPRSCALE).expect("parse");
    check_inner_outer_eta_parity(
        &m,
        &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
        &[4.0, 12.0, 2.0, 25.0],
        &[0.12, -0.08],
    );

    // Estimated lagtime with IIV (`ETA_LAG`), multi-dose — exercises the event-time
    // saltation (incl. the 2nd-dose `jg_cross`) in BOTH the Dual1 inner and the Dual2
    // outer walk, which must agree to provider tolerance (#472 review round 2 #8).
    let m = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag ODE");
    let mut s = bolus_subject(&[1.0, 3.0, 7.0, 10.0]);
    s.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
    ];
    check_inner_outer_eta_parity(&m, &s, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
}

/// The inner EBE loop must actually *resolve* to the analytic η-gradient for an
/// in-scope ODE subject (not merely be correct when called) — i.e. the wiring in
/// `analytic_inner_grad_supported` / `resolve_gradient_method` engages (#410).
#[test]
fn ode_inner_gradient_route_resolves_analytic() {
    use crate::estimation::inner_optimizer::{resolve_gradient_method, InnerGradientMethod};
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    let subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
    assert_eq!(
        resolve_gradient_method(&model, &subject),
        InnerGradientMethod::Analytic,
        "in-scope ODE subject must use the analytic inner η-gradient (#410)"
    );
    // The provider entry the inner loop actually calls (`subject_eta_grad`) must
    // route the ODE model to the light Dual1 provider, not decline.
    let g = crate::sens::provider::subject_eta_grad(
        &model,
        &subject,
        &[4.0, 12.0, 2.0, 25.0],
        &[0.1, -0.05],
    );
    assert!(
        g.is_some_and(|v| v.len() == subject.obs_times.len()),
        "subject_eta_grad must serve an in-scope ODE subject via the light provider"
    );
}

// 1-cpt oral ODE with estimated, logit-normal bioavailability F — the dose
// loads `F·AMT` into the depot, so F's derivative must flow through the
// injection. Mirrors examples/bioavailability_ode.ferx.
const BIOAV_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1,  50.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVKA(1.5,  0.05,  20.0)
  theta THETA_F(0.70, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.10
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = inv_logit(logit(THETA_F) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// Estimated bioavailability F: the provider must propagate F's derivative
/// through the `F·AMT` depot loading (and the logit/inv_logit individual-F
/// map), matching the production predictor and its FD.
#[test]
fn ode_provider_oral_bioavailability_matches_production() {
    let model = parse_model_string(BIOAV_ODE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "estimated F should be in scope"
    );
    let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    check_vs_production(&model, &subject, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);
}

// 1-cpt oral ODE with a *compartment-indexed* bioavailability `F1` (dose into the
// depot, cmt 1) carrying IIV (logit-normal). `F1` lands in the dose-attr map at its
// own slot, NOT the bare `PK_IDX_F` (which stays 1.0): so a walk that read the bare
// slot would apply F = 1.0 and diverge from production's `f_bio(cmt=1)` — both in
// value and in ∂/∂F. The IIV (`ETA_F`) routes F1 into the inner η-gradient too (#486).
const F1_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_F1(0.70, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.10
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F1 = inv_logit(logit(THETA_F1) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// 2-cpt IV ODE with *distinct* per-compartment bioavailabilities `F1` (central,
// cmt 1) and `F2` (peripheral, cmt 2). With one dose into each compartment, a walk
// that did not key F by the dose's own compartment (e.g. applied dose 0's F to
// both) would diverge from production. `obs_cmt = central`, so both F1 (direct) and
// F2 (via peripheral→central redistribution) move the observed concentration, hence
// both ∂/∂F1 and ∂/∂F2 are observable (#486).
const F1F2_IV_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  theta THETA_F1(0.80, 0.001, 0.999)
  theta THETA_F2(0.50, 0.001, 0.999)
  omega ETA_CL ~ 0.10
  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  F1 = THETA_F1
  F2 = THETA_F2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `f_bio_slot` resolves a dose's bioavailability slot per compartment: the
/// indexed `F{cmt}` slot when declared, else the bare `PK_IDX_F` (#486).
#[test]
fn f_bio_slot_resolves_indexed_then_bare() {
    let m = parse_model_string(F1F2_IV_ODE).expect("parse F1F2");
    let ode = m.ode_spec.as_ref().expect("ode_spec");
    let bare = crate::types::PK_IDX_F;
    let s1 = f_bio_slot(ode, 1);
    let s2 = f_bio_slot(ode, 2);
    assert_ne!(s1, bare, "F1 must resolve to its own indexed slot");
    assert_ne!(s2, bare, "F2 must resolve to its own indexed slot");
    assert_ne!(s1, s2, "F1 and F2 occupy distinct slots");
    assert_eq!(
        s1,
        ode.dose_attr_map
            .indexed_slot(crate::types::DoseAttr::F, 1)
            .unwrap()
    );
    // A compartment with no indexed `F` falls back to the bare slot.
    assert_eq!(f_bio_slot(ode, 3), bare);
}

/// Compartment-indexed bioavailability (`F1`/`F2`, #369) is now served analytically
/// (#486): the dual walks resolve `F` per dose compartment, so analytic f / ∂f/∂η /
/// ∂f/∂θ match the production predictor and its FD. Covers the single-indexed depot
/// case (with IIV on F1) and distinct F1≠F2 into two compartments.
#[test]
fn ode_provider_compartment_indexed_f_matches_production() {
    // Single indexed F1 into the depot, with IIV.
    let model = parse_model_string(F1_ODE).expect("parse F1");
    assert!(
        ode_analytical_supported(&model),
        "compartment-indexed F1 should now be in scope"
    );
    let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    check_vs_production(&model, &subject, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

    // Distinct F1 (central) and F2 (peripheral) with a dose into each compartment.
    let model = parse_model_string(F1F2_IV_ODE).expect("parse F1F2");
    assert!(
        ode_analytical_supported(&model),
        "distinct per-compartment F1/F2 should be in scope"
    );
    let mut subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0),
    ];
    check_vs_production(
        &model,
        &subject,
        &[4.0, 12.0, 2.0, 25.0, 0.80, 0.50],
        &[0.1],
    );
}

/// Compartment-indexed lag (`ALAG1`) IS supported (#472): it is handled on the
/// event-driven saltation walk, which reads each dose's lag from its own slot, so
/// the model passes the analytic gate and routes to the event-driven walk (not the
/// static superposition walk). (Indexed `F` is also supported — parity test above.)
#[test]
fn ode_analytical_supports_per_compartment_lag() {
    const ALAG1_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_ALAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = THETA_ALAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(ALAG1_ODE).expect("parse ALAG1");
    // Compartment-indexed `ALAG1` IS supported now: lagtime is handled on the
    // event-driven saltation walk, which reads each dose's lag from its own slot
    // (`indexed_slot(Lag, cmt)`), so per-compartment / per-dose lags are exact (#439).
    assert!(
        ode_analytical_supported(&m),
        "compartment-indexed ALAG1 is supported (event-time saltation, per-dose lag slot)"
    );
    // It routes to the event-driven walk, not the static superposition walk.
    let subj = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
    assert!(ode_tvcov_supported(&m, &subj), "ALAG1 → event-driven walk");
    assert!(
        !ode_subject_supported(&m, &subj),
        "ALAG1 not on the static walk"
    );
}

/// A **`first_order` per-route** absorption lag (`fn(..., lag=L)`, #859) is served on
/// the exact analytic path (Slice 1 of the Phase-2 follow-up to #857). The event-driven
/// walk now injects each route's onset saltation at its own `t_dose + lag_cmt + lag_route`
/// (`K_ROUTE_ONSET`), so the analytic gradient — including `∂f/∂LAG2` — matches the f64
/// predictor's finite differences. This IR+DR model (immediate `first_order(ka=KA1)` plus
/// delayed `first_order(ka=KA2, lag=LAG2)` on one compartment) exercises both the shared
/// `K_DOSE` onset (the unlagged IR route) and the new `K_ROUTE_ONSET` onset (the lagged DR
/// route) in one subject. It routes to the event-driven walk, never the static one (a
/// *pure*-route-lag model has `has_lagtime() == false`, so the static-walk decline is its
/// own check, not a fallout of the lagtime decline).
#[test]
fn first_order_per_route_lag_is_analytic() {
    const PER_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 20.0)
  theta TVKA2(0.3, 0.05, 20.0)
  theta TVLAG2(2.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FR1  = TVFR1
  FR2  = 1 - TVFR1
  KA1  = TVKA1
  KA2  = TVKA2
  LAG2 = TVLAG2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAG2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(PER_ROUTE_LAG).expect("parse per-route lag model");
    // Sanity: the model actually carries a per-route lag on exactly one forcing.
    let ode = m.ode_spec.as_ref().expect("ode spec");
    assert_eq!(
        ode.input_rate
            .iter()
            .filter(|f| f.lag_slot.is_some())
            .count(),
        1,
        "one route carries a per-route lag"
    );
    assert!(
        ode_analytical_supported(&m),
        "a first_order per-route lag is now analytic (#859)"
    );
    // Obs bracket the DR onset (LAG2 = 2.0): pre-onset (only IR active) and post-onset
    // (both routes active), so the FD check exercises `∂f/∂LAG2` across the discontinuity.
    let mut subject = bolus_subject(&[0.5, 1.0, 1.5, 2.5, 4.0, 8.0, 12.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    // Routes to the event-driven walk (the route onset is an event-time saltation), never
    // the static superposition walk.
    assert!(
        ode_tvcov_supported(&m, &subject),
        "route-lag subject routes to the event-driven walk"
    );
    assert!(
        !ode_subject_supported(&m, &subject),
        "route-lag subject is off the static walk"
    );
    // Analytic value / ∂f/∂η / ∂f/∂θ (incl. ∂f/∂LAG2) all match the f64 predictor's FD.
    // (The value is independently correct: at this tolerance it matches the exact Bateman
    // closed form for the two-route 1-cpt system to ~6e-8 — the loose-default kink
    // resolution is what a tight `ode_reltol` sharpens, not a structural error.)
    let theta = [5.0, 50.0, 0.6, 1.5, 0.3, 2.0];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    // Inner light-provider η-gradient must match the outer provider exactly (EBE-loop
    // parity), so the route onset is consistent between the inner and outer walks.
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
    // The new δlag² saltation coefficient (the route onset's 2nd-order term) is exercised
    // against a ground-truth double-FD Hessian of the predictor.
    check_hessian_vs_production_fd(&m, &subject, &theta, &eta);
}

/// **Compartment lag + per-route lag** (#859): the headline composition — a bare `ALAG1`
/// shifts *both* routes, and `lag=LAG2` shifts *one* further, so the delayed route's onset is
/// `dose + lag_cmt + lag_route`. This is the ONLY test exercising the `has_lagtime == true`
/// side: the `K_ROUTE_ONSET` `lag_cmt` term (read from `dose_lag_slot`), the combined
/// `∂/∂(lag_cmt + lag_route)` jet, AND the `K_DOSE` shared-onset **exclusion** of the
/// route-lagged forcing (the IR route still fires at `K_DOSE` under `ALAG1`, the DR route at
/// its own `K_ROUTE_ONSET`). `check_vs_production` FD-checks both `∂f/∂ALAG1` and `∂f/∂LAG2`.
#[test]
fn compartment_lag_plus_per_route_lag_is_analytic() {
    const CMT_PLUS_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 20.0)
  theta TVKA2(0.3, 0.05, 20.0)
  theta TVLAG2(2.0, 0.0, 10.0)
  theta TVALAG(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  FR1   = TVFR1
  FR2   = 1 - TVFR1
  KA1   = TVKA1
  KA2   = TVKA2
  LAG2  = TVLAG2
  ALAG1 = TVALAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAG2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(CMT_PLUS_ROUTE_LAG).expect("parse cmt-lag + route-lag model");
    assert!(m.has_lagtime(), "model carries a compartment lag (ALAG1)");
    assert!(
        ode_analytical_supported(&m),
        "compartment lag composed with a first_order route lag is analytic (#859)"
    );
    // IR onset at ALAG1 = 0.5, DR onset at ALAG1 + LAG2 = 2.5; obs straddle both.
    let mut subject = bolus_subject(&[0.3, 1.0, 2.0, 3.0, 5.0, 9.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(
        ode_tvcov_supported(&m, &subject),
        "routes to the event-driven walk"
    );
    let theta = [5.0, 50.0, 0.6, 1.5, 0.3, 2.0, 0.5];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&m, &subject, &theta, &eta);
}

/// **Multi-dose** per-route lag (#859): two doses each open their own delayed route, so the
/// per-`(dose, route)` `K_ROUTE_ONSET` events must superpose — each dose's delayed onset
/// carries its own boundary jet and its own TAD anchor (the most-recent dose), and the two
/// delayed routes overlap in time. Single-dose tests never exercise the `last_dose_eff`
/// anchor *selection* (only one dose), so this pins `∂f/∂LAG` exact across repeated dosing.
#[test]
fn multi_dose_first_order_per_route_lag_is_analytic() {
    const PER_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 20.0)
  theta TVKA2(0.3, 0.05, 20.0)
  theta TVLAG2(2.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FR1  = TVFR1
  FR2  = 1 - TVFR1
  KA1  = TVKA1
  KA2  = TVKA2
  LAG2 = TVLAG2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAG2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(PER_ROUTE_LAG).expect("parse per-route lag model");
    // Two doses at t=0 and t=8; DR onsets (LAG2=2) at t=2 and t=10. Obs straddle both.
    let mut subject = bolus_subject(&[1.0, 2.5, 4.0, 7.0, 9.0, 11.0, 14.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let theta = [5.0, 50.0, 0.6, 1.5, 0.3, 2.0];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
}

/// A **`zero_order` per-route** lag (`zero_order(..., lag=L)`, #859 Slice 2) is served on
/// the analytic path. The route-lagged window slides whole: its rate-on saltation fires at
/// the shifted `w_start = t_dose + lag_cmt + lag_route` (`K_ROUTE_ONSET`) and its rate-off at
/// `w_end = w_start + dur` (`K_ZO_END`), both carrying the `lag_route` jet — so the analytic
/// `∂f/∂LAG` and `∂f/∂DUR` match the f64 predictor's FD across both boundaries. `DUR` is
/// η-coupled so its moving-boundary η-block is FD-checked too.
#[test]
fn zero_order_per_route_lag_is_analytic() {
    const ZO_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.5, 24.0)
  theta TVLAG(2.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
  LAG = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(ZO_ROUTE_LAG).expect("parse zero-order route-lag model");
    assert!(
        ode_analytical_supported(&m),
        "a zero_order per-route lag is analytic (#859 Slice 2)"
    );
    // Window [LAG, LAG+DUR] = [2, 6]: obs straddle both the shifted start and the end.
    let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 6.5, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(
        ode_tvcov_supported(&m, &subject),
        "route-lag subject routes to the event-driven walk"
    );
    assert!(
        !ode_subject_supported(&m, &subject),
        "route-lag subject is off the static walk"
    );
    let theta = [5.0, 50.0, 4.0, 2.0];
    let eta = [0.12, -0.08];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&m, &subject, &theta, &eta);
}

/// **Compartment lag + a route-lagged `zero_order` window** (#859): `ALAG1` shifts the dose
/// arrival and `lag=LAG` shifts the window further, so the window is `[dose + ALAG1 + LAG,
/// + dur]`. This is the `has_lagtime == true` path for `zero_order`: the `K_DOSE` shared-onset
/// loop runs (under `ALAG1`) but **skips** this route-lagged window (S2-3), which instead
/// fires its rate-on at `K_ROUTE_ONSET`; and the `K_ZO_END` rate-off carries the combined
/// `∂/∂(ALAG1 + LAG + dur)` jet. `∂f/∂ALAG1`, `∂f/∂LAG`, `∂f/∂DUR` all FD-checked.
#[test]
fn compartment_lag_plus_route_lagged_zero_order_is_analytic() {
    const CMT_PLUS_ZO_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.5, 24.0)
  theta TVLAG(2.0, 0.0, 10.0)
  theta TVALAG(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  DUR   = TVDUR * exp(ETA_DUR)
  LAG   = TVLAG
  ALAG1 = TVALAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(CMT_PLUS_ZO_LAG).expect("parse cmt-lag + zo-route-lag model");
    assert!(m.has_lagtime(), "model carries a compartment lag (ALAG1)");
    assert!(
        ode_analytical_supported(&m),
        "compartment lag composed with a zero_order route lag is analytic (#859)"
    );
    // Window [ALAG1 + LAG, + DUR] = [2.5, 6.5]: obs straddle both boundaries.
    let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 7.0, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let theta = [5.0, 50.0, 4.0, 2.0, 0.5];
    let eta = [0.12, -0.08];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&m, &subject, &theta, &eta);
}

/// **TV-covariate across a route-lagged `zero_order` window end** (#859): closes the last
/// gap in the zero-order rate-off. When a time-varying covariate changes the PK params across
/// the window end `w_end`, the RHS Jacobian jumps, so `K_ZO_END` takes the **general**
/// `g⁻−g⁺` saltation branch (not the closed-form `pre == post` twin) — and there the
/// route-lag jet is carried by `route_lag_rep` on the cohort representative. A `WT`-on-`CL`
/// covariate switching between the obs bracketing `w_end = dose + LAG + DUR` forces that
/// branch; `check_vs_production` then FD-checks `∂f/∂LAG` (and `∂f/∂DUR`) through it, so the
/// `route_lag_rep` term is directly exercised, not just twin-verified against `route_lag_j`.
#[test]
fn tvcov_route_lagged_zero_order_matches_production() {
    const TVCOV_ZO_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVDUR(3.0, 0.5, 24.0)
  theta TVLAG(1.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
  LAG = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(TVCOV_ZO_ROUTE_LAG).expect("parse tvcov zo-route-lag model");
    // Window [LAG, LAG+DUR] = [1, 4]. Obs bracket w_end=4 (at t=3 and t=5) with DIFFERENT WT,
    // so the PK params jump across the window end → the general rate-off branch fires. Obs
    // deliberately avoid the window boundaries themselves (t=1, t=4): an obs coinciding with a
    // moving saltation boundary is a measure-zero point where the one-sided analytic
    // derivative and the symmetric FD disagree (FD averages the pre/post-jump sides).
    let mut subject = bolus_subject(&[0.5, 3.0, 5.0, 8.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let wt = |w: f64| std::collections::HashMap::from([("WT".to_string(), w)]);
    subject.covariates = wt(70.0);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(90.0), wt(100.0)];
    assert!(subject.has_tv_covariates(), "WT varies across records");
    assert!(
        ode_tvcov_supported(&m, &subject),
        "TV-cov + route-lagged zero_order is analytic on the event-driven walk"
    );
    let theta = [1.0, 20.0, 0.75, 3.0, 1.0];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
}

/// A **mixed** `first_order` + route-lagged `zero_order` on one compartment (#859 Slice 2):
/// the immediate `FR1*first_order` rides the smooth dual path with its onset at the dose
/// (`K_DOSE`, no route lag), while the delayed `FR2*zero_order(..., lag=L)` window opens at
/// `t_dose + lag` (`K_ROUTE_ONSET`) and closes at `+ dur` (`K_ZO_END`). One subject exercises
/// the shared `K_DOSE` onset (unlagged first-order) alongside the per-route window rate-on/off.
#[test]
fn mixed_first_order_route_lagged_zero_order_is_analytic() {
    const MIXED_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVDUR(4.0, 0.5, 24.0)
  theta TVLAG(2.0, 0.0, 10.0)
  theta TVFR(0.6, 0.05, 0.95)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  DUR = TVDUR
  LAG = TVLAG
  FR1 = TVFR
  FR2 = 1 - TVFR
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA) + FR2*zero_order(dur=DUR, lag=LAG) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(MIXED_ROUTE_LAG).expect("parse mixed route-lag model");
    assert!(
        ode_analytical_supported(&m),
        "mixed first_order + zero_order(lag) is analytic (#859 Slice 2)"
    );
    let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 6.5, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let theta = [5.0, 50.0, 1.2, 4.0, 2.0, 0.6];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
}

/// A **`transit` per-route** lag (`transit(..., lag=L)`, #859 Slice 3). The transit density
/// has a *continuous* onset (`R_in → 0` at `t_dose + lag_cmt + lag_route` for `n > 0`), so
/// no rate-on saltation is needed — the `K_ROUTE_ONSET` handler's `rate_at_zero` term is
/// zero. The route lag is served entirely by the continuous `∂R_in/∂lag_route` (through the
/// shared `add_prepared_input_rate_forcing` `t_eff` shift) plus the timeline break at the
/// onset. `∂f/∂LAG` therefore matches the f64 predictor's FD with no discontinuity term.
#[test]
fn transit_per_route_lag_is_analytic() {
    const TRANSIT_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVN(5.0, 1.0, 30.0)
  theta TVMTT(2.0, 0.1, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  LAG = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(TRANSIT_ROUTE_LAG).expect("parse transit route-lag model");
    assert!(
        ode_analytical_supported(&m),
        "a transit per-route lag is analytic (#859 Slice 3)"
    );
    let mut subject = bolus_subject(&[1.0, 2.5, 4.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(
        ode_tvcov_supported(&m, &subject),
        "route-lag subject routes to the event-driven walk"
    );
    let theta = [1.0, 20.0, 5.0, 2.0, 1.5];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
}

/// An **`igd` per-route** lag (inverse-Gaussian density, `igd(..., lag=L)`, #859 Slice 3).
/// Like `transit`, the IG onset is continuous (an essential singularity drives `R_in → 0` at
/// the onset for every valid `(mat, cv2)`), so there is no rate-on saltation — only the
/// continuous `∂R_in/∂lag_route` and the onset break. `∂f/∂LAG` matches FD.
#[test]
fn igd_per_route_lag_is_analytic() {
    const IGD_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVMAT(2.0, 0.1, 20.0)
  theta TVCV2(0.5, 0.01, 10.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MAT = TVMAT
  CV2 = TVCV2
  LAG = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(IGD_ROUTE_LAG).expect("parse igd route-lag model");
    assert!(
        ode_analytical_supported(&m),
        "an igd per-route lag is analytic (#859 Slice 3)"
    );
    let mut subject = bolus_subject(&[1.0, 2.5, 4.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let theta = [1.0, 20.0, 2.0, 0.5, 1.5];
    let eta = [0.1];
    check_vs_production(&m, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&m, &subject, &theta, &eta);
}

/// A **`weibull` per-route** lag stays on the FD fallback: `weibull`'s onset diverges for
/// shape `β < 1` (an integrable spike, not a finite `R_in(0⁺)` jump), so no closed-form
/// rate-on saltation exists — the same reason `weibull` + a compartment lagtime stays FD.
/// The permanent backstop that the kernel-classified route-lag gate (#859) admits only the
/// kernels whose onset the walk can actually emit; unlike `first_order` (analytic above),
/// `weibull` must never silently take the analytic path.
#[test]
fn weibull_per_route_lag_stays_fd() {
    const WEIBULL_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMAT(1.5, 0.05, 20.0)
  theta TVBETA(0.8, 0.1, 5.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  MAT  = TVMAT
  BETA = TVBETA
  LAG  = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=MAT, beta=BETA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(WEIBULL_ROUTE_LAG).expect("parse weibull route-lag model");
    let ode = m.ode_spec.as_ref().expect("ode spec");
    assert_eq!(
        ode.input_rate
            .iter()
            .filter(|f| f.lag_slot.is_some())
            .count(),
        1,
        "the weibull route carries a per-route lag"
    );
    assert!(
        !ode_analytical_supported(&m),
        "a weibull per-route lag must stay on FD (no closed-form onset saltation)"
    );
    let subj = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
    assert!(!ode_subject_supported(&m, &subj), "not on the static walk");
    assert!(!ode_tvcov_supported(&m, &subj), "not on the TV-cov walk");
}

/// A per-route lag **combined with IOV** (`n_kappa != 0`) is analytic on the IOV gate since #877
/// (was FD-gated under #859). The IOV walk is the same `integrate_tvcov_g`, so the per-kernel
/// `route_lag_analytic()` classifier admits `first_order` here — its κ-sensitivity rides the
/// per-occasion `pk_at_dose[k]` jet, FD-validated (value + gradient + Hessian) by the
/// `ode_iov_*_route_lag_*_matches_fd` family in `provider_tests.rs`. The non-IOV gate still
/// declines any `n_kappa != 0` model outright (its `n_kappa != 0` clause). A `weibull` route lag
/// still declines on both gates (divergent β < 1 onset — see `ode_iov_weibull_route_lag_*`).
#[test]
fn per_route_lag_under_iov_is_analytic() {
    const IOV_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta TVLAG(2.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL + KAPPA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(IOV_ROUTE_LAG).expect("parse IOV route-lag model");
    assert_eq!(m.n_kappa, 1, "model carries one IOV kappa");
    assert_eq!(
        m.ode_spec
            .as_ref()
            .unwrap()
            .input_rate
            .iter()
            .filter(|f| f.lag_slot.is_some())
            .count(),
        1,
        "one route carries a per-route lag"
    );
    // Non-IOV gate: still declined by the `n_kappa != 0` clause (an IOV model is off it).
    assert!(
        !ode_analytical_supported(&m),
        "an IOV model is off the non-IOV analytic gate"
    );
    // IOV gate: `first_order` route lag is now admitted analytically (#877).
    assert!(
        ode_iov_supported(&m),
        "#877: first_order per-route lag under IOV is analytic"
    );
}

/// Indexed `F` is now in model-level scope (parity test above), but the
/// *per-subject* gate must still route a **rate-defined infusion under `F ≠ 1`**
/// to FD. NONMEM reshapes such an infusion's window (holds the rate, scales the
/// duration to `F·amt/rate`, #419), whereas the dual walk scales the rate
/// magnitude over the *original* window — so the analytic gradient would diverge
/// from the f64 predictor. The model-level gate admits the indexed-`F` model; the
/// subject gate (`has_bioavailability() && has_rate_defined_infusion()`) declines
/// it. Crucially `has_bioavailability()` detects the indexed `F{cmt}` form too, so
/// dropping the `ode_analytical_supported` indexed-`F` decline (#486) does *not*
/// open this infusion path. A bolus of the same model stays in scope, so the
/// decline is attributable to the infusion, not the `F`.
#[test]
fn ode_subject_declines_indexed_f_rate_defined_infusion() {
    let model = parse_model_string(F1F2_IV_ODE).expect("parse F1F2");
    // Model-level scope admits indexed F (the indexed-F decline gate is gone, #486).
    assert!(
        ode_analytical_supported(&model),
        "indexed F1/F2 model is in model-level scope"
    );

    // A bolus subject of this model IS served analytically.
    let bolus = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    assert!(
        !bolus.doses[0].is_infusion(),
        "control dose must be a bolus"
    );
    assert!(
        ode_subject_supported(&model, &bolus),
        "indexed-F bolus subject should be served analytically"
    );

    // The same model with a *rate-defined* infusion (RATE>0) into the dosed
    // compartment must decline to FD: `F` reshapes the window, which the dual rate
    // scale does not reproduce (#419).
    let mut infusion = bolus.clone();
    infusion.doses = vec![DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0)];
    assert!(
        infusion.doses[0].is_infusion(),
        "rate=200 must be a rate-defined infusion"
    );
    assert!(
        !ode_subject_supported(&model, &infusion),
        "rate-defined infusion under indexed F must decline to FD (#419)"
    );
}

/// Infusion doses (RATE>0): the dual loop must add the rate forcing over the
/// infusion window and match the production predictor through during- and
/// post-infusion observations.
#[test]
fn ode_provider_2cpt_infusion_matches_production() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    // amt=1000, rate=200 → 5 h infusion into central; obs during and after.
    let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 6.0, 9.0, 24.0]);
    subject.doses = vec![DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0)];
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

// 1-cpt with a non-zero `init(central) = 1000/V` baseline (depends on V), no
// dose — exercises the dual-seeded initial state and its V derivative.
const INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 1000.0 / V
  d/dt(central) = -CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// Non-zero `init(...)`: the dual initial state (value + parameter derivative)
/// must match the production predictor + FD across the decay from baseline.
#[test]
fn ode_provider_init_matches_production() {
    let model = parse_model_string(INIT_ODE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "init(...) should be in scope"
    );
    let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    subject.doses = vec![]; // baseline comes from init, not a dose
    check_vs_production(&model, &subject, &[1.0, 20.0], &[0.1, -0.05]);
}

/// #530: a **modeled-duration infusion** (`RATE=-2` → `D1`) gets analytic
/// sensitivities on the event-driven walk. The infusion window end `t_dose + D1` is a
/// moving boundary in `D1`; the rate-off saltation carries its derivative (the
/// sign-mirror of the lagtime dose-start saltation). `D1` is η-coupled here
/// (`D1 = TVD1·exp(ETA_D1)`) so both the θ-block (`df_dtheta[TVD1]`) and the η-block
/// (`df_deta[ETA_D1]`) of the moving-boundary term are FD-checked against the production
/// predictor, which resolves `D1` per evaluation. Observations straddle the window end
/// (window `[0, 5]` at `D1=5`): 1, 3 inside; 6, 10 after — the post-window obs are the
/// ones the moving boundary moves.
#[test]
fn ode_provider_modeled_duration_matches_production() {
    const MODELED_DUR_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(MODELED_DUR_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::modeled(
        0.0,
        100.0,
        1,
        false,
        0.0,
        crate::types::RateMode::ModeledDuration,
    )];
    // A modeled dose declines the static superposition walk and routes to the
    // event-driven walk (where the moving-boundary saltation lives).
    assert!(
        ode_tvcov_supported(&model, &subject) && !ode_subject_supported(&model, &subject),
        "a modeled-duration dose must route to the event-driven walk"
    );
    let theta = [5.0, 50.0, 5.0];
    let eta = [0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    // Inner/outer scope parity: the light `Dual1` inner η-gradient shares the same
    // event-driven walk, so it must serve this subject too (never analytic-outer /
    // FD-inner) and its `∂f/∂η` must equal the outer provider's η-block exactly.
    let inner = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner served");
    let outer = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer served");
    for (io, oo) in inner.iter().zip(outer.obs.iter()) {
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                io.df_deta[k],
                oo.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// #530 mirror: a **modeled-rate infusion** (`RATE=-1` → `R1`) — the window length is
/// `F·amt/R1`, a moving boundary in `R1`, carried by the same rate-off saltation. `R1`
/// is η-coupled so both blocks are FD-checked.
#[test]
fn ode_provider_modeled_rate_matches_production() {
    const MODELED_RATE_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1 * exp(ETA_R1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(MODELED_RATE_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::modeled(
        0.0,
        100.0,
        1,
        false,
        0.0,
        crate::types::RateMode::ModeledRate,
    )];
    assert!(
        ode_tvcov_supported(&model, &subject) && !ode_subject_supported(&model, &subject),
        "a modeled-rate dose must route to the event-driven walk"
    );
    check_vs_production(&model, &subject, &[5.0, 50.0, 20.0], &[0.12, -0.08]);
}

/// #530 cross-term: a **modeled-duration dose under an estimated lagtime**. Both the
/// dose *start* (`t_dose + lag`, rate-on, moves with `lag`) and the window *end*
/// (`t_dose + lag + D1`, rate-off, moves with `lag + D1`) are moving boundaries. The
/// event-driven walk carries the start saltation with shift `δlag` and the end saltation
/// with the combined shift `δlag + δD1` (`integrate_tvcov_g` `d_off = dlag + dtinf`,
/// `ode_provider.rs:3475`). Both `D1` and `LAGTIME` are η-coupled so the cross-term's
/// θ- and η-blocks are FD-checked against the production predictor (which resolves the
/// modeled window and the lag per evaluation). This guards the combination the gate
/// admits to the analytic walk — it is *not* an FD fallback (#530 review finding 1).
/// Observations straddle the lagged window end (window `[1, 6]` at `lag=1, D1=5`): 4
/// inside, 7 and 11 after, 0.5 before the lagged dose arrival.
#[test]
fn ode_provider_modeled_duration_with_lagtime_matches_production() {
    const MODELED_DUR_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVLAG(1.0, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  omega ETA_LAG ~ 0.02
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(MODELED_DUR_LAG_ODE).expect("parse");
    let mut subject = bolus_subject(&[0.5, 4.0, 7.0, 11.0]);
    subject.doses = vec![DoseEvent::modeled(
        0.0,
        100.0,
        1,
        false,
        0.0,
        crate::types::RateMode::ModeledDuration,
    )];
    // Modeled dose + lagtime both route to the event-driven walk; the combination must
    // be admitted there (not declined to FD) — the saltation carries the cross-term.
    assert!(
        model.has_lagtime()
            && ode_tvcov_supported(&model, &subject)
            && !ode_subject_supported(&model, &subject),
        "modeled-duration + lagtime must ride the event-driven walk analytically"
    );
    let theta = [5.0, 50.0, 5.0, 1.0];
    let eta = [0.12, -0.08, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    // Inner/outer scope parity on the same walk.
    let inner = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner served");
    let outer = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer served");
    for (io, oo) in inner.iter().zip(outer.obs.iter()) {
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                io.df_deta[k],
                oo.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// #530: **zero-order absorption** (`zero_order(dur)`) gets analytic sensitivities on the
/// static walk. The forcing delivers `F·amt/dur` over `[t_dose, t_dose+dur]`; the window
/// end is a moving boundary in `dur`, carried by the rate-off saltation injected at the
/// cutoff (`integrate_g`), exactly as a modeled-duration infusion. `DUR` is η-coupled
/// (`DUR = TVDUR·exp(ETA_DUR)`) so both the θ-block (`df_dtheta[TVDUR]`) and the η-block
/// (`df_deta[ETA_DUR]`) of the moving-boundary term are FD-checked against the production
/// predictor. A bolus into `central` (the forcing's compartment) feeds `R_in`, not a
/// bolus. Observations straddle the window end (window `[0, 4]` at `DUR=4`).
#[test]
fn ode_provider_zero_order_absorption_matches_production() {
    const ZERO_ORDER_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(ZERO_ORDER_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    // A plain bolus into the forcing's compartment feeds R_in (not a bolus).
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(
        ode_subject_supported(&model, &subject),
        "a zero-order model must be served analytically on the static walk (#530)"
    );
    let theta = [5.0, 50.0, 4.0];
    let eta = [0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    // Inner/outer scope parity: the `Dual1` inner η-gradient shares `integrate_g`, so it
    // must serve this subject too and its `∂f/∂η` must equal the outer provider's η-block.
    let inner = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner served");
    let outer = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer");
    for (io, oo) in inner.iter().zip(outer.obs.iter()) {
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                io.df_deta[k],
                oo.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// #530 finding 2: **multi-dose `zero_order(dur)`**. The global `ZeroOrder` Dual2 lift
/// serves more than the single-dose case the first zero-order test covers — every dose
/// opens its own constant-rate window with its own moving end. Two doses (t=0 and t=12)
/// each deliver `F·amt/DUR` over their own `[t_dose, t_dose+DUR]` window; the second
/// window's saltation must fire independently. `DUR` is η-coupled. Observations straddle
/// both window ends (`DUR=4`: windows `[0,4]` and `[12,16]`; obs 3, 6 around the first,
/// 14, 18 around the second).
#[test]
fn ode_provider_zero_order_multi_dose_matches_production() {
    const ZERO_ORDER_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(ZERO_ORDER_ODE).expect("parse");
    let mut subject = bolus_subject(&[3.0, 6.0, 14.0, 18.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    assert!(
        ode_subject_supported(&model, &subject),
        "multi-dose zero-order must be served analytically on the static walk"
    );
    check_vs_production(&model, &subject, &[5.0, 50.0, 4.0], &[0.12, -0.08]);
}

/// #530 finding 2: **mixed (zero-order + first-order) absorption**. `FZO*zero_order(dur=DUR)`
/// `+ FZO1*first_order(ka=KA)` feeds one compartment; the docs claim the whole `mixed`
/// family is now analytic (the zero-order `∂/∂DUR` boundary term carried by the saltation,
/// the first-order pathway smooth). This checks the composite against production FD with
/// `DUR` η-coupled — the saltation must fire for the zero-order pathway while the
/// first-order forcing rides the smooth dual path alongside it.
#[test]
fn ode_provider_mixed_absorption_matches_production() {
    const MIXED_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(3.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(MIXED_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(
        ode_subject_supported(&model, &subject),
        "mixed zero+first-order absorption must be served analytically (#530)"
    );
    check_vs_production(
        &model,
        &subject,
        &[5.0, 50.0, 0.4, 1.0, 3.0],
        &[0.12, -0.08],
    );
}

/// EVID 3/4 reset: a two-occasion subject (reset + re-dose at t=10) must zero
/// the dual state at the reset and match the production event-driven path
/// across both occasions.
#[test]
fn ode_provider_2cpt_reset_matches_production() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![10.0];
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

/// **Static walk: infusion straddling an EVID 3/4 reset.** A plain subject (no TV-cov,
/// no lagtime) with an infusion window straddling a reset routes to the *static*
/// `integrate_g` walk via `ode_subject_supported` — whose `active_inf` must drop the
/// pre-reset infusion afterward (`reset_floor`), else `F·rate` leaks into the post-reset
/// segments. The PR fixed the event-driven twin; this guards the static path so a future
/// edit can't silently reintroduce the pre-PR leak (#472 review round 2 #1). The
/// post-reset observations (3, 4, 8) are the ones that catch it.
#[test]
fn ode_provider_static_infusion_reset_matches_production() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 4.0, 8.0]);
    subject.doses = vec![
        // Infusion rate 200, amt 1000 → 5 h window [0, 5], straddling the reset at t=2.
        DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
        // EVID=4 re-dose (bolus) at the reset.
        DoseEvent::new(2.0, 1000.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![2.0];
    assert!(subject.doses[0].is_infusion() && subject.has_resets());
    // Plain subject → the static `integrate_g` path, NOT the event-driven walk.
    assert!(
        !ode_tvcov_supported(&model, &subject) && ode_subject_supported(&model, &subject),
        "infusion+reset with no TV-cov/lagtime must route to the static integrate_g walk"
    );
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

/// **Time-varying covariates + EVID 3/4 reset.** A TV-cov subject with a reset +
/// re-dose routes to the event-driven walk, which must zero the dual state at the
/// reset and match production across the reset boundary (#439 reset).
#[test]
fn ode_provider_tvcov_reset_matches_production() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(5.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.dose_covariates = vec![wt(60.0), wt(75.0)];
    subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(80.0), wt(85.0)];
    subject.reset_times = vec![5.0];
    assert!(subject.has_tv_covariates() && subject.has_resets());
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
}

/// **Estimated lagtime + EVID 3/4 reset.** Lagtime routes to the event-driven walk;
/// the reset (fixed time) zeros the dual state, and the post-reset re-dose's lagtime
/// saltation lands on it. Full `SubjectSens` vs production FD (#439 lagtime × reset).
#[test]
fn ode_provider_lagtime_reset_matches_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_ODE).expect("parse oral lag ODE");
    let mut subject = bolus_subject(&[1.0, 6.0, 12.0, 25.0, 30.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![24.0];
    assert!(model.has_lagtime() && subject.has_resets());
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &[1.0, 10.0, 1.0, 0.5], &[0.12, -0.08]);
}

/// **Time-varying covariates + infusion.** A TV-cov subject with a finite-duration
/// infusion (`rate>0`, window `[0, amt/rate]`) routes to the event-driven walk, which
/// must apply the `F·rate` forcing over the in-window segments and match production
/// (#439 infusion).
#[test]
fn ode_provider_tvcov_infusion_matches_production() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    // Infusion into cmt 1: rate 50, amt 100 → 2 h window [0, 2]; obs 1 is in-window.
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)];
    subject.dose_covariates = vec![wt(70.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
    assert!(subject.doses[0].is_infusion() && subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
}

/// **Infusion straddling an EVID 3/4 reset.** An infusion window `[0, 4]` crossing a
/// reset at `t=2` must stop contributing after the reset (the reset zeroes the state and
/// turns the infusion off — production's `reset_floor`). If the dual walk kept adding
/// `F·rate` to the post-reset segments, the *prediction* (not just the gradient) would
/// diverge — the dominant defect this guards (#472 review #1). Post-reset obs (3, 6, 9)
/// are the ones that catch it; validated vs production FD.
#[test]
fn ode_provider_tvcov_infusion_reset_matches_production() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
    subject.doses = vec![
        // Infusion rate 50, amt 200 → 4 h window [0, 4], straddling the reset at t=2.
        DoseEvent::new(0.0, 200.0, 1, 50.0, false, 0.0),
        // EVID=4 re-dose (bolus) at the reset.
        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![2.0];
    subject.dose_covariates = vec![wt(70.0), wt(70.0)];
    subject.obs_covariates = vec![wt(60.0), wt(75.0), wt(80.0), wt(85.0)];
    assert!(subject.doses[0].is_infusion() && subject.has_resets());
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
}

/// **Estimated lagtime + infusion + reset.** Combines the moving infusion window with a
/// reset that cuts it off: after the reset the rate-off saltation at the window end must
/// *not* fire (the infusion was already stopped), and a post-reset re-dose infusion has
/// its own (lagged) window. Full `SubjectSens` vs production FD (#472 review #2).
#[test]
fn ode_provider_lagtime_infusion_reset_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
    let mut subject = bolus_subject(&[2.0, 4.0, 8.0, 12.0]);
    subject.doses = vec![
        // Lagged infusion (rate 50, amt 200 → 4 h window) straddling the reset at 5.
        DoseEvent::new(0.0, 200.0, 1, 50.0, false, 0.0),
        // Post-reset re-dose infusion.
        DoseEvent::new(5.0, 100.0, 1, 40.0, false, 0.0),
    ];
    subject.reset_times = vec![5.0];
    assert!(subject.doses[0].is_infusion() && subject.has_resets() && model.has_lagtime());
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
}

/// **#419: rate-defined infusion under bioavailability `F ≠ 1`.** NONMEM holds the rate
/// and scales the *window* to `F·amt/rate`, so `F`'s sensitivity is a moving rate-off
/// boundary (not a rate-magnitude scale). The subject routes to the event-driven walk
/// (`has_rate_defined_under_f`), which carries it via the rate-off saltation with
/// `δ = δt_inf`. Validated vs production FD, with `F` on IIV (`ETA_F`) for the 2nd order.
const ONECPT_IV_F_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF * exp(ETA_F)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_provider_rate_defined_infusion_under_f_matches_production() {
    let model = parse_model_string(ONECPT_IV_F_ODE).expect("parse F+inf ODE");
    assert!(model.has_bioavailability());
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    // Rate-defined infusion (rate 40): under F≈0.7 the window is F·100/40 = 1.75 h.
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
    assert!(subject.doses[0].is_infusion() && subject.has_rate_defined_infusion());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "rate-defined infusion under F → event-driven walk (#419)"
    );
    // η = [ETA_CL, ETA_F]; θ = [TVCL, TVV, TVF].
    check_vs_production(&model, &subject, &[1.0, 10.0, 0.7], &[0.1, 0.05]);
    // 2nd order: the rate-off-under-F `coef2 = -s·½·J·(Δr·e_cmt)` saltation block (#473
    // review #2 — the F-window-shift Hessian was previously unvalidated).
    check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.7], &[0.1, 0.05]);
}

/// Rate-defined infusion under `F` combined with **estimated lagtime**: the rate-on
/// boundary shifts with `lag`, the rate-off boundary with `lag` *and* `F` (combined
/// `δ = δlag + δt_inf`). Validated vs production FD (#419 × lagtime).
#[test]
fn ode_provider_rate_defined_infusion_under_f_with_lag_matches_production() {
    const ONECPT_IV_F_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVF(0.7, 0.05, 1.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(ONECPT_IV_F_LAG_ODE).expect("parse F+lag+inf ODE");
    assert!(model.has_bioavailability() && model.has_lagtime());
    let mut subject = bolus_subject(&[2.0, 4.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
    assert!(ode_tvcov_supported(&model, &subject));
    // θ = [TVCL, TVV, TVF, TVLAG].
    check_vs_production(&model, &subject, &[1.0, 10.0, 0.7, 0.5], &[0.1]);
}

/// **Estimated lagtime + infusion.** The infusion *window* `[t+lag, t+lag+dur]` shifts
/// with `lag`, so the lagtime sensitivity is the event-time saltation at **both** rate
/// boundaries (rate-on and rate-off). Full `SubjectSens` vs production FD, with lag on
/// IIV (`ETA_LAG`) to exercise the 2nd-order rate-boundary term (#439 lagtime × infusion).
const ONECPT_IV_LAG_INF_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// Steady-state × a built-in absorption **input-rate forcing** (smooth density kernel) is now
/// **analytic** on the event-driven `ode_tvcov_supported` walk (#835 — the dual carries the
/// closed-form SS trough `u_ss = (I − M)⁻¹·b`). It still declines the *static* superposition
/// walk (`ode_subject_supported`): any SS dose needs the event-driven equilibration, so the
/// routing (event-driven, never static) is unchanged — only the event-driven walk flipped from
/// the FD fallback to analytic. (SS into a `zero_order` window, or combined with an absorption
/// lagtime, stays on FD and is rejected upstream.)
#[test]
fn ode_gates_ss_into_smooth_kernel_analytic_on_event_driven_walk() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(0.5, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(M).expect("parse first_order forcing");
    assert!(
        !model.ode_spec.as_ref().unwrap().input_rate.is_empty(),
        "model must carry an input-rate forcing"
    );
    let mut subject = bolus_subject(&[1.0, 4.0, 8.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 8.0)]; // SS=1, II=8
    assert!(subject.has_periodic_ss_dose());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "#835: SS × a smooth absorption kernel is analytic on the event-driven walk"
    );
    assert!(
        !ode_subject_supported(&model, &subject),
        "SS still declines the static superposition walk (needs event-driven equilibration)"
    );
}

/// #835 belt-and-suspenders: SS into a `zero_order` window is hard-rejected upstream
/// (`E_ABSORPTION_SS_ZERO_ORDER`), so it never reaches a real fit's gate — but the non-IOV
/// `ode_tvcov_supported` gate must *independently* decline it too, so the analytic walk can't
/// silently serve the (unbuilt) periodic-window trough if that upstream check is ever relaxed.
/// (The IOV twin is covered by `ode_iov_zero_order_ss_falls_back_to_fd`.)
#[test]
fn ode_gates_ss_into_zero_order_window_declines_dual_walk() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVDUR(2.0, 0.05, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(M).expect("parse SS zero_order");
    let mut subject = bolus_subject(&[1.0, 4.0, 7.9]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 8.0)]; // SS=1, II=8
    assert!(subject.has_periodic_ss_dose());
    assert!(
        !ode_tvcov_supported(&model, &subject),
        "#835: SS into a zero_order window must decline the dual walk (no periodic-window trough)"
    );
}

/// **Steady-state (SS=1) bolus dose.** The dual SS-equilibration loads the
/// infinite-past pulse-train trough (carrying `∂SS/∂(θ,η)`) at the SS dose, then the
/// dose's own pulse applies. Validated `f`/`∂f/∂η`/`∂f/∂θ` vs the production predictor
/// (which equilibrates the same way) + FD (#439 Tier 2 steady state).
const ONECPT_IV_SS_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
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
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_provider_ss_bolus_matches_production() {
    let model = parse_model_string(ONECPT_IV_SS_ODE).expect("parse SS ODE");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
    // SS=1 bolus into central, II = 12 h.
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS bolus → event-driven walk"
    );
    check_vs_production(&model, &subject, &[1.0, 10.0], &[0.1]);
    // 2nd order: the SS equilibration's `∂²(SS state)/∂(θ,η)²` (#473 review #2 — the SS
    // dual-equilibration Hessian was previously unvalidated).
    check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0], &[0.1]);
}

/// **Linear bolus SS now uses the exact closed-form fixed point over the dual (#914).** A
/// scale-separated 2-cpt linear disposition — small `V2` puts the peripheral compartment ~50×
/// below central — equilibrates via `periodic_ss_fixed_point_g` (one recorded cycle) rather than
/// the up-to-50-cycle pulse train, so its dual value + `∂/∂η` + `∂/∂θ` + both Hessian blocks must
/// match the production predictor and its finite differences. (This test previously compared an
/// early-stopped run against a forced-full-budget run; that comparison no longer applies to a
/// linear model — the #519 pulse-train early stop is now nonlinear-only, covered by the
/// input-rate MM tests `ode_provider_ss_absorption_nonlinear_*`.)
#[test]
fn ode_provider_ss_linear_bolus_uses_exact_solve() {
    let model = parse_model_string(TWOCPT_ODE).expect("parse 2-cpt SS ODE");
    let mut subject = bolus_subject(&[1.0, 4.0, 8.0, 11.0, 20.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS bolus → analytic walk"
    );
    let theta = [4.0, 50.0, 8.0, 1.0];
    let eta = [0.1, 0.05];

    // Exact fixed point: one recorded cycle, not the pulse train.
    let _ = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    assert_eq!(
        crate::dosing::last_ss_equilibration_cycles(),
        1,
        "a linear disposition must equilibrate via the closed-form fixed point (#914)"
    );

    // Value + ∂/∂η + ∂/∂θ vs the production predictor, and both Hessian blocks vs FD-of-gradient
    // — the Dual2-vs-FD parity CLAUDE.md requires for the new exact-solve sensitivity path.
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

// 1-cpt IV with a **TAD-dependent RHS** (`-(CL/V)·central·(1+0.02·TAD)`). Used to
// verify the #473 review (13:22) finding #1: does an SS dose + a `TAD`-referencing RHS
// diverge from production for observations beyond one `II`?
const TAD_SS_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.02 * TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **SS=1 dose with a `TAD`-dependent (non-autonomous) RHS routes to FD** (#473 review #1).
/// The SS dual equilibration expands a *time-invariant* pulse train (cycle-relative time),
/// so a `TAD`/`TIME`-dependent RHS breaks the steady-state cycle recurrence — the analytic
/// walk was verified to diverge ~40× from the production predictor on this model — so the
/// gate must decline it. (A non-SS `TAD` RHS is fine — the TV-cov/static walks anchor TAD
/// correctly; see `ode_provider_tvcov_tad_dependent_rhs_matches_production`.)
#[test]
fn ode_provider_ss_tad_dependent_rhs_routes_to_fd() {
    let model = parse_model_string(TAD_SS_ODE).expect("parse TAD SS ODE");
    let mut subject = bolus_subject(&[2.0, 8.0, 20.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    assert!(subject.doses[0].ss);
    assert!(
        model
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .is_some_and(|p| p.uses_time_vars()),
        "TAD_SS_ODE's RHS must read TAD (precondition for the gate)"
    );
    assert!(
        !ode_tvcov_supported(&model, &subject),
        "SS + TAD-dependent RHS must route to FD (#473 review #1)"
    );
}

/// **Steady-state (SS=1) infusion.** Each equilibration cycle runs an active-rate
/// window (`F·rate` forcing) then a quiet decay window; the dual carries `∂SS/∂(θ,η)`
/// through both, and the SS dose's own current-cycle window is applied via the segment
/// forcing. Validated vs production FD (#439 SS infusion).
#[test]
fn ode_provider_ss_infusion_matches_production() {
    let model = parse_model_string(ONECPT_IV_SS_ODE).expect("parse SS ODE");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
    // SS=1 infusion into central: rate 40, amt 100 → 2.5 h window, II = 12 h.
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
    assert!(subject.doses[0].ss && subject.doses[0].is_infusion());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS infusion → event-driven walk"
    );
    check_vs_production(&model, &subject, &[1.0, 10.0], &[0.1]);
}

/// **SS × time-varying covariates.** The SS equilibration uses the SS dose's covariate
/// snapshot, and the post-dose obs read per-event params — both via the event-driven
/// walk. Validated vs production FD (#439 SS composing with TV-cov).
#[test]
fn ode_provider_ss_tvcov_matches_production() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    subject.dose_covariates = vec![wt(70.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
    assert!(subject.doses[0].ss && subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
}

/// **SS × estimated lagtime is now analytic (#486, PR3 sub-case (a)).** The SS dose
/// arrives at `t_dose + lag`, so observations in the pre-arrival window
/// `[t_dose, t_dose+lag)` must read the previous interval's steady-state tail — the
/// walk seeds it via a `K_SS_SEED` timeline break (`ss_state_at_phase_g`, phase
/// `II − lag`), mirroring production's `ss_state_at_phase` at the dose record time. Obs
/// at `0.2` (< lag ≈ 0.5) exercises the pre-arrival seed directly; the rest exercise the
/// dose's own (unmodified) general lagtime saltation and later cycles.
#[test]
fn ode_provider_ss_lagtime_matches_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_ODE).expect("parse oral lag ODE");
    let mut subject = bolus_subject(&[0.2, 1.0, 4.0, 8.0, 13.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    assert!(subject.doses[0].ss && model.has_lagtime());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS + lagtime → event-driven walk (#486)"
    );
    // θ = [TVCL, TVV, TVKA, TVLAG].
    let theta = [1.0, 10.0, 1.0, 0.5];
    let eta = [0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // 2nd order: the pre-arrival seed's `extend_flow_by_dual_duration_g` Taylor term and
    // the dose's own (now g_minus=0) lagtime saltation coefficient are both new — neither
    // is exercised by the existing non-SS lagtime Hessian test.
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// **SS × lagtime × infusion is also analytic now** (same pre-arrival seed, #486). Here
/// `phase = II − lag ≈ 11.5` exceeds the infusion's own duration (2.5 h), so the seed
/// exercises `ss_state_at_phase_g`'s harder branch: crossing the active/quiet boundary
/// (the `inject_rate_saltation` reuse) before extending the quiet leg by the remaining
/// dual duration. Obs at `0.3` (< lag) lands in the pre-arrival window.
#[test]
fn ode_provider_ss_lagtime_infusion_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
    let mut subject = bolus_subject(&[0.3, 3.0, 5.0, 8.0, 11.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
    assert!(subject.doses[0].ss && subject.doses[0].is_infusion() && model.has_lagtime());
    assert!(ode_tvcov_supported(&model, &subject));
    // θ = [TVCL, TVV, TVLAG]; η = [ETA_CL, ETA_LAG].
    let theta = [1.0, 10.0, 0.5];
    let eta = [0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // 2nd order: the rate-on pre-advance (`extend_flow_by_dual_duration_g`, feeding the
    // *unmodified* `inject_rate_saltation`) is new machinery, specific to this combo.
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// **Rate-defined SS infusion under `F ≠ 1` is now analytic (#486, PR3 sub-case (b)).**
/// `equilibrate_ss_state_g` reads the caller's `inf_eff` jet (window `F·duration`, rate
/// held) instead of the raw fixed `dose.duration`, and injects the same per-cycle
/// rate-off saltation the main walk's `K_INF_END` uses.
#[test]
fn ode_provider_ss_rate_defined_infusion_under_f_matches_production() {
    let model = parse_model_string(ONECPT_IV_F_ODE).expect("parse F ODE");
    assert!(model.has_bioavailability());
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0]);
    // SS=1 rate-defined infusion (rate 40) under F≈0.7.
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
    assert!(
        subject.doses[0].ss
            && subject.doses[0].is_infusion()
            && subject.has_rate_defined_infusion()
    );
    assert!(
        ode_tvcov_supported(&model, &subject),
        "rate-defined SS infusion under F → event-driven walk (#486)"
    );
    // η = [ETA_CL, ETA_F]; θ = [TVCL, TVV, TVF].
    let theta = [1.0, 10.0, 0.7];
    let eta = [0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// **Linear rate-defined infusion under `F` now uses the exact fixed point over the dual
/// (#914).** The exact-solve `advance_forced` runs the *same* per-cycle `inject_rate_saltation`
/// for the `F`-scaled active/quiet window the fallback loop did, so the `F`-window saltation's
/// gradient **and** Hessian contribution is validated here — against the production predictor and
/// its finite differences — exactly the concern of the early-stop test this replaces (#486/#642
/// review #2). One recorded cycle confirms the exact path is taken.
#[test]
fn ode_provider_ss_rate_under_f_uses_exact_solve() {
    let model = parse_model_string(ONECPT_IV_F_ODE).expect("parse F ODE");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
    assert!(ode_tvcov_supported(&model, &subject), "SS + F → analytic");
    let theta = [1.0, 10.0, 0.7];
    let eta = [0.1, 0.05];

    let _ = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    assert_eq!(
        crate::dosing::last_ss_equilibration_cycles(),
        1,
        "a linear disposition must equilibrate via the closed-form fixed point (#914)"
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// A **Michaelis–Menten disposition** (plain bolus, no absorption kernel): the #914 exact fixed
/// point declines the nonlinear map and the dual walk falls back to the pulse-train iteration —
/// the sole path that still exercises the #519/#532 dual early stop after #914 (a linear
/// disposition now short-circuits to the closed form). This pins that early stop
/// **gradient-faithful**: an early-stopped run must match a forced-full-budget run to well within
/// FD-validation precision, and the analytic dual must match the production predictor + FD. (The
/// linear analogue this replaces was `ode_provider_ss_early_stop_matches_full_budget`.)
const MM_DISPOSITION_ODE: &str = r#"
[parameters]
  theta TVVMAX(50.0, 1.0, 500.0)
  theta TVKM(30.0, 1.0, 500.0)
  omega ETA_VMAX ~ 0.09
  omega ETA_KM   ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  VMAX = TVVMAX * exp(ETA_VMAX)
  KM   = TVKM   * exp(ETA_KM)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -VMAX * central / (KM + central)
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_provider_ss_nonlinear_fallback_early_stop_matches_full_budget() {
    let model = parse_model_string(MM_DISPOSITION_ODE).expect("parse MM disposition ODE");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 7.9]);
    // Peak amount ≈ 100 ≫ KM ≈ 30 (genuinely nonlinear), mean input 100/8 = 12.5 ≪ VMAX = 50
    // (admits a steady state and converges well inside the 50-cycle cap, so the early stop fires).
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 8.0)];
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS bolus → analytic walk"
    );
    let theta = [50.0, 30.0];
    let eta = [0.1, 0.05];

    let early = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    let early_cycles = crate::dosing::last_ss_equilibration_cycles();
    let full = crate::dosing::with_full_ss_equilibration(|| {
        ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported")
    });
    let full_cycles = crate::dosing::last_ss_equilibration_cycles();

    // The nonlinear fallback (not the exact solve) must run, and its dual early stop must fire —
    // else the comparison is vacuous. `> 1` rules out the single-cycle exact-solve short-circuit.
    assert_eq!(
        full_cycles,
        crate::dosing::SS_EQUILIBRATION_CYCLES,
        "forced-full must run the whole budget"
    );
    assert!(
        (2..full_cycles).contains(&early_cycles),
        "the nonlinear fallback's early stop should run >1 and fewer cycles ({early_cycles}) than \
         the full budget ({full_cycles})"
    );

    // Value reached its fixed point → predictions preserved tightly; the dropped derivative tail
    // is below FD-validation precision.
    for (e, f) in early.obs.iter().zip(&full.obs) {
        approx::assert_relative_eq!(e.f, f.f, max_relative = 1e-9, epsilon = 1e-12);
        for (a, b) in e.df_deta.iter().zip(&f.df_deta) {
            approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
        }
        for (a, b) in e.df_dtheta.iter().zip(&f.df_dtheta) {
            approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
        }
    }
    // Teeth: the fallback dual value + gradient must match the production predictor and its FD.
    check_vs_production(&model, &subject, &theta, &eta);
}

/// **Modeled-duration dose × SS is now analytic (#486, PR3 sub-case (d) — cheapest,
/// implemented first).** `equilibrate_ss_state_g` threads the `inf_eff` jet (rebuilt
/// from the `D1` PK slot, exactly as the non-SS modeled-duration walk does) into its
/// per-cycle active/quiet split and rate-off saltation. Observations straddle the SS
/// dose's own (post-equilibration) window end at `D1=5`.
#[test]
fn ode_provider_ss_modeled_duration_matches_production() {
    const MODELED_DUR_SS_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(MODELED_DUR_SS_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::modeled(
        0.0,
        100.0,
        1,
        true,
        12.0,
        crate::types::RateMode::ModeledDuration,
    )];
    assert!(subject.doses[0].ss && !subject.doses[0].is_fixed());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "modeled-duration × SS → event-driven walk (#486)"
    );
    let theta = [5.0, 50.0, 5.0];
    let eta = [0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

#[test]
fn ode_provider_lagtime_infusion_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
    assert!(model.has_lagtime());
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0, 12.0]);
    // Infusion into central: rate 40, amt 100 → 2.5 h window, shifted by the lagtime.
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
    assert!(subject.doses[0].is_infusion());
    assert!(ode_tvcov_supported(&model, &subject));
    // η = [ETA_CL, ETA_LAG]; θ = [TVCL, TVV, TVLAG].
    check_vs_production(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
}

/// Covariate models: the provider must fold the subject's covariate-adjusted
/// typical values (here WT on CL/V1) into both `f` and `∂f/∂θ`. Validated
/// against the production predictor, which folds WT the same way.
// Same 2-cpt ODE as `TWOCPT_ODE`, but the individual parameters are declared
// with an IIV-free parameter *first* (Q, then CL, then V2, then V1). This forces
// the mixed-order axis permutation to be non-trivial (`axis_of != identity`):
// the IIV-bearing CL/V1 must be relocated to the leading dual axes 0/1.
const TWOCPT_ODE_REORDER: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  Q  = TVQ
  CL = TVCL * exp(ETA_CL)
  V2 = TVV2
  V1 = TVV1 * exp(ETA_V1)
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// The mixed-order dual (`run_subject_mixed`, dropping the IIV-free Hessian
/// block) must reproduce the full `Dual2` provider (`run_subject`) on every
/// `ObsSens` field. Covers both the identity-permutation case (`TWOCPT_ODE`:
/// CL/V1 declared first) and a non-trivial permutation (`TWOCPT_ODE_REORDER`:
/// CL/V1 relocated to the leading axes). Issue #445.
#[test]
fn ode_mixed_matches_full_dual2() {
    for src in [TWOCPT_ODE, TWOCPT_ODE_REORDER] {
        let model = parse_model_string(src).expect("parse");
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];

        // Full reference: force the 4-axis `Dual2` path directly.
        let pd = param_derivatives(&model, &subject, &theta, &eta).expect("pd");
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
        let full = run_subject::<4>(&model, &subject, &theta, &eta, &pk.values, &pd).expect("full");
        // Mixed path via the dispatcher: na = 2 (CL, V1) < n = 4 routes to
        // `run_subject_mixed::<2, 4>`, dropping the Q/V2 Hessian block.
        let mixed = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("mixed");

        assert_eq!(full.obs.len(), mixed.obs.len());
        let close = |a: &[f64], b: &[f64], what: &str| {
            assert_eq!(a.len(), b.len(), "{what} length");
            for (x, y) in a.iter().zip(b) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-9, epsilon = 1e-12);
            }
        };
        for (fo, mo) in full.obs.iter().zip(&mixed.obs) {
            approx::assert_relative_eq!(fo.f, mo.f, max_relative = 1e-12);
            close(&fo.df_deta, &mo.df_deta, "df_deta");
            close(&fo.d2f_deta2, &mo.d2f_deta2, "d2f_deta2");
            close(&fo.df_dtheta, &mo.df_dtheta, "df_dtheta");
            close(&fo.d2f_deta_dtheta, &mo.d2f_deta_dtheta, "d2f_deta_dtheta");
        }
    }
}

/// The mixed-order dual must reproduce the full `Dual2` provider for an
/// **`init(...)`-bearing** model routed through the mixed path (`na < n`), which
/// exercises the axis-mapped FD initial-state seeding in `dual_init_state` — the
/// one new numerical path the identity/reorder parity test above never hits (its
/// models have no `init` block) (#445 review #1). Here only `CL` carries IIV, so
/// `na = 1 < n = 3` (`V` and `KA` are IIV-free): `init(central) = 1000/V` seeds an
/// IIV-free gradient-only axis, and the two IIV-free parameters exercise both the
/// skipped diagonal (#448 review #8) and the both-axes-dropped cross-term
/// `continue` in the second-order FD seeding (#448 review #4).
#[test]
fn ode_mixed_init_matches_full_dual2() {
    const INIT_ODE_MIXED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.0, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  init(central) = 1000.0 / V
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(INIT_ODE_MIXED).expect("parse");
    assert_eq!(model.n_eta, 1);
    assert_eq!(model.pk_indices.len(), 3);
    let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    subject.doses = vec![]; // no dose; the `init(...)` baseline is the sole input.
    let theta = vec![1.0, 20.0, 1.0];
    let eta = vec![0.1];

    let pd = param_derivatives(&model, &subject, &theta, &eta).expect("pd");
    let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
    let full = run_subject::<3>(&model, &subject, &theta, &eta, &pk.values, &pd).expect("full");
    // na = 1 (CL) < n = 3 → run_subject_mixed::<1, 3> (2 IIV-free params).
    let mixed = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("mixed");

    assert_eq!(full.obs.len(), mixed.obs.len());
    for (fo, mo) in full.obs.iter().zip(&mixed.obs) {
        approx::assert_relative_eq!(fo.f, mo.f, max_relative = 1e-12, epsilon = 1e-12);
        for (a, b) in fo.df_deta.iter().zip(&mo.df_deta) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
        }
        for (a, b) in fo.d2f_deta2.iter().zip(&mo.d2f_deta2) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
        }
        for (a, b) in fo.df_dtheta.iter().zip(&mo.df_dtheta) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
        }
        for (a, b) in fo.d2f_deta_dtheta.iter().zip(&mo.d2f_deta_dtheta) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
        }
    }
}

/// Micro-benchmark: outer-gradient sensitivities via the full `Dual2<4>` path
/// vs the mixed `DualMixed<2, 4>` path (Q/V2 Hessian block dropped) on the
/// 2-cpt ODE. Reports ns/call and the speedup. Run with
/// `cargo test --release -- --ignored --nocapture bench_mixed_vs_full`.
#[test]
#[ignore = "micro-benchmark; run with --release --ignored --nocapture"]
fn bench_mixed_vs_full() {
    use std::hint::black_box;
    use std::time::Instant;
    let model = parse_model_string(TWOCPT_ODE).expect("parse");
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];
    let pd = param_derivatives(&model, &subject, &theta, &eta).expect("pd");
    let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
    let iiv = vec![0usize, 1]; // CL, V1
    let axis_of = vec![0usize, 1, 2, 3]; // identity (IIV declared first)

    let iters = 50_000;
    // Warm up.
    for _ in 0..1000 {
        black_box(run_subject::<4>(
            &model, &subject, &theta, &eta, &pk.values, &pd,
        ));
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(run_subject::<4>(
            &model, &subject, &theta, &eta, &pk.values, &pd,
        ));
    }
    let full = t0.elapsed();
    let t1 = Instant::now();
    for _ in 0..iters {
        black_box(run_subject_mixed::<2, 4>(
            &model, &subject, &theta, &eta, &pk.values, &pd, &axis_of, &iiv,
        ));
    }
    let mixed = t1.elapsed();
    let fns = full.as_nanos() as f64 / iters as f64;
    let mns = mixed.as_nanos() as f64 / iters as f64;
    eprintln!("full  Dual2<4>      = {fns:8.1} ns/call");
    eprintln!("mixed DualMixed<2,4> = {mns:8.1} ns/call");
    eprintln!(
        "speedup = {:.2}x  ({:.0}% faster)",
        fns / mns,
        100.0 * (fns - mns) / fns
    );
}

#[test]
fn ode_provider_2cpt_covariate_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_COV).expect("parse");
    assert!(ode_analytical_supported(&model));
    // A subject whose weight differs from the 70 kg reference, so the
    // covariate genuinely shifts CL/V1 and their θ-Jacobian.
    let subject = bolus_subject_wt(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0], 90.0);
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];

    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(&model, &subject, th, e)[j]
    };
    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            obs.f,
            pred(&eta, &theta, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
        }
        for m in 0..model.n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 1e-3, epsilon = 1e-6);
        }
    }
}

// Per-CMT Form-C readout (#439): a 2-cpt model observed at two endpoints —
// central concentration at CMT 1 (`central/V1`) and peripheral concentration
// at CMT 2 (`peripheral/V2`). Each observation reads its own CMT's output
// program over the dual state, selected by `subject.obs_cmts`.
const TWOCPT_ODE_PERCMT: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y[CMT=1] = central / V1
  y[CMT=2] = peripheral / V2
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

fn percmt_subject(times: &[f64], cmts: &[usize]) -> Subject {
    let mut s = bolus_subject(times);
    s.obs_cmts = cmts.to_vec();
    s
}

/// The per-CMT provider's `f`/`∂f/∂η`/`∂f/∂θ` must match the production predictor
/// + FD, with each observation routed through its CMT's output program.
#[test]
fn ode_provider_percmt_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "per-CMT Form-C ODE readout should be supported (#439)"
    );
    // Observations alternate between the two endpoints.
    let subject = percmt_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0], &[1, 2, 1, 2, 1, 2]);
    check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
}

/// The `PerCmt` gate's negative branches must drop the model out of analytic
/// scope (→ FD), not silently admit it: an empty endpoint map, or an endpoint
/// whose `program` is `None` (hand-constructed / non-`is_dual_evaluable`, which the dual
/// provider can't evaluate) (#446 review — patch coverage on the reject path).
#[test]
fn ode_provider_percmt_gate_rejects_incomplete_map() {
    // Empty per-CMT map → declined.
    let mut empty = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
    empty.ode_spec.as_mut().expect("ode").readout =
        OdeReadout::PerCmt(std::collections::HashMap::new());
    assert!(
        !ode_analytical_supported(&empty),
        "empty per-CMT map must decline to FD"
    );

    // One endpoint with `program: None` (keeps its f64 `out_fn`) → declined,
    // since the dual provider has no differentiable program to evaluate.
    let mut no_prog = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
    match &mut no_prog.ode_spec.as_mut().expect("ode").readout {
        OdeReadout::PerCmt(map) => {
            let any_cmt = *map.keys().next().expect("at least one endpoint");
            map.get_mut(&any_cmt).expect("entry").program = None;
        }
        _ => panic!("expected PerCmt readout"),
    }
    assert!(
        !ode_analytical_supported(&no_prog),
        "per-CMT endpoint with no differentiable program must decline to FD"
    );
}

/// An observation whose CMT is absent from the per-CMT map hits the defensive
/// NaN fallback in the readout (fit-time `validate_per_cmt_scaling` rejects this
/// upstream, so it is unreachable in a real fit — but the provider must produce
/// NaN, not panic or silently zero it) (#446 review — patch coverage on the
/// fallback arm, shared by the Dual2 and Dual1 walks via `integrate_subject_duals`).
#[test]
fn ode_provider_percmt_missing_cmt_yields_nan() {
    let model = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
    // CMT 3 has no readout entry (the map covers 1 and 2); the gate still passes
    // (map non-empty, every present program simple).
    let subject = percmt_subject(&[0.5, 1.0, 2.0], &[1, 3, 2]);
    let theta = [4.0, 12.0, 2.0, 25.0];
    let eta = [0.12, -0.08];
    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta)
        .expect("gate passes: map non-empty, programs simple");
    assert!(
        sens.obs[1].f.is_nan(),
        "obs on uncovered CMT 3 → NaN readout"
    );
    assert!(
        sens.obs[0].f.is_finite() && sens.obs[2].f.is_finite(),
        "covered CMTs stay finite"
    );
}

/// The light `Dual1` inner η-gradient must equal the full `Dual2` outer
/// `df_deta` for a per-CMT model too (each endpoint's program over both duals).
#[test]
fn ode_provider_percmt_light_matches_full() {
    let model = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
    let subject = percmt_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0], &[1, 2, 1, 2, 1, 2]);
    let theta = vec![4.0, 12.0, 2.0, 25.0];
    let eta = vec![0.12, -0.08];
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

// Time-varying covariate (#439): WT on CL changes across observations, so the
// PK params vary along the trajectory. The (θ,η)-seeded TV-cov walk must match
// production's event-driven predictor (`ode_predictions_event_driven`) + FD.
const ONECPT_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// #486: a `TIME`-switched CL on a **non-IOV ODE** model. With no TV covariates
/// the subject routes to the event-driven TV walk purely by `uses_time_builtin`
/// (`ode_tvcov_supported`); the full `ode_subject_sensitivities` value/∂η/∂θ (and
/// the 2nd-order blocks) must match central FD of `compute_predictions_with_tv`
/// (which threads the same per-event TIME), and the light inner `ode_subject_eta_grad`
/// must track the outer η-block.
#[test]
fn ode_time_builtin_provider_matches_fd_of_production() {
    const ONECPT_ODE_TIME: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(ONECPT_ODE_TIME).expect("parse ODE TIME");
    assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&model));
    let subject = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME=5
    assert!(!subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TIME ODE subject must route to the event-driven walk"
    );
    let theta = vec![1.0, 0.5, 20.0];
    let eta = vec![0.15, -0.10];

    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    let pred =
        |e: &[f64], th: &[f64], j: usize| compute_predictions_with_tv(&model, &subject, th, e)[j];
    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            obs.f,
            pred(&eta, &theta, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
        }
        for m in 0..model.n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 1e-3, epsilon = 1e-6);
        }
        // 2nd-order blocks via central FD of the analytic first-order gradient.
        let grad_at = |e: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let s = ode_subject_sensitivities(&model, &subject, &theta, e).expect("ode");
            (s.obs[j].df_deta.clone(), s.obs[j].df_dtheta.clone())
        };
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let (de_p, dt_p) = grad_at(&ep);
            let (de_m, dt_m) = grad_at(&em);
            for l in 0..model.n_eta {
                let d2 = (de_p[l] - de_m[l]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * model.n_eta + l],
                    d2,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
            for m in 0..model.n_theta {
                let d2 = (dt_p[m] - dt_m[m]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * model.n_theta + m],
                    d2,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // Inner light Dual1 η-gradient tracks the outer η-block.
    let light = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner");
    assert_eq!(light.len(), sens.obs.len());
    for (o, g) in sens.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-9, epsilon = 1e-9);
        for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-9);
        }
    }
}

/// #486: a `TIME`-switched CL combined with a **built-in `first_order` absorption
/// forcing**. The event-driven walk carries both the per-event `TIME` seeding (#637) and
/// the `R_in` forcing (#643), so this composition is analytic — the old
/// `ode_analytical_supported` decline (which assumed the walk carried no `R_in`) was
/// stale. Value / ∂η / ∂θ + the 2nd-order blocks must match FD of production, and the
/// inner Dual1 must track the outer.
#[test]
fn ode_time_builtin_with_first_order_matches_production() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(M).expect("parse TIME+first_order");
    assert!(
        ode_analytical_supported(&model),
        "TIME + input-rate forcing must be analytic now (#486)"
    );
    let subject = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME = 5
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![1.0, 0.5, 20.0, 1.0];
    let eta = vec![0.15, -0.10];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// #486: a `TIME`-switched CL combined with a non-zero **`init(...)` baseline**. The
/// event-driven walk seeds the `init` state (#662) and threads per-event `TIME` (#637)
/// together, so this composition is analytic — the old decline (assuming the walk seeded
/// compartments at zero) was stale. Value / ∂η / ∂θ + 2nd-order vs FD of production.
#[test]
fn ode_time_builtin_with_init_matches_production() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 1000.0 / V
  d/dt(central) = -(CL/V)*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(M).expect("parse TIME+init");
    assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
    assert!(
        ode_analytical_supported(&model),
        "TIME + init(...) must be analytic now (#486)"
    );
    let mut subject = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME = 5
    subject.doses = vec![]; // the init baseline is the sole input
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![1.0, 0.5, 20.0];
    let eta = vec![0.15, -0.10];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// #486 (route-completeness): the **three-way** composition of a `TIME`-switched CL,
/// a built-in `first_order` absorption **input-rate forcing**, AND an estimated
/// compartment-indexed **lagtime** (`ALAG1`). Each pair is covered elsewhere
/// (`ode_time_builtin_with_first_order_matches_production` = TIME + `R_in`;
/// `ode_provider_first_order_with_alag_matches_production` = `R_in` + lag), but the
/// gate removed in this PR newly admits all three at once, so the per-event `TIME`
/// PK-param seeding (#637), the `R_in` forcing (#643), and the lagged rate-on
/// saltation `Δr = dose·ka` at the moving arrival `t_dose + lag` must compose. If they
/// did not, `ode_subject_sensitivities` would return `Some` with a silently wrong
/// gradient (no FD fallback), so value/∂η/∂θ + the 2nd-order blocks are checked against
/// FD of production and the inner `Dual1` against the outer `Dual2`.
#[test]
fn ode_time_builtin_with_first_order_and_lagtime_matches_production() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V     = TVV
  KA    = TVKA * exp(ETA_KA)
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(M).expect("parse TIME+first_order+lagtime");
    assert!(model.has_lagtime());
    assert!(
        ode_analytical_supported(&model),
        "TIME + input-rate + lagtime must be analytic now (#486)"
    );
    // Obs deliberately straddle both the `TIME = 5` switch and the `t_dose + TVLAG`
    // arrival, while avoiding landing exactly on `t_dose + TVLAG` (a central-FD kink
    // in `lag`, see `ode_provider_first_order_with_alag_matches_production`).
    let subject = bolus_subject(&[0.1, 0.2, 0.4, 3.0, 8.0, 16.0, 30.0]);
    assert!(!ode_subject_supported(&model, &subject));
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![1.0, 0.5, 20.0, 1.0, 0.3];
    let eta = vec![0.15, -0.10];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// #486 (newly-analytic composed cell): `TIME`-switched CL + a non-zero `init(...)`
/// baseline + a **steady-state** dose. Unlike SS + input-rate (declined above), SS +
/// `init` IS supported by production: `equilibrate_ss_state` seeds the SS trough from
/// zero and overwrites the state at the SS dose (discarding `init` there), while `init`
/// still governs the **pre-SS-dose** segments — so the walk must reproduce both the
/// decaying-`init` pre-dose predictions (crossing the `TIME` switch) and the post-dose
/// steady state. This composition was FD before this PR (the removed gate) and is
/// **computed** by the walk now, so it is the one genuinely-new cell that could be
/// silently wrong; value/∂η/∂θ + 2nd-order must match FD of production.
#[test]
fn ode_time_builtin_with_init_and_ss_matches_production() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central)  = 1000.0 / V
  d/dt(central) = -(CL/V)*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(M).expect("parse TIME+init+SS");
    assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
    assert!(ode_analytical_supported(&model));
    // Pre-SS-dose obs (2, 4) decay the `init` baseline and straddle TIME = 5; the SS
    // bolus lands at t = 8 (II = 12), and later obs (10, 14, 20) read the steady state.
    let mut subject = bolus_subject(&[2.0, 4.0, 10.0, 14.0, 20.0]);
    subject.doses = vec![DoseEvent::new(8.0, 100.0, 1, 0.0, true, 12.0)];
    assert!(subject.doses[0].ss && subject.has_periodic_ss_dose());
    assert!(!ode_subject_supported(&model, &subject));
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![1.0, 0.5, 20.0];
    let eta = vec![0.15, -0.10];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// #486: an η-dependent `ExpressionScale` `obs_scale = 1000/V` on the **ODE
/// event-driven walk** — for a `TIME` switch AND a time-varying covariate. The walk
/// now applies the subject-static scale quotient (`apply_expression_scale_outer`)
/// post-walk, the SAME one the static ODE walk uses; value/∂η/∂θ (+2nd order) must
/// match FD of `compute_predictions_with_tv` (which divides by the same scale).
#[test]
fn ode_expression_scale_on_event_walk_matches_production() {
    const ODE_IV_TIME_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    const ODE_IV_TVCOV_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    // TIME switch + obs_scale (routed to the event walk by uses_time_builtin).
    let m_time = parse_model_string(ODE_IV_TIME_SCALED).expect("parse ODE TIME scaled");
    assert!(matches!(
        m_time.scaling,
        ScalingSpec::ExpressionScale { .. }
    ));
    let s_time = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME=5
    assert!(ode_tvcov_supported(&m_time, &s_time));
    check_vs_production(&m_time, &s_time, &[1.0, 0.5, 20.0], &[0.15, -0.10]);
    // Time-varying covariate + obs_scale (the broader gap this closes).
    let m_tv = parse_model_string(ODE_IV_TVCOV_SCALED).expect("parse ODE tvcov scaled");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut s_tv = bolus_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
    s_tv.dose_covariates = vec![wt(60.0)];
    s_tv.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(85.0), wt(90.0)];
    assert!(s_tv.has_tv_covariates() && ode_tvcov_supported(&m_tv, &s_tv));
    check_vs_production(&m_tv, &s_tv, &[1.0, 20.0, 0.75], &[0.15, -0.10]);

    // #637 review #2: the light inner η-gradient (`run_subject_tvcov_eta` →
    // `apply_expression_scale_inner_dispatch`) must track the outer η-block — this
    // exercises the new ODE inner ExpressionScale quotient, which `check_vs_production`
    // (outer only) does not.
    let parity = |m: &CompiledModel, s: &Subject, theta: &[f64], eta: &[f64]| {
        let full = ode_subject_sensitivities(m, s, theta, eta).expect("outer");
        let light = ode_subject_eta_grad(m, s, theta, eta).expect("inner");
        assert_eq!(full.obs.len(), light.len());
        for (o, g) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-12, epsilon = 1e-12);
            for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-10);
            }
        }
    };
    parity(&m_time, &s_time, &[1.0, 0.5, 20.0], &[0.15, -0.10]);
    parity(&m_tv, &s_tv, &[1.0, 20.0, 0.75], &[0.15, -0.10]);
}

#[test]
fn ode_provider_tvcov_matches_production() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    assert_eq!(model.n_theta, 3);
    assert_eq!(model.n_eta, 1); // M = n_theta + n_eta = 4
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
    assert!(subject.has_tv_covariates());
    let theta = vec![1.0, 20.0, 0.75];
    let eta = vec![0.1];

    let sens = run_subject_tvcov::<4>(&model, &subject, &theta, &eta).expect("tvcov supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(&model, &subject, th, e)[j]
    };
    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        // Value matches production's event-driven (per-event-cov) predictor.
        approx::assert_relative_eq!(
            obs.f,
            pred(&eta, &theta, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
        }
        for m in 0..model.n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 1e-3, epsilon = 1e-6);
        }
        // Second order (#449 review #3): the Hessian blocks `d2f_deta2` /
        // `d2f_deta_dtheta` feed FOCEI's `log|H̃|` gradient and covariance, but
        // were untested. Validate them by central-differencing the analytic
        // first-order `df_deta` / `df_dtheta` (themselves checked above) w.r.t. η.
        let grad_at = |e: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let s = run_subject_tvcov::<4>(&model, &subject, &theta, e).expect("tvcov");
            (s.obs[j].df_deta.clone(), s.obs[j].df_dtheta.clone())
        };
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let (de_p, dt_p) = grad_at(&ep);
            let (de_m, dt_m) = grad_at(&em);
            for l in 0..model.n_eta {
                let d2 = (de_p[l] - de_m[l]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * model.n_eta + l],
                    d2,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
            for m in 0..model.n_theta {
                let d2 = (dt_p[m] - dt_m[m]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * model.n_theta + m],
                    d2,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}

fn tvcov_subject() -> Subject {
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
    subject
}

// TV-cov + `init(...)` (#486): a WT covariate on CL makes it a TV-cov subject, and
// `init(central) = BASE / V` seeds an η-dependent (via `V`) and *nonlinear* baseline —
// so the event-driven walk's dual init seed carries `∂/∂(θ,η)` and a non-zero
// second-order block (`∂²(BASE/V)/∂V² ≠ 0`), exercising both Taylor terms of
// `tvcov_init_state`.
const TVCOV_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// TV-cov + `init(...)` on the non-IOV ODE walk (#486): `init(central) = BASE/V` is
/// η-dependent (through `V`) and nonlinear, so the event-driven walk's dual init seed
/// must carry `∂/∂(θ,η)` and its second-order block. The analytic `SubjectSens` (value,
/// `∂f/∂η`, `∂f/∂θ`, + 2nd order) must match FD of the production TV-cov predictor,
/// which seeds `init` at the first-record covariate snapshot (`init_pk`).
#[test]
fn ode_provider_tvcov_init_matches_production() {
    let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
    assert_eq!(model.n_theta, 4);
    assert_eq!(model.n_eta, 2); // M = n_theta + n_eta = 6
    let subject = tvcov_subject();
    assert!(subject.has_tv_covariates());
    assert!(
        subject.reset_times.is_empty(),
        "fixture must have no reset — init + reset stays FD"
    );
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TV-cov + init(...) must be analytic on the non-IOV ODE walk (#486)"
    );
    let theta = vec![1.0, 20.0, 0.75, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// Inner/outer parity for TV-cov + `init(...)` (#486): the light `Dual1` inner
/// η-gradient — which seeds `init` through the SAME `tvcov_init_state` (its second-order
/// term vanishes at `Dual1`) — must equal the full `Dual2` outer `df_deta`. Both walks
/// share `ode_tvcov_supported`, so the init seed moves in lockstep or the inner EBE
/// gradient silently diverges.
#[test]
fn ode_provider_tvcov_init_inner_eta_grad_matches_outer() {
    let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
    let subject = tvcov_subject();
    let theta = vec![1.0, 20.0, 0.75, 500.0];
    let eta = vec![0.1, -0.05];
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// TV-cov + `init(...)` analytic scope (#486): `init(...)` is **fully analytic** on the
/// non-IOV event-driven walk — it composes with a finite infusion, an EVID 3/4 reset, and
/// (via their own model fixtures) lagtime / input-rate / SS / modeled-dose. This pins the
/// two compositions constructible from the plain bolus fixture (infusion, reset), which
/// previously routed to FD; the others are validated by their dedicated
/// `ode_provider_tvcov_init_*_matches_production` tests.
#[test]
fn ode_tvcov_init_compositions_are_analytic() {
    let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
    // Baseline: plain-bolus TV-cov + init is analytic.
    assert!(
        ode_tvcov_supported(&model, &tvcov_subject()),
        "init + bolus is analytic"
    );

    // + finite infusion → analytic (#486): the walk seeds init and adds the infusion
    // forcing on top of the running state.
    let mut s = tvcov_subject();
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0)];
    assert!(
        s.doses[0].is_infusion(),
        "fixture must be a finite infusion"
    );
    assert!(
        ode_tvcov_supported(&model, &s),
        "init + finite infusion is analytic (#486)"
    );

    // + EVID 3/4 reset → analytic (#486): the K_RESET branch re-seeds init at the reset
    // ROW's own snapshot, matching production's `initial_state(&pk_at_reset[idx].values)`
    // (#1133).
    let mut s = tvcov_subject();
    s.reset_times = vec![4.0];
    assert!(
        ode_tvcov_supported(&model, &s),
        "init + EVID 3/4 reset is analytic (#486)"
    );
}

// Tight-tolerance twin of `TVCOV_INIT_ODE` for the infusion branch: the `1e-11`/`1e-13`
// solver tolerance keeps the Dual2 Hessian's value-controlled step error far below the
// `2e-3` FD-of-gradient check. `F = 1` (bare slot), so the infusion is a fixed-boundary
// rate-defined window (`amt/rate`) — the branch-A question (does the forcing accumulate
// correctly on a non-zero init baseline?) without the rate-defined-under-`F` moving
// boundary, which is a separate (#419) mechanism.
const TVCOV_INIT_INF_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

/// **TV-cov + `init(...)` + finite infusion** (#486, branch A). A 5 h IV infusion
/// (amt 100, rate 20, `F = 1`) into the init-seeded `central` compartment, under a WT
/// covariate that makes CL time-varying. The walk seeds `init(central) = BASE/V` and then
/// accumulates the `rate` forcing over the infusion window on top of that non-zero running
/// state — so the analytic `SubjectSens` (value, `∂f/∂(θ,η)`, both Hessian blocks) must
/// match FD of the production TV-cov predictor, and the light inner `Dual1` walk must match
/// the outer `Dual2` walk. Confirms the forcing branch carries no zero-baseline assumption.
#[test]
fn ode_provider_tvcov_init_infusion_matches_production() {
    let model = parse_model_string(TVCOV_INIT_INF_ODE).expect("parse");
    assert_eq!(model.n_theta, 4);
    assert_eq!(model.n_eta, 2);
    let mut subject = tvcov_subject(); // obs at 1,2,4,8; infusion runs [0,5]
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0)];
    assert!(
        subject.doses[0].is_infusion(),
        "fixture must be an infusion"
    );
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TV-cov + init + infusion must be analytic (#486)"
    );
    let theta = vec![1.0, 20.0, 0.75, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

// 1-cpt IV with an estimated lagtime carrying IIV and an `init(central) = BASE/V` baseline.
// The lagged bolus arrives at `t + lag`; before arrival the compartment decays from `init`,
// so the dose-arrival saltation's pre-arrival state `x⁻` is the *non-zero* init baseline.
// `CL` carries IIV (so the init decay — hence `x⁻` and its velocity `g(x⁻)`) depends on
// `ETA_CL`), and `LAGTIME` carries IIV (so the saltation's `∂/∂η_LAG` fires). Tight
// tolerance so the FD-of-predict Hessian check is clean.
const ONECPT_IV_LAG_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVLAG(0.5, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
  BASE    = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = -CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

/// **`init(...)` + estimated lagtime** (#486, branch C). A lagged bolus (arrives at
/// `t + lag ≈ 0.5`) on top of an `init(central) = BASE/V` decaying baseline. The obs at
/// t=0.25 sits in the pre-arrival window (pure init decay), the rest after arrival. The
/// dose-arrival saltation must use the actual pre-arrival state `x⁻ = u` (the init
/// baseline) and its velocity — validated at all orders against FD of the production
/// predictor, plus inner/outer parity. Confirms the saltation makes no zero-baseline
/// assumption.
#[test]
fn ode_provider_tvcov_init_lagtime_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_INIT_ODE).expect("parse");
    assert!(model.has_lagtime());
    let subject = bolus_subject(&[0.25, 1.0, 2.0, 4.0]);
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + lagtime must be analytic on the event-driven walk (#486)"
    );
    let theta = vec![1.0, 20.0, 0.5, 500.0];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **`init(...)` + modeled-duration dose** (#486, branch E). A `RATE=-2` / `D1`-modeled
/// infusion (window `[0, D1]`, moving boundary in `D1 = TVD1·exp(ETA_D1)`) delivering into
/// an `init(central) = BASE/V` compartment. The moving infusion-end saltation reads the
/// live `inf_eff` `D1` jet on top of the init-seeded running state; obs straddle the window
/// end (`D1 = 5`: t=1,3 inside; t=6,10 after). Validated at all orders against FD of the
/// production predictor (which resolves `D1` per evaluation), plus inner/outer parity.
#[test]
fn ode_provider_tvcov_init_modeled_duration_matches_production() {
    const MODELED_DUR_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVBASE(200.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  D1   = TVD1 * exp(ETA_D1)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = -CL/V * central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(MODELED_DUR_INIT_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    subject.doses = vec![DoseEvent::modeled(
        0.0,
        100.0,
        1,
        false,
        0.0,
        crate::types::RateMode::ModeledDuration,
    )];
    assert!(
        ode_tvcov_supported(&model, &subject) && !ode_subject_supported(&model, &subject),
        "init + modeled-duration dose must route to the event-driven walk (#486)"
    );
    let theta = vec![5.0, 50.0, 5.0, 200.0];
    let eta = vec![0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **`init(...)` + built-in `first_order` input-rate forcing + TV-cov** (#486, branch B).
/// A first-order absorption forcing `R_in = KA·(dose remaining)` feeds `central`, which also
/// carries an `init(central) = BASE/V` baseline; a WT covariate on CL makes the subject
/// time-varying (forcing the event-driven walk). `add_prepared_input_rate_forcing` adds
/// `R_in(tad)` to `du/dt` on top of the init-seeded state — validated at all orders against
/// FD of the production predictor, plus inner/outer parity.
#[test]
fn ode_provider_tvcov_init_input_rate_matches_production() {
    const FIRST_ORDER_INIT_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.0, 0.05, 10.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  KA   = TVKA * exp(ETA_KA)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = first_order(ka=KA) - CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(FIRST_ORDER_INIT_TVCOV_ODE).expect("parse");
    assert!(!model.ode_spec.as_ref().unwrap().input_rate.is_empty());
    let subject = tvcov_subject(); // WT covariate → TV-cov; obs at 1,2,4,8
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + input-rate forcing + TV-cov must be analytic (#486)"
    );
    let theta = vec![1.0, 20.0, 1.0, 0.75, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **`init(...)` + `zero_order` absorption + TV-cov** (#486, branch B). `zero_order` was
/// ported to the event-driven walk by #653; with `init` its per-segment constant `active_zero`
/// delivery (`F·amt·frac/DUR` over `[dose, dose+DUR]`, moving end in `DUR`) rides on top of the
/// init-seeded state, and the moving-end rate-off saltation reads that running state. A WT
/// covariate makes the subject TV-cov. Obs straddle the window end (`DUR=4`). Validated at all
/// orders against FD of production, plus inner/outer parity.
#[test]
fn ode_provider_tvcov_init_zero_order_matches_production() {
    const ZERO_ORDER_INIT_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(2000.0, 10.0, 50000.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  DUR  = TVDUR * exp(ETA_DUR)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(ZERO_ORDER_INIT_TVCOV_ODE).expect("parse");
    let mut subject = tvcov_subject(); // WT covariate → TV-cov
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + zero_order + TV-cov must be analytic (#486/#653)"
    );
    let theta = vec![5.0, 50.0, 4.0, 0.75, 2000.0];
    let eta = vec![0.12, -0.08];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **`init(...)` + steady-state dose** (#486, branch D). Production's `equilibrate_ss_state`
/// seeds the SS trough from zero and *overwrites the whole state* at the SS dose — so it
/// discards `init` there, exactly as the dual `equilibrate_ss_state_g` does. `init` therefore
/// only affects the **pre-SS-dose** segments: here the SS bolus is at t=5 (`II=12`), with obs
/// at t=1,2 in the pre-dose window reading the decaying `init(central) = BASE/V` baseline
/// (their `∂/∂(θ,η)` carry the init seed's derivatives) and obs at t=8,14 reading the SS
/// decay (init-independent). The walk seeds `init_state`, decays it from the true integration
/// start (t=0) to t=5, then overwrites at the SS dose — matching production at all orders.
#[test]
fn ode_provider_tvcov_init_ss_matches_production() {
    const ONECPT_IV_SS_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(ONECPT_IV_SS_INIT_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 2.0, 8.0, 14.0]);
    // SS bolus at t=5, II=12: obs at 1,2 are pre-dose (init decay); 8,14 read the SS decay.
    subject.doses = vec![DoseEvent::new(5.0, 100.0, 1, 0.0, true, 12.0)];
    assert!(
        subject.has_periodic_ss_dose(),
        "fixture must be a periodic SS dose"
    );
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + steady-state must be analytic on the event-driven walk (#486)"
    );
    let theta = vec![1.0, 20.0, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **`init(...)` + EVID 3/4 reset** (#486, branch F). Production re-applies
/// `init(&last_pk.values)` at each reset (the `K_RESET` branch re-seeds the state from the
/// reset-event snapshot rather than zeroing). Here an `init(central) = BASE/V` model under a
/// WT covariate (TV-cov, so the reset-time snapshot differs from the first-record one) takes
/// a bolus at t=0, a reset at t=4 (which restores the init baseline), and a second bolus at
/// t=5. Obs at t=2 (pre-reset decay), t=4.5 (post-reset init baseline, pre-2nd-dose), and
/// t=8,12 (2nd-dose decay on the re-seeded baseline). The re-seeded jet must carry
/// `∂init/∂(θ,η)` at the reset snapshot — validated at all orders against FD of production,
/// plus inner/outer parity.
#[test]
fn ode_provider_tvcov_init_reset_matches_production() {
    let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
    // tvcov_subject has a WT covariate (TV-cov) and a bolus at t=0. Add a reset + 2nd dose,
    // and place obs around the reset. obs_covariates must match obs_times length.
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = tvcov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(5.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.dose_covariates = vec![wt(60.0), wt(85.0)];
    subject.obs_times = vec![2.0, 4.5, 8.0, 12.0];
    subject.obs_covariates = vec![wt(65.0), wt(75.0), wt(80.0), wt(90.0)];
    subject.observations = vec![1.0; 4];
    subject.obs_cmts = vec![1; 4];
    subject.cens = vec![0; 4];
    subject.occasions = vec![1; 4];
    subject.reset_times = vec![4.0];
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + reset must be analytic on the event-driven walk (#486)"
    );
    let theta = vec![1.0, 20.0, 0.75, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

// 1-cpt IV with an `init(central) = BASE/V` baseline whose `V` (hence `init`) depends on a
// WT covariate — so the *snapshot* the reset re-seed reads is observable in the gradient.
const ONECPT_IV_INIT_WTV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV * (WT / 70) * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

/// **`init(...)` + EVID=4 reset+dose at `t = 0` (the reset is the first timeline event)**
/// (#486 review — Copilot). Production re-applies `init(&last_pk.values)` at the reset with
/// `last_pk = init_pk` (the first-record snapshot, dose-preferred on ties). The walk seeds
/// `last_params` from the *same* `first_record_pk` selection. Since #1133 the re-seed reads
/// `pk_at_reset[idx]` on both sides instead, and this subject leaves `reset_covariates`
/// empty, so `reset_cov(0)` falls back to `subject.covariates` (`WT = 60`) — the same value
/// the dose@0 snapshot carries, which is why this arm still passes and why it is degenerate
/// for the snapshot question (see the mid-timeline twin below). Here `init = BASE/V` and `V`
/// depends on WT, and the dose@0 covariate (WT=60) differs from the observations' (WT=90),
/// so a wrong snapshot at the re-seed would shift the baseline and its `∂/∂(θ,η)`. Validated
/// against FD of the production predictor, plus inner/outer parity. (Before the fix
/// `last_params` was `pk_at_obs.first()` = the WT=90 obs snapshot → mismatch.)
#[test]
fn ode_provider_init_reset_dose_at_zero_matches_production() {
    let model = parse_model_string(ONECPT_IV_INIT_WTV_ODE).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subject.dose_covariates = vec![wt(60.0)]; // dose@0 snapshot: WT=60
    subject.obs_covariates = vec![wt(90.0), wt(90.0), wt(90.0)]; // obs: WT=90
    subject.covariates = wt(60.0);
    subject.reset_times = vec![0.0]; // EVID=4 reset+dose at t=0 → reset is the first event
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + reset must be analytic (#486)"
    );
    let theta = vec![1.0, 20.0, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

// IOV twin of `ONECPT_IV_INIT_WTV_ODE`: the same WT-on-`V` init baseline, plus a per-occasion
// κ on `V`, so the reset re-seed's snapshot depends on both the covariate row and the occasion
// convention (#1133).
const ONECPT_IV_INIT_WTV_ODE_IOV: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  kappa KAPPA_V ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV * (WT / 70) * exp(ETA_V + KAPPA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

/// **`init(...)` re-seeded at a MID-TIMELINE reset whose own `WT` differs from every
/// neighbouring record** (#1133). The two existing reset twins above cannot see which
/// snapshot the re-seed reads, and it is worth writing down why, because each looks like
/// it should:
///
/// * `ode_provider_tvcov_init_reset_matches_production` is TV-cov and has a mid-timeline
///   reset, but its model is `V = TVV * exp(ETA_V)` — `WT` sits on `CL` only, so
///   `init = BASE/V` is covariate-free and every candidate snapshot seeds the same number.
/// * `ode_provider_init_reset_dose_at_zero_matches_production` *does* put `WT` on `V`, but
///   its reset is the first timeline event, where the reset row, `init_pk` and the dose@0
///   snapshot are all `WT = 60` — three conventions, one value.
///
/// This subject breaks both degeneracies at once: `WT` is on `V` (so the seed moves with
/// it) and the reset sits at `t = 3` carrying `WT = 40`, between an obs at `WT = 90` and
/// an obs at `WT = 150`. Production seeds from the reset row's own snapshot; a twin still
/// carrying `last_params` would seed from the `WT = 90` obs and disagree in `f` — and,
/// because `∂init/∂(θ,η)` is evaluated at the same snapshot, in the gradient too.
///
/// `check_vs_production` is the oracle that bites here: it FDs the production predictor,
/// so a twin that mirrors a stale arm fails on the value comparison before the derivative
/// one. (The reverse — both sides sharing a wrong convention — is the blind spot CLAUDE.md
/// records, and it is why the value path itself is anchored against NONMEM in
/// `tests/reset_init_snapshot_nonmem_anchor.rs` rather than against this twin.)
#[test]
fn ode_provider_init_reset_midtimeline_reads_the_reset_rows_snapshot() {
    let model = parse_model_string(ONECPT_IV_INIT_WTV_ODE).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subject.dose_covariates = vec![wt(60.0)];
    // The record before the reset is the t=2 obs (WT=90); the one after is t=4 (WT=150).
    // The reset row itself carries neither.
    subject.obs_covariates = vec![wt(70.0), wt(90.0), wt(150.0), wt(150.0)];
    subject.covariates = wt(60.0);
    subject.reset_times = vec![3.0];
    subject.reset_covariates = vec![wt(40.0)];
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "init + reset must be analytic (#486)"
    );
    let theta = vec![1.0, 20.0, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // The three checks above are twin-vs-production, so they cannot see the two engines
    // agreeing on a wrong snapshot. Pin WHICH map is read without re-deriving the closed
    // form: moving only `reset_covariates` must move the post-reset prediction, and a walk
    // still reading any neighbouring record would be unmoved.
    let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    for rival in [90.0, 150.0, 60.0] {
        let mut perturbed = subject.clone();
        perturbed.reset_covariates = vec![wt(rival)];
        let got = ode_subject_sensitivities(&model, &perturbed, &theta, &eta).expect("supported");
        let (a, b) = (base.obs[2].f, got.obs[2].f);
        assert!(
            (a - b).abs() > 0.05 * a.abs(),
            "changing the reset row's WT from 40 to {rival} moved the post-reset prediction \
             only from {a:.4} to {b:.4}: the walk is not reading that row's snapshot"
        );
    }
}

/// **Two resets on the sensitivity walk** (#1133). Anchor arm E pins the per-reset index
/// on the *value* engine, but it runs through `predict`, never through
/// `ode_subject_sensitivities` — so the twin's own `tl.push((rt, K_RESET, r))` and
/// `pk_at_reset[idx]` were still only ever evaluated at index 0, and reverting either to a
/// constant `0` passed the whole unit suite.
///
/// The two reset rows carry `WT = 40` and `WT = 160`, which no neighbouring record shares,
/// so indexing them apart is observable in `f` and in `∂f/∂(θ,η)` alike.
#[test]
fn ode_provider_reset_seeds_are_indexed_per_reset() {
    let model = parse_model_string(ONECPT_IV_INIT_WTV_ODE).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(70.0), wt(90.0), wt(150.0), wt(150.0)];
    subject.covariates = wt(60.0);
    subject.reset_times = vec![3.0, 6.0];
    subject.reset_covariates = vec![wt(40.0), wt(160.0)];
    assert!(subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![1.0, 20.0, 500.0];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);

    // Perturb only the SECOND reset's row. The `t = 8` observation follows it, so it must
    // move; a walk indexing `pk_at_reset[0]` for both resets would be unmoved.
    let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    let mut perturbed = subject.clone();
    perturbed.reset_covariates = vec![wt(40.0), wt(45.0)];
    let got = ode_subject_sensitivities(&model, &perturbed, &theta, &eta).expect("supported");
    let (a, b) = (base.obs[3].f, got.obs[3].f);
    assert!(
        (a - b).abs() > 0.05 * a.abs(),
        "changing only the SECOND reset row's WT moved the post-reset prediction merely \
         from {a:.4} to {b:.4}: the walk is reading one reset's snapshot for both"
    );
    // And the first reset still governs its own episode: the `t = 4` observation sits
    // between the two resets, so perturbing reset 1 must leave it alone.
    approx::assert_relative_eq!(base.obs[2].f, got.obs[2].f, max_relative = 1e-12);
}

/// **IOV × time-varying covariates × a reset that re-seeds `init(...)`** (#1133).
///
/// The twin above covers the non-IOV TV-cov walk. This one takes the *other* seeding
/// route — `seed_iov_events`' per-event branch, which fires for an IOV subject with TV
/// covariates — where the reset row's snapshot is built through the occasion-less
/// (`seed_pk_only_cov`) path, mirroring `predict_iov`: the reader stores no `OCC` for a
/// reset row, so κ = 0 there on both sides.
///
/// `WT` sits on `V`, and `init(central) = BASE/V`, so the reset row's `WT = 40` — shared
/// with no neighbouring record (the one before carries 90, the one after 150) — is
/// observable in the value and in `∂/∂(θ, stacked-η)`. `predict_iov` is the oracle, so a
/// twin that seeded from a different row fails the value comparison first.
#[test]
fn ode_provider_iov_tvcov_reset_reads_the_reset_rows_snapshot() {
    let model = parse_model_string(ONECPT_IV_INIT_WTV_ODE_IOV).expect("parse");
    assert_eq!(model.n_kappa, 1);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subj = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    subj.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subj.dose_covariates = vec![wt(60.0)];
    subj.obs_covariates = vec![wt(70.0), wt(90.0), wt(150.0), wt(150.0)];
    subj.covariates = wt(60.0);
    subj.dose_occasions = vec![1];
    subj.occasions = vec![1, 1, 2, 2];
    subj.reset_times = vec![3.0];
    subj.reset_covariates = vec![wt(40.0)];
    assert!(subj.has_tv_covariates());
    let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(groups.len(), 2);

    let theta = vec![1.0, 20.0, 500.0];
    // NON-ZERO κ, and different per occasion. With κ ≡ 0 the two occasions are numerically
    // identical, so the value comparison below is blind to *which* κ the reset seed uses —
    // and the κ convention is exactly what `reset_row_occasion` decides (#1133).
    let stacked = vec![0.1, -0.05, 0.15, -0.20];
    let sens = ode_subject_sensitivities_iov(&model, &subj, &theta, &stacked).expect("analytic");
    let pred = |st: &[f64], j: usize| -> f64 {
        let eta_bsv = st[..model.n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..groups.len())
            .map(|g| {
                st[model.n_eta + g * model.n_kappa..model.n_eta + (g + 1) * model.n_kappa].to_vec()
            })
            .collect();
        crate::pk::predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[j]
    };
    // Value parity first: this is what a stale reset snapshot breaks.
    for (j, o) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(o.f, pred(&stacked, j), max_relative = 1e-8, epsilon = 1e-10);
    }
    // Then the jets, against FD of the same production predictor.
    let h = 1e-6;
    for (j, o) in sens.obs.iter().enumerate() {
        for k in 0..stacked.len() {
            let mut sp = stacked.clone();
            sp[k] += h;
            let mut sm = stacked.clone();
            sm[k] -= h;
            let g = (pred(&sp, j) - pred(&sm, j)) / (2.0 * h);
            approx::assert_relative_eq!(o.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
        }
    }
    // Non-degeneracy without re-deriving the closed form: moving ONLY the reset row's
    // covariates must move the post-reset prediction. This also covers the subject-static
    // fallback (`WT = 60`) — if `reset_covariates` ever stopped being populated,
    // `reset_cov` would silently return `subject.covariates` on BOTH sides of the twin
    // oracle and every parity assertion above would still pass.
    let t4 = sens.obs[2].f;
    for (label, rival) in [
        ("previous record", 90.0),
        ("next record", 150.0),
        ("subject-static", 60.0),
    ] {
        let mut perturbed = subj.clone();
        perturbed.reset_covariates = vec![wt(rival)];
        let got =
            ode_subject_sensitivities_iov(&model, &perturbed, &theta, &stacked).expect("analytic");
        assert!(
            (t4 - got.obs[2].f).abs() > 0.05 * t4.abs(),
            "setting the reset row's WT to the {label} value ({rival}) moved the post-reset \
             prediction only from {t4:.4} to {:.4}: the walk is not reading that row",
            got.obs[2].f
        );
    }
}

/// **Estimated lagtime × time-varying covariates.** A 1-cpt oral ODE with a WT
/// covariate on CL *and* a bare `LAGTIME`. The static time-shift identity is invalid
/// here (WT switches on an absolute timeline), so the lag sensitivity comes from the
/// event-time saltation injected at the dose and propagated through the per-event
/// (TV-cov) params. Validates the full `SubjectSens` (value, `∂f/∂η`, `∂f/∂θ`, and the
/// 2nd-order blocks via central differences of the analytic gradient) against the
/// production TV-cov+lagtime predictor (#439 lagtime × TV-cov).
#[test]
fn ode_provider_lagtime_tvcov_matches_production() {
    const ONECPT_ORAL_LAG_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVLAG(0.5, 0.01, 5.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(ONECPT_ORAL_LAG_TVCOV_ODE).expect("parse");
    assert!(model.has_lagtime());
    let subject = tvcov_subject();
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TV-cov + lagtime supported"
    );
    let theta = vec![1.0, 20.0, 1.2, 0.5, 0.75];
    // η = [ETA_CL, ETA_LAG] — lag carries IIV, so this exercises the ∂²/∂η_LAG²
    // (event-time saltation 2nd-order) × TV-cov boundary interaction.
    let eta = vec![0.1, 0.05];
    let sens =
        ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("tvcov+lag supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(&model, &subject, th, e)[j]
    };
    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            obs.f,
            pred(&eta, &theta, j),
            max_relative = 1e-6,
            epsilon = 1e-9
        );
        for m in 0..model.n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-3, epsilon = 1e-6);
        }
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
        }
        // 2nd order via central differences of the analytic gradient w.r.t. η.
        let grad_at = |e: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let s = ode_subject_sensitivities(&model, &subject, &theta, e).expect("supported");
            (s.obs[j].df_deta.clone(), s.obs[j].df_dtheta.clone())
        };
        for k in 0..model.n_eta {
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let (de_p, dt_p) = grad_at(&ep);
            let (de_m, dt_m) = grad_at(&em);
            for l in 0..model.n_eta {
                let d2 = (de_p[l] - de_m[l]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * model.n_eta + l],
                    d2,
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
            for m in 0..model.n_theta {
                let d2 = (dt_p[m] - dt_m[m]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * model.n_theta + m],
                    d2,
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
    // The self-FD Hessian block above differentiates the analytic gradient by
    // itself, so it cannot see an error that is *consistent* across η — which is
    // exactly what #1060 was. Anchor the second order against production FD too.
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// #1060: the same 1-cpt oral model as `ONECPT_ORAL_LAG_TVCOV_ODE`, but the
// lagtime is large enough that the dose's arrival lands *past* a covariate
// record. The event-driven walk integrates the segment ending at the arrival
// under the dose record's PK snapshot and the segment starting there under the
// next record's (NONMEM end-of-interval, `predictions.rs` `Kind::Dose`), so the
// bolus saltation's post-side velocity/Jacobian belong to the *later* snapshot.
// Reading them from the dose snapshot biases `∂f/∂η_LAG` by the ratio of the two
// covariate-scaled clearances — invisible in the value (the injection is
// jet-only) and invisible to a self-FD Hessian check.
// This is issue #1060's own reproducer, transcribed. The dose lands in
// `central`, whose RHS carries the weight-scaled clearance — that is what makes
// the wrong-snapshot velocity bite at full strength. (An *oral* variant, where
// the dose lands in a `-KA * depot` compartment, leaks only a few percent into
// the Hessian, because the dose compartment's own velocity has no covariate in
// it; that variant is covered separately below.)
const ONECPT_IV_LAG_CROSS_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  LAGTIME = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// Subject for the #1060 crossing geometry, also transcribed from the issue: one
/// bolus of 100 into `CMT = 1` at `t = 1` carrying `WT = 70`, with observations
/// at 0 / 0.5 / 1.5 / 2 / 4 / 8 on a rising weight ramp. With `TVP = 0.7` and
/// `ETA_P = 0.07` the arrival is `1 + 0.7 · e^0.07 ≈ 1.751` — inside the
/// `[1.5, 2.0]` segment, whose PK snapshot is the `t = 2` record's (`WT = 76`),
/// not the dose row's (`WT = 70`), and ≥ 0.24 from either break so the FD steps
/// (1e-6 gradient, 1e-4 Hessian) never move it across one.
///
/// The `t = 0` observation is load-bearing for the `init(...)` variant: it makes
/// the subject's first record precede the dose, so the baseline is seeded at a
/// covariate snapshot both engines agree on (cf. #1046 — the seed lands at the
/// first record, not at `t = 0` by fiat).
fn lag_crossing_subject() -> Subject {
    let mut subject = bolus_subject(&[0.0, 0.5, 1.5, 2.0, 4.0, 8.0]);
    subject.doses = vec![DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0)];
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(70.0)];
    subject.obs_covariates = vec![wt(70.0), wt(72.0), wt(74.0), wt(76.0), wt(80.0), wt(85.0)];
    subject.covariates.insert("WT".to_string(), 70.0);
    subject
}

/// #1060 row C — ODE + time-varying covariates + IIV on the lagtime, with the
/// lagged arrival crossing a covariate change.
///
/// Before the saltation snapshot fix this failed on `∂f/∂η_LAG` with analytic
/// `0.34487735` against an FD reference of `0.36681859` — a relative miss of
/// `0.059815`, which is `1 − (70/76)^0.75 = 0.0598143` to five significant
/// figures. That is the arithmetic signature of evaluating the post-arrival
/// velocity under the dose row's `WT = 70` instead of the arrival segment's
/// `WT = 76`, and it is why this fixture pins the defect rather than merely
/// exercising the code path. The value stayed exact to 1e-6 throughout — the
/// injection is jet-only, so nothing in the prediction ever looked wrong.
#[test]
fn ode_provider_lagtime_tvcov_arrival_crosses_covariate_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_CROSS_ODE).expect("parse");
    assert!(model.has_lagtime());
    let subject = lag_crossing_subject();
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TV-cov + lagtime is in analytic scope (#486) — this must be a kernel \
         test, not a routing test"
    );
    let theta = vec![10.0, 50.0, 0.75, 0.7];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    // Non-vacuity: the lag must actually carry an η-jet, and the arrival must
    // land strictly inside the segment whose snapshot differs from the dose's.
    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
    let max_dlag = sens
        .obs
        .iter()
        .map(|o| o.df_deta[2].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dlag > 1e-6,
        "∂f/∂η_LAG is vacuously zero — the fixture would pass without \
         exercising the saltation at all"
    );
    let arrival = 1.0 + theta[3] * eta[2].exp();
    assert!(
        arrival > 1.5 + 1e-3 && arrival < 2.0 - 1e-3,
        "arrival {arrival} must sit strictly inside the [1.5, 2.0] segment"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

/// #1060 control — same model, same subject, but a lagtime small enough that the
/// arrival stays inside the *first* segment, whose snapshot is the dose record's
/// own. Pre- and post-arrival fields coincide, so this passes before the fix and
/// must stay bit-identical after it: it pins that the fix changes nothing when
/// the arrival does not cross a covariate change.
#[test]
fn ode_provider_lagtime_tvcov_arrival_before_covariate_change_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_CROSS_ODE).expect("parse");
    let mut subject = lag_crossing_subject();
    // Make the record that ends the arrival segment carry the dose row's own
    // weight, so the pre- and post-arrival fields coincide: the fix must be a
    // no-op here, both before and after.
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.obs_covariates[3] = wt(70.0);
    let theta = vec![10.0, 50.0, 0.75, 0.7];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    let arrival = 1.0 + theta[3] * eta[2].exp();
    assert!(
        arrival > 1.5 + 1e-3 && arrival < 2.0 - 1e-3,
        "control arrival {arrival} must sit in the same segment as the \
         crossing case — only the covariate value differs"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// #1060 row A — the crossing geometry plus an `init(...)` baseline, on a
// *three-way* covariate ramp (dose row 70, first record 73, second 76). The
// baseline makes the pre-arrival state non-zero, so `g(x⁻)` is live and the
// first-order term `(g⁻ − g⁺)·δlag` needs each side on its own snapshot: `g⁻`
// on the dose row's (the segment ending at the arrival) and `g⁺` on the
// arrival segment's. A fix that moved *both* sides to the post snapshot would
// pass the two-way fixtures above and fail here.
const ONECPT_LAG_INIT_CROSS_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  LAGTIME = TVP * exp(ETA_P)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// #1060 row A: init + TV covariates + IIV lagtime, arrival crossing a covariate
/// change on a three-way ramp. The issue measured this cell at a 588× gradient
/// margin (vs 299× without `init`) — the baseline amplifies the same defect by
/// giving the pre-arrival segment signal to carry.
#[test]
fn ode_provider_lagtime_tvcov_init_arrival_crosses_covariate_matches_production() {
    let model = parse_model_string(ONECPT_LAG_INIT_CROSS_ODE).expect("parse");
    assert!(model.has_lagtime());
    // Same crossing geometry and the same rising weight ramp: the record ending
    // the pre-arrival segment (`t = 1.5`, WT 74), the dose row (WT 70) and the
    // record ending the arrival segment (`t = 2`, WT 76) are three distinct
    // snapshots, so each evaluation is pinned to its own — a fix that moved the
    // *pre*-side velocity to the post snapshot as well would pass the
    // no-baseline fixture above and fail here.
    let subject = lag_crossing_subject();
    assert!(subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject), "init + lag + TV-cov");
    let theta = vec![10.0, 50.0, 0.75, 0.7, 500.0];
    let eta = vec![0.1, -0.05, 0.07];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

/// #1060, two doses placed on either side of the question: the first arrives
/// inside a segment whose record carries the dose row's own weight (no crossing),
/// the second arrives past a weight change with drug still in the compartment
/// from the first — so the pre-arrival velocity is live without needing an
/// `init(...)` baseline, and one subject exercises both the no-op and the
/// corrected path in a single walk.
#[test]
fn ode_provider_lagtime_tvcov_two_doses_split_segments_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_CROSS_ODE).expect("parse");
    let mut subject = bolus_subject(&[0.0, 0.5, 1.5, 2.0, 4.0, 5.5, 6.0, 9.0]);
    subject.doses = vec![
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(5.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(70.0), wt(80.0)];
    // Dose 1 arrives ≈1.751 in the segment the `t = 2` record ends; that record
    // carries the dose row's 70, so nothing changes there. Dose 2 arrives ≈5.751
    // in the segment the `t = 6` record ends, and that one steps to 88.
    subject.obs_covariates = vec![
        wt(70.0),
        wt(70.0),
        wt(70.0),
        wt(70.0),
        wt(80.0),
        wt(80.0),
        wt(88.0),
        wt(88.0),
    ];
    subject.covariates.insert("WT".to_string(), 70.0);
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "two-dose TV-cov + lag"
    );
    let theta = vec![10.0, 50.0, 0.75, 0.7];
    let eta = vec![0.1, -0.05, 0.07];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// #1060 via the compartment-indexed spelling: `ALAG1` resolves through
// `DoseAttrMap::lag_slot` rather than the bare `PK_IDX_LAGTIME` slot, so the
// per-dose lag slot lookup is a second route into the same saltation. Same
// crossing geometry, same expected arithmetic.
const ONECPT_IV_INDEXED_ALAG_CROSS_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  ALAG1 = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// #1060 with the lag declared as a compartment-indexed `ALAG1` (#369). The walk
/// resolves this dose's slot through the dose-attribute map instead of the bare
/// lagtime slot, so this pins that the corrected snapshot reaches both spellings.
#[test]
fn ode_provider_lagtime_tvcov_indexed_alag_matches_production() {
    let model = parse_model_string(ONECPT_IV_INDEXED_ALAG_CROSS_ODE).expect("parse");
    assert!(model.has_lagtime(), "ALAG1 must register as a lagtime");
    let subject = lag_crossing_subject();
    assert!(ode_tvcov_supported(&model, &subject), "ALAG1 + TV-cov");
    let theta = vec![10.0, 50.0, 0.75, 0.7];
    let eta = vec![0.1, -0.05, 0.07];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

/// The lagged-**infusion** neighbour of #1060. A finite-duration infusion under an
/// estimated lagtime switches its rate on at the same moving arrival, through
/// `inject_rate_saltation` rather than the bolus branch — and that call also reads
/// its PK snapshot from the dose row. The `init(...)` baseline makes the state
/// non-zero at the rate-on instant, which is the only condition under which the
/// bare-field half of that boundary term is non-zero, so this is the fixture that
/// would expose the same defect on the infusion path.
#[test]
fn ode_provider_lagtime_tvcov_infusion_rate_on_crosses_covariate_matches_production() {
    let model = parse_model_string(ONECPT_LAG_INIT_CROSS_ODE).expect("parse");
    let mut subject = lag_crossing_subject();
    // amt 100 at rate 50 ⇒ a 2 h window opening at the lagged arrival ≈1.751 and
    // closing ≈3.751, so the rate-on lands past the `t = 1.5` record and the
    // window spans the `t = 2` one.
    subject.doses = vec![DoseEvent::new(1.0, 100.0, 1, 50.0, false, 0.0)];
    // Widen the weight step across the arrival: the boundary term this fixture
    // is about scales with the difference between the two fields, and the gentle
    // 70→76 ramp puts it barely at the harness bound.
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.obs_covariates[3] = wt(140.0);
    subject.obs_covariates[4] = wt(150.0);
    subject.obs_covariates[5] = wt(160.0);
    assert!(
        ode_tvcov_supported(&model, &subject),
        "infusion + lag + TV-cov"
    );
    let theta = vec![10.0, 50.0, 0.75, 0.7, 500.0];
    let eta = vec![0.1, -0.05, 0.07];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// #1060 review #1: a 2-cpt twin of the crossing fixture with a **second dose** whose
// infusion window straddles the bolus arrival. `ALAG1` lags compartment 1 only, so the
// cmt-2 infusion is frozen in absolute time while the bolus arrival moves — the two sides
// of the arrival therefore differ in their Jacobian but share the forcing, which is exactly
// the configuration in which the dropped `½(J⁻−J⁺)·f` curvature is visible.
const ONECPT_IV_LAG_STRADDLE_INF_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  theta TVK21(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  K21 = TVK21
  ALAG1 = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central, periph])
[odes]
  d/dt(central) = -(CL/V) * central + K21 * periph
  d/dt(periph)  = -(CL/V) * periph
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// Subject shared by the two straddling-forcing fixtures: the #1060 record geometry with a
/// steeper weight step (70 → 140 at `t = 2`), because the term under test scales with the
/// difference between the two fields and the gentle 70 → 76 ramp puts it under the harness
/// bound. Doses are set by each caller.
fn straddling_forcing_subject() -> Subject {
    let mut subject = bolus_subject(&[0.0, 0.5, 1.5, 2.0, 4.0, 8.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.obs_covariates = vec![
        wt(70.0),
        wt(70.0),
        wt(70.0),
        wt(140.0),
        wt(150.0),
        wt(160.0),
    ];
    subject.dose_covariates = vec![wt(70.0), wt(70.0)];
    subject.covariates.insert("WT".to_string(), 70.0);
    subject
}

/// #1060 review #1 — a concurrent forcing that **straddles** the moving arrival belongs in
/// both one-sided velocities.
///
/// `ẋ̈± = J±·(g± + f)` for any forcing `f` active on both sides. `f` cancels in the
/// first-order jump `(g⁻ − g⁺)`, and while `J⁻ = J⁺` it cancelled in the curvature too —
/// which is why evaluating `g±` from the bare RHS was invisible for as long as both sides
/// read the same snapshot. Once they read different ones the leftover `½(J⁻−J⁺)·f` is real:
/// this fixture missed `∂²f/∂η_LAG²` by 1.9 % (`0.87941` vs `0.89639`) with the bare RHS and
/// by 50 % before #1060's first-order fix. Here the straddling forcing is **frozen** — a
/// cmt-2 infusion under an `ALAG1` that lags compartment 1 only — so it does not move with
/// the boundary at all.
#[test]
fn ode_provider_lagtime_tvcov_arrival_straddles_frozen_infusion_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_STRADDLE_INF_ODE).expect("parse");
    let mut subject = straddling_forcing_subject();
    // Frozen 4 h infusion into `periph` over [0, 4]; lagged bolus into `central` at t = 1,
    // arriving ≈1.751 — strictly inside the window and inside the [1.5, 2.0] segment.
    subject.doses = vec![
        DoseEvent::new(0.0, 400.0, 2, 100.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    assert!(subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject), "must stay analytic");
    let theta = vec![10.0, 50.0, 0.75, 0.7, 0.4];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    let arrival = 1.0 + theta[3] * eta[2].exp();
    assert!(
        arrival > 0.0 + 1e-3 && arrival < 4.0 - 1e-3,
        "arrival {arrival} must straddle the [0, 4] infusion window, not touch its edges"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

/// #1073 on the ODE **twin**: an infusion window whose end falls strictly between two
/// records, under a covariate that changes across it, plus a lagged arrival in the same
/// subject.
///
/// This is the oracle `Dual2`-vs-FD parity cannot supply for these lines.
/// `check_vs_production` compares the twin's own **value path** against
/// `compute_predictions_with_tv` — an independent engine — before it compares
/// derivatives against FD *of that engine*. FD parity alone perturbs the twin's own
/// value path, so a twin that resolved a segment to a different record than production
/// moves both sides together and agrees on the wrong answer.
///
/// What it covers that no other committed fixture does:
///
///   * `next_record_params` — the segment resolution itself, on both the arrival and
///     the rate-off.
///   * the `pre_params` at `K_INF_END`, which #1073 changed from `last_params` (the
///     *previous* record) to the enclosing one. That was the half measured at 4.2 % on
///     the predictions and 23.5 OFV against NONMEM 7.6.0.
///   * `post_snapshot`'s scan continuing *past* a non-record rather than stopping at
///     one, and the retirement of its co-timed-sibling-dose exception.
///
/// Records sit at 0, 0.5, 1.5, 2, 4, 8 with `WT` stepping 70 → 140 at `t = 2`. Both
/// doses go into `central`, so both ride `ALAG1 ≈ 0.751`, and **two** non-record
/// boundaries stack inside the single `(1.5, 2]` record interval:
///
/// ```text
///   1.5  obs (WT 70)  │  1.751 bolus arrival  │  1.901 infusion end  │  2.0 obs (WT 140)
/// ```
///
/// Every piece of that interval runs on the `t = 2` record. Two things follow, and the
/// geometry is built so that each is separately observable:
///
///   * the **pre**-side of the rate-off is the enclosing record (`WT = 140`), not the
///     previous one (`WT = 70`) it used to reuse — the half measured at 4.2 % on the
///     predictions and 23.5 OFV against NONMEM 7.6.0;
///   * `post_snapshot` at the arrival must **continue past** the infusion end — a
///     non-record supplies no parameters — and return the `t = 2` record. Stopping at
///     the first non-record instead falls back to the `t = 1.5` record, a different
///     field.
///
/// Two properties of the fixture are load-bearing and each was learned by a mutation
/// that initially survived:
///
///   * The infusion must ride the same `ALAG1` as the bolus. A frozen window (into
///     `periph`, which `ALAG1` does not lag) makes the rate-off a fixed boundary, its
///     `d_off` jet zero, and the whole saltation inert.
///   * A second non-record must sit between the arrival and the next record. With the
///     arrival's successor already being a record, `post_snapshot` returns it whether
///     or not it can skip past a non-record.
#[test]
fn ode_provider_stacked_non_record_boundaries_under_tvcov_match_production() {
    let model = parse_model_string(ONECPT_IV_LAG_STRADDLE_INF_ODE).expect("parse");
    let mut subject = straddling_forcing_subject();
    // 115 units at rate 100 -> a 1.15 h window into `central`, lagged by ALAG1, so its
    // END moves with the lagtime and lands at ≈1.901 — after the bolus arrival and
    // still strictly before the t=2 record.
    subject.doses = vec![
        DoseEvent::new(0.0, 115.0, 1, 100.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    assert!(subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject), "must stay analytic");
    let theta = vec![10.0, 50.0, 0.75, 0.7, 0.4];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];

    // The geometry, asserted rather than assumed: arrival then infusion end, both
    // strictly inside `(1.5, 2]`, and the two records bracketing them carrying
    // different `WT` — without which the conventions agree by construction.
    let lag = theta[3] * eta[2].exp();
    let arrival = 1.0 + lag;
    let inf_end = lag + 115.0 / 100.0;
    assert!(
        1.5 + 1e-3 < arrival && arrival < inf_end - 1e-3 && inf_end < 2.0 - 1e-3,
        "need 1.5 < arrival ({arrival}) < infusion end ({inf_end}) < 2.0, so a \
         non-record sits between the arrival and the next record"
    );
    let wt_at = |j: usize| subject.obs_covariates[j]["WT"];
    assert!(
        (wt_at(2) - wt_at(3)).abs() > 1.0,
        "the records bracketing both boundaries must carry different WT ({} vs {})",
        wt_at(2),
        wt_at(3)
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

/// Twin of the fixture above with a **co-moving** straddling forcing, and the reason the
/// membership rule is "straddles", not "is frozen".
///
/// Both doses ride the same bare `LAGTIME`, so the infusion window `[lag, 4+lag]` and the
/// bolus arrival `1+lag` shift together. It is tempting to conclude that a co-moving forcing
/// should be kept out of the saltation — the neighbouring
/// `ode_provider_bolus_concurrent_infusion_lagtime_hessian_matches_fd` looks like evidence
/// for exactly that. It is not: there the infusion *starts* at the arrival, and a window
/// toggling at the boundary is covered by its own rate saltation at the same event. A
/// co-moving window that merely straddles is grid-constant either side of the arrival, so it
/// belongs in `v±` like any frozen one — measured here as `∂²f/∂η_LAG²` `−0.65279` vs FD
/// `−0.48247` (35 %) when it is left out.
#[test]
fn ode_provider_lagtime_tvcov_arrival_straddles_comoving_infusion_matches_production() {
    let model = parse_model_string(ONECPT_IV_LAG_CROSS_ODE).expect("parse");
    let mut subject = straddling_forcing_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 400.0, 1, 100.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let theta = vec![10.0, 50.0, 0.75, 0.7];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    let lag = theta[3] * eta[2].exp();
    assert!(
        1.0 + lag > lag + 1e-3 && 1.0 + lag < 4.0 + lag - 1e-3,
        "arrival must straddle the co-moving window, not toggle at either edge"
    );
    assert!(ode_tvcov_supported(&model, &subject));
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// #1060 review #9(b): the straddling forcing is a built-in `input_rate` whose kernel reads
// the PK snapshot (`KA` carries the weight covariate). Unlike a constant infusion rate — the
// same number on both sides, so it only reaches the curvature — a snapshot-dependent `R_in`
// makes `v⁻ ≠ v⁺` in the forcing itself, moving the omission into the **first-order** jump.
const TWOCPT_LAG_STRADDLE_INPUT_RATE_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  theta TVKA(0.8, 0.01, 10.0)
  theta TVK21(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  KA = TVKA * (WT/70)^THETA_WT
  K21 = TVK21
  ALAG1 = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central, periph])
[odes]
  d/dt(central) = -(CL/V) * central + K21 * periph
  d/dt(periph)  = first_order(ka=KA) - K21 * periph
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// Largest relative miss of the analytic `∂²f/∂η²` block against double-FD of the production
/// predictor, over the elements big enough for a 4-point stencil to resolve. The
/// assert-shaped [`check_hessian_vs_production_fd`] is the normal tool; this returns the
/// number instead, for the one fixture that carries a *quantified, issue-tracked* residual
/// and must fail if it grows rather than merely being exempted.
fn max_eta_hessian_miss_vs_production_fd(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let n_eta = model.n_eta;
    let base = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
    let pred =
        |e: &[f64], j: usize| -> f64 { compute_predictions_with_tv(model, subject, theta, e)[j] };
    let h = 1e-4;
    let mut worst = 0.0_f64;
    for p in 0..n_eta {
        for (j, o) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let shift = |sk: f64, sp: f64| {
                    let mut e = eta.to_vec();
                    e[k] += sk * h;
                    e[p] += sp * h;
                    e
                };
                let fd = (pred(&shift(1.0, 1.0), j)
                    - pred(&shift(1.0, -1.0), j)
                    - pred(&shift(-1.0, 1.0), j)
                    + pred(&shift(-1.0, -1.0), j))
                    / (4.0 * h * h);
                // Same `epsilon` gate as the assert-shaped helper: an element whose absolute
                // miss is under the floor is at the resolution limit of a 4-point stencil,
                // and its relative miss carries no information.
                let a = o.d2f_deta2[k * n_eta + p];
                if (a - fd).abs() > 5e-3 {
                    worst = worst.max((a - fd).abs() / a.abs().max(fd.abs()));
                }
            }
        }
    }
    worst
}

/// #1060 review #9(b) — a **field-dependent** straddling forcing, and the boundary of what
/// this PR fixes.
///
/// The straddling forcing here is a built-in `first_order` kernel whose `KA` carries the
/// weight covariate, so unlike a constant infusion rate it takes a *different value* on the
/// two sides. That moves the omission into the first-order jump `(v⁻ − v⁺)·δ`, which is a
/// gradient error, not curvature — and the gradient is exact here.
///
/// The `∂²f/∂η²` block is **not** exact — up to 6.4 % on the late observations — and the
/// reason is a distinct term this PR does not carry (#1075). `ẋ̈ = ∂f/∂t + J·f` for an
/// explicitly time-dependent field, and a pointwise `R_in(tad)` is exactly that; carrying the
/// derivation through the shift-and-flow-back gives a `½(∂f⁻/∂t − ∂f⁺/∂t)·δ²` term alongside
/// the Jacobian ones. It vanishes when the two sides share a snapshot (every fixture before
/// this one), needs a `∂R_in/∂tad` primitive that only exists at `tad = 0` today
/// (`rate_dtad_at_zero`), and applies to all four saltation sites — so it is tracked there
/// rather than half-landed here. This bound is the guard: it must not grow, and it becomes
/// `check_hessian_vs_production_fd` when that term lands.
#[test]
fn ode_provider_lagtime_tvcov_arrival_straddles_input_rate_matches_production() {
    let model = parse_model_string(TWOCPT_LAG_STRADDLE_INPUT_RATE_ODE).expect("parse");
    assert!(
        !model.ode_spec.as_ref().unwrap().input_rate.is_empty(),
        "fixture must carry a built-in input-rate forcing"
    );
    let mut subject = straddling_forcing_subject();
    // Oral dose feeding the `first_order` kernel on `periph` (frozen — `ALAG1` lags
    // compartment 1 only), plus the lagged bolus into `central`.
    subject.doses = vec![
        DoseEvent::new(0.0, 400.0, 2, 0.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![10.0, 50.0, 0.75, 0.7, 0.8, 0.4];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    let miss = max_eta_hessian_miss_vs_production_fd(&model, &subject, &theta, &eta);
    assert!(
        miss < 8.0e-2,
        "#1075's known `∂f/∂t` residual on the second order grew to {miss:.3e} (was 6.4e-2)"
    );
}

/// Two **co-timed** lagged doses arriving together on the far side of a covariate change.
///
/// Each injects its own saltation, and the pair is exact only because they telescope: the
/// first dose's post side must be the second dose's pre side — which is the second's own
/// `pk_at_dose` row, not the next record's — so that
/// `[g(x,p₀) − g(x+Δ₁,p₁)] + [g(x+Δ₁,p₁) − g(x+Δ₁+Δ₂,p_post)]` collapses to a single jump.
/// Handing the first dose the next *record's* snapshot instead leaves an uncancelled
/// `g(x+Δ₁,p₁) − g(x+Δ₁,p_post)` behind, which is nonzero exactly when the covariates change
/// across the arrival — the case this fixture is in. Every other co-timed-dose fixture in the
/// suite has constant covariates, where the leftover is identically zero, so nothing else
/// catches it.
#[test]
fn ode_provider_lagtime_tvcov_co_timed_doses_telescope_across_covariate_change() {
    let model = parse_model_string(ONECPT_IV_LAG_CROSS_ODE).expect("parse");
    let mut subject = straddling_forcing_subject();
    // Both doses at t = 1, both lagged, arriving together at ≈1.751 — inside the [1.5, 2.0]
    // segment, whose snapshot (WT = 140) differs from the dose rows' (WT = 70).
    subject.doses = vec![
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.0, 60.0, 1, 0.0, false, 0.0),
    ];
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![10.0, 50.0, 0.75, 0.7];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

/// The control that identifies the residual above, and the only fixture in this group that
/// is green on both sides of the fix.
///
/// Same straddling `first_order` forcing, same crossing geometry — but `KA` carries no
/// covariate, so the kernel evaluates identically on the two snapshots and
/// `∂f⁻/∂t = ∂f⁺/∂t`. Both orders are then exact. Green here beside a 5 % gradient miss and a
/// 6.4 % second-order residual on the twin is what pins the twin's numbers to the kernel's
/// *snapshot dependence* specifically, rather than to straddling forcings in general or to
/// anything about the crossing geometry the two fixtures share.
#[test]
fn ode_provider_lagtime_tvcov_arrival_straddles_covariate_free_input_rate_is_exact() {
    let src =
        TWOCPT_LAG_STRADDLE_INPUT_RATE_ODE.replace("KA = TVKA * (WT/70)^THETA_WT", "KA = TVKA");
    let model = parse_model_string(&src).expect("parse");
    let mut subject = straddling_forcing_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 400.0, 2, 0.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    assert!(ode_tvcov_supported(&model, &subject));
    let theta = vec![10.0, 50.0, 0.75, 0.7, 0.8, 0.4];
    let eta: Vec<f64> = vec![0.1, -0.05, 0.07];
    // Non-vacuity: the straddling forcing must actually be carrying mass at the arrival, or
    // "exact" here says nothing about the term it is supposed to isolate.
    let with_forcing = compute_predictions_with_tv(&model, &subject, &theta, &eta);
    let mut bolus_only = subject.clone();
    bolus_only.doses = vec![subject.doses[1].clone()];
    let without = compute_predictions_with_tv(&model, &bolus_only, &theta, &eta);
    assert!(
        with_forcing
            .iter()
            .zip(&without)
            .any(|(a, b)| (a - b).abs() > 0.1 * b.abs().max(1e-6)),
        "the `first_order` forcing contributes nothing — the control is vacuous"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// #1060 review #9(c)/#4: IOV × estimated lagtime, with the occasion boundary falling between
// a lagged infusion's arrival and the next record. `pk_snapshot_equal` compares snapshots by
// VALUE, so at κ̂ = 0 — the inner-BFGS cold start and the first outer iteration — the two
// sides compare equal while carrying jets on different κ axes.
const ONECPT_IOV_LAG_INF_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVP(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_P  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
  LAGTIME = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// #1060 review #4/#9(c) — the moving rate boundary must not take a shortcut licensed by a
/// **value-only** snapshot comparison, because IOV breaks the premise that equal values imply
/// equal jets.
///
/// `pk_snapshot_equal`'s doc scopes its argument to the deterministic non-IOV PK map. Under
/// IOV each event is seeded with its own occasion's κ, so two snapshots straddling an occasion
/// boundary are value-equal at `κ̂ = 0` — the inner-BFGS cold start and the first outer
/// iteration — with jets on different κ axes. Taking the closed form there drops a
/// `(v⁻−v⁺)·δlag` term whose value is zero (predictions stay exact) and whose jet is not.
///
/// Measured before the rate-on boundary stopped consulting that predicate:
/// `∂²f/∂η_LAG∂κ_g0` analytic `+0.05207` against FD of `predict_iov` `−0.01478` — the wrong
/// **sign**, 128 % — with every non-lag element exact to 1e-7. Everything about the geometry
/// is load-bearing: without the prior bolus the state is zero at the arrival and the whole
/// term vanishes vacuously (the first version of this fixture passed for that reason), and
/// the occasion boundary sits on an observation record because `predict_iov` is not a valid
/// oracle at a boundary without one (#931).
#[test]
fn ode_provider_iov_lagtime_rate_on_across_occasion_boundary_matches_predict_iov() {
    let model = parse_model_string(ONECPT_IOV_LAG_INF_ODE).expect("parse");
    assert_eq!(model.n_kappa, 1);
    let mut subj = bolus_subject(&[0.5, 1.5, 2.0, 4.0, 8.0]);
    // A prior bolus leaves residual drug at the second dose's arrival — without it the
    // state is zero there and the whole field-difference term is vacuously zero. The
    // lagged 2 h infusion is dosed in occasion 1 and arrives ≈1.751, between the last
    // occasion-1 record (1.5) and the first occasion-2 record (2.0).
    subj.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.0, 200.0, 1, 100.0, false, 0.0),
    ];
    subj.dose_occasions = vec![1, 1];
    subj.occasions = vec![1, 1, 2, 2, 2];
    let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(groups.len(), 2);
    let theta = vec![10.0, 50.0, 0.7];
    // κ̂ = 0 on both groups: values equal, jets on different axes.
    let stacked = vec![0.1, 0.07, 0.0, 0.0];
    let sens = ode_subject_sensitivities_iov(&model, &subj, &theta, &stacked).expect("analytic");
    let pred = |st: &[f64], j: usize| -> f64 {
        let eta_bsv = st[..model.n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..groups.len())
            .map(|g| {
                st[model.n_eta + g * model.n_kappa..model.n_eta + (g + 1) * model.n_kappa].to_vec()
            })
            .collect();
        crate::pk::predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[j]
    };
    let n = stacked.len();
    // Non-vacuity: κ_g0 must actually move the late observations, or the term under test is
    // identically zero and the fixture proves nothing.
    let kappa_g0_effect = sens
        .obs
        .iter()
        .map(|o| o.df_deta[2].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        kappa_g0_effect > 1e-3,
        "κ_g0 has no effect on any observation — the fixture would pass vacuously"
    );
    let h = 1e-4;
    for (j, o) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(o.f, pred(&stacked, j), max_relative = 1e-6, epsilon = 1e-9);
        for k in 0..n {
            for p in 0..n {
                let shift = |sk: f64, sp: f64| {
                    let mut s = stacked.clone();
                    s[k] += sk * h;
                    s[p] += sp * h;
                    s
                };
                let fd = (pred(&shift(1.0, 1.0), j)
                    - pred(&shift(1.0, -1.0), j)
                    - pred(&shift(-1.0, 1.0), j)
                    + pred(&shift(-1.0, -1.0), j))
                    / (4.0 * h * h);
                approx::assert_relative_eq!(
                    o.d2f_deta2[k * n + p],
                    fd,
                    max_relative = 5e-3,
                    epsilon = 5e-3
                );
            }
        }
    }
}

/// A TV-cov model whose RHS references the `TAD` (time-after-dose) builtin, so
/// the event-driven TV-cov walk's `last_dose_eff` / time-anchoring is exercised
/// — the other TV-cov parity tests use a `t`-independent RHS, leaving the
/// anchoring covered only through a constant (#451 / #449 review #10). Same
/// parameter shape as `ONECPT_ODE_TVCOV`, so `tvcov_subject` + the same θ/η apply.
#[test]
fn ode_provider_tvcov_tad_dependent_rhs_matches_production() {
    const TVCOV_TAD_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.02 * TAD)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(TVCOV_TAD_ODE).expect("parse");
    assert!(model.ode_spec.as_ref().unwrap().input_rate.is_empty());
    let subject = tvcov_subject();
    assert!(ode_tvcov_supported(&model, &subject));
    // Analytic TV-cov walk (f / ∂η / ∂θ) must match the production predictor + FD
    // with the TAD-dependent RHS.
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
}

/// The light `Dual1` inner η-gradient must equal the full `Dual2` outer
/// `df_deta` for a TV-cov subject too — exercised through the dispatch, so this
/// also covers `ode_tvcov_supported` routing both entry points (#439).
#[test]
fn ode_provider_tvcov_light_matches_full() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let subject = tvcov_subject();
    let theta = vec![1.0, 20.0, 0.75];
    let eta = vec![0.1];
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// The **inner EBE gate** admits a TV-cov bolus subject (#449 review #2): before
/// the fix, `ode_inner_grad_supported` → `ode_subject_supported` returned false
/// for `has_tv_covariates`, so the inner loop silently ran on FD while the outer
/// was analytic. It must now be on, so the analytic `Dual1` TV-cov walk drives
/// EBE convergence — matching the outer analytic scope.
#[test]
fn ode_tvcov_inner_gate_wired() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    let s = tvcov_subject();
    assert!(ode_tvcov_supported(&model, &s));
    assert!(
        crate::sens::provider::ode_inner_grad_supported(&model, &s),
        "TV-cov bolus subject must take the analytic inner gradient, not FD"
    );
}

/// The TV-cov gate admits a bolus TV-cov subject and declines a static-covariate
/// subject (which the normal pk-seeded path serves) and an infusion (FD fallback).
#[test]
fn ode_tvcov_gate_scope() {
    let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
    // Static covariates → not the TV-cov path.
    assert!(!ode_tvcov_supported(&model, &bolus_subject(&[1.0, 2.0])));
    // Bolus TV-cov → supported.
    let tv = tvcov_subject();
    assert!(ode_tvcov_supported(&model, &tv));
    // TV-cov + a real infusion → now supported (finite-duration infusion forcing).
    let mut inf = tv.clone();
    inf.doses[0].duration = 1.0;
    inf.doses[0].rate = inf.doses[0].amt;
    assert!(crate::dosing::is_real_infusion(&inf.doses[0]));
    assert!(ode_tvcov_supported(&model, &inf));
    // TV-cov + EVID 3/4 reset → now supported (state zeroed at the reset).
    let mut rst = tv.clone();
    rst.reset_times = vec![3.0];
    assert!(ode_tvcov_supported(&model, &rst));
    // TV-cov + EVID=2 pk-only breakpoints → now supported (the walk carries `K_PKONLY`
    // events seeded at the breakpoint's covariate snapshot; no κ, so no combined-pk-only
    // analogue to build — #486).
    let mut pko = tv.clone();
    pko.pk_only_times = vec![1.5];
    pko.pk_only_covariates = vec![HashMap::from([("WT".to_string(), 75.0)])];
    assert!(ode_tvcov_supported(&model, &pko));
    // TV-cov + an η-dependent `ExpressionScale` divisor → now supported (subject-static
    // post-walk quotient, #486). The `ONECPT_ODE_TVCOV` Form-C model has no `obs_scale`,
    // so check the gate on the dedicated 2-cpt ExpressionScale + TV-cov fixture.
    let es_model = parse_model_string(TWOCPT_ODE_EXPRSCALE_TVCOV).expect("parse");
    let es_subj = exprscale_tvcov_subject();
    assert!(es_subj.has_tv_covariates());
    assert!(ode_tvcov_supported(&es_model, &es_subj));
}

/// A model whose individual-parameter program carries more than `MAX_ODE_AXES`
/// axes must make `param_derivatives_at_cov` return `None` gracefully (its dispatch
/// only specializes `1..=MAX_ODE_AXES`, hitting the `_ => None` arm) rather than
/// panic — the seeders propagate that `None` via `?`, so the caller falls back to
/// FD. (The gate caps `n_theta + n_eta ≤ MAX_ODE_AXES`, so this `_ => None` is
/// otherwise reachable only through intermediate-axis inflation — see #455.)
/// (#451 re-review #10)
#[test]
fn param_derivatives_at_cov_declines_over_max_axes_gracefully() {
    // 16 thetas + 1 eta = 17 axes (> MAX_ODE_AXES). All thetas feed CL so the
    // program carries every θ-axis.
    let n_th = MAX_ODE_AXES;
    let mut src = String::from("[parameters]\n");
    for i in 1..=n_th {
        src += &format!("  theta T{i}(1.0, 0.1, 10.0)\n");
    }
    src += "  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04 (sd)\n\
                [individual_parameters]\n  CL = exp(ETA_CL) * (";
    src += &(1..=n_th)
        .map(|i| format!("T{i}"))
        .collect::<Vec<_>>()
        .join(" + ");
    src += ")\n  V = T2\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\
                [odes]\n  d/dt(central) = -CL/V * central\n[error_model]\n  \
                DV ~ proportional(PROP_ERR)\n";
    let model = parse_model_string(&src).expect("parse");
    assert_eq!(model.n_theta, n_th);
    assert_eq!(model.n_eta, 1);
    let prog = model
        .ode_spec
        .as_ref()
        .expect("ode")
        .indiv_param_program
        .as_ref()
        .expect("prog");
    assert!(prog.n_axes() > MAX_ODE_AXES, "fixture must exceed the cap");
    let theta = vec![1.0; n_th];
    let pd = param_derivatives_at_cov(prog, &model, &HashMap::new(), &theta, &[0.1]);
    assert!(
        pd.is_none(),
        "> MAX_ODE_AXES axes must decline to FD, not panic"
    );

    // The decline is a property of the **program's axis count**, never of the covariate
    // snapshot: `param_derivatives_at_cov` reads `cov` only to *evaluate* the program, and
    // decides `Some`/`None` before that, on `prog.n_axes()` alone. `run_obs_grad_tvcov`
    // relies on exactly this — it hoists one `ParamDerivs` per event behind a single `?`,
    // which is only equivalent to the old per-consumer checks because a program that
    // resolves at one event's covariates resolves at all of them. If a future change ever
    // made the decline cov-dependent, the hoist would silently serve the *first* event's
    // answer to every other event, so pin the invariant here rather than in a comment.
    for cov in [
        HashMap::new(),
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 1.0e6), ("AGE".to_string(), -3.0)]),
    ] {
        assert!(
            param_derivatives_at_cov(prog, &model, &cov, &theta, &[0.1]).is_none(),
            "the > MAX_ODE_AXES decline must not depend on the covariate snapshot"
        );
    }
}

// ---- #430 slice 1: built-in inverse-Gaussian absorption forcing over Dual2 ----

// 1-cpt oral disposition with Freijer & Post inverse-Gaussian absorption via
// the built-in `igd()` input rate (mirrors examples/igd_inverse_gaussian.ferx).
// MAT/CV2 are θ-only and appear *only* inside `igd()`, so `∂f/∂(TVMAT,TVCV2)`
// flows entirely through the forcing — the parity check fails if the Dual2
// forcing is wrong. Tight ODE tolerances so analytic ≡ FD is clean.
const IGD_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MAT = TVMAT
  CV2 = TVCV2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// Same as IGD_ODE but with an estimated bioavailability F. The dose into the
// igd() compartment is suppressed as a bolus and fed to `R_in` as `F·amt`, so
// F appears *only* inside the forcing — `∂f/∂THETA_F` exercises the F
// derivative carried by the Dual2 forcing's `f_bio` (uncovered by IGD_ODE,
// which has no F; #430 review finding 2).
const IGD_ODE_F: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  theta THETA_F(0.7, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MAT = TVMAT
  CV2 = TVCV2
  F   = THETA_F
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// Same as IGD_ODE but with a compartment-indexed absorption lag `ALAG1` on
// the igd() compartment. The lag is wired through the `DoseAttrMap`, *not*
// `pk_indices` (and not the bare `PK_IDX_LAGTIME` slot), so the provider gate
// must consult `has_lagtime()` to exclude it (#430 review finding 1).
const IGD_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV  * exp(ETA_V)
  MAT   = TVMAT
  CV2   = TVCV2
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// Same disposition shape but with a `transit()` forcing — lifted to Dual2 in
// #430 slice 2, so it is served by the analytic provider (its `ln Γ(n+1)`
// constant rides the `ln_gamma` Dual2 rule).
// IIV on N (the gamma argument) is deliberate: it is what routes the transit
// forcing's `ln Γ(n+1)` derivatives into the FOCEI Hessian — ∂²f/∂η_N² rides the
// *trigamma* (2nd-order `ln_gamma`) rule. With IIV only on CL the second-order
// transit test would be vacuous for trigamma (∂²/∂N² would land in the dropped
// θ-θ block). Tight ODE tols so analytic ≡ central-FD is clean.
const TRANSIT_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_N  ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MTT = TVMTT
  N   = TVN * exp(ETA_N)
  KA  = TVKA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=N, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// 1-cpt disposition with Weibull absorption via the built-in `weibull()` input
// rate (Phase 2; mirrors examples/weibull_absorption.ferx). TD/BETA appear
// *only* inside `weibull()`, so `∂f/∂(TVTD,TVBETA)` and `∂f/∂ETA_BETA` flow
// entirely through the forcing — the parity check fails if the log-domain Dual2
// forcing is wrong. IIV on BETA (the forcing param) routes the forcing's `ln`/
// `exp` 2nd-order rules into the FOCEI Hessian (the transit-N analogue). β = 1.5
// (> 1) so the integrand is smooth at the dose and analytic ≡ central-FD is
// clean (the β < 1 integrable spike is unit-tested in pk/absorption.rs). Tight
// ODE tolerances so analytic ≡ FD is crisp.
const WEIBULL_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// Same as WEIBULL_ODE but with an estimated bioavailability F. The dose into
// the weibull() compartment is suppressed as a bolus and fed to `R_in` as
// `F·amt`, so F appears *only* inside the forcing — `∂f/∂THETA_F` exercises the
// F derivative carried by the Dual2 forcing (uncovered by WEIBULL_ODE).
const WEIBULL_ODE_F: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  theta THETA_F(0.7, 0.001, 0.999)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
  F    = THETA_F
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

// Same as WEIBULL_ODE but with a compartment-indexed absorption lag `ALAG1` on
// the weibull() compartment — wired through the `DoseAttrMap`, not `pk_indices`,
// so the provider gate must consult `has_lagtime()` to exclude it (the
// kind-agnostic #430 finding-1 fix, here exercised for Weibull). The dual loop
// never applies the dose-attr lag, so an admitted model would get a no-lag
// gradient diverging from the f64 predictor.
const WEIBULL_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  theta TVLAG(0.3, 0.01,   5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV  * exp(ETA_V)
  TD    = TVTD
  BETA  = TVBETA
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The kind gate: inverse-Gaussian (#430 slice 1), transit (#430 slice 2), and
/// Weibull (Phase 2 — log-domain forcing over `ln`/`exp`) are all lifted to
/// Dual2, so every built-in input-rate kind is served by the analytic provider.
#[test]
fn input_rate_kind_supported_over_dual_gates_kinds() {
    use crate::pk::absorption::InputRateKind;
    assert!(InputRateKind::InverseGaussian.supported_over_dual());
    assert!(InputRateKind::Transit.supported_over_dual());
    assert!(InputRateKind::Weibull.supported_over_dual());
}

/// With the IG forcing lifted to Dual2, an `igd()` model is served by the
/// analytic provider, and its `f`/`∂f/∂η`/`∂f/∂θ` match the production
/// predictor + central FD — including `∂f/∂(TVMAT,TVCV2)`, which flow only
/// through the forcing.
#[test]
fn ode_provider_igd_absorption_matches_production() {
    let model = parse_model_string(IGD_ODE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "igd() model should be supported once the IG forcing is lifted to Dual2"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 2.0, 0.3];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
}

/// Slice 2 lifts transit: a `transit()` model is now served by the analytic
/// provider, and `f`/`∂f/∂η`/`∂f/∂θ` match the production predictor + central
/// FD — including `∂f/∂(TVMTT,TVN)` and `∂f/∂ETA_N`, which flow only through the
/// transit forcing's `ln Γ(n+1)` constant (so this exercises the new `ln_gamma`
/// `Dual2` = digamma rule end-to-end through the ODE integration).
#[test]
fn ode_provider_transit_absorption_matches_production() {
    let model = parse_model_string(TRANSIT_ODE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "transit() should be supported once its ln_gamma forcing is lifted to Dual2 (#430 slice 2)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0]; // TVCL, TVV, TVMTT, TVN, TVKA
    let eta = vec![0.1, 0.05]; // ETA_CL, ETA_N (N feeds the forcing)
    check_vs_production(&model, &subject, &theta, &eta);
}

/// Second-order blocks of the transit forcing: FOCEI consumes `d2f_deta2` and
/// `d2f_deta_dtheta`, which for transit ride the **trigamma** (2nd-order
/// `ln_gamma`) rule. `N` carries IIV (`ETA_N`), so `∂²f/∂ETA_N²` flows through
/// `trigamma(N+1)` — a wrong trigamma rule fails here while first-order parity
/// still passes. Validated against central FD of the analytic (already
/// FD-checked) `df_deta`.
#[test]
fn ode_provider_transit_second_order_matches_fd_of_gradient() {
    let model = parse_model_string(TRANSIT_ODE).expect("parse");
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0];
    let eta = vec![0.1, 0.05];
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

    // η-η block: FD of df_deta over η (ETA_N → trigamma through the forcing).
    let he = 1e-5;
    for l in 0..n_eta {
        let mut ep = eta.clone();
        ep[l] += he;
        let mut em = eta.clone();
        em[l] -= he;
        let sp = ode_subject_sensitivities(&model, &subject, &theta, &ep).expect("supported");
        let sm = ode_subject_sensitivities(&model, &subject, &theta, &em).expect("supported");
        for (j, obs) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * n_eta + l],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // η-θ cross block: FD of df_deta over θ (TVMTT/TVN flow only through the forcing).
    for m in 0..n_theta {
        let s = 1e-5 * (1.0 + theta[m].abs());
        let mut tp = theta.clone();
        tp[m] += s;
        let mut tm = theta.clone();
        tm[m] -= s;
        let sp = ode_subject_sensitivities(&model, &subject, &tp, &eta).expect("supported");
        let sm = ode_subject_sensitivities(&model, &subject, &tm, &eta).expect("supported");
        for (j, obs) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * n_theta + m],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}

/// Phase 2 lifts Weibull: a `weibull()` model is served by the analytic
/// provider, and `f`/`∂f/∂η`/`∂f/∂θ` match the production predictor + central FD
/// — including `∂f/∂(TVTD,TVBETA)` and `∂f/∂ETA_BETA`, which flow only through the
/// Weibull forcing (so this exercises the log-domain `exp(β·ln(tad/Td))` Dual2
/// evaluation end-to-end through the ODE integration).
#[test]
fn ode_provider_weibull_absorption_matches_production() {
    let model = parse_model_string(WEIBULL_ODE).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "weibull() should be supported once its log-domain forcing is lifted to Dual2 (Phase 2)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 2.0, 1.5]; // TVCL, TVV, TVTD, TVBETA
    let eta = vec![0.1, 0.05]; // ETA_CL, ETA_BETA (BETA feeds the forcing)
    check_vs_production(&model, &subject, &theta, &eta);
}

/// Second-order blocks of the Weibull forcing: FOCEI consumes `d2f_deta2` and
/// `d2f_deta_dtheta`. `BETA` carries IIV (`ETA_BETA`) and appears only inside the
/// forcing, so `∂²f/∂ETA_BETA²` flows through the forcing's `ln`/`exp` 2nd-order
/// `Dual2` rules — a wrong 2nd-order rule fails here while first-order parity
/// still passes. Validated against central FD of the analytic (already
/// FD-checked) `df_deta`.
#[test]
fn ode_provider_weibull_second_order_matches_fd_of_gradient() {
    let model = parse_model_string(WEIBULL_ODE).expect("parse");
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 2.0, 1.5];
    let eta = vec![0.1, 0.05];
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

    // η-η block: FD of df_deta over η (ETA_BETA → forcing 2nd-order rules).
    let he = 1e-5;
    for l in 0..n_eta {
        let mut ep = eta.clone();
        ep[l] += he;
        let mut em = eta.clone();
        em[l] -= he;
        let sp = ode_subject_sensitivities(&model, &subject, &theta, &ep).expect("supported");
        let sm = ode_subject_sensitivities(&model, &subject, &theta, &em).expect("supported");
        for (j, obs) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * n_eta + l],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // η-θ cross block: FD of df_deta over θ (TVTD/TVBETA flow only through the forcing).
    for m in 0..n_theta {
        let s = 1e-5 * (1.0 + theta[m].abs());
        let mut tp = theta.clone();
        tp[m] += s;
        let mut tm = theta.clone();
        tm[m] -= s;
        let sp = ode_subject_sensitivities(&model, &subject, &tp, &eta).expect("supported");
        let sm = ode_subject_sensitivities(&model, &subject, &tm, &eta).expect("supported");
        for (j, obs) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * n_theta + m],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}

/// Bioavailability F on a weibull() model flows *only* through the input-rate
/// forcing (the bolus into the absorption compartment is suppressed and fed to
/// `R_in` as `F·amt`), so the analytic `∂f/∂THETA_F` here exercises the F
/// derivative carried by the Dual2 forcing — the path WEIBULL_ODE (no F) leaves
/// untested.
#[test]
fn ode_provider_weibull_absorption_with_f_matches_production() {
    let model = parse_model_string(WEIBULL_ODE_F).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "weibull()+F should be supported (F scales the dose as a dual)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 2.0, 1.5, 0.7];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
}

/// A `weibull()` model **with a compartment-indexed lag `ALAG1`** must stay on
/// the FD fallback: the dual loop never applies the dose-attr lag, so admitting
/// it would give a no-lag gradient diverging from the f64 predictor. The gate is
/// kind-agnostic (`has_lagtime()`), so Weibull inherits the #430 finding-1 fix —
/// this pins it (the Weibull analogue of `ode_provider_igd_with_alag_*`).
#[test]
fn ode_provider_weibull_with_alag_stays_on_fd_fallback() {
    let model = parse_model_string(WEIBULL_ALAG_ODE).expect("parse");
    assert!(
        model.has_lagtime(),
        "ALAG1 must enable has_lagtime() (precondition for the gate)"
    );
    assert!(
        !ode_analytical_supported(&model),
        "weibull()+ALAG1 must stay on the FD fallback (#430 finding 1, kind-agnostic)"
    );
}

/// A `weibull()` model **with an EVID 3/4 reset** is now analytic on both the
/// outer θ-sensitivities and the inner η-gradient (#486): the dual forcing loop
/// threads the tracked `reset_floor` into the shared `add_prepared_input_rate_forcing`
/// helper, exactly like the f64 predictor's `active_infusions` rule, so a dose's
/// pre-reset tail is correctly turned off. Reset is kind-agnostic (keyed on
/// `!input_rate.is_empty()`), so Weibull inherits the fix — pinned here (the
/// Weibull analogue of the igd reset test).
#[test]
fn ode_provider_weibull_with_reset_matches_production() {
    let model = parse_model_string(WEIBULL_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![10.0];
    let theta = vec![5.0, 50.0, 2.0, 1.5];
    let eta = vec![0.1, 0.05];
    assert!(
        ode_subject_supported(&model, &subject),
        "weibull() + reset is now shared scope for both outer and inner (#486)"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// Built-in absorption + an EVID 3/4 reset is now analytic (#486): the dual
/// forcing loop threads the tracked `reset_floor`, turning off a dose's
/// pre-reset tail exactly like the f64 predictor's `active_infusions` rule.
#[test]
fn ode_provider_igd_with_reset_matches_production() {
    let model = parse_model_string(IGD_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![10.0];
    let theta = vec![5.0, 50.0, 2.0, 0.3];
    let eta = vec![0.1, -0.05];
    // The inner η-gradient shares the scope gate, so it must be analytic too —
    // else the EBE loop would run FD while the outer runs analytic (#430 review #1
    // — same shared-scope invariant, now on the "supported" side of the gate).
    assert!(
        ode_subject_supported(&model, &subject),
        "IG + reset is now shared scope for both outer and inner (#486)"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// Bioavailability F on an igd() model flows *only* through the input-rate
/// forcing (the dose into the absorption compartment is suppressed as a bolus
/// and fed to `R_in` as `F·amt`), so the analytic `∂f/∂THETA_F` here exercises
/// the F derivative carried by the Dual2 forcing — the path IGD_ODE (no F)
/// leaves untested (#430 review finding 2).
#[test]
fn ode_provider_igd_absorption_with_f_matches_production() {
    let model = parse_model_string(IGD_ODE_F).expect("parse");
    assert!(
        ode_analytical_supported(&model),
        "igd()+F should be supported (F scales the dose as a dual)"
    );
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 2.0, 0.3, 0.7];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
}

/// Multi-dose superposition through the IG dual forcing: with two doses the
/// forcing loop sums `R_in(tad)` over both, and the analytic ∂f/∂(η,θ) must
/// still match the production predictor + FD. The single-dose IGD_ODE parity
/// test never exercises the superposition sum.
#[test]
fn ode_provider_igd_multidose_matches_production() {
    let model = parse_model_string(IGD_ODE).expect("parse");
    let mut subject = bolus_subject(&[0.5, 1.5, 4.0, 8.0, 13.0, 16.0, 25.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 80.0, 1, 0.0, false, 0.0),
    ];
    let theta = vec![5.0, 50.0, 2.0, 0.3];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subject, &theta, &eta);
}

/// Second-order blocks of the IG forcing: `check_vs_production` only checks
/// first order, but FOCEI consumes `d2f_deta2` and `d2f_deta_dtheta`. Validate
/// both against central FD of the analytic (already FD-checked) `df_deta` — if
/// the forcing's Dual2 second-order content were wrong, this fails while the
/// first-order parity still passes. TVMAT/TVCV2 are θ-only and live solely in
/// the forcing, so the θ-cross block exercises the forcing's curvature.
#[test]
fn ode_provider_igd_second_order_matches_fd_of_gradient() {
    let model = parse_model_string(IGD_ODE).expect("parse");
    let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = vec![5.0, 50.0, 2.0, 0.3];
    let eta = vec![0.1, -0.05];
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

    // η-η block: FD of df_deta over η.
    let he = 1e-5;
    for l in 0..n_eta {
        let mut ep = eta.clone();
        ep[l] += he;
        let mut em = eta.clone();
        em[l] -= he;
        let sp = ode_subject_sensitivities(&model, &subject, &theta, &ep).expect("supported");
        let sm = ode_subject_sensitivities(&model, &subject, &theta, &em).expect("supported");
        for (j, obs) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * n_eta + l],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // η-θ cross block: FD of df_deta over θ.
    for m in 0..n_theta {
        let s = 1e-5 * (1.0 + theta[m].abs());
        let mut tp = theta.clone();
        tp[m] += s;
        let mut tm = theta.clone();
        tm[m] -= s;
        let sp = ode_subject_sensitivities(&model, &subject, &tp, &eta).expect("supported");
        let sm = ode_subject_sensitivities(&model, &subject, &tm, &eta).expect("supported");
        for (j, obs) in base.obs.iter().enumerate() {
            for k in 0..n_eta {
                let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * n_theta + m],
                    fd,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}

/// Regression for #430 review finding 1, now closed by #486: an igd() model that
/// also declares a compartment-indexed `ALAG{n}` lag is analytic. The lag is
/// wired through the `DoseAttrMap` (not `pk_indices`/`PK_IDX_LAGTIME`), and
/// routes the subject to the event-driven walk (`ode_tvcov_supported`), which now
/// carries `R_in`: the continuous `∂R_in/∂lag` flows through the dual `tad`, and
/// the forcing's onset at the dose's lagged arrival is injected as an exact
/// rate-on saltation. Inverse-Gaussian's onset vanishes identically (the
/// essential singularity at `tad → 0⁺` dominates for every valid `(mat, cv2)`),
/// so this combination has no boundary-safety caveat (contrast Weibull, still FD
/// for `β < 1`, see `ode_provider_weibull_with_alag_stays_on_fd_fallback`).
#[test]
fn ode_provider_igd_with_alag_matches_production() {
    let model = parse_model_string(IGD_ALAG_ODE).expect("parse");
    assert!(
        model.has_lagtime(),
        "ALAG1 must enable has_lagtime() (precondition for the gate)"
    );
    let subj = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
    assert!(
        !ode_subject_supported(&model, &subj),
        "igd()+lagtime: the static walk still declines lagtime (routes to event-driven, #439)"
    );
    assert!(
        ode_tvcov_supported(&model, &subj),
        "igd()+lagtime: the event-driven walk now carries R_in (#486)"
    );
    let theta = vec![5.0, 50.0, 2.0, 0.3, 0.3];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subj, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
    // The onset saltation's 2nd-order coefficient (`coef2` in `inject_rate_saltation`,
    // fed `dr = rate_at_zero(...)` here) is otherwise only value/1st-order checked
    // above; FOCEI's inner Newton step and Laplace `log|H̃|` consume the 2nd-order
    // blocks, so validate them against FD of the analytic 1st-order gradient too.
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

// A one-compartment `first_order()` absorption model (feeds `central` directly,
// no separate depot) — the textbook Bateman shape with an estimated lagtime.
// Unlike inverse-Gaussian, `FirstOrder`'s onset `R_in(0⁺) = dose·ka` is *always*
// finite and nonzero, so this is the sharpest check of the new rate-on
// saltation's `Δr ≠ 0` arm (`ode_provider_igd_with_alag_matches_production`
// exercises only the `Δr = 0` case, since IG's onset always vanishes).
const FIRST_ORDER_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0,  0.05, 20.0)
  theta TVLAG(0.3, 0.01,  5.0)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA * exp(ETA_KA)
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `first_order()` + an estimated compartment-indexed lag (#486): the onset
/// saltation's `Δr = dose·ka` is always finite and nonzero, so this is the
/// sharpest check of the new rate-on injection (contrast the IG test, whose
/// onset always vanishes). Value/`∂f/∂η`/`∂f/∂θ` (incl. `∂f/∂TVLAG` and
/// `∂f/∂TVKA`) must match the production predictor + central FD, and the inner
/// `Dual1` walk must reproduce the outer `Dual2` walk's η-gradient exactly.
///
/// Observation times deliberately avoid landing exactly on `dose.time + TVLAG`
/// (`= 0.3` here): a nonzero-onset forcing's prediction has a genuine kink there
/// in `lag` (0 for `lag ≥ obs.time`, smoothly increasing for `lag < obs.time`),
/// so central FD at that exact point averages the two one-sided derivatives
/// (one of which is 0) instead of probing the analytic gradient's right-continuous
/// convention (`R_in` active from `tad ≥ 0⁺`) — a test artifact, not a provider bug.
#[test]
fn ode_provider_first_order_with_alag_matches_production() {
    let model = parse_model_string(FIRST_ORDER_ALAG_ODE).expect("parse");
    assert!(model.has_lagtime());
    let subj = bolus_subject(&[0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 4.0, 8.0]);
    assert!(!ode_subject_supported(&model, &subj));
    assert!(
        ode_tvcov_supported(&model, &subj),
        "first_order()+lagtime: the event-driven walk now carries R_in (#486)"
    );
    let theta = vec![5.0, 50.0, 1.0, 0.3];
    let eta = vec![0.1, -0.05];
    check_vs_production(&model, &subj, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
    // The sharpest 2nd-order check of the new onset saltation: `Δr = dose·ka` is
    // nonzero here (unlike IG/transit's vanishing onset), so `coef2`'s `rate_at_zero`
    // contribution is live. FOCEI's inner Newton step / Laplace `log|H̃|` consume
    // `d2f_deta2`/`d2f_deta_dtheta`, so validate them against FD of the analytic
    // 1st-order gradient (only value/1st-order is checked above).
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

// TRANSIT_ODE (depot → central, forcing on depot) with an estimated
// compartment-indexed lag on the *depot* (the forcing's own compartment). With
// `N > 0` (TVN = 3.0) the onset vanishes (any `n > 0` dominates to zero, mirroring
// IG), so this exercises the `Transit` `n > 0` arm of `rate_at_zero` end-to-end.
const TRANSIT_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVKA(1.0, 0.05, 20.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_N  ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  MTT   = TVMTT
  N     = TVN * exp(ETA_N)
  KA    = TVKA
  ALAG1 = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=N, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `transit()` + an estimated lag on the forcing's own compartment (#486): `N`
/// carries IIV, so the second-order `trigamma` forcing terms also flow through
/// the event-driven walk here (not just the static one). `N > 0` throughout, so
/// the onset saltation is the `rate_at_zero` zero-jump arm.
#[test]
fn ode_provider_transit_with_alag_matches_production() {
    let model = parse_model_string(TRANSIT_ALAG_ODE).expect("parse");
    assert!(model.has_lagtime());
    let subj = bolus_subject(&[0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    assert!(!ode_subject_supported(&model, &subj));
    assert!(
        ode_tvcov_supported(&model, &subj),
        "transit()+lagtime: the event-driven walk now carries R_in (#486)"
    );
    let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0, 0.3];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subj, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
    // 2nd-order check of the `rate_at_zero` zero-jump arm's saltation coefficient —
    // only value/1st-order is checked above, but FOCEI's inner Newton step / Laplace
    // `log|H̃|` consume the 2nd-order blocks.
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

/// `transit()` + an EVID 3/4 reset (#486, cell (b)): the static walk still
/// declines (no lagtime here, but the subject carries a reset), and the fix is
/// kind-agnostic, so transit inherits it exactly like igd/weibull.
#[test]
fn ode_provider_transit_with_reset_matches_production() {
    let model = parse_model_string(TRANSIT_ODE).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![10.0];
    let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0];
    let eta = vec![0.1, 0.05];
    assert!(
        ode_subject_supported(&model, &subject),
        "transit() + reset is shared scope for both outer and inner (#486)"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// `zero_order()` + an EVID 3/4 reset (#486 review follow-up): the kind-agnostic
/// gate relaxation in `ode_subject_supported` (dropped for every `InputRateKind`,
/// not just the "smooth density" forcings this PR's forcing loop touches) also
/// admits a pure `zero_order()` model. Its own moving-boundary cutoff is a
/// separate, pre-existing per-segment mechanism (`zero_windows`/`reset_floor`,
/// #530) untouched by this PR, so this closes an undocumented capability rather
/// than adding new machinery. The reset fires *inside* the first dose's open
/// window (`[0, DUR=4]` cut short at `t=2`) — the exact straddling case that
/// would leak pre-reset mass if `reset_floor` weren't already threaded through
/// `zero_windows`.
#[test]
fn ode_provider_zero_order_with_reset_matches_production() {
    const ZERO_ORDER_ODE_RESET: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(ZERO_ORDER_ODE_RESET).expect("parse");
    let mut subject = bolus_subject(&[1.0, 3.0, 4.0, 8.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![2.0];
    let theta = vec![5.0, 50.0, 4.0];
    let eta = vec![0.1, 0.05];
    assert!(
        ode_subject_supported(&model, &subject),
        "zero_order() + reset is shared scope for both outer and inner (#486, kind-agnostic gate)"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// `mixed` (zero-order + first-order) absorption + an EVID 3/4 reset (#486
/// review follow-up): the same kind-agnostic gate relaxation, now exercised for
/// the `mixed` combination — the `zero_order` term's window is what's exposed to
/// the reset-cutting hazard; the `first_order` term rides the general
/// smooth-density reset fix this PR adds.
#[test]
fn ode_provider_mixed_with_reset_matches_production() {
    const MIXED_ODE_RESET: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(3.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(MIXED_ODE_RESET).expect("parse");
    let mut subject = bolus_subject(&[1.0, 2.5, 4.0, 8.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.5, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.reset_times = vec![1.5];
    let theta = vec![5.0, 50.0, 0.4, 1.0, 3.0];
    let eta = vec![0.12, -0.08];
    assert!(
            ode_subject_supported(&model, &subject),
            "mixed absorption + reset is shared scope for both outer and inner (#486, kind-agnostic gate)"
        );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

// WEIBULL_ODE with an allometric weight covariate on CL — exercises cell (c):
// TV-cov + input-rate on the event-driven walk (no lagtime involved at all, so
// the dose's arrival is a fixed, non-dual boundary; the continuous forcing's
// `∂f/∂(TVTD,TVBETA,ETA_BETA)` must still match production under changing
// per-event PK params).
const WEIBULL_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `weibull()` + a time-varying covariate (#486, cell (c)): the event-driven walk
/// now carries `R_in`, hoisting the forcing's dose-invariant constants fresh per
/// segment as `CL` (hence the per-event PK snapshot) changes with `WT`.
#[test]
fn ode_provider_weibull_tvcov_matches_production() {
    let model = parse_model_string(WEIBULL_ODE_TVCOV).expect("parse");
    let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(70.0), wt(75.0), wt(80.0), wt(85.0)];
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "weibull()+TV-cov must be analytic on the event-driven walk (#486)"
    );
    let theta = vec![5.0, 50.0, 2.0, 1.5, 0.75];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

const ZERO_ORDER_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `zero_order(dur)` + a time-varying covariate (#486): the event-driven walk now
/// carries the moving-boundary window — the port of the static `integrate_g`'s
/// `zero_windows`. The constant `F·amt·frac/dur` rate is delivered as a per-segment
/// constant (`active_zero`, rebuilt per dose from `pk_at_dose[k]` as `CL` rides `WT`),
/// and the rate-off saltation at the moving end `d.time + dur` carries `∂/∂DUR` (hence
/// `∂/∂ETA_DUR` and `∂/∂TVDUR`). Value / `∂f/∂η` / `∂f/∂θ` must match the production
/// predictor + central FD, inner `Dual1` must reproduce the outer `Dual2` η-gradient,
/// and the 2nd-order blocks (the rate-off `coef2`) must match FD of the analytic gradient.
#[test]
fn ode_provider_zero_order_tvcov_matches_production() {
    let model = parse_model_string(ZERO_ORDER_ODE_TVCOV).expect("parse");
    // Obs times avoid landing on the window end `w_end ≈ TVDUR·exp(ETA_DUR) ≈ 4.2`, where
    // the prediction has a genuine kink in `dur`: central FD there would average the two
    // one-sided derivatives instead of probing the right-continuous analytic convention.
    let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(70.0), wt(75.0)];
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "zero_order()+TV-cov must be analytic on the event-driven walk (#486)"
    );
    let theta = vec![5.0, 50.0, 4.0, 0.75];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // The rate-off saltation's δdur² coefficient (`coef2`) is live here (`ETA_DUR`/`TVDUR`
    // carry IIV/θ into the moving boundary), so validate the analytic 2nd-order blocks
    // against FD of the analytic first-order gradient — the value/1st-order parity above
    // does not exercise `coef2`.
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

const ZERO_ORDER_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  DUR   = TVDUR * exp(ETA_DUR)
  ALAG1 = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `zero_order(dur)` + an estimated compartment-indexed lag (#486): the sharpest check of
/// the two coupled moving boundaries. The window is `[t_dose+lag, t_dose+lag+dur]`, so a
/// rate-on saltation (`Δr = F·amt/dur`, the same constant `active_zero` rate) fires at the
/// lagged arrival and a rate-off saltation (offset `δlag + δdur`) at the window end. Both
/// the value / `∂f/∂η` / `∂f/∂θ` (incl. `∂f/∂TVLAG`, `∂f/∂TVDUR`) and the 2nd-order blocks
/// must match the production predictor + FD, and the inner `Dual1` walk must reproduce the
/// outer `Dual2` η-gradient exactly.
///
/// Observation times avoid landing exactly on the lagged arrival (`lag = 0.3`) and the
/// window end (`lag + dur ≈ 4.5`), where the prediction has a genuine kink in `lag`/`dur`.
#[test]
fn ode_provider_zero_order_lagtime_matches_production() {
    let model = parse_model_string(ZERO_ORDER_ALAG_ODE).expect("parse");
    assert!(model.has_lagtime());
    let subj = bolus_subject(&[0.1, 0.2, 0.5, 1.0, 2.0, 6.0, 10.0]);
    assert!(
        !ode_subject_supported(&model, &subj),
        "lagtime routes off the static walk"
    );
    assert!(
        ode_tvcov_supported(&model, &subj),
        "zero_order()+lagtime: the event-driven walk now carries the moving window (#486)"
    );
    let theta = vec![5.0, 50.0, 4.0, 0.3];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subj, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
}

const MIXED_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// `mixed` (FZO1·first_order + FZO·zero_order) absorption + a time-varying covariate
/// (#653 review #1/#4). Two doses spaced closer than `DUR` overlap their zero-order
/// windows, so at each window end the *other* dose's window — plus both doses' still-
/// flowing first-order `R_in` — are concurrently active. The general rate-off saltation
/// must fold every concurrently-active forcing into its curvature term once the WT-driven
/// Jacobian jumps across the (non-record) window end; a forcing-only saltation would drop
/// those terms (`coef2` several-percent off). This is also the first regression covering
/// the advertised `mixed` pathway and its `frac` (FZO/FZO1) rate multiplier — both new
/// zero_order tests above use plain `zero_order(dur)` with `frac == 1`.
#[test]
fn ode_provider_mixed_tvcov_matches_production() {
    let model = parse_model_string(MIXED_ODE_TVCOV).expect("parse");
    // Window ends ≈ 4.2 and 6.2 (`DUR = TVDUR·exp(ETA_DUR)`); windows [0, 4.2] and
    // [2, 6.2] overlap on [2, 4.2]. Obs avoid both window ends (genuine `DUR` kinks) and
    // the dose times, and straddle each window end so WT (hence CL, hence the Jacobian)
    // changes across it.
    let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 8.0, 12.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0), wt(62.0)];
    subject.obs_covariates = vec![wt(65.0), wt(70.0), wt(75.0), wt(80.0), wt(85.0)];
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "mixed()+TV-cov must be analytic on the event-driven walk (#486/#653)"
    );
    let theta = vec![5.0, 50.0, 0.4, 1.0, 4.0, 0.75];
    let eta = vec![0.1, 0.05];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // The window ends fall between records where WT (hence the RHS Jacobian) changes, so
    // the general saltation's covariate-jump curvature term — fed by the concurrent
    // forcings — is live; validate the 2nd-order blocks against FD of the analytic
    // gradient (the value/1st-order parity above does not exercise `coef2`).
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

// ---- #880: analytic Hessian for IIV on an absorption lag feeding a rate-on
// `first_order` input-rate forcing (the `inject_rate_saltation` δ² time-partial term) ----

/// #880 regression. An η on a compartment lagtime (`ALAG1`) that shifts the onset of
/// a **decaying** `first_order` forcing lands the δlag² curvature correction in the
/// checked `d²f/∂η²` block. The old coefficient used only the state-Jacobian part
/// `½·J·(Δr·e)` and dropped the forcing's own onset time-variation `½·∂R_in/∂tad`
/// (`= −½·Δr·ka ≠ 0` for `first_order`), biasing the analytic Hessian to a near
/// sign-mirror of FD (`≈ +301` vs `≈ −304`). The 1st-order gradient was always
/// correct (`check_vs_production`), so predictions and the FOCEI *gradient* were fine;
/// only the curvature (SEs, Hessian-consuming line searches) was wrong.
#[test]
fn ode_provider_880_rate_on_first_order_lag_iiv_hessian() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(2.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  ALAG1 = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(M).expect("parse #880 model");
    let subject = bolus_subject(&[0.5, 1.0, 3.0, 5.0]);
    assert!(
        ode_analytical_supported(&m),
        "first_order + compartment ALAG1 must be on the analytic event-driven walk"
    );
    let theta = [1.0, 20.0, 1.2, 2.0];
    let eta = [0.1, 0.05];
    check_vs_production(&m, &subject, &theta, &eta); // 1st order was always fine
    check_hessian_vs_fd_of_grad(&m, &subject, &theta, &eta); // FAILED pre-#880
}

/// #880 twin: the same rate-on onset with a **time-varying covariate crossing it**.
/// `KA` (hence the onset value `Δr` and slope `∂R_in/∂tad`) carries a WT covariate that
/// changes record-to-record, and `ALAG1` carries IIV — so the onset saltation runs on
/// the TV-cov `(θ,η)`-basis walk with a per-segment PK snapshot. Guards that the δ²
/// time-partial term is fed the right per-onset `Δr`/`∂R_in/∂tad` (and that the onset
/// snapshot is self-consistent) under a covariate that moves across the onset — the
/// scenario #880 flagged as the `K_ROUTE_ONSET` snapshot concern.
#[test]
fn ode_provider_880_rate_on_first_order_lag_iiv_tvcov_hessian() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  theta KA_WT(0.01, -0.1, 0.1)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA * exp(KA_WT * (WT - 70))
  ALAG1 = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(M).expect("parse #880 TV-cov model");
    // Onset at t_dose + lag ≈ 1.5; observations straddle it (some before, some after),
    // and WT changes at each record so the RHS Jacobian (via KA) moves across the onset.
    let times = [0.75, 1.0, 2.0, 3.0, 5.0, 8.0];
    let mut subject = bolus_subject(&times);
    subject.dose_covariates = vec![HashMap::from([("WT".to_string(), 70.0)])];
    subject.obs_covariates = (0..times.len())
        .map(|i| HashMap::from([("WT".to_string(), 60.0 + 8.0 * (i as f64))]))
        .collect();
    assert!(
        subject.has_tv_covariates(),
        "WT must register as time-varying (per-obs values differ)"
    );
    assert!(
        ode_tvcov_supported(&m, &subject),
        "first_order + ALAG1 + TV-cov must be on the analytic TV-cov walk"
    );
    let theta = [1.0, 20.0, 1.2, 1.5, 0.01];
    let eta = [0.1, 0.05];
    check_vs_production(&m, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&m, &subject, &theta, &eta);
}

// ---- #880 × #859: the same rate-on onset δ² fix, on the PER-ROUTE lag `K_ROUTE_ONSET`
// path (analytic since #875). #875's own per-route tests keep the route lag η-free (η on
// ETA_CL), so their δlag² term lands in the unchecked d2f_dtheta2 region; these put an η
// ON the route lag so it lands in the FD-checked d2f_deta2 block — the gap #880 named. ----

/// #880: IIV on a per-route absorption lag (`first_order(ka=KA, lag=LAG)`, `LAG` carrying
/// an η). The onset is injected at `K_ROUTE_ONSET` (not the shared `K_DOSE`), so this guards
/// that handler's `dr_dtad` (`∂R_in/∂tad`) curvature term. Fails pre-fix identically to the
/// compartment-lag case (the issue confirms the shared `inject_rate_saltation` origin).
#[test]
fn ode_provider_880_route_lag_iiv_hessian() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(M).expect("parse #880 route-lag model");
    let ode = m.ode_spec.as_ref().expect("ode spec");
    assert_eq!(
        ode.input_rate
            .iter()
            .filter(|f| f.lag_slot.is_some())
            .count(),
        1,
        "the forcing carries a per-route lag"
    );
    let subject = bolus_subject(&[0.5, 1.0, 3.0, 5.0]);
    assert!(
        ode_tvcov_supported(&m, &subject),
        "per-route lag routes to the event-driven walk (K_ROUTE_ONSET)"
    );
    let theta = [1.0, 20.0, 1.2, 1.5];
    let eta = [0.1, 0.05];
    check_vs_production(&m, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&m, &subject, &theta, &eta);
}

/// #880 twin: IIV on a per-route lag with a TV covariate crossing the route onset. Guards
/// the `K_ROUTE_ONSET` onset-segment snapshot (kernel `ka`/`frac`/post-side Jacobian read
/// from the post-arrival record, not the pre-onset `last_params` nor the dose snapshot).
#[test]
fn ode_provider_880_route_lag_iiv_tvcov_hessian() {
    const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  theta KA_WT(0.01, -0.1, 0.1)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA * exp(KA_WT * (WT - 70))
  LAG = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let m = parse_model_string(M).expect("parse #880 route-lag TV-cov model");
    let times = [0.75, 1.0, 2.0, 3.0, 5.0, 8.0];
    let mut subject = bolus_subject(&times);
    subject.dose_covariates = vec![HashMap::from([("WT".to_string(), 70.0)])];
    subject.obs_covariates = (0..times.len())
        .map(|i| HashMap::from([("WT".to_string(), 60.0 + 8.0 * (i as f64))]))
        .collect();
    assert!(subject.has_tv_covariates(), "WT is time-varying");
    assert!(
        ode_tvcov_supported(&m, &subject),
        "per-route lag + TV-cov routes to the analytic event-driven walk"
    );
    let theta = [1.0, 20.0, 1.2, 1.5, 0.01];
    let eta = [0.1, 0.05];
    check_vs_production(&m, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&m, &subject, &theta, &eta);
}

/// #867: a nonlinear (MM) disposition that **admits** a periodic steady state (mean input
/// `100/12 ≈ 8.3 < VM·e^{η} ≈ 16.6`) is equilibrated by the Anderson-accelerated fixed-point
/// solve, and its **analytic dual** value + `∂f/∂η` + `∂f/∂θ` must match the production
/// predictor and finite differences. This is the teeth on the AA-over-dual gradient path (the
/// mixing coefficients, chosen on the value residual, must carry the correct implicit derivative
/// through the whole `[η, θ]` jet), exercised end-to-end through the real ODE — distinct from the
/// no-steady-state fallback covered by `..._nonlinear_disposition_matches_production` above.
#[test]
fn ode_provider_ss_absorption_nonlinear_converged_dual_matches_production() {
    let model = parse_model_string(MM_SS_FIRST_ORDER).expect("parse MM SS first_order");
    // SS-admitting thetas (VM=15, KM=300, KA=1) with a longer II so mean input < VM.
    let theta = vec![15.0, 300.0, 1.0];
    let eta = vec![0.1];
    let subj = ss_absorption_subject(&[1.0, 3.0, 6.0, 9.0, 11.9], 12.0);
    check_vs_production(&model, &subj, &theta, &eta);
    // Prove Anderson converged (not the linear 1-cycle fast path, not the 50-cycle cap).
    let _ = ode_subject_sensitivities(&model, &subj, &theta, &eta).expect("supported");
    let cycles = crate::dosing::last_ss_equilibration_cycles();
    assert!(
        (2..crate::dosing::SS_EQUILIBRATION_CYCLES).contains(&cycles),
        "a converging nonlinear SS must be solved by Anderson (2..cap cycles), got {cycles}"
    );
}

// ---------------------------------------------------------------------------
// #971: bucketed ODE IOV dual-width dispatch
// ---------------------------------------------------------------------------

/// 1-cpt IV with κ on CL — the cheapest fixture that can be widened to an arbitrary
/// stacked width purely by adding occasion groups (`m_dim = n_theta + n_eta + K·n_kappa`),
/// with no SS equilibration or absorption in the way.
const ONECPT_IV_IOV_WIDE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// A subject with `k_groups` occasions (one observation each) and a single bolus, so the
/// stacked IOV width is `n_eta + k_groups·n_kappa` by construction.
fn iov_wide_subject(k_groups: usize) -> Subject {
    let times: Vec<f64> = (1..=k_groups).map(|i| i as f64).collect();
    let mut s = bolus_subject(&times);
    s.occasions = (1..=k_groups as u32).collect();
    s.dose_occasions = vec![1];
    s
}

/// Every `ObsSens` field, bit-for-bit — padding a dual with zero lanes must not perturb a
/// single ULP of the live block (`assert_eq!` on `f64` is exact; the `to_bits` check pins
/// `+0.0` vs `−0.0` too, which `==` would conflate).
fn assert_subject_sens_bit_identical(a: &SubjectSens, b: &SubjectSens, what: &str) {
    assert_eq!(
        a.obs.len(),
        b.obs.len(),
        "{what}: observation count differs"
    );
    assert!(!a.obs.is_empty(), "{what}: no observations to compare");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (j, (x, y)) in a.obs.iter().zip(b.obs.iter()).enumerate() {
        assert_eq!(
            x.f.to_bits(),
            y.f.to_bits(),
            "{what}: obs {j} value differs"
        );
        assert_eq!(bits(&x.df_deta), bits(&y.df_deta), "{what}: obs {j} ∂f/∂η");
        assert_eq!(
            bits(&x.df_dtheta),
            bits(&y.df_dtheta),
            "{what}: obs {j} ∂f/∂θ"
        );
        assert_eq!(
            bits(&x.d2f_deta2),
            bits(&y.d2f_deta2),
            "{what}: obs {j} ∂²f/∂η²"
        );
        assert_eq!(
            bits(&x.d2f_deta_dtheta),
            bits(&y.d2f_deta_dtheta),
            "{what}: obs {j} ∂²f/∂η∂θ"
        );
    }
    assert!(
        a.obs.iter().any(|o| o.f.abs() > 0.0)
            && a.obs.iter().any(|o| o.df_deta.iter().any(|g| *g != 0.0)),
        "{what}: reference curve/gradient must be non-trivial, else the comparison is vacuous"
    );
}

/// The bucket ladder's contract, pinned on the real const (#971): exact through 24, rounded
/// up past it, and `None` — the FD route — outside `1..=MAX_ODE_IOV_AXES`.
#[test]
fn ode_iov_width_buckets_round_up_and_decline_past_the_cap() {
    use crate::sens::widths::bucket_for;
    let l = &ODE_IOV_WIDTH_BUCKETS;

    // The ordinary IOV model is served exactly — no padding cost at all up to the width the
    // `MAX_ODE_AXES` / `MAX_SCALE_AXES` ladders already instantiate.
    for w in 1..=24 {
        assert_eq!(bucket_for(w, l), Some(w), "widths 1..=24 must stay exact");
    }
    // Past 24 the tail rounds up. The step holds the `O(M²)` outer padding penalty near
    // 1.5×; a 33-axis subject runs 40 lanes, not the 48 a coarser ladder would give it.
    assert_eq!(bucket_for(25, l), Some(28));
    assert_eq!(bucket_for(29, l), Some(32));
    assert_eq!(bucket_for(33, l), Some(40));
    assert_eq!(bucket_for(41, l), Some(48));
    assert_eq!(bucket_for(49, l), Some(56));
    assert_eq!(bucket_for(57, l), Some(64));
    assert_eq!(bucket_for(65, l), Some(80));
    assert_eq!(bucket_for(81, l), Some(96));
    assert_eq!(bucket_for(MAX_ODE_IOV_AXES, l), Some(MAX_ODE_IOV_AXES));

    // No round-up may cost more than ~1.5× the exact `Dual2<M>` Hessian area.
    for w in 1..=MAX_ODE_IOV_AXES {
        let b = bucket_for(w, l).expect("in-cap width");
        let area = (b * b) as f64 / (w * w) as f64;
        assert!(
            area <= 1.55,
            "width {w} → bucket {b} pads {area:.2}× the M×M work"
        );
    }

    // Past the last bucket there is no instantiation to dispatch to: the ladder declines,
    // and `ode_iov_subject_supported` declines first (see the gate test below), so such a
    // subject reaches FD by the loud route, never by a silent `_ => None`.
    assert_eq!(bucket_for(MAX_ODE_IOV_AXES + 1, l), None);
    assert_eq!(
        bucket_for(0, l),
        None,
        "a zero-axis subject has nothing to seed"
    );

    // Every bucket is reachable as its own round-up target, and the ladder covers the whole
    // gate range with no hole that would drop to FD.
    for w in 1..=MAX_ODE_IOV_AXES {
        let b = bucket_for(w, l).expect("every in-cap width must have a bucket");
        assert!(b >= w && l.contains(&b), "width {w} → bucket {b}");
    }
}

/// The teeth on the bucketing itself: for a subject whose stacked width is *not* a bucket
/// boundary, the padded instantiation the ladder now selects must reproduce the exact-width
/// instantiation (what the pre-#971 ladder ran) bit-for-bit — outer `Dual2` sensitivities
/// and inner `Dual1` η-gradient alike. Padded lanes stay zero, so they can neither leak into
/// the live block nor change the value-only ODE step control.
#[test]
fn ode_iov_bucketed_dispatch_is_bit_identical_to_the_exact_width() {
    // `Dual2<24>` carries a 24×24 Hessian per value and the ODE walk holds several frames
    // at once — past the 2 MiB default test-thread stack. Production runs fits on a 32 MiB
    // Rayon stack for exactly this reason; mirror it rather than shrinking the fixture
    // below the widths the test is about (same pattern as
    // `provider_tests::ode_iov_above_legacy_axis_cap_stays_analytic`).
    std::thread::Builder::new()
        .stack_size(crate::api::FIT_RAYON_STACK_SIZE)
        .spawn(ode_iov_bucketed_dispatch_body)
        .expect("spawn wide-stack test thread")
        .join()
        .expect("bucketed-vs-exact IOV comparison panicked");
}

fn ode_iov_bucketed_dispatch_body() {
    let model = parse_model_string(ONECPT_IV_IOV_WIDE).expect("parse wide-κ IOV");
    let theta = vec![1.0, 20.0];
    // 24 occasions ⇒ n_stacked = 25 (→ bucket 28) and m_dim = 27 (→ bucket 28): neither
    // width is a bucket boundary, which is the whole point of the fixture.
    let subj = iov_wide_subject(24);
    let occ_groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(
        occ_groups.len(),
        24,
        "fixture must have 24 κ occasion groups"
    );

    let n_stacked = model.n_eta + occ_groups.len() * model.n_kappa;
    let m_dim = model.n_theta + n_stacked;
    assert_eq!(
        (m_dim, n_stacked),
        (27, 25),
        "fixture pins the two widths this test is about"
    );
    assert_eq!(
        crate::sens::widths::bucket_for(m_dim, &ODE_IOV_WIDTH_BUCKETS),
        Some(28)
    );
    assert_eq!(
        crate::sens::widths::bucket_for(n_stacked, &ODE_IOV_WIDTH_BUCKETS),
        Some(28)
    );

    let stacked: Vec<f64> = (0..n_stacked)
        .map(|k| 0.08 - 0.011 * k as f64)
        .collect::<Vec<_>>();

    // Outer (`Dual2`): exact width 27 (what the pre-#971 ladder ran) vs the padded width 28
    // the bucketed ladder picks.
    let exact = run_subject_iov::<27>(&model, &subj, &theta, &stacked, &occ_groups)
        .expect("exact-width IOV walk");
    let padded = run_subject_iov::<28>(&model, &subj, &theta, &stacked, &occ_groups)
        .expect("bucketed IOV walk");
    assert_subject_sens_bit_identical(&exact, &padded, "outer 27 vs 28");
    // …and the public entry point, which is what actually goes through the ladder.
    let dispatched = ode_subject_sensitivities_iov(&model, &subj, &theta, &stacked)
        .expect("in-scope IOV subject must stay analytic");
    assert_subject_sens_bit_identical(&exact, &dispatched, "outer 27 vs dispatched");

    // Inner (`Dual1`): exact width 25 vs the padded width 28.
    let exact_g = run_subject_iov_eta::<25>(&model, &subj, &theta, &stacked, &occ_groups)
        .expect("exact-width IOV η-gradient");
    let padded_g = run_subject_iov_eta::<28>(&model, &subj, &theta, &stacked, &occ_groups)
        .expect("bucketed IOV η-gradient");
    let dispatched_g = ode_subject_eta_grad_iov(&model, &subj, &theta, &stacked)
        .expect("in-scope IOV subject must stay analytic");
    assert_eq!(exact_g.len(), padded_g.len());
    for (j, ((x, y), z)) in exact_g
        .iter()
        .zip(padded_g.iter())
        .zip(dispatched_g.iter())
        .enumerate()
    {
        assert_eq!(
            x.f.to_bits(),
            y.f.to_bits(),
            "inner: obs {j} value (padded)"
        );
        assert_eq!(
            x.f.to_bits(),
            z.f.to_bits(),
            "inner: obs {j} value (dispatched)"
        );
        for k in 0..n_stacked {
            assert_eq!(
                x.df_deta[k].to_bits(),
                y.df_deta[k].to_bits(),
                "inner: obs {j} ∂f/∂η[{k}] (padded)"
            );
            assert_eq!(
                x.df_deta[k].to_bits(),
                z.df_deta[k].to_bits(),
                "inner: obs {j} ∂f/∂η[{k}] (dispatched)"
            );
        }
    }
    assert!(
        exact_g
            .iter()
            .any(|o| o.df_deta.iter().any(|g| g.abs() > 1e-12)),
        "η-gradient must be non-trivial, else the comparison is vacuous"
    );
}

/// A stacked width past the last bucket must decline **at the gate**, so the caller drops to
/// FD by the same loud route it always did — not by silently falling through the ladder's
/// `_ => None`. Both IOV entry points are checked; the observable consequence is `None`
/// (→ FD) plus the inner router's attribution string naming the cap, not a panic and not a
/// wrong-width analytic answer.
#[test]
fn ode_iov_subject_past_the_last_bucket_declines_to_fd() {
    let model = parse_model_string(ONECPT_IV_IOV_WIDE).expect("parse wide-κ IOV");
    let theta = vec![1.0, 20.0];
    // n_theta(2) + n_eta(1) + K·n_kappa(1) > 96 ⇒ K ≥ 94.
    let subj = iov_wide_subject(94);
    let occ_groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    let n_stacked = model.n_eta + occ_groups.len() * model.n_kappa;
    let m_dim = model.n_theta + n_stacked;
    assert!(
        m_dim > MAX_ODE_IOV_AXES,
        "fixture must exceed the cap, got {m_dim}"
    );

    let stacked = vec![0.01; n_stacked];
    assert!(
        ode_iov_subject_supported(&model, &subj).is_none(),
        "a subject past the cap must be declined by the gate, before any dispatch"
    );
    assert!(
        ode_subject_sensitivities_iov(&model, &subj, &theta, &stacked).is_none(),
        "outer IOV must decline past the cap"
    );
    assert!(
        ode_subject_eta_grad_iov(&model, &subj, &theta, &stacked).is_none(),
        "inner IOV must decline past the cap"
    );
}

// #1020: a Form C readout whose value legitimately goes negative — here a
// concentration expressed as a change from a baseline (`central/V1 - TVBASE`),
// the same shape as the `sqrt(N)*logit(p)` readout of a model-based
// meta-analysis. The overshoot clamp used to zero the negative part in both the
// f64 predictor and the dual walk.
const TWOCPT_ODE_SIGNED_READOUT: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBASE(2.0, 0.0, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma ADD_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 - TVBASE
[error_model]
  DV ~ additive(ADD_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// Regression for #1020: the negative part of a Form C `[scaling]` readout must
/// survive both the f64 predictor and the `Dual2`/`Dual1` walks, and the two must
/// still agree — the clamp gate is shared, so a gate that drifted between them
/// would show up as an analytic/FD mismatch on the negative observations (where a
/// clamped jet carries zero derivatives and an unclamped one does not).
#[test]
fn ode_provider_signed_form_c_readout_not_clamped_and_matches_production() {
    let model = parse_model_string(TWOCPT_ODE_SIGNED_READOUT).expect("parse");
    let theta = vec![4.0, 12.0, 2.0, 25.0, 2.0];
    let eta = vec![0.12, -0.08];
    // Late times sit below the 2.0 baseline; early ones above it.
    let times = [0.25, 1.0, 4.0, 12.0, 24.0];
    let subj = bolus_subject(&times);

    let preds = compute_predictions_with_tv(&model, &subj, &theta, &eta);
    assert!(
        preds.last().copied().unwrap() < -0.5,
        "fixture must produce a genuinely negative readout, got {preds:?}"
    );
    assert!(preds[0] > 0.0, "fixture must straddle zero, got {preds:?}");

    check_vs_production(&model, &subj, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
}

/// A fast-binding (TMDD-shaped) system under `ode_method = auto`. The probe reads the
/// post-dose Jacobian as stiff and starts Rodas4, so this is the analytic-sensitivity path
/// running on a *different stepper* than the one every other ODE parity test exercises.
const THREE_STATE_BINDING_AUTO: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVVC(3.0, 0.1, 100.0)
  theta TVKON(60.0, 1e-3, 1e4)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  VC = TVVC
  KON = TVKON
  KOFF = 20.0
  KINT = 0.5
  R0 = 10.0
[structural_model]
  ode(obs_cmt=central, states=[central, target, complex])
[odes]
  init(target) = R0
  d/dt(central) = -(CL/VC) * central - KON * central * target + KOFF * complex
  d/dt(target)  = -KON * central * target + KOFF * complex - 0.05 * target + 0.5
  d/dt(complex) =  KON * central * target - KOFF * complex - KINT * complex
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = auto
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// CLAUDE.md's `Dual2`-vs-FD rule, applied to the stepper `ode_method = auto` switches into
/// (#978).
///
/// `auto` is resolved inside the drivers, so the `T = Dual2` sensitivity solve and the
/// `T = f64` prediction each probe their own segment and each pick a method. Nothing outside
/// the driver coordinates them: the guarantee that they agree rests entirely on the probe
/// reading `.val()` only, and a break in it would show up here — and *only* here — as an
/// analytic gradient differentiating a trajectory the predictor never produced. That failure
/// compiles, runs, and returns a plausible number, which is why it needs a parity test rather
/// than an inspection.
#[test]
fn ode_provider_matches_fd_under_the_auto_stiff_switch() {
    let model = parse_model_string(THREE_STATE_BINDING_AUTO).expect("parse");
    assert_eq!(
        model.ode_spec.as_ref().unwrap().solver_opts.method,
        crate::ode::OdeMethod::Auto,
        "the fit option must reach the solver, or this test pins the default stepper"
    );
    assert!(
        ode_analytical_supported(&model),
        "must be served analytically, not silently dropped to FD"
    );

    let theta = vec![0.2, 3.0, 60.0];
    let eta = vec![0.15];
    // Sampled across the fast binding phase and out into the slow terminal one.
    let subject = bolus_subject(&[0.02, 0.1, 0.5, 1.0, 4.0, 12.0, 24.0]);

    // Non-vacuity: the probe must actually have escalated on this fixture. Without this the
    // parity checks below would pass by exercising plain RK45 and pin nothing about `auto`.
    let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
    let (_preds, stats) = crate::ode::ode_predictions_with_solver_stats(
        model.ode_spec.as_ref().unwrap(),
        &pk.values,
        &theta,
        &eta,
        &subject,
    );
    assert!(
        stats.auto_stiff_segments > 0,
        "fixture must read stiff, got {stats:?}"
    );
    assert_eq!(
        stats.auto_stiff_rejected, 0,
        "and the escalation must hold up, got {stats:?}"
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
}

// 1-cpt IV **infusion** at steady state with an estimated lagtime and a
// time-varying covariate on CL. Sized so `II − ALAG < T_inf`: the previous
// cycle's infusion is still running at the dose record (#1121).
const ONECPT_IV_LAG_SS_INF_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0,  0.01, 100.0)
  theta TVV(10.0,  1.0, 500.0)
  theta TVLAG(8.0, 0.01, 20.0)
  theta WTEXP(0.75, 0.01, 2.0)
  omega ETA_CL  ~ 0.1
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL    = TVCL * (WT/70)^WTEXP * exp(ETA_CL)
  V     = TVV
  ALAG1 = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

const ONECPT_ORAL_LAG_SS_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0,  0.01, 100.0)
  theta TVV(10.0,  1.0, 500.0)
  theta TVKA(1.0,  0.01, 50.0)
  theta TVLAG(0.5, 0.01, 5.0)
  theta WTEXP(0.75, 0.01, 2.0)
  omega ETA_CL  ~ 0.1
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL    = TVCL * (WT/70)^WTEXP * exp(ETA_CL)
  V     = TVV
  KA    = TVKA * (WT/70)^WTEXP
  ALAG1 = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **SS × estimated lagtime × time-varying covariates — the cell that had no test
/// (#1121).**
///
/// `ode_provider_ss_lagtime_matches_production` covers SS × lagtime with *flat*
/// covariates; `ode_provider_ss_tvcov_matches_production` covers SS × TV
/// covariates with *no* lagtime. Their intersection — where the pre-arrival window
/// `[t_dose, t_dose + ALAG)` straddles a covariate change — was uncovered, and it
/// is precisely where production and the dual walk both used to load the periodic
/// trough *at the arrival* under the dose row's snapshot instead of seeding it at
/// the dose record and flowing it forward.
///
/// **Read this before trusting it as a regression test.** Before #1121 both sides
/// carried that defect, so this passed while both were wrong; after the joint fix
/// it passes because both are right. It is therefore a **coupling guard, not an
/// oracle** — its job is to go red the moment production and the twin are fixed
/// out of step, which is the one failure a value-vs-value comparison between two
/// implementations of the same wrong idea can actually detect. The oracle for the
/// behaviour itself is NONMEM: `tests/dose_form_lag_nonmem_anchor.rs` (pointwise
/// `PRED`) and `ss_state_at_phase_matches_nonmem_at_the_dose_record` (the seed
/// alone).
///
/// Mutation-tested when it was added. Restoring the twin's `K_DOSE` trough
/// overwrite with production left fixed turns this red **on `obs.f`** — the value
/// assertion — while the flat-covariate `ode_provider_ss_lagtime{,_infusion}_…`
/// tests go red only on their derivative blocks. That split is the whole reason
/// this cell exists: a value divergence between the two walks is invisible until
/// a covariate changes inside the pre-arrival window.
#[test]
fn ode_provider_ss_lagtime_tvcov_matches_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_SS_TVCOV_ODE).expect("parse oral lag SS TV ODE");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    // θ = [TVCL, TVV, TVKA, TVLAG, WTEXP]; η = [ETA_CL, ETA_LAG] → ALAG ≈ 0.526.
    let theta = [1.0, 10.0, 1.0, 0.5, 0.75];
    let eta = [0.12_f64, 0.05];

    // Obs at 0.3 sits INSIDE the pre-arrival window; 0.8 just past the arrival;
    // the rest span the remainder of the cycle and into the next.
    let mut subject = bolus_subject(&[0.3, 0.8, 2.0, 5.0, 11.0, 13.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    // The SS record carries WT = 70; every later record carries 140 or more. The
    // window `(0, ALAG]` is therefore seeded under 70 and propagated under 140 —
    // a 2× contrast on both CL and KA.
    subject.dose_covariates = vec![wt(70.0)];
    subject.obs_covariates = vec![
        wt(140.0),
        wt(140.0),
        wt(140.0),
        wt(150.0),
        wt(150.0),
        wt(75.0),
    ];

    // Preconditions, asserted rather than assumed — this fixture is worth nothing
    // unless all three ingredients are simultaneously live.
    assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
    assert!(model.has_lagtime());
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS + lagtime + TV covariates must take the analytic event-driven walk, \
         not the FD fallback — otherwise this compares FD against itself"
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// An **SS infusion** whose lagtime pushes the seed phase inside its own window,
/// under time-varying covariates (#1121 review).
///
/// `ode_provider_ss_lagtime_infusion_matches_production` covers the infusion seed
/// with *flat* covariates, and there the two `ss_seeded_at_record`-scoped terms
/// this walk carries — the bolus saltation's `g⁻` and the infusion's hand-injected
/// `−δlag` jet — cancel exactly. A flat test therefore pins the cancellation
/// rather than either term, so the scoping change needed a cell where they do not
/// cancel.
///
/// It is also the only twin test where `II − ALAG < T_inf`, so the previous
/// cycle's infusion is still running at the dose record and the walk has to carry
/// a rate-off boundary (`K_SS_INF_END`) that belongs to no dose event — one that
/// moves with `lag` and with the window length exactly as the real infusion end
/// does, and therefore needs its own saltation, not just its own break.
///
/// `check_vs_production` asserts `obs.f` as well as the derivatives, so this is
/// simultaneously the value-coupling guard for the residual window and the
/// `Dual2`-vs-FD parity check for its moving edge.
#[test]
fn ode_provider_ss_lagtime_infusion_tvcov_matches_production() {
    let model =
        parse_model_string(ONECPT_IV_LAG_SS_INF_TVCOV_ODE).expect("parse IV lag SS inf TV ODE");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    // θ = [TVCL, TVV, TVLAG, WTEXP]; η = [ETA_CL, ETA_LAG] → ALAG ≈ 8.4.
    let theta = [1.0, 10.0, 8.0, 0.75];
    let eta = [0.12_f64, 0.05];

    // AMT 100 at RATE 25 => T_inf = 4 against II = 12, so the seed phase
    // `II − ALAG ≈ 3.6` falls INSIDE the window and the previous cycle's infusion
    // runs on into the pre-arrival window.
    let mut subject = bolus_subject(&[0.2, 1.0, 4.0, 8.0, 9.5, 13.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 25.0, true, 12.0)];
    subject.dose_covariates = vec![wt(70.0)];
    subject.obs_covariates = vec![
        wt(70.0),
        wt(140.0),
        wt(140.0),
        wt(150.0),
        wt(150.0),
        wt(75.0),
    ];

    // Preconditions, asserted rather than assumed. The third is the one this
    // fixture exists for: without it the previous cycle has already finished at
    // the record and the residual window is empty, leaving the new code unrun.
    assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
    assert!(subject.doses[0].is_infusion() && model.has_lagtime());
    let lag = theta[2] * eta[1].exp();
    let t_inf = subject.doses[0].duration;
    let phase = crate::dosing::ss_seed_phase(&subject.doses[0], lag);
    assert!(
        phase < t_inf,
        "the seed phase ({phase}) must fall inside the infusion window ({t_inf}), or \
         the previous cycle has finished at the record and this tests nothing new"
    );
    assert!(
        subject.obs_times[0] < lag,
        "the first sample must sit inside the pre-arrival window"
    );
    assert!(subject.has_tv_covariates());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "SS + lagtime + infusion + TV covariates must take the analytic walk"
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

/// An **EVID=3 reset between a seeded SS dose's record and its lagged arrival**
/// (#1121 review).
///
/// With the arrival-side re-equilibration gone, a reset inside the pre-arrival
/// window is newly observable: it zeroes the seeded steady-state load, and the
/// arrival then applies only `F·AMT` where it used to restore a full trough. That
/// reading is the one consistent with the convention — the steady state is loaded
/// at the record, so a later reset wipes it exactly as it wipes any other state —
/// but it is a behaviour change, and nothing covered it.
///
/// Deliberately an oral **bolus**, not the infusion fixture above. The infusion
/// form of this geometry trips a *separate*, pre-existing defect in the twin: an
/// `EVID=3` reset placed before a lagged infusion's arrival changes `∂f/∂η_LAG`
/// even when the reset is a physical no-op (the state is already zero). Measured
/// at `−12.53` against an FD reference of `+7.14`, with flat covariates and a
/// non-steady-state dose, so it is neither this issue's nor time-varying
/// covariates'; tracked separately. Using a bolus here keeps this test measuring
/// the thing it was written for instead of inheriting that failure.
#[test]
fn ode_provider_ss_lagtime_reset_inside_the_pre_arrival_window_matches_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_SS_TVCOV_ODE).expect("parse oral lag SS TV ODE");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    // θ = [TVCL, TVV, TVKA, TVLAG, WTEXP]; η = [ETA_CL, ETA_LAG] → ALAG ≈ 0.526.
    let theta = [1.0, 10.0, 1.0, 0.5, 0.75];
    let eta = [0.12_f64, 0.05];

    let mut subject = bolus_subject(&[0.2, 1.0, 4.0, 11.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    subject.reset_times = vec![0.35];
    subject.dose_covariates = vec![wt(70.0)];
    subject.obs_covariates = vec![wt(70.0), wt(140.0), wt(150.0), wt(75.0)];

    let lag = theta[3] * eta[1].exp();
    assert!(
        subject.reset_times[0] > subject.doses[0].time && subject.reset_times[0] < lag,
        "the reset ({}) must fall strictly between the dose record (0) and its arrival ({lag})",
        subject.reset_times[0]
    );
    assert!(
        subject.obs_times[0] < subject.reset_times[0],
        "one sample must precede the reset, so the seeded state is read before it is wiped"
    );
    assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
    assert!(subject.has_tv_covariates());
    assert!(ode_tvcov_supported(&model, &subject));

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
}

// ---------------------------------------------------------------------------
// #1070 — `TAD` is threaded through the sensitivity walks as a **dual**.
//
// `TAD`'s anchor IS the dose's lagged arrival, so `∂TAD/∂lag = −1` belongs in
// the dual chain. While `eval_rhs_anchored` lifted it as an `f64` constant that
// term was missing everywhere the trajectory is integrated, not only at a
// boundary — and the value stayed exact, so only a gradient-vs-FD check could
// see it. An interim fix routed the whole cell to FD; the term is now carried
// and that detour is gone.
//
// The tests below pin BOTH directions. The four shapes that carry the term
// (η-lag, θ-only lag, per-compartment `ALAG{n}`, IOV κ-on-lag) assert first-order
// exactness against FD of the production predictor. The neighbours that never
// carried it — `TAFD`, a `TAD` RHS with no lagtime, a pure route lag — assert
// they still take the analytic route, so a later over-widening of scope fails
// loudly. `TAD` + steady state remains FD under the separate non-autonomous-RHS
// rule and is pinned by `ode_provider_ss_tad_dependent_rhs_routes_to_fd`.
//
// Second order is NOT asserted here; see
// `ode_tad_rhs_with_estimated_lagtime_is_analytic_and_exact` for why (#1075).
// ---------------------------------------------------------------------------

/// 1-cpt IV whose RHS reads a time built-in, with a configurable lagtime
/// expression. `{TIMEVAR}` / `{LAG}` are substituted per test.
fn tad_gate_model(time_var: &str, lag_expr: &str) -> CompiledModel {
    let lag_line = if lag_expr.is_empty() {
        String::new()
    } else {
        format!("  LAGTIME = {lag_expr}\n")
    };
    let src = format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
{lag_line}[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.3 * {time_var})
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rk45
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#
    );
    parse_model_string(&src).expect("parse")
}

/// Bolus at t=0 plus a 1 h infusion at t=1, observations all strictly after the
/// first lagged arrival (~0.80). That was originally required because a record before
/// the first arrival poisoned the trajectory with `TAD = NaN`; #1073 anchors the
/// pre-arrival window at the subject's first arrival in both ODE predictors and in the
/// event-driven twin, so it is finite there.
///
/// NOT in the static twin: `integrate_g` still folds into `NEG_INFINITY` with no fallback
/// and injects `TAD = NaN` before the first dose (measured; see #1110). That walk is
/// unreachable for THIS fixture — `ode_subject_supported` declines `has_lagtime()` — which
/// is exactly why no `TAD` fixture in this file has ever exercised it. The geometry is kept: this fixture is about the
/// #1070 gate, and moving a record into that window would make it about #1110's
/// semantics instead.
fn tad_gate_subject() -> Subject {
    let mut s = bolus_subject(&[0.9, 1.5, 2.0, 4.0, 8.0]);
    let mut inf = DoseEvent::new(1.0, 60.0, 1, 0.0, false, 0.0);
    inf.duration = 1.0;
    inf.rate = 60.0;
    s.doses.push(inf);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    s.dose_covariates = vec![wt(60.0), wt(65.0)];
    s.obs_covariates = vec![wt(60.0), wt(70.0), wt(75.0), wt(80.0), wt(90.0)];
    s
}

/// **#1070, the defect this fix removes.** `TAD`'s anchor IS the dose's lagged
/// arrival, so `∂TAD/∂lag = −1` belongs in the dual chain. While `tad` was lifted
/// as an `f64` constant that term was missing everywhere the trajectory is
/// integrated — not only at a dose event — and the model was admitted as analytic
/// while returning it. Measured on this exact fixture at `df8bb8d6`, against FD of
/// the production predictor: worst `∂f/∂η` **6.6e-1** and worst `∂f/∂θ` **6.8e-1**,
/// while the *value* was exact to 4e-16. Threading `tad` as a dual takes both to
/// the FD noise floor (3.7e-9 / 6.6e-9) and leaves the value bit-identical.
///
/// Mutation guard: reverting `eval_rhs_g`'s `*d = tad` to `T::from_f64(tad.val())`
/// reproduces the 6.6e-1 figures here.
#[test]
fn ode_tad_rhs_with_estimated_lagtime_is_analytic_and_exact() {
    let model = tad_gate_model("TAD", "TVLAG * exp(ETA_LAG)");
    let subject = tad_gate_subject();
    assert!(model.has_lagtime());
    assert!(
        ode_analytical_supported(&model),
        "#1070: a TAD-reading RHS under an estimated lagtime is analytic again"
    );
    assert!(
        ode_tvcov_supported(&model, &subject),
        "the event-driven gate must admit it too"
    );
    assert!(
        crate::sens::provider::ode_inner_grad_supported_model(&model),
        "inner EBE gradient must follow the outer — never an analytic outer with an FD inner"
    );
    let theta = [1.0, 20.0, 0.75, 0.75];
    let eta = [0.1, 0.07];
    // Non-vacuity: the lag axis must actually move the predictions, or a dropped
    // `∂TAD/∂lag` term would be invisible and this fixture would prove nothing.
    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("analytic");
    let lag_axis = sens
        .obs
        .iter()
        .map(|o| o.df_deta[1].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        lag_axis > 1e-3,
        "eta_LAG has no effect on any observation - the fixture would pass vacuously"
    );
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // Second order is deliberately NOT asserted here, and the reason is worth stating
    // rather than left as an absence. In the saltation term `(g⁻ − g⁺)·δlag` the one-sided
    // velocities' own jets multiply a `δlag` whose *value* is zero, so they cannot appear
    // at first order at all — they enter only the Hessian. Carrying the anchor's jet
    // improves that element a lot (`∂²f/∂η_LAG²` on this fixture: **6.40e-1** wrong before
    // this fix, **2.23e-1** after) but does not close it: the `δlag²` coefficient is built
    // from `jdotg_value`, which returns `ẍ = J·g` — the **state** Jacobian only — while a
    // `TAD`-reading RHS is non-autonomous and its `ẍ` carries an explicit `∂f/∂t` too.
    //
    // That gap pre-dates #1070 (it reproduces with `tad` lifted as an `f64`, which is what
    // the 6.40e-1 above is), and it is invisible without a `TAD`-reading RHS *and* an η on
    // the lagtime — every other geometry in this file is exact to 2e-7, including
    // `TAD` + a θ-only lag and `TAD` + a literal lag. Seeding the anchor with `−1` inside
    // `jdotg_value` recovers part of it (2.23e-1 → 1.51e-1) but not all, so it wants the
    // full non-autonomous second-order derivation rather than a one-line patch.
    //
    // That derivation is #1075 (`ẍ = ∂f/∂t + J·f`, the explicit time-derivative term), which
    // this fixture *widens*: #1075 was measured at 6.4e-2 on a straddling input-rate kernel
    // reading a time-varying covariate, and reasoned that a bare user RHS is autonomous
    // "unless it reads `TIME`/`TAD`". This is that exception, and it is an order of
    // magnitude larger — no straddling forcing and no co-timed boundary required.
}

/// The lagtime carries NO IIV — only θ. `Dual2` seeds every θ, so the **outer**
/// θ-gradient on the lag axis was wrong (measured 6.8e-1) even though no η touches
/// the lag; the η axes were exact throughout. That asymmetry is why the interim
/// gate had to be model-level rather than conditional on the lag carrying a random
/// effect, and it is the axis this test pins now that the term is carried.
#[test]
fn ode_tad_rhs_with_theta_only_lagtime_is_analytic_and_exact() {
    let model = tad_gate_model("TAD", "TVLAG");
    let subject = tad_gate_subject();
    assert!(model.has_lagtime());
    assert!(
        ode_analytical_supported(&model),
        "#1070: a theta-only lagtime moves TAD, and that term is now in the chain"
    );
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75, 0.75], &[0.1, 0.07]);
}

/// A **literal** lagtime carries no jet at all, so this geometry was exact even
/// while `tad` was lifted — but the interim gate declined it anyway
/// (`has_lagtime()` cannot see that the lag is a constant). Kept as the control
/// that the fix did not perturb a path that was already right: the `TVLAG` axis is
/// identically zero here, so a spurious jet would show up immediately.
#[test]
fn ode_tad_rhs_with_literal_lagtime_is_analytic_and_exact() {
    let model = tad_gate_model("TAD", "0.5");
    let subject = tad_gate_subject();
    assert!(model.has_lagtime());
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75, 0.75], &[0.1, 0.07]);
}

/// **Neighbour that must stay analytic.** `TAFD` anchors at the *unlagged*
/// `min(d.time)` in both the walk and the production predictor, so it carries no
/// lag jet — measured exact on every axis. Keying the gate on the existing
/// `uses_time_vars` (which unions `TIME`/`TAFD`/`TAD`) would decline this model
/// for no reason; this test is what makes that regression loud.
#[test]
fn ode_tafd_rhs_with_estimated_lagtime_stays_analytic() {
    let model = tad_gate_model("TAFD", "TVLAG * exp(ETA_LAG)");
    let subject = tad_gate_subject();
    assert!(model.has_lagtime());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TAFD + lagtime is exact and must keep its analytic route"
    );
    // And it really is exact — the gate's premise, not just its wiring.
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75, 0.75], &[0.1, 0.07]);
}

/// **Neighbour that must not regress.** A `TAD`-reading RHS with no lagtime anchors
/// at the dose time itself, which carries no jet at all — so this cell was already
/// exact before #1070 and must stay so. It is the control for the dual `tad`: if
/// threading the anchor introduced a spurious jet, it would show up here first,
/// where the true derivative is identically zero.
#[test]
fn ode_tad_rhs_without_lagtime_stays_analytic() {
    let model = tad_gate_model("TAD", "");
    let subject = tad_gate_subject();
    assert!(!model.has_lagtime());
    assert!(
        ode_tvcov_supported(&model, &subject),
        "TAD without a lagtime is exact and must keep its analytic route"
    );
    check_vs_production(&model, &subject, &[1.0, 20.0, 0.75, 0.75], &[0.1, 0.07]);
}

/// **Neighbour that must stay analytic**, on the *static* superposition walk
/// rather than the event-driven one — the matrix cell that had no coverage at
/// all. A subject with no TV covariates and no lagtime takes `integrate_g`,
/// whose TAD anchor is a per-segment fold over unlagged `d.time`.
#[test]
fn ode_tad_rhs_without_lagtime_static_walk_stays_analytic() {
    let model = tad_gate_model("TAD", "");
    assert!(!model.has_lagtime());
    let subject = bolus_subject_wt(&[1.0, 2.0, 4.0, 8.0], 70.0);
    assert!(!subject.has_tv_covariates(), "static-walk subject");
    assert!(
        ode_analytical_supported(&model),
        "TAD with no lagtime must stay analytic on the static walk too"
    );
    assert!(ode_subject_supported(&model, &subject));
}

/// A per-route absorption lag (`fn(..., lag=L)`) never anchors TAD — the walk
/// writes `last_dose_eff` only in its `K_DOSE` arm, from the *compartment* lag,
/// and production's `tad_anchor` reads `dose_lagtimes` for the same reason. So
/// `has_route_absorption_lag()` is deliberately absent from the gate, and a pure
/// route lag must keep its analytic route.
#[test]
fn ode_tad_rhs_with_route_lag_only_stays_analytic() {
    const ROUTE_LAG_TAD: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVRLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  RLAG = TVRLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=RLAG) - (CL/V) * central * (1.0 + 0.3 * TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rk45
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(ROUTE_LAG_TAD).expect("parse");
    assert!(
        model.has_route_absorption_lag(),
        "precondition: model carries a per-route lag"
    );
    assert!(
        !model.has_lagtime(),
        "precondition: and NO compartment lagtime — the thing that anchors TAD"
    );
    assert!(
        ode_analytical_supported(&model),
        "a pure route lag does not move the TAD anchor, so it must stay analytic"
    );
}

/// Steady state keeps its own, broader decline: SS breaks the cycle recurrence for
/// ANY non-autonomous RHS, `TIME`-only ones included. It is what still routes `TAD`
/// + SS to FD now that #1070's narrower, TAD-only predicate has been deleted along
/// with the interim gate it served. Regression guard that removing the narrow one
/// left the broad one intact — and that it stayed broad when #1124 widened it from
/// `uses_time_vars` to `reads_model_time` (the union plus the bare `TIME` built-in).
#[test]
fn ode_ss_time_only_rhs_still_routes_to_fd() {
    // `T`, not `TIME`: a bare `TIME` in an `[odes]` RHS compiles to `Op::PushTime`
    // (the model-time thread-local), not to a `PushVar(time_slot)`, so
    // `stmts_read_slots` — and therefore `uses_time_vars` — does not see it. `T`
    // is the alias that does resolve to the slot.
    //
    // Until #1124 that choice was a **workaround**, not a simplification: the
    // gate asked `uses_time_vars` alone, so the `TIME` spelling of the same
    // quantity reached the SS equilibration while `T` declined. The gate now asks
    // `reads_model_time`; `ode_ss_bare_time_builtin_rhs_routes_to_fd` covers the
    // spelling this test cannot.
    let model = tad_gate_model("T", "");
    let prog = model
        .ode_spec
        .as_ref()
        .and_then(|o| o.rhs_program.as_ref())
        .expect("rhs program");
    assert!(prog.uses_time_vars(), "`T` sets the broad flag");
    let mut ss = bolus_subject_wt(&[1.0, 2.0, 4.0], 70.0);
    ss.doses[0].ss = true;
    ss.doses[0].ii = 8.0;
    assert!(
        !ode_tvcov_supported(&model, &ss),
        "SS + a time-dependent RHS must still decline (the broad uses_time_vars gate)"
    );
}

/// The spelling `ode_ss_time_only_rhs_still_routes_to_fd` works around: a **bare
/// `TIME`** built-in in the RHS.
///
/// `uses_time_vars` is structurally blind to it (`Op::PushTime`, not a
/// `PushVar(time_slot)`), so until #1124 this exact model reached
/// `equilibrate_ss_state_g` while the identical model spelled `T` declined —
/// the gate's own doc said to pair the two flags and this caller did not.
///
/// What the gate does and does not buy, measured against an explicit 40-cycle
/// pulse train (the oracle with no SS machinery in it at all):
///
/// | RHS term      | SS=1 value | vs. pulse train |
/// |---------------|------------|-----------------|
/// | autonomous    | 10.027269  | `7.6e-9`        |
/// | `0.003*T`     | 9.545089   | **17% wrong**   |
/// | `0.003*TIME`  | 9.545089   | **17% wrong**   |
/// | `0.03*TAD`    | `NaN`      | —               |
/// | `0.0*TAD`     | `NaN`      | —               |
///
/// So this gate closes an asymmetry in the **gradient** route only: `T` was
/// already declining and its value is wrong anyway. Declining is still right —
/// an analytic jet of a wrong value is worse than an FD one — but the value
/// defect is separate and is not fixed here (#1139).
#[test]
fn ode_ss_bare_time_builtin_rhs_routes_to_fd() {
    let model = tad_gate_model("TIME", "");
    let prog = model
        .ode_spec
        .as_ref()
        .and_then(|o| o.rhs_program.as_ref())
        .expect("rhs program");
    assert!(
        !prog.uses_time_vars(),
        "precondition: a bare `TIME` must NOT set the slot-backed flag — if it \
         does, this test no longer covers the spelling it exists for"
    );
    assert!(prog.reads_time_builtin(), "`TIME` sets the built-in flag");
    assert!(prog.reads_model_time(), "and the paired predicate sees it");

    let mut ss = bolus_subject_wt(&[1.0, 2.0, 4.0], 70.0);
    ss.doses[0].ss = true;
    ss.doses[0].ii = 8.0;
    assert!(
        !ode_tvcov_supported(&model, &ss),
        "SS + a bare-`TIME` RHS must decline to FD, exactly as the `T` spelling does"
    );
    // The non-SS neighbour must be unaffected: this gate is about steady state,
    // not about `TIME`, so widening it must not sweep up an ordinary subject. Use
    // a TV-covariate subject, which reaches `ode_tvcov_supported`'s trigger list
    // through a trigger that is *independent of the model-time clause* — so the
    // assert stays sharp no matter what that clause is set to. (It used to be the
    // only way to make this non-vacuous at all: the trigger list asked the narrow
    // predicate, so a plain bolus subject on this model had no trigger and took
    // the static dual walk. The list asks the wide predicate now, so a plain bolus
    // subject would also pass — but via the very clause under test, which is
    // exactly the circularity the TV-cov subject avoids.)
    let plain = tad_gate_subject();
    assert!(
        !plain.has_periodic_ss_dose() && plain.has_tv_covariates(),
        "precondition: the neighbour has no SS dose but does have a TV-cov trigger"
    );
    assert!(
        ode_tvcov_supported(&model, &plain),
        "a non-SS subject on the same `TIME`-reading model must keep the event-driven \
         dual walk — the SS clause must not sweep it up"
    );
}

/// A pre-arrival observation on a model-time-reading RHS must take the
/// **event-driven** dual walk, not the static one.
///
/// The static walk resolves `TAD`'s anchor as
/// `doses.filter(|dt| dt <= t_start).fold(NEG_INFINITY, f64::max)`, so a segment
/// starting before the first dose leaves it at `-inf`, and `eval_rhs_anchored`
/// turns a missing/non-finite anchor into `TAD = NaN` — poisoning every
/// `∂f/∂θ` and `∂f/∂η` for the subject. The event-driven walk seeds the same
/// variable at the first arrival instead (that is #1073's pre-arrival anchor,
/// which the static walk never received).
///
/// Nothing forces the routing on its own: with the model-time clause narrow, a
/// plain bolus subject on this model had no trigger at all and fell to the
/// static walk, so the `NaN` was reachable. This asserts the clause that now
/// keeps it away — and the preconditions below make the assert non-vacuous by
/// ruling out every *other* trigger, so it can only be passing for the reason
/// it claims.
#[test]
fn a_pre_arrival_observation_on_a_tad_rhs_takes_the_event_driven_walk() {
    let model = tad_gate_model("TAD", "");
    let mut subject = bolus_subject_wt(&[0.4, 2.0, 4.0], 70.0);
    // Move the dose after the first observation: that observation is now
    // pre-arrival, which is the window with no qualifying dose.
    subject.doses[0].time = 1.0;
    assert!(
        subject.obs_times[0] < subject.doses[0].time,
        "precondition: the first observation must precede the first dose"
    );
    // No other trigger — otherwise this passes for an unrelated reason.
    assert!(
        !subject.has_tv_covariates()
            && !subject.has_resets()
            && !subject.has_periodic_ss_dose()
            && !model.has_lagtime()
            && !model.has_route_absorption_lag(),
        "precondition: the model-time clause must be the only trigger in play"
    );
    assert!(
        ode_analytical_supported(&model),
        "precondition: the analytic ODE route must be available at all"
    );
    assert!(
        ode_tvcov_supported(&model, &subject),
        "a model-time-reading RHS must take the event-driven dual walk; on the \
         static walk the pre-arrival segment has no dose anchor and `TAD` is NaN"
    );
}

const PER_CMT_ALAG_TAD_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVLAG1(0.4, 0.01, 5.0)
  theta TVLAG2(0.9, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_L1 ~ 0.04
  omega ETA_L2 ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = TVLAG1 * exp(ETA_L1)
  ALAG2 = TVLAG2 * exp(ETA_L2)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central * (1.0 + 0.3 * TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rk45
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **Two arrivals whose jets live on different axes.** `ALAG1` and `ALAG2` are separate
/// compartment-indexed lags with their own θ *and* their own η, and the subject doses into
/// both compartments — so the `TAD` anchor switches from an arrival carrying `∂/∂ETA_L1` to
/// one carrying `∂/∂ETA_L2` partway through. Nothing else in the #1070 set exercises that:
/// every other fixture has a single lag slot, where "carry the anchor's jet" and "carry
/// *this* dose's jet" are the same statement.
///
/// This is the geometry `later_arrival` and `arrival_dual(k)` exist for. A version that read
/// the lag from a single slot, or that kept the incumbent arrival's jet past the second
/// arrival, would be exact on every other fixture in this file and wrong here.
///
/// Arrivals are ≈0.4205 (depot dose at `t = 0`, `ALAG1 = 0.4·e^{0.05}`) and ≈1.8476 (central
/// dose at `t = 1`, `ALAG2 = 0.9·e^{−0.06}`), and the observations straddle both — with the
/// second dose landing while the first is still visible, so its arrival has live state on the
/// incoming side. Both sit strictly inside a record interval, which the test re-derives and
/// asserts rather than trusting these numbers.
#[test]
fn ode_tad_rhs_per_compartment_lags_anchor_on_their_own_axes() {
    let model = parse_model_string(PER_CMT_ALAG_TAD_ODE).expect("parse");
    assert!(model.has_lagtime());
    assert!(
        model
            .active_dose_attr_map()
            .has_indexed_attr(crate::types::DoseAttr::Lag),
        "precondition: the lags must be compartment-indexed, not the bare slot"
    );
    let mut subject = bolus_subject(&[0.2, 0.8, 1.5, 2.5, 4.0, 8.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.0, 40.0, 2, 0.0, false, 0.0),
    ];
    // theta = [TVCL, TVV, TVKA, TVLAG1, TVLAG2]
    let theta = [1.0, 10.0, 1.2, 0.4, 0.9];
    let eta = [0.1, 0.05, -0.06];

    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("analytic");
    // Non-degeneracy: BOTH lag axes must move an observation. If only one did, a fix that
    // carried a single arrival's jet everywhere would still pass.
    for (ax, name) in [(1usize, "ETA_L1"), (2, "ETA_L2")] {
        let moved = sens
            .obs
            .iter()
            .map(|o| o.df_deta[ax].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            moved > 1e-3,
            "{name} moves no observation - the anchor switch would be untested"
        );
    }
    // ...and the anchor must actually switch: the second arrival lands strictly inside the
    // observation window, with the first dose's drug still present.
    let arrival2 = 1.0 + theta[4] * eta[2].exp();
    assert!(
        subject.obs_times[2] < arrival2 && arrival2 < subject.obs_times[3],
        "the second arrival at {arrival2} must fall between two records"
    );
    assert!(
        sens.obs[2].f > 1e-3,
        "the first dose must still be visible when the second arrives"
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    // Second order is not asserted, for the reason spelled out in
    // `ode_tad_rhs_with_estimated_lagtime_is_analytic_and_exact`: the `δlag²` coefficient
    // treats the RHS as autonomous, which a `TAD`-reading one is not. Pre-dates #1070 and
    // is tracked as #1075.
}

/// **A NaN lagtime must degrade, not abort.** The outer optimizer can step a lagtime
/// expression into its NaN domain — `(TVLAG - 2.0)^0.5` at `TVLAG < 2` is the shortest
/// example — and `integrate_tvcov_g` states the requirement explicitly: such an excursion
/// "must degrade, not panic a debug-build fit". Both pre-existing `debug_assert!`s in that
/// function are written NaN-tolerantly for exactly this reason.
///
/// The anchor `debug_assert` added with the dual `tad` was originally a `debug_assert_eq!`,
/// which cannot express that: `NaN != NaN`, so it failed on precisely the input it was meant
/// to tolerate. Reproduced as `assertion 'left == right' failed / left: NaN / right: NaN`
/// before the fix. Debug builds are what `cargo test` produces, and `[profile.ci-test]`
/// inherits `release`, so CI could never have caught this — hence a test rather than a
/// reliance on the suite.
#[test]
fn ode_tad_rhs_nan_lagtime_degrades_rather_than_panicking() {
    const NAN_LAG: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.0, -1.0, 5.0)
  theta TVLAG(3.0, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
  LAGTIME = (TVLAG - 2.0)^0.5 * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.3 * TAD)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rk45
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(NAN_LAG).expect("parse");
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(70.0), wt(75.0)];

    // Control: a real lagtime resolves and the walk serves the subject normally.
    let ok = ode_subject_sensitivities(&model, &subject, &[1.0, 20.0, 0.0, 3.0], &[0.1, 0.07])
        .expect("a resolvable lagtime must still be served");
    assert!(
        ok.obs.iter().all(|o| o.f.is_finite()),
        "control arm must be finite, got {:?}",
        ok.obs.iter().map(|o| o.f).collect::<Vec<_>>()
    );

    // The excursion: `TVLAG = 1.0` makes the lag `sqrt(-1)`. Non-vacuity — assert the model
    // really does produce a NaN lag here, or the test would pass without exercising anything.
    let theta_nan = [1.0, 20.0, 0.0, 1.0];
    let pk = (model.pk_param_fn)(&theta_nan, &[0.1, 0.07], &subject.covariates, 0.0);
    assert!(
        pk.values[crate::types::PK_IDX_LAGTIME].is_nan(),
        "fixture must produce a NaN lagtime, got {}",
        pk.values[crate::types::PK_IDX_LAGTIME]
    );

    // Must not panic. Whether it declines or returns non-finite values is the optimizer's
    // problem to reject; aborting the process is not an option.
    let out = ode_subject_sensitivities(&model, &subject, &theta_nan, &[0.1, 0.07]);
    if let Some(v) = out {
        assert!(
            v.obs.iter().any(|o| !o.f.is_finite()),
            "a NaN lagtime should not yield a finite trajectory"
        );
    }
}

/// **The route the user is actually told about.** Every other #1070 test asserts
/// the low-level predicate (`ode_analytical_supported` / `ode_iov_supported`), but
/// the *reported* `gradient_method_outer` comes from
/// `analytic_outer_gradient_for_interaction`, and the outer FOCE/FOCEI dispatch
/// from `sens_supported` — neither of which any other test pins. The θ-only-lag
/// probe measured the **outer** θ axis 6.8e-1 wrong, so this is the axis the fix
/// exists for; without this test a refactor could re-introduce the interim FD
/// detour while every parity test stayed green.
///
/// All four shapes assert the same direction now — analytic — because the dual
/// `tad` makes the two that used to decline exact. Their *values* are pinned by
/// `ode_tad_rhs_with_*_lagtime_is_analytic_and_exact`; this test pins only that the
/// user-visible route reaches them.
#[test]
fn tad_lagtime_keeps_the_analytic_outer_gradient_route() {
    for (time_var, lag_expr) in [
        ("TAD", "TVLAG * exp(ETA_LAG)"),
        ("TAD", "TVLAG"), // theta-only lag: Dual2 seeds every theta
        ("TAD", ""),
        ("TAFD", "TVLAG * exp(ETA_LAG)"),
    ] {
        let model = tad_gate_model(time_var, lag_expr);
        assert!(
            crate::sens::provider::sens_supported(&model),
            "#1070: outer gradient must stay analytic for {time_var} + `{lag_expr}`"
        );
        for interaction in [false, true] {
            assert!(
                crate::sens::provider::analytic_outer_gradient_for_interaction(&model, interaction),
                "#1070: reported outer method must be analytic for {time_var} + `{lag_expr}` \
                 (interaction = {interaction})"
            );
        }
    }
}

// #1070 under IOV. The IOV outer/inner walks share `integrate_tvcov_readout` /
// `integrate_tvcov_g` with the non-IOV TV-cov walk, so the dual `tad` reaches them by
// construction — but `ode_iov_supported` is a deliberately *parallel* gate that never calls
// `ode_analytical_supported`, and the interim #1070 fallback had to be written into it
// separately for exactly that reason. This fixture is what keeps the IOV walk pinned to the
// same standard as the non-IOV one.
const IOV_LAG_TAD_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVP(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_P  ~ 0.04
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  LAGTIME = TVP * exp(ETA_P + KAPPA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.3 * TAD)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_method = rk45
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **#1070 under IOV, with κ on the lagtime itself.** Each occasion's lag — and therefore
/// each occasion's `TAD` anchor — moves on its own κ axis, so this is the geometry where a
/// lifted `tad` is worst: measured against FD of `predict_iov` at `df8bb8d6`, the worst
/// stacked-axis error was **2.0e-1** while the *value* was exact to 0.0e0. With `tad`
/// threaded as a dual it is **8.9e-10**.
///
/// Non-degeneracy is asserted, not assumed: each κ axis must actually move an observation
/// (they move them by 0.35 and 0.61 here), and the second dose lands in occasion 2 with
/// residual drug from the first still present, so the anchor genuinely switches mid-subject
/// rather than being established once and never revisited.
#[test]
fn ode_iov_kappa_on_lagtime_tad_rhs_matches_fd_of_predict_iov() {
    let model = parse_model_string(IOV_LAG_TAD_ODE).expect("parse");
    assert_eq!(model.n_kappa, 1);
    assert!(
        ode_iov_supported(&model),
        "#1070: the parallel IOV gate must admit a TAD-reading RHS under a lagtime"
    );
    let mut subj = bolus_subject(&[1.5, 2.0, 4.0, 8.0, 12.0]);
    subj.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.0, 200.0, 1, 0.0, false, 0.0),
    ];
    subj.dose_occasions = vec![1, 2];
    subj.occasions = vec![1, 2, 2, 2, 2];
    let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(groups.len(), 2);
    let theta = vec![10.0, 50.0, 0.7];
    let stacked = vec![0.1, 0.07, 0.03, -0.02];
    let sens = ode_subject_sensitivities_iov(&model, &subj, &theta, &stacked).expect("analytic");
    // The INNER EBE route as well as the outer. These are separate walks — `run_subject_iov`
    // over `Dual2` and `run_subject_iov_eta` over `Dual1` — and the test deleted along with the
    // interim gate was the only thing asserting the inner one for this cell. Without this, a
    // change that silently dropped the IOV inner back to FD leaves every test green; that is
    // the outer/inner asymmetry the non-IOV fixture guards with `ode_inner_grad_supported_model`.
    let inner = ode_subject_eta_grad_iov(&model, &subj, &theta, &stacked)
        .expect("#1070: the IOV inner EBE gradient must be analytic too");
    assert_eq!(inner.len(), sens.obs.len());
    for (a, b) in sens.obs.iter().zip(inner.iter()) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-12);
        for k in 0..stacked.len() {
            approx::assert_relative_eq!(
                a.df_deta[k],
                b.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
    let pred = |st: &[f64], j: usize| -> f64 {
        let eta_bsv = st[..model.n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..groups.len())
            .map(|g| {
                st[model.n_eta + g * model.n_kappa..model.n_eta + (g + 1) * model.n_kappa].to_vec()
            })
            .collect();
        crate::pk::predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[j]
    };
    let n = stacked.len();
    // Non-vacuity: every kappa axis must move an observation, or a dropped anchor jet on
    // that axis is invisible and the fixture proves nothing.
    for k in model.n_eta..n {
        let moved = sens
            .obs
            .iter()
            .map(|o| o.df_deta[k].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            moved > 1e-2,
            "stacked axis {k} moves no observation - the fixture would pass vacuously"
        );
    }
    let h = 1e-6;
    for (j, o) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(o.f, pred(&stacked, j), max_relative = 1e-9, epsilon = 1e-12);
        for k in 0..n {
            let mut sp = stacked.clone();
            sp[k] += h;
            let mut sm = stacked.clone();
            sm[k] -= h;
            let g = (pred(&sp, j) - pred(&sm, j)) / (2.0 * h);
            approx::assert_relative_eq!(o.df_deta[k], g, max_relative = 1e-5, epsilon = 1e-7);
        }
    }
}

const PREARRIVAL_INIT_TAD: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.0, -5.0, 5.0) FIX
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 70.0
  d/dt(central) = -(CL/V) * central * (1.0 + 0.3 * TAD)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rk45
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **The `TAD` anchor before any dose has arrived also carries the lagtime's jet.** Since
/// #1073 both ODE predictors anchor the pre-arrival window at the subject's *first arrival*
/// rather than answering `NaN`, so `TAD` is negative-but-finite there — and that arrival
/// moves with the lag exactly like every later one. The seed is easy to overlook because it
/// is written once, outside the event loop, and because with no `init(...)` the pre-arrival
/// state is zero and the whole term is vacuous.
///
/// So the fixture is built to make it live on both counts: `init(central) = 70` gives the
/// pre-arrival window real state (it decays 70 → 68.80 across the two records at `t = 0.20`
/// and `t = 0.35`, both strictly before the arrival at ≈0.536), and the dose then lands
/// (68.80 → 163.57), so the anchor genuinely switches mid-subject.
///
/// `THETA_WT` is `FIX` at zero, so `(WT/70)^0 == 1` exactly and the time-varying `WT` cannot
/// move any prediction or contribute a `∂f/∂θ` column — this is not a covariate test. Note
/// what `WT` does *not* do: it is **not** what selects the event-driven walk. A lagtime alone
/// does that, because `ode_subject_supported` declines `has_lagtime()` outright, so the static
/// walk is unreachable for this model with or without the covariate. That distinction matters
/// — believing a TV covariate was needed to reach the event-driven walk is what left
/// `integrate_g` with no `TAD` coverage at all.
///
/// Mutation guard: seeding `last_dose_eff` from `T::from_f64(d.time + lag_val(k))` — the
/// value-only first arrival, leaving every later `K_DOSE` update dual — makes `∂f/∂η_LAG`
/// **11 % wrong at `t = 0.20`, rising to 56 % at `t = 4`**. The error is injected once in the
/// pre-arrival window and then carried multiplicatively, which is why it is *worse* after the
/// arrival than before it.
#[test]
fn ode_tad_rhs_prearrival_window_carries_the_lagtime_jet() {
    let model = parse_model_string(PREARRIVAL_INIT_TAD).expect("parse");
    let mut subject = bolus_subject(&[0.2, 0.35, 1.0, 2.0, 4.0]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(70.0), wt(75.0), wt(80.0)];
    let theta = [1.0, 20.0, 0.0, 0.5];
    let eta = [0.1, 0.07];

    let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("analytic");
    // Non-degeneracy, asserted rather than assumed: the first two records must sit strictly
    // inside the pre-arrival window, that window must carry live state, and the dose must
    // then actually move the trajectory.
    let arrival = theta[3] * eta[1].exp();
    assert!(
        subject.obs_times[1] < arrival && arrival < subject.obs_times[2],
        "records 0/1 must precede the arrival at {arrival} and record 2 must follow it"
    );
    assert!(
        sens.obs[0].f > 65.0 && sens.obs[1].f < sens.obs[0].f,
        "the pre-arrival window must carry decaying init state, got {:?}",
        (sens.obs[0].f, sens.obs[1].f)
    );
    assert!(
        sens.obs[2].f > 2.0 * sens.obs[1].f,
        "the dose must land, or the anchor never switches"
    );
    // Non-vacuity on the AXIS under test, not just on the state. The seed's jet is observable
    // only at the two pre-arrival records; from the arrival onward the anchor is the ordinary
    // `K_DOSE` fold, which is dual either way. Without this, a change that pushed the
    // pre-arrival contribution under `check_vs_production`'s tolerance would leave the fixture
    // green with the later records carrying the pass.
    for j in 0..2 {
        assert!(
            sens.obs[j].df_deta[1].abs() > 1e-2,
            "eta_LAG must move the pre-arrival record at t = {}, got {:.3e} - the seed's jet \
             would otherwise be untested",
            subject.obs_times[j],
            sens.obs[j].df_deta[1]
        );
    }

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **Two arrivals that coincide bit-exactly.** `ALAG1 = 1.5` on a dose at `t = 0` and
/// `ALAG2 = 0.5` on a dose at `t = 1`; at `η = 0` the exponentials are exactly `1.0`, so both
/// arrivals are exactly `1.5` — a genuine tie, and the test asserts the bit equality rather
/// than trusting the arithmetic. Both lags carry a live jet (`∂lag/∂η` = 1.5 and 0.5).
///
/// A tie is a kink in the `max` that defines the TAD anchor, so FD parity cannot referee it:
/// a central difference there averages two one-sided derivatives. What CAN be pinned is that
/// the analytic answer equals **one** of the one-sided limits rather than a mixture of both —
/// and specifically the one whose event ordering the walk actually uses. The timeline is
/// sorted `(time, kind)` by a stable `sort_by` with `K_DOSE` pushed in dose order, so co-timed
/// arrivals fire in ascending index: the `δ⁻` configuration, in which the lower-indexed dose
/// arrives first.
///
/// This is second-order only. The boundary velocities' jets multiply a `δlag` whose *value*
/// is zero, so they cannot reach `df_deta` at all — measuring first order here shows nothing
/// (both tie-break rules give identical `df_deta`), which is why this test reads the
/// `η`-lag block of `d2f_deta2`.
///
/// Mutation guard: flipping `later_arrival`'s comparison to `>` (keep the incumbent) moves the
/// tie to `max|tie − δ⁻| = 1.5e0` and `max|tie − δ⁺| = 8.0e-1` — matching neither limit.
#[test]
fn ode_tad_rhs_co_timed_arrivals_resolve_to_one_sided_limit() {
    fn model_for(tvlag1: f64) -> CompiledModel {
        let src = format!(
            r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVLAG1({tvlag1:.17}, 0.01, 5.0)
  theta TVLAG2(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_L1 ~ 0.04
  omega ETA_L2 ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = TVLAG1 * exp(ETA_L1)
  ALAG2 = TVLAG2 * exp(ETA_L2)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central * (1.0 + 0.3 * TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rk45
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#
        );
        parse_model_string(&src).expect("parse")
    }

    let mut subject = bolus_subject(&[0.5, 2.0, 3.0, 5.0, 8.0]);
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.0, 60.0, 2, 0.0, false, 0.0),
    ];
    let eta = [0.05, 0.0, 0.0];

    // The tie must be exact, or this test is about two nearby arrivals and proves nothing.
    assert_eq!(
        (0.0f64 + 1.5).to_bits(),
        (1.0f64 + 0.5).to_bits(),
        "fixture must produce a bit-exact tie"
    );

    // The eta-lag block of the second-order jet: [L1,L1], [L1,L2], [L2,L2].
    let hess = |tvlag1: f64| -> Vec<[f64; 3]> {
        let m = model_for(tvlag1);
        let th = [1.0, 10.0, 1.2, tvlag1, 0.5];
        let s = ode_subject_sensitivities(&m, &subject, &th, &eta).expect("analytic");
        s.obs
            .iter()
            .map(|o| [o.d2f_deta2[4], o.d2f_deta2[5], o.d2f_deta2[8]])
            .collect()
    };

    let d = 1e-6;
    let at_tie = hess(1.5);
    let plus = hess(1.5 + d); // arrival0 > arrival1: dose 0 arrives last
    let minus = hess(1.5 - d); // arrival0 < arrival1: dose 1 arrives last (the sort's ordering)

    let dist = |a: &[[f64; 3]], b: &[[f64; 3]]| -> f64 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()))
            .fold(0.0_f64, f64::max)
    };
    let to_minus = dist(&at_tie, &minus);
    let to_plus = dist(&at_tie, &plus);

    // Non-vacuity: the two limits must actually differ, or "matches one of them" is empty.
    assert!(
        dist(&plus, &minus) > 1e-2,
        "the two one-sided limits are indistinguishable ({:.3e}) - the fixture would pass \
         vacuously",
        dist(&plus, &minus)
    );
    assert!(
        to_minus < 1e-3,
        "the tie must resolve to the one-sided limit whose event ordering the stable sort \
         reproduces, but max|tie - delta-| = {to_minus:.3e}"
    );
    assert!(
        to_plus > 1e-2,
        "sanity: the tie should NOT also match the other limit, but max|tie - delta+| = \
         {to_plus:.3e}"
    );
}

/// #1070: `later_arrival`'s **incumbent** arm, asserted directly on the pure function.
///
/// The walk cannot reach it — the timeline is sorted ascending, so at a `K_DOSE` the
/// candidate is always `>=` the incumbent and the candidate arm always wins. But the fold
/// is a `max`, and its contract is that an out-of-order or NaN candidate must not drag the
/// anchor backwards or poison it. Nothing in a reachable walk state exercises that, so it
/// is pinned here rather than left to a fixture that cannot reach it.
#[test]
fn later_arrival_never_moves_the_anchor_backwards() {
    // Ascending — the only ordering the walk actually produces.
    assert_eq!(later_arrival(3.0_f64, 5.0_f64), 5.0);
    // Out of order: the incumbent must survive.
    assert_eq!(later_arrival(5.0_f64, 3.0_f64), 5.0);
    // Exact tie: the candidate, i.e. the stable sort's own ordering (#1070 round 2).
    assert_eq!(later_arrival(4.0_f64, 4.0_f64), 4.0);
    // Unseeded incumbent: any candidate is adopted.
    assert_eq!(later_arrival(f64::NAN, 2.0_f64), 2.0);
    // A NaN candidate must not silently become the anchor — `NaN >= x` is false, so the
    // incumbent is kept. This is the arm a `>` / `>=` mix-up would break.
    assert_eq!(later_arrival(2.0_f64, f64::NAN), 2.0);
}

/// A TV-cov model whose RHS reads `TAD`, for the rate-off saltation boundaries. Same
/// parameter shape as `ONECPT_ODE_TVCOV`, so the same θ/η apply.
const TVCOV_TAD_RATEOFF_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.02 * TAD)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

/// **Infusion rate-off under a time-varying covariate, with a `TAD`-reading RHS.** The
/// `K_INF_END` handler shares its `TAD` anchor between the pre- and post-boundary velocities
/// (no dose lands at a rate-off), so a wrong anchor there moves `∂f/∂η` without moving `f`.
/// The other TV-cov infusion fixtures in this module use a `t`-independent RHS, leaving that
/// sharing pinned only through a constant.
///
/// Measured scope, so nobody re-derives it: this exercises the **cheap**
/// `inject_rate_saltation` path, not the general `g⁻−g⁺` one. The general path is guarded on
/// `pk_snapshot_equal(pre, post)` being false, and it is unreachable as the walk is built —
/// `params` is `next_record_params[p]` (first record at index `>= p`) while `post_snapshot`
/// is the first record *strictly* later, and every record co-timed with a rate-off sorts
/// before it (`K_INF_END` = 7, `K_ZO_END` = 8 are the last kinds), so both resolve to the
/// same record. Verified: zero hits on that arm across the whole instrumented suite.
#[test]
fn ode_tad_rhs_infusion_end_inside_a_covariate_step_matches_production() {
    let model = parse_model_string(TVCOV_TAD_RATEOFF_ODE).expect("parse");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subject.doses[0].rate = 50.0;
    subject.doses[0].duration = 2.0;
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];

    assert!(crate::dosing::is_real_infusion(&subject.doses[0]));
    assert!(subject.has_tv_covariates() && ode_tvcov_supported(&model, &subject));
    // Non-vacuity: the window end must land strictly BETWEEN two records carrying
    // different WT, or the snapshots agree and the cheap path runs instead.
    let w_end = subject.doses[0].time + subject.doses[0].duration;
    assert!(
        subject.obs_times.contains(&w_end),
        "the infusion end must land ON a record, got {w_end}"
    );

    let theta = [1.0, 20.0, 0.75];
    let eta = [0.1];
    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// **Zero-order window rate-off under a time-varying covariate, with a `TAD`-reading RHS.**
/// The `K_ZO_END` twin of the test above, and with the same measured scope: the cheap
/// closed-form path, since the general one is unreachable for the same reason. Here the
/// window end additionally carries a `DUR` jet, so the shared anchor is checked on a moving
/// boundary rather than a fixed one.
#[test]
fn ode_tad_rhs_zero_order_end_inside_a_covariate_step_matches_production() {
    const TVCOV_TAD_ZO_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVDUR(2.5, 0.5, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V) * central * (1.0 + 0.02 * TAD)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
    let model = parse_model_string(TVCOV_TAD_ZO_ODE).expect("parse zero-order TV-cov TAD");
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];

    assert!(subject.has_tv_covariates() && ode_tvcov_supported(&model, &subject));

    // DUR = 2.5 · exp(ETA_DUR); with the η below the window ends strictly between the
    // `WT = 70` record at t = 2 and the `WT = 80` record at t = 4 — same non-vacuity
    // condition as the infusion twin, but here the boundary also carries a `DUR` jet.
    let theta: [f64; 4] = [1.0, 20.0, 0.75, 2.5];
    let eta: [f64; 2] = [0.1, 0.0];
    let dur = theta[3] * eta[1].exp();
    assert!(
        dur > 2.0 && dur < 4.0,
        "the zero-order window end must fall strictly inside a covariate step, got {dur}"
    );

    check_vs_production(&model, &subject, &theta, &eta);
    check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
}

/// A joint PK-TTE model (`d/dt(__chz_<cmt>)` accumulators) under a steady-state dose must
/// decline **at the scope gate**, so the caller drops to FD rather than reaching a dual SS
/// equilibration that has no accumulator handling (#1210).
///
/// This is the routing CLAUDE.md requires to be unit-tested: a scope gap here does not fail
/// loudly, it silently returns a jet in which the run-in's hazard has been cycled into
/// `d/deta` exactly as the f64 path did before #1210.
///
/// **How the gap was found, since it says what this test is really for.** The fit itself never
/// used that jet — `inner_optimizer` routes every `[event_model]` model to the FD inner
/// gradient — but `fd_fallback_warning` probes `subject_eta_grad` to count fallbacks, and the
/// probe reached the dual SS equilibration for real. The predicates were reporting these
/// subjects as *analytic* while the fit ran FD, which is precisely the mislabel that warning
/// exists to catch. Nothing saw it until a Tier-3 convergence fit ran: every `ss_chz_*` anchor
/// fixture routes to FD, so no anchor, no `--lib` run and no CI job could observe the dual
/// path at all.
///
/// **Both straddles are asserted**, or the test would pass on a predicate that always declines:
/// the same model without an SS dose stays in scope, and an SS dose on a model with no
/// accumulator stays in scope.
#[test]
fn a_joint_pktte_subject_under_ss_declines_to_fd_at_the_gate() {
    let model = parse_model_string(JOINT_PKTTE_ODE_SS).expect("parse joint PK-TTE ODE model");
    assert!(
        model
            .ode_spec
            .as_ref()
            .is_some_and(|o| !o.chz_state_slots.is_empty()),
        "fixture must actually carry an injected accumulator, or this tests nothing"
    );
    let theta = vec![1.0, 10.0, 1.0, 0.02, 0.3];
    let eta = vec![0.0];

    let ss = joint_ss_subject(true);
    assert!(
        ss_dual_equilibration_out_of_scope(&model, &ss),
        "a joint model + SS dose must be out of the dual providers' scope"
    );
    assert!(
        !ode_subject_supported(&model, &ss),
        "the static ODE gate must decline it"
    );
    assert!(
        !ode_tvcov_supported(&model, &ss),
        "the event-driven ODE gate must decline it too, or a subject reaches the dual SS \
         equilibration by taking the other route"
    );
    assert!(
        ode_subject_eta_grad(&model, &ss, &theta, &eta).is_none(),
        "the entry point must return None (→ FD), not a jet carrying the banked run-in"
    );

    // Straddle 1: drop the SS flag and the same model/subject is served again.
    let plain = joint_ss_subject(false);
    assert!(
        !ss_dual_equilibration_out_of_scope(&model, &plain),
        "without an SS dose the joint model must stay in scope, or the decline is unconditional \
         and this test cannot tell the two apart"
    );

    // Straddle 2: an SS dose on a model with no accumulator is still in scope.
    let no_chz = parse_model_string(ONECPT_ODE_NO_CHZ).expect("parse plain ODE model");
    assert!(
        no_chz
            .ode_spec
            .as_ref()
            .is_some_and(|o| o.chz_state_slots.is_empty()),
        "the contrast model must carry no accumulator"
    );
    assert!(
        !ss_dual_equilibration_out_of_scope(&no_chz, &ss),
        "the gate must key on the accumulator, not on SS dosing alone — an analytic-family TTE \
         model carries no accumulator and its SS dosing is served fine"
    );
}

/// Joint PK-TTE ODE model for [`a_joint_pktte_subject_under_ss_declines_to_fd_at_the_gate`].
const JOINT_PKTTE_ODE_SS: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.30, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

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
  d/dt(central) =  KA * depot - (CL/V) * central

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The same PK block with **no** `[event_model]`, so no accumulator is injected — straddle 2.
const ONECPT_ODE_NO_CHZ: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// [`bolus_subject`] whose single dose carries the `SS=1` flag, or not — the one bit the
/// gate keys on, so the two straddle it exactly.
fn joint_ss_subject(ss: bool) -> Subject {
    let mut s = bolus_subject(&[1.0, 4.0, 8.0, 12.0]);
    if ss {
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    }
    s.obs_cmts = vec![2; s.obs_times.len()];
    s
}
