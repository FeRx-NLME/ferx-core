use super::fit;
use crate::parser::model_parser::parse_full_model;
use crate::types::*;
use std::collections::HashMap;

/// 1-cpt IV ODE model with a [diffusion] block on the central compartment.
/// Sigma (ADD) is fixed so that the diffusion parameter must absorb residual
/// variance. We verify:
///   (a) uses_sde = true
///   (b) DIFF_CENTRAL is estimated positive
///   (c) OFV is finite and the fit converges
///   (d) OFV with diffusion <= OFV without diffusion (diffusion can only help)
const SDE_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
  central ~ 0.5

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

/// Same model without the [diffusion] block (for OFV comparison).
const BASE_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

fn make_sde_population() -> Population {
    // 4 subjects, single IV bolus dose=100 at t=0, observations at 3 times.
    // The ODE `d/dt(central) = -(CL/V) * central` describes the amount in
    // the central compartment (mg) — ferx adds `dose.amt` directly to the
    // state, so for an IV bolus the state IS the dose in amount units.
    // Observations must therefore also be in amount (mg), not concentration.
    // True amounts from a 1-cpt model with CL=5, V=50 (k = 0.1/h):
    //   t=1: A(t) = 100·exp(-0.1) = 90.48
    //   t=4: A(t) = 100·exp(-0.4) = 67.03
    //   t=8: A(t) = 100·exp(-0.8) = 44.93
    // Values below are symmetric ±5% perturbations of the true amounts
    // (two subjects below, two above) so the population sample remains
    // centered on the analytical trajectory.
    let obs_times = vec![1.0, 4.0, 8.0];
    let dvs: &[(&str, Vec<f64>)] = &[
        // -5% across all times
        ("S1", vec![85.96, 63.68, 42.68]),
        // +5% across all times
        ("S2", vec![95.00, 70.38, 47.18]),
        // -3% across all times
        ("S3", vec![87.77, 65.02, 43.58]),
        // +3% across all times
        ("S4", vec![93.19, 69.04, 46.28]),
    ];
    let subjects = dvs
        .iter()
        .map(|(id, obs)| Subject {
            id: id.to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: obs.clone(),
            obs_cmts: vec![1; 3],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 3],
            occasions: vec![1u32; 3],
            obs_l2: Vec::new(),
            dose_occasions: vec![1u32],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn fast_foce_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::Foce,
        methods: Vec::new(),
        outer_maxiter: 80,
        outer_gtol: 1e-3,
        inner_maxiter: 50,
        inner_tol: 1e-4,
        run_covariance_step: false,
        interaction: false,
        mu_referencing: false,
        optimizer: Optimizer::Slsqp,
        lbfgs_memory: 5,
        verbose: false,
        ..FitOptions::default()
    }
}

#[test]
fn test_sde_fit_smoke() {
    // Combined smoke test: one SDE fit, three assertions. Each EKF FOCE
    // fit takes ~30–50 min on the 2-core CI runner, so the previous
    // 3-tests-1-assertion split tripled CI wall for no extra coverage.
    let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
    let pop = make_sde_population();
    let opts = fast_foce_opts();
    let result = fit(&parsed.model, &pop, &parsed.model.default_params, &opts)
        .expect("SDE fit should succeed");
    assert!(result.uses_sde, "uses_sde must be true");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    let diff_idx = result
        .theta_names
        .iter()
        .position(|n| n == "DIFF_CENTRAL")
        .expect("DIFF_CENTRAL must be in theta_names");
    let diff_val = result.theta[diff_idx];
    assert!(
        diff_val > 0.0,
        "DIFF_CENTRAL must be positive, got {diff_val}"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_sde_ofv_le_base_ofv() {
    // Reference: the OFV from the identical model fit without the
    // [diffusion] block (BASE_MODEL_SRC). Since [diffusion] adds an extra
    // free parameter (DIFF_CENTRAL ≥ 0) and the EKF observation variance
    // collapses to the residual-only variance when DIFF_CENTRAL → 0, the
    // SDE OFV must be ≤ the base OFV at the optimum.
    // The +1 unit of slack absorbs numerical noise from finite-difference
    // gradients, NLopt's stopping tolerance (`outer_gtol = 1e-3`), and the
    // truncated `outer_maxiter = 80` cap in `fast_foce_opts`; without the
    // slack we'd flake on iterations where the SDE fit stopped a hair
    // short of the base fit's OFV.
    let pop = make_sde_population();
    let opts = fast_foce_opts();

    let parsed_base = parse_full_model(BASE_MODEL_SRC).expect("base model should parse");
    let base_result = fit(
        &parsed_base.model,
        &pop,
        &parsed_base.model.default_params,
        &opts,
    )
    .expect("base fit should succeed");

    let parsed_sde = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
    let sde_result = fit(
        &parsed_sde.model,
        &pop,
        &parsed_sde.model.default_params,
        &opts,
    )
    .expect("SDE fit should succeed");

    assert!(
        sde_result.ofv <= base_result.ofv + 1.0,
        "SDE OFV ({}) should not be worse than base OFV ({}) by more than 1 unit",
        sde_result.ofv,
        base_result.ofv,
    );
}

/// SDE + gn / gn_hybrid must fail with a clear error message.
#[test]
fn sde_gn_returns_error() {
    use crate::types::EstimationMethod;

    let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
    let pop = {
        // Minimal single-subject population (no data needed — error fires before fitting).
        use crate::types::{DoseEvent, Population, Subject};
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0],
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
        Population {
            subjects: vec![subj],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    };

    for method in [EstimationMethod::FoceGn, EstimationMethod::FoceGnHybrid] {
        let opts = FitOptions {
            method,
            ..FitOptions::default()
        };
        let result = fit(&parsed.model, &pop, &parsed.model.default_params, &opts);
        assert!(result.is_err(), "expected error for {:?} + SDE", method);
        let msg = result.unwrap_err();
        assert!(
            msg.contains("gn") || msg.contains("gn_hybrid"),
            "error message should mention gn: {msg}"
        );
    }
}

/// Issue #175: an SDE ([diffusion]) model must surface the experimental
/// feature warning, classified into the `experimental` category. The check
/// is data-independent (`check_experimental_features` takes only the model),
/// so `ferx check` reports it even without a `--data` file. Fast — no fit.
#[test]
fn sde_emits_experimental_warning() {
    let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
    let diags = super::check_experimental_features(&parsed.model);
    let exp = diags
        .iter()
        .find(|d| d.code == "W_EXPERIMENTAL_SDE")
        .expect("SDE model should emit W_EXPERIMENTAL_SDE");
    assert_eq!(exp.severity, crate::diagnostics::Severity::Warning);
    assert_eq!(
        crate::types::classify_warning(&exp.message)
            .category
            .as_str(),
        "experimental"
    );

    // Sanity: a non-SDE model must NOT emit the experimental warning.
    let base = parse_full_model(BASE_MODEL_SRC).expect("base model should parse");
    assert!(
        super::check_experimental_features(&base.model)
            .iter()
            .all(|d| d.code != "W_EXPERIMENTAL_SDE"),
        "non-SDE model should not emit W_EXPERIMENTAL_SDE"
    );
}

/// #1263: dosing features the EKF/SDE path silently drops must **warn**, mirroring the
/// `W_SDE_RESET` precedent, rather than return a plausible wrong number.
///
/// `solve_ekf` applies an `SS=1` record as a single bolus with no equilibration, and
/// never calls `DoseAttrMap::lagtime` at all. Neither gap fails, neither is visible in
/// `IPRED` (the likelihood takes only `p_obs` from the filter), and both are large:
/// measured on a 1-cpt autonomous model, one `SS=1, II=12` record gives `90.48` against
/// `200.27` for the equivalent explicit train.
///
/// Each case is asserted against its **own** control — the same population without the
/// feature — so a warning that fired unconditionally would fail here too.
mod sde_unsupported_dosing_warnings {
    use super::*;

    /// The shared SDE model with a `lagtime` bound, so `model.has_lagtime()` is true.
    const SDE_LAG_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  lagtime = TVLAG

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
  central ~ 0.5

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

    /// The same, without `[diffusion]` — the non-SDE control for the lag warning.
    const ODE_LAG_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  lagtime = TVLAG

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

    fn codes(src: &str, pop: &Population) -> Vec<String> {
        let parsed = parse_full_model(src).expect("model parses");
        let m = &parsed.model;
        super::super::check_model_data_warnings(m, pop, &m.default_params)
            .iter()
            .map(|d| d.code.clone())
            .collect()
    }

    /// `make_sde_population` with every subject's dose flipped to `SS=1, II=12`.
    fn ss_population() -> Population {
        let mut pop = make_sde_population();
        for s in &mut pop.subjects {
            s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        }
        pop
    }

    #[test]
    fn a_lagtime_under_diffusion_warns() {
        let pop = make_sde_population();
        assert!(
            codes(SDE_LAG_MODEL_SRC, &pop).contains(&"W_SDE_LAGTIME".to_string()),
            "a [diffusion] model declaring lagtime must raise W_SDE_LAGTIME"
        );
        // Control 1 — the same lagtime with no [diffusion] block. Isolates the SDE half
        // of the predicate: without this the warning could be keyed on lagtime alone.
        assert!(
            !codes(ODE_LAG_MODEL_SRC, &pop).contains(&"W_SDE_LAGTIME".to_string()),
            "an ODE model with a lagtime must NOT raise W_SDE_LAGTIME"
        );
        // Control 2 — [diffusion] with no lagtime. Isolates the other half.
        assert!(
            !codes(SDE_MODEL_SRC, &pop).contains(&"W_SDE_LAGTIME".to_string()),
            "a [diffusion] model without a lagtime must NOT raise W_SDE_LAGTIME"
        );
    }

    #[test]
    fn a_steady_state_dose_under_diffusion_warns() {
        let ss = ss_population();
        let plain = make_sde_population();
        assert!(
            codes(SDE_MODEL_SRC, &ss).contains(&"W_SDE_STEADY_STATE".to_string()),
            "a [diffusion] model with SS=1 records must raise W_SDE_STEADY_STATE"
        );
        // Control 1 — the same model, same subjects, doses not flagged SS.
        assert!(
            !codes(SDE_MODEL_SRC, &plain).contains(&"W_SDE_STEADY_STATE".to_string()),
            "a [diffusion] model with no SS record must NOT raise W_SDE_STEADY_STATE"
        );
        // Control 2 — the same SS records with no [diffusion] block; the ODE path does
        // equilibrate, so there is nothing to warn about.
        assert!(
            !codes(BASE_MODEL_SRC, &ss).contains(&"W_SDE_STEADY_STATE".to_string()),
            "an ODE model with SS=1 records must NOT raise W_SDE_STEADY_STATE"
        );
    }

    /// The message has to name the workaround, not just the gap — a warning that says
    /// only "not supported" leaves the user with a silently wrong objective and no move.
    #[test]
    fn the_messages_say_what_to_do_instead() {
        let parsed = parse_full_model(SDE_MODEL_SRC).expect("model parses");
        let m = &parsed.model;
        let diags = super::super::check_model_data_warnings(m, &ss_population(), &m.default_params);
        let ss = diags
            .iter()
            .find(|d| d.code == "W_SDE_STEADY_STATE")
            .expect("W_SDE_STEADY_STATE present");
        assert_eq!(ss.severity, crate::diagnostics::Severity::Warning);
        assert!(
            ss.message.contains("explicit dose train"),
            "the SS warning must point at the expansion workaround: {}",
            ss.message
        );

        let parsed_lag = parse_full_model(SDE_LAG_MODEL_SRC).expect("model parses");
        let ml = &parsed_lag.model;
        let lag_diags =
            super::super::check_model_data_warnings(ml, &make_sde_population(), &ml.default_params);
        let lag = lag_diags
            .iter()
            .find(|d| d.code == "W_SDE_LAGTIME")
            .expect("W_SDE_LAGTIME present");
        assert_eq!(lag.severity, crate::diagnostics::Severity::Warning);
        assert!(
            lag.message.contains("record time"),
            "the lag warning must say what happens instead: {}",
            lag.message
        );
    }
}
