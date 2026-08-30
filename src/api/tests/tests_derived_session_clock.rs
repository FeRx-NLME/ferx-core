//! Tests for the EVID=3/4 session-clock fixes in `compute_extra_output_columns`.
//!
//! All tests build a two-session subject whose second occasion has raw TIME
//! restarting from 0 (identical to the first session) but whose internal
//! `obs_times` are shifted so that ferx-core's monotonic timeline is
//! maintained.  The fixes ensure that `[derived]` columns see raw TIME,
//! not the shifted internal clock.

use super::*;
use crate::types::{
    AggFunction, BloqMethod, CompiledModel, DerivedContext, DerivedExprSpec, DerivedKind,
    ErrorModel, ErrorSpec, GradientMethod, IndivParamPartials, IntegralStep, IntegralWindow,
    ModelParameters, OmegaMatrix, PkModel, PkParams, Population, ScalingSpec, SigmaVector, Subject,
};
use nalgebra::DVector;
use std::collections::HashMap;

// ── shared helpers ────────────────────────────────────────────────────────

/// Minimal CompiledModel — 1-cpt IV, returns constant PK params, no LTO.
/// Caller supplies `derived_exprs`.
fn minimal_model(derived_exprs: Vec<DerivedExprSpec>) -> CompiledModel {
    CompiledModel {
        name: "test_session".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Additive,
        error_spec: ErrorSpec::Single(ErrorModel::Additive),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(|_, _, _, _t: f64| PkParams::default()),
        n_theta: 0,
        n_eta: 0,
        n_epsilon: 1,
        n_kappa: 0,
        kappa_names: Vec::new(),
        theta_names: Vec::new(),
        eta_names: Vec::new(),
        indiv_param_names: Vec::new(),
        indiv_param_partials: IndivParamPartials::empty(),
        default_params: ModelParameters {
            theta: Vec::new(),
            theta_names: Vec::new(),
            theta_lower: Vec::new(),
            theta_upper: Vec::new(),
            theta_fixed: Vec::new(),
            omega: OmegaMatrix::from_diagonal(&[], vec![]),
            omega_fixed: Vec::new(),
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        },
        omega_init_as_sd: Vec::new(),
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: Vec::new(),
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: Some(Box::new(|_t, _c| vec![])),
        pk_indices: Vec::new(),
        eta_map: Vec::new(),
        pk_idx_f64: Vec::new(),
        sel_flat: Vec::new(),
        ode_spec: None,
        dose_attr_map: Default::default(),
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::Fd,
        parse_warnings: Vec::new(),
        has_conditional_eta_params: false,
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs,
        output_columns: Vec::new(),
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    }
}

/// Build a two-session subject whose second occasion has raw TIME restarting
/// from 0.
///
/// Session 0: raw [0, 1, 4]  →  internal [0, 1, 4]
/// Session 1: raw [0, 1, 4]  →  internal [5, 6, 9]  (shift = 5)
///
/// `reset_times[0] = 5.0` marks the boundary.
fn two_session_subject() -> Subject {
    Subject {
        id: "S1".into(),
        doses: Vec::new(),
        // Session 0 at 0,1,4 — Session 1 shifted by 5 to 5,6,9
        obs_times: vec![0.0, 1.0, 4.0, 5.0, 6.0, 9.0],
        obs_raw_times: vec![0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
        observations: vec![1.0; 6],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: vec![5.0], // boundary at shifted t=5
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Minimal SubjectResult for a subject with `n_obs` observations and η=[] .
fn sr_for(n_obs: usize) -> SubjectResult {
    SubjectResult {
        id: "S1".into(),
        eta: DVector::from_vec(vec![]),
        ipred: vec![1.0; n_obs],
        pred: vec![1.0; n_obs],
        iwres: vec![0.0; n_obs],
        cwres: vec![0.0; n_obs],
        npde: vec![],
        npd: vec![],
        ofv_contribution: 0.0,
        cens: vec![0; n_obs],
        n_obs,
        pmix: None,
        mixest: None,
        extra_columns: Vec::new(),
        per_obs_tad: Vec::new(),
        compartment_states: Vec::new(),
        #[cfg(feature = "survival")]
        discrete_rows: Vec::new(),
    }
}

// ── tests ─────────────────────────────────────────────────────────────────

/// PerRow `[derived]` column must expose raw dataset TIME, not the internal
/// shifted clock.  For a two-session subject the second session's raw times
/// [0, 1, 4] are identical to the first session's; the shifted times are
/// [5, 6, 9].  If the fix is correct the column values are [0,1,4,0,1,4].
#[test]
fn derived_per_row_time_is_raw_clock() {
    let derived_exprs = vec![DerivedExprSpec {
        name: "T".into(),
        kind: DerivedKind::PerRow {
            eval: Box::new(|ctx: &DerivedContext| ctx.time),
        },
        uses_compartments: false,
    }];
    let model = minimal_model(derived_exprs);
    let subject = two_session_subject();
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut subjects_results = vec![sr_for(6)];
    compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results, None);
    let col = &subjects_results[0].extra_columns[0].1;
    let expected = vec![0.0, 1.0, 4.0, 0.0, 1.0, 4.0];
    for (j, (&got, &exp)) in col.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "PerRow TIME at obs {j}: got {got}, expected {exp} (raw clock)"
        );
    }
}

/// Aggregate Tmax must return raw TIME.  With the two-session subject and
/// ipred = [1,2,3,3,2,1] (session 0 peaks at raw t=4, session 1 peaks at
/// raw t=0 which is the third session-1 obs) the AggFunction::Tmax over all
/// rows should return the raw time of the global IPRED maximum.
///
/// The global max IPRED value is 3 at index j=2 (raw t=4, shifted t=4) and
/// also at j=3 (raw t=0, shifted t=5).  The first maximum encountered is
/// j=2 with raw t=4, not shifted t=4 or t=5 — both agree here so this test
/// verifies the raw path doesn't regress.  A harder variant follows.
#[test]
fn derived_aggregate_tmax_returns_raw_time() {
    let derived_exprs = vec![DerivedExprSpec {
        name: "TMAX".into(),
        kind: DerivedKind::Aggregate {
            func: AggFunction::Tmax,
            value: Box::new(|ctx: &DerivedContext| ctx.ipred),
            filter: None,
        },
        uses_compartments: false,
    }];
    let model = minimal_model(derived_exprs);
    let subject = two_session_subject();
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut sr = sr_for(6);
    // ipred peak is at j=4 (shifted t=6, raw t=1) which should give tmax=1.
    sr.ipred = vec![1.0, 2.0, 1.5, 0.5, 3.0, 1.0];
    let mut subjects_results = vec![sr];
    compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results, None);
    let col = &subjects_results[0].extra_columns[0].1;
    // All entries should be 1.0 (raw time of peak at j=4).
    for &v in col {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "Tmax should be raw time 1.0, got {v}"
        );
    }
}

/// Obs-based integral over explicit window [0, 4] must produce the correct
/// per-session AUC for a two-session (EVID=4-like) subject.
///
/// Both sessions have raw times [0, 1, 4].  Integrand = ctx.time.
/// AUC = trapezoid([(0,0),(1,1),(4,4)]) = 0·Δt₁ + (0+1)/2·1 + (1+4)/2·3 = 0.5 + 7.5 = 8.0
///
/// With the old (broken) code, session 1's shifted times [5,6,9] would all
/// fail the window filter [0,4] → NaN for every session-1 row.
#[test]
fn derived_integral_obs_per_session_explicit_window() {
    let derived_exprs = vec![DerivedExprSpec {
        name: "AUC".into(),
        kind: DerivedKind::Integral {
            integrand: Box::new(|ctx: &DerivedContext| ctx.time),
            condition: None,
            data_based: true,
            uses_compartments: false,
            window: IntegralWindow::Explicit { from: 0.0, to: 4.0 },
            step: IntegralStep::ObsTimes,
        },
        uses_compartments: false,
    }];
    let model = minimal_model(derived_exprs);
    let subject = two_session_subject();
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut subjects_results = vec![sr_for(6)];
    compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results, None);
    let col = &subjects_results[0].extra_columns[0].1;
    // Expected AUC = 8.0 for every row in each session.
    for (j, &v) in col.iter().enumerate() {
        assert!(
            (v - 8.0).abs() < 1e-12,
            "Integral obs j={j}: got {v}, expected 8.0 (per-session raw-clock AUC)"
        );
    }
}

/// Periodic integral aligns windows to raw TIME, not shifted time.
///
/// Period=5, anchor=0.  All raw obs at [0, 1, 4] satisfy floor(t/5)=0, so
/// every obs lands in the first period window [0, 5).  All three per-session
/// points contribute → AUC = trapezoid([(0,0),(1,1),(4,4)]) = 8.0.
///
/// With the old (broken) code, session-1 obs at shifted times [5, 6, 9] give
/// floor(t/5) = 1 → window [5, 10).  Integrating (5,5),(6,6),(9,9) yields 28.0,
/// not 8.0 — a clear mismatch caught by the `v == 8.0` assertion.
///
/// After the fix, session-1 obs use raw t ∈ {0, 1, 4} → floor(t/5) = 0 →
/// window [0, 5) → correct AUC = 8.0.
#[test]
fn derived_integral_periodic_uses_raw_clock() {
    let derived_exprs = vec![DerivedExprSpec {
        name: "AUC_TAU".into(),
        kind: DerivedKind::Integral {
            integrand: Box::new(|ctx: &DerivedContext| ctx.time),
            condition: None,
            data_based: true,
            uses_compartments: false,
            window: IntegralWindow::Periodic {
                period: 5.0,
                anchor: 0.0,
            },
            step: IntegralStep::ObsTimes,
        },
        uses_compartments: false,
    }];
    let model = minimal_model(derived_exprs);
    let subject = two_session_subject();
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut subjects_results = vec![sr_for(6)];
    compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results, None);
    let col = &subjects_results[0].extra_columns[0].1;
    // All obs land in the raw-clock window [0,5); all three per-session
    // points contribute → AUC=8.0 for every row.
    for (j, &v) in col.iter().enumerate() {
        assert!(
            (v - 8.0).abs() < 1e-12,
            "Periodic integral at obs {j}: got {v}, expected 8.0"
        );
    }
}

/// Single-session subjects are unaffected by the multi-session path.
///
/// A plain subject with no resets should produce the same AUC as before
/// (regression guard).
#[test]
fn derived_integral_single_session_unchanged() {
    let derived_exprs = vec![DerivedExprSpec {
        name: "AUC".into(),
        kind: DerivedKind::Integral {
            integrand: Box::new(|ctx: &DerivedContext| ctx.time),
            condition: None,
            data_based: true,
            uses_compartments: false,
            window: IntegralWindow::Explicit { from: 0.0, to: 4.0 },
            step: IntegralStep::ObsTimes,
        },
        uses_compartments: false,
    }];
    let model = minimal_model(derived_exprs);
    let subject = Subject {
        id: "SINGLE".into(),
        doses: Vec::new(),
        obs_times: vec![0.0, 1.0, 4.0],
        obs_raw_times: vec![0.0, 1.0, 4.0],
        observations: vec![1.0; 3],
        obs_cmts: vec![1; 3],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 3],
        occasions: vec![1, 1, 1],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let mut subjects_results = vec![sr_for(3)];
    compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results, None);
    let col = &subjects_results[0].extra_columns[0].1;
    // AUC = trapezoid([(0,0),(1,1),(4,4)]) = 8.0
    for &v in col {
        assert!(
            (v - 8.0).abs() < 1e-12,
            "Single-session AUC should be 8.0, got {v}"
        );
    }
}
