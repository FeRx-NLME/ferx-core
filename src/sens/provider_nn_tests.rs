//! `[covariate_nn]` (DCM) subjects on the TV-cov event-driven walk (#1300).
//!
//! Before #1300 a DCM whose network read a time-varying input sent every such subject
//! to reconverged finite-difference gradients on both loops, because
//! `tvcov_analytical_supported` bounded the walk on `model.n_theta + n_eta` and the
//! generated weight thetas alone blow past `MAX_TVCOV_AXES`. The walk now seeds the
//! *program's* axes (declared θ, η, one axis per network output), chains the weight
//! columns in through the network's backprop Jacobian per event, and walks the θ
//! columns in chunks. These tests pin:
//!
//! * the per-event derivative builder against central FD of the f64 `pk_param_fn` —
//!   every θ column (weights included) and the mixed `∂²p/∂η∂θ` block;
//! * the full outer provider against central FD of the production predictor on a
//!   fixture that **straddles the cap** (`n_theta + n_eta > MAX_TVCOV_AXES`, so the walk
//!   runs more than one θ chunk) with a multi-dose, time-varying-input subject;
//! * the inner η-gradient against the outer's η block and against FD;
//! * routing: the analytic route is reported for the whole population, and a model
//!   that genuinely exceeds the *program* cap, or reads a weight directly, still declines.

use super::tests::check_full_provider_vs_fd;
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::pk::compute_predictions_with_tv;
use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;

/// A 2 → 8 → 3 DCM on a two-compartment IV model — the shape of the vancomycin repro in
/// #1300: 51 generated weights plus one declared θ (`TVQ`), three η. With `n_eta = 3` the
/// outer walk seeds its 52 θ columns in chunks of 21, so this fixture exercises the
/// multi-chunk path, not just the NN chain.
fn dcm_src(extra_thetas: usize, direct_weight_ref: bool) -> String {
    let mut thetas = String::from("  theta TVQ(6.0, 0.1, 50.0)\n");
    for i in 0..extra_thetas {
        thetas.push_str(&format!("  theta TVX{i}(1.0, 0.1, 10.0)\n"));
    }
    let cl_line = if direct_weight_ref {
        "  CL = TYPICAL_PK.CL * exp(ETA_CL) + B_TYPICAL_PK_2_1"
    } else {
        "  CL = TYPICAL_PK.CL * exp(ETA_CL)"
    };
    format!(
        r#"
[parameters]
{thetas}  omega ETA_CL ~ 0.1
  omega ETA_V1 ~ 0.1
  omega ETA_V2 ~ 0.1
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs     = [CRCL, WT]
  center     = [90, 70]
  scale      = [30, 15]
  outputs    = [CL, V1, V2]
  layers     = [8]
  activation = tanh
  output     = softplus
  init       = [4.5, 60.0, 50.0]

[individual_parameters]
{cl_line}
  V1 = TYPICAL_PK.V1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TYPICAL_PK.V2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP)
"#
    )
}

fn dcm_model() -> CompiledModel {
    parse_model_string(&dcm_src(0, false)).expect("DCM parses")
}

/// A probe point off every symmetric point: `init` zeroes the output-layer weights, which
/// makes every hidden-layer weight gradient *exactly* zero (`∂a/∂W_hidden` runs through
/// `W_outᵀ`), so a parity test at the parsed defaults would confirm nothing about the
/// first layer. The per-weight jitter keeps the output biases near their `init` targets
/// while lighting up every column.
fn probe_theta(model: &CompiledModel) -> Vec<f64> {
    model
        .default_params
        .theta
        .iter()
        .enumerate()
        .map(|(i, &t)| t + 0.13 * ((i as f64) * 0.7).sin())
        .collect()
}

fn snap(crcl: f64, wt: f64) -> HashMap<String, f64> {
    HashMap::from([("CRCL".to_string(), crcl), ("WT".to_string(), wt)])
}

/// Multi-dose IV subject whose `CRCL` drifts across every record (dose rows included), so
/// the network output changes at every event, and later doses land with residual drug
/// present — the incoming side of each dose event is live, per CLAUDE.md's
/// non-degeneracy rule. `WT` stays put.
fn tv_subject() -> Subject {
    let obs_times = vec![1.0, 6.0, 13.0, 20.0, 30.0];
    let n = obs_times.len();
    Subject {
        id: "tv".into(),
        doses: vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 1000.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        observations: vec![20.0, 12.0, 15.0, 9.0, 8.0],
        obs_cmts: vec![1; n],
        covariates: snap(95.0, 70.0),
        dose_covariates: vec![snap(95.0, 70.0), snap(72.0, 70.0), snap(58.0, 70.0)],
        obs_covariates: vec![
            snap(95.0, 70.0),
            snap(84.0, 70.0),
            snap(70.0, 70.0),
            snap(62.0, 70.0),
            snap(55.0, 70.0),
        ],
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Default::default()
    }
}

/// Same dosing, every snapshot identical — a static-input control.
fn static_subject() -> Subject {
    let mut s = tv_subject();
    s.id = "static".into();
    s.dose_covariates = vec![snap(95.0, 70.0); 3];
    s.obs_covariates = vec![snap(95.0, 70.0); 5];
    s
}

fn population(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: vec!["CRCL".into(), "WT".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// θ indices of the generated weight block.
fn weight_cols(model: &CompiledModel) -> Vec<usize> {
    model
        .covariate_nns
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.mapper.mlp().n_weights())
        .collect()
}

/// The fixture must straddle the cap, or the multi-chunk walk is never reached and the
/// test pins nothing #1300 changed.
fn assert_straddles_cap(model: &CompiledModel) {
    assert!(
        model.n_theta + model.n_eta > MAX_TVCOV_AXES,
        "fixture must exceed MAX_TVCOV_AXES on n_theta + n_eta ({} + {}) to exercise the \
         chunked walk",
        model.n_theta,
        model.n_eta
    );
    assert!(
        tvcov_program_axes(model) <= MAX_TVCOV_AXES,
        "fixture's program width {} must fit the cap",
        tvcov_program_axes(model)
    );
}

/// The per-event builder: every θ column — declared and weight — and the η and mixed
/// blocks against central FD of the production `pk_param_fn` at one covariate snapshot.
#[test]
fn nn_param_derivatives_match_fd_of_pk_param_fn() {
    let model = dcm_model();
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("program");
    let theta = probe_theta(&model);
    let eta = vec![0.12, -0.08, 0.05];
    let cov = snap(64.0, 70.0);
    let slots = prog.pk_slots_ref();

    let pd = nn_param_derivatives_at_cov(&model, prog, &cov, &theta, &eta).expect("in scope");
    let pk_at = |th: &[f64], et: &[f64]| (model.pk_param_fn)(th, et, &cov, 0.0);

    let wcols = weight_cols(&model);
    let mut live_weight_cols = 0usize;
    for (i, &slot) in slots.iter().enumerate() {
        // ∂p/∂θ, all columns.
        for m in 0..model.n_theta {
            let h = 1e-6 * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += h;
            let mut tm = theta.clone();
            tm[m] -= h;
            let fd = (pk_at(&tp, &eta).values[slot] - pk_at(&tm, &eta).values[slot]) / (2.0 * h);
            approx::assert_relative_eq!(
                pd.dp_dtheta[i][m],
                fd,
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            if wcols.contains(&m) && pd.dp_dtheta[i][m].abs() > 1e-6 {
                live_weight_cols += 1;
            }
            // ∂²p/∂η∂θ, mixed 4-point. Both steps at `1e-4` (θ scaled): the stencil's
            // roundoff goes as `ε/(h_θ·h_η)`, so a `1e-6` θ step here would leave only
            // ~5 digits (measured 2.3e-5 relative at that step). At `1e-4` the worst
            // realised error over every entry is 2.9e-6 relative (truncation, the θ step
            // reaching 7e-4 on `TVQ ≈ 6`); the `1e-4` bound is 30× that.
            let s = 1e-4 * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            for k in 0..model.n_eta {
                let he = 1e-4;
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let fd2 = (pk_at(&tp, &ep).values[slot]
                    - pk_at(&tp, &em).values[slot]
                    - pk_at(&tm, &ep).values[slot]
                    + pk_at(&tm, &em).values[slot])
                    / (4.0 * s * he);
                approx::assert_relative_eq!(
                    pd.d2p_detadtheta[i][k][m],
                    fd2,
                    max_relative = 1e-4,
                    epsilon = 1e-9
                );
            }
        }
        // ∂p/∂η.
        for k in 0..model.n_eta {
            let he = 1e-6;
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let fd =
                (pk_at(&theta, &ep).values[slot] - pk_at(&theta, &em).values[slot]) / (2.0 * he);
            approx::assert_relative_eq!(pd.dp_deta[i][k], fd, max_relative = 1e-6, epsilon = 1e-9);
        }
    }
    assert!(
        live_weight_cols > 0,
        "no weight column carries a derivative — the chain through ∂z/∂w is dead"
    );
}

/// Outer provider on a TV-input, multi-dose DCM subject, on a fixture that straddles the
/// cap: value, `∂f/∂η`, `∂²f/∂η²`, every `∂f/∂θ` column (weights included), and the mixed
/// `∂²f/∂η∂θ` block against central FD of `compute_predictions_with_tv`.
#[test]
fn dcm_tv_input_outer_provider_matches_fd_of_production() {
    let model = dcm_model();
    assert_straddles_cap(&model);
    let subject = tv_subject();
    assert!(
        subject.has_tv_covariates(),
        "fixture must be a TV-cov subject"
    );
    assert!(
        subject_routes_to_event_walk(&model, &subject),
        "TV subject must route to the event walk"
    );
    let theta = probe_theta(&model);
    let eta = vec![0.12, -0.08, 0.05];

    let sens = subject_sensitivities(&model, &subject, &theta, &eta)
        .expect("DCM with a time-varying input must be served analytically (#1300)");
    // The weight columns must be live on this subject, or the parity below is vacuous.
    let wcols = weight_cols(&model);
    let live = sens
        .obs
        .iter()
        .flat_map(|o| wcols.iter().map(move |&m| o.df_dtheta[m].abs()))
        .fold(0.0f64, f64::max);
    assert!(
        live > 1e-6,
        "weight columns carry no sensitivity ({live:e})"
    );
    let live_mixed = sens
        .obs
        .iter()
        .flat_map(|o| wcols.iter().map(move |&m| o.d2f_deta_dtheta[m].abs()))
        .fold(0.0f64, f64::max);
    assert!(
        live_mixed > 1e-6,
        "mixed η-weight block is dead ({live_mixed:e})"
    );

    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

/// Static-input control through the same code: the walk is reached only via a TV
/// covariate, so hand the subject a time-varying covariate the network does **not**
/// read. Every event then evaluates the network at the same input, and the provider must
/// still match FD — the chain is exact whether or not `z` moves between events.
#[test]
fn dcm_static_input_on_event_walk_matches_fd_of_production() {
    let model = dcm_model();
    assert_straddles_cap(&model);
    let mut subject = static_subject();
    for (j, m) in subject.obs_covariates.iter_mut().enumerate() {
        m.insert("CONMED".into(), j as f64);
    }
    assert!(subject.has_tv_covariates());
    let theta = probe_theta(&model);
    let eta = vec![0.12, -0.08, 0.05];
    assert!(subject_sensitivities(&model, &subject, &theta, &eta).is_some());
    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

/// Inner η-gradient of the TV-input DCM subject: analytic (not `None`), equal to the outer
/// jet's η block, and equal to central FD of the production predictor.
#[test]
fn dcm_tv_input_inner_eta_grad_matches_outer_and_fd() {
    let model = dcm_model();
    let subject = tv_subject();
    let theta = probe_theta(&model);
    let eta = vec![0.12, -0.08, 0.05];

    let inner = subject_eta_grad(&model, &subject, &theta, &eta)
        .expect("DCM TV subject must take the analytic inner η-gradient (#1300)");
    let outer = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer");
    assert_eq!(inner.len(), outer.obs.len());
    let pred = |e: &[f64], j: usize| compute_predictions_with_tv(&model, &subject, &theta, e)[j];
    for (j, (i, o)) in inner.iter().zip(outer.obs.iter()).enumerate() {
        approx::assert_relative_eq!(i.f, o.f, max_relative = 1e-12);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(i.df_deta[k], o.df_deta[k], max_relative = 1e-10);
            let he = 1e-6;
            let mut ep = eta.clone();
            ep[k] += he;
            let mut em = eta.clone();
            em[k] -= he;
            let fd = (pred(&ep, j) - pred(&em, j)) / (2.0 * he);
            approx::assert_relative_eq!(i.df_deta[k], fd, max_relative = 2e-4, epsilon = 1e-7);
        }
    }
}

/// Routing: the whole population — TV-input and static-input subjects alike — reports
/// the analytic route on both loops, so the "N of M subjects use finite-difference inner
/// gradients" warning is absent; and the model-level predicates agree.
#[test]
fn dcm_tv_input_population_reports_analytic_and_no_fd_fallback() {
    let model = dcm_model();
    assert!(
        tvcov_analytical_supported(&model),
        "gate must admit the DCM"
    );
    assert!(analytical_supported(&model));
    assert!(analytic_outer_gradient_available(&model));
    let pop = population(vec![tv_subject(), static_subject()]);
    let theta = probe_theta(&model);
    assert_eq!(
        crate::estimation::inner_optimizer::fd_fallback_warning(&model, &pop, &theta),
        None,
        "no subject may fall back to FD"
    );
    let summary = crate::estimation::inner_optimizer::gradient_route_summary(
        &model,
        &pop,
        crate::types::GradientMethod::Auto,
    );
    assert!(
        summary.starts_with("analytic (Dual2)") && !summary.contains("FD"),
        "expected an all-analytic route summary, got: {summary}"
    );
}

/// A DCM whose *program* width exceeds the cap (declared θ + η + network outputs) still
/// declines — loudly, to FD — rather than dispatching past the table.
#[test]
fn dcm_exceeding_program_axis_cap_declines_to_fd() {
    // 1 + 19 declared θ + 3 η + 3 outputs = 26 > 24.
    let model = parse_model_string(&dcm_src(19, false)).expect("parses");
    assert!(tvcov_program_axes(&model) > MAX_TVCOV_AXES);
    assert!(!tvcov_analytical_supported(&model));
    let subject = tv_subject();
    let theta = probe_theta(&model);
    let eta = vec![0.12, -0.08, 0.05];
    assert!(subject_sensitivities(&model, &subject, &theta, &eta).is_none());
    assert!(subject_eta_grad(&model, &subject, &theta, &eta).is_none());
    // And with one fewer declared θ it fits again — the bound is the program width.
    let model = parse_model_string(&dcm_src(17, false)).expect("parses");
    assert_eq!(tvcov_program_axes(&model), MAX_TVCOV_AXES);
    assert!(tvcov_analytical_supported(&model));
    let theta = probe_theta(&model);
    assert!(subject_sensitivities(&model, &subject, &theta, &eta).is_some());
}

/// A model that reads a generated weight θ directly breaks the output-channel
/// factorization; the parser records it and the walk declines (same rule as
/// `nn_theta_gradient`).
#[test]
fn dcm_direct_weight_reference_declines_to_fd() {
    let model = parse_model_string(&dcm_src(0, true)).expect("parses");
    assert!(
        !nn_output_chain_supported(&model),
        "parser must have recorded the direct weight reference"
    );
    assert!(!tvcov_analytical_supported(&model));
    let subject = tv_subject();
    let theta = probe_theta(&model);
    let eta = vec![0.12, -0.08, 0.05];
    assert!(subject_sensitivities(&model, &subject, &theta, &eta).is_none());
}
