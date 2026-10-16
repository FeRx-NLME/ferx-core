//! Shared fixtures for the nested-level sensitivity tests (#1082 PR3): one subject over
//! two levels, and one model per analytic walk — compartment-free, closed form and ODE.
//!
//! The design keeps every side of the layout live:
//!
//! - **Two levels, each with several groups.** `STUDY` has two groups; `ARM` has five,
//!   numbered 1, 2 inside study 1 and 1, 2, 3 inside study 2 (nested by key).
//! - **A weight ≠ 1 that varies across groups.** `NARM` is 25, 50, 100, 200, 400 by arm,
//!   constant inside each, so the arm κ enters as `κ/√NARM` with five different scales.
//! - **An off-diagonal `Ω_IOV` entry.** The study level carries a `block_kappa` of two,
//!   so its prior and its gradient have a cross term; the arm level carries one κ.
//! - **Carryover across both boundaries** (PK models). Doses at `t = 0` (study 1, arm 1),
//!   `30` (study 1, arm 2) and `72` (study 2, arm 2); the first observation of arm 2
//!   (`t = 25`) precedes that arm's dose, and study 2's arm 1 has no dose at all, so its
//!   records see only drug carried over from study 1. Arm 3 likewise.

use std::collections::HashMap;

use crate::types::{CompiledModel, DoseEvent, Population, Subject};

use super::{assign_level_cells, NestedLayout};

/// `(STUDY, ARM, NARM, observation times)` per arm, in time order.
pub(crate) const ARMS: [(f64, f64, f64, [f64; 3]); 5] = [
    (1.0, 1.0, 25.0, [2.0, 8.0, 20.0]),
    (1.0, 2.0, 50.0, [25.0, 34.0, 44.0]),
    (2.0, 1.0, 100.0, [50.0, 60.0, 70.0]),
    (2.0, 2.0, 200.0, [74.0, 80.0, 94.0]),
    (2.0, 3.0, 400.0, [98.0, 110.0, 118.0]),
];

/// `(time, arm index into ARMS)` per dose.
const DOSES: [(f64, usize); 3] = [(0.0, 0), (30.0, 1), (72.0, 3)];

const PARAMETERS: &str = "  level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n  \
     block_kappa (K_A, K_B) = [0.04, 0.012, 0.09] at STUDY\n  \
     kappa K_ARM ~ 2.0 at ARM weight = NARM\n";

/// Compartment-free: the readout reads κ through the individual parameters only.
pub(crate) fn algebraic() -> String {
    format!(
        "[parameters]\n  theta TVE0(10.0, 0.0, 1000.0)\n  theta TVEMAX(20.0, 0.0, 1000.0)\n  \
         theta ET50(15.0, 0.1, 1000.0)\n  omega ETA ~ 0.09\n{PARAMETERS}  sigma ADD ~ 0.5\n\
         [individual_parameters]\n  E0 = TVE0 * exp(K_A)\n  \
         EMAX = TVEMAX * exp(ETA + K_B + K_ARM)\n[structural_model]\n  \
         y = E0 + EMAX * TIME / (TIME + ET50)\n[error_model]\n  DV ~ additive(ADD)\n"
    )
}

/// One-compartment oral, closed form: the study pair on `CL` and `KA`, the arm κ on `V`.
pub(crate) fn closed_form() -> String {
    format!(
        "[parameters]\n  theta TVCL(0.5, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
         theta TVKA(1.2, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n\
         {PARAMETERS}  sigma PROP ~ 0.15 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL + K_A)\n  V = TVV * exp(ETA_V + K_ARM)\n  \
         KA = TVKA * exp(K_B)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    )
}

/// [`closed_form`] written as an `[odes]` model.
pub(crate) fn ode() -> String {
    format!(
        "[parameters]\n  theta TVCL(0.5, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
         theta TVKA(1.2, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n\
         {PARAMETERS}  sigma PROP ~ 0.15 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL + K_A)\n  V = TVV * exp(ETA_V + K_ARM)\n  \
         KA = TVKA * exp(K_B)\n[structural_model]\n  ode(states=[depot, central])\n[odes]\n  \
         d/dt(depot) = -KA * depot\n  d/dt(central) = KA * depot - (CL / V) * central\n\
         [scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n\
         [fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n"
    )
}

/// [`closed_form`] with transit absorption: its IOV subjects run on the ODE twin, the
/// one nested case the analytic walks decline.
pub(crate) fn transit() -> String {
    closed_form()
        .replace(
            "theta TVKA(1.2, 0.01, 50.0)",
            "theta TVMTT(3.0, 0.05, 24.0)\n  theta TVN(3.0)",
        )
        .replace(
            "KA = TVKA * exp(K_B)",
            "MTT = TVMTT * exp(K_B)\n  NTR = TVN",
        )
        .replace(
            "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
            "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)",
        )
}

fn snap(arm: usize) -> HashMap<String, f64> {
    let (study, a, narm, _) = ARMS[arm];
    [
        ("STUDY".to_string(), study),
        ("ARM".to_string(), a),
        ("NARM".to_string(), narm),
    ]
    .into_iter()
    .collect()
}

/// The fixture subject, `id`, with doses when `doses` (the PK models) and every
/// observation set to 1 (see [`with_data`]).
fn subject(id: &str, doses: bool) -> Subject {
    let mut obs_times = Vec::new();
    let mut obs_covariates = Vec::new();
    for (arm, (_, _, _, times)) in ARMS.iter().enumerate() {
        for &t in times {
            obs_times.push(t);
            obs_covariates.push(snap(arm));
        }
    }
    let n = obs_times.len();
    let (dose_events, dose_covariates) = if doses {
        DOSES
            .iter()
            .map(|&(t, arm)| (DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0), snap(arm)))
            .unzip()
    } else {
        (Vec::new(), Vec::new())
    };
    Subject {
        id: id.to_string(),
        doses: dose_events,
        dose_covariates,
        obs_times,
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        covariates: snap(0),
        obs_covariates,
        ..Default::default()
    }
}

/// `model` with the fixture population of `n_subjects` copies of the subject, its level
/// cells assigned (what `fit()` does before estimation), and observations drawn from the
/// model's own prediction at a non-zero `b` with a deterministic perturbation, so the
/// residuals are neither zero nor one-signed.
pub(crate) fn prepared(src: &str, n_subjects: usize) -> (CompiledModel, Population) {
    let model = crate::parser::model_parser::parse_model_string(src).expect("fixture parses");
    let doses = !model.is_algebraic();
    let mut population = Population {
        subjects: (0..n_subjects)
            .map(|i| subject(&(i + 1).to_string(), doses))
            .collect(),
        covariate_names: vec!["STUDY".into(), "ARM".into(), "NARM".into()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    assign_level_cells(&model, &mut population);
    for (i, s) in population.subjects.iter_mut().enumerate() {
        with_data(&model, s, i);
    }
    (model, population)
}

fn with_data(model: &CompiledModel, s: &mut Subject, i: usize) {
    let layout = NestedLayout::of(model, s).expect("prepared");
    let b = point(&layout, 0.6 + 0.2 * i as f64);
    let f = crate::pk::predict_iov(
        model,
        s,
        &model.default_params.theta,
        &b[..model.n_eta],
        &layout.expand(&b),
    );
    let additive = model.is_algebraic();
    s.observations = f
        .iter()
        .enumerate()
        .map(|(j, &v)| {
            let wiggle = (1.7 * j as f64 + i as f64).sin();
            if additive {
                v + 0.8 * wiggle
            } else {
                v * (1.0 + 0.15 * wiggle)
            }
        })
        .collect();
}

/// A compact vector with every entry non-zero and distinct, scaled by `scale`: η first,
/// then each κ block (study, then arm).
pub(crate) fn point(layout: &NestedLayout, scale: f64) -> Vec<f64> {
    (0..layout.n_compact())
        .map(|i| scale * 0.1 * (1.0 + 0.37 * i as f64).sin())
        .collect()
}

/// What compact index `i` is, for failure messages: `η k`, or `level L group g κ NAME`.
/// A mutation of one level's axes names that level here.
pub(crate) fn axis_label(model: &CompiledModel, layout: &NestedLayout, i: usize) -> String {
    if i < layout.n_eta {
        return format!("η {i}");
    }
    let levels = model.nested_levels().expect("a nested model");
    for (l, level) in levels.levels.iter().enumerate() {
        let (off, w) = (layout.level_offset[l], layout.level_width[l]);
        if w > 0 && (off..off + layout.level_count[l] * w).contains(&i) {
            let (g, pos) = ((i - off) / w, (i - off) % w);
            let k = layout
                .kappa_slot
                .iter()
                .position(|&s| s == (l, pos))
                .expect("a κ at every in-level position");
            return format!("level {} group {g} κ {}", level.name, model.kappa_names[k]);
        }
    }
    format!("compact index {i} outside every level")
}

/// The per-cell stacked vector `[η, κ_cell0, …]` of the compact `b` — what the walks and
/// `EbeResult` read.
pub(crate) fn stacked(layout: &NestedLayout, b: &[f64]) -> Vec<f64> {
    let mut out = b[..layout.n_eta].to_vec();
    out.extend(layout.expand(b).into_iter().flatten());
    out
}
