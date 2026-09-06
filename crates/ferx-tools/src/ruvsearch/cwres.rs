//! The CWRES pre-screen (#1182): Pharmpy's `ruvsearch` fits its candidates
//! not to the data but to the parent's **conditional weighted residuals**.
//!
//! The screening dataset has one row per observation of the parent fit with
//! `DV = CWRES`, the parent's `IPRED` beside it as a covariate, and the
//! subject's dose history kept so the `TAD` built-in reads the same time after
//! dose the real model would. The screening base is `Y = θ + η + ε`, a
//! compartment-free model with one θ, one η and an additive σ; each candidate
//! is that model with the residual error reshaped:
//!
//! | candidate | CWRES model's `[error_model]` | shape parameter, init |
//! |---|---|---|
//! | `IIV_on_RUV` | `additive(SIG)` + `iiv_on_ruv = ETA_RUV` | `omega ETA_RUV ~ 0.09` |
//! | `power` | `additive(SIG * IPREDC^RUV_POW)` | `theta RUV_POW(0.1)` |
//! | `combined` | `additive(SIG * sqrt(1 + (RUV_ADD / IPREDC)²))` | `theta RUV_ADD(√(|min IPRED| / 2))` |
//! | `time_varying{i}` | `additive(SIG * (if (TAD < c_i) RUV_TV else 1.0))` | `theta RUV_TV(0.1)` |
//!
//! These are Pharmpy's CWRES models spelled in ferx: `power` is Pharmpy's
//! `ε · IPRED^power1` (a magnitude on the additive slot — the CWRES model's
//! own prediction is a constant, so the loading has to come from the
//! covariate); `combined` is Pharmpy's `ε_p + ε_a / IPRED`, whose variance
//! `σ_p² + σ_a² / IPRED²` is written as one σ with the ratio `RUV_ADD = σ_a /
//! σ_p` as a θ — the same two-parameter family, on parameters the magnitude
//! grammar can carry. `IPREDC` is the column name: `IPRED` itself is a
//! reserved word in a magnitude expression.
//!
//! What the screen hands the refit is the **shape** it estimated, mapped
//! back as Pharmpy does: `power = RUV_POW + 1` (floored at `0.02`), the
//! `ETA_RUV` variance, the time-varying θ. The combined split is not
//! transferred: its σ live on the CWRES scale, and Pharmpy's transfer of
//! them to the concentration scale is an init a full refit does not need —
//! the candidate starts from the parent's proportional σ and the
//! `(min DV / 2)²` additive default instead.

use std::collections::HashMap;

use ferx_core::edit::ModelText;
use ferx_core::{FitResult, Population, Subject};

use super::RuvFeature;
use crate::search::Candidate;

/// The covariate column the parent's `IPRED` becomes in the screening data.
pub const IPRED_COLUMN: &str = "IPREDC";

/// Pharmpy's `2.225e-307` guard: an `IPRED` of exactly zero would zero a
/// power loading and divide the combined one.
const IPRED_FLOOR: f64 = 2.225e-307;

/// The screening models of one iteration and the data they fit.
#[derive(Debug)]
pub struct Screen {
    pub population: Population,
    /// The base first, then one per feature.
    pub candidates: Vec<Candidate>,
    pub base_id: String,
    /// `(feature, candidate id)` for every screened feature.
    pub features: Vec<(RuvFeature, String)>,
}

impl Screen {
    /// The screening dataset and models for `features`, from the parent's
    /// fit on `population`.
    pub fn build(
        fit: &FitResult,
        population: &Population,
        features: &[RuvFeature],
        tad_cutoffs: &[f64],
        iteration: usize,
    ) -> Result<Screen, String> {
        let screening = screening_population(fit, population)?;
        let ipred_min = screening
            .subjects
            .iter()
            .flat_map(|s| s.obs_covariates.iter())
            .filter_map(|c| c.get(IPRED_COLUMN).copied())
            .fold(f64::INFINITY, f64::min);
        let base_id = format!("cwres-base-{iteration}");
        let mut candidates = vec![Candidate::new(&base_id, model_text(&base_model()))];
        let mut out_features = Vec::with_capacity(features.len());
        for &feature in features {
            let id = format!("cwres-{}-{iteration}", feature.label());
            let src = match feature {
                RuvFeature::IivOnRuv => iiv_on_ruv_model(),
                RuvFeature::Power => power_model(),
                RuvFeature::Combined => combined_model(ipred_min),
                RuvFeature::TimeVarying(i) => {
                    let Some(&c) = i.checked_sub(1).and_then(|k| tad_cutoffs.get(k)) else {
                        continue;
                    };
                    time_varying_model(c)
                }
            };
            candidates.push(Candidate::new(&id, model_text(&src)).parent(base_id.clone()));
            out_features.push((feature, id));
        }
        Ok(Screen {
            population: screening,
            candidates,
            base_id,
            features: out_features,
        })
    }
}

/// The parent's CWRES as a dataset: one subject per subject, one record per
/// observation with a finite CWRES and IPRED, the doses kept for `TAD`.
pub fn screening_population(
    fit: &FitResult,
    population: &Population,
) -> Result<Population, String> {
    if fit.subjects.len() != population.subjects.len() {
        return Err(format!(
            "the parent fit has {} subjects but the data {}; the CWRES cannot be matched to \
             the records",
            fit.subjects.len(),
            population.subjects.len()
        ));
    }
    let mut subjects = Vec::with_capacity(population.subjects.len());
    let mut n_obs = 0usize;
    for (sr, subj) in fit.subjects.iter().zip(&population.subjects) {
        if sr.id != subj.id {
            return Err(format!(
                "the parent fit's subject `{}` is not the data's `{}` at the same position",
                sr.id, subj.id
            ));
        }
        let keep: Vec<usize> = (0..subj.observations.len())
            .filter(|&j| {
                sr.cwres.get(j).is_some_and(|c| c.is_finite())
                    && sr.ipred.get(j).is_some_and(|f| f.is_finite())
                    && subj.cens.get(j).copied().unwrap_or(0) == 0
            })
            .collect();
        if keep.is_empty() {
            continue;
        }
        n_obs += keep.len();
        let snap = |j: usize| -> HashMap<String, f64> {
            let ipred = sr.ipred[j].abs().max(IPRED_FLOOR);
            [(IPRED_COLUMN.to_string(), ipred)].into_iter().collect()
        };
        let pick = |v: &[f64]| -> Vec<f64> {
            if v.is_empty() {
                Vec::new()
            } else {
                keep.iter().map(|&j| v[j]).collect()
            }
        };
        let pick_u32 = |v: &[u32]| -> Vec<u32> {
            if v.is_empty() {
                Vec::new()
            } else {
                keep.iter().map(|&j| v[j]).collect()
            }
        };
        subjects.push(Subject {
            id: subj.id.clone(),
            doses: subj.doses.clone(),
            obs_times: pick(&subj.obs_times),
            obs_raw_times: pick(&subj.obs_raw_times),
            observations: keep.iter().map(|&j| sr.cwres[j]).collect(),
            obs_cmts: vec![1; keep.len()],
            covariates: snap(keep[0]),
            obs_covariates: keep.iter().map(|&j| snap(j)).collect(),
            cens: vec![0; keep.len()],
            occasions: pick_u32(&subj.occasions),
            ..Default::default()
        });
    }
    if n_obs == 0 {
        return Err("the parent fit has no finite CWRES to screen on".into());
    }
    Ok(Population {
        subjects,
        covariate_names: vec![IPRED_COLUMN.to_string()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    })
}

fn model_text(src: &str) -> ModelText {
    ModelText::parse(src).expect("the screening models are well-formed")
}

/// `Y = θ + η + ε` — Pharmpy's `_create_base_model`: `theta 0.1`, `omega
/// 0.01`, `sigma 1.0`, FOCE with interaction.
fn base_model() -> String {
    screening_model("  DV ~ additive(SIG)", "")
}

fn iiv_on_ruv_model() -> String {
    screening_model(
        "  DV ~ additive(SIG)\n  iiv_on_ruv = ETA_RUV",
        "  omega ETA_RUV ~ 0.09\n",
    )
}

fn power_model() -> String {
    screening_model(
        &format!("  DV ~ additive(SIG * {IPRED_COLUMN}^RUV_POW)"),
        "  theta RUV_POW(0.1, -10.0, 10.0)\n",
    )
}

/// `σ_p² + σ_a² / IPRED²` as `SIG² · (1 + (RUV_ADD / IPRED)²)`. Pharmpy's
/// inits are `σ_p² = 1` and `σ_a² = |min IPRED| / 2`, so the ratio starts at
/// `√(|min IPRED| / 2)`.
fn combined_model(ipred_min: f64) -> String {
    let ratio = if ipred_min.is_finite() && ipred_min > 0.0 {
        (ipred_min / 2.0).sqrt()
    } else {
        0.1
    };
    screening_model(
        &format!("  DV ~ additive(SIG * sqrt(1.0 + (RUV_ADD / {IPRED_COLUMN})^2))"),
        &format!("  theta RUV_ADD({}, 0.0, 1000000.0)\n", ferx_num(ratio)),
    )
}

fn time_varying_model(cutoff: f64) -> String {
    screening_model(
        &format!(
            "  DV ~ additive(SIG * (if (TAD < {}) RUV_TV else 1.0))",
            ferx_num(cutoff)
        ),
        "  theta RUV_TV(0.1, 0.001, 100.0)\n",
    )
}

fn screening_model(error_model: &str, extra_params: &str) -> String {
    format!(
        "[parameters]
  theta TVB(0.1)
{extra_params}  omega ETA_B ~ 0.01
  sigma SIG ~ 1.0

[individual_parameters]
  B = TVB + ETA_B

[structural_model]
  y = B

[covariates]
  {IPRED_COLUMN} continuous

[error_model]
{error_model}

[fit_options]
  method     = focei
  covariance = false
  checkpoint = false
"
    )
}

/// A `.ferx` number: the value with a `.0` tail when integral.
fn ferx_num(v: f64) -> String {
    let s = format!("{v}");
    if s.contains(['.', 'e', 'E']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// The full-model init a screening fit implies for `feature` — Pharmpy's
/// `_create_best_model` mapping. `None` for `combined`, whose CWRES-scale σ
/// do not transfer (see the module docs).
pub fn init_from_screen(feature: RuvFeature, fit: &FitResult) -> Option<super::Init> {
    let theta = |name: &str| -> Option<f64> {
        fit.theta_names
            .iter()
            .position(|n| n == name)
            .and_then(|i| fit.theta.get(i).copied())
            .filter(|v| v.is_finite())
    };
    match feature {
        RuvFeature::Power => theta("RUV_POW").map(|p| {
            // Pharmpy: `power = theta + 1.0`, and `if power < 0.01: power = 0.02`.
            let p = p + 1.0;
            super::Init {
                value: if p < 0.01 { 0.02 } else { p },
            }
        }),
        RuvFeature::IivOnRuv => fit
            .eta_names
            .iter()
            .position(|n| n == "ETA_RUV")
            .map(|k| fit.omega[(k, k)])
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| super::Init { value: v }),
        RuvFeature::TimeVarying(_) => theta("RUV_TV")
            .filter(|v| *v > 0.0)
            .map(|v| super::Init { value: v }),
        RuvFeature::Combined => None,
    }
}
