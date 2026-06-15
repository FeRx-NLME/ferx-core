//! GPU-accelerated SAEM E-step support (issue #368).
//!
//! This module provides a batched evaluation of the per-subject individual
//! negative log-likelihood (NLL) for the **analytical 1-compartment IV-bolus**
//! model on the GPU, via a `cubecl` kernel (wgpu/Metal + CUDA backends). It is
//! the reusable compute primitive the SAEM E-step calls when running on GPU.
//!
//! ## Design: split the work
//!
//! The map from `(theta, eta, covariates)` to PK parameters is an arbitrary
//! DSL-compiled closure (`model.pk_param_fn`) and is **not** GPU-portable. So
//! the work is split:
//!
//! * **CPU** evaluates `pk_param_fn` once per subject to get `(CL_i, V_i)` —
//!   cheap, and faithful to whatever parameterization / covariate model the
//!   user wrote.
//! * **GPU** evaluates the per-observation analytical prediction
//!   `C(t) = Σ_doses (amt/V)·exp(-(CL/V)·(t−t_dose))` and the Gaussian data
//!   term `Σ_j [ resid²/v + ln v ]` across all subjects in parallel — the part
//!   that scales with the number of observations and is pure arithmetic.
//!
//! The η-prior term `η'Ω⁻¹η` and `ln|Ω|` are computed on the CPU in `f64`
//! (cheap, and keeps the prior in full precision), then combined with the GPU
//! data term: `nll_i = 0.5·(η'Ω⁻¹η + ln|Ω| + data_ll_i)`. This is exactly the
//! quantity [`crate::stats::likelihood::individual_nll_into`] returns, so the
//! CPU path is the reference and the GPU path must match it (within `f32`
//! tolerance — Metal has no `f64`).
//!
//! ## Fallback
//!
//! Everything outside the supported subset (other PK models, infusions,
//! steady-state doses, resets, time-varying covariates, M3 BLOQ, SDE, TTE)
//! routes to the CPU path. The GPU path is also skipped when the `gpu` feature
//! is not built or no device initialises. See [`batched_individual_nll`].

use crate::pk::EventPkParams;
use crate::stats::likelihood::individual_nll_into;
use crate::types::{
    BloqMethod, CompiledModel, ErrorModel, ErrorSpec, OmegaMatrix, PkModel, Population,
    SaemBackend, PK_IDX_CL, PK_IDX_V,
};
use nalgebra::DVector;

/// Numeric code for the error model, matching the kernel's branch.
fn error_code(em: ErrorModel) -> f32 {
    match em {
        ErrorModel::Additive => 0.0,
        ErrorModel::Proportional => 1.0,
        ErrorModel::Combined => 2.0,
    }
}

/// True when `model` is in the GPU-supported subset (model-level checks).
/// Per-subject checks (doses, covariates, resets) happen in [`flatten`].
pub fn gpu_model_supported(model: &CompiledModel) -> bool {
    if model.pk_model != PkModel::OneCptIv {
        return false;
    }
    if model.is_sde() {
        return false;
    }
    if model.bloq_method != BloqMethod::Drop {
        // M3 likelihood (censored rows) is not implemented in the kernel.
        return false;
    }
    // Single-endpoint Gaussian error only.
    matches!(model.error_spec, ErrorSpec::Single(_))
}

/// Flattened, `f32`-ready batch for the kernel. All per-subject arrays are
/// indexed by subject; observations and doses are concatenated with per-subject
/// `(offset, count)` slices.
// Fields are read only by the GPU kernel; in a non-`gpu` build the batch is
// built but never consumed (the dispatcher falls back to CPU before that).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
struct FlatBatch {
    n_subjects: usize,
    cl: Vec<f32>,
    v: Vec<f32>,
    dose_off: Vec<u32>,
    dose_cnt: Vec<u32>,
    dose_amt: Vec<f32>,
    dose_t: Vec<f32>,
    obs_off: Vec<u32>,
    obs_cnt: Vec<u32>,
    obs_y: Vec<f32>,
    obs_t: Vec<f32>,
    /// `[error_code, sigma0, sigma1]`.
    params: [f32; 3],
}

/// Build a [`FlatBatch`] for the population, or `None` when any subject is
/// outside the GPU-supported subset (caller falls back to CPU).
fn flatten(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    etas: &[Vec<f64>],
    sigma: &[f64],
) -> Option<FlatBatch> {
    if !gpu_model_supported(model) {
        return None;
    }
    let em = match &model.error_spec {
        ErrorSpec::Single(em) => *em,
        ErrorSpec::PerCmt(_) => return None,
    };
    let s0 = *sigma.first().unwrap_or(&0.0) as f32;
    let s1 = *sigma.get(1).unwrap_or(&0.0) as f32;

    let n = population.subjects.len();
    let mut cl = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut dose_off = Vec::with_capacity(n);
    let mut dose_cnt = Vec::with_capacity(n);
    let mut dose_amt = Vec::new();
    let mut dose_t = Vec::new();
    let mut obs_off = Vec::with_capacity(n);
    let mut obs_cnt = Vec::with_capacity(n);
    let mut obs_y = Vec::new();
    let mut obs_t = Vec::new();

    for (i, subject) in population.subjects.iter().enumerate() {
        // Per-subject feature gates: anything the single-CL/V, superposition
        // kernel cannot express routes the whole batch to the CPU.
        if subject.has_tv_covariates() || subject.has_resets() || subject.has_ss_doses() {
            return None;
        }
        #[cfg(feature = "survival")]
        if !subject.obs_records.is_empty() {
            return None;
        }
        for d in &subject.doses {
            if d.is_infusion() || d.ss {
                return None;
            }
        }

        let eta = &etas[i];
        let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);
        let cl_i = pk.values[PK_IDX_CL];
        let v_i = pk.values[PK_IDX_V];
        if !(cl_i.is_finite() && v_i.is_finite()) || v_i <= 0.0 {
            return None;
        }
        cl.push(cl_i as f32);
        v.push(v_i as f32);

        dose_off.push(dose_amt.len() as u32);
        dose_cnt.push(subject.doses.len() as u32);
        for d in &subject.doses {
            dose_amt.push(d.amt as f32);
            dose_t.push(d.time as f32);
        }

        obs_off.push(obs_y.len() as u32);
        obs_cnt.push(subject.observations.len() as u32);
        for (j, &y) in subject.observations.iter().enumerate() {
            obs_y.push(y as f32);
            obs_t.push(subject.obs_times[j] as f32);
        }
    }

    Some(FlatBatch {
        n_subjects: n,
        cl,
        v,
        dose_off,
        dose_cnt,
        dose_amt,
        dose_t,
        obs_off,
        obs_cnt,
        obs_y,
        obs_t,
        params: [error_code(em), s0, s1],
    })
}

/// CPU reference: per-subject individual NLL via the canonical
/// [`individual_nll_into`]. This is the behaviour the GPU path must match.
pub fn batched_individual_nll_cpu(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    etas: &[Vec<f64>],
    omega: &OmegaMatrix,
    sigma: &[f64],
) -> Vec<f64> {
    use rayon::prelude::*;
    population
        .subjects
        .par_iter()
        .zip(etas.par_iter())
        .map_init(EventPkParams::default, |scratch, (subject, eta)| {
            individual_nll_into(model, subject, theta, eta, omega, sigma, scratch)
        })
        .collect()
}

/// Combine the GPU per-subject data term with the CPU-computed η-prior to form
/// the full per-subject NLL, matching `individual_nll_into`.
fn combine_with_prior(data_ll: &[f32], etas: &[Vec<f64>], omega: &OmegaMatrix) -> Vec<f64> {
    let log_det = omega.log_det;
    if !log_det.is_finite() {
        return vec![1e20; data_ll.len()];
    }
    let omega_inv = &omega.inv;
    etas.iter()
        .zip(data_ll.iter())
        .map(|(eta, &dll)| {
            let ev = DVector::from_column_slice(eta);
            let prior = ev.dot(&(omega_inv * &ev));
            let nll = 0.5 * (prior + log_det + dll as f64);
            if nll.is_finite() {
                nll
            } else {
                1e20
            }
        })
        .collect()
}

/// Outcome of a backend dispatch.
pub struct Dispatch {
    /// Per-subject individual NLL (same ordering as `population.subjects`).
    pub nll: Vec<f64>,
    /// Whether the GPU path actually ran.
    pub used_gpu: bool,
    /// Set when the GPU was requested (`SaemBackend::Gpu`) but the engine fell
    /// back to CPU. Caller should surface this in `FitResult.warnings`.
    pub warning: Option<String>,
}

/// Evaluate the batched per-subject NLL on the requested backend, falling back
/// to CPU whenever the GPU path is unavailable or the model is unsupported.
pub fn batched_individual_nll(
    backend: SaemBackend,
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    etas: &[Vec<f64>],
    omega: &OmegaMatrix,
    sigma: &[f64],
) -> Dispatch {
    let want_gpu = matches!(backend, SaemBackend::Auto | SaemBackend::Gpu);
    if want_gpu {
        if let Some(batch) = flatten(model, population, theta, etas, sigma) {
            if let Some(data_ll) = gpu_data_ll(&batch) {
                return Dispatch {
                    nll: combine_with_prior(&data_ll, etas, omega),
                    used_gpu: true,
                    warning: None,
                };
            }
        }
    }
    // CPU fallback. Warn only when the GPU was explicitly requested.
    let warning = if backend == SaemBackend::Gpu {
        Some(gpu_unavailable_reason(model))
    } else {
        None
    };
    Dispatch {
        nll: batched_individual_nll_cpu(model, population, theta, etas, omega, sigma),
        used_gpu: false,
        warning,
    }
}

fn gpu_unavailable_reason(model: &CompiledModel) -> String {
    if cfg!(not(feature = "gpu")) {
        "saem_backend=gpu requested but ferx-core was built without the `gpu` \
         feature; using the CPU SAEM E-step."
            .to_string()
    } else if !gpu_model_supported(model) {
        "saem_backend=gpu requested but this model is outside the GPU-supported \
         subset (1-cpt IV bolus, single Gaussian endpoint, no SDE/M3); using \
         the CPU SAEM E-step."
            .to_string()
    } else {
        "saem_backend=gpu requested but no GPU device was available or the \
         population is outside the supported subset (infusions, steady-state \
         doses, resets, or time-varying covariates); using the CPU SAEM E-step."
            .to_string()
    }
}

// ---------------------------------------------------------------------------
// GPU kernel — compiled only with the `gpu` feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod kernel {
    use super::FlatBatch;
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;
    use std::sync::OnceLock;

    type R = WgpuRuntime;
    type GpuClient = ComputeClient<R>;

    /// Lazily-initialised, process-wide GPU client. `None` when no device is
    /// available (client creation panics on a headless box with no adapter —
    /// caught here so the caller falls back to CPU).
    fn client() -> Option<&'static GpuClient> {
        static CLIENT: OnceLock<Option<GpuClient>> = OnceLock::new();
        CLIENT
            .get_or_init(|| {
                std::panic::catch_unwind(|| {
                    let device = Default::default();
                    R::client(&device)
                })
                .ok()
            })
            .as_ref()
    }

    #[cube(launch_unchecked)]
    fn nll_1cpt_iv_kernel(
        cl: &Array<f32>,
        v: &Array<f32>,
        dose_off: &Array<u32>,
        dose_cnt: &Array<u32>,
        dose_amt: &Array<f32>,
        dose_t: &Array<f32>,
        obs_off: &Array<u32>,
        obs_cnt: &Array<u32>,
        obs_y: &Array<f32>,
        obs_t: &Array<f32>,
        params: &Array<f32>,
        out: &mut Array<f32>,
    ) {
        if ABSOLUTE_POS < cl.len() {
            let i = ABSOLUTE_POS;
            let vi = v[i];
            let k = cl[i] / vi;
            let err = params[0];
            let s0 = params[1];
            let s1 = params[2];
            // Offsets/counts are u32 on the device; cast to usize for indexing
            // and loop bounds (Array indexing uses usize, matching `len()`).
            let d_o = usize::cast_from(dose_off[i]);
            let d_c = usize::cast_from(dose_cnt[i]);
            let o_o = usize::cast_from(obs_off[i]);
            let o_c = usize::cast_from(obs_cnt[i]);

            let mut acc = 0.0f32;
            for j in 0..o_c {
                let t = obs_t[o_o + j];
                let y = obs_y[o_o + j];
                let mut f = 0.0f32;
                for d in 0..d_c {
                    let dt = t - dose_t[d_o + d];
                    if dt >= 0.0 {
                        f += (dose_amt[d_o + d] / vi) * (-k * dt).exp();
                    }
                }
                let mut vr = s0 * s0;
                if err >= 0.5 {
                    let fs = f * s0;
                    if err >= 1.5 {
                        vr = fs * fs + s1 * s1;
                    } else {
                        vr = fs * fs;
                    }
                }
                if vr < 1e-12 {
                    vr = 1e-12;
                }
                let resid = y - f;
                acc += resid * resid / vr + vr.ln();
            }
            out[i] = acc;
        }
    }

    /// Run the kernel; returns the per-subject data term, or `None` if no
    /// device is available.
    pub(super) fn gpu_data_ll(batch: &FlatBatch) -> Option<Vec<f32>> {
        let client = client()?;
        let n = batch.n_subjects;
        if n == 0 {
            return Some(Vec::new());
        }

        let cl = client.create_from_slice(f32::as_bytes(&batch.cl));
        let v = client.create_from_slice(f32::as_bytes(&batch.v));
        let dose_off = client.create_from_slice(u32::as_bytes(&batch.dose_off));
        let dose_cnt = client.create_from_slice(u32::as_bytes(&batch.dose_cnt));
        // Empty buffers are not allowed; pad to length 1 when a subject has no
        // doses/obs (counts still drive the loops, so the pad is never read).
        let dose_amt = client.create_from_slice(u32_pad_f32(&batch.dose_amt));
        let dose_t = client.create_from_slice(u32_pad_f32(&batch.dose_t));
        let obs_off = client.create_from_slice(u32::as_bytes(&batch.obs_off));
        let obs_cnt = client.create_from_slice(u32::as_bytes(&batch.obs_cnt));
        let obs_y = client.create_from_slice(u32_pad_f32(&batch.obs_y));
        let obs_t = client.create_from_slice(u32_pad_f32(&batch.obs_t));
        let params = client.create_from_slice(f32::as_bytes(&batch.params));
        let out = client.empty(n * core::mem::size_of::<f32>());

        let threads = 64u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        unsafe {
            nll_1cpt_iv_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(cl, batch.cl.len()),
                ArrayArg::from_raw_parts(v, batch.v.len()),
                ArrayArg::from_raw_parts(dose_off, batch.dose_off.len()),
                ArrayArg::from_raw_parts(dose_cnt, batch.dose_cnt.len()),
                ArrayArg::from_raw_parts(dose_amt, batch.dose_amt.len().max(1)),
                ArrayArg::from_raw_parts(dose_t, batch.dose_t.len().max(1)),
                ArrayArg::from_raw_parts(obs_off, batch.obs_off.len()),
                ArrayArg::from_raw_parts(obs_cnt, batch.obs_cnt.len()),
                ArrayArg::from_raw_parts(obs_y, batch.obs_y.len().max(1)),
                ArrayArg::from_raw_parts(obs_t, batch.obs_t.len().max(1)),
                ArrayArg::from_raw_parts(params, batch.params.len()),
                ArrayArg::from_raw_parts(out.clone(), n),
            );
        }
        let bytes = client.read_one(out).ok()?;
        Some(f32::from_bytes(&bytes).to_vec())
    }

    /// `f32::as_bytes` for a possibly-empty slice, padded to one element so the
    /// GPU buffer is non-empty.
    fn u32_pad_f32(data: &[f32]) -> &[u8] {
        static ONE: [f32; 1] = [0.0];
        if data.is_empty() {
            f32::as_bytes(&ONE)
        } else {
            f32::as_bytes(data)
        }
    }
}

#[cfg(feature = "gpu")]
use kernel::gpu_data_ll;

#[cfg(not(feature = "gpu"))]
fn gpu_data_ll(_batch: &FlatBatch) -> Option<Vec<f32>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{DoseEvent, Population, Subject};
    use std::collections::HashMap;

    /// A 1-cpt IV-bolus model with a combined error model and two etas
    /// (CL, V) — squarely in the GPU-supported subset.
    fn iv_combined_model() -> CompiledModel {
        parse_model_string(
            r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.1 (sd)
  sigma ADD  ~ 0.5 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP, ADD)
",
        )
        .expect("1-cpt IV combined model parses")
    }

    /// A 1-cpt *oral* model — outside the GPU-supported subset (absorption).
    fn oral_model() -> CompiledModel {
        parse_model_string(
            r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  theta TVKA(1.0, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
",
        )
        .expect("1-cpt oral model parses")
    }

    fn subject(id: &str, dose: f64, times: &[f64], obs: &[f64]) -> Subject {
        Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, dose, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: obs.to_vec(),
            obs_cmts: vec![1; times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; times.len()],
            occasions: vec![],
            dose_occasions: vec![],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    fn population() -> Population {
        Population {
            subjects: vec![
                subject(
                    "1",
                    100.0,
                    &[0.5, 1.0, 2.0, 4.0, 8.0],
                    &[8.0, 6.5, 4.0, 2.0, 0.6],
                ),
                subject("2", 120.0, &[0.5, 1.0, 3.0, 6.0], &[9.5, 7.5, 3.5, 1.2]),
                subject(
                    "3",
                    80.0,
                    &[1.0, 2.0, 4.0, 6.0, 10.0],
                    &[5.5, 4.0, 2.2, 1.1, 0.3],
                ),
            ],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    fn fixture() -> (
        CompiledModel,
        Population,
        Vec<f64>,
        Vec<f64>,
        OmegaMatrix,
        Vec<Vec<f64>>,
    ) {
        let model = iv_combined_model();
        let pop = population();
        let theta = vec![4.0_f64, 40.0];
        let sigma = vec![0.1_f64, 0.5];
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let etas: Vec<Vec<f64>> = vec![vec![0.1, -0.05], vec![-0.2, 0.15], vec![0.0, 0.0]];
        (model, pop, theta, sigma, omega, etas)
    }

    #[test]
    fn supported_subset_detection() {
        assert!(gpu_model_supported(&iv_combined_model()));
        assert!(!gpu_model_supported(&oral_model()));
    }

    #[test]
    fn cpu_dispatch_matches_individual_nll() {
        let (model, pop, theta, sigma, omega, etas) = fixture();
        let reference = batched_individual_nll_cpu(&model, &pop, &theta, &etas, &omega, &sigma);
        let d = batched_individual_nll(
            SaemBackend::Cpu,
            &model,
            &pop,
            &theta,
            &etas,
            &omega,
            &sigma,
        );
        assert!(!d.used_gpu);
        assert!(d.warning.is_none());
        assert_eq!(d.nll.len(), reference.len());
        for (a, b) in d.nll.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-9, "cpu dispatch must equal reference");
        }
    }

    #[test]
    fn unsupported_model_falls_back_with_warning() {
        let model = oral_model();
        let pop = Population {
            subjects: vec![subject("1", 100.0, &[1.0, 2.0, 4.0], &[3.0, 2.0, 1.0])],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let theta = vec![4.0_f64, 40.0, 1.0];
        let sigma = vec![0.1_f64];
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let etas = vec![vec![0.0_f64]];
        let d = batched_individual_nll(
            SaemBackend::Gpu,
            &model,
            &pop,
            &theta,
            &etas,
            &omega,
            &sigma,
        );
        // Oral model is unsupported → CPU fallback, with a warning since GPU
        // was explicitly requested.
        assert!(!d.used_gpu);
        assert!(d.warning.is_some());
        let reference = batched_individual_nll_cpu(&model, &pop, &theta, &etas, &omega, &sigma);
        for (a, b) in d.nll.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    /// GPU/CPU parity. Only meaningful with the `gpu` feature *and* a device;
    /// when no GPU is present the dispatcher falls back to CPU (`used_gpu ==
    /// false`) and the test trivially holds.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_matches_cpu_within_f32_tolerance() {
        let (model, pop, theta, sigma, omega, etas) = fixture();
        let reference = batched_individual_nll_cpu(&model, &pop, &theta, &etas, &omega, &sigma);
        let d = batched_individual_nll(
            SaemBackend::Gpu,
            &model,
            &pop,
            &theta,
            &etas,
            &omega,
            &sigma,
        );
        if !d.used_gpu {
            eprintln!("no GPU device available; parity check skipped");
            return;
        }
        assert_eq!(d.nll.len(), reference.len());
        for (i, (g, c)) in d.nll.iter().zip(reference.iter()).enumerate() {
            // f32 kernel vs f64 reference: relative tolerance.
            let rel = (g - c).abs() / c.abs().max(1.0);
            assert!(
                rel < 1e-3,
                "subject {i}: gpu={g} cpu={c} rel={rel} exceeds f32 tolerance"
            );
        }
    }
}
