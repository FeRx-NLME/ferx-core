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
    use super::{FlatBatch, MhBatch, GPU_MH_MAX_ETA};
    use cubecl::prelude::*;
    use cubecl::server::Handle;
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

    // -- MH-sweep kernel ----------------------------------------------------

    /// Integer finalizer hash (lowbias32) for the counter-based RNG.
    #[cube]
    fn hash32(x: u32) -> u32 {
        let mut h = x;
        h = h ^ (h >> 16u32);
        h = h * 2146028333u32; // 0x7feb352d
        h = h ^ (h >> 15u32);
        h = h * 2221713035u32; // 0x846ca68b
        h = h ^ (h >> 16u32);
        h
    }

    /// Uniform in [0, 1) from a (seed, subject, step, slot) counter.
    #[cube]
    fn rand_u(seed: u32, subj: u32, step: u32, slot: u32) -> f32 {
        let ctr = seed ^ hash32(subj * 2654435761u32 + (step * 64u32 + slot) * 40503u32 + 1u32);
        let h = hash32(ctr);
        f32::cast_from(h >> 8u32) * 5.9604645e-8 // 1 / 2^24
    }

    /// Standard normal via Box-Muller from two uniforms.
    #[cube]
    fn rand_n(seed: u32, subj: u32, step: u32, slot: u32) -> f32 {
        let u1 = rand_u(seed, subj, step, slot * 2u32);
        let u2 = rand_u(seed, subj, step, slot * 2u32 + 1u32);
        // Keep `u1c` cube-typed (init from `u1`, then clamp via assignment) so
        // `.ln()` resolves — a literal `if`-branch would make it ambiguous.
        let mut u1c = u1;
        if u1c < 1e-7 {
            u1c = 1e-7;
        }
        let r = (u1c.ln() * -2.0).sqrt();
        let angle = u2 * 6.2831853;
        r * angle.cos()
    }

    /// Individual NLL `0.5·(η'Ω⁻¹η + ln|Ω| + Σ[resid²/v + ln v])` for the
    /// 1-cpt IV-bolus model with diagonal Ω and log-linear `(CL, V)` in η.
    #[cube]
    #[allow(clippy::too_many_arguments)]
    fn nll_at(
        eta: &Array<f32>,
        n_eta: u32,
        cl0: f32,
        v0: f32,
        a_cl: &Array<f32>,
        a_v: &Array<f32>,
        omega_diag: &Array<f32>,
        dose_amt: &Array<f32>,
        dose_t: &Array<f32>,
        d_o: u32,
        d_c: u32,
        obs_y: &Array<f32>,
        obs_t: &Array<f32>,
        o_o: u32,
        o_c: u32,
        err: f32,
        s0: f32,
        s1: f32,
    ) -> f32 {
        let ne = usize::cast_from(n_eta);
        let mut lin_cl = 0.0f32;
        let mut lin_v = 0.0f32;
        let mut prior = 0.0f32;
        let mut logdet = 0.0f32;
        for c in 0..ne {
            let ev = eta[c];
            lin_cl += a_cl[c] * ev;
            lin_v += a_v[c] * ev;
            let od = omega_diag[c];
            prior += ev * ev / od;
            logdet += od.ln();
        }
        let cl = cl0 * lin_cl.exp();
        let vv = v0 * lin_v.exp();
        let k = cl / vv;

        let do_ = usize::cast_from(d_o);
        let dc = usize::cast_from(d_c);
        let oo = usize::cast_from(o_o);
        let oc = usize::cast_from(o_c);
        let mut data = 0.0f32;
        for j in 0..oc {
            let t = obs_t[oo + j];
            let y = obs_y[oo + j];
            let mut f = 0.0f32;
            for d in 0..dc {
                let dt = t - dose_t[do_ + d];
                if dt >= 0.0 {
                    f += (dose_amt[do_ + d] / vv) * (-k * dt).exp();
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
            data += resid * resid / vr + vr.ln();
        }
        0.5 * (prior + logdet + data)
    }

    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    fn mh_sweep_kernel(
        eta_io: &mut Array<f32>,
        cl0: &Array<f32>,
        v0: &Array<f32>,
        a_cl: &Array<f32>,
        a_v: &Array<f32>,
        omega_diag: &Array<f32>,
        step_scale: &Array<f32>,
        dose_off: &Array<u32>,
        dose_cnt: &Array<u32>,
        dose_amt: &Array<f32>,
        dose_t: &Array<f32>,
        obs_off: &Array<u32>,
        obs_cnt: &Array<u32>,
        obs_y: &Array<f32>,
        obs_t: &Array<f32>,
        fparams: &Array<f32>,
        iparams: &Array<u32>,
        accept_out: &mut Array<u32>,
        nll_out: &mut Array<f32>,
    ) {
        if ABSOLUTE_POS < cl0.len() {
            let i = ABSOLUTE_POS;
            let n_eta = iparams[0];
            let n_steps = usize::cast_from(iparams[1]);
            let seed = iparams[2];
            let ne = usize::cast_from(n_eta);
            let err = fparams[0];
            let s0 = fparams[1];
            let s1 = fparams[2];
            let ss = step_scale[i];
            let base = i * ne;
            let d_o = dose_off[i];
            let d_c = dose_cnt[i];
            let o_o = obs_off[i];
            let o_c = obs_cnt[i];
            let subj = u32::cast_from(i);

            let mut eta_loc = Array::<f32>::new(GPU_MH_MAX_ETA);
            let mut prop = Array::<f32>::new(GPU_MH_MAX_ETA);
            for c in 0..ne {
                eta_loc[c] = eta_io[base + c];
            }
            let mut nll_cur = nll_at(
                &eta_loc, n_eta, cl0[i], v0[i], a_cl, a_v, omega_diag, dose_amt, dose_t, d_o, d_c,
                obs_y, obs_t, o_o, o_c, err, s0, s1,
            );

            let mut accept = 0u32;
            for step in 0..n_steps {
                let st = u32::cast_from(step);
                for c in 0..ne {
                    let z = rand_n(seed, subj, st, u32::cast_from(c));
                    prop[c] = eta_loc[c] + ss * omega_diag[c].sqrt() * z;
                }
                let nll_prop = nll_at(
                    &prop, n_eta, cl0[i], v0[i], a_cl, a_v, omega_diag, dose_amt, dose_t, d_o, d_c,
                    obs_y, obs_t, o_o, o_c, err, s0, s1,
                );
                let u = rand_u(seed, subj, st, 63u32);
                let mut u_c = u;
                if u_c < 1e-30 {
                    u_c = 1e-30;
                }
                if u_c.ln() < nll_cur - nll_prop {
                    for c in 0..ne {
                        eta_loc[c] = prop[c];
                    }
                    nll_cur = nll_prop;
                    accept += 1u32;
                }
            }
            for c in 0..ne {
                eta_io[base + c] = eta_loc[c];
            }
            accept_out[i] = accept;
            nll_out[i] = nll_cur;
        }
    }

    /// A persistent MH-sweep session: the per-subject **static** buffers
    /// (dose/observation arrays) are uploaded once and live on the device for
    /// the whole SAEM run; only the per-iteration **dynamic** data (etas,
    /// intercepts, Ω diagonal, step scales, sigmas, seed) is uploaded each
    /// sweep. This removes the per-iteration re-upload of the largest arrays.
    pub(super) struct Session {
        n: usize,
        dose_off: Handle,
        dose_cnt: Handle,
        dose_amt: Handle,
        dose_t: Handle,
        obs_off: Handle,
        obs_cnt: Handle,
        obs_y: Handle,
        obs_t: Handle,
        l_dose_off: usize,
        l_dose_cnt: usize,
        l_dose_amt: usize,
        l_dose_t: usize,
        l_obs_off: usize,
        l_obs_cnt: usize,
        l_obs_y: usize,
        l_obs_t: usize,
    }

    impl Session {
        /// Upload the static (dose/obs) buffers once. `None` if no device.
        pub(super) fn new(batch: &MhBatch) -> Option<Session> {
            let client = client()?;
            Some(Session {
                n: batch.n_subjects,
                dose_off: client.create_from_slice(u32::as_bytes(&batch.dose_off)),
                dose_cnt: client.create_from_slice(u32::as_bytes(&batch.dose_cnt)),
                dose_amt: client.create_from_slice(u32_pad_f32(&batch.dose_amt)),
                dose_t: client.create_from_slice(u32_pad_f32(&batch.dose_t)),
                obs_off: client.create_from_slice(u32::as_bytes(&batch.obs_off)),
                obs_cnt: client.create_from_slice(u32::as_bytes(&batch.obs_cnt)),
                obs_y: client.create_from_slice(u32_pad_f32(&batch.obs_y)),
                obs_t: client.create_from_slice(u32_pad_f32(&batch.obs_t)),
                l_dose_off: batch.dose_off.len(),
                l_dose_cnt: batch.dose_cnt.len(),
                l_dose_amt: batch.dose_amt.len().max(1),
                l_dose_t: batch.dose_t.len().max(1),
                l_obs_off: batch.obs_off.len(),
                l_obs_cnt: batch.obs_cnt.len(),
                l_obs_y: batch.obs_y.len().max(1),
                l_obs_t: batch.obs_t.len().max(1),
            })
        }

        /// Run one MH sweep, reusing the resident static buffers and uploading
        /// only the dynamic arrays. Returns `(etas flat, accepts, final nll)`.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn sweep(
            &self,
            eta: &[f32],
            cl0: &[f32],
            v0: &[f32],
            a_cl: &[f32],
            a_v: &[f32],
            omega_diag: &[f32],
            scales: &[f32],
            fparams: &[f32],
            n_eta: usize,
            n_steps: usize,
            seed: u32,
        ) -> Option<(Vec<f32>, Vec<u32>, Vec<f32>)> {
            let client = client()?;
            let n = self.n;
            if n == 0 {
                return Some((Vec::new(), Vec::new(), Vec::new()));
            }
            let eta_io = client.create_from_slice(f32::as_bytes(eta));
            let cl0_h = client.create_from_slice(f32::as_bytes(cl0));
            let v0_h = client.create_from_slice(f32::as_bytes(v0));
            let a_cl_h = client.create_from_slice(f32::as_bytes(a_cl));
            let a_v_h = client.create_from_slice(f32::as_bytes(a_v));
            let omega_h = client.create_from_slice(f32::as_bytes(omega_diag));
            let scale_h = client.create_from_slice(f32::as_bytes(scales));
            let fparams_h = client.create_from_slice(f32::as_bytes(fparams));
            let iparams_vec = [n_eta as u32, n_steps as u32, seed];
            let iparams = client.create_from_slice(u32::as_bytes(&iparams_vec));
            let accept = client.empty(n * core::mem::size_of::<u32>());
            let nll = client.empty(n * core::mem::size_of::<f32>());

            let threads = 64u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            unsafe {
                mh_sweep_kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(blocks, 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(eta_io.clone(), eta.len()),
                    ArrayArg::from_raw_parts(cl0_h, cl0.len()),
                    ArrayArg::from_raw_parts(v0_h, v0.len()),
                    ArrayArg::from_raw_parts(a_cl_h, a_cl.len()),
                    ArrayArg::from_raw_parts(a_v_h, a_v.len()),
                    ArrayArg::from_raw_parts(omega_h, omega_diag.len()),
                    ArrayArg::from_raw_parts(scale_h, scales.len()),
                    ArrayArg::from_raw_parts(self.dose_off.clone(), self.l_dose_off),
                    ArrayArg::from_raw_parts(self.dose_cnt.clone(), self.l_dose_cnt),
                    ArrayArg::from_raw_parts(self.dose_amt.clone(), self.l_dose_amt),
                    ArrayArg::from_raw_parts(self.dose_t.clone(), self.l_dose_t),
                    ArrayArg::from_raw_parts(self.obs_off.clone(), self.l_obs_off),
                    ArrayArg::from_raw_parts(self.obs_cnt.clone(), self.l_obs_cnt),
                    ArrayArg::from_raw_parts(self.obs_y.clone(), self.l_obs_y),
                    ArrayArg::from_raw_parts(self.obs_t.clone(), self.l_obs_t),
                    ArrayArg::from_raw_parts(fparams_h, fparams.len()),
                    ArrayArg::from_raw_parts(iparams, iparams_vec.len()),
                    ArrayArg::from_raw_parts(accept.clone(), n),
                    ArrayArg::from_raw_parts(nll.clone(), n),
                );
            }
            // Single batched read = one device sync per sweep, not three.
            let mut out = client.read(vec![eta_io, accept, nll]);
            let nll_bytes = out.pop()?;
            let accept_bytes = out.pop()?;
            let eta_bytes = out.pop()?;
            Some((
                f32::from_bytes(&eta_bytes).to_vec(),
                u32::from_bytes(&accept_bytes).to_vec(),
                f32::from_bytes(&nll_bytes).to_vec(),
            ))
        }
    }
}

#[cfg(feature = "gpu")]
use kernel::gpu_data_ll;

#[cfg(not(feature = "gpu"))]
fn gpu_data_ll(_batch: &FlatBatch) -> Option<Vec<f32>> {
    None
}

// ---------------------------------------------------------------------------
// GPU Metropolis-Hastings E-step sweep (the proposal-loop port, issue #368).
//
// One GPU thread per subject runs the full block random-walk MH chain in
// kernel: propose η' = η + step·L·z (diagonal Ω ⇒ L = diag(√Ω_jj)), evaluate
// the individual NLL, accept on ln u < nll − nll'. Because the proposed η
// changes CL/V every step, and the θ,η→PK-param transform is an opaque DSL
// closure, the kernel reconstructs CL/V from a *log-linear* model
// `CL = CL0·exp(a_cl·η)` whose coefficients are extracted and verified on the
// host (see `extract_loglinear`). Models that are not log-linear in η fall
// back to CPU.
//
// The in-kernel RNG is a counter-based hash, so the GPU chain does not match
// the CPU chain bit-for-bit; SAEM is stochastic and the M-step averages over
// draws, so the two converge to statistically equivalent estimates (verified
// by a full-fit convergence test). The host keeps the M-step, step-size
// adaptation, and (for unsupported cases) the whole E-step.
// ---------------------------------------------------------------------------

/// Maximum number of random effects the GPU MH kernel supports (sizes the
/// per-thread η scratch, which must be a compile-time length).
pub const GPU_MH_MAX_ETA: usize = 8;

/// Log-linear reconstruction of `(CL, V)` in η: `CL_i = cl0[i]·exp(a_cl·η)`.
/// Coefficients are population-global; intercepts are per-subject (covariates).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
struct LogLinear {
    cl0: Vec<f32>,
    v0: Vec<f32>,
    a_cl: Vec<f32>,
    a_v: Vec<f32>,
}

fn rel_err(pred: f64, actual: f64) -> f64 {
    (pred - actual).abs() / actual.abs().max(1e-12)
}

/// Probe `model.pk_param_fn` to extract a log-linear `(CL, V)`-in-η model and
/// verify it reproduces the closure (at η = 0.3·1 and at each subject's current
/// η). Returns `None` when the model is not log-linear in η within tolerance —
/// the caller then falls back to the CPU E-step.
fn extract_loglinear(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    n_eta: usize,
    etas: &[Vec<f64>],
    verify: bool,
) -> Option<LogLinear> {
    let zero = vec![0.0_f64; n_eta];
    let cov0 = &population.subjects[0].covariates;
    let base = (model.pk_param_fn)(theta, &zero, cov0);
    let cl0_ref = base.values[PK_IDX_CL];
    let v0_ref = base.values[PK_IDX_V];
    if !(cl0_ref > 0.0 && v0_ref > 0.0) {
        return None;
    }
    // Finite-difference the log-sensitivities a_cl[j] = d ln CL / d eta_j.
    let h = 1e-3;
    let mut a_cl = vec![0.0_f64; n_eta];
    let mut a_v = vec![0.0_f64; n_eta];
    for j in 0..n_eta {
        let mut e = zero.clone();
        e[j] = h;
        let p = (model.pk_param_fn)(theta, &e, cov0);
        if !(p.values[PK_IDX_CL] > 0.0 && p.values[PK_IDX_V] > 0.0) {
            return None;
        }
        a_cl[j] = (p.values[PK_IDX_CL] / cl0_ref).ln() / h;
        a_v[j] = (p.values[PK_IDX_V] / v0_ref).ln() / h;
    }

    let dot = |a: &[f64], e: &[f64]| a.iter().zip(e).map(|(x, y)| x * y).sum::<f64>();

    let mut cl0 = Vec::with_capacity(population.subjects.len());
    let mut v0 = Vec::with_capacity(population.subjects.len());
    for (i, subject) in population.subjects.iter().enumerate() {
        let cov = &subject.covariates;
        let b = (model.pk_param_fn)(theta, &zero, cov);
        let cl0_i = b.values[PK_IDX_CL];
        let v0_i = b.values[PK_IDX_V];
        if !(cl0_i > 0.0 && v0_i > 0.0) {
            return None;
        }
        // Verify log-linearity for this subject at a non-trivial η and at its
        // current η: the reconstruction must match the real closure. This
        // catches non-log-normal transforms and covariate×η interactions.
        // Done once at session setup (`verify = true`); per-iteration sweeps
        // reuse the verified structure and only recompute the intercepts.
        if verify {
            for probe in [vec![0.3_f64; n_eta], etas[i].clone()] {
                let actual = (model.pk_param_fn)(theta, &probe, cov);
                let pred_cl = cl0_i * dot(&a_cl, &probe).exp();
                let pred_v = v0_i * dot(&a_v, &probe).exp();
                if rel_err(pred_cl, actual.values[PK_IDX_CL]) > 1e-4
                    || rel_err(pred_v, actual.values[PK_IDX_V]) > 1e-4
                {
                    return None;
                }
            }
        }
        cl0.push(cl0_i as f32);
        v0.push(v0_i as f32);
    }

    Some(LogLinear {
        cl0,
        v0,
        a_cl: a_cl.iter().map(|&x| x as f32).collect(),
        a_v: a_v.iter().map(|&x| x as f32).collect(),
    })
}

/// Static (per-run) layout for the MH-sweep kernel: the dose/observation
/// arrays that do not change across SAEM iterations. The dynamic per-iteration
/// data (etas, intercepts, Ω diagonal, sigmas) is supplied to each sweep.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
struct MhBatch {
    n_subjects: usize,
    n_eta: usize,
    dose_off: Vec<u32>,
    dose_cnt: Vec<u32>,
    dose_amt: Vec<f32>,
    dose_t: Vec<f32>,
    obs_off: Vec<u32>,
    obs_cnt: Vec<u32>,
    obs_y: Vec<f32>,
    obs_t: Vec<f32>,
}

/// Build the static MH batch (and run the gating + log-linear verification), or
/// `None` when the model/population/Ω are outside the GPU MH-supported subset
/// (diagonal Ω, ≤ `GPU_MH_MAX_ETA` etas, log-linear in η, plus the same
/// structural gates as [`flatten`]).
fn build_mh_batch(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    etas: &[Vec<f64>],
    _sigma: &[f64],
    omega: &OmegaMatrix,
) -> Option<MhBatch> {
    if !gpu_model_supported(model) || !omega.diagonal || population.subjects.is_empty() {
        return None;
    }
    let n_eta = omega.matrix.nrows();
    if n_eta == 0 || n_eta > GPU_MH_MAX_ETA || etas.iter().any(|e| e.len() != n_eta) {
        return None;
    }
    for j in 0..n_eta {
        if !(omega.matrix[(j, j)] > 0.0) {
            return None;
        }
    }
    if !matches!(model.error_spec, ErrorSpec::Single(_)) {
        return None;
    }

    // Gating: verify the log-linear (CL, V)-in-η reconstruction once here.
    extract_loglinear(model, population, theta, n_eta, etas, true)?;

    let n = population.subjects.len();
    let mut dose_off = Vec::with_capacity(n);
    let mut dose_cnt = Vec::with_capacity(n);
    let mut dose_amt = Vec::new();
    let mut dose_t = Vec::new();
    let mut obs_off = Vec::with_capacity(n);
    let mut obs_cnt = Vec::with_capacity(n);
    let mut obs_y = Vec::new();
    let mut obs_t = Vec::new();

    for subject in population.subjects.iter() {
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

    Some(MhBatch {
        n_subjects: n,
        n_eta,
        dose_off,
        dose_cnt,
        dose_amt,
        dose_t,
        obs_off,
        obs_cnt,
        obs_y,
        obs_t,
    })
}

/// Result of one GPU MH sweep.
pub struct MhSweepOut {
    /// Updated per-subject etas.
    pub etas: Vec<Vec<f64>>,
    /// Acceptances per subject across the sweep.
    pub accepts: Vec<usize>,
    /// Final per-subject individual NLL (for the SAEM nll cache).
    pub nll: Vec<f64>,
}

/// A persistent GPU MH-sweep session for one SAEM run. Create it once (it
/// gates the model/population and uploads the static dose/observation buffers
/// to the device); then call [`GpuMhSession::sweep`] each iteration, which
/// uploads only the dynamic per-iteration data. `None` from `new` means the
/// model is unsupported or no GPU is available — the caller uses the CPU E-step.
#[cfg(feature = "gpu")]
pub struct GpuMhSession {
    inner: kernel::Session,
    n_eta: usize,
    n_subjects: usize,
    em: ErrorModel,
}

/// Stub on non-`gpu` builds: never constructed.
#[cfg(not(feature = "gpu"))]
pub struct GpuMhSession {
    _priv: (),
}

impl GpuMhSession {
    /// Gate the model/population and upload the static buffers. `None` when the
    /// GPU MH path is unavailable or the model is unsupported.
    #[cfg(feature = "gpu")]
    pub fn new(
        model: &CompiledModel,
        population: &Population,
        theta: &[f64],
        etas: &[Vec<f64>],
        sigma: &[f64],
        omega: &OmegaMatrix,
    ) -> Option<Self> {
        let batch = build_mh_batch(model, population, theta, etas, sigma, omega)?;
        let em = match &model.error_spec {
            ErrorSpec::Single(e) => *e,
            ErrorSpec::PerCmt(_) => return None,
        };
        let inner = kernel::Session::new(&batch)?;
        Some(Self {
            inner,
            n_eta: batch.n_eta,
            n_subjects: batch.n_subjects,
            em,
        })
    }

    #[cfg(not(feature = "gpu"))]
    pub fn new(
        _model: &CompiledModel,
        _population: &Population,
        _theta: &[f64],
        _etas: &[Vec<f64>],
        _sigma: &[f64],
        _omega: &OmegaMatrix,
    ) -> Option<Self> {
        None
    }

    /// Run one block-RW MH sweep of `n_steps` proposals per subject. Recomputes
    /// the (cheap) per-subject intercepts for the current `theta`, uploads only
    /// dynamic data, and reuses the resident static buffers. `seed` should vary
    /// per SAEM iteration.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    pub fn sweep(
        &mut self,
        model: &CompiledModel,
        population: &Population,
        theta: &[f64],
        etas: &[Vec<f64>],
        sigma: &[f64],
        omega: &OmegaMatrix,
        step_scales: &[f64],
        n_steps: usize,
        seed: u64,
    ) -> Option<MhSweepOut> {
        if etas.len() != self.n_subjects {
            return None;
        }
        let n_eta = self.n_eta;
        // Reconstruct intercepts/coefficients for the current theta (no verify —
        // the structure was verified at session creation). Cheap: O(N + n_eta)
        // closure evals, no per-iteration re-upload of the static buffers.
        let ll = extract_loglinear(model, population, theta, n_eta, etas, false)?;
        let mut omega_diag = Vec::with_capacity(n_eta);
        for j in 0..n_eta {
            let d = omega.matrix[(j, j)];
            if !(d > 0.0) {
                return None;
            }
            omega_diag.push(d as f32);
        }
        let eta: Vec<f32> = etas
            .iter()
            .flat_map(|e| e.iter().map(|&x| x as f32))
            .collect();
        let scales: Vec<f32> = step_scales.iter().map(|&s| s as f32).collect();
        let s0 = *sigma.first().unwrap_or(&0.0) as f32;
        let s1 = *sigma.get(1).unwrap_or(&0.0) as f32;
        let fparams = [error_code(self.em), s0, s1];

        let (eta_out, accepts, nll) = self.inner.sweep(
            &eta,
            &ll.cl0,
            &ll.v0,
            &ll.a_cl,
            &ll.a_v,
            &omega_diag,
            &scales,
            &fparams,
            n_eta,
            n_steps,
            seed as u32,
        )?;
        Some(MhSweepOut {
            etas: eta_out
                .chunks(n_eta)
                .map(|c| c.iter().map(|&x| x as f64).collect())
                .collect(),
            accepts: accepts.iter().map(|&a| a as usize).collect(),
            nll: nll.iter().map(|&x| x as f64).collect(),
        })
    }

    #[cfg(not(feature = "gpu"))]
    #[allow(clippy::too_many_arguments)]
    pub fn sweep(
        &mut self,
        _model: &CompiledModel,
        _population: &Population,
        _theta: &[f64],
        _etas: &[Vec<f64>],
        _sigma: &[f64],
        _omega: &OmegaMatrix,
        _step_scales: &[f64],
        _n_steps: usize,
        _seed: u64,
    ) -> Option<MhSweepOut> {
        None
    }
}

/// One-shot convenience wrapper (used by tests): create a session and run a
/// single sweep. Production SAEM uses a persistent [`GpuMhSession`].
pub fn gpu_mh_sweep(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    etas: &[Vec<f64>],
    sigma: &[f64],
    omega: &OmegaMatrix,
    step_scales: &[f64],
    n_steps: usize,
    seed: u64,
) -> Option<MhSweepOut> {
    let mut session = GpuMhSession::new(model, population, theta, etas, sigma, omega)?;
    session.sweep(
        model,
        population,
        theta,
        etas,
        sigma,
        omega,
        step_scales,
        n_steps,
        seed,
    )
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

    #[test]
    fn loglinear_extraction_supported_and_rejected() {
        let (model, pop, theta, _sigma, _omega, etas) = fixture();
        // 1-cpt IV with CL=TVCL*exp(ETA_CL), V=TVV*exp(ETA_V) is log-linear.
        let ll = extract_loglinear(&model, &pop, &theta, 2, &etas, true);
        assert!(
            ll.is_some(),
            "log-normal CL/V must be detected as log-linear"
        );
        let ll = ll.unwrap();
        // a_cl ≈ [1, 0], a_v ≈ [0, 1] for this parameterization.
        assert!((ll.a_cl[0] - 1.0).abs() < 1e-2 && ll.a_cl[1].abs() < 1e-2);
        assert!(ll.a_v[0].abs() < 1e-2 && (ll.a_v[1] - 1.0).abs() < 1e-2);
    }

    /// With `n_steps = 0` the GPU MH sweep performs no proposals and must return
    /// the initial per-subject NLL — a direct check that the in-kernel NLL
    /// (log-linear CL/V reconstruction + data term) matches the CPU reference.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_mh_zero_steps_matches_cpu_nll() {
        let (model, pop, theta, sigma, omega, etas) = fixture();
        let reference = batched_individual_nll_cpu(&model, &pop, &theta, &etas, &omega, &sigma);
        let scales = vec![0.3; pop.subjects.len()];
        let out = gpu_mh_sweep(&model, &pop, &theta, &etas, &sigma, &omega, &scales, 0, 1);
        let Some(out) = out else {
            eprintln!("no GPU device; skipped");
            return;
        };
        for (i, (g, c)) in out.nll.iter().zip(reference.iter()).enumerate() {
            let rel = (g - c).abs() / c.abs().max(1.0);
            assert!(rel < 1e-3, "subject {i}: gpu_nll={g} cpu_nll={c} rel={rel}");
            // No proposals → etas unchanged and zero acceptances.
            assert_eq!(out.accepts[i], 0);
        }
        for (eo, ei) in out.etas.iter().zip(etas.iter()) {
            for (a, b) in eo.iter().zip(ei.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    /// A run of MH proposals must move the chain and accept at a sane rate.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_mh_sweep_accepts_and_moves() {
        let (model, pop, theta, sigma, omega, etas) = fixture();
        let scales = vec![0.3; pop.subjects.len()];
        let out = gpu_mh_sweep(&model, &pop, &theta, &etas, &sigma, &omega, &scales, 200, 7);
        let Some(out) = out else {
            eprintln!("no GPU device; skipped");
            return;
        };
        let total: usize = out.accepts.iter().sum();
        let proposals = 200 * pop.subjects.len();
        let rate = total as f64 / proposals as f64;
        assert!(
            rate > 0.05 && rate < 0.95,
            "implausible acceptance rate {rate}"
        );
        // At least one subject's eta moved.
        let moved = out
            .etas
            .iter()
            .zip(etas.iter())
            .any(|(eo, ei)| eo.iter().zip(ei).any(|(a, b)| (a - b).abs() > 1e-4));
        assert!(moved, "MH chain did not move any eta");
    }

    /// A deterministic synthetic 1-cpt IV dataset: per-subject CL/V vary
    /// log-normally; concentrations follow the analytical bolus decay.
    fn synthetic_population(n: usize) -> Population {
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        let dose = 100.0;
        let subjects = (0..n)
            .map(|i| {
                let fi = i as f64;
                let cl = 4.0 * (0.30 * (0.7 * fi).sin()).exp();
                let v = 40.0 * (0.20 * (0.5 * fi + 1.0).cos()).exp();
                let k = cl / v;
                let obs: Vec<f64> = times
                    .iter()
                    .enumerate()
                    .map(|(j, &t)| {
                        let c = (dose / v) * (-k * t).exp();
                        // small deterministic multiplicative perturbation
                        c * (1.0 + 0.05 * ((fi + 1.3 * j as f64).sin()))
                    })
                    .collect();
                subject(&format!("{i}"), dose, &times, &obs)
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// Full SAEM fit, CPU vs GPU backend, on the same data. The two use
    /// independent RNGs (CPU = ChaCha, GPU = counter-hash) so the chains differ;
    /// Robbins-Monro averaging should still land both near the same estimates.
    #[cfg(feature = "gpu")]
    #[test]
    fn cpu_gpu_saem_converge_similarly() {
        use crate::estimation::saem::run_saem;
        use crate::types::FitOptions;

        let model = iv_combined_model();
        let data = synthetic_population(16);

        let mut opts = FitOptions::default();
        opts.verbose = false;
        opts.run_covariance_step = false;
        opts.saem_n_exploration = 80;
        opts.saem_n_convergence = 200;
        opts.saem_seed = Some(2024);

        opts.saem_backend = SaemBackend::Cpu;
        let cpu = run_saem(&model, &data, &model.default_params, &opts).expect("cpu saem");

        opts.saem_backend = SaemBackend::Gpu;
        let gpu = run_saem(&model, &data, &model.default_params, &opts).expect("gpu saem");

        // If no device was available the GPU E-step fell back to CPU and warned;
        // the comparison is then vacuous, so skip.
        if gpu
            .warnings
            .iter()
            .any(|w| w.contains("GPU MH E-step did not run"))
        {
            eprintln!("no GPU device; convergence comparison skipped");
            return;
        }

        for (idx, (c, g)) in cpu
            .params
            .theta
            .iter()
            .zip(gpu.params.theta.iter())
            .enumerate()
        {
            let rel = (c - g).abs() / c.abs().max(1e-6);
            assert!(
                rel < 0.10,
                "theta[{idx}] cpu={c} gpu={g} rel={rel} — backends disagree"
            );
        }
        for j in 0..cpu.params.omega.matrix.nrows() {
            let c = cpu.params.omega.matrix[(j, j)];
            let g = gpu.params.omega.matrix[(j, j)];
            let rel = (c - g).abs() / c.abs().max(1e-6);
            assert!(rel < 0.40, "omega[{j}] cpu={c} gpu={g} rel={rel}");
        }
    }

    /// Wall-clock CPU vs GPU SAEM across subject counts. Not an assertion —
    /// run manually:
    /// `cargo test --release --no-default-features --features ci,gpu
    ///  gpu_saem_benchmark -- --ignored --nocapture`
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "benchmark: run manually with --ignored --nocapture"]
    fn gpu_saem_benchmark() {
        use crate::estimation::saem::run_saem;
        use crate::types::FitOptions;
        use std::time::Instant;

        let model = iv_combined_model();
        let mut opts = FitOptions::default();
        opts.verbose = false;
        opts.run_covariance_step = false;
        opts.saem_n_exploration = 50;
        opts.saem_n_convergence = 100;
        opts.saem_seed = Some(1);

        let bench = |opts: &mut FitOptions, data: &Population| -> (f64, f64, bool) {
            opts.saem_backend = SaemBackend::Cpu;
            let t = Instant::now();
            let _ = run_saem(&model, data, &model.default_params, opts).expect("cpu");
            let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
            opts.saem_backend = SaemBackend::Gpu;
            let t = Instant::now();
            let g = run_saem(&model, data, &model.default_params, opts).expect("gpu");
            let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
            let used = !g
                .warnings
                .iter()
                .any(|w| w.contains("GPU MH E-step did not run"));
            (cpu_ms, gpu_ms, used)
        };

        println!("\n  (A) vary n_subjects, n_mh_steps = 20");
        println!("  n_subj |   CPU (ms) |   GPU (ms) | speedup | gpu_used");
        println!("  -------+------------+------------+---------+---------");
        opts.saem_n_mh_steps = 20;
        for &n in &[50usize, 200, 1000, 4000, 8000] {
            let data = synthetic_population(n);
            let (cpu_ms, gpu_ms, used) = bench(&mut opts, &data);
            println!(
                "  {n:>6} | {cpu_ms:>10.1} | {gpu_ms:>10.1} | {:>6.2}x | {used}",
                cpu_ms / gpu_ms
            );
        }

        println!("\n  (B) vary n_mh_steps, n_subjects = 2000");
        println!("  mh_stp |   CPU (ms) |   GPU (ms) | speedup | gpu_used");
        println!("  -------+------------+------------+---------+---------");
        let data = synthetic_population(2000);
        for &steps in &[20usize, 100, 500, 2000] {
            opts.saem_n_mh_steps = steps;
            let (cpu_ms, gpu_ms, used) = bench(&mut opts, &data);
            println!(
                "  {steps:>6} | {cpu_ms:>10.1} | {gpu_ms:>10.1} | {:>6.2}x | {used}",
                cpu_ms / gpu_ms
            );
        }
    }
}
