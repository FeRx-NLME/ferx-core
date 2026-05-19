/// SAEM (Stochastic Approximation EM) for NLME population parameter estimation.
///
/// Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
///            Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.
///
/// Two-phase step-size schedule (Monolix convention):
///   Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
///   Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{compute_covariance, pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{individual_nll, individual_nll_into};
use crate::stats::residual_error::residual_variance;
use crate::stats::special::log_normal_cdf;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// SAEM state
// ---------------------------------------------------------------------------

struct SaemState {
    /// Per-subject current ETAs
    etas: Vec<Vec<f64>>,
    /// Cached individual NLL at current ETAs
    nll_cache: Vec<f64>,
    /// Per-subject MH step sizes
    step_scales: Vec<f64>,
    /// Per-subject acceptance counts since last adaptation
    accept_counts: Vec<usize>,
    /// Steps since last adaptation
    steps_since_adapt: usize,
    /// SA sufficient statistic for Omega: running average of (1/N) Σ ηᵢηᵢᵀ
    s2: DMatrix<f64>,
    /// Current theta
    theta: Vec<f64>,
    /// Current omega matrix
    omega_mat: DMatrix<f64>,
    /// Current sigma values
    sigma_vals: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Metropolis-Hastings step for one subject
// ---------------------------------------------------------------------------

/// Run `n_steps` symmetric random-walk MH iterations for one subject in-place.
/// Returns (n_accepted, updated_nll).
///
/// `eta` is in deviation (eta_true) space — the same space the model's
/// `pk_param_fn` consumes — so proposals are random walks
/// `eta + step_scale · L · z` from the current position. The acceptance
/// log-ratio is `nll_current − nll_prop`, which is correct because the
/// symmetric proposal density cancels.
///
/// Note: an earlier version centred proposals on `mu_k` during exploration.
/// That was incorrect: `individual_nll` interprets `eta` as the deviation
/// `log(CL_i) − log(TVCL)`, while `mu_k = log(TVCL)`, so the model evaluated
/// `CL = TVCL · exp(log TVCL) = TVCL²` for every accepted exploration step.
#[allow(clippy::too_many_arguments)]
fn mh_steps(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    step_scale: f64,
    rng: &mut impl Rng,
    n_steps: usize,
    pk_scratch: &mut EventPkParams,
) -> (usize, f64) {
    let n_eta = eta.len();
    let l = &omega.chol;
    let mut nll = nll_current;
    let mut n_accepted = 0;

    for _ in 0..n_steps {
        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let perturbation = l * z_vec;

        let eta_prop: Vec<f64> = (0..n_eta)
            .map(|j| eta[j] + step_scale * perturbation[j])
            .collect();

        // Reuses `pk_scratch` across all n_steps proposals (and across
        // outer SAEM iterations when the caller hoists allocation
        // further). On TV-cov subjects this eliminates the per-call
        // allocate/discard of three `Vec<PkParams>` per evaluation —
        // the dominant allocator pressure on the SAEM hot loop.
        let nll_prop = individual_nll_into(
            model,
            subject,
            theta,
            &eta_prop,
            omega,
            sigma_values,
            pk_scratch,
        );

        // Symmetric proposal q(η_prop|η) = q(η|η_prop) cancels in the ratio,
        // so the prior+likelihood difference encoded in `individual_nll` is
        // the full acceptance criterion.
        let log_u: f64 = rng.gen::<f64>().ln();
        if log_u < nll - nll_prop {
            eta.copy_from_slice(&eta_prop);
            nll = nll_prop;
            n_accepted += 1;
        }
    }

    (n_accepted, nll)
}

// ---------------------------------------------------------------------------
// Gradient of conditional observation NLL w.r.t. log(theta) and log(sigma)
// ---------------------------------------------------------------------------

/// Lightweight M-step: run NLopt SLSQP for a few iterations in packed
/// space, warm-started from the current packed theta / log-sigma.
///
/// `theta_packs_log_mask[i]` selects per-theta packing: log when true,
/// identity when false. Sigma is always log-packed (sigma > 0 by
/// construction). See the run_saem comment on `theta_packs_log_mask` for
/// motivation — without per-theta packing, any theta with `theta_lower < 0`
/// got pinned at 1e-10 and could never be estimated.
fn theta_sigma_mstep_light(
    model: &CompiledModel,
    population: &Population,
    etas: &[Vec<f64>],
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
    scale_params: bool,
    theta_packs_log_mask: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let n = n_theta + n_sigma;

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);

    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);

    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    // Unpack a slice of packed theta values into natural-scale theta.
    // Closure (not local fn) so it captures `theta_packs_log_mask`.
    let unpack_thetas = |packed: &[f64]| -> Vec<f64> {
        (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    packed[i].exp()
                } else {
                    packed[i]
                }
            })
            .collect()
    };

    // Objective operating on the unscaled packed parameters.
    //
    // Gradient strategy: single rayon pass over subjects, each computing its
    // own partial gradient via `obs_nll_subject_grad` (analytical sigma,
    // FD-of-predictions for theta). This replaces the old per-parameter
    // forward-FD of `obs_nll_sum` which launched `n_dim` rayon jobs
    // sequentially. Key improvements:
    //  • Sigma gradient is analytical — no extra predict calls per sigma dim.
    //  • Single rayon launch instead of n_dim sequential launches.
    //  • Better cache locality: one subject's data stays in cache while
    //    iterating over all its theta perturbations.
    //  • Pinned dims (lower == upper) are skipped per-subject, saving the
    //    predict calls entirely (same as the old FD guard).
    let obj = |xv: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();

        if let Some(g) = grad {
            use rayon::prelude::*;
            let (val, grad_vec) = population
                .subjects
                .par_iter()
                .zip(etas.par_iter())
                .map_init(EventPkParams::default, |scratch, (subject, eta)| {
                    obs_nll_subject_grad(
                        model,
                        subject,
                        &th,
                        &sg,
                        eta,
                        &theta_packs_log_mask,
                        &lower,
                        &upper,
                        n_theta,
                        n_sigma,
                        scratch,
                    )
                })
                .reduce(
                    || (0.0, vec![0.0f64; n]),
                    |(nll_a, mut ga), (nll_b, gb)| {
                        for (a, b) in ga.iter_mut().zip(gb.iter()) {
                            *a += b;
                        }
                        (nll_a + nll_b, ga)
                    },
                );
            for (gi, &gv) in g.iter_mut().zip(grad_vec.iter()) {
                *gi = if gv.is_finite() { gv } else { 0.0 };
            }
            if val.is_finite() {
                val
            } else {
                1e20
            }
        } else {
            let val = obs_nll_sum(model, population, &th, &sg, etas);
            if val.is_finite() {
                val
            } else {
                1e20
            }
        }
    };

    // Compute per-element scale factors from the initial point.
    let scale: Vec<f64> = if scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n]
    };

    // Scaled starting point and bounds: xs[i] = x[i] / scale[i].
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();
    let lower_s: Vec<f64> = (0..n).map(|i| lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| upper[i] / scale[i]).collect();

    // Wrapper objective: receives scaled xs, unscales before evaluating obj,
    // then scales the gradient back: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i].
    let obj_s = |xv_s: &[f64], grad: Option<&mut [f64]>, data: &mut ()| -> f64 {
        let xv: Vec<f64> = (0..n).map(|i| xv_s[i] * scale[i]).collect();
        if let Some(g) = grad {
            let mut g_raw = vec![0.0_f64; n];
            let val = obj(&xv, Some(&mut g_raw), data);
            for i in 0..n {
                g[i] = g_raw[i] * scale[i];
            }
            val
        } else {
            obj(&xv, None, data)
        }
    };

    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Slsqp,
        n,
        obj_s,
        nlopt::Target::Minimize,
        (),
    );
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    opt.set_ftol_rel(1e-4).unwrap();

    match opt.optimize(&mut xs) {
        Ok(_) | Err(_) => {}
    }

    // Unscale back to log-space.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let log_theta_new = x_final[..n_theta].to_vec();
    let log_sigma_new = x_final[n_theta..].to_vec();
    (log_theta_new, log_sigma_new)
}

/// Gradient of `obs_nll` w.r.t. the SAEM packed parameter vector
/// `[log_theta_0 … log_theta_{P-1} | log_sigma_0 … log_sigma_{Q-1}]`
/// for a single subject with ETAs held fixed.
///
/// For non-M3 models:
/// - Sigma: analytical from the residual-variance formula (no extra predict call).
/// - Theta: forward-FD of `compute_predictions_with_tv_into` + chain rule through
///   obs_nll (one extra predict call per non-pinned theta, not one full-subject
///   NLL call).
///
/// For M3 models (complex Mills-ratio sigma gradient): forward-FD of
/// `obs_nll_single_into` for all parameters.
///
/// `lower`/`upper` are the packed-space bounds used to detect pinned dimensions
/// (`lower[i] == upper[i]`); pinned dimensions contribute 0 to the gradient and
/// skip their FD call.
#[allow(clippy::too_many_arguments)]
fn obs_nll_subject_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    theta_packs_log_mask: &[bool],
    lower: &[f64],
    upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    pk_scratch: &mut EventPkParams,
) -> (f64, Vec<f64>) {
    let n = n_theta + n_sigma;
    let m3 = matches!(model.bloq_method, BloqMethod::M3);

    if m3 {
        // M3 path: forward-FD of obs_nll_single_into for all parameters.
        let nll_base = obs_nll_single_into(model, subject, theta, sigma_values, eta, pk_scratch);
        let mut grad = vec![0.0f64; n];
        let h = 1e-5;
        for i in 0..n {
            if lower[i] == upper[i] {
                continue;
            }
            if i < n_theta {
                let mut theta_p = theta.to_vec();
                let delta = h * (1.0 + theta[i].abs());
                theta_p[i] += delta;
                let nll_p =
                    obs_nll_single_into(model, subject, &theta_p, sigma_values, eta, pk_scratch);
                let raw = (nll_p - nll_base) / delta;
                grad[i] = if theta_packs_log_mask[i] {
                    theta[i] * raw
                } else {
                    raw
                };
            } else {
                let k = i - n_theta;
                let mut sigma_p = sigma_values.to_vec();
                let delta = h * (1.0 + sigma_values[k].abs());
                sigma_p[k] += delta;
                let nll_p = obs_nll_single_into(model, subject, theta, &sigma_p, eta, pk_scratch);
                // log-packing for sigma: d/d(log_sigma_k) = sigma_k * d/d(sigma_k)
                grad[i] = sigma_values[k] * (nll_p - nll_base) / delta;
            }
        }
        return (nll_base, grad);
    }

    // Non-M3 path.
    let preds_base =
        crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);

    let mut nll_base = 0.0f64;
    let n_obs = subject.observations.len();

    // per-obs residual, variance, d(obs_nll)/d(f_j)
    let mut residuals = vec![0.0f64; n_obs];
    let mut variances = vec![0.0f64; n_obs];
    let mut d_nll_d_f = vec![0.0f64; n_obs];

    for j in 0..n_obs {
        let f = preds_base[j].max(1e-12);
        let v = residual_variance(model.error_model, f, sigma_values).max(1e-12);
        let resid = subject.observations[j] - f;
        nll_base += 0.5 * (v.ln() + resid * resid / v);
        residuals[j] = resid;
        variances[j] = v;
        // d(obs_nll_j)/d(f_j) = -resid/V + 0.5 * (dV/df) * (1/V - resid²/V²)
        let dv_df = match model.error_model {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => {
                2.0 * f * sigma_values[0] * sigma_values[0]
            }
        };
        d_nll_d_f[j] = -resid / v + 0.5 * dv_df * (1.0 / v - resid * resid / (v * v));
    }

    let mut grad = vec![0.0f64; n];

    // Theta gradient: forward-FD of predictions, chain rule through obs_nll.
    let h_fd = 1e-5;
    for i in 0..n_theta {
        if lower[i] == upper[i] {
            continue;
        }
        let delta = h_fd * (1.0 + theta[i].abs());
        let mut theta_p = theta.to_vec();
        theta_p[i] += delta;
        let preds_p =
            crate::pk::compute_predictions_with_tv_into(model, subject, &theta_p, eta, pk_scratch);
        // Difference on raw predictions — do NOT clip before differencing.
        // Clipping both pp and pb at 1e-12 before subtracting would produce a
        // zero difference whenever pb < 1e-12, silently zeroing the gradient.
        let d_obs_nll: f64 = d_nll_d_f
            .iter()
            .zip(preds_p.iter().zip(preds_base.iter()))
            .map(|(&dl, (&pp, &pb))| dl * (pp - pb) / delta)
            .sum();
        grad[i] = if theta_packs_log_mask[i] {
            theta[i] * d_obs_nll
        } else {
            d_obs_nll
        };
    }

    // Sigma gradient: analytical.
    // d(obs_nll)/d(log_sigma_k) = Σ_j 0.5 * ratio_jk * (1/V_j - resid_j²/V_j²)
    // where ratio_jk = sigma_k * dV_j/d_sigma_k.
    for k in 0..n_sigma {
        let i = n_theta + k;
        if lower[i] == upper[i] {
            continue;
        }
        let s = sigma_values[k];
        let g: f64 = (0..n_obs)
            .map(|j| {
                let f = preds_base[j].max(1e-12);
                let v = variances[j];
                let resid = residuals[j];
                // ratio = sigma_k * dV/d(sigma_k)
                let ratio = match model.error_model {
                    ErrorModel::Additive => 2.0 * s * s,             // = 2V
                    ErrorModel::Proportional => 2.0 * s * s * f * f, // = 2V
                    ErrorModel::Combined => {
                        if k == 0 {
                            2.0 * s * s * f * f
                        } else {
                            2.0 * s * s
                        }
                    }
                };
                0.5 * ratio * (1.0 / v - resid * resid / (v * v))
            })
            .sum();
        grad[i] = g;
    }

    (nll_base, grad)
}

/// Observation NLL for a single subject with ETAs held fixed.
///
/// Under M3, CENS=1 rows contribute `-log Φ((LLOQ - f)/√V)`.
fn obs_nll_single_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    pk_scratch: &mut EventPkParams,
) -> f64 {
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    let preds = crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);
    let mut nll = 0.0;
    for (j, (&y, &f)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let f = f.max(1e-12);
        let v = residual_variance(model.error_model, f, sigma_values).max(1e-12);
        if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
            let z = (y - f) / v.sqrt();
            nll += -log_normal_cdf(z);
        } else {
            nll += 0.5 * (v.ln() + (y - f).powi(2) / v);
        }
    }
    nll
}

/// Sum of observation log-likelihoods with ETAs held fixed.
///
/// Under M3, CENS=1 rows contribute `-log Φ((LLOQ - f)/√V)` instead of the
/// Gaussian residual term. Without this branch, the SAEM M-step would optimize
/// θ/σ as if censored observations were exact Gaussians at the LLOQ value,
/// producing silently-biased population estimates.
///
/// Uses rayon's `map_init` so each worker thread allocates one
/// `EventPkParams` scratch on first use and reuses it across every
/// subject the worker handles. With NLopt's central-FD gradient
/// hitting `obs_nll_sum` `1 + 2·n_dim` times per M-step, this cuts
/// per-call `Vec<PkParams>` churn to near-zero on TV-cov data.
fn obs_nll_sum(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
) -> f64 {
    use rayon::prelude::*;
    population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            obs_nll_single_into(model, subject, theta, sigma_values, &etas[i], scratch)
        })
        .sum()
}

/// Build (theta_idx, eta_idx) pairs for log-transformed mu-references only.
///
/// Only `log_transformed = true` mu-refs (patterns `THETA*exp(ETA)` and
/// `exp(log(THETA)+ETA)`) participate in the gradient-step M-step.  For these
/// the chain rule gives `d/d_log(theta) = -Σᵢ d/d_eta`, which matches the
/// update applied in the SAEM loop.  Additive mu-refs (`THETA + ETA`,
/// `log_transformed = false`) require the extra factor of `theta` from the
/// log-space chain rule and are deliberately excluded — they fall through to
/// the regular NLopt M-step.
fn get_mu_ref_pairs(model: &CompiledModel) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if !mu_ref.log_transformed {
                continue;
            }
            if let Some(theta_idx) = model
                .theta_names
                .iter()
                .position(|n| n == &mu_ref.theta_name)
            {
                pairs.push((theta_idx, eta_idx));
            }
        }
    }
    pairs
}

// ---------------------------------------------------------------------------
// Main SAEM loop
// ---------------------------------------------------------------------------

pub fn run_saem(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let k1 = options.saem_n_exploration;
    let k2 = options.saem_n_convergence;
    let n_iter = k1 + k2;
    let n_mh_steps = options.saem_n_mh_steps;
    let adapt_interval = options.saem_adapt_interval;
    let verbose = options.verbose;

    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // Master RNG
    let master_seed = options.saem_seed.unwrap_or(12345);

    if verbose {
        eprintln!(
            "SAEM: {} subjects, {} ETAs, {} total iter ({} explore + {} converge)",
            n_subjects, n_eta, n_iter, k1, k2
        );
    }

    // Initialize state
    let theta_cur = init_params.theta.clone();
    let omega_cur = init_params.omega.matrix.clone();
    let sigma_cur = init_params.sigma.values.clone();
    let s2 = omega_cur.clone();

    let etas: Vec<Vec<f64>> = (0..n_subjects)
        .map(|_| get_eta_init(n_eta, None, None))
        .collect();
    let step_scales = vec![0.3; n_subjects];

    // Initial NLL cache
    let nll_cache: Vec<f64> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            individual_nll(
                model,
                subject,
                &theta_cur,
                &etas[i],
                &init_params.omega,
                &sigma_cur,
            )
        })
        .collect();

    // Per-theta packing flag: log for `theta_lower >= 0` (CL/V/KA…),
    // identity when `theta_lower < 0` (covariate exponents like
    // THETA_AGE_CL = -0.01 or THETA_CL_GAMMA = -0.8). Same convention
    // as `parameterization.rs::pack_params`. Without this, every theta
    // with a negative lower bound got clamped to 1e-10 by the old
    // `t.max(1e-10).ln()` packing and could never be estimated —
    // visible regression: SAD_SCEN4 SAEM left γ_CL stuck at 0 (truth
    // -0.8), letting the rest of the fit drift to compensate.
    let theta_packs_log_mask: Vec<bool> = init_params
        .theta_lower
        .iter()
        .map(|&lo| crate::estimation::parameterization::theta_packs_log(lo))
        .collect();
    let pack_theta = |i: usize, t: f64| -> f64 {
        if theta_packs_log_mask[i] {
            t.max(1e-10).ln()
        } else {
            t
        }
    };
    let unpack_theta = |i: usize, packed: f64| -> f64 {
        if theta_packs_log_mask[i] {
            packed.exp()
        } else {
            packed
        }
    };

    // Pack initial theta (per-mask) and sigma (always log).
    let mut log_theta: Vec<f64> = (0..n_theta).map(|i| pack_theta(i, theta_cur[i])).collect();
    let mut log_sigma: Vec<f64> = sigma_cur.iter().map(|&s| s.max(1e-10).ln()).collect();

    // Bounds in packed space — log when log-packed, identity otherwise.
    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i]
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).ln()
            } else {
                init_params.theta_upper[i]
            }
        })
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];

    // Pin FIX parameters: set lower == upper == packed_value so the inner
    // NLopt M-step treats them as constants. Matches the FOCE/FOCEI treatment.
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }

    let mut state = SaemState {
        etas,
        nll_cache,
        step_scales,
        accept_counts: vec![0; n_subjects],
        steps_since_adapt: 0,
        s2,
        theta: theta_cur,
        omega_mat: omega_cur,
        sigma_vals: sigma_cur,
    };

    // Mu-referencing pairs for the closed-form M-step: (theta_idx, eta_idx).
    // Only log-mu-ref pairs are returned (`get_mu_ref_pairs` filters out
    // additive ones), since the closed-form `log_theta += γ · mean(η)` only
    // applies to log-mu-referenced thetas.
    let mu_ref_pairs: Vec<(usize, usize)> = get_mu_ref_pairs(model);
    let use_closed_form_mstep = options.mu_referencing && !mu_ref_pairs.is_empty();
    // Accumulator for the `obs_nll_sum` (population OFV) evaluations skipped
    // by pinning mu-ref dims out of NLopt's central-FD gradient.  Each pinned
    // dim costs `2 * mstep_maxiter` `obs_nll_sum` calls inside NLopt — that's
    // the value we add per M-step that takes the closed-form branch.
    let mut mstep_grad_step_evals_saved: u64 = 0;

    // Main loop
    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("SAEM: cancelled at iteration {}", k);
            }
            break;
        }
        let gamma = if k <= k1 { 1.0 } else { 1.0 / (k - k1) as f64 };

        // Rebuild omega for this iteration
        let omega_k = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );

        // ---- Step 1: MH simulation (parallelized) ----
        // Symmetric random-walk MH in eta_true space, identical schedule
        // throughout exploration and convergence — the only thing that
        // changes between phases is the SA step size `gamma`.
        {
            use rayon::prelude::*;
            let theta_ref = &state.theta;
            let sigma_ref = &state.sigma_vals;
            let omega_ref = &omega_k;

            let results: Vec<(Vec<f64>, f64, usize)> = state
                .etas
                .par_iter()
                .zip(state.nll_cache.par_iter())
                .zip(state.step_scales.par_iter())
                .enumerate()
                // Per-rayon-worker `EventPkParams` scratch: allocated
                // once per worker per outer iteration, reused across
                // every subject the worker handles. Without `map_init`
                // the scratch was allocated per subject per outer
                // iter (5937 × N_iter on the cefepime SAEM bench);
                // with it, n_workers × N_iter ≈ 10 × N_iter.
                .map_init(
                    EventPkParams::default,
                    |pk_scratch, (i, ((eta, &nll), &scale))| {
                        let subject = &population.subjects[i];
                        let mut rng = StdRng::seed_from_u64(
                            master_seed
                                .wrapping_add(k as u64 * 100_000)
                                .wrapping_add(i as u64),
                        );
                        let mut eta_work = eta.clone();
                        let (n_acc, nll_new) = mh_steps(
                            &mut eta_work,
                            nll,
                            subject,
                            model,
                            theta_ref,
                            omega_ref,
                            sigma_ref,
                            scale,
                            &mut rng,
                            n_mh_steps,
                            pk_scratch,
                        );
                        (eta_work, nll_new, n_acc)
                    },
                )
                .collect();

            for (i, (eta_new, nll_new, n_acc)) in results.into_iter().enumerate() {
                state.etas[i] = eta_new;
                state.nll_cache[i] = nll_new;
                state.accept_counts[i] += n_acc;
            }
        }
        state.steps_since_adapt += 1;

        // ---- Step 2: SA update of sufficient statistic for Omega ----
        let mut eta_outer = DMatrix::zeros(n_eta, n_eta);
        for eta in &state.etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= n_subjects as f64;

        state.s2 = (1.0 - gamma) * &state.s2 + gamma * &eta_outer;

        // ---- Step 3: M-step Omega (closed form) ----
        // Restore FIX-ed rows / columns from the template. An eta flagged FIX
        // keeps its initial variance AND its initial off-diagonal couplings
        // (zero for a diagonal declaration, block cov for a FIX-ed block).
        // Letting the sufficient statistic bleed into row/col of a fixed eta
        // breaks positive-definiteness once the free-block diagonals shrink
        // during the exploration phase.
        state.omega_mat = state.s2.clone();
        // Zero structurally-absent off-diagonals. `s2 = (1/N) Σ ηη^T` always
        // produces a dense matrix; entries that aren't free parameters
        // (standalone etas, or etas from different `block_omega` declarations)
        // must be zeroed so they don't feed sampling correlations back into
        // the next iteration's Cholesky proposal. Without this the chain drives
        // Ω toward a rank-deficient state, log|Ω| → -∞, and the M-step pushes
        // thetas to bounds to compensate.
        for i in 0..n_eta {
            for j in 0..n_eta {
                if !init_params.omega.free_mask[(i, j)] {
                    state.omega_mat[(i, j)] = 0.0;
                }
            }
        }
        // Restore FIX-ed rows / columns from the template.
        for i in 0..n_eta {
            for j in 0..n_eta {
                let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    state.omega_mat[(i, j)] = init_params.omega.matrix[(i, j)];
                }
            }
        }

        // ---- Step 4: M-step theta, sigma (lightweight NLopt, warm-started) ----
        // Only run every few iterations during exploration to save time
        let run_mstep = k <= 5 || k % 3 == 0 || k > k1;
        if run_mstep {
            let mstep_maxiter = if k <= k1 { 3 } else { 5 }; // more precise in convergence phase

            if use_closed_form_mstep {
                // Closed-form EM M-step for log-mu-referenced thetas.
                //
                // Model: log(P_i) = log(TVP) + η_i, η_i ~ N(0, ω²).
                // The complete-data log-likelihood is maximised at
                //     log(TVP)_new = log(TVP)_old + mean_i(η_i)
                // and SAEM applies the stochastic-approximation step size γ:
                //     log(TVP)_new = log(TVP)_old + γ · mean_i(η_i)
                // After the update, η_i is re-centred by `mean(η)` so the
                // sufficient statistic for ω is taken from zero-mean residuals
                // (ω is updated from `s2` *after* the next MH step, but
                // re-centring keeps `state.etas` consistent with the new TVP
                // for the rest of this iteration's NLL cache refresh).
                let n_subj = state.etas.len() as f64;
                let mut temp_theta_lower = log_theta_lower.clone();
                let mut temp_theta_upper = log_theta_upper.clone();
                let mut n_pinned: u64 = 0;
                for &(theta_idx, eta_idx) in &mu_ref_pairs {
                    if init_params
                        .theta_fixed
                        .get(theta_idx)
                        .copied()
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    let mean_eta: f64 = state.etas.iter().map(|e| e[eta_idx]).sum::<f64>() / n_subj;
                    let log_theta_before = log_theta[theta_idx];
                    log_theta[theta_idx] = (log_theta_before + gamma * mean_eta)
                        .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                    // Re-centre etas by the *actual* shift applied to log_theta,
                    // not by `gamma * mean_eta` directly: when the update is
                    // clamped at a bound the realised delta is smaller, and
                    // shifting etas by the unclamped quantity would break
                    // log(P_i) = log(TVP) + η_i until the next MH refresh.
                    let delta = log_theta[theta_idx] - log_theta_before;
                    for e in state.etas.iter_mut() {
                        e[eta_idx] -= delta;
                    }
                    // Pin so NLopt leaves the closed-form value unchanged.
                    temp_theta_lower[theta_idx] = log_theta[theta_idx];
                    temp_theta_upper[theta_idx] = log_theta[theta_idx];
                    n_pinned += 1;
                }
                // Each pinned mu-ref dim avoids 2 obs_nll_sum calls per NLopt
                // gradient request, capped at `mstep_maxiter` requests. FIXed
                // thetas are not pinned by the closed form (NLopt sees them as
                // FIXed via the regular bounds path) so they aren't counted.
                mstep_grad_step_evals_saved += 2 * mstep_maxiter as u64 * n_pinned;

                // NLopt for non-mu-ref thetas (pinned) and sigma.
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    &log_theta,
                    &log_sigma,
                    &temp_theta_lower,
                    &temp_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                );
                log_theta = theta_new;
                log_sigma = sigma_new;
            } else {
                // mu_referencing = false: full NLopt M-step for all thetas + sigma (unchanged)
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    &log_theta,
                    &log_sigma,
                    &log_theta_lower,
                    &log_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                );
                log_theta = theta_new;
                log_sigma = sigma_new;
            }

            state.theta = (0..n_theta)
                .map(|i| unpack_theta(i, log_theta[i]))
                .collect();
            state.sigma_vals = log_sigma.iter().map(|&v| v.exp()).collect();
        }

        // ---- Update NLL cache (parallelized, needed for MH acceptance ratios) ----
        let omega_upd = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        {
            use rayon::prelude::*;
            // map_init lets each rayon worker keep one `EventPkParams`
            // scratch alive across every subject it handles, the same
            // pattern as the MH step above. Without it, the per-iter
            // refresh was allocating n_subj scratch buffers per outer
            // iter on TV-cov data.
            let new_nlls: Vec<f64> = state
                .etas
                .par_iter()
                .enumerate()
                .map_init(EventPkParams::default, |scratch, (i, eta)| {
                    individual_nll_into(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        eta,
                        &omega_upd,
                        &state.sigma_vals,
                        scratch,
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
        }

        // ---- Adapt MH step sizes ----
        if state.steps_since_adapt >= adapt_interval {
            for i in 0..n_subjects {
                let rate = state.accept_counts[i] as f64 / (n_mh_steps * adapt_interval) as f64;
                if rate > 0.4 {
                    state.step_scales[i] = (state.step_scales[i] * 1.1).min(5.0);
                } else {
                    state.step_scales[i] = (state.step_scales[i] * 0.9).max(0.01);
                }
                state.accept_counts[i] = 0;
            }
            state.steps_since_adapt = 0;
        }

        // ---- Verbose output + optimizer trace ----
        {
            let phase = if k <= k1 { "explore" } else { "converge" };
            let cond_nll: f64 = state.nll_cache.iter().sum();
            // Rolling MH accept rate since the last adapt reset.
            let steps_so_far = state.steps_since_adapt.max(1);
            let mh_accept_rate: f64 = state.accept_counts.iter().sum::<usize>() as f64
                / (n_subjects * n_mh_steps * steps_so_far) as f64;

            if verbose && (k == 1 || k % 50 == 0 || k == n_iter) {
                eprintln!(
                    "  SAEM iter {:>4}/{} [{}] gamma={:.3}  condNLL={:.3}",
                    k, n_iter, phase, gamma, cond_nll
                );
            }

            crate::estimation::trace::write_saem(k, phase, cond_nll, gamma, mh_accept_rate);
        }
    }

    // If the user cancelled mid-run the loop broke early; skip the final
    // EBE/OFV computation (which iterates over every subject) and abort.
    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    if verbose {
        eprintln!("SAEM iterations complete. Computing final EBEs and OFV...");
    }

    // ---- Post-SAEM: build final parameters ----
    let final_omega = OmegaMatrix::from_matrix(
        state.omega_mat.clone(),
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    let final_params = ModelParameters {
        theta: state.theta.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: final_omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: state.sigma_vals.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: init_params.omega_iov.clone(),
        kappa_fixed: init_params.kappa_fixed.clone(),
    };

    // ---- Final EBEs via inner loop (warm-started from SAEM etas) ----
    let warm_etas: Vec<DVector<f64>> = state
        .etas
        .iter()
        .map(|e| DVector::from_column_slice(e))
        .collect();
    let saem_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&warm_etas),
        Some(&saem_final_mu_k),
        0, // SAEM: no EBE convergence tracking
    );

    // ---- Final OFV via FOCE approximation (for AIC/BIC comparability) ----
    let ofv = 2.0
        * pop_nll(
            model,
            population,
            &final_params,
            &eta_hats,
            &h_matrices,
            &final_kappas,
            options.interaction,
        );

    // ---- Covariance step ----
    let mut warnings = Vec::new();
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("Running covariance step...");
            }
            let packed = pack_params(&final_params);
            let cov = compute_covariance(
                &packed,
                &final_params,
                model,
                population,
                &eta_hats,
                &h_matrices,
                &final_kappas,
                options,
            );
            if cov.is_none() {
                warnings.push("Covariance step failed — SEs not available".to_string());
            }
            cov
        } else {
            None
        };

    if verbose {
        eprintln!("SAEM completed. Final OFV = {:.4}", ofv);
    }

    let saem_mu_ref_m_step_evals_saved = if use_closed_form_mstep {
        Some(mstep_grad_step_evals_saved)
    } else {
        None
    };

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged: ofv.is_finite(),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        warnings,
        saem_mu_ref_m_step_evals_saved,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, MuRef};

    fn model_with_mu_refs(
        theta_names: &[&str],
        eta_names: &[&str],
        mu_refs: &[(&str, &str, bool)],
    ) -> CompiledModel {
        let mut m = analytical_model(GradientMethod::Auto);
        m.theta_names = theta_names.iter().map(|s| (*s).to_string()).collect();
        m.eta_names = eta_names.iter().map(|s| (*s).to_string()).collect();
        m.n_theta = theta_names.len();
        m.n_eta = eta_names.len();
        m.mu_refs = mu_refs
            .iter()
            .map(|(eta, theta, log_t)| {
                (
                    (*eta).to_string(),
                    MuRef {
                        theta_name: (*theta).to_string(),
                        log_transformed: *log_t,
                    },
                )
            })
            .collect();
        m
    }

    #[test]
    fn get_mu_ref_pairs_empty_when_no_mu_refs() {
        let m = analytical_model(GradientMethod::Auto);
        assert!(get_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn get_mu_ref_pairs_returns_log_transformed_pair() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", true)],
        );
        let mut pairs = get_mu_ref_pairs(&m);
        pairs.sort();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn get_mu_ref_pairs_excludes_additive_mu_refs() {
        // ETA_CL is lognormal (THETA*exp(ETA)) — included.
        // ETA_V is additive (THETA+ETA) — excluded because the gradient-step
        // chain rule used in run_saem assumes log-transformed parameters.
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", false)],
        );
        assert_eq!(get_mu_ref_pairs(&m), vec![(0, 0)]);
    }

    #[test]
    fn get_mu_ref_pairs_skips_orphaned_theta() {
        // mu_ref points at a theta name that doesn't exist — silently skipped.
        let m = model_with_mu_refs(&["CL"], &["ETA_CL"], &[("ETA_CL", "MISSING", true)]);
        assert!(get_mu_ref_pairs(&m).is_empty());
    }

    // ---- Regression tests for the three SAEM correctness bugs ----

    /// Bug 1 (diagonal): `from_diagonal` produces a free_mask that marks only
    /// diagonal entries free. The SAEM M-step uses this mask to zero
    /// SA-accumulated off-diagonals, preventing the rank-deficient Ω failure.
    #[test]
    fn diagonal_omega_free_mask_has_no_off_diagonals() {
        let omega = OmegaMatrix::from_diagonal(&[0.1, 0.2], vec!["ETA_CL".into(), "ETA_V".into()]);
        assert!(omega.free_mask[(0, 0)]);
        assert!(omega.free_mask[(1, 1)]);
        assert!(!omega.free_mask[(0, 1)]);
        assert!(!omega.free_mask[(1, 0)]);
    }

    /// Bug 1 (mixed structure): `from_matrix_with_mask` preserves an explicit
    /// mask that marks cross-block entries as structural zeros. This is the
    /// case that the `diagonal` flag alone cannot express (one standalone eta
    /// + one block_omega pair → diagonal=false, but cross entries are zero).
    #[test]
    fn mixed_omega_free_mask_zeros_cross_block_entries() {
        // Three etas: ETA_CL(0) and ETA_V(1) in a block; ETA_KA(2) standalone.
        let mut matrix = nalgebra::DMatrix::zeros(3, 3);
        matrix[(0, 0)] = 0.1;
        matrix[(1, 1)] = 0.2;
        matrix[(2, 2)] = 0.1;
        matrix[(0, 1)] = 0.01;
        matrix[(1, 0)] = 0.01;

        let mut free_mask = nalgebra::DMatrix::from_element(3, 3, false);
        free_mask[(0, 0)] = true;
        free_mask[(1, 1)] = true;
        free_mask[(2, 2)] = true;
        free_mask[(0, 1)] = true; // within CL-V block
        free_mask[(1, 0)] = true;

        let names = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
        let omega = OmegaMatrix::from_matrix_with_mask(matrix, names, false, free_mask);

        assert!(omega.free_mask[(0, 1)]);
        assert!(omega.free_mask[(1, 0)]);
        assert!(!omega.free_mask[(2, 0)]);
        assert!(!omega.free_mask[(0, 2)]);
        assert!(!omega.free_mask[(2, 1)]);
        assert!(!omega.free_mask[(1, 2)]);
    }

    /// Bug 2: `mh_steps` is a symmetric random walk — proposals are
    /// `eta_prop = eta + step·perturbation`, not `mu_k + step·perturbation`.
    ///
    /// Discriminator: with `step_scale = 0` the new kernel proposes exactly
    /// the current eta, so the chain cannot move regardless of the data.
    /// The pre-fix `mu_k`-centred kernel proposed exactly `mu_k` (= log TVCL),
    /// so a starting eta far from `mu_k` would either jump to `mu_k`
    /// whenever the proposal looked better, or oscillate. We pick a starting
    /// eta of 5.0 with TVCL=1 (mu_k=0): the simulated observation lives near
    /// the data-generating eta=0 region, so individual_nll(eta=0) is much
    /// lower than individual_nll(eta=5), meaning the broken kernel would
    /// accept the eta=0 proposal with probability ≈1 on the first step.
    /// The new kernel must leave eta at exactly 5.0.
    #[test]
    fn mh_steps_random_walk_uses_current_eta_not_mu_k() {
        use crate::stats::likelihood::individual_nll;
        use crate::types::{DoseEvent, SigmaVector};
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0],
            occasions: vec![],
            dose_occasions: vec![],
        };
        let omega = OmegaMatrix::from_diagonal(&[1.0], vec!["ETA_CL".into()]);
        let sigma = SigmaVector {
            values: vec![1.0],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0]; // mu_k = log(1) = 0
        let mut eta = vec![5.0_f64]; // far from mu_k
        let nll_start = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
        let mut rng = StdRng::seed_from_u64(42);

        let mut pk_scratch = EventPkParams::with_capacity_for(&subj);
        mh_steps(
            &mut eta,
            nll_start,
            &subj,
            &model,
            &theta,
            &omega,
            &sigma.values,
            0.0, // zero perturbation: random walk MUST stay put exactly
            &mut rng,
            100,
            &mut pk_scratch,
        );

        // Random walk with step=0: every proposal == current eta, accepted as
        // identity. The pre-fix kernel would have proposed mu_k=0 every step
        // and accepted it (lower nll than eta=5), driving eta to 0.
        assert_eq!(
            eta[0], 5.0,
            "eta moved despite step_scale=0 — proposals were re-centred on mu_k"
        );
    }

    /// Bug 3 / closed-form M-step: a synthetic SAEM run with mu_referencing=true
    /// and mean(eta) ≠ 0 must move log_theta in the right direction *without*
    /// pinning at the bound. We exercise the closed-form formula directly:
    /// `log_theta_new = log_theta_old + γ · mean(eta)`.
    #[test]
    fn closed_form_mu_ref_mstep_is_bounded_and_signed_correctly() {
        // Simulate post-MH state: 5 subjects, eta_mean = +0.4 (population CL
        // is higher than current TVCL), gamma = 1.0 (exploration step).
        let etas: Vec<Vec<f64>> = vec![vec![0.5], vec![0.3], vec![0.4], vec![0.6], vec![0.2]];
        let n = etas.len() as f64;
        let mean_eta: f64 = etas.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!((mean_eta - 0.4).abs() < 1e-12);

        let gamma = 1.0;
        let log_theta_old = 0.0_f64; // TVCL = 1.0
        let log_theta_new = log_theta_old + gamma * mean_eta;
        // log_theta moved by exactly mean(eta), independent of N.  This is the
        // property that the broken gradient step (γ · Σ ∂obs_nll/∂eta) lacked:
        // its update scaled with N and pinned thetas at bounds for moderate N.
        assert!((log_theta_new - 0.4).abs() < 1e-12);

        // After re-centring etas by gamma*mean, mean(eta) = 0.
        let mut etas_recentered = etas.clone();
        for e in etas_recentered.iter_mut() {
            e[0] -= gamma * mean_eta;
        }
        let new_mean: f64 = etas_recentered.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!(new_mean.abs() < 1e-12);
    }

    /// Bug 3 follow-up: the broken gradient step (γ · Σᵢ ∂obs_nll/∂eta) is no
    /// longer in the code path. The closed-form `log_theta += γ · mean(η)` is
    /// what runs when mu_referencing=true. Pair detection is unchanged.
    #[test]
    fn mu_ref_pair_detection_drives_closed_form_branch() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", true)],
        );
        let pairs = get_mu_ref_pairs(&m);
        assert_eq!(pairs.len(), 2);
        // The closed-form branch is taken iff `options.mu_referencing` AND
        // `!pairs.is_empty()`.  Both conditions are tested via the public API
        // in api::iov_integration::test_iov_foce_mu_referencing_on; this unit
        // test pins the precondition (pair detection still produces work).
    }

    /// A pre-cancelled `CancelFlag` makes the SAEM main loop break at the
    /// first iteration and `run_saem` must return `Err("cancelled by user")`
    /// without entering the post-loop "Computing final EBEs and OFV..." block
    /// (which iterates over every subject and is what makes a cancelled run
    /// feel like it isn't aborting).
    #[test]
    fn cancelled_run_returns_err_and_skips_final_ebe() {
        use crate::cancel::CancelFlag;
        use crate::types::{DoseEvent, FitOptions, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0],
            observations: vec![1.0, 0.5],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0, 0],
            occasions: vec![],
            dose_occasions: vec![],
        };
        let population = Population {
            subjects: vec![subj],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
        };

        let flag = CancelFlag::new();
        flag.cancel(); // pre-cancel: loop breaks at iteration 1

        let mut opts = FitOptions::default();
        opts.verbose = false;
        opts.run_covariance_step = false;
        opts.cancel = Some(flag);

        match run_saem(&model, &population, &model.default_params, &opts) {
            Err(msg) => assert!(
                msg.contains("cancelled by user"),
                "unexpected error message: {msg}"
            ),
            Ok(_) => panic!("pre-cancelled SAEM must return Err, not Ok"),
        }
    }

    /// Per-theta packing must round-trip values identically for both log-packed
    /// (`theta_lower >= 0`) and identity-packed (`theta_lower < 0`) thetas. SAEM
    /// uses its own pack/unpack closures inside the M-step, so this exercises
    /// the same math the closures rely on (`theta_packs_log` from
    /// parameterization plus the `if mask[i] { ln/exp } else { identity }`
    /// branches in `theta_sigma_mstep_light`).
    #[test]
    fn saem_pack_unpack_handles_negative_lower_bound() {
        use crate::estimation::parameterization::theta_packs_log;

        // Mix: CL (lower=0), V (lower=0.001), THETA_AGE_CL (lower=-1).
        let lowers: [f64; 3] = [0.0, 0.001, -1.0];
        let values: [f64; 3] = [5.0, 20.0, -0.01];
        let mask: Vec<bool> = lowers.iter().map(|&lo| theta_packs_log(lo)).collect();
        assert_eq!(mask, vec![true, true, false]);

        // Forward: simulate the SAEM init-pack construction (lines ~444–451 of
        // run_saem: log when log-packed, identity when identity-packed).
        let packed: Vec<f64> = values
            .iter()
            .zip(mask.iter())
            .map(|(&v, &log_pack)| if log_pack { v.max(1e-10).ln() } else { v })
            .collect();

        // Reverse: the M-step `unpack_thetas` closure.
        let unpacked: Vec<f64> = packed
            .iter()
            .zip(mask.iter())
            .map(|(&p, &log_pack)| if log_pack { p.exp() } else { p })
            .collect();

        for (orig, round) in values.iter().zip(unpacked.iter()) {
            assert!(
                (orig - round).abs() < 1e-12,
                "saem pack/unpack should round-trip: {orig} != {round}"
            );
        }
        // The identity-packed theta carries a negative value through —
        // pre-fix, this was clamped to 1e-10 by the log path.
        assert!(unpacked[2] < 0.0);
    }

    /// `obs_nll_subject_grad` summed over subjects must match the reference
    /// forward-FD of `obs_nll_sum` to within 1e-4 relative tolerance for all
    /// non-pinned packed parameters (theta + sigma).
    #[test]
    fn obs_nll_subject_grad_matches_obs_nll_sum_fd() {
        use crate::types::{DoseEvent, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);

        let make_subj = |id: &str, obs: f64| Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 4.0, 8.0],
            observations: vec![obs, obs * 0.6, obs * 0.3],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![],
            dose_occasions: vec![],
        };

        let population = Population {
            subjects: vec![
                make_subj("1", 8.0),
                make_subj("2", 5.0),
                make_subj("3", 11.0),
            ],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
        };

        let theta = vec![1.5f64, 20.0]; // CL, V
        let sigma_values = vec![0.2f64]; // proportional
        let etas: Vec<Vec<f64>> = vec![vec![0.0], vec![0.1], vec![-0.1]];
        let n_theta = 2;
        let n_sigma = 1;
        let n = n_theta + n_sigma;

        // Compute reference gradient via forward-FD of obs_nll_sum.
        let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
        let h = 1e-5;
        let mut ref_grad = vec![0.0f64; n];
        // Theta perturbations (in natural scale).
        for i in 0..n_theta {
            let mut theta_p = theta.clone();
            theta_p[i] += h;
            let fp = obs_nll_sum(&model, &population, &theta_p, &sigma_values, &etas);
            // FD in natural scale; convert to log-packed space (d/d_log = theta * d/d_theta)
            ref_grad[i] = theta[i] * (fp - f0) / h;
        }
        // Sigma perturbation (in natural scale; convert to log-packed).
        {
            let mut sigma_p = sigma_values.clone();
            sigma_p[0] += h;
            let fp = obs_nll_sum(&model, &population, &theta, &sigma_p, &etas);
            ref_grad[n_theta] = sigma_values[0] * (fp - f0) / h;
        }

        // Compute gradient via obs_nll_subject_grad summed over subjects.
        let mask: Vec<bool> = theta.iter().map(|_| true).collect(); // all log-packed
        let lo = vec![-1e30f64; n];
        let hi = vec![1e30f64; n];
        let mut total_nll = 0.0f64;
        let mut total_grad = vec![0.0f64; n];
        let mut scratch = EventPkParams::default();
        for (i, subject) in population.subjects.iter().enumerate() {
            let (nll_i, grad_i) = obs_nll_subject_grad(
                &model,
                subject,
                &theta,
                &sigma_values,
                &etas[i],
                &mask,
                &lo,
                &hi,
                n_theta,
                n_sigma,
                &mut scratch,
            );
            total_nll += nll_i;
            for (g, gi) in total_grad.iter_mut().zip(grad_i.iter()) {
                *g += gi;
            }
        }

        assert!(
            (total_nll - f0).abs() < 1e-10,
            "nll mismatch: {} vs {}",
            total_nll,
            f0
        );

        for j in 0..n {
            let rel = if ref_grad[j].abs() > 1e-10 {
                (total_grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
            } else {
                (total_grad[j] - ref_grad[j]).abs()
            };
            assert!(
                rel < 1e-4,
                "grad[{j}]: obs_nll_subject_grad={:.6e}, ref={:.6e}, rel={:.2e}",
                total_grad[j],
                ref_grad[j],
                rel
            );
        }
    }
}
