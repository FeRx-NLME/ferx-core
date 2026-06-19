//! Automatic differentiation gradient functions using `std::autodiff`.
//!
//! The AD functions take `tv_adjusted: &[f64]` — pre-computed typical values
//! that already incorporate covariates and theta. The inner loop computes:
//!   PK_param[i] = tv[i] * exp(eta[i])
//! so only eta is differentiated.

use crate::types::*;
use std::autodiff::{autodiff_forward, autodiff_reverse};

/// LTBS positivity floor for the AD paths. Mirrors [`crate::pk::LTBS_FLOOR`];
/// duplicated as a local `const` so the AD-instrumented code has no cross-module
/// dependency and Enzyme sees a plain literal.
const LTBS_FLOOR_AD: f64 = 1e-12;

/// Identity function that Enzyme can see through for type deduction but LLVM
/// cannot inline away. Provides an unambiguous `f64 -> f64` type boundary at
/// phi-node merge points where Enzyme's type-analysis would otherwise deadlock.
/// Currently unused (the `read_volatile` approach was chosen), kept as a
/// documented utility for future phi-node issues in AD-instrumented code.
#[inline(never)]
#[allow(dead_code)]
fn ad_type_fence(x: f64) -> f64 {
    x
}

// ─── Individual NLL: reverse-mode AD for gradient w.r.t. eta ────────────────

#[autodiff_reverse(
    individual_nll_ad_grad,
    Duplicated,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Active
)]
pub fn individual_nll_ad(
    eta: &[f64],
    tv: &[f64],             // covariate-adjusted typical values, length pk_idx_f64.len()
    omega_inv_flat: &[f64], // n_eta*n_eta, row-major
    log_det_omega: f64,
    sigma_values: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    obs_times: &[f64],
    observations: &[f64],
    cens_f64: &[f64], // per-observation censoring flag; +1 left-censored, -1 right-censored
    pk_idx_f64: &[f64], // PK parameter indices as f64 (cast to usize inside)
    sel_flat: &[f64], // n_tv × n_eta row-major one-hot eta selector
    pk_and_err_model: f64, // pk_model_id * 10 + error_model_id (+100 ⇒ LTBS)
    obs_scale: &[f64], // per-observation divisor (len = n_obs). All-ones = no-op.
) -> f64 {
    let n_eta = eta.len();
    let n_tv = tv.len();
    let n_doses = dose_times.len();
    let n_obs = obs_times.len();
    // LTBS is packed as a +100 offset on the model id (pk_model_id ≤ 5 ⇒ base
    // ≤ 52, so the offset is unambiguous). Under LTBS the effective prediction
    // is log(conc) and the error model is additive on the log scale.
    let ltbs = (pk_and_err_model as i32) >= 100;
    let base = (pk_and_err_model as i32) % 100;
    let pk_model_id = base / 10;
    let error_model_id = base % 10;

    // Eta prior: eta' * Omega_inv * eta
    let mut eta_prior = 0.0;
    for i in 0..n_eta {
        for j in 0..n_eta {
            eta_prior += eta[i] * omega_inv_flat[i * n_eta + j] * eta[j];
        }
    }

    // PK params: pk[idx] = tv[i] * exp(dot(sel_row_i, eta)). `sel_flat`
    // encodes which eta (if any) applies to each tv entry as a one-hot
    // row (length n_eta), with an all-zero row meaning "no eta". This is
    // fully branch-free on the differentiated path — Enzyme needs that
    // for reverse-mode type deduction to succeed; the earlier
    // `if has_eta` form produced NaN gradients on 2-cpt models because
    // the phi node at the if/else merge defeated Enzyme's type analysis.
    let mut pk = [0.0f64; MAX_PK_PARAMS];
    pk[PK_IDX_F] = 1.0;
    for i in 0..n_tv {
        let mut eta_contrib = 0.0;
        for j in 0..n_eta {
            eta_contrib += sel_flat[i * n_eta + j] * eta[j];
        }
        let idx = pk_idx_f64[i] as usize;
        pk[idx] = tv[i] * eta_contrib.exp();
    }

    // Volatile load prevents LLVM from merging this value with the array's
    // zero-initializer into a phi node that Enzyme cannot type-analyze.
    // Safety: `pk` is a stack-local [f64; MAX_PK_PARAMS] at a valid index.
    let lagtime = unsafe { core::ptr::read_volatile(&pk[PK_IDX_LAGTIME]) };

    // Predictions + data likelihood
    let mut data_ll = 0.0;
    for obs_idx in 0..n_obs {
        let t = obs_times[obs_idx];
        let mut conc = 0.0;
        for d in 0..n_doses {
            let tau = t - dose_times[d] - lagtime;
            conc += single_dose_ad(
                pk_model_id,
                tau,
                dose_amts[d],
                dose_rates[d],
                dose_durations[d],
                pk[PK_IDX_CL],
                pk[PK_IDX_V],
                pk[PK_IDX_Q],
                pk[PK_IDX_V2],
                pk[PK_IDX_KA],
                pk[PK_IDX_F],
                pk[PK_IDX_Q3],
                pk[PK_IDX_V3],
            );
        }
        if conc < 0.0 {
            conc = 0.0;
        }
        conc /= obs_scale[obs_idx];

        // LTBS: compare log(prediction) to the (log-scale) observation. Floor
        // with an explicit comparison — `f64::max` lowers to an LLVM intrinsic
        // Enzyme can't differentiate (see CLAUDE.md).
        if ltbs {
            let c = if conc < LTBS_FLOOR_AD {
                LTBS_FLOOR_AD
            } else {
                conc
            };
            conc = c.ln();
        }

        let v = residual_variance_ad(error_model_id, conc, sigma_values);
        let cens = cens_f64[obs_idx];
        if cens > 0.5 || cens < -0.5 {
            let z = if cens < -0.5 {
                (conc - observations[obs_idx]) / v.sqrt()
            } else {
                (observations[obs_idx] - conc) / v.sqrt()
            };
            data_ll += -2.0 * log_normal_cdf_ad(z);
        } else {
            let resid = observations[obs_idx] - conc;
            data_ll += resid * resid / v + v.ln();
        }
    }

    0.5 * (eta_prior + log_det_omega + data_ll)
}

/// AD-safe erf (Abramowitz & Stegun 7.1.26). Duplicated from stats/special.rs so
/// Enzyme sees the polynomial body inline and the LLVM IR contains no calls to
/// `llvm.maximumnum`/`llvm.minimumnum` intrinsics — see CLAUDE.md.
fn erf_ad(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// AD-safe log Φ(z). For z > -5 uses ln(max(Φ,floor)), for z ≤ -5 uses the
/// Mills-ratio asymptotic expansion. Both branches use only +, *, /, exp, ln —
/// no min/max intrinsics.
fn log_normal_cdf_ad(z: f64) -> f64 {
    // Pre-resolved constants: INV_SQRT_2 = 1/√2; LOG_SQRT_2PI = ln(√(2π)).
    const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    const MIN_PROB: f64 = 1e-300;

    if z > -5.0 {
        let p = 0.5 * (1.0 + erf_ad(z * INV_SQRT_2));
        let p_floor = if p < MIN_PROB { MIN_PROB } else { p };
        p_floor.ln()
    } else {
        let log_phi = -0.5 * z * z - LOG_SQRT_2PI;
        let inv_z2 = 1.0 / (z * z);
        let series = 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2 - 15.0 * inv_z2 * inv_z2 * inv_z2;
        log_phi - (-z).ln() + series.ln()
    }
}

// ─── Predictions: forward-mode AD for Jacobian ─────────────────────────────

#[autodiff_forward(
    predict_all_ad_tangent,
    Dual,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Dual
)]
pub fn predict_all_ad(
    eta: &[f64],
    tv: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    obs_times: &[f64],
    pk_idx_f64: &[f64], // PK parameter indices as f64 (cast to usize inside)
    sel_flat: &[f64],   // n_tv × n_eta row-major one-hot eta selector
    pk_model_id: f64,
    obs_scale: &[f64], // per-observation divisor (len = n_obs). All-ones = no-op.
    out: &mut [f64],
) {
    let n_eta = eta.len();
    let n_tv = tv.len();
    let n_doses = dose_times.len();
    let n_obs = obs_times.len();
    // Same +100 LTBS packing as `individual_nll_ad`: the Jacobian must be
    // d log(f)/dη when LTBS is active, so the forward prediction is log-wrapped
    // here — otherwise the AD Jacobian (natural scale) and the FD/AD objective
    // (log scale) would disagree and corrupt FOCEI/CWRES.
    let ltbs = (pk_model_id as i32) >= 100;
    let pk_id = (pk_model_id as i32) % 100;

    let mut pk = [0.0f64; MAX_PK_PARAMS];
    pk[PK_IDX_F] = 1.0;
    for i in 0..n_tv {
        let mut eta_contrib = 0.0;
        for j in 0..n_eta {
            eta_contrib += sel_flat[i * n_eta + j] * eta[j];
        }
        let idx = pk_idx_f64[i] as usize;
        pk[idx] = tv[i] * eta_contrib.exp();
    }

    // Volatile load — see matching comment in `individual_nll_ad`.
    let lagtime = unsafe { core::ptr::read_volatile(&pk[PK_IDX_LAGTIME]) };

    for obs_idx in 0..n_obs {
        let t = obs_times[obs_idx];
        let mut conc = 0.0;
        for d in 0..n_doses {
            let tau = t - dose_times[d] - lagtime;
            conc += single_dose_ad(
                pk_id,
                tau,
                dose_amts[d],
                dose_rates[d],
                dose_durations[d],
                pk[PK_IDX_CL],
                pk[PK_IDX_V],
                pk[PK_IDX_Q],
                pk[PK_IDX_V2],
                pk[PK_IDX_KA],
                pk[PK_IDX_F],
                pk[PK_IDX_Q3],
                pk[PK_IDX_V3],
            );
        }
        let positive = if conc > 0.0 { conc } else { 0.0 };
        let scaled = positive / obs_scale[obs_idx];
        // LTBS log-wrap with an explicit-comparison floor (no `f64::max`, per
        // CLAUDE.md — Enzyme can't differentiate the max intrinsic).
        out[obs_idx] = if ltbs {
            let c = if scaled < LTBS_FLOOR_AD {
                LTBS_FLOOR_AD
            } else {
                scaled
            };
            c.ln()
        } else {
            scaled
        };
    }
}

// ─── Inlined PK equations ───────────────────────────────────────────────────

fn single_dose_ad(
    pk_model_id: i32,
    tau: f64,
    amt: f64,
    rate: f64,
    dur: f64,
    cl: f64,
    v: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if tau < 0.0 || v <= 0.0 || cl <= 0.0 {
        return 0.0;
    }

    // Bioavailability F scales the bioavailable amount/rate on every route — IV
    // included (#327) — matching the superposition path's `route_f_scale` and the
    // event-driven path's `PkParams::bioavailable_amount`/`bioavailable_rate`.
    // Applied once here (rather than per-arm) because this `#[autodiff]` function
    // works on flat scalars and cannot call those `&self` helpers. All six arms
    // below read the pre-scaled `amt`; only the IV arms read the pre-scaled `rate`.
    //
    // The oral arms (ids 1/3/5) evaluate only the bolus Bateman form — they ignore
    // `rate`/`dur`, so they are wrong for an oral *infusion* dose (which the value
    // path routes to the IV-infusion form). That is currently safe, not a live bug:
    // the inner optimizer sends every oral model with a `rate>0` dose to FD before
    // AD is selected (`is_oral_model && rate>0 -> Fd`, see
    // `estimation/inner_optimizer.rs`), so these arms only ever see bolus doses. If
    // that guard is removed, make the oral arms infusion-aware first (#349 review).
    let amt = f_bio * amt;
    let rate = f_bio * rate;

    // Per issue #176, IV variants no longer split by administration type at
    // the model level. Each IV branch below handles bolus and infusion via
    // the per-dose `dur` (and `rate`) — the `dur <= 0.0` fall-through is
    // exactly the bolus closed form. ID mapping (see `pk_model_to_id`):
    //   0 OneCptIv, 1 OneCptOral, 2 TwoCptIv, 3 TwoCptOral,
    //   4 ThreeCptIv, 5 ThreeCptOral.
    match pk_model_id {
        0 => {
            // OneCptIv — bolus when dur<=0, infusion otherwise.
            let k = cl / v;
            if dur <= 0.0 {
                (amt / v) * (-k * tau).exp()
            } else if tau <= dur {
                (rate / cl) * (1.0 - (-k * tau).exp())
            } else {
                (rate / cl) * (1.0 - (-k * dur).exp()) * (-k * (tau - dur)).exp()
            }
        }
        1 => {
            // OneCptOral
            let k = cl / v;
            let d = amt;
            if (ka - k).abs() < 1e-6 {
                (d * ka / v) * tau * (-k * tau).exp()
            } else {
                (d * ka / (v * (ka - k))) * ((-k * tau).exp() - (-ka * tau).exp())
            }
        }
        2 => {
            // TwoCptIv — bolus when dur<=0, infusion otherwise.
            //
            // No `diff.abs() < 1e-12 ⇒ 0.0` guard on either branch:
            // branching on `.abs()` of a continuous argument poisons the
            // Enzyme reverse-mode adjoint, and the old `TwoCptInfusion`
            // arm explicitly removed it for that reason. For physical
            // positive (cl, v, q, v2) the 2-cpt discriminant is
            // strictly positive, so `diff = α - β = √disc > 0` and the
            // divisions never blow up in finite precision. The old
            // `TwoCptIvBolus` arm carried the guard but it was dead in
            // practice — keeping the bolus and infusion branches
            // symmetric here matches that prior author's decision.
            let (alpha, beta, k21) = macro_rates(cl, v, q, v2);
            let diff = alpha - beta;
            if dur <= 0.0 {
                let a = (amt / v) * (alpha - k21) / diff;
                let b = (amt / v) * (k21 - beta) / diff;
                a * (-alpha * tau).exp() + b * (-beta * tau).exp()
            } else {
                let a_c = (rate / v) * (alpha - k21) / (diff * alpha);
                let b_c = (rate / v) * (k21 - beta) / (diff * beta);
                if tau <= dur {
                    a_c * (1.0 - (-alpha * tau).exp()) + b_c * (1.0 - (-beta * tau).exp())
                } else {
                    let dt = tau - dur;
                    a_c * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
                        + b_c * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
                }
            }
        }
        3 => {
            // TwoCptOral
            let (alpha, beta, k21) = macro_rates(cl, v, q, v2);
            let diff = alpha - beta;
            if diff.abs() < 1e-12 {
                return 0.0;
            }
            let coeff = amt * ka / v;
            let p = if (ka - alpha).abs() < 1e-6 {
                coeff * (alpha - k21) / diff * tau * (-alpha * tau).exp()
            } else {
                coeff * (k21 - alpha) / ((ka - alpha) * (beta - alpha)) * (-alpha * tau).exp()
            };
            let q_val = if (ka - beta).abs() < 1e-6 {
                coeff * (k21 - beta) / diff * tau * (-beta * tau).exp()
            } else {
                coeff * (k21 - beta) / ((ka - beta) * (alpha - beta)) * (-beta * tau).exp()
            };
            let r = if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 {
                0.0
            } else {
                coeff * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-ka * tau).exp()
            };
            p + q_val + r
        }
        4 => {
            // ThreeCptIv — bolus when dur<=0, infusion otherwise.
            //
            // The guards differ between the two branches: the bolus
            // formula only divides by `ab·ag`, `ab·bg`, `ag·bg`, so it
            // only needs `ab/ag/bg ≈ 0` checks. The infusion formula
            // additionally divides by `α`, `β`, `γ` in the rate-input
            // coefficients, so it needs the three extra eigenvalue
            // checks. Folding all six into a shared guard (as a prior
            // revision did) collapses physically-valid bolus answers
            // to zero whenever a slowly-equilibrating 3-cpt has one of
            // α/β/γ near zero — see issue #176 review.
            let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_ad(cl, v, q, v2, q3, v3);
            let ab = alpha - beta;
            let ag = alpha - gamma;
            let bg = beta - gamma;
            if dur <= 0.0 {
                if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
                    return 0.0;
                }
                let d = amt / v;
                let a = d * (alpha - k21) * (alpha - k31) / (ab * ag);
                let b = d * (beta - k21) * (beta - k31) / (-ab * bg);
                let g = d * (gamma - k21) * (gamma - k31) / (ag * bg);
                a * (-alpha * tau).exp() + b * (-beta * tau).exp() + g * (-gamma * tau).exp()
            } else {
                if ab.abs() < 1e-12
                    || ag.abs() < 1e-12
                    || bg.abs() < 1e-12
                    || alpha.abs() < 1e-12
                    || beta.abs() < 1e-12
                    || gamma.abs() < 1e-12
                {
                    return 0.0;
                }
                let rv = rate / v;
                let a_c = rv * (alpha - k21) * (alpha - k31) / (ab * ag * alpha);
                let b_c = rv * (beta - k21) * (beta - k31) / (-ab * bg * beta);
                let g_c = rv * (gamma - k21) * (gamma - k31) / (ag * bg * gamma);
                if tau <= dur {
                    a_c * (1.0 - (-alpha * tau).exp())
                        + b_c * (1.0 - (-beta * tau).exp())
                        + g_c * (1.0 - (-gamma * tau).exp())
                } else {
                    let dt = tau - dur;
                    a_c * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
                        + b_c * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
                        + g_c * (1.0 - (-gamma * dur).exp()) * (-gamma * dt).exp()
                }
            }
        }
        5 => {
            // ThreeCptOral
            let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_ad(cl, v, q, v2, q3, v3);
            let ab = alpha - beta;
            let ag = alpha - gamma;
            let bg = beta - gamma;
            if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
                return 0.0;
            }
            let coeff = amt * ka / v;
            let a_c = (alpha - k21) * (alpha - k31) / (ab * ag);
            let b_c = (beta - k21) * (beta - k31) / (-ab * bg);
            let g_c = (gamma - k21) * (gamma - k31) / (ag * bg);

            let bateman_a = if (ka - alpha).abs() < 1e-6 {
                tau * (-alpha * tau).exp()
            } else {
                ((-alpha * tau).exp() - (-ka * tau).exp()) / (ka - alpha)
            };
            let bateman_b = if (ka - beta).abs() < 1e-6 {
                tau * (-beta * tau).exp()
            } else {
                ((-beta * tau).exp() - (-ka * tau).exp()) / (ka - beta)
            };
            let bateman_g = if (ka - gamma).abs() < 1e-6 {
                tau * (-gamma * tau).exp()
            } else {
                ((-gamma * tau).exp() - (-ka * tau).exp()) / (ka - gamma)
            };
            coeff * (a_c * bateman_a + b_c * bateman_b + g_c * bateman_g)
        }
        _ => 0.0,
    }
}

fn macro_rates_three_cpt_ad(
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, f64, f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q2 / v1;
    let k21 = q2 / v2;
    let k13 = q3 / v1;
    let k31 = q3 / v3;

    let s2 = k10 + k12 + k13 + k21 + k31;
    let s1 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
    let s0 = k10 * k21 * k31;

    let h = s2 / 3.0;
    let p = s1 - s2 * s2 / 3.0;
    let qq = s1 * s2 / 3.0 - 2.0 * s2 * s2 * s2 / 27.0 - s0;

    let p_safe = if p < -1e-30 { p } else { -1e-30 };
    let m = 2.0 * (-p_safe / 3.0).sqrt();
    let mut arg = 3.0 * qq / (p_safe * m);
    if arg < -1.0 {
        arg = -1.0;
    }
    if arg > 1.0 {
        arg = 1.0;
    }
    let phi = arg.acos() / 3.0;

    let pi_2_3 = 2.0 * std::f64::consts::FRAC_PI_3;
    let lambda0 = m * phi.cos() + h;
    let lambda1 = m * (phi - pi_2_3).cos() + h;
    let lambda2 = m * (phi - 2.0 * pi_2_3).cos() + h;

    let alpha = if lambda0 >= lambda1 && lambda0 >= lambda2 {
        lambda0
    } else if lambda1 >= lambda2 {
        lambda1
    } else {
        lambda2
    };
    let gamma = if lambda0 <= lambda1 && lambda0 <= lambda2 {
        lambda0
    } else if lambda1 <= lambda2 {
        lambda1
    } else {
        lambda2
    };
    let beta = s2 - alpha - gamma;

    (alpha, beta, gamma, k21, k31)
}

/// 2-cpt macro rate constants (α, β, k21).
///
/// For any positive k10, k12, k21 the discriminant
/// `s² − 4d = (k10 − k21)² + k12·(k12 + 2·k10 + 2·k21)` is non-negative
/// and `α = (s + √disc)/2 ≥ s/2 > 0`, so the old `if sq > 0` /
/// `if alpha > 1e-30` guards never fired for physical parameters — and
/// under Enzyme reverse-mode the phi nodes they created defeated type
/// deduction, producing NaN gradients.
///
/// Kept branch-free. To survive transient FP cancellation that makes
/// the discriminant a tiny negative (e.g. a line-search trial point
/// grazing a degenerate parameter configuration), `arg` is clamped to
/// `≥ 0` via `(arg + |arg|) / 2` — arithmetic only, no `.max()`
/// (which lowers to `llvm.maximumnum` and breaks the Enzyme compile)
/// and no `if`/`else` (which would reintroduce the phi-node pathology).
/// `.abs()` lowers to `llvm.fabs`, which Enzyme differentiates correctly.
fn macro_rates(cl: f64, v1: f64, q: f64, v2: f64) -> (f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let arg = s * s - 4.0 * d;
    let arg_clamped = (arg + arg.abs()) * 0.5;
    let disc = arg_clamped.sqrt();
    let alpha = (s + disc) / 2.0;
    let beta = d / alpha;
    (alpha, beta, k21)
}

fn residual_variance_ad(error_model_id: i32, f_pred: f64, sigma: &[f64]) -> f64 {
    let v = match error_model_id {
        0 => sigma[0] * sigma[0],
        1 => {
            let fs = f_pred * sigma[0];
            fs * fs
        }
        2 => {
            let p = f_pred * sigma[0];
            p * p + sigma[1] * sigma[1]
        }
        _ => sigma[0] * sigma[0],
    };
    if v < 1e-12 {
        1e-12
    } else {
        v
    }
}

// ─── Enum → ID converters ───────────────────────────────────────────────────
//
// `pk_model_id` is passed across the autodiff FFI boundary as `f64` (Enzyme
// cannot carry the Rust enum directly), and the dispatch chains in
// `event_driven_ad.rs` / `event_driven_ad_jac.rs` compare against literal
// numbers. Defining the IDs as named constants here means a future
// renumbering — or a variant rename like #176 — propagates to every dispatch
// site through the type system rather than silently misrouting.

pub const PK_ID_ONE_CPT_IV: i32 = 0;
pub const PK_ID_ONE_CPT_ORAL: i32 = 1;
pub const PK_ID_TWO_CPT_IV: i32 = 2;
pub const PK_ID_TWO_CPT_ORAL: i32 = 3;
pub const PK_ID_THREE_CPT_IV: i32 = 4;
pub const PK_ID_THREE_CPT_ORAL: i32 = 5;

pub fn pk_model_to_id(m: PkModel) -> i32 {
    match m {
        PkModel::OneCptIv => PK_ID_ONE_CPT_IV,
        PkModel::OneCptOral => PK_ID_ONE_CPT_ORAL,
        PkModel::TwoCptIv => PK_ID_TWO_CPT_IV,
        PkModel::TwoCptOral => PK_ID_TWO_CPT_ORAL,
        PkModel::ThreeCptIv => PK_ID_THREE_CPT_IV,
        PkModel::ThreeCptOral => PK_ID_THREE_CPT_ORAL,
    }
}

pub fn error_model_to_id(m: ErrorModel) -> i32 {
    match m {
        ErrorModel::Additive => 0,
        ErrorModel::Proportional => 1,
        ErrorModel::Combined => 2,
    }
}

// ─── Flat dose data ─────────────────────────────────────────────────────────

pub struct FlatDoseData {
    pub times: Vec<f64>,
    pub amts: Vec<f64>,
    pub rates: Vec<f64>,
    pub durations: Vec<f64>,
}

impl FlatDoseData {
    pub fn from_subject(subject: &Subject) -> Self {
        // Tripwire (#324 / #281 / #394): this single-snapshot AD path freezes
        // `d.rate`/`d.duration` into flat f64 arrays. A modeled-RATE dose
        // (RATE=-2 -> D{cmt}) must never reach it — `resolve_gradient_method`
        // routes any subject with a modeled dose to FD (the ODE engine has no AD
        // path; the analytical AD kernels can't carry `∂duration/∂η`), so the AD
        // path only ever sees `Fixed` doses. An unresolved one would snapshot a 0
        // rate/duration and yield a silently wrong gradient (#317).
        //
        // A real `assert!` (not `debug_assert!`): now that analytical models can be
        // BOTH AD-eligible AND carry a modeled dose (#383/#394), this path is
        // genuinely reachable, and `debug_assert!` is compiled out of `autodiff`
        // *release* builds — so only an `assert!` makes the FD-gate's invariant hold
        // across every build config (debug/release × FD/AD). It is the backstop to
        // the primary `resolve_gradient_method` gate, not the first line of defence.
        assert!(
            subject.all_doses_fixed(),
            "modeled-RATE dose reached the AD path"
        );
        Self {
            times: subject.doses.iter().map(|d| d.time).collect(),
            amts: subject.doses.iter().map(|d| d.amt).collect(),
            rates: subject.doses.iter().map(|d| d.rate).collect(),
            durations: subject.doses.iter().map(|d| d.duration).collect(),
        }
    }
}

// ─── Public interface ───────────────────────────────────────────────────────

/// Compute gradient of individual_nll w.r.t. eta using reverse-mode AD.
/// `tv_adjusted` = covariate-adjusted typical values, length n_tv
/// (parallel to `pk_idx_f64` and `sel_flat`'s row dimension — not n_eta;
/// one entry per `[individual_parameters]` assignment).
/// `cens_f64` = per-observation censoring flags (0, 1, or -1 as f64); pass all
/// zeros when M3 is disabled.
#[allow(clippy::too_many_arguments)]
pub fn compute_nll_gradient_ad(
    eta: &[f64],
    tv_adjusted: &[f64],
    omega_inv_flat: &[f64],
    log_det_omega: f64,
    sigma_values: &[f64],
    dose_data: &FlatDoseData,
    obs_times: &[f64],
    observations: &[f64],
    cens_f64: &[f64],
    pk_model: PkModel,
    error_model: ErrorModel,
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    obs_scale: &[f64],
    log_transform: bool,
) -> (f64, Vec<f64>) {
    let n_eta = eta.len();
    let mut d_eta = vec![0.0f64; n_eta];

    // +100 packs LTBS (see `individual_nll_ad`); under LTBS the error model is
    // additive (id 0) on the log scale.
    let ltbs_offset = if log_transform { 100 } else { 0 };
    let pk_and_err =
        (pk_model_to_id(pk_model) * 10 + error_model_to_id(error_model) + ltbs_offset) as f64;

    let nll = individual_nll_ad_grad(
        eta,
        &mut d_eta,
        tv_adjusted,
        omega_inv_flat,
        log_det_omega,
        sigma_values,
        &dose_data.times,
        &dose_data.amts,
        &dose_data.rates,
        &dose_data.durations,
        obs_times,
        observations,
        cens_f64,
        pk_idx_f64,
        sel_flat,
        pk_and_err,
        obs_scale,
        1.0,
    );

    (nll, d_eta)
}

/// Compute Jacobian d(predictions)/d(eta) using forward-mode AD.
#[allow(clippy::too_many_arguments)]
pub fn compute_jacobian_ad(
    eta: &[f64],
    tv_adjusted: &[f64],
    dose_data: &FlatDoseData,
    obs_times: &[f64],
    n_obs: usize,
    pk_model: PkModel,
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    obs_scale: &[f64],
    log_transform: bool,
) -> nalgebra::DMatrix<f64> {
    let n_eta = eta.len();
    // +100 packs LTBS so `predict_all_ad` log-wraps the forward prediction, making
    // this Jacobian d log(f)/dη — consistent with the log-scale objective.
    let ltbs_offset = if log_transform { 100 } else { 0 };
    let pk_id = (pk_model_to_id(pk_model) + ltbs_offset) as f64;
    let mut jac = nalgebra::DMatrix::zeros(n_obs, n_eta);

    for j in 0..n_eta {
        let mut d_eta = vec![0.0f64; n_eta];
        d_eta[j] = 1.0;

        let mut out = vec![0.0f64; n_obs];
        let mut d_out = vec![0.0f64; n_obs];

        predict_all_ad_tangent(
            eta,
            &d_eta,
            tv_adjusted,
            &dose_data.times,
            &dose_data.amts,
            &dose_data.rates,
            &dose_data.durations,
            obs_times,
            pk_idx_f64,
            sel_flat,
            pk_id,
            obs_scale,
            &mut out,
            &mut d_out,
        );

        for i in 0..n_obs {
            jac[(i, j)] = d_out[i];
        }
    }

    jac
}

#[cfg(test)]
mod id_tests {
    use super::*;

    /// `pk_model_to_id` and the named `PK_ID_*` constants must stay in
    /// lockstep — every dispatch chain in `event_driven_ad.rs` and
    /// `event_driven_ad_jac.rs` compares the AD `pk_model_id: f64`
    /// argument against these constants. A silent renumbering (e.g.
    /// adding a variant in the middle of the enum) would misroute
    /// every fit under the `autodiff` feature without a compile error,
    /// since the comparisons are on `f64` literals. This test catches
    /// that drift.
    #[test]
    fn pk_model_to_id_matches_named_constants() {
        assert_eq!(pk_model_to_id(PkModel::OneCptIv), PK_ID_ONE_CPT_IV);
        assert_eq!(pk_model_to_id(PkModel::OneCptOral), PK_ID_ONE_CPT_ORAL);
        assert_eq!(pk_model_to_id(PkModel::TwoCptIv), PK_ID_TWO_CPT_IV);
        assert_eq!(pk_model_to_id(PkModel::TwoCptOral), PK_ID_TWO_CPT_ORAL);
        assert_eq!(pk_model_to_id(PkModel::ThreeCptIv), PK_ID_THREE_CPT_IV);
        assert_eq!(pk_model_to_id(PkModel::ThreeCptOral), PK_ID_THREE_CPT_ORAL);
    }

    /// The LTBS packing `pk_id * 10 + err_id + 100` must not collide
    /// with the +100 LTBS-offset boundary. With six PK models (max id
    /// 5) and three error models (max id 2), the base range is [0, 52]
    /// and the offset-equipped range is [100, 152] — unambiguous.
    /// Verify the bound is actually met.
    #[test]
    fn ltbs_packing_does_not_collide_with_offset_boundary() {
        let max_pk = pk_model_to_id(PkModel::ThreeCptOral);
        let max_err = error_model_to_id(ErrorModel::Combined);
        let max_base = max_pk * 10 + max_err;
        assert!(
            max_base < 100,
            "pk_model_id * 10 + error_model_id overflows the +100 LTBS offset; \
             got {max_base} from pk={max_pk}, err={max_err}"
        );
    }
}

#[cfg(test)]
mod ltbs_ad_tests {
    use super::*;
    use crate::types::{ErrorModel, PkModel};

    /// The reverse-mode LTBS gradient (with the +100 log-wrap encoding) must
    /// match a central difference of the LTBS NLL — i.e. the analytic
    /// d/dη[ (log DV − log f)² / σ² ] is correct through the `conc.ln()` step.
    #[test]
    fn ltbs_ad_gradient_matches_central_difference() {
        // 1-cpt IV bolus, single eta on CL, additive error on the log scale.
        let eta = vec![0.1];
        let tv = vec![1.0, 10.0]; // CL_typical, V
        let pk_idx_f64 = vec![0.0, 1.0]; // PK_IDX_CL = 0, PK_IDX_V = 1
        let sel_flat = vec![1.0, 0.0]; // eta applies to CL only (2 tv × 1 eta)
        let omega = 0.09_f64;
        let omega_inv_flat = vec![1.0 / omega];
        let log_det = omega.ln();
        let sigma = vec![0.3];
        let dose = FlatDoseData {
            times: vec![0.0],
            amts: vec![100.0],
            rates: vec![0.0],
            durations: vec![0.0],
        };
        let obs_times = vec![1.0, 3.0, 6.0];
        let observations = vec![2.0, 1.0, 0.0]; // log-scale DV
        let cens = vec![0.0, 0.0, 0.0];
        let obs_scale = vec![1.0, 1.0, 1.0];

        let (_, grad) = compute_nll_gradient_ad(
            &eta,
            &tv,
            &omega_inv_flat,
            log_det,
            &sigma,
            &dose,
            &obs_times,
            &observations,
            &cens,
            PkModel::OneCptIv,
            ErrorModel::Additive,
            &pk_idx_f64,
            &sel_flat,
            &obs_scale,
            true, // LTBS
        );

        // Central FD of the LTBS NLL (pk_and_err = 100 ⇒ LTBS + additive id 0).
        let nll = |e: &[f64]| {
            individual_nll_ad(
                e,
                &tv,
                &omega_inv_flat,
                log_det,
                &sigma,
                &dose.times,
                &dose.amts,
                &dose.rates,
                &dose.durations,
                &obs_times,
                &observations,
                &cens,
                &pk_idx_f64,
                &sel_flat,
                100.0,
                &obs_scale,
            )
        };
        let h = 1e-6;
        let mut ep = eta.clone();
        ep[0] += h;
        let mut em = eta.clone();
        em[0] -= h;
        let fd = (nll(&ep) - nll(&em)) / (2.0 * h);

        approx::assert_relative_eq!(grad[0], fd, epsilon = 1e-5, max_relative = 1e-4);
    }
}
