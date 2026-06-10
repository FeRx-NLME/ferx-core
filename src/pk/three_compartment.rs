use crate::types::DoseEvent;

/// Compute macro-rate constants (alpha, beta, gamma) from micro-constants
/// for a three-compartment model using the trigonometric (Vieta) method.
///
/// Returns (alpha, beta, gamma, k21, k31) where alpha > beta > gamma > 0.
fn macro_rates_three_cpt(
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

    // Symmetric functions of the roots (Vieta's formulas)
    let s2 = k10 + k12 + k13 + k21 + k31;
    let s1 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
    let s0 = k10 * k21 * k31;

    // Depress the cubic: lambda = x + s2/3
    let h = s2 / 3.0;
    let p = s1 - s2 * s2 / 3.0;
    let q = s1 * s2 / 3.0 - 2.0 * s2 * s2 * s2 / 27.0 - s0;

    // Trigonometric solution (three distinct real roots guaranteed for valid PK params)
    let p_safe = p.min(-1e-30); // guard against p=0
    let m = 2.0 * (-p_safe / 3.0).sqrt();
    let arg = (3.0 * q / (p_safe * m)).clamp(-1.0, 1.0);
    let phi = arg.acos() / 3.0;

    let pi_2_3 = 2.0 * std::f64::consts::FRAC_PI_3;
    let lambda0 = m * phi.cos() + h;
    let lambda1 = m * (phi - pi_2_3).cos() + h;
    let lambda2 = m * (phi - 2.0 * pi_2_3).cos() + h;

    // Sort: alpha > beta > gamma
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

/// Three-compartment IV bolus
/// C(t) = A*exp(-alpha*t) + B*exp(-beta*t) + G*exp(-gamma*t)
pub fn three_cpt_iv_bolus(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return 0.0;
    }

    let d = dose.amt / v1;
    let a = d * (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = d * (beta - k21) * (beta - k31) / (-ab * bg);
    let g = d * (gamma - k21) * (gamma - k31) / (ag * bg);

    a * (-alpha * t).exp() + b * (-beta * t).exp() + g * (-gamma * t).exp()
}

/// Three-compartment constant-rate IV infusion
pub fn three_cpt_infusion(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12
        || ag.abs() < 1e-12
        || bg.abs() < 1e-12
        || alpha.abs() < 1e-12
        || beta.abs() < 1e-12
        || gamma.abs() < 1e-12
    {
        return 0.0;
    }

    let rate = dose.rate;
    let dur = dose.duration;
    if dur <= 0.0 {
        return three_cpt_iv_bolus(dose, t, cl, v1, q2, v2, q3, v3);
    }

    let rv = rate / v1;
    let a_coeff = rv * (alpha - k21) * (alpha - k31) / (ab * ag * alpha);
    let b_coeff = rv * (beta - k21) * (beta - k31) / (-ab * bg * beta);
    let g_coeff = rv * (gamma - k21) * (gamma - k31) / (ag * bg * gamma);

    if t <= dur {
        a_coeff * (1.0 - (-alpha * t).exp())
            + b_coeff * (1.0 - (-beta * t).exp())
            + g_coeff * (1.0 - (-gamma * t).exp())
    } else {
        let dt = t - dur;
        a_coeff * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
            + b_coeff * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
            + g_coeff * (1.0 - (-gamma * dur).exp()) * (-gamma * dt).exp()
    }
}

/// Three-compartment oral absorption with bioavailability
pub fn three_cpt_oral_f(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ka <= 0.0
    {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return 0.0;
    }

    let coeff = f_bio * dose.amt * ka / v1;

    // Normalized residue coefficients
    let a = (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = (beta - k21) * (beta - k31) / (-ab * bg);
    let c = (gamma - k21) * (gamma - k31) / (ag * bg);

    // Bateman function with L'Hôpital singularity handling
    let bateman = |lambda: f64| -> f64 {
        if (ka - lambda).abs() < 1e-6 {
            t * (-lambda * t).exp()
        } else {
            ((-lambda * t).exp() - (-ka * t).exp()) / (ka - lambda)
        }
    };

    coeff * (a * bateman(alpha) + b * bateman(beta) + c * bateman(gamma))
}

/// Three-compartment oral absorption (F=1.0 default)
pub fn three_cpt_oral(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
) -> f64 {
    three_cpt_oral_f(dose, t, cl, v1, q2, v2, q3, v3, ka, 1.0)
}

// --- Steady-state (SS=1) variants ---
//
// Same geometric-series pattern as 1-/2-cpt: each exponential A·exp(-λ·t) gets
// scaled by 1/(1 - exp(-λ·II)). For oral, the per-eigenvalue Bateman function
// is summed in closed form, including a L'Hopital limit when ka ≈ λ.

#[inline]
fn ss_coeff_3(lambda: f64, ii: f64) -> f64 {
    let denom = 1.0 - (-lambda * ii).exp();
    if denom > 0.0 {
        1.0 / denom
    } else {
        0.0
    }
}

/// Three-compartment IV bolus at steady state.
pub fn three_cpt_iv_bolus_ss(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || dose.ii <= 0.0
    {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return 0.0;
    }
    let ii = dose.ii;
    let d = dose.amt / v1;
    let a = d * (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = d * (beta - k21) * (beta - k31) / (-ab * bg);
    let g = d * (gamma - k21) * (gamma - k31) / (ag * bg);
    a * (-alpha * t).exp() * ss_coeff_3(alpha, ii)
        + b * (-beta * t).exp() * ss_coeff_3(beta, ii)
        + g * (-gamma * t).exp() * ss_coeff_3(gamma, ii)
}

/// Three-compartment infusion at steady state.
///
/// Closed form requires `T_inf ≤ II` (non-overlapping infusions); returns 0.0
/// otherwise (api.rs warns).
pub fn three_cpt_infusion_ss(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || dose.ii <= 0.0
    {
        return 0.0;
    }
    let dur = dose.duration;
    if dur <= 0.0 {
        return three_cpt_iv_bolus_ss(dose, t, cl, v1, q2, v2, q3, v3);
    }
    if dur > dose.ii {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12
        || ag.abs() < 1e-12
        || bg.abs() < 1e-12
        || alpha.abs() < 1e-12
        || beta.abs() < 1e-12
        || gamma.abs() < 1e-12
    {
        return 0.0;
    }
    let ii = dose.ii;
    let rate = dose.rate;
    let rv = rate / v1;
    let a_coeff = rv * (alpha - k21) * (alpha - k31) / (ab * ag * alpha);
    let b_coeff = rv * (beta - k21) * (beta - k31) / (-ab * bg * beta);
    let g_coeff = rv * (gamma - k21) * (gamma - k31) / (ag * bg * gamma);

    // Helper: contribution from past pulses (n ≥ 1) for one eigenvalue —
    // always "after-infusion" because τ + n·II ≥ II ≥ T_inf.
    let past = |coeff: f64, lambda: f64| -> f64 {
        coeff
            * (1.0 - (-lambda * dur).exp())
            * (-lambda * (t - dur)).exp()
            * (-lambda * ii).exp()
            * ss_coeff_3(lambda, ii)
    };

    if t <= dur {
        // n=0 is during the current infusion.
        a_coeff * (1.0 - (-alpha * t).exp())
            + b_coeff * (1.0 - (-beta * t).exp())
            + g_coeff * (1.0 - (-gamma * t).exp())
            + past(a_coeff, alpha)
            + past(b_coeff, beta)
            + past(g_coeff, gamma)
    } else {
        // All pulses are "after-infusion" — fold n=0 in by replacing the
        // tail's (n ≥ 1) sum with (n ≥ 0).
        let dt = t - dur;
        a_coeff * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp() * ss_coeff_3(alpha, ii)
            + b_coeff * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp() * ss_coeff_3(beta, ii)
            + g_coeff * (1.0 - (-gamma * dur).exp()) * (-gamma * dt).exp() * ss_coeff_3(gamma, ii)
    }
}

/// Three-compartment oral absorption at steady state (with bioavailability).
pub fn three_cpt_oral_f_ss(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ka <= 0.0
        || dose.ii <= 0.0
    {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return 0.0;
    }
    let ii = dose.ii;
    let coeff = f_bio * dose.amt * ka / v1;
    let a = (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = (beta - k21) * (beta - k31) / (-ab * bg);
    let c = (gamma - k21) * (gamma - k31) / (ag * bg);

    // SS bateman per eigenvalue λ:
    //   Σ_n [exp(-λ·(τ + n·II)) - exp(-ka·(τ + n·II))] / (ka - λ)
    //     = [exp(-λ·τ)/(1-exp(-λ·II)) - exp(-ka·τ)/(1-exp(-ka·II))] / (ka - λ)
    // L'Hopital limit when ka ≈ λ:
    //   Σ_n (τ + n·II) · exp(-λ·(τ + n·II))
    //     = exp(-λ·τ) · [τ/(1-x) + II·x/(1-x)²]  with x = exp(-λ·ii).
    let bateman_ss = |lambda: f64| -> f64 {
        if (ka - lambda).abs() < 1e-6 {
            let x = (-lambda * ii).exp();
            let one_minus_x = 1.0 - x;
            if one_minus_x <= 0.0 {
                return 0.0;
            }
            (-lambda * t).exp() * (t / one_minus_x + ii * x / (one_minus_x * one_minus_x))
        } else {
            ((-lambda * t).exp() * ss_coeff_3(lambda, ii) - (-ka * t).exp() * ss_coeff_3(ka, ii))
                / (ka - lambda)
        }
    };

    coeff * (a * bateman_ss(alpha) + b * bateman_ss(beta) + c * bateman_ss(gamma))
}

/// Three-compartment oral absorption at steady state (F = 1).
pub fn three_cpt_oral_ss(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
) -> f64 {
    three_cpt_oral_f_ss(dose, t, cl, v1, q2, v2, q3, v3, ka, 1.0)
}

// --- Peripheral compartment concentrations (for compartment_states) ---

/// Returns [C_periph1, C_periph2] for a 3-cpt IV single dose at elapsed time tau.
///
/// Peripheral 1 (V2): C2 = (q2/(v1·v2)) · dose.amt ·
///   [ −(α−k31)/(ab·ag)·e^{-αt} + (β−k31)/(ab·bg)·e^{-βt} − (γ−k31)/(ag·bg)·e^{-γt} ]
/// Peripheral 2 (V3): swap k31↔k21 and q2/v2 → q3/v3.
pub(crate) fn three_cpt_iv_peripherals(
    dose: &DoseEvent,
    tau: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> [f64; 2] {
    if tau < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return [0.0; 2];
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return [0.0; 2];
    }

    let c2_scalar = q2 / (v1 * v2); // k12/v2
    let c3_scalar = q3 / (v1 * v3); // k13/v3
    let d = dose.amt;

    if dose.ss && dose.ii > 0.0 {
        let ii = dose.ii;
        if dose.is_infusion() {
            let dur = dose.duration;
            if dur <= 0.0 {
                // Treat as bolus SS
                let c2 = c2_scalar
                    * d
                    * (-(alpha - k31) / (ab * ag) * (-alpha * tau).exp() * ss_coeff_3(alpha, ii)
                        + (beta - k31) / (ab * bg) * (-beta * tau).exp() * ss_coeff_3(beta, ii)
                        - (gamma - k31) / (ag * bg) * (-gamma * tau).exp() * ss_coeff_3(gamma, ii));
                let c3 = c3_scalar
                    * d
                    * (-(alpha - k21) / (ab * ag) * (-alpha * tau).exp() * ss_coeff_3(alpha, ii)
                        + (beta - k21) / (ab * bg) * (-beta * tau).exp() * ss_coeff_3(beta, ii)
                        - (gamma - k21) / (ag * bg) * (-gamma * tau).exp() * ss_coeff_3(gamma, ii));
                return [c2, c3];
            }
            if dur > dose.ii {
                return [0.0; 2];
            }
            // Helper for infusion SS peripheral (after-infusion formula with SS coeff)
            let infusion_periph_ss = |c_scalar: f64, k_far: f64| -> f64 {
                let r = dose.rate;
                // After-infusion peripheral formula with SS geometric series:
                // coeff_X = rate · c_scalar · (X_eigenvalue_residue) / (product · eigenvalue)
                let coeff_a = -r * c_scalar * (alpha - k_far) / (ab * ag * alpha);
                let coeff_b = r * c_scalar * (beta - k_far) / (ab * bg * beta);
                let coeff_g = -r * c_scalar * (gamma - k_far) / (ag * bg * gamma);
                if tau <= dur {
                    let cur = coeff_a * (1.0 - (-alpha * tau).exp())
                        + coeff_b * (1.0 - (-beta * tau).exp())
                        + coeff_g * (1.0 - (-gamma * tau).exp());
                    let past = coeff_a
                        * (1.0 - (-alpha * dur).exp())
                        * (-alpha * (tau - dur)).exp()
                        * (-alpha * dose.ii).exp()
                        * ss_coeff_3(alpha, ii)
                        + coeff_b
                            * (1.0 - (-beta * dur).exp())
                            * (-beta * (tau - dur)).exp()
                            * (-beta * dose.ii).exp()
                            * ss_coeff_3(beta, ii)
                        + coeff_g
                            * (1.0 - (-gamma * dur).exp())
                            * (-gamma * (tau - dur)).exp()
                            * (-gamma * dose.ii).exp()
                            * ss_coeff_3(gamma, ii);
                    cur + past
                } else {
                    let dt = tau - dur;
                    coeff_a
                        * (1.0 - (-alpha * dur).exp())
                        * (-alpha * dt).exp()
                        * ss_coeff_3(alpha, ii)
                        + coeff_b
                            * (1.0 - (-beta * dur).exp())
                            * (-beta * dt).exp()
                            * ss_coeff_3(beta, ii)
                        + coeff_g
                            * (1.0 - (-gamma * dur).exp())
                            * (-gamma * dt).exp()
                            * ss_coeff_3(gamma, ii)
                }
            };
            [
                infusion_periph_ss(c2_scalar, k31),
                infusion_periph_ss(c3_scalar, k21),
            ]
        } else {
            // Bolus SS
            let c2 = c2_scalar
                * d
                * (-(alpha - k31) / (ab * ag) * (-alpha * tau).exp() * ss_coeff_3(alpha, ii)
                    + (beta - k31) / (ab * bg) * (-beta * tau).exp() * ss_coeff_3(beta, ii)
                    - (gamma - k31) / (ag * bg) * (-gamma * tau).exp() * ss_coeff_3(gamma, ii));
            let c3 = c3_scalar
                * d
                * (-(alpha - k21) / (ab * ag) * (-alpha * tau).exp() * ss_coeff_3(alpha, ii)
                    + (beta - k21) / (ab * bg) * (-beta * tau).exp() * ss_coeff_3(beta, ii)
                    - (gamma - k21) / (ag * bg) * (-gamma * tau).exp() * ss_coeff_3(gamma, ii));
            [c2, c3]
        }
    } else if dose.is_infusion() {
        let dur = dose.duration;
        if dur <= 0.0 {
            let c2 = c2_scalar
                * d
                * (-(alpha - k31) / (ab * ag) * (-alpha * tau).exp()
                    + (beta - k31) / (ab * bg) * (-beta * tau).exp()
                    - (gamma - k31) / (ag * bg) * (-gamma * tau).exp());
            let c3 = c3_scalar
                * d
                * (-(alpha - k21) / (ab * ag) * (-alpha * tau).exp()
                    + (beta - k21) / (ab * bg) * (-beta * tau).exp()
                    - (gamma - k21) / (ag * bg) * (-gamma * tau).exp());
            return [c2, c3];
        }
        let infusion_periph = |c_scalar: f64, k_far: f64| -> f64 {
            let r = dose.rate;
            let coeff_a = -r * c_scalar * (alpha - k_far) / (ab * ag * alpha);
            let coeff_b = r * c_scalar * (beta - k_far) / (ab * bg * beta);
            let coeff_g = -r * c_scalar * (gamma - k_far) / (ag * bg * gamma);
            if tau <= dur {
                coeff_a * (1.0 - (-alpha * tau).exp())
                    + coeff_b * (1.0 - (-beta * tau).exp())
                    + coeff_g * (1.0 - (-gamma * tau).exp())
            } else {
                let dt = tau - dur;
                coeff_a * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
                    + coeff_b * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
                    + coeff_g * (1.0 - (-gamma * dur).exp()) * (-gamma * dt).exp()
            }
        };
        [
            infusion_periph(c2_scalar, k31),
            infusion_periph(c3_scalar, k21),
        ]
    } else {
        // Single-dose bolus
        let c2 = c2_scalar
            * d
            * (-(alpha - k31) / (ab * ag) * (-alpha * tau).exp()
                + (beta - k31) / (ab * bg) * (-beta * tau).exp()
                - (gamma - k31) / (ag * bg) * (-gamma * tau).exp());
        let c3 = c3_scalar
            * d
            * (-(alpha - k21) / (ab * ag) * (-alpha * tau).exp()
                + (beta - k21) / (ab * bg) * (-beta * tau).exp()
                - (gamma - k21) / (ag * bg) * (-gamma * tau).exp());
        [c2, c3]
    }
}

/// Returns [C_periph1, C_periph2] for a 3-cpt oral single dose at elapsed time tau.
///
/// C2(t) = D·(q2/v2) · [ −(α−k31)/((ka−α)·ab·ag) · e^(−ατ)
///                        + (β−k31)/((ka−β)·ab·bg) · e^(−βτ)
///                        − (γ−k31)/((ka−γ)·ag·bg) · e^(−γτ)
///                        + (k31−ka)/((α−ka)(β−ka)(γ−ka)) · e^(−kaτ) ]
/// and C3 likewise with k21 substituted for k31 and (q3/v3) for (q2/v2).
/// Each exp term gains a 1/(1−e^(−λ·II)) factor at steady state.
/// Returns 0 when ka ≈ any eigenvalue (L'Hôpital singularity, see body).
pub(crate) fn three_cpt_oral_peripherals(
    dose: &DoseEvent,
    tau: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
    f_bio: f64,
) -> [f64; 2] {
    // This function implements the bolus-only oral formula (depot → central → peripherals).
    // Infusion doses bypass the depot compartment entirely and must be routed through
    // `three_cpt_infusion`/`three_cpt_iv_peripherals` by the caller — see `three_cpt_predict`.
    debug_assert!(
        !dose.is_infusion(),
        "three_cpt_oral_peripherals called with an infusion dose — route infusions through three_cpt_iv_peripherals"
    );
    if tau < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ka <= 0.0
    {
        return [0.0; 2];
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return [0.0; 2];
    }

    let d = f_bio * dose.amt * ka / v1;
    let q2_over_v2 = q2 / v2;
    let q3_over_v3 = q3 / v3;

    // When ka ≈ any eigenvalue the (ka−λ) residue denominators below are
    // singular (a genuine L'Hôpital case). The finite limit is algebraically
    // complex; we conservatively zero the *entire* peripheral result. This
    // matches the 2-cpt oral peripheral and the edge case (ka coinciding with
    // a 3-cpt eigenvalue) is rare in real PK data. The central concentration is
    // unaffected (handled separately in the central formula).
    if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 || (ka - gamma).abs() < 1e-6 {
        return [0.0, 0.0];
    }

    // Per-exponential term, with steady-state geometric accumulation when SS.
    // The single-dose solution is a sum of exp(−λτ) terms (λ ∈ {α,β,γ,ka}); SS
    // superposition applies the 1/(1−exp(−λ·II)) factor to each independently.
    let term = |lambda: f64| -> f64 {
        let base = (-lambda * tau).exp();
        if dose.ss && dose.ii > 0.0 {
            base * ss_coeff_3(lambda, dose.ii)
        } else {
            base
        }
    };

    // Closed form (issue #205, NONMEM-validated). In Laplace space the central
    // *amount* after oral input is
    //   A1(s) = F·D·ka·(s+k21)(s+k31) / [(s+ka)(s+α)(s+β)(s+γ)],
    // so the peripheral amounts are
    //   A2(s) = k12·A1(s)/(s+k21) = F·D·ka·k12·(s+k31) / [(s+ka)(s+α)(s+β)(s+γ)]
    //   A3(s) = k13·A1(s)/(s+k31) = F·D·ka·k13·(s+k21) / [(s+ka)(s+α)(s+β)(s+γ)]
    // — the coupling pole cancels, leaving four poles. Inverting and dividing
    // by the peripheral volume gives a residue on each exp(−λτ), λ ∈ {α,β,γ,ka}.
    // (The prior implementation multiplied each eigenvalue residue by a Bateman
    // helper, introducing a spurious extra 1/(ka−λ) and a double-counted
    // exp(−ka·τ) pole — it produced wrong, often negative, peripheral states.)
    let c2 = q2_over_v2
        * d
        * (-(alpha - k31) / ((ka - alpha) * ab * ag) * term(alpha)
            + (beta - k31) / ((ka - beta) * ab * bg) * term(beta)
            - (gamma - k31) / ((ka - gamma) * ag * bg) * term(gamma)
            + (k31 - ka) / ((alpha - ka) * (beta - ka) * (gamma - ka)) * term(ka));

    let c3 = q3_over_v3
        * d
        * (-(alpha - k21) / ((ka - alpha) * ab * ag) * term(alpha)
            + (beta - k21) / ((ka - beta) * ab * bg) * term(beta)
            - (gamma - k21) / ((ka - gamma) * ag * bg) * term(gamma)
            + (k21 - ka) / ((alpha - ka) * (beta - ka) * (gamma - ka)) * term(ka));

    [c2, c3]
}

/// Predict concentration from a single dose at elapsed time t using 3-cmt model.
pub fn three_cpt_predict(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: Option<f64>,
    f_bio: Option<f64>,
) -> f64 {
    if dose.is_infusion() {
        three_cpt_infusion(dose, t, cl, v1, q2, v2, q3, v3)
    } else if let Some(ka_val) = ka {
        three_cpt_oral_f(
            dose,
            t,
            cl,
            v1,
            q2,
            v2,
            q3,
            v3,
            ka_val,
            f_bio.unwrap_or(1.0),
        )
    } else {
        three_cpt_iv_bolus(dose, t, cl, v1, q2, v2, q3, v3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn bolus_dose(amt: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)
    }

    fn infusion_dose(amt: f64, rate: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
    }

    /// Regression guard for the `three_cpt_oral_peripherals` fix (issue #205).
    /// Reference peripheral *concentrations* are NONMEM 7.5.1 ADVAN12 TRANS4
    /// amounts `A(3)`/`A(4)` divided by the peripheral volumes (V2=20, V3=30);
    /// see `tests/nonmem/oral3.ctl`. Dose 100 to depot, CL=5 V1=10 Q2=2 V2=20
    /// Q3=1.5 V3=30 KA=1. The previous (Bateman-helper) implementation returned
    /// negative, ~30× too large values here.
    #[test]
    fn oral_peripherals_match_nonmem() {
        let dose = bolus_dose(100.0);
        // (tau, periph1_conc = A(3)/20, periph2_conc = A(4)/30)
        let refs = [
            (0.5, 1.8163070319 / 20.0, 1.3744591951 / 30.0),
            (1.0, 5.3388312327 / 20.0, 4.0814462953 / 30.0),
            (4.0, 17.378583631 / 20.0, 14.498493707 / 30.0),
            (12.0, 12.175881952 / 20.0, 13.856525401 / 30.0),
        ];
        for (tau, p1_ref, p2_ref) in refs {
            let [p1, p2] =
                three_cpt_oral_peripherals(&dose, tau, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0, 1.0);
            assert_relative_eq!(p1, p1_ref, max_relative = 1e-4);
            assert_relative_eq!(p2, p2_ref, max_relative = 1e-4);
        }
    }

    // Typical 3-cpt PK parameters
    const CL: f64 = 5.0;
    const V1: f64 = 10.0;
    const Q2: f64 = 2.0;
    const V2: f64 = 20.0;
    const Q3: f64 = 1.5;
    const V3: f64 = 30.0;
    const KA: f64 = 1.5;

    // --- Macro rates ---

    #[test]
    fn test_macro_rates_positive_sorted() {
        let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert!(alpha > beta, "alpha={} should be > beta={}", alpha, beta);
        assert!(beta > gamma, "beta={} should be > gamma={}", beta, gamma);
        assert!(gamma > 0.0, "gamma={} should be > 0", gamma);
        assert!(k21 > 0.0);
        assert!(k31 > 0.0);
    }

    #[test]
    fn test_macro_rates_vieta_product() {
        // alpha * beta * gamma = S0 = k10 * k21 * k31
        let k10 = CL / V1;
        let k21 = Q2 / V2;
        let k31 = Q3 / V3;
        let (alpha, beta, gamma, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(alpha * beta * gamma, k10 * k21 * k31, epsilon = 1e-10);
    }

    #[test]
    fn test_macro_rates_vieta_sum() {
        // alpha + beta + gamma = S2
        let k10 = CL / V1;
        let k12 = Q2 / V1;
        let k21 = Q2 / V2;
        let k13 = Q3 / V1;
        let k31 = Q3 / V3;
        let s2 = k10 + k12 + k13 + k21 + k31;
        let (alpha, beta, gamma, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(alpha + beta + gamma, s2, epsilon = 1e-10);
    }

    #[test]
    fn test_macro_rates_values_vs_julia() {
        let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(alpha, 0.8844821357511055, epsilon = 1e-10);
        assert_relative_eq!(beta, 0.08033284468795598, epsilon = 1e-10);
        assert_relative_eq!(gamma, 0.03518501956093856, epsilon = 1e-10);
        assert_relative_eq!(k21, 0.1, epsilon = 1e-10);
        assert_relative_eq!(k31, 0.05, epsilon = 1e-10);
    }

    // --- IV Bolus ---

    #[test]
    fn test_iv_bolus_at_time_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_iv_bolus(&dose, 0.0, CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(c, 100.0 / V1, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_bolus_vs_julia() {
        let dose = bolus_dose(100.0);
        let cases = [
            (0.0, 10.0),
            (0.5, 6.563342046483885),
            (1.0, 4.351421728970351),
            (3.0, 1.029346912256041),
            (10.0, 0.25111455768517715),
            (50.0, 0.04607673790588789),
        ];
        for (t, expected) in cases {
            let c = three_cpt_iv_bolus(&dose, t, CL, V1, Q2, V2, Q3, V3);
            assert_relative_eq!(c, expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_iv_bolus_approaches_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_iv_bolus(&dose, 10000.0, CL, V1, Q2, V2, Q3, V3);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_iv_bolus_guard_clauses() {
        let dose = bolus_dose(100.0);
        assert_eq!(three_cpt_iv_bolus(&dose, -1.0, CL, V1, Q2, V2, Q3, V3), 0.0);
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, CL, 0.0, Q2, V2, Q3, V3), 0.0);
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, 0.0, V1, Q2, V2, Q3, V3), 0.0);
        // v2=0 and v3=0 must not produce NaN
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, CL, V1, Q2, 0.0, Q3, V3), 0.0);
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, CL, V1, Q2, V2, Q3, 0.0), 0.0);
    }

    // --- Infusion ---

    #[test]
    fn test_infusion_vs_julia() {
        let dose = infusion_dose(100.0, 50.0); // dur=2h
        let cases = [
            (0.5, 2.0389498366437318),
            (1.0, 3.383071924491621),
            (2.0, 4.888266633245966),
            (3.0, 2.229504095597091),
            (10.0, 0.26611910032293007),
        ];
        for (t, expected) in cases {
            let c = three_cpt_infusion(&dose, t, CL, V1, Q2, V2, Q3, V3);
            assert_relative_eq!(c, expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_infusion_continuity_at_end() {
        let dose = infusion_dose(100.0, 50.0); // dur=2h
        let dur = 2.0;
        let c_at = three_cpt_infusion(&dose, dur, CL, V1, Q2, V2, Q3, V3);
        let c_after = three_cpt_infusion(&dose, dur + 1e-10, CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-5);
    }

    #[test]
    fn test_infusion_after_decays() {
        let dose = infusion_dose(100.0, 50.0);
        let c1 = three_cpt_infusion(&dose, 50.0, CL, V1, Q2, V2, Q3, V3);
        let c2 = three_cpt_infusion(&dose, 100.0, CL, V1, Q2, V2, Q3, V3);
        assert!(c2 < c1);
    }

    // --- Oral ---

    #[test]
    fn test_oral_at_time_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_oral(&dose, 0.0, CL, V1, Q2, V2, Q3, V3, KA);
        assert_relative_eq!(c, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_vs_julia() {
        let dose = bolus_dose(100.0);
        let cases = [
            (0.0, 0.0),
            (0.5, 4.191965255586688),
            (1.0, 4.745318116730585),
            (3.0, 1.7475777818041354),
            (10.0, 0.2614874522040594),
        ];
        for (t, expected) in cases {
            let c = three_cpt_oral(&dose, t, CL, V1, Q2, V2, Q3, V3, KA);
            assert_relative_eq!(c, expected, epsilon = 1e-8, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_oral_approaches_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_oral(&dose, 10000.0, CL, V1, Q2, V2, Q3, V3, KA);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_oral_bioavailability_scaling() {
        let dose = bolus_dose(100.0);
        let c_full = three_cpt_oral_f(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, KA, 1.0);
        let c_half = three_cpt_oral_f(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, KA, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }

    // --- Predict dispatcher ---

    #[test]
    fn test_predict_routes_iv_bolus() {
        let dose = bolus_dose(100.0);
        let direct = three_cpt_iv_bolus(&dose, 2.0, CL, V1, Q2, V2, Q3, V3);
        let via_predict = three_cpt_predict(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_oral() {
        let dose = bolus_dose(100.0);
        let direct = three_cpt_oral(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, KA);
        let via_predict = three_cpt_predict(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, Some(KA), None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_infusion() {
        let dose = infusion_dose(100.0, 50.0);
        let direct = three_cpt_infusion(&dose, 2.0, CL, V1, Q2, V2, Q3, V3);
        let via_predict = three_cpt_predict(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    // --- Steady-state variants ---
    //
    // 3-cpt SS adds a third eigenvalue γ. The slowest tail (γ ≈ 0.035 for the
    // default test PK params) takes the most numerical-sum terms to converge,
    // so N=400 keeps the truncation tail below the test tolerances.

    fn ss_bolus_dose(amt: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, true, ii)
    }

    fn ss_infusion_dose(amt: f64, rate: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, true, ii)
    }

    fn ss_numerical_sum<F: Fn(f64) -> f64>(t: f64, ii: f64, c_single: F) -> f64 {
        const N: usize = 400;
        (0..N).map(|n| c_single(t + (n as f64) * ii)).sum()
    }

    #[test]
    fn test_ss_iv_bolus_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(100.0, ii);
        let single = bolus_dose(100.0);
        for &t in &[0.0, 0.5, 3.0, 10.0, 23.9, 24.0, 48.0] {
            let cf = three_cpt_iv_bolus_ss(&dose, t, CL, V1, Q2, V2, Q3, V3);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_iv_bolus(&single, tt, CL, V1, Q2, V2, Q3, V3)
            });
            assert_relative_eq!(cf, num, epsilon = 1e-7, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_ss_oral_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(100.0, ii);
        let single = bolus_dose(100.0);
        for &t in &[0.0, 0.5, 2.0, 5.0, 12.0, 23.0, 48.0] {
            let cf = three_cpt_oral_ss(&dose, t, CL, V1, Q2, V2, Q3, V3, KA);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_oral(&single, tt, CL, V1, Q2, V2, Q3, V3, KA)
            });
            assert_relative_eq!(cf, num, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_ss_oral_lhopital_ka_near_alpha_matches_numerical_sum() {
        let (alpha, _, _, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        let ka = alpha;
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(100.0, ii);
        let single = bolus_dose(100.0);
        for &t in &[0.5, 2.0, 5.0, 12.0, 23.0] {
            let cf = three_cpt_oral_ss(&dose, t, CL, V1, Q2, V2, Q3, V3, ka);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_oral(&single, tt, CL, V1, Q2, V2, Q3, V3, ka)
            });
            assert_relative_eq!(cf, num, epsilon = 1e-5, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_ss_oral_lhopital_ka_near_beta_matches_numerical_sum() {
        // The L'Hopital branch in `bateman_ss` activates per-eigenvalue —
        // verify the β branch independently of α (Copilot review on #77).
        let (_, beta, _, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        let ka = beta;
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(100.0, ii);
        let single = bolus_dose(100.0);
        for &t in &[0.5, 2.0, 12.0, 23.0] {
            let cf = three_cpt_oral_ss(&dose, t, CL, V1, Q2, V2, Q3, V3, ka);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_oral(&single, tt, CL, V1, Q2, V2, Q3, V3, ka)
            });
            assert_relative_eq!(cf, num, epsilon = 1e-5, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_ss_oral_lhopital_ka_near_gamma_matches_numerical_sum() {
        // γ-branch coverage (Copilot review on #77). γ is the slowest
        // disposition rate; with ka = γ the SS series tail is widest, so
        // the numerical-sum oracle uses more terms via the default N=400.
        let (_, _, gamma, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        let ka = gamma;
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(100.0, ii);
        let single = bolus_dose(100.0);
        for &t in &[0.5, 2.0, 12.0, 23.0] {
            let cf = three_cpt_oral_ss(&dose, t, CL, V1, Q2, V2, Q3, V3, ka);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_oral(&single, tt, CL, V1, Q2, V2, Q3, V3, ka)
            });
            // γ-branch: looser tolerance because the slow-tail truncation
            // in the 400-term numerical sum is the dominant residual.
            assert_relative_eq!(cf, num, epsilon = 1e-4, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_ss_infusion_during_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let rate = 50.0; // amt=100, duration=2
        let dose = ss_infusion_dose(100.0, rate, ii);
        let single = infusion_dose(100.0, rate);
        for &t in &[0.0, 0.5, 1.0, 1.9, 2.0] {
            let cf = three_cpt_infusion_ss(&dose, t, CL, V1, Q2, V2, Q3, V3);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_infusion(&single, tt, CL, V1, Q2, V2, Q3, V3)
            });
            assert_relative_eq!(cf, num, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_ss_infusion_after_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let rate = 50.0; // dur=2
        let dose = ss_infusion_dose(100.0, rate, ii);
        let single = infusion_dose(100.0, rate);
        for &t in &[2.001, 5.0, 12.0, 23.5, 48.0] {
            let cf = three_cpt_infusion_ss(&dose, t, CL, V1, Q2, V2, Q3, V3);
            let num = ss_numerical_sum(t, ii, |tt| {
                three_cpt_infusion(&single, tt, CL, V1, Q2, V2, Q3, V3)
            });
            assert_relative_eq!(cf, num, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_ss_infusion_continuity_at_end_of_infusion() {
        let dose = ss_infusion_dose(100.0, 50.0, 24.0); // dur=2
        let c_at = three_cpt_infusion_ss(&dose, 2.0, CL, V1, Q2, V2, Q3, V3);
        let c_after = three_cpt_infusion_ss(&dose, 2.0 + 1e-10, CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-5);
    }

    #[test]
    fn test_ss_infusion_with_t_inf_gt_ii_returns_zero() {
        // amt=100, rate=200 → duration=0.5; ii=0.25 → t_inf > ii.
        let dose = DoseEvent::new(0.0, 100.0, 1, 200.0, true, 0.25);
        assert_eq!(
            three_cpt_infusion_ss(&dose, 0.1, CL, V1, Q2, V2, Q3, V3),
            0.0
        );
    }

    #[test]
    fn test_ss_oral_with_bioavailability_scales() {
        let dose = ss_bolus_dose(100.0, 24.0);
        let c_full = three_cpt_oral_f_ss(&dose, 4.0, CL, V1, Q2, V2, Q3, V3, KA, 1.0);
        let c_half = three_cpt_oral_f_ss(&dose, 4.0, CL, V1, Q2, V2, Q3, V3, KA, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }
}
