/// NCA primitives: AUC, terminal slope, Cmax/Tmax, C0, Wagner-Nelson.
///
/// All functions operate on dose-normalised concentrations (divide raw obs by dose
/// before passing in). Times are assumed to be sorted ascending.
use crate::types::Subject;

/// Per-subject NCA results.  `None` fields indicate the quantity could not be
/// estimated (too few points, failed R² check, etc.).
#[derive(Debug, Clone)]
#[allow(dead_code)] // cmax/auc_inf/mrt exposed for diagnostics
pub struct SubjectNca {
    pub cl_f: Option<f64>, // CL or CL/F = dose / AUC∞
    pub v_f: Option<f64>,  // Vd or Vd/F = CL_f / lambda_z
    pub vss: Option<f64>,  // Vss = CL_f × MRT (multi-cpt)
    pub lambda_z: Option<f64>,
    pub ka: Option<f64>,   // absorption rate (oral models only)
    pub c0: Option<f64>,   // back-extrapolated initial conc (IV models only)
    pub tlag: Option<f64>, // apparent lag time (oral models only)
    pub tmax: f64,
    pub cmax: f64,
    pub auc_inf: Option<f64>,
    pub mrt: Option<f64>,
}

/// Compute linear-up / log-down trapezoidal AUC for dose-normalised
/// concentrations on the first-dose interval [t0, t_end).
///
/// Returns (AUC_obs, C_last, t_last).  Returns None if fewer than 2 valid points.
pub fn auc_trapezoid(times: &[f64], concs: &[f64]) -> Option<(f64, f64, f64)> {
    debug_assert_eq!(times.len(), concs.len());
    let pairs: Vec<(f64, f64)> = times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite() && c >= 0.0)
        .map(|(&t, &c)| (t, c))
        .collect();

    if pairs.len() < 2 {
        return None;
    }

    let mut auc = 0.0;
    for w in pairs.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        let dt = t1 - t0;
        if dt <= 0.0 {
            continue;
        }
        // Linear-up / log-down: linear when c1 >= c0, log-down otherwise.
        let trap = if c1 >= c0 || c0 <= 0.0 || c1 <= 0.0 {
            0.5 * (c0 + c1) * dt
        } else {
            (c0 - c1) / (c0.ln() - c1.ln()) * dt
        };
        auc += trap;
    }

    let &(t_last, c_last) = pairs.last().unwrap();
    Some((auc, c_last, t_last))
}

/// AUMC (area under first moment curve) via linear-up / log-down rule.
/// Returns (AUMC_obs, t_last, C_last).
pub fn aumc_trapezoid(times: &[f64], concs: &[f64]) -> Option<(f64, f64, f64)> {
    let pairs: Vec<(f64, f64)> = times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite() && c >= 0.0)
        .map(|(&t, &c)| (t, t * c))
        .collect();

    if pairs.len() < 2 {
        return None;
    }

    let mut aumc = 0.0;
    for w in pairs.windows(2) {
        let (t0, tc0) = w[0];
        let (t1, tc1) = w[1];
        let dt = t1 - t0;
        if dt <= 0.0 {
            continue;
        }
        let trap = if tc1 >= tc0 || tc0 <= 0.0 || tc1 <= 0.0 {
            0.5 * (tc0 + tc1) * dt
        } else {
            (tc0 - tc1) / (tc0.ln() - tc1.ln()) * dt
        };
        aumc += trap;
    }

    let &(t_last, tc_last) = pairs.last().unwrap();
    let c_last = if t_last > 0.0 { tc_last / t_last } else { 0.0 };
    Some((aumc, t_last, c_last))
}

/// Log-linear regression on the last `n_pts` terminal points where C > cmax/10.
///
/// Returns `(lambda_z, intercept_ln)` where `ln(C) = intercept_ln - lambda_z * t`,
/// or `None` if fewer than `min_pts` points survive or R² < `r2_threshold`.
pub fn terminal_slope(
    times: &[f64],
    concs: &[f64],
    cmax: f64,
    min_pts: usize,
    r2_threshold: f64,
) -> Option<(f64, f64)> {
    terminal_slope_frac(times, concs, cmax, min_pts, r2_threshold, 0.1)
}

/// Like [`terminal_slope`] but with a configurable `cmax_fraction` threshold.
/// Use a lower fraction (e.g. 0.01) when estimating the true terminal phase
/// of multi-compartment data, where the terminal phase sits well below 10% of Cmax.
pub fn terminal_slope_frac(
    times: &[f64],
    concs: &[f64],
    cmax: f64,
    min_pts: usize,
    r2_threshold: f64,
    cmax_fraction: f64,
) -> Option<(f64, f64)> {
    let threshold = cmax * cmax_fraction;
    let pts: Vec<(f64, f64)> = times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite() && c > threshold && c > 0.0)
        .map(|(&t, &c)| (t, c.ln()))
        .collect();

    if pts.len() < min_pts {
        return None;
    }

    // Use the last min(pts.len(), 10) points for the regression.
    let used = pts.len().min(10);
    let pts = &pts[pts.len() - used..];

    let (slope, intercept, r2) = ols_slope(pts)?;

    if r2 < r2_threshold || slope >= 0.0 {
        return None;
    }

    Some((-slope, intercept))
}

/// Ordinary least squares on (t, ln_c) pairs.
/// Returns (slope, intercept, R²).
fn ols_slope(pts: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let n = pts.len() as f64;
    if n < 2.0 {
        return None;
    }
    let sum_x: f64 = pts.iter().map(|(t, _)| t).sum();
    let sum_y: f64 = pts.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = pts.iter().map(|(t, _)| t * t).sum();
    let sum_xy: f64 = pts.iter().map(|(t, y)| t * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // R²
    let y_mean = sum_y / n;
    let ss_tot: f64 = pts.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = pts
        .iter()
        .map(|(t, y)| (y - (slope * t + intercept)).powi(2))
        .sum();
    let r2 = if ss_tot < 1e-30 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    Some((slope, intercept, r2))
}

/// Back-extrapolate C0 for IV models from the first two observations.
pub fn backextrapolate_c0(times: &[f64], concs: &[f64]) -> Option<f64> {
    let pts: Vec<(f64, f64)> = times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite() && c > 0.0)
        .take(4)
        .map(|(&t, &c)| (t, c.ln()))
        .collect();

    if pts.len() < 2 {
        return None;
    }

    // Fit log-linear to first few points, extrapolate to t=0.
    let (_, intercept, _) = ols_slope(&pts)?;
    let c0 = intercept.exp();
    if c0.is_finite() && c0 > 0.0 {
        Some(c0)
    } else {
        None
    }
}

/// Wagner-Nelson method to estimate Ka for a 1-cpt oral model.
///
/// Given dose-normalised observations and CL/F and V/F, computes
/// the amount remaining to be absorbed A(t) = (C_t - C_0)/ka + CL/V * AUC(0,t).
/// Returns `Some(ka)` estimated as -slope of ln(A) vs t, or `None` on failure.
pub fn wagner_nelson_ka(times: &[f64], concs: &[f64], cl_f: f64, v_f: f64) -> Option<f64> {
    if cl_f <= 0.0 || v_f <= 0.0 {
        return None;
    }
    let k10 = cl_f / v_f; // elimination rate constant

    let pairs: Vec<(f64, f64)> = times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite() && c >= 0.0)
        .map(|(&t, &c)| (t, c))
        .collect();

    if pairs.len() < 3 {
        return None;
    }

    // Build cumulative AUC at each time point.
    let mut cum_auc = vec![0.0_f64; pairs.len()];
    for i in 1..pairs.len() {
        let (t0, c0) = pairs[i - 1];
        let (t1, c1) = pairs[i];
        let dt = t1 - t0;
        let trap = if c1 >= c0 || c0 <= 0.0 || c1 <= 0.0 {
            0.5 * (c0 + c1) * dt
        } else {
            (c0 - c1) / (c0.ln() - c1.ln()) * dt
        };
        cum_auc[i] = cum_auc[i - 1] + trap;
    }

    let c_last = pairs.last().unwrap().1;
    let auc_inf = cum_auc.last().unwrap() + c_last / k10;

    // A(t) = dose * (1 - (k10*AUC(0,t) + C_t) / (k10*AUC_inf))
    // For dose-normalised concs, dose=1.
    let wn_pts: Vec<(f64, f64)> = pairs
        .iter()
        .zip(cum_auc.iter())
        .filter_map(|(&(t, c), &auc)| {
            let absorbed_frac = (k10 * auc + c) / (k10 * auc_inf);
            let a = 1.0 - absorbed_frac;
            if a > 0.05 && a < 1.0 {
                // Only absorption phase (A still meaningful)
                Some((t, a.ln()))
            } else {
                None
            }
        })
        .collect();

    if wn_pts.len() < 3 {
        return None;
    }

    let (slope, _, r2) = ols_slope(&wn_pts)?;
    if r2 < 0.5 || slope >= 0.0 {
        return None;
    }
    Some(-slope)
}

/// Extract Cmax and Tmax from dose-normalised concentrations.
pub fn cmax_tmax(times: &[f64], concs: &[f64]) -> (f64, f64) {
    times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite())
        .fold((0.0_f64, 0.0_f64), |(cmax, tmax), (&t, &c)| {
            if c > cmax {
                (c, t)
            } else {
                (cmax, tmax)
            }
        })
}

/// Biexponential peeling for 2-cpt IV/infusion data.
///
/// Fits C(t) = A·exp(-α·t) + B·exp(-β·t) to pooled dose-normalised data
/// using the method of residuals (two log-linear regressions).
///
/// Returns `Some((A, alpha, B, beta))` with alpha > beta, or `None` if the fit
/// fails or phases are not well separated (α/β < `min_separation`).
pub fn biexponential_peel(
    times: &[f64],
    concs: &[f64],
    min_pts: usize,
    min_separation: f64, // α/β threshold, typically 3.0
) -> Option<(f64, f64, f64, f64)> {
    // Step 1: terminal phase → β, B.
    // Classic method of residuals: use only the LAST min_pts+1 points by time for
    // the terminal regression.  This avoids including early distribution-dominated
    // points that would bias the slope estimate toward α.
    let valid_pts: Vec<(f64, f64)> = times
        .iter()
        .zip(concs.iter())
        .filter(|(_, &c)| c.is_finite() && c > 0.0)
        .map(|(&t, &c)| (t, c))
        .collect();

    if valid_pts.len() < min_pts * 2 {
        return None;
    }

    // Use the last (min_pts + 1) points for terminal regression (max 5).
    let n_terminal = (min_pts + 1).min(5).min(valid_pts.len() / 2);
    let terminal_pts: Vec<(f64, f64)> = valid_pts[valid_pts.len() - n_terminal..]
        .iter()
        .map(|&(t, c)| (t, c.ln()))
        .collect();

    let (slope_term, intercept_term, r2_term) = ols_slope(&terminal_pts)?;
    if slope_term >= 0.0 || r2_term < 0.8 {
        return None;
    }
    let beta = -slope_term;
    let b_cap = intercept_term.exp();

    // Step 2: compute residuals on the EARLY points (all except terminal).
    // These are the points dominated by the distribution phase.
    let early_pts: &[(f64, f64)] = &valid_pts[..valid_pts.len() - n_terminal];
    let residuals: Vec<(f64, f64)> = early_pts
        .iter()
        .filter_map(|&(t, c)| {
            let terminal_contrib = b_cap * (-beta * t).exp();
            let resid = c - terminal_contrib;
            if resid > 0.0 {
                Some((t, resid.ln()))
            } else {
                None
            }
        })
        .collect();

    if residuals.len() < min_pts {
        return None;
    }

    // Step 3: regression on residuals → α, A
    let (slope, intercept, _r2) = ols_slope(&residuals)?;
    if slope >= 0.0 {
        return None;
    }
    let alpha = -slope;
    let a_cap = intercept.exp();

    // Require well-separated phases
    if alpha / beta < min_separation {
        return None;
    }

    Some((a_cap, alpha, b_cap, beta))
}

/// Run per-subject NCA for an oral 1-cpt model.
pub fn nca_one_cpt_oral(subject: &Subject) -> SubjectNca {
    let dose = first_dose_amt(subject);
    if dose <= 0.0 {
        return empty_nca();
    }

    let (times, concs) = first_dose_obs(subject);
    if times.len() < 2 {
        return empty_nca();
    }

    // Dose-normalise
    let concs_norm: Vec<f64> = concs.iter().map(|&c| c / dose).collect();

    let (cmax, tmax) = cmax_tmax(&times, &concs_norm);

    // Detect lag before NCA so we can shift times for AUC/lambda_z.
    // Without shifting, near-zero pre-lag observations inflate AUC and cause
    // CL = Dose/AUC to collapse to near-zero.
    let tlag = estimate_tlag(&times, &concs_norm, cmax);
    let lag_shift = tlag.unwrap_or(0.0);

    // Drop observations that fall entirely within the lag period and shift the
    // remaining times so absorption appears to start at t=0.
    let (times_shifted, concs_shifted): (Vec<f64>, Vec<f64>) = times
        .iter()
        .zip(concs_norm.iter())
        .filter(|(&t, _)| t >= lag_shift)
        .map(|(&t, &c)| (t - lag_shift, c))
        .unzip();

    if times_shifted.len() < 2 {
        return SubjectNca {
            cmax,
            tmax,
            tlag,
            ..empty_nca()
        };
    }

    let Some((auc_obs, c_last, t_last)) = auc_trapezoid(&times_shifted, &concs_shifted) else {
        return SubjectNca {
            cmax,
            tmax,
            tlag,
            ..empty_nca()
        };
    };

    let lz = terminal_slope(&times_shifted, &concs_shifted, cmax, 3, 0.8);
    let auc_inf = lz.map(|(lz, _)| auc_obs + c_last / lz);
    let cl_f = auc_inf.map(|a| if a > 0.0 { 1.0 / a } else { f64::NAN });
    let v_f = lz.and_then(|(lz, _)| cl_f.map(|cl| cl / lz));

    // AUMC for MRT
    let (aumc_obs, _, _) =
        aumc_trapezoid(&times_shifted, &concs_shifted).unwrap_or((0.0, t_last, c_last));
    let aumc_inf =
        lz.and_then(|(lz, _)| Some(aumc_obs + c_last / lz.powi(2) + t_last * c_last / lz));
    let mrt =
        aumc_inf.and_then(|aumc| auc_inf.map(|auc| if auc > 0.0 { aumc / auc } else { f64::NAN }));
    let vss = mrt.and_then(|m| cl_f.map(|cl| cl * m));

    // Ka via Wagner-Nelson on lag-shifted times; fallback = heuristic from Tmax
    let tmax_shifted = if tmax >= lag_shift {
        tmax - lag_shift
    } else {
        tmax
    };
    let ka = if let (Some(cl), Some(v)) = (cl_f, v_f) {
        wagner_nelson_ka(&times_shifted, &concs_shifted, cl, v).or_else(|| {
            if tmax_shifted > 0.0 {
                Some((2.0_f64).ln() / (0.5 * tmax_shifted))
            } else {
                None
            }
        })
    } else if tmax_shifted > 0.0 {
        Some((2.0_f64).ln() / (0.5 * tmax_shifted))
    } else {
        None
    };

    SubjectNca {
        cl_f,
        v_f,
        vss,
        lambda_z: lz.map(|(l, _)| l),
        ka,
        c0: None,
        tlag,
        tmax,
        cmax,
        auc_inf,
        mrt,
    }
}

/// Run per-subject NCA for an IV bolus or IV infusion 1-cpt model.
///
/// For infusion doses (`rate > 0`): terminal-slope regression uses only
/// post-infusion observations (t > t_end_infusion) to avoid including the
/// rising absorption phase, and C0 back-extrapolation is skipped.
pub fn nca_one_cpt_iv(subject: &Subject) -> SubjectNca {
    let dose = first_dose_amt(subject);
    if dose <= 0.0 {
        return empty_nca();
    }

    let (times, concs) = first_dose_obs(subject);
    if times.len() < 2 {
        return empty_nca();
    }

    let concs_norm: Vec<f64> = concs.iter().map(|&c| c / dose).collect();
    let (cmax, tmax) = cmax_tmax(&times, &concs_norm);

    // Infusion end time: for rate > 0, terminal phase starts after the infusion ends.
    let t_infusion_end = subject
        .doses
        .first()
        .filter(|d| d.rate > 0.0)
        .map(|d| d.time + d.duration)
        .unwrap_or(0.0);
    let is_infusion = t_infusion_end > 0.0;

    let Some((auc_obs, c_last, t_last)) = auc_trapezoid(&times, &concs_norm) else {
        return SubjectNca {
            cmax,
            tmax,
            ..empty_nca()
        };
    };

    // Terminal slope: use only post-infusion points for infusion to avoid
    // mixing the rising phase with the declining phase.
    let lz = if is_infusion {
        let (t_post, c_post): (Vec<f64>, Vec<f64>) = times
            .iter()
            .zip(concs_norm.iter())
            .filter(|(&t, _)| t > t_infusion_end)
            .map(|(&t, &c)| (t, c))
            .unzip();
        if t_post.len() >= 3 {
            terminal_slope(&t_post, &c_post, cmax, 3, 0.8)
        } else {
            None
        }
    } else {
        terminal_slope(&times, &concs_norm, cmax, 3, 0.8)
    };

    let auc_inf = lz.map(|(lz, _)| auc_obs + c_last / lz);
    let cl_f = auc_inf.map(|a| if a > 0.0 { 1.0 / a } else { f64::NAN });
    let v_f = lz.and_then(|(lz, _)| cl_f.map(|cl| cl / lz));

    let (aumc_obs, _, _) = aumc_trapezoid(&times, &concs_norm).unwrap_or((0.0, t_last, c_last));
    let aumc_inf =
        lz.and_then(|(lz, _)| Some(aumc_obs + c_last / lz.powi(2) + t_last * c_last / lz));
    let mrt =
        aumc_inf.and_then(|aumc| auc_inf.map(|auc| if auc > 0.0 { aumc / auc } else { f64::NAN }));
    let vss = mrt.and_then(|m| cl_f.map(|cl| cl * m));
    // C0 back-extrapolation is only meaningful for bolus dosing.
    let c0 = if is_infusion {
        None
    } else {
        backextrapolate_c0(&times, &concs_norm)
    };

    SubjectNca {
        cl_f,
        v_f,
        vss,
        lambda_z: lz.map(|(l, _)| l),
        ka: None,
        c0,
        tlag: None,
        tmax,
        cmax,
        auc_inf,
        mrt,
    }
}

/// Dose amount of the first dose event.
pub fn first_dose_amt(subject: &Subject) -> f64 {
    subject.doses.first().map(|d| d.amt).unwrap_or(0.0)
}

/// Observations on the first-dose interval.
/// Returns (times, concs) excluding censored (cens==1) observations.
pub fn first_dose_obs(subject: &Subject) -> (Vec<f64>, Vec<f64>) {
    let t_dose = subject.doses.first().map(|d| d.time).unwrap_or(0.0);
    let t_next_dose = subject
        .doses
        .get(1)
        .map(|d| d.time)
        .unwrap_or(f64::INFINITY);

    let iter = subject
        .obs_times
        .iter()
        .zip(subject.observations.iter())
        .zip(subject.cens.iter())
        .filter(|&((&t, &c), &cens)| {
            t >= t_dose && t < t_next_dose && cens == 0 && c.is_finite() && c >= 0.0
        });

    let mut times = Vec::new();
    let mut concs = Vec::new();
    for ((&t, &c), _) in iter {
        times.push(t);
        concs.push(c);
    }
    (times, concs)
}

fn empty_nca() -> SubjectNca {
    SubjectNca {
        cl_f: None,
        v_f: None,
        vss: None,
        lambda_z: None,
        ka: None,
        c0: None,
        tlag: None,
        tmax: 0.0,
        cmax: 0.0,
        auc_inf: None,
        mrt: None,
    }
}

/// Estimate apparent lag time as the last observation time before Tmax where
/// concentration is below `threshold` × Cmax.  Returns None when the first
/// observation already exceeds the threshold (no detectable lag).
fn estimate_tlag(times: &[f64], concs: &[f64], cmax: f64) -> Option<f64> {
    if cmax <= 0.0 || times.is_empty() {
        return None;
    }
    let threshold = 0.05 * cmax;
    // Walk forward; stop at Cmax to avoid the elimination phase.
    let tmax = times
        .iter()
        .zip(concs.iter())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(&t, _)| t)
        .unwrap_or(f64::INFINITY);

    let last_below: Option<f64> = times
        .iter()
        .zip(concs.iter())
        .take_while(|(&t, _)| t <= tmax)
        .filter(|(_, &c)| c < threshold)
        .map(|(&t, _)| t)
        .last();

    // Only report a lag if it's after time zero (a zero-time pre-dose sample
    // at c≈0 is not a lag).
    last_below.filter(|&t| t > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monoexp(cl: f64, v: f64, dose: f64, times: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let k = cl / v;
        let concs: Vec<f64> = times.iter().map(|&t| (dose / v) * (-k * t).exp()).collect();
        (times.to_vec(), concs)
    }

    #[test]
    fn test_auc_linear_up_logdown() {
        // 1-cpt IV: C_norm(t) = (1/V) * exp(-k*t), k = CL/V.
        // auc_trapezoid returns the *observed* trapezoidal AUC from t=0 to t_last
        // (no tail extrapolation), so we compare against the analytical integral
        // from 0 to t_last:  AUC_obs = (1/(V*k)) * (1 - exp(-k*t_last)).
        let cl = 0.5_f64;
        let v = 10.0_f64;
        let dose = 100.0_f64;
        let times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let (t, c) = make_monoexp(cl, v, dose, &times);
        // Dose-normalise
        let cn: Vec<f64> = c.iter().map(|&x| x / dose).collect();
        let (auc, _, _) = auc_trapezoid(&t, &cn).unwrap();
        let k = cl / v;
        let t_last = *times.last().unwrap();
        // Analytical observed AUC (0..t_last), dose-normalised
        let analytical = (1.0 / (v * k)) * (1.0 - (-k * t_last).exp());
        assert!(
            (auc - analytical).abs() / analytical < 0.01,
            "AUC error > 1%: got {auc:.4}, expected {analytical:.4}"
        );
    }

    #[test]
    fn test_auc_multidose_uses_first_interval() {
        use crate::types::{DoseEvent, Subject};
        use std::collections::HashMap;

        let dose_amt = 100.0;
        let times_all = vec![0.5, 1.0, 2.0, 4.0, 24.5, 25.0, 26.0];
        let concs_all = vec![8.0, 7.0, 5.5, 3.5, 7.8, 6.8, 5.2]; // second dose starts at 24

        let subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent {
                    time: 0.0,
                    amt: dose_amt,
                    rate: 0.0,
                    duration: 0.0,
                    cmt: 1,
                    ss: false,
                    ii: 0.0,
                },
                DoseEvent {
                    time: 24.0,
                    amt: dose_amt,
                    rate: 0.0,
                    duration: 0.0,
                    cmt: 1,
                    ss: false,
                    ii: 0.0,
                },
            ],
            obs_times: times_all.clone(),
            obs_raw_times: Vec::new(),
            observations: concs_all.clone(),
            cens: vec![0; 7],
            obs_cmts: vec![1; 7],
            occasions: vec![],
            dose_occasions: vec![],
            #[cfg(feature = "survival")]
            obs_records: vec![],
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
        };

        let (t, c) = first_dose_obs(&subject);
        // Only first 4 time points should be included (t < 24.0)
        assert_eq!(t.len(), 4);
        assert_eq!(t.last(), Some(&4.0));
        let _ = c;
    }

    #[test]
    fn test_terminal_slope_monoexp() {
        let cl = 0.5;
        let v = 10.0;
        let dose = 100.0;
        let times = vec![2.0, 4.0, 8.0, 12.0, 24.0];
        let (t, c) = make_monoexp(cl, v, dose, &times);
        let cn: Vec<f64> = c.iter().map(|&x| x / dose).collect();
        let cmax = cn.iter().cloned().fold(0.0_f64, f64::max);
        let (lz, _) = terminal_slope(&t, &cn, cmax, 3, 0.8).unwrap();
        let k_true = cl / v; // 0.05
        assert!(
            (lz - k_true).abs() / k_true < 0.01,
            "λz error > 1%: got {lz:.5}, expected {k_true:.5}"
        );
    }

    #[test]
    fn test_terminal_slope_too_few_points() {
        let times = vec![1.0, 2.0];
        let concs = vec![0.5, 0.3];
        let result = terminal_slope(&times, &concs, 0.5, 3, 0.8);
        assert!(
            result.is_none(),
            "should return None with < 3 terminal points"
        );
    }

    #[test]
    fn test_biexp_peeling_two_cpt_iv() {
        // Well-separated 2-cpt IV: CL=5, V1=10, Q=3, V2=60.
        // k10=0.5, k12=0.3, k21=0.05 → α≈0.83, β≈0.030; α/β≈28.
        // With α/β≈28 the distribution phase (α) has decayed to <1% by t=6.
        // Use an extended time grid (up to t=72) so the last 4 terminal points
        // sit firmly in the pure-β region, making beta and B recoverable to <15%.
        let dose = 100.0_f64;
        let cl = 5.0_f64;
        let v1 = 10.0_f64;
        let q = 3.0_f64;
        let v2 = 60.0_f64;
        let k10 = cl / v1;
        let k12 = q / v1;
        let k21 = q / v2;
        let sum_ = k10 + k12 + k21;
        let disc = (sum_ * sum_ - 4.0 * k10 * k21).sqrt();
        let alpha = 0.5 * (sum_ + disc);
        let beta = 0.5 * (sum_ - disc);
        let a_cap = (dose / v1) * (alpha - k21) / (alpha - beta);
        let b_cap = (dose / v1) * (k21 - beta) / (alpha - beta);

        let times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 48.0, 72.0];
        let concs: Vec<f64> = times
            .iter()
            .map(|&t| (a_cap * (-alpha * t).exp() + b_cap * (-beta * t).exp()) / dose)
            .collect();
        let (a, al, b, be) = biexponential_peel(&times, &concs, 3, 2.0).unwrap();
        assert!(
            (a - a_cap / dose).abs() / (a_cap / dose) < 0.15,
            "A error > 15%: got {a:.5}, expected {:.5}",
            a_cap / dose
        );
        assert!(
            (b - b_cap / dose).abs() / (b_cap / dose) < 0.15,
            "B error > 15%: got {b:.5}, expected {:.5}",
            b_cap / dose
        );
        assert!(
            (al - alpha).abs() / alpha < 0.15,
            "α error > 15%: got {al:.5}, expected {alpha:.5}"
        );
        assert!(
            (be - beta).abs() / beta < 0.15,
            "β error > 15%: got {be:.5}, expected {beta:.5}"
        );
    }

    #[test]
    fn test_biexp_peeling_poor_separation_returns_none() {
        // A perfectly monoexponential decay has α/β = 1 (no two-phase structure).
        // After the terminal regression subtracts the fitted beta component, the
        // residuals on early points are all near zero (or negative), so there are
        // no valid log-residual points and biexponential_peel must return None.
        let times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let concs: Vec<f64> = times.iter().map(|&t| 1.0 * (-0.2 * t).exp()).collect();
        let result = biexponential_peel(&times, &concs, 3, 3.0);
        assert!(
            result.is_none(),
            "monoexponential data should return None (no biexponential structure)"
        );
    }

    #[test]
    fn test_wagner_nelson_ka() {
        // Simulate 1-cpt oral: CL=0.134, V=8.1, Ka=1.0
        let cl = 0.134;
        let v = 8.1;
        let ka = 1.0;
        let k = cl / v;
        let times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let concs_norm: Vec<f64> = times
            .iter()
            .map(|&t| (ka / (v * (ka - k))) * ((-k * t).exp() - (-ka * t).exp()))
            .collect();

        let ka_est = wagner_nelson_ka(&times, &concs_norm, cl, v).unwrap();
        assert!(
            (ka_est - ka).abs() / ka < 0.15,
            "Ka error > 15%: got {ka_est:.3}, expected {ka:.3}"
        );
    }
}
