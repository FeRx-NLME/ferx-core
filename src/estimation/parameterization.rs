use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, SigmaVector};
use nalgebra::DMatrix;

/// Bounds for the packed parameter vector
pub struct PackedBounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Whether to pack `theta[i]` on the log scale.
///
/// Log packing applies when `theta_lower >= 0` — i.e. the user has
/// declared the parameter as non-negative (the typical case for CL, V,
/// KA, sigma; `theta_lower = 0` is also included). When
/// `theta_lower < 0`, the user has explicitly allowed negative values —
/// typical for covariate exponents (`(DOSE/100)^γ` with γ ∈ [-3, 3]),
/// additive covariate effects (`THETA_AGE_CL ∈ [-1, 1]`), or logit-scale
/// parameters. Log-packing those silently clamps to 1e-10 and the
/// optimizer can never reach the true sign-bearing value (regression:
/// SAD_SCEN4's γ = -0.8 collapsed to 1e-10 ≈ 0, and SAD_SCEN1's
/// THETA_AGE_CL = -0.01 collapsed to the same).
///
/// Identity packing is opted into only by a *negative* lower bound. A
/// `theta_lower = 0` parameter still uses log packing (with the
/// `max(1e-10)` floor handling the boundary). This preserves the
/// optimizer conditioning that established users rely on for
/// sign-constrained parameters that can span many orders of magnitude.
#[inline]
pub(crate) fn theta_packs_log(theta_lower: f64) -> bool {
    theta_lower >= 0.0
}

/// Pack ModelParameters into a flat unconstrained vector for optimization.
///
/// Layout: [pack(theta_1), ..., pack(theta_n),
///          log(L_11), L_21, log(L_22), ...,   (Cholesky lower triangle)
///          log(sigma_1), ..., log(sigma_m)]
///
/// Theta packing depends on whether the user's `theta_lower[i]` allows
/// negatives — see [`theta_packs_log`].
pub fn pack_params(params: &ModelParameters) -> Vec<f64> {
    let mut v = Vec::new();

    // Theta: log-transformed when lower bound is non-negative; identity
    // otherwise (so negative-valued parameters like covariate exponents
    // can be expressed at all).
    for (i, &th) in params.theta.iter().enumerate() {
        if theta_packs_log(params.theta_lower[i]) {
            v.push(th.max(1e-10).ln());
        } else {
            v.push(th);
        }
    }

    // Omega Cholesky factor: diagonal as log, off-diagonal as-is. `lower_tri_iter`
    // yields only `(i,i)` when `diagonal`, so this one loop covers both cases.
    let l = &params.omega.chol;
    let n_eta = l.nrows();
    for (i, j) in lower_tri_iter(n_eta, params.omega.diagonal) {
        if i == j {
            v.push(l[(i, j)].max(1e-10).ln());
        } else {
            v.push(l[(i, j)]);
        }
    }

    // Sigma: log-transformed
    for &s in &params.sigma.values {
        v.push(s.max(1e-10).ln());
    }

    // IOV omega: diagonal elements as log; off-diagonal as-is (mirrors BSV omega).
    if let Some(ref iov) = params.omega_iov {
        let l = &iov.chol;
        for (i, j) in lower_tri_iter(iov.dim(), iov.diagonal) {
            if i == j {
                v.push(l[(i, j)].max(1e-10).ln());
            } else {
                v.push(l[(i, j)]);
            }
        }
    }

    // Mixture per-class Omega/Sigma overrides (#977). Each override is a single
    // diagonal scalar: an Omega override packs `ln` of the class matrix's
    // Cholesky diagonal (== `ln(sd)`, matching the base diagonal-Omega form), a
    // Sigma override packs `ln(sd)`. Non-overridden class entries are not packed
    // — they track the base Omega/Sigma segments above. Appended after the IOV
    // segment so all existing offsets stay put.
    if let Some(ref mix) = params.mixture {
        for &(c, e) in &mix.omega_override_addr {
            v.push(mix.omega[c].chol[(e, e)].max(1e-10).ln());
        }
        for &(c, s) in &mix.sigma_override_addr {
            v.push(mix.sigma[c].values[s].max(1e-10).ln());
        }
    }

    v
}

/// Unpack a flat unconstrained vector back into ModelParameters.
pub fn unpack_params(v: &[f64], template: &ModelParameters) -> ModelParameters {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();
    let mut idx = 0;

    // Theta — back-transform mirrors `pack_params`.
    let theta: Vec<f64> = (0..n_theta)
        .map(|i| {
            let val = if theta_packs_log(template.theta_lower[i]) {
                v[idx].exp()
            } else {
                v[idx]
            };
            idx += 1;
            val
        })
        .collect();

    // Omega Cholesky. `lower_tri_iter` yields only `(i,i)` when `diagonal`, so this
    // one loop covers both cases (same packed order as `pack_params`).
    let mut l = DMatrix::zeros(n_eta, n_eta);
    for (i, j) in lower_tri_iter(n_eta, template.omega.diagonal) {
        l[(i, j)] = if i == j { v[idx].exp() } else { v[idx] };
        idx += 1;
    }
    let omega = OmegaMatrix::from_chol_factor(
        l,
        template.omega.eta_names.clone(),
        template.omega.diagonal,
        template.omega.free_mask.clone(),
    );

    // Sigma
    let sigma_values: Vec<f64> = (0..n_sigma)
        .map(|_| {
            let val = v[idx].exp();
            idx += 1;
            val
        })
        .collect();
    let sigma = SigmaVector {
        values: sigma_values,
        names: template.sigma.names.clone(),
    };

    // IOV omega: mirrors BSV omega unpacking, checking the diagonal flag.
    let omega_iov = if let Some(ref iov_tmpl) = template.omega_iov {
        let n_iov = iov_tmpl.dim();
        if iov_tmpl.diagonal {
            let mut variances = Vec::with_capacity(n_iov);
            for _ in 0..n_iov {
                let chol_diag = v[idx].exp();
                idx += 1;
                variances.push(chol_diag * chol_diag);
            }
            Some(OmegaMatrix::from_diagonal(
                &variances,
                iov_tmpl.eta_names.clone(),
            ))
        } else {
            let mut l = DMatrix::zeros(n_iov, n_iov);
            for (i, j) in lower_tri_iter(n_iov, false) {
                l[(i, j)] = if i == j { v[idx].exp() } else { v[idx] };
                idx += 1;
            }
            Some(OmegaMatrix::from_chol_factor(
                l,
                iov_tmpl.eta_names.clone(),
                false,
                iov_tmpl.free_mask.clone(),
            ))
        }
    } else {
        None
    };

    // Mixture per-class Omega/Sigma (#977). Rebuild each class from the *newly
    // unpacked* base (`omega`/`sigma`) so non-overridden entries track the base,
    // then apply this class's overrides from the packed scalars (same order as
    // `pack_params`: all Omega overrides, then all Sigma overrides).
    let mixture = template.mixture.as_ref().map(|tmpl| {
        let k = tmpl.omega.len();
        let mut class_omega_mat: Vec<DMatrix<f64>> = vec![omega.matrix.clone(); k];
        let mut class_sigma_val: Vec<Vec<f64>> = vec![sigma.values.clone(); k];
        for &(c, e) in &tmpl.omega_override_addr {
            let chol_diag = v[idx].exp();
            idx += 1;
            class_omega_mat[c][(e, e)] = chol_diag * chol_diag;
        }
        for &(c, s) in &tmpl.sigma_override_addr {
            class_sigma_val[c][s] = v[idx].exp();
            idx += 1;
        }
        let class_omega = (0..k)
            .map(|c| {
                OmegaMatrix::from_matrix_with_mask(
                    class_omega_mat[c].clone(),
                    omega.eta_names.clone(),
                    omega.diagonal,
                    omega.free_mask.clone(),
                )
            })
            .collect();
        let class_sigma = (0..k)
            .map(|c| SigmaVector {
                values: class_sigma_val[c].clone(),
                names: sigma.names.clone(),
            })
            .collect();
        crate::types::MixtureParams {
            omega: class_omega,
            sigma: class_sigma,
            omega_override_addr: tmpl.omega_override_addr.clone(),
            omega_override_fixed: tmpl.omega_override_fixed.clone(),
            sigma_override_addr: tmpl.sigma_override_addr.clone(),
            sigma_override_fixed: tmpl.sigma_override_fixed.clone(),
        }
    });

    ModelParameters {
        theta,
        theta_names: template.theta_names.clone(),
        theta_lower: template.theta_lower.clone(),
        theta_upper: template.theta_upper.clone(),
        theta_fixed: template.theta_fixed.clone(),
        omega,
        omega_fixed: template.omega_fixed.clone(),
        sigma,
        sigma_fixed: template.sigma_fixed.clone(),
        omega_iov,
        kappa_fixed: template.kappa_fixed.clone(),
        mixture,
    }
}

/// Build a boolean mask over the packed parameter vector marking which
/// entries are held fixed. Layout mirrors [`pack_params`]:
///
/// - Theta: `template.theta_fixed[i]`.
/// - Omega Cholesky L[i,j] is fixed iff either `omega_fixed[i]` or
///   `omega_fixed[j]` is set. Pinning the whole row and column of a FIX-ed
///   eta keeps that eta uncorrelated with any other random effect (its
///   initial off-diagonals are zero for a diagonal declaration, or its block
///   off-diagonals for a FIX-ed block).
/// - Sigma: `template.sigma_fixed[i]`.
pub fn packed_fixed_mask(template: &ModelParameters) -> Vec<bool> {
    let mut mask = Vec::with_capacity(packed_len(template));

    for &f in &template.theta_fixed {
        mask.push(f);
    }

    let n_eta = template.omega.dim();
    let omega_fixed: &[bool] = &template.omega_fixed;
    // `lower_tri_iter` yields only `(i,i)` when diagonal, where `fi || fj`
    // reduces to `omega_fixed[i]` — the same value the diagonal branch pushed.
    for (i, j) in lower_tri_iter(n_eta, template.omega.diagonal) {
        let fi = omega_fixed.get(i).copied().unwrap_or(false);
        let fj = omega_fixed.get(j).copied().unwrap_or(false);
        mask.push(fi || fj);
    }

    for &f in &template.sigma_fixed {
        mask.push(f);
    }

    // IOV: mirrors BSV omega mask logic, checking the diagonal flag.
    if let Some(ref iov) = template.omega_iov {
        let kf = &template.kappa_fixed;
        for (i, j) in lower_tri_iter(iov.dim(), iov.diagonal) {
            let fi = kf.get(i).copied().unwrap_or(false);
            let fj = kf.get(j).copied().unwrap_or(false);
            mask.push(fi || fj);
        }
    }

    // Mixture overrides (#977): one packed scalar each, FIX flag carried on the
    // override. Same order as `pack_params` (Omega overrides, then Sigma).
    if let Some(ref mix) = template.mixture {
        mask.extend_from_slice(&mix.omega_override_fixed);
        mask.extend_from_slice(&mix.sigma_override_fixed);
    }

    mask
}

/// Packed-length mask marking the **structural-zero** off-diagonal entries of a
/// mixed block + diagonal Ω (and Ω_IOV) — the cross-block elements where
/// `free_mask[(i,j)] == false`. These are not estimated parameters, so the
/// covariance step excludes them from its free set exactly like FIX parameters
/// (issue #243); otherwise their flat Hessian diagonal aborts the step. Theta
/// and sigma slots, all diagonal entries, and fully diagonal/full-block Ω are
/// always `false`. Layout mirrors [`packed_fixed_mask`]:
/// `[theta, Ω (lower-tri col-major), sigma, Ω_IOV (lower-tri col-major)]`.
pub fn omega_structural_zero_mask(template: &ModelParameters) -> Vec<bool> {
    let mut mask = vec![false; packed_len(template)];
    let n_theta = template.theta.len();

    // Mark the lower-triangle off-diagonals of `om` that are structural zeros,
    // walking the same column-major order `packed_fixed_mask` / `pack_params` use.
    let mark = |mask: &mut [bool], om: &OmegaMatrix, start: usize| {
        if om.diagonal {
            return; // diagonal Ω has no off-diagonal entries to mark
        }
        let mut p = start;
        for (i, j) in lower_tri_iter(om.dim(), false) {
            if i != j && !om.free_mask[(i, j)] {
                mask[p] = true;
            }
            p += 1;
        }
    };

    mark(&mut mask, &template.omega, n_theta);

    if let Some(ref iov) = template.omega_iov {
        let n_omega = omega_packed_len(template.omega.dim(), template.omega.diagonal);
        let iov_start = n_theta + n_omega + template.sigma.values.len();
        mark(&mut mask, iov, iov_start);
    }

    mask
}

/// Compute the number of packed parameters
pub fn packed_len(template: &ModelParameters) -> usize {
    let n_theta = template.theta.len();
    let n_omega = omega_packed_len(template.omega.dim(), template.omega.diagonal);
    let n_sigma = template.sigma.values.len();
    let n_iov = template
        .omega_iov
        .as_ref()
        .map_or(0, |m| omega_packed_len(m.dim(), m.diagonal));
    let n_mixture = template.mixture.as_ref().map_or(0, |mix| {
        mix.omega_override_addr.len() + mix.sigma_override_addr.len()
    });
    n_theta + n_omega + n_sigma + n_iov + n_mixture
}

/// Compute box constraints for the packed parameter vector.
///
/// Parameters marked FIX are given `lower == upper == packed_value`, which
/// pins them for every optimizer that respects box bounds (NLopt SLSQP/L-BFGS/MMA,
/// the hand-rolled BFGS, and the Gauss-Newton clamp on proposed steps).
pub fn compute_bounds(template: &ModelParameters) -> PackedBounds {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    let mut lower = Vec::new();
    let mut upper = Vec::new();

    // Theta bounds — packed in whichever space `pack_params` uses
    // (log when sign-constrained, identity otherwise).
    for i in 0..n_theta {
        if theta_packs_log(template.theta_lower[i]) {
            lower.push(template.theta_lower[i].max(1e-10).ln());
            upper.push(template.theta_upper[i].min(1e9).ln());
        } else {
            lower.push(template.theta_lower[i]);
            upper.push(template.theta_upper[i]);
        }
    }

    // Omega Cholesky bounds
    //
    // Diagonal elements are stored as log(L_ii), so the bound constrains the
    // Cholesky diagonal in [exp(lower), exp(upper)].  The previous upper of
    // 4.0 (exp(4) ≈ 55, max variance ≈ 3 000) is too tight for FREM models
    // whose covariate omega diagonals can reach 15 000+.  With 6.0 the cap
    // is exp(6) ≈ 403, max variance ≈ 162 000 — sufficient for practical
    // FREM covariate variances while still preventing runaway.
    // `lower_tri_iter` yields only `(i,i)` when diagonal, so the `i == j` arm
    // (`[-6, 6]`) covers the diagonal case; off-diagonals get `[-10, 10]`.
    for (i, j) in lower_tri_iter(n_eta, template.omega.diagonal) {
        if i == j {
            lower.push(-6.0); // exp(-6) ≈ 0.0025 .. exp(6) ≈ 403
            upper.push(6.0);
        } else {
            lower.push(-10.0);
            upper.push(10.0);
        }
    }

    // Sigma bounds (log-transformed)
    for _ in 0..n_sigma {
        lower.push(-8.0); // exp(-8) ≈ 3e-4
        upper.push(5.0); // exp(5) ≈ 148
    }

    // IOV bounds: diagonal same as BSV diagonal; off-diagonal same as BSV off-diagonal.
    if let Some(ref iov) = template.omega_iov {
        for (i, j) in lower_tri_iter(iov.dim(), iov.diagonal) {
            if i == j {
                lower.push(-6.0);
                upper.push(6.0);
            } else {
                lower.push(-10.0);
                upper.push(10.0);
            }
        }
    }

    // Mixture override bounds (#977): each is a diagonal scalar packed on the log
    // scale — Omega overrides use the log-Cholesky-diagonal bound `[-6, 6]`, Sigma
    // overrides the log-sigma bound `[-8, 5]`, matching their base counterparts.
    if let Some(ref mix) = template.mixture {
        for _ in 0..mix.omega_override_addr.len() {
            lower.push(-6.0);
            upper.push(6.0);
        }
        for _ in 0..mix.sigma_override_addr.len() {
            lower.push(-8.0);
            upper.push(5.0);
        }
    }

    // Pin any FIX parameters to their packed (log-space) initial value.
    // We pack first, then overwrite lower=upper=packed[i] for fixed indices.
    // Pack-before-overwrite is correct even for block Cholesky off-diagonals,
    // whose "packed" value is the raw L[i,j] (not log-transformed).
    let packed = pack_params(template);
    let fixed_mask = packed_fixed_mask(template);
    for i in 0..fixed_mask.len() {
        if fixed_mask[i] {
            lower[i] = packed[i];
            upper[i] = packed[i];
        }
    }

    PackedBounds { lower, upper }
}

/// Per-coordinate **display names** for the optimizer trace, in the same order
/// as [`pack_params`]: `[theta…, Ω (lower-tri col-major)…, sigma…, Ω_IOV…]`.
///
/// Declared names are preferred (`TVCL`, `ETA_CL`, `EPS_PROP`); an Ω
/// off-diagonal couples two etas as `ETA_i~ETA_j` (row eta ~ column eta). When a
/// name is missing the NONMEM-style fallback is used: `THETA1`, `OMEGA(2,1)`,
/// `SIGMA(1)`.
pub fn coordinate_names(params: &ModelParameters) -> Vec<String> {
    let mut names = Vec::with_capacity(packed_len(params));
    for i in 0..params.theta.len() {
        names.push(named_or(&params.theta_names, i, || {
            format!("THETA{}", i + 1)
        }));
    }
    push_omega_names(&mut names, &params.omega);
    for i in 0..params.sigma.values.len() {
        names.push(named_or(&params.sigma.names, i, || {
            format!("SIGMA({})", i + 1)
        }));
    }
    if let Some(ref iov) = params.omega_iov {
        push_omega_names(&mut names, iov);
    }
    // Mixture overrides (#977): `<ETA|SIGMA>_MIX{class}`, same order as pack.
    if let Some(ref mix) = params.mixture {
        for &(c, e) in &mix.omega_override_addr {
            let base = named_or(&params.omega.eta_names, e, || {
                format!("OMEGA({},{})", e + 1, e + 1)
            });
            names.push(format!("{base}_MIX{}", c + 1));
        }
        for &(c, s) in &mix.sigma_override_addr {
            let base = named_or(&params.sigma.names, s, || format!("SIGMA({})", s + 1));
            names.push(format!("{base}_MIX{}", c + 1));
        }
    }
    names
}

fn named_or(v: &[String], i: usize, fallback: impl FnOnce() -> String) -> String {
    match v.get(i) {
        Some(s) if !s.is_empty() => s.clone(),
        _ => fallback(),
    }
}

fn push_omega_names(names: &mut Vec<String>, om: &OmegaMatrix) {
    let n = om.dim();
    let diag_name = |i: usize| named_or(&om.eta_names, i, || format!("OMEGA({},{})", i + 1, i + 1));
    // Column-major lower triangle — mirrors `pack_params`. `lower_tri_iter` yields
    // only `(i,i)` when diagonal, so the `i == j` arm covers the diagonal case.
    for (i, j) in lower_tri_iter(n, om.diagonal) {
        if i == j {
            names.push(diag_name(i));
        } else {
            let ni = om.eta_names.get(i).filter(|s| !s.is_empty());
            let nj = om.eta_names.get(j).filter(|s| !s.is_empty());
            match (ni, nj) {
                (Some(a), Some(b)) => names.push(format!("{}~{}", a, b)),
                _ => names.push(format!("OMEGA({},{})", i + 1, j + 1)),
            }
        }
    }
}

/// Per-coordinate **natural / reporting-scale** values, ordered like
/// [`pack_params`]. Theta are natural values, Ω entries are variances
/// (diagonal) / covariances (off-diagonal); sigma are the stored **SD-scale**
/// values (`parse_parameters` `sqrt`s variance-scale input at parse time), and a
/// mixture sigma override reports on that same SD scale. This is the
/// back-transformed space the trace's `val:*` columns report.
pub fn coordinate_values(params: &ModelParameters) -> Vec<f64> {
    let mut v = coordinate_values_raw(
        &params.theta,
        &params.omega.matrix,
        params.omega.diagonal,
        &params.sigma.values,
        params.omega_iov.as_ref().map(|m| (&m.matrix, m.diagonal)),
    );
    // Mixture overrides (#977): Omega natural value = class variance, Sigma
    // natural value = class sigma (same scale the base sigma reports).
    if let Some(ref mix) = params.mixture {
        for &(c, e) in &mix.omega_override_addr {
            v.push(mix.omega[c].matrix[(e, e)]);
        }
        for &(c, s) in &mix.sigma_override_addr {
            v.push(mix.sigma[c].values[s]);
        }
    }
    v
}

/// Assemble the natural-scale coordinate vector directly from raw pieces, for
/// callers (e.g. SAEM) that hold parameters as loose matrices/vectors rather
/// than a `ModelParameters`. Order matches [`pack_params`].
pub fn coordinate_values_raw(
    theta: &[f64],
    omega_mat: &DMatrix<f64>,
    omega_diagonal: bool,
    sigma: &[f64],
    iov: Option<(&DMatrix<f64>, bool)>,
) -> Vec<f64> {
    let mut v = Vec::new();
    v.extend_from_slice(theta);
    push_omega_vals(&mut v, omega_mat, omega_diagonal);
    v.extend_from_slice(sigma);
    if let Some((m, diag)) = iov {
        push_omega_vals(&mut v, m, diag);
    }
    v
}

fn push_omega_vals(v: &mut Vec<f64>, m: &DMatrix<f64>, diagonal: bool) {
    for (i, j) in lower_tri_iter(m.nrows(), diagonal) {
        v.push(m[(i, j)]);
    }
}

/// Return initial ETA vector: warm-start if available, else mu_refs, else zeros.
pub fn get_eta_init(n_eta: usize, warm_start: Option<&[f64]>, mu_refs: Option<&[f64]>) -> Vec<f64> {
    if let Some(ws) = warm_start {
        ws.to_vec()
    } else if let Some(mu) = mu_refs {
        mu.to_vec()
    } else {
        vec![0.0; n_eta]
    }
}

/// Compute the mu_k shift vector from current theta for mu-referenced ETAs.
///
/// For each ETA that has a detected mu-reference, mu[i] = log(theta) or theta
/// depending on whether the relationship is log-transformed.  ETAs without a
/// mu-reference get mu[i] = 0 (no shift), preserving the standard behaviour.
/// When `enabled` is false, returns a zero vector (disables mu-referencing).
pub fn compute_mu_k(model: &CompiledModel, theta: &[f64], enabled: bool) -> Vec<f64> {
    if !enabled {
        return vec![0.0; model.n_eta];
    }
    let mut mu = vec![0.0; model.n_eta];
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if let Some(theta_idx) = model
                .theta_names
                .iter()
                .position(|n| n == &mu_ref.theta_name)
            {
                let theta_val = theta[theta_idx];
                mu[eta_idx] = if mu_ref.log_transformed {
                    theta_val.max(1e-10).ln()
                } else {
                    theta_val
                };
            }
        }
    }
    mu
}

/// Compute a scale vector for a packed log/Cholesky parameter vector.
///
/// Returns |v| for elements whose absolute value exceeds 0.1 (normalises
/// log-space parameters to be O(1) for the outer optimizer), and 1.0
/// otherwise. The threshold 0.1 is appropriate because a log-space value
/// near zero means the natural-scale parameter is near 1.0 — no scaling
/// needed there.
pub fn compute_scale(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| if v.abs() > 0.1 { v.abs() } else { 1.0 })
        .collect()
}

/// Divide each element of `x` by the corresponding scale factor.
/// `x_s = x / scale` — the representation seen by the outer optimizer.
pub fn apply_scale(x: &[f64], scale: &[f64]) -> Vec<f64> {
    x.iter().zip(scale).map(|(v, s)| v / s).collect()
}

/// Multiply each element of `x_scaled` by the corresponding scale factor.
/// `x = x_s * scale` — recovers the real packed vector.
pub fn remove_scale(x_scaled: &[f64], scale: &[f64]) -> Vec<f64> {
    x_scaled.iter().zip(scale).map(|(v, s)| v * s).collect()
}

/// Clamp a vector to box constraints
pub fn clamp_to_bounds(x: &mut [f64], bounds: &PackedBounds) {
    for i in 0..x.len() {
        x[i] = x[i].clamp(bounds.lower[i], bounds.upper[i]);
    }
}

// ===== Cholesky-Ω packing layout — single source of truth =====
// These express the column-major lower-triangle convention that `pack_params`
// (above) is the authority for. They were previously re-derived independently in
// `sens_outer_gradient` (lower_tri_entries / chol_pack / block_chol_full) and
// `api` (chol_lt_idx); centralised here so the packing order has exactly one
// definition. Pure integer/`f64` algebra moved verbatim — no result is reordered.

/// Column-major lower-triangle entry list `(row, col)` with `row >= col`,
/// matching `pack_params` order (diagonal → `(i, i)`).
pub(crate) fn lower_tri_entries(n: usize, diagonal: bool) -> Vec<(usize, usize)> {
    lower_tri_iter(n, diagonal).collect()
}

/// Non-allocating iterator over the column-major lower-triangle entries `(row, col)`
/// (`row >= col`), in `pack_params` order. `diagonal` restricts each column to its
/// single `(c, c)` entry. This is the single source of the `for j in 0..n { for i
/// in j..n }` packing convention: `pack_params`/`unpack_params` and the mask/bounds
/// walkers all iterate through it, so a change to the packing order is a one-place
/// edit. Returns an iterator (not a `Vec`) so the hot `pack`/`unpack` paths allocate
/// nothing.
pub(crate) fn lower_tri_iter(n: usize, diagonal: bool) -> impl Iterator<Item = (usize, usize)> {
    (0..n).flat_map(move |c| {
        let end = if diagonal { c + 1 } else { n };
        (c..end).map(move |r| (r, c))
    })
}

/// Number of packed entries for an Ω / Ω_iov of dimension `n`: `n` if `diagonal`
/// (variances only), else the full lower-triangle `n*(n+1)/2`. Single source of the
/// triangular-length formula that was re-derived at ~10 sites (api/covariance/output/
/// types/parameterization). Equals `lower_tri_iter(n, diagonal).count()`, without allocating.
#[inline]
pub(crate) fn omega_packed_len(n: usize, diagonal: bool) -> usize {
    if diagonal {
        n
    } else {
        n * (n + 1) / 2
    }
}

/// Flat index of `L[i,j]` (i ≥ j) in the column-major lower-triangle packing.
///
/// Layout: `for j in 0..n { for i in j..n { .. } }`, so column `j` starts at
/// offset `Σ_{k<j}(n−k) = j·n − j·(j−1)/2`.
#[inline]
pub(crate) fn chol_lt_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i >= j && i < n);
    let col_offset = if j == 0 { 0 } else { j * n - j * (j - 1) / 2 };
    col_offset + (i - j)
}

/// Map a sub-block natural symmetric Ω-gradient to packed Cholesky space:
/// `∂F/∂L = 2·M_sub·L` (L lower-triangular), with the diagonal log-chain
/// (`x_ii = ln L_ii ⇒ ×L_ii`) and raw off-diagonals — the same convention/order
/// as `pack_params`. Shared with `crate::estimation::agq`.
pub(crate) fn chol_pack(m_sub: &DMatrix<f64>, l: &DMatrix<f64>, diagonal: bool) -> Vec<f64> {
    let n = l.nrows();
    let gl = (m_sub * l).scale(2.0);
    // `lower_tri_iter` yields only `(i,i)` when diagonal, where the `i == j` arm
    // (diagonal log-chain `×L_ii`) applies; off-diagonals are raw.
    lower_tri_iter(n, diagonal)
        .map(|(i, j)| {
            if i == j {
                gl[(i, j)] * l[(i, j)]
            } else {
                gl[(i, j)]
            }
        })
        .collect()
}

/// Block-diagonal Cholesky factor `L_Σb = blkdiag(L_bsv, L_iov × K)` of the IOV
/// prior `Σ_b = Ω_bsv ⊕ K·Ω_iov`.
pub(crate) fn block_chol_full(
    l_bsv: &DMatrix<f64>,
    l_iov: &DMatrix<f64>,
    k: usize,
    n_eta: usize,
    n_iov: usize,
) -> DMatrix<f64> {
    let n = n_eta + k * n_iov;
    let mut l = DMatrix::zeros(n, n);
    for r in 0..n_eta {
        for c in 0..n_eta {
            l[(r, c)] = l_bsv[(r, c)];
        }
    }
    for kk in 0..k {
        let off = n_eta + kk * n_iov;
        for r in 0..n_iov {
            for c in 0..n_iov {
                l[(off + r, off + c)] = l_iov[(r, c)];
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lower_tri_entries_frozen_order() {
        assert_eq!(lower_tri_entries(1, false), vec![(0, 0)]);
        assert_eq!(lower_tri_entries(2, false), vec![(0, 0), (1, 0), (1, 1)]);
        assert_eq!(
            lower_tri_entries(3, false),
            vec![(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
        );
        assert_eq!(lower_tri_entries(3, true), vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_chol_lt_idx_roundtrips_lower_tri_entries() {
        for n in 1..=4 {
            for (pos, &(i, j)) in lower_tri_entries(n, false).iter().enumerate() {
                assert_eq!(chol_lt_idx(i, j, n), pos, "n={n} (i,j)=({i},{j})");
            }
        }
    }

    fn make_template() -> ModelParameters {
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["eta_cl".into(), "eta_v".into()]);
        let sigma = SigmaVector {
            values: vec![0.3],
            names: vec!["sigma_prop".into()],
        };
        ModelParameters {
            theta: vec![10.0, 100.0],
            theta_names: vec!["cl".into(), "v".into()],
            theta_lower: vec![0.01, 0.01],
            theta_upper: vec![1000.0, 10000.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false; 2],
            sigma,
            sigma_fixed: vec![false; 1],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        }
    }

    #[test]
    fn test_packed_len_diagonal() {
        let template = make_template();
        // 2 theta + 2 diagonal omega + 1 sigma = 5
        assert_eq!(packed_len(&template), 5);
    }

    #[test]
    fn test_pack_unpack_round_trip() {
        let template = make_template();
        let packed = pack_params(&template);
        assert_eq!(packed.len(), packed_len(&template));

        let recovered = unpack_params(&packed, &template);

        // Theta values should round-trip
        for (orig, rec) in template.theta.iter().zip(recovered.theta.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }

        // Omega diagonal should round-trip
        let n = template.omega.dim();
        for i in 0..n {
            assert_relative_eq!(
                template.omega.matrix[(i, i)],
                recovered.omega.matrix[(i, i)],
                epsilon = 1e-8
            );
        }

        // Sigma should round-trip
        for (orig, rec) in template
            .sigma
            .values
            .iter()
            .zip(recovered.sigma.values.iter())
        {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pack_values_are_log_transformed() {
        let template = make_template();
        let packed = pack_params(&template);
        // First packed value should be log(theta[0]) = log(10)
        assert_relative_eq!(packed[0], 10.0_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(packed[1], 100.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_pack_negative_lower_bound_uses_identity_packing() {
        // Regression: SAD_SCEN3/SAD_SCEN4 in the astra-testdata-simulator
        // benchmark have thetas like `THETA_CL_GAMMA(-0.8, -3.0, 3.0)` and
        // `THETA_AGE_CL(-0.01, -1.0, 1.0)`. The original `pack_params` ran
        // `th.max(1e-10).ln()` on every theta, silently clamping negative
        // values to 1e-10 and back-transforming through `exp()` so the
        // optimizer could never reach a sign-bearing optimum. SCEN4 was the
        // most visible: γ = -0.8 (truth) collapsed to ≈ 0 and the rest of
        // the fit drifted by 30-50% to compensate.
        //
        // Identity packing kicks in whenever the user-supplied `theta_lower`
        // allows negatives (i.e. < 0). Positive-only parameters keep their
        // log-scale conditioning.
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["eta_cl".into()]);
        let sigma = SigmaVector {
            values: vec![0.3],
            names: vec!["sigma_prop".into()],
        };
        let template = ModelParameters {
            theta: vec![5.0, -0.8, -0.01],
            theta_names: vec!["tvcl".into(), "gamma".into(), "age_eff".into()],
            theta_lower: vec![0.1, -3.0, -1.0],
            theta_upper: vec![100.0, 3.0, 1.0],
            theta_fixed: vec![false; 3],
            omega,
            omega_fixed: vec![false; 1],
            sigma,
            sigma_fixed: vec![false; 1],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        };
        let packed = pack_params(&template);
        // theta[0] is sign-constrained (lower=0.1) → log-packed.
        assert_relative_eq!(packed[0], 5.0_f64.ln(), epsilon = 1e-12);
        // theta[1] (lower=-3.0) and theta[2] (lower=-1.0) → identity-packed,
        // so the *negative* initial values survive the round-trip.
        assert_relative_eq!(packed[1], -0.8, epsilon = 1e-12);
        assert_relative_eq!(packed[2], -0.01, epsilon = 1e-12);

        let recovered = unpack_params(&packed, &template);
        assert_relative_eq!(recovered.theta[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(recovered.theta[1], -0.8, epsilon = 1e-12);
        assert_relative_eq!(recovered.theta[2], -0.01, epsilon = 1e-12);

        // Bounds packed in matching space: log for theta[0], identity for
        // the others. compute_bounds must agree with pack_params or
        // clamp_to_bounds will silently reject legal points.
        let bounds = compute_bounds(&template);
        assert_relative_eq!(bounds.lower[0], 0.1_f64.ln(), epsilon = 1e-12);
        assert_relative_eq!(bounds.upper[0], 100.0_f64.ln(), epsilon = 1e-12);
        assert_relative_eq!(bounds.lower[1], -3.0, epsilon = 1e-12);
        assert_relative_eq!(bounds.upper[1], 3.0, epsilon = 1e-12);
        assert_relative_eq!(bounds.lower[2], -1.0, epsilon = 1e-12);
        assert_relative_eq!(bounds.upper[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_bounds_dimensions() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        let expected_len = packed_len(&template);
        assert_eq!(bounds.lower.len(), expected_len);
        assert_eq!(bounds.upper.len(), expected_len);
    }

    #[test]
    fn test_bounds_lower_less_than_upper() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        for (lo, hi) in bounds.lower.iter().zip(bounds.upper.iter()) {
            assert!(lo < hi, "lower {} should be < upper {}", lo, hi);
        }
    }

    #[test]
    fn test_clamp_to_bounds() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        let mut x = vec![100.0; packed_len(&template)]; // way above upper bounds
        clamp_to_bounds(&mut x, &bounds);
        for (val, hi) in x.iter().zip(bounds.upper.iter()) {
            assert!(*val <= *hi + 1e-12);
        }
    }

    #[test]
    fn test_clamp_to_bounds_below() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        let mut x = vec![-100.0; packed_len(&template)]; // way below lower bounds
        clamp_to_bounds(&mut x, &bounds);
        for (val, lo) in x.iter().zip(bounds.lower.iter()) {
            assert!(*val >= *lo - 1e-12);
        }
    }

    fn make_block_template() -> ModelParameters {
        // Build a 2x2 block omega with covariance
        let mut m = DMatrix::zeros(2, 2);
        m[(0, 0)] = 0.09; // var(eta_cl)
        m[(1, 1)] = 0.04; // var(eta_v)
        m[(0, 1)] = 0.02; // cov(eta_cl, eta_v)
        m[(1, 0)] = 0.02;
        let omega = OmegaMatrix::from_matrix(m, vec!["eta_cl".into(), "eta_v".into()], false);
        let sigma = SigmaVector {
            values: vec![0.3],
            names: vec!["sigma_prop".into()],
        };
        ModelParameters {
            theta: vec![10.0, 100.0],
            theta_names: vec!["cl".into(), "v".into()],
            theta_lower: vec![0.01, 0.01],
            theta_upper: vec![1000.0, 10000.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false; 2],
            sigma,
            sigma_fixed: vec![false; 1],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        }
    }

    #[test]
    fn test_packed_len_block() {
        let template = make_block_template();
        // 2 theta + 3 omega (lower triangle of 2x2) + 1 sigma = 6
        assert_eq!(packed_len(&template), 6);
    }

    /// 3×3 mixed Ω: a 2×2 block on (CL,V) plus a separate diagonal KA. The
    /// cross-block elements (KA,CL)/(KA,V) are structural zeros (`free_mask ==
    /// false`).
    fn make_block_plus_diag_omega() -> OmegaMatrix {
        let mut m = DMatrix::zeros(3, 3);
        m[(0, 0)] = 0.09;
        m[(1, 1)] = 0.04;
        m[(2, 2)] = 0.30;
        m[(0, 1)] = 0.02;
        m[(1, 0)] = 0.02;
        let mut fm = DMatrix::from_element(3, 3, false);
        // diagonal + the (CL,V) block are free; the cross-block off-diagonals
        // (2,0)/(2,1) (and their transposes) stay false → structural zeros.
        for i in 0..3 {
            fm[(i, i)] = true;
        }
        fm[(0, 1)] = true;
        fm[(1, 0)] = true;
        OmegaMatrix::from_matrix_with_mask(
            m,
            vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
            false,
            fm,
        )
    }

    #[test]
    fn test_omega_structural_zero_mask_block_plus_diagonal() {
        // 2 theta + block+diag Ω (col-major lower-tri: (0,0)(1,0)(2,0)(1,1)(2,1)(2,2))
        // + 1 sigma. Structural zeros are (2,0) and (2,1).
        let template = ModelParameters {
            theta: vec![1.0, 2.0],
            theta_names: vec!["a".into(), "b".into()],
            theta_lower: vec![0.0, 0.0],
            theta_upper: vec![10.0, 10.0],
            theta_fixed: vec![false, false],
            omega: make_block_plus_diag_omega(),
            omega_fixed: vec![false, false, false],
            sigma: SigmaVector {
                values: vec![0.3],
                names: vec!["s".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        };
        let mask = omega_structural_zero_mask(&template);
        assert_eq!(mask.len(), packed_len(&template)); // 2 + 6 + 1 = 9
        let n_theta = 2;
        // omega packed offsets: (0,0)=0 (1,0)=1 (2,0)=2 (1,1)=3 (2,1)=4 (2,2)=5
        let expected_true = [n_theta + 2, n_theta + 4]; // (2,0) and (2,1)
        for (i, &m) in mask.iter().enumerate() {
            assert_eq!(
                m,
                expected_true.contains(&i),
                "mask[{i}] should be {}",
                expected_true.contains(&i)
            );
        }
    }

    #[test]
    fn test_omega_structural_zero_mask_diagonal_is_all_false() {
        // Pure diagonal Ω has no off-diagonals → nothing structural-zero.
        let template = make_template();
        let mask = omega_structural_zero_mask(&template);
        assert_eq!(mask.len(), packed_len(&template));
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_omega_structural_zero_mask_full_block_is_all_false() {
        // A fully-free 2×2 block has no structural zeros.
        let template = make_block_template();
        let mask = omega_structural_zero_mask(&template);
        assert_eq!(mask.len(), packed_len(&template));
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_omega_structural_zero_mask_block_iov() {
        // Diagonal BSV (1 eta) + sigma, then a block+diagonal Ω_IOV. The IOV
        // structural zeros must be marked in the IOV region of the packed vector.
        // Layout: theta(1) + bsvΩ(1) + sigma(1) + iovΩ(6) = 9.
        //   iov packed offset 3: (0,0)=3 (1,0)=4 (2,0)=5 (1,1)=6 (2,1)=7 (2,2)=8
        let template = ModelParameters {
            theta: vec![1.0],
            theta_names: vec!["a".into()],
            theta_lower: vec![0.0],
            theta_upper: vec![10.0],
            theta_fixed: vec![false],
            omega: OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.3],
                names: vec!["s".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(make_block_plus_diag_omega()),
            kappa_fixed: vec![false, false, false],
            mixture: None,
        };
        let mask = omega_structural_zero_mask(&template);
        assert_eq!(mask.len(), packed_len(&template)); // 1 + 1 + 1 + 6 = 9
        let expected_true = [5usize, 7]; // iov (2,0) and (2,1)
        for (i, &m) in mask.iter().enumerate() {
            assert_eq!(
                m,
                expected_true.contains(&i),
                "mask[{i}] should be {}",
                expected_true.contains(&i)
            );
        }
    }

    #[test]
    fn test_pack_unpack_block_round_trip() {
        let template = make_block_template();
        let packed = pack_params(&template);
        assert_eq!(packed.len(), packed_len(&template));

        let recovered = unpack_params(&packed, &template);

        // Theta round-trip
        for (orig, rec) in template.theta.iter().zip(recovered.theta.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }

        // Full omega matrix round-trip (including off-diagonals)
        let n = template.omega.dim();
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    template.omega.matrix[(i, j)],
                    recovered.omega.matrix[(i, j)],
                    epsilon = 1e-6
                );
            }
        }

        // Sigma round-trip
        for (orig, rec) in template
            .sigma
            .values
            .iter()
            .zip(recovered.sigma.values.iter())
        {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }
    }

    // ── coordinate names / values (trace #640) ──────────────────────────────

    #[test]
    fn test_coordinate_names_diagonal() {
        // Layout: theta(cl,v), diagonal Ω(eta_cl,eta_v), sigma(sigma_prop).
        let t = make_template();
        let names = coordinate_names(&t);
        assert_eq!(names, vec!["cl", "v", "eta_cl", "eta_v", "sigma_prop"]);
        assert_eq!(names.len(), packed_len(&t));
    }

    #[test]
    fn test_coordinate_values_diagonal_are_natural_scale() {
        // Values: theta as-is, Ω diagonal as variances, sigma as variance.
        let t = make_template();
        let v = coordinate_values(&t);
        assert_eq!(v.len(), packed_len(&t));
        assert_relative_eq!(v[0], 10.0, epsilon = 1e-12); // cl
        assert_relative_eq!(v[1], 100.0, epsilon = 1e-12); // v
        assert_relative_eq!(v[2], 0.09, epsilon = 1e-12); // var(eta_cl)
        assert_relative_eq!(v[3], 0.04, epsilon = 1e-12); // var(eta_v)
        assert_relative_eq!(v[4], 0.3, epsilon = 1e-12); // sigma
    }

    #[test]
    fn test_coordinate_names_block_off_diagonal() {
        // Block Ω off-diagonal couples row~col eta in packed (col-major) order:
        // (0,0)=eta_cl, (1,0)=eta_v~eta_cl, (1,1)=eta_v.
        let t = make_block_template();
        let names = coordinate_names(&t);
        assert_eq!(
            names,
            vec!["cl", "v", "eta_cl", "eta_v~eta_cl", "eta_v", "sigma_prop"]
        );
    }

    #[test]
    fn test_coordinate_values_block_off_diagonal_is_covariance() {
        let t = make_block_template();
        let v = coordinate_values(&t);
        // Packed omega order: var(cl)=0.09, cov=0.02, var(v)=0.04.
        assert_relative_eq!(v[2], 0.09, epsilon = 1e-12);
        assert_relative_eq!(v[3], 0.02, epsilon = 1e-12);
        assert_relative_eq!(v[4], 0.04, epsilon = 1e-12);
    }

    #[test]
    fn test_coordinate_names_fallbacks_when_unnamed() {
        // Empty declared names → NONMEM-style THETA1 / OMEGA(2,1) / SIGMA(1).
        let mut m = DMatrix::zeros(2, 2);
        m[(0, 0)] = 0.09;
        m[(1, 1)] = 0.04;
        m[(0, 1)] = 0.02;
        m[(1, 0)] = 0.02;
        let omega = OmegaMatrix::from_matrix(m, vec![String::new(), String::new()], false);
        let t = ModelParameters {
            theta: vec![1.0],
            theta_names: vec![String::new()],
            theta_lower: vec![0.0],
            theta_upper: vec![10.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false, false],
            sigma: SigmaVector {
                values: vec![0.3],
                names: vec![String::new()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        };
        let names = coordinate_names(&t);
        assert_eq!(
            names,
            vec![
                "THETA1",
                "OMEGA(1,1)",
                "OMEGA(2,1)",
                "OMEGA(2,2)",
                "SIGMA(1)"
            ]
        );
    }

    #[test]
    fn test_coordinate_names_values_include_iov() {
        // IOV coordinates append after sigma, mirroring pack_params.
        let t = make_iov_template();
        let names = coordinate_names(&t);
        assert_eq!(names, vec!["TVCL", "ETA_CL", "PROP_ERR", "KAPPA_CL"]);
        let v = coordinate_values(&t);
        assert_eq!(v.len(), packed_len(&t));
        assert_relative_eq!(v[3], 0.01, epsilon = 1e-12); // var(kappa_cl)
    }

    #[test]
    fn test_coordinate_values_matches_packed_len_for_all_shapes() {
        for t in [make_template(), make_block_template(), make_iov_template()] {
            assert_eq!(coordinate_values(&t).len(), packed_len(&t));
            assert_eq!(coordinate_names(&t).len(), packed_len(&t));
        }
    }

    #[test]
    fn test_block_omega_not_diagonal() {
        let template = make_block_template();
        assert!(!template.omega.diagonal);
    }

    // ── mu-referencing helpers ──────────────────────────────────────────

    use crate::types::{
        BloqMethod, CompiledModel, ErrorModel, GradientMethod, MuRef, PkModel, PkParams,
        ScalingSpec,
    };
    use std::collections::HashMap;

    /// Build a minimal CompiledModel with the given mu-refs. Only fields
    /// that `compute_mu_k` actually reads need to be meaningful; the rest
    /// are filled with defaults.
    fn make_model_with_mu_refs(mu_refs: Vec<(&str, &str, bool)>) -> CompiledModel {
        let theta_names: Vec<String> = vec!["TVCL".into(), "TVV".into(), "TVKA".into()];
        let eta_names: Vec<String> = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
        let mut refs = HashMap::new();
        for (eta, theta, log_t) in mu_refs {
            refs.insert(
                eta.to_string(),
                MuRef {
                    theta_name: theta.to_string(),
                    log_transformed: log_t,
                },
            );
        }
        let omega = OmegaMatrix::from_diagonal(&[0.09, 0.04, 0.30], eta_names.clone());
        let sigma = SigmaVector {
            values: vec![0.02],
            names: vec!["PROP_ERR".into()],
        };
        let default_params = ModelParameters {
            theta: vec![0.2, 10.0, 1.5],
            theta_names: theta_names.clone(),
            theta_lower: vec![0.001, 0.1, 0.01],
            theta_upper: vec![10.0, 500.0, 50.0],
            theta_fixed: vec![false; 3],
            omega,
            omega_fixed: vec![false; 3],
            sigma,
            sigma_fixed: vec![false; 1],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        };
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|_, _, _, _t: f64| PkParams::default()),
            n_theta: 3,
            n_eta: 3,
            n_epsilon: 1,
            theta_names,
            eta_names,
            indiv_param_names: vec!["CL".into(), "V".into(), "KA".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            theta_blocks: crate::types::ThetaBlocks::empty(),
            default_params,
            omega_init_as_sd: vec![false; 3],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            kappa_weights: Vec::new(),
            mu_refs: refs,
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1, 4],

            eta_map: (0..3).map(|i| i as i32).collect(),

            pk_idx_f64: vec![0.0, 1.0, 4.0],

            sel_flat: {
                let mut v = vec![0.0f64; 3 * 3];
                for i in 0..3 {
                    v[i * 3 + i] = 1.0;
                }
                v
            },
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::default(),
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            n_kappa: 0,
            kappa_names: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
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

    #[test]
    fn test_compute_mu_k_no_refs_returns_zeros() {
        // Model with no detected mu-refs → every shift is zero, even when enabled.
        let model = make_model_with_mu_refs(vec![]);
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], true);
        assert_eq!(mu.len(), 3);
        for v in &mu {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_compute_mu_k_disabled_returns_zeros() {
        // `enabled = false` must short-circuit even if mu-refs exist.
        let model = make_model_with_mu_refs(vec![("ETA_CL", "TVCL", true), ("ETA_V", "TVV", true)]);
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], false);
        assert_eq!(mu, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_mu_k_log_transformed() {
        // log-transformed mu-ref (exp / multiplicative pattern) → mu = ln(theta).
        let model = make_model_with_mu_refs(vec![("ETA_CL", "TVCL", true), ("ETA_V", "TVV", true)]);
        let theta = vec![0.2_f64, 10.0_f64, 1.5_f64];
        let mu = compute_mu_k(&model, &theta, true);
        assert_relative_eq!(mu[0], 0.2_f64.ln(), epsilon = 1e-12);
        assert_relative_eq!(mu[1], 10.0_f64.ln(), epsilon = 1e-12);
        // ETA_KA has no mu-ref → zero shift.
        assert_eq!(mu[2], 0.0);
    }

    #[test]
    fn test_compute_mu_k_additive_uses_theta_directly() {
        // Additive pattern (THETA + ETA) → mu = theta (no log).
        let model = make_model_with_mu_refs(vec![("ETA_CL", "TVCL", false)]);
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], true);
        assert_relative_eq!(mu[0], 0.2, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_mu_k_clamps_log_of_nonpositive_theta() {
        // ln() of a non-positive theta would be -inf or NaN — the
        // implementation clamps to 1e-10 first. Verify that guard holds.
        let model = make_model_with_mu_refs(vec![("ETA_CL", "TVCL", true)]);
        let mu = compute_mu_k(&model, &[0.0, 10.0, 1.5], true);
        assert!(mu[0].is_finite());
        assert_relative_eq!(mu[0], 1e-10_f64.ln(), epsilon = 1e-6);
    }

    #[test]
    fn test_compute_mu_k_unknown_theta_name_is_ignored() {
        // If the recorded theta_name doesn't exist in theta_names
        // (shouldn't happen in practice, but guard is real), shift stays zero.
        let mut model = make_model_with_mu_refs(vec![]);
        model.mu_refs.insert(
            "ETA_CL".into(),
            MuRef {
                theta_name: "NON_EXISTENT".into(),
                log_transformed: true,
            },
        );
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], true);
        assert_eq!(mu, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_get_eta_init_warm_start_preferred() {
        // Warm start wins over mu_refs.
        let warm = vec![0.5, -0.1, 0.2];
        let mu = vec![1.0, 1.0, 1.0];
        let out = get_eta_init(3, Some(&warm), Some(&mu));
        assert_eq!(out, warm);
    }

    #[test]
    fn test_get_eta_init_falls_back_to_mu_refs() {
        // No warm start → use mu_refs.
        let mu = vec![0.1, 0.2, 0.3];
        let out = get_eta_init(3, None, Some(&mu));
        assert_eq!(out, mu);
    }

    #[test]
    fn test_get_eta_init_falls_back_to_zeros() {
        // Nothing provided → zeros of the requested length.
        let out = get_eta_init(4, None, None);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn test_compute_bounds_block_dimensions() {
        let template = make_block_template();
        let bounds = compute_bounds(&template);
        let expected_len = packed_len(&template);
        assert_eq!(bounds.lower.len(), expected_len);
        assert_eq!(bounds.upper.len(), expected_len);
    }

    // ── FIX-parameter behavior ─────────────────────────────────────────────

    #[test]
    fn test_fixed_theta_pins_bounds_to_packed_value() {
        let mut template = make_template();
        template.theta_fixed[0] = true; // fix first theta (TVCL = 10)
        let bounds = compute_bounds(&template);
        let packed = pack_params(&template);
        // Lower == upper == packed value (log-space) for the fixed theta
        assert_relative_eq!(bounds.lower[0], packed[0], epsilon = 1e-12);
        assert_relative_eq!(bounds.upper[0], packed[0], epsilon = 1e-12);
        // Free theta still has a nontrivial box
        assert!(bounds.lower[1] < bounds.upper[1]);
    }

    #[test]
    fn test_fixed_sigma_pins_bounds() {
        let mut template = make_template();
        template.sigma_fixed[0] = true;
        let bounds = compute_bounds(&template);
        let packed = pack_params(&template);
        let sigma_idx = packed.len() - 1;
        assert_relative_eq!(bounds.lower[sigma_idx], packed[sigma_idx], epsilon = 1e-12);
        assert_relative_eq!(bounds.upper[sigma_idx], packed[sigma_idx], epsilon = 1e-12);
    }

    #[test]
    fn test_fixed_omega_diagonal_pins_bounds() {
        let mut template = make_template();
        template.omega_fixed[0] = true; // fix eta_cl variance
        let bounds = compute_bounds(&template);
        let packed = pack_params(&template);
        let omega0_idx = template.theta.len(); // first omega entry after theta
        assert_relative_eq!(
            bounds.lower[omega0_idx],
            packed[omega0_idx],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            bounds.upper[omega0_idx],
            packed[omega0_idx],
            epsilon = 1e-12
        );
        // The other omega (free) still has a real interval
        assert!(bounds.lower[omega0_idx + 1] < bounds.upper[omega0_idx + 1]);
    }

    #[test]
    fn test_fixed_block_omega_pins_all_cholesky_entries() {
        // 2×2 block, both etas fixed => every Cholesky entry pinned.
        let mut template = make_block_template();
        template.omega_fixed = vec![true, true];
        let bounds = compute_bounds(&template);
        let packed = pack_params(&template);
        // Theta entries 0,1 are free; omega entries 2,3,4 are the Cholesky
        // lower-triangle (L11, L21, L22); sigma entry 5 is free.
        for i in 2..=4 {
            assert_relative_eq!(bounds.lower[i], packed[i], epsilon = 1e-12);
            assert_relative_eq!(bounds.upper[i], packed[i], epsilon = 1e-12);
        }
        assert!(bounds.lower[0] < bounds.upper[0]); // theta 0 free
        assert!(bounds.lower[5] < bounds.upper[5]); // sigma free
    }

    // ── scaling helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_scale_above_threshold() {
        // |v| > 0.1 → scale = |v|
        let x = vec![2.3, -4.5, 0.0, 0.05, -0.11];
        let s = compute_scale(&x);
        assert_relative_eq!(s[0], 2.3, epsilon = 1e-12);
        assert_relative_eq!(s[1], 4.5, epsilon = 1e-12);
        assert_relative_eq!(s[2], 1.0, epsilon = 1e-12); // 0.0 → 1.0
        assert_relative_eq!(s[3], 1.0, epsilon = 1e-12); // 0.05 ≤ 0.1 → 1.0
        assert_relative_eq!(s[4], 0.11, epsilon = 1e-12); // 0.11 > 0.1 → 0.11
    }

    #[test]
    fn test_apply_remove_scale_round_trip() {
        let x = vec![6.9, -2.3, 0.0, 1.5];
        let s = compute_scale(&x);
        let xs = apply_scale(&x, &s);
        let xr = remove_scale(&xs, &s);
        for (orig, rec) in x.iter().zip(xr.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_apply_scale_normalises_to_unit_magnitude() {
        // After apply_scale, all elements with |v| > 0.1 should have |x_s| ≈ 1
        let x = vec![6.9, -2.3, 1.5, -0.05];
        let s = compute_scale(&x);
        let xs = apply_scale(&x, &s);
        assert_relative_eq!(xs[0].abs(), 1.0, epsilon = 1e-12); // 6.9/6.9
        assert_relative_eq!(xs[1].abs(), 1.0, epsilon = 1e-12); // -2.3/2.3
        assert_relative_eq!(xs[2].abs(), 1.0, epsilon = 1e-12); // 1.5/1.5
        assert_relative_eq!(xs[3], -0.05, epsilon = 1e-12); // |v|≤0.1 → scale=1
    }

    #[test]
    fn test_packed_fixed_mask_length() {
        let template = make_template();
        let mask = packed_fixed_mask(&template);
        assert_eq!(mask.len(), packed_len(&template));
        assert!(mask.iter().all(|&b| !b)); // default: nothing fixed
    }

    fn make_iov_template() -> ModelParameters {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.01], vec!["KAPPA_CL".into()]);
        let sigma = SigmaVector {
            values: vec![0.02],
            names: vec!["PROP_ERR".into()],
        };
        ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.01],
            theta_upper: vec![100.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false],
            sigma,
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false],
            mixture: None,
        }
    }

    #[test]
    fn test_packed_len_with_kappa() {
        let template = make_iov_template();
        // 1 theta + 1 bsv omega diag + 1 sigma + 1 kappa omega diag = 4
        assert_eq!(packed_len(&template), 4);
    }

    #[test]
    fn test_pack_unpack_with_omega_iov() {
        let template = make_iov_template();
        let packed = pack_params(&template);
        assert_eq!(packed.len(), packed_len(&template));

        let recovered = unpack_params(&packed, &template);

        // Theta round-trips
        assert_relative_eq!(template.theta[0], recovered.theta[0], epsilon = 1e-8);

        // BSV omega diagonal round-trips
        assert_relative_eq!(
            template.omega.matrix[(0, 0)],
            recovered.omega.matrix[(0, 0)],
            epsilon = 1e-8
        );

        // IOV omega diagonal round-trips
        let iov_orig = template.omega_iov.as_ref().unwrap().matrix[(0, 0)];
        let iov_rec = recovered.omega_iov.as_ref().unwrap().matrix[(0, 0)];
        assert_relative_eq!(iov_orig, iov_rec, epsilon = 1e-8);
    }

    #[test]
    fn test_unpack_omega_iov_depends_only_on_vector_and_structure() {
        // The mechanism behind the IOV `run_covariance` bit-exactness (#823):
        // `unpack_params` rebuilds `omega_iov` from the packed vector and the
        // template's *structure* (diagonal flag, names, free_mask) alone — the
        // template's numeric Ω_IOV **values** never leak in. So the inline
        // covariance step (template = the fit's init params) and the standalone
        // step (template = `fitted_params_from_result`, carrying the *converged*
        // Ω_IOV) reconstruct a byte-identical Ω_IOV from the same `packed_estimate`,
        // even though their templates hold different variances. The diagonal-IOV
        // branch runs `OmegaMatrix::from_diagonal` (square-then-re-decompose)
        // rather than the BSV's `from_chol_factor`, so pin every cached field
        // (`matrix`, `chol`, `inv`, `log_det`) it derives, not just the variance.
        let t_init = make_iov_template(); // IOV variance 0.01

        // Structurally identical, different values (variance 0.05, and shifted
        // theta/omega/sigma). `theta_lower` must match `t_init` — it drives the
        // log-vs-identity packing decision, part of the "structure".
        let mut t_conv = make_iov_template();
        t_conv.theta = vec![7.3];
        t_conv.omega = OmegaMatrix::from_diagonal(&[0.21], vec!["ETA_CL".into()]);
        t_conv.sigma.values = vec![0.11];
        t_conv.omega_iov = Some(OmegaMatrix::from_diagonal(&[0.05], vec!["KAPPA_CL".into()]));

        // A packed vector at a converged-ish point (not `pack(t_init)`), so the
        // unpacked Ω_IOV is a genuine reconstruction, not an identity round-trip.
        let v = pack_params(&t_conv);
        assert_eq!(v.len(), packed_len(&t_init));

        let from_init = unpack_params(&v, &t_init);
        let from_conv = unpack_params(&v, &t_conv);

        let a = from_init.omega_iov.as_ref().unwrap();
        let b = from_conv.omega_iov.as_ref().unwrap();
        // Bit-for-bit on every cached field — this is the `1e-12` the fit-level
        // parity test observes, reduced to its root cause.
        assert_eq!(
            a.matrix, b.matrix,
            "Ω_IOV matrix must not depend on template values"
        );
        assert_eq!(
            a.chol, b.chol,
            "Ω_IOV chol must not depend on template values"
        );
        assert_eq!(a.inv, b.inv, "Ω_IOV inv must not depend on template values");
        assert_eq!(
            a.log_det.to_bits(),
            b.log_det.to_bits(),
            "Ω_IOV log_det must not depend on template values"
        );
    }

    #[test]
    fn test_fixed_kappa_pins_bounds() {
        let mut template = make_iov_template();
        template.kappa_fixed[0] = true;
        let bounds = compute_bounds(&template);
        let packed = pack_params(&template);
        // kappa is the last packed element
        let kappa_idx = packed.len() - 1;
        assert_relative_eq!(bounds.lower[kappa_idx], packed[kappa_idx], epsilon = 1e-12);
        assert_relative_eq!(bounds.upper[kappa_idx], packed[kappa_idx], epsilon = 1e-12);
    }

    #[test]
    fn test_packed_fixed_mask_with_kappa() {
        let mut template = make_iov_template();
        template.kappa_fixed[0] = true;
        let mask = packed_fixed_mask(&template);
        assert_eq!(mask.len(), packed_len(&template));
        assert!(mask[mask.len() - 1]); // kappa is fixed
        assert!(!mask[0]); // theta is free
    }

    #[test]
    fn test_packed_fixed_mask_block_off_diagonal() {
        // One eta fixed, the other free. The whole row/col of a fixed eta is
        // pinned — this keeps the fixed eta uncorrelated with free etas and
        // prevents SAEM's closed-form omega M-step from breaking PD.
        let mut template = make_block_template();
        template.omega_fixed = vec![true, false];
        let mask = packed_fixed_mask(&template);
        // Layout: theta(0,1), omega-chol(2=L11, 3=L21, 4=L22), sigma(5)
        assert!(mask[2]); // L11 (eta0 diagonal) — fixed
        assert!(mask[3]); // L21 (couples eta0-fixed to eta1) — pinned
        assert!(!mask[4]); // L22 (eta1 diagonal) — free
    }

    // ── block_kappa (Option B) ─────────────────────────────────────────────

    fn make_block_kappa_iov_template() -> ModelParameters {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        // 2×2 block kappa: [[0.01, 0.002], [0.002, 0.005]]
        // Build via Cholesky like OmegaMatrix::from_diagonal but full.
        use nalgebra::DMatrix;
        let mut mat = DMatrix::zeros(2, 2);
        mat[(0, 0)] = 0.01;
        mat[(0, 1)] = 0.002;
        mat[(1, 0)] = 0.002;
        mat[(1, 1)] = 0.005;
        let _chol = mat.clone().cholesky().unwrap().l();
        let omega_iov =
            OmegaMatrix::from_matrix(mat, vec!["KAPPA_CL".into(), "KAPPA_V".into()], false);
        let sigma = SigmaVector {
            values: vec![0.02],
            names: vec!["PROP_ERR".into()],
        };
        ModelParameters {
            theta: vec![0.2],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.01],
            theta_upper: vec![100.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false],
            sigma,
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false, false],
            mixture: None,
        }
    }

    #[test]
    fn test_packed_len_block_kappa() {
        let template = make_block_kappa_iov_template();
        // 1 theta + 1 bsv omega diag + 1 sigma + 3 block-kappa chol entries = 6
        assert_eq!(packed_len(&template), 6);
    }

    #[test]
    fn test_pack_unpack_block_kappa_round_trip() {
        let template = make_block_kappa_iov_template();
        let packed = pack_params(&template);
        assert_eq!(packed.len(), packed_len(&template));

        let recovered = unpack_params(&packed, &template);
        let iov_orig = template.omega_iov.as_ref().unwrap();
        let iov_rec = recovered.omega_iov.as_ref().unwrap();

        assert!(!iov_rec.diagonal);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    iov_orig.matrix[(i, j)],
                    iov_rec.matrix[(i, j)],
                    epsilon = 1e-8
                );
            }
        }
    }

    #[test]
    fn test_packed_fixed_mask_block_kappa() {
        let mut template = make_block_kappa_iov_template();
        // Fix the first kappa — its whole row/col in the Cholesky should be pinned.
        template.kappa_fixed = vec![true, false];
        let mask = packed_fixed_mask(&template);
        assert_eq!(mask.len(), packed_len(&template));
        // IOV chol layout (after theta+omega+sigma): L11, L21, L22
        let iov_start = 1 + 1 + 1; // theta + bsv diag + sigma
        assert!(mask[iov_start]); // L11 — kappa_fixed[0]=true
        assert!(mask[iov_start + 1]); // L21 — kappa_fixed[0]||kappa_fixed[1]=true
        assert!(!mask[iov_start + 2]); // L22 — kappa_fixed[1]=false
    }

    #[test]
    fn test_block_kappa_bounds_off_diagonal() {
        let template = make_block_kappa_iov_template();
        let bounds = compute_bounds(&template);
        assert_eq!(bounds.lower.len(), packed_len(&template));
        // IOV chol layout after theta+omega+sigma: L11, L21, L22
        let iov_start = 1 + 1 + 1;
        assert_relative_eq!(bounds.lower[iov_start], -6.0, epsilon = 1e-12); // L11 diag
        assert_relative_eq!(bounds.lower[iov_start + 1], -10.0, epsilon = 1e-12); // L21 off-diag
        assert_relative_eq!(bounds.lower[iov_start + 2], -6.0, epsilon = 1e-12);
        // L22 diag
    }
}
