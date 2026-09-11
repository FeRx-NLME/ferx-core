//! Per-parameter frequentist priors — penalized ML / MAP point estimation (#254).
//!
//! A prior is declared next to the parameter it constrains as a central value
//! plus a relative uncertainty (`prior(0.15, rse = 25%)`), and contributes a
//! quadratic penalty to the objective:
//!
//! ```text
//! OFV_total = OFV_data + Σ_k ((x_k − m_k) / s_k)²
//! ```
//!
//! # Why the penalty lives in packed space
//!
//! `x` here is the **packed** parameter vector — the optimizer's own
//! coordinates ([`pack_params`](super::parameterization::pack_params)), not the
//! natural values. Three things follow, and they are the whole reason this
//! module is 200 lines rather than a new sensitivity kernel:
//!
//! - The gradient `2(x−m)/s²` and the Hessian `2/s²` are exact and closed form,
//!   and the Hessian is a **constant** diagonal. Nothing here goes through
//!   `sens/`, so no `Dual2` instantiation and no added monomorphization on a
//!   compile path that is already 93% LLVM (#971).
//! - The prior family falls out of the packing rather than being a second
//!   convention: a θ with a non-negative lower bound packs as `ln θ`, so a
//!   normal prior there *is* a lognormal prior on θ — which is exactly what the
//!   issue asks for ("if the parameter is estimated on the log scale, apply the
//!   prior on that scale"). An Ω diagonal packs as `ln(chol) = ln(SD) =
//!   ½·ln(variance)`, so a normal prior there is a lognormal prior on the
//!   variance, positive and symmetric with no matrix algebra.
//! - Every consumer — the outer objective, the Gauss-Newton and trust-region
//!   optimizers, and the covariance step — already works in packed space, so
//!   there is one form of the penalty rather than one per caller.
//!
//! The coupling that buys: **a θ's prior family is decided by its declared lower
//! bound**, so changing `theta HILL(1.0, 0.0, 5.0)` to `theta HILL(1.0, -5.0,
//! 5.0)` silently moves its prior from lognormal to normal. That is real, and
//! the mitigation is visibility rather than cleverness — [`PriorSummary`]
//! reports the realised family and the implied natural-scale 95% interval for
//! every priored parameter, so the flip shows up in the fit report instead of
//! having to be inferred from the model file.
//!
//! # Scale convention
//!
//! **The prior's central value and spread are always on the scale the parameter
//! was declared on.** `omega ETA_CL ~ 0.09 prior(0.09, rse = 40%)` reads its
//! RSE off the *variance* row of a paper's parameter table; under `(sd)` both
//! the declaration and the prior are SDs. The user never converts anything and
//! never meets a second convention.

use crate::estimation::parameterization::{
    coordinate_kinds, coordinate_names, packed_fixed_mask, packed_segments, theta_packs_log,
    PackedCoordKind,
};
use crate::types::{CompiledModel, ModelParameters, PriorSpread, PriorSummary};

/// How a coordinate's **declared** value maps onto its packed value.
///
/// This is the single derivation of "what scale is this prior on", read by the
/// mean, the spread, the reporting inverse and the 95% interval alike. Keeping
/// it in one place is what stops the mean and the spread from disagreeing about
/// whether a variance-declared Ω is on the log-SD or the log-variance scale —
/// a disagreement that would be a factor of two and would still look plausible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PriorScale {
    /// `packed = value`. A θ whose declared lower bound is negative.
    Identity,
    /// `packed = ln(value)`. A log-packed θ, or an Ω/Σ/κ declared `(sd)`.
    Log,
    /// `packed = ½·ln(value)`. An Ω/Σ/κ declared on the (default) variance
    /// scale, whose packed coordinate is `ln(SD)`.
    HalfLog,
}

impl PriorScale {
    /// Declared value → packed coordinate.
    pub(crate) fn to_packed(self, value: f64) -> f64 {
        match self {
            PriorScale::Identity => value,
            PriorScale::Log => value.ln(),
            PriorScale::HalfLog => 0.5 * value.ln(),
        }
    }

    /// Packed coordinate → declared value. Exact inverse of [`Self::to_packed`].
    pub(crate) fn from_packed(self, packed: f64) -> f64 {
        match self {
            PriorScale::Identity => packed,
            PriorScale::Log => packed.exp(),
            PriorScale::HalfLog => (2.0 * packed).exp(),
        }
    }

    /// `d(packed) / d(ln value)` — the factor that carries a spread expressed
    /// on the declared log scale onto the packed scale. `Identity` has no log
    /// scale of its own and never uses this.
    fn log_spread_factor(self) -> f64 {
        match self {
            PriorScale::Identity => 1.0,
            PriorScale::Log => 1.0,
            PriorScale::HalfLog => 0.5,
        }
    }

    /// `true` when the parameter is strictly positive on its declared scale, so
    /// a non-positive prior central value is a declaration error rather than a
    /// legitimate point on the real line.
    fn requires_positive(self) -> bool {
        !matches!(self, PriorScale::Identity)
    }

    /// Human-readable family name for the fit report.
    fn family(self) -> &'static str {
        match self {
            PriorScale::Identity => "normal",
            PriorScale::Log | PriorScale::HalfLog => "lognormal",
        }
    }
}

/// One resolved prior: a packed coordinate plus the normal `(m, s)` that
/// governs it there.
#[derive(Debug, Clone)]
pub(crate) struct PriorTerm {
    /// Index into the packed parameter vector.
    pub(crate) coord: usize,
    /// Parameter name as declared, for diagnostics and the fit report.
    pub(crate) name: String,
    /// Prior mean, on the packed scale.
    pub(crate) mean: f64,
    /// Prior SD, on the packed scale. Strictly positive.
    pub(crate) sd: f64,
    /// How this coordinate's declared value maps to its packed value.
    pub(crate) scale: PriorScale,
    /// Prior central value as the user wrote it (declared scale), carried so
    /// the report can echo it without inverting `mean`.
    pub(crate) declared_value: f64,
}

/// Every prior in force for one fit, resolved against the packed layout.
///
/// Built once per estimation stage — never per objective evaluation — so the
/// name resolution and the `ln`/`sqrt` conversions cost nothing on the hot path.
#[derive(Debug, Clone, Default)]
pub(crate) struct PriorSet {
    terms: Vec<PriorTerm>,
}

impl PriorSet {
    /// Resolve a model's declared priors against the packed layout of
    /// `template`.
    ///
    /// Every rejection is an `Err` naming the parameter: a prior that cannot be
    /// applied must stop the fit, because an unapplied prior returns a
    /// plausible-looking unpenalized fit with nothing to say the prior was
    /// dropped.
    pub(crate) fn build(model: &CompiledModel, template: &ModelParameters) -> Result<Self, String> {
        if model.priors.is_empty() {
            return Ok(Self::default());
        }
        let coords = coordinate_table(model, template);
        let fixed = packed_fixed_mask(template);
        let mut terms: Vec<PriorTerm> = Vec::with_capacity(model.priors.len());

        for prior in &model.priors {
            let matches: Vec<&CoordInfo> = coords
                .iter()
                .filter(|c| c.name.eq_ignore_ascii_case(&prior.name))
                .collect();
            let info = match matches.as_slice() {
                [] => {
                    return Err(format!(
                        "prior on `{}`: no theta, omega, sigma or kappa by that name. \
                         A prior must name a parameter declared in `[parameters]`.",
                        prior.name
                    ))
                }
                [one] => *one,
                // Two coordinates can share a name only via a `[mixture]`
                // per-class override, which v1 rejects anyway — but resolving
                // ambiguously would silently prior the wrong one.
                _ => {
                    return Err(format!(
                        "prior on `{}`: the name resolves to {} packed coordinates; \
                         priors on per-class `[mixture]` overrides are not supported.",
                        prior.name,
                        matches.len()
                    ))
                }
            };

            if let Some(reason) = info.rejection.as_deref() {
                return Err(format!("prior on `{}`: {reason}", prior.name));
            }
            if fixed.get(info.coord).copied().unwrap_or(false) {
                return Err(format!(
                    "prior on `{}`: the parameter is FIXed, so the prior can never move it. \
                     Drop the prior or drop the FIX.",
                    prior.name
                ));
            }

            let scale = info.scale;
            // Finiteness first: `!(NaN > 0.0)` is `true`, so the positivity check
            // below would otherwise report a `NaN` as "must be > 0" and point the
            // user at the wrong problem.
            if !prior.value.is_finite() {
                return Err(format!(
                    "prior on `{}`: central value must be finite, got {}.",
                    prior.name, prior.value
                ));
            }
            if scale.requires_positive() && !(prior.value > 0.0) {
                return Err(format!(
                    "prior on `{}`: central value must be > 0 on the declared scale \
                     (this parameter is estimated on the log scale), got {}.",
                    prior.name, prior.value
                ));
            }

            let sd = packed_sd(&prior.name, prior.value, prior.spread, scale)?;
            terms.push(PriorTerm {
                coord: info.coord,
                name: info.name.clone(),
                mean: scale.to_packed(prior.value),
                sd,
                scale,
                declared_value: prior.value,
            });
        }

        // A second prior on the same coordinate would silently multiply the two
        // densities, which is never what a duplicated line means.
        terms.sort_by_key(|t| t.coord);
        if let Some(w) = terms.windows(2).find(|w| w[0].coord == w[1].coord) {
            return Err(format!(
                "prior on `{}`: declared more than once.",
                w[0].name
            ));
        }
        Ok(Self { terms })
    }

    /// `true` when at least one prior is in force. Call sites use this to keep
    /// an unpriored fit bit-identical to one built before this feature existed.
    pub(crate) fn is_active(&self) -> bool {
        !self.terms.is_empty()
    }

    /// `Σ ((x−m)/s)²` — the prior contribution to the OFV.
    ///
    /// Normalisation constants are omitted, matching the data side (ferx's OFV
    /// already drops `N·log(2π)`). A priored ferx OFV is therefore not
    /// absolutely comparable to a priored NONMEM OFV; compare ΔOFV against a
    /// null twin.
    pub(crate) fn penalty(&self, packed: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|t| {
                let z = (packed[t.coord] - t.mean) / t.sd;
                z * z
            })
            .sum()
    }

    /// Add `∂penalty/∂x = 2(x−m)/s²` into `grad`, which must be in **packed**
    /// (unscaled) space — the same space the likelihood gradient is assembled
    /// in before the optimizer's `* scale[k]` chain rule.
    pub(crate) fn add_gradient(&self, packed: &[f64], grad: &mut [f64]) {
        for t in &self.terms {
            grad[t.coord] += 2.0 * (packed[t.coord] - t.mean) / (t.sd * t.sd);
        }
    }

    /// [`Self::penalty`] and [`Self::add_gradient`] in one call.
    ///
    /// Exists for the same reason `NnRegularizer::penalty_and_gradient` does,
    /// plus a test-shaped one: at a call site that takes the value and the
    /// gradient separately, deleting only the gradient half leaves an optimizer
    /// minimising a penalized objective with an unpenalized gradient — a defect
    /// no *outcome* assertion reliably catches, because the line search still
    /// drags the estimate toward the prior, just badly. Measured: dropping the
    /// splice in `outer_optimizer` left every integration test in
    /// `tests/parameter_priors.rs` green. Binding the two together means one
    /// mutation removes both, and `a_prior_moves_the_estimates_and_not_only_the_report`
    /// then reddens.
    pub(crate) fn penalty_and_gradient(&self, packed: &[f64], grad: &mut [f64]) -> f64 {
        self.add_gradient(packed, grad);
        self.penalty(packed)
    }

    /// Add `∂²penalty/∂x² = 2/s²` on the diagonal.
    ///
    /// Constant, so it takes no `packed` argument and needs no finite
    /// differences: the covariance step adds it to the FD-of-OFV `R` matrix for
    /// free. That matters — an SE computed from the *unpenalized* curvature is
    /// wrong precisely on the sparse fits this feature exists for, and keeps
    /// going non-PD in exactly the directions the prior was added to identify.
    pub(crate) fn add_hessian(&self, add: &mut dyn FnMut(usize, usize, f64)) {
        for t in &self.terms {
            add(t.coord, t.coord, 2.0 / (t.sd * t.sd));
        }
    }

    /// Per-parameter report at the final packed estimate.
    pub(crate) fn summarize(&self, packed: &[f64]) -> Vec<PriorSummary> {
        self.terms
            .iter()
            .map(|t| {
                let x = packed[t.coord];
                let z = (x - t.mean) / t.sd;
                PriorSummary {
                    name: t.name.clone(),
                    prior_value: t.declared_value,
                    estimate: t.scale.from_packed(x),
                    shift_in_prior_sds: z,
                    penalty: z * z,
                    family: t.scale.family().to_string(),
                    prior_lower_95: t.scale.from_packed(t.mean - 1.959_963_984_540_054 * t.sd),
                    prior_upper_95: t.scale.from_packed(t.mean + 1.959_963_984_540_054 * t.sd),
                }
            })
            .collect()
    }
}

/// Convert a declared spread to a packed-scale SD.
fn packed_sd(
    name: &str,
    value: f64,
    spread: PriorSpread,
    scale: PriorScale,
) -> Result<f64, String> {
    // An absolute SD on a log-packed coordinate is read as the relative spread
    // it implies, so `sd = 0.03` on a value of 0.15 lands in exactly the same
    // formula as `rse = 20%`. One conversion, not two that could disagree.
    let rse = match spread {
        PriorSpread::Rse(r) => r,
        PriorSpread::Sd(s) => {
            if !(s > 0.0) || !s.is_finite() {
                return Err(format!(
                    "prior on `{name}`: `sd` must be > 0 and finite, got {s}."
                ));
            }
            if matches!(scale, PriorScale::Identity) {
                // No log scale to relate to — the absolute SD *is* the packed SD.
                return Ok(s);
            }
            s / value.abs()
        }
    };
    if !(rse > 0.0) || !rse.is_finite() {
        return Err(format!(
            "prior on `{name}`: `rse` must be > 0 and finite, got {rse}."
        ));
    }
    let sd = match scale {
        // Normal prior on the natural scale: SD = |m| · RSE.
        PriorScale::Identity => value.abs() * rse,
        // Lognormal: the exact CV → log-SD conversion, then onto the packed
        // scale (a factor ½ when the declaration is a variance and the packed
        // coordinate is ln(SD)).
        PriorScale::Log | PriorScale::HalfLog => {
            (1.0 + rse * rse).ln().sqrt() * scale.log_spread_factor()
        }
    };
    if !(sd > 0.0) || !sd.is_finite() {
        return Err(format!(
            "prior on `{name}`: the declared spread gives a non-positive or \
             non-finite prior SD ({sd})."
        ));
    }
    Ok(sd)
}

/// What one packed coordinate is, as far as a prior is concerned.
struct CoordInfo {
    coord: usize,
    name: String,
    scale: PriorScale,
    /// `Some(reason)` when a prior on this coordinate is rejected in v1.
    /// Carried per coordinate rather than tested at the use site so the scope
    /// gate is a single predicate — two gates rejecting the same inputs is a
    /// test hole, not belt-and-braces.
    rejection: Option<String>,
}

/// Describe every packed coordinate, in [`pack_params`] order.
///
/// This is the one place the packed layout is walked for priors. It leans on
/// [`coordinate_names`] and [`coordinate_kinds`] for the layout itself, so a
/// future segment appears here automatically (as an unnamed, rejected
/// coordinate) rather than being silently mis-indexed.
fn coordinate_table(model: &CompiledModel, template: &ModelParameters) -> Vec<CoordInfo> {
    let names = coordinate_names(template);
    let kinds = coordinate_kinds(template);
    let segs = packed_segments(template);
    let omega_start = segs.omega_start();
    let sigma_start = segs.sigma_start();
    let iov_start = segs.iov_start();
    let iov_end = segs.mixture_omega_start();

    let omega_is_block = !template.omega.diagonal;
    let iov_is_block = template.omega_iov.as_ref().is_some_and(|m| !m.diagonal);

    (0..names.len())
        .map(|i| {
            let kind = kinds.get(i).copied().unwrap_or(PackedCoordKind::Theta);
            let (scale, rejection) = match kind {
                PackedCoordKind::Theta => {
                    let scale = if theta_packs_log(template.theta_lower[i]) {
                        PriorScale::Log
                    } else {
                        PriorScale::Identity
                    };
                    (scale, None)
                }
                PackedCoordKind::OmegaOffDiagonal => (
                    PriorScale::Identity,
                    Some(
                        "priors on a `block_omega` / `block_sigma` off-diagonal are not \
                         supported. A correlated block needs an inverse-Wishart prior, \
                         which v1 deliberately leaves out; declare the diagonal \
                         variances separately to prior them."
                            .to_string(),
                    ),
                ),
                PackedCoordKind::OmegaDiagonal => {
                    // Which Ω family this diagonal belongs to decides both the
                    // declared scale (`(sd)` vs the variance default) and
                    // whether it sits inside a block.
                    if i < sigma_start {
                        let as_sd = flag_at(&model.omega_init_as_sd, i - omega_start);
                        (
                            sd_or_var_scale(as_sd),
                            omega_is_block.then(|| block_rejection("block_omega")),
                        )
                    } else if i < iov_end {
                        let as_sd = flag_at(&model.kappa_init_as_sd, i - iov_start);
                        (
                            sd_or_var_scale(as_sd),
                            iov_is_block.then(|| block_rejection("block_kappa")),
                        )
                    } else {
                        (
                            PriorScale::HalfLog,
                            Some(
                                "priors on a per-class `[mixture]` Ω/Σ override are not \
                                 supported."
                                    .to_string(),
                            ),
                        )
                    }
                }
                PackedCoordKind::Sigma => {
                    if i < iov_start {
                        let s = i - sigma_start;
                        (sd_or_var_scale(flag_at(&model.sigma_init_as_sd, s)), None)
                    } else {
                        (
                            PriorScale::HalfLog,
                            Some(
                                "priors on a per-class `[mixture]` Ω/Σ override are not \
                                 supported."
                                    .to_string(),
                            ),
                        )
                    }
                }
            };
            CoordInfo {
                coord: i,
                name: names[i].clone(),
                scale,
                rejection,
            }
        })
        .collect()
}

fn block_rejection(kind: &str) -> String {
    format!(
        "the parameter belongs to a `{kind}`, whose elements are estimated as a \
         Cholesky factor rather than as variances. Priors on block elements are \
         not supported in v1."
    )
}

fn sd_or_var_scale(declared_as_sd: bool) -> PriorScale {
    if declared_as_sd {
        PriorScale::Log
    } else {
        PriorScale::HalfLog
    }
}

/// Read one `*_init_as_sd` flag, defaulting to the variance scale.
///
/// The flags live on [`CompiledModel`], parallel to the Ω diagonal / Σ / κ
/// declarations, so they are indexed by the coordinate's offset **within its
/// segment**, never by its packed index. Spelled once so the "missing flag means
/// the variance default" fallback cannot drift between the three families.
fn flag_at(flags: &[bool], within_segment: usize) -> bool {
    flags.get(within_segment).copied().unwrap_or(false)
}

#[cfg(test)]
#[path = "priors_tests.rs"]
mod tests;
