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

use crate::estimation::parameterization::packed_fixed_mask;
#[cfg(test)]
pub(crate) use crate::estimation::prior_import::expand_prior_from_fit;
use crate::types::{CompiledModel, ModelParameters, PriorSpread, PriorSummary};

pub(crate) use crate::estimation::prior_import::{coordinate_table, CoordInfo, PriorScale};

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
        // `[priors] from_fit` is consumed by [`expand_prior_from_fit`] at parse
        // time, which appends the imported priors to `model.priors` and leaves
        // this field `None`. A model that still carries a path was built by hand
        // and never parsed, so nothing has read the file — refuse rather than
        // fit unpenalized, which is the one outcome that looks identical to
        // success (#254 phase 2 review).
        if let Some(path) = model.prior_from_fit.as_deref() {
            return Err(format!(
                "[priors] from_fit = `{path}` has not been expanded. A `CompiledModel` \
                 built by hand must call \
                 `parser::model_parser::expand_prior_from_fit` before it is fit; \
                 a model parsed from a file or a string has already done so."
            ));
        }
        if model.priors.is_empty() {
            return Ok(Self::default());
        }
        let coords = coordinate_table(model, template);
        let fixed = packed_fixed_mask(template);
        let mut terms: Vec<PriorTerm> = Vec::with_capacity(model.priors.len());

        for prior in model.priors.iter() {
            // Narrow by family first when the prior carries one. θ names and η
            // names are separate namespaces, so `theta CL` alongside `omega CL`
            // is a legal model; without this the prior resolves to two
            // coordinates and the fit is refused as ambiguous even though the
            // producer knew exactly which one it meant.
            let matches: Vec<&CoordInfo> = coords
                .iter()
                .filter(|c| {
                    prior.kind.is_none_or(|k| c.kind == k)
                        && c.name.eq_ignore_ascii_case(&prior.name)
                })
                .collect();
            let info = match matches.as_slice() {
                [] => {
                    return Err(format!(
                        "prior on `{}`: no {} by that name. \
                         A prior must name a parameter declared in `[parameters]`.",
                        prior.name,
                        match prior.kind {
                            Some(k) => k.keyword(),
                            None => "theta, omega, sigma or kappa",
                        }
                    ))
                }
                [one] => *one,
                // Only reachable for a prior that names no family: a `[mixture]`
                // per-class override (which v1 rejects anyway), or a name a
                // hand-built `ParameterPrior` shares across two families. Both
                // are genuinely undecidable here — resolving either way would
                // silently prior the wrong coordinate.
                _ => {
                    return Err(format!(
                        "prior on `{}`: the name resolves to {} packed coordinates \
                         ({}). Set `ParameterPrior::kind` to say which family is meant; \
                         priors on per-class `[mixture]` overrides are not supported.",
                        prior.name,
                        matches.len(),
                        matches.iter().map(|c| c.kind.keyword()).collect::<Vec<_>>().join(", ")
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
            return Err(format!("prior on `{}`: declared more than once.", w[0].name));
        }
        Ok(Self { terms })
    }

    /// `true` when at least one prior is in force. Call sites use this to keep
    /// an unpriored fit bit-identical to one built before this feature existed.
    pub(crate) fn is_active(&self) -> bool {
        !self.terms.is_empty()
    }

    /// How many parameters carry a prior — typed and imported together.
    pub(crate) fn len(&self) -> usize {
        self.terms.len()
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
    name: &str, value: f64, spread: PriorSpread, scale: PriorScale,
) -> Result<f64, String> {
    // An absolute SD on a log-packed coordinate is read as the relative spread
    // it implies, so `sd = 0.03` on a value of 0.15 lands in exactly the same
    // formula as `rse = 20%`. One conversion, not two that could disagree.
    let rse = match spread {
        PriorSpread::Rse(r) => r,
        PriorSpread::Sd(s) => {
            if !(s > 0.0) || !s.is_finite() {
                return Err(format!("prior on `{name}`: `sd` must be > 0 and finite, got {s}."));
            }
            if matches!(scale, PriorScale::Identity) {
                // No log scale to relate to — the absolute SD *is* the packed SD.
                return Ok(s);
            }
            s / value.abs()
        }
    };
    if !(rse > 0.0) || !rse.is_finite() {
        return Err(format!("prior on `{name}`: `rse` must be > 0 and finite, got {rse}."));
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

#[cfg(test)]
#[path = "priors_tests.rs"]
mod tests;
