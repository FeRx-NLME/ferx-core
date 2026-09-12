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
    coordinate_names, lower_tri_iter, packed_fixed_mask, packed_segments, theta_packs_log,
};
use crate::io::fit_estimates::EstimateKind;
use crate::types::{CompiledModel, ModelParameters, ParameterPrior, PriorSpread, PriorSummary};

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
                        matches
                            .iter()
                            .map(|c| c.kind.keyword())
                            .collect::<Vec<_>>()
                            .join(", ")
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

/// Consume `[priors] from_fit = "<path>"` into ordinary [`ParameterPrior`]s
/// (#254 phase 2).
///
/// Every free parameter the source fit reports with a usable standard error, and
/// whose **name and family** both match a coordinate of this model, is appended
/// to `model.priors` as `prior(estimate, rse = SE/estimate)` on *this* model's
/// declared scale. Everything else is skipped with a note pushed onto
/// `model.parse_warnings` — unlike a hand-written prior, which is a hard error
/// when it cannot be applied. The asymmetry is deliberate: a typed prior names
/// one parameter the user meant, while an import is a bulk operation over a
/// source model that legitimately has parameters this one does not.
///
/// # Why this runs once, at parse time
///
/// The file read used to sit in [`PriorSet::build`], which every estimation
/// stage and the covariance and SIR steps call. That was wrong twice over
/// (review of #1363): `build_prior_set` turns a build error into an *empty* set,
/// and `run_covariance` / `run_sir` are public entry points that do not run
/// `check_parameter_priors` — so a source fit that had moved or become
/// unreadable between the fit and the post-fit step made every prior silently
/// vanish, and those APIs went on reporting unpenalized standard errors. It also
/// left a window in which the file could change *during* a fit, so the prior the
/// optimizer minimised and the prior the report printed need not be the same one.
///
/// Doing it here means the file is read exactly once, the error reaches the user
/// through the parser's own `Result`, and `PriorSet::build` is pure — it refuses
/// a model that still carries an unexpanded path rather than reading anything.
///
/// # Why matching is on name *and* kind
///
/// A source θ called `CL` and a target Ω called `CL` are both legal, and the
/// numbers are on unrelated scales: importing the θ's natural value as an Ω
/// variance prior would be silently, quietly wrong. The matched
/// [`crate::types::ParameterKind`] is carried onto the resulting
/// `ParameterPrior`, so the packed resolution lands on the same coordinate
/// instead of re-deriving it from the name and finding two.
///
/// # Scale
///
/// A [`crate::types::FitResult`] reports Ω/κ as a **variance** and Σ as an
/// **SD**, whatever the source model declared them as — so the source's own
/// `(sd)` spelling is invisible here and is not guessed at. The target's
/// declared scale is the only one that matters, and the two conversions are the
/// delta method on a power: the relative SE of `xᵖ` is `|p|` times the relative
/// SE of `x`, so variance → SD halves the RSE and SD → variance doubles it.
pub(crate) fn expand_prior_from_fit(model: &mut CompiledModel) -> Result<(), String> {
    let Some(path) = model.prior_from_fit.take() else {
        return Ok(());
    };
    // Resolved against the model's own declarations, which is the layout every
    // optimizer packs and the only one available before a dataset is read.
    let template = model.default_params.clone();
    let coords = coordinate_table(model, &template);
    let fixed = packed_fixed_mask(&template);
    let mut notes: Vec<String> = Vec::new();

    let imported = match import_from_fit(&path, model, &coords, &fixed, &mut notes) {
        Ok(v) => v,
        Err(e) => {
            // Put the path back so a caller that recovers from the error can see
            // what was asked for, and so `PriorSet::build` still refuses.
            model.prior_from_fit = Some(path);
            return Err(e);
        }
    };
    model.priors.extend(imported);
    if !notes.is_empty() {
        model.parse_warnings.push(format!(
            "[priors] from_fit: {} parameter(s) not imported:\n  - {}",
            notes.len(),
            notes.join("\n  - ")
        ));
    }
    Ok(())
}

/// The body of [`expand_prior_from_fit`], against an already-built coordinate
/// table.
fn import_from_fit(
    path: &str,
    model: &CompiledModel,
    coords: &[CoordInfo],
    fixed: &[bool],
    notes: &mut Vec<String>,
) -> Result<Vec<ParameterPrior>, String> {
    let source = crate::io::fit_estimates::read_fit_estimates(std::path::Path::new(path))
        .map_err(|e| format!("[priors] from_fit: {e}"))?;

    let mut out: Vec<ParameterPrior> = Vec::new();
    for est in &source {
        let matches: Vec<&CoordInfo> = coords
            .iter()
            .filter(|c| c.kind == est.kind && c.name.eq_ignore_ascii_case(&est.name))
            .collect();
        let info = match matches.as_slice() {
            [one] => *one,
            // Not in this model at all. Silent, and the **only** silent skip
            // that is not a deliberate user choice: a source model bigger than
            // the one being updated is the normal case, and one note per absent
            // parameter would bury the ones that matter. The "nothing landed"
            // guard below is what catches a wholesale mismatch.
            [] => continue,
            // Two coordinates of the same family under one name. Unreachable
            // today — the `[mixture]` overrides that could share a name are
            // suffixed `_MIX{n}` — but folding it into the `[]` arm would make
            // a future layout change *silently* drop the prior, which is the one
            // outcome this feature must never produce. A note, not an error:
            // this is still a bulk import over a model the user did not write
            // for it.
            _ => {
                notes.push(format!(
                    "{} {}: the name resolves to {} packed coordinates in this \
                     model, so which one the source estimate refers to is \
                     undecidable.",
                    est.kind.keyword(),
                    est.name,
                    matches.len()
                ));
                continue;
            }
        };
        // The `[mixture]` / `block_sigma` tail of the coordinate table carries a
        // placeholder `kind` (see `coordinate_table`), so it *can* survive the
        // match above; this is what stops it, and every other out-of-scope
        // coordinate, from being imported onto.
        if let Some(reason) = info.rejection.as_deref() {
            notes.push(format!("{} {}: {reason}", est.kind.keyword(), est.name));
            continue;
        }
        // An inline `prior(...)` on the same parameter wins, so one imported
        // prior can be overridden without giving up the rest. Silent: the user
        // wrote the override on purpose, and it is visible in `prior_summary`.
        //
        // Matched on **family as well as name**, like everything else here. A
        // model may carry `theta CL` and `omega CL`, and before the family was
        // consulted a typed prior on one of them suppressed the import of the
        // other — and did so silently, which is exactly the failure this feature
        // is built to avoid. A typed prior that names no family suppresses by
        // name alone, conservatively: it cannot say which it meant, and in a
        // colliding model it will refuse to resolve anyway.
        if model.priors.iter().any(|p| {
            p.name.eq_ignore_ascii_case(&info.name) && p.kind.is_none_or(|k| k == info.kind)
        }) {
            continue;
        }
        if fixed.get(info.coord).copied().unwrap_or(false) {
            notes.push(format!(
                "{} {}: FIXed in this model, so a prior could never move it.",
                est.kind.keyword(),
                est.name
            ));
            continue;
        }
        let Some(se) = est.se else {
            notes.push(format!(
                "{} {}: the source fit reports no standard error for it (a FIXed \
                 parameter, or a fit whose covariance step did not run).",
                est.kind.keyword(),
                est.name
            ));
            continue;
        };
        if !est.value.is_finite() || (info.scale.requires_positive() && !(est.value > 0.0)) {
            notes.push(format!(
                "{} {}: the source estimate is {}, which is not a usable prior \
                 centre for a parameter estimated on the log scale.",
                est.kind.keyword(),
                est.name,
                est.value
            ));
            continue;
        }

        // Source scale → this model's declared scale. `Identity` is a θ that may
        // be negative, where a *relative* standard error is meaningless (and
        // undefined at zero) — carry the absolute SE instead, which is the
        // escape hatch `PriorSpread::Sd` exists for.
        let rel = se / est.value.abs();
        let (value, spread) = match (est.kind, info.scale) {
            (_, PriorScale::Identity) => (est.value, PriorSpread::Sd(se)),
            (EstimateKind::Theta, _) => (est.value, PriorSpread::Rse(rel)),
            // Ω / κ: the source reports a variance.
            (EstimateKind::Omega | EstimateKind::Kappa, PriorScale::HalfLog) => {
                (est.value, PriorSpread::Rse(rel))
            }
            (EstimateKind::Omega | EstimateKind::Kappa, PriorScale::Log) => {
                (est.value.sqrt(), PriorSpread::Rse(0.5 * rel))
            }
            // Σ: the source reports an SD.
            (EstimateKind::Sigma, PriorScale::Log) => (est.value, PriorSpread::Rse(rel)),
            (EstimateKind::Sigma, PriorScale::HalfLog) => {
                (est.value * est.value, PriorSpread::Rse(2.0 * rel))
            }
        };
        out.push(ParameterPrior {
            name: info.name.clone(),
            value,
            spread,
            // The family this was matched on, carried through so the shared
            // resolution below lands on the same coordinate rather than
            // re-deriving it from the name and possibly finding two.
            kind: Some(info.kind),
        });
    }

    // An import that lands nothing leaves a fit that looks exactly like one the
    // import shaped — the same failure mode phase 1 hard-errors on for a typed
    // prior that cannot be applied. Unconditional, and deliberately not weakened
    // to "…unless some typed prior landed": the user wrote `from_fit`, and a
    // wrong path or a model whose parameter names have all been renamed is
    // otherwise invisible.
    if out.is_empty() {
        let why = if notes.is_empty() {
            format!(
                "none of its {} parameters has a name and family matching a \
                 `[parameters]` declaration in this model",
                source.len()
            )
        } else {
            format!("every candidate was skipped:\n  - {}", notes.join("\n  - "))
        };
        return Err(format!(
            "[priors] from_fit = `{path}`: no prior could be imported — {why}."
        ));
    }
    Ok(out)
}

/// What one packed coordinate is, as far as a prior is concerned.
struct CoordInfo {
    coord: usize,
    name: String,
    /// Which `[parameters]` family this coordinate belongs to, for matching a
    /// `from_fit` source estimate on name *and* kind.
    kind: EstimateKind,
    scale: PriorScale,
    /// `Some(reason)` when a prior on this coordinate is rejected in v1.
    /// Carried per coordinate rather than tested at the use site so the scope
    /// gate is a single predicate — two gates rejecting the same inputs is a
    /// test hole, not belt-and-braces.
    rejection: Option<String>,
}

/// Describe every packed coordinate, in [`pack_params`] order.
///
/// This is the one place the packed layout is walked for priors. The Ω / Ω_IOV
/// segments are walked with [`lower_tri_iter`] rather than by offset arithmetic,
/// because for a **non-diagonal** Ω the packed index is a position in the
/// column-major lower triangle and is *not* the eta index — so
/// `omega_init_as_sd[i - omega_start]` reads the wrong flag (or runs off the
/// end) the moment a `block_omega` is present. Walking the triangle gives each
/// coordinate its real `(row, col)`, which is what both the scale lookup and the
/// block test need.
fn coordinate_table(model: &CompiledModel, template: &ModelParameters) -> Vec<CoordInfo> {
    let names = coordinate_names(template);
    let segs = packed_segments(template);
    let mut out: Vec<CoordInfo> = Vec::with_capacity(names.len());

    let name_at = |i: usize| names.get(i).cloned().unwrap_or_else(|| format!("#{i}"));

    // θ — the prior family follows the packing, which follows the lower bound.
    for i in 0..segs.theta {
        let scale = if theta_packs_log(template.theta_lower[i]) {
            PriorScale::Log
        } else {
            PriorScale::Identity
        };
        out.push(CoordInfo {
            coord: i,
            name: name_at(i),
            kind: EstimateKind::Theta,
            scale,
            rejection: None,
        });
    }

    // Ω, then (below) Ω_IOV: same shape, different `init_as_sd` table and
    // different keyword in the diagnostic.
    push_omega_coords(
        &mut out,
        &template.omega,
        &model.omega_init_as_sd,
        EstimateKind::Omega,
        "block_omega",
        &name_at,
    );

    // Σ — always diagonal, packed as `ln(SD)`; declared as a variance unless `(sd)`.
    for s_i in 0..segs.sigma {
        let i = out.len();
        out.push(CoordInfo {
            coord: i,
            name: name_at(i),
            kind: EstimateKind::Sigma,
            scale: sd_or_var_scale(flag_at(&model.sigma_init_as_sd, s_i)),
            rejection: None,
        });
    }

    if let Some(iov) = template.omega_iov.as_ref() {
        push_omega_coords(
            &mut out,
            iov,
            &model.kappa_init_as_sd,
            EstimateKind::Kappa,
            "block_kappa",
            &name_at,
        );
    }

    // Everything past here — `[mixture]` per-class Ω/Σ overrides and the
    // `block_sigma` ρ coordinates — is out of scope for v1. Filling the tail
    // generically means a future segment arrives as a *rejected* coordinate
    // rather than being silently mis-indexed onto one of the tables above.
    while out.len() < names.len() {
        let i = out.len();
        out.push(CoordInfo {
            coord: i,
            name: name_at(i),
            // Placeholder: every coordinate here carries a `rejection`, and both
            // consumers test that first, so the kind is never read. Spelled as
            // the Ω family rather than left meaningful-looking because a
            // `[mixture]` override *is* an Ω/Σ override — but it must never be
            // matched as one.
            kind: EstimateKind::Omega,
            scale: PriorScale::HalfLog,
            rejection: Some(
                "priors on a per-class `[mixture]` Ω/Σ override or a `block_sigma` \
                 correlation are not supported."
                    .to_string(),
            ),
        });
    }
    out
}

/// Append one Ω-family segment (Ω or Ω_IOV) to the coordinate table.
///
/// Block membership is decided **per coordinate**, not per matrix. A mixed model
/// — `block_omega (A, B)` alongside a standalone `omega C ~ …` — is one
/// non-diagonal `OmegaMatrix`, so a matrix-level `!diagonal` test rejects a prior
/// on `C` even though `C` is an ordinary independent variance. That is exactly
/// the arrangement the block diagnostic tells users to reach for ("declare the
/// diagonal variances separately to prior them"), so getting it wrong makes the
/// advice impossible to follow.
///
/// `free_mask` is what encodes the real structure: a diagonal `(e, e)` is inside
/// a block iff some off-diagonal in its row or column is a *free* parameter.
fn push_omega_coords(
    out: &mut Vec<CoordInfo>,
    om: &crate::types::OmegaMatrix,
    init_as_sd: &[bool],
    kind: EstimateKind,
    block_keyword: &str,
    name_at: &dyn Fn(usize) -> String,
) {
    let n = om.dim();
    for (r, c) in lower_tri_iter(n, om.diagonal) {
        let i = out.len();
        let (scale, rejection) = if r == c {
            // `init_as_sd` is parallel to the *eta* list, so it is indexed by the
            // row, never by the packed position.
            // A *free* off-diagonal is the usual marker, but it is not the only
            // one: a fixed non-zero covariance is still a covariance, and the
            // packed coordinate for row `r > 0` of such a block is `ln(L[r,r])`,
            // which is not `ln(SD_r)` (`SD_r² = Σⱼ L[r,j]²`) — so scaling a prior
            // as if it were would be wrong with nothing to say so. Today
            // `packed_fixed_mask` fixes the whole row/col of a FIXed eta, which
            // makes the case unreachable; testing the matrix as well as the mask
            // means this stays correct without depending on that.
            let in_block = (0..n).any(|j| {
                j != r
                    && (om.free_mask[(r, j)]
                        || om.free_mask[(j, r)]
                        || om.matrix[(r, j)] != 0.0
                        || om.matrix[(j, r)] != 0.0)
            });
            (
                sd_or_var_scale(flag_at(init_as_sd, r)),
                in_block.then(|| block_rejection(block_keyword)),
            )
        } else {
            (
                PriorScale::Identity,
                Some(
                    "priors on a `block_omega` / `block_sigma` off-diagonal are not \
                     supported. A correlated block needs an inverse-Wishart prior, \
                     which v1 deliberately leaves out; declare the diagonal \
                     variances separately to prior them."
                        .to_string(),
                ),
            )
        };
        out.push(CoordInfo {
            coord: i,
            name: name_at(i),
            kind,
            scale,
            rejection,
        });
    }
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
