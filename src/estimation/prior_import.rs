//! Expanding a prior from a previous fit (`[priors] from_fit`), as the parser resolves it.
//! Below the estimators (#3041).
use crate::estimation::parameterization::{
    coordinate_names, lower_tri_iter, packed_fixed_mask, packed_segments, theta_packs_log,
};
use crate::io::fit_estimates::EstimateKind;
use crate::types::{CompiledModel, ModelParameters, ParameterPrior, PriorSpread};

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
    pub(crate) fn log_spread_factor(self) -> f64 {
        match self {
            PriorScale::Identity => 1.0,
            PriorScale::Log => 1.0,
            PriorScale::HalfLog => 0.5,
        }
    }

    /// `true` when the parameter is strictly positive on its declared scale, so
    /// a non-positive prior central value is a declaration error rather than a
    /// legitimate point on the real line.
    pub(crate) fn requires_positive(self) -> bool {
        !matches!(self, PriorScale::Identity)
    }

    /// Human-readable family name for the fit report.
    pub(crate) fn family(self) -> &'static str {
        match self {
            PriorScale::Identity => "normal",
            PriorScale::Log | PriorScale::HalfLog => "lognormal",
        }
    }
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
pub(crate) fn import_from_fit(
    path: &str, model: &CompiledModel, coords: &[CoordInfo], fixed: &[bool],
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
        return Err(format!("[priors] from_fit = `{path}`: no prior could be imported — {why}."));
    }
    Ok(out)
}

/// What one packed coordinate is, as far as a prior is concerned.
pub(crate) struct CoordInfo {
    pub(crate) coord: usize,
    pub(crate) name: String,
    /// Which `[parameters]` family this coordinate belongs to, for matching a
    /// `from_fit` source estimate on name *and* kind.
    pub(crate) kind: EstimateKind,
    pub(crate) scale: PriorScale,
    /// `Some(reason)` when a prior on this coordinate is rejected in v1.
    /// Carried per coordinate rather than tested at the use site so the scope
    /// gate is a single predicate — two gates rejecting the same inputs is a
    /// test hole, not belt-and-braces.
    pub(crate) rejection: Option<String>,
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
pub(crate) fn coordinate_table(
    model: &CompiledModel, template: &ModelParameters,
) -> Vec<CoordInfo> {
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
pub(crate) fn push_omega_coords(
    out: &mut Vec<CoordInfo>, om: &crate::types::OmegaMatrix, init_as_sd: &[bool],
    kind: EstimateKind, block_keyword: &str, name_at: &dyn Fn(usize) -> String,
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
        out.push(CoordInfo { coord: i, name: name_at(i), kind, scale, rejection });
    }
}

pub(crate) fn block_rejection(kind: &str) -> String {
    format!(
        "the parameter belongs to a `{kind}`, whose elements are estimated as a \
         Cholesky factor rather than as variances. Priors on block elements are \
         not supported in v1."
    )
}

pub(crate) fn sd_or_var_scale(declared_as_sd: bool) -> PriorScale {
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
pub(crate) fn flag_at(flags: &[bool], within_segment: usize) -> bool {
    flags.get(within_segment).copied().unwrap_or(false)
}
