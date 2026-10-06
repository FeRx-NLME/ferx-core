//! Random-effect shape transforms on the η scale (#1716, tracker #359):
//! `boxcox(η, λ)`, `tdist(η, ν)` and `johndraper(η, λ)`, after Petersson et
//! al. 2009 (*Pharm Res* 26:2174) — the formulas Pharmpy's `transform_etas_*`
//! generate. η stays `N(0, ω)`; the model reads `h(η)` where it read η, the
//! NONMEM spelling `CL = TVCL * EXP(ETATR)`.
//!
//! Each transform is written **once**, as [`shape_g`] over [`PkNum`]: `T = f64`
//! is the value every evaluator computes, and `T = Dual2` gives the exact
//! `∂/∂η`, `∂/∂shape` the sensitivity providers consume. There is no second copy
//! of any formula — the symbolic partials ([`ShapeOut::DEta`] /
//! [`ShapeOut::DShape`]) are the same kernel over `Dual1`.

use crate::sens::dual1::Dual1;
use crate::sens::num::PkNum;

/// Which shape transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeKind {
    /// `(exp(η)^λ − 1)/λ = expm1(λη)/λ`; λ → 0 is the identity (log-normal).
    BoxCox,
    /// Student-t-like heavy tails (Petersson's third-order series); ν → ∞ is
    /// the identity.
    TDist,
    /// `sign(η)·((|η| + 1)^λ − 1)/λ`; λ = 1 is the identity.
    JohnDraper,
}

impl ShapeKind {
    /// Every kind, for name lookup and tests.
    pub(crate) const ALL: [ShapeKind; 3] =
        [ShapeKind::BoxCox, ShapeKind::TDist, ShapeKind::JohnDraper];

    /// The model-file function name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            ShapeKind::BoxCox => "boxcox",
            ShapeKind::TDist => "tdist",
            ShapeKind::JohnDraper => "johndraper",
        }
    }

    /// The kind a (lower-cased) function name spells, if any.
    pub(crate) fn from_name(name: &str) -> Option<ShapeKind> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// `(init, lower, upper)` of an auto-declared shape θ. Box-Cox and t-dist
    /// are Pharmpy's (`_create_new_thetas` in `modeling/parameter_variability.py`:
    /// λ `(0.01, −3, 3)`, ν `(80, 3, 100)`). John-Draper starts at its identity
    /// λ = 1, so a fit starts at the unshaped model; Pharmpy starts it at 0.01.
    pub(crate) fn default_theta(self) -> (f64, f64, f64) {
        match self {
            ShapeKind::BoxCox => (0.01, -3.0, 3.0),
            ShapeKind::TDist => (80.0, 3.0, 100.0),
            ShapeKind::JohnDraper => (1.0, -3.0, 3.0),
        }
    }
}

/// What a shape node evaluates to: the transform, or one of its first partials
/// (only the symbolic partials builder produces the latter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeOut {
    Value,
    /// `∂h/∂η`.
    DEta,
    /// `∂h/∂shape`.
    DShape,
}

/// Below this `|x| = |λu|`, `expm1(x)/λ` is evaluated by its series (see
/// [`expm1_over`]); at λ = 0 exactly the closed form is `0/0`.
const EXPM1_SERIES_BELOW: f64 = 0.1;

/// `expm1(x)` over `T`: the value from libm's `exp_m1` (no cancellation), the
/// derivatives from `exp`, which are `expm1`'s. The `e − e.val()` term has value
/// exactly 0 and carries the jets.
fn expm1_g<T: PkNum>(x: T) -> T {
    let e = x.exp();
    (e - T::from_f64(e.val())) + T::from_f64(x.val().exp_m1())
}

/// `expm1(λ·u)/λ`, exact through λ = 0 (where it is `u`).
///
/// For `|λu| < 0.1` it is the series `u · Σ_{k<12} xᵏ/(k+1)!`, truncated at
/// `0.1¹²/13! ≈ 1.6e-22`. The closed form `expm1(x)/λ` has an exact value, but
/// its **jets** are a quotient: `∂/∂λ` cancels to `eps/x²` and the Hessian to
/// `eps/x³`, so it is used only where `x ≥ 0.1` keeps those near `1e-13`.
fn expm1_over<T: PkNum>(u: T, lambda: T) -> T {
    let x = lambda * u;
    if x.val().abs() < EXPM1_SERIES_BELOW {
        // Horner over the coefficients 1/(k+1)!, highest first.
        let mut coef = [0.0f64; 12];
        let mut fact = 1.0;
        for (k, c) in coef.iter_mut().enumerate() {
            fact *= (k + 1) as f64;
            *c = 1.0 / fact;
        }
        let mut acc = T::from_f64(coef[11]);
        for c in coef[..11].iter().rev() {
            acc = T::from_f64(*c) + x * acc;
        }
        u * acc
    } else {
        expm1_g(x) * recip(lambda)
    }
}

/// `1/b`. The kernels divide only through this, as `a · (1/b)`: a dual `a / b`
/// is `a × recip(b)`, so writing `a / b` would put the `f64` value one ulp off
/// the dual value of the same expression.
fn recip<T: PkNum>(b: T) -> T {
    T::from_f64(1.0) / b
}

/// The transform `h(η; s)` of kind `kind`.
pub(crate) fn shape_g<T: PkNum>(kind: ShapeKind, eta: T, s: T) -> T {
    let c = |v: f64| T::from_f64(v);
    match kind {
        ShapeKind::BoxCox => expm1_over(eta, s),
        // η·(1 + (η²+1)/(4ν) + (5η⁴+16η²+3)/(96ν²) + (3η⁶+19η⁴+17η²−15)/(384ν³)).
        ShapeKind::TDist => {
            let e2 = eta * eta;
            let e4 = e2 * e2;
            let e6 = e4 * e2;
            let inv = recip(s);
            let inv2 = inv * inv;
            let inv3 = inv2 * inv;
            eta * (c(1.0)
                + (e2 + c(1.0)) * c(0.25) * inv
                + (c(5.0) * e4 + c(16.0) * e2 + c(3.0)) * c(1.0 / 96.0) * inv2
                + (c(3.0) * e6 + c(19.0) * e4 + c(17.0) * e2 - c(15.0)) * c(1.0 / 384.0) * inv3)
        }
        // Odd by construction, branching on the sign rather than multiplying
        // by `sign(η)`: a dual `sign` is flat, so `η · 0` at η = 0 — the inner
        // loop's warm start — would hand it `∂h/∂η = 0` instead of 1.
        ShapeKind::JohnDraper => {
            let half = |u: T| expm1_over((c(1.0) + u).ln(), s);
            if eta.val() >= 0.0 {
                half(eta)
            } else {
                -half(-eta)
            }
        }
    }
}

/// The value of a shape node over `f64`: the transform, or one of its first
/// partials, which are the same kernel over `Dual1`.
pub(crate) fn shape_f64(kind: ShapeKind, out: ShapeOut, eta: f64, s: f64) -> f64 {
    match out {
        ShapeOut::Value => shape_g(kind, eta, s),
        ShapeOut::DEta | ShapeOut::DShape => {
            let d = shape_g(kind, Dual1::<2>::var(eta, 0), Dual1::<2>::var(s, 1));
            d.grad[if out == ShapeOut::DEta { 0 } else { 1 }]
        }
    }
}

// ── The `[eta_shape]` block ─────────────────────────────────────────────────

/// What the `[eta_shape]` desugaring needs to know about `[parameters]`.
pub(crate) struct EtaShapeContext<'a> {
    /// BSV ETA names, declaration order.
    pub(crate) eta_names: &'a [String],
    /// IOV kappa names.
    pub(crate) kappa_names: &'a [String],
    /// Every declared θ name.
    pub(crate) theta_names: &'a [String],
}

/// Blocks that compute a prediction or a likelihood and may read an η: a
/// shaped ETA read in one of them would see η where `[individual_parameters]`
/// sees `h(η)`, so it is refused. `[derived]` / `[output]` report a value, and
/// reporting the random effect itself there is meaningful.
const PREDICTION_BLOCKS: &[&str] = &[
    "structural_model",
    "odes",
    "scaling",
    "initial_conditions",
    "error_model",
    "event_model",
    "binary_model",
    "markov_model",
];

/// Whether `line` reads the identifier `name` as a whole word.
fn reads_word(line: &str, name: &str) -> bool {
    !word_spans(line, name).is_empty()
}

/// Byte ranges of `name` in `line` where it stands as a whole identifier.
fn word_spans(line: &str, name: &str) -> Vec<(usize, usize)> {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = line[from..].find(name) {
        let (s, e) = (from + i, from + i + name.len());
        let before = line[..s].chars().next_back().is_some_and(is_ident);
        let after = line[e..].chars().next().is_some_and(is_ident);
        if !before && !after {
            out.push((s, e));
        }
        from = e;
    }
    out
}

/// One parsed `[eta_shape]` line: `ETA ~ kind(SHAPE)`, `SHAPE` empty when the
/// θ is to be auto-declared.
struct ShapeLine {
    eta: String,
    kind: ShapeKind,
    shape: String,
}

fn parse_shape_line(line: &str) -> Result<ShapeLine, String> {
    let bad = || {
        format!(
            "[eta_shape]: cannot read `{line}` — write `ETA_NAME ~ boxcox(THETA)`, \
             `tdist(THETA)` or `johndraper(THETA)`, or leave the parentheses empty to \
             declare the shape θ automatically"
        )
    };
    let (eta, rhs) = line.split_once('~').ok_or_else(bad)?;
    let (eta, rhs) = (eta.trim(), rhs.trim());
    let (func, rest) = rhs.split_once('(').ok_or_else(bad)?;
    let shape = rest.strip_suffix(')').ok_or_else(bad)?.trim();
    let func = func.trim().to_ascii_lowercase();
    let kind = ShapeKind::from_name(&func).ok_or_else(|| {
        format!(
            "[eta_shape]: unknown shape `{func}` in `{line}` — the shapes are `boxcox`, \
             `tdist` and `johndraper`"
        )
    })?;
    if eta.is_empty() || !eta.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(bad());
    }
    Ok(ShapeLine {
        eta: eta.to_string(),
        kind,
        shape: shape.to_string(),
    })
}

/// Desugar `[eta_shape]` into the inline form (#1716): every whole-word read of
/// a shaped ETA in `[individual_parameters]` becomes `kind(ETA, SHAPE)`, and a
/// line with empty parentheses appends `theta {LAMBDA|NU}_{ETA}` to
/// `[parameters]` at [`ShapeKind::default_theta`]. The rest of the parser then
/// sees an ordinary model, so the two spellings cannot drift.
///
/// `other_blocks` are the remaining blocks as `(block type, lines)`, unnamed
/// and named (`[event_model NAME]`) alike, checked so that a shaped ETA is not
/// also read raw where a prediction is computed ([`PREDICTION_BLOCKS`]).
pub(crate) fn apply_eta_shape(
    block: &[String],
    parameters: &mut Vec<String>,
    individual_parameters: &mut [String],
    other_blocks: &[(&str, &[String])],
    cx: &EtaShapeContext<'_>,
) -> Result<(), String> {
    let mut seen: Vec<String> = Vec::new();
    let mut rewrites: Vec<(String, String)> = Vec::new();
    for line in block {
        let l = parse_shape_line(line)?;
        if cx.kappa_names.contains(&l.eta) {
            return Err(format!(
                "[eta_shape]: `{}` is an IOV kappa; shaping a kappa is not supported yet \
                 (#1717) — shape the ETA it is summed with, or write the transform inline",
                l.eta
            ));
        }
        if !cx.eta_names.contains(&l.eta) {
            return Err(format!(
                "[eta_shape]: `{}` is not a declared ETA (an `omega` in [parameters]); \
                 declared: {}",
                l.eta,
                cx.eta_names.join(", ")
            ));
        }
        if seen.contains(&l.eta) {
            return Err(format!(
                "[eta_shape]: `{}` is shaped twice — an ETA takes one shape",
                l.eta
            ));
        }
        seen.push(l.eta.clone());
        let shape = if l.shape.is_empty() {
            let prefix = match l.kind {
                ShapeKind::TDist => "NU",
                ShapeKind::BoxCox | ShapeKind::JohnDraper => "LAMBDA",
            };
            let name = format!("{prefix}_{}", l.eta);
            if cx.theta_names.contains(&name) {
                return Err(format!(
                    "[eta_shape]: `{}` would declare θ `{name}`, which [parameters] already \
                     declares — name it inside the parentheses, `{}({name})`",
                    l.eta,
                    l.kind.name()
                ));
            }
            let (init, lo, hi) = l.kind.default_theta();
            parameters.push(format!("theta {name}({init}, {lo}, {hi})"));
            name
        } else {
            l.shape
        };
        rewrites.push((l.eta, format!("{}({{}}, {shape})", l.kind.name())));
    }
    for &(key, lines) in other_blocks {
        if !PREDICTION_BLOCKS.contains(&key) {
            continue;
        }
        for (eta, _) in &rewrites {
            if let Some(line) = lines.iter().find(|l| reads_word(l, eta)) {
                return Err(format!(
                    "[eta_shape]: `{eta}` is shaped, but [{key}] reads it directly (`{line}`); \
                     the shape applies in [individual_parameters] only, so read it through an \
                     individual parameter there"
                ));
            }
        }
    }
    for line in individual_parameters.iter_mut() {
        for (eta, template) in &rewrites {
            let spans = word_spans(line, eta);
            if spans.is_empty() {
                continue;
            }
            let call = template.replacen("{}", eta, 1);
            let mut out = String::with_capacity(line.len() + spans.len() * call.len());
            let mut at = 0;
            for (s, e) in spans {
                out.push_str(&line[at..s]);
                out.push_str(&call);
                at = e;
            }
            out.push_str(&line[at..]);
            *line = out;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "eta_shape_tests.rs"]
mod tests;
