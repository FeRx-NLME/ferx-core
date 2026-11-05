//! Random-effect shape transforms on the η scale (#1716, tracker #359):
//! `boxcox(η, λ)`, `tdist(η, ν)` and `johndraper(η, λ)`, after Petersson et
//! al. 2009 (*Pharm Res* 26:2174) — the formulas Pharmpy's `transform_etas_*`
//! generate. η stays `N(0, ω)`; the model reads `h(η)` where it read η, the
//! NONMEM spelling `CL = TVCL * EXP(ETATR)`.
//!
//! Each transform is written **once**, as [`shape_g`] over [`PkNum`]: `T = f64`
//! is the value every evaluator computes, and the dual and taped types give the
//! exact `∂/∂η`, `∂/∂shape` the adjoint gradient consumes. There is no second copy
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
    if !e.val().is_finite() {
        // `inf − inf` would be NaN; past the overflow `expm1 = exp` (#1721 r1).
        return e;
    }
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
/// The smallest ν a `tdist` shape θ may take (#1731): Petersson et al.'s and Pharmpy's
/// lower bound, the one an auto-declared θ already gets.
pub(crate) const TDIST_NU_MIN: f64 = 3.0;

/// A `[parameters]` θ's lower bound, and its value when it is `FIX`ed: `(lo, Some(v))`
/// for `theta NAME(v) FIX`, `(lo, None)` otherwise, `−∞` for an unbounded θ. `None` when
/// no `theta NAME(…)` line declares it (a shape written as an expression).
fn theta_lower_bound(parameters: &[String], name: &str) -> Option<(f64, Option<f64>)> {
    for line in parameters {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("theta") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(after) = rest.strip_prefix(name) else {
            continue;
        };
        let after = after.trim_start();
        let Some(args) = after.strip_prefix('(') else {
            continue;
        };
        let close = args.find(')')?;
        let vals: Vec<f64> = args[..close]
            .split(',')
            .map(|v| v.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let fixed = args[close + 1..]
            .split_whitespace()
            .any(|w| w.eq_ignore_ascii_case("fix"));
        let lo = if vals.len() >= 3 {
            vals[1]
        } else {
            f64::NEG_INFINITY
        };
        return Some((lo, fixed.then(|| vals[0])));
    }
    None
}

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
            // #1731: `tdist` divides by ν, ν², ν³, so a user θ that may reach small ν lets
            // `h(η)` blow up. Refuse a lower bound below 3 (Petersson's and Pharmpy's), or a
            // FIXed value below it. A shape expression that is not a bare θ cannot be
            // checked here.
            if l.kind == ShapeKind::TDist {
                if let Some((lo, fixed_at)) = theta_lower_bound(parameters, &l.shape) {
                    let reach = fixed_at.unwrap_or(lo);
                    if !(reach >= TDIST_NU_MIN) {
                        return Err(format!(
                            "[eta_shape]: `{} ~ tdist({})` needs ν ≥ {TDIST_NU_MIN}, but θ `{}` \
                             may reach {reach} — `tdist` divides by ν³, so h(η) explodes as ν \
                             falls; declare it with a lower bound of at least {TDIST_NU_MIN}, \
                             e.g. `theta {}(10, {TDIST_NU_MIN}, 100)`",
                            l.eta, l.shape, l.shape, l.shape
                        ));
                    }
                }
            }
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

// ── `density` / `transform` on the omega line ───────────────────────────────

/// A `density KIND(ARGS)` tail on an `omega` / `block_omega` / `kappa` declaration
/// (#1716): what the ETA's distribution is. A `transform KIND(ARGS)` tail is read too, so
/// that it can be refused with the `[eta_shape]` line to write instead.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DistClause {
    /// `transform` (η ~ N, the model reads `h(η)`) rather than `density` (η itself
    /// has the density).
    pub(crate) transform: bool,
    /// The kind, lower-cased (`t`, `boxcox`, …), unchecked.
    pub(crate) kind: String,
    /// The comma-separated arguments, trimmed; empty for `kind()`.
    pub(crate) args: Vec<String>,
}

/// A `density` clause and the ETAs of the declaration it sits on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OmegaDensityDecl {
    pub(crate) etas: Vec<String>,
    /// The names are IOV κs (a `kappa` / `block_kappa` line), not BSV ETAs.
    pub(crate) kappa: bool,
    pub(crate) kind: String,
    pub(crate) args: Vec<String>,
    /// The declaration, for messages.
    pub(crate) line: String,
}

/// The byte range `[start, end)` of a `density …(…)` / `transform …(…)` tail in a
/// `[parameters]` line, and the clause. The keyword is matched as a whole word at
/// bracket depth 0, and not in a declaration's name slot (`omega density ~ 0.1` is an
/// ETA named `density`). Only a `prior(...)` tail may follow it.
pub(crate) fn find_dist_clause(line: &str) -> Result<Option<(usize, usize, DistClause)>, String> {
    let bytes = line.as_bytes();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut depth = 0i32;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            _ => {}
        }
        if depth != 0 || (i > 0 && is_ident(bytes[i - 1])) {
            continue;
        }
        let Some(kw) = ["density", "transform"].into_iter().find(|kw| {
            bytes.len() >= i + kw.len()
                && bytes[i..i + kw.len()].eq_ignore_ascii_case(kw.as_bytes())
                && !bytes.get(i + kw.len()).copied().is_some_and(is_ident)
        }) else {
            continue;
        };
        // In the name slot (right after the declaration keyword), it is a name.
        let before = line[..i].trim_end();
        let prev_word = before
            .rsplit(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        if before.len() == line[..i].len()
            || ["theta", "omega", "sigma", "kappa"]
                .contains(&prev_word.to_ascii_lowercase().as_str())
        {
            continue;
        }
        let bad = || {
            format!(
                "[parameters] cannot read the `{kw}` clause in `{}` — write `{kw} KIND`, or \
                 `{kw} KIND(SHAPE)` for a kind with a shape θ, e.g. \
                 `omega ETA_CL ~ 0.09 density gamma` or `density t(NU)`; \
                 `KIND()` declares the shape θ automatically",
                line.trim()
            )
        };
        let rest = &line[i + kw.len()..];
        // A kind without a shape may be written bare: `density gamma`.
        let word_end = rest
            .trim_start()
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(rest.len(), |e| e + (rest.len() - rest.trim_start().len()));
        let kind = rest[..word_end].trim();
        if kind.is_empty() || !kind.bytes().all(is_ident) {
            return Err(bad());
        }
        let after_kind = rest[word_end..].trim_start();
        if !after_kind.starts_with('(') {
            let prior_follows = after_kind.len() >= 5
                && after_kind[..5].eq_ignore_ascii_case("prior")
                && !after_kind.as_bytes().get(5).copied().is_some_and(is_ident);
            if !after_kind.is_empty() && !prior_follows {
                return Err(bad());
            }
            let end = i + kw.len() + word_end;
            let clause = DistClause {
                transform: kw == "transform",
                kind: kind.to_ascii_lowercase(),
                args: Vec::new(),
            };
            return Ok(Some((i, end, clause)));
        }
        let open_rel = rest.find('(').ok_or_else(bad)?;
        let open = i + kw.len() + open_rel;
        let mut d = 0i32;
        let mut close = None;
        for (j, &b) in bytes.iter().enumerate().skip(open) {
            match b {
                b'(' => d += 1,
                b')' => {
                    d -= 1;
                    if d == 0 {
                        close = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(bad)?;
        let after = line[close + 1..].trim_start();
        let prior_follows = after.len() >= 5
            && after[..5].eq_ignore_ascii_case("prior")
            && !after.as_bytes().get(5).copied().is_some_and(is_ident);
        if !after.is_empty() && !prior_follows {
            return Err(format!(
                "[parameters] the `{kw}` clause must end the declaration (only a `prior(...)` \
                 may follow it); found `{after}` after it in `{}`",
                line.trim()
            ));
        }
        let inner = line[open + 1..close].trim();
        let args = if inner.is_empty() {
            Vec::new()
        } else {
            split_top_level(inner)
        };
        // `truncated(KIND[(…)], LO, HI)` takes a kind and two bounds (numbers, `inf` or
        // θ names), read by `parse_truncated`; every other kind takes θ names.
        let truncated =
            kind.eq_ignore_ascii_case("truncated") || kind.eq_ignore_ascii_case("copula");
        if !truncated
            && args
                .iter()
                .any(|a| a.is_empty() || !a.bytes().all(is_ident))
        {
            return Err(format!(
                "[parameters] the `{kw}` clause takes θ names, comma-separated: `{}`",
                line.trim()
            ));
        }
        let clause = DistClause {
            transform: kw == "transform",
            kind: kind.to_ascii_lowercase(),
            args,
        };
        return Ok(Some((i, close + 1, clause)));
    }
    Ok(None)
}

/// `s` split at its top-level commas (not inside parentheses), each part trimmed.
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim().to_string());
    out
}

/// A kind spec `KIND` or `KIND(SHAPES)`: its (lower-cased) name and its shape arguments,
/// `None` without parentheses.
pub(crate) fn parse_kind_spec(spec: &str) -> Option<(String, Option<Vec<String>>)> {
    let spec = spec.trim();
    match spec.find('(') {
        Some(o) => {
            let inner = spec[o + 1..].strip_suffix(')')?.trim();
            let sh = if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|a| a.trim().to_string()).collect()
            };
            Some((spec[..o].trim().to_ascii_lowercase(), Some(sh)))
        }
        None => Some((spec.to_ascii_lowercase(), None)),
    }
}

/// A `truncated(KIND[(SHAPES)], LO, HI)` clause's parts: the inner kind's name, its shape
/// arguments (`None` without parentheses), and the two bounds as written.
pub(crate) fn parse_truncated(
    args: &[String],
    line: &str,
) -> Result<(String, Option<Vec<String>>, String, String), String> {
    let bad = || {
        format!(
            "[parameters] `truncated` takes a distribution and two bounds, e.g. \
             `density truncated(gamma, 0.2, 5)` or `truncated(t(NU), -1, inf)`: `{}`",
            line.trim()
        )
    };
    if args.len() != 3 {
        return Err(bad());
    }
    let spec = args[0].trim();
    let (name, shapes) = match spec.find('(') {
        Some(o) => {
            let inner = spec[o + 1..].strip_suffix(')').ok_or_else(bad)?.trim();
            let sh = if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|a| a.trim().to_string()).collect()
            };
            (spec[..o].trim().to_ascii_lowercase(), Some(sh))
        }
        None => (spec.to_ascii_lowercase(), None),
    };
    if name.is_empty() || name == "truncated" {
        return Err(bad());
    }
    Ok((name, shapes, args[1].clone(), args[2].clone()))
}

/// `line` without its distribution clause, and the clause.
pub(crate) fn split_dist_clause(line: &str) -> Result<(String, Option<DistClause>), String> {
    Ok(match find_dist_clause(line)? {
        Some((s, e, c)) => (format!("{}{}", line[..s].trim_end(), &line[e..]), Some(c)),
        None => (line.to_string(), None),
    })
}

/// The ETA names an `omega` / `block_omega` declaration line declares, or `None` for
/// any other line.
/// The random-effect names an `omega` / `block_omega` / `kappa` / `block_kappa`
/// declaration declares: whether it is a block, the names, and whether they are IOV κs.
fn declared_random_effects(line: &str) -> Option<(bool, Vec<String>, bool)> {
    let t = line.trim_start();
    let lower = t.to_ascii_lowercase();
    for (kw, kappa) in [("block_omega", false), ("block_kappa", true)] {
        if let Some(rest) = lower.strip_prefix(kw) {
            let open = rest.find('(')?;
            let close = rest.find(')')?;
            let names = t[kw.len() + open + 1..kw.len() + close]
                .split(',')
                .map(|n| n.trim().to_string())
                .collect();
            return Some((true, names, kappa));
        }
    }
    for (kw, kappa) in [("omega", false), ("kappa", true)] {
        if let Some(rest) = lower.strip_prefix(kw) {
            if !rest.starts_with(char::is_whitespace) {
                continue;
            }
            let name = t[kw.len()..]
                .trim_start()
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next()?;
            return Some((false, vec![name.to_string()], kappa));
        }
    }
    None
}

/// The keyword of a `[parameters]` line (`theta`, `kappa`, …), lower-cased.
pub(crate) fn decl_keyword(line: &str) -> String {
    line.trim_start()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// The text pre-pass for omega-line distributions (#1716), run on `[parameters]` before
/// anything parses it:
///
/// - an omega-line `transform KIND(ARGS)` or `density boxcox(λ)` is refused, naming the
///   `[eta_shape]` line to write: that block is a transform's one declarative home.
/// - `density KIND()` gets its shape θ named (`NU_<ETA>` / `LAMBDA_<ETA>`), and a
///   `theta` line is appended at [`crate::stats::eta_density::DensityKind::default_theta`].
///   The clause stays on the line, explicit, so every later parse of the text (the
///   absorption twin's included) reads the same density.
///
/// Returns the generated `[eta_shape]` lines (none since transforms left the omega line;
/// kept so the caller's splice stays one code path).
/// The message for an unknown `density` kind, listing the kinds.
pub(crate) fn unknown_density(kind: &str, line: &str) -> String {
    let kinds: Vec<&str> = crate::stats::eta_density::DensityKind::ALL
        .iter()
        .map(|k| k.name())
        .collect();
    format!(
        "[parameters] unknown density `{kind}` in `{}` — the densities are {} (and \
         `boxcox`, which is the Box-Cox transform)",
        line.trim(),
        kinds.join(", ")
    )
}

pub(crate) fn apply_omega_dist_clauses(
    parameters: &mut Vec<String>,
) -> Result<Vec<String>, String> {
    use crate::stats::eta_density::DensityKind;
    let theta_names: Vec<String> = parameters
        .iter()
        .filter(|l| decl_keyword(l) == "theta")
        .filter_map(|l| {
            l.trim_start()[5..]
                .trim_start()
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next()
                .map(str::to_string)
        })
        .collect();
    let shape_lines: Vec<String> = Vec::new();
    let mut new_thetas = Vec::new();
    for line in parameters.iter_mut() {
        let Some((s, e, clause)) = find_dist_clause(line)? else {
            continue;
        };
        let kw = if clause.transform {
            "transform"
        } else {
            "density"
        };
        let Some((_, etas, _)) = declared_random_effects(line) else {
            return Err(format!(
                "[parameters] a `{kw}` clause belongs on an `omega`, `block_omega`, `kappa` or \
                 `block_kappa` declaration: `{}`",
                line.trim()
            ));
        };
        // One declarative home for a shape transform (#1721 review): `[eta_shape]`, the
        // syntax agreed on #1716. The omega-line `transform` and the `density boxcox` alias
        // (on its latent scale the same model) would be a third and fourth spelling.
        if clause.transform || clause.kind == "boxcox" {
            return Err(format!(
                "[parameters] a shape transform is declared in [eta_shape] — `{} ~ {}({})` — \
                 not on the omega line: `{}`",
                etas.first().map_or("ETA", |e| e.as_str()),
                clause.kind,
                clause.args.join(", "),
                line.trim()
            ));
        }
        if clause.kind == "copula" {
            let specs = named_copula_margins(&clause.args, &etas, line)?;
            let mut rewritten = Vec::new();
            let mut changed = false;
            for (spec, eta) in specs.iter().zip(&etas) {
                let (name, shapes) = parse_kind_spec(spec).ok_or_else(|| {
                    format!(
                        "[parameters] cannot read the copula margin `{spec}`: `{}`",
                        line.trim()
                    )
                })?;
                let kind =
                    DensityKind::from_name(&name).ok_or_else(|| unknown_density(&name, line))?;
                if kind.shapes().is_empty() || shapes.as_ref().is_some_and(|v| !v.is_empty()) {
                    rewritten.push(format!("{eta} = {spec}"));
                    continue;
                }
                let mut names = Vec::new();
                for sp in kind.shapes() {
                    let n = format!("{}_{eta}", sp.prefix);
                    if theta_names.contains(&n) {
                        return Err(format!(
                            "[parameters] the copula margin `{name}()` would declare θ `{n}`, \
                             which [parameters] already declares — name it: `{}`",
                            line.trim()
                        ));
                    }
                    new_thetas.push(format!(
                        "theta {n}({}, {}, {})",
                        sp.init, sp.lower, sp.upper
                    ));
                    names.push(n);
                }
                rewritten.push(format!("{eta} = {name}({})", names.join(", ")));
                changed = true;
            }
            if changed {
                *line = format!(
                    "{} density copula({}){}",
                    line[..s].trim_end(),
                    rewritten.join(", "),
                    &line[e..]
                );
            }
            continue;
        }
        if clause.kind == "truncated" {
            let (name, shapes, lo, hi) = parse_truncated(&clause.args, line)?;
            let kind = DensityKind::from_name(&name).ok_or_else(|| unknown_density(&name, line))?;
            if kind.shapes().is_empty() || shapes.as_ref().is_some_and(|v| !v.is_empty()) {
                continue;
            }
            let mut names = Vec::new();
            for sp in kind.shapes() {
                let n = format!("{}_{}", sp.prefix, etas[0]);
                if theta_names.contains(&n) {
                    return Err(format!(
                        "[parameters] `density truncated({name}(), …)` would declare θ `{n}`, \
                         which [parameters] already declares — name it inside the parentheses: \
                         `{}`",
                        line.trim()
                    ));
                }
                new_thetas.push(format!(
                    "theta {n}({}, {}, {})",
                    sp.init, sp.lower, sp.upper
                ));
                names.push(n);
            }
            *line = format!(
                "{} density truncated({name}({}), {lo}, {hi}){}",
                line[..s].trim_end(),
                names.join(", "),
                &line[e..]
            );
            continue;
        }
        let kind = DensityKind::from_name(&clause.kind)
            .ok_or_else(|| unknown_density(&clause.kind, line))?;
        let specs = kind.shapes();
        if specs.is_empty() || !clause.args.is_empty() {
            continue;
        }
        let names: Vec<String> = specs
            .iter()
            .map(|sp| format!("{}_{}", sp.prefix, etas[0]))
            .collect();
        for (name, sp) in names.iter().zip(specs) {
            let (init, lo, hi) = (sp.init, sp.lower, sp.upper);
            if theta_names.contains(name)
                || new_thetas
                    .iter()
                    .any(|t: &String| t.contains(&format!(" {name}(")))
            {
                return Err(format!(
                    "[parameters] `density {}()` would declare θ `{name}`, which [parameters] \
                     already declares — name it inside the parentheses: `{}`",
                    clause.kind,
                    line.trim()
                ));
            }
            new_thetas.push(format!("theta {name}({init}, {lo}, {hi})"));
        }
        let _ = &names;
        *line = format!(
            "{} density {}({}){}",
            line[..s].trim_end(),
            clause.kind,
            names.join(", "),
            &line[e..]
        );
    }
    parameters.extend(new_thetas);
    Ok(shape_lines)
}

/// A `copula(ETA_A = KIND, ETA_B = KIND(…), …)` clause's margins, in the block's ETA
/// order. The margins are named, not positional, so reordering a `block_omega` cannot
/// silently re-bind them (#1721 review): every ETA of the block must be named exactly once.
pub(crate) fn named_copula_margins(
    args: &[String],
    etas: &[String],
    line: &str,
) -> Result<Vec<String>, String> {
    let mut by_eta: Vec<Option<String>> = vec![None; etas.len()];
    for a in args {
        let Some((name, spec)) = a.split_once('=') else {
            return Err(format!(
                "[parameters] copula margins are named — `copula({} = gamma, …)` — so a \
                 reordered block cannot re-bind them; `{a}` names no ETA: `{}`",
                etas.first().map_or("ETA", |e| e.as_str()),
                line.trim()
            ));
        };
        let name = name.trim();
        let Some(k) = etas.iter().position(|e| e.eq_ignore_ascii_case(name)) else {
            return Err(format!(
                "[parameters] the copula margin `{name}` is not an ETA of this block ({}): `{}`",
                etas.join(", "),
                line.trim()
            ));
        };
        if by_eta[k].replace(spec.trim().to_string()).is_some() {
            return Err(format!(
                "[parameters] the copula names `{name}` twice: `{}`",
                line.trim()
            ));
        }
    }
    by_eta
        .into_iter()
        .zip(etas)
        .map(|(m, e)| {
            m.ok_or_else(|| {
                format!(
                    "[parameters] the copula gives `{e}` no margin; name every ETA of the \
                     block: `{}`",
                    line.trim()
                )
            })
        })
        .collect()
}

/// Every `density` clause in `[parameters]`, with the ETAs of its declaration. Run after
/// [`apply_omega_dist_clauses`], so a `transform` is gone and every argument is named.
pub(crate) fn omega_density_decls(parameters: &[String]) -> Result<Vec<OmegaDensityDecl>, String> {
    let mut out = Vec::new();
    for line in parameters {
        let Some((_, _, clause)) = find_dist_clause(line)? else {
            continue;
        };
        if clause.transform {
            continue;
        }
        if let Some((_, etas, kappa)) = declared_random_effects(line) {
            out.push(OmegaDensityDecl {
                etas,
                kappa,
                kind: clause.kind,
                args: clause.args,
                line: line.trim().to_string(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "eta_shape_tests.rs"]
mod tests;
