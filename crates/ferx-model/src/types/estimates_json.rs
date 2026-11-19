//! The parameter-estimate sections of a fit's output, built once and rendered twice
//! (#2911): as the `theta:` / `omega:` / `sigma:` / `block_sigma:` / `omega_iov:` /
//! `mixture_overrides:` sections of `{model}-fit.yaml`, and as the `estimates` object of
//! `{model}-fit.json` ([`FitResult::to_json_value`]). The YAML also writes them a second
//! time, nested under its own `estimates:` key, so the one path
//! `estimates.<section>.<name>.<field>` reads both files.
//!
//! Before #2911 the JSON carried only `FitResult`'s own fields — bare `theta` vectors
//! beside parallel `theta_names` lists, Ω as a flat `{rows, cols, data}` matrix — so
//! anything reading final estimates back had to branch on which file it was handed. Now
//! `x["estimates"]["theta"]["CL"]["estimate"]` reads either file (the YAML keeps its
//! top-level `theta:` … sections too, as aliases of its `estimates:` block):
//! the same sections, entry names and field names, because both are rendered from the one
//! [`Section`] list built here. They differ only in precision — the YAML rounds as it
//! always has (`{:.6}`, `{:.2}` for a percentage) and writes `~` for "none", the JSON
//! carries the full `f64` and writes `null`, including for a non-finite value, matching
//! [`FitResult::to_json_value`]'s convention for every other float.

use crate::types::*;

/// One value of an estimate entry, tagged with how the YAML prints it.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A number, `{:.6}` in the YAML.
    Num(f64),
    /// A percentage (`rse_pct`, a σ `cv_pct`), `{:.2}` in the YAML.
    Pct(f64),
    /// A random-effect CV%: `{:.2}`, or `.inf` when it overflows ([`yaml_cv_pct`]).
    Cv(f64),
    /// No value: `~` in the YAML, `null` in the JSON.
    Null,
    Bool(bool),
    Int(usize),
    /// A bare word from ferx's own vocabulary: a σ `type`, a `held` reason, a `kind`.
    Token(String),
    /// A name the model file chose (a `mixture_overrides` entry's `base`): quoted under
    /// `estimates:`, and in a top-level section only when YAML would misread it bare
    /// ([`yaml_name`]).
    Name(String),
    /// Free-form text, quoted in the YAML ([`yaml_quote`]).
    Text(String),
}

impl Value {
    /// An optional number: [`Value::Num`] or [`Value::Null`].
    fn num_or_null(v: Option<f64>) -> Value {
        v.map_or(Value::Null, Value::Num)
    }

    /// The value as a YAML scalar. `quote_names` is set under `estimates:`, where every
    /// [`Value::Name`] is quoted.
    pub fn yaml(&self, quote_names: bool) -> String {
        match self {
            Value::Num(v) => format!("{v:.6}"),
            Value::Pct(v) => format!("{v:.2}"),
            Value::Cv(v) => yaml_cv_pct(*v),
            Value::Null => "~".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int(n) => n.to_string(),
            Value::Token(s) => s.clone(),
            Value::Name(s) => yaml_name(s, quote_names),
            Value::Text(s) => yaml_quote(s),
        }
    }

    pub fn json(&self) -> serde_json::Value {
        let float =
            |v: f64| serde_json::Number::from_f64(v).map_or(serde_json::Value::Null, Into::into);
        match self {
            Value::Num(v) | Value::Pct(v) | Value::Cv(v) => float(*v),
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => (*b).into(),
            Value::Int(n) => (*n).into(),
            Value::Token(s) | Value::Name(s) | Value::Text(s) => s.clone().into(),
        }
    }
}

/// A name the model file chose (a θ, η, σ or κ name, an `A__B` off-diagonal, a base
/// name) as a YAML **block** key or scalar: always quoted when `always` (under
/// `estimates:`), otherwise only when YAML 1.1 — which R's `yaml` package implements —
/// would read it bare as something other than that string, or could not read it at all.
///
/// Every legal identifier is a plain scalar except the boolean and null spellings: `YES`,
/// `no`, `On`, `OFF`, `y`, `N`, `TRUE`, `NULL`, … would come back as `TRUE` / `FALSE` /
/// `NULL` and address a different key than the JSON's `"YES"` (#2911). The check is
/// case-insensitive, so it also quotes a few mixed-case spellings YAML would read as
/// strings; that costs only the quotes.
///
/// A level name (`TEFF[TIME=0.5]`) is not an identifier, but in block context a plain
/// scalar may contain `[`, `]`, `=` and `,` as long as it does not *start* with an
/// indicator, so it stays bare. What is quoted is anything that would change the
/// document's shape: a leading indicator or space, `: ` or ` #` inside, a trailing `:`
/// or space, or a control character ([`needs_quotes_in_block`]). In a flow sequence
/// (`[a, b]`) more characters are structural — use [`yaml_flow_name`] there (#3006).
pub fn yaml_name(name: &str, always: bool) -> String {
    if always || needs_quotes_in_block(name) {
        yaml_quote(name)
    } else {
        name.to_string()
    }
}

/// [`yaml_name`] for an element of a YAML **flow** sequence (`parameters: [A, B]`). Inside
/// `[...]` the flow indicators `[`, `]`, `{`, `}` and `,` are structural even mid-scalar,
/// so `[INT, TEFF[TIME=0.5]]` opens a nested sequence and the document fails to parse
/// (#3006). Such a name — and one holding `:`, which YAML 1.1 parsers treat
/// inconsistently inside flow context — is quoted; a plain identifier stays bare.
pub fn yaml_flow_name(name: &str) -> String {
    const FLOW: &[char] = &['[', ']', '{', '}', ',', ':'];
    if needs_quotes_in_block(name) || name.contains(FLOW) {
        yaml_quote(name)
    } else {
        name.to_string()
    }
}

/// A comma-joined YAML flow sequence of names, each through [`yaml_flow_name`].
pub fn yaml_flow_names<S: AsRef<str>>(names: &[S]) -> String {
    let items: Vec<String> = names.iter().map(|n| yaml_flow_name(n.as_ref())).collect();
    format!("[{}]", items.join(", "))
}

/// Whether `name` must be quoted to round-trip as that string in YAML block context.
fn needs_quotes_in_block(name: &str) -> bool {
    const IMPLICIT: &[&str] =
        &["y", "n", "yes", "no", "true", "false", "on", "off", "null", "~", ""];
    // Characters that start a non-plain node, or are reserved, at the head of a scalar.
    const INDICATORS: &[char] = &[
        '-', '?', ':', ',', '[', ']', '{', '}', '#', '&', '*', '!', '|', '>', '\'', '"', '%', '@',
        '`',
    ];
    IMPLICIT.contains(&name.to_ascii_lowercase().as_str())
        || name.starts_with(INDICATORS)
        || name.starts_with(char::is_whitespace)
        || name.ends_with(char::is_whitespace)
        || name.ends_with(':')
        || name.contains(": ")
        || name.contains(" #")
        || name.contains(char::is_control)
}

/// One named entry of a section: a parameter (`CL`), or an off-diagonal (`A__B`).
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: String,
    /// In the order the YAML writes them.
    pub fields: Vec<(&'static str, Value)>,
}

impl Entry {
    fn new(name: impl Into<String>) -> Self {
        Entry { name: name.into(), fields: Vec::new() }
    }

    fn push(&mut self, key: &'static str, value: Value) {
        self.fields.push((key, value));
    }

    /// The `se` field, and the `fixed:` / `held:` key that explains a missing one, in the
    /// order every family writes them: `fixed: true` before `se: ~`, `held:` after it.
    fn push_se(&mut self, se: SeState) {
        match se {
            SeState::Fixed => {
                self.push("fixed", Value::Bool(true));
                self.push("se", Value::Null);
            }
            SeState::Held(reason) => {
                self.push("se", Value::Null);
                self.push("held", Value::Token(reason.as_str().to_string()));
            }
            SeState::Free(se) => self.push("se", Value::num_or_null(se)),
        }
    }
}

/// Why a parameter does or does not report an SE, in render precedence.
#[derive(Debug, Clone, Copy)]
enum SeState {
    Fixed,
    Held(HeldReason),
    /// Free: its SE, or `None` when the covariance step produced none.
    Free(Option<f64>),
}

impl SeState {
    /// A base θ/Ω/σ/κ's state: FIX first, then [`se_held`], else its SE.
    fn of(result: &FitResult, kind: ParameterKind, i: usize, fixed: bool, se: Option<f64>) -> Self {
        if fixed {
            SeState::Fixed
        } else if let Some(reason) = se_held(result, kind, i) {
            SeState::Held(reason)
        } else {
            SeState::Free(se)
        }
    }
}

/// One top-level section: its key, an optional comment on the header line, its entries.
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub key: &'static str,
    /// Written after the header in the YAML (`sigma:  # error model: …`); not in the JSON.
    pub comment: Option<String>,
    pub entries: Vec<Entry>,
}

impl Section {
    fn new(key: &'static str) -> Self {
        Section { key, comment: None, entries: Vec::new() }
    }

    /// The section as a top-level YAML key, from the blank line before its header to its
    /// last field.
    pub fn yaml(&self) -> String {
        format!("\n{}", self.yaml_at(""))
    }

    /// The section's header and entries, every line prefixed by `indent`.
    ///
    /// Under `estimates:` (a non-empty `indent`) every user-chosen name is quoted; the
    /// top-level copy quotes only the names YAML would misread bare, keeping its bytes.
    fn yaml_at(&self, indent: &str) -> String {
        use std::fmt::Write;
        let nested = !indent.is_empty();
        let mut out = format!("{indent}{}:", self.key);
        // An empty section nested under `estimates:` is `{}`, so a YAML parser reads the
        // empty map the JSON has rather than `null`. The top-level copy keeps its old bytes.
        if self.entries.is_empty() && !indent.is_empty() {
            out.push_str(" {}");
        }
        if let Some(c) = &self.comment {
            let _ = write!(out, "  # {c}");
        }
        out.push('\n');
        for e in &self.entries {
            let _ = writeln!(out, "{indent}  {}:", yaml_name(&e.name, nested));
            for (k, v) in &e.fields {
                let _ = writeln!(out, "{indent}    {k}: {}", v.yaml(nested));
            }
        }
        out
    }

    /// The section as a JSON object keyed by entry name, each entry an object keyed by
    /// field name. Keys sort (the crate's `serde_json` has no `preserve_order`), which a
    /// keyed reader does not see.
    pub fn json(&self) -> serde_json::Value {
        let entries = self
            .entries
            .iter()
            .map(|e| {
                let fields = e
                    .fields
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.json()))
                    .collect::<serde_json::Map<_, _>>();
                (e.name.clone(), serde_json::Value::Object(fields))
            })
            .collect::<serde_json::Map<_, _>>();
        serde_json::Value::Object(entries)
    }
}

/// The θ indices the `theta:` section lists, in order: every θ except a `[covariate_nn]`
/// weight and a coefficient of a compacted `theta NAME[COL]` level block, which the YAML
/// summarizes under `neural_networks:` / `theta_blocks:` instead. Shared with the
/// `[priors] from_fit` reader, so it imports exactly the set a fit file lists.
pub fn listed_theta_indices(result: &FitResult) -> Vec<usize> {
    #[cfg(feature = "nn")]
    let nn: std::collections::HashSet<usize> = result
        .neural_networks
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.n_weights)
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let blocked: std::collections::HashSet<usize> =
        compact_theta_blocks(&result.theta_names).into_iter().flat_map(|(_, r)| r).collect();
    (0..result.theta_names.len()).filter(|i| !nn.contains(i) && !blocked.contains(i)).collect()
}

/// The spread field of a random-effect variance on its η type's scale: `cv_pct` for
/// log-normal and unknown, `sd` for additive and logit, none for a custom expression.
pub fn spread_field(t: Option<EtaParamType>, var: f64) -> Option<(&'static str, Value)> {
    match t {
        None | Some(EtaParamType::LogNormal) => Some(("cv_pct", Value::Cv(cv_pct(var)))),
        Some(EtaParamType::Additive | EtaParamType::Logit | EtaParamType::LogitProbability) => {
            Some(("sd", Value::Num(sd_from_var(var))))
        }
        Some(EtaParamType::Custom) => None,
    }
}

/// A weighted κ's `cv_pct_at_typical_weight`: the CV% of the typical arm's variance
/// `var_at`, for log-normal and unknown types only — the exact CV% is not
/// `100 · sd_at_typical_weight` (#1858), while for the other types that SD already is
/// the console's figure.
pub fn typical_cv_field(t: Option<EtaParamType>, var_at: f64) -> Option<(&'static str, Value)> {
    matches!(t, None | Some(EtaParamType::LogNormal))
        .then(|| ("cv_pct_at_typical_weight", Value::Cv(cv_pct(var_at))))
}

/// The `theta:` section: `estimate`, then `fixed`/`at_bound`, `se`, `rse_pct`, `held`.
pub fn theta_section(result: &FitResult) -> Section {
    let mut section = Section::new("theta");
    for i in listed_theta_indices(result) {
        let est = result.theta[i];
        let fixed = result.theta_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_theta.as_ref().and_then(|v| v.get(i).copied());
        let mut e = Entry::new(result.theta_names[i].clone());
        e.push("estimate", Value::Num(est));
        match SeState::of(result, ParameterKind::Theta, i, fixed, se) {
            SeState::Fixed => {
                e.push("fixed", Value::Bool(true));
                e.push("se", Value::Null);
                e.push("rse_pct", Value::Null);
            }
            SeState::Held(reason) => {
                if reason == HeldReason::ProfiledBound {
                    e.push("at_bound", Value::Bool(true));
                }
                e.push("se", Value::Null);
                e.push("rse_pct", Value::Null);
                e.push("held", Value::Token(reason.as_str().to_string()));
            }
            SeState::Free(Some(s)) => {
                e.push("se", Value::Num(s));
                e.push("rse_pct", Value::Pct(rse_pct(est, s)));
            }
            SeState::Free(None) => {
                e.push("se", Value::Null);
                e.push("rse_pct", Value::Null);
            }
        }
        section.entries.push(e);
    }
    section
}

/// The `omega:` section: one entry per η (`variance`, its spread, the SE), then one
/// `ETA_I__ETA_J` entry per non-zero covariance (`covariance`, `se`, `correlation`).
pub fn omega_section(result: &FitResult) -> Section {
    let mut section = Section::new("omega");
    let n_eta = result.omega.nrows();
    for i in 0..n_eta {
        let var = result.omega[(i, i)];
        let fixed = result.omega_fixed.get(i).copied().unwrap_or(false);
        let se = crate::types::omega_se_at(&result.se_omega, n_eta, i, i);
        let name = result
            .eta_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("omega_{}_{}", i + 1, i + 1));
        let mut e = Entry::new(name.clone());
        e.push("variance", Value::Num(var));
        if let Some((k, v)) = spread_field(eta_type(result, &name), var) {
            e.push(k, v);
        }
        e.push_se(SeState::of(result, ParameterKind::Omega, i, fixed, se));
        section.entries.push(e);
    }
    let eta =
        |i: usize| result.eta_names.get(i).cloned().unwrap_or_else(|| format!("eta_{}", i + 1));
    for i in 0..n_eta {
        for j in 0..i {
            let cov = result.omega[(i, j)];
            if cov.abs() <= 1e-15 {
                continue;
            }
            let corr = param_corr_fallback(
                result.omega_param_corr.as_ref(),
                cov,
                result.omega[(i, i)],
                result.omega[(j, j)],
                i,
                j,
            );
            let mut e = Entry::new(format!("{}__{}", eta(i), eta(j)));
            e.push("covariance", Value::Num(cov));
            e.push(
                "se",
                Value::num_or_null(crate::types::omega_se_at(&result.se_omega, n_eta, i, j)),
            );
            e.push("correlation", Value::Num(corr));
            section.entries.push(e);
        }
    }
    section
}

/// The `sigma:` section. σ is stored as an SD, so `estimate` is the SD and `variance` its
/// square for both component types; `cv_pct` (`σ·100`) only for a proportional one, where
/// it is the coefficient of variation directly — an additive σ is in observation units.
pub fn sigma_section(result: &FitResult) -> Section {
    let mut section = Section::new("sigma");
    section.comment = Some(format!(
        "error model: {}",
        match result.error_model {
            ErrorModel::Additive => "additive",
            ErrorModel::Proportional => "proportional",
            ErrorModel::Combined => "combined",
            ErrorModel::LogCombined => "log_combined",
        }
    ));
    for (i, &s) in result.sigma.iter().enumerate() {
        let fixed = result.sigma_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_sigma.as_ref().and_then(|v| v.get(i).copied());
        let name = result.sigma_names.get(i).cloned().unwrap_or_else(|| format!("sigma_{}", i + 1));
        let sigma_type = result.sigma_types.get(i).copied();
        let mut e = Entry::new(name);
        e.push("estimate", Value::Num(s));
        e.push("variance", Value::Num(s * s));
        e.push(
            "type",
            Value::Token(
                match sigma_type {
                    Some(SigmaType::Proportional) => "proportional",
                    Some(SigmaType::Additive) => "additive",
                    None => "unknown",
                }
                .to_string(),
            ),
        );
        if matches!(sigma_type, Some(SigmaType::Proportional)) {
            e.push("cv_pct", Value::Pct(s * 100.0));
        }
        e.push_se(SeState::of(result, ParameterKind::Sigma, i, fixed, se));
        section.entries.push(e);
    }
    section
}

/// The `block_sigma:` section, one `SIGMA_I__SIGMA_J` entry per residual correlation, or
/// `None` without one. An error names a correlation whose σ index is out of bounds.
pub fn block_sigma_section(result: &FitResult) -> Result<Option<Section>, String> {
    if result.residual_correlations.is_empty() {
        return Ok(None);
    }
    let mut section = Section::new("block_sigma");
    for (k, corr) in result.residual_correlations.iter().enumerate() {
        let name_i = result.sigma_names.get(corr.sigma_i).ok_or_else(|| {
            format!("residual correlation sigma_i index {} is out of bounds", corr.sigma_i)
        })?;
        let name_j = result.sigma_names.get(corr.sigma_j).ok_or_else(|| {
            format!("residual correlation sigma_j index {} is out of bounds", corr.sigma_j)
        })?;
        let sigma_i = result.sigma.get(corr.sigma_i).ok_or_else(|| {
            format!("residual correlation sigma_i index {} has no estimate", corr.sigma_i)
        })?;
        let sigma_j = result.sigma.get(corr.sigma_j).ok_or_else(|| {
            format!("residual correlation sigma_j index {} has no estimate", corr.sigma_j)
        })?;
        // Since #847 a bare `block_sigma (...) = [...]` **estimates** the off-diagonal
        // (NONMEM `$SIGMA BLOCK(n)`); only `... FIX` pins it. A pre-#847 `FitResult`
        // carries no flags at all, and back then every correlation was fixed by
        // construction — hence the `true` default.
        let correlation_fixed = result.residual_correlation_fixed.get(k).copied().unwrap_or(true);
        // The covariance `rho·sigma_i·sigma_j` is fixed only when *all three* factors
        // are: a free rho or a free SD moves it during the fit.
        let sigma_is_fixed = |i: usize| result.sigma_fixed.get(i).copied().unwrap_or(false);
        let covariance_fixed =
            correlation_fixed && sigma_is_fixed(corr.sigma_i) && sigma_is_fixed(corr.sigma_j);
        // SE on the natural rho scale; none for a FIXed correlation and for a fit whose
        // covariance step did not run.
        let corr_se = if correlation_fixed {
            None
        } else {
            result.se_residual_correlations.as_ref().and_then(|v| v.get(k)).copied()
        };
        let mut e = Entry::new(format!("{name_i}__{name_j}"));
        e.push("covariance", Value::Num(corr.rho * sigma_i * sigma_j));
        e.push("covariance_fixed", Value::Bool(covariance_fixed));
        e.push("correlation", Value::Num(corr.rho));
        e.push("correlation_fixed", Value::Bool(correlation_fixed));
        e.push("correlation_se", Value::num_or_null(corr_se));
        section.entries.push(e);
    }
    Ok(Some(section))
}

/// The `omega_iov:` section (κ), or `None` for a fit without IOV: one entry per κ, then
/// one `KAPPA_I__KAPPA_J` entry per non-zero covariance.
pub fn omega_iov_section(result: &FitResult) -> Option<Section> {
    let iov = result.omega_iov.as_ref()?;
    let mut section = Section::new("omega_iov");
    let n_kappa = iov.nrows();
    let kappa =
        |i: usize| result.kappa_names.get(i).cloned().unwrap_or_else(|| format!("kappa_{}", i + 1));
    for i in 0..n_kappa {
        let var = iov[(i, i)];
        let fixed = result.kappa_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_kappa.as_ref().and_then(|v| v.get(i).copied());
        let kappa_type = result.kappa_param_types.get(i).copied();
        let mut e = Entry::new(kappa(i));
        // Nested levels (#1082 PR4): the level the κ is drawn at.
        if let Some(level) = result.kappa_levels.get(i) {
            e.push("level", Value::Text(level.clone()));
        }
        e.push("variance", Value::Num(var));
        if let Some((k, v)) = spread_field(kappa_type, var) {
            e.push(k, v);
        }
        // A weighted κ (#1660): `variance` / its spread are the weight-1 γ², so add the
        // weight and the typical arm's spread the console prints: its SD always, and for
        // a CV% type its exact CV% (#1858).
        if let Some((w, typical)) =
            kappa_weight_facts(&result.kappa_weights, &result.kappa_weight_typical, i, var)
        {
            e.push("weight", Value::Text(w.to_string()));
            if let Some((n, var_at)) = typical {
                e.push("weight_typical", Value::Num(n));
                e.push("sd_at_typical_weight", Value::Num(sd_from_var(var_at)));
                if let Some((k, v)) = typical_cv_field(kappa_type, var_at) {
                    e.push(k, v);
                }
            }
        }
        e.push_se(SeState::of(result, ParameterKind::Kappa, i, fixed, se));
        section.entries.push(e);
    }
    for i in 0..n_kappa {
        for j in 0..i {
            let cov = iov[(i, j)];
            if cov.abs() <= 1e-15 {
                continue;
            }
            let corr = param_corr_fallback(
                result.omega_iov_param_corr.as_ref(),
                cov,
                iov[(i, i)],
                iov[(j, j)],
                i,
                j,
            );
            let mut e = Entry::new(format!("{}__{}", kappa(i), kappa(j)));
            e.push("covariance", Value::Num(cov));
            e.push("correlation", Value::Num(corr));
            section.entries.push(e);
        }
    }
    Some(section)
}

/// The `mixture_overrides:` section (#2833), or `None` without an override. Its own
/// section, never nested under `omega:` / `sigma:`: the `[priors] from_fit` reader takes
/// every key there as a base parameter. Each entry carries its `class`, `base` name and
/// `kind`, an Ω override's `variance` (and the spread field of its base η's type) or a σ
/// override's `estimate`, then `se` with the same `fixed:` / `held:` keys the base
/// sections use.
pub fn mixture_overrides_section(result: &FitResult) -> Option<Section> {
    let rows = override_rows(result);
    if rows.is_empty() {
        return None;
    }
    let mut section = Section::new("mixture_overrides");
    for row in &rows {
        let p = row.param;
        let base = row.base_name(result);
        let mut e = Entry::new(p.name.clone());
        e.push("class", Value::Int(p.class));
        e.push("base", Value::Name(base.to_string()));
        match p.kind {
            ParameterKind::Omega => {
                e.push("kind", Value::Token("omega".into()));
                e.push("variance", Value::Num(row.value));
                if let Some((k, v)) = spread_field(eta_type(result, base), row.value) {
                    e.push(k, v);
                }
            }
            _ => {
                e.push("kind", Value::Token("sigma".into()));
                e.push("estimate", Value::Num(row.value));
            }
        }
        e.push_se(match row.se {
            OverrideSe::Fixed => SeState::Fixed,
            OverrideSe::Held(reason) => SeState::Held(reason),
            OverrideSe::Free(se) => SeState::Free(se),
        });
        section.entries.push(e);
    }
    Some(section)
}

/// Every estimate section of `result`, in the order the fit YAML writes them.
pub fn estimate_sections(result: &FitResult) -> Result<Vec<Section>, String> {
    let mut out = vec![theta_section(result), omega_section(result), sigma_section(result)];
    out.extend(block_sigma_section(result)?);
    out.extend(omega_iov_section(result));
    out.extend(mixture_overrides_section(result));
    Ok(out)
}

/// The `estimates:` block of the fit YAML (#2911): every section of [`estimate_sections`]
/// again, nested one level under `estimates`, so `estimates.theta.CL.estimate` reads the
/// same in `{model}-fit.yaml` and `{model}-fit.json`. The top-level `theta:` / `omega:` /
/// … sections stay where they were, for the readers that already use them.
pub fn estimates_yaml(result: &FitResult) -> Result<String, String> {
    let mut out = String::from("\nestimates:\n");
    for s in estimate_sections(result)? {
        out.push_str(&s.yaml_at("  "));
    }
    Ok(out)
}

/// The `estimates` object of the fit JSON (#2911): each section of [`estimate_sections`]
/// under its YAML key. A result whose `block_sigma` cannot be rendered (a correlation
/// naming a σ index the result does not have, on which the YAML writer fails) gets
/// `estimates: null` rather than a partial table.
pub fn estimates_json(result: &FitResult) -> serde_json::Value {
    match estimate_sections(result) {
        Ok(sections) => serde_json::Value::Object(
            sections.iter().map(|s| (s.key.to_string(), s.json())).collect(),
        ),
        Err(_) => serde_json::Value::Null,
    }
}

/// Relative standard error as a percent: `|se / est| · 100`, or `NaN` when the
/// estimate is ~0 (guards the divide-by-zero for a parameter pinned at 0).
pub fn rse_pct(est: f64, se: f64) -> f64 {
    if est.abs() > 1e-12 {
        (se / est.abs()) * 100.0
    } else {
        f64::NAN
    }
}

/// Why parameter `i` of family `kind` has no standard error (#2797), or `None`: it is
/// in `FitResult::held_at_bound` (a profiled level on its bound, #2112, or a coordinate
/// collapsed on the internal guard). The one predicate every renderer reads. A
/// hand-built result setting only the deprecated `profiled_levels_at_bound` is honoured.
pub fn se_held(result: &FitResult, kind: ParameterKind, i: usize) -> Option<HeldReason> {
    result
        .held_at_bound
        .iter()
        // A `[mixture]` override's `(kind, index)` names its base parameter (#2809).
        .find(|h| h.class.is_none() && h.kind == kind && h.index == i)
        .map(|h| h.reason)
        .or_else(|| {
            (kind == ParameterKind::Theta && result.profiled_levels_at_bound.contains(&i))
                .then_some(HeldReason::ProfiledBound)
        })
}

/// The CV% of `P = TV · exp(η)`, `η ~ N(0, var)`: `sqrt(exp(var) − 1) · 100` exactly
/// (#1858), `0` for a non-positive variance. Not the first-order `sqrt(var) · 100`
/// (0.2: 44.7 vs 47.1; 2.37: 154 vs 311). `exp_m1` keeps a tiny variance's digits;
/// above `var ≈ 709.78` this is `+∞` (YAML: [`yaml_cv_pct`]).
pub fn cv_pct(var: f64) -> f64 {
    if var > 0.0 {
        var.exp_m1().sqrt() * 100.0
    } else {
        0.0
    }
}

/// A CV% as a YAML scalar: `{:.2}` when finite, else `.inf` — YAML's own
/// spelling of infinity, which a numeric reader parses as a float. Rust's
/// `inf` would be read back as a string.
pub fn yaml_cv_pct(cv: f64) -> String {
    if cv.is_finite() {
        format!("{:.2}", cv)
    } else {
        ".inf".to_string()
    }
}

/// SD from a variance: `sqrt(var)`, or `0` for a non-positive variance.
pub fn sd_from_var(var: f64) -> f64 {
    if var > 0.0 {
        var.sqrt()
    } else {
        0.0
    }
}

/// The type of the ETA called `name`, looked up **by name** in `eta_param_info`
/// (statement order, one entry per use): all agree → that type; disagreement →
/// `Custom`; none → `None`.
pub fn eta_type(result: &FitResult, name: &str) -> Option<EtaParamType> {
    let mut types =
        result.eta_param_info.iter().filter(|i| i.eta_name == name).map(|i| i.param_type);
    let first = types.next()?;
    Some(if types.all(|t| t == first) { first } else { EtaParamType::Custom })
}

/// Why `[mixture]` override `o` has no SE (#2833), or `None`: a
/// [`FitResult::held_at_bound`] entry of **its** class, family and base index.
pub fn override_held(result: &FitResult, o: &MixtureOverrideParam) -> Option<HeldReason> {
    result
        .held_at_bound
        .iter()
        .find(|h| h.class == Some(o.class) && h.kind == o.kind && h.index == o.index)
        .map(|h| h.reason)
}

/// The SE state of one override row, in render precedence (#2833).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverrideSe {
    /// In `held_at_bound`: `N/A` (console, table), `se: ~` + `held:` (YAML).
    Held(HeldReason),
    /// Fixed by the model or held at its initial value by the estimator: `---`.
    Fixed,
    /// Free: its SE, or `None` when the covariance step produced none.
    Free(Option<f64>),
}

/// One `[mixture]` per-class override as every renderer shows it (#2833).
pub struct OverrideRow<'a> {
    pub param: &'a MixtureOverrideParam,
    /// The fitted value: a class variance (Ω) or σ (Σ), as `mixture_overrides` reports it.
    pub value: f64,
    pub se: OverrideSe,
}

impl OverrideRow<'_> {
    /// The base parameter's name, e.g. `ETA_CL` for `ETA_CL_MIX2`.
    pub fn base_name<'r>(&self, result: &'r FitResult) -> &'r str {
        let names = match self.param.kind {
            ParameterKind::Omega => &result.eta_names,
            _ => &result.sigma_names,
        };
        names.get(self.param.index).map_or("", String::as_str)
    }

    /// `---` / `N/A` / the SE, for the console.
    pub fn se_text(&self) -> String {
        match self.se {
            OverrideSe::Held(_) => "N/A".to_string(),
            OverrideSe::Fixed => "---".to_string(),
            OverrideSe::Free(se) => se_cell(se),
        }
    }
}

/// The override rows of `result` (#2833), in `mixture_overrides` order; empty
/// without one descriptor per value (such a row cannot be named).
pub fn override_rows(result: &FitResult) -> Vec<OverrideRow<'_>> {
    let Some(values) = result.mixture_overrides.as_deref() else {
        return Vec::new();
    };
    if result.mixture_override_params.len() != values.len() {
        return Vec::new();
    }
    result
        .mixture_override_params
        .iter()
        .zip(values)
        .enumerate()
        .map(|(k, (param, &value))| {
            // Held first: a held override is never also fixed (the guard hold is not in
            // the model's FIX mask), so the order matters only for a hand-built result.
            let se = if let Some(reason) = override_held(result, param) {
                OverrideSe::Held(reason)
            } else if param.fixed {
                OverrideSe::Fixed
            } else {
                OverrideSe::Free(
                    result.se_mixture_overrides.as_ref().and_then(|v| v.get(k).copied()),
                )
            };
            OverrideRow { param, value, se }
        })
        .collect()
}

/// A weighted kappa's (#1031) weight source and, with a typical weight `n > 0` and
/// non-negative variance, `(n, var / n)`. `None` when unweighted. Shared by the
/// console weight line and the YAML `weight*` keys (#1660).
pub fn kappa_weight_facts<'a>(
    weights: &'a [Option<String>], weight_typical: &[Option<f64>], i: usize, var: f64,
) -> Option<(&'a str, Option<(f64, f64)>)> {
    let w = weights.get(i).and_then(|w| w.as_deref())?;
    let typical = match weight_typical.get(i).copied().flatten() {
        Some(n) if n > 0.0 && var >= 0.0 => Some((n, var / n)),
        _ => None,
    };
    Some((w, typical))
}

/// Parameter correlation for the off-diagonal `(i, j)`: prefer the precomputed
/// `param_corr` matrix entry when present, else fall back to
/// `cov / (sqrt(var_i) · sqrt(var_j))` (0 when either variance is non-positive).
pub fn param_corr_fallback(
    m: Option<&nalgebra::DMatrix<f64>>, cov: f64, var_i: f64, var_j: f64, i: usize, j: usize,
) -> f64 {
    m.map(|m| m[(i, j)]).unwrap_or_else(|| {
        if var_i > 0.0 && var_j > 0.0 {
            cov / (var_i.sqrt() * var_j.sqrt())
        } else {
            0.0
        }
    })
}

/// Number of coefficients at or above which a θ vector or level block (#1064) is
/// reported in its own compact section. Counts the block's θ in `theta_names`
/// (`FIX` included, #1891), not its levels: a 21-level `sum_to_zero` block (20 θ)
/// is compacted. Use [`compact_theta_blocks`] rather than this constant (#1623).
pub const THETA_BLOCK_COMPACT_MIN: usize = 20;

/// The θ index ranges of the vector / level blocks in `names` with
/// [`THETA_BLOCK_COMPACT_MIN`] or more coefficients. Blocks are runs of
/// `NAME[…]` names (a scalar θ name never contains a bracket); levels are
/// contiguous by construction.
pub fn compact_theta_blocks(names: &[String]) -> Vec<(String, std::ops::Range<usize>)> {
    let block_of = |n: &String| -> Option<String> {
        n.strip_suffix(']').and_then(|r| r.split_once('[')).map(|(b, _)| b.to_string())
    };
    let mut out: Vec<(String, std::ops::Range<usize>)> = Vec::new();
    let mut i = 0;
    while i < names.len() {
        let Some(block) = block_of(&names[i]) else {
            i += 1;
            continue;
        };
        let start = i;
        while i < names.len() && block_of(&names[i]).as_deref() == Some(block.as_str()) {
            i += 1;
        }
        if i - start >= THETA_BLOCK_COMPACT_MIN {
            out.push((block, start..i));
        }
    }
    out
}

/// Quote and escape a free-form string for embedding as a YAML double-quoted
/// scalar. Used for values sourced from the OS/environment (e.g.
/// `environment.username`) that aren't under ferx's control and may contain
/// YAML-significant characters — `:`/`#` are harmless once quoted, but a
/// literal newline or backslash/quote would otherwise corrupt the line or
/// (for a raw newline) get silently folded away by YAML's line-folding rules.
pub fn yaml_quote(s: &str) -> String {
    let escaped =
        s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r");
    format!("\"{escaped}\"")
}

/// The `SE = …` text of a console row: the SE to six decimals, or `N/A` when
/// there is none or it is not finite (#2225) — a `NaN` entry in an otherwise
/// present `se_*` vector prints the same as a missing one, not `NaN`/`inf`.
pub fn se_cell(se: Option<f64>) -> String {
    match se.filter(|s| s.is_finite()) {
        Some(s) => format!("{:.6}", s),
        None => "N/A".to_string(),
    }
}
