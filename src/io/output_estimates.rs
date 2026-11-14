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

use super::*;

/// One value of an estimate entry, tagged with how the YAML prints it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
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
    pub(crate) fn yaml(&self, quote_names: bool) -> String {
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

    pub(crate) fn json(&self) -> serde_json::Value {
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
/// name) as a YAML key or scalar: always quoted when `always` (under `estimates:`),
/// otherwise only when YAML 1.1 — which R's `yaml` package implements — would read it
/// bare as something other than that string. Every legal identifier is a plain scalar
/// except the boolean and null spellings: `YES`, `no`, `On`, `OFF`, `y`, `N`, `TRUE`,
/// `NULL`, … would come back as `TRUE` / `FALSE` / `NULL` and address a different key than
/// the JSON's `"YES"` (#2911). The check is case-insensitive, so it also quotes a few
/// mixed-case spellings YAML would read as strings; that costs only the quotes.
pub(crate) fn yaml_name(name: &str, always: bool) -> String {
    const IMPLICIT: &[&str] = &[
        "y", "n", "yes", "no", "true", "false", "on", "off", "null", "~", "",
    ];
    if always || IMPLICIT.contains(&name.to_ascii_lowercase().as_str()) {
        yaml_quote(name)
    } else {
        name.to_string()
    }
}

/// One named entry of a section: a parameter (`CL`), or an off-diagonal (`A__B`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Entry {
    pub(crate) name: String,
    /// In the order the YAML writes them.
    pub(crate) fields: Vec<(&'static str, Value)>,
}

impl Entry {
    fn new(name: impl Into<String>) -> Self {
        Entry {
            name: name.into(),
            fields: Vec::new(),
        }
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
pub(crate) struct Section {
    pub(crate) key: &'static str,
    /// Written after the header in the YAML (`sigma:  # error model: …`); not in the JSON.
    pub(crate) comment: Option<String>,
    pub(crate) entries: Vec<Entry>,
}

impl Section {
    fn new(key: &'static str) -> Self {
        Section {
            key,
            comment: None,
            entries: Vec::new(),
        }
    }

    /// The section as a top-level YAML key, from the blank line before its header to its
    /// last field.
    pub(crate) fn yaml(&self) -> String {
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
    pub(crate) fn json(&self) -> serde_json::Value {
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
pub(crate) fn listed_theta_indices(result: &FitResult) -> Vec<usize> {
    #[cfg(feature = "nn")]
    let nn: std::collections::HashSet<usize> = result
        .neural_networks
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.n_weights)
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let blocked: std::collections::HashSet<usize> = compact_theta_blocks(&result.theta_names)
        .into_iter()
        .flat_map(|(_, r)| r)
        .collect();
    (0..result.theta_names.len())
        .filter(|i| !nn.contains(i) && !blocked.contains(i))
        .collect()
}

/// The spread field of a random-effect variance on its η type's scale: `cv_pct` for
/// log-normal and unknown, `sd` for additive and logit, none for a custom expression.
pub(crate) fn spread_field(t: Option<EtaParamType>, var: f64) -> Option<(&'static str, Value)> {
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
pub(crate) fn typical_cv_field(
    t: Option<EtaParamType>,
    var_at: f64,
) -> Option<(&'static str, Value)> {
    matches!(t, None | Some(EtaParamType::LogNormal))
        .then(|| ("cv_pct_at_typical_weight", Value::Cv(cv_pct(var_at))))
}

/// The `theta:` section: `estimate`, then `fixed`/`at_bound`, `se`, `rse_pct`, `held`.
pub(crate) fn theta_section(result: &FitResult) -> Section {
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
pub(crate) fn omega_section(result: &FitResult) -> Section {
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
    let eta = |i: usize| {
        result
            .eta_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("eta_{}", i + 1))
    };
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
pub(crate) fn sigma_section(result: &FitResult) -> Section {
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
        let name = result
            .sigma_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("sigma_{}", i + 1));
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
pub(crate) fn block_sigma_section(result: &FitResult) -> Result<Option<Section>, String> {
    if result.residual_correlations.is_empty() {
        return Ok(None);
    }
    let mut section = Section::new("block_sigma");
    for (k, corr) in result.residual_correlations.iter().enumerate() {
        let name_i = result.sigma_names.get(corr.sigma_i).ok_or_else(|| {
            format!(
                "residual correlation sigma_i index {} is out of bounds",
                corr.sigma_i
            )
        })?;
        let name_j = result.sigma_names.get(corr.sigma_j).ok_or_else(|| {
            format!(
                "residual correlation sigma_j index {} is out of bounds",
                corr.sigma_j
            )
        })?;
        let sigma_i = result.sigma.get(corr.sigma_i).ok_or_else(|| {
            format!(
                "residual correlation sigma_i index {} has no estimate",
                corr.sigma_i
            )
        })?;
        let sigma_j = result.sigma.get(corr.sigma_j).ok_or_else(|| {
            format!(
                "residual correlation sigma_j index {} has no estimate",
                corr.sigma_j
            )
        })?;
        // Since #847 a bare `block_sigma (...) = [...]` **estimates** the off-diagonal
        // (NONMEM `$SIGMA BLOCK(n)`); only `... FIX` pins it. A pre-#847 `FitResult`
        // carries no flags at all, and back then every correlation was fixed by
        // construction — hence the `true` default.
        let correlation_fixed = result
            .residual_correlation_fixed
            .get(k)
            .copied()
            .unwrap_or(true);
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
            result
                .se_residual_correlations
                .as_ref()
                .and_then(|v| v.get(k))
                .copied()
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
pub(crate) fn omega_iov_section(result: &FitResult) -> Option<Section> {
    let iov = result.omega_iov.as_ref()?;
    let mut section = Section::new("omega_iov");
    let n_kappa = iov.nrows();
    let kappa = |i: usize| {
        result
            .kappa_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("kappa_{}", i + 1))
    };
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
pub(crate) fn mixture_overrides_section(result: &FitResult) -> Option<Section> {
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
pub(crate) fn estimate_sections(result: &FitResult) -> Result<Vec<Section>, String> {
    let mut out = vec![
        theta_section(result),
        omega_section(result),
        sigma_section(result),
    ];
    out.extend(block_sigma_section(result)?);
    out.extend(omega_iov_section(result));
    out.extend(mixture_overrides_section(result));
    Ok(out)
}

/// The `estimates:` block of the fit YAML (#2911): every section of [`estimate_sections`]
/// again, nested one level under `estimates`, so `estimates.theta.CL.estimate` reads the
/// same in `{model}-fit.yaml` and `{model}-fit.json`. The top-level `theta:` / `omega:` /
/// … sections stay where they were, for the readers that already use them.
pub(crate) fn estimates_yaml(result: &FitResult) -> Result<String, String> {
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
pub(crate) fn estimates_json(result: &FitResult) -> serde_json::Value {
    match estimate_sections(result) {
        Ok(sections) => serde_json::Value::Object(
            sections
                .iter()
                .map(|s| (s.key.to_string(), s.json()))
                .collect(),
        ),
        Err(_) => serde_json::Value::Null,
    }
}

#[cfg(test)]
#[path = "output_estimates_tests.rs"]
mod tests;
