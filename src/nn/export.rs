//! Export a fitted deep compartment model (DCM) as a plain population PK model
//! (#1331).
//!
//! After estimation the weights of a `[covariate_nn NAME]` block are constants,
//! and a multilayer perceptron at fixed weights is a closed-form algebraic
//! function of its covariates. [`export_dcm_as_analytic`] writes that function
//! out as ordinary `[individual_parameters]` statements, so the result has no
//! `[covariate_nn]` block, loads in a build without the `nn` feature, and can be
//! read, audited and handed to tools that have no network spelling at all.
//!
//! # A text transform, not a serializer
//!
//! There is no model-file writer to build on, and this does not need one. The
//! transform deletes the `[covariate_nn]` blocks, inserts the generated
//! statements at the top of `[individual_parameters]`, and rewrites each
//! `NAME.OUTPUT` reference there to the generated variable; every other line is
//! copied verbatim, so the export inherits the original's fidelity for free.
//! Block boundaries are found with the parser's own header reader
//! (`block_header`), and each block's fields with its own field reader
//! (`parse_nn_block_fields`) — one parse of the block, shared with the `nn`
//! build that evaluates it.
//!
//! None of this evaluates a network, which is why the module is compiled in
//! every build: the point of the export is to hand a plain model to someone
//! whose build has no DCM support (ferx-r ships without `nn`).
//!
//! # Exactness
//!
//! The emitted statements are the network's forward pass in its own
//! association: `z_i = ((w_i1·a_1 + w_i2·a_2) + …) + b_i`, the order
//! `MlpMapper`'s forward pass accumulates in, then the activation through the
//! same code the network runs (`tanh` and `softplus` are expression builtins
//! wired to [`Activation::apply`]). Weights are printed in Rust's shortest
//! round-trip form, so each literal reads back as the same `f64`. The exported
//! model therefore reproduces the network bit for bit rather than to a
//! tolerance — see the parity tests in `export_tests.rs`.
//!
//! # Missing covariates
//!
//! A covariate the data leaves blank is `NaN` in the subject's covariate map, and
//! the network reads it as `(NaN − center) / scale = NaN`, exactly as the emitted
//! `(X − center) / scale` does. So the normalisation is emitted *without* a
//! `present(X)` guard: a guard would turn a subject the network gives `NaN`
//! predictions into one the export gives finite ones — a different model.
//!
//! A covariate *absent from the map* (a sparse in-memory snapshot, a direct
//! `pk_param_fn` call) is `0.0` to every expression, and the network reads it
//! the same way: `forward_raw` feeds `normalize(0.0) = (0 − center) / scale`,
//! which is what the emitted equation computes from the `0.0` it is handed.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::Activation;
use crate::parser::model_parser::{
    block_disabled_feature, block_header, folds_onto_block_declaration, is_plain_identifier,
    parse_model_string, parse_nn_block_fields, scaling_intermediate_reserved, strip_line_comment,
    NnBlockSpec, CONTINUATION_BLOCKS, CONTINUATION_LEADERS, CONTINUATION_TRAILERS,
};

#[path = "export_values.rs"]
mod values;

/// How the exported model carries the fitted weights.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DcmWeightForm {
    /// Weights inlined as numeric literals. The default: the exported model is
    /// frozen at the fitted network.
    #[default]
    Literal,
    /// Weights declared as `theta W_<NAME>_<l>_<i>_<j>(value, lower, upper) FIX`
    /// in `[parameters]` and referenced by name, so the exported model still
    /// names each weight and one can be perturbed by hand. The θ are fixed:
    /// re-fitting them estimates a wildly over-parameterised model with no
    /// identifiability, which is not what the export is for.
    FixedTheta,
}

/// Options for [`export_dcm_as_analytic`].
///
/// `#[non_exhaustive]`: build one with `DcmExportOptions::default()` and assign
/// the fields.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct DcmExportOptions {
    /// How the weights appear in the exported model.
    pub weight_form: DcmWeightForm,
    /// [`export_dcm_from_fit`] only: keep every `theta` / `omega` / `sigma` /
    /// `kappa` at the value the model text declares instead of writing the
    /// fit's estimates — the fitted network inside the model as written. For a
    /// model with a declaration the estimates cannot be written into (a vector
    /// θ, a `[covariate_model]` θ, a `[mixture]` class override), which
    /// `export_dcm_from_fit` otherwise refuses.
    pub network_only: bool,
}

/// Expand every `[covariate_nn]` block of `model_text` into explicit
/// `[individual_parameters]` statements at the weights in `theta`.
///
/// `theta_names` / `theta` are a fit's `FitResult::theta_names` /
/// `FitResult::theta`. The weights are looked up **by name** — the
/// `W_<NAME>_<l>_<i>_<j>` / `B_<NAME>_<l>_<i>` names the parser generates —
/// rather than by offset, so the lookup does not depend on how many θ the rest
/// of the model declares (a `[covariate_model]` block or a level block adds θ
/// the text alone does not reveal).
///
/// Only the networks are rewritten: every other `theta`, `omega` and `sigma`
/// keeps the value the model text declares, and `theta` is read for the network
/// weights alone.
///
/// Returns the exported model text, or an error naming what stopped the export:
/// a model with no `[covariate_nn]` block, a weight missing from `theta_names`
/// or non-finite, a generated name that collides with one the model already
/// uses, or a block the parser would itself reject.
pub fn export_dcm_as_analytic(
    model_text: &str,
    theta_names: &[String],
    theta: &[f64],
    opts: &DcmExportOptions,
) -> Result<String, String> {
    export_networks(model_text, theta_names, theta, opts, OTHERS_AS_WRITTEN)
}

/// The header line on the other parameters, for each entry point.
const OTHERS_AS_WRITTEN: &str =
    "# Every other theta / omega / sigma keeps the value this file declared.\n";
const OTHERS_FITTED: &str = "# Every other theta / omega / sigma is the fit's estimate.\n";

fn export_networks(
    model_text: &str,
    theta_names: &[String],
    theta: &[f64],
    opts: &DcmExportOptions,
    others: &str,
) -> Result<String, String> {
    if theta_names.len() != theta.len() {
        return Err(format!(
            "theta_names has {} entries but theta has {}; pass a fit's `theta_names` \
             and `theta` together",
            theta_names.len(),
            theta.len()
        ));
    }
    let lines: Vec<&str> = model_text.lines().collect();
    let layout = BlockLayout::scan(&lines);
    if layout.nn_blocks.is_empty() {
        return Err(
            "the model has no [covariate_nn] block; there is no network to export".to_string(),
        );
    }

    // The fields, read by the parser's own reader. Sorted by name — the order the
    // parser registers the blocks' weights in.
    let mut nets = Vec::with_capacity(layout.nn_blocks.len());
    for (name, body) in &layout.nn_blocks {
        let spec = parse_nn_block_fields(name, body)?;
        let weights = lookup_weights(&spec, theta_names, theta)?;
        check_exportable(&spec)?;
        nets.push((spec, weights));
    }

    let names = NameTable::build(&nets, &layout, &lines, opts.weight_form)?;

    let mut out = String::with_capacity(model_text.len() * 2);
    out.push_str(&format!(
        "# Exported by `ferx export-dcm` (#1331): the [covariate_nn] block(s) {} written\n\
         # out as plain equations at the fitted weights. Predictions equal the fitted\n\
         # network's; the weights are {}, not estimated — do not refit them.\n",
        nets.iter()
            .map(|(s, _)| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        match opts.weight_form {
            DcmWeightForm::Literal => "inlined as constants",
            DcmWeightForm::FixedTheta => "FIXed thetas",
        }
    ));
    out.push_str(others);
    // A model may read a weight θ directly (the parser allows it, #1016). Those
    // θ disappear with the block, so under `Literal` each reference becomes the
    // weight's value; under `FixedTheta` the θ is declared again and the
    // reference is left alone.
    let weight_literals: HashMap<String, f64> = match opts.weight_form {
        DcmWeightForm::Literal => nets
            .iter()
            .flat_map(|(spec, w)| spec.weight_theta_names().into_iter().zip(w.iter().copied()))
            .collect(),
        DcmWeightForm::FixedTheta => HashMap::new(),
    };
    for (idx, line) in lines.iter().enumerate() {
        match layout.role[idx] {
            LineRole::NnBlock => continue,
            LineRole::Verbatim => out.push_str(line),
            LineRole::Other | LineRole::IndivParams => {
                out.push_str(&rewrite_refs(line, &nets, &names, &weight_literals)?)
            }
        }
        out.push('\n');
        if Some(idx) == layout.indiv_header {
            for (spec, weights) in &nets {
                emit_network(&mut out, spec, weights, &names, opts.weight_form);
            }
        }
        if Some(idx) == layout.parameters_last && opts.weight_form == DcmWeightForm::FixedTheta {
            for (spec, weights) in &nets {
                for (name, &w) in spec.weight_theta_names().iter().zip(weights) {
                    out.push_str(&format!("  {}\n", fixed_theta_declaration(name, w)));
                }
            }
        }
    }
    check_exported(&out, &nets, &names)?;
    Ok(out)
}

/// Parse the export and confirm every network input is read as a data covariate.
///
/// The emitted text is only worth handing back if it parses, and the parser —
/// not a second list kept here — is what decides how a name resolves: an input
/// that shares its name with a θ, an η, a σ or a variable the model assigns
/// reads that instead of the column, which is a different model with nothing to
/// show for it. `check_exportable` refuses the names that cannot work at all;
/// this catches the ones only the whole model can.
fn check_exported(
    text: &str,
    nets: &[(NnBlockSpec, Vec<f64>)],
    names: &NameTable,
) -> Result<(), String> {
    let model = parse_for_validation(text)?;
    // A `NAME.OUTPUT` read where the generated statements are not in scope (a
    // `[parameters]` kappa weight, say) becomes a bare name there, which parses
    // as a data column the dataset does not have.
    for ((net, out), var) in &names.outputs {
        if model.referenced_covariates.iter().any(|c| c == var) {
            return Err(format!(
                "`{net}.{out}` is read outside [individual_parameters], where the \
                 exported network is not in scope; the export would read `{var}` as a \
                 data column. Move that read into [individual_parameters] before exporting"
            ));
        }
    }
    // A weight θ the model read directly must not have survived as a name: with
    // its declaration gone it would parse as a data column.
    for (spec, _) in nets {
        for w in spec.weight_theta_names() {
            if model.referenced_covariates.iter().any(|c| *c == w) {
                return Err(format!(
                    "[covariate_nn {}] the exported model reads `{w}` as a data column; \
                     the reference to the weight was not rewritten",
                    spec.name
                ));
            }
        }
    }
    // What a bare name resolves to ahead of a data column, in the parser's
    // order: a θ, an η or κ, then a variable assigned earlier in the block. The
    // generated normalisation is the block's first statement, so a variable
    // assigned later is a forward reference the validation parse has already
    // refused; that leaves θ, η and κ. Being read as a column *elsewhere* — a
    // `correlation … by ARM` groups on the column whatever else `ARM` names —
    // does not make the normalisation read it, so the input must name none.
    let p = &model.default_params;
    let mut shadowing: Vec<(&str, &str)> = Vec::new();
    shadowing.extend(p.theta_names.iter().map(|n| (n.as_str(), "a theta")));
    shadowing.extend(
        p.omega
            .eta_names
            .iter()
            .map(|n| (n.as_str(), "a random effect")),
    );
    if let Some(iov) = &p.omega_iov {
        shadowing.extend(iov.eta_names.iter().map(|n| (n.as_str(), "a kappa")));
    }
    for (spec, _) in nets {
        for input in &spec.inputs {
            if let Some((_, what)) = shadowing.iter().find(|(n, _)| n == input) {
                return Err(format!(
                    "[covariate_nn {}] input `{input}` is also the name of {what}, which \
                     the exported equation `{}` would read instead of the data column. \
                     Rename the column or the other `{input}` before exporting",
                    spec.name,
                    input_var(&spec.name, input)
                ));
            }
        }
    }
    Ok(())
}

/// `text` without the blocks the validation parse leaves out.
///
/// `[priors] from_fit = "…"` is expanded during a parse, reading the named fit
/// file against the process's working directory — not the model's, which this
/// text transform does not know. Priors decide nothing the validation looks at
/// (whether the text parses, how each name resolves), so they are left out of it
/// rather than read from a path that may not resolve here.
///
/// A block this build cannot parse (`[event_model]` without `survival`, say) is
/// left out too: the export is a text transform that never evaluates one, and
/// refusing it would make a joint DCM unexportable from the default-feature
/// `ferx` binary. The rest of the model is still validated. The returned export
/// keeps every block verbatim either way.
fn validation_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skip = false;
    for line in text.lines() {
        if let Some((ty, _)) = block_header(strip_line_comment(line).trim()) {
            skip = ty == "priors" || block_disabled_feature(&ty).is_some();
        }
        if !skip {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Parse the export for validation (see [`validation_text`]).
fn parse_for_validation(text: &str) -> Result<crate::types::CompiledModel, String> {
    parse_model_string(&validation_text(text))
        .map_err(|e| format!("the exported model does not parse, so it is not returned: {e}"))
}

/// Export the fitted model: [`export_dcm_as_analytic`] at a fit's weights, with
/// every other `theta` / `omega` / `sigma` / `kappa` declaration set to the fit's
/// estimate, after confirming the fit was made with the networks `model_text`
/// declares.
///
/// Each declaration keeps the scale it is written on (a `(sd)` tag, an inline
/// `prior(...)`), and the export is parsed again to confirm it declares every
/// estimate — bit for bit where the declared scale is the one the fit reports
/// (variances for Ω and κ, SDs for Σ), within a few ULP of the conversion where
/// it is not. A declaration the estimate cannot be written into is an error
/// naming it; [`DcmExportOptions::network_only`] keeps the model's own values
/// instead.
///
/// The weight θ names carry only the block name and the matrix indices, so a
/// model whose network kept its name and shape but changed an input, an
/// activation, the normalisation or an output still finds every weight it looks
/// for in an unrelated fit — and would export those weights under the wrong
/// network, claiming predictions it does not have. So the fit's own model text
/// (`FitResult::model_text`, which every fit made from a file and every `.fitrx`
/// carries) is read with the same block reader, and each `[covariate_nn]` block
/// must define the same network in both: the same blocks by name, and in each
/// the same inputs and outputs in order, hidden widths, both activations and
/// `center` / `scale` bit for bit. Comments, layout and `init` (a starting value
/// the fitted weights supersede) may differ. The rest of the model must match
/// too — every equation, the error model, which parameters exist and which are
/// `FIX` — up to comments, layout, the values in `[parameters]`, `[data] path`
/// and `[priors]` (see `model_lines`).
///
/// This is the entry point to use whenever the model text and the fit arrive
/// separately (`ferx export-dcm model.ferx --fit run.json`). A fit that carries
/// no model text cannot be checked and is refused.
pub fn export_dcm_from_fit(
    model_text: &str,
    fit: &crate::types::FitResult,
    opts: &DcmExportOptions,
) -> Result<String, String> {
    let fitted = fit.model_text.as_deref().ok_or(
        "the fit carries no model text, so the network it was made with cannot be \
         checked against this model; export from the fit's `.fitrx` bundle instead",
    )?;
    check_same_networks(model_text, fitted)?;
    check_same_model(model_text, fitted)?;
    if opts.network_only {
        return export_dcm_as_analytic(model_text, &fit.theta_names, &fit.theta, opts);
    }
    let networks = export_networks(
        model_text,
        &fit.theta_names,
        &fit.theta,
        opts,
        OTHERS_FITTED,
    )?;
    let exported = values::write_fitted_values(&networks, fit);
    let weights: HashSet<String> = network_specs(model_text)?
        .iter()
        .flat_map(NnBlockSpec::weight_theta_names)
        .collect();
    values::check_fitted_values(&parse_for_validation(&exported)?, fit, &weights)?;
    Ok(exported)
}

/// Every `[covariate_nn]` block of `text`, read by the parser's field reader.
fn network_specs(text: &str) -> Result<Vec<NnBlockSpec>, String> {
    let lines: Vec<&str> = text.lines().collect();
    BlockLayout::scan(&lines)
        .nn_blocks
        .iter()
        .map(|(name, body)| parse_nn_block_fields(name, body))
        .collect()
}

/// The definition check behind [`export_dcm_from_fit`].
fn check_same_networks(model_text: &str, fitted_text: &str) -> Result<(), String> {
    let ours = network_specs(model_text)?;
    let theirs = network_specs(fitted_text)
        .map_err(|e| format!("the fit's own model text does not read: {e}"))?;
    let names = |v: &[NnBlockSpec]| v.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
    if names(&ours) != names(&theirs) {
        return Err(format!(
            "the model declares [covariate_nn] block(s) [{}] but the fit was made with \
             [{}]; the fit does not belong to this model",
            names(&ours).join(", "),
            names(&theirs).join(", ")
        ));
    }
    let bits = |v: &Option<Vec<f64>>, n: usize, id: f64| -> Vec<u64> {
        v.clone()
            .unwrap_or_else(|| vec![id; n])
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    for (a, b) in ours.iter().zip(&theirs) {
        let n = a.inputs.len();
        let mut differences = Vec::new();
        let mut differ = |field: &str, same: bool, model: String, fitted: String| {
            if !same {
                differences.push(format!("{field}: model `{model}`, fit `{fitted}`"));
            }
        };
        differ(
            "inputs",
            a.inputs == b.inputs,
            a.inputs.join(", "),
            b.inputs.join(", "),
        );
        differ(
            "outputs",
            a.outputs == b.outputs,
            a.outputs.join(", "),
            b.outputs.join(", "),
        );
        differ(
            "layers",
            a.hidden == b.hidden,
            format!("{:?}", a.hidden),
            format!("{:?}", b.hidden),
        );
        differ(
            "activation",
            a.hidden_activation == b.hidden_activation,
            a.hidden_activation.as_str().into(),
            b.hidden_activation.as_str().into(),
        );
        differ(
            "output",
            a.output_activation == b.output_activation,
            a.output_activation.as_str().into(),
            b.output_activation.as_str().into(),
        );
        differ(
            "center",
            bits(&a.center, n, 0.0) == bits(&b.center, b.inputs.len(), 0.0),
            format!("{:?}", a.center),
            format!("{:?}", b.center),
        );
        differ(
            "scale",
            bits(&a.scale, n, 1.0) == bits(&b.scale, b.inputs.len(), 1.0),
            format!("{:?}", a.scale),
            format!("{:?}", b.scale),
        );
        if !differences.is_empty() {
            return Err(format!(
                "[covariate_nn {}] is not the network the fit was made with ({}); \
                 exporting its weights under this definition would not reproduce the fit",
                a.name,
                differences.join("; ")
            ));
        }
    }
    Ok(())
}

/// The lines of `text` that define the model, each with whitespace removed:
/// `(normalised, original)`.
///
/// Left out, since they cannot change what the model predicts at the fit:
/// comments and blank lines; the values in `[parameters]` (starting values,
/// bounds and inline priors — the export writes the fit's estimates over them;
/// a kappa `weight` and a vector θ's size are compared);
/// a `[covariate_nn]` `init` (a starting value the fitted weights supersede);
/// `[data] path` (which file, not how it is read); and `[priors]`, which
/// shapes only a refit. Everything else — every equation, the error model,
/// which parameters exist and which are `FIX`, `[fit_options]` — must match.
fn model_lines(text: &str) -> Vec<(String, &str)> {
    let mut out = Vec::new();
    // A header counts only once a line of its block does, so a block left empty
    // by the exclusions (a `[data]` holding only a path) is no difference.
    let mut header: Option<(String, &str)> = None;
    for (block, code, line) in logical_lines(text) {
        if block_header(&code).is_some() {
            header = Some((without_whitespace(&code), line));
            continue;
        }
        let mut norm = without_whitespace(&code);
        let key = norm.to_ascii_lowercase();
        match block.as_str() {
            "priors" => continue,
            "covariate_nn" if key.starts_with("init=") => continue,
            "data" if key.starts_with("path=") => continue,
            "parameters" => norm = mask_parameter_values(&norm),
            _ => {}
        }
        out.extend(header.take());
        out.push((norm, line));
    }
    out
}

/// `text` as the parser groups it: `(block, code, first physical line)` per
/// logical line, comments and bare `end` lines dropped. A line continues the one
/// before while a `(` or `[` is open; in `[parameters]` when it is a bare `FIX`
/// or scale tag after a `block_*` declaration; and in an expression block
/// (`CONTINUATION_BLOCKS`) when it starts with an operator or the one before
/// ends with one — the parser's own rules, so a statement laid out over several
/// lines compares equal to the same statement on one.
fn logical_lines(text: &str) -> Vec<(String, String, &str)> {
    let mut out: Vec<(String, String, &str)> = Vec::new();
    let mut block = String::new();
    for line in text.lines() {
        let code = strip_line_comment(line).trim();
        // `end` is a bare terminator the parser drops wherever it appears.
        if code.is_empty() || code == "end" {
            continue;
        }
        if let Some((ty, _)) = block_header(code) {
            block = ty;
            out.push((block.clone(), code.to_string(), line));
            continue;
        }
        let continues = out.last().is_some_and(|(b, prev, _)| {
            *b == block
                && block_header(prev).is_none()
                && (open_groups(prev) > 0
                    || (block == "parameters" && folds_onto_block_declaration(prev, code))
                    || (CONTINUATION_BLOCKS.contains(&block.as_str())
                        && (code.starts_with(CONTINUATION_LEADERS)
                            || prev.ends_with(CONTINUATION_TRAILERS))))
        });
        match out.last_mut() {
            Some((_, prev, _)) if continues => {
                prev.push(' ');
                prev.push_str(code);
            }
            _ => out.push((block.clone(), code.to_string(), line)),
        }
    }
    out
}

/// `(` / `[` left open in `code`, outside quoted spans.
fn open_groups(code: &str) -> i32 {
    let bytes = code.as_bytes();
    let (mut depth, mut i) = (0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'"' | b'\'' => {
                i = quoted_span_end(bytes, i);
                continue;
            }
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    depth
}

/// `code` with its whitespace removed outside quoted spans: a quoted string —
/// a column header such as `"BODY WEIGHT"` — keeps its spaces.
fn without_whitespace(code: &str) -> String {
    let bytes = code.as_bytes();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'"' | b'\'') {
            let end = quoted_span_end(bytes, i);
            out.push_str(&code[i..end]);
            i = end;
            continue;
        }
        let ch = code[i..].chars().next().expect("in bounds");
        if !ch.is_whitespace() {
            out.push(ch);
        }
        i += ch.len_utf8();
    }
    out
}

/// A whitespace-free `[parameters]` line with its *values* masked: the starting
/// value, bounds and inline `prior(...)` arguments, which the export overwrites
/// or which shape only a refit. A `weight = …` expression (it scales κ, so it is
/// part of the model) and a vector θ's `[size]` are kept as written.
fn mask_parameter_values(norm: &str) -> String {
    static NUMBER: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"[A-Za-z_]\w*|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?").unwrap()
    });
    static SIGNED: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"([(,~=\[])[-+]#").unwrap());
    static PRIOR: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)prior\([^)]*\)").unwrap());
    static VECTOR: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)^theta\w+\[[^\]]*\]").unwrap());
    let mask = |s: &str| {
        let masked = NUMBER.replace_all(s, |c: &regex::Captures| {
            let t = &c[0];
            if t.starts_with(|ch: char| ch.is_ascii_digit() || ch == '.') {
                "#".to_string()
            } else {
                t.to_string()
            }
        });
        SIGNED.replace_all(&masked, "$1#").into_owned()
    };
    let line = PRIOR.replace_all(norm, "prior(#)");
    let (head, tail) = match line.to_ascii_lowercase().find("weight=") {
        Some(i) => line.split_at(i),
        None => (line.as_ref(), ""),
    };
    let keep = VECTOR.find(head).map_or(0, |m| m.end());
    format!("{}{}{tail}", &head[..keep], mask(&head[keep..]))
}

/// The whole-model half of the check behind [`export_dcm_from_fit`]: the model
/// text must be the one the fit was made with, up to what [`model_lines`]
/// leaves out. The network check alone would let a changed structural equation
/// or error model through and export it at estimates made for another model.
fn check_same_model(model_text: &str, fitted_text: &str) -> Result<(), String> {
    let ours = model_lines(model_text);
    let theirs = model_lines(fitted_text);
    let differ = ours
        .iter()
        .map(Some)
        .chain(std::iter::repeat(None))
        .zip(theirs.iter().map(Some).chain(std::iter::repeat(None)))
        .take(ours.len().max(theirs.len()))
        .find(|(a, b)| a.map(|x| &x.0) != b.map(|x| &x.0));
    match differ {
        None => Ok(()),
        Some((a, b)) => {
            let show = |x: Option<&(String, &str)>| {
                x.map_or("(nothing)".to_string(), |(_, l)| format!("`{}`", l.trim()))
            };
            Err(format!(
                "the model is not the one the fit was made with: the model has {} where \
                 the fit's model has {}; export from the fit's `.fitrx` bundle, or from \
                 the model file the fit was made with. Only comments, layout, the values \
                 in [parameters], a [covariate_nn] `init`, `[data] path` and [priors] may \
                 differ",
                show(a),
                show(b)
            ))
        }
    }
}

/// What one physical line of the model is, for the rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineRole {
    /// Header or body of a `[covariate_nn]` block — dropped.
    NnBlock,
    /// Header or body of `[individual_parameters]` — `NAME.OUT` rewritten.
    IndivParams,
    /// A block that holds no expressions (`VERBATIM_BLOCKS`) — copied exactly,
    /// never rewritten, and not scanned for name collisions.
    Verbatim,
    /// Every other block — weight-θ reads rewritten, the rest copied as is.
    Other,
}

/// Blocks whose lines are settings and paths, not expressions: nothing in them
/// can read a network output or a weight θ, and a path such as
/// `path = "TYPICAL_PK.CL.csv"` must survive byte for byte.
const VERBATIM_BLOCKS: &[&str] = &["data", "priors", "fit_options"];

/// Where the blocks of the model sit, line by line.
struct BlockLayout {
    role: Vec<LineRole>,
    /// Each `[covariate_nn NAME]` block's body as the parser sees it (comment
    /// stripped, trimmed, blank lines and `end` dropped), keyed by name so
    /// iteration is the parser's name order.
    nn_blocks: BTreeMap<String, Vec<String>>,
    /// The first `[individual_parameters]` header line, where the generated
    /// statements go.
    indiv_header: Option<usize>,
    /// The last non-blank line of the first `[parameters]` block, after which
    /// `FixedTheta` declarations go.
    parameters_last: Option<usize>,
}

impl BlockLayout {
    fn scan(lines: &[&str]) -> Self {
        let mut role = Vec::with_capacity(lines.len());
        let mut nn_blocks: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut indiv_header = None;
        let mut parameters_last = None;
        let mut current = LineRole::Other;
        let mut current_nn: Option<String> = None;
        let mut in_first_parameters = false;
        let mut seen_parameters = false;
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = strip_line_comment(line).trim();
            if let Some((ty, instance)) = block_header(trimmed) {
                in_first_parameters = false;
                current_nn = None;
                current = match (ty.as_str(), instance) {
                    ("covariate_nn", Some(name)) => {
                        nn_blocks.entry(name.clone()).or_default();
                        current_nn = Some(name);
                        LineRole::NnBlock
                    }
                    ("individual_parameters", _) => {
                        indiv_header.get_or_insert(idx);
                        LineRole::IndivParams
                    }
                    ("parameters", _) => {
                        if !seen_parameters {
                            seen_parameters = true;
                            in_first_parameters = true;
                            parameters_last = Some(idx);
                        }
                        LineRole::Other
                    }
                    (ty, _) if VERBATIM_BLOCKS.contains(&ty) => LineRole::Verbatim,
                    _ => LineRole::Other,
                };
                role.push(current);
                continue;
            }
            role.push(current);
            if in_first_parameters && !line.trim().is_empty() {
                parameters_last = Some(idx);
            }
            if let Some(name) = &current_nn {
                if !trimmed.is_empty() && trimmed != "end" {
                    nn_blocks
                        .get_mut(name)
                        .expect("registered at its header")
                        .push(trimmed.to_string());
                }
            }
        }
        BlockLayout {
            role,
            nn_blocks,
            indiv_header,
            parameters_last,
        }
    }
}

/// The block's weights out of the fit, in the network's flat order.
fn lookup_weights(
    spec: &NnBlockSpec,
    theta_names: &[String],
    theta: &[f64],
) -> Result<Vec<f64>, String> {
    spec.weight_theta_names()
        .iter()
        .map(|w| {
            let i = theta_names.iter().position(|n| n == w).ok_or_else(|| {
                format!(
                    "[covariate_nn {}] weight `{w}` is not among the fit's theta names; \
                     the fit does not belong to this model text",
                    spec.name
                )
            })?;
            let v = theta[i];
            if v.is_finite() {
                Ok(v)
            } else {
                Err(format!(
                    "[covariate_nn {}] weight `{w}` is {v}; a non-finite weight has no \
                     equation to export",
                    spec.name
                ))
            }
        })
        .collect()
}

/// What only an exported network has to satisfy, on top of the block parsing.
///
/// The normalisation checks repeat `NamedMlpMapper::with_normalization`'s, which
/// the `nn` build applies to the same values: in a build without `nn` nothing
/// else has looked at them, and the emission indexes them by input.
fn check_exportable(spec: &NnBlockSpec) -> Result<(), String> {
    // Each generates one variable per entry, so a repeat (in any case — the
    // collision scan folds case) would be declared twice.
    for (field, names) in [("inputs", &spec.inputs), ("outputs", &spec.outputs)] {
        let mut seen = HashSet::new();
        if let Some(dup) = names.iter().find(|n| !seen.insert(n.to_ascii_uppercase())) {
            return Err(format!(
                "[covariate_nn {}] `{field}` lists `{dup}` more than once; list each \
                 once before exporting",
                spec.name
            ));
        }
    }
    let n_in = spec.inputs.len();
    for (field, values) in [("center", &spec.center), ("scale", &spec.scale)] {
        if let Some(v) = values {
            if v.len() != n_in {
                return Err(format!(
                    "[covariate_nn {}] `{field}` must have one entry per input: expected \
                     {n_in}, got {}",
                    spec.name,
                    v.len()
                ));
            }
            if let Some((input, x)) = spec.inputs.iter().zip(v).find(|(_, x)| !x.is_finite()) {
                return Err(format!(
                    "[covariate_nn {}] `{field}` for input '{input}' must be finite; got {x}",
                    spec.name
                ));
            }
        }
    }
    if let Some(scale) = &spec.scale {
        // The expression language's `/` returns 0 for a divisor below 1e-30 in
        // magnitude; the network divides. Below that the two are different
        // functions, so refuse rather than emit one that is not the network.
        if let Some((input, s)) = spec.inputs.iter().zip(scale).find(|(_, s)| s.abs() < 1e-30) {
            return Err(format!(
                "[covariate_nn {}] `scale` for input '{input}' is {s}; below 1e-30 in \
                 magnitude the expression language's guarded `/` is not the network's \
                 division, so the export would not reproduce it",
                spec.name
            ));
        }
    }
    for input in &spec.inputs {
        // The network reads each input from the data by name; the export writes
        // the name into an expression, where the parser resolves it as θ, η, a
        // variable or a built-in before it falls back to a data column. Refuse
        // every name that would not come back as the column.
        if !is_plain_identifier(input) {
            return Err(format!(
                "[covariate_nn {}] input `{input}` is not an identifier (letters, digits \
                 and `_`), so it cannot be written into an equation; rename the data \
                 column",
                spec.name
            ));
        }
        if let Some(what) = scaling_intermediate_reserved(input) {
            return Err(format!(
                "[covariate_nn {}] input `{input}`: inside an expression `{input}` is \
                 {what}, while the network reads the data column of that name, so the \
                 export could not promise the same values (#1942); rename the column",
                spec.name
            ));
        }
    }
    Ok(())
}

/// The generated variable names, fixed up front so collisions are found before
/// anything is written.
struct NameTable {
    /// `(network, output)` → the variable a `NAME.OUTPUT` reference becomes.
    outputs: BTreeMap<(String, String), String>,
}

/// Normalised input `j` of network `name`.
fn input_var(name: &str, input: &str) -> String {
    format!("Z_{name}_{input}")
}

/// Post-activation of unit `i` (1-based) of hidden layer `l` (1-based).
fn hidden_var(name: &str, l: usize, i: usize) -> String {
    format!("H{l}_{name}_{i}")
}

/// Pre-activation of unit `i` of layer `l`, emitted only for `relu`, whose
/// `if (z > 0) z else 0` reads `z` twice.
fn preact_var(name: &str, l: usize, i: usize) -> String {
    format!("P{l}_{name}_{i}")
}

/// The variable holding output `out` of network `name`.
fn output_var(name: &str, out: &str) -> String {
    format!("{name}_{out}")
}

impl NameTable {
    fn build(
        nets: &[(NnBlockSpec, Vec<f64>)],
        layout: &BlockLayout,
        lines: &[&str],
        form: DcmWeightForm,
    ) -> Result<Self, String> {
        // Every identifier the retained model text already uses, case-folded:
        // a generated name may not shadow or be shadowed by any of them.
        let mut taken: HashSet<String> = HashSet::new();
        for (idx, line) in lines.iter().enumerate() {
            if matches!(layout.role[idx], LineRole::NnBlock | LineRole::Verbatim) {
                continue;
            }
            for word in identifiers(strip_line_comment(line)) {
                taken.insert(word.to_ascii_uppercase());
            }
        }
        // `NAME.OUT` references are about to be rewritten away, and a direct
        // read of a weight θ is that same θ (inlined under `Literal`, declared
        // again under `FixedTheta`): neither is a collision.
        for (spec, _) in nets {
            taken.remove(&spec.name.to_ascii_uppercase());
            for w in spec.weight_theta_names() {
                taken.remove(&w.to_ascii_uppercase());
            }
        }

        let mut generated: Vec<String> = Vec::new();
        let mut outputs = BTreeMap::new();
        for (spec, _) in nets {
            let sizes = spec.layer_sizes();
            let n_layers = sizes.len() - 1;
            for input in &spec.inputs {
                generated.push(input_var(&spec.name, input));
            }
            for l in 1..n_layers {
                for i in 1..=sizes[l] {
                    generated.push(hidden_var(&spec.name, l, i));
                    if spec.hidden_activation == Activation::Relu {
                        generated.push(preact_var(&spec.name, l, i));
                    }
                }
            }
            for (i, out) in spec.outputs.iter().enumerate() {
                let var = output_var(&spec.name, out);
                generated.push(var.clone());
                if spec.output_activation == Activation::Relu {
                    generated.push(preact_var(&spec.name, n_layers, i + 1));
                }
                outputs.insert((spec.name.clone(), out.clone()), var);
            }
            if form == DcmWeightForm::FixedTheta {
                generated.extend(spec.weight_theta_names());
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        for g in &generated {
            let key = g.to_ascii_uppercase();
            if taken.contains(&key) {
                return Err(format!(
                    "the export would declare `{g}`, a name the model already uses; \
                     rename it in the model before exporting"
                ));
            }
            if !seen.insert(key) {
                return Err(format!(
                    "the export would declare `{g}` twice: two [covariate_nn] blocks \
                     generate the same name; rename one of the blocks before exporting"
                ));
            }
        }
        Ok(NameTable { outputs })
    }
}

/// The end (exclusive) of the quoted span opening at `i` (`"…"` or `'…'`), or
/// the end of `bytes` when it is not closed. A quoted span is a string — a path,
/// a label — never a name, so neither the rewrite nor the collision scan looks
/// inside one.
fn quoted_span_end(bytes: &[u8], i: usize) -> usize {
    let q = bytes[i];
    bytes[i + 1..]
        .iter()
        .position(|&b| b == q)
        .map_or(bytes.len(), |k| i + 1 + k + 1)
}

/// The identifiers in a line of code (letters, digits, `_`, not starting with a
/// digit). A number's exponent (`1e5`) is skipped with the number, and a quoted
/// string is skipped whole.
fn identifiers(code: &str) -> impl Iterator<Item = &str> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'"' || c == b'\'' {
            i = quoted_span_end(bytes, i);
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push(&code[start..i]);
        } else if c.is_ascii_digit() {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'.') {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out.into_iter()
}

/// One retained line with every `NAME.OUTPUT` (`NAME . OUTPUT`) in its code
/// replaced by the generated output variable, and every weight θ named in
/// `weight_literals` by its value. Quoted strings and the comment, if any, are
/// kept verbatim.
fn rewrite_refs(
    line: &str,
    nets: &[(NnBlockSpec, Vec<f64>)],
    names: &NameTable,
    weight_literals: &HashMap<String, f64>,
) -> Result<String, String> {
    let code_len = strip_line_comment(line).len();
    let (code, comment) = line.split_at(code_len);
    let bytes = code.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < bytes.len() {
        if !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            // A number such as `1.5e3` is consumed whole so its `e3` is never
            // read as an identifier, and a quoted string is copied untouched.
            let start = i;
            if bytes[i] == b'"' || bytes[i] == b'\'' {
                i = quoted_span_end(bytes, i);
            } else if bytes[i].is_ascii_digit() {
                while i < bytes.len() && (is_ident(bytes[i]) || bytes[i] == b'.') {
                    i += 1;
                }
            } else {
                i += 1;
            }
            out.push_str(&code[start..i]);
            continue;
        }
        let start = i;
        while i < bytes.len() && is_ident(bytes[i]) {
            i += 1;
        }
        let word = &code[start..i];
        if let Some(&v) = weight_literals.get(word) {
            // Parenthesised when negative, so `x - W` stays `x - (-0.5)`; the
            // tokenizer reads `-0.5` after `(` as one literal, so the value — and
            // the sign of a zero — is the θ's exactly.
            if v.is_sign_negative() {
                out.push_str(&format!("(-{})", magnitude(v)));
            } else {
                out.push_str(&magnitude(v));
            }
            continue;
        }
        let Some((spec, _)) = nets.iter().find(|(s, _)| s.name == word) else {
            out.push_str(word);
            continue;
        };
        // `NAME` followed by `.` and an identifier, whitespace allowed — the
        // tokenizer drops it between the three tokens.
        let mut j = i;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'.' {
            out.push_str(word);
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        let out_start = j;
        while j < bytes.len() && is_ident(bytes[j]) {
            j += 1;
        }
        let output = &code[out_start..j];
        let var = names
            .outputs
            .get(&(spec.name.clone(), output.to_string()))
            .ok_or_else(|| {
                format!(
                    "`{word}.{output}` is not declared as an output of [covariate_nn {word}]. \
                     Known outputs: {}",
                    spec.outputs.join(", ")
                )
            })?;
        out.push_str(var);
        i = j;
    }
    out.push_str(comment);
    Ok(out)
}

/// A weight or bias as the emitted text refers to it: its θ name under
/// `FixedTheta`, its value under `Literal`.
fn coefficient(form: DcmWeightForm, theta_name: &str, value: f64) -> Coefficient<'_> {
    match form {
        DcmWeightForm::Literal => Coefficient::Literal(value),
        DcmWeightForm::FixedTheta => Coefficient::Theta(theta_name),
    }
}

enum Coefficient<'a> {
    Literal(f64),
    Theta(&'a str),
}

/// Shortest round-trip spelling of `|v|` (`{:?}`): reads back as the same `f64`.
/// Never a fixed precision — a truncated weight is a different model, and the
/// truncation would be invisible in the output.
fn magnitude(v: f64) -> String {
    format!("{:?}", v.abs())
}

/// The pre-activation `((c_1·a_1 + c_2·a_2) + …) + b`, one term per line, in the
/// association the network's forward pass accumulates in.
///
/// A negative literal after the first term is written `- |c|·a`: `x − y` is
/// `x + (−y)` exactly in IEEE arithmetic (signed zeros included), and `−(|c|·a)`
/// is `(−|c|)·a` exactly, so the sign moves into the operator without changing a
/// bit. The first term keeps its sign on the literal, which the tokenizer reads
/// as one negative number. The opening `(` is the caller's ([`act_call`]).
fn preactivation(
    terms: &[(Coefficient<'_>, String)],
    bias: Coefficient<'_>,
    indent: &str,
) -> String {
    let mut s = String::new();
    for (k, (c, a)) in terms.iter().enumerate() {
        let lead = k == 0;
        match c {
            Coefficient::Literal(v) => {
                let neg = v.is_sign_negative();
                match (lead, neg) {
                    (true, false) => s.push_str(&format!("{} * {a}", magnitude(*v))),
                    (true, true) => s.push_str(&format!("-{} * {a}", magnitude(*v))),
                    (false, false) => s.push_str(&format!("\n{indent}+ {} * {a}", magnitude(*v))),
                    (false, true) => s.push_str(&format!("\n{indent}- {} * {a}", magnitude(*v))),
                }
            }
            Coefficient::Theta(t) => {
                if lead {
                    s.push_str(&format!("{t} * {a}"));
                } else {
                    s.push_str(&format!("\n{indent}+ {t} * {a}"));
                }
            }
        }
    }
    match bias {
        Coefficient::Literal(v) if v.is_sign_negative() => {
            s.push_str(&format!("\n{indent}- {}", magnitude(v)))
        }
        Coefficient::Literal(v) => s.push_str(&format!("\n{indent}+ {}", magnitude(v))),
        Coefficient::Theta(t) => s.push_str(&format!("\n{indent}+ {t}")),
    }
    s.push(')');
    s
}

/// The function an activation is written as, up to and including the `(` that
/// opens the pre-activation (`relu` and `identity` are a bare parenthesis).
fn act_call(act: Activation) -> &'static str {
    match act {
        Activation::Identity | Activation::Relu => "(",
        Activation::Softplus => "softplus(",
        Activation::Tanh => "tanh(",
        // The DSL's `inv_logit` is `Activation::Sigmoid` itself, so this is bit-exact.
        Activation::Sigmoid => "inv_logit(",
        Activation::Exp => "exp(",
    }
}

/// The statement(s) assigning `var = activation(pre)`. `relu` reads its argument
/// twice, so it gets the pre-activation as its own variable `pre_var`.
fn activate(out: &mut String, act: Activation, var: &str, pre: &str, pre_var: &str) {
    match act {
        // `Activation::Relu` is `if z > 0 { z } else { 0.0 }`; `max(z, 0)` is
        // `z >= 0`, which differs at `z = -0.0`. Spell the network's own branch.
        Activation::Relu => {
            out.push_str(&format!("{BODY}{pre_var} = ({pre}\n"));
            out.push_str(&format!(
                "{BODY}{var} = if ({pre_var} > 0) {pre_var} else 0\n"
            ));
        }
        _ => out.push_str(&format!("{BODY}{var} = {}{pre}\n", act_call(act))),
    }
}

/// `theta NAME(value, lower, upper) FIX`, with a box that contains `value` (the
/// parser's default box is positive, and most weights are not).
fn fixed_theta_declaration(name: &str, value: f64) -> String {
    let lo = if value > -1e9 { -1e9 } else { 2.0 * value };
    let hi = if value < 1e9 { 1e9 } else { 2.0 * value };
    format!("theta {name}({value:?}, {lo:?}, {hi:?}) FIX")
}

/// Indent of the statements inside a network's `if (0 == 0) { … }` block.
const BODY: &str = "    ";

/// Write one network's forward pass as statements.
///
/// The statements sit inside an always-taken `if (0 == 0) { … } else { … }`.
/// That is not decoration: every *top-level* assignment in
/// `[individual_parameters]` is an individual parameter, and an ODE model gives
/// each one a slot in the fixed `PkParams` layout (fewer than
/// `MAX_PK_PARAMS` = 128). A network of two 64-unit hidden layers has 130 units,
/// so emitted at top level its export would not parse. A name assigned inside an
/// `if` stays branch-local unless a downstream block reads it, so the units take
/// no slot. Only the outputs are read after the block, which is why the `else`
/// arm assigns exactly those — a name read after an `if` must be assigned in
/// every arm. The arm is never taken; it exists to satisfy that rule.
fn emit_network(
    out: &mut String,
    spec: &NnBlockSpec,
    weights: &[f64],
    names: &NameTable,
    form: DcmWeightForm,
) {
    let name = spec.name.as_str();
    let sizes = spec.layer_sizes();
    let n_layers = sizes.len() - 1;
    let shape = sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" -> ");
    out.push_str(&format!(
        "  # --- [covariate_nn {name}] expanded: {shape}, {} hidden / {} output ---\n\
         \x20 # Always taken: the `if` keeps the network's units out of the individual\n\
         \x20 # parameters, so an ODE model does not give each one a PK slot.\n\
         \x20 if (0 == 0) {{\n",
        spec.hidden_activation.as_str(),
        spec.output_activation.as_str()
    ));

    // Inputs: `(X - center) / scale`, exactly the network's `normalize`. Only the
    // identity transform (`center = +0`, `scale = 1`) is written as the bare
    // covariate, which it equals bit for bit; `x - (-0.0)` would not (−0 → +0).
    for (j, input) in spec.inputs.iter().enumerate() {
        let c = spec.center.as_ref().map_or(0.0, |v| v[j]);
        let s = spec.scale.as_ref().map_or(1.0, |v| v[j]);
        let var = input_var(name, input);
        if c.to_bits() == 0.0f64.to_bits() && s == 1.0 {
            out.push_str(&format!("{BODY}{var} = {input}\n"));
        } else {
            let shift = if c.is_sign_negative() {
                format!("{input} + {}", magnitude(c))
            } else {
                format!("{input} - {}", magnitude(c))
            };
            let div = if s.is_sign_negative() {
                format!("(-{})", magnitude(s))
            } else {
                magnitude(s)
            };
            out.push_str(&format!("{BODY}{var} = ({shift}) / {div}\n"));
        }
    }

    let theta_names = spec.weight_theta_names();
    let mut offset = 0;
    let mut prev: Vec<String> = spec.inputs.iter().map(|i| input_var(name, i)).collect();
    for l in 1..=n_layers {
        let (n_l, n_lm1) = (sizes[l], sizes[l - 1]);
        let w_at = offset;
        let b_at = offset + n_l * n_lm1;
        let is_output = l == n_layers;
        let act = if is_output {
            spec.output_activation
        } else {
            spec.hidden_activation
        };
        let mut next = Vec::with_capacity(n_l);
        for i in 0..n_l {
            let var = if is_output {
                names.outputs[&(spec.name.clone(), spec.outputs[i].clone())].clone()
            } else {
                hidden_var(name, l, i + 1)
            };
            // Row-major `W` (`n_l × n_{l-1}`): unit `i`'s row is `i·n_{l-1} ..`.
            let terms: Vec<(Coefficient<'_>, String)> = (0..n_lm1)
                .map(|j| {
                    let k = w_at + i * n_lm1 + j;
                    (
                        coefficient(form, &theta_names[k], weights[k]),
                        prev[j].clone(),
                    )
                })
                .collect();
            let k = b_at + i;
            let bias = coefficient(form, &theta_names[k], weights[k]);
            // Continuation lines line up under the first term: `  VAR = call(`.
            let call = if act == Activation::Relu {
                act_call(Activation::Identity)
            } else {
                act_call(act)
            };
            let lead = if act == Activation::Relu {
                preact_var(name, l, i + 1)
            } else {
                var.clone()
            };
            let indent = " ".repeat(BODY.len() + lead.len() + 3 + call.len());
            let pre = preactivation(&terms, bias, &indent);
            activate(out, act, &var, &pre, &preact_var(name, l, i + 1));
            next.push(var);
        }
        offset = b_at + n_l;
        prev = next;
    }
    debug_assert_eq!(offset, weights.len());
    out.push_str("  } else {\n");
    for out_var in &prev {
        out.push_str(&format!("{BODY}{out_var} = 0\n"));
    }
    out.push_str("  }\n");
}

#[cfg(test)]
#[path = "export_tests.rs"]
mod tests;
