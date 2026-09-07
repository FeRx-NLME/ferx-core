//! [`VariabilityText::read`]: a model's η / κ structure as written (#1183).
//!
//! The reading side of the random-effect edits. `apply.rs` owns the line
//! grammars and the canonical-form splitter; this module only walks the two
//! blocks with them and reports what it finds, so a search over variability
//! structures can start from *what the model says* rather than from what a
//! `FitResult` implies — a `FIX`ed ω, a `block_omega`, a parameter whose η
//! is written in a form the edits cannot rewrite, are all facts about the
//! text.

use super::apply::{
    block_names, declared_variance, is_top_level_product, mentions, split_trailing_exp,
    RandomEffectKind, BLOCK_KAPPA_RE, BLOCK_OMEGA_RE, KAPPA_RE, OMEGA_RE,
};
use super::spec::{ParameterVariability, RandomEffectBlock, RandomEffectDecl, VariabilityText};
use super::ModelText;
use std::sync::LazyLock;

use regex::Regex;

/// `NAME = <expr>` — the same assignment grammar `apply.rs` uses.
static ASSIGN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Za-z_]\w*)\s*=\s*([^=].*)$").unwrap());
static FIX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bFIX\b").unwrap());

pub(crate) fn read(text: &ModelText) -> Result<VariabilityText, String> {
    let mut omegas = Vec::new();
    let mut omega_blocks = Vec::new();
    let mut kappas = Vec::new();
    let mut kappa_blocks = Vec::new();
    for (_, code) in text.logical_lines("parameters") {
        if let Some(caps) = OMEGA_RE.captures(&code) {
            omegas.push(RandomEffectDecl {
                name: caps[2].to_string(),
                variance: declared_variance(&code, RandomEffectKind::Eta)?,
                fixed: FIX_RE.is_match(&code),
            });
        } else if let Some(caps) = KAPPA_RE.captures(&code) {
            kappas.push(RandomEffectDecl {
                name: caps[2].to_string(),
                variance: declared_variance(&code, RandomEffectKind::Kappa)?,
                fixed: FIX_RE.is_match(&code),
            });
        } else if let Some(caps) = BLOCK_OMEGA_RE.captures(&code) {
            omega_blocks.push(RandomEffectBlock {
                names: block_names(&caps[2]),
                fixed: FIX_RE.is_match(&caps[4]),
            });
        } else if let Some(caps) = BLOCK_KAPPA_RE.captures(&code) {
            kappa_blocks.push(RandomEffectBlock {
                names: block_names(&caps[2]),
                fixed: FIX_RE.is_match(&caps[4]),
            });
        }
    }
    let etas: Vec<String> = omegas
        .iter()
        .map(|d| d.name.clone())
        .chain(omega_blocks.iter().flat_map(|b| b.names.iter().cloned()))
        .collect();
    let kappa_names: Vec<String> = kappas
        .iter()
        .map(|d| d.name.clone())
        .chain(kappa_blocks.iter().flat_map(|b| b.names.iter().cloned()))
        .collect();
    let all: Vec<String> = etas.iter().chain(kappa_names.iter()).cloned().collect();

    let mut parameters = Vec::new();
    for (_, code) in text.logical_lines("individual_parameters") {
        let Some(caps) = ASSIGN_RE.captures(&code) else {
            continue;
        };
        let name = caps[1].to_string();
        let rhs = caps[2].trim().to_string();
        let mut mentioned: Vec<String> = Vec::new();
        for m in super::IDENT_RE.find_iter(&rhs) {
            let s = m.as_str();
            if all.iter().any(|a| a == s) && !mentioned.iter().any(|a| a == s) {
                mentioned.push(s.to_string());
            }
        }
        // Canonical: `HEAD * exp(terms)` with every random effect the line
        // mentions inside that one factor, at most one η and one κ — or a
        // line that mentions no random effect and is a product an η could be
        // appended to.
        let (eta, kappa, canonical) = match split_trailing_exp(&rhs, &all) {
            Some((head, terms)) => {
                let outside = all.iter().any(|a| mentions(&head, a));
                let eta_terms: Vec<&String> = terms.iter().filter(|t| etas.contains(t)).collect();
                let kappa_terms: Vec<&String> =
                    terms.iter().filter(|t| kappa_names.contains(t)).collect();
                let canonical = !outside && eta_terms.len() <= 1 && kappa_terms.len() <= 1;
                (
                    eta_terms.first().map(|s| (*s).clone()),
                    kappa_terms.first().map(|s| (*s).clone()),
                    canonical,
                )
            }
            None => {
                let eta: Vec<&String> = mentioned.iter().filter(|m| etas.contains(m)).collect();
                let kappa: Vec<&String> = mentioned
                    .iter()
                    .filter(|m| kappa_names.contains(m))
                    .collect();
                (
                    (eta.len() == 1).then(|| eta[0].clone()),
                    (kappa.len() == 1).then(|| kappa[0].clone()),
                    mentioned.is_empty() && is_top_level_product(&rhs),
                )
            }
        };
        parameters.push(ParameterVariability {
            name,
            eta,
            kappa,
            canonical,
            mentions: mentioned,
        });
    }
    Ok(VariabilityText {
        parameters,
        omegas,
        omega_blocks,
        kappas,
        kappa_blocks,
    })
}
