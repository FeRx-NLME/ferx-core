//! Pandoc's `auto_identifiers` algorithm — the ids Quarto actually renders.
//!
//! Getting this exactly right is the whole game: a slug model that is even
//! slightly off turns every anchor check (R4) into noise. The first prototype
//! for #1163 assumed an em-dash produced a *double* hyphen and reported four
//! healthy links as broken, and four broken ones as healthy.
//!
//! The algorithm, verified against `docs/_site/**/*.html` (`data-anchor-id`):
//!
//! 1. An explicit `{#id}` attribute on the heading wins outright.
//! 2. Otherwise: strip inline markup (code spans, emphasis, link syntax), then
//!    **delete** — not replace — every character that is not alphanumeric,
//!    `_`, `-` or `.`; map runs of whitespace to a single `-`; lowercase; drop
//!    everything before the first letter.
//! 3. A repeat of an id already used on the page gets `-1`, `-2`, … in
//!    document order.
//!
//! Two consequences worth spelling out, because both bit us:
//!
//! - `### 3. Communication` → `communication`. The number is dropped by (2)'s
//!   last step, so `#3-communication` is a dead link.
//! - `### Branching model — trunk-based` → `branching-model-trunk-based`, a
//!   *single* hyphen: the em-dash is deleted and the spaces that surrounded it
//!   collapse into one separator. Same for `(`ode_method = auto`)` →
//!   `ode_method-auto`. Deleting rather than replacing is also why
//!   `` `D{n}` `` → `dn` and not `d-n`.

use std::collections::HashMap;

/// Strip a trailing pandoc attribute block (`{#id .class key=val}`) from a
/// heading, returning `(text_without_attrs, explicit_id)`.
pub fn split_attributes(text: &str) -> (String, Option<String>) {
    let trimmed = text.trim_end();
    if !trimmed.ends_with('}') {
        return (trimmed.to_string(), None);
    }
    let Some(open) = trimmed.rfind('{') else {
        return (trimmed.to_string(), None);
    };
    let attrs = &trimmed[open + 1..trimmed.len() - 1];
    let id = attrs
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix('#'))
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    (trimmed[..open].trim_end().to_string(), id)
}

/// Remove the inline markup pandoc resolves before it builds an identifier:
/// code spans, emphasis markers, and link syntax (keeping the link *text*).
fn strip_inline_markup(text: &str) -> String {
    // Link syntax: `[text](target)` and `[text][ref]` keep only `text`.
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '[' => {
                // Copy the link text, then skip a following (…) or […] target.
                let mut depth = 1;
                i += 1;
                while i < chars.len() && depth > 0 {
                    match chars[i] {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        c => out.push(c),
                    }
                    i += 1;
                }
                i += 1; // consume the closing ']'
                if i < chars.len() && (chars[i] == '(' || chars[i] == '[') {
                    let close = if chars[i] == '(' { ')' } else { ']' };
                    while i < chars.len() && chars[i] != close {
                        i += 1;
                    }
                    i += 1; // consume the closing delimiter
                }
            }
            '`' | '*' => i += 1,
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// The identifier pandoc generates for a heading, ignoring collisions.
///
/// Pass the raw heading text (without the leading `#`s); an explicit `{#id}`
/// is honoured.
pub fn slug(heading_text: &str) -> String {
    let (text, explicit) = split_attributes(heading_text);
    if let Some(id) = explicit {
        return id;
    }
    let stripped = strip_inline_markup(&text);

    // Delete disallowed characters (do not replace them), keep whitespace as a
    // separator, then collapse whitespace runs into single hyphens.
    let mut kept = String::with_capacity(stripped.len());
    for ch in stripped.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            kept.extend(ch.to_lowercase());
        } else if ch.is_whitespace() {
            kept.push(' ');
        }
    }
    let joined = kept.split_whitespace().collect::<Vec<_>>().join("-");

    // Drop everything up to the first letter.
    match joined.char_indices().find(|(_, c)| c.is_alphabetic()) {
        Some((start, _)) => joined[start..].to_string(),
        None => String::new(),
    }
}

/// Assigns page-unique ids in document order, mirroring pandoc's `-1`, `-2`, …
/// disambiguation.
#[derive(Default)]
pub struct IdAssigner {
    seen: HashMap<String, usize>,
}

impl IdAssigner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `(assigned_id, base_id)`. They differ exactly when the base id
    /// had already been used on this page — which is what R3 reports.
    pub fn assign(&mut self, heading_text: &str) -> (String, String) {
        let base = slug(heading_text);
        let n = self.seen.entry(base.clone()).or_insert(0);
        let assigned = if *n == 0 {
            base.clone()
        } else {
            format!("{base}-{n}")
        };
        *n += 1;
        (assigned, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expectation below is a `data-anchor-id` copied out of a Quarto
    /// render of this repo's docs, not a guess about what pandoc does.
    #[test]
    fn matches_rendered_quarto_ids() {
        let cases = [
            // Numbered heading: the number is dropped entirely.
            ("3. Communication", "communication"),
            ("8.7 The public-API gate", "the-public-api-gate"),
            // Em-dash: deleted, surrounding spaces collapse to ONE hyphen.
            (
                "5.1 Branching model — trunk-based",
                "branching-model-trunk-based",
            ),
            (
                "No omega at all — fixed-effects-only (naive-pooled) fits",
                "no-omega-at-all-fixed-effects-only-naive-pooled-fits",
            ),
            // Code span + `=`: deleted, spaces collapse.
            (
                "Letting ferx pick the stepper (`ode_method = auto`)",
                "letting-ferx-pick-the-stepper-ode_method-auto",
            ),
            // Braces are deleted without inserting a separator.
            (
                "Modeled infusion duration (`D{n}`, RATE = −2)",
                "modeled-infusion-duration-dn-rate-2",
            ),
            (
                "Modeled infusion rate (`R{n}`, RATE = −1)",
                "modeled-infusion-rate-rn-rate-1",
            ),
            (
                "Analytic FOCE / FOCEI gradient (analytical PK models)",
                "analytic-foce-focei-gradient-analytical-pk-models",
            ),
            (
                "Do I need to use mu-referencing in my model definitions, like in NONMEM / nlmixr2?",
                "do-i-need-to-use-mu-referencing-in-my-model-definitions-like-in-nonmem-nlmixr2",
            ),
            ("CI/CD and quality gates", "cicd-and-quality-gates"),
            ("Zero-order (`zero_order`)", "zero-order-zero_order"),
        ];
        for (heading, want) in cases {
            assert_eq!(slug(heading), want, "heading: {heading}");
        }
    }

    #[test]
    fn explicit_id_wins() {
        assert_eq!(slug("Warning codes {#warning-codes}"), "warning-codes");
        assert_eq!(slug("Anything at all {#custom .unnumbered}"), "custom");
    }

    #[test]
    fn attribute_block_without_id_is_stripped() {
        assert_eq!(slug("Appendix {.unnumbered}"), "appendix");
        let (text, id) = split_attributes("Appendix {.unnumbered}");
        assert_eq!(text, "Appendix");
        assert_eq!(id, None);
    }

    #[test]
    fn link_text_survives_the_target() {
        assert_eq!(slug("See [the FAQ](../faq.qmd) first"), "see-the-faq-first");
    }

    #[test]
    fn emphasis_is_transparent() {
        assert_eq!(slug("**Bold** and *italic*"), "bold-and-italic");
    }

    #[test]
    fn heading_with_no_letters_yields_an_empty_id() {
        assert_eq!(slug("123 456"), "");
    }

    #[test]
    fn collisions_get_pandoc_suffixes() {
        let mut ids = IdAssigner::new();
        assert_eq!(ids.assign("Syntax"), ("syntax".into(), "syntax".into()));
        assert_eq!(ids.assign("Syntax"), ("syntax-1".into(), "syntax".into()));
        assert_eq!(ids.assign("Syntax"), ("syntax-2".into(), "syntax".into()));
        assert_eq!(ids.assign("Other"), ("other".into(), "other".into()));
    }
}
