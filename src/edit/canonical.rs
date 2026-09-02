//! Candidate identity — the canonical form behind
//! [`ModelText::canonical_hash`](super::ModelText::canonical_hash).
//!
//! A search fits hundreds of candidates and must not fit the same one twice.
//! Hashing the raw bytes would miss on a re-indent or an added comment, and
//! comparing `CompiledModel`s is impossible (they are closures). So identity
//! is a *normalised* view of the source:
//!
//! - comments are removed and blank lines dropped, so annotating a model does
//!   not make it a new model;
//! - each line's whitespace collapses to nothing except between two
//!   word characters, where a single space survives — `TVCL*exp(ETA_CL)` and
//!   `TVCL * exp(ETA_CL)` are one model, and `theta TVCL` does not become
//!   `thetaTVCL`;
//! - `[block]` headers lowercase, matching what the parser does to them.
//!
//! Deliberately **not** normalised: line order, and the spelling of numbers.
//! Reordering `[parameters]` reorders the θ vector, so it is a different model
//! and must hash differently. `0.15` versus `1.5e-1` is the same model and
//! will hash differently — a false miss, which costs a fit; the alternative is
//! parsing every literal, which risks a false *hit*, which costs correctness.

use sha2::{Digest, Sha256};

use super::{code_of, BLOCK_HEADER_RE};

/// The normalised text `canonical_hash` digests, one line per surviving
/// source line.
pub(crate) fn canonical_form(lines: &[String]) -> String {
    let mut out = String::new();
    for line in lines {
        let code = code_of(line);
        if code.is_empty() {
            continue;
        }
        let normalized = squeeze(code);
        if BLOCK_HEADER_RE.is_match(&normalized) {
            out.push_str(&normalized.to_lowercase());
        } else {
            out.push_str(&normalized);
        }
        out.push('\n');
    }
    out
}

pub(crate) fn canonical_hash(lines: &[String]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(canonical_form(lines).as_bytes());
    hasher.finalize().into()
}

/// Drop every run of whitespace, except between two word characters where one
/// space is kept — the boundary that separates `theta TVCL` (two tokens) from
/// `TVCL * exp(...)` (one expression written loosely).
fn squeeze(code: &str) -> String {
    let mut out = String::with_capacity(code.len());
    let mut pending_space = false;
    for c in code.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space && is_word(out.chars().next_back().unwrap_or(' ')) && is_word(c) {
            out.push(' ');
        }
        pending_space = false;
        out.push(c);
    }
    out
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}
