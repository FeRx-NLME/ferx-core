//! Typed model transformation — `ModelText` and `ModelEdit` (#1176).
//!
//! Text → [`CompiledModel`](crate::types::CompiledModel) has always been a
//! one-way street. ferx can *fit* a model and *rank* it, but every model-space
//! search tool is `generate candidate → fit → rank`, and there was nothing in
//! the engine that could **generate** one. This module is that missing third
//! verb.
//!
//! # Why text surgery and not an AST
//!
//! A `CompiledModel` is closures. It cannot be rendered back to `.ferx`, and a
//! second, renderable AST would be a parallel definition of the language that
//! drifts from the one that actually compiles. So a candidate model here is
//! *the source file*, edited in place:
//!
//! ```text
//! ModelText::parse(src)  →  apply(edit)…  →  render()  →  parse_full_model
//! ```
//!
//! [`ModelText`] keeps every byte of the original — comments, blank lines,
//! alignment, CRLF, a missing final newline — and an unedited round-trip is
//! **byte-identical** (pinned over all of `examples/**` in
//! `tests/model_text_roundtrip.rs`). That property is what makes the surgery
//! safe: anything this module does not deliberately touch is provably
//! unchanged, so a bad edit shows up as a visible diff rather than a silent
//! rewrite of the rest of the file.
//!
//! # Canonical form is a precondition, not a guess
//!
//! η surgery ([`ModelEdit::AddIiv`] / [`ModelEdit::DropIiv`]) is expression
//! editing, not line editing. Rather than pattern-match an arbitrary
//! right-hand side, a search-eligible parameter must be written in the
//! canonical product form
//!
//! ```text
//! CL = TVCL * exp(ETA_CL)
//! ```
//!
//! and anything else is a hard error naming the offending parameter. This is
//! the posture `[covariate_model]` already takes on a non-product right-hand
//! side (#1111), so it is a rule users have met before.
//!
//! # Identity
//!
//! [`ModelText::canonical_hash`] is the candidate's identity — a cache key for
//! "have I already fitted this model?". It is stable under comments and
//! whitespace and changes for every token-level edit.

mod apply;
mod canonical;
mod spec;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

pub use spec::{
    ErrorForm, ErrorSpecText, IivForm, ModelEdit, NewParameter, Relation, RelationTheta, SigmaDecl,
    StructuralSpec,
};

use std::collections::HashSet;
use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

/// The `[block]` / `[block INSTANCE]` header form, matched on the
/// comment-stripped, trimmed line — the same anchor
/// `parser::model_parser::extract_blocks` uses, so this module and the parser
/// can never disagree about where a block starts.
static BLOCK_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[(\w+)(?:\s+(\w+))?\]$").unwrap());

/// Any `.ferx` identifier token.
static IDENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z_]\w*").unwrap());

/// One `[block]` in the source: its header line and the half-open range of
/// body lines that follow it (up to the next header, or end of file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BlockSpan {
    /// Block type, lowercased — `parameters`, `covariate_model`, …
    pub(crate) name: String,
    /// The header line's index.
    pub(crate) header: usize,
    /// Body lines: `header + 1 .. next_header_or_eof`.
    pub(crate) body: Range<usize>,
}

/// Block-addressable `.ferx` source with every byte preserved.
///
/// Construct with [`ModelText::parse`], transform with [`ModelText::apply`],
/// and get the source back with [`ModelText::render`]. See the [module
/// docs](self) for why the representation is text rather than a syntax tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelText {
    /// Every source line **including its terminator**, so concatenating them
    /// reproduces the input exactly — CRLF, a missing final newline and all.
    lines: Vec<String>,
}

impl ModelText {
    /// Read `.ferx` source into block-addressable form.
    ///
    /// This is a *structural* parse: it locates block headers and nothing more.
    /// It deliberately does **not** compile the model, so a model that needs
    /// data-derived bindings (`center = median`, `levels = auto`) round-trips
    /// and edits like any other. Compile the result of [`render`](Self::render)
    /// when you want the model checked.
    ///
    /// # Errors
    ///
    /// When the source contains no `[block]` header at all — there is nothing
    /// addressable to edit, and accepting it silently would turn every
    /// subsequent edit into a confusing "block not found".
    pub fn parse(src: &str) -> Result<Self, String> {
        let lines = split_keeping_terminators(src);
        let text = ModelText { lines };
        if text.block_spans().is_empty() {
            return Err(
                "ferx-core::edit: the source contains no `[block]` header, so there is nothing \
                 to address — expected at least `[parameters]`, `[individual_parameters]` and \
                 `[structural_model]`"
                    .to_string(),
            );
        }
        Ok(text)
    }

    /// The source. Byte-identical to what [`parse`](Self::parse) was given
    /// when no edit has been applied.
    pub fn render(&self) -> String {
        self.lines.concat()
    }

    /// Apply one transformation in place.
    ///
    /// On `Err` the text is left **untouched** — every variant validates its
    /// preconditions before it mutates anything, so a rejected edit cannot
    /// leave a half-written model behind.
    pub fn apply(&mut self, edit: ModelEdit<'_>) -> Result<(), String> {
        apply::apply(self, edit)
    }

    /// A stable identity for the model this text describes.
    ///
    /// Equal for two files that differ only in comments, indentation, blank
    /// lines, line endings or spacing inside an expression; different for any
    /// token-level change. Use it as a candidate cache key — "have I already
    /// fitted this model?" — rather than hashing the raw bytes, which would
    /// miss the cache on a reflow.
    pub fn canonical_hash(&self) -> [u8; 32] {
        canonical::canonical_hash(&self.lines)
    }

    /// The canonicalised text [`canonical_hash`](Self::canonical_hash) digests.
    ///
    /// Exposed because a cache miss you cannot explain is a cache you stop
    /// trusting: diffing two canonical forms says exactly which token made two
    /// candidates different.
    pub fn canonical_form(&self) -> String {
        canonical::canonical_form(&self.lines)
    }

    /// The `[block]` names present, lowercased, in source order. A block that
    /// appears twice appears twice here.
    pub fn block_names(&self) -> Vec<String> {
        self.block_spans().into_iter().map(|b| b.name).collect()
    }

    /// The body lines of every `[name]` block, comment-stripped and trimmed,
    /// with blank lines dropped — the view the parser gets.
    pub fn block_lines(&self, name: &str) -> Vec<String> {
        self.block_spans()
            .iter()
            .filter(|b| b.name == name)
            .flat_map(|b| self.lines[b.body.clone()].iter())
            .map(|l| code_of(l).to_string())
            .filter(|l| !l.is_empty())
            .collect()
    }

    // ── Internals shared by the appliers ────────────────────────────────────

    /// Every block header in source order.
    pub(crate) fn block_spans(&self) -> Vec<BlockSpan> {
        let mut headers: Vec<(usize, String)> = Vec::new();
        for (i, line) in self.lines.iter().enumerate() {
            if let Some(caps) = BLOCK_HEADER_RE.captures(code_of(line)) {
                headers.push((i, caps[1].to_lowercase()));
            }
        }
        let mut out = Vec::with_capacity(headers.len());
        for (k, (header, name)) in headers.iter().enumerate() {
            let end = headers.get(k + 1).map(|h| h.0).unwrap_or(self.lines.len());
            out.push(BlockSpan {
                name: name.clone(),
                header: *header,
                body: header + 1..end,
            });
        }
        out
    }

    /// The **last** `[name]` block, which is where new lines go.
    ///
    /// A block may legally appear more than once — the parser concatenates the
    /// bodies — and appending to the last occurrence is the only choice that
    /// preserves declaration order for every duplicate arrangement.
    pub(crate) fn last_block(&self, name: &str) -> Option<BlockSpan> {
        self.block_spans()
            .into_iter()
            .filter(|b| b.name == name)
            .next_back()
    }

    /// Line indices of every body line of every `[name]` block, in source order.
    pub(crate) fn body_indices(&self, name: &str) -> Vec<usize> {
        self.block_spans()
            .iter()
            .filter(|b| b.name == name)
            .flat_map(|b| b.body.clone())
            .collect()
    }

    /// The comment-stripped, trimmed content of line `i`.
    pub(crate) fn code(&self, i: usize) -> &str {
        code_of(&self.lines[i])
    }

    /// Replace line `i`, keeping its original indentation and terminator.
    pub(crate) fn set_code(&mut self, i: usize, code: &str) {
        let indent = leading_ws(&self.lines[i]).to_string();
        let term = terminator_of(&self.lines[i]).to_string();
        self.lines[i] = format!("{indent}{code}{term}");
    }

    /// Delete the given line indices.
    pub(crate) fn delete_lines(&mut self, mut idx: Vec<usize>) {
        idx.sort_unstable();
        idx.dedup();
        for i in idx.into_iter().rev() {
            self.lines.remove(i);
        }
    }

    /// Append `code` to the end of the last `[block]`, creating the block at
    /// end of file if it does not exist.
    ///
    /// "End of the block" means after its last non-blank line, so a blank
    /// separator before the next header stays where the author put it.
    pub(crate) fn append_to_block(&mut self, block: &str, code: &str) {
        let (at, indent) = match self.last_block(block) {
            Some(span) => {
                let mut at = span.body.end;
                while at > span.body.start && self.code(at - 1).is_empty() {
                    at -= 1;
                }
                (at, self.block_indent(&span))
            }
            None => {
                self.append_block_header(block);
                (self.lines.len(), "  ".to_string())
            }
        };
        let term = self.newline();
        self.lines.insert(at, format!("{indent}{code}{term}"));
    }

    /// Start a new `[block]` at end of file, preceded by a blank line.
    fn append_block_header(&mut self, block: &str) {
        let term = self.newline();
        // Guarantee the previous line is terminated, or the new header would be
        // glued onto it (a source file may end without a final newline).
        if let Some(last) = self.lines.last_mut() {
            if terminator_of(last).is_empty() {
                last.push_str(&term);
            }
        }
        if self.lines.last().is_some_and(|l| !code_of(l).is_empty()) {
            self.lines.push(term.clone());
        }
        self.lines.push(format!("[{block}]{term}"));
    }

    /// The indentation to give a new line in `span`: whatever its first
    /// non-blank body line uses, or two spaces for an empty block.
    fn block_indent(&self, span: &BlockSpan) -> String {
        self.lines[span.body.clone()]
            .iter()
            .find(|l| !code_of(l).is_empty())
            .map(|l| leading_ws(l).to_string())
            .unwrap_or_else(|| "  ".to_string())
    }

    /// The file's line terminator: CRLF if any line uses it, else LF.
    pub(crate) fn newline(&self) -> String {
        if self.lines.iter().any(|l| l.ends_with("\r\n")) {
            "\r\n".to_string()
        } else {
            "\n".to_string()
        }
    }

    /// Every identifier mentioned in code, excluding the lines in `skip`.
    ///
    /// This is the reachability test the declaration pruner runs: a `theta` or
    /// `omega` whose name survives nowhere but its own declaration is dead.
    pub(crate) fn identifiers_excluding(&self, skip: &[usize]) -> HashSet<String> {
        let mut out = HashSet::new();
        for (i, line) in self.lines.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            for m in IDENT_RE.find_iter(code_of(line)) {
                out.insert(m.as_str().to_string());
            }
        }
        out
    }
}

/// Split into lines that each keep their own terminator, so `concat()` is the
/// identity. `str::lines` cannot do this — it discards `\n` and `\r\n` alike
/// and cannot tell a final newline from its absence.
fn split_keeping_terminators(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, b) in src.bytes().enumerate() {
        if b == b'\n' {
            out.push(src[start..=i].to_string());
            start = i + 1;
        }
    }
    if start < src.len() {
        out.push(src[start..].to_string());
    }
    out
}

/// A line with its comment and surrounding whitespace removed — what the
/// parser reads. `#` and `//` both start a comment, whichever comes first.
pub(crate) fn code_of(line: &str) -> &str {
    let cut = line
        .find('#')
        .into_iter()
        .chain(line.find("//"))
        .min()
        .unwrap_or(line.len());
    line[..cut].trim()
}

/// The leading whitespace of a line, excluding its own terminator (a blank
/// line is all terminator and must not donate a newline as "indentation").
fn leading_ws(line: &str) -> &str {
    let end = line
        .find(|c: char| !c.is_whitespace() || c == '\n' || c == '\r')
        .unwrap_or(line.len());
    &line[..end]
}

fn terminator_of(line: &str) -> &str {
    if line.ends_with("\r\n") {
        "\r\n"
    } else if line.ends_with('\n') {
        "\n"
    } else {
        ""
    }
}
