//! Split a raw mrgsolve source file into a `Vec<RawBlock>`.
//!
//! A mrgsolve block looks like `$NAME header_args\nbody_line_1\nbody_line_2\n...`
//! and runs until the next `$NAME` opener or end-of-file. Comments (`//` and
//! `/* ... */`) are stripped before splitting so they don't leak into block
//! bodies. Source line numbers are preserved so downstream parsers can report
//! `file:line:` errors that point back at the original `.cpp`/`.mod` file.
//!
//! This layer does no semantic interpretation — it just chops the file into
//! labelled chunks. `blocks.rs` consumes the chunks and dispatches per-block.

/// A single `$BLOCK` chunk extracted from the source file. The body is split
/// into lines so block parsers can either rejoin them (C++ body blocks like
/// `$MAIN`) or process them line-by-line (declaration blocks like `$OMEGA`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawBlock {
    /// Block name in uppercase, without the leading `$` (e.g. `"PARAM"`).
    pub(crate) name: String,
    /// Anything that appeared on the same line as the `$BLOCK` opener,
    /// after the block name. Useful for `$PKMODEL ncmt=1, depot=TRUE` or
    /// inline `$OMEGA 0.04 0.09` forms.
    pub(crate) header_args: String,
    /// Body lines after the opener, comment-stripped, with line numbers
    /// matching the original 1-indexed source positions.
    pub(crate) body_lines: Vec<BodyLine>,
    /// 1-indexed line number of the `$BLOCK` opener line itself.
    pub(crate) start_line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BodyLine {
    pub(crate) line_no: usize,
    pub(crate) text: String,
}

/// Strip `//`-to-end-of-line and `/* ... */` comments from the source. The
/// stripped string preserves line breaks (block comments collapse to a
/// single newline per spanned line) so source line numbers stay aligned.
fn strip_comments(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            // Line comment — skip until newline (but keep the newline).
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            // Block comment — skip until `*/`, preserving any newlines so
            // line numbers downstream stay aligned with the original file.
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                if chars[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            if i + 1 < chars.len() {
                i += 2; // consume the closing */
            } else {
                // Unterminated block comment — bail to EOF.
                i = chars.len();
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Split a mrgsolve source string into its constituent `$BLOCK` chunks.
/// Returns an error if a `$BLOCK` opener is followed by anything that isn't
/// an identifier, or if no `$BLOCK` openers appear at all.
pub(crate) fn extract(src: &str) -> Result<Vec<RawBlock>, String> {
    let stripped = strip_comments(src);
    let mut blocks: Vec<RawBlock> = Vec::new();
    let mut current: Option<RawBlock> = None;

    for (idx, raw_line) in stripped.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('$') {
            // New block opener. Parse `NAME header_args...`.
            let mut chars = rest.char_indices();
            let name_end = loop {
                match chars.next() {
                    Some((i, c)) if c.is_ascii_alphanumeric() || c == '_' => {
                        let _ = i;
                    }
                    Some((i, _)) => break i,
                    None => break rest.len(),
                }
            };
            if name_end == 0 {
                return Err(format!(
                    "line {}: `$` is not followed by a block name",
                    line_no
                ));
            }
            let name = rest[..name_end].to_ascii_uppercase();
            let header_args = rest[name_end..].trim().to_string();

            // Stash the previous block before starting the new one.
            if let Some(b) = current.take() {
                blocks.push(b);
            }
            current = Some(RawBlock {
                name,
                header_args,
                body_lines: Vec::new(),
                start_line: line_no,
            });
        } else if let Some(b) = current.as_mut() {
            // Body of the current block. Trim trailing whitespace but keep
            // the line — even blank lines — so the C++ tokenizer can see
            // statement boundaries.
            b.body_lines.push(BodyLine {
                line_no,
                text: raw_line.trim_end().to_string(),
            });
        } else {
            // Pre-block content (e.g. file header). Allow it if it's
            // whitespace; otherwise error.
            if !raw_line.trim().is_empty() {
                return Err(format!(
                    "line {}: unexpected text before any `$BLOCK` opener: {:?}",
                    line_no,
                    raw_line.trim()
                ));
            }
        }
    }
    if let Some(b) = current.take() {
        blocks.push(b);
    }
    if blocks.is_empty() {
        return Err("no `$BLOCK` openers found — is this an mrgsolve model file?".to_string());
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_text(b: &RawBlock) -> Vec<&str> {
        b.body_lines.iter().map(|l| l.text.as_str()).collect()
    }

    #[test]
    fn extracts_three_blocks_with_correct_line_numbers() {
        let src = "\
$PROB warfarin demo
$PARAM TVCL = 1, TVV = 10
$CMT GUT CENT
";
        let blocks = extract(src).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].name, "PROB");
        assert_eq!(blocks[0].header_args, "warfarin demo");
        assert_eq!(blocks[0].start_line, 1);
        assert_eq!(blocks[1].name, "PARAM");
        assert_eq!(blocks[1].header_args, "TVCL = 1, TVV = 10");
        assert_eq!(blocks[2].name, "CMT");
        assert_eq!(blocks[2].header_args, "GUT CENT");
    }

    #[test]
    fn captures_body_lines_with_line_numbers() {
        let src = "\
$PARAM TVCL = 1
$MAIN
double CL = TVCL * exp(ETA(1));
double V  = TVV;
";
        let blocks = extract(src).unwrap();
        assert_eq!(blocks[1].name, "MAIN");
        assert_eq!(blocks[1].body_lines.len(), 2);
        assert_eq!(blocks[1].body_lines[0].line_no, 3);
        assert_eq!(blocks[1].body_lines[0].text, "double CL = TVCL * exp(ETA(1));");
        assert_eq!(blocks[1].body_lines[1].line_no, 4);
    }

    #[test]
    fn strips_line_and_block_comments() {
        let src = "\
$PARAM TVCL = 1 // pop CL
/* block
   comment
*/$MAIN
double CL = TVCL; // here too
";
        let blocks = extract(src).unwrap();
        // `$MAIN` is on the same line as the closing */ — block comment
        // collapses to a single token, leaving `$MAIN` as the opener.
        let names: Vec<_> = blocks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["PARAM", "MAIN"]);
        assert_eq!(blocks[0].header_args, "TVCL = 1");
        let main_body = body_text(&blocks[1]);
        assert_eq!(main_body, vec!["double CL = TVCL;"]);
    }

    #[test]
    fn rejects_source_with_no_blocks() {
        let err = extract("// nothing here\n").unwrap_err();
        assert!(err.contains("no `$BLOCK` openers"));
    }

    #[test]
    fn rejects_text_before_first_block() {
        let err = extract("garbage\n$PARAM CL = 1\n").unwrap_err();
        assert!(err.contains("line 1"));
        assert!(err.contains("unexpected text"));
    }

    #[test]
    fn block_name_uppercased() {
        let src = "$param TVCL=1\n$Cmt GUT\n";
        let blocks = extract(src).unwrap();
        assert_eq!(blocks[0].name, "PARAM");
        assert_eq!(blocks[1].name, "CMT");
    }
}
