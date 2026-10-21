//! #1051: the `E_*` / `W_*` code registry is exhaustive. Every code the engine emits
//! has a row in **both** registries — the table in `src/diagnostics.rs`'s module doc
//! and the code tables of `docs/file-formats/check-report.qmd` — and neither lists a
//! code nothing emits. Before this check the rule was prose only, and 62 of 98 emitted
//! codes had drifted out of the in-code table.

use super::uncoded_sites::{production_files, production_lines};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const REGISTRY: &str = "src/diagnostics.rs";
const DOCS: &str = "docs/file-formats/check-report.qmd";

/// Whether `word` is a code: `E_` or `W_`, an uppercase letter, then `[A-Z0-9_]`, not
/// ending in `_` — a family prefix such as `"E_ENDPOINT_"` in a `starts_with` is not one.
fn is_code(word: &str) -> bool {
    let b = word.as_bytes();
    b.len() >= 3
        && (b[0] == b'E' || b[0] == b'W')
        && b[1] == b'_'
        && b[2].is_ascii_uppercase()
        && b.iter()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == b'_')
        && b[b.len() - 1] != b'_'
}

/// The codes in `line`: its maximal `[A-Za-z0-9_]` runs that are codes, so
/// `W_TYPICAL_PK_{}` (an NN weight's name template) and `xE_FOO` are not.
fn code_tokens(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| is_code(w))
}

/// Code → first `file:line` emitting it in the non-test code under `root/src`.
///
/// Comment lines are skipped — doc prose cites codes the line does not emit — but the
/// rest of a line is not, because a code reaches the user in three spellings: the code
/// of a `Diagnostic` or `EngineError` (`Diagnostic::error("E_X", …)`), the head of a
/// `FitResult.warnings` or data-reader message (`"W_MISSING_DV: …"`), and a code quoted
/// in a parse error's text (`"… (E_DERIVED_NAME_CONFLICT)"`). A code computed at run
/// time (`format!("E_{kind}")`) would be invisible here; keep every code a literal.
fn emitted(root: &Path) -> BTreeMap<String, String> {
    let mut found = BTreeMap::new();
    for (rel, text) in production_files(root) {
        for (i, line) in production_lines(&text).iter().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for code in code_tokens(line) {
                found
                    .entry(code.to_string())
                    .or_insert_with(|| format!("{rel}:{}", i + 1));
            }
        }
    }
    found
}

/// Code → the second cell of every table row of `text` whose first cell is that code
/// in backticks, after stripping `prefix` (`"//! "` in the registry's module doc). A
/// code's prose mention outside a row's first cell does not count as a row.
fn table_rows(text: &str, prefix: &str) -> BTreeMap<String, Vec<String>> {
    let mut rows: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in text.lines() {
        let Some(row) = line.strip_prefix(prefix) else {
            continue;
        };
        let mut cells = row.split('|').map(str::trim);
        if cells.next() != Some("") {
            continue;
        }
        let Some(first) = cells.next() else { continue };
        let Some(code) = first.strip_prefix('`').and_then(|c| c.strip_suffix('`')) else {
            continue;
        };
        if is_code(code) {
            let second = cells.next().unwrap_or("").to_string();
            rows.entry(code.to_string()).or_default().push(second);
        }
    }
    rows
}

/// Every way the two registries can disagree with the emitted codes, one line each.
fn registry_problems(
    emitted: &BTreeMap<String, String>,
    registry: &BTreeMap<String, Vec<String>>,
    docs: &BTreeMap<String, Vec<String>>,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (code, site) in emitted {
        if !registry.contains_key(code) {
            problems.push(format!(
                "{code} (emitted at {site}) has no row in {REGISTRY}'s registry table"
            ));
        }
        if !docs.contains_key(code) {
            problems.push(format!(
                "{code} (emitted at {site}) has no row in {DOCS}'s code tables"
            ));
        }
    }
    for (table, rows) in [(REGISTRY, registry), (DOCS, docs)] {
        for (code, cells) in rows {
            if !emitted.contains_key(code) {
                problems.push(format!(
                    "{code} has a row in {table} but nothing in src/ emits it — remove the stale row"
                ));
            }
            if cells.len() > 1 {
                problems.push(format!("{code} has {} rows in {table}", cells.len()));
            }
        }
    }
    // The docs tables carry a severity column; the code's prefix is its severity.
    for (code, cells) in docs {
        let want = if code.starts_with("E_") {
            "error"
        } else {
            "warning"
        };
        for got in cells.iter().filter(|c| c.as_str() != want) {
            problems.push(format!(
                "{code}'s row in {DOCS} says severity `{got}`; an {} code is `{want}`",
                &code[..2]
            ));
        }
    }
    problems
}

/// Mutations — emit a new code in `src/` with no rows → red (named twice: once per
/// table); delete a row from either table → red; leave a row for a code nothing emits
/// → red; list a code twice, or under the wrong severity in the docs → red.
#[test]
fn every_emitted_code_has_a_row_in_both_registries() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let emitted = emitted(root);
    let registry = table_rows(
        &std::fs::read_to_string(root.join(REGISTRY)).expect("read the registry"),
        "//! ",
    );
    let docs = table_rows(
        &std::fs::read_to_string(root.join(DOCS)).expect("read the check-report page"),
        "",
    );
    // Non-degeneracy: each scan found codes in each spelling it claims to read, so a
    // green result is not three empty sets agreeing.
    for code in ["E_PARSE", "W_MISSING_DV", "E_DERIVED_NAME_CONFLICT"] {
        assert!(emitted.contains_key(code), "the scan missed {code}");
        assert!(
            registry.contains_key(code),
            "the registry parse missed {code}"
        );
        assert!(docs.contains_key(code), "the docs parse missed {code}");
    }
    let problems = registry_problems(&emitted, &registry, &docs);
    assert!(
        problems.is_empty(),
        "the error-code registry has drifted from the emitted codes (#1051). Every code \
         needs a row in the table of {REGISTRY}'s module doc and in a code table of {DOCS}:\n{}",
        problems.join("\n")
    );
}

/// The scan reads each spelling of an emitted code, also after a mid-file test module,
/// and skips comments, family prefixes, name templates and test code — so a green
/// `every_emitted_code_has_a_row_in_both_registries` is not a scan that sees nothing.
#[test]
fn the_scan_sees_planted_codes_and_skips_the_rest() {
    let dir = std::env::temp_dir().join(format!("ferx-code-registry-{}", std::process::id()));
    let src = dir.join("src");
    std::fs::create_dir_all(src.join("tests")).unwrap();
    std::fs::write(
        src.join("a.rs"),
        "pub fn f() -> R {\n    // E_IN_COMMENT is only prose\n    \
         let _ = Diagnostic::error(\"E_PLANTED\", \"x\");\n    \
         w.push(format!(\"W_PLANTED_HEAD: {n} subjects\"));\n    \
         Err(format!(\"clash (E_PLANTED_QUOTED)\"))\n}\n\
         fn g(c: &str) -> bool { c.starts_with(\"E_FAMILY_\") }\n\
         fn h(l: usize) -> String { format!(\"W_TEMPLATE_{}\", l) }\n\
         \n#[cfg(test)]\nmod tests {\n    fn t() { e(\"E_IN_TEST_MODULE\"); }\n}\n\
         \npub(crate) fn after() {\n    let _ = Diagnostic::warning(\"W_AFTER_TESTS\", \"y\");\n}\n",
    )
    .unwrap();
    std::fs::write(src.join("a_tests.rs"), "fn t() { e(\"E_IN_SIBLING\"); }\n").unwrap();
    std::fs::write(
        src.join("tests").join("b.rs"),
        "fn t() { e(\"E_IN_TESTS_DIR\"); }\n",
    )
    .unwrap();
    let found = emitted(&dir);
    std::fs::remove_dir_all(&dir).ok();
    let codes: BTreeSet<&str> = found.keys().map(String::as_str).collect();
    assert_eq!(
        codes,
        BTreeSet::from([
            "E_PLANTED",
            "E_PLANTED_QUOTED",
            "W_AFTER_TESTS",
            "W_PLANTED_HEAD"
        ])
    );
    assert_eq!(found["E_PLANTED"], "src/a.rs:3");
}

/// The row parser takes a row's first cell only, in both table spellings, and the
/// check names each kind of drift.
#[test]
fn the_check_names_each_kind_of_drift() {
    let registry = table_rows(
        "//! | code | meaning |\n//! |---|---|\n//! | `E_A_ROW` | see `E_PROSE` |\n\
         //! | `E_STALE` | gone |\n| `E_NOT_IN_DOC_COMMENT` | x |\n",
        "//! ",
    );
    assert_eq!(
        registry.keys().map(String::as_str).collect::<Vec<_>>(),
        ["E_A_ROW", "E_STALE"]
    );
    let docs = table_rows(
        "| Code | Severity | Meaning |\n|---|---|---|\n| `E_A_ROW` | warning | x |\n\
         | `E_A_ROW` | error | again |\nProse naming `W_UNROWED`.\n",
        "",
    );
    let emitted: BTreeMap<String, String> = [
        ("E_A_ROW".to_string(), "src/a.rs:1".to_string()),
        ("W_UNROWED".to_string(), "src/a.rs:2".to_string()),
    ]
    .into();
    let problems = registry_problems(&emitted, &registry, &docs);
    let want = [
        "W_UNROWED (emitted at src/a.rs:2) has no row in src/diagnostics.rs's registry table",
        "W_UNROWED (emitted at src/a.rs:2) has no row in docs/file-formats/check-report.qmd's code tables",
        "E_STALE has a row in src/diagnostics.rs but nothing in src/ emits it — remove the stale row",
        "E_A_ROW has 2 rows in docs/file-formats/check-report.qmd",
        "E_A_ROW's row in docs/file-formats/check-report.qmd says severity `warning`; an E_ code is `error`",
    ];
    assert_eq!(problems, want);
}
