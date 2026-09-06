//! Guards for the `ferx-core` / `ferx-tools` / `ferx-cli` split (#1114).
//!
//! The committed baseline at `api/ferx-core-public-api.txt` (A2) is the main
//! enforcement of the API boundary, and CI diffs it via
//! `tools/update-public-api.sh --check`. The invariants that baseline *cannot*
//! see are pinned here instead — plus one that has nothing to do with the API
//! surface but everything to do with the split: the checked-in commands that
//! invoke the `ferx` binary.
//!
//! Not feature-gated: these must run in the base `--features ci` job, the one
//! job guaranteed to build on every PR.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable directory") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// **A3 — no `#[doc(hidden)] pub` escape hatch.**
///
/// This is not style policing: `cargo public-api` *omits* `#[doc(hidden)]`
/// items entirely, so marking a new `pub fn` hidden makes it invisible to the
/// A2 baseline diff. Verified directly — adding `#[doc(hidden)] pub fn probe()`
/// to `src/lib.rs` leaves `--check` green, while the same `pub fn` without the
/// attribute fails it with exactly one added line. So the attribute is a
/// working bypass of the boundary gate, and the only place to close it is here.
///
/// An item is either public API — documented, in the baseline, usable from
/// ferx-r — or it stays `pub(crate)` and the caller does without.
///
/// Matched in *attribute position* rather than as a raw substring, so the ban
/// can be written down (in a doc comment, a string literal, this file's own
/// prose) without tripping its own guard, and `#[cfg_attr(…, doc(hidden))]` —
/// the same bypass wearing a hat — is caught too. Remaining gap: an attribute
/// broken across source lines, which `rustfmt` does not produce for one this
/// short.
#[test]
fn no_doc_hidden_escape_hatch() {
    let mut sources = Vec::new();
    rust_sources(&repo_root().join("src"), &mut sources);
    assert!(!sources.is_empty(), "found no sources under src/");

    let mut offenders = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("source file is valid UTF-8");
        for (i, line) in src.lines().enumerate() {
            let trimmed = line.trim_start();
            let hides = trimmed.starts_with("#[doc(hidden)]")
                || (trimmed.starts_with("#[cfg_attr(") && trimmed.contains("doc(hidden)"));
            if hides {
                offenders.push(format!(
                    "{}:{}",
                    path.strip_prefix(repo_root()).unwrap_or(path).display(),
                    i + 1
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "`#[doc(hidden)]` hides an item from the `api/ferx-core-public-api.txt` \
         baseline, so it is a silent bypass of the A2 API gate (#1114 A3). \
         Make the item genuinely public (document it and regenerate the \
         baseline with `tools/update-public-api.sh`) or make it `pub(crate)`. \
         Found at: {offenders:?}"
    );
}

/// Drop a trailing `#` comment from a TOML line, honouring quotes so a `#`
/// inside a string value survives.
fn strip_toml_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut quote: Option<u8> = None;
    for (i, &b) in bytes.iter().enumerate() {
        match (quote, b) {
            (None, b'\'' | b'"') => quote = Some(b),
            (Some(q), c) if c == q => quote = None,
            (None, b'#') => return &line[..i],
            _ => {}
        }
    }
    line
}

/// **A1 — one-way dependency.** `ferx-core` never depends on `ferx-tools` or
/// `ferx-cli`.
///
/// Cargo would reject an actual cycle, but not every violation is a cycle: a
/// `dev-dependency` on `ferx-cli` compiles fine and would quietly make the core
/// test suite depend on the layer above it, which is exactly the coupling the
/// split exists to prevent.
#[test]
fn ferx_core_does_not_depend_on_the_layers_above_it() {
    let manifest =
        std::fs::read_to_string(repo_root().join("Cargo.toml")).expect("root Cargo.toml is UTF-8");
    // Trim the `[workspace]` table — it names the members by path on purpose.
    let package_section = manifest
        .split_once("\n[package]")
        .map(|(_, rest)| rest)
        .expect("root manifest has a [package] section");
    // Comments are prose, not dependencies. `exclude`/`[features]` are an
    // obvious place to explain why `crates/` needs no entry, and a bare
    // substring match would read that explanation as a forbidden dependency.
    let package_code = package_section
        .lines()
        .map(strip_toml_comment)
        .collect::<Vec<_>>()
        .join("\n");

    for forbidden in ["ferx-tools", "ferx-cli"] {
        assert!(
            !package_code.contains(forbidden),
            "the `ferx-core` package must not depend on `{forbidden}` in any \
             dependency table (#1114 A1) — the dependency is strictly one-way"
        );
    }
}

// ── Checked-in `ferx` CLI invocations ───────────────────────────────────────

/// Text files that can plausibly carry a shell command a reader would run.
const SCANNED_EXTENSIONS: [&str; 6] = ["md", "qmd", "sh", "ferx", "R", "rs"];

/// Every **tracked** file with a scannable extension.
///
/// `git ls-files` rather than a filesystem walk, for the same reason
/// `slow_test_path_filter.rs` uses it: a walk scans whatever happens to be
/// lying in the working tree, and only committed text can carry an instruction
/// a reader would copy. Concretely — this repo's worktree workflow parks
/// sibling checkouts at `<repo>/.claude/worktrees/<branch>/`, and `.claude/` is
/// git-ignored, so a walk from the main checkout reads every *other* branch's
/// copy of the whole tree and reports thousands of offenders that no edit to a
/// tracked file can fix. Extension-filtering an ignored tree is not a filter
/// anyone can act on; tracking status is.
fn scannable_files(root: &Path) -> Vec<PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .expect("`git ls-files` runs (the guard scans committed files only)");
    assert!(
        out.status.success(),
        "`git ls-files` failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("tracked paths are UTF-8")
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(|s| root.join(s))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| SCANNED_EXTENSIONS.contains(&e))
        })
        .collect()
}

/// **The `ferx` binary lives in `ferx-cli`, so every checked-in invocation of it
/// must say so.**
///
/// Moving the binary out of the root package silently invalidated ~53 commands
/// spread across `examples/*.ferx` headers, `nonmem_anchor/`, `plans/`,
/// `tests/reference/`, and the executable harnesses under `tools/`. They are
/// documentation and shell scripts, so nothing compiled them and nothing failed:
/// a build naming the root-package bin now dies with *"no bin target named
/// `ferx` in default-run packages"*, and a package-less `cargo run --release`
/// passing a model through `--` dies the same way. The first sweep of this PR
/// caught the `docs/` copies and missed every other tree, which is why the
/// invariant is pinned in a test rather than trusted to a grep done once.
///
/// Two failure shapes are checked:
///
/// * a run/build that reaches the CLI without selecting its package, and
/// * a `ferx-cli`-scoped command passing a bare `--features ci|survival|…`,
///   which fails with *"none of the selected packages contains this feature"*
///   because those features belong to `ferx-core` and need qualifying.
///
/// Invocations of the binaries that genuinely stayed in `ferx-core`
/// (`generate_data`, `pktte_sim_anchor`, `rtte_sim_anchor`) name their own
/// `--bin` and are left alone.
/// Whether one line of checked-in text is a `cargo run`/`cargo build` that no
/// longer works now that the binary lives in `ferx-cli`.
///
/// Split out of the test so the classification itself is pinned by
/// `invocation_classifier_excuses_the_explicit_form` below. The excusals are
/// where a rule like this gets it wrong, and a scan over the tree only proves
/// that *today's* tracked files happen not to contain the excused shapes.
fn is_broken_ferx_invocation(line: &str) -> bool {
    if !(line.contains("cargo run") || line.contains("cargo build")) {
        return false;
    }
    // Needles are built with `format!` rather than written literally so this
    // file does not match itself — the same trick
    // `ci_workflow_endpoint_coverage.rs` uses, and what lets the doc comments
    // here describe the broken commands without becoming offenders.
    let cli_pkg = format!("-p {}", "ferx-cli");
    let root_bin = format!("--bin {}", "ferx");
    let core_features = ["ci", "survival", "markov", "nn", "slow-tests"];

    // The root-package `ferx` bin, and not a longer name that merely starts
    // with it. A `--bin ferx` that ALSO selects the package is the most
    // explicit *correct* form of the invocation, so it is excused here exactly
    // as it is in `unscoped_run` below.
    let names_root_bin = !line.contains(cli_pkg.as_str())
        && line
            .split_once(root_bin.as_str())
            .is_some_and(|(_, rest)| !rest.starts_with('-') && !rest.starts_with('_'));
    // A run that passes arguments through `--` but never selects a package.
    // `--bin <other>` excuses it: that is one of the bins still in core.
    //
    // Any explicit `-p <pkg>` / `--package <pkg>` excuses it too, and the check
    // is on the *form* rather than on a list of package names: the workspace
    // gained `docs-lint` (#1163), whose `cargo run -p docs-lint -- --check` is
    // correct precisely because it selects its package. Enumerating the members
    // here would make every new one a false positive on its first commit.
    //
    // Only the CARGO arguments count. Everything after the `--` separator goes
    // to the program, so a `run --release -- model.ferx -p 8` — where the `-p`
    // is the *model's* flag, not cargo's — selects no package and stays an
    // offender. (Spelled without the leading verb so this line is not itself
    // scanned as one, the same reason the needles below use `format!`.)
    let cargo_args = line.split(" -- ").next().unwrap_or(line);
    let selects_a_package = cargo_args.contains("-p ") || cargo_args.contains("--package ");
    let unscoped_run = line.contains(" -- ") && !selects_a_package && !line.contains("--bin ");
    // Feature flags belong to `ferx-core`, so a member-scoped command has to
    // qualify them.
    let unqualified_feature = line.contains(cli_pkg.as_str())
        && core_features.iter().any(|f| {
            line.contains(&format!("--features {f}")) || line.contains(&format!("--features={f}"))
        });

    names_root_bin || unscoped_run || unqualified_feature
}

#[test]
fn checked_in_ferx_invocations_select_the_ferx_cli_package() {
    let files = scannable_files(&repo_root());
    assert!(!files.is_empty(), "found no scannable files");

    let mut offenders = Vec::new();
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        for (i, line) in src.lines().enumerate() {
            if is_broken_ferx_invocation(line) {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(repo_root()).unwrap_or(path).display(),
                    i + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these checked-in commands no longer run: the `ferx` binary moved to the \
         `ferx-cli` package (#1114), so a run/build must select it with \
         `-p ferx-cli`, and any `ferx-core` feature must be written qualified \
         (`--features ferx-core/survival`). Offenders:\n  {}",
        offenders.join("\n  ")
    );
}

/// The classifier's *excusals*, which the tree scan above cannot exercise: it
/// only shows that today's tracked files contain no correct-but-flagged command,
/// so a false positive stays invisible until someone writes one into a doc and
/// gets a red test they cannot explain.
///
/// Every fixture is assembled with `format!` for the same reason the needles
/// are — a literal broken command written here would be scanned, and this file
/// would flag itself.
#[test]
fn invocation_classifier_excuses_the_explicit_form() {
    let run = format!("cargo {}", "run --release");
    let build = format!("cargo {}", "build --release");
    let pkg = format!("-p {}", "ferx-cli");
    let bin = format!("--bin {}", "ferx");

    // Broken: the root package no longer has a `ferx` bin, and a package-less
    // run passing arguments through `--` resolves against the root package.
    assert!(is_broken_ferx_invocation(&format!("{build} {bin}")));
    assert!(is_broken_ferx_invocation(&format!(
        "{run} -- examples/warfarin.ferx"
    )));
    // Member-scoped, but the feature belongs to `ferx-core` and is unqualified.
    assert!(is_broken_ferx_invocation(&format!(
        "{run} {pkg} --features ci -- model.ferx"
    )));

    // Correct, and must NOT be flagged. Naming the bin alongside the package is
    // redundant but legal, and it is the most explicit form a reader can write.
    assert!(!is_broken_ferx_invocation(&format!(
        "{run} {pkg} {bin} -- model.ferx"
    )));
    assert!(!is_broken_ferx_invocation(&format!(
        "{run} {pkg} -- model.ferx"
    )));
    // A bin that genuinely stayed in `ferx-core`.
    assert!(!is_broken_ferx_invocation(&format!(
        "{run} --bin generate_data -- out.csv"
    )));
    // Another workspace member, selected explicitly: correct for the same
    // reason `-p ferx-cli` is, and the guard must not read "not the CLI" as
    // "no package" (#1163).
    assert!(!is_broken_ferx_invocation(
        "cargo run -p docs-lint -- --check"
    ));
    assert!(!is_broken_ferx_invocation(
        "cargo run --package docs-lint -- --update-baseline"
    ));
    // A `-p` that belongs to the PROGRAM, not to cargo, selects no package and
    // must stay flagged — otherwise the excusal above swallows exactly the
    // invocation shape this guard exists to catch (#1114).
    assert!(is_broken_ferx_invocation(&format!(
        "{run} -- model.ferx -p 8"
    )));
    assert!(is_broken_ferx_invocation(&format!(
        "{run} -- model.ferx --package foo"
    )));
    // Features qualified for a member-scoped build.
    assert!(!is_broken_ferx_invocation(&format!(
        "{build} {pkg} --features ferx-core/ci"
    )));
    // Prose that merely mentions the binary.
    assert!(!is_broken_ferx_invocation(
        "the ferx binary reads a .ferx model file"
    ));
}

#[test]
fn toml_comments_are_stripped_but_string_hashes_survive() {
    // The A1 guard must read dependency tables, not prose: an explanatory
    // comment below `[package]` that names a member is not a dependency on it.
    let member = "ferx-cli";
    assert_eq!(
        strip_toml_comment(&format!("exclude = [] # crates/ holds {member}")),
        "exclude = [] "
    );
    assert_eq!(strip_toml_comment(&format!("# {member} is a member")), "");
    // A `#` inside a quoted value is not a comment.
    let url = "repository = \"https://example.com/a#b\"";
    assert_eq!(strip_toml_comment(url), url);
    let name = "name = \"ferx-core\"";
    assert_eq!(strip_toml_comment(name), name);
}

// ── A4 — every `pub *Options` type implements `Default` (#529) ───────────────

/// One `pub … Options` type declaration found by the source scan.
#[derive(Debug, PartialEq, Eq)]
struct OptionsDecl {
    name: String,
    /// `Default` appears in a `#[derive(...)]` attached to the declaration.
    derives_default: bool,
    /// 1-indexed line of the `pub struct` / `pub enum` itself.
    line: usize,
}

/// The attributes attached to the item declared at `decl_idx`, as one string.
///
/// Walks back to the nearest blank line — the block rustfmt keeps contiguous
/// above an item — and keeps only genuine attribute lines. Doc comments are
/// **dropped on purpose**: several of the types this guard covers explain their
/// `Default` in prose ("Derives `Default` so the wrapper can spread it"), and a
/// substring scan that read documentation would let a comment satisfy a check
/// about code. Continuation lines of a multi-line attribute are kept via the
/// paren depth, so a `#[derive(` split across lines is still seen whole.
fn attribute_block_above(lines: &[&str], decl_idx: usize) -> String {
    let mut start = decl_idx;
    while start > 0 && !lines[start - 1].trim().is_empty() {
        start -= 1;
    }
    let mut attrs = String::new();
    let mut depth = 0usize;
    for line in &lines[start..decl_idx] {
        let trimmed = line.trim();
        if depth == 0 && !trimmed.starts_with("#[") && !trimmed.starts_with("#![") {
            continue;
        }
        attrs.push_str(trimmed);
        attrs.push('\n');
        depth += trimmed.matches('(').count();
        depth = depth.saturating_sub(trimmed.matches(')').count());
    }
    attrs
}

/// Whether `attrs` contains a `#[derive(...)]` naming `Default` **as a whole
/// trait**, not as a substring: a hypothetical `DefaultDisplay` derive must not
/// satisfy the guard.
fn derive_list_names_default(attrs: &str) -> bool {
    let mut rest = attrs;
    while let Some(at) = rest.find("derive(") {
        let after = &rest[at + "derive(".len()..];
        let mut depth = 1usize;
        let mut end = after.len();
        for (i, c) in after.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let inner = &after[..end];
        if inner
            .split(',')
            .any(|t| matches!(t.trim(), "Default" | "core::default::Default"))
        {
            return true;
        }
        rest = &after[end.min(after.len())..];
    }
    false
}

/// Identifier starting at the front of `s`, or `None` if `s` does not start
/// with one.
fn leading_ident(s: &str) -> Option<&str> {
    let end = s
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    (end > 0).then(|| &s[..end])
}

/// Every `struct`/`enum` in `src` whose name ends in `Options`.
///
/// Deliberately name-driven rather than baseline-driven: `cargo public-api`
/// omits *derived* impls, so `api/ferx-core-public-api.txt` shows no
/// `impl Default` row for a type that derives it (only for the hand-written
/// ones). The committed baseline therefore cannot see this convention at all,
/// which is why it is scanned from source — the same argument that puts the
/// `#[doc(hidden)]` ban in this file.
fn scan_options_decls(src: &str) -> Vec<OptionsDecl> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let after_kind = trimmed
            .strip_prefix("pub struct ")
            .or_else(|| trimmed.strip_prefix("pub enum "));
        let Some(after_kind) = after_kind else {
            continue;
        };
        let Some(name) = leading_ident(after_kind) else {
            continue;
        };
        if !name.ends_with("Options") {
            continue;
        }
        out.push(OptionsDecl {
            name: name.to_string(),
            derives_default: derive_list_names_default(&attribute_block_above(&lines, idx)),
            line: idx + 1,
        });
    }
    out
}

/// Type names carrying a hand-written `impl Default for …` in `src`.
///
/// Six of the options types in this workspace implement `Default` manually
/// (`FitOptions`, `OdeSolverOptions`, `AdaptiveSimulateOptions`,
/// `BootstrapOptions`, `GamOptions`, `RunOptions`), so a derive-only scan would
/// report six false offenders.
fn scan_manual_default_impls(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim_start();
        for prefix in ["impl Default for ", "impl core::default::Default for "] {
            let Some(after) = trimmed.strip_prefix(prefix) else {
                continue;
            };
            // Take the last path segment so `impl Default for crate::a::B` and
            // `impl Default for B` agree, and drop any generic argument list.
            let head = after.split(['<', ' ', '{']).next().unwrap_or(after);
            if let Some(name) = head.rsplit("::").next().and_then(leading_ident) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// Rust sources under `dir`, skipping any `target/` build directory.
fn rust_sources_skipping_target(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable directory") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("target") {
                continue;
            }
            rust_sources_skipping_target(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// **A4 — every `pub *Options` type in the workspace implements `Default`
/// (#529).**
///
/// `ferx-r` builds these structs with exhaustive literals, so a field added
/// here breaks the wrapper's build with `error[E0063]: missing field`. The fix
/// is for the wrapper to spread `..Default::default()`, which it can only do if
/// the struct implements `Default` — so "options types implement `Default`" is
/// a cross-repo contract, not a style preference.
///
/// This is an **inventory** check, not a list: it discovers every `struct` /
/// `enum` named `*Options` under `src/` and `crates/`, so a newly added one
/// that forgets the trait fails here without anyone remembering to register it.
/// The scan is pinned by `options_scan_flags_a_new_struct_without_default`
/// below, which feeds it exactly that mutation.
#[test]
fn every_public_options_type_implements_default() {
    let root = repo_root();
    let mut sources = Vec::new();
    rust_sources_skipping_target(&root.join("src"), &mut sources);
    rust_sources_skipping_target(&root.join("crates"), &mut sources);
    assert!(
        !sources.is_empty(),
        "found no sources under src/ or crates/"
    );

    let mut decls: Vec<(PathBuf, OptionsDecl)> = Vec::new();
    let mut manual_impls: Vec<String> = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("source file is valid UTF-8");
        manual_impls.extend(scan_manual_default_impls(&src));
        for decl in scan_options_decls(&src) {
            decls.push((path.clone(), decl));
        }
    }

    // Non-degeneracy: a scan that silently found nothing, or that classified
    // every type the same way, would pass vacuously. Both spellings of the
    // convention must be present and correctly told apart — `SimulateOptions`
    // derives `Default`, `FitOptions` implements it by hand.
    let found = |name: &str| decls.iter().any(|(_, d)| d.name == name);
    for anchor in [
        "FitOptions",
        "SaveFitOptions",
        "OdeSolverOptions",
        "SimulateOptions",
        "SimulateUncertaintyOptions",
        "AdaptiveSimulateOptions",
        "BootstrapOptions",
        "GamOptions",
        "RunOptions",
    ] {
        assert!(found(anchor), "the scan did not find `{anchor}`");
    }
    assert!(
        decls
            .iter()
            .any(|(_, d)| d.name == "SimulateOptions" && d.derives_default),
        "`SimulateOptions` derives `Default`; the derive arm of the scan is broken"
    );
    assert!(
        decls
            .iter()
            .any(|(_, d)| d.name == "FitOptions" && !d.derives_default),
        "`FitOptions` implements `Default` by hand, not by derive; \
         the scan is reading a derive that is not there"
    );
    assert!(
        manual_impls.iter().any(|n| n == "FitOptions"),
        "the manual-`impl` arm of the scan is broken"
    );

    let offenders: Vec<String> = decls
        .iter()
        .filter(|(_, d)| !d.derives_default && !manual_impls.contains(&d.name))
        .map(|(path, d)| {
            format!(
                "{}:{} ({})",
                path.strip_prefix(&root).unwrap_or(path).display(),
                d.line,
                d.name
            )
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "every `pub *Options` type must implement `Default` so a wrapper can \
         build it with `..Default::default()` and survive an added field \
         (#529). Add `Default` to the derive list, or write an `impl Default` \
         with meaningful values. Missing at: {offenders:?}"
    );
}

/// The mutation the A4 guard exists to catch, run against the scan itself:
/// introducing a `pub *Options` type with no `Default` must be reported.
///
/// This is the check a hand-written `assert_default::<T>()` list cannot make —
/// adding a type does not change a hard-coded list, so such a test stays green.
/// Here the offender is the *only* thing that differs between the two fixtures,
/// and the guard's verdict has to flip with it.
#[test]
fn options_scan_flags_a_new_struct_without_default() {
    const FIXTURE: &str = r#"
/// Derives `Default` in prose only — this documentation must not rescue the
/// type below, which is the point of dropping doc comments from the scan.
#[derive(Debug, Clone)]
pub struct ForgottenOptions {
    pub a: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DerivedOptions {
    pub b: usize,
}

#[derive(Debug, Clone)]
pub struct ManualOptions {
    pub c: usize,
}

impl Default for ManualOptions {
    fn default() -> Self {
        Self { c: 7 }
    }
}

#[derive(
    Debug,
    Clone,
    Default,
)]
pub enum SplitDeriveOptions {
    #[default]
    One,
}

/// Not an options type, so not this guard's business.
#[derive(Debug)]
pub struct Something {
    pub d: usize,
}
"#;

    let decls = scan_options_decls(FIXTURE);
    let names: Vec<&str> = decls.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "ForgottenOptions",
            "DerivedOptions",
            "ManualOptions",
            "SplitDeriveOptions"
        ],
        "the scan must find every `pub *Options` declaration and nothing else"
    );

    let manual = scan_manual_default_impls(FIXTURE);
    assert_eq!(manual, vec!["ManualOptions".to_string()]);

    let flagged: Vec<&str> = decls
        .iter()
        .filter(|d| !d.derives_default && !manual.contains(&d.name))
        .map(|d| d.name.as_str())
        .collect();
    assert_eq!(
        flagged,
        vec!["ForgottenOptions"],
        "exactly the type that forgot `Default` must be reported — a prose \
         mention of the trait must not satisfy the guard, and a multi-line \
         derive must not be missed"
    );

    // The straddle: the *same* fixture with the derive fixed must come back
    // clean, so the guard's verdict tracks the code rather than always firing.
    let fixed = FIXTURE.replace(
        "#[derive(Debug, Clone)]\npub struct ForgottenOptions",
        "#[derive(Debug, Clone, Default)]\npub struct ForgottenOptions",
    );
    assert_ne!(
        fixed, FIXTURE,
        "the mutation must actually change the source"
    );
    let manual_fixed = scan_manual_default_impls(&fixed);
    let still_flagged: Vec<String> = scan_options_decls(&fixed)
        .into_iter()
        .filter(|d| !d.derives_default && !manual_fixed.contains(&d.name))
        .map(|d| d.name)
        .collect();
    assert!(
        still_flagged.is_empty(),
        "adding the derive must clear the finding, got {still_flagged:?}"
    );
}

/// `Default` must be matched as a derive entry, not as a substring, and a
/// `derive(...)` must not be read out of an unrelated attribute.
#[test]
fn derive_list_matches_default_exactly() {
    assert!(derive_list_names_default("#[derive(Debug, Default)]"));
    assert!(derive_list_names_default(
        "#[derive(Debug,\n Default,\n Clone)]"
    ));
    assert!(!derive_list_names_default("#[derive(Debug, Clone)]"));
    // A trait whose name merely starts with `Default`.
    assert!(!derive_list_names_default("#[derive(DefaultDisplay)]"));
    // Nested parens inside an earlier attribute must not end the scan early.
    assert!(derive_list_names_default(
        "#[derive(Debug)]\n#[serde(rename_all(x))]\n#[derive(Default)]"
    ));
    assert!(!derive_list_names_default("#[non_exhaustive]"));
}
