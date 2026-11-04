//! Which code is **test code** — the only code these rules look at.
//!
//! The defect the rules hunt is a fold *inside an assertion*. The ~450 `max` /
//! `min` folds in production code are, almost all of them, correct (a running
//! maximum of step sizes, a tolerance floor), and linting them would mean a
//! large baseline plus suppression comments everywhere — so the scope is the
//! test tree, and it is decided here, once, for every rule.
//!
//! A file is test code **as a whole** when any of these holds:
//!
//! - it lives under a crate's `tests/` directory (`tests/**`, `crates/*/tests/**`);
//! - its name ends in `_tests.rs`, or a directory on its path inside `src/` is
//!   named `tests` (`src/api/tests/*.rs`) — the repo's sibling-test-file
//!   convention, see AGENTS.md "Tier 1";
//! - it is the target of an out-of-line `mod` declaration that is itself test
//!   code — `#[cfg(test)] mod test_helpers;`, or any `mod x;` inside a test file.
//!   That last rule is resolved by following the declarations, so a test-only
//!   helper file is covered whatever it is called.
//!
//! Inside any other file, only **items** that are test code are linted: an item
//! carrying `#[test]` (or `#[<path>::test]`), or a `#[cfg(...)]` that *requires*
//! `test` — `cfg(test)`, `cfg(all(test, …))`, but not `cfg(any(test, …))`,
//! which also compiles into a production build.

use std::path::{Path, PathBuf};

use syn::punctuated::Punctuated;
use syn::{Attribute, Item, Meta, Token};

/// One source file and whether it is test code from its first line.
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// Repo-relative, `/`-separated — the form findings and baseline keys use.
    pub rel: String,
    pub abs: PathBuf,
    pub whole_test: bool,
}

/// Does `#[cfg(<meta>)]` exclude every non-test build?
pub fn cfg_meta_requires_test(meta: &Meta) -> bool {
    match meta {
        Meta::Path(p) => p.is_ident("test"),
        Meta::NameValue(_) => false,
        Meta::List(list) => {
            let Ok(nested) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                return false;
            };
            if list.path.is_ident("all") {
                nested.iter().any(cfg_meta_requires_test)
            } else if list.path.is_ident("any") {
                !nested.is_empty() && nested.iter().all(cfg_meta_requires_test)
            } else {
                // `not(...)` and anything unknown: not test-only.
                false
            }
        }
    }
}

/// Is an item with these attributes test code by its own attributes?
pub fn attrs_mark_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        let path = a.path();
        if path.is_ident("cfg") {
            return a
                .parse_args::<Meta>()
                .map(|m| cfg_meta_requires_test(&m))
                .unwrap_or(false);
        }
        // `#[test]`, `#[tokio::test]`, `#[rstest::test]`, …
        path.segments.last().is_some_and(|s| s.ident == "test")
    })
}

/// Is this path test code by name/location alone?
pub fn path_is_test_by_name(rel: &str) -> bool {
    let parts: Vec<&str> = rel.split('/').collect();
    let file = parts.last().copied().unwrap_or_default();
    if file.ends_with("_tests.rs") {
        return true;
    }
    // Position of the crate's `src`/`tests` directory: `tests/..` at the repo
    // root, or `crates/<name>/tests/..`.
    let crate_dir_len = if parts.first() == Some(&"crates") {
        2
    } else {
        0
    };
    match parts.get(crate_dir_len) {
        Some(&"tests") => true,
        Some(&"src") => parts[crate_dir_len + 1..parts.len() - 1].contains(&"tests"),
        _ => false,
    }
}

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            if e.file_name() == "target" {
                continue;
            }
            walk_rs(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    Ok(())
}

fn rel_of(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The directory a file's out-of-line child modules resolve against.
fn module_dir(file: &Path, rel: &str) -> PathBuf {
    let parent = file.parent().unwrap_or(Path::new("")).to_path_buf();
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let parts: Vec<&str> = rel.split('/').collect();
    let crate_dir_len = if parts.first() == Some(&"crates") {
        2
    } else {
        0
    };
    let in_crate = &parts[crate_dir_len.min(parts.len())..];
    // Crate roots behave like `mod.rs`: `src/lib.rs`, `src/main.rs`,
    // `src/bin/x.rs`, and every top-level `tests/x.rs` integration target.
    let is_root = matches!(name, "mod.rs" | "lib.rs" | "main.rs")
        || (in_crate.len() == 2 && in_crate[0] == "tests")
        || (in_crate.len() == 3 && in_crate[0] == "src" && in_crate[1] == "bin");
    if is_root {
        parent
    } else {
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        parent.join(stem)
    }
}

fn path_attr(attrs: &[Attribute]) -> Option<String> {
    attrs.iter().find_map(|a| match &a.meta {
        Meta::NameValue(nv) if nv.path.is_ident("path") => match &nv.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) => Some(s.value()),
            _ => None,
        },
        _ => None,
    })
}

/// Out-of-line `mod x;` declarations in `items`, resolved to files, with
/// whether each declaration sits in test code.
fn child_modules(
    items: &[Item],
    file_dir: &Path,
    mod_dir: &Path,
    in_test: bool,
    out: &mut Vec<(PathBuf, bool)>,
) {
    for item in items {
        let Item::Mod(m) = item else { continue };
        let test = in_test || attrs_mark_test(&m.attrs);
        let name = m.ident.to_string();
        match &m.content {
            Some((_, inner)) => {
                let dir = match path_attr(&m.attrs) {
                    Some(p) => mod_dir.join(p),
                    None => mod_dir.join(&name),
                };
                child_modules(inner, &dir, &dir, test, out);
            }
            None => {
                let target = match path_attr(&m.attrs) {
                    Some(p) => file_dir.join(p),
                    None => {
                        let flat = mod_dir.join(format!("{name}.rs"));
                        if flat.is_file() {
                            flat
                        } else {
                            mod_dir.join(&name).join("mod.rs")
                        }
                    }
                };
                out.push((target, test));
            }
        }
    }
}

/// Every `.rs` file under `src/`, `tests/`, `crates/*/src/` and
/// `crates/*/tests/`, as `(repo-relative, absolute)` pairs in a stable order.
pub fn source_paths(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut dirs = vec![root.join("src"), root.join("tests")];
    let crates = root.join("crates");
    if crates.is_dir() {
        let mut members: Vec<_> = std::fs::read_dir(&crates)
            .map_err(|e| format!("{}: {e}", crates.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        members.sort();
        for m in members {
            dirs.push(m.join("src"));
            dirs.push(m.join("tests"));
        }
    }
    let mut paths = Vec::new();
    for d in &dirs {
        walk_rs(d, &mut paths).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    Ok(paths
        .into_iter()
        .map(|abs| (rel_of(root, &abs), normalize(&abs)))
        .collect())
}

/// The files this file's out-of-line `mod x;` declarations resolve to, each
/// with whether its declaration sits in test code.
pub fn child_edges(abs: &Path, rel: &str, ast: &syn::File) -> Vec<(PathBuf, bool)> {
    let file_dir = abs.parent().unwrap_or(Path::new("")).to_path_buf();
    let mod_dir = module_dir(abs, rel);
    let mut out = Vec::new();
    child_modules(&ast.items, &file_dir, &mod_dir, false, &mut out);
    out.into_iter().map(|(p, t)| (normalize(&p), t)).collect()
}

/// Propagate whole-file test status along `mod x;` edges to a fixed point: a
/// file is test code if it is by name, if a test-code declaration names it, or
/// if any declaration in a test file names it.
pub fn propagate(mut test: Vec<bool>, edges: &[Vec<(usize, bool)>]) -> Vec<bool> {
    loop {
        let mut changed = false;
        for (from, children) in edges.iter().enumerate() {
            for &(to, decl_test) in children {
                if (test[from] || decl_test) && !test[to] {
                    test[to] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            return test;
        }
    }
}

/// Collapse `a/./b` and `a/x/../b` without touching the filesystem, so a
/// `#[path = "../foo.rs"]` target matches the walked path.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(src: &str) -> Meta {
        syn::parse_str::<Meta>(src).unwrap()
    }

    #[test]
    fn cfg_requires_test_only_when_every_build_that_compiles_it_is_a_test_build() {
        assert!(cfg_meta_requires_test(&meta("test")));
        assert!(cfg_meta_requires_test(&meta("all(test, feature = \"nn\")")));
        assert!(cfg_meta_requires_test(&meta("any(test, all(test, unix))")));
        // `any(test, feature)` also compiles into a production build with the feature on.
        assert!(!cfg_meta_requires_test(&meta(
            "any(test, feature = \"nn\")"
        )));
        assert!(!cfg_meta_requires_test(&meta("not(test)")));
        assert!(!cfg_meta_requires_test(&meta("feature = \"test\"")));
        assert!(!cfg_meta_requires_test(&meta("any()")));
    }

    #[test]
    fn test_attribute_by_last_segment() {
        let f: syn::ItemFn = syn::parse_str("#[tokio::test] fn f() {}").unwrap();
        assert!(attrs_mark_test(&f.attrs));
        let f: syn::ItemFn = syn::parse_str("#[inline] fn f() {}").unwrap();
        assert!(!attrs_mark_test(&f.attrs));
        let f: syn::ItemFn = syn::parse_str("#[cfg(test)] fn f() {}").unwrap();
        assert!(attrs_mark_test(&f.attrs));
    }

    #[test]
    fn file_names_that_are_test_code() {
        assert!(path_is_test_by_name("tests/foo.rs"));
        assert!(path_is_test_by_name("tests/common/mod.rs"));
        assert!(path_is_test_by_name("crates/ferx-cli/tests/cli.rs"));
        assert!(path_is_test_by_name("src/estimation/saem_tests.rs"));
        assert!(path_is_test_by_name("src/api/tests/fit.rs"));
        assert!(!path_is_test_by_name("src/api/fit.rs"));
        assert!(!path_is_test_by_name("crates/ferx-tools/src/lib.rs"));
        // A file merely *named* tests.rs inside src is resolved via its `mod`
        // declaration, not by name.
        assert!(!path_is_test_by_name("src/tests.rs"));
    }

    #[test]
    fn module_dir_follows_rustc() {
        let d = module_dir(Path::new("/r/src/lib.rs"), "src/lib.rs");
        assert_eq!(d, Path::new("/r/src"));
        let d = module_dir(Path::new("/r/src/types.rs"), "src/types.rs");
        assert_eq!(d, Path::new("/r/src/types"));
        let d = module_dir(Path::new("/r/tests/foo.rs"), "tests/foo.rs");
        assert_eq!(d, Path::new("/r/tests"));
        let d = module_dir(
            Path::new("/r/crates/c/src/bin/x.rs"),
            "crates/c/src/bin/x.rs",
        );
        assert_eq!(d, Path::new("/r/crates/c/src/bin"));
    }

    /// The propagation is what reaches a test-only helper whose name says
    /// nothing: `#[cfg(test)] mod helpers;` makes `helpers.rs` test code, and a
    /// plain `mod deeper;` inside it inherits that.
    #[test]
    fn cfg_test_mod_declarations_propagate_to_their_files() {
        let root = std::env::temp_dir().join(format!("patterns-lint-scope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src/a/helpers")).unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "mod a;\n#[cfg(test)]\n#[path = \"odd_name.rs\"]\nmod t;\n",
        )
        .unwrap();
        std::fs::write(root.join("src/odd_name.rs"), "").unwrap();
        std::fs::write(
            root.join("src/a.rs"),
            "#[cfg(test)] mod helpers;\nmod prod;\n",
        )
        .unwrap();
        std::fs::write(root.join("src/a/prod.rs"), "").unwrap();
        std::fs::write(root.join("src/a/helpers.rs"), "mod deeper;\n").unwrap();
        std::fs::write(root.join("src/a/helpers/deeper.rs"), "").unwrap();

        let files = crate::scan(&root).unwrap().files;
        let test_of = |rel: &str| {
            files
                .iter()
                .find(|f| f.rel == rel)
                .unwrap_or_else(|| panic!("{rel} not discovered"))
                .whole_test
        };
        assert!(!test_of("src/lib.rs"));
        assert!(!test_of("src/a.rs"));
        assert!(!test_of("src/a/prod.rs"));
        assert!(test_of("src/odd_name.rs"));
        assert!(test_of("src/a/helpers.rs"));
        assert!(test_of("src/a/helpers/deeper.rs"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unparseable_file_is_an_error_not_a_skip() {
        let root = std::env::temp_dir().join(format!("patterns-lint-parse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(root.join("tests/bad.rs"), "fn f( {\n").unwrap();
        let err = crate::scan(&root).err().expect("parse error must surface");
        assert!(err.contains("tests/bad.rs:"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
