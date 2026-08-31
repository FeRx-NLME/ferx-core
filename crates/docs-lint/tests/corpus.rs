//! The corpus check: `docs/` itself must pass all four rules.
//!
//! This is what the `docs-lint` CI job runs, so a PR that adds an oversized
//! section, skips a heading level, repeats an id, or writes a dead anchor fails
//! before review.

use docs_lint::baseline::{self, Baseline};
use docs_lint::{default_baseline_path, lint_tree, repo_root, Config};

#[test]
fn docs_tree_passes_all_four_rules() {
    let root = repo_root();
    let docs = root.join("docs");
    assert!(docs.is_dir(), "no docs/ at {}", docs.display());

    let violations = lint_tree(&docs, &root, &Config::default()).expect("read docs/");
    let baseline = Baseline::load(&default_baseline_path(&root)).expect("parse baseline");
    let filtered = baseline::apply(&baseline, violations);

    let mut report = String::new();
    for v in &filtered.failing {
        report.push_str(&format!(
            "{}:{}: {}: {}\n",
            v.file,
            v.line,
            v.rule.as_str(),
            v.message
        ));
    }
    for key in &filtered.stale {
        report.push_str(&format!(
            "{key}: R1: no longer over the limit — remove this line from docs/.lint-baseline\n"
        ));
    }
    assert!(
        report.is_empty(),
        "docs-lint found {} violation(s):\n{report}\nRun `cargo run -p docs-lint` for the same report.",
        filtered.failing.len() + filtered.stale.len()
    );
}

/// Every baseline entry names a section that still exists, so the file cannot
/// rot into a list of keys nothing resolves to.
#[test]
fn baseline_entries_all_name_a_live_section() {
    let root = repo_root();
    let baseline = Baseline::load(&default_baseline_path(&root)).expect("parse baseline");
    if baseline.entries.is_empty() {
        return;
    }
    // A very large limit yields every chunk in the corpus, keyed the same way.
    let all = lint_tree(
        &root.join("docs"),
        &root,
        &Config {
            max_chars: usize::MAX,
        },
    )
    .expect("read docs/");
    // With no size violations possible, anything left is another rule; the keys
    // we need come from a zero-limit run instead.
    assert!(all.iter().all(|v| v.chunk.is_none()));

    let every_chunk = lint_tree(&root.join("docs"), &root, &Config { max_chars: 0 })
        .expect("read docs/")
        .into_iter()
        .filter_map(|v| v.chunk.map(|c| c.key))
        .collect::<Vec<_>>();

    for key in baseline.entries.keys() {
        assert!(
            every_chunk.contains(key),
            "baseline entry `{key}` names a section that no longer exists — \
             remove or re-key it (headings renamed?)"
        );
    }
}

/// Opt-in cross-check of the id algorithm against a real Quarto render.
///
/// `docs/_site/` is generated and git-ignored, so this cannot run in CI and
/// must not fail a developer with a stale render. Refresh with `quarto render
/// docs`, then:
///
/// ```text
/// DOCS_LINT_VERIFY_RENDER=1 cargo test -p docs-lint -- --ignored
/// ```
#[test]
#[ignore = "needs a fresh `quarto render docs`; set DOCS_LINT_VERIFY_RENDER=1"]
fn computed_ids_match_the_rendered_site() {
    if std::env::var("DOCS_LINT_VERIFY_RENDER").is_err() {
        eprintln!("skipped: set DOCS_LINT_VERIFY_RENDER=1 with a fresh render");
        return;
    }
    let root = repo_root();
    let site = root.join("docs/_site");
    assert!(site.is_dir(), "no render at {}", site.display());

    let mut checked = 0usize;
    let mut stale = 0usize;
    let mut mismatches = Vec::new();
    for path in docs_lint::parse::collect_qmd(&root.join("docs")).expect("read docs/") {
        let doc = docs_lint::parse::parse_file(&path, &root).expect("parse");
        let html = site.join(doc.rel.trim_start_matches("docs/").replace(".qmd", ".html"));
        let Ok(rendered) = std::fs::read_to_string(&html) else {
            continue;
        };
        // A page edited since the render says nothing about the algorithm.
        if let (Ok(src), Ok(out)) = (
            std::fs::metadata(&path).and_then(|m| m.modified()),
            std::fs::metadata(&html).and_then(|m| m.modified()),
        ) {
            if src > out {
                stale += 1;
                continue;
            }
        }
        let ids = rendered
            .match_indices("data-anchor-id=\"")
            .map(|(i, m)| {
                let rest = &rendered[i + m.len()..];
                rest[..rest.find('"').unwrap_or(0)].to_string()
            })
            .collect::<Vec<_>>();
        if ids.is_empty() {
            continue;
        }
        for h in &doc.headings {
            // The page title (h1) is not rendered as an anchored heading.
            if h.level == 1 {
                continue;
            }
            if !ids.contains(&h.id) {
                mismatches.push(format!(
                    "{}:{}: computed `{}` for \"{}\" is not in {}",
                    doc.rel,
                    h.line,
                    h.id,
                    h.text,
                    html.display()
                ));
            }
            checked += 1;
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} id(s) disagree with the render:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert!(
        checked > 0,
        "no ids checked ({stale} page(s) skipped as newer than their render) — re-render first"
    );
}
