//! `ModelText::parse` → `render` is the identity, over every checked-in model
//! (#1176).
//!
//! This one test is what makes the text surgery in `ferx-core::edit` safe to
//! use. Every [`ModelEdit`](ferx_core::edit::ModelEdit) rewrites a handful of
//! lines and leaves the rest of the file alone — but "leaves the rest alone"
//! is only a guarantee if the representation is lossless to begin with. If a
//! round-trip could drop a comment, re-indent a block or normalise a line
//! ending, then every generated candidate would carry an invisible rewrite of
//! everything the search did *not* mean to change, and a diff between parent
//! and child would stop being readable.
//!
//! `examples/**` is the corpus rather than a handful of fixtures because it is
//! the widest spread of real syntax the repo has — ODE blocks, `if`/`else`,
//! multi-line `block_omega`, `[covariate_nn]`, adaptive-dosing blocks, models
//! with no `[fit_options]` at all — and it grows on its own as features land.

use std::fs;
use std::path::{Path, PathBuf};

use ferx_core::edit::ModelText;

/// The shipped multi-line `block_omega` is one declaration, not six lines.
///
/// `prepare_frem()` emits the block form broken over a line per row, and the
/// parser rejoins it before any declaration regex runs. An editor that reads
/// physical lines sees no `block_omega` at all here: it would rewrite the
/// `block_omega (…) = [` line and leave the five rows of triangle below it.
#[test]
fn the_shipped_multi_line_block_omega_reads_as_one_declaration() {
    let src = fs::read_to_string("examples/warfarin_frem.ferx").expect("the example must exist");
    let text = ModelText::parse(&src).expect("the example must parse");
    let blocks: Vec<String> = text
        .block_lines("parameters")
        .into_iter()
        .filter(|l| l.starts_with("block_omega"))
        .collect();
    assert_eq!(blocks.len(), 1, "{blocks:?}");
    let block = &blocks[0];
    for eta in ["ETA_CL", "ETA_V", "ETA_KA", "ETA_WT_FREM", "ETA_AGE_FREM"] {
        assert!(block.contains(eta), "`{eta}` missing from `{block}`");
    }
    // The whole triangle came with it — the last row included.
    assert!(block.contains("99.38") && block.ends_with(']'), "{block}");
}

/// Every `.ferx` file under `examples/`, recursively.
fn example_models() -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect(Path::new("examples"), &mut out);
    out.sort();
    out
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "ferx") {
            out.push(path);
        }
    }
}

#[test]
fn every_example_model_round_trips_byte_for_byte() {
    let models = example_models();
    assert!(
        models.len() > 50,
        "the example corpus looks truncated ({} files) — this test is only \
         meaningful over the whole of examples/**",
        models.len()
    );
    for path in &models {
        let src = fs::read_to_string(path).expect("example model must be readable");
        let text = ModelText::parse(&src)
            .unwrap_or_else(|e| panic!("{} must parse as ModelText: {e}", path.display()));
        let rendered = text.render();
        assert_eq!(
            rendered,
            src,
            "{} did not round-trip; first difference at byte {}",
            path.display(),
            rendered
                .bytes()
                .zip(src.bytes())
                .position(|(a, b)| a != b)
                .map(|i| i.to_string())
                .unwrap_or_else(|| "the end (lengths differ)".into())
        );
    }
}

#[test]
fn dropping_one_line_changes_every_example_model_s_identity() {
    // The other half of the contract: stable under formatting, *sensitive* to
    // content. A hash that missed a change would silently collapse two
    // branches of a search into one cached fit — the expensive direction of
    // the error, since the second candidate never gets fitted at all.
    //
    // Deleting the first code line is the smallest semantic edit that is legal
    // on every file in the corpus, whatever blocks it happens to use.
    for path in example_models() {
        let src = fs::read_to_string(&path).expect("example model must be readable");
        let Some(victim) = src
            .lines()
            .position(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        else {
            continue;
        };
        let shortened: String = src
            .lines()
            .enumerate()
            .filter(|(i, _)| *i != victim)
            .map(|(_, l)| format!("{l}\n"))
            .collect();
        let before = ModelText::parse(&src).unwrap().canonical_hash();
        let after = match ModelText::parse(&shortened) {
            Ok(t) => t.canonical_hash(),
            // Removing the only block header leaves nothing addressable; that
            // is a rejection, not a collision.
            Err(_) => continue,
        };
        assert_ne!(
            before,
            after,
            "{} kept its identity after a line was deleted",
            path.display()
        );
    }
}

#[test]
fn two_example_models_hash_alike_only_when_they_are_the_same_model() {
    // `examples/one_cpt_infusion.ferx` and `examples/dose_rate.ferx` are the
    // same model documented two different ways, and they *should* collide —
    // that is the cache hit the hash exists to produce. What must not happen
    // is two files with different code hashing alike.
    let mut seen: Vec<(PathBuf, [u8; 32], String)> = Vec::new();
    for path in example_models() {
        let src = fs::read_to_string(&path).expect("example model must be readable");
        let text = ModelText::parse(&src).unwrap();
        let (hash, form) = (text.canonical_hash(), text.canonical_form());
        if let Some((other, _, other_form)) = seen.iter().find(|(_, h, _)| *h == hash) {
            assert_eq!(
                &form,
                other_form,
                "{} and {} hash alike but are different models",
                path.display(),
                other.display()
            );
        }
        seen.push((path, hash, form));
    }
}

#[test]
fn canonical_hash_survives_a_reflow_of_every_example_model() {
    // The mechanical version of the unit test's hand-written reflow: doubling
    // every run of spaces, adding a trailing comment to each line and blank
    // lines between them must not change any model's identity.
    for path in example_models() {
        let src = fs::read_to_string(&path).expect("example model must be readable");
        let reflowed: String = src
            .lines()
            .map(|l| format!("{}   # noise\n\n", l.replace("  ", "    ")))
            .collect();
        // A `#` inside a line would already have started a comment, so the
        // appended one is inert; a line that *is* a comment stays one.
        let original = ModelText::parse(&src).unwrap().canonical_hash();
        let after = ModelText::parse(&reflowed).unwrap().canonical_hash();
        assert_eq!(
            original,
            after,
            "{} changed identity under a pure reflow",
            path.display()
        );
    }
}
