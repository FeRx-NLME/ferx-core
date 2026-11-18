//! Seed derivation for independent random streams (#3021).
//!
//! Every per-subject, per-chain or per-start stream is `seed_from_u64` of a value derived
//! from the user's seed and the stream's coordinates. Deriving it as `seed + i` made
//! stream `(s, i)` and `(s + 1, i - 1)` the *same* stream, so runs under consecutive seeds
//! shared all but one subject's draws (IMP seeds 1..5 shared 499 of 500 subject streams).
//! [`stream_seed`] hashes the coordinates instead, so two coordinate tuples collide only by
//! 64-bit hash chance. It is a pure function of its arguments: results stay independent of
//! the rayon schedule.

/// SplitMix64 finaliser (Steele, Lea & Flood 2014).
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The RNG seed of the stream at `words` (subject index, iteration, chain, purpose tag, …)
/// under the user's `seed`. Order-sensitive; the seed is hashed before any word is folded
/// in, so `(s, [i])` and `(s ± d, [i ∓ d])` do not coincide.
pub(crate) fn stream_seed(seed: u64, words: &[u64]) -> u64 {
    words.iter().fold(splitmix64(seed), |acc, &w| {
        splitmix64(acc ^ splitmix64(w ^ 0xA076_1D64_78BD_642F))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The defect: additive derivation maps `(s, i)` and `(s + 1, i - 1)` to one stream.
    #[test]
    fn consecutive_seeds_share_no_stream() {
        let mut seen = HashSet::new();
        for s in 0..8u64 {
            for i in 0..500u64 {
                assert!(
                    seen.insert(stream_seed(s, &[i])),
                    "stream ({s},{i}) repeats"
                );
            }
        }
        for s in 1..8u64 {
            for i in 1..500u64 {
                assert_ne!(stream_seed(s, &[i]), stream_seed(s - 1, &[i + 1]));
                assert_ne!(stream_seed(s, &[i]), stream_seed(s + 1, &[i - 1]));
            }
        }
    }

    /// Every site derives its stream seed through [`stream_seed`]: no non-test source line
    /// seeds from `seed ± coordinate` arithmetic (the #3021 shape — `seed.wrapping_add(i as
    /// u64)`, `seed + i`), which no unit test of one estimator would catch for the others.
    #[test]
    fn no_site_derives_a_stream_seed_by_addition() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        assert!(files.len() > 100, "scan found only {} files", files.len());
        let mut bad = Vec::new();
        for f in files {
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            if name == "rng.rs" {
                continue;
            }
            // Join method-chain continuations (`seed\n    .wrapping_add(i as u64)`) and
            // arguments rustfmt puts on their own lines (`.wrapping_add(\n    i as u64,\n)`)
            // into one logical line, so a derivation split over several lines is scanned whole.
            let mut logical: Vec<(usize, String)> = Vec::new();
            for (n, line) in std::fs::read_to_string(&f).unwrap().lines().enumerate() {
                let t = line.trim();
                if t.starts_with("//") {
                    continue;
                }
                match logical.last_mut() {
                    Some((_, prev))
                        if t.starts_with('.') || t.starts_with(')') || prev.ends_with('(') =>
                    {
                        prev.push_str(t)
                    }
                    _ => logical.push((n + 1, t.to_string())),
                }
            }
            for (n, line) in logical {
                // A seed multiplied by an odd constant before the coordinates are added
                // (VI's `crn_eps`) is a multiplicative mix, not `seed + i`: `(s, i)` and
                // `(s + 1, i')` meet only at a 64-bit-sized `i' - i`.
                if !line.contains("seed") || line.contains("seed.wrapping_mul(") {
                    continue;
                }
                // ` as u64` with no closing paren also catches a scaled coordinate
                // (`seed.wrapping_add(chain as u64 * K)`, Bayes' pre-#3021 shape).
                let additive = line.contains(".wrapping_add(") && line.contains(" as u64");
                let plus = line.contains("seed + ") && line.contains(" as u64");
                if additive || plus {
                    bad.push(format!("{}:{}: {}", f.display(), n, line));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "additive stream-seed derivation:\n{}",
            bad.join("\n")
        );
        // Every file that seeds per-subject / per-chain / per-start streams must also name
        // the helper.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for f in [
            "estimation/importance_sampling.rs",
            "estimation/impmap.rs",
            "estimation/saem.rs",
            "estimation/saem_conddist.rs",
            "estimation/bayes.rs",
            "stats/npde.rs",
            "api/fit.rs",
        ] {
            let text = std::fs::read_to_string(src.join(f)).unwrap();
            assert!(
                text.contains("stream_seed("),
                "{f} no longer derives its streams via stream_seed"
            );
        }
    }

    #[test]
    fn words_are_order_and_arity_sensitive() {
        assert_ne!(stream_seed(7, &[1, 2]), stream_seed(7, &[2, 1]));
        assert_ne!(stream_seed(7, &[1]), stream_seed(7, &[1, 0]));
        assert_ne!(stream_seed(7, &[0]), stream_seed(7, &[]));
        assert_eq!(stream_seed(7, &[3, 4]), stream_seed(7, &[3, 4]));
    }
}
