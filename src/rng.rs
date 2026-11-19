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
    words
        .iter()
        .fold(splitmix64(seed), |acc, &w| splitmix64(acc ^ splitmix64(w ^ 0xA076_1D64_78BD_642F)))
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
                assert!(seen.insert(stream_seed(s, &[i])), "stream ({s},{i}) repeats");
            }
        }
        for s in 1..8u64 {
            for i in 1..500u64 {
                assert_ne!(stream_seed(s, &[i]), stream_seed(s - 1, &[i + 1]));
                assert_ne!(stream_seed(s, &[i]), stream_seed(s + 1, &[i - 1]));
            }
        }
    }

    /// Every site derives its stream seed through [`stream_seed`]: no source line under
    /// `src/` or `crates/` (test code included) seeds from `seed ± coordinate` or
    /// `seed ^ coordinate` arithmetic (the #3021 shape: `seed.wrapping_add(i as u64)`,
    /// `seed + i`), which no unit test of one estimator would catch for the others. A
    /// second, name-independent check flags any `seed_from_u64(…)` argument that mixes by
    /// `+`, `-`, `^` or `wrapping_add/sub` without a hash. The exemptions are listed with
    /// their reason, and each must still match, so a stale one fails too.
    #[test]
    fn no_site_derives_a_stream_seed_by_addition() {
        use regex::Regex;
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
        /// `(path suffix, text on the logical line, why it is not the #3021 defect)`.
        const EXEMPT: &[(&str, &str, &str)] = &[
            (
                "estimation/importance_sampling.rs",
                "splitmix64(seed ^ IS_ADAPT_SEED_TAG ^ pass as u64)",
                "pilot_seed: `seed` is already a stream_seed hash, and the result is finalised",
            ),
            (
                "estimation/importance_sampling.rs",
                "seed.wrapping_add(0x534F_424F_4C00_0000u64)",
                "test oracle: a copy of the pre-#1965 shift code, one stream, no coordinate",
            ),
            (
                "estimation/vi/elbo.rs",
                "let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)",
                "crn_eps: odd-multiplier mix, collides only at a 64-bit-sized coordinate gap",
            ),
            (
                "api/vpc_tte.rs",
                "let mut z = seed ^ (sim as u64).wrapping_mul(",
                "censoring_uniform: XOR of odd-multiplied coordinates, then a finaliser",
            ),
            (
                "stats/npde_tte.rs",
                "combine_seed(seed ^ IMPUTE_SALT, index as u64)",
                "constant salt; the coordinate goes through combine_seed, a hash",
            ),
            (
                "stats/npde_tte.rs",
                "combine_seed(seed ^ BATCH_SALT, batch as u64)",
                "constant salt; the coordinate goes through combine_seed, a hash",
            ),
            (
                "bootstrap/resample.rs",
                "seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)",
                "replicate_seed: collides only for seeds differing by a multiple of the constant",
            ),
        ];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        walk(&root.join("src"), &mut files);
        walk(&root.join("crates"), &mut files);
        assert!(files.len() > 100, "scan found only {} files", files.len());
        let coord = Regex::new(r"\bas u64\b|u64::from\(").unwrap();
        let add_sub = Regex::new(r"\.wrapping_(add|sub)\(").unwrap();
        let op_seed = Regex::new(r"seed\w*\s*[-+^]\s|[-+^]\s*\w*seed\b").unwrap();
        let sfu = Regex::new(r"seed_from_u64\((.*)\)").unwrap();
        let hashed = Regex::new(r"stream_seed|splitmix|combine_seed|hash").unwrap();
        let mixed = Regex::new(r"wrapping_(add|sub)\(|\s[-+^]\s").unwrap();
        let mut bad = Vec::new();
        let mut used = vec![0usize; EXEMPT.len()];
        for f in files {
            if f.file_name().is_some_and(|n| n == "rng.rs") {
                continue;
            }
            let path = f.to_string_lossy().replace('\\', "/");
            // One logical line per statement: join method-chain and operator
            // continuations (`seed\n    .wrapping_add(i as u64)`, `master_seed\n    + i as
            // u64`) and arguments rustfmt puts on their own lines, so a derivation split
            // over several lines is scanned whole.
            let mut logical: Vec<(usize, String)> = Vec::new();
            for (n, line) in std::fs::read_to_string(&f).unwrap().lines().enumerate() {
                let t = line.trim();
                if t.starts_with("//") {
                    continue;
                }
                match logical.last_mut() {
                    Some((_, prev)) if t.starts_with(['.', ')']) => prev.push_str(t),
                    Some((_, prev))
                        if t.starts_with(['+', '-', '^', '*'])
                            || prev.ends_with(['(', '+', '-', '^', '*', '=']) =>
                    {
                        prev.push(' ');
                        prev.push_str(t)
                    }
                    _ => logical.push((n + 1, t.to_string())),
                }
            }
            for (n, line) in logical {
                let named = line.contains("seed")
                    && coord.is_match(&line)
                    && (add_sub.is_match(&line) || op_seed.is_match(&line));
                let unhashed = sfu
                    .captures(&line)
                    .is_some_and(|c| !hashed.is_match(&c[1]) && mixed.is_match(&c[1]));
                if !(named || unhashed) {
                    continue;
                }
                match EXEMPT
                    .iter()
                    .position(|(p, text, _)| path.ends_with(p) && line.contains(text))
                {
                    Some(k) => used[k] += 1,
                    None => bad.push(format!("{path}:{n}: {line}")),
                }
            }
        }
        assert!(
            bad.is_empty(),
            "additive / XOR stream-seed derivation (use rng::stream_seed):\n{}",
            bad.join("\n")
        );
        for ((p, text, why), n) in EXEMPT.iter().zip(&used) {
            assert_eq!(*n, 1, "exemption `{p}` / `{text}` ({why}) matched {n} lines");
        }
        // Every file that seeds per-subject / per-chain / per-start streams must also name
        // the helper.
        let src = root.join("src");
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
