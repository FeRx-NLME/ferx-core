//! Drawing bootstrap replicates: strata, sample sizes, and the seeded RNG.
//!
//! Resampling is always over **whole subjects** — for each ID every record is
//! either in or out of a replicate, never split. That is what makes the
//! non-parametric case bootstrap valid for a hierarchical model, and it is what
//! PsN does.

use std::collections::BTreeMap;

use ferx_core::Population;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// How many subjects each replicate draws.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleSize {
    /// The number of subjects in the original dataset — PsN's default, and the
    /// only case where a stratified proportional split is guaranteed to sum
    /// back to the requested total.
    Original,
    /// One number for the whole replicate. Under stratification the strata are
    /// allocated *in proportion to their size*, rounded to the nearest integer;
    /// PsN documents that the rounded parts need not sum to the request.
    Total(usize),
    /// An explicit count per stratum, PsN's `-sample_size=1001=>12,1002=>24`
    /// syntax. Only meaningful with `stratify_on`.
    PerStratum(BTreeMap<String, usize>),
}

impl SampleSize {
    /// Parse PsN's `-sample_size` argument: either a bare integer or a
    /// comma-separated list of `STRATUM=>COUNT` pairs.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err("empty --sample-size".to_string());
        }
        if !spec.contains("=>") {
            return spec.parse::<usize>().map(SampleSize::Total).map_err(|_| {
                format!("--sample-size: `{spec}` is neither a number nor a `STRATUM=>COUNT` list")
            });
        }
        let mut map = BTreeMap::new();
        for part in spec.split(',') {
            let part = part.trim();
            let (stratum, count) = part.split_once("=>").ok_or_else(|| {
                format!(
                    "--sample-size: `{part}` is not a `STRATUM=>COUNT` pair. Mixing a bare \
                     number with per-stratum counts is not allowed — give one or the other."
                )
            })?;
            let stratum = stratum.trim().to_string();
            let count: usize = count.trim().parse().map_err(|_| {
                format!("--sample-size: `{part}` has a non-numeric count after `=>`")
            })?;
            if map.insert(stratum.clone(), count).is_some() {
                return Err(format!(
                    "--sample-size: stratum `{stratum}` is listed more than once"
                ));
            }
        }
        Ok(SampleSize::PerStratum(map))
    }
}

/// The subjects of the original dataset, grouped for resampling.
///
/// The unstratified case is represented as a single unnamed group rather than as
/// a separate code path, so the draw loop below has one shape.
#[derive(Debug, Clone)]
pub struct Strata {
    /// Stratum label → ordinals into `Population::subjects`, ascending.
    pub groups: BTreeMap<String, Vec<usize>>,
    /// `None` for an unstratified bootstrap.
    pub column: Option<String>,
}

impl Strata {
    /// One group holding every subject, in dataset order.
    pub fn unstratified(n_subjects: usize) -> Self {
        let mut groups = BTreeMap::new();
        groups.insert(String::new(), (0..n_subjects).collect());
        Strata {
            groups,
            column: None,
        }
    }

    /// Group subjects by a per-subject label.
    ///
    /// `labels[i]` is the stratum of `Population::subjects[i]`. Callers get the
    /// labels from the dataset (see [`super::strata_from_csv`]); the
    /// one-value-per-subject check happens there, where the offending record can
    /// still be named.
    pub fn from_labels(labels: &[String], column: &str) -> Self {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, label) in labels.iter().enumerate() {
            groups.entry(label.clone()).or_default().push(i);
        }
        Strata {
            groups,
            column: Some(column.to_string()),
        }
    }

    fn n_subjects(&self) -> usize {
        self.groups.values().map(|g| g.len()).sum()
    }

    /// How many subjects to draw from each stratum, resolving [`SampleSize`].
    ///
    /// Returned in the same key order as `groups` so the draw is deterministic.
    pub fn allocation(&self, size: &SampleSize) -> Result<Vec<(String, usize)>, String> {
        let total_subjects = self.n_subjects();
        match size {
            SampleSize::Original => Ok(self
                .groups
                .iter()
                .map(|(k, g)| (k.clone(), g.len()))
                .collect()),
            SampleSize::Total(m) => {
                // Proportional allocation. With one stratum this is just `m`, so
                // the unstratified path needs no special case.
                if self.groups.len() == 1 {
                    return Ok(vec![(
                        self.groups.keys().next().cloned().unwrap_or_default(),
                        *m,
                    )]);
                }
                Ok(self
                    .groups
                    .iter()
                    .map(|(k, g)| {
                        let share = (*m as f64) * (g.len() as f64) / (total_subjects as f64);
                        (k.clone(), share.round() as usize)
                    })
                    .collect())
            }
            SampleSize::PerStratum(map) => {
                if self.column.is_none() {
                    return Err(
                        "--sample-size was given per-stratum counts but --stratify-on was not set"
                            .to_string(),
                    );
                }
                let unknown: Vec<&str> = map
                    .keys()
                    .filter(|k| !self.groups.contains_key(*k))
                    .map(|s| s.as_str())
                    .collect();
                if !unknown.is_empty() {
                    return Err(format!(
                        "--sample-size names stratum/strata {:?} that do not occur in column \
                         `{}`. Present strata: {:?}",
                        unknown,
                        self.column.as_deref().unwrap_or(""),
                        self.groups.keys().collect::<Vec<_>>()
                    ));
                }
                // A stratum the user did not name contributes nothing, rather
                // than silently falling back to its own size — an unmentioned
                // stratum in an explicit list is far more likely a typo than an
                // intent to include it whole.
                let missing: Vec<&String> = self
                    .groups
                    .keys()
                    .filter(|k| !map.contains_key(*k))
                    .collect();
                if !missing.is_empty() {
                    return Err(format!(
                        "--sample-size lists per-stratum counts but omits {missing:?}. List every \
                         stratum explicitly (use 0 to exclude one)."
                    ));
                }
                Ok(map.iter().map(|(k, v)| (k.clone(), *v)).collect())
            }
        }
    }
}

/// One drawn replicate: ordinals into the original `Population::subjects`.
///
/// Held **ascending**, so a subject drawn twice appears twice in a row. That is
/// not cosmetic: PsN builds a replicate by walking the original data file and
/// emitting each selected individual as many times as it was drawn, so dataset
/// order and draw order coincide there. Sorting reproduces that, which is what
/// lets `included_individuals` and `included_keys` agree the way the PsN guide
/// describes them. It is also a no-op statistically — a multiset has no order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replicate {
    /// 1-based; replicate 0 is reserved for the original dataset.
    pub index: usize,
    pub keys: Vec<usize>,
}

impl Replicate {
    /// Times each original subject was drawn — PsN's `sample_keys`.
    pub fn counts(&self, n_subjects: usize) -> Vec<usize> {
        let mut counts = vec![0usize; n_subjects];
        for &k in &self.keys {
            counts[k] += 1;
        }
        counts
    }
}

/// Seed for replicate `index`, derived from the run seed.
///
/// Deliberately a pure function of `(seed, index)` and nothing else. PsN's guide
/// documents the opposite behaviour as a known wart — "the results of two runs
/// will be different even if the seed is the same if the lst-file of the base
/// model is present at the start of one run but not the other", because running
/// the base model advances a shared RNG. Deriving per replicate means the drawn
/// datasets depend only on `--seed` and the design: not on `--threads`, not on
/// completion order, not on whether the base fit ran.
pub fn replicate_seed(seed: u64, index: usize) -> u64 {
    // Odd multiplier (the golden-ratio constant used by SplitMix64) so distinct
    // indices stay distinct after wrapping.
    seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Draw one replicate: sample each stratum with replacement to its allocation.
pub fn draw(strata: &Strata, allocation: &[(String, usize)], seed: u64, index: usize) -> Replicate {
    let mut rng = StdRng::seed_from_u64(replicate_seed(seed, index));
    let mut keys = Vec::new();
    for (label, n_draw) in allocation {
        let Some(pool) = strata.groups.get(label) else {
            continue;
        };
        if pool.is_empty() {
            continue;
        }
        for _ in 0..*n_draw {
            keys.push(pool[rng.random_range(0..pool.len())]);
        }
    }
    keys.sort_unstable();
    Replicate { index, keys }
}

/// Build a replicate population by cloning each drawn subject.
///
/// A subject drawn more than once must enter the fit as *independent*
/// individuals — the whole point of resampling with replacement — so the copies
/// get distinct IDs. PsN achieves the same by renumbering IDs while writing the
/// replicate data file; here the original ID is kept and a `#2`, `#3`, … suffix
/// added, so a diagnostic can still be traced back to the subject it came from.
pub fn build_population(original: &Population, replicate: &Replicate) -> Population {
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut subjects = Vec::with_capacity(replicate.keys.len());
    for &k in &replicate.keys {
        let mut subject = original.subjects[k].clone();
        let n = seen.entry(original.subjects[k].id.as_str()).or_insert(0);
        *n += 1;
        if *n > 1 {
            subject.id = format!("{}#{}", subject.id, n);
        }
        subjects.push(subject);
    }
    Population {
        subjects,
        covariate_names: original.covariate_names.clone(),
        dv_column: original.dv_column.clone(),
        input_columns: original.input_columns.clone(),
        exclusions: original.exclusions.clone(),
        warnings: Vec::new(),
    }
}

#[cfg(test)]
#[path = "resample_tests.rs"]
mod tests;
