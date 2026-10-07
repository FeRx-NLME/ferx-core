//! A fingerprint of the population a fit was given (#1685).
//!
//! `run_sir`, `run_covariance` and `load_fit` evaluate a fit's θ against a
//! population they rebuild or are handed. Matching the subject IDs is not enough:
//! a row filter the model file does not state, a dose-row filter, or a population
//! that was never bound for a level block all leave the subjects as they were and
//! change the objective. `fit()` records this fingerprint of its population
//! argument, and every post-hoc step compares the population it is about to use
//! against it.
//!
//! The encoding is SHA-256 over a canonical little-endian byte stream: `f64`s by
//! their bits, every `HashMap` in key order, every `Vec` with its length. So two
//! reads of the same file give the same fingerprint on every platform, and nothing
//! depends on a map's iteration order.
//!
//! [`Subject`], [`DoseEvent`] and [`ObsRecord`] are destructured **without `..`**,
//! so adding a field to any of them is a compile error here until someone decides
//! whether the field is hashed. Hashing it changes every digest, so it is also a
//! [`SCHEME`] bump: a fingerprint of another scheme is treated as absent, with a
//! warning, never as a mismatch.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "survival")]
use super::EventType;
use super::{DoseEvent, InfusionDef, ObsRecord, Population, RateMode, Subject};

/// The encoding version. Bump it whenever what [`PopulationFingerprint::of`]
/// hashes, or how, changes.
pub(crate) const SCHEME: u32 = 1;

/// A fingerprint of the population a fit was given: per subject, its ID, its
/// record and dose counts, and a digest of each of its records, its doses and its
/// covariate values. Opaque: it is compared, never read. Stored on
/// [`FitResult::population_fingerprint`](crate::types::FitResult::population_fingerprint)
/// and carried through `.fitrx` and the R fit object as JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PopulationFingerprint {
    scheme: u32,
    /// The fit derived its occasion labels from a model-side `iov_occasion` rule,
    /// so `occasions` and `dose_occasions` are left out of the records and doses
    /// digests, here and in every population compared against this fingerprint.
    #[serde(default)]
    occasions_derived: bool,
    /// `Population::covariate_names`, sorted.
    covariate_names: Vec<String>,
    subjects: Vec<SubjectPrint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SubjectPrint {
    id: String,
    /// Gaussian observations plus non-Gaussian observation records.
    n_obs: usize,
    n_doses: usize,
    /// Observation records, `EVID = 2` rows and reset rows, with their clocks,
    /// compartments, censoring, occasions, `L2` and `FREMTYPE`.
    records: String,
    /// Dose events and their occasions.
    doses: String,
    /// The subject's covariate map and every per-record snapshot.
    covariates: String,
}

/// The first way a population differs from a fingerprint, in the order a caller
/// should hear about it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Difference {
    SubjectCount {
        population: usize,
        fit: usize,
    },
    SubjectId {
        /// 0-based.
        position: usize,
        population: String,
        fit: String,
    },
    CovariateNames {
        /// In the fit, not in the population.
        missing: Vec<String>,
        /// In the population, not in the fit.
        extra: Vec<String>,
    },
    Records {
        id: String,
        population: usize,
        fit: usize,
    },
    Doses {
        id: String,
        population: usize,
        fit: usize,
    },
    Covariates {
        id: String,
    },
}

/// What differs, as a clause: "this population" against "the fit's". Shared by every
/// refusal that names a difference (`run_sir` / `run_covariance`, `load_fit`).
impl std::fmt::Display for Difference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let list = |names: &[String]| {
            if names.is_empty() {
                "none".to_string()
            } else {
                names
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        match self {
            Difference::SubjectCount { population, fit } => {
                write!(f, "it has {population} subjects, the fit's has {fit}")
            }
            Difference::SubjectId {
                position,
                population,
                fit,
            } => write!(
                f,
                "its subject {} is `{population}`, the fit's is `{fit}`",
                position + 1
            ),
            Difference::CovariateNames { missing, extra } => write!(
                f,
                "its covariate columns are not the fit's: missing {}, extra {}",
                list(missing),
                list(extra)
            ),
            Difference::Records {
                id,
                population,
                fit,
            } if population != fit => write!(
                f,
                "the records of subject `{id}` differ: {population} observations in this \
                 population, {fit} in the fit's"
            ),
            // The digest covers more than the observation count, so equal counts
            // need their own wording (review r1 #4).
            Difference::Records { id, population, .. } => write!(
                f,
                "the records of subject `{id}` differ with the same {population} \
                 observations: an observation time, value, compartment, censoring flag or \
                 occasion, or an `EVID = 2` or reset row"
            ),
            Difference::Doses {
                id,
                population,
                fit,
            } => write!(
                f,
                "the doses of subject `{id}` differ: {population} in this population, {fit} \
                 in the fit's, with the same observation records"
            ),
            Difference::Covariates { id } => write!(
                f,
                "the covariate values of subject `{id}` differ, with the same records and doses"
            ),
        }
    }
}

impl PopulationFingerprint {
    /// The fingerprint of `population`, occasion labels included.
    #[cfg(test)]
    pub(crate) fn of(population: &Population) -> Self {
        Self::of_with(population, false)
    }

    /// The fingerprint of `population`. With `occasions_derived` (the fit derives its
    /// occasion labels from a model-side rule, `run::occasions_are_derived`), the
    /// labels the population carries are left out: the fit overwrites them.
    pub(crate) fn of_with(population: &Population, occasions_derived: bool) -> Self {
        let Population {
            subjects,
            covariate_names,
            // How the file was read, not what was read: the DV column's name, the
            // header echo, the filter summary and reader warnings do not enter the
            // objective.
            dv_column: _,
            input_columns: _,
            exclusions: _,
            warnings: _,
        } = population;
        let mut names = covariate_names.clone();
        names.sort();
        PopulationFingerprint {
            scheme: SCHEME,
            occasions_derived,
            covariate_names: names,
            subjects: subjects
                .iter()
                .map(|s| subject_print(s, occasions_derived))
                .collect(),
        }
    }

    /// Whether this fingerprint was made with the current encoding. One that was
    /// not cannot be compared, and is treated as absent.
    pub(crate) fn is_current(&self) -> bool {
        self.scheme == SCHEME
    }

    /// The first difference between `population` and the population this
    /// fingerprint was made of, or `None` when they are the same.
    pub(crate) fn first_difference(&self, population: &Population) -> Option<Difference> {
        let got = Self::of_with(population, self.occasions_derived);
        if got.subjects.len() != self.subjects.len() {
            return Some(Difference::SubjectCount {
                population: got.subjects.len(),
                fit: self.subjects.len(),
            });
        }
        let pairs = || got.subjects.iter().zip(&self.subjects);
        if let Some((position, (p, f))) = pairs().enumerate().find(|(_, (p, f))| p.id != f.id) {
            return Some(Difference::SubjectId {
                position,
                population: p.id.clone(),
                fit: f.id.clone(),
            });
        }
        if got.covariate_names != self.covariate_names {
            let not_in = |a: &[String], b: &[String]| -> Vec<String> {
                a.iter().filter(|n| !b.contains(n)).cloned().collect()
            };
            return Some(Difference::CovariateNames {
                missing: not_in(&self.covariate_names, &got.covariate_names),
                extra: not_in(&got.covariate_names, &self.covariate_names),
            });
        }
        for (p, f) in pairs() {
            if p.n_obs != f.n_obs || p.records != f.records {
                return Some(Difference::Records {
                    id: p.id.clone(),
                    population: p.n_obs,
                    fit: f.n_obs,
                });
            }
        }
        for (p, f) in pairs() {
            if p.n_doses != f.n_doses || p.doses != f.doses {
                return Some(Difference::Doses {
                    id: p.id.clone(),
                    population: p.n_doses,
                    fit: f.n_doses,
                });
            }
        }
        pairs()
            .find(|(p, f)| p.covariates != f.covariates)
            .map(|(p, _)| Difference::Covariates { id: p.id.clone() })
    }

    /// This fingerprint under another scheme, for the tests of the scheme gate.
    #[cfg(test)]
    pub(crate) fn with_scheme(mut self, scheme: u32) -> Self {
        self.scheme = scheme;
        self
    }
}

fn subject_print(s: &Subject, occasions_derived: bool) -> SubjectPrint {
    let Subject {
        id,
        doses,
        obs_times,
        obs_raw_times,
        observations,
        obs_cmts,
        covariates,
        dose_covariates,
        obs_covariates,
        pk_only_times,
        pk_only_covariates,
        reset_times,
        reset_covariates,
        cens,
        occasions,
        obs_l2,
        dose_occasions,
        reset_occasions,
        fremtype,
        obs_records,
    } = s;

    let mut r = Encoder::default();
    r.f64s(obs_times);
    r.f64s(obs_raw_times);
    r.f64s(observations);
    r.len(obs_cmts.len());
    obs_cmts.iter().for_each(|&c| r.usize(c));
    r.len(cens.len());
    cens.iter().for_each(|&c| r.bytes(&c.to_le_bytes()));
    // Derived labels are the fit's, not the population's: both sides hash none.
    let derived_out: &[u32] = &[];
    r.u32s(if occasions_derived {
        derived_out
    } else {
        occasions
    });
    r.len(obs_l2.len());
    obs_l2.iter().for_each(|&l| r.bytes(&l.to_le_bytes()));
    r.len(fremtype.len());
    fremtype.iter().for_each(|&t| r.bytes(&t.to_le_bytes()));
    r.len(obs_records.len());
    obs_records.iter().for_each(|o| obs_record(&mut r, o));
    r.f64s(pk_only_times);
    r.f64s(reset_times);
    r.u32s(reset_occasions);

    let mut d = Encoder::default();
    d.len(doses.len());
    doses.iter().for_each(|x| dose(&mut d, x));
    d.u32s(if occasions_derived {
        derived_out
    } else {
        dose_occasions
    });

    let mut c = Encoder::default();
    c.map(covariates);
    for snapshots in [
        dose_covariates,
        obs_covariates,
        pk_only_covariates,
        reset_covariates,
    ] {
        c.len(snapshots.len());
        snapshots.iter().for_each(|m| c.map(m));
    }

    SubjectPrint {
        id: id.clone(),
        n_obs: observations.len() + obs_records.len(),
        n_doses: doses.len(),
        records: r.finish(),
        doses: d.finish(),
        covariates: c.finish(),
    }
}

fn dose(e: &mut Encoder, d: &DoseEvent) {
    let DoseEvent {
        time,
        amt,
        cmt,
        rate,
        duration,
        ss,
        ii,
        rate_mode,
        infusion_def,
    } = d;
    e.f64(*time);
    e.f64(*amt);
    e.usize(*cmt);
    e.f64(*rate);
    e.f64(*duration);
    e.bytes(&[u8::from(*ss)]);
    e.f64(*ii);
    e.bytes(&[match rate_mode {
        RateMode::Fixed => 0,
        RateMode::ModeledDuration => 1,
        RateMode::ModeledRate => 2,
    }]);
    e.bytes(&[match infusion_def {
        InfusionDef::RateDefined => 0,
        InfusionDef::DurationDefined => 1,
    }]);
}

fn obs_record(e: &mut Encoder, o: &ObsRecord) {
    match o {
        #[cfg(feature = "survival")]
        ObsRecord::Event {
            time,
            event_type,
            entry_time,
            cmt,
        } => {
            e.bytes(&[0]);
            e.f64(*time);
            match event_type {
                EventType::Exact => e.bytes(&[0]),
                EventType::RightCensored => e.bytes(&[1]),
                EventType::IntervalCensored { left, right } => {
                    e.bytes(&[2]);
                    e.f64(*left);
                    e.f64(*right);
                }
            }
            e.f64(*entry_time);
            e.usize(*cmt);
        }
        ObsRecord::DiscreteState {
            time,
            raw_time,
            state,
            cmt,
        } => {
            e.bytes(&[1]);
            e.f64(*time);
            e.f64(*raw_time);
            e.usize(*state);
            e.usize(*cmt);
        }
        ObsRecord::Count {
            time,
            raw_time,
            count,
            cmt,
        } => {
            e.bytes(&[2]);
            e.f64(*time);
            e.f64(*raw_time);
            e.bytes(&count.to_le_bytes());
            e.usize(*cmt);
        }
    }
}

/// SHA-256 over a canonical little-endian stream.
#[derive(Default)]
struct Encoder(Sha256);

impl Encoder {
    fn bytes(&mut self, b: &[u8]) {
        self.0.update(b);
    }
    fn usize(&mut self, x: usize) {
        self.bytes(&(x as u64).to_le_bytes());
    }
    fn len(&mut self, n: usize) {
        self.usize(n);
    }
    fn f64(&mut self, x: f64) {
        self.bytes(&x.to_bits().to_le_bytes());
    }
    fn f64s(&mut self, xs: &[f64]) {
        self.len(xs.len());
        xs.iter().for_each(|&x| self.f64(x));
    }
    fn u32s(&mut self, xs: &[u32]) {
        self.len(xs.len());
        xs.iter().for_each(|&x| self.bytes(&x.to_le_bytes()));
    }
    fn str(&mut self, s: &str) {
        self.len(s.len());
        self.bytes(s.as_bytes());
    }
    fn map(&mut self, m: &HashMap<String, f64>) {
        let mut keys: Vec<&String> = m.keys().collect();
        keys.sort();
        self.len(keys.len());
        for k in keys {
            self.str(k);
            self.f64(m[k]);
        }
    }
    fn finish(self) -> String {
        crate::io::hash::hex_lower(self.0.finalize().as_ref())
    }
}

#[cfg(test)]
#[path = "population_fingerprint_tests.rs"]
mod tests;
