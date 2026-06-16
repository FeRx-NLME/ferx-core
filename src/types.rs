use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// Re-export the milestone-2 sensitivity partials placeholder so
// `CompiledModel` can hold one and external callers (test fixtures and the
// `generate_data` data-generation binary) can construct an empty one. The
// inner `Expression` AST stays parser-private — only `IndivParamPartials::empty`
// and the `Debug`/`Clone` derives are reachable from outside the crate.
pub use crate::parser::model_parser::IndivParamPartials;

/// A single dose event (bolus, infusion, or oral)
#[derive(Debug, Clone)]
pub struct DoseEvent {
    pub time: f64,
    pub amt: f64,
    pub cmt: usize,
    pub rate: f64,
    pub duration: f64,
    pub ss: bool,
    pub ii: f64,
}

impl DoseEvent {
    pub fn new(time: f64, amt: f64, cmt: usize, rate: f64, ss: bool, ii: f64) -> Self {
        let duration = if rate > 0.0 { amt / rate } else { 0.0 };
        Self {
            time,
            amt,
            cmt,
            rate,
            duration,
            ss,
            ii,
        }
    }

    pub fn is_infusion(&self) -> bool {
        self.rate > 0.0
    }
}

/// Fixed-layout PK parameters — replaces HashMap<String, f64> for AD compatibility.
///
/// Index convention:
///   0: CL      (clearance)
///   1: V       (volume, or V1 for 2-cmt)
///   2: Q/Q2    (intercompartmental clearance, central ↔ peripheral 1; 2-cmt and 3-cmt)
///   3: V2      (peripheral volume 1; 2-cmt and 3-cmt)
///   4: KA      (absorption rate constant, oral only)
///   5: F       (bioavailability, default 1.0)
///   6: Q3      (intercompartmental clearance, 3-cmt: central ↔ peripheral 2)
///   7: V3      (peripheral volume 2, 3-cmt only)
///   8: LAGTIME (dose/absorption lagtime, default 0.0; equivalent to NONMEM ALAG)
///
/// Slots 0–8 are the named PK parameters above. Slots 9.. are spare capacity
/// for ODE models, whose `[individual_parameters]` may declare additional
/// "structural" parameters (rate constants, Emax/EC50, baselines, …) beyond the
/// named PK slots. `ode_param_slots` routes canonical names to slots 0–8 and
/// structural names to the remaining free slots, while keeping slots
/// `PK_IDX_F` and `PK_IDX_LAGTIME` reserved so an undeclared F/lagtime keeps its
/// default rather than being aliased by a structural parameter (issue #122).
/// The headroom here therefore bounds how many structural parameters an ODE
/// model can declare.
pub const MAX_PK_PARAMS: usize = 16;

pub const PK_IDX_CL: usize = 0;
pub const PK_IDX_V: usize = 1;
pub const PK_IDX_Q: usize = 2;
pub const PK_IDX_V2: usize = 3;
pub const PK_IDX_KA: usize = 4;
pub const PK_IDX_F: usize = 5;
pub const PK_IDX_Q3: usize = 6;
pub const PK_IDX_V3: usize = 7;
pub const PK_IDX_LAGTIME: usize = 8;

/// The engine-reserved PK slots: bioavailability (`F`) and absorption lag
/// (`lagtime`). `ode_param_slots` keeps these free for an undeclared F/lagtime
/// (issue #122), both the analytical and ODE engines apply them to the dose
/// itself rather than the RHS, and the "computed but never used" census exempts
/// parameters routed here. Single source of truth so those sites can't drift.
pub(crate) const RESERVED_PK_SLOTS: [usize; 2] = [PK_IDX_F, PK_IDX_LAGTIME];

#[derive(Debug, Clone, Copy)]
pub struct PkParams {
    pub values: [f64; MAX_PK_PARAMS],
}

impl Default for PkParams {
    fn default() -> Self {
        let mut v = [0.0; MAX_PK_PARAMS];
        v[PK_IDX_F] = 1.0; // bioavailability defaults to 1
        Self { values: v }
    }
}

impl PkParams {
    pub fn cl(&self) -> f64 {
        self.values[PK_IDX_CL]
    }
    pub fn v(&self) -> f64 {
        self.values[PK_IDX_V]
    }
    pub fn q(&self) -> f64 {
        self.values[PK_IDX_Q]
    }
    pub fn v2(&self) -> f64 {
        self.values[PK_IDX_V2]
    }
    pub fn ka(&self) -> f64 {
        self.values[PK_IDX_KA]
    }
    pub fn f_bio(&self) -> f64 {
        self.values[PK_IDX_F]
    }

    /// Bioavailable dose amount: `F · amount`. `F` scales the input on **every**
    /// route — IV bolus, infusion, and oral depot alike — matching NONMEM's `F1`
    /// (#327).
    ///
    /// This pair is the single source of truth for the F-on-dose rule. The three
    /// prediction paths derive their `F` handling from it:
    /// * event-driven (`pk/event_driven.rs`) calls these directly;
    /// * the analytical superposition path (`pk/mod.rs`) applies the same rule as
    ///   a post-multiply via `route_f_scale` — the oral-depot closed forms bake
    ///   `F` in, so they take the `1.0` branch;
    /// * the autodiff path (`ad/ad_gradients.rs`) inlines `f_bio * amt` /
    ///   `f_bio * rate` once, because under `#[autodiff]` it works on flat scalars
    ///   and cannot call `&self` methods.
    ///
    /// A change to the rule must be mirrored in those sites.
    pub(crate) fn bioavailable_amount(&self, amount: f64) -> f64 {
        self.f_bio() * amount
    }

    /// Bioavailable infusion rate: `F · rate`. See [`Self::bioavailable_amount`].
    pub(crate) fn bioavailable_rate(&self, rate: f64) -> f64 {
        self.f_bio() * rate
    }
    pub fn q3(&self) -> f64 {
        self.values[PK_IDX_Q3]
    }
    pub fn v3(&self) -> f64 {
        self.values[PK_IDX_V3]
    }
    pub fn lagtime(&self) -> f64 {
        self.values[PK_IDX_LAGTIME]
    }

    /// Map a PK parameter name to its index in the fixed-size array.
    ///
    /// `"alag"` is accepted as an alias for `"lagtime"` for NONMEM familiarity.
    pub fn name_to_index(name: &str) -> Option<usize> {
        match name {
            "cl" => Some(PK_IDX_CL),
            "v" | "v1" => Some(PK_IDX_V),
            "q" | "q2" => Some(PK_IDX_Q),
            "v2" => Some(PK_IDX_V2),
            "ka" => Some(PK_IDX_KA),
            "f" => Some(PK_IDX_F),
            "q3" => Some(PK_IDX_Q3),
            "v3" => Some(PK_IDX_V3),
            "lagtime" | "alag" => Some(PK_IDX_LAGTIME),
            _ => None,
        }
    }

    /// Build from named HashMap (bridge for parser compatibility)
    pub fn from_hashmap(map: &HashMap<String, f64>) -> Self {
        let mut p = Self::default();
        if let Some(&v) = map.get("cl") {
            p.values[PK_IDX_CL] = v;
        }
        if let Some(&v) = map.get("v") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("v1") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("q") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("q2") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("v2") {
            p.values[PK_IDX_V2] = v;
        }
        if let Some(&v) = map.get("ka") {
            p.values[PK_IDX_KA] = v;
        }
        if let Some(&v) = map.get("f") {
            p.values[PK_IDX_F] = v;
        }
        if let Some(&v) = map.get("q3") {
            p.values[PK_IDX_Q3] = v;
        }
        if let Some(&v) = map.get("v3") {
            p.values[PK_IDX_V3] = v;
        }
        if let Some(&v) = map.get("lagtime").or_else(|| map.get("alag")) {
            p.values[PK_IDX_LAGTIME] = v;
        }
        p
    }
}

/// A single subject with dosing and observation data
#[derive(Debug, Clone)]
pub struct Subject {
    pub id: String,
    pub doses: Vec<DoseEvent>,
    pub obs_times: Vec<f64>,
    /// Original (unshifted) observation times from the data file's TIME column,
    /// parallel to `obs_times`. For subjects with stacked reset occasions whose
    /// TIME restarts (see `io/datareader`), `obs_times` carries the internal
    /// monotonic timeline while this keeps the raw value for the user-clock
    /// diagnostics: sdtab/covtab TIME and `predict()`/`simulate()` TIME.
    /// `[derived]` integral windows use raw times so per-occasion AUC is
    /// correct. Populated for every observation read from a CSV (equal to
    /// `obs_times` when no shift occurred); empty only for in-memory subjects,
    /// where consumers fall back to `obs_times`.
    pub obs_raw_times: Vec<f64>,
    pub observations: Vec<f64>,
    pub obs_cmts: Vec<usize>,
    /// Subject-representative covariate values (first non-missing value per
    /// covariate). Used by the AD fast path and as a fallback when neither
    /// `dose_covariates` nor `obs_covariates` is populated.
    pub covariates: HashMap<String, f64>,
    /// Per-dose covariate snapshot (LOCF), parallel to `doses`. When the
    /// dataset has no time-varying covariates, this is empty and consumers
    /// fall back to `covariates`. NONMEM-equivalent: the value of `$PK`
    /// inputs at each dose record.
    pub dose_covariates: Vec<HashMap<String, f64>>,
    /// Per-observation covariate snapshot (LOCF), parallel to `obs_times`.
    /// Same fallback semantics as `dose_covariates`.
    pub obs_covariates: Vec<HashMap<String, f64>>,
    /// Times of EVID=2 "other event" rows (typically covariate-change
    /// markers). Only populated when the subject has time-varying
    /// covariates — for time-constant covariates these rows are no-ops
    /// (NONMEM-equivalent: $PK runs but with the same values).
    /// The event-driven propagators see them as a third event kind that
    /// does not mutate compartment amounts but does refresh the
    /// piecewise-constant rate matrix from the row's covariate values.
    pub pk_only_times: Vec<f64>,
    /// Per-EVID-2 covariate snapshot (LOCF), parallel to `pk_only_times`.
    /// Empty when no TV covariates.
    pub pk_only_covariates: Vec<HashMap<String, f64>>,
    /// Times of system-reset events (NONMEM EVID=3 "reset" and EVID=4
    /// "reset + dose"). At each of these times every compartment amount is
    /// set back to zero. For EVID=4 the row's dose is *also* recorded in
    /// `doses`; the reset is applied first (state-propagating paths use a
    /// `Reset < Dose` tie-break at a shared time). Empty for the common case
    /// of no reset rows — superposition stays valid and consumers skip the
    /// state-propagating path. Resets break dose superposition, so a subject
    /// with any reset is forced onto the event-driven analytical / ODE path.
    pub reset_times: Vec<f64>,
    /// Censoring flag per observation (0 = quantified, 1 = below LLOQ).
    /// When `cens[j] == 1`, `observations[j]` holds the LLOQ value (NONMEM convention).
    pub cens: Vec<u8>,
    /// Occasion index per observation row (parallel to `obs_times`).
    /// Empty when no IOV column is present in the data.
    pub occasions: Vec<u32>,
    /// Occasion index per dose event (parallel to `doses`).
    /// Empty when no IOV column is present in the data.
    pub dose_occasions: Vec<u32>,
    /// Non-Gaussian observation records (TTE events, discrete states, counts).
    /// Empty for all-Gaussian subjects. Populated by the data reader when the
    /// model declares a non-Gaussian endpoint for the row's CMT.
    #[cfg(feature = "survival")]
    pub obs_records: Vec<ObsRecord>,
}

impl Subject {
    pub fn has_bloq(&self) -> bool {
        self.cens.iter().any(|&c| c != 0)
    }

    /// True when the subject carries per-event covariate snapshots (i.e. at
    /// least one covariate was time-varying in the source data). When false,
    /// callers can use `covariates` directly and skip the per-event evaluation
    /// loop.
    pub fn has_tv_covariates(&self) -> bool {
        !self.dose_covariates.is_empty() || !self.obs_covariates.is_empty()
    }

    /// True when this subject carries any system-reset event (EVID=3 or
    /// EVID=4). Resets zero all compartment amounts, which dose superposition
    /// cannot express, so the prediction dispatcher routes these subjects onto
    /// the state-propagating event-driven analytical / ODE path regardless of
    /// whether they have time-varying covariates.
    pub fn has_resets(&self) -> bool {
        !self.reset_times.is_empty()
    }

    /// True when any dose record on this subject is flagged steady-state
    /// (SS=1). Used to gate paths that don't yet honour SS (event-driven,
    /// AD propagators, ODE) — see `estimation/inner_optimizer.rs` and the
    /// SS warning in `api.rs`.
    pub fn has_ss_doses(&self) -> bool {
        self.doses.iter().any(|d| d.ss)
    }

    /// Time of the first dose of the reset-occasion containing `obs_time`,
    /// used as the TAFD (time-after-first-dose) reference. For a subject with
    /// no resets this is just the earliest dose (the conventional TAFD origin);
    /// for stacked reset occasions it is the first dose at or after the most
    /// recent reset, so TAFD resets per occasion instead of measuring from the
    /// very first occasion across the shifted timeline (issue #195 review).
    /// `obs_time` is on the same (internal, possibly shifted) clock as
    /// `doses[*].time` and `reset_times`, so the result is correct regardless
    /// of any occasion shift. Returns `f64::INFINITY` when the occasion
    /// containing `obs_time` has no dose at or after its reset — which includes
    /// the no-doses-at-all case and a pure-reset (EVID=3) occasion with no
    /// subsequent dose; callers treat a non-finite result as an undefined TAFD
    /// (NaN).
    pub fn occasion_first_dose_time(&self, obs_time: f64) -> f64 {
        let seg_start = self
            .reset_times
            .iter()
            .copied()
            .filter(|&r| r <= obs_time + 1e-9)
            .fold(f64::NEG_INFINITY, f64::max);
        self.doses
            .iter()
            .map(|d| d.time)
            .filter(|&t| t >= seg_start - 1e-9)
            .fold(f64::INFINITY, f64::min)
    }

    /// Covariate snapshot at observation index `j`. Falls back to the
    /// subject-static `covariates` map when per-event snapshots aren't present.
    pub fn obs_cov(&self, j: usize) -> &HashMap<String, f64> {
        self.obs_covariates.get(j).unwrap_or(&self.covariates)
    }

    /// Covariate snapshot at dose index `k`. Same fallback as `obs_cov`.
    pub fn dose_cov(&self, k: usize) -> &HashMap<String, f64> {
        self.dose_covariates.get(k).unwrap_or(&self.covariates)
    }

    /// Covariate snapshot at EVID=2 row index `m`. Same fallback as
    /// the others — for time-constant covariates this returns the
    /// subject-static map.
    pub fn pk_only_cov(&self, m: usize) -> &HashMap<String, f64> {
        self.pk_only_covariates.get(m).unwrap_or(&self.covariates)
    }
}

/// Summary of records excluded by `[data_selection]` `ignore`/`accept` rules.
#[derive(Debug, Clone, Default)]
pub struct ExclusionSummary {
    /// Subject IDs that had all records removed (zero doses and observations remaining).
    pub excluded_subject_ids: Vec<String>,
    /// Number of observation records (EVID==0, MDV==0) excluded.
    pub n_obs_excluded: usize,
    /// Number of dose records (EVID 1/4) excluded.
    pub n_dose_excluded: usize,
    /// Number of other records excluded that are neither a scored observation
    /// nor a dose — EVID==2 (other event), EVID==3 (reset), and missing-DV
    /// observation rows (EVID==0, MDV==1). Tracked so the reported counts sum to
    /// every excluded record and the summary can't read all-zeros while rows
    /// were dropped.
    pub n_other_excluded: usize,
    /// Total CSV records read before any filtering.
    pub n_records_total: usize,
    /// `ignore` / `ignore_subjects` clauses that matched at least one record,
    /// in declaration order.
    pub fired_ignore: Vec<String>,
    /// `accept` clauses that rejected at least one record, in declaration order.
    pub fired_accept: Vec<String>,
}

/// A collection of subjects
#[derive(Debug, Clone)]
pub struct Population {
    pub subjects: Vec<Subject>,
    pub covariate_names: Vec<String>,
    pub dv_column: String,
    /// All column headers from the data CSV in original order (ID, TIME, DV, AMT, ...,
    /// covariates). Preserved verbatim so downstream consumers can echo a NONMEM-style
    /// `$INPUT` line. Empty for in-memory `Population` values that were not read from a file.
    pub input_columns: Vec<String>,
    /// Present when `[data_selection]` rules were applied; `None` when no filtering
    /// was requested or the population was constructed in-memory.
    pub exclusions: Option<ExclusionSummary>,
    /// Non-fatal warnings generated while reading the dataset (e.g. ADDL with missing II,
    /// unparseable OCC values). Propagated into `FitResult.warnings` by `fit()`.
    pub warnings: Vec<String>,
}

impl Population {
    pub fn n_obs(&self) -> usize {
        self.subjects.iter().map(|s| s.observations.len()).sum()
    }

    /// Drop per-event covariate snapshots that don't carry any
    /// variation in covariates the model actually references.
    ///
    /// The data reader populates `dose_covariates` / `obs_covariates` /
    /// `pk_only_covariates` whenever *any* non-standard CSV column
    /// varies within a subject — including pure time-tracker columns
    /// like `DAY` or `STIME` that no model expression touches. The
    /// downstream prediction path then takes the heavy event-driven
    /// route (one `pk_param_fn` call per event, plus state propagation
    /// across each interval) instead of the analytical superposition
    /// fast path that runs `pk_param_fn` once per subject. This pre-fit
    /// pass clears the snapshots for any subject whose model-referenced
    /// covariates are all time-constant, so the existing
    /// `has_tv_covariates()`-based dispatcher routes those subjects
    /// through the fast path automatically.
    ///
    /// Returns the number of subjects whose snapshots were cleared
    /// (for diagnostic / warning purposes).
    pub fn prune_irrelevant_tv_covariates(&mut self, referenced: &[String]) -> usize {
        let mut pruned = 0;
        for subj in &mut self.subjects {
            if subj.dose_covariates.is_empty()
                && subj.obs_covariates.is_empty()
                && subj.pk_only_covariates.is_empty()
            {
                continue; // already on the fast path
            }
            let mut any_relevant_tv = false;
            'covs: for cov in referenced {
                let base = subj.covariates.get(cov).copied();
                for snap in subj
                    .dose_covariates
                    .iter()
                    .chain(subj.obs_covariates.iter())
                    .chain(subj.pk_only_covariates.iter())
                {
                    if snap.get(cov).copied() != base {
                        any_relevant_tv = true;
                        break 'covs;
                    }
                }
            }
            if !any_relevant_tv {
                subj.dose_covariates.clear();
                subj.obs_covariates.clear();
                subj.pk_only_covariates.clear();
                pruned += 1;
            }
        }
        pruned
    }
}

/// Whether a declared covariate is continuous or categorical.
///
/// Both kinds are carried as `f64` in the data path (categoricals must be
/// numerically coded — see [`CovariateDecl`]). The distinction is metadata for
/// downstream consumers (R-side summary statistics, covariate-search
/// algorithms) that treat the two differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CovariateKind {
    Continuous,
    Categorical,
}

impl CovariateKind {
    /// Lowercase label used in the `[covariates]` block and in output.
    pub fn label(&self) -> &'static str {
        match self {
            CovariateKind::Continuous => "continuous",
            CovariateKind::Categorical => "categorical",
        }
    }
}

/// One entry from the optional `[covariates]` DSL block: a dataset column the
/// modeller declares as a covariate, tagged continuous or categorical. This is
/// a declaration of *availability* — it does not imply the covariate is used in
/// the structural model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CovariateDecl {
    /// Column name, case-sensitive, matching the CSV header.
    pub name: String,
    pub kind: CovariateKind,
}

/// A single row of the [`CovariateTable`], echoing one input dataset record.
#[derive(Debug, Clone, PartialEq)]
pub struct CovariateRow {
    pub id: String,
    pub time: f64,
    /// EVID of the source row (0=obs, 1=dose, 2=other, 3=reset, 4=reset+dose).
    pub evid: u32,
    /// Covariate values, parallel to [`CovariateTable::names`]. A missing value
    /// (blank / `.` / `NA` in the source) is encoded as `f64::NAN`.
    pub values: Vec<f64>,
}

/// Echo of the declared covariate columns from the input dataset, one row per
/// input record (including dose / EVID rows — unlike the observation-only
/// sdtab). Produced at data-read time when a `[covariates]` block is present,
/// and attached to [`FitResult::covariate_table`]. Missing values are
/// `f64::NAN`. Restricted to declared columns to bound memory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CovariateTable {
    /// Declared covariate names, in declaration order. Parallel to each row's
    /// `values` and to `kinds`.
    pub names: Vec<String>,
    /// Continuous/categorical tag per covariate, parallel to `names`.
    pub kinds: Vec<CovariateKind>,
    pub rows: Vec<CovariateRow>,
}

/// Between-subject variability matrix (Omega)
#[derive(Debug, Clone)]
pub struct OmegaMatrix {
    pub matrix: DMatrix<f64>,
    pub chol: DMatrix<f64>,
    pub eta_names: Vec<String>,
    pub diagonal: bool,
    /// Which (i,j) entries are free parameters (not structural zeros).
    /// Diagonal entries are always free. Off-diagonals are free only when
    /// both etas belong to the same `block_omega` declaration; cross-block
    /// and standalone-vs-block entries are structural zeros and stay false.
    /// Used by the SAEM M-step to zero sampling correlations that bleed into
    /// structurally-absent entries via `(1/N) Σ ηη^T`.
    pub free_mask: DMatrix<bool>,
    /// Pre-computed Ω⁻¹. Cached at construction so per-call code paths
    /// (`individual_nll_into`, SAEM MH proposals) don't have to clone the
    /// matrix, run Cholesky, and invert on every evaluation.
    pub inv: DMatrix<f64>,
    /// Pre-computed `log|Ω| = 2·Σᵢ log(L_ii)`. Same motivation as `inv`.
    pub log_det: f64,
}

impl OmegaMatrix {
    pub fn from_matrix_with_mask(
        m: DMatrix<f64>,
        names: Vec<String>,
        diagonal: bool,
        free_mask: DMatrix<bool>,
    ) -> Self {
        let n = m.nrows();
        // If the input matrix is PD we use it as-is. If Cholesky fails we
        // regularise (eigenvalue floor) and switch to the regularised
        // matrix from here on, so `matrix`, `chol`, `inv`, and `log_det`
        // all describe the *same* matrix downstream — otherwise the
        // FOCE inner loop's eta_prior (read from `matrix`) would be
        // inconsistent with the cached `inv` (computed from `m_reg`).
        let (matrix, chol, inv) = match m.clone().cholesky() {
            Some(c) => (m, c.l(), c.inverse()),
            None => {
                let eig = m.clone().symmetric_eigen();
                let min_eig = eig.eigenvalues.min();
                let reg = if min_eig < 0.0 { -min_eig + 1e-8 } else { 1e-8 };
                let m_reg = &m + DMatrix::identity(n, n) * reg;
                let c = m_reg
                    .clone()
                    .cholesky()
                    .expect("Regularized matrix must be PD");
                (m_reg, c.l(), c.inverse())
            }
        };
        // log|Ω| = 2·Σᵢ log(L_ii). Negative or zero diagonals shouldn't
        // happen post-regularisation but we fall back to f64::INFINITY so
        // downstream NLL code can short-circuit cleanly instead of NaNing.
        let mut log_det = 0.0;
        for i in 0..n {
            let lii = chol[(i, i)];
            if lii > 0.0 {
                log_det += lii.ln();
            } else {
                log_det = f64::INFINITY;
                break;
            }
        }
        log_det *= 2.0;
        Self {
            matrix,
            chol,
            eta_names: names,
            diagonal,
            free_mask,
            inv,
            log_det,
        }
    }

    pub fn from_matrix(m: DMatrix<f64>, names: Vec<String>, diagonal: bool) -> Self {
        let n = m.nrows();
        // Infer free_mask: diagonal entries always free; for non-diagonal
        // matrices, off-diagonals are free iff non-zero. This is the correct
        // inference when reconstructing an OmegaMatrix from a final estimate
        // matrix where the original block structure has already been imposed.
        // For initial parsing of `block_omega` declarations, use
        // `from_matrix_with_mask` directly so cross-block zeros are preserved.
        let mut free_mask = DMatrix::from_element(n, n, false);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    free_mask[(i, j)] = true;
                } else if !diagonal && m[(i, j)] != 0.0 {
                    free_mask[(i, j)] = true;
                }
            }
        }
        Self::from_matrix_with_mask(m, names, diagonal, free_mask)
    }

    pub fn from_diagonal(variances: &[f64], names: Vec<String>) -> Self {
        let n = variances.len();
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = variances[i];
        }
        Self::from_matrix(m, names, true)
    }

    /// Construct from a known lower-Cholesky factor `L` such that Ω = L Lᵀ.
    /// Avoids the Cholesky factorisation that `from_matrix*` runs, which the
    /// SAEM/FOCEI hot paths hit on every NLopt M-step and every outer
    /// iteration via `unpack_params`. Ω⁻¹ is computed from `L` directly:
    /// Ω⁻¹ = L⁻ᵀ L⁻¹, where L⁻¹ comes from one lower-triangular solve
    /// against the identity.
    pub fn from_chol_factor(
        l: DMatrix<f64>,
        names: Vec<String>,
        diagonal: bool,
        free_mask: DMatrix<bool>,
    ) -> Self {
        let n = l.nrows();
        let matrix = &l * l.transpose();
        // L⁻¹ via lower-triangular solve: L · X = I ⇒ X = L⁻¹.
        // `solve_lower_triangular(&I)` returns Some(_) iff L is non-singular;
        // L Lᵀ is PD by construction so a positive-diagonal L is guaranteed
        // unless the caller hands us a degenerate factor — fall back to a
        // full Cholesky on the materialised matrix in that degenerate case
        // so the cache is at least populated rather than panicking.
        let identity = DMatrix::<f64>::identity(n, n);
        let inv = match l.solve_lower_triangular(&identity) {
            Some(l_inv) => &l_inv.transpose() * &l_inv,
            None => matrix
                .clone()
                .cholesky()
                .expect("L Lᵀ should be PD by construction")
                .inverse(),
        };
        let mut log_det = 0.0;
        for i in 0..n {
            let lii = l[(i, i)];
            if lii > 0.0 {
                log_det += lii.ln();
            } else {
                log_det = f64::INFINITY;
                break;
            }
        }
        log_det *= 2.0;
        Self {
            matrix,
            chol: l,
            eta_names: names,
            diagonal,
            free_mask,
            inv,
            log_det,
        }
    }

    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Residual error parameters (Sigma)
#[derive(Debug, Clone)]
pub struct SigmaVector {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

/// Full set of model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    pub theta_lower: Vec<f64>,
    pub theta_upper: Vec<f64>,
    /// Per-theta FIX flags (NONMEM-style). Fixed thetas are not estimated;
    /// they retain their initial value and receive SE = 0 in the cov step.
    pub theta_fixed: Vec<bool>,
    pub omega: OmegaMatrix,
    /// Per-eta FIX flags. For diagonal omegas: flag fixes the variance.
    /// For block omegas: all etas within a fixed block share `true`, and
    /// every Cholesky element whose row *or* column is flagged is held
    /// constant during optimization. Parser rejects fixing an eta that
    /// belongs to a block unless the whole block is fixed.
    pub omega_fixed: Vec<bool>,
    pub sigma: SigmaVector,
    /// Per-sigma FIX flags.
    pub sigma_fixed: Vec<bool>,
    /// Inter-occasion variability matrix (Omega_IOV). `None` when no `kappa`
    /// declarations appear in the model file.  Always diagonal for Option A.
    pub omega_iov: Option<OmegaMatrix>,
    /// Per-kappa FIX flags (parallel to `omega_iov` diagonal).
    pub kappa_fixed: Vec<bool>,
}

impl ModelParameters {
    /// Return `true` if any parameter is marked FIX.
    pub fn has_any_fixed(&self) -> bool {
        self.theta_fixed.iter().any(|&b| b)
            || self.omega_fixed.iter().any(|&b| b)
            || self.sigma_fixed.iter().any(|&b| b)
            || self.kappa_fixed.iter().any(|&b| b)
    }
}

/// Supported PK structural models.
///
/// IV (bolus and/or infusion) administration is represented by a single
/// variant per compartment count; the bolus-vs-infusion choice is made
/// per dose event from the dataset's RATE column (see
/// `DoseEvent::is_infusion`). This mirrors NONMEM, nlmixr2, and Monolix
/// and lets a subject mix bolus and infusion doses in one record.
/// Oral routes remain a separate variant because they have a distinct
/// model structure (absorption rate constant KA, bioavailability F).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkModel {
    OneCptIv,
    OneCptOral,
    TwoCptIv,
    TwoCptOral,
    ThreeCptIv,
    ThreeCptOral,
}

impl PkModel {
    /// Canonical PK slots that MUST be mapped in a `[structural_model]` `pk(...)`
    /// line for this model, each paired with the conventional name shown in
    /// parser errors. `f`/`lagtime` are intentionally absent — they are optional
    /// and default to 1.0 / 0.0 (see `PkParams::default`).
    ///
    /// Slots are canonical (`name_to_index` values), so the `v`/`v1` and `q`/`q2`
    /// aliases satisfy the same requirement. The display names mirror the
    /// "Required Parameters" table in `docs/src/model-file/structural-model.md`;
    /// the parser enforces what that table documents (issue #309).
    pub(crate) fn required_pk_params(&self) -> &'static [(usize, &'static str)] {
        match self {
            PkModel::OneCptIv => &[(PK_IDX_CL, "cl"), (PK_IDX_V, "v")],
            PkModel::OneCptOral => &[(PK_IDX_CL, "cl"), (PK_IDX_V, "v"), (PK_IDX_KA, "ka")],
            PkModel::TwoCptIv => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q"),
                (PK_IDX_V2, "v2"),
            ],
            PkModel::TwoCptOral => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_KA, "ka"),
            ],
            PkModel::ThreeCptIv => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q2"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_Q3, "q3"),
                (PK_IDX_V3, "v3"),
            ],
            PkModel::ThreeCptOral => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q2"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_Q3, "q3"),
                (PK_IDX_V3, "v3"),
                (PK_IDX_KA, "ka"),
            ],
        }
    }

    /// The canonical short model name (e.g. `one_cpt_oral`), used in parser
    /// diagnostics. Long-form aliases (`one_compartment_oral`) normalise to this.
    ///
    /// Deliberately the inverse of the string→`PkModel` match in
    /// `parse_structural_model` (which additionally accepts the long-form
    /// aliases, so the two can't be a single bidirectional table);
    /// `canonical_name_round_trips_through_parser` guards them against drift.
    pub(crate) fn canonical_name(&self) -> &'static str {
        match self {
            PkModel::OneCptIv => "one_cpt_iv",
            PkModel::OneCptOral => "one_cpt_oral",
            PkModel::TwoCptIv => "two_cpt_iv",
            PkModel::TwoCptOral => "two_cpt_oral",
            PkModel::ThreeCptIv => "three_cpt_iv",
            PkModel::ThreeCptOral => "three_cpt_oral",
        }
    }

    /// Whether this is a first-order-absorption (oral) model. Oral models read
    /// `ka`; IV models do not. (`f` is read by every model since #327 — it scales
    /// IV bolus/infusion doses too.) The canonical home for this predicate.
    ///
    /// The `#[cfg(feature = "autodiff")]` free fn `is_oral_model` in
    /// `estimation/inner_optimizer.rs` is a pre-existing identical copy; it should
    /// delegate here, but that path isn't compiled in the default/CI build, so the
    /// change can't be verified yet — fold it in when autodiff runs in CI (#281).
    pub(crate) fn is_oral(&self) -> bool {
        matches!(
            self,
            PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
        )
    }

    /// Whether the analytical solver for this model actually reads the given PK
    /// slot. This is the single source of truth for "is a mapped param used", and
    /// it mirrors what the `pk/` closed forms consume — pinned to real solver
    /// behaviour by `consumes_pk_slot_matches_solver` in `pk/mod.rs`, so a future
    /// variant that reads a new slot can't silently drift from this:
    ///   - every required structural slot (`required_pk_params`);
    ///   - `lagtime`, applied to *every* dose (`predict_concentration` shifts the
    ///     effective dose time for IV and oral alike);
    ///   - `f` (bioavailability), applied to *every* dose and route — IV bolus,
    ///     infusion, and oral depot — scaling the bioavailable amount/rate
    ///     (#327). Both the superposition and event-driven analytical paths read
    ///     it for every model, IV included, so it is never inert.
    pub(crate) fn consumes_pk_slot(&self, slot: usize) -> bool {
        slot == PK_IDX_LAGTIME
            || slot == PK_IDX_F
            || self.required_pk_params().iter().any(|(s, _)| *s == slot)
    }
}

/// Supported residual error models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    Additive,
    Proportional,
    Combined,
}

/// How a sigma parameter enters the residual error model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaType {
    Proportional,
    Additive,
}

impl ErrorModel {
    /// Return the `SigmaType` for each sigma, in the order they appear in `FitResult.sigma`.
    pub fn sigma_types(self) -> Vec<SigmaType> {
        match self {
            ErrorModel::Proportional => vec![SigmaType::Proportional],
            ErrorModel::Additive => vec![SigmaType::Additive],
            ErrorModel::Combined => vec![SigmaType::Proportional, SigmaType::Additive],
        }
    }

    /// Number of sigma parameters this error model consumes.
    pub fn n_sigma(self) -> usize {
        match self {
            ErrorModel::Additive | ErrorModel::Proportional => 1,
            ErrorModel::Combined => 2,
        }
    }
}

/// Residual error specification for a model.
///
/// `Single` applies one error model to every observation (the default and the
/// only form analytical-PK models support). `PerCmt` dispatches a distinct
/// error model per observed compartment — the multi-endpoint / simultaneous
/// PK-PD case (issue #14). The map key is the 1-based CMT index from the data
/// file's CMT column, matching `subject.obs_cmts[i]`.
#[derive(Debug, Clone)]
pub enum ErrorSpec {
    /// One error model for all observations.
    Single(ErrorModel),
    /// Per-CMT error models. Each endpoint carries its own `ErrorModel` and the
    /// indices into the flat global `sigma.values` vector that supply its
    /// sigmas (declaration order in the `[parameters]` block).
    PerCmt(HashMap<usize, EndpointError>),
}

impl Default for ErrorSpec {
    fn default() -> Self {
        ErrorSpec::Single(ErrorModel::Additive)
    }
}

impl ErrorSpec {
    /// `SigmaType` for each entry of the flat global sigma vector, as a `Vec`
    /// of length `n_sigma`, so `FitResult` can label/scale every sigma.
    ///
    /// `Single` stamps the error model's own sigma types into the leading
    /// slots and leaves any further declared sigmas as `Additive`. `PerCmt`
    /// stamps the type of every sigma index each endpoint owns; sigmas not
    /// referenced by any endpoint default to `Additive`. Either way the
    /// returned length always equals `n_sigma`.
    pub fn sigma_types(&self, n_sigma: usize) -> Vec<SigmaType> {
        let mut out = vec![SigmaType::Additive; n_sigma];
        match self {
            ErrorSpec::Single(em) => {
                for (i, t) in em.sigma_types().into_iter().enumerate() {
                    if i < out.len() {
                        out[i] = t;
                    }
                }
            }
            ErrorSpec::PerCmt(map) => {
                for ep in map.values() {
                    let types = ep.error_model.sigma_types();
                    for (k, &idx) in ep.sigma_idx.iter().enumerate() {
                        if idx < out.len() {
                            out[idx] = types[k];
                        }
                    }
                }
            }
        }
        out
    }

    /// `d(residual variance)/d(prediction f)` for one observation at `cmt`.
    ///
    /// The score term the SAEM M-step needs alongside the variance. Additive
    /// endpoints contribute 0; proportional/combined endpoints contribute
    /// `2·f·σ_prop²` (σ_prop is the endpoint's proportional sigma, which is the
    /// first sigma for both `Proportional` and `Combined`). `Single` ignores
    /// `cmt`; `PerCmt` dispatches on the endpoint registered for `cmt`.
    pub fn dvar_df(&self, cmt: usize, f: f64, sigma: &[f64]) -> f64 {
        let (em, prop_sigma) = match self {
            ErrorSpec::Single(em) => (*em, sigma.first().copied().unwrap_or(0.0)),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => (
                    ep.error_model,
                    ep.sigma_idx
                        .first()
                        .and_then(|&i| sigma.get(i))
                        .copied()
                        .unwrap_or(0.0),
                ),
                None => return 0.0,
            },
        };
        match em {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => 2.0 * f * prop_sigma * prop_sigma,
        }
    }

    /// `d²(residual variance)/d(prediction f)²` for one observation at `cmt`.
    ///
    /// Additive endpoints contribute 0 (variance is `σ_add²`, independent of f).
    /// Proportional and combined endpoints contribute `2·σ_prop²` (variance has
    /// a `f²·σ_prop²` term, so the second derivative w.r.t. f is constant).
    /// `Single` ignores `cmt`; `PerCmt` dispatches on the endpoint registered
    /// for `cmt`. Used by the Almquist Laplace FOCEI gradient's θ-axis β_j
    /// chain — keeping the per-CMT routing here lets the same closed-form
    /// gradient handle multi-endpoint models without changing the call site.
    pub fn d2var_df2(&self, cmt: usize, sigma: &[f64]) -> f64 {
        let (em, prop_sigma) = match self {
            ErrorSpec::Single(em) => (*em, sigma.first().copied().unwrap_or(0.0)),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => (
                    ep.error_model,
                    ep.sigma_idx
                        .first()
                        .and_then(|&i| sigma.get(i))
                        .copied()
                        .unwrap_or(0.0),
                ),
                None => return 0.0,
            },
        };
        match em {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => 2.0 * prop_sigma * prop_sigma,
        }
    }

    /// `d(residual variance)/d(log σ_k)` for one observation at `cmt`, where
    /// `k` indexes the flat global sigma vector. Zero when `σ_k` does not enter
    /// this observation's endpoint, so the SAEM sigma-gradient can sum this over
    /// every observation and have each sigma pick up only its own endpoint's
    /// contributions. (`Proportional` slot → `2·σ_k²·f²`, `Additive` slot →
    /// `2·σ_k²`.)
    pub fn dvar_dlogsigma(&self, cmt: usize, k: usize, f: f64, sigma: &[f64]) -> f64 {
        let sk = match sigma.get(k) {
            Some(&s) => s,
            None => return 0.0,
        };
        let sk2 = sk * sk;
        // Resolve which `SigmaType` (if any) global index `k` plays for this
        // observation's endpoint.
        let stype = match self {
            ErrorSpec::Single(em) => em.sigma_types().get(k).copied(),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => ep
                    .sigma_idx
                    .iter()
                    .position(|&i| i == k)
                    .and_then(|p| ep.error_model.sigma_types().get(p).copied()),
                None => None,
            },
        };
        match stype {
            Some(SigmaType::Proportional) => 2.0 * sk2 * f * f,
            Some(SigmaType::Additive) => 2.0 * sk2,
            None => 0.0,
        }
    }

    /// Residual variance for one observation, dispatching on its compartment.
    ///
    /// For `Single` the `cmt` is ignored and the full `sigma` slice is used
    /// (the back-compat path). For `PerCmt` the endpoint registered for `cmt`
    /// selects the error model and slices `sigma` by its `sigma_idx`. Returns
    /// `NaN` when `cmt` has no registered endpoint, or when an endpoint's
    /// `sigma_idx` points outside `sigma` — defensive guards mirroring the
    /// scaling path. Fit-time validation rejects an uncovered CMT up front,
    /// and `build_error_spec` resolves indices against the real sigma vector,
    /// so a `NaN` here is only reachable via a hand-constructed model.
    /// Whether the residual variance depends on the prediction `f` for any
    /// endpoint (proportional or combined). When `false` (purely additive),
    /// the variance is constant in `f`, so FOCE's choice of evaluation point
    /// (linearized `f0` vs population `f(η=0)`) is irrelevant and the cheap
    /// path stays bit-identical. Used to gate the FOCE population-variance
    /// (`f(η=0)`) treatment in the marginal, analytical gradient, and
    /// covariance step.
    pub fn has_f_dependent_variance(&self) -> bool {
        match self {
            ErrorSpec::Single(em) => !matches!(em, ErrorModel::Additive),
            ErrorSpec::PerCmt(map) => map
                .values()
                .any(|ep| !matches!(ep.error_model, ErrorModel::Additive)),
        }
    }

    pub fn variance_at(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64 {
        use crate::stats::residual_error::residual_variance;
        match self {
            ErrorSpec::Single(em) => residual_variance(*em, f_pred, sigma),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => {
                    // Slice length is tied to the endpoint's error model
                    // (1 for additive/proportional, 2 for combined); the max
                    // is 2, so a stack buffer avoids a per-observation alloc.
                    let n = ep.error_model.n_sigma();
                    let mut buf = [0.0f64; 2];
                    for k in 0..n.min(2) {
                        match ep.sigma_idx.get(k).and_then(|&i| sigma.get(i)) {
                            Some(&v) => buf[k] = v,
                            None => return f64::NAN, // malformed spec / sigma length
                        }
                    }
                    residual_variance(ep.error_model, f_pred, &buf[..n.min(2)])
                }
                None => f64::NAN,
            },
        }
    }
}

/// One endpoint of a multi-endpoint (`ErrorSpec::PerCmt`) residual error model.
#[derive(Debug, Clone)]
pub struct EndpointError {
    pub error_model: ErrorModel,
    /// Positions into the flat global `sigma.values` supplying this endpoint's
    /// sigmas. Length equals `error_model.n_sigma()`.
    pub sigma_idx: Vec<usize>,
}

/// Transformation applied to a theta on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaTransform {
    /// Theta is on the natural scale (no transformation).
    Identity,
    /// Theta is on the log scale; back-transform = exp(theta).
    Log,
    /// Theta is on the logit scale: `inv_logit(THETA + ETA)`. User sets THETA
    /// on the logit scale (e.g. logit(0.7) ≈ 0.847).
    Logit,
    /// Theta is on the probability scale: `inv_logit(logit(THETA) + ETA)`.
    /// User sets THETA directly in (0,1) (e.g. 0.70 for 70% bioavailability).
    LogitProbability,
}

/// Distribution / parameterisation of an ETA random effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaParamType {
    /// `TVCL * exp(ETA)` or `exp(THETA + ETA)` — log-normal.
    LogNormal,
    /// `TVCL + ETA` — normal (additive).
    Additive,
    /// `inv_logit(THETA + ETA)` — logit-normal; THETA on the logit scale.
    Logit,
    /// `inv_logit(logit(THETA) + ETA)` — logit-normal; THETA on the (0,1) scale.
    LogitProbability,
    /// Pattern not automatically recognised.
    Custom,
}

/// Per-ETA transformation metadata, carried in `FitResult`.
#[derive(Debug, Clone)]
pub struct EtaParamInfo {
    pub eta_name: String,
    pub param_type: EtaParamType,
    /// Theta paired with this ETA. Set only when the ETA is added directly to a single THETA in
    /// the same expression (e.g. `THETA * exp(ETA)` or `inv_logit(THETA + ETA)`).
    /// Not set for mu-ref patterns like `TVCL * exp(ETA)` where the THETA is a scale factor.
    pub linked_theta: Option<String>,
    /// Name of the individual parameter this ETA appears in (e.g. `"CL"`).
    pub individual_param_name: String,
}

/// PK parameter function: maps (theta, eta, covariates) -> PkParams
pub type PkParamFn = Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> PkParams + Send + Sync>;

/// Closure signature for `[scaling] obs_scale = <expr>` (Form B). Receives
/// `(theta, eta, covariates, pk_params)` and returns the per-subject scale
/// factor used to divide the raw prediction. `pk_params` is the subject-
/// static evaluation of `model.pk_param_fn`, so the scale expression can
/// reference individual parameters (e.g. `obs_scale = 1000 / V`) — the
/// closure looks up V via its PK slot in `pk_params.values`.
pub type ScaleFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>, &PkParams) -> f64 + Send + Sync>;

/// How the structural model's raw output is mapped to the observed `DV`.
///
/// Set by the `[scaling]` block in `.ferx` model files. The convention is
/// **divisive**: `pred_scaled = pred_raw / scale`. This matches the natural
/// reading of `obs_scale = V/1000` as "divide amount by V/1000 to get
/// concentration in the user's units."
///
/// Forms A/B (this enum) post-multiply analytical and ODE predictions
/// uniformly at the end of the prediction dispatcher. Form C (ODE-only
/// `y = <expr>`) is handled inside the ODE timeline loop via
/// `OdeSpec::output_fn` instead — it replaces the state readout entirely,
/// so it doesn't share the post-multiply path.
pub enum ScalingSpec {
    /// No scaling: prediction is returned as-is.
    None,
    /// Constant divisor applied to every prediction.
    ScalarScale(f64),
    /// Per-subject divisor evaluated from `(theta, eta, covariates, pk)`.
    /// Used for expressions like `obs_scale = 1000 / V`.
    ExpressionScale { scale_fn: ScaleFn },
    /// Per-CMT dispatch for multi-analyte models (parent+metabolite,
    /// sum-of-moieties, free vs total, ...). Key is the 1-based CMT
    /// index from the data file's CMT column (matches
    /// `subject.obs_cmts[i]`, which is `usize`). Each entry is one of
    /// `None` / `ScalarScale` / `ExpressionScale` (no nested `PerCmt`
    /// — parser enforces).
    ///
    /// Fit-time validation requires every observed CMT in the population
    /// to have an entry; missing entries fall through to NaN predictions
    /// at runtime as a defensive guard against hand-constructed
    /// CompiledModels.
    PerCmt(HashMap<usize, ScalingSpec>),
}

impl Default for ScalingSpec {
    fn default() -> Self {
        Self::None
    }
}

impl ScalingSpec {
    /// Materialise the per-observation scale factors for one subject.
    ///
    /// Returns a `Vec<f64>` of length `obs_cmts.len()`. The vector is the
    /// canonical source of truth used by both the FD path
    /// (`pk::apply_scaling`) and the AD path (threaded as a `Const` slice
    /// into the four AD entry points). Keeping a single materialiser
    /// avoids drift between the two — every change to the scaling
    /// semantics is felt by both paths automatically.
    ///
    /// All variants are subject-static: the closure (`ExpressionScale`)
    /// is evaluated at most once per subject. AD therefore treats the
    /// scale as a constant w.r.t. eta. For a genuinely eta-independent
    /// scale (`WT/70`, `TVV/1000` — covariates/thetas only) AD and FD give
    /// identical gradients. For a scale that depends on eta (e.g.
    /// `obs_scale = V` with `V = TVV*exp(ETA_V)`, or `1000/V`) the frozen
    /// scale drops `d obs_scale / d eta`, so the inner loop is routed to FD
    /// by `inner_optimizer::analytical_ad_unsupported`
    /// (`ScalingSpec::breaks_ad_inner_gradient`) rather than silently
    /// producing a wrong AD gradient. See `docs/src/model-file/scaling.md`.
    ///
    /// Invalid scale values (0, negative, NaN, inf — e.g. from a covariate
    /// that's missing, or from a `1/(TVV-x)` near a singularity) propagate
    /// as `NaN` so the downstream divide produces a NaN prediction and
    /// the outer NLL goes NaN, matching the established loud-failure
    /// semantic used everywhere else in the scaling path.
    pub fn build_obs_scale_array(
        &self,
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
        pk: &PkParams,
        obs_cmts: &[usize],
    ) -> Vec<f64> {
        let n = obs_cmts.len();
        match self {
            Self::None => vec![1.0; n],
            Self::ScalarScale(k) => {
                let v = if *k > 0.0 && k.is_finite() {
                    *k
                } else {
                    f64::NAN
                };
                vec![v; n]
            }
            Self::ExpressionScale { scale_fn } => {
                let s = scale_fn(theta, eta, covariates, pk);
                let v = if s > 0.0 && s.is_finite() {
                    s
                } else {
                    f64::NAN
                };
                vec![v; n]
            }
            Self::PerCmt(map) => obs_cmts
                .iter()
                .map(|cmt| match map.get(cmt) {
                    Some(inner) => {
                        let s = match inner {
                            Self::None => 1.0,
                            Self::ScalarScale(k) => *k,
                            Self::ExpressionScale { scale_fn } => {
                                scale_fn(theta, eta, covariates, pk)
                            }
                            Self::PerCmt(_) => {
                                // Nested PerCmt is rejected at parse time;
                                // a hand-constructed CompiledModel that
                                // bypasses validation lands here. Return
                                // NaN (loud failure → NaN OFV) rather than
                                // 1.0, which would silently produce a
                                // mis-scaled fit in release builds where
                                // the debug_assert is stripped. (Caught by
                                // Copilot review on PR #85.)
                                debug_assert!(false, "nested PerCmt rejected at parse time");
                                f64::NAN
                            }
                        };
                        if s > 0.0 && s.is_finite() {
                            s
                        } else {
                            f64::NAN
                        }
                    }
                    // Validation in `fit()` catches this; defensive NaN.
                    None => f64::NAN,
                })
                .collect(),
        }
    }

    /// Returns true if this spec is a Form C readout shape that the AD
    /// path can't represent (currently only Form C lives outside
    /// `ScalingSpec` — see `OdeReadout::requires_fd`).
    ///
    /// `ScalingSpec` variants — including `ExpressionScale` and
    /// `PerCmt` — are all AD-compatible now via the per-observation scale
    /// array produced by `build_obs_scale_array`. Always returns `false`.
    /// Kept for symmetry with `OdeReadout::requires_fd` and forward
    /// compatibility (e.g. a future variant that genuinely can't fold
    /// into a Const slice).
    #[inline]
    pub fn requires_fd(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) | Self::ExpressionScale { .. } | Self::PerCmt(_) => {
                false
            }
        }
    }

    /// Returns true when materialising this spec needs a `pk_param_fn`
    /// evaluation. Only `ExpressionScale` consults `pk` (either directly
    /// or as an inner entry inside `PerCmt`). Lets callers skip the
    /// pk eval — which may be expensive on models with parsed expressions
    /// or NN forward passes — for the common `None` / `ScalarScale`
    /// cases. (Caught by Copilot review on PR #85.)
    #[inline]
    pub fn needs_pk_eval(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) => false,
            Self::ExpressionScale { .. } => true,
            Self::PerCmt(map) => map.values().any(Self::needs_pk_eval),
        }
    }

    /// Returns true when an `ExpressionScale` makes the analytical AD inner
    /// gradient unsafe, so the inner loop must use finite differences.
    ///
    /// `build_obs_scale_array` materialises the scale **subject-static** (once
    /// per gradient call), so the AD Jacobian treats `obs_scale` as constant
    /// w.r.t. eta. When the scale expression actually depends on eta (e.g.
    /// `obs_scale = V` with `V = TVV*exp(ETA_V)`, or `1000/V`), that drops
    /// `d obs_scale / d eta` and the AD gradient disagrees with the objective -
    /// observed as a ~12 OFV gap on the bundled `scaling_expression` example
    /// (issue #278 follow-up).
    ///
    /// This is **conservative**: it returns true for *any* `ExpressionScale`,
    /// including the eta-independent ones (`WT/70`, `TVV/1000`) that are AD-exact
    /// - routing those to FD costs a little speed but never correctness. A
    /// precise eta-dependence check would need the parser to record whether the
    /// scale expression reads an eta-bearing quantity; tracked as a follow-up.
    /// `None` / `ScalarScale` are always eta-independent and stay on AD.
    #[inline]
    pub fn breaks_ad_inner_gradient(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) => false,
            Self::ExpressionScale { .. } => true,
            Self::PerCmt(map) => map.values().any(Self::breaks_ad_inner_gradient),
        }
    }
}

impl std::fmt::Debug for ScalingSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "ScalingSpec::None"),
            Self::ScalarScale(k) => write!(f, "ScalingSpec::ScalarScale({})", k),
            Self::ExpressionScale { .. } => write!(f, "ScalingSpec::ExpressionScale {{ .. }}"),
            Self::PerCmt(map) => {
                let cmts: Vec<usize> = {
                    let mut v: Vec<usize> = map.keys().copied().collect();
                    v.sort();
                    v
                };
                write!(f, "ScalingSpec::PerCmt({:?})", cmts)
            }
        }
    }
}

/// Associates an ETA with its mu-referencing anchor theta.
#[derive(Debug, Clone)]
pub struct MuRef {
    pub theta_name: String,
    /// true for patterns THETA*exp(ETA) or exp(log(THETA)+ETA); false for THETA+ETA
    pub log_transformed: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Non-Gaussian endpoint types  (Phase 1: TTE / survival)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "survival")]
/// Censoring type for a TTE event record.
#[derive(Debug, Clone)]
pub enum EventType {
    Exact,
    RightCensored,
    /// Event occurred in the half-open interval (left, right].
    IntervalCensored {
        left: f64,
        right: f64,
    },
}

#[cfg(feature = "survival")]
/// A single non-Gaussian observation record on a subject.
#[derive(Debug, Clone)]
pub enum ObsRecord {
    Event {
        time: f64,
        event_type: EventType,
        /// Left truncation / delayed entry time (0.0 when none).
        /// The likelihood conditions on survival past entry_time:
        ///   H_eff(T) = H(T) − H(entry_time)
        entry_time: f64,
        cmt: usize,
    },
    // DiscreteState and Count variants deferred to Phase 4/5
}

#[cfg(feature = "survival")]
/// Analytic parametric hazard families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardFamily {
    Exponential,
    Weibull,
    Gompertz,
}

#[cfg(feature = "survival")]
/// Closure type for computing hazard parameters from (theta, eta, covariates).
///
/// Return layout by family:
///   Exponential: `[lambda]`
///   Weibull:     `[scale, shape]`
///   Gompertz:    `[alpha, gamma, loghr_term]`  (loghr_term cumulates log-hazard
///                 contributions from covariates; 0.0 when no covariates)
pub type HazardParamFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>;

#[cfg(feature = "survival")]
/// Hazard specification for a TTE endpoint.
pub enum HazardSpec {
    Analytic {
        family: HazardFamily,
        param_fn: HazardParamFn,
    },
    // OdeAccumulated deferred to Phase 2
}

#[cfg(feature = "survival")]
impl std::fmt::Debug for HazardSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HazardSpec::Analytic { family, .. } => {
                write!(f, "HazardSpec::Analytic({family:?})")
            }
        }
    }
}

#[cfg(feature = "survival")]
/// Per-CMT endpoint likelihood specification.
pub enum EndpointLikelihood {
    Gaussian(EndpointError),
    Tte { hazard: HazardSpec },
    // Binary, Ordinal, Poisson, NegBin, Ctmm, Dtmm deferred to Phase 4/5
}

#[cfg(feature = "survival")]
impl std::fmt::Debug for EndpointLikelihood {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointLikelihood::Gaussian(e) => write!(f, "Gaussian({e:?})"),
            EndpointLikelihood::Tte { hazard } => write!(f, "Tte({hazard:?})"),
        }
    }
}

/// Outcome of a single simulated observation — replaces the previous `dv_sim: f64` field.
///
/// `Continuous` preserves the existing Gaussian path unchanged.
/// `Event` carries TTE-specific outputs (gated behind `survival` feature).
#[derive(Debug, Clone)]
pub enum SimOutcome {
    /// Gaussian continuous prediction + residual noise (the only variant before Phase 1).
    Continuous { value: f64 },
    /// TTE event: simulated event time and whether it occurred before the censoring horizon.
    #[cfg(feature = "survival")]
    Event { time: f64, observed: bool },
}

impl SimOutcome {
    /// Extract the continuous value for Gaussian outcomes; NAN for all others.
    ///
    /// Calling this on a TTE `Event` row is a logic error; a `debug_assert` fires
    /// in debug builds to catch misuse early.
    pub fn continuous_value(&self) -> f64 {
        match self {
            SimOutcome::Continuous { value } => *value,
            #[cfg(feature = "survival")]
            SimOutcome::Event { .. } => {
                debug_assert!(
                    false,
                    "continuous_value() called on a TTE Event row — filter by CMT type first"
                );
                f64::NAN
            }
        }
    }
}

/// A compiled model ready for estimation
pub struct CompiledModel {
    pub name: String,
    pub pk_model: PkModel,
    pub error_model: ErrorModel,
    /// Residual error specification. For single-endpoint models this is
    /// `ErrorSpec::Single(error_model)`; for multi-endpoint models it carries
    /// the per-CMT dispatch and `error_model` holds a representative endpoint.
    pub error_spec: ErrorSpec,
    pub pk_param_fn: PkParamFn,
    pub n_theta: usize,
    /// Number of between-subject variability (BSV) ETAs.
    pub n_eta: usize,
    /// Number of inter-occasion variability (IOV) kappa parameters.
    /// Zero when no `kappa` declarations are present.
    pub n_kappa: usize,
    pub n_epsilon: usize,
    pub theta_names: Vec<String>,
    /// BSV ETA names only (length == n_eta).
    pub eta_names: Vec<String>,
    /// IOV kappa names (length == n_kappa). Empty when no IOV.
    pub kappa_names: Vec<String>,
    /// Names of the individual parameters declared at the top level of the
    /// `[individual_parameters]` block, in declaration order. Parallel to
    /// `pk_indices`; for analytical models the i-th name is the variable
    /// whose value lands in `PkParams.values[pk_indices[i]]`. For ODE
    /// models the i-th name is written sequentially into slot `i` by
    /// `pk_param_fn`. Used by the FFI to label per-subject EBE individual
    /// parameter values (e.g. `CL`, `V`, `Ka`).
    ///
    /// Bound: `pk_param_fn` writes at most `MAX_PK_PARAMS` slots (the size
    /// of the fixed `PkParams.values` array). For analytical models the
    /// parser already routes assignments through that fixed slot table, so
    /// excess names are not possible. For ODE models with more than
    /// `MAX_PK_PARAMS` top-level `[individual_parameters]` assignments,
    /// names beyond index `MAX_PK_PARAMS - 1` will appear in this list but
    /// `pk_param_fn` won't store their values — downstream consumers will
    /// read either zero or NaN for those slots. In practice no PK model
    /// approaches this limit.
    pub indiv_param_names: Vec<String>,
    /// Symbolic partial derivatives of every top-level `[individual_parameters]`
    /// assignment w.r.t. each θ and η axis, precomputed at parse time. Outer
    /// Vec is parallel to `indiv_param_names`; inner Vecs have length
    /// `n_theta` (user-declared θ) and `n_eta + n_kappa` (extended η) on the
    /// θ and η sides respectively.
    ///
    /// Reserved for a future analytical-η-gradient path. The original
    /// downstream consumers (Tier 4a milestones 3-5: augmented ODE RHS,
    /// Form C readout sensitivities, `gradient = sens` estimator wiring)
    /// were reverted in #145 after the `gradient = sens` path failed to
    /// deliver a wall-time win at low n_η. The partials themselves are
    /// still produced at parse time and are exercised by parser unit
    /// tests; they're kept on `CompiledModel` so the primitive is in
    /// place when a future symbolic-gradient consumer lands.
    ///
    /// Field itself is `pub` so external test fixtures and the
    /// `generate_data` binary can write `IndivParamPartials::empty()` into
    /// it. The inner Expression AST stays private — outside callers can
    /// only stuff in an empty placeholder, not read or mutate the
    /// parser-produced partials.
    #[allow(dead_code)] // no runtime consumer after #145; see field doc.
    pub indiv_param_partials: IndivParamPartials,
    pub default_params: ModelParameters,
    /// Per-eta flag (parallel to `eta_names` / omega diagonal): `true` when
    /// the user wrote `omega NAME ~ X (sd)` and the parser squared the value.
    /// Pure display metadata — the stored `default_params.omega` is always on
    /// the variance scale. Always `false` for etas declared inside a
    /// `block_omega`, which is variance-only.
    pub omega_init_as_sd: Vec<bool>,
    /// Per-sigma flag (parallel to `default_params.sigma.values`): `true` when
    /// the user wrote `sigma NAME ~ X (sd)`. Since #56 the .ferx default for
    /// sigma is variance — the parser `sqrt`s the variance-scale input into
    /// the internal SD representation. With `(sd)` the value is stored as-is.
    pub sigma_init_as_sd: Vec<bool>,
    /// Per-kappa flag (parallel to `kappa_names`): `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Empty when no kappa declarations are present.
    pub kappa_init_as_sd: Vec<bool>,
    /// Detected mu-referencing relationships: eta_name → (theta_name, log_transformed).
    /// Populated by the parser; empty map means no mu-referencing detected.
    pub mu_refs: HashMap<String, MuRef>,
    /// Same as `mu_refs` but for IOV kappa parameters (kappa_name → MuRef).
    pub kappa_mu_refs: HashMap<String, MuRef>,
    /// Computes covariate-adjusted typical values per subject for AD.
    /// Returns one value per `[individual_parameters]` assignment (in
    /// declaration order), evaluated with eta = 0. Covariates and theta are
    /// folded in; only eta is differentiated. The AD inner loop then
    /// computes `pk[pk_indices[i]] = tv[i] * exp(dot(sel_flat[i,:], eta))`,
    /// so `tv.len() == pk_indices.len() == eta_map.len() == sel_flat.len() / n_eta`,
    /// and the eta application is driven by `sel_flat` rather than being
    /// positional. When `Some`, enables AD gradient computation in the
    /// inner loop; when `None` (e.g. ODE models), falls back to FD.
    pub tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>>,
    /// Maps each `[individual_parameters]` assignment (by declaration order)
    /// to its PK parameter slot. E.g. for a model with CL, V, KA:
    /// `[PK_IDX_CL, PK_IDX_V, PK_IDX_KA] = [0, 1, 4]`. Parallel to the
    /// output of `tv_fn` and to `eta_map`; used by AD to route each tv
    /// value to the correct PK slot. Note: the index here is the
    /// assignment/tv index, *not* the eta index — see `eta_map` for the
    /// latter (they diverge when some params are eta-free).
    pub pk_indices: Vec<usize>,
    /// Per-tv eta index: `eta_map[i]` is the eta index referenced by the
    /// i-th [individual_parameters] assignment, or -1 if the assignment
    /// references no eta (e.g. `Q = TVQ`). Parallel to `pk_indices` and the
    /// output of `tv_fn`; used by the AD path to correctly combine eta
    /// with each tv slot. Before this field existed the AD loop assumed
    /// `pk_indices.len() == n_eta` with 1:1 positional correspondence,
    /// which silently misaligned eta and produced NaN gradients for models
    /// with eta-free PK parameters like 2-cpt where `Q` is fixed.
    pub eta_map: Vec<i32>,
    /// Precomputed `pk_indices` as `Vec<f64>` — the form the AD functions
    /// actually want. Cached here so each BFGS gradient call doesn't
    /// reallocate and recast a tiny vector; on a 110k-find_ebe fit that
    /// saves several million allocations.
    pub pk_idx_f64: Vec<f64>,
    /// Precomputed one-hot eta selector (row-major, n_tv × n_eta) derived
    /// from `eta_map`. Same motivation as `pk_idx_f64`: built once, reused
    /// for every AD gradient evaluation.
    pub sel_flat: Vec<f64>,
    /// ODE specification. When `Some`, predictions use ODE integration instead of
    /// analytical PK equations. The `pk_param_fn` output is flattened and passed
    /// to the ODE RHS function as the parameter vector.
    pub ode_spec: Option<crate::ode::OdeSpec>,
    /// Index of the first diffusion theta in the theta vector, and the parallel
    /// mapping from diffusion-theta index to ODE state index.
    /// `None` when no `[diffusion]` block is present.
    /// Used by `ekf_p_obs` to read current diffusion variances from `theta`
    /// without requiring mutation of `ode_spec` during estimation.
    pub diffusion_theta_start: Option<usize>,
    /// For each diffusion theta (offset from `diffusion_theta_start`),
    /// the index of the ODE state it applies to. Parallel to the diffusion
    /// theta slice of `theta`. Empty when `diffusion_theta_start` is `None`.
    pub diffusion_state_indices: Vec<usize>,
    /// Mirror of [`FitOptions::bloq_method`] so likelihood/AD paths can read
    /// it without threading the options struct through every call site.
    /// Set by [`fit_from_files`](crate::fit_from_files) automatically;
    /// callers invoking [`fit`](crate::fit) with a hand-built `CompiledModel`
    /// must set this field to match `options.bloq_method` themselves.
    pub bloq_method: BloqMethod,
    /// Covariate names referenced by any expression in the model (preserved
    /// in the case the modeller wrote). Validated against the data's covariate
    /// columns before a fit so that a missing/misspelt covariate fails loudly
    /// instead of silently evaluating to zero.
    pub referenced_covariates: Vec<String>,
    /// Mirror of [`FitOptions::gradient_method`] so the inner loop can
    /// dispatch at runtime without threading the options struct through
    /// every call site. Set by [`fit_from_files`](crate::fit_from_files)
    /// automatically; callers invoking [`fit`](crate::fit) with a
    /// hand-built `CompiledModel` must set this field to match
    /// `options.gradient_method` themselves. A mismatch is not detected —
    /// `find_ebe` reads this field, not `options`.
    pub gradient_method: GradientMethod,
    /// Warnings generated at parse time (e.g. mu-referencing disabled for
    /// conditional parameters).  Prepended to `FitResult.warnings` by `fit()`.
    pub parse_warnings: Vec<String>,
    /// True when an individual parameter is assigned inside an `if`-branch that
    /// references an ETA (e.g. `if (WT>70) { CL = TVCL*exp(ETA_CL) } else {...}`).
    /// Set by the parser. The analytical AD kernels can't represent the branch
    /// structure, so `inner_optimizer::analytical_ad_unsupported` routes such
    /// models to FD. (Structured replacement for matching the "conditional
    /// parameter" `parse_warnings` string.)
    pub has_conditional_eta_params: bool,
    /// Per-ETA transformation metadata derived from the `[individual_parameters]`
    /// expressions at parse time. Length ≤ n_eta (only ETAs whose expression was
    /// classified are present). Forwarded into `FitResult`.
    pub eta_param_info: Vec<EtaParamInfo>,
    /// Per-theta transformation: `theta_transform[i]` describes whether theta i
    /// is used on the natural (Identity), log, or logit scale. Length == n_theta.
    pub theta_transform: Vec<ThetaTransform>,
    /// Parsed `[covariate_nn NAME]` blocks (one entry per block in the model
    /// file). Empty when the `nn` feature is off or no block is present.
    ///
    /// Consumed by `build_pk_param_fn` and `tv_fn`: each NN's forward output is
    /// pre-computed once per call and looked up by `Expression::NnOutput` via
    /// the `NAME.OUTPUT` dot-access syntax. Weights live in `theta` starting at
    /// `weights_offset` for `mapper.n_weights()` slots, so they participate in
    /// the optimizer vector like any other theta.
    #[cfg(feature = "nn")]
    pub covariate_nns: Vec<crate::nn::CovariateNn>,
    /// How the structural model's raw output is mapped to the observed `DV`.
    /// Default `ScalingSpec::None` preserves the historical behaviour where
    /// the prediction is returned unchanged. Forms A/B from the `[scaling]`
    /// block populate this field; Form C lives on `ode_spec.output_fn`.
    pub scaling: ScalingSpec,
    /// Log-transform-both-sides (LTBS) active. When `true`, the effective
    /// prediction is `log(f)` and the residual error is additive on the log
    /// scale (matching NONMEM's `Y = LOG(F) + EPS`). Set by the parser when
    /// the `[error_model]` uses `log(DV) ~ additive(...)` or
    /// `DV ~ log_additive(...)`. The prediction sinks log-wrap their output
    /// when this is set, so IPRED/PRED/IWRES/CWRES and simulated DV are all
    /// reported on the log scale.
    pub log_transform: bool,
    /// Only meaningful when `log_transform` is `true`. `true` means the data's
    /// `DV` column is *already* on the log scale (case 1, `DV ~ log_additive`),
    /// so `fit()` must NOT log-transform it again; `false` means `DV` is on the
    /// natural scale (case 2, `log(DV) ~ additive`) and `fit()` log-transforms
    /// the observations once at load. Ignored when `log_transform` is `false`.
    pub dv_pre_logged: bool,
    /// Derived expression specifications from [derived] block.
    /// Empty when no [derived] block is present. Evaluated post-fit.
    pub derived_exprs: Vec<DerivedExprSpec>,
    /// Column names from [output] block. Validated at fit time.
    pub output_columns: Vec<String>,
    /// Per-CMT non-Gaussian endpoint specifications.
    /// Empty for models with only Gaussian observations.
    /// Keyed by the CMT value declared in `[event_model]` / future blocks.
    #[cfg(feature = "survival")]
    pub endpoints: HashMap<usize, EndpointLikelihood>,
}

/// Inner-loop (per-subject EBE) gradient method.
///
/// The inner optimizer is BFGS; what differs across variants is how the
/// gradient of the individual NLL w.r.t. ETA is computed.
///
/// - `Ad`: reverse-mode automatic differentiation via Enzyme. One forward
///   pass + one reverse pass per gradient, regardless of `n_eta`. Requires
///   the crate to be compiled with the `autodiff` feature and the model to
///   have an analytical PK path (`tv_fn` populated). Falls back to `Fd`
///   automatically when either condition isn't met (e.g. ODE models, which
///   currently have no AD path).
/// - `Fd`: central finite differences on the forward NLL. Performs `2·n_eta`
///   forward evaluations per gradient, so cost scales linearly with the
///   number of random effects.
/// - `Auto` (default): pick `Ad` whenever it is available, else `Fd`.
///
/// ## When each wins
///
/// AD's relative advantage over FD grows with:
/// 1. **Number of etas.** FD cost scales as `O(n_eta)`; AD stays roughly
///    flat. For `n_eta ≥ 3` AD is already faster per gradient call on every
///    analytical PK model tested.
/// 2. **Forward-pass cost.** Many observations per subject, many doses per
///    subject, 2- or 3-compartment analytical formulas, and (when
///    implemented) ODE-based models all amortize AD's fixed reverse-pass
///    overhead and make the per-gradient gap wider.
///
/// On small analytical problems (`n_eta ≈ 3`, few observations, 1-cpt PK)
/// the wall-clock difference can be small because gradient work is only a
/// fraction of total fit time — NLopt, population NLL reduction, and
/// parallel scheduling dominate. Relative gradient-call speedups we have
/// measured range from ~1.5× (3-cpt infusion) to ~5× (1-cpt oral).
///
/// ## Numerical equivalence
///
/// For well-conditioned problems both methods converge to the same OFV
/// within line-search tolerance. FD introduces `O(1e-9)` noise per
/// component; AD is exact up to floating-point roundoff. Rare disagreements
/// at the 2nd-decimal level of OFV usually reflect different trajectories
/// to the same optimum rather than a correctness gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientMethod {
    Auto,
    Ad,
    Fd,
}

impl Default for GradientMethod {
    fn default() -> Self {
        Self::Auto
    }
}

impl CompiledModel {
    /// Returns true when this model uses ODE integration; false for analytical PK.
    pub fn is_ode_based(&self) -> bool {
        self.ode_spec.is_some()
    }

    /// Copy the configured ODE solver tolerances from `opts` onto this model's
    /// [`OdeSpec`] (no-op for analytical models). Call this once after the
    /// model file's `[fit_options]` and any call-time `settings` overrides have
    /// been merged into `opts`, so the integrator uses the requested accuracy.
    /// The parser calls it at parse time, so `.ferx` `[fit_options]` and any
    /// entry that integrates the parsed spec as-is (`predict`, `fit_from_files`)
    /// already use the configured accuracy.
    ///
    /// Note: [`fit`](crate::fit) takes `&CompiledModel` and does **not** call
    /// this. The integrator reads [`OdeSpec::solver_opts`], never
    /// `FitOptions::ode_reltol` directly, so a caller that merges call-time
    /// `settings` into its own `FitOptions` (as the R wrapper's `ferx_fit`
    /// does) must re-apply this on an owned model *before* `fit` for those
    /// overrides to reach the solver. Idempotent.
    pub fn sync_ode_solver_opts(&mut self, opts: &FitOptions) {
        if let Some(ode) = self.ode_spec.as_mut() {
            ode.solver_opts.reltol = opts.ode_reltol;
            ode.solver_opts.abstol = opts.ode_abstol;
            ode.solver_opts.max_steps = opts.ode_max_steps;
        }
    }

    /// Returns true when the model has a `[diffusion]` block (SDE / EKF path).
    pub fn is_sde(&self) -> bool {
        self.diffusion_theta_start.is_some()
    }

    /// Returns true when the model has a time-to-event (`[event_model]`)
    /// endpoint. The analytical single-snapshot AD kernel computes the
    /// PK-observation NLL, not the hazard/survival likelihood, so its
    /// eta-gradient through the hazard (especially the shape parameters) is
    /// wrong - `tte_weibull` / `tte_gompertz` diverged ~2-5 OFV from FD under
    /// AD. `inner_optimizer::analytical_ad_unsupported` routes these to FD.
    #[cfg(feature = "survival")]
    pub fn has_tte(&self) -> bool {
        self.endpoints
            .values()
            .any(|e| matches!(e, EndpointLikelihood::Tte { .. }))
    }

    /// Always false without the `survival` feature - TTE endpoints can't be
    /// parsed, so no model can carry one.
    #[cfg(not(feature = "survival"))]
    pub fn has_tte(&self) -> bool {
        false
    }

    /// Returns true when `[individual_parameters]` declares `LAGTIME` (or its
    /// `ALAG` alias). Used by the prediction dispatcher and inner optimizer
    /// to choose between cached-schedule / AD fast paths and the lagtime-
    /// aware slow paths.
    ///
    /// Checks both routes by which lagtime can be wired in:
    ///   1. Analytical PK: `pk_indices` contains `PK_IDX_LAGTIME` when the
    ///      `[structural_model]` line includes `lagtime=` / `alag=`.
    ///   2. ODE: the LAGTIME/ALAG slot is populated by name in
    ///      `build_pk_param_fn`'s ODE branch (sequential pk_indices do not
    ///      reflect this), so we fall back to scanning `indiv_param_names`.
    pub fn has_lagtime(&self) -> bool {
        if self.pk_indices.iter().any(|&i| i == PK_IDX_LAGTIME) {
            return true;
        }
        self.indiv_param_names.iter().any(|n| {
            let u = n.to_uppercase();
            u == "LAGTIME" || u == "ALAG"
        })
    }

    /// Residual variance for one observation, dispatching on its compartment.
    /// Thin convenience wrapper over [`ErrorSpec::variance_at`] for call sites
    /// that already hold the `&CompiledModel`.
    pub fn residual_variance_at(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64 {
        self.error_spec.variance_at(cmt, f_pred, sigma)
    }

    /// Canonical compartment names for analytical models, used in `[derived]` expressions.
    /// For ODE models use `ode_spec.state_names` instead.
    /// Returns a `'static` slice so it can be used in `DerivedContext` without lifetime issues.
    pub fn analytical_compartment_names(&self) -> &'static [String] {
        debug_assert!(
            self.ode_spec.is_none(),
            "analytical_compartment_names called on an ODE model — use ode_spec.state_names instead"
        );
        use std::sync::OnceLock;
        macro_rules! names {
            ($lock:ident, $($name:expr),+) => {{
                static $lock: OnceLock<Vec<String>> = OnceLock::new();
                $lock.get_or_init(|| vec![$($name.to_string()),+]).as_slice()
            }};
        }
        match self.pk_model {
            PkModel::OneCptIv => names!(ONE_CMT_IV, "central"),
            PkModel::OneCptOral => names!(ONE_CMT_ORAL, "depot", "central"),
            PkModel::TwoCptIv => names!(TWO_CMT_IV, "central", "peripheral"),
            PkModel::TwoCptOral => names!(TWO_CMT_ORAL, "depot", "central", "peripheral"),
            PkModel::ThreeCptIv => names!(THREE_CMT_IV, "central", "peripheral1", "peripheral2"),
            PkModel::ThreeCptOral => names!(
                THREE_CMT_ORAL,
                "depot",
                "central",
                "peripheral1",
                "peripheral2"
            ),
        }
    }
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("name", &self.name)
            .field("pk_model", &self.pk_model)
            .field("error_model", &self.error_model)
            .field("error_spec", &self.error_spec)
            .field("n_theta", &self.n_theta)
            .field("n_eta", &self.n_eta)
            .field("n_kappa", &self.n_kappa)
            .finish()
    }
}

/// Per-subject estimation results
#[derive(Debug, Clone)]
pub struct SubjectResult {
    pub id: String,
    pub eta: DVector<f64>,
    pub ipred: Vec<f64>,
    pub pred: Vec<f64>,
    pub iwres: Vec<f64>,
    pub cwres: Vec<f64>,
    pub ofv_contribution: f64,
    pub cens: Vec<u8>,
    /// Number of observations for this subject (MDV=0 rows).
    pub n_obs: usize,
    /// Extra sdtab columns from [derived] and [output] blocks, computed
    /// post-fit. Each entry is (column_name, per-observation values). Subject-
    /// level aggregates (max, AUC, tmax) are repeated across all observation rows.
    pub extra_columns: Vec<(String, Vec<f64>)>,
    /// Per-observation TAD computed with individual lagtime. Populated by
    /// `compute_extra_output_columns` whenever the model has a lagtime or [derived]/[output]
    /// blocks exist. Empty if those conditions are not met; output.rs falls back to
    /// a lagtime=0 approximation in that case (correct for the common case).
    pub per_obs_tad: Vec<f64>,
    /// Full compartment state vector at each observation time. For ODE models this
    /// is the raw solver state `u[i]` in whatever units the ODE defines (typically
    /// amounts for standard PK ODEs). For SDE models (`[diffusion]` block), these are
    /// the deterministic ODE states, not EKF-filtered states. For analytical models
    /// this is the natural per-compartment value (amounts for depot, concentrations for
    /// central/peripheral). Indexed `[obs_j][cmt_i]`.
    /// Empty for IOV subjects and analytical TV-covariate subjects (those cases yield
    /// NaN in `[derived]`; see W_DERIVED_CMT_IOV_UNSUPPORTED / W_DERIVED_CMT_TV_ANALYTICAL).
    /// Scaling (`apply_scaling`, Form A/C) is never applied here — only `ipred` is scaled.
    pub compartment_states: Vec<Vec<f64>>,
}

// ── Derived expression types ──────────────────────────────────────────────────

/// Context threaded into every [derived] expression evaluation.
pub struct DerivedContext<'a> {
    pub theta: &'a [f64],
    pub eta: &'a [f64],
    pub indiv_params: &'a HashMap<String, f64>,
    pub covariates: &'a HashMap<String, f64>,
    pub ipred: f64,
    pub pred: f64,
    pub dv: f64,
    pub time: f64,
    pub tafd: f64,
    pub tad: f64,
    pub prev_derived: &'a HashMap<String, f64>,
    /// Raw compartment state at this observation time. For ODE models this is
    /// the solver state `u[i]`; for SDE models these are deterministic ODE states
    /// (not EKF-filtered). For analytical models it follows the convention in
    /// the `compartment_states` docs on `SubjectResult`. Empty slice for IOV subjects,
    /// analytical TV-covariate subjects, and grid-integral points when
    /// `uses_compartments` is false.
    pub compartments: &'a [f64],
    /// Names parallel to `compartments` — ODE state names or analytical names.
    pub compartment_names: &'a [String],
}

pub type DerivedEvalFn = Box<dyn Fn(&DerivedContext<'_>) -> f64 + Send + Sync>;
pub type DerivedFilterFn = Box<dyn Fn(&DerivedContext<'_>) -> bool + Send + Sync>;

pub struct DerivedExprSpec {
    pub name: String,
    pub kind: DerivedKind,
    /// True if any sub-expression in `kind` references `compartments[i]` or a
    /// named ODE state variable.  Used to gate `W_DERIVED_CMT_*` warnings so
    /// they fire only when compartment states are actually requested.
    pub uses_compartments: bool,
}

pub enum DerivedKind {
    /// Evaluated independently at each observation row.
    PerRow { eval: DerivedEvalFn },
    /// Reduction over observation rows (max/min/tmax), optionally filtered.
    /// The scalar is repeated for all rows of the subject.
    Aggregate {
        func: AggFunction,
        value: DerivedEvalFn,
        filter: Option<DerivedFilterFn>,
    },
    /// Numeric integration over a time window.
    Integral {
        integrand: DerivedEvalFn,
        /// When `Some`, only time points where this evaluates to true contribute.
        condition: Option<DerivedFilterFn>,
        /// True when integrand references DV → use observation times only.
        data_based: bool,
        /// True when the integrand references `compartments[i]` or a named ODE
        /// state variable. When true and the integral uses a fine grid, the ODE
        /// solver (or analytical formula) is re-run at grid points for accuracy.
        uses_compartments: bool,
        window: IntegralWindow,
        step: IntegralStep,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunction {
    Max,
    Min,
    Tmax,
}

#[derive(Debug, Clone)]
pub enum IntegralWindow {
    Explicit {
        from: f64,
        to: f64,
    },
    /// One integral per period-aligned window; each observation gets its window's value.
    Periodic {
        period: f64,
        anchor: f64,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum IntegralStep {
    /// Use observation times only (DV integrals; fallback when model unavailable).
    ObsTimes,
    /// Fine internal grid with this step size (hours).
    Fixed(f64),
    /// Auto: (to − from) / 500.
    Auto,
}

/// How per-occasion kappa random effects are treated in the IS marginal likelihood.
///
/// `Marginalized` — kappa is jointly sampled with eta using a block-diagonal
/// posterior Hessian. The IS -2LL integrates over both η and κ uncertainty,
/// making it directly comparable to NONMEM's `$EST METHOD=IMP LAPLACIAN=1`.
/// `FixedAtMode` — kappa is held at its EBE (κ̂) and only eta is sampled. This is a
/// *partial* marginal likelihood that ignores κ uncertainty. (Legacy; no longer
/// used for IOV models.)
/// `NotApplicable` — model has no kappa declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaTreatment {
    NotApplicable,
    FixedAtMode,
    Marginalized,
}

/// Result of the importance-sampling marginal log-likelihood step.
///
/// Produced by the `Imp` stage in a method chain (`methods = [..., imp]`).
/// Surfaced on `FitResult.importance_sampling`.
#[derive(Debug, Clone)]
pub struct ImportanceSamplingResult {
    /// `−2 · Σᵢ log p(yᵢ | θ)` estimated by importance sampling. Lower-bias
    /// alternative to the FOCE/Laplace OFV when subject posteriors are
    /// non-Gaussian (sparse data, strong nonlinearity). For IOV models,
    /// this uses joint (eta, kappa) sampling and is directly comparable
    /// to the FOCE OFV.
    pub minus2_log_likelihood: f64,
    /// Monte-Carlo standard error on `minus2_log_likelihood`. Scales with
    /// `1/sqrt(n_samples)`; halve by quadrupling `is_samples`.
    pub mc_standard_error: f64,
    /// `(subject_id, ESS/K)` for every subject whose normalized effective sample
    /// size fraction fell below `FitOptions::is_low_ess_threshold`. Empty list
    /// means every subject's proposal matched its posterior well.
    pub low_ess_subjects: Vec<(String, f64)>,
    /// Number of importance samples drawn per subject (`FitOptions::is_samples`).
    pub n_samples: usize,
    /// Student-t proposal degrees of freedom (`FitOptions::is_proposal_df`).
    pub proposal_df: f64,
    /// Minimum across-subject normalized ESS fraction (ESS / K). 1.0 = ideal,
    /// near 0 = degenerate proposal for at least one subject.
    pub ess_min: f64,
    /// Median across-subject normalized ESS fraction.
    pub ess_median: f64,
    /// Treatment of per-occasion kappa random effects.  See [`KappaTreatment`].
    pub kappa_treatment: KappaTreatment,
}

/// Outcome of the post-estimation covariance step.
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceStatus {
    /// User set `covariance = false`; step was not attempted.
    NotRequested,
    /// Covariance matrix was successfully computed.
    Computed,
    /// Step was attempted but failed (e.g. singular Hessian).
    Failed,
    /// FD Hessian was non-PD; SIR was run as a fallback and succeeded.
    SirFallback,
}

/// What to do when the covariance step produces a non-positive-definite Hessian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceFallback {
    /// Do nothing; leave the covariance step as failed (default).
    #[default]
    None,
    /// Run SIR with a proposal built from the rectified (|eigenvalue|) Hessian,
    /// inflated 4× for heavier tails. Parameter uncertainty is then reported as
    /// 95% credible intervals from the SIR posterior quantiles instead of
    /// `H⁻¹`-based standard errors.
    Sir,
}

/// Which estimator to use for the parameter covariance matrix, mirroring
/// NONMEM's `$COVARIANCE MATRIX=` options. All three share the same FD Hessian
/// `R` (the observed information) and per-subject score cross-product
/// `S = Σᵢ gᵢgᵢᵀ`; they differ only in how those are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceMethod {
    /// `R⁻¹` — inverse observed-information (Hessian) matrix. The model-based
    /// covariance; assumes the model is correctly specified (default, NONMEM
    /// `MATRIX=R`).
    #[default]
    Hessian,
    /// `S⁻¹` — inverse cross-product (outer-product-of-gradients) matrix. The
    /// empirical-information covariance (NONMEM `MATRIX=S`).
    CrossProduct,
    /// `R⁻¹ S R⁻¹` — the Huber–White "sandwich". Robust to model
    /// mis-specification; NONMEM's default (`MATRIX=RSR`).
    Sandwich,
}

/// Severity level for a structured warning entry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WarningSeverity {
    Critical,
    Warning,
    Info,
}

/// A structured warning with severity, category, and message.
///
/// Populated in parallel with `FitResult.warnings` (which remains for
/// backward compatibility). The `category` is a fixed lowercase vocabulary:
/// `convergence`, `covariance_step`, `optimizer_health`, `dw_autocorrelation`,
/// `bloq_method`, `sir`, `importance_sampling`, `data_quality`,
/// `omega_structure`, `ebe_convergence`, `gradient_fallback`,
/// `mu_referencing`, `optimizer_config`, `multi_start`, `cancelled`,
/// `threads`, `condition_number`, `eta_normality`, `eps_shrinkage`,
/// `experimental`, `general` (fallback for unrecognised messages).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarningEntry {
    pub severity: WarningSeverity,
    /// Fixed lowercase category string (see type-level docs).
    pub category: String,
    /// Human-readable message. For messages that carry a multi-stage chain
    /// prefix such as `[FOCEI] ...`, only the body after the prefix is stored
    /// here; the method tag is moved into `source_method`. For unprefixed
    /// messages this is identical to the corresponding entry in
    /// `FitResult.warnings`.
    pub message: String,
    /// For multi-stage chains, the method that produced this warning.
    pub source_method: Option<String>,
}

/// Classify a free-text warning message into a structured `WarningEntry`.
///
/// This is the single source of truth for warning severity/category in
/// ferx-core. The R wrapper consumes the structured output directly and never
/// re-classifies message text. Multi-stage chain prefixes (`[FOCEI] ...`) are
/// stripped into `source_method`; the remaining message is matched against the
/// fixed category vocabulary. Unrecognised messages fall back to
/// `Warning`/`general`.
pub fn classify_warning(raw: &str) -> WarningEntry {
    // Strip a leading "[METHOD] " chain prefix into source_method.
    let (source_method, msg) = if let Some(rest) = raw.strip_prefix('[') {
        if let Some(idx) = rest.find(']') {
            let tag = &rest[..idx];
            let body = rest[idx + 1..].trim_start();
            (Some(tag.to_string()), body.to_string())
        } else {
            (None, raw.to_string())
        }
    } else {
        (None, raw.to_string())
    };

    let lower = msg.to_lowercase();

    // (severity, category) keyed off distinctive substrings. Order matters:
    // more specific patterns first.
    let (severity, category) = if lower.contains("did not converge")
        || lower.contains("without convergence")
        || lower.contains("no multi-start run converged")
    {
        (WarningSeverity::Critical, "convergence")
    } else if lower.contains("covariance step failed")
        || lower.contains("covariance failed")
        || (lower.contains("covariance step") && lower.contains("not positive definite"))
    {
        // "ses not available" intentionally omitted — too broad.
        // The compound check catches format_non_pd_warning ("Covariance step:
        // Hessian is not positive definite") whose prefix differs from "failed:"
        // messages. It must be compound so that "SIR failed: covariance not
        // positive definite" (no "covariance step" token) still routes to "sir".
        (WarningSeverity::Critical, "covariance_step")
    } else if lower.contains("covariance step regularized")
        || lower.contains("off-diagonal fd stencil")
    {
        // Regularisation warning and off-diagonal NaN soft warning both indicate a
        // degraded-but-present covariance step result. Severity within the message
        // (minor/moderate/severe, or "over-optimistic") informs guidance; the
        // category is always covariance_step.
        (WarningSeverity::Warning, "covariance_step")
    } else if lower.contains("ill-conditioned") || lower.contains("condition number") {
        // Note: "covariance step failed: Hessian has ill-conditioned entries" contains
        // "ill-conditioned" but is caught by "covariance step failed" above (else-if chain).
        // Any future covariance message that contains "ill-conditioned" but NOT "failed:"
        // would land here instead — keep this ordering in mind when adding new messages.
        (WarningSeverity::Critical, "condition_number")
    } else if lower.contains("trust radius") || lower.contains("degenerate") {
        (WarningSeverity::Warning, "optimizer_health")
    } else if lower.contains("autocorrelation") || lower.contains("durbin") {
        (WarningSeverity::Warning, "dw_autocorrelation")
    } else if lower.contains("shapiro") || lower.contains("non-normal") {
        (WarningSeverity::Warning, "eta_normality")
    } else if lower.contains("experimental feature") {
        // Experimental-feature notices (issue #175): SDE and neural-network
        // components emit a runtime warning so results are applied with caution.
        (WarningSeverity::Warning, "experimental")
    } else if lower.contains("m3 bloq") || lower.contains("bloq handling") {
        (WarningSeverity::Warning, "bloq_method")
    } else if lower.contains("sir failed") || lower.contains("sir requested") {
        (WarningSeverity::Warning, "sir")
    } else if lower.contains("ess = 0") || lower.contains("proposal collapse") {
        (WarningSeverity::Warning, "importance_sampling")
    } else if lower.contains("eps shrinkage") {
        (WarningSeverity::Warning, "eps_shrinkage")
    } else if lower.starts_with("w_addl_missing_ii") || lower.contains("addl > 0 but ii") {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.starts_with("w_iov_occ_missing")
        || lower.contains("missing or unparseable values in iov_column")
    {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.starts_with("w_missing_dv") {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.contains("ltbs")
        || lower.contains("non-positive dv")
        || lower.contains("ss=1 dose")
        || lower.contains("ss=1 infusion")
        || lower.contains("evid=3/4")
        || lower.contains("lagtime evaluates")
    {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.contains("mixed lognormal") || lower.contains("mixed log-normal") {
        (WarningSeverity::Warning, "omega_structure")
    } else if lower.contains("hmc is unavailable") {
        // "falls back to" intentionally removed: no emitted message uses that exact
        // phrase. The SAEM HMC message is fully covered by "hmc is unavailable".
        (WarningSeverity::Info, "gradient_fallback")
    } else if lower.contains("mu-ref") || lower.contains("mu-referencing") {
        (WarningSeverity::Info, "mu_referencing")
    } else if lower.contains("global_search disabled") {
        // Runtime failure: CRS2-LM init failed — the optimiser ran without global search.
        (WarningSeverity::Warning, "optimizer_config")
    } else if lower.contains("global_search") {
        (WarningSeverity::Info, "optimizer_config")
    } else if lower.contains("multi-start") {
        (WarningSeverity::Info, "multi_start")
    } else if lower.contains("cancelled by user") {
        (WarningSeverity::Info, "cancelled")
    } else if lower.contains("threads configured") || lower.contains("threads than subjects") {
        (WarningSeverity::Info, "threads")
    } else if lower.contains("n\u{00b2} ofv")
        || lower.contains("n^2 ofv")
        || (lower.contains("parameters") && lower.contains("covariance step:"))
    {
        (WarningSeverity::Info, "covariance_step")
    } else {
        (WarningSeverity::Warning, "general")
    };

    WarningEntry {
        severity,
        category: category.to_string(),
        message: msg,
        source_method,
    }
}

/// Full fit result
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Final method in the chain (same as `method_chain.last()`).
    pub method: EstimationMethod,
    /// Full sequence of methods executed, in order. Always has at least one entry.
    pub method_chain: Vec<EstimationMethod>,
    pub converged: bool,
    pub ofv: f64,
    pub aic: f64,
    pub bic: f64,
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    /// Names of the random effects (etas), parallel to the omega diagonal.
    pub eta_names: Vec<String>,
    pub omega: DMatrix<f64>,
    pub sigma: Vec<f64>,
    /// Names of the sigma parameters, parallel to `sigma`.
    pub sigma_names: Vec<String>,
    /// Residual error model (additive, proportional, combined).
    ///
    /// For multi-endpoint (per-CMT) models this is only the *representative*
    /// endpoint's error model; it does not describe the other endpoints.
    /// `sigma_types` (parallel to `sigma`/`sigma_names`) carries the correct
    /// per-sigma classification and should be preferred by consumers that need
    /// to distinguish endpoints.
    pub error_model: ErrorModel,
    pub covariance_matrix: Option<DMatrix<f64>>,
    pub se_theta: Option<Vec<f64>>,
    pub se_omega: Option<Vec<f64>>,
    pub se_sigma: Option<Vec<f64>>,
    /// FIX flags carried through from the model so the output layer can
    /// render `FIXED` for SE columns rather than the (meaningless) zero
    /// they acquire from the reduced-Hessian covariance step.
    pub theta_fixed: Vec<bool>,
    pub omega_fixed: Vec<bool>,
    pub sigma_fixed: Vec<bool>,
    /// Per-eta SD-init flag (parallel to the omega diagonal): `true` when the
    /// initial value was written as `omega NAME ~ X (sd)`. Lets downstream
    /// printers annotate the estimate with `[initial specified as SD]`.
    /// Always `false` for block-omega entries.
    pub omega_init_as_sd: Vec<bool>,
    /// Per-sigma SD-init flag (parallel to `sigma`): `true` when the initial
    /// value was written as `sigma NAME ~ X (sd)`, `false` for the variance
    /// default.
    pub sigma_init_as_sd: Vec<bool>,
    pub subjects: Vec<SubjectResult>,
    pub n_obs: usize,
    pub n_subjects: usize,
    pub n_parameters: usize,
    pub n_iterations: usize,
    pub interaction: bool,
    pub warnings: Vec<String>,
    /// Structured counterpart to `warnings` — same entries with severity and category metadata.
    pub warnings_structured: Vec<WarningEntry>,
    // SIR results (optional)
    pub sir_ci_theta: Option<Vec<(f64, f64)>>,
    pub sir_ci_omega: Option<Vec<(f64, f64)>>,
    pub sir_ci_sigma: Option<Vec<(f64, f64)>>,
    pub sir_ess: Option<f64>,
    /// Resampled packed parameter vectors retained from the SIR step, available
    /// when `FitOptions.sir_keep_samples = true`. Each `Vec<f64>` is a draw in
    /// the packed parameter space — same layout as `pack_params`:
    /// `[log-theta, Cholesky-omega, log-sigma]`, with the IOV Cholesky block
    /// appended when the model has kappa declarations.
    /// Consumed by `simulate_with_uncertainty()` with `UncertaintyMethod::Sir`.
    pub sir_resamples_packed: Option<Vec<Vec<f64>>>,
    /// Importance-sampling marginal log-likelihood result. `Some` when an
    /// `Imp` stage ran in the method chain (`methods = [..., imp]`). The
    /// `−2 log L_IS` value is lower-bias than the FOCE/Laplace `ofv` for
    /// sparsely-sampled subjects and is the preferred quantity for AIC/BIC
    /// model comparison in those settings. See [`ImportanceSamplingResult`].
    pub importance_sampling: Option<ImportanceSamplingResult>,
    // IOV results (present when kappa declarations exist in the model)
    pub omega_iov: Option<DMatrix<f64>>,
    pub kappa_names: Vec<String>,
    pub kappa_fixed: Vec<bool>,
    /// Per-kappa SD-init flag (parallel to the `omega_iov` diagonal). Same
    /// semantics as `omega_init_as_sd` — `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Always `false` for block_kappa entries.
    pub kappa_init_as_sd: Vec<bool>,
    pub se_kappa: Option<Vec<f64>>,
    /// Pooled kappa shrinkage: one value per kappa parameter, averaged over all
    /// subject-occasion pairs.  Empty when `n_kappa == 0`.
    pub shrinkage_kappa: Vec<f64>,
    /// Per-occasion kappa shrinkage: `shrinkage_kappa_by_occ[occ_idx][kappa_idx]`.
    /// `occ_idx` is the 0-based position within each subject's own occasion list
    /// (order in which distinct OCC values first appear in that subject's rows),
    /// **not** the raw OCC column value.  For unbalanced designs where subjects
    /// have different OCC sequences, a given `occ_idx` may map to different OCC
    /// values across subjects — use `shrinkage_kappa` (pooled) in that case.
    /// Empty when `n_kappa == 0` or only one occasion is present.
    pub shrinkage_kappa_by_occ: Vec<Vec<f64>>,
    /// Per-subject, per-occasion kappa EBEs.
    /// `ebe_kappas[i][k]` is the kappa vector for subject i, occasion k.
    /// Outer vec is empty when `n_kappa == 0`.
    pub ebe_kappas: Vec<Vec<DVector<f64>>>,
    /// Estimated OFV evaluations saved by the SAEM mu-ref gradient step M-step.
    /// Non-None only when method=saem and mu_referencing=true.
    pub saem_mu_ref_m_step_evals_saved: Option<u64>,
    /// Number of subjects that used HMC at least once during the SAEM E-step.
    /// `None` when `n_leapfrog = 0` (MH-only) or for non-SAEM methods.
    pub saem_n_subjects_hmc: Option<usize>,
    /// Gradient method used in the inner (per-subject EBE) BFGS loop.
    pub gradient_method_inner: String,
    /// Gradient method used in the outer (population parameter) optimizer.
    pub gradient_method_outer: String,
    /// True when the model uses ODE integration; false for analytical PK.
    pub uses_ode_solver: bool,
    /// True when the model has a `[diffusion]` block (SDE / EKF likelihood).
    pub uses_sde: bool,
    /// Number of Rayon worker threads used during this fit.
    pub n_threads_used: usize,
    /// NLopt algorithms requested but not available in this platform build.
    pub nlopt_missing_algorithms: Vec<String>,
    /// Estimated OFV evaluations for the covariance step (n_params²), set
    /// when `run_covariance_step = true` and `n_parameters > 30`.
    pub covariance_n_evals_estimated: Option<usize>,
    /// Path to the per-iteration optimizer trace CSV, present when
    /// `FitOptions::optimizer_trace = true`.
    pub trace_path: Option<String>,
    /// Number of outer iterations in which at least one subject had an
    /// unconverged EBE.  Always `0` for SAEM (which uses MH sampling).
    pub ebe_convergence_warnings: u32,
    /// Worst-case number of unconverged subjects in a single outer iteration.
    pub max_unconverged_subjects: u32,
    /// Total number of times the Nelder-Mead fallback was invoked across all
    /// subjects and all outer iterations.  Always `0` for SAEM.
    pub total_ebe_fallbacks: u32,
    /// Outcome of the post-estimation covariance step.
    pub covariance_status: CovarianceStatus,
    /// ETA shrinkage per random effect: `1 - SD(eta_hat_k) / sqrt(omega_kk)`.
    /// `NaN` when `omega_kk` is zero.
    pub shrinkage_eta: Vec<f64>,
    /// EPS shrinkage: `1 - SD(IWRES)`.  `NaN` when fewer than 2 valid residuals.
    pub shrinkage_eps: f64,
    /// Pooled lag-1 Pearson correlation of IWRES across subjects.
    /// `NaN` when no subject has ≥ 2 valid IWRES values.
    pub iwres_lag1_r: f64,
    /// Pooled Durbin-Watson statistic for IWRES within subjects.
    /// 2.0 = no autocorrelation; < 1.5 = positive; > 2.5 = negative.
    /// `NaN` when no subject has ≥ 2 valid IWRES values.
    pub dw_statistic: f64,
    /// Wall-clock time for the complete fit in seconds.
    pub wall_time_secs: f64,
    /// Model name (from the `.ferx` file or "Unnamed").
    pub model_name: String,
    /// ferx-core library version (from Cargo.toml at compile time).
    pub ferx_version: String,
    /// Per-ETA transformation metadata (see `EtaParamInfo`). Used by the R
    /// layer to pick the correct CI / CV% formula for each random effect.
    pub eta_param_info: Vec<EtaParamInfo>,
    /// Per-theta transformation (Identity / Log / Logit), parallel to `theta`.
    /// Tells the R layer whether a theta must be back-transformed before display.
    pub theta_transform: Vec<ThetaTransform>,
    /// Per-sigma type (Proportional / Additive), parallel to `sigma`.
    pub sigma_types: Vec<SigmaType>,
    /// Eigenvalues of the correlation matrix of free (non-fixed) parameters,
    /// sorted descending. `None` when the covariance step was not run, failed,
    /// or fewer than two free parameters exist.
    pub cov_eigenvalues: Option<Vec<f64>>,
    /// Ratio of the largest to smallest eigenvalue of the correlation matrix of
    /// free parameters. `f64::INFINITY` when the smallest eigenvalue is
    /// non-positive (signals a near-singular parameter space). `None` when
    /// `cov_eigenvalues` is `None`.
    pub cov_condition_number: Option<f64>,
    /// Whether each BSV eta is lognormally parameterised (`true`) or
    /// additive/unknown (`false`). Parallel to `eta_names` / omega diagonal.
    pub eta_log_transformed: Vec<bool>,
    /// Parameter-level correlation matrix for BSV omega.  Entry `[i,j]` uses
    /// the lognormal formula `(exp(ω_ij)−1)/√((exp(ω_ii)−1)(exp(ω_jj)−1))`
    /// when both etas are lognormal, otherwise falls back to
    /// `ω_ij/√(ω_ii·ω_jj)`.  `None` when omega is diagonal (no off-diagonals).
    pub omega_param_corr: Option<DMatrix<f64>>,
    /// Parameter-level correlation matrix for IOV block kappa, analogous to
    /// `omega_param_corr`.  `None` when `omega_iov` is absent or diagonal.
    pub omega_iov_param_corr: Option<DMatrix<f64>>,
    /// Path to the `.ferx` model file used for this fit, as supplied by the
    /// caller. `Some` when the fit was launched via `fit_from_files` or
    /// `run_model_with_data`; `None` when `fit()` was called with an in-memory
    /// `CompiledModel`. Stored verbatim (no canonicalisation) so paths don't
    /// leak the runner's home directory into shared `.fitrx` bundles.
    pub model_path: Option<String>,
    /// Path to the NONMEM-format CSV data file used for this fit, as supplied
    /// by the caller. `Some` / `None` follows the same rules as `model_path`.
    pub data_path: Option<String>,
    /// SHA-256 hex digest (64 chars, lowercase) of the model file bytes at
    /// fit time. Used by `run_sir` to refuse stale data when the caller
    /// re-supplies a model or asks the function to re-read from `model_path`.
    /// Computed only when the fit was launched from a file path.
    pub model_hash: Option<String>,
    /// SHA-256 hex digest of the data file bytes at fit time. Same semantics
    /// as `model_hash`.
    pub data_hash: Option<String>,
    /// Verbatim content of the `.ferx` model file. `Some` when the fit was
    /// launched via `fit_from_files` / CLI or loaded from a `.fitrx` bundle;
    /// `None` for in-memory `fit()` callers who never had a file path.
    pub model_text: Option<String>,
    /// Initial theta values as supplied to the optimizer, parallel to `theta`
    /// and `theta_names`.
    pub theta_init: Vec<f64>,
    /// Initial omega matrix (variance scale), same layout as `omega`.
    pub omega_init: DMatrix<f64>,
    /// Initial sigma values, parallel to `sigma` and `sigma_names`.
    pub sigma_init: Vec<f64>,
    /// `(min_time, max_time)` across all observation records. `None` only when
    /// there are no observations at all.
    pub obs_time_range: Option<(f64, f64)>,
    /// Gradient of the objective function at the best-OFV parameter point,
    /// in the packed parameter space (log-theta, Cholesky-omega, log-sigma).
    /// `Some` only for NLopt gradient-based runs (SLSQP, L-BFGS, MMA) when at
    /// least one gradient-requesting iteration improved the OFV; `None` for
    /// BOBYQA (derivative-free), built-in BFGS, GN, and SAEM.
    pub final_gradient: Option<Vec<f64>>,
    // ── Run settings (for runlog / reproducibility) ──────────────────────────
    /// Outer optimizer used for this fit, as a short lowercase label
    /// ("bobyqa", "slsqp", "nlopt_lbfgs", "mma", "bfgs", "lbfgs",
    /// "trust_region").  Always populated; the label is the same regardless of
    /// method chain length.
    pub optimizer: String,
    /// Number of random multi-starts attempted. 1 means a single fit from
    /// the model-file initial values (no multi-start).
    pub n_starts: usize,
    /// Seed used to perturb initial values across multi-starts.  `None` when
    /// `n_starts == 1` (no perturbation applied) or when no seed was set and
    /// the run used a random seed derived from the system clock.
    pub multi_start_seed: Option<u64>,
    /// Seed used for the SAEM MCMC E-step.  `None` for non-SAEM methods or
    /// when no explicit seed was set in `[fit_options]`.
    pub saem_seed: Option<u64>,
    /// Seed used for the SIR resampling step.  `None` when SIR was not run or
    /// no explicit seed was set.
    pub sir_seed: Option<u64>,
    /// Seed used for the importance-sampling Monte Carlo step.  `None` when IS
    /// was not run or no explicit seed was set.
    pub is_seed: Option<u64>,
    /// BLOQ handling method: "drop" (observations below LOQ are excluded) or
    /// "m3" (M3 likelihood for censored observations).
    pub bloq_method: String,
    /// Maximum number of outer optimizer iterations allowed.
    pub outer_maxiter: usize,
    /// Gradient-norm convergence tolerance for the outer optimizer.
    pub outer_gtol: f64,
    /// NCA initialisation method used to derive starting values, if any.
    /// One of "nca", "nca_sweep", "nca_ebe", or `None` when the model-file
    /// initial values were used directly.
    pub inits_from_nca: Option<String>,
    /// Names of covariate columns present in the dataset, in the order they
    /// appear in the data file's header.  Mirrors the NONMEM `$INPUT` echo —
    /// lets `ferx_runlog()` report which covariates were available without
    /// requiring the caller to re-read the CSV.  Empty for in-memory `fit()`
    /// calls that never touch a file.
    pub covariate_names: Vec<String>,
    /// All column headers from the data CSV in original order (ID, TIME, DV, AMT, ...,
    /// covariates), analogous to NONMEM `$INPUT`. Empty for in-memory `fit()` calls.
    pub input_columns: Vec<String>,
    /// One entry per `[covariate_nn NAME]` block in the model, populated by
    /// `fit()` from `CompiledModel.covariate_nns`. Empty when the `nn`
    /// feature is off or no block is declared. Output writers
    /// (`write_estimates_yaml`, `print_results`, `.fitrx`) use this to
    /// collapse the wall of per-weight thetas (`W_NN_l_i_j`, `B_NN_l_i`)
    /// into a single readable `neural_networks:` summary section — see
    /// `plans/dcm-and-low-dim-node.md` "Option E".
    #[cfg(feature = "nn")]
    pub neural_networks: Vec<NeuralNetworkInfo>,
    /// Echo of the declared covariate columns from the input dataset (ID, TIME,
    /// EVID + one column per declared covariate, one row per input record).
    /// `Some` only when the model has a `[covariates]` block AND the fit was
    /// launched from a data file; `None` for the in-memory [`fit`] entry point
    /// (which has no raw rows) or when no `[covariates]` block is declared.
    /// Missing values are `f64::NAN`. See [`CovariateTable`].
    pub covariate_table: Option<CovariateTable>,
    /// Record-level exclusion statistics; `Some` when `[data_selection]` rules
    /// were active during the fit (or the caller supplied `ignore`/`accept`
    /// expressions).  `None` means no filtering was requested.
    pub exclusions: Option<ExclusionSummary>,
}

/// Minimal per-NN metadata carried on `FitResult` so output writers can
/// summarise NN weights without re-walking `theta_names` to detect them.
#[cfg(feature = "nn")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralNetworkInfo {
    /// Block name from the model file (e.g. `TYPICAL_PK`).
    pub name: String,
    /// Layer shape including input and output dimensions
    /// (e.g. `[2, 16, 5]` for 2 inputs → 16 hidden → 5 outputs).
    pub shape: Vec<usize>,
    /// Hidden-layer activation name (e.g. `"tanh"`, `"relu"`).
    pub hidden_activation: String,
    /// Output-layer activation name.
    pub output_activation: String,
    /// Total weight + bias count for this NN.
    pub n_weights: usize,
    /// Index into `FitResult.theta` (and `FitResult.theta_names`) where
    /// this NN's contiguous weight block starts.
    pub weights_offset: usize,
    /// Input covariate names in declaration order.
    pub input_names: Vec<String>,
    /// PK output names in declaration order.
    pub output_names: Vec<String>,
}

/// Options for fit()
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Primary estimation method (used when `methods` is empty).
    /// When `methods` is non-empty, `method` is ignored for execution and
    /// is set to the final method in the chain for backwards-compatible reporting.
    pub method: EstimationMethod,
    /// Sequence of estimation methods to run. Each stage's converged parameters
    /// are used as the initial values for the next stage. The final stage
    /// produces the reported fit (covariance, diagnostics, OFV). Leave empty
    /// to run a single stage using `method`.
    pub methods: Vec<EstimationMethod>,
    pub outer_maxiter: usize,
    pub outer_gtol: f64,
    pub inner_maxiter: usize,
    pub inner_tol: f64,
    /// RK45 ODE solver relative tolerance (`[fit_options] ode_reltol`, or via
    /// `ferx_fit(settings = list(ode_reltol = ...))`). Default `1e-4`. Only
    /// affects ODE models. The default reproduces analytical closed forms in
    /// PRED to ~1e-4, but the FOCE OFV amplifies solver error, so a tighter
    /// value (e.g. `1e-10`) is needed for the ODE-form OFV to match the
    /// analytical OFV. Copied onto `OdeSpec::solver_opts` via
    /// [`CompiledModel::sync_ode_solver_opts`].
    pub ode_reltol: f64,
    /// RK45 ODE solver absolute tolerance (`[fit_options] ode_abstol`).
    /// Default `1e-6`. See [`FitOptions::ode_reltol`].
    pub ode_abstol: f64,
    /// RK45 ODE solver maximum step count per integration segment
    /// (`[fit_options] ode_max_steps`). Default `10000`. Raise if a tight
    /// `ode_reltol` exhausts the step budget on stiff multi-compartment
    /// segments. See [`FitOptions::ode_reltol`].
    pub ode_max_steps: usize,
    pub run_covariance_step: bool,
    /// *Initial* relative step size for the finite-difference Hessian in the
    /// covariance step. The actual step for parameter i is
    /// `fd_hessian_step * (1 + |x_hat[i]|)`. Default `1e-2`. ferx halves this
    /// automatically (up to 8×) if a diagonal stencil comes back non-finite, so
    /// manual tuning is rarely needed for overflow; decrease (e.g. `1e-3`) for
    /// smoother OFV surfaces where FD noise is the main concern.
    pub fd_hessian_step: f64,
    /// What to do when the FD Hessian is non-positive-definite.
    /// Default [`CovarianceFallback::None`] leaves the covariance step as failed.
    /// [`CovarianceFallback::Sir`] runs SIR with a fallback proposal covariance
    /// built from the rectified (`|eigenvalue|`) Hessian, inflated 4×.
    pub covariance_fallback: CovarianceFallback,
    /// Which covariance estimator to assemble, mirroring NONMEM `$COV MATRIX=`.
    /// Default [`CovarianceMethod::Hessian`] (`R⁻¹`). [`CovarianceMethod::CrossProduct`]
    /// (`S⁻¹`) and [`CovarianceMethod::Sandwich`] (`R⁻¹SR⁻¹`) add the per-subject
    /// score cross-product `S`; currently supported for FOCEI and IOV fits.
    pub covariance_method: CovarianceMethod,
    /// Build the covariance R-matrix (Hessian) from second differences of the
    /// reconverged marginal OFV, rather than from a central difference of the
    /// analytical population gradient. **Default `true`.** The analytical stencil
    /// holds the H-matrix `a = ∂f/∂η` fixed in the `log|H̃|` θ-gradient (it omits
    /// `∂a/∂θ = ∂²f/∂η∂θ`), which biases the SE of *weakly-identified* structural
    /// parameters — e.g. TVKA on warfarin reads ~9% high versus a Richardson
    /// FD-of-OFV ground truth. The OFV-Hessian stencil recomputes `a` (and
    /// everything else) at every perturbed point, so it captures that curvature
    /// exactly (up to the FD step) and matches the ground truth to <1%; it is the
    /// same stencil used for IOV and f-dependent FOCE. It costs O(n²) reconverged
    /// OFV evaluations versus O(n) gradient evaluations, but both stencils
    /// parallelise over perturbation points so the wall-clock cost is ≈ equal in
    /// practice. Set `false` to force the faster analytical-gradient stencil
    /// (e.g. on very high-dimensional models where the O(n²) point count
    /// dominates).
    pub covariance_ofv_hessian: bool,
    pub interaction: bool,
    pub verbose: bool,
    pub optimizer: Optimizer,
    pub lbfgs_memory: usize,
    /// Run a gradient-free global pre-search (NLopt GN_CRS2_LM) before local optimization.
    pub global_search: bool,
    /// Max evaluations for the global pre-search (0 = auto).
    pub global_maxeval: usize,
    // SAEM-specific options
    pub saem_n_exploration: usize,
    pub saem_n_convergence: usize,
    /// Number of block-kernel MH proposals per subject per SAEM outer
    /// iteration. The default 20 mixes well on hard cold-start surfaces
    /// (e.g. Emax PKPD with stressful initial values, where chains at 3
    /// proposals can lock the M-step into a degenerate basin with the
    /// PD-curve thetas at boundary values) and, together with the
    /// componentwise kernel (run automatically for multi-η models), keeps a
    /// block Ω from collapsing to a near rank-1 correlation matrix. The
    /// componentwise sweep count `max(2, n_mh_steps / n_eta)` is derived from
    /// this value (the kernel is skipped entirely for single-η models).
    /// Reduce to 3-10 for the older/faster behaviour on simpler
    /// well-identified models; raise (30-50) only when the diagnostic shows
    /// the M-step is still tracking correlated samples.
    pub saem_n_mh_steps: usize,
    pub saem_adapt_interval: usize,
    /// Number of initial exploration iterations during which the BSV/IOV Ω
    /// M-step is suppressed (Ω held at its initial value) while the MH chain
    /// warms up. Prevents the iteration-1 Ω collapse on sparse data, where a
    /// cold-start chain (η = 0, few MH steps) yields a tiny `(1/N)Σηηᵀ` that
    /// the γ=1 M-step would otherwise install as Ω, starving the proposal.
    /// Clamped to `saem_n_exploration` at use. `0` disables the burn-in.
    pub saem_omega_burnin: usize,
    pub saem_seed: Option<u64>,
    /// Number of leapfrog steps per HMC proposal in the SAEM E-step.
    /// `0` (default) uses the Metropolis-Hastings random-walk sampler.
    /// A positive value (e.g. `3`) enables HMC; requires the `autodiff`
    /// feature and an analytical PK model — falls back to MH otherwise.
    pub saem_n_leapfrog: usize,
    /// Levenberg-Marquardt damping factor for Gauss-Newton (0 = pure GN).
    pub gn_lambda: f64,
    // SIR options
    pub sir: bool,
    pub sir_samples: usize,
    pub sir_resamples: usize,
    pub sir_seed: Option<u64>,
    /// When `true` and SIR is enabled, the resampled packed parameter vectors
    /// are retained on `FitResult.sir_resamples_packed` for downstream use by
    /// `simulate_with_uncertainty()`. Adds `n_resamples * n_packed * 8` bytes
    /// to the result; default `false`.
    pub sir_keep_samples: bool,
    /// Degrees of freedom for the Student-t SIR proposal distribution.
    /// Heavier tails than the normal improve ESS for parameters near boundaries
    /// (omega variances, constrained thetas). Default 5.0 follows Dosne (2017).
    /// Set to a large value (e.g. 100.0) to recover near-normal behaviour.
    pub sir_df: f64,
    // Importance-sampling marginal log-likelihood options (consumed by the
    // `Imp` chain stage; ignored otherwise). The Imp stage estimates
    // `−2 log L = −2 Σᵢ log ∫ p(yᵢ|η,θ)p(η|θ) dη` by Monte Carlo with a
    // Student-t proposal centred on each subject's EBE.
    /// Number of importance samples per subject. Default 1000. Recommended
    /// 2000–5000 for publication-quality MC SE (cost scales linearly).
    pub is_samples: usize,
    /// Degrees of freedom for the Student-t proposal. Default 5.0 (heavy-tailed
    /// — robust to mild proposal misspecification). Must be ≥ 1.
    pub is_proposal_df: f64,
    /// RNG seed for the IS sampling. `None` falls back to a fixed default so
    /// runs are reproducible across invocations.
    pub is_seed: Option<u64>,
    /// Subjects with normalized effective sample size below this fraction
    /// (ESS / K) are flagged in the result. Default 0.1. Set to 0 to silence
    /// the flag entirely.
    pub is_low_ess_threshold: f64,
    // IMPMAP (Importance Sampling assisted by Mode A Posteriori) options,
    // consumed by the `Impmap` estimating stage. IMPMAP runs a Monte-Carlo EM
    // loop: each iteration re-centers a per-subject importance-sampling proposal
    // at the freshly-computed conditional mode (MAP) and first-order variance,
    // then updates θ/Ω/σ from the importance-weighted posterior moments.
    /// Number of MCEM iterations (M-step parameter updates). Default 200.
    pub impmap_iterations: usize,
    /// Importance samples drawn per subject per iteration (K). Default 300.
    /// Larger K reduces Monte-Carlo noise in the M-step at linear cost.
    pub impmap_samples: usize,
    /// Proposal degrees of freedom. `f64::INFINITY` selects a multivariate
    /// normal proposal (NONMEM's IMPMAP default; parsed from `normal`); a finite
    /// value selects a heavier-tailed Student-t. Default `INFINITY` (MVN).
    pub impmap_proposal_df: f64,
    /// RNG seed for the IMPMAP sampling. `None` falls back to a fixed default so
    /// runs are reproducible across invocations.
    pub impmap_seed: Option<u64>,
    /// Number of terminal iterations whose parameters are averaged to form the
    /// reported estimate (Monte-Carlo variance reduction). Default 50.
    pub impmap_averaging: usize,
    /// Subjects whose normalized effective sample size (ESS / K) falls below
    /// this fraction are flagged as poorly-sampled. Default 0.1.
    pub impmap_low_ess_threshold: f64,
    /// How BLOQ (Below Limit of Quantification) observations are handled.
    /// See [`BloqMethod`]. Defaults to `Drop` (backward-compatible: no effect
    /// when the data has no CENS column).
    pub bloq_method: BloqMethod,
    /// Maximum CG iterations for the Steihaug subproblem solver (trust-region only).
    /// `None` (default) uses a size-adaptive budget of `ceil(sqrt(n_params)).clamp(5, n_params)`,
    /// which is 5 for typical NLME problems (n_params ≈ 7–15) and grows with model size.
    /// `Some(n)` pins the budget to `n` — set to `50` to recover the previous fixed behaviour.
    pub steihaug_max_iters: Option<usize>,
    /// If true (default), use automatically detected mu-referencing to centre
    /// ETA starting points on the current population mean at each outer step.
    /// Set to false to disable for comparison purposes.
    pub mu_referencing: bool,
    /// Number of rayon worker threads used for the per-subject parallel loops
    /// (inner EBE search, SAEM MH steps, SIR weighting, likelihood reductions).
    /// `None` (default) leaves rayon's global pool alone, which means one
    /// worker per logical CPU. `Some(n)` runs the fit inside a scoped local
    /// pool of `n` threads — so the setting is per-call, not process-wide,
    /// and different fits can use different thread counts.
    pub threads: Option<usize>,
    /// Number of independent optimizations to run from perturbed starting values.
    /// `1` (default) is a single run — no behaviour change. When `> 1`, runs are
    /// launched in parallel via rayon; the result with the lowest OFV among
    /// converged runs is returned. Start 0 always uses the exact user initials;
    /// starts 1..n are log-space or additive perturbations of size `start_sigma`.
    /// Useful for models with local minima (nonlinear elimination, full-block omega,
    /// many covariates). Nested rayon parallelism (multi-start × per-subject) is
    /// safe — rayon's work-stealing pool handles it without oversubscription.
    pub n_starts: usize,
    /// Log-space standard deviation of the perturbation applied to initial theta
    /// values for starts 1..n_starts. Log-packed thetas are multiplied by
    /// `exp(N(0, start_sigma))`; identity-packed thetas (negative lower bound)
    /// are shifted by `start_sigma * N(0,1)`. Default `0.3` (≈ 30% CV).
    pub start_sigma: f64,
    /// RNG seed for the multi-start theta perturbations. Independent of
    /// `saem_seed` so that changing the SAEM seed for SAEM convergence does
    /// not silently alter which perturbed starts are tried for FOCE multi-start
    /// runs. Default `None` falls back to `42`.
    pub multi_start_seed: Option<u64>,
    /// Name of the column in the dataset that identifies the occasion for each row.
    /// When `Some`, `read_nonmem_csv` populates `Subject::occasions` / `dose_occasions`
    /// and the inner loop estimates per-occasion kappas alongside the BSV etas.
    /// Requires at least one `kappa` declaration in the model's `[parameters]` block.
    pub iov_column: Option<String>,
    /// Optional cooperative cancellation token. When present and flipped by
    /// another thread, the outer/inner/SAEM/GN loops exit at the next safe
    /// point and `fit()` returns `Err("cancelled by user")`. Default `None`.
    pub cancel: Option<crate::cancel::CancelFlag>,
    /// Keys the user explicitly set, in the order they were applied. Populated
    /// by `parse_fit_options` / `apply_fit_option`. Used by `fit()` to warn
    /// when a key is set that the selected estimation method does not consume.
    pub user_set_keys: Vec<String>,
    /// Inner-loop gradient method. Default [`GradientMethod::Auto`] prefers
    /// AD whenever the crate was built with the `autodiff` feature and the
    /// model has an analytical PK path (`tv_fn` populated); otherwise falls
    /// back to FD. See [`GradientMethod`] for the full contract.
    pub gradient_method: GradientMethod,
    /// How often, in gradient evaluations, to re-solve each subject's inner
    /// EBE loop (η̂ and the FOCE Hessian) during the population gradient
    /// instead of holding it fixed.
    ///
    /// - `0` (default) — never reconverge on non-IOV models: use the cheap
    ///   fixed-EBE analytical/AD gradient.
    /// - `1` — reconverge on every gradient evaluation.
    /// - `N` — reconverge on evals `0, N, 2N, …` and use the cheap fixed-EBE
    ///   gradient in between.
    ///
    /// The reconverged gradient captures the inner-solution response term that
    /// the fixed-EBE gradient omits — the term whose absence stalls SLSQP well
    /// above the derivative-free optimum on ill-conditioned non-IOV fits (see
    /// the `focei-slsqp-fixed-ebe-gradient-bias` note) — at ~5–6× the
    /// per-gradient cost (the inner loop runs once per perturbed component).
    /// A larger `N` amortizes that cost while still periodically correcting the
    /// search direction. IOV models (`n_kappa > 0`) always reconverge and
    /// ignore this setting.
    pub reconverge_gradient_interval: usize,
    /// When `true`, write a per-iteration optimizer trace CSV to a temp file
    /// and store its path in `FitResult::trace_path`. Default: `false`.
    pub optimizer_trace: bool,
    /// Apply an additional scaling layer on top of the existing log/Cholesky
    /// parameterization, dividing each transformed coordinate by its initial
    /// magnitude so they are O(1) when passed to the outer optimizer.
    ///
    /// **Default: `false`** (changed in issue #99). The scaling layer is *not*
    /// trajectory-transparent: although the OFV value is unchanged at any
    /// fixed point, the layer rescales the gradient the optimizer sees, and
    /// that gradient feeds the SLSQP overshoot cap (`cap_slsqp_gradient`), the
    /// quasi-Newton Hessian estimate, and the xtol/ftol termination — all of
    /// which act in the scaled coordinate system, so the *trajectory* and stop
    /// point differ. Because the scaling-enabled path only ever runs on
    /// log/Cholesky-packed coordinates (it auto-disables when any
    /// identity-packed theta is present), dividing by `|log value|` is
    /// counterproductive: a coordinate like `ln(V) = ln(20) ≈ 3` gets scale 3,
    /// so the optimizer's unit step becomes a 3-unit move in log space — an
    /// e³ ≈ 20× multiplicative jump in V. That large step both overshoots and,
    /// through the uniform gradient cap, starves the step in every other
    /// dimension (notably OMEGA), which then halts on xtol ~2.5 OFV units
    /// short of the minimum (issue #99; the earlier SAD_SCEN1 slowdown noted
    /// in `optimize_nlopt` is the same effect). The `false` default reproduces
    /// the well-tested pre-scaling-layer behaviour; `true` is left as an
    /// opt-in for experimentation.
    pub scale_params: bool,
    /// Parameter-scaling strategy for the outer optimizer. When non-`None` this
    /// supersedes [`scale_params`]: `Rescale2` (nlmixr2-style bound-half-width
    /// normalisation) is the recommended setting for gradient-based optimizers
    /// and substantially improves cold-start convergence (see
    /// [`ParameterScaling`]). **Default: `Auto`** — applies `Rescale2` to the
    /// gradient-based optimizers that benefit (`Bfgs`/`Lbfgs`/`NloptLbfgs`/`Slsqp`)
    /// and leaves the derivative-free default `Bobyqa` unscaled (where `Rescale2`
    /// distorts its trust-region model). Set via `[fit_options]` key
    /// `parameter_scaling = none|abs|rescale2` to override.
    pub parameter_scaling: ParameterScaling,
    /// Fraction of subjects allowed to have unconverged EBEs before the outer
    /// optimizer rejects the current parameter step (returns OFV = ∞).  Set to
    /// `1.0` to disable the guard (old behaviour).  Default: `0.1`.
    pub max_unconverged_frac: f64,
    /// Minimum number of observations a subject must have for its EBE to count
    /// toward `max_unconverged_frac`.  Subjects below this threshold are
    /// excluded from the convergence fraction but still run normally.
    /// Default: `2`.
    pub min_obs_for_convergence_check: u32,
    /// Enable the outer-loop stagnation guard. When `true` (default), the
    /// NLopt-based outer optimizers short-circuit once recent evals show
    /// no OFV improvement above 1e-3 over a window of `3*(n+1).max(50)`
    /// evals — letting SLSQP / L-BFGS terminate in microseconds via their
    /// own xtol/ftol instead of burning through the remaining maxeval
    /// budget at full inner-loop cost. Set to `false` to disable when
    /// you want the optimizer to run to its natural termination criterion
    /// (or to `outer_maxiter`), e.g. for debugging or for problems with
    /// very slow-but-real OFV improvements below the stagnation threshold.
    pub stagnation_guard: bool,
    /// When `Some(_)`, call [`crate::suggest_start::inits_from_nca`] with the
    /// selected strategy before the optimizer loop to derive NCA-based starting
    /// values from the data. Useful when the model file's defaults are far from
    /// the truth. `None` (the default) disables it.
    pub inits_from_nca: Option<crate::suggest_start::NcaInit>,
    /// Expression strings for `[data_selection] ignore = ...` / `ignore_subjects`.
    /// Each string may contain `&&`-joined sub-expressions (all must hold).
    /// A record is excluded when any clause evaluates to `true`.
    /// Stored verbatim for logging; compiled to `FilterClause` at read time.
    pub ignore_exprs: Vec<String>,
    /// Expression strings for `[data_selection] accept = ...`.
    /// A record is excluded when any clause evaluates to `false`.
    pub accept_exprs: Vec<String>,
    /// Subject IDs to exclude wholesale (syntactic sugar for `ignore = ID == X`).
    /// Compared as strings against `Subject::id`.
    pub ignore_subjects: Vec<String>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: EstimationMethod::FoceI,
            methods: Vec::new(),
            outer_maxiter: 500,
            outer_gtol: 1e-6,
            inner_maxiter: 200,
            // 1e-5, not the looser 1e-4 that an earlier comment justified as
            // matching "NONMEM's ~3-SIGDIGITS inner loop" — that conflated the
            // *outer* control (NSIG/SIGDIGITS, default 3) with the *inner*
            // conditional precision (SIGL, default ~10 significant digits), which
            // NONMEM runs far tighter. A loose inner tolerance leaves residual
            // noise in each subject's EBE solution, which propagates into the
            // marginal OFV the *outer* optimizer sees. On models with a noisy or
            // flat marginal surface (FD-inner FOCE such as LTBS) that noise made
            // the derivative-free BOBYQA outer optimizer false-converge a few OFV
            // units above the true minimum. Tightening to 1e-5 removes enough of
            // that noise to reach NONMEM's minimum (LTBS now matches to <0.001
            // OFV), at ~1.5x the per-fit cost. Note tighter is not uniformly
            // better: at 1e-6 some ill-conditioned fits (3x3 block-Ω,
            // KA=KE+exp(...) coupling) over-converge the inner Hessian and BOBYQA
            // navigates into a worse basin — 1e-5 sits below the LTBS noise floor
            // while staying clear of that pathology. It does change the converged
            // point versus 1e-4; the previous "no measurable OFV change" claim
            // only held for well-conditioned fits. Override via `inner_tol = ...`
            // in `[fit_options]` (loosen for speed; tighten with care).
            inner_tol: 1e-5,
            // ODE solver tolerances: match OdeSolverOptions::default() so the
            // engine default is unchanged. Opt into tighter accuracy per model
            // via `[fit_options] ode_reltol = ...` (see FitOptions::ode_reltol).
            ode_reltol: 1e-4,
            ode_abstol: 1e-6,
            ode_max_steps: 10_000,
            run_covariance_step: true,
            fd_hessian_step: 1e-2,
            covariance_fallback: CovarianceFallback::None,
            covariance_method: CovarianceMethod::Hessian,
            covariance_ofv_hessian: true,
            interaction: true,
            verbose: true,
            // BOBYQA — derivative-free quadratic trust-region. Chosen as the
            // default because the fixed-EBE FD gradient that SLSQP/L-BFGS rely
            // on is biased on ill-conditioned fits (ODE/PD models, sparse data,
            // Hill-ridge identifiability), and SLSQP can declare convergence
            // hundreds of OFV units above the true minimum. BOBYQA re-evaluates
            // EBEs at every trial point and routinely reaches a lower OFV
            // without the cost of `reconverge_gradient_interval = 1`. See
            // `docs/src/estimation/optimizers.md` for the cefepime and Emax
            // PKPD validations and guidance on when to switch back to SLSQP.
            optimizer: Optimizer::Bobyqa,
            lbfgs_memory: 5,
            global_search: false,
            global_maxeval: 0,
            saem_n_exploration: 150,
            saem_n_convergence: 250,
            saem_n_mh_steps: 20,
            saem_adapt_interval: 50,
            saem_omega_burnin: 20,
            saem_seed: None,
            saem_n_leapfrog: 0,
            gn_lambda: 0.01,
            sir: false,
            sir_samples: 1000,
            sir_resamples: 250,
            sir_seed: None,
            sir_keep_samples: false,
            sir_df: 5.0,
            is_samples: 1000,
            is_proposal_df: 5.0,
            is_seed: None,
            is_low_ess_threshold: 0.1,
            impmap_iterations: 200,
            impmap_samples: 300,
            impmap_proposal_df: f64::INFINITY,
            impmap_seed: None,
            impmap_averaging: 50,
            impmap_low_ess_threshold: 0.1,
            bloq_method: BloqMethod::Drop,
            steihaug_max_iters: None,
            mu_referencing: true,
            threads: None,
            n_starts: 1,
            start_sigma: 0.3,
            multi_start_seed: None,
            iov_column: None,
            cancel: None,
            user_set_keys: Vec::new(),
            gradient_method: GradientMethod::default(),
            reconverge_gradient_interval: 0,
            optimizer_trace: false,
            scale_params: false,
            parameter_scaling: ParameterScaling::Auto,
            max_unconverged_frac: 0.1,
            min_obs_for_convergence_check: 2,
            stagnation_guard: true,
            inits_from_nca: None,
            ignore_exprs: Vec::new(),
            accept_exprs: Vec::new(),
            ignore_subjects: Vec::new(),
        }
    }
}

/// BLOQ (Below Limit of Quantification) handling.
///
/// `Drop` — CENS rows are kept as ordinary observations (no special treatment). If
/// the dataset has no CENS column, every row is treated as quantified and this is
/// equivalent to the pre-M3 behavior.
///
/// `M3` — Beal's M3 method: each BLOQ observation contributes
/// `P(y < LLOQ | θ,η) = Φ((LLOQ - f)/√V)` to the likelihood instead of a
/// Gaussian residual term. LLOQ is read from DV on CENS=1 rows (NONMEM convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloqMethod {
    Drop,
    M3,
}

impl BloqMethod {
    pub fn label(self) -> &'static str {
        match self {
            BloqMethod::Drop => "drop",
            BloqMethod::M3 => "m3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    Bfgs,
    Lbfgs,
    /// NLopt LD_SLSQP — Sequential Least Squares Programming. Gradient-based;
    /// fast per iteration on smooth, well-conditioned analytical PK models where
    /// the fixed-EBE finite-difference gradient is a faithful proxy for the true
    /// gradient. On ill-conditioned fits (ODE/PD models, sparse data, Hill-ridge
    /// identifiability) the fixed-EBE bias can drive SLSQP to declare convergence
    /// hundreds of OFV units above the true minimum — pair with
    /// `reconverge_gradient_interval = 1` if it stalls, or switch to `Bobyqa`
    /// (the default; see `FitOptions::default`).
    Slsqp,
    /// NLopt LD_LBFGS
    NloptLbfgs,
    /// NLopt LD_MMA — Method of Moving Asymptotes
    Mma,
    /// NLopt LN_BOBYQA — derivative-free quadratic interpolation, default outer
    /// optimizer. Re-evaluates the FOCE objective (and the inner EBE loop) at
    /// every trial point, so it never sees the fixed-EBE gradient bias that can
    /// stall gradient-based optimizers; consistently reaches a lower OFV than
    /// SLSQP on ODE/PD models, sparse data, and Hill-ridge problems. Needs more
    /// outer evaluations than SLSQP to triangulate a quadratic from scratch, but
    /// each evaluation is cheap (no FD gradient sweep). See
    /// `docs/src/estimation/optimizers.md` for the cefepime and Emax PKPD
    /// validations behind the default choice.
    Bobyqa,
    /// Newton trust-region with Steihaug CG subproblem (via argmin)
    TrustRegion,
}

/// Parameter-scaling strategy for the outer optimizer. Maps the packed
/// parameter vector into a better-conditioned space before the optimizer sees
/// it (and maps gradients/bounds back). Distinct from the legacy
/// [`FitOptions::scale_params`] bool, which it supersedes when set to a
/// non-`None` value.
///
/// On `two_cpt_oral_cov` (FOCEI, mu-referencing), `Rescale2` + `bfgs` reaches
/// OFV −1198.97 from a cold start — matching nlmixr2 (−1199.24) — where the
/// unscaled gradient-based optimizers stall near −1152/−1192. This mirrors
/// nlmixr2's finding that parameter scaling, not gradient exactness, is the
/// lever for cold-start robustness of *gradient-based* optimizers.
///
/// Crucially, `Rescale2` is **harmful to the derivative-free default `Bobyqa`**
/// (e.g. it drops `emax_pkpd` from OFV −36.76 to −13.51 and `three_cpt_iv` from
/// −730.6 to −715.9): rescaling the trust region of a gradient-free optimizer
/// distorts its quadratic model. Hence the default is [`Auto`](Self::Auto),
/// which applies `Rescale2` only to the gradient-based optimizers that benefit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ParameterScaling {
    /// **Default.** Apply `Rescale2` for the gradient-based optimizers that
    /// benefit from it (`Bfgs`, `Lbfgs`, `NloptLbfgs`, `Slsqp`) and no scaling
    /// otherwise — so the derivative-free `Bobyqa` default (where `Rescale2` is
    /// harmful) and `Mma`/`TrustRegion` are left unscaled, with the legacy
    /// `scale_params` / IOV-auto-enable still applying in that unscaled branch.
    /// `Slsqp` is scaled because the bound-half-width rescaling fixes its
    /// cold-start convergence on IOV models (#335).
    #[default]
    Auto,
    /// No scaling: fall back to the legacy `scale_params` bool (and the IOV+SLSQP
    /// auto-enable). Preserves the pre-`Auto` unscaled behaviour for any optimizer.
    None,
    /// Normalise each coordinate by `|packed value|` (the legacy `compute_scale`
    /// strategy). O(1) for log-packed thetas; 1.0 fallback near zero.
    Abs,
    /// nlmixr2-style: normalise each coordinate by the half-width of its bound
    /// range, mapping it toward `(−1, 1)`. The recommended scaling for
    /// gradient-based optimizers (`bfgs`/`lbfgs`); harmful to `Bobyqa`.
    Rescale2,
}

impl Optimizer {
    pub fn label(self) -> &'static str {
        match self {
            Optimizer::Bfgs => "bfgs",
            Optimizer::Lbfgs => "lbfgs",
            Optimizer::Slsqp => "slsqp",
            Optimizer::NloptLbfgs => "nlopt_lbfgs",
            Optimizer::Mma => "mma",
            Optimizer::Bobyqa => "bobyqa",
            Optimizer::TrustRegion => "trust_region",
        }
    }
}

/// Estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    Foce,
    FoceI,
    FoceGn,
    FoceGnHybrid,
    Saem,
    /// Importance-sampling marginal log-likelihood. Not an estimator — does not
    /// update parameters. Must follow another stage in `methods` (consumes that
    /// stage's params + EBEs + per-subject Hessians as the proposal centre /
    /// scale) and is typically terminal. Reports `−2 log L_IS` on
    /// `FitResult.importance_sampling`, distinct from the Laplace OFV on `ofv`.
    Imp,
    /// Importance Sampling assisted by Mode A Posteriori (NONMEM `METHOD=IMPMAP`).
    /// Unlike [`Imp`](Self::Imp), this *is* an estimator: a Monte-Carlo EM loop
    /// whose E-step re-evaluates each subject's conditional mode and first-order
    /// variance every iteration (as in FOCE/ITS) to center a multivariate-normal
    /// importance-sampling proposal, and whose M-step updates θ/Ω/σ from the
    /// importance-weighted posterior moments.
    Impmap,
}

impl EstimationMethod {
    pub fn label(self) -> &'static str {
        match self {
            EstimationMethod::Foce => "FOCE",
            EstimationMethod::FoceI => "FOCEI",
            EstimationMethod::FoceGn => "FOCE-GN",
            EstimationMethod::FoceGnHybrid => "FOCE-GN-Hybrid",
            EstimationMethod::Saem => "SAEM",
            EstimationMethod::Imp => "IMP",
            EstimationMethod::Impmap => "IMPMAP",
        }
    }
}

impl FitOptions {
    /// Returns the sequence of methods to execute. If `methods` is non-empty it
    /// is returned as-is; otherwise a single-element chain wrapping `method`.
    pub fn method_chain(&self) -> Vec<EstimationMethod> {
        if self.methods.is_empty() {
            vec![self.method]
        } else {
            self.methods.clone()
        }
    }

    /// Check `user_set_keys` against the selected method chain. Returns one
    /// warning per key that isn't consumed by any method in the chain, listing
    /// the method-specific keys that *are* applicable so the user can correct
    /// the mistake. Framework-level keys (covariance/verbose/sir/bloq/threads/
    /// mu_referencing) are omitted from the suggestion list — they apply to
    /// every method and are exposed as top-level arguments in the wrappers.
    pub fn unsupported_keys_warnings(&self) -> Vec<String> {
        if self.user_set_keys.is_empty() {
            return Vec::new();
        }
        let chain = self.method_chain();
        // Applicability = framework keys ∪ (method-specific keys for each
        // stage in the chain). A key is legit as long as *some* stage
        // consumes it.
        let mut applicable: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        applicable.extend(framework_keys().iter().copied());
        for &m in &chain {
            applicable.extend(method_specific_keys(m).iter().copied());
        }
        // Only method-specific keys get surfaced as "available" — listing
        // framework keys here would conflate the two layers.
        let mut method_only: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        for &m in &chain {
            method_only.extend(method_specific_keys(m).iter().copied());
        }
        let chain_label: String = if chain.len() == 1 {
            chain[0].label().to_string()
        } else {
            chain
                .iter()
                .map(|m| m.label())
                .collect::<Vec<_>>()
                .join(" → ")
        };
        let available: Vec<&'static str> = method_only.iter().copied().collect();

        let mut seen = std::collections::HashSet::new();
        let mut warnings = Vec::new();
        for key in &self.user_set_keys {
            // `method` / `methods` select the chain itself — they can't be
            // "wrong for the method" in the way other options can.
            if key == "method" || key == "methods" {
                continue;
            }
            if applicable.contains(key.as_str()) {
                continue;
            }
            if !seen.insert(key.clone()) {
                continue;
            }
            warnings.push(format!(
                "fit option `{}` is not used by method `{}` and will be ignored. \
                 Method-specific options for `{}`: {}",
                key,
                chain_label,
                chain_label,
                available.join(", ")
            ));
        }
        warnings
    }
}

/// Framework-level fit-option keys: consumed by every method and typically
/// exposed as dedicated top-level arguments in the language wrappers
/// (`covariance`, `verbose`, `bloq_method`, `threads`, `sir`, ...). Kept
/// separate from `method_specific_keys` so the "unsupported option" warning
/// can list only method-specific suggestions without conflating the layers.
pub fn framework_keys() -> &'static [&'static str] {
    &[
        "covariance",
        "covariance_method",
        "covariance_fallback",
        "covariance_ofv_hessian",
        "fd_hessian_step",
        "verbose",
        "sir",
        "sir_samples",
        "sir_resamples",
        "sir_seed",
        "sir_keep_samples",
        "sir_df",
        "bloq_method",
        "bloq",
        "mu_referencing",
        "threads",
        "n_starts",
        "start_sigma",
        "multi_start_seed",
        "gradient",
        "gradient_method",
        "iov_column",
        "optimizer_trace",
        "scale_params",
        "parameter_scaling",
        "max_unconverged_frac",
        "min_obs_for_convergence_check",
        "inits_from_nca",
    ]
}

/// Fit-option keys that are meaningful only for a particular estimation
/// method (or family of methods). `method` / `methods` are omitted — those
/// select the chain itself and can't be "wrong for the method". Framework-
/// wide keys live in `framework_keys`.
pub fn method_specific_keys(m: EstimationMethod) -> &'static [&'static str] {
    match m {
        EstimationMethod::Foce | EstimationMethod::FoceI => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "optimizer",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "reconverge_gradient_interval",
        ],
        EstimationMethod::FoceGn => &["maxiter", "inner_maxiter", "inner_tol", "gn_lambda"],
        EstimationMethod::FoceGnHybrid => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "optimizer",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "gn_lambda",
            "reconverge_gradient_interval",
        ],
        EstimationMethod::Saem => &[
            "inner_maxiter",
            "inner_tol",
            "n_exploration",
            "n_convergence",
            "n_mh_steps",
            "n_leapfrog",
            "saem_n_leapfrog",
            "adapt_interval",
            "omega_burnin",
            "seed",
            "saem_seed",
        ],
        EstimationMethod::Imp => &[
            "is_samples",
            "is_proposal_df",
            "is_seed",
            "is_low_ess_threshold",
        ],
        EstimationMethod::Impmap => &[
            "inner_maxiter",
            "inner_tol",
            "impmap_iterations",
            "impmap_samples",
            "impmap_proposal_df",
            "impmap_seed",
            "impmap_averaging",
            "impmap_low_ess_threshold",
        ],
    }
}

/// Trial design specification parsed from [simulation] block
#[derive(Debug, Clone)]
pub struct SimulationSpec {
    pub n_subjects: usize,
    pub dose_amt: f64,
    pub dose_cmt: usize,
    pub obs_times: Vec<f64>,
    pub seed: u64,
    /// Optional per-subject covariates: (name, values) — length must equal n_subjects
    pub covariates: Vec<(String, Vec<f64>)>,
}

/// Full parsed model including simulation spec and fit options
pub struct ParsedModel {
    pub model: CompiledModel,
    pub simulation: Option<SimulationSpec>,
    pub fit_options: FitOptions,
    /// Declarations from the optional `[covariates]` block. `None` when the
    /// block is absent (legacy auto-detect: every non-standard CSV column is a
    /// covariate). `Some` (possibly empty) when present, in which case it is
    /// authoritative — only listed columns are covariates, and each must exist
    /// in the data and be numerically coded. Drives the file-based readers and
    /// the [`FitResult::covariate_table`].
    pub covariate_decls: Option<Vec<CovariateDecl>>,
    /// 1-based source line of each unnamed `[block]` header, keyed by the
    /// lowercased block type (e.g. `"individual_parameters" -> 7`). Used by
    /// `ferx check` to attach a block-level location to diagnostics. Empty when
    /// a model is constructed programmatically rather than parsed from text.
    pub block_lines: std::collections::HashMap<String, usize>,
}

/// Factories that build minimal `CompiledModel` instances for unit tests.
/// Exposed `pub(crate)` (gated on `#[cfg(test)]`) so other modules' tests
/// can construct models without duplicating the boilerplate.
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;
    use std::collections::HashMap;

    /// Build an analytical-PK model (`tv_fn = Some`, `ode_spec = None`).
    pub(crate) fn analytical_model(gradient_method: GradientMethod) -> CompiledModel {
        make_compiled_model(false, gradient_method)
    }

    /// Build an ODE-backed model (`tv_fn = None`, `ode_spec = Some`).
    pub(crate) fn ode_model(gradient_method: GradientMethod) -> CompiledModel {
        make_compiled_model(true, gradient_method)
    }

    fn make_compiled_model(with_ode: bool, gradient_method: GradientMethod) -> CompiledModel {
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptOral,
            error_model: ErrorModel::Additive,
            error_spec: ErrorSpec::Single(ErrorModel::Additive),
            pk_param_fn: Box::new(|_, _, _| PkParams::default()),
            n_theta: 1,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            theta_names: vec!["CL".into()],
            eta_names: vec!["ETA_CL".into()],
            kappa_names: Vec::new(),
            indiv_param_names: vec!["CL".into()],
            indiv_param_partials: IndivParamPartials::empty(),
            default_params: ModelParameters {
                theta: vec![1.0],
                theta_names: vec!["CL".into()],
                theta_lower: vec![0.0],
                theta_upper: vec![f64::INFINITY],
                theta_fixed: vec![false],
                omega: OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_CL".into()]),
                omega_fixed: vec![false],
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["EPS".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            // Analytical models populate tv_fn; ODE models leave it None.
            tv_fn: if with_ode {
                None
            } else {
                Some(Box::new(|_t, _c| vec![1.0]))
            },
            pk_indices: vec![],
            eta_map: vec![],
            pk_idx_f64: vec![],
            sel_flat: vec![],
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            ode_spec: if with_ode {
                Some(crate::ode::OdeSpec {
                    rhs: Box::new(|_y, _p, _t, _dy| {}),
                    n_states: 2,
                    state_names: vec!["depot".into(), "central".into()],
                    readout: crate::ode::OdeReadout::ObsCmt(0),
                    diffusion_var: Vec::new(),
                    init_fn: None,
                    solver_opts: crate::ode::OdeSolverOptions::default(),
                    input_rate: Vec::new(),
                })
            } else {
                None
            },
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec![],
            gradient_method,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_pk_params_match_docs_table() {
        // Locks the per-model required slots to the "Required Parameters" table
        // in docs/src/model-file/structural-model.md (issue #309).
        use PkModel::*;
        let cases: &[(PkModel, &[&str])] = &[
            (OneCptIv, &["cl", "v"]),
            (OneCptOral, &["cl", "v", "ka"]),
            (TwoCptIv, &["cl", "v1", "q", "v2"]),
            (TwoCptOral, &["cl", "v1", "q", "v2", "ka"]),
            (ThreeCptIv, &["cl", "v1", "q2", "v2", "q3", "v3"]),
            (ThreeCptOral, &["cl", "v1", "q2", "v2", "q3", "v3", "ka"]),
        ];
        for (model, expected_names) in cases {
            let req = model.required_pk_params();
            let names: Vec<&str> = req.iter().map(|(_, n)| *n).collect();
            assert_eq!(&names, expected_names, "wrong required names for {model:?}");
            // Every (slot, name) pair must be self-consistent with name_to_index,
            // so the parser's key→slot canonicalisation lines up with this table.
            for (slot, name) in req {
                assert_eq!(
                    PkParams::name_to_index(name),
                    Some(*slot),
                    "slot/name mismatch for `{name}` in {model:?}"
                );
            }
            // f / lagtime are optional and must never appear as required.
            assert!(
                !req.iter()
                    .any(|(s, _)| *s == PK_IDX_F || *s == PK_IDX_LAGTIME),
                "{model:?} must not require f/lagtime"
            );
        }
    }

    #[test]
    fn canonical_name_covers_all_variants() {
        use PkModel::*;
        assert_eq!(OneCptIv.canonical_name(), "one_cpt_iv");
        assert_eq!(OneCptOral.canonical_name(), "one_cpt_oral");
        assert_eq!(TwoCptIv.canonical_name(), "two_cpt_iv");
        assert_eq!(TwoCptOral.canonical_name(), "two_cpt_oral");
        assert_eq!(ThreeCptIv.canonical_name(), "three_cpt_iv");
        assert_eq!(ThreeCptOral.canonical_name(), "three_cpt_oral");
    }

    #[test]
    fn error_spec_has_f_dependent_variance() {
        use std::collections::HashMap;
        // Single endpoint: only additive is f-independent.
        assert!(!ErrorSpec::Single(ErrorModel::Additive).has_f_dependent_variance());
        assert!(ErrorSpec::Single(ErrorModel::Proportional).has_f_dependent_variance());
        assert!(ErrorSpec::Single(ErrorModel::Combined).has_f_dependent_variance());

        // PerCmt: f-dependent if ANY endpoint is non-additive.
        let ep = |em: ErrorModel| EndpointError {
            error_model: em,
            sigma_idx: vec![0],
        };
        let mut all_additive = HashMap::new();
        all_additive.insert(1usize, ep(ErrorModel::Additive));
        all_additive.insert(2usize, ep(ErrorModel::Additive));
        assert!(!ErrorSpec::PerCmt(all_additive).has_f_dependent_variance());

        let mut mixed = HashMap::new();
        mixed.insert(1usize, ep(ErrorModel::Additive));
        mixed.insert(2usize, ep(ErrorModel::Proportional));
        assert!(ErrorSpec::PerCmt(mixed).has_f_dependent_variance());
    }

    #[test]
    fn classify_warning_convergence_is_critical() {
        let w = classify_warning("Outer optimization did not converge");
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert_eq!(w.category, "convergence");
        assert!(w.source_method.is_none());
    }

    #[test]
    fn classify_warning_covariance_is_critical() {
        let w = classify_warning("Covariance step failed");
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert_eq!(w.category, "covariance_step");
    }

    #[test]
    fn classify_warning_dw_is_warning() {
        let w = classify_warning("Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).");
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_eq!(w.category, "dw_autocorrelation");
    }

    #[test]
    fn classify_warning_mu_ref_is_info() {
        let w = classify_warning("mu-ref: CL, V");
        assert_eq!(w.severity, WarningSeverity::Info);
        assert_eq!(w.category, "mu_referencing");
    }

    #[test]
    fn classify_warning_strips_chain_prefix() {
        let w = classify_warning("[FOCEI] Covariance step failed");
        assert_eq!(w.source_method.as_deref(), Some("FOCEI"));
        assert_eq!(w.message, "Covariance step failed");
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert_eq!(w.category, "covariance_step");
    }

    #[test]
    fn classify_warning_unknown_falls_back_to_general() {
        let w = classify_warning("some entirely novel message");
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_eq!(w.category, "general");
    }

    /// Round-trip table covering every literal warning message emitted by the
    /// engine, with the (severity, category) it should classify to.
    ///
    /// This protects the classifier contract from quiet regressions when a
    /// message gets a typo fix or a wording change. If you edit a message
    /// string in `outer_optimizer.rs`, `saem.rs`, `gauss_newton.rs`,
    /// `trust_region.rs`, or `api.rs`, mirror the edit here so the test
    /// keeps reflecting the engine's actual output.
    ///
    /// Messages built by `format!(..)` are exercised with a representative
    /// instantiation: the substrings the classifier matches on must remain
    /// present after interpolation.
    #[test]
    fn classify_warning_roundtrips_every_engine_message() {
        use WarningSeverity::*;
        let table: &[(&str, WarningSeverity, &str)] = &[
            // -- outer_optimizer.rs ----------------------------------------
            (
                "Outer optimization did not converge",
                Critical,
                "convergence",
            ),
            ("Covariance step failed", Critical, "covariance_step"),
            // global_search has two arms: explicit "disabled" is a runtime
            // failure (Warning); a bare mention without "disabled" is
            // informational.
            (
                "global_search disabled: bad seed",
                Warning,
                "optimizer_config",
            ),
            ("global_search reached eval cap", Info, "optimizer_config"),
            ("cancelled by user", Info, "cancelled"),
            // -- gauss_newton.rs -------------------------------------------
            (
                "Gauss-Newton: trust radius collapsed",
                Warning,
                "optimizer_health",
            ),
            (
                "Gauss-Newton: degenerate BHHH Hessian, trust radius collapsed",
                Warning,
                "optimizer_health",
            ),
            (
                "Gauss-Newton: max iterations reached without convergence",
                Critical,
                "convergence",
            ),
            // -- saem.rs ---------------------------------------------------
            (
                "Covariance step failed \u{2014} SEs not available",
                Critical,
                "covariance_step",
            ),
            (
                "saem_n_leapfrog > 0 but HMC is unavailable (requires `autodiff` feature)",
                Info,
                "gradient_fallback",
            ),
            // -- trust_region.rs ------------------------------------------
            (
                "Trust-region did not converge: line search stalled",
                Critical,
                "convergence",
            ),
            // -- api.rs ----------------------------------------------------
            (
                "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).",
                Warning,
                "dw_autocorrelation",
            ),
            (
                "Negative IWRES autocorrelation detected (Durbin-Watson = 2.80).",
                Warning,
                "dw_autocorrelation",
            ),
            (
                "M3 BLOQ handling requires FOCEI semantics",
                Warning,
                "bloq_method",
            ),
            (
                "SIR failed: covariance not positive definite",
                Warning,
                "sir",
            ),
            (
                "SIR requested but covariance matrix is not available",
                Warning,
                "sir",
            ),
            (
                "IMP: 2 subject(s) had ESS = 0 (proposal collapse)",
                Warning,
                "importance_sampling",
            ),
            (
                "LTBS (log(DV) ~ ...): 3 observation(s) with non-positive DV",
                Warning,
                "data_quality",
            ),
            (
                "W_MISSING_DV: 2 observation row(s) (EVID=0) had a missing DV",
                Warning,
                "data_quality",
            ),
            (
                "Stochastic differential equations ([diffusion] / Extended Kalman \
                 Filter) are an EXPERIMENTAL feature: validated only on a small set",
                Warning,
                "experimental",
            ),
            (
                "Neural-network model components ([covariate_nn] / deep compartment \
                 models) are an EXPERIMENTAL feature: validated only on a small set",
                Warning,
                "experimental",
            ),
            (
                "block omega: ETA_CL x ETA_V have mixed lognormal / additive parameterisations",
                Warning,
                "omega_structure",
            ),
            ("mu-ref: CL, V, KA", Info, "mu_referencing"),
            (
                "Multi-start: best result from start 3/8",
                Info,
                "multi_start",
            ),
            (
                "12 threads configured but only 10 subject(s)",
                Info,
                "threads",
            ),
            ("SAEM with more threads than subjects/2", Info, "threads"),
            (
                "Covariance step: 35 parameters \u{2192} n\u{00b2} OFV evaluations",
                Info,
                "covariance_step",
            ),
            (
                "Covariance step: 35 parameters -> n^2 OFV evaluations",
                Info,
                "covariance_step",
            ),
            // format_non_pd_warning path (NonPdHessian variant).
            (
                "Covariance step: Hessian is not positive definite. \
                 Eigenvalues: [8.4000, 2.1000, -0.0100]. SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // Covariance step regularised — present in all three severity tiers.
            (
                "Covariance step regularized: eigenvalue floor applied to FD Hessian \
                 (1 of 3 free-block eigenvalues clipped; min eig = 1.2e-6, floor = 8.4e-14; \
                 severity: minor). Standard errors are likely reliable.",
                Warning,
                "covariance_step",
            ),
            (
                "Covariance step regularized: eigenvalue floor applied to FD Hessian \
                 (3 of 5 free-block eigenvalues clipped; min eig = 1.2e-6, floor = 8.4e-14; \
                 severity: severe). Standard errors are likely unreliable; \
                 SIR-based confidence intervals are recommended.",
                Warning,
                "covariance_step",
            ),
            // Off-diagonal NaN soft warning — Success result, but correlation missing.
            (
                "Covariance step: off-diagonal FD stencil(s) non-finite for theta[CL], sigma[1]. \
                 Cross-partial correlation set to 0; SE for these parameter(s) \
                 may be over-optimistic. Try tuning fd_hessian_step.",
                Warning,
                "covariance_step",
            ),
            // Chain-prefixed off-diagonal warning.
            (
                "[FOCEI] Covariance step: off-diagonal FD stencil(s) non-finite for theta[CL]. \
                 Cross-partial correlation set to 0; SE for these parameter(s) \
                 may be over-optimistic. Try tuning fd_hessian_step.",
                Warning,
                "covariance_step",
            ),
            // Unusable messages introduced in commit 2.
            (
                "Covariance step failed: base OFV is non-finite at convergence \
                 (likely numerical overflow or underflow in model evaluation). \
                 SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            (
                "Covariance step failed: Hessian has ill-conditioned entries for the \
                 following parameter(s) — theta[CL] (non-finite diagonal); \
                 sigma[1] (non-finite off-diagonal). SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // Omega near-singular (tiny positive eigenvalue) — "near-singular" descriptor.
            (
                "Covariance step failed: Omega matrix is near-singular at \
                 convergence (min eigenvalue = 1.2e-10; eigenvalues: [0.5000, 1.2e-10]). \
                 SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // Omega truly non-PD (negative eigenvalue) — "not positive definite" descriptor.
            (
                "Covariance step failed: Omega matrix is not positive definite at \
                 convergence (min eigenvalue = -1.0e-3; eigenvalues: [0.5000, -1.0e-3]). \
                 SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // SIR message that also contains "not positive definite" — must
            // still route to "sir", NOT to covariance_step.
            (
                "SIR failed: covariance not positive definite",
                Warning,
                "sir",
            ),
            // -- chain-prefixed (multi-stage) -----------------------------
            (
                "[FOCEI] Covariance step failed",
                Critical,
                "covariance_step",
            ),
            (
                "[FOCEI] Covariance step: Hessian is not positive definite. \
                 Eigenvalues: [2.1000, -0.0100]. SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            ("[SAEM] mu-ref: CL", Info, "mu_referencing"),
            (
                "[FOCEI] Outer optimization did not converge",
                Critical,
                "convergence",
            ),
        ];

        let mut failures: Vec<String> = Vec::new();
        for (msg, want_sev, want_cat) in table {
            let got = classify_warning(msg);
            if got.severity != *want_sev || got.category != *want_cat {
                failures.push(format!(
                    "  {msg:?} -> expected ({:?}, {:?}), got ({:?}, {:?})",
                    want_sev, want_cat, got.severity, got.category
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "classify_warning round-trip failures:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn is_ode_based_false_for_analytical() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert!(!m.is_ode_based());
    }

    #[test]
    fn is_ode_based_true_for_ode() {
        let m = test_helpers::ode_model(GradientMethod::Auto);
        assert!(m.is_ode_based());
    }

    #[test]
    fn error_spec_single_ignores_cmt() {
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        // Proportional: V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0, regardless of CMT.
        assert!((spec.variance_at(1, 10.0, &[0.1]) - 1.0).abs() < 1e-12);
        assert!((spec.variance_at(7, 10.0, &[0.1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn error_spec_per_cmt_dispatches_model_and_sigma_slice() {
        // Flat sigma vector [prop_pk, add_pd]; CMT=2 uses idx 0 (proportional),
        // CMT=3 uses idx 1 (additive).
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        let sigma = [0.1, 2.0];

        // CMT=2 proportional: (10 * 0.1)^2 = 1.0 (independent of additive sigma).
        assert!((spec.variance_at(2, 10.0, &sigma) - 1.0).abs() < 1e-12);
        // CMT=3 additive: 2.0^2 = 4.0 (independent of prediction).
        assert!((spec.variance_at(3, 10.0, &sigma) - 4.0).abs() < 1e-12);
        assert!((spec.variance_at(3, 999.0, &sigma) - 4.0).abs() < 1e-12);

        // Unregistered CMT → NaN guard.
        assert!(spec.variance_at(1, 10.0, &sigma).is_nan());
    }

    #[test]
    fn error_spec_single_sigma_types_padded_to_n_sigma() {
        // Single proportional with extra declared sigmas: leading slot is
        // proportional, the rest default to additive, length == n_sigma.
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        assert_eq!(
            spec.sigma_types(3),
            vec![
                SigmaType::Proportional,
                SigmaType::Additive,
                SigmaType::Additive
            ]
        );
        // Combined stamps both leading slots.
        let combined = ErrorSpec::Single(ErrorModel::Combined);
        assert_eq!(
            combined.sigma_types(2),
            vec![SigmaType::Proportional, SigmaType::Additive]
        );
    }

    #[test]
    fn error_spec_per_cmt_variance_at_out_of_range_idx_is_nan() {
        // Hand-constructed endpoint whose sigma_idx points past the sigma slice
        // must yield NaN, not panic.
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![5], // out of range for a length-1 sigma vector
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        assert!(spec.variance_at(2, 10.0, &[0.1]).is_nan());
    }

    #[test]
    fn error_spec_dvar_df_matches_legacy_single_formulas() {
        let f = 7.0;
        // Additive: 0. Proportional/Combined: 2*f*sigma_prop^2.
        assert_eq!(
            ErrorSpec::Single(ErrorModel::Additive).dvar_df(1, f, &[0.5]),
            0.0
        );
        assert!(
            (ErrorSpec::Single(ErrorModel::Proportional).dvar_df(1, f, &[0.3]) - 2.0 * f * 0.09)
                .abs()
                < 1e-12
        );
        // Combined uses the first (proportional) sigma.
        assert!(
            (ErrorSpec::Single(ErrorModel::Combined).dvar_df(1, f, &[0.3, 2.0]) - 2.0 * f * 0.09)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn error_spec_per_cmt_dvar_df_and_dlogsigma_dispatch_by_endpoint() {
        // CMT=2 proportional on sigma idx 0; CMT=3 additive on sigma idx 1.
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        let sigma = [0.3, 2.0];
        let f = 7.0;

        // dvar_df: proportional endpoint -> 2*f*0.3^2; additive -> 0.
        assert!((spec.dvar_df(2, f, &sigma) - 2.0 * f * 0.09).abs() < 1e-12);
        assert_eq!(spec.dvar_df(3, f, &sigma), 0.0);

        // dvar_dlogsigma: each sigma only contributes through its own endpoint.
        // sigma 0 (proportional, CMT=2): 2*0.3^2*f^2 at CMT=2, 0 at CMT=3.
        assert!((spec.dvar_dlogsigma(2, 0, f, &sigma) - 2.0 * 0.09 * f * f).abs() < 1e-12);
        assert_eq!(spec.dvar_dlogsigma(3, 0, f, &sigma), 0.0);
        // sigma 1 (additive, CMT=3): 2*2.0^2 at CMT=3, 0 at CMT=2.
        assert!((spec.dvar_dlogsigma(3, 1, f, &sigma) - 2.0 * 4.0).abs() < 1e-12);
        assert_eq!(spec.dvar_dlogsigma(2, 1, f, &sigma), 0.0);
    }

    #[test]
    fn error_spec_per_cmt_sigma_types_maps_each_slot() {
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        assert_eq!(
            spec.sigma_types(2),
            vec![SigmaType::Proportional, SigmaType::Additive]
        );
    }

    #[test]
    fn test_lagtime_name_to_index_and_default() {
        assert_eq!(PkParams::name_to_index("lagtime"), Some(PK_IDX_LAGTIME));
        // NONMEM-style alias maps to the same slot.
        assert_eq!(PkParams::name_to_index("alag"), Some(PK_IDX_LAGTIME));
        assert_eq!(PK_IDX_LAGTIME, 8);
        // Slots 0–8 are named PK params; the rest are ODE structural-param
        // headroom (see MAX_PK_PARAMS docs / issue #122).
        assert!(MAX_PK_PARAMS > PK_IDX_LAGTIME);

        let default = PkParams::default();
        assert_eq!(default.lagtime(), 0.0);
        // F still defaults to 1.0 (unchanged).
        assert_eq!(default.f_bio(), 1.0);
    }

    #[test]
    fn test_lagtime_from_hashmap_primary_and_alias() {
        let mut m = HashMap::new();
        m.insert("lagtime".to_string(), 1.5);
        let p = PkParams::from_hashmap(&m);
        assert_eq!(p.lagtime(), 1.5);

        let mut m_alias = HashMap::new();
        m_alias.insert("alag".to_string(), 2.0);
        let p_alias = PkParams::from_hashmap(&m_alias);
        assert_eq!(p_alias.lagtime(), 2.0);
    }

    /// Guard the SAEM MH-step default. An early value (3) was too low for hard
    /// cold-start surfaces — the chain didn't decorrelate between SAEM outer
    /// iterations, so the single-draw stochastic M-step received sticky
    /// correlated ETAs and locked the population-θ M-step into a degenerate
    /// basin (observed on Emax PKPD: PD-curve thetas pinned to boundary, ~150
    /// OFV units worse than the correct basin). The default was raised to 10,
    /// then to 20 alongside the componentwise eta kernel and the damped Ω
    /// stochastic-approximation step — both added to stop a block (correlated)
    /// Ω collapsing to a near rank-1 correlation matrix (UVM 2-cpt: every
    /// off-diagonal correlation → ~0.99, one variance → 0). The larger default
    /// also sizes the componentwise sweep count (`max(2, n_mh_steps / n_eta)`).
    ///
    /// If a future change drops the default below ~5, re-run both the Emax PKPD
    /// basin regression and the UVM block-Ω collapse regression in the
    /// experiment repo before merging — both fail silently (OFV looks fine;
    /// parameters wrong).
    #[test]
    fn saem_n_mh_steps_default_is_20() {
        let opts = FitOptions::default();
        assert_eq!(
            opts.saem_n_mh_steps, 20,
            "saem_n_mh_steps default changed — see comment above this test \
             for the basin-trap and block-Ω-collapse regression rationale \
             before adjusting."
        );
    }

    // ── small pure helpers ───────────────────────────────────────────────────

    /// Bare subject with no doses/observations; tests mutate the fields they
    /// exercise.
    fn bare_subject(id: &str) -> Subject {
        Subject {
            id: id.to_string(),
            doses: Vec::new(),
            obs_times: Vec::new(),
            obs_raw_times: Vec::new(),
            observations: Vec::new(),
            obs_cmts: Vec::new(),
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: Vec::new(),
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: Vec::new(),
        }
    }

    fn dose(ss: bool) -> DoseEvent {
        DoseEvent {
            time: 0.0,
            amt: 100.0,
            cmt: 1,
            rate: 0.0,
            duration: 0.0,
            ss,
            ii: 0.0,
        }
    }

    #[test]
    fn subject_has_bloq_reflects_cens_flags() {
        let mut s = bare_subject("1");
        s.cens = vec![0, 0];
        assert!(!s.has_bloq());
        s.cens = vec![0, 1];
        assert!(s.has_bloq());
    }

    #[test]
    fn subject_has_ss_doses_detects_steady_state_flag() {
        let mut s = bare_subject("1");
        s.doses = vec![dose(false), dose(false)];
        assert!(!s.has_ss_doses());
        s.doses.push(dose(true));
        assert!(s.has_ss_doses());
    }

    #[test]
    fn subject_pk_only_cov_falls_back_to_static_map() {
        let mut s = bare_subject("1");
        s.covariates.insert("WT".to_string(), 70.0);
        // No per-EVID-2 snapshots → static map.
        assert_eq!(s.pk_only_cov(0).get("WT"), Some(&70.0));
        // With a snapshot at index 0 → that snapshot.
        let mut snap = HashMap::new();
        snap.insert("WT".to_string(), 80.0);
        s.pk_only_covariates = vec![snap];
        assert_eq!(s.pk_only_cov(0).get("WT"), Some(&80.0));
        // Out-of-range index → fall back to static map.
        assert_eq!(s.pk_only_cov(5).get("WT"), Some(&70.0));
    }

    #[test]
    fn prune_irrelevant_tv_covariates_clears_only_constant_referenced_covs() {
        // Subject A: the referenced covariate (WT) is constant across snapshots
        // but an unreferenced one (DAY) varies → snapshots should be pruned.
        let mut a = bare_subject("A");
        a.covariates = HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 1.0)]);
        a.obs_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 1.0)]),
            HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 2.0)]),
        ];
        // Subject B: the referenced covariate (WT) genuinely varies → keep.
        let mut b = bare_subject("B");
        b.covariates = HashMap::from([("WT".to_string(), 70.0)]);
        b.obs_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 90.0)]),
        ];

        let mut pop = Population {
            subjects: vec![a, b],
            covariate_names: vec!["WT".to_string(), "DAY".to_string()],
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let pruned = pop.prune_irrelevant_tv_covariates(&["WT".to_string()]);
        assert_eq!(pruned, 1, "only subject A should be pruned");
        assert!(
            pop.subjects[0].obs_covariates.is_empty(),
            "A pruned to fast path"
        );
        assert!(
            !pop.subjects[1].obs_covariates.is_empty(),
            "B keeps TV snapshots"
        );
    }

    #[test]
    fn covariate_kind_label_strings() {
        assert_eq!(CovariateKind::Continuous.label(), "continuous");
        assert_eq!(CovariateKind::Categorical.label(), "categorical");
    }

    #[test]
    fn scaling_spec_requires_fd_is_always_false() {
        assert!(!ScalingSpec::None.requires_fd());
        assert!(!ScalingSpec::ScalarScale(1000.0).requires_fd());
        assert!(!ScalingSpec::PerCmt(HashMap::new()).requires_fd());
    }

    #[test]
    fn estimation_method_label_covers_all_variants() {
        assert_eq!(EstimationMethod::Foce.label(), "FOCE");
        assert_eq!(EstimationMethod::FoceI.label(), "FOCEI");
        assert_eq!(EstimationMethod::FoceGn.label(), "FOCE-GN");
        assert_eq!(EstimationMethod::FoceGnHybrid.label(), "FOCE-GN-Hybrid");
        assert_eq!(EstimationMethod::Saem.label(), "SAEM");
        assert_eq!(EstimationMethod::Imp.label(), "IMP");
    }

    #[test]
    fn model_parameters_has_any_fixed_tracks_every_flag_vec() {
        let mut mp = test_helpers::analytical_model(GradientMethod::Fd).default_params;
        mp.theta_fixed = vec![false];
        mp.omega_fixed = vec![false];
        mp.sigma_fixed = vec![false];
        mp.kappa_fixed = Vec::new();
        assert!(!mp.has_any_fixed());
        mp.sigma_fixed = vec![true];
        assert!(mp.has_any_fixed());
    }

    #[test]
    fn pk_params_from_hashmap_maps_named_fields_and_aliases() {
        let map = HashMap::from([
            ("cl".to_string(), 5.0),
            ("v1".to_string(), 30.0), // v1 alias → V index
            ("q".to_string(), 2.0),
            ("v2".to_string(), 50.0),
            ("ka".to_string(), 1.2),
            ("f".to_string(), 0.8),
            ("q3".to_string(), 0.5),
            ("v3".to_string(), 100.0),
        ]);
        let p = PkParams::from_hashmap(&map);
        assert_eq!(p.values[PK_IDX_CL], 5.0);
        assert_eq!(p.values[PK_IDX_V], 30.0);
        assert_eq!(p.values[PK_IDX_Q], 2.0);
        assert_eq!(p.values[PK_IDX_V2], 50.0);
        assert_eq!(p.values[PK_IDX_KA], 1.2);
        assert_eq!(p.values[PK_IDX_F], 0.8);
        assert_eq!(p.values[PK_IDX_Q3], 0.5);
        assert_eq!(p.values[PK_IDX_V3], 100.0);
    }
}
