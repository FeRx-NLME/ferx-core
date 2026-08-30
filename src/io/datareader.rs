use crate::io::filter_expr::{FilterClause, RowContext};
use crate::types::{
    CovariateDecl, CovariateRow, CovariateTable, DoseEvent, ExclusionSummary, Population, RateMode,
    Subject,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Compiled data-selection filter built from `FitOptions` ignore/accept fields.
/// Passed into `read_nonmem_csv_impl` so filtering happens at read time.
pub struct SelectionFilter {
    pub ignore: Vec<FilterClause>,
    pub accept: Vec<FilterClause>,
    /// Subject IDs to exclude wholesale (from `ignore_subjects`).
    pub ignore_subject_ids: Vec<String>,
}

impl SelectionFilter {
    /// Build from the raw expression strings stored in `FitOptions`.
    /// Returns `Err` if any expression fails to parse.
    pub fn from_opts(
        ignore_exprs: &[String],
        accept_exprs: &[String],
        ignore_subjects: &[String],
    ) -> Result<Self, String> {
        let ignore = ignore_exprs
            .iter()
            .map(|s| FilterClause::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        let accept = accept_exprs
            .iter()
            .map(|s| FilterClause::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SelectionFilter {
            ignore,
            accept,
            ignore_subject_ids: ignore_subjects.to_vec(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.ignore.is_empty() && self.accept.is_empty() && self.ignore_subject_ids.is_empty()
    }

    /// De-duplicated, lowercased covariate column names referenced by any
    /// ignore/accept clause (standard NONMEM columns excluded). The declared
    /// `[covariates]` reader uses this to guarantee a filtered column is read
    /// into each subject's covariate map even when the model file did not
    /// declare it — otherwise `ignore = STUDY == 2` against an undeclared
    /// `STUDY` column would silently never fire.
    pub fn referenced_covariate_columns(&self) -> Vec<String> {
        let mut cols: Vec<String> = Vec::new();
        for clause in self.ignore.iter().chain(self.accept.iter()) {
            for c in clause.covariate_columns() {
                if !cols.iter().any(|existing| existing == c) {
                    cols.push(c.to_string());
                }
            }
        }
        cols
    }

    /// True when any ignore/accept clause compares a covariate column as a raw
    /// string, so the reader must build the per-row `str_covariates` map. Lets a
    /// purely numeric filter (the common case) skip that per-row allocation.
    pub fn needs_str_covariates(&self) -> bool {
        self.ignore
            .iter()
            .chain(self.accept.iter())
            .any(|c| c.needs_str_covariates())
    }

    /// Returns `(excluded, which)`:
    /// - `excluded = true` when the row should be dropped.
    /// - `which` is the source string of the first clause that fired (for logging).
    ///
    /// Checks short-circuit on the first match, so a record is attributed to the
    /// first rule that excludes it. A rule that only ever matches records already
    /// removed by an earlier rule therefore never appears in the fired-condition
    /// summary — see `docs/model-file/data-selection.qmd`.
    pub fn should_exclude(&self, ctx: &RowContext<'_>) -> (bool, Option<String>) {
        // 1. ignore_subjects shorthand.
        if self.ignore_subject_ids.iter().any(|id| id == ctx.id) {
            return (true, Some(format!("ignore_subjects: {}", ctx.id)));
        }
        // 2. ignore clauses (any match → excluded).
        for clause in &self.ignore {
            if clause.eval(ctx) {
                return (true, Some(format!("ignore: {}", clause.source)));
            }
        }
        // 3. accept clauses (all must pass → if any fails, excluded).
        for clause in &self.accept {
            if !clause.eval(ctx) {
                return (true, Some(format!("accept: {}", clause.source)));
            }
        }
        (false, None)
    }
}

/// Per-subject exclusion counts returned by `parse_subject` when a filter is active.
pub(crate) struct SubjectExclusion {
    pub n_obs_excluded: usize,
    pub n_dose_excluded: usize,
    /// Records excluded that are neither scored obs nor doses (EVID 2/3, or
    /// missing-DV obs).
    pub n_other_excluded: usize,
    /// Sources that matched at least one row ("ignore: DV < 0.001", etc.).
    pub fired: Vec<String>,
}

/// Leading text of the "declared covariate column absent from data" error.
/// Shared so the `ferx check` layer can classify the reader's error into the
/// right diagnostic code without matching on the full (formatted) message.
pub(crate) const ERR_COV_MISSING_COLUMNS: &str =
    "[covariates]: declared covariate column(s) not found in data";
/// Leading text of the "declared covariate value is not numeric" error.
pub(crate) const ERR_COV_NON_NUMERIC: &str = "[covariates]: non-numeric value";

/// Wall-clock gap inserted between reset-delimited occasion segments when a
/// subject's TIME column restarts (see the segmentation logic in
/// `parse_subject`). The reset zeros every compartment at the boundary, so no
/// drug carries across the gap and its magnitude is numerically irrelevant
/// (it cancels in every dose/observation time difference within a segment); a
/// small positive value simply keeps the two occasions from colliding on the
/// sorted absolute timeline.
const RESET_SEGMENT_GAP: f64 = 1.0;

/// True when a CSV cell represents a missing value (blank / `.` / `NA` / `NaN`).
/// NONMEM convention uses `.` for missing.
fn is_missing_cell(s: &str) -> bool {
    let t = s.trim();
    t.is_empty() || t == "." || t.eq_ignore_ascii_case("na") || t.eq_ignore_ascii_case("nan")
}

/// Read a NONMEM-format CSV file into a Population.
///
/// Expected columns (case-insensitive):
///   ID, TIME, DV, EVID, AMT, CMT, RATE, MDV, II, SS, CENS, [covariates...]
///
/// EVID: 0=observation, 1=dose, 2=other event (covariate change),
///       3=system reset (zero all compartments), 4=reset + dose
/// MDV: 1=missing dependent variable
/// CENS: 1=observation is below LLOQ (DV carries LLOQ), -1=above ULOQ (DV carries ULOQ); 0 otherwise
///
/// `iov_column`: when `Some(name)`, that column is read as the occasion index
/// (integer) and stored in `Subject::occasions` / `Subject::dose_occasions`.
/// The column is excluded from the covariate auto-detection list.
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
) -> Result<Population, String> {
    read_nonmem_csv_mapped(path, covariate_columns, iov_column, &[])
}

/// Like [`read_nonmem_csv`] but with a `[data]` canonical-role → header
/// remapping (#730). `column_map` entries are `(canonical_role, actual_header)`;
/// an empty slice is the no-remap identity used by the public wrapper.
pub(crate) fn read_nonmem_csv_mapped(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    column_map: &[(String, String)],
) -> Result<Population, String> {
    read_nonmem_csv_routed(
        path,
        covariate_columns,
        None,
        &[],
        iov_column,
        None,
        &ObsRouting::default(),
        column_map,
    )
    .map(|(pop, _)| pop)
}

/// Read a NONMEM-format CSV with a `[covariates]` declaration.
///
/// `decls` are the declared covariates: each must exist as a column and be
/// numerically coded (a non-numeric value is a hard error, not a silent `0.0`),
/// and they populate the returned [`CovariateTable`].
///
/// `extra_columns` are covariates *used by the model but not declared*. They are
/// still read into the [`Population`] (leniently, like the auto-detect path) so
/// the model works, but they are not strictly validated and do not appear in the
/// table. The parser emits a warning recommending they be declared.
///
/// The table echoes the declared columns: one row per input record (including
/// dose / EVID rows), with `f64::NAN` for missing values.
pub fn read_nonmem_csv_with_covariates(
    path: &Path,
    decls: &[CovariateDecl],
    extra_columns: &[String],
    iov_column: Option<&str>,
) -> Result<(Population, CovariateTable), String> {
    read_nonmem_csv_with_covariates_mapped(path, decls, extra_columns, iov_column, &[])
}

/// Build the covariate-column read set: declared names first (so the covariate
/// table's column order matches the declaration), then each `extra` column not
/// already present. Dedup is case-sensitive, matching the declared spelling.
fn declared_union(decls: &[CovariateDecl], extra: &[String]) -> Vec<String> {
    let mut union: Vec<String> = decls.iter().map(|d| d.name.clone()).collect();
    for c in extra {
        if !union.iter().any(|n| n == c) {
            union.push(c.clone());
        }
    }
    union
}

/// Ensure every covariate column referenced by a `[data_selection]` filter is
/// present in `cols`, so a filter on an otherwise-unread column still fires.
/// Case-insensitive dedup: referenced names are lowercased, declared names may
/// not be. No-op when `filter` is `None`.
fn augment_with_filter(cols: &mut Vec<String>, filter: Option<&SelectionFilter>) {
    if let Some(f) = filter {
        for c in f.referenced_covariate_columns() {
            if !cols.iter().any(|n| n.eq_ignore_ascii_case(&c)) {
                cols.push(c);
            }
        }
    }
}

/// Like [`read_nonmem_csv_with_covariates`] but with a `[data]` column
/// remapping (#730). See [`read_nonmem_csv_mapped`].
pub(crate) fn read_nonmem_csv_with_covariates_mapped(
    path: &Path,
    decls: &[CovariateDecl],
    extra_columns: &[String],
    iov_column: Option<&str>,
    column_map: &[(String, String)],
) -> Result<(Population, CovariateTable), String> {
    // Population reads the union of declared + referenced-but-undeclared columns,
    // declared first so the table's column order matches the declaration.
    let (pop, table) = read_nonmem_csv_routed(
        path,
        None,
        Some(decls),
        extra_columns,
        iov_column,
        None,
        &ObsRouting::default(),
        column_map,
    )?;
    Ok((
        pop,
        table.expect("covariate table is built whenever table_decls is Some"),
    ))
}

/// Like [`read_nonmem_csv`] but applies `[data_selection]` filtering at read time.
/// Called from `api::read_population_for` when `FitOptions` carries selection rules.
pub fn read_nonmem_csv_filtered(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    filter: &SelectionFilter,
) -> Result<Population, String> {
    read_nonmem_csv_filtered_mapped(path, covariate_columns, iov_column, filter, &[])
}

/// Like [`read_nonmem_csv_filtered`] but with a `[data]` column remapping
/// (#730). See [`read_nonmem_csv_mapped`].
pub(crate) fn read_nonmem_csv_filtered_mapped(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    filter: &SelectionFilter,
    column_map: &[(String, String)],
) -> Result<Population, String> {
    // When an explicit covariate list is supplied, `read_nonmem_csv_routed` makes
    // sure every covariate the filter references is in it — otherwise a filtered
    // column outside the list would not be read and the condition would silently
    // never fire. (With `None`, the auto-detect path already reads every
    // non-standard column, so no augmentation is needed.)
    read_nonmem_csv_routed(
        path,
        covariate_columns,
        None,
        &[],
        iov_column,
        Some(filter),
        &ObsRouting::default(),
        column_map,
    )
    .map(|(pop, _)| pop)
}

/// Like [`read_nonmem_csv_with_covariates`] but applies `[data_selection]` filtering.
pub fn read_nonmem_csv_with_covariates_filtered(
    path: &Path,
    decls: &[CovariateDecl],
    extra_columns: &[String],
    iov_column: Option<&str>,
    filter: &SelectionFilter,
) -> Result<(Population, CovariateTable), String> {
    read_nonmem_csv_with_covariates_filtered_mapped(
        path,
        decls,
        extra_columns,
        iov_column,
        filter,
        &[],
    )
}

/// Like [`read_nonmem_csv_with_covariates_filtered`] but with a `[data]` column
/// remapping (#730). See [`read_nonmem_csv_mapped`].
pub(crate) fn read_nonmem_csv_with_covariates_filtered_mapped(
    path: &Path,
    decls: &[CovariateDecl],
    extra_columns: &[String],
    iov_column: Option<&str>,
    filter: &SelectionFilter,
    column_map: &[(String, String)],
) -> Result<(Population, CovariateTable), String> {
    // `read_nonmem_csv_routed` also reads any covariate column referenced by an
    // ignore/accept clause but never declared — without that, a filter on an
    // undeclared column would silently never fire on this path (the declared
    // union would lack the column, so it would be absent from `locf_state`).
    let (pop, table) = read_nonmem_csv_routed(
        path,
        None,
        Some(decls),
        extra_columns,
        iov_column,
        Some(filter),
        &ObsRouting::default(),
        column_map,
    )?;
    Ok((
        pop,
        table.expect("covariate table is built whenever table_decls is Some"),
    ))
}

/// How an `EVID=0`, `MDV=0` record whose `DV` cell is missing (`.` / `NA` /
/// blank) is read.
///
/// The two readings are both right, for different callers: when the `DV` is an
/// *input* (fitting) a missing one means "nothing to score here"; when it is the
/// *output* (simulation) it means "produce a value here" (#957).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MissingDvPolicy {
    /// Skip the row and count it for the single `W_MISSING_DV` summary (#258) —
    /// a forgotten `MDV=1` must never inject a phantom `DV=0` into the
    /// likelihood. The default, and the only policy on every fitting path.
    #[default]
    Skip,
    /// Keep the row as a design point: the sampling time is real, only the
    /// observation is absent because it has not been generated yet. The DV is a
    /// placeholder (`NaN` for a Gaussian row, `0` for an integer-coded one) and
    /// is overwritten by the simulated value. Used by the simulation readers
    /// (#957) so a `DV = .` template — the natural way to write a design —
    /// simulates instead of returning zero rows. `MDV=1` still excludes the row:
    /// that is the user explicitly saying it is not an observation.
    KeepAsDesign,
}

/// How each `EVID=0` observation row is turned into an observation record:
/// which non-Gaussian [`ObsRecord`](crate::types::ObsRecord) variant the row's
/// CMT routes to, plus the missing-DV policy. Empty sets ⇒ every observation row
/// takes the Gaussian parallel-Vec path (the all-Gaussian default).
///
/// The three sets must be pairwise disjoint — a CMT has exactly one endpoint kind
/// (§8.1: routing is by the CMT's declared endpoint, never guessed from the DV).
/// [`ObsRouting::validate`] enforces disjointness before any row is read.
///
/// Phase 4.0 introduces the discrete/count routing but no parser yet populates
/// those sets; `discrete`/`count` are reachable only through
/// [`read_nonmem_csv_filtered_routed`] (used by the reader unit tests). The
/// production `api::read_population_for` path still builds a TTE-only routing.
#[derive(Debug, Clone, Default)]
pub(crate) struct ObsRouting {
    /// CMTs whose rows become `ObsRecord::Event` (TTE / RTTE; `survival` feature).
    pub tte: HashSet<usize>,
    /// CMTs whose integer-DV rows become `ObsRecord::DiscreteState`
    /// (binary / ordinal / Markov state index).
    pub discrete: HashSet<usize>,
    /// CMTs whose non-negative-integer-DV rows become `ObsRecord::Count`
    /// (Poisson / negative-binomial).
    pub count: HashSet<usize>,
    /// What a missing `DV` on a scored row means — see [`MissingDvPolicy`].
    pub missing_dv: MissingDvPolicy,
    /// Placeholder DV code to write on an integer-coded design-point row, per
    /// CMT. Only consulted under [`MissingDvPolicy::KeepAsDesign`]; a CMT absent
    /// from the map falls back to `0`. Endpoints whose declared `state_codes`
    /// are not 0-based (e.g. `1`/`2`) would otherwise get an out-of-range `0`
    /// placeholder that no `state_codes` lookup can map back to a generator
    /// index (#957 review).
    pub design_states: HashMap<usize, usize>,
}

impl ObsRouting {
    /// Routing for a model with TTE and/or binary/categorical endpoints: `tte` →
    /// `ObsRecord::Event`, `discrete` → `ObsRecord::DiscreteState` (binary/categorical,
    /// #760). Count endpoints are not produced yet, so that set stays empty. Pass an
    /// empty `discrete` set for the pre-Phase-4.0 TTE-only behaviour.
    pub(crate) fn tte_and_discrete(tte: &HashSet<usize>, discrete: &HashSet<usize>) -> Self {
        Self {
            tte: tte.clone(),
            discrete: discrete.clone(),
            ..Default::default()
        }
    }

    /// Register the per-CMT placeholder DV code used for integer-coded design
    /// rows (builder form). `codes` maps a CMT to the DV value written when its
    /// `DV` cell is missing under [`MissingDvPolicy::KeepAsDesign`] — normally
    /// the endpoint's first declared state code, so the placeholder is always a
    /// value the endpoint can decode. CMTs left out keep the `0` default.
    pub(crate) fn with_design_states(mut self, codes: HashMap<usize, usize>) -> Self {
        self.design_states = codes;
        self
    }

    /// Set the missing-DV policy (builder form), leaving the routing sets alone.
    /// `api::read_population_for_simulation` uses this to read a `DV = .` design
    /// template as sampling times rather than as skipped rows (#957).
    pub(crate) fn with_missing_dv(mut self, policy: MissingDvPolicy) -> Self {
        self.missing_dv = policy;
        self
    }

    /// The integer-coded non-Gaussian endpoint kind (discrete-state or count) a
    /// CMT routes to, if any. `None` ⇒ the CMT is TTE or Gaussian. The routing
    /// sets are pairwise disjoint (validated up front), so this is unambiguous.
    fn integer_kind(&self, cmt: usize) -> Option<IntDvKind> {
        if self.discrete.contains(&cmt) {
            Some(IntDvKind::DiscreteState)
        } else if self.count.contains(&cmt) {
            Some(IntDvKind::Count)
        } else {
            None
        }
    }

    /// Reject a CMT declared under more than one endpoint kind — that would make
    /// row routing ambiguous (one endpoint per CMT, §8.1).
    fn validate(&self) -> Result<(), String> {
        let overlap =
            |a: &HashSet<usize>, b: &HashSet<usize>| a.iter().find(|c| b.contains(c)).copied();
        if let Some(cmt) = overlap(&self.tte, &self.discrete) {
            return Err(format!(
                "CMT={cmt} is routed to both a TTE and a discrete-state endpoint; \
                 each CMT may have only one endpoint type"
            ));
        }
        if let Some(cmt) = overlap(&self.tte, &self.count) {
            return Err(format!(
                "CMT={cmt} is routed to both a TTE and a count endpoint; \
                 each CMT may have only one endpoint type"
            ));
        }
        if let Some(cmt) = overlap(&self.discrete, &self.count) {
            return Err(format!(
                "CMT={cmt} is routed to both a discrete-state and a count endpoint; \
                 each CMT may have only one endpoint type"
            ));
        }
        Ok(())
    }
}

/// An integer-coded non-Gaussian observation endpoint: a discrete-state index
/// (binary / ordinal / Markov state — unbounded `usize`) or a `u32` count
/// (Poisson / negative-binomial). Selects the DV upper bound and the error
/// wording in [`checked_integer_dv`].
#[derive(Debug, Clone, Copy)]
enum IntDvKind {
    DiscreteState,
    Count,
}

impl IntDvKind {
    fn label(self) -> &'static str {
        match self {
            IntDvKind::DiscreteState => "discrete-state",
            IntDvKind::Count => "count",
        }
    }

    fn unit(self) -> &'static str {
        match self {
            IntDvKind::DiscreteState => "state index",
            IntDvKind::Count => "count",
        }
    }

    /// Exclusive upper bound on the rounded DV: a value `>=` this overflows the
    /// `as usize` / `as u32` cast and is rejected (no domain cap in Phase 4.0 —
    /// the bound only rejects values the cast could not represent). `type::MAX as
    /// f64 + 1.0` is exact on both 32- and 64-bit: when `type::MAX` is
    /// f64-representable the `+1` lands on the next integer; when it rounds up
    /// (`usize::MAX` on 64-bit: `2^64 - 1` → `2^64`) the `+1` is absorbed, leaving
    /// exactly the first saturating value. So `u32::MAX` / `usize::MAX` themselves
    /// are accepted and only genuine overflow is rejected. Non-finite DVs are
    /// rejected earlier.
    fn exclusive_max(self) -> f64 {
        match self {
            IntDvKind::DiscreteState => usize::MAX as f64 + 1.0,
            IntDvKind::Count => u32::MAX as f64 + 1.0,
        }
    }

    /// Human-readable valid-range suffix for the out-of-range message; empty for
    /// the state index (its `usize::MAX` bound is not a meaningful domain limit).
    fn range_hint(self) -> &'static str {
        match self {
            IntDvKind::DiscreteState => "",
            IntDvKind::Count => " (0..=4294967295)",
        }
    }
}

/// Validate a non-Gaussian integer-coded DV (discrete-state index or count) and
/// return the rounded, non-negative magnitude. Rejects — with a caller-shaped
/// message — non-finite, fractional, negative, and out-of-range (cast-overflowing)
/// DVs (§8.1 integer-code rule, #192).
///
/// A **missing** DV (`.`/`NA`/blank) is skipped by the caller as MDV=1 (#258) and
/// never reaches here. This function is the guard that stops the three silent
/// mis-records a raw `dv.round() as usize`/`as u32` cast would otherwise produce:
/// a `.`-coerced `0.0` → phantom `state:0`/`count:0`; an `inf` → saturated
/// `usize::MAX`/`u32::MAX`; a `NaN` (whose every comparison is false) → `0`.
fn checked_integer_dv(
    dv: f64,
    kind: IntDvKind,
    id: &str,
    cmt: usize,
    time: f64,
) -> Result<f64, String> {
    let (label, unit) = (kind.label(), kind.unit());
    if !dv.is_finite() {
        return Err(format!(
            "Subject {id}: {label} endpoint CMT={cmt} has non-finite DV={dv} at \
             TIME={time}; the DV must be a non-negative integer {unit}"
        ));
    }
    let dv_rounded = dv.round();
    if (dv - dv_rounded).abs() > 1e-9 {
        return Err(format!(
            "Subject {id}: {label} endpoint CMT={cmt} has non-integer DV={dv} at \
             TIME={time}; the DV must be a non-negative integer {unit}"
        ));
    }
    if dv_rounded < 0.0 || dv_rounded >= kind.exclusive_max() {
        return Err(format!(
            "Subject {id}: {label} endpoint CMT={cmt} has out-of-range DV={dv} at \
             TIME={time}; the DV must be a non-negative integer {unit}{hint}",
            hint = kind.range_hint()
        ));
    }
    Ok(dv_rounded)
}

/// Covariate-free shorthand for [`read_nonmem_csv_routed`]. Phase 4.0 exposes it
/// so the discrete-state / count routing can be exercised directly from unit
/// tests (no parser produces those sets yet). Disjointness is validated inside
/// [`read_nonmem_csv_impl`].
#[cfg(test)]
pub(crate) fn read_nonmem_csv_filtered_routed(
    path: &Path,
    routing: &ObsRouting,
) -> Result<Population, String> {
    read_nonmem_csv_impl(path, None, None, None, None, routing, &[]).map(|(pop, _)| pop)
}

/// The one reader every wrapper above delegates to: it builds the covariate
/// read set (declared union, or the explicit lenient list) and augments it with
/// any `[data_selection]`-referenced column before calling
/// [`read_nonmem_csv_impl`].
///
/// `decls` is `Some` on the strict `[covariates]` path — declared columns are
/// validated and a [`CovariateTable`] is returned; `None` returns no table and
/// reads `covariate_columns` leniently (`None` in turn is auto-detect).
/// `routing` carries both the per-CMT endpoint routing and the missing-DV
/// policy, so a caller that needs a non-default policy (simulation, #957) can
/// reach every covariate/filter combination through this single entry point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_nonmem_csv_routed(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    decls: Option<&[CovariateDecl]>,
    extra_columns: &[String],
    iov_column: Option<&str>,
    filter: Option<&SelectionFilter>,
    routing: &ObsRouting,
    column_map: &[(String, String)],
) -> Result<(Population, Option<CovariateTable>), String> {
    // Declared covariates come first (so the table's column order matches the
    // declaration); otherwise the caller's explicit list, or `None` for
    // auto-detect. Either way, a filter's referenced columns must be read or the
    // condition would silently never fire.
    let cols: Option<Vec<String>> = match decls {
        Some(d) => {
            let mut union = declared_union(d, extra_columns);
            augment_with_filter(&mut union, filter);
            Some(union)
        }
        None => covariate_columns.map(|cols| {
            let mut v: Vec<String> = cols.iter().map(|s| s.to_string()).collect();
            augment_with_filter(&mut v, filter);
            v
        }),
    };
    let cols_ref: Option<Vec<&str>> = cols
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    read_nonmem_csv_impl(
        path,
        cols_ref.as_deref(),
        iov_column,
        decls,
        filter,
        routing,
        column_map,
    )
}

/// Shared CSV reader. `table_decls`, when `Some`, requests building a
/// [`CovariateTable`] over exactly those declared covariates — each must exist
/// as a column and is validated as numeric (non-numeric → hard error). The
/// columns in `covariate_columns` (a superset, including referenced-but-
/// undeclared covariates) are read into the [`Population`] leniently. `None` on
/// both is the legacy auto-detect [`read_nonmem_csv`] path.
///
/// `routing`: per-CMT non-Gaussian endpoint routing (`ObsRecord::Event` /
/// `DiscreteState` / `Count`). Default (all sets empty) ⇒ every observation row
/// takes the Gaussian parallel-Vec path. Disjointness is validated up front.
fn read_nonmem_csv_impl(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    table_decls: Option<&[CovariateDecl]>,
    filter: Option<&SelectionFilter>,
    routing: &ObsRouting,
    column_map: &[(String, String)],
) -> Result<(Population, Option<CovariateTable>), String> {
    routing.validate()?;
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV: {}", e))?;

    // Preserve original header casing for covariate names. Standard NONMEM
    // columns are matched case-insensitively so that legacy CSVs (e.g. `Id`,
    // `TIME`) keep working; covariate lookups remain case-sensitive.
    let mut headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Failed to read headers: {}", e))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    // Apply `[data]` column remapping (#730, #742): rename each mapped actual
    // header to its target name (case-insensitive match on the actual header).
    // Downstream canonical lookups (`col_idx_ci("time")`) and covariate
    // auto-detection (`is_standard`) then treat a renamed column as that role /
    // covariate with no further special-casing, and the original header no
    // longer leaks in under its old name. Resolve every actual header against
    // the *original* headers first, then apply the renames, so one mapping can
    // never shadow another's lookup.
    if !column_map.is_empty() {
        let mut planned: Vec<(usize, String)> = Vec::with_capacity(column_map.len());
        for (target, actual) in column_map {
            // A mapped header that is absent is a hard error — the mapping would
            // silently do nothing (e.g. TIME would still be reported missing).
            let idx = headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(actual))
                .ok_or_else(|| {
                    format!(
                        "[data]: mapped column `{actual}` (renamed to `{target}`) not found in \
                         dataset headers: {}",
                        headers.join(", ")
                    )
                })?;
            // Reject clobbering the IOV occasion column: renaming it would make
            // the later `iov_column` lookup fail with a misleading "not found".
            if let Some(iov) = iov_column {
                if actual.eq_ignore_ascii_case(iov) {
                    return Err(format!(
                        "[data]: mapped column `{actual}` (renamed to `{target}`) is also the \
                         iov_column `{iov}`"
                    ));
                }
            }
            planned.push((idx, target.clone()));
        }
        // A column being renamed *away* frees its old name — the raw `dv` column
        // renamed to `ODV` no longer collides with a `DV = lndv` mapping (#742).
        // So a target may collide only with a *surviving* (non-source) header.
        let sources: std::collections::HashSet<usize> = planned.iter().map(|(i, _)| *i).collect();
        for (idx, target) in &planned {
            if let Some(other) = headers.iter().position(|h| h.eq_ignore_ascii_case(target)) {
                if other != *idx && !sources.contains(&other) {
                    return Err(format!(
                        "[data]: renaming to `{target}` collides — the dataset already has a `{}` \
                         column",
                        headers[other]
                    ));
                }
            }
        }
        for (idx, name) in planned {
            headers[idx] = name;
        }
    }

    let col_idx_ci =
        |name: &str| -> Option<usize> { headers.iter().position(|h| h.eq_ignore_ascii_case(name)) };
    let col_idx_cs = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

    let id_col = col_idx_ci("id").ok_or("Missing ID column")?;
    let time_col = col_idx_ci("time").ok_or("Missing TIME column")?;
    let dv_col = col_idx_ci("dv").ok_or("Missing DV column")?;
    let evid_col = col_idx_ci("evid");
    let amt_col = col_idx_ci("amt");
    let cmt_col = col_idx_ci("cmt");
    let rate_col = col_idx_ci("rate");
    let mdv_col = col_idx_ci("mdv");
    let ii_col = col_idx_ci("ii");
    let ss_col = col_idx_ci("ss");
    let cens_col = col_idx_ci("cens");
    let addl_col = col_idx_ci("addl");
    // TENTRY column: left-truncation / delayed-entry time for TTE rows.
    // Absent in Gaussian-only datasets; only read for `routing.tte` (Event) rows.
    let tentry_col = col_idx_ci("tentry");
    // L2 column: NONMEM level-2 grouping id. Records sharing an L2 value form one
    // correlated observation unit (`block_sigma` cross covariance). Optional.
    let l2_col = col_idx_ci("l2");

    // FREMTYPE column (case-insensitive)
    let fremtype_col: Option<usize> = col_idx_ci("fremtype");

    // IOV occasion column (case-insensitive lookup of user-specified name)
    let occ_col: Option<usize> = iov_column.and_then(|name| col_idx_ci(name));
    if iov_column.is_some() && occ_col.is_none() {
        return Err(format!(
            "iov_column '{}' not found in dataset headers",
            iov_column.unwrap()
        ));
    }

    const STANDARD_COLS: &[&str] = &[
        "id", "time", "dv", "evid", "amt", "cmt", "rate", "mdv", "ii", "ss", "cens", "addl",
        "tentry", "fremtype", "l2",
    ];
    let is_standard = |h: &str| {
        STANDARD_COLS.iter().any(|s| h.eq_ignore_ascii_case(s))
            || iov_column.map_or(false, |iov| h.eq_ignore_ascii_case(iov))
    };

    // Identify covariate columns (names preserved in their original case).
    let cov_names: Vec<String> = match covariate_columns {
        Some(cols) => cols.iter().map(|c| c.to_string()).collect(),
        None => headers
            .iter()
            .filter(|h| !is_standard(h))
            .cloned()
            .collect(),
    };
    let cov_indices: Vec<(String, usize)> = cov_names
        .iter()
        .filter_map(|name| {
            // Prefer an exact header match; fall back to case-insensitive so a
            // filter-injected lowercase name (e.g. "study" from
            // `referenced_covariate_columns`) still resolves to a "STUDY" header.
            // Store the actual header name so covariate keys match the dataset.
            col_idx_cs(name)
                .or_else(|| col_idx_ci(name))
                .map(|idx| (headers[idx].clone(), idx))
        })
        .collect();

    // A filter that references a covariate column absent from the data can never
    // fire (the column is missing from every row's covariate map, so the clause
    // is a silent no-op). This commonly means a typo in the column name — e.g.
    // `ignore = Coment` instead of `Comment`, or the bare `ignore = C` shorthand
    // naming a column the file does not have. Collect those names here and warn
    // once below so the user is not left with an unfiltered fit. ID is handled
    // separately and is never a covariate, so exclude it.
    let filter_absent_cols: Vec<String> = filter
        .map(|f| {
            f.referenced_covariate_columns()
                .into_iter()
                .filter(|c| c != "id" && col_idx_cs(c).is_none() && col_idx_ci(c).is_none())
                .collect()
        })
        .unwrap_or_default();

    // Optional covariate table over the *declared* covariates. Every declared
    // column must exist — otherwise it would silently vanish and evaluate to
    // nothing — so resolve indices up front and fail loudly on any miss.
    let table_indices: Vec<(String, usize)> = if let Some(decls) = table_decls {
        let missing: Vec<&str> = decls
            .iter()
            .filter(|d| col_idx_cs(&d.name).is_none())
            .map(|d| d.name.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{ERR_COV_MISSING_COLUMNS} (case-sensitive): {}. Available columns: {}.",
                missing.join(", "),
                headers.join(", ")
            ));
        }
        decls
            .iter()
            .map(|d| (d.name.clone(), col_idx_cs(&d.name).unwrap()))
            .collect()
    } else {
        Vec::new()
    };

    // Covariate table: one row per input record, in file order. Only built when
    // declarations were supplied (authoritative `[covariates]` path).
    let build_table = table_decls.is_some();
    let mut table_rows: Vec<CovariateRow> = Vec::new();

    // Parse rows grouped by ID
    let mut rows_by_id: Vec<(String, Vec<Vec<String>>)> = Vec::new();
    let mut current_id = String::new();

    for result in rdr.records() {
        let record = result.map_err(|e| format!("CSV parse error: {}", e))?;
        let fields: Vec<String> = record.iter().map(|f| f.trim().to_string()).collect();

        let id = fields.get(id_col).cloned().unwrap_or_default();

        if build_table {
            let time = parse_f64(fields.get(time_col).map(|s| s.as_str()).unwrap_or("0"));
            // Mirror `parse_subject`'s EVID computation (incl. AMT-based dose
            // inference when EVID is absent) so the table's EVID agrees with how
            // each row was classified. #262
            let evid = effective_evid(&fields, evid_col, amt_col);
            let mut values = Vec::with_capacity(table_indices.len());
            for (name, idx) in &table_indices {
                let cell = fields.get(*idx).map(|s| s.as_str()).unwrap_or("");
                if is_missing_cell(cell) {
                    values.push(f64::NAN);
                } else {
                    match cell.trim().parse::<f64>() {
                        Ok(v) => values.push(v),
                        Err(_) => {
                            return Err(format!(
                                "{ERR_COV_NON_NUMERIC} '{}' for covariate '{}' (ID {}, TIME {}). \
                                 Covariates must be numerically coded — encode categoricals as \
                                 integer levels.",
                                cell.trim(),
                                name,
                                id,
                                time
                            ));
                        }
                    }
                }
            }
            table_rows.push(CovariateRow {
                id: id.clone(),
                time,
                evid,
                values,
            });
        }

        if id != current_id {
            current_id = id.clone();
            rows_by_id.push((id, Vec::new()));
        }
        rows_by_id.last_mut().unwrap().1.push(fields);
    }

    // Build subjects, applying selection filter if present.
    let mut subjects = Vec::new();
    let mut total_occ_failures: usize = 0;
    let mut total_missing_dv: usize = 0;
    // Rows dropped despite a nonzero AMT, summed across subjects (#262).
    let mut total_amt_ignored: usize = 0;
    let mut subjects_with_amt_ignored: usize = 0;
    let mut population_warnings: Vec<String> = Vec::new();
    if !filter_absent_cols.is_empty() {
        population_warnings.push(format!(
            "W_FILTER_COLUMN_ABSENT: data-selection filter references column(s) not found in the \
             data: {}. These conditions never match, so no rows are excluded for them — check for \
             a typo in the column name. Available columns: {}.",
            filter_absent_cols.join(", "),
            headers.join(", ")
        ));
    }
    let n_records_total: usize = rows_by_id.iter().map(|(_, rows)| rows.len()).sum();
    let mut excl_summary = ExclusionSummary {
        n_records_total,
        ..Default::default()
    };
    for (id, rows) in &rows_by_id {
        let (subject, occ_failures, missing_dv, subj_excl, subj_warnings, amt_ignored) =
            parse_subject(
                id,
                rows,
                time_col,
                dv_col,
                evid_col,
                amt_col,
                cmt_col,
                rate_col,
                mdv_col,
                ii_col,
                ss_col,
                cens_col,
                occ_col,
                addl_col,
                fremtype_col,
                l2_col,
                &cov_indices,
                filter,
                routing,
                tentry_col,
            )?;
        total_occ_failures += occ_failures;
        total_missing_dv += missing_dv;
        total_amt_ignored += amt_ignored;
        if amt_ignored > 0 {
            subjects_with_amt_ignored += 1;
        }
        population_warnings.extend(subj_warnings);

        // Accumulate filter statistics.
        excl_summary.n_obs_excluded += subj_excl.n_obs_excluded;
        excl_summary.n_dose_excluded += subj_excl.n_dose_excluded;
        excl_summary.n_other_excluded += subj_excl.n_other_excluded;
        for src in subj_excl.fired {
            if !excl_summary.fired_ignore.contains(&src)
                && !excl_summary.fired_accept.contains(&src)
            {
                if src.starts_with("accept:") {
                    excl_summary.fired_accept.push(src);
                } else {
                    excl_summary.fired_ignore.push(src);
                }
            }
        }

        // Warn about pathological partial exclusions. Collected into
        // `population_warnings` (not printed) per the warning convention —
        // they surface via `FitResult.warnings`.
        let has_doses = !subject.doses.is_empty();
        let has_obs = !subject.observations.is_empty();
        let excluded_doses = subj_excl.n_dose_excluded > 0;
        let excluded_obs = subj_excl.n_obs_excluded > 0;

        if excluded_doses && !has_doses && has_obs {
            population_warnings.push(format!(
                "subject {id}: all dose records were excluded by [data_selection] but \
                 observations remain — predictions will be undefined."
            ));
        }
        if excluded_obs && !has_obs && has_doses {
            population_warnings.push(format!(
                "subject {id}: all observation records were excluded by [data_selection] but \
                 dose records remain — subject contributes nothing to the likelihood."
            ));
        }

        // Only drop a subject as "excluded by data selection" when the filter
        // actually removed at least one of its records and that left it empty.
        // Without this guard we would also drop subjects that are empty for
        // unrelated reasons (e.g. only EVID=2/3 rows) on the no-filter path,
        // which is a behavior change vs. the pre-feature reader (it pushed every
        // subject unconditionally). `had_exclusions` is only ever > 0 when a
        // filter is active.
        let had_exclusions =
            subj_excl.n_obs_excluded + subj_excl.n_dose_excluded + subj_excl.n_other_excluded > 0;
        if had_exclusions && subject.doses.is_empty() && subject.observations.is_empty() {
            // Subject entirely excluded by the filter — do not add to the list.
            excl_summary.excluded_subject_ids.push(id.clone());
            continue;
        }
        subjects.push(subject);
    }

    // Hard error (#753): dose rows are present (EVID=1/4 classified as doses) but
    // the dataset has no `AMT` column, so every dose was created with `amt = 0`.
    // The fit would run silently with no drug in the system — a flat objective
    // that pins every parameter at its initial estimate. This most often means the
    // amount column is named something other than `AMT` (e.g. a NONMEM export using
    // `DOSE`); rename it in the [data] block or the CSV header. Scoped to dose rows
    // present + no AMT column: dose-free models (#262) and TTE/survival have no dose
    // events, and the EVID-absent path infers no doses without an AMT column, so
    // none of them trip this.
    if amt_col.is_none() && subjects.iter().any(|s| !s.doses.is_empty()) {
        return Err(
            "E_DOSE_NO_AMT: the dataset has dose records (EVID=1/4) but no `AMT` column, so \
             every dose amount is zero and the fit would run with no drug in the system. If the \
             amount column has a different name (e.g. `DOSE`), rename it to `AMT` in the CSV \
             header or via a rename in the [data] block."
                .to_string(),
        );
    }

    // Accumulate OCC warning into population_warnings (surfaced via FitResult.warnings).
    if let Some(name) = iov_column {
        if total_occ_failures > 0 {
            population_warnings.push(format!(
                "W_IOV_OCC_MISSING: {} row(s) had missing or unparseable values in \
                 iov_column '{}'; these rows were assigned occasion=0 and may be grouped \
                 with valid occ=0 rows. Consider cleaning the dataset.",
                total_occ_failures, name
            ));
        }
    }

    // Missing-DV summary: scored observation rows (EVID=0, MDV=0) whose DV cell
    // was missing. Both readings change the row count relative to a dataset with
    // no missing cells, so both get a summary line — under `Skip` the rows are
    // dropped (issue #258), under `KeepAsDesign` they are kept as extra design
    // points (#957). A simulation run off an *observed* dataset (a VPC, say)
    // silently carries more rows than the fit did without this second line.
    // Surfaced via FitResult.warnings and `ferx check` (data path).
    if total_missing_dv > 0 {
        population_warnings.push(match routing.missing_dv {
            MissingDvPolicy::Skip => format!(
                "W_MISSING_DV: {} observation row(s) (EVID=0) had a missing DV (`.`/`NA`/blank) \
                 but were not marked MDV=1; they were skipped (not scored as DV=0). Set MDV=1 \
                 on intentionally-missing observations to silence this, or check for data errors.",
                total_missing_dv
            ),
            MissingDvPolicy::KeepAsDesign => format!(
                "W_DESIGN_DV: {} observation row(s) (EVID=0) had a missing DV (`.`/`NA`/blank) \
                 and were kept as design points to simulate at. A fit of the same dataset would \
                 skip these rows (W_MISSING_DV), so the simulated dataset has more rows than the \
                 fitted one. Set MDV=1 on rows that are not sampling times.",
                total_missing_dv
            ),
        });
    }

    // Dose-coverage warnings (#262), surfaced via FitResult.warnings. Most
    // specific wins so a dataset never gets both: W_AMT_NOT_DOSED pinpoints AMT
    // that was dropped; W_NO_DOSES is the generic "no doses parsed at all"
    // backstop for datasets that carry no AMT signal to begin with.
    if total_amt_ignored > 0 {
        population_warnings.push(format!(
            "W_AMT_NOT_DOSED: {} record(s) across {} subject(s) carry AMT != 0 but were not \
             treated as dose events (EVID is not 1 or 4); their AMT was ignored. If the dataset \
             has no EVID column, a dose row must carry a nonzero AMT to be inferred as a dose; \
             otherwise code dose rows as EVID=1 (or EVID=4).",
            total_amt_ignored, subjects_with_amt_ignored
        ));
    } else if subjects.iter().all(|s| s.doses.is_empty()) {
        // Zero dose events across the whole population. Warn only when scored
        // observations are present (an all-EVID=2 / covariate-only dataset is not
        // a fit) and the dataset carries no non-Gaussian observations — TTE/survival
        // and discrete/count endpoints legitimately have no PK doses, so suppressing
        // the warning for them avoids a noisy false positive. `obs_records` is
        // unconditional since Phase 4.0, so this needs no feature split.
        let total_scored_obs: usize = subjects.iter().map(|s| s.observations.len()).sum();
        let any_nongaussian_obs = subjects.iter().any(|s| !s.obs_records.is_empty());
        if total_scored_obs > 0 && !any_nongaussian_obs {
            population_warnings.push(format!(
                "W_NO_DOSES: parsed zero dose events across all {} subject(s) although scored \
                 observations are present. If this is a PK model, check that the dataset has an \
                 AMT column with EVID=1/4 dose rows (or a nonzero AMT when EVID is absent).",
                subjects.len()
            ));
        }
    }

    // All-zero-dose warning (#753): an `AMT` column exists (so E_DOSE_NO_AMT did not
    // fire) but every parsed dose amount is exactly zero — e.g. a placebo-only typo
    // or a mis-scaled amount column. Like the `E_DOSE_NO_AMT` error above, this yields a flat
    // objective; here we warn rather than error because the column is present and a
    // genuinely dose-free-but-AMT-columned dataset is conceivable. Only fires when at
    // least one dose event exists (else W_NO_DOSES already covered it).
    if amt_col.is_some() {
        let any_dose = subjects.iter().any(|s| !s.doses.is_empty());
        let all_zero = subjects
            .iter()
            .all(|s| s.doses.iter().all(|d| d.amt == 0.0));
        if any_dose && all_zero {
            population_warnings.push(
                "W_ALL_DOSES_ZERO: every dose record has AMT = 0, so no drug enters the system \
                 and the objective is flat (parameters will not move from their initial values). \
                 Check the `AMT` column values and scaling."
                    .to_string(),
            );
        }
    }

    let exclusions = if filter.is_some() {
        Some(excl_summary)
    } else {
        None
    };

    let table = if let Some(decls) = table_decls {
        // `table_indices` (and hence each row's `values`) is in declaration
        // order, so names/kinds taken from `decls` stay aligned.
        Some(CovariateTable {
            names: decls.iter().map(|d| d.name.clone()).collect(),
            kinds: decls.iter().map(|d| d.kind).collect(),
            rows: table_rows,
        })
    } else {
        None
    };

    // `covariate_names` reports only columns that actually exist in the data
    // (derived from `cov_indices`). A requested column that isn't in the CSV —
    // e.g. a referenced-but-undeclared covariate passed in the union that turns
    // out to be absent — must NOT appear here, otherwise `check_covariates`
    // would treat it as present and let the fit run with that covariate at 0.0
    // instead of failing with E_MISSING_COVARIATE. (For the auto-detect path
    // `cov_names` is already existing-only, so this is a no-op there.)
    let existing_cov_names: Vec<String> = cov_indices.iter().map(|(n, _)| n.clone()).collect();

    Ok((
        Population {
            subjects,
            covariate_names: existing_cov_names,
            dv_column: "dv".to_string(),
            input_columns: headers,
            exclusions,
            warnings: population_warnings,
        },
        table,
    ))
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

/// Parse a numeric cell for the data-selection filter, mapping missing/blank
/// cells (`.`, `NA`, empty) to `NaN`. Because every IEEE comparison against
/// `NaN` is false (see `cmp_f64`), a record whose value for a referenced column
/// is missing never matches that condition — so `ignore = DV < 0.001` skips
/// dose rows (where `DV` is `.`) instead of silently treating them as `0`.
fn parse_f64_or_nan(s: &str) -> f64 {
    let t = s.trim();
    if is_missing_cell(t) {
        f64::NAN
    } else {
        t.parse::<f64>().unwrap_or(f64::NAN)
    }
}

fn parse_usize(s: &str) -> usize {
    s.parse::<usize>().unwrap_or(0)
}

/// Parse an `L2` grouping-id cell. NONMEM writes an integer, but pandas/R
/// exports commonly float-format the whole column (`"10.0"`, `"11.0"`) once any
/// row is blank — so a strict `i64` parse would silently ungroup every record
/// and discard the user's block_sigma pairing (#830). Accept an integer literal
/// or a *float-formatted integer* (fractional part exactly 0, within `i64`
/// range). A genuinely non-integer value (`"2.4"`), an out-of-range magnitude, a
/// blank, or an unparseable cell means "ungrouped" (`None` → 0) — left ungrouped
/// rather than silently rounded/saturated into a group, which would mis-pair the
/// residual correlation in a hard-to-debug way.
fn parse_l2_id(s: &str) -> Option<i64> {
    let t = s.trim();
    if is_missing_cell(t) {
        return None;
    }
    if let Ok(i) = t.parse::<i64>() {
        return Some(i);
    }
    let f = t.parse::<f64>().ok()?;
    if f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
        Some(f as i64)
    } else {
        None
    }
}

fn parse_cens(s: &str) -> i8 {
    let t = s.trim();
    if is_missing_cell(t) {
        0
    } else {
        t.parse::<i8>().unwrap_or(0)
    }
}

/// Parse an EVID cell. A missing / blank / unparseable value maps to 0
/// (observation) — NONMEM's documented default. (`parse_usize` defaults to 1,
/// which would mislabel a blank-EVID observation row as a dose.)
fn parse_evid(s: &str) -> u32 {
    let t = s.trim();
    if is_missing_cell(t) {
        return 0;
    }
    t.parse::<u32>().unwrap_or(0)
}

/// True for EVID values that administer a dose (1 = dose, 4 = reset + dose).
/// Single source of truth for the dose test, shared by the dose-record arm, the
/// data-selection exclusion tally, and the ignored-AMT counter.
fn is_dose_evid(evid: u32) -> bool {
    evid == 1 || evid == 4
}

/// True when an `AMT` value denotes an actual dose: **finite and nonzero**. A
/// missing cell (or absent column) parses to `0.0` — not a dose. A literal
/// `nan`/`inf`/`infinity` parses to a non-finite value (Rust's `f64::from_str`
/// accepts those, and [`parse_f64`] does not route through `is_missing_cell`),
/// which is malformed and is also rejected here — so a stray non-finite AMT
/// never silently becomes an infinite/NaN-amount dose (#262).
fn is_dosing_amt(amt: f64) -> bool {
    amt.is_finite() && amt != 0.0
}

/// Classify (and validate) the `RATE` cell of a *dose* record into the
/// [`RateMode`] its [`DoseEvent`] should carry.
///
/// NONMEM overloads `RATE` with coded values:
///   - `0`  → bolus (route set by the dose compartment)
///   - `>0` → constant-rate infusion (duration = `AMT/RATE`)
///   - `-1` → infusion **rate** is *modeled* (a `$PK` `R{n}` parameter)
///   - `-2` → infusion **duration** is *modeled* (a `$PK` `D{n}` parameter)
///
/// `-1` is accepted as [`RateMode::ModeledRate`] and `-2` as
/// [`RateMode::ModeledDuration`] (#324). The datareader has no model, so it
/// cannot yet know whether a matching `R{cmt}`/`D{cmt}` parameter exists or
/// whether the model can infuse into that compartment — those checks move to the
/// model+data join ([`crate::api::check_model_data`]). Any other negative and
/// non-finite values are rejected here: they are unconditionally invalid
/// regardless of the model. (Previously `-1`/`-2` fell through to `rate > 0.0`
/// and were silently treated as boluses — #324.)
fn validate_dose_rate(rate: f64, id: &str, time: f64) -> Result<RateMode, String> {
    if !rate.is_finite() {
        return Err(format!(
            "subject {id}, time {time}: RATE={rate} is not finite; expected 0 \
             (bolus), a positive infusion rate, -1 (modeled rate), or -2 \
             (modeled duration)"
        ));
    }
    if rate >= 0.0 {
        return Ok(RateMode::Fixed);
    }
    // rate < 0 → a NONMEM coded value, which is always an *exact negative
    // integer*. Match on the integer form so the arms read as the codes they are
    // and a new code is one more arm. A non-integer negative (e.g. -1.5) is not a
    // code: `fract() != 0.0` rejects it rather than rounding it into one (`round()`
    // would map -1.5 → -2 and silently accept it as modeled duration). Comparison
    // against `0.0` is exempt from clippy::float_cmp; `rate as i64` saturates, so
    // an out-of-range integer can't alias -1/-2.
    let code = if rate.fract() == 0.0 {
        Some(rate as i64)
    } else {
        None
    };
    let detail = match code {
        Some(-2) => return Ok(RateMode::ModeledDuration),
        Some(-1) => return Ok(RateMode::ModeledRate),
        _ => format!("RATE={rate} is a negative value that is not a recognised NONMEM code"),
    };
    Err(format!(
        "subject {id}, time {time}: {detail}. Recognised RATE values are 0 \
         (bolus), >0 (infusion rate), -1 (modeled infusion rate), and -2 \
         (modeled infusion duration)."
    ))
}

/// Validate a `SS` (steady-state) dataset cell.
///
/// NONMEM overloads `SS`: `0` = not steady state, `1` = reset the dose
/// compartment to zero and initialise it at the steady state of the given
/// regimen, `2` = *superimpose* that steady state on top of the compartment's
/// pre-existing amounts (no reset). ferx implements `SS=1` only.
///
/// The engine carries steady state as a single `DoseEvent.ss: bool`, so once a
/// cell is reduced to that flag the `1`-vs-`2` distinction is gone. Previously
/// any `SS >= 0.5` — including `SS=2` — was collapsed to `ss = true` and run
/// with `SS=1` (reset) semantics, so an `SS=2` record produced a wrong (reset)
/// profile with no error. `0` maps to `false`, `1` to `true`; every other value
/// (including `SS=2`) is rejected here. Full `SS=2` support is tracked in #694.
///
/// A missing / blank / non-numeric cell arrives as `0.0` (via [`parse_f64`]) and
/// is treated as "not steady state", matching the NONMEM default. `time` is the
/// user-written (`raw_time`) value so the message names the row's own time.
fn validate_ss(ss: f64, id: &str, time: f64) -> Result<bool, String> {
    if !ss.is_finite() {
        return Err(format!(
            "subject {id}, time {time}: SS={ss} is not finite; expected 0 (not \
             steady state) or 1 (reset then dose to steady state)."
        ));
    }
    // `SS` is a NONMEM integer code. Match on the integer form (mirrors
    // `validate_dose_rate`); a non-integer `SS` is not a code. `ss as i64`
    // saturates, so an out-of-range integer can't alias 0/1. Comparison against
    // `0.0` is exempt from clippy::float_cmp.
    let code = if ss.fract() == 0.0 {
        Some(ss as i64)
    } else {
        None
    };
    match code {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => {
            let hint = if code == Some(2) {
                " SS=2 (superimpose the steady state without resetting the \
                 compartment) is not yet supported — only SS=1 (reset then dose \
                 to steady state) is implemented; see issue #694."
            } else {
                ""
            };
            Err(format!(
                "subject {id}, time {time}: SS={ss} is not a supported \
                 steady-state code; expected 0 (not steady state) or 1 (reset \
                 then dose to steady state).{hint}"
            ))
        }
    }
}

/// Build a dose that honors the row's `RATE` classification (#324/#722).
///
/// A coded `RATE=-1`/`-2` row yields a *modeled* infusion ([`DoseEvent::modeled`],
/// whose concrete rate/duration are resolved per iteration from `R{cmt}`/`D{cmt}`);
/// an ordinary row yields a [`RateMode::Fixed`] dose. Passing the raw `-1`/`-2`
/// sentinel to [`DoseEvent::new`] instead would set `rate_mode = Fixed` with
/// `duration = 0` (since `rate <= 0`), so `is_infusion()` is false and the dose
/// silently collapses to an instantaneous bolus — and `check_modeled_dose_rates`
/// skips `Fixed` doses, so the missing slot is never caught.
///
/// **Single construction site** for the primary dose *and* its `ADDL`-expanded
/// copies, so the two cannot diverge — that divergence is exactly what silently
/// collapsed `ADDL` modeled infusions to boluses (#722). `ADDL` doses pass
/// `ss = false` (an expanded dose is never steady-state itself); everything else
/// is identical to the primary row.
fn dose_for_rate_mode(
    time: f64,
    amt: f64,
    cmt: usize,
    rate: f64,
    ss: bool,
    ii: f64,
    rate_mode: RateMode,
) -> DoseEvent {
    match rate_mode {
        RateMode::Fixed => DoseEvent::new(time, amt, cmt, rate, ss, ii),
        RateMode::ModeledDuration | RateMode::ModeledRate => {
            DoseEvent::modeled(time, amt, cmt, ss, ii, rate_mode)
        }
    }
}

/// Compute a record's effective EVID.
///
/// When an `EVID` column is present its value governs (a blank / `.` /
/// unparseable cell is the documented NONMEM default of 0 = observation, via
/// [`parse_evid`]).
///
/// When the `EVID` column is **absent**, NONMEM infers the record type from
/// `AMT`: a row with a nonzero `AMT` is a dose (EVID 1); everything else is an
/// observation (EVID 0). Without this, an EVID-less dataset silently drops every
/// `AMT` row — it is neither a dose (needs EVID 1/4) nor an observation (needs
/// EVID 0 and MDV 0, but dose rows carry MDV=1) — and fits a degenerate
/// dose-free model (#262). Inference keys on `AMT` only: a NONMEM dose always
/// carries a nonzero `AMT` (infusions too — `RATE` is the rate, `AMT` the
/// amount), so a `RATE`-only row would just create a no-op zero-amount dose.
///
/// Only a finite, nonzero `AMT` infers a dose: a missing cell parses to `0.0`
/// and a non-finite `nan`/`inf` is rejected, both via [`is_dosing_amt`].
fn effective_evid(row: &[String], evid_col: Option<usize>, amt_col: Option<usize>) -> u32 {
    match evid_col {
        Some(c) => row.get(c).map(|s| parse_evid(s)).unwrap_or(0),
        None => {
            let amt = amt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            if is_dosing_amt(amt) {
                1
            } else {
                0
            }
        }
    }
}

/// Parse an occasion-column cell. Returns `None` for blank / `.` / NA / non-integer
/// values so the caller can warn about silently dropped rows. NONMEM convention
/// uses `.` for missing.
fn parse_occ(s: &str) -> Option<u32> {
    let t = s.trim();
    if is_missing_cell(t) {
        return None;
    }
    t.parse::<u32>().ok()
}

#[allow(clippy::too_many_arguments)]
fn parse_subject(
    id: &str,
    rows: &[Vec<String>],
    time_col: usize,
    dv_col: usize,
    evid_col: Option<usize>,
    amt_col: Option<usize>,
    cmt_col: Option<usize>,
    rate_col: Option<usize>,
    mdv_col: Option<usize>,
    ii_col: Option<usize>,
    ss_col: Option<usize>,
    cens_col: Option<usize>,
    occ_col: Option<usize>,
    addl_col: Option<usize>,
    fremtype_col: Option<usize>,
    l2_col: Option<usize>,
    cov_indices: &[(String, usize)],
    filter: Option<&SelectionFilter>,
    // Per-CMT non-Gaussian endpoint routing (Event / DiscreteState / Count).
    // Empty for Gaussian-only models. Always available; the TTE (`Event`) arm is
    // feature-gated behind `survival`, the discrete/count arms compile unconditionally.
    routing: &ObsRouting,
    // Column index of the TENTRY (left-truncation time) column, if present.
    _tentry_col: Option<usize>,
) -> Result<(Subject, usize, usize, SubjectExclusion, Vec<String>, usize), String> {
    let mut doses = Vec::new();
    // File-order record index (position in `rows`) of each dose / observation,
    // parallel to `doses` (pre-sort) and `obs_times`. Used after the loop to
    // honor NONMEM record order at equal TIME (see the obs/dose tie-break pass).
    let mut dose_rec: Vec<usize> = Vec::new();
    let mut obs_rec: Vec<usize> = Vec::new();
    let mut obs_times = Vec::new();
    let mut obs_raw_times = Vec::new();
    let mut observations = Vec::new();
    let mut obs_cmts = Vec::new();
    let mut cens = Vec::new();
    let mut occasions: Vec<u32> = Vec::new();
    let mut dose_occasions: Vec<u32> = Vec::new();
    let mut fremtype: Vec<u16> = Vec::new();
    let mut obs_l2: Vec<i64> = Vec::new();
    let mut occ_parse_failures: usize = 0;
    // EVID=0/MDV=0 rows whose DV cell was missing, counted under **both**
    // policies: skipped under `Skip` (issue #258, `W_MISSING_DV`) and kept as
    // design points under `KeepAsDesign` (#957, `W_DESIGN_DV`). Either reading
    // changes how many rows the dataset contributes, so either one is worth a
    // summary line.
    let mut missing_dv_rows: usize = 0;
    let mut excl_n_obs: usize = 0;
    let mut excl_n_dose: usize = 0;
    let mut excl_n_other: usize = 0;
    let mut excl_fired: Vec<String> = Vec::new();
    let mut parse_warnings: Vec<String> = Vec::new();
    let mut addl_missing_ii_warned = false;
    let mut cens_invalid_warned = false;
    // Rows that survived the data-selection filter, carry a nonzero AMT, yet
    // were not classified as a dose (EVID not 1/4) — their AMT was silently
    // dropped. Reported as a population summary so a degenerate dose-free fit
    // can't pass unnoticed (#262). Counted post-filter so deliberately excluded
    // dose rows don't trip the warning.
    let mut amt_ignored_rows: usize = 0;

    // Non-Gaussian observation records for this subject. Holds TTE `Event`s
    // (pushed only under `survival`) and, unconditionally, discrete-state / count
    // rows. Unconditional since Phase 4.0 so the categorical (Track C) and Markov
    // (Track D) endpoints share one stream on the default build.
    let mut obs_records: Vec<crate::types::ObsRecord> = Vec::new();
    // tte_pending_left: per-CMT pending DV=0 row (may be a left-bound for an interval
    //   or a right-censored event, depending on whether the next row is DV=2).
    //   Map value is (time, entry_time). TTE-only, hence `survival`-gated.
    #[cfg(feature = "survival")]
    let mut tte_pending_left: HashMap<usize, (f64, f64)> = HashMap::new();

    // Time-constant covariates: first non-missing value across all rows.
    // Used as the subject-static fallback (and for the AD fast path, which
    // does not yet read per-event snapshots).
    let mut covariates: HashMap<String, f64> = HashMap::new();
    for (name, idx) in cov_indices {
        for row in rows {
            if let Some(val_str) = row.get(*idx) {
                if let Ok(val) = val_str.parse::<f64>() {
                    if val.is_finite() {
                        covariates.insert(name.clone(), val);
                        break;
                    }
                }
            }
        }
    }

    // Detect which covariates are time-varying within this subject. Per-event
    // snapshots are only built when at least one is — keeps memory flat for
    // models with no TV covariates.
    let mut tv_names: Vec<&str> = Vec::new();
    let mut tv_indices: Vec<usize> = Vec::new();
    for (name, idx) in cov_indices {
        let mut first_val: Option<f64> = None;
        let mut is_tv = false;
        for row in rows {
            let v_opt = row
                .get(*idx)
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|v| v.is_finite());
            if let Some(v) = v_opt {
                match first_val {
                    None => first_val = Some(v),
                    Some(fv) if (v - fv).abs() > 1e-12 => {
                        is_tv = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
        if is_tv {
            tv_names.push(name.as_str());
            tv_indices.push(*idx);
        }
    }
    let any_tv = !tv_names.is_empty();

    // LOCF state for the per-event snapshot path. Initialized from the
    // subject-static `covariates` map so the first event sees something
    // sensible even if the row's own value is missing.
    let mut locf_state: HashMap<String, f64> = covariates.clone();

    // Per-event covariate snapshots (only populated when any_tv is true).
    let mut dose_covariates: Vec<HashMap<String, f64>> = Vec::new();
    let mut obs_covariates: Vec<HashMap<String, f64>> = Vec::new();
    // EVID=2 ("other event") rows — typically covariate-change markers.
    // Only worth tracking when there are TV covariates, since otherwise
    // re-evaluating $PK with unchanged values is a no-op.
    let mut pk_only_times: Vec<f64> = Vec::new();
    let mut pk_only_covariates: Vec<HashMap<String, f64>> = Vec::new();
    // EVID=3 (reset) and EVID=4 (reset + dose) rows. Both zero every
    // compartment amount at `time`; EVID=4 additionally records a dose
    // (handled in the `evid == 1 || evid == 4` arm below).
    let mut reset_times: Vec<f64> = Vec::new();
    let mut reset_covariates: Vec<HashMap<String, f64>> = Vec::new();

    // Reset-delimited occasion segmentation. NONMEM processes records
    // sequentially, so an EVID=3/4 reset whose TIME restarts at/below the
    // running timeline begins a fresh occasion that reuses the previous
    // occasion's wall-clock (e.g. two infusion occasions both timed from 0,
    // stacked under one ID). Our event engine sorts events by absolute time,
    // which would interleave such occasions and double the administered dose.
    // We instead shift each restarting segment — and every event after it,
    // until the next restart — past the prior segment onto a single monotonic
    // timeline. The reset zeros all compartments at the boundary, so the
    // inserted gap carries no drug: predictions are identical to integrating
    // each occasion independently, while the subject keeps one shared set of
    // random effects (matching NONMEM's EVID=4 semantics). `time_offset` is
    // the running shift; `max_eff_time` is the largest effective (shifted)
    // event time emitted so far.
    let mut time_offset = 0.0f64;
    let mut max_eff_time = f64::NEG_INFINITY;

    // Only string-valued covariate filters (IGNORE(C.EQ.C)) read the raw-cell
    // map; numeric-only filters never touch it, so skip the per-row allocation.
    let needs_str_cov = filter.is_some_and(|f| f.needs_str_covariates());

    for (row_seq, row) in rows.iter().enumerate() {
        // Update LOCF state from this row's TV-covariate values *before*
        // classifying the event, matching NONMEM's "$PK runs at the record
        // with this record's covariate values" semantics.
        if any_tv {
            for (name, idx) in tv_names.iter().zip(tv_indices.iter()) {
                if let Some(s) = row.get(*idx) {
                    if let Ok(v) = s.parse::<f64>() {
                        if v.is_finite() {
                            locf_state.insert((*name).to_string(), v);
                        }
                    }
                }
            }
        }

        let time = parse_f64(row.get(time_col).map(|s| s.as_str()).unwrap_or("0"));
        // Effective EVID: the column value if present, else inferred from AMT
        // (NONMEM's rule for EVID-less datasets — see `effective_evid`). #262
        let evid = effective_evid(row, evid_col, amt_col);
        let mdv = mdv_col
            .and_then(|c| row.get(c))
            .map(|s| parse_usize(s))
            .unwrap_or(0);
        // Parse OCC. When iov_column is set but a row's value is missing or
        // unparseable, count it (caller emits a single summary warning) and
        // fall back to 0 — matching pre-warning behavior so existing fits
        // don't change. With no iov_column, parse failures are not tracked.
        let occ = if let Some(c) = occ_col {
            match row.get(c).and_then(|s| parse_occ(s)) {
                Some(n) => n,
                None => {
                    occ_parse_failures += 1;
                    0
                }
            }
        } else {
            0
        };

        // ── Data selection filter ─────────────────────────────────────────────
        // Evaluated after LOCF update so the filter sees current covariate
        // values, matching NONMEM's per-record semantics.
        if let Some(sel) = filter {
            // Missing-sensitive numeric columns map `.`/blank to NaN so the
            // filter never matches them (see `parse_f64_or_nan`).
            let dv_for_ctx = parse_f64_or_nan(row.get(dv_col).map(|s| s.as_str()).unwrap_or(""));
            let amt_for_ctx = amt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64_or_nan(s))
                .unwrap_or(f64::NAN);
            let cmt_for_ctx = cmt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(1);
            let rate_for_ctx = rate_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64_or_nan(s))
                .unwrap_or(f64::NAN);
            let ii_for_ctx = ii_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64_or_nan(s))
                .unwrap_or(f64::NAN);
            let ss_for_ctx = ss_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s) > 0)
                .unwrap_or(false);
            let cens_for_ctx = cens_col
                .and_then(|c| row.get(c))
                .map(|s| parse_cens(s))
                .unwrap_or(0);
            // Raw (unparsed) covariate cell strings for this row, keyed
            // lowercased. Lets string filters compare a non-numeric label column
            // (NONMEM's `IGNORE(C.EQ.C)`) that the numeric `locf_state` map drops.
            // Built only when a string-covariate filter is active; an empty
            // `HashMap` does not allocate, so numeric-only filters pay nothing.
            let str_covariates: HashMap<String, String> = if needs_str_cov {
                cov_indices
                    .iter()
                    .filter_map(|(name, idx)| {
                        row.get(*idx)
                            .map(|cell| (name.to_lowercase(), cell.trim().to_string()))
                    })
                    .collect()
            } else {
                HashMap::new()
            };
            let ctx = RowContext {
                id,
                time,
                dv: dv_for_ctx,
                evid,
                amt: amt_for_ctx,
                cmt: cmt_for_ctx,
                rate: rate_for_ctx,
                mdv: mdv as u32,
                cens: cens_for_ctx,
                ii: ii_for_ctx,
                ss: ss_for_ctx,
                covariates: &locf_state,
                str_covariates: &str_covariates,
            };
            let (excluded, which) = sel.should_exclude(&ctx);
            if excluded {
                if let Some(src) = which {
                    if !excl_fired.contains(&src) {
                        excl_fired.push(src);
                    }
                }
                // Count by record type for the summary. The catch-all `other`
                // bucket (EVID 2/3, missing-DV obs) ensures every excluded
                // record is reflected in some counter.
                if is_dose_evid(evid) {
                    excl_n_dose += 1;
                } else if evid == 0 && mdv == 0 {
                    excl_n_obs += 1;
                } else {
                    excl_n_other += 1;
                }
                continue; // skip this row
            }
        }

        // AMT for this row (parsed once, post-filter; reused by the dose arm).
        // A missing column or `.` cell parses to 0.0 (see `parse_f64`).
        let row_amt = amt_col
            .and_then(|c| row.get(c))
            .map(|s| parse_f64(s))
            .unwrap_or(0.0);
        // Track AMT that won't be administered: a dose-like AMT (finite,
        // nonzero) on a record that is neither a dose (EVID 1/4) nor a *scored*
        // observation (`mdv != 0`). The `mdv != 0` gate is what keeps this from
        // false-firing: a scored observation (MDV=0) that merely carries a
        // redundant / forward-filled AMT is benign — a real dropped dose is a
        // non-scored record (a NONMEM dose row is MDV=1). With no EVID column
        // `effective_evid` already promoted dose rows to doses, so this fires
        // mainly on an EVID-present dataset whose dose row was mistyped (e.g.
        // EVID=0, MDV=1, AMT=5000). Surfaced as a population warning. #262
        if is_dosing_amt(row_amt) && !is_dose_evid(evid) && mdv != 0 {
            amt_ignored_rows += 1;
        }

        // Raw (unshifted) TIME for this row, preserved before the occasion
        // shift below so the user-clock diagnostics (sdtab/covtab TIME and
        // predict/simulate TIME) report the value the user wrote, while the
        // engine uses the shifted monotonic `time`.
        let raw_time = time;

        // Reset-delimited occasion segmentation (see `time_offset` above).
        // When an EVID=3/4 reset's TIME would land at or before the running
        // timeline, start a new segment by shifting it (and the rest of this
        // occasion) just past the latest event seen so far. `time` is then the
        // effective, monotonic event time used everywhere downstream; `raw_time`
        // keeps the original column value for the diagnostic outputs.
        if (evid == 3 || evid == 4) && time + time_offset <= max_eff_time {
            time_offset = max_eff_time + RESET_SEGMENT_GAP - time;
        }
        let time = time + time_offset;
        if time > max_eff_time {
            max_eff_time = time;
        }

        // EVID=3 (reset) and EVID=4 (reset + dose) both zero the compartment
        // state at this time. Record the reset before the dose arm runs so
        // EVID=4 captures both the reset and its dose.
        if evid == 3 || evid == 4 {
            reset_times.push(time);
            // A reset row is a data record: NONMEM runs `$PK` at it, so an
            // `[odes] init(...)` re-seeded here is evaluated with THIS row's
            // covariates (#1133). `locf_state` was refreshed from this row at the
            // top of the loop, so it is already the reset row's own snapshot.
            // Skipped without TV covariates, exactly like `pk_only_covariates`:
            // every snapshot is then the subject-static map and `reset_cov`'s
            // fallback returns it.
            if any_tv {
                reset_covariates.push(locf_state.clone());
            }
        }

        if evid == 3 {
            // Pure system reset: no dose, no observation. Nothing else to do.
        } else if is_dose_evid(evid) {
            // Dose record
            let amt = row_amt;
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .and_then(|s| {
                    let t = s.trim();
                    if t == "." || t.is_empty() {
                        None
                    } else {
                        t.parse::<usize>().ok()
                    }
                })
                .unwrap_or(1);
            let rate = rate_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            // Classify the RATE cell (#324): >=0 -> Fixed (data-driven rate),
            // -2 -> ModeledDuration (duration is a `D{cmt}` parameter), -1 ->
            // ModeledRate (rate is an `R{cmt}` parameter) — both resolved at the
            // model+data join; other negative / non-finite are rejected. Uses
            // `raw_time` so the message names the value the user wrote, not the
            // occasion-shifted engine time. `?` bubbles up through `parse_subject`.
            let rate_mode = validate_dose_rate(rate, id, raw_time)?;
            let ii = ii_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            // Validate the `SS` cell: only `SS=0`/`SS=1` are supported. A missing
            // or blank cell parses to `0.0` (not steady state). `SS=2` and other
            // codes are rejected here (via `validate_ss`) rather than silently
            // collapsed into the single `ss = true` flag and run with `SS=1`
            // (reset) semantics. `raw_time` names the value the user wrote.
            let ss = validate_ss(
                ss_col
                    .and_then(|c| row.get(c))
                    .map(|s| parse_f64(s.trim()))
                    .unwrap_or(0.0),
                id,
                raw_time,
            )?;

            doses.push(dose_for_rate_mode(time, amt, cmt, rate, ss, ii, rate_mode));
            dose_rec.push(row_seq);
            if occ_col.is_some() {
                dose_occasions.push(occ);
            }
            if any_tv {
                dose_covariates.push(locf_state.clone());
            }
            // Advance the occasion watermark past this dose's *end* (start +
            // infusion duration), not just its start. A later reset-restarting
            // occasion is shifted past `max_eff_time`; if a dose here ends after
            // the last observation, the watermark must reflect that so the next
            // occasion doesn't land inside this one's dosing window. Reuses the
            // duration `DoseEvent::new` already computed (single source of truth).
            // NOTE: dose lagtime (ALAG) is a model parameter unknown at parse
            // time, so the watermark uses unlagged times; a heavily-lagged dose
            // whose effective start crosses an occasion boundary is not covered.
            let dose_end = {
                let d = doses.last().unwrap();
                d.time + d.duration
            };
            if dose_end > max_eff_time {
                max_eff_time = dose_end;
            }

            // ADDL expansion: add additional doses at time + k*II for k=1..=addl.
            let addl = addl_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(0);
            if addl > 0 {
                if ii <= 0.0 {
                    if !addl_missing_ii_warned {
                        parse_warnings.push(format!(
                            "W_ADDL_MISSING_II subject {}: ADDL > 0 but II is zero or \
                             missing; additional doses not expanded",
                            id
                        ));
                        addl_missing_ii_warned = true;
                    }
                } else {
                    for k in 1..=(addl as u32) {
                        let addl_time = time + (k as f64) * ii;
                        // Same construction as the primary dose (see
                        // `dose_for_rate_mode`): an expanded dose of a coded
                        // `RATE=-1`/`-2` row stays a modeled infusion rather than
                        // collapsing to a `Fixed` bolus, and is never steady-state
                        // itself (`ss = false`).
                        doses.push(dose_for_rate_mode(
                            addl_time, amt, cmt, rate, false, ii, rate_mode,
                        ));
                        dose_rec.push(row_seq);
                        if occ_col.is_some() {
                            dose_occasions.push(occ);
                        }
                        if any_tv {
                            dose_covariates.push(locf_state.clone());
                        }
                        // Fold each expanded dose's end into the watermark so a
                        // following reset-restarting occasion is shifted past the
                        // whole ADDL train (issue #195 review): ADDL bolus doses
                        // landing after the next occasion's reset would otherwise
                        // fire onto it, since boluses aren't gated by reset_floor.
                        // Reuses the just-pushed dose's stored duration.
                        let addl_end = {
                            let d = doses.last().unwrap();
                            d.time + d.duration
                        };
                        if addl_end > max_eff_time {
                            max_eff_time = addl_end;
                        }
                    }
                }
            }
        } else if evid == 0 && mdv == 0 {
            // Observation record
            let dv = parse_f64(row.get(dv_col).map(|s| s.as_str()).unwrap_or("0"));
            // Guard "." / blank the same way the dose path does: parse_usize maps
            // these to 0 (an invalid compartment), but a missing CMT on an
            // observation row must default to compartment 1.
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .and_then(|s| {
                    let t = s.trim();
                    if t == "." || t.is_empty() {
                        None
                    } else {
                        t.parse::<usize>().ok()
                    }
                })
                .unwrap_or(1);

            // A missing DV cell (`.` / `NA` / blank) means "no observation" when
            // the DV is an input: `parse_f64` coerced it to `0.0` above, so
            // without a guard a forgotten MDV would inject a phantom scored row.
            // NONMEM convention marks such rows MDV=1; treat a forgotten one the
            // same way — skip it and count it for the single W_MISSING_DV summary
            // (#258). Under `MissingDvPolicy::KeepAsDesign` the DV is instead the
            // *output* (simulation, #957) and the row is kept as a design point
            // with a placeholder DV. Applied to every scored endpoint below
            // (Gaussian, discrete-state, count); the TTE `Event` arm keeps its own
            // DV-code semantics and does not use it.
            let dv_missing = is_missing_cell(row.get(dv_col).map(|s| s.as_str()).unwrap_or(""));
            let skip_missing_dv = dv_missing && routing.missing_dv == MissingDvPolicy::Skip;

            // Non-Gaussian row routing: when this CMT belongs to a declared TTE /
            // discrete-state / count endpoint, route the row to `obs_records`
            // instead of the Gaussian parallel Vecs. The routing `.contains`
            // checks are always compiled; only the TTE `Event` construction is
            // `survival`-gated. The routing sets are disjoint (validated up front).
            if routing.tte.contains(&cmt) {
                #[cfg(feature = "survival")]
                {
                    use crate::types::{EventType, ObsRecord};
                    let raw_entry = _tentry_col
                        .and_then(|c| row.get(c))
                        .map(|s| parse_f64(s))
                        .unwrap_or(0.0)
                        .max(0.0);
                    if raw_entry > time + 1e-12 {
                        parse_warnings.push(format!(
                            "Subject {id}: TENTRY={raw_entry} > TIME={time} on CMT={cmt} \
                             — entry time after the event/censoring time yields a negative \
                             effective cumulative hazard; row skipped"
                        ));
                        // Skip this malformed row rather than producing an invalid NLL.
                        continue;
                    }
                    let entry_time = raw_entry;
                    // DV must be an integer code (0/1/2).  Reject fractional values
                    // explicitly: a DV of 1.9 would silently truncate to 1 (Exact event),
                    // misclassifying a censored observation.
                    let dv_rounded = dv.round();
                    if (dv - dv_rounded).abs() > 1e-9 {
                        return Err(format!(
                            "Subject {id}: TTE endpoint CMT={cmt} has non-integer DV={dv} \
                             at TIME={time}; DV must be 0 (right-censored), \
                             1 (exact event), or 2 (interval-censored right bound)"
                        ));
                    }
                    let dv_code = dv_rounded as i64;
                    match dv_code {
                        0 => {
                            // DV=0: tentatively a right-censored event, or left-bound of
                            // interval-censored pair. Save as pending; flush on next row.
                            // Flush any existing pending for this CMT first.
                            if let Some((t_left, e_left)) = tte_pending_left.remove(&cmt) {
                                obs_records.push(ObsRecord::Event {
                                    time: t_left,
                                    event_type: EventType::RightCensored,
                                    entry_time: e_left,
                                    cmt,
                                });
                            }
                            tte_pending_left.insert(cmt, (time, entry_time));
                        }
                        1 => {
                            // DV=1: exact event. Flush any pending left for this CMT.
                            if let Some((t_left, e_left)) = tte_pending_left.remove(&cmt) {
                                obs_records.push(ObsRecord::Event {
                                    time: t_left,
                                    event_type: EventType::RightCensored,
                                    entry_time: e_left,
                                    cmt,
                                });
                            }
                            obs_records.push(ObsRecord::Event {
                                time,
                                event_type: EventType::Exact,
                                entry_time,
                                cmt,
                            });
                        }
                        2 => {
                            // DV=2: interval-censored right-bound. Must follow a DV=0.
                            let left = tte_pending_left.remove(&cmt).ok_or_else(|| {
                                format!(
                                    "Subject {id}: DV=2 row at TIME={time} on CMT={cmt} \
                                     not preceded by a DV=0 row on the same CMT — \
                                     DV=2 marks the right bound of an interval-censored event"
                                )
                            })?;
                            let (t_left, e_left) = left;
                            obs_records.push(ObsRecord::Event {
                                time,
                                event_type: EventType::IntervalCensored {
                                    left: t_left,
                                    right: time,
                                },
                                entry_time: e_left,
                                cmt,
                            });
                        }
                        other => {
                            return Err(format!(
                                "Subject {id}: TTE endpoint CMT={cmt} has DV={other} \
                                 at TIME={time}; valid DV codes are 0 (right-censored), \
                                 1 (exact event), 2 (interval-censored right bound)"
                            ));
                        }
                    }
                }
                // No fallback arm needed: `routing.tte` is always empty when the
                // `survival` feature is off (its only producer is a TTE endpoint),
                // so this branch is never entered in that build.
            } else if let Some(kind) = routing.integer_kind(cmt) {
                // Non-Gaussian integer-coded observation (discrete-state index or
                // count). A missing DV is skipped as MDV=1 — the same #258
                // phantom-zero guard the Gaussian branch uses — so a forgotten MDV
                // never records a spurious `state:0` / `count:0`. Otherwise the DV
                // must be a finite, non-negative, in-range integer (`checked_integer_dv`,
                // #192); no endpoint math here (Phase 4.0).
                if dv_missing {
                    missing_dv_rows += 1;
                }
                if skip_missing_dv {
                    continue;
                }
                // Design-point row under `KeepAsDesign`: there is no user DV, so
                // write the endpoint's registered placeholder code (its first
                // declared state code; `0` for a count or an unregistered CMT).
                // The simulated outcome overwrites it. Skipping `checked_integer_dv`
                // here is deliberate — it is the *user's* DV that must be a valid
                // integer, and there isn't one — but the placeholder must still be
                // a code the endpoint can decode, so that a mis-routed design
                // population cannot silently score as an out-of-range state.
                let magnitude = if dv_missing {
                    routing.design_states.get(&cmt).copied().unwrap_or(0) as f64
                } else {
                    checked_integer_dv(dv, kind, id, cmt, time)?
                };
                obs_records.push(match kind {
                    IntDvKind::DiscreteState => crate::types::ObsRecord::DiscreteState {
                        time,
                        raw_time,
                        state: magnitude as usize,
                        cmt,
                    },
                    IntDvKind::Count => crate::types::ObsRecord::Count {
                        time,
                        raw_time,
                        count: magnitude as u32,
                        cmt,
                    },
                });
            } else {
                // Gaussian path. A missing DV (`.`/`NA`/blank) is skipped as MDV=1
                // (see the `dv_missing` note above; #258) so a forgotten MDV never
                // injects a phantom zero observation.
                if dv_missing {
                    missing_dv_rows += 1;
                }
                if skip_missing_dv {
                    continue;
                }
                // Design-point row under `KeepAsDesign` (#957): the sampling time
                // is real, the observation does not exist yet. `NaN` rather than
                // the `0.0` the missing cell parsed to, so the placeholder can
                // never be mistaken for a measured zero — the simulator reads only
                // the times, and every downstream |DV| consumer already guards on
                // `is_finite`.
                let dv = if dv_missing { f64::NAN } else { dv };
                let cens_flag = cens_col
                    .and_then(|c| row.get(c))
                    .map(|s| parse_cens(s))
                    .unwrap_or(0);
                obs_times.push(time);
                obs_rec.push(row_seq);
                obs_raw_times.push(raw_time);
                observations.push(dv);
                obs_cmts.push(cmt);
                // Only -1 (above ULOQ), 0 (quantified), and 1 (below LLOQ) are
                // meaningful. Any other value is coerced to left-censored by the
                // M3 likelihood (`m3_logcdf` treats every nonzero as a tail), so
                // flag it rather than silently mis-scoring the row.
                if !matches!(cens_flag, -1 | 0 | 1) && !cens_invalid_warned {
                    parse_warnings.push(format!(
                        "W_CENS_UNEXPECTED subject {}: CENS={} is not -1, 0, or 1; \
                         treated as censored (left tail) under M3",
                        id, cens_flag
                    ));
                    cens_invalid_warned = true;
                }
                cens.push(cens_flag);
                if occ_col.is_some() {
                    occasions.push(occ);
                }
                if fremtype_col.is_some() {
                    let ft = fremtype_col
                        .and_then(|c| row.get(c))
                        .and_then(|s| s.parse::<u16>().ok())
                        .unwrap_or(0);
                    fremtype.push(ft);
                }
                if l2_col.is_some() {
                    // NONMEM writes L2 as an integer id; a blank / unparsable
                    // cell means "ungrouped" (0), matching the empty-vector case.
                    let l2 = l2_col
                        .and_then(|c| row.get(c))
                        .and_then(|s| parse_l2_id(s))
                        .unwrap_or(0);
                    obs_l2.push(l2);
                }
                if any_tv {
                    obs_covariates.push(locf_state.clone());
                }
            }
        } else if evid == 2 && any_tv {
            // EVID=2 "other event" — typically a covariate-change marker.
            // NONMEM/nlmixr2 run $PK at this time with this row's
            // covariate values, so the rate matrix switches at this
            // time even though the row is neither a dose nor an obs.
            // We capture it as a pk-only event; the analytical / AD /
            // ODE event walkers will refresh `current_pk` from this
            // row's covariates without mutating the compartment state.
            //
            // Skipped entirely when there are no TV covariates: with
            // constant covariates re-evaluating $PK gives the same
            // values, so the row is a true no-op and adding it to the
            // event timeline would just be wasted work.
            pk_only_times.push(time);
            pk_only_covariates.push(locf_state.clone());
        }
    }

    // Sort doses by time (keeping dose_occasions and dose_covariates in sync).
    // Stable sort would be safer when two events share a time, but PartialOrd
    // sort_by gives us a stable answer for f64 ordered times, and matches
    // pre-existing behavior.
    let n_doses = doses.len();
    let mut perm: Vec<usize> = (0..n_doses).collect();
    perm.sort_by(|&a, &b| doses[a].time.partial_cmp(&doses[b].time).unwrap());
    let sorted_doses: Vec<DoseEvent> = perm.iter().map(|&i| doses[i].clone()).collect();
    let sorted_dose_rec: Vec<usize> = perm.iter().map(|&i| dose_rec[i]).collect();
    let sorted_dose_occ: Vec<u32> = if occ_col.is_some() {
        perm.iter().map(|&i| dose_occasions[i]).collect()
    } else {
        Vec::new()
    };
    let sorted_dose_cov: Vec<HashMap<String, f64>> = if any_tv {
        perm.iter().map(|&i| dose_covariates[i].clone()).collect()
    } else {
        Vec::new()
    };

    // Reset events are recorded in row order, which is usually time order;
    // sort defensively so the event-driven propagators see them in order. The
    // covariate snapshots ride the same permutation (#1133) — sorting the times
    // alone would silently pair each reset with another reset's `$PK` row.
    {
        let mut rperm: Vec<usize> = (0..reset_times.len()).collect();
        rperm.sort_by(|&a, &b| {
            reset_times[a]
                .partial_cmp(&reset_times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reset_times = rperm.iter().map(|&i| reset_times[i]).collect();
        if any_tv {
            reset_covariates = rperm.iter().map(|&i| reset_covariates[i].clone()).collect();
        }
    }

    // ── Honor NONMEM record order for a same-TIME observation/dose ──────────
    // NONMEM processes data records in file order: an observation listed BEFORE
    // a dose at the *same* TIME is a pre-dose (trough) sample and must not see
    // that dose; one listed AFTER is a post-dose sample that does. ferx's
    // solvers instead order events by time, and at equal time place the dose
    // first — both the event-driven sort (`kind_order`: Dose < Obs) and the
    // analytical superposition gate (`t_eff <= t`, so a dose at exactly the obs
    // time contributes). That turns every coincident trough into a post-dose
    // peak (e.g. infliximab run55: eval OFV 3751 vs NONMEM 662), railing fits to
    // their bounds.
    //
    // We restore record-order semantics by nudging a pre-dose observation one
    // ULP below the coincident dose time, so both solver paths evaluate it just
    // before the dose. The change is numerically inert for the prediction (a
    // single ULP) and leaves `obs_raw_times` — the user-clock value reported in
    // sdtab/covtab and by predict()/simulate() — untouched.
    //
    // Only a "pure trough" is nudged: every coincident dose is non-SS and comes
    // *after* the observation in the file. If any coincident dose precedes the
    // observation (a genuine post-dose sample, or a rare dose/obs/dose straddle)
    // it is left alone. SS doses are excluded: a sub-dose-time sample would read
    // 0 instead of the periodic pre-arrival trough the SS closed form supplies.
    for j in 0..obs_times.len() {
        let t = obs_times[j];
        let mut later_nonss = false;
        let mut earlier_any = false;
        for d in 0..sorted_doses.len() {
            if sorted_doses[d].time == t && !sorted_doses[d].ss {
                if sorted_dose_rec[d] > obs_rec[j] {
                    later_nonss = true;
                } else if sorted_dose_rec[d] < obs_rec[j] {
                    earlier_any = true;
                }
            }
        }
        if later_nonss && !earlier_any {
            obs_times[j] = t.next_down();
        }
    }

    // Flush any remaining pending TTE left-bounds as right-censored events.
    // This handles the common case: final row is a right-censored DV=0 with no
    // following DV=2 — the subject was censored at its last observation time.
    #[cfg(feature = "survival")]
    for (cmt, (t_left, e_left)) in tte_pending_left {
        obs_records.push(crate::types::ObsRecord::Event {
            time: t_left,
            event_type: crate::types::EventType::RightCensored,
            entry_time: e_left,
            cmt,
        });
    }

    Ok((
        Subject {
            id: id.to_string(),
            doses: sorted_doses,
            obs_times,
            obs_raw_times,
            observations,
            obs_cmts,
            covariates,
            dose_covariates: sorted_dose_cov,
            obs_covariates,
            pk_only_times,
            pk_only_covariates,
            reset_times,
            reset_covariates,
            cens,
            occasions,
            obs_l2,
            dose_occasions: sorted_dose_occ,
            fremtype,
            obs_records,
        },
        occ_parse_failures,
        missing_dv_rows,
        SubjectExclusion {
            n_obs_excluded: excl_n_obs,
            n_dose_excluded: excl_n_dose,
            n_other_excluded: excl_n_other,
            fired: excl_fired,
        },
        parse_warnings,
        amt_ignored_rows,
    ))
}

#[cfg(test)]
#[path = "datareader_tests.rs"]
mod tests;
