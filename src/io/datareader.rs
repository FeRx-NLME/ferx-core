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

    /// Returns `(excluded, which)`:
    /// - `excluded = true` when the row should be dropped.
    /// - `which` is the source string of the first clause that fired (for logging).
    ///
    /// Checks short-circuit on the first match, so a record is attributed to the
    /// first rule that excludes it. A rule that only ever matches records already
    /// removed by an earlier rule therefore never appears in the fired-condition
    /// summary — see `docs/src/model-file/data-selection.md`.
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
/// CENS: 1=observation is below LLOQ (DV carries the LLOQ value); 0 otherwise
///
/// `iov_column`: when `Some(name)`, that column is read as the occasion index
/// (integer) and stored in `Subject::occasions` / `Subject::dose_occasions`.
/// The column is excluded from the covariate auto-detection list.
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
) -> Result<Population, String> {
    read_nonmem_csv_impl(
        path,
        covariate_columns,
        iov_column,
        None,
        None,
        &HashSet::new(),
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
    // Population reads the union of declared + referenced-but-undeclared columns,
    // declared first so the table's column order matches the declaration.
    let mut union: Vec<String> = decls.iter().map(|d| d.name.clone()).collect();
    for c in extra_columns {
        if !union.iter().any(|n| n == c) {
            union.push(c.clone());
        }
    }
    let union_refs: Vec<&str> = union.iter().map(|s| s.as_str()).collect();
    let (pop, table) = read_nonmem_csv_impl(
        path,
        Some(&union_refs),
        iov_column,
        Some(decls),
        None,
        &HashSet::new(),
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
    // When an explicit covariate list is supplied, make sure every covariate the
    // filter references is in it — otherwise a filtered column outside the list
    // would not be read and the condition would silently never fire. (With
    // `None`, the auto-detect path already reads every non-standard column, so no
    // augmentation is needed.) Symmetric with the `[covariates]` reader.
    let augmented: Option<Vec<String>> = covariate_columns.map(|cols| {
        let mut v: Vec<String> = cols.iter().map(|s| s.to_string()).collect();
        for c in filter.referenced_covariate_columns() {
            if !v.iter().any(|n| n.eq_ignore_ascii_case(&c)) {
                v.push(c);
            }
        }
        v
    });
    let cols_ref: Option<Vec<&str>> = augmented
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    read_nonmem_csv_impl(
        path,
        cols_ref.as_deref(),
        iov_column,
        None,
        Some(filter),
        &HashSet::new(),
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
    let mut union: Vec<String> = decls.iter().map(|d| d.name.clone()).collect();
    for c in extra_columns {
        if !union.iter().any(|n| n == c) {
            union.push(c.clone());
        }
    }
    // Ensure any covariate column referenced by an ignore/accept clause is read
    // into each subject's covariate map, even if the model never declared it.
    // Without this, a filter on an undeclared column would silently never fire
    // on the declared-`[covariates]` read path (`union` would lack the column,
    // so it would be absent from `locf_state`). Case-insensitive dedup: the
    // referenced names are lowercased, declared names may not be.
    for c in filter.referenced_covariate_columns() {
        if !union.iter().any(|n| n.eq_ignore_ascii_case(&c)) {
            union.push(c);
        }
    }
    let union_refs: Vec<&str> = union.iter().map(|s| s.as_str()).collect();
    let (pop, table) = read_nonmem_csv_impl(
        path,
        Some(&union_refs),
        iov_column,
        Some(decls),
        Some(filter),
        &HashSet::new(),
    )?;
    Ok((
        pop,
        table.expect("covariate table is built whenever table_decls is Some"),
    ))
}

// ── TTE-aware readers (pub(crate) — used by api::read_population_for) ────────

/// Like [`read_nonmem_csv_filtered`] but routes EVID=0 rows on `tte_cmts` to
/// `Subject::obs_records` instead of the Gaussian parallel Vecs.
///
/// Used by `api::read_population_for` when the model has one or more TTE endpoints.
pub(crate) fn read_nonmem_csv_filtered_tte(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    filter: Option<&SelectionFilter>,
    tte_cmts: &HashSet<usize>,
) -> Result<Population, String> {
    let augmented: Option<Vec<String>> = covariate_columns.map(|cols| {
        let mut v: Vec<String> = cols.iter().map(|s| s.to_string()).collect();
        if let Some(f) = filter {
            for c in f.referenced_covariate_columns() {
                if !v.iter().any(|n| n.eq_ignore_ascii_case(&c)) {
                    v.push(c);
                }
            }
        }
        v
    });
    let cols_ref: Option<Vec<&str>> = augmented
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    read_nonmem_csv_impl(
        path,
        cols_ref.as_deref(),
        iov_column,
        None,
        filter,
        tte_cmts,
    )
    .map(|(pop, _)| pop)
}

/// Like [`read_nonmem_csv_with_covariates_filtered`] but routes EVID=0 rows on
/// `tte_cmts` to `Subject::obs_records`.
pub(crate) fn read_nonmem_csv_with_covariates_tte(
    path: &Path,
    decls: &[CovariateDecl],
    extra_columns: &[String],
    iov_column: Option<&str>,
    filter: Option<&SelectionFilter>,
    tte_cmts: &HashSet<usize>,
) -> Result<(Population, CovariateTable), String> {
    let mut union: Vec<String> = decls.iter().map(|d| d.name.clone()).collect();
    for c in extra_columns {
        if !union.iter().any(|n| n == c) {
            union.push(c.clone());
        }
    }
    if let Some(f) = filter {
        for c in f.referenced_covariate_columns() {
            if !union.iter().any(|n| n.eq_ignore_ascii_case(&c)) {
                union.push(c);
            }
        }
    }
    let union_refs: Vec<&str> = union.iter().map(|s| s.as_str()).collect();
    let (pop, table) = read_nonmem_csv_impl(
        path,
        Some(&union_refs),
        iov_column,
        Some(decls),
        filter,
        tte_cmts,
    )?;
    Ok((
        pop,
        table.expect("covariate table is built whenever table_decls is Some"),
    ))
}

/// Shared CSV reader. `table_decls`, when `Some`, requests building a
/// [`CovariateTable`] over exactly those declared covariates — each must exist
/// as a column and is validated as numeric (non-numeric → hard error). The
/// columns in `covariate_columns` (a superset, including referenced-but-
/// undeclared covariates) are read into the [`Population`] leniently. `None` on
/// both is the legacy auto-detect [`read_nonmem_csv`] path.
///
/// `tte_cmts`: CMTs whose EVID=0 rows should be routed to `Subject::obs_records`
/// (TTE endpoint) instead of the Gaussian parallel Vecs. Empty for all-Gaussian models.
fn read_nonmem_csv_impl(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    table_decls: Option<&[CovariateDecl]>,
    filter: Option<&SelectionFilter>,
    tte_cmts: &HashSet<usize>,
) -> Result<(Population, Option<CovariateTable>), String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV: {}", e))?;

    // Preserve original header casing for covariate names. Standard NONMEM
    // columns are matched case-insensitively so that legacy CSVs (e.g. `Id`,
    // `TIME`) keep working; covariate lookups remain case-sensitive.
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Failed to read headers: {}", e))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

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
    // Absent in Gaussian-only datasets; only used when tte_cmts is non-empty.
    let tentry_col = col_idx_ci("tentry");

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
        "tentry", "fremtype",
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
                &cov_indices,
                filter,
                tte_cmts,
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

    // Missing-DV summary (issue #258): scored observation rows (EVID=0, MDV=0)
    // whose DV cell was missing were skipped rather than read as DV=0. Surfaced
    // via FitResult.warnings and `ferx check` (data path).
    if total_missing_dv > 0 {
        population_warnings.push(format!(
            "W_MISSING_DV: {} observation row(s) (EVID=0) had a missing DV (`.`/`NA`/blank) \
             but were not marked MDV=1; they were skipped (not scored as DV=0). Set MDV=1 \
             on intentionally-missing observations to silence this, or check for data errors.",
            total_missing_dv
        ));
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
        // a fit) and the dataset isn't TTE/survival (which legitimately has no PK
        // doses) — otherwise this would be a noisy false positive.
        let total_scored_obs: usize = subjects.iter().map(|s| s.observations.len()).sum();
        #[cfg(feature = "survival")]
        let any_tte = subjects.iter().any(|s| !s.obs_records.is_empty());
        #[cfg(not(feature = "survival"))]
        let any_tte = false;
        if total_scored_obs > 0 && !any_tte {
            population_warnings.push(format!(
                "W_NO_DOSES: parsed zero dose events across all {} subject(s) although scored \
                 observations are present. If this is a PK model, check that the dataset has an \
                 AMT column with EVID=1/4 dose rows (or a nonzero AMT when EVID is absent).",
                subjects.len()
            ));
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
/// `-2` is accepted as [`RateMode::ModeledDuration`] (#324). The datareader has
/// no model, so it cannot yet know whether a matching `D{cmt}` parameter exists
/// or whether the model is an ODE model — those checks move to the model+data
/// join ([`crate::api::check_model_data`]). `-1` (modeled rate, #324 Phase B),
/// any other negative, and non-finite values are rejected here: they are
/// unconditionally invalid regardless of the model. (Previously `-1`/`-2` fell
/// through to `rate > 0.0` and were silently treated as boluses — #324.)
fn validate_dose_rate(rate: f64, id: &str, time: f64) -> Result<RateMode, String> {
    if !rate.is_finite() {
        return Err(format!(
            "subject {id}, time {time}: RATE={rate} is not finite; expected 0 \
             (bolus), a positive infusion rate, or -2 (modeled duration)"
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
        Some(-1) => "RATE=-1 (NONMEM: infusion RATE modeled via R1 in $PK) is not yet \
             supported (tracked in #324); use RATE=-2 (modeled duration) or an \
             explicit positive RATE"
            .to_string(),
        _ => format!("RATE={rate} is a negative value that is not a recognised NONMEM code"),
    };
    Err(format!(
        "subject {id}, time {time}: {detail}. Recognised RATE values are 0 \
         (bolus), >0 (infusion rate), and -2 (modeled infusion duration)."
    ))
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
    cov_indices: &[(String, usize)],
    filter: Option<&SelectionFilter>,
    // CMTs to route to `obs_records` instead of the Gaussian parallel Vecs.
    // Empty for Gaussian-only models. Always available; feature-gated routing
    // runs inside `#[cfg(feature = "survival")]` blocks.
    tte_cmts: &HashSet<usize>,
    // Column index of the TENTRY (left-truncation time) column, if present.
    tentry_col: Option<usize>,
) -> Result<(Subject, usize, usize, SubjectExclusion, Vec<String>, usize), String> {
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    let mut obs_raw_times = Vec::new();
    let mut observations = Vec::new();
    let mut obs_cmts = Vec::new();
    let mut cens = Vec::new();
    let mut occasions: Vec<u32> = Vec::new();
    let mut dose_occasions: Vec<u32> = Vec::new();
    let mut fremtype: Vec<u16> = Vec::new();
    let mut occ_parse_failures: usize = 0;
    // EVID=0/MDV=0 rows whose DV cell was missing and were skipped (issue #258).
    let mut missing_dv_skipped: usize = 0;
    let mut excl_n_obs: usize = 0;
    let mut excl_n_dose: usize = 0;
    let mut excl_n_other: usize = 0;
    let mut excl_fired: Vec<String> = Vec::new();
    let mut parse_warnings: Vec<String> = Vec::new();
    let mut addl_missing_ii_warned = false;
    // Rows that survived the data-selection filter, carry a nonzero AMT, yet
    // were not classified as a dose (EVID not 1/4) — their AMT was silently
    // dropped. Reported as a population summary so a degenerate dose-free fit
    // can't pass unnoticed (#262). Counted post-filter so deliberately excluded
    // dose rows don't trip the warning.
    let mut amt_ignored_rows: usize = 0;

    // TTE state — only meaningful when tte_cmts is non-empty.
    // obs_records: finalised TTE observation records for this subject.
    // tte_pending_left: per-CMT pending DV=0 row (may be a left-bound for an interval
    //   or a right-censored event, depending on whether the next row is DV=2).
    //   Map value is (time, entry_time).
    #[cfg(feature = "survival")]
    let mut tte_obs_records: Vec<crate::types::ObsRecord> = Vec::new();
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

    for row in rows {
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
                .map(|s| parse_usize(s))
                .unwrap_or(0);
            let ctx = RowContext {
                id,
                time,
                dv: dv_for_ctx,
                evid,
                amt: amt_for_ctx,
                cmt: cmt_for_ctx,
                rate: rate_for_ctx,
                mdv: mdv as u32,
                cens: if cens_for_ctx > 0 { 1u8 } else { 0u8 },
                ii: ii_for_ctx,
                ss: ss_for_ctx,
                covariates: &locf_state,
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
            // -2 -> ModeledDuration (the duration is a `D{cmt}` parameter,
            // resolved at the model+data join); -1 / other negative / non-finite
            // are rejected. Uses `raw_time` so the message names the value the
            // user wrote, not the occasion-shifted engine time. `?` bubbles up
            // through `parse_subject`.
            let rate_mode = validate_dose_rate(rate, id, raw_time)?;
            let ii = ii_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let ss = ss_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s.trim()) >= 0.5)
                .unwrap_or(false);

            doses.push(match rate_mode {
                RateMode::Fixed => DoseEvent::new(time, amt, cmt, rate, ss, ii),
                RateMode::ModeledDuration => DoseEvent::modeled(time, amt, cmt, ss, ii, rate_mode),
            });
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
                        doses.push(DoseEvent::new(
                            addl_time, amt, cmt, rate,
                            false, // expanded doses are never SS themselves
                            ii,
                        ));
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

            // TTE row routing: when this CMT belongs to a TTE endpoint, route to
            // obs_records instead of the Gaussian parallel Vecs.
            // The `tte_cmts.contains` check is always compiled (HashSet is not
            // feature-gated); the inner ObsRecord construction is cfg-gated.
            if tte_cmts.contains(&cmt) {
                #[cfg(feature = "survival")]
                {
                    use crate::types::{EventType, ObsRecord};
                    let raw_entry = tentry_col
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
                                tte_obs_records.push(ObsRecord::Event {
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
                                tte_obs_records.push(ObsRecord::Event {
                                    time: t_left,
                                    event_type: EventType::RightCensored,
                                    entry_time: e_left,
                                    cmt,
                                });
                            }
                            tte_obs_records.push(ObsRecord::Event {
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
                            tte_obs_records.push(ObsRecord::Event {
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
                // Note: no fallback needed here. `tte_cmts` is always empty when the
                // `survival` feature is off (callers pass `&HashSet::new()`), so this
                // branch is never entered in that build. The dead cfg block was removed.
            } else {
                // Gaussian path.
                // Missing DV (`.` / `NA` / blank) on a scored observation row
                // (EVID=0, MDV=0): NONMEM convention is to mark these MDV=1, but
                // if the user didn't, `parse_f64` would coerce the cell to 0.0
                // and inject a phantom zero observation into the likelihood.
                // Treat a missing DV as MDV=1 — skip the row — and count it for a
                // single summary warning (W_MISSING_DV; issue #258).
                let dv_cell = row.get(dv_col).map(|s| s.as_str()).unwrap_or("");
                if is_missing_cell(dv_cell) {
                    missing_dv_skipped += 1;
                    continue;
                }
                let cens_flag = cens_col
                    .and_then(|c| row.get(c))
                    .map(|s| parse_usize(s))
                    .unwrap_or(0);
                obs_times.push(time);
                obs_raw_times.push(raw_time);
                observations.push(dv);
                obs_cmts.push(cmt);
                cens.push(if cens_flag > 0 { 1u8 } else { 0u8 });
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
    // sort defensively so the event-driven propagators see them in order.
    reset_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Flush any remaining pending TTE left-bounds as right-censored events.
    // This handles the common case: final row is a right-censored DV=0 with no
    // following DV=2 — the subject was censored at its last observation time.
    #[cfg(feature = "survival")]
    for (cmt, (t_left, e_left)) in tte_pending_left {
        tte_obs_records.push(crate::types::ObsRecord::Event {
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
            cens,
            occasions,
            dose_occasions: sorted_dose_occ,
            fremtype,
            #[cfg(feature = "survival")]
            obs_records: tte_obs_records,
        },
        occ_parse_failures,
        missing_dv_skipped,
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
mod tests {
    use super::*;
    use crate::types::CovariateKind;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // ── "no happy paths": malformed-input rejection ──────────────────────────
    // The reader's validation surface — branches where a bug silently corrupts
    // (or crashes on) real NONMEM datasets. All deterministic and fit-free:
    // feed malformed CSV, assert the exact error or warning. These error paths
    // are otherwise exercised only indirectly, if at all.

    #[test]
    fn missing_required_columns_are_rejected() {
        // ID / TIME / DV are mandatory; each missing one is a hard error that
        // names the absent column.
        let f = write_csv("TIME,DV,EVID,AMT\n0,.,1,100\n");
        let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
        assert!(err.contains("Missing ID column"), "{err}");

        let f = write_csv("ID,DV,EVID,AMT\n1,.,1,100\n");
        let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
        assert!(err.contains("Missing TIME column"), "{err}");

        let f = write_csv("ID,TIME,EVID,AMT\n1,0,1,100\n");
        let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
        assert!(err.contains("Missing DV column"), "{err}");
    }

    #[test]
    fn unknown_iov_column_is_rejected() {
        // A requested IOV column that isn't in the header is a hard error, not a
        // silent "no occasions" — otherwise an IOV model would quietly collapse.
        let f = write_csv("ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n");
        let err = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap_err();
        assert!(
            err.contains("iov_column 'OCC'") && err.contains("not found"),
            "{err}"
        );
    }

    #[test]
    fn declared_covariate_column_missing_is_rejected() {
        // `[covariates]` declares WT but the dataset has no WT column → hard
        // error (a silently-vanished covariate would evaluate to nothing).
        let f = write_csv("ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n");
        let decls = vec![CovariateDecl {
            name: "WT".to_string(),
            kind: CovariateKind::Continuous,
        }];
        let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
        assert!(err.contains(ERR_COV_MISSING_COLUMNS), "{err}");
        assert!(err.contains("WT"), "missing column should be named: {err}");
    }

    #[test]
    fn declared_covariate_non_numeric_value_is_rejected() {
        // A declared covariate must be numerically coded; a text value is a hard
        // error rather than a silent 0.0 that would bias the fit.
        let f = write_csv("ID,TIME,DV,EVID,AMT,WT\n1,0,.,1,100,heavy\n1,1,5.0,0,.,heavy\n");
        let decls = vec![CovariateDecl {
            name: "WT".to_string(),
            kind: CovariateKind::Continuous,
        }];
        let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
        assert!(err.contains(ERR_COV_NON_NUMERIC), "{err}");
        assert!(
            err.contains("WT"),
            "offending covariate should be named: {err}"
        );
    }

    #[test]
    fn unparseable_iov_occasion_values_warn_not_fail() {
        // A non-numeric occasion value doesn't abort the read: the row is
        // assigned occ=0 and surfaced as a W_IOV_OCC_MISSING population warning
        // so the user can clean the data (a hard error here would be too brittle).
        let f = write_csv("ID,TIME,DV,EVID,AMT,OCC\n1,0,.,1,100,x\n1,1,5.0,0,.,x\n");
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        assert!(
            pop.warnings.iter().any(|w| w.contains("W_IOV_OCC_MISSING")),
            "expected an IOV-occasion warning, got {:?}",
            pop.warnings
        );
    }

    #[test]
    fn test_obs_cmt_dot_defaults_to_compartment_one() {
        // Regression: a "." (missing) CMT on an observation row must default to
        // compartment 1, not 0. `parse_usize(".")` yields 0 — an invalid
        // compartment — so the observation path must guard "." / blank exactly
        // like the dose path does.
        let csv = "ID,TIME,DV,EVID,AMT,CMT\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert_eq!(
            subj.obs_cmts,
            vec![1],
            "obs CMT='.' must default to compartment 1, not 0"
        );
    }

    #[test]
    fn test_occ_absent_gives_empty_occasions() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n1,2,3.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(pop.subjects[0].occasions.is_empty());
        assert!(pop.subjects[0].dose_occasions.is_empty());
    }

    // ── #262: EVID-absent dose inference + dose-coverage warnings ─────────────

    #[test]
    fn no_evid_column_infers_dose_from_amt() {
        // No EVID column: NONMEM infers a dose from a nonzero AMT. Dose rows here
        // carry AMT>0 with MDV=1 (the #154 shape), which without inference are
        // neither dose (needs EVID 1/4) nor obs (needs EVID 0 & MDV 0) — silently
        // dropped. With inference they administer and the fit is non-degenerate.
        let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,.,1,100\n\
                   1,1,9.5,0,.\n\
                   1,2,7.3,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert_eq!(
            subj.doses.len(),
            1,
            "AMT>0 row should be inferred as a dose"
        );
        assert_eq!(subj.doses[0].amt, 100.0);
        assert_eq!(subj.observations, vec![9.5, 7.3]);
        // The dataset "just works" — no dose-coverage warnings.
        assert!(
            !pop.warnings
                .iter()
                .any(|w| w.contains("W_AMT_NOT_DOSED") || w.contains("W_NO_DOSES")),
            "inferred-dose dataset must not warn, got {:?}",
            pop.warnings
        );
    }

    #[test]
    fn no_evid_column_infers_multiple_doses_across_subjects() {
        // Two subjects, each with an AMT-coded dose and observations; no EVID.
        let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,.,1,10000\n\
                   1,1,4.2,0,.\n\
                   2,0,.,1,5000\n\
                   2,1,2.1,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert_eq!(pop.subjects.len(), 2);
        assert_eq!(pop.subjects[0].doses.len(), 1);
        assert_eq!(pop.subjects[0].doses[0].amt, 10000.0);
        assert_eq!(pop.subjects[1].doses[0].amt, 5000.0);
    }

    #[test]
    fn no_evid_zero_amt_all_observations_warns_no_doses() {
        // No EVID column and no nonzero AMT anywhere: nothing to infer, so the
        // population parses zero doses. With scored observations present this is
        // almost always a data error — surface the generic W_NO_DOSES backstop.
        let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,1.0,0,.\n\
                   1,1,5.0,0,0\n\
                   1,2,3.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(pop.subjects[0].doses.is_empty());
        assert_eq!(pop.subjects[0].observations.len(), 3);
        assert!(
            pop.warnings.iter().any(|w| w.contains("W_NO_DOSES")),
            "zero-dose population with observations should warn, got {:?}",
            pop.warnings
        );
        // Generic only — no AMT was ignored, so the specific warning stays silent.
        assert!(!pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")));
    }

    #[test]
    fn evid_present_amt_on_nondose_row_warns_amt_not_dosed() {
        // EVID column present (so no inference), but a dose row is mistyped
        // EVID=0 with AMT=5000 and MDV=1 — dropped entirely (not dose, not obs).
        // Its AMT is silently ignored; W_AMT_NOT_DOSED must catch it. The real
        // EVID=1 dose still administers.
        let csv = "ID,TIME,DV,EVID,AMT,MDV\n\
                   1,0,.,1,100,1\n\
                   1,0,.,0,5000,1\n\
                   1,1,5.0,0,.,0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert_eq!(
            subj.doses.len(),
            1,
            "the mistyped AMT=5000 row is not a dose"
        );
        assert_eq!(subj.doses[0].amt, 100.0);
        assert!(
            pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
            "ignored-AMT row should warn, got {:?}",
            pop.warnings
        );
        // Specific wins — the generic backstop must not also fire.
        assert!(!pop.warnings.iter().any(|w| w.contains("W_NO_DOSES")));
    }

    #[test]
    fn wellformed_evid_dataset_emits_no_dose_warnings() {
        // Regression: a normal EVID dataset (dose EVID=1, obs EVID=0) is wholly
        // unaffected — neither dose-coverage warning fires.
        let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,9.5,0,.\n\
                   1,2,7.3,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert_eq!(pop.subjects[0].doses.len(), 1);
        assert!(
            !pop.warnings
                .iter()
                .any(|w| w.contains("W_AMT_NOT_DOSED") || w.contains("W_NO_DOSES")),
            "well-formed EVID data must not warn, got {:?}",
            pop.warnings
        );
    }

    #[test]
    fn no_evid_inference_mirrored_in_covariate_table() {
        // The covariate table's per-row EVID must agree with how parse_subject
        // classified the row, including AMT-based inference when EVID is absent.
        let csv = "ID,TIME,DV,AMT,MDV,WT\n\
                   1,0,.,100,1,70\n\
                   1,1,5.0,.,0,70\n";
        let f = write_csv(csv);
        let decls = vec![CovariateDecl {
            name: "WT".to_string(),
            kind: CovariateKind::Continuous,
        }];
        let (pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
        assert_eq!(
            pop.subjects[0].doses.len(),
            1,
            "dose inferred on table path too"
        );
        assert_eq!(table.rows[0].evid, 1, "AMT>0 row's table EVID should be 1");
        assert_eq!(table.rows[1].evid, 0, "obs row's table EVID should be 0");
    }

    #[test]
    fn amt_not_dosed_counted_after_data_selection_filter() {
        // The AMT-ignored count is taken post-filter: a mistyped AMT row that the
        // data-selection filter removes must NOT trip W_AMT_NOT_DOSED, while the
        // same dataset read unfiltered does trip it. Locks the post-filter
        // placement so deliberately excluded dose rows don't cause false alarms.
        let csv = "ID,TIME,DV,EVID,AMT,MDV,STUDY\n\
                   1,0,.,1,100,1,1\n\
                   1,0,.,0,5000,1,2\n\
                   1,1,5.0,0,.,0,1\n";
        let f = write_csv(csv);

        // Unfiltered: the EVID=0/AMT=5000 row is dropped and its AMT flagged.
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(
            pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
            "unfiltered read should flag the ignored AMT, got {:?}",
            pop.warnings
        );

        // Filtered to exclude that row (STUDY==2): nothing is silently dropped,
        // so no warning.
        let filter = SelectionFilter::from_opts(&["STUDY == 2".to_string()], &[], &[]).unwrap();
        let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
        assert!(
            !pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
            "a filter-excluded AMT row must not warn, got {:?}",
            pop.warnings
        );
        assert_eq!(pop.subjects[0].doses.len(), 1);
    }

    #[test]
    fn scored_obs_carrying_amt_does_not_warn_amt_not_dosed() {
        // A *scored* observation (EVID=0, MDV=0) that carries a nonzero AMT —
        // e.g. a pipeline that forward-fills / LOCFs the AMT column across all
        // rows — must NOT trip W_AMT_NOT_DOSED: it is a real observation, not a
        // dropped dose (a NONMEM dose row is MDV=1). The EVID=1 dose administers
        // and both observations are recorded.
        let csv = "ID,TIME,DV,EVID,AMT,MDV\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,100,0\n\
                   1,2,3.0,0,100,0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert_eq!(subj.doses.len(), 1);
        assert_eq!(subj.observations, vec![5.0, 3.0]);
        assert!(
            !pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")),
            "a scored obs carrying a forward-filled AMT must not warn, got {:?}",
            pop.warnings
        );
    }

    #[test]
    fn nonfinite_amt_is_not_inferred_as_a_dose() {
        // Robustness: a stray non-finite AMT ('inf'/'nan') must not become an
        // infinite/NaN-amount dose. parse_f64 accepts 'inf' (Rust FromStr), so
        // without the is_dosing_amt finiteness guard `amt != 0.0` would be true
        // and the row would infer a bogus dose. With no EVID column it is
        // instead rejected, leaving zero doses.
        let csv = "ID,TIME,DV,MDV,AMT\n\
                   1,0,.,1,inf\n\
                   1,1,5.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(
            pop.subjects[0].doses.is_empty(),
            "a non-finite AMT must not be inferred as a dose, got {:?}",
            pop.subjects[0].doses
        );
        // The non-finite AMT is also not counted as an ignored dose-like AMT.
        assert!(!pop.warnings.iter().any(|w| w.contains("W_AMT_NOT_DOSED")));
    }

    // ── NONMEM coded RATE values (#324) ──────────────────────────────────────
    // `RATE` is overloaded: 0 = bolus, >0 = infusion rate, -1 = modeled rate
    // (R{n} in $PK), -2 = modeled duration (D{n} in $PK). `-2` is accepted as
    // `ModeledDuration` (the D{cmt}/engine check happens later at the model+data
    // join); `-1` and malformed values are still rejected loudly.
    // `validate_dose_rate` is the unit under test.

    #[test]
    fn validate_dose_rate_classifies_coded_and_malformed_values() {
        // -2 → modeled duration: accepted here; the D{cmt} existence / ODE-engine
        // check happens later at the model+data join, where the model is known.
        assert_eq!(
            validate_dose_rate(-2.0, "7", 0.0).unwrap(),
            RateMode::ModeledDuration
        );

        // -1 → modeled rate: not yet supported (#324 Phase B). The message names
        // RATE=-1 and R1 so a NONMEM user recognises it, plus subject/time.
        let e = validate_dose_rate(-1.0, "1", 2.5).unwrap_err();
        assert!(e.contains("RATE=-1") && e.contains("R1"), "{e}");
        assert!(e.contains("subject 1") && e.contains("time 2.5"), "{e}");

        // Other negatives are not recognised NONMEM codes; the message echoes
        // the offending value so the bad row is identifiable. `-1.5`/`-2.5` are
        // the regression guard for the integer-match: a non-integer must NOT be
        // rounded into the -1/-2 codes (it is rejected, not silently accepted as
        // modeled duration).
        for r in [-0.5, -1.5, -2.5, -3.0, -100.0] {
            let e = validate_dose_rate(r, "1", 0.0).unwrap_err();
            assert!(
                e.contains(&format!("RATE={r}")) && e.contains("negative value"),
                "r={r}: {e}"
            );
        }

        // Non-finite RATE on a dose row is malformed.
        for r in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let e = validate_dose_rate(r, "1", 0.0).unwrap_err();
            assert!(e.contains("not finite"), "r={r}: {e}");
        }

        // Ordinary data-driven rates classify as Fixed.
        assert_eq!(validate_dose_rate(0.0, "1", 0.0).unwrap(), RateMode::Fixed);
        assert_eq!(validate_dose_rate(50.0, "1", 0.0).unwrap(), RateMode::Fixed);
    }

    #[test]
    fn coded_rate_minus_one_on_dose_row_is_rejected() {
        // End-to-end regression for the silent-bolus bug: a RATE=-1 dose must
        // error at read time, naming the subject/time, not load as a bolus.
        let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                   1,0,.,1,100,1,-1,1\n\
                   1,1,5.0,0,.,.,.,0\n";
        let f = write_csv(csv);
        let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
        assert!(
            err.contains("RATE=-1") && err.contains("subject 1"),
            "{err}"
        );
    }

    #[test]
    fn positive_and_zero_rate_doses_still_parse() {
        // Don't break normal infusions/boluses: RATE=50 → duration = amt/rate,
        // RATE=0 → bolus (duration 0).
        let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                   1,0,.,1,500,1,50,1\n\
                   2,0,.,1,500,1,0,1\n\
                   1,1,5.0,0,.,.,.,0\n\
                   2,1,5.0,0,.,.,.,0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let inf = &pop.subjects[0].doses[0];
        assert!(inf.is_infusion() && (inf.duration - 10.0).abs() < 1e-12);
        let bolus = &pop.subjects[1].doses[0];
        assert!(!bolus.is_infusion() && bolus.duration == 0.0);
    }

    #[test]
    fn coded_rate_on_observation_row_is_ignored() {
        // NONMEM only interprets RATE on dose records. A coded RATE on an EVID=0
        // observation row must not error (it is never administered).
        let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                   1,0,.,1,100,1,0,1\n\
                   1,1,5.0,0,.,.,-1,0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert_eq!(pop.subjects[0].doses.len(), 1);
    }

    #[test]
    fn coded_rate_on_filtered_out_dose_row_does_not_error() {
        // The RATE check runs in the dose arm, after the data-selection filter
        // (`continue` on an excluded row). A coded RATE on a row the user IGNOREs
        // must not error — only administered doses are validated.
        let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,FLAG\n\
                   1,0,.,1,100,1,0,1,1\n\
                   1,0.5,.,1,100,1,-2,1,9\n\
                   1,1,5.0,0,.,.,.,0,1\n";
        let f = write_csv(csv);
        let filter = SelectionFilter::from_opts(&["FLAG == 9".to_string()], &[], &[]).unwrap();
        let pop = read_nonmem_csv_filtered(f.path(), None, None, &filter).unwrap();
        // The coded-RATE dose row was filtered out; the normal dose survives.
        assert_eq!(pop.subjects[0].doses.len(), 1);
        assert!(!pop.subjects[0].doses[0].is_infusion());
    }

    #[test]
    fn test_evid3_reset_recorded_not_dose_or_obs() {
        // EVID=3 is a pure system reset: it must land in `reset_times` and
        // must NOT be parsed as a dose or an observation.
        let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,5.0,0,.\n\
                   1,5,.,3,.\n\
                   1,6,2.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        // Also the no-shift guard for a *forward* reset (TIME=5 advances past
        // the prior obs at TIME=1): the occasion-segmentation shift must only
        // fire on a restarting clock, so reset_times and obs_times keep their
        // raw values here (a spurious shift would push the reset past 5 and
        // move the t=6 obs).
        assert_eq!(subj.reset_times, vec![5.0]);
        assert!(subj.has_resets());
        // One dose (t=0), two observations (t=1, t=6) — the reset row is neither.
        assert_eq!(subj.doses.len(), 1);
        assert_eq!(subj.doses[0].time, 0.0);
        assert_eq!(subj.obs_times, vec![1.0, 6.0]);
    }

    #[test]
    fn test_filter_on_undeclared_covariate_via_declared_path() {
        // Regression: a `[data_selection]` condition on a covariate that the
        // `[covariates]` block did NOT declare must still fire on the declared
        // read path. Here only WT is declared; `ignore = STUDY == 2` references
        // the undeclared STUDY column. `referenced_covariate_columns()` must
        // pull STUDY into the read union so it lands in each subject's covariate
        // map — otherwise the condition would silently never match.
        // STUDY is a covariate independent of ID: subjects 1 & 2 are in STUDY 1,
        // subjects 3 & 4 in STUDY 2. Filtering on STUDY (not ID) must drop a whole
        // study (3 and 4) while keeping the other (1 and 2). The not-1:1 mapping
        // ensures the test exercises covariate filtering, not ID matching.
        let csv = "ID,TIME,DV,EVID,AMT,CMT,WT,STUDY\n\
                   1,0,.,1,100,1,70,1\n\
                   1,1,5.0,0,.,1,70,1\n\
                   2,0,.,1,100,1,72,1\n\
                   2,1,4.5,0,.,1,72,1\n\
                   3,0,.,1,100,1,80,2\n\
                   3,1,4.0,0,.,1,80,2\n\
                   4,0,.,1,100,1,85,2\n\
                   4,1,3.5,0,.,1,85,2\n";
        let f = write_csv(csv);
        let decls = vec![CovariateDecl {
            name: "WT".to_string(),
            kind: CovariateKind::Continuous,
        }];
        let filter = SelectionFilter::from_opts(&["STUDY == 2".to_string()], &[], &[]).unwrap();
        let (pop, _table) =
            read_nonmem_csv_with_covariates_filtered(f.path(), &decls, &[], None, &filter).unwrap();
        // Both STUDY==2 subjects (3 and 4) are excluded; STUDY==1 subjects remain.
        let ids: Vec<&str> = pop.subjects.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["1", "2"], "only STUDY==1 subjects should remain");
        let excl = pop.exclusions.as_ref().expect("exclusions present");
        assert_eq!(
            excl.excluded_subject_ids,
            vec!["3".to_string(), "4".to_string()]
        );
        assert!(excl.fired_ignore.iter().any(|s| s.contains("STUDY == 2")));
    }

    #[test]
    fn test_no_filter_keeps_subject_with_only_other_events() {
        // Regression: without a [data_selection] filter, a subject made up solely
        // of EVID=2 (other-event) rows has empty doses+observations but must
        // still be retained — matching the pre-feature reader, which pushed every
        // subject unconditionally. The empty-subject skip must be gated on the
        // filter actually having excluded records.
        let csv = "ID,TIME,DV,EVID,AMT,CMT\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   2,0,.,2,.,1\n\
                   2,1,.,2,.,1\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let ids: Vec<&str> = pop.subjects.iter().map(|s| s.id.as_str()).collect();
        assert!(
            ids.contains(&"2"),
            "subject with only EVID=2 rows must be retained when no filter is active; got {ids:?}"
        );
        assert!(pop.exclusions.is_none(), "no filter → no exclusion summary");
    }

    #[test]
    fn test_evid4_reset_plus_dose_recorded_as_both() {
        // EVID=4 is reset + dose: it records both a reset time and a dose.
        let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,1,100\n\
                   1,1,5.0,0,.\n\
                   1,10,.,4,200\n\
                   1,11,3.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert_eq!(subj.reset_times, vec![10.0]);
        // Two doses: the t=0 dose and the EVID=4 dose at t=10.
        assert_eq!(subj.doses.len(), 2);
        assert!(subj.doses.iter().any(|d| d.time == 10.0 && d.amt == 200.0));
        assert_eq!(subj.obs_times, vec![1.0, 11.0]);
    }

    #[test]
    fn test_no_resets_leaves_reset_times_empty() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(pop.subjects[0].reset_times.is_empty());
        assert!(!pop.subjects[0].has_resets());
    }

    #[test]
    fn test_evid4_restart_shifts_second_occasion_onto_monotonic_timeline() {
        // Two dosing occasions stacked under one ID, each opened by an EVID=4
        // reset whose TIME column restarts at 0 (NONMEM processes records
        // sequentially, so this is a fresh occasion sharing the first's clock).
        // The reader must shift the second occasion past the first so the two
        // don't collide on the sorted absolute timeline — otherwise both doses
        // land at t=0 and the subject is double-dosed (issue #195).
        let csv = "ID,TIME,DV,EVID,AMT\n\
                   1,0,.,4,100\n\
                   1,2,5.0,0,.\n\
                   1,8,2.0,0,.\n\
                   1,0,.,4,100\n\
                   1,2,4.0,0,.\n\
                   1,8,1.5,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];

        // Two distinct reset times (the second shifted past the first occasion).
        assert_eq!(subj.reset_times.len(), 2);
        assert_eq!(subj.reset_times[0], 0.0);
        assert!(
            subj.reset_times[1] > 8.0,
            "second reset must be shifted past the first occasion's last event (t=8), got {}",
            subj.reset_times[1]
        );

        // Two doses, no longer colliding at t=0.
        assert_eq!(subj.doses.len(), 2);
        assert_eq!(subj.doses[0].time, 0.0);
        assert_eq!(subj.doses[1].time, subj.reset_times[1]);

        // Observation times are strictly increasing across the occasion
        // boundary (the second occasion's relative spacing is preserved).
        assert_eq!(subj.obs_times.len(), 4);
        for w in subj.obs_times.windows(2) {
            assert!(
                w[1] > w[0],
                "obs times must be monotonic: {:?}",
                subj.obs_times
            );
        }
        // Within-occasion spacing is unchanged: second occasion's two obs are
        // still 6 time units apart (raw t=2 and t=8).
        let gap2 = subj.obs_times[3] - subj.obs_times[2];
        assert!(
            (gap2 - 6.0).abs() < 1e-9,
            "second occasion spacing preserved"
        );
    }

    #[test]
    fn test_addl_train_advances_occasion_watermark() {
        // Regression for the issue #195 review: occasion 1 carries an ADDL dose
        // train (II=10, ADDL=3 → boluses at 0,10,20,30) but its only observation
        // is at t=5. A following EVID=4 occasion restarts at TIME=0. The
        // occasion shift must place the new reset past the *whole* ADDL train,
        // not just past the last observation — otherwise the later ADDL boluses
        // (which a reset does not cancel) would land after the reset and
        // contaminate occasion 2.
        let csv = "ID,TIME,DV,EVID,AMT,II,ADDL\n\
                   1,0,.,1,100,10,3\n\
                   1,5,5.0,0,.,.,.\n\
                   1,0,.,4,100,0,0\n\
                   1,5,4.0,0,.,.,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];

        // Occasion 1 doses: 0,10,20,30 (raw). The single reset (occasion 2)
        // must be shifted strictly past the last ADDL dose at t=30.
        assert_eq!(subj.reset_times.len(), 1);
        let reset = subj.reset_times[0];
        assert!(
            reset > 30.0,
            "occasion-2 reset must be shifted past the ADDL train (last dose t=30), got {reset}"
        );
        // No occasion-1 dose lands at or after the reset (which would inject it
        // into occasion 2). Exactly one dose — occasion 2's own — is >= reset.
        let after_reset = subj.doses.iter().filter(|d| d.time >= reset - 1e-9).count();
        assert_eq!(
            after_reset,
            1,
            "only occasion 2's own dose may sit at/after its reset; doses={:?}",
            subj.doses.iter().map(|d| d.time).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_parse_subject_reads_occ_column() {
        let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,1\n\
                   1,7,.,1,100,2\n\
                   1,8,4.0,0,.,2\n\
                   1,9,2.5,0,.,2\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        let subj = &pop.subjects[0];
        // Two obs in occ 1, two in occ 2 (dose rows are stripped from occasions)
        assert_eq!(subj.occasions, vec![1, 1, 2, 2]);
        assert_eq!(subj.dose_occasions, vec![1, 2]);
    }

    #[test]
    fn test_occ_column_excluded_from_covariates() {
        let csv = "ID,TIME,DV,EVID,AMT,OCC,WT\n\
                   1,0,.,1,100,1,70\n\
                   1,1,5.0,0,.,1,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        // OCC should NOT appear as a covariate; WT should
        assert!(!pop.covariate_names.contains(&"OCC".to_string()));
        assert!(pop.covariate_names.contains(&"WT".to_string()));
    }

    #[test]
    fn test_missing_iov_column_errors() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n";
        let f = write_csv(csv);
        let result = read_nonmem_csv(f.path(), None, Some("OCC"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("iov_column"));
    }

    #[test]
    fn test_parse_occ_recognizes_missing_sentinels() {
        // NONMEM-style "." plus blanks/NAs should parse as None.
        assert_eq!(parse_occ(""), None);
        assert_eq!(parse_occ("."), None);
        assert_eq!(parse_occ("  "), None);
        assert_eq!(parse_occ("NA"), None);
        assert_eq!(parse_occ("nan"), None);
        // Non-integer or signed values that u32 can't parse
        assert_eq!(parse_occ("1.5"), None);
        assert_eq!(parse_occ("-1"), None);
        // Valid u32 round-trips
        assert_eq!(parse_occ("1"), Some(1));
        assert_eq!(parse_occ("42"), Some(42));
    }

    #[test]
    fn test_missing_occ_value_does_not_break_load_but_falls_back_to_zero() {
        // Row with OCC = "." gets occ=0; load still succeeds (warning is
        // emitted to stderr, not asserted here).
        let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        let subj = &pop.subjects[0];
        // Two obs: first has OCC=1, second had "." → 0
        assert_eq!(subj.occasions, vec![1, 0]);
    }

    #[test]
    fn test_no_tv_covariates_leaves_per_event_snapshots_empty() {
        // WT is constant — no per-event snapshots should be allocated.
        let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,0,.,70\n\
                   1,2,3.0,0,.,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert!(!subj.has_tv_covariates());
        assert!(subj.dose_covariates.is_empty());
        assert!(subj.obs_covariates.is_empty());
        // Static covariate map still populated.
        assert_eq!(subj.covariates["WT"], 70.0);
        // Fallback helpers return the static map.
        assert_eq!(subj.dose_cov(0)["WT"], 70.0);
        assert_eq!(subj.obs_cov(0)["WT"], 70.0);
    }

    #[test]
    fn test_tv_covariate_locf_per_event_snapshot() {
        // CR changes mid-subject. Each event must see the LOCF value at its
        // own row's time (NONMEM $PK semantics).
        let csv = "ID,TIME,DV,EVID,AMT,WT,CR\n\
                   1,0,.,1,100,70,1.0\n\
                   1,1,5.0,0,.,70,1.0\n\
                   1,2,3.0,0,.,70,1.5\n\
                   1,3,.,1,100,70,1.5\n\
                   1,4,2.5,0,.,70,2.0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert!(subj.has_tv_covariates());
        // 2 doses, 3 obs.
        assert_eq!(subj.dose_covariates.len(), 2);
        assert_eq!(subj.obs_covariates.len(), 3);
        // Static covariate map is the *first* CR value (1.0), kept for
        // the AD fast-path / fallback.
        assert_eq!(subj.covariates["CR"], 1.0);
        // Dose 1 (t=0): CR=1.0; Dose 2 (t=3): CR=1.5
        assert_eq!(subj.dose_covariates[0]["CR"], 1.0);
        assert_eq!(subj.dose_covariates[1]["CR"], 1.5);
        // Obs at t=1: CR=1.0; t=2: CR=1.5; t=4: CR=2.0
        assert_eq!(subj.obs_covariates[0]["CR"], 1.0);
        assert_eq!(subj.obs_covariates[1]["CR"], 1.5);
        assert_eq!(subj.obs_covariates[2]["CR"], 2.0);
        // WT is constant — but appears in every snapshot at its constant value.
        assert_eq!(subj.dose_covariates[0]["WT"], 70.0);
        assert_eq!(subj.obs_covariates[2]["WT"], 70.0);
    }

    #[test]
    fn test_tv_covariate_snapshot_keeps_dose_sort_alignment() {
        // Doses arrive in non-time order in the CSV. After sorting doses by
        // time, dose_covariates must follow the same permutation so each
        // dose still pairs with its own snapshot.
        let csv = "ID,TIME,DV,EVID,AMT,CR\n\
                   1,5,.,1,100,2.0\n\
                   1,0,.,1,100,1.0\n\
                   1,6,5.0,0,.,2.0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert!(subj.has_tv_covariates());
        // After sorting: dose 0 = t=0 (CR=1.0), dose 1 = t=5 (CR=2.0).
        assert_eq!(subj.doses[0].time, 0.0);
        assert_eq!(subj.doses[1].time, 5.0);
        assert_eq!(subj.dose_covariates[0]["CR"], 1.0);
        assert_eq!(subj.dose_covariates[1]["CR"], 2.0);
    }

    #[test]
    fn test_evid2_rows_captured_with_locf_covariates() {
        // EVID=2 ("other event") rows with TV covariates should be
        // captured into pk_only_times / pk_only_covariates so the
        // event-driven propagator can refresh the rate matrix at the
        // EVID=2 time. NONMEM/nlmixr2 equivalent: $PK runs at every
        // record (including EVID=2), so a covariate change marker
        // should switch CL/V immediately at its time.
        let csv = "ID,TIME,DV,EVID,MDV,AMT,CR\n\
                   1,0,.,1,1,100,1.0\n\
                   1,5,.,2,1,0,1.5\n\
                   1,10,5.0,0,0,.,1.5\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];

        // 1 dose, 1 obs, 1 pk-only event.
        assert_eq!(subj.doses.len(), 1);
        assert_eq!(subj.obs_times.len(), 1);
        assert_eq!(subj.pk_only_times.len(), 1);
        assert_eq!(subj.pk_only_times[0], 5.0);
        // EVID=2 row carries CR=1.5 — must end up in the snapshot.
        assert_eq!(subj.pk_only_covariates[0]["CR"], 1.5);
        // Subsequent obs sees the LOCF value.
        assert_eq!(subj.obs_covariates[0]["CR"], 1.5);
    }

    #[test]
    fn test_evid2_rows_skipped_when_no_tv_covariates() {
        // With time-constant covariates, EVID=2 rows are no-ops in
        // NONMEM ($PK gives the same values), so we don't bother
        // building snapshots for them — saves allocation. This test
        // locks in that optimization.
        let csv = "ID,TIME,DV,EVID,MDV,AMT,WT\n\
                   1,0,.,1,1,100,70\n\
                   1,5,.,2,1,0,70\n\
                   1,10,5.0,0,0,.,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];

        assert!(!subj.has_tv_covariates());
        assert!(subj.pk_only_times.is_empty());
        assert!(subj.pk_only_covariates.is_empty());
    }

    #[test]
    fn test_missing_dv_obs_skipped_and_warned() {
        // Issue #258: an EVID=0 row with a missing DV and no MDV=1 must be
        // skipped (not scored as DV=0), and a single W_MISSING_DV warning fires.
        let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,5.0,0,0,.,1\n\
                   1,2,.,0,0,.,1\n\
                   1,3,7.0,0,0,.,1\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];

        // Only the two valid observations are scored; the missing-DV row at t=2
        // is skipped (no phantom 0.0 observation, no t=2 entry).
        assert_eq!(subj.observations, vec![5.0, 7.0]);
        assert_eq!(subj.obs_times, vec![1.0, 3.0]);

        // Exactly one summary warning, reporting a single skipped row.
        let warns: Vec<&String> = pop
            .warnings
            .iter()
            .filter(|w| w.starts_with("W_MISSING_DV"))
            .collect();
        assert_eq!(warns.len(), 1, "expected one W_MISSING_DV summary warning");
        assert!(warns[0].contains("1 observation row"), "got: {}", warns[0]);
    }

    #[test]
    fn test_missing_dv_with_mdv1_no_warning() {
        // The same missing-DV row marked MDV=1 is the documented convention and
        // must NOT trigger the W_MISSING_DV warning (it's already handled).
        let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,5.0,0,0,.,1\n\
                   1,2,.,0,1,.,1\n\
                   1,3,7.0,0,0,.,1\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert_eq!(subj.observations, vec![5.0, 7.0]);
        assert!(
            !pop.warnings.iter().any(|w| w.starts_with("W_MISSING_DV")),
            "MDV=1 missing-DV row should not warn"
        );
    }

    #[test]
    fn test_missing_dv_count_aggregates_across_subjects() {
        // Issue #258: the per-subject missing-DV counts are summed into ONE
        // population warning. Two subjects, one skipped row each → a single
        // W_MISSING_DV reporting two rows (plural), not two warnings.
        let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,.,0,0,.,1\n\
                   1,2,7.0,0,0,.,1\n\
                   2,0,.,1,1,100,1\n\
                   2,1,5.0,0,0,.,1\n\
                   2,2,.,0,0,.,1\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();

        // Each subject keeps only its single valid observation.
        assert_eq!(pop.subjects[0].observations, vec![7.0]);
        assert_eq!(pop.subjects[1].observations, vec![5.0]);

        let warns: Vec<&String> = pop
            .warnings
            .iter()
            .filter(|w| w.starts_with("W_MISSING_DV"))
            .collect();
        assert_eq!(warns.len(), 1, "expected one aggregated W_MISSING_DV");
        assert!(
            warns[0].contains("2 observation row"),
            "expected aggregated count of 2, got: {}",
            warns[0]
        );
    }

    #[test]
    fn test_missing_dv_recognizes_na_nan_and_blank_sentinels() {
        // `is_missing_cell` treats `.`, `NA`/`na`, `NaN`/`nan`, and blank as
        // missing — all of these on a scored obs row must be skipped and counted,
        // not just the `.` sentinel exercised by the other tests.
        let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,NA,0,0,.,1\n\
                   1,2,nan,0,0,.,1\n\
                   1,3,,0,0,.,1\n\
                   1,4,6.0,0,0,.,1\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];

        // Only the single numeric observation survives.
        assert_eq!(subj.observations, vec![6.0]);
        assert_eq!(subj.obs_times, vec![4.0]);

        let warns: Vec<&String> = pop
            .warnings
            .iter()
            .filter(|w| w.starts_with("W_MISSING_DV"))
            .collect();
        assert_eq!(warns.len(), 1);
        assert!(
            warns[0].contains("3 observation row"),
            "expected 3 skipped (NA, nan, blank), got: {}",
            warns[0]
        );
    }

    #[test]
    fn test_missing_dv_and_amt_not_dosed_warnings_coexist() {
        // The missing-DV summary (#258) and the dose-coverage summary (#262) are
        // independent population warnings and must both fire when a dataset trips
        // both: a missing-DV scored obs row AND a nonzero-AMT row that is not a
        // dose (EVID=2, MDV=1) so its AMT is ignored.
        let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT\n\
                   1,0,.,1,1,100,1\n\
                   1,1,.,0,0,.,1\n\
                   1,2,5.0,0,0,.,1\n\
                   1,3,.,2,1,5000,1\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();

        assert_eq!(pop.subjects[0].observations, vec![5.0]);
        assert!(
            pop.warnings.iter().any(|w| w.starts_with("W_MISSING_DV")),
            "missing-DV warning should fire; warnings: {:?}",
            pop.warnings
        );
        assert!(
            pop.warnings
                .iter()
                .any(|w| w.starts_with("W_AMT_NOT_DOSED")),
            "AMT-not-dosed warning should fire; warnings: {:?}",
            pop.warnings
        );
    }

    fn decl(name: &str, kind: CovariateKind) -> CovariateDecl {
        CovariateDecl {
            name: name.to_string(),
            kind,
        }
    }

    #[test]
    fn test_covariate_table_one_row_per_input_record() {
        // 1 dose + 2 obs = 3 input rows → 3 table rows, including the dose row.
        let csv = "ID,TIME,DV,EVID,AMT,WT,SEX\n\
                   1,0,.,1,100,70,1\n\
                   1,1,5.0,0,.,70,1\n\
                   1,2,3.0,0,.,70,1\n";
        let f = write_csv(csv);
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        let (_pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
        assert_eq!(table.names, vec!["WT", "SEX"]);
        assert_eq!(
            table.kinds,
            vec![CovariateKind::Continuous, CovariateKind::Categorical]
        );
        assert_eq!(table.rows.len(), 3);
        // Dose row preserved with EVID=1.
        assert_eq!(table.rows[0].evid, 1);
        assert_eq!(table.rows[0].time, 0.0);
        assert_eq!(table.rows[0].values, vec![70.0, 1.0]);
        assert_eq!(table.rows[1].evid, 0);
        assert_eq!(table.rows[2].time, 2.0);
    }

    #[test]
    fn test_covariate_table_missing_value_is_nan() {
        let csv = "ID,TIME,DV,EVID,AMT,WT,SEX\n\
                   1,0,.,1,100,70,.\n\
                   1,1,5.0,0,.,,1\n";
        let f = write_csv(csv);
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        let (_pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
        // Row 0: SEX is "." → NaN. Row 1: WT is blank → NaN.
        assert!(table.rows[0].values[1].is_nan());
        assert!(table.rows[1].values[0].is_nan());
        assert_eq!(table.rows[0].values[0], 70.0);
    }

    #[test]
    fn test_covariate_strict_numeric_errors_on_non_numeric() {
        let csv = "ID,TIME,DV,EVID,AMT,WT,SEX\n\
                   1,0,.,1,100,70,M\n\
                   1,1,5.0,0,.,70,M\n";
        let f = write_csv(csv);
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
        assert!(err.contains("non-numeric"), "got: {err}");
        assert!(err.contains("SEX"), "got: {err}");
    }

    #[test]
    fn test_covariate_declared_column_missing_errors() {
        let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n";
        let f = write_csv(csv);
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("CRCL", CovariateKind::Continuous),
        ];
        let err = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
        assert!(err.contains("CRCL"), "got: {err}");
    }

    #[test]
    fn test_parse_evid_defaults_to_observation() {
        // parse_usize defaults to 1; parse_evid must default to 0 (observation)
        // for blank / missing / unparseable cells.
        assert_eq!(parse_evid("1"), 1);
        assert_eq!(parse_evid("0"), 0);
        assert_eq!(parse_evid(""), 0);
        assert_eq!(parse_evid("."), 0);
        assert_eq!(parse_evid("NA"), 0);
        assert_eq!(parse_evid("x"), 0);
    }

    #[test]
    fn test_covtab_blank_evid_is_observation_not_dose() {
        // A blank EVID cell on an observation row must be EVID=0 in the covtab,
        // not 1 (which parse_usize would have produced).
        let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,,.,70\n";
        let f = write_csv(csv);
        let decls = vec![decl("WT", CovariateKind::Continuous)];
        let (_pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &[], None).unwrap();
        assert_eq!(table.rows[0].evid, 1); // explicit dose row
        assert_eq!(table.rows[1].evid, 0); // blank EVID → observation
    }

    #[test]
    fn test_absent_extra_covariate_excluded_from_covariate_names() {
        // A referenced-but-undeclared covariate passed in `extra` that is NOT a
        // real column must not appear in covariate_names — otherwise the fit's
        // E_MISSING_COVARIATE guard would be masked and it would silently read
        // as 0.0. (Regression test for the masking bug.)
        let csv = "ID,TIME,DV,EVID,AMT,WT,CRCL\n\
                   1,0,.,1,100,70,80\n\
                   1,1,5.0,0,.,70,80\n";
        let f = write_csv(csv);
        let decls = vec![decl("WT", CovariateKind::Continuous)];
        let extra = vec!["GHOST".to_string()]; // not a column in the CSV
        let (pop, table) = read_nonmem_csv_with_covariates(f.path(), &decls, &extra, None).unwrap();
        assert!(!pop.covariate_names.contains(&"GHOST".to_string()));
        assert!(pop.covariate_names.contains(&"WT".to_string()));
        // The table still reflects only declared columns.
        assert_eq!(table.names, vec!["WT"]);
    }

    #[test]
    fn test_legacy_read_still_succeeds_with_non_numeric_covariate() {
        // The legacy auto-detect path must remain unchanged (no strict numeric
        // check): a non-numeric covariate column loads without erroring.
        let csv = "ID,TIME,DV,EVID,AMT,SEX\n\
                   1,0,.,1,100,M\n\
                   1,1,5.0,0,.,M\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(pop.covariate_names.contains(&"SEX".to_string()));
    }

    #[test]
    fn test_tv_covariate_locf_handles_missing_intermediate() {
        // Missing CR values are filled forward (LOCF), matching NONMEM.
        let csv = "ID,TIME,DV,EVID,AMT,CR\n\
                   1,0,.,1,100,1.0\n\
                   1,1,5.0,0,.,.\n\
                   1,2,3.0,0,.,2.0\n\
                   1,3,2.0,0,.,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        // Obs 0 (t=1, CR missing) → LOCF → 1.0.
        assert_eq!(subj.obs_covariates[0]["CR"], 1.0);
        // Obs 1 (t=2, CR=2.0) → 2.0.
        assert_eq!(subj.obs_covariates[1]["CR"], 2.0);
        // Obs 2 (t=3, CR missing) → LOCF → 2.0.
        assert_eq!(subj.obs_covariates[2]["CR"], 2.0);
    }

    #[test]
    fn test_input_columns_preserves_full_header_order() {
        // input_columns must carry every column in original order and case,
        // including standard columns (ID, TIME, DV, …) that are excluded from
        // covariate_names, and IOV columns.
        let csv = "ID,TIME,DV,EVID,AMT,OCC,WT\n\
                   1,0,.,1,100,1,70\n\
                   1,1,5.0,0,.,1,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        assert_eq!(
            pop.input_columns,
            vec!["ID", "TIME", "DV", "EVID", "AMT", "OCC", "WT"]
        );
        // Standard and IOV columns must not appear in covariate_names.
        assert_eq!(pop.covariate_names, vec!["WT"]);
    }

    #[test]
    fn tte_aware_readers_route_through_gaussian_path_with_empty_tte_cmts() {
        // `read_nonmem_csv_filtered_tte` / `_with_covariates_tte` (used by
        // api::read_population_for for [event_model] models) are always compiled
        // but only *called* on the TTE path, so they read as uncovered in the
        // FD-only build. With an empty tte_cmts set they delegate to the Gaussian
        // reader; drive them directly to cover the column-augmentation / union /
        // delegation lines. (The cfg(survival) row-routing inside the impl is
        // exercised by the survival job, not here.)
        let no_tte = std::collections::HashSet::new();
        let csv = "ID,TIME,DV,EVID,AMT,WT,STUDY,AGE\n\
                   1,0,.,1,100,70,1,30\n\
                   1,1,5.0,0,.,70,1,30\n\
                   2,0,.,1,100,80,2,40\n\
                   2,1,4.0,0,.,80,2,40\n";
        let f = write_csv(csv);

        // filtered_tte: explicit covariate list, augmented by a filter that
        // references an out-of-list column (STUDY) — exercises the augmentation
        // branch; the filter then drops STUDY==2 (subject 2).
        let cols: &[&str] = &["WT"];
        let filter = SelectionFilter::from_opts(&["STUDY == 2".to_string()], &[], &[]).unwrap();
        let pop = read_nonmem_csv_filtered_tte(f.path(), Some(cols), None, Some(&filter), &no_tte)
            .unwrap();
        assert_eq!(
            pop.subjects
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["1"],
            "STUDY==2 subject should be filtered out via the augmented column"
        );

        // with_covariates_tte: declared WT + an undeclared `extra` (STUDY) + a
        // filter referencing a *third* column (AGE) — exercises BOTH the
        // extra-columns dedup loop and the filter-referenced-column merge, and
        // *validates* the merge: AGE==40 can only drop subject 2 if AGE was
        // actually pulled into the read union, so the assertion fails if the
        // merge regresses.
        let decls = vec![CovariateDecl {
            name: "WT".to_string(),
            kind: CovariateKind::Continuous,
        }];
        let extra = ["STUDY".to_string()];
        let drop_age40 = SelectionFilter::from_opts(&["AGE == 40".to_string()], &[], &[]).unwrap();
        let (pop2, _table) = read_nonmem_csv_with_covariates_tte(
            f.path(),
            &decls,
            &extra,
            None,
            Some(&drop_age40),
            &no_tte,
        )
        .unwrap();
        // Subject 2 (AGE=40) is dropped via the merged AGE column; subject 1 remains.
        assert_eq!(
            pop2.subjects
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["1"],
            "AGE==40 must drop subject 2 — proving AGE was pulled into the read union"
        );
    }
}
