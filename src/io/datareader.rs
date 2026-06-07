use crate::io::filter_expr::{FilterClause, RowContext};
use crate::types::{
    CovariateDecl, CovariateRow, CovariateTable, DoseEvent, ExclusionSummary, Population, Subject,
};
use std::collections::HashMap;
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
    read_nonmem_csv_impl(path, covariate_columns, iov_column, None, None).map(|(pop, _)| pop)
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
    let (pop, table) =
        read_nonmem_csv_impl(path, Some(&union_refs), iov_column, Some(decls), None)?;
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
    read_nonmem_csv_impl(path, cols_ref.as_deref(), iov_column, None, Some(filter))
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
fn read_nonmem_csv_impl(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    table_decls: Option<&[CovariateDecl]>,
    filter: Option<&SelectionFilter>,
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
            // Mirror `parse_subject`'s EVID computation so the table's EVID
            // agrees with how each row was classified.
            let evid = evid_col
                .and_then(|c| fields.get(c))
                .map(|s| parse_evid(s))
                .unwrap_or(0);
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
    let mut population_warnings: Vec<String> = Vec::new();
    let n_records_total: usize = rows_by_id.iter().map(|(_, rows)| rows.len()).sum();
    let mut excl_summary = ExclusionSummary {
        n_records_total,
        ..Default::default()
    };
    for (id, rows) in &rows_by_id {
        let (subject, occ_failures, subj_excl, subj_warnings) = parse_subject(
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
            &cov_indices,
            filter,
        )?;
        total_occ_failures += occ_failures;
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
    cov_indices: &[(String, usize)],
    filter: Option<&SelectionFilter>,
) -> Result<(Subject, usize, SubjectExclusion, Vec<String>), String> {
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    let mut observations = Vec::new();
    let mut obs_cmts = Vec::new();
    let mut cens = Vec::new();
    let mut occasions: Vec<u32> = Vec::new();
    let mut dose_occasions: Vec<u32> = Vec::new();
    let mut occ_parse_failures: usize = 0;
    let mut excl_n_obs: usize = 0;
    let mut excl_n_dose: usize = 0;
    let mut excl_n_other: usize = 0;
    let mut excl_fired: Vec<String> = Vec::new();
    let mut parse_warnings: Vec<String> = Vec::new();
    let mut addl_missing_ii_warned = false;

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
        let evid = evid_col
            .and_then(|c| row.get(c))
            .map(|s| parse_evid(s))
            .unwrap_or(0);
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
                if evid == 1 || evid == 4 {
                    excl_n_dose += 1;
                } else if evid == 0 && mdv == 0 {
                    excl_n_obs += 1;
                } else {
                    excl_n_other += 1;
                }
                continue; // skip this row
            }
        }

        // EVID=3 (reset) and EVID=4 (reset + dose) both zero the compartment
        // state at this time. Record the reset before the dose arm runs so
        // EVID=4 captures both the reset and its dose.
        if evid == 3 || evid == 4 {
            reset_times.push(time);
        }

        if evid == 3 {
            // Pure system reset: no dose, no observation. Nothing else to do.
        } else if evid == 1 || evid == 4 {
            // Dose record
            let amt = amt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
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
            let ii = ii_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let ss = ss_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s.trim()) >= 0.5)
                .unwrap_or(false);

            doses.push(DoseEvent::new(time, amt, cmt, rate, ss, ii));
            if occ_col.is_some() {
                dose_occasions.push(occ);
            }
            if any_tv {
                dose_covariates.push(locf_state.clone());
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
                        doses.push(DoseEvent::new(
                            time + (k as f64) * ii,
                            amt,
                            cmt,
                            rate,
                            false, // expanded doses are never SS themselves
                            ii,
                        ));
                        if occ_col.is_some() {
                            dose_occasions.push(occ);
                        }
                        if any_tv {
                            dose_covariates.push(locf_state.clone());
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
            let cens_flag = cens_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(0);
            obs_times.push(time);
            observations.push(dv);
            obs_cmts.push(cmt);
            cens.push(if cens_flag > 0 { 1u8 } else { 0u8 });
            if occ_col.is_some() {
                occasions.push(occ);
            }
            if any_tv {
                obs_covariates.push(locf_state.clone());
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

    Ok((
        Subject {
            id: id.to_string(),
            doses: sorted_doses,
            obs_times,
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
            #[cfg(feature = "survival")]
            obs_records: vec![],
        },
        occ_parse_failures,
        SubjectExclusion {
            n_obs_excluded: excl_n_obs,
            n_dose_excluded: excl_n_dose,
            n_other_excluded: excl_n_other,
            fired: excl_fired,
        },
        parse_warnings,
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
        assert_eq!(subj.reset_times, vec![5.0]);
        assert!(subj.has_resets());
        // One dose (t=0), two observations (t=1, t=6) — the reset row is neither.
        assert_eq!(subj.doses.len(), 1);
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
}
