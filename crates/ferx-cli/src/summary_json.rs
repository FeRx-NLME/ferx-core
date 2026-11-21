//! `ferx summary --json` (#3074): the run summary as one JSON document.
//!
//! Every per-run value is copied out of [`ferx_core::FitResult::to_json_value`] —
//! the `{model}-fit.json` serializer — under the same key, so a field reads the
//! same here and in the fit JSON and the two cannot drift: `estimates`, `status`,
//! `ofv`, `covariance_status`, … are that writer's values, not a second mapping.
//! Non-finite floats are therefore `null`, as in the fit JSON.

use serde_json::{json, Map, Value};

/// Schema version of the `ferx summary --json` document. Same policy as
/// `FitResult::JSON_SCHEMA_VERSION`: bumped only on a breaking change (a renamed,
/// removed or re-nested key); additive keys do not bump it.
pub(crate) const SUMMARY_JSON_SCHEMA_VERSION: u32 = 1;

/// The fit-JSON keys a summary run carries, in the order `format_summary` prints
/// them. Each is a key of `FitResult::to_json_value()`, copied verbatim.
pub(crate) const RUN_KEYS: &[&str] = &[
    // Run status.
    "model_name",
    "method",
    "method_chain",
    "converged",
    "status",
    "final_gradient",
    "final_gradient_source",
    "n_iterations",
    "n_subjects",
    "n_obs",
    "n_parameters",
    // Objective function.
    "ofv",
    "ofv_data",
    "ofv_prior",
    "aic",
    "bic",
    "prior_summary",
    "saem_objective",
    "importance_sampling",
    // Parameter estimates (θ with SE / %RSE, Ω with CV% and correlations, κ, σ;
    // FIX flagged), in the fit YAML/JSON's name-keyed layout.
    "estimates",
    // Every θ, by position: `estimates.theta` leaves out the coefficients of a
    // compact θ block (20+ levels, summarized under `theta_blocks`) and
    // `[covariate_nn]` weights, as the text table does; these vectors carry them.
    "theta",
    "theta_names",
    "theta_fixed",
    "se_theta",
    // Coordinates the covariance step held on a bound (no SE although `se_theta`
    // may carry a number for them), and the θ blocks declared `profile`.
    "held_at_bound",
    "profiled_blocks",
    "error_model",
    // Diagnostics. Shrinkages are fractions, indexed like `eta_names` /
    // `kappa_names`.
    "covariance_status",
    "covariance_method",
    "cov_condition_number",
    "eta_names",
    "kappa_names",
    "shrinkage_eta",
    "shrinkage_kappa",
    "shrinkage_kappa_by_occ",
    "shrinkage_eps",
    // Run info.
    "wall_time_secs",
    "ferx_version",
    "warnings_structured",
];

/// One bundle named on the command line: its path, its label (the file stem, as
/// the comparison table's column header) and the loaded fit or the load error.
pub(crate) struct SummaryEntry<'a> {
    pub path: &'a str,
    pub label: String,
    pub fit: Result<&'a ferx_core::FitResult, String>,
}

/// The per-run object: `label` and `path`, then either the [`RUN_KEYS`] of the
/// fit JSON or, for a bundle that failed to load, `error`.
fn run_object(entry: &SummaryEntry<'_>) -> Map<String, Value> {
    let mut run = Map::new();
    run.insert("label".into(), json!(entry.label));
    run.insert("path".into(), json!(entry.path));
    match &entry.fit {
        Ok(fit) => {
            let full = fit.to_json_value();
            for key in RUN_KEYS {
                // A key the serializer skips when empty (`prior_summary`) is `null`
                // here, so every run object carries the same keys.
                run.insert((*key).into(), full.get(*key).cloned().unwrap_or(Value::Null));
            }
            run.insert("theta_blocks".into(), theta_blocks(fit));
        }
        Err(e) => {
            run.insert("error".into(), json!(e));
        }
    }
    run
}

/// A finite float as JSON, else `null` (the fit JSON's convention).
fn num(v: f64) -> Value {
    serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
}

/// The compact θ blocks (`ferx_core::io::output::compact_theta_blocks`, 20+
/// coefficients) the text summary reports under `--- THETA BLOCKS ---` instead of
/// row by row, keyed by block name: the same free / FIX counts, `profiled` flag
/// and min / median / max (`block_summary`, over every coefficient, FIX included),
/// plus `theta_start` / `theta_end` — the half-open index range of the block's
/// coefficients in `theta` / `theta_names` / `theta_fixed` / `se_theta`, so every
/// coefficient is reachable. `{}` when the fit has none.
fn theta_blocks(fit: &ferx_core::FitResult) -> Value {
    let mut blocks = Map::new();
    for (block, range) in ferx_core::io::output::compact_theta_blocks(&fit.theta_names) {
        let n_fixed =
            range.clone().filter(|&i| fit.theta_fixed.get(i).copied().unwrap_or(false)).count();
        let (min, median, max) = ferx_core::io::output::block_summary(&fit.theta[range.clone()]);
        let profiled = fit.profiled_blocks.contains(&block);
        blocks.insert(
            block,
            json!({
                "n_free_coefficients": range.len() - n_fixed,
                "n_fixed_coefficients": n_fixed,
                "profiled": profiled,
                "min": num(min),
                "median": num(median),
                "max": num(max),
                "theta_start": range.start,
                "theta_end": range.end,
            }),
        );
    }
    Value::Object(blocks)
}

/// The whole document: `schema_version`, `fit_schema_version` (the fit JSON
/// layout the copied fields follow), and `runs`, one object per bundle in
/// command-line order. With two or more bundles each loaded run also carries
/// `delta_ofv` against the first run (`reference_run` = index 0), as the
/// comparison table's ΔOFV row — `null` when either OFV is non-finite or the
/// first bundle failed to load.
pub(crate) fn summary_json(entries: &[SummaryEntry<'_>]) -> Value {
    let reference_ofv = entries.first().and_then(|e| e.fit.as_ref().ok()).map(|f| f.ofv);
    let compare = entries.len() > 1;
    let runs: Vec<Value> = entries
        .iter()
        .map(|entry| {
            let mut run = run_object(entry);
            if compare {
                if let Ok(fit) = &entry.fit {
                    let delta = reference_ofv
                        .map(|r| fit.ofv - r)
                        .and_then(serde_json::Number::from_f64)
                        .map_or(Value::Null, Value::Number);
                    run.insert("delta_ofv".into(), delta);
                }
            }
            Value::Object(run)
        })
        .collect();
    let mut doc = Map::new();
    doc.insert("schema_version".into(), json!(SUMMARY_JSON_SCHEMA_VERSION));
    doc.insert("fit_schema_version".into(), json!(ferx_core::FitResult::JSON_SCHEMA_VERSION));
    if compare {
        doc.insert("reference_run".into(), json!(0));
    }
    doc.insert("runs".into(), Value::Array(runs));
    Value::Object(doc)
}

#[cfg(test)]
mod tests {
    use super::{summary_json, SummaryEntry, RUN_KEYS, SUMMARY_JSON_SCHEMA_VERSION};
    use serde_json::Value;

    const WARFARIN_MODEL: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/warfarin.ferx");

    /// A simulated warfarin fit (no optimization loop), with the diagnostics a
    /// simulation leaves empty filled in so every line of the text summary is
    /// printed: standard errors, a covariance estimator and condition number,
    /// η / ε shrinkage and a warning.
    fn populated_fit() -> ferx_core::FitResult {
        let (mut fit, _) = ferx_core::run_model_simulate(WARFARIN_MODEL).expect("simulate");
        let n_theta = fit.theta.len();
        fit.se_theta = Some(fit.theta.iter().map(|t| 0.1 * t.abs()).collect());
        assert_eq!(fit.se_theta.as_ref().unwrap().len(), n_theta);
        *fit.theta_fixed.last_mut().expect("a theta") = true;
        fit.covariance_status = ferx_core::CovarianceStatus::Computed;
        fit.covariance_method = Some(ferx_core::CovarianceMethod::Hessian);
        fit.cov_condition_number = Some(123.5);
        fit.shrinkage_eta = vec![0.125; fit.eta_names.len()];
        fit.shrinkage_eps = 0.0625;
        let w = "covariance step: a test warning";
        fit.warnings.push(w.to_string());
        fit.warnings_structured.push(ferx_core::types::classify_warning(w));
        fit
    }

    fn one(fit: &ferx_core::FitResult) -> Value {
        summary_json(&[SummaryEntry { path: "a/run.fitrx", label: "run".into(), fit: Ok(fit) }])
    }

    #[test]
    fn single_run_carries_schema_versions_and_every_run_key() {
        let fit = populated_fit();
        let doc = one(&fit);
        assert_eq!(doc["schema_version"], SUMMARY_JSON_SCHEMA_VERSION);
        assert_eq!(doc["fit_schema_version"], ferx_core::FitResult::JSON_SCHEMA_VERSION);
        // A single run is not a comparison: no reference, no ΔOFV.
        assert!(doc.get("reference_run").is_none());
        let runs = doc["runs"].as_array().expect("runs array");
        assert_eq!(runs.len(), 1);
        let run = runs[0].as_object().unwrap();
        assert_eq!(run["label"], "run");
        assert_eq!(run["path"], "a/run.fitrx");
        assert!(run.get("delta_ofv").is_none());
        assert!(run.get("error").is_none());
        for key in RUN_KEYS {
            assert!(run.contains_key(*key), "run object lacks `{key}`");
        }
        // Copied verbatim from the fit JSON serializer, not re-mapped.
        let full = fit.to_json_value();
        for key in RUN_KEYS {
            assert_eq!(
                run[*key],
                full.get(*key).cloned().unwrap_or(Value::Null),
                "`{key}` differs from FitResult::to_json_value()"
            );
        }
    }

    /// Every field the text summary (`format_summary`) prints has a JSON home: each
    /// row pairs a marker of a text line with the JSON key carrying its value, and
    /// both must be present. A key dropped from `RUN_KEYS` fails its row here.
    #[test]
    fn every_text_summary_field_is_in_the_json() {
        let fit = populated_fit();
        let text = ferx_core::io::output::format_summary(&fit);
        let doc = one(&fit);
        let run = &doc["runs"][0];
        let pairs: &[(&str, &str)] = &[
            ("ferx run summary — ", "model_name"),
            ("Method:", "method"),
            ("Converged:", "converged"),
            ("Status:", "status"),
            ("Iterations:", "n_iterations"),
            ("Subjects:", "n_subjects"),
            ("Observations:", "n_obs"),
            ("Parameters:", "n_parameters"),
            ("OFV", "ofv"),
            ("AIC:", "aic"),
            ("BIC:", "bic"),
            ("--- THETA ---", "estimates"),
            ("--- SIGMA (", "error_model"),
            ("Covariance:", "covariance_status"),
            ("covariance_method = r", "covariance_method"),
            ("Condition number:", "cov_condition_number"),
            ("Eta shrinkage:", "shrinkage_eta"),
            ("EPS shrinkage:", "shrinkage_eps"),
            ("Wall time:", "wall_time_secs"),
            ("ferx v", "ferx_version"),
            ("Warnings:", "warnings_structured"),
        ];
        for (marker, key) in pairs {
            assert!(text.contains(marker), "text summary lacks `{marker}`:\n{text}");
            assert!(!run[*key].is_null(), "JSON `{key}` (text `{marker}`) is null or absent");
        }
        assert_eq!(run["ofv"].as_f64(), Some(fit.ofv));
        assert_eq!(run["cov_condition_number"].as_f64(), Some(123.5));
        assert_eq!(run["shrinkage_eps"].as_f64(), Some(0.0625));
        let warnings = run["warnings_structured"].as_array().expect("warnings array");
        assert_eq!(warnings.len(), fit.warnings.len());
        let last = warnings.last().unwrap();
        assert_eq!(last["message"], "covariance step: a test warning");
        assert!(last["severity"].is_string() && last["category"].is_string(), "{last}");
        // Each parameter row of the text has a name-keyed estimate.
        let est = &run["estimates"];
        for (names, section) in
            [(&fit.theta_names, "theta"), (&fit.eta_names, "omega"), (&fit.sigma_names, "sigma")]
        {
            assert!(!names.is_empty(), "warfarin has {section} parameters");
            for name in names {
                assert!(text.contains(name.as_str()), "text lacks {section} `{name}`");
                assert!(
                    est[section][name].is_object(),
                    "estimates.{section}.{name} missing: {est}"
                );
            }
        }
        // θ with SE / %RSE and the FIX flag, as the fit JSON writes them.
        let first = &est["theta"][&fit.theta_names[0]];
        assert_eq!(first["se"].as_f64(), Some(fit.se_theta.as_ref().unwrap()[0]));
        assert!(first["rse_pct"].is_number(), "{first}");
        // The text marks a FIX parameter in its label; the JSON flags it.
        let last_name = fit.theta_names.last().unwrap();
        assert!(text.contains(&format!("{last_name} [FIX]")), "{text}");
        assert_eq!(est["theta"][last_name]["fixed"], true);
    }

    /// A compact θ block (20+ levels) is summarized, not listed, by the text and by
    /// `estimates.theta`; the JSON must still carry its summary row and every
    /// coefficient (review of #3074).
    #[test]
    fn compact_theta_blocks_are_summarized_and_every_coefficient_is_reachable() {
        let mut fit = populated_fit();
        let n_scalar = fit.theta.len();
        let n_block = ferx_core::io::output::THETA_BLOCK_COMPACT_MIN + 1;
        for k in 0..n_block {
            fit.theta_names.push(format!("PBO[{}]", k + 1));
            fit.theta.push(0.5 + k as f64);
            fit.theta_fixed.push(k == 0);
            fit.se_theta.as_mut().unwrap().push(0.01);
        }
        fit.profiled_blocks = vec!["PBO".to_string()];
        let text = ferx_core::io::output::format_summary(&fit);
        assert!(text.contains("--- THETA BLOCKS ---") && text.contains("PBO"), "{text}");
        let doc = one(&fit);
        let run = &doc["runs"][0];
        // Not row by row in `estimates.theta`, as in the text table …
        assert!(run["estimates"]["theta"].get("PBO[1]").is_none());
        // … but the block's summary row, with the text's counts and statistics.
        let b = &run["theta_blocks"]["PBO"];
        assert_eq!(b["n_free_coefficients"], n_block - 1, "{b}");
        assert_eq!(b["n_fixed_coefficients"], 1);
        assert_eq!(b["profiled"], true);
        assert!(text.contains(&format!("{} free, 1 FIX coefficients", n_block - 1)), "{text}");
        assert!(text.contains("profiled (not searched)"), "{text}");
        assert_eq!(b["min"].as_f64(), Some(0.5));
        assert_eq!(b["median"].as_f64(), Some(0.5 + ((n_block - 1) / 2) as f64));
        assert_eq!(b["max"].as_f64(), Some(0.5 + (n_block - 1) as f64));
        // Every coefficient, through the block's index range into the θ vectors.
        let (start, end) = (b["theta_start"].as_u64().unwrap(), b["theta_end"].as_u64().unwrap());
        assert_eq!((start as usize, end as usize), (n_scalar, n_scalar + n_block));
        let names = run["theta_names"].as_array().unwrap();
        let theta = run["theta"].as_array().unwrap();
        for (k, i) in (start as usize..end as usize).enumerate() {
            assert_eq!(names[i], format!("PBO[{}]", k + 1));
            assert_eq!(theta[i].as_f64(), Some(0.5 + k as f64));
        }
        assert_eq!(run["theta_fixed"][n_scalar], true);
        assert_eq!(run["se_theta"][n_scalar + 1].as_f64(), Some(0.01));
        assert_eq!(run["profiled_blocks"][0], "PBO");
    }

    #[test]
    fn a_fit_without_compact_blocks_has_an_empty_theta_blocks_object() {
        let fit = populated_fit();
        let run = &one(&fit)["runs"][0];
        assert_eq!(run["theta_blocks"], serde_json::json!({}));
        assert_eq!(run["theta"].as_array().unwrap().len(), fit.theta.len());
    }

    #[test]
    fn several_runs_carry_delta_ofv_against_the_first() {
        let a = populated_fit();
        let mut b = a.clone();
        b.ofv = a.ofv - 12.5;
        let mut c = a.clone();
        c.ofv = f64::NAN;
        let doc = summary_json(&[
            SummaryEntry { path: "a.fitrx", label: "a".into(), fit: Ok(&a) },
            SummaryEntry { path: "b.fitrx", label: "b".into(), fit: Ok(&b) },
            SummaryEntry { path: "c.fitrx", label: "c".into(), fit: Ok(&c) },
            SummaryEntry { path: "d.fitrx", label: "d".into(), fit: Err("boom".into()) },
        ]);
        assert_eq!(doc["reference_run"], 0);
        let runs = doc["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0]["delta_ofv"].as_f64(), Some(0.0));
        assert_eq!(runs[1]["delta_ofv"].as_f64(), Some(-12.5));
        // Non-finite → null, as everywhere in the fit JSON.
        assert!(runs[2]["delta_ofv"].is_null());
        assert!(runs[2]["ofv"].is_null());
        // A failed load is an error entry, with no fit fields and no ΔOFV.
        let d = runs[3].as_object().unwrap();
        assert_eq!(d["error"], "boom");
        assert_eq!(d["label"], "d");
        assert!(!d.contains_key("delta_ofv") && !d.contains_key("ofv"));
    }

    #[test]
    fn a_failed_reference_run_leaves_delta_ofv_null() {
        let a = populated_fit();
        let doc = summary_json(&[
            SummaryEntry { path: "x.fitrx", label: "x".into(), fit: Err("gone".into()) },
            SummaryEntry { path: "a.fitrx", label: "a".into(), fit: Ok(&a) },
        ]);
        assert!(doc["runs"][1]["delta_ofv"].is_null());
        assert!(doc["runs"][1]["ofv"].is_number());
    }
}
