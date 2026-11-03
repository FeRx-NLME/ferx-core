//! Guard for the golden fit set (#2635): `tools/golden-fits/models.tsv`,
//! `tools/golden-fits/golden.py` and `.github/workflows/golden-fits.yml`.
//!
//! The nightly job fits ~20 reference models and diffs them against the last release.
//! What can go wrong without anyone noticing is the *report*: a diff that cannot see a
//! perturbation still prints a tidy table. So the verdicts are pinned here on synthetic
//! results, each failure verdict paired with a twin that differs in the one property
//! under test (perturbed vs not, inside vs outside the band, before vs after the date),
//! so a report that always says "=" or always fails cannot satisfy both.
//!
//! No fit runs: `run` is exercised against a fake `ferx` that writes a canned fit JSON.
//! Not feature-gated — Tier 2, runs in the base `--features ci` job.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script() -> PathBuf {
    repo_root().join("tools/golden-fits/golden.py")
}

fn scratch(slot: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("golden-{slot}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn golden(args: &[&str]) -> Output {
    Command::new("python3")
        .arg("-I")
        .arg(script())
        .args(args)
        .output()
        .expect("python3 runs tools/golden-fits/golden.py")
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// The `FAIL_ON_BAND_FROM = "YYYY-MM-DD"` line of golden.py.
fn fail_date() -> String {
    let src = std::fs::read_to_string(script()).unwrap();
    let line = src
        .lines()
        .find(|l| l.starts_with("FAIL_ON_BAND_FROM = "))
        .expect("golden.py defines FAIL_ON_BAND_FROM at module level");
    line.split('"').nth(1).unwrap().to_string()
}

/// The day before `date` (YYYY-MM-DD), without a date crate: the tests only ever need
/// one step back, and FAIL_ON_BAND_FROM is pinned below to a day-of-month > 1.
fn day_before(date: &str) -> String {
    let day: u32 = date[8..].parse().unwrap();
    assert!(
        day > 1,
        "pick a FAIL_ON_BAND_FROM that is not the 1st: {date}"
    );
    format!("{}{:02}", &date[..8], day - 1)
}

// -------------------------------------------------------------------------------------
// Synthetic results.json
// -------------------------------------------------------------------------------------

/// One model record: OFV, iterations, and `(key, est, se)` parameters. `ofv` is spliced
/// in verbatim so a test can write `NaN` (what `json.dumps(allow_nan=True)` emits).
fn model_rec(ofv: &str, iters: u32, params: &[(&str, f64, f64)]) -> String {
    let ps: Vec<String> = params
        .iter()
        .map(|(k, e, s)| format!(r#""{k}": {{"est": {e:e}, "se": {s:e}}}"#))
        .collect();
    format!(
        r#"{{"status": "ok", "family": "FOCE, test", "model_sha256": "m", "data_sha256": "d",
            "ofv": {ofv}, "converged": true, "n_iterations": {iters},
            "covariance_status": "Computed", "wall_s": 1.0, "params": {{{}}}}}"#,
        ps.join(", ")
    )
}

fn results(label: &str, models: &[(&str, String)]) -> String {
    let ms: Vec<String> = models
        .iter()
        .map(|(id, r)| format!(r#""{id}": {r}"#))
        .collect();
    format!(
        r#"{{"schema": 1, "label": "{label}", "platform": "Linux/x86_64", "models": {{{}}}}}"#,
        ms.join(", ")
    )
}

const BASE_PARAMS: [(&str, f64, f64); 3] = [
    ("theta:CL", 0.134, 0.0071),
    ("theta:V", 8.11, 0.21),
    ("omega:ETA_CL", 0.071, 0.019),
];

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p.to_string_lossy().into_owned()
}

/// `report` over a base with `warfarin` at BASE_PARAMS and a head with `head_params`.
fn report(
    slot: &str,
    head_ofv: &str,
    head_params: &[(&str, f64, f64)],
    bands: Option<&str>,
    today: &str,
) -> (Output, String) {
    report_recs(
        slot,
        model_rec("1234.5", 20, &BASE_PARAMS),
        model_rec(head_ofv, 20, head_params),
        bands.map(|b| (b, None)),
        today,
    )
}

/// `report` over arbitrary `warfarin` records. `bands` is the bands.json body and,
/// optionally, `(head results, reason)` to pin with `golden.py accept` first — the
/// real path a contributor takes, so the test exercises what `accept` writes.
fn report_recs(
    slot: &str,
    base_rec: String,
    head_rec: String,
    bands: Option<(&str, Option<(String, &str)>)>,
    today: &str,
) -> (Output, String) {
    let dir = scratch(slot);
    let base = write(
        &dir,
        "base.json",
        &results("v0.4.0", &[("warfarin", base_rec)]),
    );
    let head = write(
        &dir,
        "head.json",
        &results("main@abc", &[("warfarin", head_rec)]),
    );
    let bands_path = match bands {
        Some((b, accept)) => {
            let path = write(&dir, "bands.json", b);
            if let Some((accepted_results, reason)) = accept {
                let acc = write(&dir, "accepted.json", &accepted_results);
                let out = golden(&[
                    "accept", "warfarin", "--head", &acc, "--reason", reason, "--bands", &path,
                ]);
                assert!(out.status.success(), "accept: {}", text(&out));
            }
            path
        }
        None => dir.join("no-bands.json").to_string_lossy().into_owned(),
    };
    let summary = dir.join("summary.md");
    let out = golden(&[
        "report",
        "--head",
        &head,
        "--base",
        &base,
        "--bands",
        &bands_path,
        "--today",
        today,
        "--summary",
        summary.to_str().unwrap(),
    ]);
    let md = std::fs::read_to_string(&summary).unwrap_or_default();
    (out, md)
}

fn warfarin_row(md: &str) -> &str {
    md.lines()
        .find(|l| l.contains("| `warfarin` |"))
        .unwrap_or_else(|| panic!("no warfarin row in the report:\n{md}"))
}

fn perturbed_cl() -> Vec<(&'static str, f64, f64)> {
    let mut p = BASE_PARAMS.to_vec();
    p[0].1 *= 1.01;
    p
}

const BEFORE: &str = "2026-11-10";

// -------------------------------------------------------------------------------------
// The report sees a perturbation — and only where it is
// -------------------------------------------------------------------------------------

/// A 1% change on one θ must show as a change on THAT θ; the unperturbed twin must show
/// `=` with no "Changed models" section. Either half alone is satisfied by a report that
/// prints a constant.
#[test]
fn report_flags_a_perturbed_theta_and_not_its_unperturbed_twin() {
    let (out, md) = report("perturbed", "1234.5", &perturbed_cl(), None, BEFORE);
    assert!(
        out.status.success(),
        "report-only must exit 0: {}",
        text(&out)
    );
    let row = warfarin_row(&md);
    assert!(row.starts_with("| ≠ |"), "perturbed row not flagged: {row}");
    assert!(
        row.contains("`theta:CL`"),
        "row must name the parameter that moved: {row}"
    );
    assert!(
        row.contains("1.00e-02"),
        "a 1% bump, relative to the release: {row}"
    );
    assert!(md.contains("### Changed models") && md.contains("| theta:CL | 0.134 | 0.13534 |"));
    assert!(
        !md.contains("theta:V |"),
        "an unchanged θ must not be listed:\n{md}"
    );

    let (out, md) = report("unperturbed", "1234.5", &BASE_PARAMS, None, BEFORE);
    assert!(out.status.success());
    let row = warfarin_row(&md);
    assert!(row.starts_with("| = |"), "unchanged row flagged: {row}");
    assert!(!md.contains("### Changed models"), "{md}");
}

/// A change in an SE alone lands in the SE column, not the estimate column. Pins that
/// the two are compared separately — a report diffing only estimates would miss it.
#[test]
fn an_se_only_change_is_reported_in_the_se_column() {
    let mut p = BASE_PARAMS.to_vec();
    p[1].2 *= 1.5;
    let (_, md) = report("se-only", "1234.5", &p, None, BEFORE);
    let row = warfarin_row(&md);
    let cols: Vec<&str> = row.split('|').map(str::trim).collect();
    // | glyph | model | family | OFV | ΔOFV | Δ est | Δ SE | ...
    assert_eq!(cols[1], "≠", "{row}");
    assert_eq!(cols[6], "=", "estimates did not move: {row}");
    assert!(
        cols[7].contains("`theta:V`"),
        "SE change must name theta:V: {row}"
    );
}

/// An OFV that went NaN is a failure glyph and printed as NaN — never folded into "="
/// by a `max` that drops it (AGENTS.md, `f64::max` discards NaN).
#[test]
fn a_nan_ofv_is_reported_not_swallowed() {
    let (_, md) = report("nan", "NaN", &BASE_PARAMS, None, BEFORE);
    let row = warfarin_row(&md);
    assert!(row.starts_with("| ❌ |"), "{row}");
    assert!(row.contains("**NaN**"), "{row}");

    // NaN on one parameter estimate, finite elsewhere: the worst-change column must
    // name the NaN, not the largest finite change.
    let mut p = perturbed_cl();
    p[1].1 = f64::NAN;
    let dir_md = {
        let dir = scratch("nan-param");
        let base = write(
            &dir,
            "base.json",
            &results("v", &[("warfarin", model_rec("1.0", 20, &BASE_PARAMS))]),
        );
        // `{:e}` of f64::NAN is `NaN`, the token `json.dumps(allow_nan=True)` writes.
        let head_body = results("h", &[("warfarin", model_rec("1.0", 20, &p))]);
        assert!(head_body.contains(r#""est": NaN"#), "{head_body}");
        let head = write(&dir, "head.json", &head_body);
        let out = golden(&[
            "report", "--head", &head, "--base", &base, "--today", BEFORE,
        ]);
        assert!(out.status.success(), "{}", text(&out));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let row = warfarin_row(&dir_md);
    assert!(
        row.contains("**NaN** `theta:V`"),
        "NaN must outrank the 1% θ change: {row}"
    );
}

// -------------------------------------------------------------------------------------
// Report-only until the date, then fail-on-band
// -------------------------------------------------------------------------------------

/// The day before FAIL_ON_BAND_FROM a perturbed head exits 0; on the day, with no
/// bands.json committed, it fails and says which file is missing.
#[test]
fn report_only_ends_on_the_date_and_a_missing_bands_file_fails() {
    let date = fail_date();
    let (out, md) = report(
        "before",
        "1234.5",
        &perturbed_cl(),
        None,
        &day_before(&date),
    );
    assert!(out.status.success(), "{}", text(&out));
    assert!(
        md.contains(&format!("**Report-only** until {date} (1 days)")),
        "{md}"
    );

    let (out, md) = report("on-date", "1234.5", &BASE_PARAMS, None, &date);
    assert!(
        !out.status.success(),
        "no bands on the date must fail:\n{md}"
    );
    assert!(
        md.contains("tools/golden-fits/bands.json is not committed"),
        "{md}"
    );
    assert!(
        text(&out).contains("::error title=golden fit set::"),
        "{}",
        text(&out)
    );
}

fn bands(cl_band: f64) -> String {
    format!(
        r#"{{"n_nightlies": 30, "accepted": {{}}, "models": {{"warfarin": {{
            "ofv": 0.001, "n_iterations": 2,
            "est": {{"theta:CL": {cl_band:e}, "theta:V": 1e-6, "omega:ETA_CL": 1e-6}},
            "se": {{"theta:CL": 1e-6, "theta:V": 1e-6, "omega:ETA_CL": 1e-6}}}}}}}}"#
    )
}

/// In fail-on-band mode the same 1% θ bump passes inside a 2% band, fails outside a
/// 0.5% band naming the model and quantity, and passes again when accepted.
#[test]
fn fail_on_band_straddles_the_band_and_honours_accepted() {
    let date = fail_date();
    let (out, md) = report(
        "in-band",
        "1234.5",
        &perturbed_cl(),
        Some(&bands(0.02)),
        &date,
    );
    assert!(out.status.success(), "1% inside a 2% band must pass:\n{md}");
    assert!(warfarin_row(&md).starts_with("| ≠ |"), "{md}");

    let (out, md) = report(
        "out-band",
        "1234.5",
        &perturbed_cl(),
        Some(&bands(0.005)),
        &date,
    );
    assert!(
        !out.status.success(),
        "1% outside a 0.5% band must fail:\n{md}"
    );
    assert!(warfarin_row(&md).starts_with("| 🔺 |"), "{md}");
    assert!(
        md.contains("- warfarin: est(theta:CL): 0.01 > band 0.005"),
        "{md}"
    );
    assert!(
        !md.contains("est(theta:V)"),
        "only the quantity that moved fails:\n{md}"
    );

    // Accepted at exactly this state: outside the band vs the release, yet green.
    let state = || {
        results(
            "pr",
            &[("warfarin", model_rec("1234.5", 20, &perturbed_cl()))],
        )
    };
    let (out, md) = report_recs(
        "accepted",
        model_rec("1234.5", 20, &BASE_PARAMS),
        model_rec("1234.5", 20, &perturbed_cl()),
        Some((&bands(0.005), Some((state(), "#9999 deliberate")))),
        &date,
    );
    assert!(
        out.status.success(),
        "an accepted change must not fail:\n{md}"
    );
    assert!(warfarin_row(&md).starts_with("| ✔︎ |"), "{md}");
    assert!(md.contains("#9999 deliberate"), "{md}");

    // Accepted, then drifts again on an UNRELATED quantity: the acceptance pins a state,
    // it does not exempt the model, so this must fail and name the new drift.
    let mut drifted = perturbed_cl();
    drifted[1].1 *= 1.01;
    let (out, md) = report_recs(
        "accepted-drift",
        model_rec("1234.5", 20, &BASE_PARAMS),
        model_rec("1234.5", 20, &drifted),
        Some((&bands(0.005), Some((state(), "#9999 deliberate")))),
        &date,
    );
    assert!(
        !out.status.success(),
        "drift after acceptance must fail:\n{md}"
    );
    assert!(warfarin_row(&md).starts_with("| 🔺 |"), "{md}");
    assert!(
        md.contains("- warfarin: drifted from its accepted state: est(theta:V): 0.01 > band 1e-06"),
        "{md}"
    );
    assert!(
        !md.contains("accepted state: est(theta:CL)"),
        "the accepted θ_CL move is not a drift from the accepted state:\n{md}"
    );

    // An entry with no pinned state (the old, exempt-everything shape) fails.
    let b = bands(0.005).replace(
        r#""accepted": {}"#,
        r##""accepted": {"warfarin": {"reason": "#9999 deliberate"}}"##,
    );
    let (out, md) = report(
        "accepted-no-state",
        "1234.5",
        &perturbed_cl(),
        Some(&b),
        &date,
    );
    assert!(!out.status.success(), "{md}");
    assert!(md.contains("accepted entry pins no state"), "{md}");
}

/// In fail-on-band mode a NaN estimate fails even inside a generous band: `NaN > tol`
/// is false, so the check has to ask for NaN explicitly.
#[test]
fn fail_on_band_fails_a_nan_estimate() {
    let date = fail_date();
    let mut p = BASE_PARAMS.to_vec();
    p[1].1 = f64::NAN;
    let (out, md) = report("nan-band", "1234.5", &p, Some(&bands(0.02)), &date);
    assert!(
        !out.status.success(),
        "a NaN θ must fail in band mode:\n{md}"
    );
    assert!(md.contains("est(theta:V): nan > band"), "{md}");
}

/// A fit that fails at head but succeeded at the release fails the band mode; its twin,
/// failing at both, is a known state and does not.
#[test]
fn a_fit_that_broke_since_the_release_fails_the_band_mode() {
    let date = fail_date();
    let failed = r#"{"status": "failed", "family": "FOCE, test", "error": "exit 1: boom"}"#;
    let ok = model_rec("1234.5", 20, &BASE_PARAMS);
    for (slot, base_rec, should_fail) in [("broke", ok.as_str(), true), ("still", failed, false)] {
        let dir = scratch(slot);
        let base = write(
            &dir,
            "base.json",
            &results("v", &[("warfarin", base_rec.to_string())]),
        );
        let head = write(
            &dir,
            "head.json",
            &results("h", &[("warfarin", failed.to_string())]),
        );
        let bands_path = write(&dir, "bands.json", &bands(0.02));
        let out = golden(&[
            "report",
            "--head",
            &head,
            "--base",
            &base,
            "--bands",
            &bands_path,
            "--today",
            &date,
        ]);
        let md = text(&out);
        assert!(warfarin_row(&md).starts_with("| ❌ |"), "{md}");
        assert_eq!(!out.status.success(), should_fail, "{slot}:\n{md}");
        assert_eq!(
            md.contains("warfarin: fit failed at head, ok at the release"),
            should_fail,
            "{slot}:\n{md}"
        );
    }
}

/// A NaN band — what `bands` writes for a quantity that went NaN in its window — is
/// no band: `v > NaN` is false for every v, so it must not let a finite drift pass.
#[test]
fn a_non_finite_band_is_no_band() {
    let date = fail_date();
    let b = bands(0.02).replace(r#""theta:CL": 2e-2"#, r#""theta:CL": NaN"#);
    assert!(b.contains(r#""theta:CL": NaN"#), "{b}");
    let (out, md) = report("nan-tol", "1234.5", &perturbed_cl(), Some(&b), &date);
    assert!(
        !out.status.success(),
        "a finite drift against a NaN band must fail:\n{md}"
    );
    assert!(md.contains("est(theta:CL): band is not finite"), "{md}");
    // Twin: the same drift against the finite 2% band passes (straddle).
    let (out, _) = report(
        "nan-tol-twin",
        "1234.5",
        &perturbed_cl(),
        Some(&bands(0.02)),
        &date,
    );
    assert!(out.status.success());
}

/// NaN at the release AND at head is still NaN, not "unchanged": a quantity that has
/// been NaN since the release must not read as a stable pass.
#[test]
fn nan_on_both_sides_is_a_change_and_fails_the_band_mode() {
    let date = fail_date();
    let mut p = BASE_PARAMS.to_vec();
    p[1].1 = f64::NAN;
    let (out, md) = report_recs(
        "nan-both",
        model_rec("NaN", 20, &p),
        model_rec("NaN", 20, &p),
        Some((&bands(0.02), None)),
        &date,
    );
    assert!(!out.status.success(), "{md}");
    assert!(md.contains("OFV: nan > band"), "{md}");
    assert!(md.contains("est(theta:V): nan > band"), "{md}");
    let (out, md) = report_recs(
        "nan-both-report-only",
        model_rec("1.0", 20, &p),
        model_rec("1.0", 20, &p),
        None,
        BEFORE,
    );
    assert!(out.status.success());
    let row = warfarin_row(&md);
    assert!(
        row.starts_with("| ≠ |") && row.contains("**NaN** `theta:V`"),
        "{row}"
    );
}

/// A parameter with no measured band fails rather than being skipped: a new θ must
/// force a re-measure, not slip through.
#[test]
fn a_quantity_without_a_band_is_out_of_band() {
    let date = fail_date();
    let b = bands(0.02).replace(r#""theta:V": 1e-6, "omega"#, r#""omega"#);
    let (out, md) = report("unbanded", "1234.5", &BASE_PARAMS, Some(&b), &date);
    assert!(!out.status.success(), "{md}");
    assert!(md.contains("est(theta:V): no measured band"), "{md}");
}

// -------------------------------------------------------------------------------------
// bands
// -------------------------------------------------------------------------------------

#[test]
fn bands_is_headroom_times_the_spread_and_refuses_a_short_window() {
    let dir = scratch("bands");
    let hist: Vec<String> = ["1234.5", "1234.7", "1234.4"]
        .iter()
        .enumerate()
        .map(|(k, ofv)| {
            let mut p = BASE_PARAMS.to_vec();
            p[0].1 = [0.134, 0.135, 0.134][k];
            write(
                &dir,
                &format!("n{k}.json"),
                &results(
                    &format!("n{k}"),
                    &[("warfarin", model_rec(ofv, 20 + k as u32, &p))],
                ),
            )
        })
        .collect();
    let out_path = dir.join("bands.json");
    let mut args = vec![
        "bands",
        "--min-nightlies",
        "3",
        "--out",
        out_path.to_str().unwrap(),
    ];
    args.extend(hist.iter().map(String::as_str));
    let out = golden(&args);
    assert!(out.status.success(), "{}", text(&out));
    let b: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let w = &b["models"]["warfarin"];
    // OFV median 1234.5, max deviation 0.2, headroom 2.
    assert!((w["ofv"].as_f64().unwrap() - 0.4).abs() < 1e-9, "{w}");
    // θ_CL median 0.134, max deviation 0.001 → relative 0.001/0.134, × 2.
    let want = 2.0 * 0.001 / 0.134;
    assert!(
        (w["est"]["theta:CL"].as_f64().unwrap() - want).abs() < 1e-12,
        "{w}"
    );
    // θ_V never moved: the floor, not 0. Same for every SE (constant in the fixture).
    assert_eq!(w["est"]["theta:V"].as_f64().unwrap(), 1e-6, "{w}");
    assert_eq!(w["se"]["theta:CL"].as_f64().unwrap(), 1e-6, "{w}");
    // Iterations 20, 21, 22: deviation 1 × 2 = 2 (also the floor), as an integer.
    assert_eq!(w["n_iterations"].as_u64().unwrap(), 2, "{w}");

    let mut short = vec!["bands", "--out", out_path.to_str().unwrap()];
    short.extend(hist.iter().map(String::as_str));
    let out = golden(&short);
    assert!(
        !out.status.success(),
        "3 nightlies against the default 30 must refuse"
    );
    assert!(
        text(&out).contains("3 nightlies given, 30 required"),
        "{}",
        text(&out)
    );
}

// -------------------------------------------------------------------------------------
// run: extraction from a fit JSON, against a fake ferx
// -------------------------------------------------------------------------------------

/// A block omega's off-diagonal is tracked, its SE read from the column-major packed
/// lower triangle the way `types::omega_se_at` reads it, and a `null` OFV (the JSON
/// writer's spelling of non-finite) comes back as NaN. A failing binary is a `failed`
/// row, not an aborted run.
#[cfg(unix)]
#[test]
fn run_extracts_block_omega_and_reports_a_failed_fit() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("run");
    std::fs::create_dir_all(dir.join("examples")).unwrap();
    std::fs::create_dir_all(dir.join("data")).unwrap();
    std::fs::write(dir.join("examples/ok.ferx"), "x").unwrap();
    std::fs::write(dir.join("examples/bad.ferx"), "x").unwrap();
    std::fs::write(dir.join("data/d.csv"), "x").unwrap();
    let fit = r#"{"ofv": null, "converged": true, "n_iterations": 7, "method": "Focei",
        "covariance_status": "Computed", "warnings": [],
        "theta": [1.5], "theta_names": ["CL"], "se_theta": [0.1],
        "eta_names": ["ETA_CL", "ETA_V", "ETA_KA"],
        "omega": {"rows": 3, "cols": 3,
                  "data": [0.1, 0.02, 0.05, 0.02, 0.3, 0.06, 0.05, 0.06, 0.7]},
        "se_omega": [0.01, 0.002, 0.005, 0.03, 0.006, 0.07],
        "sigma": [0.04], "sigma_names": ["PROP"], "se_sigma": [0.004],
        "omega_iov": null}"#;
    std::fs::write(dir.join("fit.json"), fit).unwrap();
    let fake = dir.join("ferx");
    std::fs::write(
        &fake,
        format!(
            // Refuses unless asked for JSON: `run` must request the file it reads.
            "#!/bin/sh\n\
             case \"$*\" in *'--output-format json'*) ;; *) echo 'no json' >&2; exit 9;; esac\n\
             case \"$1\" in *bad.ferx) echo boom >&2; exit 3;; esac\n\
             cp '{}' ./ok-fit.json\n",
            dir.join("fit.json").display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let manifest = write(
        &dir,
        "models.tsv",
        "# c\nid\tmodel\tdata\tfamily\nok\texamples/ok.ferx\tdata/d.csv\tA\n\
         bad\texamples/bad.ferx\tdata/d.csv\tB\ngone\texamples/gone.ferx\tdata/d.csv\tC\n",
    );
    let out_path = dir.join("results.json");
    let out = golden(&[
        "run",
        "--ferx",
        fake.to_str().unwrap(),
        "--root",
        dir.to_str().unwrap(),
        "--manifest",
        &manifest,
        "--out",
        out_path.to_str().unwrap(),
        "--label",
        "t",
    ]);
    assert!(out.status.success(), "{}", text(&out));
    let raw = std::fs::read_to_string(&out_path).unwrap();
    assert!(
        raw.contains(r#""ofv": NaN"#),
        "null OFV must read back as NaN:\n{raw}"
    );
    let r: serde_json::Value = serde_json::from_str(&raw.replace("NaN", "null")).unwrap();
    let ok = &r["models"]["ok"];
    assert_eq!(ok["status"], "ok");
    assert_eq!(ok["n_iterations"], 7);
    let p = &ok["params"];
    assert_eq!(p["theta:CL"]["se"], 0.1);
    assert_eq!(p["omega:ETA_CL"]["se"], 0.01);
    assert_eq!(p["omega:ETA_V,ETA_CL"]["est"], 0.02);
    // Column-major packing: (0,0) (1,0) (2,0) (1,1) (2,1) (2,2). A 3×3 block is the
    // smallest that tells it from row-major, which swaps (2,0) and (1,1).
    for (key, want) in [
        ("omega:ETA_V,ETA_CL", 0.002),
        ("omega:ETA_KA,ETA_CL", 0.005),
        ("omega:ETA_V", 0.03),
        ("omega:ETA_KA,ETA_V", 0.006),
        ("omega:ETA_KA", 0.07),
    ] {
        assert_eq!(p[key]["se"], want, "packed lower-tri SE of {key}");
    }
    assert_eq!(p["sigma:PROP"]["est"], 0.04);
    assert_eq!(r["models"]["bad"]["status"], "failed");
    assert!(r["models"]["bad"]["error"]
        .as_str()
        .unwrap()
        .contains("exit 3: boom"));
    assert_eq!(r["models"]["gone"]["status"], "missing");
}

// -------------------------------------------------------------------------------------
// The manifest, the workflow, the docs
// -------------------------------------------------------------------------------------

struct Row {
    id: String,
    model: String,
    data: String,
    family: String,
}

fn manifest() -> Vec<Row> {
    let src = std::fs::read_to_string(repo_root().join("tools/golden-fits/models.tsv")).unwrap();
    let mut lines = src
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    assert_eq!(lines.next(), Some("id\tmodel\tdata\tfamily"));
    lines
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            assert_eq!(c.len(), 4, "four tab-separated columns: {l:?}");
            Row {
                id: c[0].into(),
                model: c[1].into(),
                data: c[2].into(),
                family: c[3].into(),
            }
        })
        .collect()
}

/// About 20 rows, each a tracked `examples/` model with tracked data, with distinct ids
/// and distinct families (the set is one model per family, not twenty warfarins).
#[test]
fn manifest_names_tracked_example_models_one_per_family() {
    let rows = manifest();
    assert!(
        (18..=30).contains(&rows.len()),
        "the issue asks for about 20 models, got {}",
        rows.len()
    );
    let mut ids = std::collections::BTreeSet::new();
    let mut fams = std::collections::BTreeSet::new();
    for r in &rows {
        assert!(ids.insert(&r.id), "duplicate id {}", r.id);
        assert!(fams.insert(&r.family), "duplicate family {:?}", r.family);
        assert!(
            r.model.starts_with("examples/") && r.model.ends_with(".ferx"),
            "golden models come from examples/: {}",
            r.model
        );
        for path in [&r.model, &r.data] {
            let tracked = Command::new("git")
                .args(["ls-files", "--error-unmatch", path])
                .current_dir(repo_root())
                .output()
                .unwrap();
            assert!(
                tracked.status.success(),
                "{} ({}) is not a tracked file",
                path,
                r.id
            );
        }
    }
}

#[test]
fn workflow_is_a_linux_nightly_that_posts_the_table() {
    let wf =
        std::fs::read_to_string(repo_root().join(".github/workflows/golden-fits.yml")).unwrap();
    let runs_on: Vec<&str> = wf
        .lines()
        .filter(|l| l.trim_start().starts_with("runs-on:"))
        .collect();
    assert_eq!(
        runs_on,
        ["    runs-on: ubuntu-latest"],
        "Linux only (#1688)"
    );
    assert!(
        wf.contains("\n  schedule:\n    - cron: "),
        "must run nightly"
    );
    // The `on:` block alone — the header comment names the triggers it avoids.
    let on: Vec<&str> = wf
        .lines()
        .skip_while(|l| *l != "on:")
        .skip(1)
        .take_while(|l| l.is_empty() || l.starts_with(' '))
        .collect();
    let triggers: Vec<&str> = on
        .iter()
        .filter(|l| l.starts_with("  ") && !l.starts_with("   ") && l.trim_end().ends_with(':'))
        .map(|l| l.trim().trim_end_matches(':'))
        .collect();
    assert_eq!(
        triggers,
        ["schedule", "workflow_dispatch"],
        "not a per-PR job: {on:?}"
    );
    assert!(wf.contains("RAYON_NUM_THREADS: 1"));
    // Two `run`s — the release on its own tree, head on this one — then the report into
    // the job summary.
    assert!(wf.contains("--ferx ../release/target/ci-test/ferx --root ../release"));
    assert!(wf.contains("--ferx target/ci-test/ferx --root ."));
    assert!(wf.contains("golden.py report") && wf.contains("--summary \"$GITHUB_STEP_SUMMARY\""));
    assert!(wf.contains("name: golden-fits-results") && wf.contains("retention-days: 90"));
}

/// The switch date is a real date, lies after this test was written, and the docs say
/// the same date — so moving it is one reviewed edit to both.
#[test]
fn the_fail_on_band_date_is_set_and_documented() {
    let date = fail_date();
    assert_eq!(date.len(), 10, "{date}");
    assert!(
        date.as_str() > "2026-11-03",
        "the switch must be in the future of #2635"
    );
    for page in [
        "docs/development/golden-fits.qmd",
        "docs/development/sdlc.qmd",
    ] {
        let doc = std::fs::read_to_string(repo_root().join(page)).unwrap();
        assert!(
            doc.contains(&format!("report-only until {date}"))
                || doc.contains(&format!("report-only until **{date}**"))
                || doc.contains(&format!("**report-only until {date}**")),
            "{page} must state the date {date}"
        );
    }
}
