//! Guard for the peak-memory probe (#3081): `tools/peak-memory/peakmem.py`, its model
//! and data, `baseline.json` and `.github/workflows/peak-memory.yml`.
//!
//! The nightly measures the shipped `ferx` binary's peak RSS on a synthetic stiff ODE
//! model and fails on growth over a committed baseline. What can go wrong without anyone
//! noticing is the *verdict*: a report that cannot see a regression still prints a tidy
//! table. So each verdict is pinned here on synthetic results, paired with a twin that
//! differs only in the property under test (above vs inside the band, same vs other
//! platform, escalated vs not), so a report that always passes or always fails cannot
//! satisfy both.
//!
//! No fit runs: `run` is exercised against a fake `ferx` that prints a canned
//! `/usr/bin/time` report and writes a canned fit JSON. Not feature-gated — Tier 2, runs
//! in the base `--features ci` job.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn probe_dir() -> PathBuf {
    repo_root().join("tools").join("peak-memory")
}

fn script() -> PathBuf {
    probe_dir().join("peakmem.py")
}

fn scratch(slot: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("peakmem-{slot}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// peakmem.py, with the GitHub Actions variables that name the runner removed, so a
/// guard run inside CI records `runner: local` like a run anywhere else.
fn peakmem(args: &[&str]) -> Output {
    peakmem_on(None, args)
}

/// peakmem.py as if on a GitHub Actions runner of `runner` kind (`None`: outside CI).
fn peakmem_on(runner: Option<&str>, args: &[&str]) -> Output {
    let mut cmd = Command::new("python3");
    cmd.env_remove("GITHUB_ACTIONS").env_remove("RUNNER_ENVIRONMENT");
    if let Some(r) = runner {
        cmd.env("GITHUB_ACTIONS", "true").env("RUNNER_ENVIRONMENT", r);
    }
    cmd.arg("-I")
        .arg(script())
        .args(args)
        .output()
        .expect("python3 runs tools/peak-memory/peakmem.py")
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p.to_string_lossy().into_owned()
}

/// The module-level `NAME = <value>` of peakmem.py, as text.
fn constant(name: &str) -> String {
    let src = std::fs::read_to_string(script()).unwrap();
    let prefix = format!("{name} = ");
    src.lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("peakmem.py defines {name} at module level"))
        .split('#')
        .next()
        .unwrap()
        .trim()
        .to_string()
}

const MIB: u64 = 1 << 20;

/// One case record as `run` writes it.
/// `cov` is the `covariance_status` the record carries: `NOT_REQUESTED` is the `eval`
/// case's expected value, `COMPUTED` the `fit` case's.
fn case(peak: u64, escalated: bool, gradient: &str, cov: &str) -> String {
    format!(
        r#"{{"status": "ok", "peak_rss_bytes": {peak}, "wall_s": 70.0, "ofv": -1212.9,
            "escalated": {escalated}, "gradient_method_outer": "{gradient}",
            "covariance_status": "{cov}", "plan_fallback": []}}"#
    )
}

const NOT_REQUESTED: &str = "NotRequested";
const COMPUTED: &str = "Computed";

fn results(platform: &str, cases: &[(&str, String)]) -> String {
    let cs: Vec<String> = cases.iter().map(|(n, c)| format!(r#""{n}": {c}"#)).collect();
    format!(
        r#"{{"schema": 1, "label": "main@abc", "platform": "{platform}", "threads": 4,
            "runner": "github-hosted",
            "model_sha256": "m", "data_sha256": "d", "cases": {{{}}}}}"#,
        cs.join(", ")
    )
}

fn baseline(platform: &str, eval: u64, fit: u64) -> String {
    format!(
        r#"{{"schema": 1, "platform": "{platform}", "threads": 4, "model_sha256": "m",
            "runner": "github-hosted",
            "data_sha256": "d", "reason": "t", "cases": {{
            "eval": {{"peak_rss_bytes": {eval}}}, "fit": {{"peak_rss_bytes": {fit}}}}}}}"#
    )
}

/// `report` of a head with the given eval/fit records against a Linux baseline of
/// 100 MB / 400 MB.
fn report(slot: &str, platform: &str, eval: String, fit: String) -> Output {
    let dir = scratch(slot);
    let res = write(&dir, "results.json", &results(platform, &[("eval", eval), ("fit", fit)]));
    let base = write(&dir, "baseline.json", &baseline("Linux/x86_64", 100 * MIB, 400 * MIB));
    peakmem(&["report", "--results", &res, "--baseline", &base])
}

fn row<'a>(out: &'a str, case: &str) -> &'a str {
    let prefix = format!("| {case} |");
    out.lines().find(|l| l.starts_with(&prefix)).unwrap_or_else(|| panic!("no {case} row:\n{out}"))
}

/// peakmem.py's `CASE_TIMEOUT_S = {"eval": 1800, "fit": 8100}`, parsed.
fn case_timeouts() -> std::collections::BTreeMap<String, u64> {
    let v: serde_json::Value = serde_json::from_str(&constant("CASE_TIMEOUT_S")).unwrap();
    v.as_object().unwrap().iter().map(|(k, s)| (k.clone(), s.as_u64().unwrap())).collect()
}

fn tolerance() -> f64 {
    constant("TOLERANCE").parse().unwrap()
}

/// Peak at `factor` × the 100 MB eval baseline.
fn eval_at(factor: f64) -> String {
    case((100.0 * factor * MIB as f64) as u64, true, "analytic", NOT_REQUESTED)
}

fn fit_ok() -> String {
    case(400 * MIB, true, "analytic", COMPUTED)
}

// -------------------------------------------------------------------------------------
// report
// -------------------------------------------------------------------------------------

/// The regression this exists for (#2939 was 10–40×), and the band edge: just above the
/// tolerance fails, just inside passes. Only the eval case moves, so the fit row of the
/// red report must stay green — the failure names the case that grew.
#[test]
fn a_peak_above_the_band_fails_and_its_twin_inside_passes() {
    let tol = tolerance();
    assert!(tol > 0.0 && tol < 1.0, "TOLERANCE {tol}");

    let red = report("above", "Linux/x86_64", eval_at(1.0 + tol + 0.02), fit_ok());
    let t = text(&red);
    assert!(!red.status.success(), "a peak above the band must fail:\n{t}");
    assert!(row(&t, "eval").contains("regression"), "{t}");
    assert!(row(&t, "fit").contains("within tolerance"), "{t}");
    assert!(t.contains("eval failed"), "the failure names the case: {t}");

    let green = report("inside", "Linux/x86_64", eval_at(1.0 + tol - 0.02), fit_ok());
    let t = text(&green);
    assert!(green.status.success(), "a peak inside the band must pass:\n{t}");
    assert!(row(&t, "eval").contains("within tolerance"), "{t}");

    let blowup = report("blowup", "Linux/x86_64", eval_at(30.0), fit_ok());
    assert!(!blowup.status.success(), "{}", text(&blowup));
    assert!(row(&text(&blowup), "eval").contains("+2900.0%"), "{}", text(&blowup));
}

/// The ratchet: a peak more than the tolerance below the baseline passes but asks to be
/// locked in; its twin inside the band says nothing of the kind.
#[test]
fn an_improvement_passes_and_asks_to_be_accepted() {
    let tol = tolerance();
    let out = report("better", "Linux/x86_64", eval_at(1.0 - tol - 0.05), fit_ok());
    let t = text(&out);
    assert!(out.status.success(), "{t}");
    assert!(row(&t, "eval").contains("improvement") && row(&t, "eval").contains("accept"), "{t}");

    let same = report("same", "Linux/x86_64", eval_at(1.0 - tol + 0.05), fit_ok());
    assert!(!row(&text(&same), "eval").contains("improvement"), "{}", text(&same));
}

/// Another platform's peak is shown, not judged: a 30× blow-up on macOS is not red
/// against a Linux baseline — and the same blow-up on Linux is.
#[test]
fn another_platform_is_shown_not_judged() {
    let out = report("mac", "Darwin/arm64", eval_at(30.0), fit_ok());
    let t = text(&out);
    assert!(out.status.success(), "{t}");
    assert!(t.contains("shown, not judged"), "{t}");
    assert!(row(&t, "eval").contains("not judged"), "{t}");

    let linux = report("linux", "Linux/x86_64", eval_at(30.0), fit_ok());
    assert!(!linux.status.success(), "{}", text(&linux));
}

/// A failed run or a missing case is red, never folded into "no data, pass" — on any
/// platform, since a crash is not a platform difference.
#[test]
fn a_failed_or_missing_case_fails() {
    let failed = r#"{"status": "failed", "error": "exit 137: Killed"}"#.to_string();
    for platform in ["Linux/x86_64", "Darwin/arm64"] {
        let out = report(&format!("failed-{platform}"), platform, failed.clone(), fit_ok());
        let t = text(&out);
        assert!(!out.status.success(), "{t}");
        assert!(row(&t, "eval").contains("exit 137"), "{t}");
    }

    let dir = scratch("missing");
    let res = write(&dir, "results.json", &results("Linux/x86_64", &[("eval", eval_at(1.0))]));
    let base = write(&dir, "baseline.json", &baseline("Linux/x86_64", 100 * MIB, 400 * MIB));
    let out = peakmem(&["report", "--results", &res, "--baseline", &base]);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(row(&text(&out), "fit").contains("missing"), "{}", text(&out));
}

/// A probe that stopped being a probe — `auto` no longer escalates, or the model left
/// the compiled-plan (analytic) path — is red even when its peak is *lower*: that drop
/// is not a memory win, it is a different measurement. The twin that still escalates
/// at the same peak is an ordinary pass.
#[test]
fn a_drifted_probe_fails_even_with_a_lower_peak() {
    let low = (100 * MIB) / 2;
    let out =
        report("noescalate", "Linux/x86_64", case(low, false, "analytic", NOT_REQUESTED), fit_ok());
    let t = text(&out);
    assert!(!out.status.success(), "{t}");
    assert!(row(&t, "eval").contains("no longer escalates"), "{t}");
    assert!(!row(&t, "eval").contains("improvement"), "{t}");

    let fd = report(
        "fd",
        "Linux/x86_64",
        case(100 * MIB, true, "finite_difference", NOT_REQUESTED),
        fit_ok(),
    );
    assert!(!fd.status.success(), "{}", text(&fd));
    assert!(row(&text(&fd), "eval").contains("not analytic"), "{}", text(&fd));

    // Fail closed: a record with no `gradient_method_outer` (renamed, dropped, null) is
    // no evidence of the analytic plan path.
    let missing = case(100 * MIB, true, "analytic", NOT_REQUESTED)
        .replace(r#""gradient_method_outer": "analytic","#, "");
    assert!(!missing.contains("gradient_method_outer"), "{missing}");
    let gone = report("nogradient", "Linux/x86_64", missing, fit_ok());
    assert!(!gone.status.success(), "{}", text(&gone));
    assert!(row(&text(&gone), "eval").contains("not analytic"), "{}", text(&gone));

    let ok = report(
        "escalates",
        "Linux/x86_64",
        case(100 * MIB, true, "analytic", NOT_REQUESTED),
        fit_ok(),
    );
    assert!(ok.status.success(), "{}", text(&ok));
}

/// The `fit` case exists for the covariance step (e7070's post-fit spike). A fit whose
/// step was skipped (`NotRequested`) or failed (anything but `Computed`) at a *lower*
/// peak is drift, not an improvement — while its twin that reports `Computed` at the
/// same peak is the ordinary `▼`. The mirror for `eval`: a covariance step that runs
/// there is drift too. Each assertion names the case and the status it saw.
#[test]
fn a_fit_without_its_covariance_step_is_drift_not_an_improvement() {
    let low = 400 * MIB / 2;
    for status in [NOT_REQUESTED, "Failed"] {
        let out = report(
            &format!("nocov-{status}"),
            "Linux/x86_64",
            eval_at(1.0),
            case(low, true, "analytic", status),
        );
        let t = text(&out);
        assert!(!out.status.success(), "{status}: a fit without its covariance step passed:\n{t}");
        let fit = row(&t, "fit");
        assert!(fit.contains(&format!("covariance status is `{status}`, not `Computed`")), "{t}");
        assert!(!fit.contains("improvement"), "{t}");
        assert!(row(&t, "eval").contains("within tolerance"), "{t}");
    }
    let twin =
        report("withcov", "Linux/x86_64", eval_at(1.0), case(low, true, "analytic", COMPUTED));
    let t = text(&twin);
    assert!(twin.status.success(), "{t}");
    assert!(row(&t, "fit").contains("improvement"), "{t}");

    let evalcov =
        report("evalcov", "Linux/x86_64", case(100 * MIB, true, "analytic", COMPUTED), fit_ok());
    let t = text(&evalcov);
    assert!(!evalcov.status.success(), "{t}");
    assert!(row(&t, "eval").contains("not `NotRequested`"), "{t}");
}

/// Changing the model or the data invalidates the baseline: shown, not judged, with the
/// instruction to re-measure.
#[test]
fn changed_inputs_are_not_judged_against_the_old_baseline() {
    let dir = scratch("inputs");
    let res = results("Linux/x86_64", &[("eval", eval_at(30.0)), ("fit", fit_ok())])
        .replace(r#""model_sha256": "m""#, r#""model_sha256": "m2""#);
    let res = write(&dir, "results.json", &res);
    let base = write(&dir, "baseline.json", &baseline("Linux/x86_64", 100 * MIB, 400 * MIB));
    let out = peakmem(&["report", "--results", &res, "--baseline", &base]);
    let t = text(&out);
    assert!(out.status.success(), "{t}");
    assert!(t.contains("model_sha256") && t.contains("Re-measure"), "{t}");
}

// -------------------------------------------------------------------------------------
// run + accept, against a fake ferx
// -------------------------------------------------------------------------------------

/// `run` rewrites the model's `[fit_options]` per case, runs the binary under
/// `/usr/bin/time`, and reads the peak back; `accept` turns that into a baseline.
/// The fake binary checks the case's spelling of `maxiter` / `covariance`, so a rewrite
/// that measured the eval twice would fail here; it reports a different peak per case,
/// so a parser that read the wrong number would too.
/// A covariance step whose compiled plans failed falls back to finite differences: it
/// still reports `Computed` with `gradient_method_outer = analytic`, so only the
/// `W_PLAN_FD_FALLBACK` warning shows that the spike site was not measured. Such a `fit`
/// at a lower peak is drift, not an improvement, and `accept` refuses it; a record that
/// never checked (`plan_fallback` absent) fails closed. The clean twin at the same peak
/// is the ordinary `▼`.
#[test]
fn a_covariance_plan_fallback_is_drift_not_an_improvement() {
    let low = 100 * MIB;
    let msg = "W_PLAN_FD_FALLBACK: 3 of 100 subjects fell back to finite-difference \
               derivatives (the covariance step)";
    let fallback = case(low, true, "analytic", COMPUTED)
        .replace(r#""plan_fallback": []"#, &format!(r#""plan_fallback": ["{msg}"]"#));
    assert!(fallback.contains("W_PLAN_FD_FALLBACK"), "{fallback}");
    let out = report("covfallback", "Linux/x86_64", eval_at(1.0), fallback.clone());
    let t = text(&out);
    assert!(!out.status.success(), "{t}");
    assert!(
        row(&t, "fit").contains("fell back") && row(&t, "fit").contains("covariance step"),
        "{t}"
    );
    assert!(!row(&t, "fit").contains("improvement"), "{t}");

    let unchecked = case(low, true, "analytic", COMPUTED).replace(r#", "plan_fallback": []"#, "");
    assert!(!unchecked.contains("plan_fallback"), "{unchecked}");
    let out = report("covunchecked", "Linux/x86_64", eval_at(1.0), unchecked);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(row(&text(&out), "fit").contains("not recorded"), "{}", text(&out));

    let clean =
        report("covclean", "Linux/x86_64", eval_at(1.0), case(low, true, "analytic", COMPUTED));
    assert!(clean.status.success(), "{}", text(&clean));
    assert!(row(&text(&clean), "fit").contains("improvement"), "{}", text(&clean));

    let dir = scratch("covfallback-accept");
    let res = write(
        &dir,
        "results.json",
        &results("Linux/x86_64", &[("eval", eval_at(1.0)), ("fit", fallback)]),
    );
    let base = dir.join("baseline.json");
    let acc = peakmem(&[
        "accept",
        "--results",
        &res,
        "--baseline",
        base.to_str().unwrap(),
        "--reason",
        "r",
    ]);
    assert!(!acc.status.success(), "{}", text(&acc));
    assert!(!base.exists(), "a refused accept must not write the baseline");
}

/// A baseline measured off the reference runner (a laptop, a container, emulation) is
/// provisional. The first reference run against it fails, however well its numbers
/// agree, and says to accept its artifact; a local run against the same baseline, and a
/// reference run against a reference baseline, are judged as usual. Each twin differs
/// from the failing one in exactly one runner field.
#[test]
fn a_provisional_baseline_fails_the_first_reference_run() {
    let dir = scratch("provisional");
    let local = |body: String| body.replace(r#""runner": "github-hosted""#, r#""runner": "local""#);
    let head = results("Linux/x86_64", &[("eval", eval_at(1.0)), ("fit", fit_ok())]);
    let base = baseline("Linux/x86_64", 100 * MIB, 400 * MIB);
    let go = |name: &str, head: String, base: String| {
        let res = write(&dir, &format!("{name}-results.json"), &head);
        let b = write(&dir, &format!("{name}-baseline.json"), &base);
        peakmem(&["report", "--results", &res, "--baseline", &b])
    };

    let first = go("first", head.clone(), local(base.clone()));
    let t = text(&first);
    assert!(!first.status.success(), "{t}");
    assert!(t.contains("provisional") && t.contains("accept"), "{t}");
    assert!(row(&t, "eval").contains("within tolerance"), "{t}");

    let both_local = go("bothlocal", local(head.clone()), local(base.clone()));
    assert!(both_local.status.success(), "{}", text(&both_local));
    assert!(!text(&both_local).contains("provisional"), "{}", text(&both_local));

    let reference = go("reference", head, base);
    assert!(reference.status.success(), "{}", text(&reference));
    assert!(!text(&reference).contains("provisional"), "{}", text(&reference));
}

/// The timing table flags a case past `TIMEOUT_WARN` of the per-case timeout, and only
/// that case: the twins sit either side of the gate (the fit at 0.81, the eval at 0.79
/// of it). A flag is a warning, never a verdict, so the report still passes.
#[test]
fn a_case_near_its_timeout_is_flagged_and_still_passes() {
    let warn: f64 = constant("TIMEOUT_WARN").parse().unwrap();
    // Different timeouts per case, so a share computed against the wrong case's (or one
    // shared) timeout lands on the wrong side of the gate.
    let at = |c: String, share: f64, timeout: f64| {
        let out = c.replace(
            r#""wall_s": 70.0"#,
            &format!(r#""wall_s": {}, "timeout_s": {timeout}"#, share * timeout),
        );
        assert_ne!(out, c, "the case record no longer carries `\"wall_s\": 70.0`");
        out
    };
    let (eval, fit) = (at(eval_at(1.0), warn - 0.01, 1000.0), at(fit_ok(), warn + 0.01, 8000.0));
    let dir = scratch("timeout-warn");
    let head = results("Linux/x86_64", &[("eval", eval), ("fit", fit)]);
    let res = write(&dir, "results.json", &head);
    let base = write(&dir, "baseline.json", &baseline("Linux/x86_64", 100 * MIB, 400 * MIB));
    let out = peakmem(&["report", "--results", &res, "--baseline", &base]);
    let t = text(&out);
    assert!(out.status.success(), "a flag is not a failure:\n{t}");
    let timing = |case: &str| {
        t.lines()
            .filter(|l| l.starts_with(&format!("| {case} |")))
            .nth(1)
            .unwrap_or_else(|| panic!("no {case} timing row:\n{t}"))
            .to_string()
    };
    assert!(timing("fit").contains("81% of 8000 s ⚠"), "{}", timing("fit"));
    assert!(
        timing("eval").contains("79% of 1000 s") && !timing("eval").contains('⚠'),
        "{}",
        timing("eval")
    );
}

#[cfg(unix)]
#[test]
fn run_measures_each_case_and_accept_writes_the_baseline() {
    use std::os::unix::fs::PermissionsExt;
    if !Path::new("/usr/bin/time").exists() {
        eprintln!("skipping: no /usr/bin/time on this machine");
        return;
    }
    let dir = scratch("run");
    let fit_json = |cov: &str| {
        format!(
            r#"{{"ofv": -1212.9, "n_iterations": 0, "covariance_status": "{cov}",
            "gradient_method_outer": "analytic", "n_threads_used": 4,
            "wall_time_secs": 61.5, "method_wall_times_secs": [55.25],
            "covariance_wall_time_secs": 0.0,
            "warnings": ["W_ODE_SOLVER_ESCALATION_NOTE: ode_method = auto escalated 3000"]}}"#
        )
    };
    let eval_fit = write(&dir, "eval-fit.json", &fit_json(NOT_REQUESTED));
    let fit_fit = write(&dir, "fit-fit.json", &fit_json(COMPUTED));
    // Allocates per case (eval small, fit large) so /usr/bin/time sees two different
    // peaks: python holding a bytearray of the given size.
    let fake = dir.join("ferx");
    std::fs::write(
        &fake,
        format!(
            "#!/bin/sh\n\
             case \"$*\" in *'--threads 4'*) ;; *) echo 'no --threads 4' >&2; exit 8;; esac\n\
             case \"$*\" in *'--output-format json'*) ;; *) echo 'no json' >&2; exit 9;; esac\n\
             if grep -q '^  maxiter    = 0$' \"$1\" && grep -q '^  covariance = false$' \"$1\"; then\n\
               mb=8; out='{eval_fit}'\n\
             elif grep -q '^  maxiter    = 1$' \"$1\" && grep -q '^  covariance = true$' \"$1\"; then\n\
               mb=160; out='{fit_fit}'\n\
             else echo 'unexpected fit options' >&2; exit 7; fi\n\
             python3 -c \"import time; b = bytearray($mb << 20); b[::4096] = b'x' * len(b[::4096]); time.sleep(0.2)\"\n\
             cp \"$out\" ./stiff_pkpd-fit.json\n"
        ),
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();

    // In its own directory, as the workflow's `peak-memory/` is: the case logs land
    // next to results.json, and must not collide with the fake's canned fit JSONs.
    let out_dir = dir.join("pm");
    let res = out_dir.join("results.json");
    let run_args = ["run", "--ferx", fake.to_str().unwrap(), "--out", res.to_str().unwrap()];
    // Outside GitHub Actions the run is `local`, whatever the platform says.
    let out = peakmem(&[&run_args[..], &["--cases", "eval"]].concat());
    assert!(out.status.success(), "{}", text(&out));
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&res).unwrap()).unwrap();
    assert_eq!(r["runner"], "local", "{r}");

    // The artifact attributes a slow night without a rerun: the case's stderr, each line
    // timestamped (the time report is the last thing ferx's process writes there), and
    // the fit JSON with its warnings, next to results.json.
    let log = std::fs::read_to_string(out_dir.join("eval.stderr.log"))
        .unwrap_or_else(|e| panic!("no eval.stderr.log: {e}"));
    let rss_line = log
        .lines()
        .find(|l| l.contains("aximum resident set size"))
        .unwrap_or_else(|| panic!("time report not in the log:\n{log}"));
    assert!(rss_line.starts_with('[') && rss_line.contains("s] "), "not timestamped: {rss_line}");
    let kept: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.join("eval-fit.json")).expect("fit JSON kept"),
    )
    .unwrap();
    assert_eq!(kept["covariance_status"], NOT_REQUESTED, "{kept}");
    assert!(!out_dir.join("fit.stderr.log").exists(), "a case that was not run left a log");
    // Each case carries the timeout it ran under: its own, from CASE_TIMEOUT_S.
    assert_eq!(r["cases"]["eval"]["timeout_s"].as_u64(), Some(case_timeouts()["eval"]), "{r}");
    // The fit's own phase clock, lifted out of the fit JSON.
    assert_eq!(r["cases"]["eval"]["fit_wall_s"], 61.5, "{r}");
    assert_eq!(r["cases"]["eval"]["method_wall_s"], serde_json::json!([55.25]), "{r}");
    assert_eq!(r["cases"]["eval"]["covariance_wall_s"], 0.0, "{r}");
    let w = r["cases"]["eval"]["warnings"].as_array().unwrap_or_else(|| panic!("{r}"));
    assert!(w.iter().any(|x| x.as_str().unwrap().contains("W_ODE_SOLVER_ESCALATION_NOTE")));

    // `run` lifts a plan-fallback warning out of the fit JSON into `plan_fallback`.
    let clean_eval = std::fs::read_to_string(&eval_fit).unwrap();
    let with_fallback = clean_eval.replace(
        r#""warnings": ["#,
        r#""warnings": ["W_PLAN_FD_FALLBACK: 2 of 100 subjects fell back (the covariance step)", "#,
    );
    assert_ne!(with_fallback, clean_eval);
    std::fs::write(&eval_fit, &with_fallback).unwrap();
    let out = peakmem(&[&run_args[..], &["--cases", "eval"]].concat());
    assert!(out.status.success(), "{}", text(&out));
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&res).unwrap()).unwrap();
    let fb = r["cases"]["eval"]["plan_fallback"].as_array().unwrap_or_else(|| panic!("{r}"));
    assert!(fb.len() == 1 && fb[0].as_str().unwrap().contains("covariance step"), "{r}");
    std::fs::write(&eval_fit, &clean_eval).unwrap();

    let out = peakmem_on(Some("github-hosted"), &run_args);
    assert!(out.status.success(), "{}", text(&out));
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&res).unwrap()).unwrap();
    assert_eq!(r["runner"], "github-hosted", "{r}");
    for c in ["eval", "fit"] {
        assert_eq!(r["cases"][c]["status"], "ok", "{c}: {r}");
        assert_eq!(r["cases"][c]["escalated"], true, "{c}: {r}");
    }
    // Recorded as the plain variant name, which is what `report` compares against.
    assert_eq!(r["cases"]["eval"]["covariance_status"], NOT_REQUESTED, "{r}");
    assert_eq!(r["cases"]["fit"]["covariance_status"], COMPUTED, "{r}");
    for c in ["eval", "fit"] {
        assert_eq!(r["cases"][c]["plan_fallback"], serde_json::json!([]), "{c}: {r}");
    }
    let eval = r["cases"]["eval"]["peak_rss_bytes"].as_u64().unwrap();
    let fit_peak = r["cases"]["fit"]["peak_rss_bytes"].as_u64().unwrap();
    // The fit case allocates 152 MiB more than the eval case.
    assert!(fit_peak > eval + 100 * MIB, "eval {eval} fit {fit_peak}: wrong case or wrong unit");
    assert!(fit_peak < 2048 * MIB, "fit {fit_peak}: a kbytes/bytes mix-up");

    let base = dir.join("baseline.json");
    let acc = peakmem(&[
        "accept",
        "--results",
        res.to_str().unwrap(),
        "--baseline",
        base.to_str().unwrap(),
        "--reason",
        "test",
    ]);
    assert!(acc.status.success(), "{}", text(&acc));
    let b: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&base).unwrap()).unwrap();
    assert_eq!(b["cases"]["fit"]["peak_rss_bytes"].as_u64(), Some(fit_peak));
    assert_eq!(b["reason"], "test");
    assert_eq!(b["runner"], "github-hosted", "accept carries the run's provenance: {b}");
    // The accepted baseline judges its own run green.
    let again = peakmem(&[
        "report",
        "--results",
        res.to_str().unwrap(),
        "--baseline",
        base.to_str().unwrap(),
    ]);
    assert!(again.status.success(), "{}", text(&again));
}

/// A case past its timeout is killed then, not when ferx finishes, and is still
/// recorded with its log. ferx runs under `/usr/bin/time`, so it is a grandchild:
/// killing only `time` leaves ferx holding the pipes, and the stderr reader blocks
/// until ferx exits on its own (PR #3147 review r1 #1). The fake's work is a `sleep`
/// child of the shell, a great-grandchild, so only a kill of the whole process group
/// ends it.
#[cfg(unix)]
#[test]
fn a_case_past_its_timeout_is_killed_and_still_recorded() {
    use std::os::unix::fs::PermissionsExt;
    if !Path::new("/usr/bin/time").exists() {
        eprintln!("skipping: no /usr/bin/time on this machine");
        return;
    }
    let dir = scratch("timeout");
    let fake = dir.join("ferx");
    std::fs::write(&fake, "#!/bin/sh\necho 'fake start' >&2\nsleep 30\necho 'fake end' >&2\n")
        .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out_dir = dir.join("pm");
    let res = out_dir.join("results.json");
    let t0 = std::time::Instant::now();
    let out = peakmem(&[
        "run",
        "--ferx",
        fake.to_str().unwrap(),
        "--out",
        res.to_str().unwrap(),
        "--cases",
        "eval",
        "--timeout",
        "2",
    ]);
    let took = t0.elapsed().as_secs_f64();
    assert!(out.status.success(), "{}", text(&out));
    // 2 s timeout against a 30 s sleep: anything near 30 s means ferx was waited for.
    assert!(took < 15.0, "run returned after {took:.1} s: the timeout did not kill ferx");
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&res).unwrap()).unwrap();
    let rec = &r["cases"]["eval"];
    assert_eq!(rec["status"], "timeout", "{r}");
    assert_eq!(rec["timeout_s"].as_u64(), Some(2), "{r}");
    let wall = rec["wall_s"].as_f64().unwrap_or_else(|| panic!("{r}"));
    assert!((2.0..15.0).contains(&wall), "wall_s {wall}: not the time of the kill");
    let log = std::fs::read_to_string(out_dir.join("eval.stderr.log"))
        .unwrap_or_else(|e| panic!("no eval.stderr.log after a timeout: {e}"));
    assert!(log.contains("] fake start"), "log lost what ferx wrote before the kill:\n{log}");
    assert!(!log.contains("fake end"), "ferx ran to the end:\n{log}");
}

/// `accept` refuses what would make a meaningless baseline: no reason, a failed case,
/// a run where `auto` did not escalate, or a fit whose covariance step did not run.
#[test]
fn accept_refuses_a_failed_or_drifted_run() {
    let dir = scratch("accept");
    let base = dir.join("baseline.json");
    let try_accept = |name: &str, body: String, reason: &str| {
        let res = write(&dir, name, &body);
        peakmem(&[
            "accept",
            "--results",
            &res,
            "--baseline",
            base.to_str().unwrap(),
            "--reason",
            reason,
        ])
    };
    let good = results("Linux/x86_64", &[("eval", eval_at(1.0)), ("fit", fit_ok())]);
    assert!(!try_accept("noreason.json", good.clone(), " ").status.success());
    let failed = results(
        "Linux/x86_64",
        &[("eval", r#"{"status": "failed", "error": "x"}"#.to_string()), ("fit", fit_ok())],
    );
    assert!(!try_accept("failed.json", failed, "r").status.success());
    let drift = results(
        "Linux/x86_64",
        &[("eval", case(MIB, false, "analytic", NOT_REQUESTED)), ("fit", fit_ok())],
    );
    assert!(!try_accept("drift.json", drift, "r").status.success());
    let nocov = results(
        "Linux/x86_64",
        &[("eval", eval_at(1.0)), ("fit", case(400 * MIB, true, "analytic", NOT_REQUESTED))],
    );
    assert!(!try_accept("nocov.json", nocov, "r").status.success());
    assert!(!base.exists(), "a refused accept must not write the baseline");
    let ok = try_accept("good.json", good, "r");
    assert!(ok.status.success(), "{}", text(&ok));
}

// -------------------------------------------------------------------------------------
// The committed probe, baseline and workflow
// -------------------------------------------------------------------------------------

/// The model carries the `eval` spelling of the two lines `run` rewrites, exactly once
/// each, and the committed baseline was taken on these very inputs at the script's
/// thread count, on Linux — so the first nightly judges instead of saying "not judged".
#[test]
fn the_committed_baseline_matches_the_committed_probe() {
    let model = std::fs::read_to_string(probe_dir().join("stiff_pkpd.ferx")).unwrap();
    assert_eq!(model.matches("\n  maxiter    = 0\n").count(), 1, "eval spelling of maxiter");
    assert_eq!(model.matches("\n  covariance = false\n").count(), 1, "eval spelling of covariance");
    assert!(
        model.contains("ode(states=[depot, central, periph, effect, prol, tr1, tr2, tr3, circ])")
    );

    let base: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(probe_dir().join("baseline.json")).unwrap())
            .unwrap();
    let sha = |name: &str| {
        let out = Command::new("python3")
            .args(["-I", "-c"])
            .arg("import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())")
            .arg(probe_dir().join(name))
            .output()
            .unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    };
    assert_eq!(
        base["model_sha256"],
        sha("stiff_pkpd.ferx"),
        "re-measure and accept after editing the model"
    );
    assert_eq!(
        base["data_sha256"],
        sha("stiff_pkpd.csv"),
        "re-measure and accept after editing the data"
    );
    assert_eq!(base["threads"].to_string(), constant("FERX_THREADS"));
    assert!(
        base["platform"].as_str().unwrap().starts_with("Linux/"),
        "Linux is the reference (#1688)"
    );
    assert!(!base["reason"].as_str().unwrap().trim().is_empty());
    assert!(base["runner"].is_string(), "the baseline names the runner it was measured on");
    for c in ["eval", "fit"] {
        let p = base["cases"][c]["peak_rss_bytes"].as_u64().unwrap_or(0);
        assert!(p > 10 * MIB, "{c} baseline {p} bytes is not a plausible peak");
    }
}

#[test]
fn workflow_is_a_linux_nightly_on_the_shipped_binary() {
    let wf =
        std::fs::read_to_string(repo_root().join(".github/workflows/peak-memory.yml")).unwrap();
    let runs_on: Vec<&str> =
        wf.lines().filter(|l| l.trim_start().starts_with("runs-on:")).collect();
    assert_eq!(runs_on, ["    runs-on: ubuntu-latest"], "Linux only (#1688)");
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
    assert_eq!(triggers, ["schedule", "workflow_dispatch"], "not a per-PR job: {on:?}");
    // The shipped profile (fat LTO + mimalloc), not ci-test.
    assert!(wf.contains("cargo build --release -p ferx-cli"));
    assert!(wf.contains("--ferx target/release/ferx"));
    assert!(wf.contains("peakmem.py report") && wf.contains("--summary \"$GITHUB_STEP_SUMMARY\""));
    assert!(wf.contains("name: peak-memory-results"));

    // The job outlives every case's own timeout plus the build: otherwise the slow night
    // that most needs its logs is killed by GitHub before peakmem.py records a `timeout`
    // and the artifact is uploaded. 20 min covers a cold fat-LTO build (5.5 min warm).
    let job_min: u64 = wf
        .lines()
        .find_map(|l| l.trim().strip_prefix("timeout-minutes:"))
        .expect("the job sets timeout-minutes")
        .trim()
        .parse()
        .unwrap();
    let cases: u64 = case_timeouts().values().sum();
    assert_eq!(case_timeouts().len(), 2, "a timeout for each of eval and fit");
    assert!(
        job_min * 60 >= cases + 20 * 60,
        "timeout-minutes {job_min} does not cover the case timeouts ({cases} s) + 20 min build"
    );
}
