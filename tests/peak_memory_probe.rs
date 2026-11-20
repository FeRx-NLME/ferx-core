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

fn peakmem(args: &[&str]) -> Output {
    Command::new("python3")
        .arg("-I")
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
            "covariance_status": "{cov}"}}"#
    )
}

const NOT_REQUESTED: &str = "NotRequested";
const COMPUTED: &str = "Computed";

fn results(platform: &str, cases: &[(&str, String)]) -> String {
    let cs: Vec<String> = cases.iter().map(|(n, c)| format!(r#""{n}": {c}"#)).collect();
    format!(
        r#"{{"schema": 1, "label": "main@abc", "platform": "{platform}", "threads": 4,
            "model_sha256": "m", "data_sha256": "d", "cases": {{{}}}}}"#,
        cs.join(", ")
    )
}

fn baseline(platform: &str, eval: u64, fit: u64) -> String {
    format!(
        r#"{{"schema": 1, "platform": "{platform}", "threads": 4, "model_sha256": "m",
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

    let res = dir.join("results.json");
    let out = peakmem(&["run", "--ferx", fake.to_str().unwrap(), "--out", res.to_str().unwrap()]);
    assert!(out.status.success(), "{}", text(&out));
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&res).unwrap()).unwrap();
    for c in ["eval", "fit"] {
        assert_eq!(r["cases"][c]["status"], "ok", "{c}: {r}");
        assert_eq!(r["cases"][c]["escalated"], true, "{c}: {r}");
    }
    // Recorded as the plain variant name, which is what `report` compares against.
    assert_eq!(r["cases"]["eval"]["covariance_status"], NOT_REQUESTED, "{r}");
    assert_eq!(r["cases"]["fit"]["covariance_status"], COMPUTED, "{r}");
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
}
