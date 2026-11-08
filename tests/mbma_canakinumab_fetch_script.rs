//! `tools/mbma_canakinumab_fetch.py`'s retry loop against a fake Europe PMC (#2818, #1938).
//!
//! The nightly `slow-tests.yml` core leg fetches the canakinumab data with that script, and
//! `FERX_REQUIRE_EXTERNAL_DATA=1` turns a failed fetch into a red night. Two failure modes
//! made that red for no reason: a payload that downloaded but hashed wrong exited at once
//! with attempts 2–4 unused (#2818), and `urlopen(timeout=…)` bounded each socket read, not
//! the download, so a trickling reply ran into the step's cap and no retry ever ran (#1938).
//!
//! Nothing here touches the network. `tests/data/mbma_canakinumab/fetch_fake_server.py`
//! starts an `http.server` on `127.0.0.1` that replies per scenario, loads the script as a
//! module, points `URL` at it and the two pins at a synthetic `data_9studies.csv` (the real
//! one may not be committed, #1913), records `time.sleep` instead of sleeping, and runs
//! `main()`; a watchdog fails the harness at 25 s, so a fetch that ignores its deadline is
//! a red test, not a hung binary. The re-pin verdict ("one wrong hash on ≥ 2 attempts,
//! every attempt") has three conditions, and each has a twin on its other side:
//! `same_wrong` × `error_then_same_wrong` (a transport error among the hashes),
//! × `always_500` (no hash at all), × `same_wrong --attempts 1` (one attempt).
//!
//! Each sentence of the exit message has a test that dies when it is deleted (measured by
//! deleting each in turn):
//!
//! | sentence | dies in |
//! |---|---|
//! | "… hashed sha256 X on all N attempts, not the pinned P." | `the_same_wrong_hash_every_time_says_re_pin` |
//! | "The same wrong hash … really changed … re-pin …" | `the_same_wrong_hash_every_time_says_re_pin` |
//! | "no attempt of N delivered the pinned upstream …" | every flaky-verdict test |
//! | "Mixed hashes or transport errors … flaky … do not re-pin." | every flaky-verdict test |
//! | "Stopped at the D s --deadline-seconds budget." | `a_trickling_download_stops_at_the_deadline`, `a_silent_server_stops_at_the_deadline` |
//! | "  attempt K: …" (one per attempt) | every failing-fetch test |
//!
//! Not feature-gated: must run in the base `--features ci` job.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// What the harness reported for one scenario.
struct Run {
    log: String,
    message: String,
    exit: i32,
    hits: usize,
    sleeps: usize,
    elapsed: f64,
    wrote_pinned: bool,
    pin: String,
    wrong_a: String,
    wrong_b: String,
}

fn field<'a>(out: &'a str, key: &str) -> &'a str {
    out.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("harness printed no {key}:\n{out}"))
}

fn run(scenario: &str, args: &[&str]) -> Run {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cana-fetch-{scenario}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = Command::new("python3")
        .arg("-I")
        .arg(repo_root().join("tests/data/mbma_canakinumab/fetch_fake_server.py"))
        .arg(repo_root().join("tools/mbma_canakinumab_fetch.py"))
        .arg(&dir)
        .arg(scenario)
        .args(args)
        .output()
        .expect("python3 on PATH");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let log = format!("{stdout}{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        out.status.success() && !log.contains("Traceback"),
        "the harness itself failed:\n{log}"
    );
    let message = stdout
        .split("MESSAGE-BEGIN\n")
        .nth(1)
        .and_then(|s| s.split("MESSAGE-END").next())
        .unwrap_or_else(|| panic!("no message block:\n{log}"))
        .trim_end()
        .to_string();
    Run {
        message,
        exit: field(&stdout, "EXIT").parse().unwrap(),
        hits: field(&stdout, "HITS").parse().unwrap(),
        sleeps: field(&stdout, "SLEEPS").parse().unwrap(),
        elapsed: field(&stdout, "ELAPSED").parse().unwrap(),
        wrote_pinned: field(&stdout, "WROTE_PINNED") == "True",
        pin: field(&stdout, "PIN").to_string(),
        wrong_a: field(&stdout, "WRONG_A").to_string(),
        wrong_b: field(&stdout, "WRONG_B").to_string(),
        log,
    }
}

/// #2818's night: attempt 1 hashes wrong, attempt 2 is right. The fetch succeeds, writes
/// the pinned file, and logs attempt 1's hash so the flake is visible in the CI log.
#[test]
fn a_wrong_hash_then_the_right_one_succeeds_and_logs_the_wrong_hash() {
    let r = run("wrong_then_right", &[]);
    assert_eq!(r.exit, 0, "{}", r.log);
    assert!(r.wrote_pinned, "{}", r.log);
    assert_eq!(r.hits, 2, "{}", r.log);
    assert!(
        r.log.contains(&format!("attempt 1: sha256 {}", r.wrong_a)),
        "attempt 1's wrong hash must be logged:\n{}",
        r.log
    );
}

/// A reply that is not a zip at all fails one attempt and is reported as unreadable, not
/// hashed as if it were the bare CSV; the next attempt succeeds.
#[test]
fn a_garbled_payload_is_a_failed_attempt_not_a_hash() {
    let r = run("garbled_then_right", &[]);
    assert_eq!(r.exit, 0, "{}", r.log);
    assert!(r.wrote_pinned, "{}", r.log);
    assert!(
        r.log
            .contains("attempt 1: unreadable payload: not a zip (27 bytes)"),
        "{}",
        r.log
    );
}

/// One wrong hash on every attempt is the signature of a real upstream change: the message
/// names that hash and the pin and says to re-pin, not that the endpoint is flaky. Also
/// pins two invariants on the same run: `--allow-new-hash` never waives the upstream pin,
/// and there is no sleep after the last attempt (3 attempts, 2 backoffs).
#[test]
fn the_same_wrong_hash_every_time_says_re_pin() {
    let r = run("same_wrong", &["--attempts", "3", "--allow-new-hash"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(!r.wrote_pinned, "{}", r.log);
    assert_eq!(r.hits, 3, "{}", r.log);
    assert_eq!(
        r.sleeps, 2,
        "one backoff between attempts, none after the last:\n{}",
        r.log
    );
    assert!(
        m.contains(&format!(
            "hashed sha256 {} on all 3 attempts, not the pinned {}.",
            r.wrong_a, r.pin
        )),
        "{m}"
    );
    assert!(
        m.contains("really changed") && m.contains("re-pin UPSTREAM_SHA256"),
        "{m}"
    );
    assert!(!m.contains("flaky") && !m.contains("no attempt of"), "{m}");
    for k in 1..=3 {
        assert!(
            m.contains(&format!("  attempt {k}: sha256 {}", r.wrong_a)),
            "{m}"
        );
    }
}

/// Two different wrong hashes cannot both be the new upstream file: the verdict is a
/// flaky endpoint, and every attempt's hash is listed.
#[test]
fn two_alternating_wrong_hashes_say_flaky() {
    let r = run("alternating_wrong", &["--attempts", "3"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        m.contains(&format!(
            "no attempt of 3 delivered the pinned upstream supplemental/case_study_2/data_9studies.csv (sha256 {}).",
            r.pin
        )),
        "{m}"
    );
    assert!(
        m.contains("endpoint is flaky") && m.contains("do not re-pin"),
        "{m}"
    );
    assert!(!m.contains("really changed"), "{m}");
    assert!(
        m.contains(&format!("  attempt 1: sha256 {}", r.wrong_a))
            && m.contains(&format!("  attempt 2: sha256 {}", r.wrong_b))
            && m.contains(&format!("  attempt 3: sha256 {}", r.wrong_a)),
        "{m}"
    );
}

/// The other side of the re-pin gate: the same wrong hash as `same_wrong` on attempts 2–3,
/// but attempt 1 was an HTTP 500, so the run is not "one hash on every attempt" and the
/// verdict is flaky.
#[test]
fn a_transport_error_makes_one_repeated_hash_flaky() {
    let r = run("error_then_same_wrong", &["--attempts", "3"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        m.contains("no attempt of 3") && m.contains("endpoint is flaky"),
        "{m}"
    );
    assert!(!m.contains("really changed"), "{m}");
    assert!(m.contains("  attempt 1: error: HTTPError"), "{m}");
    assert!(
        m.contains(&format!("  attempt 3: sha256 {}", r.wrong_a)),
        "{m}"
    );
}

/// The re-pin verdict needs a hash on every attempt: when every attempt is a transport
/// error there is one distinct "hash" (none), and the verdict must still be flaky.
#[test]
fn transport_errors_on_every_attempt_say_flaky() {
    let r = run("always_500", &["--attempts", "3"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_eq!(r.hits, 3, "{}", r.log);
    assert!(
        m.contains("no attempt of 3") && m.contains("endpoint is flaky"),
        "{m}"
    );
    assert!(!m.contains("really changed") && !m.contains("None"), "{m}");
    for k in 1..=3 {
        assert!(
            m.contains(&format!("  attempt {k}: error: HTTPError")),
            "{m}"
        );
    }
}

/// One attempt cannot tell a changed file from a flake, so a single wrong hash is not
/// told to re-pin; `same_wrong` with three attempts is the twin that is.
#[test]
fn a_single_wrong_hash_is_not_enough_to_say_re_pin() {
    let r = run("same_wrong", &["--attempts", "1"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_eq!(r.sleeps, 0, "{}", r.log);
    assert!(
        m.contains("no attempt of 1") && m.contains("endpoint is flaky"),
        "{m}"
    );
    assert!(!m.contains("really changed"), "{m}");
    assert!(
        m.contains(&format!("  attempt 1: sha256 {}", r.wrong_a)),
        "{m}"
    );
}

/// A server that accepts the request and never answers: the read timeout handed to
/// `urlopen` is capped by the time left, so the 2 s budget ends it, not the 30 s
/// `--timeout`. (The trickler below sends bytes, so it cannot see that cap.)
#[test]
fn a_silent_server_stops_at_the_deadline() {
    let r = run("silent", &["--deadline-seconds", "2", "--timeout", "30"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        (1.9..10.0).contains(&r.elapsed),
        "must stop at the 2 s deadline, not the 30 s read timeout: {} s\n{}",
        r.elapsed,
        r.log
    );
    assert_eq!(r.hits, 1, "{}", r.log);
    assert!(
        m.contains("Stopped at the 2 s --deadline-seconds budget."),
        "{m}"
    );
    assert!(m.contains("  attempt 1: error: "), "{m}");
}

/// The minutes in `.github/workflows/slow-tests.yml` add up (#1938): the script's
/// `--deadline-seconds` sits inside the fetch step's cap, and the core job ceiling holds
/// setup (15) + that cap + the cargo step's inner `timeout`. A job killed by its ceiling
/// reads as `cancelled`, not red, so an overrun here would hide the night's failures.
#[test]
fn slow_tests_fetch_budget_fits_the_core_job_ceiling() {
    let wf = std::fs::read_to_string(repo_root().join(".github/workflows/slow-tests.yml"))
        .expect("slow-tests.yml");
    let num_after = |hay: &str, key: &str| -> f64 {
        let rest = &hay[hay
            .find(key)
            .unwrap_or_else(|| panic!("no `{key}` in slow-tests.yml"))
            + key.len()..];
        let n: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        n.parse()
            .unwrap_or_else(|_| panic!("no number after `{key}`"))
    };
    let step = &wf[wf
        .find("- name: Fetch the canakinumab MBMA data")
        .expect("fetch step")..];
    let step = &step[..step.find("\n      - name:").expect("next step")];
    let deadline_min = num_after(step, "mbma_canakinumab_fetch.py --deadline-seconds") / 60.0;
    let step_cap = num_after(step, "timeout-minutes:");
    let ceiling = num_after(&wf, "timeout-minutes: ${{ matrix.leg == 'core' &&");
    let inner = num_after(&wf, "--kill-after=2m ${{ matrix.leg == 'core' && '");
    assert!(
        deadline_min < step_cap,
        "script deadline {deadline_min} min must end before the step cap {step_cap} min"
    );
    assert!(
        15.0 + step_cap + inner <= ceiling,
        "core ceiling {ceiling} min < 15 setup + {step_cap} fetch + {inner} inner"
    );
}

/// #1938: a reply that keeps trickling bytes never trips a per-read timeout (30 s here,
/// far above the deadline). The 2 s `--deadline-seconds` budget stops it, the message
/// says so, and no further attempt or backoff runs past the budget.
#[test]
fn a_trickling_download_stops_at_the_deadline() {
    let r = run("trickle", &["--deadline-seconds", "2", "--timeout", "30"]);
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        (2.0..10.0).contains(&r.elapsed),
        "must stop at the 2 s deadline, not the 30 s read timeout: {} s\n{}",
        r.elapsed,
        r.log
    );
    assert_eq!(r.hits, 1, "{}", r.log);
    assert_eq!(r.sleeps, 0, "{}", r.log);
    assert!(
        m.contains("Stopped at the 2 s --deadline-seconds budget."),
        "{m}"
    );
    assert!(
        m.contains("  attempt 1: error: TimeoutError: deadline reached mid-download"),
        "{m}"
    );
}
