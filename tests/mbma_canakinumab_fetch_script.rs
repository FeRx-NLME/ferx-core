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
//! one may not be committed, #1913), gives the script a virtual clock whose `sleep` returns
//! at once but spends deadline, and runs `main()`. A watchdog fails the harness at 25 s, so
//! a fetch that ignores its deadline is a red test, not a hung binary.
//!
//! The verdict is decided on the **complete downloads' hashes** alone; a transport error or
//! an unreadable payload says nothing about the file. One wrong hash on ≥ 2 downloads →
//! re-pin; ≥ 2 different wrong hashes → flaky; fewer than 2 downloads → inconclusive. Each
//! arm has a twin across each of its boundaries: `same_wrong` (re-pin) × `alternating_wrong`
//! (flaky) × `same_wrong --attempts 1` (inconclusive); `error_then_same_wrong` and
//! `same_wrong_then_trickle` show an error or a deadline does not hide a repeated hash, and
//! `always_500` that errors alone decide nothing.
//!
//! Each sentence of the exit message has a test that dies when it is deleted (measured by
//! deleting each in turn):
//!
//! | sentence | dies in |
//! |---|---|
//! | "… hashed sha256 X on all K complete downloads, not the pinned P." | every re-pin test |
//! | "The same wrong hash … really changed … re-pin …" | every re-pin test |
//! | "no attempt of N delivered the pinned upstream …" | the flaky and inconclusive tests |
//! | "D different wrong hashes mean the endpoint is flaky … do not re-pin." | `two_alternating_wrong_hashes_say_flaky` |
//! | "Fewer than two complete downloads cannot tell … re-run the fetch." | the inconclusive tests |
//! | "Stopped at the D s --deadline-seconds budget." | every deadline test |
//! | "  attempt K: …" (one per attempt) | every failing-fetch test |
//! | "--attempts: must be at least 1, got N" | `zero_attempts_is_a_usage_error` |
//! | "--timeout and --deadline-seconds must be positive" | `a_zero_deadline_is_a_usage_error` |
//! | "--from F: neither … nor … found in the zip (N entries)" | `from_a_zip_without_the_csv_names_what_is_missing` |
//!
//! Not covered, by design: a server that trickles its *headers* forever. `urlopen` reads
//! them with the timeout set at open, so each header byte resets it; the step's 15-min cap
//! is the bound there (`headers_arriving_after_the_deadline_stop_before_the_body` covers
//! headers that end late, not ones that never end).
//!
//! Not feature-gated: must run in the base `--features ci` job.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// What the harness reported for one run.
struct Run {
    log: String,
    message: String,
    exit: i32,
    hits: usize,
    sleeps: usize,
    sleep_total: f64,
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

/// Run the harness on `scenario`, in a scratch dir of its own named `slot` (one per test,
/// so two tests on one scenario never share it).
fn run(slot: &str, scenario: &str, args: &[&str]) -> Run {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cana-fetch-{slot}"));
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
        sleep_total: field(&stdout, "SLEEP_TOTAL").parse().unwrap(),
        elapsed: field(&stdout, "ELAPSED").parse().unwrap(),
        wrote_pinned: field(&stdout, "WROTE_PINNED") == "True",
        pin: field(&stdout, "PIN").to_string(),
        wrong_a: field(&stdout, "WRONG_A").to_string(),
        wrong_b: field(&stdout, "WRONG_B").to_string(),
        log,
    }
}

/// The re-pin verdict's two sentences, for `k` complete downloads of `r.wrong_a`.
fn assert_re_pin(r: &Run, k: usize) {
    let m = &r.message;
    assert!(
        m.contains(&format!(
            "hashed sha256 {} on all {k} complete downloads, not the pinned {}.",
            r.wrong_a, r.pin
        )),
        "{m}"
    );
    assert!(
        m.contains("really changed") && m.contains("re-pin UPSTREAM_SHA256"),
        "{m}"
    );
    assert!(
        !m.contains("flaky") && !m.contains("no attempt of") && !m.contains("Fewer than two"),
        "{m}"
    );
}

/// The inconclusive verdict's two sentences, after `n` attempts.
fn assert_inconclusive(r: &Run, n: usize) {
    let m = &r.message;
    assert!(
        m.contains(&format!("no attempt of {n} delivered the pinned upstream")),
        "{m}"
    );
    assert!(
        m.contains(
            "Fewer than two complete downloads cannot tell a changed file from a flaky \
             endpoint: re-run the fetch."
        ),
        "{m}"
    );
    assert!(
        !m.contains("really changed") && !m.contains("flaky,"),
        "{m}"
    );
}

/// #2818's night: attempt 1 hashes wrong, attempt 2 is right. The fetch succeeds, writes
/// the pinned file, and logs attempt 1's hash so the flake is visible in the CI log.
#[test]
fn a_wrong_hash_then_the_right_one_succeeds_and_logs_the_wrong_hash() {
    let r = run("wrong-then-right", "wrong_then_right", &[]);
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
/// hashed as if it were the bare CSV; the next attempt succeeds. The log line quotes the
/// start of the reply, the only clue to what the endpoint sent instead.
#[test]
fn a_garbled_payload_is_a_failed_attempt_not_a_hash() {
    let r = run("garbled", "garbled_then_right", &[]);
    assert_eq!(r.exit, 0, "{}", r.log);
    assert!(r.wrote_pinned, "{}", r.log);
    assert!(
        r.log.contains(
            "attempt 1: unreadable payload: not a zip (27 bytes): b'<html>gateway hiccup</html>'"
        ),
        "{}",
        r.log
    );
}

/// A connection closed before `Content-Length` is a short transfer, and is logged as one,
/// not as "not a zip" (`read1` returns `b""` on an early close where `read()` raised).
#[test]
fn a_truncated_transfer_is_logged_as_incomplete() {
    let r = run("truncated", "truncated_then_right", &[]);
    assert_eq!(r.exit, 0, "{}", r.log);
    assert!(r.wrote_pinned, "{}", r.log);
    assert!(
        r.log.contains("attempt 1: error: IncompleteRead") && !r.log.contains("not a zip"),
        "{}",
        r.log
    );
}

/// One wrong hash on every attempt is the signature of a real upstream change: the message
/// names that hash and the pin and says to re-pin. Also pins two invariants on the same
/// run: `--allow-new-hash` never waives the upstream pin, and there is no sleep after the
/// last attempt (3 attempts, 2 backoffs).
#[test]
fn the_same_wrong_hash_every_time_says_re_pin() {
    let r = run(
        "same-wrong",
        "same_wrong",
        &["--attempts", "3", "--allow-new-hash"],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(!r.wrote_pinned, "{}", r.log);
    assert_eq!(r.hits, 3, "{}", r.log);
    assert_eq!(
        r.sleeps, 2,
        "one backoff between attempts, none after the last:\n{}",
        r.log
    );
    assert_re_pin(&r, 3);
    for k in 1..=3 {
        assert!(
            r.message
                .contains(&format!("  attempt {k}: sha256 {}", r.wrong_a)),
            "{}",
            r.message
        );
    }
}

/// Two different wrong hashes cannot both be the new upstream file: the verdict is a
/// flaky endpoint, and every attempt's hash is listed.
#[test]
fn two_alternating_wrong_hashes_say_flaky() {
    let r = run("alternating", "alternating_wrong", &["--attempts", "3"]);
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
        m.contains(
            "2 different wrong hashes mean the endpoint is flaky, not that the file \
             changed: re-run the fetch, and do not re-pin."
        ),
        "{m}"
    );
    assert!(
        !m.contains("really changed") && !m.contains("Fewer than two"),
        "{m}"
    );
    assert!(
        m.contains(&format!("  attempt 1: sha256 {}", r.wrong_a))
            && m.contains(&format!("  attempt 2: sha256 {}", r.wrong_b))
            && m.contains(&format!("  attempt 3: sha256 {}", r.wrong_a)),
        "{m}"
    );
}

/// A transport error says nothing about the file: an HTTP 500 on attempt 1 followed by the
/// same wrong hash twice is still "re-pin", counted over the 2 complete downloads.
#[test]
fn a_transport_error_does_not_hide_a_repeated_hash() {
    let r = run(
        "error-then-same-wrong",
        "error_then_same_wrong",
        &["--attempts", "3"],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_re_pin(&r, 2);
    assert!(
        r.message.contains("  attempt 1: error: HTTPError"),
        "{}",
        r.message
    );
}

/// Neither does the deadline: two complete downloads hash the same wrong value, and the
/// third trickles into the 5 s budget (backoffs 1 s + 2 s on the virtual clock, then ~2 s of
/// trickle). The verdict is re-pin and the deadline is reported. The loop's one deadline
/// gate also stops a 4th attempt from starting (3 hits, not 4).
#[test]
fn a_deadline_does_not_hide_a_repeated_hash() {
    let r = run(
        "same-wrong-then-trickle",
        "same_wrong_then_trickle",
        &["--deadline-seconds", "5", "--timeout", "30"],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_eq!(r.hits, 3, "{}", r.log);
    assert!(
        !r.log.contains("attempt 4/4") && !r.message.contains("attempt 4:"),
        "no attempt may start after the deadline:\n{}",
        r.log
    );
    assert_re_pin(&r, 2);
    assert!(
        r.message
            .contains("Stopped at the 5 s --deadline-seconds budget."),
        "{}",
        r.message
    );
}

/// When every attempt is a transport error there is no hash to judge: inconclusive.
#[test]
fn transport_errors_on_every_attempt_are_inconclusive() {
    let r = run("always-500", "always_500", &["--attempts", "3"]);
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_eq!(r.hits, 3, "{}", r.log);
    assert_inconclusive(&r, 3);
    assert!(!r.message.contains("None"), "{}", r.message);
    for k in 1..=3 {
        assert!(
            r.message
                .contains(&format!("  attempt {k}: error: HTTPError")),
            "{}",
            r.message
        );
    }
}

/// One download cannot tell a changed file from a flake, so a single wrong hash is
/// inconclusive; `same_wrong` with three attempts is the twin that says re-pin.
#[test]
fn a_single_wrong_hash_is_inconclusive() {
    let r = run("single-wrong", "same_wrong", &["--attempts", "1"]);
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_eq!(r.sleeps, 0, "{}", r.log);
    assert_inconclusive(&r, 1);
    assert!(
        r.message
            .contains(&format!("  attempt 1: sha256 {}", r.wrong_a)),
        "{}",
        r.message
    );
}

/// Each backoff is capped by the time left. With a 2.5 s budget and a 1 s backoff base the
/// sleeps are 1 s, then min(2 s, 1.5 s left): 2.5 s in all, and the next attempt never
/// starts. Uncapped they would be 1 + 2 = 3 s, past the budget.
#[test]
fn the_backoff_never_sleeps_past_the_deadline() {
    let r = run(
        "backoff-cap",
        "same_wrong",
        &["--attempts", "4", "--deadline-seconds", "2.5"],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        r.sleep_total.is_finite() && r.sleep_total <= 2.5 + 1e-3,
        "slept {} s on a 2.5 s budget\n{}",
        r.sleep_total,
        r.log
    );
    assert_eq!(r.hits, 2, "{}", r.log);
    assert!(
        !r.log.contains("attempt 3/4") && !r.message.contains("attempt 3:"),
        "no attempt may start after the deadline:\n{}",
        r.log
    );
    assert_re_pin(&r, 2);
    assert!(
        r.message
            .contains("Stopped at the 2.5 s --deadline-seconds budget."),
        "{}",
        r.message
    );
}

/// A server that accepts the request and never answers: the read timeout handed to
/// `urlopen` is capped by the time left, so the 2 s budget ends it, not the 30 s
/// `--timeout`. (The trickler sends bytes, so it cannot see that cap.)
#[test]
fn a_silent_server_stops_at_the_deadline() {
    let r = run(
        "silent",
        "silent",
        &["--deadline-seconds", "2", "--timeout", "30"],
    );
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

/// The socket timeout is re-capped before every read, not once at open. This server sends
/// bytes for 1.8 s, then goes silent: a timeout fixed at open (min(30, 2) = 2 s) would let
/// the stalled read run to ~3.8 s; re-capped, it ends at the 2 s deadline.
#[test]
fn a_stall_after_a_trickle_stops_at_the_deadline() {
    let r = run(
        "trickle-then-stall",
        "trickle_then_stall",
        &[
            "--attempts",
            "1",
            "--deadline-seconds",
            "2",
            "--timeout",
            "30",
        ],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        (1.9..3.0).contains(&r.elapsed),
        "must stop at the 2 s deadline, not 2 s after the stall began: {} s\n{}",
        r.elapsed,
        r.log
    );
    assert!(
        r.message
            .contains("Stopped at the 2 s --deadline-seconds budget.")
            && r.message
                .contains("  attempt 1: error: TimeoutError: deadline reached mid-download"),
        "{}",
        r.message
    );
}

/// A response whose headers arrive after the deadline is refused before the body is
/// read: `urlopen` returns at ~2.5 s on a 2 s budget, and the body loop reports the
/// deadline with 0 bytes received instead of handing a negative timeout to the socket.
#[test]
fn headers_arriving_after_the_deadline_stop_before_the_body() {
    let r = run(
        "slow-headers",
        "slow_headers",
        &[
            "--attempts",
            "1",
            "--deadline-seconds",
            "2",
            "--timeout",
            "30",
        ],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(!r.wrote_pinned, "{}", r.log);
    assert!(
        r.message.contains(
            "  attempt 1: error: TimeoutError: deadline reached mid-download (0 bytes received)"
        ) && r
            .message
            .contains("Stopped at the 2 s --deadline-seconds budget."),
        "{}",
        r.message
    );
}

/// The twin across the deadline-or-timeout gate: the same stall, but `--timeout 0.5` cuts
/// it long before the 10 s budget. That is a stalled server, reported as a socket timeout,
/// not as the deadline (and no "Stopped at" line).
#[test]
fn a_stall_cut_by_the_read_timeout_is_not_the_deadline() {
    let r = run(
        "trickle-then-stall-timeout",
        "trickle_then_stall",
        &[
            "--attempts",
            "1",
            "--deadline-seconds",
            "10",
            "--timeout",
            "0.5",
        ],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        (1.9..5.0).contains(&r.elapsed),
        "the 0.5 s read timeout, not the 10 s deadline, ends it: {} s\n{}",
        r.elapsed,
        r.log
    );
    assert!(
        // The class name differs by Python: `timeout: timed out` on 3.9,
        // `TimeoutError: timed out` on ≥ 3.10 (CI). "timed out" is both.
        r.message.contains("  attempt 1: error: ")
            && r.message.contains("timed out")
            && !r.message.contains("deadline reached")
            && !r.message.contains("Stopped at"),
        "{}",
        r.message
    );
}

/// #1938: a reply that keeps trickling bytes never trips a per-read timeout (30 s here,
/// far above the deadline). The 2 s `--deadline-seconds` budget stops it, the message
/// says so, and no further attempt or backoff runs past the budget.
#[test]
fn a_trickling_download_stops_at_the_deadline() {
    let r = run(
        "trickle",
        "trickle",
        &["--deadline-seconds", "2", "--timeout", "30"],
    );
    let m = &r.message;
    assert_eq!(r.exit, 1, "{}", r.log);
    assert!(
        (2.0..10.0).contains(&r.elapsed),
        "must stop at the 2 s deadline, not the 30 s read timeout: {} s\n{}",
        r.elapsed,
        r.log
    );
    assert_eq!(r.hits, 1, "{}", r.log);
    assert!(r.sleep_total == 0.0, "{}", r.log);
    assert!(
        m.contains("Stopped at the 2 s --deadline-seconds budget."),
        "{m}"
    );
    assert!(
        m.contains("  attempt 1: error: TimeoutError: deadline reached mid-download"),
        "{m}"
    );
}

/// The deadline expiring during the *last* attempt is reported too: that is the attempt
/// most likely to meet it on CI, and it leaves the loop without a backoff.
#[test]
fn a_deadline_on_the_last_attempt_is_reported() {
    let r = run(
        "trickle-last",
        "trickle",
        &[
            "--attempts",
            "1",
            "--deadline-seconds",
            "2",
            "--timeout",
            "30",
        ],
    );
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_inconclusive(&r, 1);
    assert!(
        r.message
            .contains("Stopped at the 2 s --deadline-seconds budget."),
        "{}",
        r.message
    );
}

/// `--attempts` below 1 is a usage error, not a report that "no attempt of 0" succeeded.
#[test]
fn zero_attempts_is_a_usage_error() {
    let r = run("zero-attempts", "same_wrong", &["--attempts", "0"]);
    assert_eq!(r.exit, 2, "{}", r.log);
    assert_eq!(r.hits, 0, "{}", r.log);
    assert!(
        r.log.contains("--attempts: must be at least 1, got 0"),
        "{}",
        r.log
    );
}

/// A zero time budget is a usage error too.
#[test]
fn a_zero_deadline_is_a_usage_error() {
    let r = run("zero-deadline", "same_wrong", &["--deadline-seconds", "0"]);
    assert_eq!(r.exit, 2, "{}", r.log);
    assert_eq!(r.hits, 0, "{}", r.log);
    assert!(
        r.log
            .contains("--timeout and --deadline-seconds must be positive"),
        "{}",
        r.log
    );
}

/// `--from` a zip holding neither the CSV nor the inner zip exits with a message naming
/// the file and what is missing, not a traceback.
#[test]
fn from_a_zip_without_the_csv_names_what_is_missing() {
    let zip = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cana-fetch-empty.zip");
    // The 22-byte end-of-central-directory record: a valid, empty zip.
    let mut eocd = b"PK\x05\x06".to_vec();
    eocd.resize(22, 0);
    std::fs::write(&zip, eocd).unwrap();
    let zip = zip.to_str().unwrap();
    let r = run("from-empty-zip", "same_wrong", &["--from", zip]);
    assert_eq!(r.exit, 1, "{}", r.log);
    assert_eq!(r.hits, 0, "{}", r.log);
    assert_eq!(
        r.message,
        format!(
            "--from {zip}: neither supplemental/case_study_2/data_9studies.csv nor \
             PSP4-15-e70158-s002.zip found in the zip (0 entries)"
        )
    );
}

/// The minutes in `.github/workflows/slow-tests.yml` add up (#1938). The script's worst
/// case is its deadline plus one read that started just before it (`--timeout`), and that
/// must end before the fetch step's cap; the core job ceiling must hold setup (15) + that
/// cap + the cargo step's inner `timeout` + its `--kill-after` grace. A job killed by its
/// ceiling reads as `cancelled`, not red, so an overrun here would hide the night's failures.
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
    // Read the flags from the command, not from the comment above it, which quotes them.
    let cmd = &step[step.find("run: |").expect("fetch step command")..];
    let deadline_s = num_after(cmd, "mbma_canakinumab_fetch.py --deadline-seconds");
    let timeout_s = num_after(cmd, "--timeout");
    let step_cap = num_after(step, "timeout-minutes:");
    let ceiling = num_after(&wf, "timeout-minutes: ${{ matrix.leg == 'core' &&");
    let kill_after = num_after(&wf, "timeout --signal=INT --kill-after=");
    let cargo_timeout = &wf[wf
        .find("timeout --signal=INT --kill-after=")
        .expect("cargo step timeout")..];
    let inner = num_after(cargo_timeout, "${{ matrix.leg == 'core' && '");
    assert!(
        deadline_s + timeout_s < step_cap * 60.0,
        "script worst case {deadline_s} s + {timeout_s} s must end before the step cap \
         {step_cap} min"
    );
    assert!(
        15.0 + step_cap + inner + kill_after <= ceiling,
        "core ceiling {ceiling} min < 15 setup + {step_cap} fetch + {inner} inner + \
         {kill_after} kill-after"
    );
}
