//! `tools/check-external-links.py` and `.github/workflows/external-links.yml` (#2487).
//!
//! docs-lint's R4 checks internal links only, because that crate makes no network call
//! (#969, #1163). The script checks the external ones: a 2xx after redirects, and for a
//! `#fragment` an `id=`/`name=` on the final page. The workflow runs it weekly and on
//! dispatch, never per PR.
//!
//! Nothing here touches the network. The checker is pointed at a throwaway HTTP server on
//! `127.0.0.1` through `--url`, and the collector at a scratch `docs/` under
//! `CARGO_TARGET_TMPDIR`. Each failure verdict is paired with a twin on the same server that
//! differs in the one property under test (anchor present vs absent, 404 vs 200), so a
//! checker that always passes or always fails cannot satisfy both.
//!
//! The skip entries (`SKIP_URLS`, and `HOST_POLICY`'s Wiley skip) are pinned without a
//! fetch, since a skipped URL is never requested. Not pinned: the `doi.org` no-follow entry,
//! which needs the real resolver to answer.
//!
//! Not feature-gated: must run in the base `--features ci` job.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script() -> PathBuf {
    repo_root().join("tools").join("check-external-links.py")
}

fn scratch(slot: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ext-links-{slot}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run the script with `--docs <docs>` and fast retry settings.
fn run(docs: &Path, args: &[&str]) -> Output {
    Command::new("python3")
        .arg("-I")
        .arg(script())
        .arg("--docs")
        .arg(docs)
        .args(["--retries", "2", "--backoff", "0", "--timeout", "5"])
        .args(args)
        .output()
        .expect("python3 on PATH")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// A local HTTP/1.0 server. Routes:
/// `/page` 200 with `id="present"` and `name="named"`; `/moved` 301 → `/page`;
/// `/gone` 404; `/down` 500. Returns the base URL and per-route hit counters
/// `[gone, down]`.
fn server() -> (String, Arc<[AtomicUsize; 2]>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let hits: Arc<[AtomicUsize; 2]> = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let h = Arc::clone(&hits);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut line = String::new();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            // Drain headers.
            loop {
                let mut hdr = String::new();
                if reader.read_line(&mut hdr).map(|n| n == 0).unwrap_or(true) || hdr == "\r\n" {
                    break;
                }
            }
            let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
            let (status, extra, body) = match path.as_str() {
                "/page" => (
                    "200 OK",
                    String::new(),
                    "<html><h2 id=\"present\">x</h2><a name=\"named\"></a></html>",
                ),
                "/moved" => ("301 Moved", "Location: /page\r\n".to_string(), ""),
                "/gone" => {
                    h[0].fetch_add(1, Ordering::SeqCst);
                    ("404 Not Found", String::new(), "")
                }
                "/down" => {
                    h[1].fetch_add(1, Ordering::SeqCst);
                    ("500 Internal Server Error", String::new(), "")
                }
                _ => ("404 Not Found", String::new(), ""),
            };
            let _ = write!(
                stream,
                "HTTP/1.0 {status}\r\n{extra}Content-Type: text/html\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (base, hits)
}

#[test]
fn the_script_is_executable() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(script()).unwrap().permissions().mode();
        assert!(
            mode & 0o111 != 0,
            "tools/check-external-links.py is not executable"
        );
    }
}

/// The collector takes `https://` URLs from markdown links, autolinks and code, strips
/// trailing sentence punctuation, records every location, and ignores `http://` and the
/// rendered `_site/`.
#[test]
fn list_collects_https_urls_with_locations() {
    let docs = scratch("list");
    std::fs::create_dir_all(docs.join("sub")).unwrap();
    std::fs::create_dir_all(docs.join("_site")).unwrap();
    std::fs::write(
        docs.join("a.qmd"),
        "See [the book](https://example.org/book/ch.html#sec-a).\n\
         Autolink <https://example.org/auto>, and https://example.org/plain.\n\
         ```bash\ncurl -sSf https://example.org/code | sh\n```\n\
         Not this: http://example.org/insecure\n",
    )
    .unwrap();
    std::fs::write(
        docs.join("sub").join("b.qmd"),
        "Again: https://example.org/plain\n",
    )
    .unwrap();
    std::fs::write(
        docs.join("_site").join("c.qmd"),
        "https://example.org/rendered\n",
    )
    .unwrap();
    std::fs::write(docs.join("d.md"), "https://example.org/not-qmd\n").unwrap();

    let out = run(&docs, &["--list"]);
    assert!(out.status.success(), "{}", text(&out));
    let listed = String::from_utf8(out.stdout).unwrap();
    let urls: Vec<&str> = listed
        .lines()
        .map(|l| l.split('\t').next().unwrap())
        .collect();
    assert_eq!(
        urls,
        [
            "https://example.org/auto",
            "https://example.org/book/ch.html#sec-a",
            "https://example.org/code",
            "https://example.org/plain",
        ],
        "full listing:\n{listed}"
    );
    let plain = listed
        .lines()
        .find(|l| l.starts_with("https://example.org/plain\t"))
        .unwrap();
    assert!(
        plain.contains("a.qmd:2") && plain.contains("b.qmd:1"),
        "both locations of a repeated URL must be listed: {plain}"
    );
}

/// The anchor straddle: the same page, one fragment present and one absent. `id=` and
/// `name=` both count, and the check runs on the page a redirect lands on.
#[test]
fn a_present_anchor_passes_and_a_missing_one_fails() {
    let (base, _) = server();
    let docs = scratch("anchor");

    let good = format!("{base}/page {base}/page#present {base}/page#named {base}/moved#present");
    let out = run(&docs, &["--url", &good]);
    assert!(
        out.status.success(),
        "all present must pass:\n{}",
        text(&out)
    );
    assert!(
        text(&out).contains("4 URLs checked, 0 failed"),
        "{}",
        text(&out)
    );

    let bad = format!("{base}/page#present {base}/moved#absent");
    let out = run(&docs, &["--url", &bad]);
    let t = text(&out);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a missing anchor must fail:\n{t}"
    );
    assert!(t.contains(&format!("ok   {base}/page#present")), "{t}");
    assert!(
        t.contains(&format!("FAIL {base}/moved#absent"))
            && t.contains("no id=\"absent\"")
            && t.contains(&format!("redirected to {base}/page")),
        "{t}"
    );
}

/// A 404 fails at once (no retries: it will not get better), a 500 is retried
/// `--retries` times and then fails. The 200 twin in the same run passes.
#[test]
fn a_404_fails_without_retry_and_a_500_after_retries() {
    let (base, hits) = server();
    let docs = scratch("status");
    let urls = format!("{base}/page {base}/gone {base}/down");
    let out = run(&docs, &["--url", &urls]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(1), "{t}");
    assert!(t.contains(&format!("ok   {base}/page")), "{t}");
    assert!(t.contains(&format!("FAIL {base}/gone -- HTTP 404")), "{t}");
    assert!(
        t.contains(&format!("FAIL {base}/down -- HTTP 500 (after 3 attempts)")),
        "{t}"
    );
    assert_eq!(hits[0].load(Ordering::SeqCst), 1, "404 must not be retried");
    assert_eq!(
        hits[1].load(Ordering::SeqCst),
        3,
        "500 must be retried twice"
    );
}

/// The markdown report names each failing URL and where in docs/ it is linked from; a
/// passing URL does not appear in the failure table.
#[test]
fn summary_names_the_failing_link_and_its_location() {
    let (base, _) = server();
    let docs = scratch("summary");
    std::fs::write(docs.join("p.qmd"), "x\n").unwrap();
    let summary = docs.join("report.md");
    let urls = format!("{base}/page {base}/gone");
    let out = run(
        &docs,
        &["--url", &urls, "--summary", summary.to_str().unwrap()],
    );
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let rep = std::fs::read_to_string(&summary).unwrap();
    assert!(rep.contains("**1 passed**, **1 failed**"), "{rep}");
    assert!(
        rep.contains(&format!("| <{base}/gone> | HTTP 404 | `--url` |")),
        "{rep}"
    );
    assert!(!rep.contains(&format!("<{base}/page>")), "{rep}");
}

/// Own-repo issue/PR numbers and Wiley are skipped without a request (an offline run would
/// otherwise fail them), and a skip never counts as a failure. The local-server URL in the
/// same run is still fetched and passes.
#[test]
fn skip_entries_are_reported_and_never_fetched() {
    let (base, _) = server();
    let docs = scratch("skip");
    let urls = format!(
        "https://github.com/FeRx-NLME/ferx-core/issues/1 \
         https://github.com/FeRx-NLME/ferx-r/pull/2#discussion \
         https://ascpt.onlinelibrary.wiley.com/doi/x {base}/page"
    );
    let out = run(&docs, &["--url", &urls, "--timeout", "0.01"]);
    let t = text(&out);
    assert!(out.status.success(), "{t}");
    assert!(
        t.contains("SKIP https://github.com/FeRx-NLME/ferx-core/issues/1 -- skipped: own-repo issue/PR number")
            && t.contains("SKIP https://github.com/FeRx-NLME/ferx-r/pull/2#discussion -- skipped: own-repo")
            && t.contains("SKIP https://ascpt.onlinelibrary.wiley.com/doi/x -- skipped: host onlinelibrary.wiley.com"),
        "{t}"
    );
    assert!(t.contains(&format!("ok   {base}/page")), "{t}");
}

#[test]
fn a_non_http_url_is_a_usage_error() {
    let docs = scratch("usage");
    let out = run(&docs, &["--url", "ftp://example.org/x"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    let out = run(&docs.join("missing"), &["--list"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
}

/// The workflow runs weekly and on dispatch, never on `pull_request` or `push` — a flaky
/// external site must not redden a PR — and it delegates to the script rather than
/// carrying its own copy of the check.
#[test]
fn workflow_is_scheduled_dispatchable_and_off_the_pr_path() {
    let wf =
        std::fs::read_to_string(repo_root().join(".github/workflows/external-links.yml")).unwrap();
    let on_block: Vec<&str> = wf
        .lines()
        .skip_while(|l| *l != "on:")
        .skip(1)
        .take_while(|l| l.is_empty() || l.starts_with(' ') || l.starts_with('#'))
        .collect();
    let triggers: Vec<&str> = on_block
        .iter()
        .filter(|l| l.starts_with("  ") && !l.starts_with("   "))
        .map(|l| l.trim().trim_end_matches(':'))
        .collect();
    assert_eq!(
        triggers,
        ["schedule", "workflow_dispatch"],
        "external-links.yml triggers"
    );
    assert!(
        on_block
            .iter()
            .any(|l| l.trim_start().starts_with("- cron:")),
        "no cron line under schedule"
    );
    assert!(
        wf.contains("tools/check-external-links.py"),
        "workflow must run the script"
    );
}
