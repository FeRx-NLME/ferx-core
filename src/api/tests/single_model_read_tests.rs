//! #1752: every entry point that parses a model file reads it **once**, and the
//! parse, the binding to the data, `model_hash` and `model_text` all come from
//! that one read.
//!
//! The model file is a FIFO served by a thread that hands out text A on the first
//! open and an edited text B on every later one, counting the opens. An entry point
//! that read the file again — to bind levels, to hash it, to store its text, to
//! generate a FREM model — opens it a second time and sees B. Each entry point has
//! its own test, so a failure names the site.
//!
//! Mutation, per test: restore any one of that entry point's re-reads (`fit.rs`
//! bind / hash / text; `run.rs` bind / hash / text for `prepare_run`,
//! `run_model_with_overrides` and the simulate path; the check's bind; the FREM
//! generation read). The open count goes to 2.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Text A: fast to fit (`maxiter = 0`, no covariance) and with no checkpoint, so
/// `run_model_with_overrides` writes nothing into the working directory. `WT` is
/// declared for the FREM conversion, which folds declared covariates.
const MODEL_A: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[covariates]
  WT continuous

[fit_options]
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false

[simulation]
  n_subjects = 3
  dose_amt   = 100.0
  dose_cmt   = 1
  times      = [1.0, 4.0, 12.0]
  seed       = 1
"#;

/// Text B: A with `KA = TVKA` rewritten as `KA = TVKA * 1.0` — the same model,
/// different bytes, so a second read shows up in the hash and the stored text as
/// well as in the count, and in the FREM model, which copies the
/// `[individual_parameters]` lines from the text it is given.
fn model_b() -> String {
    let b = MODEL_A.replace("  KA = TVKA\n", "  KA = TVKA * 1.0\n");
    assert_ne!(b, MODEL_A);
    b
}

fn data(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}

/// `O_NONBLOCK`, which `std` does not export. A non-blocking write-open of a FIFO
/// fails (`ENXIO`) when no reader holds it open, and a non-blocking read-open
/// succeeds whether or not a writer does.
#[cfg(target_os = "linux")]
const O_NONBLOCK: i32 = 0o4000;
#[cfg(target_os = "macos")]
const O_NONBLOCK: i32 = 0x0004;

fn open_nonblocking(path: &Path, write: bool) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(!write)
        .write(write)
        .custom_flags(O_NONBLOCK)
        .open(path)
}

/// A model-file FIFO that serves A on its first open and B on any later one.
///
/// After serving a read the server closes its end, so the reader sees EOF, and
/// then waits for that reader to **close** before it opens for the next one: a
/// write-open succeeds at once while any reader holds the pipe, and writing then
/// would append the next text to the stream the first reader is still reading.
/// A reader's close shows as `ENXIO` on a non-blocking write-open (a probe that
/// writes nothing, so a reader still draining sees only EOF).
///
/// Known blind spot (#1760 review r1, finding 2): a second open that lands
/// *inside* one probe — within about a millisecond of the first reader's close —
/// connects to the probe, reads an empty file, and is not counted. A probe cannot
/// tell that reader from the first one still closing. Such a re-read is still
/// caught when its text is hashed or stored (`sha(A)` / `model_text == A` fail
/// on `""`), and FREM generation fails on `""`. Only a re-read whose text the
/// binders ignore — `MODEL_A` has no level block or symbolic statistic — would
/// pass, and only if it came within that millisecond. Every entry point parses
/// the text and builds its population (or simulation design) between the first
/// read and any bind, which takes longer than that.
/// Each re-read mutation in the PR was killed with a count of 2.
struct ModelFifo {
    path: PathBuf,
    opens: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    server: Option<std::thread::JoinHandle<()>>,
    _dir: tempfile::TempDir,
}

impl ModelFifo {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.ferx");
        let made = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .expect("run mkfifo");
        assert!(made.success(), "mkfifo");
        let opens = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let server = {
            let (path, opens, stop) = (path.clone(), opens.clone(), stop.clone());
            let b = model_b();
            std::thread::spawn(move || loop {
                // Blocks until a reader opens the file.
                let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                let n = opens.fetch_add(1, Ordering::SeqCst) + 1;
                // A reader that stops early closes the pipe: not this test's concern.
                let _ = f.write_all(if n == 1 {
                    MODEL_A.as_bytes()
                } else {
                    b.as_bytes()
                });
                drop(f);
                // Wait for the reader to close before serving the next open.
                while open_nonblocking(&path, true).is_ok() {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            })
        };
        Self {
            path,
            opens,
            stop,
            server: Some(server),
            _dir: dir,
        }
    }

    fn path(&self) -> &str {
        self.path.to_str().unwrap()
    }

    /// Stop the server and return how many times the file was opened. The server
    /// is blocked in a write-open, waiting for a reader, or probing for the last
    /// reader's close; a non-blocking read-open releases the first, and the stop
    /// flag ends the second.
    fn finish(mut self) -> usize {
        self.stop.store(true, Ordering::SeqCst);
        let server = self.server.take().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !server.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "the FIFO server did not stop"
            );
            drop(open_nonblocking(&self.path, false));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        server.join().unwrap();
        self.opens.load(Ordering::SeqCst)
    }
}

fn sha_a() -> String {
    crate::io::hash::sha256_bytes(MODEL_A.as_bytes())
}

/// The straddle: a second open really is served B, so a hash or text taken from a
/// second read would differ from A's, and the count sees it. The pause between the
/// reads stands in for the work an entry point does between its parse and a
/// re-read (reading the data): the server probes for the first reader's close
/// every millisecond, and an open that lands inside a probe would read empty.
#[test]
fn the_fixture_serves_a_different_text_on_a_second_open() {
    let fifo = ModelFifo::new();
    let first = std::fs::read_to_string(fifo.path()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let second = std::fs::read_to_string(fifo.path()).unwrap();
    assert_eq!(fifo.finish(), 2);
    assert_eq!(first, MODEL_A);
    assert_eq!(second, model_b());
    assert_ne!(crate::io::hash::sha256_bytes(second.as_bytes()), sha_a());
}

#[test]
fn fit_from_files_reads_the_model_file_once() {
    let fifo = ModelFifo::new();
    let opts = crate::types::FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    };
    let r = crate::api::fit_from_files(
        fifo.path(),
        Some(&data("two_cpt_oral_cov.csv")),
        None,
        Some(opts),
    );
    let opens = fifo.finish();
    let r = r.expect("fit_from_files");
    assert_eq!(
        opens, 1,
        "fit_from_files opened the model file {opens} times"
    );
    assert_eq!(r.model_hash.as_deref(), Some(sha_a().as_str()));
    assert_eq!(r.model_text.as_deref(), Some(MODEL_A));
}

#[test]
fn prepare_run_reads_the_model_file_once() {
    let fifo = ModelFifo::new();
    let r = crate::api::prepare_run(fifo.path(), Some(&data("two_cpt_oral_cov.csv")));
    let opens = fifo.finish();
    let r = r.expect("prepare_run");
    assert_eq!(opens, 1, "prepare_run opened the model file {opens} times");
    assert_eq!(r.model_hash.as_deref(), Some(sha_a().as_str()));
    assert_eq!(r.model_text, MODEL_A);
}

#[test]
fn run_model_with_overrides_reads_the_model_file_once() {
    let fifo = ModelFifo::new();
    let r = crate::api::run_model_with_overrides(
        fifo.path(),
        Some(&data("two_cpt_oral_cov.csv")),
        &crate::api::RunOverrides::default(),
    );
    let opens = fifo.finish();
    let (r, _) = r.expect("run_model_with_overrides");
    assert_eq!(
        opens, 1,
        "run_model_with_overrides opened the model file {opens} times"
    );
    assert_eq!(r.model_hash.as_deref(), Some(sha_a().as_str()));
    assert_eq!(r.model_text.as_deref(), Some(MODEL_A));
}

#[test]
fn the_simulate_entry_point_reads_the_model_file_once() {
    let fifo = ModelFifo::new();
    let r = crate::api::run_model_simulate_with_overrides(
        fifo.path(),
        &crate::api::RunOverrides::default(),
    );
    let opens = fifo.finish();
    let (r, _) = r.expect("run_model_simulate_with_overrides");
    assert_eq!(
        opens, 1,
        "the simulate path opened the model file {opens} times"
    );
    assert_eq!(r.model_hash.as_deref(), Some(sha_a().as_str()));
    assert_eq!(r.model_text.as_deref(), Some(MODEL_A));
}

#[test]
fn the_check_reads_the_model_file_once() {
    let fifo = ModelFifo::new();
    let report = crate::api::validate_model_file(fifo.path(), Some(&data("two_cpt_oral_cov.csv")));
    let opens = fifo.finish();
    assert_eq!(opens, 1, "the check opened the model file {opens} times");
    assert!(
        report
            .diagnostics
            .iter()
            .all(|d| d.severity != crate::diagnostics::Severity::Error),
        "{:?}",
        report.diagnostics
    );
}

#[test]
fn prepare_frem_reads_the_model_file_once() {
    let fifo = ModelFifo::new();
    let out = tempfile::tempdir().unwrap();
    let (out_model, out_data) = (out.path().join("f.ferx"), out.path().join("f.csv"));
    let r = crate::frem::prepare_frem(
        &fifo.path,
        Path::new(&data("two_cpt_oral_cov.csv")),
        &["WT".to_string()],
        None,
        Some(&out_model),
        Some(&out_data),
        None,
        None,
    );
    let opens = fifo.finish();
    r.expect("prepare_frem");
    assert_eq!(opens, 1, "prepare_frem opened the model file {opens} times");
    // The generated model is built from A's text, not B's.
    let generated = std::fs::read_to_string(&out_model).unwrap();
    assert!(generated.contains("KA = TVKA\n"), "{generated}");
    assert!(!generated.contains("TVKA * 1.0"), "{generated}");
}
