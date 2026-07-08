/// Per-iteration optimizer trace for diagnostic observability.
///
/// A thread-local TraceWriter is initialised by `api::fit_inner` when
/// `FitOptions::optimizer_trace = true` and is written to by each estimator
/// (NLopt, BFGS, GN, SAEM) as it iterates.  The outer fit collects the file
/// path and stores it in `FitResult::trace_path`.
///
/// The thread-local design means no signature changes are needed in the inner
/// estimator functions.  All outer-loop code is single-threaded (rayon is used
/// only for per-subject inner-loop parallelism), so there are no data races.
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

// ─── writer ─────────────────────────────────────────────────────────────────

pub struct TraceWriter {
    pub path: String,
    writer: BufWriter<File>,
    start: Instant,
    /// Number of optimized coordinates. Each row carries `n_coords` `val:*`
    /// columns (natural scale) followed by `n_coords` `grad:*` columns (scaled
    /// space), appended after the fixed 17 columns. See #640.
    n_coords: usize,
}

impl TraceWriter {
    /// `coord_names` are the declared parameter names (one per optimized
    /// coordinate, in packed order). The header appends `val:<name>` then
    /// `grad:<name>` columns for each; all row writers emit the same set.
    fn new(path: String, coord_names: &[String]) -> std::io::Result<Self> {
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        let mut header = String::from(
            "iter,method,phase,ofv,wall_ms,grad_norm,step_norm,inner_iter_count,\
             optimizer,lm_lambda,ofv_delta,step_accepted,cond_nll,gamma,mh_accept_rate,\
             n_ebe_unconverged,n_ebe_fallback",
        );
        for name in coord_names {
            header.push(',');
            header.push_str(&csv_field(&format!("val:{}", name)));
        }
        for name in coord_names {
            header.push(',');
            header.push_str(&csv_field(&format!("grad:{}", name)));
        }
        writeln!(writer, "{}", header)?;
        Ok(Self {
            path,
            writer,
            start: Instant::now(),
            n_coords: coord_names.len(),
        })
    }

    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Append the per-parameter `val:*` and `grad:*` columns (no leading/trailing
    /// newline). `grads = None` (SAEM, or a derivative-free eval) writes `NA` for
    /// every gradient column. Missing/non-finite entries serialise as `NA`.
    ///
    /// These use a scientific format (`fmt_opt_sci`), not the fixed `{:.6}` of
    /// the scalar columns: per-parameter estimates (variances down to ~1e-8) and
    /// near-converged gradient coordinates span many orders of magnitude, and a
    /// fixed six-decimal format would round them to `0.000000`, erasing exactly
    /// the signal the per-parameter view exists to show (#640 review).
    fn write_param_cols(&mut self, values: &[f64], grads: Option<&[f64]>) {
        for i in 0..self.n_coords {
            let _ = write!(self.writer, ",{}", fmt_opt_sci(values.get(i).copied()));
        }
        for i in 0..self.n_coords {
            let _ = write!(
                self.writer,
                ",{}",
                fmt_opt_sci(grads.and_then(|g| g.get(i).copied()))
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_foce_row(
        &mut self,
        iter: usize,
        method: &str,
        phase: &str,
        ofv: f64,
        grad_norm: Option<f64>,
        step_norm: Option<f64>,
        optimizer: &str,
        n_ebe_unconverged: Option<usize>,
        n_ebe_fallback: Option<usize>,
        values: &[f64],
        grads: Option<&[f64]>,
    ) {
        let wall_ms = self.elapsed_ms();
        let _ = write!(
            self.writer,
            "{},{},{},{:.6},{},{},{},NA,{},NA,NA,NA,NA,NA,NA,{},{}",
            iter,
            method,
            phase,
            ofv,
            wall_ms,
            fmt_opt(grad_norm),
            fmt_opt(step_norm),
            optimizer,
            fmt_opt_usize(n_ebe_unconverged),
            fmt_opt_usize(n_ebe_fallback),
        );
        self.write_param_cols(values, grads);
        let _ = writeln!(self.writer);
        // Flush each row so live consumers (e.g. the trace UI) see iterations
        // as they happen. Gradient methods emit few rows (< the BufWriter's
        // buffer), so without this the file would not appear until finish().
        let _ = self.writer.flush();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_gn_row(
        &mut self,
        iter: usize,
        method: &str,
        phase: &str,
        ofv: f64,
        grad_norm: Option<f64>,
        lm_lambda: f64,
        ofv_delta: f64,
        step_accepted: bool,
        n_ebe_unconverged: Option<usize>,
        n_ebe_fallback: Option<usize>,
        values: &[f64],
        grads: Option<&[f64]>,
    ) {
        let wall_ms = self.elapsed_ms();
        // grad_norm (position 6) is now populated for GN: the scaled BHHH
        // gradient is available, so `sqrt(sum(grad:*^2)) == grad_norm` holds for
        // GN rows too, matching the FOCE writers (#640 review).
        let _ = write!(
            self.writer,
            "{},{},{},{:.6},{},{},NA,NA,NA,{:.6},{:.6},{},NA,NA,NA,{},{}",
            iter,
            method,
            phase,
            ofv,
            wall_ms,
            fmt_opt(grad_norm),
            lm_lambda,
            ofv_delta,
            i32::from(step_accepted),
            fmt_opt_usize(n_ebe_unconverged),
            fmt_opt_usize(n_ebe_fallback),
        );
        self.write_param_cols(values, grads);
        let _ = writeln!(self.writer);
        let _ = self.writer.flush();
    }

    pub fn write_saem_row(
        &mut self,
        iter: usize,
        phase: &str,
        cond_nll: f64,
        gamma: f64,
        mh_accept_rate: f64,
        values: &[f64],
    ) {
        let wall_ms = self.elapsed_ms();
        // During SAEM iterations the FOCE OFV is not yet available;
        // use condNLL as the ofv-column proxy (documented in PR1).
        let _ = write!(
            self.writer,
            "{},saem,{},{:.6},{},NA,NA,NA,NA,NA,NA,NA,{:.6},{:.6},{:.4},NA,NA",
            iter, phase, cond_nll, wall_ms, cond_nll, gamma, mh_accept_rate
        );
        // SAEM is derivative-free w.r.t. the OFV → every gradient column is NA.
        self.write_param_cols(values, None);
        let _ = writeln!(self.writer);
        let _ = self.writer.flush();
    }

    pub fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(f) if f.is_finite() => format!("{:.6}", f),
        _ => "NA".to_string(),
    }
}

/// Scientific-notation variant of [`fmt_opt`] for the per-parameter `val:*` /
/// `grad:*` columns, which span many orders of magnitude. `1.5e-8` survives
/// where `{:.6}` would print `0.000000`. Ten significant figures keep large
/// gradient components (O(1e6)) precise enough that `sqrt(sum(grad:*^2))`
/// reconstructs `grad_norm`. Non-finite → `NA`.
fn fmt_opt_sci(v: Option<f64>) -> String {
    match v {
        Some(f) if f.is_finite() => format!("{:.9e}", f),
        _ => "NA".to_string(),
    }
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "NA".to_string(),
    }
}

/// Minimal CSV field quoting for header names. Parameter names can contain a
/// comma (the `OMEGA(2,1)` fallback), which would otherwise shift columns.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ─── thread-local state ──────────────────────────────────────────────────────

#[derive(Default)]
struct TraceState {
    writer: Option<TraceWriter>,
    /// Overrides the `method` column written by estimator code.
    /// Set by `run_foce_gn` before calling the FOCEI polish so that
    /// all polish rows say "gn_hybrid" rather than "foce".
    method_override: Option<&'static str>,
    /// Overrides the `phase` column.  Set to "focei" for the GN-hybrid
    /// polish phase and to "gn" / "" by the GN loop itself.
    phase_override: Option<&'static str>,
}

thread_local! {
    static TRACE: RefCell<TraceState> = RefCell::new(TraceState::default());
}

// ─── public API ─────────────────────────────────────────────────────────────

/// Initialise the trace for this fit.  The CSV file is created immediately and
/// its header row is written.  Called once per `fit_inner` invocation.
pub fn init(path: String, coord_names: &[String]) -> std::io::Result<()> {
    let writer = TraceWriter::new(path, coord_names)?;
    TRACE.with(|t| {
        let mut s = t.borrow_mut();
        s.writer = Some(writer);
        s.method_override = None;
        s.phase_override = None;
    });
    Ok(())
}

/// Override the method/phase columns for subsequent `write_*` calls.
/// Pass `None` to clear an override.
pub fn set_overrides(method: Option<&'static str>, phase: Option<&'static str>) {
    TRACE.with(|t| {
        let mut s = t.borrow_mut();
        s.method_override = method;
        s.phase_override = phase;
    });
}

/// Returns `true` when a trace file is open for this thread.
pub fn is_active() -> bool {
    TRACE.with(|t| t.borrow().writer.is_some())
}

/// Flush and close the trace file.  Returns the path so `fit_inner` can
/// store it in `FitResult::trace_path`.
pub fn finish() -> Option<String> {
    TRACE.with(|t| {
        let mut s = t.borrow_mut();
        if let Some(ref mut w) = s.writer {
            w.flush();
        }
        let path = s.writer.take().map(|w| w.path);
        s.method_override = None;
        s.phase_override = None;
        path
    })
}

/// Write one FOCE/FOCEI trace row.
///
/// `method`           — "foce" or "focei" (may be overridden by `set_overrides`)
/// `ofv`              — current OFV
/// `grad_norm`        — L2 norm of the gradient (None for derivative-free optimizers)
/// `step_norm`        — L2 norm of the parameter step (None when unavailable)
/// `optimizer`        — optimizer name string, e.g. "slsqp", "bobyqa", "bfgs"
/// `n_ebe_unconverged`— subjects that did not meet EBE tolerance (None = unavailable)
/// `n_ebe_fallback`   — subjects that used Nelder-Mead fallback (None = unavailable)
///
/// `values`           — per-coordinate estimates, natural scale (packed order)
/// `grads`            — per-coordinate gradient, scaled space (None for
///                      derivative-free optimizers; matches `grad_norm`)
#[allow(clippy::too_many_arguments)]
pub fn write_foce(
    iter: usize,
    method: &str,
    ofv: f64,
    grad_norm: Option<f64>,
    step_norm: Option<f64>,
    optimizer: &str,
    n_ebe_unconverged: Option<usize>,
    n_ebe_fallback: Option<usize>,
    values: &[f64],
    grads: Option<&[f64]>,
) {
    TRACE.with(|t| {
        let mut s = t.borrow_mut();
        let method = s.method_override.unwrap_or(method);
        let phase = s.phase_override.unwrap_or("");
        if let Some(ref mut w) = s.writer {
            w.write_foce_row(
                iter,
                method,
                phase,
                ofv,
                grad_norm,
                step_norm,
                optimizer,
                n_ebe_unconverged,
                n_ebe_fallback,
                values,
                grads,
            );
        }
    });
}

/// Write one GN/GN-hybrid trace row.
///
/// `method`           — "gn" or "gn_hybrid"
/// `phase`            — "" for pure GN, "gn" for GN phase of hybrid
/// `n_ebe_unconverged`— subjects that did not meet EBE tolerance (None = unavailable)
/// `n_ebe_fallback`   — subjects that used Nelder-Mead fallback (None = unavailable)
#[allow(clippy::too_many_arguments)]
pub fn write_gn(
    iter: usize,
    method: &str,
    phase: &str,
    ofv: f64,
    grad_norm: Option<f64>,
    lm_lambda: f64,
    ofv_delta: f64,
    step_accepted: bool,
    n_ebe_unconverged: Option<usize>,
    n_ebe_fallback: Option<usize>,
    values: &[f64],
    grads: Option<&[f64]>,
) {
    TRACE.with(|t| {
        let mut s = t.borrow_mut();
        let method = s.method_override.unwrap_or(method);
        // GN rows always carry the caller-supplied phase; phase_override is
        // reserved for the FOCEI polish phase and doesn't apply here.
        if let Some(ref mut w) = s.writer {
            w.write_gn_row(
                iter,
                method,
                phase,
                ofv,
                grad_norm,
                lm_lambda,
                ofv_delta,
                step_accepted,
                n_ebe_unconverged,
                n_ebe_fallback,
                values,
                grads,
            );
        }
    });
}

/// Write one SAEM trace row. `values` are per-coordinate estimates (natural
/// scale); SAEM has no OFV gradient so every `grad:*` column is written `NA`.
pub fn write_saem(
    iter: usize,
    phase: &str,
    cond_nll: f64,
    gamma: f64,
    mh_accept_rate: f64,
    values: &[f64],
) {
    TRACE.with(|t| {
        let mut s = t.borrow_mut();
        if let Some(ref mut w) = s.writer {
            w.write_saem_row(iter, phase, cond_nll, gamma, mh_accept_rate, values);
        }
    });
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn read_file(path: &str) -> String {
        let mut f = File::open(path).unwrap();
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn test_header_written() {
        let path = format!("/tmp/ferx_trace_hdr_{}.csv", std::process::id());
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.flush();
        let contents = read_file(&path);
        assert!(contents.starts_with("iter,method,phase,ofv,wall_ms,grad_norm"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_foce_row_format() {
        let path = format!("/tmp/ferx_trace_foce_{}.csv", std::process::id());
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.write_foce_row(
            1,
            "foce",
            "",
            100.5,
            Some(0.25),
            Some(0.01),
            "slsqp",
            None,
            None,
            &[],
            None,
        );
        w.flush();
        let contents = read_file(&path);
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("1,foce,,100.5"));
        assert!(lines[1].contains(",0.250000,0.010000,NA,slsqp,"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_na_for_missing_grad() {
        let path = format!("/tmp/ferx_trace_na_{}.csv", std::process::id());
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.write_foce_row(
            1,
            "focei",
            "",
            99.0,
            None,
            None,
            "bobyqa",
            None,
            None,
            &[],
            None,
        );
        w.flush();
        let contents = read_file(&path);
        // grad_norm and step_norm should be NA
        let row = contents.lines().nth(1).unwrap();
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols[5], "NA"); // grad_norm
        assert_eq!(cols[6], "NA"); // step_norm
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gn_row_format() {
        let path = format!("/tmp/ferx_trace_gn_{}.csv", std::process::id());
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.write_gn_row(
            3,
            "gn",
            "",
            200.0,
            Some(0.5),
            0.01,
            -5.0,
            true,
            None,
            None,
            &[],
            None,
        );
        w.flush();
        let contents = read_file(&path);
        let row = contents.lines().nth(1).unwrap();
        assert!(row.starts_with("3,gn,,200."));
        // grad_norm now populated for GN (position 6), then lm_lambda, ofv_delta.
        assert!(row.contains(",0.500000,NA,NA,NA,0.010000,-5.000000,1,"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_saem_row_format() {
        let path = format!("/tmp/ferx_trace_saem_{}.csv", std::process::id());
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.write_saem_row(10, "explore", 55.3, 1.0, 0.35, &[]);
        w.flush();
        let contents = read_file(&path);
        let row = contents.lines().nth(1).unwrap();
        assert!(row.starts_with("10,saem,explore,55.3"));
        assert!(row.contains(",1.000000,0.3500"));
        std::fs::remove_file(&path).ok();
    }

    // Per-test path that won't collide with other tests sharing the same
    // thread-local TRACE on parallel cargo test runs.  cargo test runs each
    // #[test] on its own thread, so TLS is isolated, but file paths still
    // need to be unique on disk.
    fn unique_path(tag: &str) -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!(
            "/tmp/ferx_trace_{}_{}_{}.csv",
            tag,
            std::process::id(),
            nanos
        )
    }

    #[test]
    fn test_is_active_default_false() {
        // Fresh thread => no writer initialised => not active.
        assert!(!is_active());
    }

    #[test]
    fn test_init_then_finish_lifecycle() {
        let path = unique_path("lifecycle");
        assert!(!is_active());

        init(path.clone(), &[]).unwrap();
        assert!(is_active(), "is_active should be true after init");

        let returned = finish().expect("finish should return the path");
        assert_eq!(returned, path);
        assert!(!is_active(), "is_active should be false after finish");
        assert!(
            std::path::Path::new(&path).exists(),
            "trace file should still exist on disk after finish"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_finish_without_init_returns_none() {
        // No init => finish is a no-op that returns None.
        assert!(finish().is_none());
    }

    #[test]
    fn test_write_foce_via_thread_local() {
        let path = unique_path("tl_foce");
        init(path.clone(), &[]).unwrap();
        write_foce(
            7,
            "focei",
            42.5,
            Some(0.125),
            None,
            "bobyqa",
            None,
            None,
            &[],
            None,
        );
        finish();

        let contents = read_file(&path);
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "header + one data row");
        assert!(lines[1].starts_with("7,focei,,42.5"));
        assert!(lines[1].contains(",0.125000,NA,NA,bobyqa,"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_when_no_writer_is_noop() {
        // Calling write_* with no active trace must not panic and must not
        // create any files.  This is the contract estimators rely on so they
        // can call trace::write_* unconditionally.
        assert!(!is_active());
        write_foce(1, "foce", 1.0, None, None, "slsqp", None, None, &[], None);
        write_gn(
            1,
            "gn",
            "",
            1.0,
            None,
            0.0,
            0.0,
            true,
            None,
            None,
            &[],
            None,
        );
        write_saem(1, "explore", 1.0, 1.0, 0.5, &[]);
        // No assertion on files — the assertion is "didn't panic".
    }

    #[test]
    fn test_method_override_applied() {
        let path = unique_path("override");
        init(path.clone(), &[]).unwrap();
        // Caller passes "foce" but override forces "gn_hybrid" + phase "focei".
        set_overrides(Some("gn_hybrid"), Some("focei"));
        write_foce(
            2,
            "foce",
            10.0,
            Some(0.1),
            Some(0.01),
            "slsqp",
            None,
            None,
            &[],
            None,
        );
        set_overrides(None, None);
        // After clearing, caller-supplied method/phase apply.
        write_foce(
            3,
            "foce",
            9.0,
            Some(0.05),
            Some(0.005),
            "slsqp",
            None,
            None,
            &[],
            None,
        );
        finish();

        let contents = read_file(&path);
        let rows: Vec<&str> = contents.lines().skip(1).collect();
        assert_eq!(rows.len(), 2);

        let cols0: Vec<&str> = rows[0].split(',').collect();
        assert_eq!(cols0[1], "gn_hybrid", "method overridden");
        assert_eq!(cols0[2], "focei", "phase overridden");

        let cols1: Vec<&str> = rows[1].split(',').collect();
        assert_eq!(cols1[1], "foce", "method override cleared");
        assert_eq!(cols1[2], "", "phase override cleared");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_init_overwrites_previous_writer() {
        // Calling init twice in a row (e.g. back-to-back fits on the same
        // worker thread) should drop the first writer and start fresh.
        let path1 = unique_path("init1");
        let path2 = unique_path("init2");
        init(path1.clone(), &[]).unwrap();
        write_foce(1, "foce", 1.0, None, None, "slsqp", None, None, &[], None);
        init(path2.clone(), &[]).unwrap();
        write_foce(1, "foce", 2.0, None, None, "slsqp", None, None, &[], None);
        let returned = finish().unwrap();
        assert_eq!(returned, path2, "finish returns the most recent path");

        // Second file has just the one row written after re-init.
        let c2 = read_file(&path2);
        assert_eq!(c2.lines().count(), 2);
        std::fs::remove_file(&path1).ok();
        std::fs::remove_file(&path2).ok();
    }

    #[test]
    fn test_init_fails_for_unwritable_path() {
        // A path under a non-existent directory should fail at File::create
        // and leave the writer uninitialised.
        let bad = "/definitely/not/a/real/dir/ferx_trace.csv";
        let err = init(bad.to_string(), &[]);
        assert!(err.is_err());
        assert!(!is_active(), "failed init must leave writer uninitialised");
    }

    #[test]
    fn test_na_for_non_finite_grad() {
        // NaN/Inf gradients should be serialised as "NA", not "NaN"/"inf".
        let path = unique_path("nonfinite");
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.write_foce_row(
            1,
            "foce",
            "",
            1.0,
            Some(f64::NAN),
            Some(f64::INFINITY),
            "bfgs",
            None,
            None,
            &[],
            None,
        );
        w.flush();
        let row = read_file(&path).lines().nth(1).unwrap().to_string();
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols[5], "NA", "NaN grad_norm => NA");
        assert_eq!(cols[6], "NA", "Inf step_norm => NA");
        std::fs::remove_file(&path).ok();
    }

    // ── per-parameter columns (#640) ────────────────────────────────────────

    #[test]
    fn test_header_appends_val_and_grad_columns() {
        let path = unique_path("hdr_param");
        let names = vec!["TVCL".to_string(), "ETA_CL".to_string()];
        let w = TraceWriter::new(path.clone(), &names).unwrap();
        drop(w);
        let contents = read_file(&path);
        let header = contents.lines().next().unwrap();
        // Fixed columns, then all val:* columns, then all grad:* columns.
        assert!(header.ends_with("val:TVCL,val:ETA_CL,grad:TVCL,grad:ETA_CL"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_header_quotes_names_with_commas() {
        // The OMEGA(2,1) fallback contains a comma → must be CSV-quoted so it
        // stays one column.
        let path = unique_path("hdr_quote");
        let names = vec!["OMEGA(2,1)".to_string()];
        let w = TraceWriter::new(path.clone(), &names).unwrap();
        drop(w);
        let header = read_file(&path).lines().next().unwrap().to_string();
        assert!(header.contains("\"val:OMEGA(2,1)\""));
        assert!(header.contains("\"grad:OMEGA(2,1)\""));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_foce_row_writes_values_and_grads() {
        let path = unique_path("foce_param");
        let names = vec!["TVCL".to_string(), "ETA_CL".to_string()];
        let mut w = TraceWriter::new(path.clone(), &names).unwrap();
        w.write_foce_row(
            1,
            "foce",
            "",
            10.0,
            Some(0.5),
            Some(0.1),
            "slsqp",
            None,
            None,
            &[2.5, 0.09],
            Some(&[0.4, -0.3]),
        );
        w.flush();
        let row = read_file(&path).lines().nth(1).unwrap().to_string();
        // 17 fixed + 2 val + 2 grad = 21 columns.
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols.len(), 21);
        assert_eq!(
            &cols[17..],
            &[
                "2.500000000e0",
                "9.000000000e-2",
                "4.000000000e-1",
                "-3.000000000e-1"
            ]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_foce_row_grads_none_writes_na() {
        // Derivative-free eval: values present, every gradient column NA.
        let path = unique_path("foce_nograd");
        let names = vec!["TVCL".to_string(), "ETA_CL".to_string()];
        let mut w = TraceWriter::new(path.clone(), &names).unwrap();
        w.write_foce_row(
            1,
            "focei",
            "",
            9.0,
            None,
            None,
            "bobyqa",
            None,
            None,
            &[2.5, 0.09],
            None,
        );
        w.flush();
        let cols: Vec<String> = read_file(&path)
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .map(String::from)
            .collect();
        assert_eq!(
            &cols[17..],
            &["2.500000000e0", "9.000000000e-2", "NA", "NA"]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_saem_row_values_present_grads_na() {
        let path = unique_path("saem_param");
        let names = vec!["TVCL".to_string(), "ETA_CL".to_string()];
        let mut w = TraceWriter::new(path.clone(), &names).unwrap();
        w.write_saem_row(3, "explore", 55.0, 1.0, 0.4, &[2.5, 0.09]);
        w.flush();
        let cols: Vec<String> = read_file(&path)
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .map(String::from)
            .collect();
        assert_eq!(
            &cols[17..],
            &["2.500000000e0", "9.000000000e-2", "NA", "NA"]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_grad_columns_reconstruct_grad_norm() {
        // Invariant from #640: sqrt(Σ gᵢ²) == grad_norm. Feed a gradient whose
        // components match the grad_norm column and confirm the round-trip.
        let path = unique_path("grad_recon");
        let names = vec!["A".to_string(), "B".to_string()];
        let mut w = TraceWriter::new(path.clone(), &names).unwrap();
        let g = [0.3_f64, 0.4];
        let gn = (g[0] * g[0] + g[1] * g[1]).sqrt(); // 0.5
        w.write_foce_row(
            1,
            "foce",
            "",
            10.0,
            Some(gn),
            None,
            "slsqp",
            None,
            None,
            &[1.0, 2.0],
            Some(&g),
        );
        w.flush();
        let cols: Vec<f64> = read_file(&path)
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .enumerate()
            .filter_map(|(i, s)| if i >= 19 { s.parse().ok() } else { None })
            .collect();
        let recon = (cols[0] * cols[0] + cols[1] * cols[1]).sqrt();
        assert!((recon - gn).abs() < 1e-6);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_missing_value_entries_padded_with_na() {
        // Defensive: fewer values than n_coords → remaining columns NA rather
        // than shifting the CSV.
        let path = unique_path("short_vals");
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut w = TraceWriter::new(path.clone(), &names).unwrap();
        w.write_saem_row(1, "explore", 1.0, 1.0, 0.5, &[1.0]);
        w.flush();
        let cols: Vec<String> = read_file(&path)
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .map(String::from)
            .collect();
        // 17 fixed + 3 val + 3 grad = 23.
        assert_eq!(cols.len(), 23);
        assert_eq!(&cols[17..20], &["1.000000000e0", "NA", "NA"]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gn_step_rejected_serialised_as_zero() {
        let path = unique_path("gn_reject");
        let mut w = TraceWriter::new(path.clone(), &[]).unwrap();
        w.write_gn_row(
            5,
            "gn",
            "",
            100.0,
            None,
            1.0,
            2.0,
            false,
            None,
            None,
            &[],
            None,
        );
        w.flush();
        let row = read_file(&path).lines().nth(1).unwrap().to_string();
        // step_accepted is the column right before cond_nll's NA stretch.
        assert!(row.contains(",1.000000,2.000000,0,"));
        std::fs::remove_file(&path).ok();
    }
}
