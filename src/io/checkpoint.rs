//! Periodic checkpoint / restart for interrupted fits (#755).
//!
//! A running fit writes a small JSON `.tmp` file holding the current population
//! estimates and where it is in the method chain, no more often than a
//! configurable interval. If the process is killed, the next run on the same
//! model + data resumes from that point instead of starting over; on successful
//! completion the file is removed. Because the write is throttled, a run that
//! finishes before the first interval elapses never leaves a `.tmp` behind
//! (short runs are cheap to re-run, so they are not checkpointed).
//!
//! **Fidelity is coarse.** Only the packed population-parameter vector
//! (log-theta / Cholesky-omega / log-sigma, reconstructed on resume via
//! [`crate::estimation::parameterization::unpack_params`]) plus the stage index,
//! iteration and OFV are stored. Resume re-seeds the fit from those estimates
//! and continues — it is not a bit-exact optimizer-state restore. NLopt (the
//! FOCE/FOCEI driver) owns its internal trust-region / quasi-Newton state and
//! does not expose it, so a bit-exact resume is impossible there anyway; a
//! coarse warm-restart is the one scheme that works uniformly across every
//! estimation method. Since the stored estimates sit near the optimum,
//! re-convergence is fast.
//!
//! Like [`crate::estimation::trace`], the writer lives in a thread-local: the
//! outer fit loop is single-threaded (rayon parallelism is confined to the
//! per-subject inner loop), so there are no data races and the estimators need
//! no signature changes to feed it.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::time::Instant;

/// On-disk schema version. Bump on any incompatible change to [`Checkpoint`];
/// [`load`] rejects a checkpoint whose version this build doesn't understand.
pub const SCHEMA_VERSION: u32 = 1;

// ─── NaN-safe f64 serde (JSON has no NaN/±Inf) ───────────────────────────────
// Mirrors the approach in `io/fitrx.rs`: non-finite serialises to `null` and
// `null` deserialises back to NaN, so the round-trip never fails on a value
// that is legitimately NaN early in a fit (e.g. the OFV before the first eval).

mod nan_safe_f64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_finite() {
            s.serialize_f64(*v)
        } else {
            s.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

mod nan_safe_vec {
    use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec<f64>, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for x in v {
            if x.is_finite() {
                seq.serialize_element(x)?;
            } else {
                seq.serialize_element(&Option::<f64>::None)?;
            }
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<Option<f64>>::deserialize(d)?
            .into_iter()
            .map(|o| o.unwrap_or(f64::NAN))
            .collect())
    }
}

/// The on-disk checkpoint payload (serialised as pretty JSON).
///
/// `packed` is the unconstrained optimizer vector; on resume it is turned back
/// into `ModelParameters` via `unpack_params` against the run's init-params
/// template, so `coord_names` and `packed.len()` are validated against that
/// template before use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
    pub ferx_version: String,
    /// SHA-256 of the model file at save time (the resume integrity check).
    pub model_hash: Option<String>,
    /// SHA-256 of the data CSV at save time.
    pub data_hash: Option<String>,
    /// Human-readable method chain, e.g. `["saem", "focei"]`.
    pub method_chain: Vec<String>,
    /// Index into `method_chain` of the stage that was running when saved.
    pub stage_idx: usize,
    /// Packed unconstrained parameter vector (see struct docs).
    #[serde(with = "nan_safe_vec")]
    pub packed: Vec<f64>,
    /// Optimizer coordinate names in packed order; length- and name-checked on
    /// resume so a structurally different model can never be resumed by mistake.
    pub coord_names: Vec<String>,
    /// Iteration / eval counter within the running stage at save time.
    pub iter: usize,
    /// Best OFV seen so far (informational; shown in the resume banner).
    #[serde(with = "nan_safe_f64")]
    pub ofv: f64,
    /// Unix seconds at save (informational).
    pub unix_time: u64,
}

// ─── file I/O (all best-effort; a checkpoint must never abort a fit) ─────────

/// Serialise `cp` to `path` atomically: write `{path}.new`, then rename it over
/// `path`. Rename is atomic within a filesystem, so an interrupt mid-write can
/// never corrupt an existing checkpoint (the old one stays intact until the new
/// one is fully written).
pub fn save_atomic(path: &str, cp: &Checkpoint) -> std::io::Result<()> {
    let json = serde_json::to_vec_pretty(cp).map_err(std::io::Error::other)?;
    let tmp = format!("{path}.new");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load and parse a checkpoint. Returns `None` on *any* problem — missing,
/// unreadable, corrupt JSON, or a schema version this build doesn't understand
/// — because resuming is always optional: a bad checkpoint just means "start
/// fresh."
pub fn load(path: &str) -> Option<Checkpoint> {
    let bytes = std::fs::read(path).ok()?;
    let cp: Checkpoint = serde_json::from_slice(&bytes).ok()?;
    if cp.schema_version != SCHEMA_VERSION {
        return None;
    }
    Some(cp)
}

/// Remove the checkpoint file (and any stray `.new` temp) if present.
/// Best-effort; a missing file is not an error.
pub fn remove(path: &str) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(format!("{path}.new"));
}

// ─── thread-local writer ─────────────────────────────────────────────────────

struct Sink {
    path: String,
    ferx_version: String,
    model_hash: Option<String>,
    data_hash: Option<String>,
    method_chain: Vec<String>,
    coord_names: Vec<String>,
    interval_secs: u64,
    stage_idx: usize,
    /// Instant of the last write, or of `init` before the first write. A write
    /// happens only when `elapsed >= interval_secs`, which is what makes a short
    /// run leave no checkpoint behind.
    last: Instant,
}

thread_local! {
    static SINK: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

/// Begin checkpointing for this fit on the current thread. Replaces any prior
/// sink (e.g. back-to-back fits on a pooled worker). Does **not** touch the file
/// — the first write only happens after `interval_secs` elapse, so a checkpoint
/// being resumed from is preserved until a newer state replaces it.
#[allow(clippy::too_many_arguments)]
pub fn init(
    path: String,
    ferx_version: String,
    model_hash: Option<String>,
    data_hash: Option<String>,
    method_chain: Vec<String>,
    coord_names: Vec<String>,
    interval_secs: u64,
) {
    SINK.with(|s| {
        *s.borrow_mut() = Some(Sink {
            path,
            ferx_version,
            model_hash,
            data_hash,
            method_chain,
            coord_names,
            interval_secs,
            stage_idx: 0,
            last: Instant::now(),
        });
    });
}

/// True when a checkpoint sink is active on this thread.
pub fn is_active() -> bool {
    SINK.with(|s| s.borrow().is_some())
}

/// True when a write is due (the interval has elapsed) — lets hot estimator
/// loops skip building the packed vector except when a checkpoint will actually
/// be written. Returns `false` when no sink is active.
pub fn is_due() -> bool {
    SINK.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sink| sink.last.elapsed().as_secs() >= sink.interval_secs)
            .unwrap_or(false)
    })
}

/// Record which chain stage is now running; written into subsequent checkpoints.
pub fn set_stage(stage_idx: usize) {
    SINK.with(|s| {
        if let Some(sink) = s.borrow_mut().as_mut() {
            sink.stage_idx = stage_idx;
        }
    });
}

/// Write a checkpoint iff at least `interval_secs` have elapsed since the last
/// write (or since `init`). `packed` is the current packed optimizer vector;
/// `iter` / `ofv` are informational. No-op when no sink is active. Best-effort:
/// I/O errors are swallowed so a fit never fails because a checkpoint couldn't
/// be written (the clock only advances on a successful write, so a transient
/// failure is retried next iteration).
pub fn maybe_write(iter: usize, ofv: f64, packed: &[f64]) {
    SINK.with(|s| {
        let mut guard = s.borrow_mut();
        let Some(sink) = guard.as_mut() else {
            return;
        };
        if sink.last.elapsed().as_secs() < sink.interval_secs {
            return;
        }
        let cp = Checkpoint {
            schema_version: SCHEMA_VERSION,
            ferx_version: sink.ferx_version.clone(),
            model_hash: sink.model_hash.clone(),
            data_hash: sink.data_hash.clone(),
            method_chain: sink.method_chain.clone(),
            stage_idx: sink.stage_idx,
            packed: packed.to_vec(),
            coord_names: sink.coord_names.clone(),
            iter,
            ofv,
            unix_time: now_unix(),
        };
        if save_atomic(&sink.path, &cp).is_ok() {
            sink.last = Instant::now();
        }
    });
}

/// Successful completion: drop the sink and delete the checkpoint file. The
/// run finished, so there is nothing left to resume.
pub fn finish_success() {
    SINK.with(|s| {
        if let Some(sink) = s.borrow_mut().take() {
            remove(&sink.path);
        }
    });
}

/// Drop the sink *without* deleting the file — used when a fit ends abnormally
/// (error / cancel) so the last checkpoint survives for a later resume.
pub fn abandon() {
    SINK.with(|s| {
        s.borrow_mut().take();
    });
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(packed: Vec<f64>) -> Checkpoint {
        Checkpoint {
            schema_version: SCHEMA_VERSION,
            ferx_version: "test".to_string(),
            model_hash: Some("mh".to_string()),
            data_hash: Some("dh".to_string()),
            method_chain: vec!["focei".to_string()],
            stage_idx: 0,
            coord_names: (0..packed.len()).map(|i| format!("c{i}")).collect(),
            packed,
            iter: 7,
            ofv: 123.5,
            unix_time: 42,
        }
    }

    fn tmp_path(tag: &str) -> String {
        // Unique per process; tests clean up after themselves.
        format!("/tmp/ferx_ckpt_{}_{}.tmp", tag, std::process::id())
    }

    #[test]
    fn save_load_round_trip() {
        let path = tmp_path("roundtrip");
        let cp = sample(vec![0.1, -0.2, 1.3]);
        save_atomic(&path, &cp).unwrap();
        let back = load(&path).expect("load should succeed");
        assert_eq!(back.packed, cp.packed);
        assert_eq!(back.coord_names, cp.coord_names);
        assert_eq!(back.stage_idx, cp.stage_idx);
        assert_eq!(back.iter, cp.iter);
        assert_eq!(back.method_chain, cp.method_chain);
        remove(&path);
    }

    #[test]
    fn nonfinite_ofv_round_trips_as_nan() {
        let path = tmp_path("nanofv");
        let mut cp = sample(vec![1.0]);
        cp.ofv = f64::NAN;
        save_atomic(&path, &cp).unwrap();
        let back = load(&path).unwrap();
        assert!(back.ofv.is_nan(), "non-finite OFV should round-trip to NaN");
        remove(&path);
    }

    #[test]
    fn load_missing_returns_none() {
        assert!(load(&tmp_path("does_not_exist")).is_none());
    }

    #[test]
    fn load_corrupt_returns_none() {
        let path = tmp_path("corrupt");
        std::fs::write(&path, b"{ not valid json").unwrap();
        assert!(load(&path).is_none());
        remove(&path);
    }

    #[test]
    fn load_wrong_schema_version_returns_none() {
        let path = tmp_path("badver");
        let mut cp = sample(vec![1.0]);
        cp.schema_version = SCHEMA_VERSION + 99;
        // Serialise directly (save_atomic would keep the bumped version).
        std::fs::write(&path, serde_json::to_vec(&cp).unwrap()).unwrap();
        assert!(load(&path).is_none());
        remove(&path);
    }

    #[test]
    fn remove_is_idempotent() {
        let path = tmp_path("rm");
        save_atomic(&path, &sample(vec![1.0])).unwrap();
        remove(&path);
        remove(&path); // second remove on a missing file must not panic
        assert!(load(&path).is_none());
    }

    #[test]
    fn short_run_writes_nothing() {
        // Large interval → maybe_write never fires → no file created.
        let path = tmp_path("short");
        remove(&path);
        init(
            path.clone(),
            "test".to_string(),
            None,
            None,
            vec!["focei".to_string()],
            vec!["c0".to_string()],
            3600,
        );
        assert!(is_active());
        assert!(!is_due(), "not due immediately with a 1h interval");
        maybe_write(1, 1.0, &[0.5]);
        maybe_write(2, 0.9, &[0.4]);
        assert!(
            load(&path).is_none(),
            "no checkpoint should exist for a sub-interval run"
        );
        abandon();
        remove(&path);
    }

    #[test]
    fn zero_interval_writes_immediately() {
        let path = tmp_path("zero");
        remove(&path);
        init(
            path.clone(),
            "test".to_string(),
            Some("mh".to_string()),
            Some("dh".to_string()),
            vec!["saem".to_string(), "focei".to_string()],
            vec!["a".to_string(), "b".to_string()],
            0,
        );
        set_stage(1);
        assert!(is_due());
        maybe_write(5, 88.0, &[0.11, 0.22]);
        let cp = load(&path).expect("interval 0 → write happens");
        assert_eq!(cp.stage_idx, 1);
        assert_eq!(cp.iter, 5);
        assert_eq!(cp.packed, vec![0.11, 0.22]);
        assert_eq!(cp.method_chain, vec!["saem", "focei"]);

        finish_success();
        assert!(!is_active(), "finish_success drops the sink");
        assert!(load(&path).is_none(), "finish_success deletes the file");
    }

    #[test]
    fn abandon_keeps_file() {
        let path = tmp_path("abandon");
        remove(&path);
        init(
            path.clone(),
            "test".to_string(),
            None,
            None,
            vec!["focei".to_string()],
            vec!["c0".to_string()],
            0,
        );
        maybe_write(1, 1.0, &[0.5]);
        assert!(load(&path).is_some());
        abandon();
        assert!(!is_active());
        assert!(
            load(&path).is_some(),
            "abandon must leave the checkpoint for later resume"
        );
        remove(&path);
    }

    #[test]
    fn maybe_write_is_noop_without_sink() {
        // Contract the estimators rely on: calling into a fresh thread with no
        // active sink must not panic and must not create files.
        assert!(!is_active());
        assert!(!is_due());
        maybe_write(1, 1.0, &[0.1]);
        set_stage(3);
        finish_success();
        abandon();
    }
}
