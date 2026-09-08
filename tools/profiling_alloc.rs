//! Example-only Rust allocation counter and sampled stacks. Never linked into the engine.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

pub struct Allocator;
static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static MAX_REQUEST: AtomicU64 = AtomicU64::new(0);
static SAMPLE_BYTES: AtomicBool = AtomicBool::new(false);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static STRIDE: AtomicU64 = AtomicU64::new(0);
static SAMPLES: AtomicU64 = AtomicU64::new(0);
static BUCKETS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static BUCKET_BYTES: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static STACKS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
thread_local! { static CAPTURING: Cell<bool> = const { Cell::new(false) }; }

fn record(bytes: usize, realloc: bool) {
    if !ENABLED.load(Relaxed) {
        return;
    }
    let _ = CAPTURING.try_with(|busy| {
        if busy.get() {
            return;
        }
        let n = CALLS.fetch_add(1, Relaxed).wrapping_add(1);
        let previous_bytes = BYTES.fetch_add(bytes as u64, Relaxed);
        MAX_REQUEST.fetch_max(bytes as u64, Relaxed);
        if realloc {
            REALLOCS.fetch_add(1, Relaxed);
        }
        let bucket = [32, 128, 512, 2048, 8192, 32768, 131072]
            .iter()
            .position(|&limit| bytes <= limit)
            .unwrap_or(7);
        BUCKETS[bucket].fetch_add(1, Relaxed);
        BUCKET_BYTES[bucket].fetch_add(bytes as u64, Relaxed);
        let stride = STRIDE.load(Relaxed);
        if stride == 0 {
            return;
        }
        let sample = if SAMPLE_BYTES.load(Relaxed) {
            previous_bytes / stride != previous_bytes.wrapping_add(bytes as u64) / stride
        } else {
            n % stride == 0
        };
        if !sample {
            return;
        }
        if SAMPLES.fetch_add(1, Relaxed) >= 512 {
            return;
        }
        // Exclude profiler bookkeeping and prevent recursive stack sampling.
        // GlobalAlloc must never unwind: abort if optional symbolization panics.
        busy.set(true);
        let captured = std::panic::catch_unwind(|| {
            let trace = std::backtrace::Backtrace::force_capture().to_string();
            let mut selected = Vec::new();
            let mut include_location = false;
            let mut frames = 0;
            for line in trace.lines() {
                if line.contains("ferx_core::") && frames < 10 {
                    let trimmed = line.trim();
                    selected.push(
                        trimmed
                            .split_once(": ")
                            .map_or(trimmed, |(_, symbol)| symbol),
                    );
                    include_location = true;
                    frames += 1;
                } else if include_location && line.trim().starts_with("at ") {
                    selected.push(line.trim());
                    include_location = false;
                } else {
                    include_location = false;
                }
            }
            let mut stack = selected.join("\n");
            if stack.is_empty() {
                stack = trace;
            }
            let mut stacks = STACKS
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *stacks.entry(stack).or_insert(0) += 1;
        });
        busy.set(false);
        if captured.is_err() {
            std::process::abort();
        }
    });
}

// SAFETY: every pointer/layout/size is forwarded unchanged to System. The
// observer neither dereferences payloads nor changes allocation ownership.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            record(layout.size(), false);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            record(layout.size(), false);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(p, layout, new_size) };
        if !p.is_null() {
            record(new_size, true);
        }
        p
    }
}

pub fn start(stride: u64) {
    ENABLED.store(false, Relaxed);
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    MAX_REQUEST.store(0, Relaxed);
    REALLOCS.store(0, Relaxed);
    SAMPLES.store(0, Relaxed);
    STRIDE.store(stride, Relaxed);
    for b in &BUCKETS {
        b.store(0, Relaxed);
    }
    for b in &BUCKET_BYTES {
        b.store(0, Relaxed);
    }
    STACKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    ENABLED.store(true, Relaxed);
}

pub fn sample_bytes(on: bool) {
    SAMPLE_BYTES.store(on, Relaxed);
}

pub fn stop() -> serde_json::Value {
    ENABLED.store(false, Relaxed);
    let mut stacks: Vec<_> = STACKS
        .get()
        .unwrap()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(stack, &samples)| serde_json::json!({"stack": stack, "samples": samples}))
        .collect();
    stacks.sort_by_key(|v| std::cmp::Reverse(v["samples"].as_u64().unwrap()));
    serde_json::json!({
        "allocation_calls": CALLS.load(Relaxed), "requested_bytes": BYTES.load(Relaxed),
        "max_request_bytes": MAX_REQUEST.load(Relaxed),
        "realloc_calls": REALLOCS.load(Relaxed), "sample_stride": STRIDE.load(Relaxed),
        "sample_basis": if SAMPLE_BYTES.load(Relaxed) { "requested_bytes" } else { "allocation_calls" },
        "samples_capped": SAMPLES.load(Relaxed) > 512,
        "allocation_size_buckets": BUCKETS.iter().map(|v| v.load(Relaxed)).collect::<Vec<_>>(),
        "bytes_by_size_bucket": BUCKET_BYTES.iter().map(|v| v.load(Relaxed)).collect::<Vec<_>>(),
        "bucket_upper_bytes": [32,128,512,2048,8192,32768,131072,u64::MAX], "stacks": stacks,
    })
}
