//! DEV ONLY (removed before the PR): counters and switches for attributing the #2706
//! follow-up speedups. `FERX_DEV_COUNT=1` prints the counters at the end of `fit`;
//! `FERX_DEV_OFF=a,b` turns named features off.
use std::sync::atomic::{AtomicUsize, Ordering};

pub const N: usize = 28;
pub static COUNTS: [AtomicUsize; N] = [const { AtomicUsize::new(0) }; N];
pub const NAMES: [&str; N] = [
    "iov_solves_warm",
    "iov_solves_cold",
    "iov_nm_warm",
    "iov_nm_cold",
    "iov_bfgs_failed",
    "iov_bfgs_stuck",
    "iov_runaway_nm",
    "lin_spectral",
    "lin_spectral_rejected",
    "lin_spectral_declined",
    "ls_accept",
    "ls_rej_nonfinite",
    "ls_rej_increase",
    "ls_rej_small_decrease",
    "ls_floor_stop",
    "ls_rej_inc_reset",
    "k0r0",
    "k0r1",
    "k0r2_3",
    "k0r4p",
    "k1r0",
    "k1r1",
    "k1r2_3",
    "k1r4p",
    "k2r0",
    "k2r1",
    "k2r2_3",
    "k2r4p",
];

#[inline]
pub fn bump(i: usize) {
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
}

pub fn off(name: &str) -> bool {
    static OFF: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    OFF.get_or_init(|| {
        std::env::var("FERX_DEV_OFF")
            .map(|v| v.split(',').map(str::to_string).collect())
            .unwrap_or_default()
    })
    .iter()
    .any(|n| n == name)
}

pub fn report() {
    if std::env::var_os("FERX_DEV_COUNT").is_none() {
        return;
    }
    let parts: Vec<String> = (0..N)
        .filter(|&i| COUNTS[i].load(Ordering::Relaxed) > 0)
        .map(|i| format!("{}={}", NAMES[i], COUNTS[i].load(Ordering::Relaxed)))
        .collect();
    eprintln!("DEVCOUNT {}", parts.join(" "));
}
