//! DEV-ONLY (#2549): per-feature off switches for attributing a batch of speedups from one
//! build. `FERX_DEV_OFF=pool,iov,...` turns the named fast paths off. Read once per process.
//! This module is removed before the PR leaves draft.

use std::sync::OnceLock;

pub(crate) const POOL: u32 = 1 << 0;
pub(crate) const IOVBUF: u32 = 1 << 1;
pub(crate) const SOLPT: u32 = 1 << 2;
pub(crate) const REGEX: u32 = 1 << 3;
pub(crate) const COVSLOT: u32 = 1 << 4;
pub(crate) const F5: u32 = 1 << 5;
pub(crate) const F6: u32 = 1 << 6;
pub(crate) const F7: u32 = 1 << 7;

fn off_mask() -> u32 {
    static MASK: OnceLock<u32> = OnceLock::new();
    *MASK.get_or_init(|| {
        let Ok(s) = std::env::var("FERX_DEV_OFF") else {
            return 0;
        };
        s.split(',')
            .map(|t| match t.trim() {
                "pool" => POOL,
                "iov" => IOVBUF,
                "solpt" => SOLPT,
                "regex" => REGEX,
                "covslot" => COVSLOT,
                "f5" => F5,
                "f6" => F6,
                "f7" => F7,
                _ => 0,
            })
            .fold(0, |a, b| a | b)
    })
}

#[inline]
pub(crate) fn on(flag: u32) -> bool {
    off_mask() & flag == 0
}
