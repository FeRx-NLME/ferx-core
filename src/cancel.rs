//! Cooperative cancellation token for long-running fits.
//!
//! Motivation: when ferx-core is called from R (via extendr) or any other
//! FFI host, the host cannot interrupt our Rust hot loops. `CancelFlag` is a
//! cheap shared atomic that callers clone into [`FitOptions`](crate::types::FitOptions);
//! the estimation loops poll it at safe points and return a partial/empty
//! result quickly when set.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Clone-share this into `FitOptions.cancel` and call [`CancelFlag::cancel`]
/// from another thread to request a cooperative abort of an in-flight fit.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(
    Arc<AtomicBool>,
    /// Test-only countdown: polls left before the flag trips itself (#574). Lets a test
    /// land a cancel at an exact poll inside a fit instead of racing a second thread.
    #[cfg(test)]
    Option<Arc<std::sync::atomic::AtomicUsize>>,
);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        #[cfg(test)]
        if let Some(left) = &self.1 {
            let before = left
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .unwrap_or(0);
            if before == 0 {
                self.cancel();
            }
        }
        self.0.load(Ordering::Relaxed)
    }

    /// A flag that reads `false` on its first `polls` polls and cancels itself on the
    /// next one (`trip_after(0)` is a pre-cancelled flag).
    #[cfg(test)]
    pub(crate) fn trip_after(polls: usize) -> Self {
        Self(
            Arc::default(),
            Some(Arc::new(std::sync::atomic::AtomicUsize::new(polls))),
        )
    }
}

/// Cheap helper: returns `true` if the optional flag is set.
///
/// Used inside tight loops to keep the polling call-site a one-liner.
#[inline]
pub fn is_cancelled(flag: &Option<CancelFlag>) -> bool {
    matches!(flag, Some(f) if f.is_cancelled())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trip_after_cancels_on_the_poll_after_its_budget() {
        let flag = CancelFlag::trip_after(2);
        assert!(!flag.is_cancelled());
        assert!(!flag.is_cancelled());
        assert!(flag.is_cancelled());
        assert!(flag.is_cancelled(), "a tripped flag stays cancelled");
        assert!(CancelFlag::trip_after(0).is_cancelled());
        assert!(!CancelFlag::new().is_cancelled());
    }
}
