#![allow(unused_imports)]
//! Extracted verbatim from `api/mod.rs` (production peel). See the module-
//! doc / Key Modules table for the split rationale.
use super::*;
use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::{
    chol_lt_idx, lower_tri_iter, omega_packed_len, theta_packs_log,
};
use crate::estimation::saem;
use crate::io::datareader::{
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_mapped,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    SelectionFilter, ERR_COV_MISSING_COLUMNS, ERR_COV_NON_NUMERIC,
};
use crate::pk;
use crate::propensity_match::MatchMethod;
use crate::sim::adaptive::{
    AdaptiveRun, AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DoseAction,
    DoseLedgerEntry, MonitorSpec,
};
use crate::stats::likelihood::{
    build_frem_r_override, compute_cwres, foce_subject_nll, foce_subject_nll_iov,
};
use crate::stats::residual_error::{
    compute_iwres_with_correlations, compute_r_matrix_with_correlations,
    compute_r_matrix_with_correlations_scaled, iwres_autocorrelation,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Worker-stack size for ferx-managed Rayon pools.
///
/// Wide ODE+IOV analytic sensitivities instantiate `Dual2<M>` with `M` close to 100.
/// Each dual carries an `M x M` Hessian, and the ODE/event-walk frames hold several
/// of them at once. Rayon/default pthread stacks can be as small as 2 MiB on macOS,
/// which overflows before Rust can unwind. Use a larger fit-scoped stack for Rayon
/// workers that may evaluate these gradients.
pub const FIT_RAYON_STACK_SIZE: usize = 32 * 1024 * 1024;

pub(crate) fn fit_thread_pool_builder() -> rayon::ThreadPoolBuilder {
    rayon::ThreadPoolBuilder::new().stack_size(FIT_RAYON_STACK_SIZE)
}

/// Ceiling applied to the unpinned default thread count (#707).
const DEFAULT_THREADS_CAP: usize = 8;

/// Core cap arithmetic for the unpinned default thread count, split out from
/// [`default_thread_count`] so it is testable without depending on the host's actual
/// core count: leave one core free for the OS/other work, and don't scale past
/// [`DEFAULT_THREADS_CAP`] even on much larger machines, since most fits see no benefit
/// from spreading across every core and (notably on Apple Silicon) not all cores are equal.
pub(crate) fn cap_default_threads(available: usize) -> usize {
    available.saturating_sub(1).clamp(1, DEFAULT_THREADS_CAP)
}

/// Default worker-thread count used when nothing pins an explicit count (`threads` unset
/// or `auto`/`0`, and no explicit `--threads`/[`configure_global_thread_pool`] call). See
/// [`cap_default_threads`] for the cap logic (#707).
pub(crate) fn default_thread_count() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cap_default_threads(available)
}

/// Set when a caller has explicitly sized the process-wide Rayon pool (currently: the CLI's
/// `--threads N` via [`configure_global_thread_pool`]), so [`default_fit_pool`] knows to
/// honor that explicit choice rather than applying the [`default_thread_count`] cap (#707).
static GLOBAL_THREADS_EXPLICIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Explicitly size the process-wide Rayon pool and mark it as user-chosen. Intended for a
/// CLI binary sizing its one process-wide pool from `--threads N` before the first fit;
/// library callers that want a pinned thread count for a single fit should use
/// `FitOptions::threads` instead, which scopes a fit-local pool via `build_fit_pool`.
///
/// `n_threads` must be positive — a caller wanting the engine's own default should simply
/// not call this at all, rather than pass `0` (which Rayon would otherwise silently treat
/// as "pick automatically", masking the caller's intent). The explicit-override flag is
/// only set once `build_global` actually succeeds, so a failed call (e.g. the global pool
/// was already initialized elsewhere) leaves `default_fit_pool` applying the #707 cap
/// rather than incorrectly deferring to whatever the ambient pool happens to be.
pub fn configure_global_thread_pool(n_threads: usize) -> Result<(), String> {
    if n_threads == 0 {
        return Err("thread count must be positive".to_string());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|e| format!("failed to configure thread pool with {n_threads} threads: {e}"))?;
    GLOBAL_THREADS_EXPLICIT.store(true, std::sync::atomic::Ordering::Release);
    Ok(())
}

/// Build a fit-scoped Rayon pool with the ferx worker stack and an explicit thread
/// count. Used only when the caller pins `options.threads` to a positive value; the
/// common (unpinned) path reuses the shared [`default_fit_pool`] instead.
pub(crate) fn build_fit_pool(n_threads: usize) -> Result<rayon::ThreadPool, String> {
    fit_thread_pool_builder()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("failed to build rayon pool with {n_threads} threads: {e}"))
}

/// The process-wide fit pool, built once with the ferx worker stack (32 MiB) so wide
/// ODE+IOV analytic gradients do not overflow the platform-default worker stack. Shared
/// across every default-threads `fit()` call so batch / concurrent callers do not each
/// spawn-and-tear-down a fresh `N × 32 MiB` pool (which oversubscribes CPUs and can
/// exhaust address space).
///
/// Sized by [`default_thread_count`] (available cores minus one, capped at 8 — #707): most
/// fits gain little from spreading across every core, and not all cores are equal on
/// asymmetric platforms (e.g. Apple Silicon E-cores). A caller that explicitly sized the
/// global pool via [`configure_global_thread_pool`] (the CLI's `--threads N`) is honored
/// instead — that call marks [`GLOBAL_THREADS_EXPLICIT`] before this pool is built.
///
/// Returns `None` only if the one-time build fails (e.g. resource limits); callers then
/// run on the ambient pool rather than aborting the fit.
pub(crate) fn default_fit_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n_threads = if GLOBAL_THREADS_EXPLICIT.load(std::sync::atomic::Ordering::Acquire) {
            rayon::current_num_threads()
        } else {
            default_thread_count()
        };
        fit_thread_pool_builder()
            .num_threads(n_threads)
            .build()
            .ok()
    })
    .as_ref()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Nested-pool planning for tools that run many fits (#1115)
// ─────────────────────────────────────────────────────────────────────────────

/// How a total thread budget is split between the two levels of parallelism a
/// many-fit tool (bootstrap, SCM, model search) has available.
///
/// `fit()` already `par_iter`s over subjects, so a tool that naively `par_iter`s
/// over replicates nests one pool on top of another and the two levels compete
/// for the same workers. A `PoolPlan` makes the split explicit:
///
/// * [`replicates`](Self::replicates) — width of the **outer** pool, i.e. how
///   many fits run concurrently;
/// * [`threads_per_fit`](Self::threads_per_fit) — what each inner `fit()` gets,
///   applied via [`apply_to`](Self::apply_to) (which sets `FitOptions::threads`).
///
/// Both are always ≥ 1, so a plan can never silently mean "let Rayon decide".
///
/// The outer pool is built by [`install`](Self::install) with the ferx worker
/// stack ([`FIT_RAYON_STACK_SIZE`]), **not** the platform default: wide ODE+IOV
/// analytic-gradient models overflow a stock 2 MiB Rayon worker stack, and an
/// outer pool built from a bare `rayon::ThreadPoolBuilder::new()` would inherit
/// that default and fault on exactly the models that need the bigger stack.
/// Using this type is how a tool gets the right stack without remembering to.
///
/// ```no_run
/// use ferx_core::{FitOptions, PoolPlan};
///
/// // 200 bootstrap replicates over an 8-thread budget: 8 fits at a time,
/// // each single-threaded (replicate-level parallelism is the better axis —
/// // embarrassingly parallel, no per-subject synchronisation).
/// let plan = PoolPlan::from_budget(8, 200);
/// assert_eq!((plan.replicates(), plan.threads_per_fit()), (8, 1));
///
/// let mut opts = FitOptions::default().quiet();
/// plan.apply_to(&mut opts);
/// # let run_one = |_i: usize| ();
/// plan.install(|| {
///     use rayon::prelude::*;
///     (0..200usize).into_par_iter().for_each(run_one);
/// })
/// .expect("outer pool");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolPlan {
    replicates: usize,
    threads_per_fit: usize,
}

impl PoolPlan {
    /// A plan with an explicit split. Both arguments are clamped to at least 1,
    /// since `0` would mean "Rayon picks" — the ambiguity this type exists to
    /// remove.
    pub fn new(replicates: usize, threads_per_fit: usize) -> Self {
        Self {
            replicates: replicates.max(1),
            threads_per_fit: threads_per_fit.max(1),
        }
    }

    /// Split `total_threads` across `n_units` fits, spending the budget on the
    /// **outer** level first:
    ///
    /// ```text
    /// replicates      = min(total_threads, n_units)     (at least 1)
    /// threads_per_fit = total_threads / replicates      (at least 1, floored)
    /// ```
    ///
    /// So a 200-replicate bootstrap on 8 threads runs 8 single-threaded fits at
    /// a time, while a 2-replicate run on the same budget gives each fit 4
    /// subject-level workers. The division floors: 8 threads over 3 units is
    /// `3 × 2`, leaving two threads unused rather than oversubscribing.
    ///
    /// `total_threads = 0` means "the engine default" — the same
    /// available-cores-minus-one, capped at 8, that an unpinned `fit()` uses.
    /// `n_units = 0` yields a single-replicate plan.
    pub fn from_budget(total_threads: usize, n_units: usize) -> Self {
        let total = if total_threads == 0 {
            default_thread_count()
        } else {
            total_threads
        };
        let replicates = total.min(n_units).max(1);
        Self::new(replicates, total / replicates)
    }

    /// Width of the outer pool: how many fits run concurrently.
    pub fn replicates(&self) -> usize {
        self.replicates
    }

    /// Subject-level worker threads handed to each inner `fit()`.
    pub fn threads_per_fit(&self) -> usize {
        self.threads_per_fit
    }

    /// Worker threads in flight when every replicate is running:
    /// `replicates * threads_per_fit`.
    pub fn total_threads(&self) -> usize {
        self.replicates * self.threads_per_fit
    }

    /// Worker-stack size of the outer pool [`install`](Self::install) builds —
    /// always [`FIT_RAYON_STACK_SIZE`], never the platform default.
    pub fn worker_stack_size(&self) -> usize {
        FIT_RAYON_STACK_SIZE
    }

    /// Pin the inner fit's subject-level parallelism to
    /// [`threads_per_fit`](Self::threads_per_fit) by setting
    /// `FitOptions::threads`.
    ///
    /// With `threads_per_fit == 1` this is the documented single-threaded inner
    /// mode: `fit()` runs its per-subject loops on a one-worker pool, so no
    /// subject-level fan-out competes with the outer replicate loop, and the
    /// resulting `FitResult::n_threads_used` is `1`.
    pub fn apply_to(&self, options: &mut FitOptions) {
        options.threads = Some(self.threads_per_fit);
    }

    /// Run `op` on an outer Rayon pool of [`replicates`](Self::replicates)
    /// workers, each with the ferx worker stack ([`FIT_RAYON_STACK_SIZE`]).
    ///
    /// Any `par_iter` inside `op` fans out over that pool. Returns `Err` if the
    /// pool cannot be built (e.g. resource limits).
    pub fn install<R, OP>(&self, op: OP) -> Result<R, String>
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        Ok(build_fit_pool(self.replicates)?.install(op))
    }
}

impl Default for PoolPlan {
    /// The engine's default budget spread entirely over the outer level:
    /// `from_budget(0, usize::MAX)`, i.e. one single-threaded fit per default
    /// worker thread.
    fn default() -> Self {
        Self::from_budget(0, usize::MAX)
    }
}

#[cfg(test)]
#[path = "tests/pool_plan_tests.rs"]
mod pool_plan_tests;
