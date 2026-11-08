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
/// core count: leave one core free for the OS/other work (but never drop below two workers where
/// two cores exist), and don't scale past
/// [`DEFAULT_THREADS_CAP`] even on much larger machines, since most fits see no benefit
/// from spreading across every core and (notably on Apple Silicon) not all cores are equal.
///
/// Never fewer than two workers where two cores exist: leaving one free on a two-core host
/// left a single worker, every fit serial (measured equal to one core on a 2-core
/// affinity, #2706 follow-up), so the free core bought nothing but half the throughput.
pub(crate) fn cap_default_threads(available: usize) -> usize {
    available
        .saturating_sub(1)
        .max(available.min(2))
        .clamp(1, DEFAULT_THREADS_CAP)
}

/// Default worker-thread count used when nothing pins an explicit count (`threads` unset
/// or `auto`/`0`, and no explicit `--threads`/[`configure_global_thread_pool`] call). See
/// [`cap_default_threads`] for the cap logic (#707).
pub(crate) fn default_thread_count() -> usize {
    cap_default_threads(available_cpus())
}

/// `std::thread::available_parallelism()`, `1` when it cannot be read. Respects cgroup
/// quotas and affinity masks, so it is the host's usable width inside a container too.
pub(crate) fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The process-wide worker count a caller explicitly asked for (currently: the CLI's
/// `--threads N` via [`configure_global_thread_pool`]), or `0` when nothing has — so
/// [`effective_default_threads`] honors that explicit choice rather than applying the
/// [`default_thread_count`] cap (#707).
///
/// A count rather than a flag because the width is now *recorded* here instead of being
/// read back off Rayon's global pool: that pool is never built, so there is nothing to
/// read (#1460).
static EXPLICIT_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Declare the worker count this process should fit with, overriding the automatic default
/// (available cores minus one, at least two where two cores exist, capped at 8). Intended for
/// a CLI binary applying its `--threads N` before the first fit; library callers that want a
/// pinned thread count for a single fit should use `FitOptions::threads` instead, which
/// exclusively leases a pool for that call.
///
/// **This does not build Rayon's global pool** (#1460), despite the name it has carried
/// since #707. It used to: `--threads N` called `build_global` with `N`, and then every
/// fit ran on a *separate* N-worker pool — `fit()` leases one of its own, with the 32 MiB
/// worker stack that wide `Dual2` ODE+IOV gradients need, which Rayon's global pool does
/// not have. Per-thread `samply` profiles of a `--threads 2` SAEM fit measured the two
/// fit-pool workers at 85–99 % busy and the two global-pool threads at 0.0 % for the whole
/// run: `--threads N` was spending `2N` OS threads to do `N` threads of work. Recording
/// the count instead leaves one pool — the one the fit actually runs on — sized to `N`.
///
/// `n_threads` must be positive — a caller wanting the engine's own default should simply
/// not call this at all, rather than pass `0` (which would otherwise read as "pick
/// automatically", masking the caller's intent).
///
/// **Once per process**, like the `build_global` it replaces. The shared default pool an
/// unpinned fit runs on is sized from this on first use and cannot be resized afterwards,
/// so a second, *differing* count would be reported as configured while the pool kept the
/// first one — and `PoolPlan::from_budget(0, …)` would then disagree with the pool an
/// unpinned fit actually runs on, which is exactly the agreement #1115 requires. A repeat
/// call with the same count is accepted (nothing changes); a differing one is an `Err`
/// naming both, as the second `build_global` used to be.
pub fn configure_global_thread_pool(n_threads: usize) -> Result<(), String> {
    if n_threads == 0 {
        return Err("thread count must be positive".to_string());
    }
    match EXPLICIT_THREADS.compare_exchange(
        0,
        n_threads,
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
    ) {
        Ok(_) => Ok(()),
        Err(already) => reconfiguration_result(already, n_threads),
    }
}

/// Outcome of a second [`configure_global_thread_pool`] call: idempotent when it names the
/// count already in force, an error naming both when it does not.
///
/// Split out so the message is unit-testable without writing the process-global static
/// every other test in the binary shares.
pub(crate) fn reconfiguration_result(already: usize, requested: usize) -> Result<(), String> {
    if already == requested {
        return Ok(());
    }
    Err(format!(
        "thread count is already configured to {already} and cannot be changed to \
         {requested}: pools sized from the first value may already exist"
    ))
}

/// The process-wide fit pool, built once with the ferx worker stack (32 MiB) so wide
/// ODE+IOV analytic gradients do not overflow the platform-default worker stack. Shared
/// across every default-threads `fit()` call so batch / concurrent callers do not each
/// spawn-and-tear-down a fresh `N × 32 MiB` pool (which oversubscribes CPUs and can
/// exhaust address space).
///
/// Sized by [`effective_default_threads`]: a width declared via
/// [`configure_global_thread_pool`] if any — that call records [`EXPLICIT_THREADS`] before
/// this pool is built, and this is then the only pool `N` sizes (#1460) — else a positive
/// `RAYON_NUM_THREADS` (#2800), else [`default_thread_count`] (available cores minus one, at
/// least two where two cores exist, capped at 8 — #707): most fits gain little from
/// spreading across every core, and not all cores are equal on asymmetric platforms (e.g.
/// Apple Silicon E-cores).
///
/// Returns `None` only if the one-time build fails (e.g. resource limits); callers then
/// run on the ambient pool rather than aborting the fit.
pub(crate) fn default_fit_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        fit_thread_pool_builder()
            .num_threads(effective_default_threads())
            .build()
            .ok()
    })
    .as_ref()
}

#[path = "pool_cache.rs"]
mod cache;
use cache::{FitPoolLease, PoolCache, SharedPoolCache};

/// Everything a fit-scoped call's workers must carry so that a per-subject solve reads the
/// call's own settings: the ODE solver override (#1212), the inner-loop settings (#426),
/// whether the call asked for `gradient = fd` (#1613) and its `bloq_method` override (#1824).
/// It is the key of the fit pools, so a worker serves only calls with exactly this value and
/// "which call am I serving?" has one answer per thread. The empty scope (no ODE override,
/// default inner settings, gradient not forced, no `bloq_method` override) is what the shared
/// pool's workers carry, by installing nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct FitScope {
    pub(crate) ode: crate::ode::solver::OdeSolverOverride,
    pub(crate) inner: crate::estimation::inner_optimizer::InnerSettings,
    /// Beside `inner` rather than inside it: it gates the outer gradient, the `auto` optimizer
    /// pick and the covariance R-matrix scope too, not only the EBE solve.
    pub(crate) fd: bool,
    /// `FitOptions::bloq_method`: `None` scores each model under its own method.
    pub(crate) bloq: Option<crate::types::BloqMethod>,
}

impl FitScope {
    pub(crate) fn of(options: &FitOptions) -> Self {
        Self {
            ode: options.ode_solver_override(),
            inner: options.inner_settings(),
            fd: options.forces_fd(),
            bloq: options.bloq_method,
        }
    }

    /// True when a worker carrying this scope installs nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.ode.is_empty() && self.inner.is_default() && !self.fd && self.bloq.is_none()
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        self.ode.validate()
    }

    /// Pool-key equality: the ODE half bit-compared (see
    /// [`OdeSolverOverride::same_pool_key`](crate::ode::solver::OdeSolverOverride)), the
    /// inner half, the forced-FD flag and the `bloq_method` override by value.
    pub(crate) fn same_pool_key(&self, other: &Self) -> bool {
        self.ode.same_pool_key(&other.ode)
            && self.inner == other.inner
            && self.fd == other.fd
            && self.bloq == other.bloq
    }

    /// `f` with every member armed on whichever thread runs it, for the duration of the call.
    ///
    /// Arm inside the closure an `install` moves, never on the calling thread before it. A
    /// Rayon worker blocked in a cross-pool `install` keeps running its own pool's queued jobs
    /// while it waits, and those jobs belong to *its* pool's scope: a guard armed on it across
    /// the `install` is what they would read (#1801 review, finding 1 — a job stolen during a
    /// default-options nested call on an `(Lbfgs, warm)` worker read `(Auto, false)`).
    fn armed<R>(self, f: impl FnOnce() -> R + Send) -> impl FnOnce() -> R + Send {
        // Test seam: the solver's forced linear decline follows `f` onto the worker that runs
        // it, so a one-thread scope sees what its caller set (#520 review round 3).
        #[cfg(test)]
        let declined = crate::ode::solver::FORCE_LINEAR_DECLINE.get();
        // A test recording inner solves on the calling thread follows the call onto the
        // thread that runs it (#2261).
        #[cfg(test)]
        let f = {
            let recording = crate::estimation::inner_optimizer::recording_handle();
            move || crate::estimation::inner_optimizer::with_recording_handle(recording, f)
        };
        // So is the warm-start predictor's test switch (#2549).
        #[cfg(test)]
        let f = {
            let off = crate::estimation::outer_optimizer::warm_predictor_off();
            move || {
                let _off = off.then(crate::estimation::outer_optimizer::WarmPredictorOff::new);
                f()
            }
        };
        move || {
            #[cfg(test)]
            let _declined = ForcedDeclineGuard::set(declined);
            let _ode = crate::ode::solver::arm_ode_solver_override(self.ode);
            let _inner = crate::estimation::inner_optimizer::arm_inner_settings(self.inner);
            let _fd = crate::types::arm_forced_fd(self.fd);
            let _bloq = crate::types::arm_bloq_override(self.bloq);
            f()
        }
    }

    /// Install this scope for a pool worker's whole lifetime (its `start_handler`).
    pub(super) fn install_on_worker(self) {
        if !self.ode.is_empty() {
            crate::ode::solver::install_worker_ode_override(self.ode);
        }
        if !self.inner.is_default() {
            crate::estimation::inner_optimizer::install_worker_inner_settings(self.inner);
        }
        if self.fd {
            crate::types::install_worker_forced_fd(true);
        }
        if self.bloq.is_some() {
            crate::types::install_worker_bloq_override(self.bloq);
        }
        WORKER_SCOPE.set(Some(self));
    }
}

/// Sets `FORCE_LINEAR_DECLINE` on the current thread and restores the prior value on drop.
#[cfg(test)]
struct ForcedDeclineGuard(bool);

#[cfg(test)]
impl ForcedDeclineGuard {
    fn set(on: bool) -> Self {
        Self(crate::ode::solver::FORCE_LINEAR_DECLINE.replace(on))
    }
}

#[cfg(test)]
impl Drop for ForcedDeclineGuard {
    fn drop(&mut self) {
        crate::ode::solver::FORCE_LINEAR_DECLINE.set(self.0);
    }
}

impl From<crate::ode::solver::OdeSolverOverride> for FitScope {
    fn from(ode: crate::ode::solver::OdeSolverOverride) -> Self {
        Self {
            ode,
            ..Default::default()
        }
    }
}

thread_local! {
    /// The scope this thread's pool installed at worker start. `None` on any thread that is
    /// not a worker of a fit pool built by `pool_cache::build_pool` — the shared default pool,
    /// a plain Rayon pool, a caller thread — all of which carry the empty scope. Distinct from
    /// what a guard has armed on this thread: the jobs a `par_iter` spawns run on this
    /// thread's *siblings*, which read what the pool installed, not this thread's guards.
    static WORKER_SCOPE: std::cell::Cell<Option<FitScope>> = const { std::cell::Cell::new(None) };
}

/// The scope the current thread's pool gave its workers (empty when there is none).
fn worker_scope() -> FitScope {
    WORKER_SCOPE.get().unwrap_or_default()
}

fn fit_pool_cache() -> &'static PoolCache {
    static CACHE: std::sync::OnceLock<PoolCache> = std::sync::OnceLock::new();
    // Ordinarily retain at most twice the automatic worker budget (at most 16
    // workers, 512 MiB reserved stack capacity). A single most-recent wider
    // pool is retained so explicit wide configurations still get reuse.
    CACHE.get_or_init(|| PoolCache::new(default_thread_count() * 2))
}

fn shared_override_pool_cache() -> &'static SharedPoolCache {
    static CACHE: std::sync::OnceLock<SharedPoolCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| SharedPoolCache::new(default_thread_count() * 2))
}

/// Lease a pool carrying exactly this scope. Idle pools are reused, but active
/// callers never share one: identical settings must not collapse a batch's
/// independently budgeted fits onto a single pool. Workers keep immutable ODE and
/// inner settings for their lifetime; eviction bounds ordinary idle workers across
/// all keys while retaining one most-recent oversized pool.
/// Acquisition never waits for a busy pool, including inside nested fits.
pub(crate) fn fit_scope_pool(
    scope: FitScope,
    n_threads: usize,
) -> Result<FitPoolLease<'static>, String> {
    fit_pool_cache().acquire(n_threads, scope)
}

fn shared_fit_scope_pool(
    scope: FitScope,
    n_threads: usize,
) -> Result<std::sync::Arc<rayon::ThreadPool>, String> {
    shared_override_pool_cache().acquire(n_threads, scope)
}

/// Run `f` with `options`' fit scope — its ODE solver settings, its inner-loop settings, its
/// `gradient = fd` and its `bloq_method` override —
/// reaching every thread that can solve for it: armed on whichever thread runs `f`, and run on
/// a pool whose workers carry the same value (see [`fit_scope_pool`]).
///
/// Used by the standalone `run_covariance` / `run_sir` / `run_sir_core` entry points, which
/// re-solve the inner loop exactly as `fit` does but may have no pool of their own. Runs
/// inline only when the current thread is a worker of a pool that already carries this exact
/// scope — `run_sir_core` reached from inside `fit` — so no second full-width pool is leased.
/// A worker carrying a *different* scope is left, also when the requested one is empty:
/// otherwise the `par_iter` jobs inside `f` would run on that worker's siblings and read the
/// enclosing call's settings (#426).
///
/// Refuses ODE solver settings no solver can run with as `E_ODE_SOLVER_OPTIONS` (#2044),
/// before any pool is leased. A pool that cannot be built is a runtime failure, uncoded.
pub(crate) fn with_fit_scope<R: Send>(
    options: &FitOptions,
    f: impl FnOnce() -> R + Send,
) -> Result<R, crate::diagnostics::EngineError> {
    with_scope(FitScope::of(options), options.threads, f)
}

/// [`with_fit_scope`] with `options`' ODE solver override replaced by `ode`, every other
/// member of the scope kept. For the stagnation guard's Newton decrement, which re-solves the
/// incumbent at a tighter integrator tolerance than the fit's own (#2001); `ode` is the whole
/// override, not a merge onto the fit's.
pub(crate) fn with_fit_scope_ode<R: Send>(
    options: &FitOptions,
    ode: crate::ode::solver::OdeSolverOverride,
    f: impl FnOnce() -> R + Send,
) -> Result<R, crate::diagnostics::EngineError> {
    let scope = FitScope {
        ode,
        ..FitScope::of(options)
    };
    with_scope(scope, options.threads, f)
}

fn with_scope<R: Send>(
    scope: FitScope,
    threads: Option<usize>,
    f: impl FnOnce() -> R + Send,
) -> Result<R, crate::diagnostics::EngineError> {
    use crate::diagnostics::EngineError;
    scope
        .validate()
        .map_err(super::entry_refusals::ode_solver_options)?;
    let requested_threads = threads
        .filter(|&n| n > 0)
        .unwrap_or_else(effective_default_threads);
    // `current_num_threads()` is only meaningful — and only safe to call — on a worker:
    // off one it reports (and *initializes*) Rayon's global pool, at one worker per logical
    // CPU, which is the pool this engine never wants to own (#1460).
    let carried = rayon::current_thread_index().is_some() && worker_scope().same_pool_key(&scope);
    let f = scope.armed(f);
    if scope.is_empty() {
        // Nothing for workers to carry, but `f` still `par_iter`s (SIR weighting, the
        // covariance step's per-subject passes). On a worker that also carries nothing, stay
        // there, keeping an enclosing tool's budget as `install_on_pool_sized` does; anywhere
        // else, a ferx pool of the width the caller asked for.
        return Ok(if carried {
            f()
        } else {
            install_on_ferx_pool_sized(requested_threads, f)
        });
    }
    if carried && rayon::current_num_threads() == requested_threads {
        return Ok(f());
    }
    match threads.filter(|&n| n > 0) {
        Some(n) => Ok(fit_scope_pool(scope, n)
            .map_err(EngineError::uncoded)?
            .install(f)),
        None => Ok(shared_fit_scope_pool(scope, requested_threads)
            .map_err(EngineError::uncoded)?
            .install(f)),
    }
}

/// Run population-wide parallel work that is **not** a `fit()` on the engine's worker pool.
///
/// For the parallel entry points a caller can reach directly — GAM covariate screening,
/// standalone `run_sir` / `run_covariance`, the NCA initial-estimate pass, `npde`. A bare
/// `par_iter` in those places runs on whatever pool the calling thread belongs to, and off
/// a Rayon worker that is Rayon's *global* pool: one thread per logical CPU, sized by
/// Rayon, ignoring the count the caller declared. Measured on a 15-core host after
/// `configure_global_thread_pool(2)`: 15 workers, not 2 (#1460).
///
/// Already on a worker → runs inline, so work reached from inside a fit (or from inside a
/// tool's replicate pool) keeps that budget instead of nesting a second pool underneath it.
pub fn install_on_engine_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    install_on_pool_sized(effective_default_threads(), f)
}

/// [`install_on_engine_pool`] at an explicit width, for a caller that pinned one.
///
/// Prefers the shared default pool when it is already that wide — the common case, where
/// this adds no threads at all — and otherwise leases from the same cache `fit()` uses, so
/// a pool just released by a fit is reused rather than doubled. Falls back to running
/// inline if no pool can be built: ambient parallelism is worse than the declared width,
/// but it is not worth failing the caller's work over.
pub(crate) fn install_on_pool_sized<R: Send>(n_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return f();
    }
    install_on_ferx_pool_sized(n_threads, f)
}

/// [`install_on_pool_sized`] without the stay-on-the-current-worker shortcut: always a pool
/// whose workers carry the empty scope.
fn install_on_ferx_pool_sized<R: Send>(n_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    match default_fit_pool() {
        Some(pool) if pool.current_num_threads() == n_threads => pool.install(f),
        _ => match fit_pool_cache().acquire(n_threads, FitScope::default()) {
            Ok(lease) => lease.install(f),
            Err(_) => f(),
        },
    }
}

/// Run `f` on the pool this `fit()` call should use, with its fit scope armed on the thread
/// that runs it (never on this one across the `install`; see `FitScope::armed`).
///
/// Pool choice, in order: a pool carrying this call's scope when it is not empty — an ODE
/// override, non-default inner settings, `gradient = fd` or a `bloq_method` override (shared for unpinned fits, exclusive for a positive
/// `threads` budget); an exclusively leased plain pool sized to a pinned `threads`; otherwise
/// the shared big-stack pool, falling back to the ambient one only if that one-time build
/// failed. One pool serves both levels of a multi-start fan-out, which is why this is called
/// once per `fit()` and not per start.
pub(crate) fn install_on_fit_pool<R: Send>(
    options: &FitOptions,
    f: impl FnOnce() -> R + Send,
) -> Result<R, crate::diagnostics::EngineError> {
    use crate::diagnostics::EngineError;
    let scope = FitScope::of(options);
    // ODE solver settings no solver can run with are `E_ODE_SOLVER_OPTIONS`, as in
    // `with_fit_scope` (#2044), so `fit()` carries the code too (#2062 review 2, F2).
    scope
        .validate()
        .map_err(super::entry_refusals::ode_solver_options)?;
    let f = scope.armed(f);
    let resolved = fit_thread_resolution(options.threads);
    let pinned = resolved.source == ThreadSource::Pinned;
    if !scope.is_empty() {
        return if pinned {
            Ok(fit_scope_pool(scope, resolved.n)
                .map_err(EngineError::uncoded)?
                .install(f))
        } else {
            Ok(shared_fit_scope_pool(scope, resolved.n)
                .map_err(EngineError::uncoded)?
                .install(f))
        };
    }
    if pinned {
        // A pinned positive `threads` leases an idle pool or builds a new one,
        // preserving independently budgeted concurrent fits.
        Ok(fit_pool_cache()
            .acquire(resolved.n, FitScope::default())
            .map_err(EngineError::uncoded)?
            .install(f))
    } else {
        Ok(match default_fit_pool() {
            Some(pool) => pool.install(f),
            None => f(),
        })
    }
}

/// The thread count an unpinned `fit()` actually runs on: the explicitly configured
/// process-wide width when the CLI called [`configure_global_thread_pool`], else a positive
/// `RAYON_NUM_THREADS` (#2800), otherwise the capped [`default_thread_count`] (#707).
///
/// [`default_fit_pool`] sizes itself with this, and [`PoolPlan::from_budget`] resolves a
/// `total_threads = 0` budget with it — the two must agree, or a `from_budget(0, …)` plan
/// would silently disagree with the pool an unpinned fit uses (#1115).
pub(crate) fn effective_default_threads() -> usize {
    effective_default_resolution().n
}

/// [`effective_default_threads`] together with where the count came from, for the fit's
/// thread warnings.
pub(crate) fn effective_default_resolution() -> DefaultThreads {
    fit_thread_resolution(None)
}

/// The worker count a `fit()` with this `FitOptions::threads` runs on, and who chose it.
/// [`install_on_fit_pool`] picks its pool from this and the fit's thread warnings name its
/// source, so the width a warning reports and the pool the fit leased cannot disagree.
pub(crate) fn fit_thread_resolution(threads: Option<usize>) -> DefaultThreads {
    resolve_fit_threads(
        threads,
        EXPLICIT_THREADS.load(std::sync::atomic::Ordering::Acquire),
        rayon_num_threads_env(),
        available_cpus(),
    )
}

/// The whole precedence for one fit, with every input injected (#2800):
///
/// 1. a positive `pinned` (`FitOptions::threads`: the R argument, `[fit_options] threads`,
///    or a CLI `--threads N` that overrode the file — #1416 merges those upstream);
/// 2. a process-wide width declared through [`configure_global_thread_pool`];
/// 3. a positive `RAYON_NUM_THREADS`;
/// 4. the capped default of `available`.
///
/// `pinned = Some(0)` is "the engine default" — the CLI's `--threads auto` and R's
/// `threads = 0` — so it falls through to 2–4 and the environment applies to it.
pub(crate) fn resolve_fit_threads(
    pinned: Option<usize>,
    explicit: usize,
    env: Option<&str>,
    available: usize,
) -> DefaultThreads {
    match pinned.filter(|&n| n > 0) {
        Some(n) => DefaultThreads {
            n,
            source: ThreadSource::Pinned,
            env_ignored: None,
        },
        None => resolve_default_threads(explicit, env, available),
    }
}

/// The environment variable an unpinned fit honours as a worker count (#2800).
pub(crate) const RAYON_NUM_THREADS_ENV: &str = "RAYON_NUM_THREADS";

/// `RAYON_NUM_THREADS` as this process first saw it, read **once**.
///
/// Once per process because the shared [`default_fit_pool`] is sized on first use and can
/// never be resized: re-reading the variable per fit would report a width no pool has — the
/// disagreement #1115 forbids. Rayon reads it once for its global pool for the same reason.
fn rayon_num_threads_env() -> Option<&'static str> {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    RAW.get_or_init(|| std::env::var(RAYON_NUM_THREADS_ENV).ok())
        .as_deref()
}

/// Who chose the worker count of a fit (#2800). Reported in the fit's thread warnings so a
/// user can see why a fit ran the width it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThreadSource {
    /// A positive `FitOptions::threads`: `[fit_options] threads`, the R `threads` argument,
    /// or a CLI `--threads N` overriding the file (#1416).
    Pinned,
    /// A process-wide width declared through [`configure_global_thread_pool`].
    Declared,
    /// A positive integer in `RAYON_NUM_THREADS`.
    Env,
    /// The engine default: cores minus one, at least two where two exist, capped at 8.
    Auto,
}

impl ThreadSource {
    /// The parenthetical the thread warnings carry.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            ThreadSource::Pinned => "pinned by `threads`",
            ThreadSource::Declared => "pinned process-wide by `--threads`",
            ThreadSource::Env => "from `RAYON_NUM_THREADS`",
            ThreadSource::Auto => "engine default",
        }
    }
}

/// The resolved worker count of an unpinned fit, where it came from, and the
/// `RAYON_NUM_THREADS` value that was set but could not be used, if any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DefaultThreads {
    pub(crate) n: usize,
    pub(crate) source: ThreadSource,
    /// The raw variable when it was set, non-empty, and not a non-negative integer — it was
    /// ignored and the fit must say so. `None` for a valid pin, for `0` (Rayon's own
    /// spelling of "the default", as `threads = 0` is ferx's) and for an empty value.
    pub(crate) env_ignored: Option<String>,
}

/// [`effective_default_resolution`] without the process-global reads, so the precedence is
/// testable with the environment injected — never `std::env::set_var`, which races every
/// other test in the binary.
///
/// Precedence: a declared process-wide width (`explicit > 0`), then a positive
/// `RAYON_NUM_THREADS`, then the capped default of `available`. An env pin is an explicit
/// request, so — like `explicit` — it is used as given: no "minus one", no cap at 8.
///
/// The variable is parsed exactly as Rayon parses it (`usize::from_str`, no trimming, `0`
/// meaning "the default"), so the fits and anything still sized by Rayon's global pool
/// (a bootstrap's replicate lanes) agree on what counts as a pin.
///
/// Deliberately *not* `rayon::current_num_threads()` on the explicit branch: that reads the
/// ambient pool, and reading it is what made `--threads N` build a global pool nobody ran on
/// (#1460). It also silently builds Rayon's global pool when called off-worker, so an
/// engine that never wants one must not ask.
pub(crate) fn resolve_default_threads(
    explicit: usize,
    env: Option<&str>,
    available: usize,
) -> DefaultThreads {
    let auto = |env_ignored| DefaultThreads {
        n: cap_default_threads(available),
        source: ThreadSource::Auto,
        env_ignored,
    };
    if explicit > 0 {
        return DefaultThreads {
            n: explicit,
            source: ThreadSource::Declared,
            env_ignored: None,
        };
    }
    match env {
        None | Some("") => auto(None),
        Some(raw) => match raw.parse::<usize>() {
            Ok(0) => auto(None),
            Ok(n) => DefaultThreads {
                n,
                source: ThreadSource::Env,
                env_ignored: None,
            },
            Err(_) => auto(Some(raw.to_string())),
        },
    }
}

/// The thread warnings `fit()` reports after its chain: [`fit_thread_resolution`] for this
/// call's `threads`, handed to [`thread_warnings`]. One function so the hand-off — which
/// source, which ignored env value — is the code [`fit_thread_warnings_with`] tests, not a
/// copy of it in `fit.rs` (#2813 review row 1).
pub(crate) fn fit_thread_warnings(
    threads: Option<usize>,
    n_used: usize,
    n_subjects: usize,
    saem: bool,
) -> Vec<String> {
    fit_thread_warnings_with(
        threads,
        EXPLICIT_THREADS.load(std::sync::atomic::Ordering::Acquire),
        rayon_num_threads_env(),
        available_cpus(),
        n_used,
        n_subjects,
        saem,
    )
}

/// [`fit_thread_warnings`] with every process-global input injected.
pub(crate) fn fit_thread_warnings_with(
    threads: Option<usize>,
    explicit: usize,
    env: Option<&str>,
    available: usize,
    n_used: usize,
    n_subjects: usize,
    saem: bool,
) -> Vec<String> {
    let r = resolve_fit_threads(threads, explicit, env, available);
    thread_warnings(
        n_used,
        r.source,
        r.env_ignored.as_deref(),
        available,
        n_subjects,
        saem,
    )
}

/// The thread warnings a fit reports after its chain (#2800, #1811), as a pure function
/// of what was measured so each sentence is testable without a fit.
///
/// `requested` is the resolved count and `source` who chose it; `n_used` is the width the
/// fit actually ran on (they differ only if the shared pool's build failed and the fit fell
/// back to the ambient pool). `env_ignored` is consulted only for an unpinned fit, the one
/// case in which the variable was read at all.
pub(crate) fn thread_warnings(
    n_used: usize,
    source: ThreadSource,
    env_ignored: Option<&str>,
    available: usize,
    n_subjects: usize,
    saem: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(raw) = env_ignored {
        out.push(format!(
            "{RAYON_NUM_THREADS_ENV} = \"{raw}\" is not a positive integer and was ignored; \
             the fit used the engine default of {n_used} thread(s)"
        ));
    }
    // An `Auto` count never exceeds `available` by construction, so only an explicit
    // request can oversubscribe. No clamp: a pin is honoured as given (#1811).
    if source != ThreadSource::Auto && n_used > available {
        let advice = match source {
            ThreadSource::Env => {
                format!("unset {RAYON_NUM_THREADS_ENV} or lower it to at most {available}")
            }
            // `threads = 0` resolves to the declared width, so it cannot lower it.
            ThreadSource::Declared => format!("lower --threads to at most {available}"),
            _ => format!("consider threads = 0 (engine default) or at most {available}"),
        };
        out.push(format!(
            "{n_used} threads requested ({}) exceed the {available} logical CPU(s) available, \
             so workers time-share the cores — {advice}",
            source.describe()
        ));
    }
    if n_subjects > 0 && n_used > n_subjects {
        // `threads = 0` is not a valid Rayon pool size, so for n_subjects = 1
        // we still suggest a 1-thread pool.
        let suggested = n_subjects.max(1);
        out.push(format!(
            "{n_used} threads configured ({}) but only {n_subjects} subject(s) — consider \
             threads = {suggested} to reduce scheduling overhead (no speed benefit beyond \
             n_subjects)",
            source.describe()
        ));
    }
    // SAEM-specific: MH scheduling has higher per-subject overhead than FOCE.
    // Skip when n_subjects < 2 (n_subjects/2 = 0 is meaningless and the prior
    // warning already covers the n_threads > n_subjects case).
    if saem && n_subjects >= 2 {
        let suggested = (n_subjects / 2).max(1);
        if n_used > suggested {
            out.push(format!(
                "SAEM with more threads than subjects/2 may be slower due to MH scheduling \
                 overhead. Consider threads = {suggested} for SAEM."
            ));
        }
    }
    out
}

/// Whether a cheap population-wide pass has enough work to repay Rayon dispatch.
///
/// This is deliberately limited to cheap loops such as evaluating an already-converged
/// FOCE likelihood. It
/// must not gate inner EBE solves, ODE sensitivities, or finite differences: one subject in
/// those loops can contain enough work to benefit from every available worker only when the
/// algorithm exposes finer-grained parallelism.
///
/// Eight subjects per worker are required: below that granularity the fixed scheduling/join
/// cost dominates the small closed-form fixtures used by interactive and test callers. The
/// threshold scales with pool width and leaves expensive subject kernels free to use their
/// existing parallel paths.
pub(crate) fn parallelize_cheap_subject_pass(n_subjects: usize) -> bool {
    let workers = rayon::current_num_threads();
    workers > 1 && n_subjects >= workers.saturating_mul(8)
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
/// The outer pool is leased by [`install`](Self::install) with the ferx worker
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
    /// `total_threads = 0` means "the engine default" — whatever an unpinned
    /// `fit()` would run on, which is available-cores-minus-one capped at 8
    /// unless a CLI sized the process-wide pool via
    /// [`configure_global_thread_pool`], in which case that explicit width wins.
    /// `n_units = 0` yields a single-replicate plan.
    pub fn from_budget(total_threads: usize, n_units: usize) -> Self {
        let total = if total_threads == 0 {
            effective_default_threads()
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
    /// `replicates * threads_per_fit`, saturating rather than overflowing on an
    /// absurd hand-built plan.
    ///
    /// This counts the workers doing the fitting. Each replicate additionally
    /// *blocks* one outer worker while its inner pool runs, so the live OS
    /// thread count is `replicates + total_threads()`, each reserving
    /// [`FIT_RAYON_STACK_SIZE`] of stack address space — see
    /// [`apply_to`](Self::apply_to).
    pub fn total_threads(&self) -> usize {
        self.replicates.saturating_mul(self.threads_per_fit)
    }

    /// Worker-stack size of the outer pool [`install`](Self::install) leases —
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
    ///
    /// The pin is an upper bound on the whole `fit()` call, `n_starts > 1`
    /// included: a multi-start fit runs its start fan-out and the per-subject
    /// loops underneath it on the *same* pinned pool, so a plan's budget is not
    /// multiplied by `n_starts` behind the caller's back.
    ///
    /// The inner pool is a real Rayon pool even at width 1, and the outer worker
    /// that entered it stays blocked meanwhile — so `replicates` plan-wide costs
    /// `replicates` blocked threads on top of [`total_threads`](Self::total_threads)
    /// live workers, each reserving [`FIT_RAYON_STACK_SIZE`] of stack. Running the
    /// inner fit inline on the outer pool instead would be cheaper but wrong: the
    /// per-subject `par_iter` would then fan out over the *outer* pool's
    /// `replicates` workers, which is exactly the nesting the pin removes.
    pub fn apply_to(&self, options: &mut FitOptions) {
        options.threads = Some(self.threads_per_fit);
    }

    /// Run `op` on an exclusively leased outer Rayon pool of
    /// [`replicates`](Self::replicates) workers, each with the ferx worker stack
    /// ([`FIT_RAYON_STACK_SIZE`]). Idle pools are reused within a bounded cache.
    /// The lease lasts until `op` returns. `op` must not let work submitted with
    /// `rayon::spawn` / `spawn_fifo` outlive it: after return another caller may
    /// reuse the workers. Scoped work and parallel iterators satisfy this because
    /// they join before returning; these are also the forms used by ferx-tools.
    ///
    /// Any `par_iter` inside `op` fans out over that pool. Returns `Err` if the
    /// pool cannot be built (e.g. resource limits).
    pub fn install<R, OP>(&self, op: OP) -> Result<R, String>
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        Ok(fit_pool_cache()
            .acquire(self.replicates, Default::default())?
            .install(op))
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
