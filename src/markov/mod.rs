//! Matrix-exponential primitives for continuous-time Markov models (CTMM).
//!
//! Gated behind the `markov` cargo feature. **Zero new dependencies**: the
//! matrix exponential is nalgebra's built-in Higham (2005) scaling-and-squaring
//! Padé approximant (`DMatrix::exp`), and the parameter gradient is the exact
//! Van Loan (1978) block-matrix Fréchet derivative built on top of it. There is
//! no hand-rolled Padé here — `plans/tte-survival-markov.md` D4 predates the
//! discovery that nalgebra already ships a correct, well-tested `expm`.
//!
//! This module is a **pure numerical leaf**. It knows nothing about the model
//! DSL, random effects, covariates, or the outer/inner optimizer. It provides:
//!
//! - [`matrix_exp`] — `P(Δt) = expm(Q·Δt)`, the CTMM transition-probability
//!   matrix over an interval.
//! - [`matrix_exp_frechet`] — `L(A, E) = d/dt expm(A + tE)|₀`, the *exact*
//!   directional derivative of the matrix exponential (§3.4, §8.7, §9.7). This
//!   is what differentiates the CTMM likelihood w.r.t. rate parameters and
//!   random effects.
//! - [`generator_rate_direction`] — the constrained perturbation direction for
//!   one rate parameter of a generator (see "Why directional" below).
//! - [`ctmm_data_term`] — the individual CTMM negative log-likelihood
//!   `−Σ log P(Δtₘ)[sₘ, sₘ₊₁]` for a *prebuilt* generator, with a fail-loud
//!   guard layer: structural/data problems return [`MarkovError`]; a
//!   parameter-driven degeneracy returns the finite `1e20` sentinel (matching
//!   the `survival` module's convention) rather than a silent `−inf`.
//!
//! Building the generator `Q(η, θ)` from the model, the FOCEI/SAEM wiring, the
//! `Population → StateObs` data reader, and the `msm`-anchored validation are
//! **Phase 5** — see the plan. Because this leaf is a pure numerical primitive,
//! its comparator is not NONMEM/`msm` (there is no fit here to anchor); it is
//! validated in-repo, license-free, against the Taylor series, the closed-form
//! 2-state transition matrix, and the *exact* Daleckii–Krein Fréchet reference.
//!
//! ## Why a directional Fréchet, not per-entry gradients
//!
//! A CTMM generator obeys the row-sum-zero constraint: the diagonal is fixed by
//! the off-diagonals, `q_jj = −Σ_{k≠j} q_jk`. So a free **rate parameter**
//! `q_jk` (an off-diagonal, `j ≠ k`) perturbs *two* entries of `Q` at once:
//! `q_jk` up and `q_jj` down. The derivative of `expm(Q·Δt)` w.r.t. that rate is
//! therefore the Fréchet derivative in the direction `(E_jk − E_jj)·Δt`, **not**
//! the derivative w.r.t. a single independent matrix entry. An API that returns
//! "one gradient per `Q` entry" (as an earlier draft of §8.7 proposed) silently
//! computes the wrong thing for a constrained generator. [`matrix_exp_frechet`]
//! takes the direction matrix explicitly, and [`generator_rate_direction`]
//! constructs exactly the constrained direction, so the caller cannot get the
//! constraint wrong.

pub mod endpoint;

use nalgebra::DMatrix;

/// Finite sentinel for a numerically ill-defined individual likelihood.
///
/// Mirrors the `survival` module's `1e20` convention: a parameter guess that
/// makes an *observed* transition have (numerically) zero probability must
/// repel the optimizer with a large **finite** objective, never a `−inf` or a
/// panic that would abort the whole fit. Structural/data errors — which do not
/// depend on the parameters and cannot be optimized away — are reported as
/// [`MarkovError`] instead.
const SENTINEL_NLL: f64 = 1e20;

/// Upper bound on the magnitude of any entry of the scaled generator `Q·Δt`
/// before it is handed to `expm`.
///
/// nalgebra's `DMatrix::exp` forms internal matrix powers of its argument (up to
/// `A^8`/`A^10`) to choose a scaling factor. For an argument whose entries exceed
/// roughly `1e38`, those powers overflow to `+∞` **even though the argument
/// itself is finite**; the overflow drives nalgebra's internal squaring count to
/// `u64::MAX`, so `expm` then spins forever in a `for _ in 0..s` loop that never
/// returns — a **hang**, not a catchable panic, and not something the post-`expm`
/// finiteness check can rescue. A `Q·Δt` this large is a diverged parameter,
/// never a real CTMM regime (it is orders of magnitude beyond any physical
/// rate×time), so it is treated as the degenerate case and repels the optimizer.
/// The bound sits far below the overflow point (`1e18^10 = 1e180 ≪ f64::MAX`) and
/// far above any legitimate argument.
const MAX_EXP_ARG_ABS: f64 = 1e18;

/// A single continuous-time Markov state observation: the process was seen in
/// integer `state` at `time`. The CTMM likelihood consumes *consecutive* pairs
/// of these within a subject (`(ID, TIME, STATE)` data, no `EVID=3` rows —
/// §3.2 of the plan).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateObs {
    /// Observation time. Within a subject these must be non-decreasing.
    pub time: f64,
    /// Zero-based state index; must be `< S` for an `S`-state generator.
    pub state: usize,
}

/// Structural / data-invariant violations in a CTMM data term.
///
/// These are distinct from a numerically ill-defined likelihood (which returns
/// [`SENTINEL_NLL`]): a `MarkovError` does not depend on the parameter values
/// and signals a malformed generator or malformed data — a caller bug that no
/// amount of optimization can fix, so it fails loud instead of being buried in
/// the objective.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MarkovError {
    /// The generator matrix is not square.
    #[error("generator must be square; got {rows}×{cols}")]
    NonSquareGenerator { rows: usize, cols: usize },

    /// A CTMM needs at least two states.
    #[error("generator must have at least 2 states; got {n}")]
    TooFewStates { n: usize },

    /// An observation references a state index outside the generator.
    #[error("state index {state} out of range for {n_states}-state generator")]
    StateOutOfRange { state: usize, n_states: usize },

    /// An observation time is NaN or infinite.
    #[error("observation time at index {index} is not finite: {time}")]
    NonFiniteTime { index: usize, time: f64 },

    /// Observation times decreased (`Δt < 0`) between consecutive records.
    #[error("observation times must be non-decreasing; {prev} → {next} at index {index}")]
    TimeDecreased { index: usize, prev: f64, next: f64 },

    /// Two records share a timestamp (`Δt = 0`) but report different states —
    /// physically impossible and would make the likelihood `−inf`.
    #[error(
        "two distinct states ({from} → {to}) at identical time {time} (index {index}); \
         a state transition needs Δt > 0"
    )]
    ZeroDtStateChange {
        index: usize,
        time: f64,
        from: usize,
        to: usize,
    },
}

/// Transition-probability matrix `P(Δt) = expm(A)` for `A = Q·Δt`.
///
/// Thin, documented wrapper over nalgebra's Higham (2005) scaling-and-squaring
/// Padé approximant (`DMatrix::exp`). The caller passes the already-scaled
/// argument `A = Q·Δt`; for a valid generator `Q` (off-diagonals `≥ 0`,
/// row sums `0`) the result is a stochastic matrix (rows sum to 1, entries in
/// `[0, 1]`) up to floating round-off. **Round-off is not clamped here** — the
/// likelihood in [`ctmm_data_term`] inspects only the single selected entry and
/// treats a non-positive value as the degenerate case, so a global clamp would
/// add a code path without changing any result.
///
/// # Panics (debug builds only)
/// Debug-asserts that `a` is square. `expm` is only defined for square matrices;
/// nalgebra itself would panic on a non-square input.
#[must_use]
pub fn matrix_exp(a: &DMatrix<f64>) -> DMatrix<f64> {
    debug_assert!(a.is_square(), "matrix_exp requires a square matrix");
    // `DMatrix::exp` takes `&self` and clones internally; no extra copy here.
    a.exp()
}

/// Fréchet derivative of the matrix exponential in direction `E`:
/// `L(A, E) = d/dt expm(A + t·E)|_{t=0}`.
///
/// Computed with the Van Loan (1978) block identity — this is **exact** (up to
/// the exponential's own ~1e-15 round-off), not a finite difference:
///
/// ```text
///                 ⎛ ⎡ A  E ⎤ ⎞
///   L(A, E) = expm⎜ ⎢      ⎥ ⎟          ← upper-right S×S block
///                 ⎝ ⎣ 0  A ⎦ ⎠
/// ```
///
/// For a generator rate parameter, pass `E = (E_jk − E_jj)·Δt`
/// (see [`generator_rate_direction`]) and `A = Q·Δt`; then `L(A, E)` is exactly
/// `∂P(Δt)/∂q_jk`. `L` is linear in `E`, so directions compose:
/// `L(A, αE₁ + βE₂) = α·L(A, E₁) + β·L(A, E₂)`.
///
/// Cost: one `expm` of a `2S×2S` matrix per direction. Negligible for `S ≤ 10`.
///
/// # Panics (debug builds only)
/// Debug-asserts that `a` is square and that `e` has the same shape as `a`.
#[must_use]
pub fn matrix_exp_frechet(a: &DMatrix<f64>, e: &DMatrix<f64>) -> DMatrix<f64> {
    debug_assert!(a.is_square(), "matrix_exp_frechet requires a square A");
    debug_assert_eq!(
        a.shape(),
        e.shape(),
        "matrix_exp_frechet requires E to match A's shape"
    );
    let s = a.nrows();
    // Build the 2S×2S block matrix [[A, E], [0, A]].
    let mut c = DMatrix::<f64>::zeros(2 * s, 2 * s);
    c.view_mut((0, 0), (s, s)).copy_from(a);
    c.view_mut((s, s), (s, s)).copy_from(a);
    c.view_mut((0, s), (s, s)).copy_from(e);
    // The upper-right S×S block of expm(C) is L(A, E).
    matrix_exp(&c).view((0, s), (s, s)).into_owned()
}

/// The constrained perturbation direction for one **rate parameter** `q_jk` of
/// an `s`-state generator, scaled by `dt`: `(E_jk − E_jj)·dt`.
///
/// This encodes the row-sum-zero constraint (`q_jj = −Σ_{k≠j} q_jk`): raising
/// the off-diagonal rate `q_jk` by `δ` lowers the diagonal `q_jj` by `δ` so the
/// row still sums to zero. Feeding this direction (and `A = Q·dt`) to
/// [`matrix_exp_frechet`] yields exactly `∂P(dt)/∂q_jk`.
///
/// Keeping this as its own function means Phase 5 assembles gradients without
/// re-deriving — and cannot accidentally differentiate w.r.t. a bare matrix
/// entry, which would ignore the constraint.
///
/// # Panics
/// Panics if `j == k` (a rate is strictly off-diagonal) or if `j` or `k` is
/// `≥ s`. These are setup-time programming errors, checked unconditionally
/// (the function is called `O(#rate params)` times, not in the hot loop).
#[must_use]
pub fn generator_rate_direction(s: usize, j: usize, k: usize, dt: f64) -> DMatrix<f64> {
    assert!(
        j != k,
        "a generator rate parameter is off-diagonal; got j == k == {j}"
    );
    assert!(
        j < s && k < s,
        "rate index ({j},{k}) out of range for {s}-state generator"
    );
    let mut e = DMatrix::<f64>::zeros(s, s);
    e[(j, k)] = dt; //  +E_jk·dt
    e[(j, j)] = -dt; //  −E_jj·dt
    e
}

/// Individual CTMM negative log-likelihood for a **prebuilt** generator `q`:
/// `−Σ_m log P(Δt_m)[s_m, s_{m+1}]` over consecutive observation pairs.
///
/// The generator is taken as given (Phase 5 builds `Q(η, θ)`); this function is
/// the pure likelihood kernel plus its guard layer. It never returns `−inf` or
/// panics on bad input:
///
/// - **Structural / data errors** → [`MarkovError`] (fail loud): non-square or
///   too-small generator, a state index out of range, a non-finite time, times
///   that decrease, or two different states reported at the same instant.
/// - **Numerically degenerate likelihood** → `Ok(`[`SENTINEL_NLL`]`)`: a
///   parameter guess under which an *observed* transition has non-positive
///   (underflowed) probability. This is a large finite objective that repels the
///   optimizer, exactly as the `survival` module does with its `1e20` sentinel.
/// - Fewer than two observations carry no transition, so the term is `0.0`.
/// - A zero-length interval between two *identical* states contributes `0`
///   (`P(0) = I`, `log 1 = 0`); the same interval between *different* states is
///   the [`MarkovError::ZeroDtStateChange`] error above.
///
/// `q` must be a valid generator; validity of its *entries* (off-diagonals
/// `≥ 0`, rows summing to `0`) is the model builder's responsibility and is not
/// re-checked per-evaluation here.
pub fn ctmm_data_term(q: &DMatrix<f64>, obs: &[StateObs]) -> Result<f64, MarkovError> {
    if !q.is_square() {
        return Err(MarkovError::NonSquareGenerator {
            rows: q.nrows(),
            cols: q.ncols(),
        });
    }
    let s = q.nrows();
    if s < 2 {
        return Err(MarkovError::TooFewStates { n: s });
    }

    // Validate every state index and every time up front, so a malformed record
    // fails loud regardless of where it sits (not only if the loop reaches it).
    for (i, o) in obs.iter().enumerate() {
        if o.state >= s {
            return Err(MarkovError::StateOutOfRange {
                state: o.state,
                n_states: s,
            });
        }
        if !o.time.is_finite() {
            return Err(MarkovError::NonFiniteTime {
                index: i,
                time: o.time,
            });
        }
    }

    if obs.len() < 2 {
        return Ok(0.0);
    }

    let mut nll = 0.0;
    for (i, pair) in obs.windows(2).enumerate() {
        let (a, b) = (pair[0], pair[1]);
        let dt = b.time - a.time;

        if dt < 0.0 {
            return Err(MarkovError::TimeDecreased {
                index: i + 1,
                prev: a.time,
                next: b.time,
            });
        }
        if dt == 0.0 {
            if a.state != b.state {
                return Err(MarkovError::ZeroDtStateChange {
                    index: i + 1,
                    time: b.time,
                    from: a.state,
                    to: b.state,
                });
            }
            // Δt = 0, same state: P(0) = I ⇒ log P[s,s] = log 1 = 0.
            continue;
        }

        // Guard the exponential's argument before it reaches nalgebra's `expm`.
        // Two failure modes, both from a rate that diverged during optimization:
        //   • a non-finite entry (∞/NaN, e.g. an infinite rate from a bad η) —
        //     `expm`'s internal LU solve `.unwrap()` can *panic* on it; and
        //   • a finite-but-enormous entry (|q·dt| ≳ 1e38) — `expm` forms internal
        //     matrix powers that overflow to ∞, driving its scaling count to
        //     `u64::MAX` so it *hangs* in an unbounded squaring loop.
        // A hang survives neither the post-`expm` finiteness check nor
        // `catch_unwind`, so both are rejected here up front. Either is the
        // degenerate case (repel the optimizer), not a crash or a frozen fit.
        let arg = q * dt;
        if arg
            .iter()
            .any(|x| !x.is_finite() || x.abs() > MAX_EXP_ARG_ABS)
        {
            return Ok(SENTINEL_NLL);
        }
        let p_mat = matrix_exp(&arg);
        let p = p_mat[(a.state, b.state)];
        // Underflow / non-positive probability for an *observed* transition:
        // this is the degenerate case, not a hard error — repel the optimizer.
        if !p.is_finite() || p <= 0.0 {
            return Ok(SENTINEL_NLL);
        }
        // A valid generator gives P ∈ [0, 1]; clamp to 1 so floating round-off on
        // a near-certain transition (p = 1 + ε) contributes ~0 rather than a
        // spurious *negative* NLL, and a malformed generator (a P entry > 1)
        // cannot be *rewarded* with a negative term. Entry-level generator
        // validity stays the builder's responsibility (not re-checked here).
        nll -= p.min(1.0).ln();
    }
    Ok(nll)
}

/// Time-**inhomogeneous** transition-probability matrix `P(Δt)` for a generator
/// that varies within the interval — the drug-driven CTMM (`Q(t) = f(C(t))`),
/// Phase 6, Track D (#817).
///
/// Solves the matrix occupancy ODE (forward Kolmogorov, right multiplication)
///
/// ```text
///   dP/dτ = P·Q(τ),   P(0) = I,   τ ∈ [0, Δt]
/// ```
///
/// so `P(Δt)[from, to] = Pr(state = to at Δt | state = from at 0)`, matching the
/// homogeneous `matrix_exp(Q·Δt)` indexing. (The backward form `Q(τ)·P` gives the
/// transpose-ordered result and is only equal for a generator that commutes across
/// time — constant `Q`, or `Q(τ)=g(τ)·Q₀`.)
///
/// on the shared Dormand–Prince RK45 engine ([`crate::ode::solve_ode`]) and
/// returns `P(Δt)`. `q_at(τ)` supplies the `n_states × n_states` generator at
/// **elapsed time `τ` within the interval** (`0` at the left endpoint); the
/// caller maps `τ` to an absolute model time and hence to the PK state / drug
/// concentration that drives `Q`. Unlike the time-homogeneous
/// [`ctmm_data_term`], there is no closed form here (the matrix exponential
/// `expm(Q·Δt)` is exact only when `Q` is constant), so the interval transition
/// is obtained by integration.
///
/// **Reduction to the matrix exponential.** When `q_at` returns a *constant* `Q`,
/// the time-ordered solution collapses to `P(Δt) = expm(Q·Δt)` — this is the
/// Tier-1 anchor (`ctmm_inhomogeneous_matches_matrix_exp_for_constant_q`), so the
/// homogeneous and inhomogeneous paths agree in the limit they share. More
/// generally, if `Q(τ) = g(τ)·Q₀` for a scalar `g` and a fixed `Q₀` (the family
/// commutes across time), `P(Δt) = expm(Q₀ · ∫₀^{Δt} g)` — a second closed-form
/// anchor independent of `expm`'s own definition.
///
/// The likelihood term is unchanged: `−Σ_m log P(Δt_m)[s_m, s_{m+1}]`, with each
/// `P(Δt_m)` from this function instead of `matrix_exp`.
///
/// The occupancy matrix is flattened **row-major** into the ODE state vector
/// (length `n_states²`); the RHS reshapes it, forms `P·Q(τ)`, and flattens back.
/// A non-positive `delta_t` returns the identity (`P(0) = I`; a zero-length gap
/// between identical states contributes `log 1 = 0`, exactly as the homogeneous
/// kernel treats it).
///
/// # Panics (debug builds only)
/// Debug-asserts that `q_at(0)` is `n_states × n_states`.
///
/// Integrates to near machine precision (`reltol 1e-10`) — the default for the
/// Tier-1 closed-form anchors. The **fit path** instead uses
/// [`ctmm_inhomogeneous_transition_with_opts`] with the model's own ODE tolerance
/// (the same `reltol` the PK/TTE solves use), since integrating the whole
/// likelihood to `1e-10` on every EBE/FD evaluation is needlessly slow.
#[must_use]
pub fn ctmm_inhomogeneous_transition(
    q_at: impl Fn(f64) -> DMatrix<f64>,
    delta_t: f64,
    n_states: usize,
) -> DMatrix<f64> {
    let opts = crate::ode::OdeSolverOptions {
        abstol: 1e-12,
        reltol: 1e-10,
        ..Default::default()
    };
    ctmm_inhomogeneous_transition_with_opts(q_at, delta_t, n_states, &opts)
}

/// [`ctmm_inhomogeneous_transition`] with explicit solver options — the fit path
/// passes the model's ODE `solver_opts` so the occupancy integration uses the same
/// accuracy as the PK solve that drives it (the user's `TOL`), keeping the
/// per-evaluation cost in line with the rest of the ODE machinery.
#[must_use]
pub fn ctmm_inhomogeneous_transition_with_opts(
    q_at: impl Fn(f64) -> DMatrix<f64>,
    delta_t: f64,
    n_states: usize,
    opts: &crate::ode::OdeSolverOptions,
) -> DMatrix<f64> {
    let s = n_states;
    if delta_t <= 0.0 || delta_t.is_nan() {
        // P(0) = I; also the guard for a NaN/negative gap (validated upstream).
        return DMatrix::identity(s, s);
    }
    debug_assert_eq!(
        q_at(0.0).shape(),
        (s, s),
        "q_at must return an n_states×n_states generator"
    );

    // Flattened matrix ODE: u ↔ P row-major (u[i*s + j] = P[i, j]).
    // Forward Kolmogorov `dP/dt = P·Q(t)` (right multiplication), so
    // `P[from, to] = Pr(state=to at Δt | state=from at 0)` — the same indexing the
    // homogeneous `matrix_exp(Q·dt)` path yields. Using the backward form `Q·P`
    // would only coincide for a generator that commutes across time (constant Q,
    // or a scalar-scaled Q(τ)=g(τ)·Q₀); it is wrong for a genuinely non-commuting
    // time-varying generator. `params` is unused — `q_at` captures what it needs.
    let rhs = |u: &[f64], _params: &[f64], t: f64, du: &mut [f64]| {
        let p = DMatrix::from_row_slice(s, s, u);
        let dp = p * q_at(t);
        for i in 0..s {
            for j in 0..s {
                du[i * s + j] = dp[(i, j)];
            }
        }
    };

    // u0 = flattened identity.
    let mut u0 = vec![0.0_f64; s * s];
    for i in 0..s {
        u0[i * s + i] = 1.0;
    }

    let sol = crate::ode::solve_ode(&rhs, &u0, (0.0, delta_t), &[], &[delta_t], opts);

    match sol.last() {
        Some(pt) => DMatrix::from_row_slice(s, s, &pt.u),
        // The solver always returns at least the final save; identity is the
        // conservative fallback (never observed) rather than a panic in a hot loop.
        None => DMatrix::identity(s, s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    // ---- test helpers -----------------------------------------------------

    /// Reference exponential via truncated Taylor series Σ A^k/k!, k = 0..=n.
    fn series_exp(a: &DMatrix<f64>, n: usize) -> DMatrix<f64> {
        let s = a.nrows();
        let mut term = DMatrix::<f64>::identity(s, s);
        let mut acc = DMatrix::<f64>::identity(s, s);
        for k in 1..=n {
            term = (&term * a) / (k as f64);
            acc += &term;
        }
        acc
    }

    /// Closed-form 2-state generator transition matrix for `Q=[[-a,a],[b,-b]]`.
    fn exact_2state(a: f64, b: f64, t: f64) -> DMatrix<f64> {
        let lam = a + b;
        let e = (-lam * t).exp();
        DMatrix::from_row_slice(
            2,
            2,
            &[
                (b + a * e) / lam,
                (a - a * e) / lam,
                (b - b * e) / lam,
                (a + b * e) / lam,
            ],
        )
    }

    /// Closed-form transition matrix for the acyclic 3-state chain `0 → 1 → 2`
    /// (state 2 absorbing) with `Q = [[-a, a, 0], [0, -b, b], [0, 0, 0]]`. Derived
    /// by solving the Kolmogorov forward ODE directly (occupancy of a two-stage
    /// process), so it is **independent of `expm`'s own `Σ Aᵏ/k!` definition** —
    /// unlike `series_exp`.
    ///
    /// Only valid for `a != b`: the `a/(b−a)` term is singular at `a == b`, where
    /// the occupancy has a *different* (confluent) closed form `P01 = a·t·e^{-at}`
    /// — see `expm_matches_closed_form_3state_confluent`. The helper asserts this
    /// rather than silently returning `NaN`, matching the module's fail-loud stance.
    fn exact_seq3(a: f64, b: f64, t: f64) -> DMatrix<f64> {
        assert!(
            (a - b).abs() > 1e-12,
            "exact_seq3 requires a != b (singular a/(b−a) term); got a == b == {a}"
        );
        let (ea, eb) = ((-a * t).exp(), (-b * t).exp());
        let p00 = ea;
        let p01 = a / (b - a) * (ea - eb);
        let p02 = 1.0 - p00 - p01;
        let p12 = 1.0 - eb;
        DMatrix::from_row_slice(
            3,
            3,
            &[
                p00, p01, p02, // row 0: from state 0
                0.0, eb, p12, // row 1: from state 1
                0.0, 0.0, 1.0, // row 2: absorbing
            ],
        )
    }

    /// Build a valid 3-state generator from off-diagonal rates (diagonal filled
    /// to make each row sum to zero).
    fn generator_3() -> DMatrix<f64> {
        let mut q = DMatrix::from_row_slice(3, 3, &[0.0, 0.5, 0.2, 0.3, 0.0, 0.4, 0.1, 0.6, 0.0]);
        for i in 0..3 {
            let rs: f64 = q.row(i).sum();
            q[(i, i)] = -rs;
        }
        q
    }

    fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    // ---- matrix_exp: value ------------------------------------------------

    #[test]
    fn expm_matches_series_and_closed_form_2state() {
        let (a, b, dt) = (0.7, 0.3, 2.0);
        let q = DMatrix::from_row_slice(2, 2, &[-a, a, b, -b]);
        let qdt = &q * dt;
        let p = matrix_exp(&qdt);
        assert!(max_abs_diff(&p, &exact_2state(a, b, dt)) < 1e-13);
        assert!(max_abs_diff(&p, &series_exp(&qdt, 40)) < 1e-12);
    }

    #[test]
    fn expm_matches_series_3state_dense() {
        let qdt = generator_3() * 1.5;
        assert!(max_abs_diff(&matrix_exp(&qdt), &series_exp(&qdt, 50)) < 1e-12);
    }

    #[test]
    fn expm_matches_closed_form_3state_chain() {
        // Independent analytic anchor for the 3-state case: the acyclic 0→1→2
        // chain has a closed-form P(t) from the Kolmogorov ODE, NOT from expm's own
        // Σ Aᵏ/k! definition. `expm_matches_series_3state_dense` only compares
        // against that series; this pins the 3-state result to a separate source.
        let (a, b, t) = (0.6, 0.9, 1.75);
        let q = DMatrix::from_row_slice(3, 3, &[-a, a, 0.0, 0.0, -b, b, 0.0, 0.0, 0.0]);
        let reference = exact_seq3(a, b, t);

        // Validate the reference *on its own terms* first — properties checkable
        // without touching expm — so the cross-check below is "two independently
        // valid objects agree", not "trust the hand-derived algebra":
        //   • each row is a probability distribution over next states (sums to 1);
        //   • the acyclic 0→1→2 structure forbids every back-transition, so the
        //     strict lower triangle is exactly zero and state 2 is absorbing.
        for i in 0..3 {
            assert!(
                (reference.row(i).sum() - 1.0).abs() < 1e-14,
                "reference row {i} must sum to 1"
            );
        }
        assert_eq!(reference[(1, 0)], 0.0, "no 1→0 transition");
        assert_eq!(reference[(2, 0)], 0.0, "no 2→0 transition");
        assert_eq!(reference[(2, 1)], 0.0, "no 2→1 transition");
        assert!(
            (reference[(2, 2)] - 1.0).abs() < 1e-15,
            "state 2 is absorbing"
        );

        // Now the anchor itself: expm agrees with the independent closed form.
        let p = matrix_exp(&(&q * t));
        assert!(max_abs_diff(&p, &reference) < 1e-12);
        // A generator's exponential is stochastic: rows sum to 1.
        for i in 0..3 {
            assert!((p.row(i).sum() - 1.0).abs() < 1e-13, "row {i} sums to 1");
        }
    }

    #[test]
    fn expm_matches_closed_form_3state_confluent() {
        // The a == b case is the *defective* generator: Q has one repeated
        // eigenvalue (−a, algebraic multiplicity 2, a single 2×2 Jordan block).
        // An eigendecomposition-based expm would divide by the zero eigenvalue gap
        // and return NaN here; nalgebra's scaling-and-squaring Padé has no such
        // failure mode. Pin it to the *confluent limit* of the occupancy solution
        // (l'Hôpital of a/(b−a)·(e^{-at}−e^{-bt}) as b → a):
        //   P00 = e^{-at},  P01 = a·t·e^{-at} (Erlang-2),  P02 = 1 − P00 − P01,
        //   P11 = e^{-at},  P12 = 1 − e^{-at},             P22 = 1.
        // This is exactly the regime `exact_seq3`'s a != b form cannot represent,
        // so it is a genuinely separate anchor, not a happy-path repeat.
        let (a, t) = (0.7, 1.3);
        let q = DMatrix::from_row_slice(3, 3, &[-a, a, 0.0, 0.0, -a, a, 0.0, 0.0, 0.0]);
        let p = matrix_exp(&(&q * t));
        // Named like `exact_seq3` so the matrix literal stays row-grouped.
        let ea = (-a * t).exp();
        let p00 = ea;
        let p01 = a * t * ea; // Erlang-2: lim_{b→a} a/(b−a)·(e^{-at} − e^{-bt})
        let p02 = 1.0 - p00 - p01;
        let p12 = 1.0 - ea;
        let confluent = DMatrix::from_row_slice(
            3,
            3,
            &[
                p00, p01, p02, // row 0: from state 0
                0.0, ea, p12, // row 1: from state 1
                0.0, 0.0, 1.0, // row 2: absorbing
            ],
        );
        assert!(
            max_abs_diff(&p, &confluent) < 1e-12,
            "defective (a == b) generator must match the confluent limit form"
        );
        for i in 0..3 {
            assert!((p.row(i).sum() - 1.0).abs() < 1e-13, "row {i} sums to 1");
        }
    }

    #[test]
    #[should_panic(expected = "a != b")]
    fn exact_seq3_rejects_equal_rates() {
        // The closed form is singular at a == b (division by b − a = 0). The helper
        // must fail loud, never silently return NaN — the confluent regime has its
        // own limit form, anchored in `expm_matches_closed_form_3state_confluent`.
        let _ = exact_seq3(0.5, 0.5, 1.0);
    }

    #[test]
    fn expm_zero_is_identity_exactly() {
        let z = DMatrix::<f64>::zeros(3, 3);
        assert_eq!(matrix_exp(&z), DMatrix::identity(3, 3));
    }

    // ---- matrix_exp: invariants of a generator ----------------------------

    #[test]
    fn expm_of_generator_is_stochastic() {
        let p = matrix_exp(&(generator_3() * 1.5));
        for i in 0..3 {
            assert!(
                (p.row(i).sum() - 1.0).abs() < 1e-13,
                "row {i} must sum to 1"
            );
        }
        assert!(p.iter().all(|&x| x >= 0.0), "entries must be non-negative");
        assert!(p.iter().all(|&x| x <= 1.0 + 1e-13), "entries must be ≤ 1");
    }

    #[test]
    fn expm_semigroup_inverse_identity() {
        let a = generator_3() * 1.5;
        let prod = matrix_exp(&a) * matrix_exp(&(-&a));
        assert!(max_abs_diff(&prod, &DMatrix::identity(3, 3)) < 1e-11);
    }

    #[test]
    fn expm_det_equals_exp_trace() {
        let a = generator_3() * 1.5;
        assert!((matrix_exp(&a).determinant() - a.trace().exp()).abs() < 1e-13);
    }

    #[test]
    fn expm_large_norm_rows_still_sum_to_one() {
        // ‖A‖ ~ 1e4 with a near-absorbing structure — the scaling-and-squaring
        // regime. Rows must still sum to 1 even where individual entries
        // underflow to exactly 0.
        let mut q = DMatrix::from_row_slice(
            3,
            3,
            &[-1000.0, 900.0, 100.0, 50.0, -60.0, 10.0, 0.0, 0.0, 0.0],
        );
        // (already a valid generator: rows sum to 0)
        let _ = &mut q;
        let p = matrix_exp(&(&q * 10.0));
        for i in 0..3 {
            assert!((p.row(i).sum() - 1.0).abs() < 1e-10);
        }
    }

    // ---- matrix_exp_frechet: exactness ------------------------------------

    /// Exact Daleckii–Krein reference for a **symmetric** `A` (real eigenvalues,
    /// orthogonal eigenvectors): `L(A,E) = V (Ψ ∘ VᵀEV) Vᵀ`, with the
    /// divided-difference matrix `Ψ_ij = (e^{λi} − e^{λj})/(λi − λj)` (and
    /// `e^{λi}` on the diagonal). Independent of both `matrix_exp` internals and
    /// finite differences, so it pins the Van Loan block trick to ~1e-13.
    #[test]
    fn frechet_matches_daleckii_krein_exact() {
        let a = DMatrix::from_row_slice(3, 3, &[0.4, 0.1, -0.2, 0.1, -0.3, 0.05, -0.2, 0.05, 0.6]);
        assert!(
            (a.clone() - a.transpose()).abs().max() < 1e-15,
            "A symmetric"
        );
        let e = DMatrix::from_row_slice(
            3,
            3,
            &[0.11, -0.22, 0.05, 0.07, 0.13, -0.09, -0.04, 0.02, 0.15],
        );

        let se = a.clone().symmetric_eigen();
        let lam = &se.eigenvalues;
        let v = &se.eigenvectors;
        let vt_e_v = v.transpose() * &e * v;
        let mut psi = DMatrix::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let li: f64 = lam[i];
                let lj: f64 = lam[j];
                psi[(i, j)] = if (li - lj).abs() < 1e-9 {
                    li.exp()
                } else {
                    (li.exp() - lj.exp()) / (li - lj)
                };
            }
        }
        let dk = v * psi.component_mul(&vt_e_v) * v.transpose();
        assert!(max_abs_diff(&matrix_exp_frechet(&a, &e), &dk) < 1e-13);
    }

    #[test]
    fn frechet_is_linear_in_direction() {
        let a = generator_3() * 1.5;
        let e = DMatrix::from_row_slice(
            3,
            3,
            &[0.11, -0.22, 0.05, 0.07, 0.13, -0.09, -0.04, 0.02, 0.15],
        );
        let l = matrix_exp_frechet(&a, &e);
        let l2 = matrix_exp_frechet(&a, &(&e * 2.0));
        assert!(max_abs_diff(&l2, &(&l * 2.0)) < 1e-11);
    }

    #[test]
    fn frechet_matches_central_fd() {
        let a = generator_3() * 1.5;
        let e = DMatrix::from_row_slice(
            3,
            3,
            &[0.11, -0.22, 0.05, 0.07, 0.13, -0.09, -0.04, 0.02, 0.15],
        );
        let h = 1e-6;
        let fd = (matrix_exp(&(&a + &e * h)) - matrix_exp(&(&a - &e * h))) / (2.0 * h);
        assert!(max_abs_diff(&matrix_exp_frechet(&a, &e), &fd) < 1e-8);
    }

    // ---- generator_rate_direction + constrained gradient ------------------

    #[test]
    fn rate_direction_encodes_row_sum_zero_constraint() {
        let dir = generator_rate_direction(3, 0, 2, 1.5);
        let mut expected = DMatrix::<f64>::zeros(3, 3);
        expected[(0, 2)] = 1.5;
        expected[(0, 0)] = -1.5;
        assert_eq!(dir, expected);
        // Every row of the direction sums to zero (constraint preserved).
        for i in 0..3 {
            assert!(dir.row(i).sum().abs() < 1e-15);
        }
    }

    #[test]
    fn constrained_rate_gradient_matches_fd() {
        // d/da expm(Q(a)·dt) for the 2-state model, Q(a)=[[-a,a],[b,-b]].
        let (a, b, dt) = (0.7, 0.3, 2.0);
        let q = DMatrix::from_row_slice(2, 2, &[-a, a, b, -b]);
        let grad_vl = matrix_exp_frechet(&(&q * dt), &generator_rate_direction(2, 0, 1, dt));
        let ha = 1e-6;
        let qp = DMatrix::from_row_slice(2, 2, &[-(a + ha), a + ha, b, -b]) * dt;
        let qm = DMatrix::from_row_slice(2, 2, &[-(a - ha), a - ha, b, -b]) * dt;
        let grad_fd = (matrix_exp(&qp) - matrix_exp(&qm)) / (2.0 * ha);
        assert!(max_abs_diff(&grad_vl, &grad_fd) < 1e-8);
    }

    #[test]
    #[should_panic(expected = "off-diagonal")]
    fn rate_direction_rejects_diagonal() {
        let _ = generator_rate_direction(3, 1, 1, 1.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rate_direction_rejects_out_of_range() {
        let _ = generator_rate_direction(2, 0, 5, 1.0);
    }

    // ---- ctmm_data_term: value --------------------------------------------

    #[test]
    fn data_term_matches_hand_computed() {
        // 2-state chain, three observations with two transitions.
        let (a, b) = (0.7, 0.3);
        let q = DMatrix::from_row_slice(2, 2, &[-a, a, b, -b]);
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            },
            StateObs {
                time: 3.0,
                state: 1,
            },
        ];
        let p1 = exact_2state(a, b, 1.0)[(0, 1)];
        let p2 = exact_2state(a, b, 2.0)[(1, 1)];
        let expected = -(p1.ln()) - (p2.ln());
        let got = ctmm_data_term(&q, &obs).unwrap();
        assert!((got - expected).abs() < 1e-12, "got {got}, want {expected}");
    }

    #[test]
    fn data_term_matches_hand_computed_3state() {
        // 3-state acyclic chain 0→1→2 with two *off-diagonal* transitions (0→1
        // then 1→2) — richer than the 2-state case, and hand-computed against the
        // independent closed form `exact_seq3` (not against `matrix_exp` itself).
        let (a, b) = (0.6, 0.9);
        let q = DMatrix::from_row_slice(3, 3, &[-a, a, 0.0, 0.0, -b, b, 0.0, 0.0, 0.0]);
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            },
            StateObs {
                time: 2.5,
                state: 2,
            },
        ];
        let p01 = exact_seq3(a, b, 1.0)[(0, 1)]; // 0 → 1 over Δt = 1.0
        let p12 = exact_seq3(a, b, 1.5)[(1, 2)]; // 1 → 2 over Δt = 1.5
        let expected = -(p01.ln()) - (p12.ln());
        let got = ctmm_data_term(&q, &obs).unwrap();
        assert!((got - expected).abs() < 1e-12, "got {got}, want {expected}");
    }

    #[test]
    fn data_term_fewer_than_two_obs_is_zero() {
        let q = generator_3();
        assert_eq!(ctmm_data_term(&q, &[]).unwrap(), 0.0);
        assert_eq!(
            ctmm_data_term(
                &q,
                &[StateObs {
                    time: 0.0,
                    state: 1
                }]
            )
            .unwrap(),
            0.0
        );
    }

    #[test]
    fn data_term_zero_dt_same_state_contributes_nothing() {
        let (a, b) = (0.7, 0.3);
        let q = DMatrix::from_row_slice(2, 2, &[-a, a, b, -b]);
        let with_dup = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            },
            StateObs {
                time: 1.0,
                state: 1,
            }, // duplicate timestamp, same state
            StateObs {
                time: 3.0,
                state: 1,
            },
        ];
        let without = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            },
            StateObs {
                time: 3.0,
                state: 1,
            },
        ];
        let a1 = ctmm_data_term(&q, &with_dup).unwrap();
        let a2 = ctmm_data_term(&q, &without).unwrap();
        assert!((a1 - a2).abs() < 1e-15);
    }

    // ---- ctmm_data_term: degenerate → sentinel ----------------------------

    #[test]
    fn data_term_underflowed_transition_returns_sentinel() {
        // State 1 absorbing; P(dt)[0,0] = e^{-1000·dt} underflows to 0. Observing
        // 0 → 0 across dt=1 is a (numerically) impossible transition.
        let q = DMatrix::from_row_slice(2, 2, &[-1000.0, 1000.0, 0.0, 0.0]);
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 0,
            },
        ];
        assert_eq!(ctmm_data_term(&q, &obs).unwrap(), SENTINEL_NLL);
    }

    #[test]
    fn data_term_non_finite_generator_returns_sentinel() {
        // An infinite rate (e.g. a diverged η during optimization) must NOT reach
        // nalgebra's expm — whose internal LU-solve `.unwrap()` can panic on a
        // non-finite input. It is the degenerate case → sentinel, not a panic.
        let q = DMatrix::from_row_slice(2, 2, &[f64::NEG_INFINITY, f64::INFINITY, 0.0, 0.0]);
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            },
        ];
        assert_eq!(ctmm_data_term(&q, &obs).unwrap(), SENTINEL_NLL);
    }

    #[test]
    fn data_term_finite_huge_generator_returns_sentinel() {
        // Regression: a FINITE but enormous rate (|q·dt| ≳ 1e38) must not reach
        // nalgebra's `expm`. Its argument is all-finite (so the is_finite() check
        // alone lets it through), but `expm` overflows an internal matrix power to
        // ∞, sets its scaling count to u64::MAX, and HANGS forever in
        // `for _ in 0..s`. The magnitude guard treats it as the degenerate case so
        // the call returns promptly — this test would never terminate without it.
        let q = DMatrix::from_row_slice(2, 2, &[-1e40, 1e40, 0.0, 0.0]);
        // Precondition: the argument really is finite, so finiteness alone is not
        // enough to catch it — the magnitude bound is what does.
        assert!((&q * 1.0).iter().all(|x: &f64| x.is_finite()));
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            },
        ];
        assert_eq!(ctmm_data_term(&q, &obs).unwrap(), SENTINEL_NLL);
    }

    #[test]
    fn data_term_superstochastic_entry_never_negative() {
        // Regression: an invalid generator (positive diagonal ⇒ a P entry > 1)
        // must not produce a NEGATIVE nll via log(p) > 0, which would *reward* the
        // optimizer toward the malformed input. The `p.min(1.0)` clamp makes the
        // contribution exactly 0, never negative.
        let q = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 5.0]); // P[0,0] = e^5 > 1
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 0,
            },
        ];
        let nll = ctmm_data_term(&q, &obs).unwrap();
        assert!(nll >= 0.0, "nll must never be negative, got {nll}");
        assert_eq!(nll, 0.0, "clamped p = 1 ⇒ −log 1 = 0");
    }

    // ---- ctmm_data_term: structural errors are loud (check messages) ------

    #[test]
    fn data_term_rejects_non_square_generator() {
        let q = DMatrix::from_row_slice(2, 3, &[0.0; 6]);
        let err = ctmm_data_term(&q, &[]).unwrap_err();
        assert_eq!(err, MarkovError::NonSquareGenerator { rows: 2, cols: 3 });
        assert!(err.to_string().contains("2×3"));
    }

    #[test]
    fn data_term_rejects_too_few_states() {
        let q = DMatrix::from_row_slice(1, 1, &[0.0]);
        assert_eq!(
            ctmm_data_term(&q, &[]).unwrap_err(),
            MarkovError::TooFewStates { n: 1 }
        );
    }

    #[test]
    fn data_term_rejects_state_out_of_range() {
        let q = generator_3();
        let obs = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 3,
            }, // valid states are 0..=2
        ];
        let err = ctmm_data_term(&q, &obs).unwrap_err();
        assert_eq!(
            err,
            MarkovError::StateOutOfRange {
                state: 3,
                n_states: 3
            }
        );
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn data_term_rejects_non_finite_time() {
        let q = generator_3();
        // NaN: `NaN != NaN`, so match the variant + `is_nan()` rather than
        // `assert_eq!` (which can never hold for a NaN-bearing error).
        let obs_nan = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: f64::NAN,
                state: 1,
            },
        ];
        match ctmm_data_term(&q, &obs_nan).unwrap_err() {
            MarkovError::NonFiniteTime { index, time } => {
                assert_eq!(index, 1);
                assert!(time.is_nan());
            }
            other => panic!("expected NonFiniteTime, got {other:?}"),
        }
        // +inf is `PartialEq`-comparable and exercises the Display message.
        let obs_inf = [
            StateObs {
                time: 0.0,
                state: 0,
            },
            StateObs {
                time: f64::INFINITY,
                state: 1,
            },
        ];
        let err = ctmm_data_term(&q, &obs_inf).unwrap_err();
        assert_eq!(
            err,
            MarkovError::NonFiniteTime {
                index: 1,
                time: f64::INFINITY
            }
        );
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn data_term_rejects_decreasing_time() {
        let q = generator_3();
        let obs = [
            StateObs {
                time: 2.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 1,
            }, // goes backwards
        ];
        let err = ctmm_data_term(&q, &obs).unwrap_err();
        assert_eq!(
            err,
            MarkovError::TimeDecreased {
                index: 1,
                prev: 2.0,
                next: 1.0
            }
        );
        assert!(err.to_string().contains("non-decreasing"));
    }

    #[test]
    fn data_term_rejects_zero_dt_state_change() {
        let q = generator_3();
        let obs = [
            StateObs {
                time: 1.0,
                state: 0,
            },
            StateObs {
                time: 1.0,
                state: 2,
            }, // same instant, different state
        ];
        let err = ctmm_data_term(&q, &obs).unwrap_err();
        assert_eq!(
            err,
            MarkovError::ZeroDtStateChange {
                index: 1,
                time: 1.0,
                from: 0,
                to: 2
            }
        );
        assert!(err.to_string().contains("Δt > 0"));
    }

    // ---- ctmm_inhomogeneous_transition (Phase 6, #817) --------------------

    /// The defining reduction: a **constant** `Q(τ)` makes the time-ordered
    /// occupancy ODE collapse to `P(Δt) = expm(Q·Δt)`. The RK45-integrated
    /// transition must therefore match `matrix_exp` (the homogeneous path) — both
    /// the 2-state closed form and a dense 3-state generator.
    #[test]
    fn ctmm_inhomogeneous_matches_matrix_exp_for_constant_q() {
        // 2-state, closed form.
        let (a, b, dt) = (0.7, 0.3, 2.0);
        let q2 = DMatrix::from_row_slice(2, 2, &[-a, a, b, -b]);
        let p2 = ctmm_inhomogeneous_transition(|_t| q2.clone(), dt, 2);
        assert!(
            max_abs_diff(&p2, &exact_2state(a, b, dt)) < 1e-8,
            "constant-Q 2-state must match the closed form; got {p2}"
        );

        // 3-state dense generator vs. the exact matrix exponential.
        let q3 = generator_3();
        let p3 = ctmm_inhomogeneous_transition(|_t| q3.clone(), 1.5, 3);
        assert!(
            max_abs_diff(&p3, &matrix_exp(&(&q3 * 1.5))) < 1e-8,
            "constant-Q 3-state must match expm(Q·Δt)"
        );
        // A valid generator's transition matrix is stochastic (rows sum to 1).
        for i in 0..3 {
            assert!((p3.row(i).sum() - 1.0).abs() < 1e-8, "row {i} sums to 1");
        }
    }

    /// A genuinely time-varying but **scalar-commuting** generator `Q(τ) = g(τ)·Q₀`
    /// has the closed form `P(Δt) = expm(Q₀ · ∫₀^{Δt} g)` — the family commutes
    /// across time, so the time-ordered exponential reduces to an ordinary one.
    /// This anchors the integrator on a non-constant `Q` **independently of `expm`'s
    /// own reduction** (the integral is hand-computed), so it is not a repeat of the
    /// constant-Q test.
    #[test]
    fn ctmm_inhomogeneous_scalar_time_varying_matches_closed_form() {
        // Q₀ off-diagonals (0.7, 0.3); g(τ) = 1 + τ ⇒ ∫₀^{Δt}(1+τ)dτ = Δt + Δt²/2.
        let (a, b, dt) = (0.7_f64, 0.3, 2.0);
        let q0 = DMatrix::from_row_slice(2, 2, &[-a, a, b, -b]);
        let integral = dt + 0.5 * dt * dt; // 2 + 2 = 4
        let expected = exact_2state(a, b, integral); // expm(Q₀·∫g)

        let p = ctmm_inhomogeneous_transition(|t| (1.0 + t) * q0.clone(), dt, 2);
        assert!(
            max_abs_diff(&p, &expected) < 1e-7,
            "scalar-time-varying Q must match expm(Q₀·∫g); got {p}, want {expected}"
        );
        for i in 0..2 {
            assert!((p.row(i).sum() - 1.0).abs() < 1e-8, "row {i} sums to 1");
        }
    }

    /// A genuinely **non-commuting** time-varying generator: piecewise-constant
    /// `Q(τ) = Q₁` on `[0, h)`, `Q₂` on `[h, 2h)`, with `Q₁·Q₂ ≠ Q₂·Q₁`. The
    /// forward Kolmogorov solution `dP/dτ = P·Q(τ)`, `P(0)=I` is the *time-ordered*
    /// product `P(2h) = expm(Q₁·h) · expm(Q₂·h)` (left-to-right in time). This is the
    /// anchor the two commuting tests above cannot supply: the backward form
    /// `dP/dτ = Q(τ)·P` would instead give `expm(Q₂·h)·expm(Q₁·h)` (reversed order),
    /// which differs precisely because `Q₁,Q₂` do not commute — so this test fails
    /// unless the integrator uses the correct right-multiplication.
    #[test]
    fn ctmm_inhomogeneous_matches_time_ordered_product_for_noncommuting_q() {
        let h = 1.3_f64;
        let q1 = generator_3();
        // A second, structurally different generator that does not commute with q1.
        let q2 = {
            let mut q =
                DMatrix::from_row_slice(3, 3, &[0.0, 0.1, 0.7, 0.6, 0.0, 0.2, 0.4, 0.3, 0.0]);
            for i in 0..3 {
                let rs: f64 = q.row(i).sum();
                q[(i, i)] = -rs;
            }
            q
        };
        // Sanity: the generators genuinely do not commute, so forward vs backward
        // integration give different answers — the test has teeth.
        assert!(
            max_abs_diff(&(&q1 * &q2), &(&q2 * &q1)) > 1e-3,
            "test generators must not commute"
        );

        let e1 = matrix_exp(&(&q1 * h));
        let e2 = matrix_exp(&(&q2 * h));
        let expected = &e1 * &e2; // forward/right-multiplication time order
        let wrong = &e2 * &e1; // backward/left-multiplication order (must NOT match)

        let p = ctmm_inhomogeneous_transition(
            |t| if t < h { q1.clone() } else { q2.clone() },
            2.0 * h,
            3,
        );
        assert!(
            max_abs_diff(&p, &expected) < 1e-6,
            "non-commuting Q must match the time-ordered product expm(Q₁h)·expm(Q₂h); got {p}, want {expected}"
        );
        assert!(
            max_abs_diff(&p, &wrong) > 1e-3,
            "must NOT match the reversed (backward-Kolmogorov) product"
        );
        for i in 0..3 {
            assert!((p.row(i).sum() - 1.0).abs() < 1e-8, "row {i} sums to 1");
        }
    }

    /// A non-positive interval returns the identity (`P(0) = I`) without touching the
    /// solver — the zero-length-gap case the homogeneous kernel also treats as `log 1 = 0`.
    #[test]
    fn ctmm_inhomogeneous_zero_dt_is_identity() {
        // `q_at` is deliberately a panic: a zero-length gap must short-circuit before
        // evaluating the generator at all.
        let p = ctmm_inhomogeneous_transition(
            |_t| panic!("q_at must not be called for Δt = 0"),
            0.0,
            3,
        );
        assert_eq!(p, DMatrix::identity(3, 3));
    }
}
