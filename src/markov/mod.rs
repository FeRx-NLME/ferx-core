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
}
