//! `PkNum` — the small numeric interface a closed-form PK solution needs, so a
//! single generic implementation serves both the scalar prediction (`T = f64`)
//! and its exact sensitivities (`T = Dual2<N>`). No Enzyme, no codegen: the same
//! source is monomorphised for each numeric type.

use super::dual1::Dual1;
use super::dual_mixed::DualMixed;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Size above which a dual's arithmetic is called out of line rather than inlined at every
/// use. A wide dual's operation is a loop over tens to hundreds of lanes, far more work
/// than a call, while inlining it into every closed form, event walk and ODE step was most
/// of the binary (#486); narrow duals — small models' hot path — stay inlined. Decided
/// per type at compile time from `size_of::<Self>()`, so nested duals are covered too.
/// The narrow duals' operations are `#[inline]` hints, not `#[inline(always)]`: forcing them
/// into every kernel made LLVM's optimisation of the recorders most of the crate's compile
/// time (610 s vs 298 s for the lib, and 84 vs 37 MB of text, measured on #2093).
pub(crate) const OUTLINE_BYTES: usize = 256;

/// Direction count from which a dual is outlined whatever its size: every `N ≥ 8` bucket
/// serves a rare wide fit, and inlining its lane loops cost 4–7 MB of binary per
/// instantiation where the outlined copy costs ~0.5 MB (measured on the release binary,
/// #2093 review). Below it, [`OUTLINE_BYTES`] decides.
pub(crate) const OUTLINE_DIRECTIONS: usize = 8;

/// Whether a dual with `n` directions and `size` bytes runs its arithmetic out of line.
pub(crate) const fn outlined(n: usize, size: usize) -> bool {
    n >= OUTLINE_DIRECTIONS || size > OUTLINE_BYTES
}

/// The largest `|x|` over `xs`, or `NaN` when any is `NaN` — the fold behind every
/// [`PkNum::jet_max_abs`] override, so a `NaN` jet fails the comparison it feeds rather than
/// being discarded by `f64::max`.
pub(crate) fn nan_max_abs(xs: impl IntoIterator<Item = f64>) -> f64 {
    xs.into_iter().fold(0.0_f64, |m, x| {
        if x.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

pub trait PkNum:
    Copy
    // Every implementor is a plain numeric value type (`f64`, `Dual1`, `Dual2`, `DualMixed`)
    // that borrows nothing, so `'static` costs nothing and lets a `PkNum` be boxed behind a
    // trait object — how `crate::ode::solver::make_stepper` hands the shared integration
    // drivers whichever stepper the method selects.
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Whether this type carries a second-order (Hessian) jet. `false` for `f64` and the
    /// gradient-only `Dual1`, `true` for the `Dual2`/`DualMixed` outer type. Lets a caller
    /// skip building a Hessian it would discard (e.g. `tvcov_init_state` on the inner walk).
    const SECOND_ORDER: bool;
    /// Whether every derivative this type carries is of total order ≤ 2 in its inputs, so a
    /// function of a few intermediates may be replaced by its second-order Taylor polynomial
    /// in them without changing any jet (the adjoint's PK-space closed form,
    /// `sens::ode_adjoint`). `false` unless an implementor opts in; a type whose taped or
    /// nested parts differentiate a second-order jet again must not.
    const TAYLOR2_EXACT: bool = false;
    /// Whether the forward jet carries second derivatives in its own directions (the
    /// adjoint's η-Hessian payload): what a [`Self::TAYLOR2_EXACT`] stand-in has to reproduce
    /// besides the tape's first θ-derivatives. Defaults to [`Self::SECOND_ORDER`].
    const ETA_HESSIAN: bool = Self::SECOND_ORDER;
    /// Whether this value is on a reverse tape, i.e. something downstream differentiates
    /// it once more. `false` for every type without one.
    #[inline]
    fn is_taped(self) -> bool {
        false
    }
    /// Whether this value carries no derivative at all (no jet, no tape), so a function of
    /// it may be evaluated on `f64` and lifted with [`Self::from_f64`]. Conservative:
    /// `false` unless an implementor can tell cheaply.
    #[inline]
    fn is_constant(self) -> bool {
        false
    }
    /// Lift a constant into the numeric type (zero derivatives for duals).
    fn from_f64(x: f64) -> Self;
    /// Seed dual dimension `dim` as an independent variable at value `x`. `f64`
    /// carries no derivatives, so it ignores `dim` and behaves like `from_f64`.
    fn var(x: f64, dim: usize) -> Self;
    /// The underlying value — for guards / branch conditions only.
    fn val(self) -> f64;
    /// `exp`.
    fn exp(self) -> Self;
    /// `ln`.
    fn ln(self) -> Self;
    /// `sqrt`.
    fn sqrt(self) -> Self;
    /// `self^e` (binary power; the bytecode `Op::Pow`).
    fn pow(self, e: Self) -> Self;
    /// `|self|`.
    fn abs(self) -> Self;
    /// `inv_logit(self) = 1/(1+e^{−x})` (logistic sigmoid).
    fn inv_logit(self) -> Self;
    /// `logit(self) = ln(x/(1−x))`.
    fn logit(self) -> Self;
    /// `cos` (3-cpt trigonometric cubic root solve).
    fn cos(self) -> Self;
    /// `acos` (3-cpt trigonometric cubic root solve).
    fn acos(self) -> Self;
    /// Lower-clamp the value to `lo` (branch on `.val()`), used by the bytecode
    /// VM to reproduce the `Op::Ln`/`Op::Sqrt` domain guards (`v.max(lo)`) before
    /// the transcendental call. For duals the clamped region is flat (zero jet).
    fn guard_floor(self, lo: f64) -> Self;
    /// `ln Γ(self)` (log-gamma). The transit absorption forcing's `ln Γ(n + 1)`
    /// constant rides this on the analytic ODE sensitivity path, so for the dual
    /// types it must carry 1st/2nd-order derivatives (digamma/trigamma) (#430).
    fn ln_gamma(self) -> Self;
    /// Whether every **derivative** component is finite — the gradient, and the Hessian
    /// block when the type carries one. Says nothing about [`val`](Self::val), which a
    /// caller checks separately.
    ///
    /// Exists because value-finiteness and jet-finiteness genuinely come apart (#1204). A
    /// dual's jets carry higher powers of the same quantities its value carries linearly:
    /// `recip` scales the gradient by `1/x²` and the Hessian by `1/x³`, and integrating
    /// `u' = p·u` gives `∂u/∂p = t·u` and `∂²u/∂p² = t²·u`. So a state can sit at a finite
    /// `7.2e303` with a gradient of `1.4e306` and a Hessian that has already overflowed to
    /// `inf`/`NaN` — a solve that reports success, returns finite predictions, and hands
    /// FOCEI a non-finite gradient.
    ///
    /// `f64` carries no jets and so is vacuously `true`; the guard that consumes this is
    /// therefore inert on the prediction path and only ever fires on a dual instantiation
    /// (see `crate::ode::solver`'s escalation guard for why that asymmetry is deliberate).
    fn jets_finite(self) -> bool;

    /// The largest absolute **derivative** component — gradient, and the Hessian block when
    /// the type carries one; `0.0` for `f64`. Like [`jets_finite`](Self::jets_finite) it
    /// ignores [`val`](Self::val).
    ///
    /// For a check that has to hold in every component, not just the value: the exact linear
    /// propagator (#852) reads `A` off differences of the RHS, and a cancellation can wipe a
    /// derivative component while the value survives intact, so its self-check compares the
    /// jets of `f(x)` and `A·x + b` too, against a scale built from these magnitudes.
    ///
    /// Provided, returning `0.0`, so adding it did not break an implementor outside this
    /// crate. That default is only correct for a jet-free type; every dual in this crate
    /// overrides it. An external type cannot reach the exact propagator (only a parser-certified
    /// spec does, and that runs on this crate's types), so the default cannot weaken its check.
    fn jet_max_abs(self) -> f64 {
        0.0
    }

    /// `φ(self)` for a scalar function known through its value and first two derivatives
    /// at `self.val()` (`v = φ(x)`, `d1 = φ′(x)`, `d2 = φ″(x)`): the special functions the
    /// likelihood needs (e.g. `log Φ` for M3 rows) and a residual variance as a function of
    /// the prediction. Exact to second order on every dual and tape type.
    ///
    /// Provided, as the second-order Taylor polynomial in `δ = self − self.val()` (value `0`,
    /// `self`'s jet), so an implementor outside this crate keeps compiling (#2093 review): it is
    /// exact for every type tracking at most second derivatives, which every external `PkNum`
    /// predating this member is. Every type in this crate overrides it.
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        let dx = self - Self::from_f64(self.val());
        Self::from_f64(v) + dx * Self::from_f64(d1) + dx * dx * Self::from_f64(0.5 * d2)
    }

    /// [`Self::lift2`] with the third derivative `d3 = φ‴` too, for types that carry a
    /// second-order jet over a differentiable base (`sens::rtape::DualH`): their
    /// `φ″` factor needs its own derivative. The default drops `d3`, which is exact for
    /// every type tracking fewer orders.
    #[inline]
    fn lift3(self, v: f64, d1: f64, d2: f64, _d3: f64) -> Self {
        self.lift2(v, d1, d2)
    }

    /// [`Self::lift3`] with the fourth derivative `d4 = φ⁗` too, for the covariance step's
    /// nested dual (`sens::rtape::DualH` over `DualT` over the reverse tape), which carries
    /// four orders: a Laplace exact-anchor R-matrix differentiates `∂²f/∂η²` twice more.
    /// The default drops `d4`, which is exact for every type tracking fewer orders.
    #[inline]
    fn lift4(self, v: f64, d1: f64, d2: f64, d3: f64, _d4: f64) -> Self {
        self.lift3(v, d1, d2, d3)
    }

    /// `u + h·(c₀k₀ + c₁k₁ + …)` with exactly this association (left-to-right sum,
    /// then `h·`, then `u +`): an explicit RK stage combination. The default *is* that
    /// expression, so `f64` and the forward duals keep their trajectories bit for bit.
    /// The adjoint tape type (`sens::vtape::DualV`) overrides it to record one
    /// node instead of `~3j` binary ones.
    #[inline]
    fn lincomb(u: Self, h: f64, cs: &[f64], ks: &[Self]) -> Self {
        let mut s = Self::from_f64(cs[0]) * ks[0];
        for j in 1..cs.len() {
            s = s + Self::from_f64(cs[j]) * ks[j];
        }
        u + Self::from_f64(h) * s
    }

    /// `((acc + x₀·w₀) + x₁·w₁) + …`, each term `xⱼ · from_f64(wⱼ)`, left to right: a
    /// Rosenbrock stage combination. The default *is* that expression; the adjoint tape type
    /// (`sens::vtape::DualV`) computes the same values and records one linear node instead
    /// of two per term.
    #[inline]
    fn axpy_fold(acc: Self, terms: &[(Self, f64)]) -> Self {
        let mut a = acc;
        for &(x, w) in terms {
            a = a + x * Self::from_f64(w);
        }
        a
    }

    /// `((acc + from_f64(w₀)·x₀) + from_f64(w₁)·x₁) + …`: [`Self::axpy_fold`] with the
    /// constant on the left, as a matrix combination `Σ cₖ·Mₖ` writes it.
    #[inline]
    fn caxpy_fold(acc: Self, terms: &[(f64, Self)]) -> Self {
        let mut a = acc;
        for &(w, x) in terms {
            a = a + Self::from_f64(w) * x;
        }
        a
    }

    /// Whether this type's folds ([`Self::dot_fold`], [`Self::caxpy_fold`],
    /// [`Self::axpy_fold`]) record fewer tape nodes than the per-operation expressions, so a
    /// kernel should call them per entry; otherwise it keeps its own loop (same values).
    const FUSED_FOLDS: bool = false;

    /// `((acc + a₀·b₀) + a₁·b₁) + …`, left to right: one entry of a matrix product. The
    /// default *is* that expression; `DualV` computes the same values and records one node
    /// whose rule is the product rule per pair, instead of two nodes per term.
    #[inline]
    fn dot_fold(acc: Self, pairs: &[(Self, Self)]) -> Self {
        let mut r = acc;
        for &(a, b) in pairs {
            r = r + a * b;
        }
        r
    }

    /// The same number off any tape: a value-only check computed on it records nothing.
    #[inline]
    fn detach(self) -> Self {
        self
    }

    /// `(a − b) · from_f64(s)`: a forward-difference quotient. The default *is* that
    /// expression; `DualV` computes the same values and records one linear node.
    #[inline]
    fn scaled_diff(a: Self, b: Self, s: f64) -> Self {
        (a - b) * Self::from_f64(s)
    }

    /// `base + Σ αₜ·xₜ·yₜ`, left to right: one entry of a weighted Gram sum such as the
    /// FOCEI `H̃ = Ω⁻¹ + Σ_j α_j g_j g_jᵀ`. The reverse tape (`sens::rtape::R`)
    /// overrides it to record one node instead of three per term.
    #[inline]
    fn wsum3(base: f64, terms: impl Iterator<Item = (Self, Self, Self)>) -> Self {
        let mut s = Self::from_f64(base);
        for (a, x, y) in terms {
            s = s + a * x * y;
        }
        s
    }

    /// Tape position for [`Self::tape_rollback`]; `0` for types that record nothing.
    #[inline]
    fn tape_mark() -> usize {
        0
    }

    /// Discard every record made after `mark` (a rejected ODE step attempt). No-op for
    /// types that record nothing.
    #[inline]
    fn tape_rollback(_mark: usize) {}

    /// LU factorisation hook for a taping type: `Some(ok)` when the type factored `m`
    /// itself (it may leave `m` unfactored and keep its state behind `piv`), `None` to
    /// run the generic elimination (`sens::linsolve`).
    #[inline]
    fn lu_factor_hook(_m: &mut [Self], _piv: &mut [usize], _n: usize) -> Option<bool> {
        None
    }

    /// Solve hook paired with [`Self::lu_factor_hook`]; `true` when handled.
    #[inline]
    fn lu_solve_hook(_m: &[Self], _piv: &[usize], _n: usize, _b: &mut [Self]) -> bool {
        false
    }
}

impl PkNum for f64 {
    const SECOND_ORDER: bool = false;
    #[inline]
    fn is_constant(self) -> bool {
        true
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn var(x: f64, _dim: usize) -> Self {
        x
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn pow(self, e: Self) -> Self {
        f64::powf(self, e)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn inv_logit(self) -> Self {
        1.0 / (1.0 + f64::exp(-self))
    }
    #[inline]
    fn logit(self) -> Self {
        f64::ln(self / (1.0 - self))
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        self.max(lo)
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::stats::special::ln_gamma(self)
    }
    #[inline]
    fn lift2(self, v: f64, _d1: f64, _d2: f64) -> Self {
        v
    }
    #[inline]
    fn jets_finite(self) -> bool {
        // No derivative components to be non-finite. Const-folds away in the guard, so the
        // `f64` prediction path pays nothing for a check only the dual paths can fail.
        true
    }
}

/// Generate the `PkNum` impl for a dual type whose every method is a one-line
/// delegation to the inherent method of the same name. `Dual1<N>` and
/// `DualMixed<NA, N>` differ only in their const-generic list, the
/// `SECOND_ORDER` flag, and (implicitly) which inherent method the delegator
/// resolves to — everything else is identical, including the `guard_floor`
/// NaN-flooring body. The scalar `f64` impl stays hand-written (its bodies are
/// inline formulas, not delegation).
macro_rules! impl_pknum_delegate {
    ($ty:ident<$(const $c:ident: usize),+>, second_order = $so:expr) => {
        impl<$(const $c: usize),+> PkNum for $ty<$($c),+> {
            const SECOND_ORDER: bool = $so;
            #[inline]
            fn from_f64(x: f64) -> Self {
                $ty::constant(x)
            }
            #[inline]
            fn var(x: f64, dim: usize) -> Self {
                $ty::var(x, dim)
            }
            #[inline]
            fn val(self) -> f64 {
                self.value
            }
            #[inline]
            fn exp(self) -> Self {
                $ty::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                $ty::ln(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                $ty::sqrt(self)
            }
            #[inline]
            fn pow(self, e: Self) -> Self {
                $ty::powd(self, e)
            }
            #[inline]
            fn abs(self) -> Self {
                $ty::abs(self)
            }
            #[inline]
            fn inv_logit(self) -> Self {
                $ty::inv_logit(self)
            }
            #[inline]
            fn logit(self) -> Self {
                $ty::logit(self)
            }
            #[inline]
            fn cos(self) -> Self {
                $ty::cos(self)
            }
            #[inline]
            fn acos(self) -> Self {
                $ty::acos(self)
            }
            #[inline]
            fn guard_floor(self, lo: f64) -> Self {
                // `!(value >= lo)` (not `value < lo`) so a `NaN` value floors too,
                // matching `f64::max(lo)` on the scalar path (`NaN.max(lo) == lo`) —
                // otherwise a transient `NaN` would give a finite f64 prediction but a
                // `NaN` dual gradient (the clamped region is flat, so the floored jet
                // is zero) (#430).
                if !(self.value >= lo) {
                    $ty::constant(lo)
                } else {
                    self
                }
            }
            #[inline]
            fn ln_gamma(self) -> Self {
                $ty::ln_gamma(self)
            }
            #[inline]
            fn jets_finite(self) -> bool {
                $ty::jets_finite(self)
            }
            #[inline]
            fn jet_max_abs(self) -> f64 {
                $ty::jet_max_abs(self)
            }
            #[inline]
            fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
                $ty::lift2(self, v, d1, d2)
            }
        }
    };
}

impl_pknum_delegate!(Dual1<const N: usize>, second_order = false);
impl_pknum_delegate!(DualMixed<const NA: usize, const N: usize>, second_order = true);

/// In-place arithmetic for the generic bytecode VM (`eval_bytecode_g`).
///
/// The VM keeps its operands on a `Vec<T>` stack. For the outer `DualMixed<NA, N>`
/// type a value is several KB (`N + NA·N + 1` doubles), so the by-value
/// `pop, pop, push(a ∘ b)` shape spent about a third of a busulfan ODE+IOV FOCEI fit
/// in `memmove`/`memset` — moving operands and materialising all-zero jets for
/// constants. These update the left operand where it sits on the stack and take the
/// right one by reference, or as a bare `f64` when it is a constant, covariate or
/// `TIME`.
///
/// **Every method is bit-identical to the by-value operator it replaces**, the scalar
/// ones included: they evaluate the formula the operator would apply to a
/// `from_f64(c)` operand term by term — `x·0.0`, `+ 0.0` and all — just without
/// building that operand. So signed zeros and `inf · 0 = NaN` come out exactly as
/// before. `f64` and `Dual1` (small enough that moves are cheap) use the operators
/// themselves; `DualMixed` writes the same formulas as in-place loops.
///
/// In a crate-private module, so it is not public API and does not constrain
/// `PkNum`: the VM and the generic code that drives it bound on [`vm::VmNum`]
/// (`PkNum + VmArith`) instead.
pub(crate) mod vm {
    use super::{Dual1, DualMixed};

    /// A number type's lanes as the RHS's native code sees them (`parser::rhs_ir`): `f64`
    /// (`n = m = 0`), or a dual of `n` tangents and an `m`-entry second-order payload,
    /// laid out `[value, g_0 … g_{n−1}, h_0 … h_{m−1}]`.
    ///
    /// `th` is `DualV`'s directional mode (`m = n`, `h_k = ∂²/∂t_k²`); otherwise a
    /// non-empty payload is the packed lower triangle (`m = n(n+1)/2`). `wide` is
    /// [`super::outlined`]: such a dual applies a constant operand of `×`/`÷` as a full
    /// product with the constant dual, a different operation order.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct Layout {
        pub(crate) n: u16,
        pub(crate) m: u16,
        pub(crate) th: bool,
        pub(crate) wide: bool,
    }

    impl Layout {
        pub(crate) const F64: Layout = Layout {
            n: 0,
            m: 0,
            th: false,
            wide: false,
        };
        pub(crate) const fn lanes(&self) -> usize {
            1 + self.n as usize + self.m as usize
        }
    }

    /// A compiled RHS a tape can record whole and differentiate later (`parser::rhs_jit`):
    /// its native forward and reverse pass over a [`Layout`]'s lanes, `fn(u, params, seeds,
    /// du)` and `fn(u, params, seeds, du, ū, p̄, s̄, dū)`.
    pub(crate) trait RhsKernel: Send + Sync {
        /// How many elements of `u`, `params` and `du` the program touches.
        fn extent(&self) -> (usize, usize, usize);
        /// Whether every derivative below the extent is written, so an output never
        /// depends on what `du` held before the call.
        fn writes_every_du(&self) -> bool;
        /// Whether a native forward exists (or can be built) for `layout`.
        fn has_forward(&self, layout: Layout) -> bool;
        /// Whether a reverse pass exists (or can be built) for `layout`.
        fn has_reverse(&self, layout: Layout) -> bool;
        /// The forward over `layout`.
        ///
        /// # Safety
        /// Each pointer covers [`Self::extent`]'s elements (4 seeds) of `layout`'s lanes,
        /// and `du` does not overlap the inputs; [`Self::has_forward`] holds.
        unsafe fn forward(
            &self,
            layout: Layout,
            u: *const f64,
            params: *const f64,
            seeds: *const f64,
            du: *mut f64,
        );
        /// Add `dūᵀ·∂du/∂(u, params, seeds)` into `ū`, `p̄`, `s̄`.
        ///
        /// # Safety
        /// As [`Self::forward`] for the primal inputs; each adjoint buffer is laid out like
        /// its primal; [`Self::has_reverse`] holds.
        #[allow(clippy::too_many_arguments)]
        unsafe fn reverse(
            &self,
            layout: Layout,
            u: *const f64,
            params: *const f64,
            seeds: *const f64,
            u_bar: *mut f64,
            p_bar: *mut f64,
            s_bar: *mut f64,
            du_bar: *const f64,
        );
    }

    pub trait VmArith: Sized {
        /// Whether a value of this type can carry a tape node, so evaluating the same
        /// expression twice records it twice. Code that reuses a value instead of
        /// recomputing it (the ODE RHS's hoist) changes such a type's tape.
        const RECORDS_TAPE: bool = false;
        /// This type's lanes for the RHS's native code, when it has them: `f64`, and the
        /// duals whose every part is an `f64` and whose arithmetic is the lane arithmetic
        /// `parser::rhs_ir` transcribes.
        const LANES: Option<Layout> = None;
        /// Whether a value sits in memory as exactly its lanes (`#[repr(C)]`, value first),
        /// so a slice of them passes to native code as it is. Otherwise the lanes are
        /// copied out and back ([`Self::put_lanes`], [`Self::from_lanes`]).
        const CONTIGUOUS: bool = false;
        /// Whether this value may run through native code: it is not on a tape.
        #[inline]
        fn native_ok(&self) -> bool {
            true
        }
        /// Write this value's [`Self::LANES`] into `out` (a non-contiguous type).
        fn put_lanes(&self, _out: &mut [f64]) {
            unreachable!("only a non-contiguous type with LANES copies its lanes")
        }
        /// A value, off any tape, from its lanes (a non-contiguous type).
        fn from_lanes(_lanes: &[f64]) -> Self {
            unreachable!("only a non-contiguous type with LANES copies its lanes")
        }
        /// Record one whole RHS call on this type's tape: `du` from `kernel`'s native
        /// forward, as tape nodes whose reverse rule is `kernel`'s reverse pass, instead of
        /// one node per operation. `false`, with nothing written or recorded, when the type
        /// has no such record or the kernel cannot serve it.
        fn tape_rhs(
            _kernel: &std::sync::Arc<dyn RhsKernel>,
            _u: &[Self],
            _params: &[Self],
            _seeds: &[Self; 4],
            _du: &mut [Self],
        ) -> bool {
            false
        }
        /// `self = self + rhs`.
        fn add_in(&mut self, rhs: &Self);
        /// `self = self - rhs`.
        fn sub_in(&mut self, rhs: &Self);
        /// `self = rhs - self` (the bytecode `RSub`).
        fn rsub_in(&mut self, rhs: &Self);
        /// `self = self * rhs`.
        fn mul_in(&mut self, rhs: &Self);
        /// `self = self / rhs` (no zero guard — the caller owns it).
        fn div_in(&mut self, rhs: &Self);
        /// `self = self + from_f64(c)`.
        fn add_scalar_in(&mut self, c: f64);
        /// `self = self - from_f64(c)`.
        fn sub_scalar_in(&mut self, c: f64);
        /// `self = from_f64(c) - self`.
        fn rsub_scalar_in(&mut self, c: f64);
        /// `self = self * from_f64(c)`.
        fn mul_scalar_in(&mut self, c: f64);
        /// `self = self / from_f64(c)` (no zero guard — the caller owns it).
        fn div_scalar_in(&mut self, c: f64);
        /// Whether `self` and `other` are the same number bit for bit — the value
        /// and, for a dual, every jet entry. Keys the ODE RHS's lifted-subexpression
        /// cache (`RhsHoist` in the parser): a lifted value is a pure function of the
        /// inputs it reads, so bit-identical inputs give a bit-identical value.
        fn same_bits(&self, other: &Self) -> bool;
    }

    /// [`super::PkNum`] plus the VM's in-place arithmetic: the bound for code that runs
    /// the bytecode VM over a generic numeric type. Kept off `PkNum` itself so `PkNum`
    /// stays implementable outside the crate; every in-crate `PkNum` type has it.
    pub trait VmNum: super::PkNum + VmArith + 'static {}
    impl<T: super::PkNum + VmArith + 'static> VmNum for T {}

    /// `VmArith` as the by-value operators, for types small enough that moving them
    /// costs nothing worth saving.
    macro_rules! impl_vm_by_value {
        ($ty:ty, $lift:expr, $same:expr $(, $c:ident)* $(; $($extra:item)*)?) => {
            impl<$(const $c: usize),*> VmArith for $ty {
                $($($extra)*)?
                #[inline]
                fn add_in(&mut self, rhs: &Self) {
                    *self = *self + *rhs;
                }
                #[inline]
                fn sub_in(&mut self, rhs: &Self) {
                    *self = *self - *rhs;
                }
                #[inline]
                fn rsub_in(&mut self, rhs: &Self) {
                    *self = *rhs - *self;
                }
                #[inline]
                fn mul_in(&mut self, rhs: &Self) {
                    *self = *self * *rhs;
                }
                #[inline]
                fn div_in(&mut self, rhs: &Self) {
                    *self = *self / *rhs;
                }
                #[inline]
                fn add_scalar_in(&mut self, c: f64) {
                    *self = *self + $lift(c);
                }
                #[inline]
                fn sub_scalar_in(&mut self, c: f64) {
                    *self = *self - $lift(c);
                }
                #[inline]
                fn rsub_scalar_in(&mut self, c: f64) {
                    *self = $lift(c) - *self;
                }
                #[inline]
                fn mul_scalar_in(&mut self, c: f64) {
                    *self = *self * $lift(c);
                }
                #[inline]
                fn div_scalar_in(&mut self, c: f64) {
                    *self = *self / $lift(c);
                }
                #[inline]
                fn same_bits(&self, other: &Self) -> bool {
                    $same(self, other)
                }
            }
        };
    }

    impl_vm_by_value!(f64, |c: f64| c, |a: &f64, b: &f64| a.to_bits()
        == b.to_bits();
        const LANES: Option<Layout> = Some(Layout::F64);
        const CONTIGUOUS: bool = true;
    );
    impl_vm_by_value!(Dual1<N>, Dual1::<N>::constant, Dual1::<N>::same_bits, N);
    /// By-value `VmArith` for a type whose operators and `from_f64` are its own.
    macro_rules! vm_ops_body {
        ($same:expr) => {
            #[inline]
            fn add_in(&mut self, rhs: &Self) {
                *self = *self + *rhs;
            }
            #[inline]
            fn sub_in(&mut self, rhs: &Self) {
                *self = *self - *rhs;
            }
            #[inline]
            fn rsub_in(&mut self, rhs: &Self) {
                *self = *rhs - *self;
            }
            #[inline]
            fn mul_in(&mut self, rhs: &Self) {
                *self = *self * *rhs;
            }
            #[inline]
            fn div_in(&mut self, rhs: &Self) {
                *self = *self / *rhs;
            }
            #[inline]
            fn add_scalar_in(&mut self, c: f64) {
                *self = *self + <Self as super::PkNum>::from_f64(c);
            }
            #[inline]
            fn sub_scalar_in(&mut self, c: f64) {
                *self = *self - <Self as super::PkNum>::from_f64(c);
            }
            #[inline]
            fn rsub_scalar_in(&mut self, c: f64) {
                *self = <Self as super::PkNum>::from_f64(c) - *self;
            }
            #[inline]
            fn mul_scalar_in(&mut self, c: f64) {
                *self = *self * <Self as super::PkNum>::from_f64(c);
            }
            #[inline]
            fn div_scalar_in(&mut self, c: f64) {
                *self = *self / <Self as super::PkNum>::from_f64(c);
            }
            #[inline]
            fn same_bits(&self, other: &Self) -> bool {
                ($same)(self, other)
            }
        };
    }

    /// The scalar reverse tape: the same value *and* the same tape node.
    impl VmArith for crate::sens::rtape::R {
        const RECORDS_TAPE: bool = true;
        vm_ops_body!(
            |a: &crate::sens::rtape::R, b: &crate::sens::rtape::R| a.v.to_bits() == b.v.to_bits()
                && a.i == b.i
        );
    }

    impl<const N: usize, B: VmNum> VmArith for crate::sens::rtape::DualT<N, B> {
        const RECORDS_TAPE: bool = B::RECORDS_TAPE;
        vm_ops_body!(|a: &Self, b: &Self| a.v.same_bits(&b.v)
            && a.g.iter().zip(&b.g).all(|(x, y)| x.same_bits(y)));
    }

    impl<const N: usize, const M: usize, B: VmNum, const D: bool> VmArith
        for crate::sens::rtape::DualH<N, M, B, D>
    {
        const RECORDS_TAPE: bool = B::RECORDS_TAPE;
        vm_ops_body!(|a: &Self, b: &Self| a.v.same_bits(&b.v)
            && a.g.iter().zip(&b.g).all(|(x, y)| x.same_bits(y))
            && a.h.iter().zip(&b.h).all(|(x, y)| x.same_bits(y)));
    }

    /// Both tape modes of [`crate::sens::vtape::DualV`] (the macro takes `usize` consts only).
    impl<const N: usize, const M: usize, const TH: bool> VmArith
        for crate::sens::vtape::DualV<N, M, TH>
    {
        const RECORDS_TAPE: bool = true;
        const LANES: Option<Layout> = Some(Layout {
            n: N as u16,
            m: M as u16,
            th: TH,
            wide: crate::sens::vtape::DualV::<N, M, TH>::WIDE,
        });
        #[inline]
        fn native_ok(&self) -> bool {
            !<Self as super::PkNum>::is_taped(*self)
        }
        #[inline]
        fn put_lanes(&self, out: &mut [f64]) {
            out[0] = self.v;
            out[1..1 + N].copy_from_slice(&self.g);
            out[1 + N..1 + N + M].copy_from_slice(&self.h);
        }
        #[inline]
        fn from_lanes(l: &[f64]) -> Self {
            let mut g = [0.0; N];
            let mut h = [0.0; M];
            g.copy_from_slice(&l[1..1 + N]);
            h.copy_from_slice(&l[1 + N..1 + N + M]);
            crate::sens::vtape::DualV::free(l[0], g, h)
        }
        #[inline]
        fn tape_rhs(
            kernel: &std::sync::Arc<dyn RhsKernel>,
            u: &[Self],
            params: &[Self],
            seeds: &[Self; 4],
            du: &mut [Self],
        ) -> bool {
            crate::sens::vtape::record_rhs(kernel, u, params, seeds, du)
        }
        #[inline]
        fn add_in(&mut self, rhs: &Self) {
            *self = *self + *rhs;
        }
        #[inline]
        fn sub_in(&mut self, rhs: &Self) {
            *self = *self - *rhs;
        }
        #[inline]
        fn rsub_in(&mut self, rhs: &Self) {
            *self = *rhs - *self;
        }
        #[inline]
        fn mul_in(&mut self, rhs: &Self) {
            *self = *self * *rhs;
        }
        #[inline]
        fn div_in(&mut self, rhs: &Self) {
            *self = *self / *rhs;
        }
        #[inline]
        fn add_scalar_in(&mut self, c: f64) {
            *self = self.add_c(c);
        }
        #[inline]
        fn sub_scalar_in(&mut self, c: f64) {
            *self = self.sub_c(c);
        }
        #[inline]
        fn rsub_scalar_in(&mut self, c: f64) {
            *self = self.rsub_c(c);
        }
        #[inline]
        fn mul_scalar_in(&mut self, c: f64) {
            *self = self.mul_c(c);
        }
        #[inline]
        fn div_scalar_in(&mut self, c: f64) {
            *self = self.div_c(c);
        }
        #[inline]
        fn same_bits(&self, other: &Self) -> bool {
            crate::sens::vtape::DualV::same_bits(self, other)
        }
    }

    impl<const NA: usize, const N: usize> VmArith for DualMixed<NA, N> {
        #[inline]
        fn add_in(&mut self, rhs: &Self) {
            DualMixed::add_in(self, rhs)
        }
        #[inline]
        fn sub_in(&mut self, rhs: &Self) {
            DualMixed::sub_in(self, rhs)
        }
        #[inline]
        fn rsub_in(&mut self, rhs: &Self) {
            DualMixed::rsub_in(self, rhs)
        }
        #[inline]
        fn mul_in(&mut self, rhs: &Self) {
            DualMixed::mul_in(self, rhs)
        }
        #[inline]
        fn div_in(&mut self, rhs: &Self) {
            DualMixed::div_in(self, rhs)
        }
        #[inline]
        fn add_scalar_in(&mut self, c: f64) {
            DualMixed::add_scalar_in(self, c)
        }
        #[inline]
        fn sub_scalar_in(&mut self, c: f64) {
            DualMixed::sub_scalar_in(self, c)
        }
        #[inline]
        fn rsub_scalar_in(&mut self, c: f64) {
            DualMixed::rsub_scalar_in(self, c)
        }
        #[inline]
        fn mul_scalar_in(&mut self, c: f64) {
            DualMixed::mul_scalar_in(self, c)
        }
        #[inline]
        fn div_scalar_in(&mut self, c: f64) {
            DualMixed::div_scalar_in(self, c)
        }
        #[inline]
        fn same_bits(&self, other: &Self) -> bool {
            DualMixed::same_bits(self, other)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual1::Dual1;
    use crate::sens::dual2::Dual2;
    use crate::sens::dual_mixed::DualMixed;

    /// Exercise every `PkNum` method on a value once, so the per-impl delegators
    /// (and the underlying `Dual1`/`Dual2` ops they call) are covered. `0.7` is a
    /// safe argument for all: positive (exp/ln/sqrt/pow), in `(0,1)` (logit/
    /// inv_logit) and in `[-1,1]` (acos).
    fn exercise<T: PkNum>(x: T) {
        let _ = T::from_f64(1.0).val();
        let _ = x.val();
        let _ = x.exp().val();
        let _ = x.ln().val();
        let _ = x.sqrt().val();
        let _ = x.pow(T::from_f64(2.0)).val();
        let _ = x.abs().val();
        let _ = (-x.abs()).abs().val(); // negative branch of abs
        let _ = x.cos().val();
        let _ = x.acos().val();
        let _ = x.inv_logit().val();
        let _ = x.logit().val();
        let _ = x.guard_floor(1e-6).val();
        let _ = x.guard_floor(10.0).val(); // floor-active branch
        let _ = x.ln_gamma().val(); // 0.7 > 0, in the ln_gamma domain
    }

    #[test]
    fn pknum_all_methods_covered_for_each_impl() {
        exercise::<f64>(0.7);
        exercise::<Dual1<1>>(Dual1::var(0.7, 0));
        exercise::<Dual2<1>>(Dual2::var(0.7, 0));
        exercise::<DualMixed<1, 2>>(DualMixed::var(0.7, 0));
    }

    /// `guard_floor(NaN)` must floor (return `lo`) on the duals exactly as it does on
    /// `f64` (`NaN.max(lo) == lo`) — a sub-floor / `NaN` value lands in the flat
    /// clamped region, so the dual's value is `lo` and its jet is zero. Without this,
    /// a transient `NaN` would give a finite f64 prediction but a `NaN` dual gradient
    /// (#430 review #2).
    #[test]
    fn guard_floor_floors_nan_consistently_across_impls() {
        let lo = 1e-6;
        assert_eq!(f64::NAN.guard_floor(lo), lo);
        assert_eq!(Dual1::<1>::var(f64::NAN, 0).guard_floor(lo).value, lo);
        assert_eq!(Dual2::<1>::var(f64::NAN, 0).guard_floor(lo).value, lo);
        assert_eq!(
            DualMixed::<1, 2>::var(f64::NAN, 0).guard_floor(lo).value,
            lo
        );
        // The floored dual carries a zero jet (flat clamped region).
        let g = Dual1::<1>::var(f64::NAN, 0).guard_floor(lo);
        assert_eq!(g.grad[0], 0.0);
        // A value already above the floor is untouched.
        assert_eq!(Dual1::<1>::var(5.0, 0).guard_floor(lo).value, 5.0);
    }

    /// The `ln_gamma` `Dual2` rule: its analytic 1st (digamma) and 2nd (trigamma)
    /// derivatives must match central finite differences of `special::ln_gamma`,
    /// since the transit forcing's `ln Γ(n + 1)` constant rides this on the analytic
    /// ODE sensitivity path (#430 slice 2). This is exactly the `Dual2`-rule
    /// regression an FD-only check would miss (cf. the `guard_floor` NaN bug, #2).
    #[test]
    fn ln_gamma_dual2_matches_central_fd() {
        use crate::stats::special::ln_gamma;
        for &x in &[0.6, 1.0, 2.5, 4.0, 7.3] {
            let d = Dual2::<1>::var(x, 0).ln_gamma();
            approx::assert_relative_eq!(d.value, ln_gamma(x), max_relative = 1e-12);
            // 1st derivative (digamma) vs central FD of ln_gamma.
            let h1 = 1e-5;
            let fd1 = (ln_gamma(x + h1) - ln_gamma(x - h1)) / (2.0 * h1);
            approx::assert_relative_eq!(d.grad[0], fd1, max_relative = 1e-6, epsilon = 1e-9);
            // 2nd derivative (trigamma) vs central second difference of ln_gamma.
            let h2 = 1e-4;
            let fd2 = (ln_gamma(x + h2) - 2.0 * ln_gamma(x) + ln_gamma(x - h2)) / (h2 * h2);
            approx::assert_relative_eq!(d.hess[0][0], fd2, max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    /// The dual `ln_gamma` chain rule has two second-order terms — `ψ′·x′⊗x′`
    /// **and** `ψ·x″`. `ln_gamma_dual2_matches_central_fd` seeds a bare `var`
    /// (x″ = 0), so it exercises only the first. Feed a dual with a NONZERO input
    /// Hessian via `x ↦ ln Γ(eˣ)` (the inner `exp` carries x′ and x″) so the
    /// `ψ·x″` term is exercised too — the path real transit fits hit when an
    /// absorption param is composite (e.g. `MTT = TVMTT·exp(ETA_MTT)`). Negative
    /// `x` (eˣ < 0.5) additionally drives the **reflection** branch of the dual
    /// rule (#458 review #1/#2). Without `ψ·x″`, the main-branch points fail; with
    /// a wrong reflection rule, the negative points fail.
    #[test]
    fn ln_gamma_dual2_chain_rule_nonzero_input_hessian_and_reflection() {
        use crate::stats::special::ln_gamma;
        let f = |x: f64| ln_gamma(x.exp());
        // x = −1.5, −1.0 → eˣ = 0.22, 0.37 < 0.5 (reflection); the rest main branch.
        for &x in &[-1.5_f64, -1.0, 0.1, 0.5, 1.2, 2.0] {
            let u = Dual2::<1>::var(x, 0).exp().ln_gamma();
            approx::assert_relative_eq!(u.value, f(x), max_relative = 1e-12);
            let h = 1e-5;
            let fd1 = (f(x + h) - f(x - h)) / (2.0 * h);
            approx::assert_relative_eq!(u.grad[0], fd1, max_relative = 1e-6, epsilon = 1e-9);
            let h2 = 1e-4;
            let fd2 = (f(x + h2) - 2.0 * f(x) + f(x - h2)) / (h2 * h2);
            approx::assert_relative_eq!(u.hess[0][0], fd2, max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    /// `f64` has no jets, so the check is vacuously true — which is exactly what keeps the
    /// #1204 guard inert on the prediction path.
    #[test]
    fn f64_has_no_jets_to_be_non_finite() {
        assert!(PkNum::jets_finite(1.0_f64));
        assert!(PkNum::jets_finite(0.0_f64));
        assert!(PkNum::jets_finite(f64::MAX));
        // Even a non-finite *value*: this predicate is about derivatives only, and the
        // caller checks `val()` separately.
        assert!(PkNum::jets_finite(f64::NAN));
        assert!(PkNum::jets_finite(f64::INFINITY));
    }

    /// `PkNum::lift2`'s provided default (the second-order Taylor polynomial) for an
    /// implementor outside this crate, which supplies only the required members: on a
    /// second-order dual it reproduces the crate's own exact `lift2` (value, gradient and
    /// Hessian), so adding the member broke no external implementor (#2093 review).
    #[test]
    fn lift2_default_is_exact_for_an_external_second_order_type() {
        #[derive(Clone, Copy, Debug)]
        struct Ext(Dual2<1>);
        macro_rules! op {
            ($tr:ident, $f:ident) => {
                impl std::ops::$tr for Ext {
                    type Output = Ext;
                    fn $f(self, o: Ext) -> Ext {
                        Ext(std::ops::$tr::$f(self.0, o.0))
                    }
                }
            };
        }
        op!(Add, add);
        op!(Sub, sub);
        op!(Mul, mul);
        op!(Div, div);
        impl std::ops::Neg for Ext {
            type Output = Ext;
            fn neg(self) -> Ext {
                Ext(-self.0)
            }
        }
        macro_rules! un {
            ($($f:ident),*) => { $(fn $f(self) -> Self { Ext(PkNum::$f(self.0)) })* };
        }
        impl PkNum for Ext {
            const SECOND_ORDER: bool = true;
            fn from_f64(x: f64) -> Self {
                Ext(<Dual2<1> as PkNum>::from_f64(x))
            }
            fn var(x: f64, dim: usize) -> Self {
                Ext(<Dual2<1> as PkNum>::var(x, dim))
            }
            fn val(self) -> f64 {
                PkNum::val(self.0)
            }
            un!(exp, ln, sqrt, abs, inv_logit, logit, cos, acos, ln_gamma);
            fn pow(self, e: Self) -> Self {
                Ext(PkNum::pow(self.0, e.0))
            }
            fn guard_floor(self, lo: f64) -> Self {
                Ext(PkNum::guard_floor(self.0, lo))
            }
            fn jets_finite(self) -> bool {
                PkNum::jets_finite(self.0)
            }
        }
        let x = <Dual2<1> as PkNum>::var(0.7, 0) * <Dual2<1> as PkNum>::var(0.7, 0);
        let (v, d1, d2) = (0.3, -1.7, 2.9);
        let want = PkNum::lift2(x, v, d1, d2);
        let got = PkNum::lift2(Ext(x), v, d1, d2).0;
        assert!(
            (got.value - want.value).abs() < 1e-15,
            "{got:?} vs {want:?}"
        );
        assert!(
            (got.grad[0] - want.grad[0]).abs() < 1e-14,
            "{got:?} vs {want:?}"
        );
        assert!(
            (got.hess[0][0] - want.hess[0][0]).abs() < 1e-13,
            "{got:?} vs {want:?}"
        );
        assert!(
            want.hess[0][0].abs() > 1.0,
            "a curved jet, so the d2 term is live"
        );
    }
}
