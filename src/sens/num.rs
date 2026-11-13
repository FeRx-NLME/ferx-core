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

/// An `f64` analysis the generic code runs on values ([`PkNum::host`]): a decision, or a
/// quantity it uses without differentiating through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HostFn {
    /// The spectral form of a linear ODE segment's generator
    /// (`ode::linear::spectral_analysis`).
    Spectral,
    /// The scaling-and-squaring count of a matrix exponential (`ode::linear::pade_squarings`).
    PadeSquarings,
    /// A Rosenbrock Jacobian's forward-difference steps (`ode::rosenbrock::fd_steps`).
    FdSteps,
    /// A partial-pivoted LU factorization of the values (`sens::linsolve::lu_factor_host`).
    LuFactor,
    /// The forcing column's scale in an augmented generator (`ode::linear::forcing_scale`).
    ForcingScale,
}

impl HostFn {
    /// Every analysis, by index (a native call passes the index).
    const ALL: [HostFn; 5] = [
        HostFn::Spectral,
        HostFn::PadeSquarings,
        HostFn::FdSteps,
        HostFn::LuFactor,
        HostFn::ForcingScale,
    ];
    pub(crate) fn index(self) -> u16 {
        Self::ALL.iter().position(|&f| f == self).expect("listed") as u16
    }
    pub(crate) fn from_index(i: u16) -> HostFn {
        Self::ALL[i as usize]
    }
    /// Run the analysis on `args`' values into `out` — or reuse the result of the same
    /// analysis on the same bits: one point's generator is analysed by its `f64` prediction and
    /// by every derivative layout of its gradient, and the analyses are pure.
    pub(crate) fn run(self, args: &[f64], out: &mut Vec<f64>) {
        use std::cell::RefCell;
        thread_local! {
            /// The last few analyses: `(f, argument bits, result)`, most recent last.
            static MEMO: RefCell<Vec<(HostFn, Vec<u64>, Vec<f64>)>> =
                const { RefCell::new(Vec::new()) };
        }
        const KEEP: usize = 8;
        let hit = MEMO.with(|m| {
            let m = m.borrow();
            m.iter().rev().find_map(|(g, bits, res)| {
                (*g == self
                    && bits.len() == args.len()
                    && bits.iter().zip(args).all(|(b, a)| *b == a.to_bits()))
                .then(|| res.clone())
            })
        });
        if let Some(res) = hit {
            out.clear();
            out.extend_from_slice(&res);
            return;
        }
        self.compute(args, out);
        MEMO.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() == KEEP {
                m.remove(0);
            }
            m.push((
                self,
                args.iter().map(|a| a.to_bits()).collect(),
                out.clone(),
            ));
        });
    }

    fn compute(self, args: &[f64], out: &mut Vec<f64>) {
        match self {
            HostFn::Spectral => crate::ode::linear::spectral_analysis(args, out),
            HostFn::PadeSquarings => crate::ode::linear::pade_squarings(args, out),
            HostFn::FdSteps => crate::ode::rosenbrock::fd_steps(args, out),
            HostFn::LuFactor => crate::sens::linsolve::lu_factor_host(args, out),
            HostFn::ForcingScale => crate::ode::linear::forcing_scale(args, out),
        }
    }
}

/// A scalar function given by its own jet: value and first four derivatives at a point
/// ([`JetFn::jet`]). A recording type (#2786) records `f(x)` as one primitive whose `k`-th
/// derivative is the jet's entry `k`; every other type lifts the jet ([`PkNum::jet_unary`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JetFn {
    /// `−log Φ(x)` (the probit record term).
    NegLogNormCdf,
    /// `ln(1 + eˣ)` (the logit record term, the softplus activation).
    Softplus,
    /// `Φ(x)`, the standard normal CDF (the model language's `pnorm`).
    Pnorm,
    /// `tanh(x)` (an activation).
    Tanh,
    /// `1/(1 + e⁻ˣ)` (the sigmoid activation).
    Logistic,
    /// `log Φ(x)` (the M3 censored row's likelihood).
    LogNormCdf,
    /// The inverse Mills ratio `φ(x)/Φ(x)` (the M3 censored row's curvature).
    InvMills,
    /// `eˣ − 1` (an interval-censored record's probability).
    ExpM1,
}

impl JetFn {
    /// `[f, f′, f″, f‴, f⁗]` at `x`.
    pub(crate) fn jet(self, x: f64) -> [f64; 5] {
        match self {
            #[cfg(feature = "survival")]
            JetFn::NegLogNormCdf => crate::categorical::neg_log_normal_cdf_jet(x),
            // Only the binary endpoint's probit link builds it.
            #[cfg(not(feature = "survival"))]
            JetFn::NegLogNormCdf => [f64::NAN; 5],
            JetFn::Softplus => crate::stats::special::softplus_jet(x),
            JetFn::Pnorm => crate::stats::special::pnorm_jet(x),
            JetFn::Tanh => crate::stats::special::tanh_jet(x),
            JetFn::Logistic => crate::stats::special::logistic_jet(x),
            JetFn::LogNormCdf => crate::stats::special::log_normal_cdf_jet(x),
            JetFn::InvMills => crate::stats::special::inv_mills_jet(x),
            JetFn::ExpM1 => {
                let e = x.exp();
                [x.exp_m1(), e, e, e, e]
            }
        }
    }
    pub(crate) const fn index(self) -> u8 {
        self as u8
    }
    /// The function of [`Self::index`] `i`.
    pub(crate) const fn from_index(i: u8) -> JetFn {
        match i {
            0 => JetFn::NegLogNormCdf,
            1 => JetFn::Softplus,
            2 => JetFn::Pnorm,
            3 => JetFn::Tanh,
            4 => JetFn::Logistic,
            5 => JetFn::LogNormCdf,
            6 => JetFn::InvMills,
            _ => JetFn::ExpM1,
        }
    }
}

/// A comparison of two values, for [`PkNum::br_cmp`]: IEEE's ordered comparisons, and `Ne`,
/// which holds when either side is `NaN` (Rust's `!=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchCmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl BranchCmp {
    /// `a cc b`, as Rust's operators decide it.
    #[inline(always)]
    pub fn holds(self, a: f64, b: f64) -> bool {
        match self {
            BranchCmp::Lt => a < b,
            BranchCmp::Le => a <= b,
            BranchCmp::Gt => a > b,
            BranchCmp::Ge => a >= b,
            BranchCmp::Eq => a == b,
            BranchCmp::Ne => a != b,
        }
    }
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
    /// The highest total derivative order this type can observe of a function it is
    /// evaluated through — its own jet's order, plus one for each tape that differentiates
    /// it again. A function of a few intermediates may be replaced by its Taylor polynomial
    /// of this order in them without changing anything the type carries (the adjoint's
    /// PK-space lift, `sens::ode_adjoint`). `0` for a plain number.
    const JET_ORDER: usize = 0;
    /// The `f64` work one arithmetic operation on this type costs, in lanes: the lanes it
    /// carries, plus two for each lane a tape records (the node, and its sweep). A cost proxy
    /// for choosing between two recordings of one function (the adjoint's PK-space lift), not
    /// a timing; `1` for a plain number.
    const WORK: usize = 1;
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
    /// Whether `self` and `other` are the same number bit for bit (value and every jet entry),
    /// for a type that carries no tape; `None` for one that can, whose results a caller must
    /// recompute rather than reuse (a reused result drops the tape records the recomputation
    /// writes). Keys the exact linear path's reuse of a Padé approximant: a deterministic
    /// function of equal bits gives equal bits.
    #[inline]
    fn same_bits_untaped(&self, _other: &Self) -> Option<bool> {
        None
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
    /// `sin` (the model language's, #2720): its jet lifted from the value, which a
    /// recording type (#2786) records as one primitive instead.
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = (self.val().sin(), self.val().cos());
        self.lift4(s, c, -s, -c, s)
    }
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
    /// `f` on the values of `args`: outputs that are constants of the computation (no
    /// derivative flows through them). A recording type (#2786) records a call that runs `f`
    /// again on every evaluation.
    fn host(f: HostFn, args: &[Self]) -> Vec<Self> {
        let vals: Vec<f64> = args.iter().map(|x| x.val()).collect();
        let mut out = Vec::new();
        f.run(&vals, &mut out);
        out.into_iter().map(Self::from_f64).collect()
    }
    /// One adaptive ODE segment, for a recording type (#2786) that records it as a call the
    /// solver's driver runs at run time; `None` — the generic integration — for every number
    /// type. `auto` is the driver's auto-switch state.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn ode_segment(
        _rhs: &dyn Fn(&[Self], &[Self], Self, &mut [Self]),
        _u0: &[Self],
        _t_span: (f64, f64),
        _params: &[Self],
        _saveat: &[f64],
        _interp_at: &[f64],
        _opts: &crate::ode::OdeSolverOptions,
        _auto: &mut dyn std::any::Any,
    ) -> Option<(Vec<crate::ode::SolPointG<Self>>, Vec<crate::ode::SolPointG<Self>>)> {
        None
    }
    /// `⌊self⌋` as a constant (no derivative): a recording type (#2786) records it, so it
    /// follows the value on every evaluation.
    #[inline]
    fn floor_c(self) -> Self {
        Self::from_f64(self.val().floor())
    }
    /// `f(self)` for a function given by its jet ([`JetFn`]): the jet lifted, on every type
    /// but a recording one (#2786), which records one primitive.
    #[inline]
    fn jet_unary(self, f: JetFn) -> Self {
        let [v, d1, d2, d3, d4] = f.jet(self.val());
        self.lift4(v, d1, d2, d3, d4)
    }
    /// The next ODE segment's span is `[t0, t1]`, times that may depend on the inputs (a
    /// lagged arrival, a zero-order window's end), value-only. A recording type (#2786) keeps
    /// them so the segment it records takes its span as run-time inputs ([`Self::take_span`])
    /// instead of the values it saw; every other type ignores it.
    #[inline]
    fn span_hint(_t0: Self, _t1: Self) {}
    /// The span [`Self::span_hint`] left, taken (`None` once taken, and on every type but a
    /// recording one).
    #[inline]
    fn take_span() -> Option<(Self, Self)> {
        None
    }
    /// A segment time's value, for the solver's `f64` span: the value on every type; on a
    /// recording type the value it saw, allowed only for a time [`Self::span_hint`] named (the
    /// segment it is recording takes the time itself).
    #[inline]
    fn span_value(self) -> f64 {
        self.val()
    }
    /// The value, for a decision the code takes on it — a discrete outcome of a
    /// [`host`](Self::host) analysis. A recording type checks it comes out the same on every
    /// evaluation.
    #[inline]
    fn br_value(self) -> f64 {
        self.val()
    }
    /// `body` applied `count` times to `x0`, `count` a whole number the recording reads at run
    /// time (#2786, `kernel_jit::record_repeat`): `None` for every type that is not a recording
    /// type (the caller loops), `Some(None)` when the recording cannot take the call.
    fn repeat(
        _x0: &[Self],
        _count: Self,
        _body: &dyn Fn(&[Self]) -> Vec<Self>,
    ) -> Option<Option<Vec<Self>>> {
        None
    }
    /// A steady state recorded as one fixed-point call (#2786, `kernel_jit::record_fixed`):
    /// `maps` the cycle maps the search runs (each recorded once, from `u0`), `solve` the
    /// search itself on the values, `held` the coordinates every map leaves where they are
    /// (the search holds them, so they take no derivative). `None` for every type that is not
    /// a recording type — the caller solves — and `Some(None)` when the recording cannot take
    /// the call (it fails).
    #[allow(clippy::type_complexity)]
    fn ss_fixed_point(
        _n: usize,
        _held: &[usize],
        _u0: &[f64],
        _maps: &[&dyn Fn(&[Self]) -> Option<Vec<Self>>],
        _solve: Box<crate::parser::model_parser::rhs_regcode::kernel_jit::SolveFn>,
    ) -> Option<Option<Vec<Self>>> {
        None
    }
    /// The value, for a computation whose result is checked not to depend on it — a
    /// parameter slot the tape shows constant in θ and η, evaluated with every other slot.
    /// A recording type (#2786) reads the value it saw, with no guard.
    #[inline]
    fn seen_value(self) -> f64 {
        self.val()
    }
    /// A magnitude to scale a computation by whose result does not depend on it — a probe
    /// step, a column scaling: the value's size, and the jets' where the type carries them. A
    /// recording type (#2786) keeps the one it saw; any scale gives the same exact result.
    #[inline]
    fn magnitude_hint(self) -> f64 {
        self.val().abs().max(self.jet_max_abs())
    }
    /// Of two spellings of one quantity, `skip` and `take` — `take` the one with refinement
    /// step `k` applied, a step that makes derivative order `k` exact (`linsolve`'s solve, a
    /// spectral chord): `take` for every type but a recording type (#2786), whose compiled
    /// variant takes it only when its derivative order (a reverse pass counting one more)
    /// reaches `k`, so no variant runs a step it does not need.
    #[inline]
    fn order_gate(_k: u8, _skip: Self, take: Self) -> Self {
        take
    }
    /// A segment of the exact linear path is integrated instead: a recording type (#2786)
    /// counts it into its plan, which reports it on every run; every other type's caller
    /// counts it itself.
    #[inline]
    fn note_linear_decline() {}
    /// Whether `self` and `other` are the same recorded value (a recording type, #2786): an
    /// `if`'s arm that left a variable as it was. `false` on every other type.
    #[inline]
    fn same_node(&self, _other: &Self) -> bool {
        false
    }
    /// The value as a constant: no derivative.
    #[inline]
    fn value_only(self) -> Self {
        Self::from_f64(self.val())
    }
    /// `exprel(z) = (e^z − 1)/z` at `self`, through the derivative orders the type carries
    /// (`orders`, as `ode::linear` decides them): the spectral propagation's `φ(λ, Δ)/Δ`. A
    /// recording type (#2786) records it as one primitive.
    #[inline]
    fn exprel(self, orders: usize) -> Self {
        let [g0, g1, g2, g3, g4] = crate::ode::linear::exprel_derivs(self.val(), orders);
        self.lift4(g0, g1, g2, g3, g4)
    }
    /// `A = ∂f/∂u` (row-major, `n = b.len()`) and `b = f(0)` of a right-hand side affine in the
    /// state, read off the recording by a recording type (#2786); `false` — the caller probes
    /// the RHS itself — for every number type.
    #[inline]
    fn affine_parts(
        _rhs: &dyn Fn(&[Self], &[Self], Self, &mut [Self]),
        _params: &[Self],
        _t: f64,
        _a: &mut [Self],
        _b: &mut [Self],
    ) -> bool {
        false
    }
    /// The regularized incomplete gamma `P(self, x)`, or `Q = 1 − P` when `upper`
    /// (`stats::special::regularized_gamma_p` / `_q`). A recording type (#2786) records it
    /// as one primitive rather than its value-dependent iteration.
    #[inline]
    fn gamma_pq(self, x: Self, upper: bool) -> Self {
        crate::stats::special::gamma_pq_g(self, x, upper)
    }
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
    /// `self cc other` on the values, for a branch the code takes on them. A recording
    /// type (#2786 prototype) turns it into a check its compiled recording re-evaluates,
    /// so a branch that would go the other way falls back instead of being baked in.
    #[inline]
    fn br_cmp(self, cc: BranchCmp, other: Self) -> bool {
        cc.holds(self.val(), other.val())
    }
    /// [`Self::br_cmp`] against a literal.
    #[inline]
    fn br_cmp_c(self, cc: BranchCmp, c: f64) -> bool {
        cc.holds(self.val(), c)
    }
    /// `base + self cc c` on the values, for a branch (see [`Self::br_cmp`]): a time shifted
    /// by a parameter, such as a dose's lagged arrival `t_dose + lag` against a time.
    #[inline]
    fn br_shift_cmp(self, base: f64, cc: BranchCmp, c: f64) -> bool {
        cc.holds(base + self.val(), c)
    }
    /// Whether this type records its branches (a recording type, #2786), so code that
    /// decides on an `f64` it derived from a parameter must decide through the branch API
    /// instead. `false` for every number type: their decisions stay as they are.
    const RECORDS_BRANCHES: bool = false;
    /// `|self| < tol` on the value, for a branch (see [`Self::br_cmp`]).
    #[inline]
    fn br_abs_lt(self, tol: f64) -> bool {
        self.val().abs() < tol
    }
    /// `|self − other| < tol` on the values, for a branch (see [`Self::br_cmp`]).
    #[inline]
    fn br_absdiff_lt(self, other: Self, tol: f64) -> bool {
        (self.val() - other.val()).abs() < tol
    }
    /// An event boundary of the computation being evaluated: a recording type (#2786)
    /// cuts its recording here into chunks it compiles once and shares across subjects.
    /// Nothing for a number.
    #[inline(always)]
    fn chunk_mark() {}
    /// `if x cc y { a } else { b }` on the values, as a selection rather than a branch: a
    /// recording type (#2786) records it, so the choice is made again on every evaluation —
    /// for a condition that changes inside the code it records (a time window inside an ODE
    /// step), where a branch's guard would fail.
    #[inline]
    fn select_if(cc: BranchCmp, x: Self, y: Self, a: Self, b: Self) -> Self {
        if cc.holds(x.val(), y.val()) {
            a
        } else {
            b
        }
    }
    /// Whether the value is `NaN`, for a branch (see [`Self::br_cmp`]).
    #[inline]
    fn br_is_nan(self) -> bool {
        self.val().is_nan()
    }

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

    /// `x₁·y₁ + x₂·y₂`: a product rule's tangent (`g·v′ + v·g′`). The reverse tape
    /// (`sens::rtape::R`) overrides it to record **one** node with the four partials instead
    /// of three (two products and a sum, #2481 §2.2); every other type keeps exactly this
    /// arithmetic, so its values do not move.
    #[inline]
    fn dot2(x1: Self, y1: Self, x2: Self, y2: Self) -> Self {
        x1 * y1 + x2 * y2
    }

    /// `((x₁·y₁ + x₂·y₂) + x₃·y₃) + x₄·y₄`, left to right: a second-order product rule's
    /// Hessian entry. As [`Self::dot2`]: one node on the reverse tape, this arithmetic
    /// everywhere else.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dot4(x1: Self, y1: Self, x2: Self, y2: Self, x3: Self, y3: Self, x4: Self, y4: Self) -> Self {
        x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4
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
    fn lu_factor_hook(_m: &mut [Self], _piv: &mut [Self], _n: usize) -> Option<bool> {
        None
    }

    /// Solve hook paired with [`Self::lu_factor_hook`]; `true` when handled.
    #[inline]
    fn lu_solve_hook(_m: &[Self], _piv: &[Self], _n: usize, _b: &mut [Self]) -> bool {
        false
    }
}

impl PkNum for f64 {
    const SECOND_ORDER: bool = false;
    #[inline]
    fn same_bits_untaped(&self, other: &Self) -> Option<bool> {
        Some(self.to_bits() == other.to_bits())
    }
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
    fn sin(self) -> Self {
        f64::sin(self)
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
            const JET_ORDER: usize = if $so { 2 } else { 1 };
            #[inline]
            fn same_bits_untaped(&self, other: &Self) -> Option<bool> {
                Some($ty::same_bits(self, other))
            }
            /// Every jet entry `+0.0`: the constant of its value, bit for bit (no tape to
            /// carry). A `NaN` or `−0.0` entry reads non-constant, the conservative side.
            #[inline]
            fn is_constant(self) -> bool {
                $ty::same_bits(&self, &$ty::constant(self.val()))
            }
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
        /// Forward-mode tangent directions (#2786): each slot holds the base layout's lanes,
        /// then `tan` more blocks of them, block `j` the tangent along direction `j`
        /// (`rhs_ir::tangent_packed`). `0` for every number type.
        pub(crate) tan: u8,
    }

    impl Layout {
        pub(crate) const F64: Layout = Layout {
            n: 0,
            m: 0,
            th: false,
            wide: false,
            tan: 0,
        };
        pub(crate) const fn lanes(&self) -> usize {
            self.base_lanes() * (1 + self.tan as usize)
        }
        /// The lanes of one block: the value, the tangents, the payload.
        pub(crate) const fn base_lanes(&self) -> usize {
            1 + self.n as usize + self.m as usize
        }
        /// The layout without its tangent blocks.
        pub(crate) const fn base(&self) -> Layout {
            Layout { tan: 0, ..*self }
        }
        /// The highest derivative order the lanes carry: 2 with a second-order payload, 1
        /// with first-order tangents alone, 0 for `f64`; tangent blocks one more.
        pub(crate) const fn order(&self) -> usize {
            (if self.m > 0 {
                2
            } else if self.n > 0 {
                1
            } else {
                0
            }) + (self.tan > 0) as usize
        }
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
