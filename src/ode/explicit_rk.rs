//! Explicit Runge-Kutta steppers driven by a Butcher tableau — everything that is not RK45 and
//! not stiff.
//!
//! One ships here — `vern7`, the high-order option for the **accuracy-limited** regime (see
//! below). The tableau/stepper split is deliberate: another method is one `const` away.
//!
//! # Which problem this solves
//!
//! An accuracy-limited fit spends its steps meeting `ode_reltol`, not staying stable: nearly
//! every step is accepted and nothing is min-step clamped. A stiff method does not help there
//! (it buys stability the fit already has, and pays a Jacobian and a factorization per step for
//! it); what helps is a **higher order**, because the step count scales as `tol^(−1/p)`. Going
//! from RK45's `p = 5` to `p = 7` at `ode_reltol = 1e-9` predicts roughly a third of the steps,
//! against 10 stages per step instead of 6 — a net win that grows as the tolerance tightens.
//!
//! That regime is not hypothetical: the Savic transit NONMEM anchor at `TOL=9`-equivalent
//! accuracy sits squarely in it (`tests/transit_nonmem_anchor.rs`, ~97 % of steps accepted,
//! zero min-step clamps), which is exactly why a stiff method is the wrong tool for it.
//!
//! # Method
//!
//! **Vern7** — Verner's "most efficient" 7(6) pair: 10 stages, order 7 with an embedded order-6
//! error estimate.
//!
//! ## Measured negative result: Tsit5 is not here on purpose
//!
//! Tsitouras 5(4) (2011, *CAMWA* 62:770–775) is the obvious same-cost upgrade over the default
//! Dormand-Prince tableau — same 7 stages and FSAL structure, smaller truncation-error
//! constant. It was implemented and measured against the Savic transit anchor
//! (`tests/transit_nonmem_anchor.rs`) before being removed again: at `1e-9` it needed 3 720
//! accepted steps against RK45's 3 940 — a ~6 % gain — and was *slower* in wall-clock, because
//! [`super::solver`]'s RK45 stepper is hand-unrolled while this generic tableau-driven one
//! walks index loops. Vern7's 64 % step reduction clears that overhead; Tsit5's 6 % does not.
//!
//! Reviving it is a single `ErkTableau` const, and worth doing if the generic stepper is ever
//! made competitive with the unrolled one — or, better, if RK45 itself is expressed as an
//! `ErkTableau` (it fits this struct exactly: 7 stages, FSAL) and the unrolled copy retired.
//! That would unify every explicit method behind one implementation, at the cost of RK45 no
//! longer being bit-identical to its historical trajectory — a deliberate re-anchoring
//! decision, not a drive-by. Coefficients are transcribed from `OrdinaryDiffEq.jl`'s `Vern7Tableau`, and — as
//! for the Rosenbrock tableaus — the digits are pinned by a **measured convergence order** test
//! rather than by eye. SciML's work-precision benchmarks put the Verner family ahead of DOP853
//! over roughly `1e-8 … 1e-12`, which is the band a NONMEM-equivalent `TOL=9` fit lives in.
//!
//! # Dense output
//!
//! Both interpolate with the **cubic Hermite** built from the derivatives at each end of the
//! step — the same interpolant [`super::solver`]'s RK45 uses. For an FSAL method (Tsit5) the
//! end derivative is the last stage, so this is free and order-matched to the step. Vern7 is
//! not FSAL and its own continuous extension would need three extra stages and a coefficient
//! block larger than the method, so it pays one extra evaluation — and only when a caller has
//! actually asked for in-step reads
//! ([`Stepper::set_dense_required`](super::solver::Stepper::set_dense_required)), keeping the
//! ordinary `saveat` path at exactly 10 evaluations per step. Its in-step reads are therefore
//! 3rd-order accurate while its steps are 7th-order accurate.

use super::solver::{hermite_g, scale_tol, OdeSolverOptions, Stepper};
use crate::sens::num::PkNum;

/// Widest tableau here (Vern7's 10 stages).
const MAX_STAGES: usize = 10;

/// An explicit Runge-Kutta pair: stages, weights, and how its error estimate is scaled.
pub(crate) struct ErkTableau {
    /// Number of stages `s`.
    stages: usize,
    /// Stage weights `a_ij` (`j < i`), row-major and zero-padded.
    a: [[f64; MAX_STAGES]; MAX_STAGES],
    /// Solution weights. For an FSAL pair this equals row `s−1` of `a`.
    b: [f64; MAX_STAGES],
    /// Embedded error weights `b − b̂`.
    b_tilde: [f64; MAX_STAGES],
    /// Stage times `α_i` (`α₁ = 0`).
    c: [f64; MAX_STAGES],
    /// I-controller exponent `1/(p̂+1)`.
    err_exp: f64,
    /// First Same As Last: the final stage is `f(u_new, t + h)`, so it doubles as the next
    /// step's first stage *and* as the end derivative the Hermite interpolant needs. A
    /// non-FSAL pair pays one extra evaluation for each of those.
    fsal: bool,
}

/// Verner "most efficient" 7(6) — 10 stages, order 7, not FSAL. Coefficients from
/// `OrdinaryDiffEq.jl`'s `Vern7Tableau`. The structural zeros (every later stage skips stage 2)
/// are where its efficiency comes from.
pub(crate) const VERN7: ErkTableau = ErkTableau {
    stages: 10,
    a: [
        [0.0; MAX_STAGES],
        [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -1.076_790_123_456_79,
            1.185_679_012_345_679,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.040_833_333_333_333_33,
            0.0,
            0.1225,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.638_913_923_625_572_6,
            0.0,
            -2.455_672_638_223_657,
            2.272_258_714_598_084,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -2.661_577_375_018_757_2,
            0.0,
            10.804_513_886_456_137,
            -8.353_914_657_396_2,
            0.820_487_594_956_657,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            6.067_741_434_696_772,
            0.0,
            -24.711_273_635_911_088,
            20.427_517_930_788_895,
            -1.906_157_978_816_647_2,
            1.006_172_249_242_068,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            12.054_670_076_253_203,
            0.0,
            -49.754_784_950_468_99,
            41.142_888_638_604_674,
            -4.461_760_149_974_004,
            2.042_334_822_239_175,
            -0.098_348_436_654_061_07,
            0.0,
            0.0,
            0.0,
        ],
        [
            10.138_146_522_881_808,
            0.0,
            -42.641_136_031_717_5,
            35.763_840_039_922_57,
            -4.348_022_840_392_907_5,
            2.009_862_268_377_035_7,
            0.348_749_046_033_827_2,
            -0.271_439_005_104_831_27,
            0.0,
            0.0,
        ],
        [
            -45.030_072_034_298_676,
            0.0,
            187.327_243_765_458_9,
            -154.028_823_693_501_86,
            18.564_653_063_475_36,
            -7.141_809_679_295_079,
            1.308_808_578_161_378_7,
            0.0,
            0.0,
            0.0,
        ],
    ],
    b: [
        0.047_155_618_486_272_22,
        0.0,
        0.0,
        0.257_505_642_984_341_53,
        0.262_166_539_774_126_24,
        0.152_160_926_567_385_58,
        0.493_996_917_003_248_5,
        -0.294_303_117_140_325_03,
        0.081_317_472_324_951_11,
        0.0,
    ],
    b_tilde: [
        0.002_547_011_879_931_045,
        0.0,
        0.0,
        -0.009_658_394_872_795_75,
        0.042_064_709_756_396_91,
        -0.066_682_243_746_930_1,
        0.265_009_746_462_128_1,
        -0.294_303_117_140_325_03,
        0.081_317_472_324_951_11,
        -0.020_295_184_663_356_28,
    ],
    c: [
        0.0,
        0.005,
        0.108_888_888_888_888_88,
        0.163_333_333_333_333_33,
        0.4555,
        0.609_509_448_997_838_1,
        0.884,
        0.925,
        1.0,
        1.0,
    ],
    err_exp: 1.0 / 7.0,
    fsal: false,
};

/// Any [`ErkTableau`] as a [`Stepper`]. Shares every driver with the other methods.
pub(crate) struct ErkStepper<T> {
    tab: &'static ErkTableau,
    n: usize,
    ks: Vec<Vec<T>>,
    u_tmp: Vec<T>,
    u_new: Vec<T>,
    /// `f(u_new, t + h)` for a non-FSAL pair; unused when the last stage already is it.
    f_end: Vec<T>,
    /// Whether `ks[0]` already holds `f(u, t)` for the current `(u, t)`.
    have_k1: bool,
    dense_required: bool,
}

impl<T: PkNum> ErkStepper<T> {
    pub(crate) fn new(n: usize, tab: &'static ErkTableau) -> Self {
        let z = T::from_f64(0.0);
        Self {
            tab,
            n,
            ks: vec![vec![z; n]; tab.stages],
            u_tmp: vec![z; n],
            u_new: vec![z; n],
            f_end: vec![z; n],
            have_k1: false,
            dense_required: false,
        }
    }
}

impl<T: PkNum> Stepper<T> for ErkStepper<T> {
    // Indexed loops walk `u` alongside every stage vector; zipping would obscure the tableau.
    #[allow(clippy::needless_range_loop)]
    fn attempt(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        let n = self.n;
        let s = self.tab.stages;
        let h = T::from_f64(dt);

        // `ks[0] = f(u, t)`. Carried across a *rejected* step (which leaves `(u, t)` in place)
        // and, for an FSAL pair, across an accepted one too.
        if !self.have_k1 {
            rhs(u, params, t, &mut self.ks[0]);
            self.have_k1 = true;
        }

        // An FSAL pair's last stage is `f(u_new, t + h)`, so it is evaluated after the solution
        // below rather than in this loop.
        let explicit_stages = if self.tab.fsal { s - 1 } else { s };
        for stage in 1..explicit_stages {
            for i in 0..n {
                let mut acc = T::from_f64(0.0);
                for j in 0..stage {
                    let w = self.tab.a[stage][j];
                    if w != 0.0 {
                        acc = acc + self.ks[j][i] * T::from_f64(w);
                    }
                }
                self.u_tmp[i] = u[i] + h * acc;
            }
            let ti = t + self.tab.c[stage] * dt;
            rhs(&self.u_tmp, params, ti, &mut self.ks[stage]);
        }

        for i in 0..n {
            let mut acc = T::from_f64(0.0);
            for stage in 0..s {
                if self.tab.b[stage] != 0.0 {
                    acc = acc + self.ks[stage][i] * T::from_f64(self.tab.b[stage]);
                }
            }
            self.u_new[i] = u[i] + h * acc;
        }

        if self.tab.fsal {
            rhs(&self.u_new, params, t + dt, &mut self.ks[s - 1]);
        }

        // Error estimate on **values only** — the step sequence must not depend on derivative
        // components (see the `Stepper` contract).
        let mut err_norm = 0.0;
        for i in 0..n {
            let mut e = 0.0;
            for stage in 0..s {
                if self.tab.b_tilde[stage] != 0.0 {
                    e += self.tab.b_tilde[stage] * self.ks[stage][i].val();
                }
            }
            let err_i = dt * e;
            let scale = scale_tol(opts.abstol, opts.reltol, self.u_new[i].val(), u[i].val());
            err_norm += (err_i / scale) * (err_i / scale);
        }

        // A non-FSAL pair has to buy the end derivative the Hermite interpolant needs — but
        // only when someone actually reads inside steps.
        if !self.tab.fsal && self.dense_required {
            rhs(&self.u_new, params, t + dt, &mut self.f_end);
        }

        (err_norm / n as f64).sqrt()
    }

    fn u_new(&self) -> &[T] {
        &self.u_new
    }

    fn interpolate_component(&self, theta: f64, u_old: &[T], dt: f64, i: usize) -> T {
        let d_end = if self.tab.fsal {
            self.ks[self.tab.stages - 1][i]
        } else {
            self.f_end[i]
        };
        hermite_g(theta, dt, u_old[i], self.ks[0][i], self.u_new[i], d_end)
    }

    fn on_accept(&mut self) {
        if self.tab.fsal {
            // The last stage is `f` at the accepted end point — exactly the next step's first.
            let last = self.tab.stages - 1;
            self.ks.swap(0, last);
        } else {
            // `(u, t)` moved and nothing was evaluated there, so `ks[0]` is stale.
            self.have_k1 = false;
        }
    }

    fn err_exp(&self) -> f64 {
        self.tab.err_exp
    }

    fn attempt_usable(&self) -> bool {
        true
    }

    fn set_dense_required(&mut self, yes: bool) {
        self.dense_required = yes;
    }
}
