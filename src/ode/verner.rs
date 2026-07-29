//! Verner's 7(6) explicit Runge-Kutta stepper — the **high-order** option, for fits that are
//! accuracy-limited rather than stability-limited.
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
//! Verner's "most efficient" 7(6) pair: 10 stages, order 7 with an embedded order-6 error
//! estimate. Coefficients are transcribed from `OrdinaryDiffEq.jl`'s `Vern7Tableau`, and — as
//! for the Rosenbrock tableaus — the digits are pinned by a **measured convergence order** test
//! rather than by eye. SciML's work-precision benchmarks put the Verner family ahead of DOP853
//! over roughly `1e-8 … 1e-12`, which is the band a NONMEM-equivalent `TOL=9` fit lives in.
//!
//! # Dense output
//!
//! Verner's own continuous extension needs three extra stages and a coefficient block larger
//! than the method itself, so this stepper interpolates with the **cubic Hermite** built from
//! the derivatives at both ends of the step — the same interpolant [`super::solver`]'s RK45
//! uses. Unlike RK45 the pair is not FSAL, so the end derivative is not free: it is evaluated
//! only when a caller has actually asked for in-step reads
//! ([`Stepper::set_dense_required`](super::solver::Stepper::set_dense_required)), keeping the
//! ordinary `saveat` path at exactly 10 evaluations per step.

use super::solver::{hermite_g, scale_tol, OdeSolverOptions, Stepper};
use crate::sens::num::PkNum;

/// Stage count of the 7(6) pair.
const STAGES: usize = 10;

/// Stage times `α_i` (`α₁ = 0`; stages 9 and 10 both sit at the step end).
const C: [f64; STAGES] = [
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
];

/// Stage weights `a_ij` (`j < i`), row-major and zero-padded. The zeros are structural — the
/// pair skips stage 2 in every later stage, which is where its efficiency comes from.
const A: [[f64; STAGES]; STAGES] = [
    [0.0; STAGES],
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
];

/// Order-7 solution weights (`b₂ = b₃ = b₁₀ = 0`).
const B: [f64; STAGES] = [
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
];

/// Embedded error weights `b − b̂` (`btilde` in the source tableau).
const B_TILDE: [f64; STAGES] = [
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
];

/// Verner 7(6) as a [`Stepper`]. Shares every driver with the other methods.
pub(crate) struct Vern7Stepper<T> {
    n: usize,
    ks: Vec<Vec<T>>,
    u_tmp: Vec<T>,
    u_new: Vec<T>,
    /// `f(u_new, t + h)` — only evaluated when a caller needs in-step reads.
    f_end: Vec<T>,
    dense_required: bool,
}

impl<T: PkNum> Vern7Stepper<T> {
    pub(crate) fn new(n: usize) -> Self {
        let z = T::from_f64(0.0);
        Self {
            n,
            ks: vec![vec![z; n]; STAGES],
            u_tmp: vec![z; n],
            u_new: vec![z; n],
            f_end: vec![z; n],
            dense_required: false,
        }
    }
}

impl<T: PkNum> Stepper<T> for Vern7Stepper<T> {
    // Indexed loops walk `u` alongside ten stage vectors; zipping would obscure the tableau.
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
        let h = T::from_f64(dt);

        for stage in 0..STAGES {
            if stage == 0 {
                rhs(u, params, t, &mut self.ks[0]);
                continue;
            }
            for i in 0..n {
                let mut acc = T::from_f64(0.0);
                for j in 0..stage {
                    let w = A[stage][j];
                    if w != 0.0 {
                        acc = acc + self.ks[j][i] * T::from_f64(w);
                    }
                }
                self.u_tmp[i] = u[i] + h * acc;
            }
            rhs(&self.u_tmp, params, t + C[stage] * dt, &mut self.ks[stage]);
        }

        for i in 0..n {
            let mut acc = T::from_f64(0.0);
            for stage in 0..STAGES {
                if B[stage] != 0.0 {
                    acc = acc + self.ks[stage][i] * T::from_f64(B[stage]);
                }
            }
            self.u_new[i] = u[i] + h * acc;
        }

        // Error estimate on **values only** — the step sequence must not depend on derivative
        // components (see the `Stepper` contract).
        let mut err_norm = 0.0;
        for i in 0..n {
            let mut e = 0.0;
            for stage in 0..STAGES {
                if B_TILDE[stage] != 0.0 {
                    e += B_TILDE[stage] * self.ks[stage][i].val();
                }
            }
            let err_i = dt * e;
            let scale = scale_tol(opts.abstol, opts.reltol, self.u_new[i].val(), u[i].val());
            err_norm += (err_i / scale) * (err_i / scale);
        }

        // The Hermite interpolant needs the derivative at the step end, which this pair — unlike
        // FSAL RK45 — does not produce as a by-product. Pay for it only when someone reads
        // inside steps.
        if self.dense_required {
            rhs(&self.u_new, params, t + dt, &mut self.f_end);
        }

        (err_norm / n as f64).sqrt()
    }

    fn u_new(&self) -> &[T] {
        &self.u_new
    }

    fn interpolate_component(&self, theta: f64, u_old: &[T], dt: f64, i: usize) -> T {
        hermite_g(
            theta,
            dt,
            u_old[i],
            self.ks[0][i],
            self.u_new[i],
            self.f_end[i],
        )
    }

    fn on_accept(&mut self) {}

    fn err_exp(&self) -> f64 {
        // 1/(p̂+1) for the embedded order-6 estimate.
        1.0 / 7.0
    }

    fn attempt_usable(&self) -> bool {
        true
    }

    fn set_dense_required(&mut self, yes: bool) {
        self.dense_required = yes;
    }
}
