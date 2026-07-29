//! Linearly-implicit **Rosenbrock** steppers — the stiff counterpart to the explicit
//! Dormand-Prince RK45 in [`super::solver`].
//!
//! Three methods ship, selected by [`OdeMethod`] (`[fit_options] ode_method`):
//!
//! | method | order (embedded) | stages | use for |
//! |---|---|---|---|
//! | [`OdeMethod::Rosenbrock23`] | 2(3) | 3 | crude tolerances, very rough / discontinuous problems |
//! | [`OdeMethod::Rodas4`] | 4(3) | 6 | the stiff workhorse at PK tolerances (1e-6 … 1e-9) |
//! | [`OdeMethod::Rodas5P`] | 5(4) | 8 | tight tolerances (≤ 1e-9), the FOCEI-OFV-matching regime |
//!
//! # Why Rosenbrock and not BDF / ESDIRK / Radau
//!
//! Every ferx integrator has to run **twice over**: at `T = f64` for the prediction and at
//! `T = Dual2<N>` for the analytic sensitivities, from one generic body (`crate::sens`).
//! Rosenbrock methods are *linearly implicit* — one Jacobian, one LU factorization, and a
//! fixed number of back-solves per step, with **no Newton iteration**. A step is therefore a
//! straight-line sequence of `+ − × ÷`, so instantiating it at `Dual2` differentiates the
//! discrete step exactly (discretize-then-differentiate) with no convergence-dependent
//! semantics. A Newton-based method (BDF, ESDIRK, Radau) would need either the
//! implicit-function-theorem correction or a fully converged inner solve before its
//! derivatives were even meaningful.
//!
//! The cost argument points the same way for PK-sized systems (2–20 states): the LU is
//! cheap, so BDF's "one solve per step" advantage buys nothing, and BDF loses L-stability
//! above order 2.
//!
//! # Formulation
//!
//! With `J = ∂f/∂u` and `f_t = ∂f/∂t` at the step start, `W = I/(γh) − J`, and for stages
//! `i = 1…s`:
//!
//! ```text
//! u_i  = u + Σ_{j<i} a_ij k_j
//! W k_i = f(u_i, t + α_i h) + h γ_i f_t + (1/h) Σ_{j<i} c_ij k_j
//! u_new = u + Σ_i b_i k_i
//! ```
//!
//! This is Hairer & Wanner's `RODAS` convention (*Solving ODEs II*, IV.7), and the Rodas
//! tableaus are transcribed from `OrdinaryDiffEq.jl`'s (`Rodas4Tableau` / `Rodas5PTableau`;
//! Rodas5P is Steinebach, *BIT* 63:27, 2023). Both are **stiffly accurate** with
//! `b_i = a_{s,i}` and `b_s = 1`, so `u_new` is the last stage's `u_s` plus `k_s`, and the
//! embedded lower-order solution is `u_new − k_s` — i.e. the error estimate *is* the last
//! stage increment. [`OdeMethod::Rosenbrock23`] is Shampine & Reichelt's `ode23s`
//! (*SIAM J. Sci. Comput.* 18:1, 1997), which reuses `f(u, t)` across stages and so does not
//! fit the tableau form; it is stepped by its own routine below over the same `W`.
//!
//! The tableau digits and the stage-3 `h·γ·f_t` scaling are pinned by measured convergence
//! order in this module's tests — a mistyped coefficient shows up immediately as order loss.
//!
//! # Shared discipline with RK45
//!
//! Step-size control, the `min_dt` force-accept, the non-finite divergence break, the
//! `saveat` clipping and [`OdeSolverStats`] accounting are all deliberately identical to
//! [`super::solver::solve_ode`], and — as there — **every comparison branches on
//! [`PkNum::val`] only**, so the derivative rides a step sequence fixed by the value part.
//! That discipline extends here to the LU pivot selection (see
//! [`crate::sens::linsolve::lu_factor_in_place_g`]).

use super::solver::{scale_tol, OdeMethod, OdeSolverOptions};
use crate::sens::linsolve::{lu_factor_in_place_g, lu_solve_in_place_g};
use crate::sens::num::PkNum;

/// Widest tableau shipped (Rodas5P). Sizing the coefficient block as a fixed array keeps the
/// tableaus `const` and allocation-free.
const MAX_ROS_STAGES: usize = 8;

/// Relative perturbation floor for the finite-difference Jacobian / time derivative:
/// `δ = √(ε·max(FD_FLOOR, |x|))`, Hairer's `rodas.f` `numjac` choice. The floor keeps the
/// step from collapsing onto `√ε·0 = 0` for a state sitting at (or near) zero — a depot
/// before its dose, or any compartment that has not been reached yet.
const FD_FLOOR: f64 = 1e-5;

/// A Rosenbrock tableau in the `RODAS` convention (see the module docs). Rows `i` use only
/// entries `j < i`; the unused tail is zero-padded to [`MAX_ROS_STAGES`].
struct RosTableau {
    /// Stage count `s`.
    stages: usize,
    /// `a_ij` — stage state weights.
    a: [[f64; MAX_ROS_STAGES]; MAX_ROS_STAGES],
    /// `c_ij` — stage coupling weights, entering as `(1/h)·Σ c_ij k_j`.
    c: [[f64; MAX_ROS_STAGES]; MAX_ROS_STAGES],
    /// `α_i` — stage `i` evaluates `f` at `t + α_i·h`.
    alpha: [f64; MAX_ROS_STAGES],
    /// `γ_i` — coefficient of the `h·f_t` term in stage `i`'s right-hand side.
    d: [f64; MAX_ROS_STAGES],
    /// Diagonal `γ`: `W = I/(γh) − J`.
    gamma: f64,
    /// I-controller exponent `1/(p̂+1)` for the embedded pair (`p̂` = embedded order).
    err_exp: f64,
    /// Dense-output (continuous extension) block: row `j` combines the stage increments into
    /// the interpolation coefficient `κ_j = Σ_i h[j][i]·k_i`. See [`RosTableau::interpolate`].
    h: [[f64; MAX_ROS_STAGES]; MAX_INTERP_ROWS],
    /// Rows of `h` this method uses — 2 for Rodas4 (quadratic in `Θ`), 3 for Rodas5P (cubic).
    h_rows: usize,
}

/// Widest continuous extension shipped (Rodas5P's 3 rows).
const MAX_INTERP_ROWS: usize = 3;

impl RosTableau {
    /// Continuous extension of one accepted step, evaluated for a single state component:
    ///
    /// ```text
    /// u(t + Θh) = (1−Θ)·y₀ + Θ·(y₁ + (1−Θ)·(κ₁ + Θ·(κ₂ + Θ·κ₃)))     (κ₃ only when h_rows = 3)
    /// ```
    ///
    /// with `κ_j = Σ_i h[j][i]·k_i` over the step's stage increments. `Θ = 0` gives `y₀` and
    /// `Θ = 1` gives `y₁` exactly, so the interpolant is continuous with the saved states at
    /// both ends of every step.
    ///
    /// Evaluated per component and on demand: a step that is never interpolated pays nothing,
    /// and one that is pays `O(h_rows · s)` for the components asked for, with no allocation.
    // Indexed over stages to combine one component across several stage vectors; zipping
    // would obscure the `Σ_i h[j][i]·k_i` this transcribes.
    #[allow(clippy::needless_range_loop)]
    fn interpolate<T: PkNum>(&self, theta: f64, y0: T, y1: T, ks: &[Vec<T>], i: usize) -> T {
        let th = T::from_f64(theta);
        let th1 = T::from_f64(1.0 - theta);
        let kappa = |row: usize| -> T {
            let mut acc = T::from_f64(0.0);
            for j in 0..self.stages {
                acc = acc + ks[j][i] * T::from_f64(self.h[row][j]);
            }
            acc
        };
        let inner = if self.h_rows >= 3 {
            kappa(0) + th * (kappa(1) + th * kappa(2))
        } else {
            kappa(0) + th * kappa(1)
        };
        th1 * y0 + th * (y1 + th1 * inner)
    }
}

/// Rodas4 (Hairer & Wanner, *Solving ODEs II* IV.7): 6 stages, order 4(3), L-stable and
/// stiffly accurate.
const RODAS4: RosTableau = RosTableau {
    stages: 6,
    a: [
        [0.0; 8],
        [1.544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.946_678_528_081_582_6,
            0.255_701_169_898_328_4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            3.314_825_187_068_521,
            2.896_124_015_972_201,
            0.998_641_913_997_781_7,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.221_224_509_226_641,
            6.019_134_481_288_629,
            12.537_083_329_320_87,
            -0.687_886_036_105_895,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.221_224_509_226_641,
            6.019_134_481_288_629,
            12.537_083_329_320_87,
            -0.687_886_036_105_895,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [0.0; 8],
        [0.0; 8],
    ],
    c: [
        [0.0; 8],
        [-5.6688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -2.430_093_356_833_875,
            -0.206_359_915_709_191_5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.107_352_905_815_137_5,
            -9.594_562_251_023_355,
            -20.470_286_148_096_16,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            7.496_443_313_967_647,
            -10.246_804_314_643_52,
            -33.999_903_528_199_05,
            11.708_908_932_061_6,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            8.083_246_795_921_522,
            -7.981_132_988_064_893,
            -31.521_594_328_743_71,
            16.319_305_431_231_36,
            -6.058_818_238_834_054,
            0.0,
            0.0,
            0.0,
        ],
        [0.0; 8],
        [0.0; 8],
    ],
    alpha: [0.0, 0.386, 0.21, 0.63, 1.0, 1.0, 0.0, 0.0],
    d: [0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0, 0.0, 0.0],
    gamma: 0.25,
    err_exp: 1.0 / 4.0,
    h: [
        [
            10.126_235_083_445_86,
            -7.487_995_877_610_167,
            -34.800_918_615_557_47,
            -7.992_771_707_568_823,
            1.025_137_723_295_662,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.676_280_339_280_125_3,
            6.087_714_651_680_015,
            16.430_843_208_924_78,
            24.767_225_114_183_86,
            -6.594_389_125_716_872,
            0.0,
            0.0,
            0.0,
        ],
        [0.0; 8],
    ],
    h_rows: 2,
};

/// Rodas5P (Steinebach 2023): 8 stages, order 5(4), L-stable and stiffly accurate. Built for
/// better behaviour than Rodas5 at tight tolerances and on index-1 DAE-like stiffness.
const RODAS5P: RosTableau = RosTableau {
    stages: 8,
    a: [
        [0.0; 8],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            2.849_394_379_747_939,
            0.458_422_422_044_639_23,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -6.954_028_509_809_101,
            2.489_845_061_869_568,
            -10.358_996_098_473_584,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.802_998_627_562_896_4,
            0.507_246_473_622_820_6,
            -0.398_831_254_177_052_4,
            -0.047_211_872_304_046_41,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -7.502_846_399_306_121,
            2.561_846_144_803_919,
            -11.627_539_656_261_098,
            -0.182_687_676_599_422_56,
            0.030_198_172_008_377_946,
            0.0,
            0.0,
            0.0,
        ],
        [
            -7.502_846_399_306_121,
            2.561_846_144_803_919,
            -11.627_539_656_261_098,
            -0.182_687_676_599_422_56,
            0.030_198_172_008_377_946,
            1.0,
            0.0,
            0.0,
        ],
        [
            -7.502_846_399_306_121,
            2.561_846_144_803_919,
            -11.627_539_656_261_098,
            -0.182_687_676_599_422_56,
            0.030_198_172_008_377_946,
            1.0,
            1.0,
            0.0,
        ],
    ],
    c: [
        [0.0; 8],
        [-14.155_112_264_123_755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -17.972_960_358_859_52,
            -2.859_693_295_451_294,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            147.121_502_757_117_16,
            -1.412_214_027_182_13,
            71.689_402_513_023_58,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            165.435_170_248_716_76,
            -0.459_282_345_649_112_6,
            42.909_383_369_586_03,
            -5.961_986_721_573_306,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            24.854_864_614_690_072,
            -3.000_922_700_283_218_6,
            47.493_111_002_076_8,
            5.581_419_782_155_812_5,
            -0.661_069_182_524_947_1,
            0.0,
            0.0,
            0.0,
        ],
        [
            30.912_732_140_285_99,
            -3.120_824_334_993_797_4,
            77.799_546_460_708_92,
            34.286_460_282_947_83,
            -19.097_331_116_725_623,
            -28.087_943_162_872_662,
            0.0,
            0.0,
        ],
        [
            37.802_771_233_905_63,
            -3.257_196_902_907_227_6,
            112.269_188_494_963_27,
            66.934_723_124_404_7,
            -40.066_189_370_910_02,
            -54.667_802_628_779_68,
            -9.488_616_523_096_27,
            0.0,
        ],
    ],
    alpha: [
        0.0,
        0.635_812_689_582_870_4,
        0.409_579_839_339_753_5,
        0.976_930_672_506_071_6,
        0.428_840_360_955_866_4,
        1.0,
        1.0,
        1.0,
    ],
    d: [
        0.211_937_563_194_290_14,
        -0.423_875_126_388_580_27,
        -0.338_462_712_623_592_4,
        1.804_645_287_288_273_4,
        2.325_825_639_765_069,
        0.0,
        0.0,
        0.0,
    ],
    gamma: 0.211_937_563_194_290_14,
    err_exp: 1.0 / 5.0,
    h: [
        [
            25.948_786_856_663_858,
            -2.557_972_484_584_623_5,
            10.433_815_404_888_879,
            -2.367_925_102_268_520_4,
            0.524_948_541_321_073,
            1.124_108_831_045_040_4,
            0.427_287_619_443_187_4,
            -0.172_022_210_701_554_93,
        ],
        [
            -9.915_688_506_951_71,
            -0.968_994_459_411_515_4,
            3.043_803_724_297_845_3,
            -24.495_224_566_215_796,
            20.176_138_334_709_044,
            15.980_663_614_246_51,
            -6.789_040_303_419_874,
            -6.710_236_069_923_372,
        ],
        [
            11.419_903_575_922_262,
            2.887_964_514_613_699_4,
            72.921_379_959_960_29,
            80.125_118_346_226_43,
            -52.072_871_366_152_654,
            -59.789_936_252_667_29,
            -0.155_826_842_827_519_13,
            4.883_087_185_713_722,
        ],
    ],
    h_rows: 3,
};

/// `γ = 1/(2+√2)` — the (L-stable) diagonal of `ode23s`.
const ROS23_GAMMA: f64 = 1.0 / (2.0 + std::f64::consts::SQRT_2);
/// `c₃₂ = 6+√2` — the stage-3 coupling of `ode23s`.
const ROS23_C32: f64 = 6.0 + std::f64::consts::SQRT_2;
/// I-controller exponent for `ode23s`: the 2nd-order solution's local error is `O(h³)`.
const ROS23_ERR_EXP: f64 = 1.0 / 3.0;

/// Per-method `γ`, needed before a step to build `W`.
fn method_gamma(method: OdeMethod) -> f64 {
    match method {
        OdeMethod::Rosenbrock23 => ROS23_GAMMA,
        OdeMethod::Rodas4 => RODAS4.gamma,
        OdeMethod::Rodas5P => RODAS5P.gamma,
        other => unreachable!("{other:?} does not use the Rosenbrock stepper"),
    }
}

/// Per-method I-controller exponent.
fn method_err_exp(method: OdeMethod) -> f64 {
    match method {
        OdeMethod::Rosenbrock23 => ROS23_ERR_EXP,
        OdeMethod::Rodas4 => RODAS4.err_exp,
        OdeMethod::Rodas5P => RODAS5P.err_exp,
        other => unreachable!("{other:?} does not use the Rosenbrock stepper"),
    }
}

/// Reusable per-integration scratch: the Jacobian, the `W` factorization, the stage
/// increments and the stage/temporary vectors. Allocated once per `solve_ros_g` call and
/// reused across every step attempt, so the stepper allocates nothing in the loop.
struct RosWorkspace<T> {
    n: usize,
    /// `∂f/∂u` at the step start, row-major `n×n`.
    jac: Vec<T>,
    /// `W = I/(γh) − J`, overwritten by its LU factorization each attempt.
    w: Vec<T>,
    piv: Vec<usize>,
    /// `f(u, t)` at the step start.
    f0: Vec<T>,
    /// `∂f/∂t` at the step start.
    ft: Vec<T>,
    /// Stage increments `k_i` (Rodas) or stage derivatives `k_i` (`ode23s`).
    ks: Vec<Vec<T>>,
    /// Stage state / perturbed state.
    u_tmp: Vec<T>,
    /// Stage derivative / perturbed derivative.
    f_tmp: Vec<T>,
    /// `ode23s` only: the midpoint derivative `f₁`, which stage 3 needs after `f_tmp` has
    /// been overwritten by `f₂`.
    f_mid: Vec<T>,
    /// Stage right-hand side handed to the back-solve.
    rhs_tmp: Vec<T>,
    /// Proposed state at `t + h`.
    u_new: Vec<T>,
    /// Embedded error estimate for the proposed step.
    err: Vec<T>,
}

/// Why a step attempt could not be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepFailure {
    /// `W = I/(γh) − J` was singular (or non-finite) to the LU's scaled pivot tolerance, so
    /// no stage could be solved. Distinct from "the step was too inaccurate": there is no
    /// `u_new` to force-accept, so the driver must shrink `dt` or give up.
    SingularW,
}

impl<T: PkNum> RosWorkspace<T> {
    fn new(n: usize, stages: usize) -> Self {
        let zero = T::from_f64(0.0);
        Self {
            n,
            jac: vec![zero; n * n],
            w: vec![zero; n * n],
            piv: vec![0; n],
            f0: vec![zero; n],
            ft: vec![zero; n],
            ks: vec![vec![zero; n]; stages],
            u_tmp: vec![zero; n],
            f_tmp: vec![zero; n],
            f_mid: vec![zero; n],
            rhs_tmp: vec![zero; n],
            u_new: vec![zero; n],
            err: vec![zero; n],
        }
    }

    /// `f(u,t)`, `J = ∂f/∂u`, `f_t = ∂f/∂t`, then `W = I/(γh) − J` factored in place.
    ///
    /// `J` and `f_t` are forward differences with Hairer's `rodas.f` perturbation
    /// `δ = √(ε·max(1e-5, |x|))`. Differentiating the FD quotient is exactly what the dual
    /// instantiation does, so the sensitivities stay consistent with the step actually taken
    /// (the alternative — an exact `J` for the value part and an FD one for the jet — would
    /// not be).
    fn build_and_factor_w(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt: f64,
        gamma: f64,
    ) -> Result<(), StepFailure> {
        let n = self.n;
        rhs(u, params, t, &mut self.f0);

        for j in 0..n {
            let delt = (f64::EPSILON * u[j].val().abs().max(FD_FLOOR)).sqrt();
            self.u_tmp.copy_from_slice(u);
            self.u_tmp[j] = u[j] + T::from_f64(delt);
            rhs(&self.u_tmp, params, t, &mut self.f_tmp);
            let inv = T::from_f64(1.0 / delt);
            for i in 0..n {
                self.jac[i * n + j] = (self.f_tmp[i] - self.f0[i]) * inv;
            }
        }

        let delt = (f64::EPSILON * t.abs().max(FD_FLOOR)).sqrt();
        rhs(u, params, t + delt, &mut self.f_tmp);
        let inv = T::from_f64(1.0 / delt);
        for i in 0..n {
            self.ft[i] = (self.f_tmp[i] - self.f0[i]) * inv;
        }

        let inv_gamma_h = T::from_f64(1.0 / (gamma * dt));
        for idx in 0..n * n {
            self.w[idx] = -self.jac[idx];
        }
        for i in 0..n {
            self.w[i * n + i] = self.w[i * n + i] + inv_gamma_h;
        }
        if lu_factor_in_place_g(&mut self.w, &mut self.piv, n) {
            Ok(())
        } else {
            Err(StepFailure::SingularW)
        }
    }

    /// One Rodas-family step from `(u, t)` over `dt`, filling `u_new` and `err`. Assumes
    /// [`build_and_factor_w`](Self::build_and_factor_w) already ran for this `(u, t, dt)`.
    fn rodas_stages(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt: f64,
        tab: &RosTableau,
    ) {
        let n = self.n;
        let s = tab.stages;
        for i in 0..s {
            // u_i = u + Σ_{j<i} a_ij k_j
            self.u_tmp.copy_from_slice(u);
            for j in 0..i {
                let w = T::from_f64(tab.a[i][j]);
                for idx in 0..n {
                    self.u_tmp[idx] = self.u_tmp[idx] + self.ks[j][idx] * w;
                }
            }
            if i == 0 {
                // Stage 1 evaluates at (u, t), already in `f0`.
                self.f_tmp.copy_from_slice(&self.f0);
            } else {
                rhs(&self.u_tmp, params, t + tab.alpha[i] * dt, &mut self.f_tmp);
            }
            // rhs_i = f_i + h·γ_i·f_t + (1/h)·Σ_{j<i} c_ij k_j
            let hd = T::from_f64(dt * tab.d[i]);
            for idx in 0..n {
                self.rhs_tmp[idx] = self.f_tmp[idx] + self.ft[idx] * hd;
            }
            for j in 0..i {
                let w = T::from_f64(tab.c[i][j] / dt);
                for idx in 0..n {
                    self.rhs_tmp[idx] = self.rhs_tmp[idx] + self.ks[j][idx] * w;
                }
            }
            self.ks[i].copy_from_slice(&self.rhs_tmp);
            lu_solve_in_place_g(&self.w, &self.piv, n, &mut self.ks[i]);
        }
        // Stiffly accurate: b_i = a_{s,i} (i < s) and b_s = 1, so the solution is the last
        // stage's own state plus its increment — `u_tmp` still holds `u + Σ_{j<s} a_sj k_j`.
        // The embedded (order p−1) solution is `u_new − k_s`, hence `err = k_s`.
        for idx in 0..n {
            self.u_new[idx] = self.u_tmp[idx] + self.ks[s - 1][idx];
            self.err[idx] = self.ks[s - 1][idx];
        }
    }

    /// One `ode23s` step (Shampine & Reichelt 1997) from `(u, t)` over `dt`.
    ///
    /// The `k_i` here are **derivatives**, not increments: Shampine's `W̃ = I − γh·J` relates
    /// to the `W = I/(γh) − J` this workspace factors by `W̃ = γh·W`, so each of his solves
    /// is one back-solve against `W` scaled by `1/(γh)`.
    ///
    /// The stage-3 time-derivative term is `h·γ·f_t`. Some implementations (e.g.
    /// `OrdinaryDiffEq.jl`'s `Rosenbrock23`) scale it by `h` instead; `k₃` feeds only the
    /// error estimate, so the *solution* is identical either way, but the `h` variant was
    /// measured to give an `O(h²)`-per-step estimate of an `O(h³)` local error on a
    /// non-autonomous problem — ~100× too large, which would cost the controller a factor of
    /// ten in step size for nothing. `ros23_error_estimator_is_third_order` pins the choice.
    // Indexed loops walk several state vectors at once (`u`, `ks`, `rhs_tmp`, …); zipping
    // them would read worse than the formula they transcribe — same call as `sens::dual2`.
    #[allow(clippy::needless_range_loop)]
    fn ros23_stages(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt: f64,
    ) {
        let n = self.n;
        let gh = ROS23_GAMMA * dt;
        let inv_gh = T::from_f64(1.0 / gh);
        let hgamma = T::from_f64(gh);
        let half_h = T::from_f64(0.5 * dt);
        let h = T::from_f64(dt);

        // k₁ = W̃⁻¹(f₀ + hγ·f_t)
        for i in 0..n {
            self.rhs_tmp[i] = self.f0[i] + self.ft[i] * hgamma;
        }
        lu_solve_in_place_g(&self.w, &self.piv, n, &mut self.rhs_tmp);
        for i in 0..n {
            self.ks[0][i] = self.rhs_tmp[i] * inv_gh;
        }

        // f₁ = f(u + (h/2)·k₁, t + h/2)
        for i in 0..n {
            self.u_tmp[i] = u[i] + self.ks[0][i] * half_h;
        }
        rhs(&self.u_tmp, params, t + 0.5 * dt, &mut self.f_mid);

        // k₂ = W̃⁻¹(f₁ − k₁) + k₁
        for i in 0..n {
            self.rhs_tmp[i] = self.f_mid[i] - self.ks[0][i];
        }
        lu_solve_in_place_g(&self.w, &self.piv, n, &mut self.rhs_tmp);
        for i in 0..n {
            self.ks[1][i] = self.rhs_tmp[i] * inv_gh + self.ks[0][i];
        }

        // u_new = u + h·k₂  (the 2nd-order solution)
        for i in 0..n {
            self.u_new[i] = u[i] + self.ks[1][i] * h;
        }

        // k₃ = W̃⁻¹(f₂ − c₃₂(k₂ − f₁) − 2(k₁ − f₀) + hγ·f_t)
        rhs(&self.u_new, params, t + dt, &mut self.f_tmp);
        let c32 = T::from_f64(ROS23_C32);
        let two = T::from_f64(2.0);
        for i in 0..n {
            self.rhs_tmp[i] = self.f_tmp[i] + self.ft[i] * hgamma
                - (self.ks[1][i] - self.f_mid[i]) * c32
                - (self.ks[0][i] - self.f0[i]) * two;
        }
        lu_solve_in_place_g(&self.w, &self.piv, n, &mut self.rhs_tmp);
        let sixth_h = T::from_f64(dt / 6.0);
        for i in 0..n {
            let k3 = self.rhs_tmp[i] * inv_gh;
            self.err[i] = (self.ks[0][i] - self.ks[1][i] * two + k3) * sixth_h;
        }
    }

    /// RMS error norm over the scale band `abstol + reltol·max(|u_new|, |u|)`, on **values
    /// only** — identical in form to the RK45 control so a model that converges under one
    /// integrator sees the same accuracy contract under the other.
    // Indexed to read `u`, `u_new` and `err` for the same component together.
    #[allow(clippy::needless_range_loop)]
    fn err_norm(&self, u: &[T], opts: &OdeSolverOptions) -> f64 {
        let n = self.n;
        let mut acc = 0.0;
        for i in 0..n {
            let scale = scale_tol(opts.abstol, opts.reltol, self.u_new[i].val(), u[i].val());
            let e = self.err[i].val() / scale;
            acc += e * e;
        }
        (acc / n as f64).sqrt()
    }
}

/// The Rosenbrock family packaged as a [`Stepper`](super::solver::Stepper): workspace,
/// tableau, and the method's continuous extension. Everything else — step-size control, the
/// `saveat` contract, `min_dt` force-accept, soft sampling, event root-finding, statistics —
/// comes from the shared drivers in [`super::solver`], so a new consumer written against
/// `Stepper` gets every method at once and no method needs per-consumer wiring.
pub(crate) struct RosStepper<T> {
    ws: RosWorkspace<T>,
    method: OdeMethod,
    /// Whether the last attempt produced a usable `u_new` (false only on a singular `W`).
    usable: bool,
}

impl<T: PkNum> RosStepper<T> {
    pub(crate) fn new(n: usize, method: OdeMethod) -> Self {
        // `ode23s` stores only k₁ and k₂ (k₃ feeds the error estimate and dies with the step).
        let stages = match method {
            OdeMethod::Rosenbrock23 => 2,
            OdeMethod::Rodas4 => RODAS4.stages,
            OdeMethod::Rodas5P => RODAS5P.stages,
            other => unreachable!("{other:?} does not use the Rosenbrock stepper"),
        };
        Self {
            ws: RosWorkspace::new(n, stages),
            method,
            usable: false,
        }
    }

    /// The Rodas tableau backing this method, or `None` for `ode23s` (which is not in tableau
    /// form — see [`RosWorkspace::ros23_stages`]).
    fn tableau(&self) -> Option<&'static RosTableau> {
        match self.method {
            OdeMethod::Rodas4 => Some(&RODAS4),
            OdeMethod::Rodas5P => Some(&RODAS5P),
            OdeMethod::Rosenbrock23 | OdeMethod::Rk45 | OdeMethod::Vern7 => None,
        }
    }
}

impl<T: PkNum> super::solver::Stepper<T> for RosStepper<T> {
    fn attempt(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        let gamma = method_gamma(self.method);
        if self
            .ws
            .build_and_factor_w(rhs, u, params, t, dt, gamma)
            .is_err()
        {
            // Singular `W`: no stage could be solved, so there is no `u_new` to score or to
            // force-accept. The driver shrinks `dt`, or gives up at `min_dt`.
            self.usable = false;
            return f64::INFINITY;
        }
        self.usable = true;
        match self.method {
            OdeMethod::Rosenbrock23 => self.ws.ros23_stages(rhs, u, params, t, dt),
            OdeMethod::Rodas4 => self.ws.rodas_stages(rhs, u, params, t, dt, &RODAS4),
            OdeMethod::Rodas5P => self.ws.rodas_stages(rhs, u, params, t, dt, &RODAS5P),
            other => unreachable!("{other:?} does not use the Rosenbrock stepper"),
        }
        self.ws.err_norm(u, opts)
    }

    fn u_new(&self) -> &[T] {
        &self.ws.u_new
    }

    fn interpolate_component(&self, theta: f64, u_old: &[T], dt: f64, i: usize) -> T {
        match self.tableau() {
            Some(tab) => tab.interpolate(theta, u_old[i], self.ws.u_new[i], &self.ws.ks, i),
            // `ode23s`'s continuous extension (Shampine & Reichelt): a quadratic in `Θ` over
            // the two stage *derivatives*, exact at both ends (`Θ = 0 → y₀`;
            // `Θ = 1 → y₀ + h·k₂ = u_new`).
            None => {
                let d = ROS23_GAMMA;
                let c1 = theta * (1.0 - theta) / (1.0 - 2.0 * d);
                let c2 = theta * (theta - 2.0 * d) / (1.0 - 2.0 * d);
                u_old[i]
                    + (self.ws.ks[0][i] * T::from_f64(c1) + self.ws.ks[1][i] * T::from_f64(c2))
                        * T::from_f64(dt)
            }
        }
    }

    fn on_accept(&mut self) {}

    fn err_exp(&self) -> f64 {
        method_err_exp(self.method)
    }

    fn attempt_usable(&self) -> bool {
        self.usable
    }
}

#[cfg(test)]
#[path = "rosenbrock_tests.rs"]
mod tests;
