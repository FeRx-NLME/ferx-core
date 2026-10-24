//! The profile solve (#1065): Newton on the total gradient, the envelope gradient and
//! the layout, each against the unprofiled objective of the same model.
//!
//! Fixtures (both synthetic, both committed):
//! - **F1** `tests/data/profile_mbma/f1_profile.ferx` on `mbma_placebo`: linear in
//!   (η, κ), so the solve is one Newton step.
//! - **F2** `nonmem_anchor/profile_mbma_fit.ferx` on `profile_mbma.csv`, cut to its first
//!   two or three studies here: the drug effect is `EMAX·exp(ETA_EM)`, nonlinear
//!   in η with a sensitivity that varies by visit, so the frozen-EBE shortcut the solve
//!   avoids lands measurably above the minimum.
//!
//! The oracle for "the minimum over the levels" is a dense Newton over every level at
//! once on the population objective, with every EBE re-solved cold at each point and
//! finite differences for both derivatives — the M3 probe of #1065, in Rust. It shares
//! nothing with the solve but the objective.

use super::*;
use crate::types::FitOptions;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const F1: &str = "tests/data/profile_mbma/f1_profile.ferx";
const F1_DATA: &str = "tests/data/mbma_placebo/mbma_placebo.csv";
const F2: &str = "nonmem_anchor/profile_mbma_fit.ferx";
const F2_DATA: &str = "nonmem_anchor/profile_mbma.csv";

/// A fresh temp path per call, so concurrent tests never share a file (#2090).
fn temp(name: &str) -> PathBuf {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "ferx_profile_{}_{}_{name}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// `model` on `data`, keeping the studies `<= max_study` when given; `edit` rewrites the
/// model text first.
fn prep(
    model: &str,
    data: &str,
    max_study: Option<usize>,
    edit: impl Fn(String) -> String,
) -> crate::api::PreparedRun {
    let text = edit(std::fs::read_to_string(root().join(model)).unwrap());
    let model_path = temp("model.ferx");
    std::fs::write(&model_path, text).unwrap();
    let csv = std::fs::read_to_string(root().join(data)).unwrap();
    let data_path = temp("data.csv");
    let kept: String =
        match max_study {
            None => csv,
            Some(n) => {
                let mut lines = csv.lines();
                let header = lines.next().unwrap();
                let col = header
                    .split(',')
                    .position(|c| c.trim_matches('"') == "STUDY")
                    .unwrap();
                std::iter::once(header)
                    .chain(lines.filter(|l| {
                        l.split(',').nth(col).unwrap().parse::<f64>().unwrap() <= n as f64
                    }))
                    .map(|l| format!("{l}\n"))
                    .collect()
            }
        };
    std::fs::write(&data_path, kept).unwrap();
    crate::api::prepare_run(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn options(p: &crate::api::PreparedRun) -> FitOptions {
    FitOptions {
        verbose: false,
        ..p.parsed.fit_options.clone()
    }
}

fn ctx<'a>(
    p: &'a crate::api::PreparedRun,
    layout: &'a ProfileLayout,
    options: &'a FitOptions,
) -> ProfileCtx<'a> {
    ProfileCtx {
        model: &p.parsed.model,
        population: &p.population,
        layout,
        options,
        inner_tol: PROFILE_INNER_TOL,
    }
}

/// The population objective with the layout's coordinates at `beta`, every EBE re-solved
/// cold.
fn total_ofv(c: &ProfileCtx<'_>, params: &ModelParameters, beta: &[f64]) -> f64 {
    let mut p = params.clone();
    for (&k, &b) in c.layout.coords.iter().zip(beta) {
        p.theta[k] = b;
    }
    let zero = DVector::zeros(c.model.n_eta);
    2.0 * (0..c.population.subjects.len())
        .into_par_iter()
        .map(|i| c.resolve(i, &p, &zero).nll)
        .sum::<f64>()
}

/// The oracle: Newton over every coordinate at once on [`total_ofv`], FD gradient at
/// every step and an FD Hessian taken once at the start (the objective is close to
/// quadratic in the levels, so the fixed matrix still contracts fast, and the fixed
/// point is the gradient's zero whatever the matrix). Returns the minimum and its
/// argument.
fn oracle_min(c: &ProfileCtx<'_>, params: &ModelParameters, start: &[f64]) -> (f64, Vec<f64>) {
    let n = start.len();
    let mut b = start.to_vec();
    let f = |x: &[f64]| total_ofv(c, params, x);
    let at = |b: &[f64], d: &[(usize, f64)]| {
        let mut x = b.to_vec();
        for &(j, v) in d {
            x[j] += v;
        }
        f(&x)
    };
    let hh = 1e-2;
    let f0 = f(&b);
    let mut h = DMatrix::zeros(n, n);
    for j in 0..n {
        h[(j, j)] = (at(&b, &[(j, hh)]) - 2.0 * f0 + at(&b, &[(j, -hh)])) / (hh * hh);
        for k in 0..j {
            let v = (at(&b, &[(j, hh), (k, hh)])
                - at(&b, &[(j, hh), (k, -hh)])
                - at(&b, &[(j, -hh), (k, hh)])
                + at(&b, &[(j, -hh), (k, -hh)]))
                / (4.0 * hh * hh);
            h[(j, k)] = v;
            h[(k, j)] = v;
        }
    }
    let chol = h.cholesky().expect("oracle Hessian PD");
    for _ in 0..60 {
        let hg = 1e-4;
        let g: Vec<f64> = (0..n)
            .map(|j| (at(&b, &[(j, hg)]) - at(&b, &[(j, -hg)])) / (2.0 * hg))
            .collect();
        let step = chol.solve(&DVector::from_vec(g));
        for j in 0..n {
            b[j] -= step[j];
        }
        if step.amax() < 1e-9 {
            break;
        }
    }
    (f(&b), b)
}

/// The frozen-EBE fixed point the solve exists to avoid (M3 (a) on #1065): solve the
/// levels at the current η̂, H and κ̂ held, re-solve the EBEs, repeat. Each inner step is
/// Newton on the held objective with the Newton matrix; returns its objective.
fn frozen_fixed_point(c: &ProfileCtx<'_>, params: &ModelParameters, start: &[f64]) -> f64 {
    let mut b = start.to_vec();
    let n = b.len();
    let zero = DVector::zeros(c.model.n_eta);
    for _ in 0..200 {
        let mut p = params.clone();
        for (&k, &v) in c.layout.coords.iter().zip(&b) {
            p.theta[k] = v;
        }
        let states: Vec<SubjectState> = (0..c.population.subjects.len())
            .map(|i| c.resolve(i, &p, &zero))
            .collect();
        // The held objective's gradient and its exact (quadratic) Newton step.
        let held = |x: &[f64]| -> f64 {
            let mut q = params.clone();
            for (&k, &v) in c.layout.coords.iter().zip(x) {
                q.theta[k] = v;
            }
            states
                .iter()
                .enumerate()
                .map(|(i, s)| c.held(i, &q, s, c.options.interaction))
                .sum()
        };
        let h = 1e-3;
        let at = |j: usize, d: f64| {
            let mut x = b.clone();
            x[j] += d;
            held(&x)
        };
        let g: Vec<f64> = (0..n).map(|j| (at(j, h) - at(j, -h)) / (2.0 * h)).collect();
        let f0 = held(&b);
        let mut m = DMatrix::zeros(n, n);
        for j in 0..n {
            m[(j, j)] = (at(j, h) - 2.0 * f0 + at(j, -h)) / (h * h);
            for k in 0..j {
                let mut x = b.clone();
                x[j] += h;
                x[k] += h;
                let v = (held(&x) - at(j, h) - at(k, h) + f0) / (h * h);
                m[(j, k)] = v;
                m[(k, j)] = v;
            }
        }
        let step = m.cholesky().unwrap().solve(&DVector::from_vec(g));
        for j in 0..n {
            b[j] -= step[j];
        }
        if step.amax() < 1e-10 {
            break;
        }
    }
    total_ofv(c, params, &b)
}

fn layout_of(p: &crate::api::PreparedRun) -> ProfileLayout {
    ProfileLayout::build(&p.parsed.model, &p.population, &p.init_params).unwrap()
}

// ── N1: one Newton step on a model linear in (η, κ) ────────────────────────────────

/// On F1 the Newton step is the exact GLS solution, so the solve stops after **one**
/// step, at the tolerance. Dies when the Newton matrix is taken from the fit's own FOCEI
/// term (`XᵀR⁻¹X`, measured ~1/30 contraction per step) or any other approximation of
/// `XᵀC̃⁻¹X`.
#[test]
fn a_model_linear_in_its_random_effects_profiles_in_one_newton_step() {
    let p = prep(F1, F1_DATA, None, |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    assert_eq!(
        layout.groups.len(),
        6,
        "one group per study under sum_to_zero_within"
    );
    let s = c
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .expect("the solve converges");
    assert_eq!(s.iterations, 1, "criterion {:.3e}", s.criterion);
    assert!(
        s.criterion.is_finite() && s.criterion <= PROFILE_TOL,
        "{}",
        s.criterion
    );
}

/// The dense oracle's minimum on F2 cut to two studies, started from the template's
/// levels at the initial parameters: the point D3, the bad start and N2's
/// `sum_to_zero_within` case each compare against, from the identical model, data and
/// start. Its FD Hessian is the bulk of each of those tests, so it is computed once
/// (#2167) — the solves under test still run per test.
fn f2_oracle() -> &'static (f64, Vec<f64>) {
    static O: OnceLock<(f64, Vec<f64>)> = OnceLock::new();
    O.get_or_init(|| {
        let p = prep(F2, F2_DATA, Some(2), |t| t);
        let layout = layout_of(&p);
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let warm = ProfileWarm::from_template(&layout, &p.init_params);
        oracle_min(&c, &p.init_params, &warm.beta)
    })
}

// ── D3: the profile objective is the minimum over the levels ───────────────────────

/// On F2 (nonlinear in η), the solve's objective and levels equal the dense oracle's,
/// and the frozen-EBE fixed point lands above both by more than the bound, so the
/// fixture can tell the two apart (the straddle). Dies when the total gradient is
/// replaced by the held-EBE partial.
#[test]
fn the_profile_objective_is_the_minimum_over_the_levels() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let s = c.solve(&p.init_params, &warm).expect("the solve converges");
    let (o_star, b_star) = f2_oracle();
    let o_star = *o_star;
    let frozen = frozen_fixed_point(&c, &p.init_params, &warm.beta);
    assert!(s.ofv.is_finite() && o_star.is_finite() && frozen.is_finite());
    let worst_beta = layout
        .coords
        .iter()
        .zip(b_star)
        .zip(&s.se_scale)
        .map(|((&k, &b), &se)| ((s.params.theta[k] - b) / se).abs())
        .fold(0.0_f64, |a, v| {
            assert!(v.is_finite());
            a.max(v)
        });
    eprintln!(
        "D3: profile {:.10} oracle {:.10} (Δ {:.2e}), worst level gap {:.2e} SE, \
         frozen-EBE Δ {:.3e}, {} steps",
        s.ofv,
        o_star,
        s.ofv - o_star,
        worst_beta,
        frozen - o_star,
        s.iterations
    );
    assert!(
        s.iterations > 1,
        "a nonlinear fixture takes more than one step"
    );
    assert!(
        (s.ofv - o_star).abs() < D3_OFV_BOUND,
        "{} vs {}",
        s.ofv,
        o_star
    );
    assert!(worst_beta < D3_BETA_BOUND, "{worst_beta}");
    assert!(
        frozen - o_star > 10.0 * D3_OFV_BOUND,
        "the straddle: {}",
        frozen - o_star
    );
}

/// Worst realised |ΔOFV| 1.8e-8 (N2 `ref`, macOS arm64; 6.2e-9 Linux aarch64; D3 6.1e-9
/// Linux), the oracle's own FD and inner-solve noise
/// included; 14× headroom. The frozen-EBE point is 1.44e-2 above the minimum on this
/// fixture, five orders of magnitude past the bound.
const D3_OFV_BOUND: f64 = 1e-7;
/// Worst realised level gap 1.08e-5 conditional SEs; ~9× headroom.
const D3_BETA_BOUND: f64 = 1e-4;

// ── D0: the profiled objective is the unprofiled one at the joint point ────────────

/// The profile solve's objective equals the ordinary population objective — a cold
/// inner loop and `pop_nll_opts`, the unprofiled fit's own scoring — evaluated at the
/// joint point `(θ, β̂)` it returns. Independent of any optimizer converging.
#[test]
fn the_profiled_objective_is_the_unprofiled_objective_at_its_own_point() {
    for (model, data, n) in [(F1, F1_DATA, None), (F2, F2_DATA, Some(3))] {
        let p = prep(model, data, n, |t| t);
        let layout = layout_of(&p);
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let s = c
            .solve(
                &p.init_params,
                &ProfileWarm::from_template(&layout, &p.init_params),
            )
            .unwrap();
        let mu = compute_mu_k(c.model, &s.params.theta, o.mu_referencing);
        let (e, h, _, k) = run_inner_loop_warm_seeded(
            c.model,
            c.population,
            &s.params,
            o.inner_maxiter,
            PROFILE_INNER_TOL,
            None,
            Some(&mu),
            0,
            o.inner_restarts,
            InnerHessianSeed::for_options(&o),
            &o,
        );
        let plain = 2.0
            * crate::estimation::outer_optimizer::pop_nll_opts(
                c.model,
                c.population,
                &s.params,
                &e,
                &h,
                &k,
                &o,
            );
        eprintln!("D0 {model}: profiled {:.12} plain {:.12}", s.ofv, plain);
        assert!(s.ofv.is_finite() && plain.is_finite());
        assert!((s.ofv - plain).abs() < D0_BOUND, "{} vs {}", s.ofv, plain);
    }
}

/// Worst realised 1.1e-8 (F1; F2 1.0e-9), the two inner solves' difference at
/// `PROFILE_INNER_TOL`; 9× headroom.
const D0_BOUND: f64 = 1e-7;

// ── N3: a level ferx froze is solved around, not moved ─────────────────────────────

/// One level of F2 FIXed in the template (a level ferx froze, #1891): it keeps its value
/// exactly, and the others are the oracle's minimum with it held. Dies when the layout
/// keeps the frozen column.
#[test]
fn a_frozen_level_is_an_offset() {
    let mut p = prep(F2, F2_DATA, Some(2), |t| t);
    let all = layout_of(&p);
    let k = all.coords[1];
    p.init_params.theta[k] = 0.3;
    p.init_params.theta_fixed[k] = true;
    let layout = layout_of(&p);
    assert!(!layout.coords.contains(&k) && layout.coords.len() == all.coords.len() - 1);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let s = c.solve(&p.init_params, &warm).unwrap();
    assert_eq!(s.params.theta[k], 0.3);
    let (o_star, _) = oracle_min(&c, &p.init_params, &warm.beta);
    assert!(
        (s.ofv - o_star).abs() < D3_OFV_BOUND,
        "{} vs {}",
        s.ofv,
        o_star
    );
}

// ── G1: the envelope gradient is the gradient of the profile objective ─────────────

/// At F2's initial point (away from the optimum), the gradient the optimizer is handed —
/// the production gradient with the levels held at `β̂` — equals central FD of the
/// profile objective, every point re-profiled. Dies when the levels are held at a stale
/// value instead of `β̂`.
fn envelope_matches_fd(edit: impl Fn(String) -> String, analytic: bool) {
    let p = prep(F2, F2_DATA, Some(3), edit);
    assert_eq!(
        crate::sens::provider::analytic_outer_gradient_for_interaction(
            &p.parsed.model,
            p.parsed.fit_options.interaction
        ),
        analytic,
        "the arm runs on the engine it names"
    );
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let hold = hold_template(&layout, &p.init_params);
    let PackedStart {
        packed: x,
        bounds,
        fixed,
        ..
    } = pack_with_bounds(&hold);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let s = c.solve(&unpack_params(&x, &hold), &warm).unwrap();
    let declines = OuterFdDeclineLog::new(p.population.subjects.len());
    let g = envelope_gradient(&c, &hold, &x, &s, &bounds, &mut 0, &declines);
    let w = ProfileWarm::from_solve(&layout, &s);
    let beta: Vec<f64> = layout.coords.iter().map(|&k| s.params.theta[k]).collect();
    let (mut worst_handed, mut worst_identity) = (0.0_f64, 0.0_f64);
    for i in (0..x.len()).filter(|&i| !fixed[i]) {
        let h = 1e-4 * (1.0 + x[i].abs());
        let moved = |d: f64| {
            let mut y = x.clone();
            y[i] += d;
            y
        };
        // dO_p/dθ: the levels re-profiled at every point.
        let fd = (c.profile_ofv(&hold, &moved(h), &w) - c.profile_ofv(&hold, &moved(-h), &w))
            / (2.0 * h);
        // ∂O/∂θ at β held at β̂: what the envelope theorem says that equals.
        let held = (total_ofv(&c, &unpack_params(&moved(h), &hold), &beta)
            - total_ofv(&c, &unpack_params(&moved(-h), &hold), &beta))
            / (2.0 * h);
        assert!(
            fd.is_finite() && held.is_finite() && g[i].is_finite(),
            "{i}: {fd} / {held} / {}",
            g[i]
        );
        worst_handed = worst_handed.max((g[i] - fd).abs() / (1.0 + fd.abs()));
        worst_identity = worst_identity.max((held - fd).abs() / (1.0 + fd.abs()));
    }
    eprintln!(
        "G1 analytic={analytic}: handed-vs-profile {worst_handed:.3e}, \
         envelope identity {worst_identity:.3e}"
    );
    assert!(worst_identity < G1_IDENTITY_BOUND, "{worst_identity}");
    assert!(worst_handed < G1_HANDED_BOUND, "{worst_handed}");
}

/// The envelope identity `dO_p/dθ = ∂O/∂θ|β̂`, both sides FD, no engine involved. Worst
/// realised 3.1e-5 (F2, three studies, Linux aarch64 and macOS arm64), the β residual at `PROFILE_TOL` and
/// FD noise; ~3× headroom.
const G1_IDENTITY_BOUND: f64 = 1e-4;
/// The gradient the optimizer is handed against FD of the profile objective. This also
/// carries the production gradient's own distance from its objective, which is the larger
/// part: on the analytic arm `∂O/∂TVE0` reads 25.58695 against an FD of 25.59225 at held
/// `β̂` (2.0e-4 relative), a property of the unprofiled engine, not of the profile. Worst
/// realised 1.78e-4 (analytic) and 5.5e-5 (FD); ~3× headroom.
const G1_HANDED_BOUND: f64 = 5e-4;

#[test]
fn the_envelope_gradient_is_the_profile_gradient_on_the_fd_engine() {
    envelope_matches_fd(
        |t| {
            t.replace(
                "  method     = focei\n",
                "  method     = focei\n  gradient   = fd\n",
            )
        },
        false,
    );
}

/// No kappa and `gradient = fd`: the non-IOV FD route, whose default is the fixed-EBE
/// gradient (`dη̂/dθ` dropped). The envelope reconverges it; dies when it does not.
#[test]
fn the_envelope_gradient_is_the_profile_gradient_on_the_non_iov_fd_engine() {
    envelope_matches_fd(
        |t| {
            t.replace("  kappa KAPPA_ARM ~ 1.0\n", "")
                .replace(" + KAPPA_ARM", "")
                .replace("  iov_column = OCC\n", "")
                .replace(
                    "  method     = focei\n",
                    "  method     = focei\n  gradient   = fd\n",
                )
        },
        false,
    );
}

/// The same with no kappa, so the production gradient is the analytic one.
#[test]
fn the_envelope_gradient_is_the_profile_gradient_on_the_analytic_engine() {
    envelope_matches_fd(
        |t| {
            t.replace("  kappa KAPPA_ARM ~ 1.0\n", "")
                .replace(" + KAPPA_ARM", "")
                .replace("  iov_column = OCC\n", "")
        },
        true,
    );
}

// ── Backtracking: a start far from the minimum ─────────────────────────────────────

/// From levels 20 units off (the data's spread is ~1.5) the solve still lands on the
/// oracle's minimum: the Newton matrix taken at the start's η̂ is a long way from the
/// one at the optimum.
///
/// It does **not** kill the removal of the backtracking, and no fixture found here does:
/// on F2 a full Newton step never raised the objective — starts at ±5 … ±3000, Ω_EM up to
/// 9 and EMAX 40 all converged in 3–4 steps to the same minimum, with or without the
/// line search (measured on #1065). The backtracking stays as the safeguard the plan
/// specifies for a model whose GLS matrix is a poorer stand-in for the Hessian.
#[test]
fn a_bad_start_still_reaches_the_minimum() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let bad = ProfileWarm {
        beta: vec![BAD_START; layout.coords.len()],
        etas: None,
    };
    let s = c.solve(&p.init_params, &bad).expect("the solve converges");
    let o_star = f2_oracle().0;
    eprintln!(
        "backtracking: from {BAD_START} {:.10} vs oracle {:.10} in {} steps",
        s.ofv, o_star, s.iterations
    );
    assert!(
        (s.ofv - o_star).abs() < D3_OFV_BOUND,
        "{} vs {}",
        s.ofv,
        o_star
    );
}

const BAD_START: f64 = 20.0;

// ── N2: every contrast ─────────────────────────────────────────────────────────────

/// Each contrast's profile objective is the oracle's minimum: `sum_to_zero_within` (one
/// group per study), the global `sum_to_zero` (its dependent level couples every study,
/// so the whole block is one group), `ref`, and `none` — the last on a `[TIME]` block
/// with no intercept, since on `[STUDY, TIME]` the binder refuses `none` beside a
/// subject-level η (it would carry the same study mean). Each `[TIME]` level is read by
/// both studies, one group again. Dies when coupled levels are solved as separate
/// groups.
#[test]
fn every_contrast_profiles_to_the_minimum() {
    let within = "PLACEBO[STUDY, TIME, contrast = sum_to_zero_within]";
    for (block, n_groups) in [
        ("PLACEBO[STUDY, TIME, contrast = sum_to_zero_within]", 2),
        ("PLACEBO[STUDY, TIME, contrast = sum_to_zero]", 1),
        ("PLACEBO[STUDY, TIME, contrast = ref]", 2),
        ("PLACEBO[TIME, contrast = none]", 1),
    ] {
        let contrast = block;
        let p = prep(F2, F2_DATA, Some(2), |t| {
            let t = t.replace(within, block);
            if block.contains("none") {
                t.replace("BASE  = TVE0 + PLACEBO", "BASE  = PLACEBO")
            } else {
                t
            }
        });
        let layout = layout_of(&p);
        assert_eq!(layout.groups.len(), n_groups, "{contrast}");
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let warm = ProfileWarm::from_template(&layout, &p.init_params);
        let s = c
            .solve(&p.init_params, &warm)
            .unwrap_or_else(|e| panic!("{contrast}: {e:?}"));
        let o_star = if block == within {
            f2_oracle().0
        } else {
            oracle_min(&c, &p.init_params, &warm.beta).0
        };
        eprintln!(
            "N2 {contrast}: {} coords, profile {:.10} oracle {:.10}",
            layout.coords.len(),
            s.ofv,
            o_star
        );
        assert!(
            (s.ofv - o_star).abs() < D3_OFV_BOUND,
            "{contrast}: {} vs {}",
            s.ofv,
            o_star
        );
    }
}

// ── Replay: the covariance step's solve failure ────────────────────────────────────

/// `replay_covariance` on a start the profile solve cannot use (a `NaN` level; measured,
/// the solve fails as `ProfileFailure::NonFinite`) returns no matrix and the one warning
/// `profile_solve_failed_msg` builds — and that warning, as emitted, classifies
/// `covariance_failed` (#2158). The `classify_warning` table in `types_tests.rs` builds its
/// profile rows from the same function; this test is what ties the function to the call
/// site, so an inline message put back there (with the prefix reworded, say) fails here.
///
/// The failure arm is `NonFinite` since #2093 (it was `Singular` when #2176 added this
/// test): the `NaN` level makes subject 0's FOCE objective `NaN`, and the solve rejects
/// its non-finite start. The FOCE objective propagates `NaN` by design — the fit's
/// non-finite arm (#1303, `nonfinite_objective_tests`) keys on it — and the IOV path
/// returned `1e20` only by accident: its κ columns of `H_full` were central differences
/// of the `NaN` predictions, so `H_full` went `NaN` and the linearized objective fell into
/// its `1e20` sentinel (measured: subject 0 scores `1e20` on `main`, `NaN` here). #2093
/// builds them exactly (∂f/∂κ = 1 on F1 at any θ), so the IOV path now behaves like the
/// non-IOV one. Which arm fires is incidental here; the classification is the claim.
#[test]
fn a_failed_replay_solve_is_a_covariance_failure() {
    let p = prep(F1, F1_DATA, None, |t| t);
    let layout = layout_of(&p);
    let mut params = p.init_params.clone();
    params.theta[layout.coords[0]] = f64::NAN;
    let o = FitOptions {
        covariance_method: CovarianceMethod::Hessian,
        ..options(&p)
    };
    let out = replay_covariance(
        &p.parsed.model,
        &p.population,
        &params,
        &pack_with_bounds(&params).packed,
        &[],
        &o,
    )
    .expect("F1 profiles a level block");
    assert!(out.matrix.is_none());
    assert_eq!(
        out.warnings,
        vec![profile_solve_failed_msg(&ProfileFailure::NonFinite)]
    );
    let w = crate::types::classify_warning(&out.warnings[0]);
    assert_eq!(
        (w.severity, w.category.as_str()),
        (crate::types::WarningSeverity::Critical, "covariance_failed"),
        "{}",
        out.warnings[0]
    );
}

// ── #2285: the profiled outer stop ─────────────────────────────────────────────────

/// A profiled gradient fit stops on [`PROFILED_GRADIENT_FTOL_REL`], not the unprofiled
/// `1e-6` that let L-BFGS report `converged: true` 3.9e-4 above A2's optimum; BOBYQA
/// keeps the unprofiled rule and a user's `outer_ftol` wins over both. The fit-level
/// regression is `a2_a_nearby_start_reaches_nonmems_optimum` (slow).
#[test]
fn a_profiled_gradient_fit_stops_on_its_own_ftol() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let model = &p.parsed.model;
    let o = options(&p);
    assert_eq!(o.outer_ftol, None, "fixture sets outer_ftol");
    let unprofiled = resolve_outer_ftol(model.has_non_gaussian(), model.is_ode_based(), None);
    assert_eq!(unprofiled, 1e-6);
    for algo in [nlopt::Algorithm::Lbfgs, nlopt::Algorithm::Slsqp] {
        assert_eq!(profiled_ftol(algo, model, &o), PROFILED_GRADIENT_FTOL_REL);
    }
    assert!(PROFILED_GRADIENT_FTOL_REL < unprofiled);
    assert_eq!(
        profiled_ftol(nlopt::Algorithm::Bobyqa, model, &o),
        unprofiled
    );
    let user = FitOptions {
        outer_ftol: Some(3e-7),
        ..o.clone()
    };
    for algo in [nlopt::Algorithm::Lbfgs, nlopt::Algorithm::Bobyqa] {
        assert_eq!(profiled_ftol(algo, model, &user), 3e-7);
    }
    assert!(
        matches!(profiled_algorithm(&o), nlopt::Algorithm::Lbfgs),
        "auto"
    );
}

// ── #2135: a global sum_to_zero is differenced over each subject's own levels ──────

const GLOBAL: &str = "PLACEBO[STUDY, TIME, contrast = sum_to_zero]";
const WITHIN: &str = "PLACEBO[STUDY, TIME, contrast = sum_to_zero_within]";

/// F2 cut to `studies`, its block under the global `sum_to_zero`.
fn global_prep(studies: usize) -> crate::api::PreparedRun {
    prep(F2, F2_DATA, Some(studies), |t| {
        assert!(t.contains(WITHIN), "fixture moved");
        t.replace(WITHIN, GLOBAL)
    })
}

/// Under the global contrast the whole block is one group, its dependent level one
/// border, and each subject is differenced over **its own** levels — the cost #2135 is
/// about: in θ, a subject reading the dependent read every θ of the block. Each read's
/// moves shift exactly its own level by the step and leave every other level the subject
/// reads where it was (`eval_gather`, the prediction's own evaluator, is the judge). Dies
/// when the slack's sign flips, when a slack the subject reads is chosen, when a read
/// sharing the dependent's group loses its slack, or when the layout falls back to
/// expanding the dependent to its range.
#[test]
fn a_global_sum_to_zero_differences_each_subject_over_its_own_levels() {
    let p = global_prep(3);
    let layout = layout_of(&p);
    let model = &p.parsed.model;
    let decl = model
        .indiv_param_partials
        .theta_blocks
        .decls
        .iter()
        .find(|d| d.name == "PLACEBO")
        .unwrap();
    let al = crate::api::active_levels(model, &p.population);
    let per_subject = &al
        .iter()
        .find(|a| a.block == "PLACEBO")
        .unwrap()
        .per_subject;
    assert_eq!(
        layout.groups.len(),
        1,
        "the global contrast ties every level"
    );
    assert_eq!(layout.borders.len(), 1, "one dependent level");
    let n_levels = decl.spec.levels.len();
    assert_eq!(layout.coords.len(), n_levels - 1);
    let theta = &p.init_params.theta;
    let level_values = |theta: &[f64], levels: &[usize]| -> Vec<f64> {
        levels
            .iter()
            .map(|&l| crate::parser::model_parser::eval_gather(&decl.spec, theta, (l + 1) as f64))
            .collect()
    };
    let (mut dependent_readers, mut max_reads) = (0, 0);
    for (i, levels) in per_subject.iter().enumerate() {
        let reads = &layout.reads[i];
        assert_eq!(
            reads.len(),
            levels.len(),
            "subject {i}: its own levels only"
        );
        max_reads = max_reads.max(reads.len());
        if levels.contains(&(n_levels - 1)) {
            dependent_readers += 1;
            assert!(
                reads.iter().any(|r| r.level >= layout.coords.len()),
                "subject {i}"
            );
        }
        let before = level_values(theta, levels);
        for (j, _) in reads.iter().enumerate() {
            let d = 0.5;
            let after = level_values(&shifted(&p.init_params, reads, &[(j, d)]).theta, levels);
            let moved: Vec<f64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
            assert_eq!(
                moved.iter().filter(|m| m.abs() > 1e-12).count(),
                1,
                "subject {i} read {j}: {moved:?}"
            );
            assert!(moved.iter().any(|m| (m - d).abs() < 1e-12), "{moved:?}");
        }
    }
    assert!(
        dependent_readers > 0,
        "the fixture reads the dependent level"
    );
    assert!(
        max_reads < layout.coords.len() / 2,
        "{max_reads} of {} levels",
        layout.coords.len()
    );
}

/// Under `sum_to_zero_within` every F2 subject reads all of its study's levels, so no θ
/// of the dependent's group is left for a slack: no border, and the θ form, unchanged.
#[test]
fn a_subject_reading_its_whole_group_keeps_the_theta_form() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    assert!(layout.borders.is_empty());
    for (i, reads) in layout.reads.iter().enumerate() {
        assert!(
            reads
                .iter()
                .all(|r| r.level < layout.coords.len()
                    && r.moves == vec![(layout.coords[r.level], 1.0)]),
            "subject {i}"
        );
    }
}

/// A group of four θ and `borders` dependents over a block-diagonal level matrix.
fn bordered_fixture(border_of: Vec<Option<usize>>, nb: usize) -> ProfileGroup {
    ProfileGroup {
        coords: (0..border_of.len()).collect(),
        subjects: Vec::new(),
        borders: (0..nb).collect(),
        border_of,
    }
}

/// An SPD matrix over `n` levels, block diagonal over `blocks`.
fn block_spd(n: usize, blocks: &[Vec<usize>], seed: u64) -> DMatrix<f64> {
    let mut x = seed;
    let mut next = || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mut m = DMatrix::zeros(n, n);
    for blk in blocks {
        let k = blk.len();
        let a = DMatrix::from_fn(k, k, |_, _| next());
        let s = &a * a.transpose() + DMatrix::identity(k, k) * 0.5;
        for (i, &li) in blk.iter().enumerate() {
            for (j, &lj) in blk.iter().enumerate() {
                m[(li, lj)] = s[(i, j)];
            }
        }
    }
    m
}

/// The bordered step is the θ-form Newton step `−(TᵀMT)⁻¹Tᵀg`, with one border and with
/// two, whose blocks each straddle a border's range. Dies on a dropped multiplier term,
/// a wrong sign in `T`, or a border row that misses its own level.
#[test]
fn the_bordered_step_is_the_theta_newton_step() {
    let cases: Vec<(Vec<Option<usize>>, usize, Vec<Vec<usize>>)> = vec![
        // θ 1..4 summed by border 0 (level 4); θ 0 outside it.
        (
            vec![None, Some(0), Some(0), Some(0)],
            1,
            vec![vec![0, 1], vec![2, 4], vec![3]],
        ),
        // Two dependents: θ 0, 1 → level 4; θ 2, 3 → level 5.
        (
            vec![Some(0), Some(0), Some(1), Some(1)],
            2,
            vec![vec![0, 2], vec![1, 4], vec![3, 5]],
        ),
    ];
    for (seed, (border_of, nb, blocks)) in cases.into_iter().enumerate() {
        let g = bordered_fixture(border_of, nb);
        let n = g.n_levels();
        let m = block_spd(n, &blocks, seed as u64 + 7);
        let gy: Vec<f64> = (0..n).map(|l| (l as f64 * 0.7).sin() + 0.3).collect();
        let step = bordered_step(&g, &m, &gy, &blocks).expect("PD blocks");
        let dense = -g
            .theta_matrix(&m)
            .cholesky()
            .unwrap()
            .solve(&DVector::from_vec(g.theta_gradient(&gy)));
        let diag = g.theta_diagonal(&m);
        for a in 0..g.coords.len() {
            assert!((diag[a] - g.theta_matrix(&m)[(a, a)]).abs() < 1e-12);
            assert!(
                (step[a] - dense[a]).abs() < 1e-10 * (1.0 + dense[a].abs()),
                "case {seed}, θ {a}: {} vs {}",
                step[a],
                dense[a]
            );
        }
        // A level with no curvature of its own: the bordered form declines.
        let mut flat = m.clone();
        for l in 0..n {
            flat[(blocks[0][0], l)] = 0.0;
            flat[(l, blocks[0][0])] = 0.0;
        }
        assert!(bordered_step(&g, &flat, &gy, &blocks).is_none());
    }
}

/// The covariance step's `D` and `B`, differenced over each subject's own levels and
/// mapped to θ, equal the dense θ-space FD of the population objective (every EBE
/// re-solved cold) at the solved point, under the global contrast. Dies on a wrong sign
/// or a dropped term in either mapping (`TᵀDT`, `BT`).
#[test]
fn level_blocks_over_levels_are_the_theta_hessian() {
    let p = global_prep(2);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let hold = hold_template(&layout, &p.init_params);
    let PackedStart {
        packed: mut x,
        fixed,
        ..
    } = pack_with_bounds(&hold);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let s = c.solve(&unpack_params(&x, &hold), &warm).unwrap();
    for &k in &layout.coords {
        x[k] = packed_theta(&hold, k, s.params.theta[k]);
    }
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let (d, b) = c.level_blocks(&hold, &x, &free, &s).unwrap();
    assert_eq!(d.len(), 1);
    let n = layout.coords.len();
    let beta: Vec<f64> = layout.coords.iter().map(|&k| s.params.theta[k]).collect();
    let at = |ds: &[(usize, f64)], dl: &[(usize, f64)]| {
        let mut y = x.clone();
        for &(a, v) in ds {
            y[free[a]] += v;
        }
        let mut bb = beta.clone();
        for &(j, v) in dl {
            bb[j] += v;
        }
        total_ofv(&c, &unpack_params(&y, &hold), &bb)
    };
    let hl: Vec<f64> = s.se_scale.iter().map(|v| 0.1 * v).collect();
    let mut worst_d = 0.0_f64;
    let scale_d = d[0].amax();
    for a in 0..n {
        for cc in 0..=a {
            let (h, g) = (hl[a], hl[cc]);
            let fd = if a == cc {
                (at(&[], &[(a, h)]) - 2.0 * at(&[], &[]) + at(&[], &[(a, -h)])) / (h * h)
            } else {
                (at(&[], &[(a, h), (cc, g)])
                    - at(&[], &[(a, h), (cc, -g)])
                    - at(&[], &[(a, -h), (cc, g)])
                    + at(&[], &[(a, -h), (cc, -g)]))
                    / (4.0 * h * g)
            };
            assert!(fd.is_finite() && d[0][(a, cc)].is_finite());
            worst_d = worst_d.max((fd - d[0][(a, cc)]).abs() / scale_d);
        }
    }
    let mut worst_b = 0.0_f64;
    let scale_b = b.amax();
    let mut worst_at = (0, 0, 0.0, 0.0);
    for sidx in 0..free.len() {
        // The structural step production takes, so the two sides share their truncation.
        let floor = fd_step_floor(&hold, &c.model.default_params, free[sidx]);
        let hs = fd_step(o.fd_hessian_step, x[free[sidx]], floor);
        for a in 0..n {
            let h = hl[a];
            let fd = (at(&[(sidx, hs)], &[(a, h)])
                - at(&[(sidx, hs)], &[(a, -h)])
                - at(&[(sidx, -hs)], &[(a, h)])
                + at(&[(sidx, -hs)], &[(a, -h)]))
                / (4.0 * hs * h);
            assert!(fd.is_finite() && b[(sidx, a)].is_finite());
            let gap = (fd - b[(sidx, a)]).abs() / scale_b;
            if gap > worst_b {
                worst_at = (sidx, a, fd, b[(sidx, a)]);
            }
            worst_b = worst_b.max(gap);
        }
    }
    eprintln!(
        "#2135 level blocks: D {worst_d:.3e}, B {worst_b:.3e} (relative to max; worst B at \
         {worst_at:?}, max {scale_b:.3e})"
    );
    assert!(worst_d < LEVEL_BLOCKS_BOUND, "D {worst_d}");
    assert!(worst_b < LEVEL_BLOCKS_BOUND, "B {worst_b}");
}

/// Worst realised gap relative to the largest entry: `D` 3.6e-7, `B` 3.4e-6 (macOS arm64),
/// both FD of one objective at shared steps; ~8× headroom over `B`. Dropping the border's
/// term from the `B` mapping moves it to 4.1e-1 (measured).
const LEVEL_BLOCKS_BOUND: f64 = 3e-5;
